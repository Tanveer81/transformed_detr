# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List
import os
import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .pytorch_pretrained_vit.utils import non_strict_load_state_dict
from timm.models.layers import DropPath

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, return_intermediate_dec=False,
                  backbone_name = 'resnet', cross_first=False, drop_path=0, use_proj_in_dec=False,
                 bkbone_dim=768, hierarchical_pool=False, pool_size=[None, None, 24,24,14,14]):
        super().__init__()

        # In case of ViT backbone, self.backbone changes to "ViT" from detr
        self.backbone = backbone_name

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(num_decoder_layers, decoder_norm, return_intermediate_dec,
                                          drop_path, d_model,nhead, dim_feedforward,
                                          dropout, activation, normalize_before, cross_first,
                                          use_proj_in_dec, bkbone_dim, hierarchical_pool,
                                          pool_size=pool_size)

        self._reset_parameters()
        self.dim_feedforward = dim_feedforward
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # If ViT is used then src shape is (N,C,HW)
        bs, hw, c = src.shape[1:] if len(src.shape) > 3 else src.shape
        #h = w = int(math.sqrt(hw))
        src = src.permute(0,2, 1, 3) if len(src.shape) > 3 else src.permute(1, 0, 2)
        pos_embed = pos_embed.permute(1, 0, 2)

        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)

        hs = self.decoder(tgt, src, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)

        return hs.transpose(1, 2)

class TransformerDecoder(nn.Module):

    def __init__(self, num_layers, norm=None, return_intermediate=False, drop_path=0, d_model=512,
                 nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False, cross_first=False, use_proj_in_dec=False,
                 bkbone_dim=768, hierarchical_pool=None, reduce_backbone=None, pool_size=[None, None, 24,24,14,14]):
        super().__init__()
        if d_model!=bkbone_dim: # reduce the prjection decoder layer wise
            reduce_backbone = nn.Linear(bkbone_dim, d_model)
            torch.nn.init.xavier_uniform_(reduce_backbone.weight)
        #use multiple pool for multiscale feature map
        assert len(pool_size)==num_layers
        # drop path rate
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)]  # stochastic depth decay rule
        # create decoder layer woth drop path
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before,
                                                             cross_first, use_proj_in_dec, reduce_backbone, i, hierarchical_pool,
                                                             pool_size, drop_path=dpr[i])
                                    for i in range(num_layers)])

        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate


    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []
        for i, layer in enumerate(self.layers):
            if len(memory.shape)>3: #for features coming from lower label
                layer_wise_enc = memory[i if i <memory.shape[0] -1 else memory.shape[0]-1,:,:,:]
            else:
                layer_wise_enc = memory
            output = layer(output, layer_wise_enc, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, cross_first=False, use_proj_in_dec=False, reduce_backbone=None,
                 layer_num=None, hierarchical_pool=None, pool_size=None, drop_path=0.):
        super().__init__()
        #assert not (dropout>0. and drop_path>0.), 'dropout and drop_path cannot both be greater than 0.'
        self.cross_first=cross_first
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.d_model = d_model
        self.use_proj_in_dec = use_proj_in_dec
        self.reduce_backbone = reduce_backbone
        if self.use_proj_in_dec:
            torch.Assert(self.d_model==256,"Reduce decoder dim to 256 ")
            self.input_proj = nn.Linear(768, self.d_model)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_number = layer_num

        if hierarchical_pool !='None' and pool_size[layer_num]!="_":
            self.pool =  eval("nn.{}(({},{}))".format(hierarchical_pool,int(pool_size[layer_num]),int(pool_size[layer_num])))

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        if self.use_proj_in_dec and not self.d_model==tensor.shape[-1]:
            return self.input_proj(tensor if pos is None else tensor + pos)
        elif self.reduce_backbone != None and not self.d_model==tensor.shape[-1]:
            return self.reduce_backbone(tensor if pos is None else tensor + pos)
        else:
            return tensor if pos is None else tensor + pos #todo with no positional embedding for testing

    def wd_pool(self, tensor):
        if hasattr(self, "pool"):
            token_len, bs, dim = tensor.shape
            im_size = int(np.sqrt(token_len))
            tensor = self.pool(tensor.view(im_size, im_size, bs, dim).permute(2, 3, 0, 1))
            return tensor.view(bs, dim, -1).permute(2,0,1)
        else:
            return  tensor

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        # Perform cross attention between encoder values and decoder queries
        # Then perform self attention between decoder object queries
        if self.cross_first: # 1st aplly decoder to all image attn
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=self.with_pos_embed(memory, None), attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

            q = k = self.with_pos_embed(tgt, query_pos) #tgt if self.cross_first else  todo experiment wdout pos emebeding as for cross attn its already done earlier
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.drop_path(self.dropout1(tgt2))
            tgt = self.norm1(tgt)

        # pPerform self attention between decoder object queries first
        else:
            q = k = self.with_pos_embed(tgt, query_pos)  # tgt if self.cross_first else  todo experiment wdout pos emebeding as for cross attn its already done earlier
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.wd_pool(self.with_pos_embed(memory, pos)),
                                       value=self.wd_pool(self.with_pos_embed(memory, None)), attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
            tgt = tgt + self.drop_path(self.dropout2(tgt2))
            tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.drop_path(self.dropout3(tgt2))
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=self.with_pos_embed(memory, None), attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args, bkbone_dim=768):
    transformer = Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.detr_nheads,
        dim_feedforward=args.dim_feedforward,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        backbone_name=args.pretrained_model,
        activation="gelu",
        cross_first = args.cross_first,
        drop_path = args.drop_path,
        use_proj_in_dec=args.use_proj_in_dec,
        bkbone_dim=bkbone_dim,
        hierarchical_pool = args.use_ms_dec,
        pool_size=args.pool_size
    )

    return transformer


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
