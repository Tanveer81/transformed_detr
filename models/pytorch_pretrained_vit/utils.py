"""utils.py - Helper functions
"""

import numpy as np
import torch
from torch.utils import model_zoo

from .configs import PRETRAINED_MODELS


def load_pretrained_weights(
    model, 
    model_name=None, 
    weights_path=None, 
    load_first_conv=True, 
    load_fc=True, 
    load_repr_layer=False,
    resize_positional_embedding=False,
    verbose=True,
    strict=False,
    old_img = None,
    new_img = None,
    deit=False
):
    """Loads pretrained weights from weights path or download using url.
    Args:
        model (Module): Full model (a nn.Module)
        model_name (str): Model name (e.g. B_16)
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_first_conv (bool): Whether to load patch embedding.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
        resize_positional_embedding=False,
        verbose (bool): Whether to print on completion
    """
    assert bool(model_name) ^ bool(weights_path), 'Expected exactly one of model_name or weights_path'
    
    # Load or download weights
    if weights_path is None:
        url = PRETRAINED_MODELS[model_name]['url']
        if url:
            state_dict = model_zoo.load_url(url)
        else:
            raise ValueError(f'Pretrained model for {model_name} has not yet been released')
    else:
        print("Loading weightfrom:",weights_path)
        state_dict = torch.load(weights_path)

    # Modifications to load partial state dict
    expected_missing_keys = []
    if not load_first_conv and 'patch_embedding.weight' in state_dict:
        expected_missing_keys += ['patch_embedding.weight', 'patch_embedding.bias']
    if not load_fc and 'fc.weight' in state_dict:
        expected_missing_keys += ['fc.weight', 'fc.bias']
    if not load_repr_layer and 'pre_logits.weight' in state_dict:
        expected_missing_keys += ['pre_logits.weight', 'pre_logits.bias']
    for key in expected_missing_keys:
        state_dict.pop(key)

    # Change checkpoint dictionary
    if deit:
        old_img = (14,14)
        num_heads_model = int([n for (n, p) in model.transformer.blocks.named_parameters()][-1].split('.')[0]) + 1 #todo RK, its not head, its number of layer
        num_heads_state_dict = int((len(state_dict['model']) - 8) / 12)
        if num_heads_model != num_heads_state_dict:
            raise ValueError(f'Pretrained model has different number of heads: {num_heads_state_dict} than defined models heads: {num_heads_model}')
        state_dict['class_token'] = state_dict['model'].pop('cls_token')
        state_dict['positional_embedding.pos_embedding'] = state_dict['model'].pop('pos_embed')
        state_dict['patch_embedding.weight'] = state_dict['model'].pop('patch_embed.proj.weight')
        state_dict['patch_embedding.bias'] = state_dict['model'].pop('patch_embed.proj.bias')
        state_dict['fc.weight'] = state_dict['model'].pop('head.weight')
        state_dict['fc.bias'] = state_dict['model'].pop('head.bias')
        state_dict['norm.weight'] = state_dict['model'].pop('norm.weight')
        state_dict['norm.bias'] = state_dict['model'].pop('norm.bias')

        for i in range(num_heads_model):
            qkv_w = state_dict['model'].pop(f'blocks.{i}.attn.qkv.weight').reshape(model.dim, 3, -1).permute(1, 0, 2)
            qkv_b = state_dict['model'].pop(f'blocks.{i}.attn.qkv.bias').reshape(model.dim, 3, -1).permute(1, 0, 2)
            state_dict[f'transformer.blocks.{i}.attn.proj_q.weight'] = qkv_w[0]
            state_dict[f'transformer.blocks.{i}.attn.proj_q.bias'] = qkv_b[0].squeeze()
            state_dict[f'transformer.blocks.{i}.attn.proj_k.weight'] = qkv_w[1]
            state_dict[f'transformer.blocks.{i}.attn.proj_k.bias'] = qkv_b[1].squeeze()
            state_dict[f'transformer.blocks.{i}.attn.proj_v.weight'] = qkv_w[2]
            state_dict[f'transformer.blocks.{i}.attn.proj_v.bias'] = qkv_b[2].squeeze()
            state_dict[f'transformer.blocks.{i}.pwff.fc1.weight'] = state_dict['model'].pop(f'blocks.{i}.mlp.fc1.weight')
            state_dict[f'transformer.blocks.{i}.pwff.fc1.bias'] = state_dict['model'].pop(f'blocks.{i}.mlp.fc1.bias')
            state_dict[f'transformer.blocks.{i}.pwff.fc2.weight'] = state_dict['model'].pop(f'blocks.{i}.mlp.fc2.weight')
            state_dict[f'transformer.blocks.{i}.pwff.fc2.bias'] = state_dict['model'].pop(f'blocks.{i}.mlp.fc2.bias')
            state_dict[f'transformer.blocks.{i}.proj.weight'] = state_dict['model'].pop(f'blocks.{i}.attn.proj.weight')
            state_dict[f'transformer.blocks.{i}.proj.bias'] = state_dict['model'].pop(f'blocks.{i}.attn.proj.bias')
            state_dict[f'transformer.blocks.{i}.norm1.weight'] = state_dict['model'].pop(f'blocks.{i}.norm1.weight')
            state_dict[f'transformer.blocks.{i}.norm1.bias'] = state_dict['model'].pop(f'blocks.{i}.norm1.bias')
            state_dict[f'transformer.blocks.{i}.norm2.weight'] = state_dict['model'].pop(f'blocks.{i}.norm2.weight')
            state_dict[f'transformer.blocks.{i}.norm2.bias'] = state_dict['model'].pop(f'blocks.{i}.norm2.bias')
    # Change size of positional embeddings
    if resize_positional_embedding: 
        posemb = state_dict['positional_embedding.pos_embedding']
        posemb_new = model.state_dict()['positional_embedding.pos_embedding']
        state_dict['positional_embedding.pos_embedding'] = \
            resize_positional_embedding_(posemb=posemb, posemb_new=posemb_new, 
                has_class_token=hasattr(model, 'class_token'),gs_old = old_img, gs_new = new_img)
        maybe_print('Resized positional embeddings from {} to {}'.format(
                    posemb.shape, posemb_new.shape), verbose)

    # Load state dict
    ret = model.load_state_dict(state_dict, strict=False)
    if strict:
        assert set(ret.missing_keys) == set(expected_missing_keys), \
            'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
        assert not ret.unexpected_keys, \
            'Missing keys when loading pretrained weights: {}'.format(ret.unexpected_keys)
        maybe_print('Loaded pretrained weights.', verbose)
    else:
        maybe_print('Missing keys when loading pretrained weights: {}'.format(ret.missing_keys), verbose)
        maybe_print('Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys), verbose)
        return ret


def maybe_print(s: str, flag: bool):
    if flag:
        print(s)


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


def resize_positional_embedding_(posemb, posemb_new, has_class_token=True, gs_old=[24,24], gs_new=[38,50]): #todo exp wd width and height mayb mispalced
    """Rescale the grid of position embeddings in a sensible manner"""
    from scipy.ndimage import zoom

    # Deal with class token
    ntok_new = posemb_new.shape[1]
    if has_class_token:  # this means classifier == 'token'
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

    # Get old and new grid sizes
    posemb_grid = posemb_grid.reshape(gs_old[0], gs_old[1], -1)

    #todo experiment with gs 0 and 1 wht is width and height
    if False: # Rescale grid wd zoom
        zoom_factor = (gs_new[0] / gs_old[0], gs_new[1] / gs_old[1], 1)
        posemb_grid = zoom(posemb_grid, zoom_factor, order=1)
        posemb_grid = posemb_grid.reshape(1, gs_new[0]*gs_new[1], -1)
        posemb_grid = torch.from_numpy(posemb_grid)
    else :
        posemb = torch.unsqueeze(posemb_grid.permute(2, 0, 1),dim =0)
        pos_emb = torch.nn.functional.interpolate(posemb, size=(gs_new[0], gs_new[1]), mode='bicubic', align_corners=False)
        posemb_grid = pos_emb.permute(0,2,3,1).flatten(1,2)

    # Deal with class token and return
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


