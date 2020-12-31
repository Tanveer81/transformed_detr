"""model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""

import math
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F

from .transformer import Transformer
from .utils import load_pretrained_weights, as_tuple
from .configs import PRETRAINED_MODELS
from positional_encodings import PositionalEncodingPermute2D


class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim, include_class_token=True):
        super().__init__()
        self.include_class_token = include_class_token
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""

        if self.include_class_token:
            return x + self.pos_embedding

        # else removed class token
        return x + self.pos_embedding[:, 1:, :]


class ViT(nn.Module):
    """
    Args:
        name (str): Model name, e.g. 'B_16'
        pretrained (bool): Load pretrained weights
        in_channels (int): Number of channels in input data
        num_classes (int): Number of classes, default 1000

        # Control Parameters for integrating DETR
        weight_path (str): Path of weights downloaded previously
        detr_compatibility (bool): Is the model compatible with DETR?


    References:
        [1] https://openreview.net/forum?id=YicbFdNTTy
    """

    def __init__(
            self,
            name: Optional[str] = None,
            pretrained: bool = False,
            position_embedding: str = "sine",  # ('sine', 'learned')
            weight_path: Optional[str] = None,
            detr_compatibility: bool = False,
            patches: int = 16,
            dim: int = 768,
            ff_dim: int = 3072,
            num_heads: int = 12,
            num_layers: int = 12,
            attention_dropout_rate: float = 0.0,
            dropout_rate: float = 0.1,
            representation_size: Optional[int] = None,
            load_repr_layer: bool = False,
            classifier: str = 'token',
            positional_embedding: str = '1d',
            in_channels: int = 3,
            image_size: Optional[tuple] = (608, 800),
            num_classes: Optional[int] = None,
            include_class_token: bool = True,
            skip_connection: bool = False
    ):

        super().__init__()

        # Configuration
        self.include_class_token = include_class_token
        self.skip_connection = skip_connection
        self.weight_path = weight_path
        self.detr_compatibility = detr_compatibility
        self.position_embedding = position_embedding
        if name is None:
            check_msg = 'must specify name of pretrained model'
            assert not pretrained, check_msg
            assert not resize_positional_embedding, check_msg
            if num_classes is None:
                num_classes = 1000
            if image_size is None:
                image_size = 384
        else:  # load pretrained model
            assert name in PRETRAINED_MODELS.keys(), \
                'name should be in: ' + ', '.join(PRETRAINED_MODELS.keys())
            config = PRETRAINED_MODELS[name]['config']
            patches = config['patches']
            dim = config['dim']
            ff_dim = config['ff_dim']
            num_heads = config['num_heads']
            num_layers = config['num_layers']
            attention_dropout_rate = config['attention_dropout_rate']
            dropout_rate = config['dropout_rate']
            representation_size = config['representation_size']
            classifier = config['classifier']
            if image_size is None:
                image_size = as_tuple(PRETRAINED_MODELS[name]['image_size'],
                                      PRETRAINED_MODELS[name]['image_size'])
            if num_classes is None:
                num_classes = PRETRAINED_MODELS[name]['num_classes']
        self.image_size = image_size
        self.dim = dim
        # Image and patch sizes
        h, w = image_size  # image sizes
        fh, fw = as_tuple(patches)  # patch sizes
        gh, gw = h // fh, w // fw  # number of patches
        seq_len = gh * gw

        # Patch embedding
        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))
        # self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((gh, gw))

        # Class token
        if classifier == 'token' and self.include_class_token:
            self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
            seq_len += 1

        # Positional embedding
        self.positional_embedding_type = positional_embedding
        if positional_embedding.lower() == '1d':
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        else:
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
            self.positional_embedding_2d = PositionalEncodingPermute2D(dim)

        # Transformer
        self.transformer = Transformer(num_layers=num_layers, dim=dim, num_heads=num_heads,
                                       ff_dim=ff_dim, dropout=dropout_rate, skip_connection=self.skip_connection)

        # todo not needed! Representation layer
        if representation_size and load_repr_layer:
            self.pre_logits = nn.Linear(dim, representation_size)
            pre_logits_size = representation_size
        else:
            pre_logits_size = dim

            # Classifier head
            self.norm = nn.LayerNorm(pre_logits_size, eps=1e-6)
            self.fc = nn.Linear(pre_logits_size, num_classes)

        # Initialize weights
        self.init_weights()

        # Load pretrained model
        if pretrained:
            pretrained_num_channels = 3
            pretrained_num_classes = PRETRAINED_MODELS[name]['num_classes']
            pretrained_image_size = PRETRAINED_MODELS[name]['image_size']
            load_pretrained_weights(
                self,
                # If weights_path given, no need to give model name
                name if self.weight_path is None else None,
                # If the weights are already downloaded previously, provide the path
                weights_path=self.weight_path,
                load_first_conv=(in_channels == pretrained_num_channels),
                load_fc=(num_classes == pretrained_num_classes),
                load_repr_layer=load_repr_layer,
                resize_positional_embedding=(image_size != pretrained_image_size),
                old_img=(pretrained_image_size[0] // fh, pretrained_image_size[1] // fw),
                # original vit 384x384
                new_img=(gh, gw),
            )

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)

        self.apply(_init)
        if not self.detr_compatibility:
            nn.init.constant_(self.fc.weight, 0)
            nn.init.constant_(self.fc.bias, 0)
        nn.init.normal_(self.positional_embedding.pos_embedding,
                        std=0.02)  # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)
        nn.init.constant_(self.class_token, 0)

    def forward(self, x):
        """Breaks image into patches, applies transformer, applies MLP head.

        Args:
            x (tensor): `b,c,fh,fw`
        """
        if not isinstance(x, torch.Tensor):
            x = x.tensors
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  # b,d,gh,gw
        # x = self.AdaptiveAvgPool2d(x)
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d

        if hasattr(self, 'class_token') and self.include_class_token:
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        if hasattr(self, 'positional_embedding'):
            x = self.positional_embedding(x)  # b,gh*gw+1,d

        x = self.transformer(x)  # b,gh*gw+1,d
        if self.positional_embedding_type.lower() == '2d':
            pos_embed_2d = self.positional_embedding_2d(x.permute(0, 2, 1)[:, :, :x.shape[1]].reshape(x.shape[0], x.shape[2], int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1]))))
            pos_embed_2d = pos_embed_2d.reshape([pos_embed_2d.shape[0], pos_embed_2d.shape[1], -1]).permute(0, 2, 1)
            x = x + pos_embed_2d

        if self.detr_compatibility:
            # if self.position_embedding == "sine":
            #     return x, PositionEmbeddingSine(self.dim/2, normalize=True)
            # else:
            if self.positional_embedding_type.lower() == '1d':
                if self.include_class_token:
                    return x, self.positional_embedding.pos_embedding
                return x, self.positional_embedding.pos_embedding[:, 1:, :]

            return x, pos_embed_2d

        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)
        if hasattr(self, 'fc'):
            x = self.norm(x)[:, 0]  # b,d
            x = self.fc(x)  # b,num_classes

        return x


class hierarchicalViT(nn.Module):
    def __init__(self, args):
        self.backbone1 = ViT(args.backbone,
                       pretrained=args.pretrained_vit,
                       weight_path=f"{args.pretrain_dir}/{args.backbone}.pth",
                       detr_compatibility=True,
                       position_embedding=args.position_embedding,
                       image_size=args.img_size,
                       num_heads=args.vit_heads,
                       num_layers=args.vit_layer,
                       include_class_token=args.include_class_token,
                       skip_connection=args.skip_connection
                       )

        self.backbone2 = ViT(args.backbone,
                       pretrained=args.pretrained_vit,
                       weight_path=f"{args.pretrain_dir}/{args.backbone}.pth",
                       detr_compatibility=True,
                       position_embedding=args.position_embedding,
                       image_size=args.img_size,
                       num_heads=args.vit_heads,
                       num_layers=args.vit_layer,
                       include_class_token=args.include_class_token,
                       skip_connection=args.skip_connection
                       )

        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((gh, gw))

        def forward(self, x):
            x = self.backbone1(x)
            x = make_2d(x)
            x = self.AdaptiveAvgPool2d(x)
            x = make_1d(x)
            x = self.backbone2(x)

            return x