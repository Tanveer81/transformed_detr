"""
Adapted from https://github.com/lukemelas/simple-bert
"""

import numpy as np
import torch
from torch import nn
from torch import Tensor 
from torch.nn import functional as F


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, num_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.drop(self.proj(self.attn(self.norm1(x), mask)))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, imsize, skip_connection=False, include_class_token=True, hierarchy=False, pool=None, fh=None, fw=None, gh=None, gw=None):
        super().__init__()
        self.gh = gh
        self.gw = gw
        self.fh = fh
        self.fw = fw
        self.pool = pool
        self.hierarchy = hierarchy
        self.include_class_token = include_class_token
        self.skip_connection = skip_connection
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.norm_layer = []
        for i in range(4):
            self.norm_layer.append(nn.LayerNorm(dim, eps=1e-6))
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        if self.pool == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=(self.fh, self.fw), stride=(self.fh, self.fw))
        elif self.pool == 'max':
            self.pool = nn.MaxPool2d(kernel_size=(self.fh, self.fw), stride=(self.fh, self.fw))
        self.imsize = imsize

    def hour_glass(self, x):
        # x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[2], self.gh, self.gw)
        # x = self.pool(x)
        # x = x.flatten(2).transpose(1, 2)
        x = torch.reshape(x.transpose(1, 2), (x.shape[0], x.shape[2], self.gh, self.gw)).contiguous()
        x = torch.reshape(self.pool(x), (x.shape[0], x.shape[2], -1)).transpose(1, 2)
        return x

    def forward(self, x, mask=None):
        # Use only 6 layers for hierarchical structure
        if self.hierarchy:
            residual_connections = []
            for i, block in zip(range(len(self.blocks)), self.blocks):
                x = block(x, mask)
#                 print('Block',i+1,'Min:', x.data.min().cpu().numpy(), 'Max', x.data.max().cpu().numpy())

                # skip connections
                if i in [2, 5, 8]:
                    if i == 2 or 5:
                        if self.include_class_token:
                            token = x[:, 0:1, :]
                            y = self.hour_glass(x[:, 1:, :])
                            y = torch.cat([token, y], 1).contiguous()
                        else:
                            y = self.hour_glass(x)
                        residual_connections.append(y)
                    residual_connections.append(x)

                # downsample resolution -- here we may add another pos-embedding!
                if i == 5:
                    if self.include_class_token:
                        token = x[:, 0:1, :]
                        x = self.hour_glass(x[:, 1:, :])
                        x = torch.cat([token, x], 1).contiguous()
                    else:
                        x = self.hour_glass(x)

            # add skip connections with a pre-norm
            x = self.norm_layer[-1](residual)
            for i,residual in enumerate(residual_connections):
                x = x + self.norm_layer[-i-2](residual)


        # Added residual connection from layer 3, 6 and 9
        elif self.skip_connection:
            residual_connections = []
            for i, block in zip(range(len(self.blocks)), self.blocks):
                x = block(x, mask)
                if i in [2, 5, 8]:
                    residual_connections.append(x)

            for residual in residual_connections:
                x = x + residual
        # No skip connection
        else:
            for block in self.blocks:
#                 print('Min:', x.data.min().cpu().numpy(), 'Max', x.data.max().cpu().numpy())
                x = block(x, mask)

        return self.norm(x)