# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------
from functools import partial
import timm.models.vision_transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.models.vision_transformer import Attention
import math
from einops import rearrange

class WindowAttention(nn.Module):
    """
    非重叠窗口注意力（non-overlapping window attention）
    - 参数结构与 timm 原生 Attention 完全一致 → 权重完美加载
    - CLS token 全局可见
    - 计算量真正降低
    - 支持任意 window_size（需能整除 H/W，或用方案1改 win_size）
    """

    def __init__(self, dim, num_heads=8, window_size=7, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.window_size = window_size
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        num_patches = N - 1
        H = W = int(math.sqrt(num_patches))
        assert H * W == num_patches
        assert H % self.window_size == 0 and W % self.window_size == 0, f'H={H}, W={W} must be divisible by window_size={self.window_size}. Use win_size=8 or 16 for 1024 input.'
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q_cls = q[:, :, 0:1, :]
        k_cls = k[:, :, 0:1, :]
        v_cls = v[:, :, 0:1, :]
        q_patch = q[:, :, 1:, :]
        k_patch = k[:, :, 1:, :]
        v_patch = v[:, :, 1:, :]
        q_patch = rearrange(q_patch, 'b h (gh gw) d -> b h gh gw d', gh=H, gw=W)
        k_patch = rearrange(k_patch, 'b h (gh gw) d -> b h gh gw d', gh=H, gw=W)
        v_patch = rearrange(v_patch, 'b h (gh gw) d -> b h gh gw d', gh=H, gw=W)
        q_win = rearrange(q_patch, 'b h (nh wh) (nw ww) d -> b h nh nw (wh ww) d', wh=self.window_size, ww=self.window_size)
        k_win = rearrange(k_patch, 'b h (nh wh) (nw ww) d -> b h nh nw (wh ww) d', wh=self.window_size, ww=self.window_size)
        v_win = rearrange(v_patch, 'b h (nh wh) (nw ww) d -> b h nh nw (wh ww) d', wh=self.window_size, ww=self.window_size)
        attn_win = q_win @ k_win.transpose(-2, -1) * self.scale
        attn_win = attn_win.softmax(dim=-1)
        attn_win = self.attn_drop(attn_win)
        x_win = attn_win @ v_win
        x_patch = rearrange(x_win, 'b h nh nw (wh ww) d -> b h (nh wh) (nw ww) d', wh=self.window_size, ww=self.window_size)
        x_patch = rearrange(x_patch, 'b h gh gw d -> b h (gh gw) d')
        attn_cls = q_cls @ k.transpose(-2, -1) * self.scale
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)
        x_cls = attn_cls @ v
        x = torch.cat([x_cls, x_patch], dim=2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def replace_attn_with_win_attn(vit: nn.Module, win_size: int=7, cls_global: bool=True):
    """
    替换为真正的 WindowAttention（不使用 mask）
    cls_global 参数暂时忽略，因为我们固定让 CLS 全局可见
    """
    for blk in vit.blocks:
        old_attn = blk.attn
        new_attn = WindowAttention(dim=old_attn.qkv.in_features, num_heads=old_attn.num_heads, window_size=win_size, qkv_bias=old_attn.qkv.bias is not None, attn_drop=old_attn.attn_drop.p if hasattr(old_attn.attn_drop, 'p') else 0.0, proj_drop=old_attn.proj_drop.p if hasattr(old_attn.proj_drop, 'p') else 0.0)
        new_attn.qkv.load_state_dict(old_attn.qkv.state_dict())
        new_attn.proj.load_state_dict(old_attn.proj.state_dict())
        blk.attn = new_attn.to(next(vit.parameters()).device)
    return vit

def RETFound_mae(**kwargs):
    if 'global_pool' in kwargs:
        if kwargs['global_pool'] is True:
            kwargs['global_pool'] = 'avg'
        elif kwargs['global_pool'] is False:
            kwargs['global_pool'] = ''
        elif kwargs['global_pool'] not in ('', 'avg', 'token', 'map'):
            raise ValueError(f"Invalid global_pool: {kwargs['global_pool']}")
    model = VisionTransformer(patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-06), **kwargs)
    return model
import math
import torch
import torch.nn as nn
from functools import partial
import timm
from timm.models.vision_transformer import Attention, VisionTransformer

def _build_local_mask(n_tokens: int, win_size: int, cls_global: bool=True, device=None, dtype=None):
    """
    构造 additive mask, 形状 (1, 1, N, N), 允许的位置为 0, 屏蔽的位置为 -inf
    完全向量化实现，无 Python 循环，显著提升速度（尤其在大分辨率如 64x64 或更高时）
    """
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    assert n_tokens >= 2, '序列长度必须包含 [CLS] + 至少一个 patch'
    P = n_tokens - 1
    H = W = int(math.sqrt(P))
    assert H * W == P and H > 0, f'Patch tokens ({P}) 不是正方形网格'
    radius = (win_size - 1) // 2
    rows = torch.arange(H, device=device)
    cols = torch.arange(W, device=device)
    row_grid, col_grid = torch.meshgrid(rows, cols, indexing='ij')
    patch_pos = torch.stack((row_grid.flatten(), col_grid.flatten()), dim=1)
    diff = patch_pos.unsqueeze(1) - patch_pos.unsqueeze(0)
    cheb_dist = diff.abs().max(dim=2).values
    patch_local = cheb_dist <= radius
    mask = torch.full((n_tokens, n_tokens), float('-inf'), device=device, dtype=dtype)
    mask[1:, 1:][patch_local] = 0.0
    mask.diagonal(dim1=0, dim2=1).fill_(0.0)
    if cls_global:
        mask[0, :] = 0.0
        mask[:, 0] = 0.0
    return mask.unsqueeze(0).unsqueeze(0)

class LocalMaskAttention(Attention):
    """
    继承自 timm 的 Attention,不改变参数结构,只在 forward 时加上局部掩码。
    这样能保证和原始 ViT 权重完全兼容。
    """

    def __init__(self, *args, win_size: int=7, cls_global: bool=True, **kwargs):
        super().__init__(*args, **kwargs)
        assert win_size % 2 == 1 and win_size >= 1, 'win_size 应为奇数且 >= 1'
        self.win_size = win_size
        self.cls_global = cls_global
        self.register_buffer('_local_mask', None, persistent=False)
        self._cached_n = None

    @torch.no_grad()
    def _maybe_build_mask(self, N: int, device, dtype):
        if self._local_mask is None or self._cached_n != N or self._local_mask.device != device:
            self._local_mask = _build_local_mask(N, self.win_size, self.cls_global, device=device, dtype=dtype)
            self._cached_n = N

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = q * self.scale @ k.transpose(-2, -1)
        self._maybe_build_mask(N, attn.device, attn.dtype)
        attn = attn + self._local_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def replace_attn_with_local_mask(vit: nn.Module, win_size: int=7, cls_global: bool=True):
    """
    就地替换 ViT 中每个 Block 的 Attention 为 LocalMaskAttention。
    - 保留 qkv/proj 权重,保证能直接加载预训练权重
    - 替换后,旧权重已被拷贝到新 Attention 中
    """
    for blk in vit.blocks:
        old_attn = blk.attn
        new_attn = LocalMaskAttention(dim=old_attn.qkv.in_features, num_heads=old_attn.num_heads, qkv_bias=True if old_attn.qkv.bias is not None else False, attn_drop=old_attn.attn_drop.p if hasattr(old_attn.attn_drop, 'p') else 0.0, proj_drop=old_attn.proj_drop.p if hasattr(old_attn.proj_drop, 'p') else 0.0, win_size=win_size, cls_global=cls_global)
        new_attn.qkv.load_state_dict(old_attn.qkv.state_dict())
        new_attn.proj.load_state_dict(old_attn.proj.state_dict())
        device = next(vit.parameters()).device
        new_attn = new_attn.to(device)
        blk.attn = new_attn
    return vit
