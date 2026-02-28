import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional, List


"""Swin Transformer with multi-scale gated fusion and CBAM.

A PyTorch implementation based on:
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030

This customized version introduces:
    1) multi-scale feature collection across stages,
    2) learnable gated weighting for feature fusion,
    3) CBAM-based attention refinement after concatenation.

Code/weights adapted from:
    https://github.com/microsoft/Swin-Transformer
"""


def drop_path_f(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample.

    This is applied to the main residual branch.
    """
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # support arbitrary tensor rank
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    """
    Partition a feature map into non-overlapping windows.

    Args:
        x: Tensor of shape (B, H, W, C)
        window_size: Window size

    Returns:
        windows: Tensor of shape (num_windows * B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Restore a feature map from windows.

    Args:
        windows: Tensor of shape (num_windows * B, window_size, window_size, C)
        window_size: Window size
        H: Output height
        W: Output width

    Returns:
        x: Tensor of shape (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class CBAMBlock(nn.Module):
    """
    CBAM: Convolutional Block Attention Module
    https://arxiv.org/abs/1807.06521

    Includes:
        1) Channel Attention
        2) Spatial Attention
    """
    def __init__(self, channels, reduction=16, spatial_kernel_size=7):
        super().__init__()

        hidden_channels = max(channels // reduction, 1)

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )

        # Spatial attention
        self.spatial_conv = nn.Conv2d(
            2, 1,
            kernel_size=spatial_kernel_size,
            stride=1,
            padding=spatial_kernel_size // 2
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # -------- Channel attention --------
        avg_feat = self.avg_pool(x)
        avg_out = self.channel_mlp(avg_feat)

        max_feat = self.max_pool(x)
        max_out = self.channel_mlp(max_feat)

        channel_attn = self.sigmoid(avg_out + max_out)
        x = x * channel_attn

        # -------- Spatial attention --------
        avg_feat = torch.mean(x, dim=1, keepdim=True)
        max_feat, _ = torch.max(x, dim=1, keepdim=True)
        spatial_pool = torch.cat([avg_feat, max_feat], dim=1)

        spatial_attn = self.sigmoid(self.spatial_conv(spatial_pool))
        x = x * spatial_attn

        return x


class PatchEmbed(nn.Module):
    """2D image to patch embedding."""
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # Pad input if H or W is not divisible by patch size
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            x = F.pad(
                x,
                (
                    0, self.patch_size[1] - W % self.patch_size[1],
                    0, self.patch_size[0] - H % self.patch_size[0],
                    0, 0
                )
            )

        x = self.proj(x)
        _, _, H, W = x.shape

        # [B, C, H, W] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    """Patch Merging Layer.

    Args:
        dim: Number of input channels.
        norm_layer: Normalization layer. Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        Args:
            x: Tensor of shape [B, H*W, C]
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # Pad if H or W is odd
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # Current layout is [B, H, W, C]
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)   # [B, H/2, W/2, 4C]
        x = x.view(B, -1, 4 * C)                  # [B, H/2*W/2, 4C]

        x = self.norm(x)
        x = self.reduction(x)                     # [B, H/2*W/2, 2C]
        return x


class Mlp(nn.Module):
    """MLP block used in Vision Transformer-like architectures."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias.

    Supports both shifted and non-shifted windows.

    Args:
        dim: Number of input channels.
        window_size: Height and width of the window.
        num_heads: Number of attention heads.
        qkv_bias: If True, add learnable bias to query, key, value.
        attn_drop: Dropout ratio for attention weights.
        proj_drop: Dropout ratio for output projection.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                num_heads
            )
        )

        # Pair-wise relative position index inside each window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: Tensor of shape (num_windows * B, Mh * Mw, C)
            mask: Tensor of shape (num_windows, Mh*Mw, Mh*Mw), or None
        """
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(
            B_, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift for SW-MSA
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2)
            )
        else:
            shifted_x = x
            attn_mask = None

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows back
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2)
            )
        else:
            x = shifted_x

        # Remove padded region
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # Residual + MLP
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer stage."""
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def create_mask(self, x, H, W):
        """Create attention mask for shifted window attention."""
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size

        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)

        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        attn_mask = self.create_mask(x, H, W)

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W


class SwinTransformer(nn.Module):
    """Swin Transformer with gated multi-scale fusion and CBAM."""
    def __init__(self,
                 in_chans=3,
                 patch_size=4,
                 window_size=7,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 num_classes=1000):
        super().__init__()

        self.patch_norm = patch_norm
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio

        # Patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_c=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Learnable gate parameters for multi-scale fusion
        self.gate_params = nn.Parameter(torch.ones(self.num_layers))

        # Build Swin stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()

        self.stage_out_dims: List[int] = []
        dim_current = embed_dim
        idx_drop = 0

        for i_layer in range(self.num_layers):
            use_downsample = i_layer < self.num_layers - 1

            layer = BasicLayer(
                dim=dim_current,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[idx_drop: idx_drop + depths[i_layer]],
                norm_layer=norm_layer,
                downsample=PatchMerging if use_downsample else None,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)

            dim_out = dim_current * 2 if use_downsample else dim_current
            self.stage_out_dims.append(dim_out)

            dim_current = dim_out
            idx_drop += depths[i_layer]

        # Final feature dimension
        self.num_features = self.stage_out_dims[-1]

        # Keep final norm for compatibility with pretrained Swin checkpoints
        # even though the customized fusion head does not explicitly use it.
        self.norm = norm_layer(self.num_features)

        # Fusion head
        fuse_in_channels = sum(self.stage_out_dims)
        self.fuse_cbam = CBAMBlock(fuse_in_channels)
        self.fuse_conv = nn.Conv2d(fuse_in_channels, self.num_features, kernel_size=1)
        self.fuse_bn = nn.BatchNorm2d(self.num_features)
        self.fuse_act = nn.ReLU(inplace=True)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, in_chans, H, W]

        Returns:
            logits: Tensor of shape [B, num_classes]
        """
        # ---- A) Patch embedding
        x, H, W = self.patch_embed(x)               # [B, H*W, C]
        x = self.pos_drop(x)
        x = x.view(-1, H, W, x.shape[-1])           # [B, H, W, C]

        # ---- B) Forward through all stages and collect multi-scale features
        skip_feats = []
        cur_H, cur_W = H, W

        for layer in self.layers:
            x_3d = x.view(x.shape[0], cur_H * cur_W, x.shape[-1])
            x_out, out_H, out_W = layer(x_3d, cur_H, cur_W)

            x_out_4d = x_out.view(x_out.shape[0], out_H, out_W, x_out.shape[-1])
            skip_feats.append(x_out_4d)

            x, cur_H, cur_W = x_out_4d, out_H, out_W

        # ---- C) Resize all earlier-stage features to the final-stage resolution
        # To preserve the original implementation logic, the fusion order is:
        # [Stage4, Stage1, Stage2, Stage3]
        x_s4 = skip_feats[-1]
        _, H4, W4, _ = x_s4.shape

        downsampled_feats = [x_s4]
        for feat_i in skip_feats[:-1]:
            feat_i_down = F.interpolate(
                feat_i.permute(0, 3, 1, 2),
                size=(H4, W4),
                mode="bilinear",
                align_corners=False
            ).permute(0, 2, 3, 1)
            downsampled_feats.append(feat_i_down)

        # ---- D) Learnable gated weighting for each scale
        weights = F.softmax(self.gate_params, dim=0)

        weighted_feats = []
        for i in range(self.num_layers):
            feat_i = downsampled_feats[i].permute(0, 3, 1, 2)  # [B, C_i, H4, W4]
            feat_i = weights[i] * feat_i
            weighted_feats.append(feat_i)

        fused_4d = torch.cat(weighted_feats, dim=1)            # [B, sumC, H4, W4]

        # ---- E) Attention refinement + channel projection
        fused_4d = self.fuse_cbam(fused_4d)
        fused_4d = self.fuse_conv(fused_4d)
        fused_4d = self.fuse_bn(fused_4d)
        fused_4d = self.fuse_act(fused_4d)

        # ---- F) Global pooling + classifier
        fused_4d = self.avgpool(fused_4d)
        fused_4d = fused_4d.flatten(1)
        logits = self.head(fused_4d)

        return logits


def swin_tiny_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # Trained on ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    model = SwinTransformer(
        in_chans=3,
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        num_classes=num_classes,
        **kwargs
    )
    return model


def swin_small_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # Trained on ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
    model = SwinTransformer(
        in_chans=3,
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        num_classes=num_classes,
        **kwargs
    )
    return model


def swin_base_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # Trained on ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
    model = SwinTransformer(
        in_chans=3,
        patch_size=4,
        window_size=7,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        num_classes=num_classes,
        **kwargs
    )
    return model


def swin_base_patch4_window12_384(num_classes: int = 1000, **kwargs):
    # Trained on ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth
    model = SwinTransformer(
        in_chans=3,
        patch_size=4,
        window_size=12,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        num_classes=num_classes,
        **kwargs
    )
    return model


def swin_base_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):
    # Trained on ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
    model = SwinTransformer(
        in_chans=3,
        patch_size=4,
        window_size=7,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        num_classes=num_classes,
        **kwargs
    )
    return model


def swin_base_patch4_window12_384_in22k(num_classes: int = 21841, **kwargs):
    # Trained on ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
    model = SwinTransformer(
        in_chans=3,
        patch_size=4,
        window_size=12,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        num_classes=num_classes,
        **kwargs
    )
    return model


def swin_large_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):
    # Trained on ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
    model = SwinTransformer(
        in_chans=3,
        patch_size=4,
        window_size=7,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        num_classes=num_classes,
        **kwargs
    )
    return model


def swin_large_patch4_window12_384_in22k(num_classes: int = 21841, **kwargs):
    # Trained on ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
    model = SwinTransformer(
        in_chans=3,
        patch_size=4,
        window_size=12,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        num_classes=num_classes,
        **kwargs
    )
    return model


if __name__ == "__main__":
    model = swin_tiny_patch4_window7_224(num_classes=5)
    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    print(out.shape)  # expected: [2, 5]