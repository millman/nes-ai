from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import _group_count, default


class Residual(nn.Module):
    """Wrap a module with a residual connection, optionally learning the skip."""

    def __init__(
        self,
        fn: nn.Module,
        *,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        learned_skip = (
            in_channels is not None
            and out_channels is not None
            and in_channels != out_channels
        )
        self.fn = fn
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if learned_skip
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self.skip(x) + self.fn(x, **kwargs)


class ResNetBlock2d(nn.Module):
    """Pre-activation residual block with SiLU activations and GroupNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        *,
        groups: int = 32,
    ) -> None:
        super().__init__()
        out_channels = default(out_channels, in_channels)
        self.norm1 = nn.GroupNorm(_group_count(in_channels, groups), in_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels, groups), out_channels)
        self.act2 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Multi-head self-attention operating over spatial tokens."""

    def __init__(
        self,
        channels: int,
        *,
        heads: int = 4,
        dim_head: int = 64,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__()
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        self.heads = heads
        inner = heads * dim_head
        self.norm = nn.GroupNorm(1, channels)
        self.to_qkv = nn.Conv2d(channels, inner * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(inner, channels, kernel_size=1)
        self.scale = dim_head ** -0.5
        self.max_tokens = max_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_norm = self.norm(x)
        tokens = h * w
        if tokens > self.max_tokens:
            pool_factor = int(math.ceil(math.sqrt(tokens / self.max_tokens)))
            pooled = F.avg_pool2d(
                x_norm, kernel_size=pool_factor, stride=pool_factor, ceil_mode=True
            )
            target_size = pooled.shape[-2:]
        else:
            pooled = x_norm
            target_size = (h, w)

        qkv = self.to_qkv(pooled)
        q, k, v = qkv.chunk(3, dim=1)

        def reshape_heads(t: torch.Tensor) -> torch.Tensor:
            bsz, ch, height, width = t.shape
            return t.view(bsz, self.heads, ch // self.heads, height * width)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)
        attn = torch.einsum("bhdi,bhdj->bhij", q * self.scale, k)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhij,bhdj->bhdi", attn, v)
        out = out.contiguous().view(b, -1, target_size[0] * target_size[1]).view(
            b, -1, target_size[0], target_size[1]
        )
        out = self.proj(out)
        if target_size != (h, w):
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out + x


class ModernResNetAttnBlock(nn.Module):
    """Residual block that combines modern conv block with spatial attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        *,
        heads: int = 4,
        dim_head: int = 64,
        groups: int = 32,
    ) -> None:
        super().__init__()
        out_channels = default(out_channels, in_channels)
        self.residual_block = ResNetBlock2d(
            in_channels, out_channels, groups=groups
        )
        self.attn = Residual(
            SelfAttention2d(out_channels, heads=heads, dim_head=dim_head),
            in_channels=out_channels,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual_block(x)
        x = self.attn(x)
        return x


class ModernResNetAttnDown(nn.Module):
    """Downsampling block built around ModernResNetAttnBlock."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        heads: int = 4,
        dim_head: int = 64,
        groups: int = 32,
    ) -> None:
        super().__init__()
        self.block = ModernResNetAttnBlock(
            in_channels,
            out_channels,
            heads=heads,
            dim_head=dim_head,
            groups=groups,
        )
        self.down = nn.Sequential(
            nn.GroupNorm(_group_count(out_channels, groups), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return self.down(x)


class ModernResNetAttnUp(nn.Module):
    """Upsampling block for the modern attention autoencoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        heads: int = 4,
        dim_head: int = 64,
        groups: int = 32,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ModernResNetAttnBlock(
            out_channels,
            out_channels,
            heads=heads,
            dim_head=dim_head,
            groups=groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.block(x)


class ModernResNetAttnAutoencoder(nn.Module):
    """Autoencoder with ResNet-style residual blocks and attention."""

    def __init__(self, base_channels: int = 64, latent_channels: int = 256) -> None:
        super().__init__()
        if base_channels <= 0 or latent_channels <= 0:
            raise ValueError("base_channels and latent_channels must be positive.")
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(inplace=True),
        )
        self.down1 = ModernResNetAttnDown(base_channels, base_channels * 2)
        self.down2 = ModernResNetAttnDown(base_channels * 2, base_channels * 4)
        self.down3 = ModernResNetAttnDown(base_channels * 4, latent_channels)
        self.mid = ModernResNetAttnBlock(latent_channels)
        self.up1 = ModernResNetAttnUp(latent_channels, base_channels * 4)
        self.up2 = ModernResNetAttnUp(base_channels * 4, base_channels * 2)
        self.up3 = ModernResNetAttnUp(base_channels * 2, base_channels)
        self.head = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels // 2, 3, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_hw = x.shape[-2:]
        h = self.stem(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.mid(h)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        out = self.head(h)
        if out.shape[-2:] != target_hw:
            out = F.interpolate(out, size=target_hw, mode="bilinear", align_corners=False)
        return out


__all__ = ["ModernResNetAttnAutoencoder"]
