from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DownBlock, UpBlock


class TextureAwareAutoencoderUNet(nn.Module):
    """Higher-capacity autoencoder tuned for style and patch contrastive training."""

    def __init__(self, base_channels: int = 64, latent_channels: int = 192) -> None:
        super().__init__()
        if base_channels % 8 != 0:
            raise ValueError("base_channels must be divisible by 8 for GroupNorm stability.")
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(inplace=True),
        )
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 3)
        self.down3 = DownBlock(base_channels * 3, base_channels * 4)
        self.down4 = DownBlock(base_channels * 4, latent_channels)
        groups = 8 if latent_channels % 8 == 0 else 1
        self.bottleneck = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, latent_channels),
            nn.SiLU(inplace=True),
        )
        self.up1 = UpBlock(latent_channels, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 3)
        self.up3 = UpBlock(base_channels * 3, base_channels * 2)
        self.up4 = UpBlock(base_channels * 2, base_channels)
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels // 2, 3, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.stem(x)
        h1 = self.down1(h0)
        h2 = self.down2(h1)
        h3 = self.down3(h2)
        h4 = self.down4(h3)
        b = self.bottleneck(h4)
        u1 = self.up1(b) + h3
        u2 = self.up2(u1) + h2
        u3 = self.up3(u2) + h1
        u4 = self.up4(u3) + h0
        out = self.head(u4)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


__all__ = ["TextureAwareAutoencoderUNet"]
