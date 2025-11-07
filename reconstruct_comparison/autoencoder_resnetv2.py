from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import PreActResidualBlock, PreActResidualUpBlock
from .model_utils import _norm_groups


class ResNetV2Autoencoder(nn.Module):
    """Pre-activation residual autoencoder inspired by ResNet v2."""

    def __init__(self, base_channels: int = 48, latent_channels: int = 128) -> None:
        super().__init__()
        if base_channels <= 0 or latent_channels <= 0:
            raise ValueError("base_channels and latent_channels must be positive.")
        self.stem = nn.Conv2d(3, base_channels, kernel_size=3, padding=1, bias=False)
        self.block0 = PreActResidualBlock(base_channels, base_channels)
        self.down1 = PreActResidualBlock(base_channels, base_channels * 2, stride=2)
        self.down2 = PreActResidualBlock(base_channels * 2, base_channels * 3, stride=2)
        self.down3 = PreActResidualBlock(base_channels * 3, latent_channels, stride=2)
        self.bottleneck = PreActResidualBlock(latent_channels, latent_channels)
        self.up1 = PreActResidualUpBlock(latent_channels, base_channels * 3)
        self.up2 = PreActResidualUpBlock(base_channels * 3, base_channels * 2)
        self.up3 = PreActResidualUpBlock(base_channels * 2, base_channels)
        self.head = nn.Sequential(
            nn.GroupNorm(_norm_groups(base_channels), base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels // 2, 3, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_hw = x.shape[-2:]
        h = self.stem(x)
        h = self.block0(h)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.bottleneck(h)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        out = self.head(h)
        if out.shape[-2:] != target_hw:
            out = F.interpolate(out, size=target_hw, mode="bilinear", align_corners=False)
        return out


__all__ = ["ResNetV2Autoencoder"]
