from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import UpBlock


class Decoder(nn.Module):
    """Shared decoder architecture parameterised by encoder channel width."""

    def __init__(self, in_channels: int, *, base_channels: int = 512) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=1),
            nn.SiLU(inplace=True),
        )
        self.up1 = UpBlock(base_channels, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 96)
        self.up4 = UpBlock(96, 64)
        self.up5 = UpBlock(64, 48)
        self.head = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = self.proj(feat)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        h = self.up5(h)
        out = self.head(h)
        if out.shape[-2:] != (224, 224):
            out = F.interpolate(out, size=(224, 224), mode="bilinear", align_corners=False)
        return out


__all__ = ["Decoder"]
