from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DownBlock, UpBlock


class LightweightAutoencoderUNetSkipTrain(nn.Module):
    """Trains with skip connections but removes them during evaluation/inference."""

    def __init__(self, base_channels: int = 48, latent_channels: int = 128) -> None:
        super().__init__()
        if base_channels % 8 != 0:
            raise ValueError("base_channels must be divisible by 8 for GroupNorm stability.")
        self.output_hw = (224, 224)
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(inplace=True),
        )
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 3)
        self.down3 = DownBlock(base_channels * 3, latent_channels)
        groups = 8 if latent_channels % 8 == 0 else 1
        self.bottleneck = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, latent_channels),
            nn.SiLU(inplace=True),
        )
        self.up1 = UpBlock(latent_channels, base_channels * 3)
        self.up2 = UpBlock(base_channels * 3, base_channels * 2)
        self.up3 = UpBlock(base_channels * 2, base_channels)
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels // 2, 3, kernel_size=1),
        )

    def _encode_with_skips(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if x.shape[-2:] != self.output_hw:
            raise RuntimeError(
                f"Expected input spatial size {self.output_hw}, got {tuple(x.shape[-2:])}"
            )
        h0 = self.stem(x)
        h1 = self.down1(h0)
        h2 = self.down2(h1)
        h3 = self.down3(h2)
        latent = self.bottleneck(h3)
        return latent, (h0, h1, h2)

    def _decode(
        self,
        latent: torch.Tensor,
        *,
        skips: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        h0 = h1 = h2 = None
        if skips is not None:
            h0, h1, h2 = skips
        u1 = self.up1(latent)
        if h2 is not None:
            u1 = u1 + h2
        u2 = self.up2(u1)
        if h1 is not None:
            u2 = u2 + h1
        u3 = self.up3(u2)
        if h0 is not None:
            u3 = u3 + h0
        out = self.head(u3)
        final_hw = target_hw or self.output_hw
        if out.shape[-2:] != final_hw:
            out = F.interpolate(out, size=final_hw, mode="bilinear", align_corners=False)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, _ = self._encode_with_skips(x)
        return self._decode(latent, target_hw=x.shape[-2:])

    def train_forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, skips = self._encode_with_skips(x)
        return self._decode(latent, skips=skips, target_hw=x.shape[-2:])


__all__ = ["LightweightAutoencoderUNetSkipTrain"]
