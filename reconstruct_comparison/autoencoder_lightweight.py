from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DownBlock, UpBlock


class LightweightEncoder(nn.Module):
    """Encoder counterpart without exposing skip connections."""

    def __init__(self, base_channels: int = 48, latent_channels: int = 128) -> None:
        super().__init__()
        if base_channels % 8 != 0:
            raise ValueError("base_channels must be divisible by 8 for GroupNorm stability.")
        self.input_hw = (224, 224)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != self.input_hw:
            raise RuntimeError(
                f"Expected input spatial size {self.input_hw}, got {tuple(x.shape[-2:])}"
            )
        h = self.stem(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        return self.bottleneck(h)


class LightweightDecoder(nn.Module):
    """Decoder that mirrors LightweightEncoder without skip inputs."""

    def __init__(
        self,
        base_channels: int = 48,
        latent_channels: int = 128,
        output_hw: Tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        self.output_hw = output_hw
        self.up1 = UpBlock(latent_channels, base_channels * 3)
        self.up2 = UpBlock(base_channels * 3, base_channels * 2)
        self.up3 = UpBlock(base_channels * 2, base_channels)
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels // 2, 3, kernel_size=1),
        )

    def forward(
        self,
        latent: torch.Tensor,
        *,
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        h = self.up1(latent)
        h = self.up2(h)
        h = self.up3(h)
        out = self.head(h)
        final_hw = target_hw or self.output_hw
        if out.shape[-2:] != final_hw:
            out = F.interpolate(out, size=final_hw, mode="bilinear", align_corners=False)
        return out


class LightweightAutoencoder(nn.Module):
    """Stride-2 lightweight autoencoder without skip connections.

    Rationale:
    - Mirrors the encoder/down-sampler stack (48→96→144→128 channels) with
      matching transpose-convolution blocks so the latent stays 28×28×128 and
      decoding remains inexpensive.
    - Keeps GroupNorm/Sigmoid Linear Units throughout, giving it the same
      inductive bias as the BasicAutoencoder while scaling channel capacity.
    - Encoder output latent: [B, latent_channels, 28, 28] → 28×28×latent_channels
      elements (100,352 values with the default latent_channels=128).

    Total parameters: ≈1.8M learnable weights when base_channels=48 and
    latent_channels=128; changing the base width scales quadratically with the
    channel multipliers defined in the DownBlock/UpBlock pyramid.
    """

    def __init__(self, base_channels: int = 48, latent_channels: int = 128) -> None:
        super().__init__()
        self.encoder = LightweightEncoder(base_channels, latent_channels)
        self.decoder = LightweightDecoder(base_channels, latent_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        recon = self.decoder(latent, target_hw=x.shape[-2:])
        return recon


class LightweightAutoencoderPatch(LightweightAutoencoder):
    """Variant that reuses LightweightAutoencoder for patch-based losses."""

    def __init__(self, base_channels: int = 48, latent_channels: int = 128) -> None:
        super().__init__(base_channels, latent_channels)


__all__ = [
    "LightweightEncoder",
    "LightweightDecoder",
    "LightweightAutoencoder",
    "LightweightAutoencoderPatch",
]
