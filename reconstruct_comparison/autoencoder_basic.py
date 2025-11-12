from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicEncoder(nn.Module):
    """Minimal stride-2 convolutional encoder used by the basic autoencoder."""

    def __init__(
        self,
        latent_channels: int = 128,
        input_hw: Tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        height, width = input_hw
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("input_hw must be divisible by 8 to match encoder strides.")
        self.input_hw = input_hw
        self.latent_hw = (height // 8, width // 8)
        self.net = nn.Sequential(
            # [B, 3, H, W] -> [B, 32, H/2, W/2]; stride-2 conv halves spatial size
            # while expanding to 32 channels for basic edge/colour detectors.
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # [B, 32, H/2, W/2] -> [B, 48, H/4, W/4]; maintains cheap compute while
            # increasing channels for richer mid-level features.
            nn.Conv2d(32, 48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # [B, 48, H/4, W/4] -> [B, latent_channels, H/8, W/8]; latent tensor stays
            # spatially compact yet preserves layout for easy decoding.
            nn.Conv2d(48, latent_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != self.input_hw:
            raise RuntimeError(
                f"Expected input spatial size {self.input_hw}, got {tuple(x.shape[-2:])}"
            )
        return self.net(x)


class BasicDecoder(nn.Module):
    """Transpose-convolution decoder that mirrors :class:`BasicEncoder`."""

    def __init__(
        self,
        latent_channels: int = 128,
        latent_hw: Tuple[int, int] = (28, 28),
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_hw = latent_hw
        self.net = nn.Sequential(
            # Mirrors encoder: [B, latent_channels, H, W] -> [B, 48, 2H, 2W].
            nn.ConvTranspose2d(latent_channels, 48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # [B, 48, 2H, 2W] -> [B, 32, 4H, 4W]; progressive up-sampling.
            nn.ConvTranspose2d(48, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # [B, 32, 4H, 4W] -> [B, 3, 8H, 8W]; reconstruct RGB image.
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[-3] != self.latent_channels:
            raise RuntimeError(
                f"Expected {self.latent_channels} latent channels, got {latent.shape[-3]}"
            )
        if latent.shape[-2:] != self.latent_hw:
            raise RuntimeError(
                f"Expected latent spatial size {self.latent_hw}, got {tuple(latent.shape[-2:])}"
            )
        return self.net(latent)


class BasicAutoencoder(nn.Module):
    """Minimal stride-2 convolutional autoencoder composed from encoder+decoder.

    Rationale:
    - Uses only strided convolutions with ReLU activations so the encoder is
      extremely cheap while still down-sampling spatially for compression.
    - Mirrors the encoder with transpose convolutions so every latent feature
      directly informs pixels without expensive skip connections.
    - Encoder output latent: [B, latent_channels, 28, 28] → 28×28×latent_channels
      elements (100,352 values when latent_channels=128).

    Total parameters: ≈250k learnable weights with the default latent_channels=128.
    Each extra latent channel adds ≈1.5k weights because only the last encoder
    and first decoder layers widen, while the count stays independent of the
    224×224 RGB input resolution.
    """

    def __init__(
        self,
        latent_channels: int = 128,
        input_hw: Tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        if input_hw[0] % 8 != 0 or input_hw[1] % 8 != 0:
            raise ValueError("input_hw must be divisible by 8 to match encoder strides.")
        latent_hw = (input_hw[0] // 8, input_hw[1] // 8)
        self.encoder = BasicEncoder(latent_channels, input_hw)
        self.decoder = BasicDecoder(latent_channels, latent_hw)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        recon = self.decoder(latent)
        scale = 2 ** 3
        target_hw = (latent.shape[-2] * scale, latent.shape[-1] * scale)
        if recon.shape[-2:] != target_hw:
            # ConvTranspose restores the canonical 224×224 resolution when inputs follow
            # the expected spatial stride progression. Guard against mismatched shapes
            # during validation or scripted export by interpolating.
            recon = F.interpolate(recon, size=target_hw, mode="bilinear", align_corners=False)
        return recon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        recon = self.decode(latent)
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(recon, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return recon


__all__ = ["BasicAutoencoder", "BasicEncoder", "BasicDecoder"]
