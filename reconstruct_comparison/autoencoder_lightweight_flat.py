from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .autoencoder_lightweight import LightweightDecoder, LightweightEncoder


class LightweightFlatAutoencoder(nn.Module):
    """Lightweight autoencoder that compresses to a flat latent vector.

    Rationale:
    - Reuses the LightweightEncoder/Decoder stack (28×28×128 latent grid) but
      pools to a channel-wise summary before projecting into a compact vector.
    - The latent vector expands back to the 28×28×128 grid with a single linear
      layer, mirroring the BasicFlatAutoencoder workflow while keeping the
      convolutional trunk lightweight.
    - Encoder output latent: [B, latent_dim] (default 256) distilled from the
      pooled [B, latent_channels, 28, 28] tensor (100,352 pre-pooled values with
      default geometry).

    Total parameters: ≈27.6M learnable weights when base_channels=48,
    latent_channels=128, and latent_dim=256; ≈25.8M of those live in the
    latent expansion Linear layer, so reducing `latent_dim` or the encoder’s
    spatial resolution quickly shrinks the overall count.
    """

    def __init__(
        self,
        base_channels: int = 48,
        latent_channels: int = 128,
        latent_dim: int = 256,
        input_hw: Tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        self.encoder = LightweightEncoder(base_channels, latent_channels)
        self.decoder = LightweightDecoder(base_channels, latent_channels, output_hw=input_hw)
        self.latent_channels = latent_channels
        height, width = input_hw
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("input_hw dimensions must be divisible by 8 to match encoder strides.")
        self.latent_hw = (height // 8, width // 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pre_latent_norm = nn.LayerNorm(latent_channels)
        self.latent_proj = nn.Linear(latent_channels, latent_dim)
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.expand = nn.Linear(latent_dim, latent_channels * self.latent_hw[0] * self.latent_hw[1])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        pooled = self.pool(features).flatten(1)
        pooled = self.pre_latent_norm(pooled)
        latent = self.latent_proj(pooled)
        return self.latent_norm(latent)

    def decode(
        self,
        latent: torch.Tensor,
        *,
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        latent = self.latent_norm(latent)
        expanded = self.expand(latent)
        expanded = expanded.view(
            latent.size(0),
            self.latent_channels,
            self.latent_hw[0],
            self.latent_hw[1],
        )
        return self.decoder(expanded, target_hw=target_hw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        return self.decode(latent, target_hw=x.shape[-2:])


__all__ = ["LightweightFlatAutoencoder"]
