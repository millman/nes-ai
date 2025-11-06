from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .lightweight_autoencoder import LightweightDecoder, LightweightEncoder
from .spatial_softmax import SpatialSoftmax


class LightweightFlatLatentAutoencoder(nn.Module):
    """Lightweight autoencoder that compresses to a flat latent vector."""

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
        self.spatial = SpatialSoftmax(latent_channels)
        combined_dim = latent_channels + self.spatial.output_dim
        self.pre_latent_norm = nn.LayerNorm(combined_dim)
        self.latent_proj = nn.Linear(combined_dim, latent_dim)
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.expand = nn.Linear(latent_dim, latent_channels * self.latent_hw[0] * self.latent_hw[1])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        pooled = self.pool(features).flatten(1)
        spatial = self.spatial(features)
        combined = torch.cat([pooled, spatial], dim=-1)
        combined = self.pre_latent_norm(combined)
        latent = self.latent_proj(combined)
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


__all__ = ["LightweightFlatLatentAutoencoder"]
