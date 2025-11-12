from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .autoencoder_lightweight import LightweightDecoder, LightweightEncoder


class LightweightFlatAutoencoder(nn.Module):
    """Lightweight autoencoder that compresses to a flat latent vector.

    Rationale:
    - Reuses the LightweightEncoder/Decoder stack (28×28×128 latent grid) but
      pools to a smaller latent_spatial×latent_spatial tensor before flattening,
      mirroring the BasicFlatAutoencoder workflow.
    - A single Linear layer maps the pooled tensor to the latent vector and a
      matching Linear reverses the operation before resizing back to the encoder
      grid, making the flatten/unflatten path easy to follow.
    - Encoder output latent: [B, latent_dim] (default 256) distilled from the
      pooled [B, latent_channels, latent_spatial, latent_spatial] tensor (e.g.
      25,088 values when latent_spatial=14 and latent_channels=128).

    Total parameters: ≈27.6M learnable weights when base_channels=48,
    latent_channels=128, latent_spatial=14, and latent_dim=256; ≈25.8M of those
    live in the latent MLP, so shrinking latent_spatial or latent_dim quickly
    reduces the total while leaving the convolutional trunk unchanged.
    """

    def __init__(
        self,
        base_channels: int = 48,
        latent_channels: int = 128,
        latent_dim: int = 256,
        latent_spatial: int = 14,
        input_hw: Tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        height, width = input_hw
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("input_hw dimensions must be divisible by 8 to match encoder strides.")
        self.latent_hw = (height // 8, width // 8)
        if latent_spatial > self.latent_hw[0] or latent_spatial > self.latent_hw[1]:
            raise ValueError(
                f"latent_spatial ({latent_spatial}) must be <= encoder latent grid {self.latent_hw}"
            )
        self.encoder = LightweightEncoder(base_channels, latent_channels)
        self.decoder = LightweightDecoder(base_channels, latent_channels, output_hw=input_hw)
        self.latent_channels = latent_channels
        self.latent_dim = latent_dim
        self.latent_spatial = latent_spatial
        self.pool = nn.AdaptiveAvgPool2d((latent_spatial, latent_spatial))
        self.flatten = nn.Flatten()
        self.to_latent = nn.Linear(
            latent_channels * latent_spatial * latent_spatial,
            latent_dim,
        )
        self.from_latent = nn.Linear(
            latent_dim,
            latent_channels * latent_spatial * latent_spatial,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        pooled = self.pool(features)
        flat = self.flatten(pooled)
        return self.to_latent(flat)

    def decode(
        self,
        latent: torch.Tensor,
        *,
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        expanded = self.from_latent(latent)
        expanded = expanded.view(
            latent.size(0),
            self.latent_channels,
            self.latent_spatial,
            self.latent_spatial,
        )
        expanded = F.interpolate(
            expanded,
            size=self.latent_hw,
            mode="bilinear",
            align_corners=False,
        )
        return self.decoder(expanded, target_hw=target_hw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        return self.decode(latent, target_hw=x.shape[-2:])


__all__ = ["LightweightFlatAutoencoder"]
