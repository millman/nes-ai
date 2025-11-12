from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .autoencoder_basic import BasicDecoder, BasicEncoder


class BasicFlatAutoencoder(nn.Module):
    """Basic autoencoder that exposes a flattened latent vector.

    Rationale:
    - Reuses :class:`BasicEncoder`/:class:`BasicDecoder` so the convolutional trunk
      matches the spatial autoencoder while remaining easily testable in isolation.
    - Uses adaptive pooling before the linear bottleneck so the latent vector
      stays tractable; flattening the full 28×28×128 tensor would otherwise add
      ≈52M parameters to the latent MLP and proved numerically brittle.
    - Encoder output latent: [B, latent_dim] (default 256) produced from a pooled
      [B, latent_channels, pooled_spatial, pooled_spatial] tensor (25,088 values
      before the linear layer when using the defaults).

    Total parameters: ≈13M learnable weights when latent_channels=128,
    latent_dim=256, and latent_spatial=14; almost all of those weights live in
    the latent MLP, so shrinking latent_spatial or latent_dim further cuts the
    total while leaving the convolutional trunk unchanged.
    """

    def __init__(
        self,
        latent_channels: int = 128,
        latent_dim: int = 256,
        latent_spatial: int = 14,
        input_hw: Tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        if input_hw[0] % 8 != 0 or input_hw[1] % 8 != 0:
            raise ValueError("input_hw must be divisible by 8 to match encoder strides.")
        latent_hw = (input_hw[0] // 8, input_hw[1] // 8)
        if latent_spatial > latent_hw[0] or latent_spatial > latent_hw[1]:
            raise ValueError(
                f"latent_spatial ({latent_spatial}) must be <= latent grid {latent_hw}"
            )
        self.latent_channels = latent_channels
        self.latent_dim = latent_dim
        self.latent_spatial = latent_spatial
        self.latent_hw = latent_hw
        self.encoder = BasicEncoder(latent_channels, input_hw)
        self.decoder = BasicDecoder(latent_channels, latent_hw)
        # Pool to smaller spatial grid: [B, latent_channels, H/8, W/8] ->
        # [B, latent_channels, latent_spatial, latent_spatial].
        self.pool = nn.AdaptiveAvgPool2d((latent_spatial, latent_spatial))
        # Flatten pooled features and map to latent vector.
        self.flatten = nn.Flatten()
        # Linear bottleneck: [B, latent_channels*latent_spatial^2] -> [B, latent_dim].
        self.to_latent = nn.Linear(
            latent_channels * latent_spatial * latent_spatial, latent_dim
        )
        # Linear projection back to pooled spatial tensor for the decoder.
        self.from_latent = nn.Linear(
            latent_dim, latent_channels * latent_spatial * latent_spatial
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        pooled = self.pool(feats)
        flat = self.flatten(pooled)
        latent = self.to_latent(flat)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        feats = self.from_latent(latent)
        feats = feats.view(
            latent.shape[0], self.latent_channels, self.latent_spatial, self.latent_spatial
        )
        feats = F.interpolate(
            feats,
            size=self.latent_hw,
            mode="bilinear",
            align_corners=False,
        )
        return self.decoder(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        recon = self.decode(latent)
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(recon, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return recon
__all__ = ["BasicFlatAutoencoder"]
