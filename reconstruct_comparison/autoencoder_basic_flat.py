from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .autoencoder_basic import BasicDecoder, BasicEncoder
from .latent_vector_adapter import SpatialLatentProjector


class BasicFlatAutoencoder(nn.Module):
    """Basic autoencoder that exposes a flattened latent vector.

    Rationale:
    - Reuses :class:`BasicEncoder`/:class:`BasicDecoder` so the convolutional trunk
      matches the spatial autoencoder while remaining easily testable in isolation.
    - Uses the shared SpatialLatentProjector (pool → 1×1 conv stack → flatten)
      so the latent dimensionality is purely conv-controlled rather than driven
      by a dense bottleneck.
    - Encoder output latent: [B, latent_dim] produced from a pooled
      [B, latent_channels, pooled_spatial, pooled_spatial] tensor.
      1. Adaptive pooling shrinks 28×28×latent_channels → latent_spatial²×latent_channels.
      2. A stack of 1×1 convolutions reduces channels to latent_conv_channels.
      3. Flattening yields latent_conv_channels × latent_spatial² scalars, which
         define latent_dim automatically.
      With the defaults (latent_spatial=14, latent_conv_channels=128) this becomes
      28×28 → 14×14 pooling, 128 projected channels, and a 128×14×14 = 25,088 element latent.

    The total parameters are now dominated by the basic conv trunk plus the
    small projector, so dialing projection settings up or down remains cheap.

    Parameter accounting (defaults: latent_channels=128, latent_conv_channels=128,
    latent_spatial=14):
    - The BasicEncoder supplies 124,624 weights (three stride-2 Conv2d layers).
      The mirrored BasicDecoder adds 124,499 weights (three ConvTranspose2d
      layers), for 249,123 parameters across the convolutional trunk.
    - The SpatialLatentProjector contributes 33,024 parameters: each 1×1
      Conv2d(128→128) used in the down/up stacks holds 128×128 weights + 128
      biases = 16,512 params, and there are two such stacks.
    - Total learnable parameters: 249,123 (encoder/decoder) + 33,024 (projector)
      = 282,147.
    """

    def __init__(
        self,
        latent_channels: int = 128,
        latent_spatial: int = 14,
        input_hw: Tuple[int, int] = (224, 224),
        latent_conv_channels: int = 128,
        latent_proj_layers: int = 1,
    ) -> None:
        super().__init__()
        if input_hw[0] % 8 != 0 or input_hw[1] % 8 != 0:
            raise ValueError("input_hw must be divisible by 8 to match encoder strides.")
        latent_hw = (input_hw[0] // 8, input_hw[1] // 8)
        if latent_spatial > latent_hw[0] or latent_spatial > latent_hw[1]:
            raise ValueError(
                f"latent_spatial ({latent_spatial}) must be <= latent grid {latent_hw}"
            )
        if latent_conv_channels <= 0:
            raise ValueError("latent_conv_channels must be positive")
        spatial_area = latent_spatial * latent_spatial
        self.latent_dim = latent_conv_channels * spatial_area
        self.latent_channels = latent_channels
        self.latent_spatial = latent_spatial
        self.latent_hw = latent_hw
        self.latent_adapter = SpatialLatentProjector(
            latent_channels,
            latent_hw,
            latent_spatial,
            projection_channels=latent_conv_channels,
            proj_layers=latent_proj_layers,
        )
        self.encoder = BasicEncoder(latent_channels, input_hw)
        self.decoder = BasicDecoder(latent_channels, latent_hw)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return self.latent_adapter.grid_to_vector(feats)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        feats = self.latent_adapter.vector_to_grid(latent)
        return self.decoder(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        recon = self.decode(latent)
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(recon, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return recon
__all__ = ["BasicFlatAutoencoder"]
