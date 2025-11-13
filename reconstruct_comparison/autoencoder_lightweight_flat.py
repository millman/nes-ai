from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .autoencoder_lightweight import LightweightDecoder, LightweightEncoder
from .latent_vector_adapter import SpatialLatentProjector


class LightweightFlatAutoencoder(nn.Module):
    """Lightweight autoencoder that compresses to a flat latent vector.

    Rationale:
    - Reuses the LightweightEncoder/Decoder stack (28×28×128 latent grid) but
      pools to a smaller latent_spatial×latent_spatial tensor before flattening,
      mirroring the BasicFlatAutoencoder workflow.
    - A shared SpatialLatentProjector handles pooling, 1×1 channel projections,
      flattening, and the inverse reshape so both the basic and lightweight
      variants can opt into large latent vectors without massive dense layers.
    - Encoder output latent: [B, latent_dim] distilled from the pooled
      [B, latent_channels, latent_spatial, latent_spatial] tensor via:
      1. Adaptive pooling from 28×28×latent_channels to latent_spatial²×latent_channels.
      2. 1×1 conv projection down to latent_conv_channels channels.
      3. Flattening the tensor, which yields latent_conv_channels × latent_spatial²
         scalars and therefore sets latent_dim implicitly.
      With the defaults (latent_spatial=14, latent_conv_channels=128) the latent
      is 14×14×128 = 25,088 elements.

    Total parameters stay near the ≈1.8M conv trunk plus the projector, which
    can now swap between tiny (≈256-d) and large (≈16k-d) latents by only
    adjusting projection settings (latent_spatial/projection_channels).

    Parameter accounting (defaults: base_channels=48, latent_channels=128,
    latent_conv_channels=128, latent_spatial=14):
    - The LightweightEncoder/Decoder pair contains 1,792,611 weights. This sum
      comes from the 3×3 stem (1,440 params), three DownBlocks (124,992 +
      311,904 + 314,112), the bottleneck (147,840), three UpBlocks (482,400 +
      304,704 + 94,752), and the two-layer head (10,467).
    - The SpatialLatentProjector adds 33,024 params. Each of the symmetric 1×1
      stacks is a single Conv2d(128→128, k=1) containing 128×128 weights + 128
      biases = 16,512 parameters, so the down/up pair doubles that figure.
    - Total learnable parameters: 1,792,611 (conv trunk) + 33,024 (projector) =
      1,825,635.
    """

    def __init__(
        self,
        base_channels: int = 48,
        latent_channels: int = 128,
        latent_spatial: int = 14,
        input_hw: Tuple[int, int] = (224, 224),
        latent_conv_channels: int = 128,
        latent_proj_layers: int = 1,
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
        if latent_conv_channels <= 0:
            raise ValueError("latent_conv_channels must be positive")
        spatial_area = latent_spatial * latent_spatial
        self.latent_dim = latent_conv_channels * spatial_area
        self.latent_channels = latent_channels
        self.latent_spatial = latent_spatial
        self.latent_adapter = SpatialLatentProjector(
            latent_channels,
            self.latent_hw,
            latent_spatial,
            projection_channels=latent_conv_channels,
            proj_layers=latent_proj_layers,
        )
        self.encoder = LightweightEncoder(base_channels, latent_channels)
        self.decoder = LightweightDecoder(base_channels, latent_channels, output_hw=input_hw)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.latent_adapter.grid_to_vector(features)

    def decode(
        self,
        latent: torch.Tensor,
        *,
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        expanded = self.latent_adapter.vector_to_grid(latent)
        return self.decoder(expanded, target_hw=target_hw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        return self.decode(latent, target_hw=x.shape[-2:])


__all__ = ["LightweightFlatAutoencoder"]
