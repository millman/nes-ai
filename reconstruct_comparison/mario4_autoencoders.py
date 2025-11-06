from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from predict_mario4 import (
    ImageDecoder as Mario4ImageDecoder,
    ImageDecoderMirrored as Mario4ImageDecoderMirrored,
    ImageEncoder as Mario4ImageEncoder,
)

from .spatial_softmax import SpatialSoftmax


class _Mario4AutoencoderBase(nn.Module):
    """Shared normalisation wrapper for Mario4-derived autoencoders."""

    def __init__(
        self,
        *,
        decoder: nn.Module,
        latent_dim: int = 192,
        encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else Mario4ImageEncoder(latent_dim)
        self.decoder = decoder
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        self.register_buffer("_mean", mean.view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", std.view(1, 3, 1, 1), persistent=False)

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._std.to(dtype=x.dtype, device=x.device) + self._mean.to(
            dtype=x.dtype, device=x.device
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean.to(dtype=x.dtype, device=x.device)) / self._std.to(
            dtype=x.dtype, device=x.device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self._denormalize(x)
        latent = self.encoder(raw)
        recon_raw = self.decoder(latent)
        return self._normalize(recon_raw)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raw = self._denormalize(x)
        return self.encoder(raw)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        recon_raw = self.decoder(latent)
        return self._normalize(recon_raw)


class Mario4Autoencoder(_Mario4AutoencoderBase):
    """Autoencoder built from predict_mario4's baseline decoder."""

    def __init__(self, latent_dim: int = 192, base_channels: int = 32) -> None:
        decoder = Mario4ImageDecoder(latent_dim, base=base_channels)
        super().__init__(decoder=decoder, latent_dim=latent_dim)


class Mario4MirroredAutoencoder(_Mario4AutoencoderBase):
    """Autoencoder using the mirrored ImageDecoder for spatial upsampling."""

    def __init__(self, latent_dim: int = 192, initial_hw: int = 14) -> None:
        decoder = Mario4ImageDecoderMirrored(latent_dim, initial_hw=initial_hw)
        super().__init__(decoder=decoder, latent_dim=latent_dim)


class Mario4ImageDecoderMirroredLarge(Mario4ImageDecoderMirrored):
    """Mario4 mirrored decoder defaulting to a 1024-dimensional latent."""

    def __init__(self, latent_dim: int = 1024, initial_hw: int = 14) -> None:
        super().__init__(latent_dim=latent_dim, initial_hw=initial_hw)


class Mario4SpatialSoftmaxEncoder(nn.Module):
    """Mario4 encoder variant that applies spatial softmax before projection."""

    def __init__(self, latent_dim: int = 192) -> None:
        super().__init__()
        base = Mario4ImageEncoder(latent_dim)
        self.features = base.features
        self.pool = base.pool
        feature_dim = base.proj.in_features
        self.spatial_pool = SpatialSoftmax(feature_dim)
        concat_dim = feature_dim + self.spatial_pool.output_dim
        self.proj = nn.Linear(concat_dim, latent_dim)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        x = self.features(frame)
        pooled = self.pool(x).flatten(1)
        spatial = self.spatial_pool(x)
        combined = torch.cat([pooled, spatial], dim=-1)
        return self.proj(combined)


class Mario4SpatialSoftmaxAutoencoder(_Mario4AutoencoderBase):
    """Autoencoder pairing the Mario4 decoder with a spatial-softmax encoder head."""

    def __init__(self, latent_dim: int = 192, initial_hw: int = 14) -> None:
        encoder = Mario4SpatialSoftmaxEncoder(latent_dim)
        decoder = Mario4ImageDecoderMirrored(latent_dim, initial_hw=initial_hw)
        super().__init__(decoder=decoder, latent_dim=latent_dim, encoder=encoder)


class Mario4SpatialSoftmaxLargeAutoencoder(_Mario4AutoencoderBase):
    """Spatial-softmax Mario4 autoencoder variant with a 1024-D latent."""

    def __init__(self, latent_dim: int = 1024, initial_hw: int = 14) -> None:
        encoder = Mario4SpatialSoftmaxEncoder(latent_dim)
        decoder = Mario4ImageDecoderMirroredLarge(latent_dim=latent_dim, initial_hw=initial_hw)
        super().__init__(decoder=decoder, latent_dim=latent_dim, encoder=encoder)


class Mario4LargeAutoencoder(_Mario4AutoencoderBase):
    """Mario4 autoencoder variant with enlarged latent dimensionality."""

    def __init__(self, latent_dim: int = 1024, base_channels: int = 32) -> None:
        decoder = Mario4ImageDecoder(latent_dim, base=base_channels)
        super().__init__(decoder=decoder, latent_dim=latent_dim)


__all__ = [
    "Mario4Autoencoder",
    "Mario4MirroredAutoencoder",
    "Mario4SpatialSoftmaxAutoencoder",
    "Mario4SpatialSoftmaxLargeAutoencoder",
    "Mario4LargeAutoencoder",
]
