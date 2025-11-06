from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAutoencoderTrainer


class BarebonesVectorAutoencoder(nn.Module):
    """Barebones autoencoder that exposes a flattened latent vector.

    Rationale:
    - Keeps the same light-weight convolutional tower as the spatial model so
      features remain cheap to compute and easy to debug.
    - Adds a bottleneck MLP so downstream components can consume a single
      flattened latent vector without any spatial reasoning.

    Total parameters: ≈1.03e8 learnable weights (dominated by the linear heads).
    """

    def __init__(self, latent_channels: int = 64, latent_dim: int = 1024) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_dim = latent_dim
        self.encoder_conv = nn.Sequential(
            # [B, 3, 224, 224] -> [B, 32, 112, 112]; stride-2 conv for cheap
            # down-sampling while expanding representational capacity.
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # [B, 32, 112, 112] -> [B, 48, 56, 56]; builds mid-level features.
            nn.Conv2d(32, 48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # [B, 48, 56, 56] -> [B, latent_channels, 28, 28]; compact spatial
            # latent keeps decoding simple while feeding the MLP.
            nn.Conv2d(48, latent_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.flatten = nn.Flatten()
        # Linear bottleneck: [B, latent_channels*28*28] -> [B, latent_dim].
        self.to_latent = nn.Linear(latent_channels * 28 * 28, latent_dim)
        # Linear projection back to spatial tensor for the decoder.
        self.from_latent = nn.Linear(latent_dim, latent_channels * 28 * 28)
        self.decoder_conv = nn.Sequential(
            # [B, latent_channels, 28, 28] -> [B, 48, 56, 56].
            nn.ConvTranspose2d(latent_channels, 48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # [B, 48, 56, 56] -> [B, 32, 112, 112].
            nn.ConvTranspose2d(48, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # [B, 32, 112, 112] -> [B, 3, 224, 224].
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder_conv(x)
        flat = self.flatten(feats)
        latent = self.to_latent(flat)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        feats = self.from_latent(latent)
        feats = feats.view(latent.shape[0], self.latent_channels, 28, 28)
        recon = self.decoder_conv(feats)
        if recon.shape[-2:] != (224, 224):
            recon = F.interpolate(recon, size=(224, 224), mode="bilinear", align_corners=False)
        return recon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        recon = self.decode(latent)
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(recon, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return recon


class BarebonesVectorAutoencoderTrainer(BaseAutoencoderTrainer):
    """Trainer for the barebones vector-latent autoencoder."""

    def __init__(
        self,
        *,
        device: torch.device,
        lr: float,
        loss_fn: Optional[nn.Module] = None,
        latent_channels: int = 64,
        latent_dim: int = 1024,
        weight_decay: float = 0.0,
        name: str = "barebones_vector_autoencoder",
    ) -> None:
        model = BarebonesVectorAutoencoder(
            latent_channels=latent_channels,
            latent_dim=latent_dim,
        )
        super().__init__(
            name,
            model,
            device=device,
            lr=lr,
            loss_fn=loss_fn,
            weight_decay=weight_decay,
        )


__all__ = ["BarebonesVectorAutoencoder", "BarebonesVectorAutoencoderTrainer"]
