from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAutoencoderTrainer
from .best_practice_autoencoder import BestPracticeAutoencoder


class BestPracticeVectorAutoencoder(nn.Module):
    """Best-practice autoencoder with a flattened latent representation.

    Rationale:
    - Reuses the expressive convolutional backbone so the encoder maintains
      spatial awareness and attention-driven detail.
    - Adds projection heads so downstream agents can consume/produce a compact
      latent vector while the decoder still benefits from the structured latent
      grid.

    Total parameters: ≈2.35e8 learnable weights (dominated by the latent MLP).
    """

    def __init__(
        self,
        base_channels: int = 64,
        latent_channels: int = 256,
        latent_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.backbone = BestPracticeAutoencoder(
            base_channels=base_channels,
            latent_channels=latent_channels,
        )
        self.latent_channels = latent_channels
        self.spatial_hw = 28
        self.latent_dim = latent_dim
        # Flattened latent projection: [B, C=256, H=W=28] -> [B, latent_dim].
        self.to_latent = nn.Linear(latent_channels * self.spatial_hw * self.spatial_hw, latent_dim)
        # Learned unflattening back to spatial tensor for the decoder path.
        self.from_latent = nn.Linear(latent_dim, latent_channels * self.spatial_hw * self.spatial_hw)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.encode(x)
        flat = feats.view(feats.shape[0], -1)
        return self.to_latent(flat)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        feats = self.from_latent(latent)
        feats = feats.view(latent.shape[0], self.latent_channels, self.spatial_hw, self.spatial_hw)
        return self.backbone.decode(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        recon = self.decode(latent)
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(recon, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return recon


class BestPracticeVectorAutoencoderTrainer(BaseAutoencoderTrainer):
    """Trainer for the vector-latent best-practice autoencoder."""

    def __init__(
        self,
        *,
        device: torch.device,
        lr: float,
        loss_fn: Optional[nn.Module] = None,
        base_channels: int = 64,
        latent_channels: int = 256,
        latent_dim: int = 2048,
        weight_decay: float = 1e-4,
        name: str = "best_practice_vector_autoencoder",
    ) -> None:
        model = BestPracticeVectorAutoencoder(
            base_channels=base_channels,
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


__all__ = ["BestPracticeVectorAutoencoder", "BestPracticeVectorAutoencoderTrainer"]
