from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAutoencoderTrainer
from .autoencoder_best_practice import BestPracticeAutoencoder, _norm_groups


class BestPracticeVectorAutoencoder(nn.Module):
    """Best-practice autoencoder with a flattened latent representation.

    Rationale:
    - Reuses the expressive convolutional backbone so the encoder maintains
      spatial awareness and attention-driven detail.
    - Down-samples the latent grid before flattening so the vector head adds
      only ≈2.8e7 parameters instead of hundreds of millions.

    Total parameters: ≈5.6e7 learnable weights (vector head + backbone).
    """

    def __init__(
        self,
        base_channels: int = 64,
        latent_channels: int = 256,
        latent_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.backbone = BestPracticeAutoencoder(
            base_channels=base_channels,
            latent_channels=latent_channels,
        )
        self.latent_channels = latent_channels
        self.latent_dim = latent_dim
        # Reduce the spatial latent grid: [B, 256, 28, 28] -> [B, 256, 7, 7].
        self.reducer = nn.Sequential(
            # [B, C=256, 28, 28] -> [B, 256, 14, 14]; strided conv halves spatial
            # size while maintaining channels for context gathering.
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_norm_groups(latent_channels), latent_channels),
            nn.SiLU(inplace=True),
            # [B, 256, 14, 14] -> [B, 256, 7, 7]; second stride prepares compact grid
            # for the latent projection without overwhelming parameter count.
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_norm_groups(latent_channels), latent_channels),
            nn.SiLU(inplace=True),
        )
        self.reduced_hw = 7
        reduced_dim = latent_channels * self.reduced_hw * self.reduced_hw
        # Flattened latent projection: [B, reduced_dim] -> [B, latent_dim].
        self.to_latent = nn.Linear(reduced_dim, latent_dim)
        # Learned unflattening back to pooled spatial tensor for the decoder path.
        self.from_latent = nn.Linear(latent_dim, reduced_dim)
        self.expander = nn.Sequential(
            # [B, 256, 7, 7] -> [B, 256, 14, 14]; learnable up-sampling mirrors reducer.
            nn.ConvTranspose2d(latent_channels, latent_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_norm_groups(latent_channels), latent_channels),
            nn.SiLU(inplace=True),
            # [B, 256, 14, 14] -> [B, 256, 28, 28]; restores latent grid for decoder.
            nn.ConvTranspose2d(latent_channels, latent_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_norm_groups(latent_channels), latent_channels),
            nn.SiLU(inplace=True),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.encode(x)
        reduced = self.reducer(feats)
        flat = reduced.view(reduced.shape[0], -1)
        return self.to_latent(flat)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        feats = self.from_latent(latent)
        feats = feats.view(
            latent.shape[0], self.latent_channels, self.reduced_hw, self.reduced_hw
        )
        expanded = self.expander(feats)
        return self.backbone.decode(expanded)

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
        loss_fn: nn.Module,
        base_channels: int = 64,
        latent_channels: int = 256,
        latent_dim: int = 1024,
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
