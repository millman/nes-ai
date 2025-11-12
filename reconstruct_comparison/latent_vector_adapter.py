from __future__ import annotations

from typing import Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialLatentProjector(nn.Module):
    """Shared module that maps encoder grids to latent vectors and back.

    The projector reduces the encoder's latent grid via adaptive pooling and a
    stack of 1×1 convolutions, then flattens to a vector whose width is fully
    determined by ``projection_channels × latent_spatial²``. The inverse path
    reshapes the vector back into a spatial tensor, expands the channels with
    another 1×1 stack, and upsamples to the decoder's expected ``latent_hw``.
    """

    def __init__(
        self,
        in_channels: int,
        latent_hw: Tuple[int, int],
        latent_spatial: int,
        *,
        projection_channels: int,
        proj_layers: int = 1,
        activation: Type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        if proj_layers < 0:
            raise ValueError("proj_layers must be >= 0")
        height, width = latent_hw
        if height <= 0 or width <= 0:
            raise ValueError("latent_hw must be positive")
        if latent_spatial <= 0:
            raise ValueError("latent_spatial must be positive")
        if projection_channels <= 0:
            raise ValueError("projection_channels must be positive")
        self.latent_hw = latent_hw
        self.in_channels = in_channels
        self.latent_spatial = latent_spatial
        self.pool = nn.AdaptiveAvgPool2d((latent_spatial, latent_spatial))
        spatial_area = latent_spatial * latent_spatial
        flat_dim = projection_channels * spatial_area
        self.projection_channels = projection_channels
        self.flat_dim = flat_dim
        self.latent_dim = flat_dim
        self.channel_down = self._build_channel_stack(
            in_channels, projection_channels, proj_layers, activation
        )
        self.channel_up = self._build_channel_stack(
            projection_channels, in_channels, proj_layers, activation
        )

    @staticmethod
    def _build_channel_stack(
        in_channels: int,
        out_channels: int,
        depth: int,
        activation: Type[nn.Module],
    ) -> nn.Module:
        if depth == 0:
            if in_channels != out_channels:
                raise ValueError(
                    "proj_layers must be >=1 when in_channels != out_channels"
                )
            return nn.Identity()
        layers = []
        current = in_channels
        for idx in range(depth - 1):
            layers.append(nn.Conv2d(current, current, kernel_size=1))
            layers.append(activation())
        layers.append(nn.Conv2d(current, out_channels, kernel_size=1))
        return nn.Sequential(*layers)

    def grid_to_vector(self, grid: torch.Tensor) -> torch.Tensor:
        if grid.dim() != 4:
            raise RuntimeError(
                f"SpatialLatentProjector.grid_to_vector expected 4D tensor, got shape {grid.shape}"
            )
        if grid.size(1) != self.in_channels:
            raise RuntimeError(
                f"Expected input with {self.in_channels} channels, got {grid.size(1)}"
            )
        pooled = self.pool(grid)
        projected = self.channel_down(pooled)
        flat = torch.flatten(projected, start_dim=1)
        return flat

    def vector_to_grid(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.dim() != 2:
            raise RuntimeError(
                f"SpatialLatentProjector.vector_to_grid expected [B, latent_dim], got {latent.shape}"
            )
        if latent.size(1) != self.latent_dim:
            raise RuntimeError(
                f"Expected latent_dim={self.latent_dim}, got {latent.size(1)}"
            )
        feats = latent.view(
            latent.size(0),
            self.projection_channels,
            self.latent_spatial,
            self.latent_spatial,
        )
        feats = self.channel_up(feats)
        if feats.shape[-2:] != self.latent_hw:
            feats = F.interpolate(
                feats,
                size=self.latent_hw,
                mode="bilinear",
                align_corners=False,
            )
        return feats


__all__ = ["SpatialLatentProjector"]
