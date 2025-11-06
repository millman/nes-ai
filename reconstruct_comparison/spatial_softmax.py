from __future__ import annotations

import torch
import torch.nn as nn


class SpatialSoftmax(nn.Module):
    """Compute per-channel spatial softmax expectations."""

    def __init__(self, channels: int, *, normalize: bool = True) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        self.channels = channels
        self.normalize = normalize

    @property
    def output_dim(self) -> int:
        return self.channels * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W); got {x.shape}.")
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(f"Expected {self.channels} channels, received {c}.")
        flat = x.view(b, c, h * w)
        weights = torch.softmax(flat, dim=-1)
        if self.normalize:
            ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
            xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
        else:
            ys = torch.linspace(0.0, float(h - 1), h, device=x.device, dtype=x.dtype)
            xs = torch.linspace(0.0, float(w - 1), w, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_x = grid_x.reshape(1, 1, -1)
        grid_y = grid_y.reshape(1, 1, -1)
        exp_x = torch.sum(weights * grid_x, dim=-1)
        exp_y = torch.sum(weights * grid_y, dim=-1)
        return torch.cat([exp_x, exp_y], dim=-1)


__all__ = ["SpatialSoftmax"]
