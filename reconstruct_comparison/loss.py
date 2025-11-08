from __future__ import annotations

import torch
import torch.nn as nn


class FocalL1Loss(nn.Module):
    """Pixel-wise focal weighting applied to an L1 reconstruction objective."""

    def __init__(self, gamma: float = 2.0, max_weight: float = 5.0, eps: float = 1e-6) -> None:
        super().__init__()
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        if max_weight <= 0:
            raise ValueError("max_weight must be positive.")
        self.gamma = gamma
        self.max_weight = max_weight
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = torch.abs(input - target)
        norm = l1.detach().mean(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps)
        weight = torch.pow(l1 / norm, self.gamma).clamp(max=self.max_weight)
        loss = weight * l1
        return loss.mean()


class HardnessWeightedL1Loss(nn.Module):
    """L1 loss reweighted by per-pixel hardness relative to mean error."""

    def __init__(self, beta: float = 1.5, max_weight: float = 10.0, eps: float = 1e-6) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive.")
        if max_weight <= 0:
            raise ValueError("max_weight must be positive.")
        self.beta = beta
        self.max_weight = max_weight
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = torch.abs(input - target)
        norm = l1.detach().mean(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps)
        hardness = (l1.detach() / norm).clamp_min(0.0)
        weight = torch.pow(hardness, self.beta).clamp(max=self.max_weight)
        return (weight * l1).mean()


__all__ = ["FocalL1Loss", "HardnessWeightedL1Loss"]
