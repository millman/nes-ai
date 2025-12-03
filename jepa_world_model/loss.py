from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "HardnessWeightedL1Loss",
    "HardnessWeightedMSELoss",
    "HardnessWeightedMedianLoss",
    "FocalL1Loss",
]


class HardnessWeightedL1Loss(nn.Module):
    """L1 loss with per-sample hardness weighting derived from mean error."""

    def __init__(self, beta: float = 2.0, max_weight: float = 100.0, eps: float = 1e-6) -> None:
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
        dims = tuple(range(1, l1.dim()))
        norm = l1.detach().mean(dim=dims, keepdim=True).clamp_min(self.eps)
        rel_error = l1.detach() / norm
        weight = rel_error.pow(self.beta).clamp(max=self.max_weight)
        return (weight * l1).mean()


class HardnessWeightedMSELoss(nn.Module):
    """Simple hardness-weighted MSE mirroring the L1 variant."""

    def __init__(self, beta: float = 2.0, max_weight: float = 100.0, eps: float = 1e-6) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive.")
        if max_weight <= 0:
            raise ValueError("max_weight must be positive.")
        self.beta = beta
        self.max_weight = max_weight
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.shape != target.shape:
            raise ValueError("input and target must share shape")
        mse = (input - target).pow(2)
        dims = tuple(range(1, mse.dim()))
        norm = mse.detach().mean(dim=dims, keepdim=True).clamp_min(self.eps)
        rel_error = mse.detach() / norm
        weight = rel_error.pow(self.beta).clamp(max=self.max_weight)
        return (weight * mse).mean()


class HardnessWeightedMedianLoss(nn.Module):
    """Twin of :class:`HardnessWeightedL1Loss` using a median baseline on residual deltas."""

    def __init__(self, beta: float = 2.0, max_weight: float = 100.0, eps: float = 1e-6) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive.")
        if max_weight <= 0:
            raise ValueError("max_weight must be positive.")
        self.beta = beta
        self.max_weight = max_weight
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.shape != target.shape:
            raise ValueError("input and target must share shape")
        residual = (input - target).abs()
        b = residual.shape[0]
        flat = residual.detach().flatten(start_dim=1)
        median = flat.median(dim=1).values.view(b, *((1,) * (residual.dim() - 1))).clamp_min(self.eps)
        rel_error = residual.detach() / median
        weight = rel_error.pow(self.beta).clamp(max=self.max_weight)
        return (weight * residual).mean()


class FocalL1Loss(nn.Module):
    """Focal-style L1 reconstruction loss that upweights harder pixels."""

    def __init__(self, gamma: float = 2.0, max_weight: float = 100.0, eps: float = 1e-6) -> None:
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
        dims = tuple(range(1, l1.dim()))
        norm = l1.detach().mean(dim=dims, keepdim=True).clamp_min(self.eps)
        rel_error = l1 / norm
        weight = rel_error.pow(self.gamma).clamp(max=self.max_weight)
        return (weight * l1).mean()
