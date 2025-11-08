from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CauchyLoss(nn.Module):
    """Robust loss based on the negative log-likelihood of the Cauchy distribution."""

    def __init__(self, sigma: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.sigma = sigma
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = (input - target) / self.sigma
        return torch.log1p(diff.pow(2)).mean().clamp_min(self.eps)


class MultiScalePatchLoss(nn.Module):
    """Aggregate MSE over local patches across a pyramid of spatial scales."""

    def __init__(
        self,
        patch_sizes: Sequence[int] = (7, 11, 15),
        pool_scales: Sequence[int] = (1, 2, 4),
    ) -> None:
        super().__init__()
        if not patch_sizes:
            raise ValueError("patch_sizes must be non-empty.")
        if not pool_scales:
            raise ValueError("pool_scales must be non-empty.")
        if len(patch_sizes) != len(pool_scales):
            raise ValueError("patch_sizes and pool_scales must be the same length.")
        if any(size <= 0 for size in patch_sizes):
            raise ValueError("patch_sizes must contain positive integers.")
        if any(scale <= 0 for scale in pool_scales):
            raise ValueError("pool_scales must contain positive integers.")
        self.patch_sizes = tuple(int(size) for size in patch_sizes)
        self.pool_scales = tuple(int(scale) for scale in pool_scales)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.shape != target.shape:
            raise ValueError("input and target must share the same shape.")
        total_loss = input.new_tensor(0.0)
        total_weight = 0
        for patch_size, pool_scale in zip(self.patch_sizes, self.pool_scales):
            if pool_scale > 1:
                pooled_input = F.avg_pool2d(input, kernel_size=pool_scale, stride=pool_scale)
                pooled_target = F.avg_pool2d(target, kernel_size=pool_scale, stride=pool_scale)
            else:
                pooled_input = input
                pooled_target = target
            k = min(patch_size, pooled_input.shape[-2], pooled_input.shape[-1])
            if k <= 0:
                raise RuntimeError("Patch size became non-positive after clamping.")
            stride = max(1, k // 2)
            padding = k // 2
            unfolded_input = F.unfold(pooled_input, kernel_size=k, stride=stride, padding=padding)
            unfolded_target = F.unfold(pooled_target, kernel_size=k, stride=stride, padding=padding)
            if unfolded_input.shape[-1] == 0:
                patch_loss = F.mse_loss(pooled_input, pooled_target)
            else:
                diff = unfolded_input - unfolded_target
                patch_loss = diff.pow(2).mean()
            total_loss = total_loss + patch_loss
            total_weight += 1
        if total_weight == 0:
            raise RuntimeError("No scales contributed to the loss computation.")
        return total_loss / total_weight


__all__ = [
    "FocalL1Loss",
    "HardnessWeightedL1Loss",
    "CauchyLoss",
    "MultiScalePatchLoss",
]
