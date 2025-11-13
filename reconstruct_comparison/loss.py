from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial_softmax import SpatialSoftmax


class FocalL1Loss(nn.Module):
    """Focal-style L1 reconstruction loss that upweights harder pixels.

    Loss:
        L = mean( w_i * |x_i - y_i| )
    with weights
        w_i = min(max_weight, (|x_i - y_i| / (mean_batch + eps)) ** gamma)
    where mean_batch is the per-sample mean absolute error over all pixels.

    The numerator stays attached to the graph so the focal weights receive
    gradients, mirroring the original focal-loss behaviour of encouraging the
    network to focus on hard pixels. The denominator is detached to keep the
    normalisation statistic fixed during backprop.
    """

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
        norm = l1.detach().mean(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps)
        rel_error = l1 / norm
        # Implements w_i = min(max_weight, (|x_i - y_i| / norm) ** gamma).
        weight = rel_error.pow(self.gamma).clamp(max=self.max_weight)
        return (weight * l1).mean()


class HardnessWeightedL1Loss(nn.Module):
    """L1 loss reweighted by per-pixel hardness relative to mean error.

    Unlike :class:`FocalL1Loss`, both the numerator and denominator are detached
    so the hardness weights act as fixed, data-dependent importance weights
    rather than additional learnable factors. The exponent is named ``beta`` to
    distinguish this milder hardness weighting from the focal ``gamma``, though
    they now share the same default of 2.0 for easier comparison.
    """

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
        norm = l1.detach().mean(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps)
        rel_error = l1.detach() / norm
        weight = rel_error.pow(self.beta).clamp(max=self.max_weight)
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


class FocalSpatialSoftmaxLoss(nn.Module):
    """Focal L1 reconstruction loss augmented with spatial softmax alignment."""

    def __init__(self, spatial_weight: float = 0.1, channels: int = 3) -> None:
        super().__init__()
        if spatial_weight < 0:
            raise ValueError("spatial_weight must be non-negative.")
        self.focal = FocalL1Loss()
        self.spatial_weight = spatial_weight
        self.spatial = SpatialSoftmax(channels)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base = self.focal(input, target)
        if self.spatial_weight == 0.0:
            return base
        spatial_pred = self.spatial(input)
        spatial_target = self.spatial(target)
        spatial_loss = F.l1_loss(spatial_pred, spatial_target)
        return base + self.spatial_weight * spatial_loss


__all__ = [
    "FocalL1Loss",
    "HardnessWeightedL1Loss",
    "CauchyLoss",
    "MultiScalePatchLoss",
    "FocalSpatialSoftmaxLoss",
]
