#!/usr/bin/env python3
"""Reconstruction-oriented loss helpers for the JEPA world model."""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "HardnessWeightedL1Loss",
    "HardnessWeightedMSELoss",
    "HardnessWeightedMedianLoss",
    "FocalL1Loss",
    "patch_recon_loss",
    "gaussian_kernel_1d",
    "gaussian_blur_separable_2d",
    "box_kernel_2d",
    "build_feature_pyramid",
    "multi_scale_hardness_loss_gaussian",
    "multi_scale_hardness_loss_box",
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


def patch_recon_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    patch_sizes: Sequence[int],
    loss_fn: Optional[callable] = None,
) -> torch.Tensor:
    """
    Compute reconstruction loss over a grid of overlapping patches for multiple sizes.

    Rationale: keep supervision in image space without adding feature taps or extra
    forward passes—cheap to bolt on and works with the existing decoder output.
    A more traditional multi-scale hardness term could sample patches from intermediate
    CNN layers (feature pyramids, perceptual losses) with size-aware weights, but that
    would require exposing/retaining feature maps and increase memory/compute.
    """
    if not patch_sizes:
        raise ValueError("patch_recon_loss requires at least one patch size.")
    h, w = recon.shape[-2], recon.shape[-1]
    total = recon.new_tensor(0.0)
    count = 0
    if loss_fn is None:
        loss_fn = HardnessWeightedL1Loss()

    def _grid_indices(limit: int, size: int) -> Iterable[int]:
        step = max(1, size // 2)  # 50% overlap by default
        positions = list(range(0, limit - size + 1, step))
        if positions and positions[-1] != limit - size:
            positions.append(limit - size)
        elif not positions:
            positions = [0]
        return positions

    for patch_size in patch_sizes:
        if patch_size <= 0:
            raise ValueError("patch_recon_loss requires all patch sizes to be > 0.")
        if patch_size > h or patch_size > w:
            raise ValueError(f"patch_size={patch_size} exceeds recon dimensions {(h, w)}.")
        row_starts = _grid_indices(h, patch_size)
        col_starts = _grid_indices(w, patch_size)
        for rs in row_starts:
            for cs in col_starts:
                recon_patch = recon[..., rs : rs + patch_size, cs : cs + patch_size]
                target_patch = target[..., rs : rs + patch_size, cs : cs + patch_size]
                total = total + loss_fn(recon_patch, target_patch)
                count += 1
    return total / count if count > 0 else recon.new_tensor(0.0)


def gaussian_kernel_1d(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if size <= 0:
        raise ValueError("Gaussian kernel size must be positive.")
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    kernel = torch.exp(-0.5 * (coords / max(sigma, 1e-6)) ** 2)
    kernel = kernel / kernel.sum().clamp_min(1e-6)
    return kernel


def gaussian_blur_separable_2d(x: torch.Tensor, kernel_1d: torch.Tensor, stride: int = 1) -> torch.Tensor:
    """Apply separable Gaussian blur to a single-channel map."""
    if stride <= 0:
        raise ValueError("Gaussian blur stride must be positive.")
    k = kernel_1d.numel()
    pad = k // 2
    vert = F.conv2d(x, kernel_1d.view(1, 1, k, 1), padding=pad, stride=stride)
    horiz = F.conv2d(vert, kernel_1d.view(1, 1, 1, k), padding=pad, stride=stride)
    return horiz


def box_kernel_2d(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Normalized box filter over an NxN window."""
    if size <= 0:
        raise ValueError("Box kernel size must be positive.")
    kernel = torch.ones((1, 1, size, size), device=device, dtype=dtype)
    return kernel / kernel.sum().clamp_min(1e-6)


def build_feature_pyramid(
    pred: torch.Tensor, target: torch.Tensor, num_scales: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    preds = [pred]
    targets = [target]
    for _ in range(1, num_scales):
        pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
        target = F.avg_pool2d(target, kernel_size=2, stride=2)
        preds.append(pred)
        targets.append(target)
    return preds, targets


def multi_scale_hardness_loss_gaussian(
    preds: List[torch.Tensor],
    targets: List[torch.Tensor],
    kernel_sizes: Sequence[int],
    sigmas: Sequence[float],
    betas: Sequence[float],
    lambdas: Sequence[float],
    strides: Sequence[int],
    max_weight: float = 100.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute hardness-weighted loss over multiple scales.

    • kernel_sizes: spatial support per scale (similar to patch size).
    • sigmas: Gaussian blur stddev per scale (controls how far hardness spreads).
    • betas: hardness exponents; higher beta emphasizes harder regions.
    • lambdas: per-scale weights to balance contributions.
    • strides: blur stride per scale (reduces compute; resampled back to original size if needed).
    """
    if max_weight <= 0:
        raise ValueError("max_weight must be positive.")
    if not (
        len(preds)
        == len(targets)
        == len(kernel_sizes)
        == len(sigmas)
        == len(betas)
        == len(lambdas)
        == len(strides)
    ):
        raise ValueError("preds, targets, kernel_sizes, sigmas, betas, lambdas, and strides must share length.")
    if not preds:
        return torch.tensor(0.0, device="cpu")
    total = preds[0].new_tensor(0.0)
    for idx, (p, t) in enumerate(zip(preds, targets)):
        if p.shape != t.shape:
            raise ValueError(f"Pred/target shape mismatch at scale {idx}: {p.shape} vs {t.shape}")
        k = int(kernel_sizes[idx])
        sigma = float(sigmas[idx])
        beta = float(betas[idx])
        lam = float(lambdas[idx])
        stride = int(strides[idx])
        if stride <= 0:
            raise ValueError("Gaussian hardness stride must be positive.")
        per_pixel = ((p - t) ** 2).mean(dim=1, keepdim=True)
        per_pixel_detached = per_pixel.detach()
        g1d = gaussian_kernel_1d(k, sigma, device=per_pixel.device, dtype=per_pixel.dtype)
        blurred_weight = gaussian_blur_separable_2d(per_pixel_detached, g1d, stride=stride)
        if blurred_weight.shape[-2:] != per_pixel.shape[-2:]:
            blurred_weight = F.interpolate(blurred_weight, size=per_pixel.shape[-2:], mode="nearest")
        weight = (blurred_weight + eps).pow(beta).clamp(max=max_weight)
        scale_loss = (weight * (per_pixel + eps)).mean()
        total = total + lam * scale_loss
    return total


def multi_scale_hardness_loss_box(
    preds: List[torch.Tensor],
    targets: List[torch.Tensor],
    kernel_sizes: Sequence[int],
    betas: Sequence[float],
    lambdas: Sequence[float],
    strides: Sequence[int],
    max_weight: float = 100.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute hardness-weighted loss over multiple scales with a convolutional approximation of patch-style hardness.
    """
    if max_weight <= 0:
        raise ValueError("max_weight must be positive.")
    if not (
        len(preds)
        == len(targets)
        == len(kernel_sizes)
        == len(betas)
        == len(lambdas)
        == len(strides)
    ):
        raise ValueError("preds, targets, kernel_sizes, betas, lambdas, and strides must share length.")
    if not preds:
        return torch.tensor(0.0, device="cpu")
    total = preds[0].new_tensor(0.0)
    for idx, (p, t) in enumerate(zip(preds, targets)):
        if p.shape != t.shape:
            raise ValueError(f"Pred/target shape mismatch at scale {idx}: {p.shape} vs {t.shape}")
        k = int(kernel_sizes[idx])
        beta = float(betas[idx])
        lam = float(lambdas[idx])
        stride = int(strides[idx])
        if stride <= 0:
            raise ValueError("Box hardness stride must be positive.")
        _, _, h, w = p.shape
        k_eff = min(k, h, w)
        # Clamp kernel/stride to valid spatial support so deeper pyramid levels still contribute.
        stride_eff = max(1, min(stride, k_eff))
        per_pixel_l1 = (p - t).abs().mean(dim=1, keepdim=True)  # Bx1xHxW
        per_pixel_detached = per_pixel_l1.detach()
        # Valid pooling (no padding) to avoid border bleed; stride can mimic patch overlap (e.g., k//2).
        norm = F.avg_pool2d(per_pixel_detached, kernel_size=k_eff, stride=stride_eff, padding=0)
        coverage = F.avg_pool2d(torch.ones_like(per_pixel_detached), kernel_size=k_eff, stride=stride_eff, padding=0)
        # Upsample pooled maps back to full resolution for per-pixel weighting.
        if norm.shape[-2:] != per_pixel_l1.shape[-2:]:
            norm = F.interpolate(norm, size=per_pixel_l1.shape[-2:], mode="bilinear", align_corners=False)
            coverage = F.interpolate(coverage, size=per_pixel_l1.shape[-2:], mode="bilinear", align_corners=False)
        norm = norm.clamp_min(eps)
        coverage = coverage.clamp_min(eps)
        weight = (per_pixel_detached / norm).pow(beta).clamp(max=max_weight)
        weighted_sum = (weight * per_pixel_l1).sum()
        scale_loss = weighted_sum / coverage.sum()
        total = total + lam * scale_loss
    return total
