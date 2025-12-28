#!/usr/bin/env python3
"""Multi-scale hardness losses and utilities."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F


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
