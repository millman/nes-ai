#!/usr/bin/env python3
"""Patch-based reconstruction loss helpers."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch

from jepa_world_model.loss import HardnessWeightedL1Loss


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


__all__ = ["patch_recon_loss"]
