#!/usr/bin/env python3
"""Backward-compatible import shim for multi-scale hardness losses."""
from __future__ import annotations

from jepa_world_model.loss_recon import (
    build_feature_pyramid,
    gaussian_blur_separable_2d,
    gaussian_kernel_1d,
    multi_scale_hardness_loss_box,
    multi_scale_hardness_loss_gaussian,
)

__all__ = [
    "gaussian_kernel_1d",
    "gaussian_blur_separable_2d",
    "build_feature_pyramid",
    "multi_scale_hardness_loss_gaussian",
    "multi_scale_hardness_loss_box",
]
