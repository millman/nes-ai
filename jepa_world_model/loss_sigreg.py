#!/usr/bin/env python3
"""SIGReg loss helpers."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sigreg_loss(
    embeddings: torch.Tensor,
    num_projections: int,
) -> torch.Tensor:
    b, t, d = embeddings.shape
    flat = embeddings.reshape(b * t, d)
    device = embeddings.device
    directions = torch.randn(num_projections, d, device=device)
    directions = F.normalize(directions, dim=-1)
    projected = flat @ directions.t()
    projected = projected.t()
    sorted_proj, _ = torch.sort(projected, dim=1)
    normal_samples = torch.randn_like(projected)
    sorted_normal, _ = torch.sort(normal_samples, dim=1)
    return F.mse_loss(sorted_proj, sorted_normal)


__all__ = ["sigreg_loss"]
