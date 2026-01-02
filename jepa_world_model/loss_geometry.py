#!/usr/bin/env python3
"""Geometry ranking loss for g(h) embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass
class GeometryLossConfig:
    margin: float = 0.1
    max_pairs: int = 0


def geometry_ranking_loss(states: torch.Tensor, cfg: GeometryLossConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rank distances to the goal (last step) so later states are closer.

    Returns (loss, accuracy).
    """
    if states.ndim != 3:
        raise ValueError("states must have shape [B, T, D].")
    bsz, seq_len, _ = states.shape
    if seq_len < 3:
        zero = states.new_tensor(0.0)
        return zero, zero

    goal = states[:, -1]
    dist = torch.norm(states[:, :-1] - goal[:, None, :], dim=-1)
    steps = dist.shape[1]
    if steps < 2:
        zero = states.new_tensor(0.0)
        return zero, zero

    diff = dist[:, :, None] - dist[:, None, :]
    mask = torch.triu(torch.ones(steps, steps, device=states.device, dtype=torch.bool), diagonal=1)
    diffs = diff[:, mask]
    if diffs.numel() == 0:
        zero = states.new_tensor(0.0)
        return zero, zero

    if cfg.max_pairs and diffs.numel() > cfg.max_pairs * bsz:
        per_batch = min(cfg.max_pairs, diffs.shape[1])
        idx = torch.randperm(diffs.shape[1], device=states.device)[:per_batch]
        diffs = diffs[:, idx]

    loss = F.relu(cfg.margin - diffs).mean()
    accuracy = (diffs > 0).float().mean()
    return loss, accuracy
