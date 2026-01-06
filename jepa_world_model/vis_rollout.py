#!/usr/bin/env python3
"""Model-agnostic rollout visualization helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import torch


@dataclass
class VisualizationSelection:
    row_indices: torch.Tensor
    time_indices: torch.Tensor


@dataclass
class VisualizationSequence:
    ground_truth: torch.Tensor
    rollout: List[Optional[torch.Tensor]]
    gradients: List[Optional[np.ndarray]]
    reconstructions: torch.Tensor
    reencoded: List[Optional[torch.Tensor]]
    labels: List[str]
    actions: List[str] = field(default_factory=list)


@dataclass
class RolloutBatchItem:
    row_index: int
    start_idx: int
    ground_truth: torch.Tensor
    reconstructions: torch.Tensor
    rollout: List[Optional[torch.Tensor]]
    reencoded: List[Optional[torch.Tensor]]
    actions: List[str]


def render_rollout_batch(
    *,
    vis_frames: torch.Tensor,
    decoded_frames: torch.Tensor,
    row_indices: torch.Tensor,
    start_indices: torch.Tensor,
    max_window: int,
    action_texts: Sequence[Sequence[str]],
    rollout_frames: Sequence[List[Optional[torch.Tensor]]],
    reencoded_frames: Sequence[List[Optional[torch.Tensor]]],
) -> List[RolloutBatchItem]:
    if vis_frames.shape[0] == 0:
        raise ValueError("Visualization batch must include at least one sequence.")
    if vis_frames.shape[1] < 2:
        raise ValueError("Visualization batch must include at least two frames.")
    if max_window < 2:
        raise ValueError("Visualization window must be at least two frames wide.")
    items: List[RolloutBatchItem] = []
    for row_offset, idx in enumerate(row_indices):
        start_idx = int(start_indices[row_offset].item())
        gt_slice = vis_frames[idx, start_idx : start_idx + max_window]
        if gt_slice.shape[0] < max_window:
            continue
        recon_tensor = decoded_frames[idx, start_idx : start_idx + max_window].clamp(0, 1)
        items.append(
            RolloutBatchItem(
                row_index=int(idx.item()),
                start_idx=start_idx,
                ground_truth=gt_slice.detach().cpu(),
                reconstructions=recon_tensor.detach().cpu(),
                rollout=list(rollout_frames[row_offset]),
                reencoded=list(reencoded_frames[row_offset]),
                actions=list(action_texts[row_offset]),
            )
        )
    if not items:
        raise AssertionError("Failed to build any visualization sequences.")
    return items


__all__ = [
    "RolloutBatchItem",
    "VisualizationSelection",
    "VisualizationSequence",
    "render_rollout_batch",
]
