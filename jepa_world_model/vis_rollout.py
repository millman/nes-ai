#!/usr/bin/env python3
"""Model-agnostic rollout visualization helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


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
    labels: List[str]
    actions: List[str] = field(default_factory=list)


@dataclass
class RolloutBatchItem:
    row_index: int
    start_idx: int
    ground_truth: torch.Tensor
    reconstructions: torch.Tensor
    rollout: List[Optional[torch.Tensor]]
    actions: List[str]


def render_rollout_batch(
    *,
    vis_frames: torch.Tensor,
    vis_actions: torch.Tensor,
    embeddings: torch.Tensor,
    decoded_frames: torch.Tensor,
    rows: int,
    rollout_steps: int,
    max_columns: Optional[int],
    selection: Optional[VisualizationSelection],
    log_deltas: bool,
    predictor: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    paired_actions: torch.Tensor,
    state_dim: int,
    action_text_fn: Callable[[torch.Tensor], str],
    rng: Optional[torch.Generator] = None,
) -> List[RolloutBatchItem]:
    if vis_frames.shape[0] == 0:
        raise ValueError("Visualization batch must include at least one sequence.")
    if vis_frames.shape[1] < 2:
        raise ValueError("Visualization batch must include at least two frames.")
    batch_size = vis_frames.shape[0]
    min_start = 0
    target_window = max(2, rollout_steps + 1)
    if max_columns is not None:
        target_window = max(target_window, max(2, max_columns))
    max_window = min(target_window, vis_frames.shape[1] - min_start)
    if max_window < 2:
        raise ValueError("Visualization window must be at least two frames wide.")
    max_start = max(min_start, vis_frames.shape[1] - max_window)
    if selection is not None and selection.row_indices.numel() > 0:
        num_rows = min(rows, selection.row_indices.numel())
        row_indices = selection.row_indices[:num_rows].to(device=vis_frames.device)
        base_starts = selection.time_indices[:num_rows].to(device=vis_frames.device)
    else:
        num_rows = min(rows, batch_size)
        row_indices = torch.randperm(batch_size, generator=rng, device=vis_frames.device)[:num_rows]
        base_starts = torch.randint(min_start, max_start + 1, (num_rows,), device=vis_frames.device, generator=rng)
    items: List[RolloutBatchItem] = []
    debug_lines: List[str] = []
    for row_offset, idx in enumerate(row_indices):
        start_idx = int(base_starts[row_offset].item()) if base_starts is not None else min_start
        start_idx = max(min_start, min(start_idx, max_start))
        gt_slice = vis_frames[idx, start_idx : start_idx + max_window]
        if gt_slice.shape[0] < max_window:
            continue
        action_texts: List[str] = []
        for offset in range(max_window):
            action_idx = min(start_idx + offset, vis_actions.shape[1] - 1)
            action_texts.append(action_text_fn(vis_actions[idx, action_idx]))
        recon_tensor = decoded_frames[idx, start_idx : start_idx + max_window].clamp(0, 1)
        rollout_frames: List[Optional[torch.Tensor]] = [None for _ in range(max_window)]
        current_embed = embeddings[idx, start_idx].unsqueeze(0)
        current_hidden = current_embed.new_zeros(1, state_dim)
        prev_pred_frame = decoded_frames[idx, start_idx].detach()
        current_frame = prev_pred_frame
        for step in range(1, max_window):
            action = paired_actions[idx, start_idx + step - 1].unsqueeze(0)
            prev_embed = current_embed
            next_embed, next_hidden = predictor(current_embed, current_hidden, action)
            decoded_next = decode_fn(next_embed)[0]
            current_frame = decoded_next.clamp(0, 1)
            rollout_frames[step] = current_frame.detach().cpu()
            if log_deltas and row_offset < 2:
                latent_norm = (next_embed - prev_embed).norm().item()
                pixel_delta = (current_frame - prev_pred_frame).abs().mean().item()
                frame_mse = F.mse_loss(current_frame, gt_slice[step]).item()
                debug_lines.append(
                    (
                        f"[viz] row={int(idx)} step={step} "
                        f"latent_norm={latent_norm:.4f} pixel_delta={pixel_delta:.4f} "
                        f"frame_mse={frame_mse:.4f}"
                    )
                )
            prev_pred_frame = current_frame.detach()
            current_embed = next_embed
            current_hidden = next_hidden
        items.append(
            RolloutBatchItem(
                row_index=int(idx.item()),
                start_idx=start_idx,
                ground_truth=gt_slice.detach().cpu(),
                reconstructions=recon_tensor.detach().cpu(),
                rollout=rollout_frames,
                actions=action_texts,
            )
        )
    if not items:
        raise AssertionError("Failed to build any visualization sequences.")
    if debug_lines:
        print("\n".join(debug_lines))
    return items


__all__ = [
    "RolloutBatchItem",
    "VisualizationSelection",
    "VisualizationSequence",
    "render_rollout_batch",
]
