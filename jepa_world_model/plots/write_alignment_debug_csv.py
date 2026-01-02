"""Diagnostics alignment debug CSV helper."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, List

import torch

from jepa_world_model.actions import compress_actions_to_ids, decode_action_id


def write_alignment_debug_csv(
    frames: torch.Tensor,
    actions: torch.Tensor,
    paths: Optional[List[List[str]]],
    out_dir: Path,
    global_step: int,
) -> None:
    """Log per-frame checksums and action context to spot indexing issues."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bsz, seq_len = frames.shape[0], frames.shape[1]
    action_ids = compress_actions_to_ids(actions.cpu().numpy().reshape(-1, actions.shape[-1])).reshape(bsz, seq_len)
    csv_path = out_dir / f"alignment_debug_{global_step:07d}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "batch_index",
                "time_index",
                "frame_mean",
                "frame_std",
                "same_as_prev_frame",
                "action_to_this_id",
                "action_to_this_label",
                "action_from_this_id",
                "action_from_this_label",
                "frame_path",
            ]
        )
        for b in range(bsz):
            for t in range(seq_len):
                frame = frames[b, t]
                mean = float(frame.mean().item())
                std = float(frame.std().item())
                same_prev = False
                if t > 0:
                    same_prev = bool(torch.equal(frame, frames[b, t - 1]))
                action_to_this_id = action_ids[b, t - 1] if t > 0 else None
                action_from_this_id = action_ids[b, t] if t < seq_len - 1 else None
                frame_path = paths[b][t] if paths and b < len(paths) and t < len(paths[b]) else ""
                writer.writerow(
                    [
                        b,
                        t,
                        mean,
                        std,
                        int(same_prev),
                        "" if action_to_this_id is None else int(action_to_this_id),
                        "" if action_to_this_id is None else decode_action_id(int(action_to_this_id), actions.shape[-1]),
                        "" if action_from_this_id is None else int(action_from_this_id),
                        "" if action_from_this_id is None else decode_action_id(int(action_from_this_id), actions.shape[-1]),
                        frame_path,
                    ]
                )
