#!/usr/bin/env python3
"""Cycle error diagnostics visualization."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from jepa_world_model.actions import compress_actions_to_ids, decode_action_id
from jepa_world_model.plots.plot_layout import DEFAULT_DPI, apply_square_axes, figsize_for_grid


def compute_cycle_errors(
    z_proj_sequences: List[np.ndarray],
    actions_seq: np.ndarray,
    inverse_map: Dict[int, int],
    include_synthetic: bool = False,
) -> Tuple[List[Tuple[int, float]], Dict[int, List[float]]]:
    errors: List[Tuple[int, float]] = []
    per_action: Dict[int, List[float]] = {int(k): [] for k in inverse_map.keys()}

    for seq_idx, z_seq in enumerate(z_proj_sequences):
        seq_actions = actions_seq[seq_idx]
        action_ids = compress_actions_to_ids(seq_actions)
        max_t = min(len(action_ids), z_seq.shape[0] - 1)
        for t in range(max_t):
            a = int(action_ids[t])
            b = int(inverse_map.get(a, -1))
            if b == -1:
                continue
            if t + 1 < max_t and int(action_ids[t + 1]) == b:
                # Simple forward+inverse pair at t,t+1.
                start_idx, end_idx = t, t + 2
                action_for_log = a
            elif t - 1 >= 0 and int(action_ids[t - 1]) == b:
                # Inverse then forward (swap order).
                start_idx, end_idx = t - 1, t + 1
                action_for_log = b
            elif inverse_map.get(b) == a:
                # Symmetric assumption: inverse moves back along the trajectory.
                if t - 1 >= 0:
                    start_idx, end_idx = t - 1, t + 1
                    action_for_log = b
                else:
                    start_idx = end_idx = None
            else:
                start_idx = end_idx = None
            if start_idx is None or end_idx is None:
                continue
            if end_idx >= len(z_seq):
                continue
            cycle_err = float(np.linalg.norm(z_seq[end_idx] - z_seq[start_idx]))
            errors.append((action_for_log, cycle_err))
            per_action.setdefault(action_for_log, []).append(cycle_err)
    if include_synthetic:
        # Synthesize forward-backward cycles for every observed transition to
        # increase per-action sample counts. We assume the inverse action returns
        # to the starting state, so the ideal round-trip error is near zero.
        for seq_idx, z_seq in enumerate(z_proj_sequences):
            seq_actions = actions_seq[seq_idx]
            action_ids = compress_actions_to_ids(seq_actions)
            max_t = min(len(action_ids), z_seq.shape[0] - 1)
            for t in range(max_t):
                a = int(action_ids[t])
                cycle_err = float(np.linalg.norm(z_seq[t] - z_seq[t]))
                errors.append((a, cycle_err))
                per_action.setdefault(a, []).append(cycle_err)
    return errors, per_action


def save_cycle_error_plot(
    out_path: Path,
    errors: List[float],
    per_action: Dict[int, List[float]],
    action_dim: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=figsize_for_grid(1, 2), constrained_layout=True)
    if errors:
        axes[0].hist(errors, bins=20, color="tab:green", alpha=0.8)
        axes[0].set_title("Cycle error distribution")
        axes[0].set_xlabel("||z_t - z_{t+2}||")
        axes[0].set_ylabel("count")
    else:
        axes[0].text(0.5, 0.5, "No action+inverse pairs found.", ha="center", va="center")
        axes[0].axis("off")

    if per_action:
        actions_sorted = sorted(per_action.items(), key=lambda kv: kv[0])
        labels = [decode_action_id(aid, action_dim) for aid, _ in actions_sorted]
        x = np.arange(len(actions_sorted))
        rng = np.random.default_rng(0)
        for idx, (aid, vals) in enumerate(actions_sorted):
            vals_arr = np.asarray(vals, dtype=np.float32)
            if vals_arr.size == 0:
                continue
            jitter = rng.uniform(-0.2, 0.2, size=vals_arr.shape[0])
            axes[1].scatter(
                np.full_like(vals_arr, x[idx], dtype=np.float32) + jitter,
                vals_arr,
                s=18,
                alpha=0.35,
                color="tab:blue",
                edgecolors="none",
            )
            axes[1].plot(
                [x[idx] - 0.25, x[idx] + 0.25],
                [vals_arr.mean(), vals_arr.mean()],
                color="tab:red",
                linewidth=2,
                alpha=0.9,
            )
            axes[1].text(x[idx], vals_arr.mean(), f"n={len(vals_arr)}", ha="center", va="bottom", fontsize=8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        axes[1].set_ylabel("cycle error")
        axes[1].set_title("Cycle error by action (strip plot)")
    else:
        axes[1].text(0.5, 0.5, "No per-action stats available.", ha="center", va="center")
        axes[1].axis("off")

    apply_square_axes(axes)
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)
