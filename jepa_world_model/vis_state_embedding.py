#!/usr/bin/env python3
"""State embedding diagnostics outputs."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from jepa_world_model.actions import compress_actions_to_ids, decode_action_id


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    denom = torch.norm(a, dim=-1) * torch.norm(b, dim=-1)
    denom = torch.clamp(denom, min=1e-8)
    cos = (a * b).sum(dim=-1) / denom
    return 1.0 - cos.clamp(-1.0, 1.0)


def write_state_embedding_csv(
    csv_path: Path,
    traj_inputs,
    steps: List[int],
    dist_first_np,
    dist_prior_np,
    dist_first_cos_np,
    dist_prior_cos_np,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trajectory",
        "trajectory_label",
        "timestep",
        "distance_to_first",
        "distance_to_prior",
        "cosine_distance_to_first",
        "cosine_distance_to_prior",
        "frame_path",
        "first_frame_path",
        "prior_frame_path",
        "frame_label",
        "action_id",
        "action_label",
    ]
    first_frame = traj_inputs.frame_paths[0].as_posix()
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, step in enumerate(steps):
            prior_idx = max(0, idx - 1)
            action_id = int(compress_actions_to_ids(traj_inputs.actions[idx : idx + 1])[0])
            action_label = (
                traj_inputs.action_labels[idx]
                if idx < len(traj_inputs.action_labels)
                else decode_action_id(action_id, traj_inputs.action_dim)
            )
            writer.writerow(
                {
                    "trajectory": traj_inputs.trajectory_label,
                    "trajectory_label": traj_inputs.trajectory_label,
                    "timestep": step,
                    "distance_to_first": float(dist_first_np[idx]),
                    "distance_to_prior": float(dist_prior_np[idx]),
                    "cosine_distance_to_first": float(dist_first_cos_np[idx]),
                    "cosine_distance_to_prior": float(dist_prior_cos_np[idx]),
                    "frame_path": traj_inputs.frame_paths[idx].as_posix(),
                    "first_frame_path": first_frame,
                    "prior_frame_path": traj_inputs.frame_paths[prior_idx].as_posix(),
                    "frame_label": traj_inputs.frame_labels[idx]
                    if idx < len(traj_inputs.frame_labels)
                    else f"t={step}",
                    "action_id": action_id,
                    "action_label": action_label,
                }
            )


def write_state_embedding_plots(
    plot_dir: Path,
    trajectory_label: str,
    steps: List[int],
    dist_first_np,
    dist_prior_np,
    dist_first_cos_np,
    dist_prior_cos_np,
    global_step: int,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    axes[0, 0].plot(steps, dist_first_np, marker="o")
    axes[0, 0].set_title("Distance to first")
    axes[0, 0].set_xlabel("timestep")
    axes[0, 0].set_ylabel("||s0 - s_t||")
    axes[0, 1].plot(steps, dist_prior_np, marker="o", color="tab:orange")
    axes[0, 1].set_title("Distance to prior")
    axes[0, 1].set_xlabel("timestep")
    axes[0, 1].set_ylabel("||s(t-1) - s_t||")
    sc0 = axes[0, 2].scatter(dist_first_np, dist_prior_np, c=steps, cmap="viridis", s=20)
    axes[0, 2].set_title("Distance to first vs prior")
    axes[0, 2].set_xlabel("||s0 - s_t||")
    axes[0, 2].set_ylabel("||s(t-1) - s_t||")

    axes[1, 0].plot(steps, dist_first_cos_np, marker="o", color="tab:green")
    axes[1, 0].set_title("Cosine distance to first")
    axes[1, 0].set_xlabel("timestep")
    axes[1, 0].set_ylabel("1 - cos(s0, s_t)")
    axes[1, 1].plot(steps, dist_prior_cos_np, marker="o", color="tab:red")
    axes[1, 1].set_title("Cosine distance to prior")
    axes[1, 1].set_xlabel("timestep")
    axes[1, 1].set_ylabel("1 - cos(s(t-1), s_t)")
    sc1 = axes[1, 2].scatter(dist_first_cos_np, dist_prior_cos_np, c=steps, cmap="plasma", s=20)
    axes[1, 2].set_title("Cosine distance to first vs prior")
    axes[1, 2].set_xlabel("1 - cos(s0, s_t)")
    axes[1, 2].set_ylabel("1 - cos(s(t-1), s_t)")

    fig.suptitle(f"State embedding: {trajectory_label}", fontsize=12)
    fig.colorbar(sc0, ax=axes[0, 2], label="timestep")
    fig.colorbar(sc1, ax=axes[1, 2], label="timestep")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = plot_dir / f"state_embedding_{global_step:07d}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_state_embedding_histogram(
    plot_dir: Path,
    s_norm_np,
    global_step: int,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_hist, ax = plt.subplots(figsize=(7, 4))
    if s_norm_np.size:
        finite = np.asarray(s_norm_np)[np.isfinite(s_norm_np)]
        if finite.size:
            min_val = float(finite.min())
            max_val = float(finite.max())
            if finite.size < 2 or np.isclose(min_val, max_val):
                ax.hist(finite, bins=1, color="tab:purple", alpha=0.8)
            else:
                ax.hist(finite, bins=40, color="tab:purple", alpha=0.8)
        else:
            ax.text(0.5, 0.5, "No finite state norms.", ha="center", va="center")
            ax.axis("off")
        ax.set_xlabel("||s||")
        ax.set_ylabel("count")
    else:
        ax.text(0.5, 0.5, "No state embeddings.", ha="center", va="center")
        ax.axis("off")
    ax.set_title("State embedding norm distribution")
    fig_hist.tight_layout()
    fig_hist.savefig(plot_dir / f"state_embedding_hist_{global_step:07d}.png", dpi=200, bbox_inches="tight")
    plt.close(fig_hist)


def write_state_embedding_outputs(
    model,
    traj_inputs,
    device: torch.device,
    csv_dir: Path,
    plot_dir: Path,
    global_step: int,
    hist_frames_cpu: Optional[torch.Tensor] = None,
) -> None:
    if traj_inputs.frames.shape[1] < 2:
        return
    frames = traj_inputs.frames.to(device)
    with torch.no_grad():
        embeddings = model.encode_sequence(frames)["embeddings"][0]
        h = model.z_to_h(embeddings)
        s = model.state_head(h)

    dist_to_first = torch.norm(s - s[0:1], dim=-1)
    dist_to_first_cos = _cosine_distance(s, s[0:1])

    deltas = s[1:] - s[:-1]
    dist_to_prior = torch.cat(
        [
            dist_to_first.new_zeros(1),
            torch.norm(deltas, dim=-1),
        ],
        dim=0,
    )
    dist_to_prior_cos = torch.cat(
        [
            dist_to_first_cos.new_zeros(1),
            _cosine_distance(s[1:], s[:-1]),
        ],
        dim=0,
    )
    steps = list(range(dist_to_first.shape[0]))
    dist_first_np = dist_to_first.detach().cpu().numpy()
    dist_prior_np = dist_to_prior.detach().cpu().numpy()
    dist_first_cos_np = dist_to_first_cos.detach().cpu().numpy()
    dist_prior_cos_np = dist_to_prior_cos.detach().cpu().numpy()
    csv_path = csv_dir / f"state_embedding_{global_step:06d}.csv"
    write_state_embedding_csv(
        csv_path,
        traj_inputs,
        steps,
        dist_first_np,
        dist_prior_np,
        dist_first_cos_np,
        dist_prior_cos_np,
    )
    write_state_embedding_plots(
        plot_dir,
        traj_inputs.trajectory_label,
        steps,
        dist_first_np,
        dist_prior_np,
        dist_first_cos_np,
        dist_prior_cos_np,
        global_step,
    )

    hist_frames = hist_frames_cpu if hist_frames_cpu is not None else traj_inputs.frames
    if hist_frames.ndim != 5 or hist_frames.shape[1] < 1:
        return
    hist_frames = hist_frames.to(device)
    with torch.no_grad():
        hist_embeddings = model.encode_sequence(hist_frames)["embeddings"]
        hist_h = model.z_to_h(hist_embeddings)
        hist_s = model.state_head(hist_h)
    hist_flat = hist_s.reshape(-1, hist_s.shape[-1])
    hist_norm = torch.norm(hist_flat, dim=-1).detach().cpu().numpy()
    write_state_embedding_histogram(plot_dir, hist_norm, global_step)
