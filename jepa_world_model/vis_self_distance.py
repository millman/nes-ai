#!/usr/bin/env python3
"""Self-distance diagnostics outputs."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch

from jepa_world_model.actions import compress_actions_to_ids, decode_action_id


def write_self_distance_csv(
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


def write_self_distance_plots(
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
    axes[0, 0].set_ylabel("||z0 - zt||")
    axes[0, 1].plot(steps, dist_prior_np, marker="o", color="tab:orange")
    axes[0, 1].set_title("Distance to prior")
    axes[0, 1].set_xlabel("timestep")
    axes[0, 1].set_ylabel("||z(t-1) - zt||")
    sc0 = axes[0, 2].scatter(dist_first_np, dist_prior_np, c=steps, cmap="viridis", s=20)
    axes[0, 2].set_title("Distance to first vs prior")
    axes[0, 2].set_xlabel("||z0 - zt||")
    axes[0, 2].set_ylabel("||z(t-1) - zt||")

    axes[1, 0].plot(steps, dist_first_cos_np, marker="o", color="tab:green")
    axes[1, 0].set_title("Cosine distance to first")
    axes[1, 0].set_xlabel("timestep")
    axes[1, 0].set_ylabel("1 - cos(z0, zt)")
    axes[1, 1].plot(steps, dist_prior_cos_np, marker="o", color="tab:red")
    axes[1, 1].set_title("Cosine distance to prior")
    axes[1, 1].set_xlabel("timestep")
    axes[1, 1].set_ylabel("1 - cos(z(t-1), zt)")
    sc1 = axes[1, 2].scatter(dist_first_cos_np, dist_prior_cos_np, c=steps, cmap="plasma", s=20)
    axes[1, 2].set_title("Cosine distance to first vs prior")
    axes[1, 2].set_xlabel("1 - cos(z0, zt)")
    axes[1, 2].set_ylabel("1 - cos(z(t-1), zt)")

    fig.suptitle(f"Self-distance: {trajectory_label}", fontsize=12)
    fig.colorbar(sc0, ax=axes[0, 2], label="timestep")
    fig.colorbar(sc1, ax=axes[1, 2], label="timestep")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = plot_dir / f"self_distance_{global_step:07d}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Save a cosine-only PNG for quick inspection.
    fig_cos, axes_cos = plt.subplots(1, 3, figsize=(14, 4.2))
    axes_cos[0].plot(steps, dist_first_cos_np, marker="o", color="tab:green")
    axes_cos[0].set_title("Cosine distance to first")
    axes_cos[0].set_xlabel("timestep")
    axes_cos[0].set_ylabel("1 - cos(z0, zt)")
    axes_cos[1].plot(steps, dist_prior_cos_np, marker="o", color="tab:red")
    axes_cos[1].set_title("Cosine distance to prior")
    axes_cos[1].set_xlabel("timestep")
    axes_cos[1].set_ylabel("1 - cos(z(t-1), zt)")
    sc_cos = axes_cos[2].scatter(dist_first_cos_np, dist_prior_cos_np, c=steps, cmap="plasma", s=20)
    axes_cos[2].set_title("Cosine distance to first vs prior")
    axes_cos[2].set_xlabel("1 - cos(z0, zt)")
    axes_cos[2].set_ylabel("1 - cos(z(t-1), zt)")
    fig_cos.suptitle(f"Self-distance (cosine): {trajectory_label}", fontsize=12)
    fig_cos.colorbar(sc_cos, ax=axes_cos[2], label="timestep")
    fig_cos.tight_layout(rect=[0, 0, 1, 0.93])
    out_path_cos = plot_dir / f"self_distance_cosine_{global_step:07d}.png"
    fig_cos.savefig(out_path_cos, dpi=200, bbox_inches="tight")
    plt.close(fig_cos)


def write_self_distance_outputs(
    model,
    traj_inputs,
    device: torch.device,
    csv_dir: Path,
    plot_dir: Path,
    global_step: int,
) -> None:
    if traj_inputs.frames.shape[1] < 2:
        return
    frames = traj_inputs.frames.to(device)
    with torch.no_grad():
        embeddings = model.encode_sequence(frames)["embeddings"][0]

    def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return 1 - cosine similarity with numeric safeguards."""
        denom = torch.norm(a, dim=-1) * torch.norm(b, dim=-1)
        denom = torch.clamp(denom, min=1e-8)
        cos = (a * b).sum(dim=-1) / denom
        return 1.0 - cos.clamp(-1.0, 1.0)

    dist_to_first = torch.norm(embeddings - embeddings[0:1], dim=-1)
    dist_to_first_cos = _cosine_distance(embeddings, embeddings[0:1])

    deltas = embeddings[1:] - embeddings[:-1]
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
            _cosine_distance(embeddings[1:], embeddings[:-1]),
        ],
        dim=0,
    )
    steps = list(range(dist_to_first.shape[0]))
    dist_first_np = dist_to_first.detach().cpu().numpy()
    dist_prior_np = dist_to_prior.detach().cpu().numpy()
    dist_first_cos_np = dist_to_first_cos.detach().cpu().numpy()
    dist_prior_cos_np = dist_to_prior_cos.detach().cpu().numpy()
    csv_path = csv_dir / f"self_distance_{global_step:06d}.csv"
    write_self_distance_csv(
        csv_path,
        traj_inputs,
        steps,
        dist_first_np,
        dist_prior_np,
        dist_first_cos_np,
        dist_prior_cos_np,
    )
    write_self_distance_plots(
        plot_dir,
        traj_inputs.trajectory_label,
        steps,
        dist_first_np,
        dist_prior_np,
        dist_first_cos_np,
        dist_prior_cos_np,
        global_step,
    )
