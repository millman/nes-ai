#!/usr/bin/env python3
"""Self-distance diagnostics outputs."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import torch

from jepa_world_model.actions import compress_actions_to_ids, decode_action_id
from jepa_world_model.plots.plot_layout import DEFAULT_DPI, figsize_for_grid


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return 1 - cosine similarity with numeric safeguards."""
    denom = torch.norm(a, dim=-1) * torch.norm(b, dim=-1)
    denom = torch.clamp(denom, min=1e-8)
    cos = (a * b).sum(dim=-1) / denom
    return 1.0 - cos.clamp(-1.0, 1.0)


def _compute_self_distance_arrays(
    embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return dist_to_first, dist_to_prior, dist_to_first_cos, dist_to_prior_cos


def write_self_distance_csv(
    csv_path: Path,
    traj_inputs,
    steps: List[int],
    dist_first_np,
    dist_prior_np,
    dist_first_cos_np,
    dist_prior_cos_np,
    frame_paths: Optional[List[Path]] = None,
    frame_labels: Optional[List[str]] = None,
    actions: Optional[torch.Tensor] = None,
    action_labels: Optional[List[str]] = None,
    action_dim: Optional[int] = None,
    trajectory_label: Optional[str] = None,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame_paths = frame_paths if frame_paths is not None else traj_inputs.frame_paths
    frame_labels = frame_labels if frame_labels is not None else traj_inputs.frame_labels
    actions = actions if actions is not None else traj_inputs.actions
    action_labels = action_labels if action_labels is not None else traj_inputs.action_labels
    action_dim = action_dim if action_dim is not None else traj_inputs.action_dim
    trajectory_label = trajectory_label if trajectory_label is not None else traj_inputs.trajectory_label
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
    if not frame_paths:
        return
    first_frame = frame_paths[0].as_posix()
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, step in enumerate(steps):
            prior_idx = max(0, idx - 1)
            action_id = int(compress_actions_to_ids(actions[idx : idx + 1])[0])
            action_label = (
                action_labels[idx]
                if idx < len(action_labels)
                else decode_action_id(action_id, action_dim)
            )
            writer.writerow(
                {
                    "trajectory": trajectory_label,
                    "trajectory_label": trajectory_label,
                    "timestep": step,
                    "distance_to_first": float(dist_first_np[idx]),
                    "distance_to_prior": float(dist_prior_np[idx]),
                    "cosine_distance_to_first": float(dist_first_cos_np[idx]),
                    "cosine_distance_to_prior": float(dist_prior_cos_np[idx]),
                    "frame_path": frame_paths[idx].as_posix(),
                    "first_frame_path": first_frame,
                    "prior_frame_path": frame_paths[prior_idx].as_posix(),
                    "frame_label": frame_labels[idx]
                    if idx < len(frame_labels)
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
    embedding_label: str,
    title_prefix: str,
    file_prefix: str,
    cosine_prefix: Optional[str],
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=figsize_for_grid(2, 3))
    axes[0, 0].plot(steps, dist_first_np, marker="o")
    axes[0, 0].set_title("Distance to first")
    axes[0, 0].set_xlabel("timestep")
    axes[0, 0].set_ylabel(f"||{embedding_label}0 - {embedding_label}t||")
    axes[0, 1].plot(steps, dist_prior_np, marker="o", color="tab:orange")
    axes[0, 1].set_title("Distance to prior")
    axes[0, 1].set_xlabel("timestep")
    axes[0, 1].set_ylabel(f"||{embedding_label}(t-1) - {embedding_label}t||")
    sc0 = axes[0, 2].scatter(dist_first_np, dist_prior_np, c=steps, cmap="viridis", s=20)
    axes[0, 2].set_title("Distance to first vs prior")
    axes[0, 2].set_xlabel(f"||{embedding_label}0 - {embedding_label}t||")
    axes[0, 2].set_ylabel(f"||{embedding_label}(t-1) - {embedding_label}t||")

    axes[1, 0].plot(steps, dist_first_cos_np, marker="o", color="tab:green")
    axes[1, 0].set_title("Cosine distance to first")
    axes[1, 0].set_xlabel("timestep")
    axes[1, 0].set_ylabel(f"1 - cos({embedding_label}0, {embedding_label}t)")
    axes[1, 1].plot(steps, dist_prior_cos_np, marker="o", color="tab:red")
    axes[1, 1].set_title("Cosine distance to prior")
    axes[1, 1].set_xlabel("timestep")
    axes[1, 1].set_ylabel(f"1 - cos({embedding_label}(t-1), {embedding_label}t)")
    sc1 = axes[1, 2].scatter(dist_first_cos_np, dist_prior_cos_np, c=steps, cmap="plasma", s=20)
    axes[1, 2].set_title("Cosine distance to first vs prior")
    axes[1, 2].set_xlabel(f"1 - cos({embedding_label}0, {embedding_label}t)")
    axes[1, 2].set_ylabel(f"1 - cos({embedding_label}(t-1), {embedding_label}t)")

    fig.suptitle(f"{title_prefix}: {trajectory_label}", fontsize=12)
    fig.colorbar(sc0, ax=axes[0, 2], label="timestep")
    fig.colorbar(sc1, ax=axes[1, 2], label="timestep")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = plot_dir / f"{file_prefix}_{global_step:07d}.png"
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)

    # Save a cosine-only PNG for quick inspection.
    fig_cos, axes_cos = plt.subplots(1, 3, figsize=figsize_for_grid(1, 3))
    axes_cos[0].plot(steps, dist_first_cos_np, marker="o", color="tab:green")
    axes_cos[0].set_title("Cosine distance to first")
    axes_cos[0].set_xlabel("timestep")
    axes_cos[0].set_ylabel(f"1 - cos({embedding_label}0, {embedding_label}t)")
    axes_cos[1].plot(steps, dist_prior_cos_np, marker="o", color="tab:red")
    axes_cos[1].set_title("Cosine distance to prior")
    axes_cos[1].set_xlabel("timestep")
    axes_cos[1].set_ylabel(f"1 - cos({embedding_label}(t-1), {embedding_label}t)")
    sc_cos = axes_cos[2].scatter(dist_first_cos_np, dist_prior_cos_np, c=steps, cmap="plasma", s=20)
    axes_cos[2].set_title("Cosine distance to first vs prior")
    axes_cos[2].set_xlabel(f"1 - cos({embedding_label}0, {embedding_label}t)")
    axes_cos[2].set_ylabel(f"1 - cos({embedding_label}(t-1), {embedding_label}t)")
    fig_cos.suptitle(f"{title_prefix} (cosine): {trajectory_label}", fontsize=12)
    fig_cos.colorbar(sc_cos, ax=axes_cos[2], label="timestep")
    fig_cos.tight_layout(rect=[0, 0, 1, 0.93])
    cosine_prefix = cosine_prefix or f"{file_prefix}_cosine"
    out_path_cos = plot_dir / f"{cosine_prefix}_{global_step:07d}.png"
    fig_cos.savefig(out_path_cos, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig_cos)


def write_self_distance_outputs_from_embeddings(
    embeddings: torch.Tensor,
    traj_inputs,
    csv_dir: Path,
    plot_dir: Path,
    global_step: int,
    embedding_label: str,
    title_prefix: str,
    file_prefix: str,
    cosine_prefix: Optional[str],
    start_index: int = 0,
) -> None:
    if embeddings.shape[0] < 2:
        return
    dist_to_first, dist_to_prior, dist_to_first_cos, dist_to_prior_cos = _compute_self_distance_arrays(embeddings)
    steps = list(range(start_index, start_index + dist_to_first.shape[0]))
    dist_first_np = dist_to_first.detach().cpu().numpy()
    dist_prior_np = dist_to_prior.detach().cpu().numpy()
    dist_first_cos_np = dist_to_first_cos.detach().cpu().numpy()
    dist_prior_cos_np = dist_to_prior_cos.detach().cpu().numpy()
    csv_path = csv_dir / f"{file_prefix}_{global_step:06d}.csv"
    frame_paths = traj_inputs.frame_paths[start_index:]
    frame_labels = traj_inputs.frame_labels[start_index:]
    actions = traj_inputs.actions[start_index:]
    action_labels = traj_inputs.action_labels[start_index:]
    write_self_distance_csv(
        csv_path,
        traj_inputs,
        steps,
        dist_first_np,
        dist_prior_np,
        dist_first_cos_np,
        dist_prior_cos_np,
        frame_paths=frame_paths,
        frame_labels=frame_labels,
        actions=actions,
        action_labels=action_labels,
        action_dim=traj_inputs.action_dim,
        trajectory_label=traj_inputs.trajectory_label,
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
        embedding_label=embedding_label,
        title_prefix=title_prefix,
        file_prefix=file_prefix,
        cosine_prefix=cosine_prefix,
    )


def write_self_distance_outputs(
    embeddings: torch.Tensor,
    traj_inputs,
    csv_dir: Path,
    plot_dir: Path,
    global_step: int,
    embedding_label: str,
    title_prefix: str,
    file_prefix: str,
    cosine_prefix: Optional[str],
) -> None:
    if embeddings.ndim == 3:
        embeddings = embeddings[0]
    if embeddings.shape[0] < 2:
        return
    write_self_distance_outputs_from_embeddings(
        embeddings,
        traj_inputs,
        csv_dir,
        plot_dir,
        global_step,
        embedding_label=embedding_label,
        title_prefix=title_prefix,
        file_prefix=file_prefix,
        cosine_prefix=cosine_prefix,
        start_index=0,
    )
