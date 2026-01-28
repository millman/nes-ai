from __future__ import annotations

from pathlib import Path

import torch

from jepa_world_model.diagnostics_utils import should_use_z2h_init
from jepa_world_model.pose_rollout import rollout_pose
from jepa_world_model.rollout import rollout_teacher_forced
from jepa_world_model.plots.plot_neighborhood_stability import save_neighborhood_stability_plot
from jepa_world_model.plots.plot_smoothness_knn_distance_eigenvalue_spectrum import (
    save_smoothness_knn_distance_eigenvalue_spectrum_plot,
)
from jepa_world_model.plots.plot_two_step_composition_error import save_two_step_composition_error_plot
from jepa_world_model.vis_vis_ctrl_metrics import compute_vis_ctrl_metrics, write_vis_ctrl_metrics_csv


def compute_vis_ctrl_state(
    *,
    model,
    weights,
    device: torch.device,
    vis_ctrl_batch_cpu,
    force_h_zero: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    vis_frames = vis_ctrl_batch_cpu[0].to(device)
    vis_actions = vis_ctrl_batch_cpu[1].to(device)
    vis_embeddings = model.encode_sequence(vis_frames)["embeddings"]
    _, _, vis_h_states = rollout_teacher_forced(
        model,
        vis_embeddings,
        vis_actions,
        use_z2h_init=should_use_z2h_init(weights),
        force_h_zero=force_h_zero,
    )
    vis_p_embeddings = None
    if model.p_action_delta_projector is not None:
        _, vis_p_embeddings, _ = rollout_pose(
            model,
            vis_h_states,
            vis_actions,
            z_embeddings=vis_embeddings,
        )
    return vis_embeddings, vis_h_states, vis_actions, vis_p_embeddings


def run_vis_ctrl_for_kind(
    *,
    vis_ctrl_cfg,
    kind: str,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    warmup_frames: int,
    global_step: int,
    vis_ctrl_dir: Path,
) -> dict[str, float]:
    metrics = compute_vis_ctrl_metrics(
        embeddings,
        actions,
        vis_ctrl_cfg.knn_k_values,
        warmup_frames,
        vis_ctrl_cfg.min_action_count,
        vis_ctrl_cfg.stability_delta,
        vis_ctrl_cfg.knn_chunk_size,
    )
    save_smoothness_knn_distance_eigenvalue_spectrum_plot(
        vis_ctrl_dir / f"smoothness_{kind}_{global_step:07d}.png",
        metrics,
        kind,
    )
    save_two_step_composition_error_plot(
        vis_ctrl_dir / f"composition_error_{kind}_{global_step:07d}.png",
        metrics,
        kind,
    )
    save_neighborhood_stability_plot(
        vis_ctrl_dir / f"stability_{kind}_{global_step:07d}.png",
        metrics,
        kind,
    )
    return metrics


def write_vis_ctrl_summary(
    *,
    metrics_dir: Path,
    global_step: int,
    metrics_z: dict[str, float],
    metrics_p: dict[str, float] | None,
    metrics_h: dict[str, float],
) -> None:
    write_vis_ctrl_metrics_csv(
        metrics_dir / "vis_ctrl_metrics.csv",
        global_step,
        metrics_z,
        metrics_p,
        metrics_h,
    )
