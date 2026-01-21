from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from jepa_world_model.actions import compress_actions_to_ids
from jepa_world_model.diagnostics_alignment import (
    write_alignment_artifacts,
    write_cycle_error_artifacts,
    write_motion_pca_artifacts,
)
from jepa_world_model.diagnostics_consistency import (
    compute_path_independence_diffs,
    compute_z_monotonicity_distances,
)
from jepa_world_model.diagnostics_graph import prepare_graph_diagnostics
from jepa_world_model.diagnostics_prepare import prepare_diagnostics_batch_state
from jepa_world_model.diagnostics_rollout_divergence import (
    compute_h_ablation_divergence,
    compute_rollout_divergence_metrics,
)
from jepa_world_model.diagnostics_rollout_visuals import (
    build_visualization_sequences,
    compute_off_manifold_errors,
)
from jepa_world_model.diagnostics_utils import append_csv_row, compute_norm_stats, write_step_csv
from jepa_world_model.diagnostics_vis_ctrl import compute_vis_ctrl_state, write_vis_ctrl_summary
from jepa_world_model.plots.plot_action_vector_field import (
    save_action_time_slice_plot,
    save_action_vector_field_plot,
)
from jepa_world_model.plots.plot_diagnostics_extra import (
    StraightLineTrajectory,
    save_ablation_divergence_plot,
    save_drift_by_action_plot,
    save_monotonicity_plot,
    save_norm_timeseries_plot,
    save_path_independence_plot,
    save_rollout_divergence_plot,
    save_straightline_plot,
    save_z_consistency_plot,
)
from jepa_world_model.plots.plot_diagnostics_frames import save_diagnostics_frames
from jepa_world_model.plots.plot_edge_consistency_hist import save_edge_consistency_hist_plot
from jepa_world_model.plots.plot_graph_diagnostics import (
    compute_graph_diagnostics_stats,
    update_graph_diagnostics_history,
)
from jepa_world_model.plots.plot_in_degree_hist import save_in_degree_hist_plot
from jepa_world_model.plots.plot_neff_violin import save_neff_violin_plot
from jepa_world_model.plots.plot_neighborhood_stability import save_neighborhood_stability_plot
from jepa_world_model.plots.plot_off_manifold_error import save_off_manifold_visualization
from jepa_world_model.plots.plot_rank_cdf import save_rank_cdf_plot
from jepa_world_model.plots.plot_smoothness_knn_distance_eigenvalue_spectrum import (
    save_smoothness_knn_distance_eigenvalue_spectrum_plot,
)
from jepa_world_model.plots.plot_two_step_composition_error import save_two_step_composition_error_plot
from jepa_world_model.plots.write_alignment_debug_csv import write_alignment_debug_csv
from jepa_world_model.pose_rollout import rollout_pose
from jepa_world_model.rollout import rollout_teacher_forced
from jepa_world_model.vis_composability import save_composability_plot
from jepa_world_model.vis_embedding_projection import save_embedding_projection
from jepa_world_model.vis_hard_samples import save_hard_example_grid
from jepa_world_model.vis_rollout_batch import save_rollout_sequence_batch
from jepa_world_model.vis_self_distance import write_self_distance_outputs
from jepa_world_model.vis_state_embedding import write_state_embedding_outputs
from jepa_world_model.vis_vis_ctrl_metrics import compute_vis_ctrl_metrics


def _enabled_kinds(weights, model) -> dict[str, bool]:
    z_enabled = any(
        getattr(weights, name, 0.0) > 0
        for name in (
            "jepa",
            "inverse_dynamics_z",
            "action_delta_z",
            "rollout_kstep_z",
            "rollout_recon_z",
            "rollout_recon_multi_box_z",
            "rollout_recon_delta_z",
            "rollout_recon_multi_box_delta_z",
            "rollout_project_z",
        )
    )
    h_enabled = any(
        getattr(weights, name, 0.0) > 0
        for name in (
            "h2z",
            "z2h",
            "inverse_dynamics_h",
            "action_delta_h",
            "additivity_h",
            "rollout_kstep_h",
            "rollout_kstep_delta_h",
        )
    )
    p_enabled = (
        model.p_action_delta_projector is not None
        and any(
            getattr(weights, name, 0.0) > 0
            for name in (
                "inverse_dynamics_p",
                "action_delta_p",
                "additivity_p",
                "rollout_kstep_p",
                "scale_p",
                "geometry_rank_p",
            )
        )
    )
    return {"z": z_enabled, "h": h_enabled, "p": p_enabled}


def run_diagnostics_step(
    *,
    diagnostics_cfg: Any,
    vis_cfg: Any,
    hard_example_cfg: Any,
    vis_ctrl_cfg: Any,
    graph_cfg: Any,
    model: Any,
    ema_model: Optional[Any],
    decoder: Any,
    device: torch.device,
    weights: Any,
    global_step: int,
    fixed_batch_cpu: Tuple[torch.Tensor, torch.Tensor, List[List[str]]],
    fixed_selection: Optional[Any],
    rolling_batch_cpu: Tuple[torch.Tensor, torch.Tensor, List[List[str]]],
    off_manifold_batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor, List[List[str]]]],
    off_manifold_steps: int,
    embedding_batch_cpu: Tuple[torch.Tensor, torch.Tensor, List[List[str]]],
    diagnostics_batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor, List[List[str]]]],
    vis_ctrl_batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor, List[List[str]]]],
    graph_diag_batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor, List[List[str]]]],
    hard_reservoir: Optional[Any],
    hard_reservoir_val: Optional[Any],
    dataset: Any,
    self_distance_inputs: Any,
    diagnostics_generator: torch.Generator,
    vis_selection_generator: torch.Generator,
    run_dir: Path,
) -> None:
    if diagnostics_cfg.enabled and diagnostics_batch_cpu is None:
        raise AssertionError("Diagnostics requested but diagnostics_batch_cpu is missing.")
    if vis_ctrl_cfg.enabled and vis_ctrl_batch_cpu is None:
        raise AssertionError("Vis-ctrl requested but vis_ctrl_batch_cpu is missing.")
    if graph_cfg.enabled and graph_diag_batch_cpu is None:
        raise AssertionError("Graph diagnostics requested but graph_diag_batch_cpu is missing.")

    run_dir = Path(run_dir)
    metrics_dir = run_dir / "metrics"
    fixed_vis_dir = run_dir / "vis_fixed"
    rolling_vis_dir = run_dir / "vis_rolling"
    pca_z_dir = run_dir / "pca_z"
    pca_p_dir = run_dir / "pca_p"
    pca_h_dir = run_dir / "pca_h"
    vis_off_manifold_dir = run_dir / "vis_off_manifold"
    samples_hard_dir = run_dir / "samples_hard"
    samples_hard_val_dir = run_dir / "samples_hard_val"
    vis_self_distance_z_dir = run_dir / "vis_self_distance_z"
    vis_self_distance_p_dir = run_dir / "vis_self_distance_p"
    vis_self_distance_h_dir = run_dir / "vis_self_distance_h"
    vis_state_embedding_dir = run_dir / "vis_state_embedding"
    vis_odometry_dir = run_dir / "vis_odometry"
    self_distance_z_dir = run_dir / "self_distance_z"
    self_distance_p_dir = run_dir / "self_distance_p"
    self_distance_h_dir = run_dir / "self_distance_h"
    diagnostics_delta_dir = run_dir / "vis_delta_z_pca"
    diagnostics_delta_p_dir = run_dir / "vis_delta_p_pca"
    diagnostics_delta_h_dir = run_dir / "vis_delta_h_pca"
    diagnostics_alignment_dir = run_dir / "vis_action_alignment_z"
    diagnostics_alignment_raw_dir = run_dir / "vis_action_alignment_z_raw"
    diagnostics_alignment_centered_dir = run_dir / "vis_action_alignment_z_centered"
    diagnostics_alignment_p_dir = run_dir / "vis_action_alignment_p"
    diagnostics_alignment_p_raw_dir = run_dir / "vis_action_alignment_p_raw"
    diagnostics_alignment_p_centered_dir = run_dir / "vis_action_alignment_p_centered"
    diagnostics_alignment_h_dir = run_dir / "vis_action_alignment_h"
    diagnostics_alignment_h_raw_dir = run_dir / "vis_action_alignment_h_raw"
    diagnostics_alignment_h_centered_dir = run_dir / "vis_action_alignment_h_centered"
    diagnostics_cycle_dir = run_dir / "vis_cycle_error_z"
    diagnostics_cycle_p_dir = run_dir / "vis_cycle_error_p"
    diagnostics_cycle_h_dir = run_dir / "vis_cycle_error_h"
    diagnostics_frames_dir = run_dir / "vis_diagnostics_frames"
    vis_composability_z_dir = run_dir / "vis_composability_z"
    vis_composability_p_dir = run_dir / "vis_composability_p"
    vis_composability_h_dir = run_dir / "vis_composability_h"
    diagnostics_rollout_divergence_z_dir = run_dir / "vis_rollout_divergence_z"
    diagnostics_rollout_divergence_h_dir = run_dir / "vis_rollout_divergence_h"
    diagnostics_rollout_divergence_p_dir = run_dir / "vis_rollout_divergence_p"
    diagnostics_action_field_z_dir = run_dir / "vis_action_field_z"
    diagnostics_action_field_h_dir = run_dir / "vis_action_field_h"
    diagnostics_action_field_p_dir = run_dir / "vis_action_field_p"
    diagnostics_action_time_z_dir = run_dir / "vis_action_time_z"
    diagnostics_action_time_h_dir = run_dir / "vis_action_time_h"
    diagnostics_action_time_p_dir = run_dir / "vis_action_time_p"
    diagnostics_straightline_p_dir = run_dir / "vis_straightline_p"
    diagnostics_z_consistency_dir = run_dir / "vis_z_consistency"
    diagnostics_z_monotonicity_dir = run_dir / "vis_z_monotonicity"
    diagnostics_path_independence_dir = run_dir / "vis_path_independence"
    diagnostics_h_ablation_dir = run_dir / "vis_h_ablation"
    diagnostics_h_drift_dir = run_dir / "vis_h_drift_by_action"
    diagnostics_norm_timeseries_dir = run_dir / "vis_norm_timeseries"
    vis_ctrl_dir = run_dir / "vis_vis_ctrl"
    graph_diagnostics_dir = run_dir / "graph_diagnostics_z"
    graph_diagnostics_p_dir = run_dir / "graph_diagnostics_p"
    graph_diagnostics_h_dir = run_dir / "graph_diagnostics_h"

    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fixed_vis_dir.mkdir(parents=True, exist_ok=True)
    rolling_vis_dir.mkdir(parents=True, exist_ok=True)
    pca_z_dir.mkdir(parents=True, exist_ok=True)
    pca_p_dir.mkdir(parents=True, exist_ok=True)
    pca_h_dir.mkdir(parents=True, exist_ok=True)
    vis_off_manifold_dir.mkdir(parents=True, exist_ok=True)
    samples_hard_dir.mkdir(parents=True, exist_ok=True)
    samples_hard_val_dir.mkdir(parents=True, exist_ok=True)
    vis_self_distance_z_dir.mkdir(parents=True, exist_ok=True)
    vis_self_distance_p_dir.mkdir(parents=True, exist_ok=True)
    vis_self_distance_h_dir.mkdir(parents=True, exist_ok=True)
    vis_state_embedding_dir.mkdir(parents=True, exist_ok=True)
    vis_odometry_dir.mkdir(parents=True, exist_ok=True)
    self_distance_z_dir.mkdir(parents=True, exist_ok=True)
    self_distance_p_dir.mkdir(parents=True, exist_ok=True)
    self_distance_h_dir.mkdir(parents=True, exist_ok=True)
    if diagnostics_cfg.enabled:
        diagnostics_delta_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_delta_p_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_delta_h_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_alignment_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_alignment_raw_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_alignment_centered_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_alignment_p_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_alignment_p_raw_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_alignment_p_centered_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_alignment_h_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_alignment_h_raw_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_alignment_h_centered_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_cycle_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_cycle_p_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_cycle_h_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_frames_dir.mkdir(parents=True, exist_ok=True)
        vis_composability_z_dir.mkdir(parents=True, exist_ok=True)
        vis_composability_p_dir.mkdir(parents=True, exist_ok=True)
        vis_composability_h_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_rollout_divergence_z_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_rollout_divergence_h_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_rollout_divergence_p_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_action_field_z_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_action_field_h_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_action_field_p_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_action_time_z_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_action_time_h_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_action_time_p_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_straightline_p_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_z_consistency_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_z_monotonicity_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_path_independence_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_h_ablation_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_h_drift_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_norm_timeseries_dir.mkdir(parents=True, exist_ok=True)
    if graph_cfg.enabled:
        graph_diagnostics_dir.mkdir(parents=True, exist_ok=True)
        graph_diagnostics_p_dir.mkdir(parents=True, exist_ok=True)
        graph_diagnostics_h_dir.mkdir(parents=True, exist_ok=True)
    if vis_ctrl_cfg.enabled:
        vis_ctrl_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    sequences, grad_label = build_visualization_sequences(
        batch_cpu=fixed_batch_cpu,
        selection=fixed_selection,
        model=model,
        decoder=decoder,
        device=device,
        vis_cfg=vis_cfg,
        vis_selection_generator=vis_selection_generator,
        use_z2h_init=weights.z2h > 0,
    )
    save_rollout_sequence_batch(
        fixed_vis_dir,
        sequences,
        grad_label,
        global_step,
        include_pixel_delta=(weights.pixel_delta > 0 or weights.pixel_delta_multi_box > 0),
    )

    sequences, grad_label = build_visualization_sequences(
        batch_cpu=rolling_batch_cpu,
        selection=None,
        model=model,
        decoder=decoder,
        device=device,
        vis_cfg=vis_cfg,
        vis_selection_generator=vis_selection_generator,
        use_z2h_init=weights.z2h > 0,
    )
    save_rollout_sequence_batch(
        rolling_vis_dir,
        sequences,
        grad_label,
        global_step,
        include_pixel_delta=(weights.pixel_delta > 0 or weights.pixel_delta_multi_box > 0),
    )
    if off_manifold_batch_cpu is not None:
        assert torch.is_grad_enabled()
        with torch.no_grad():
            step_indices, errors = compute_off_manifold_errors(
                model=model,
                decoder=decoder,
                batch_cpu=off_manifold_batch_cpu,
                device=device,
                rollout_steps=off_manifold_steps,
            )
        assert torch.is_grad_enabled()
        save_off_manifold_visualization(
            vis_off_manifold_dir,
            step_indices,
            errors,
            global_step,
        )

    assert torch.is_grad_enabled()
    with torch.no_grad():
        embed_frames = embedding_batch_cpu[0].to(device)
        embed_actions = embedding_batch_cpu[1].to(device)
        embed_outputs = model.encode_sequence(embed_frames)
        _, _, h_states = rollout_teacher_forced(
            model,
            embed_outputs["embeddings"],
            embed_actions,
            use_z2h_init=weights.z2h > 0,
        )

        if model.p_action_delta_projector is not None:
            _, p_embeddings, _ = rollout_pose(
                model,
                h_states,
                embed_actions,
                z_embeddings=embed_outputs["embeddings"],
            )
        else:
            p_embeddings = None
    assert torch.is_grad_enabled()

    save_embedding_projection(
        embed_outputs["embeddings"],
        pca_z_dir / f"pca_z_{global_step:07d}.png",
        "PCA z",
    )
    if p_embeddings is not None:
        save_embedding_projection(
            p_embeddings,
            pca_p_dir / f"pca_p_{global_step:07d}.png",
            "PCA p",
        )
    save_embedding_projection(
        h_states,
        pca_h_dir / f"pca_h_{global_step:07d}.png",
        "PCA h",
    )

    if hard_reservoir is not None:
        hard_samples = hard_reservoir.topk(hard_example_cfg.vis_rows * hard_example_cfg.vis_columns)
        save_hard_example_grid(
            samples_hard_dir / f"hard_{global_step:07d}.png",
            hard_samples,
            hard_example_cfg.vis_columns,
            hard_example_cfg.vis_rows,
            dataset.image_hw,
        )
    if hard_reservoir_val is not None:
        hard_samples_val = hard_reservoir_val.topk(hard_example_cfg.vis_rows * hard_example_cfg.vis_columns)
        save_hard_example_grid(
            samples_hard_val_dir / f"hard_{global_step:07d}.png",
            hard_samples_val,
            hard_example_cfg.vis_columns,
            hard_example_cfg.vis_rows,
            dataset.image_hw,
        )

    assert torch.is_grad_enabled()
    with torch.no_grad():
        self_dist_frames = self_distance_inputs.frames.to(device)
        self_dist_actions = torch.from_numpy(self_distance_inputs.actions).unsqueeze(0).to(device)
        self_dist_embeddings_full = model.encode_sequence(self_dist_frames)["embeddings"]
        self_dist_embeddings = self_dist_embeddings_full[0]
        _, _, self_dist_h_states_batch = rollout_teacher_forced(
            model,
            self_dist_embeddings_full,
            self_dist_actions,
            use_z2h_init=weights.z2h > 0,
        )
        self_dist_h_states = self_dist_h_states_batch[0]
        self_dist_p = None
        if model.p_action_delta_projector is not None:
            _, self_dist_p_batch, _ = rollout_pose(
                model,
                self_dist_h_states_batch,
                self_dist_actions,
                z_embeddings=self_dist_embeddings_full,
            )
            self_dist_p = self_dist_p_batch[0]
    assert torch.is_grad_enabled()

    write_self_distance_outputs(
        self_dist_embeddings,
        self_distance_inputs,
        self_distance_z_dir,
        vis_self_distance_z_dir,
        global_step,
        embedding_label="z",
        title_prefix="Self-distance (Z)",
        file_prefix="self_distance_z",
        cosine_prefix="self_distance_z_cosine",
    )
    write_self_distance_outputs(
        self_dist_h_states,
        self_distance_inputs,
        self_distance_h_dir,
        vis_self_distance_h_dir,
        global_step,
        embedding_label="h",
        title_prefix="Self-distance (H)",
        file_prefix="self_distance_h",
        cosine_prefix="self_distance_h_cosine",
    )
    if self_dist_p is not None:
        write_self_distance_outputs(
            self_dist_p,
            self_distance_inputs,
            self_distance_p_dir,
            vis_self_distance_p_dir,
            global_step,
            embedding_label="p",
            title_prefix="Self-distance (P)",
            file_prefix="self_distance_p",
            cosine_prefix="self_distance_p_cosine",
        )
    if model.p_action_delta_projector is not None:
        write_state_embedding_outputs(
            model,
            self_distance_inputs,
            device,
            self_distance_p_dir,
            vis_self_distance_p_dir,
            vis_state_embedding_dir,
            vis_odometry_dir,
            global_step,
            use_z2h_init=weights.z2h > 0,
            hist_frames_cpu=rolling_batch_cpu[0],
            hist_actions_cpu=rolling_batch_cpu[1],
        )

    if diagnostics_cfg.enabled:
        diag_state = prepare_diagnostics_batch_state(
            model=model,
            diagnostics_cfg=diagnostics_cfg,
            weights=weights,
            diagnostics_batch_cpu=diagnostics_batch_cpu,
            device=device,
        )
        enabled_kinds = _enabled_kinds(weights, model)
        z_enabled = enabled_kinds.get("z", False)
        h_enabled = enabled_kinds.get("h", False)
        p_enabled = enabled_kinds.get("p", False)

        kinds = [("z", z_enabled), ("h", h_enabled), ("p", p_enabled)]
        enabled_kinds_list = [kind for kind, enabled in kinds if enabled]
        if enabled_kinds_list:
            action_ids_flat = diag_state.action_metadata.action_ids_flat
            for kind in ["z", "h", "p"]:
                if kind not in enabled_kinds_list:
                    continue
                if kind == "z":
                    embed_np = diag_state.embeddings.detach().cpu().numpy()
                    deltas = embed_np[:, 1:] - embed_np[:, :-1]
                    action_dim = diag_state.motion_z.action_dim
                    action_field_dir = diagnostics_action_field_z_dir
                    action_time_dir = diagnostics_action_time_z_dir
                    title = "Action-conditioned vector field (Z)"
                    time_title = "Action delta time slices (Z)"
                elif kind == "h":
                    embed_np = diag_state.h_states.detach().cpu().numpy()
                    deltas = embed_np[:, 1:] - embed_np[:, :-1]
                    action_dim = diag_state.motion_h.action_dim
                    action_field_dir = diagnostics_action_field_h_dir
                    action_time_dir = diagnostics_action_time_h_dir
                    title = "Action-conditioned vector field (H)"
                    time_title = "Action delta time slices (H)"
                else:
                    if not diag_state.has_p or diag_state.motion_p is None or diag_state.p_embeddings is None:
                        raise AssertionError("Action-field diagnostics for p require pose embeddings.")
                    embed_np = diag_state.p_embeddings.detach().cpu().numpy()
                    deltas = embed_np[:, 1:] - embed_np[:, :-1]
                    action_dim = diag_state.motion_p.action_dim
                    action_field_dir = diagnostics_action_field_p_dir
                    action_time_dir = diagnostics_action_time_p_dir
                    title = "Action-conditioned vector field (P)"
                    time_title = "Action delta time slices (P)"
                save_action_vector_field_plot(
                    action_field_dir / f"action_field_{kind}_{global_step:07d}.png",
                    embed_np[:, :-1].reshape(-1, embed_np.shape[-1]),
                    deltas.reshape(-1, deltas.shape[-1]),
                    action_ids_flat,
                    action_dim,
                    max_actions=diagnostics_cfg.max_actions_to_plot,
                    min_count=diagnostics_cfg.min_action_count,
                    title=title,
                )
                save_action_time_slice_plot(
                    action_time_dir / f"action_time_{kind}_{global_step:07d}.png",
                    deltas,
                    diag_state.action_ids_seq[:, :-1],
                    action_dim,
                    max_actions=diagnostics_cfg.max_actions_to_plot,
                    min_count=diagnostics_cfg.min_action_count,
                    title=time_title,
                )

        if diag_state.composability is not None:
            for kind in ["z", "h", "p"]:
                if kind not in enabled_kinds_list:
                    continue
                if kind == "z":
                    save_composability_plot(
                        vis_composability_z_dir / f"composability_z_{global_step:07d}.png",
                        diag_state.composability["z"],
                        "z",
                    )
                elif kind == "h":
                    save_composability_plot(
                        vis_composability_h_dir / f"composability_h_{global_step:07d}.png",
                        diag_state.composability["h"],
                        "h",
                    )
                else:
                    if diag_state.p_series is None:
                        raise AssertionError("Composability diagnostics for p require p_series.")
                    save_composability_plot(
                        vis_composability_p_dir / f"composability_p_{global_step:07d}.png",
                        diag_state.p_series,
                        "p",
                    )
        if enabled_kinds_list:
            for kind in ["z", "h", "p"]:
                if kind not in enabled_kinds_list:
                    continue
                if kind == "z":
                    motion = diag_state.motion_z
                    delta_dir = diagnostics_delta_dir
                    align_dir = diagnostics_alignment_dir
                    align_raw_dir = diagnostics_alignment_raw_dir
                    align_center_dir = diagnostics_alignment_centered_dir
                    cycle_dir = diagnostics_cycle_dir
                elif kind == "h":
                    motion = diag_state.motion_h
                    delta_dir = diagnostics_delta_h_dir
                    align_dir = diagnostics_alignment_h_dir
                    align_raw_dir = diagnostics_alignment_h_raw_dir
                    align_center_dir = diagnostics_alignment_h_centered_dir
                    cycle_dir = diagnostics_cycle_h_dir
                else:
                    if not diag_state.has_p or diag_state.motion_p is None:
                        raise AssertionError("Alignment diagnostics for p require pose embeddings.")
                    motion = diag_state.motion_p
                    delta_dir = diagnostics_delta_p_dir
                    align_dir = diagnostics_alignment_p_dir
                    align_raw_dir = diagnostics_alignment_p_raw_dir
                    align_center_dir = diagnostics_alignment_p_centered_dir
                    cycle_dir = diagnostics_cycle_p_dir
                write_motion_pca_artifacts(
                    diagnostics_cfg=diagnostics_cfg,
                    global_step=global_step,
                    name=kind,
                    motion=motion,
                    delta_dir=delta_dir,
                )
                write_alignment_artifacts(
                    diagnostics_cfg=diagnostics_cfg,
                    global_step=global_step,
                    name=kind,
                    motion=motion,
                    inverse_map=diag_state.inverse_map,
                    alignment_dir=align_dir,
                    alignment_raw_dir=align_raw_dir,
                    alignment_centered_dir=align_center_dir,
                )
                write_cycle_error_artifacts(
                    diagnostics_cfg=diagnostics_cfg,
                    global_step=global_step,
                    name=kind,
                    motion=motion,
                    inverse_map=diag_state.inverse_map,
                    cycle_dir=cycle_dir,
                )
        write_alignment_debug_csv(
            diag_state.frames,
            diag_state.actions,
            diag_state.paths,
            diagnostics_frames_dir,
            global_step,
        )
        save_diagnostics_frames(
            diag_state.frames,
            diag_state.paths,
            diag_state.actions,
            diagnostics_frames_dir,
            global_step,
        )
        diagnostics_scalars_path = metrics_dir / "diagnostics_scalars.csv"
        z_norm_mean, z_norm_p95 = compute_norm_stats(diag_state.embeddings)
        h_norm_mean, h_norm_p95 = compute_norm_stats(diag_state.h_states)
        if diag_state.has_p and diag_state.p_embeddings is not None:
            p_norm_mean, p_norm_p95 = compute_norm_stats(diag_state.p_embeddings)
        else:
            p_norm_mean, p_norm_p95 = 0.0, 0.0

        if diag_state.h_states.shape[1] >= 2:
            h_drift = diag_state.h_states[:, 1:] - diag_state.h_states[:, :-1]
            h_drift_mean = float(h_drift.norm(dim=-1).mean().item())
            if diag_state.has_p and diag_state.p_deltas is not None and diag_state.p_deltas.numel() > 0:
                p_drift_mean = float(diag_state.p_deltas.norm(dim=-1).mean().item())
            else:
                p_drift_mean = 0.0
        else:
            h_drift_mean = 0.0
            p_drift_mean = 0.0

        id_acc_z = 0.0
        id_acc_h = 0.0
        id_acc_p = 0.0
        if weights.inverse_dynamics_z > 0 and diag_state.embeddings.shape[1] >= 2:
            action_logits_z = model.inverse_dynamics_z(
                diag_state.embeddings[:, :-1],
                diag_state.embeddings[:, 1:],
            )
            action_preds_z = (torch.sigmoid(action_logits_z) > 0.5).to(diag_state.actions_device.dtype)
            id_acc_z = float((action_preds_z == diag_state.actions_device[:, :-1]).float().mean().item())
        if weights.inverse_dynamics_h > 0 and diag_state.h_states.shape[1] >= 2:
            action_logits_h = model.inverse_dynamics_h(
                diag_state.h_states[:, :-1],
                diag_state.h_states[:, 1:],
            )
            action_preds_h = (torch.sigmoid(action_logits_h) > 0.5).to(diag_state.actions_device.dtype)
            id_acc_h = float((action_preds_h == diag_state.actions_device[:, :-1]).float().mean().item())
        if (
            diag_state.has_p
            and weights.inverse_dynamics_p > 0
            and diag_state.p_embeddings is not None
            and diag_state.p_embeddings.shape[1] >= 2
        ):
            action_logits_p = model.inverse_dynamics_p(
                diag_state.p_embeddings[:, :-1],
                diag_state.p_embeddings[:, 1:],
            )
            action_preds = (torch.sigmoid(action_logits_p) > 0.5).to(diag_state.actions_device.dtype)
            id_acc_p = float((action_preds == diag_state.actions_device[:, :-1]).float().mean().item())

        append_csv_row(
            diagnostics_scalars_path,
            [
                "step",
                "z_norm_mean",
                "z_norm_p95",
                "h_norm_mean",
                "h_norm_p95",
                "p_norm_mean",
                "p_norm_p95",
                "h_drift_mean",
                "p_drift_mean",
                "id_acc_z",
                "id_acc_h",
                "id_acc_p",
            ],
            [
                global_step,
                z_norm_mean,
                z_norm_p95,
                h_norm_mean,
                h_norm_p95,
                p_norm_mean,
                p_norm_p95,
                h_drift_mean,
                p_drift_mean,
                id_acc_z,
                id_acc_h,
                id_acc_p,
            ],
        )

        try:
            with diagnostics_scalars_path.open("r", newline="") as handle:
                reader = csv.DictReader(handle)
                steps_list: List[int] = []
                z_mean_list: List[float] = []
                z_p95_list: List[float] = []
                h_mean_list: List[float] = []
                h_p95_list: List[float] = []
                p_mean_list: List[float] = []
                p_p95_list: List[float] = []
                for row in reader:
                    steps_list.append(int(float(row["step"])))
                    z_mean_list.append(float(row["z_norm_mean"]))
                    z_p95_list.append(float(row["z_norm_p95"]))
                    h_mean_list.append(float(row["h_norm_mean"]))
                    h_p95_list.append(float(row["h_norm_p95"]))
                    p_mean_list.append(float(row.get("p_norm_mean", row.get("s_norm_mean", 0.0))))
                    p_p95_list.append(float(row.get("p_norm_p95", row.get("s_norm_p95", 0.0))))
            if steps_list:
                save_norm_timeseries_plot(
                    diagnostics_norm_timeseries_dir / f"norm_timeseries_{global_step:07d}.png",
                    steps_list,
                    z_mean_list,
                    z_p95_list,
                    h_mean_list,
                    h_p95_list,
                    p_mean_list,
                    p_p95_list,
                )
        except OSError:
            pass
        if p_enabled:
            for kind in ["p"]:
                if diag_state.p_embeddings is None or not diag_state.has_p:
                    raise AssertionError("Straightline diagnostics require p_embeddings.")
                if diag_state.p_embeddings.shape[1] < 2:
                    raise AssertionError("Straightline diagnostics require at least two timesteps.")
                max_starts = min(diagnostics_cfg.straightline_starts, diag_state.p_embeddings.shape[0])
                if max_starts <= 0 or len(diag_state.action_metadata.unique_actions) == 0:
                    continue
                action_labels = diag_state.action_metadata.action_labels
                action_vectors = diag_state.action_metadata.action_vectors
                unique_actions = diag_state.action_metadata.unique_actions
                action_counts = diag_state.action_metadata.action_counts

                right_id = next((aid for aid, label in action_labels.items() if "RIGHT" in label), None)
                left_id = next((aid for aid, label in action_labels.items() if "LEFT" in label), None)
                up_id = next((aid for aid, label in action_labels.items() if "UP" in label), None)
                down_id = next((aid for aid, label in action_labels.items() if "DOWN" in label), None)

                sorted_ids = [
                    int(aid)
                    for aid, _ in sorted(
                        zip(unique_actions.tolist(), action_counts.tolist()),
                        key=lambda pair: pair[1],
                        reverse=True,
                    )
                ]
                if right_id is not None and left_id is not None:
                    straightline_ids = [right_id, left_id]
                elif up_id is not None and down_id is not None:
                    straightline_ids = [up_id, down_id]
                else:
                    straightline_ids = sorted_ids[:2]

                if not straightline_ids:
                    raise AssertionError("Straightline diagnostics require at least one action ID.")

                p_flat = (
                    diag_state.p_embeddings.detach()
                    .cpu()
                    .reshape(-1, diag_state.p_embeddings.shape[-1])
                    .numpy()
                )
                p_center = p_flat.mean(axis=0, keepdims=True)
                centered = p_flat - p_center
                try:
                    _, _, vt = np.linalg.svd(centered, full_matrices=False)
                except np.linalg.LinAlgError:
                    jitter = np.random.normal(scale=1e-6, size=centered.shape)
                    _, _, vt = np.linalg.svd(centered + jitter, full_matrices=False)
                projection = vt[:2].T if vt.shape[0] >= 2 else None
                if projection is None:
                    raise AssertionError("Straightline diagnostics require a 2D projection.")

                seq_len = diag_state.p_embeddings.shape[1]
                start_frame = max(min(diag_state.warmup_frames, seq_len - 2), 0)
                perm = torch.randperm(diag_state.p_embeddings.shape[0], generator=diagnostics_generator)[:max_starts]
                trajectories: List[StraightLineTrajectory] = []
                palette = ["#4c72b0", "#55a868", "#c44e52", "#8172b3"]
                for action_idx, action_id in enumerate(straightline_ids):
                    action_vec = action_vectors.get(action_id)
                    if action_vec is None:
                        continue
                    label = action_labels.get(action_id, f"action {action_id}")
                    color = palette[action_idx % len(palette)]
                    for row_offset, b_idx in enumerate(perm):
                        b = int(b_idx.item())
                        z_t = diag_state.embeddings[b, start_frame]
                        h_t = diag_state.h_states[b, start_frame]
                        p_t = diag_state.p_embeddings[b, start_frame]
                        p_points = [p_t.detach().cpu().numpy()]
                        for _ in range(diagnostics_cfg.straightline_steps):
                            h_in = h_t.detach() if model.cfg.pose_delta_detach_h else h_t
                            delta = model.p_action_delta_projector(
                                p_t.unsqueeze(0),
                                h_in.unsqueeze(0),
                                action_vec.unsqueeze(0),
                            ).squeeze(0)
                            p_t = p_t + delta
                            h_next = model.predictor(
                                z_t.unsqueeze(0),
                                h_t.unsqueeze(0),
                                action_vec.unsqueeze(0),
                            )
                            z_t = model.h_to_z(h_next).squeeze(0)
                            h_t = h_next.squeeze(0)
                            p_points.append(p_t.detach().cpu().numpy())
                        p_points_np = np.stack(p_points, axis=0)
                        proj = (p_points_np - p_center) @ projection
                        traj_label = f"{label} (start {row_offset + 1})"
                        trajectories.append(
                            StraightLineTrajectory(points=proj, label=traj_label, color=color)
                        )
                if trajectories:
                    save_straightline_plot(
                        diagnostics_straightline_p_dir / f"straightline_p_{global_step:07d}.png",
                        trajectories,
                    )
        rollout_horizon = min(diagnostics_cfg.rollout_divergence_horizon, diag_state.frames.shape[1] - 1)
        start_span = diag_state.frames.shape[1] - 1 - diag_state.warmup_frames
        if enabled_kinds_list and rollout_horizon > 0 and start_span > 0:
            horizons, pixel_mean, z_mean, h_mean, p_mean = compute_rollout_divergence_metrics(
                model=model,
                decoder=decoder,
                diag_embeddings=diag_state.embeddings,
                diag_h_states=diag_state.h_states,
                diag_p_embeddings=diag_state.p_embeddings,
                diag_actions_device=diag_state.actions_device,
                diag_frames_device=diag_state.frames_device,
                rollout_horizon=rollout_horizon,
                warmup_frames=diag_state.warmup_frames,
                start_span=start_span,
                rollout_divergence_samples=diagnostics_cfg.rollout_divergence_samples,
                diagnostics_generator=diagnostics_generator,
            )
            for kind in ["z", "h", "p"]:
                if kind not in enabled_kinds_list:
                    continue
                if kind == "z":
                    metric = z_mean
                    out_dir = diagnostics_rollout_divergence_z_dir
                    latent_label = "Z error"
                    title = "Rollout divergence (Z)"
                    csv_cols = ["k", "pixel_error", "z_error"]
                elif kind == "h":
                    metric = h_mean
                    out_dir = diagnostics_rollout_divergence_h_dir
                    latent_label = "H error"
                    title = "Rollout divergence (H)"
                    csv_cols = ["k", "pixel_error", "h_error"]
                else:
                    if not diag_state.has_p:
                        raise AssertionError("Rollout divergence for p requires pose embeddings.")
                    metric = p_mean
                    out_dir = diagnostics_rollout_divergence_p_dir
                    latent_label = "P error"
                    title = "Rollout divergence (P)"
                    csv_cols = ["k", "pixel_error", "p_error"]
                save_rollout_divergence_plot(
                    out_dir / f"rollout_divergence_{kind}_{global_step:07d}.png",
                    horizons,
                    pixel_mean,
                    metric,
                    latent_label=latent_label,
                    title=title,
                )
                write_step_csv(
                    out_dir,
                    f"rollout_divergence_{kind}_{global_step:07d}.csv",
                    csv_cols,
                    zip(horizons, pixel_mean, metric),
                )
            if diag_state.has_p and diag_state.p_embeddings is not None and diag_state.p_embeddings.shape[1] >= 2:
                horizons, pixel_mean, pixel_zero_mean, latent_mean, latent_zero_mean = compute_h_ablation_divergence(
                    model=model,
                    decoder=decoder,
                    diag_embeddings=diag_state.embeddings,
                    diag_h_states=diag_state.h_states,
                    diag_p_embeddings=diag_state.p_embeddings,
                    diag_actions_device=diag_state.actions_device,
                    diag_frames_device=diag_state.frames_device,
                    rollout_horizon=rollout_horizon,
                    warmup_frames=diag_state.warmup_frames,
                    start_span=start_span,
                    rollout_divergence_samples=diagnostics_cfg.rollout_divergence_samples,
                    diagnostics_generator=diagnostics_generator,
                )
                save_ablation_divergence_plot(
                    diagnostics_h_ablation_dir / f"h_ablation_{global_step:07d}.png",
                    horizons,
                    pixel_mean,
                    pixel_zero_mean,
                    latent_mean,
                    latent_zero_mean,
                )
                write_step_csv(
                    diagnostics_h_ablation_dir,
                    f"h_ablation_{global_step:07d}.csv",
                    ["k", "pixel_error", "pixel_error_zero", "latent_error", "latent_error_zero"],
                    zip(horizons, pixel_mean, pixel_zero_mean, latent_mean, latent_zero_mean),
                )
        if z_enabled:
            for kind in ["z"]:
                z_consistency_samples = min(
                    diagnostics_cfg.z_consistency_samples,
                    diag_state.frames.shape[0] * diag_state.frames.shape[1],
                )
                if z_consistency_samples > 0:
                    frame_count = diag_state.frames.shape[0] * diag_state.frames.shape[1]
                    perm = torch.randperm(frame_count, generator=diagnostics_generator)[:z_consistency_samples]
                    distances: List[float] = []
                    cosines: List[float] = []
                    for flat_idx in perm.tolist():
                        b = flat_idx // diag_state.frames.shape[1]
                        t0 = flat_idx % diag_state.frames.shape[1]
                        frame = diag_state.frames_device[b, t0]
                        repeats = diagnostics_cfg.z_consistency_repeats
                        noise = torch.randn(
                            (repeats, *frame.shape),
                            device=frame.device,
                        ) * diagnostics_cfg.z_consistency_noise_std
                        noisy = (frame.unsqueeze(0) + noise).clamp(0, 1)
                        z_samples = model.encoder(noisy)
                        z_mean = z_samples.mean(dim=0, keepdim=True)
                        dist = (z_samples - z_mean).norm(dim=-1)
                        cos = F.cosine_similarity(z_samples, z_mean, dim=-1)
                        distances.extend(dist.detach().cpu().numpy().tolist())
                        cosines.extend(cos.detach().cpu().numpy().tolist())
                    if distances and cosines:
                        save_z_consistency_plot(
                            diagnostics_z_consistency_dir / f"z_consistency_{global_step:07d}.png",
                            distances,
                            cosines,
                        )
                        write_step_csv(
                            diagnostics_z_consistency_dir,
                            f"z_consistency_{global_step:07d}.csv",
                            ["idx", "distance", "cosine"],
                            [(idx, d, c) for idx, (d, c) in enumerate(zip(distances, cosines))],
                        )
                monotonicity_samples = min(
                    diagnostics_cfg.z_monotonicity_samples,
                    diag_state.frames.shape[0] * diag_state.frames.shape[1],
                )
                if monotonicity_samples > 0:
                    max_shift = max(1, diagnostics_cfg.z_monotonicity_max_shift)
                    shifts, distances = compute_z_monotonicity_distances(
                        model=model,
                        diag_frames_device=diag_state.frames_device,
                        max_shift=max_shift,
                        monotonicity_samples=monotonicity_samples,
                        diagnostics_generator=diagnostics_generator,
                    )
                    save_monotonicity_plot(
                        diagnostics_z_monotonicity_dir / f"z_monotonicity_{global_step:07d}.png",
                        shifts,
                        distances,
                    )
                    write_step_csv(
                        diagnostics_z_monotonicity_dir,
                        f"z_monotonicity_{global_step:07d}.csv",
                        ["shift", "distance"],
                        zip(shifts, distances),
                    )
        if p_enabled:
            for kind in ["p"]:
                if not diag_state.has_p or diag_state.p_embeddings is None:
                    raise AssertionError("Path independence diagnostics require pose embeddings.")
                if diag_state.embeddings.shape[1] < 2:
                    raise AssertionError("Path independence diagnostics require at least two timesteps.")
                if len(diag_state.action_metadata.unique_actions) < 2:
                    raise AssertionError("Path independence diagnostics require at least two action IDs.")
                if diagnostics_cfg.path_independence_samples <= 0:
                    continue
                action_labels = diag_state.action_metadata.action_labels
                action_vectors = diag_state.action_metadata.action_vectors
                unique_actions = diag_state.action_metadata.unique_actions
                action_counts = diag_state.action_metadata.action_counts

                right_id = next((aid for aid, label in action_labels.items() if "RIGHT" in label), None)
                left_id = next((aid for aid, label in action_labels.items() if "LEFT" in label), None)
                up_id = next((aid for aid, label in action_labels.items() if "UP" in label), None)
                down_id = next((aid for aid, label in action_labels.items() if "DOWN" in label), None)

                sorted_ids = [
                    int(aid)
                    for aid, _ in sorted(
                        zip(unique_actions.tolist(), action_counts.tolist()),
                        key=lambda pair: pair[1],
                        reverse=True,
                    )
                ]
                if right_id is not None and left_id is not None:
                    straightline_ids = [right_id, left_id]
                elif up_id is not None and down_id is not None:
                    straightline_ids = [up_id, down_id]
                else:
                    straightline_ids = sorted_ids[:2]
                if len(straightline_ids) < 2:
                    raise AssertionError("Path independence diagnostics require at least two action IDs.")
                a_id = straightline_ids[0]
                b_id = straightline_ids[1] if len(straightline_ids) > 1 else straightline_ids[0]
                action_a = action_vectors.get(a_id)
                action_b = action_vectors.get(b_id)
                if action_a is None or action_b is None:
                    raise AssertionError("Path independence diagnostics require action vectors for both actions.")

                seq_len = diag_state.embeddings.shape[1]
                start_frame = max(min(diag_state.warmup_frames, seq_len - 2), 0)
                max_starts = min(diagnostics_cfg.path_independence_samples, diag_state.embeddings.shape[0])
                if max_starts <= 0:
                    raise AssertionError("Path independence diagnostics require at least one start.")
                use_noop = (
                    diag_state.action_metadata.noop_id is not None
                    and diag_state.action_metadata.noop_id in action_vectors
                )
                action_b_first = action_vectors.get(diag_state.action_metadata.noop_id) if use_noop else action_b
                action_b_second = action_vectors.get(diag_state.action_metadata.noop_id) if use_noop else action_a
                z_diffs, p_diffs = compute_path_independence_diffs(
                    model=model,
                    diag_embeddings=diag_state.embeddings,
                    diag_h_states=diag_state.h_states,
                    diag_p_embeddings=diag_state.p_embeddings,
                    action_a=action_a,
                    action_b=action_b,
                    action_b_first=action_b_first,
                    action_b_second=action_b_second,
                    start_frame=start_frame,
                    max_starts=max_starts,
                    path_independence_steps=diagnostics_cfg.path_independence_steps,
                    diagnostics_generator=diagnostics_generator,
                )
                if z_diffs and p_diffs:
                    label_a = action_labels.get(a_id, f"action {a_id}")
                    label_b = action_labels.get(b_id, f"action {b_id}")
                    if use_noop:
                        label_b_path = action_labels.get(diag_state.action_metadata.noop_id, "NOOP")
                    else:
                        label_b_path = f"{label_b}+{label_a}"
                    labels = [f"{label_a}+{label_b} vs {label_b_path}"]
                    save_path_independence_plot(
                        diagnostics_path_independence_dir / f"path_independence_{global_step:07d}.png",
                        labels,
                        [float(np.mean(z_diffs))],
                        [float(np.mean(p_diffs))],
                    )
                    write_step_csv(
                        diagnostics_path_independence_dir,
                        f"path_independence_{global_step:07d}.csv",
                        ["label", "z_distance", "p_distance"],
                        [(labels[0], float(np.mean(z_diffs)), float(np.mean(p_diffs)))],
                    )
        if h_enabled:
            for kind in ["h"]:
                if diag_state.h_states.shape[1] < 2:
                    raise AssertionError("H-drift diagnostics require at least two timesteps.")
                h_deltas = diag_state.h_states[:, 1:] - diag_state.h_states[:, :-1]
                action_ids = compress_actions_to_ids(diag_state.actions[:, :-1].detach().cpu().numpy())
                action_ids_flat = action_ids.reshape(-1)
                h_drift_flat = h_deltas.detach().cpu().reshape(-1, h_deltas.shape[-1]).norm(dim=-1).numpy()
                drift_stats = []
                for aid in np.unique(action_ids_flat):
                    mask = action_ids_flat == aid
                    count = int(mask.sum())
                    if count == 0:
                        continue
                    samples = h_drift_flat[mask]
                    mean_drift = float(samples.mean())
                    drift_stats.append((int(aid), count, mean_drift, samples.tolist()))
                drift_stats.sort(key=lambda row: row[1], reverse=True)
                drift_stats = drift_stats[: diagnostics_cfg.h_drift_max_actions]
                if drift_stats:
                    action_labels = diag_state.action_metadata.action_labels
                    labels = [action_labels.get(aid, f"action {aid}") for aid, _, _, _ in drift_stats]
                    drift_samples = [samples for _, _, _, samples in drift_stats]
                    save_drift_by_action_plot(
                        diagnostics_h_drift_dir / f"h_drift_by_action_{global_step:07d}.png",
                        labels,
                        drift_samples,
                    )
                    write_step_csv(
                        diagnostics_h_drift_dir,
                        f"h_drift_by_action_{global_step:07d}.csv",
                        ["action_id", "label", "count", "mean_drift"],
                        [
                            (aid, action_labels.get(aid, f"action {aid}"), count, drift)
                            for aid, count, drift, _ in drift_stats
                        ],
                    )

    if vis_ctrl_cfg.enabled:
        vis_embeddings, vis_h_states, vis_actions, vis_p_embeddings = compute_vis_ctrl_state(
            model=model,
            weights=weights,
            device=device,
            vis_ctrl_batch_cpu=vis_ctrl_batch_cpu,
        )
        warmup_frames = max(model.cfg.warmup_frames_h, 0)
        vis_kind_inputs = {
            "z": vis_embeddings,
            "h": vis_h_states,
            "p": vis_p_embeddings,
        }
        metrics_by_kind = {}
        for kind in ["z", "h", "p"]:
            embeddings = vis_kind_inputs.get(kind)
            if embeddings is None:
                if kind == "p":
                    continue
                raise AssertionError(f"Vis-ctrl diagnostics for {kind} require embeddings.")
            metrics = compute_vis_ctrl_metrics(
                embeddings,
                vis_actions,
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
            metrics_by_kind[kind] = metrics
        write_vis_ctrl_summary(
            metrics_dir=metrics_dir,
            global_step=global_step,
            metrics_z=metrics_by_kind["z"],
            metrics_p=metrics_by_kind.get("p"),
            metrics_h=metrics_by_kind["h"],
        )

    if graph_cfg.enabled:
        graph_diag = prepare_graph_diagnostics(
            graph_frames=graph_diag_batch_cpu[0],
            graph_actions=graph_diag_batch_cpu[1],
            model=model,
            ema_model=ema_model,
            graph_cfg=graph_cfg,
            device=device,
            use_z2h_init=weights.z2h > 0,
        )
        graph_kinds = ["z", "h"]
        if model.p_action_delta_projector is not None:
            graph_kinds.append("p")
        for kind in ["z", "h", "p"]:
            if kind not in graph_kinds:
                continue
            if kind == "z":
                graph_dir = graph_diagnostics_dir
            elif kind == "h":
                graph_dir = graph_diagnostics_h_dir
            else:
                graph_dir = graph_diagnostics_p_dir
            if kind == "p":
                if model.p_action_delta_projector is None:
                    raise AssertionError("Graph diagnostics for p require p_action_delta_projector.")
                _, p_targets, _ = rollout_pose(
                    model,
                    graph_diag.graph_h_states,
                    graph_diag.graph_actions,
                    z_embeddings=graph_diag.graph_embeddings,
                )
                h_pred_states = graph_diag.graph_h_preds
                if h_pred_states.shape[1] + 1 == graph_diag.graph_h_states.shape[1]:
                    h_pred_states = torch.cat([graph_diag.graph_h_states[:, :1], h_pred_states], dim=1)
                if graph_diag.graph_preds.shape[1] + 1 == graph_diag.graph_h_states.shape[1]:
                    z_pred_embeddings = torch.cat([graph_diag.graph_embeddings[:, :1], graph_diag.graph_preds], dim=1)
                elif graph_diag.graph_preds.shape[1] == graph_diag.graph_h_states.shape[1]:
                    z_pred_embeddings = graph_diag.graph_preds
                else:
                    raise AssertionError(
                        "graph_preds must match graph_h_states in time (T or T-1) to build pose rollouts."
                    )
                _, p_hat, _ = rollout_pose(
                    model,
                    h_pred_states,
                    graph_diag.graph_actions,
                    z_embeddings=z_pred_embeddings,
                )
                p_hat_full = torch.cat([p_hat, p_targets[:, -1:, :]], dim=1)
                if (
                    graph_cfg.use_ema_targets
                    and ema_model is not None
                    and graph_diag.ema_h_states is not None
                    and graph_diag.ema_embeddings is not None
                ):
                    if ema_model.p_action_delta_projector is None:
                        raise AssertionError("Graph diagnostics for p require ema p_action_delta_projector.")
                    _, targets, _ = rollout_pose(
                        ema_model,
                        graph_diag.ema_h_states,
                        graph_diag.graph_actions,
                        z_embeddings=graph_diag.ema_embeddings,
                    )
                else:
                    targets = p_targets
                z_flat = p_targets.reshape(-1, p_targets.shape[-1])
                target_flat = targets.reshape(-1, targets.shape[-1])
                zhat_full = p_hat_full.reshape(-1, p_hat_full.shape[-1])
            elif kind == "h":
                targets = graph_diag.ema_h_states if graph_cfg.use_ema_targets and graph_diag.ema_h_states is not None else graph_diag.graph_h_states
                z_flat = graph_diag.graph_h_states.reshape(-1, graph_diag.graph_h_states.shape[-1])
                target_flat = targets.reshape(-1, targets.shape[-1])
                zhat_full = torch.cat(
                    [graph_diag.graph_h_preds, graph_diag.graph_h_states[:, -1:, :]],
                    dim=1,
                ).reshape(-1, graph_diag.graph_h_states.shape[-1])
            else:
                targets = (
                    graph_diag.ema_embeddings
                    if graph_cfg.use_ema_targets and graph_diag.ema_embeddings is not None
                    else graph_diag.graph_embeddings
                )
                z_flat = graph_diag.graph_embeddings.reshape(-1, graph_diag.graph_embeddings.shape[-1])
                target_flat = targets.reshape(-1, targets.shape[-1])
                zhat_full = torch.cat(
                    [graph_diag.graph_preds, graph_diag.graph_embeddings[:, -1:, :]],
                    dim=1,
                ).reshape(-1, graph_diag.graph_embeddings.shape[-1])

            if graph_cfg.normalize_latents:
                z_flat = F.normalize(z_flat, dim=-1)
                target_flat = F.normalize(target_flat, dim=-1)
                zhat_full = F.normalize(zhat_full, dim=-1)

            queries = zhat_full if graph_cfg.use_predictor_scores else z_flat
            stats = compute_graph_diagnostics_stats(
                queries,
                target_flat,
                zhat_full,
                graph_diag.next_index,
                graph_diag.next2_index,
                graph_diag.chunk_ids,
                graph_cfg,
                global_step,
            )
            save_rank_cdf_plot(
                graph_dir / f"rank1_cdf_{global_step:07d}.png",
                stats.ranks1,
                stats.k,
                "1-step rank CDF",
            )
            save_rank_cdf_plot(
                graph_dir / f"rank2_cdf_{global_step:07d}.png",
                stats.ranks2,
                stats.k,
                "2-hop rank CDF",
            )
            save_neff_violin_plot(
                graph_dir / f"neff_violin_{global_step:07d}.png",
                stats.neff1,
                stats.neff2,
            )
            save_in_degree_hist_plot(
                graph_dir / f"in_degree_hist_{global_step:07d}.png",
                stats.in_degree,
            )
            save_edge_consistency_hist_plot(
                graph_dir / f"edge_consistency_{global_step:07d}.png",
                stats.edge_errors,
                embedding_label=kind,
            )
            update_graph_diagnostics_history(
                graph_dir,
                stats,
                global_step,
                metrics_dir / f"graph_diagnostics_{kind}.csv",
            )

    model.train()
