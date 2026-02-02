from __future__ import annotations

import csv
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from jepa_world_model.actions import compress_actions_to_ids
from jepa_world_model.config_planning import PlanningDiagnosticsConfig
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
from jepa_world_model.diagnostics_utils import (
    append_csv_row,
    compute_norm_stats,
    should_use_z2h_init,
    write_step_csv,
)
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
from jepa_world_model.plots.plot_planning_graph import save_planning_graph_plot
from jepa_world_model.plots.plot_rank_cdf import save_rank_cdf_plot
from jepa_world_model.plots.plot_reachable_fraction_hist import save_reachable_fraction_hist_plot
from jepa_world_model.plots.plot_smoothness_knn_distance_eigenvalue_spectrum import (
    save_smoothness_knn_distance_eigenvalue_spectrum_plot,
)
from jepa_world_model.plots.plot_two_step_composition_error import save_two_step_composition_error_plot
from jepa_world_model.plots.write_alignment_debug_csv import write_alignment_debug_csv
from jepa_world_model.planning.planning_eval import (
    DIRECTION_ORDER,
    ActionDeltaStats,
    DatasetGraph,
    PlanningTestResult,
    action_labels_from_vectors,
    bfs_plan,
    build_dataset_graph,
    cluster_latents,
    compute_action_delta_stats,
    delta_lattice_astar,
    plot_action_stats,
    plot_action_strip,
    plot_grid_trace,
    plot_pca_path,
    reachable_fractions,
    run_plan_in_env,
)
from gridworldkey_env import GridworldKeyEnv
from jepa_world_model.pose_rollout import rollout_pose
from jepa_world_model.rollout import rollout_teacher_forced
from jepa_world_model.vis_composability import save_composability_plot
from jepa_world_model.vis_embedding_projection import save_embedding_projection
from jepa_world_model.vis_hard_samples import save_hard_example_grid
from jepa_world_model.vis_rollout_batch import save_rollout_sequence_batch
from jepa_world_model.vis_self_distance import write_self_distance_outputs
from jepa_world_model.vis_state_embedding import write_state_embedding_outputs
from jepa_world_model.vis_vis_ctrl_metrics import compute_vis_ctrl_metrics


@dataclass(frozen=True)
class DiagnosticsOutputSpec:
    enabled: bool
    description: str


DIAGNOSTICS_OUTPUT_CATALOG = {
    "vis_fixed_0": DiagnosticsOutputSpec(True, "Fixed batch rollout sequence grid (view 0)."),
    "vis_fixed_1": DiagnosticsOutputSpec(True, "Fixed batch rollout sequence grid (view 1)."),
    "vis_rolling_0": DiagnosticsOutputSpec(True, "Rolling batch rollout sequence grid (view 0)."),
    "vis_rolling_1": DiagnosticsOutputSpec(True, "Rolling batch rollout sequence grid (view 1)."),
    "vis_off_manifold": DiagnosticsOutputSpec(True, "Off-manifold rollout error visualization."),
    "pca_z": DiagnosticsOutputSpec(True, "PCA projection plot for z embeddings."),
    "pca_h": DiagnosticsOutputSpec(True, "PCA projection plot for h states."),
    "pca_p": DiagnosticsOutputSpec(True, "PCA projection plot for p embeddings."),
    "samples_hard": DiagnosticsOutputSpec(True, "Hard-example grid from the training reservoir."),
    "samples_hard_val": DiagnosticsOutputSpec(True, "Hard-example grid from the validation reservoir."),
    "vis_self_distance_z": DiagnosticsOutputSpec(True, "Self-distance metrics + visuals for z."),
    "vis_self_distance_h": DiagnosticsOutputSpec(True, "Self-distance metrics + visuals for h."),
    "vis_self_distance_p": DiagnosticsOutputSpec(True, "Self-distance metrics + visuals for p."),
    "vis_state_embedding": DiagnosticsOutputSpec(True, "State embedding histogram."),
    "vis_odometry_current_z": DiagnosticsOutputSpec(True, "Odometry cumulative Δz plots."),
    "vis_odometry_current_p": DiagnosticsOutputSpec(True, "Odometry cumulative Δp plots."),
    "vis_odometry_current_h": DiagnosticsOutputSpec(True, "Odometry cumulative Δh plots."),
    "vis_odometry_z_vs_z_hat": DiagnosticsOutputSpec(True, "Odometry z vs z_hat plots."),
    "vis_odometry_p_vs_p_hat": DiagnosticsOutputSpec(True, "Odometry p vs p_hat plots."),
    "vis_odometry_h_vs_h_hat": DiagnosticsOutputSpec(True, "Odometry h vs h_hat plots."),
    "vis_action_field_z": DiagnosticsOutputSpec(True, "Action-conditioned vector field plots for z."),
    "vis_action_field_h": DiagnosticsOutputSpec(True, "Action-conditioned vector field plots for h."),
    "vis_action_field_p": DiagnosticsOutputSpec(True, "Action-conditioned vector field plots for p."),
    "vis_action_time_z": DiagnosticsOutputSpec(True, "Action delta time-slice plots for z."),
    "vis_action_time_h": DiagnosticsOutputSpec(True, "Action delta time-slice plots for h."),
    "vis_action_time_p": DiagnosticsOutputSpec(True, "Action delta time-slice plots for p."),
    "vis_composability_z": DiagnosticsOutputSpec(True, "Composability plots for z."),
    "vis_composability_h": DiagnosticsOutputSpec(True, "Composability plots for h."),
    "vis_composability_p": DiagnosticsOutputSpec(True, "Composability plots for p."),
    "vis_delta_z_pca": DiagnosticsOutputSpec(True, "Motion PCA artifacts for z."),
    "vis_delta_h_pca": DiagnosticsOutputSpec(True, "Motion PCA artifacts for h."),
    "vis_delta_p_pca": DiagnosticsOutputSpec(True, "Motion PCA artifacts for p."),
    "vis_action_alignment_detail_z": DiagnosticsOutputSpec(True, "Action-alignment detail plot for z (PCA)."),
    "vis_action_alignment_detail_raw_z": DiagnosticsOutputSpec(True, "Action-alignment detail plot for z (raw)."),
    "vis_action_alignment_detail_centered_z": DiagnosticsOutputSpec(True, "Action-alignment detail plot for z (centered)."),
    "vis_action_alignment_detail_h": DiagnosticsOutputSpec(True, "Action-alignment detail plot for h (PCA)."),
    "vis_action_alignment_detail_raw_h": DiagnosticsOutputSpec(True, "Action-alignment detail plot for h (raw)."),
    "vis_action_alignment_detail_centered_h": DiagnosticsOutputSpec(True, "Action-alignment detail plot for h (centered)."),
    "vis_action_alignment_detail_p": DiagnosticsOutputSpec(True, "Action-alignment detail plot for p (PCA)."),
    "vis_action_alignment_detail_raw_p": DiagnosticsOutputSpec(True, "Action-alignment detail plot for p (raw)."),
    "vis_action_alignment_detail_centered_p": DiagnosticsOutputSpec(True, "Action-alignment detail plot for p (centered)."),
    "vis_cycle_error_z": DiagnosticsOutputSpec(False, "Cycle-error artifacts for z."),
    "vis_cycle_error_h": DiagnosticsOutputSpec(False, "Cycle-error artifacts for h."),
    "vis_cycle_error_p": DiagnosticsOutputSpec(False, "Cycle-error artifacts for p."),
    "vis_straightline_p": DiagnosticsOutputSpec(True, "Straight-line rollout plot in p."),
    "vis_rollout_divergence_z": DiagnosticsOutputSpec(True, "Rollout divergence plots/CSVs for z."),
    "vis_rollout_divergence_h": DiagnosticsOutputSpec(True, "Rollout divergence plots/CSVs for h."),
    "vis_rollout_divergence_p": DiagnosticsOutputSpec(True, "Rollout divergence plots/CSVs for p."),
    "vis_rollout_divergence_excess_z": DiagnosticsOutputSpec(True, "Rollout divergence excess plots/CSVs for z."),
    "vis_rollout_divergence_excess_h": DiagnosticsOutputSpec(True, "Rollout divergence excess plots/CSVs for h."),
    "vis_rollout_divergence_excess_p": DiagnosticsOutputSpec(True, "Rollout divergence excess plots/CSVs for p."),
    "vis_h_ablation": DiagnosticsOutputSpec(True, "H-ablation divergence plots/CSVs."),
    "vis_z_consistency": DiagnosticsOutputSpec(True, "Z consistency plots/CSVs."),
    "vis_z_monotonicity": DiagnosticsOutputSpec(True, "Z monotonicity plots/CSVs."),
    "vis_path_independence": DiagnosticsOutputSpec(True, "Path-independence plots/CSVs."),
    "vis_h_drift_by_action": DiagnosticsOutputSpec(True, "H drift-by-action plot/CSV."),
    "vis_norm_timeseries": DiagnosticsOutputSpec(True, "Norm timeseries plot."),
    "diagnostics_frames": DiagnosticsOutputSpec(False, "Diagnostics frame dumps + CSV."),
    "diagnostics_scalars": DiagnosticsOutputSpec(True, "Diagnostics scalar CSV summary."),
    "vis_ctrl_smoothness_z": DiagnosticsOutputSpec(False, "Vis-ctrl smoothness spectrum for z."),
    "vis_ctrl_smoothness_h": DiagnosticsOutputSpec(False, "Vis-ctrl smoothness spectrum for h."),
    "vis_ctrl_smoothness_p": DiagnosticsOutputSpec(False, "Vis-ctrl smoothness spectrum for p."),
    "vis_ctrl_composition_z": DiagnosticsOutputSpec(False, "Vis-ctrl two-step composition error for z."),
    "vis_ctrl_composition_h": DiagnosticsOutputSpec(False, "Vis-ctrl two-step composition error for h."),
    "vis_ctrl_composition_p": DiagnosticsOutputSpec(False, "Vis-ctrl two-step composition error for p."),
    "vis_ctrl_stability_z": DiagnosticsOutputSpec(False, "Vis-ctrl neighborhood stability plot for z."),
    "vis_ctrl_stability_h": DiagnosticsOutputSpec(False, "Vis-ctrl neighborhood stability plot for h."),
    "vis_ctrl_stability_p": DiagnosticsOutputSpec(False, "Vis-ctrl neighborhood stability plot for p."),
    "vis_ctrl_summary": DiagnosticsOutputSpec(False, "Vis-ctrl metrics CSV summary."),
    "vis_planning_action_stats": DiagnosticsOutputSpec(True, "Planning action delta stats plot (P)."),
    "vis_planning_action_stats_strip": DiagnosticsOutputSpec(True, "Planning action delta strip plot (P)."),
    "vis_planning_pca_test1": DiagnosticsOutputSpec(True, "Planning PCA path visualization (test1)."),
    "vis_planning_pca_test2": DiagnosticsOutputSpec(True, "Planning PCA path visualization (test2)."),
    "vis_planning_exec_test1": DiagnosticsOutputSpec(True, "Planning execution trace (test1)."),
    "vis_planning_exec_test2": DiagnosticsOutputSpec(True, "Planning execution trace (test2)."),
    "vis_planning_reachable_h": DiagnosticsOutputSpec(True, "Planning reachable fraction histogram (H)."),
    "vis_planning_reachable_p": DiagnosticsOutputSpec(True, "Planning reachable fraction histogram (P)."),
    "vis_planning_graph_h": DiagnosticsOutputSpec(True, "Planning graph visualization (H)."),
    "vis_planning_graph_p": DiagnosticsOutputSpec(True, "Planning graph visualization (P)."),
    "graph_rank_cdf_z": DiagnosticsOutputSpec(False, "Graph diagnostics rank CDF plots for z."),
    "graph_rank_cdf_h": DiagnosticsOutputSpec(False, "Graph diagnostics rank CDF plots for h."),
    "graph_rank_cdf_p": DiagnosticsOutputSpec(False, "Graph diagnostics rank CDF plots for p."),
    "graph_neff_violin_z": DiagnosticsOutputSpec(False, "Graph diagnostics Neff violin plot for z."),
    "graph_neff_violin_h": DiagnosticsOutputSpec(False, "Graph diagnostics Neff violin plot for h."),
    "graph_neff_violin_p": DiagnosticsOutputSpec(False, "Graph diagnostics Neff violin plot for p."),
    "graph_in_degree_z": DiagnosticsOutputSpec(False, "Graph diagnostics in-degree histogram for z."),
    "graph_in_degree_h": DiagnosticsOutputSpec(False, "Graph diagnostics in-degree histogram for h."),
    "graph_in_degree_p": DiagnosticsOutputSpec(False, "Graph diagnostics in-degree histogram for p."),
    "graph_edge_consistency_z": DiagnosticsOutputSpec(False, "Graph diagnostics edge consistency histogram for z."),
    "graph_edge_consistency_h": DiagnosticsOutputSpec(False, "Graph diagnostics edge consistency histogram for h."),
    "graph_edge_consistency_p": DiagnosticsOutputSpec(False, "Graph diagnostics edge consistency histogram for p."),
    "graph_history_z": DiagnosticsOutputSpec(False, "Graph diagnostics history CSV for z."),
    "graph_history_h": DiagnosticsOutputSpec(False, "Graph diagnostics history CSV for h."),
    "graph_history_p": DiagnosticsOutputSpec(False, "Graph diagnostics history CSV for p."),
}

DIAGNOSTICS_OUTPUT_GROUPS = {
    "rollouts": ("vis_fixed_0", "vis_fixed_1", "vis_rolling_0", "vis_rolling_1"),
    "pca": ("pca_z", "pca_h", "pca_p"),
    "hard_examples": ("samples_hard", "samples_hard_val"),
    "self_distance": (
        "vis_self_distance_z",
        "vis_self_distance_h",
        "vis_self_distance_p",
        "vis_state_embedding",
        "vis_odometry_current_z",
        "vis_odometry_current_p",
        "vis_odometry_current_h",
        "vis_odometry_z_vs_z_hat",
        "vis_odometry_p_vs_p_hat",
        "vis_odometry_h_vs_h_hat",
    ),
    "diagnostics": tuple(
        key
        for key in DIAGNOSTICS_OUTPUT_CATALOG.keys()
        if key.startswith("vis_")
        and not key.startswith("vis_ctrl_")
        and key
        not in (
            "vis_fixed_0",
            "vis_fixed_1",
            "vis_rolling_0",
            "vis_rolling_1",
            "vis_off_manifold",
            "vis_self_distance_z",
            "vis_self_distance_h",
            "vis_self_distance_p",
            "vis_state_embedding",
            "vis_odometry_current_z",
            "vis_odometry_current_p",
            "vis_odometry_current_h",
            "vis_odometry_z_vs_z_hat",
            "vis_odometry_p_vs_p_hat",
            "vis_odometry_h_vs_h_hat",
            "vis_planning_action_stats",
            "vis_planning_action_stats_strip",
            "vis_planning_pca_test1",
            "vis_planning_pca_test2",
            "vis_planning_exec_test1",
            "vis_planning_exec_test2",
            "vis_planning_reachable_h",
            "vis_planning_reachable_p",
            "vis_planning_graph_h",
            "vis_planning_graph_p",
        )
    ),
    "planning": (
        "vis_planning_action_stats",
        "vis_planning_action_stats_strip",
        "vis_planning_pca_test1",
        "vis_planning_pca_test2",
        "vis_planning_exec_test1",
        "vis_planning_exec_test2",
        "vis_planning_reachable_h",
        "vis_planning_reachable_p",
        "vis_planning_graph_h",
        "vis_planning_graph_p",
    ),
    "vis_ctrl": (
        "vis_ctrl_smoothness_z",
        "vis_ctrl_smoothness_h",
        "vis_ctrl_smoothness_p",
        "vis_ctrl_composition_z",
        "vis_ctrl_composition_h",
        "vis_ctrl_composition_p",
        "vis_ctrl_stability_z",
        "vis_ctrl_stability_h",
        "vis_ctrl_stability_p",
        "vis_ctrl_summary",
    ),
    "graph": tuple(key for key in DIAGNOSTICS_OUTPUT_CATALOG.keys() if key.startswith("graph_")),
}

OUTPUT_KIND_OVERRIDES = {
    "vis_h_ablation": "p",
    "vis_path_independence": "p",
    "vis_state_embedding": "p",
    "vis_h_drift_by_action": "h",
    "vis_odometry_current_z": "z",
    "vis_odometry_current_p": "p",
    "vis_odometry_current_h": "h",
    "vis_odometry_z_vs_z_hat": "z",
    "vis_odometry_p_vs_p_hat": "p",
    "vis_odometry_h_vs_h_hat": "h",
    "vis_delta_z_pca": "z",
    "vis_delta_h_pca": "h",
    "vis_delta_p_pca": "p",
    "vis_z_consistency": "z",
    "vis_z_monotonicity": "z",
    "vis_planning_action_stats": "p",
    "vis_planning_action_stats_strip": "p",
    "vis_planning_pca_test1": "p",
    "vis_planning_pca_test2": "p",
    "vis_planning_exec_test1": "p",
    "vis_planning_exec_test2": "p",
    "vis_planning_reachable_h": "h",
    "vis_planning_reachable_p": "p",
    "vis_planning_graph_h": "h",
    "vis_planning_graph_p": "p",
}


def _output_kind(key: str) -> Optional[str]:
    override = OUTPUT_KIND_OVERRIDES.get(key)
    if override is not None:
        return override
    if key.endswith("_z"):
        return "z"
    if key.endswith("_h"):
        return "h"
    if key.endswith("_p"):
        return "p"
    return None


def _resolve_outputs(
    *,
    weights: Any,
    model: Any,
    hard_example_cfg: Any,
    graph_cfg: Any,
    planning_cfg: Any,
) -> dict[str, DiagnosticsOutputSpec]:
    enabled_kinds = _enabled_kinds(weights, model)
    group_enabled = {
        "hard_examples": getattr(hard_example_cfg, "reservoir", 0) > 0,
        "planning": bool(getattr(planning_cfg, "enabled", True)),
        "vis_ctrl": True,
        "graph": bool(getattr(graph_cfg, "enabled", True)),
    }
    resolved: dict[str, DiagnosticsOutputSpec] = {}
    for key, spec in DIAGNOSTICS_OUTPUT_CATALOG.items():
        enabled = spec.enabled
        for group_name, group_keys in DIAGNOSTICS_OUTPUT_GROUPS.items():
            if key in group_keys and not group_enabled.get(group_name, True):
                enabled = False
                break
        if enabled:
            if key == "vis_ctrl_summary" and (
                not enabled_kinds.get("z", False) or not enabled_kinds.get("h", False)
            ):
                enabled = False
            if not enabled:
                resolved[key] = DiagnosticsOutputSpec(enabled, spec.description)
                continue
            kind = _output_kind(key)
        if kind is not None and not enabled_kinds.get(kind, False):
            enabled = False
        resolved[key] = DiagnosticsOutputSpec(enabled, spec.description)
    return resolved


def planning_outputs_enabled(
    *,
    weights: Any,
    model: Any,
    hard_example_cfg: Any,
    graph_cfg: Any,
    planning_cfg: Any,
) -> bool:
    resolved_outputs = _resolve_outputs(
        weights=weights,
        model=model,
        hard_example_cfg=hard_example_cfg,
        graph_cfg=graph_cfg,
        planning_cfg=planning_cfg,
    )
    return _any_outputs_enabled(resolved_outputs, DIAGNOSTICS_OUTPUT_GROUPS["planning"])


def _frame_to_tensor(frame: np.ndarray, image_size: int) -> torch.Tensor:
    pil = Image.fromarray(frame)
    pil = pil.resize((image_size, image_size), Image.BILINEAR)
    arr = np.array(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _pose_from_frames(
    frames: Sequence[np.ndarray],
    model: Any,
    model_cfg: Any,
    device: torch.device,
    *,
    use_z2h_init: bool,
    action_dim: int,
    force_h_zero: bool = False,
) -> np.ndarray:
    if len(frames) < 2:
        raise AssertionError("Pose extraction requires at least two frames.")
    frames_tensor = torch.stack(
        [_frame_to_tensor(f, model_cfg.image_size) for f in frames],
        dim=0,
    ).unsqueeze(0)
    frames_tensor = frames_tensor.to(device)
    actions_zero = torch.zeros((1, frames_tensor.shape[1] - 1, action_dim), device=device)
    with torch.no_grad():
        embeds = model.encode_sequence(frames_tensor)["embeddings"]
        _, _, h_states = rollout_teacher_forced(
            model,
            embeds,
            actions_zero,
            use_z2h_init=use_z2h_init,
            force_h_zero=force_h_zero,
        )
        _, pose_pred, _ = rollout_pose(
            model,
            h_states,
            actions_zero,
            z_embeddings=embeds,
        )
    return pose_pred[0].detach().cpu().numpy()


def _extract_planning_latents(
    model: Any,
    plan_frames: torch.Tensor,
    plan_actions: torch.Tensor,
    device: torch.device,
    *,
    use_z2h_init: bool,
    force_h_zero: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Optional[str]], np.ndarray, torch.Tensor]:
    assert torch.is_grad_enabled()
    with torch.no_grad():
        plan_frames_device = plan_frames.to(device)
        plan_actions_device = plan_actions.to(device)
        plan_embeddings = model.encode_sequence(plan_frames_device)["embeddings"]
        _, _, plan_h_states = rollout_teacher_forced(
            model,
            plan_embeddings,
            plan_actions_device,
            use_z2h_init=use_z2h_init,
            force_h_zero=force_h_zero,
        )
        _, plan_p_embeddings, _ = rollout_pose(
            model,
            plan_h_states,
            plan_actions_device,
            z_embeddings=plan_embeddings,
        )
    p_t = plan_p_embeddings[:, :-1].detach().cpu().reshape(-1, plan_p_embeddings.shape[-1]).numpy()
    p_tp1 = plan_p_embeddings[:, 1:].detach().cpu().reshape(-1, plan_p_embeddings.shape[-1]).numpy()
    h_t = plan_h_states[:, :-1].detach().cpu().reshape(-1, plan_h_states.shape[-1]).numpy()
    h_tp1 = plan_h_states[:, 1:].detach().cpu().reshape(-1, plan_h_states.shape[-1]).numpy()
    actions_np = plan_actions[:, :-1].detach().cpu().reshape(-1, plan_actions.shape[-1]).numpy()
    action_labels = action_labels_from_vectors(actions_np)
    deltas = p_tp1 - p_t
    return p_t, p_tp1, h_t, h_tp1, actions_np, action_labels, deltas, plan_h_states


def _compute_planning_graphs(
    p_t: np.ndarray,
    p_tp1: np.ndarray,
    h_t: np.ndarray,
    h_tp1: np.ndarray,
    actions_np: np.ndarray,
    action_labels: Sequence[Optional[str]],
    *,
    min_action_count: int,
) -> Tuple[ActionDeltaStats, DatasetGraph, DatasetGraph, float]:
    stats = compute_action_delta_stats(
        p_t,
        p_tp1,
        actions_np,
        min_action_count=min_action_count,
    )
    non_noop = np.array([lbl in DIRECTION_ORDER for lbl in action_labels], dtype=bool)
    if not np.any(non_noop):
        raise AssertionError("Planning diagnostics require non-noop actions for h graph thresholds.")
    h_dot = (h_t * h_tp1).sum(axis=1)
    h_norms = np.maximum(np.linalg.norm(h_t, axis=1) * np.linalg.norm(h_tp1, axis=1), 1e-8)
    h_cos = h_dot / h_norms
    d_nn = float(np.median(1.0 - h_cos[non_noop]))
    tau_h_merge = min(max(3.0 * d_nn, 0.02), 0.08)
    graph_h = build_dataset_graph(
        h_t,
        h_tp1,
        actions_np,
        radius=tau_h_merge,
        metric="cosine",
    )
    graph_p = build_dataset_graph(
        p_t,
        p_tp1,
        actions_np,
        radius=stats.r_cluster_p,
        metric="l2",
    )
    return stats, graph_h, graph_p, tau_h_merge


def _run_h_local_sanity(
    graph_h: DatasetGraph,
    plan_h_states: torch.Tensor,
    tau_h_merge: float,
    cfg: PlanningDiagnosticsConfig,
    rng: random.Random,
) -> bool:
    seq_len = plan_h_states.shape[1]
    if seq_len <= cfg.local_k_min:
        raise AssertionError("Planning diagnostics require seq_len > local_k_min.")
    h_all = plan_h_states.detach().cpu().numpy().reshape(-1, plan_h_states.shape[-1])
    _, h_nodes = cluster_latents(h_all, radius=tau_h_merge, metric="cosine")
    h_nodes = h_nodes.reshape(plan_h_states.shape[0], seq_len)
    b_idx = rng.randrange(plan_h_states.shape[0])
    max_start = seq_len - cfg.local_k_min - 1
    if max_start <= 0:
        raise AssertionError("Planning diagnostics require room for local k-step test.")
    t0 = rng.randrange(max_start)
    k_max = min(cfg.local_k_max, seq_len - 1 - t0)
    k = rng.randint(cfg.local_k_min, k_max)
    h_local_actions = bfs_plan(graph_h, int(h_nodes[b_idx, t0]), int(h_nodes[b_idx, t0 + k]))
    return h_local_actions is not None


def _build_planning_tests(
    env: GridworldKeyEnv,
    cfg: PlanningDiagnosticsConfig,
) -> List[Tuple[str, Tuple[int, int], Tuple[int, int]]]:
    start_loop = (env.grid_rows - 2, 1)
    goal_center = (env.grid_rows // 2, env.grid_cols // 2)
    goal_mid = (
        env.grid_rows // 2,
        min(env.grid_cols - 2, env.grid_cols // 2 + cfg.interior_goal_col_offset),
    )
    return [
        ("test1", start_loop, goal_center),
        ("test2", goal_center, goal_mid),
    ]


def _write_planning_metrics_row(
    metrics_dir: Path,
    stats: ActionDeltaStats,
    graph_h: DatasetGraph,
    graph_p: DatasetGraph,
    h_reach: np.ndarray,
    p_reach: np.ndarray,
    h_local_success: bool,
    test_results: Dict[str, PlanningTestResult],
    *,
    global_step: int,
) -> None:
    def _pct(values: np.ndarray, q: float) -> float:
        if values.size == 0:
            return float("nan")
        return float(np.quantile(values, q))

    planning_metrics_path = metrics_dir / "planning_metrics.csv"
    header = [
        "step",
        "L",
        "inv_lr",
        "inv_ud",
        "noop_ratio",
        "q_L",
        "q_R",
        "q_U",
        "q_D",
        "num_nodes_h",
        "num_nodes_p",
        "reach_h_median",
        "reach_h_p10",
        "reach_h_p90",
        "reach_p_median",
        "reach_p_p10",
        "reach_p_p90",
        "h_local_success",
        "test1_success",
        "test1_steps",
        "test1_final_p_dist",
        "test1_goal_dist",
        "test2_success",
        "test2_steps",
        "test2_final_p_dist",
        "test2_goal_dist",
    ]
    row = [
        global_step,
        stats.L_scale,
        stats.inv_lr,
        stats.inv_ud,
        stats.noop_ratio,
        stats.q.get("L", float("nan")),
        stats.q.get("R", float("nan")),
        stats.q.get("U", float("nan")),
        stats.q.get("D", float("nan")),
        graph_h.centers.shape[0],
        graph_p.centers.shape[0],
        float(np.median(h_reach)) if h_reach.size else float("nan"),
        _pct(h_reach, 0.10),
        _pct(h_reach, 0.90),
        float(np.median(p_reach)) if p_reach.size else float("nan"),
        _pct(p_reach, 0.10),
        _pct(p_reach, 0.90),
        float(h_local_success),
        float(test_results["test1"].success),
        test_results["test1"].steps,
        test_results["test1"].final_p_distance,
        test_results["test1"].goal_distance,
        float(test_results["test2"].success),
        test_results["test2"].steps,
        test_results["test2"].final_p_distance,
        test_results["test2"].goal_distance,
    ]
    append_csv_row(planning_metrics_path, header, row)


def _build_straightline_trajectories(
    *,
    diagnostics_cfg: Any,
    diag_state: Any,
    model: Any,
    diagnostics_generator: torch.Generator,
) -> List[StraightLineTrajectory]:
    if diag_state.p_embeddings is None or not diag_state.has_p:
        raise AssertionError("Straightline diagnostics require p_embeddings.")
    if diag_state.p_embeddings.shape[1] < 2:
        raise AssertionError("Straightline diagnostics require at least two timesteps.")

    max_starts = min(diagnostics_cfg.straightline_starts, diag_state.p_embeddings.shape[0])
    if max_starts <= 0:
        raise AssertionError("Straightline diagnostics require at least one start.")
    if len(diag_state.action_metadata.unique_actions) == 0:
        raise AssertionError("Straightline diagnostics require at least one action ID.")

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

    p_flat = diag_state.p_embeddings.detach().cpu().reshape(-1, diag_state.p_embeddings.shape[-1]).numpy()
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
            trajectories.append(StraightLineTrajectory(points=proj, label=traj_label, color=color))
    return trajectories


def _compute_z_consistency_samples(
    *,
    diagnostics_cfg: Any,
    diag_state: Any,
    model: Any,
    diagnostics_generator: torch.Generator,
    z_consistency_samples: int,
) -> Tuple[List[float], List[float]]:
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
    return distances, cosines


def _compute_path_independence_stats(
    *,
    diagnostics_cfg: Any,
    diag_state: Any,
    model: Any,
    diagnostics_generator: torch.Generator,
) -> Optional[Tuple[str, float, float]]:
    if not diag_state.has_p or diag_state.p_embeddings is None:
        raise AssertionError("Path independence diagnostics require pose embeddings.")
    if diag_state.embeddings.shape[1] < 2:
        raise AssertionError("Path independence diagnostics require at least two timesteps.")
    if len(diag_state.action_metadata.unique_actions) < 2:
        raise AssertionError("Path independence diagnostics require at least two action IDs.")
    if diagnostics_cfg.path_independence_samples <= 0:
        return None

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
    if not z_diffs or not p_diffs:
        return None

    label_a = action_labels.get(a_id, f"action {a_id}")
    label_b = action_labels.get(b_id, f"action {b_id}")
    if use_noop:
        label_b_path = action_labels.get(diag_state.action_metadata.noop_id, "NOOP")
    else:
        label_b_path = f"{label_b}+{label_a}"
    label = f"{label_a}+{label_b} vs {label_b_path}"
    return label, float(np.mean(z_diffs)), float(np.mean(p_diffs))


def _compute_h_drift_stats(
    *,
    diagnostics_cfg: Any,
    diag_state: Any,
) -> List[Tuple[int, int, float, List[float]]]:
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
    return drift_stats[: diagnostics_cfg.h_drift_max_actions]


def _compute_graph_kind_latents(
    *,
    kind: str,
    graph_diag: Any,
    model: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        targets = p_targets
        z_flat = p_targets.reshape(-1, p_targets.shape[-1])
        target_flat = targets.reshape(-1, targets.shape[-1])
        zhat_full = p_hat_full.reshape(-1, p_hat_full.shape[-1])
    elif kind == "h":
        targets = graph_diag.graph_h_states
        z_flat = graph_diag.graph_h_states.reshape(-1, graph_diag.graph_h_states.shape[-1])
        target_flat = targets.reshape(-1, targets.shape[-1])
        zhat_full = torch.cat(
            [graph_diag.graph_h_preds, graph_diag.graph_h_states[:, -1:, :]],
            dim=1,
        ).reshape(-1, graph_diag.graph_h_states.shape[-1])
    else:
        targets = graph_diag.graph_embeddings
        z_flat = graph_diag.graph_embeddings.reshape(-1, graph_diag.graph_embeddings.shape[-1])
        target_flat = targets.reshape(-1, targets.shape[-1])
        zhat_full = torch.cat(
            [graph_diag.graph_preds, graph_diag.graph_embeddings[:, -1:, :]],
            dim=1,
        ).reshape(-1, graph_diag.graph_embeddings.shape[-1])
    return z_flat, target_flat, zhat_full


def _any_outputs_enabled(resolved_outputs: dict[str, DiagnosticsOutputSpec], keys: Tuple[str, ...]) -> bool:
    return any(resolved_outputs[key].enabled for key in keys)


def _enabled_kinds(weights, model) -> dict[str, bool]:
    z_enabled = any(
        getattr(weights, name, 0.0) > 0
        for name in (
            "jepa",
            "inverse_dynamics_z",
            "recon",
            "recon_patch",
            "recon_multi_gauss",
            "recon_multi_box",
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
            "jepa",
            "jepa_open_loop",
            "h2z",
            "z2h",
            "z2h_init_zero",
            "h2z_delta",
            "inverse_dynamics_h",
            "action_delta_h",
            "additivity_h",
            "rollout_kstep_h",
            "rollout_kstep_delta_h",
            "rollout_recon_h",
            "rollout_recon_multi_box_h",
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
    planning_cfg: Any,
    model: Any,
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
    render_mode: str,
    force_h_zero: bool = False,
) -> None:
    resolved_outputs = _resolve_outputs(
        weights=weights,
        model=model,
        hard_example_cfg=hard_example_cfg,
        graph_cfg=graph_cfg,
        planning_cfg=planning_cfg,
    )
    if (
        _any_outputs_enabled(resolved_outputs, DIAGNOSTICS_OUTPUT_GROUPS["diagnostics"])
        and diagnostics_batch_cpu is None
    ):
        raise AssertionError("Diagnostics requested but diagnostics_batch_cpu is missing.")
    if _any_outputs_enabled(resolved_outputs, DIAGNOSTICS_OUTPUT_GROUPS["vis_ctrl"]) and vis_ctrl_batch_cpu is None:
        raise AssertionError("Vis-ctrl requested but vis_ctrl_batch_cpu is missing.")
    if (
        graph_cfg.enabled
        and _any_outputs_enabled(resolved_outputs, DIAGNOSTICS_OUTPUT_GROUPS["graph"])
        and graph_diag_batch_cpu is None
    ):
        raise AssertionError("Graph diagnostics requested but graph_diag_batch_cpu is missing.")
    if resolved_outputs["vis_off_manifold"].enabled and off_manifold_batch_cpu is None:
        raise AssertionError("Off-manifold outputs requested but off_manifold_batch_cpu is missing.")

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
    diagnostics_delta_z_dir = run_dir / "vis_delta_z_pca"
    diagnostics_delta_p_dir = run_dir / "vis_delta_p_pca"
    diagnostics_delta_h_dir = run_dir / "vis_delta_h_pca"
    diagnostics_alignment_z_dir = run_dir / "vis_action_alignment_z"
    diagnostics_alignment_z_raw_dir = run_dir / "vis_action_alignment_z_raw"
    diagnostics_alignment_z_centered_dir = run_dir / "vis_action_alignment_z_centered"
    diagnostics_alignment_p_dir = run_dir / "vis_action_alignment_p"
    diagnostics_alignment_p_raw_dir = run_dir / "vis_action_alignment_p_raw"
    diagnostics_alignment_p_centered_dir = run_dir / "vis_action_alignment_p_centered"
    diagnostics_alignment_h_dir = run_dir / "vis_action_alignment_h"
    diagnostics_alignment_h_raw_dir = run_dir / "vis_action_alignment_h_raw"
    diagnostics_alignment_h_centered_dir = run_dir / "vis_action_alignment_h_centered"
    diagnostics_cycle_z_dir = run_dir / "vis_cycle_error_z"
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

    alignment_detail_z_enabled = any(
        resolved_outputs[key].enabled
        for key in (
            "vis_action_alignment_detail_z",
            "vis_action_alignment_detail_raw_z",
            "vis_action_alignment_detail_centered_z",
        )
    )
    alignment_detail_h_enabled = any(
        resolved_outputs[key].enabled
        for key in (
            "vis_action_alignment_detail_h",
            "vis_action_alignment_detail_raw_h",
            "vis_action_alignment_detail_centered_h",
        )
    )
    alignment_detail_p_enabled = any(
        resolved_outputs[key].enabled
        for key in (
            "vis_action_alignment_detail_p",
            "vis_action_alignment_detail_raw_p",
            "vis_action_alignment_detail_centered_p",
        )
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["vis_fixed_0"].enabled or resolved_outputs["vis_fixed_1"].enabled:
        fixed_vis_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["vis_rolling_0"].enabled or resolved_outputs["vis_rolling_1"].enabled:
        rolling_vis_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["pca_z"].enabled:
        pca_z_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["pca_p"].enabled:
        pca_p_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["pca_h"].enabled:
        pca_h_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["vis_off_manifold"].enabled:
        vis_off_manifold_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["samples_hard"].enabled:
        samples_hard_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["samples_hard_val"].enabled:
        samples_hard_val_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["vis_self_distance_z"].enabled:
        vis_self_distance_z_dir.mkdir(parents=True, exist_ok=True)
        self_distance_z_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["vis_self_distance_p"].enabled:
        vis_self_distance_p_dir.mkdir(parents=True, exist_ok=True)
        self_distance_p_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["vis_self_distance_h"].enabled:
        vis_self_distance_h_dir.mkdir(parents=True, exist_ok=True)
        self_distance_h_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["vis_state_embedding"].enabled:
        vis_state_embedding_dir.mkdir(parents=True, exist_ok=True)
    if (
        resolved_outputs["vis_odometry_current_z"].enabled
        or resolved_outputs["vis_odometry_current_p"].enabled
        or resolved_outputs["vis_odometry_current_h"].enabled
        or resolved_outputs["vis_odometry_z_vs_z_hat"].enabled
        or resolved_outputs["vis_odometry_p_vs_p_hat"].enabled
        or resolved_outputs["vis_odometry_h_vs_h_hat"].enabled
    ):
        vis_odometry_dir.mkdir(parents=True, exist_ok=True)
    if _any_outputs_enabled(resolved_outputs, DIAGNOSTICS_OUTPUT_GROUPS["diagnostics"]):
        if resolved_outputs["vis_delta_z_pca"].enabled:
            diagnostics_delta_z_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_delta_p_pca"].enabled:
            diagnostics_delta_p_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_delta_h_pca"].enabled:
            diagnostics_delta_h_dir.mkdir(parents=True, exist_ok=True)
        if alignment_detail_z_enabled:
            diagnostics_alignment_z_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_action_alignment_detail_raw_z"].enabled:
            diagnostics_alignment_z_raw_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_action_alignment_detail_centered_z"].enabled:
            diagnostics_alignment_z_centered_dir.mkdir(parents=True, exist_ok=True)
        if alignment_detail_p_enabled:
            diagnostics_alignment_p_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_action_alignment_detail_raw_p"].enabled:
            diagnostics_alignment_p_raw_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_action_alignment_detail_centered_p"].enabled:
            diagnostics_alignment_p_centered_dir.mkdir(parents=True, exist_ok=True)
        if alignment_detail_h_enabled:
            diagnostics_alignment_h_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_action_alignment_detail_raw_h"].enabled:
            diagnostics_alignment_h_raw_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_action_alignment_detail_centered_h"].enabled:
            diagnostics_alignment_h_centered_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_cycle_error_z"].enabled:
            diagnostics_cycle_z_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_cycle_error_p"].enabled:
            diagnostics_cycle_p_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_cycle_error_h"].enabled:
            diagnostics_cycle_h_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_frames"].enabled:
            diagnostics_frames_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_composability_z"].enabled:
            vis_composability_z_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_composability_p"].enabled:
            vis_composability_p_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_composability_h"].enabled:
            vis_composability_h_dir.mkdir(parents=True, exist_ok=True)
        if (
            resolved_outputs["vis_rollout_divergence_z"].enabled
            or resolved_outputs["vis_rollout_divergence_excess_z"].enabled
        ):
            diagnostics_rollout_divergence_z_dir.mkdir(parents=True, exist_ok=True)
        if (
            resolved_outputs["vis_rollout_divergence_h"].enabled
            or resolved_outputs["vis_rollout_divergence_excess_h"].enabled
        ):
            diagnostics_rollout_divergence_h_dir.mkdir(parents=True, exist_ok=True)
        if (
            resolved_outputs["vis_rollout_divergence_p"].enabled
            or resolved_outputs["vis_rollout_divergence_excess_p"].enabled
        ):
            diagnostics_rollout_divergence_p_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_action_field_z"].enabled:
            diagnostics_action_field_z_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_action_field_h"].enabled:
            diagnostics_action_field_h_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_action_field_p"].enabled:
            diagnostics_action_field_p_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_action_time_z"].enabled:
            diagnostics_action_time_z_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_action_time_h"].enabled:
            diagnostics_action_time_h_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_action_time_p"].enabled:
            diagnostics_action_time_p_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_straightline_p"].enabled:
            diagnostics_straightline_p_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_z_consistency"].enabled:
            diagnostics_z_consistency_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_z_monotonicity"].enabled:
            diagnostics_z_monotonicity_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_path_independence"].enabled:
            diagnostics_path_independence_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_h_ablation"].enabled:
            diagnostics_h_ablation_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_h_drift_by_action"].enabled:
            diagnostics_h_drift_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["vis_norm_timeseries"].enabled:
            diagnostics_norm_timeseries_dir.mkdir(parents=True, exist_ok=True)
    if graph_cfg.enabled:
        if _any_outputs_enabled(resolved_outputs, ("graph_rank_cdf_z", "graph_neff_violin_z", "graph_in_degree_z", "graph_edge_consistency_z", "graph_history_z")):
            graph_diagnostics_dir.mkdir(parents=True, exist_ok=True)
        if _any_outputs_enabled(resolved_outputs, ("graph_rank_cdf_p", "graph_neff_violin_p", "graph_in_degree_p", "graph_edge_consistency_p", "graph_history_p")):
            graph_diagnostics_p_dir.mkdir(parents=True, exist_ok=True)
        if _any_outputs_enabled(resolved_outputs, ("graph_rank_cdf_h", "graph_neff_violin_h", "graph_in_degree_h", "graph_edge_consistency_h", "graph_history_h")):
            graph_diagnostics_h_dir.mkdir(parents=True, exist_ok=True)
    if _any_outputs_enabled(resolved_outputs, DIAGNOSTICS_OUTPUT_GROUPS["vis_ctrl"]):
        vis_ctrl_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    fixed_indices = []
    if resolved_outputs["vis_fixed_0"].enabled:
        fixed_indices.append(0)
    if resolved_outputs["vis_fixed_1"].enabled:
        fixed_indices.append(1)
    if fixed_indices:
        sequences, grad_label = build_visualization_sequences(
            batch_cpu=fixed_batch_cpu,
            selection=fixed_selection,
            model=model,
            decoder=decoder,
            device=device,
            vis_cfg=vis_cfg,
            vis_selection_generator=vis_selection_generator,
            use_z2h_init=should_use_z2h_init(weights),
            render_mode=render_mode,
            force_h_zero=force_h_zero,
        )
        save_rollout_sequence_batch(
            fixed_vis_dir,
            sequences,
            grad_label,
            global_step,
            include_pixel_delta=(weights.pixel_delta > 0 or weights.pixel_delta_multi_box > 0),
            indices=fixed_indices,
        )

    rolling_indices = []
    if resolved_outputs["vis_rolling_0"].enabled:
        rolling_indices.append(0)
    if resolved_outputs["vis_rolling_1"].enabled:
        rolling_indices.append(1)
    if rolling_indices:
        sequences, grad_label = build_visualization_sequences(
            batch_cpu=rolling_batch_cpu,
            selection=None,
            model=model,
            decoder=decoder,
            device=device,
            vis_cfg=vis_cfg,
            vis_selection_generator=vis_selection_generator,
            use_z2h_init=should_use_z2h_init(weights),
            render_mode=render_mode,
            force_h_zero=force_h_zero,
        )
        save_rollout_sequence_batch(
            rolling_vis_dir,
            sequences,
            grad_label,
            global_step,
            include_pixel_delta=(weights.pixel_delta > 0 or weights.pixel_delta_multi_box > 0),
            indices=rolling_indices,
        )

    if resolved_outputs["vis_off_manifold"].enabled and off_manifold_batch_cpu is not None:
        with torch.no_grad():
            step_indices, errors = compute_off_manifold_errors(
                model=model,
                decoder=decoder,
                batch_cpu=off_manifold_batch_cpu,
                device=device,
                rollout_steps=off_manifold_steps,
            )
        save_off_manifold_visualization(
            vis_off_manifold_dir,
            step_indices,
            errors,
            global_step,
        )

    if _any_outputs_enabled(resolved_outputs, DIAGNOSTICS_OUTPUT_GROUPS["pca"]):
        with torch.no_grad():
            embed_frames = embedding_batch_cpu[0].to(device)
            embed_actions = embedding_batch_cpu[1].to(device)
            embed_outputs = model.encode_sequence(embed_frames)
            _, _, h_states = rollout_teacher_forced(
                model,
                embed_outputs["embeddings"],
                embed_actions,
                use_z2h_init=should_use_z2h_init(weights),
            )

        if resolved_outputs["pca_z"].enabled:
            save_embedding_projection(
                embed_outputs["embeddings"],
                pca_z_dir / f"pca_z_{global_step:07d}.png",
                "PCA z",
            )
        if resolved_outputs["pca_h"].enabled:
            save_embedding_projection(
                h_states,
                pca_h_dir / f"pca_h_{global_step:07d}.png",
                "PCA h",
            )
        if resolved_outputs["pca_p"].enabled:
            _, p_embeddings, _ = rollout_pose(
                model,
                h_states,
                embed_actions,
                z_embeddings=embed_outputs["embeddings"],
            )
            save_embedding_projection(
                p_embeddings,
                pca_p_dir / f"pca_p_{global_step:07d}.png",
                "PCA p",
            )

    if resolved_outputs["samples_hard"].enabled:
        hard_samples = hard_reservoir.topk(hard_example_cfg.vis_rows * hard_example_cfg.vis_columns)
        save_hard_example_grid(
            samples_hard_dir / f"hard_{global_step:07d}.png",
            hard_samples,
            hard_example_cfg.vis_columns,
            hard_example_cfg.vis_rows,
            dataset.image_hw,
        )
    if resolved_outputs["samples_hard_val"].enabled:
        hard_samples_val = hard_reservoir_val.topk(hard_example_cfg.vis_rows * hard_example_cfg.vis_columns)
        save_hard_example_grid(
            samples_hard_val_dir / f"hard_{global_step:07d}.png",
            hard_samples_val,
            hard_example_cfg.vis_columns,
            hard_example_cfg.vis_rows,
            dataset.image_hw,
        )

    if _any_outputs_enabled(resolved_outputs, DIAGNOSTICS_OUTPUT_GROUPS["self_distance"]):
        with torch.no_grad():
            self_dist_frames = self_distance_inputs.frames.to(device)
            self_dist_actions = torch.from_numpy(self_distance_inputs.actions).unsqueeze(0).to(device)
            self_dist_embeddings_full = model.encode_sequence(self_dist_frames)["embeddings"]
            self_dist_embeddings = self_dist_embeddings_full[0]
            _, _, self_dist_h_states_batch = rollout_teacher_forced(
                model,
                self_dist_embeddings_full,
                self_dist_actions,
                use_z2h_init=should_use_z2h_init(weights),
            )
            self_dist_h_states = self_dist_h_states_batch[0]

        if resolved_outputs["vis_self_distance_z"].enabled:
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
        if resolved_outputs["vis_self_distance_h"].enabled:
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
        if resolved_outputs["vis_self_distance_p"].enabled:
            _, self_dist_p_batch, _ = rollout_pose(
                model,
                self_dist_h_states_batch,
                self_dist_actions,
                z_embeddings=self_dist_embeddings_full,
            )
            self_dist_p = self_dist_p_batch[0]

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
        if (
            resolved_outputs["vis_state_embedding"].enabled
            or resolved_outputs["vis_odometry_current_z"].enabled
            or resolved_outputs["vis_odometry_current_p"].enabled
            or resolved_outputs["vis_odometry_current_h"].enabled
            or resolved_outputs["vis_odometry_z_vs_z_hat"].enabled
            or resolved_outputs["vis_odometry_p_vs_p_hat"].enabled
            or resolved_outputs["vis_odometry_h_vs_h_hat"].enabled
        ):
            if model.p_action_delta_projector is None:
                raise AssertionError("State embedding outputs require p_action_delta_projector.")
            write_state_embedding_outputs(
                model,
                self_distance_inputs,
                device,
                self_distance_p_dir,
                vis_self_distance_p_dir,
                vis_state_embedding_dir,
                vis_odometry_dir,
                global_step,
                use_z2h_init=should_use_z2h_init(weights),
                force_h_zero=force_h_zero,
                hist_frames_cpu=rolling_batch_cpu[0],
                hist_actions_cpu=rolling_batch_cpu[1],
                write_self_distance=False,
                write_state_hist=resolved_outputs["vis_state_embedding"].enabled,
                write_odometry_current_z=resolved_outputs["vis_odometry_current_z"].enabled,
                write_odometry_current_p=resolved_outputs["vis_odometry_current_p"].enabled,
                write_odometry_current_h=resolved_outputs["vis_odometry_current_h"].enabled,
                write_odometry_z_vs_z_hat=resolved_outputs["vis_odometry_z_vs_z_hat"].enabled,
                write_odometry_p_vs_p_hat=resolved_outputs["vis_odometry_p_vs_p_hat"].enabled,
                write_odometry_h_vs_h_hat=resolved_outputs["vis_odometry_h_vs_h_hat"].enabled,
            )

    if _any_outputs_enabled(resolved_outputs, DIAGNOSTICS_OUTPUT_GROUPS["diagnostics"]):
        diag_state = prepare_diagnostics_batch_state(
            model=model,
            diagnostics_cfg=diagnostics_cfg,
            weights=weights,
            diagnostics_batch_cpu=diagnostics_batch_cpu,
            device=device,
            force_h_zero=force_h_zero,
        )

        action_ids_flat = diag_state.action_metadata.action_ids_flat
        if (
            resolved_outputs["vis_action_field_z"].enabled
            or resolved_outputs["vis_action_time_z"].enabled
        ):
            z_embed_np = diag_state.embeddings.detach().cpu().numpy()
            z_deltas = z_embed_np[:, 1:] - z_embed_np[:, :-1]
            z_action_dim = diag_state.motion_z.action_dim

            if resolved_outputs["vis_action_field_z"].enabled:
                save_action_vector_field_plot(
                    diagnostics_action_field_z_dir / f"action_field_z_{global_step:07d}.png",
                    z_embed_np[:, :-1].reshape(-1, z_embed_np.shape[-1]),
                    z_deltas.reshape(-1, z_deltas.shape[-1]),
                    action_ids_flat,
                    z_action_dim,
                    max_actions=diagnostics_cfg.max_actions_to_plot,
                    min_count=diagnostics_cfg.min_action_count,
                    title="Action-conditioned vector field (Z)",
                )
            if resolved_outputs["vis_action_time_z"].enabled:
                save_action_time_slice_plot(
                    diagnostics_action_time_z_dir / f"action_time_z_{global_step:07d}.png",
                    z_deltas,
                    diag_state.action_ids_seq[:, :-1],
                    z_action_dim,
                    max_actions=diagnostics_cfg.max_actions_to_plot,
                    min_count=diagnostics_cfg.min_action_count,
                    title="Action delta time slices (Z)",
                )

        if (
            resolved_outputs["vis_action_field_h"].enabled
            or resolved_outputs["vis_action_time_h"].enabled
        ):
            h_embed_np = diag_state.h_states.detach().cpu().numpy()
            h_deltas = h_embed_np[:, 1:] - h_embed_np[:, :-1]
            h_action_dim = diag_state.motion_h.action_dim

            if resolved_outputs["vis_action_field_h"].enabled:
                save_action_vector_field_plot(
                    diagnostics_action_field_h_dir / f"action_field_h_{global_step:07d}.png",
                    h_embed_np[:, :-1].reshape(-1, h_embed_np.shape[-1]),
                    h_deltas.reshape(-1, h_deltas.shape[-1]),
                    action_ids_flat,
                    h_action_dim,
                    max_actions=diagnostics_cfg.max_actions_to_plot,
                    min_count=diagnostics_cfg.min_action_count,
                    title="Action-conditioned vector field (H)",
                )
            if resolved_outputs["vis_action_time_h"].enabled:
                save_action_time_slice_plot(
                    diagnostics_action_time_h_dir / f"action_time_h_{global_step:07d}.png",
                    h_deltas,
                    diag_state.action_ids_seq[:, :-1],
                    h_action_dim,
                    max_actions=diagnostics_cfg.max_actions_to_plot,
                    min_count=diagnostics_cfg.min_action_count,
                    title="Action delta time slices (H)",
                )
        if (
            resolved_outputs["vis_action_field_p"].enabled
            or resolved_outputs["vis_action_time_p"].enabled
        ):
            p_embed_np = diag_state.p_embeddings.detach().cpu().numpy()
            p_deltas = p_embed_np[:, 1:] - p_embed_np[:, :-1]
            p_action_dim = diag_state.motion_p.action_dim

            if resolved_outputs["vis_action_field_p"].enabled:
                save_action_vector_field_plot(
                    diagnostics_action_field_p_dir / f"action_field_p_{global_step:07d}.png",
                    p_embed_np[:, :-1].reshape(-1, p_embed_np.shape[-1]),
                    p_deltas.reshape(-1, p_deltas.shape[-1]),
                    action_ids_flat,
                    p_action_dim,
                    max_actions=diagnostics_cfg.max_actions_to_plot,
                    min_count=diagnostics_cfg.min_action_count,
                    title="Action-conditioned vector field (P)",
                )
            if resolved_outputs["vis_action_time_p"].enabled:
                save_action_time_slice_plot(
                    diagnostics_action_time_p_dir / f"action_time_p_{global_step:07d}.png",
                    p_deltas,
                    diag_state.action_ids_seq[:, :-1],
                    p_action_dim,
                    max_actions=diagnostics_cfg.max_actions_to_plot,
                    min_count=diagnostics_cfg.min_action_count,
                    title="Action delta time slices (P)",
                )

        if diag_state.composability is not None:
            if resolved_outputs[f"vis_composability_z"].enabled:
                save_composability_plot(
                    vis_composability_z_dir / f"composability_z_{global_step:07d}.png",
                    diag_state.composability["z"],
                    "z",
                )
            if resolved_outputs[f"vis_composability_h"].enabled:
                save_composability_plot(
                    vis_composability_h_dir / f"composability_h_{global_step:07d}.png",
                    diag_state.composability["h"],
                    "h",
                )
            if resolved_outputs[f"vis_composability_p"].enabled:
                save_composability_plot(
                    vis_composability_p_dir / f"composability_p_{global_step:07d}.png",
                    diag_state.p_series,
                    "p",
                )

        for kind in ["z", "h", "p"]:
            motion_pca_enabled = resolved_outputs[f"vis_delta_{kind}_pca"].enabled
            alignment_enabled = any(
                resolved_outputs[key].enabled
                for key in (
                    f"vis_action_alignment_detail_{kind}",
                    f"vis_action_alignment_detail_raw_{kind}",
                    f"vis_action_alignment_detail_centered_{kind}",
                )
            )
            cycle_enabled = resolved_outputs[f"vis_cycle_error_{kind}"].enabled

            if not (motion_pca_enabled or alignment_enabled or cycle_enabled):
                continue

            if kind == "z":
                motion = diag_state.motion_z
                delta_dir = diagnostics_delta_z_dir
                align_dir = diagnostics_alignment_z_dir
                align_raw_dir = diagnostics_alignment_z_raw_dir
                align_center_dir = diagnostics_alignment_z_centered_dir
                cycle_dir = diagnostics_cycle_z_dir
            elif kind == "h":
                motion = diag_state.motion_h
                delta_dir = diagnostics_delta_h_dir
                align_dir = diagnostics_alignment_h_dir
                align_raw_dir = diagnostics_alignment_h_raw_dir
                align_center_dir = diagnostics_alignment_h_centered_dir
                cycle_dir = diagnostics_cycle_h_dir
            else:
                motion = diag_state.motion_p
                delta_dir = diagnostics_delta_p_dir
                align_dir = diagnostics_alignment_p_dir
                align_raw_dir = diagnostics_alignment_p_raw_dir
                align_center_dir = diagnostics_alignment_p_centered_dir
                cycle_dir = diagnostics_cycle_p_dir

            if motion_pca_enabled:
                write_motion_pca_artifacts(
                    diagnostics_cfg=diagnostics_cfg,
                    global_step=global_step,
                    name=kind,
                    motion=motion,
                    delta_dir=delta_dir,
                )
            if alignment_enabled:
                write_alignment_artifacts(
                    diagnostics_cfg=diagnostics_cfg,
                    global_step=global_step,
                    name=kind,
                    motion=motion,
                    inverse_map=diag_state.inverse_map,
                    alignment_dir=align_dir,
                    alignment_raw_dir=align_raw_dir,
                    alignment_centered_dir=align_center_dir,
                    write_pca=resolved_outputs[f"vis_action_alignment_detail_{kind}"].enabled,
                    write_raw=resolved_outputs[f"vis_action_alignment_detail_raw_{kind}"].enabled,
                    write_centered=resolved_outputs[f"vis_action_alignment_detail_centered_{kind}"].enabled,
                )
            if cycle_enabled:
                write_cycle_error_artifacts(
                    diagnostics_cfg=diagnostics_cfg,
                    global_step=global_step,
                    name=kind,
                    motion=motion,
                    inverse_map=diag_state.inverse_map,
                    cycle_dir=cycle_dir,
                )

        if resolved_outputs["diagnostics_frames"].enabled:
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

        scalars_enabled = resolved_outputs["diagnostics_scalars"].enabled
        norm_timeseries_enabled = resolved_outputs["vis_norm_timeseries"].enabled
        if norm_timeseries_enabled and not scalars_enabled:
            raise AssertionError("Norm timeseries requires diagnostics_scalars to be enabled.")
        if scalars_enabled or norm_timeseries_enabled:
            diagnostics_scalars_path = metrics_dir / "diagnostics_scalars.csv"
            z_norm_mean, z_norm_p95 = compute_norm_stats(diag_state.embeddings)
            h_norm_mean, h_norm_p95 = compute_norm_stats(diag_state.h_states)
            if diag_state.has_p and diag_state.p_embeddings is not None:
                p_norm_mean, p_norm_p95 = compute_norm_stats(diag_state.p_embeddings)
            else:
                p_norm_mean, p_norm_p95 = 0.0, 0.0

        if scalars_enabled or norm_timeseries_enabled:
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

            if scalars_enabled:
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

            if norm_timeseries_enabled:
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

        if resolved_outputs["vis_straightline_p"].enabled:
            trajectories = _build_straightline_trajectories(
                diagnostics_cfg=diagnostics_cfg,
                diag_state=diag_state,
                model=model,
                diagnostics_generator=diagnostics_generator,
            )
            save_straightline_plot(
                diagnostics_straightline_p_dir / f"straightline_p_{global_step:07d}.png",
                trajectories,
            )

        rollout_horizon = min(diagnostics_cfg.rollout_divergence_horizon, diag_state.frames.shape[1] - 1)
        start_span = diag_state.frames.shape[1] - 1 - diag_state.warmup_frames
        can_rollout = rollout_horizon > 0 and start_span > 0
        rollout_outputs_enabled = any(
            resolved_outputs[f"vis_rollout_divergence_{kind}"].enabled
            or resolved_outputs[f"vis_rollout_divergence_excess_{kind}"].enabled
            for kind in ["z", "h", "p"]
        )
        if can_rollout and rollout_outputs_enabled:
            horizons, pixel_mean, pixel_teacher_mean, z_mean, h_mean, p_mean = compute_rollout_divergence_metrics(
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
                force_h_zero=force_h_zero,
            )
            pixel_excess = (np.maximum(np.asarray(pixel_mean) - np.asarray(pixel_teacher_mean), 0.0)).tolist()

            for kind in ["z", "h", "p"]:
                base_enabled = resolved_outputs[f"vis_rollout_divergence_{kind}"].enabled
                excess_enabled = resolved_outputs[f"vis_rollout_divergence_excess_{kind}"].enabled
                if not (base_enabled or excess_enabled):
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
                    metric = p_mean
                    out_dir = diagnostics_rollout_divergence_p_dir
                    latent_label = "P error"
                    title = "Rollout divergence (P)"
                    csv_cols = ["k", "pixel_error", "p_error"]

                if base_enabled:
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
                if excess_enabled:
                    save_rollout_divergence_plot(
                        out_dir / f"rollout_divergence_excess_{kind}_{global_step:07d}.png",
                        horizons,
                        pixel_excess,
                        metric,
                        pixel_label="Excess pixel error (MSE - recon)",
                        latent_label=latent_label,
                        title=f"{title} (excess)",
                    )
                    write_step_csv(
                        out_dir,
                        f"rollout_divergence_excess_{kind}_{global_step:07d}.csv",
                        csv_cols,
                        zip(horizons, pixel_excess, metric),
                    )
        if (
            can_rollout
            and resolved_outputs["vis_h_ablation"].enabled
            and diag_state.has_p
            and diag_state.p_embeddings is not None
            and diag_state.p_embeddings.shape[1] >= 2
        ):
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
                force_h_zero=force_h_zero,
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

        if resolved_outputs["vis_z_consistency"].enabled:
            z_consistency_samples = min(
                diagnostics_cfg.z_consistency_samples,
                diag_state.frames.shape[0] * diag_state.frames.shape[1],
            )
            distances, cosines = _compute_z_consistency_samples(
                diagnostics_cfg=diagnostics_cfg,
                diag_state=diag_state,
                model=model,
                diagnostics_generator=diagnostics_generator,
                z_consistency_samples=z_consistency_samples,
            )

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

        if resolved_outputs["vis_z_monotonicity"].enabled:
            monotonicity_samples = min(
                diagnostics_cfg.z_monotonicity_samples,
                diag_state.frames.shape[0] * diag_state.frames.shape[1],
            )

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

        if resolved_outputs["vis_path_independence"].enabled:
            stats = _compute_path_independence_stats(
                diagnostics_cfg=diagnostics_cfg,
                diag_state=diag_state,
                model=model,
                diagnostics_generator=diagnostics_generator,
            )

            label, z_mean, p_mean = stats
            save_path_independence_plot(
                diagnostics_path_independence_dir / f"path_independence_{global_step:07d}.png",
                [label],
                [z_mean],
                [p_mean],
            )
            write_step_csv(
                diagnostics_path_independence_dir,
                f"path_independence_{global_step:07d}.csv",
                ["label", "z_distance", "p_distance"],
                [(label, z_mean, p_mean)],
            )

        if resolved_outputs["vis_h_drift_by_action"].enabled:
            drift_stats = _compute_h_drift_stats(
                diagnostics_cfg=diagnostics_cfg,
                diag_state=diag_state,
            )

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

    if _any_outputs_enabled(resolved_outputs, DIAGNOSTICS_OUTPUT_GROUPS["vis_ctrl"]):
        vis_embeddings, vis_h_states, vis_actions, vis_p_embeddings = compute_vis_ctrl_state(
            model=model,
            weights=weights,
            device=device,
            vis_ctrl_batch_cpu=vis_ctrl_batch_cpu,
            force_h_zero=force_h_zero,
        )

        warmup_frames = max(model.cfg.warmup_frames_h, 0)
        vis_kind_inputs = {
            "z": vis_embeddings,
            "h": vis_h_states,
            "p": vis_p_embeddings,
        }

        summary_enabled = resolved_outputs["vis_ctrl_summary"].enabled
        metrics_by_kind = {}

        for kind in ["z", "h", "p"]:
            needs_metrics = summary_enabled and kind in ("z", "h")
            needs_metrics = needs_metrics or resolved_outputs[f"vis_ctrl_smoothness_{kind}"].enabled
            needs_metrics = needs_metrics or resolved_outputs[f"vis_ctrl_composition_{kind}"].enabled
            needs_metrics = needs_metrics or resolved_outputs[f"vis_ctrl_stability_{kind}"].enabled
            if kind == "p" and summary_enabled and vis_kind_inputs.get("p") is not None:
                needs_metrics = True
            if not needs_metrics:
                continue

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
            if resolved_outputs[f"vis_ctrl_smoothness_{kind}"].enabled:
                save_smoothness_knn_distance_eigenvalue_spectrum_plot(
                    vis_ctrl_dir / f"smoothness_{kind}_{global_step:07d}.png",
                    metrics,
                    kind,
                )
            if resolved_outputs[f"vis_ctrl_composition_{kind}"].enabled:
                save_two_step_composition_error_plot(
                    vis_ctrl_dir / f"composition_error_{kind}_{global_step:07d}.png",
                    metrics,
                    kind,
                )
            if resolved_outputs[f"vis_ctrl_stability_{kind}"].enabled:
                save_neighborhood_stability_plot(
                    vis_ctrl_dir / f"stability_{kind}_{global_step:07d}.png",
                    metrics,
                    kind,
                )
            metrics_by_kind[kind] = metrics

        if summary_enabled:
            if "z" not in metrics_by_kind or "h" not in metrics_by_kind:
                raise AssertionError("Vis-ctrl summary requires metrics for z and h.")
            write_vis_ctrl_summary(
                metrics_dir=metrics_dir,
                global_step=global_step,
                metrics_z=metrics_by_kind["z"],
                metrics_p=metrics_by_kind.get("p"),
                metrics_h=metrics_by_kind["h"],
            )

    if _any_outputs_enabled(resolved_outputs, DIAGNOSTICS_OUTPUT_GROUPS["graph"]):
        graph_diag = prepare_graph_diagnostics(
            graph_frames=graph_diag_batch_cpu[0],
            graph_actions=graph_diag_batch_cpu[1],
            model=model,
            graph_cfg=graph_cfg,
            device=device,
            use_z2h_init=should_use_z2h_init(weights),
            force_h_zero=force_h_zero,
        )

        graph_kinds = ["z", "h"]
        if model.p_action_delta_projector is not None:
            graph_kinds.append("p")

        for kind in ["z", "h", "p"]:
            if kind not in graph_kinds:
                continue
            kind_outputs_enabled = any(
                resolved_outputs[f"{prefix}_{kind}"].enabled
                for prefix in (
                    "graph_rank_cdf",
                    "graph_neff_violin",
                    "graph_in_degree",
                    "graph_edge_consistency",
                    "graph_history",
                )
            )
            if not kind_outputs_enabled:
                continue

            if kind == "z":
                graph_dir = graph_diagnostics_dir
            elif kind == "h":
                graph_dir = graph_diagnostics_h_dir
            else:
                graph_dir = graph_diagnostics_p_dir

            z_flat, target_flat, zhat_full = _compute_graph_kind_latents(
                kind=kind,
                graph_diag=graph_diag,
                model=model,
            )

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

            if resolved_outputs[f"graph_rank_cdf_{kind}"].enabled:
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
            if resolved_outputs[f"graph_neff_violin_{kind}"].enabled:
                save_neff_violin_plot(
                    graph_dir / f"neff_violin_{global_step:07d}.png",
                    stats.neff1,
                    stats.neff2,
                )
            if resolved_outputs[f"graph_in_degree_{kind}"].enabled:
                save_in_degree_hist_plot(
                    graph_dir / f"in_degree_hist_{global_step:07d}.png",
                    stats.in_degree,
                )
            if resolved_outputs[f"graph_edge_consistency_{kind}"].enabled:
                save_edge_consistency_hist_plot(
                    graph_dir / f"edge_consistency_{global_step:07d}.png",
                    stats.edge_errors,
                    embedding_label=kind,
                )
            if resolved_outputs[f"graph_history_{kind}"].enabled:
                update_graph_diagnostics_history(
                    graph_dir,
                    stats,
                    global_step,
                    metrics_dir / f"graph_diagnostics_{kind}.csv",
                )

    model.train()


def run_planning_diagnostics_step(
    *,
    planning_cfg: PlanningDiagnosticsConfig,
    hard_example_cfg: Any,
    graph_cfg: Any,
    model_cfg: Any,
    model: Any,
    device: torch.device,
    weights: Any,
    global_step: int,
    planning_batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor, List[List[str]]]],
    planning_env: Optional[GridworldKeyEnv],
    run_dir: Path,
    force_h_zero: bool = False,
) -> None:
    resolved_outputs = _resolve_outputs(
        weights=weights,
        model=model,
        hard_example_cfg=hard_example_cfg,
        graph_cfg=graph_cfg,
        planning_cfg=planning_cfg,
    )
    if not _any_outputs_enabled(resolved_outputs, DIAGNOSTICS_OUTPUT_GROUPS["planning"]):
        return
    if planning_batch_cpu is None:
        raise AssertionError("Planning diagnostics requested but planning_batch_cpu is missing.")
    if planning_env is None:
        raise AssertionError("Planning diagnostics requested but planning_env is missing.")
    plan_frames = planning_batch_cpu[0]
    plan_actions = planning_batch_cpu[1]
    if plan_frames.shape[1] < 2:
        raise AssertionError("Planning diagnostics require sequences with at least two frames.")

    run_dir = Path(run_dir)
    metrics_dir = run_dir / "metrics"
    planning_action_stats_dir = run_dir / "vis_planning_action_stats"
    planning_pca_dir = run_dir / "vis_planning_pca"
    planning_exec_dir = run_dir / "vis_planning_exec"
    planning_reachable_dir = run_dir / "vis_planning_reachable"
    planning_graph_dir = run_dir / "vis_planning_graph"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["vis_planning_action_stats"].enabled or resolved_outputs["vis_planning_action_stats_strip"].enabled:
        planning_action_stats_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["vis_planning_pca_test1"].enabled or resolved_outputs["vis_planning_pca_test2"].enabled:
        planning_pca_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["vis_planning_exec_test1"].enabled or resolved_outputs["vis_planning_exec_test2"].enabled:
        planning_exec_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["vis_planning_reachable_h"].enabled or resolved_outputs["vis_planning_reachable_p"].enabled:
        planning_reachable_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["vis_planning_graph_h"].enabled or resolved_outputs["vis_planning_graph_p"].enabled:
        planning_graph_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    (
        p_t,
        p_tp1,
        h_t,
        h_tp1,
        actions_np,
        action_labels,
        deltas,
        plan_h_states,
    ) = _extract_planning_latents(
        model,
        plan_frames,
        plan_actions,
        device,
        use_z2h_init=should_use_z2h_init(weights),
        force_h_zero=force_h_zero,
    )
    stats, graph_h, graph_p, tau_h_merge = _compute_planning_graphs(
        p_t,
        p_tp1,
        h_t,
        h_tp1,
        actions_np,
        action_labels,
        min_action_count=planning_cfg.min_action_count,
    )

    if resolved_outputs["vis_planning_action_stats"].enabled:
        plot_action_stats(
            planning_action_stats_dir / f"action_stats_{global_step:07d}.png",
            deltas,
            action_labels,
            stats.mu,
        )
    if resolved_outputs["vis_planning_action_stats_strip"].enabled:
        plot_action_strip(
            planning_action_stats_dir / f"action_stats_strip_{global_step:07d}.png",
            deltas,
            action_labels,
            stats.mu,
        )

    rng = random.Random(global_step)
    h_reach = reachable_fractions(
        graph_h,
        sample_limit=planning_cfg.reachable_fraction_samples,
        rng=rng,
    )
    p_reach = reachable_fractions(
        graph_p,
        sample_limit=planning_cfg.reachable_fraction_samples,
        rng=rng,
    )

    if resolved_outputs["vis_planning_graph_h"].enabled:
        save_planning_graph_plot(
            planning_graph_dir / f"graph_h_{global_step:07d}.png",
            h_t,
            graph_h.centers,
            graph_h.edges,
            title="Planning graph (h)",
            max_samples=planning_cfg.pca_samples,
            max_edges=4000,
        )
    if resolved_outputs["vis_planning_graph_p"].enabled:
        save_planning_graph_plot(
            planning_graph_dir / f"graph_p_{global_step:07d}.png",
            p_t,
            graph_p.centers,
            graph_p.edges,
            title="Planning graph (p)",
            max_samples=planning_cfg.pca_samples,
            max_edges=4000,
        )

    h_local_success = _run_h_local_sanity(
        graph_h,
        plan_h_states,
        tau_h_merge,
        planning_cfg,
        rng,
    )
    test_cases = _build_planning_tests(planning_env, planning_cfg)

    test_results: Dict[str, PlanningTestResult] = {}
    plan_nodes_for_plot: Dict[str, Optional[np.ndarray]] = {}
    action_dim = plan_actions.shape[-1]
    for label, start_tile, goal_tile in test_cases:
        obs_start, _ = planning_env.reset(options={"start_tile": start_tile})
        obs_goal, _ = planning_env.reset(options={"start_tile": goal_tile})
        pose_seq = _pose_from_frames(
            [obs_start, obs_goal],
            model,
            model_cfg,
            device,
            use_z2h_init=should_use_z2h_init(weights),
            action_dim=action_dim,
            force_h_zero=force_h_zero,
        )
        p_start = pose_seq[0]
        p_goal = pose_seq[1]
        plan = delta_lattice_astar(
            p_start,
            p_goal,
            stats.mu,
            r_goal=stats.r_goal,
            r_merge=stats.r_merge,
            step_scale=stats.L_scale,
            max_nodes=planning_cfg.astar_max_nodes,
        )
        visited: List[Tuple[int, int]] = [start_tile]
        final_frame = None
        if plan is None:
            test_results[label] = PlanningTestResult(
                success=False,
                steps=0,
                final_p_distance=float("inf"),
                goal_distance=float(np.linalg.norm(p_goal - p_start)),
                visited_cells=visited,
            )
            plan_nodes_for_plot[label] = None
        else:
            visited, final_frame = run_plan_in_env(
                planning_env,
                plan.actions,
                start_tile=start_tile,
            )
            final_cell = visited[-1] if visited else start_tile
            success = final_cell == goal_tile
            final_p_distance = float("inf")
            if final_frame is not None:
                final_pose = _pose_from_frames(
                    [obs_start, final_frame],
                    model,
                    model_cfg,
                    device,
                    use_z2h_init=should_use_z2h_init(weights),
                    action_dim=action_dim,
                    force_h_zero=force_h_zero,
                )
                final_p_distance = float(np.linalg.norm(final_pose[1] - p_goal))
            test_results[label] = PlanningTestResult(
                success=success,
                steps=len(plan.actions),
                final_p_distance=final_p_distance,
                goal_distance=float(np.linalg.norm(p_goal - p_start)),
                visited_cells=visited,
            )
            plan_nodes_for_plot[label] = np.stack(plan.nodes, axis=0) if plan.nodes else None

        if label == "test1" and resolved_outputs["vis_planning_exec_test1"].enabled:
            plot_grid_trace(
                planning_exec_dir / f"exec_{label}_{global_step:07d}.png",
                planning_env.grid_rows,
                planning_env.grid_cols,
                visited,
                start_tile,
                goal_tile,
            )
        if label == "test2" and resolved_outputs["vis_planning_exec_test2"].enabled:
            plot_grid_trace(
                planning_exec_dir / f"exec_{label}_{global_step:07d}.png",
                planning_env.grid_rows,
                planning_env.grid_cols,
                visited,
                start_tile,
                goal_tile,
            )
        if label == "test1" and resolved_outputs["vis_planning_pca_test1"].enabled:
            plot_pca_path(
                planning_pca_dir / f"pca_{label}_{global_step:07d}.png",
                p_t,
                plan_nodes_for_plot[label],
                p_start,
                p_goal,
                max_samples=planning_cfg.pca_samples,
            )
        if label == "test2" and resolved_outputs["vis_planning_pca_test2"].enabled:
            plot_pca_path(
                planning_pca_dir / f"pca_{label}_{global_step:07d}.png",
                p_t,
                plan_nodes_for_plot[label],
                p_start,
                p_goal,
                max_samples=planning_cfg.pca_samples,
            )

    if resolved_outputs["vis_planning_reachable_h"].enabled:
        save_reachable_fraction_hist_plot(
            planning_reachable_dir / f"reachable_h_{global_step:07d}.png",
            h_reach,
            "Reachable fraction (h)",
        )
    if resolved_outputs["vis_planning_reachable_p"].enabled:
        save_reachable_fraction_hist_plot(
            planning_reachable_dir / f"reachable_p_{global_step:07d}.png",
            p_reach,
            "Reachable fraction (p)",
        )

    _write_planning_metrics_row(
        metrics_dir,
        stats,
        graph_h,
        graph_p,
        h_reach,
        p_reach,
        h_local_success,
        test_results,
        global_step=global_step,
    )

    model.train()
