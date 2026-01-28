from __future__ import annotations

import csv
from pathlib import Path
from dataclasses import dataclass
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


@dataclass(frozen=True)
class DiagnosticsOutputSpec:
    enabled: bool
    description: str


DIAGNOSTICS_OUTPUT_CATALOG = {
    "rollout_fixed": DiagnosticsOutputSpec(True, "Fixed batch rollout sequence grid."),
    "rollout_rolling": DiagnosticsOutputSpec(True, "Rolling batch rollout sequence grid."),
    "off_manifold": DiagnosticsOutputSpec(True, "Off-manifold rollout error visualization."),
    "pca_z": DiagnosticsOutputSpec(True, "PCA projection plot for z embeddings."),
    "pca_h": DiagnosticsOutputSpec(True, "PCA projection plot for h states."),
    "pca_p": DiagnosticsOutputSpec(True, "PCA projection plot for p embeddings."),
    "hard_examples_train": DiagnosticsOutputSpec(True, "Hard-example grid from the training reservoir."),
    "hard_examples_val": DiagnosticsOutputSpec(True, "Hard-example grid from the validation reservoir."),
    "self_distance_z": DiagnosticsOutputSpec(True, "Self-distance metrics + visuals for z."),
    "self_distance_h": DiagnosticsOutputSpec(True, "Self-distance metrics + visuals for h."),
    "self_distance_p": DiagnosticsOutputSpec(True, "Self-distance metrics + visuals for p."),
    "state_embedding": DiagnosticsOutputSpec(False, "State embedding projections + odometry visuals."),
    "diagnostics_action_field_z": DiagnosticsOutputSpec(True, "Action-conditioned vector field plots for z."),
    "diagnostics_action_field_h": DiagnosticsOutputSpec(True, "Action-conditioned vector field plots for h."),
    "diagnostics_action_field_p": DiagnosticsOutputSpec(True, "Action-conditioned vector field plots for p."),
    "diagnostics_action_time_z": DiagnosticsOutputSpec(True, "Action delta time-slice plots for z."),
    "diagnostics_action_time_h": DiagnosticsOutputSpec(True, "Action delta time-slice plots for h."),
    "diagnostics_action_time_p": DiagnosticsOutputSpec(True, "Action delta time-slice plots for p."),
    "diagnostics_composability_z": DiagnosticsOutputSpec(True, "Composability plots for z."),
    "diagnostics_composability_h": DiagnosticsOutputSpec(True, "Composability plots for h."),
    "diagnostics_composability_p": DiagnosticsOutputSpec(True, "Composability plots for p."),
    "diagnostics_motion_pca_z": DiagnosticsOutputSpec(True, "Motion PCA artifacts for z."),
    "diagnostics_motion_pca_h": DiagnosticsOutputSpec(True, "Motion PCA artifacts for h."),
    "diagnostics_motion_pca_p": DiagnosticsOutputSpec(True, "Motion PCA artifacts for p."),
    "diagnostics_alignment_z": DiagnosticsOutputSpec(True, "Action-alignment artifacts for z."),
    "diagnostics_alignment_h": DiagnosticsOutputSpec(True, "Action-alignment artifacts for h."),
    "diagnostics_alignment_p": DiagnosticsOutputSpec(True, "Action-alignment artifacts for p."),
    "diagnostics_cycle_z": DiagnosticsOutputSpec(False, "Cycle-error artifacts for z."),
    "diagnostics_cycle_h": DiagnosticsOutputSpec(False, "Cycle-error artifacts for h."),
    "diagnostics_cycle_p": DiagnosticsOutputSpec(False, "Cycle-error artifacts for p."),
    "diagnostics_frames": DiagnosticsOutputSpec(False, "Diagnostics frame dumps + CSV."),
    "diagnostics_scalars": DiagnosticsOutputSpec(True, "Diagnostics scalar CSV summary."),
    "diagnostics_norm_timeseries": DiagnosticsOutputSpec(True, "Norm timeseries plot."),
    "diagnostics_straightline_p": DiagnosticsOutputSpec(True, "Straight-line rollout plot in p."),
    "diagnostics_rollout_divergence_z": DiagnosticsOutputSpec(True, "Rollout divergence plots/CSVs for z."),
    "diagnostics_rollout_divergence_h": DiagnosticsOutputSpec(True, "Rollout divergence plots/CSVs for h."),
    "diagnostics_rollout_divergence_p": DiagnosticsOutputSpec(True, "Rollout divergence plots/CSVs for p."),
    "diagnostics_h_ablation": DiagnosticsOutputSpec(True, "H-ablation divergence plots/CSVs."),
    "diagnostics_z_consistency": DiagnosticsOutputSpec(True, "Z consistency plots/CSVs."),
    "diagnostics_z_monotonicity": DiagnosticsOutputSpec(True, "Z monotonicity plots/CSVs."),
    "diagnostics_path_independence": DiagnosticsOutputSpec(True, "Path-independence plots/CSVs."),
    "diagnostics_h_drift": DiagnosticsOutputSpec(True, "H drift-by-action plot/CSV."),
    "vis_ctrl_smoothness_z": DiagnosticsOutputSpec(False, "Vis-ctrl smoothness spectrum for z."),
    "vis_ctrl_smoothness_h": DiagnosticsOutputSpec(False, "Vis-ctrl smoothness spectrum for h."),
    "vis_ctrl_smoothness_p": DiagnosticsOutputSpec(False, "Vis-ctrl smoothness spectrum for p."),
    "vis_ctrl_composition_error_z": DiagnosticsOutputSpec(False, "Vis-ctrl two-step composition error for z."),
    "vis_ctrl_composition_error_h": DiagnosticsOutputSpec(False, "Vis-ctrl two-step composition error for h."),
    "vis_ctrl_composition_error_p": DiagnosticsOutputSpec(False, "Vis-ctrl two-step composition error for p."),
    "vis_ctrl_stability_z": DiagnosticsOutputSpec(False, "Vis-ctrl neighborhood stability plot for z."),
    "vis_ctrl_stability_h": DiagnosticsOutputSpec(False, "Vis-ctrl neighborhood stability plot for h."),
    "vis_ctrl_stability_p": DiagnosticsOutputSpec(False, "Vis-ctrl neighborhood stability plot for p."),
    "vis_ctrl_summary": DiagnosticsOutputSpec(False, "Vis-ctrl metrics CSV summary."),
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
    "rollouts": ("rollout_fixed", "rollout_rolling"),
    "pca": ("pca_z", "pca_h", "pca_p"),
    "hard_examples": ("hard_examples_train", "hard_examples_val"),
    "self_distance": ("self_distance_z", "self_distance_h", "self_distance_p", "state_embedding"),
    "diagnostics": tuple(key for key in DIAGNOSTICS_OUTPUT_CATALOG.keys() if key.startswith("diagnostics_")),
    "vis_ctrl": tuple(key for key in DIAGNOSTICS_OUTPUT_CATALOG.keys() if key.startswith("vis_ctrl_")),
    "graph": tuple(key for key in DIAGNOSTICS_OUTPUT_CATALOG.keys() if key.startswith("graph_")),
}

OUTPUT_KIND_OVERRIDES = {
    "diagnostics_h_ablation": "p",
    "diagnostics_path_independence": "p",
    "state_embedding": "p",
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
) -> dict[str, DiagnosticsOutputSpec]:
    enabled_kinds = _enabled_kinds(weights, model)
    group_enabled = {
        "hard_examples": getattr(hard_example_cfg, "reservoir", 0) > 0,
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
    if resolved_outputs["off_manifold"].enabled and off_manifold_batch_cpu is None:
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

    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["rollout_fixed"].enabled:
        fixed_vis_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["rollout_rolling"].enabled:
        rolling_vis_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["pca_z"].enabled:
        pca_z_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["pca_p"].enabled:
        pca_p_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["pca_h"].enabled:
        pca_h_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["off_manifold"].enabled:
        vis_off_manifold_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["hard_examples_train"].enabled:
        samples_hard_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["hard_examples_val"].enabled:
        samples_hard_val_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["self_distance_z"].enabled:
        vis_self_distance_z_dir.mkdir(parents=True, exist_ok=True)
        self_distance_z_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["self_distance_p"].enabled or resolved_outputs["state_embedding"].enabled:
        vis_self_distance_p_dir.mkdir(parents=True, exist_ok=True)
        self_distance_p_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["self_distance_h"].enabled:
        vis_self_distance_h_dir.mkdir(parents=True, exist_ok=True)
        self_distance_h_dir.mkdir(parents=True, exist_ok=True)
    if resolved_outputs["state_embedding"].enabled:
        vis_state_embedding_dir.mkdir(parents=True, exist_ok=True)
        vis_odometry_dir.mkdir(parents=True, exist_ok=True)
    if _any_outputs_enabled(resolved_outputs, DIAGNOSTICS_OUTPUT_GROUPS["diagnostics"]):
        if resolved_outputs["diagnostics_motion_pca_z"].enabled:
            diagnostics_delta_z_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_motion_pca_p"].enabled:
            diagnostics_delta_p_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_motion_pca_h"].enabled:
            diagnostics_delta_h_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_alignment_z"].enabled:
            diagnostics_alignment_z_dir.mkdir(parents=True, exist_ok=True)
            diagnostics_alignment_z_raw_dir.mkdir(parents=True, exist_ok=True)
            diagnostics_alignment_z_centered_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_alignment_p"].enabled:
            diagnostics_alignment_p_dir.mkdir(parents=True, exist_ok=True)
            diagnostics_alignment_p_raw_dir.mkdir(parents=True, exist_ok=True)
            diagnostics_alignment_p_centered_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_alignment_h"].enabled:
            diagnostics_alignment_h_dir.mkdir(parents=True, exist_ok=True)
            diagnostics_alignment_h_raw_dir.mkdir(parents=True, exist_ok=True)
            diagnostics_alignment_h_centered_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_cycle_z"].enabled:
            diagnostics_cycle_z_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_cycle_p"].enabled:
            diagnostics_cycle_p_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_cycle_h"].enabled:
            diagnostics_cycle_h_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_frames"].enabled:
            diagnostics_frames_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_composability_z"].enabled:
            vis_composability_z_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_composability_p"].enabled:
            vis_composability_p_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_composability_h"].enabled:
            vis_composability_h_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_rollout_divergence_z"].enabled:
            diagnostics_rollout_divergence_z_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_rollout_divergence_h"].enabled:
            diagnostics_rollout_divergence_h_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_rollout_divergence_p"].enabled:
            diagnostics_rollout_divergence_p_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_action_field_z"].enabled:
            diagnostics_action_field_z_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_action_field_h"].enabled:
            diagnostics_action_field_h_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_action_field_p"].enabled:
            diagnostics_action_field_p_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_action_time_z"].enabled:
            diagnostics_action_time_z_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_action_time_h"].enabled:
            diagnostics_action_time_h_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_action_time_p"].enabled:
            diagnostics_action_time_p_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_straightline_p"].enabled:
            diagnostics_straightline_p_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_z_consistency"].enabled:
            diagnostics_z_consistency_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_z_monotonicity"].enabled:
            diagnostics_z_monotonicity_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_path_independence"].enabled:
            diagnostics_path_independence_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_h_ablation"].enabled:
            diagnostics_h_ablation_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_h_drift"].enabled:
            diagnostics_h_drift_dir.mkdir(parents=True, exist_ok=True)
        if resolved_outputs["diagnostics_norm_timeseries"].enabled:
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

    if resolved_outputs["rollout_fixed"].enabled:
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
        )

    if resolved_outputs["rollout_rolling"].enabled:
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
        )

    if resolved_outputs["off_manifold"].enabled and off_manifold_batch_cpu is not None:
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

    if resolved_outputs["hard_examples_train"].enabled:
        hard_samples = hard_reservoir.topk(hard_example_cfg.vis_rows * hard_example_cfg.vis_columns)
        save_hard_example_grid(
            samples_hard_dir / f"hard_{global_step:07d}.png",
            hard_samples,
            hard_example_cfg.vis_columns,
            hard_example_cfg.vis_rows,
            dataset.image_hw,
        )
    if resolved_outputs["hard_examples_val"].enabled:
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

        if resolved_outputs["self_distance_z"].enabled:
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
        if resolved_outputs["self_distance_h"].enabled:
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
        if resolved_outputs["self_distance_p"].enabled:
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
        if resolved_outputs["state_embedding"].enabled:
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
            resolved_outputs["diagnostics_action_field_z"].enabled
            or resolved_outputs["diagnostics_action_time_z"].enabled
        ):
            z_embed_np = diag_state.embeddings.detach().cpu().numpy()
            z_deltas = z_embed_np[:, 1:] - z_embed_np[:, :-1]
            z_action_dim = diag_state.motion_z.action_dim

            if resolved_outputs["diagnostics_action_field_z"].enabled:
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
            if resolved_outputs["diagnostics_action_time_z"].enabled:
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
            resolved_outputs["diagnostics_action_field_h"].enabled
            or resolved_outputs["diagnostics_action_time_h"].enabled
        ):
            h_embed_np = diag_state.h_states.detach().cpu().numpy()
            h_deltas = h_embed_np[:, 1:] - h_embed_np[:, :-1]
            h_action_dim = diag_state.motion_h.action_dim

            if resolved_outputs["diagnostics_action_field_h"].enabled:
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
            if resolved_outputs["diagnostics_action_time_h"].enabled:
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
            resolved_outputs["diagnostics_action_field_p"].enabled
            or resolved_outputs["diagnostics_action_time_p"].enabled
        ):
            p_embed_np = diag_state.p_embeddings.detach().cpu().numpy()
            p_deltas = p_embed_np[:, 1:] - p_embed_np[:, :-1]
            p_action_dim = diag_state.motion_p.action_dim

            if resolved_outputs["diagnostics_action_field_p"].enabled:
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
            if resolved_outputs["diagnostics_action_time_p"].enabled:
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
            if resolved_outputs[f"diagnostics_composability_z"].enabled:
                save_composability_plot(
                    vis_composability_z_dir / f"composability_z_{global_step:07d}.png",
                    diag_state.composability["z"],
                    "z",
                )
            if resolved_outputs[f"diagnostics_composability_h"].enabled:
                save_composability_plot(
                    vis_composability_h_dir / f"composability_h_{global_step:07d}.png",
                    diag_state.composability["h"],
                    "h",
                )
            if resolved_outputs[f"diagnostics_composability_p"].enabled:
                save_composability_plot(
                    vis_composability_p_dir / f"composability_p_{global_step:07d}.png",
                    diag_state.p_series,
                    "p",
                )

        for kind in ["z", "h", "p"]:
            motion_pca_enabled = resolved_outputs[f"diagnostics_motion_pca_{kind}"].enabled
            alignment_enabled = resolved_outputs[f"diagnostics_alignment_{kind}"].enabled
            cycle_enabled = resolved_outputs[f"diagnostics_cycle_{kind}"].enabled

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
        norm_timeseries_enabled = resolved_outputs["diagnostics_norm_timeseries"].enabled
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

        if resolved_outputs["diagnostics_straightline_p"].enabled:
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
            resolved_outputs[f"diagnostics_rollout_divergence_{kind}"].enabled for kind in ["z", "h", "p"]
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
                if not resolved_outputs[f"diagnostics_rollout_divergence_{kind}"].enabled:
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

                save_rollout_divergence_plot(
                    out_dir / f"rollout_divergence_{kind}_{global_step:07d}.png",
                    horizons,
                    pixel_mean,
                    metric,
                    latent_label=latent_label,
                    title=title,
                )
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
                    f"rollout_divergence_{kind}_{global_step:07d}.csv",
                    csv_cols,
                    zip(horizons, pixel_mean, metric),
                )
                write_step_csv(
                    out_dir,
                    f"rollout_divergence_excess_{kind}_{global_step:07d}.csv",
                    csv_cols,
                    zip(horizons, pixel_excess, metric),
                )
        if (
            can_rollout
            and resolved_outputs["diagnostics_h_ablation"].enabled
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

        if resolved_outputs["diagnostics_z_consistency"].enabled:
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

        if resolved_outputs["diagnostics_z_monotonicity"].enabled:
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

        if resolved_outputs["diagnostics_path_independence"].enabled:
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

        if resolved_outputs["diagnostics_h_drift"].enabled:
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
            needs_metrics = needs_metrics or resolved_outputs[f"vis_ctrl_composition_error_{kind}"].enabled
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
            if resolved_outputs[f"vis_ctrl_composition_error_{kind}"].enabled:
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
