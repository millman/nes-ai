from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from jepa_world_model.plots.plot_graph_diagnostics import GraphDiagnosticsConfig, build_graph_diag_indices
from jepa_world_model.plots.plot_graph_diagnostics import compute_graph_diagnostics_stats, update_graph_diagnostics_history
from jepa_world_model.plots.plot_edge_consistency_hist import save_edge_consistency_hist_plot
from jepa_world_model.plots.plot_in_degree_hist import save_in_degree_hist_plot
from jepa_world_model.plots.plot_neff_violin import save_neff_violin_plot
from jepa_world_model.plots.plot_rank_cdf import save_rank_cdf_plot
from jepa_world_model.pose_rollout import rollout_pose
from jepa_world_model.rollout import rollout_teacher_forced


@dataclass
class GraphDiagnosticsBatch:
    graph_embeddings: torch.Tensor
    graph_preds: torch.Tensor
    graph_h_preds: torch.Tensor
    graph_h_states: torch.Tensor
    ema_embeddings: Optional[torch.Tensor]
    ema_h_states: Optional[torch.Tensor]
    graph_actions: torch.Tensor
    next_index: torch.Tensor
    next2_index: torch.Tensor
    chunk_ids: torch.Tensor


def prepare_graph_diagnostics(
    *,
    graph_frames: torch.Tensor,
    graph_actions: torch.Tensor,
    model,
    ema_model,
    graph_cfg: GraphDiagnosticsConfig,
    device: torch.device,
    use_z2h_init: bool,
) -> GraphDiagnosticsBatch:
    if graph_frames.shape[1] < 3:
        raise AssertionError("Graph diagnostics require sequences with at least three frames.")
    assert torch.is_grad_enabled()
    with torch.no_grad():
        graph_frames_device = graph_frames.to(device)
        graph_actions_device = graph_actions.to(device)
        graph_embeddings = model.encode_sequence(graph_frames_device)["embeddings"]
        graph_preds, graph_h_preds, graph_h_states = rollout_teacher_forced(
            model,
            graph_embeddings,
            graph_actions_device,
            use_z2h_init=use_z2h_init,
        )
        ema_embeddings: Optional[torch.Tensor] = None
        ema_h_states: Optional[torch.Tensor] = None
        if graph_cfg.use_ema_targets and ema_model is not None:
            ema_embeddings = ema_model.encode_sequence(graph_frames_device)["embeddings"]
            _, _, ema_h_states = rollout_teacher_forced(
                ema_model,
                ema_embeddings,
                graph_actions_device,
                use_z2h_init=use_z2h_init,
            )
    assert torch.is_grad_enabled()
    next_index, next2_index, chunk_ids = build_graph_diag_indices(graph_frames_device)
    return GraphDiagnosticsBatch(
        graph_embeddings=graph_embeddings,
        graph_preds=graph_preds,
        graph_h_preds=graph_h_preds,
        graph_h_states=graph_h_states,
        ema_embeddings=ema_embeddings,
        ema_h_states=ema_h_states,
        graph_actions=graph_actions_device,
        next_index=next_index,
        next2_index=next2_index,
        chunk_ids=chunk_ids,
    )


def run_graph_diag(
    *,
    embedding_kind: str,
    graph_cfg: GraphDiagnosticsConfig,
    model,
    ema_model,
    graph_embeddings: torch.Tensor,
    graph_preds: torch.Tensor,
    graph_h_preds: torch.Tensor,
    graph_h_states: torch.Tensor,
    graph_actions: torch.Tensor,
    ema_embeddings: Optional[torch.Tensor],
    ema_h_states: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if embedding_kind == "p":
        if model.p_action_delta_projector is None:
            raise AssertionError("Graph diagnostics for p require p_action_delta_projector.")
        _, p_targets, _ = rollout_pose(
            model,
            graph_h_states,
            graph_actions,
            z_embeddings=graph_embeddings,
        )
        h_pred_states = graph_h_preds
        if h_pred_states.shape[1] + 1 == graph_h_states.shape[1]:
            h_pred_states = torch.cat([graph_h_states[:, :1], h_pred_states], dim=1)
        if graph_preds.shape[1] + 1 == graph_h_states.shape[1]:
            z_pred_embeddings = torch.cat([graph_embeddings[:, :1], graph_preds], dim=1)
        elif graph_preds.shape[1] == graph_h_states.shape[1]:
            z_pred_embeddings = graph_preds
        else:
            raise AssertionError(
                "graph_preds must match graph_h_states in time (T or T-1) to build pose rollouts."
            )
        _, p_hat, _ = rollout_pose(
            model,
            h_pred_states,
            graph_actions,
            z_embeddings=z_pred_embeddings,
        )
        p_hat_full = torch.cat([p_hat, p_targets[:, -1:, :]], dim=1)
        if graph_cfg.use_ema_targets and ema_model is not None and ema_h_states is not None and ema_embeddings is not None:
            if ema_model.p_action_delta_projector is None:
                raise AssertionError("Graph diagnostics for p require ema p_action_delta_projector.")
            _, targets, _ = rollout_pose(
                ema_model,
                ema_h_states,
                graph_actions,
                z_embeddings=ema_embeddings,
            )
        else:
            targets = p_targets
        z_flat = p_targets.reshape(-1, p_targets.shape[-1])
        target_flat = targets.reshape(-1, targets.shape[-1])
        zhat_full = p_hat_full.reshape(-1, p_hat_full.shape[-1])
    elif embedding_kind == "h":
        targets = ema_h_states if graph_cfg.use_ema_targets and ema_h_states is not None else graph_h_states
        z_flat = graph_h_states.reshape(-1, graph_h_states.shape[-1])
        target_flat = targets.reshape(-1, targets.shape[-1])
        zhat_full = torch.cat(
            [graph_h_preds, graph_h_states[:, -1:, :]],
            dim=1,
        ).reshape(-1, graph_h_states.shape[-1])
    else:
        targets = ema_embeddings if graph_cfg.use_ema_targets and ema_embeddings is not None else graph_embeddings
        z_flat = graph_embeddings.reshape(-1, graph_embeddings.shape[-1])
        target_flat = targets.reshape(-1, targets.shape[-1])
        zhat_full = torch.cat(
            [graph_preds, graph_embeddings[:, -1:, :]],
            dim=1,
        ).reshape(-1, graph_embeddings.shape[-1])

    if graph_cfg.normalize_latents:
        z_flat = F.normalize(z_flat, dim=-1)
        target_flat = F.normalize(target_flat, dim=-1)
        zhat_full = F.normalize(zhat_full, dim=-1)

    queries = zhat_full if graph_cfg.use_predictor_scores else z_flat
    return queries, target_flat, zhat_full


def run_graph_diagnostics_for_kind(
    *,
    kind: str,
    graph_cfg,
    model,
    ema_model,
    global_step: int,
    graph_diag: GraphDiagnosticsBatch,
    metrics_dir: Path,
    graph_diagnostics_dir: Path,
) -> None:
    queries, targets, predictions = run_graph_diag(
        embedding_kind=kind,
        graph_cfg=graph_cfg,
        model=model,
        ema_model=ema_model,
        graph_embeddings=graph_diag.graph_embeddings,
        graph_preds=graph_diag.graph_preds,
        graph_h_preds=graph_diag.graph_h_preds,
        graph_h_states=graph_diag.graph_h_states,
        graph_actions=graph_diag.graph_actions,
        ema_embeddings=graph_diag.ema_embeddings,
        ema_h_states=graph_diag.ema_h_states,
    )
    stats = compute_graph_diagnostics_stats(
        queries,
        targets,
        predictions,
        graph_diag.next_index,
        graph_diag.next2_index,
        graph_diag.chunk_ids,
        graph_cfg,
        global_step,
    )
    save_rank_cdf_plot(
        graph_diagnostics_dir / f"rank1_cdf_{global_step:07d}.png",
        stats.ranks1,
        stats.k,
        "1-step rank CDF",
    )
    save_rank_cdf_plot(
        graph_diagnostics_dir / f"rank2_cdf_{global_step:07d}.png",
        stats.ranks2,
        stats.k,
        "2-hop rank CDF",
    )
    save_neff_violin_plot(
        graph_diagnostics_dir / f"neff_violin_{global_step:07d}.png",
        stats.neff1,
        stats.neff2,
    )
    save_in_degree_hist_plot(
        graph_diagnostics_dir / f"in_degree_hist_{global_step:07d}.png",
        stats.in_degree,
    )
    save_edge_consistency_hist_plot(
        graph_diagnostics_dir / f"edge_consistency_{global_step:07d}.png",
        stats.edge_errors,
        embedding_label=kind,
    )
    update_graph_diagnostics_history(
        graph_diagnostics_dir,
        stats,
        global_step,
        metrics_dir / f"graph_diagnostics_{kind}.csv",
    )
