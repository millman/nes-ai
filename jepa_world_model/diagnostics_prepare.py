from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from jepa_world_model.actions import compress_actions_to_ids, decode_action_id
from jepa_world_model.plots.build_motion_subspace import MotionSubspace, build_motion_subspace
from jepa_world_model.plots.plot_action_inverse_map import build_action_inverse_map
from jepa_world_model.pose_rollout import rollout_pose
from jepa_world_model.rollout import rollout_teacher_forced
from jepa_world_model.vis_composability import compute_composability_series


@dataclass
class ActionMetadata:
    action_ids_flat: np.ndarray
    unique_actions: np.ndarray
    action_counts: np.ndarray
    action_labels: Dict[int, str]
    action_vectors: Dict[int, torch.Tensor]
    noop_id: Optional[int]


@dataclass
class DiagnosticsBatchState:
    frames: torch.Tensor
    actions: torch.Tensor
    paths: List[List[str]]
    frames_device: torch.Tensor
    actions_device: torch.Tensor
    embeddings: torch.Tensor
    h_states: torch.Tensor
    p_embeddings: Optional[torch.Tensor]
    p_deltas: Optional[torch.Tensor]
    warmup_frames: int
    has_p: bool
    inverse_map: Dict[int, Dict[int, int]]
    motion_z: MotionSubspace
    motion_h: MotionSubspace
    motion_p: Optional[MotionSubspace]
    action_ids_seq: np.ndarray
    action_metadata: ActionMetadata
    composability: Optional[Dict[str, np.ndarray]]
    p_series: Optional[np.ndarray]


def prepare_diagnostics_batch_state(
    *,
    model,
    diagnostics_cfg,
    weights,
    diagnostics_batch_cpu: Tuple[torch.Tensor, torch.Tensor, List[List[str]]],
    device: torch.device,
) -> DiagnosticsBatchState:
    diag_frames = diagnostics_batch_cpu[0]
    diag_actions = diagnostics_batch_cpu[1]
    diag_paths = diagnostics_batch_cpu[2]
    if not (diag_frames.shape[0] > 0 and diag_frames.shape[1] >= 2):
        raise AssertionError("Diagnostics require at least one sequence with two frames.")

    warmup_frames = max(model.cfg.warmup_frames_h, 0)
    assert torch.is_grad_enabled()
    with torch.no_grad():
        diag_frames_device = diag_frames.to(device)
        diag_actions_device = diag_actions.to(device)
        diag_embeddings = model.encode_sequence(diag_frames_device)["embeddings"]
        _, _, diag_h_states = rollout_teacher_forced(
            model,
            diag_embeddings,
            diag_actions_device,
            use_z2h_init=weights.z2h > 0,
        )
        diag_p_embeddings = None
        diag_p_deltas = None
        if model.p_action_delta_projector is not None:
            _, diag_p_embeddings, diag_p_deltas = rollout_pose(
                model,
                diag_h_states,
                diag_actions_device,
                z_embeddings=diag_embeddings,
            )
    assert torch.is_grad_enabled()

    has_p = diag_p_embeddings is not None and model.p_action_delta_projector is not None
    diag_composability = None
    diag_p_series = None
    if (
        has_p
        and model.z_action_delta_projector is not None
        and model.h_action_delta_projector is not None
    ):
        diag_composability = compute_composability_series(
            model,
            diag_embeddings,
            diag_h_states,
            diag_actions_device,
            warmup_frames,
            diagnostics_cfg.min_action_count,
        )
        diag_p_series = diag_composability.get("p") or diag_composability.get("s")

    inverse_map = build_action_inverse_map(diag_actions.detach().cpu().numpy())
    motion_z = build_motion_subspace(
        diag_embeddings,
        diag_actions,
        diagnostics_cfg.top_k_components,
        diag_paths,
    )
    motion_h = build_motion_subspace(
        diag_h_states,
        diag_actions,
        diagnostics_cfg.top_k_components,
        diag_paths,
    )
    motion_p = None
    if has_p:
        motion_p = build_motion_subspace(
            diag_p_embeddings,
            diag_actions,
            diagnostics_cfg.top_k_components,
            diag_paths,
        )

    action_ids_seq = compress_actions_to_ids(diag_actions.detach().cpu().numpy())
    if action_ids_seq.ndim == 1:
        action_ids_seq = action_ids_seq.reshape(diag_actions.shape[0], diag_actions.shape[1])
    action_ids_flat = action_ids_seq[:, :-1].reshape(-1)
    unique_actions, action_counts = np.unique(action_ids_flat, return_counts=True)
    action_labels = {int(aid): decode_action_id(int(aid), motion_z.action_dim) for aid in unique_actions}
    action_vectors: Dict[int, torch.Tensor] = {}
    flat_actions = diag_actions.detach().cpu().reshape(-1, diag_actions.shape[-1])
    for idx, aid in enumerate(action_ids_flat):
        if int(aid) not in action_vectors:
            action_vectors[int(aid)] = flat_actions[idx].to(device)
    noop_id = None
    for aid, label in action_labels.items():
        if "NOOP" in label:
            noop_id = aid
            break

    action_metadata = ActionMetadata(
        action_ids_flat=action_ids_flat,
        unique_actions=unique_actions,
        action_counts=action_counts,
        action_labels=action_labels,
        action_vectors=action_vectors,
        noop_id=noop_id,
    )

    return DiagnosticsBatchState(
        frames=diag_frames,
        actions=diag_actions,
        paths=diag_paths,
        frames_device=diag_frames_device,
        actions_device=diag_actions_device,
        embeddings=diag_embeddings,
        h_states=diag_h_states,
        p_embeddings=diag_p_embeddings,
        p_deltas=diag_p_deltas,
        warmup_frames=warmup_frames,
        has_p=has_p,
        inverse_map=inverse_map,
        motion_z=motion_z,
        motion_h=motion_h,
        motion_p=motion_p,
        action_ids_seq=action_ids_seq,
        action_metadata=action_metadata,
        composability=diag_composability,
        p_series=diag_p_series,
    )
