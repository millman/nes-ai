#!/usr/bin/env python3
"""Pose rollout utilities for odometry-style pose integration."""
from __future__ import annotations

from typing import Tuple

import torch


def rollout_pose_sequence(
    model,
    h_states: torch.Tensor,
    actions: torch.Tensor,
    z_embeddings: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Roll out pose by integrating Δp from the pose delta projector.

    Returns:
        pose_seq: [B, T, pose_dim] pose trajectory
        pose_deltas: [B, T-1, pose_dim] per-step increments
    """
    if h_states.ndim != 3:
        raise AssertionError("h_states must have shape [B, T, D].")
    if actions.ndim != 3:
        raise AssertionError("actions must have shape [B, T, A].")
    if h_states.shape[0] != actions.shape[0]:
        raise AssertionError("h_states and actions must share the batch dimension.")
    if h_states.shape[1] < 1:
        raise AssertionError("h_states must include at least one timestep.")
    if actions.shape[1] < 1:
        raise AssertionError("actions must include at least one timestep.")
    if model.p_action_delta_projector is None:
        raise AssertionError("p_action_delta_projector is required to roll out pose deltas.")
    if z_embeddings is not None and z_embeddings.ndim != 3:
        raise AssertionError("z_embeddings must have shape [B, T, D].")
    if z_embeddings is not None and z_embeddings.shape[0] != h_states.shape[0]:
        raise AssertionError("z_embeddings and h_states must share the batch dimension.")
    bsz, steps, _ = h_states.shape
    pose_dim = model.p_action_delta_projector.pose_dim
    pose_start = h_states.new_zeros((bsz, pose_dim))
    if steps <= 1 or actions.shape[1] < 1:
        pose_single = pose_start.unsqueeze(1).repeat(1, steps, 1)
        zero_delta = pose_start.new_zeros((bsz, 0, pose_dim))
        return pose_single, zero_delta
    max_steps = min(actions.shape[1], steps - 1)
    pose_pred = [pose_start]
    pose_deltas = []
    for idx in range(max_steps):
        h_in = h_states[:, idx]
        if model.cfg.pose_delta_detach_h:
            h_in = h_in.detach()
        delta = model.p_action_delta_projector(pose_pred[-1], h_in, actions[:, idx])
        pose_next = pose_pred[-1] + delta
        if model.p_correction_projector is not None and z_embeddings is not None:
            if idx + 1 < z_embeddings.shape[1]:
                z_in = z_embeddings[:, idx + 1]
                if model.cfg.pose_correction_detach_z:
                    z_in = z_in.detach()
                pose_next = pose_next + model.p_correction_projector(pose_next, z_in)
        pose_deltas.append(delta)
        pose_pred.append(pose_next)
    pose_pred_tensor = torch.stack(pose_pred, dim=1)
    if pose_pred_tensor.shape[1] < steps:
        pad = pose_pred_tensor[:, -1:].repeat(1, steps - pose_pred_tensor.shape[1], 1)
        pose_pred_tensor = torch.cat([pose_pred_tensor, pad], dim=1)
    return pose_pred_tensor, torch.stack(pose_deltas, dim=1)
