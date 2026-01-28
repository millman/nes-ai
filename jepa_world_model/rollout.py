"""Rollout helpers for JEPA world model diagnostics."""
from __future__ import annotations

from typing import Tuple

import torch


def rollout_teacher_forced_z(
    model,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    *,
    use_z2h_init: bool = False,
    force_h_zero: bool = False,
) -> torch.Tensor:
    """Teacher-forced rollout that returns z_hat (enc(x_t) used at each step)."""
    b, t, _ = embeddings.shape
    if t < 2:
        return embeddings.new_zeros((b, 0, embeddings.shape[-1]))
    z_preds = []
    if force_h_zero:
        h_t = embeddings.new_zeros((b, model.state_dim))
    elif use_z2h_init and t > 0:
        h_t = model.z_to_h(embeddings[:, 0].detach())
    else:
        h_t = embeddings.new_zeros((b, model.state_dim))
    for step in range(t - 1):
        z_t = embeddings[:, step]
        act_t = actions[:, step]
        h_next = model.predictor(z_t, h_t, act_t)
        z_pred = model.h_to_z(h_next)
        z_preds.append(z_pred)
        if force_h_zero:
            h_t = embeddings.new_zeros((b, model.state_dim))
        else:
            h_t = h_next
    return torch.stack(z_preds, dim=1)


def rollout_self_fed(
    model,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    *,
    use_z2h_init: bool = False,
    force_h_zero: bool = False,
) -> torch.Tensor:
    """Roll predictor using its own predictions (no teacher forcing) to get hidden states.

    This is open-loop w.r.t. real observations (no enc(x_t) after t=0), but closed-loop
    w.r.t. the model's own predictions (z_t = h2z(h_t)).
    """
    b, t, _ = embeddings.shape
    if t < 2:
        return embeddings.new_zeros((b, 0, model.state_dim))
    if force_h_zero:
        h_t = embeddings.new_zeros((b, model.state_dim))
    elif use_z2h_init and t > 0:
        h_t = model.z_to_h(embeddings[:, 0].detach())
    else:
        h_t = embeddings.new_zeros((b, model.state_dim))
    z_t = embeddings[:, 0]
    h_preds = []
    for step in range(t - 1):
        act_t = actions[:, step]
        h_next = model.predictor(z_t, h_t, act_t)
        z_next = model.h_to_z(h_next)
        if force_h_zero:
            h_preds.append(embeddings.new_zeros((b, model.state_dim)))
        else:
            h_preds.append(h_next)
        z_t = z_next
        if force_h_zero:
            h_t = embeddings.new_zeros((b, model.state_dim))
        else:
            h_t = h_next
    return torch.stack(h_preds, dim=1)


def rollout_teacher_forced(
    model,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    *,
    use_z2h_init: bool = False,
    force_h_zero: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Teacher-forced rollout that returns z_hat plus hidden state sequences."""
    b, t, _ = embeddings.shape
    h0 = None
    if force_h_zero:
        h0 = embeddings.new_zeros((b, model.state_dim))
    elif use_z2h_init and t > 0:
        h0 = model.z_to_h(embeddings[:, 0].detach())
    if t < 2:
        z_preds = embeddings.new_zeros((b, 0, embeddings.shape[-1]))
        h_preds = embeddings.new_zeros((b, 0, model.state_dim))
        h_states = embeddings.new_zeros((b, t, model.state_dim))
        if h0 is not None and t > 0:
            h_states[:, 0] = h0
        return (
            z_preds,
            h_preds,
            h_states,
        )
    z_preds = []
    h_preds = []
    # h_preds holds the predicted next hidden states (length T-1); h_states includes h0 and all subsequent states (length T).
    h_states = [h0 if h0 is not None else embeddings.new_zeros((b, model.state_dim))]
    for step in range(t - 1):
        z_t = embeddings[:, step]
        h_t = h_states[-1]
        act_t = actions[:, step]
        h_next = model.predictor(z_t, h_t, act_t)
        z_preds.append(model.h_to_z(h_next))
        if force_h_zero:
            h_zero = embeddings.new_zeros((b, model.state_dim))
            h_preds.append(h_zero)
            h_states.append(h_zero)
        else:
            h_preds.append(h_next)
            h_states.append(h_next)
    return (
        torch.stack(z_preds, dim=1),
        torch.stack(h_preds, dim=1),
        torch.stack(h_states, dim=1),
    )
