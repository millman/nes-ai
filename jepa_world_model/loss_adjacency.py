#!/usr/bin/env python3
"""Adjacency losses and utilities for planning/state embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from jepa_world_model_trainer import JEPAWorldModel, LossWeights


@dataclass
class AdjacencyConfig:
    temp_score: float = 0.1
    sinkhorn_iters: int = 7
    sinkhorn_eps: float = 1e-3
    mask_self_edges: bool = False
    topk_candidates: int = 0
    sigma_aug: float = 0.05
    detach_encoder: bool = False
    epsilon: float = 1e-4


def gaussian_augment(x: torch.Tensor, sigma: float, clip_min: float = 0.0, clip_max: float = 1.0) -> torch.Tensor:
    """Additive Gaussian noise augmentation clamped to a valid pixel range."""
    if sigma <= 0:
        return x
    noise = torch.randn_like(x) * sigma
    return (x + noise).clamp(clip_min, clip_max)


def _pair_actions(actions: torch.Tensor) -> torch.Tensor:
    """Concatenate current and prior actions for predictor conditioning."""
    if actions.ndim != 3:
        raise ValueError("Actions must have shape [B, T, action_dim].")
    if actions.shape[1] == 0:
        return actions.new_zeros((actions.shape[0], 0, actions.shape[2] * 2))
    zeros = actions.new_zeros((actions.shape[0], 1, actions.shape[2]))
    prev = torch.cat([zeros, actions[:, :-1]], dim=1)
    return torch.cat([actions, prev], dim=-1)


def _mask_topk_with_diagonal(scores: torch.Tensor, topk: int) -> torch.Tensor:
    """Keep top-k scores per row while always retaining the diagonal entry."""
    if topk <= 0 or topk >= scores.shape[1]:
        return scores
    keep = torch.zeros_like(scores, dtype=torch.bool)
    _, idx = torch.topk(scores, k=topk, dim=1)
    keep.scatter_(1, idx, True)
    diag = torch.arange(scores.shape[0], device=scores.device)
    keep[diag, diag] = True
    masked = scores.masked_fill(~keep, float("-inf"))
    return masked


def _sinkhorn_transport(kernel: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    """Compute doubly-stochastic matrix via Sinkhorn-Knopp in value domain."""
    if kernel.ndim != 2:
        raise ValueError("Sinkhorn kernel must be 2D.")
    u = torch.ones(kernel.shape[0], device=kernel.device, dtype=kernel.dtype)
    v = torch.ones(kernel.shape[1], device=kernel.device, dtype=kernel.dtype)
    for _ in range(max(iters, 0)):
        Kv = torch.matmul(kernel, v)
        u = 1.0 / (Kv + eps)
        KTu = torch.matmul(kernel.t(), u)
        v = 1.0 / (KTu + eps)
    return u[:, None] * kernel * v[None, :]


def _transport_from_scores(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    temp_score: float,
    mask_self_edges: bool,
    topk_candidates: int,
    sinkhorn_iters: int,
    sinkhorn_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if queries.shape[1] != candidates.shape[1]:
        raise ValueError("Query and candidate dimensions must match for adjacency scoring.")
    if temp_score <= 0:
        raise ValueError("temp_score must be positive for adjacency scoring.")
    # Avoid torch.cdist to stay compatible with MPS backward (cdist backward is unsupported there).
    q_norm = (queries * queries).sum(dim=1, keepdim=True)
    c_norm = (candidates * candidates).sum(dim=1, keepdim=True)
    # dist^2 = ||q||^2 + ||c||^2 - 2 q·c
    dist_sq = q_norm + c_norm.t() - 2.0 * queries @ candidates.t()
    dist_sq = dist_sq.clamp_min(0.0)
    scores = -dist_sq / temp_score
    if mask_self_edges:
        scores = scores.clone()
        scores.fill_diagonal_(float("-inf"))
    scores = _mask_topk_with_diagonal(scores, topk_candidates)
    row_max = torch.max(scores, dim=1, keepdim=True).values
    scores = scores - row_max
    kernel = torch.exp(scores)
    kernel = torch.where(torch.isfinite(kernel), kernel, torch.zeros_like(kernel))
    transport = _sinkhorn_transport(kernel, sinkhorn_iters, sinkhorn_eps)
    return transport, scores


def adjacency_loss_adj0(
    states: torch.Tensor,
    states_noisy: torch.Tensor,
    weights: LossWeights,
    cfg: AdjacencyConfig,
) -> torch.Tensor:
    if weights.adj0 <= 0 or states.shape[1] < 2:
        return states.new_tensor(0.0)
    return F.mse_loss(F.normalize(states[:, :-1], dim=-1), F.normalize(states_noisy[:, :-1], dim=-1))


def adjacency_loss_adj1(
    state_preds: torch.Tensor,
    state_targets: torch.Tensor,
    weights: LossWeights,
    cfg: AdjacencyConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zero = state_preds.new_tensor(0.0)
    if weights.adj1 <= 0 or state_preds.shape[1] < 1 or state_targets.shape[1] < 1:
        return zero, zero, zero
    preds_flat = state_preds.reshape(-1, state_preds.shape[-1])
    cand_flat = state_targets.reshape(-1, state_targets.shape[-1]).detach()
    transport, _ = _transport_from_scores(
        F.normalize(preds_flat, dim=-1),
        F.normalize(cand_flat, dim=-1),
        temp_score=cfg.temp_score,
        mask_self_edges=cfg.mask_self_edges,
        topk_candidates=cfg.topk_candidates,
        sinkhorn_iters=cfg.sinkhorn_iters,
        sinkhorn_eps=cfg.sinkhorn_eps,
    )
    diag_idx = torch.arange(transport.shape[0], device=transport.device)
    loss_adj1 = -torch.log(transport[diag_idx, diag_idx] + cfg.epsilon).mean()
    with torch.no_grad():
        row_entropy = -(transport * torch.log(transport + cfg.epsilon)).sum(dim=1)
        adj_entropy = row_entropy.mean()
        adj_hit = (transport.argmax(dim=1) == diag_idx).float().mean()
    return loss_adj1, adj_entropy, adj_hit


def adjacency_loss_adj2(
    state_preds: torch.Tensor,
    state_targets: torch.Tensor,
    weights: LossWeights,
    cfg: AdjacencyConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    zero = state_preds.new_tensor(0.0)
    if weights.adj2 <= 0 or state_preds.shape[1] < 1 or state_targets.shape[1] < 1:
        return zero, zero
    q2_flat = state_preds.reshape(-1, state_preds.shape[-1])
    cand2_flat = state_targets.reshape(-1, state_targets.shape[-1]).detach()
    transport2, _ = _transport_from_scores(
        F.normalize(q2_flat, dim=-1),
        F.normalize(cand2_flat, dim=-1),
        temp_score=cfg.temp_score,
        mask_self_edges=cfg.mask_self_edges,
        topk_candidates=cfg.topk_candidates,
        sinkhorn_iters=cfg.sinkhorn_iters,
        sinkhorn_eps=cfg.sinkhorn_eps,
    )
    if transport2.numel() == 0:
        return zero, zero
    diag2 = torch.arange(transport2.shape[0], device=transport2.device)
    loss_adj2 = -torch.log(transport2[diag2, diag2] + cfg.epsilon).mean()
    with torch.no_grad():
        adj2_hit = (transport2.argmax(dim=1) == diag2).float().mean()
    return loss_adj2, adj2_hit


def adjacency_losses(
    model: JEPAWorldModel,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    images: torch.Tensor,
    weights: LossWeights,
    cfg: AdjacencyConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute adjacency-related losses and metrics."""
    zero = embeddings.new_tensor(0.0)
    loss_adj0 = zero
    loss_adj1 = zero
    loss_adj2 = zero
    adj_entropy = zero
    adj_hit = zero
    adj2_hit = zero

    use_adjacency = (weights.adj0 > 0) or (weights.adj1 > 0) or (weights.adj2 > 0)
    if not use_adjacency or embeddings.shape[1] < 2:
        return loss_adj0, loss_adj1, loss_adj2, adj_entropy, adj_hit, adj2_hit

    plan_embeddings = embeddings.detach() if cfg.detach_encoder else embeddings
    h_plan = model.z_to_h(plan_embeddings)
    states = model.predictor.state_head(h_plan)

    loss_adj0 = zero
    if weights.adj0 > 0:
        sigma = max(cfg.sigma_aug, 0.0)
        noisy_images = gaussian_augment(images, sigma)
        noisy_outputs = model.encode_sequence(noisy_images)
        z_noisy = noisy_outputs["embeddings"]
        if cfg.detach_encoder:
            z_noisy = z_noisy.detach()
        h_noisy = model.z_to_h(z_noisy)
        states_noisy = model.predictor.state_head(h_noisy)
        loss_adj0 = adjacency_loss_adj0(states, states_noisy, weights, cfg)

    state_dim = states.shape[-1]
    state_preds = states.new_zeros((states.shape[0], max(states.shape[1] - 1, 0), state_dim))
    z_pred1: Optional[torch.Tensor] = None
    h_pred1: Optional[torch.Tensor] = None
    paired_actions = _pair_actions(actions)
    if (weights.adj1 > 0 or weights.adj2 > 0) and actions.shape[1] >= 2:
        z_pred1, _, h_pred1, state_preds = model.predictor(
            plan_embeddings[:, :-1],
            h_plan[:, :-1],
            paired_actions[:, :-1],
        )

    if weights.adj1 > 0:
        loss_adj1, adj_entropy, adj_hit = adjacency_loss_adj1(
            state_preds,
            states[:, 1:],
            weights,
            cfg,
        )

    if weights.adj2 > 0:
        if z_pred1 is not None and h_pred1 is not None and actions.shape[1] >= 3 and z_pred1.shape[1] >= 2:
            _, _, _, state_preds2 = model.predictor(
                z_pred1[:, :-1],
                h_pred1[:, :-1],
                paired_actions[:, 1:-1],
            )
        else:
            state_preds2 = state_preds.new_zeros((states.shape[0], max(states.shape[1] - 2, 0), state_dim))
        loss_adj2, adj2_hit = adjacency_loss_adj2(
            state_preds2,
            states[:, 2:],
            weights,
            cfg,
        )

    return loss_adj0, loss_adj1, loss_adj2, adj_entropy, adj_hit, adj2_hit
