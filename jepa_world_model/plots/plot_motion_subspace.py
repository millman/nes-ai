"""Motion subspace helpers for diagnostics."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from jepa_world_model.actions import compress_actions_to_ids


def _compute_pca(delta_z_centered: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if delta_z_centered.shape[0] < 2:
        raise ValueError("Need at least two delta steps to compute PCA.")
    try:
        _, s, vt = np.linalg.svd(delta_z_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        jitter = np.random.normal(scale=1e-6, size=delta_z_centered.shape)
        _, s, vt = np.linalg.svd(delta_z_centered + jitter, full_matrices=False)
    eigvals = (s ** 2) / max(1, delta_z_centered.shape[0] - 1)
    total_var = float(eigvals.sum()) if eigvals.size else 0.0
    if total_var <= 0:
        var_ratio = np.zeros_like(eigvals)
    else:
        var_ratio = eigvals / total_var
    return vt, var_ratio


def build_motion_subspace(
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    top_k_components: int,
    paths: Optional[List[List[str]]],
) -> Optional[Dict[str, Any]]:
    if embeddings.shape[1] < 2:
        return None
    embed_np = embeddings.detach().cpu().numpy()
    action_np = actions.detach().cpu().numpy()
    batch, seq_len, latent_dim = embed_np.shape
    delta_list: List[np.ndarray] = []
    action_vecs: List[np.ndarray] = []
    for b in range(batch):
        delta_list.append(embed_np[b, 1:] - embed_np[b, :-1])
        action_vecs.append(action_np[b, :-1])
    delta_embed = np.concatenate(delta_list, axis=0)
    if delta_embed.shape[0] < 2:
        return None
    actions_flat = np.concatenate(action_vecs, axis=0)
    action_ids = compress_actions_to_ids(actions_flat)
    delta_mean = delta_embed.mean(axis=0, keepdims=True)
    delta_centered = delta_embed - delta_mean
    components, variance_ratio = _compute_pca(delta_centered)
    use_k = max(1, min(top_k_components, components.shape[0]))
    projection = components[:use_k].T
    flat_embed = embed_np.reshape(-1, latent_dim)
    embed_centered = flat_embed - flat_embed.mean(axis=0, keepdims=True)
    delta_proj = delta_centered @ projection
    proj_flat = embed_centered @ projection
    proj_sequences: List[np.ndarray] = []
    offset = 0
    for _ in range(batch):
        proj_sequences.append(proj_flat[offset : offset + seq_len])
        offset += seq_len
    return {
        "delta_proj": delta_proj,
        "proj_flat": proj_flat,
        "proj_sequences": proj_sequences,
        "variance_ratio": variance_ratio,
        "components": components,
        "action_ids": action_ids,
        "action_dim": action_np.shape[-1],
        "actions_seq": action_np,
        "paths": paths,
    }
