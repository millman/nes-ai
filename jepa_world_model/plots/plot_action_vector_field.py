"""Action-conditioned vector field and time-slice plots."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from jepa_world_model.actions import decode_action_id
from jepa_world_model.plots.plot_layout import DEFAULT_DPI, apply_square_axes, figsize_for_grid


def _compute_pca_components(data: np.ndarray, *, k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    if data.ndim != 2:
        raise ValueError("PCA expects a 2D array.")
    if data.shape[0] < 2:
        raise ValueError("Need at least two samples for PCA.")
    centered = data - data.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    k = max(1, min(k, vt.shape[0]))
    eigvals = (s ** 2) / max(1, centered.shape[0] - 1)
    total = float(eigvals.sum()) if eigvals.size else 0.0
    var_ratio = eigvals[:k] / total if total > 0 else np.zeros((k,), dtype=np.float32)
    return vt[:k], var_ratio


def _select_top_actions(action_ids: np.ndarray, max_actions: int) -> List[int]:
    unique, counts = np.unique(action_ids, return_counts=True)
    order = np.argsort(counts)[::-1]
    top = [int(unique[idx]) for idx in order[:max_actions]]
    return top


def save_action_vector_field_plot(
    out_path: Path,
    embeddings: np.ndarray,
    deltas: np.ndarray,
    action_ids: np.ndarray,
    action_dim: int,
    *,
    max_actions: int = 4,
    grid_size: int = 10,
    min_count: int = 5,
    title: str = "Action-conditioned vector field (latent space)",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if embeddings.shape[0] == 0 or deltas.shape[0] == 0:
        raise ValueError("Empty embeddings or deltas for vector field plot.")

    components, _ = _compute_pca_components(embeddings, k=2)
    embed_centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    embed_proj = embed_centered @ components.T
    delta_proj = deltas @ components.T

    actions = _select_top_actions(action_ids, max_actions=max_actions)
    rows = int(np.ceil(len(actions) / 2)) if actions else 1
    cols = 2 if len(actions) > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=figsize_for_grid(rows, cols), constrained_layout=True)
    axes_arr = np.atleast_2d(axes)

    x_min, x_max = float(embed_proj[:, 0].min()), float(embed_proj[:, 0].max())
    y_min, y_max = float(embed_proj[:, 1].min()), float(embed_proj[:, 1].max())
    x_edges = np.linspace(x_min, x_max, grid_size + 1)
    y_edges = np.linspace(y_min, y_max, grid_size + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) * 0.5
    y_centers = (y_edges[:-1] + y_edges[1:]) * 0.5

    for idx, aid in enumerate(actions):
        ax = axes_arr[idx // cols, idx % cols]
        mask = action_ids == aid
        if not np.any(mask):
            ax.set_title(f"{decode_action_id(aid, action_dim)} (no samples)")
            ax.axis("off")
            continue
        ax.scatter(embed_proj[:, 0], embed_proj[:, 1], s=6, alpha=0.08, color="gray")
        delta_means = np.zeros((grid_size, grid_size, 2), dtype=np.float32)
        counts = np.zeros((grid_size, grid_size), dtype=np.int32)
        for point, delta in zip(embed_proj[mask], delta_proj[mask]):
            xi = int(np.searchsorted(x_edges, point[0], side="right") - 1)
            yi = int(np.searchsorted(y_edges, point[1], side="right") - 1)
            if xi < 0 or xi >= grid_size or yi < 0 or yi >= grid_size:
                continue
            delta_means[yi, xi] += delta
            counts[yi, xi] += 1
        valid = counts >= min_count
        if np.any(valid):
            delta_means[valid] /= counts[valid][..., None]
            xx, yy = np.meshgrid(x_centers, y_centers)
            ax.quiver(
                xx[valid],
                yy[valid],
                delta_means[..., 0][valid],
                delta_means[..., 1][valid],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.003,
                color="tab:blue",
            )
        ax.set_title(decode_action_id(aid, action_dim))
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    fig.suptitle(title, fontsize=10)
    apply_square_axes(axes)
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)


def save_action_time_slice_plot(
    out_path: Path,
    deltas: np.ndarray,
    action_ids: np.ndarray,
    action_dim: int,
    *,
    max_actions: int = 4,
    min_count: int = 5,
    title: str = "Action delta time slices (latent PCA)",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if deltas.ndim != 3:
        raise ValueError("Expected deltas with shape [B, T-1, D].")
    bsz, steps, _ = deltas.shape
    flat = deltas.reshape(-1, deltas.shape[-1])
    components, _ = _compute_pca_components(flat, k=2)
    delta_proj = deltas @ components.T

    actions = _select_top_actions(action_ids.reshape(-1), max_actions=max_actions)
    rows = int(np.ceil(len(actions) / 2)) if actions else 1
    cols = 2 if len(actions) > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=figsize_for_grid(rows, cols), constrained_layout=True)
    axes_arr = np.atleast_2d(axes)

    for idx, aid in enumerate(actions):
        ax = axes_arr[idx // cols, idx % cols]
        mean_pc1 = np.full((steps,), np.nan, dtype=np.float32)
        mean_pc2 = np.full((steps,), np.nan, dtype=np.float32)
        for t in range(steps):
            mask = action_ids[:, t] == aid
            if mask.sum() < min_count:
                continue
            mean_delta = delta_proj[mask, t].mean(axis=0)
            mean_pc1[t] = mean_delta[0]
            mean_pc2[t] = mean_delta[1]
        ax.plot(np.arange(steps), mean_pc1, marker="o", label="PC1")
        ax.plot(np.arange(steps), mean_pc2, marker="o", label="PC2")
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("time index")
        ax.set_ylabel("mean delta")
        ax.set_title(decode_action_id(aid, action_dim))
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    apply_square_axes(axes)
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)
