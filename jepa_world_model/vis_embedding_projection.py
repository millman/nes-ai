from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from jepa_world_model.plots.plot_layout import DEFAULT_DPI, apply_square_axes, figsize_for_grid
from jepa_world_model.plots.plot_grid_overlay import GridOverlay, draw_grid_overlay

def save_embedding_projection(
    embeddings: torch.Tensor,
    path: Path,
    title: str,
    *,
    grid_overlay: Optional[GridOverlay] = None,
) -> None:
    flat = embeddings.detach().cpu().numpy()
    b, t, dim = flat.shape
    flat = flat.reshape(b * t, dim)
    mean = flat.mean(axis=0, keepdims=True)
    flat = flat - mean
    if flat.shape[0] < 2:
        return
    try:
        _, _, vt = np.linalg.svd(flat, full_matrices=False)
    except np.linalg.LinAlgError:
        jitter = np.random.normal(scale=1e-6, size=flat.shape)
        try:
            _, _, vt = np.linalg.svd(flat + jitter, full_matrices=False)
        except np.linalg.LinAlgError:
            print("Warning: embedding SVD failed to converge; skipping projection.")
            return
    coords = flat @ vt[:2].T
    colors = np.tile(np.arange(t), b)
    colors = colors / max(1, t - 1)
    fig, ax = plt.subplots(figsize=figsize_for_grid(1, 1), constrained_layout=True)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=colors, cmap="viridis", s=10, alpha=0.7)
    if grid_overlay is not None:
        grid_proj = (grid_overlay.points - mean) @ vt[:2].T
        draw_grid_overlay(
            ax,
            grid_proj,
            grid_overlay.positions,
            grid_overlay.grid_rows,
            grid_overlay.grid_cols,
        )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=ax, label="Time step")
    path.parent.mkdir(parents=True, exist_ok=True)
    apply_square_axes(ax)
    fig.savefig(path, dpi=DEFAULT_DPI)
    plt.close(fig)


__all__ = ["save_embedding_projection"]
