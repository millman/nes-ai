"""Planning lattice visualization with PCA projection."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from jepa_world_model.plots.plot_grid_overlay import GridOverlay, draw_grid_overlay

def _pca_project(
    points: np.ndarray,
    *,
    max_samples: int,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if points.ndim != 2:
        raise AssertionError("points must be 2D for PCA.")
    if points.shape[0] == 0:
        raise AssertionError("points must be non-empty for PCA.")
    sample = points
    if points.shape[0] > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(points.shape[0], size=max_samples, replace=False)
        sample = points[idx]
    mean = sample.mean(axis=0, keepdims=True)
    centered = sample - mean
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    num_components = min(2, vt.shape[0])
    comps = vt[:num_components]
    if num_components < 2:
        pad = np.zeros((2 - num_components, comps.shape[1]), dtype=comps.dtype)
        comps = np.vstack([comps, pad])
    proj = centered @ comps.T
    denom = max(sample.shape[0] - 1, 1)
    variances = (s**2) / denom
    total = float(variances.sum())
    if total <= 0:
        ratios = np.array([0.0, 0.0], dtype=np.float32)
    else:
        ratios = (variances / total)[:num_components]
        if ratios.shape[0] < 2:
            ratios = np.pad(ratios, (0, 2 - ratios.shape[0]))
    return proj, (mean, comps, ratios)


def _project_with_pca(points: np.ndarray, pca: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    mean, comps, _ = pca
    return (points - mean) @ comps.T


def save_planning_lattice_plot(
    out_path: Path,
    nodes: np.ndarray,
    edges: np.ndarray,
    *,
    title: str,
    max_samples: int,
    max_edges: int,
    grid_overlay: Optional[GridOverlay] = None,
) -> None:
    if nodes.ndim != 2:
        raise AssertionError("nodes must be 2D for plotting.")
    if nodes.shape[0] == 0:
        raise AssertionError("nodes must be non-empty for plotting.")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise AssertionError("edges must be shaped (N, 2).")
    if max_samples <= 0:
        raise AssertionError("max_samples must be positive.")
    if max_edges < 0:
        raise AssertionError("max_edges must be non-negative.")

    _, pca = _pca_project(nodes, max_samples=max_samples)
    proj_nodes = _project_with_pca(nodes, pca)
    _, _, ratios = pca

    edge_list = edges.tolist()
    if max_edges and len(edge_list) > max_edges:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(edge_list), size=max_edges, replace=False)
        edge_list = [edge_list[i] for i in idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(proj_nodes[:, 0], proj_nodes[:, 1], s=8, alpha=0.35, label="nodes")

    if grid_overlay is not None:
        grid_proj = _project_with_pca(grid_overlay.points, pca)
        draw_grid_overlay(
            ax,
            grid_proj,
            grid_overlay.positions,
            grid_overlay.grid_rows,
            grid_overlay.grid_cols,
        )

    for src, dst in edge_list:
        if src >= proj_nodes.shape[0] or dst >= proj_nodes.shape[0]:
            continue
        xs = [proj_nodes[src, 0], proj_nodes[dst, 0]]
        ys = [proj_nodes[src, 1], proj_nodes[dst, 1]]
        ax.plot(xs, ys, color="tab:gray", alpha=0.2, linewidth=0.8)

    ax.set_xlabel(f"PC1 ({ratios[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ratios[1] * 100:.1f}%)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
