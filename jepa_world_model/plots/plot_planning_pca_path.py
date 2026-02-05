"""PCA projection plot for planning paths."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from jepa_world_model.plots.plot_grid_overlay import GridOverlay, draw_grid_overlay

def _pca_project(points: np.ndarray, *, max_samples: int) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:2].T
    return proj, (mean, vt[:2])


def _project_with_pca(points: np.ndarray, pca: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mean, comps = pca
    return (points - mean) @ comps.T


def save_planning_pca_path_plot(
    out_path: Path,
    p_points: np.ndarray,
    plan_nodes: Optional[np.ndarray],
    start: np.ndarray,
    goal: np.ndarray,
    *,
    max_samples: int,
    grid_overlay: Optional[GridOverlay] = None,
    title: str = "PCA(p) plan",
) -> None:
    proj, pca = _pca_project(p_points, max_samples=max_samples)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(proj[:, 0], proj[:, 1], s=8, alpha=0.3, label="dataset")
    if grid_overlay is not None:
        grid_proj = _project_with_pca(grid_overlay.points, pca)
        draw_grid_overlay(
            ax,
            grid_proj,
            grid_overlay.positions,
            grid_overlay.grid_rows,
            grid_overlay.grid_cols,
        )
    start_proj = _project_with_pca(start[None, :], pca)[0]
    goal_proj = _project_with_pca(goal[None, :], pca)[0]
    ax.scatter([start_proj[0]], [start_proj[1]], s=60, marker="o", label="start")
    ax.scatter([goal_proj[0]], [goal_proj[1]], s=60, marker="x", label="goal")
    if plan_nodes is not None and plan_nodes.size:
        path_proj = _project_with_pca(plan_nodes, pca)
        ax.plot(path_proj[:, 0], path_proj[:, 1], color="tab:orange", linewidth=2, label="plan")
    ax.legend(fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
