"""Planning graph visualization with PCA projection."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


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
    proj = centered @ vt[:2].T
    denom = max(sample.shape[0] - 1, 1)
    variances = (s**2) / denom
    total = float(variances.sum())
    if total <= 0:
        ratios = np.array([0.0, 0.0], dtype=np.float32)
    else:
        ratios = (variances / total)[:2]
    return proj, (mean, vt[:2], ratios)


def _project_with_pca(points: np.ndarray, pca: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    mean, comps, _ = pca
    return (points - mean) @ comps.T


def _edge_pairs(edges: Dict[int, Dict[str, List[int]]]) -> Iterable[Tuple[int, int]]:
    for src, by_action in edges.items():
        for dsts in by_action.values():
            for dst in dsts:
                yield int(src), int(dst)


def save_planning_graph_plot(
    out_path: Path,
    samples: np.ndarray,
    centers: np.ndarray,
    edges: Dict[int, Dict[str, List[int]]],
    *,
    title: str,
    max_samples: int,
    max_edges: int,
) -> None:
    if samples.ndim != 2:
        raise AssertionError("samples must be 2D for plotting.")
    if centers.ndim != 2:
        raise AssertionError("centers must be 2D for plotting.")
    if centers.shape[0] == 0:
        raise AssertionError("centers must be non-empty for plotting.")
    if max_samples <= 0:
        raise AssertionError("max_samples must be positive.")
    if max_edges < 0:
        raise AssertionError("max_edges must be non-negative.")

    proj_samples, pca = _pca_project(samples, max_samples=max_samples)
    proj_centers = _project_with_pca(centers, pca)
    _, _, ratios = pca

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(proj_samples[:, 0], proj_samples[:, 1], s=6, alpha=0.2, label="samples")
    ax.scatter(proj_centers[:, 0], proj_centers[:, 1], s=30, alpha=0.9, label="centers")

    edge_list = list(_edge_pairs(edges))
    if max_edges and len(edge_list) > max_edges:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(edge_list), size=max_edges, replace=False)
        edge_list = [edge_list[i] for i in idx]

    for src, dst in edge_list:
        if src >= proj_centers.shape[0] or dst >= proj_centers.shape[0]:
            continue
        xs = [proj_centers[src, 0], proj_centers[dst, 0]]
        ys = [proj_centers[src, 1], proj_centers[dst, 1]]
        ax.plot(xs, ys, color="tab:gray", alpha=0.25, linewidth=0.8)

    ax.set_xlabel(f"PC1 ({ratios[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ratios[1] * 100:.1f}%)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
