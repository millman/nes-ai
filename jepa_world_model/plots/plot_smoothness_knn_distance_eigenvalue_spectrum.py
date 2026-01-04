"""Smoothness plot helper for vis-vs-ctrl diagnostics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from jepa_world_model.plot_raincloud import plot_raincloud
from jepa_world_model.vis_vis_ctrl_metrics import VisCtrlMetrics
from jepa_world_model.plots.plot_layout import DEFAULT_DPI, apply_square_axes, figsize_for_grid


def save_smoothness_knn_distance_eigenvalue_spectrum_plot(
    out_path: Path,
    metrics: VisCtrlMetrics,
    embedding_label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ks = sorted(metrics.knn_mean_distances.keys())
    fig, axes = plt.subplots(1, 2, figsize=figsize_for_grid(1, 2))
    if ks:
        samples = [metrics.knn_distance_samples.get(k, np.zeros(0, dtype=np.float32)) for k in ks]
        labels = [f"k={k}" for k in ks]
        plot_raincloud(axes[0], samples, labels, "kNN distance")
        axes[0].set_title(f"kNN distance raincloud ({embedding_label})")
    else:
        axes[0].text(0.5, 0.5, "No kNN data.", ha="center", va="center")
        axes[0].axis("off")
    eigvals = metrics.eigenvalues if metrics.eigenvalues.size else np.zeros(0, dtype=np.float32)
    if eigvals.size:
        axes[1].bar(np.arange(1, eigvals.size + 1), eigvals, color="tab:green", alpha=0.8)
        axes[1].set_xlabel("eigenvalue rank")
        axes[1].set_ylabel("eigenvalue")
        variance_text = metrics.global_variance
        if np.isfinite(variance_text):
            title = f"Eigenvalue spectrum ({embedding_label})\n(global variance = {variance_text:.4f})"
        else:
            title = f"Eigenvalue spectrum ({embedding_label})"
        axes[1].set_title(title)
    else:
        axes[1].text(0.5, 0.5, "No eigenvalues.", ha="center", va="center")
        axes[1].axis("off")
    apply_square_axes(axes)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
