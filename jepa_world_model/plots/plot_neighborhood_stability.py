"""Stability plot helper for vis-vs-ctrl diagnostics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from jepa_world_model.plot_raincloud import plot_raincloud
from jepa_world_model.vis_vis_ctrl_metrics import VisCtrlMetrics
from jepa_world_model.plots.plot_layout import apply_square_axes, figsize_for_grid


def save_neighborhood_stability_plot(
    out_path: Path,
    metrics: VisCtrlMetrics,
    embedding_label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ks = sorted(metrics.jaccard_means.keys())
    fig, ax = plt.subplots(figsize=figsize_for_grid(1, 1))
    if ks:
        samples = [metrics.jaccard_samples.get(k, np.zeros(0, dtype=np.float32)) for k in ks]
        labels = [f"k={k}" for k in ks]
        plot_raincloud(ax, samples, labels, "Jaccard")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Neighborhood stability ({embedding_label})")
    else:
        ax.text(0.5, 0.5, "No stability data.", ha="center", va="center")
        ax.axis("off")
    apply_square_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
