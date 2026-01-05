"""Composition error plot helper for vis-vs-ctrl diagnostics."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jepa_world_model.vis_vis_ctrl_metrics import VisCtrlMetrics
from jepa_world_model.plots.plot_layout import DEFAULT_DPI, apply_square_axes, figsize_for_grid


def save_two_step_composition_error_plot(
    out_path: Path,
    metrics: VisCtrlMetrics,
    embedding_label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize_for_grid(1, 1), constrained_layout=True)
    if metrics.composition_errors.size:
        ax.hist(metrics.composition_errors, bins=20, color="tab:purple", alpha=0.8)
        ax.set_xlabel("two-step error")
        ax.set_ylabel("count")
        mean_val = metrics.composition_error_mean
        if np.isfinite(mean_val):
            ax.axvline(mean_val, color="tab:red", linestyle="--", linewidth=1.5)
            ax.set_title(f"Two-step composition error ({embedding_label})")
        else:
            ax.set_title(f"Two-step composition error ({embedding_label})")
    else:
        ax.text(0.5, 0.5, "No composition errors.", ha="center", va="center")
        ax.axis("off")
    apply_square_axes(ax)
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)
