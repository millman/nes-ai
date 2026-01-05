"""Variance spectrum plot helper for diagnostics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from jepa_world_model.plots.plot_layout import DEFAULT_DPI, apply_square_axes, figsize_for_grid

def save_variance_spectrum_plot(
    variance_ratio: np.ndarray,
    out_dir: Path,
    global_step: int,
    embedding_label: str,
) -> None:
    out_path = out_dir / f"delta_{embedding_label}_variance_spectrum_{global_step:07d}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    k = min(len(variance_ratio), 32)
    if k == 0:
        return
    fig, ax = plt.subplots(figsize=figsize_for_grid(1, 1), constrained_layout=True)
    x = np.arange(k)
    ax.bar(x, variance_ratio[:k], color="tab:blue", alpha=0.8, label="variance ratio")
    cumulative = np.cumsum(variance_ratio[:k])
    ax.plot(x, cumulative, color="tab:red", marker="o", linewidth=2, label="cumulative")
    ax.set_xlabel("component")
    ax.set_ylabel("variance ratio")
    ax.set_ylim(0, max(1.05, float(cumulative[-1]) + 0.05))
    ax.set_title("Motion PCA spectrum")
    ax.legend()
    apply_square_axes(ax)
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)
