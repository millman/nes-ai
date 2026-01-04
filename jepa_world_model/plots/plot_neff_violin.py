"""Neighborhood size violin plot helper for graph diagnostics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from jepa_world_model.plot_raincloud import plot_raincloud
from jepa_world_model.plots.plot_layout import DEFAULT_DPI, apply_square_axes, figsize_for_grid


def save_neff_violin_plot(out_path: Path, neff1: np.ndarray, neff2: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize_for_grid(1, 1))

    def _clean(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return finite
        return finite[finite > 0]

    data = [_clean(neff1), _clean(neff2)]
    labels = ["Neff1", "Neff2"]
    if all(arr.size == 0 for arr in data):
        ax.text(0.5, 0.5, "No neighborhood stats available.", ha="center", va="center")
    else:
        plot_raincloud(
            ax,
            data,
            labels,
            "Effective neighborhood size",
            log_scale=True,
            colors=["tab:blue", "tab:green"],
        )
    ax.set_title("Neighborhood size (exp entropy)")
    apply_square_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
