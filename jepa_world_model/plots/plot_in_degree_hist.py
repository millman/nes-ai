"""In-degree histogram plot helper for graph diagnostics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from jepa_world_model.plots.plot_layout import DEFAULT_DPI, apply_square_axes, figsize_for_grid

def save_in_degree_hist_plot(out_path: Path, in_degree: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize_for_grid(1, 1))
    if in_degree.size == 0:
        ax.text(0.5, 0.5, "No edges to compute in-degree.", ha="center", va="center")
    else:
        bins = min(50, max(5, int(np.sqrt(in_degree.size))))
        ax.hist(in_degree, bins=bins, color="tab:purple", alpha=0.8)
        ax.set_xlabel("In-degree (top-K graph)")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
    ax.set_title("Hubness / in-degree distribution")
    apply_square_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
