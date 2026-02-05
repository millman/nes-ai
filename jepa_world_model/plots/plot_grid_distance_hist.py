"""Histogram of nearest-grid distances in latent space."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def save_grid_distance_hist(
    out_path: Path,
    distances: np.ndarray,
    *,
    title: str,
    l2_distances: np.ndarray | None = None,
) -> None:
    if l2_distances is None:
        fig, ax = plt.subplots(figsize=(5, 4))
        if distances.size:
            ax.hist(distances, bins=30, color="tab:blue", alpha=0.75)
        ax.set_title(title)
        ax.set_xlabel("cosine distance")
        ax.set_ylabel("count")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        if distances.size:
            axes[0].hist(distances, bins=30, color="tab:blue", alpha=0.75)
        axes[0].set_title("Cosine")
        axes[0].set_xlabel("cosine distance")
        axes[0].set_ylabel("count")
        if l2_distances.size:
            axes[1].hist(l2_distances, bins=30, color="tab:orange", alpha=0.75)
        axes[1].set_title("L2")
        axes[1].set_xlabel("l2 distance")
        axes[1].set_ylabel("count")
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
