"""Scatter plot for ||z_i - z_j|| vs ||p_i - p_j||."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from jepa_world_model.plots.plot_layout import DEFAULT_DPI


def save_zp_distance_scatter(
    out_path: Path,
    z_distances: Sequence[float],
    p_distances: Sequence[float],
    title: str = "Z vs P distance (fixed pairs)",
) -> None:
    z_vals = np.asarray(z_distances, dtype=np.float32)
    p_vals = np.asarray(p_distances, dtype=np.float32)
    if z_vals.shape != p_vals.shape:
        raise AssertionError("z_distances and p_distances must share shape.")
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
    ax.scatter(z_vals, p_vals, s=8, alpha=0.35, color="#4c72b0", edgecolors="none")
    ax.set_xlabel("||z_i - z_j||")
    ax.set_ylabel("||p_i - p_j||")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)
