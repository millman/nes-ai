"""Reachable fraction histogram plot for planning diagnostics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def save_reachable_fraction_hist_plot(out_path: Path, values: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    if values.size:
        ax.hist(values, bins=20, color="tab:blue", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("reachable fraction")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
