"""Rank CDF plot helper for graph diagnostics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def save_rank_cdf_plot(out_path: Path, ranks: np.ndarray, k: int, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    if ranks.size == 0:
        ax.text(0.5, 0.5, "No valid transitions.", ha="center", va="center")
    else:
        ranks_sorted = np.sort(ranks)
        y = np.arange(1, len(ranks_sorted) + 1) / len(ranks_sorted)
        ax.plot(ranks_sorted, y, label="CDF", color="tab:blue")
        if k > 0:
            ax.axvline(k, color="tab:orange", linestyle="--", label=f"K={k}")
        ax.set_xscale("log")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Fraction <= rank")
        ax.grid(True, alpha=0.3)
        ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
