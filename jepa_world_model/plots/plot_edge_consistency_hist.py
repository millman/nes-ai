"""Edge consistency histogram plot helper for graph diagnostics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def save_edge_consistency_hist_plot(
    out_path: Path,
    edge_errors: np.ndarray,
    embedding_label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(edge_errors, bins=30, color="tab:gray", alpha=0.85)
    ax.set_title(f"Predictor-edge consistency (||{embedding_label}hat - {embedding_label}T||^2)")
    ax.set_xlabel("error")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
