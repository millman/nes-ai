"""Action delta statistics plotting for planning diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt


def save_action_delta_stats_plot(
    out_path: Path,
    deltas: np.ndarray,
    labels: Sequence[Optional[str]],
    mu: Dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    axes = axes.flatten()
    for idx, action in enumerate(("L", "R", "U", "D")):
        ax = axes[idx]
        mask = np.array([lbl == action for lbl in labels], dtype=bool)
        if not mask.any():
            ax.set_title(f"{action} (no data)")
            continue
        d = deltas[mask]
        norms = np.linalg.norm(d, axis=1)
        mu_vec = mu[action]
        denom = np.maximum(np.linalg.norm(d, axis=1) * np.linalg.norm(mu_vec), 1e-8)
        cos = (d @ mu_vec) / denom
        ax.hist(norms, bins=30, alpha=0.6, label="||d||")
        ax.hist(cos, bins=30, alpha=0.6, label="cos(d,mu)")
        ax.set_title(action)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
