"""Graph diagnostics history plot helper."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from jepa_world_model.plots.plot_layout import figsize_for_grid

def save_graph_history_plot(out_path: Path, history: List[Dict[str, float]], k: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=figsize_for_grid(3, 1), sharex=True)
    if not history:
        for ax in axes:
            ax.text(0.5, 0.5, "History unavailable.", ha="center", va="center")
            ax.axis("off")
    else:
        steps = [row.get("step", float("nan")) for row in history]
        hit1 = [row.get("hit1_at_k", float("nan")) for row in history]
        hit2 = [row.get("hit2_at_k", float("nan")) for row in history]
        median_neff1 = [row.get("median_neff1", float("nan")) for row in history]
        median_neff2 = [row.get("median_neff2", float("nan")) for row in history]
        ratio = [row.get("neff_ratio", float("nan")) for row in history]
        long_gap = [row.get("long_gap_rate", float("nan")) for row in history]
        mutual = [row.get("mutual_rate", float("nan")) for row in history]
        max_in = [row.get("max_in_degree", float("nan")) for row in history]

        axes[0].plot(steps, hit1, marker="o", label=f"hit1@{k}", color="tab:blue")
        axes[0].plot(steps, hit2, marker="o", label=f"hit2@{k}", color="tab:orange")
        axes[0].set_ylabel("Hit rate")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(steps, median_neff1, marker="o", label="median Neff1", color="tab:green")
        axes[1].plot(steps, median_neff2, marker="o", label="median Neff2", color="tab:red")
        axes[1].plot(steps, ratio, marker="o", label="Neff2/Neff1", color="tab:purple")
        axes[1].set_ylabel("Neighborhood size")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(steps, long_gap, marker="o", label="long-gap rate", color="tab:brown")
        axes[2].plot(steps, mutual, marker="o", label="mutual kNN rate", color="tab:cyan")
        axes[2].plot(steps, max_in, marker="o", label="max in-degree", color="tab:gray")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Graph health")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
