"""Grid execution trace plot for planning diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


def save_grid_execution_trace_plot(
    out_path: Path,
    grid_rows: int,
    grid_cols: int,
    visited: Sequence[Tuple[int, int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.5, grid_cols - 0.5)
    ax.set_ylim(grid_rows - 0.5, -0.5)
    ax.set_xticks(range(grid_cols))
    ax.set_yticks(range(grid_rows))
    ax.grid(True, color="0.85", linestyle="-", linewidth=0.5)
    if visited:
        heat = np.zeros((grid_rows, grid_cols), dtype=np.float32)
        for row, col in visited:
            if 0 <= row < grid_rows and 0 <= col < grid_cols:
                heat[row, col] += 1.0
        ax.imshow(
            heat,
            cmap="Blues",
            interpolation="nearest",
            alpha=0.55,
            vmin=0.0,
            vmax=max(float(heat.max()), 1.0),
        )
        rows, cols = zip(*visited)
        ax.plot(cols, rows, marker="o", markersize=3, linewidth=1)
    ax.scatter([start[1]], [start[0]], s=60, marker="o", color="green", label="start")
    ax.scatter([goal[1]], [goal[0]], s=60, marker="x", color="red", label="goal")
    ax.legend(fontsize=8)
    ax.set_title("Execution trace")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
