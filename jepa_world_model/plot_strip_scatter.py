#!/usr/bin/env python3
"""Strip plot helper."""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt


def plot_strip_scatter(
    ax: plt.Axes,
    samples: Sequence[np.ndarray],
    labels: Sequence[str],
    *,
    stat_fn: Callable[[np.ndarray], float] = lambda arr: float(np.mean(arr)),
    count_y_fn: Optional[Callable[[np.ndarray, float], float]] = None,
    jitter: float = 0.2,
    point_size: int = 14,
    point_alpha: float = 0.35,
    point_color: str = "tab:blue",
    line_color: str = "tab:red",
    line_width: float = 2.0,
    line_alpha: float = 0.9,
    count_fontsize: int = 7,
    x_label_rotation: int = 35,
    x_label_ha: str = "right",
) -> List[float]:
    if not samples:
        return []
    x = np.arange(len(samples))
    rng = np.random.default_rng(0)
    stats: List[float] = []
    for idx, arr in enumerate(samples):
        if arr is None or arr.size == 0:
            continue
        jitter_vals = rng.uniform(-jitter, jitter, size=arr.shape[0])
        ax.scatter(
            np.full_like(arr, x[idx], dtype=np.float32) + jitter_vals,
            arr,
            s=point_size,
            alpha=point_alpha,
            color=point_color,
            edgecolors="none",
        )
        stat_val = float(stat_fn(arr))
        stats.append(stat_val)
        ax.plot(
            [x[idx] - 0.25, x[idx] + 0.25],
            [stat_val, stat_val],
            color=line_color,
            linewidth=line_width,
            alpha=line_alpha,
        )
        if count_y_fn is not None:
            count_y = float(count_y_fn(arr, stat_val))
            ax.text(x[idx], count_y, f"n={arr.shape[0]}", ha="center", va="bottom", fontsize=count_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=x_label_rotation, ha=x_label_ha, fontsize=9)
    return stats


__all__ = ["plot_strip_scatter"]
