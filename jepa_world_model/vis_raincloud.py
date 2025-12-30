#!/usr/bin/env python3
"""Shared raincloud plotting helpers."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt


def plot_half_raincloud(
    ax: plt.Axes,
    data: Sequence[np.ndarray],
    labels: Sequence[str],
    ylabel: str,
    log_scale: bool = False,
    colors: Iterable[str] | None = None,
    mean_color: str = "tab:red",
) -> None:
    if all(arr.size == 0 for arr in data):
        ax.text(0.5, 0.5, "No samples available.", ha="center", va="center")
        ax.axis("off")
        return
    palette = list(colors) if colors is not None else ["tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:red"]
    violin_parts = ax.violinplot(
        data,
        showmeans=False,
        showextrema=False,
        showmedians=False,
        widths=0.7,
    )
    for idx, body in enumerate(violin_parts.get("bodies", [])):
        body.set_facecolor(palette[idx % len(palette)])
        body.set_edgecolor("black")
        body.set_alpha(0.25)
        verts = body.get_paths()[0].vertices
        center_x = float(np.mean(verts[:, 0]))
        verts[:, 0] = np.minimum(verts[:, 0], center_x)

    rng = np.random.default_rng(0)
    for idx, arr in enumerate(data):
        if arr.size == 0:
            continue
        jitter = rng.normal(loc=0.0, scale=0.04, size=arr.size)
        offset = 0.18
        ax.scatter(
            np.full_like(arr, idx + 1, dtype=np.float32) + offset + jitter,
            arr,
            s=10,
            alpha=0.25,
            color=palette[idx % len(palette)],
            edgecolor="none",
        )
        mean_val = float(np.mean(arr))
        line_center = idx + 1 + offset
        line_half_width = 0.12
        ax.plot(
            [line_center - line_half_width, line_center + line_half_width],
            [mean_val, mean_val],
            color=mean_color,
            linewidth=2,
            alpha=0.9,
        )

    ax.set_xticks(list(range(1, len(labels) + 1)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
