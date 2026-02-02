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
    def _safe_hist(ax, values: np.ndarray, bins: int, label: str) -> None:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return
        vmin = float(finite.min())
        vmax = float(finite.max())
        if np.isclose(vmin, vmax):
            ax.hist([vmin], bins=1, alpha=0.6, label=label)
        else:
            ax.hist(finite, bins=bins, alpha=0.6, label=label)

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
        _safe_hist(ax, norms, bins=30, label="||d||")
        _safe_hist(ax, cos, bins=30, label="cos(d,mu)")
        ax.set_title(action)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_action_delta_strip_plot(
    out_path: Path,
    deltas: np.ndarray,
    labels: Sequence[Optional[str]],
    mu: Dict[str, np.ndarray],
) -> None:
    actions = ("L", "R", "U", "D")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    rng = np.random.default_rng(0)
    for ax, metric in zip(axes, ("norm", "cos")):
        for idx, action in enumerate(actions):
            mask = np.array([lbl == action for lbl in labels], dtype=bool)
            if not mask.any():
                continue
            d = deltas[mask]
            if metric == "norm":
                values = np.linalg.norm(d, axis=1)
                ylabel = "||d_p||"
            else:
                mu_vec = mu[action]
                denom = np.maximum(np.linalg.norm(d, axis=1) * np.linalg.norm(mu_vec), 1e-8)
                values = (d @ mu_vec) / denom
                ylabel = "cos(d_p, mu[a])"
            jitter = rng.normal(scale=0.08, size=values.shape[0])
            ax.scatter(np.full(values.shape, idx) + jitter, values, s=10, alpha=0.5)
            mean_val = float(np.mean(values))
            ax.plot(
                [idx - 0.18, idx + 0.18],
                [mean_val, mean_val],
                color="tab:red",
                linewidth=2,
                zorder=3,
            )
        ax.set_xticks(range(len(actions)))
        ax.set_xticklabels(actions)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by action")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
