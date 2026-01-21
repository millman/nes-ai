from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from jepa_world_model.plots.plot_layout import DEFAULT_DPI, figsize_for_grid


def save_off_manifold_visualization(
    out_dir: Path,
    step_indices: np.ndarray,
    errors: np.ndarray,
    global_step: int,
) -> None:
    if step_indices.size == 0 or errors.size == 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=figsize_for_grid(1, 2), constrained_layout=True)
    rng = np.random.default_rng(0)
    jitter = rng.normal(scale=0.06, size=step_indices.shape)
    axes[0].scatter(step_indices + jitter, errors, s=10, alpha=0.6, color="tab:blue")
    axes[0].set_title("Off-manifold error by rollout index")
    axes[0].set_xlabel("rollout index")
    axes[0].set_ylabel("||enc(dec(z_roll)) - z_roll||")
    axes[1].hist(errors, bins=30, color="tab:orange", alpha=0.8)
    axes[1].set_title("Off-manifold error histogram")
    axes[1].set_xlabel("error")
    axes[1].set_ylabel("count")
    out_path = out_dir / f"off_manifold_{global_step:07d}.png"
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)
