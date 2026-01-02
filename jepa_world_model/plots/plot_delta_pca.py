"""Delta PCA plot helper for diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from jepa_world_model.actions import decode_action_id


def save_delta_pca_plot(
    out_path: Path,
    variance_ratio: np.ndarray,
    delta_proj: np.ndarray,
    proj_flat: np.ndarray,
    action_ids: np.ndarray,
    action_dim: int,
    embedding_label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    num_var = min(10, variance_ratio.shape[0])
    axes[0, 0].bar(np.arange(num_var), variance_ratio[:num_var], color="tab:blue")
    axes[0, 0].set_title(f"Delta-{embedding_label} PCA variance ratio")
    axes[0, 0].set_xlabel("component")
    axes[0, 0].set_ylabel("explained variance")

    if delta_proj.shape[1] >= 2:
        unique_actions = sorted({int(a) for a in np.asarray(action_ids).reshape(-1)})
        action_to_index = {aid: idx for idx, aid in enumerate(unique_actions)}
        color_indices = (
            np.array([action_to_index.get(int(a), 0) for a in np.asarray(action_ids).reshape(-1)], dtype=np.float32)
            if unique_actions
            else np.asarray(action_ids, dtype=np.float32)
        )
        palette = plt.get_cmap("tab20").colors
        color_count = max(1, min(len(palette), len(unique_actions) if unique_actions else 1))
        color_list = list(palette[:color_count])
        cmap = mcolors.ListedColormap(color_list)
        bounds = np.arange(color_count + 1) - 0.5
        color_indices_mapped = np.mod(color_indices, color_count) if color_count else color_indices
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        scatter = axes[0, 1].scatter(
            delta_proj[:, 0],
            delta_proj[:, 1],
            c=color_indices_mapped,
            cmap=cmap,
            norm=norm,
            s=8,
            alpha=0.7,
        )
        axes[0, 1].set_xlabel(f"PC1 (delta {embedding_label})")
        axes[0, 1].set_ylabel(f"PC2 (delta {embedding_label})")
        cbar = fig.colorbar(scatter, ax=axes[0, 1], fraction=0.046, pad=0.04, boundaries=bounds)
        ticks = list(range(color_count))
        cbar.set_ticks(ticks)
        tick_labels = (
            [decode_action_id(aid, action_dim) for aid in unique_actions[:color_count]] if unique_actions else ["NOOP"]
        )
        cbar.set_ticklabels(tick_labels)
        cbar.set_label("action")
    else:
        axes[0, 1].plot(delta_proj[:, 0], np.zeros_like(delta_proj[:, 0]), ".", alpha=0.6)
        axes[0, 1].set_xlabel(f"PC1 (delta {embedding_label})")
        axes[0, 1].set_ylabel("density")
    axes[0, 1].set_title(f"Delta-{embedding_label} projections")

    cumulative = np.cumsum(variance_ratio)
    axes[1, 0].plot(np.arange(len(cumulative)), cumulative, marker="o", color="tab:green")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_xlabel("component")
    axes[1, 0].set_ylabel("cumulative variance")
    axes[1, 0].set_title("Cumulative explained variance")

    if proj_flat.shape[1] >= 2:
        t = np.linspace(0, 1, num=proj_flat.shape[0])
        sc2 = axes[1, 1].scatter(proj_flat[:, 0], proj_flat[:, 1], c=t, cmap="viridis", s=6, alpha=0.6)
        axes[1, 1].set_xlabel(f"PC1 ({embedding_label})")
        axes[1, 1].set_ylabel(f"PC2 ({embedding_label})")
        fig.colorbar(sc2, ax=axes[1, 1], fraction=0.046, pad=0.04, label="time (normalized)")
    else:
        axes[1, 1].plot(proj_flat[:, 0], np.zeros_like(proj_flat[:, 0]), ".", alpha=0.6)
        axes[1, 1].set_xlabel(f"PC1 ({embedding_label})")
        axes[1, 1].set_ylabel("density")
    upper_label = embedding_label.upper()
    axes[1, 1].set_title(f"PCA of {upper_label} on motion-defined Δ{upper_label} basis")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
