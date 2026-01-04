#!/usr/bin/env python3
"""Visualization helpers for action alignment diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt

from jepa_world_model.actions import decode_action_id
from jepa_world_model.plots.plot_layout import DEFAULT_DPI, apply_square_axes, figsize_for_grid
from jepa_world_model.plot_strip_scatter import plot_strip_scatter


def save_action_alignment_plot(
    out_path: Path,
    stats: List[Dict[str, Any]],
    cosine_high_threshold: float,
    action_dim: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not stats:
        fig, ax = plt.subplots(figsize=figsize_for_grid(1, 1))
        ax.text(0.5, 0.5, "No actions met alignment criteria.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
        return
    labels = [decode_action_id(s["action_id"], action_dim) for s in stats]
    samples: List[np.ndarray] = []
    for stat in stats:
        cos_values = stat.get("cosines")
        samples.append(np.asarray([] if cos_values is None else cos_values, dtype=np.float32))
    fig, ax = plt.subplots(figsize=figsize_for_grid(1, 1))
    plot_strip_scatter(
        ax,
        samples,
        labels,
        stat_fn=lambda arr: float(np.mean(arr)),
        count_y_fn=lambda arr, stat: 1.02,
        point_size=18,
        count_fontsize=8,
    )

    ax.axhline(cosine_high_threshold, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_ylabel("cosine alignment")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Action-conditioned cosine alignment (strip plot)")
    apply_square_axes(ax)
    apply_square_axes(axes)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


def save_action_alignment_detail_plot(
    out_path: Path,
    debug_data: Dict[str, Any],
    cosine_high_threshold: float,
    action_dim: int,
    alignment_label: str = "PCA",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=figsize_for_grid(2, 2))

    overall_cos_raw = debug_data.get("overall_cos")
    overall_norms_raw = debug_data.get("overall_norms")
    per_action_cos: Dict[int, np.ndarray] = debug_data.get("per_action_cos") or {}
    pairwise_raw = debug_data.get("pairwise")
    overall_cos = np.asarray([] if overall_cos_raw is None else overall_cos_raw, dtype=np.float32)
    overall_norms = np.asarray([] if overall_norms_raw is None else overall_norms_raw, dtype=np.float32)
    pairwise = np.asarray([] if pairwise_raw is None else pairwise_raw, dtype=np.float32)
    actions_sorted: List[int] = list(debug_data.get("actions_sorted") or [])
    per_action_norms: Dict[int, np.ndarray] = debug_data.get("per_action_norms") or {}

    label_suffix = f"{alignment_label.strip()}, " if alignment_label.strip() else ""

    # (0,0): per-action cosine strip plot (mirrors cosine alignment view)
    ax0 = axes[0, 0]
    if actions_sorted and any(per_action_cos.get(aid) is not None for aid in actions_sorted):
        actions_sorted = sorted(actions_sorted)
        labels = [decode_action_id(aid, action_dim) for aid in actions_sorted]
        samples = []
        for aid in actions_sorted:
            values = per_action_cos.get(aid)
            samples.append(np.asarray([] if values is None else values, dtype=np.float32))
        plot_strip_scatter(
            ax0,
            samples,
            labels,
            stat_fn=lambda arr: float(np.mean(arr)),
            count_y_fn=lambda arr, stat: 1.02,
            point_size=14,
        )
        ax0.axhline(cosine_high_threshold, color="gray", linestyle="--", linewidth=1, alpha=0.8)
        ax0.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.8)
        ax0.set_ylabel("cosine alignment")
        ax0.set_ylim(-1.05, 1.05)
        ax0.set_title(f"Cosine alignment ({label_suffix}per-action strip)")
    else:
        ax0.text(0.5, 0.5, "No valid cosine samples.", ha="center", va="center")
        ax0.axis("off")

    # (0,1): scatter of cosine vs delta norm
    ax1 = axes[0, 1]
    if actions_sorted and per_action_norms:
        scatter_x = np.asarray([], dtype=np.float32)
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        color_ids: List[np.ndarray] = []
        color_actions = sorted(actions_sorted)
        palette = plt.get_cmap("tab20").colors
        color_count = max(1, min(len(palette), len(color_actions)))
        cmap = mcolors.ListedColormap(list(palette[:color_count]))
        bounds = np.arange(color_count + 1) - 0.5
        norm = mcolors.BoundaryNorm(bounds, color_count)
        color_map = {aid: (idx % color_count) for idx, aid in enumerate(color_actions)}
        for idx, aid in enumerate(actions_sorted):
            norms = per_action_norms.get(aid)
            cos_vals = per_action_cos.get(aid)
            if norms is None or cos_vals is None or norms.size == 0 or cos_vals.size == 0:
                continue
            count = min(norms.shape[0], cos_vals.shape[0])
            xs.append(norms[:count])
            ys.append(cos_vals[:count])
            color_ids.append(np.full(count, color_map.get(aid, idx), dtype=np.float32))
        if xs and ys and color_ids:
            scatter_x = np.concatenate(xs)
            scatter_y = np.concatenate(ys)
            scatter_c = np.concatenate(color_ids)
            sc = ax1.scatter(
                scatter_x,
                scatter_y,
                c=scatter_c,
                cmap=cmap,
                norm=norm,
                s=8,
                alpha=0.35,
                edgecolors="none",
            )
            cbar = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04, boundaries=bounds)
            ticks = list(range(color_count))
            cbar.set_ticks(ticks)
            tick_labels = [decode_action_id(aid, action_dim) for aid in color_actions[:color_count]]
            cbar.set_ticklabels(tick_labels)
            cbar.set_label("action")
        else:
            scatter_x = np.asarray([], dtype=np.float32)
        ax1.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax1.axhline(cosine_high_threshold, color="tab:green", linestyle="--", linewidth=1)
        if scatter_x.size and np.all(scatter_x > 0):
            ax1.set_xscale("log")
        ax1.set_xlabel("delta norm")
        ax1.set_ylabel("cosine alignment")
        ax1.set_title(f"Alignment vs. delta magnitude ({alignment_label.strip()})")
    else:
        ax1.text(0.5, 0.5, "Insufficient samples for cosine/norm scatter.", ha="center", va="center")
        ax1.axis("off")

    # (1,0): per-action delta norm strip plot
    ax2 = axes[1, 0]
    if actions_sorted and any(per_action_norms.get(aid) is not None for aid in actions_sorted):
        labels = [decode_action_id(aid, action_dim) for aid in actions_sorted]
        samples = []
        for aid in actions_sorted:
            values = per_action_norms.get(aid)
            samples.append(np.asarray([] if values is None else values, dtype=np.float32))
        medians = plot_strip_scatter(
            ax2,
            samples,
            labels,
            stat_fn=lambda arr: float(np.median(arr)),
            count_y_fn=lambda arr, stat: stat,
            point_size=10,
        )
        if medians and all(m > 0 for m in medians):
            ax2.set_yscale("log")
        ax2.set_ylabel("delta norm")
        ax2.set_title(f"Delta norm by action ({alignment_label.strip()})")
    else:
        ax2.text(0.5, 0.5, "No usable per-action norms.", ha="center", va="center")
        ax2.axis("off")

    # (1,1): pairwise similarity heatmap of mean directions
    ax3 = axes[1, 1]
    if pairwise.size and actions_sorted:
        im = ax3.imshow(pairwise, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        labels = [decode_action_id(aid, action_dim) for aid in actions_sorted]
        ax3.set_xticks(np.arange(len(actions_sorted)))
        ax3.set_yticks(np.arange(len(actions_sorted)))
        ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax3.set_yticklabels(labels, fontsize=8)
        ax3.set_title(f"Mean direction similarity ({alignment_label.strip()})")
        fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label="cosine")
    else:
        ax3.text(0.5, 0.5, "Mean direction similarity unavailable.", ha="center", va="center")
        ax3.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
