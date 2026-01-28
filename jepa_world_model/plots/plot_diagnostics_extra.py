"""Extra diagnostics plot helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from jepa_world_model.plots.plot_layout import DEFAULT_DPI


@dataclass
class StraightLineTrajectory:
    points: np.ndarray  # [T, 2]
    label: str
    color: str


def save_rollout_divergence_plot(
    out_path: Path,
    horizons: Sequence[int],
    pixel_errors: Sequence[float],
    latent_errors: Sequence[float],
    pixel_label: str = "Pixel error (MSE)",
    latent_label: str = "Latent error",
    title: str = "Rollout divergence",
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.2))
    ax.plot(horizons, pixel_errors, marker="o", label=pixel_label)
    ax.plot(horizons, latent_errors, marker="o", label=latent_label)
    ax.set_xlabel("Rollout step k")
    ax.set_ylabel("Error")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)


def save_ablation_divergence_plot(
    out_path: Path,
    horizons: Sequence[int],
    pixel_errors: Sequence[float],
    pixel_errors_zero: Sequence[float],
    latent_errors: Sequence[float],
    latent_errors_zero: Sequence[float],
    title: str = "H ablation divergence",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].plot(horizons, pixel_errors, marker="o", label="normal")
    axes[0].plot(horizons, pixel_errors_zero, marker="o", label="h=0")
    axes[0].set_title("Pixel error")
    axes[0].set_xlabel("Rollout step k")
    axes[0].set_ylabel("Error")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(horizons, latent_errors, marker="o", label="normal")
    axes[1].plot(horizons, latent_errors_zero, marker="o", label="h=0")
    axes[1].set_title("Latent error")
    axes[1].set_xlabel("Rollout step k")
    axes[1].set_ylabel("Error")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)


def save_straightline_plot(
    out_path: Path,
    trajectories: Sequence[StraightLineTrajectory],
    title: str = "Straight-line action rays",
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 4.2))
    for traj in trajectories:
        pts = traj.points
        if pts.shape[0] < 2:
            continue
        ax.plot(pts[:, 0], pts[:, 1], marker="o", markersize=3, color=traj.color, label=traj.label)
        ax.annotate(
            "",
            xy=(pts[-1, 0], pts[-1, 1]),
            xytext=(pts[-2, 0], pts[-2, 1]),
            arrowprops=dict(arrowstyle="->", color=traj.color, lw=1),
        )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)


def save_z_consistency_plot(
    out_path: Path,
    distances: Sequence[float],
    cosines: Sequence[float],
    title: str = "Z same-frame consistency",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].hist(distances, bins=24, color="#4c72b0", alpha=0.9)
    axes[0].set_title("||z - mean(z)||")
    axes[0].set_xlabel("Distance")
    axes[0].set_ylabel("Count")

    axes[1].hist(cosines, bins=24, color="#55a868", alpha=0.9)
    axes[1].set_title("cos(z, mean(z))")
    axes[1].set_xlabel("Cosine")
    axes[1].set_ylabel("Count")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)


def save_monotonicity_plot(
    out_path: Path,
    shifts: Sequence[int],
    distances: Sequence[float],
    title: str = "Z distance vs pixel shift",
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.2))
    ax.plot(shifts, distances, marker="o")
    ax.set_xlabel("Pixel shift (d)")
    ax.set_ylabel("E[||z(x) - z(shift(x,d))||]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)


def save_path_independence_plot(
    out_path: Path,
    labels: Sequence[str],
    z_distances: Sequence[float],
    s_distances: Sequence[float],
    title: str = "Path independence",
) -> None:
    idx = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.2))
    ax.bar(idx - width / 2, z_distances, width, label="Z distance")
    ax.bar(idx + width / 2, s_distances, width, label="S distance")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Distance")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)


def save_drift_by_action_plot(
    out_path: Path,
    labels: Sequence[str],
    drift_samples: Sequence[Sequence[float]],
    title: str = "H drift by action",
) -> None:
    idx = np.arange(len(labels))
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 3.2))
    for i, samples in enumerate(drift_samples):
        if not samples:
            continue
        samples_np = np.asarray(samples, dtype=np.float32)
        if samples_np.size > 200:
            take = np.linspace(0, samples_np.size - 1, num=200, dtype=int)
            samples_np = samples_np[take]
        jitter = (np.random.rand(samples_np.size) - 0.5) * 0.2
        ax.scatter(idx[i] + jitter, samples_np, s=10, alpha=0.35, color="#8172b3", edgecolors="none")
        mean_val = float(np.mean(samples_np))
        ax.hlines(mean_val, idx[i] - 0.3, idx[i] + 0.3, color="#c44e52", linewidth=2)
        ax.text(idx[i], mean_val, f"{len(samples)}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("||h_{t+1}-h_t||")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)


def save_norm_timeseries_plot(
    out_path: Path,
    steps: Sequence[int],
    z_mean: Sequence[float],
    z_p95: Sequence[float],
    h_mean: Sequence[float],
    h_p95: Sequence[float],
    p_mean: Sequence[float],
    p_p95: Sequence[float],
    title: str = "Norm stability over steps",
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.4))
    ax.plot(steps, z_mean, label="z mean", color="#4c72b0")
    ax.plot(steps, z_p95, label="z p95", color="#4c72b0", linestyle="--")
    ax.plot(steps, h_mean, label="h mean", color="#55a868")
    ax.plot(steps, h_p95, label="h p95", color="#55a868", linestyle="--")
    ax.plot(steps, p_mean, label="p mean", color="#c44e52")
    ax.plot(steps, p_p95, label="p p95", color="#c44e52", linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Norm")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)


__all__ = [
    "StraightLineTrajectory",
    "save_rollout_divergence_plot",
    "save_ablation_divergence_plot",
    "save_straightline_plot",
    "save_z_consistency_plot",
    "save_monotonicity_plot",
    "save_path_independence_plot",
    "save_drift_by_action_plot",
    "save_norm_timeseries_plot",
]
