#!/usr/bin/env python3
"""Odometry-style latent diagnostics."""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from sklearn.decomposition import FastICA
    from sklearn.manifold import TSNE
except ImportError:
    FastICA = None
    TSNE = None


def _rollout_predictions(
    model,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Roll predictor forward to obtain z_hat for each next step."""
    b, t, _ = embeddings.shape
    if t < 2:
        return embeddings.new_zeros((b, 0, embeddings.shape[-1]))
    preds = []
    h_t = embeddings.new_zeros((b, model.state_dim))
    for step in range(t - 1):
        z_t = embeddings[:, step]
        act_t = actions[:, step]
        h_next = model.predictor(z_t, h_t, act_t)
        z_pred = model.h_to_z(h_next)
        preds.append(z_pred)
        h_t = h_next
    return torch.stack(preds, dim=1)


def _rollout_open_loop(
    model,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Roll predictor using its own predictions to get open-loop hidden states."""
    b, t, _ = embeddings.shape
    if t < 2:
        return embeddings.new_zeros((b, 0, model.state_dim))
    h_t = embeddings.new_zeros((b, model.state_dim))
    z_t = embeddings[:, 0]
    h_preds = []
    for step in range(t - 1):
        act_t = actions[:, step]
        h_next = model.predictor(z_t, h_t, act_t)
        z_next = model.h_to_z(h_next)
        h_preds.append(h_next)
        z_t = z_next
        h_t = h_next
    return torch.stack(h_preds, dim=1)


def _compute_pca_2d(data: np.ndarray) -> Optional[np.ndarray]:
    if data.shape[0] < 2:
        return None
    if not np.isfinite(data).all():
        return None
    centered = data - data.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        jitter = np.random.normal(scale=1e-6, size=centered.shape)
        _, _, vt = np.linalg.svd(centered + jitter, full_matrices=False)
    if vt.shape[0] < 2:
        return None
    return centered @ vt[:2].T


def _compute_ica_2d(data: np.ndarray) -> Optional[np.ndarray]:
    if FastICA is None or data.shape[0] < 3:
        return None
    try:
        ica = FastICA(n_components=2, random_state=0, max_iter=1000)
        return ica.fit_transform(data)
    except Exception:
        return None


def _compute_tsne_2d(data: np.ndarray) -> Optional[np.ndarray]:
    if TSNE is None or data.shape[0] < 3:
        return None
    n_samples = data.shape[0]
    perplexity = max(2, min(30, (n_samples - 1) / 3))
    if perplexity >= n_samples:
        return None
    try:
        tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
        return tsne.fit_transform(data)
    except Exception:
        return None


def _plot_odometry_embeddings(
    out_path,
    history: np.ndarray,
    embedding_label: str,
) -> None:
    if history.shape[0] < 2:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 1, num=history.shape[0])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, title, coords in (
        (axes[0], "PCA", _compute_pca_2d(history)),
        (axes[1], "ICA", _compute_ica_2d(history)),
        (axes[2], "t-SNE", _compute_tsne_2d(history)),
    ):
        if coords is None:
            ax.text(0.5, 0.5, "Unavailable", ha="center", va="center")
            ax.axis("off")
            continue
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=t, cmap="viridis", s=14, alpha=0.75)
        ax.set_title(f"{embedding_label} {title}")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_latent_prediction_comparison(
    out_path,
    z_next: np.ndarray,
    z_hat: np.ndarray,
    embedding_label: str,
) -> None:
    if z_next.shape[0] < 2:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    hat_label = f"{embedding_label}_hat"
    t = np.arange(z_next.shape[0])
    diff = z_next - z_hat
    diff_norm = np.linalg.norm(diff, axis=1)
    axes[0].plot(t, diff_norm, marker="o", markersize=4, linewidth=1.0, color="tab:blue")
    axes[0].set_title(f"||{embedding_label} - {hat_label}|| over time")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel(f"||{embedding_label} - {hat_label}||")

    axes[1].scatter(z_next.reshape(-1), z_hat.reshape(-1), s=6, alpha=0.4, color="tab:blue")
    axes[1].set_title(f"{embedding_label} vs {hat_label} scatter")
    axes[1].set_xlabel(f"{embedding_label}")
    axes[1].set_ylabel(f"{hat_label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
