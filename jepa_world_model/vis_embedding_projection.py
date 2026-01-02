from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_embedding_projection(embeddings: torch.Tensor, path: Path, title: str) -> None:
    flat = embeddings.detach().cpu().numpy()
    b, t, dim = flat.shape
    flat = flat.reshape(b * t, dim)
    flat = flat - flat.mean(axis=0, keepdims=True)
    if flat.shape[0] < 2:
        return
    try:
        _, _, vt = np.linalg.svd(flat, full_matrices=False)
    except np.linalg.LinAlgError:
        jitter = np.random.normal(scale=1e-6, size=flat.shape)
        try:
            _, _, vt = np.linalg.svd(flat + jitter, full_matrices=False)
        except np.linalg.LinAlgError:
            print("Warning: embedding SVD failed to converge; skipping projection.")
            return
    coords = flat @ vt[:2].T
    colors = np.tile(np.arange(t), b)
    colors = colors / max(1, t - 1)
    plt.figure(figsize=(6, 5))
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, cmap="viridis", s=10, alpha=0.7)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="Time step")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


__all__ = ["save_embedding_projection"]
