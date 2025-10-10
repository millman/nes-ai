import math
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np


def save_image_grid(images: Iterable[np.ndarray], titles: Optional[Iterable[str]], out_path: str, ncol: int = 8) -> None:
    images = list(images)
    titles = list(titles) if titles is not None else None
    if titles is not None and len(titles) != len(images):
        raise ValueError("titles length must match images length")

    if not images:
        return

    ncol = max(1, min(ncol, len(images)))
    nrow = math.ceil(len(images) / ncol)
    plt.figure(figsize=(1.8 * ncol, 1.8 * nrow))
    for idx, img in enumerate(images):
        plt.subplot(nrow, ncol, idx + 1)
        plt.imshow(img)
        if titles is not None:
            plt.title(titles[idx], fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if heatmap.ndim == 3 and heatmap.shape[0] == 1:
        heatmap = heatmap[0]
    if image.shape[:2] != heatmap.shape:
        raise ValueError("image and heatmap spatial dimensions must match")
    heatmap = np.clip(heatmap, 0.0, 1.0)
    colored = plt.get_cmap('jet')(heatmap)[..., :3]
    return np.clip(0.6 * image + 0.4 * colored, 0.0, 1.0)


def plot_self_distance(t_steps: np.ndarray, y_vals: np.ndarray, color_vals: np.ndarray, title: str, ylabel: str, color_label: str, out_path: str) -> None:
    plt.figure(figsize=(6, 4))
    sc = plt.scatter(t_steps, y_vals, c=color_vals, cmap='viridis')
    plt.colorbar(sc, label=color_label)
    plt.xlabel('time step t')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_line(values: np.ndarray, title: str, ylabel: str, out_path: str) -> None:
    if values.ndim != 1:
        raise ValueError("values for plot_line must be 1D")
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(len(values)), values, marker='o')
    plt.xlabel('time step t')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
