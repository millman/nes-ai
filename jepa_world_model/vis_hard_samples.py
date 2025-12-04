from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from recon.data import load_frame_as_tensor
from .vis import tensor_to_uint8_image


class HardSampleLike(Protocol):
    label: str
    score: float
    sequence_paths: Sequence[str]
    frame_index: int
    frame_path: str


def _motion_blur_image(record: HardSampleLike, image_hw: Tuple[int, int], window: int = 4) -> np.ndarray:
    weights: list[float] = []
    frames: list[torch.Tensor] = []
    weight = 1.0
    for offset in range(window):
        frame_idx = record.frame_index - offset
        if frame_idx < 0 or frame_idx >= len(record.sequence_paths):
            break
        path = Path(record.sequence_paths[frame_idx])
        tensor = load_frame_as_tensor(path, size=image_hw)
        frames.append(tensor)
        weights.append(weight)
        weight *= 0.5
    if not frames:
        tensor = load_frame_as_tensor(Path(record.frame_path), size=image_hw)
        return tensor_to_uint8_image(tensor)
    stacked = torch.stack(frames, dim=0)
    weight_tensor = torch.tensor(weights, dtype=stacked.dtype).view(-1, 1, 1, 1)
    weight_tensor = weight_tensor / weight_tensor.sum()
    ema = (stacked * weight_tensor).sum(dim=0).clamp(0, 1)
    return tensor_to_uint8_image(ema)


def save_hard_example_grid(
    out_path: Path,
    hard_samples: Sequence[HardSampleLike],
    columns: int,
    rows: int,
    image_hw: Tuple[int, int],
) -> None:
    if not hard_samples:
        return
    columns = max(1, columns)
    rows = max(1, rows)
    limit = columns * rows
    subset = list(hard_samples)[:limit]
    blank = np.zeros((image_hw[0], image_hw[1], 3), dtype=np.uint8)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2, rows * 2))
    axes = np.atleast_2d(axes)
    for idx in range(rows * columns):
        ax = axes[idx // columns, idx % columns]
        if idx < len(subset):
            record = subset[idx]
            image = _motion_blur_image(record, image_hw)
            ax.imshow(image)
            ax.set_title(f"{record.label}\n diff {record.score:.4f}", fontsize=8)
        else:
            ax.imshow(blank)
            ax.set_title("")
        ax.axis("off")
    fig.suptitle("Hard Examples", fontsize=12)
    plt.tight_layout(rect=(0, 0.02, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


__all__ = ["save_hard_example_grid"]
