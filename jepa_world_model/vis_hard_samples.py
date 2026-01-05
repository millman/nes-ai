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


def motion_trail(frames: torch.Tensor, trail_strength: float = 0.7) -> torch.Tensor:
    """Blend past frames into a fading trail while keeping the latest frame sharp."""
    if frames.ndim != 4:
        raise ValueError(f"Expected frames with shape [T, C, H, W], got {frames.shape}")
    if frames.shape[0] == 0:
        raise ValueError("motion_trail requires at least one frame")

    orig_dtype = frames.dtype
    frames_f = frames.to(torch.float32)
    if not torch.is_floating_point(frames):
        frames_f = frames_f / 255.0
    frames_f = frames_f.clamp(0.0, 1.0)

    current = frames_f[-1]
    if frames_f.shape[0] == 1:
        blended = current
    else:
        past = frames_f[:-1]
        N = past.shape[0]
        weights = torch.arange(N, 0, -1, device=past.device, dtype=past.dtype)
        weights = weights / weights.sum()
        trail = (past.flip(0) * weights.view(-1, 1, 1, 1)).sum(dim=0)
        strength = float(max(0.0, min(1.0, trail_strength)))
        if strength <= 0.0:
            blended = current
        else:
            total = 1.0 + strength
            blended = torch.clamp((current + strength * trail) / total, 0.0, 1.0)

    if orig_dtype == torch.uint8:
        return (blended * 255.0).round().to(torch.uint8)
    if torch.is_floating_point(frames):
        return blended.to(orig_dtype)
    return blended


def _motion_blur_image(record: HardSampleLike, image_hw: Tuple[int, int], window: int = 4) -> np.ndarray:
    frames: list[torch.Tensor] = []
    for offset in range(window):
        frame_idx = record.frame_index - offset
        if frame_idx < 0 or frame_idx >= len(record.sequence_paths):
            break
        path = Path(record.sequence_paths[frame_idx])
        tensor = load_frame_as_tensor(path, size=image_hw)
        frames.append(tensor)
    if not frames:
        tensor = load_frame_as_tensor(Path(record.frame_path), size=image_hw)
        return tensor_to_uint8_image(tensor)
    frames.reverse()
    stacked = torch.stack(frames, dim=0)
    trailed = motion_trail(stacked)
    if trailed.dtype == torch.uint8:
        return trailed.permute(1, 2, 0).cpu().numpy()
    return tensor_to_uint8_image(trailed)


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
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2, rows * 2), constrained_layout=True)
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


__all__ = ["save_hard_example_grid", "motion_trail"]
