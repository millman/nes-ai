from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from jepa_world_model.vis import _annotate_with_text, tensor_to_uint8_image
from jepa_world_model.vis_rollout import VisualizationSequence


def save_rollout_visualization(
    out_path: Path,
    sequence: VisualizationSequence,
    grad_label: str,
) -> None:
    seq_cols = len(sequence.labels)
    if seq_cols == 0:
        return
    row_block = 4
    fig, axes = plt.subplots(
        row_block,
        seq_cols,
        figsize=(seq_cols * 2, row_block * 1.5),
    )
    axes = np.atleast_2d(axes)

    def _imshow_tensor(ax: plt.Axes, tensor: torch.Tensor) -> None:
        ax.imshow(tensor.clamp(0, 1).permute(1, 2, 0).numpy())
        ax.axis("off")

    def _imshow_array(ax: plt.Axes, array: np.ndarray) -> None:
        if array.dtype != np.float32 and array.dtype != np.float64:
            data = array.astype(np.float32) / 255.0
        else:
            data = array
        ax.imshow(data)
        ax.axis("off")

    sample_tensor = sequence.ground_truth[0]
    height, width = sample_tensor.shape[1], sample_tensor.shape[2]
    blank = np.full((height, width, 3), 0.5, dtype=np.float32)
    row_labels = [
        "Ground Truth",
        "Rollout Prediction",
        grad_label,
        "Direct Reconstruction",
    ]

    for row_idx in range(row_block):
        axes[row_idx, 0].text(
            -0.12,
            0.5,
            row_labels[row_idx],
            transform=axes[row_idx, 0].transAxes,
            va="center",
            ha="right",
            fontsize=8,
            rotation=90,
        )
    for col in range(seq_cols):
        gt_ax = axes[0, col]
        rollout_ax = axes[1, col]
        grad_ax = axes[2, col]
        recon_ax = axes[3, col]
        gt_img = tensor_to_uint8_image(sequence.ground_truth[col])
        if sequence.actions and col < len(sequence.actions):
            gt_img = _annotate_with_text(gt_img, sequence.actions[col])
        gt_ax.imshow(gt_img)
        gt_ax.axis("off")
        gt_ax.set_title(sequence.labels[col], fontsize=8)
        recon_tensor = sequence.reconstructions[col]
        _imshow_tensor(recon_ax, recon_tensor)
        rollout_frame = sequence.rollout[col]
        if rollout_frame is None:
            _imshow_array(rollout_ax, blank)
        else:
            _imshow_tensor(rollout_ax, rollout_frame)
        grad_map = sequence.gradients[col]
        if grad_map is None:
            _imshow_array(grad_ax, blank)
        else:
            _imshow_array(grad_ax, grad_map)
    fig.suptitle("JEPA Rollout Visualization", fontsize=12)
    fig.tight_layout(rect=(0.08, 0.02, 1.0, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


__all__ = ["save_rollout_visualization"]
