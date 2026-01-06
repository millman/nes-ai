from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from jepa_world_model.vis import _annotate_with_text, delta_to_uint8_image, tensor_to_uint8_image
from jepa_world_model.vis_rollout import VisualizationSequence


def save_rollout_visualization(
    out_path: Path,
    sequence: VisualizationSequence,
    grad_label: str,
    include_pixel_delta: bool,
) -> None:
    seq_cols = len(sequence.labels)
    if seq_cols == 0:
        return
    row_labels = [
        "Ground Truth",
        "Direct Reconstruction",
        "Rollout Prediction",
        "Re-encoded Rollout",
        grad_label,
    ]
    if include_pixel_delta:
        row_labels.extend(["Delta Target", "Delta Recon"])
    row_block = len(row_labels)
    fig, axes = plt.subplots(
        row_block,
        seq_cols,
        figsize=(seq_cols * 2, row_block * 1.5),
        constrained_layout=True,
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
        recon_ax = axes[1, col]
        rollout_ax = axes[2, col]
        reenc_ax = axes[3, col]
        grad_ax = axes[4, col]
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
        reenc_frame = sequence.reencoded[col] if sequence.reencoded else None
        if reenc_frame is None:
            _imshow_array(reenc_ax, blank)
        else:
            _imshow_tensor(reenc_ax, reenc_frame)
        if include_pixel_delta:
            delta_target_ax = axes[5, col]
            delta_recon_ax = axes[6, col]
            if col == 0:
                _imshow_array(delta_target_ax, blank)
                _imshow_array(delta_recon_ax, blank)
            else:
                delta_target = sequence.ground_truth[col] - sequence.ground_truth[col - 1]
                delta_recon = sequence.reconstructions[col] - sequence.reconstructions[col - 1]
                _imshow_array(delta_target_ax, delta_to_uint8_image(delta_target))
                _imshow_array(delta_recon_ax, delta_to_uint8_image(delta_recon))
    fig.suptitle("JEPA Rollout Visualization", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


__all__ = ["save_rollout_visualization"]
