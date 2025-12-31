from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from nes_controller import _describe_controller_vector_compact


class VisualizationSequenceLike(Protocol):
    ground_truth: Sequence[torch.Tensor]
    reconstructions: Sequence[torch.Tensor]
    rollout: Sequence[Optional[torch.Tensor]]
    gradients: Sequence[Optional[np.ndarray]]
    labels: Sequence[str]
    actions: Sequence[str]


TEXT_FONT = ImageFont.load_default()
BASE_FONT_SIZE = 10
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def _get_font(font_size: int) -> ImageFont.FreeTypeFont:
    cached = _FONT_CACHE.get(font_size)
    if cached is not None:
        return cached
    font = TEXT_FONT
    _FONT_CACHE[font_size] = font
    return font


def describe_action_tensor(action: torch.Tensor) -> str:
    vector = action.detach().cpu().numpy().reshape(-1)
    binary = (vector > 0.5).astype(np.uint8)
    return _describe_controller_vector_compact(binary)


def tensor_to_uint8_image(frame: torch.Tensor) -> np.ndarray:
    array = frame.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (array * 255.0).round().astype(np.uint8)


def _annotate_with_text(image: np.ndarray, text: str) -> np.ndarray:
    if not text:
        return image
    height = image.shape[0]
    scale = max(0.25, height / 128.0)
    font_size = max(6, int(round(BASE_FONT_SIZE * scale)))
    font = _get_font(font_size)
    padding = max(1, int(round(2 * scale)))
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    text = text.strip()
    bbox = draw.textbbox((padding, padding), text, font=font)
    rect = (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding)
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.text((padding, padding), text, fill=(255, 255, 255), font=font)
    return np.array(pil_image)


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


def save_temporal_pair_visualization(
    out_path: Path,
    frames: torch.Tensor,
    actions: torch.Tensor,
    rows: int,
    generator: torch.Generator | None = None,
) -> None:
    if frames.shape[1] < 2:
        return
    frames = frames.detach().cpu()
    actions = actions.detach().cpu()
    batch_size = frames.shape[0]
    num_rows = min(rows, batch_size)
    order = torch.randperm(batch_size, generator=generator)[:num_rows]
    pairs: list[np.ndarray] = []
    for idx in order:
        time_idx = torch.randint(1, frames.shape[1], (), generator=generator).item()
        prev_frame = tensor_to_uint8_image(frames[idx, time_idx - 1])
        next_frame = tensor_to_uint8_image(frames[idx, time_idx])
        prev_frame = _annotate_with_text(prev_frame, describe_action_tensor(actions[idx, time_idx - 1]))
        next_frame = _annotate_with_text(next_frame, describe_action_tensor(actions[idx, time_idx]))
        pairs.append(np.concatenate([prev_frame, next_frame], axis=1))
    grid = np.concatenate(pairs, axis=0) if pairs else np.zeros((1, 1, 3), dtype=np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path)


def save_input_batch_visualization(
    out_path: Path,
    frames: torch.Tensor,
    actions: torch.Tensor,
    rows: int,
) -> None:
    frames = frames.detach().cpu()
    actions = actions.detach().cpu()
    batch_size, seq_len = frames.shape[0], frames.shape[1]
    num_rows = min(rows, batch_size)
    if num_rows <= 0:
        return
    grid_rows: list[np.ndarray] = []
    for row_idx in range(num_rows):
        columns: list[np.ndarray] = []
        for step in range(seq_len):
            frame_img = tensor_to_uint8_image(frames[row_idx, step])
            desc = describe_action_tensor(actions[row_idx, step])
            frame_img = _annotate_with_text(frame_img, desc)
            columns.append(frame_img)
        row_image = np.concatenate(columns, axis=1)
        grid_rows.append(row_image)
    grid = np.concatenate(grid_rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path)


def save_adjacency_input_visualization(
    out_path: Path,
    clean_frames: torch.Tensor,
    noisy_frames: torch.Tensor,
    rows: int,
    max_steps: int | None = None,
) -> None:
    """Visualize clean vs. noisy pairs used for adjacency supervision."""
    if clean_frames.shape != noisy_frames.shape:
        raise ValueError("Clean and noisy frames must share shape for adjacency visualization.")
    clean = clean_frames.detach().cpu()
    noisy = noisy_frames.detach().cpu()
    batch_size, seq_len = clean.shape[0], clean.shape[1]
    steps = seq_len if max_steps is None else max(1, min(seq_len, int(max_steps)))
    num_rows = min(rows, batch_size)
    if num_rows <= 0 or steps <= 0:
        return
    grid_rows: list[np.ndarray] = []
    for row_idx in range(num_rows):
        columns: list[np.ndarray] = []
        for step in range(steps):
            clean_img = tensor_to_uint8_image(clean[row_idx, step])
            noisy_img = tensor_to_uint8_image(noisy[row_idx, step])
            clean_img = _annotate_with_text(clean_img, f"t{step} clean")
            noisy_img = _annotate_with_text(noisy_img, f"t{step} noisy")
            columns.append(clean_img)
            columns.append(noisy_img)
        row_image = np.concatenate(columns, axis=1)
        grid_rows.append(row_image)
    grid = np.concatenate(grid_rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path)


def save_rollout_visualization(
    out_path: Path,
    sequence: VisualizationSequenceLike,
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


def save_rollout_sequence_batch(
    template_dir: Path,
    sequences: Sequence[VisualizationSequenceLike],
    grad_label: str,
    global_step: int,
) -> None:
    if not sequences:
        return
    base_parent = template_dir.parent
    base_name = template_dir.name
    for idx, sequence in enumerate(sequences):
        sample_dir = base_parent / f"{base_name}_{idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        out_path = sample_dir / f"rollout_{global_step:07d}.png"
        save_rollout_visualization(out_path, sequence, grad_label)


__all__ = [
    "describe_action_tensor",
    "tensor_to_uint8_image",
    "save_embedding_projection",
    "save_temporal_pair_visualization",
    "save_input_batch_visualization",
    "save_adjacency_input_visualization",
    "save_rollout_visualization",
    "save_rollout_sequence_batch",
]
