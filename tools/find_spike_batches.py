#!/usr/bin/env python3
"""Reconstruct shuffled batch membership for specific training steps.

This mirrors the training DataLoader shuffle behavior without loading frames.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import random

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore

from jepa_world_model.actions import describe_action_tensor
from jepa_world_model.data import TrajectorySequenceDataset
from jepa_world_model.vis import _annotate_with_text, tensor_to_uint8_image
from recon.data import list_trajectories, load_frame_as_tensor, short_traj_state_label


def _parse_steps(steps_arg: str) -> List[int]:
    steps: List[int] = []
    for chunk in steps_arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_str, end_str = chunk.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise AssertionError(f"Invalid step range '{chunk}' (end < start).")
            steps.extend(range(start, end + 1))
        else:
            steps.append(int(chunk))
    if not steps:
        raise AssertionError("No steps provided.")
    return sorted(set(steps))


def _normalize_optional(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"null", "none"}:
        return None
    return value


def _split_trajectories(
    root: Path, max_traj: int | None, val_fraction: float, seed: int
) -> Tuple[List[str], List[str]]:
    traj_names = sorted(list(list_trajectories(root).keys()))
    if max_traj is not None:
        traj_names = traj_names[:max_traj]
    if not traj_names:
        return [], []
    rng = random.Random(seed)
    rng.shuffle(traj_names)
    if val_fraction <= 0:
        return traj_names, []
    if val_fraction >= 1.0:
        return [], traj_names
    val_count = max(1, int(len(traj_names) * val_fraction)) if len(traj_names) > 1 else 0
    val_count = min(val_count, max(0, len(traj_names) - 1))
    val_names = traj_names[:val_count]
    train_names = traj_names[val_count:]
    return train_names, val_names


def _iter_batches(
    dataset_size: int,
    batch_size: int,
    generator: torch.Generator,
) -> Iterable[Tuple[int, int, List[int]]]:
    """Yield (epoch_idx, batch_idx, indices) in the same order as RandomSampler."""
    epoch = 0
    while True:
        order = torch.randperm(dataset_size, generator=generator).tolist()
        batch_idx = 0
        for offset in range(0, dataset_size, batch_size):
            yield epoch, batch_idx, order[offset : offset + batch_size]
            batch_idx += 1
        epoch += 1


def _load_metadata(run_dir: Path) -> dict:
    metadata_path = run_dir / "metadata.txt"
    if not metadata_path.is_file():
        raise AssertionError(f"Missing metadata.txt in {run_dir}")
    with metadata_path.open("rb") as f:
        return tomllib.load(f)


def _resolve_data_root(run_dir: Path, data_root: str) -> Path:
    root_path = Path(data_root)
    if root_path.is_absolute():
        return root_path
    # Prefer relative to repo root (cwd), then relative to run directory.
    cwd_path = Path(os.getcwd()) / root_path
    if cwd_path.exists():
        return cwd_path
    run_relative = run_dir / root_path
    if run_relative.exists():
        return run_relative
    return cwd_path


def _format_sample(dataset: TrajectorySequenceDataset, index: int) -> str:
    frame_paths, _, start = dataset.samples[index]
    first_label = short_traj_state_label(str(frame_paths[start]))
    last_label = short_traj_state_label(str(frame_paths[start + dataset.seq_len - 1]))
    traj_name = Path(frame_paths[0]).parts[-3]
    return f"{index}: {traj_name} start={start} ({first_label} -> {last_label})"


def _choose_grid_shape(num_items: int, target_ratio: float = 8 / 5) -> Tuple[int, int]:
    if num_items <= 0:
        raise AssertionError("num_items must be positive.")
    best_cols = 1
    best_rows = num_items
    best_score = float("inf")
    for cols in range(1, num_items + 1):
        rows = int(math.ceil(num_items / cols))
        ratio = cols / rows
        score = abs(ratio - target_ratio)
        area = cols * rows
        best_area = best_cols * best_rows
        if score < best_score or (score == best_score and area < best_area):
            best_score = score
            best_cols = cols
            best_rows = rows
    return best_cols, best_rows


def _pad_image(image: np.ndarray, pad: int, color: int = 0) -> np.ndarray:
    if pad <= 0:
        return image
    height, width = image.shape[:2]
    canvas = np.full((height + pad, width + pad, 3), color, dtype=image.dtype)
    canvas[:height, :width] = image
    return canvas


def _make_title_strip(text: str, width: int, height: int) -> np.ndarray:
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    return _annotate_with_text(strip, text)


def _render_batch_grid(
    dataset: TrajectorySequenceDataset,
    indices: Sequence[int],
    image_size: int,
    out_path: Path,
) -> None:
    if not indices:
        raise AssertionError("No indices provided for visualization.")
    columns, rows = _choose_grid_shape(len(indices))
    images: List[np.ndarray] = []
    title_height = max(12, int(round(image_size * 0.2)))
    for idx in indices:
        frame_paths, actions, start = dataset.samples[idx]
        frame_path = frame_paths[start]
        frame = load_frame_as_tensor(frame_path, size=(image_size, image_size))
        img = tensor_to_uint8_image(frame)
        action = torch.from_numpy(actions[start])
        action_desc = describe_action_tensor(action)
        img = _annotate_with_text(img, action_desc)
        title = short_traj_state_label(str(frame_path))
        title_strip = _make_title_strip(title, img.shape[1], title_height)
        tile = np.concatenate([title_strip, img], axis=0)
        images.append(tile)

    if len(images) < rows * columns:
        blank = np.zeros_like(images[0])
        images.extend([blank] * (rows * columns - len(images)))

    pad = max(2, image_size // 48)
    row_images: List[np.ndarray] = []
    for row_idx in range(rows):
        row_slice = images[row_idx * columns : (row_idx + 1) * columns]
        padded = [_pad_image(img, pad) for img in row_slice]
        row = np.concatenate(padded, axis=1)
        row_images.append(row)
    grid = np.concatenate(row_images, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct shuffled batch indices for specific training steps."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Experiment run directory (e.g., out.jepa_world_model_trainer/2026-01-30_14-09-41)",
    )
    parser.add_argument(
        "--steps",
        required=True,
        help="Comma-separated steps or ranges (e.g., 4330 or 4320-4340,5000)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override training seed")
    parser.add_argument("--val-split-seed", type=int, default=None, help="Override val split seed")
    parser.add_argument("--data-root", type=str, default=None, help="Override data_root")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--seq-len", type=int, default=None, help="Override seq_len")
    parser.add_argument("--max-trajectories", type=int, default=None, help="Override max trajectories")
    parser.add_argument(
        "--grid-dir",
        type=Path,
        default=None,
        help="Directory to save batch grids (default: <run_dir>/vis_spike_batches)",
    )
    args = parser.parse_args()

    steps = _parse_steps(args.steps)
    metadata = _load_metadata(args.run_dir)
    train_cfg = metadata.get("train_config")
    if train_cfg is None:
        raise AssertionError("metadata.txt missing train_config")

    model_cfg = metadata.get("model_config")
    if model_cfg is None:
        raise AssertionError("metadata.txt missing model_config")

    data_root = args.data_root or train_cfg.get("data_root")
    if not data_root:
        raise AssertionError("data_root missing from metadata and not provided")
    batch_size = args.batch_size or int(train_cfg.get("batch_size"))
    seq_len = args.seq_len or int(train_cfg.get("seq_len"))
    seed = args.seed if args.seed is not None else int(train_cfg.get("seed"))
    val_split_seed = (
        args.val_split_seed if args.val_split_seed is not None else int(train_cfg.get("val_split_seed"))
    )
    val_fraction = float(train_cfg.get("val_fraction", 0.0))
    image_size = int(model_cfg.get("image_size"))
    if image_size <= 0:
        raise AssertionError("image_size must be positive.")

    max_traj_raw = _normalize_optional(train_cfg.get("max_trajectories"))
    max_traj = args.max_trajectories if args.max_trajectories is not None else max_traj_raw
    if isinstance(max_traj, str):
        max_traj = _normalize_optional(max_traj)
    if max_traj is not None:
        max_traj = int(max_traj)

    data_root_path = _resolve_data_root(args.run_dir, data_root)

    train_trajs, _ = _split_trajectories(data_root_path, max_traj, val_fraction, val_split_seed)
    dataset = TrajectorySequenceDataset(
        root=data_root_path,
        seq_len=seq_len,
        image_hw=(1, 1),
        max_traj=None,
        included_trajectories=train_trajs,
    )

    dataset_size = len(dataset)
    if dataset_size <= 0:
        raise AssertionError("Dataset is empty after splitting.")
    if batch_size <= 0:
        raise AssertionError("batch_size must be positive.")

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    target_steps = set(steps)
    max_step = max(steps)

    batches_per_epoch = math.ceil(dataset_size / batch_size)
    print(f"dataset_size={dataset_size} batch_size={batch_size} batches_per_epoch={batches_per_epoch}")
    print(f"seed={seed} val_split_seed={val_split_seed} val_fraction={val_fraction}")
    print(f"data_root={data_root_path}")
    print("-")

    grid_dir = args.grid_dir or (args.run_dir / "vis_spike_batches")

    current_step = 0
    for epoch, batch_idx, batch_indices in _iter_batches(dataset_size, batch_size, generator):
        if current_step in target_steps:
            print(f"step {current_step}: epoch {epoch} batch {batch_idx} size {len(batch_indices)}")
            for idx in batch_indices:
                print(f"  { _format_sample(dataset, idx) }")
            print("-")
            grid_path = grid_dir / f"batch_{current_step:07d}.png"
            _render_batch_grid(dataset, batch_indices, image_size, grid_path)
        if current_step >= max_step:
            break
        current_step += 1


if __name__ == "__main__":
    main()
