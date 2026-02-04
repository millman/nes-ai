#!/usr/bin/env python3
"""Dataset utilities for JEPA world model training."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import warnings
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from recon.data import list_trajectories, load_frame_as_tensor


def collate_batch(
    batch: Iterable[Tuple[torch.Tensor, torch.Tensor, List[str], int]]
) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]], torch.Tensor]:
    obs, actions, paths, indices = zip(*batch)
    obs_tensor = torch.stack(obs, dim=0)
    act_tensor = torch.stack(actions, dim=0)
    path_batch = [list(seq) for seq in paths]
    idx_tensor = torch.tensor(indices, dtype=torch.long)
    return obs_tensor, act_tensor, path_batch, idx_tensor


def load_actions_for_trajectory(traj_dir: Path, expected_length: Optional[int] = None) -> np.ndarray:
    """Load actions.npz for a trajectory and ensure alignment."""
    actions_path = Path(traj_dir) / "actions.npz"
    if not actions_path.is_file():
        raise FileNotFoundError(f"Missing actions.npz for {traj_dir}")
    with np.load(actions_path) as data:
        action_arr = data["actions"] if "actions" in data else data[list(data.files)[0]]
    if action_arr.ndim == 1:
        action_arr = action_arr[:, None]
    action_arr = action_arr.astype(np.float32, copy=False)
    if expected_length is not None and action_arr.shape[0] != expected_length:
        raise ValueError(
            f"Action count {action_arr.shape[0]} does not match frame count {expected_length} in {traj_dir}"
        )
    return action_arr


class TrajectorySequenceDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, List[str], int]]):
    """Load contiguous frame/action sequences from recorded trajectories."""

    def __init__(
        self,
        root: Path,
        seq_len: int,
        image_hw: Tuple[int, int],
        max_traj: Optional[int] = None,
        included_trajectories: Optional[Sequence[str]] = None,
    ) -> None:
        self.root = Path(root)
        self.seq_len = seq_len
        self.image_hw = image_hw
        if self.seq_len < 1:
            raise ValueError("seq_len must be positive.")
        trajectories = list_trajectories(self.root)
        if included_trajectories is not None:
            include_set = set(included_trajectories)
            items = [(name, frames) for name, frames in trajectories.items() if name in include_set]
        else:
            items = list(trajectories.items())
        if max_traj is not None:
            items = items[:max_traj]
        self.samples: List[Tuple[List[Path], np.ndarray, int]] = []
        self.action_dim: Optional[int] = None
        for traj_name, frame_paths in tqdm(items, desc="Preloading trajectories", unit="traj"):
            if len(frame_paths) < self.seq_len:
                warnings.warn(
                    f"Skipping trajectory {traj_name} shorter than seq_len {self.seq_len}",
                    RuntimeWarning,
                )
                continue
            action_arr = load_actions_for_trajectory(self.root / traj_name, expected_length=len(frame_paths))
            if self.action_dim is None:
                self.action_dim = action_arr.shape[1]
            elif self.action_dim != action_arr.shape[1]:
                raise ValueError(
                    f"Inconsistent action dimension for {traj_name}: expected {self.action_dim}, got {action_arr.shape[1]}"
                )
            if len(frame_paths) < self.seq_len:
                raise ValueError(f"Trajectory {traj_name} shorter than seq_len {self.seq_len}")
            max_start = len(frame_paths) - self.seq_len
            for start in range(max_start + 1):
                self.samples.append((frame_paths, action_arr, start))
        if not self.samples:
            raise AssertionError(f"No usable sequences found under {self.root}")
        if self.action_dim is None:
            raise AssertionError("Failed to infer action dimensionality.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, List[str], int]:
        frame_paths, actions, start = self.samples[index]
        frames: List[torch.Tensor] = []
        path_slice: List[str] = []
        for offset in range(self.seq_len):
            path = frame_paths[start + offset]
            frame = load_frame_as_tensor(path, size=self.image_hw)
            frames.append(frame)
            path_slice.append(str(path))
        action_slice = actions[start : start + self.seq_len]
        # Each frame/action pair must stay aligned so the predictor knows which action follows each observation.
        assert len(frames) == self.seq_len, f"Expected {self.seq_len} frames, got {len(frames)}"
        assert action_slice.shape[0] == self.seq_len, (
            f"Expected {self.seq_len} actions, got {action_slice.shape[0]}"
        )
        assert action_slice.shape[1] == actions.shape[1], "Action dimensionality changed unexpectedly."
        return torch.stack(frames, dim=0), torch.from_numpy(action_slice), path_slice, index


def _estimate_tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _estimate_strings_bytes(strings: Sequence[str]) -> int:
    total = 0
    for value in strings:
        total += sys.getsizeof(value)
    return total


class PreloadedTrajectorySequenceDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, List[str], int]]):
    """Preload contiguous frame/action sequences from recorded trajectories."""

    def __init__(
        self,
        root: Path,
        seq_len: int,
        image_hw: Tuple[int, int],
        max_traj: Optional[int] = None,
        included_trajectories: Optional[Sequence[str]] = None,
    ) -> None:
        self.root = Path(root)
        self.seq_len = seq_len
        self.image_hw = image_hw
        if self.seq_len < 1:
            raise ValueError("seq_len must be positive.")
        trajectories = list_trajectories(self.root)
        if included_trajectories is not None:
            include_set = set(included_trajectories)
            items = [(name, frames) for name, frames in trajectories.items() if name in include_set]
        else:
            items = list(trajectories.items())
        if max_traj is not None:
            items = items[:max_traj]

        self.trajs: List[Tuple[torch.Tensor, torch.Tensor, List[str]]] = []
        self.samples: List[Tuple[int, int]] = []
        self.action_dim: Optional[int] = None

        for traj_name, frame_paths in items:
            if len(frame_paths) < self.seq_len:
                warnings.warn(
                    f"Skipping trajectory {traj_name} shorter than seq_len {self.seq_len}",
                    RuntimeWarning,
                )
                continue
            action_arr = load_actions_for_trajectory(self.root / traj_name, expected_length=len(frame_paths))
            if self.action_dim is None:
                self.action_dim = action_arr.shape[1]
            elif self.action_dim != action_arr.shape[1]:
                raise ValueError(
                    f"Inconsistent action dimension for {traj_name}: expected {self.action_dim}, got {action_arr.shape[1]}"
                )
            frames: List[torch.Tensor] = []
            paths: List[str] = []
            for path in tqdm(frame_paths, desc=f"Preloading {traj_name}", unit="frame", leave=False):
                frames.append(load_frame_as_tensor(path, size=self.image_hw))
                paths.append(str(path))
            traj_frames = torch.stack(frames, dim=0)
            traj_actions = torch.from_numpy(action_arr)
            traj_index = len(self.trajs)
            self.trajs.append((traj_frames, traj_actions, paths))
            max_start = len(frame_paths) - self.seq_len
            for start in range(max_start + 1):
                self.samples.append((traj_index, start))
        if not self.samples:
            raise AssertionError(f"No usable sequences found under {self.root}")
        if self.action_dim is None:
            raise AssertionError("Failed to infer action dimensionality.")

        self._print_memory_estimate()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, List[str], int]:
        traj_idx, start = self.samples[index]
        traj_frames, traj_actions, traj_paths = self.trajs[traj_idx]
        frames = traj_frames[start : start + self.seq_len]
        actions = traj_actions[start : start + self.seq_len]
        paths = traj_paths[start : start + self.seq_len]
        # Each frame/action pair must stay aligned so the predictor knows which action follows each observation.
        assert frames.shape[0] == self.seq_len, f"Expected {self.seq_len} frames, got {frames.shape[0]}"
        assert actions.shape[0] == self.seq_len, f"Expected {self.seq_len} actions, got {actions.shape[0]}"
        assert actions.shape[1] == traj_actions.shape[1], "Action dimensionality changed unexpectedly."
        return frames, actions, list(paths), index

    def _print_memory_estimate(self) -> None:
        traj_tensor_bytes = 0
        traj_paths_bytes = 0
        for traj_frames, traj_actions, traj_paths in self.trajs:
            traj_tensor_bytes += _estimate_tensor_bytes(traj_frames)
            traj_tensor_bytes += _estimate_tensor_bytes(traj_actions)
            traj_paths_bytes += _estimate_strings_bytes(traj_paths)
        traj_list_bytes = sys.getsizeof(self.trajs)
        sample_list_bytes = sys.getsizeof(self.samples)
        sample_tuple_bytes = 0
        if self.samples:
            sample_tuple_bytes = sys.getsizeof(self.samples[0]) * len(self.samples)
        underlying_bytes = traj_tensor_bytes + traj_paths_bytes + traj_list_bytes
        samples_bytes = sample_list_bytes + sample_tuple_bytes
        total_bytes = underlying_bytes + samples_bytes
        print(
            "[dataset preload memory] "
            f"total={total_bytes / (1024 ** 2):.2f} MiB "
            f"underlying={underlying_bytes / (1024 ** 2):.2f} MiB "
            f"samples={samples_bytes / (1024 ** 2):.2f} MiB"
        )


__all__ = [
    "TrajectorySequenceDataset",
    "PreloadedTrajectorySequenceDataset",
    "collate_batch",
    "load_actions_for_trajectory",
]
