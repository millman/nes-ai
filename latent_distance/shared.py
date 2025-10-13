"""Shared utilities for latent-distance scripts."""
from __future__ import annotations

import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, TypeVar

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from trajectory_utils import list_state_frames, list_traj_dirs


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TrajectoryIndex:
    """Index frames across trajectories for random access."""

    def __init__(self, data_root: Path, img_suffixes=(".png", ".jpg", ".jpeg"), min_len: int = 2):
        traj_dirs = list_traj_dirs(Path(data_root))
        if len(traj_dirs) == 0:
            raise RuntimeError(f"No trajectory directories found in {data_root}")

        self.traj_paths: List[Path] = []
        self.frames: List[List[Path]] = []
        for traj_path in traj_dirs:
            states_dir = traj_path / "states"
            if not states_dir.is_dir():
                continue
            imgs = list_state_frames(states_dir, img_suffixes)
            if len(imgs) < min_len:
                continue
            self.traj_paths.append(traj_path)
            self.frames.append(imgs)

        if len(self.frames) == 0:
            raise RuntimeError(f"No trajectories with >= {min_len} frames found in {data_root}")

        self.cum_counts = np.cumsum([0] + [len(f) for f in self.frames])

    def num_traj(self) -> int:
        return len(self.frames)

    def len_traj(self, idx: int) -> int:
        return len(self.frames[idx])

    def get_path(self, traj_idx: int, t: int) -> Path:
        return self.frames[traj_idx][t]


class FramesDataset(Dataset):
    """Dataset yielding individual frames for evaluation."""

    def __init__(self, index: TrajectoryIndex, image_size: int = 128):
        self.index = index
        self.image_size = image_size
        self.tr = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return sum(len(f) for f in self.index.frames)

    def __getitem__(self, global_idx: int):
        traj_idx = int(np.searchsorted(self.index.cum_counts, global_idx, side='right') - 1)
        local_offset = global_idx - self.index.cum_counts[traj_idx]
        path = self.index.get_path(traj_idx, local_offset)
        img = Image.open(path).convert('RGB').resize((self.image_size, self.image_size), Image.BILINEAR)
        x = self.tr(img)
        return {
            'x': x,
            'traj_idx': traj_idx,
            't': local_offset,
            'path': str(path),
        }


def pick_device(pref: Optional[str]) -> str:
    if pref:
        return pref
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


T = TypeVar('T')


def attach_run_output_dir(args: T) -> T:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.out_dir) / f"run__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(run_dir)
    return args


def format_elapsed(start_time: float) -> str:
    elapsed = int(time.time() - start_time)
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
