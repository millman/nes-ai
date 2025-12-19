"""Data utilities shared by action-distance reconstruction scripts."""

from __future__ import annotations

import itertools
import logging
import os
import random
import threading
import traceback
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, get_worker_info

from .constants import H, W

NormalizeFn = Callable[[torch.Tensor], torch.Tensor]
LoadFrameFn = Callable[[Path], torch.Tensor]


logger = logging.getLogger(__name__)


# Toggle via NESAI_LOG_FILE_OPENS to help diagnose leaked file descriptors.
_FILE_OPEN_LOG_ENV = "NESAI_LOG_FILE_OPENS"
_file_open_mode = os.environ.get(_FILE_OPEN_LOG_ENV)
_log_file_opens = bool(_file_open_mode)
_log_file_stack = bool(_file_open_mode and "stack" in _file_open_mode.lower())
_open_sequence = itertools.count()
_open_lock = threading.Lock()
_active_opens: Dict[int, Path] = {}
_PROC_FD_CANDIDATES = (
    Path(f"/proc/{os.getpid()}/fd"),
    Path("/proc/self/fd"),
    Path("/dev/fd"),
)


def _get_thread_native_id() -> Optional[int]:
    get_native_id = getattr(threading, "get_native_id", None)
    if get_native_id is None:
        return None
    try:
        return get_native_id()
    except RuntimeError:
        return None


def _scan_fd_dir(path: Path) -> Optional[Tuple[int, Counter[str]]]:
    try:
        iterator = os.scandir(path)
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
        return None
    counter: Counter[str] = Counter()
    total = 0
    try:
        for entry in iterator:
            total += 1
            target: Optional[str]
            try:
                target = os.readlink(entry.path)
            except OSError:
                target = None
            category = "other"
            if target:
                lower = target.lower()
                if "torch_shm_manager" in lower:
                    category = "torch_shm_manager"
                elif "pipe" in lower:
                    category = "pipe"
                elif "socket" in lower:
                    category = "socket"
            counter[category] += 1
        return total, counter
    except OSError:
        return None
    finally:
        iterator.close()


def _collect_fd_info(candidates: Tuple[Path, ...]) -> Tuple[Optional[int], Optional[Counter[str]]]:
    for candidate in candidates:
        info = _scan_fd_dir(candidate)
        if info is not None:
            return info
    return None, None


def _collect_thread_fd_info(native_thread_id: Optional[int]) -> Tuple[Optional[int], Optional[Counter[str]]]:
    if native_thread_id is None:
        return None, None
    task_dir = Path(f"/proc/{os.getpid()}/task/{native_thread_id}/fd")
    info = _scan_fd_dir(task_dir)
    if info is None:
        return None, None
    return info


def _format_fd_categories(counter: Optional[Counter[str]]) -> str:
    if not counter:
        return ""
    items = [f"{k}:{counter[k]}" for k in (
        "torch_shm_manager",
        "pipe",
        "socket",
        "other",
    ) if counter.get(k)]
    if not items:
        return ""
    return f" fd_types={'/'.join(items)}"


def _log_file_event(phase: str, path: Path, *, idx: int, active_count: Optional[int] = None) -> None:
    if not _log_file_opens:
        return
    worker = get_worker_info()
    worker_id = worker.id if worker is not None else None
    thread_name = threading.current_thread().name
    native_thread_id = _get_thread_native_id()
    proc_fds, proc_fd_types = _collect_fd_info(_PROC_FD_CANDIDATES)
    thread_fds, thread_fd_types = _collect_thread_fd_info(native_thread_id)
    image_opens = f" image_opens={active_count}" if active_count is not None else ""
    proc_fd_summary = _format_fd_categories(proc_fd_types)
    thread_fd_summary = _format_fd_categories(thread_fd_types)
    message = (
        f"{phase} file idx={idx} pid={os.getpid()} thread={thread_name}"
        f" native_tid={native_thread_id} worker={worker_id}"
        f" proc_fds={proc_fds}{proc_fd_summary}"
        f" thread_fds={thread_fds}{thread_fd_summary}{image_opens} path={path}"
    )
    if _log_file_stack:
        stack = "".join(traceback.format_stack(limit=6))
        logger.warning("%s\nTraceback (most recent call last):\n%s", message, stack.rstrip())
    else:
        logger.warning(message)


@contextmanager
def _logged_image_open(path: Path):
    idx = next(_open_sequence)
    with _open_lock:
        _active_opens[idx] = path
        active_count = len(_active_opens)
    _log_file_event("OPEN", path, idx=idx, active_count=active_count)
    try:
        with Image.open(path) as img:
            yield img
    finally:
        with _open_lock:
            _active_opens.pop(idx, None)
            active_count = len(_active_opens)
        _log_file_event("CLOSE", path, idx=idx, active_count=active_count)


def _assert_unit_range(t: torch.Tensor, *, source: Optional[Path] = None) -> None:
    """Ensure tensors that are expected to be in [0, 1] actually are."""
    if not torch.isfinite(t).all():
        loc = f" from {source}" if source is not None else ""
        raise ValueError(f"Non-finite value encountered in image tensor{loc}.")
    min_val = float(t.min())
    max_val = float(t.max())
    if min_val < -1e-5 or max_val > 1.0 + 1e-5:
        loc = f" from {source}" if source is not None else ""
        raise ValueError(
            f"Image tensor{loc} has values outside [0, 1]: min={min_val:.6f}, max={max_val:.6f}"
        )


def list_trajectories(data_root: Path) -> Dict[str, List[Path]]:
    """Return mapping of trajectory directory -> ordered state image paths."""
    out: Dict[str, List[Path]] = {}
    for traj_dir in sorted(data_root.glob("traj_*")):
        state_dir = traj_dir / "states"
        if not state_dir.is_dir():
            continue
        paths = sorted(state_dir.glob("state_*.png"), key=lambda p: int(p.stem.split("_")[1]))
        if paths:
            out[traj_dir.name] = paths
    if not out:
        raise FileNotFoundError(f"No trajectories under {data_root} (expected traj_*/states/state_*.png)")
    return out


def pil_to_tensor(
    img: Image.Image,
    *,
    normalize: Optional[NormalizeFn] = None,
) -> torch.Tensor:
    """Convert PIL image to CHW float tensor in [0, 1], then apply optional normalize."""
    arr = np.array(img, dtype=np.float32, copy=True) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    if normalize is not None:
        t = normalize(t)
    _assert_unit_range(t)
    return t


def load_frame_as_tensor(
    path: Path,
    *,
    size: Tuple[int, int] = (H, W),
    normalize: Optional[NormalizeFn] = None,
    resample: int = Image.BILINEAR,
) -> torch.Tensor:
    """Read state PNG and return normalized tensor."""
    open_ctx = _logged_image_open(path) if _log_file_opens else Image.open(path)
    with open_ctx as img:
        img = img.convert("RGB").resize((size[1], size[0]), resample=resample)
        tensor = pil_to_tensor(img, normalize=normalize)
        _assert_unit_range(tensor, source=path)
    return tensor


def short_traj_state_label(path_str: str) -> str:
    """Return short traj_X/state_Y label from a full filesystem path."""
    base = os.path.normpath(path_str)
    parts = base.split(os.sep)
    traj_idx = next((i for i, p in enumerate(parts) if p.startswith("traj_")), None)
    if (
        traj_idx is not None
        and traj_idx + 2 < len(parts)
        and parts[traj_idx + 1] == "states"
        and parts[traj_idx + 2].startswith("state_")
    ):
        return f"{parts[traj_idx]}/{os.path.splitext(parts[traj_idx + 2])[0]}"
    if len(parts) >= 2:
        return f"{parts[-2]}/{os.path.splitext(parts[-1])[0]}"
    return os.path.splitext(os.path.basename(base))[0]


class PairFromTrajDataset(Dataset):
    """Sample A/B frame pairs from trajectories, optionally across trajectories."""

    def __init__(
        self,
        data_root: Path,
        split: str = "train",
        train_frac: float = 0.95,
        seed: int = 0,
        max_step_gap: int = 20,
        allow_cross_traj: bool = False,
        p_cross_traj: float = 0.0,
        load_frame: Optional[LoadFrameFn] = None,
    ):
        super().__init__()
        self.trajs = list_trajectories(data_root)
        self.traj_items = list(self.trajs.items())
        all_paths = [p for paths in self.trajs.values() for p in paths]

        self._base_seed = seed
        self._main_rng = random.Random(seed)
        self._worker_rngs: Dict[int, random.Random] = {}

        self._main_rng.shuffle(all_paths)
        n_train = int(round(len(all_paths) * train_frac))
        self.pool = all_paths[:n_train] if split == "train" else all_paths[n_train:]
        self.max_step_gap = max_step_gap
        self.allow_cross_traj = allow_cross_traj
        self.p_cross_traj = p_cross_traj if allow_cross_traj else 0.0
        self._load_frame = load_frame if load_frame is not None else load_frame_as_tensor

        pool_set = set(map(str, self.pool))
        self.split_trajs: Dict[str, List[Path]] = {}
        for traj_name, paths in self.traj_items:
            kept = [p for p in paths if str(p) in pool_set]
            if len(kept) >= 2:
                self.split_trajs[traj_name] = kept
        if not self.split_trajs:
            raise RuntimeError(f"No trajectories with >=2 frames in split='{split}'")
        self.split_traj_items = list(self.split_trajs.items())

    def __len__(self) -> int:
        return sum(len(v) for v in self.split_trajs.values())

    def _get_worker_rng(self) -> random.Random:
        info = get_worker_info()
        if info is None:
            return self._main_rng
        wid = info.id
        rng = self._worker_rngs.get(wid)
        if rng is None:
            rng = random.Random(info.seed)
            self._worker_rngs[wid] = rng
        return rng

    def _sample_same_traj_pair(self, rng: random.Random) -> Tuple[Path, Path]:
        traj_idx = rng.randrange(len(self.split_traj_items))
        _, paths = self.split_traj_items[traj_idx]
        if len(paths) < 2:
            return paths[0], paths[-1]
        i0 = rng.randrange(0, len(paths) - 1)
        gap = rng.randint(1, min(max(1, self.max_step_gap), len(paths) - 1 - i0))
        j0 = i0 + gap
        return paths[i0], paths[j0]

    def _sample_cross_traj_pair(self, rng: random.Random) -> Tuple[Path, Path]:
        idx1 = rng.randrange(len(self.split_traj_items))
        idx2 = rng.randrange(len(self.split_traj_items))
        p1s = self.split_traj_items[idx1][1]
        p2s = self.split_traj_items[idx2][1]
        return p1s[rng.randrange(len(p1s))], p2s[rng.randrange(len(p2s))]

    def __getitem__(self, idx: int):  # type: ignore[override]
        rng = self._get_worker_rng()
        if self.allow_cross_traj and (rng.random() < self.p_cross_traj):
            p1, p2 = self._sample_cross_traj_pair(rng)
        else:
            p1, p2 = self._sample_same_traj_pair(rng)
        a = self._load_frame(p1)
        b = self._load_frame(p2)
        return a, b, str(p1), str(p2)


__all__ = [
    "H",
    "W",
    "list_trajectories",
    "pil_to_tensor",
    "load_frame_as_tensor",
    "short_traj_state_label",
    "PairFromTrajDataset",
]
