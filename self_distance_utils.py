"""Utilities for computing self-distance metrics across trajectory frames."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from predict_mario_ms_ssim import default_transform
from trajectory_utils import list_state_frames, list_traj_dirs
from torchvision.models import ResNet18_Weights, resnet18


@dataclass
class TrajectorySelfDistance:
    """Self-distance results for a single trajectory."""

    traj_name: str
    relative_dir: Path
    frame_paths: List[Path]
    embeddings: torch.Tensor  # [N, D]
    l2_distances: torch.Tensor  # [N]
    cosine_distances: torch.Tensor  # [N]


def load_backbone(device: torch.device) -> nn.Module:
    """Return a pretrained ResNet18 backbone without the classifier head."""
    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    backbone.eval().to(device)
    return backbone


def compute_self_distance_results(
    traj_dir: Path,
    device: torch.device,
    max_trajs: Optional[int] = None,
    progress: bool = True,
) -> List[TrajectorySelfDistance]:
    """Compute embeddings and self-distances for trajectories under `traj_dir`.

    Returns a list of `TrajectorySelfDistance`, one per processed trajectory.
    """

    transform = default_transform()
    backbone = load_backbone(device)

    traj_paths = list_traj_dirs(traj_dir)
    if max_trajs is not None:
        traj_paths = traj_paths[:max_trajs]
    if not traj_paths:
        return []

    iterator: Iterable[Path]
    if progress:
        iterator = tqdm(traj_paths, desc="Trajectories", unit="traj")
    else:
        iterator = traj_paths

    results: List[TrajectorySelfDistance] = []
    for traj_path in iterator:
        states_dir = traj_path / "states"
        if not states_dir.is_dir():
            continue
        frame_paths = list_state_frames(states_dir)
        if len(frame_paths) < 2:
            continue

        embeddings: List[torch.Tensor] = []
        frames_iter: Iterable[Path]
        if progress:
            frames_iter = tqdm(
                frame_paths,
                desc=f"Embedding {traj_path.name}",
                unit="frame",
                leave=False,
            )
        else:
            frames_iter = frame_paths

        with torch.no_grad():
            for frame_path in frames_iter:
                with Image.open(frame_path).convert("RGB") as img:
                    frame_tensor = transform(img).unsqueeze(0).to(device)
                feat = backbone(frame_tensor).squeeze(0).cpu()
                embeddings.append(feat)

        if not embeddings:
            continue

        feats = torch.stack(embeddings)
        ref = feats[0]
        diffs = feats - ref
        l2 = torch.norm(diffs, dim=1)
        cosine = 1.0 - F.cosine_similarity(feats, ref.unsqueeze(0), dim=1)

        rel_name = traj_path.relative_to(traj_dir)
        traj_name = rel_name.as_posix().replace('/', '_')

        results.append(
            TrajectorySelfDistance(
                traj_name=traj_name,
                relative_dir=rel_name,
                frame_paths=frame_paths,
                embeddings=feats,
                l2_distances=l2,
                cosine_distances=cosine,
            )
        )

    return results


def copy_frames_for_visualization(
    traj_results: Sequence[TrajectorySelfDistance],
    dest_root: Path,
) -> List[List[Path]]:
    """Copy frames referenced in `traj_results` into `dest_root` preserving structure.

    Returns a nested list of destination paths aligned with `traj_results[i].frame_paths`.
    """
    copied_paths: List[List[Path]] = []
    for res in traj_results:
        traj_dir = dest_root / res.relative_dir
        traj_dir.mkdir(parents=True, exist_ok=True)
        dst_paths: List[Path] = []
        for frame_path in res.frame_paths:
            dst = traj_dir / frame_path.name
            if not dst.exists():
                dst.write_bytes(frame_path.read_bytes())
            dst_paths.append(dst)
        copied_paths.append(dst_paths)
    return copied_paths
