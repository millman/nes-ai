#!/usr/bin/env python3
"""
Slice a trajectory directory down to a specific range of steps.

Given a directory that contains actions.npz, infos.npz, and a states/ folder of
state_{i}.png frames, this script writes a NEW directory containing only the
frames/actions/infos that fall within the requested [start, end) range.

The new directory is placed next to the input folder and named
<input_name>__slice_<start>_<end>. The input directory is never modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated
import shutil
from pathlib import Path

import numpy as np
import tyro


@dataclass
class Args:
    """Arguments for slicing a trajectory."""

    trajectory_dir: Annotated[Path, tyro.conf.Positional]
    end: int
    start: int = 0


def validate_range(start: int, end: int):
    if start < 0:
        raise ValueError("start must be non-negative")
    if end <= start:
        raise ValueError("end must be greater than start")


def slice_npz_file(npz_path: Path, start: int, end: int, output_path: Path):
    if not npz_path.exists():
        raise FileNotFoundError(f"Expected file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        sliced = {}
        for key in data.files:
            array = data[key]
            if array.shape[0] < end:
                raise ValueError(
                    f"{npz_path.name}:{key} has only {array.shape[0]} entries, cannot slice up to {end}"
                )
            sliced[key] = array[start:end]

    np.savez_compressed(output_path, **sliced)


def slice_states_folder(states_dir: Path, start: int, end: int, output_states_dir: Path):
    if not states_dir.exists():
        raise FileNotFoundError(f"States directory not found: {states_dir}")

    state_files = sorted(
        states_dir.glob("state_*.png"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not state_files:
        raise ValueError(f"No state_*.png files found in {states_dir}")
    if end > len(state_files):
        raise ValueError(
            f"States folder has only {len(state_files)} frames, cannot slice up to {end}"
        )

    selected = state_files[start:end]
    output_states_dir.mkdir(parents=True, exist_ok=False)
    for new_idx, src in enumerate(selected):
        dst = output_states_dir / f"state_{new_idx}.png"
        shutil.copy2(src, dst)


def main():
    args = tyro.cli(Args)
    trajectory_dir: Path = args.trajectory_dir
    validate_range(args.start, args.end)

    actions_path = trajectory_dir / "actions.npz"
    infos_path = trajectory_dir / "infos.npz"
    states_dir = trajectory_dir / "states"

    output_dir = trajectory_dir.with_name(
        f"{trajectory_dir.name}__slice_{args.start}_{args.end}"
    )
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    output_dir.mkdir(parents=True)
    slice_npz_file(actions_path, args.start, args.end, output_dir / "actions.npz")
    slice_npz_file(infos_path, args.start, args.end, output_dir / "infos.npz")
    slice_states_folder(states_dir, args.start, args.end, output_dir / "states")

    print(
        f"Wrote sliced trajectory to {output_dir} for range [{args.start}, {args.end}) "
        f"({args.end - args.start} steps retained)"
    )


if __name__ == "__main__":
    main()
