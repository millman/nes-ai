#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import tyro

from super_mario_env import _describe_controller_vector


@dataclass
class Args:
    traj_dir: Annotated[Path, tyro.conf.Positional] = Path("runs/smb-search-v0__search_mario__1__2025-10-28_22-20-11/traj_dumps/traj_0")
    """Path to a trajectory dump directory (expects actions.npz inside)."""


def main():
    args = tyro.cli(Args)
    actions_path = args.traj_dir / "actions.npz"

    with np.load(actions_path, allow_pickle=True) as data:
        actions = data[data.files[0]]

    num_frames = len(actions)
    frame_width = len(str(num_frames - 1)) if num_frames else 1

    for frame_idx, controller in enumerate(actions):
        controller_desc = _describe_controller_vector(controller)
        print(f"[{frame_idx:>{frame_width}}]: {controller_desc}")


if __name__ == "__main__":
    main()
