#!/usr/bin/env python3
"""Shared trajectory dump utility used by gridworld and Mario search scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class TrajectoryStore:
    def __init__(self, dump_dir: Path, image_shape: tuple[int, int, int] = (224, 240, 3)):
        self.dump_dir = dump_dir
        self.image_shape = image_shape

        self.traj_id = 0
        self.states = []
        self.actions = []
        self.infos = {}

    def record_state_action(self, state: np.ndarray, action: np.ndarray, info: dict[str, Any]) -> None:
        assert state.shape == self.image_shape, f"Unexpected state shape: {state.shape} != {self.image_shape}"

        self.states.append(state.copy())
        self.actions.append(action.copy())

        if not self.infos:
            for key in info.keys():
                self.infos[key] = []

        assert self.infos.keys() == info.keys(), f"Mismatched keys: {self.infos.keys()} != {info.keys()}"
        for key, value in info.items():
            self.infos[key].append(value)

    def save(self, traj_subdir: str | None = None) -> None:
        if not traj_subdir:
            traj_subdir = f"traj_{self.traj_id}"

        dump_states_dir = self.dump_dir / traj_subdir / "states"
        dump_actions_path = self.dump_dir / traj_subdir / "actions.npz"
        dump_infos_path = self.dump_dir / traj_subdir / "infos.npz"

        print(f"Writing trajectory[{self.traj_id}] (#={len(self.actions)}) to: {self.dump_dir}")
        print(f"  {dump_states_dir}")
        print(f"  {dump_actions_path}")
        print(f"  {dump_infos_path}")

        dump_states_dir.mkdir(parents=True, exist_ok=True)

        for i, state in enumerate(self.states):
            dump_states_path = dump_states_dir / f"state_{i}.png"
            obs_img = Image.fromarray(state, mode="RGB")
            obs_img.save(dump_states_path)

        np.savez(dump_actions_path, self.actions)
        np.savez(dump_infos_path, **self.infos)

        self.states = []
        self.actions = []
        self.infos = {}
        self.traj_id += 1
