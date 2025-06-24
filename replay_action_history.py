#!/usr/bin/env python3

import os
import pickle
import random
import time
from dataclasses import dataclass
from datetime import datetime

import gymnasium as gym
import numpy as np
import pygame
import tyro

from super_mario_env_search import SuperMarioEnv, _to_controller_presses

from gymnasium.envs.registration import register

register(
    id="smb-search-v0",
    entry_point=SuperMarioEnv,
    max_episode_steps=None,
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""

    # Configuration.
    action_history_filename: str = "runs/smb-search-v0__search_mario__1__2025-06-23_19-31-20/action_history/level_1-1_3008_end.pkl"

    # Algorithm specific arguments
    env_id: str = "smb-search-v0"


def make_env(env_id: str, idx: int, capture_video: bool, run_name: str, headless: bool, world_level: tuple[int, int]):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            raise RuntimeError("STOP")
        else:
            render_mode = "rgb" if headless else "human"
            env = gym.make(env_id, render_mode=render_mode, world_level=world_level, screen_rc=(1,1))

        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env

    return thunk


def main():
    args = tyro.cli(Args)

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_prefix = f"{args.env_id}__{args.exp_name}__{args.seed}"
    run_name = f"{run_prefix}__{date_str}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load action history.
    with open(args.action_history_filename, "rb") as infile:
        action_info = pickle.load(infile)

    save_state = action_info['start_save_state']
    action_history = action_info['action_history']
    start_world_level = action_info['start_world_level']

    print(f"Loaded action history: {args.action_history_filename}")
    print(f"  level: {start_world_level}")
    print(f"  action history: #={len(action_history)}")

    # env setup
    capture_video = False
    headless = False
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, 0, capture_video, run_name, headless, start_world_level)],
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )

    first_env = envs.envs[0].unwrapped
    nes = first_env.nes

    controller = _to_controller_presses([])

    # Initialize game state.
    envs.reset()

    nes.load(save_state)
    envs.step((controller,))

    user_control = False

    if False:
        print("Replay action history:")
        for i, action in enumerate(action_history):
            print(f"  [{i}] {action}")

    for i, action in enumerate(action_history):
        controller = action

        # Execute action.
        _next_obs, reward, termination, truncation, info = envs.step((controller,))

        if pygame.K_x in nes.keys_pressed:
            user_control = not user_control

        # Clear out user key presses.
        nes.keys_pressed = []

        # Delay frame for user control.
        if user_control:
            time.sleep(1.0 / 60)


if __name__ == "__main__":
    main()
