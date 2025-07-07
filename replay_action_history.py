#!/usr/bin/env python3

import os
import pickle
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame
import tyro
from PIL import Image

from search_mario import _str_level
from super_mario_env_search import SuperMarioEnv, _to_controller_presses, get_x_pos, get_y_pos, get_level, get_world

from gymnasium.envs.registration import register

register(
    id="smb-search-v0",
    entry_point=SuperMarioEnv,
    max_episode_steps=None,
)

class FrameRecordingWrapper(gym.Wrapper):
    """Copy numpy output frames and transpose for recording"""

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        obs_space = self.observation_space

        new_shape = (obs_space.shape[1], obs_space.shape[0]) + obs_space.shape[2:]
        axes = (1, 0, 2)

        self.axes = axes

        self.observation_space = gym.spaces.Box(
            low=np.transpose(obs_space.low, axes),
            high=np.transpose(obs_space.high, axes),
            shape=new_shape,
            dtype=obs_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return np.transpose(obs, self.axes).copy(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return np.transpose(obs, self.axes).copy(), reward, done, truncated, info

    def render(self, *args, **kwargs):
        frame = self.env.render(*args, **kwargs)
        return np.transpose(frame, self.axes).copy()


_EXAMPLE_RUNS = {
    "1-1": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_1-1_2563_end.pkl",
    "1-2": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_1-2_2060_end.pkl",
    "2-1": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_2-1_2983_end.pkl",
    "2-2": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_2-2_3609_end.pkl",
    "2-3": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_2-3_2253_end.pkl",
    "2-4": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_2-4_1541_end.pkl",
    "3-1": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_3-1_2729_end.pkl",
    "3-2": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_3-2_2251_end.pkl",
    "3-3": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_3-3_1720_end.pkl",
    "3-4": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_3-4_1140_end.pkl",
    "4-1": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_4-1_2214_end.pkl",
    "4-2": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_4-2_2844_end.pkl",
    "4-3": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_4-3_1814_end.pkl",
    "4-4": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_4-4_1912_end.pkl",
    "5-1": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_5-1_2059_end.pkl",
    "5-2": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_5-2_2299_end.pkl",
    "5-3": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_5-3_1387_end.pkl",
    "5-4": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_5-4_1238_end.pkl",
    "6-1": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_6-1_1700_end.pkl",
    "6-2": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_6-2_2689_end.pkl",
    "6-3": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_6-3_1345_end.pkl",
    "6-4": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_6-4_1262_end.pkl",
    "7-1": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_7-1_2416_end.pkl",
    "7-2": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_7-2_3121_end.pkl",
    "7-3": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_7-3_2075_end.pkl",
    "7-4": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_7-4_2084_end.pkl",
    "8-1": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_8-1_3353_end.pkl",
    "8-2": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_8-2_2437_end.pkl",
    "8-3": "runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_8-3_2120_end.pkl",
    "8-4": "runs/smb-search-v0__search_mario__1__2025-06-25_19-42-02/action_history/level_8-4_7739_end.pkl",
}

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Configuration.
    headless: bool = False
    dump_trajectories: bool = False

    action_history_filename: str = _EXAMPLE_RUNS['8-4']

    skip_animation: bool = False
    debug_frames: bool = False

    # Algorithm specific arguments
    env_id: str = "smb-search-v0"


def make_env(env_id: str, idx: int, capture_video: bool, run_name: str, headless: bool, world_level: tuple[int, int], skip_animation: bool):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", world_level=world_level, screen_rc=(1,1), skip_animation=skip_animation)
            env = FrameRecordingWrapper(env)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            render_mode = "rgb_array" if headless else "human"
            env = gym.make(env_id, render_mode=render_mode, world_level=world_level, screen_rc=(1,1), skip_animation=skip_animation)

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
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, 0, args.capture_video, run_name, args.headless, start_world_level, args.skip_animation)],
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )

    first_env = envs.envs[0].unwrapped
    nes = first_env.nes

    # Initialize game state.
    envs.reset()

    nes.load(save_state)

    user_control = False

    if False:
        print("Replay action history:")
        for i, action in enumerate(action_history):
            print(f"  [{i}] {action}")

    # Trajectory dumps.
    if args.dump_trajectories:
        # Make a dump dir named the same as the input directory.  Input example:
        #   runs/smb-search-v0__search_mario__1__2025-06-25_18-03-35/action_history/level_1-1_2563_end.pkl
        action_history_path = Path(args.action_history_filename)
        parts = action_history_path.parts
        assert parts[0] == "runs", f"Expected input file starting with 'runs', found: {parts}"
        assert parts[2] == "action_history", f"Expected input file part[2] == 'action_history', found: {parts[2]}"
        assert parts[3].endswith('.pkl'), f"Expected input file part[3] suffix .pkl, found: {parts[3]}"
        input_run_id = parts[1]
        traj_id = action_history_path.stem

        dump_dir = Path("traj_dumps") / input_run_id / traj_id
        dump_states_dir = dump_dir / 'states'
        dump_actions_path = dump_dir / 'actions.npz'

        dump_states = []
        dump_actions = []

    i = 0
    num_resets = 0
    frames = 0

    # NOTE: We extend the action history by 1 for the even when the last action is at the end of
    #   a level.  If we have end-of-level animations, they will be skipped frames that we can display.
    #
    #   We'll continue until we start a new level.  For the very end of the game, the level value
    #   happens to change, so that works too.
    #
    stop_after_world_level = None

    while i < len(action_history) + 1:
        frames += 1

        if i < len(action_history):
            action = action_history[i]
        else:
            action = _to_controller_presses([])

        ram = nes.ram()

        world = get_world(ram)
        level = get_level(ram)

        if args.debug_frames:
            x = get_x_pos(ram)
            y = get_y_pos(ram)
            print(f"step: {i}, world={world} level={level} x={x} y={y}")

        if stop_after_world_level is not None and (world, level) != stop_after_world_level:
            print(f"Stopping after level change: {_str_level(*stop_after_world_level)} -> {_str_level(world, level)}")
            break

        # Get the state before we apply an action.
        if args.dump_trajectories:
            # Dump state (observation).
            dump_states_path = dump_states_dir / f'state_{frames}.png'
            dump_states_path.parent.mkdir(parents=True, exist_ok=True)
            obs_img = Image.fromarray(first_env._get_obs(), mode='RGB')
            obs_img.save(dump_states_path)

            # Dump action (controller).
            dump_actions.append(action)

        # Execute action.
        _next_obs, reward, termination, truncation, info = envs.step((action,))

        if info.get('skipped_frame'):
            # This frame is skipped, don't consume an action.
            if args.debug_frames:
                print("(skipped frame)")
        else:
            i += 1

            if i == len(action_history):
                # We're past our recording.  Stop at the next level change.
                stop_after_world_level = (world, level)

        if termination:
            num_resets += 1
            print(f"Hit termination, resetting at frame {i}: #resets={num_resets}")
            envs.reset()

        if pygame.K_x in nes.keys_pressed:
            user_control = not user_control

        # Clear out user key presses.
        nes.keys_pressed = []

        # Delay frame for user control.
        if user_control:
            time.sleep(1.0 / 60)


    if args.capture_video:
        print(f"Writing video (frames={frames} actions={len(action_history)}) to: {run_name}")

    if args.dump_trajectories:
        print(f"Writing trajectory (#={len(dump_actions)}) to: {dump_dir}")
        print(f"  {dump_states_dir}")
        print(f"  {dump_actions_path}")

        np.savez(dump_actions_path, dump_actions)


if __name__ == "__main__":
    main()
