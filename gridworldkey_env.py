from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import random

import gymnasium as gym
import pygame
import tyro
from gymnasium import spaces
from search_mario import TrajectoryStore
import numpy as np

ENV_ID = "gridworldkey_env"


TILE_SIZE = 14
GRID_ROWS = 15
GRID_COLS = 16
INVENTORY_HEIGHT = TILE_SIZE
DISPLAY_HEIGHT = INVENTORY_HEIGHT + GRID_ROWS * TILE_SIZE
DISPLAY_WIDTH = GRID_COLS * TILE_SIZE
AGENT_SPEED = 2
AGENT_SIZE = TILE_SIZE
AGENT_RENDER_INSET = 2
AGENT_COLLISION_INSET = 2
KEY_RENDER_INSET = 3
KEY_COLLISION_INSET = 3

COLOR_FLOOR = np.array([233, 220, 188], dtype=np.uint8)
COLOR_WALL = np.array([148, 101, 64], dtype=np.uint8)
COLOR_AGENT = np.array([66, 167, 70], dtype=np.uint8)
COLOR_KEY = np.array([250, 220, 90], dtype=np.uint8)
COLOR_INVENTORY_BG = np.array([60, 60, 60], dtype=np.uint8)

CONTROLLER_STATE_DESC = ["A", "B", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT"]
BUTTON_A = 0
BUTTON_B = 1
BUTTON_SELECT = 2
BUTTON_START = 3
BUTTON_UP = 4
BUTTON_DOWN = 5
BUTTON_LEFT = 6
BUTTON_RIGHT = 7
NUM_CONTROLLER_BUTTONS = len(CONTROLLER_STATE_DESC)


def _default_grid() -> np.ndarray:
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int8)

    # Surrounding walls.
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    # Internal pillars / partitions.
    grid[3:12, 5] = 1
    grid[4:11, 10] = 1
    grid[8, 2:7] = 1
    grid[9, 9:13] = 1

    return grid


class GridworldKeyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, keyboard_override: bool = True):
        self.render_mode = render_mode
        self.keyboard_override = keyboard_override
        self.grid = _default_grid()
        self.action_space = spaces.MultiBinary(NUM_CONTROLLER_BUTTONS)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8
        )

        self.window: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None

        self.agent_x = 0
        self.agent_y = 0
        self.key_tile = (5, 12)
        self.key_present = True
        self.inventory_has_key = False

        self.base_frame = self._build_base_frame()

        self._episode_steps = 0
        self._keyboard_toggle_prev = False
        self._manual_control = False

    def _tile_top_left(self, row: int, col: int) -> tuple[int, int]:
        y = INVENTORY_HEIGHT + row * TILE_SIZE
        x = col * TILE_SIZE
        return x, y

    def _build_base_frame(self) -> np.ndarray:
        frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        frame[:INVENTORY_HEIGHT, :] = COLOR_INVENTORY_BG

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                color = COLOR_WALL if self.grid[row, col] else COLOR_FLOOR
                x0, y0 = self._tile_top_left(row, col)
                frame[y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE] = color

        return frame

    def _reset_agent(self):
        start_tile = (GRID_ROWS - 2, 1)
        x, y = self._tile_top_left(*start_tile)
        self.agent_x = x
        self.agent_y = y

    def _draw_key(self, frame: np.ndarray):
        if not self.key_present:
            return

        key_col = max(0, min(GRID_COLS - 1, self.key_tile[1]))
        key_row = max(0, min(GRID_ROWS - 1, self.key_tile[0]))
        x0, y0 = self._tile_top_left(key_row, key_col)

        inset = KEY_RENDER_INSET
        frame[y0 + inset : y0 + TILE_SIZE - inset, x0 + inset : x0 + TILE_SIZE - inset] = COLOR_KEY

    def _draw_agent(self, frame: np.ndarray):
        x0 = int(self.agent_x)
        y0 = int(self.agent_y)
        inset = AGENT_RENDER_INSET
        frame[
            y0 + inset : y0 + AGENT_SIZE - inset,
            x0 + inset : x0 + AGENT_SIZE - inset,
        ] = COLOR_AGENT

    def _draw_inventory(self, frame: np.ndarray):
        if not self.inventory_has_key:
            return

        inset = 2
        x0 = DISPLAY_WIDTH - TILE_SIZE + inset
        y0 = inset
        frame[y0 : INVENTORY_HEIGHT - inset, x0 : DISPLAY_WIDTH - inset] = COLOR_KEY

    def _render_frame(self) -> np.ndarray:
        frame = self.base_frame.copy()
        self._draw_key(frame)
        self._draw_agent(frame)
        self._draw_inventory(frame)
        return frame

    def _read_keyboard_override(self) -> Optional[np.ndarray]:
        if not self.keyboard_override or self.render_mode != "human":
            return None
        if not pygame.get_init():
            return None

        pygame.event.pump()
        keys = pygame.key.get_pressed()
        if not keys:
            return None

        toggle_pressed = bool(keys[pygame.K_x])
        if toggle_pressed and not self._keyboard_toggle_prev:
            self._manual_control = not self._manual_control
        self._keyboard_toggle_prev = toggle_pressed

        mapping = [
            (pygame.K_w, BUTTON_UP),
            (pygame.K_a, BUTTON_LEFT),
            (pygame.K_s, BUTTON_DOWN),
            (pygame.K_d, BUTTON_RIGHT),
        ]
        manual_press_detected = any(keys[key_code] for key_code, _ in mapping)

        if manual_press_detected and not self._manual_control:
            self._manual_control = True

        if not self._manual_control:
            return None

        action = np.zeros(NUM_CONTROLLER_BUTTONS, dtype=np.uint8)
        for key_code, action_index in mapping:
            if keys[key_code]:
                action[action_index] = 1

        return action

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._episode_steps = 0
        self.key_present = True
        self.inventory_has_key = False
        self._manual_control = False
        self._keyboard_toggle_prev = False
        self._reset_agent()
        observation = self._render_frame()
        return observation, {}

    def _can_occupy(self, x: int, y: int) -> bool:
        left = x + AGENT_COLLISION_INSET
        top = y + AGENT_COLLISION_INSET
        right = x + AGENT_SIZE - AGENT_COLLISION_INSET
        bottom = y + AGENT_SIZE - AGENT_COLLISION_INSET

        if left < 0 or top < INVENTORY_HEIGHT:
            return False
        if right > DISPLAY_WIDTH or bottom > DISPLAY_HEIGHT:
            return False

        row_start = int((top - INVENTORY_HEIGHT) // TILE_SIZE)
        row_end = int((bottom - 1 - INVENTORY_HEIGHT) // TILE_SIZE)
        col_start = int(left // TILE_SIZE)
        col_end = int((right - 1) // TILE_SIZE)

        row_start = max(0, min(GRID_ROWS - 1, row_start))
        row_end = max(0, min(GRID_ROWS - 1, row_end))
        col_start = max(0, min(GRID_COLS - 1, col_start))
        col_end = max(0, min(GRID_COLS - 1, col_end))

        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                if self.grid[row, col] == 1:
                    return False
        return True

    def _maybe_collect_key(self) -> bool:
        if not self.key_present:
            return False

        key_x, key_y = self._tile_top_left(*self.key_tile)
        agent_rect = (
            self.agent_x + AGENT_COLLISION_INSET,
            self.agent_y + AGENT_COLLISION_INSET,
            self.agent_x + AGENT_SIZE - AGENT_COLLISION_INSET,
            self.agent_y + AGENT_SIZE - AGENT_COLLISION_INSET,
        )
        key_rect = (
            key_x + KEY_COLLISION_INSET,
            key_y + KEY_COLLISION_INSET,
            key_x + TILE_SIZE - KEY_COLLISION_INSET,
            key_y + TILE_SIZE - KEY_COLLISION_INSET,
        )

        overlap = not (
            agent_rect[2] <= key_rect[0]
            or agent_rect[0] >= key_rect[2]
            or agent_rect[3] <= key_rect[1]
            or agent_rect[1] >= key_rect[3]
        )

        if overlap:
            self.key_present = False
            self.inventory_has_key = True
            return True

        return False

    def step(self, action: np.ndarray):
        manual_action = self._read_keyboard_override()

        if self._manual_control:
            controller_action = manual_action if manual_action is not None else np.zeros(NUM_CONTROLLER_BUTTONS, dtype=np.uint8)
        else:
            controller_action = np.array(action, dtype=np.uint8).reshape(NUM_CONTROLLER_BUTTONS)

        dx = (int(controller_action[BUTTON_RIGHT]) - int(controller_action[BUTTON_LEFT])) * AGENT_SPEED
        dy = (int(controller_action[BUTTON_DOWN]) - int(controller_action[BUTTON_UP])) * AGENT_SPEED

        target_x = int(self.agent_x + dx)
        target_y = int(self.agent_y + dy)

        if self._can_occupy(target_x, target_y):
            self.agent_x = target_x
            self.agent_y = target_y

        key_collected_now = self._maybe_collect_key()
        reward = 1.0 if key_collected_now else 0.0
        terminated = False
        truncated = False
        self._episode_steps += 1

        observation = self._render_frame()
        info = {
            "has_key": self.inventory_has_key,
            "key_collected_this_step": key_collected_now,
            "controller": controller_action.copy(),
            "manual_control": bool(self._manual_control),
            "step_index": int(self._episode_steps),
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        frame = self._render_frame()

        if self.render_mode == "rgb_array":
            return frame

        if self.render_mode != "human":
            return None

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surface = pygame.surfarray.make_surface(np.swapaxes(frame, 0, 1))
        self.window.blit(surface, (0, 0))
        pygame.display.flip()
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


@dataclass
class GridworldRunnerArgs:
    exp_name: str = Path(__file__).stem
    seed: int = 0
    episodes: int = 3
    max_steps: int = 1500
    keyboard_override: bool = True
    dump_trajectories: bool = True
    render_mode: str = "human"
    run_root: Path = Path("runs")


def _build_run_directory(args: GridworldRunnerArgs) -> tuple[Path, str]:
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_prefix = f"{ENV_ID}__{args.exp_name}__{args.seed}"
    run_name = f"{run_prefix}__{date_str}"
    run_dir = Path(args.run_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_name

def main():
    args = tyro.cli(GridworldRunnerArgs)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    run_dir, run_name = _build_run_directory(args)
    print(f"Run dir: {run_dir}")
    print(f"Run name: {run_name}")

    trajectory_store = None
    if args.dump_trajectories:
        traj_dir = run_dir / "traj_dumps"
        trajectory_store = TrajectoryStore(traj_dir, image_shape=(DISPLAY_HEIGHT, DISPLAY_WIDTH, 3))

    env = GridworldKeyEnv(render_mode=args.render_mode, keyboard_override=args.keyboard_override)

    stop_running = False

    for episode_idx in range(args.episodes):
        if stop_running:
            break

        observation, _ = env.reset()
        if env.render_mode == "human":
            env.render()

        for step_idx in range(args.max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    stop_running = True
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    print("Detected 'q' key press. Exiting run.")
                    stop_running = True
                    break

            if stop_running:
                break

            random_action = rng.integers(0, 2, size=NUM_CONTROLLER_BUTTONS, dtype=np.uint8)
            current_observation = observation

            observation, reward, terminated, truncated, info = env.step(random_action)

            executed_action = info["controller"]

            if trajectory_store is not None:
                action_array = executed_action.copy()
                info_payload = info.copy()
                info_payload["step_index"] = step_idx
                trajectory_store.record_state_action(current_observation, action_array, info_payload)

            if terminated or truncated:
                break

        if trajectory_store is not None and trajectory_store.states:
            trajectory_store.save()

    if trajectory_store is not None and trajectory_store.states:
        trajectory_store.save()

    env.close()


if __name__ == "__main__":
    main()
