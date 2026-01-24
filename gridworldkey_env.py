from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import random

import gymnasium as gym
import pygame
import tyro
from gymnasium import spaces
from search_mario import TrajectoryStore
import numpy as np

ENV_ID = "gridworldkey_env"


TILE_SIZE = 15
GRID_PLAY_ROWS = 14
GRID_COLS = 16
DEFAULT_DISPLAY_WIDTH = GRID_COLS * TILE_SIZE
DEFAULT_DISPLAY_HEIGHT = 224
DEFAULT_INVENTORY_HEIGHT = DEFAULT_DISPLAY_HEIGHT - GRID_PLAY_ROWS * TILE_SIZE
BASE_WORLD_SIZE = 32
AGENT_SPEED = 2
AGENT_SIZE = 4
AGENT_RENDER_INSET = 2
AGENT_COLLISION_INSET = 2
KEY_RENDER_INSET = 3
KEY_COLLISION_INSET = 3
ALLOWED_WORLD_SIZES = {224, 128, 64, 32}
MOVEMENT_PATTERN_CHOICES = ("right_only", "loop", "loop_imperfect", "random", "right_left", "random_corner_loops")
THEME_CHOICES = ("basic", "zelda")

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
ACTION_NOOP = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4


def _to_controller_presses(buttons: list[str]) -> np.ndarray:
    presses = np.zeros(NUM_CONTROLLER_BUTTONS, dtype=np.uint8)
    for button in buttons:
        idx = CONTROLLER_STATE_DESC.index(button.upper())
        presses[idx] = 1
    return presses


COMPLEX_DIRECTIONS = [
    [],
    ["up"],
    ["down"],
    ["left"],
    ["right"],
    ["up", "left"],
    ["up", "right"],
    ["down", "left"],
    ["down", "right"],
]
DISCRETE_ACTIONS = [_to_controller_presses(buttons) for buttons in COMPLEX_DIRECTIONS]


@dataclass(frozen=True)
class GridworldLayout:
    tile_size: int
    grid_rows: int
    grid_cols: int
    display_width: int
    display_height: int
    inventory_height: int


def _build_layout(world_size: Optional[int], include_inventory: bool) -> GridworldLayout:
    grid_rows = GRID_PLAY_ROWS
    grid_cols = GRID_COLS
    if world_size is None:
        tile_size = TILE_SIZE
        display_width = DEFAULT_DISPLAY_WIDTH
        if include_inventory:
            display_height = DEFAULT_DISPLAY_HEIGHT
            inventory_height = DEFAULT_INVENTORY_HEIGHT
        else:
            display_height = grid_rows * tile_size
            inventory_height = 0
    else:
        if world_size not in ALLOWED_WORLD_SIZES:
            raise ValueError(f"world_size must be one of {sorted(ALLOWED_WORLD_SIZES)}")
        tile_size = max(1, world_size // grid_cols)
        display_width = grid_cols * tile_size
        if include_inventory:
            display_height = world_size
            inventory_height = max(0, display_height - grid_rows * tile_size)
        else:
            display_height = grid_rows * tile_size
            inventory_height = 0

    return GridworldLayout(
        tile_size=tile_size,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        display_width=display_width,
        display_height=display_height,
        inventory_height=inventory_height,
    )


def _default_grid(rows: int, cols: int, with_obstacles: bool = True) -> np.ndarray:
    grid = np.zeros((rows, cols), dtype=np.int8)

    if not with_obstacles:
        return grid

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

    def __init__(
        self,
        render_mode: Optional[str] = None,
        keyboard_override: bool = True,
        obstacles: bool = True,
        start_manual_control: bool = False,
        agent_size: int = AGENT_SIZE,
        agent_speed: int = AGENT_SPEED,
        world_size: Optional[int] = None,
        hide_key_and_inventory: bool = False,
        black_background: bool = False,
        background_color: Optional[tuple[int, int, int]] = None,
        box_color: Optional[tuple[int, int, int]] = None,
        key_color: Optional[tuple[int, int, int]] = None,
    ):
        self.render_mode = render_mode
        self.keyboard_override = keyboard_override
        self.start_manual_control = start_manual_control
        self.obstacles_enabled = obstacles
        self.include_key = not hide_key_and_inventory
        self.include_inventory = not hide_key_and_inventory
        self.layout = _build_layout(world_size, self.include_inventory)
        self.tile_size = self.layout.tile_size
        self.grid_rows = self.layout.grid_rows
        self.grid_cols = self.layout.grid_cols
        self.display_width = self.layout.display_width
        self.display_height = self.layout.display_height
        self.inventory_height = self.layout.inventory_height
        size_scale = self.display_width / BASE_WORLD_SIZE
        self.agent_size = max(2, int(round(agent_size * size_scale)))
        self.agent_speed = max(1, int(round(agent_speed * size_scale)))
        floor_color = background_color if background_color is not None else ([0, 0, 0] if black_background else COLOR_FLOOR)
        self.color_floor = np.array(floor_color, dtype=np.uint8)
        background_is_black = bool(np.array_equal(self.color_floor, np.array([0, 0, 0], dtype=np.uint8)))
        self.color_wall = np.array([255, 255, 255], dtype=np.uint8) if background_is_black else COLOR_WALL
        self.color_agent = np.array(box_color, dtype=np.uint8) if box_color is not None else (
            np.array([255, 255, 255], dtype=np.uint8) if background_is_black else COLOR_AGENT
        )
        self.color_key = np.array(key_color, dtype=np.uint8) if key_color is not None else COLOR_KEY
        self.color_inventory_bg = (
            np.array([0, 0, 0], dtype=np.uint8) if background_is_black else COLOR_INVENTORY_BG
        )
        self.grid = _default_grid(self.grid_rows, self.grid_cols, with_obstacles=self.obstacles_enabled)
        self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.display_height, self.display_width, 3), dtype=np.uint8
        )

        self.window: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None

        self.agent_x = 0
        self.agent_y = 0
        self.key_tile = (5, 12)
        self.key_present = self.include_key
        self.inventory_has_key = False

        self.base_frame = self._build_base_frame()

        self._episode_steps = 0
        self._keyboard_toggle_prev = False
        self._manual_control = False

    def _tile_top_left(self, row: int, col: int) -> tuple[int, int]:
        y = self.inventory_height + row * self.tile_size
        x = col * self.tile_size
        return x, y

    def _build_base_frame(self) -> np.ndarray:
        frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        frame[:] = self.color_floor
        if self.include_inventory and self.inventory_height > 0:
            frame[: self.inventory_height, :] = self.color_inventory_bg

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                color = self.color_wall if self.grid[row, col] else self.color_floor
                x0, y0 = self._tile_top_left(row, col)
                frame[y0 : y0 + self.tile_size, x0 : x0 + self.tile_size] = color

        return frame

    def _reset_agent(self):
        self._place_agent_at_tile(self.grid_rows - 2, 1)

    def _clamp_agent_position(self, x: int, y: int) -> tuple[int, int]:
        max_x = max(0, self.display_width - self.agent_size)
        max_y = max(self.inventory_height, self.display_height - self.agent_size)
        x = max(0, min(x, max_x))
        y = max(self.inventory_height, min(y, max_y))
        return x, y

    def _place_agent_at_tile(self, row: int, col: int):
        row = max(0, min(self.grid_rows - 1, row))
        col = max(0, min(self.grid_cols - 1, col))
        max_row = int((self.display_height - self.agent_size - self.inventory_height) // self.tile_size)
        max_col = int((self.display_width - self.agent_size) // self.tile_size)
        max_row = max(0, min(self.grid_rows - 1, max_row))
        max_col = max(0, min(self.grid_cols - 1, max_col))
        row = min(row, max_row)
        col = min(col, max_col)
        x, y = self._tile_top_left(row, col)
        self.agent_x, self.agent_y = self._clamp_agent_position(x, y)

    def _draw_key(self, frame: np.ndarray):
        if not self.include_key or not self.key_present:
            return

        key_col = max(0, min(self.grid_cols - 1, self.key_tile[1]))
        key_row = max(0, min(self.grid_rows - 1, self.key_tile[0]))
        x0, y0 = self._tile_top_left(key_row, key_col)

        inset = KEY_RENDER_INSET
        frame[
            y0 + inset : y0 + self.tile_size - inset,
            x0 + inset : x0 + self.tile_size - inset,
        ] = self.color_key

    def _draw_agent(self, frame: np.ndarray):
        x0 = int(self.agent_x)
        y0 = int(self.agent_y)
        inset = AGENT_RENDER_INSET
        frame[
            y0 + inset : y0 + self.agent_size - inset,
            x0 + inset : x0 + self.agent_size - inset,
        ] = self.color_agent

    def _draw_inventory(self, frame: np.ndarray):
        if not self.include_inventory or self.inventory_height <= 0:
            return
        frame[: self.inventory_height, :] = self.color_inventory_bg
        inset = 2
        icon_size = max(1, self.tile_size - inset * 2)
        if self.inventory_has_key and self.include_key:
            x0 = self.display_width - icon_size - inset
            frame[inset : inset + icon_size, x0 : x0 + icon_size] = self.color_key

    def _render_frame(self) -> np.ndarray:
        frame = self.base_frame.copy()
        self._draw_key(frame)
        self._draw_agent(frame)
        self._draw_inventory(frame)
        return frame

    def _slide_axis(self, delta: int, axis: str):
        if delta == 0:
            return
        step = 1 if delta > 0 else -1
        remaining = abs(delta)
        while remaining > 0:
            next_x = self.agent_x + step if axis == "x" else self.agent_x
            next_y = self.agent_y + step if axis == "y" else self.agent_y
            if self._can_occupy(next_x, next_y):
                if axis == "x":
                    self.agent_x = next_x
                else:
                    self.agent_y = next_y
                remaining -= 1
            else:
                break

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
        self.key_present = self.include_key
        self.inventory_has_key = False
        self._manual_control = self.start_manual_control
        self._keyboard_toggle_prev = False
        start_tile = None
        if options is not None:
            start_tile = options.get("start_tile")
        if isinstance(start_tile, tuple) and len(start_tile) == 2:
            self._place_agent_at_tile(int(start_tile[0]), int(start_tile[1]))
        else:
            self._reset_agent()
        observation = self._render_frame()
        return observation, {}

    def _can_occupy(self, x: int, y: int) -> bool:
        left = x + AGENT_COLLISION_INSET
        top = y + AGENT_COLLISION_INSET
        right = x + self.agent_size - AGENT_COLLISION_INSET
        bottom = y + self.agent_size - AGENT_COLLISION_INSET

        if left < 0 or top < self.inventory_height:
            return False
        if right > self.display_width or bottom > self.display_height:
            return False

        row_start = int((top - self.inventory_height) // self.tile_size)
        row_end = int((bottom - 1 - self.inventory_height) // self.tile_size)
        col_start = int(left // self.tile_size)
        col_end = int((right - 1) // self.tile_size)

        row_start = max(0, min(self.grid_rows - 1, row_start))
        row_end = max(0, min(self.grid_rows - 1, row_end))
        col_start = max(0, min(self.grid_cols - 1, col_start))
        col_end = max(0, min(self.grid_cols - 1, col_end))

        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                if self.grid[row, col] == 1:
                    return False
        return True

    def _maybe_collect_key(self) -> bool:
        if not self.include_key or not self.key_present:
            return False

        key_x, key_y = self._tile_top_left(*self.key_tile)
        agent_rect = (
            self.agent_x + AGENT_COLLISION_INSET,
            self.agent_y + AGENT_COLLISION_INSET,
            self.agent_x + self.agent_size - AGENT_COLLISION_INSET,
            self.agent_y + self.agent_size - AGENT_COLLISION_INSET,
        )
        key_rect = (
            key_x + KEY_COLLISION_INSET,
            key_y + KEY_COLLISION_INSET,
            key_x + self.tile_size - KEY_COLLISION_INSET,
            key_y + self.tile_size - KEY_COLLISION_INSET,
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

    def step(self, action: int):
        manual_action = self._read_keyboard_override()

        if self._manual_control:
            controller_action = manual_action if manual_action is not None else np.zeros(NUM_CONTROLLER_BUTTONS, dtype=np.uint8)
        else:
            controller_action = DISCRETE_ACTIONS[int(action)]

        dx = (int(controller_action[BUTTON_RIGHT]) - int(controller_action[BUTTON_LEFT])) * self.agent_speed
        dy = (int(controller_action[BUTTON_DOWN]) - int(controller_action[BUTTON_UP])) * self.agent_speed

        self._slide_axis(dx, axis="x")
        self._slide_axis(dy, axis="y")

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
            "agent_x": int(self.agent_x),
            "agent_y": int(self.agent_y),
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
            self.window = pygame.display.set_mode((self.display_width, self.display_height))
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
class GridworldWorldConfig:
    obstacles: bool = True
    agent_size: int = AGENT_SIZE
    agent_speed: int = AGENT_SPEED
    world_size: Optional[int] = None
    hide_key_and_inventory: bool = False
    black_background: bool = False
    background_color: Optional[tuple[int, int, int]] = None
    box_color: Optional[tuple[int, int, int]] = None
    key_color: Optional[tuple[int, int, int]] = None

    def validate(self):
        if self.world_size is not None and self.world_size not in ALLOWED_WORLD_SIZES:
            raise ValueError(f"world_size must be one of {sorted(ALLOWED_WORLD_SIZES)}")
        for name, value in (
            ("background_color", self.background_color),
            ("box_color", self.box_color),
            ("key_color", self.key_color),
        ):
            if value is None:
                continue
            if len(value) != 3 or any(channel < 0 or channel > 255 for channel in value):
                raise ValueError(f"{name} must be a 3-tuple of 0-255 values")


@dataclass
class GridworldConfig:
    # Run options.
    exp_name: str = Path(__file__).stem
    seed: int = 0
    episodes: int = 3
    max_steps: int = 1500
    keyboard_override: bool = True
    disable_random_movement: bool = False
    dump_trajectories: bool = True
    render_mode: str = "human"
    run_root: Path = Path("runs")

    # Environment theme.
    theme: Optional[Literal["basic", "zelda"]] = None

    # Environment overrides (applied after theme).
    disable_obstacles: Optional[bool] = None
    agent_size: Optional[int] = None
    agent_speed: Optional[int] = None
    world_size: Optional[int] = None
    hide_key_and_inventory: Optional[bool] = None
    black_background: Optional[bool] = None
    movement_pattern: Optional[
        Literal["right_only", "loop", "loop_imperfect", "random", "right_left", "random_corner_loops"]
    ] = None


def _build_run_directory(args: GridworldConfig) -> tuple[Path, str]:
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_prefix = f"{ENV_ID}__{args.exp_name}__{args.seed}"
    run_name = f"{run_prefix}__{date_str}"
    run_dir = Path(args.run_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_name


def _color_tuple(color: np.ndarray) -> tuple[int, int, int]:
    return (int(color[0]), int(color[1]), int(color[2]))


def _apply_theme(config: GridworldWorldConfig, theme: Optional[str]):
    if theme is None:
        return
    elif theme == "basic":
        config.obstacles = False
        config.world_size = 64
        config.hide_key_and_inventory = True
        config.black_background = True
        config.background_color = (0, 0, 0)
        config.box_color = (255, 255, 255)
        config.key_color = _color_tuple(COLOR_KEY)
        return
    elif theme == "zelda":
        config.obstacles = True
        config.world_size = 224
        config.hide_key_and_inventory = False
        config.black_background = False
        config.background_color = _color_tuple(COLOR_FLOOR)
        config.box_color = _color_tuple(COLOR_AGENT)
        config.key_color = _color_tuple(COLOR_KEY)
        return
    else:
        raise ValueError(f"theme must be one of: {', '.join(THEME_CHOICES)}")


def _build_world_config(args: GridworldConfig) -> GridworldWorldConfig:
    config = GridworldWorldConfig()
    _apply_theme(config, args.theme)
    if args.disable_obstacles is not None:
        config.obstacles = not args.disable_obstacles
    if args.agent_size is not None:
        config.agent_size = args.agent_size
    if args.agent_speed is not None:
        config.agent_speed = args.agent_speed
    if args.world_size is not None:
        config.world_size = args.world_size
    if args.hide_key_and_inventory is not None:
        config.hide_key_and_inventory = args.hide_key_and_inventory
    if args.black_background is not None:
        config.black_background = args.black_background
    if config.black_background:
        config.background_color = config.background_color or (0, 0, 0)
        config.box_color = config.box_color or (255, 255, 255)
    config.background_color = config.background_color or _color_tuple(COLOR_FLOOR)
    config.box_color = config.box_color or _color_tuple(COLOR_AGENT)
    config.key_color = config.key_color or _color_tuple(COLOR_KEY)
    config.validate()
    return config


def _with_noops(actions: list[int], rng: np.random.Generator, chance: float = 0.25, max_noops: int = 2) -> list[int]:
    output: list[int] = []
    for action in actions:
        output.append(action)
        if rng.random() < chance:
            output.extend([ACTION_NOOP] * int(rng.integers(1, max_noops + 1)))
    return output


def _imperfect(actions: list[int], rng: np.random.Generator, chance: float = 0.08) -> list[int]:
    output: list[int] = []
    for action in actions:
        output.append(action)
        if rng.random() < chance:
            output.append(int(rng.choice([ACTION_NOOP, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT])))
    return output


def _steps_for_tiles(tiles: int, tile_size: int, speed: int) -> int:
    pixels = max(0, tiles) * tile_size
    return max(1, int(round(pixels / max(1, speed))))


def _build_pattern_actions(
    name: str, env: GridworldKeyEnv, rng: np.random.Generator, max_steps: int
) -> tuple[list[int], tuple[int, int]]:
    pattern = name.lower()
    start_lower_left = (env.grid_rows - 2, 1)
    start_middle = (env.grid_rows // 2, env.grid_cols // 2)

    if pattern == "right_only":
        start_tile = start_lower_left
        start_x, _ = env._tile_top_left(*start_tile)
        max_x = env.display_width - env.agent_size - 1
        steps_right = max(1, int((max_x - start_x) / max(1, env.agent_speed)))
        base_actions = [ACTION_RIGHT] * steps_right
        return _with_noops(base_actions, rng), start_tile

    elif pattern == "right_left":
        start_tile = start_lower_left
        start_x, _ = env._tile_top_left(*start_tile)
        max_x = env.display_width - env.agent_size - 1
        steps_right = max(1, int((max_x - start_x) / max(1, env.agent_speed)))
        base_actions = [ACTION_RIGHT] * steps_right + [ACTION_LEFT] * steps_right
        return _with_noops(base_actions, rng), start_tile

    elif pattern == "random":
        start_tile = start_middle
        actions = [int(rng.choice([ACTION_NOOP, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT])) for _ in range(max_steps)]
        return actions, start_tile

    elif pattern == "loop":
        start_tile = start_lower_left
        actions: list[int] = []
        base_sizes = [(5, 4), (7, 6), (6, 8)]
        for width_tiles, height_tiles in base_sizes:
            right_steps = _steps_for_tiles(width_tiles, env.tile_size, env.agent_speed)
            up_steps = _steps_for_tiles(height_tiles, env.tile_size, env.agent_speed)
            segment_actions = (
                [ACTION_RIGHT] * right_steps
                + [ACTION_NOOP] * 2
                + [ACTION_UP] * up_steps
                + [ACTION_NOOP] * 2
                + [ACTION_LEFT] * right_steps
                + [ACTION_NOOP] * 2
                + [ACTION_DOWN] * up_steps
                + [ACTION_NOOP] * 2
            )
            actions.extend(segment_actions)
        return actions, start_tile

    elif pattern == "loop_imperfect":
        start_tile = start_lower_left
        loops = int(rng.integers(2, 4))
        actions: list[int] = []
        max_loop = min(env.grid_cols - 3, env.grid_rows - 3)
        min_loop = max(3, max_loop // 2)
        for _ in range(loops):
            loop_size = int(rng.integers(min_loop, max_loop + 1))
            width_tiles = max(2, loop_size + int(rng.integers(-1, 2)))
            height_tiles = max(2, loop_size + int(rng.integers(-1, 2)))
            right_steps = _steps_for_tiles(width_tiles, env.tile_size, env.agent_speed)
            up_steps = _steps_for_tiles(height_tiles, env.tile_size, env.agent_speed)
            segment_actions = (
                [ACTION_RIGHT] * right_steps
                + [ACTION_UP] * up_steps
                + [ACTION_LEFT] * right_steps
                + [ACTION_DOWN] * up_steps
            )
            actions.extend(_imperfect(_with_noops(segment_actions, rng, chance=0.2, max_noops=2), rng, chance=0.06))
        return actions, start_tile

    elif pattern == "random_corner_loops":
        start_tile = start_middle
        actions: list[int] = []
        random_steps = min(max_steps, max(12, int(max_steps * 0.25)))
        actions.extend(
            [int(rng.choice([ACTION_NOOP, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT])) for _ in range(random_steps)]
        )
        left_steps = _steps_for_tiles(env.grid_cols, env.tile_size, env.agent_speed)
        down_steps = _steps_for_tiles(env.grid_rows, env.tile_size, env.agent_speed)
        actions.extend([ACTION_LEFT] * left_steps + [ACTION_DOWN] * down_steps)

        nudge_right = _steps_for_tiles(1, env.tile_size, env.agent_speed)
        nudge_up = _steps_for_tiles(1, env.tile_size, env.agent_speed)
        actions.extend([ACTION_RIGHT] * nudge_right + [ACTION_UP] * nudge_up)

        loop_sizes = [
            (env.grid_cols - 3, env.grid_rows - 3),
            (env.grid_cols - 5, env.grid_rows - 5),
            (env.grid_cols - 4, env.grid_rows - 6),
        ]
        for width_tiles, height_tiles in loop_sizes:
            width_tiles = max(2, width_tiles)
            height_tiles = max(2, height_tiles)
            right_steps = _steps_for_tiles(width_tiles, env.tile_size, env.agent_speed)
            up_steps = _steps_for_tiles(height_tiles, env.tile_size, env.agent_speed)
            segment_actions = (
                [ACTION_RIGHT] * right_steps
                + [ACTION_UP] * up_steps
                + [ACTION_LEFT] * right_steps
                + [ACTION_DOWN] * up_steps
            )
            actions.extend(_with_noops(segment_actions, rng, chance=0.15, max_noops=2))
        return actions, start_tile

    else:
        raise ValueError(f"movement_pattern must be one of: {', '.join(MOVEMENT_PATTERN_CHOICES)}")


def main():
    args = tyro.cli(GridworldConfig)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    run_dir, run_name = _build_run_directory(args)
    print(f"Run dir: {run_dir}")
    print(f"Run name: {run_name}")

    world_config = _build_world_config(args)
    env = GridworldKeyEnv(
        render_mode=args.render_mode,
        keyboard_override=args.keyboard_override,
        obstacles=world_config.obstacles,
        start_manual_control=args.disable_random_movement,
        agent_size=world_config.agent_size,
        agent_speed=world_config.agent_speed,
        world_size=world_config.world_size,
        hide_key_and_inventory=world_config.hide_key_and_inventory,
        black_background=world_config.black_background,
        background_color=world_config.background_color,
        box_color=world_config.box_color,
        key_color=world_config.key_color,
    )

    trajectory_store = None
    if args.dump_trajectories:
        traj_dir = run_dir / "traj_dumps"
        trajectory_store = TrajectoryStore(
            traj_dir, image_shape=(env.display_height, env.display_width, 3)
        )

    stop_running = False
    pattern_actions = None
    pattern_start_tile = None
    if args.movement_pattern is not None:
        pattern_actions, pattern_start_tile = _build_pattern_actions(
            args.movement_pattern, env, rng, args.max_steps
        )

    for episode_idx in range(args.episodes):
        if stop_running:
            break

        if pattern_start_tile is not None:
            observation, _ = env.reset(options={"start_tile": pattern_start_tile})
        else:
            observation, _ = env.reset()
        if env.render_mode == "human":
            env.render()

        if pattern_actions is not None:
            action_sequence = pattern_actions
        else:
            action_sequence = [int(rng.integers(0, len(DISCRETE_ACTIONS))) for _ in range(args.max_steps)]

        for step_idx, selected_action in enumerate(action_sequence):
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

            current_observation = observation

            observation, reward, terminated, truncated, info = env.step(selected_action)

            executed_action = info["controller"]

            if trajectory_store is not None:
                action_array = executed_action.copy()
                info_payload = info.copy()
                info_payload["step_index"] = step_idx
                trajectory_store.record_state_action(current_observation, action_array, info_payload)

            if terminated or truncated:
                break

        if pattern_actions is not None:
            stop_running = True

        if trajectory_store is not None and trajectory_store.states:
            trajectory_store.save()

    if trajectory_store is not None and trajectory_store.states:
        trajectory_store.save()

    env.close()


if __name__ == "__main__":
    main()
