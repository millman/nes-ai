#!/usr/bin/env python3

import copy
import os
import pickle
import random
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import gymnasium as gym
import numpy as np
import pygame
import tyro
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from search_mario_actions import ACTION_INDEX_TO_CONTROLLER, CONTROLLER_TO_ACTION_INDEX, build_controller_transition_matrix, flip_buttons_by_action_in_place
from search_mario_viz import build_patch_histogram_rgb, draw_patch_grid, draw_patch_path, optimal_patch_layout
from super_mario_env_search import SuperMarioEnv, SCREEN_H, SCREEN_W, get_x_pos, get_y_pos, get_level, get_world, _to_controller_presses, get_time_left, life
from super_mario_env_ram_hacks import encode_world_level

from gymnasium.envs.registration import register

register(
    id="smb-search-v0",
    entry_point=SuperMarioEnv,
    max_episode_steps=None,
)


NdArrayUint8 = np.ndarray[np.dtype[np.uint8]]
NdArrayRGB8 = np.ndarray[tuple[Literal[3]], np.dtype[np.uint8]]


@dataclass(frozen=True)
class PatchId:
    patch_x: int
    patch_y: int
    action_bucket: int
    jump_count: int

    def __post_init__(self):
        # Convert value from np.uint8 to int.
        object.__setattr__(self, 'patch_x', int(self.patch_x))
        object.__setattr__(self, 'patch_y', int(self.patch_y))

    def __repr__(self) -> str:
        jump_str = f",j:{self.jump_count}" if self.jump_count else ""
        return f"PatchId({self.patch_x},{self.patch_y},a:{self.action_bucket}{jump_str})"

    __str__ = __repr__


@dataclass(frozen=True)
class PatchIdX:
    patch_x: int
    action_bucket: int
    jump_count: int

    @staticmethod
    def from_patch(patch_id: PatchId) -> 'PatchIdX':
        return PatchIdX(patch_id.patch_x, patch_id.action_bucket, patch_id.jump_count)


@dataclass(frozen=True)
class SaveInfo:
    save_id: int
    x: int
    y: int
    level: int
    world: int
    level_ticks: int
    ticks_left: int
    jump_count: int
    save_state: Any
    action_history: list
    patch_history: deque[PatchId]
    visited_patches_x: set[PatchIdX]
    controller_state: NdArrayUint8
    controller_state_user: NdArrayUint8

    def __post_init__(self):
        # Convert value from np.uint8 to int.
        object.__setattr__(self, 'world', int(self.world))
        object.__setattr__(self, 'level', int(self.level))
        object.__setattr__(self, 'x', int(self.x))
        object.__setattr__(self, 'y', int(self.y))

        # Convert value from list to np.uint8.
        assert self.controller_state.dtype == np.uint8, f"Unexpected controller state type: {self.controller_state.dtype} != np.uint8"

        assert self.jump_count == self.patch_history[-1].jump_count, f"Mismatched patch jump_count on save: history[-1]:{self.patch_history[-1].jump_count} != jump_count:{self.jump_count}"


@dataclass(frozen=True)
class ReservoirId:
    patch_history: tuple[PatchId, ...]

    def __repr__(self) -> str:
        s = "ReservoirId["
        for i, p in enumerate(self.patch_history):
            if i > 0:
                s += ", "
            s += f"({p.patch_x},{p.patch_y})"
        s += "]"
        return s

    __str__ = __repr__


@dataclass
class ReservoirStats:

    # Number of times visited from any trajectories.
    num_visited: int = 0

    # Number of times selected as a start point.
    num_selected: int = 0

    # The total number of unique, previously unvisited cells discovered by exploring from this cell.
    num_children: int = 0

    # Number of new children found from this specific cell since last choosing this cell as a start point.
    num_children_since_last_selected: int = 0

    transitioned_to_reservoir: Counter[ReservoirId] = field(default_factory=Counter)


class PatchReservoir:

    def __init__(self, patch_size: int, action_bucket_size: int, reservoir_history_length: int, max_saves_per_reservoir: int = 1):
        self.patch_size = patch_size
        self.action_bucket_size = action_bucket_size
        self.reservoir_history_length = reservoir_history_length
        self.max_saves_per_reservoir = max_saves_per_reservoir

        self._reservoir_to_saves = {}

    def reservoir_id_from_state(self, patch_history: list[PatchId]) -> ReservoirId:
        assert len(patch_history) <= self.reservoir_history_length
        return ReservoirId(tuple(patch_history))

    def reservoir_id_from_save(self, save: SaveInfo) -> ReservoirId:
        return self.reservoir_id_from_state(save.patch_history)

    def add(self, save: SaveInfo):
        reservoir_id = self.reservoir_id_from_save(save)

        saves_in_reservoir = self._reservoir_to_saves.setdefault(reservoir_id, [])

        if len(saves_in_reservoir) < self.max_saves_per_reservoir:
            # Reservoir is still small, add it.
            saves_in_reservoir.append(save)
        else:
            # Replace the save that took the longest to reach this patch.
            if True:
                # Find the save state with the most action steps.  We assume that it's better to
                # get to a state with fewer action steps.
                max_index, max_item = max(enumerate(saves_in_reservoir), key=lambda i_and_save: len(i_and_save[1].action_history))

                # Replace the save state with the most action steps.  We assume that it's better to
                # get to a state with fewer action steps.
                #
                # Bucket the action history size, since we only want to consider meaningfully better histories,
                # not just a second or 2 difference.

                if len(save.action_history) < len(max_item.action_history):
                    # Replace item, remove from patch index.
                    saves_in_reservoir[max_index] = save
                else:
                    # Don't replace item, do nothing.
                    pass


    def values(self) -> list[SaveInfo]:
        return (
            save
            for saves in self._reservoir_to_saves.values()
            for save in saves
        )

    def __len__(self) -> int:
        return len(self._reservoir_to_saves)


@dataclass
class ActionHistory:

    parent_history: Optional['ActionHistory']

    # The size of the parent history and all of its parents.  Does not include this node's action history.
    # Store to avoid needing a recursive call to find out the current total action history size.
    ancestors_history_size: int

    action_history: list[NdArrayUint8]

    def append(self, action: NdArrayUint8):
        self.action_history.append(action)

    def __len__(self) -> int:
        return self.ancestors_history_size + len(self.action_history)

    @staticmethod
    def continue_from_parent(parent: 'ActionHistory') -> 'ActionHistory':
        return ActionHistory(
            parent_history=parent,
            ancestors_history_size=len(parent),
            action_history=[],
        )

    def copy(self) -> 'ActionHistory':
        return ActionHistory(
            parent_history=self.parent_history,
            ancestors_history_size=self.__len__(),
            action_history=self.action_history.copy(),
        )


_DEBUG_SCORE_PATCH = False


def _score_reservoir(
    res_id: ReservoirId,
    res_stats: ReservoirStats,
    depth: int,
    max_possible_transitions: int,
    max_num_children_since_last_selected: int,
) -> float:
    # Score recommended by Go-Explore paper: https://arxiv.org/abs/1901.10995, an estimate of recent productivity.
    # If the patch has recently produced children, keep exploring.  As the patch gets selected more,
    # its productivity will drop.
    #
    # Here are the metrics we're considering.  Some metrics are "higher the better" and some are "lower the better".
    # We need to transform each metric in a way such that it's range is in line with other scores, and all scores
    # are "higher the better".
    #
    #     num_selected: (lower is better)
    #       Number of times selected from the archive, lower means less explored, which is more desirable.
    #
    #     num_visited: (lower is better)
    #       Number of times we've ever visited this state, lower means less explored, which is more desirable.
    #
    #     num_children_since_last_selected: (higher is better)
    #       Number of children produced since this cell was last selected.
    #       Higher indicates a productive cell, so continue selecting until it's not productive
    #
    #     [Not implemented] num_selected_since_new_children: (lower is better)
    #       Number of times selected since this cell produced anything.  Lower means we haven't explored it since
    #       we've found new children elsewhere.
    #
    #     [Not implemented] number of unique actions we've tried: (lower is better)
    #       Lower means we haven't tried all of the possible actions from this cell.  There is still more to be
    #       explored here.  (Is this approx equivalent to number of times selected + visited?)
    #
    #     number of unexplored neighbors: (lower is better)
    #       More unexplored neighbors means there's more to be potentially be found searching from this cell.
    #
    #     entropy of transitions to neighbors: (higher is better)
    #       Entropy is higher when we have more reachable states from a cell.  Entropy is 0 when all transitions
    #       lead to the same cell.  For example, when falling in a pit, we can only fall to the state below the
    #       current cell.
    #
    #       Note that this is different from number of unexplored neighbors because entropy considers the *count*
    #       of transitions, not just unique transitions.
    #
    #     action history length / depth of trajectory: higher the better
    #       The longer the action history goes, the more likely we've made progress through the level.
    #
    #       Note that this is a pretty coarse proxy for progress, because if we take a whole lot of noop actions,
    #       we'll still have a long action history, but have not gone anywhere.
    #
    #
    # One possible method of combining all metrics.  Higher-is-better metrics on top, lower-is-better values on bottom:
    #
    #   a*num_children_since_last_selected + b*entropy to neighbors + c*depth
    #   ------------------------------------------------------------
    #   d*num_selected + e*num_visited + f*num_selected_since_new_children
    #
    #
    # Another method, more like what's in the Go-Explore paper, convert each metric to a higher-is-better score independently:
    #
    #   a*(1/(num_selected+1)) +                      (0 -> 1)
    #   b*(1/(num_visited+1)) +                       (0 -> 1)
    #   c*(1/num_selected_since_new_children) +       (0 -> 1)
    #   d*num_children_since_last_selected +          (0 -> inf)?
    #   e*entropy_to_neighbors +                      (0 -> max_entropy)
    #   f*depth                                       (0 -> depth)
    #
    # For scores that have a large range (i.e. not 0->1), we need to further transform:
    #
    # Convert depth to a reasonable range, but must be a penalty for shorter, not reward for longer.
    #   - 1/(depth / 256 + 1)                         (-1 -> 0)
    #
    #
    # There is a max number of children based on how much we let a trajectory advance at once.
    # We can make this a relative measure, by comparing to the maximum number of children since last
    # selected by any cell.
    #   num_children_since_last_selected / (max_num_children_since_last_selected+1)  (0 -> 1)
    #

    productivity_score = res_stats.num_children_since_last_selected / (max_num_children_since_last_selected + 1)

    # Less selection means higher score.  We want to prefer exploring unselected cells.
    selected_score = 1 / (res_stats.num_selected + 1)

    # Less visits means higher score.  We want to prefer exploring unvisited cells.
    visited_score = 1 / (res_stats.num_visited + 1)

    # Higher depth means higher score.  Note that the range is (-1 -> 0), -1 for 0 depth, and 0 for infinite depth.
    # Realistically, depth will range to about 6000.
    depth_score = - 1 / (depth + 1)


    # If the patch always transitions to the same next patch, that's an indicator that we can't explore from there.
    # For example, if Mario is falling in a pit, Mario can't really control where he will end up.
    #
    # Example calculation:
    #  transitions = [
    #    (1,2,3),
    #    (1,2,3),
    #    (4,5,6),
    #    (4,5,6),
    #    (7,8,9),
    #  ]
    #
    # (1,2,3) → 2 times
    # (4,5,6) → 2 times
    # (7,8,9) → 1 time
    #
    # So the probabilities are [2/5, 2/5, 1/5], and the entropy is:
    #
    # - (2/5) * log2(2/5) * 2 - (1/5) * log2(1/5) ≈ 1.5219 bits
    total = res_stats.transitioned_to_reservoir.total()
    counts = np.fromiter(res_stats.transitioned_to_reservoir.values(), dtype=np.float64)

    # The problem with using entropy directly is that it confuses productivity.  If we've taken only a single
    # transition from a patch, then it will have entropy of zero.  There are too few transitions to make a good
    # determination of entropy.  Instead, we need to assume that entropy is high when we have few transitions,
    # and calculate entropy directly when we have a lot of transitions.
    #
    # We want an uncertainy-aware entropy.

    # Laplace smoothing is simple, so we'll use it here.
    if False:
        if len(counts) == 0:
            # We have no transitions.  Assume there's a single transition.
            counts = np.full(1, fill_value=1, dtype=np.float64)

        alpha = 1.0
        k = max_possible_transitions

        probs = (counts + alpha) / (total + k*alpha)
        transition_entropy = -np.sum(probs * np.log2(probs))

    # Threshold entropy.  When we have fewer than the max possible transitions, just assume we have max entropy.
    if True:
        threshold = max_possible_transitions / 2.0

        max_entropy = np.log2(max_possible_transitions)

        if total < threshold:
            # Max possible entropy is a single count for every possible transition.
            transition_entropy = max_entropy
        else:
            probs = counts / total
            # Reminder: entropy is positive, because it's a negative times a negative from the log.
            transition_entropy = -np.sum(probs * np.log2(probs))        # TODO(millman): this is slow

    # Ensure the transition score includes the number of times the state is selected.  It's
    # possible that Mario is about to die, in which case we want to reduce the probability
    # that this cell gets selected.  Otherwise, Mario can get stuck on this cell if it's
    # entropy score is very high.
    transition_score = transition_entropy / (max_entropy + 1)

    # Prefer states that have unexplored neighbors.  This makes it more likely that we pick patches
    # near the frontier of exploration.
    #
    # Range is (-1 -> 0), -1 for cells with all transitions, 0 for cells with no transitions.
    frontier_score = - len(res_stats.transitioned_to_reservoir) / max_possible_transitions

    # Some sample values for score parts:
    #   productivity_score=1.0 transition_entropy=0.7219 total=5 max_possible_transitions=4
    #   productivity_score=0.5 transition_entropy=2.0000 total=2 max_possible_transitions=4
    #   productivity_score=1.0 transition_entropy=1.9219 total=5 max_possible_transitions=4
    #   productivity_score=1.0 transition_entropy=1.0000 total=6 max_possible_transitions=4
    #   productivity_score=0.5 transition_entropy=1.9219 total=5 max_possible_transitions=4
    #   productivity_score=0.5 transition_entropy=1.4488 total=7 max_possible_transitions=4
    #   productivity_score=1.0 transition_entropy=0.9852 total=7 max_possible_transitions=4

    if _DEBUG_SCORE_PATCH:
        print(f"Scored reservoir: {res_id}: productivity_score={productivity_score:.4f} transition_entropy={transition_entropy:.4f} transition_score={transition_score:.4f} frontier_score={frontier_score:.4f} {total=} {max_possible_transitions=}")

    # TODO(millman): separate the frontier into x and y components, then weight them differently.  Vertical frontier is not
    #   really interesting.

    p_coef = 1.0
    t_coef = 0.2
    f_coef = 0.2
    s_coef = 1.0
    v_coef = 1.0
    d_coef = 0.1
    score = (
        p_coef * productivity_score +
        t_coef * transition_score +
        f_coef * frontier_score +
        s_coef * selected_score +
        v_coef * visited_score +
        d_coef * depth_score
    )

    return score


def _choose_save_from_stats(saves_reservoir: PatchReservoir, reservoirs_stats: ReservoirStats, rng: Any) -> SaveInfo:
    valid_res_ids = []
    valid_res_stats = []
    valid_res_depths = []

    max_possible_transitions = 1
    max_num_children_since_last_selected = 1

    reservoirs_with_missing_saves = {}
    for res_id, res_stats in reservoirs_stats.items():
        saves_in_res = saves_reservoir._reservoir_to_saves.get(res_id, [])
        if saves_in_res:
            valid_res_ids.append(res_id)
            valid_res_stats.append(res_stats)
            valid_res_depths.append(len(saves_in_res[0].action_history))

            max_possible_transitions = max(max_possible_transitions, len(res_stats.transitioned_to_reservoir))
            max_num_children_since_last_selected = max(max_num_children_since_last_selected, res_stats.num_children_since_last_selected)
        else:
            reservoirs_with_missing_saves[res_id] = res_stats

    # TODO(millman): I think we can end up with a reservoir without a save if we track stats for
    #   terminate.  So I think this is ok.
    if _DEBUG_SCORE_PATCH and reservoirs_with_missing_saves:
        print("Missing saves for reservoirs:")
        for res_id, res_stats in reservoirs_with_missing_saves.items():
            print(f"  {res_id}: {res_stats}")
        raise AssertionError(f"Missing saves for reservoirs: #={len(reservoirs_with_missing_saves)}")

    # For a given patch size and Mario movement speed, we can have different max possible transitions.
    # Since mario can jump a number of pixels at a time, he may transition from 1, 2, or even more
    # patches away if the patch size is small.  And, some levels teleport mario back to different
    # locations, either through pipes or mazes.
    #
    # So, we'll just assume the max possible transitions is the max that we've seen so far, not the max
    # theoretical.  Note we want the number of unique transitions, not the count of transitions.

    # Calculate reservoir scores.
    scores = np.fromiter((
        _score_reservoir(
            res_id,
            res_stats,
            depth=res_depth,
            max_possible_transitions=max_possible_transitions,
            max_num_children_since_last_selected=max_num_children_since_last_selected,
        )
        for res_id, res_stats, res_depth in zip(valid_res_ids, valid_res_stats, valid_res_depths)
    ), dtype=np.float64)

    # Pick reservoir based on score.  Deterministic.
    chosen_res_index = np.argmax(scores)
    chosen_res = valid_res_ids[chosen_res_index]

    if _DEBUG_SCORE_PATCH:
        print(f"Picked reservoir: {chosen_res} score={scores[chosen_res_index]} {chosen_res=}")

    # Pick the first item out of the reservoir.
    saves_list = saves_reservoir._reservoir_to_saves[chosen_res]
    assert len(saves_list) == 1, f"Implement sampling within the reservoir, found: {len(saves_list)}"

    sample = saves_list[0]

    patch_id_and_weight_pairs = Counter()
    for res_id, score in zip(valid_res_ids, scores):
        current_score = patch_id_and_weight_pairs[res_id.patch_history[-1]]
        patch_id_and_weight_pairs[res_id.patch_history[-1]] = max(current_score, score)

    return sample, list(patch_id_and_weight_pairs.items())


def _validate_save_state(save_info: SaveInfo, ram: NdArrayUint8):
    world = get_world(ram)
    level = get_level(ram)
    x = get_x_pos(ram)
    y = get_y_pos(ram)
    lives = life(ram)
    ticks_left = get_time_left(ram)

    print(f"Validate save state:")
    print(f"  world: {save_info.world} =? {world} -> {save_info.world == world}")
    print(f"  level: {save_info.level} =? {level} -> {save_info.level == level}")
    print(f"  x:     {save_info.x} =? {x} -> {save_info.x == x}")
    print(f"  y:     {save_info.y} =? {y} -> {save_info.y == y}")
    print(f"  lives: ??? =? {lives} -> ???")
    assert save_info.world == world, f"Mismatched save state!"
    assert save_info.level == level, f"Mismatched save state!"
    assert save_info.x == x, f"Mismatched save state!"
    assert save_info.y == y, f"Mismatched save state!"


def _str_level(world_ram: int, level_ram: int) -> str:
    world, level = encode_world_level(world_ram, level_ram)
    return f"{world}-{level}"


def _seconds_to_hms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{int(hours):02}:{int(minutes):02}:{int(secs):02}"


def _print_saves_list(saves: list[SaveInfo], N: int = 2):
    # Print bottom-N and top-N saves.
    for i, s in enumerate(saves[:N]):
        print(f"  [{i}] {_str_level(s.world, s.level)} x={s.x} y={s.y} save_id={s.save_id}")

    num_top = min(len(saves) - N, N)
    if num_top > 0:
        print('  ...')
        for e, (s, w) in enumerate(saves[-num_top:]):
            i = len(saves) - num_top + e
            print(f"  [{i}] {_str_level(s.world, s.level)} x={s.x} y={s.y} save_id={s.save_id}")


def _print_info(
    time_since_start: float,
    time_since_load: float,
    world: int,
    level: int,
    x: int,
    y: int,
    ticks_left: int,
    saves: PatchReservoir,
    step: int,
    steps_since_load: int,
    patches_x_since_load: int,
):
    if time_since_load == 0:
        steps_per_sec = float('inf')
    else:
        steps_per_sec = steps_since_load / time_since_load

    approx_max_actions = max((
        len(saves_in_res[-1].action_history)
        for _r_id, saves_in_res in saves._reservoir_to_saves.items()
    ), default=0)

    # screen_x = ram[0x006D]
    # level_screen_x = ram[0x071A]
    # screen_pos = ram[0x0086]

    print(
        f"{_seconds_to_hms(time_since_start)} "
        f"level={_str_level(world, level)} "
        f"x={x} y={y} ticks_left={ticks_left} "
        f"saves={len(saves)} "
        f"steps/sec={steps_per_sec:.4f} "
        f"steps_since_load={steps_since_load} "
        f"patches_x_since_load={patches_x_since_load} "
        f"max_actions={approx_max_actions}"
    )


def _print_histogram(patch_id_and_weight_pairs: list[tuple[PatchId], float], n_bins: int = 10):
    min_patch_and_weight = min(patch_id_and_weight_pairs, key=lambda p_id_and_w: p_id_and_w[1])
    max_patch_and_weight = max(patch_id_and_weight_pairs, key=lambda p_id_and_w: p_id_and_w[1])

    b_min = min_patch_and_weight[1]
    b_max = max_patch_and_weight[1]

    # Quick and dirty histogram.
    bin_width = (b_max - b_min) / n_bins
    bins = np.zeros(n_bins + 1, dtype=np.int64)

    for p_id, w in patch_id_and_weight_pairs:
        bin = int((w  - b_min) / bin_width)
        bins[bin] += 1

    print("patch_id_and_weight_pairs")
    print(f"  min: {min_patch_and_weight[0]}, {min_patch_and_weight[1]:.4f}")
    print(f"  max: {max_patch_and_weight[0]}, {max_patch_and_weight[1]:.4f}")
    print()
    for i, count in enumerate(bins):
        b0 = b_min + i*bin_width
        b1 = b_min + (i+1)*bin_width
        print(f"  [{b0:.2f} -> {b1:.2f}]: {count}")


def _print_ram_debug(ram: NdArrayUint8):
    # Debug printouts for level pos
    print(f"DEBUG POS:")
    print(f"  world:      ram[0x75f]: {ram[0x75f]}")
    print(f"  stage:      ram[0x75c]: {ram[0x75f]}")
    print(f"  area:       ram[0x760]: {ram[0x760]}")
    print(f"  screen_x:   ram[0x 6d]: {ram[0x6d]}")
    print(f"  offset_x:   ram[0x 86]: {ram[0x86]}")
    print(f"  y_pixel:    ram[0x3b8]: {ram[0x3b8]}")
    print(f"  y_pos:      ram[0x ce]: {ram[0xce]}")
    print(f"  y_viewport: ram[0x b5]: {ram[0xb5]}")
    print(f"  screen:     ram[0x71a]: {ram[0x71a]}")
    print(f"  next scrn:  ram[0x71b]: {ram[0x71b]}")

    # Nothing seems like a problem?

    # Before pipe:
    #     DEBUG POS:
    #     world:      ram[0x75f]: 1
    #     stage:      ram[0x75c]: 1
    #     area:       ram[0x760]: 2
    #     screen_x:   ram[0x 6d]: 11
    #     offset_x:   ram[0x 86]: 56
    #     y_pixel:    ram[0x3b8]: 144
    #     y_pos:      ram[0x ce]: 144
    #     y_pixel:    ram[0x b5]: 1
    #     screen:     ram[0x71a]: 11
    #     next scrn:  ram[0x71b]: 11

    # After pipe:
    #     DEBUG POS:
    #     world:      ram[0x75f]: 1
    #     stage:      ram[0x75c]: 1
    #     area:       ram[0x760]: 2
    #     screen_x:   ram[0x 6d]: 11
    #     offset_x:   ram[0x 86]: 82
    #     y_pixel:    ram[0x3b8]: 160
    #     y_pos:      ram[0x ce]: 160
    #     y_viewport: ram[0x b5]: 1
    #     screen:     ram[0x71a]: 11
    #     next scrn:  ram[0x71b]: 12


_MAX_LEVEL_DIST = 6400


def _history_length_for_level(default_history_length: int, natural_world_level: tuple[int, int]) -> int:
    WORLD_LEVEL_TO_LEN = {
        # (7, 4): 20,
        # (8, 4): 3,
    }

    world, level = natural_world_level

    if (world, level) not in WORLD_LEVEL_TO_LEN:
        return default_history_length

    custom_len = WORLD_LEVEL_TO_LEN[(world, level)]
    if default_history_length < custom_len:
        print(f"Increasing history length, required for level {world}-{level}: {custom_len}")
        return custom_len
    else:
        print(f"Satisfied history length, required for level {world}-{level}: {default_history_length}")
        return default_history_length


def _save_ram_snapshot(ram: NdArrayUint8, filename: str):
    with open(filename, 'w') as f:
        for i, byte in enumerate(ram):
            f.write(f"[0x{i:04x}]: {byte}\n")
    print(f"Wrote ram snapshot to: {filename}")


def _save_xy_transitions(filepath: Path, world_ram: int, level_ram: int, xy_transitions: Counter, desc: str = None):
    transitions_count = xy_transitions.total()
    desc_with_sep = "_" + desc if desc else ""

    path_xy_transitions = filepath / f"level_{_str_level(world_ram, level_ram)}_{transitions_count}{desc_with_sep}.pkl"
    path_xy_transitions.parent.mkdir(parents=True, exist_ok=True)

    with open(path_xy_transitions, "wb") as out:
        pickle.dump(xy_transitions, out)

    print(f"Wrote xy transitions to: {path_xy_transitions}")


def _save_action_history(filepath: Path, world_ram: int, level_ram: int, start_save_state: Any, action_history: ActionHistory, desc: str = None):
    action_history_size = len(action_history)
    desc_with_sep = "_" + desc if desc else ""

    filepath_action_history = filepath / f"level_{_str_level(world_ram, level_ram)}_{action_history_size}{desc_with_sep}.pkl"
    filepath_action_history.parent.mkdir(parents=True, exist_ok=True)

    current_node = action_history

    # Convert action history into a list.
    segments_newest_to_oldest = []
    while current_node.parent_history:
        segments_newest_to_oldest.append(current_node.action_history)
        current_node = current_node.parent_history

    flattened_action_history = [
        action
        for segment in reversed(segments_newest_to_oldest)
        for action in segment
    ]

    with open(filepath_action_history, "wb") as out:
        pickle.dump({
            'action_history': flattened_action_history,
            'start_save_state': start_save_state,
        }, out)

    print(f"Wrote action history to: {filepath_action_history}")


@dataclass
class Args:
    r"""
    Run example:
        > WANDB_API_KEY=<key> python3 ppo_nes.py --wandb-project-name mariorl --track

        ...
        wandb: Tracking run with wandb version 0.19.9
        wandb: Run data is saved locally in /Users/dave/rl/nes-ai/wandb/run-20250418_130130-SuperMarioBros-v0__ppo_nes__1__2025-04-18_13-01-30
        wandb: Run `wandb offline` to turn off syncing.
        wandb: Syncing run SuperMarioBros-v0__ppo_nes__1__2025-04-18_13-01-30
        wandb: ⭐️ View project at https://wandb.ai/millman-none/mariorl
        wandb: 🚀 View run at https://wandb.ai/millman-none/mariorl/runs/SuperMarioBros-v0__ppo_nes__1__2025-04-18_13-01-30

    Resume example:
        > WANDB_API_KEY=<key> WANDB_RUN_ID=SuperMarioBros-v0__ppo_nes__1__2025-04-18_13-01-30 WANDB_RESUME=must python3 ppo_nes.py --wandb-project-name mariorl --track

        ...
        wandb: Tracking run with wandb version 0.19.9
        wandb: Run data is saved locally in /Users/dave/rl/nes-ai/wandb/run-20250418_133317-SuperMarioBros-v0__ppo_nes__1__2025-04-18_13-01-30
        wandb: Run `wandb offline` to turn off syncing.
    --> wandb: Resuming run SuperMarioBros-v0__ppo_nes__1__2025-04-18_13-01-30
        wandb: ⭐️ View project at https://wandb.ai/millman-none/mariorl
        wandb: 🚀 View run at https://wandb.ai/millman-none/mariorl/runs/SuperMarioBros-v0__ppo_nes__1__2025-04-18_13-01-30
        ...
    --> resumed at update 9
        ...
    """

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "MarioRL"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    wandb_run_id: str | None = None
    """the id of a wandb run to resume"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    checkpoint_frequency: float = 30
    """create a checkpoint every N seconds"""
    train_agent: bool = True
    """enable or disable training of the agent"""

    # Configuration.
    headless: bool = False
    print_freq_sec: float = 1.0
    start_level: tuple[int,int] = (7,4)

    # Specific experiments
    patch_size: int = 20

     # About 30 frames/tick
    action_bucket_size: int = 1500

    # NOTE: Need enough history to distinguish between paths for 7-4 and 8-4; works with len=20.
    #   All other levels are much faster with history of length 3, or even 1.  This value may be
    #   overwritten elsewhere based on the level, if the specified value is too small.
    #
    # NOTE: Level 2-2, when entering the pipe at the end, there is not an obvious way to tell that
    #   Mario is in a different position.  The x and y coordinates in the above-ground (after pipe)
    #   overlap with the in-water (before pipe).  There must be some other indicator to show where
    #   Mario is in the level.  Until then, we can use a history>1 to make the x,y position unique.
    #   The jump into the x,y patch will be discontinuous, from the right side of the screen.
    #
    reservoir_history_length: int = 2


    # TODO(millman): We should consider the visitation counts based on a patch that includes some amount of history.
    #   Otherwise, we won't know to expand jumps.

    max_trajectory_steps: int = 150
    max_trajectory_patches_x: int = -1
    max_trajectory_revisit_x: int = 2
    max_trajectory_jump_count: int = 1

    flip_prob: float = 0.03

    # Visualization
    vis_freq_sec: float = 0.15

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
            env = gym.make(env_id, render_mode=render_mode, world_level=world_level, screen_rc=(2,2))

        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env

    return thunk


def main():
    args = tyro.cli(Args)

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # NOTE: Run name should be descriptive, but not unique.
    # In particular, we don't include the date because the date does not affect the results.
    # Date prefixes are handled by wandb automatically.

    if not args.wandb_run_id:
        run_prefix = f"{args.env_id}__{args.exp_name}__{args.seed}"
        run_name = f"{run_prefix}__{date_str}"
        args.wandb_run_id = run_name

    run_name = args.wandb_run_id

    if args.track:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            #name=run_name,
            monitor_gym=True,
            save_code=True,
            id=run_name,
        )
        assert run.dir == f"runs/{run_name}"
        run_dir = run.dir
    else:
        run_dir = f"runs/{run_name}"

    filepath_xy_transitions = Path(f"{run_dir}/xy_transitions")
    filepath_action_history = Path(f"{run_dir}/action_history")

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng()

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, 0, args.capture_video, run_name, args.headless, args.start_level)],
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )

    first_env = envs.envs[0].unwrapped
    nes = first_env.nes
    screen = first_env.screen

    # Action setup.
    transition_matrix = build_controller_transition_matrix(actions=ACTION_INDEX_TO_CONTROLLER, flip_prob=args.flip_prob)

    # Global state.
    patch_size = args.patch_size
    step = 0
    next_save_id = 0
    start_time = time.time()
    last_print_time = time.time()
    last_vis_time = time.time()

    # Per-trajectory state.  Resets after every death/level.
    action_history = ActionHistory(ancestors_history_size=0, parent_history=None, action_history=[])
    controller = _to_controller_presses([])

    # Histogram visualization.

    # Approximate the size of the histogram based on how many patches we need.
    _MAX_PATCHES_X = int(np.ceil(_MAX_LEVEL_DIST / patch_size))
    _MAX_PATCHES_Y = int(np.ceil(240 / patch_size))
    _NUM_MAX_PATCHES = _MAX_PATCHES_X * _MAX_PATCHES_Y

    # We'll wrap rows around if they hit the edge of the screen.
    # Figure out how many wraps we need by:
    #
    #   (actual_screen_w / pixel_size) * num_rows = (actual_screen_w / pixel_size) * (actual_screen_h / pixel_size)
    #
    #   patches_per_row * num_rows = patches_per_row * patches_per_col
    #
    #
    # We want the pixel size to be maximized, as long as everything fits on the screen.
    # When everything fits exactly,
    # We don't know the patches_per_row or the patch_pixel_size.
    _HIST_ROWS, _HIST_COLS, _HIST_PIXEL_SIZE = optimal_patch_layout(screen_width=240, screen_height=224, n_patches=_NUM_MAX_PATCHES)

    # Start searching the Mario game tree.
    envs.reset()

    ram = nes.ram()

    reservoir_history_length = _history_length_for_level(args.reservoir_history_length, args.start_level)

    # Initialize to invalid values.  The first step of the loop should act as a new level.
    world = -1
    level = -1
    x = -1
    y = -1
    lives = -1
    ticks_left = -1

    level_ticks = -1

    first_save_in_level = nes.save()
    xy_transitions_in_level = Counter()
    visited_reservoirs_in_level = set()
    patch_history = deque(maxlen=reservoir_history_length)
    visited_patches_x = set()

    jump_count = 0
    revisited_x = 0


    action_bucket_size = args.action_bucket_size

    saves = PatchReservoir(patch_size=patch_size, action_bucket_size=action_bucket_size, reservoir_history_length=reservoir_history_length)
    reservoirs_stats = defaultdict(ReservoirStats)
    force_terminate = False
    steps_since_load = 0
    load_time = time.time()
    patches_x_since_load = 0
    last_selected_res_id = None

    patch_id_and_weight_pairs = []

    original_args = None
    user_args = None

    while True:
        step += 1
        steps_since_load += 1
        now = time.time()

        # Remember previous states.
        prev_world = world
        prev_level = level
        prev_x = x
        prev_y = y
        prev_lives = lives

        # Update action every frame.
        controller = flip_buttons_by_action_in_place(controller, transition_matrix=transition_matrix, action_index_to_controller=ACTION_INDEX_TO_CONTROLLER, controller_to_action_index=CONTROLLER_TO_ACTION_INDEX)

        action_history.append(controller)

        # Execute action.
        _next_obs, reward, termination, truncation, info = envs.step((controller,))

        # Handle user config changes.
        if pygame.K_x in nes.keys_pressed:
            # Switch to user args.
            if user_args is None:
                original_args = copy.deepcopy(args)

                # Update args.
                print("Changing to user config:")

                # TODO(millman): show changed values from namespace.
                args.max_trajectory_steps = -1
                args.max_trajectory_patches_x = -1
                args.max_trajectory_revisit_x = -1

                user_args = args

            # Restore original args.
            else:
                args = copy.deepcopy(original_args)

                # Update args.
                print("Changing to original config:")

                user_args = None

        if pygame.K_1 in nes.keys_pressed:
            _save_ram_snapshot(ram, '/tmp/a.txt')

        if pygame.K_2 in nes.keys_pressed:
            _save_ram_snapshot(ram, '/tmp/b.txt')

        if pygame.K_v in nes.keys_pressed:
            _save_xy_transitions(filepath_xy_transitions, world, level, xy_transitions_in_level)
            _save_action_history(filepath_action_history, world, level, first_save_in_level, action_history)

        # Clear out user key presses.
        nes.keys_pressed = []

        # Read current state.
        world = get_world(ram)
        level = get_level(ram)
        x = get_x_pos(ram)
        y = get_y_pos(ram)
        lives = life(ram)
        ticks_left = get_time_left(ram)

        # If we get teleported, or if the level boundary is discontinuous, the change in x position isn't meaningful.
        if abs(x - prev_x) > 50:
            if world != prev_world or level != prev_level:
                # print(f"Discountinuous x position, level change: {prev_world},{prev_level} -> {world},{level}, x: {prev_x} -> {x}")
                pass
            elif lives != prev_lives:
                # print(f"Discontinuous x position, died, lives: {prev_lives} -> {lives}")
                pass
            else:
                assert world == prev_world and level == prev_level, f"Mismatched level change: {prev_world},{prev_level} -> {world},{level}, x: {prev_x} -> {x}"

                if not args.headless:
                    print(f"Discountinuous x position: {prev_x} -> {x}, jump_count: {jump_count}")

                # Consider jumps a problem only if jumping backwards.  Forward jumps are ok.
                if x - prev_x < 0:
                    jump_count += 1

        patch_id = PatchId(x // patch_size, y // patch_size, len(action_history) // action_bucket_size, jump_count)

        # Update histogram counts every frame.
        xy_transitions_in_level[((prev_x, prev_y), (x, y))] += 1

        # ---------------------------------------------------------------------
        # Trajectory ending criteria
        # ---------------------------------------------------------------------

        if lives < prev_lives and not termination:
            print(f"Lost a life: x={x} ticks_left={ticks_left}")
            raise AssertionError("Missing termination flag for lost life")

        # Stop after some fixed number of steps.  This will force the sampling logic to run more often,
        # which means we won't waste as much time running through old states.
        if args.max_trajectory_steps > 0 and steps_since_load >= args.max_trajectory_steps:
            if not args.headless:
                print(f"Ending trajectory, max steps for trajectory: {steps_since_load}: x={x} ticks_left={ticks_left}")
            force_terminate = True

        elif args.max_trajectory_patches_x > 0 and patches_x_since_load >= args.max_trajectory_patches_x:
            if not args.headless:
                print(f"Ending trajectory, max patches x for trajectory: {patches_x_since_load}: x={x} ticks_left={ticks_left}")
            force_terminate = True

        elif args.max_trajectory_jump_count >= 0 and jump_count > args.max_trajectory_jump_count:
            if not args.headless:
                print(f"Ending trajectory, max jump count for trajectory: jump_count={jump_count} x={x} ticks_left={ticks_left}")
            force_terminate = True

        # If we died, skip.
        elif lives < prev_lives:
            if not args.headless:
                print(f"Lost a life: x={x} ticks_left={ticks_left}")
            force_terminate = True

        # Stop if we double-back to the same x patch within a trajectory.
        #
        # Must check only the x patch, since we can't avoid visiting a state by changing y.
        # If we jump up and down, we'll get back to the same state we were just on.
        #
        # TODO(millman): This seems really powerful for preventing wasting search space, but there
        #   are a few places in the game where mario needs to go back and forth on x, when advancing
        #   the y position.  Maybe if total number of patches revisited is too many?
        #   This is also still not easily getting past 7-4; the specific path and jump is a really
        #   narrow sequence.
        #
        #   Overall, seems like there needs to be a smarter algorithm to find states that are
        #   accessible, but not easily accessible.
        #
        #   Related, level 2-2 has a jump backwards after entering the pipe.
        elif (
            patch_history and
            PatchIdX.from_patch(patch_id) != PatchIdX.from_patch(patch_history[-1]) and
            PatchIdX.from_patch(patch_id) in visited_patches_x and
            world == prev_world and level == prev_level
        ):
            revisited_x += 1

            if args.max_trajectory_revisit_x > 0 and revisited_x > args.max_trajectory_revisit_x:
                if not args.headless:
                    print(f"Ending trajectory, revisited x patch: {patch_id}: x={x} ticks_left={ticks_left}")
                force_terminate = True

        # ---------------------------------------------------------------------
        # Handle new level reached
        # ---------------------------------------------------------------------

        # If we reached a new level, serialize all of the states to disk, then clear the save state buffer.
        # Also dump state histogram.
        if world != prev_world or level != prev_level:
            # Print before-level-end info.
            if True:
                _print_info(time_since_start=now-start_time, time_since_load=now-load_time, world=world, level=level, x=x, y=y, ticks_left=ticks_left, saves=saves, step=step, steps_since_load=steps_since_load, patches_x_since_load=patches_x_since_load)

            # Save end-of-level info.
            if prev_level != -1:
                _save_xy_transitions(filepath_xy_transitions, world, level, xy_transitions_in_level, desc="end")
                _save_action_history(filepath_action_history, world, level, first_save_in_level, action_history)

            # Set number of ticks in level to the current ticks.
            level_ticks = get_time_left(ram)

            # Clear state.
            action_history = ActionHistory(ancestors_history_size=0, parent_history=None, action_history=[])
            xy_transitions_in_level = Counter()
            visited_reservoirs_in_level = set()
            visited_patches_x = set()

            revisited_x = 0
            jump_count = 0

            # We're at the starting patch on a new level, but we may have arrived via a jump.
            # Force the first patch of the new level to have a zero jump count.
            #
            # Observed new level via jump on the transition from 1-3 to 2-1.
            #
            first_patch_id_in_level = patch_id = PatchId(x // patch_size, y // patch_size, len(action_history) // action_bucket_size, jump_count)
            if patch_id != first_patch_id_in_level:
                # NOTE: We actually expect to have an incorrect patch when reaching a new level, because the
                #   patch was created with the action_history and jump_count at the end of the last level.
                print(f"Reached new level from patch, rewrite patch: {patch_id} -> {first_patch_id_in_level}")
                patch_id = first_patch_id_in_level

            # Update history length for this level.  Note that we must get the history length before setting the
            # deque size, since the deque size may change between levels.
            natural_world_level = encode_world_level(world, level)
            reservoir_history_length = _history_length_for_level(args.reservoir_history_length, natural_world_level)

            patch_history = deque(maxlen=reservoir_history_length)
            patch_history.append(patch_id)

            reservoir_id = saves.reservoir_id_from_state(patch_history)

            visited_reservoirs_in_level.add(reservoir_id)
            visited_patches_x.add(PatchIdX.from_patch(patch_id))

            print(f"Starting level: {_str_level(world, level)}")

            # Print after-level-start info.
            if True:
                _print_info(time_since_start=now-start_time, time_since_load=now-load_time, world=world, level=level, x=x, y=y, ticks_left=ticks_left, saves=saves, step=step, steps_since_load=steps_since_load, patches_x_since_load=patches_x_since_load)

            assert lives > 1 and lives < 100, f"How did we end up with lives?: {lives}"

            saves = PatchReservoir(patch_size=patch_size, action_bucket_size=action_bucket_size, reservoir_history_length=reservoir_history_length)
            save_info = SaveInfo(
                save_id=next_save_id,
                x=x,
                y=y,
                level=level,
                world=world,
                level_ticks=level_ticks,
                ticks_left=ticks_left,
                jump_count=jump_count,
                save_state=nes.save(),
                action_history=action_history.copy(),
                patch_history=patch_history.copy(),
                visited_patches_x=visited_patches_x.copy(),
                controller_state=controller.copy(),
                controller_state_user=nes.controller1.is_pressed_user.copy(),
            )
            saves.add(save_info)
            next_save_id += 1

            reservoirs_stats = defaultdict(ReservoirStats)

            # Fill out stats for current patch.  Do not include transitions.
            r_stats = reservoirs_stats[reservoir_id]
            r_stats.num_visited += 1
            r_stats.num_selected += 1
            r_stats.last_selected_step = step
            r_stats.last_visited_step = step

            last_selected_res_id = saves.reservoir_id_from_save(save_info)

        # ---------------------------------------------------------------------
        # Handle patch transitions
        # ---------------------------------------------------------------------
        #
        # Track patch stats even if we terminate on this patch.

        if patch_id != patch_history[-1]:
            if _DEBUG_SCORE_PATCH and (termination or force_terminate):
                print(f"Updating patch stats on terminate: {patch_id}")

            prev_res_id = saves.reservoir_id_from_state(patch_history)

            next_patch_history = patch_history.copy()
            next_patch_history.append(patch_id)

            res_id = saves.reservoir_id_from_state(next_patch_history)

            # Update reservoir stats.
            r_stats = reservoirs_stats[res_id]
            r_stats.num_visited += 1
            r_stats.last_visited_step = step

            if res_id not in visited_reservoirs_in_level:
                r_stats.num_children += 1

                res_last_selected_stats = reservoirs_stats[last_selected_res_id]
                res_last_selected_stats.num_children += 1
                res_last_selected_stats.num_children_since_last_selected += 1

                visited_reservoirs_in_level.add(res_id)

            r_prev_stats = reservoirs_stats[prev_res_id]
            r_prev_stats.transitioned_to_reservoir[res_id] += 1

        # Save state info on transition.  It's ok to save when we intentionally end a trajectory
        # early, but not when the termination is due to the environment (e.g. Mario dying).
        #
        if patch_id != patch_history[-1] and not termination:

            valid_lives = lives > 1 and lives < 100
            valid_x = True  # x < 65500

            # NOTE: Some levels (like 4-4) are discontinuous.  We can get x values of > 65500.
            if not valid_x:
                # TODO(millman): how did we get into a weird x state?  Happens on 4-4.
                print(f"RAM values: ram[0x006D]={ram[0x006D]=} * 256 + ram[0x0086]={ram[0x0086]=}")
                print(f"Something is wrong with the x position, don't save this state: level={_str_level(world, level)} x={x} y={y} lives={lives} ticks-left={ticks_left} saves={len(saves)} actions={len(action_history)}")
                raise AssertionError("STOP")

            if not valid_lives:
                # TODO(millman): how did we get to a state where we don't have full lives?
                print(f"Something is wrong with the lives, don't save this state: level={_str_level(world, level)} x={x} y={y} ticks-left={ticks_left} lives={lives} steps_since_load={steps_since_load} patches_x_since_load={patches_x_since_load}")
                raise AssertionError("STOP")

            patch_id_x = PatchIdX.from_patch(patch_id)

            if patch_id_x not in visited_patches_x:
                visited_patches_x.add(patch_id_x)

            # NOTE: These are any patches since load, not *new* patches load.
            if patch_id_x != PatchIdX.from_patch(patch_history[-1]):
                patches_x_since_load += 1

            patch_history.append(patch_id)

            assert patch_id.jump_count == jump_count

            save_info = SaveInfo(
                save_id=next_save_id,
                x=x,
                y=y,
                level=level,
                world=world,
                level_ticks=level_ticks,
                ticks_left=ticks_left,
                jump_count=jump_count,
                save_state=nes.save(),
                action_history=action_history.copy(),
                patch_history=patch_history.copy(),
                visited_patches_x=visited_patches_x.copy(),
                controller_state=controller.copy(),
                controller_state_user=nes.controller1.is_pressed_user.copy(),
            )
            saves.add(save_info)
            next_save_id += 1

        # ---------------------------------------------------------------------
        # Handle trajectory end
        # ---------------------------------------------------------------------
        #
        # Trajectory can end from either from Mario dying or a forced ending criteria.
        #
        # Note that we want to handle ending trajectories after we've processed the transition.
        # We need to ensure that stats for the new state are handled correctly.

        # If we died, reload from a game state based on heuristic.
        if termination or force_terminate:

            # If we reached a new level, serialize all of the states to disk, then clear the save state buffer.
            # Also dump state histogram.
            if (world != prev_world or level != prev_level) and termination:
                raise AssertionError(f"Reached a new world ({prev_world}-{prev_level} -> {world}-{level}), but also terminated?")

            # In AutoresetMode.DISABLED, we have to reset ourselves.
            # Reset only if we hit a termination state.  Otherwise, we can just reload.
            if termination:
                resets_before = first_env.resets
                envs.reset()

                # Don't count this reset by us, we're trying to find where other things are calling reset.
                first_env.resets -= 1

                resets_after = first_env.resets

                # Check that stepping after termination resets appropriately.
                if resets_after == resets_before and level != prev_level:
                    raise AssertionError("Failed to reset on level change")

            # Start reload process.

             # Reset variables that change on load.
            steps_since_load = 0
            load_time = time.time()
            patches_x_since_load = 0
            force_terminate = False

            # Choose save.
            save_info, patch_id_and_weight_pairs = _choose_save_from_stats(saves, reservoirs_stats, rng=rng)

            # Reload and re-initialize.
            nes.load(save_info.save_state)
            ram = nes.ram()

            # Restore controller to what it was at the point of saving.  The user may have been pressing something here.
            controller[:] = nes.controller1.is_pressed[:]

            # Ensure loading from ram is the same as loading from the controller state.
            if (controller != save_info.controller_state).any():
                if (controller == save_info.controller_state_user).all():
                    print(f"User pressed controls on save: controller:{save_info.controller_state} user:{save_info.controller_state_user}")
                else:
                    raise AssertionError(f"Mismatched controller on load: {controller} != {save_info.controller_state}")

            if False:
                # Flip the buttons with some probability.  If we're loading a state, we don't want to
                # be required to use the same action state that was tried before.  To get faster coverage
                # We flip buttons here with much higher probability than during a trajectory.
                controller = flip_buttons_by_action_in_place(controller, transition_matrix=transition_matrix, action_index_to_controller=ACTION_INDEX_TO_CONTROLLER, controller_to_action_index=CONTROLLER_TO_ACTION_INDEX)

            # Create a new action history with the save as an ancestor.
            action_history = ActionHistory.continue_from_parent(save_info.action_history)

            visited_patches_x = save_info.visited_patches_x.copy()

            revisited_x = 0
            jump_count = save_info.jump_count

            # Read current state.
            world = get_world(ram)
            level = get_level(ram)
            x = get_x_pos(ram)
            y = get_y_pos(ram)
            lives = life(ram)
            ticks_left = get_time_left(ram)

            if save_info.world != world or save_info.level != level or save_info.x != x or save_info.y != y:
                _validate_save_state(save_info, ram)

            # Set prior frame values to current.  There is no difference at load.
            prev_world = world
            prev_level = level
            prev_x = x
            prev_y = y
            prev_lives = lives
            patch_history = save_info.patch_history.copy()

            reservoir_id = saves.reservoir_id_from_save(save_info)
            patch_id = reservoir_id.patch_history[-1]

            assert patch_id == patch_history[-1], f"Mismatched patch history: history[-1]:{patch_history[-1]} != patch_id:{patch_id}"

            level_ticks = save_info.level_ticks

            last_selected_res_id = reservoir_id

            if True:
                res_visited = reservoirs_stats[reservoir_id].num_visited

                if not args.headless:
                    print(f"Loaded save: save_id={save_info.save_id} level={_str_level(world, level)}, x={x} y={y} lives={lives} saves={len(saves)} res_visited={res_visited}")

                if False:
                    _print_saves_list(saves.values())

            # Update reservoir stats.
            res_stats = reservoirs_stats[reservoir_id]
            res_stats.num_selected += 1
            res_stats.num_visited += 1
            res_stats.num_children_since_last_selected = 0
            res_stats.last_selected_step = step
            res_stats.last_visited_step = step

            assert reservoir_id in visited_reservoirs_in_level, f"Missing reservoir_id in visited, even though chosen as start point: {reservoir_id}"

        # ---------------------------------------------------------------------
        # Loop updates
        # ---------------------------------------------------------------------

        # Print stats every second:
        #   * Current position: (x, y)
        #   * Number of states in memory.
        #   * Elapsed time since level start.
        #   * Novel states found (across all trajectories)
        #   * Novel states/sec
        if args.print_freq_sec > 0 and now - last_print_time > args.print_freq_sec:
            _print_info(time_since_start=now-start_time, time_since_load=now-load_time, world=world, level=level, x=x, y=y, ticks_left=ticks_left, saves=saves, step=step, steps_since_load=steps_since_load, patches_x_since_load=patches_x_since_load)
            last_print_time = now

        # Visualize the distribution of save states.
        if not args.headless and args.vis_freq_sec > 0 and now - last_vis_time > args.vis_freq_sec:

            # Draw grid on screen.
            if False:
                obs_hwc = _next_obs[0]
                img_rgb_240 = Image.fromarray(obs_hwc.swapaxes(0, 1), mode='RGB')
                expected_size = (SCREEN_W, SCREEN_H)
                assert img_rgb_240.size == expected_size, f"Unexpected img_rgb_240 size: {img_rgb_240.size} != {expected_size}"
                draw_patch_grid(img_rgb_240, patch_size=patch_size, ram=ram, x=x, y=y)
                screen.blit_image(img_rgb_240, screen_index=4)

            # Histogram of number of saves in each patch reservoir.
            # Collect reservoirs into patches.
            patch_id_to_num_saves = Counter()
            for res_id, saves_in_reservoir in saves._reservoir_to_saves.items():
                patch_id_to_num_saves[res_id.patch_history[-1]] += len(saves_in_reservoir)

            patch_id_and_num_saves_pairs = list(patch_id_to_num_saves.items())

            img_rgb_240 = build_patch_histogram_rgb(
                patch_id_and_num_saves_pairs,
                current_patch=patch_id,
                patch_size=patch_size,
                max_patch_x=_MAX_PATCHES_X,
                max_patch_y=_MAX_PATCHES_Y,
                hist_rows=_HIST_ROWS,
                hist_cols=_HIST_COLS,
                pixel_size=_HIST_PIXEL_SIZE,
            )
            screen.blit_image(img_rgb_240, screen_index=1)

            # Histogram of reservoir visits.
            patch_id_to_num_visits = Counter()
            for res_id, r_stats in reservoirs_stats.items():
                patch_id_to_num_visits[res_id.patch_history[-1]] += r_stats.num_visited

            patch_id_and_seen_count_pairs = list(patch_id_to_num_visits.items())

            img_rgb_240 = build_patch_histogram_rgb(
                patch_id_and_seen_count_pairs,
                current_patch=patch_id,
                patch_size=patch_size,
                max_patch_x=_MAX_PATCHES_X,
                max_patch_y=_MAX_PATCHES_Y,
                hist_rows=_HIST_ROWS,
                hist_cols=_HIST_COLS,
                pixel_size=_HIST_PIXEL_SIZE,
            )
            screen.blit_image(img_rgb_240, screen_index=3)

            if False and patch_id_and_weight_pairs:
                _print_histogram(patch_id_and_weight_pairs)

            # Histogram of sampling weight.
            img_rgb_240 = build_patch_histogram_rgb(
                patch_id_and_weight_pairs,
                current_patch=patch_id,
                patch_size=patch_size,
                max_patch_x=_MAX_PATCHES_X,
                max_patch_y=_MAX_PATCHES_Y,
                hist_rows=_HIST_ROWS,
                hist_cols=_HIST_COLS,
                pixel_size=_HIST_PIXEL_SIZE,
            )
            draw_patch_path(
                img_rgb_240,
                patch_history,
                patch_size=patch_size,
                max_patch_x=_MAX_PATCHES_X,
                max_patch_y=_MAX_PATCHES_Y,
                hist_rows=_HIST_ROWS,
                hist_cols=_HIST_COLS,
                pixel_size=_HIST_PIXEL_SIZE,
            )

            screen.blit_image(img_rgb_240, screen_index=2)

            # Update display.
            screen.show()

            last_vis_time = now

        # Delay frame for user control.
        if user_args is not None:
            time.sleep(1.0 / 60)

            if False:
                _print_ram_debug(ram)


if __name__ == "__main__":
    main()
