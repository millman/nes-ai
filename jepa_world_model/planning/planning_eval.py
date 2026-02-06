#!/usr/bin/env python3
"""Planning diagnostics utilities for JEPA world model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import math

import numpy as np
import torch
from jepa_world_model.plots.plot_action_delta_stats import (
    save_action_delta_stats_plot,
    save_action_delta_strip_plot,
)
from jepa_world_model.plots.plot_grid_execution_trace import save_grid_execution_trace_plot
from jepa_world_model.plots.plot_planning_pca_path import save_planning_pca_path_plot
from gridworldkey_env import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    BUTTON_DOWN,
    BUTTON_LEFT,
    BUTTON_RIGHT,
    BUTTON_UP,
    GridworldKeyEnv,
)
from jepa_world_model.config_planning import PlanningDiagnosticsConfig


DIRECTION_ORDER = ("L", "R", "U", "D")
DIRECTION_TO_ENV_ACTION = {
    "L": ACTION_LEFT,
    "R": ACTION_RIGHT,
    "U": ACTION_UP,
    "D": ACTION_DOWN,
}


@dataclass
class ActionDeltaStats:
    mu: Dict[str, np.ndarray]
    L: Dict[str, float]
    S: Dict[str, float]
    L_scale: float
    r_goal: float
    r_merge: float
    r_cluster_p: float
    q: Dict[str, float]
    inv_lr: float
    inv_ud: float
    noop_ratio: float
    counts: Dict[str, int]


@dataclass
class DatasetGraph:
    centers: np.ndarray
    node_ids_t: np.ndarray
    node_ids_tp1: np.ndarray
    edges: Dict[int, Dict[str, List[int]]]


@dataclass
class PlanResult:
    actions: List[str]
    nodes: List[np.ndarray]


@dataclass
class PlanningTestResult:
    success: bool
    steps: int
    final_p_distance: float
    goal_distance: float
    visited_cells: List[Tuple[int, int]]


def grid_cell_from_env(env: GridworldKeyEnv) -> Tuple[int, int]:
    if env.tile_size <= 0:
        raise AssertionError("env.tile_size must be positive to compute grid cell.")
    row = int((env.agent_y - env.inventory_height) // env.tile_size)
    col = int(env.agent_x // env.tile_size)
    row = max(0, min(env.grid_rows - 1, row))
    col = max(0, min(env.grid_cols - 1, col))
    return row, col


def _action_to_direction(action_vec: np.ndarray) -> Optional[str]:
    if action_vec.ndim != 1:
        raise AssertionError("action_vec must be 1D.")
    pressed = action_vec > 0.5
    if pressed[:BUTTON_UP].any():
        return None
    up = bool(pressed[BUTTON_UP])
    down = bool(pressed[BUTTON_DOWN])
    left = bool(pressed[BUTTON_LEFT])
    right = bool(pressed[BUTTON_RIGHT])
    count = int(up) + int(down) + int(left) + int(right)
    if count == 0:
        return "NOOP"
    if count > 1:
        return None
    if left:
        return "L"
    if right:
        return "R"
    if up:
        return "U"
    if down:
        return "D"
    return None


def action_labels_from_vectors(actions: np.ndarray) -> List[Optional[str]]:
    if actions.ndim != 2:
        raise AssertionError("actions must be 2D for label extraction.")
    return [_action_to_direction(vec) for vec in actions]


def compute_action_delta_stats(
    p_t: np.ndarray,
    p_tp1: np.ndarray,
    actions: np.ndarray,
    *,
    min_action_count: int,
) -> ActionDeltaStats:
    if p_t.shape != p_tp1.shape:
        raise AssertionError("p_t and p_tp1 must have matching shapes.")
    if p_t.ndim != 2:
        raise AssertionError("p_t must be 2D [N, D].")
    if actions.ndim != 2 or actions.shape[0] != p_t.shape[0]:
        raise AssertionError("actions must be 2D and aligned with p_t.")
    deltas = p_tp1 - p_t
    labels: List[Optional[str]] = action_labels_from_vectors(actions)

    mu: Dict[str, np.ndarray] = {}
    L: Dict[str, float] = {}
    S: Dict[str, float] = {}
    q: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for label in [*DIRECTION_ORDER, "NOOP"]:
        mask = np.array([lbl == label for lbl in labels], dtype=bool)
        count = int(mask.sum())
        counts[label] = count
        if count == 0:
            continue
        d = deltas[mask]
        mu[label] = np.median(d, axis=0)
        L[label] = float(np.median(np.linalg.norm(d, axis=1)))
        spread = d - mu[label][None, :]
        S[label] = float(np.median(np.linalg.norm(spread, axis=1)))
        if label != "NOOP":
            denom = max(np.linalg.norm(mu[label]), 1e-8)
            q[label] = S[label] / denom
    for label in DIRECTION_ORDER:
        if counts.get(label, 0) < min_action_count:
            raise AssertionError(f"Not enough '{label}' actions for planning stats: {counts.get(label, 0)}")
    L_scale = float(np.median([L[label] for label in DIRECTION_ORDER]))
    if L_scale <= 0:
        raise AssertionError("Computed L_scale must be positive for planning stats.")
    r_goal = 0.35 * L_scale
    r_merge = 0.25 * L_scale
    r_cluster_p = 0.35 * L_scale
    inv_lr = np.linalg.norm(mu.get("L", 0.0) + mu.get("R", 0.0)) / L_scale
    inv_ud = np.linalg.norm(mu.get("U", 0.0) + mu.get("D", 0.0)) / L_scale
    noop_ratio = L.get("NOOP", 0.0) / L_scale if L_scale > 0 else float("nan")
    return ActionDeltaStats(
        mu=mu,
        L=L,
        S=S,
        L_scale=L_scale,
        r_goal=r_goal,
        r_merge=r_merge,
        r_cluster_p=r_cluster_p,
        q=q,
        inv_lr=float(inv_lr),
        inv_ud=float(inv_ud),
        noop_ratio=float(noop_ratio),
        counts=counts,
    )


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8)
    return 1.0 - float(np.dot(a, b) / denom)


def cluster_latents(latents: np.ndarray, radius: float, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    if latents.ndim != 2:
        raise AssertionError("latents must be [N, D] for clustering.")
    if radius <= 0:
        raise AssertionError("radius must be positive for clustering.")
    centers: List[np.ndarray] = []
    node_ids = np.empty((latents.shape[0],), dtype=np.int64)
    for idx, vec in enumerate(latents):
        if not centers:
            centers.append(vec.copy())
            node_ids[idx] = 0
            continue
        if metric == "cosine":
            distances = np.array([_cosine_distance(vec, c) for c in centers])
        else:
            distances = np.array([np.linalg.norm(vec - c) for c in centers])
        nearest = int(np.argmin(distances))
        if distances[nearest] <= radius:
            node_ids[idx] = nearest
        else:
            node_ids[idx] = len(centers)
            centers.append(vec.copy())
    return np.stack(centers, axis=0), node_ids


def build_dataset_graph(
    latents_t: np.ndarray,
    latents_tp1: np.ndarray,
    actions: np.ndarray,
    *,
    radius: float,
    metric: str,
    max_edge_distance: Optional[float] = None,
    edge_metric: Optional[str] = None,
) -> DatasetGraph:
    if latents_t.shape != latents_tp1.shape:
        raise AssertionError("latents_t and latents_tp1 must share shape.")
    combined = np.concatenate([latents_t, latents_tp1], axis=0)
    centers, node_ids = cluster_latents(combined, radius=radius, metric=metric)
    node_ids_t = node_ids[: latents_t.shape[0]]
    node_ids_tp1 = node_ids[latents_t.shape[0] :]
    edges: Dict[int, Dict[str, List[int]]] = {}
    labels = [_action_to_direction(a) for a in actions]
    for src, dst, label in zip(node_ids_t, node_ids_tp1, labels):
        if label is None:
            continue
        if max_edge_distance is not None:
            if max_edge_distance <= 0:
                raise AssertionError("max_edge_distance must be positive when provided.")
            src_center = centers[int(src)]
            dst_center = centers[int(dst)]
            metric_name = edge_metric or metric
            if metric_name == "cosine":
                dist = _cosine_distance(src_center, dst_center)
            elif metric_name == "l2":
                dist = float(np.linalg.norm(src_center - dst_center))
            else:
                raise AssertionError(f"Unsupported edge metric: {metric_name!r}")
            if dist > max_edge_distance:
                continue
        bucket = edges.setdefault(int(src), {})
        bucket.setdefault(label, []).append(int(dst))
    return DatasetGraph(centers=centers, node_ids_t=node_ids_t, node_ids_tp1=node_ids_tp1, edges=edges)


def bfs_plan(graph: DatasetGraph, start: int, goal: int) -> Optional[List[str]]:
    if start == goal:
        return []
    from collections import deque

    queue = deque([start])
    parents: Dict[int, int] = {start: -1}
    parent_action: Dict[int, str] = {}
    while queue:
        node = queue.popleft()
        for action, dsts in graph.edges.get(node, {}).items():
            for nxt in dsts:
                if nxt in parents:
                    continue
                parents[nxt] = node
                parent_action[nxt] = action
                if nxt == goal:
                    queue.clear()
                    break
                queue.append(nxt)
    if goal not in parents:
        return None
    actions: List[str] = []
    cur = goal
    while parents[cur] != -1:
        actions.append(parent_action[cur])
        cur = parents[cur]
    actions.reverse()
    return actions


def reachable_fractions(graph: DatasetGraph, *, sample_limit: int, rng: random.Random) -> np.ndarray:
    num_nodes = graph.centers.shape[0]
    if num_nodes == 0:
        return np.asarray([], dtype=np.float32)
    sample = min(sample_limit, num_nodes)
    starts = rng.sample(range(num_nodes), sample)
    fractions: List[float] = []
    from collections import deque

    adjacency: Dict[int, List[int]] = {}
    for src, by_action in graph.edges.items():
        targets: List[int] = []
        for dsts in by_action.values():
            targets.extend(dsts)
        adjacency[src] = targets
    for start in starts:
        visited = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nxt in adjacency.get(node, []):
                if nxt in visited:
                    continue
                visited.add(nxt)
                queue.append(nxt)
        fractions.append(len(visited) / num_nodes)
    return np.asarray(fractions, dtype=np.float32)


def delta_lattice_astar(
    start: np.ndarray,
    goal: np.ndarray,
    mu: Dict[str, np.ndarray],
    *,
    r_goal: float,
    r_merge: float,
    step_scale: float,
    max_nodes: int,
    lattice_dump: Optional[Dict[str, np.ndarray]] = None,
) -> Optional[PlanResult]:
    if start.shape != goal.shape:
        raise AssertionError("start and goal must have matching shapes.")
    if step_scale <= 0:
        raise AssertionError("step_scale must be positive.")
    import heapq

    def heuristic(p: np.ndarray) -> float:
        return float(np.linalg.norm(p - goal) / step_scale)

    open_heap: List[Tuple[float, int, int, np.ndarray]] = []
    tie = 0
    heapq.heappush(open_heap, (heuristic(start), 0, tie, start))
    parents: Dict[int, Tuple[int, str]] = {}
    nodes: List[np.ndarray] = [start]
    collect_lattice = lattice_dump is not None
    edges: List[Tuple[int, int]] = []

    def find_or_add(node: np.ndarray) -> int:
        for idx, existing in enumerate(nodes):
            if np.linalg.norm(existing - node) <= r_merge:
                return idx
        nodes.append(node)
        return len(nodes) - 1

    def _finalize_lattice() -> None:
        if not collect_lattice:
            return
        if not nodes:
            raise AssertionError("Lattice dump requires at least one node.")
        lattice_dump.clear()
        lattice_dump["nodes"] = np.stack(nodes, axis=0)
        if edges:
            lattice_dump["edges"] = np.asarray(edges, dtype=np.int32)
        else:
            lattice_dump["edges"] = np.zeros((0, 2), dtype=np.int32)

    start_idx = 0
    g_scores = {start_idx: 0}
    while open_heap and len(nodes) < max_nodes:
        _, g, _, current = heapq.heappop(open_heap)
        if np.linalg.norm(current - goal) <= r_goal:
            goal_idx = find_or_add(current)
            actions: List[str] = []
            cur = goal_idx
            while cur != start_idx:
                parent_idx, action = parents[cur]
                actions.append(action)
                cur = parent_idx
            actions.reverse()
            path = [nodes[0]]
            cur = goal_idx
            stack = [cur]
            while cur != start_idx:
                cur = parents[cur][0]
                stack.append(cur)
            for idx in reversed(stack[:-1]):
                path.append(nodes[idx])
            _finalize_lattice()
            return PlanResult(actions=actions, nodes=path)
        current_idx = find_or_add(current)
        for action in DIRECTION_ORDER:
            if action not in mu:
                continue
            nxt = current + mu[action]
            nxt_idx = find_or_add(nxt)
            if collect_lattice:
                edges.append((current_idx, nxt_idx))
            tentative = g + 1
            if tentative >= g_scores.get(nxt_idx, math.inf):
                continue
            g_scores[nxt_idx] = tentative
            parents[nxt_idx] = (current_idx, action)
            f_score = tentative + heuristic(nxt)
            tie += 1
            heapq.heappush(open_heap, (f_score, tentative, tie, nxt))
    _finalize_lattice()
    return None


def plot_action_stats(out_path: Path, deltas: np.ndarray, labels: Sequence[Optional[str]], mu: Dict[str, np.ndarray]) -> None:
    save_action_delta_stats_plot(out_path, deltas, labels, mu)


def plot_action_strip(
    out_path: Path,
    deltas: np.ndarray,
    labels: Sequence[Optional[str]],
    mu: Dict[str, np.ndarray],
    *,
    delta_label: str = "d_p",
    title_prefix: Optional[str] = None,
) -> None:
    save_action_delta_strip_plot(
        out_path,
        deltas,
        labels,
        mu,
        delta_label=delta_label,
        title_prefix=title_prefix,
    )


def plot_pca_path(
    out_path: Path,
    p_points: np.ndarray,
    plan_nodes: Optional[np.ndarray],
    start: np.ndarray,
    goal: np.ndarray,
    *,
    max_samples: int,
    grid_overlay: Optional["GridOverlay"] = None,
    title: str = "PCA(p) plan",
) -> None:
    save_planning_pca_path_plot(
        out_path,
        p_points,
        plan_nodes,
        start,
        goal,
        max_samples=max_samples,
        grid_overlay=grid_overlay,
        title=title,
    )


def plot_grid_trace(
    out_path: Path,
    grid_rows: int,
    grid_cols: int,
    visited: Sequence[Tuple[int, int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    *,
    title: str = "Execution trace",
) -> None:
    save_grid_execution_trace_plot(
        out_path,
        grid_rows,
        grid_cols,
        visited,
        start,
        goal,
        title=title,
    )


def run_plan_in_env(
    env: GridworldKeyEnv,
    actions: Sequence[str],
    *,
    start_tile: Tuple[int, int],
) -> Tuple[List[Tuple[int, int]], Optional[np.ndarray]]:
    env.reset(options={"start_tile": start_tile})
    visited = [grid_cell_from_env(env)]
    last_frame = env._render_frame()
    for action in actions:
        env_action = DIRECTION_TO_ENV_ACTION.get(action)
        if env_action is None:
            continue
        obs, _, _, _, _ = env.step(env_action)
        last_frame = obs
        visited.append(grid_cell_from_env(env))
    return visited, last_frame
