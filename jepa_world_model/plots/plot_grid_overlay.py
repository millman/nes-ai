"""Grid overlay helper for PCA-style plots."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class GridOverlay:
    points: np.ndarray
    positions: np.ndarray
    grid_rows: int
    grid_cols: int


def _build_index(positions: np.ndarray) -> Dict[Tuple[int, int], int]:
    index: Dict[Tuple[int, int], int] = {}
    for i, (row, col) in enumerate(positions):
        index[(int(row), int(col))] = i
    return index


def _grid_edges(
    grid_rows: int,
    grid_cols: int,
    index: Dict[Tuple[int, int], int],
) -> Iterable[Tuple[int, int]]:
    for row in range(grid_rows):
        for col in range(grid_cols - 1):
            a = index.get((row, col))
            b = index.get((row, col + 1))
            if a is not None and b is not None:
                yield a, b
    for col in range(grid_cols):
        for row in range(grid_rows - 1):
            a = index.get((row, col))
            b = index.get((row + 1, col))
            if a is not None and b is not None:
                yield a, b


def draw_grid_overlay(
    ax,
    points_2d: np.ndarray,
    positions: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    *,
    color: str = "tab:orange",
    alpha: float = 0.55,
    linewidth: float = 0.8,
    marker_size: float = 10.0,
) -> None:
    if points_2d.ndim != 2 or points_2d.shape[1] != 2:
        raise AssertionError("points_2d must be shaped (N, 2).")
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise AssertionError("positions must be shaped (N, 2).")
    if points_2d.shape[0] != positions.shape[0]:
        raise AssertionError("points_2d and positions must have matching length.")
    if grid_rows <= 0 or grid_cols <= 0:
        raise AssertionError("grid_rows/grid_cols must be positive.")

    index = _build_index(positions)
    for a, b in _grid_edges(grid_rows, grid_cols, index):
        xs = [points_2d[a, 0], points_2d[b, 0]]
        ys = [points_2d[a, 1], points_2d[b, 1]]
        ax.plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth, zorder=1)
    ax.scatter(
        points_2d[:, 0],
        points_2d[:, 1],
        s=marker_size,
        color=color,
        alpha=min(1.0, alpha + 0.2),
        label="grid",
        zorder=2,
    )


__all__ = ["GridOverlay", "draw_grid_overlay"]
