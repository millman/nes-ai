"""Standard plot sizing helpers for diagnostics outputs."""
from __future__ import annotations

from typing import Iterable, Iterator

DEFAULT_DPI = 200
BASE_PX = (800, 500)


def figsize_for_grid(rows: int, cols: int, base_px: tuple[int, int] = BASE_PX, dpi: int = DEFAULT_DPI) -> tuple[float, float]:
    width_px = base_px[0] * cols
    height_px = base_px[1] * rows
    return (width_px / dpi, height_px / dpi)


def _iter_axes(axes: object) -> Iterator[object]:
    if isinstance(axes, (list, tuple)):
        for item in axes:
            yield from _iter_axes(item)
        return
    try:
        iterator = axes.flat  # type: ignore[attr-defined]
    except AttributeError:
        yield axes
    else:
        for item in iterator:
            yield item


def apply_square_axes(axes: object) -> None:
    for ax in _iter_axes(axes):
        if ax is None:
            continue
        set_box = getattr(ax, "set_box_aspect", None)
        if callable(set_box):
            set_box(1)
