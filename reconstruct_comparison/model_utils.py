from __future__ import annotations

from typing import Optional


def _norm_groups(channels: int) -> int:
    return 8 if channels % 8 == 0 else 1


def default(val: Optional[int], fallback: int) -> int:
    return fallback if val is None else val


def _group_count(channels: int, max_groups: int) -> int:
    """Largest group count ≤ max_groups that evenly divides channels."""
    limit = min(max_groups, channels)
    for groups in range(limit, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


__all__ = ["_norm_groups", "_group_count", "default"]
