#!/usr/bin/env python3
"""Step schedule parsing utilities."""
from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple, Union


def _parse_schedule(
    raw: Union[str, Sequence[Sequence[Optional[Union[int, str]]]]]
) -> Tuple[Tuple[int, Optional[int]], ...]:
    def parse_token(token: str) -> Tuple[int, Optional[int]]:
        if ":" not in token:
            raise ValueError(f"Invalid schedule entry '{token}'. Expected every_steps:max_step.")
        every_str, max_str = token.split(":", 1)
        every_steps = int(every_str)
        max_step = None if max_str.lower() == "none" else int(max_str)
        return every_steps, max_step

    if isinstance(raw, str):
        tokens = [token for token in re.split(r"[,\s]+", raw.strip()) if token]
        return tuple(parse_token(token) for token in tokens)

    schedule: List[Tuple[int, Optional[int]]] = []
    for entry in raw:
        if isinstance(entry, str):
            schedule.append(parse_token(entry))
            continue
        if len(entry) != 2:
            raise ValueError(
                f"Invalid schedule entry '{entry}'. Expected (every_steps, max_step)."
            )
        every_steps = int(entry[0])
        max_raw = entry[1]
        if isinstance(max_raw, str):
            max_step = None if max_raw.lower() == "none" else int(max_raw)
        elif max_raw is None:
            max_step = None
        else:
            max_step = int(max_raw)
        schedule.append((every_steps, max_step))
    return tuple(schedule)


def _should_run_schedule(global_step: int, schedule: Tuple[Tuple[int, Optional[int]], ...]) -> bool:
    if global_step < 0:
        return False
    for every_steps, max_step in schedule:
        if every_steps <= 0:
            continue
        if max_step is None or global_step <= max_step:
            return global_step % every_steps == 0
    return False


__all__ = ["_parse_schedule", "_should_run_schedule"]
