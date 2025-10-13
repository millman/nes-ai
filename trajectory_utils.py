"""Filesystem helpers for natural trajectory and frame ordering."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence


_NUMERIC_PATTERN = re.compile(r'(\d+)')


def _natural_key(path: Path) -> tuple:
    """Case-insensitive natural sort key treating embedded digits numerically."""
    parts = _NUMERIC_PATTERN.split(path.name)
    key = []
    for part in parts:
        if not part:
            continue
        key.append(int(part) if part.isdigit() else part.lower())
    return tuple(key)


def list_traj_dirs(root: Path) -> List[Path]:
    """Return trajectory subdirectories sorted by natural filename order."""
    return sorted((p for p in root.iterdir() if p.is_dir()), key=_natural_key)


def list_state_frames(
    states_dir: Path,
    suffixes: Sequence[str] = (".png", ".jpg", ".jpeg"),
) -> List[Path]:
    """Return state frame image paths sorted by natural filename order."""
    suffix_set = {s.lower() for s in suffixes}
    return sorted(
        (p for p in states_dir.iterdir() if p.suffix.lower() in suffix_set),
        key=_natural_key,
    )
