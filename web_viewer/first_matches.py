from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def _first_matching_file(folder: Path, *, exact_name: Optional[str], pattern: str) -> Optional[Path]:
    if exact_name:
        exact_path = folder / exact_name
        if exact_path.exists():
            return exact_path
    try:
        with os.scandir(folder) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                if fnmatch.fnmatch(entry.name, pattern):
                    return Path(entry.path)
    except OSError:
        return None
    return None


def _first_existing_matches(
    root: Path,
    candidates: Sequence[Tuple[str, Sequence[str]]],
    *,
    conflict_label: str = "candidates",
) -> List[Path]:
    hits: List[List[Path]] = []
    for folder_name, patterns in candidates:
        folder = root / folder_name
        if not folder.exists():
            continue
        files: List[Path] = []
        for pattern in patterns:
            files.extend(sorted(folder.glob(pattern)))
        if files:
            hits.append(files)
            break
    if len(hits) > 1:
        raise ValueError(f"Multiple {conflict_label} contain files in {root}.")
    return hits[0] if hits else []


def _first_matching_csv_candidate(
    root: Path,
    candidates: Sequence[Tuple[str, str, str, str]],
    *,
    conflict_label: str,
) -> Optional[Path]:
    matches: List[Tuple[str, Path]] = []
    for label, folder_name, exact_name, pattern in candidates:
        folder = root / folder_name
        if not folder.exists():
            continue
        match = _first_matching_file(folder, exact_name=exact_name, pattern=pattern)
        if match is not None:
            matches.append((label, match))
    if len(matches) > 1:
        labels = ", ".join(label for label, _ in matches)
        raise ValueError(f"Multiple {conflict_label} candidates contain CSVs in {root}: {labels}.")
    return matches[0][1] if matches else None


def _first_existing_steps(
    root: Path,
    candidates: Sequence[Tuple[str, str, str]],
) -> List[int]:
    for folder_name, pattern, prefix in candidates:
        folder = root / folder_name
        if not folder.exists():
            continue
        steps: set[int] = set()
        for png in folder.glob(pattern):
            stem = png.stem
            if prefix and stem.startswith(prefix):
                suffix = stem[len(prefix) :]
            else:
                parts = stem.split("_")
                suffix = parts[-1] if parts else stem
            try:
                steps.add(int(suffix))
            except ValueError:
                continue
        if steps:
            return sorted(steps)
    return []
