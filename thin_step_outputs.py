#!/usr/bin/env python3
"""
Utility to thin out step image files (e.g., step_00010.png) within a directory tree.

Given a root directory and a keep interval, the script walks every subdirectory,
identifies files that match the pattern `step_<number>.<ext>`, and removes files whose
step number is not aligned with the requested interval. Use --dry-run to preview the
deletions before actually removing anything.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

STEP_REGEX = re.compile(r"step_(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove intermediate step_* files so only every Nth step remains "
            "inside each directory."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path(".")],
        help="Root directories to scan (defaults to current directory).",
    )
    parser.add_argument(
        "--keep-every",
        type=int,
        required=True,
        help="Keep only steps divisible by this interval (e.g., 100 keeps step_00100).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help=(
            "Optional offset to align keeps (step %% keep == offset). "
            "Use when numbering does not start at 0."
        ),
    )
    parser.add_argument(
        "--glob",
        action="append",
        default=["step_*.png"],
        help=(
            "Glob pattern to match files (relative to each path). "
            "May be supplied multiple times. Defaults to step_*.png."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be removed without deleting anything.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file output; still prints summary.",
    )
    return parser.parse_args()


def collect_candidates(
    roots: Sequence[Path], patterns: Sequence[str]
) -> Dict[Path, List[Path]]:
    per_dir: Dict[Path, List[Path]] = defaultdict(list)
    seen: set[Path] = set()

    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.rglob(pattern):
                if path in seen or not path.is_file():
                    continue
                seen.add(path)
                per_dir[path.parent].append(path)
    return per_dir


def filter_files(
    files: Iterable[Path], keep_every: int, offset: int
) -> Tuple[List[Path], List[Path]]:
    keep: List[Path] = []
    remove: List[Path] = []
    for path in files:
        match = STEP_REGEX.search(path.name)
        if not match:
            continue
        step = int(match.group(1))
        if step % keep_every == offset % keep_every:
            keep.append(path)
        else:
            remove.append(path)
    return keep, remove


def main() -> None:
    args = parse_args()
    if args.keep_every <= 0:
        raise SystemExit("--keep-every must be a positive integer")

    candidates = collect_candidates(args.paths, args.glob)
    total_dirs = len(candidates)
    total_remove = 0
    total_keep = 0

    for directory, files in sorted(candidates.items(), key=lambda item: str(item[0])):
        keep, remove = filter_files(files, args.keep_every, args.offset)
        total_keep += len(keep)
        total_remove += len(remove)
        if not remove:
            continue
        if not args.quiet:
            print(f"{directory}: removing {len(remove)} files, keeping {len(keep)}")
        for path in remove:
            if args.dry_run:
                if not args.quiet:
                    print(f"  DRY RUN: would remove {path}")
            else:
                path.unlink(missing_ok=True)
                if not args.quiet:
                    print(f"  removed {path}")

    print(
        f"Processed {total_dirs} directories | kept {total_keep} files | "
        f"{'would remove' if args.dry_run else 'removed'} {total_remove} files"
    )


if __name__ == "__main__":
    main()
