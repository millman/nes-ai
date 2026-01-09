#!/usr/bin/env python3
"""
Copy out.* directories from one or more roots into a destination, preserving
their relative hierarchy.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy directories matching a pattern (default: out.*) from input roots "
            "into a destination, preserving the directory hierarchy."
        )
    )
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="Root directories to scan for matching folders.",
    )
    parser.add_argument(
        "--dest",
        required=True,
        type=Path,
        help="Destination directory to copy into.",
    )
    parser.add_argument(
        "--pattern",
        default="out.*",
        help="Directory name pattern to match (default: %(default)s).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Replace existing destination folders if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print actions without copying any data.",
    )
    return parser.parse_args()


def iter_matching_dirs(root: Path, pattern: str) -> Iterable[Path]:
    if root.is_dir() and fnmatch.fnmatch(root.name, pattern):
        yield root
        return
    for current, dirnames, _ in os.walk(root):
        for name in list(dirnames):
            if fnmatch.fnmatch(name, pattern):
                yield Path(current) / name
                dirnames.remove(name)


def copy_dir(src: Path, dest: Path, overwrite: bool, dry_run: bool) -> str:
    if dest.exists():
        if not overwrite:
            return "skipped (exists)"
        if not dry_run:
            shutil.rmtree(dest)
    if not dry_run:
        shutil.copytree(src, dest)
    return "copied"


def main() -> int:
    args = parse_args()
    dest_root = args.dest
    roots = [root.resolve() for root in args.roots]

    for root in roots:
        if not root.exists():
            print(f"Missing root: {root}", file=sys.stderr)
            return 2
        if not root.is_dir():
            print(f"Root is not a directory: {root}", file=sys.stderr)
            return 2

    if not args.dry_run:
        dest_root.mkdir(parents=True, exist_ok=True)

    total_found = 0
    total_copied = 0
    total_skipped = 0
    for root in roots:
        matches = sorted(iter_matching_dirs(root, args.pattern))
        for src in matches:
            rel_path = src.relative_to(root)
            dest_path = dest_root / rel_path
            total_found += 1
            action = copy_dir(src, dest_path, args.overwrite, args.dry_run)
            if action == "copied":
                total_copied += 1
            else:
                total_skipped += 1
            print(f"{action}: {src} -> {dest_path}")

    print(
        f"Done. matched={total_found}, copied={total_copied}, "
        f"skipped={total_skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
