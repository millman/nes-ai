#!/usr/bin/env python3
"""
Find and optionally delete checkpoint files (.pt, .ckpt) in experiment directories.

Given one or more root directories, this script recursively scans for checkpoint files,
displays their sizes, and can delete them when --DELETE is specified.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


CHECKPOINT_EXTENSIONS = {".pt", ".ckpt"}


@dataclass(frozen=True)
class CheckpointFile:
    path: Path
    size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find checkpoint files (.pt, .ckpt) recursively and optionally delete them."
        )
    )
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path(".")],
        help="Root directories to search for checkpoint files.",
    )
    parser.add_argument(
        "--DELETE",
        action="store_true",
        dest="delete",
        default=False,
        help=(
            "Actually delete checkpoint files. "
            "By default, only report what would be removed."
        ),
    )
    return parser.parse_args()


def format_size(size: int) -> str:
    """Format byte size into human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def find_checkpoints(root: Path) -> list[CheckpointFile]:
    """Recursively find all checkpoint files in root directory."""
    checkpoints = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in CHECKPOINT_EXTENSIONS:
            try:
                size = path.stat().st_size
                checkpoints.append(CheckpointFile(path=path, size=size))
            except OSError:
                # Skip files we can't stat
                continue
    return checkpoints


def render_checkpoint_summary(
    root: Path, checkpoints: Sequence[CheckpointFile], delete_mode: bool
) -> None:
    """Display summary of checkpoint files found."""
    print(f"\nRoot: {root}")
    if not checkpoints:
        print("  (no checkpoint files found)")
        return

    # Sort by path for consistent output
    ordered = sorted(checkpoints, key=lambda c: str(c.path))

    total_size = sum(c.size for c in ordered)
    action = "DELETE" if delete_mode else "would delete"

    print(f"  Found {len(ordered)} checkpoint file(s) - Total: {format_size(total_size)}")

    last_index = len(ordered) - 1
    for idx, checkpoint in enumerate(ordered):
        branch = "\\--" if idx == last_index else "|--"
        try:
            rel_path = checkpoint.path.relative_to(root)
        except ValueError:
            rel_path = checkpoint.path
        print(f"  {branch} {rel_path} ({format_size(checkpoint.size)}) -> {action}")


def delete_checkpoint(path: Path) -> None:
    """Delete a single checkpoint file."""
    path.unlink()


def main() -> None:
    args = parse_args()
    delete_mode = args.delete

    total_files = 0
    total_size = 0

    for root in args.roots:
        root = root.expanduser().resolve()
        if not root.exists():
            print(f"\nRoot: {root} (missing)")
            continue

        checkpoints = find_checkpoints(root)
        render_checkpoint_summary(root, checkpoints, delete_mode)

        total_files += len(checkpoints)
        total_size += sum(c.size for c in checkpoints)

        if delete_mode:
            for checkpoint in checkpoints:
                delete_checkpoint(checkpoint.path)

    print(f"\n{'=' * 60}")
    print(
        f"Processed {total_files} checkpoint file(s) | "
        f"Total size: {format_size(total_size)}"
    )
    if delete_mode:
        print(f"Deleted {total_files} checkpoint file(s)")
    else:
        print("No files were deleted. Re-run with --DELETE to remove checkpoint files.")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass