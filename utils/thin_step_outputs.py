#!/usr/bin/env python3
"""
Utility to thin out step image files (e.g., step_00010.png or ep01_step00010.png)
within a directory tree.

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
from typing import Dict, Iterable, List, Optional, Pattern, Sequence, Tuple

# Regex patterns to find png/csv files with step-like or trailing numeric identifiers.
DEFAULT_PATTERNS = [
    # Trailing numbers, e.g.:
    #   step_00010.png
    #   losses_step_00010.csv
    #   rollout_0005700.png
    r"^.*_\d+\.(?:png|csv)$",

    # Trailing numbers, with word attached.
    # step000010.png
    # ep01_step000010.png
    r"^_?.+\d+\.(?:png|csv)$",
]
DEFAULT_PATTERN_REGEX: List[Pattern[str]] = [re.compile(p) for p in DEFAULT_PATTERNS]
DEFAULT_KEEP_BEFORE = 1000
DEFAULT_KEEP_SCHEDULE_SPEC = "1000:100,10000:1000"
KeepSchedule = List[Tuple[int, int]]


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
        help="Root directories to scan (default: %(default)s).",
    )
    parser.add_argument(
        "--keep-every",
        type=int,
        default=None,
        help=(
            "Keep only steps divisible by this interval (e.g., 100 keeps step_00100) "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help=(
            "Optional offset to align keeps (step %% keep == offset). "
            "Use when numbering does not start at 0. "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--keep-schedule",
        action="append",
        default=None,
        help=(
            f"Tiered keep intervals defined as start:interval pairs. "
            f"Example: --keep-schedule \"1000:200,10000:1000\" keeps all steps below "
            f"1000, then every 200 until 10000, and every 1000 thereafter. "
            f"Cannot be combined with --keep-every. "
            f"Defaults to {DEFAULT_KEEP_SCHEDULE_SPEC} when neither keep option is given. "
            f"(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--keep-before",
        type=int,
        default=DEFAULT_KEEP_BEFORE,
        help=(
            "Never delete files with step numbers below this threshold. "
            "Use alongside --keep-every or --keep-schedule to preserve early steps. "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file output; still prints summary. "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help=(
            "Print per-file actions. By default only directory-level summaries are shown. "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--DELETE",
        dest="delete",
        action="store_true",
        default=False,
        help=(
            "Actually delete files. By default, the script only lists files it would "
            "remove. (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--summary-depth",
        type=int,
        default=2,
        help=(
            "Maximum directory depth to display in the summary tree "
            "(0 disables output, 1 shows immediate children, etc.). "
            "(default: %(default)s)."
        ),
    )
    return parser.parse_args()


def parse_keep_schedule(raw: Optional[Sequence[str]]) -> Optional[KeepSchedule]:
    if not raw:
        return None

    entries: KeepSchedule = []
    for item in raw:
        pieces = [part.strip() for part in item.split(",") if part.strip()]
        for piece in pieces:
            if ":" not in piece:
                raise SystemExit(
                    f"Invalid keep schedule entry '{piece}'. Use start:interval."
                )
            start_s, interval_s = (token.strip() for token in piece.split(":", 1))
            try:
                start = int(start_s)
                interval = int(interval_s)
            except ValueError:
                raise SystemExit(
                    f"Invalid keep schedule entry '{piece}'. start and interval must be integers."
                )
            if start < 0:
                raise SystemExit("Keep schedule start values must be non-negative.")
            if interval <= 0:
                raise SystemExit("Keep schedule intervals must be positive.")
            entries.append((start, interval))

    entries.sort(key=lambda pair: pair[0])
    seen_starts = set()
    for start, _ in entries:
        if start in seen_starts:
            raise SystemExit(
                f"Duplicate keep schedule start value {start} is not allowed."
            )
        seen_starts.add(start)
    return entries


def extract_step_number(name: str) -> Optional[int]:
    """Extract the final numeric chunk in a filename (prefer trailing before extension)."""
    match = re.search(r"(\d+)(?=\.[^.]+$)", name)
    if match:
        return int(match.group(1))
    digits = re.findall(r"\d+", name)
    if digits:
        return int(digits[-1])
    return None


def collect_candidates(
    roots: Sequence[Path], patterns: Sequence[Pattern[str]]
) -> Dict[Path, List[Path]]:
    per_dir: Dict[Path, List[Path]] = defaultdict(list)
    seen: set[Path] = set()

    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path in seen or not path.is_file():
                continue
            name = path.name
            if any(regex.search(name) for regex in patterns):
                seen.add(path)
                per_dir[path.parent].append(path)
    return per_dir


def filter_files(
    files: Iterable[Path],
    keep_every: Optional[int],
    offset: int,
    keep_before: Optional[int] = None,
    keep_schedule: Optional[KeepSchedule] = None,
) -> Tuple[List[Path], List[Path]]:
    keep: List[Path] = []
    remove: List[Path] = []

    def choose_interval(step: int) -> int:
        if keep_schedule:
            # Default to keeping everything before the first schedule start.
            interval = 1
            for start, scheduled_interval in keep_schedule:
                if step < start:
                    break
                interval = scheduled_interval
            return interval
        assert keep_every is not None
        return keep_every

    for path in files:
        step = extract_step_number(path.name)
        if step is None:
            continue
        if keep_before is not None and step < keep_before:
            keep.append(path)
            continue
        interval = choose_interval(step)
        if step % interval == offset % interval:
            keep.append(path)
        else:
            remove.append(path)
    return keep, remove


def choose_unit(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return unit
        value /= 1024
    return "PB"


def format_bytes(size: int, unit: Optional[str] = None) -> str:
    if unit is None:
        unit = choose_unit(size)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(size)
    for current in units:
        if current == unit:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}{unit}"


def summarize_tree(
    stats: Dict[Path, Tuple[int, int, int, int, int, int]], max_depth: int
) -> None:
    """Render a tree of directories with before/after counts."""

    def rel_parts(path: Path) -> Sequence[str]:
        try:
            rel = path.relative_to(Path.cwd())
        except ValueError:
            return path.parts
        return rel.parts or (".",)

    tree: Dict = {"children": {}, "stats": None, "agg": None}
    for path, counts in stats.items():
        parts = rel_parts(path)
        node = tree
        for part in parts:
            node = node.setdefault("children", {}).setdefault(
                part, {"children": {}, "stats": None, "agg": None}
            )
        node["stats"] = counts

    def aggregate(node: Dict) -> Tuple[int, int, int, int, int, int]:
        before = after = removed = kept = 0
        before_bytes = after_bytes = 0
        if node.get("stats"):
            b, a, r, k, bb, ab = node["stats"]
            before += b
            after += a
            removed += r
            kept += k
            before_bytes += bb
            after_bytes += ab
        for child in node.get("children", {}).values():
            cb, ca, cr, ck, cbb, cab = aggregate(child)
            before += cb
            after += ca
            removed += cr
            kept += ck
            before_bytes += cbb
            after_bytes += cab
        node["agg"] = (before, after, removed, kept, before_bytes, after_bytes)
        return node["agg"]

    aggregate(tree)

    def render(node: Dict, prefix: str, level: int) -> None:
        children = sorted(node.get("children", {}).items(), key=lambda item: item[0])
        for idx, (name, child) in enumerate(children):
            is_last = idx == len(children) - 1
            connector = "└─ " if is_last else "├─ "
            stats = child.get("agg")
            summary = ""
            if stats:
                before, after, removed, kept, before_bytes, after_bytes = stats
                if removed > 0:
                    pct = 100.0 if before == 0 else (after / before) * 100
                    unit = choose_unit(max(before_bytes, after_bytes))
                    summary = (
                        f" ({before} -> {after}, {pct:.1f}% remaining, "
                        f"{format_bytes(before_bytes, unit)} -> "
                        f"{format_bytes(after_bytes, unit)})"
                    )
            print(f"{prefix}{connector}{name}{summary}")
            if level + 1 < max_depth:
                render(child, prefix + ("   " if is_last else "│  "), level + 1)

    print("Directory summary (tree):")
    render(tree, "", 0)


def build_schedule_summary(
    keep_every: Optional[int],
    keep_schedule: Optional[KeepSchedule],
    keep_before: Optional[int],
    offset: int,
) -> List[str]:
    lines: List[str] = []

    if keep_every is not None:
        if keep_before is not None:
            lines.append(f"Steps < {keep_before}: keep all steps.")
        offset_value = offset % keep_every
        lines.append(
            f"Keep every {keep_every} steps (step % {keep_every} == {offset_value})."
        )
        return lines

    if not keep_schedule:
        lines.append("Keep schedule not specified; no thinning applied.")
        return lines

    if offset != 0:
        lines.append("Offset applies per interval: keep when step % interval == offset.")

    keep_all_until = 0
    schedule_first_start = keep_schedule[0][0]
    if schedule_first_start > 0:
        keep_all_until = schedule_first_start
    if keep_before is not None:
        keep_all_until = max(keep_all_until, keep_before)
    if keep_all_until > 0:
        lines.append(f"Steps < {keep_all_until}: keep all steps.")

    if keep_all_until < schedule_first_start:
        effective_schedule = list(keep_schedule)
    else:
        interval_at_keep_all = keep_schedule[0][1]
        for start, interval in keep_schedule:
            if start <= keep_all_until:
                interval_at_keep_all = interval
            else:
                break
        effective_schedule = [(keep_all_until, interval_at_keep_all)]
        for start, interval in keep_schedule:
            if start > keep_all_until:
                effective_schedule.append((start, interval))

    for idx, (start, interval) in enumerate(effective_schedule):
        end = None
        if idx + 1 < len(effective_schedule):
            end = effective_schedule[idx + 1][0] - 1
        range_label = f"Steps {start}+" if end is None else f"Steps {start}-{end}"
        if interval == 1:
            action = "keep all steps"
        else:
            action = f"keep every {interval} steps"
        lines.append(f"{range_label}: {action}.")
    return lines


def main() -> None:
    args = parse_args()
    delete_mode = args.delete
    patterns = DEFAULT_PATTERN_REGEX
    keep_schedule = parse_keep_schedule(args.keep_schedule)
    if keep_schedule and args.keep_every is not None:
        raise SystemExit("Use either --keep-every or --keep-schedule, not both.")
    if not keep_schedule and args.keep_every is None:
        keep_schedule = parse_keep_schedule([DEFAULT_KEEP_SCHEDULE_SPEC])
    if args.keep_every is not None and args.keep_every <= 0:
        raise SystemExit("--keep-every must be a positive integer")
    if args.keep_before is not None and args.keep_before < 0:
        raise SystemExit("--keep-before must be non-negative if provided")
    if args.summary_depth < 0:
        raise SystemExit("--summary-depth must be non-negative")

    candidates = collect_candidates(args.paths, patterns)
    total_dirs = len(candidates)
    total_remove = 0
    total_keep = 0
    per_dir_stats: Dict[Path, Tuple[int, int, int, int, int, int]] = {}

    def total_size(paths: Iterable[Path]) -> int:
        size = 0
        for path in paths:
            try:
                size += path.stat().st_size
            except FileNotFoundError:
                continue
        return size

    for directory, files in sorted(candidates.items(), key=lambda item: str(item[0])):
        keep, remove = filter_files(
            files, args.keep_every, args.offset, args.keep_before, keep_schedule
        )
        before_count = len(files)
        removed_count = len(remove)
        kept_count = len(keep)
        after_count = before_count - removed_count
        before_bytes = total_size(files)
        removed_bytes = total_size(remove)
        after_bytes = max(0, before_bytes - removed_bytes)
        per_dir_stats[directory] = (
            before_count,
            after_count,
            removed_count,
            kept_count,
            before_bytes,
            after_bytes,
        )
        total_keep += len(keep)
        total_remove += len(remove)
        if not remove:
            continue
        if args.verbose and not args.quiet:
            print(f"{directory}: removing {len(remove)} files, keeping {len(keep)}")
        for path in sorted(remove):
            if not delete_mode:
                if args.verbose and not args.quiet:
                    print(f"  DRY RUN: would remove {path}")
            else:
                path.unlink(missing_ok=True)
                if args.verbose and not args.quiet:
                    print(f"  removed {path}")

    schedule_lines = build_schedule_summary(
        args.keep_every, keep_schedule, args.keep_before, args.offset
    )
    print("Schedule summary:")
    for line in schedule_lines:
        print(f"  - {line}")
    print(
        f"Processed {total_dirs} directories | kept {total_keep} files | "
        f"{'removed' if delete_mode else 'would remove'} {total_remove} files"
    )
    if per_dir_stats and args.summary_depth > 0:
        summarize_tree(per_dir_stats, args.summary_depth)
    if not delete_mode:
        print("No files were deleted. Re-run with --DELETE to apply removals.")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # Allow piping to tools like `head` without noisy tracebacks.
        pass
