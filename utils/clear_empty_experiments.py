#!/usr/bin/env python3
"""
Summarize experiment step counts and optionally delete empty experiments.

Given one or more root directories (e.g., out.jepa_world_model_trainer),
this scans each immediate experiment subdirectory for metrics/loss.csv and
for any files with trailing numeric identifiers (e.g., state_embedding_hist_0000040.png).
Experiments with 0 or 1 total steps are eligible for removal when --delete is set.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from web_viewer.csv_utils import get_max_step

STEP_NUMBER_REGEX = re.compile(r"(\d+)(?=\.[^.]+$)")
DEFAULT_MAX_DELETE_STEPS = 1


@dataclass(frozen=True)
class ExperimentSummary:
    path: Path
    csv_steps: int
    file_steps: int
    max_steps: int
    max_step_source: Optional[Path]
    remove_candidate: bool
    loss_csv: Optional[Path]
    error: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report experiment step counts from metrics/loss.csv and filenames with "
            "trailing numbers, optionally deleting empty experiments."
        )
    )
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path(".")],
        help="Root directories that contain experiment subdirectories.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        default=False,
        help=(
            "Actually delete experiments at or below the configured step threshold. "
            "By default, only report what would be removed."
        ),
    )
    parser.add_argument(
        "--max-delete-steps",
        type=int,
        default=DEFAULT_MAX_DELETE_STEPS,
        help=(
            "Delete experiments with steps <= this threshold "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Show all experiments, not just deletion candidates.",
    )
    return parser.parse_args()


def extract_step_number(name: str) -> Optional[int]:
    match = STEP_NUMBER_REGEX.search(name)
    if match:
        return int(match.group(1))
    digits = re.findall(r"\d+", name)
    if digits:
        return int(digits[-1])
    return None


def scan_file_steps(
    root: Path, stop_above: Optional[int]
) -> tuple[Optional[int], Optional[Path], bool]:
    max_step: Optional[int] = None
    max_path: Optional[Path] = None
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if "self_distance_frames" in path.parts:
            continue
        step = extract_step_number(path.name)
        if step is None:
            continue
        if max_step is None or step > max_step:
            max_step = step
            max_path = path
        if stop_above is not None and step > stop_above:
            return max_step, max_path, True
    return max_step, max_path, False


def summarize_experiment(
    exp_dir: Path, max_delete_steps: int, verbose: bool
) -> ExperimentSummary:
    loss_csv = exp_dir / "metrics" / "loss.csv"
    error: Optional[str] = None
    csv_max: Optional[int] = None
    if loss_csv.exists():
        try:
            csv_max = get_max_step(loss_csv)
        except Exception as exc:
            error = f"loss.csv error: {exc}"
    csv_steps = csv_max + 1 if csv_max is not None else 0
    if error:
        file_steps = 0
        max_steps = csv_steps
        max_step_source = loss_csv if csv_steps > 0 else None
        remove_candidate = False
    elif not verbose and csv_steps > max_delete_steps:
        file_steps = 0
        max_steps = csv_steps
        max_step_source = loss_csv
        remove_candidate = False
    else:
        stop_above = None if verbose else max_delete_steps
        file_max, file_path, early_exit = scan_file_steps(exp_dir, stop_above)
        file_steps = file_max + 1 if file_max is not None else 0
        max_steps = max(csv_steps, file_steps)
        if max_steps == 0:
            max_step_source = None
        elif csv_steps >= file_steps and csv_steps > 0:
            max_step_source = loss_csv
        else:
            max_step_source = file_path
        remove_candidate = max_steps <= max_delete_steps
        if early_exit:
            remove_candidate = False
    return ExperimentSummary(
        path=exp_dir,
        csv_steps=csv_steps,
        file_steps=file_steps,
        max_steps=max_steps,
        max_step_source=max_step_source,
        remove_candidate=remove_candidate,
        loss_csv=loss_csv if loss_csv.exists() else None,
        error=error,
    )


def format_summary(summary: ExperimentSummary, delete_mode: bool) -> str:
    action = "keep"
    if summary.remove_candidate:
        action = "delete" if delete_mode else "would delete"
    csv_note = f"csv={summary.csv_steps}"
    if summary.loss_csv is None:
        csv_note = "csv=missing"
    source = "source=missing"
    if summary.max_step_source is not None:
        try:
            source_path = summary.max_step_source.relative_to(summary.path)
        except ValueError:
            source_path = summary.max_step_source
        source = f"source={source_path}"
    return (
        f"{summary.path.name} (steps={summary.max_steps}, {csv_note}, "
        f"files={summary.file_steps}, {source}) -> {action}"
    )


def _is_experiment_dir(path: Path) -> bool:
    return (path / "metrics").is_dir() and (path / "metadata.txt").is_file()


def iter_experiment_dirs(root: Path) -> Iterable[Path]:
    if _is_experiment_dir(root):
        yield root
        return
    seen: set[Path] = set()
    for metadata_file in root.rglob("metadata.txt"):
        exp_dir = metadata_file.parent
        if exp_dir in seen:
            continue
        if not _is_experiment_dir(exp_dir):
            continue
        seen.add(exp_dir)
        yield exp_dir


def render_root_summary(
    root: Path, summaries: Sequence[ExperimentSummary], delete_mode: bool
) -> None:
    print(f"Root: {root}")
    if not summaries:
        print("  (no matching experiments)")
        return
    last_index = len(summaries) - 1
    for idx, summary in enumerate(summaries):
        branch = "\\--" if idx == last_index else "|--"
        print(f"  {branch} {format_summary(summary, delete_mode)}")


def remove_experiment(path: Path) -> None:
    shutil.rmtree(path)


def main() -> None:
    args = parse_args()
    delete_mode = args.delete
    if args.max_delete_steps < 0:
        raise SystemExit("--max-delete-steps must be non-negative")
    total_experiments = 0
    total_remove = 0

    for root in args.roots:
        root = root.expanduser().resolve()
        if not root.exists():
            print(f"Root: {root} (missing)")
            continue
        exp_dirs = list(iter_experiment_dirs(root))
        all_summaries = [
            summarize_experiment(exp_dir, args.max_delete_steps, args.verbose)
            for exp_dir in exp_dirs
        ]
        summaries = all_summaries
        if not args.verbose:
            summaries = [summary for summary in summaries if summary.remove_candidate]
        render_root_summary(root, summaries, delete_mode)
        total_experiments += len(all_summaries)
        total_remove += sum(summary.remove_candidate for summary in all_summaries)
        for summary in all_summaries:
            if summary.error:
                print(f"Warning: {summary.path} skipped for deletion ({summary.error})")

        if delete_mode:
            for summary in all_summaries:
                if summary.remove_candidate:
                    remove_experiment(summary.path)

    print(
        f"Processed {total_experiments} experiments | "
        f"{'removed' if delete_mode else 'would remove'} {total_remove}"
    )
    if not delete_mode:
        print("No experiments were deleted. Re-run with --delete to apply removals.")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass
