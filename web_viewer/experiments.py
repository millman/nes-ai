from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import csv
import tomli
import tomli_w

ALL_ZERO_EPS = 1e-12


@dataclass
class Experiment:
    """Metadata bundled for rendering experiment summaries."""

    id: str
    name: str
    path: Path
    metadata_text: str
    git_metadata_text: str
    git_commit: str
    notes_text: str
    title: str
    loss_image: Optional[Path]
    loss_csv: Optional[Path]
    rollout_steps: List[int]
    max_step: Optional[int]

    def asset_exists(self, relative: str) -> bool:
        return (self.path / relative).exists()


@dataclass
class LossCurveData:
    steps: List[float]
    series: Dict[str, List[float]]


def list_experiments(output_dir: Path) -> List[Experiment]:
    experiments: List[Experiment] = []
    if not output_dir.exists():
        return experiments
    for subdir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        exp = load_experiment(subdir)
        if exp is not None:
            experiments.append(exp)
    experiments.sort(key=lambda e: e.name, reverse=True)
    return experiments


def load_experiment(path: Path) -> Optional[Experiment]:
    if not path.is_dir():
        return None
    metadata_path = path / "metadata.txt"
    metadata_git_path = path / "metadata_git.txt"
    metrics_dir = path / "metrics"
    loss_png = metrics_dir / "loss_curves.png"
    loss_csv = _resolve_loss_csv(metrics_dir)
    notes_path = path / "notes.txt"
    metadata_custom_path = path / "experiment_metadata.txt"
    metadata_text = metadata_path.read_text() if metadata_path.exists() else "metadata.txt missing."
    git_metadata_text = metadata_git_path.read_text() if metadata_git_path.exists() else "metadata_git.txt missing."
    git_commit = _extract_git_commit(git_metadata_text)
    notes_text = _read_or_create_notes(notes_path)
    title = _read_title(metadata_custom_path)
    rollout_steps = _collect_rollout_steps(path)
    max_step = _get_max_step(loss_csv) if loss_csv and loss_csv.exists() else None
    return Experiment(
        id=path.name,
        name=path.name,
        path=path,
        metadata_text=metadata_text,
        git_metadata_text=git_metadata_text,
        git_commit=git_commit or "Unknown commit",
        notes_text=notes_text,
        title=title,
        loss_image=loss_png if loss_png.exists() else None,
        loss_csv=loss_csv if loss_csv and loss_csv.exists() else None,
        rollout_steps=rollout_steps,
        max_step=max_step,
    )


def load_loss_curves(csv_path: Path) -> Optional[LossCurveData]:
    if not csv_path.exists():
        return None
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "step" not in reader.fieldnames:
            return None
        other_fields = [f for f in reader.fieldnames if f != "step"]
        rows = list(reader)
    if not rows:
        return None
    steps: List[float] = []
    series: Dict[str, List[float]] = {field: [] for field in other_fields}
    for row in rows:
        steps.append(float(row.get("step", 0.0)))
        for field in other_fields:
            try:
                series[field].append(float(row.get(field, "0") or 0.0))
            except ValueError:
                series[field].append(0.0)
    filtered_series = {
        field: values
        for field, values in series.items()
        if not _is_all_zero(values)
    }
    if not filtered_series:
        return None
    return LossCurveData(steps=steps, series=filtered_series)


def write_notes(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = text.replace("\r\n", "\n")
    path.write_text(normalized)


def write_title(path: Path, title: str) -> None:
    clean = title.strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"title": clean or "Untitled"}
    path.write_text(tomli_w.dumps(payload))


def _read_or_create_notes(path: Path) -> str:
    if not path.exists():
        path.write_text("")
        return ""
    return path.read_text()


def _read_title(path: Path) -> str:
    if not path.exists():
        return "Untitled"
    try:
        data = tomli.loads(path.read_text())
    except (tomli.TOMLDecodeError, OSError):
        return "Untitled"
    title = data.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return "Untitled"


def _extract_git_commit(metadata_git_text: str) -> str:
    lines = metadata_git_text.splitlines()
    found_commit_block = False
    for line in lines:
        stripped = line.strip()
        if not found_commit_block:
            if stripped.lower().startswith("git commit"):
                found_commit_block = True
            continue
        if stripped:
            return stripped
    return ""


def _resolve_loss_csv(metrics_dir: Path) -> Optional[Path]:
    if not metrics_dir.exists():
        return None
    preferred = metrics_dir / "loss_curves.csv"
    if preferred.exists():
        return preferred
    legacy = metrics_dir / "loss.csv"
    if legacy.exists():
        return legacy
    return None


def _collect_rollout_steps(root: Path) -> List[int]:
    steps: set[int] = set()
    for png in root.glob("vis_fixed_*/rollout_*.png"):
        stem = png.stem
        if not stem.startswith("rollout_"):
            continue
        suffix = stem[len("rollout_") :]
        try:
            steps.add(int(suffix))
        except ValueError:
            continue
    return sorted(steps)


def _get_max_step(csv_path: Path) -> Optional[int]:
    """Get the maximum step value from the loss CSV file."""
    if not csv_path.exists():
        return None
    try:
        with csv_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "step" not in reader.fieldnames:
                return None
            max_step = None
            for row in reader:
                try:
                    step = int(float(row.get("step", 0)))
                    if max_step is None or step > max_step:
                        max_step = step
                except (ValueError, TypeError):
                    continue
            return max_step
    except (OSError, csv.Error):
        return None


def _is_all_zero(values: Iterable[float]) -> bool:
    return all(abs(v) <= ALL_ZERO_EPS for v in values)
