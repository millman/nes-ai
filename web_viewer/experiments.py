from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import csv
import re
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
    tags: str
    loss_image: Optional[Path]
    loss_csv: Optional[Path]
    rollout_steps: List[int]
    max_step: Optional[int]
    last_modified: Optional[datetime]
    total_params: Optional[int]
    flops_per_step: Optional[int]
    self_distance_csv: Optional[Path]
    self_distance_images: List[Path]
    self_distance_csvs: List[Path]
    diagnostics_images: Dict[str, List[Path]]
    diagnostics_steps: List[int]
    diagnostics_csvs: Dict[str, List[Path]]
    diagnostics_frames: Dict[int, List[Path]]

    def asset_exists(self, relative: str) -> bool:
        return (self.path / relative).exists()


@dataclass
class ExperimentIndex:
    """Lightweight index row for pagination."""

    id: str
    path: Path
    last_modified: Optional[datetime]


@dataclass
class LossCurveData:
    steps: List[float]
    cumulative_flops: List[float]
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


def build_experiment_index(output_dir: Path) -> List[ExperimentIndex]:
    """Build a lightweight listing without reading full metadata."""
    index: List[ExperimentIndex] = []
    if not output_dir.exists():
        return index
    for subdir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        last_modified = _quick_last_modified(subdir)
        index.append(ExperimentIndex(id=subdir.name, path=subdir, last_modified=last_modified))
    index.sort(key=lambda e: e.id, reverse=True)
    return index


def load_experiment(path: Path) -> Optional[Experiment]:
    if not path.is_dir():
        return None
    metadata_path = path / "metadata.txt"
    metadata_git_path = path / "metadata_git.txt"
    metadata_model_path = path / "metadata_model.txt"
    metrics_dir = path / "metrics"
    loss_png = metrics_dir / "loss_curves.png"
    loss_csv = _resolve_loss_csv(metrics_dir)
    self_distance_csvs = _collect_self_distance_csvs(path)
    latest_self_distance_csv = self_distance_csvs[-1] if self_distance_csvs else None
    self_distance_images = _collect_self_distance_images(path)
    diagnostics_images = _collect_diagnostics_images(path)
    diagnostics_steps = _collect_diagnostics_steps(diagnostics_images)
    diagnostics_csvs = _collect_diagnostics_csvs(path)
    diagnostics_frames = _collect_diagnostics_frames(path)
    notes_path = path / "notes.txt"
    metadata_custom_path = path / "experiment_metadata.txt"
    title, tags = _read_metadata(metadata_custom_path)
    metadata_text = metadata_path.read_text() if metadata_path.exists() else "metadata.txt missing."
    git_metadata_text = metadata_git_path.read_text() if metadata_git_path.exists() else "metadata_git.txt missing."
    git_commit = _extract_git_commit(git_metadata_text)
    notes_text = _read_or_create_notes(notes_path)
    rollout_steps = _collect_rollout_steps(path)
    max_step = _get_max_step(loss_csv) if loss_csv and loss_csv.exists() else None
    last_modified = _get_last_modified(path)
    total_params, flops_per_step = _read_model_metadata(metadata_model_path)
    return Experiment(
        id=path.name,
        name=path.name,
        path=path,
        metadata_text=metadata_text,
        git_metadata_text=git_metadata_text,
        git_commit=git_commit or "Unknown commit",
        notes_text=notes_text,
        title=title,
        tags=tags,
        loss_image=loss_png if loss_png.exists() else None,
        loss_csv=loss_csv if loss_csv and loss_csv.exists() else None,
        rollout_steps=rollout_steps,
        max_step=max_step,
        last_modified=last_modified,
        total_params=total_params,
        flops_per_step=flops_per_step,
        self_distance_csv=latest_self_distance_csv,
        self_distance_images=self_distance_images,
        self_distance_csvs=self_distance_csvs,
        diagnostics_images=diagnostics_images,
        diagnostics_steps=diagnostics_steps,
        diagnostics_csvs=diagnostics_csvs,
        diagnostics_frames=diagnostics_frames,
    )


def load_loss_curves(csv_path: Path) -> Optional[LossCurveData]:
    if not csv_path.exists():
        return None
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "step" not in reader.fieldnames:
            return None
        # Exclude step and cumulative_flops from series
        excluded_fields = {"step", "cumulative_flops"}
        other_fields = [f for f in reader.fieldnames if f not in excluded_fields]
        has_cumulative_flops = "cumulative_flops" in reader.fieldnames
        rows = list(reader)
    if not rows:
        return None
    steps: List[float] = []
    cumulative_flops: List[float] = []
    series: Dict[str, List[float]] = {field: [] for field in other_fields}
    for row in rows:
        steps.append(float(row.get("step", 0.0)))
        if has_cumulative_flops:
            cumulative_flops.append(float(row.get("cumulative_flops", "0") or 0.0))
        for field in other_fields:
            try:
                series[field].append(float(row.get(field, "0") or 0.0))
            except ValueError:
                series[field].append(0.0)
    # If no cumulative_flops column, generate placeholder (step-based)
    if not cumulative_flops:
        cumulative_flops = [s for s in steps]
    filtered_series = {
        field: values
        for field, values in series.items()
        if not _is_all_zero(values)
    }
    if not filtered_series:
        return None
    return LossCurveData(steps=steps, cumulative_flops=cumulative_flops, series=filtered_series)


def write_notes(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = text.replace("\r\n", "\n")
    path.write_text(normalized)


def write_title(path: Path, title: str) -> None:
    _write_metadata(path, title=title)


def write_tags(path: Path, tags: str) -> None:
    _write_metadata(path, tags=tags)


def _read_or_create_notes(path: Path) -> str:
    if not path.exists():
        path.write_text("")
        return ""
    return path.read_text()


def _read_title(path: Path) -> str:
    title, _ = _read_metadata(path)
    return title


def _read_metadata(path: Path) -> tuple[str, str]:
    """Read custom metadata (title, tags) with sane defaults."""
    if not path.exists():
        return "Untitled", ""
    try:
        data = tomli.loads(path.read_text())
    except (tomli.TOMLDecodeError, OSError):
        return "Untitled", ""
    raw_title = data.get("title")
    raw_tags = data.get("tags")
    title = raw_title.strip() if isinstance(raw_title, str) and raw_title.strip() else "Untitled"
    tags = _normalize_tags(raw_tags)
    return title, tags


def _normalize_tags(raw_tags) -> str:
    """Normalize tags from string or list to a single string."""
    if isinstance(raw_tags, str):
        return raw_tags.strip()
    if isinstance(raw_tags, list):
        cleaned = []
        for item in raw_tags:
            if isinstance(item, str) and item.strip():
                cleaned.append(item.strip())
        return ", ".join(cleaned)
    return ""


def _write_metadata(path: Path, title: Optional[str] = None, tags: Optional[str] = None) -> None:
    """Write combined title/tags metadata, preserving existing values."""
    current_title, current_tags = _read_metadata(path)
    next_title = title.strip() if isinstance(title, str) else None
    next_tags = tags.strip() if isinstance(tags, str) else None
    payload = {
        "title": (next_title if next_title is not None and next_title else current_title or "Untitled"),
        "tags": (next_tags if next_tags is not None else current_tags),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(payload))


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


def _read_model_metadata(path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Read total_params and flops_per_step from metadata_model.txt."""
    if not path.exists():
        return None, None
    try:
        data = tomli.loads(path.read_text())
    except (tomli.TOMLDecodeError, OSError):
        return None, None
    total_params = None
    flops_per_step = None
    params_section = data.get("parameters")
    if isinstance(params_section, dict):
        total = params_section.get("total")
        if isinstance(total, int):
            total_params = total
    flops_section = data.get("flops")
    if isinstance(flops_section, dict):
        per_step = flops_section.get("per_step")
        if isinstance(per_step, int):
            flops_per_step = per_step
    return total_params, flops_per_step


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


def _collect_self_distance_images(root: Path) -> List[Path]:
    folder = root / "vis_self_distance"
    if not folder.exists():
        return []
    return sorted(folder.glob("self_distance_*.png"))


def _collect_self_distance_csvs(root: Path) -> List[Path]:
    folder = root / "self_distance"
    if not folder.exists():
        return []
    return sorted(folder.glob("self_distance_*.csv"))


def _collect_diagnostics_images(root: Path) -> Dict[str, List[Path]]:
    diag_specs = [
        ("delta_z_pca", root / "vis_delta_z_pca", "delta_z_pca_*.png"),
        ("variance_spectrum", root / "vis_delta_z_pca", "delta_z_variance_spectrum_*.png"),
        ("action_alignment", root / "vis_action_alignment", "action_alignment_[0-9]*.png"),
        ("action_alignment_detail", root / "vis_action_alignment", "action_alignment_detail_*.png"),
        ("cycle_error", root / "vis_cycle_error", "*.png"),
    ]
    images: Dict[str, List[Path]] = {}
    for name, folder, pattern in diag_specs:
        if folder.exists():
            imgs = sorted(folder.glob(pattern))
            if imgs:
                images[name] = imgs
    return images


def _collect_diagnostics_steps(diagnostics_images: Dict[str, List[Path]]) -> List[int]:
    steps: set[int] = set()
    for paths in diagnostics_images.values():
        for path in paths:
            stem = path.stem
            parts = stem.split("_")
            if not parts:
                continue
            suffix = parts[-1]
            try:
                steps.add(int(suffix))
            except ValueError:
                continue
    return sorted(steps)


def _collect_diagnostics_csvs(root: Path) -> Dict[str, List[Path]]:
    diag_dirs = {
        "delta_z_pca": root / "vis_delta_z_pca",
        "action_alignment": root / "vis_action_alignment",
        "cycle_error": root / "vis_cycle_error",
    }
    csvs: Dict[str, List[Path]] = {}
    for name, folder in diag_dirs.items():
        if folder.exists():
            files = sorted(folder.glob("*.csv")) + sorted(folder.glob("*.txt"))
            if files:
                csvs[name] = files
    return csvs


def _collect_diagnostics_frames(root: Path) -> Dict[int, List[Path]]:
    frames_root = root / "vis_diagnostics_frames"
    if not frames_root.exists():
        return {}
    frame_map: Dict[int, List[Path]] = {}

    def _parse_step(src: str) -> int:
        # Prefer numeric token at end of filename (e.g., state_123.png -> 123).
        match = re.search(r"(\d+)(?:\\.[A-Za-z0-9]+)?$", src)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return 10**9  # fallback large value to push unknowns to end

    for csv_path in sorted(frames_root.glob("frames_*.csv")):
        stem = csv_path.stem
        suffix = stem.split("_")[-1]
        try:
            step = int(suffix)
        except ValueError:
            continue
        try:
            with csv_path.open("r", newline="") as handle:
                reader = csv.DictReader(handle)
                sorted_entries: List[Tuple[Tuple[str, int, int], Path]] = []
                for idx, row in enumerate(reader):
                    rel = row.get("image_path")
                    src = row.get("source_path", "")
                    if not rel:
                        continue
                    prefix = str(Path(src).parent)
                    sort_key = (prefix, _parse_step(src), idx)
                    sorted_entries.append((sort_key, frames_root / rel))
                if sorted_entries:
                    sorted_entries.sort(key=lambda t: t[0])
                    frame_map[step] = [p for _, p in sorted_entries]
        except (OSError, csv.Error):
            continue
    return frame_map


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


def _get_last_modified(path: Path) -> Optional[datetime]:
    """Get the most recent modification time of any file in the directory tree."""
    if not path.is_dir():
        return None
    latest_mtime: Optional[float] = None
    try:
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    mtime = item.stat().st_mtime
                    if latest_mtime is None or mtime > latest_mtime:
                        latest_mtime = mtime
                except OSError:
                    continue
    except OSError:
        return None
    if latest_mtime is None:
        return None
    return datetime.fromtimestamp(latest_mtime)


def _quick_last_modified(path: Path) -> Optional[datetime]:
    """Fast last-modified using directory mtime."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return datetime.fromtimestamp(stat.st_mtime)


def _is_all_zero(values: Iterable[float]) -> bool:
    return all(abs(v) <= ALL_ZERO_EPS for v in values)
