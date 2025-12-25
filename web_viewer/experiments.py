from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import csv
import re
import tomli
import tomli_w

ALL_ZERO_EPS = 1e-12
PROFILE_ENABLED = os.environ.get("VIEWER_PROFILE", "").lower() in {"1", "true", "yes", "on"}
PROFILE_LOGGER = logging.getLogger("web_viewer.profile")
PROFILE_LOGGER.setLevel(logging.INFO)


def _profile(label: str, start_time: float, path: Optional[Path] = None, **fields) -> None:
    """Log lightweight profiling info when VIEWER_PROFILE is enabled."""
    if not PROFILE_ENABLED:
        return
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    parts = [f"{label} {elapsed_ms:.1f}ms"]
    if path is not None:
        parts.append(f"path={path}")
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    PROFILE_LOGGER.info(" ".join(parts))


_MODEL_DIFF_SHORTNAMES: Dict[str, str] = {
    "loss_normalization_enabled": "norm",
    "image_size": "img",
}


def _parse_model_diff_items(text: str) -> List[Tuple[str, str, bool]]:
    """Return display, full text, shortened? tuples for model diff entries."""
    stripped = text.strip()
    if not stripped or stripped.startswith("model_diff.txt missing"):
        return []
    items: List[Tuple[str, str, bool]] = []
    for line in text.splitlines():
        trimmed = line.strip()
        if not trimmed:
            continue
        normalized = trimmed.lstrip("+- ").lower()
        if normalized.startswith("data_root"):
            continue
        display = trimmed
        shortened = False
        sep = None
        key = None
        for candidate in ("=", ":"):
            if candidate in trimmed:
                key, rest = trimmed.split(candidate, 1)
                sep = candidate
                break
        if key:
            key_clean = key.strip()
            short = _MODEL_DIFF_SHORTNAMES.get(key_clean)
            if short and sep is not None:
                display = f"{short}{sep}{rest.strip()}"
                shortened = True
        items.append((display, trimmed, shortened))
    return items


def _render_model_diff(text: str) -> str:
    items = _parse_model_diff_items(text)
    if not text.strip():
        return "—"
    if text.strip().startswith("model_diff.txt missing"):
        return text.strip()
    if not items:
        return "—"
    return ", ".join(display for display, _, _ in items)


@dataclass
class Experiment:
    """Metadata bundled for rendering experiment summaries."""

    id: str
    name: str
    path: Path
    metadata_text: str
    data_root: Optional[str]
    model_diff_text: str
    model_diff_items: List[Tuple[str, str, bool]]
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
    diagnostics_frames: Dict[int, List[Tuple[Path, str, str, Optional[int]]]]

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
    elapsed_seconds: List[float]
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


def load_experiment(
    path: Path,
    include_self_distance: bool = True,
    include_diagnostics_images: bool = True,
    include_diagnostics_frames: bool = True,
    include_last_modified: bool = True,
) -> Optional[Experiment]:
    if not path.is_dir():
        return None
    start_time = time.perf_counter()
    section_start = start_time
    metadata_path = path / "metadata.txt"
    metadata_git_path = path / "metadata_git.txt"
    metadata_model_diff_path = path / "server_cache" / "model_diff.txt"
    metadata_model_path = path / "metadata_model.txt"
    metrics_dir = path / "metrics"
    loss_png = metrics_dir / "loss_curves.png"
    loss_csv = _resolve_loss_csv(metrics_dir)
    _profile("load_experiment.paths", start_time, path, metrics_dir=metrics_dir)
    section_start = time.perf_counter()

    self_distance_csvs: List[Path] = []
    latest_self_distance_csv: Optional[Path] = None
    self_distance_images: List[Path] = []
    if include_self_distance:
        self_distance_csvs = _collect_self_distance_csvs(path)
        latest_self_distance_csv = self_distance_csvs[-1] if self_distance_csvs else None
        self_distance_images = _collect_self_distance_images(path)
    else:
        latest_self_distance_csv = _quick_self_distance_csv(path)
    _profile(
        "load_experiment.self_distance",
        section_start,
        path,
        csvs=len(self_distance_csvs),
        images=len(self_distance_images),
        included=include_self_distance,
    )
    section_start = time.perf_counter()

    diagnostics_images: Dict[str, List[Path]] = {}
    diagnostics_steps: List[int] = []
    diagnostics_csvs: Dict[str, List[Path]] = {}
    diagnostics_frames: Dict[int, List[Tuple[Path, str]]] = {}
    if include_diagnostics_images:
        diagnostics_images = _collect_diagnostics_images(path)
        diagnostics_steps = _collect_diagnostics_steps(diagnostics_images)
        diagnostics_csvs = _collect_diagnostics_csvs(path)
    else:
        diagnostics_steps = [0] if _diagnostics_exists(path) else []
    diagnostics_frames = _collect_diagnostics_frames(path) if include_diagnostics_frames else {}
    _profile(
        "load_experiment.diagnostics",
        section_start,
        path,
        images=sum(len(v) for v in diagnostics_images.values()),
        steps=len(diagnostics_steps),
        csvs=sum(len(v) for v in diagnostics_csvs.values()),
        frames=sum(len(v) for v in diagnostics_frames.values()),
        include_frames=include_diagnostics_frames,
        include_images=include_diagnostics_images,
    )
    section_start = time.perf_counter()

    notes_path = path / "notes.txt"
    metadata_custom_path = path / "experiment_metadata.txt"
    title, tags = _read_metadata(metadata_custom_path)
    metadata_text = metadata_path.read_text() if metadata_path.exists() else "metadata.txt missing."
    metadata_data_root = _extract_data_root_from_metadata(metadata_text) if metadata_text else None
    if metadata_model_diff_path.exists():
        metadata_model_diff_text_raw = metadata_model_diff_path.read_text()
    else:
        metadata_model_diff_text_raw = _ensure_model_diff(path)
    metadata_model_diff_items = _parse_model_diff_items(metadata_model_diff_text_raw)
    metadata_model_diff_text = _render_model_diff(metadata_model_diff_text_raw)
    git_metadata_text = metadata_git_path.read_text() if metadata_git_path.exists() else "metadata_git.txt missing."
    git_commit = _extract_git_commit(git_metadata_text)
    notes_text = _read_or_create_notes(notes_path)
    _profile("load_experiment.text_meta", section_start, path)
    section_start = time.perf_counter()

    rollout_steps = _collect_rollout_steps(path)
    max_step = _get_max_step(loss_csv) if loss_csv and loss_csv.exists() else None
    last_modified = _get_last_modified(path) if include_last_modified else _quick_last_modified(path)
    total_params, flops_per_step = _read_model_metadata(metadata_model_path)
    _profile(
        "load_experiment.metrics",
        section_start,
        path,
        rollout_steps=len(rollout_steps),
        max_step=max_step if max_step is not None else "",
        loss_csv=bool(loss_csv and loss_csv.exists()),
        last_modified_mode="deep" if include_last_modified else "quick",
    )
    _profile(
        "load_experiment",
        start_time,
        path,
        diagnostics_images=sum(len(v) for v in diagnostics_images.values()),
        diagnostics_frames=sum(len(v) for v in diagnostics_frames.values()),
        diagnostics_csvs=sum(len(v) for v in diagnostics_csvs.values()),
        include_frames=include_diagnostics_frames,
    )
    return Experiment(
        id=path.name,
        name=path.name,
        path=path,
        metadata_text=metadata_text,
        data_root=metadata_data_root,
        model_diff_text=metadata_model_diff_text,
        model_diff_items=metadata_model_diff_items,
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
        # Exclude step, cumulative_flops, and elapsed_seconds from series
        excluded_fields = {"step", "cumulative_flops", "elapsed_seconds"}
        other_fields = [f for f in reader.fieldnames if f not in excluded_fields]
        has_cumulative_flops = "cumulative_flops" in reader.fieldnames
        has_elapsed_seconds = "elapsed_seconds" in reader.fieldnames
        rows = list(reader)
    if not rows:
        return None
    steps: List[float] = []
    cumulative_flops: List[float] = []
    elapsed_seconds: List[float] = []
    series: Dict[str, List[float]] = {field: [] for field in other_fields}
    for row in rows:
        steps.append(float(row.get("step", 0.0)))
        if has_cumulative_flops:
            cumulative_flops.append(float(row.get("cumulative_flops", "0") or 0.0))
        if has_elapsed_seconds:
            elapsed_seconds.append(float(row.get("elapsed_seconds", "0") or 0.0))
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
    return LossCurveData(
        steps=steps,
        cumulative_flops=cumulative_flops,
        elapsed_seconds=elapsed_seconds,
        series=filtered_series,
    )


def _step_from_filename(path: Path) -> Optional[int]:
    """Extract trailing integer step from a filename like foo_0000100.txt."""
    stem = path.stem
    suffix = stem.split("_")[-1] if "_" in stem else stem
    try:
        return int(suffix)
    except ValueError:
        return None


def _parse_tabbed_lines(text: str) -> List[Dict[str, str]]:
    """Parse tab-delimited diagnostics text into dictionaries."""
    rows: List[Dict[str, str]] = []
    headers: Optional[List[str]] = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("Action alignment") or line.startswith("Per-action"):
            continue
        if line.lower().startswith("no actions"):
            break
        if line.startswith("action_id"):
            headers = line.split("\t")
            continue
        if headers is None:
            continue
        parts = line.split("\t")
        if len(parts) < len(headers):
            parts.extend([""] * (len(headers) - len(parts)))
        rows.append({key: value for key, value in zip(headers, parts)})
    return rows


def _to_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_action_alignment_report(path: Path) -> List[Dict[str, object]]:
    """Return parsed rows from action_alignment_report_*.txt."""
    try:
        content = path.read_text()
    except OSError:
        return []
    rows = _parse_tabbed_lines(content)
    parsed: List[Dict[str, object]] = []
    for row in rows:
        aid = _to_int(row.get("action_id", ""))
        if aid is None:
            continue
        parsed.append(
            {
                "action_id": aid,
                "label": row.get("label", ""),
                "count": _to_int(row.get("count", "")) or 0,
                "mean": _to_float(row.get("mean", "")),
                "median": _to_float(row.get("median", "")),
                "pct_high": _to_float(row.get("pct_gt_thr", "")),
                "std": _to_float(row.get("std", "")),
                "frac_neg": _to_float(row.get("frac_neg", "")),
                "delta_norm_median": _to_float(row.get("delta_norm_median", "")),
                "delta_norm_p90": _to_float(row.get("delta_norm_p90", "")),
                "inverse_alignment": _to_float(row.get("inverse_alignment", "")),
                "note": row.get("notes", "") or "",
            }
        )
    return parsed


def _parse_action_alignment_strength(path: Path) -> List[Dict[str, object]]:
    """Return parsed rows from action_alignment_strength_*.txt."""
    try:
        content = path.read_text()
    except OSError:
        return []
    rows = _parse_tabbed_lines(content)
    parsed: List[Dict[str, object]] = []
    for row in rows:
        aid = _to_int(row.get("action_id", ""))
        if aid is None:
            continue
        parsed.append(
            {
                "action_id": aid,
                "label": row.get("label", ""),
                "count": _to_int(row.get("count", "")) or 0,
                "mean": _to_float(row.get("mean_cos", "")),
                "std": _to_float(row.get("std", "")),
                "frac_neg": _to_float(row.get("frac_neg", "")),
                "mean_dir_norm": _to_float(row.get("mean_dir_norm", "")),
                "delta_norm_median": _to_float(row.get("delta_norm_median", "")),
                "strength_ratio": _to_float(row.get("strength_ratio", "")),
                "note": row.get("note", "") or "",
            }
        )
    return parsed


def extract_alignment_summary(experiment: Experiment) -> Optional[Dict[str, object]]:
    """Build a merged per-action summary from the latest alignment report/strength files."""
    files = experiment.diagnostics_csvs.get("action_alignment", [])
    if not files:
        return None

    report_files = [p for p in files if "action_alignment_report_" in p.name]
    strength_files = [p for p in files if "action_alignment_strength_" in p.name]
    if not report_files:
        return None

    def _latest(paths: List[Path]) -> Optional[Path]:
        return max(paths, key=lambda p: _step_from_filename(p) or -1, default=None)

    latest_report = _latest(report_files)
    if latest_report is None:
        return None
    report_step = _step_from_filename(latest_report)

    strength_for_step = None
    if report_step is not None:
        for candidate in strength_files:
            if _step_from_filename(candidate) == report_step:
                strength_for_step = candidate
                break
    if strength_for_step is None and strength_files:
        strength_for_step = _latest(strength_files)

    report_rows = _parse_action_alignment_report(latest_report)
    strength_rows = _parse_action_alignment_strength(strength_for_step) if strength_for_step else []
    if not report_rows:
        return None

    strength_by_id = {row["action_id"]: row for row in strength_rows if "action_id" in row}
    merged_rows: List[Dict[str, object]] = []
    for row in report_rows:
        aid = row.get("action_id")
        merged = dict(row)
        strength_row = strength_by_id.get(aid)
        if strength_row:
            merged["strength_ratio"] = strength_row.get("strength_ratio")
            if strength_row.get("note"):
                merged["strength_note"] = strength_row["note"]
        merged_rows.append(merged)

    merged_rows.sort(key=lambda r: r.get("count", 0), reverse=True)
    return {
        "step": report_step,
        "rows": merged_rows,
        "report_path": latest_report,
        "strength_path": strength_for_step,
    }


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


def _extract_data_root_from_metadata(metadata_text: str) -> Optional[str]:
    """Return the first data_root value found within a TOML metadata blob."""
    try:
        parsed = tomli.loads(metadata_text)
    except (tomli.TOMLDecodeError, OSError):
        return None

    def _walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "data_root" and isinstance(value, str) and value.strip():
                    return value.strip()
                found = _walk(value)
                if found:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = _walk(item)
                if found:
                    return found
        return None

    return _walk(parsed)


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

VIS_STEP_SPECS = [
    ("vis_fixed_0", "vis_fixed_0", "rollout_*.png", "rollout_"),
    ("vis_fixed_1", "vis_fixed_1", "rollout_*.png", "rollout_"),
    ("vis_rolling_0", "vis_rolling_0", "rollout_*.png", "rollout_"),
    ("vis_rolling_1", "vis_rolling_1", "rollout_*.png", "rollout_"),
    ("embeddings", "embeddings", "embeddings_*.png", "embeddings_"),
    ("samples_hard", "samples_hard", "hard_*.png", "hard_"),
    ("vis_self_distance", "vis_self_distance", "self_distance_*.png", "self_distance_"),
    ("vis_delta_z_pca", "vis_delta_z_pca", "delta_z_pca_*.png", "delta_z_pca_"),
    ("vis_action_alignment", "vis_action_alignment", "action_alignment_*.png", "action_alignment_"),
    ("vis_action_alignment_detail", "vis_action_alignment", "action_alignment_detail_*.png", "action_alignment_detail_"),
    ("vis_cycle_error", "vis_cycle_error", "cycle_error_*.png", "cycle_error_"),
    ("vis_adjacency", "vis_adjacency", "adjacency_*.png", "adjacency_"),
]


def _parse_step_from_stem(stem: str, prefix: str) -> Optional[int]:
    if prefix and stem.startswith(prefix):
        suffix = stem[len(prefix) :]
    else:
        parts = stem.split("_")
        suffix = parts[-1] if parts else stem
    try:
        return int(suffix)
    except ValueError:
        return None


def _collect_visualization_steps(root: Path) -> Dict[str, List[int]]:
    """Gather per-visualization step lists used for comparison previews."""
    step_map: Dict[str, List[int]] = {}
    for key, folder_name, pattern, prefix in VIS_STEP_SPECS:
        folder = root / folder_name
        if not folder.exists():
            continue
        steps: set[int] = set()
        try:
            for png in folder.glob(pattern):
                step = _parse_step_from_stem(png.stem, prefix)
                if step is None:
                    continue
                steps.add(step)
        except OSError:
            continue
        if steps:
            step_map[key] = sorted(steps)
    return step_map


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


def _ensure_model_diff(path: Path) -> str:
    cache_path = path / "server_cache" / "model_diff.txt"
    if cache_path.exists():
        try:
            return cache_path.read_text()
        except OSError:
            pass

    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "web_viewer" / "model_diff.py"
    cmd = [
        "uv",
        "run",
        "python",
        str(script_path),
        "--experiment",
        str(path),
        "--repo-root",
        str(repo_root),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
    if result.returncode != 0:
        failure = result.stderr.strip() or result.stdout.strip() or "unknown failure"
        return f"model diff generation failed: {failure}"
    try:
        return cache_path.read_text()
    except OSError as exc:
        return f"model diff cache missing after generation: {exc}"


def _quick_self_distance_csv(root: Path) -> Optional[Path]:
    """Cheap existence check for self-distance outputs."""
    folder = root / "self_distance"
    if not folder.exists():
        return None
    for path in folder.glob("self_distance_*.csv"):
        return path
    return None


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


def _diagnostics_exists(root: Path) -> bool:
    """Cheap check for any diagnostics output without globbing everything."""
    diag_dirs = [
        (root / "vis_delta_z_pca", "*.png"),
        (root / "vis_delta_z_pca", "*.csv"),
        (root / "vis_action_alignment", "*.png"),
        (root / "vis_action_alignment", "*.csv"),
        (root / "vis_cycle_error", "*.png"),
        (root / "vis_cycle_error", "*.csv"),
        (root / "vis_diagnostics_frames", "*.csv"),
    ]
    for folder, pattern in diag_dirs:
        if not folder.exists():
            continue
        try:
            for _ in folder.glob(pattern):
                return True
        except OSError:
            continue
    return False


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
        "frame_alignment": root / "vis_diagnostics_frames",
    }
    csvs: Dict[str, List[Path]] = {}
    for name, folder in diag_dirs.items():
        if folder.exists():
            files = sorted(folder.glob("*.csv")) + sorted(folder.glob("*.txt"))
            if files:
                csvs[name] = files
    return csvs


def _collect_diagnostics_frames(root: Path) -> Dict[int, List[Tuple[Path, str, str, Optional[int]]]]:
    frames_root = root / "vis_diagnostics_frames"
    if not frames_root.exists():
        return {}
    start_time = time.perf_counter()
    frame_map: Dict[int, List[Tuple[Path, str, str, Optional[int]]]] = {}
    csv_count = 0
    frame_count = 0

    def _natural_sort_key(path_str: str) -> Tuple:
        """Split a path into text/int chunks so state_2 comes before state_10."""
        parts = re.split(r"(\d+)", path_str)
        key: List = []
        for part in parts:
            if not part:
                continue
            if part.isdigit():
                key.append(int(part))
            else:
                key.append(part.lower())
        return tuple(key)

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
                sorted_entries: List[Tuple[Tuple, Path, str, str, Optional[int]]] = []
                csv_count += 1
                for idx, row in enumerate(reader):
                    rel = row.get("image_path")
                    src = (row.get("source_path") or "").strip()
                    action_label = (row.get("action_label") or "").strip()
                    aid_raw = row.get("action_id")
                    try:
                        action_id: Optional[int] = int(aid_raw) if aid_raw not in (None, "", "None") else None
                    except (TypeError, ValueError):
                        action_id = None
                    if not rel:
                        continue
                    sort_basis = src if src else rel
                    sort_key = (_natural_sort_key(sort_basis), idx)
                    sorted_entries.append((sort_key, frames_root / rel, src, action_label, action_id))
                if sorted_entries:
                    sorted_entries.sort(key=lambda t: t[0])
                    frame_count += len(sorted_entries)
                    frame_map[step] = [(p, src, action_label, action_id) for _, p, src, action_label, action_id in sorted_entries]
        except (OSError, csv.Error):
            continue
    _profile("diagnostics.frames", start_time, root, csv_files=csv_count, frames=frame_count)
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
    start_time = time.perf_counter()
    latest_mtime: Optional[float] = None
    files_scanned = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    mtime = item.stat().st_mtime
                    files_scanned += 1
                    if latest_mtime is None or mtime > latest_mtime:
                        latest_mtime = mtime
                except OSError:
                    continue
    except OSError:
        return None
    if latest_mtime is None:
        return None
    _profile("last_modified.rglob", start_time, path, files=files_scanned)
    return datetime.fromtimestamp(latest_mtime)


def _quick_last_modified(path: Path) -> Optional[datetime]:
    """Fast last-modified using depth-1 files and dirs."""
    latest_mtime: Optional[float] = None
    try:
        for child in path.iterdir():
            try:
                mtime = child.stat().st_mtime
                if latest_mtime is None or mtime > latest_mtime:
                    latest_mtime = mtime
            except OSError:
                continue
    except OSError:
        return None
    if latest_mtime is None:
        try:
            return datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            return None
    return datetime.fromtimestamp(latest_mtime)


def _is_all_zero(values: Iterable[float]) -> bool:
    return all(abs(v) <= ALL_ZERO_EPS for v in values)
