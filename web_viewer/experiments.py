from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import csv
import re
import tomli
import tomli_w

from .csv_utils import get_max_step

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
    starred: bool
    archived: bool
    loss_image: Optional[Path]
    loss_csv: Optional[Path]
    rollout_steps: List[int]
    max_step: Optional[Union[int, str]]
    last_modified: Optional[datetime]
    total_params: Optional[int]
    flops_per_step: Optional[int]
    self_distance_csv: Optional[Path]
    self_distance_images: List[Path]
    self_distance_csvs: List[Path]
    state_embedding_csv: Optional[Path]
    state_embedding_images: List[Path]
    state_embedding_csvs: List[Path]
    odometry_images: List[Path]
    diagnostics_images: Dict[str, List[Path]]
    diagnostics_steps: List[int]
    diagnostics_csvs: Dict[str, List[Path]]
    diagnostics_frames: Dict[int, List[Tuple[Path, str, str, Optional[int]]]]
    diagnostics_s_images: Dict[str, List[Path]]
    diagnostics_s_steps: List[int]
    diagnostics_s_csvs: Dict[str, List[Path]]
    graph_diagnostics_images: Dict[str, List[Path]]
    graph_diagnostics_steps: List[int]
    graph_diagnostics_csvs: Dict[str, List[Path]]
    graph_diagnostics_h_images: Dict[str, List[Path]]
    graph_diagnostics_h_steps: List[int]
    graph_diagnostics_h_csvs: Dict[str, List[Path]]
    graph_diagnostics_s_images: Dict[str, List[Path]]
    graph_diagnostics_s_steps: List[int]
    graph_diagnostics_s_csvs: Dict[str, List[Path]]
    vis_ctrl_images: Dict[str, List[Path]]
    vis_ctrl_steps: List[int]
    vis_ctrl_csvs: List[Path]

    def asset_exists(self, relative: str) -> bool:
        return (self.path / relative).exists()


@dataclass
class ExperimentIndex:
    """Lightweight index row for pagination."""

    id: str
    path: Path
    last_modified: Optional[datetime]
    starred: bool
    archived: bool


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
        metadata_custom_path = subdir / "experiment_metadata.txt"
        _, _, starred, archived = _read_metadata(metadata_custom_path)
        index.append(
            ExperimentIndex(
                id=subdir.name,
                path=subdir,
                last_modified=last_modified,
                starred=starred,
                archived=archived,
            )
        )
    index.sort(key=lambda e: e.id, reverse=True)
    return index


def load_experiment(
    path: Path,
    include_self_distance: bool = False,
    include_diagnostics_images: bool = False,
    include_diagnostics_frames: bool = False,
    include_diagnostics_s: bool = False,
    include_graph_diagnostics: bool = False,
    include_vis_ctrl: bool = False,
    include_state_embedding: bool = False,
    include_odometry: bool = False,
    include_graph_diagnostics_s: bool = False,
    include_graph_diagnostics_h: bool = False,
    include_last_modified: bool = False,
    include_rollout_steps: bool = False,
    include_max_step: bool = False,
    include_model_diff: bool = False,
    include_model_diff_generation: bool = False,
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

    state_embedding_images: List[Path] = []
    state_embedding_csvs: List[Path] = []
    latest_state_embedding_csv: Optional[Path] = None
    if include_state_embedding:
        state_embedding_images = _collect_state_embedding_images(path)
        state_embedding_csvs = _collect_state_embedding_csvs(path)
        latest_state_embedding_csv = state_embedding_csvs[-1] if state_embedding_csvs else None
    else:
        latest_state_embedding_csv = _quick_state_embedding_csv(path)
    _profile(
        "load_experiment.state_embedding",
        section_start,
        path,
        images=len(state_embedding_images),
        csvs=len(state_embedding_csvs),
        included=include_state_embedding,
    )
    section_start = time.perf_counter()

    odometry_images: List[Path] = []
    if include_odometry:
        odometry_images = _collect_odometry_images(path)
    _profile(
        "load_experiment.odometry",
        section_start,
        path,
        images=len(odometry_images),
        included=include_odometry,
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

    diagnostics_s_images: Dict[str, List[Path]] = {}
    diagnostics_s_steps: List[int] = []
    diagnostics_s_csvs: Dict[str, List[Path]] = {}
    if include_diagnostics_s:
        diagnostics_s_images = _collect_diagnostics_images_s(path)
        diagnostics_s_steps = _collect_diagnostics_steps(diagnostics_s_images)
        diagnostics_s_csvs = _collect_diagnostics_csvs_s(path)
    else:
        diagnostics_s_steps = [0] if _diagnostics_s_exists(path) else []
    graph_diagnostics_images: Dict[str, List[Path]] = {}
    graph_diagnostics_steps: List[int] = []
    graph_diagnostics_csvs: Dict[str, List[Path]] = {}
    graph_diagnostics_h_images: Dict[str, List[Path]] = {}
    graph_diagnostics_h_steps: List[int] = []
    graph_diagnostics_h_csvs: Dict[str, List[Path]] = {}
    graph_diagnostics_s_images: Dict[str, List[Path]] = {}
    graph_diagnostics_s_steps: List[int] = []
    graph_diagnostics_s_csvs: Dict[str, List[Path]] = {}
    if include_graph_diagnostics:
        graph_diagnostics_images = _collect_graph_diagnostics_images(path)
        graph_diagnostics_steps = _collect_graph_diagnostics_steps(graph_diagnostics_images)
        graph_diagnostics_csvs = _collect_graph_diagnostics_csvs(path)
    else:
        graph_diagnostics_steps = [0] if _graph_diagnostics_exists(path) else []
    if include_graph_diagnostics_h:
        graph_diagnostics_h_images = _collect_graph_diagnostics_images(path, folder_name="graph_diagnostics_h")
        graph_diagnostics_h_steps = _collect_graph_diagnostics_steps(graph_diagnostics_h_images)
        graph_diagnostics_h_csvs = _collect_graph_diagnostics_csvs(path, folder_name="graph_diagnostics_h")
    else:
        graph_diagnostics_h_steps = [0] if _graph_diagnostics_exists(path, folder_name="graph_diagnostics_h") else []
    if include_graph_diagnostics_s:
        graph_diagnostics_s_images = _collect_graph_diagnostics_images(path, folder_name="graph_diagnostics_s")
        graph_diagnostics_s_steps = _collect_graph_diagnostics_steps(graph_diagnostics_s_images)
        graph_diagnostics_s_csvs = _collect_graph_diagnostics_csvs(path, folder_name="graph_diagnostics_s")
    else:
        graph_diagnostics_s_steps = [0] if _graph_diagnostics_exists(path, folder_name="graph_diagnostics_s") else []
    vis_ctrl_images: Dict[str, List[Path]] = {}
    vis_ctrl_steps: List[int] = []
    vis_ctrl_csvs: List[Path] = []
    if include_vis_ctrl:
        vis_ctrl_images = _collect_vis_ctrl_images(path)
        vis_ctrl_steps = _collect_diagnostics_steps(vis_ctrl_images)
        vis_ctrl_csvs = _collect_vis_ctrl_csvs(path)
    else:
        vis_ctrl_steps = [0] if _vis_ctrl_exists(path) else []
    _profile(
        "load_experiment.diagnostics",
        section_start,
        path,
        images=sum(len(v) for v in diagnostics_images.values()),
        steps=len(diagnostics_steps),
        csvs=sum(len(v) for v in diagnostics_csvs.values()),
        frames=sum(len(v) for v in diagnostics_frames.values()),
        include_frames=include_diagnostics_frames,
        graph_images=sum(len(v) for v in graph_diagnostics_images.values()),
        graph_steps=len(graph_diagnostics_steps),
        graph_csvs=sum(len(v) for v in graph_diagnostics_csvs.values()),
        graph_h_images=sum(len(v) for v in graph_diagnostics_h_images.values()),
        graph_h_steps=len(graph_diagnostics_h_steps),
        graph_h_csvs=sum(len(v) for v in graph_diagnostics_h_csvs.values()),
        graph_s_images=sum(len(v) for v in graph_diagnostics_s_images.values()),
        graph_s_steps=len(graph_diagnostics_s_steps),
        graph_s_csvs=sum(len(v) for v in graph_diagnostics_s_csvs.values()),
        include_graph=include_graph_diagnostics,
        include_images=include_diagnostics_images,
    )
    section_start = time.perf_counter()

    notes_path = path / "notes.txt"
    metadata_custom_path = path / "experiment_metadata.txt"
    meta_start = time.perf_counter()
    title, tags, starred, archived = _read_metadata(metadata_custom_path)
    metadata_exists = metadata_path.exists()
    metadata_text = metadata_path.read_text() if metadata_exists else "metadata.txt missing."
    metadata_data_root = _extract_data_root_from_metadata(metadata_text) if metadata_exists else None
    _profile("load_experiment.text_meta.base", meta_start, path)
    meta_start = time.perf_counter()
    metadata_model_diff_text_raw = "—"
    metadata_model_diff_items: List[Tuple[str, str, bool]] = []
    metadata_model_diff_text = "—"
    if include_model_diff:
        if metadata_model_diff_path.exists():
            metadata_model_diff_text_raw = metadata_model_diff_path.read_text()
        elif include_model_diff_generation:
            metadata_model_diff_text_raw = _ensure_model_diff(path)
        else:
            metadata_model_diff_text_raw = "model_diff.txt missing."
        metadata_model_diff_items = _parse_model_diff_items(metadata_model_diff_text_raw)
        metadata_model_diff_text = _render_model_diff(metadata_model_diff_text_raw)
    _profile("load_experiment.text_meta.model_diff", meta_start, path)
    meta_start = time.perf_counter()
    git_metadata_text = metadata_git_path.read_text() if metadata_git_path.exists() else "metadata_git.txt missing."
    git_commit = _extract_git_commit(git_metadata_text)
    notes_text = _read_or_create_notes(notes_path)
    _profile("load_experiment.text_meta.notes", meta_start, path)
    _profile("load_experiment.text_meta", section_start, path)
    section_start = time.perf_counter()

    rollout_steps = _collect_rollout_steps(path) if include_rollout_steps else []
    max_step = get_max_step(loss_csv) if include_max_step and loss_csv and loss_csv.exists() else None
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
        starred=starred,
        archived=archived,
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
        state_embedding_csv=latest_state_embedding_csv,
        state_embedding_images=state_embedding_images,
        state_embedding_csvs=state_embedding_csvs,
        odometry_images=odometry_images,
        diagnostics_images=diagnostics_images,
        diagnostics_steps=diagnostics_steps,
        diagnostics_csvs=diagnostics_csvs,
        diagnostics_frames=diagnostics_frames,
        diagnostics_s_images=diagnostics_s_images,
        diagnostics_s_steps=diagnostics_s_steps,
        diagnostics_s_csvs=diagnostics_s_csvs,
        graph_diagnostics_images=graph_diagnostics_images,
        graph_diagnostics_steps=graph_diagnostics_steps,
        graph_diagnostics_csvs=graph_diagnostics_csvs,
        graph_diagnostics_h_images=graph_diagnostics_h_images,
        graph_diagnostics_h_steps=graph_diagnostics_h_steps,
        graph_diagnostics_h_csvs=graph_diagnostics_h_csvs,
        graph_diagnostics_s_images=graph_diagnostics_s_images,
        graph_diagnostics_s_steps=graph_diagnostics_s_steps,
        graph_diagnostics_s_csvs=graph_diagnostics_s_csvs,
        vis_ctrl_images=vis_ctrl_images,
        vis_ctrl_steps=vis_ctrl_steps,
        vis_ctrl_csvs=vis_ctrl_csvs,
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
    title, _, _, _ = _read_metadata(path)
    return title


def _read_metadata(path: Path) -> tuple[str, str, bool, bool]:
    """Read custom metadata (title, tags, starred, archived) with sane defaults."""
    if not path.exists():
        return "Untitled", "", False, False
    try:
        data = tomli.loads(path.read_text())
    except (tomli.TOMLDecodeError, OSError) as exc:
        raise ValueError(f"Invalid metadata file: {path}") from exc
    raw_title = data.get("title")
    raw_tags = data.get("tags")
    title = raw_title.strip() if isinstance(raw_title, str) and raw_title.strip() else "Untitled"
    tags = _normalize_tags(raw_tags)
    starred = _coerce_bool(data.get("starred"))
    archived = _coerce_bool(data.get("archived"))
    return title, tags, starred, archived


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


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, int):
        return value != 0
    return False


def _write_metadata(
    path: Path,
    title: Optional[str] = None,
    tags: Optional[str] = None,
    starred: Optional[bool] = None,
    archived: Optional[bool] = None,
) -> None:
    """Write combined metadata, preserving existing values."""
    current_title, current_tags, current_starred, current_archived = _read_metadata(path)
    existing_data: Dict[str, object] = {}
    if path.exists():
        try:
            parsed = tomli.loads(path.read_text())
            if isinstance(parsed, dict):
                existing_data = dict(parsed)
        except (tomli.TOMLDecodeError, OSError) as exc:
            raise ValueError(f"Invalid metadata file: {path}") from exc
    next_title = title.strip() if isinstance(title, str) else None
    next_tags = tags.strip() if isinstance(tags, str) else None
    next_starred = current_starred if starred is None else _coerce_bool(starred)
    next_archived = current_archived if archived is None else _coerce_bool(archived)
    payload = dict(existing_data)
    payload.update({
        "title": (next_title if next_title is not None and next_title else current_title or "Untitled"),
        "tags": (next_tags if next_tags is not None else current_tags),
        "starred": next_starred,
        "archived": next_archived,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(payload))


def write_starred(path: Path, starred: bool) -> None:
    _write_metadata(path, starred=starred)


def write_archived(path: Path, archived: bool) -> None:
    _write_metadata(path, archived=archived)


def _extract_data_root_from_metadata(metadata_text: str) -> Optional[str]:
    """Return the first data_root value found within a TOML metadata blob."""
    try:
        parsed = tomli.loads(metadata_text)
    except (tomli.TOMLDecodeError, OSError) as exc:
        raise ValueError("Invalid metadata.txt TOML") from exc

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
    except (tomli.TOMLDecodeError, OSError) as exc:
        raise ValueError(f"Invalid metadata_model.txt TOML: {path}") from exc
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
    ("pca_z", "pca_z", "pca_z_*.png", "pca_z_"),
    ("pca_s", "pca_s", "pca_s_*.png", "pca_s_"),
    ("pca_h", "pca_h", "pca_h_*.png", "pca_h_"),
    ("samples_hard", "samples_hard", "hard_*.png", "hard_"),
    ("vis_self_distance_z", "vis_self_distance_z", "self_distance_z_*.png", "self_distance_z_"),
    ("vis_self_distance_s", "vis_self_distance_s", "self_distance_s_*.png", "self_distance_s_"),
    ("vis_self_distance_h", "vis_self_distance_h", "self_distance_h_*.png", "self_distance_h_"),
    ("vis_delta_z_pca", "vis_delta_z_pca", "delta_z_pca_*.png", "delta_z_pca_"),
    ("vis_delta_s_pca", "vis_delta_s_pca", "delta_s_pca_*.png", "delta_s_pca_"),
    ("vis_delta_h_pca", "vis_delta_h_pca", "delta_h_pca_*.png", "delta_h_pca_"),
    ("vis_odometry_current_z", "vis_odometry", "odometry_z_*.png", "odometry_z_"),
    ("vis_odometry_current_s", "vis_odometry", "odometry_s_*.png", "odometry_s_"),
    ("vis_odometry_current_h", "vis_odometry", "odometry_h_*.png", "odometry_h_"),
    ("vis_odometry_z_vs_z_hat", "vis_odometry", "z_vs_z_hat_*.png", "z_vs_z_hat_"),
    ("vis_odometry_s_vs_s_hat", "vis_odometry", "s_vs_s_hat_*.png", "s_vs_s_hat_"),
    ("vis_odometry_h_vs_h_hat", "vis_odometry", "h_vs_h_hat_*.png", "h_vs_h_hat_"),
    ("vis_action_alignment_detail_z", "vis_action_alignment_z", "action_alignment_detail_*.png", "action_alignment_detail_"),
    ("vis_action_alignment_detail_raw_z", "vis_action_alignment_z_raw", "action_alignment_detail_*.png", "action_alignment_detail_"),
    (
        "vis_action_alignment_detail_centered_z",
        "vis_action_alignment_z_centered",
        "action_alignment_detail_*.png",
        "action_alignment_detail_",
    ),
    ("vis_action_alignment_detail_s", "vis_action_alignment_s", "action_alignment_detail_*.png", "action_alignment_detail_"),
    ("vis_action_alignment_detail_raw_s", "vis_action_alignment_s_raw", "action_alignment_detail_*.png", "action_alignment_detail_"),
    (
        "vis_action_alignment_detail_centered_s",
        "vis_action_alignment_s_centered",
        "action_alignment_detail_*.png",
        "action_alignment_detail_",
    ),
    ("vis_action_alignment_detail_h", "vis_action_alignment_h", "action_alignment_detail_*.png", "action_alignment_detail_"),
    ("vis_action_alignment_detail_raw_h", "vis_action_alignment_h_raw", "action_alignment_detail_*.png", "action_alignment_detail_"),
    (
        "vis_action_alignment_detail_centered_h",
        "vis_action_alignment_h_centered",
        "action_alignment_detail_*.png",
        "action_alignment_detail_",
    ),
    ("vis_cycle_error", "vis_cycle_error_z", "cycle_error_*.png", "cycle_error_"),
    ("vis_cycle_error_s", "vis_cycle_error_s", "cycle_error_*.png", "cycle_error_"),
    ("vis_cycle_error_h", "vis_cycle_error_h", "cycle_error_*.png", "cycle_error_"),
    ("vis_graph_rank1_cdf_z", "graph_diagnostics_z", "rank1_cdf_*.png", "rank1_cdf_"),
    ("vis_graph_rank2_cdf_z", "graph_diagnostics_z", "rank2_cdf_*.png", "rank2_cdf_"),
    ("vis_graph_neff_violin_z", "graph_diagnostics_z", "neff_violin_*.png", "neff_violin_"),
    ("vis_graph_in_degree_hist_z", "graph_diagnostics_z", "in_degree_hist_*.png", "in_degree_hist_"),
    ("vis_graph_edge_consistency_z", "graph_diagnostics_z", "edge_consistency_*.png", "edge_consistency_"),
    ("vis_graph_metrics_history_z", "graph_diagnostics_z", "metrics_history_*.png", "metrics_history_"),
    ("vis_graph_rank1_cdf_h", "graph_diagnostics_h", "rank1_cdf_*.png", "rank1_cdf_"),
    ("vis_graph_rank2_cdf_h", "graph_diagnostics_h", "rank2_cdf_*.png", "rank2_cdf_"),
    ("vis_graph_neff_violin_h", "graph_diagnostics_h", "neff_violin_*.png", "neff_violin_"),
    ("vis_graph_in_degree_hist_h", "graph_diagnostics_h", "in_degree_hist_*.png", "in_degree_hist_"),
    ("vis_graph_edge_consistency_h", "graph_diagnostics_h", "edge_consistency_*.png", "edge_consistency_"),
    ("vis_graph_metrics_history_h", "graph_diagnostics_h", "metrics_history_*.png", "metrics_history_"),
    ("vis_graph_rank1_cdf_s", "graph_diagnostics_s", "rank1_cdf_*.png", "rank1_cdf_"),
    ("vis_graph_rank2_cdf_s", "graph_diagnostics_s", "rank2_cdf_*.png", "rank2_cdf_"),
    ("vis_graph_neff_violin_s", "graph_diagnostics_s", "neff_violin_*.png", "neff_violin_"),
    ("vis_graph_in_degree_hist_s", "graph_diagnostics_s", "in_degree_hist_*.png", "in_degree_hist_"),
    ("vis_graph_edge_consistency_s", "graph_diagnostics_s", "edge_consistency_*.png", "edge_consistency_"),
    ("vis_graph_metrics_history_s", "graph_diagnostics_s", "metrics_history_*.png", "metrics_history_"),
    ("vis_ctrl_smoothness_z", "vis_vis_ctrl", "smoothness_z_*.png", "smoothness_z_"),
    ("vis_ctrl_smoothness_s", "vis_vis_ctrl", "smoothness_s_*.png", "smoothness_s_"),
    ("vis_ctrl_smoothness_h", "vis_vis_ctrl", "smoothness_h_*.png", "smoothness_h_"),
    ("vis_ctrl_composition_z", "vis_vis_ctrl", "composition_error_z_*.png", "composition_error_z_"),
    ("vis_ctrl_composition_s", "vis_vis_ctrl", "composition_error_s_*.png", "composition_error_s_"),
    ("vis_ctrl_composition_h", "vis_vis_ctrl", "composition_error_h_*.png", "composition_error_h_"),
    ("vis_composability_z", "vis_composability_z", "composability_z_*.png", "composability_z_"),
    ("vis_composability_s", "vis_composability_s", "composability_s_*.png", "composability_s_"),
    ("vis_composability_h", "vis_composability_h", "composability_h_*.png", "composability_h_"),
    ("vis_ctrl_stability_z", "vis_vis_ctrl", "stability_z_*.png", "stability_z_"),
    ("vis_ctrl_stability_s", "vis_vis_ctrl", "stability_s_*.png", "stability_s_"),
    ("vis_ctrl_stability_h", "vis_vis_ctrl", "stability_h_*.png", "stability_h_"),
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
    if "vis_self_distance_z" not in step_map:
        fallback_folder = root / "vis_self_distance"
        if fallback_folder.exists():
            steps = set()
            for png in fallback_folder.glob("self_distance_*.png"):
                step = _parse_step_from_stem(png.stem, "self_distance_")
                if step is not None:
                    steps.add(step)
            if steps:
                step_map["vis_self_distance_z"] = sorted(steps)
    if "vis_self_distance_s" not in step_map:
        fallback_folder = root / "vis_state_embedding"
        if fallback_folder.exists():
            steps = set()
            for png in fallback_folder.glob("state_embedding_[0-9]*.png"):
                step = _parse_step_from_stem(png.stem, "state_embedding_")
                if step is not None:
                    steps.add(step)
            if steps:
                step_map["vis_self_distance_s"] = sorted(steps)
    if "vis_action_alignment_detail_z" not in step_map:
        fallback_folder = root / "vis_action_alignment"
        if fallback_folder.exists():
            steps = set()
            for png in fallback_folder.glob("action_alignment_detail_*.png"):
                step = _parse_step_from_stem(png.stem, "action_alignment_detail_")
                if step is not None:
                    steps.add(step)
            if steps:
                step_map["vis_action_alignment_detail_z"] = sorted(steps)
    if "vis_cycle_error" not in step_map:
        fallback_folder = root / "vis_cycle_error"
        if fallback_folder.exists():
            steps = set()
            for png in fallback_folder.glob("cycle_error_*.png"):
                step = _parse_step_from_stem(png.stem, "cycle_error_")
                if step is not None:
                    steps.add(step)
            if steps:
                step_map["vis_cycle_error"] = sorted(steps)
    legacy_graph = root / "graph_diagnostics"
    if legacy_graph.exists():
        legacy_specs = [
            ("vis_graph_rank1_cdf_z", "rank1_cdf_", "rank1_cdf_*.png"),
            ("vis_graph_rank2_cdf_z", "rank2_cdf_", "rank2_cdf_*.png"),
            ("vis_graph_neff_violin_z", "neff_violin_", "neff_violin_*.png"),
            ("vis_graph_in_degree_hist_z", "in_degree_hist_", "in_degree_hist_*.png"),
            ("vis_graph_edge_consistency_z", "edge_consistency_", "edge_consistency_*.png"),
            ("vis_graph_metrics_history_z", "metrics_history_", "metrics_history_*.png"),
        ]
        for key, prefix, pattern in legacy_specs:
            if key in step_map:
                continue
            steps = set()
            for png in legacy_graph.glob(pattern):
                step = _parse_step_from_stem(png.stem, prefix)
                if step is not None:
                    steps.add(step)
            if steps:
                step_map[key] = sorted(steps)
    return step_map


def _collect_self_distance_images(root: Path) -> List[Path]:
    new_folder = root / "vis_self_distance_z"
    old_folder = root / "vis_self_distance"
    new_files = sorted(new_folder.glob("self_distance_z_*.png")) if new_folder.exists() else []
    old_files = sorted(old_folder.glob("self_distance_*.png")) if old_folder.exists() else []
    if new_files and old_files:
        raise ValueError("Both vis_self_distance_z and vis_self_distance contain self-distance images.")
    return new_files or old_files


def _collect_state_embedding_images(root: Path) -> List[Path]:
    hist_folder = root / "vis_state_embedding"
    hist_files = sorted(hist_folder.glob("state_embedding_hist_*.png")) if hist_folder.exists() else []
    new_folder = root / "vis_self_distance_s"
    new_files = []
    if new_folder.exists():
        new_files = sorted(new_folder.glob("self_distance_cosine_*.png")) + sorted(new_folder.glob("self_distance_s_*.png"))
    old_files = []
    if hist_folder.exists():
        old_files = (
            sorted(hist_folder.glob("state_embedding_cosine_*.png"))
            + sorted(hist_folder.glob("state_embedding_[0-9]*.png"))
        )
    if new_files and old_files:
        raise ValueError("Both vis_state_embedding and vis_self_distance_s contain self-distance images.")
    return hist_files + (new_files or old_files)


def _collect_odometry_images(root: Path) -> List[Path]:
    folder = root / "vis_odometry"
    if not folder.exists():
        return []
    return sorted(folder.glob("*.png"))


def _collect_state_embedding_csvs(root: Path) -> List[Path]:
    new_folder = root / "self_distance_s"
    old_folder = root / "state_embedding"
    new_files = sorted(new_folder.glob("self_distance_s_*.csv")) if new_folder.exists() else []
    old_files = sorted(old_folder.glob("state_embedding_*.csv")) if old_folder.exists() else []
    if new_files and old_files:
        raise ValueError("Both self_distance_s and state_embedding contain self-distance CSVs.")
    return new_files or old_files


def _collect_self_distance_csvs(root: Path) -> List[Path]:
    new_folder = root / "self_distance_z"
    old_folder = root / "self_distance"
    new_files = sorted(new_folder.glob("self_distance_z_*.csv")) if new_folder.exists() else []
    old_files = sorted(old_folder.glob("self_distance_*.csv")) if old_folder.exists() else []
    if new_files and old_files:
        raise ValueError("Both self_distance_z and self_distance contain self-distance CSVs.")
    return new_files or old_files


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
    new_folder = root / "self_distance_z"
    old_folder = root / "self_distance"
    new_file = next(new_folder.glob("self_distance_z_*.csv"), None) if new_folder.exists() else None
    old_file = next(old_folder.glob("self_distance_*.csv"), None) if old_folder.exists() else None
    if new_file and old_file:
        raise ValueError("Both self_distance_z and self_distance contain self-distance CSVs.")
    if new_file:
        return new_file
    if old_file:
        return old_file
    return None


def _quick_state_embedding_csv(root: Path) -> Optional[Path]:
    """Cheap existence check for state embedding outputs."""
    new_folder = root / "self_distance_s"
    old_folder = root / "state_embedding"
    new_file = next(new_folder.glob("self_distance_s_*.csv"), None) if new_folder.exists() else None
    old_file = next(old_folder.glob("state_embedding_*.csv"), None) if old_folder.exists() else None
    if new_file and old_file:
        raise ValueError("Both self_distance_s and state_embedding contain self-distance CSVs.")
    if new_file:
        return new_file
    if old_file:
        return old_file
    return None


def _collect_diagnostics_images(root: Path) -> Dict[str, List[Path]]:
    images: Dict[str, List[Path]] = {}
    delta_dir = root / "vis_delta_z_pca"
    if delta_dir.exists():
        delta_imgs = sorted(delta_dir.glob("delta_z_pca_*.png"))
        if delta_imgs:
            images["delta_z_pca"] = delta_imgs
        spectrum_imgs = sorted(delta_dir.glob("delta_z_variance_spectrum_*.png"))
        if spectrum_imgs:
            images["variance_spectrum"] = spectrum_imgs

    alignment_dir = root / "vis_action_alignment_z"
    alignment_imgs = sorted(alignment_dir.glob("action_alignment_detail_*.png")) if alignment_dir.exists() else []
    if not alignment_imgs:
        fallback_dir = root / "vis_action_alignment"
        alignment_imgs = sorted(fallback_dir.glob("action_alignment_detail_*.png")) if fallback_dir.exists() else []
    if alignment_imgs:
        images["action_alignment_detail"] = alignment_imgs

    cycle_dir = root / "vis_cycle_error_z"
    cycle_imgs = sorted(cycle_dir.glob("*.png")) if cycle_dir.exists() else []
    if not cycle_imgs:
        fallback_dir = root / "vis_cycle_error"
        cycle_imgs = sorted(fallback_dir.glob("*.png")) if fallback_dir.exists() else []
    if cycle_imgs:
        images["cycle_error"] = cycle_imgs
    return images


def _collect_diagnostics_images_s(root: Path) -> Dict[str, List[Path]]:
    diag_specs = [
        ("delta_s_pca", root / "vis_delta_s_pca", "delta_s_pca_*.png"),
        ("variance_spectrum_s", root / "vis_delta_s_pca", "delta_s_variance_spectrum_*.png"),
        ("action_alignment_detail_s", root / "vis_action_alignment_s", "action_alignment_detail_*.png"),
        ("cycle_error_s", root / "vis_cycle_error_s", "*.png"),
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
        root / "vis_delta_z_pca",
        root / "vis_action_alignment_z",
        root / "vis_action_alignment",
        root / "vis_cycle_error_z",
        root / "vis_cycle_error",
        root / "vis_diagnostics_frames",
    ]
    suffixes = (".png", ".csv", ".txt")
    for folder in diag_dirs:
        if not folder.exists():
            continue
        if _folder_has_any_file(folder, suffixes):
            return True
    return False


def _diagnostics_s_exists(root: Path) -> bool:
    diag_dirs = [
        root / "vis_delta_s_pca",
        root / "vis_action_alignment_s",
        root / "vis_cycle_error_s",
    ]
    suffixes = (".png", ".csv", ".txt")
    for folder in diag_dirs:
        if not folder.exists():
            continue
        if _folder_has_any_file(folder, suffixes):
            return True
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
    alignment_dir = root / "vis_action_alignment_z"
    if not alignment_dir.exists():
        alignment_dir = root / "vis_action_alignment"
    cycle_dir = root / "vis_cycle_error_z"
    if not cycle_dir.exists():
        cycle_dir = root / "vis_cycle_error"
    diag_dirs = {
        "delta_z_pca": root / "vis_delta_z_pca",
        "action_alignment": alignment_dir,
        "cycle_error": cycle_dir,
        "frame_alignment": root / "vis_diagnostics_frames",
    }
    csvs: Dict[str, List[Path]] = {}
    for name, folder in diag_dirs.items():
        if folder.exists():
            files = sorted(folder.glob("*.csv")) + sorted(folder.glob("*.txt"))
            if files:
                csvs[name] = files
    return csvs


def _collect_diagnostics_csvs_s(root: Path) -> Dict[str, List[Path]]:
    diag_dirs = {
        "delta_s_pca": root / "vis_delta_s_pca",
        "action_alignment_s": root / "vis_action_alignment_s",
        "cycle_error_s": root / "vis_cycle_error_s",
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


def _collect_vis_ctrl_images(root: Path) -> Dict[str, List[Path]]:
    images: Dict[str, List[Path]] = {}
    vis_dir = root / "vis_vis_ctrl"
    specs = [
        ("smoothness_z", "smoothness_z_*.png"),
        ("smoothness_s", "smoothness_s_*.png"),
        ("smoothness_h", "smoothness_h_*.png"),
        ("composition_z", "composition_error_z_*.png"),
        ("composition_s", "composition_error_s_*.png"),
        ("composition_h", "composition_error_h_*.png"),
        ("stability_z", "stability_z_*.png"),
        ("stability_s", "stability_s_*.png"),
        ("stability_h", "stability_h_*.png"),
    ]
    if vis_dir.exists():
        for name, pattern in specs:
            files = sorted(vis_dir.glob(pattern))
            if files:
                images[name] = files
    alignment_z = root / "vis_action_alignment_z"
    alignment_s = root / "vis_action_alignment_s"
    alignment_h = root / "vis_action_alignment_h"
    if alignment_z.exists():
        files = sorted(alignment_z.glob("action_alignment_detail_*.png"))
        if files:
            images["alignment_z"] = files
    if alignment_s.exists():
        files = sorted(alignment_s.glob("action_alignment_detail_*.png"))
        if files:
            images["alignment_s"] = files
    if alignment_h.exists():
        files = sorted(alignment_h.glob("action_alignment_detail_*.png"))
        if files:
            images["alignment_h"] = files
    return images


def _collect_vis_ctrl_csvs(root: Path) -> List[Path]:
    metrics_dir = root / "metrics"
    if metrics_dir.exists():
        history = metrics_dir / "vis_ctrl_metrics.csv"
        if history.exists():
            return [history]
    folder = root / "vis_ctrl"
    if not folder.exists():
        return []
    return sorted(folder.glob("vis_ctrl_metrics_*.csv"))


def _vis_ctrl_exists(root: Path) -> bool:
    vis_dir = root / "vis_vis_ctrl"
    if vis_dir.exists() and any(vis_dir.glob("smoothness_z_*.png")):
        return True
    metrics_dir = root / "metrics"
    if metrics_dir.exists() and (metrics_dir / "vis_ctrl_metrics.csv").exists():
        return True
    return False


def _resolve_graph_diagnostics_folder(root: Path, folder_name: str) -> Optional[Path]:
    if folder_name == "graph_diagnostics_z":
        preferred = root / "graph_diagnostics_z"
        legacy = root / "graph_diagnostics"
        if preferred.exists():
            return preferred
        if legacy.exists():
            return legacy
        return None
    folder = root / folder_name
    return folder if folder.exists() else None


def _collect_graph_diagnostics_images(root: Path, folder_name: str = "graph_diagnostics_z") -> Dict[str, List[Path]]:
    folder = _resolve_graph_diagnostics_folder(root, folder_name)
    if folder is None:
        return {}
    specs = [
        ("rank1_cdf", "rank1_cdf_*.png"),
        ("rank2_cdf", "rank2_cdf_*.png"),
        ("neff_violin", "neff_violin_*.png"),
        ("in_degree_hist", "in_degree_hist_*.png"),
        ("edge_consistency", "edge_consistency_*.png"),
        ("metrics_history", "metrics_history_*.png"),
    ]
    images: Dict[str, List[Path]] = {}
    for name, pattern in specs:
        files = sorted(folder.glob(pattern))
        if files:
            images[name] = files
    latest = folder / "metrics_history_latest.png"
    if latest.exists():
        images.setdefault("metrics_history_latest", []).append(latest)
    return images


def _collect_graph_diagnostics_steps(graph_images: Dict[str, List[Path]]) -> List[int]:
    steps: set[int] = set()
    for paths in graph_images.values():
        for path in paths:
            stem = path.stem
            suffix = stem.split("_")[-1] if "_" in stem else stem
            try:
                steps.add(int(suffix))
            except ValueError:
                continue
    return sorted(steps)


def _collect_graph_diagnostics_csvs(root: Path, folder_name: str = "graph_diagnostics_z") -> Dict[str, List[Path]]:
    folder = _resolve_graph_diagnostics_folder(root, folder_name)
    if folder is None:
        return {}
    csvs: Dict[str, List[Path]] = {}
    metrics_csvs = sorted(folder.glob("metrics_history*.csv"))
    if metrics_csvs:
        csvs["metrics_history"] = metrics_csvs
    return csvs


def _graph_diagnostics_exists(root: Path, folder_name: str = "graph_diagnostics_z") -> bool:
    folder = _resolve_graph_diagnostics_folder(root, folder_name)
    if folder is None:
        return False
    return _folder_has_any_file(folder, (".png", ".csv"))


def _folder_has_any_file(folder: Path, suffixes: Tuple[str, ...]) -> bool:
    """Return True if folder contains any file with a matching suffix."""
    try:
        with os.scandir(folder) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                if entry.name.lower().endswith(suffixes):
                    return True
    except OSError:
        return False
    return False


def _get_last_modified(path: Path) -> Optional[datetime]:
    """Get the most recent modification time for files under metrics/."""
    if not path.is_dir():
        return None
    metrics_dir = path / "metrics"
    if not metrics_dir.is_dir():
        return None
    start_time = time.perf_counter()
    latest_mtime: Optional[float] = None
    files_scanned = 0
    try:
        for item in metrics_dir.rglob("*"):
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
        try:
            return datetime.fromtimestamp(metrics_dir.stat().st_mtime)
        except OSError:
            return None
    _profile("last_modified.rglob", start_time, metrics_dir, files=files_scanned)
    return datetime.fromtimestamp(latest_mtime)


def _quick_last_modified(path: Path) -> Optional[datetime]:
    """Fast last-modified using depth-1 files under metrics/."""
    metrics_dir = path / "metrics"
    if not metrics_dir.is_dir():
        return None
    latest_mtime: Optional[float] = None
    try:
        for child in metrics_dir.iterdir():
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
            return datetime.fromtimestamp(metrics_dir.stat().st_mtime)
        except OSError:
            return None
    return datetime.fromtimestamp(latest_mtime)


def _is_all_zero(values: Iterable[float]) -> bool:
    return all(abs(v) <= ALL_ZERO_EPS for v in values)
