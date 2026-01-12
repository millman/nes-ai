from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .experiments import Experiment


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
    files = experiment.diagnostics_action_alignment_csvs
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
