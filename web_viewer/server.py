from __future__ import annotations

import fnmatch
import logging
import os
import time
from datetime import datetime, timedelta
import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from math import ceil

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import safe_join

from .config import ViewerConfig
from .csv_utils import get_max_step
from .experiments import DIAGNOSTICS_H_PATTERNS
from .diffing import build_metadata_diff_views, diff_metadata
from .experiments import (
    Experiment,
    ExperimentIndex,
    LossCurveData,
    build_experiment_index,
    list_experiments,
    load_experiment,
    load_loss_curves,
    write_notes,
    write_tags,
    write_title,
    write_starred,
    write_archived,
    _collect_visualization_steps,
    extract_alignment_summary,
    _diagnostics_suffix_exists,
    DIAGNOSTICS_P_DIRS,
    DIAGNOSTICS_Z_DIRS,
    _first_matching_csv_candidate,
    QUICK_SELF_DISTANCE_P_CSV_CANDIDATES,
    _any_existing_paths,
    _any_folder_pattern_exists,
    GRAPH_DIAGNOSTICS_H_FOLDER_CANDIDATES,
    GRAPH_DIAGNOSTICS_P_FOLDER_CANDIDATES,
    GRAPH_DIAGNOSTICS_Z_FOLDER_CANDIDATES,
    _resolve_first_existing_folder,
    _folder_has_any_file,
    get_image_folder_specs,
)
from .plots import build_overlay, build_ranking_accuracy_plot

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_PAGE_SIZE = 25

SELF_DISTANCE_Z_CSV_CANDIDATES = [
    ("self_distance_z", "self_distance_z", "self_distance_z_0000000.csv", "self_distance_z_*.csv"),
    ("self_distance", "self_distance", "self_distance_0000000.csv", "self_distance_*.csv"),
]
SELF_DISTANCE_S_CSV_CANDIDATES = [
    ("self_distance_s", "self_distance_s", "self_distance_s_0000000.csv", "self_distance_s_*.csv"),
    ("state_embedding", "state_embedding", "state_embedding_0000000.csv", "state_embedding_*.csv"),
]
SELF_DISTANCE_P_CSV_CANDIDATES = [
    ("self_distance_p", "self_distance_p", "self_distance_p_0000000.csv", "self_distance_p_*.csv"),
    ("self_distance_s", "self_distance_s", "self_distance_s_0000000.csv", "self_distance_s_*.csv"),
    ("state_embedding", "state_embedding", "state_embedding_0000000.csv", "state_embedding_*.csv"),
]
SELF_DISTANCE_H_CSV_CANDIDATES = [
    ("self_distance_h", "self_distance_h", "self_distance_h_0000000.csv", "self_distance_h_*.csv"),
]
SELF_DISTANCE_Z_IMAGE_CANDIDATES = [
    ("vis_self_distance_z", "vis_self_distance_z", "self_distance_z_0000000.png", "self_distance_z_*.png"),
    ("vis_self_distance", "vis_self_distance", "self_distance_0000000.png", "self_distance_*.png"),
]
SELF_DISTANCE_S_IMAGE_CANDIDATES = [
    ("vis_state_embedding", "vis_state_embedding", "state_embedding_0000000.png", "state_embedding_[0-9]*.png"),
    ("vis_self_distance_s", "vis_self_distance_s", "self_distance_s_0000000.png", "self_distance_s_*.png"),
]
SELF_DISTANCE_P_IMAGE_CANDIDATES = [
    ("vis_self_distance_p", "vis_self_distance_p", "self_distance_p_0000000.png", "self_distance_p_*.png"),
    ("vis_self_distance_s", "vis_self_distance_s", "self_distance_s_0000000.png", "self_distance_s_*.png"),
    ("vis_state_embedding", "vis_state_embedding", "state_embedding_0000000.png", "state_embedding_[0-9]*.png"),
]


def _first_matching_file(folder: Path, *, exact_name: Optional[str], pattern: str) -> Optional[Path]:
    if exact_name:
        exact_path = folder / exact_name
        if exact_path.exists():
            return exact_path
    try:
        with os.scandir(folder) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                if fnmatch.fnmatch(entry.name, pattern):
                    return Path(entry.path)
    except OSError:
        return None
    return None


def _has_matching_file(folder: Path, *, exact_name: Optional[str], pattern: str) -> bool:
    if not folder.exists():
        return False
    return _first_matching_file(folder, exact_name=exact_name, pattern=pattern) is not None


def _latest_experiment_id_from_index(
    output_dir: Path,
    *,
    match: Callable[[Path], bool],
    fallback: Optional[Callable[[Path], bool]] = None,
    fallback_to_latest: bool = False,
) -> Optional[str]:
    index_rows = build_experiment_index(output_dir)
    if not index_rows:
        return None
    sorted_rows = sorted(
        index_rows,
        key=lambda row: row.last_modified or datetime.fromtimestamp(0),
        reverse=True,
    )
    fallback_id = None
    for row in sorted_rows:
        if match(row.path):
            return row.id
        if fallback is not None and fallback_id is None and fallback(row.path):
            fallback_id = row.id
    if fallback_id is not None:
        return fallback_id
    if fallback_to_latest:
        return sorted_rows[0].id
    return None


def _diagnostics_z_exists(exp_path: Path) -> bool:
    return _diagnostics_suffix_exists(exp_path, DIAGNOSTICS_Z_DIRS)


def _diagnostics_p_exists(exp_path: Path) -> bool:
    return _diagnostics_suffix_exists(exp_path, DIAGNOSTICS_P_DIRS)


def _diagnostics_h_exists(exp_path: Path) -> bool:
    for folder_name, pattern in DIAGNOSTICS_H_PATTERNS:
        folder = exp_path / folder_name
        if _first_matching_file(folder, exact_name=None, pattern=pattern):
            return True
    return False


def _diagnostics_zhp_match(exp_path: Path) -> bool:
    return _diagnostics_z_exists(exp_path) and _diagnostics_p_exists(exp_path)


def _diagnostics_zhp_fallback(exp_path: Path) -> bool:
    return _diagnostics_z_exists(exp_path) or _diagnostics_p_exists(exp_path)


def _find_candidate_dir(
    exp_path: Path,
    candidates: List[Tuple[str, str, str, str]],
    *,
    conflict_label: str,
) -> Optional[Path]:
    matches: List[Tuple[str, Path]] = []
    for label, folder_name, exact_name, pattern in candidates:
        folder = exp_path / folder_name
        if _has_matching_file(folder, exact_name=exact_name, pattern=pattern):
            matches.append((label, folder))
    if len(matches) > 1:
        labels = ", ".join(label for label, _ in matches)
        raise RuntimeError(f"Found multiple {conflict_label} outputs in {exp_path}: {labels}.")
    return matches[0][1] if matches else None


def _format_last_modified(dt: Optional[datetime]) -> str:
    """Format datetime as relative time for same-day, or date string otherwise."""
    if dt is None:
        return "—"
    now = datetime.now()
    if dt.date() == now.date():
        # Same day - show relative time
        delta = now - dt
        total_seconds = int(delta.total_seconds())
        if total_seconds < 60:
            return "just now"
        minutes = total_seconds // 60
        if minutes < 60:
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        hours = minutes // 60
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        # Different day - show date
        return dt.strftime("%Y-%m-%d %H:%M")


def _safe_get_max_step(loss_csv: Optional[Path]) -> Optional[Union[int, str]]:
    if loss_csv is None or not loss_csv.exists():
        return None
    try:
        return get_max_step(loss_csv)
    except Exception:
        return "ERR"


def create_app(config: Optional[ViewerConfig] = None) -> Flask:
    cfg = config or ViewerConfig()
    profile_enabled = os.environ.get("VIEWER_PROFILE", "").lower() in {"1", "true", "yes", "on"}
    app = Flask(
        __name__,
        template_folder=str(PACKAGE_ROOT / "templates"),
        static_folder=str(PACKAGE_ROOT / "static"),
        static_url_path="/static/web_viewer",
    )
    app.config["VIEWER_CONFIG"] = cfg

    # Register custom Jinja filters
    app.jinja_env.filters["format_last_modified"] = _format_last_modified

    if profile_enabled:
        profile_logger = logging.getLogger("web_viewer.profile")
        profile_logger.setLevel(logging.INFO)
        profile_logger.propagate = True
        # Mirror app logger handlers so profile logs show up alongside request logs.
        if not profile_logger.handlers:
            for handler in app.logger.handlers:
                profile_logger.addHandler(handler)
        if app.logger.level > logging.INFO:
            app.logger.setLevel(logging.INFO)

    def _log_timing(label: str, start_time: float, **fields) -> None:
        """Log timing info when profiling is enabled."""
        if not profile_enabled:
            return
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        field_text = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
        app.logger.info("profile:%s %.1fms %s", label, elapsed_ms, field_text)

    def _parse_graph_history_csv(path: Path) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        try:
            with path.open("r", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    parsed: Dict[str, float] = {}
                    for key, value in row.items():
                        try:
                            parsed[key] = float(value)
                        except (TypeError, ValueError):
                            continue
                    rows.append(parsed)
        except (OSError, csv.Error):
            return []
        rows.sort(key=lambda r: r.get("step", float("inf")))
        return rows

    def _build_vis_ctrl_figure(rows: List[Dict[str, float]]) -> Optional[Dict]:
        if not rows:
            return None
        steps = [row.get("step", float("nan")) for row in rows]
        metric_keys = sorted({key for row in rows for key in row.keys() if key != "step"})
        knn_keys = [key for key in metric_keys if "knn_mean" in key]
        jaccard_keys = [key for key in metric_keys if "jaccard" in key]

        def mean_for_keys(keys: List[str]) -> List[float]:
            combined: List[float] = []
            for row in rows:
                values = [row.get(key) for key in keys if row.get(key) is not None]
                valid = [value for value in values if value == value]
                combined.append(sum(valid) / len(valid) if valid else float("nan"))
            return combined

        comp_keys = [key for key in metric_keys if key.endswith("_composition_error")]
        var_keys = [key for key in metric_keys if key.endswith("_global_variance")]
        knn_by_prefix = {
            "z": [key for key in knn_keys if key.startswith("z_")],
            "s": [key for key in knn_keys if key.startswith("s_")],
            "h": [key for key in knn_keys if key.startswith("h_")],
        }
        jaccard_by_prefix = {
            "z": [key for key in jaccard_keys if key.startswith("z_")],
            "s": [key for key in jaccard_keys if key.startswith("s_")],
            "h": [key for key in jaccard_keys if key.startswith("h_")],
        }

        def mean_values(values: List[float]) -> float:
            valid = [value for value in values if value == value]
            return sum(valid) / len(valid) if valid else float("nan")

        def combined_series(prefix: str) -> List[float]:
            knn_values = mean_for_keys(knn_by_prefix.get(prefix, []))
            jaccard_values = mean_for_keys(jaccard_by_prefix.get(prefix, []))
            comp_values = [row.get(f"{prefix}_composition_error", float("nan")) for row in rows]
            combined: List[float] = []
            for idx in range(len(rows)):
                combined.append(
                    mean_values([knn_values[idx], comp_values[idx], jaccard_values[idx]])
                )
            return combined

        data = [
            {
                "type": "scatter",
                "mode": "lines",
                "name": "z_combined",
                "x": steps,
                "y": combined_series("z"),
                "hovertemplate": "%{y}<extra></extra>",
            },
            {
                "type": "scatter",
                "mode": "lines",
                "name": "s_combined",
                "x": steps,
                "y": combined_series("s"),
                "hovertemplate": "%{y}<extra></extra>",
            },
            {
                "type": "scatter",
                "mode": "lines",
                "name": "h_combined",
                "x": steps,
                "y": combined_series("h"),
                "hovertemplate": "%{y}<extra></extra>",
            },
        ]
        layout = {
            "template": "plotly_white",
            "xaxis": {"title": "Step"},
            "yaxis": {"title": "Combined local smoothness + composition error + stability"},
            "margin": {"t": 30, "b": 40, "l": 60, "r": 20},
            "hovermode": "x unified",
            "hoverlabel": {"namelength": -1, "align": "left"},
            "legend": {"title": {"text": "Metric"}},
        }
        return {"data": data, "layout": layout}

    def _load_all() -> List[Experiment]:
        return list_experiments(cfg.output_dir)

    def _get_experiment_or_404(exp_id: str, *, include_rollout_steps: bool = False) -> Experiment:
        exp_path = cfg.output_dir / exp_id
        experiment = load_experiment(
            exp_path,
            include_rollout_steps=include_rollout_steps,
        )
        if experiment is None:
            abort(404, f"Experiment {exp_id} not found.")
        return experiment

    @app.route("/")
    def index():
        return redirect(url_for("dashboard"))

    @app.route("/dashboard")
    def dashboard():
        route_start = time.perf_counter()
        experiments = []
        starred_experiments = []
        index_start = time.perf_counter()
        index_rows = build_experiment_index(cfg.output_dir)
        _log_timing("dashboard.index", index_start, rows=len(index_rows))
        show_archived = request.args.get("show_archived", "").lower() in {"1", "true", "yes", "on"}
        archived_count = sum(1 for row in index_rows if row.archived)
        visible_rows = index_rows if show_archived else [row for row in index_rows if not row.archived]
        starred_rows = [row for row in visible_rows if row.starred]
        unstarred_rows = [row for row in visible_rows if not row.starred]
        total_items = len(unstarred_rows)
        requested_page = request.args.get("page", type=int)
        requested_page_size = request.args.get("page_size", type=int)
        current_page = 1
        page_size = DEFAULT_PAGE_SIZE

        if requested_page_size and requested_page_size > 0:
            page_size = requested_page_size
        if requested_page and requested_page > 0:
            current_page = requested_page

        total_pages = max(1, ceil(total_items / page_size)) if total_items else 1
        if current_page > total_pages:
            current_page = total_pages

        start_idx = (current_page - 1) * page_size
        end_idx = start_idx + page_size
        selected_ids: List[str] = [row.id for row in unstarred_rows[start_idx:end_idx]]

        for row in starred_rows:
            exp_start = time.perf_counter()
            exp = load_experiment(
                row.path,
                include_max_step=False,
                include_model_diff=True,
                include_model_diff_generation=True,
            )
            if exp is not None:
                exp.max_step = _safe_get_max_step(exp.loss_csv)
                starred_experiments.append(exp)
            _log_timing("dashboard.experiment", exp_start, exp_id=row.id)

        load_block_start = time.perf_counter()
        load_times = []
        for exp_id in selected_ids:
            exp_start = time.perf_counter()
            exp = load_experiment(
                cfg.output_dir / exp_id,
                include_max_step=False,
                include_model_diff=True,
                include_model_diff_generation=True,
            )
            if exp is not None:
                exp.max_step = _safe_get_max_step(exp.loss_csv)
                experiments.append(exp)
            load_times.append((exp_id, (time.perf_counter() - exp_start) * 1000.0))
            _log_timing("dashboard.experiment", exp_start, exp_id=exp_id)
        if profile_enabled and load_times:
            total_load_ms = sum(ms for _, ms in load_times)
            slowest = sorted(load_times, key=lambda t: t[1], reverse=True)[:5]
            slowest_summary = ", ".join(f"{eid}:{ms:.1f}ms" for eid, ms in slowest)
            _log_timing(
                "dashboard.load_experiments",
                load_block_start,
                count=len(load_times),
                total_ms=f"{total_load_ms:.1f}",
                slowest=slowest_summary,
            )
        _log_timing("dashboard.total", route_start, rows=len(selected_ids))
        return render_template(
            "dashboard.html",
            experiments=experiments,
            starred_experiments=starred_experiments,
            dashboard_experiments=starred_experiments + experiments,
            cfg=cfg,
            total_pages=total_pages,
            current_page=current_page,
            page_size=page_size,
            show_archived=show_archived,
            archived_count=archived_count,
            active_nav="dashboard",
            first_experiment_id=(starred_experiments[0].id if starred_experiments else (unstarred_rows[0].id if unstarred_rows else None)),
        )

    @app.route("/grid")
    @app.route("/experiments")
    def experiments_index():
        experiments = _load_all()
        return render_template(
            "experiments_grid.html",
            experiments=experiments,
            cfg=cfg,
            active_nav="experiments",
            first_experiment_id=experiments[0].id if experiments else None,
        )

    @app.route("/comparison")
    def comparison():
        index_rows = build_experiment_index(cfg.output_dir)
        row_by_id = {row.id: row for row in index_rows}

        # Parse comma-separated ids parameter
        ids_param = request.args.get("ids", "")
        raw_ids = [id.strip() for id in ids_param.split(",") if id.strip()] if ids_param else []

        selected_ids = [exp_id for exp_id in raw_ids if exp_id and exp_id in row_by_id]
        if len(selected_ids) < 2 and len(index_rows) >= 2:
            selected_ids = [row.id for row in sorted(index_rows, key=lambda r: r.last_modified or datetime.min, reverse=True)[:2]]

        experiments: List[Experiment] = []
        for exp_id in selected_ids:
            row = row_by_id.get(exp_id)
            if not row:
                continue
            exp = load_experiment(
                row.path,
            )
            if exp is not None:
                experiments.append(exp)

        selected_map = {exp.id: exp for exp in experiments}
        selected_ids = [eid for eid in selected_ids if eid in selected_map]
        image_folder_specs = get_image_folder_specs(experiments[0].path) if experiments else []
        return render_template(
            "comparison.html",
            experiments=experiments,
            cfg=cfg,
            selected_ids=selected_ids,
            selected_map=selected_map,
            image_folder_specs=image_folder_specs,
            active_nav="comparison",
            first_experiment_id=selected_ids[0] if selected_ids else (index_rows[0].id if index_rows else None),
        )

    @app.route("/experiment/<exp_id>")
    def experiment_detail(exp_id: str):
        experiment = _get_experiment_or_404(exp_id, include_rollout_steps=True)
        figure = _build_single_experiment_figure(experiment)
        viz_steps = _collect_visualization_steps(experiment.path)
        image_folder_specs = get_image_folder_specs(experiment.path)
        if experiment.rollout_steps:
            viz_steps.setdefault("vis_fixed_0", experiment.rollout_steps)
            viz_steps.setdefault("__fallback", experiment.rollout_steps)
        return render_template(
            "experiment_detail.html",
            experiment=experiment,
            cfg=cfg,
            figure=figure,
            visualization_steps=viz_steps,
            image_folder_specs=image_folder_specs,
            active_nav="detail",
            active_experiment_id=experiment.id,
            first_experiment_id=experiment.id,
        )

    @app.route("/experiments/<exp_id>")
    def experiment_detail_redirect(exp_id: str):
        return redirect(url_for("experiment_detail", exp_id=exp_id))

    @app.post("/experiments/<exp_id>/notes")
    def update_notes(exp_id: str):
        experiment = _get_experiment_or_404(exp_id)
        payload = request.get_json(force=True, silent=True) or {}
        new_notes = payload.get("notes", "")
        write_notes(experiment.path / "notes.txt", new_notes)
        return jsonify({"status": "ok"})

    @app.post("/experiments/<exp_id>/title")
    def update_title(exp_id: str):
        experiment = _get_experiment_or_404(exp_id)
        payload = request.get_json(force=True, silent=True) or {}
        new_title = payload.get("title", "")
        metadata_path = experiment.path / "experiment_metadata.txt"
        write_title(metadata_path, new_title)
        return jsonify({"status": "ok"})

    @app.post("/experiments/<exp_id>/tags")
    def update_tags(exp_id: str):
        experiment = _get_experiment_or_404(exp_id)
        payload = request.get_json(force=True, silent=True) or {}
        new_tags = payload.get("tags", "")
        metadata_path = experiment.path / "experiment_metadata.txt"
        write_tags(metadata_path, new_tags)
        return jsonify({"status": "ok"})

    def _parse_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(value, int):
            return value != 0
        return False

    @app.post("/experiments/<exp_id>/starred")
    def update_starred(exp_id: str):
        experiment = _get_experiment_or_404(exp_id)
        payload = request.get_json(force=True, silent=True) or {}
        starred = _parse_bool(payload.get("starred"))
        metadata_path = experiment.path / "experiment_metadata.txt"
        write_starred(metadata_path, starred)
        return jsonify({"status": "ok", "starred": starred})

    @app.post("/experiments/<exp_id>/archived")
    def update_archived(exp_id: str):
        experiment = _get_experiment_or_404(exp_id)
        payload = request.get_json(force=True, silent=True) or {}
        archived = _parse_bool(payload.get("archived"))
        metadata_path = experiment.path / "experiment_metadata.txt"
        write_archived(metadata_path, archived)
        return jsonify({"status": "ok", "archived": archived})

    @app.post("/experiments/archive_selected")
    def archive_selected():
        payload = request.get_json(force=True, silent=True) or {}
        exp_ids = payload.get("ids") or []
        archived = _parse_bool(payload.get("archived", True))
        if not isinstance(exp_ids, list) or not exp_ids:
            abort(400, "Provide at least one experiment id.")
        for exp_id in exp_ids:
            exp_path = cfg.output_dir / str(exp_id)
            if not exp_path.is_dir():
                continue
            write_archived(exp_path / "experiment_metadata.txt", archived)
        return jsonify({"status": "ok", "archived": archived, "count": len(exp_ids)})

    @app.post("/comparison/data")
    def comparison_data():
        payload = request.get_json(force=True, silent=True) or {}
        exp_ids = payload.get("ids") or []
        if not isinstance(exp_ids, list) or len(exp_ids) < 2:
            abort(400, "Provide at least two experiment ids.")
        # Preserve the requested order (matches the title row and dataset ids on the page).
        experiments = []
        for exp_id in exp_ids:
            exp_path = cfg.output_dir / exp_id
            if not exp_path.is_dir():
                abort(404, f"Experiment {exp_id} not found.")
            exp = load_experiment(
                exp_path,
                include_rollout_steps=True,
            )
            if exp is None:
                abort(404, f"Experiment {exp_id} not found.")
            experiments.append(exp)
        overlay_data = _build_overlay_data(experiments)
        comparison_rows = _build_comparison_rows(experiments)
        return jsonify(
            {
                "figure": overlay_data,
                "experiments": comparison_rows,
            }
        )

    @app.route("/self_distance_z", defaults={"exp_id": None})
    @app.route("/self_distance_z/<exp_id>")
    def self_distance_z(exp_id: Optional[str]):
        requested = exp_id or request.args.get("id")

        def _latest_self_distance_id() -> Optional[str]:
            index_rows = build_experiment_index(cfg.output_dir)
            if not index_rows:
                return None
            sorted_rows = sorted(
                index_rows,
                key=lambda row: row.last_modified or datetime.fromtimestamp(0),
                reverse=True,
            )
            for row in sorted_rows:
                csv_dir = _find_candidate_dir(
                    row.path,
                    SELF_DISTANCE_Z_CSV_CANDIDATES,
                    conflict_label="self-distance CSV",
                )
                if csv_dir is not None:
                    return row.id
            return None

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir():
                csv_dir = _find_candidate_dir(
                    requested_path,
                    SELF_DISTANCE_Z_CSV_CANDIDATES,
                    conflict_label="self-distance CSV",
                )
                if csv_dir is not None:
                    selected_id = requested
        if selected_id is None:
            selected_id = _latest_self_distance_id()

        if selected_id is None:
            return render_template(
                "self_distance_page.html",
                experiments=[],
                experiment=None,
                page_title="Self-distance (Z)",
                csv_name="self_distance_z.csv",
                cfg=cfg,
                active_nav="self_distance_z",
                active_experiment_id=None,
                first_experiment_id=None,
            )

        selected = load_experiment(
            cfg.output_dir / selected_id,
            include_self_distance=True,
        )
        if selected is None or selected.self_distance_z_csv is None:
            abort(404, "Experiment not found for self-distance.")
        _find_candidate_dir(selected.path, SELF_DISTANCE_Z_IMAGE_CANDIDATES, conflict_label="self-distance image")

        figure = _build_single_experiment_figure(selected)

        return render_template(
            "self_distance_page.html",
            experiments=[],
            experiment=selected,
            figure=figure,
            page_title="Self-distance (Z)",
            csv_name="self_distance_z.csv",
            csv_file=selected.self_distance_z_csv,
            csv_url=url_for('serve_asset', relative_path=f"{selected.id}/{selected.self_distance_z_csv.relative_to(selected.path)}"),
            images_list=selected.self_distance_z_images,
            images_folder="vis_self_distance_z",
            redirect_url_base=url_for('self_distance_z'),
            cfg=cfg,
            active_nav="self_distance_z",
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
        )

    @app.route("/self_distance_h", defaults={"exp_id": None})
    @app.route("/self_distance_h/<exp_id>")
    def self_distance_h(exp_id: Optional[str]):
        requested = exp_id or request.args.get("id")

        def _latest_self_distance_h_id() -> Optional[str]:
            index_rows = build_experiment_index(cfg.output_dir)
            if not index_rows:
                return None
            sorted_rows = sorted(
                index_rows,
                key=lambda row: row.last_modified or datetime.fromtimestamp(0),
                reverse=True,
            )
            for row in sorted_rows:
                csv_dir = _find_candidate_dir(
                    row.path,
                    SELF_DISTANCE_H_CSV_CANDIDATES,
                    conflict_label="self-distance (H) CSV",
                )
                if csv_dir is not None:
                    return row.id
            return None

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir():
                csv_dir = _find_candidate_dir(
                    requested_path,
                    SELF_DISTANCE_H_CSV_CANDIDATES,
                    conflict_label="self-distance (H) CSV",
                )
                if csv_dir is not None:
                    selected_id = requested
        if selected_id is None:
            selected_id = _latest_self_distance_h_id()

        if selected_id is None:
            return render_template(
                "self_distance_page.html",
                experiments=[],
                experiment=None,
                page_title="Self-distance (H)",
                csv_name="self_distance_h.csv",
                cfg=cfg,
                active_nav="self_distance_h",
                active_experiment_id=None,
                first_experiment_id=None,
            )

        selected = load_experiment(
            cfg.output_dir / selected_id,
        )
        if selected is None:
            abort(404, "Experiment not found for self-distance (H).")
        csv_dir = _find_candidate_dir(
            selected.path,
            SELF_DISTANCE_H_CSV_CANDIDATES,
            conflict_label="self-distance (H) CSV",
        )
        h_csvs = sorted(csv_dir.glob("self_distance_h_*.csv")) if csv_dir is not None else []
        latest_h_csv = h_csvs[-1] if h_csvs else None
        if latest_h_csv is None:
            abort(404, "Experiment not found for self-distance (H).")
        self_distance_h_images = sorted((selected.path / "vis_self_distance_h").glob("self_distance_h_*.png"))

        figure = _build_single_experiment_figure(selected)

        return render_template(
            "self_distance_page.html",
            experiments=[],
            experiment=selected,
            figure=figure,
            page_title="Self-distance (H)",
            csv_name="self_distance_h.csv",
            csv_file=latest_h_csv,
            csv_url=url_for('serve_asset', relative_path=f"{selected.id}/{latest_h_csv.relative_to(selected.path)}"),
            images_list=self_distance_h_images,
            images_folder="vis_self_distance_h",
            redirect_url_base=url_for('self_distance_h'),
            cfg=cfg,
            active_nav="self_distance_h",
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
        )

    @app.route("/self_distance_p", defaults={"exp_id": None})
    @app.route("/self_distance_p/<exp_id>")
    def self_distance_p(exp_id: Optional[str]):
        requested = exp_id or request.args.get("id")

        def _latest_self_distance_p_id() -> Optional[str]:
            index_rows = build_experiment_index(cfg.output_dir)
            if not index_rows:
                return None
            sorted_rows = sorted(
                index_rows,
                key=lambda row: row.last_modified or datetime.fromtimestamp(0),
                reverse=True,
            )
            for row in sorted_rows:
                if _first_matching_csv_candidate(
                    row.path,
                    QUICK_SELF_DISTANCE_P_CSV_CANDIDATES,
                    conflict_label="self-distance (P)",
                ):
                    return row.id
            return None

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir():
                if _first_matching_csv_candidate(
                    requested_path,
                    QUICK_SELF_DISTANCE_P_CSV_CANDIDATES,
                    conflict_label="self-distance (P)",
                ):
                    selected_id = requested
        if selected_id is None:
            selected_id = _latest_self_distance_p_id()

        if selected_id is None:
            return render_template(
                "self_distance_page.html",
                experiments=[],
                experiment=None,
                page_title="Self-distance (P)",
                csv_name="self_distance_p.csv",
                cfg=cfg,
                active_nav="self_distance_p",
                active_experiment_id=None,
                first_experiment_id=None,
            )

        selected = load_experiment(
            cfg.output_dir / selected_id,
            include_self_distance_p=True,
        )
        if selected is None or selected.self_distance_p_csv is None:
            abort(404, "Experiment not found for self-distance (P).")

        figure = _build_single_experiment_figure(selected)

        return render_template(
            "self_distance_page.html",
            experiments=[],
            experiment=selected,
            figure=figure,
            page_title="Self-distance (P)",
            csv_name="self_distance_p.csv",
            csv_file=selected.self_distance_p_csv,
            csv_url=url_for(
                "serve_asset",
                relative_path=f"{selected.id}/{selected.self_distance_p_csv.relative_to(selected.path)}",
            ),
            images_list=selected.self_distance_p_images,
            images_folder="vis_self_distance_p",
            redirect_url_base=url_for("self_distance_p"),
            cfg=cfg,
            active_nav="self_distance_p",
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
        )

    @app.route("/self_distance_s", defaults={"exp_id": None})
    @app.route("/self_distance_s/<exp_id>")
    def self_distance_s(exp_id: Optional[str]):
        requested = exp_id or request.args.get("id")

        def _latest_state_embedding_id() -> Optional[str]:
            index_rows = build_experiment_index(cfg.output_dir)
            if not index_rows:
                return None
            sorted_rows = sorted(
                index_rows,
                key=lambda row: row.last_modified or datetime.fromtimestamp(0),
                reverse=True,
            )
            for row in sorted_rows:
                csv_dir = _find_candidate_dir(
                    row.path,
                    SELF_DISTANCE_S_CSV_CANDIDATES,
                    conflict_label="self-distance (S) CSV",
                )
                if csv_dir is not None:
                    return row.id
            return None

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir():
                csv_dir = _find_candidate_dir(
                    requested_path,
                    SELF_DISTANCE_S_CSV_CANDIDATES,
                    conflict_label="self-distance (S) CSV",
                )
                if csv_dir is not None:
                    selected_id = requested
        if selected_id is None:
            selected_id = _latest_state_embedding_id()

        index_rows = build_experiment_index(cfg.output_dir)
        experiment_ids = [row.id for row in sorted(index_rows, key=lambda r: r.id, reverse=True)]

        if selected_id is None:
            return render_template(
                "state_embedding_page.html",
                experiments=experiment_ids,
                experiment=None,
                state_embedding_map={},
                state_embedding_steps=[],
                page_title="Self-distance (S)",
                cfg=cfg,
                active_nav="self_distance_s",
                active_experiment_id=None,
                first_experiment_id=None,
            )

        selected = load_experiment(
            cfg.output_dir / selected_id,
            include_state_embedding=True,
        )
        if selected is None or selected.state_embedding_csv is None:
            abort(404, "Experiment not found for state embedding diagnostics.")
        _find_candidate_dir(selected.path, SELF_DISTANCE_S_IMAGE_CANDIDATES, conflict_label="self-distance (S) image")

        figure = _build_single_experiment_figure(selected)

        distance_paths: List[Path] = []
        for path in selected.state_embedding_images:
            stem = path.stem
            if stem.startswith("self_distance_cosine_") or stem.startswith("state_embedding_cosine_"):
                continue
            if stem.startswith("self_distance_s_") or stem.startswith("state_embedding_"):
                distance_paths.append(path)
        state_map = {
            "state_embedding": _build_step_map(distance_paths, selected.id, selected.path),
            "state_embedding_hist": _build_step_map(
                selected.state_embedding_hist_images,
                selected.id,
                selected.path,
                prefix="state_embedding_hist_",
            ),
        }
        steps = sorted(state_map["state_embedding"].keys())

        return render_template(
            "state_embedding_page.html",
            experiments=experiment_ids,
            experiment=selected,
            figure=figure,
            state_embedding_map=state_map,
            state_embedding_steps=steps,
            page_title="Self-distance (P)",
            cfg=cfg,
            active_nav="self_distance_p",
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
        )

    @app.route("/self_distance_zhp", defaults={"exp_id": None})
    @app.route("/self_distance_zhp/<exp_id>")
    def self_distance_zhp(exp_id: Optional[str]):
        requested = exp_id or request.args.get("id")

        def _latest_self_distance_zhp_id() -> Optional[str]:
            index_rows = build_experiment_index(cfg.output_dir)
            if not index_rows:
                return None
            sorted_rows = sorted(
                index_rows,
                key=lambda row: row.last_modified or datetime.fromtimestamp(0),
                reverse=True,
            )
            for row in sorted_rows:
                z_dir = _find_candidate_dir(
                    row.path,
                    SELF_DISTANCE_Z_CSV_CANDIDATES,
                    conflict_label="self-distance CSV",
                )
                p_dir = _find_candidate_dir(
                    row.path,
                    SELF_DISTANCE_P_CSV_CANDIDATES,
                    conflict_label="self-distance (P) CSV",
                )
                h_dir = _find_candidate_dir(
                    row.path,
                    SELF_DISTANCE_H_CSV_CANDIDATES,
                    conflict_label="self-distance (H) CSV",
                )
                z_has = (
                    z_dir is not None
                    and (
                        _has_matching_file(
                            z_dir,
                            exact_name="self_distance_z_0000000.csv",
                            pattern="self_distance_z_*.csv",
                        )
                        or _has_matching_file(
                            z_dir,
                            exact_name="self_distance_0000000.csv",
                            pattern="self_distance_*.csv",
                        )
                    )
                )
                h_has = (
                    h_dir is not None
                    and _has_matching_file(
                        h_dir,
                        exact_name="self_distance_h_0000000.csv",
                        pattern="self_distance_h_*.csv",
                    )
                )
                p_has = (
                    p_dir is not None
                    and (
                        _has_matching_file(
                            p_dir,
                            exact_name="self_distance_p_0000000.csv",
                            pattern="self_distance_p_*.csv",
                        )
                        or _has_matching_file(
                            p_dir,
                            exact_name="self_distance_s_0000000.csv",
                            pattern="self_distance_s_*.csv",
                        )
                        or _has_matching_file(
                            p_dir,
                            exact_name="state_embedding_0000000.csv",
                            pattern="state_embedding_*.csv",
                        )
                    )
                )
                if z_has or h_has or p_has:
                    return row.id
            return None

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if not requested_path.is_dir():
                abort(404, f"Requested experiment not found: {requested}")
            selected_id = requested
        if selected_id is None:
            selected_id = _latest_self_distance_zhp_id()

        if selected_id is None:
            return render_template(
                "self_distance_zhp.html",
                experiments=[],
                experiment=None,
                page_title="Self-distance (Z/H/P)",
                cfg=cfg,
                active_nav="self_distance_zhp",
                active_experiment_id=None,
                first_experiment_id=None,
            )

        selected = load_experiment(
            cfg.output_dir / selected_id,
            include_self_distance=True,
            include_self_distance_p=True,
            include_state_embedding=True,
        )
        h_dir = (
            _find_candidate_dir(
                selected.path,
                SELF_DISTANCE_H_CSV_CANDIDATES,
                conflict_label="self-distance (H) CSV",
            )
            if selected
            else None
        )
        h_csvs = sorted(h_dir.glob("self_distance_h_*.csv")) if h_dir is not None else []
        latest_h_csv = h_csvs[-1] if h_csvs else None
        _find_candidate_dir(selected.path, SELF_DISTANCE_Z_IMAGE_CANDIDATES, conflict_label="self-distance image")
        _find_candidate_dir(selected.path, SELF_DISTANCE_S_IMAGE_CANDIDATES, conflict_label="self-distance (S) image")
        _find_candidate_dir(selected.path, SELF_DISTANCE_P_IMAGE_CANDIDATES, conflict_label="self-distance (P) image")
        self_distance_h_images = sorted((selected.path / "vis_self_distance_h").glob("self_distance_h_*.png"))

        figure = _build_single_experiment_figure(selected)
        p_csv = selected.self_distance_p_csv or selected.state_embedding_csv
        p_images = selected.self_distance_p_images or selected.state_embedding_images
        missing_parts = []
        if selected.self_distance_z_csv is None:
            missing_parts.append("Z")
        if latest_h_csv is None:
            missing_parts.append("H")
        if p_csv is None:
            missing_parts.append("P")

        return render_template(
            "self_distance_zhp.html",
            experiments=[],
            experiment=selected,
            figure=figure,
            page_title="Self-distance (Z/H/P)",
            self_distance_h_csv=latest_h_csv,
            self_distance_h_images=self_distance_h_images,
            self_distance_p_csv=p_csv,
            self_distance_p_images=p_images,
            cfg=cfg,
            active_nav="self_distance_zhp",
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
            missing_parts=missing_parts,
        )

    @app.route("/odometry", defaults={"exp_id": None})
    @app.route("/odometry/<exp_id>")
    def odometry(exp_id: Optional[str]):
        requested = exp_id or request.args.get("id")

        def _latest_experiment_id() -> Optional[str]:
            index_rows = build_experiment_index(cfg.output_dir)
            if not index_rows:
                return None
            sorted_rows = sorted(
                index_rows,
                key=lambda row: row.last_modified or datetime.fromtimestamp(0),
                reverse=True,
            )
            return sorted_rows[0].id if sorted_rows else None

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir():
                selected_id = requested
        if selected_id is None:
            selected_id = _latest_experiment_id()

        index_rows = build_experiment_index(cfg.output_dir)
        experiment_ids = [row.id for row in sorted(index_rows, key=lambda r: r.id, reverse=True)]

        if selected_id is None:
            return render_template(
                "odometry_page.html",
                experiments=experiment_ids,
                experiment=None,
                figure=None,
                odometry_map={},
                odometry_steps=[],
                page_title="Odometry",
                cfg=cfg,
                active_nav="odometry",
                active_experiment_id=None,
                first_experiment_id=None,
            )

        selected = load_experiment(
            cfg.output_dir / selected_id,
            include_odometry=True,
        )
        if selected is None:
            abort(404, "Experiment not found for odometry.")

        odometry_map: Dict[int, Dict[str, str]] = {}
        extra_paths = []
        pca_z_dir = selected.path / "pca_z"
        if pca_z_dir.exists():
            extra_paths.extend((path, "pca_z", "pca_z_") for path in pca_z_dir.glob("pca_z_*.png"))
        embeddings_dir = selected.path / "embeddings"
        if embeddings_dir.exists():
            extra_paths.extend((path, "pca_z", "embeddings_") for path in embeddings_dir.glob("embeddings_*.png"))
        pca_s_dir = selected.path / "pca_s"
        if pca_s_dir.exists():
            extra_paths.extend((path, "pca_s", "pca_s_") for path in pca_s_dir.glob("pca_s_*.png"))
        pca_h_dir = selected.path / "pca_h"
        if pca_h_dir.exists():
            extra_paths.extend((path, "pca_h", "pca_h_") for path in pca_h_dir.glob("pca_h_*.png"))
        for path in selected.odometry_images:
            extra_paths.append((path, None, None))

        for path, override_label, override_prefix in extra_paths:
            stem = path.stem
            try:
                rel = path.relative_to(selected.path)
            except ValueError:
                continue
            label = override_label
            prefix = override_prefix
            if label is None or prefix is None:
                if stem.startswith("odometry_z_"):
                    label = "odometry_z"
                    prefix = "odometry_z_"
                elif stem.startswith("odometry_s_"):
                    label = "odometry_s"
                    prefix = "odometry_s_"
                elif stem.startswith("odometry_h_"):
                    label = "odometry_h"
                    prefix = "odometry_h_"
                elif stem.startswith("z_vs_z_hat_"):
                    label = "z_vs_z_hat"
                    prefix = "z_vs_z_hat_"
                elif stem.startswith("s_vs_s_hat_"):
                    label = "s_vs_s_hat"
                    prefix = "s_vs_s_hat_"
                elif stem.startswith("h_vs_h_hat_"):
                    label = "h_vs_h_hat"
                    prefix = "h_vs_h_hat_"
            if label is None or prefix is None:
                continue
            suffix = stem[len(prefix) :]
            try:
                step = int(suffix)
            except ValueError:
                continue
            per_step = odometry_map.setdefault(step, {})
            if label in per_step and label == "pca_z" and prefix == "embeddings_":
                continue
            per_step[label] = url_for(
                "serve_asset", relative_path=f"{selected.id}/{rel}"
            )

        steps = sorted(odometry_map.keys())
        figure = _build_single_experiment_figure(selected)

        return render_template(
            "odometry_page.html",
            experiments=experiment_ids,
            experiment=selected,
            figure=figure,
            odometry_map=odometry_map,
            odometry_steps=steps,
            page_title="Odometry",
            cfg=cfg,
            active_nav="odometry",
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
        )

    @app.route("/diagnostics_z", defaults={"exp_id": None})
    @app.route("/diagnostics_z/<exp_id>")
    def diagnostics_z(exp_id: Optional[str]):
        route_start = time.perf_counter()
        requested = exp_id or request.args.get("id")

        def _latest_diagnostics_id() -> Optional[str]:
            return _latest_experiment_id_from_index(cfg.output_dir, match=_diagnostics_z_exists)

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir() and _diagnostics_z_exists(requested_path):
                selected_id = requested
        if selected_id is None:
            selected_id = _latest_diagnostics_id()

        if selected_id is None:
            return render_template(
                "diagnostics_page.html",
                view_label="z",
                page_title="Diagnostics",
                no_experiment_message="No experiments with diagnostics outputs.",
                experiments=[],
                experiment=None,
                diagnostics_map={},
                diagnostics_csv_map={},
                diagnostics_scalars={},
                frame_map={},
                cfg=cfg,
                active_nav="diagnostics",
                active_experiment_id=None,
                first_experiment_id=None,
            )

        selected = load_experiment(
            cfg.output_dir / selected_id,
            include_self_distance=True,
            include_diagnostics_images=True,
            include_diagnostics_frames=True,
            include_state_embedding=True,
        )
        if selected is None:
            abort(404, "Experiment not found for diagnostics.")

        build_maps_start = time.perf_counter()
        figure = _build_single_experiment_figure(selected)
        diagnostics_images = {
            "delta_z_pca": selected.diagnostics_delta_z_pca_images,
            "variance_spectrum": selected.diagnostics_variance_spectrum_images,
            "action_alignment_detail": selected.diagnostics_action_alignment_detail_images,
            "cycle_error": selected.diagnostics_cycle_error_images,
            "straightline_z": selected.diagnostics_z_straightline_images,
            "rollout_divergence": selected.diagnostics_rollout_divergence_images,
            "rollout_divergence_z": selected.diagnostics_rollout_divergence_z_images,
            "z_consistency": selected.diagnostics_z_consistency_images,
            "z_monotonicity": selected.diagnostics_z_monotonicity_images,
            "path_independence": selected.diagnostics_path_independence_images,
            "zp_distance_scatter": selected.diagnostics_zp_distance_scatter_images,
        }
        diagnostics_map: Dict[str, Dict[int, str]] = {}
        for name, paths in diagnostics_images.items():
            per_step = _build_step_map(paths, selected.id, selected.path)
            if per_step:
                diagnostics_map[name] = per_step

        # Add self-distance images keyed by step (matching self_distance_page)
        if selected.self_distance_z_images:
            per_step = _build_step_map(
                selected.self_distance_z_images,
                selected.id,
                selected.path,
                skip_tokens=["cosine"],
            )
            if per_step:
                diagnostics_map["self_distance"] = per_step

        # Add state embedding self-distance images (S) keyed by step.
        if selected.state_embedding_images:
            per_step = _build_step_map(
                selected.state_embedding_images,
                selected.id,
                selected.path,
                skip_tokens=["hist", "cosine"],
            )
            if per_step:
                diagnostics_map["self_distance_s"] = per_step

        diagnostics_csvs = {
            "delta_z_pca": selected.diagnostics_delta_z_pca_csvs,
            "action_alignment": selected.diagnostics_action_alignment_csvs,
            "cycle_error": selected.diagnostics_cycle_error_csvs,
            "frame_alignment": selected.diagnostics_frame_alignment_csvs,
            "rollout_divergence": selected.diagnostics_rollout_divergence_csvs,
            "rollout_divergence_z": selected.diagnostics_rollout_divergence_z_csvs,
            "z_consistency": selected.diagnostics_z_consistency_csvs,
            "z_monotonicity": selected.diagnostics_z_monotonicity_csvs,
            "path_independence": selected.diagnostics_path_independence_csvs,
        }
        diagnostics_csv_map: Dict[str, Dict[int, str]] = {}
        for name, paths in diagnostics_csvs.items():
            per_step = _build_step_map(paths, selected.id, selected.path)
            if per_step:
                diagnostics_csv_map[name] = per_step

        frame_map: Dict[int, List[Dict[str, str]]] = {}
        for step, frames in zip(selected.diagnostics_frame_steps, selected.diagnostics_frame_entries):
            entries: List[Dict[str, str]] = []
            for img_path, src_path, action_label, action_id in frames:
                try:
                    rel = img_path.relative_to(selected.path)
                except ValueError:
                    continue
                entries.append(
                    {
                        "url": url_for("serve_asset", relative_path=f"{selected.id}/{rel}"),
                        "source": src_path or rel.as_posix(),
                        "action": action_label or "",
                        "action_id": "" if action_id is None else str(action_id),
                    }
                )
            if entries:
                frame_map[step] = entries

        alignment_summary_raw = extract_alignment_summary(selected)

        def _maybe_url(path: Optional[Path]) -> Optional[str]:
            if path is None:
                return None
            try:
                rel = path.relative_to(selected.path)
            except ValueError:
                return None
            return url_for("serve_asset", relative_path=f"{selected.id}/{rel}")

        def _maybe_rel(path: Optional[Path]) -> Optional[str]:
            if path is None:
                return None
            try:
                return str(path.relative_to(selected.path))
            except ValueError:
                return path.name

        alignment_summary = None
        if alignment_summary_raw:
            alignment_summary = {
                "step": alignment_summary_raw.get("step"),
                "rows": alignment_summary_raw.get("rows", []),
                "report_url": _maybe_url(alignment_summary_raw.get("report_path")),
                "strength_url": _maybe_url(alignment_summary_raw.get("strength_path")),
                "report_name": _maybe_rel(alignment_summary_raw.get("report_path")),
                "strength_name": _maybe_rel(alignment_summary_raw.get("strength_path")),
            }

        _log_timing(
            "diagnostics.build_maps",
            build_maps_start,
            images=sum(len(v) for v in diagnostics_map.values()),
            csvs=sum(len(v) for v in diagnostics_csv_map.values()),
            frames=sum(len(v) for v in frame_map.values()),
        )
        _log_timing("diagnostics.total", route_start, selected=selected.id)
        return render_template(
            "diagnostics_page.html",
            view_label="z",
            page_title="Diagnostics",
            no_experiment_message="No experiments with diagnostics outputs.",
            experiments=[],
            experiment=selected,
            diagnostics_map=diagnostics_map,
            diagnostics_csv_map=diagnostics_csv_map,
            diagnostics_scalars=_load_diagnostics_scalars(selected.path),
            frame_map=frame_map,
            figure=figure,
            diagnostics_summary=alignment_summary,
            cfg=cfg,
            active_nav="diagnostics",
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
        )

    def _render_diagnostics_pose(exp_id: Optional[str], active_nav: str) -> str:
        route_start = time.perf_counter()
        requested = exp_id or request.args.get("id")

        def _latest_diagnostics_p_id() -> Optional[str]:
            return _latest_experiment_id_from_index(
                cfg.output_dir,
                match=_diagnostics_p_exists,
                fallback_to_latest=True,
            )

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir():
                selected_id = requested
        if selected_id is None:
            selected_id = _latest_diagnostics_p_id()

        if selected_id is None:
            return render_template(
                "diagnostics_page.html",
                view_label="p",
                page_title="Diagnostics (P)",
                no_experiment_message="No experiment selected.",
                experiments=[],
                experiment=None,
                diagnostics_map={},
                diagnostics_csv_map={},
                diagnostics_scalars={},
                frame_map={},
                cfg=cfg,
                active_nav=active_nav,
                active_experiment_id=None,
                first_experiment_id=None,
            )

        selected = load_experiment(
            cfg.output_dir / selected_id,
            include_diagnostics_frames=True,
            include_diagnostics_p=True,
            include_state_embedding=True,
            include_self_distance_p=True,
        )
        if selected is None:
            abort(404, "Experiment not found for diagnostics (P).")

        build_maps_start = time.perf_counter()
        figure = _build_single_experiment_figure(selected)
        diagnostics_p_images = {
            "delta_p_pca": selected.diagnostics_p_delta_p_pca_images,
            "variance_spectrum_p": selected.diagnostics_p_variance_spectrum_images,
            "action_alignment_detail_p": selected.diagnostics_p_action_alignment_detail_images,
            "cycle_error_p": selected.diagnostics_p_cycle_error_images,
            "straightline_p": selected.diagnostics_p_straightline_images,
            "rollout_divergence_p": selected.diagnostics_p_rollout_divergence_images,
        }
        diagnostics_map: Dict[str, Dict[int, str]] = {}
        for name, paths in diagnostics_p_images.items():
            per_step = _build_step_map(paths, selected.id, selected.path)
            if per_step:
                diagnostics_map[name] = per_step

        # Add pose self-distance images keyed by step (fallback to legacy state embedding images).
        pose_images = selected.self_distance_p_images or selected.state_embedding_images
        if pose_images:
            per_step = _build_step_map(
                pose_images,
                selected.id,
                selected.path,
                skip_tokens=["hist", "cosine"],
            )
            if per_step:
                diagnostics_map["self_distance_p"] = per_step

        diagnostics_p_csvs = {
            "delta_p_pca": selected.diagnostics_p_delta_p_pca_csvs,
            "action_alignment_p": selected.diagnostics_p_action_alignment_csvs,
            "cycle_error_p": selected.diagnostics_p_cycle_error_csvs,
            "rollout_divergence_p": selected.diagnostics_p_rollout_divergence_csvs,
        }
        diagnostics_csv_map: Dict[str, Dict[int, str]] = {}
        for name, paths in diagnostics_p_csvs.items():
            per_step = _build_step_map(paths, selected.id, selected.path)
            if per_step:
                diagnostics_csv_map[name] = per_step

        step_set: set[int] = set()
        for per_step in diagnostics_map.values():
            step_set.update(per_step.keys())
        for per_step in diagnostics_csv_map.values():
            step_set.update(per_step.keys())
        diagnostics_steps = sorted(step_set) if step_set else selected.diagnostics_p_steps

        frame_map: Dict[int, List[Dict[str, str]]] = {}
        for step, frames in zip(selected.diagnostics_frame_steps, selected.diagnostics_frame_entries):
            entries: List[Dict[str, str]] = []
            for img_path, src_path, action_label, action_id in frames:
                try:
                    rel = img_path.relative_to(selected.path)
                except ValueError:
                    continue
                entries.append(
                    {
                        "url": url_for("serve_asset", relative_path=f"{selected.id}/{rel}"),
                        "source": src_path or rel.as_posix(),
                        "action": action_label or "",
                        "action_id": "" if action_id is None else str(action_id),
                    }
                )
            if entries:
                frame_map[step] = entries

        _log_timing(
            "diagnostics_p.build_maps",
            build_maps_start,
            images=sum(len(v) for v in diagnostics_map.values()),
            csvs=sum(len(v) for v in diagnostics_csv_map.values()),
            frames=sum(len(v) for v in frame_map.values()),
        )
        _log_timing("diagnostics_p.total", route_start, selected=selected.id)
        return render_template(
            "diagnostics_page.html",
            view_label="p",
            page_title="Diagnostics (P)",
            no_experiment_message="No experiment selected.",
            experiments=[],
            experiment=selected,
            diagnostic_steps=diagnostics_steps,
            diagnostics_map=diagnostics_map,
            diagnostics_csv_map=diagnostics_csv_map,
            diagnostics_scalars=_load_diagnostics_scalars(selected.path),
            frame_map=frame_map,
            figure=figure,
            cfg=cfg,
            active_nav=active_nav,
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
        )

    @app.route("/diagnostics_p", defaults={"exp_id": None})
    @app.route("/diagnostics_p/<exp_id>")
    def diagnostics_p(exp_id: Optional[str]):
        return _render_diagnostics_pose(exp_id, active_nav="diagnostics_p")

    @app.route("/diagnostics_s", defaults={"exp_id": None})
    @app.route("/diagnostics_s/<exp_id>")
    def diagnostics_s(exp_id: Optional[str]):
        return _render_diagnostics_pose(exp_id, active_nav="diagnostics_p")

    @app.route("/diagnostics_h", defaults={"exp_id": None})
    @app.route("/diagnostics_h/<exp_id>")
    def diagnostics_h(exp_id: Optional[str]):
        route_start = time.perf_counter()
        requested = exp_id or request.args.get("id")

        def _latest_diagnostics_h_id() -> Optional[str]:
            return _latest_experiment_id_from_index(
                cfg.output_dir,
                match=_diagnostics_h_exists,
                fallback_to_latest=True,
            )

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir():
                selected_id = requested
        if selected_id is None:
            selected_id = _latest_diagnostics_h_id()

        if selected_id is None:
            return render_template(
                "diagnostics_page.html",
                view_label="h",
                page_title="Diagnostics (H)",
                no_experiment_message="No experiment selected.",
                experiments=[],
                experiment=None,
                diagnostics_map={},
                diagnostics_csv_map={},
                diagnostics_scalars={},
                frame_map={},
                diagnostic_steps=[],
                cfg=cfg,
                active_nav="diagnostics_h",
                active_experiment_id=None,
                first_experiment_id=None,
            )

        selected = load_experiment(
            cfg.output_dir / selected_id,
            include_diagnostics_frames=True,
        )
        if selected is None:
            abort(404, "Experiment not found for diagnostics (H).")

        build_maps_start = time.perf_counter()
        figure = _build_single_experiment_figure(selected)
        diagnostics_map: Dict[str, Dict[int, str]] = {}
        diagnostics_map["delta_h_pca"] = _collect_step_map_from_dir(
            selected.id, selected.path, "vis_delta_h_pca", "delta_h_pca_*.png", "delta_h_pca_"
        )
        diagnostics_map["action_alignment_detail_h"] = _collect_step_map_from_dir(
            selected.id, selected.path, "vis_action_alignment_h", "action_alignment_detail_*.png", "action_alignment_detail_"
        )
        diagnostics_map["cycle_error_h"] = _collect_step_map_from_dir(
            selected.id, selected.path, "vis_cycle_error_h", "cycle_error_*.png", "cycle_error_"
        )
        diagnostics_map["self_distance_h"] = _collect_step_map_from_dir(
            selected.id, selected.path, "vis_self_distance_h", "self_distance_h_*.png", "self_distance_h_"
        )
        diagnostics_map["h_ablation"] = _collect_step_map_from_dir(
            selected.id, selected.path, "vis_h_ablation", "h_ablation_*.png", "h_ablation_"
        )
        diagnostics_map["h_drift_by_action"] = _collect_step_map_from_dir(
            selected.id, selected.path, "vis_h_drift_by_action", "h_drift_by_action_*.png", "h_drift_by_action_"
        )
        diagnostics_map["norm_timeseries"] = _collect_step_map_from_dir(
            selected.id, selected.path, "vis_norm_timeseries", "norm_timeseries_*.png", "norm_timeseries_"
        )

        step_set: set[int] = set()
        for per_step in diagnostics_map.values():
            step_set.update(per_step.keys())
        diagnostic_steps = sorted(step_set)

        frame_map: Dict[int, List[Dict[str, str]]] = {}
        for step, frames in zip(selected.diagnostics_frame_steps, selected.diagnostics_frame_entries):
            entries: List[Dict[str, str]] = []
            for img_path, src_path, action_label, action_id in frames:
                try:
                    rel = img_path.relative_to(selected.path)
                except ValueError:
                    continue
                entries.append(
                    {
                        "url": url_for("serve_asset", relative_path=f"{selected.id}/{rel}"),
                        "source": src_path or rel.as_posix(),
                        "action": action_label or "",
                        "action_id": "" if action_id is None else str(action_id),
                    }
                )
            if entries:
                frame_map[step] = entries

        _log_timing(
            "diagnostics_h.build_maps",
            build_maps_start,
            images=sum(len(v) for v in diagnostics_map.values()),
            frames=sum(len(v) for v in frame_map.values()),
        )
        _log_timing("diagnostics_h.total", route_start, selected=selected.id)
        return render_template(
            "diagnostics_page.html",
            view_label="h",
            page_title="Diagnostics (H)",
            no_experiment_message="No experiment selected.",
            experiments=[],
            experiment=selected,
            diagnostics_map=diagnostics_map,
            diagnostics_csv_map={},
            diagnostics_scalars=_load_diagnostics_scalars(selected.path),
            frame_map=frame_map,
            diagnostic_steps=diagnostic_steps,
            figure=figure,
            cfg=cfg,
            active_nav="diagnostics_h",
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
        )

    @app.route("/diagnostics_zhp", defaults={"exp_id": None})
    @app.route("/diagnostics_zhp/<exp_id>")
    def diagnostics_zhp(exp_id: Optional[str]):
        route_start = time.perf_counter()
        requested = exp_id or request.args.get("id")

        def _latest_diagnostics_zhp_id() -> Optional[str]:
            return _latest_experiment_id_from_index(
                cfg.output_dir,
                match=_diagnostics_zhp_match,
                fallback=_diagnostics_zhp_fallback,
            )

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir():
                selected_id = requested
        if selected_id is None:
            selected_id = _latest_diagnostics_zhp_id()

        if selected_id is None:
            return render_template(
                "diagnostics_page_zhp.html",
                experiments=[],
                experiment=None,
                diagnostics_map_z={},
                diagnostics_map_h={},
                diagnostics_map_p={},
                diagnostics_scalars={},
                diagnostic_steps=[],
                figure=None,
                cfg=cfg,
                active_nav="diagnostics_zhp",
                active_experiment_id=None,
                first_experiment_id=None,
            )

        selected = load_experiment(
            cfg.output_dir / selected_id,
            include_self_distance=True,
            include_diagnostics_images=True,
            include_diagnostics_p=True,
            include_state_embedding=True,
            include_self_distance_p=True,
        )
        if selected is None:
            abort(404, "Experiment not found for diagnostics.")

        build_maps_start = time.perf_counter()
        figure = _build_single_experiment_figure(selected)

        diagnostics_images = {
            "delta_z_pca": selected.diagnostics_delta_z_pca_images,
            "variance_spectrum": selected.diagnostics_variance_spectrum_images,
            "action_alignment_detail": selected.diagnostics_action_alignment_detail_images,
            "cycle_error": selected.diagnostics_cycle_error_images,
            "rollout_divergence": selected.diagnostics_rollout_divergence_images,
            "rollout_divergence_z": selected.diagnostics_rollout_divergence_z_images,
            "z_consistency": selected.diagnostics_z_consistency_images,
            "z_monotonicity": selected.diagnostics_z_monotonicity_images,
            "path_independence": selected.diagnostics_path_independence_images,
            "zp_distance_scatter": selected.diagnostics_zp_distance_scatter_images,
        }
        diagnostics_map_z: Dict[str, Dict[int, str]] = {}
        for name, paths in diagnostics_images.items():
            per_step = _build_step_map(paths, selected.id, selected.path)
            if per_step:
                diagnostics_map_z[name] = per_step

        if selected.self_distance_z_images:
            per_step = _build_step_map(
                selected.self_distance_z_images,
                selected.id,
                selected.path,
                skip_tokens=["cosine"],
            )
            if per_step:
                diagnostics_map_z["self_distance"] = per_step

        diagnostics_map_p: Dict[str, Dict[int, str]] = {}
        diagnostics_p_images = {
            "delta_p_pca": selected.diagnostics_p_delta_p_pca_images,
            "variance_spectrum_p": selected.diagnostics_p_variance_spectrum_images,
            "action_alignment_detail_p": selected.diagnostics_p_action_alignment_detail_images,
            "cycle_error_p": selected.diagnostics_p_cycle_error_images,
            "straightline_p": selected.diagnostics_p_straightline_images,
            "rollout_divergence_p": selected.diagnostics_p_rollout_divergence_images,
        }
        for name, paths in diagnostics_p_images.items():
            per_step = _build_step_map(paths, selected.id, selected.path)
            if per_step:
                diagnostics_map_p[name] = per_step

        pose_images = selected.self_distance_p_images or selected.state_embedding_images
        if pose_images:
            per_step = _build_step_map(
                pose_images,
                selected.id,
                selected.path,
                skip_tokens=["hist", "cosine"],
            )
            if per_step:
                diagnostics_map_p["self_distance_p"] = per_step

        diagnostics_map_h: Dict[str, Dict[int, str]] = {}
        diagnostics_map_h["delta_h_pca"] = _collect_step_map_from_dir(
            selected.id, selected.path, "vis_delta_h_pca", "delta_h_pca_*.png", "delta_h_pca_"
        )
        diagnostics_map_h["action_alignment_detail_h"] = _collect_step_map_from_dir(
            selected.id, selected.path, "vis_action_alignment_h", "action_alignment_detail_*.png", "action_alignment_detail_"
        )
        diagnostics_map_h["cycle_error_h"] = _collect_step_map_from_dir(
            selected.id, selected.path, "vis_cycle_error_h", "cycle_error_*.png", "cycle_error_"
        )
        diagnostics_map_h["straightline_h"] = _collect_step_map_from_dir(
            selected.id, selected.path, "vis_straightline_h", "straightline_h_*.png", "straightline_h_"
        )
        diagnostics_map_h["self_distance_h"] = _collect_step_map_from_dir(
            selected.id, selected.path, "vis_self_distance_h", "self_distance_h_*.png", "self_distance_h_"
        )
        diagnostics_map_h["rollout_divergence_h"] = _collect_step_map_from_dir(
            selected.id,
            selected.path,
            "vis_rollout_divergence_h",
            "rollout_divergence_h_*.png",
            "rollout_divergence_h_",
        )
        diagnostics_map_h["h_ablation"] = _collect_step_map_from_dir(
            selected.id, selected.path, "vis_h_ablation", "h_ablation_*.png", "h_ablation_"
        )
        diagnostics_map_h["h_drift_by_action"] = _collect_step_map_from_dir(
            selected.id, selected.path, "vis_h_drift_by_action", "h_drift_by_action_*.png", "h_drift_by_action_"
        )
        diagnostics_map_h["norm_timeseries"] = _collect_step_map_from_dir(
            selected.id, selected.path, "vis_norm_timeseries", "norm_timeseries_*.png", "norm_timeseries_"
        )

        step_set: set[int] = set()
        for per_step in diagnostics_map_z.values():
            step_set.update(per_step.keys())
        for per_step in diagnostics_map_h.values():
            step_set.update(per_step.keys())
        for per_step in diagnostics_map_p.values():
            step_set.update(per_step.keys())
        diagnostic_steps = sorted(step_set)

        _log_timing(
            "diagnostics_zhp.build_maps",
            build_maps_start,
            images=(
                sum(len(v) for v in diagnostics_map_z.values())
                + sum(len(v) for v in diagnostics_map_h.values())
                + sum(len(v) for v in diagnostics_map_p.values())
            ),
        )
        _log_timing("diagnostics_zhp.total", route_start, selected=selected.id)
        return render_template(
            "diagnostics_page_zhp.html",
            experiments=[],
            experiment=selected,
            diagnostics_map_z=diagnostics_map_z,
            diagnostics_map_h=diagnostics_map_h,
            diagnostics_map_p=diagnostics_map_p,
            diagnostics_scalars=_load_diagnostics_scalars(selected.path),
            diagnostic_steps=diagnostic_steps,
            figure=figure,
            cfg=cfg,
            active_nav="diagnostics_zhp",
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
        )

    @app.route("/graph_diagnostics_z", defaults={"exp_id": None})
    @app.route("/graph_diagnostics_z/<exp_id>")
    def graph_diagnostics_z(exp_id: Optional[str]):
        route_start = time.perf_counter()
        requested = exp_id or request.args.get("id")

        def _latest_graph_id() -> Optional[str]:
            index_rows = build_experiment_index(cfg.output_dir)
            if not index_rows:
                return None
            sorted_rows = sorted(
                index_rows,
                key=lambda row: row.last_modified or datetime.fromtimestamp(0),
                reverse=True,
            )
            for row in sorted_rows:
                graph_folder = _resolve_first_existing_folder(
                    row.path,
                    GRAPH_DIAGNOSTICS_Z_FOLDER_CANDIDATES,
                )
                if graph_folder and _folder_has_any_file(graph_folder, (".png", ".csv")):
                    return row.id
            return None

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir():
                selected_id = requested
        if selected_id is None:
            selected_id = _latest_graph_id()

        if selected_id is None:
            return render_template(
                "graph_diagnostics_page.html",
                experiments=[],
                experiment=None,
                graph_map={},
                graph_steps=[],
                history=[],
                history_url=None,
                history_plot_url=None,
                page_title="Graph Diagnostics (Z)",
                embedding_label="z",
                cfg=cfg,
                active_nav="graph_diagnostics_z",
                active_experiment_id=None,
                first_experiment_id=None,
            )

        selected = load_experiment(
            cfg.output_dir / selected_id,
            include_graph_diagnostics=True,
        )
        if selected is None:
            abort(404, "Experiment not found for graph diagnostics.")

        figure = _build_single_experiment_figure(selected)

        graph_images = {
            "rank1_cdf": selected.graph_diagnostics_rank1_cdf_z_images,
            "rank2_cdf": selected.graph_diagnostics_rank2_cdf_z_images,
            "neff_violin": selected.graph_diagnostics_neff_violin_z_images,
            "in_degree_hist": selected.graph_diagnostics_in_degree_hist_z_images,
            "edge_consistency": selected.graph_diagnostics_edge_consistency_z_images,
            "metrics_history": selected.graph_diagnostics_metrics_history_z_images,
            "metrics_history_latest": selected.graph_diagnostics_metrics_history_latest_z_images,
        }
        graph_map: Dict[str, Dict[int, str]] = {}
        for name, paths in graph_images.items():
            per_step = _build_step_map(paths, selected.id, selected.path)
            if per_step:
                graph_map[name] = per_step

        graph_steps = selected.graph_diagnostics_z_steps
        history_rows: List[Dict[str, float]] = []
        history_url: Optional[str] = None
        history_plot_url: Optional[str] = None
        history_files = selected.graph_diagnostics_metrics_history_z_csvs
        if history_files:
            latest_hist = history_files[-1]
            history_rows = _parse_graph_history_csv(latest_hist)
            try:
                rel = latest_hist.relative_to(selected.path)
                history_url = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            except ValueError:
                history_url = None
        history_images = graph_images.get("metrics_history", [])
        latest_history_images = graph_images.get("metrics_history_latest", [])
        plot_path: Optional[Path] = None
        if latest_history_images:
            plot_path = latest_history_images[-1]
        elif history_images:
            plot_path = history_images[-1]
        if plot_path is not None:
            try:
                rel = plot_path.relative_to(selected.path)
                history_plot_url = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            except ValueError:
                history_plot_url = None
        if not graph_steps and history_rows:
            graph_steps = [int(row.get("step", 0)) for row in history_rows if "step" in row]

        _log_timing(
            "graph_diagnostics.total",
            route_start,
            selected=selected.id,
            images=sum(len(v) for v in graph_map.values()),
            steps=len(graph_steps),
            history=len(history_rows),
        )
        return render_template(
            "graph_diagnostics_page.html",
            experiments=[],
            experiment=selected,
            graph_map=graph_map,
            graph_steps=graph_steps,
            history=history_rows,
            history_url=history_url,
            history_plot_url=history_plot_url,
            figure=figure,
            page_title="Graph Diagnostics (Z)",
            embedding_label="z",
            cfg=cfg,
            active_nav="graph_diagnostics_z",
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
        )

    @app.route("/graph_diagnostics_h", defaults={"exp_id": None})
    @app.route("/graph_diagnostics_h/<exp_id>")
    def graph_diagnostics_h(exp_id: Optional[str]):
        route_start = time.perf_counter()
        requested = exp_id or request.args.get("id")

        def _latest_graph_id() -> Optional[str]:
            index_rows = build_experiment_index(cfg.output_dir)
            if not index_rows:
                return None
            sorted_rows = sorted(
                index_rows,
                key=lambda row: row.last_modified or datetime.fromtimestamp(0),
                reverse=True,
            )
            for row in sorted_rows:
                graph_folder = _resolve_first_existing_folder(
                    row.path,
                    GRAPH_DIAGNOSTICS_H_FOLDER_CANDIDATES,
                )
                if graph_folder and _folder_has_any_file(graph_folder, (".png", ".csv")):
                    return row.id
            return None

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir():
                selected_id = requested
        if selected_id is None:
            selected_id = _latest_graph_id()

        if selected_id is None:
            return render_template(
                "graph_diagnostics_page.html",
                experiments=[],
                experiment=None,
                graph_map={},
                graph_steps=[],
                history=[],
                history_url=None,
                history_plot_url=None,
                page_title="Graph Diagnostics (H)",
                embedding_label="h",
                cfg=cfg,
                active_nav="graph_diagnostics_h",
                active_experiment_id=None,
                first_experiment_id=None,
            )

        selected = load_experiment(
            cfg.output_dir / selected_id,
            include_graph_diagnostics_h=True,
        )
        if selected is None:
            abort(404, "Experiment not found for graph diagnostics.")

        figure = _build_single_experiment_figure(selected)

        graph_images = {
            "rank1_cdf": selected.graph_diagnostics_rank1_cdf_h_images,
            "rank2_cdf": selected.graph_diagnostics_rank2_cdf_h_images,
            "neff_violin": selected.graph_diagnostics_neff_violin_h_images,
            "in_degree_hist": selected.graph_diagnostics_in_degree_hist_h_images,
            "edge_consistency": selected.graph_diagnostics_edge_consistency_h_images,
            "metrics_history": selected.graph_diagnostics_metrics_history_h_images,
            "metrics_history_latest": selected.graph_diagnostics_metrics_history_latest_h_images,
        }
        graph_map: Dict[str, Dict[int, str]] = {}
        for name, paths in graph_images.items():
            per_step = _build_step_map(paths, selected.id, selected.path)
            if per_step:
                graph_map[name] = per_step

        graph_steps = selected.graph_diagnostics_h_steps
        history_rows: List[Dict[str, float]] = []
        history_url: Optional[str] = None
        history_plot_url: Optional[str] = None
        history_files = selected.graph_diagnostics_metrics_history_h_csvs
        if history_files:
            latest_hist = history_files[-1]
            history_rows = _parse_graph_history_csv(latest_hist)
            try:
                rel = latest_hist.relative_to(selected.path)
                history_url = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            except ValueError:
                history_url = None
        history_images = graph_images.get("metrics_history", [])
        latest_history_images = graph_images.get("metrics_history_latest", [])
        plot_path: Optional[Path] = None
        if latest_history_images:
            plot_path = latest_history_images[-1]
        elif history_images:
            plot_path = history_images[-1]
        if plot_path is not None:
            try:
                rel = plot_path.relative_to(selected.path)
                history_plot_url = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            except ValueError:
                history_plot_url = None
        if not graph_steps and history_rows:
            graph_steps = [int(row.get("step", 0)) for row in history_rows if "step" in row]

        _log_timing(
            "graph_diagnostics_h.total",
            route_start,
            selected=selected.id,
            images=sum(len(v) for v in graph_map.values()),
            steps=len(graph_steps),
            history=len(history_rows),
        )
        return render_template(
            "graph_diagnostics_page.html",
            experiments=[],
            experiment=selected,
            graph_map=graph_map,
            graph_steps=graph_steps,
            history=history_rows,
            history_url=history_url,
            history_plot_url=history_plot_url,
            figure=figure,
            page_title="Graph Diagnostics (H)",
            embedding_label="h",
            cfg=cfg,
            active_nav="graph_diagnostics_h",
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
        )

    @app.route("/graph_diagnostics_s", defaults={"exp_id": None})
    @app.route("/graph_diagnostics_s/<exp_id>")
    def graph_diagnostics_s(exp_id: Optional[str]):
        route_start = time.perf_counter()
        requested = exp_id or request.args.get("id")

        def _latest_graph_id() -> Optional[str]:
            index_rows = build_experiment_index(cfg.output_dir)
            if not index_rows:
                return None
            sorted_rows = sorted(
                index_rows,
                key=lambda row: row.last_modified or datetime.fromtimestamp(0),
                reverse=True,
            )
            for row in sorted_rows:
                graph_folder = _resolve_first_existing_folder(
                    row.path,
                    GRAPH_DIAGNOSTICS_P_FOLDER_CANDIDATES,
                )
                if graph_folder and _folder_has_any_file(graph_folder, (".png", ".csv")):
                    return row.id
            return None

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir():
                selected_id = requested
        if selected_id is None:
            selected_id = _latest_graph_id()

        if selected_id is None:
            return render_template(
                "graph_diagnostics_page.html",
                experiments=[],
                experiment=None,
                graph_map={},
                graph_steps=[],
                history=[],
                history_url=None,
                history_plot_url=None,
                page_title="Graph Diagnostics (P)",
                embedding_label="p",
                cfg=cfg,
                active_nav="graph_diagnostics_s",
                active_experiment_id=None,
                first_experiment_id=None,
            )

        selected = load_experiment(
            cfg.output_dir / selected_id,
            include_graph_diagnostics_p=True,
        )
        if selected is None:
            abort(404, "Experiment not found for graph diagnostics.")

        figure = _build_single_experiment_figure(selected)

        graph_images = {
            "rank1_cdf": selected.graph_diagnostics_rank1_cdf_p_images,
            "rank2_cdf": selected.graph_diagnostics_rank2_cdf_p_images,
            "neff_violin": selected.graph_diagnostics_neff_violin_p_images,
            "in_degree_hist": selected.graph_diagnostics_in_degree_hist_p_images,
            "edge_consistency": selected.graph_diagnostics_edge_consistency_p_images,
            "metrics_history": selected.graph_diagnostics_metrics_history_p_images,
            "metrics_history_latest": selected.graph_diagnostics_metrics_history_latest_p_images,
        }
        graph_map: Dict[str, Dict[int, str]] = {}
        for name, paths in graph_images.items():
            per_step = _build_step_map(paths, selected.id, selected.path)
            if per_step:
                graph_map[name] = per_step

        graph_steps = selected.graph_diagnostics_p_steps
        history_rows: List[Dict[str, float]] = []
        history_url: Optional[str] = None
        history_plot_url: Optional[str] = None
        history_files = selected.graph_diagnostics_metrics_history_p_csvs
        if history_files:
            latest_hist = history_files[-1]
            history_rows = _parse_graph_history_csv(latest_hist)
            try:
                rel = latest_hist.relative_to(selected.path)
                history_url = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            except ValueError:
                history_url = None
        history_images = graph_images.get("metrics_history", [])
        latest_history_images = graph_images.get("metrics_history_latest", [])
        plot_path: Optional[Path] = None
        if latest_history_images:
            plot_path = latest_history_images[-1]
        elif history_images:
            plot_path = history_images[-1]
        if plot_path is not None:
            try:
                rel = plot_path.relative_to(selected.path)
                history_plot_url = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            except ValueError:
                history_plot_url = None
        if not graph_steps and history_rows:
            graph_steps = [int(row.get("step", 0)) for row in history_rows if "step" in row]

        _log_timing(
            "graph_diagnostics_p.total",
            route_start,
            selected=selected.id,
            images=sum(len(v) for v in graph_map.values()),
            steps=len(graph_steps),
            history=len(history_rows),
        )
        return render_template(
            "graph_diagnostics_page.html",
            experiments=[],
            experiment=selected,
            graph_map=graph_map,
            graph_steps=graph_steps,
            history=history_rows,
            history_url=history_url,
            history_plot_url=history_plot_url,
            figure=figure,
            page_title="Graph Diagnostics (P)",
            embedding_label="p",
            cfg=cfg,
            active_nav="graph_diagnostics_s",
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
        )

    @app.route("/graph_diagnostics_zhp", defaults={"exp_id": None})
    @app.route("/graph_diagnostics_zhp/<exp_id>")
    def graph_diagnostics_zhp(exp_id: Optional[str]):
        route_start = time.perf_counter()
        requested = exp_id or request.args.get("id")

        def _latest_graph_zhp_id() -> Optional[str]:
            index_rows = build_experiment_index(cfg.output_dir)
            if not index_rows:
                return None
            sorted_rows = sorted(
                index_rows,
                key=lambda row: row.last_modified or datetime.fromtimestamp(0),
                reverse=True,
            )
            for row in sorted_rows:
                graph_z_folder = _resolve_first_existing_folder(
                    row.path,
                    GRAPH_DIAGNOSTICS_Z_FOLDER_CANDIDATES,
                )
                graph_s_folder = _resolve_first_existing_folder(
                    row.path,
                    GRAPH_DIAGNOSTICS_P_FOLDER_CANDIDATES,
                )
                if (
                    graph_z_folder
                    and _folder_has_any_file(graph_z_folder, (".png", ".csv"))
                    and graph_s_folder
                    and _folder_has_any_file(graph_s_folder, (".png", ".csv"))
                ):
                    return row.id
            return None

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir():
                graph_z_folder = _resolve_first_existing_folder(
                    requested_path,
                    GRAPH_DIAGNOSTICS_Z_FOLDER_CANDIDATES,
                )
                graph_s_folder = _resolve_first_existing_folder(
                    requested_path,
                    GRAPH_DIAGNOSTICS_P_FOLDER_CANDIDATES,
                )
                if (
                    graph_z_folder
                    and _folder_has_any_file(graph_z_folder, (".png", ".csv"))
                    and graph_s_folder
                    and _folder_has_any_file(graph_s_folder, (".png", ".csv"))
                ):
                    selected_id = requested
        if selected_id is None:
            selected_id = _latest_graph_zhp_id()

        if selected_id is None:
            return render_template(
                "graph_diagnostics_zhp.html",
                experiments=[],
                experiment=None,
                graph_map_z={},
                graph_map_h={},
                graph_map_s={},
                steps_z=[],
                steps_h=[],
                steps_s=[],
                history_z=[],
                history_h=[],
                history_s=[],
                history_url_z=None,
                history_url_h=None,
                history_url_s=None,
                history_plot_url_z=None,
                history_plot_url_h=None,
                history_plot_url_s=None,
                figure=None,
                cfg=cfg,
                active_nav="graph_diagnostics_zhp",
                active_experiment_id=None,
                first_experiment_id=None,
            )

        selected = load_experiment(
            cfg.output_dir / selected_id,
            include_graph_diagnostics=True,
            include_graph_diagnostics_h=True,
            include_graph_diagnostics_p=True,
        )
        if selected is None:
            abort(404, "Experiment not found for graph diagnostics (Z/H/P).")

        figure = _build_single_experiment_figure(selected)
        graph_images_z = {
            "rank1_cdf": selected.graph_diagnostics_rank1_cdf_z_images,
            "rank2_cdf": selected.graph_diagnostics_rank2_cdf_z_images,
            "neff_violin": selected.graph_diagnostics_neff_violin_z_images,
            "in_degree_hist": selected.graph_diagnostics_in_degree_hist_z_images,
            "edge_consistency": selected.graph_diagnostics_edge_consistency_z_images,
            "metrics_history": selected.graph_diagnostics_metrics_history_z_images,
            "metrics_history_latest": selected.graph_diagnostics_metrics_history_latest_z_images,
        }
        graph_images_h = {
            "rank1_cdf": selected.graph_diagnostics_rank1_cdf_h_images,
            "rank2_cdf": selected.graph_diagnostics_rank2_cdf_h_images,
            "neff_violin": selected.graph_diagnostics_neff_violin_h_images,
            "in_degree_hist": selected.graph_diagnostics_in_degree_hist_h_images,
            "edge_consistency": selected.graph_diagnostics_edge_consistency_h_images,
            "metrics_history": selected.graph_diagnostics_metrics_history_h_images,
            "metrics_history_latest": selected.graph_diagnostics_metrics_history_latest_h_images,
        }
        graph_images_p = {
            "rank1_cdf": selected.graph_diagnostics_rank1_cdf_p_images,
            "rank2_cdf": selected.graph_diagnostics_rank2_cdf_p_images,
            "neff_violin": selected.graph_diagnostics_neff_violin_p_images,
            "in_degree_hist": selected.graph_diagnostics_in_degree_hist_p_images,
            "edge_consistency": selected.graph_diagnostics_edge_consistency_p_images,
            "metrics_history": selected.graph_diagnostics_metrics_history_p_images,
            "metrics_history_latest": selected.graph_diagnostics_metrics_history_latest_p_images,
        }

        def _build_graph_map(paths_by_name: Dict[str, List[Path]]) -> Dict[str, Dict[int, str]]:
            out: Dict[str, Dict[int, str]] = {}
            for name, paths in paths_by_name.items():
                per_step = _build_step_map(paths, selected.id, selected.path)
                if per_step:
                    out[name] = per_step
            return out

        graph_map_z = _build_graph_map(graph_images_z)
        graph_map_h = _build_graph_map(graph_images_h)
        graph_map_s = _build_graph_map(graph_images_p)
        steps_z = selected.graph_diagnostics_z_steps
        steps_h = selected.graph_diagnostics_h_steps
        steps_s = selected.graph_diagnostics_p_steps

        history_rows_z: List[Dict[str, float]] = []
        history_rows_h: List[Dict[str, float]] = []
        history_rows_s: List[Dict[str, float]] = []
        history_url_z: Optional[str] = None
        history_url_h: Optional[str] = None
        history_url_s: Optional[str] = None
        history_plot_url_z: Optional[str] = None
        history_plot_url_h: Optional[str] = None
        history_plot_url_s: Optional[str] = None

        history_files_z = selected.graph_diagnostics_metrics_history_z_csvs
        if history_files_z:
            latest_hist = history_files_z[-1]
            history_rows_z = _parse_graph_history_csv(latest_hist)
            try:
                rel = latest_hist.relative_to(selected.path)
                history_url_z = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            except ValueError:
                history_url_z = None

        history_files_s = selected.graph_diagnostics_metrics_history_p_csvs
        if history_files_s:
            latest_hist = history_files_s[-1]
            history_rows_s = _parse_graph_history_csv(latest_hist)
            try:
                rel = latest_hist.relative_to(selected.path)
                history_url_s = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            except ValueError:
                history_url_s = None

        history_files_h = selected.graph_diagnostics_metrics_history_h_csvs
        if history_files_h:
            latest_hist = history_files_h[-1]
            history_rows_h = _parse_graph_history_csv(latest_hist)
            try:
                rel = latest_hist.relative_to(selected.path)
                history_url_h = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            except ValueError:
                history_url_h = None

        def _latest_history_plot(images: Dict[str, List[Path]]) -> Optional[str]:
            plot_path: Optional[Path] = None
            latest_history_images = images.get("metrics_history_latest", [])
            history_images = images.get("metrics_history", [])
            if latest_history_images:
                plot_path = latest_history_images[-1]
            elif history_images:
                plot_path = history_images[-1]
            if plot_path is None:
                return None
            try:
                rel = plot_path.relative_to(selected.path)
                return url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            except ValueError:
                return None

        history_plot_url_z = _latest_history_plot(graph_images_z)
        history_plot_url_h = _latest_history_plot(graph_images_h)
        history_plot_url_s = _latest_history_plot(graph_images_p)

        _log_timing(
            "graph_diagnostics_zhp.total",
            route_start,
            selected=selected.id,
            images=sum(len(v) for v in graph_map_z.values())
            + sum(len(v) for v in graph_map_h.values())
            + sum(len(v) for v in graph_map_s.values()),
            steps=len(steps_z) + len(steps_h) + len(steps_s),
            history=len(history_rows_z) + len(history_rows_h) + len(history_rows_s),
        )
        return render_template(
            "graph_diagnostics_zhp.html",
            experiments=[],
            experiment=selected,
            graph_map_z=graph_map_z,
            graph_map_h=graph_map_h,
            graph_map_s=graph_map_s,
            steps_z=steps_z,
            steps_h=steps_h,
            steps_s=steps_s,
            history_z=history_rows_z,
            history_h=history_rows_h,
            history_s=history_rows_s,
            history_url_z=history_url_z,
            history_url_h=history_url_h,
            history_url_s=history_url_s,
            history_plot_url_z=history_plot_url_z,
            history_plot_url_h=history_plot_url_h,
            history_plot_url_s=history_plot_url_s,
            figure=figure,
            cfg=cfg,
            active_nav="graph_diagnostics_zhp",
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
        )

    @app.route("/assessment", defaults={"exp_id": None})
    @app.route("/assessment/<exp_id>")
    def assessment(exp_id: Optional[str]):
        requested = exp_id or request.args.get("id")

        def _latest_experiment_id() -> Optional[str]:
            index_rows = build_experiment_index(cfg.output_dir)
            if not index_rows:
                return None
            sorted_rows = sorted(
                index_rows,
                key=lambda row: row.last_modified or datetime.fromtimestamp(0),
                reverse=True,
            )
            return sorted_rows[0].id if sorted_rows else None

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir():
                selected_id = requested
        if selected_id is None:
            selected_id = _latest_experiment_id()

        if selected_id is None:
            return render_template(
                "assessment.html",
                experiments=[],
                experiment=None,
                assessment_rows=[],
                figure=None,
                cfg=cfg,
                active_nav="assessment",
                active_experiment_id=None,
                first_experiment_id=None,
            )
        if exp_id is None:
            return redirect(url_for("assessment", exp_id=selected_id))

        selected = load_experiment(
            cfg.output_dir / selected_id,
            include_self_distance=True,
            include_self_distance_p=True,
            include_state_embedding=True,
            include_diagnostics_images=True,
            include_diagnostics_p=True,
        )
        if selected is None:
            abort(404, "Experiment not found for assessment.")

        h_csvs = sorted((selected.path / "self_distance_h").glob("self_distance_h_*.csv"))
        h_images = sorted((selected.path / "vis_self_distance_h").glob("self_distance_h_*.png"))
        latest_h_csv = h_csvs[-1] if h_csvs else None

        figure = _build_single_experiment_figure(selected)

        def _maybe_url(path: Optional[Path]) -> Optional[str]:
            if path is None:
                return None
            try:
                rel = path.relative_to(selected.path)
            except ValueError:
                return None
            return url_for("serve_asset", relative_path=f"{selected.id}/{rel}")

        def _latest_csv_entry(
            files: List[Path],
            prefixes: Optional[List[str]] = None,
        ) -> Tuple[Optional[Path], Optional[int]]:
            if not files:
                return None, None
            latest = files[-1]
            if prefixes:
                for prefix in prefixes:
                    step = _parse_step_suffix(latest.stem, prefix)
                    if step is not None:
                        return latest, step
            return latest, _parse_step_suffix(latest.stem, "")

        def _build_step_map_with_prefixes(
            paths: List[Path],
            prefixes: List[str],
            *,
            skip_tokens: Optional[List[str]] = None,
        ) -> Dict[int, str]:
            per_step: Dict[int, str] = {}
            for path in paths:
                stem = path.stem
                if skip_tokens and any(token in stem for token in skip_tokens):
                    continue
                step = None
                for prefix in prefixes:
                    step = _parse_step_suffix(stem, prefix)
                    if step is not None:
                        break
                if step is None:
                    continue
                try:
                    rel = path.relative_to(selected.path)
                except ValueError:
                    continue
                per_step[step] = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            return per_step

        assessment_rows: List[Dict[str, object]] = []

        def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
            try:
                if path.suffix.lower() == ".txt":
                    text = path.read_text()
                    lines = text.splitlines()
                    header_idx = 0
                    for idx, line in enumerate(lines):
                        if "\t" in line and "action_id" in line:
                            header_idx = idx
                            break
                    reader = csv.DictReader(lines[header_idx:], delimiter="\t")
                    return [row for row in reader if row]
                with path.open(newline="") as handle:
                    reader = csv.DictReader(handle)
                    return [row for row in reader]
            except OSError:
                return []

        def _to_float(value: object) -> Optional[float]:
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _weighted_mean(rows: List[Dict[str, str]], value_key: str, weight_key: str = "count") -> Optional[float]:
            total = 0.0
            weight = 0.0
            for row in rows:
                val = _to_float(row.get(value_key))
                wt = _to_float(row.get(weight_key)) or 0.0
                if val is None or wt <= 0:
                    continue
                total += val * wt
                weight += wt
            if weight <= 0:
                return None
            return total / weight

        def _last_value(rows: List[Dict[str, str]], keys: List[str]) -> Optional[float]:
            if not rows:
                return None
            row = rows[-1]
            for key in keys:
                val = _to_float(row.get(key))
                if val is not None:
                    return val
            return None

        def _delta_variance_topk(rows: List[Dict[str, str]], k: int = 3) -> Optional[float]:
            if not rows:
                return None
            parsed: List[Tuple[int, float]] = []
            for row in rows:
                comp = _to_float(row.get("component"))
                val = _to_float(row.get("variance_ratio"))
                if comp is None or val is None:
                    continue
                parsed.append((int(comp), val))
            if not parsed:
                return None
            parsed.sort(key=lambda item: item[0])
            return sum(val for _, val in parsed[:k])

        def _judge_low(metric: Optional[float], good: float, ok: float) -> Tuple[str, str]:
            if metric is None:
                return "Unknown", f"Good ≤ {good:.3f}, Ok ≤ {ok:.3f}"
            if metric <= good:
                return "Good", f"Good ≤ {good:.3f}, Ok ≤ {ok:.3f}"
            if metric <= ok:
                return "Ok", f"Good ≤ {good:.3f}, Ok ≤ {ok:.3f}"
            return "Bad", f"Good ≤ {good:.3f}, Ok ≤ {ok:.3f}"

        def _judge_high(metric: Optional[float], good: float, ok: float) -> Tuple[str, str]:
            if metric is None:
                return "Unknown", f"Good ≥ {good:.3f}, Ok ≥ {ok:.3f}"
            if metric >= good:
                return "Good", f"Good ≥ {good:.3f}, Ok ≥ {ok:.3f}"
            if metric >= ok:
                return "Ok", f"Good ≥ {good:.3f}, Ok ≥ {ok:.3f}"
            return "Bad", f"Good ≥ {good:.3f}, Ok ≥ {ok:.3f}"

        def _add_row(
            key: str,
            label: str,
            *,
            csv_files: List[Path],
            csv_prefixes: List[str],
            image_paths: List[Path],
            image_prefixes: List[str],
            image_skip: Optional[List[str]] = None,
            diagnostic: str,
            metric_label: str,
            metric_value: Optional[float],
            judgement: str,
            judgement_detail: str,
        ) -> None:
            csv_path, csv_step = _latest_csv_entry(csv_files, csv_prefixes)
            if csv_path is None:
                return
            image_map = _build_step_map_with_prefixes(image_paths, image_prefixes, skip_tokens=image_skip)
            row = {
                "key": key,
                "label": label,
                "csv_name": csv_path.name,
                "csv_url": _maybe_url(csv_path),
                "csv_step": csv_step,
                "image_map": image_map,
                "diagnostic": diagnostic,
                "metric_label": metric_label,
                "metric_value": metric_value,
                "judgement": judgement,
                "judgement_detail": judgement_detail,
            }
            assessment_rows.append(row)

        def _pick_preferred_csv(files: List[Path], preferred_tokens: List[str]) -> List[Path]:
            if not files:
                return []
            for token in preferred_tokens:
                matches = [p for p in files if token in p.name]
                if matches:
                    return sorted(matches)
            return sorted(files)

        def _self_distance_metric(csv_path: Optional[Path]) -> Optional[float]:
            if csv_path is None:
                return None
            rows = _read_csv_rows(csv_path)
            return _last_value(
                rows,
                ["cosine_distance_to_prior", "distance_to_prior_cosine", "distance_to_prior"],
            )

        def _action_alignment_metric(csv_path: Optional[Path]) -> Optional[float]:
            if csv_path is None:
                return None
            rows = _read_csv_rows(csv_path)
            mean_val = _weighted_mean(rows, "mean_cos", "count")
            if mean_val is not None:
                return mean_val
            mean_val = _weighted_mean(rows, "mean", "count")
            if mean_val is not None:
                return mean_val
            return _last_value(rows, ["mean_cos", "mean"])

        def _cycle_error_metric(csv_path: Optional[Path]) -> Optional[float]:
            if csv_path is None:
                return None
            rows = _read_csv_rows(csv_path)
            mean_val = _weighted_mean(rows, "mean_cycle_error", "count")
            if mean_val is not None:
                return mean_val
            return _last_value(rows, ["mean_cycle_error", "cycle_error"])

        def _delta_variance_metric(csv_path: Optional[Path]) -> Optional[float]:
            if csv_path is None:
                return None
            rows = _read_csv_rows(csv_path)
            return _delta_variance_topk(rows, 3)

        def _z_consistency_metric(csv_path: Optional[Path]) -> Tuple[Optional[float], str, str]:
            if csv_path is None:
                return None, "mean cosine", "Unknown"
            rows = _read_csv_rows(csv_path)
            cosine = _last_value(rows, ["cosine_mean", "mean_cosine", "cosine"])
            if cosine is not None:
                return cosine, "mean cosine", _judge_high(cosine, 0.9, 0.8)
            dist = _last_value(rows, ["distance_mean", "mean_distance", "distance"])
            return dist, "mean distance", _judge_low(dist, 0.2, 0.4)

        z_metric = _self_distance_metric(selected.self_distance_z_csv)
        z_judgement, z_thresholds = _judge_low(z_metric, 0.2, 0.4)
        _add_row(
            "self_distance_z",
            "Self-distance (Z)",
            csv_files=[selected.self_distance_z_csv] if selected.self_distance_z_csv else [],
            csv_prefixes=["self_distance_z_", "self_distance_"],
            image_paths=selected.self_distance_z_images,
            image_prefixes=["self_distance_z_", "self_distance_"],
            image_skip=["cosine"],
            diagnostic="Stability of Z embeddings across time; lower drift indicates smoother dynamics.",
            metric_label="cosine distance to prior",
            metric_value=z_metric,
            judgement=z_judgement,
            judgement_detail=z_thresholds,
        )
        h_metric = _self_distance_metric(latest_h_csv) if latest_h_csv else None
        h_judgement, h_thresholds = _judge_low(h_metric, 0.2, 0.4)
        _add_row(
            "self_distance_h",
            "Self-distance (H)",
            csv_files=h_csvs,
            csv_prefixes=["self_distance_h_"],
            image_paths=h_images,
            image_prefixes=["self_distance_h_"],
            diagnostic="Stability of hidden dynamics state; lower drift suggests stable memory.",
            metric_label="cosine distance to prior",
            metric_value=h_metric,
            judgement=h_judgement,
            judgement_detail=h_thresholds,
        )
        pose_csv = selected.self_distance_p_csv or selected.state_embedding_csv
        pose_csv_files = [pose_csv] if pose_csv else []
        pose_images = selected.self_distance_p_images or selected.state_embedding_images
        p_metric = _self_distance_metric(pose_csv) if pose_csv else None
        p_judgement, p_thresholds = _judge_low(p_metric, 0.2, 0.4)
        _add_row(
            "self_distance_p",
            "Self-distance (P)",
            csv_files=pose_csv_files,
            csv_prefixes=["self_distance_p_", "self_distance_s_", "state_embedding_"],
            image_paths=pose_images,
            image_prefixes=["self_distance_p_", "self_distance_s_", "state_embedding_"],
            image_skip=["cosine", "hist"],
            diagnostic="Pose drift vs prior/first frame; smoother, lower drift is better.",
            metric_label="cosine distance to prior",
            metric_value=p_metric,
            judgement=p_judgement,
            judgement_detail=p_thresholds,
        )
        delta_csvs = _pick_preferred_csv(
            selected.diagnostics_p_delta_p_pca_csvs,
            ["delta_p_pca_variance_", "delta_s_pca_variance_"],
        )
        delta_metric = _delta_variance_metric(delta_csvs[-1]) if delta_csvs else None
        delta_judgement, delta_thresholds = _judge_high(delta_metric, 0.4, 0.25)
        _add_row(
            "delta_p_pca",
            "Delta-p PCA",
            csv_files=delta_csvs,
            csv_prefixes=["delta_p_pca_variance_", "delta_s_pca_variance_"],
            image_paths=selected.diagnostics_p_delta_p_pca_images,
            image_prefixes=["delta_p_pca_", "delta_s_pca_"],
            diagnostic="Pose deltas clustered by action; separation suggests distinct action effects.",
            metric_label="top-3 variance ratio",
            metric_value=delta_metric,
            judgement=delta_judgement,
            judgement_detail=delta_thresholds,
        )
        align_csvs = _pick_preferred_csv(
            selected.diagnostics_p_action_alignment_csvs,
            ["action_alignment_", "action_alignment_report_", "action_alignment_strength_"],
        )
        align_metric = _action_alignment_metric(align_csvs[-1]) if align_csvs else None
        align_judgement, align_thresholds = _judge_high(align_metric, 0.7, 0.5)
        _add_row(
            "action_alignment_p",
            "Action alignment (P)",
            csv_files=align_csvs,
            csv_prefixes=["action_alignment_"],
            image_paths=selected.diagnostics_p_action_alignment_detail_images,
            image_prefixes=["action_alignment_detail_"],
            diagnostic="Mean alignment of pose deltas per action; higher mean cosine is better.",
            metric_label="mean cosine",
            metric_value=align_metric,
            judgement=align_judgement,
            judgement_detail=align_thresholds,
        )
        cycle_csvs = _pick_preferred_csv(
            selected.diagnostics_p_cycle_error_csvs,
            ["cycle_error_summary_", "cycle_error_values_"],
        )
        cycle_metric = _cycle_error_metric(cycle_csvs[-1]) if cycle_csvs else None
        cycle_judgement, cycle_thresholds = _judge_low(cycle_metric, 0.1, 0.2)
        _add_row(
            "cycle_error_p",
            "Cycle error (P)",
            csv_files=cycle_csvs,
            csv_prefixes=["cycle_error_"],
            image_paths=selected.diagnostics_p_cycle_error_images,
            image_prefixes=["cycle_error_"],
            diagnostic="Average cycle error by action; lower means inverse actions cancel cleanly.",
            metric_label="mean cycle error",
            metric_value=cycle_metric,
            judgement=cycle_judgement,
            judgement_detail=cycle_thresholds,
        )
        straight_csvs = _pick_preferred_csv(
            [],
            ["straightline_", "straightline_summary_"],
        )
        straight_metric = _last_value(_read_csv_rows(straight_csvs[-1]), ["curvature_mean", "mean_curvature"]) if straight_csvs else None
        straight_judgement, straight_thresholds = _judge_low(straight_metric, 0.2, 0.4)
        _add_row(
            "straightline_p",
            "Straight-line rays (P)",
            csv_files=straight_csvs,
            csv_prefixes=["straightline_p_", "straightline_s_"],
            image_paths=selected.diagnostics_p_straightline_images,
            image_prefixes=["straightline_p_", "straightline_s_"],
            diagnostic="Repeated actions should trace straight rays; curvature indicates drift.",
            metric_label="mean curvature",
            metric_value=straight_metric,
            judgement=straight_judgement,
            judgement_detail=straight_thresholds,
        )
        z_consistency_csvs = selected.diagnostics_z_consistency_csvs
        z_consistency_metric, z_metric_label, z_judgement = _z_consistency_metric(
            z_consistency_csvs[-1] if z_consistency_csvs else None
        )
        z_thresholds = "Good ≥ 0.900, Ok ≥ 0.800" if z_metric_label == "mean cosine" else "Good ≤ 0.200, Ok ≤ 0.400"
        _add_row(
            "z_consistency",
            "Z consistency",
            csv_files=z_consistency_csvs,
            csv_prefixes=["z_consistency_"],
            image_paths=selected.diagnostics_z_consistency_images,
            image_prefixes=["z_consistency_"],
            diagnostic="Same-frame consistency under noise; higher cosine (or lower distance) is better.",
            metric_label=z_metric_label,
            metric_value=z_consistency_metric,
            judgement=z_judgement,
            judgement_detail=z_thresholds,
        )

        return render_template(
            "assessment.html",
            experiments=[],
            experiment=selected,
            assessment_rows=assessment_rows,
            figure=figure,
            cfg=cfg,
            active_nav="assessment",
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
        )

    @app.route("/vis_ctrl", defaults={"exp_id": None})
    @app.route("/vis_ctrl/<exp_id>")
    def vis_ctrl(exp_id: Optional[str]):
        requested = exp_id or request.args.get("id")

        def _latest_vis_ctrl_id() -> Optional[str]:
            index_rows = build_experiment_index(cfg.output_dir)
            if not index_rows:
                return None
            sorted_rows = sorted(
                index_rows,
                key=lambda row: row.last_modified or datetime.fromtimestamp(0),
                reverse=True,
            )
            for row in sorted_rows:
                if _any_folder_pattern_exists(
                    row.path,
                    [
                        ("vis_vis_ctrl", "smoothness_z_0000000.png", "smoothness_z_*.png"),
                    ],
                ) or _any_existing_paths(row.path, ["metrics/vis_ctrl_metrics.csv"]):
                    return row.id
            return None

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if not requested_path.is_dir():
                abort(404, f"Experiment {requested} not found.")
            selected_id = requested
        if selected_id is None:
            selected_id = _latest_vis_ctrl_id()

        if selected_id is None:
            return render_template(
                "vis_ctrl_page.html",
                experiment=None,
                steps=[],
                vis_ctrl_map_z={},
                vis_ctrl_map_p={},
                vis_ctrl_map_h={},
                csv_url=None,
                cfg=cfg,
                active_nav="vis_ctrl",
                active_experiment_id=None,
                first_experiment_id=None,
            )

        selected = load_experiment(
            cfg.output_dir / selected_id,
            include_vis_ctrl=True,
        )
        if selected is None:
            abort(404, "Experiment not found for Vis v Ctrl outputs.")

        def _build_vis_ctrl_map(
            *paths_by_name_list: Dict[str, List[Path]],
        ) -> Tuple[Dict[str, Dict[int, str]], Dict[str, Dict[int, str]], Dict[str, Dict[int, str]]]:
            map_z: Dict[str, Dict[int, str]] = {}
            map_p: Dict[str, Dict[int, str]] = {}
            map_h: Dict[str, Dict[int, str]] = {}
            for paths_by_name in paths_by_name_list:
                for name, paths in paths_by_name.items():
                    target = (
                        map_z if name.endswith("_z") else map_p if name.endswith("_p") else map_h if name.endswith("_h") else None
                    )
                    if target is None:
                        continue
                    per_step = _build_step_map(paths, selected.id, selected.path)
                    if per_step:
                        target[name] = per_step
            return map_z, map_p, map_h

        vis_ctrl_core_images = {
            "smoothness_z": selected.vis_ctrl_smoothness_z_images,
            "smoothness_p": selected.vis_ctrl_smoothness_p_images,
            "smoothness_h": selected.vis_ctrl_smoothness_h_images,
            "composition_z": selected.vis_ctrl_composition_z_images,
            "composition_p": selected.vis_ctrl_composition_p_images,
            "composition_h": selected.vis_ctrl_composition_h_images,
            "stability_z": selected.vis_ctrl_stability_z_images,
            "stability_p": selected.vis_ctrl_stability_p_images,
            "stability_h": selected.vis_ctrl_stability_h_images,
        }
        vis_ctrl_alignment_z = {"alignment_z": selected.vis_ctrl_alignment_z_images}
        vis_ctrl_alignment_p = {"alignment_p": selected.vis_ctrl_alignment_p_images}
        vis_ctrl_alignment_h = {"alignment_h": selected.vis_ctrl_alignment_h_images}
        vis_ctrl_map_z, vis_ctrl_map_p, vis_ctrl_map_h = _build_vis_ctrl_map(
            vis_ctrl_core_images,
            vis_ctrl_alignment_z,
            vis_ctrl_alignment_p,
            vis_ctrl_alignment_h,
        )
        steps = selected.vis_ctrl_steps
        csv_url: Optional[str] = None
        history_rows: List[Dict[str, float]] = []
        figure: Optional[Dict] = None
        if selected.vis_ctrl_csvs:
            latest_csv = selected.vis_ctrl_csvs[-1]
            history_rows = _parse_graph_history_csv(latest_csv)
            figure = _build_vis_ctrl_figure(history_rows)
            try:
                rel = latest_csv.relative_to(selected.path)
                csv_url = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            except ValueError:
                csv_url = None

        return render_template(
            "vis_ctrl_page.html",
            experiment=selected,
            steps=steps,
            vis_ctrl_map_z=vis_ctrl_map_z,
            vis_ctrl_map_p=vis_ctrl_map_p,
            vis_ctrl_map_h=vis_ctrl_map_h,
            csv_url=csv_url,
            history=history_rows,
            figure=figure,
            cfg=cfg,
            active_nav="vis_ctrl",
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
        )

    @app.route("/assets/<path:relative_path>")
    def serve_asset(relative_path: str):
        target = _resolve_asset_path(cfg.output_dir, relative_path)
        if not target.exists():
            abort(404)
        return send_from_directory(str(target.parent), target.name)

    return app


def _build_overlay_data(experiments: List[Experiment]):
    curve_map: Dict[str, LossCurveData] = {}
    trace_ids: Dict[str, str] = {}
    seen_labels = set()
    for experiment in experiments:
        if experiment.loss_csv is None:
            continue
        curves = load_loss_curves(experiment.loss_csv)
        if curves is None:
            continue
        filtered = {
            name: values
            for name, values in curves.series.items()
            if "loss" in name.lower()
        }
        if not filtered:
            continue
        label = experiment.title if experiment.title and experiment.title != "Untitled" else experiment.name
        unique_label = label if label not in seen_labels else f"{label} ({experiment.id})"
        seen_labels.add(label)
        curve_map[unique_label] = LossCurveData(
            steps=curves.steps,
            cumulative_flops=curves.cumulative_flops,
            elapsed_seconds=curves.elapsed_seconds,
            series=filtered,
        )
        trace_ids[unique_label] = experiment.id
    return build_overlay(curve_map, trace_ids=trace_ids)


def _build_comparison_rows(experiments: List[Experiment]):
    if not experiments:
        return []
    metadata_texts = [exp.metadata_text for exp in experiments]
    metadata_diff_views = build_metadata_diff_views(metadata_texts)
    base_metadata = experiments[0].metadata_text
    rows = []
    for idx, exp in enumerate(experiments):
        diff_text = None
        if idx > 0:
            diff_text = diff_metadata(base_metadata, exp.metadata_text)
        viz_steps = _collect_visualization_steps(exp.path)
        image_folder_specs = get_image_folder_specs(exp.path)
        max_step = exp.max_step
        if max_step is None:
            max_step = _safe_get_max_step(exp.loss_csv)
        rows.append(
            {
                "id": exp.id,
                "name": exp.name,
                "git_commit": exp.git_commit,
                "title": exp.title,
                "tags": exp.tags,
                "rollout_steps": exp.rollout_steps,
                "visualization_steps": viz_steps,
                "image_folder_specs": image_folder_specs,
                "max_step": max_step,
                "metadata": exp.metadata_text,
                "metadata_diff": diff_text,
                "metadata_differences": metadata_diff_views[idx],
                "git_metadata": exp.git_metadata_text,
                "total_params": exp.total_params,
                "flops_per_step": exp.flops_per_step,
                "loss_image": url_for("serve_asset", relative_path=f"{exp.id}/metrics/loss_curves.png")
                if exp.loss_image
                else None,
            }
        )
    return rows


def _build_single_experiment_figure(experiment: Experiment):
    if experiment.loss_csv is None:
        return None
    curves = load_loss_curves(experiment.loss_csv)
    if curves is None:
        return None
    filtered = {
        name: values
        for name, values in curves.series.items()
        if "loss" in name.lower()
    }
    if not filtered:
        return None
    loss_curves = LossCurveData(
        steps=curves.steps,
        cumulative_flops=curves.cumulative_flops,
        elapsed_seconds=curves.elapsed_seconds,
        series=filtered,
    )
    label = experiment.title or experiment.name
    return build_overlay({label: loss_curves}, include_experiment_in_trace=False, trace_ids={label: experiment.id})


def _load_diagnostics_scalars(exp_path: Path) -> Dict[int, Dict[str, float]]:
    scalars_path = exp_path / "metrics" / "diagnostics_scalars.csv"
    if not scalars_path.exists():
        return {}
    scalars: Dict[int, Dict[str, float]] = {}
    try:
        with scalars_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                step_raw = row.get("step")
                if step_raw is None:
                    continue
                try:
                    step = int(float(step_raw))
                except ValueError:
                    continue
                entry = {}
                for key, value in row.items():
                    if key == "step" or value is None:
                        continue
                    try:
                        entry[key] = float(value)
                    except ValueError:
                        continue
                if "p_norm_mean" not in entry and "s_norm_mean" in entry:
                    entry["p_norm_mean"] = entry["s_norm_mean"]
                if "p_norm_p95" not in entry and "s_norm_p95" in entry:
                    entry["p_norm_p95"] = entry["s_norm_p95"]
                if "p_drift_mean" not in entry and "s_drift_mean" in entry:
                    entry["p_drift_mean"] = entry["s_drift_mean"]
                if "id_acc_p" not in entry and "id_acc_s" in entry:
                    entry["id_acc_p"] = entry["id_acc_s"]
                if entry:
                    scalars[step] = entry
    except OSError:
        return {}
    return scalars


def _parse_step_suffix(stem: str, prefix: str) -> Optional[int]:
    if prefix and stem.startswith(prefix):
        suffix = stem[len(prefix) :]
    else:
        suffix = stem.split("_")[-1] if "_" in stem else stem
    try:
        return int(suffix)
    except ValueError:
        return None


def _build_step_map(
    paths: List[Path],
    exp_id: str,
    exp_path: Path,
    prefix: str = "",
    skip_tokens: Optional[List[str]] = None,
) -> Dict[int, str]:
    per_step: Dict[int, str] = {}
    for path in paths:
        stem = path.stem
        if skip_tokens and any(token in stem for token in skip_tokens):
            continue
        step = _parse_step_suffix(stem, prefix)
        if step is None:
            continue
        try:
            rel = path.relative_to(exp_path)
        except ValueError:
            continue
        per_step[step] = url_for("serve_asset", relative_path=f"{exp_id}/{rel}")
    return per_step


def _collect_step_map_from_dir(
    exp_id: str,
    exp_path: Path,
    folder: str,
    pattern: str,
    prefix: str,
) -> Dict[int, str]:
    target_dir = exp_path / folder
    if not target_dir.exists():
        return {}
    paths = sorted(target_dir.glob(pattern))
    return _build_step_map(paths, exp_id, exp_path, prefix=prefix)


def _resolve_asset_path(root: Path, relative_path: str) -> Path:
    root_path = root.resolve()
    raw_target = root_path / relative_path
    resolved = raw_target.resolve()
    parts = Path(relative_path).parts
    allowed_root = root_path
    if parts:
        exp_root = root_path / parts[0]
        if exp_root.is_symlink():
            allowed_root = exp_root.resolve()

    def _is_allowed(path: Path) -> bool:
        try:
            path.relative_to(root_path)
            return True
        except ValueError:
            pass
        if allowed_root != root_path:
            try:
                path.relative_to(allowed_root)
                return True
            except ValueError:
                return False
        return False

    if _is_allowed(resolved) and resolved.exists():
        return resolved
    if "graph_diagnostics_z" in resolved.parts:
        fallback = Path(
            *[("graph_diagnostics" if part == "graph_diagnostics_z" else part) for part in resolved.parts]
        )
        if _is_allowed(fallback) and fallback.exists():
            return fallback
    if not _is_allowed(resolved):
        abort(404)
    return resolved
