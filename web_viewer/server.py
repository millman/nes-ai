from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
import csv
from pathlib import Path
from typing import Dict, List, Optional
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
from .diffing import diff_metadata
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
    _diagnostics_exists,
    _diagnostics_s_exists,
    _graph_diagnostics_exists,
    _vis_ctrl_exists,
    _collect_visualization_steps,
    extract_alignment_summary,
)
from .plots import build_overlay, build_ranking_accuracy_plot

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_PAGE_SIZE = 25


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
                include_odometry=True,
                include_max_step=True,
                include_model_diff=True,
                include_model_diff_generation=True,
            )
            if exp is not None:
                starred_experiments.append(exp)
            _log_timing("dashboard.experiment", exp_start, exp_id=row.id)

        load_block_start = time.perf_counter()
        load_times = []
        for exp_id in selected_ids:
            exp_start = time.perf_counter()
            exp = load_experiment(
                cfg.output_dir / exp_id,
                include_odometry=True,
                include_max_step=True,
                include_model_diff=True,
                include_model_diff_generation=True,
            )
            if exp is not None:
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
            "experiments.html",
            experiments=experiments,
            cfg=cfg,
            active_nav="experiments",
            first_experiment_id=experiments[0].id if experiments else None,
        )

    @app.route("/comparison")
    def comparison():
        index_rows = build_experiment_index(cfg.output_dir)
        row_by_id = {row.id: row for row in index_rows}

        raw_ids = request.args.getlist("ids")
        if not raw_ids:
            ids_param = request.args.get("ids", "")
            raw_ids = ids_param.split(",") if ids_param else []

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
        return render_template(
            "comparison.html",
            experiments=experiments,
            cfg=cfg,
            selected_ids=selected_ids,
            selected_map=selected_map,
            active_nav="comparison",
            first_experiment_id=selected_ids[0] if selected_ids else (index_rows[0].id if index_rows else None),
        )

    @app.route("/experiment/<exp_id>")
    def experiment_detail(exp_id: str):
        experiment = _get_experiment_or_404(exp_id, include_rollout_steps=True)
        figure = _build_single_experiment_figure(experiment)
        viz_steps = _collect_visualization_steps(experiment.path)
        if experiment.rollout_steps:
            viz_steps.setdefault("vis_fixed_0", experiment.rollout_steps)
            viz_steps.setdefault("__fallback", experiment.rollout_steps)
        return render_template(
            "experiment_detail.html",
            experiment=experiment,
            cfg=cfg,
            figure=figure,
            visualization_steps=viz_steps,
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

        def _resolve_self_distance_csv_dir(exp_path: Path) -> Optional[Path]:
            """Support legacy self_distance outputs while preferring self_distance_z; error if both exist."""
            new_dir = exp_path / "self_distance_z"
            old_dir = exp_path / "self_distance"
            new_has = new_dir.exists() and any(new_dir.glob("self_distance_z_*.csv"))
            old_has = old_dir.exists() and any(old_dir.glob("self_distance_*.csv"))
            if new_has and old_has:
                raise RuntimeError(f"Found both self_distance_z and self_distance outputs in {exp_path}.")
            if new_has:
                return new_dir
            if old_has:
                return old_dir
            return None

        def _self_distance_z_image_conflict(exp_path: Path) -> None:
            """Prefer vis_self_distance_z images but fail if legacy+new images both exist."""
            new_dir = exp_path / "vis_self_distance_z"
            old_dir = exp_path / "vis_self_distance"
            new_has = new_dir.exists() and any(new_dir.glob("self_distance_z_*.png"))
            old_has = old_dir.exists() and any(old_dir.glob("self_distance_*.png"))
            if new_has and old_has:
                raise RuntimeError(f"Found both vis_self_distance_z and vis_self_distance images in {exp_path}.")

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
                csv_dir = _resolve_self_distance_csv_dir(row.path)
                if csv_dir is None:
                    continue
                if any(csv_dir.glob("self_distance_z_*.csv")) or any(csv_dir.glob("self_distance_*.csv")):
                    return row.id
            return None

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir():
                csv_dir = _resolve_self_distance_csv_dir(requested_path)
                if csv_dir is not None and (
                    any(csv_dir.glob("self_distance_z_*.csv")) or any(csv_dir.glob("self_distance_*.csv"))
                ):
                    selected_id = requested
        if selected_id is None:
            selected_id = _latest_self_distance_id()

        if selected_id is None:
            return render_template(
                "self_distance_page.html",
                experiments=[],
                experiment=None,
                page_title="Self-distance (Z)",
                cfg=cfg,
                active_nav="self_distance_z",
                active_experiment_id=None,
                first_experiment_id=None,
            )

        selected = load_experiment(
            cfg.output_dir / selected_id,
            include_self_distance=True,
        )
        if selected is None or selected.self_distance_csv is None:
            abort(404, "Experiment not found for self-distance.")
        _self_distance_z_image_conflict(selected.path)

        figure = _build_single_experiment_figure(selected)

        return render_template(
            "self_distance_page.html",
            experiments=[],
            experiment=selected,
            figure=figure,
            page_title="Self-distance (Z)",
            cfg=cfg,
            active_nav="self_distance_z",
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
        )

    @app.route("/self_distance_s", defaults={"exp_id": None})
    @app.route("/self_distance_s/<exp_id>")
    def self_distance_s(exp_id: Optional[str]):
        requested = exp_id or request.args.get("id")

        def _resolve_self_distance_s_csv_dir(exp_path: Path) -> Optional[Path]:
            """Support legacy state_embedding outputs while preferring self_distance_s; error if both exist."""
            new_dir = exp_path / "self_distance_s"
            old_dir = exp_path / "state_embedding"
            new_has = new_dir.exists() and any(new_dir.glob("self_distance_s_*.csv"))
            old_has = old_dir.exists() and any(old_dir.glob("state_embedding_*.csv"))
            if new_has and old_has:
                raise RuntimeError(f"Found both self_distance_s and state_embedding outputs in {exp_path}.")
            if new_has:
                return new_dir
            if old_has:
                return old_dir
            return None

        def _self_distance_s_image_conflict(exp_path: Path) -> None:
            """Avoid ambiguous self-distance (S) image sources between legacy and renamed folders."""
            old_dir = exp_path / "vis_state_embedding"
            new_dir = exp_path / "vis_self_distance_s"
            old_has = old_dir.exists() and any(old_dir.glob("state_embedding_[0-9]*.png"))
            new_has = new_dir.exists() and any(new_dir.glob("self_distance_s_*.png"))
            if old_has and new_has:
                raise RuntimeError(f"Found both vis_state_embedding and vis_self_distance_s images in {exp_path}.")

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
                csv_dir = _resolve_self_distance_s_csv_dir(row.path)
                if csv_dir is not None and (
                    any(csv_dir.glob("self_distance_s_*.csv")) or any(csv_dir.glob("state_embedding_*.csv"))
                ):
                    return row.id
            return None

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir():
                csv_dir = _resolve_self_distance_s_csv_dir(requested_path)
                if csv_dir is not None and (
                    any(csv_dir.glob("self_distance_s_*.csv")) or any(csv_dir.glob("state_embedding_*.csv"))
                ):
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
        _self_distance_s_image_conflict(selected.path)

        figure = _build_single_experiment_figure(selected)

        state_map: Dict[str, Dict[int, str]] = {"state_embedding": {}, "state_embedding_hist": {}}
        steps: List[int] = []
        for path in selected.state_embedding_images:
            stem = path.stem
            try:
                rel = path.relative_to(selected.path)
            except ValueError:
                continue
            if stem.startswith("state_embedding_hist_"):
                key = "state_embedding_hist"
                prefix = "state_embedding_hist_"
            elif stem.startswith("self_distance_s_") or stem.startswith("state_embedding_"):
                key = "state_embedding"
                prefix = "self_distance_s_" if stem.startswith("self_distance_s_") else "state_embedding_"
            elif stem.startswith("self_distance_cosine_") or stem.startswith("state_embedding_cosine_"):
                continue
            else:
                continue
            suffix = stem[len(prefix) :]
            try:
                step = int(suffix)
            except ValueError:
                continue
            state_map[key][step] = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            if key == "state_embedding":
                steps.append(step)
        steps = sorted(set(steps))

        return render_template(
            "state_embedding_page.html",
            experiments=experiment_ids,
            experiment=selected,
            figure=figure,
            state_embedding_map=state_map,
            state_embedding_steps=steps,
            page_title="Self-distance (S)",
            cfg=cfg,
            active_nav="self_distance_s",
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
        )

    @app.route("/self_distance_zvs", defaults={"exp_id": None})
    @app.route("/self_distance_zvs/<exp_id>")
    def self_distance_zvs(exp_id: Optional[str]):
        requested = exp_id or request.args.get("id")

        def _resolve_self_distance_z_csv_dir(exp_path: Path) -> Optional[Path]:
            new_dir = exp_path / "self_distance_z"
            old_dir = exp_path / "self_distance"
            new_has = new_dir.exists() and any(new_dir.glob("self_distance_z_*.csv"))
            old_has = old_dir.exists() and any(old_dir.glob("self_distance_*.csv"))
            if new_has and old_has:
                raise RuntimeError(f"Found both self_distance_z and self_distance outputs in {exp_path}.")
            if new_has:
                return new_dir
            if old_has:
                return old_dir
            return None

        def _resolve_self_distance_s_csv_dir(exp_path: Path) -> Optional[Path]:
            new_dir = exp_path / "self_distance_s"
            old_dir = exp_path / "state_embedding"
            new_has = new_dir.exists() and any(new_dir.glob("self_distance_s_*.csv"))
            old_has = old_dir.exists() and any(old_dir.glob("state_embedding_*.csv"))
            if new_has and old_has:
                raise RuntimeError(f"Found both self_distance_s and state_embedding outputs in {exp_path}.")
            if new_has:
                return new_dir
            if old_has:
                return old_dir
            return None

        def _self_distance_z_image_conflict(exp_path: Path) -> None:
            new_dir = exp_path / "vis_self_distance_z"
            old_dir = exp_path / "vis_self_distance"
            new_has = new_dir.exists() and any(new_dir.glob("self_distance_z_*.png"))
            old_has = old_dir.exists() and any(old_dir.glob("self_distance_*.png"))
            if new_has and old_has:
                raise RuntimeError(f"Found both vis_self_distance_z and vis_self_distance images in {exp_path}.")

        def _self_distance_s_image_conflict(exp_path: Path) -> None:
            old_dir = exp_path / "vis_state_embedding"
            new_dir = exp_path / "vis_self_distance_s"
            old_has = old_dir.exists() and any(old_dir.glob("state_embedding_[0-9]*.png"))
            new_has = new_dir.exists() and any(new_dir.glob("self_distance_s_*.png"))
            if old_has and new_has:
                raise RuntimeError(f"Found both vis_state_embedding and vis_self_distance_s images in {exp_path}.")

        def _latest_self_distance_zvs_id() -> Optional[str]:
            index_rows = build_experiment_index(cfg.output_dir)
            if not index_rows:
                return None
            sorted_rows = sorted(
                index_rows,
                key=lambda row: row.last_modified or datetime.fromtimestamp(0),
                reverse=True,
            )
            for row in sorted_rows:
                z_dir = _resolve_self_distance_z_csv_dir(row.path)
                s_dir = _resolve_self_distance_s_csv_dir(row.path)
                if z_dir is None or s_dir is None:
                    continue
                if not (any(z_dir.glob("self_distance_z_*.csv")) or any(z_dir.glob("self_distance_*.csv"))):
                    continue
                if not (any(s_dir.glob("self_distance_s_*.csv")) or any(s_dir.glob("state_embedding_*.csv"))):
                    continue
                return row.id
            return None

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir():
                z_dir = _resolve_self_distance_z_csv_dir(requested_path)
                s_dir = _resolve_self_distance_s_csv_dir(requested_path)
                if z_dir is not None and s_dir is not None:
                    selected_id = requested
        if selected_id is None:
            selected_id = _latest_self_distance_zvs_id()

        if selected_id is None:
            return render_template(
                "self_distance_zs.html",
                experiments=[],
                experiment=None,
                page_title="Self-distance (Z vs S)",
                cfg=cfg,
                active_nav="self_distance_zvs",
                active_experiment_id=None,
                first_experiment_id=None,
            )

        selected = load_experiment(
            cfg.output_dir / selected_id,
            include_self_distance=True,
            include_state_embedding=True,
        )
        if selected is None or selected.self_distance_csv is None or selected.state_embedding_csv is None:
            abort(404, "Experiment not found for self-distance (Z vs S).")
        _self_distance_z_image_conflict(selected.path)
        _self_distance_s_image_conflict(selected.path)

        figure = _build_single_experiment_figure(selected)

        return render_template(
            "self_distance_zs.html",
            experiments=[],
            experiment=selected,
            figure=figure,
            page_title="Self-distance (Z vs S)",
            cfg=cfg,
            active_nav="self_distance_zvs",
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
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
                elif stem.startswith("z_vs_z_hat_"):
                    label = "z_vs_z_hat"
                    prefix = "z_vs_z_hat_"
                elif stem.startswith("s_vs_s_hat_"):
                    label = "s_vs_s_hat"
                    prefix = "s_vs_s_hat_"
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
            index_rows = build_experiment_index(cfg.output_dir)
            if not index_rows:
                return None
            sorted_rows = sorted(
                index_rows,
                key=lambda row: row.last_modified or datetime.fromtimestamp(0),
                reverse=True,
            )
            for row in sorted_rows:
                if _diagnostics_exists(row.path):
                    return row.id
            return None

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir() and _diagnostics_exists(requested_path):
                selected_id = requested
        if selected_id is None:
            selected_id = _latest_diagnostics_id()

        if selected_id is None:
            return render_template(
            "diagnostics_page.html",
            experiments=[],
            experiment=None,
            diagnostics_map={},
                diagnostics_csv_map={},
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
        diagnostics_map: Dict[str, Dict[int, str]] = {}
        for name, paths in selected.diagnostics_images.items():
            per_step: Dict[int, str] = {}
            for path in paths:
                stem = path.stem
                suffix = stem.split("_")[-1] if "_" in stem else stem
                try:
                    step = int(suffix)
                except ValueError:
                    continue
                rel = path.relative_to(selected.path)
                per_step[step] = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            if per_step:
                diagnostics_map[name] = per_step

        # Add self-distance images keyed by step (matching self_distance_page)
        if selected.self_distance_images:
            per_step: Dict[int, str] = {}
            for path in selected.self_distance_images:
                stem = path.stem
                if "cosine" in stem:
                    continue
                suffix = stem.split("_")[-1] if "_" in stem else stem
                try:
                    step = int(suffix)
                except ValueError:
                    continue
                rel = path.relative_to(selected.path)
                per_step[step] = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            if per_step:
                diagnostics_map["self_distance"] = per_step

        # Add state embedding self-distance images (S) keyed by step.
        if selected.state_embedding_images:
            per_step = {}
            for path in selected.state_embedding_images:
                stem = path.stem
                if "hist" in stem or "cosine" in stem:
                    continue
                suffix = stem.split("_")[-1] if "_" in stem else stem
                try:
                    step = int(suffix)
                except ValueError:
                    continue
                rel = path.relative_to(selected.path)
                per_step[step] = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            if per_step:
                diagnostics_map["self_distance_s"] = per_step

        diagnostics_csv_map: Dict[str, Dict[int, str]] = {}
        for name, paths in selected.diagnostics_csvs.items():
            per_step: Dict[int, str] = {}
            for path in paths:
                stem = path.stem
                suffix = stem.split("_")[-1] if "_" in stem else stem
                try:
                    step = int(suffix)
                except ValueError:
                    continue
                rel = path.relative_to(selected.path)
                per_step[step] = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            if per_step:
                diagnostics_csv_map[name] = per_step

        frame_map: Dict[int, List[Dict[str, str]]] = {}
        for step, frames in selected.diagnostics_frames.items():
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
            experiments=[],
            experiment=selected,
            diagnostics_map=diagnostics_map,
            diagnostics_csv_map=diagnostics_csv_map,
            frame_map=frame_map,
            figure=figure,
            diagnostics_summary=alignment_summary,
            cfg=cfg,
            active_nav="diagnostics",
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
        )

    @app.route("/diagnostics_s", defaults={"exp_id": None})
    @app.route("/diagnostics_s/<exp_id>")
    def diagnostics_s(exp_id: Optional[str]):
        route_start = time.perf_counter()
        requested = exp_id or request.args.get("id")

        def _latest_diagnostics_s_id() -> Optional[str]:
            index_rows = build_experiment_index(cfg.output_dir)
            if not index_rows:
                return None
            sorted_rows = sorted(
                index_rows,
                key=lambda row: row.last_modified or datetime.fromtimestamp(0),
                reverse=True,
            )
            for row in sorted_rows:
                if _diagnostics_s_exists(row.path):
                    return row.id
            return sorted_rows[0].id if sorted_rows else None

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir():
                selected_id = requested
        if selected_id is None:
            selected_id = _latest_diagnostics_s_id()

        if selected_id is None:
            return render_template(
                "diagnostics_page_s.html",
                experiments=[],
                experiment=None,
                diagnostics_map={},
                diagnostics_csv_map={},
                frame_map={},
                cfg=cfg,
                active_nav="diagnostics_s",
                active_experiment_id=None,
                first_experiment_id=None,
            )

        selected = load_experiment(
            cfg.output_dir / selected_id,
            include_diagnostics_frames=True,
            include_diagnostics_s=True,
            include_state_embedding=True,
        )
        if selected is None:
            abort(404, "Experiment not found for diagnostics (S).")

        build_maps_start = time.perf_counter()
        figure = _build_single_experiment_figure(selected)
        diagnostics_map: Dict[str, Dict[int, str]] = {}
        for name, paths in selected.diagnostics_s_images.items():
            per_step: Dict[int, str] = {}
            for path in paths:
                stem = path.stem
                suffix = stem.split("_")[-1] if "_" in stem else stem
                try:
                    step = int(suffix)
                except ValueError:
                    continue
                rel = path.relative_to(selected.path)
                per_step[step] = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            if per_step:
                diagnostics_map[name] = per_step

        # Add state embedding self-distance images (S) keyed by step.
        if selected.state_embedding_images:
            per_step = {}
            for path in selected.state_embedding_images:
                stem = path.stem
                if "hist" in stem or "cosine" in stem:
                    continue
                suffix = stem.split("_")[-1] if "_" in stem else stem
                try:
                    step = int(suffix)
                except ValueError:
                    continue
                rel = path.relative_to(selected.path)
                per_step[step] = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            if per_step:
                diagnostics_map["self_distance_s"] = per_step

        diagnostics_csv_map: Dict[str, Dict[int, str]] = {}
        for name, paths in selected.diagnostics_s_csvs.items():
            per_step: Dict[int, str] = {}
            for path in paths:
                stem = path.stem
                suffix = stem.split("_")[-1] if "_" in stem else stem
                try:
                    step = int(suffix)
                except ValueError:
                    continue
                rel = path.relative_to(selected.path)
                per_step[step] = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            if per_step:
                diagnostics_csv_map[name] = per_step

        frame_map: Dict[int, List[Dict[str, str]]] = {}
        for step, frames in selected.diagnostics_frames.items():
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
            "diagnostics_s.build_maps",
            build_maps_start,
            images=sum(len(v) for v in diagnostics_map.values()),
            csvs=sum(len(v) for v in diagnostics_csv_map.values()),
            frames=sum(len(v) for v in frame_map.values()),
        )
        _log_timing("diagnostics_s.total", route_start, selected=selected.id)
        return render_template(
            "diagnostics_page_s.html",
            experiments=[],
            experiment=selected,
            diagnostics_map=diagnostics_map,
            diagnostics_csv_map=diagnostics_csv_map,
            frame_map=frame_map,
            figure=figure,
            cfg=cfg,
            active_nav="diagnostics_s",
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
        )

    @app.route("/diagnostics_zvs", defaults={"exp_id": None})
    @app.route("/diagnostics_zvs/<exp_id>")
    def diagnostics_zvs(exp_id: Optional[str]):
        route_start = time.perf_counter()
        requested = exp_id or request.args.get("id")

        def _latest_diagnostics_zvs_id() -> Optional[str]:
            index_rows = build_experiment_index(cfg.output_dir)
            if not index_rows:
                return None
            sorted_rows = sorted(
                index_rows,
                key=lambda row: row.last_modified or datetime.fromtimestamp(0),
                reverse=True,
            )
            fallback = None
            for row in sorted_rows:
                has_z = _diagnostics_exists(row.path)
                has_s = _diagnostics_s_exists(row.path)
                if has_z and has_s:
                    return row.id
                if fallback is None and (has_z or has_s):
                    fallback = row.id
            return fallback

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir():
                selected_id = requested
        if selected_id is None:
            selected_id = _latest_diagnostics_zvs_id()

        if selected_id is None:
            return render_template(
                "diagnostics_page_zs.html",
                experiments=[],
                experiment=None,
                diagnostics_map_z={},
                diagnostics_map_s={},
                diagnostic_steps=[],
                figure=None,
                cfg=cfg,
                active_nav="diagnostics_zvs",
                active_experiment_id=None,
                first_experiment_id=None,
            )

        selected = load_experiment(
            cfg.output_dir / selected_id,
            include_self_distance=True,
            include_diagnostics_images=True,
            include_diagnostics_s=True,
            include_state_embedding=True,
        )
        if selected is None:
            abort(404, "Experiment not found for diagnostics.")

        build_maps_start = time.perf_counter()
        figure = _build_single_experiment_figure(selected)

        diagnostics_map_z: Dict[str, Dict[int, str]] = {}
        for name, paths in selected.diagnostics_images.items():
            per_step: Dict[int, str] = {}
            for path in paths:
                stem = path.stem
                suffix = stem.split("_")[-1] if "_" in stem else stem
                try:
                    step = int(suffix)
                except ValueError:
                    continue
                rel = path.relative_to(selected.path)
                per_step[step] = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            if per_step:
                diagnostics_map_z[name] = per_step

        if selected.self_distance_images:
            per_step = {}
            for path in selected.self_distance_images:
                stem = path.stem
                if "cosine" in stem:
                    continue
                suffix = stem.split("_")[-1] if "_" in stem else stem
                try:
                    step = int(suffix)
                except ValueError:
                    continue
                rel = path.relative_to(selected.path)
                per_step[step] = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            if per_step:
                diagnostics_map_z["self_distance"] = per_step

        diagnostics_map_s: Dict[str, Dict[int, str]] = {}
        for name, paths in selected.diagnostics_s_images.items():
            per_step = {}
            for path in paths:
                stem = path.stem
                suffix = stem.split("_")[-1] if "_" in stem else stem
                try:
                    step = int(suffix)
                except ValueError:
                    continue
                rel = path.relative_to(selected.path)
                per_step[step] = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            if per_step:
                diagnostics_map_s[name] = per_step

        if selected.state_embedding_images:
            per_step = {}
            for path in selected.state_embedding_images:
                stem = path.stem
                if "hist" in stem or "cosine" in stem:
                    continue
                suffix = stem.split("_")[-1] if "_" in stem else stem
                try:
                    step = int(suffix)
                except ValueError:
                    continue
                rel = path.relative_to(selected.path)
                per_step[step] = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            if per_step:
                diagnostics_map_s["self_distance_s"] = per_step

        step_set: set[int] = set()
        for per_step in diagnostics_map_z.values():
            step_set.update(per_step.keys())
        for per_step in diagnostics_map_s.values():
            step_set.update(per_step.keys())
        diagnostic_steps = sorted(step_set)

        _log_timing(
            "diagnostics_zvs.build_maps",
            build_maps_start,
            images=sum(len(v) for v in diagnostics_map_z.values()) + sum(len(v) for v in diagnostics_map_s.values()),
        )
        _log_timing("diagnostics_zvs.total", route_start, selected=selected.id)
        return render_template(
            "diagnostics_page_zs.html",
            experiments=[],
            experiment=selected,
            diagnostics_map_z=diagnostics_map_z,
            diagnostics_map_s=diagnostics_map_s,
            diagnostic_steps=diagnostic_steps,
            figure=figure,
            cfg=cfg,
            active_nav="diagnostics_zvs",
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
        )

    @app.route("/debug_z", defaults={"exp_id": None})
    @app.route("/debug_z/<exp_id>")
    def debug_z(exp_id: Optional[str]):
        requested = exp_id or request.args.get("id")
        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir():
                selected_id = requested
        if selected_id is None:
            index_rows = build_experiment_index(cfg.output_dir)
            selected_id = index_rows[0].id if index_rows else None
        if selected_id is None:
            return render_template(
                "debug_z.html",
                experiment=None,
                debug_map={},
                steps=[],
                cfg=cfg,
                active_nav="debug_z",
                active_experiment_id=None,
                first_experiment_id=None,
            )

        selected = load_experiment(
            cfg.output_dir / selected_id,
            include_self_distance=True,
            include_diagnostics_images=True,
            include_vis_ctrl=True,
        )
        if selected is None:
            abort(404, "Experiment not found for debug z.")

        debug_map: Dict[str, Dict[int, str]] = {}
        debug_map["inputs"] = _collect_step_map_from_dir(
            selected.id, selected.path, "vis_inputs", "inputs_*.png", "inputs_"
        )
        debug_map["pairs"] = _collect_step_map_from_dir(
            selected.id, selected.path, "vis_pairs", "pairs_*.png", "pairs_"
        )
        debug_map["pca_z"] = _collect_step_map_from_dir(
            selected.id, selected.path, "pca_z", "pca_z_*.png", "pca_z_"
        )
        debug_map["samples_hard"] = _collect_step_map_from_dir(
            selected.id, selected.path, "samples_hard", "hard_*.png", "hard_"
        )
        for name, paths in selected.diagnostics_images.items():
            debug_map[name] = _build_step_map(paths, selected.id, selected.path)
        for name, paths in selected.vis_ctrl_images.items():
            if name not in {"alignment_z", "smoothness_z", "composition_z", "stability_z"}:
                continue
            debug_map[name] = _build_step_map(paths, selected.id, selected.path)

        if selected.self_distance_images:
            filtered = [p for p in selected.self_distance_images if "cosine" not in p.stem]
            debug_map["self_distance"] = _build_step_map(filtered, selected.id, selected.path)

        steps = sorted({step for per_map in debug_map.values() for step in per_map.keys()})
        figure = _build_single_experiment_figure(selected)
        return render_template(
            "debug_z.html",
            experiment=selected,
            debug_map=debug_map,
            steps=steps,
            figure=figure,
            cfg=cfg,
            active_nav="debug_z",
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
        )

    @app.route("/debug_h", defaults={"exp_id": None})
    @app.route("/debug_h/<exp_id>")
    def debug_h(exp_id: Optional[str]):
        requested = exp_id or request.args.get("id")
        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir():
                selected_id = requested
        if selected_id is None:
            index_rows = build_experiment_index(cfg.output_dir)
            selected_id = index_rows[0].id if index_rows else None
        if selected_id is None:
            return render_template(
                "debug_h.html",
                experiment=None,
                debug_map={},
                steps=[],
                cfg=cfg,
                active_nav="debug_h",
                active_experiment_id=None,
                first_experiment_id=None,
            )

        selected = load_experiment(
            cfg.output_dir / selected_id,
            include_vis_ctrl=True,
        )
        if selected is None:
            abort(404, "Experiment not found for debug h.")

        debug_map: Dict[str, Dict[int, str]] = {}
        for name, paths in selected.vis_ctrl_images.items():
            if name not in {"alignment_h", "smoothness_h", "composition_h", "stability_h"}:
                continue
            debug_map[name] = _build_step_map(paths, selected.id, selected.path)

        steps = sorted({step for per_map in debug_map.values() for step in per_map.keys()})
        figure = _build_single_experiment_figure(selected)
        return render_template(
            "debug_h.html",
            experiment=selected,
            debug_map=debug_map,
            steps=steps,
            figure=figure,
            cfg=cfg,
            active_nav="debug_h",
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
        )

    @app.route("/debug_s", defaults={"exp_id": None})
    @app.route("/debug_s/<exp_id>")
    def debug_s(exp_id: Optional[str]):
        requested = exp_id or request.args.get("id")
        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir():
                selected_id = requested
        if selected_id is None:
            index_rows = build_experiment_index(cfg.output_dir)
            selected_id = index_rows[0].id if index_rows else None
        if selected_id is None:
            return render_template(
                "debug_s.html",
                experiment=None,
                debug_map={},
                steps=[],
                ranking_figure=None,
                cfg=cfg,
                active_nav="debug_s",
                active_experiment_id=None,
                first_experiment_id=None,
            )

        selected = load_experiment(
            cfg.output_dir / selected_id,
            include_state_embedding=True,
            include_diagnostics_s=True,
            include_vis_ctrl=True,
        )
        if selected is None:
            abort(404, "Experiment not found for debug s.")

        debug_map: Dict[str, Dict[int, str]] = {}
        debug_map["pca_s"] = _collect_step_map_from_dir(
            selected.id, selected.path, "pca_s", "pca_s_*.png", "pca_s_"
        )
        for name, paths in selected.diagnostics_s_images.items():
            debug_map[name] = _build_step_map(paths, selected.id, selected.path)
        for name, paths in selected.vis_ctrl_images.items():
            if name not in {"alignment_s", "smoothness_s", "composition_s", "stability_s"}:
                continue
            debug_map[name] = _build_step_map(paths, selected.id, selected.path)

        if selected.state_embedding_images:
            filtered = [p for p in selected.state_embedding_images if "hist" not in p.stem and "cosine" not in p.stem]
            debug_map["self_distance_s"] = _build_step_map(filtered, selected.id, selected.path)

        curves = load_loss_curves(selected.loss_csv) if selected.loss_csv else None
        ranking_figure = build_ranking_accuracy_plot(curves) if curves else None

        steps = sorted({step for per_map in debug_map.values() for step in per_map.keys()})
        figure = _build_single_experiment_figure(selected)
        return render_template(
            "debug_s.html",
            experiment=selected,
            debug_map=debug_map,
            steps=steps,
            ranking_figure=ranking_figure,
            figure=figure,
            cfg=cfg,
            active_nav="debug_s",
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
                if _graph_diagnostics_exists(row.path):
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

        graph_map: Dict[str, Dict[int, str]] = {}
        for name, paths in selected.graph_diagnostics_images.items():
            per_step: Dict[int, str] = {}
            for path in paths:
                stem = path.stem
                suffix = stem.split("_")[-1] if "_" in stem else stem
                try:
                    step = int(suffix)
                except ValueError:
                    continue
                try:
                    rel = path.relative_to(selected.path)
                except ValueError:
                    continue
                per_step[step] = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            if per_step:
                graph_map[name] = per_step

        graph_steps = selected.graph_diagnostics_steps
        history_rows: List[Dict[str, float]] = []
        history_url: Optional[str] = None
        history_plot_url: Optional[str] = None
        history_files = selected.graph_diagnostics_csvs.get("metrics_history", [])
        if history_files:
            latest_hist = history_files[-1]
            history_rows = _parse_graph_history_csv(latest_hist)
            try:
                rel = latest_hist.relative_to(selected.path)
                history_url = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            except ValueError:
                history_url = None
        history_images = selected.graph_diagnostics_images.get("metrics_history", [])
        latest_history_images = selected.graph_diagnostics_images.get("metrics_history_latest", [])
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
                if _graph_diagnostics_exists(row.path, folder_name="graph_diagnostics_s"):
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
                page_title="Graph Diagnostics (S)",
                embedding_label="s",
                cfg=cfg,
                active_nav="graph_diagnostics_s",
                active_experiment_id=None,
                first_experiment_id=None,
            )

        selected = load_experiment(
            cfg.output_dir / selected_id,
            include_graph_diagnostics_s=True,
        )
        if selected is None:
            abort(404, "Experiment not found for graph diagnostics.")

        figure = _build_single_experiment_figure(selected)

        graph_map: Dict[str, Dict[int, str]] = {}
        for name, paths in selected.graph_diagnostics_s_images.items():
            per_step: Dict[int, str] = {}
            for path in paths:
                stem = path.stem
                suffix = stem.split("_")[-1] if "_" in stem else stem
                try:
                    step = int(suffix)
                except ValueError:
                    continue
                try:
                    rel = path.relative_to(selected.path)
                except ValueError:
                    continue
                per_step[step] = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            if per_step:
                graph_map[name] = per_step

        graph_steps = selected.graph_diagnostics_s_steps
        history_rows: List[Dict[str, float]] = []
        history_url: Optional[str] = None
        history_plot_url: Optional[str] = None
        history_files = selected.graph_diagnostics_s_csvs.get("metrics_history", [])
        if history_files:
            latest_hist = history_files[-1]
            history_rows = _parse_graph_history_csv(latest_hist)
            try:
                rel = latest_hist.relative_to(selected.path)
                history_url = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            except ValueError:
                history_url = None
        history_images = selected.graph_diagnostics_s_images.get("metrics_history", [])
        latest_history_images = selected.graph_diagnostics_s_images.get("metrics_history_latest", [])
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
            "graph_diagnostics_s.total",
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
            page_title="Graph Diagnostics (S)",
            embedding_label="s",
            cfg=cfg,
            active_nav="graph_diagnostics_s",
            active_experiment_id=selected.id,
            first_experiment_id=selected.id,
        )

    @app.route("/graph_diagnostics_zvs", defaults={"exp_id": None})
    @app.route("/graph_diagnostics_zvs/<exp_id>")
    def graph_diagnostics_zvs(exp_id: Optional[str]):
        route_start = time.perf_counter()
        requested = exp_id or request.args.get("id")

        def _latest_graph_zvs_id() -> Optional[str]:
            index_rows = build_experiment_index(cfg.output_dir)
            if not index_rows:
                return None
            sorted_rows = sorted(
                index_rows,
                key=lambda row: row.last_modified or datetime.fromtimestamp(0),
                reverse=True,
            )
            for row in sorted_rows:
                if _graph_diagnostics_exists(row.path) and _graph_diagnostics_exists(row.path, folder_name="graph_diagnostics_s"):
                    return row.id
            return None

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir():
                if _graph_diagnostics_exists(requested_path) and _graph_diagnostics_exists(requested_path, folder_name="graph_diagnostics_s"):
                    selected_id = requested
        if selected_id is None:
            selected_id = _latest_graph_zvs_id()

        if selected_id is None:
            return render_template(
                "graph_diagnostics_zs.html",
                experiments=[],
                experiment=None,
                graph_map_z={},
                graph_map_s={},
                steps_z=[],
                steps_s=[],
                history_z=[],
                history_s=[],
                history_url_z=None,
                history_url_s=None,
                history_plot_url_z=None,
                history_plot_url_s=None,
                figure=None,
                cfg=cfg,
                active_nav="graph_diagnostics_zvs",
                active_experiment_id=None,
                first_experiment_id=None,
            )

        selected = load_experiment(
            cfg.output_dir / selected_id,
            include_graph_diagnostics=True,
            include_graph_diagnostics_s=True,
        )
        if selected is None:
            abort(404, "Experiment not found for graph diagnostics (Z vs S).")

        figure = _build_single_experiment_figure(selected)

        def _build_graph_map(paths_by_name: Dict[str, List[Path]]) -> Dict[str, Dict[int, str]]:
            out: Dict[str, Dict[int, str]] = {}
            for name, paths in paths_by_name.items():
                per_step: Dict[int, str] = {}
                for path in paths:
                    stem = path.stem
                    suffix = stem.split("_")[-1] if "_" in stem else stem
                    try:
                        step = int(suffix)
                    except ValueError:
                        continue
                    try:
                        rel = path.relative_to(selected.path)
                    except ValueError:
                        continue
                    per_step[step] = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
                if per_step:
                    out[name] = per_step
            return out

        graph_map_z = _build_graph_map(selected.graph_diagnostics_images)
        graph_map_s = _build_graph_map(selected.graph_diagnostics_s_images)
        steps_z = selected.graph_diagnostics_steps
        steps_s = selected.graph_diagnostics_s_steps

        history_rows_z: List[Dict[str, float]] = []
        history_rows_s: List[Dict[str, float]] = []
        history_url_z: Optional[str] = None
        history_url_s: Optional[str] = None
        history_plot_url_z: Optional[str] = None
        history_plot_url_s: Optional[str] = None

        history_files_z = selected.graph_diagnostics_csvs.get("metrics_history", [])
        if history_files_z:
            latest_hist = history_files_z[-1]
            history_rows_z = _parse_graph_history_csv(latest_hist)
            try:
                rel = latest_hist.relative_to(selected.path)
                history_url_z = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            except ValueError:
                history_url_z = None

        history_files_s = selected.graph_diagnostics_s_csvs.get("metrics_history", [])
        if history_files_s:
            latest_hist = history_files_s[-1]
            history_rows_s = _parse_graph_history_csv(latest_hist)
            try:
                rel = latest_hist.relative_to(selected.path)
                history_url_s = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
            except ValueError:
                history_url_s = None

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

        history_plot_url_z = _latest_history_plot(selected.graph_diagnostics_images)
        history_plot_url_s = _latest_history_plot(selected.graph_diagnostics_s_images)

        _log_timing(
            "graph_diagnostics_zvs.total",
            route_start,
            selected=selected.id,
            images=sum(len(v) for v in graph_map_z.values()) + sum(len(v) for v in graph_map_s.values()),
            steps=len(steps_z) + len(steps_s),
            history=len(history_rows_z) + len(history_rows_s),
        )
        return render_template(
            "graph_diagnostics_zs.html",
            experiments=[],
            experiment=selected,
            graph_map_z=graph_map_z,
            graph_map_s=graph_map_s,
            steps_z=steps_z,
            steps_s=steps_s,
            history_z=history_rows_z,
            history_s=history_rows_s,
            history_url_z=history_url_z,
            history_url_s=history_url_s,
            history_plot_url_z=history_plot_url_z,
            history_plot_url_s=history_plot_url_s,
            figure=figure,
            cfg=cfg,
            active_nav="graph_diagnostics_zvs",
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
                if _vis_ctrl_exists(row.path):
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
                vis_ctrl_map_s={},
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
            paths_by_name: Dict[str, List[Path]]
        ) -> Tuple[Dict[str, Dict[int, str]], Dict[str, Dict[int, str]], Dict[str, Dict[int, str]]]:
            map_z: Dict[str, Dict[int, str]] = {}
            map_s: Dict[str, Dict[int, str]] = {}
            map_h: Dict[str, Dict[int, str]] = {}
            for name, paths in paths_by_name.items():
                target = (
                    map_z if name.endswith("_z") else map_s if name.endswith("_s") else map_h if name.endswith("_h") else None
                )
                if target is None:
                    continue
                per_step: Dict[int, str] = {}
                for path in paths:
                    stem = path.stem
                    suffix = stem.split("_")[-1] if "_" in stem else stem
                    try:
                        step = int(suffix)
                    except ValueError:
                        continue
                    try:
                        rel = path.relative_to(selected.path)
                    except ValueError:
                        continue
                    per_step[step] = url_for("serve_asset", relative_path=f"{selected.id}/{rel}")
                if per_step:
                    target[name] = per_step
            return map_z, map_s, map_h

        vis_ctrl_map_z, vis_ctrl_map_s, vis_ctrl_map_h = _build_vis_ctrl_map(selected.vis_ctrl_images)
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
            vis_ctrl_map_s=vis_ctrl_map_s,
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
    base_metadata = experiments[0].metadata_text
    rows = []
    for idx, exp in enumerate(experiments):
        diff_text = None
        if idx > 0:
            diff_text = diff_metadata(base_metadata, exp.metadata_text)
        viz_steps = _collect_visualization_steps(exp.path)
        rows.append(
            {
                "id": exp.id,
                "name": exp.name,
                "git_commit": exp.git_commit,
                "title": exp.title,
                "tags": exp.tags,
                "rollout_steps": exp.rollout_steps,
                "visualization_steps": viz_steps,
                "metadata": exp.metadata_text,
                "metadata_diff": diff_text,
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
) -> Dict[int, str]:
    per_step: Dict[int, str] = {}
    for path in paths:
        step = _parse_step_suffix(path.stem, prefix)
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
    target = (root_path / relative_path).resolve()
    try:
        target.relative_to(root_path)
    except ValueError:
        abort(404)
    if target.exists():
        return target
    if "graph_diagnostics_z" in target.parts:
        fallback_rel = Path(*[("graph_diagnostics" if part == "graph_diagnostics_z" else part) for part in target.parts])
        try:
            fallback_rel = fallback_rel.relative_to(root_path)
        except ValueError:
            return target
        fallback = (root_path / fallback_rel).resolve()
        try:
            fallback.relative_to(root_path)
        except ValueError:
            return target
        if fallback.exists():
            return fallback
    return target
