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
    _diagnostics_exists,
    _graph_diagnostics_exists,
    _collect_visualization_steps,
    extract_alignment_summary,
)
from .plots import build_overlay

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

    def _load_all() -> List[Experiment]:
        return list_experiments(cfg.output_dir)

    def _get_experiment_or_404(exp_id: str) -> Experiment:
        exp_path = cfg.output_dir / exp_id
        experiment = load_experiment(exp_path)
        if experiment is None:
            abort(404, f"Experiment {exp_id} not found.")
        return experiment

    @app.route("/")
    def dashboard():
        route_start = time.perf_counter()
        experiments = []
        index_start = time.perf_counter()
        index_rows = build_experiment_index(cfg.output_dir)
        _log_timing("dashboard.index", index_start, rows=len(index_rows))
        total_items = len(index_rows)
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
        selected_ids: List[str] = [row.id for row in index_rows[start_idx:end_idx]]

        load_block_start = time.perf_counter()
        load_times = []
        for exp_id in selected_ids:
            exp_start = time.perf_counter()
            exp = load_experiment(
                cfg.output_dir / exp_id,
                include_self_distance=False,
                include_diagnostics_images=False,
                include_diagnostics_frames=False,
                include_graph_diagnostics=False,
                include_last_modified=False,
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
            cfg=cfg,
            total_pages=total_pages,
            current_page=current_page,
            page_size=page_size,
            active_nav="dashboard",
            first_experiment_id=index_rows[0].id if index_rows else None,
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
                include_self_distance=False,
                include_diagnostics_images=False,
                include_diagnostics_frames=False,
                include_graph_diagnostics=False,
                include_last_modified=False,
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

    @app.route("/experiments/<exp_id>")
    def experiment_detail(exp_id: str):
        experiment = _get_experiment_or_404(exp_id)
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
                include_self_distance=False,
                include_diagnostics_images=False,
                include_diagnostics_frames=False,
                include_graph_diagnostics=False,
                include_last_modified=False,
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

    @app.route("/self_distance", defaults={"exp_id": None})
    @app.route("/self_distance/<exp_id>")
    def self_distance(exp_id: Optional[str]):
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
                csv_path = row.path / "self_distance" / "self_distance_000000.csv"
                if csv_path.exists():
                    return row.id
            return None

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir():
                csv_dir = requested_path / "self_distance"
                if csv_dir.exists() and any(csv_dir.glob("self_distance_*.csv")):
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
            include_diagnostics_images=False,
            include_diagnostics_frames=False,
            include_graph_diagnostics=False,
        )
        if selected is None or selected.self_distance_csv is None:
            abort(404, "Experiment not found for self-distance.")

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

    @app.route("/state_embedding", defaults={"exp_id": None})
    @app.route("/state_embedding/<exp_id>")
    def state_embedding(exp_id: Optional[str]):
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
                csv_dir = row.path / "state_embedding"
                if csv_dir.exists() and any(csv_dir.glob("state_embedding_*.csv")):
                    return row.id
            return None

        selected_id: Optional[str] = None
        if requested:
            requested_path = cfg.output_dir / requested
            if requested_path.is_dir():
                csv_dir = requested_path / "state_embedding"
                if csv_dir.exists() and any(csv_dir.glob("state_embedding_*.csv")):
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
            include_self_distance=False,
            include_diagnostics_images=False,
            include_diagnostics_frames=False,
            include_graph_diagnostics=False,
        )
        if selected is None or selected.state_embedding_csv is None:
            abort(404, "Experiment not found for state embedding diagnostics.")

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
            elif stem.startswith("state_embedding_"):
                key = "state_embedding"
                prefix = "state_embedding_"
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

    @app.route("/diagnostics", defaults={"exp_id": None})
    @app.route("/diagnostics/<exp_id>")
    def diagnostics(exp_id: Optional[str]):
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

        selected = load_experiment(cfg.output_dir / selected_id, include_graph_diagnostics=False)
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
                if "hist" in stem:
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

    @app.route("/graph_diagnostics", defaults={"exp_id": None})
    @app.route("/graph_diagnostics/<exp_id>")
    def graph_diagnostics(exp_id: Optional[str]):
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
            include_self_distance=False,
            include_diagnostics_images=False,
            include_diagnostics_frames=False,
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
            include_self_distance=False,
            include_diagnostics_images=False,
            include_diagnostics_frames=False,
            include_graph_diagnostics=False,
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


def _resolve_asset_path(root: Path, relative_path: str) -> Path:
    root_path = root.resolve()
    target = (root_path / relative_path).resolve()
    try:
        target.relative_to(root_path)
    except ValueError:
        abort(404)
    return target
