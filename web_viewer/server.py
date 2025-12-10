from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
    LossCurveData,
    list_experiments,
    load_experiment,
    load_loss_curves,
    write_notes,
    write_title,
)
from .plots import build_overlay

PACKAGE_ROOT = Path(__file__).resolve().parent


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
    app = Flask(
        __name__,
        template_folder=str(PACKAGE_ROOT / "templates"),
        static_folder=str(PACKAGE_ROOT / "static"),
        static_url_path="/static/web_viewer",
    )
    app.config["VIEWER_CONFIG"] = cfg

    # Register custom Jinja filters
    app.jinja_env.filters["format_last_modified"] = _format_last_modified

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
        experiments = _load_all()
        return render_template(
            "dashboard.html",
            experiments=experiments,
            cfg=cfg,
            active_nav="dashboard",
        )

    @app.route("/experiments")
    def experiments_index():
        experiments = _load_all()
        return render_template(
            "experiments.html",
            experiments=experiments,
            cfg=cfg,
            active_nav="experiments",
        )

    @app.route("/comparison")
    def comparison():
        experiments = _load_all()
        raw_ids = request.args.getlist("ids")
        if not raw_ids:
            ids_param = request.args.get("ids", "")
            raw_ids = ids_param.split(",") if ids_param else []
        selected_ids = [exp_id for exp_id in raw_ids if exp_id]
        if len(selected_ids) < 2 and len(experiments) >= 2:
            selected_ids = [exp.id for exp in experiments[:2]]
        selected_map = {exp.id: exp for exp in experiments if exp.id in selected_ids}
        # Sort selected_ids by last_modified date (most recent first)
        selected_ids = sorted(
            selected_ids,
            key=lambda eid: selected_map[eid].last_modified or datetime.min,
            reverse=True,
        )
        return render_template(
            "comparison.html",
            experiments=experiments,
            cfg=cfg,
            selected_ids=selected_ids,
            selected_map=selected_map,
            active_nav="comparison",
        )

    @app.route("/experiments/<exp_id>")
    def experiment_detail(exp_id: str):
        experiment = _get_experiment_or_404(exp_id)
        figure = _build_single_experiment_figure(experiment)
        return render_template(
            "experiment_detail.html",
            experiment=experiment,
            cfg=cfg,
            figure=figure,
            active_nav="detail",
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

    @app.post("/comparison/data")
    def comparison_data():
        payload = request.get_json(force=True, silent=True) or {}
        exp_ids = payload.get("ids") or []
        if not isinstance(exp_ids, list) or len(exp_ids) < 2:
            abort(400, "Provide at least two experiment ids.")
        experiments = [_get_experiment_or_404(exp_id) for exp_id in exp_ids]
        # Sort by last_modified date (most recent first)
        experiments = sorted(
            experiments,
            key=lambda exp: exp.last_modified or datetime.min,
            reverse=True,
        )
        overlay_data = _build_overlay_data(experiments)
        comparison_rows = _build_comparison_rows(experiments)
        return jsonify(
            {
                "figure": overlay_data,
                "experiments": comparison_rows,
            }
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
        curve_map[label] = LossCurveData(steps=curves.steps, cumulative_flops=curves.cumulative_flops, series=filtered)
        trace_ids[label] = experiment.id
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
        rows.append(
            {
                "id": exp.id,
                "name": exp.name,
                "git_commit": exp.git_commit,
                "title": exp.title,
                "rollout_steps": exp.rollout_steps,
                "metadata": exp.metadata_text,
                "metadata_diff": diff_text,
                "git_metadata": exp.git_metadata_text,
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
    loss_curves = LossCurveData(steps=curves.steps, cumulative_flops=curves.cumulative_flops, series=filtered)
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
