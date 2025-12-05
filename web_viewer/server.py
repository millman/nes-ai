from __future__ import annotations

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
)
from .plots import build_overlay

PACKAGE_ROOT = Path(__file__).resolve().parent


def create_app(config: Optional[ViewerConfig] = None) -> Flask:
    cfg = config or ViewerConfig()
    app = Flask(
        __name__,
        template_folder=str(PACKAGE_ROOT / "templates"),
        static_folder=str(PACKAGE_ROOT / "static"),
        static_url_path="/static/web_viewer",
    )
    app.config["VIEWER_CONFIG"] = cfg

    def _load_all() -> List[Experiment]:
        return list_experiments(cfg.output_dir)

    def _get_experiment_or_404(exp_id: str) -> Experiment:
        exp_path = cfg.output_dir / exp_id
        experiment = load_experiment(exp_path)
        if experiment is None:
            abort(404, f"Experiment {exp_id} not found.")
        return experiment

    @app.route("/")
    def index():
        experiments = _load_all()
        return render_template(
            "experiments.html",
            experiments=experiments,
            cfg=cfg,
        )

    @app.route("/comparison")
    def comparison():
        experiments = _load_all()
        return render_template(
            "comparison.html",
            experiments=experiments,
            cfg=cfg,
        )

    @app.post("/experiments/<exp_id>/notes")
    def update_notes(exp_id: str):
        experiment = _get_experiment_or_404(exp_id)
        payload = request.get_json(force=True, silent=True) or {}
        new_notes = payload.get("notes", "")
        write_notes(experiment.path / "notes.txt", new_notes)
        return jsonify({"status": "ok"})

    @app.post("/comparison/data")
    def comparison_data():
        payload = request.get_json(force=True, silent=True) or {}
        exp_ids = payload.get("ids") or []
        if not isinstance(exp_ids, list) or len(exp_ids) < 2:
            abort(400, "Provide at least two experiment ids.")
        experiments = [_get_experiment_or_404(exp_id) for exp_id in exp_ids]
        overlay_data = _build_overlay_data(experiments)
        comparison_rows = _build_comparison_rows(experiments)
        return jsonify(
            {
                "figure": overlay_data,
                "experiments": comparison_rows,
            }
        )

    @app.route("/assets/<exp_id>/<path:filename>")
    def serve_asset(exp_id: str, filename: str):
        experiment = _get_experiment_or_404(exp_id)
        safe_path = safe_join(str(experiment.path), filename)
        if safe_path is None:
            abort(404)
        target = Path(safe_path)
        if not target.exists():
            abort(404)
        directory = target.parent
        return send_from_directory(directory, target.name)

    return app


def _build_overlay_data(experiments: List[Experiment]):
    curve_map: Dict[str, LossCurveData] = {}
    for experiment in experiments:
        if experiment.loss_csv is None:
            continue
        curves = load_loss_curves(experiment.loss_csv)
        if curves is None:
            continue
        curve_map[experiment.name] = curves
    return build_overlay(curve_map)


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
                "metadata": exp.metadata_text,
                "metadata_diff": diff_text,
                "loss_image": url_for("serve_asset", exp_id=exp.id, filename="metrics/loss_curves.png")
                if exp.loss_image
                else None,
            }
        )
    return rows
