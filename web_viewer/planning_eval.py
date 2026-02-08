from __future__ import annotations

import json
from dataclasses import asdict, fields, replace
from datetime import datetime
from pathlib import Path
import tomllib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import tomli_w
import torch

from gridworldkey_env import create_env_with_theme
from jepa_world_model.data import PreloadedTrajectorySequenceDataset
from jepa_world_model.diagnostics.grid_overlay import build_grid_overlay_frames
from jepa_world_model.diagnostics_runner import run_planning_diagnostics_step
from jepa_world_model.model import JEPAWorldModel
from jepa_world_model.model_config import LayerNormConfig, ModelConfig
from jepa_world_model_trainer import LossWeights, TrainConfig, _build_embedding_batch, _split_trajectories


def _coerce_scalar(value: Any, target_example: Any) -> Any:
    if isinstance(target_example, bool):
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return bool(value)
    if isinstance(target_example, int) and not isinstance(target_example, bool):
        return int(value)
    if isinstance(target_example, float):
        return float(value)
    if isinstance(target_example, str):
        return str(value)
    return value


def _apply_overrides_dataclass(obj: Any, overrides: Dict[str, Any]) -> None:
    for field in fields(obj):
        name = field.name
        if name not in overrides:
            continue
        current = getattr(obj, name)
        incoming = overrides[name]
        if hasattr(current, "__dataclass_fields__") and isinstance(incoming, dict):
            _apply_overrides_dataclass(current, incoming)
            continue
        setattr(obj, name, _coerce_scalar(incoming, current))


def _parse_model_config(model_cfg_raw: Dict[str, Any]) -> ModelConfig:
    layer_norms_raw = model_cfg_raw.get("layer_norms") or {}
    layer_norms = LayerNormConfig(**layer_norms_raw)
    encoder_schedule = tuple(model_cfg_raw.get("encoder_schedule") or ())
    decoder_schedule_raw = model_cfg_raw.get("decoder_schedule")
    decoder_schedule = None if decoder_schedule_raw is None else tuple(decoder_schedule_raw)
    model = ModelConfig(
        in_channels=int(model_cfg_raw.get("in_channels", 3)),
        image_size=int(model_cfg_raw.get("image_size", 64)),
        hidden_dim=int(model_cfg_raw.get("hidden_dim", 512)),
        encoder_schedule=encoder_schedule,
        decoder_schedule=decoder_schedule,
        action_dim=int(model_cfg_raw.get("action_dim", 8)),
        state_dim=int(model_cfg_raw.get("state_dim", 256)),
        warmup_frames_h=int(model_cfg_raw.get("warmup_frames_h", 0)),
        pose_dim=model_cfg_raw.get("pose_dim"),
        pose_delta_detach_h=bool(model_cfg_raw.get("pose_delta_detach_h", True)),
        layer_norms=layer_norms,
        predictor_spectral_norm=bool(model_cfg_raw.get("predictor_spectral_norm", True)),
        enable_inverse_dynamics_z=bool(model_cfg_raw.get("enable_inverse_dynamics_z", False)),
        enable_inverse_dynamics_h=bool(model_cfg_raw.get("enable_inverse_dynamics_h", False)),
        enable_inverse_dynamics_p=bool(model_cfg_raw.get("enable_inverse_dynamics_p", False)),
        enable_inverse_dynamics_dp=bool(model_cfg_raw.get("enable_inverse_dynamics_dp", False)),
        enable_action_delta_z=bool(model_cfg_raw.get("enable_action_delta_z", False)),
        enable_action_delta_h=bool(model_cfg_raw.get("enable_action_delta_h", False)),
        enable_action_delta_p=bool(model_cfg_raw.get("enable_action_delta_p", False)),
        enable_dz_to_dp_projector=bool(model_cfg_raw.get("enable_dz_to_dp_projector", False)),
        enable_h2z_delta=bool(model_cfg_raw.get("enable_h2z_delta", False)),
    )
    return model


def _load_from_metadata(run_dir: Path) -> Tuple[TrainConfig, ModelConfig, LossWeights]:
    metadata_path = run_dir / "metadata.txt"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.txt in {run_dir}")
    payload = tomllib.loads(metadata_path.read_text())
    train_raw = payload.get("train_config") or {}
    model_raw = payload.get("model_config") or {}

    cfg = TrainConfig()
    cfg.data_root = Path(train_raw.get("data_root", cfg.data_root))
    cfg.seq_len = int(train_raw.get("seq_len", cfg.seq_len))
    cfg.max_trajectories = train_raw.get("max_trajectories", cfg.max_trajectories)
    cfg.val_fraction = float(train_raw.get("val_fraction", cfg.val_fraction))
    cfg.val_split_seed = int(train_raw.get("val_split_seed", cfg.val_split_seed))
    cfg.z_norm = str(train_raw.get("z_norm", cfg.z_norm))
    cfg.force_h_zero = bool(train_raw.get("force_h_zero", cfg.force_h_zero))

    _apply_overrides_dataclass(cfg.hard_example, train_raw.get("hard_example") or {})
    _apply_overrides_dataclass(cfg.graph_diagnostics, train_raw.get("graph_diagnostics") or {})
    _apply_overrides_dataclass(cfg.planning_diagnostics, train_raw.get("planning_diagnostics") or {})

    weights = LossWeights()
    _apply_overrides_dataclass(weights, train_raw.get("loss_weights") or {})
    cfg.loss_weights = weights

    model_cfg = _parse_model_config(model_raw)
    return cfg, model_cfg, weights


def _model_cfg_runtime(cfg: TrainConfig, model_cfg: ModelConfig, weights: LossWeights) -> ModelConfig:
    return replace(
        model_cfg,
        z_norm=cfg.z_norm,
        enable_inverse_dynamics_z=weights.inverse_dynamics_z > 0,
        enable_inverse_dynamics_h=weights.inverse_dynamics_h > 0,
        enable_inverse_dynamics_p=weights.inverse_dynamics_p > 0,
        enable_inverse_dynamics_dp=weights.inverse_dynamics_dp > 0,
        enable_action_delta_z=weights.action_delta_z > 0,
        enable_action_delta_h=(weights.action_delta_h > 0 or weights.additivity_h > 0),
        enable_action_delta_p=(
            weights.action_delta_dp > 0
            or weights.additivity_dp > 0
            or weights.rollout_kstep_p > 0
            or weights.dz_anchor_dp > 0
            or weights.loop_closure_p > 0
            or weights.distance_corr_p > 0
            or weights.noop_residual_dp > 0
            or weights.noop_residual_dh > 0
            or weights.h_smooth > 0
            or weights.scale_dp > 0
            or weights.geometry_rank_p > 0
            or weights.inverse_cycle_dp > 0
            or weights.inverse_dynamics_p > 0
            or weights.inverse_dynamics_dp > 0
        ),
        enable_dz_to_dp_projector=weights.dz_anchor_dp > 0,
        enable_h2z_delta=weights.h2z_delta > 0,
    )


def _checkpoint_step(checkpoint_path: Path, payload: Dict[str, Any]) -> int:
    if "global_step" in payload:
        return int(payload["global_step"])
    stem = checkpoint_path.stem
    if stem.startswith("step_"):
        try:
            return int(stem.split("_", 1)[1])
        except ValueError:
            return 0
    return 0


def _pick_latest_checkpoint(run_dir: Path) -> Path:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints directory in {run_dir}")
    step_files = sorted(checkpoints_dir.glob("step_*.pt"))
    if step_files:
        return step_files[-1]
    last = checkpoints_dir / "last.pt"
    if last.exists():
        return last
    final = checkpoints_dir / "final.pt"
    if final.exists():
        return final
    raise FileNotFoundError(f"No checkpoint files found in {checkpoints_dir}")


def default_trace_specs(grid_rows: int, grid_cols: int) -> List[Dict[str, Any]]:
    return [
        {
            "label": "test1",
            "start": [grid_rows - 2, 1],
            "goal": [grid_rows // 2, grid_cols // 2],
        },
        {
            "label": "test2",
            "start": [grid_rows // 2, grid_cols // 2],
            "goal": [grid_rows // 2, min(grid_cols - 2, grid_cols // 2 + 2)],
        },
        {
            "label": "test3",
            "start": [1, 1],
            "goal": [grid_rows - 2, grid_cols - 2],
        },
    ]


def _normalize_trace_specs(
    traces: Sequence[Dict[str, Any]],
    *,
    grid_rows: int,
    grid_cols: int,
) -> List[Tuple[str, Tuple[int, int], Tuple[int, int]]]:
    normalized: List[Tuple[str, Tuple[int, int], Tuple[int, int]]] = []
    seen_labels: set[str] = set()
    for idx, trace in enumerate(traces):
        label = str(trace.get("label") or f"test{idx + 1}")
        if label in seen_labels:
            raise AssertionError(f"Duplicate planning trace label: {label}")
        start_raw = trace.get("start")
        goal_raw = trace.get("goal")
        if not (isinstance(start_raw, (list, tuple)) and len(start_raw) == 2):
            raise AssertionError(f"Trace {label} start must be [row, col].")
        if not (isinstance(goal_raw, (list, tuple)) and len(goal_raw) == 2):
            raise AssertionError(f"Trace {label} goal must be [row, col].")
        start = (int(start_raw[0]), int(start_raw[1]))
        goal = (int(goal_raw[0]), int(goal_raw[1]))
        for name, tile in (("start", start), ("goal", goal)):
            if tile[0] < 0 or tile[0] >= grid_rows or tile[1] < 0 or tile[1] >= grid_cols:
                raise AssertionError(f"Trace {label} {name}={tile} out of bounds for grid {grid_rows}x{grid_cols}.")
        seen_labels.add(label)
        normalized.append((label, start, goal))
    return normalized


def run_planning_eval(
    *,
    experiment_dir: Path,
    planning_overrides: Dict[str, Any],
    traces: Optional[Sequence[Dict[str, Any]]] = None,
    checkpoint_path: Optional[Path] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    experiment_dir = experiment_dir.resolve()
    cfg, model_cfg, weights = _load_from_metadata(experiment_dir)
    cfg.planning_diagnostics.enabled = True
    cfg.planning_diagnostics.latent_kind = "h"
    _apply_overrides_dataclass(cfg.planning_diagnostics, planning_overrides)

    selected_checkpoint = checkpoint_path or _pick_latest_checkpoint(experiment_dir)
    if not selected_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {selected_checkpoint}")

    device_obj = torch.device(device)
    payload = torch.load(selected_checkpoint, map_location=device_obj)
    if "model_state" not in payload:
        raise AssertionError(f"Checkpoint missing model_state: {selected_checkpoint}")
    step = _checkpoint_step(selected_checkpoint, payload)

    model_cfg_runtime = _model_cfg_runtime(cfg, model_cfg, weights)
    model = JEPAWorldModel(model_cfg_runtime).to(device_obj)
    model.load_state_dict(payload["model_state"])
    model.eval()

    train_trajs, _ = _split_trajectories(cfg.data_root, cfg.max_trajectories, cfg.val_fraction, cfg.val_split_seed)
    dataset = PreloadedTrajectorySequenceDataset(
        root=cfg.data_root,
        seq_len=cfg.seq_len,
        image_hw=(model_cfg.image_size, model_cfg.image_size),
        max_traj=None,
        included_trajectories=train_trajs,
    )
    if len(dataset) == 0:
        raise AssertionError(f"No training samples available in dataset at {cfg.data_root}")

    planning_generator = torch.Generator()
    planning_generator.manual_seed(0)
    planning_batch_cpu = _build_embedding_batch(
        dataset,
        cfg.planning_diagnostics.sample_sequences,
        generator=planning_generator,
    )

    planning_env = create_env_with_theme(
        theme=cfg.planning_diagnostics.env_theme,
        render_mode="rgb_array",
        keyboard_override=False,
        start_manual_control=False,
    )
    try:
        grid_overlay_frames = build_grid_overlay_frames(theme=cfg.planning_diagnostics.env_theme)
        trace_specs = traces or default_trace_specs(planning_env.grid_rows, planning_env.grid_cols)
        planning_tests = _normalize_trace_specs(
            trace_specs,
            grid_rows=planning_env.grid_rows,
            grid_cols=planning_env.grid_cols,
        )

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        eval_dir = experiment_dir / "planning_eval" / timestamp
        eval_dir.mkdir(parents=True, exist_ok=True)

        run_planning_diagnostics_step(
            planning_cfg=cfg.planning_diagnostics,
            hard_example_cfg=cfg.hard_example,
            graph_cfg=cfg.graph_diagnostics,
            model_cfg=model_cfg_runtime,
            model=model,
            device=device_obj,
            weights=weights,
            global_step=step,
            planning_batch_cpu=planning_batch_cpu,
            planning_env=planning_env,
            grid_overlay_frames=grid_overlay_frames,
            run_dir=eval_dir,
            force_h_zero=cfg.force_h_zero,
            planning_tests=planning_tests,
            emit_exec_for_all_tests=True,
        )

        config_payload = {
            "checkpoint": str(selected_checkpoint),
            "checkpoint_step": step,
            "planning_diagnostics": asdict(cfg.planning_diagnostics),
            "traces": [
                {"label": label, "start": list(start), "goal": list(goal)}
                for label, start, goal in planning_tests
            ],
        }
        (eval_dir / "planning_config.txt").write_text(tomli_w.dumps(config_payload))

        run_info = {
            "timestamp": timestamp,
            "experiment_dir": str(experiment_dir),
            "checkpoint": str(selected_checkpoint),
            "checkpoint_step": int(step),
            "planning_eval_dir": str(eval_dir),
            "traces": config_payload["traces"],
        }
        (eval_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, sort_keys=True))
        return run_info
    finally:
        planning_env.close()
