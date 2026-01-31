#!/usr/bin/env python3
"""Compute per-sample inverse dynamics loss for a specific training step."""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import random

import torch

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jepa_world_model.data import TrajectorySequenceDataset, collate_batch
from jepa_world_model.diagnostics_utils import should_use_z2h_init
from jepa_world_model.model import JEPAWorldModel
from jepa_world_model.model_config import LayerNormConfig, ModelConfig
from jepa_world_model_trainer import LossWeights, jepa_loss
from recon.data import list_trajectories, short_traj_state_label


BCE_PER_SAMPLE = torch.nn.BCEWithLogitsLoss(reduction="none")


def _normalize_optional(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"null", "none"}:
        return None
    return value


def _split_trajectories(
    root: Path, max_traj: int | None, val_fraction: float, seed: int
) -> Tuple[List[str], List[str]]:
    traj_names = sorted(list(list_trajectories(root).keys()))
    if max_traj is not None:
        traj_names = traj_names[:max_traj]
    if not traj_names:
        return [], []
    rng = random.Random(seed)
    rng.shuffle(traj_names)
    if val_fraction <= 0:
        return traj_names, []
    if val_fraction >= 1.0:
        return [], traj_names
    val_count = max(1, int(len(traj_names) * val_fraction)) if len(traj_names) > 1 else 0
    val_count = min(val_count, max(0, len(traj_names) - 1))
    val_names = traj_names[:val_count]
    train_names = traj_names[val_count:]
    return train_names, val_names


def _load_metadata(run_dir: Path) -> dict:
    metadata_path = run_dir / "metadata.txt"
    if not metadata_path.is_file():
        raise AssertionError(f"Missing metadata.txt in {run_dir}")
    with metadata_path.open("rb") as f:
        return tomllib.load(f)


def _resolve_data_root(run_dir: Path, data_root: str) -> Path:
    root_path = Path(data_root)
    if root_path.is_absolute():
        return root_path
    cwd_path = Path(os.getcwd()) / root_path
    if cwd_path.exists():
        return cwd_path
    run_relative = run_dir / root_path
    if run_relative.exists():
        return run_relative
    return cwd_path


def _iter_batches(
    dataset_size: int,
    batch_size: int,
    generator: torch.Generator,
) -> Iterable[Tuple[int, int, List[int]]]:
    epoch = 0
    while True:
        order = torch.randperm(dataset_size, generator=generator).tolist()
        batch_idx = 0
        for offset in range(0, dataset_size, batch_size):
            yield epoch, batch_idx, order[offset : offset + batch_size]
            batch_idx += 1
        epoch += 1


def _get_batch_indices(step: int, dataset_size: int, batch_size: int, seed: int) -> Tuple[int, int, List[int]]:
    if step < 0:
        raise AssertionError("step must be non-negative.")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    current_step = 0
    for epoch, batch_idx, batch_indices in _iter_batches(dataset_size, batch_size, generator):
        if current_step == step:
            return epoch, batch_idx, batch_indices
        current_step += 1
    raise AssertionError("Failed to resolve batch indices for step.")


def _parse_model_config(cfg: dict, train_cfg: dict, weights: LossWeights) -> ModelConfig:
    layer_norms_raw = cfg.get("layer_norms", {})
    layer_norms = LayerNormConfig(**layer_norms_raw)
    encoder_schedule = tuple(cfg.get("encoder_schedule"))
    decoder_schedule_raw = cfg.get("decoder_schedule")
    decoder_schedule = tuple(decoder_schedule_raw) if decoder_schedule_raw is not None else None
    model_cfg = ModelConfig(
        in_channels=int(cfg.get("in_channels")),
        image_size=int(cfg.get("image_size")),
        hidden_dim=int(cfg.get("hidden_dim")),
        encoder_schedule=encoder_schedule,
        decoder_schedule=decoder_schedule,
        action_dim=int(cfg.get("action_dim")),
        state_dim=int(cfg.get("state_dim")),
        warmup_frames_h=int(cfg.get("warmup_frames_h", 0)),
        pose_dim=_normalize_optional(cfg.get("pose_dim")),
        pose_delta_detach_h=bool(cfg.get("pose_delta_detach_h")),
        pose_correction_use_z=bool(cfg.get("pose_correction_use_z")),
        pose_correction_detach_z=bool(cfg.get("pose_correction_detach_z")),
        layer_norms=layer_norms,
        predictor_spectral_norm=bool(cfg.get("predictor_spectral_norm")),
        z_norm=bool(cfg.get("z_norm")),
        enable_inverse_dynamics_z=_normalize_optional(cfg.get("enable_inverse_dynamics_z")),
        enable_inverse_dynamics_h=_normalize_optional(cfg.get("enable_inverse_dynamics_h")),
        enable_inverse_dynamics_p=_normalize_optional(cfg.get("enable_inverse_dynamics_p")),
        enable_action_delta_z=_normalize_optional(cfg.get("enable_action_delta_z")),
        enable_action_delta_h=_normalize_optional(cfg.get("enable_action_delta_h")),
        enable_action_delta_p=_normalize_optional(cfg.get("enable_action_delta_p")),
        enable_h2z_delta=_normalize_optional(cfg.get("enable_h2z_delta")),
    )
    model_cfg_runtime = replace(
        model_cfg,
        z_norm=bool(train_cfg.get("z_norm")),
        enable_inverse_dynamics_z=weights.inverse_dynamics_z > 0,
        enable_inverse_dynamics_h=weights.inverse_dynamics_h > 0,
        enable_inverse_dynamics_p=weights.inverse_dynamics_p > 0,
        enable_action_delta_z=weights.action_delta_z > 0,
        enable_action_delta_h=(weights.action_delta_h > 0 or weights.additivity_h > 0),
        enable_action_delta_p=(weights.action_delta_p > 0 or weights.additivity_p > 0 or weights.rollout_kstep_p > 0),
        enable_h2z_delta=weights.h2z_delta > 0,
    )
    return model_cfg_runtime


def _format_sample_path(path: str) -> str:
    label = short_traj_state_label(path)
    return label


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect per-sample inverse dynamics loss for a training step."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Experiment run directory (e.g., out.jepa_world_model_trainer/2026-01-30_14-09-41)",
    )
    parser.add_argument("--step", type=int, required=True, help="Training step to inspect")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: <run_dir>/checkpoints/last.pt)",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on")
    parser.add_argument("--top-k", type=int, default=8, help="How many samples to print")
    args = parser.parse_args()

    metadata = _load_metadata(args.run_dir)
    train_cfg = metadata.get("train_config")
    if train_cfg is None:
        raise AssertionError("metadata.txt missing train_config")
    model_cfg_raw = metadata.get("model_config")
    if model_cfg_raw is None:
        raise AssertionError("metadata.txt missing model_config")

    weights_raw = train_cfg.get("loss_weights")
    if weights_raw is None:
        raise AssertionError("metadata.txt missing train_config.loss_weights")
    weights = LossWeights(**weights_raw)

    data_root = train_cfg.get("data_root")
    if not data_root:
        raise AssertionError("data_root missing from metadata")
    batch_size = int(train_cfg.get("batch_size"))
    seq_len = int(train_cfg.get("seq_len"))
    seed = int(train_cfg.get("seed"))
    val_split_seed = int(train_cfg.get("val_split_seed"))
    val_fraction = float(train_cfg.get("val_fraction", 0.0))
    max_traj_raw = _normalize_optional(train_cfg.get("max_trajectories"))
    max_traj = int(max_traj_raw) if max_traj_raw is not None else None

    data_root_path = _resolve_data_root(args.run_dir, data_root)

    train_trajs, _ = _split_trajectories(data_root_path, max_traj, val_fraction, val_split_seed)
    dataset = TrajectorySequenceDataset(
        root=data_root_path,
        seq_len=seq_len,
        image_hw=(int(model_cfg_raw.get("image_size")), int(model_cfg_raw.get("image_size"))),
        max_traj=None,
        included_trajectories=train_trajs,
    )

    dataset_size = len(dataset)
    if dataset_size <= 0:
        raise AssertionError("Dataset is empty after splitting.")
    if batch_size <= 0:
        raise AssertionError("batch_size must be positive.")

    epoch, batch_idx, batch_indices = _get_batch_indices(args.step, dataset_size, batch_size, seed)
    batch_items = [dataset[idx] for idx in batch_indices]
    batch = collate_batch(batch_items)

    model_cfg = _parse_model_config(model_cfg_raw, train_cfg, weights)
    model = JEPAWorldModel(model_cfg).to(args.device)

    checkpoint_path = args.checkpoint or (args.run_dir / "checkpoints" / "last.pt")
    if not checkpoint_path.is_file():
        raise AssertionError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    x_frames, a_seq, paths, idx_tensor = batch
    x_frames = x_frames.to(args.device)
    a_seq = a_seq.to(args.device)

    encode_outputs = model.encode_sequence(x_frames)
    _, _, _, h_states = jepa_loss(
        model,
        encode_outputs,
        a_seq,
        use_z2h_init=should_use_z2h_init(weights),
        detach_z_inputs=bool(train_cfg.get("detach_z_from_h_and_p")),
        force_h_zero=bool(train_cfg.get("force_h_zero")),
    )

    if model.inverse_dynamics_h is None:
        raise AssertionError("inverse_dynamics_h head is disabled in the loaded model.")

    action_logits_h = model.inverse_dynamics_h(h_states[:, :-1], h_states[:, 1:])
    per_entry = BCE_PER_SAMPLE(action_logits_h, a_seq[:, :-1])
    per_sample_loss = per_entry.mean(dim=(1, 2)).detach().cpu()
    h_norms = h_states.norm(dim=-1).detach().cpu()
    per_sample_h_mean = h_norms.mean(dim=1)
    per_sample_h_max = h_norms.max(dim=1).values

    order = torch.argsort(per_sample_loss, descending=True)
    top_k = min(args.top_k, per_sample_loss.numel())

    print(f"step={args.step} epoch={epoch} batch={batch_idx} size={len(batch_indices)}")
    print(f"checkpoint={checkpoint_path}")
    print("top per-sample inverse_dynamics_h losses:")
    for rank in range(top_k):
        idx = int(order[rank])
        sample_idx = int(idx_tensor[idx])
        path_label = _format_sample_path(paths[idx][0])
        print(
            f"  rank {rank+1}: sample_index={sample_idx} loss={per_sample_loss[idx]:.6f} "
            f"h_mean={per_sample_h_mean[idx]:.4f} h_max={per_sample_h_max[idx]:.4f} "
            f"path={path_label}"
        )


if __name__ == "__main__":
    main()
