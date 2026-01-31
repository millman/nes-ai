#!/usr/bin/env python3
"""Bucket action alignment stats by grid position for a training run."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from jepa_world_model.actions import decode_action_id
from jepa_world_model.data import TrajectorySequenceDataset, collate_batch
from jepa_world_model.model_config import LayerNormConfig, ModelConfig
from jepa_world_model.plots.build_motion_subspace import build_motion_subspace
from jepa_world_model.plots.plot_action_alignment_stats import compute_action_alignment_stats
from jepa_world_model.rollout import rollout_teacher_forced
from jepa_world_model.model import JEPAWorldModel


@dataclass
class RunMetadata:
    data_root: Path
    seq_len: int
    sample_sequences: int
    min_action_count: int
    top_k_components: int
    model_config: ModelConfig


def _read_metadata(run_dir: Path) -> RunMetadata:
    metadata_path = run_dir / "metadata.txt"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata.txt in {run_dir}")
    payload = tomllib.loads(metadata_path.read_text())
    train_cfg = payload.get("train_config") or {}
    diagnostics_cfg = train_cfg.get("diagnostics") or {}
    loss_weights = train_cfg.get("loss_weights") or {}
    model_cfg = payload.get("model_config") or {}
    layer_norms_raw = model_cfg.get("layer_norms") or {}
    layer_norms = LayerNormConfig(**layer_norms_raw)
    encoder_schedule = tuple(model_cfg.get("encoder_schedule") or ())
    decoder_schedule_raw = model_cfg.get("decoder_schedule")
    decoder_schedule = None if decoder_schedule_raw is None else tuple(decoder_schedule_raw)
    def _resolve_toggle(key: str, derived: bool) -> bool:
        raw = model_cfg.get(key)
        if isinstance(raw, str) and raw.lower() == "null":
            raw = None
        if raw is None:
            return derived
        return bool(raw)

    enable_inverse_dynamics_z = loss_weights.get("inverse_dynamics_z", 0) > 0
    enable_inverse_dynamics_h = loss_weights.get("inverse_dynamics_h", 0) > 0
    enable_inverse_dynamics_p = loss_weights.get("inverse_dynamics_p", 0) > 0
    enable_action_delta_z = loss_weights.get("action_delta_z", 0) > 0
    enable_action_delta_h = (
        loss_weights.get("action_delta_h", 0) > 0
        or loss_weights.get("additivity_h", 0) > 0
    )
    enable_action_delta_p = (
        loss_weights.get("action_delta_p", 0) > 0
        or loss_weights.get("additivity_p", 0) > 0
        or loss_weights.get("rollout_kstep_p", 0) > 0
    )
    enable_h2z_delta = loss_weights.get("h2z_delta", 0) > 0
    model = ModelConfig(
        in_channels=int(model_cfg.get("in_channels", 3)),
        image_size=int(model_cfg.get("image_size", 64)),
        hidden_dim=int(model_cfg.get("hidden_dim", 512)),
        encoder_schedule=encoder_schedule,
        decoder_schedule=decoder_schedule,
        action_dim=int(model_cfg.get("action_dim", 8)),
        state_dim=int(model_cfg.get("state_dim", 256)),
        warmup_frames_h=int(model_cfg.get("warmup_frames_h", 0)),
        pose_dim=model_cfg.get("pose_dim"),
        pose_delta_detach_h=bool(model_cfg.get("pose_delta_detach_h", True)),
        pose_correction_use_z=bool(model_cfg.get("pose_correction_use_z", False)),
        pose_correction_detach_z=bool(model_cfg.get("pose_correction_detach_z", True)),
        layer_norms=layer_norms,
        predictor_spectral_norm=bool(model_cfg.get("predictor_spectral_norm", True)),
        enable_inverse_dynamics_z=_resolve_toggle("enable_inverse_dynamics_z", enable_inverse_dynamics_z),
        enable_inverse_dynamics_h=_resolve_toggle("enable_inverse_dynamics_h", enable_inverse_dynamics_h),
        enable_inverse_dynamics_p=_resolve_toggle("enable_inverse_dynamics_p", enable_inverse_dynamics_p),
        enable_action_delta_z=_resolve_toggle("enable_action_delta_z", enable_action_delta_z),
        enable_action_delta_h=_resolve_toggle("enable_action_delta_h", enable_action_delta_h),
        enable_action_delta_p=_resolve_toggle("enable_action_delta_p", enable_action_delta_p),
        enable_h2z_delta=_resolve_toggle("enable_h2z_delta", enable_h2z_delta),
    )
    data_root = Path(train_cfg.get("data_root") or "")
    if not data_root.exists():
        raise FileNotFoundError(f"data_root={data_root} does not exist.")
    return RunMetadata(
        data_root=data_root,
        seq_len=int(train_cfg.get("seq_len", 8)),
        sample_sequences=int(diagnostics_cfg.get("sample_sequences", 128)),
        min_action_count=int(diagnostics_cfg.get("min_action_count", 5)),
        top_k_components=int(diagnostics_cfg.get("top_k_components", 8)),
        model_config=model,
    )


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint at {path}")
    payload = torch.load(path, map_location=device)
    if "model_state" not in payload:
        raise AssertionError("Checkpoint missing model_state.")
    return payload


class PositionResolver:
    def __init__(
        self,
        grid_rows: int,
        grid_cols: int,
        agent_color: Tuple[int, int, int],
        inventory_height: Optional[int] = None,
    ) -> None:
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.agent_color = np.array(agent_color, dtype=np.uint8)
        self.inventory_height = inventory_height
        self._info_cache: Dict[Path, Dict[str, np.ndarray]] = {}

    def _load_infos(self, traj_dir: Path) -> Dict[str, np.ndarray]:
        cached = self._info_cache.get(traj_dir)
        if cached is not None:
            return cached
        infos_path = traj_dir / "infos.npz"
        if not infos_path.is_file():
            self._info_cache[traj_dir] = {}
            return {}
        with np.load(infos_path) as data:
            info = {key: data[key] for key in data.files}
        self._info_cache[traj_dir] = info
        return info

    def _infer_inventory_height(self, height: int, tile_size: int) -> int:
        if self.inventory_height is not None:
            return self.inventory_height
        inventory_height = height - self.grid_rows * tile_size
        return max(0, inventory_height)

    def _tile_from_xy(self, x: int, y: int, height: int, width: int) -> Tuple[int, int]:
        tile_size = width // self.grid_cols
        if tile_size <= 0:
            raise AssertionError(f"Invalid tile_size={tile_size} from width={width} grid_cols={self.grid_cols}")
        inventory_height = self._infer_inventory_height(height, tile_size)
        row = int((y - inventory_height) // tile_size)
        col = int(x // tile_size)
        if row < 0 or col < 0 or row >= self.grid_rows or col >= self.grid_cols:
            raise AssertionError(
                f"Position out of bounds for grid {self.grid_rows}x{self.grid_cols}: row={row}, col={col}"
            )
        return row, col

    def resolve(self, frame_path: Path) -> Tuple[int, int]:
        traj_dir = frame_path.parent.parent
        infos = self._load_infos(traj_dir)
        frame_index = int(frame_path.stem.split("_")[1])
        if "agent_x" in infos and "agent_y" in infos:
            x = int(infos["agent_x"][frame_index])
            y = int(infos["agent_y"][frame_index])
            with Image.open(frame_path) as img:
                width, height = img.size
            return self._tile_from_xy(x, y, height, width)

        with Image.open(frame_path) as img:
            img = img.convert("RGB")
            arr = np.asarray(img)
        mask = np.all(arr == self.agent_color, axis=-1)
        if not np.any(mask):
            raise AssertionError(f"No agent pixels found in {frame_path}")
        ys, xs = np.where(mask)
        y = int(np.round(ys.mean()))
        x = int(np.round(xs.mean()))
        height, width = arr.shape[:2]
        return self._tile_from_xy(x, y, height, width)


def _extract_embeddings(
    model: JEPAWorldModel,
    frames: torch.Tensor,
    actions: torch.Tensor,
    target: str,
) -> torch.Tensor:
    outputs = model.encode_sequence(frames)
    z_embeddings = outputs["embeddings"]
    if target == "z":
        return z_embeddings
    if target == "h":
        _, _, h_states = rollout_teacher_forced(model, z_embeddings, actions)
        return h_states
    raise ValueError(f"Unsupported target={target}")


def _flatten_positions(
    positions: Sequence[Sequence[Tuple[int, int]]],
    seq_len: int,
) -> List[Tuple[int, int]]:
    flat: List[Tuple[int, int]] = []
    for seq in positions:
        if len(seq) != seq_len:
            raise AssertionError(f"Expected seq_len={seq_len} positions, got {len(seq)}")
        for idx in range(seq_len - 1):
            flat.append(seq[idx])
    return flat


def _write_bucket_stats(
    out_path: Path,
    buckets: Dict[Tuple[int, int], List[int]],
    delta_proj: np.ndarray,
    action_ids: np.ndarray,
    action_dim: int,
    min_action_count: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "row",
                "col",
                "bucket_count",
                "action_id",
                "action_label",
                "count",
                "mean_cos",
                "median_cos",
                "std_cos",
                "pct_high",
                "frac_negative",
                "mean_dir_norm",
                "delta_norm_mean",
                "delta_norm_median",
                "delta_norm_p10",
                "delta_norm_p90",
                "frac_low_delta_norm",
            ]
        )
        for (row, col), indices in sorted(buckets.items()):
            if not indices:
                continue
            delta_bucket = delta_proj[indices]
            actions_bucket = action_ids[indices]
            stats = compute_action_alignment_stats(
                delta_bucket,
                actions_bucket,
                min_action_count,
                cosine_high_threshold=0.7,
            )
            for stat in stats:
                writer.writerow(
                    [
                        row,
                        col,
                        len(indices),
                        stat.get("action_id"),
                        decode_action_id(stat.get("action_id", -1), action_dim),
                        stat.get("count", 0),
                        stat.get("mean", float("nan")),
                        stat.get("median", float("nan")),
                        stat.get("std", float("nan")),
                        stat.get("pct_high", float("nan")),
                        stat.get("frac_neg", float("nan")),
                        stat.get("mean_dir_norm", float("nan")),
                        stat.get("delta_norm_mean", float("nan")),
                        stat.get("delta_norm_median", float("nan")),
                        stat.get("delta_norm_p10", float("nan")),
                        stat.get("delta_norm_p90", float("nan")),
                        stat.get("frac_low_delta_norm", float("nan")),
                    ]
                )


def _write_bucket_summary(
    out_path: Path,
    buckets: Dict[Tuple[int, int], List[int]],
    delta_proj: np.ndarray,
    action_ids: np.ndarray,
    min_action_count: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "row",
                "col",
                "bucket_count",
                "mean_alignment",
                "median_alignment",
                "frac_negative",
            ]
        )
        for (row, col), indices in sorted(buckets.items()):
            if not indices:
                continue
            delta_bucket = delta_proj[indices]
            actions_bucket = action_ids[indices]
            stats = compute_action_alignment_stats(
                delta_bucket,
                actions_bucket,
                min_action_count,
                cosine_high_threshold=0.7,
            )
            counts = np.array([stat.get("count", 0) for stat in stats], dtype=np.float32)
            means = np.array([stat.get("mean", float("nan")) for stat in stats], dtype=np.float32)
            medians = np.array([stat.get("median", float("nan")) for stat in stats], dtype=np.float32)
            frac_negs = np.array([stat.get("frac_neg", float("nan")) for stat in stats], dtype=np.float32)
            total = float(counts.sum())
            if total <= 0:
                mean_alignment = float("nan")
                median_alignment = float("nan")
                frac_negative = float("nan")
            else:
                mean_alignment = float((means * counts).sum() / total)
                median_alignment = float((medians * counts).sum() / total)
                frac_negative = float((frac_negs * counts).sum() / total)
            writer.writerow(
                [
                    row,
                    col,
                    len(indices),
                    mean_alignment,
                    median_alignment,
                    frac_negative,
                ]
            )


def _parse_color(value: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError("agent_color must be a comma-separated RGB triple, e.g. 66,167,70")
    return tuple(int(p) for p in parts)


def _collect_positions(
    resolver: PositionResolver,
    paths: Sequence[Sequence[str]],
) -> List[List[Tuple[int, int]]]:
    all_positions: List[List[Tuple[int, int]]] = []
    for seq_paths in paths:
        seq_positions: List[Tuple[int, int]] = []
        for path_str in seq_paths:
            seq_positions.append(resolver.resolve(Path(path_str)))
        all_positions.append(seq_positions)
    return all_positions


def main() -> None:
    parser = argparse.ArgumentParser(description="Bucket action alignment by grid position.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target", choices=("h", "z"), default="h")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-sequences", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grid-rows", type=int, default=14)
    parser.add_argument("--grid-cols", type=int, default=16)
    parser.add_argument("--agent-color", type=str, default="66,167,70")
    parser.add_argument("--inventory-height", type=int, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    metadata = _read_metadata(run_dir)
    sample_sequences = metadata.sample_sequences if args.sample_sequences is None else args.sample_sequences
    device = torch.device(args.device)
    checkpoint_path = args.checkpoint or (run_dir / "checkpoints" / "last.pt")
    payload = _load_checkpoint(checkpoint_path, device)

    model = JEPAWorldModel(metadata.model_config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    dataset = TrajectorySequenceDataset(
        metadata.data_root,
        seq_len=metadata.seq_len,
        image_hw=(metadata.model_config.image_size, metadata.model_config.image_size),
        max_traj=None,
    )
    sample_count = min(sample_sequences, len(dataset))
    subset = torch.utils.data.Subset(dataset, list(range(sample_count)))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    embeddings_list: List[torch.Tensor] = []
    actions_list: List[torch.Tensor] = []
    paths_list: List[List[str]] = []
    resolver = PositionResolver(
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        agent_color=_parse_color(args.agent_color),
        inventory_height=args.inventory_height,
    )
    positions_by_seq: List[List[Tuple[int, int]]] = []

    with torch.no_grad():
        for frames, actions, paths, _ in loader:
            frames = frames.to(device)
            actions = actions.to(device)
            embeddings = _extract_embeddings(model, frames, actions, args.target)
            embeddings_list.append(embeddings.cpu())
            actions_list.append(actions.cpu())
            paths_list.extend(paths)
            positions_by_seq.extend(_collect_positions(resolver, paths))

    if not embeddings_list:
        raise AssertionError("No embeddings collected; check dataset or sample size.")
    embeddings_all = torch.cat(embeddings_list, dim=0)
    actions_all = torch.cat(actions_list, dim=0)
    motion = build_motion_subspace(
        embeddings_all,
        actions_all,
        metadata.top_k_components,
        paths_list,
    )
    positions_flat = _flatten_positions(positions_by_seq, metadata.seq_len)
    if len(positions_flat) != motion.delta_proj.shape[0]:
        raise AssertionError(
            f"Position count {len(positions_flat)} does not match deltas {motion.delta_proj.shape[0]}"
        )
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for idx, pos in enumerate(positions_flat):
        buckets.setdefault(pos, []).append(idx)

    output_dir = args.output_dir or (run_dir / f"vis_action_alignment_{args.target}_by_pos")
    step = int(payload.get("global_step", 0))
    stats_path = output_dir / f"action_alignment_by_pos_{args.target}_{step:07d}.csv"
    summary_path = output_dir / f"action_alignment_by_pos_{args.target}_summary_{step:07d}.csv"
    _write_bucket_stats(
        stats_path,
        buckets,
        motion.delta_proj,
        motion.action_ids,
        motion.action_dim,
        metadata.min_action_count,
    )
    _write_bucket_summary(
        summary_path,
        buckets,
        motion.delta_proj,
        motion.action_ids,
        metadata.min_action_count,
    )
    print(f"Wrote per-position stats to {stats_path}")
    print(f"Wrote per-position summary to {summary_path}")


if __name__ == "__main__":
    main()
