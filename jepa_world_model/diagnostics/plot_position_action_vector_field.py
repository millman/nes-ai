#!/usr/bin/env python3
"""Plot per-cell action vector fields in PCA space."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from jepa_world_model.actions import decode_action_id
from jepa_world_model.data import TrajectorySequenceDataset, collate_batch
from jepa_world_model.diagnostics.run_position_bucket_alignment import (
    PositionResolver,
    _read_metadata,
)
from jepa_world_model.plots.build_motion_subspace import build_motion_subspace
from jepa_world_model.pose_rollout import rollout_pose_sequence
from jepa_world_model.rollout import rollout_teacher_forced
from jepa_world_model.model import JEPAWorldModel


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint at {path}")
    payload = torch.load(path, map_location=device)
    if "model_state" not in payload:
        raise AssertionError("Checkpoint missing model_state.")
    return payload


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
    if target == "p":
        _, _, h_states = rollout_teacher_forced(model, z_embeddings, actions)
        pose_pred, _ = rollout_pose_sequence(model, h_states, actions)
        return pose_pred
    raise ValueError(f"Unsupported target={target}")


def _flatten_positions(
    positions: List[List[Tuple[int, int]]],
    seq_len: int,
) -> List[Tuple[int, int]]:
    flat: List[Tuple[int, int]] = []
    for seq in positions:
        if len(seq) != seq_len:
            raise AssertionError(f"Expected seq_len={seq_len} positions, got {len(seq)}")
        for idx in range(seq_len - 1):
            flat.append(seq[idx])
    return flat


def _collect_positions(
    resolver: PositionResolver,
    paths: List[List[str]],
) -> List[List[Tuple[int, int]]]:
    all_positions: List[List[Tuple[int, int]]] = []
    for seq_paths in paths:
        seq_positions: List[Tuple[int, int]] = []
        for path_str in seq_paths:
            seq_positions.append(resolver.resolve(Path(path_str)))
        all_positions.append(seq_positions)
    return all_positions


def _plot_vector_field(
    out_path: Path,
    grid_rows: int,
    grid_cols: int,
    vectors: Dict[Tuple[int, int], Dict[int, Tuple[float, float, int]]],
    action_dim: int,
    scale_factor: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    action_ids = sorted({aid for per_cell in vectors.values() for aid in per_cell})
    for idx, aid in enumerate(action_ids):
        xs: List[float] = []
        ys: List[float] = []
        us: List[float] = []
        vs: List[float] = []
        for (row, col), per_action in vectors.items():
            if aid not in per_action:
                continue
            u, v, _ = per_action[aid]
            xs.append(col)
            ys.append(row)
            us.append(u * scale_factor)
            vs.append(v * scale_factor)
        if not xs:
            continue
        ax.quiver(
            xs,
            ys,
            us,
            vs,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=colors[idx % len(colors)],
            alpha=0.8,
            label=decode_action_id(aid, action_dim),
            width=0.006,
        )
    ax.set_xlim(-0.5, grid_cols - 0.5)
    ax.set_ylim(grid_rows - 0.5, -0.5)
    ax.set_xticks(range(grid_cols))
    ax.set_yticks(range(grid_rows))
    ax.grid(True, color="0.85", linestyle="-", linewidth=0.6)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.set_title("Per-cell mean delta (PCA components 0/1)")
    ax.legend(fontsize=8, loc="upper right", frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _write_vectors_csv(
    out_path: Path,
    vectors: Dict[Tuple[int, int], Dict[int, Tuple[float, float, int]]],
    action_dim: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "col", "action_id", "action_label", "count", "mean_pc0", "mean_pc1"])
        for (row, col), per_action in sorted(vectors.items()):
            for aid, (u, v, count) in sorted(per_action.items()):
                writer.writerow([row, col, aid, decode_action_id(aid, action_dim), count, u, v])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-cell action vector field in PCA space.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target", choices=("h", "z", "p"), default="h")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-sequences", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grid-rows", type=int, default=14)
    parser.add_argument("--grid-cols", type=int, default=16)
    parser.add_argument("--agent-color", type=str, default="66,167,70")
    parser.add_argument("--inventory-height", type=int, default=None)
    parser.add_argument("--min-action-count", type=int, default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    run_dir = args.run_dir
    metadata = _read_metadata(run_dir)
    sample_sequences = metadata.sample_sequences if args.sample_sequences is None else args.sample_sequences
    min_action_count = metadata.min_action_count if args.min_action_count is None else args.min_action_count
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
    )
    sample_count = min(sample_sequences, len(dataset))
    subset = torch.utils.data.Subset(dataset, list(range(sample_count)))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    resolver = PositionResolver(
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        agent_color=tuple(int(p) for p in args.agent_color.split(",")),
        inventory_height=args.inventory_height,
    )

    embeddings_list: List[torch.Tensor] = []
    actions_list: List[torch.Tensor] = []
    paths_list: List[List[str]] = []
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
        max(2, metadata.top_k_components),
        paths_list,
    )
    positions_flat = _flatten_positions(positions_by_seq, metadata.seq_len)
    if len(positions_flat) != motion.delta_proj.shape[0]:
        raise AssertionError(
            f"Position count {len(positions_flat)} does not match deltas {motion.delta_proj.shape[0]}"
        )

    deltas_2d = motion.delta_proj[:, :2]
    vectors: Dict[Tuple[int, int], Dict[int, Tuple[float, float, int]]] = {}
    for idx, pos in enumerate(positions_flat):
        aid = int(motion.action_ids[idx])
        vectors.setdefault(pos, {}).setdefault(aid, [])
        vectors[pos][aid].append(deltas_2d[idx])

    summarized: Dict[Tuple[int, int], Dict[int, Tuple[float, float, int]]] = {}
    for pos, per_action in vectors.items():
        for aid, items in per_action.items():
            arr = np.vstack(items)
            count = arr.shape[0]
            if count < min_action_count:
                continue
            mean_vec = arr.mean(axis=0)
            summarized.setdefault(pos, {})[aid] = (float(mean_vec[0]), float(mean_vec[1]), count)

    output_dir = args.output_dir or (run_dir / f"vis_action_vector_field_{args.target}_by_pos")
    step = int(payload.get("global_step", 0))
    img_path = output_dir / f"action_vector_field_{args.target}_{step:07d}.png"
    csv_path = output_dir / f"action_vector_field_{args.target}_{step:07d}.csv"
    _plot_vector_field(
        img_path,
        args.grid_rows,
        args.grid_cols,
        summarized,
        motion.action_dim,
        args.scale,
    )
    _write_vectors_csv(csv_path, summarized, motion.action_dim)
    print(f"Wrote vector field plot to {img_path}")
    print(f"Wrote vector field CSV to {csv_path}")


if __name__ == "__main__":
    main()
