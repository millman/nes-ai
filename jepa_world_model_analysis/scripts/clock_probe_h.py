#!/usr/bin/env python3
"""Probe whether planning hidden state h carries timestep/clock information."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch

from jepa_world_model.data import PreloadedTrajectorySequenceDataset
from jepa_world_model.diagnostics_utils import should_use_z2h_init
from jepa_world_model.model import JEPAWorldModel
from jepa_world_model.model_config import LayerNormConfig, ModelConfig
from jepa_world_model.rollout import rollout_teacher_forced


STATE_RE = re.compile(r"state_(\d+)\.png$")


def _action_to_direction(action_vec: np.ndarray) -> str | None:
    if action_vec.ndim != 1:
        raise AssertionError("action_vec must be 1D.")
    pressed = action_vec > 0.5
    if pressed.shape[0] < 8:
        raise AssertionError("Expected at least 8 action dimensions for direction parsing.")
    if pressed[:4].any():
        return None
    up = bool(pressed[4])
    down = bool(pressed[5])
    left = bool(pressed[6])
    right = bool(pressed[7])
    count = int(up) + int(down) + int(left) + int(right)
    if count == 0:
        return "NOOP"
    if count > 1:
        return None
    if left:
        return "L"
    if right:
        return "R"
    if up:
        return "U"
    if down:
        return "D"
    return None


def _action_labels_from_vectors(actions: np.ndarray) -> List[str | None]:
    if actions.ndim != 2:
        raise AssertionError("actions must be 2D [N, action_dim].")
    return [_action_to_direction(row) for row in actions]


def _load_model_cfg(metadata_path: Path) -> Dict[str, object]:
    payload = tomllib.loads(metadata_path.read_text())
    if "model_config" not in payload or "train_config" not in payload:
        raise AssertionError("metadata.txt must include model_config and train_config.")
    return payload


def _materialize_model_config(model_cfg_raw: Dict[str, object]) -> ModelConfig:
    model_cfg_data = dict(model_cfg_raw)
    if isinstance(model_cfg_data.get("encoder_schedule"), list):
        model_cfg_data["encoder_schedule"] = tuple(model_cfg_data["encoder_schedule"])
    if isinstance(model_cfg_data.get("decoder_schedule"), list):
        model_cfg_data["decoder_schedule"] = tuple(model_cfg_data["decoder_schedule"])
    if isinstance(model_cfg_data.get("layer_norms"), dict):
        model_cfg_data["layer_norms"] = LayerNormConfig(**model_cfg_data["layer_norms"])
    return ModelConfig(**model_cfg_data)


def _ridge_r2(X: np.ndarray, y: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray, lam: float = 10.0) -> float:
    x_train = X[train_mask].astype(np.float64)
    x_test = X[test_mask].astype(np.float64)
    y_train = y[train_mask].astype(np.float64)
    y_test = y[test_mask].astype(np.float64)
    if x_train.shape[0] == 0 or x_test.shape[0] == 0:
        raise AssertionError("Need both train and test rows for ridge probe.")
    mu = x_train.mean(axis=0, keepdims=True)
    sigma = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mu) / sigma
    x_test = (x_test - mu) / sigma
    x_train = np.concatenate([np.ones((x_train.shape[0], 1)), x_train], axis=1)
    x_test = np.concatenate([np.ones((x_test.shape[0], 1)), x_test], axis=1)
    eye = np.eye(x_train.shape[1], dtype=np.float64)
    eye[0, 0] = 0.0
    beta = np.linalg.solve(x_train.T @ x_train + lam * eye, x_train.T @ y_train)
    pred = x_test @ beta
    ss_res = float(np.sum((y_test - pred) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _centroid_timestep_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> float:
    x_train = X[train_mask]
    x_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    if x_train.shape[0] == 0 or x_test.shape[0] == 0:
        raise AssertionError("Need both train and test rows for centroid probe.")
    centroids: List[np.ndarray] = []
    for t in range(seq_len):
        timestep_rows = x_train[y_train == t]
        if timestep_rows.shape[0] == 0:
            raise AssertionError(f"No train rows available for timestep {t}.")
        centroids.append(timestep_rows.mean(axis=0))
    C = np.stack(centroids, axis=0)
    d2 = ((x_test[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    pred = np.argmin(d2, axis=1)
    return float((pred == y_test).mean())


def summarize_clock_probe(run_dir: Path, sample_sequences: int, seed: int) -> Dict[str, float]:
    metadata_path = run_dir / "metadata.txt"
    ckpt_path = run_dir / "checkpoints" / "last.pt"
    if not metadata_path.exists():
        raise AssertionError(f"Missing metadata file: {metadata_path}")
    if not ckpt_path.exists():
        raise AssertionError(f"Missing checkpoint file: {ckpt_path}")

    payload = _load_model_cfg(metadata_path)
    train_cfg = payload["train_config"]
    model_cfg = _materialize_model_config(payload["model_config"])
    weights = SimpleNamespace(**train_cfg["loss_weights"])

    model = JEPAWorldModel(model_cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    data_root = Path(train_cfg["data_root"])
    if not data_root.is_absolute():
        data_root = Path.cwd() / data_root

    seq_len = int(train_cfg["seq_len"])
    if seq_len <= 1:
        raise AssertionError("seq_len must be > 1 for clock probe.")

    dataset = PreloadedTrajectorySequenceDataset(
        root=data_root,
        seq_len=seq_len,
        image_hw=(model_cfg.image_size, model_cfg.image_size),
        max_traj=(train_cfg["max_trajectories"] if isinstance(train_cfg.get("max_trajectories"), int) else None),
    )
    if len(dataset) < 2:
        raise AssertionError("Need at least two sequences for train/test split.")

    rng = np.random.default_rng(seed)
    count = len(dataset) if sample_sequences <= 0 else min(sample_sequences, len(dataset))
    indices = np.arange(len(dataset), dtype=np.int64)
    rng.shuffle(indices)
    indices = indices[:count]

    use_z2h_init = should_use_z2h_init(weights)
    force_h_zero = bool(train_cfg.get("force_h_zero", False))

    all_h = []
    all_z = []
    all_t = []
    all_state = []
    all_seq = []
    all_dh = []
    all_labels = []
    same_state_mask = []

    batch_size = 64
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        frames = []
        actions = []
        paths = []
        for idx in batch_indices.tolist():
            x, a, p, _ = dataset[idx]
            frames.append(x)
            actions.append(a)
            paths.append(p)
        x_batch = torch.stack(frames, dim=0)
        a_batch = torch.stack(actions, dim=0)
        with torch.no_grad():
            z_batch = model.encode_sequence(x_batch)["embeddings"]
            _, _, h_batch = rollout_teacher_forced(
                model,
                z_batch,
                a_batch,
                use_z2h_init=use_z2h_init,
                force_h_zero=force_h_zero,
            )
        all_h.append(h_batch.numpy())
        all_z.append(z_batch.numpy())

        B, T, _ = h_batch.shape
        for bi in range(B):
            seq_id = int(batch_indices[bi])
            state_ids: List[int] = []
            for frame_path in paths[bi]:
                match = STATE_RE.search(frame_path)
                state_ids.append(int(match.group(1)) if match else -1)
            for t in range(T):
                all_t.append(t)
                all_state.append(state_ids[t])
                all_seq.append(seq_id)
            dh = h_batch[bi, 1:] - h_batch[bi, :-1]
            all_dh.extend(torch.norm(dh, dim=-1).numpy().tolist())
            labels = _action_labels_from_vectors(a_batch[bi, :-1].numpy())
            all_labels.extend(labels)
            same_state_mask.extend([state_ids[t + 1] == state_ids[t] for t in range(T - 1)])

    H = np.concatenate(all_h, axis=0).reshape(-1, all_h[0].shape[-1])
    Z = np.concatenate(all_z, axis=0).reshape(-1, all_z[0].shape[-1])
    timestep = np.asarray(all_t, dtype=np.int64)
    seq_ids = np.asarray(all_seq, dtype=np.int64)
    state_ids = np.asarray(all_state, dtype=np.int64)

    unique_seq = np.unique(seq_ids)
    rng.shuffle(unique_seq)
    train_count = max(1, int(round(0.8 * len(unique_seq))))
    train_ids = set(unique_seq[:train_count].tolist())
    train_mask = np.array([sid in train_ids for sid in seq_ids], dtype=bool)
    test_mask = ~train_mask
    if not train_mask.any() or not test_mask.any():
        raise AssertionError("Failed to create non-empty train/test split.")

    r2_h = _ridge_r2(H, timestep, train_mask, test_mask)
    r2_z = _ridge_r2(Z, timestep, train_mask, test_mask)
    acc_h = _centroid_timestep_accuracy(H, timestep, seq_len, train_mask, test_mask)
    acc_z = _centroid_timestep_accuracy(Z, timestep, seq_len, train_mask, test_mask)

    by_state: Dict[int, List[int]] = defaultdict(list)
    for i, state in enumerate(state_ids.tolist()):
        if state >= 0:
            by_state[state].append(i)
    pair_dt = []
    pair_dist_h = []
    for idxs in by_state.values():
        if len(idxs) < 2:
            continue
        max_pairs = min(200, len(idxs) * (len(idxs) - 1) // 2)
        chosen = 0
        seen = set()
        while chosen < max_pairs:
            i, j = rng.choice(idxs, size=2, replace=False)
            a_idx, b_idx = (int(i), int(j)) if i < j else (int(j), int(i))
            if a_idx == b_idx or (a_idx, b_idx) in seen:
                continue
            seen.add((a_idx, b_idx))
            pair_dt.append(abs(int(timestep[a_idx]) - int(timestep[b_idx])))
            pair_dist_h.append(float(np.linalg.norm(H[a_idx] - H[b_idx])))
            chosen += 1
    if not pair_dt:
        raise AssertionError("No repeated-state pairs available for drift correlation.")
    pair_dt_arr = np.asarray(pair_dt, dtype=np.float64)
    pair_dist_arr = np.asarray(pair_dist_h, dtype=np.float64)
    corr = float(np.corrcoef(pair_dt_arr, pair_dist_arr)[0, 1])
    X = np.stack([np.ones_like(pair_dt_arr), pair_dt_arr], axis=1)
    beta, *_ = np.linalg.lstsq(X, pair_dist_arr, rcond=None)
    slope = float(beta[1])

    dh = np.asarray(all_dh, dtype=np.float64)
    labels = np.asarray(all_labels, dtype=object)
    noop_mask = labels == "NOOP"
    move_mask = np.isin(labels, ["L", "R", "U", "D"])
    if not noop_mask.any():
        raise AssertionError("No NOOP transitions found in sampled sequences.")
    if not move_mask.any():
        raise AssertionError("No move transitions found in sampled sequences.")
    same_state = np.asarray(same_state_mask, dtype=bool)
    if same_state.shape != dh.shape:
        raise AssertionError("same_state mask must align with dh transitions.")

    summary = {
        "sampled_sequences": int(len(indices)),
        "seq_len": int(seq_len),
        "num_points": int(H.shape[0]),
        "time_probe_ridge_r2_h": float(r2_h),
        "time_probe_ridge_r2_z": float(r2_z),
        "time_probe_centroid_acc_h": float(acc_h),
        "time_probe_centroid_acc_z": float(acc_z),
        "time_probe_chance_acc": float(1.0 / seq_len),
        "within_state_pairs": int(pair_dt_arr.size),
        "within_state_corr_dt_dist_h": float(corr),
        "within_state_slope_dist_per_dt": float(slope),
        "dh_noop_median": float(np.median(dh[noop_mask])),
        "dh_move_median": float(np.median(dh[move_mask])),
        "dh_noop_ratio": float(np.median(dh[noop_mask]) / max(np.median(dh[move_mask]), 1e-12)),
        "dh_noop_p90": float(np.quantile(dh[noop_mask], 0.90)),
        "dh_move_p90": float(np.quantile(dh[move_mask], 0.90)),
        "dh_same_state_p90": float(np.quantile(dh[same_state], 0.90)) if same_state.any() else float("nan"),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory (out.jepa_world_model_trainer/<run>)")
    parser.add_argument("--sample-sequences", type=int, default=1024, help="Number of sequence windows to sample (<=0 uses all).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling and pair selection.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args()

    summary = summarize_clock_probe(args.run_dir, args.sample_sequences, args.seed)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print(f"run: {args.run_dir}")
    print(
        "time_probe: "
        f"ridge_r2_h={summary['time_probe_ridge_r2_h']:.4f} "
        f"ridge_r2_z={summary['time_probe_ridge_r2_z']:.4f} "
        f"centroid_acc_h={summary['time_probe_centroid_acc_h']:.4f} "
        f"centroid_acc_z={summary['time_probe_centroid_acc_z']:.4f} "
        f"chance={summary['time_probe_chance_acc']:.4f}"
    )
    print(
        "revisit_drift: "
        f"pairs={summary['within_state_pairs']} "
        f"corr_dt_dist_h={summary['within_state_corr_dt_dist_h']:.4f} "
        f"slope_dist_per_dt={summary['within_state_slope_dist_per_dt']:.6f}"
    )
    print(
        "delta_h: "
        f"noop_median={summary['dh_noop_median']:.6f} "
        f"move_median={summary['dh_move_median']:.6f} "
        f"noop_ratio={summary['dh_noop_ratio']:.4f} "
        f"noop_p90={summary['dh_noop_p90']:.6f} "
        f"move_p90={summary['dh_move_p90']:.6f} "
        f"same_state_p90={summary['dh_same_state_p90']:.6f}"
    )


if __name__ == "__main__":
    main()
