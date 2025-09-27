
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image_distance_vs_steps.py

Evaluate an image-distance model by comparing its **inferred pixel distance**
(defined as the magnitude of the model's predicted motion vector) to the
**action distance** (number of trajectory steps between two frames).

Pairs can be sampled either **within the same trajectory** (known step gap)
or **across different trajectories** (unknown step gap). We log all pairs,
make a scatter plot of known step gaps vs inferred pixel distance, and make a
histogram of inferred pixel distances for the unknown (cross-trajectory) pairs.

Outputs (under --out_dir):
  - CSV:      eval_pairs.csv
              columns:
                traj_i, i, path_i, traj_j, j, path_j,
                known_steps (0/1), step_gap (int or ''),
                pred_pixel_dist (float)
  - PNG:      scatter_steps_vs_pixels.png      (known pairs only)
  - PNG:      hist_unknown_pixel_distance.png  (unknown pairs only)
  - TXT:      stats.txt                        (summary/correlation stats)

Example:
  python image_distance_vs_steps.py \
    --data_root traj_dumps \
    --ckpt out.image_distance/best.ckpt \
    --out_dir out.image_distance/eval_vs_steps \
    --num_pairs 10000 --p_cross 0.2 --max_step_gap 60
"""
from __future__ import annotations
import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import csv
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")  # for headless environments
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Constants (must match training resolution assumptions)
# ---------------------------------------------------------------------
H, W = 240, 224

# ---------------------------------------------------------------------
# Minimal model definition (must match training)
# NOTE: This mirrors image_distance_train.py / image_distance_eval.py.
# ---------------------------------------------------------------------
class ConvEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        h = h.flatten(1)
        z = self.fc(h)
        return F.normalize(z, dim=1)


class DistanceRegressor(nn.Module):
    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.encoder = ConvEncoder(out_dim=emb_dim)
        self.head_vec = nn.Sequential(
            nn.Linear(emb_dim * 3, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        feat = torch.cat([z1, z2, torch.abs(z1 - z2)], dim=1)
        return self.head_vec(feat)  # (B,2)


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------
def list_trajectories(data_root: Path) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for traj_dir in sorted(data_root.glob("traj_*")):
        state_dir = traj_dir / "states"
        if not state_dir.is_dir():
            continue
        paths = sorted(state_dir.glob("state_*.png"), key=lambda p: int(p.stem.split("_")[1]))
        if paths:
            out[traj_dir.name] = paths
    if not out:
        raise FileNotFoundError(f"No trajectories found under {data_root} (expected traj_*/states/state_*.png)")
    return out


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    arr = arr.astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t


def load_frame_as_tensor(p: Path) -> torch.Tensor:
    img = Image.open(p).convert("RGB").resize((W, H), resample=Image.BICUBIC)
    return pil_to_tensor(img)


@torch.no_grad()
def predict_distance(model: DistanceRegressor, device: torch.device,
                     img1_t: torch.Tensor, img2_t: torch.Tensor,
                     bidirectional: bool = True) -> float:
    """Return scalar pixel distance as ‖v‖ where v is the model's (dx,dy)."""
    v_ab = model(img1_t.to(device), img2_t.to(device))   # [1,2]
    d_ab = torch.linalg.norm(v_ab, dim=1)                # [1]
    if bidirectional:
        v_ba = model(img2_t.to(device), img1_t.to(device))
        d_ba = torch.linalg.norm(v_ba, dim=1)
        d = 0.5 * (d_ab + d_ba)
    else:
        d = d_ab
    return float(d.item())


def _rankdata(a: np.ndarray) -> np.ndarray:
    """
    Simple rankdata with average ranks for ties (Spearman helper).
    """
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    # Find run starts
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    run_starts = np.flatnonzero(obs)
    run_ends = np.r_[run_starts[1:], len(a_sorted)]
    ranks = np.empty(len(a), dtype=float)
    for s, e in zip(run_starts, run_ends):
        avg_rank = 0.5*(s + (e-1)) + 1.0  # 1-based average
        ranks[sorter[s:e]] = avg_rank
    return ranks


def compute_correlations(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Returns Pearson r and Spearman rho. If either array is constant or
    insufficient length, values may be NaN. We propagate and report NaNs.
    """
    stats = {"pearson_r": float('nan'), "spearman_rho": float('nan')}
    if x.size < 2 or y.size < 2:
        return stats
    # Pearson
    try:
        r = np.corrcoef(x, y)[0, 1]
    except Exception:
        r = float('nan')
    stats["pearson_r"] = float(r)
    # Spearman
    try:
        rx = _rankdata(x)
        ry = _rankdata(y)
        rho = np.corrcoef(rx, ry)[0, 1]
    except Exception:
        rho = float('nan')
    stats["spearman_rho"] = float(rho)
    return stats


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def eval_vs_steps(
    model: DistanceRegressor,
    device: torch.device,
    data_root: Path,
    out_dir: Path,
    num_pairs: int = 10_000,
    p_cross: float = 0.25,
    max_step_gap: int = 60,
    bidirectional: bool = True,
    seed: int = 0,
) -> None:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    trajs = list_trajectories(data_root)
    traj_items = list(trajs.items())
    traj_names = [k for k, _ in traj_items]

    csv_path = out_dir / "eval_pairs.csv"
    rows = []

    known_x_steps = []  # x-axis (steps) for known pairs
    known_y_pix   = []  # y-axis (pixel distance)
    unknown_pix   = []  # pixel distances for cross-trajectory pairs

    for t in range(num_pairs):
        is_cross = (rng.random() < p_cross) and (len(traj_items) >= 2)
        if not is_cross:
            # sample within one trajectory (known step gap)
            traj_name, paths = rng.choice(traj_items)
            if len(paths) < 2:
                continue
            i = rng.randrange(0, len(paths) - 1)
            # sample positive gap up to max_step_gap (bounded by traj length)
            max_gap_here = min(max_step_gap, len(paths) - 1 - i)
            gap = rng.randint(1, max_gap_here)
            j = i + gap
            p_i, p_j = paths[i], paths[j]
            a = load_frame_as_tensor(p_i).unsqueeze(0)
            b = load_frame_as_tensor(p_j).unsqueeze(0)
            pred = predict_distance(model, device, a, b, bidirectional=bidirectional)
            rows.append({
                "traj_i": traj_name, "i": i, "path_i": str(p_i),
                "traj_j": traj_name, "j": j, "path_j": str(p_j),
                "known_steps": 1, "step_gap": gap,
                "pred_pixel_dist": pred,
            })
            known_x_steps.append(gap)
            known_y_pix.append(pred)
        else:
            # sample across two different trajectories (unknown step gap)
            traj_i, paths_i = rng.choice(traj_items)
            traj_j, paths_j = rng.choice(traj_items)
            # ensure different
            tries = 0
            while traj_j == traj_i and tries < 5:
                traj_j, paths_j = rng.choice(traj_items)
                tries += 1
            if traj_j == traj_i:
                # degenerate fallback: treat as within-traj
                if len(paths_i) < 2:
                    continue
                i = rng.randrange(0, len(paths_i) - 1)
                j = rng.randint(i+1, len(paths_i) - 1)
                p_i, p_j = paths_i[i], paths_i[j]
                a = load_frame_as_tensor(p_i).unsqueeze(0)
                b = load_frame_as_tensor(p_j).unsqueeze(0)
                pred = predict_distance(model, device, a, b, bidirectional=bidirectional)
                rows.append({
                    "traj_i": traj_i, "i": i, "path_i": str(p_i),
                    "traj_j": traj_i, "j": j, "path_j": str(p_j),
                    "known_steps": 1, "step_gap": abs(j-i),
                    "pred_pixel_dist": pred,
                })
                known_x_steps.append(abs(j-i))
                known_y_pix.append(pred)
            else:
                i = rng.randrange(0, len(paths_i))
                j = rng.randrange(0, len(paths_j))
                p_i, p_j = paths_i[i], paths_j[j]
                a = load_frame_as_tensor(p_i).unsqueeze(0)
                b = load_frame_as_tensor(p_j).unsqueeze(0)
                pred = predict_distance(model, device, a, b, bidirectional=bidirectional)
                rows.append({
                    "traj_i": traj_i, "i": i, "path_i": str(p_i),
                    "traj_j": traj_j, "j": j, "path_j": str(p_j),
                    "known_steps": 0, "step_gap": "",  # unknown
                    "pred_pixel_dist": pred,
                })
                unknown_pix.append(pred)

        if (t+1) % max(1, num_pairs // 10) == 0:
            print(f"[eval] {t+1}/{num_pairs}")

    # Write CSV
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "traj_i","i","path_i","traj_j","j","path_j",
            "known_steps","step_gap","pred_pixel_dist"
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"[eval] wrote CSV: {csv_path}")

    # Scatter plot (known only)
    if known_x_steps and known_y_pix:
        sc_path = out_dir / "scatter_steps_vs_pixels.png"
        plt.figure(figsize=(7,5))
        plt.scatter(known_x_steps, known_y_pix, s=8, alpha=0.5)
        plt.xlabel("Trajectory steps (Δt)")
        plt.ylabel("Inferred pixel distance (‖v‖)")
        plt.title(f"Steps vs Pixel Distance (known pairs N={len(known_x_steps)})")
        plt.tight_layout()
        plt.savefig(sc_path, dpi=150)
        plt.close()
        print(f"[eval] wrote scatter: {sc_path}")
    else:
        print("[eval] no known pairs collected; skipping scatter.")

    # Histogram for unknown pixel distances
    if unknown_pix:
        hist_path = out_dir / "hist_unknown_pixel_distance.png"
        plt.figure(figsize=(7,5))
        plt.hist(unknown_pix, bins=60)
        plt.xlabel("Inferred pixel distance (‖v‖)")
        plt.ylabel("Count")
        plt.title(f"Unknown (cross-traj) pairs pixel-distance histogram (N={len(unknown_pix)})")
        plt.tight_layout()
        plt.savefig(hist_path, dpi=150)
        plt.close()
        print(f"[eval] wrote unknown hist: {hist_path}")
    else:
        print("[eval] no unknown pairs; skipping unknown histogram.")

    # Correlation stats on known pairs
    stats_path = out_dir / "stats.txt"
    with stats_path.open("w") as f:
        f.write("Image distance vs action (step) distance\n")
        f.write(f"total_pairs={len(rows)}  known_pairs={len(known_x_steps)}  unknown_pairs={len(unknown_pix)}\n")

        if known_x_steps and known_y_pix:
            x = np.asarray(known_x_steps, dtype=np.float64)
            y = np.asarray(known_y_pix, dtype=np.float64)
            # Basic summaries
            mae = float(np.mean(np.abs(y - x)))
            mse = float(np.mean((y - x) ** 2))
            f.write(f"MAE_known={mae:.6f}\n")
            f.write(f"MSE_known={mse:.6f}\n")
            # Correlations (may be NaN)
            corr = compute_correlations(x, y)
            f.write(f"Pearson_r={corr['pearson_r']}\n")
            f.write(f"Spearman_rho={corr['spearman_rho']}\n")
            # In case either correlation is NaN, explicitly record a NaN count stat
            num_nans = int(np.isnan([corr['pearson_r'], corr['spearman_rho']]).sum())
            f.write(f"Num_NaN_correlation_stats={num_nans}\n")
        else:
            f.write("MAE_known=nan\nMSE_known=nan\nPearson_r=nan\nSpearman_rho=nan\nNum_NaN_correlation_stats=2\n")
    print(f"[eval] wrote stats: {stats_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True, help="Path containing traj_*/states/state_*.png")
    ap.add_argument("--ckpt", type=Path, required=True, help="Path to best.ckpt or last.ckpt")
    ap.add_argument("--out_dir", type=Path, default=Path("out.image_distance/eval_vs_steps"))
    ap.add_argument("--num_pairs", type=int, default=10000)
    ap.add_argument("--p_cross", type=float, default=0.25, help="Probability of sampling a cross-trajectory pair (unknown step gap)")
    ap.add_argument("--max_step_gap", type=int, default=60, help="Max Δ steps when sampling within a trajectory")
    g2 = ap.add_mutually_exclusive_group()
    g2.add_argument("--bidirectional", dest="bidirectional", action="store_true", help="average A→B and B→A predictions")
    g2.add_argument("--no-bidirectional", dest="bidirectional", action="store_false", help="use only A→B predictions")
    ap.set_defaults(bidirectional=True)
    ap.add_argument("--device", type=str, default=None, help="cpu|cuda|mps (auto if None)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # device selection
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    print(f"[Device] {device}")

    # model
    model = DistanceRegressor().to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"[Model] loaded from {args.ckpt}")

    eval_vs_steps(
        model=model,
        device=device,
        data_root=args.data_root,
        out_dir=args.out_dir,
        num_pairs=args.num_pairs,
        p_cross=args.p_cross,
        max_step_gap=args.max_step_gap,
        bidirectional=args.bidirectional,
        seed=args.seed,
    )

    print("[done]")


if __name__ == "__main__":
    main()
