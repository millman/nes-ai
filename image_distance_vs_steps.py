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

Additionally, this script SAVES VISUAL SAMPLES of the evaluated pairs:
- Side-by-side (A | B) images
- On A: draw the predicted vector A→B
- On B: draw the predicted vector B→A
- Text overlays include |v_ab|, |v_ba|, mean |v|, and the trajectory step gap (or "unknown")

Outputs (under --out_dir):
  - CSV:      eval_pairs.csv
              columns:
                traj_i, i, path_i, traj_j, j, path_j,
                known_steps (0/1), step_gap (int or ''),
                pred_pixel_dist (float), v_ab_x, v_ab_y, v_ba_x, v_ba_y
  - PNG:      scatter_steps_vs_pixels.png      (known pairs only)
  - PNG:      hist_unknown_pixel_distance.png  (unknown pairs only)
  - TXT:      stats.txt                        (summary/correlation stats)
  - DIR:      samples/known/*.png              (sampled known-gap pairs with overlays)
  - DIR:      samples/unknown/*.png            (sampled cross-traj pairs with overlays)

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
from PIL import Image, ImageDraw, ImageFont

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
# Model definition (must match training)
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
    with Image.open(p) as img:
        img = img.convert("RGB").resize((W, H), resample=Image.BICUBIC)
        return pil_to_tensor(img)


def load_frame_both(p: Path) -> Tuple[Image.Image, torch.Tensor]:
    """Load, resize to (W,H), return (PIL_RGB, Tensor[1,3,H,W])."""
    with Image.open(p) as img:
        img = img.convert("RGB").resize((W, H), resample=Image.BICUBIC)
        t = pil_to_tensor(img).unsqueeze(0)
        return img, t


# ---------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------
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


@torch.no_grad()
def predict_vectors(model: DistanceRegressor, device: torch.device,
                    a_t: torch.Tensor, b_t: torch.Tensor,
                    bidirectional: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns (v_ab, v_ba) as numpy arrays of shape (2,), where v = (dx, dy).
    If bidirectional=False, v_ba is None.
    """
    v_ab = model(a_t.to(device), b_t.to(device))[0].detach().cpu().numpy()
    if bidirectional:
        v_ba = model(b_t.to(device), a_t.to(device))[0].detach().cpu().numpy()
    else:
        v_ba = None
    return v_ab, v_ba


def vector_norm(v: np.ndarray) -> float:
    return float(np.sqrt((v**2).sum()))


# ---------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------
def draw_arrow(draw: ImageDraw.ImageDraw, start: Tuple[float, float], end: Tuple[float, float],
               width: int = 3, head_len: float = 10.0, head_width: float = 6.0):
    """Draw an arrow from start to end on a PIL ImageDraw context."""
    draw.line([start, end], width=width, fill=(255, 255, 0))
    # Arrow head
    dx, dy = (end[0] - start[0], end[1] - start[1])
    L = math.hypot(dx, dy) + 1e-6
    ux, uy = dx / L, dy / L
    # Base of head
    hx, hy = end[0] - ux * head_len, end[1] - uy * head_len
    # Perp
    px, py = -uy, ux
    left = (hx + px * head_width, hy + py * head_width)
    right = (hx - px * head_width, hy - py * head_width)
    draw.polygon([end, left, right], fill=(255, 255, 0))


def make_sample_image(img_a: Image.Image,
                      img_b: Image.Image,
                      v_ab: Optional[np.ndarray],
                      v_ba: Optional[np.ndarray],
                      gap_text: str,
                      save_path: Path,
                      scale: float = 40.0):
    """
    Compose A|B side-by-side. On A draw v_ab; on B draw v_ba.
    Overlay text with |v_ab|, |v_ba|, mean|v|, and Δt info.
    """
    pad = 6
    canvas = Image.new("RGB", (W*2 + pad, H), (20,20,20))
    canvas.paste(img_a, (0,0))
    canvas.paste(img_b, (W+pad, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    cx_a, cy_a = W/2, H/2
    cx_b, cy_b = W + pad + W/2, H/2

    lines = []
    if v_ab is not None:
        mag_ab = vector_norm(v_ab)
        end_a = (cx_a + v_ab[0]*scale, cy_a + v_ab[1]*scale)
        draw_arrow(draw, (cx_a, cy_a), end_a, width=3)
        lines.append(f"|v_ab|={mag_ab:.3f}")
    else:
        lines.append("|v_ab|=n/a")

    if v_ba is not None:
        mag_ba = vector_norm(v_ba)
        end_b = (cx_b + v_ba[0]*scale, cy_b + v_ba[1]*scale)
        draw_arrow(draw, (cx_b, cy_b), end_b, width=3)
        lines.append(f"|v_ba|={mag_ba:.3f}")
    else:
        lines.append("|v_ba|=n/a")

    # Mean magnitude
    mags = []
    if v_ab is not None: mags.append(vector_norm(v_ab))
    if v_ba is not None: mags.append(vector_norm(v_ba))
    mean_mag = (sum(mags)/len(mags)) if mags else float('nan')
    lines.append(f"|v|_mean={mean_mag:.3f}")
    lines.append(f"Δt={gap_text}")

    # Text box
    text = "  ".join(lines)
    draw.rectangle([2, 2, 2 + draw.textlength(text, font=font) + 6, 22], fill=(0,0,0))
    draw.text((5,5), text, fill=(255,255,255), font=font)

    # Labels
    draw.text((5, H-18), "A", fill=(255,255,0), font=font)
    draw.text((W+pad+5, H-18), "B", fill=(255,255,0), font=font)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


# ---------------------------------------------------------------------
# Correlation utils
# ---------------------------------------------------------------------
def _rankdata(a: np.ndarray) -> np.ndarray:
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    run_starts = np.flatnonzero(obs)
    run_ends = np.r_[run_starts[1:], len(a_sorted)]
    ranks = np.empty(len(a), dtype=float)
    for s, e in zip(run_starts, run_ends):
        avg_rank = 0.5*(s + (e-1)) + 1.0  # 1-based average
        ranks[sorter[s:e]] = avg_rank
    return ranks


def compute_correlations(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    stats = {"pearson_r": float('nan'), "spearman_rho": float('nan')}
    if x.size < 2 or y.size < 2:
        return stats
    try:
        r = np.corrcoef(x, y)[0, 1]
    except Exception:
        r = float('nan')
    stats["pearson_r"] = float(r)
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
    save_samples: int = 200,
    seed: int = 0,
) -> None:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    trajs = list_trajectories(data_root)
    traj_items = list(trajs.items())

    csv_path = out_dir / "eval_pairs.csv"
    rows = []

    known_x_steps = []  # steps
    known_y_pix   = []  # inferred pixel distance (mean magnitude)
    unknown_pix   = []  # inferred pixel distance (mean magnitude for cross-traj)

    # sample saving
    save_known_dir = out_dir / "samples" / "known"
    save_unknown_dir = out_dir / "samples" / "unknown"
    n_saved_known = 0
    n_saved_unknown = 0

    for t in range(num_pairs):
        is_cross = (rng.random() < p_cross) and (len(traj_items) >= 2)
        if not is_cross:
            traj_name, paths = rng.choice(traj_items)
            if len(paths) < 2:
                continue
            i = rng.randrange(0, len(paths) - 1)
            max_gap_here = min(max_step_gap, len(paths) - 1 - i)
            gap = rng.randint(1, max_gap_here)
            j = i + gap
            p_i, p_j = paths[i], paths[j]

            img_a, a_t = load_frame_both(p_i)
            img_b, b_t = load_frame_both(p_j)

            v_ab, v_ba = predict_vectors(model, device, a_t, b_t, bidirectional=True)
            mags = [vector_norm(v_ab)]
            if v_ba is not None: mags.append(vector_norm(v_ba))
            pred_mean = float(sum(mags)/len(mags))

            rows.append({
                "traj_i": traj_name, "i": i, "path_i": str(p_i),
                "traj_j": traj_name, "j": j, "path_j": str(p_j),
                "known_steps": 1, "step_gap": gap,
                "pred_pixel_dist": pred_mean,
                "v_ab_x": v_ab[0], "v_ab_y": v_ab[1],
                "v_ba_x": (v_ba[0] if v_ba is not None else ""),
                "v_ba_y": (v_ba[1] if v_ba is not None else ""),
            })
            known_x_steps.append(gap)
            known_y_pix.append(pred_mean)

            if n_saved_known < save_samples:
                fname = f"{t:06d}_{traj_name}_{i:06d}__{traj_name}_{j:06d}.png"
                make_sample_image(
                    img_a, img_b, v_ab, v_ba, gap_text=str(gap),
                    save_path=save_known_dir / fname, scale=40.0
                )
                n_saved_known += 1

        else:
            traj_i, paths_i = rng.choice(traj_items)
            traj_j, paths_j = rng.choice(traj_items)
            # ensure different
            tries = 0
            while traj_j == traj_i and tries < 5:
                traj_j, paths_j = rng.choice(traj_items)
                tries += 1
            if traj_j == traj_i:
                # fallback: treat as within-traj
                if len(paths_i) < 2:
                    continue
                i = rng.randrange(0, len(paths_i) - 1)
                j = rng.randint(i+1, len(paths_i) - 1)
                p_i, p_j = paths_i[i], paths_i[j]

                img_a, a_t = load_frame_both(p_i)
                img_b, b_t = load_frame_both(p_j)

                v_ab, v_ba = predict_vectors(model, device, a_t, b_t, bidirectional=True)
                mags = [vector_norm(v_ab)]
                if v_ba is not None: mags.append(vector_norm(v_ba))
                pred_mean = float(sum(mags)/len(mags))

                rows.append({
                    "traj_i": traj_i, "i": i, "path_i": str(p_i),
                    "traj_j": traj_i, "j": j, "path_j": str(p_j),
                    "known_steps": 1, "step_gap": abs(j-i),
                    "pred_pixel_dist": pred_mean,
                    "v_ab_x": v_ab[0], "v_ab_y": v_ab[1],
                    "v_ba_x": (v_ba[0] if v_ba is not None else ""),
                    "v_ba_y": (v_ba[1] if v_ba is not None else ""),
                })
                known_x_steps.append(abs(j-i))
                known_y_pix.append(pred_mean)

                if n_saved_known < save_samples:
                    fname = f"{t:06d}_{traj_i}_{i:06d}__{traj_i}_{j:06d}.png"
                    make_sample_image(
                        img_a, img_b, v_ab, v_ba, gap_text=str(abs(j-i)),
                        save_path=save_known_dir / fname, scale=40.0
                    )
                    n_saved_known += 1
            else:
                i = rng.randrange(0, len(paths_i))
                j = rng.randrange(0, len(paths_j))
                p_i, p_j = paths_i[i], paths_j[j]

                img_a, a_t = load_frame_both(p_i)
                img_b, b_t = load_frame_both(p_j)

                v_ab, v_ba = predict_vectors(model, device, a_t, b_t, bidirectional=True)
                mags = [vector_norm(v_ab)]
                if v_ba is not None: mags.append(vector_norm(v_ba))
                pred_mean = float(sum(mags)/len(mags))

                rows.append({
                    "traj_i": traj_i, "i": i, "path_i": str(p_i),
                    "traj_j": traj_j, "j": j, "path_j": str(p_j),
                    "known_steps": 0, "step_gap": "",
                    "pred_pixel_dist": pred_mean,
                    "v_ab_x": v_ab[0], "v_ab_y": v_ab[1],
                    "v_ba_x": (v_ba[0] if v_ba is not None else ""),
                    "v_ba_y": (v_ba[1] if v_ba is not None else ""),
                })
                unknown_pix.append(pred_mean)

                if n_saved_unknown < save_samples:
                    fname = f"{t:06d}_{traj_i}_{i:06d}__{traj_j}_{j:06d}.png"
                    make_sample_image(
                        img_a, img_b, v_ab, v_ba, gap_text="unknown",
                        save_path=save_unknown_dir / fname, scale=40.0
                    )
                    n_saved_unknown += 1

        if (t+1) % max(1, num_pairs // 10) == 0:
            print(f"[eval] {t+1}/{num_pairs}")

    # Write CSV
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "traj_i","i","path_i","traj_j","j","path_j",
            "known_steps","step_gap","pred_pixel_dist",
            "v_ab_x","v_ab_y","v_ba_x","v_ba_y"
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
        plt.ylabel("Inferred pixel distance (‖v‖ mean)")
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
        plt.xlabel("Inferred pixel distance (‖v‖ mean)")
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
        f.write("Image distance vs action (step) distance")
        f.write(f"total_pairs={len(rows)}  known_pairs={len(known_x_steps)}  unknown_pairs={len(unknown_pix)}")

        if known_x_steps and known_y_pix:
            x = np.asarray(known_x_steps, dtype=np.float64)
            y = np.asarray(known_y_pix, dtype=np.float64)
            mae = float(np.mean(np.abs(y - x)))
            mse = float(np.mean((y - x) ** 2))
            f.write(f"MAE_known={mae:.6f}")
            f.write(f"MSE_known={mse:.6f}")
            corr = compute_correlations(x, y)
            f.write(f"Pearson_r={corr['pearson_r']}")
            f.write(f"Spearman_rho={corr['spearman_rho']}")
            num_nans = int(np.isnan([corr['pearson_r'], corr['spearman_rho']]).sum())
            f.write(f"Num_NaN_correlation_stats={num_nans}")
        else:
            f.write("MAE_known=nan MSE_known=nan Pearson_r=nan Spearman_rho=nan Num_NaN_correlation_stats=2")
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
    ap.add_argument("--save_samples", type=int, default=200, help="Max samples to save for known and unknown each")
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
        save_samples=args.save_samples,
        seed=args.seed,
    )

    print("[done]")


if __name__ == "__main__":
    main()
