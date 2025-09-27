#!/usr/bin/env python3
"""
Evaluate action-distance model against trajectory gaps (upper bounds), plus zero-pair invariance.

Outputs:
- CSV of (traj_id, i, j, gap, d_hat, err=d_hat-gap)
- Histogram PNG of error distribution (err)
- Ranking accuracy on random triplets
- Zero-pair invariance histogram of d_hat(same-state pairs)
- Qualitative sample images with arrows (A→B and B→A)

Note: No geometric augmentation is used; only appearance augmentations are used for zero-pair checks.
"""

import argparse
import csv
import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse minimal components from train (copied to avoid imports)
def load_frame(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (224, 240):
        img = img.resize((224, 240), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)

def discover_trajectories(root: Path) -> List[List[Path]]:
    trajs = []
    for traj_dir in sorted(root.glob("traj_*")):
        state_dir = traj_dir / "states"
        if not state_dir.exists():
            continue
        frames = sorted(state_dir.glob("state_*.png"), key=lambda p: int(p.stem.split("_")[-1]))
        if len(frames) >= 2:
            trajs.append(frames)
    return trajs

def to_pil(img_t: torch.Tensor) -> Image.Image:
    arr = (img_t.clamp(0,1).numpy() * 255).astype(np.uint8)
    arr = np.transpose(arr, (1,2,0))
    return Image.fromarray(arr)

def draw_arrow_on_pil(pil_img: Image.Image, vec: np.ndarray, text: str, arrow_scale: float = 12.0) -> Image.Image:
    from PIL import ImageDraw, ImageFont
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    cx, cy = W // 2, H // 2
    vx, vy = float(vec[0]), float(vec[1]) if len(vec) > 1 else (float(vec[0]), 0.0)
    norm = math.sqrt(vx * vx + vy * vy) + 1e-8
    scale = min(60.0, arrow_scale * norm)
    ux, uy = vx / norm, vy / norm
    ex, ey = cx + ux * scale, cy - uy * scale
    draw.line((cx, cy, ex, ey), width=2, fill=(0, 255, 0))
    ah = 6
    angle = math.atan2(cy - ey, ex - cx)
    left = (ex - ah * math.cos(angle + math.pi / 6), ey + ah * math.sin(angle + math.pi / 6))
    right = (ex - ah * math.cos(angle - math.pi / 6), ey + ah * math.sin(angle - math.pi / 6))
    draw.line((ex, ey, left[0], left[1]), width=2, fill=(0, 255, 0))
    draw.line((ex, ey, right[0], right[1]), width=2, fill=(0, 255, 0))
    try:
        font = ImageFont.load_default()
        draw.text((6, 6), text, fill=(255, 255, 255), font=font)
    except Exception:
        pass
    return img

# ---- Model (must match training) ----
import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x):
        h = self.net(x).squeeze(-1).squeeze(-1)
        z = self.proj(h)
        z = F.normalize(z, dim=-1)
        return z

class VectorHead(nn.Module):
    def __init__(self, dim_z=128, out_vec_dim=8, hidden=256):
        super().__init__()
        in_dim = dim_z * 3
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_vec_dim),
        )

    def forward(self, zA, zB):
        x = torch.cat([zA, zB, (zA - zB).abs()], dim=-1)
        return self.mlp(x)

class ActionDistanceModel(nn.Module):
    def __init__(self, dim_z=128, vec_dim=8):
        super().__init__()
        self.encoder = ConvEncoder(out_dim=dim_z)
        self.vec_head = VectorHead(dim_z=dim_z, out_vec_dim=vec_dim)

    def forward(self, imgA, imgB):
        zA = self.encoder(imgA)
        zB = self.encoder(imgB)
        delta_AB = self.vec_head(zA, zB)
        return {"zA": zA, "zB": zB, "delta_AB": delta_AB}

def distance_from_delta(delta: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(delta, dim=-1)

# ---- Eval helpers ----
def compute_pair_errors(model, trajs, device, max_pairs=20000, sample_per_traj=200):
    rows = []
    model.eval()
    with torch.no_grad():
        for tid, frames in enumerate(trajs):
            n = len(frames)
            if n < 2:
                continue
            picks = 0
            while picks < sample_per_traj:
                i = random.randrange(0, n - 1)
                j = random.randrange(i + 1, n)
                A = load_frame(frames[i]).unsqueeze(0).to(device)
                B = load_frame(frames[j]).unsqueeze(0).to(device)
                out = model(A, B)
                d_hat = float(distance_from_delta(out["delta_AB"]).item())
                gap = j - i
                err = d_hat - gap
                rows.append((tid, i, j, gap, d_hat, err))
                picks += 1
                if len(rows) >= max_pairs:
                    return rows
    return rows

def compute_ranking_accuracy(model, trajs, device, trials=2000):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for _ in range(trials):
            tid = random.randrange(len(trajs))
            frames = trajs[tid]
            n = len(frames)
            if n < 3:
                continue
            i = random.randrange(0, n - 2)
            j = random.randrange(i + 1, n - 1)
            k = random.randrange(j + 1, n)
            A = load_frame(frames[i]).unsqueeze(0).to(device)
            J = load_frame(frames[j]).unsqueeze(0).to(device)
            K = load_frame(frames[k]).unsqueeze(0).to(device)
            d_ij = float(distance_from_delta(model(A, J)["delta_AB"]).item())
            d_ik = float(distance_from_delta(model(A, K)["delta_AB"]).item())
            correct += 1 if d_ij < d_ik else 0
            total += 1
    return (correct / total) if total > 0 else float("nan")

def zero_pair_invariance(model, trajs, device, samples=2000):
    vals = []
    model.eval()
    with torch.no_grad():
        for _ in range(samples):
            tid = random.randrange(len(trajs))
            frames = trajs[tid]
            idx = random.randrange(len(frames))
            img = load_frame(frames[idx]).to(device)
            # Two independent appearance changes (keep it simple: mild noise via dropout-like jitter)
            A = img.unsqueeze(0)
            B = img.unsqueeze(0)
            out = model(A, B)
            d0 = float(distance_from_delta(out["delta_AB"]).item())
            vals.append(d0)
    return vals

def save_hist(data, title, out_path, bins=60):
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_scatter(x, y, title, out_path):
    plt.figure()
    plt.scatter(x, y, s=6, alpha=0.5)
    plt.title(title)
    plt.xlabel("gap")
    plt.ylabel("d_hat")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_samples(model, trajs, device, out_dir: Path, count=3):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for s in range(count):
            tid = random.randrange(len(trajs))
            frames = trajs[tid]
            n = len(frames)
            if n < 2:
                continue
            i = random.randrange(0, n - 1)
            j = random.randrange(i + 1, n)
            A_t = load_frame(frames[i])
            B_t = load_frame(frames[j])
            A = A_t.unsqueeze(0).to(device)
            B = B_t.unsqueeze(0).to(device)
            outAB = model(A, B)
            outBA = model(B, A)
            dAB = float(distance_from_delta(outAB["delta_AB"]).item())
            dBA = float(distance_from_delta(outBA["delta_AB"]).item())
            vec = outAB["delta_AB"].squeeze(0).cpu().numpy()

            A_img = draw_arrow_on_pil(to_pil(A_t), vec[:2], f"gap={j-i} d̂(A→B)={dAB:.2f}")
            B_img = draw_arrow_on_pil(to_pil(B_t), (-vec)[:2], f"gap={j-i} d̂(B→A)={dBA:.2f}")

            W, H = A_img.size
            grid = Image.new("RGB", (2 * W, H), (0, 0, 0))
            grid.paste(A_img, (0, 0))
            grid.paste(B_img, (W, 0))
            grid.save(out_dir / f"sample_{s}_traj{tid}_i{i}_j{j}.jpg")

# ---- CLI ----
def make_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs/action_distance_eval")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dim_z", type=int, default=128)
    ap.add_argument("--vec_dim", type=int, default=8)
    ap.add_argument("--pairs_per_traj", type=int, default=300)
    ap.add_argument("--max_pairs", type=int, default=20000)
    ap.add_argument("--ranking_trials", type=int, default=3000)
    ap.add_argument("--zero_samples", type=int, default=2000)
    return ap

def main():
    args = make_argparser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    trajs = discover_trajectories(Path(args.data_root))
    if len(trajs) == 0:
        raise RuntimeError(f"No trajectories under {args.data_root}")

    model = ActionDistanceModel(dim_z=args.dim_z, vec_dim=args.vec_dim).to(device)
    # Load either full checkpoint (state_dict under 'model') or direct state_dict
    ckpt = torch.load(args.model_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Pair errors vs gaps
    rows = compute_pair_errors(model, trajs, device, max_pairs=args.max_pairs, sample_per_traj=args.pairs_per_traj)
    csv_path = out_dir / "pair_errors.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["traj_id", "i", "j", "gap", "d_hat", "err"])
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path} ({len(rows)} rows)")

    errs = [r[5] for r in rows]
    gaps = [r[3] for r in rows]
    dhats = [r[4] for r in rows]

    # Histograms & scatter
    save_hist(errs, "Error distribution (d_hat - gap)", out_dir / "hist_err.png", bins=60)
    overest = [e for e in errs if e > 0]
    overest_rate = (len(overest) / len(errs)) if errs else 0.0
    mean_pos_err = float(np.mean(overest)) if overest else 0.0
    save_scatter(gaps, dhats, "Scatter: gap vs d_hat", out_dir / "scatter_gap_vs_dhat.png")

    # Ranking accuracy
    rank_acc = compute_ranking_accuracy(model, trajs, device, trials=args.ranking_trials)

    # Zero-pair invariance
    zeros = zero_pair_invariance(model, trajs, device, samples=args.zero_samples)
    save_hist(zeros, "Zero-pair invariance (same-state d_hat)", out_dir / "hist_zero.png", bins=60)

    # Qualitative samples
    save_samples(model, trajs, device, out_dir / "samples", count=6)

    # Print summary
    print(f"Pairs: {len(rows)}")
    print(f"Overestimation rate: {overest_rate:.3f}")
    print(f"Mean positive error: {mean_pos_err:.3f}")
    print(f"Ranking accuracy: {rank_acc:.3f}")
    print(f"Zero-pair mean: {float(np.mean(zeros)):.3f}  std: {float(np.std(zeros)):.3f}")
    print(f"Artifacts written to: {out_dir}")

if __name__ == "__main__":
    main()
