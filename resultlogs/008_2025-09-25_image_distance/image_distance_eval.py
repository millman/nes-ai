#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate an image-distance model that predicts a motion vector (dx, dy).
Distance is defined as the magnitude ‖(dx,dy)‖.

It runs two evaluations:

1) Augmentation sanity:
   - Create synthetic pairs via random augmentations (noise, translate, scale).
   - Use the known augmentation distance as ground truth.
   - Output:
       * CSV: out_dir/aug_eval_errors.csv  (gt, pred, error, mode, params)
       * PNG: out_dir/aug_eval_hist.png    (histogram of |pred-gt|)
       * Console summary: MAE / MSE

2) Trajectory step distance:
   - For pairs of frames within the same trajectory, define GT as step gap |j-i|.
   - Model predicts distance from raw frames; compare to step gap.
   - Output:
       * CSV: out_dir/traj_eval_errors.csv (gt_steps, pred, error, traj_id, i, j)
       * PNG: out_dir/traj_eval_hist.png   (histogram of |pred-gt|)
       * Console summary: MAE / MSE

Assumptions:
- Frames are at 240x224 (H x W), RGB.
- Data layout: traj_dumps/traj_<n>/states/state_<m>.png
- Checkpoint: produced by image_distance_train.py (top-level keys include 'model').

Example:
python image_distance_eval.py \
  --data_root traj_dumps \
  --ckpt out.image_distance/best.ckpt \
  --out_dir out.image_distance/eval \
  --aug_pairs 5000 \
  --traj_pairs 5000
"""
from __future__ import annotations
import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
H, W = 240, 224

# ---------------------------------------------------------------------
# I/O Utils
# ---------------------------------------------------------------------
def list_trajectories(data_root: Path) -> Dict[str, List[Path]]:
    """Return {traj_name: [sorted state paths]}."""
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
    """Convert PIL->CHW float32 in [0,1]."""
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

# ---------------------------------------------------------------------
# Augmentations (match training semantics)
# ---------------------------------------------------------------------
def add_gaussian_noise(img: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return img
    return torch.clamp(img + torch.randn_like(img) * sigma, 0.0, 1.0)

def affine_translate(img: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """
    Integer translate with noise-filled edges (same as training).
    """
    c, h, w = img.shape
    ix, iy = int(round(dx)), int(round(dy))
    out = torch.roll(img, shifts=(iy, ix), dims=(1, 2))
    if iy > 0:
        out[:, :iy, :] = torch.rand((c, iy, w), device=img.device, dtype=img.dtype)
    elif iy < 0:
        out[:, h+iy:, :] = torch.rand((c, -iy, w), device=img.device, dtype=img.dtype)
    if ix > 0:
        out[:, :, :ix] = torch.rand((c, h, ix), device=img.device, dtype=img.dtype)
    elif ix < 0:
        out[:, :, w+ix:] = torch.rand((c, h, -ix), device=img.device, dtype=img.dtype)
    return out

def scale_around_center(img: torch.Tensor, scale: float) -> torch.Tensor:
    c, h, w = img.shape
    arr = (img.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pil_scaled = pil.resize((new_w, new_h), resample=Image.BICUBIC)
    canvas = Image.fromarray((np.random.rand(h, w, 3) * 255).astype(np.uint8))
    ox = (w - new_w) // 2
    oy = (h - new_h) // 2
    canvas.paste(pil_scaled, (ox, oy))
    return pil_to_tensor(canvas).to(img.device)

def approx_scale_to_pixel_distance(scale: float, coeff: float = 0.5) -> float:
    """
    Heuristic pixel distance from uniform scale delta: |s-1| * coeff * image_diagonal.
    coeff default matches training.
    """
    diag = math.sqrt(H * H + W * W)
    return abs(scale - 1.0) * coeff * diag

class EvalPairAugmentor:
    """
    Generates evaluation pairs with known ground truth distance.
    Mirrors training behavior, including optional two-sided application.
    """
    def __init__(
        self,
        noise_sigma: float = 0.02,
        max_shift: int = 30,
        max_scale_delta: float = 0.20,
        p_noise_only: float = 0.20,
        p_translate: float = 0.50,
        p_scale: float = 0.30,
        scale_coeff: float = 0.5,
        two_sided: bool = True,
    ):
        total = p_noise_only + p_translate + p_scale
        self.p_noise_only = p_noise_only / total
        self.p_translate  = p_translate  / total
        self.p_scale      = p_scale      / total
        self.noise_sigma = noise_sigma
        self.max_shift = max_shift
        self.max_scale_delta = max_scale_delta
        self.scale_coeff = scale_coeff
        self.two_sided = two_sided

    def __call__(self, img: torch.Tensor):
        base1 = add_gaussian_noise(img, self.noise_sigma)
        base2 = add_gaussian_noise(img, self.noise_sigma)
        r = random.random()
        if r < self.p_noise_only:
            img1, img2 = base1, base2
            dist = 0.0
            info = {"mode": "noise_only", "dx": 0, "dy": 0}
        elif r < self.p_noise_only + self.p_translate:
            if self.two_sided:
                dx1 = random.randint(-self.max_shift, self.max_shift)
                dy1 = random.randint(-self.max_shift, self.max_shift)
                dx2 = random.randint(-self.max_shift, self.max_shift)
                dy2 = random.randint(-self.max_shift, self.max_shift)
                img1 = affine_translate(base1, dx1, dy1)
                img2 = affine_translate(base2, dx2, dy2)
                rdx, rdy = dx2 - dx1, dy2 - dy1
                dist = math.sqrt(rdx * rdx + rdy * rdy)
                info = {"mode": "translate", "dx": rdx, "dy": rdy, "dx1": dx1, "dy1": dy1, "dx2": dx2, "dy2": dy2}
            else:
                dx = random.randint(-self.max_shift, self.max_shift)
                dy = random.randint(-self.max_shift, self.max_shift)
                img1, img2 = base1, affine_translate(base2, dx, dy)
                dist = math.sqrt(dx * dx + dy * dy)
                info = {"mode": "translate", "dx": dx, "dy": dy}
        else:
            if self.two_sided:
                s1 = 1.0 + random.uniform(-self.max_scale_delta, self.max_scale_delta)
                s2 = 1.0 + random.uniform(-self.max_scale_delta, self.max_scale_delta)
                img1 = scale_around_center(base1, s1)
                img2 = scale_around_center(base2, s2)
                dist = approx_scale_to_pixel_distance(s2 - (s1 - 1.0), coeff=self.scale_coeff) if False else abs(s2 - s1) * self.scale_coeff * math.sqrt(H*H + W*W)
                info = {"mode": "scale", "scale1": s1, "scale2": s2}
            else:
                s = 1.0 + random.uniform(-self.max_scale_delta, self.max_scale_delta)
                img1, img2 = base1, scale_around_center(base2, s)
                dist = approx_scale_to_pixel_distance(s, coeff=self.scale_coeff)
                info = {"mode": "scale", "scale": s}
        return img1, img2, float(dist), info

# ---------------------------------------------------------------------
# Model (must match training)
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
# Prediction helpers
# ---------------------------------------------------------------------
@torch.no_grad()
def predict_vector(model: DistanceRegressor, device: torch.device,
                   img1_t: torch.Tensor, img2_t: torch.Tensor) -> Tuple[float, float]:
    """Return (dx, dy) predicted for a single pair (expects [1,3,H,W] tensors)."""
    v = model(img1_t.to(device), img2_t.to(device))  # [1,2]
    dx, dy = float(v[0, 0].item()), float(v[0, 1].item())
    return dx, dy

@torch.no_grad()
def predict_distance(model: DistanceRegressor, device: torch.device,
                     img1_t: torch.Tensor, img2_t: torch.Tensor,
                     bidirectional: bool = True) -> float:
    """Return scalar distance as norm of predicted vector; optionally average A->B and B->A."""
    v_ab = model(img1_t.to(device), img2_t.to(device))   # [1,2]
    d_ab = torch.linalg.norm(v_ab, dim=1)               # [1]
    if bidirectional:
        v_ba = model(img2_t.to(device), img1_t.to(device))
        d_ba = torch.linalg.norm(v_ba, dim=1)
        d = 0.5 * (d_ab + d_ba)
    else:
        d = d_ab
    return float(d.item())

# ---------------------------------------------------------------------
# Evaluation routines
# ---------------------------------------------------------------------
def eval_augmentations(model: DistanceRegressor, device: torch.device, data_root: Path,
                       n_pairs: int, out_dir: Path, two_sided: bool = True,
                       bidirectional: bool = True, seed: int = 0) -> None:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    ck = out_dir / "aug_eval_errors.csv"

    # Gather all frames to sample from uniformly
    trajs = list_trajectories(data_root)
    all_paths: List[Path] = [p for lst in trajs.values() for p in lst]
    if not all_paths:
        raise RuntimeError("No frames found for augmentation eval.")

    aug = EvalPairAugmentor(two_sided=two_sided)

    rows = []
    abs_errors = []
    for i in range(n_pairs):
        p = rng.choice(all_paths)
        base = load_frame_as_tensor(p).unsqueeze(0)
        img1, img2, gt, info = aug(base[0])

        a = img1.unsqueeze(0)
        b = img2.unsqueeze(0)
        pred = predict_distance(model, device, a, b, bidirectional=bidirectional)
        err = abs(pred - gt)
        abs_errors.append(err)

        row = {
            "gt": gt,
            "pred": pred,
            "abs_error": err,
            "mode": info.get("mode", "?"),
            "dx": info.get("dx", ""),
            "dy": info.get("dy", ""),
            "dx1": info.get("dx1", ""),
            "dy1": info.get("dy1", ""),
            "dx2": info.get("dx2", ""),
            "dy2": info.get("dy2", ""),
            "scale": info.get("scale", ""),
            "scale1": info.get("scale1", ""),
            "scale2": info.get("scale2", ""),
            "base_path": str(p),
        }
        rows.append(row)
        if (i+1) % max(1, n_pairs // 10) == 0:
            print(f"[aug] {i+1}/{n_pairs}   running MAE={np.mean(abs_errors):.3f}")

    # Write CSV
    import csv
    with ck.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[aug] wrote CSV: {ck}")

    # Histogram
    hist_path = out_dir / "aug_eval_hist.png"
    plt.figure()
    plt.hist(abs_errors, bins=50)
    plt.xlabel("|pred - gt| (pixels)")
    plt.ylabel("count")
    plt.title(f"Augmentation distance errors (N={len(abs_errors)})")
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()
    print(f"[aug] wrote hist: {hist_path}")

    # Summary
    mae = float(np.mean(abs_errors))
    mse = float(np.mean(np.square(abs_errors)))
    print(f"[aug] MAE={mae:.3f}  MSE={mse:.3f}")

def eval_trajectories(model: DistanceRegressor, device: torch.device, data_root: Path,
                      n_pairs: int, out_dir: Path, max_step_gap: int = 20,
                      bidirectional: bool = True, seed: int = 0) -> None:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    ck = out_dir / "traj_eval_errors.csv"

    trajs = list_trajectories(data_root)
    traj_items = list(trajs.items())

    rows = []
    abs_errors = []
    for i in range(n_pairs):
        traj_name, paths = rng.choice(traj_items)
        if len(paths) < 2:
            continue
        # choose a pair within a limited step gap
        i0 = rng.randrange(0, len(paths))
        gap = rng.randint(1, min(max_step_gap, len(paths)-1))
        j0 = min(len(paths)-1, i0 + gap)
        if j0 == i0:
            j0 = min(len(paths)-1, i0+1)

        p1, p2 = paths[i0], paths[j0]
        a = load_frame_as_tensor(p1).unsqueeze(0)
        b = load_frame_as_tensor(p2).unsqueeze(0)
        gt_steps = abs(j0 - i0)

        pred = predict_distance(model, device, a, b, bidirectional=bidirectional)
        err = abs(pred - gt_steps)
        abs_errors.append(err)
        rows.append({
            "traj": traj_name,
            "i": i0,
            "j": j0,
            "gt_steps": gt_steps,
            "pred": pred,
            "abs_error": err,
            "path_i": str(p1),
            "path_j": str(p2),
        })

        if (i+1) % max(1, n_pairs // 10) == 0:
            print(f"[traj] {i+1}/{n_pairs}   running MAE={np.mean(abs_errors):.3f}")

    # CSV
    import csv
    with ck.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[traj] wrote CSV: {ck}")

    # Histogram
    hist_path = out_dir / "traj_eval_hist.png"
    plt.figure()
    plt.hist(abs_errors, bins=50)
    plt.xlabel("|pred - step_gap|")
    plt.ylabel("count")
    plt.title(f"Trajectory step-gap errors (N={len(abs_errors)})")
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()
    print(f"[traj] wrote hist: {hist_path}")

    mae = float(np.mean(abs_errors))
    mse = float(np.mean(np.square(abs_errors)))
    print(f"[traj] MAE={mae:.3f}  MSE={mse:.3f}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True, help="Path to best.ckpt or last.ckpt")
    ap.add_argument("--out_dir", type=Path, default=Path("out.image_distance/eval"))

    ap.add_argument("--aug_pairs", type=int, default=5000, help="# synthetic augmentation pairs")
    ap.add_argument("--traj_pairs", type=int, default=5000, help="# trajectory frame pairs")
    ap.add_argument("--max_step_gap", type=int, default=20, help="max Δ steps for trajectory sampling")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--two_sided", dest="two_sided", action="store_true", help="apply transforms to both views in augmentation eval")
    g.add_argument("--one_sided", dest="two_sided", action="store_false", help="apply transforms only to second view")
    ap.set_defaults(two_sided=True)

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

    # device
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

    # 1) Augmentation sanity
    aug_dir = args.out_dir / "augmentations"
    aug_dir.mkdir(parents=True, exist_ok=True)
    eval_augmentations(model, device, args.data_root, n_pairs=args.aug_pairs,
                       out_dir=aug_dir, two_sided=args.two_sided,
                       bidirectional=args.bidirectional, seed=args.seed)

    # 2) Trajectory step-gap eval
    traj_dir = args.out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    eval_trajectories(model, device, args.data_root, n_pairs=args.traj_pairs,
                      out_dir=traj_dir, max_step_gap=args.max_step_gap,
                      bidirectional=args.bidirectional, seed=args.seed)

    print("[done]")

if __name__ == "__main__":
    main()
