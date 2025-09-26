#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the trained image-distance model.

Outputs:
  1) Augmentation evaluation (known GT):
     - CSV: aug_pairs_errors.csv with columns [mode,gt,pred,abs_err,extra_json]
     - PNG: aug_pairs_hist.png (histogram of abs errors)
  2) Trajectory step evaluation (proxy "action distance" = |Δt|):
     - CSV: traj_step_errors.csv with columns [traj_id,t1,t2,gt_steps,pred,abs_err]
     - PNG: traj_step_hist.png

Usage:
  python eval_image_distance.py --data_root traj_dumps --ckpt runs/image_distance/best.pt
"""
from __future__ import annotations
import json
import math
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from train_image_distance import (
    H, W, pil_to_tensor, PairAugmentor, DistanceRegressor, list_state_images
)

def load_model(ckpt_path: Path, device: torch.device) -> DistanceRegressor:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = DistanceRegressor()
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def load_trajectories(data_root: Path):
    trajs = []
    for traj_dir in sorted(data_root.glob("traj_*")):
        states_dir = traj_dir / "states"
        frames = sorted(states_dir.glob("state_*.png"))
        if frames:
            trajs.append((traj_dir.name, frames))
    if not trajs:
        raise FileNotFoundError(f"No trajectories found under {data_root}.")
    return trajs


def predict_distance(model, device, img1_t: torch.Tensor, img2_t: torch.Tensor) -> float:
    with torch.no_grad():
        img1_t = img1_t.unsqueeze(0).to(device)
        img2_t = img2_t.unsqueeze(0).to(device)
        pred = model(img1_t, img2_t).item()
    return float(pred)


def eval_augmentations(model, device, data_root: Path, n_pairs: int = 5000, seed: int = 0, out_dir: Path = Path(".")):
    rng = np.random.RandomState(seed)
    all_imgs = list_state_images(data_root)
    if not all_imgs:
        raise FileNotFoundError("No images found for augmentation eval.")

    aug = PairAugmentor()
    rows = []
    for i in range(n_pairs):
        p = all_imgs[rng.randint(0, len(all_imgs))]
        img = Image.open(p).convert("RGB")
        img = img.resize((W, H), resample=Image.BICUBIC)
        base = pil_to_tensor(img)

        a, b, gt, info = aug(base)
        pred = predict_distance(model, device, a, b)
        abs_err = abs(pred - gt)
        rows.append((info["mode"], gt, pred, abs_err, json.dumps(info)))

        if (i % 500) == 0:
            print(f"[aug-eval] {i}/{n_pairs}  last abs_err={abs_err:.2f}")

    # save CSV
    out_csv = out_dir / "aug_pairs_errors.csv"
    with out_csv.open("w") as f:
        f.write("mode,gt,pred,abs_err,info\n")
        for mode, gt, pred, abs_err, info in rows:
            f.write(f"{mode},{gt:.6f},{pred:.6f},{abs_err:.6f},{info}\n")
    print(f"[aug-eval] wrote {out_csv}")

    # histogram
    errs = np.array([r[3] for r in rows], dtype=np.float32)
    plt.figure(figsize=(6,4))
    plt.hist(errs, bins=50)
    plt.xlabel("Absolute error (pixels-equivalent)")
    plt.ylabel("Count")
    plt.title("Augmented pairs: distance errors")
    plt.tight_layout()
    out_png = out_dir / "aug_pairs_hist.png"
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[aug-eval] wrote {out_png}")


def eval_traj_steps(model, device, data_root: Path, max_pairs_per_traj: int = 2000, max_dt: int = 10, out_dir: Path = Path(".")):
    trajs = load_trajectories(data_root)
    rows = []
    for traj_name, frames in trajs:
        n = len(frames)
        if n < 2:
            continue
        # sample pairs (t, t+dt) with dt in [1, max_dt]
        sampled = 0
        for t in range(n - 1):
            for dt in range(1, min(max_dt, n - t - 1) + 1):
                p1 = frames[t]
                p2 = frames[t + dt]
                img1 = Image.open(p1).convert("RGB").resize((W, H), resample=Image.BICUBIC)
                img2 = Image.open(p2).convert("RGB").resize((W, H), resample=Image.BICUBIC)
                t1 = pil_to_tensor(img1)
                t2 = pil_to_tensor(img2)
                pred = predict_distance(model, device, t1, t2)
                gt_steps = float(dt)
                abs_err = abs(pred - gt_steps)
                rows.append((traj_name, t, t + dt, gt_steps, pred, abs_err))
                sampled += 1
                if sampled >= max_pairs_per_traj:
                    break
            if sampled >= max_pairs_per_traj:
                break

        print(f"[traj-eval] {traj_name}: sampled {sampled} pairs")

    # save CSV
    out_csv = out_dir / "traj_step_errors.csv"
    with out_csv.open("w") as f:
        f.write("traj_id,t1,t2,gt_steps,pred,abs_err\n")
        for traj_id, t1, t2, gt, pred, err in rows:
            f.write(f"{traj_id},{t1},{t2},{gt:.6f},{pred:.6f},{err:.6f}\n")
    print(f"[traj-eval] wrote {out_csv}")

    # histogram
    errs = np.array([r[5] for r in rows], dtype=np.float32)
    plt.figure(figsize=(6,4))
    plt.hist(errs, bins=50)
    plt.xlabel("Absolute error vs. |Δt|")
    plt.ylabel("Count")
    plt.title("Trajectory step pairs: distance errors")
    plt.tight_layout()
    out_png = out_dir / "traj_step_hist.png"
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[traj-eval] wrote {out_png}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("out/image_distance_eval"))
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--aug_pairs", type=int, default=5000)
    ap.add_argument("--traj_pairs_per_traj", type=int, default=2000)
    ap.add_argument("--max_dt", type=int, default=10)
    return ap.parse_args()


if __name__ == "__main__":
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

    model = load_model(args.ckpt, device)
    eval_augmentations(model, device, args.data_root, n_pairs=args.aug_pairs, out_dir=args.out_dir)
    eval_traj_steps(model, device, args.data_root, max_pairs_per_traj=args.traj_pairs_per_traj, max_dt=args.max_dt, out_dir=args.out_dir)
    print("[done]")