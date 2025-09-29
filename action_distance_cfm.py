#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Action-distance (CFM) toy trainer with a tiny encoder/decoder for visualization.

Data layout (same as your other scripts):
  traj_dumps/
    traj_0/states/state_00000.png
    traj_0/states/state_00001.png
    ...
    traj_1/states/state_00000.png
    ...

What it does:
  - Encodes two frames A,B -> (zA, zB)
  - Samples t ~ U[0,1] and forms a *line* latent: z_t = (1 - t) zA + t zB
  - Trains a conditional vector field vθ(z_t, t | cond(A,B)) to match the *target* d/dt z_t = zB - zA
      L_CFM = E[ || vθ( z_t, t, cond ) - (zB - zA) ||^2 ]
  - Adds a tiny autoencoder reconstruction loss on A and B for sanity/viz:
      L_rec = ||dec(zA) - A||_1 + ||dec(zB) - B||_1
  - Optional latent regularization (small) to keep scales tame.
  - Debug viz: decodes interpolants between A and B and saves a grid.
    Also shows an Euler rollout using the learned vector field (optional).

You can treat ||zB - zA|| as a first "action-distance" proxy if you want a scalar,
or later attach a small head to map that to predicted step gaps.

CLI:
  python action_distance_cfm.py --data_root traj_dumps --out_dir out.action_distance_cfm
"""

from __future__ import annotations
import argparse, math, random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler  # <-- add near your imports
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
H, W = 240, 224  # (height, width)

# --------------------------------------------------------------------------------------
# IO Utils
# --------------------------------------------------------------------------------------
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
        raise FileNotFoundError(f"No trajectories under {data_root} (expected traj_*/states/state_*.png)")
    return out

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    arr = arr.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2,0,1).contiguous()

def load_frame_as_tensor(p: Path) -> torch.Tensor:
    img = Image.open(p).convert("RGB").resize((W, H), resample=Image.BICUBIC)
    return pil_to_tensor(img)

# --------------------------------------------------------------------------------------
# Dataset that yields pairs (A,B) from (usually) the same trajectory
# --------------------------------------------------------------------------------------
class PairFromTrajDataset(Dataset):
    def __init__(self, data_root: Path, split: str = "train", train_frac: float = 0.95, seed: int = 0,
                 max_step_gap: int = 20, p_cross_traj: float = 0.1):
        super().__init__()
        self.trajs = list_trajectories(data_root)
        self.traj_items = list(self.trajs.items())
        all_paths = [p for lst in self.trajs.values() for p in lst]
        rng = random.Random(seed)
        rng.shuffle(all_paths)
        n_train = int(round(len(all_paths) * train_frac))
        self.pool = all_paths[:n_train] if split == "train" else all_paths[n_train:]
        self.split = split
        self.rng = rng
        self.max_step_gap = max_step_gap
        self.p_cross_traj = p_cross_traj

    def __len__(self):
        return max(1, len(self.pool))

    def _sample_same_traj_pair(self) -> Tuple[Path, Path]:
        traj_name, paths = self.rng.choice(self.traj_items)
        if len(paths) < 2:
            return paths[0], paths[-1]
        i0 = self.rng.randrange(0, len(paths))
        gap = self.rng.randint(1, min(max(1,self.max_step_gap), len(paths)-1))
        j0 = min(len(paths)-1, i0 + gap)
        if j0 == i0:
            j0 = min(len(paths)-1, i0+1)
        return paths[i0], paths[j0]

    def _sample_cross_traj_pair(self) -> Tuple[Path, Path]:
        (t1, p1s) = self.rng.choice(self.traj_items)
        (t2, p2s) = self.rng.choice(self.traj_items)
        p1 = self.rng.choice(p1s)
        p2 = self.rng.choice(p2s)
        return p1, p2

    def __getitem__(self, idx: int):
        if self.rng.random() < self.p_cross_traj:
            p1, p2 = self._sample_cross_traj_pair()
        else:
            p1, p2 = self._sample_same_traj_pair()
        a = load_frame_as_tensor(p1)
        b = load_frame_as_tensor(p2)
        return a, b, str(p1), str(p2)

# --------------------------------------------------------------------------------------
# Small Encoder/Decoder
# --------------------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, z_dim: int = 128):
        super().__init__()
        # (3,240,224) -> (z)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),   # 120x112
            nn.Conv2d(32,64, 3, stride=2, padding=1), nn.ReLU(inplace=True),   # 60x56
            nn.Conv2d(64,128,3, stride=2, padding=1), nn.ReLU(inplace=True),   # 30x28
            nn.Conv2d(128,256,3,stride=2, padding=1), nn.ReLU(inplace=True),   # 15x14
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(256, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).flatten(1)
        z = self.fc(h)
        return z  # do NOT normalize; we want meaningful scale for interpolation

class Decoder(nn.Module):
    def __init__(self, z_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256,128, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),  # 2x2
            nn.ConvTranspose2d(128,64,  kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),  # 4x4
            nn.ConvTranspose2d(64, 64,  kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),  # 8x8
            nn.ConvTranspose2d(64, 32,  kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),  # 16x16
            nn.ConvTranspose2d(32, 16,  kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),  # 32x32
        )
        # From 32x32 to 240x224 with upsampling + convs (keeps it simple & stable)
        self.final = nn.Sequential(
            nn.Upsample(size=(60,56), mode="bilinear", align_corners=False),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(size=(120,112), mode="bilinear", align_corners=False),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(size=(240,224), mode="bilinear", align_corners=False),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1,256,1,1)
        h = self.up(h)
        x = self.final(h)
        return x

# --------------------------------------------------------------------------------------
# Conditional Vector Field for CFM
# vθ(z_t, t | cond(A,B)) -> R^{z_dim}
# We'll condition using [z_t, (zB - zA), (zA + zB)/2, t].
# --------------------------------------------------------------------------------------
class VectorField(nn.Module):
    def __init__(self, z_dim: int = 128):
        super().__init__()
        in_dim = z_dim * 3 + 1  # zt, delta, mid, and scalar t
        hidden = 512
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, zt: torch.Tensor, t: torch.Tensor, zA: torch.Tensor, zB: torch.Tensor) -> torch.Tensor:
        delta = zB - zA
        mid   = 0.5 * (zA + zB)
        feat  = torch.cat([zt, delta, mid, t[:, None]], dim=1)
        return self.net(feat)

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().clamp(0,1)
    arr = (t.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)

# --------------------------------------------------------------------------------------
# Visualization: decode linear interpolants (and optional flow rollouts)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def save_interpolation_grid(
    enc: Encoder, dec: Decoder, vf: VectorField,
    A: torch.Tensor, B: torch.Tensor,
    out_path: Path, device: torch.device,
    cols: int = 9, euler_rollout: bool = True, rollout_steps: Optional[int] = None
):
    """
    For the first N rows in batch:
      Row shows: [A] [dec(zA)] [dec(z_t1)] ... [dec(z_tK)] [dec(zB)] [B]
      Also an Euler rollout using vθ starting at zA over K steps (optional).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Bsz = min(A.shape[0], 8)
    K = max(2, cols)  # number of *internal* samples between endpoints
    steps = K
    if rollout_steps is None:
        rollout_steps = K

    A = A[:Bsz].to(device); B = B[:Bsz].to(device)
    zA = enc(A); zB = enc(B)

    # linear interpolants
    ts = torch.linspace(0, 1, steps+2, device=device)  # include 0 and 1
    z_seq = []
    for t in ts:
        zt = (1.0 - t) * zA + t * zB
        z_seq.append(dec(zt))
    # Euler rollout with learned field (optional)
    rollout_seq = []
    if euler_rollout:
        for b in range(Bsz):
            z = zA[b:b+1].clone()
            rollout_imgs = [dec(z)]
            for s in range(steps):
                t = torch.full((1,), (s+1)/(steps+1), device=device)
                v = vf(z, t, zA[b:b+1], zB[b:b+1])
                z = z + (1.0/(steps+1)) * v  # simple Euler step on unit-time interval
                rollout_imgs.append(dec(z))
            rollout_imgs.append(dec(zB[b:b+1]))
            rollout_seq.append(torch.cat(rollout_imgs, dim=0))  # (steps+2, 3,H,W)

    # Compose canvas
    # Each row: A | dec(zA) | interp_1 .. interp_K | dec(zB) | B  (=> steps+4 tiles)
    tiles_per_row = steps + 4
    tile_w, tile_h = W, H
    total_w = tile_w * tiles_per_row
    total_h = tile_h * (Bsz + (Bsz if euler_rollout else 0))
    canvas = Image.new("RGB", (total_w, total_h), (0,0,0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    #test
    # Put linear interpolation rows
    for r in range(Bsz):
        x = 0; y = r * tile_h
        canvas.paste(_to_pil(A[r]), (x, y)); x += tile_w
        canvas.paste(_to_pil(dec(zA[r:r+1])[0]), (x, y)); x += tile_w
        for s in range(steps):
            canvas.paste(_to_pil(z_seq[s+1][r]), (x, y)); x += tile_w
        canvas.paste(_to_pil(dec(zB[r:r+1])[0]), (x, y)); x += tile_w
        canvas.paste(_to_pil(B[r]), (x, y))

        # annotate
        txt = f"||zB-zA||={torch.linalg.norm(zB[r]-zA[r]).item():.2f}"
        if font:
            bbox = draw.textbbox((0, 0), txt, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            tw, th = len(txt) * 6, 10
        draw.rectangle([2, y+2, 2+tw+2, y+2+th+2], fill=(0,0,0))
        draw.text((4, y+4), txt, font=font, fill=(255,255,255))

    # Put rollout rows
    if euler_rollout:
        base_y = Bsz * tile_h
        for r in range(Bsz):
            x = 0; y = base_y + r * tile_h
            # A and dec(zA)
            canvas.paste(_to_pil(A[r]), (x, y)); x += tile_w
            canvas.paste(_to_pil(dec(zA[r:r+1])[0]), (x, y)); x += tile_w
            seq = rollout_seq[r]  # (steps+2, 3,H,W)
            for s in range(1, steps+1):
                canvas.paste(_to_pil(seq[s]), (x, y)); x += tile_w
            canvas.paste(_to_pil(dec(zB[r:r+1])[0]), (x, y)); x += tile_w
            canvas.paste(_to_pil(B[r]), (x, y))

            txt = "Euler rollout via vθ"
            if font:
                bbox = draw.textbbox((0, 0), txt, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            else:
                tw, th = len(txt) * 6, 10
            draw.rectangle([2, y+2, 2+tw+2, y+2+th+2], fill=(0,0,0))
            draw.text((4, y+4), txt, font=font, fill=(200,255,200))

    canvas.save(out_path)

# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
@dataclass
class TrainCfg:
    data_root: Path
    out_dir: Path = Path("out.action_distance_cfm")
    z_dim: int = 128
    batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-3
    seed: int = 0
    num_workers: int = 4
    device: Optional[str] = None
    max_step_gap: int = 20
    p_cross_traj: float = 0.1

    # losses
    lambda_cfm: float = 1.0
    lambda_rec: float = 0.1
    lambda_latent_l2: float = 1e-4

    # viz/debug
    viz_every: int = 500
    viz_rows: int = 6
    viz_cols: int = 9
    save_every_ep: bool = True

def pick_device(pref: Optional[str]) -> torch.device:
    if pref:
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train(cfg: TrainCfg):
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "checkpoints").mkdir(exist_ok=True, parents=True)
    (cfg.out_dir / "viz").mkdir(exist_ok=True, parents=True)

    ds_tr = PairFromTrajDataset(cfg.data_root, "train", 0.95, cfg.seed, cfg.max_step_gap, cfg.p_cross_traj)
    ds_va = PairFromTrajDataset(cfg.data_root, "val",   0.95, cfg.seed, cfg.max_step_gap, cfg.p_cross_traj)

    use_pin = (device.type == "cuda")  # no pinning on cpu/mps
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, pin_memory=use_pin, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=use_pin, drop_last=False)
    print(f"[Data] train pairs≈{len(ds_tr)}  val pairs≈{len(ds_va)}")

    enc = Encoder(cfg.z_dim).to(device)
    dec = Decoder(cfg.z_dim).to(device)
    vf  = VectorField(cfg.z_dim).to(device)

    params = list(enc.parameters()) + list(dec.parameters()) + list(vf.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    global_step = 0
    best_val = float("inf")

    for ep in range(1, cfg.epochs+1):
        enc.train(); dec.train(); vf.train()
        run_loss_cfm = run_loss_rec = run_n = 0.0

        for A, B, _, _ in dl_tr:
            A = A.to(device, non_blocking=True)
            B = B.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                zA = enc(A)
                zB = enc(B)

                # Sample t and construct z_t (line between zA and zB)
                t = torch.rand(A.shape[0], device=device)
                zt = (1.0 - t[:, None]) * zA + t[:, None] * zB

                # Target velocity for linear interpolation is constant: d/dt z_t = zB - zA
                u = (zB - zA)

                # Predict vector field
                v = vf(zt, t, zA, zB)

                # Losses
                loss_cfm = F.mse_loss(v, u)  # core CFM objective
                xA_rec = dec(zA)
                xB_rec = dec(zB)
                loss_rec = F.l1_loss(xA_rec, A) + F.l1_loss(xB_rec, B)

                # small latent l2 to stabilize scale
                loss_lat = 0.5 * (zA.pow(2).mean() + zB.pow(2).mean())

                loss = cfg.lambda_cfm * loss_cfm + cfg.lambda_rec * loss_rec + cfg.lambda_latent_l2 * loss_lat

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            run_loss_cfm += loss_cfm.item() * A.shape[0]
            run_loss_rec += loss_rec.item() * A.shape[0]
            run_n += A.shape[0]
            global_step += 1

            if (cfg.viz_every > 0) and (global_step % cfg.viz_every == 0):
                enc.eval(); dec.eval(); vf.eval()
                save_interpolation_grid(enc, dec, vf, A.cpu(), B.cpu(),
                                        cfg.out_dir / "viz" / f"ep{ep:02d}_step{global_step:06d}.png",
                                        device=device, cols=cfg.viz_cols, euler_rollout=True)
                enc.train(); dec.train(); vf.train()

        tr_cfm = run_loss_cfm / max(1, run_n)
        tr_rec = run_loss_rec / max(1, run_n)
        print(f"[ep {ep:02d}] train: Lcfm={tr_cfm:.4f}  Lrec={tr_rec:.4f}")

        # ---- Validation (CFM only + recon) ----
        enc.eval(); dec.eval(); vf.eval()
        va_cfm = va_rec = va_n = 0.0
        with torch.no_grad():
            for A, B, _, _ in dl_va:
                A = A.to(device); B = B.to(device)
                zA = enc(A); zB = enc(B)
                t = torch.rand(A.shape[0], device=device)
                zt = (1.0 - t[:, None]) * zA + t[:, None] * zB
                u  = (zB - zA)
                v  = vf(zt, t, zA, zB)
                loss_cfm = F.mse_loss(v, u, reduction="sum")
                loss_rec = F.l1_loss(dec(zA), A, reduction="sum") + F.l1_loss(dec(zB), B, reduction="sum")
                va_cfm += loss_cfm.item()
                va_rec += loss_rec.item()
                va_n   += A.shape[0]
        va_cfm /= max(1, va_n)
        va_rec /= max(1, va_n)
        print(f"[ep {ep:02d}]   val: Lcfm={va_cfm:.4f}  Lrec={va_rec:.4f}")

        # Save checkpoint
        ckpt = {
            "epoch": ep,
            "enc": enc.state_dict(),
            "dec": dec.state_dict(),
            "vf":  vf.state_dict(),
            "val_cfm": va_cfm,
            "val_rec": va_rec,
            "cfg": vars(cfg),
        }
        torch.save(ckpt, cfg.out_dir / "last.ckpt")
        if cfg.save_every_ep or (va_cfm < best_val):
            if va_cfm < best_val:
                best_val = va_cfm
                torch.save(ckpt, cfg.out_dir / "best.ckpt")
                print(f"[ep {ep:02d}] saved best (val Lcfm={best_val:.4f})")
            torch.save(ckpt, cfg.out_dir / "checkpoints" / f"ep{ep:02d}.ckpt")

    print("[done]")

# --------------------------------------------------------------------------------------
# Simple “distance” export (optional): encode pairs and write ||zB - zA|| to CSV
# --------------------------------------------------------------------------------------
@torch.no_grad()
def export_latent_distances(ckpt_path: Path, data_root: Path, n_pairs: int, out_csv: Path, device_str: Optional[str] = None):
    device = pick_device(device_str)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    z_dim = ckpt.get("cfg", {}).get("z_dim", 128)

    enc = Encoder(z_dim).to(device)
    enc.load_state_dict(ckpt["enc"])
    enc.eval()

    ds = PairFromTrajDataset(data_root, "val", 0.95, 0, 20, 0.1)
    rng = random.Random(0)

    rows = [("path_a","path_b","latent_l2")]
    for _ in range(n_pairs):
        a, b, pa, pb = ds[rng.randrange(0, len(ds))]
        A = a.unsqueeze(0).to(device)
        B = b.unsqueeze(0).to(device)
        zA = enc(A); zB = enc(B)
        d = torch.linalg.norm(zB - zA, dim=1).item()
        rows.append((pa, pb, f"{d:.6f}"))

    import csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f); w.writerows(rows)
    print(f"[export] wrote {out_csv}")

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("out.action_distance_cfm"))
    ap.add_argument("--z_dim", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--max_step_gap", type=int, default=20)
    ap.add_argument("--p_cross_traj", type=float, default=0.10)

    ap.add_argument("--lambda_cfm", type=float, default=1.0)
    ap.add_argument("--lambda_rec", type=float, default=0.1)
    ap.add_argument("--lambda_latent_l2", type=float, default=1e-4)

    ap.add_argument("--viz_every", type=int, default=100)
    ap.add_argument("--viz_rows", type=int, default=6)   # not used directly, but kept for parity
    ap.add_argument("--viz_cols", type=int, default=9)

    # optional export
    ap.add_argument("--export_pairs", type=int, default=0, help="if >0, export N pairs’ ||zB-zA|| to CSV after training")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = TrainCfg(
        data_root=args.data_root, out_dir=args.out_dir, z_dim=args.z_dim,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, seed=args.seed,
        num_workers=args.num_workers, device=args.device, max_step_gap=args.max_step_gap,
        p_cross_traj=args.p_cross_traj, lambda_cfm=args.lambda_cfm, lambda_rec=args.lambda_rec,
        lambda_latent_l2=args.lambda_latent_l2, viz_every=args.viz_every,
        viz_rows=args.viz_rows, viz_cols=args.viz_cols
    )
    train(cfg)

    if args.export_pairs > 0:
        export_latent_distances(
            ckpt_path=cfg.out_dir / "best.ckpt",
            data_root=cfg.data_root,
            n_pairs=args.export_pairs,
            out_csv=cfg.out_dir / "latent_distances.csv",
            device_str=cfg.device,
        )

if __name__ == "__main__":
    main()
