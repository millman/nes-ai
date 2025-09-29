#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Action-distance (CFM) toy trainer with a tiny encoder/decoder for visualization.

Debug visualization: full latent interpolation from A→B decoded as images:
  [A] [dec(zA)] [t-samples ...] [dec(zB)] [B]

CLI:
  python action_distance_cfm.py --data_root traj_dumps --out_dir out.action_distance_cfm
"""

from __future__ import annotations
import argparse, math, random, contextlib
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

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
        return z  # keep scale (no normalization)

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
# Utilities & Debug helpers
# --------------------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().clamp(0,1)
    arr = (t.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)

def _psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # assuming x,y in [0,1]
    mse = F.mse_loss(x, y, reduction="mean").clamp_min(eps)
    return 10.0 * torch.log10(1.0 / mse)

def _grad_norm(module: nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().data.pow(2).sum().cpu())
    return math.sqrt(total) if total > 0 else 0.0

# --------------------------------------------------------------------------------------
# Visualization: FULL interpolation from A→B (decoded)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def save_full_interpolation_grid(
    enc: Encoder, dec: Decoder,
    A: torch.Tensor, B: torch.Tensor,
    out_path: Path, device: torch.device,
    interp_steps: int = 12
):
    """
    For the first up to 8 rows in batch:
      Row shows: [A] [dec(zA)] [dec(z_t1)] ... [dec(z_tK)] [dec(zB)] [B]
      where z_tk = (1 - t_k) * zA + t_k * zB,  t_k in (0,1), K = interp_steps
      Endpoints (t=0, t=1) are shown as dec(zA), dec(zB) next to raw A, B.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Bsz = min(A.shape[0], 8)
    K = max(1, int(interp_steps))  # number of *internal* samples (excluding endpoints)

    A = A[:Bsz].to(device); B = B[:Bsz].to(device)
    zA = enc(A); zB = enc(B)

    # t values in (0,1), excluding endpoints; shape (K,)
    t_vals = torch.linspace(0, 1, K+2, device=device)[1:-1]
    # Precompute decoded samples for efficiency: list of (B,3,H,W)
    interp_decoded = []
    for t in t_vals:
        zt = (1.0 - t) * zA + t * zB
        interp_decoded.append(dec(zt))  # (B,3,H,W)

    # Compose canvas
    tiles_per_row = 2 + K + 2  # A | dec(zA) | K interps | dec(zB) | B
    tile_w, tile_h = W, H
    total_w = tile_w * tiles_per_row
    total_h = tile_h * Bsz
    canvas = Image.new("RGB", (total_w, total_h), (0,0,0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Place rows
    for r in range(Bsz):
        x = 0; y = r * tile_h
        # Raw A
        canvas.paste(_to_pil(A[r]), (x, y)); x += tile_w
        # dec(zA)
        canvas.paste(_to_pil(dec(zA[r:r+1])[0]), (x, y)); x += tile_w
        # K internal samples
        for k in range(K):
            canvas.paste(_to_pil(interp_decoded[k][r]), (x, y))
            # annotate t on first row only to reduce clutter
            if r == 0:
                tt = float(t_vals[k].item())
                txt = f"t={tt:.2f}"
                if font:
                    bbox = draw.textbbox((0, 0), txt, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                else:
                    tw, th = len(txt)*6, 10
                draw.rectangle([x+2, y+2, x+2+tw+2, y+2+th+2], fill=(0,0,0))
                draw.text((x+4, y+4), txt, font=font, fill=(220,220,255))
            x += tile_w
        # dec(zB)
        canvas.paste(_to_pil(dec(zB[r:r+1])[0]), (x, y)); x += tile_w
        # Raw B
        canvas.paste(_to_pil(B[r]), (x, y)); x += tile_w

        # row annotation: ||zB-zA||
        txt = f"||zB-zA||={torch.linalg.norm(zB[r]-zA[r]).item():.2f}"
        if font:
            bbox = draw.textbbox((0, 0), txt, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            tw, th = len(txt)*6, 10
        draw.rectangle([2, y+2, 2+tw+2, y+2+th+2], fill=(0,0,0))
        draw.text((4, y+4), txt, font=font, fill=(255,255,255))

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
    interp_steps: int = 12   # <--- controls density of the full interpolation
    save_every_ep: bool = True
    log_every: int = 50

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

    use_pin = (device.type == "cuda")  # pinning not useful on CPU/MPS; avoids MPS warning
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers, pin_memory=use_pin, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=use_pin, drop_last=False)

    print(f"[Device] {device} | [Data] train pairs≈{len(ds_tr)}  val pairs≈{len(ds_va)}")

    enc = Encoder(cfg.z_dim).to(device)
    dec = Decoder(cfg.z_dim).to(device)
    vf  = VectorField(cfg.z_dim).to(device)

    params = list(enc.parameters()) + list(dec.parameters()) + list(vf.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr)

    # AMP setup: CUDA only
    if device.type == "cuda":
        amp_autocast = lambda: autocast(device_type="cuda")
        scaler = GradScaler(enabled=True)
    else:
        amp_autocast = contextlib.nullcontext
        scaler = GradScaler(enabled=False)

    global_step = 0
    best_val = float("inf")

    # running windows for pretty logs
    win = 50
    q_lcfm, q_lrec, q_llat, q_ltot = deque(maxlen=win), deque(maxlen=win), deque(maxlen=win), deque(maxlen=win)
    q_dnorm, q_vnorm, q_cos = deque(maxlen=win), deque(maxlen=win), deque(maxlen=win)
    q_psnrA, q_psnrB = deque(maxlen=win), deque(maxlen=win)
    q_genc, q_gdec, q_gvf = deque(maxlen=win), deque(maxlen=win), deque(maxlen=win)

    for ep in range(1, cfg.epochs+1):
        enc.train(); dec.train(); vf.train()
        run_loss_cfm = run_loss_rec = run_n = 0.0

        for A, B, _, _ in dl_tr:
            A = A.to(device, non_blocking=True)
            B = B.to(device, non_blocking=True)

            with amp_autocast():
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
            # gradient norms (after backward, before step)
            g_enc = _grad_norm(enc)
            g_dec = _grad_norm(dec)
            g_vf  = _grad_norm(vf)
            scaler.step(opt)
            scaler.update()

            run_loss_cfm += loss_cfm.item() * A.shape[0]
            run_loss_rec += loss_rec.item() * A.shape[0]
            run_n += A.shape[0]
            global_step += 1

            # --------- Light-weight debug stats ----------
            with torch.no_grad():
                d = (zB - zA)
                dnorm = torch.linalg.norm(d, dim=1).mean().item()
                vnorm = torch.linalg.norm(v, dim=1).mean().item()
                cosvu = F.cosine_similarity(v, u, dim=1).mean().item()
                psnrA = _psnr(xA_rec, A).item()
                psnrB = _psnr(xB_rec, B).item()

                q_lcfm.append(float(loss_cfm.item()))
                q_lrec.append(float(loss_rec.item()))
                q_llat.append(float(loss_lat.item()))
                q_ltot.append(float(loss.item()))
                q_dnorm.append(dnorm); q_vnorm.append(vnorm); q_cos.append(cosvu)
                q_psnrA.append(psnrA); q_psnrB.append(psnrB)
                q_genc.append(g_enc); q_gdec.append(g_dec); q_gvf.append(g_vf)

            if (cfg.log_every > 0) and (global_step % cfg.log_every == 0):
                def avg(q):
                    return (sum(q) / len(q)) if len(q) else 0.0
                print(
                    f"ep {ep:02d} step {global_step:06d} | "
                    f"loss {avg(q_ltot):.4f} | "
                    f"Lcfm {avg(q_lcfm):.4f} Lrec {avg(q_lrec):.4f} Llat {avg(q_llat):.5f} | "
                    f"‖Δ‖ {avg(q_dnorm):.3f} ‖v‖ {avg(q_vnorm):.3f} cos(v,u) {avg(q_cos):.3f} | "
                    f"PSNR A {avg(q_psnrA):.2f}dB B {avg(q_psnrB):.2f}dB | "
                    f"∥g_enc∥ {avg(q_genc):.3e} ∥g_dec∥ {avg(q_gdec):.3e} ∥g_vf∥ {avg(q_gvf):.3e}"
                )

            if (cfg.viz_every > 0) and (global_step % cfg.viz_every == 0):
                enc.eval(); dec.eval()
                save_full_interpolation_grid(
                    enc, dec,
                    A.cpu(), B.cpu(),
                    cfg.out_dir / "viz" / f"ep{ep:02d}_step{global_step:06d}.png",
                    device=device, interp_steps=cfg.interp_steps
                )
                enc.train(); dec.train()

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
    ap.add_argument("--epochs", type=int, default=1000)
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
    ap.add_argument("--interp_steps", type=int, default=12, help="number of internal interpolation samples between dec(zA) and dec(zB)")
    ap.add_argument("--log_every", type=int, default=50, help="print running debug stats every N steps")

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
        interp_steps=args.interp_steps, log_every=args.log_every
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
