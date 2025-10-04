#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Action-distance (CFM) trainer with a lightweight CNN autoencoder, AE warmup, and
full A→B latent interpolation visualization with trajectory/state labels.

Run:
  python action_distance_cfm.py --data_root traj_dumps --out_dir out.action_distance_cfm
"""

from __future__ import annotations

import argparse
import contextlib
import math
import random
import time
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from recon import (
    H,
    W,
    Decoder,
    DownBlock,
    PairFromTrajDataset,
    load_frame_as_tensor as base_load_frame_as_tensor,
    set_seed,
    short_traj_state_label,
    to_float01,
)
from recon.utils import grad_norm, psnr_01, tensor_to_pil

def _normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    # Deprecated placeholder (kept for compatibility with older checkpoints)
    return t


def _denormalize_tensor(t: torch.Tensor) -> torch.Tensor:
    # Deprecated placeholder (kept for compatibility with older checkpoints)
    return t


def load_frame_as_tensor(path: Path) -> torch.Tensor:
    return base_load_frame_as_tensor(path, normalize=_normalize_tensor)



class Encoder(nn.Module):
    def __init__(self, z_dim: int = 128, pretrained: bool = False, freeze_backbone: bool = False):
        super().__init__()
        if pretrained or freeze_backbone:
            warnings.warn(
                "Lightweight encoder does not support pretrained/freeze flags; ignoring.",
                RuntimeWarning,
            )

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # 120x112
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.down1 = DownBlock(32, 64)   # 60x56
        self.down2 = DownBlock(64, 128)  # 30x28
        self.down3 = DownBlock(128, 256) # 15x14
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, z_dim)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.pool(h).flatten(1)
        return self.fc(h)

# --------------------------------------------------------------------------------------
# Conditional vector field for CFM
# --------------------------------------------------------------------------------------
class VectorField(nn.Module):
    def __init__(self, z_dim: int = 128):
        super().__init__()
        in_dim = z_dim * 3 + 1  # [z_t, (zB-zA), (zA+zB)/2, t]
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
# Utilities & debug helpers
# --------------------------------------------------------------------------------------


def _to_pil(t: torch.Tensor) -> Image.Image:
    return tensor_to_pil(t, denormalize=_denormalize_tensor)


def _psnr_01(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return psnr_01(_denormalize_tensor(x), _denormalize_tensor(y), eps)


def _grad_norm(module: nn.Module) -> float:
    return grad_norm(module)

# --------------------------------------------------------------------------------------
# Debug visualization: FULL latent interpolation from A→B with labels
# --------------------------------------------------------------------------------------
@torch.no_grad()
def save_full_interpolation_grid(
    enc: Encoder, dec: Decoder,
    A: torch.Tensor, B: torch.Tensor,
    pathsA: List[str], pathsB: List[str],
    out_path: Path, device: torch.device,
    interp_steps: int = 12
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Bsz = min(A.shape[0], 8)
    K = max(1, int(interp_steps))

    A = A[:Bsz].contiguous()
    B = B[:Bsz].contiguous()
    pA = pathsA[:Bsz]
    pB = pathsB[:Bsz]

    A_dev = to_float01(A, device, non_blocking=False)
    B_dev = to_float01(B, device, non_blocking=False)
    zA = enc(A_dev)
    zB = enc(B_dev)
    pair_recon = dec(torch.cat([zA, zB], dim=0))
    dec_zA, dec_zB = pair_recon.chunk(2, dim=0)

    t_vals = torch.linspace(0, 1, K + 2, device=device)[1:-1]
    if K > 0:
        z_interp = torch.stack([(1.0 - t) * zA + t * zB for t in t_vals], dim=0)
        interp_decoded = dec(z_interp.view(-1, zA.shape[1])).view(K, Bsz, 3, H, W)
    else:
        interp_decoded = torch.empty(0, Bsz, 3, H, W, device=device)

    tiles_per_row = 2 + K + 2  # A | dec(zA) | K interps | dec(zB) | B
    tile_w, tile_h = W, H
    total_w = tile_w * tiles_per_row
    total_h = tile_h * Bsz
    canvas = Image.new("RGB", (total_w, total_h), (0,0,0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=18)
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    def annotate(draw, x, y, txt, fill=(255,255,255)):
        if font:
            bbox = draw.textbbox((0, 0), txt, font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        else:
            tw, th = len(txt)*6, 10
        pad = 4
        draw.rectangle([x+2, y+2, x+2+tw+pad*2, y+2+th+pad*2], fill=(0,0,0))
        draw.text((x+2+pad, y+2+pad), txt, font=font, fill=fill)

    for r in range(Bsz):
        x = 0; y = r * tile_h
        labelA = short_traj_state_label(pA[r])
        labelB = short_traj_state_label(pB[r])

        canvas.paste(_to_pil(A[r]), (x, y)); annotate(draw, x, y, labelA); x += tile_w
        canvas.paste(_to_pil(dec_zA[r]), (x, y)); annotate(draw, x, y, labelA + " (dec)", (220,220,255)); x += tile_w

        for k in range(K):
            canvas.paste(_to_pil(interp_decoded[k, r]), (x, y))
            annotate(draw, x, y, f"t={float(t_vals[k]):.2f}", (200,255,200))
            x += tile_w

        canvas.paste(_to_pil(dec_zB[r]), (x, y)); annotate(draw, x, y, labelB + " (dec)", (220,220,255)); x += tile_w
        canvas.paste(_to_pil(B[r]), (x, y)); annotate(draw, x, y, labelB)

        txt = f"‖zB−zA‖={torch.linalg.norm(zB[r]-zA[r]).item():.2f}"
        annotate(draw, 2, y + tile_h - 22, txt, (255,255,0))

    canvas.save(out_path)


@torch.no_grad()
def save_simple_debug_grid(
    enc: Encoder, dec: Decoder,
    A: torch.Tensor, B: torch.Tensor,
    pathsA: List[str], pathsB: List[str],
    out_path: Path, device: torch.device
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Bsz = min(A.shape[0], 4)
    A = A[:Bsz].contiguous()
    B = B[:Bsz].contiguous()

    # Stage samples through device and back to isolate where artefacts appear
    A_dev = to_float01(A, device, non_blocking=False)
    B_dev = to_float01(B, device, non_blocking=False)
    A_dev_back = A_dev.to("cpu")
    B_dev_back = B_dev.to("cpu")

    with torch.no_grad():
        zA = enc(A_dev)
        zB = enc(B_dev)
        rec_pair = dec(torch.cat([zA, zB], dim=0)).to("cpu")
        A_rec, B_rec = rec_pair.chunk(2, dim=0)

    cols = [
        ("A raw", A),
        ("A recon", A_rec),
        ("A gpu→cpu", A_dev_back),
        ("B raw", B),
        ("B recon", B_rec),
        ("B gpu→cpu", B_dev_back),
    ]

    tile_w, tile_h = W, H
    canvas = Image.new("RGB", (tile_w * len(cols), tile_h * Bsz), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    for r in range(Bsz):
        y = r * tile_h
        for c, (label, stack) in enumerate(cols):
            x = c * tile_w
            canvas.paste(_to_pil(stack[r]), (x, y))
            draw.rectangle([x + 2, y + 2, x + tile_w - 2, y + 26], outline=(255,255,0))
            draw.text((x + 6, y + 6), label, fill=(255, 255, 0))

    canvas.save(out_path)

# --------------------------------------------------------------------------------------
# Training configuration
# --------------------------------------------------------------------------------------
@dataclass
class TrainCfg:
    data_root: Path
    out_dir: Path = Path("out.action_distance_cfm")
    z_dim: int = 128
    batch_size: int = 32
    epochs: int = 5
    lr: float = 3e-4
    vf_lr_mult: float = 0.5
    seed: int = 0
    num_workers: int = 4
    device: Optional[str] = None
    max_step_gap: int = 20
    allow_cross_traj: bool = False
    p_cross_traj: float = 0.0
    encoder_pretrained: bool = False  # no-op with lightweight encoder (kept for CLI compatibility)
    freeze_backbone: bool = False     # no-op with lightweight encoder
    use_foreach_optim: bool = True
    detach_decoder_from_encoder: bool = False  # prevents recon gradients from updating encoder/CFM

    # Loss weights
    lambda_cfm: float = 0.3
    lambda_rec: float = 2.0
    lambda_latent_l2: float = 0.0
    lambda_zcal: float = 0.05

    # Schedules / stability
    warmup_steps: int = 2000         # AE-only steps (CFM/lat/zcal off)
    vf_grad_clip: float = 1.0        # clip VF grads

    # Viz/debug
    viz_every: int = 500
    interp_steps: int = 12
    log_every: int = 50
    simple_viz: bool = False

# --------------------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------------------
def pick_device(pref: Optional[str]) -> torch.device:
    if pref:
        dev = torch.device(pref)
        if dev.type == "cuda":
            raise ValueError("CUDA is not supported by this trainer; use 'cpu' or 'mps'.")
        return dev
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def kaiming_init(m):
    # Use ReLU gain for SiLU blocks (safe & standard).
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu", mode="fan_in")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def train(cfg: TrainCfg):
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "checkpoints").mkdir(exist_ok=True, parents=True)
    (cfg.out_dir / "viz").mkdir(exist_ok=True, parents=True)

    ds_tr = PairFromTrajDataset(
        cfg.data_root,
        "train",
        0.95,
        cfg.seed,
        cfg.max_step_gap,
        cfg.allow_cross_traj,
        cfg.p_cross_traj,
        load_frame=load_frame_as_tensor,
    )
    ds_va = PairFromTrajDataset(
        cfg.data_root,
        "val",
        0.95,
        cfg.seed,
        cfg.max_step_gap,
        cfg.allow_cross_traj,
        cfg.p_cross_traj,
        load_frame=load_frame_as_tensor,
    )

    use_pin = False
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers, pin_memory=use_pin, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=use_pin, drop_last=False)

    print(f"[Device] {device} | [Data] train pairs≈{len(ds_tr)}  val pairs≈{len(ds_va)}")

    enc = Encoder(cfg.z_dim, pretrained=cfg.encoder_pretrained, freeze_backbone=cfg.freeze_backbone).to(device)
    dec = Decoder(cfg.z_dim).to(device)
    vf  = VectorField(cfg.z_dim).to(device)
    dec.apply(kaiming_init); vf.apply(kaiming_init)

    # Per-module LRs (VF slower)
    optim_kwargs = {"lr": cfg.lr, "weight_decay": 1e-4}
    if cfg.use_foreach_optim:
        optim_kwargs["foreach"] = True
    try:
        opt = torch.optim.AdamW([
            {"params": enc.parameters(), "lr": cfg.lr},
            {"params": dec.parameters(), "lr": cfg.lr},
            {"params": vf.parameters(),  "lr": cfg.lr * cfg.vf_lr_mult},
        ], **optim_kwargs)
    except TypeError:
        if optim_kwargs.pop("foreach", None) is not None:
            opt = torch.optim.AdamW([
                {"params": enc.parameters(), "lr": cfg.lr},
                {"params": dec.parameters(), "lr": cfg.lr},
                {"params": vf.parameters(),  "lr": cfg.lr * cfg.vf_lr_mult},
            ], **optim_kwargs)
        else:
            raise

    # AMP only on CUDA
    amp_autocast = contextlib.nullcontext
    scaler = GradScaler(enabled=False)

    global_step = 0
    best_val = float("inf")

    # Running windows
    win = 50
    q_lcfm, q_lrec, q_llat, q_ltot = deque(maxlen=win), deque(maxlen=win), deque(maxlen=win), deque(maxlen=win)
    q_dnorm, q_vnorm, q_cos = deque(maxlen=win), deque(maxlen=win), deque(maxlen=win)
    q_psnrA, q_psnrB = deque(maxlen=win), deque(maxlen=win)
    q_genc, q_gdec, q_gvf = deque(maxlen=win), deque(maxlen=win), deque(maxlen=win)
    # Timing queues
    q_data_time, q_forward_time, q_backward_time = deque(maxlen=win), deque(maxlen=win), deque(maxlen=win)
    step_time_accum = 0.0
    step_time_count = 0

    for ep in range(1, cfg.epochs+1):
        enc.train(); dec.train(); vf.train()
        run_loss_cfm = run_loss_rec = run_n = 0.0

        for A, B, pathsA, pathsB in dl_tr:
            step_start = time.perf_counter()
            need_viz = (cfg.viz_every > 0) and (global_step % cfg.viz_every == 0)
            A_cpu = A.detach().clone() if need_viz else None
            B_cpu = B.detach().clone() if need_viz else None

            # TIMING: Data transfer
            data_start = time.perf_counter()
            A = to_float01(A, device)
            B = to_float01(B, device)
            data_time = time.perf_counter() - data_start

            # TIMING: Forward pass
            forward_start = time.perf_counter()
            with amp_autocast():
                zA = enc(A)
                zB = enc(B)

                # --- CFM core: detached + scale-invariant (cos + normalized magnitude) ---
                t = torch.rand(A.shape[0], device=device)
                zA_d, zB_d = zA.detach(), zB.detach()
                zt_d = (1.0 - t[:, None]) * zA_d + t[:, None] * zB_d
                u_tgt = (zB_d - zA_d)

                v = vf(zt_d, t, zA_d, zB_d)

                eps = 1e-6
                cos_term = 1.0 - F.cosine_similarity(v, u_tgt, dim=1).mean()
                mag_term = (torch.linalg.norm(v - u_tgt, dim=1) / (torch.linalg.norm(u_tgt, dim=1) + eps)).mean()
                loss_cfm = cos_term + mag_term

                # Reconstruction
                if cfg.detach_decoder_from_encoder:
                    z_pair = torch.cat([zA.detach(), zB.detach()], dim=0)
                else:
                    z_pair = torch.cat([zA, zB], dim=0)
                x_pair = dec(z_pair)
                xA_rec, xB_rec = x_pair.chunk(2, dim=0)
                loss_rec = F.l1_loss(xA_rec, A) + F.l1_loss(xB_rec, B)

                # small latent penalties
                loss_lat = 0.5 * (zA.pow(2).mean() + zB.pow(2).mean())  # can be 0 during warmup

                # latent scale calibration (keep per-dim std near 1)
                z_all = torch.cat([zA, zB], dim=0)
                z_std = z_all.std(dim=0)
                loss_zcal = (z_std - 1.0).abs().mean()

                # Warmup gating
                cfm_w = 0.0 if global_step < cfg.warmup_steps else cfg.lambda_cfm
                lat_w = 0.0 if global_step < cfg.warmup_steps else cfg.lambda_latent_l2
                zcl_w = 0.0 if global_step < cfg.warmup_steps else cfg.lambda_zcal

                loss = cfm_w*loss_cfm + cfg.lambda_rec*loss_rec + lat_w*loss_lat + zcl_w*loss_zcal
            forward_time = time.perf_counter() - forward_start

            # TIMING: Backward pass
            backward_start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # Clip VF grads (after unscale)
            if scaler.is_enabled():
                scaler.unscale_(opt)
            if cfg.vf_grad_clip and cfg.vf_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(vf.parameters(), max_norm=cfg.vf_grad_clip)

            # grad norms for logs
            g_enc = _grad_norm(enc)
            g_dec = _grad_norm(dec)
            g_vff = _grad_norm(vf)

            scaler.step(opt)
            scaler.update()
            backward_time = time.perf_counter() - backward_start

            step_time = time.perf_counter() - step_start
            step_time_accum += step_time
            step_time_count += 1

            run_loss_cfm += loss_cfm.item() * A.shape[0]
            run_loss_rec += loss_rec.item() * A.shape[0]
            run_n += A.shape[0]

            with torch.no_grad():
                d = (zB - zA)
                dnorm = torch.linalg.norm(d, dim=1).mean().item()
                vnorm = torch.linalg.norm(v, dim=1).mean().item()
                cosvu = F.cosine_similarity(v, u_tgt, dim=1).mean().item()
                psnrA = _psnr_01(xA_rec, A).item()
                psnrB = _psnr_01(xB_rec, B).item()

                q_lcfm.append(float(loss_cfm.item()))
                q_lrec.append(float(loss_rec.item()))
                q_llat.append(float(loss_lat.item()))
                q_ltot.append(float(loss.item()))
                q_dnorm.append(dnorm); q_vnorm.append(vnorm); q_cos.append(cosvu)
                q_psnrA.append(psnrA); q_psnrB.append(psnrB)
                q_genc.append(g_enc); q_gdec.append(g_dec); q_gvf.append(g_vff)
                # Add timing data
                q_data_time.append(data_time * 1000)  # Convert to ms
                q_forward_time.append(forward_time * 1000)  # Convert to ms
                q_backward_time.append(backward_time * 1000)  # Convert to ms

            if (cfg.log_every > 0) and (global_step % cfg.log_every == 0):
                avg = lambda q: (sum(q)/len(q)) if len(q) else 0.0
                avg_step_time = (step_time_accum / step_time_count) if step_time_count else 0.0
                step_time_accum = 0.0
                step_time_count = 0
                if avg_step_time > 0:
                    throughput = (cfg.batch_size / avg_step_time)
                else:
                    throughput = 0.0
                print(
                    f"ep {ep:02d} step {global_step:06d} | "
                    f"loss {avg(q_ltot):.4f} | "
                    f"Lcfm {avg(q_lcfm):.4f} Lrec {avg(q_lrec):.4f} Llat {avg(q_llat):.5f} | "
                    f"‖Δ‖ {avg(q_dnorm):.3f} ‖v‖ {avg(q_vnorm):.3f} cos(v,u) {avg(q_cos):.3f} | "
                    f"PSNR A {avg(q_psnrA):.2f}dB B {avg(q_psnrB):.2f}dB | "
                    f"∥g_enc∥ {avg(q_genc):.3e} ∥g_dec∥ {avg(q_gdec):.3e} ∥g_vf∥ {avg(q_gvf):.3e} | "
                    f"timing: data {avg(q_data_time):.1f}ms fwd {avg(q_forward_time):.1f}ms bwd {avg(q_backward_time):.1f}ms | "
                    f"step_time {avg_step_time*1000:.1f} ms  ({throughput:.1f} samples/s)"
                )

            if global_step % cfg.viz_every == 0:
                enc.eval(); dec.eval()

                if cfg.simple_viz:
                    out_path = cfg.out_dir / "viz" / f"ep{ep:02d}_step{global_step:06d}_simple.png"
                    save_simple_debug_grid(
                        enc, dec,
                        A_cpu if A_cpu is not None else A.detach().cpu(),
                        B_cpu if B_cpu is not None else B.detach().cpu(),
                        list(pathsA), list(pathsB),
                        out_path,
                        device=device,
                    )

                out_path = cfg.out_dir / "viz" / f"ep{ep:02d}_step{global_step:06d}.png"
                save_full_interpolation_grid(
                    enc, dec,
                    A_cpu if A_cpu is not None else A.detach().cpu(),
                    B_cpu if B_cpu is not None else B.detach().cpu(),
                    list(pathsA), list(pathsB),
                    out_path,
                    device=device, interp_steps=cfg.interp_steps
                )
                enc.train(); dec.train()

            global_step += 1

        tr_cfm = run_loss_cfm / max(1, run_n)
        tr_rec = run_loss_rec / max(1, run_n)
        print(f"[ep {ep:02d}] train: Lcfm={tr_cfm:.4f}  Lrec={tr_rec:.4f}")

        # ---- Validation ----
        enc.eval(); dec.eval(); vf.eval()
        va_cfm = va_rec = va_n = 0.0
        with torch.no_grad():
            for A, B, _, _ in dl_va:
                A = to_float01(A, device, non_blocking=False)
                B = to_float01(B, device, non_blocking=False)
                zA = enc(A); zB = enc(B)
                t = torch.rand(A.shape[0], device=device)

                zA_d, zB_d = zA.detach(), zB.detach()
                zt_d = (1.0 - t[:, None]) * zA_d + t[:, None] * zB_d
                u_tgt = (zB_d - zA_d)
                v  = vf(zt_d, t, zA_d, zB_d)
                eps = 1e-6
                cos_term = 1.0 - F.cosine_similarity(v, u_tgt, dim=1).mean()
                mag_term = (torch.linalg.norm(v - u_tgt, dim=1) / (torch.linalg.norm(u_tgt, dim=1) + eps)).mean()
                loss_cfm = (cos_term + mag_term) * A.shape[0]

                x_pair = dec(torch.cat([zA, zB], dim=0))
                xA_rec, xB_rec = x_pair.chunk(2, dim=0)
                loss_rec = F.l1_loss(xA_rec, A, reduction="sum") + F.l1_loss(xB_rec, B, reduction="sum")
                va_cfm += loss_cfm.item()
                va_rec += loss_rec.item()
                va_n   += A.shape[0]
        va_cfm /= max(1, va_n)
        va_rec /= max(1, va_n)
        print(f"[ep {ep:02d}]   val: Lcfm={va_cfm:.4f}  Lrec={va_rec:.4f}")

        # Save checkpoint
        ckpt = {"epoch": ep, "enc": enc.state_dict(), "dec": dec.state_dict(), "vf": vf.state_dict(),
                "val_cfm": va_cfm, "val_rec": va_rec, "cfg": vars(cfg)}
        torch.save(ckpt, cfg.out_dir / "last.ckpt")
        if va_cfm < best_val:
            best_val = va_cfm
            torch.save(ckpt, cfg.out_dir / "best.ckpt")
            print(f"[ep {ep:02d}] saved best (val Lcfm={best_val:.4f})")
        torch.save(ckpt, cfg.out_dir / "checkpoints" / f"ep{ep:02d}.ckpt")

    print("[done]")

# --------------------------------------------------------------------------------------
# Export (optional): encode pairs and write ||zB - zA|| to CSV
# --------------------------------------------------------------------------------------
@torch.no_grad()
def export_latent_distances(ckpt_path: Path, data_root: Path, n_pairs: int, out_csv: Path, device_str: Optional[str] = None):
    device = pick_device(device_str)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    z_dim = ckpt.get("cfg", {}).get("z_dim", 128)
    enc = Encoder(z_dim, pretrained=False).to(device)
    enc.load_state_dict(ckpt["enc"]); enc.eval()
    ds = PairFromTrajDataset(
        data_root,
        "val",
        0.95,
        0,
        20,
        False,
        0.0,
        load_frame=load_frame_as_tensor,
    )
    rng = random.Random(0)

    rows = [("path_a","path_b","latent_l2")]
    for _ in range(n_pairs):
        a, b, pa, pb = ds[rng.randrange(0, len(ds))]
        A = to_float01(a.unsqueeze(0), device, non_blocking=False)
        B = to_float01(b.unsqueeze(0), device, non_blocking=False)
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
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--vf_lr_mult", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--max_step_gap", type=int, default=20)
    ap.add_argument("--allow_cross_traj", action="store_true")
    ap.add_argument("--p_cross_traj", type=float, default=0.0)

    ap.add_argument("--lambda_cfm", type=float, default=0.3)
    ap.add_argument("--lambda_rec", type=float, default=2.0)
    ap.add_argument("--lambda_latent_l2", type=float, default=0.0)
    ap.add_argument("--lambda_zcal", type=float, default=0.05)
    ap.add_argument("--warmup_steps", type=int, default=2000)
    ap.add_argument("--vf_grad_clip", type=float, default=1.0)

    ap.add_argument("--viz_every", type=int, default=10)
    ap.add_argument("--interp_steps", type=int, default=12)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--simple_viz", action="store_true", help="emit minimal GPU debug grids instead of full interpolation viz")
    ap.add_argument("--encoder_pretrained", dest="encoder_pretrained", action="store_true",
                    help="(deprecated) no-op with lightweight encoder; kept for CLI compatibility")
    ap.add_argument("--no_encoder_pretrained", dest="encoder_pretrained", action="store_false")
    ap.set_defaults(encoder_pretrained=False)
    ap.add_argument("--freeze_backbone", action="store_true",
                    help="(deprecated) no-op with lightweight encoder; kept for CLI compatibility")
    ap.add_argument("--no_foreach_optim", dest="use_foreach_optim", action="store_false",
                    help="disable foreach AdamW updates (use if PyTorch build lacks foreach support)")
    ap.add_argument("--foreach_optim", dest="use_foreach_optim", action="store_true",
                    help="force-enable foreach AdamW updates")
    ap.set_defaults(use_foreach_optim=True)
    ap.add_argument("--detach_decoder_from_encoder", action="store_true",
                    help="stop reconstruction gradients from updating encoder/CFM; decoder trains for viz only")

    ap.add_argument("--export_pairs", type=int, default=0)
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = TrainCfg(
        data_root=args.data_root, out_dir=args.out_dir, z_dim=args.z_dim,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, vf_lr_mult=args.vf_lr_mult,
        seed=args.seed, num_workers=args.num_workers, device=args.device, max_step_gap=args.max_step_gap,
        allow_cross_traj=args.allow_cross_traj, p_cross_traj=args.p_cross_traj,
        lambda_cfm=args.lambda_cfm, lambda_rec=args.lambda_rec, lambda_latent_l2=args.lambda_latent_l2,
        lambda_zcal=args.lambda_zcal, warmup_steps=args.warmup_steps, vf_grad_clip=args.vf_grad_clip,
        viz_every=args.viz_every, interp_steps=args.interp_steps, log_every=args.log_every,
        simple_viz=args.simple_viz, encoder_pretrained=args.encoder_pretrained, freeze_backbone=args.freeze_backbone,
        use_foreach_optim=args.use_foreach_optim,
        detach_decoder_from_encoder=args.detach_decoder_from_encoder,
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
