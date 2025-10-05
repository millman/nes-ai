#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Action-distance reconstruction trainer with a lightweight CNN autoencoder and
full A→B latent interpolation visualization with trajectory/state labels.

Run:
  python action_distance_recon.py --data_root traj_dumps --out_dir out.action_distance_recon
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from recon import (
    H,
    W,
    PairFromTrajDataset,
    load_frame_as_tensor as base_load_frame_as_tensor,
    set_seed,
    short_traj_state_label,
    to_float01,
)
from recon.utils import psnr_01, tensor_to_pil


def _normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    return t  # kept for compatibility with legacy checkpoints


def _denormalize_tensor(t: torch.Tensor) -> torch.Tensor:
    return t  # kept for compatibility with legacy checkpoints


def load_frame_as_tensor(path: Path) -> torch.Tensor:
    return base_load_frame_as_tensor(path, normalize=_normalize_tensor)


class Encoder(nn.Module):
    """Compact convolutional encoder for NES frames."""

    def __init__(self, z_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, z_dim)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.fc(h)


class Decoder(nn.Module):
    """Symmetric transpose-conv decoder with sigmoid output."""

    def __init__(self, z_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256 * 15 * 14)
        self.pre = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
        )
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 256, 15, 14)
        h = self.pre(h)
        x = self.net(h)
        return torch.sigmoid(x)


def _to_pil(t: torch.Tensor) -> Image.Image:
    return tensor_to_pil(t, denormalize=_denormalize_tensor)


def _psnr_01(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return psnr_01(_denormalize_tensor(x), _denormalize_tensor(y), eps)


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = (gauss / gauss.sum()).unsqueeze(0)
    window_2d = (gauss.t() @ gauss).unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def _ssim_components(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    padding = window.shape[-1] // 2
    mu_x = F.conv2d(x, window, padding=padding, groups=x.shape[1])
    mu_y = F.conv2d(y, window, padding=padding, groups=y.shape[1])

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=padding, groups=x.shape[1]) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=padding, groups=y.shape[1]) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=x.shape[1]) - mu_xy

    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2

    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    ssim_map = numerator / denominator
    cs_map = (2 * sigma_xy + C2) / (sigma_x_sq + sigma_y_sq + C2)

    ssim_val = ssim_map.mean(dim=(1, 2, 3))
    cs_val = cs_map.mean(dim=(1, 2, 3))
    return ssim_val, cs_val


def ms_ssim(x: torch.Tensor, y: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weights is None:
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=x.device, dtype=x.dtype)
    window_size = 11
    sigma = 1.5
    channels = x.shape[1]
    window = _gaussian_window(window_size, sigma, channels, x.device)
    levels = weights.shape[0]
    mssim: List[torch.Tensor] = []
    mcs: List[torch.Tensor] = []

    x_scaled, y_scaled = x, y
    for _ in range(levels):
        ssim_val, cs_val = _ssim_components(x_scaled, y_scaled, window)
        mssim.append(ssim_val)
        mcs.append(cs_val)
        x_scaled = F.avg_pool2d(x_scaled, kernel_size=2, stride=2, padding=0, ceil_mode=False)
        y_scaled = F.avg_pool2d(y_scaled, kernel_size=2, stride=2, padding=0, ceil_mode=False)

    mssim_tensor = torch.stack(mssim, dim=0)
    mcs_tensor = torch.stack(mcs[:-1], dim=0)

    pow1 = weights[:-1].unsqueeze(1)
    pow2 = weights[-1]
    ms_prod = torch.prod(mcs_tensor ** pow1, dim=0) * (mssim_tensor[-1] ** pow2)
    return ms_prod.mean()


def ms_ssim_loss(x: torch.Tensor, y: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    return 1.0 - ms_ssim(x, y, weights=weights)


def focal_patch_l1_loss(
    x_hat: torch.Tensor,
    x_true: torch.Tensor,
    *,
    patch_size: int,
    stride: int,
    gamma: float,
    eps: float,
) -> torch.Tensor:
    """Compute focal-weighted mean L1 across spatial patches."""

    stride = max(1, stride)
    err = (x_hat - x_true).abs().mean(dim=1, keepdim=True)
    pooled = F.avg_pool2d(err, kernel_size=patch_size, stride=stride, padding=0)
    if gamma > 0:
        weights = (pooled + eps).pow(gamma)
        weighted = (weights * pooled).sum(dim=(1, 2, 3))
        norm = weights.sum(dim=(1, 2, 3)).clamp_min(eps)
        loss = weighted / norm
    else:
        loss = pooled.mean(dim=(1, 2, 3))
    return loss.mean()


@torch.no_grad()
def save_full_interpolation_grid(
    enc: Encoder,
    dec: Decoder,
    A: torch.Tensor,
    B: torch.Tensor,
    pathsA: List[str],
    pathsB: List[str],
    out_path: Path,
    device: torch.device,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Bsz = min(A.shape[0], 8)

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

    t_vals = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], device=device)
    mid_t_vals = t_vals[1:-1]
    if mid_t_vals.numel() > 0:
        z_interp = torch.stack([(1.0 - t) * zA + t * zB for t in mid_t_vals], dim=0)
        interp_decoded = dec(z_interp.view(-1, zA.shape[1])).view(mid_t_vals.numel(), Bsz, 3, H, W)
    else:
        interp_decoded = torch.empty(0, Bsz, 3, H, W, device=device)

    tiles_per_row = 2 + len(t_vals)  # A raw | t columns | B raw
    tile_w, tile_h = W, H
    total_w = tile_w * tiles_per_row
    total_h = tile_h * Bsz
    canvas = Image.new("RGB", (total_w, total_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=18)
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    def annotate(draw_ctx, x, y, txt, fill=(255, 255, 255)):
        if font:
            bbox = draw_ctx.textbbox((0, 0), txt, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            tw, th = len(txt) * 6, 10
        pad = 4
        draw_ctx.rectangle([x + 2, y + 2, x + 2 + tw + pad * 2, y + 2 + th + pad * 2], fill=(0, 0, 0))
        draw_ctx.text((x + 2 + pad, y + 2 + pad), txt, font=font, fill=fill)

    for r in range(Bsz):
        x = 0
        y = r * tile_h
        labelA = short_traj_state_label(pA[r])
        labelB = short_traj_state_label(pB[r])

        canvas.paste(_to_pil(A[r]), (x, y))
        draw.line([(x, y), (x, y + tile_h - 1)], fill=(255, 255, 255))
        draw.line([(x + tile_w - 1, y), (x + tile_w - 1, y + tile_h - 1)], fill=(0, 0, 0))
        annotate(draw, x, y, labelA)
        x += tile_w
        canvas.paste(_to_pil(dec_zA[r]), (x, y))
        draw.line([(x, y), (x, y + tile_h - 1)], fill=(255, 255, 255))
        draw.line([(x + tile_w - 1, y), (x + tile_w - 1, y + tile_h - 1)], fill=(0, 0, 0))
        annotate(draw, x, y, "t=0.0 (A)", (220, 220, 255))
        x += tile_w

        for k in range(mid_t_vals.numel()):
            canvas.paste(_to_pil(interp_decoded[k, r]), (x, y))
            draw.line([(x, y), (x, y + tile_h - 1)], fill=(255, 255, 255))
            draw.line([(x + tile_w - 1, y), (x + tile_w - 1, y + tile_h - 1)], fill=(0, 0, 0))
            annotate(draw, x, y, f"t={float(mid_t_vals[k]):.1f}", (200, 255, 200))
            x += tile_w

        canvas.paste(_to_pil(dec_zB[r]), (x, y))
        draw.line([(x, y), (x, y + tile_h - 1)], fill=(255, 255, 255))
        draw.line([(x + tile_w - 1, y), (x + tile_w - 1, y + tile_h - 1)], fill=(0, 0, 0))
        annotate(draw, x, y, "t=1.0 (B)", (220, 220, 255))
        x += tile_w
        canvas.paste(_to_pil(B[r]), (x, y))
        draw.line([(x, y), (x, y + tile_h - 1)], fill=(255, 255, 255))
        draw.line([(x + tile_w - 1, y), (x + tile_w - 1, y + tile_h - 1)], fill=(0, 0, 0))
        annotate(draw, x, y, labelB)

        txt = f"‖zB−zA‖={torch.linalg.norm(zB[r] - zA[r]).item():.2f}"
        annotate(draw, 2, y + tile_h - 22, txt, (255, 255, 0))

    canvas.save(out_path)


@torch.no_grad()
def save_simple_debug_grid(
    enc: Encoder,
    dec: Decoder,
    A: torch.Tensor,
    B: torch.Tensor,
    pathsA: List[str],
    pathsB: List[str],
    out_path: Path,
    device: torch.device,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Bsz = min(A.shape[0], 4)
    A = A[:Bsz].contiguous()
    B = B[:Bsz].contiguous()

    A_dev = to_float01(A, device, non_blocking=False)
    B_dev = to_float01(B, device, non_blocking=False)
    A_dev_back = A_dev.to("cpu")
    B_dev_back = B_dev.to("cpu")

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
            draw.line([(x, y), (x, y + tile_h - 1)], fill=(255, 255, 255))
            draw.line([(x + tile_w - 1, y), (x + tile_w - 1, y + tile_h - 1)], fill=(0, 0, 0))
            draw.rectangle([x + 2, y + 2, x + tile_w - 2, y + 26], outline=(255, 255, 0))
            draw.text((x + 6, y + 6), label, fill=(255, 255, 0))

    canvas.save(out_path)


# --------------------------------------------------------------------------------------
# Training configuration
# --------------------------------------------------------------------------------------
@dataclass
class TrainCfg:
    data_root: Path
    out_dir: Path = Path("out.action_distance_recon")
    z_dim: int = 64
    batch_size: int = 16
    epochs: int = 20
    lr: float = 1e-3
    seed: int = 0
    num_workers: int = 2
    device: Optional[str] = None
    max_step_gap: int = 10
    viz_every: int = 200
    log_every: int = 50
    simple_viz: bool = False
    lambda_l1: float = 0.5
    lambda_ms_ssim: float = 0.5
    lambda_patch: float = 0.0
    patch_size: int = 32
    patch_stride: int = 16
    patch_focal_gamma: float = 1.5
    patch_eps: float = 1e-6


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

    start_time = time.monotonic()

    ds_tr = PairFromTrajDataset(
        cfg.data_root,
        "train",
        0.95,
        cfg.seed,
        cfg.max_step_gap,
        False,
        0.0,
        load_frame=load_frame_as_tensor,
    )
    ds_va = PairFromTrajDataset(
        cfg.data_root,
        "val",
        0.95,
        cfg.seed,
        cfg.max_step_gap,
        False,
        0.0,
        load_frame=load_frame_as_tensor,
    )

    use_pin = False
    dl_tr = DataLoader(
        ds_tr,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=use_pin,
        drop_last=True,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=use_pin,
        drop_last=False,
    )

    print(f"[Device] {device} | [Data] train pairs≈{len(ds_tr)}  val pairs≈{len(ds_va)}")

    enc = Encoder(cfg.z_dim).to(device)
    dec = Decoder(cfg.z_dim).to(device)
    enc.apply(kaiming_init)
    dec.apply(kaiming_init)

    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=cfg.lr)

    global_step = 0
    best_val = float("inf")

    win = 50
    q_loss = deque(maxlen=win)
    q_l1 = deque(maxlen=win)
    q_ms = deque(maxlen=win)
    q_patch = deque(maxlen=win) if cfg.lambda_patch > 0 else None
    q_psnrA, q_psnrB = deque(maxlen=win), deque(maxlen=win)
    q_step_ms = deque(maxlen=win)

    for ep in range(1, cfg.epochs + 1):
        enc.train()
        dec.train()
        run_loss_rec = run_n = 0.0

        for A, B, pathsA, pathsB in dl_tr:
            step_start = time.perf_counter()
            need_viz = (cfg.viz_every > 0) and (global_step % cfg.viz_every == 0)
            A_cpu = A.detach().clone() if need_viz else None
            B_cpu = B.detach().clone() if need_viz else None

            A = to_float01(A, device)
            B = to_float01(B, device)
            zA = enc(A)
            zB = enc(B)

            xA_rec = dec(zA)
            xB_rec = dec(zB)
            loss_l1 = F.l1_loss(xA_rec, A) + F.l1_loss(xB_rec, B)

            if cfg.lambda_ms_ssim > 0:
                loss_ms = ms_ssim_loss(xA_rec, A) + ms_ssim_loss(xB_rec, B)
            else:
                loss_ms = torch.tensor([float('nan')], device=xA_rec.device)

            if cfg.lambda_patch > 0:
                loss_patch = focal_patch_l1_loss(
                    xA_rec,
                    A,
                    patch_size=cfg.patch_size,
                    stride=cfg.patch_stride,
                    gamma=cfg.patch_focal_gamma,
                    eps=cfg.patch_eps,
                )
                loss_patch = loss_patch + focal_patch_l1_loss(
                    xB_rec,
                    B,
                    patch_size=cfg.patch_size,
                    stride=cfg.patch_stride,
                    gamma=cfg.patch_focal_gamma,
                    eps=cfg.patch_eps,
                )
            else:
                loss_patch = torch.tensor([float('nan')], device=xA_rec.device)

            loss = cfg.lambda_l1 * loss_l1
            if cfg.lambda_ms_ssim > 0:
                loss = loss + cfg.lambda_ms_ssim * loss_ms
            if cfg.lambda_patch > 0:
                loss = loss + cfg.lambda_patch * loss_patch

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            step_time = time.perf_counter() - step_start

            with torch.no_grad():
                psnrA = _psnr_01(xA_rec, A).item()
                psnrB = _psnr_01(xB_rec, B).item()

            batch_mean = 0.5 * loss_l1.item()
            run_loss_rec += batch_mean * A.shape[0]
            run_n += A.shape[0]

            q_loss.append(float(loss.item()))
            q_l1.append(float(loss_l1.item()))
            if cfg.lambda_ms_ssim > 0:
                q_ms.append(float(loss_ms.item()))
            if cfg.lambda_patch > 0 and q_patch is not None:
                q_patch.append(float(loss_patch.item()))
            q_psnrA.append(psnrA)
            q_psnrB.append(psnrB)
            q_step_ms.append(step_time * 1000.0)

            if (cfg.log_every > 0) and (global_step % cfg.log_every == 0):
                avg_loss = (sum(q_loss) / len(q_loss)) if q_loss else 0.0
                avg_l1 = (sum(q_l1) / len(q_l1)) if q_l1 else 0.0
                avg_ms = (sum(q_ms) / len(q_ms)) if q_ms else 0.0
                avg_patch = (sum(q_patch) / len(q_patch)) if (q_patch and len(q_patch)) else 0.0
                avg_psnrA = (sum(q_psnrA) / len(q_psnrA)) if q_psnrA else 0.0
                avg_psnrB = (sum(q_psnrB) / len(q_psnrB)) if q_psnrB else 0.0
                avg_step_ms = (sum(q_step_ms) / len(q_step_ms)) if q_step_ms else 0.0
                throughput = (cfg.batch_size / (avg_step_ms / 1000.0)) if avg_step_ms > 0 else 0.0
                elapsed = int(time.monotonic() - start_time)
                h = elapsed // 3600
                m = (elapsed % 3600) // 60
                s = elapsed % 60
                loss_terms = [f"L1 {avg_l1:.4f}"]
                if cfg.lambda_ms_ssim > 0:
                    loss_terms.append(f"MS-SSIM {avg_ms:.4f}")
                if cfg.lambda_patch > 0:
                    loss_terms.append(f"Patch {avg_patch:.4f}")
                print(
                    f"[{h:02d}:{m:02d}:{s:02d}] "
                    f"ep {ep:02d} step {global_step:06d} | "
                    f"loss {avg_loss:.4f} | "
                    " ".join(loss_terms) + " | "
                    f"PSNR A {avg_psnrA:.2f}dB B {avg_psnrB:.2f}dB | "
                    f"step {avg_step_ms:.1f} ms ({throughput:.1f} samples/s)"
                )

            if global_step % cfg.viz_every == 0:
                enc.eval()
                dec.eval()

                if cfg.simple_viz:
                    out_path = cfg.out_dir / "viz" / f"ep{ep:02d}_step{global_step:06d}_simple.png"
                    save_simple_debug_grid(
                        enc,
                        dec,
                        A_cpu if A_cpu is not None else A.detach().cpu(),
                        B_cpu if B_cpu is not None else B.detach().cpu(),
                        list(pathsA),
                        list(pathsB),
                        out_path,
                        device=device,
                    )

                out_path = cfg.out_dir / "viz" / f"ep{ep:02d}_step{global_step:06d}.png"
                save_full_interpolation_grid(
                    enc,
                    dec,
                    A_cpu if A_cpu is not None else A.detach().cpu(),
                    B_cpu if B_cpu is not None else B.detach().cpu(),
                    list(pathsA),
                    list(pathsB),
                    out_path,
                    device=device,
                )
                enc.train()
                dec.train()

            global_step += 1

        tr_rec = run_loss_rec / max(1, run_n)
        print(f"[ep {ep:02d}] train: Lrec={tr_rec:.4f}")

        enc.eval()
        dec.eval()
        va_rec = va_n = 0.0
        va_patch = 0.0 if cfg.lambda_patch > 0 else None
        with torch.no_grad():
            for A, B, _, _ in dl_va:
                A = to_float01(A, device, non_blocking=False)
                B = to_float01(B, device, non_blocking=False)
                zA = enc(A)
                zB = enc(B)
                xA_rec = dec(zA)
                xB_rec = dec(zB)
                batch_mean = 0.5 * (F.l1_loss(xA_rec, A).item() + F.l1_loss(xB_rec, B).item())
                va_rec += batch_mean * A.shape[0]
                if cfg.lambda_patch > 0 and va_patch is not None:
                    batch_patch = focal_patch_l1_loss(
                        xA_rec,
                        A,
                        patch_size=cfg.patch_size,
                        stride=cfg.patch_stride,
                        gamma=cfg.patch_focal_gamma,
                        eps=cfg.patch_eps,
                    )
                    batch_patch = batch_patch + focal_patch_l1_loss(
                        xB_rec,
                        B,
                        patch_size=cfg.patch_size,
                        stride=cfg.patch_stride,
                        gamma=cfg.patch_focal_gamma,
                        eps=cfg.patch_eps,
                    )
                    va_patch += float(batch_patch.item()) * A.shape[0]
                va_n += A.shape[0]
        va_rec = va_rec / max(1, va_n)
        if cfg.lambda_patch > 0 and va_patch is not None:
            va_patch_mean = va_patch / max(1, va_n)
            print(f"[ep {ep:02d}]   val: Lrec={va_rec:.4f} Patch={va_patch_mean:.4f}")
        else:
            print(f"[ep {ep:02d}]   val: Lrec={va_rec:.4f}")

        ckpt = {
            "epoch": ep,
            "enc": enc.state_dict(),
            "dec": dec.state_dict(),
            "val_rec": va_rec,
            "val_patch": va_patch / max(1, va_n) if (cfg.lambda_patch > 0 and va_patch is not None) else None,
            "cfg": vars(cfg),
        }
        torch.save(ckpt, cfg.out_dir / "last.ckpt")
        if va_rec < best_val:
            best_val = va_rec
            torch.save(ckpt, cfg.out_dir / "best.ckpt")
            print(f"[ep {ep:02d}] saved best (val Lrec={best_val:.4f})")
        torch.save(ckpt, cfg.out_dir / "checkpoints" / f"ep{ep:02d}.ckpt")

    print("[done]")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("out.action_distance_recon"))
    ap.add_argument("--z_dim", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--max_step_gap", type=int, default=10)
    ap.add_argument("--viz_every", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument(
        "--simple_viz",
        action="store_true",
        help="emit a smaller debug grid instead of the full interpolation layout",
    )
    ap.add_argument("--lambda_l1", type=float, default=0.5)
    ap.add_argument("--lambda_ms_ssim", type=float, default=0.5)
    ap.add_argument("--lambda_patch", type=float, default=0.0)
    ap.add_argument("--patch_size", type=int, default=32)
    ap.add_argument("--patch_stride", type=int, default=16)
    ap.add_argument("--patch_focal_gamma", type=float, default=1.5)
    ap.add_argument("--patch_eps", type=float, default=1e-6)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = TrainCfg(
        data_root=args.data_root,
        out_dir=args.out_dir,
        z_dim=args.z_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        max_step_gap=args.max_step_gap,
        viz_every=args.viz_every,
        log_every=args.log_every,
        simple_viz=args.simple_viz,
        lambda_l1=args.lambda_l1,
        lambda_ms_ssim=args.lambda_ms_ssim,
        lambda_patch=args.lambda_patch,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        patch_focal_gamma=args.patch_focal_gamma,
        patch_eps=args.patch_eps,
    )
    train(cfg)


if __name__ == "__main__":
    main()
