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
    TileSpec,
    load_frame_as_tensor as base_load_frame_as_tensor,
    render_image_grid,
    set_seed,
    short_traj_state_label,
    to_float01,
)
from recon.utils import psnr_01, tensor_to_pil

torch.multiprocessing.set_sharing_strategy('file_system')


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

    def __init__(self, z_dim: int = 64, *, use_refine: bool, self_refine_mids: bool, refine_ch: int = 64):

        super().__init__()
        self.use_refine = use_refine
        self.refine_ch = refine_ch
        self.self_refine_mids = self_refine_mids
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
        # --- Residual/refinement head: predicts r from concat([x, x_hat]) ---
        # Always build it so we can self-refine mids even if --use_refine wasn't passed.
        gn_groups = 8 if (refine_ch % 8 == 0 and refine_ch >= 8) else 1
        self.refine = nn.Sequential(
            nn.Conv2d(3 * 2, refine_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=gn_groups, num_channels=refine_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(refine_ch, 3, kernel_size=3, padding=1),
        )

        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, z: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        h = self.fc(z).view(-1, 256, 15, 14)
        h = self.pre(h)
        x_hat0 = torch.sigmoid(self.net(h))

        # Optional refinement: prefer real image x; else (if enabled) self-refine with x_hat as context
        if self.use_refine and (x is not None):
            r = self.refine(torch.cat([x, x_hat0], dim=1))
            x_hat = (x_hat0 + r).clamp(0.0, 1.0)
        elif self.self_refine_mids:
            x_ctx = x_hat0.detach()
            r = self.refine(torch.cat([x_ctx, x_hat0], dim=1))
            x_hat = (x_hat0 + r).clamp(0.0, 1.0)
        else:
            x_hat = x_hat0

        return x_hat

    @torch.no_grad()
    def decode_no_refine(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latents WITHOUT applying refinement, even if use_refine=True.
        Useful for latent interpolations where we don't have a ground-truth x.
        """
        h = self.fc(z).view(-1, 256, 15, 14)
        h = self.pre(h)
        return torch.sigmoid(self.net(h))

    @torch.no_grad()
    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        """
        Return pre-sigmoid logits of the base decoder for diagnostics.
        """
        h = self.fc(z).view(-1, 256, 15, 14)
        h = self.pre(h)
        return self.net(h)  # (N,3,H,W) pre-sigmoid

    @torch.no_grad()
    def decode_self_refine(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latents and apply a single self-refine pass: uses x_hat as its own context.
        This is purely for visualization of mids (no ground-truth context available).
        """
        h = self.fc(z).view(-1, 256, 15, 14)
        h = self.pre(h)
        x_hat0 = torch.sigmoid(self.net(h))
        # self-refine even if use_refine=False (refine head is always built now)
        r = self.refine(torch.cat([x_hat0.detach(), x_hat0], dim=1))
        return (x_hat0 + r).clamp(0.0, 1.0)

    def self_refine_train(self, z: torch.Tensor) -> torch.Tensor:
        """
        Trainable self-refine: decode base x_hat0, then refine using x_hat0 as context
        (no torch.no_grad), so gradients flow through refine and base paths.
        """
        h = self.fc(z).view(-1, 256, 15, 14)
        h = self.pre(h)
        x_hat0 = torch.sigmoid(self.net(h))
        r = self.refine(torch.cat([x_hat0, x_hat0], dim=1))
        x_hat = (x_hat0 + r).clamp(0.0, 1.0)
        return x_hat


def _to_pil(t: torch.Tensor) -> Image.Image:
    return tensor_to_pil(t, denormalize=_denormalize_tensor)

def _to_uint8_bchw(imgs: torch.Tensor) -> torch.Tensor:
    """
    imgs: (N,C,H,W) in [0,1] float -> uint8 tensor (N,C,H,W) without normalization.
    """
    return (imgs.clamp(0,1) * 255.0 + 0.5).floor().to(torch.uint8)

def _safe_norm01(img: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Per-image normalization to [0,1] that avoids collapse when max≈min.
    img: (C,H,W) float
    """
    mn = img.amin(dim=(1,2), keepdim=True)
    mx = img.amax(dim=(1,2), keepdim=True)
    rng = torch.maximum(mx - mn, torch.tensor(eps, device=img.device, dtype=img.dtype))
    return (img - mn) / rng

def _boost_contrast01(img: torch.Tensor, k: float) -> torch.Tensor:
    """
    Linear contrast boost around 0.5 in [0,1]: y = clamp((x-0.5)*k + 0.5, 0, 1)
    """
    if k is None or k <= 1.0:
        return img
    return ((img - 0.5) * k + 0.5).clamp(0, 1)

def _save_tensor_png(path: Path, img_chw: torch.Tensor) -> None:
    """Save a single (C,H,W) float [0,1] as PNG without any normalization magic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = img_chw.detach().clamp(0,1).to("cpu")
    pil = Image.fromarray((img.permute(1,2,0).numpy()*255.0 + 0.5).astype("uint8"))
    pil.save(path)

def _draw_border_uint8(img_chw_u8: torch.Tensor, thickness: int = 1) -> torch.Tensor:
    """Draw a white border on a uint8 (C,H,W) image tensor."""
    c,h,w = img_chw_u8.shape
    img = img_chw_u8.clone()
    img[:, :thickness, :] = 255
    img[:, -thickness:, :] = 255
    img[:, :, :thickness] = 255
    img[:, :, -thickness:] = 255
    return img

def _annotate_minmax(tile: torch.Tensor) -> str:
    return f"min={tile.amin().item():.4f} max={tile.amax().item():.4f} mean={tile.mean().item():.4f}"


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


# --------------------------- Latent helpers ---------------------------------
def _slerp(zA: torch.Tensor, zB: torch.Tensor, t: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Spherical interpolation along unit hypersphere per pair in the batch.
    zA,zB: (B, Z); t: scalar or (T,) or (T,B)
    Returns: (T,B,Z) if t is (T,) else (B,Z) for scalar.
    """
    def _unit(x): return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))
    a = _unit(zA); b = _unit(zB)
    dot = (a * b).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)  # (B,)
    omega = torch.acos(dot)                                 # (B,)
    so = torch.sin(omega).clamp_min(eps)
    if t.ndim == 0:
        t = t.view(1)
    if t.ndim == 1:
        t = t[:, None]                                     # (T,1)
    t = t.to(zA.device, zA.dtype)
    # (T,B) weights
    w1 = torch.sin((1 - t) * omega[None, :]) / so[None, :]
    w2 = torch.sin(t * omega[None, :]) / so[None, :]
    out = w1[..., None] * a[None, :, :] + w2[..., None] * b[None, :, :]
    return out


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
    viz_diag: bool,
    interp_context: str,
    normalize_mids: bool,
    interp_vis_contrast: float,
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
    # Decode endpoints with refinement context (when available)
    dec_zA = dec(zA, x=A_dev)
    dec_zB = dec(zB, x=B_dev)

    t_vals = torch.tensor([0.0, 0.01, 0.2, 0.4, 0.6, 0.8, 0.99, 1.0], device=device)
    mid_t_vals = t_vals[1:-1]
    if mid_t_vals.numel() > 0:
        if False: # cfg.interp_mode.lower() == "slerp":
            z_interp = _slerp(zA, zB, mid_t_vals)  # (Tmid, B, Z) on unit sphere
        else:
            z_interp = torch.stack([(1.0 - t) * zA + t * zB for t in mid_t_vals], dim=0)  # (Tmid, B, Z)

            #z_interp = torch.stack([(1.0 - t) * zA.detach() + t * zB.detach() for t in mid_t_vals], dim=0)

        Tmid, Bsz, zdim = z_interp.shape
        _, C, H, W = A_dev.shape  # A_dev: (B, 3, H, W)
        z_interp_flat = z_interp.reshape(Tmid * Bsz, zdim)

        # ---- Choose how to decode mids ----
        interp_context = interp_context.lower()
        assert interp_context in ("none", "self", "blend", "endpoints"), f"bad interp_context={interp_context}"
        if interp_context == "none":
            interp_decoded_flat = dec.decode_no_refine(z_interp_flat)
            ctx_used = "none"
        elif interp_context == "self":
            interp_decoded_flat = dec.decode_self_refine(z_interp_flat)
            ctx_used = "self"
        elif interp_context == "blend":
            # build blended context x = (1-t)*A + t*B matching each z_interp (Tmid*B, 3, H, W)
            # expand A_dev/B_dev to (Tmid,B,3,H,W) with per-row t
            lam = mid_t_vals.view(Tmid, 1, 1, 1, 1)  # (Tmid,1,1,1,1)
            x_blend = (1.0 - lam) * A_dev[None, ...] + lam * B_dev[None, ...]  # (Tmid,B,3,H,W)
            x_blend_flat = x_blend.reshape(Tmid * Bsz, C, H, W)
            interp_decoded_flat = dec(z_interp_flat, x=x_blend_flat)
            ctx_used = "blend"
        else:  # 'endpoints'
            # Use A as context for t<0.5, else B
            # Build per-sample contexts aligned with z_interp_flat ordering
            x_ctx = []
            for ti, tval in enumerate(mid_t_vals.tolist()):
                if tval < 0.5:
                    x_ctx.append(A_dev)
                else:
                    x_ctx.append(B_dev)
            x_ctx = torch.stack(x_ctx, dim=0).reshape(Tmid * Bsz, C, H, W)
            interp_decoded_flat = dec(z_interp_flat, x=x_ctx)
            ctx_used = "endpoints"

        # Prepare a copy for optional per-tile normalization/contrast (viz only)
        mids_flat = interp_decoded_flat
        if normalize_mids:
            mids_flat = torch.stack([_safe_norm01(m) for m in mids_flat], dim=0)
            mids_flat = _boost_contrast01(mids_flat, interp_vis_contrast)

        # Quick sanity check to help diagnose "empty" outputs
        if True:
            mn, mx = interp_decoded_flat.min().item(), interp_decoded_flat.max().item()
            #print(f"[interp sanity] min={mn:.4f} max={mx:.4f} shape={tuple(interp_decoded_flat.shape)}")
            print(f"[interp sanity] ctx={ctx_used} min={mn:.4f} max={mx:.4f} shape={tuple(mids_flat.shape)} (vis k={interp_vis_contrast})")

        # Final tensor for grid layout (no normalization here; keep the true values)
        interp_decoded = interp_decoded_flat.reshape(Tmid, Bsz, C, H, W)

        # --- Viz diagnostics: dump raw mids and a synthetic pattern through the same pipe ---
        if viz_diag:
            print("VIZ DIAG")
            dump_dir = out_path.parent / (out_path.stem + "_mids_dump")
            dump_dir.mkdir(parents=True, exist_ok=True)
            # save first 8 mids as standalone PNGs (no grids, no normalize)
            n_dump = min(8, mids_flat.shape[0])
            for i in range(n_dump):
                _save_tensor_png(dump_dir / f"mid_{i:02d}.png", mids_flat[i])
                print("[viz_diag]", i, _annotate_minmax(mids_flat[i]))
            # also create a synthetic checker/ramp and save it via the SAME path
            yy = torch.linspace(0,1,H, device=mids_flat.device).view(1,H,1).expand(1,H,W)
            xx = torch.linspace(0,1,W, device=mids_flat.device).view(1,1,W).expand(1,H,W)
            checker = (((xx*16).floor() + (yy*16).floor()) % 2)  # 0/1 checker
            synth = torch.cat([yy, xx, checker], dim=0).clamp(0,1)  # (3,H,W)
            _save_tensor_png(dump_dir / "synthetic_check.png", synth)
            print("[viz_diag] wrote synthetic_check.png to ensure save path renders contrast")
    else:
        interp_decoded = torch.empty(0, Bsz, 3, H, W, device=device)

    z_norm = torch.linalg.norm(zB - zA, dim=1).cpu()
    mid_t_vals_cpu = mid_t_vals.detach().cpu().tolist()

    rows: List[List[TileSpec]] = []
    for idx in range(Bsz):
        labelA = short_traj_state_label(pA[idx])
        labelB = short_traj_state_label(pB[idx])

        row: List[TileSpec] = [
            TileSpec(
                image=_to_pil(A[idx]),
                top_label=labelA,
                bottom_label=f"‖zB−zA‖={z_norm[idx]:.2f}",
                bottom_color=(255, 255, 0),
            ),
            TileSpec(
                image=_to_pil(dec_zA[idx]),
                top_label="t=0.0 (A)",
                top_color=(220, 220, 255),
            ),
        ]

        for interp_idx, t_val in enumerate(mid_t_vals_cpu):
            row.append(
                TileSpec(
                    image=_to_pil(interp_decoded[interp_idx, idx]),
                    top_label=f"t={t_val:.2f} ({ctx_used})",
                    top_color=(200, 255, 200),
                )
            )

        row.extend(
            [
                TileSpec(
                    image=_to_pil(dec_zB[idx]),
                    top_label="t=1.0 (B)",
                    top_color=(220, 220, 255),
                ),
                TileSpec(
                    image=_to_pil(B[idx]),
                    top_label=labelB,
                ),
            ]
        )

        rows.append(row)

    render_image_grid(rows, out_path, tile_size=(W, H))


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
    A_rec = dec(zA, x=A_dev).to("cpu")
    B_rec = dec(zB, x=B_dev).to("cpu")

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
    # Visualization-only knobs
    interp_vis_contrast: float = 1.0
    normalize_mids: bool = False
    interp_context: str = "endpoints"  # 'none' | 'self' | 'blend' | 'endpoints'

    # Refinement options
    use_refine: bool = False
    refine_ch: int = 64
    self_refine_mids: bool = False

    # Teach base decoder (no-refine) to reconstruct a bit
    lambda_no_refine: float = 0.1

    # Teach refinement to use BLENDED context x=(1-t)A+tB for mixed latents
    lambda_blend_ctx: float = 0.0    # try 0.1 to turn on

    # Teach self-refine path (no external context) to do something useful
    lambda_self_refine: float = 0.0  # try 0.05–0.1

    # Latent mixup (training only)
    lambda_latent_mix: float = 0.0   # set >0 to enable (e.g., 0.1)
    mixup_alpha: float = 0.5         # Beta(alpha,alpha), or fixed 0<lambda<1 if <=0

    # Visualization diagnostics (no training effect)
    viz_diag: bool = False           # dump raw mids, add borders, synthetic mids,



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
    dec = Decoder(cfg.z_dim, use_refine=cfg.use_refine, refine_ch=cfg.refine_ch, self_refine_mids=cfg.self_refine_mids).to(device)
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

            # Pass original images as context so the decoder can refine against [x, x_hat]
            xA_rec = dec(zA, x=A)
            xB_rec = dec(zB, x=B)
            loss_l1 = F.l1_loss(xA_rec, A) + F.l1_loss(xB_rec, B)

            # --- Base-path supervision (no refine) so mids aren’t gray ---
            loss_no_ref = 0.0
            if cfg.lambda_no_refine > 0.0:
                xA_base = dec.decode_no_refine(zA)
                xB_base = dec.decode_no_refine(zB)
                loss_no_ref = F.l1_loss(xA_base, A) + F.l1_loss(xB_base, B)

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

            # --- Base-path supervision (no refine) so mids aren’t gray ---
            loss_no_ref = 0.0
            if cfg.lambda_no_refine > 0.0:
                xA_base = dec.decode_no_refine(zA)
                xB_base = dec.decode_no_refine(zB)
                loss_no_ref = F.l1_loss(xA_base, A) + F.l1_loss(xB_base, B)

            # ---- Latent mixup regularizer (teach decoder to handle interpolations) ----
            mixup_loss = 0.0
            if cfg.lambda_latent_mix > 0.0:
                Bsz = A.shape[0]
                if cfg.mixup_alpha > 0:
                    lam = torch.distributions.Beta(cfg.mixup_alpha, cfg.mixup_alpha).sample((Bsz,)).to(device)
                else:
                    lam = torch.full((Bsz,), 0.5, device=device)  # fixed
                lam = lam.view(Bsz, 1, 1, 1)  # broadcast over CHW
                # Latent + image mix
                z_mix = (lam.view(Bsz,1) * zA + (1 - lam).view(Bsz,1) * zB)
                x_mix = lam * A + (1 - lam) * B
                # Decode with (optional) self-refine using x_mix as context if you want; safer is no context:
                x_mix_rec = dec.decode_no_refine(z_mix)
                mixup_loss = F.l1_loss(x_mix_rec, x_mix)

            # ---- Blended-context supervision (teaches the REFINE path for "blend" viz) ----
            blend_ctx_loss = 0.0
            if cfg.lambda_blend_ctx > 0.0:
                Bsz = A.shape[0]
                # use the same lambda as above if available, otherwise sample fresh
                if not ('lam' in locals()):
                    if cfg.mixup_alpha > 0:
                        lam = torch.distributions.Beta(cfg.mixup_alpha, cfg.mixup_alpha).sample((Bsz,)).to(device)
                    else:
                        lam = torch.full((Bsz,), 0.5, device=device)
                    lam = lam.view(Bsz, 1, 1, 1)
                # build blended latent and blended context image
                z_blend = (lam.view(Bsz,1) * zA + (1 - lam).view(Bsz,1) * zB)
                x_blend = lam * A + (1 - lam) * B
                # decode WITH blended context through the normal forward (refine) path
                x_blend_rec = dec(z_blend, x=x_blend)
                blend_ctx_loss = F.l1_loss(x_blend_rec, x_blend)

            # ---- Self-refine supervision (teach 'self' viz mode) ----
            self_refine_loss = 0.0
            if cfg.lambda_self_refine > 0.0:
                # endpoints
                xA_self = dec.self_refine_train(zA)
                xB_self = dec.self_refine_train(zB)
                self_refine_loss = F.l1_loss(xA_self, A) + F.l1_loss(xB_self, B)
                # optional: blended mids too (pairs with blend_ctx supervision)
                if cfg.lambda_blend_ctx > 0.0:
                    if not ('lam' in locals()):
                        Bsz = A.shape[0]
                        if cfg.mixup_alpha > 0:
                            lam = torch.distributions.Beta(cfg.mixup_alpha, cfg.mixup_alpha).sample((Bsz,)).to(device)
                        else:
                            lam = torch.full((Bsz,), 0.5, device=device)
                        lam = lam.view(Bsz, 1, 1, 1)
                    z_blend = lam.view(Bsz,1) * zA + (1 - lam).view(Bsz,1) * zB
                    x_blend = lam * A + (1 - lam) * B
                    x_blend_self = dec.self_refine_train(z_blend)
                    self_refine_loss = self_refine_loss + F.l1_loss(x_blend_self, x_blend)

            loss = (
                cfg.lambda_l1 * loss_l1 +
                cfg.lambda_ms_ssim * loss_ms +
                cfg.lambda_patch * loss_patch +
                cfg.lambda_latent_mix * mixup_loss +
                cfg.lambda_no_refine * loss_no_ref +
                cfg.lambda_self_refine * self_refine_loss +
                cfg.lambda_blend_ctx * blend_ctx_loss
            )


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
                if cfg.lambda_latent_mix > 0.0:
                    loss_terms.append(f"Mix {float(mixup_loss):.4f}")
                if cfg.lambda_no_refine > 0.0:
                    loss_terms.append(f"NoRef {float(loss_no_ref):.4f}")
                if cfg.lambda_blend_ctx > 0.0:
                    loss_terms.append(f"BlendCtx {float(blend_ctx_loss):.4f}")
                loss_detail = " ".join(loss_terms)

                print(
                    f"[{h:02d}:{m:02d}:{s:02d}] "
                    f"ep {ep:02d} step {global_step:06d} | "
                    f"loss {avg_loss:.4f} | {loss_detail} |"
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
                    viz_diag=cfg.viz_diag,
                    interp_context=cfg.interp_context,
                    normalize_mids=cfg.normalize_mids,
                    interp_vis_contrast=cfg.interp_vis_contrast,
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
                xA_rec = dec(zA, x=A)
                xB_rec = dec(zB, x=B)
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
    ap.add_argument("--viz_every", type=int, default=50)
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
    ap.add_argument("--interp_context", type=str, default="endpoints",
                    choices=["none","self","blend","endpoints"],
                    help="Decode mids with: no context, self-refine, blended A/B context, or endpoint context.")
    ap.add_argument("--use_refine", action="store_true",
                    help="Enable residual/refinement head: predicts r from [x, x_hat] and applies x_hat+=r.")
    ap.add_argument("--refine_ch", type=int, default=64,
                    help="Hidden channels for the refinement head.")
    ap.add_argument("--interp_mode", type=str, default="lerp", choices=["lerp", "slerp"],
                    help="Interpolation mode in latent space for visualization.")
    ap.add_argument("--lambda_latent_mix", type=float, default=0.0,
                    help="Weight for latent mixup reconstruction loss. Try 0.1.")
    ap.add_argument("--mixup_alpha", type=float, default=0.5,
                    help="Beta(alpha,alpha) for mixup λ. Use <=0 for fixed 0.5.")
    ap.add_argument("--self_refine_mids", action="store_true",
                    help="Allow a self-refine pass for interpolated tiles (uses x_hat as its own context).")
    ap.add_argument("--lambda_self_refine", type=float, default=0.0,
                    help="Weight for self-refine loss (endpoints + optional blended mids). Try 0.05.")
    ap.add_argument("--viz_diag", action="store_true",
                    help="Visualization diagnostics: dump raw mids, add borders, and write a synthetic check image.")
    ap.add_argument("--lambda_blend_ctx", type=float, default=0.0,
                    help="Weight for blended-context refine loss: supervise dec(z_mix, x=(1-t)A+tB) to match the blended image. Try 0.1.")


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
        use_refine=args.use_refine,
        interp_context=args.interp_context,
        refine_ch=args.refine_ch,
        self_refine_mids=args.self_refine_mids,
        lambda_latent_mix=args.lambda_latent_mix,
        mixup_alpha=args.mixup_alpha,
        viz_diag=args.viz_diag,
        lambda_blend_ctx=args.lambda_blend_ctx,
        lambda_self_refine=args.lambda_self_refine,
    )
    train(cfg)


if __name__ == "__main__":
    main()
