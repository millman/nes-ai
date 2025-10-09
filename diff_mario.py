#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mario difference vectors (bucket-free) with spatial factor heatmaps.

Architecture:
  - Shared encoder -> latent vectors z_a, z_b AND a mid-level feature map (from ResNet18 layer2)
  - Self-attention mixer on [z_a, z_b, z_b - z_a] -> h
  - Low-rank PSD metric head: B(h) -> factor projections -> energies m (size r), scalar D (train-only)
  - Localizer head (optional): from mid-level feature maps (F_a, F_b) -> K=r heatmaps (per-factor spatial maps)

Losses:
  - Uncertainty-weighted robust regression: D ≈ log1p(Δt)
  - Gated ranking using MS-SSIM proxy (loop-closure tolerant)
  - Factor decorrelation (VICReg/BT-style) and entropy over factor energy distribution (continuous rank emergence)
  - Photometric invariance on factors (optional)
  - Heatmap consistency: spatial sums of heatmaps ≈ factor energies; global consistency: sum over maps ≈ D
  - Tiny-shift equivariance for heatmaps (integer pixel roll)

Visualization (saved periodically):
  - side-by-side x_a | x_b (unnormalized)
  - scree / top-K factor bars
  - top-K heatmaps overlaid on x_b

Run:
  python diffvec_mario_train_vis_maps.py --traj_dir data.image_distance.train_levels_1_2 --out_dir out.diffvec_maps
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tyro
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision.models import resnet18, ResNet18_Weights

# ----------------------------- Device -----------------------------------------
def pick_device(pref: Optional[str]) -> torch.device:
    if pref:
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ------------------------ Normalization helpers --------------------------------
INV_MEAN = torch.tensor([0.485, 0.456, 0.406])
INV_STD  = torch.tensor([0.229, 0.224, 0.225])

def default_transform(img_size: Tuple[int,int]=(224,224)) -> T.Compose:
    weights = ResNet18_Weights.DEFAULT
    base = weights.transforms()
    if img_size != (224,224):
        base = T.Compose([
            T.Resize(min(img_size)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=weights.meta["mean"], std=weights.meta["std"]),
        ])
    return base

def unnormalize(img: torch.Tensor) -> torch.Tensor:
    if img.ndim == 4:  # batched NCHW
        mean = INV_MEAN[None, :, None, None]
        std = INV_STD[None, :, None, None]
    elif img.ndim == 3:  # single CHW image
        mean = INV_MEAN[:, None, None]
        std = INV_STD[:, None, None]
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(img.shape)}")
    mean = mean.to(img.device, img.dtype)
    std = std.to(img.device, img.dtype)
    return (img * std + mean).clamp(0, 1)

# ----------------------- MS-SSIM proxy (stop-grad) -----------------------------
def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = (gauss / gauss.sum()).unsqueeze(0)
    window_2d = (gauss.t() @ gauss).unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()

def _ssim_components(x: torch.Tensor, y: torch.Tensor, window: torch.Tensor,
                     data_range: float=1.0, k1: float=0.01, k2: float=0.03) -> Tuple[torch.Tensor, torch.Tensor]:
    padding = window.shape[-1] // 2
    mu_x = F.conv2d(x, window, padding=padding, groups=x.shape[1])
    mu_y = F.conv2d(y, window, padding=padding, groups=y.shape[1])

    mu_x_sq = mu_x.pow(2);  mu_y_sq = mu_y.pow(2);  mu_xy = mu_x * mu_y
    sigma_x_sq = F.conv2d(x * x, window, padding=padding, groups=x.shape[1]) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=padding, groups=y.shape[1]) - mu_y_sq
    sigma_xy   = F.conv2d(x * y, window, padding=padding, groups=x.shape[1]) - mu_xy

    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2

    numerator   = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    ssim_map = numerator / denominator
    cs_map   = (2 * sigma_xy + C2) / (sigma_x_sq + sigma_y_sq + C2)

    ssim_val = ssim_map.mean(dim=(1,2,3))
    cs_val   = cs_map.mean(dim=(1,2,3))
    return ssim_val, cs_val

def ms_ssim(x: torch.Tensor, y: torch.Tensor, weights: Optional[torch.Tensor]=None) -> torch.Tensor:
    if weights is None:
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=x.device, dtype=x.dtype)
    window_size = 11
    sigma = 1.5
    channels = x.shape[1]
    window = _gaussian_window(window_size, sigma, channels, x.device, x.dtype)
    levels = weights.shape[0]
    mssim, mcs = [], []

    x_scaled, y_scaled = x, y
    for _ in range(levels):
        ssim_val, cs_val = _ssim_components(x_scaled, y_scaled, window)
        mssim.append(ssim_val); mcs.append(cs_val)
        x_scaled = F.avg_pool2d(x_scaled, kernel_size=2, stride=2)
        y_scaled = F.avg_pool2d(y_scaled, kernel_size=2, stride=2)

    mssim_tensor = torch.stack(mssim, dim=0)
    mcs_tensor   = torch.stack(mcs[:-1], dim=0)
    eps = torch.finfo(mssim_tensor.dtype).eps
    mssim_tensor = mssim_tensor.clamp(min=eps, max=1.0)
    mcs_tensor = mcs_tensor.clamp(min=eps, max=1.0)
    pow1 = weights[:-1].unsqueeze(1)
    pow2 = weights[-1]
    ms_prod = torch.prod(mcs_tensor ** pow1, dim=0) * (mssim_tensor[-1] ** pow2)
    return ms_prod.mean()

def ms_ssim_distance(x_norm: torch.Tensor, y_norm: torch.Tensor) -> torch.Tensor:
    return 1.0 - ms_ssim(unnormalize(x_norm), unnormalize(y_norm))

# ----------------------------- Dataset -----------------------------------------
class PairDataset(Dataset):
    """
    Expects:
      traj_*/states/*.png
    Returns (x_a, x_b, delta_t, episode_id)
    """
    def __init__(self, root_dir: str,
                 image_size: Tuple[int,int]=(224,224),
                 sample_prob: Dict[str, float] | None = None,
                 small: Tuple[int,int]=(1,4),
                 mid: Tuple[int,int]=(5,60),
                 large: Tuple[int,int]=(61,400),
                 cross_episode_prob: float=0.10,
                 photometric_jitter: bool=True,
                 max_trajs: Optional[int]=None) -> None:
        self.transform = default_transform(image_size)
        self.photometric_jitter = photometric_jitter
        self.sample_prob = sample_prob or {"small":0.5, "mid":0.3, "large":0.2}
        self.small, self.mid, self.large = small, mid, large
        self.cross_episode_prob = cross_episode_prob

        self.trajs: List[List[Path]] = []
        traj_count = 0
        for traj_path in sorted(Path(root_dir).iterdir()):
            if not traj_path.is_dir(): continue
            states_dir = traj_path / "states"
            if not states_dir.is_dir(): continue
            files = sorted([p for p in states_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
            if len(files) < 6:
                continue
            self.trajs.append(files)
            traj_count += 1
            if max_trajs and traj_count >= max_trajs: break

        if not self.trajs:
            raise ValueError(f"No trajectories with images found under {root_dir}")

        self.all_len = sum(len(t) for t in self.trajs)
        self.jitter = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02)

    def __len__(self) -> int:
        return self.all_len

    def _pick_gap(self) -> int:
        r = random.random()
        if r < self.sample_prob["small"]:
            lo, hi = self.small
        elif r < self.sample_prob["small"] + self.sample_prob["mid"]:
            lo, hi = self.mid
        else:
            lo, hi = self.large
        if hi < lo: lo, hi = hi, lo
        return random.randint(lo, hi)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        ep = random.randrange(len(self.trajs))
        files = self.trajs[ep]
        n = len(files)
        i = random.randrange(n - 2)
        gap = self._pick_gap()
        j = i + gap
        if j >= n:
            j = n - 1
            gap = j - i

        if random.random() < self.cross_episode_prob:
            ep2 = ep
            trials = 0
            while ep2 == ep and trials < 5:
                ep2 = random.randrange(len(self.trajs)); trials += 1
            files_b = self.trajs[ep2]
            j = min(j, len(files_b)-1)
            xb_path = files_b[j]; ep_id_b = ep2
        else:
            xb_path = files[j];   ep_id_b = ep

        xa_path = files[i]

        with Image.open(xa_path).convert("RGB") as img_a:
            if self.photometric_jitter and random.random() < 0.2:
                img_a = self.jitter(img_a)
            xa = self.transform(img_a)
        with Image.open(xb_path).convert("RGB") as img_b:
            if self.photometric_jitter and random.random() < 0.2:
                img_b = self.jitter(img_b)
            xb = self.transform(img_b)

        return xa, xb, int(gap), int(ep_id_b)

# ----------------------------- Model -------------------------------------------
class ResNet18Mid(nn.Module):
    """
    ResNet18 backbone producing:
      - mid-level feature map (after layer2): [B, C_mid=128, H/8, W/8]
      - vector latent z: GAP after layer4 -> Linear -> z_dim
    """
    def __init__(self, z_dim: int=256):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        net = resnet18(weights=weights)
        # Keep stem -> layer1 -> layer2 (mid), plus layer3 and layer4 for vector
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)   # /4
        self.layer1 = net.layer1                                               # /4
        self.layer2 = net.layer2                                               # /8  (mid)
        self.layer3 = net.layer3                                               # /16
        self.layer4 = net.layer4                                               # /32
        self.avgpool = net.avgpool
        self.proj = nn.Linear(512, z_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        mid = self.layer2(x)      # [B,128,H/8,W/8]
        y = self.layer3(mid)
        y = self.layer4(y)
        y = self.avgpool(y)       # [B,512,1,1]
        y = torch.flatten(y, 1)   # [B,512]
        z = self.proj(y)          # [B,z_dim]
        return z, mid

class DiffTransformer(nn.Module):
    def __init__(self, d: int, heads: int=8, depth: int=3):
        super().__init__()
        self.embed = nn.Linear(3*d, d)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d, nhead=heads,
                                       dim_feedforward=4*d,
                                       activation="gelu",
                                       batch_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d)

    def forward(self, za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([za, zb, zb - za], dim=-1)
        x = self.embed(x).unsqueeze(1)  # [B,1,d]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x.squeeze(1))  # [B,d]

class MetricHead(nn.Module):
    def __init__(self, d: int, r: int=16):
        super().__init__()
        self.B_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d*r))
        self.sigma = nn.Linear(d, 1)
        self.d, self.r = d, r

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = self.B_mlp(h).view(-1, self.d, self.r)                          # [B,d,r]
        proj = torch.bmm(B.transpose(1,2), h.unsqueeze(-1)).squeeze(-1)     # [B,r]
        m = proj.pow(2)                                                     # [B,r]
        D = m.sum(dim=-1)                                                   # [B]
        log_sigma = self.sigma(h).squeeze(-1)
        return {"B": B, "proj": proj, "m": m, "D": D, "log_sigma": log_sigma}

class Localizer(nn.Module):
    """
    Tiny UNet-ish head to produce K=r heatmaps from mid-level feature maps of both frames.
    Input: concat [F_a, F_b, F_a - F_b, F_a * F_b] along channels.
    Output: maps [B, r, Hm, Wm] in R+ via softplus.
    """
    def __init__(self, c_in: int=128, r: int=16):
        super().__init__()
        cin = c_in * 4
        self.down1 = nn.Sequential(nn.Conv2d(cin, 128, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(128, 128, 3, padding=1), nn.GELU())
        self.pool1 = nn.MaxPool2d(2)     # /2
        self.down2 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(256, 256, 3, padding=1), nn.GELU())
        self.up1   = nn.ConvTranspose2d(256, 128, 2, stride=2)   # x2
        self.fuse1 = nn.Sequential(nn.Conv2d(128+128, 128, 3, padding=1), nn.GELU())
        self.out   = nn.Conv2d(128, r, 1)

    def forward(self, Fa: torch.Tensor, Fb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([Fa, Fb, Fa - Fb, Fa * Fb], dim=1)  # [B,4C,H,W]
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        y  = self.up1(x2)
        y  = self.fuse1(torch.cat([y, x1], dim=1))
        maps = F.softplus(self.out(y))  # [B,r,Hm,Wm], non-negative
        return maps

class DiffVecModel(nn.Module):
    def __init__(self, z_dim: int=256, heads: int=8, depth: int=3, r: int=16, use_localizer: bool=True):
        super().__init__()
        self.enc = ResNet18Mid(z_dim)
        self.mix = DiffTransformer(z_dim, heads=heads, depth=depth)
        self.head = MetricHead(z_dim, r=r)
        self.use_localizer = use_localizer
        if use_localizer:
            self.loc = Localizer(c_in=128, r=r)

    def forward(self, xa: torch.Tensor, xb: torch.Tensor, return_maps: bool=False) -> Dict[str, torch.Tensor]:
        za, Fa = self.enc(xa)    # za: [B,d], Fa: [B,128,H/8,W/8]
        zb, Fb = self.enc(xb)
        h = self.mix(za, zb)
        out = self.head(h)
        out["h"] = h
        if self.use_localizer or return_maps:
            maps = self.loc(Fa, Fb)   # [B,r,Hm,Wm]
            out["maps"] = maps
        return out

# ------------------------------- Losses ----------------------------------------
def g_fn(delta_t: torch.Tensor) -> torch.Tensor:
    return torch.log1p(delta_t.float())

def gaussian_nll_like(D: torch.Tensor, target: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    sigma = F.softplus(log_sigma) + 1e-6
    resid = torch.abs(D - target)
    return (resid / sigma + torch.log(sigma)).mean()

def gated_margin_ranking(D: torch.Tensor, xa: torch.Tensor, xb: torch.Tensor, eps: float=0.02) -> torch.Tensor:
    B = D.shape[0]
    if B < 2:
        return D.new_zeros(())
    D2  = torch.roll(D,  shifts=1, dims=0)
    xa2 = torch.roll(xa, shifts=1, dims=0)
    xb2 = torch.roll(xb, shifts=1, dims=0)
    with torch.no_grad():
        S1 = ms_ssim_distance(xa, xb)
        S2 = ms_ssim_distance(xa2, xb2)
    y = torch.sign(S1 - S2)
    mask = (torch.abs(S1 - S2) > eps) & (y != 0)
    if mask.sum() == 0:
        return D.new_zeros(())
    loss = F.softplus(-y[mask] * (D[mask] - D2[mask])).mean()
    return loss

def decorrelation_loss(proj: torch.Tensor) -> torch.Tensor:
    B, r = proj.shape
    if B < 2:
        return proj.new_zeros(())
    X = proj - proj.mean(dim=0, keepdim=True)
    std = X.std(dim=0, unbiased=False, keepdim=True) + 1e-6
    X = X / std
    C = (X.T @ X) / (B - 1)
    off = C - torch.diag(torch.diag(C))
    return off.pow(2).mean()

def entropy_on_factors(m: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    s = m.sum(dim=-1, keepdim=True) + eps
    p = m / s
    H = -(p * torch.log(p + eps)).sum(dim=-1)
    return H.mean()

def heatmap_mass_consistency(maps: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """
    Encourage per-factor map mass (after batch-wise scaling) to match energy m.
    Scaling factor is detached so gradients flow only into maps.
    """
    eps = 1e-6
    mass = maps.flatten(2).sum(dim=-1)  # [B,r]
    with torch.no_grad():
        scale = (m.sum(dim=-1, keepdim=True) + eps) / (mass.sum(dim=-1, keepdim=True) + eps)
    mass_scaled = mass * scale
    return F.l1_loss(mass_scaled, m.detach())

def heatmap_global_consistency(maps: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Encourage global map energy (after normalised scaling) to match scalar distance.
    The scaling factor is computed from running batch statistics but treated as
    constant for the backward pass to avoid collapsing towards degenerate zeros.
    """
    eps = 1e-6
    D_map = maps.mean(dim=(2, 3)).sum(dim=1)  # [B]
    with torch.no_grad():
        scale = (D.mean() + eps) / (D_map.mean() + eps)
    D_map_scaled = D_map * scale
    return F.l1_loss(D_map_scaled, D.detach())

def heatmap_spatial_entropy(maps: torch.Tensor) -> torch.Tensor:
    """
    Maximize spatial entropy of each heatmap to avoid collapse to tiny blobs.
    Returns negative entropy so adding with positive lambda encourages spread.
    """
    eps = 1e-8
    flat = maps.flatten(2)
    mass = flat.sum(dim=-1, keepdim=True)
    p = flat / (mass + eps)
    entropy = -(p * torch.log(p + eps)).sum(dim=-1)  # [B,r]
    return -entropy.mean()

def heatmap_shift_equivariance(xb: torch.Tensor, model: DiffVecModel, shift_px: int=2) -> torch.Tensor:
    """
    Tiny integer roll: predict maps for xb and for rolled-xb (xa stays same),
    require rolled maps ≈ roll(maps).
    """
    if shift_px <= 0:
        return xb.new_zeros(())
    B, C, H, W = xb.shape
    # Use a small subset for compute
    idx = torch.arange(min(B, 8), device=xb.device)
    xa_s, xb_s = xb[idx], xb[idx]  # reuse xb as xa for equivariance probe (cheap)
    out0 = model(xa_s, xb_s, return_maps=True)
    maps0 = out0["maps"].detach()  # [b,r,Hm,Wm]
    # roll inputs
    dx = random.choice([-shift_px, shift_px]); dy = random.choice([-shift_px, shift_px])
    xb_roll = torch.roll(xb_s, shifts=(dy, dx), dims=(-2, -1))
    out1 = model(xa_s, xb_roll, return_maps=True)
    maps1 = out1["maps"]
    # up/downsample difference in grid resolution
    # maps are at Hm,Wm; roll by proportional amount
    Hm, Wm = maps0.shape[-2:]
    dy_m = int(round(dy * (Hm / H)))
    dx_m = int(round(dx * (Wm / W)))
    maps0r = torch.roll(maps0, shifts=(dy_m, dx_m), dims=(-2, -1))
    return F.l1_loss(maps1, maps0r)

# ------------------------------- Viz -------------------------------------------
def summarize_maps(maps: torch.Tensor) -> Dict[str, float]:
    B, r, Hm, Wm = maps.shape
    m = maps.detach()
    flat = m.view(B * r, -1)
    mass = m.flatten(2).sum(dim=-1)
    stats = {
        "mean": float(m.mean().item()),
        "std": float(m.std().item()),
        "min": float(m.min().item()),
        "max": float(m.max().item()),
        "mass_mean": float(mass.mean().item()),
        "mass_std": float(mass.std().item()),
        "spatial_std": float(flat.std(dim=-1).mean().item()),
    }
    return stats

def plot_scree_and_bars(m: torch.Tensor, out_path: Path, title: str, topk: int = 10) -> None:
    r = m.numel()
    vals, _ = torch.sort(m, descending=True)
    xs = list(range(1, r+1))
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(xs, vals.numpy())
    plt.title(f"Scree (sorted factor energies)\n{title}")
    plt.xlabel("factor index (sorted)"); plt.ylabel("energy")
    plt.subplot(1,2,2)
    k = min(topk, r)
    plt.bar(list(range(k)), vals[:k].numpy())
    plt.title("Top factors")
    plt.xlabel("rank"); plt.ylabel("energy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, bbox_inches="tight"); plt.close()

def save_pair(xa: torch.Tensor, xb: torch.Tensor, out_path: Path, caption: str="", row_captions: Optional[List[str]] = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_rows = min(4, xa.shape[0])
    to_pil = T.ToPILImage()
    xa0 = unnormalize(xa[0:1]).squeeze(0).cpu()
    xb0 = unnormalize(xb[0:1]).squeeze(0).cpu()
    im_a0 = to_pil(xa0)
    im_b0 = to_pil(xb0)
    W, H = im_a0.size
    pad = 6
    canvas_w = W * 2 + pad
    canvas_h = H * max_rows + pad * (max_rows - 1)
    canvas = Image.new('RGB', (canvas_w, canvas_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for row in range(max_rows):
        xa_i = unnormalize(xa[row:row+1]).squeeze(0).cpu()
        xb_i = unnormalize(xb[row:row+1]).squeeze(0).cpu()
        im_a = to_pil(xa_i)
        im_b = to_pil(xb_i)
        y = row * (H + pad)
        canvas.paste(im_a, (0, y))
        canvas.paste(im_b, (W + pad, y))
        if row_captions and row < len(row_captions):
            draw.text((5, y + 5), row_captions[row], fill=(255, 255, 0))
    if caption:
        draw.text((5, 5), caption, fill=(255, 255, 0))
    canvas.save(out_path)

def overlay_heatmaps_on_xb(xb: torch.Tensor, maps: torch.Tensor, out_path: Path, topk: int=6) -> None:
    """
    xb: [B,3,H,W] (normalized); maps: [B,r,Hm,Wm] (non-negative).
    Saves a grid with xb and top-k factor overlays.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    batch = xb.shape[0]
    rows = min(4, batch)
    k = min(topk, maps.shape[1])
    pad = 6
    to_pil = T.ToPILImage()
    xb0 = unnormalize(xb[0:1]).squeeze(0).cpu()
    base0 = to_pil(xb0)
    W, H = base0.size
    grid_w = (W + pad) * (k + 1) - pad
    grid_h = (H + pad) * rows - pad
    grid = Image.new('RGB', (grid_w, grid_h), (0, 0, 0))
    for row in range(rows):
        xb_i = unnormalize(xb[row:row+1]).squeeze(0).cpu()
        base = to_pil(xb_i)
        maps_i = maps[row].detach().cpu()
        mass = maps_i.flatten(1).sum(-1)
        _, idx = torch.sort(mass, descending=True)
        idx = idx[:k]
        maps_sel = maps_i[idx]
        maps_up = F.interpolate(maps_sel.unsqueeze(0), size=xb_i.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
        hm_flat = maps_up.view(maps_up.shape[0], -1)
        hm_min = hm_flat.min(dim=1, keepdim=True).values.view(-1, 1, 1)
        hm_max = hm_flat.max(dim=1, keepdim=True).values.view(-1, 1, 1)
        hm = (maps_up - hm_min) / (hm_max - hm_min + 1e-6)
        y = row * (H + pad)
        grid.paste(base, (0, y))
        for col in range(k):
            cmap = plt.get_cmap('magma')
            hm_i = hm[col].cpu().numpy()
            rgba = cmap(hm_i)
            heat = (rgba[..., :3] * 255).astype(np.uint8)
            over = Image.fromarray(heat)
            alpha = (hm_i * 200).clip(0, 255).astype(np.uint8)
            over.putalpha(Image.fromarray(alpha))
            comp = base.copy()
            comp.paste(over, (0, 0), over)
            x = (col + 1) * (W + pad)
            grid.paste(comp, (x, y))
    grid.save(out_path)

def write_loss_csv(hist: List[Tuple[int, float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "loss_history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step","loss"])
        w.writerows(hist)

def plot_loss_grid(histories: Dict[str, List[Tuple[int,float]]], out_dir: Path, step: int) -> None:
    items = [(name, hist) for name, hist in histories.items() if hist]
    if not items:
        return
    cols = min(3, len(items))
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
    axes_flat = list(axes.flat)
    for idx, (name, hist) in enumerate(items):
        ax = axes_flat[idx]
        steps, losses = zip(*hist)
        if all(l > 0 for l in losses):
            ax.semilogy(steps, losses)
        else:
            ax.plot(steps, losses)
        ax.set_title(name)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, which="both", linestyle="--", linewidth=0.4)
    for ax in axes_flat[len(items):]:
        ax.set_visible(False)
    fig.suptitle(f"Loss curves up to step {step}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"losses_step_{step:06d}.png", bbox_inches="tight")
    plt.close(fig)

# ------------------------------- Train -----------------------------------------
@dataclass
class Args:
    traj_dir: str = "data.image_distance.train_levels_1_2"
    out_dir: str = "out.diffvec_mario_maps"
    batch_size: int = 64
    lr: float = 3e-4
    epochs: int = 200
    steps_per_epoch: int = 120
    num_workers: int = 0
    device: Optional[str] = None
    max_trajs: Optional[int] = None
    save_every: int = 100
    image_size: Tuple[int,int] = (224,224)

    # model
    z_dim: int = 256
    heads: int = 8
    depth: int = 3
    r: int = 16
    use_localizer: bool = True

    # data sampling (continuous; buckets not passed to model)
    sample_prob: Dict[str,float] = field(default_factory=lambda: {"small":0.5, "mid":0.3, "large":0.2})
    small: Tuple[int,int] = (1,4)
    mid: Tuple[int,int] = (5,60)
    large: Tuple[int,int] = (61,400)
    cross_episode_prob: float = 0.10
    photometric_jitter: bool = True

    # losses
    lambda_reg: float = 1.0
    lambda_rank: float = 0.5
    lambda_decor: float = 0.1
    lambda_ent: float = 0.05
    lambda_inv: float = 0.10
    lambda_map_mass: float = 0.25
    lambda_map_D: float = 0.10
    lambda_map_eq: float = 0.10
    lambda_map_spread: float = 0.02
    rank_margin_eps: float = 0.02
    topk_vis: int = 6

def main():
    args = tyro.cli(Args)
    device = pick_device(args.device)
    print(f"[Device] {device}")

    ds = PairDataset(
        root_dir=args.traj_dir,
        image_size=args.image_size,
        sample_prob=args.sample_prob,
        small=args.small, mid=args.mid, large=args.large,
        cross_episode_prob=args.cross_episode_prob,
        photometric_jitter=args.photometric_jitter,
        max_trajs=args.max_trajs,
    )
    print(f"Dataset: ~{len(ds)} pairs across {len(ds.trajs)} trajectories.")

    sampler = RandomSampler(ds, replacement=False, num_samples=args.steps_per_epoch * args.batch_size)
    dl = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)

    model = DiffVecModel(z_dim=args.z_dim, heads=args.heads, depth=args.depth, r=args.r,
                         use_localizer=args.use_localizer).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path(args.out_dir) / f"run__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    total_hist:     List[Tuple[int,float]] = []
    reg_hist:       List[Tuple[int,float]] = []
    rank_hist:      List[Tuple[int,float]] = []
    decor_hist:     List[Tuple[int,float]] = []
    ent_hist:       List[Tuple[int,float]] = []
    map_mass_hist:  List[Tuple[int,float]] = []
    map_D_hist:     List[Tuple[int,float]] = []
    map_eq_hist:    List[Tuple[int,float]] = []
    map_total_hist: List[Tuple[int,float]] = []
    map_spread_hist: List[Tuple[int,float]] = []

    global_step = 0
    start_time = time.monotonic()

    for ep in range(1, args.epochs+1):
        model.train()
        for batch in dl:
            xa, xb, dt, _ = batch
            xa = xa.to(device); xb = xb.to(device); dt = dt.to(device)

            out = model(xa, xb, return_maps=args.use_localizer)
            D = out["D"]; m = out["m"]; proj = out["proj"]; log_sigma = out["log_sigma"]
            maps = out.get("maps", None)
            target = g_fn(dt)

            # Base losses
            L_reg   = gaussian_nll_like(D, target, log_sigma)
            L_rank  = gated_margin_ranking(D, xa, xb, eps=args.rank_margin_eps)
            L_decor = decorrelation_loss(proj)
            L_ent   = entropy_on_factors(m)

            # Invariance on factors (photometric jitter)
            if args.lambda_inv > 0.0 and random.random() < 0.5:
                jitter = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02)
                xa_j = torch.stack([default_transform(args.image_size)(jitter(T.ToPILImage()(unnormalize(xa[i]).cpu()))) for i in range(xa.shape[0])]).to(device)
                xb_j = torch.stack([default_transform(args.image_size)(jitter(T.ToPILImage()(unnormalize(xb[i]).cpu()))) for i in range(xb.shape[0])]).to(device)
                m_j = model(xa_j, xb_j)["m"]
                L_inv = F.l1_loss(m, m_j)
            else:
                L_inv = D.new_zeros(())

            # Map losses
            L_map_mass = L_map_D = L_map_eq = L_map_spread = D.new_zeros(())
            if args.use_localizer and maps is not None:
                L_map_mass = heatmap_mass_consistency(maps, m)
                L_map_D    = heatmap_global_consistency(maps, D)
                if args.lambda_map_eq > 0.0 and random.random() < 0.25:
                    L_map_eq = heatmap_shift_equivariance(xb, model, shift_px=2)
                if args.lambda_map_spread > 0.0:
                    L_map_spread = heatmap_spatial_entropy(maps)

            loss = (args.lambda_reg     * L_reg
                  + args.lambda_rank    * L_rank
                  + args.lambda_decor   * L_decor
                  + args.lambda_ent     * L_ent
                  + args.lambda_inv     * L_inv
                  + args.lambda_map_mass* L_map_mass
                  + args.lambda_map_D   * L_map_D
                  + args.lambda_map_eq  * L_map_eq
                  + args.lambda_map_spread * L_map_spread)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            global_step += 1
            total_hist.append((global_step, float(loss.item())))
            reg_hist.append((global_step, float(L_reg.item())))
            rank_hist.append((global_step, float(L_rank.item())))
            decor_hist.append((global_step, float(L_decor.item())))
            ent_hist.append((global_step, float(L_ent.item())))
            if args.use_localizer:
                map_mass_hist.append((global_step, float(L_map_mass.item())))
                map_D_hist.append((global_step, float(L_map_D.item())))
                map_eq_hist.append((global_step, float(L_map_eq.item())))
                map_spread_hist.append((global_step, float(L_map_spread.item())))
                map_total_hist.append((global_step, float((L_map_mass + L_map_D + L_map_eq + L_map_spread).item())))

            if global_step % 10 == 0:
                elapsed = time.monotonic() - start_time
                print(
                    f"[ep {ep:02d}] step {global_step:06d} | "
                    f"loss={loss.item():.4f} "
                    f"(reg={L_reg.item():.3f}, rank={L_rank.item():.3f}, decor={L_decor.item():.3f}, "
                    f"ent={L_ent.item():.3f}, inv={L_inv.item():.3f}, "
                    f"map_mass={L_map_mass.item():.3f}, map_D={L_map_D.item():.3f}, map_eq={L_map_eq.item():.3f}, "
                    f"map_spread={L_map_spread.item():.3f}) | "
                    f"Δt mean={dt.float().mean().item():.2f} | elapsed={elapsed/60:.2f} min"
                )

            # Save visualizations
            if global_step % args.save_every == 0:
                if args.use_localizer and maps is not None:
                    ms = summarize_maps(maps)
                    print(
                        f"[maps] μ={ms['mean']:.4f} σ={ms['std']:.4f} min={ms['min']:.4f} max={ms['max']:.4f} "
                        f"massμ={ms['mass_mean']:.2f} massσ={ms['mass_std']:.2f} spatialσ={ms['spatial_std']:.4f}"
                    )
                with torch.no_grad():
                    mi = m[0].detach().cpu()
                    plot_scree_and_bars(mi, out_dir / "samples" / f"scree_step_{global_step:06d}.png",
                                        title=f"step {global_step}", topk=min(args.topk_vis, args.r))
                    max_rows = min(4, xa.shape[0])
                    row_caps = [
                        f"Δt={int(dt[i].item())}, D={float(D[i].item()):.3f}"
                        for i in range(max_rows)
                    ]
                    save_pair(
                        xa,
                        xb,
                        out_dir / "samples" / f"pair_step_{global_step:06d}.png",
                        caption=f"step {global_step}",
                        row_captions=row_caps,
                    )
                    if args.use_localizer and ("maps" in out):
                        overlay_heatmaps_on_xb(xb, out["maps"], out_dir / "samples" / f"maps_step_{global_step:06d}.png",
                                               topk=args.topk_vis)

                loss_histories = {
                    "total": total_hist,
                    "reg": reg_hist,
                    "rank": rank_hist,
                    "decor": decor_hist,
                    "entropy": ent_hist,
                }
                if args.use_localizer:
                    loss_histories.update({
                        "map_total": map_total_hist,
                        "map_mass": map_mass_hist,
                        "map_D": map_D_hist,
                        "map_eq": map_eq_hist,
                        "map_spread": map_spread_hist,
                    })
                plot_loss_grid(loss_histories, out_dir, global_step)
                torch.save({"ep": ep, "step": global_step, "model": model.state_dict()},
                           out_dir / "last.pt")
                print(f"[ep {ep:02d}] saved viz/checkpoint at step {global_step:06d}")

        print(f"[ep {ep:02d}] done.")

    write_loss_csv(total_hist, out_dir)
    torch.save({"ep": ep, "step": global_step, "model": model.state_dict()}, out_dir / "final.pt")
    print("Training complete.")

if __name__ == "__main__":
    main()
