#!/usr/bin/env python3
"""
Train a model to predict the observed action distance (step gap) between two frames.

Stabilized training:
- DistanceHead outputs ŷ = f(zA,zB) in log-space; target y = log1p(gap); loss = Smooth L1(ŷ,y).
- Report d̂ = expm1(ŷ) for metrics/visuals.
- Pairwise ranking loss (small hinge) to enforce monotonicity by gap.
- Light calibration to tie ||zB - zA|| to y; tiny Δ-norm calibration kept (0.02×).
- Noise-only augmentations (Gaussian + shot).
- Conditional Flow Matching (CFM) auxiliary loss (velocity matching).
- Lightweight decoder + small reconstruction loss (for latent interpolation visuals).
- Balanced gap sampler by default with max_gap 6.

Visualizations (saved to <out_dir>/vis/ at end of each epoch):
  1) Tiny push along z-direction (decoded frames + |·-A| heatmaps)
  2) Counterfactual line search along z-direction (best λ*, decoded, |·-B| heatmap)
  3) Orthogonal directions (around zA; decoded + diffs)
  4) PCA on Δ vectors (2D scatter colored by gap)
  5) Feature-space correspondence (cosine-sim heatmap A->B features)
  6) Occlusion sensitivity (masking heatmap for d̂ on A and B)
  7) Gradient attribution (|∂d̂/∂pixels| heatmaps on A and B)
  8) Decoder Jacobian probes (decode zA ± ε e_k; difference maps)
  9) Δ-length calibration (gap vs ||Δ|| bands)
 10) Neighborhood morphs along z-direction (decode neighbors and neighbors+βu)

Defaults:
- Output: out.action_distance (checkpoints/debug) and out.action_distance/vis (visuals)
- Device preference: MPS > CUDA > CPU
- Fail fast: minimal exception handling by design
"""

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------
# Repro
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------------------
# IO
# -------------------------
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
    arr = (img_t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    return Image.fromarray(arr)

# -------------------------
# Noise-only augmentations
# -------------------------
def add_gaussian_noise(img: torch.Tensor, std_range=(0.0, 0.10)) -> torch.Tensor:
    std = random.uniform(*std_range)
    if std == 0.0:
        return img
    noise = torch.randn_like(img) * std
    return torch.clamp(img + noise, 0.0, 1.0)

def add_shot_noise(img: torch.Tensor, scale_range=(0.0, 0.10)) -> torch.Tensor:
    scale = random.uniform(*scale_range)
    if scale == 0.0:
        return img
    lam = torch.clamp(img * 255.0, 0.0, 255.0)
    noisy = torch.poisson(lam) / 255.0
    return torch.clamp((1 - scale) * img + scale * noisy, 0.0, 1.0)

@dataclass
class AugmentConfig:
    p_gauss_noise: float = 0.6
    p_shot_noise: float = 0.3
    gauss_std_range: Tuple[float, float] = (0.0, 0.10)
    shot_scale_range: Tuple[float, float] = (0.0, 0.10)

def apply_noise_aug(img: torch.Tensor, cfg: AugmentConfig) -> torch.Tensor:
    x = img
    if random.random() < cfg.p_gauss_noise:
        x = add_gaussian_noise(x, cfg.gauss_std_range)
    if random.random() < cfg.p_shot_noise:
        x = add_shot_noise(x, cfg.shot_scale_range)
    return x

# -------------------------
# Dataset & sampling
# -------------------------
class TrajectorySet(Dataset):
    def __init__(self, root: Path):
        self.trajs = discover_trajectories(root)
        self.index = []
        for tid, frames in enumerate(self.trajs):
            for i in range(len(frames)):
                self.index.append((tid, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        tid, i = self.index[idx]
        return tid, i

def load_image_from_index(ds: TrajectorySet, tid: int, i: int) -> torch.Tensor:
    return load_frame(ds.trajs[tid][i])

def sample_pairs(
    ds: TrajectorySet,
    batch_size: int,
    max_gap: Optional[int] = None,
    balanced: bool = True,
    gap_bins: Tuple[int, ...] = (1,2,3,4,5,6,8)
):
    """
    Return list of tuples: (img_i, img_j, gap, tid, i, j)
    If balanced=True, try to sample gaps roughly uniformly across bins.
    """
    if not balanced:
        pairs = []
        trials = 0
        while len(pairs) < batch_size and trials < batch_size * 200:
            trials += 1
            tid = random.randrange(len(ds.trajs))
            frames = ds.trajs[tid]; n = len(frames)
            if n < 2:
                continue
            i = random.randrange(0, n - 1)
            j = random.randrange(i + 1, n)
            gap = j - i
            if max_gap is not None and gap > max_gap:
                continue
            img_i = load_image_from_index(ds, tid, i)
            img_j = load_image_from_index(ds, tid, j)
            pairs.append((img_i, img_j, gap, tid, i, j))
        return pairs[:batch_size]

    # Balanced across bins
    pairs = []
    bin_edges = list(gap_bins) + [10**9]
    targets_per_bin = max(1, batch_size // len(bin_edges))
    counts = [0] * len(bin_edges)
    trials = 0
    while sum(counts) < targets_per_bin * len(bin_edges) and trials < batch_size * 2000:
        trials += 1
        tid = random.randrange(len(ds.trajs))
        frames = ds.trajs[tid]; n = len(frames)
        if n < 2:
            continue
        i = random.randrange(0, n - 1)
        j = random.randrange(i + 1, n)
        gap = j - i
        if max_gap is not None and gap > max_gap:
            continue
        # find bin
        b = 0
        while gap > bin_edges[b]:
            b += 1
        if counts[b] >= targets_per_bin:
            continue
        img_i = load_image_from_index(ds, tid, i)
        img_j = load_image_from_index(ds, tid, j)
        pairs.append((img_i, img_j, gap, tid, i, j))
        counts[b] += 1

    # If underfilled, resample uniformly under constraints
    safety_trials = 0
    while len(pairs) < batch_size and safety_trials < batch_size * 200:
        safety_trials += 1
        tid = random.randrange(len(ds.trajs))
        frames = ds.trajs[tid]; n = len(frames)
        if n < 2:
            continue
        i = random.randrange(0, n - 1)
        j = random.randrange(i + 1, n)
        gap = j - i
        if max_gap is not None and gap > max_gap:
            continue
        img_i = load_image_from_index(ds, tid, i)
        img_j = load_image_from_index(ds, tid, j)
        pairs.append((img_i, img_j, gap, tid, i, j))

    return pairs[:batch_size]

# -------------------------
# Model with CFM + Decoder
# -------------------------
class ConvEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(inplace=True),   # 120x112
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),  # 60x56
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True), # 30x28
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(inplace=True),# 15x14
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
        in_dim = dim_z * 3  # zA, zB, |zA - zB|
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_vec_dim),
        )

    def forward(self, zA, zB):
        x = torch.cat([zA, zB, (zA - zB).abs()], dim=-1)
        return self.mlp(x)

class DistanceHead(nn.Module):
    def __init__(self, dim_z=128, hidden=256):
        super().__init__()
        in_dim = dim_z * 3
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1),
        )
    def forward(self, zA, zB):
        x = torch.cat([zA, zB, (zA - zB).abs()], dim=-1)
        return self.mlp(x).squeeze(-1)  # ŷ (log-space), unbounded

class CFMHead(nn.Module):
    def __init__(self, dim_z=128, hidden=256):
        super().__init__()
        in_dim = dim_z * 3 + 1  # z_t, zA, zB, t
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, dim_z),
        )

    def forward(self, z_t, t, zA, zB):
        x = torch.cat([z_t, zA, zB, t], dim=-1)
        return self.mlp(x)

class ConvDecoder(nn.Module):
    def __init__(self, dim_z=128):
        super().__init__()
        self.fc = nn.Linear(dim_z, 256 * 15 * 14)
        self.net = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 30x28
            nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 60x56
            nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 120x112
            nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 240x224
            nn.Conv2d(32, 16, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, 1, 1),
        )

    def forward(self, z):
        B = z.size(0)
        x = self.fc(z).view(B, 256, 15, 14)
        x = self.net(x)
        x = torch.sigmoid(x)
        return x

class ActionDistanceModel(nn.Module):
    def __init__(self, dim_z=128, vec_dim=8):
        super().__init__()
        self.encoder = ConvEncoder(out_dim=dim_z)
        self.vec_head = VectorHead(dim_z=dim_z, out_vec_dim=vec_dim)
        self.dist_head = DistanceHead(dim_z=dim_z)
        self.cfm = CFMHead(dim_z=dim_z)
        self.decoder = ConvDecoder(dim_z=dim_z)
        # light calibrations
        self.calib = nn.Linear(1, 1)   # for ||Δ||
        self.zcalib = nn.Linear(1, 1)  # for ||zB - zA|| -> y
        with torch.no_grad():
            last = self.vec_head.mlp[-1]
            if isinstance(last, nn.Linear):
                nn.init.normal_(last.weight, mean=0.0, std=1e-3)
                nn.init.constant_(last.bias, 0.0)
            self.calib.weight.fill_(1.0); self.calib.bias.zero_()
            self.zcalib.weight.fill_(1.0); self.zcalib.bias.zero_()

    def forward(self, imgA, imgB, t_for_cfm: torch.Tensor):
        zA = self.encoder(imgA)
        zB = self.encoder(imgB)
        delta_AB = self.vec_head(zA, zB)
        y_hat = self.dist_head(zA, zB)  # log-space scalar
        t = t_for_cfm.view(-1, 1)
        z_t = (1 - t) * zA + t * zB
        v_hat = self.cfm(z_t, t, zA, zB)
        return {"zA": zA, "zB": zB, "delta_AB": delta_AB, "dist_hat": y_hat, "cfm_v": v_hat}

    def decode(self, z):
        return self.decoder(z)

# -------------------------
# Distances & Losses
# -------------------------
EPS = 1e-8
def distance_from_delta(delta: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp((delta * delta).sum(dim=-1), min=EPS))

def cfm_loss(v_hat: torch.Tensor, zA: torch.Tensor, zB: torch.Tensor) -> torch.Tensor:
    v_target = zB - zA
    return F.mse_loss(v_hat, v_target)

# -------------------------
# Debug helpers (two-column A/B)
# -------------------------
def draw_vec_panel(pil_img: Image.Image, vec2: np.ndarray, text: str) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    cx, cy = W // 2, H // 2
    vx = float(vec2[0])
    vy = float(vec2[1]) if len(vec2) > 1 else 0.0
    norm = math.sqrt(vx * vx + vy * vy)
    pixels_per_step = 10.0
    length_px = pixels_per_step * norm
    ux, uy = (0.0, 0.0) if norm < 1e-8 else (vx / norm, vy / norm)
    ex, ey = cx + ux * length_px, cy - uy * length_px
    draw.line((cx, cy, ex, ey), width=2, fill=(0, 255, 0))
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_h = bbox[3] - bbox[1]
    tx, ty = 6, H - text_h - 6
    draw.text((tx, ty), text, fill=(255, 255, 255), font=font)
    return img

def save_debug_columns(pairs, preds, deltas, out_path: Path):
    if not pairs:
        return
    A_panels, B_panels = [], []
    for (A, B, gap, *_), d_hat, dvec in zip(pairs, preds, deltas):
        A_pil = to_pil(A)
        B_pil = to_pil(B)
        textA = f"gap={gap}  d̂(A→B)={d_hat:.2f}"
        textB = f"gap={gap}  d̂(B→A)≈{d_hat:.2f}"
        vec2 = dvec[:2]
        A_panels.append(draw_vec_panel(A_pil, vec2, textA))
        B_panels.append(draw_vec_panel(B_pil, (-vec2), textB))
    W, H = A_panels[0].size
    rows = len(A_panels)
    grid = Image.new("RGB", (2 * W, rows * H), (0, 0, 0))
    for r in range(rows):
        grid.paste(A_panels[r], (0, r * H))
        grid.paste(B_panels[r], (W, r * H))
    grid.save(out_path)

# -------------------------
# Visualization utilities (10 views)
# -------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def im_to_heat(arr: np.ndarray) -> Image.Image:
    plt.figure(figsize=(2.24, 2.40), dpi=100)
    plt.axis("off")
    plt.imshow(arr, cmap="magma")
    plt.tight_layout(pad=0.0)
    from io import BytesIO
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize((224, 240), Image.BILINEAR)

def stack_horiz(pils: List[Image.Image]) -> Image.Image:
    H = max(im.size[1] for im in pils)
    W = sum(im.size[0] for im in pils)
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    x = 0
    for im in pils:
        canvas.paste(im, (x, 0)); x += im.size[0]
    return canvas

def stack_vert(pils: List[Image.Image]) -> Image.Image:
    W = max(im.size[0] for im in pils)
    H = sum(im.size[1] for im in pils)
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    y = 0
    for im in pils:
        canvas.paste(im, (0, y)); y += im.size[1]
    return canvas

# 1) Tiny push (along z-direction)
def vis_tiny_push(model, device, pairs, out_path: Path, alphas=(0.1, 0.2, 0.3, 0.4)):
    rows = []
    with torch.no_grad():
        for (A, B, gap, *_ ) in pairs:
            A1 = A.unsqueeze(0).to(device)
            B1 = B.unsqueeze(0).to(device)
            out = model(A1, B1, torch.rand(1, device=device))
            zA = out["zA"].squeeze(0)
            zB = out["zB"].squeeze(0)
            u = zB - zA
            u = u / (torch.linalg.norm(u) + 1e-8)

            row_imgs = [to_pil(A)]
            for a in alphas:
                z_t = zA + a * u
                dec = model.decode(z_t.unsqueeze(0)).squeeze(0).cpu()
                dec_p = to_pil(dec)
                diff = torch.abs(dec - A).mean(0).numpy()
                diff_p = im_to_heat(diff)
                row_imgs.append(stack_vert([dec_p, diff_p]))
            rows.append(stack_horiz(row_imgs))
    stack_vert(rows).save(out_path)

# 2) Counterfactual line search (along z-direction)
def vis_line_search(model, device, pairs, out_path: Path, num_lambdas=16, max_lambda=6.0):
    rows = []
    lambdas = torch.linspace(0, max_lambda, num_lambdas)
    with torch.no_grad():
        for (A, B, gap, *_ ) in pairs:
            A1 = A.unsqueeze(0).to(device)
            B1 = B.unsqueeze(0).to(device)
            out = model(A1, B1, torch.rand(1, device=device))
            zA = out["zA"].squeeze(0)
            zB = out["zB"].squeeze(0)
            u = zB - zA
            u = u / (torch.linalg.norm(u) + 1e-8)

            best_loss = 1e9; best_dec = None
            for lam in lambdas:
                z_t = zA + lam.item() * u
                dec = model.decode(z_t.unsqueeze(0)).squeeze(0).cpu()
                loss = torch.mean(torch.abs(dec - B)).item()
                if loss < best_loss:
                    best_loss = loss; best_dec = dec
            rows.append(stack_horiz([to_pil(A), to_pil(B), to_pil(best_dec),
                                     im_to_heat(torch.abs(best_dec - B).mean(0).numpy())]))
    stack_vert(rows).save(out_path)

# 3) Orthogonal directions around zA
def gram_schmidt(u, basis: List[torch.Tensor]):
    v = u.clone()
    for b in basis:
        v -= (v @ b) * b
    n = torch.linalg.norm(v) + 1e-8
    return v / n

def vis_orthogonals(model, device, pairs, out_path: Path, step=0.5):
    rows = []
    with torch.no_grad():
        for (A, B, gap, *_ ) in pairs:
            A1 = A.unsqueeze(0).to(device)
            B1 = B.unsqueeze(0).to(device)
            out = model(A1, B1, torch.rand(1, device=device))
            zA = out["zA"].squeeze(0)
            zB = out["zB"].squeeze(0)

            e1 = zB - zA; e1 = e1 / (torch.linalg.norm(e1) + 1e-8)
            e2 = gram_schmidt(torch.randn_like(e1), [e1])
            e3 = gram_schmidt(torch.randn_like(e1), [e1, e2])

            dec_e1 = model.decode((zA + step * e1).unsqueeze(0)).squeeze(0).cpu()
            dec_e2 = model.decode((zA + step * e2).unsqueeze(0)).squeeze(0).cpu()
            dec_e3 = model.decode((zA + step * e3).unsqueeze(0)).squeeze(0).cpu()

            rows.append(stack_horiz([
                to_pil(A),
                stack_vert([to_pil(dec_e1), im_to_heat(torch.abs(dec_e1 - A).mean(0).numpy())]),
                stack_vert([to_pil(dec_e2), im_to_heat(torch.abs(dec_e2 - A).mean(0).numpy())]),
                stack_vert([to_pil(dec_e3), im_to_heat(torch.abs(dec_e3 - A).mean(0).numpy())]),
            ]))
    stack_vert(rows).save(out_path)

# 4) PCA on many Δ’s
def vis_pca_deltas(delta_cache: List[Tuple[np.ndarray, int]], out_path: Path):
    if len(delta_cache) < 10:
        return
    X = np.stack([d for d, _ in delta_cache], 0)
    gaps = np.array([g for _, g in delta_cache])
    X = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Z = X @ Vt[:2].T
    plt.figure(figsize=(5, 4))
    sc = plt.scatter(Z[:,0], Z[:,1], c=gaps, s=10, cmap="viridis")
    plt.colorbar(sc, label="gap")
    plt.title("PCA of Δ vectors")
    plt.tight_layout(); plt.savefig(out_path); plt.close()

# 5) Feature-space correspondence heatmap (A->B)
def extract_conv_feat(model: ActionDistanceModel, x: torch.Tensor):
    feats = []
    def hook(_, __, output):
        feats.append(output.detach())
    h = model.encoder.net[-2].register_forward_hook(hook)  # last conv layer feature
    _ = model.encoder(x)
    h.remove()
    return feats[0]  # [B,C,H',W']

def vis_feature_corr(model, device, pairs, out_path: Path):
    rows = []
    with torch.no_grad():
        for (A, B, gap, *_ ) in pairs:
            A1, B1 = A.unsqueeze(0).to(device), B.unsqueeze(0).to(device)
            fA = extract_conv_feat(model, A1)
            fB = extract_conv_feat(model, B1)
            fA_n = F.normalize(fA, dim=1); fB_n = F.normalize(fB, dim=1)
            C, Ha, Wa = fA_n.shape[1:]
            Hb, Wb = fB_n.shape[2], fB_n.shape[3]
            fA_flat = fA_n.view(1, C, Ha * Wa)
            fB_flat = fB_n.view(1, C, Hb * Wb)
            sim = torch.einsum("bcn,bcm->bnm", fA_flat, fB_flat).squeeze(0)  # [N,M]
            sim_max, _ = torch.max(sim, dim=1)
            sim_map = sim_max.view(Ha, Wa).cpu().numpy()
            rng = np.ptp(sim_map)  # NumPy 2.0 safe
            sim_map = np.clip((sim_map - sim_map.min()) / (rng + 1e-8), 0, 1)
            rows.append(stack_horiz([to_pil(A), to_pil(B), im_to_heat(sim_map)]))
    stack_vert(rows).save(out_path)

# 6) Occlusion sensitivity
def vis_occlusion(model, device, pairs, out_path: Path, patch=24, stride=12):
    rows = []
    for (A, B, gap, *_ ) in pairs:
        A1 = A.unsqueeze(0).to(device)
        B1 = B.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(A1, B1, torch.rand(1, device=device))
            y_hat = out["dist_hat"].item()
        H, W = A.shape[1], A.shape[2]
        A_occ = np.zeros((H, W), dtype=np.float32)
        for y in range(0, H - patch + 1, stride):
            for x in range(0, W - patch + 1, stride):
                Ao = A.clone(); Ao[:, y:y+patch, x:x+patch] = Ao.mean()
                Ao1 = Ao.unsqueeze(0).to(device)
                with torch.no_grad():
                    d = model(Ao1, B1, torch.rand(1, device=device))["dist_hat"].item()
                A_occ[y:y+patch, x:x+patch] = max(A_occ[y:y+patch, x:x+patch].max(), y_hat - d)
        B_occ = np.zeros((H, W), dtype=np.float32)
        for y in range(0, H - patch + 1, stride):
            for x in range(0, W - patch + 1, stride):
                Bo = B.clone(); Bo[:, y:y+patch, x:x+patch] = Bo.mean()
                Bo1 = Bo.unsqueeze(0).to(device)
                with torch.no_grad():
                    d = model(A1, Bo1, torch.rand(1, device=device))["dist_hat"].item()
                B_occ[y:y+patch, x:x+patch] = max(B_occ[y:y+patch, x:x+patch].max(), y_hat - d)
        rows.append(stack_horiz([to_pil(A), im_to_heat(A_occ), to_pil(B), im_to_heat(B_occ)]))
    stack_vert(rows).save(out_path)

# 7) Gradient attribution (on ŷ)
def vis_grad_attr(model, device, pairs, out_path: Path):
    rows = []
    for (A, B, gap, *_ ) in pairs:
        A1 = A.unsqueeze(0).to(device).clone().requires_grad_(True)
        B1 = B.unsqueeze(0).to(device).clone().requires_grad_(True)
        out = model(A1, B1, torch.rand(1, device=device))
        y_hat = out["dist_hat"].mean()
        y_hat.backward()
        gA = A1.grad.detach().abs().mean(1).squeeze(0).cpu().numpy()
        gB = B1.grad.detach().abs().mean(1).squeeze(0).cpu().numpy()
        rows.append(stack_horiz([to_pil(A), im_to_heat(gA), to_pil(B), im_to_heat(gB)]))
    stack_vert(rows).save(out_path)

# 8) Decoder Jacobian probes
def vis_jacobian_probes(model, device, pairs, out_path: Path, dims=4, eps=0.15):
    rows = []
    with torch.no_grad():
        for (A, B, gap, *_ ) in pairs:
            A1 = A.unsqueeze(0).to(device)
            zA = model.encoder(A1).squeeze(0)
            idx = torch.topk(torch.abs(zA), k=min(dims, zA.numel())).indices.cpu().numpy().tolist()
            tiles = [to_pil(A)]
            for k in idx:
                e = torch.zeros_like(zA); e[k] = 1.0
                dec_p = model.decode((zA + eps * e).unsqueeze(0)).squeeze(0).cpu()
                dec_m = model.decode((zA - eps * e).unsqueeze(0)).squeeze(0).cpu()
                diff = torch.abs(dec_p - dec_m).mean(0).numpy()
                tiles.append(stack_vert([to_pil(dec_p), to_pil(dec_m), im_to_heat(diff)]))
            rows.append(stack_horiz(tiles))
    stack_vert(rows).save(out_path)

# 9) Δ-length calibration (uses cached Δ)
def vis_length_calibration(delta_cache: List[Tuple[np.ndarray, int]], out_path: Path):
    if len(delta_cache) == 0:
        return
    dhat = np.array([np.linalg.norm(d) for d, _ in delta_cache])
    gaps = np.array([g for _, g in delta_cache])
    plt.figure(figsize=(6,4))
    plt.scatter(gaps, dhat, s=8, alpha=0.4)
    bins = np.unique(np.linspace(gaps.min(), gaps.max(), num=12, dtype=int))
    centers, means, stds = [], [], []
    for b in range(len(bins)-1):
        m = (gaps >= bins[b]) & (gaps < bins[b+1])
        if m.sum() < 5:
            continue
        centers.append(0.5*(bins[b]+bins[b+1]))
        means.append(dhat[m].mean())
        stds.append(dhat[m].std())
    if centers:
        centers = np.array(centers); means = np.array(means); stds = np.array(stds)
        plt.plot(centers, means, lw=2)
        plt.fill_between(centers, means-stds, means+stds, alpha=0.2)
    plt.xlabel("gap"); plt.ylabel("||Δ||")
    plt.title("Δ-length calibration")
    plt.tight_layout(); plt.savefig(out_path); plt.close()

# 10) Neighborhood morphs along z-direction
def vis_neighbor_morphs(model, device, ds: TrajectorySet, pairs, out_path: Path, K=3, beta=0.7, pool_samples=400):
    # pool
    all_paths = []
    for tid, frames in enumerate(ds.trajs):
        for i, p in enumerate(frames):
            all_paths.append(p)
            if len(all_paths) >= pool_samples: break
        if len(all_paths) >= pool_samples: break
    imgs = [load_frame(p) for p in all_paths]
    X = torch.stack(imgs, 0).to(device)
    with torch.no_grad():
        Z = model.encoder(X)  # [N,D]
    rows = []
    with torch.no_grad():
        for (A, B, gap, *_ ) in pairs:
            zA = model.encoder(A.unsqueeze(0).to(device)).squeeze(0)
            zB = model.encoder(B.unsqueeze(0).to(device)).squeeze(0)
            u = zB - zA
            u = u / (torch.linalg.norm(u) + 1e-8)
            sims = (Z @ zA)
            idx = torch.topk(sims, k=min(K, Z.size(0))).indices
            tiles = [to_pil(A)]
            for ix in idx.tolist():
                zn = Z[ix]
                dec_n = model.decode(zn.unsqueeze(0)).squeeze(0).cpu()
                dec_n_plus = model.decode((zn + beta * u).unsqueeze(0)).squeeze(0).cpu()
                tiles.append(stack_vert([to_pil(dec_n), to_pil(dec_n_plus),
                                         im_to_heat(torch.abs(dec_n_plus - dec_n).mean(0).numpy())]))
            rows.append(stack_horiz(tiles))
    stack_vert(rows).save(out_path)

# -------------------------
# CLI
# -------------------------
def default_device_str() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def make_argparser():
    ap = argparse.ArgumentParser(description="Train action-distance model by regressing log-steps with CFM + decoder (stabilized).")
    ap.add_argument("--data_root", type=str, required=True,
                    help="Root directory containing traj_*/states/state_*.png (required).")
    ap.add_argument("--out_dir", type=str, default="out.action_distance",
                    help="Directory for checkpoints, debug, and visuals (default: %(default)s)")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Mini-batch size (default: %(default)s)")
    ap.add_argument("--epochs", type=int, default=5,
                    help="Number of training epochs (default: %(default)s)")
    ap.add_argument("--steps_per_epoch", type=int, default=500,
                    help="Optimization steps per epoch (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: %(default)s)")
    ap.add_argument("--device", type=str, default=default_device_str(),
                    help="Compute device: mps/cuda/cpu (auto-prefers MPS, then CUDA) (default: %(default)s)")
    ap.add_argument("--dim_z", type=int, default=128,
                    help="Embedding dimension (default: %(default)s)")
    ap.add_argument("--vec_dim", type=int, default=8,
                    help="Displacement vector dimensionality (default: %(default)s)")
    ap.add_argument("--max_gap", type=int, default=6,
                    help="Max step gap when sampling pairs to stabilize early training (default: %(default)s)")
    ap.add_argument("--balanced_gaps", action="store_true", default=True,
                    help="Sample pairs roughly uniformly across gap bins (default: %(default)s)")
    ap.add_argument("--debug_every", type=int, default=50,
                    help="Steps between saving a two-column A/B debug grid (default: %(default)s)")
    ap.add_argument("--save_every", type=int, default=1000,
                    help="Steps between saving a checkpoint (default: %(default)s)")
    # Visualization
    ap.add_argument("--vis_every_epoch", type=int, default=1,
                    help="Epoch interval for producing visualization suite (default: %(default)s)")
    ap.add_argument("--vis_pairs", type=int, default=4,
                    help="Number of random pairs to visualize per view (default: %(default)s)")
    ap.add_argument("--pca_cache_pairs", type=int, default=2000,
                    help="How many Δ samples to cache across training for PCA/calibration (default: %(default)s)")
    return ap

# -------------------------
# Training
# -------------------------
def grad_norm(module: nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total) if total > 0 else 0.0

def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean(); b = b - b.mean()
    num = (a * b).mean()
    den = (a.std() + 1e-8) * (b.std() + 1e-8)
    return float(num / den)

def main():
    args = make_argparser().parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "vis"; ensure_dir(vis_dir)

    device = torch.device(args.device)
    ds = TrajectorySet(Path(args.data_root))
    if len(ds.trajs) == 0:
        raise RuntimeError(f"No trajectories found under {args.data_root}")

    aug_cfg = AugmentConfig()
    model = ActionDistanceModel(dim_z=args.dim_z, vec_dim=args.vec_dim).to(device)

    # Param groups: encoder lower LR, heads/decoder higher LR
    opt = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
            {"params": list(model.vec_head.parameters())
                    + list(model.dist_head.parameters())
                    + list(model.cfm.parameters())
                    + list(model.decoder.parameters())
                    + list(model.zcalib.parameters())
                    + list(model.calib.parameters()),
             "lr": 3e-4, "weight_decay": 1e-5},
        ]
    )

    global_step = 0
    delta_cache: List[Tuple[np.ndarray, int]] = []

    for epoch in range(args.epochs):
        model.train()
        for step in range(args.steps_per_epoch):
            pairs = sample_pairs(ds, args.batch_size, args.max_gap, balanced=args.balanced_gaps)
            imgsA, imgsB, gaps = [], [], []
            for A, B, g, *_ in pairs:
                imgsA.append(apply_noise_aug(A, aug_cfg))
                imgsB.append(apply_noise_aug(B, aug_cfg))
                gaps.append(g)
            A_t = torch.stack(imgsA, 0).to(device)
            B_t = torch.stack(imgsB, 0).to(device)
            gap_t = torch.tensor(gaps, dtype=torch.float32, device=device)

            t_for_cfm = torch.rand(A_t.size(0), device=device)
            out = model(A_t, B_t, t_for_cfm)

            # Targets in log-space
            y = torch.log1p(gap_t)          # [B]
            y_hat = out["dist_hat"]         # [B] (unbounded)

            # Main regression loss (Huber)
            L_reg = F.smooth_l1_loss(y_hat, y)

            # Pairwise ranking hinge to enforce monotonicity
            with torch.no_grad():
                num_pairs = max(1, A_t.size(0) // 2)
                idx_i = torch.randint(0, A_t.size(0), (num_pairs,), device=device)
                idx_j = torch.randint(0, A_t.size(0), (num_pairs,), device=device)
                sign = torch.sign((y[idx_i] - y[idx_j]).clamp(min=-1, max=1))
            margin = 0.05
            rank_term = ( (y_hat[idx_i] - y_hat[idx_j]) * sign )
            L_rank = F.relu(margin - rank_term).mean()

            # Calibrations
            # Δ-norm (very tiny weight)
            delta_norm = torch.linalg.norm(out["delta_AB"], dim=-1, keepdim=True)  # [B,1]
            y_from_delta = model.calib(delta_norm).squeeze(-1)
            L_cal = 0.02 * F.smooth_l1_loss(y_from_delta, y)
            # z-distance (main light calibration)
            zdist = torch.linalg.norm(out["zB"] - out["zA"], dim=-1, keepdim=True)  # [B,1]
            y_from_z = model.zcalib(zdist).squeeze(-1)
            L_zcal = 0.1 * F.smooth_l1_loss(y_from_z, y)

            # CFM & reconstruction
            L_cfm = cfm_loss(out["cfm_v"], out["zA"], out["zB"])
            decA = model.decode(out["zA"]); decB = model.decode(out["zB"])
            L_rec = 0.5 * (F.l1_loss(decA, A_t) + F.l1_loss(decB, B_t))

            loss = L_reg + 0.5 * L_cfm + 0.1 * L_rec + L_zcal + L_cal + 0.2 * L_rank

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn_enc  = grad_norm(model.encoder)
            gn_heads = (grad_norm(model.vec_head) + grad_norm(model.dist_head) +
                        grad_norm(model.cfm) + grad_norm(model.decoder) +
                        grad_norm(model.zcalib) + grad_norm(model.calib))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

            # Metrics (report in steps using d̂ = expm1(ŷ))
            with torch.no_grad():
                d_hat_steps = torch.expm1(y_hat.detach())
                dn = torch.linalg.norm(out["delta_AB"], dim=-1)
                mae = float(torch.abs(d_hat_steps - gap_t).mean())
                corr_scalar = pearson_corr(d_hat_steps, gap_t)
                ycorr = pearson_corr(y_hat.detach(), y)
                corr_deln   = pearson_corr(dn.detach(), gap_t)
                gap_mean = float(gap_t.mean()); gap_std = float(gap_t.std())
                gmin, gmax = int(gap_t.min().item()), int(gap_t.max().item())
                d_mean = float(d_hat_steps.mean()); d_std = float(d_hat_steps.std())

            if (global_step % 50) == 0:
                print(
                    f"ep {epoch} step {step} | loss {loss.item():.4f} | "
                    f"Lreg {L_reg.item():.3f} Lrank {L_rank.item():.3f} Lcfm {L_cfm.item():.3f} Lrec {L_rec.item():.3f} "
                    f"Lzcal {L_zcal.item():.3f} Lcal {L_cal.item():.3f} | "
                    f"gap {gap_mean:.2f}±{gap_std:.2f} [{gmin},{gmax}] | d̂ {d_mean:.2f}±{d_std:.2f} | "
                    f"MAE {mae:.3f} | corr(ŷ,log1p(gap)) {ycorr:.3f} | corr(d̂,gap) {corr_scalar:.3f} | corr(||Δ||,gap) {corr_deln:.3f} | "
                    f"∥g_enc∥ {gn_enc:.3e} ∥g_heads∥ {gn_heads:.3e}"
                )

            # Two-column debug grid
            if (global_step % args.debug_every) == 0:
                k = min(6, A_t.size(0))
                dbg_pairs = [(A_t[i].detach().cpu(), B_t[i].detach().cpu(), int(gap_t[i].item()), None, None, None) for i in range(k)]
                preds = [float(d_hat_steps[i].item()) for i in range(k)]
                deltas = [out["delta_AB"][i].detach().cpu().numpy() for i in range(k)]
                save_debug_columns(dbg_pairs, preds, deltas, out_dir / f"debug_cols_step_{global_step}.jpg")

            # Cache deltas for PCA & calibration
            for i in range(A_t.size(0)):
                if len(delta_cache) >= args.pca_cache_pairs:
                    break
                delta_cache.append((out["delta_AB"][i].detach().cpu().numpy(), int(gap_t[i].item())))

            # Periodic checkpoints
            if (global_step % args.save_every) == 0 and global_step > 0:
                ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "args": vars(args),
                        "step": global_step, "epoch": epoch}
                torch.save(ckpt, out_dir / f"ckpt_step_{global_step}.pt")

            global_step += 1

        # --------- End-of-epoch visualization suite ---------
        if ((epoch + 1) % args.vis_every_epoch) == 0:
            model.eval()
            base_pairs = sample_pairs(ds, batch_size=args.vis_pairs, max_gap=args.max_gap, balanced=True)
            base_pairs = [(load_image_from_index(ds, tid, i).cpu(),
                           load_image_from_index(ds, tid, j).cpu(),
                           gap, tid, i, j)
                          for (_, _, gap, tid, i, j) in base_pairs]

            vis_dir_epoch = vis_dir
            vis_tiny_push(model, device, base_pairs, vis_dir_epoch / f"ep{epoch:02d}_1_tiny_push.jpg")
            vis_line_search(model, device, base_pairs, vis_dir_epoch / f"ep{epoch:02d}_2_line_search.jpg")
            vis_orthogonals(model, device, base_pairs, vis_dir_epoch / f"ep{epoch:02d}_3_orthogonals.jpg")
            vis_pca_deltas(delta_cache, vis_dir_epoch / f"ep{epoch:02d}_4_pca_deltas.png")
            vis_feature_corr(model, device, base_pairs, vis_dir_epoch / f"ep{epoch:02d}_5_feat_corr.jpg")
            vis_occlusion(model, device, base_pairs, vis_dir_epoch / f"ep{epoch:02d}_6_occlusion.jpg")
            vis_grad_attr(model, device, base_pairs, vis_dir_epoch / f"ep{epoch:02d}_7_grad_attr.jpg")
            vis_jacobian_probes(model, device, base_pairs, vis_dir_epoch / f"ep{epoch:02d}_8_jacobian.jpg")
            vis_length_calibration(delta_cache, vis_dir_epoch / f"ep{epoch:02d}_9_length_calib.png")
            vis_neighbor_morphs(model, device, ds, base_pairs, vis_dir_epoch / f"ep{epoch:02d}_10_neighbor_morphs.jpg")

        # save per-epoch model
        torch.save(model.state_dict(), out_dir / f"epoch_{epoch}_model.pt")

    # final save
    torch.save(model.state_dict(), out_dir / "final.pt")
    print("Training complete.")

if __name__ == "__main__":
    main()
