#!/usr/bin/env python3
"""
Train a model to predict (minimum) action-distance between two frames.

Design:
- Only appearance augments (no geometric): Gaussian/shot noise, global Gaussian blur,
  and spatially-varying focus blur. Optional mild color jitter.
- Supervision:
  * Zero pairs (same frame, different appearance) -> distance 0 (invariance).
  * Within-trajectory pairs -> use observed step gap as an UPPER BOUND via hinge loss.
  * Triplets (i<j<k) -> ranking/ordinal loss on *squared* distances to avoid zero-gradient collapse.
  * Bidirectional consistency on vectors + distance symmetry.
  * (Optional) Conditional Flow Matching (CFM) auxiliary velocity field (flag --use_cfm).
  * (Small) Lower-bound hinge to discourage trivial Δ=0 collapse on gapped pairs.

Debugging/Visualization:
- Periodic grid with latent vector drawn from center, scaled so 1 action step = 10 pixels.
- Bottom-aligned text annotation (gap, prediction).
- Rich training diagnostics: means/std, % tiny, grad norms, overestimation EMA.

Default output directory: out.action_distance
Device preference: MPS (Apple) > CUDA > CPU
"""

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# -------------------------
# Utility: reproducibility
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------------------
# Data loading
# -------------------------
def load_frame(path: Path) -> torch.Tensor:
    """Load an RGB frame (H=240, W=224) -> float tensor [3,H,W] in [0,1]."""
    img = Image.open(path).convert("RGB")
    if img.size != (224, 240):
        img = img.resize((224, 240), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return torch.from_numpy(arr)

def discover_trajectories(root: Path) -> List[List[Path]]:
    """
    Return list of trajectories; each is a list of frame paths sorted by state index.
    Expects: traj_dumps/traj_<n>/states/state_<m>.png
    """
    trajs = []
    for traj_dir in sorted(root.glob("traj_*")):
        state_dir = traj_dir / "states"
        if not state_dir.exists():
            continue
        frames = sorted(state_dir.glob("state_*.png"), key=lambda p: int(p.stem.split("_")[-1]))
        if len(frames) >= 2:
            trajs.append(frames)
    return trajs

# -------------------------
# Appearance augmentations
# -------------------------
def add_gaussian_noise(img: torch.Tensor, std_range=(0.0, 0.15)) -> torch.Tensor:
    std = random.uniform(*std_range)
    if std == 0.0:
        return img
    noise = torch.randn_like(img) * std
    out = torch.clamp(img + noise, 0.0, 1.0)
    return out

def add_shot_noise(img: torch.Tensor, scale_range=(0.0, 0.15)) -> torch.Tensor:
    # Poisson-like shot noise approximation
    scale = random.uniform(*scale_range)
    if scale == 0.0:
        return img
    lam = torch.clamp(img * 255.0, 0.0, 255.0)
    noisy = torch.poisson(lam) / 255.0
    out = torch.clamp((1 - scale) * img + scale * noisy, 0.0, 1.0)
    return out

def gaussian_blur_pil(pil_img: Image.Image, sigma_range=(0.0, 3.0)) -> Image.Image:
    sigma = random.uniform(*sigma_range)
    if sigma <= 0.0:
        return pil_img
    return pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))

def to_pil(img_t: torch.Tensor) -> Image.Image:
    arr = (img_t.clamp(0, 1).numpy() * 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))  # HWC
    return Image.fromarray(arr)

def to_tensor(pil: Image.Image) -> torch.Tensor:
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)

def focus_blur(img_t: torch.Tensor, sigma_range=(0.5, 4.0)) -> torch.Tensor:
    """
    Create a spatially-varying blur by blending a globally blurred image
    with the original using a radial mask centered at random (cx, cy).
    """
    H, W = img_t.shape[1], img_t.shape[2]
    pil = to_pil(img_t)
    blurred = gaussian_blur_pil(pil, sigma_range)
    blurred_t = to_tensor(blurred)

    # Random focus point and radius
    cx = random.uniform(0.0, W - 1.0)
    cy = random.uniform(0.0, H - 1.0)
    max_r = math.sqrt(W * W + H * H)
    focus_radius = random.uniform(0.15, 0.45) * max_r  # fraction of diag
    hard_center = random.random() < 0.5  # sharp center vs sharp periphery

    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    dx = (xx - cx)
    dy = (yy - cy)
    dist = torch.sqrt(dx * dx + dy * dy).float()
    # Smooth radial mask in [0,1]
    k = 3.0
    mask = torch.sigmoid(k * (focus_radius - dist))  # near center ~1, far ~0

    if not hard_center:
        # invert: sharp outside, blur center
        mask = 1.0 - mask

    mask = mask[None, ...]  # 1xHxW
    out = mask * img_t + (1 - mask) * blurred_t
    return out.clamp(0.0, 1.0)

@dataclass
class AugmentConfig:
    p_gauss_noise: float = 0.5
    p_shot_noise: float = 0.2
    p_global_blur: float = 0.5
    p_focus_blur: float = 0.5
    # sigma/noise ranges
    gauss_std_range: Tuple[float, float] = (0.0, 0.12)
    shot_scale_range: Tuple[float, float] = (0.0, 0.12)
    blur_sigma_range: Tuple[float, float] = (0.0, 3.0)
    focus_blur_sigma_range: Tuple[float, float] = (0.6, 4.0)
    mild_color_jitter: bool = True
    jitter_strength: float = 0.06

def apply_appearance_aug(img: torch.Tensor, cfg: AugmentConfig) -> torch.Tensor:
    x = img
    if random.random() < cfg.p_gauss_noise:
        x = add_gaussian_noise(x, cfg.gauss_std_range)
    if random.random() < cfg.p_shot_noise:
        x = add_shot_noise(x, cfg.shot_scale_range)
    if random.random() < cfg.p_global_blur:
        x = to_tensor(gaussian_blur_pil(to_pil(x), cfg.blur_sigma_range))
    if random.random() < cfg.p_focus_blur:
        x = focus_blur(x, cfg.focus_blur_sigma_range)
    if cfg.mild_color_jitter:
        # brightness/contrast jitter (very mild)
        b = 1.0 + random.uniform(-cfg.jitter_strength, cfg.jitter_strength)
        c = 1.0 + random.uniform(-cfg.jitter_strength, cfg.jitter_strength)
        mean = x.mean(dim=(1, 2), keepdim=True)
        x = torch.clamp((x - mean) * c + mean, 0, 1)
        x = torch.clamp(x * b, 0, 1)
    return x

# -------------------------
# Dataset & Sampler
# -------------------------
class TrajectorySet(Dataset):
    """
    Yields items for pair & triplet sampling:
      - For batch construction, we'll randomly pick trajectories and indices inside collate_fn.
    """
    def __init__(self, root: Path):
        self.trajs = discover_trajectories(root)
        self.index = []  # (traj_id, frame_idx)
        for tid, frames in enumerate(self.trajs):
            for i in range(len(frames)):
                self.index.append((tid, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        tid, i = self.index[idx]
        return tid, i  # we load images lazily

def load_image_from_index(ds: TrajectorySet, tid: int, i: int) -> torch.Tensor:
    path = ds.trajs[tid][i]
    return load_frame(path)

def sample_pairs_and_triplets(
    ds: TrajectorySet,
    batch_size: int,
    max_gap: Optional[int] = None,
) -> Dict[str, List]:
    """
    Assemble:
      - zero_pairs: (img, img') of same state with appearance aug only
      - pairs: (imgA, imgB, gap)
      - triplets: (img_i, img_j, img_k) for ranking
    """
    out = {"zero_pairs": [], "pairs": [], "triplets": []}
    # Sample trajectories
    for _ in range(batch_size):
        tid = random.randrange(len(ds.trajs))
        frames = ds.trajs[tid]
        n = len(frames)

        if n < 3:
            i = random.randrange(n - 1)
            j = random.randrange(i + 1, n)
            k = j
        else:
            i = random.randrange(n - 2)
            j = random.randrange(i + 1, n - 1)
            k = random.randrange(j + 1, n)

        if max_gap is not None:
            # resample j,k until within bound
            tries = 0
            while (j - i) > max_gap and tries < 10:
                j = random.randrange(i + 1, n)
                tries += 1
            tries = 0
            while (k - i) > max_gap and tries < 10:
                k = random.randrange(j + 1, n) if j + 1 < n else n - 1
                tries += 1

        # Load base images (no aug yet)
        img_i = load_image_from_index(ds, tid, i)
        img_j = load_image_from_index(ds, tid, j)
        img_k = load_image_from_index(ds, tid, k)

        out["pairs"].append((img_i, img_j, j - i))
        if k > j:
            out["triplets"].append((img_i, img_j, img_k, (j - i), (k - i)))

        # zero pair (same state i)
        img_i2 = img_i.clone()
        out["zero_pairs"].append(img_i2)

    return out

# -------------------------
# Model
# -------------------------
class ConvEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        # Lightweight ConvNet tuned for 240x224 inputs
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
        h = self.net(x).squeeze(-1).squeeze(-1)  # [B,256]
        z = self.proj(h)
        z = F.normalize(z, dim=-1)
        return z  # [B,D], unit norm

class VectorHead(nn.Module):
    """Predict a displacement vector Δ in latent space for A->B."""
    def __init__(self, dim_z=128, out_vec_dim=8, hidden=256):
        super().__init__()
        in_dim = dim_z * 3  # zA, zB, |zA-zB|
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_vec_dim),
        )

    def forward(self, zA, zB):
        x = torch.cat([zA, zB, (zA - zB).abs()], dim=-1)
        delta = self.mlp(x)
        return delta  # Δ_AB

class CFMHead(nn.Module):
    """Predict velocity field v(z_t, t; A, B)."""
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
        v = self.mlp(x)
        return v

class ActionDistanceModel(nn.Module):
    def __init__(self, dim_z=128, vec_dim=8, use_cfm=False):
        super().__init__()
        self.encoder = ConvEncoder(out_dim=dim_z)
        self.vec_head = VectorHead(dim_z=dim_z, out_vec_dim=vec_dim)
        # init last layer to avoid exact-zero field
        with torch.no_grad():
            last = self.vec_head.mlp[-1]
            if isinstance(last, nn.Linear):
                nn.init.normal_(last.weight, mean=0.0, std=1e-3)
                nn.init.constant_(last.bias, 0.0)
        self.use_cfm = use_cfm
        if use_cfm:
            self.cfm = CFMHead(dim_z=dim_z)

    def forward(self, imgA, imgB, for_cfm_t: Optional[torch.Tensor] = None):
        zA = self.encoder(imgA)
        zB = self.encoder(imgB)
        delta_AB = self.vec_head(zA, zB)
        outputs = {"zA": zA, "zB": zB, "delta_AB": delta_AB}
        if self.use_cfm and for_cfm_t is not None:
            t = for_cfm_t.view(-1, 1)
            z_t = (1 - t) * zA + t * zB
            v_hat = self.cfm(z_t, t, zA, zB)
            outputs["cfm_v"] = v_hat
            outputs["z_t"] = z_t
            outputs["t"] = t
        return outputs

# -------------------------
# Distances & Losses
# -------------------------
EPS = 1e-8

def distance_from_delta(delta: torch.Tensor) -> torch.Tensor:
    # magnitude in "action steps"
    return torch.sqrt(torch.clamp((delta * delta).sum(dim=-1), min=EPS))

def squared_distance_from_delta(delta: torch.Tensor) -> torch.Tensor:
    # squared magnitude; non-vanishing grads at zero for ranking
    return (delta * delta).sum(dim=-1)

def upper_bound_hinge(pred: torch.Tensor, gap: torch.Tensor) -> torch.Tensor:
    # penalize only overestimation: (max(0, pred - gap))^2
    return F.relu(pred - gap).pow(2.0)

def ranking_loss_sq(d2_ij: torch.Tensor, d2_ik: torch.Tensor, margin_steps: float = 1.0) -> torch.Tensor:
    # margin in *steps*, converted to squared margin
    m2 = margin_steps * margin_steps
    return F.relu(m2 + d2_ij - d2_ik)

def bidir_consistency(delta_AB: torch.Tensor, delta_BA: torch.Tensor) -> torch.Tensor:
    # Δ_AB ≈ -Δ_BA
    return F.smooth_l1_loss(delta_AB + delta_BA, torch.zeros_like(delta_AB))

def distance_symmetry(d_ab: torch.Tensor, d_ba: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(d_ab, d_ba)

def cfm_loss(v_hat: torch.Tensor, zA: torch.Tensor, zB: torch.Tensor) -> torch.Tensor:
    # target is constant vector (zB - zA)
    v_target = zB - zA
    return F.mse_loss(v_hat, v_target)

# -------------------------
# Debug visualization
# -------------------------
def draw_arrow_on_pil(pil_img: Image.Image, vec: np.ndarray, text: str) -> Image.Image:
    """
    Draw a straight line from the image center in the direction of the latent vector.
    Visualization scale: 1 action step (‖vec‖=1) -> 10 pixels.
    Text annotation is placed at the bottom of the image.
    """
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    cx, cy = W // 2, H // 2

    # Vector components (use first 2 dims if higher dimensional)
    vx = float(vec[0])
    vy = float(vec[1]) if len(vec) > 1 else 0.0
    norm = math.sqrt(vx * vx + vy * vy)

    # Scale: 1 action step -> 10 pixels
    pixels_per_step = 10.0
    length_px = pixels_per_step * norm

    # Unit direction (avoid div-by-zero)
    if norm < 1e-8:
        ux, uy = 0.0, 0.0
    else:
        ux, uy = vx / norm, vy / norm

    # Note: image y-axis grows down, so invert vy
    ex, ey = cx + ux * length_px, cy - uy * length_px

    # Draw line body only
    draw.line((cx, cy, ex, ey), width=2, fill=(0, 255, 0))

    # Bottom-left text (robust across Pillow versions)
    try:
        font = ImageFont.load_default()
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            text_w, text_h = font.getsize(text)
        tx, ty = 6, H - text_h - 6
        draw.text((tx, ty), text, fill=(255, 255, 255), font=font)
    except Exception:
        pass

    return img

def save_debug_grid(pairs, preds, deltas, out_path: Path):
    """
    Save a 2xN grid: top row A panels, bottom row B panels, with lines and overlay text.
    pairs: list of (imgA_t, imgB_t, gap)
    preds: list of predicted distances (A->B magnitude)
    deltas: list of delta vectors (A->B)
    """
    panels = []
    for (A, B, gap), d_hat, dvec in zip(pairs, preds, deltas):
        A_pil = to_pil(A.cpu())
        B_pil = to_pil(B.cpu())
        textA = f"gap={gap}  d̂(A→B)={d_hat:.2f}"
        textB = f"gap={gap}  d̂(B→A)≈{d_hat:.2f}"
        # Use first 2 dims for drawing
        vec2 = dvec[:2]
        A_p = draw_arrow_on_pil(A_pil, vec2, textA)
        B_p = draw_arrow_on_pil(B_pil, (-vec2), textB)
        panels.append((A_p, B_p))

    if not panels:
        return
    W, H = panels[0][0].size
    cols = len(panels)
    grid = Image.new("RGB", (cols * W, 2 * H), (0, 0, 0))
    for c, (A_p, B_p) in enumerate(panels):
        grid.paste(A_p, (c * W, 0))
        grid.paste(B_p, (c * W, H))
    grid.save(out_path)

# -------------------------
# CLI helpers
# -------------------------
def default_device_str() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def make_argparser():
    ap = argparse.ArgumentParser(description="Train action-distance model (appearance-only aug, ranking + upper-bound).")
    ap.add_argument("--data_root", type=str, required=True,
                    help="Root directory containing traj_*/states/state_*.png (required).")
    ap.add_argument("--out_dir", type=str, default="out.action_distance",
                    help="Directory for checkpoints and debug images (default: %(default)s)")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Mini-batch size for sampled pairs/triplets (default: %(default)s)")
    ap.add_argument("--epochs", type=int, default=5,
                    help="Number of training epochs (default: %(default)s)")
    ap.add_argument("--steps_per_epoch", type=int, default=500,
                    help="Optimization steps per epoch (default: %(default)s)")
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="AdamW learning rate (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: %(default)s)")
    ap.add_argument("--device", type=str, default=default_device_str(),
                    help="Compute device: mps/cuda/cpu (auto-prefers MPS, then CUDA) (default: %(default)s)")
    ap.add_argument("--dim_z", type=int, default=128,
                    help="Embedding dimension of encoder output (default: %(default)s)")
    ap.add_argument("--vec_dim", type=int, default=8,
                    help="Output displacement vector dimensionality (default: %(default)s)")
    ap.add_argument("--use_cfm", action="store_true",
                    help="Enable Conditional Flow Matching auxiliary loss (default: %(default)s)")
    ap.add_argument("--max_gap", type=int, default=None,
                    help="Max step gap when sampling pairs/triplets (None means unbounded) (default: %(default)s)")
    ap.add_argument("--debug_every", type=int, default=200,
                    help="Steps between saving a debug pair grid (default: %(default)s)")
    ap.add_argument("--save_every", type=int, default=1000,
                    help="Steps between saving a checkpoint (default: %(default)s)")
    ap.add_argument("--w_zero", type=float, default=1.0,
                    help="Weight for zero-pair invariance loss (default: %(default)s)")
    ap.add_argument("--w_upper", type=float, default=1.0,
                    help="Weight for upper-bound hinge loss (default: %(default)s)")
    ap.add_argument("--w_rank", type=float, default=1.0,
                    help="Weight for ranking/ordinal loss (default: %(default)s)")
    ap.add_argument("--w_bidir", type=float, default=0.5,
                    help="Weight for bidirectional vector consistency loss (default: %(default)s)")
    ap.add_argument("--w_sym", type=float, default=0.2,
                    help="Weight for distance symmetry loss (default: %(default)s)")
    ap.add_argument("--w_cfm", type=float, default=0.5,
                    help="Weight for CFM velocity loss (used when --use_cfm) (default: %(default)s)")
    ap.add_argument("--rank_margin_steps", type=float, default=1.0,
                    help="Ranking margin in action steps (uses squared margin internally) (default: %(default)s)")
    ap.add_argument("--w_lower", type=float, default=0.05,
                    help="Weight for anti-collapse lower-bound hinge (default: %(default)s)")
    ap.add_argument("--lower_margin_steps", type=float, default=0.1,
                    help="Lower-bound margin (steps) to discourage d̂=0 on gapped pairs (default: %(default)s)")
    return ap

# -------------------------
# Training loop
# -------------------------
def grad_norm(module: nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total) if total > 0 else 0.0

def main():
    args = make_argparser().parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    ds = TrajectorySet(Path(args.data_root))
    if len(ds.trajs) == 0:
        raise RuntimeError(f"No trajectories found under {args.data_root}")

    aug_cfg = AugmentConfig()

    model = ActionDistanceModel(dim_z=args.dim_z, vec_dim=args.vec_dim, use_cfm=args.use_cfm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    global_step = 0
    best_rank_acc = 0.0
    overest_rate_ema = None

    for epoch in range(args.epochs):
        model.train()
        for step in range(args.steps_per_epoch):
            batch = sample_pairs_and_triplets(ds, args.batch_size, args.max_gap)

            # Build tensors with appearance-only aug
            # Pairs
            imgsA, imgsB, gaps = [], [], []
            for A, B, g in batch["pairs"]:
                imgsA.append(apply_appearance_aug(A, aug_cfg))
                imgsB.append(apply_appearance_aug(B, aug_cfg))
                gaps.append(g)
            if len(imgsA) == 0:
                continue
            A_t = torch.stack(imgsA, 0).to(device)
            B_t = torch.stack(imgsB, 0).to(device)
            gap_t = torch.tensor(gaps, dtype=torch.float32, device=device)

            # Zero pairs (same state i vs i')
            zeroA = []
            zeroB = []
            for base in batch["zero_pairs"]:
                zeroA.append(apply_appearance_aug(base, aug_cfg))
                zeroB.append(apply_appearance_aug(base, aug_cfg))
            ZA = torch.stack(zeroA, 0).to(device)
            ZB = torch.stack(zeroB, 0).to(device)

            # Forward main pairs
            t_for_cfm = torch.rand(A_t.size(0), device=device) if args.use_cfm else None
            out_AB = model(A_t, B_t, t_for_cfm)
            out_BA = model(B_t, A_t, t_for_cfm)

            d_AB  = distance_from_delta(out_AB["delta_AB"])
            d_BA  = distance_from_delta(out_BA["delta_AB"])
            d2_AB = squared_distance_from_delta(out_AB["delta_AB"])

            # Zero-pair forward
            out_zero = model(ZA, ZB, None)
            d_zero = distance_from_delta(out_zero["delta_AB"])
            L_zero = F.smooth_l1_loss(d_zero, torch.zeros_like(d_zero))

            # Upper-bound hinge (no penalty if model predicts shorter than observed path)
            L_upper = upper_bound_hinge(d_AB, gap_t).mean()

            # Lower-bound anti-collapse (light touch; only meaningful when gap>0)
            if args.w_lower > 0:
                # mask only gapped pairs
                mask = (gap_t > 0).float()
                lb = torch.full_like(d_AB, args.lower_margin_steps)
                L_lower = (F.relu(lb - d_AB) * mask).sum() / (mask.sum() + 1e-8)
            else:
                L_lower = torch.tensor(0.0, device=device)

            # Ranking on squared distances
            if len(batch["triplets"]) > 0:
                triA, triJ, triK = [], [], []
                for i_img, j_img, k_img, _, _ in batch["triplets"]:
                    triA.append(apply_appearance_aug(i_img, aug_cfg))
                    triJ.append(apply_appearance_aug(j_img, aug_cfg))
                    triK.append(apply_appearance_aug(k_img, aug_cfg))
                triA = torch.stack(triA, 0).to(device)
                triJ = torch.stack(triJ, 0).to(device)
                triK = torch.stack(triK, 0).to(device)

                d2_ij = squared_distance_from_delta(model(triA, triJ)["delta_AB"])
                d2_ik = squared_distance_from_delta(model(triA, triK)["delta_AB"])
                L_rank = ranking_loss_sq(d2_ij, d2_ik, margin_steps=args.rank_margin_steps).mean()
                rank_acc = (d2_ij < d2_ik).float().mean().item()
            else:
                L_rank = torch.tensor(0.0, device=device)
                rank_acc = float("nan")

            L_bidir = bidir_consistency(out_AB["delta_AB"], out_BA["delta_AB"])
            L_sym   = distance_symmetry(d_AB, d_BA)
            L_cfm   = cfm_loss(out_AB["cfm_v"], out_AB["zA"], out_AB["zB"]) if args.use_cfm else torch.tensor(0.0, device=device)

            total_loss = (
                args.w_zero * L_zero
                + args.w_upper * L_upper
                + args.w_rank * L_rank
                + args.w_bidir * L_bidir
                + args.w_sym * L_sym
                + args.w_cfm * L_cfm
                + args.w_lower * L_lower
            )

            opt.zero_grad(set_to_none=True)
            total_loss.backward()

            # Gradient norms (before clipping/step for visibility)
            gn_enc  = grad_norm(model.encoder)
            gn_head = grad_norm(model.vec_head)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # Metrics
            with torch.no_grad():
                overest = (d_AB > gap_t).float().mean().item()
                overest_rate_ema = overest if overest_rate_ema is None else 0.95 * overest_rate_ema + 0.05 * overest

                mean_d   = d_AB.mean().item()
                std_d    = d_AB.std().item()
                pct_tiny = (d_AB < 1e-3).float().mean().item()
                mean_d0  = d_zero.mean().item()
                std_d0   = d_zero.std().item()

            if (global_step % 50) == 0:
                print(
                    f"ep {epoch} step {step} | loss {total_loss.item():.4f} | "
                    f"L0 {L_zero.item():.3f} Lu {L_upper.item():.3f} Lr {L_rank.item():.3f} "
                    f"Llo {L_lower.item():.3f} Lbi {L_bidir.item():.3f} Lsym {L_sym.item():.3f} "
                    f"{'Lcfm '+str(L_cfm.item()) if args.use_cfm else ''} | "
                    f"rank_acc {rank_acc:.3f} overest_ema {overest_rate_ema:.3f} | "
                    f"d̂ mean {mean_d:.3f}±{std_d:.3f} zero mean {mean_d0:.3f} pct<1e-3 {pct_tiny:.2f} | "
                    f"∥g_enc∥ {gn_enc:.3e} ∥g_head∥ {gn_head:.3e}"
                )

            # Debug visualization
            if (global_step % args.debug_every) == 0:
                # Save a small grid from first 4 pairs
                k = min(4, A_t.size(0))
                pairs = [(A_t[i].cpu(), B_t[i].cpu(), int(gap_t[i].item())) for i in range(k)]
                preds = [float(d_AB[i].item()) for i in range(k)]
                deltas = [out_AB["delta_AB"][i].detach().cpu().numpy() for i in range(k)]
                dbg_path = out_dir / f"debug_step_{global_step}.jpg"
                save_debug_grid(pairs, preds, deltas, dbg_path)

            # Save periodic checkpoints
            if (global_step % args.save_every) == 0 and global_step > 0:
                ckpt = {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "args": vars(args),
                    "step": global_step,
                    "epoch": epoch,
                }
                torch.save(ckpt, out_dir / f"ckpt_step_{global_step}.pt")

            global_step += 1

        # End epoch checkpoint by ranking accuracy (approx via last batch)
        if not math.isnan(rank_acc) and rank_acc > best_rank_acc:
            best_rank_acc = rank_acc
            torch.save(model.state_dict(), out_dir / "best_by_rankacc.pt")

    torch.save(model.state_dict(), out_dir / "final.pt")
    print("Training complete.")

if __name__ == "__main__":
    main()
