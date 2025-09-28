#!/usr/bin/env python3
"""
Train a model to predict the observed action distance (step gap) between two frames.

Simplifications:
- Objective: regress d̂(A,B) to the observed gap |j - i| with MSE.
- Appearance augmentations: noise only (Gaussian + shot). No blur/jitter/geom.
- Always use Conditional Flow Matching (CFM) auxiliary velocity loss.
- Add a decoder and reconstruction auxiliary loss so we can decode latent interpolations.
- Fail fast: no conservative exception handling.

Debug:
- Column layout: left column = all A images, right column = all B images (pairs read left->right).
- More frequent debug grids (default --debug_every 50).
- Latent interpolation visualization: interpolate zA->zB and decode each interpolated embedding to an image.
  Saved as a grid (rows=pairs, cols=interp steps).

Default output directory: out.action_distance
Device preference: MPS (Apple) > CUDA > CPU
"""

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

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
    """Load an RGB frame (H=240, W=224) -> float tensor [3,H,W] in [0,1]."""
    img = Image.open(path).convert("RGB")
    if img.size != (224, 240):
        img = img.resize((224, 240), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # CHW
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
    arr = (img_t.clamp(0, 1).numpy() * 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    return Image.fromarray(arr)

# -------------------------
# Noise-only augmentations
# -------------------------
def add_gaussian_noise(img: torch.Tensor, std_range=(0.0, 0.15)) -> torch.Tensor:
    std = random.uniform(*std_range)
    if std == 0.0:
        return img
    noise = torch.randn_like(img) * std
    return torch.clamp(img + noise, 0.0, 1.0)

def add_shot_noise(img: torch.Tensor, scale_range=(0.0, 0.15)) -> torch.Tensor:
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
    gauss_std_range: Tuple[float, float] = (0.0, 0.12)
    shot_scale_range: Tuple[float, float] = (0.0, 0.12)

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

def sample_pairs(ds: TrajectorySet, batch_size: int, max_gap: Optional[int] = None):
    pairs = []
    for _ in range(batch_size):
        tid = random.randrange(len(ds.trajs))
        frames = ds.trajs[tid]
        n = len(frames)
        i = random.randrange(0, n - 1)
        j = random.randrange(i + 1, n)
        if max_gap is not None:
            tries = 0
            while (j - i) > max_gap and tries < 10:
                j = random.randrange(i + 1, n)
                tries += 1
        img_i = load_image_from_index(ds, tid, i)
        img_j = load_image_from_index(ds, tid, j)
        pairs.append((img_i, img_j, j - i))
    return pairs

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
    """Lightweight decoder: z -> 256x15x14 -> upsample x2 x4 -> 240x224 -> tanh-ish via sigmoid."""
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
        x = torch.sigmoid(x)  # map to [0,1]
        return x

class ActionDistanceModel(nn.Module):
    def __init__(self, dim_z=128, vec_dim=8):
        super().__init__()
        self.encoder = ConvEncoder(out_dim=dim_z)
        self.vec_head = VectorHead(dim_z=dim_z, out_vec_dim=vec_dim)
        with torch.no_grad():
            last = self.vec_head.mlp[-1]
            if isinstance(last, nn.Linear):
                nn.init.normal_(last.weight, mean=0.0, std=1e-3)
                nn.init.constant_(last.bias, 0.0)
        self.cfm = CFMHead(dim_z=dim_z)
        self.decoder = ConvDecoder(dim_z=dim_z)

    def forward(self, imgA, imgB, t_for_cfm: torch.Tensor):
        zA = self.encoder(imgA)
        zB = self.encoder(imgB)
        delta_AB = self.vec_head(zA, zB)
        # CFM branch
        t = t_for_cfm.view(-1, 1)
        z_t = (1 - t) * zA + t * zB
        v_hat = self.cfm(z_t, t, zA, zB)
        return {"zA": zA, "zB": zB, "delta_AB": delta_AB, "cfm_v": v_hat}

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

# -------------------------
# Distances & Losses
# -------------------------
EPS = 1e-8

def distance_from_delta(delta: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp((delta * delta).sum(dim=-1), min=EPS))

def cfm_loss(v_hat: torch.Tensor, zA: torch.Tensor, zB: torch.Tensor) -> torch.Tensor:
    v_target = zB - zA  # constant target field
    return F.mse_loss(v_hat, v_target)

# -------------------------
# Debug viz
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

    if norm < 1e-8:
        ux, uy = 0.0, 0.0
    else:
        ux, uy = vx / norm, vy / norm

    ex, ey = cx + ux * length_px, cy - uy * length_px
    draw.line((cx, cy, ex, ey), width=2, fill=(0, 255, 0))

    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx, ty = 6, H - text_h - 6
    draw.text((tx, ty), text, fill=(255, 255, 255), font=font)
    return img

def save_debug_columns(pairs, preds, deltas, out_path: Path):
    """
    Two-column debug: left column all A panels, right column all B panels.
    Each row is a pair (read left->right).
    """
    if not pairs:
        return
    # Build panels with arrows/text
    A_panels, B_panels = [], []
    for (A, B, gap), d_hat, dvec in zip(pairs, preds, deltas):
        A_pil = to_pil(A.cpu())
        B_pil = to_pil(B.cpu())
        textA = f"gap={gap}  d̂(A→B)={d_hat:.2f}"
        textB = f"gap={gap}  d̂(B→A)≈{d_hat:.2f}"
        vec2 = dvec[:2]
        A_panels.append(draw_vec_panel(A_pil, vec2, textA))
        B_panels.append(draw_vec_panel(B_pil, (-vec2), textB))

    W, H = A_panels[0].size
    rows = len(A_panels)
    grid = Image.new("RGB", (2 * W, rows * H), (0, 0, 0))
    for r in range(rows):
        grid.paste(A_panels[r], (0, r * H))     # left column
        grid.paste(B_panels[r], (W, r * H))     # right column
    grid.save(out_path)

def save_latent_interpolations(model, pairs, out_path: Path, device, steps: int = 6):
    """
    For each pair (A, B), interpolate zA->zB with 'steps' samples (including endpoints),
    decode each latent to an image, and save as a grid: rows=pairs, cols=steps.
    """
    if not pairs:
        return
    model.eval()
    with torch.no_grad():
        # compute zA, zB
        zA_list, zB_list = [], []
        for (A, B, _) in pairs:
            A1 = A.unsqueeze(0).to(device)
            B1 = B.unsqueeze(0).to(device)
            zA = model.encoder(A1)
            zB = model.encoder(B1)
            zA_list.append(zA.squeeze(0))
            zB_list.append(zB.squeeze(0))

        # build grid images
        all_rows = []
        for zA, zB in zip(zA_list, zB_list):
            row_imgs = []
            for t in torch.linspace(0, 1, steps, device=device):
                z_t = (1 - t) * zA + t * zB
                img_t = model.decode(z_t.unsqueeze(0)).squeeze(0).cpu()
                row_imgs.append(to_pil(img_t))
            all_rows.append(row_imgs)

        # compose into a single image
        W, H = all_rows[0][0].size
        rows, cols = len(all_rows), steps
        canvas = Image.new("RGB", (cols * W, rows * H), (0, 0, 0))
        for r in range(rows):
            for c in range(cols):
                canvas.paste(all_rows[r][c], (c * W, r * H))
        canvas.save(out_path)

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
    ap = argparse.ArgumentParser(description="Train action-distance model by regressing observed step gaps (with CFM + decoder).")
    ap.add_argument("--data_root", type=str, required=True,
                    help="Root directory containing traj_*/states/state_*.png (required).")
    ap.add_argument("--out_dir", type=str, default="out.action_distance",
                    help="Directory for checkpoints and debug images (default: %(default)s)")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Mini-batch size (default: %(default)s)")
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
                    help="Embedding dimension (default: %(default)s)")
    ap.add_argument("--vec_dim", type=int, default=8,
                    help="Displacement vector dimensionality (default: %(default)s)")
    ap.add_argument("--max_gap", type=int, default=None,
                    help="Max step gap when sampling pairs (None means unbounded) (default: %(default)s)")
    ap.add_argument("--debug_every", type=int, default=100,
                    help="Steps between saving a two-column A/B debug grid (default: %(default)s)")
    ap.add_argument("--interp_every", type=int, default=100,
                    help="Steps between saving latent interpolation grids (default: %(default)s)")
    ap.add_argument("--interp_steps", type=int, default=6,
                    help="Number of columns (interpolation steps) per row in latent interp grids (default: %(default)s)")
    ap.add_argument("--w_rec", type=float, default=0.1,
                    help="Weight for image reconstruction (decoder) loss (default: %(default)s)")
    ap.add_argument("--w_cfm", type=float, default=0.5,
                    help="Weight for CFM auxiliary loss (default: %(default)s)")
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
    model = ActionDistanceModel(dim_z=args.dim_z, vec_dim=args.vec_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        for step in range(args.steps_per_epoch):
            pairs = sample_pairs(ds, args.batch_size, args.max_gap)

            imgsA, imgsB, gaps = [], [], []
            for A, B, g in pairs:
                imgsA.append(apply_noise_aug(A, aug_cfg))
                imgsB.append(apply_noise_aug(B, aug_cfg))
                gaps.append(g)
            A_t = torch.stack(imgsA, 0).to(device)
            B_t = torch.stack(imgsB, 0).to(device)
            gap_t = torch.tensor(gaps, dtype=torch.float32, device=device)

            t_for_cfm = torch.rand(A_t.size(0), device=device)
            out = model(A_t, B_t, t_for_cfm)

            # losses
            d_hat = distance_from_delta(out["delta_AB"])
            L_reg = F.mse_loss(d_hat, gap_t)                     # main objective
            L_cfm = cfm_loss(out["cfm_v"], out["zA"], out["zB"]) # auxiliary CFM

            # decoder recon loss on both A and B
            decA = model.decode(out["zA"])
            decB = model.decode(out["zB"])
            L_rec = (F.l1_loss(decA, A_t) + F.l1_loss(decB, B_t)) * 0.5

            loss = L_reg + args.w_cfm * L_cfm + args.w_rec * L_rec

            opt.zero_grad(set_to_none=True)
            loss.backward()

            gn_enc  = grad_norm(model.encoder)
            gn_head = grad_norm(model.vec_head)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if (global_step % 50) == 0:
                print(
                    f"ep {epoch} step {step} | loss {loss.item():.4f} | "
                    f"Lreg {L_reg.item():.3f} Lcfm {L_cfm.item():.3f} Lrec {L_rec.item():.3f} | "
                    f"d̂ mean {d_hat.mean().item():.3f}±{d_hat.std().item():.3f} | "
                    f"∥g_enc∥ {gn_enc:.3e} ∥g_head∥ {gn_head:.3e}"
                )

            # Two-column A/B debug grid (more frequent)
            if (global_step % args.debug_every) == 0:
                k = min(6, A_t.size(0))
                dbg_pairs = [(A_t[i].cpu(), B_t[i].cpu(), int(gap_t[i].item())) for i in range(k)]
                preds = [float(d_hat[i].item()) for i in range(k)]
                deltas = [out["delta_AB"][i].detach().cpu().numpy() for i in range(k)]
                dbg_path = out_dir / f"debug_cols_step_{global_step}.jpg"
                save_debug_columns(dbg_pairs, preds, deltas, dbg_path)

            # Latent interpolation grid (rows=pairs, cols=interp steps)
            if (global_step % args.interp_every) == 0:
                k = min(6, A_t.size(0))
                interp_pairs = [(A_t[i].cpu(), B_t[i].cpu(), int(gap_t[i].item())) for i in range(k)]
                interp_path = out_dir / f"interp_step_{global_step}.jpg"
                save_latent_interpolations(model, interp_pairs, interp_path, device, steps=args.interp_steps)

            # Periodic checkpoints
            if (global_step % 1000) == 0 and global_step > 0:
                ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "args": vars(args),
                        "step": global_step, "epoch": epoch}
                torch.save(ckpt, out_dir / f"ckpt_step_{global_step}.pt")

            global_step += 1

        # save per-epoch
        torch.save(model.state_dict(), out_dir / f"epoch_{epoch}_model.pt")

    torch.save(model.state_dict(), out_dir / "final.pt")
    print("Training complete.")

if __name__ == "__main__":
    main()
