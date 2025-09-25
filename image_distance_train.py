#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a self-supervised "image distance" regressor on 240x224 frames.

- Input: folder of trajectories: traj_dumps/traj_<n>/states/state_<m>.png
- Augmentations on same base image create (img1, img2, gt_distance) pairs:
  * noise-only           -> distance = 0
  * translate (dx,dy)    -> distance = sqrt(dx^2 + dy^2)
  * crop+shift (+pad)    -> equivalent to translation above
  * scale around center  -> distance contributes ≈ |s-1| * scale_coeff
      where scale_coeff defaults to 0.5 * image_diagonal
- Slight Gaussian noise is added to (almost) all variants.
- Model: Siamese encoder (shared CNN) -> embeddings -> small head predicts scalar distance.
- Loss: MSE; also report MAE and % within tolerance.

Usage:
  python train_image_distance.py --data_root traj_dumps --epochs 10
"""
from __future__ import annotations
import math
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from functools import partial

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

H, W = 240, 224  # (height, width) of input frames

# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_state_images(data_root: Path) -> List[Path]:
    # Find .../traj_*/states/state_*.png
    img_paths = []
    for traj_dir in sorted(data_root.glob("traj_*")):
        state_dir = traj_dir / "states"
        if state_dir.is_dir():
            img_paths.extend(sorted(state_dir.glob("state_*.png")))
    return img_paths


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    # to [0,1] float32 CHW
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    arr = arr.astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t


def add_gaussian_noise(img: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return img
    noise = torch.randn_like(img) * sigma
    out = img + noise
    return torch.clamp(out, 0.0, 1.0)


def pad_with_noise(canvas: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
    # canvas is CHW target size (3,H,W). Fill with random noise; caller can paste content.
    if seed is not None:
        gen = torch.Generator(device=canvas.device).manual_seed(seed)
        canvas.copy_(torch.rand_like(canvas, generator=gen))
    else:
        canvas.copy_(torch.rand_like(canvas))
    return canvas


def affine_translate(img: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """
    Translate img by (dx,dy) pixels (x=right, y=down). Pad with noise.
    """
    c, h, w = img.shape
    base = torch.empty_like(img)
    pad_with_noise(base)
    # integer grid copy (simple, fast). For subpixel, we could grid_sample; here keep simple.
    ix = torch.arange(w, device=img.device).view(1, w).expand(h, w)
    iy = torch.arange(h, device=img.device).view(h, 1).expand(h, w)
    src_x = ix - int(round(dx))
    src_y = iy - int(round(dy))
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)

    # Get valid coordinates
    valid_y, valid_x = torch.where(valid)
    src_y_valid = src_y[valid]
    src_x_valid = src_x[valid]

    # Copy valid pixels
    base[:, valid_y, valid_x] = img[:, src_y_valid, src_x_valid]
    return base


def scale_around_center(img: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Uniform scale around image center, output size fixed (H,W), pad with noise.
    Implemented via PIL for decent quality + determinism, then back to tensor.
    """
    c, h, w = img.shape
    # to PIL
    arr = (img.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pil_scaled = pil.resize((new_w, new_h), resample=Image.BICUBIC)
    # paste on noise canvas centered
    canvas = Image.fromarray((np.random.rand(h, w, 3) * 255).astype(np.uint8))
    ox = (w - new_w) // 2
    oy = (h - new_h) // 2
    canvas.paste(pil_scaled, (ox, oy))
    out = pil_to_tensor(canvas)
    return out.to(img.device)


def approx_scale_to_pixel_distance(scale: float, h: int, w: int, coeff: float = 0.5) -> float:
    """
    Convert uniform scale to a pixel-equivalent distance.
    Heuristic: |scale-1| * coeff * image_diagonal
    coeff defaults to 0.5 (average displacement over the image).
    """
    diag = math.sqrt(h * h + w * w)
    return abs(scale - 1.0) * coeff * diag


# -------------------------
# Pair generator (self-supervised)
# -------------------------
class PairAugmentor:
    """
    Generates (img1, img2, gt_dist) from a single base image using:
      - noise-only (dist=0)
      - translate (dx,dy)
      - crop+shift (implemented as translate; cropping is implicit when we translate and pad)
      - scale +- up to 20%
    """
    def __init__(
        self,
        noise_sigma: float = 0.02,
        max_shift: int = 30,       # pixels
        max_scale_delta: float = 0.20,
        p_noise_only: float = 0.20,
        p_translate: float = 0.50,
        p_scale: float = 0.30,
        scale_coeff: float = 0.5,
    ):
        total = p_noise_only + p_translate + p_scale
        self.p_noise_only = p_noise_only / total
        self.p_translate  = p_translate  / total
        self.p_scale      = p_scale      / total

        self.noise_sigma = noise_sigma
        self.max_shift = max_shift
        self.max_scale_delta = max_scale_delta
        self.scale_coeff = scale_coeff

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float, Dict]:
        """
        Returns: img1, img2, gt_distance, info
        """
        # base noise on both views
        img1 = add_gaussian_noise(img, self.noise_sigma)

        r = random.random()
        if r < self.p_noise_only:
            # Same image + noise -> 0 distance
            img2 = add_gaussian_noise(img, self.noise_sigma)
            dist = 0.0
            info = {"mode": "noise_only"}
        elif r < self.p_noise_only + self.p_translate:
            # Translate (dx,dy)
            dx = random.randint(-self.max_shift, self.max_shift)
            dy = random.randint(-self.max_shift, self.max_shift)
            img2 = affine_translate(img1, dx=dx, dy=dy)
            img2 = add_gaussian_noise(img2, self.noise_sigma)
            dist = math.sqrt(dx * dx + dy * dy)
            info = {"mode": "translate", "dx": dx, "dy": dy}
        else:
            # Scale around center
            s = 1.0 + random.uniform(-self.max_scale_delta, self.max_scale_delta)
            img2 = scale_around_center(img1, s)
            img2 = add_gaussian_noise(img2, self.noise_sigma)
            dist = approx_scale_to_pixel_distance(s, H, W, coeff=self.scale_coeff)
            info = {"mode": "scale", "scale": s}
        return img1, img2, float(dist), info


# -------------------------
# Dataset
# -------------------------
class FramesDataset(Dataset):
    """
    Iterates over all frames (as singletons). Pairing is done on-the-fly by PairAugmentor.
    """
    def __init__(self, data_root: Path, split: str = "train", train_frac: float = 0.95, seed: int = 0):
        img_paths = list_state_images(data_root)
        if not img_paths:
            raise FileNotFoundError(f"No images found under {data_root} (expected traj_*/states/state_*.png).")

        rng = random.Random(seed)
        rng.shuffle(img_paths)
        n_train = int(round(len(img_paths) * train_frac))
        if split == "train":
            self.paths = img_paths[:n_train]
        else:
            self.paths = img_paths[n_train:]
        self.split = split

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        img = img.resize((W, H), resample=Image.BICUBIC)  # guard
        t = pil_to_tensor(img)
        return t


def collate_fn(batch_imgs, aug):
    """
    Collate function for DataLoader that applies augmentations.

    Args:
        batch_imgs: List[Tensor CxHxW]
        aug: PairAugmentor instance

    Returns:
        Tuple of (img1_batch, img2_batch, distances, infos)
    """
    imgs = torch.stack(batch_imgs, dim=0)
    img1s, img2s, dists = [], [], []
    infos = []
    for i in range(imgs.shape[0]):
        a, b, dist, info = aug(imgs[i])
        img1s.append(a); img2s.append(b); dists.append(dist); infos.append(info)
    return torch.stack(img1s, 0), torch.stack(img2s, 0), torch.tensor(dists, dtype=torch.float32), infos


# -------------------------
# Model
# -------------------------
class ConvEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        # Simple but decent CNN; you can swap for a stronger backbone easily.
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        h = h.flatten(1)
        z = self.fc(h)
        z = F.normalize(z, dim=1)  # normalized embedding
        return z


class DistanceRegressor(nn.Module):
    """
    Siamese: shared encoder -> combine -> MLP -> scalar distance (>=0).
    """
    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.encoder = ConvEncoder(out_dim=emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim * 3, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        feat = torch.cat([z1, z2, torch.abs(z1 - z2)], dim=1)
        y = self.head(feat).squeeze(1)
        return F.relu(y)  # clamp negatives to 0


# -------------------------
# Training
# -------------------------
def train(
    data_root: Path,
    out_dir: Path,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    log_every: int = 100,
    num_workers: int = 4,
    device_str: Optional[str] = None,
):
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # device
    if device_str:
        device = torch.device(device_str)
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    print(f"[Device] {device}")

    # data
    ds_train = FramesDataset(data_root, split="train", train_frac=0.95, seed=seed)
    ds_val   = FramesDataset(data_root, split="val",   train_frac=0.95, seed=seed)
    aug = PairAugmentor()
    print(f"[Data] train={len(ds_train)}  val={len(ds_val)}")

    collate = partial(collate_fn, aug=aug)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, collate_fn=collate, drop_last=True)
    dl_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True, collate_fn=collate, drop_last=False)

    # model/opt
    model = DistanceRegressor().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type in ["cuda"]))
    best_val = float("inf")

    global_step = 0
    for ep in range(1, epochs + 1):
        model.train()
        running = {"mse": 0.0, "mae": 0.0, "n": 0, "acc@5": 0, "acc@10": 0}
        for img1, img2, dist, _ in dl_train:
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            dist = dist.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type in ["cuda"])):
                pred = model(img1, img2)
                loss = F.mse_loss(pred, dist)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                mae = torch.mean(torch.abs(pred - dist)).item()
                mse = torch.mean((pred - dist)**2).item()
                acc5 = torch.mean((torch.abs(pred - dist) <= 5.0).float()).item()
                acc10 = torch.mean((torch.abs(pred - dist) <= 10.0).float()).item()
            running["mse"] += mse * dist.numel()
            running["mae"] += mae * dist.numel()
            running["acc@5"] += acc5 * dist.numel()
            running["acc@10"] += acc10 * dist.numel()
            running["n"] += dist.numel()

            if (global_step % log_every) == 0:
                print(f"[ep {ep:02d} | step {global_step:06d}] "
                      f"train MSE={mse:.3f} MAE={mae:.3f} acc@5={acc5*100:.1f}% acc@10={acc10*100:.1f}%")
            global_step += 1

        # epoch summary
        n = max(1, running["n"])
        print(f"[ep {ep:02d}] train: MSE={running['mse']/n:.3f} MAE={running['mae']/n:.3f} "
              f"acc@5={(running['acc@5']/n)*100:.1f}% acc@10={(running['acc@10']/n)*100:.1f}%")

        # validation
        model.eval()
        val_mse, val_mae, val_n = 0.0, 0.0, 0
        with torch.no_grad():
            for img1, img2, dist, _ in dl_val:
                img1 = img1.to(device, non_blocking=True)
                img2 = img2.to(device, non_blocking=True)
                dist = dist.to(device, non_blocking=True)
                pred = model(img1, img2)
                val_mse += torch.sum((pred - dist) ** 2).item()
                val_mae += torch.sum(torch.abs(pred - dist)).item()
                val_n += dist.numel()
        val_mse /= max(1, val_n)
        val_mae /= max(1, val_n)
        print(f"[ep {ep:02d}]   val: MSE={val_mse:.3f} MAE={val_mae:.3f}")

        # checkpoint
        ckpt = {
            "epoch": ep,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "val_mse": val_mse,
            "val_mae": val_mae,
            "args": {
                "data_root": str(data_root),
                "seed": seed,
            },
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_mse < best_val:
            best_val = val_mse
            torch.save(ckpt, out_dir / "best.pt")
            print(f"[ep {ep:02d}] saved best checkpoint (val MSE={best_val:.3f})")

    print("[done]")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True, help="Path to traj_dumps")
    ap.add_argument("--out_dir", type=Path, default=Path("runs/image_distance"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default=None, help="cpu|cuda|mps (auto if None)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_root=args.data_root,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        log_every=args.log_every,
        num_workers=args.num_workers,
        device_str=args.device,
    )