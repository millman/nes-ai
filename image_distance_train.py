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
- Model: Siamese encoder (shared CNN) → embeddings → small MLP predicts a **motion vector** (dx, dy). The **vector magnitude** ‖(dx,dy)‖ is the image-distance signal.
- Loss: vector-component loss on (dx,dy) where defined (translate/noise), plus a magnitude loss so ‖(dx,dy)‖ matches the ground-truth pixel-equivalent distance (including scale). We report MAE/MSE on the magnitude and vector MAE on supervised subsets.

Usage:
  python train_image_distance.py --data_root traj_dumps --epochs 10
"""
from __future__ import annotations
import math
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from functools import partial

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
    Translate img by integer (dx, dy) pixels (x=right, y=down).
    Newly exposed regions are filled with random noise in [0,1].
    This uses torch.roll for speed and then overwrites the wrapped edges
    with fresh noise so there's no wraparound artifact.
    """
    c, h, w = img.shape
    ix = int(round(dx))
    iy = int(round(dy))

    # Roll the image; this wraps around, which we'll fix by replacing
    # the wrapped strips with noise so the output is a true translate+pad.
    out = torch.roll(img, shifts=(iy, ix), dims=(1, 2))

    # Vertical exposed bands
    if iy > 0:  # shifted down -> expose a band at the top
        out[:, :iy, :] = torch.rand((c, iy, w), device=img.device, dtype=img.dtype)
    elif iy < 0:  # shifted up -> expose a band at the bottom
        out[:, h+iy:, :] = torch.rand((c, -iy, w), device=img.device, dtype=img.dtype)

    # Horizontal exposed bands
    if ix > 0:  # shifted right -> expose a band on the left
        out[:, :, :ix] = torch.rand((c, h, ix), device=img.device, dtype=img.dtype)
    elif ix < 0:  # shifted left -> expose a band on the right
        out[:, :, w+ix:] = torch.rand((c, h, -ix), device=img.device, dtype=img.dtype)

    return out


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


def collate_with_aug(batch_imgs, aug):
    """Top-level, picklable collate that applies PairAugmentor on-the-fly."""
    imgs = torch.stack(batch_imgs, dim=0)
    img1s, img2s, dists, infos = [], [], [], []
    for i in range(imgs.shape[0]):
        a, b, dist, info = aug(imgs[i])
        img1s.append(a); img2s.append(b); dists.append(dist); infos.append(info)
    return (
        torch.stack(img1s, 0),
        torch.stack(img2s, 0),
        torch.tensor(dists, dtype=torch.float32),
        infos,
    )

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
    Siamese: shared encoder -> combine -> MLP -> motion vector (dx, dy).
    The image-distance is defined as the vector magnitude ‖(dx,dy)‖.
    """
    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.encoder = ConvEncoder(out_dim=emb_dim)
        self.head_vec = nn.Sequential(
            nn.Linear(emb_dim * 3, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Return motion vector (dx, dy)."""
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        feat = torch.cat([z1, z2, torch.abs(z1 - z2)], dim=1)
        v = self.head_vec(feat)
        return v


# -------------------------
# Debug utilities
# -------------------------

def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().clamp(0, 1)
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)



def save_debug_pairs(img1: torch.Tensor,
                      img2: torch.Tensor,
                      pred: torch.Tensor,
                      gt: torch.Tensor,
                      infos: list,
                      out_path: Path,
                      max_rows: int = 8,
                      *,
                      model: Optional[DistanceRegressor] = None,
                      device: Optional[torch.device] = None,
                      ) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = min(max_rows, img1.shape[0])
    canvas = Image.new("RGB", (W * 2, H * rows), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    cx, cy = W // 2, H // 2

    for r in range(rows):
        a = _tensor_to_pil(img1[r])
        b = _tensor_to_pil(img2[r])
        canvas.paste(a, (0, r * H))
        canvas.paste(b, (W, r * H))

        # text overlay
        txt = f"gt={gt[r]:.1f} pred={pred[r]:.1f} mode={infos[r].get('mode','?')}"
        if 'dx' in infos[r] and 'dy' in infos[r]:
            txt += f" dx={infos[r]['dx']} dy={infos[r]['dy']}"
        if 'scale' in infos[r]:
            txt += f" s={infos[r]['scale']:.3f}"
        draw.text((4, r * H + 4), txt, fill=(255, 255, 255), font=font)

        # optional: show model-estimated motion vectors A->B and B->A (direct model output)
        if model is not None and device is not None:
            try:
                # Predict vectors directly from the vector head
                # Predict vectors directly from the vector head
                v_ab = model(img1[r].unsqueeze(0).to(device))
                v_ba = model(img2[r].unsqueeze(0).to(device))
                dx_ab, dy_ab = float(v_ab[0,0].item()), float(v_ab[0,1].item())
                dx_ba, dy_ba = float(v_ba[0,0].item()), float(v_ba[0,1].item())
                d_ab = math.sqrt(dx_ab*dx_ab + dy_ab*dy_ab)
                d_ba = math.sqrt(dx_ba*dx_ba + dy_ba*dy_ba)

                # A panel (left): draw from center to center+(dx_ab, dy_ab)
                draw.line([(cx, r * H + cy), (cx + int(round(dx_ab)), r * H + cy + int(round(dy_ab)))], fill=(0, 255, 0), width=2)
                draw.ellipse([(cx + int(round(dx_ab)) - 2, r * H + cy + int(round(dy_ab)) - 2), (cx + int(round(dx_ab)) + 2, r * H + cy + int(round(dy_ab)) + 2)], outline=(0, 255, 0))
                draw.text((cx + 4, r * H + cy + 4), f"→ ({dx_ab:.1f},{dy_ab:.1f}) d={d_ab:.1f}", fill=(0, 255, 0), font=font)

                # B panel (right): draw from center to center+(dx_ba, dy_ba)
                draw.line([(W + cx, r * H + cy), (W + cx + int(round(dx_ba)), r * H + cy + int(round(dy_ba)))], fill=(255, 0, 0), width=2)
                draw.ellipse([(W + cx + int(round(dx_ba)) - 2, r * H + cy + int(round(dy_ba)) - 2), (W + cx + int(round(dx_ba)) + 2, r * H + cy + int(round(dy_ba)) + 2)], outline=(255, 0, 0))
                draw.text((W + cx + 4, r * H + cy + 4), f"→ ({dx_ba:.1f},{dy_ba:.1f}) d={d_ba:.1f}", fill=(255, 0, 0), font=font)
            except Exception as e:
                draw.text((4, r * H + 18), f"vec est failed: {e}", fill=(255, 80, 80), font=font)

    canvas.save(out_path)

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
    loss_type: str = "smoothl1",
    debug_every: int = 1000,
    debug_samples: int = 8,
    save_every_images: int = 0,
    lambda_comp: float = 1.0,
    lambda_mag: float = 1.0,
):
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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

    def collate(batch_imgs):
        # batch_imgs: List[Tensor CxHxW]
        imgs = torch.stack(batch_imgs, dim=0)
        img1s, img2s, dists = [], [], []
        infos = []
        for i in range(imgs.shape[0]):
            a, b, dist, info = aug(imgs[i])
            img1s.append(a); img2s.append(b); dists.append(dist); infos.append(info)
        return torch.stack(img1s, 0), torch.stack(img2s, 0), torch.tensor(dists, dtype=torch.float32), infos

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, collate_fn=partial(collate_with_aug, aug=aug), drop_last=True)
    dl_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True, collate_fn=partial(collate_with_aug, aug=aug), drop_last=False)

    # model/opt
    model = DistanceRegressor().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type in ["cuda"]))
    best_val = float("inf")

    processed_images = 0  # counts images seen by the network (2 per pair)
    next_ckpt_at = save_every_images if save_every_images and save_every_images > 0 else None

    global_step = 0
    for ep in range(1, epochs + 1):
        model.train()
        running = {"mse": 0.0, "mae": 0.0, "n": 0, "acc@5": 0, "acc@10": 0}
        for img1, img2, dist, infos in dl_train:
            # count images processed (each pair contributes two images)
            images_this_step = int(img1.shape[0]) * 2
            processed_images += images_this_step

            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            dist = dist.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type in ["cuda"])):
                pred_v = model(img1, img2)
                pred_mag = torch.linalg.norm(pred_v, dim=1)
                # choose loss
                # magnitude supervision for all modes (translate/scale/noise)
                if loss_type == "mse":
                    loss_mag = F.mse_loss(pred_mag, dist)
                else:
                    loss_mag = F.smooth_l1_loss(pred_mag, dist)

                # vector supervision: only on translate + noise_only (scale direction is ambiguous)
                B = dist.shape[0]
                gt_vec = torch.zeros((B, 2), device=device, dtype=pred_v.dtype)
                vec_mask = torch.zeros((B,), device=device, dtype=torch.bool)
                for i, info in enumerate(infos):
                    m = info.get("mode", "?")
                    if m == "translate":
                        gt_vec[i, 0] = float(info.get("dx", 0))
                        gt_vec[i, 1] = float(info.get("dy", 0))
                        vec_mask[i] = True
                    elif m == "noise_only":
                        vec_mask[i] = True
                if vec_mask.any():
                    loss_vec = F.smooth_l1_loss(pred_v[vec_mask], gt_vec[vec_mask])
                else:
                    loss_vec = torch.zeros((), device=device)

                # total loss: vector components (when defined) + magnitude always
                loss = lambda_comp * loss_vec + lambda_mag * loss_mag

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                mae = torch.mean(torch.abs(pred_mag - dist)).item()
                mse = torch.mean((pred_mag - dist)**2).item()
                acc5 = torch.mean((torch.abs(pred_mag - dist) <= 5.0).float()).item()
                acc10 = torch.mean((torch.abs(pred_mag - dist) <= 10.0).float()).item()
                # vector MAE on supervised subset
                if 'vec_mask' in locals() and vec_mask.any():
                    vmae = torch.mean(torch.linalg.norm(pred_v[vec_mask] - gt_vec[vec_mask], dim=1)).item()
                else:
                    vmae = float('nan')
            running["mse"] += mse * dist.numel()
            running["mae"] += mae * dist.numel()
            running["acc@5"] += acc5 * dist.numel()
            running["acc@10"] += acc10 * dist.numel()
            running["n"] += dist.numel()

            if (global_step % log_every) == 0:
                # per-mode diagnostics
                modes = [x.get("mode", "?") for x in infos]
                _counts = {}
                for m in modes:
                    _counts[m] = _counts.get(m, 0) + 1
                mode_str = " ".join([f"{k}:{v}" for k, v in _counts.items()])
                print(f"[ep {ep:02d} | step {global_step:06d}] train MSE={mse:.3f} MAE={mae:.3f} vMAE={vmae:.3f} acc@5={acc5*100:.1f}% acc@10={acc10*100:.1f}% | modes {mode_str}")
            # periodic debug image grid (independent of log_every)
            if (global_step % debug_every) == 0:
                try:
                    out_path = out_dir / "debug_pairs" / f"step_{global_step:06d}.png"
                    save_debug_pairs(img1.cpu(), img2.cpu(), pred_mag.detach().cpu(), dist.detach().cpu(), infos, out_path, max_rows=debug_samples, model=model, device=device)
                except Exception as e:
                    print(f"[debug] failed to save pair grid: {e}")
            # periodic checkpoint by images processed
            if next_ckpt_at is not None and processed_images >= next_ckpt_at:
                try:
                    ckpt = {
                        "epoch": ep,
                        "global_step": global_step,
                        "images_seen": processed_images,
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "val_mse": None,
                        "val_mae": None,
                        "args": {"data_root": str(data_root), "seed": seed},
                    }
                    out_path = ckpt_dir / f"imgs_\{next_ckpt_at:012d\}\.ckpt"
                    torch.save(ckpt, out_path)
                    print(f"[ckpt] saved {out_path} (images_seen={processed_images})")
                except Exception as e:
                    print(f"[ckpt] failed to save periodic checkpoint: {e}")
                finally:
                    next_ckpt_at += save_every_images

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
                pred_v = model(img1, img2)
                pred_mag = torch.linalg.norm(pred_v, dim=1)
                val_mse += torch.sum((pred_mag - dist) ** 2).item()
                val_mae += torch.sum(torch.abs(pred_mag - dist)).item()
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
        torch.save(ckpt, out_dir / "last.ckpt")
        if val_mse < best_val:
            best_val = val_mse
            torch.save(ckpt, out_dir / "best.ckpt")
            print(f"[ep {ep:02d}] saved best checkpoint (val MSE={best_val:.3f})")

    print("[done]")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True, help="Path to traj_dumps")
    ap.add_argument("--out_dir", type=Path, default=Path("out.image_distance"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default=None, help="cpu|cuda|mps (auto if None)")
    ap.add_argument("--loss", type=str, default="smoothl1", choices=["mse", "smoothl1"], help="regression loss")
    ap.add_argument("--debug_every", type=int, default=1000, help="save a grid of training pairs every N steps")
    ap.add_argument("--debug_samples", type=int, default=8, help="rows in the debug grid")
    ap.add_argument("--save_every_images", type=int, default=10000, help="save a checkpoint every N images seen (0 disables)")
    ap.add_argument("--lambda_comp", type=float, default=1.0, help="weight of vector component loss on (dx,dy) for translate/noise")
    ap.add_argument("--lambda_mag", type=float, default=1.0, help="weight of magnitude loss so ‖(dx,dy)‖ matches pixel distance")
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
        loss_type=args.loss,
        debug_every=args.debug_every,
        debug_samples=args.debug_samples,
        save_every_images=args.save_every_images,
        # loss weights
        lambda_comp=args.lambda_comp,
        lambda_mag=args.lambda_mag,
    )
