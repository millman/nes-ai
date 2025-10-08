#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4-to-1 next-frame prediction with U-Net decoder and MS-SSIM loss.
- Input: stack of 4 RGB frames (12xH×W)
- Output: next RGB frame (3xH×W)
- Loss: 1 - MS-SSIM only
- Saves sample grids, loss plots, and CSV history.

Run:
  python video_ms_ssim_recon.py --traj_dir /path/to/traj_dumps --out_dir out.ms_ssim_unet
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights


# --------------------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------------------
def pick_device(pref: Optional[str]) -> torch.device:
    if pref:
        dev = torch.device(pref)
        return dev
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
def default_transform() -> T.Compose:
    # Use ImageNet normalization (ResNet18 default) with resize+center-crop to 224
    return ResNet18_Weights.DEFAULT.transforms()

State = torch.Tensor  # (12,H,W) for 4 stacked RGB frames
Next = torch.Tensor   # (3,H,W) next RGB frame
class Mario4to1Dataset(Dataset):
    """
    Expects root_dir/:
      traj_*/
        states/*.png
        actions.npz   (ignored here; we are in the 'no actions' setting)
    We slide a 5-frame window: take frames i..i+3 as input, i+4 as next target.
    """
    def __init__(self, root_dir: str, transform: Optional[T.Compose]=None,
                 max_trajs: Optional[int]=None) -> None:
        self.transform = transform or default_transform()
        self.index: List[Tuple[List[Path], int]] = []
        traj_count = 0
        for traj_path in sorted(Path(root_dir).iterdir()):
            if not traj_path.is_dir():
                continue
            states_dir = traj_path / "states"
            if not states_dir.is_dir():
                continue
            files = sorted(states_dir.iterdir())
            if len(files) < 5:
                continue
            # slide window: 4 in, 1 out => need 5 consecutive frames
            for i in range(len(files)-4):
                self.index.append((files, i))
            traj_count += 1
            if max_trajs and traj_count >= max_trajs:
                break

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[State, Next]:
        files, offset = self.index[idx]
        inputs = []
        for i in range(4):
            with Image.open(files[offset+i]).convert("RGB") as img:
                inputs.append(self.transform(img))  # (3,H,W) normalized
        x = torch.cat(inputs, dim=0)               # (12,H,W)

        with Image.open(files[offset+4]).convert("RGB") as img:
            y = self.transform(img)                # (3,H,W) normalized
        return x, y


# --------------------------------------------------------------------------------------
# Model (ResNet18 encoder over 12ch + U-Net style decoder with skips from enc feature pyramid)
# --------------------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, groups: int=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(num_groups=groups if c_out % groups == 0 else 1, num_channels=c_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(num_groups=groups if c_out % groups == 0 else 1, num_channels=c_out),
            nn.SiLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, 2, 2, 0)
        self.fuse = ConvBlock(c_out + c_skip, c_out)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)

class UNetPredictor(nn.Module):
    def __init__(self, out_ch: int = 3) -> None:
        super().__init__()
        # ResNet18 backbone; swap first conv to accept 12 channels
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        backbone.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Encoder (feature pyramid like in predict_mario.py)
        self.enc1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2
        self.enc2 = nn.Sequential(backbone.maxpool, backbone.layer1)            # /4
        self.enc3 = backbone.layer2                                             # /8
        self.enc4 = backbone.layer3                                             # /16
        self.enc5 = backbone.layer4                                             # /32
        # Bottleneck to seed decoder
        self.seed = ConvBlock(512, 512)
        # Decoder with U-Net skips
        self.up4 = Up(512, 256, 256)   # fuse enc4
        self.up3 = Up(256, 128, 128)   # fuse enc3
        self.up2 = Up(128, 64,  64)    # fuse enc2
        self.up1 = Up(64,  64,  64)    # fuse enc1
        self.head = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):  # x: (B,12,H,W), normalized
        target_hw = x.shape[-2:]
        f1 = self.enc1(x)               # (B,64, H/2,  W/2)
        f2 = self.enc2(f1)              # (B,64, H/4,  W/4)
        f3 = self.enc3(f2)              # (B,128,H/8,  W/8)
        f4 = self.enc4(f3)              # (B,256,H/16, W/16)
        f5 = self.enc5(f4)              # (B,512,H/32, W/32)
        b  = self.seed(f5)
        d4 = self.up4(b, f4)
        d3 = self.up3(d4, f3)
        d2 = self.up2(d3, f2)
        d1 = self.up1(d2, f1)
        y  = self.head(d1)  # outputs stay in normalized space
        if y.shape[-2:] != target_hw:
            y = F.interpolate(y, size=target_hw, mode="bilinear", align_corners=False)
        return y


# --------------------------------------------------------------------------------------
# MS-SSIM (5 scales), adapted to take normalized inputs; we unnormalize before MS-SSIM.
# --------------------------------------------------------------------------------------
INV_MEAN = torch.tensor([0.485, 0.456, 0.406])  # ImageNet
INV_STD  = torch.tensor([0.229, 0.224, 0.225])

def unnormalize(img: torch.Tensor) -> torch.Tensor:
    # img: (B,3,H,W) normalized
    mean = INV_MEAN[None, :, None, None].to(img.device, img.dtype)
    std  = INV_STD [None, :, None, None].to(img.device, img.dtype)
    return (img * std + mean).clamp(0,1)

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

    pow1 = weights[:-1].unsqueeze(1)
    pow2 = weights[-1]
    ms_prod = torch.prod(mcs_tensor ** pow1, dim=0) * (mssim_tensor[-1] ** pow2)
    return ms_prod.mean()

def ms_ssim_loss(x_hat_norm: torch.Tensor, x_true_norm: torch.Tensor) -> torch.Tensor:
    # Unnormalize to [0,1] before MS-SSIM
    xh = unnormalize(x_hat_norm)
    xt = unnormalize(x_true_norm)
    return 1.0 - ms_ssim(xh, xt)


# --------------------------------------------------------------------------------------
# Utils: saving samples and loss history
# --------------------------------------------------------------------------------------
def save_samples_grid(x4: torch.Tensor, y_hat: torch.Tensor, y_true: torch.Tensor, out_dir: Path, step: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    to_pil = T.ToPILImage()
    # split inputs
    B, C, H, W = x4.shape
    frames = x4.view(B, 4, 3, H, W)  # (B,4,3,H,W)

    # unnormalize everything for visualization
    def unnorm3(t):
        if t.dim() == 3:
            mean = INV_MEAN[:, None, None].to(t.device, t.dtype)
            std  = INV_STD [:, None, None].to(t.device, t.dtype)
            return (t * std + mean).clamp(0, 1)
        if t.dim() == 4:
            mean = INV_MEAN[None, :, None, None].to(t.device, t.dtype)
            std  = INV_STD [None, :, None, None].to(t.device, t.dtype)
            return (t * std + mean).clamp(0, 1)
        raise ValueError(f"Expected 3D or 4D tensor, got {t.dim()}D tensor")

    for i in range(min(4, B)):
        row1 = [to_pil(unnorm3(frames[i,j])) for j in range(4)] + [to_pil(unnorm3(y_true[i]))]
        row2 = [Image.new('RGB', (row1[0].width, row1[0].height), (0,0,0))]*4 + [to_pil(unnorm3(y_hat[i]))]

        Ww, Hh = row1[0].size
        canvas = Image.new('RGB', (Ww*5, Hh*2))
        for k, im in enumerate(row1): canvas.paste(im, (k*Ww, 0))
        for k, im in enumerate(row2): canvas.paste(im, (k*Ww, Hh))
        canvas.save(out_dir / f"sample_step_{step:06d}_idx_{i}.png")

def write_loss_csv(hist: List[Tuple[int,float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "loss_history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step","loss"])
        w.writerows(hist)

def plot_loss(hist: List[Tuple[int,float]], out_dir: Path, step: int) -> None:
    if not hist: return
    steps, losses = zip(*hist)
    plt.figure()
    plt.semilogy(steps, losses)
    plt.xlabel("Step"); plt.ylabel("Loss (1 - MS-SSIM)")
    plt.title(f"Loss up to step {step}")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"loss_step_{step:06d}.png", bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------------------
@dataclass
class Args:
    traj_dir: str
    out_dir: str = "out.ms_ssim_unet"
    batch_size: int = 32
    lr: float = 1e-4
    epochs: int = 2
    steps_per_epoch: int = 100
    num_workers: int = 2
    device: Optional[str] = None
    max_trajs: Optional[int] = None
    save_every: int = 50

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="out.ms_ssim_unet")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--steps_per_epoch", type=int, default=100)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--max_trajs", type=int, default=None)
    ap.add_argument("--save_every", type=int, default=50)
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[Device] {device}")

    ds = Mario4to1Dataset(args.traj_dir, max_trajs=args.max_trajs)
    print(f"Dataset: {len(ds)} samples.")
    # random subset per epoch using RandomSampler (similar flavor to predict_mario.py)
    sampler = RandomSampler(ds, replacement=False, num_samples=args.steps_per_epoch * args.batch_size)
    dl = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)

    model = UNetPredictor().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir) / f"run__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    global_step = 0
    loss_hist: List[Tuple[int,float]] = []

    for ep in range(1, args.epochs+1):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device)            # (B,12,H,W) normalized
            yb = yb.to(device)            # (B,3,H,W)  normalized
            y_hat = model(xb)             # (B,3,H,W) still in normalized space
            # NOTE: model outputs normalized tensors; MS-SSIM will unnormalize before scoring.
            loss = ms_ssim_loss(y_hat, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            loss_hist.append((global_step, float(loss.item())))

            if global_step % 10 == 0:
                print(f"[ep {ep:02d}] step {global_step:06d} | loss={loss.item():.4f}")

            if global_step % args.save_every == 0:
                with torch.no_grad():
                    save_samples_grid(xb.cpu(), y_hat.cpu(), yb.cpu(), out_dir / "samples", global_step)
                plot_loss(loss_hist, out_dir, global_step)
                torch.save({"ep": ep, "step": global_step, "model": model.state_dict()},
                           out_dir / "last.pt")

        print(f"[ep {ep:02d}] done.")

    write_loss_csv(loss_hist, out_dir)
    torch.save({"ep": ep, "step": global_step, "model": model.state_dict()}, out_dir / "final.pt")
    print("Training complete.")

if __name__ == "__main__":
    main()
