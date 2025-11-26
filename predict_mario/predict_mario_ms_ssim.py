#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4-to-1 next-frame prediction with U-Net decoder, MS-SSIM + L1 loss, and
optional multi-step scheduled sampling.
- Input: stack of 4 RGB frames (12×H×W)
- Output: one or more future RGB frames (rollout×3×H×W)
- Loss: weighted (1 - MS-SSIM) + L1, averaged across rollout steps
- Extras: mixed teacher forcing and auto-regressive rollouts during training

Run:
  python predict_mario_ms_ssim.py --traj_dir /path/to/traj_dumps --out_dir out.predict_mario_ms_ssim
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import time
from pathlib import Path
from typing import List, Optional, Tuple

import tyro
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision.models import resnet18, ResNet18_Weights

from trajectory_utils import list_state_frames, list_traj_dirs
from utils.device_utils import pick_device

# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
def default_transform() -> T.Compose:
    # Use ImageNet normalization (ResNet18 default) with resize+center-crop to 224
    return ResNet18_Weights.DEFAULT.transforms()

State = torch.Tensor  # (12,H,W) for 4 stacked RGB frames
NextSeq = torch.Tensor   # (K,3,H,W) next RGB frame sequence
class Mario4to1Dataset(Dataset):
    """
    Expects root_dir/:
      traj_*/
        states/*.png
        actions.npz   (ignored here; we are in the 'no actions' setting)
    We slide a (4+rollout)-frame window: frames i..i+3 seed the model,
    and the next `rollout` frames provide supervision targets.
    """
    def __init__(self, root_dir: str, transform: Optional[T.Compose]=None,
                 max_trajs: Optional[int]=None, rollout: int = 1) -> None:
        self.transform = transform or default_transform()
        if rollout < 1:
            raise ValueError("rollout must be >= 1")
        self.rollout = rollout
        self.index: List[Tuple[List[Path], int]] = []
        traj_count = 0
        for traj_path in list_traj_dirs(Path(root_dir)):
            if not traj_path.is_dir():
                continue
            states_dir = traj_path / "states"
            if not states_dir.is_dir():
                continue
            files = list_state_frames(states_dir)
            needed = 4 + self.rollout
            if len(files) < needed:
                continue
            # slide window: 4 in, rollout out => need 4+rollout consecutive frames
            for i in range(len(files)-needed+1):
                self.index.append((files, i))
            traj_count += 1
            if max_trajs and traj_count >= max_trajs:
                break

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[State, NextSeq]:
        files, offset = self.index[idx]
        inputs = []
        for i in range(4):
            with Image.open(files[offset+i]).convert("RGB") as img:
                inputs.append(self.transform(img))  # (3,H,W) normalized
        x = torch.cat(inputs, dim=0)               # (12,H,W)

        targets = []
        for j in range(self.rollout):
            with Image.open(files[offset+4+j]).convert("RGB") as img:
                targets.append(self.transform(img))
        y = torch.stack(targets, dim=0)            # (rollout,3,H,W)
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

    # clamp to keep values in a numerically safe range before fractional powers
    eps = torch.finfo(mssim_tensor.dtype).eps
    mssim_tensor = mssim_tensor.clamp(min=eps, max=1.0)
    mcs_tensor = mcs_tensor.clamp(min=eps, max=1.0)

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
def save_samples_grid(context4: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor,
                      out_dir: Path, step: int) -> None:
    """Visualize context frames alongside multi-step targets and predictions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    to_pil = T.ToPILImage()

    if context4.dim() != 5:
        raise ValueError(f"Expected context4 as (B,4,3,H,W); got shape {context4.shape}")

    B, ctx_len, _, H, W = context4.shape
    if preds.dim() != 5 or targets.dim() != 5:
        raise ValueError("preds/targets must have shape (B,S,3,H,W)")

    steps = preds.shape[1]
    if targets.shape[1] != steps:
        raise ValueError("preds and targets must share rollout length")

    # unnormalize helper handles 3D (C,H,W) tensors
    def unnorm3(t: torch.Tensor) -> torch.Tensor:
        if t.dim() != 3:
            raise ValueError("Expected 3D tensor for frame visualization")
        mean = INV_MEAN[:, None, None].to(t.device, t.dtype)
        std  = INV_STD [:, None, None].to(t.device, t.dtype)
        return (t * std + mean).clamp(0, 1)

    max_rows = min(4, B)
    for i in range(max_rows):
        ctx_imgs = [to_pil(unnorm3(context4[i, j])) for j in range(ctx_len)]
        tgt_imgs = [to_pil(unnorm3(targets[i, s])) for s in range(steps)]
        pred_imgs = [to_pil(unnorm3(preds[i, s])) for s in range(steps)]

        # Align predictions beneath targets; show context along the left columns
        blank = Image.new('RGB', ctx_imgs[0].size, (0, 0, 0))
        top_row = ctx_imgs + tgt_imgs
        bottom_row = [blank] * ctx_len + pred_imgs

        tile_w, tile_h = ctx_imgs[0].size
        cols = ctx_len + steps
        canvas = Image.new('RGB', (tile_w * cols, tile_h * 2))
        for k, im in enumerate(top_row):
            canvas.paste(im, (k * tile_w, 0))
        for k, im in enumerate(bottom_row):
            canvas.paste(im, (k * tile_w, tile_h))

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
    traj_dir: str = "data.image_distance.train_levels_1_2"
    out_dir: str = "out.predict_mario_ms_ssim"
    batch_size: int = 32
    lr: float = 1e-4
    epochs: int = 1000
    steps_per_epoch: int = 100
    num_workers: int = 0
    device: Optional[str] = None
    max_trajs: Optional[int] = None
    save_every: int = 50
    ms_weight: float = 1.0
    l1_weight: float = 0.1
    rollout_steps: int = 4
    teacher_forcing_prob: float = 0.7


def main():
    args = tyro.cli(Args)

    if not (0.0 <= args.teacher_forcing_prob <= 1.0):
        raise ValueError("teacher_forcing_prob must be in [0, 1]")
    if args.rollout_steps < 1:
        raise ValueError("rollout_steps must be >= 1")

    device = pick_device(args.device)
    print(f"[Device] {device}")

    ds = Mario4to1Dataset(args.traj_dir, max_trajs=args.max_trajs, rollout=args.rollout_steps)
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
    start_time = time.monotonic()

    for ep in range(1, args.epochs+1):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device)            # (B,12,H,W) normalized
            yb = yb.to(device)            # (B,S,3,H,W) normalized
            if yb.dim() == 4:             # Handle legacy case with single frame targets
                yb = yb.unsqueeze(1)

            B, S, C, H, W = yb.shape
            context = xb.reshape(B, 4, 3, H, W)
            seed_context = context.clone()
            ms_losses: List[torch.Tensor] = []
            l1_losses: List[torch.Tensor] = []
            pred_seq: List[torch.Tensor] = []
            target_seq: List[torch.Tensor] = []

            for step in range(S):
                model_input = context.reshape(B, 12, H, W)
                preds = model(model_input)
                target = yb[:, step]

                ms_step = ms_ssim_loss(preds, target)
                l1_step = F.l1_loss(preds, target)
                ms_losses.append(ms_step)
                l1_losses.append(l1_step)

                pred_seq.append(preds.detach())
                target_seq.append(target.detach())

                if step + 1 == S:
                    continue

                if args.teacher_forcing_prob >= 1.0:
                    next_frame = target
                elif args.teacher_forcing_prob <= 0.0:
                    next_frame = preds
                else:
                    mask = (torch.rand(B, device=device) < args.teacher_forcing_prob).float().view(B, 1, 1, 1)
                    next_frame = mask * target + (1.0 - mask) * preds

                context = torch.cat([context[:, 1:], next_frame.unsqueeze(1)], dim=1)

            ms_loss = torch.stack(ms_losses).mean()
            l1_loss = torch.stack(l1_losses).mean()
            loss = args.ms_weight * ms_loss + args.l1_weight * l1_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            loss_hist.append((global_step, float(loss.item())))

            if global_step % 10 == 0:
                elapsed = time.monotonic() - start_time
                print(
                    f"[ep {ep:02d}] step {global_step:06d} | loss={loss.item():.4f} "
                    f"(ms={ms_loss.item():.4f}, l1={l1_loss.item():.4f}) | "
                    f"elapsed={elapsed/60:.2f} min"
                )

            if global_step % args.save_every == 0 and pred_seq:
                with torch.no_grad():
                    preds_tensor = torch.stack(pred_seq, dim=1).cpu()
                    targets_tensor = torch.stack(target_seq, dim=1).cpu()
                    save_samples_grid(seed_context.cpu(), preds_tensor, targets_tensor,
                                      out_dir / "samples", global_step)
                elapsed = time.monotonic() - start_time
                plot_loss(loss_hist, out_dir, global_step)
                torch.save({"ep": ep, "step": global_step, "model": model.state_dict()},
                           out_dir / "last.pt")
                print(f"[ep {ep:02d}] saved samples/checkpoint at step {global_step:06d} | elapsed={elapsed/60:.2f} min")

        print(f"[ep {ep:02d}] done.")

    write_loss_csv(loss_hist, out_dir)
    torch.save({"ep": ep, "step": global_step, "model": model.state_dict()}, out_dir / "final.pt")
    print("Training complete.")

if __name__ == "__main__":
    main()
