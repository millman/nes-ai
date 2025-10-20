#!/usr/bin/env python3
"""Multi-task Mario predictor with shared latent.

This script extends ``predict_mario_ms_ssim.py`` by jointly training:
  * a reconstruction decoder that reconstructs the latest input frame (autoencoder head)
  * a future-frame decoder that predicts the next ``rollout`` frames (forecast head)

The encoder learns a latent embedding that supports both tasks. Use
``--lambda-recon``/``--lambda-pred`` to balance losses.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import tyro

from predict_mario_ms_ssim import (
    pick_device,
    default_transform,
    Mario4to1Dataset,
    ms_ssim_loss,
    ConvBlock,
    Up,
    INV_MEAN,
    INV_STD,
    unnormalize,
)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class MultiHeadPredictor(nn.Module):
    def __init__(self, rollout: int = 1, latent_dim: int = 256) -> None:
        super().__init__()
        self.rollout = rollout
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        backbone.conv1 = nn.Conv2d(12, 64, 7, stride=2, padding=3, bias=False)
        self.enc1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # (B,64,H/2,W/2)
        self.enc2 = nn.Sequential(backbone.maxpool, backbone.layer1)            # (B,64,H/4,W/4)
        self.enc3 = backbone.layer2                                            # (B,128,H/8,W/8)
        self.enc4 = backbone.layer3                                            # (B,256,H/16,W/16)
        self.enc5 = backbone.layer4                                            # (B,512,H/32,W/32)
        self.seed = ConvBlock(512, 512)
        self.up4 = Up(512, 256, 256)
        self.up3 = Up(256, 128, 128)
        self.up2 = Up(128, 64, 64)
        self.up1 = Up(64, 64, 64)
        self.recon_head = nn.Conv2d(64, 3, 1)
        self.pred_head = nn.Conv2d(64, rollout * 3, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.latent_fc = nn.Linear(512, latent_dim)
        self.latent_norm = nn.LayerNorm(latent_dim)
        self._latent_hw: Optional[Tuple[int, int]] = (7, 7)
        self.latent_to_bottleneck = nn.Linear(latent_dim, 512 * self._latent_hw[0] * self._latent_hw[1])
        self._skip_shapes: dict[str, Tuple[int, int, int]] = {
            "f1": (64, 112, 112),
            "f2": (64, 56, 56),
            "f3": (128, 28, 28),
            "f4": (256, 14, 14),
        }
        self._input_hw: Tuple[int, int] = (224, 224)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, H, W = x.shape
        if (H, W) != self._input_hw:
            raise RuntimeError(f"Expected input spatial size {self._input_hw}, got {(H, W)}")
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        f5 = self.enc5(f4)
        expected_shapes = {
            "f1": (64, H // 2, W // 2),
            "f2": (64, H // 4, W // 4),
            "f3": (128, H // 8, W // 8),
            "f4": (256, H // 16, W // 16),
            "f5": (512, H // 32, W // 32),
        }
        actual_shapes = {
            "f1": f1.shape[1:],
            "f2": f2.shape[1:],
            "f3": f3.shape[1:],
            "f4": f4.shape[1:],
            "f5": f5.shape[1:],
        }
        for key, expected in expected_shapes.items():
            if actual_shapes[key] != expected:
                raise RuntimeError(f"Expected {key} shape {expected}, got {actual_shapes[key]}")
        bottleneck = self.seed(f5)
        pooled = self.pool(bottleneck).view(B, -1)
        latent = self.latent_norm(self.latent_fc(pooled))
        if (H, W) != self._input_hw:
            raise RuntimeError(f"Expected input spatial size {self._input_hw}, got {(H, W)}")
        expected_shapes = {
            "f1": (64, H // 2, W // 2),
            "f2": (64, H // 4, W // 4),
            "f3": (128, H // 8, W // 8),
            "f4": (256, H // 16, W // 16),
        }
        actual_shapes = {
            "f1": f1.shape[1:],
            "f2": f2.shape[1:],
            "f3": f3.shape[1:],
            "f4": f4.shape[1:],
        }
        for key, expected in expected_shapes.items():
            if actual_shapes[key] != expected:
                raise RuntimeError(f"Expected {key} shape {expected}, got {actual_shapes[key]}")

        # Prediction branch (uses skips as usual)
        d4 = self.up4(bottleneck, f4)
        d3 = self.up3(d4, f3)
        d2 = self.up2(d3, f2)
        d1 = self.up1(d2, f1)
        preds = self.pred_head(d1)
        if preds.shape[-2:] != (H, W):
            preds = F.interpolate(preds, size=(H, W), mode="bilinear", align_corners=False)
        preds = preds.view(B, self.rollout, 3, H, W)
        # ensure reconstruction head only sees latent-derived features
        h5, w5 = f5.shape[-2:]
        if (h5, w5) != self._latent_hw:
            raise RuntimeError(
                f"Expected encoder bottleneck spatial size {self._latent_hw}, got {(h5, w5)}"
            )
        bottleneck_lat = self.latent_to_bottleneck(latent).view(B, 512, h5, w5)
        def make_zero(key: str) -> torch.Tensor:
            c, h, w = self._skip_shapes[key]
            return torch.zeros((B, c, h, w), device=bottleneck_lat.device, dtype=bottleneck_lat.dtype)

        zero4 = make_zero("f4")
        zero3 = make_zero("f3")
        zero2 = make_zero("f2")
        zero1 = make_zero("f1")
        d4_recon = self.up4(bottleneck_lat, zero4)
        d3_recon = self.up3(d4_recon, zero3)
        d2_recon = self.up2(d3_recon, zero2)
        d1_recon = self.up1(d2_recon, zero1)
        recon = self.recon_head(d1_recon)
        if recon.shape[-2:] != (H, W):
            recon = F.interpolate(recon, size=(H, W), mode="bilinear", align_corners=False)
        return recon, preds, latent

    def decode_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self._skip_shapes is None or self._input_hw is None:
            raise RuntimeError("Decoder skip shapes unavailable. Run a forward pass first to capture them.")
        h5, w5 = self._latent_hw
        B = latent.shape[0]
        device = latent.device
        dtype = latent.dtype
        bottleneck_lat = self.latent_to_bottleneck(latent).view(B, 512, h5, w5)

        def make_zero(key: str) -> torch.Tensor:
            c, h, w = self._skip_shapes[key]
            return torch.zeros((B, c, h, w), device=device, dtype=dtype)

        zero4 = make_zero("f4")
        zero3 = make_zero("f3")
        zero2 = make_zero("f2")
        zero1 = make_zero("f1")
        d4_recon = self.up4(bottleneck_lat, zero4)
        d3_recon = self.up3(d4_recon, zero3)
        d2_recon = self.up2(d3_recon, zero2)
        d1_recon = self.up1(d2_recon, zero1)
        recon = self.recon_head(d1_recon)
        if self._input_hw is not None and recon.shape[-2:] != self._input_hw:
            recon = F.interpolate(recon, size=self._input_hw, mode="bilinear", align_corners=False)
        return recon

# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

def save_multi_samples(context4: torch.Tensor, recon: torch.Tensor,
                       preds: torch.Tensor, targets: torch.Tensor,
                       out_dir: Path, step: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    to_pil = T.ToPILImage()

    def unnorm(frame: torch.Tensor) -> torch.Tensor:
        mean = INV_MEAN[:, None, None].to(frame.device, frame.dtype)
        std = INV_STD[:, None, None].to(frame.device, frame.dtype)
        return (frame * std + mean).clamp(0, 1)

    B, ctx_len, _, H, W = context4.shape
    steps = preds.shape[1]
    rows = min(B, 4)
    for i in range(rows):
        ctx_imgs = [to_pil(unnorm(context4[i, j])) for j in range(ctx_len)]
        recon_img = to_pil(unnorm(recon[i]))
        tgt_imgs = [to_pil(unnorm(targets[i, s])) for s in range(steps)]
        pred_imgs = [to_pil(unnorm(preds[i, s])) for s in range(steps)]
        tile_w, tile_h = ctx_imgs[0].size
        cols = ctx_len + steps
        canvas = Image.new('RGB', (tile_w * cols, tile_h * 3))
        blank = Image.new('RGB', (tile_w, tile_h), (0, 0, 0))

        # Row 0: context frames followed by rollout targets
        for k in range(cols):
            if k < ctx_len:
                canvas.paste(ctx_imgs[k], (k * tile_w, 0))
            else:
                tgt_idx = k - ctx_len
                if tgt_idx < len(tgt_imgs):
                    canvas.paste(tgt_imgs[tgt_idx], (k * tile_w, 0))
                else:
                    canvas.paste(blank, (k * tile_w, 0))

        # Row 1: predictions aligned with targets
        for k in range(cols):
            if k < ctx_len:
                canvas.paste(blank, (k * tile_w, tile_h))
            else:
                pred_idx = k - ctx_len
                if pred_idx < len(pred_imgs):
                    canvas.paste(pred_imgs[pred_idx], (k * tile_w, tile_h))
                else:
                    canvas.paste(blank, (k * tile_w, tile_h))

        # Row 2: reconstruction under the last context frame column
        for k in range(cols):
            if k == ctx_len - 1:
                canvas.paste(recon_img, (k * tile_w, tile_h * 2))
            else:
                canvas.paste(blank, (k * tile_w, tile_h * 2))

        canvas.save(out_dir / f"sample_step_{step:06d}_idx_{i}.png")

def write_loss_csv(hist: List[Tuple[int, float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "loss_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        writer.writerows(hist)

def plot_loss(hist: List[Tuple[int, float]], out_dir: Path, step: int) -> None:
    if not hist:
        return
    steps, losses = zip(*hist)
    plt.figure()
    plt.semilogy(steps, losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training loss")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"loss_step_{step:06d}.png")
    plt.close()

# -----------------------------------------------------------------------------
# Training CLI
# -----------------------------------------------------------------------------

@dataclass
class Args:
    traj_dir: str = "data.image_distance.train_levels_1_2"
    out_dir: str = "out.predict_mario_multi"
    batch_size: int = 32
    lr: float = 1e-4
    epochs: int = 1000
    steps_per_epoch: int = 100
    rollout_steps: int = 4
    max_trajs: Optional[int] = None
    save_every: int = 50
    num_workers: int = 0
    device: Optional[str] = None
    lambda_recon: float = 1.0
    lambda_pred: float = 1.0
    ms_weight: float = 1.0
    l1_weight: float = 0.1
    latent_dim: int = 256
    teacher_forcing_prob: float = 0.5


def main() -> None:
    args = tyro.cli(Args)
    if not (0.0 <= args.teacher_forcing_prob <= 1.0):
        raise ValueError("teacher_forcing_prob must be in [0,1]")
    device = pick_device(args.device)
    print(f"[Device] {device}")

    dataset = Mario4to1Dataset(args.traj_dir, max_trajs=args.max_trajs, rollout=args.rollout_steps)
    print(f"Dataset: {len(dataset)} samples")
    sampler = RandomSampler(dataset, replacement=False,
                            num_samples=args.steps_per_epoch * args.batch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers)

    model = MultiHeadPredictor(args.rollout_steps, args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    run_dir = Path(args.out_dir) / f"run__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)

    loss_hist: List[Tuple[int, float]] = []
    global_step = 0
    start_time = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            B = xb.shape[0]
            # latest frame (most recent input) for reconstruction target
            recon_target = xb.view(B, 4, 3, xb.shape[-2], xb.shape[-1])[:, -1]
            recon_pred, preds, latent = model(xb)
            preds = preds[:, :yb.shape[1]]

            recon_ms = ms_ssim_loss(recon_pred, recon_target)
            recon_l1 = F.l1_loss(recon_pred, recon_target)
            recon_loss = args.lambda_recon * (args.ms_weight * recon_ms + args.l1_weight * recon_l1)

            pred_ms_terms: List[torch.Tensor] = []
            pred_l1_terms: List[torch.Tensor] = []
            for step in range(yb.shape[1]):
                pred_frame = preds[:, step]
                target_frame = yb[:, step]
                pred_ms_terms.append(ms_ssim_loss(pred_frame, target_frame))
                pred_l1_terms.append(F.l1_loss(pred_frame, target_frame))
            pred_ms = torch.stack(pred_ms_terms).mean()
            pred_l1 = torch.stack(pred_l1_terms).mean()
            pred_loss = args.lambda_pred * (args.ms_weight * pred_ms + args.l1_weight * pred_l1)

            loss = recon_loss + pred_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            loss_hist.append((global_step, float(loss.item())))

            if global_step % 10 == 0:
                elapsed = (time.monotonic() - start_time) / 60
                print(
                    f"[ep {epoch:03d}] step {global_step:06d} | "
                    f"loss={loss.item():.4f} | recon(ms={recon_ms.item():.4f}, l1={recon_l1.item():.4f}) "
                    f"pred(ms={pred_ms.item():.4f}, l1={pred_l1.item():.4f}) | elapsed={elapsed:.2f} min"
                )

            if global_step % args.save_every == 0:
                with torch.no_grad():
                    context = xb.view(xb.size(0), 4, 3, preds.size(-2), preds.size(-1))
                    save_multi_samples(context.cpu(), recon_pred.cpu(), preds.cpu(), yb.cpu(),
                                       run_dir / "samples", global_step)
                plot_loss(loss_hist, run_dir, global_step)
                torch.save({"epoch": epoch, "step": global_step, "model": model.state_dict()},
                           run_dir / "checkpoint.pt")

        print(f"[ep {epoch:03d}] done.")

    write_loss_csv(loss_hist, run_dir)
    torch.save({"epoch": epoch, "step": global_step, "model": model.state_dict()},
               run_dir / "final.pt")
    print("Training complete.")

if __name__ == "__main__":
    main()
