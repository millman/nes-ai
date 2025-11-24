#!/usr/bin/env python3
"""Mario predictor with explicit camera-viewpoint factorization.

This variant separates the latent space into a canonical background image and
per-step sprite residuals that ride on top of a predicted camera translation.
The goal is to make the model explain bulk motion via the camera pose rather
than by redrawing the background each step. Visualizations include the
canonical background, pose trajectories, and sprite residual heatmaps.
"""
from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler
from torchvision.models import ResNet18_Weights, resnet18
import torchvision.transforms as T
import tyro

from predict_mario_ms_ssim import (
    ConvBlock,
    INV_MEAN,
    INV_STD,
    Mario4to1Dataset,
    Up,
    ms_ssim_loss,
    pick_device,
)


logger = logging.getLogger(__name__)


def grad_l2_norm(parameters) -> float:
    """Compute the L2 norm of gradients for the provided parameters."""

    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += float(param.grad.detach().pow(2).sum().item())
    return total ** 0.5


def latent_summary(latent: torch.Tensor) -> dict[str, float]:
    """Return lightweight stats for a latent batch."""

    lat = latent.detach()
    return {
        "mean": float(lat.mean().item()),
        "std": float(lat.std(unbiased=False).item()),
        "min": float(lat.min().item()),
        "max": float(lat.max().item()),
    }


def compute_high_freq_energy(img: torch.Tensor, low_freq_ratio: float = 0.25) -> float:
    """Estimate per-batch high-frequency energy via FFT.

    Args:
        img: Tensor shaped (B, C, H, W).
        low_freq_ratio: Fraction of the shorter side treated as low-frequency radius.
    """

    if img.dim() != 4:
        raise ValueError("Expected 4D tensor for frequency energy computation")
    if not 0.0 < low_freq_ratio < 1.0:
        raise ValueError("low_freq_ratio must lie in (0, 1)")
    B, C, H, W = img.shape
    if H < 2 or W < 2:
        return 0.0
    freq = torch.fft.fftshift(torch.fft.fftn(img.float(), dim=(-2, -1)), dim=(-2, -1))
    cy, cx = H // 2, W // 2
    radius = max(1, int(min(H, W) * low_freq_ratio * 0.5))
    mask = torch.ones((H, W), device=img.device, dtype=torch.bool)
    y0, y1 = max(0, cy - radius), min(H, cy + radius)
    x0, x1 = max(0, cx - radius), min(W, cx + radius)
    mask[y0:y1, x0:x1] = False
    energy = freq.abs().mean(dim=1)
    masked = energy[:, mask]
    if masked.numel() == 0:
        return 0.0
    return float(masked.mean().item())


def _unnorm(frame: torch.Tensor) -> torch.Tensor:
    """Undo dataset normalization back to 0-1 range."""

    mean = INV_MEAN[:, None, None].to(frame.device, frame.dtype)
    std = INV_STD[:, None, None].to(frame.device, frame.dtype)
    return (frame * std + mean).clamp(0, 1)


def tensor_to_pil(frame: torch.Tensor) -> Image.Image:
    """Convert a CxHxW tensor in dataset space to a PIL image."""

    return T.ToPILImage()(_unnorm(frame).cpu())


def sprite_to_heatmap(sprite: torch.Tensor) -> Image.Image:
    """Map sprite residuals (C,H,W) to a monochrome heatmap for logging."""

    energy = sprite.abs().mean(dim=0, keepdim=False)
    if torch.isnan(energy).any():
        energy = torch.zeros_like(energy)
    energy = energy / (energy.max().clamp(min=1e-6))
    array = (energy.cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def plot_pose_traj(poses: torch.Tensor, out_path: Path) -> None:
    """Save a quick plot of camera translations."""

    steps = poses.shape[0]
    xs = poses[:, 0].cpu().numpy()
    ys = poses[:, 1].cpu().numpy()
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.plot(range(steps), xs, marker="o", label="dx")
    ax.plot(range(steps), ys, marker="s", label="dy")
    ax.axhline(0.0, color="gray", linewidth=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("translation")
    ax.legend(loc="upper right", fontsize=6)
    ax.set_title("Pose trajectory")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _sample_shift(background: torch.Tensor, shift_x: torch.Tensor, shift_y: torch.Tensor, pad: int) -> torch.Tensor:
    """Sample integer shifts with replicate padding."""

    B, C, H, W = background.shape
    padded = F.pad(background, (pad, pad, pad, pad), mode="replicate")
    out = torch.empty_like(background)
    shift_x_list = shift_x.to(torch.int64).cpu().tolist()
    shift_y_list = shift_y.to(torch.int64).cpu().tolist()
    for b, (sx, sy) in enumerate(zip(shift_x_list, shift_y_list)):
        x0 = pad + sx
        y0 = pad + sy
        out[b] = padded[b, :, y0 : y0 + H, x0 : x0 + W]
    return out


def apply_translation(background: torch.Tensor, pose: torch.Tensor, max_translation: float) -> torch.Tensor:
    """Approximate pure translations without using grid_sample.

    Args:
        background: (B, 3, H, W) canonical frame.
        pose: (B, 2) tensor with normalized offsets in [-max_translation, max_translation].
        max_translation: maximum absolute normalized translation.
    """

    if max_translation <= 0:
        return background

    B, _, H, W = background.shape
    max_shift = int(math.ceil(max_translation * 0.5 * max(H, W))) + 2

    dx = pose[:, 0] * (W / 2.0)
    dy = pose[:, 1] * (H / 2.0)

    base_x = torch.floor(dx)
    base_y = torch.floor(dy)
    frac_x = (dx - base_x).view(B, 1, 1, 1)
    frac_y = (dy - base_y).view(B, 1, 1, 1)

    base_x = base_x.to(torch.int64)
    base_y = base_y.to(torch.int64)

    shift00 = _sample_shift(background, base_x, base_y, max_shift)
    shift01 = _sample_shift(background, base_x + 1, base_y, max_shift)
    shift10 = _sample_shift(background, base_x, base_y + 1, max_shift)
    shift11 = _sample_shift(background, base_x + 1, base_y + 1, max_shift)

    out = (
        (1 - frac_x) * (1 - frac_y) * shift00
        + frac_x * (1 - frac_y) * shift01
        + (1 - frac_x) * frac_y * shift10
        + frac_x * frac_y * shift11
    )
    return out


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class ViewpointPredictor(nn.Module):
    def __init__(self, rollout: int = 1, latent_dim: int = 256, max_translation: float = 0.15) -> None:
        super().__init__()
        self.rollout = rollout
        self.steps = rollout + 1  # include reconstruction frame
        self.max_translation = max_translation

        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        backbone.conv1 = nn.Conv2d(12, 64, 7, stride=2, padding=3, bias=False)
        self.enc1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.enc2 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.enc3 = backbone.layer2
        self.enc4 = backbone.layer3
        self.enc5 = backbone.layer4

        self.seed = ConvBlock(512, 512)
        self.up4 = Up(512, 256, 256)
        self.up3 = Up(256, 128, 128)
        self.up2 = Up(128, 64, 64)
        self.up1 = Up(64, 64, 64)

        self.sprite_head = nn.Conv2d(64, self.steps * 3, 1)
        self.background_head = nn.Conv2d(64, 3, 1)
        self.latent_conv = nn.Conv2d(512, latent_dim, 1)
        self.latent_norm = nn.InstanceNorm2d(latent_dim, affine=True)
        self.latent_to_bottleneck = nn.Conv2d(latent_dim, 512, 1)

        self.pose_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.steps * 2),
        )

        self._skip_shapes: dict[str, Tuple[int, int, int]] = {
            "f1": (64, 112, 112),
            "f2": (64, 56, 56),
            "f3": (128, 28, 28),
            "f4": (256, 14, 14),
        }
        self._input_hw: Tuple[int, int] = (224, 224)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        latent_spatial = self.latent_norm(self.latent_conv(bottleneck))

        # Sprite path with skip connections.
        d4 = self.up4(bottleneck, f4)
        d3 = self.up3(d4, f3)
        d2 = self.up2(d3, f2)
        d1 = self.up1(d2, f1)

        sprite = self.sprite_head(d1)
        if sprite.shape[-2:] != (H, W):
            sprite = F.interpolate(sprite, size=(H, W), mode="bilinear", align_corners=False)
        sprite = sprite.contiguous().view(B, self.steps, 3, H, W)

        # Background path only sees latent features.
        def make_zero(key: str) -> torch.Tensor:
            c, h, w = self._skip_shapes[key]
            return torch.zeros((B, c, h, w), device=bottleneck.device, dtype=bottleneck.dtype)

        bottleneck_lat = self.latent_to_bottleneck(latent_spatial)
        zero4 = make_zero("f4")
        zero3 = make_zero("f3")
        zero2 = make_zero("f2")
        zero1 = make_zero("f1")
        bg4 = self.up4(bottleneck_lat, zero4)
        bg3 = self.up3(bg4, zero3)
        bg2 = self.up2(bg3, zero2)
        bg1 = self.up1(bg2, zero1)
        background = self.background_head(bg1)
        if background.shape[-2:] != (H, W):
            background = F.interpolate(background, size=(H, W), mode="bilinear", align_corners=False)

        poses = self.pose_head(bottleneck)
        poses = poses.view(B, self.steps, 2).tanh() * self.max_translation

        background_views: List[torch.Tensor] = []
        for step in range(self.steps):
            view = apply_translation(background, poses[:, step], self.max_translation)
            background_views.append(view)
        background_views_t = torch.stack(background_views, dim=1)

        frames = background_views_t + sprite

        return frames, background, sprite, poses, latent_spatial


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------


def save_viewpoint_samples(
    context4: torch.Tensor,
    frames: torch.Tensor,
    background: torch.Tensor,
    sprite: torch.Tensor,
    poses: torch.Tensor,
    targets: Tuple[torch.Tensor, torch.Tensor],
    out_dir: Path,
    step: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    B, ctx_len, _, H, W = context4.shape
    steps = frames.shape[1]
    rows = min(B, 4)

    for i in range(rows):
        ctx_imgs = [tensor_to_pil(context4[i, j]) for j in range(ctx_len)]
        pred_imgs = [tensor_to_pil(frames[i, s]) for s in range(steps)]
        sprite_maps = [sprite_to_heatmap(sprite[i, s]) for s in range(steps)]
        target_imgs = []
        for s in range(steps):
            if s == 0:
                target_imgs.append(tensor_to_pil(targets[0][i]))
            else:
                tgt_idx = s - 1
                if tgt_idx < targets[1].shape[1]:
                    target_imgs.append(tensor_to_pil(targets[1][i, tgt_idx]))
                else:
                    target_imgs.append(Image.new("RGB", ctx_imgs[0].size, (0, 0, 0)))
        bg_img = tensor_to_pil(background[i])

        tile_w, tile_h = ctx_imgs[0].size
        cols = max(ctx_len, steps)
        canvas = Image.new("RGB", (tile_w * cols, tile_h * 4), (0, 0, 0))
        blank = Image.new("RGB", (tile_w, tile_h), (0, 0, 0))

        # Row 0: context frames (right aligned to last observation)
        start_col = cols - ctx_len
        for k in range(cols):
            if k >= start_col:
                ctx_idx = k - start_col
                canvas.paste(ctx_imgs[ctx_idx], (k * tile_w, 0))
            else:
                canvas.paste(blank, (k * tile_w, 0))

        # Row 1: targets (step0 uses recon target)
        for k in range(steps):
            if k < cols:
                canvas.paste(target_imgs[k], (k * tile_w, tile_h))

        # Row 2: predicted frames
        for k in range(steps):
            if k < cols:
                canvas.paste(pred_imgs[k], (k * tile_w, tile_h * 2))

        # Row 3: sprite residual magnitude heatmaps
        for k in range(steps):
            if k < cols:
                canvas.paste(sprite_maps[k], (k * tile_w, tile_h * 3))

        canvas.save(out_dir / f"sample_step_{step:06d}_idx_{i}.png")

        pose_plot = out_dir / f"pose_step_{step:06d}_idx_{i}.png"
        plot_pose_traj(poses[i], pose_plot)

        bg_img.save(out_dir / f"background_step_{step:06d}_idx_{i}.png")


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
    out_dir: str = "out.predict_mario_viewpoint"
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
    log_every: int = 10
    log_debug_every: int = 50
    log_grad_norms: bool = True
    log_latent_stats: bool = True
    log_frequency_energy: bool = False
    low_freq_ratio: float = 0.25
    max_translation: float = 0.15
    pose_smooth_weight: float = 0.1
    sprite_sparsity_weight: float = 0.05


def main() -> None:
    args = tyro.cli(Args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if not (0.0 <= args.teacher_forcing_prob <= 1.0):
        raise ValueError("teacher_forcing_prob must be in [0,1]")
    device = pick_device(args.device)
    logger.info("Using device: %s", device)

    dataset = Mario4to1Dataset(args.traj_dir, max_trajs=args.max_trajs, rollout=args.rollout_steps)
    logger.info("Dataset size: %d", len(dataset))
    sampler = RandomSampler(dataset, replacement=False, num_samples=args.steps_per_epoch * args.batch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)

    model = ViewpointPredictor(args.rollout_steps, args.latent_dim, max_translation=args.max_translation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    run_dir = Path(args.out_dir) / f"run__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)

    loss_hist: List[Tuple[int, float]] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            B, _, H, W = xb.shape
            recon_target = xb.view(B, 4, 3, H, W)[:, -1]

            frames, background, sprite, poses, latent = model(xb)
            preds = frames[:, 1 : yb.shape[1] + 1]
            recon_pred = frames[:, 0]

            recon_ms = ms_ssim_loss(recon_pred, recon_target)
            recon_l1 = F.l1_loss(recon_pred, recon_target)
            recon_loss = args.lambda_recon * (args.ms_weight * recon_ms + args.l1_weight * recon_l1)

            pred_ms_terms: List[torch.Tensor] = []
            pred_l1_terms: List[torch.Tensor] = []
            for step_idx in range(preds.shape[1]):
                pred_frame = preds[:, step_idx]
                target_frame = yb[:, step_idx]
                pred_ms_terms.append(ms_ssim_loss(pred_frame, target_frame))
                pred_l1_terms.append(F.l1_loss(pred_frame, target_frame))
            if pred_ms_terms:
                pred_ms = torch.stack(pred_ms_terms).mean()
                pred_l1 = torch.stack(pred_l1_terms).mean()
                pred_loss = args.lambda_pred * (args.ms_weight * pred_ms + args.l1_weight * pred_l1)
            else:
                pred_loss = torch.zeros_like(recon_loss)

            pose_smooth = (poses[:, 1:] - poses[:, :-1]).pow(2).mean()
            sprite_sparsity = sprite.abs().mean()

            loss = recon_loss + pred_loss
            loss = loss + args.pose_smooth_weight * pose_smooth + args.sprite_sparsity_weight * sprite_sparsity

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            loss_hist.append((global_step, float(loss.detach().item())))

            if args.log_every > 0 and global_step % args.log_every == 0:
                logger.info(
                    "step=%d epoch=%d loss=%.4f recon=%.4f pred=%.4f pose_smooth=%.4f sprite=%.4f",
                    global_step,
                    epoch,
                    float(loss.detach().item()),
                    float(recon_loss.detach().item()),
                    float(pred_loss.detach().item()),
                    float(pose_smooth.detach().item()),
                    float(sprite_sparsity.detach().item()),
                )

            debug_due = args.log_debug_every > 0 and global_step % args.log_debug_every == 0
            if debug_due:
                debug_pieces: List[str] = []
                if args.log_grad_norms:
                    grad_norm = grad_l2_norm(model.parameters())
                    debug_pieces.append(f"|grad|={grad_norm:.3f}")
                if args.log_latent_stats:
                    stats = latent_summary(latent)
                    debug_pieces.append(
                        "latent(mean={mean:.4f},std={std:.4f},min={min:.4f},max={max:.4f})".format(**stats)
                    )
                if args.log_frequency_energy:
                    freq_energy = compute_high_freq_energy(frames.reshape(B * frames.shape[1], 3, H, W))
                    debug_pieces.append(f"hf={freq_energy:.4f}")
                logger.info("debug %s", " ".join(debug_pieces))

            if global_step % args.save_every == 0:
                context = xb.view(B, 4, 3, H, W)
                targets_tuple = (recon_target.detach(), yb.detach())
                save_viewpoint_samples(
                    context,
                    frames.detach(),
                    background.detach(),
                    sprite.detach(),
                    poses.detach(),
                    targets_tuple,
                    run_dir / "samples",
                    global_step,
                )
                write_loss_csv(loss_hist, run_dir)
                plot_loss(loss_hist, run_dir, global_step)

        logger.info("Completed epoch %d (step %d)", epoch, global_step)


if __name__ == "__main__":
    main()
