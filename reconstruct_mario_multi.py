#!/usr/bin/env python3
"""Latent-only Mario frame reconstruction with object segmentation readout.

This script trains an object-centric autoencoder on NES Mario frames. During
training we feed 4 sequential frames so the encoder can see temporal context,
while the decoder is forced to reconstruct each frame using only per-frame
latents (no encoder skip connections). The model disentangles a small set of
object slots whose soft segmentation maps are used both to compose the base
image and to visualize the learned prototypes.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
import tyro

from predict_mario_ms_ssim import (
    pick_device,
    ms_ssim_loss,
    INV_MEAN,
    INV_STD,
    unnormalize,
)
from trajectory_utils import list_state_frames, list_traj_dirs

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

def pixel_transform() -> T.Compose:
    """Return a transform that preserves the native 240x224 resolution."""
    mean = INV_MEAN.tolist()
    std = INV_STD.tolist()
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


class MarioSequentialDataset(Dataset):
    """Slide a fixed-length window of sequential frames for reconstruction."""

    def __init__(self, root_dir: str, sequence_len: int = 4,
                 transform: Optional[T.Compose] = None,
                 max_trajs: Optional[int] = None) -> None:
        if sequence_len < 1:
            raise ValueError("sequence_len must be >= 1")
        self.sequence_len = sequence_len
        self.transform = transform or pixel_transform()
        self.index: List[Tuple[List[Path], int]] = []
        traj_count = 0
        for traj_path in list_traj_dirs(Path(root_dir)):
            if not traj_path.is_dir():
                continue
            states_dir = traj_path / "states"
            if not states_dir.is_dir():
                continue
            files = list_state_frames(states_dir)
            if len(files) < sequence_len:
                continue
            for start in range(len(files) - sequence_len + 1):
                self.index.append((files, start))
            traj_count += 1
            if max_trajs and traj_count >= max_trajs:
                break
        if not self.index:
            raise RuntimeError(f"No trajectories found under {root_dir}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        files, start = self.index[idx]
        frames: List[torch.Tensor] = []
        for offset in range(self.sequence_len):
            with Image.open(files[start + offset]).convert("RGB") as img:
                frames.append(self.transform(img))
        return torch.stack(frames, dim=0)  # (T,3,H,W)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class UpDecoderBlock(nn.Module):
    """Nearest-neighbor upsample + conv stack to keep edges crisp."""

    def __init__(self, c_in: int, c_out: int, groups: int = 8) -> None:
        super().__init__()
        norm_groups = groups if c_out % groups == 0 else 1
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, c_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MarioObjectReconstructor(nn.Module):
    """Encode frames and reconstruct via slot-based palettes and residuals."""

    def __init__(self, latent_dim: int = 256, num_objects: int = 16) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.enc1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2
        self.enc2 = nn.Sequential(backbone.maxpool, backbone.layer1)            # /4
        self.enc3 = backbone.layer2                                            # /8
        self.enc4 = backbone.layer3                                            # /16 (15x14 from 240x224)
        self.latent_proj = nn.Conv2d(256, latent_dim, kernel_size=1)
        self.latent_norm = nn.GroupNorm(1, latent_dim)

        self.dec1 = UpDecoderBlock(latent_dim, 256)
        self.dec2 = UpDecoderBlock(256, 128)
        self.dec3 = UpDecoderBlock(128, 64)
        self.dec4 = UpDecoderBlock(64, 64)
        self.seg_head = nn.Conv2d(64, num_objects, kernel_size=1)
        self.residual_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
        )
        self.object_palette = nn.Parameter(torch.randn(num_objects, 3))
        self.num_objects = num_objects
        self.latent_dim = latent_dim
        # NES frames arrive as (H=224, W=240); latent grid matches encoder downsampling.
        self._latent_hw = (14, 15)
        self._target_hw = (224, 240)

    # Encoder -----------------------------------------------------------------
    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode (B,3,H,W) frames into spatial latents."""
        if frames.dim() != 4:
            raise ValueError("Expected frames with shape (B,3,H,W)")
        f1 = self.enc1(frames)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        latent = self.latent_norm(self.latent_proj(f4))
        return latent

    # Decoder -----------------------------------------------------------------
    def _decode_latent_map(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if latent.shape[-2:] != self._latent_hw:
            latent = F.interpolate(latent, size=self._latent_hw, mode="nearest")
        x = self.dec1(latent)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        if x.shape[-2:] != self._target_hw:
            x = F.interpolate(x, size=self._target_hw, mode="nearest")
        seg_logits = self.seg_head(x)
        seg_probs = seg_logits.softmax(dim=1)
        palette = self.object_palette.view(1, self.num_objects, 3, 1, 1)
        palette_base = (seg_probs.unsqueeze(2) * palette).sum(dim=1)
        residual = self.residual_head(x)
        recon = palette_base + residual
        return recon, seg_logits, palette_base, residual

    def decode_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        recon, _, _, _ = self._decode_latent_map(latent)
        return recon

    # Full forward ------------------------------------------------------------
    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct each frame independently without encoder skips.

        Args:
            frames: Tensor shaped (B,T,3,H,W).
        Returns:
            recon:     (B,T,3,H,W)
            seg_logits:(B,T,num_objects,H,W)
            palette:   (num_objects,3)
            residual:  (B,T,3,H,W) residual contribution
            latents:   (B,T,latent_dim,15,14)
        """
        if frames.dim() != 5:
            raise ValueError("Expected frames with shape (B,T,3,H,W)")
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)
        latent = self.encode_frames(flat)
        recon, seg_logits, palette_base, residual = self._decode_latent_map(latent)
        recon = recon.view(B, T, 3, self._target_hw[0], self._target_hw[1])
        seg_logits = seg_logits.view(B, T, self.num_objects, self._target_hw[0], self._target_hw[1])
        palette_base = palette_base.view(B, T, 3, self._target_hw[0], self._target_hw[1])
        residual = residual.view(B, T, 3, self._target_hw[0], self._target_hw[1])
        latents = latent.view(B, T, self.latent_dim, *self._latent_hw)
        return recon, seg_logits, palette_base, residual, latents

    @torch.no_grad()
    def objects_palette(self) -> torch.Tensor:
        """Return current object palette in normalized RGB space (num_objects,3)."""
        return self.object_palette.detach().clone()


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    """Convert normalized (3,H,W) tensor to a PIL image."""
    if img.dim() != 3:
        raise ValueError("Expected (3,H,W) tensor")
    img = img.unsqueeze(0)
    img = unnormalize(img)[0].clamp(0, 1)
    arr = (img.mul(255).round().byte().cpu().permute(1, 2, 0).numpy())
    return Image.fromarray(arr, mode="RGB")


def segmentation_to_pil(seg_logits: torch.Tensor, palette: Optional[np.ndarray] = None) -> Image.Image:
    """Visualize segmentation logits (num_objects,H,W) as a colored mask."""
    if seg_logits.dim() != 3:
        raise ValueError("Expected (num_objects,H,W) logits")
    seg_idx = seg_logits.argmax(dim=0).cpu().numpy().astype(np.int32)
    num_objects = seg_logits.shape[0]
    if palette is None:
        cmap = plt.get_cmap("tab20")
        base_colors = (np.array([cmap(i % cmap.N)[:3] for i in range(num_objects)]) * 255).astype(np.uint8)
    else:
        base_colors = palette.astype(np.uint8)
    color_img = base_colors[seg_idx]
    return Image.fromarray(color_img, mode="RGB")


def save_samples(frames: torch.Tensor, recon: torch.Tensor, seg_logits: torch.Tensor,
                 out_dir: Path, step: int, max_items: int = 3) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    B, T = frames.shape[0], frames.shape[1]
    rows = min(B, max_items)
    for i in range(rows):
        panes: List[Image.Image] = []
        for t in range(T):
            original = tensor_to_pil(frames[i, t])
            reconstructed = tensor_to_pil(recon[i, t])
            seg_img = segmentation_to_pil(seg_logits[i, t])
            panes.extend([original, reconstructed, seg_img])
        w, h = panes[0].size
        cols = 3
        canvas = Image.new("RGB", (cols * w, T * h))
        for idx, pane in enumerate(panes):
            col = idx % cols
            row = idx // cols
            canvas.paste(pane, (col * w, row * h))
        canvas.save(out_dir / f"sample_step_{step:06d}_idx_{i}.png")


def save_palette_image(palette: torch.Tensor, out_dir: Path, step: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = palette.detach().cpu()
    mean = INV_MEAN.view(1, 3)
    std = INV_STD.view(1, 3)
    rgb = ((palette * std) + mean).clamp(0, 1)
    tiles: List[Image.Image] = []
    for color in rgb:
        arr = (color.clamp(0, 1).numpy() * 255).astype(np.uint8)
        patch = np.ones((32, 32, 3), dtype=np.uint8) * arr[None, None, :]
        tiles.append(Image.fromarray(patch, mode="RGB"))
    cols = 8
    rows = int(np.ceil(len(tiles) / cols))
    tile_w, tile_h = tiles[0].size
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h))
    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        canvas.paste(tile, (c * tile_w, r * tile_h))
    canvas.save(out_dir / f"palette_step_{step:06d}.png")


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
# Training
# -----------------------------------------------------------------------------

@dataclass
class Args:
    traj_dir: str = "data.image_distance.train_levels_1_2"
    out_dir: str = "out.reconstruct_mario_multi"
    batch_size: int = 16
    lr: float = 1e-4
    epochs: int = 1000
    steps_per_epoch: int = 100
    num_frames: int = 4
    max_trajs: Optional[int] = None
    save_every: int = 50
    log_every: int = 10
    log_debug_every: int = 50
    num_workers: int = 0
    device: Optional[str] = None
    latent_dim: int = 256
    num_objects: int = 16
    ms_weight: float = 1.0
    l1_weight: float = 0.1
    entropy_weight: float = 0.001
    residual_l2_weight: float = 0.0005


def latent_summary(latent: torch.Tensor) -> dict[str, float]:
    lat = latent.detach()
    return {
        "mean": float(lat.mean().item()),
        "std": float(lat.std(unbiased=False).item()),
        "min": float(lat.min().item()),
        "max": float(lat.max().item()),
    }


def main() -> None:
    args = tyro.cli(Args)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    device = pick_device(args.device)
    logger.info("Using device: %s", device)

    dataset = MarioSequentialDataset(args.traj_dir, sequence_len=args.num_frames,
                                     max_trajs=args.max_trajs)
    logger.info("Dataset size: %d", len(dataset))
    sampler = RandomSampler(dataset, replacement=False,
                            num_samples=args.steps_per_epoch * args.batch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers)

    model = MarioObjectReconstructor(args.latent_dim, args.num_objects).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    run_dir = Path(args.out_dir) / f"run__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    samples_dir = run_dir / "samples"
    palette_dir = run_dir / "palettes"
    samples_dir.mkdir(parents=True, exist_ok=True)
    palette_dir.mkdir(parents=True, exist_ok=True)

    loss_hist: List[Tuple[int, float]] = []
    global_step = 0
    start_time = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb in loader:
            xb = xb.to(device)  # (B,T,3,H,W)
            recon, seg_logits, palette_base, residual, latents = model(xb)

            ms_terms: List[torch.Tensor] = []
            l1_terms: List[torch.Tensor] = []
            for t in range(xb.shape[1]):
                ms_terms.append(ms_ssim_loss(recon[:, t], xb[:, t]))
                l1_terms.append(F.l1_loss(recon[:, t], xb[:, t]))
            ms_loss = torch.stack(ms_terms).mean()
            l1_loss = torch.stack(l1_terms).mean()
            loss = args.ms_weight * ms_loss + args.l1_weight * l1_loss

            if args.entropy_weight > 0.0:
                seg_probs = seg_logits.softmax(dim=2)
                entropy = -(seg_probs * (seg_probs.clamp_min(1e-8).log())).sum(dim=2).mean()
                loss = loss + args.entropy_weight * entropy

            if args.residual_l2_weight > 0.0:
                residual_penalty = residual.pow(2).mean()
                loss = loss + args.residual_l2_weight * residual_penalty

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            loss_hist.append((global_step, float(loss.item())))

            if args.log_every > 0 and global_step % args.log_every == 0:
                elapsed = (time.monotonic() - start_time) / 60
                logger.info(
                    "[ep %03d] step %06d | loss=%.4f | ms=%.4f | l1=%.4f | elapsed=%.2f min",
                    epoch,
                    global_step,
                    loss.item(),
                    ms_loss.item(),
                    l1_loss.item(),
                    elapsed,
                )

            if args.log_debug_every > 0 and global_step % args.log_debug_every == 0:
                stats = latent_summary(latents)
                logger.info("[step %06d] latent stats | mean=%.4f std=%.4f min=%.4f max=%.4f",
                            global_step, stats["mean"], stats["std"], stats["min"], stats["max"])

            if args.save_every > 0 and global_step % args.save_every == 0:
                with torch.no_grad():
                    save_samples(xb.cpu(), recon.cpu(), seg_logits.cpu(), samples_dir, global_step)
                    save_palette_image(model.objects_palette(), palette_dir, global_step)
                plot_loss(loss_hist, run_dir, global_step)
                torch.save({"epoch": epoch, "step": global_step, "model": model.state_dict()},
                           run_dir / "checkpoint.pt")

        logger.info("[ep %03d] done.", epoch)

    write_loss_csv(loss_hist, run_dir)
    torch.save({"epoch": epoch, "step": global_step, "model": model.state_dict()},
               run_dir / "final.pt")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
