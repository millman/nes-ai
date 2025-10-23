#!/usr/bin/env python3
"""CFM-based single-frame reconstruction for NES Mario trajectories.

This script trains a latent autoencoder that reconstructs individual frames
without relying on encoder/decoder skip connections. The latent representation
is regularised with Conditional Flow Matching (CFM) so that the encoder maps
frames onto a flow-accessible manifold. Visualisations include:

  • input vs. reconstruction comparisons
  • latent PCA traversals decoded back to pixel space
"""
from __future__ import annotations

import csv
import logging
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torchvision.transforms as T
import tyro

from predict_mario_ms_ssim import (
    INV_MEAN,
    INV_STD,
    default_transform,
    ms_ssim_loss,
    pick_device,
    unnormalize,
)
from trajectory_utils import list_state_frames, list_traj_dirs

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class MarioFrameDataset(Dataset):
    """Flat list of Mario frames, normalised with ImageNet statistics."""

    def __init__(
        self,
        root_dir: str,
        *,
        transform: Optional[T.Compose] = None,
        max_trajs: Optional[int] = None,
    ) -> None:
        self.transform = transform or default_transform()
        self.paths: List[Path] = []
        traj_count = 0
        for traj_dir in list_traj_dirs(Path(root_dir)):
            if not traj_dir.is_dir():
                continue
            states_dir = traj_dir / "states"
            if not states_dir.is_dir():
                continue
            for frame_path in list_state_frames(states_dir):
                self.paths.append(frame_path)
            traj_count += 1
            if max_trajs is not None and traj_count >= max_trajs:
                break
        if not self.paths:
            raise RuntimeError(f"No frames found under {root_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.paths[idx]
        with Image.open(path).convert("RGB") as img:
            tensor = self.transform(img)
        return tensor, str(path)


# -----------------------------------------------------------------------------
# Model components
# -----------------------------------------------------------------------------


def _make_group_norm(num_channels: int) -> nn.GroupNorm:
    groups = 8 if num_channels % 8 == 0 else 1
    return nn.GroupNorm(groups, num_channels)


class UpBlock(nn.Module):
    """Conv-transpose up-sampling block without skip connections."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
            _make_group_norm(c_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            _make_group_norm(c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    """Lightweight residual block with optional down-sampling."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1)
        self.norm1 = _make_group_norm(c_out)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1)
        self.norm2 = _make_group_norm(c_out)
        if stride != 1 or c_in != c_out:
            self.skip = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride),
                _make_group_norm(c_out),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.skip(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return self.act(out + shortcut)


class LatentEncoder(nn.Module):
    """Custom ResNet-style encoder that preserves spatial detail."""

    def __init__(
        self,
        latent_dim: int,
        *,
        base_channels: int = 64,
        input_hw: Tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        if input_hw[0] % 16 != 0 or input_hw[1] % 16 != 0:
            raise ValueError("input_hw dimensions must be divisible by 16.")
        if base_channels % 2 != 0:
            raise ValueError("base_channels must be an even number.")
        self.input_hw = input_hw
        self.latent_hw = (input_hw[0] // 16, input_hw[1] // 16)

        stem_width = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_width, kernel_size=3, stride=1, padding=1),
            _make_group_norm(stem_width),
            nn.SiLU(inplace=True),
        )

        stage_channels = [
            stem_width,
            stem_width * 3 // 2,
            stem_width * 2,
            stem_width * 5 // 2,
        ]
        self.stages = nn.ModuleList()
        in_ch = stem_width
        for out_ch in stage_channels:
            block = nn.Sequential(
                ResidualBlock(in_ch, out_ch, stride=2),
                ResidualBlock(out_ch, out_ch, stride=1),
            )
            self.stages.append(block)
            in_ch = out_ch

        self.post = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            _make_group_norm(in_ch),
            nn.SiLU(inplace=True),
        )
        flat_dim = in_ch * self.latent_hw[0] * self.latent_hw[1]
        self.fc = nn.Linear(flat_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != self.input_hw:
            raise RuntimeError(f"Expected input spatial size {self.input_hw}, got {tuple(x.shape[-2:])}")
        h = self.stem(x)
        for stage in self.stages:
            h = stage(h)
        h = self.post(h)
        if h.shape[-2:] != self.latent_hw:
            raise RuntimeError(f"Expected latent feature map {self.latent_hw}, got {tuple(h.shape[-2:])}")
        h = h.flatten(1)
        latent = self.fc(h)
        return self.norm(latent)


class LatentDecoder(nn.Module):
    """Lightweight CNN decoder that relies solely on the latent vector."""

    def __init__(self, latent_dim: int, out_hw: Tuple[int, int] = (224, 224)) -> None:
        super().__init__()
        self.out_hw = out_hw
        base_hw = (7, 7)
        base_ch = 512
        self.fc = nn.Linear(latent_dim, base_ch * base_hw[0] * base_hw[1])
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)
        self.pre = nn.SiLU(inplace=True)
        self.up1 = UpBlock(base_ch, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up4 = UpBlock(64, 32)
        self.up5 = UpBlock(32, 16)
        self.head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        h = self.fc(latent)
        h = self.pre(h).view(-1, 512, 7, 7)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        h = self.up5(h)
        out = self.head(h)
        if out.shape[-2:] != self.out_hw:
            out = F.interpolate(out, size=self.out_hw, mode="bilinear", align_corners=False)
        return out


class ConditionalVectorField(nn.Module):
    """Implements the conditional velocity field used for flow matching."""

    def __init__(self, latent_dim: int, hidden: int = 512) -> None:
        super().__init__()
        in_dim = latent_dim * 3 + 1  # [z_t, delta, midpoint, t]
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        z_data: torch.Tensor,
        z_base: torch.Tensor,
    ) -> torch.Tensor:
        delta = z_data - z_base
        midpoint = 0.5 * (z_data + z_base)
        feat = torch.cat([z_t, delta, midpoint, t[:, None]], dim=1)
        return self.net(feat)


class CFMAutoencoder(nn.Module):
    """Encoder / decoder pair trained with CFM regularisation."""

    def __init__(
        self,
        latent_dim: int,
        *,
        encoder_base_channels: int = 64,
        input_hw: Tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        self.encoder = LatentEncoder(
            latent_dim,
            base_channels=encoder_base_channels,
            input_hw=input_hw,
        )
        self.decoder = LatentDecoder(latent_dim, out_hw=input_hw)
        self.vector_field = ConditionalVectorField(latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


# -----------------------------------------------------------------------------
# Visualisation helpers
# -----------------------------------------------------------------------------


def latent_summary(latent: torch.Tensor) -> dict[str, float]:
    data = latent.detach()
    return {
        "mean": float(data.mean().item()),
        "std": float(data.std(unbiased=False).item()),
        "min": float(data.min().item()),
        "max": float(data.max().item()),
    }


@torch.no_grad()
def collect_latents(
    model: CFMAutoencoder,
    dataset: Dataset,
    device: torch.device,
    *,
    max_samples: int,
    batch_size: int,
) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    latents: List[torch.Tensor] = []
    total = 0
    for xb, _ in loader:
        xb = xb.to(device)
        lat = model.encode(xb).cpu()
        latents.append(lat)
        total += lat.shape[0]
        if total >= max_samples:
            break
    if not latents:
        raise RuntimeError("Failed to collect latents for PCA.")
    return torch.cat(latents, dim=0)[:max_samples]


def compute_pca_components(
    latents: torch.Tensor, n_components: int
) -> Tuple[List[torch.Tensor], List[float], List[torch.Tensor]]:
    """Return PCA directions, their std devs, and projection samples."""
    if latents.dim() != 2:
        raise ValueError("Expected 2D latent matrix")
    centered = latents - latents.mean(dim=0, keepdim=True)
    max_components = min(n_components, centered.shape[1])
    if max_components == 0:
        raise ValueError("No components available for PCA traversal")
    _, _, V = torch.pca_lowrank(centered, q=max_components)
    projections = centered @ V[:, :max_components]
    directions: List[torch.Tensor] = []
    stds: List[float] = []
    projection_samples: List[torch.Tensor] = []
    for idx in range(max_components):
        directions.append(V[:, idx])
        comp_proj = projections[:, idx].detach().cpu()
        std = float(comp_proj.std(unbiased=False).item())
        stds.append(1.0 if std == 0.0 else std)
        projection_samples.append(comp_proj)
    return directions, stds, projection_samples


def to_image(tensor: torch.Tensor) -> Image.Image:
    if tensor.dim() != 3:
        raise ValueError("Expected image tensor with shape (C,H,W)")
    # The tensor is normalised; unnormalise before converting to PIL.
    img = unnormalize(tensor.unsqueeze(0)).squeeze(0).clip(0, 1)
    to_pil = T.ToPILImage()
    return to_pil(img.cpu())


@torch.no_grad()
def save_pca_traversal_grid(
    model: CFMAutoencoder,
    input_frame: torch.Tensor,
    recon_frame: torch.Tensor,
    latent: torch.Tensor,
    directions: Sequence[torch.Tensor],
    stds: Sequence[float],
    percent_levels: Sequence[float],
    projections: Sequence[torch.Tensor],
    out_path: Path,
    device: torch.device,
) -> None:
    """Save a grid showing PCA traversals for a single sample."""
    if not directions:
        raise ValueError("No PCA directions provided")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_img = to_image(input_frame)
    recon_img = to_image(recon_frame)
    tile_w, tile_h = base_img.size
    num_rows = len(directions)
    num_cols = 1 + len(percent_levels)
    canvas = Image.new("RGB", (num_cols * tile_w, num_rows * tile_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    latent_device = latent.to(device)
    percent_list = [float(p) for p in percent_levels]
    label_height = 24

    for row_idx, (direction, std, comp_proj) in enumerate(
        zip(directions, stds, projections), start=1
    ):
        y = (row_idx - 1) * tile_h
        # First column: source frame
        canvas.paste(base_img, (0, y))
        draw.rectangle([4, y + 4, tile_w - 4, y + 28], outline=(255, 255, 0))
        draw.text((8, y + 8), f"PC{row_idx}", fill=(255, 255, 0))
        # Annotate source (0%)
        draw.rectangle(
            [0, y + tile_h - label_height, tile_w, y + tile_h],
            fill=(0, 0, 0),
        )
        draw.text(
            (6, y + tile_h - label_height + 4),
            "+0.000 (+0%)",
            fill=(255, 255, 0),
        )

        direction_device = direction.to(device)
        comp_proj_float = comp_proj.to(dtype=torch.float32)
        for col_idx, percent in enumerate(percent_list, start=1):
            x = col_idx * tile_w
            if abs(percent) < 1e-6:
                tile = recon_img
                delta = 0.0
            else:
                q = 0.5 + 0.5 * percent
                q = max(0.0, min(1.0, q))
                delta = float(torch.quantile(comp_proj_float, q))
                shifted = latent_device + delta * direction_device
                decoded = model.decode(shifted.unsqueeze(0)).cpu()[0]
                tile = to_image(decoded)
            canvas.paste(tile, (x, y))
            draw.rectangle(
                [x, y + tile_h - label_height, x + tile_w, y + tile_h],
                fill=(0, 0, 0),
            )
            draw.text(
                (x + 6, y + tile_h - label_height + 4),
                f"{delta:+.3f} ({percent*100:+.0f}%)",
                fill=(255, 255, 0),
            )
    canvas.save(out_path)


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
    plt.ylabel("Total loss")
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
    out_dir: str = "out.reconstruct_mario_cfm"
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 1000
    steps_per_epoch: int = 100
    latent_dim: int = 1024
    lambda_recon: float = 1.0
    lambda_cfm: float = 0.1
    lambda_latent_l2: float = 0.0
    ms_weight: float = 1.0
    l1_weight: float = 0.1
    grad_clip: Optional[float] = 1.0
    max_trajs: Optional[int] = None
    encoder_base_channels: int = 64
    num_workers: int = 0
    device: Optional[str] = None
    log_every: int = 10
    viz_every: int = 50
    viz_samples: int = 4
    pca_batch_size: int = 64
    pca_sample_size: int = 2048
    pca_std_multiplier: float = 2.0
    pca_traverse_steps: int = 5
    seed: int = 42
    resume_checkpoint: Optional[str] = None
    checkpoint_every: int = 50  # 0 disables periodic last-checkpoint updates


def main() -> None:
    args = tyro.cli(Args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if args.pca_traverse_steps < 2:
        raise ValueError("pca_traverse_steps must be ≥ 2")
    device = pick_device(args.device)
    logger.info("Using device: %s", device)

    dataset = MarioFrameDataset(args.traj_dir, max_trajs=args.max_trajs)
    logger.info("Dataset loaded: %d frames", len(dataset))
    num_samples = args.steps_per_epoch * args.batch_size
    sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    sample_hw = dataset[0][0].shape[-2:]
    model = CFMAutoencoder(
        args.latent_dim,
        encoder_base_channels=args.encoder_base_channels,
        input_hw=sample_hw,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_run_dir = Path(args.out_dir) / f"run__{timestamp}"
    run_dir = default_run_dir

    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint).resolve()
        if resume_path.parent.name == "checkpoints":
            resume_run_dir = resume_path.parent.parent
        else:
            resume_run_dir = resume_path.parent
        if resume_run_dir.is_dir():
            run_dir = resume_run_dir

    samples_dir = run_dir / "samples"
    metrics_dir = run_dir / "metrics"
    checkpoints_dir = run_dir / "checkpoints"
    samples_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    loss_hist: List[Tuple[int, float]] = []
    start_epoch = 0
    global_step = 0
    best_metric = float("inf")

    if args.resume_checkpoint:
        ckpt_path = Path(args.resume_checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        logger.info("Resuming from checkpoint: %s", ckpt_path)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("step", 0))
        loss_hist = list(ckpt.get("loss_hist", []))
        best_metric = float(ckpt.get("best_metric", best_metric))

    if start_epoch >= args.epochs:
        logger.warning(
            "Start epoch %d is >= target epochs %d; nothing to train.", start_epoch, args.epochs
        )
        write_loss_csv(loss_hist, metrics_dir)
        torch.save({
            "epoch": start_epoch,
            "step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss_hist": loss_hist,
            "best_metric": best_metric,
        }, run_dir / "final.pt")
        return

    final_epoch = start_epoch
    checkpoint_last_path = checkpoints_dir / "checkpoint_last.pt"
    checkpoint_best_path = checkpoints_dir / "checkpoint_best.pt"

    def save_checkpoint(path: Path, epoch_val: int, step_val: int, best_val: float) -> None:
        payload = {
            "epoch": epoch_val,
            "step": step_val,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss_hist": loss_hist,
            "best_metric": best_val,
        }
        torch.save(payload, path)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        for xb, _ in loader:
            xb = xb.to(device)
            recon, latent = model(xb)

            ms = ms_ssim_loss(recon, xb)
            l1 = F.l1_loss(recon, xb)
            recon_loss = args.ms_weight * ms + args.l1_weight * l1

            z_base = torch.randn_like(latent)
            t = torch.rand(latent.shape[0], device=device)
            z_t = (1.0 - t)[:, None] * z_base + t[:, None] * latent
            v_pred = model.vector_field(z_t, t, latent, z_base)
            v_target = latent - z_base
            cfm_loss = F.mse_loss(v_pred, v_target)

            latent_reg = latent.pow(2).mean()
            total_loss = (
                args.lambda_recon * recon_loss
                + args.lambda_cfm * cfm_loss
                + args.lambda_latent_l2 * latent_reg
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            global_step += 1
            total_value = float(total_loss.item())
            loss_hist.append((global_step, total_value))

            if global_step % args.log_every == 0:
                stats = latent_summary(latent)
                logger.info(
                    "[epoch %03d | step %06d] loss=%.4f recon=%.4f cfm=%.4f ms=%.4f l1=%.4f "
                    "latent_std=%.3f",
                    epoch,
                    global_step,
                    total_value,
                    float(recon_loss.item()),
                    float(cfm_loss.item()),
                    float(ms.item()),
                    float(l1.item()),
                    stats["std"],
                )

            if global_step % args.viz_every == 0:
                model.eval()
                with torch.no_grad():
                    sample_indices = random.sample(
                        range(len(dataset)), k=min(args.viz_samples, len(dataset))
                    )
                    batch = torch.stack([dataset[i][0] for i in sample_indices], dim=0)
                    recon_batch, latent_batch = model(batch.to(device))
                recon_batch = recon_batch.cpu()
                latent_batch = latent_batch.cpu()

                latents_for_pca = collect_latents(
                    model,
                    dataset,
                    device,
                    max_samples=args.pca_sample_size,
                    batch_size=args.pca_batch_size,
                )
                directions, stds, proj_samples = compute_pca_components(
                    latents_for_pca, n_components=5
                )
                if not directions:
                    logger.warning("Skipping PCA traversal: no directions available.")
                else:
                    steps = args.pca_traverse_steps
                    if steps % 2 == 0:
                        steps += 1
                    max_percent = math.erf(
                        abs(float(args.pca_std_multiplier)) / math.sqrt(2.0)
                    )
                    if max_percent <= 0.0:
                        max_percent = 1.0
                    max_percent = min(max_percent, 1.0)
                    percent_levels = torch.linspace(
                        -max_percent,
                        max_percent,
                        steps=steps,
                    )
                    for idx in range(batch.shape[0]):
                        out_path = samples_dir / (
                            f"pca_step_{global_step:06d}_idx_{idx}.png"
                        )
                        save_pca_traversal_grid(
                            model,
                            batch[idx],
                            recon_batch[idx],
                            latent_batch[idx],
                            directions,
                            stds,
                             percent_levels.tolist(),
                             proj_samples,
                            out_path,
                            device,
                        )
                plot_loss(loss_hist, metrics_dir, global_step)
                model.train()

            updated_best = False
            if total_value < best_metric:
                best_metric = total_value
                save_checkpoint(checkpoint_best_path, epoch, global_step, best_metric)
                updated_best = True

            save_last = updated_best or (
                args.checkpoint_every > 0
                and global_step % args.checkpoint_every == 0
            )
            if save_last:
                save_checkpoint(checkpoint_last_path, epoch, global_step, best_metric)

        final_epoch = epoch
        # Recreate sampler each epoch to avoid exhaustion with replacement=False fallback.
        sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=False,
        )

    write_loss_csv(loss_hist, metrics_dir)
    save_checkpoint(run_dir / "final.pt", final_epoch, global_step, best_metric)
    logger.info("Training finished. Artifacts written to %s", run_dir)


if __name__ == "__main__":
    main()
