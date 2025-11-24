#!/usr/bin/env python3
"""Conditional flow matching across Mario frame sequences.

The model observes four context frames and learns a conditional vector field
whose flow passes through those frames before rolling out future predictions.
Structure, logging, and visualisation mirror ``predict_mario_multi2.py`` where
practical, while the network remains a small pixel-space field trained via the
conditional flow matching objective on a piecewise-linear path.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import tyro
from torchvision.transforms import Normalize

from predict_mario_ms_ssim import pick_device, unnormalize
from recon.data import list_trajectories, load_frame_as_tensor, short_traj_state_label

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_latent_tensor(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalise over spatial/channel axes while preserving leading dims."""
    if tensor.dim() < 3:
        raise ValueError(f"Expected latent tensor with >=3 dims, got {tuple(tensor.shape)}")
    flat = tensor.flatten(start_dim=-3)
    norms = flat.norm(dim=-1, keepdim=True).clamp_min(eps)
    normalised = flat / norms
    return normalised.view_as(tensor)


class FrameEncoder(nn.Module):
    """Shared convolutional encoder yielding latent maps and optional skip features."""

    def __init__(self, base_channels: int = 32, latent_channels: int = 96) -> None:
        super().__init__()
        if base_channels % 8 != 0:
            raise ValueError("base_channels must be divisible by 8")
        groups = 8
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, base_channels),
            nn.SiLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(groups, base_channels * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(groups, base_channels * 2),
            nn.SiLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 3, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(groups, base_channels * 3),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels * 3, base_channels * 3, kernel_size=3, padding=1),
            nn.GroupNorm(groups, base_channels * 3),
            nn.SiLU(inplace=True),
        )
        groups_latent = 8 if latent_channels % 8 == 0 else 1
        self.down3 = nn.Sequential(
            nn.Conv2d(base_channels * 3, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(groups_latent, latent_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups_latent, latent_channels),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, frame: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        h0 = self.stem(frame)
        h1 = self.down1(h0)
        h2 = self.down2(h1)
        latent = self.down3(h2)
        pooled = self.pool(_normalize_latent_tensor(latent)).flatten(1)
        skips: Tuple[torch.Tensor, ...] = (h0, h1, h2)
        return latent, pooled, skips


class LatentDecoder(nn.Module):
    """Decoder mirroring FrameEncoder to reconstruct RGB frames from latents."""

    def __init__(self, latent_channels: int = 96, base_channels: int = 32) -> None:
        super().__init__()
        if base_channels % 8 != 0:
            raise ValueError("base_channels must be divisible by 8")
        groups = 8
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, base_channels * 3, kernel_size=2, stride=2),
            nn.GroupNorm(groups, base_channels * 3),
            nn.SiLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 3, base_channels * 2, kernel_size=2, stride=2),
            nn.GroupNorm(groups, base_channels * 2),
            nn.SiLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2),
            nn.GroupNorm(groups, base_channels),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels // 2, 3, kernel_size=1),
        )

    def forward(
        self,
        latents: torch.Tensor,
        skip_pyramids: Tuple[torch.Tensor, ...] = (),
        *,
        target_hw: Tuple[int, int],
    ) -> torch.Tensor:
        if latents.dim() != 4:
            raise ValueError("Expected latents shaped (B, C, H, W).")
        h = self.up1(latents)
        if skip_pyramids:
            skip2 = F.interpolate(skip_pyramids[2], size=h.shape[-2:], mode="bilinear", align_corners=False)
            h = h + skip2
        h = self.up2(h)
        if skip_pyramids:
            skip1 = F.interpolate(skip_pyramids[1], size=h.shape[-2:], mode="bilinear", align_corners=False)
            h = h + skip1
        h = self.up3(h)
        if skip_pyramids:
            skip0 = F.interpolate(skip_pyramids[0], size=h.shape[-2:], mode="bilinear", align_corners=False)
            h = h + skip0
        frame = self.head(h)
        if frame.shape[-2:] != target_hw:
            frame = F.interpolate(frame, size=target_hw, mode="bilinear", align_corners=False)
        return frame


LOSS_COLUMNS = ["step", "train_loss", "val_loss", "flow_loss", "align_loss", "pixel_loss"]


def write_loss_csv(history: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LOSS_COLUMNS)
        for entry in history:
            writer.writerow([entry.get(col, float("nan")) for col in LOSS_COLUMNS])


def plot_loss_curves(history: List[Dict[str, float]], out_dir: Path) -> None:
    if not history:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = [entry["step"] for entry in history]
    metrics = {
        "train_loss": [entry.get("train_loss", float("nan")) for entry in history],
        "val_loss": [entry.get("val_loss", float("nan")) for entry in history],
        "flow_loss": [entry.get("flow_loss", float("nan")) for entry in history],
        "align_loss": [entry.get("align_loss", float("nan")) for entry in history],
        "pixel_loss": [entry.get("pixel_loss", float("nan")) for entry in history],
    }

    plt.figure(figsize=(7, 4))
    for label, values in metrics.items():
        if any(np.isfinite(v) for v in values):
            plt.plot(steps, values, label=label)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Flow Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=200)
    plt.close()


def _tensor_to_image(img: torch.Tensor) -> np.ndarray:
    """Convert a (C, H, W) tensor in normalised space to a H×W×C numpy image."""
    if img.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got shape {tuple(img.shape)}")
    with torch.no_grad():
        data = unnormalize(img.unsqueeze(0)).clamp(0.0, 1.0).squeeze(0)
    return data.permute(1, 2, 0).cpu().numpy()


def _ensure_dirs(config: "TrainConfig") -> Dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.resume:
        existing_runs = sorted(
            [
                path
                for path in config.output_dir.iterdir()
                if path.is_dir() and path.name and path.name[0].isdigit()
            ],
            reverse=True,
        )
        run_dir = existing_runs[0] if existing_runs else config.output_dir / datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = config.output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    vis_dir = run_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    return {"run": run_dir, "checkpoints": ckpt_dir, "plots": plots_dir, "visualizations": vis_dir}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    encoder: nn.Module,
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    best_val: float,
    history: List[Dict[str, float]],
) -> None:
    payload = {
        "model": model.state_dict(),
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val": best_val,
        "history": history,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    encoder: nn.Module,
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[int, int, float, List[Dict[str, float]]]:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    if "encoder" in payload:
        encoder.load_state_dict(payload["encoder"])
    if "decoder" in payload:
        decoder.load_state_dict(payload["decoder"])
    optimizer.load_state_dict(payload["optimizer"])
    return (
        int(payload.get("epoch", 0)),
        int(payload.get("step", 0)),
        float(payload.get("best_val", math.inf)),
        list(payload.get("history", [])),
    )


# -----------------------------------------------------------------------------
# Data handling
# -----------------------------------------------------------------------------


IMAGENET_NORMALIZE = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class MarioSequenceCFMDataset(Dataset):
    """Return sliding windows of context+future frames for conditional flow training."""

    def __init__(
        self,
        root_dir: Path,
        *,
        image_hw: Tuple[int, int],
        context_len: int,
        future_len: int,
        normalize: Optional[Normalize] = None,
        max_trajs: Optional[int] = None,
    ) -> None:
        if context_len <= 0 or future_len <= 0:
            raise ValueError("context_len and future_len must be positive.")
        self.root_dir = Path(root_dir)
        self.image_hw = image_hw
        self.context_len = context_len
        self.future_len = future_len
        self.normalize = normalize
        self.entries: List[Tuple[List[Path], int]] = []

        traj_map = list_trajectories(self.root_dir)
        traj_items = list(traj_map.items())
        if max_trajs is not None:
            traj_items = traj_items[:max_trajs]
        window = context_len + future_len
        for traj_name, frames in traj_items:
            if len(frames) < window:
                continue
            for start in range(len(frames) - window + 1):
                self.entries.append((frames, start))
        if not self.entries:
            raise RuntimeError(f"No valid frame windows under {self.root_dir}")

    def __len__(self) -> int:
        return len(self.entries)

    def _load_frame(self, path: Path) -> torch.Tensor:
        tensor = load_frame_as_tensor(path, size=self.image_hw, normalize=None)
        if self.normalize is not None:
            tensor = self.normalize(tensor)
        return tensor

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        frames, start = self.entries[index]
        context_frames: List[torch.Tensor] = []
        future_frames: List[torch.Tensor] = []

        context_labels: List[str] = []
        future_labels: List[str] = []

        for i in range(self.context_len):
            path = frames[start + i]
            context_frames.append(self._load_frame(path))
            context_labels.append(short_traj_state_label(str(path)))
        for j in range(self.future_len):
            path = frames[start + self.context_len + j]
            future_frames.append(self._load_frame(path))
            future_labels.append(short_traj_state_label(str(path)))

        return {
            "context": torch.stack(context_frames, dim=0),
            "future": torch.stack(future_frames, dim=0),
            "context_labels": context_labels,
            "future_labels": future_labels,
        }


def prepare_dataloaders(config: "TrainConfig") -> Tuple[DataLoader, DataLoader, MarioSequenceCFMDataset]:
    dataset = MarioSequenceCFMDataset(
        config.data_root,
        image_hw=config.image_hw,
        context_len=config.context_len,
        future_len=config.future_len,
        normalize=IMAGENET_NORMALIZE,
        max_trajs=config.max_trajs,
    )
    val_len = max(1, int(len(dataset) * config.val_fraction))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(config.seed))

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader, dataset


def sample_path_state(
    frames: torch.Tensor,
    times: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample path state, velocity, and segment index for piecewise-linear paths."""
    if frames.dim() != 5:
        raise ValueError("frames must be shaped (B, T, C, H, W).")
    batch, total_nodes, _, _, _ = frames.shape
    if total_nodes < 2:
        raise ValueError("Need at least two frames to define a path.")
    if times.shape[0] != batch:
        raise ValueError("Batch size mismatch between frames and times.")

    total_segments = total_nodes - 1
    scaled = times * total_segments
    segment_idx = torch.clamp(scaled.long(), min=0, max=total_segments - 1)
    local_tau = scaled - segment_idx.float()
    batch_indices = torch.arange(batch, device=frames.device)
    frame_a = frames[batch_indices, segment_idx]
    frame_b = frames[batch_indices, segment_idx + 1]
    x_t = (1.0 - local_tau)[:, None, None, None] * frame_a + local_tau[:, None, None, None] * frame_b
    velocity = (frame_b - frame_a) * total_segments
    return x_t, velocity, segment_idx


def gather_window_frames(
    frames: torch.Tensor,
    segment_idx: torch.Tensor,
    window_len: int,
) -> torch.Tensor:
    """Collect the latest `window_len` frames preceding each segment index."""
    if frames.dim() != 5:
        raise ValueError("frames must be shaped (B, T, C, H, W).")
    batch, total_nodes, channels, height, width = frames.shape
    if window_len <= 0:
        raise ValueError("window_len must be positive.")
    if segment_idx.shape[0] != batch:
        raise ValueError("Batch size mismatch for segment indices.")

    device = frames.device
    steps = torch.arange(window_len, device=device, dtype=segment_idx.dtype)
    idx = segment_idx.unsqueeze(1) - (window_len - 1 - steps).unsqueeze(0)
    idx = idx.clamp(min=0, max=total_nodes - 1)
    batch_indices = torch.arange(batch, device=device).unsqueeze(1).expand_as(idx)
    window = frames[batch_indices, idx]
    return window.reshape(batch, window_len, channels, height, width)


def encode_sequence(
    encoder: FrameEncoder,
    frames: torch.Tensor,
    *,
    return_skips: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Encode a (B, T, 3, H, W) tensor into latent maps shaped (B, T, C, h, w).

    When ``return_skips`` is True, also return the encoder skip activations
    reshaped to (B, T, C_i, h_i, w_i) tuples matching the decoder expectations.
    """
    if frames.dim() != 5:
        raise ValueError("Expected frames shaped (B, T, 3, H, W).")
    batch, steps, _, height, width = frames.shape
    flat = frames.view(batch * steps, 3, height, width)
    latents, _, skips = encoder(flat)
    latent_shape = latents.shape[1:]
    latents = latents.view(batch, steps, *latent_shape)
    if not return_skips:
        return latents
    skip_maps: Tuple[torch.Tensor, ...] = tuple(
        skip.view(batch, steps, *skip.shape[1:]) for skip in skips
    )
    return latents, skip_maps


def decode_sequence(
    decoder: LatentDecoder,
    latents: torch.Tensor,
    *,
    image_hw: Tuple[int, int],
    skip_pyramids: Optional[Tuple[torch.Tensor, ...]] = None,
) -> torch.Tensor:
    """Decode (B, T, C, h, w) latents back to normalised RGB frames."""
    if latents.dim() != 5:
        raise ValueError("Expected latents shaped (B, T, C, h, w).")
    batch, steps, channels, height, width = latents.shape
    flat = latents.view(batch * steps, channels, height, width)
    skip_flat: Tuple[torch.Tensor, ...] = ()
    if skip_pyramids is not None:
        skip_flat = tuple(
            skip.view(batch * steps, *skip.shape[2:]) for skip in skip_pyramids
        )
    frames = decoder(flat, skip_flat, target_hw=image_hw)
    return frames.view(batch, steps, 3, *image_hw)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class LatentCFMField(nn.Module):
    """Latent-space conditional flow field conditioned on sliding window latents."""

    def __init__(
        self,
        context_len: int,
        latent_channels: int,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        if context_len <= 0:
            raise ValueError("context_len must be positive.")
        if latent_channels <= 0:
            raise ValueError("latent_channels must be positive.")
        if hidden_channels % 8 != 0:
            raise ValueError("hidden_channels must be divisible by 8 for GroupNorm.")
        self.context_len = context_len
        in_channels = latent_channels + context_len * latent_channels + 1
        groups = 8 if hidden_channels % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, context_flat: torch.Tensor) -> torch.Tensor:
        if x_t.dim() != 4 or context_flat.dim() != 4:
            raise ValueError("Expected tensors shaped (B, C, H, W).")
        if x_t.shape[0] != context_flat.shape[0]:
            raise ValueError("Batch size mismatch between x_t and context.")
        if x_t.shape[-2:] != context_flat.shape[-2:]:
            raise ValueError("Spatial shape mismatch between x_t and context.")
        batch, _, height, width = x_t.shape
        if t.shape[0] != batch:
            raise ValueError("Time tensor batch dimension mismatch.")
        t_img = t[:, None, None, None].expand(batch, 1, height, width)
        cond = torch.cat([x_t, context_flat, t_img], dim=1)
        return self.net(cond)


def rollout_future_latents(
    model: LatentCFMField,
    context_latents: torch.Tensor,
    *,
    future_len: int,
    flow_steps: int,
) -> torch.Tensor:
    """Integrate the learned field through context transitions and future rollouts."""
    if future_len <= 0:
        raise ValueError("future_len must be positive.")
    if flow_steps <= 0:
        raise ValueError("flow_steps must be positive.")

    model.eval()
    device = context_latents.device
    batch, ctx_len, channels, height, width = context_latents.shape
    total_nodes = ctx_len + future_len
    total_segments = total_nodes - 1
    if total_segments <= 0:
        raise ValueError("Not enough nodes to define a path.")
    base_steps = max(1, flow_steps // total_segments)
    remainder = flow_steps - base_steps * total_segments

    window = context_latents.clone()
    predictions: List[torch.Tensor] = []

    with torch.no_grad():
        for segment in range(total_segments):
            segment_steps = base_steps + (1 if segment < remainder else 0)
            start_time = segment / total_segments
            end_time = (segment + 1) / total_segments
            cond_flat = window.view(batch, ctx_len * channels, height, width)

            if segment < ctx_len - 1:
                state = context_latents[:, segment]
            elif segment == ctx_len - 1:
                state = window[:, -1]
            elif predictions:
                state = predictions[-1]
            else:
                state = window[:, -1]

            times = torch.linspace(
                start_time,
                end_time,
                segment_steps + 1,
                device=device,
                dtype=context_latents.dtype,
            )
            for t0, t1 in zip(times[:-1], times[1:]):
                t = torch.full((batch,), float(t0.item()), device=device, dtype=context_latents.dtype)
                v = model(state, t, cond_flat)
                dt = float(t1.item() - t0.item())
                state = state + dt * v

            if segment < ctx_len - 1:
                continue

            predictions.append(state)
            window = torch.cat([window[:, 1:], state.unsqueeze(1)], dim=1)

    if not predictions:
        return context_latents.new_zeros(batch, 0, channels, height, width)
    return torch.stack(predictions, dim=1)


# -----------------------------------------------------------------------------
# Visualisation
# -----------------------------------------------------------------------------


def render_visualization_grid(
    model: LatentCFMField,
    encoder: FrameEncoder,
    decoder: LatentDecoder,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    output_path: Path,
    *,
    integration_steps: int,
    image_hw: Tuple[int, int],
) -> None:
    model.eval()
    encoder.eval()
    decoder.eval()
    context = batch["context"][0].unsqueeze(0).to(device)
    future = batch["future"][0].unsqueeze(0).to(device)
    context_labels = batch.get("context_labels", None)
    future_labels = batch.get("future_labels", None)
    if context_labels is not None and isinstance(context_labels, (list, tuple)):
        context_labels = [lbl[0] if isinstance(lbl, (list, tuple)) else lbl for lbl in context_labels]
    if future_labels is not None and isinstance(future_labels, (list, tuple)):
        future_labels = [lbl[0] if isinstance(lbl, (list, tuple)) else lbl for lbl in future_labels]

    with torch.no_grad():
        context_latents = encode_sequence(encoder, context)
        predictions_latents = rollout_future_latents(
            model,
            context_latents,
            future_len=future.shape[1],
            flow_steps=integration_steps,
        )
        if predictions_latents.size(1) > 0:
            pred_frames = decode_sequence(
                decoder,
                predictions_latents,
                image_hw=image_hw,
            )
        else:
            pred_frames = future.new_zeros((future.shape[0], 0, 3, *image_hw))

    context_np = context.squeeze(0).cpu()
    future_np = future.squeeze(0).cpu()
    pred_np = pred_frames.squeeze(0).cpu()
    # Align prediction length with available futures.
    future_steps = future_np.shape[0]
    pred_steps = pred_np.shape[0]
    usable_steps = min(future_steps, pred_steps)

    abs_error = torch.zeros(usable_steps, image_hw[0], image_hw[1])
    if usable_steps > 0:
        abs_error = torch.abs(pred_np[:usable_steps] - future_np[:usable_steps]).mean(dim=1)
    error_max = float(abs_error.max().item()) if usable_steps > 0 else 1e-6
    if error_max <= 0.0:
        error_max = 1e-6

    total_cols = context_np.shape[0] + future_steps
    blank_rgb = np.zeros((*image_hw, 3), dtype=np.float32)

    fig, axes = plt.subplots(3, total_cols, figsize=(max(12, total_cols * 2), 6))
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    # Row 1: context frames followed by ground-truth future frames.
    for idx in range(context_np.shape[0]):
        axes[0, idx].imshow(_tensor_to_image(context_np[idx]))
        if context_labels is not None and idx < len(context_labels):
            axes[0, idx].set_title(str(context_labels[idx]), fontsize=6)
    for j in range(future_steps):
        col = context_np.shape[0] + j
        axes[0, col].imshow(_tensor_to_image(future_np[j]))
        if future_labels is not None and j < len(future_labels):
            axes[0, col].set_title(str(future_labels[j]), fontsize=6)

    # Row 2: blanks under context, predictions under future columns.
    for idx in range(context_np.shape[0]):
        axes[1, idx].imshow(blank_rgb)
    for j in range(future_steps):
        col = context_np.shape[0] + j
        if j < pred_steps:
            axes[1, col].imshow(_tensor_to_image(pred_np[j]))
        else:
            axes[1, col].imshow(blank_rgb)

    # Row 3: blanks under context, error heatmaps for predicted steps.
    cmap = plt.get_cmap("magma")
    for idx in range(context_np.shape[0]):
        axes[2, idx].imshow(blank_rgb)
    for j in range(future_steps):
        col = context_np.shape[0] + j
        if j < usable_steps:
            err = abs_error[j].cpu().numpy()
            im = axes[2, col].imshow(err, cmap=cmap, vmin=0.0, vmax=error_max)
        else:
            axes[2, col].imshow(np.zeros(image_hw, dtype=np.float32), cmap=cmap, vmin=0.0, vmax=error_max)

    if usable_steps > 0:
        fig.colorbar(im, ax=axes[2, context_np.shape[0]:], fraction=0.025, pad=0.04)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


@dataclass
class TrainConfig:
    data_root: Path = Path("data.image_distance.train_levels_1_2")
    output_dir: Path = Path("out.predict_mario_cfm")
    epochs: int = 1000
    batch_size: int = 12
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    val_fraction: float = 0.1
    seed: int = 42
    device: str = "auto"
    mixed_precision: bool = False
    context_len: int = 4
    future_len: int = 4
    flow_steps: int = 64
    hidden_channels: int = 64
    base_channels: int = 32
    latent_channels: int = 96
    lambda_align: float = 1.0
    lambda_pixel: float = 0.1
    decoder_lr: float = 3e-4
    decoder_weight_decay: float = 1e-4
    lambda_pixel: float = 0.1
    log_every: int = 50
    vis_every: int = 50
    checkpoint_every: int = 200
    resume: bool = False
    max_trajs: Optional[int] = None
    image_hw: Tuple[int, int] = (224, 224)

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.output_dir = Path(self.output_dir)
        if self.context_len <= 0:
            raise ValueError("context_len must be positive.")
        if self.future_len <= 0:
            raise ValueError("future_len must be positive.")
        if self.hidden_channels % 8 != 0:
            raise ValueError("hidden_channels must be divisible by 8.")
        if self.base_channels % 8 != 0:
            raise ValueError("base_channels must be divisible by 8.")
        if self.latent_channels % 8 != 0:
            raise ValueError("latent_channels must be divisible by 8.")
        if self.lambda_align < 0.0:
            raise ValueError("lambda_align must be non-negative.")
        if self.lambda_pixel < 0.0:
            raise ValueError("lambda_pixel must be non-negative.")
        if self.decoder_lr <= 0.0:
            raise ValueError("decoder_lr must be positive.")
        if self.decoder_weight_decay < 0.0:
            raise ValueError("decoder_weight_decay must be non-negative.")


def evaluate(
    model: LatentCFMField,
    encoder: FrameEncoder,
    loader: DataLoader,
    device: torch.device,
    *,
    context_len: int,
) -> float:
    model.eval()
    encoder.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in loader:
            context = batch["context"].to(device)
            future = batch["future"].to(device)
            context_latents = encode_sequence(encoder, context)
            future_latents = encode_sequence(encoder, future)
            latent_frames = torch.cat([context_latents, future_latents], dim=1)
            times = torch.rand(context.shape[0], device=device, dtype=latent_frames.dtype)
            x_t, velocity_target, segment_idx = sample_path_state(latent_frames, times)
            window = gather_window_frames(latent_frames, segment_idx, context_len)
            latent_channels = latent_frames.shape[2]
            cond_flat = window.view(
                context.shape[0],
                context_len * latent_channels,
                window.shape[-2],
                window.shape[-1],
            )
            pred = model(x_t, times, cond_flat)
            loss = F.mse_loss(pred, velocity_target)
            total_loss += float(loss.item())
            total_batches += 1
    if total_batches == 0:
        return float("nan")
    return total_loss / total_batches


def train(config: TrainConfig) -> None:
    set_seed(config.seed)
    dirs = _ensure_dirs(config)
    run_dir = dirs["run"]
    logger.info("Writing outputs to %s", run_dir)
    train_loader, val_loader, full_dataset = prepare_dataloaders(config)
    device_pref = None if str(config.device).lower() == "auto" else config.device
    device = pick_device(device_pref)
    logger.info("Using device %s", device)
    logger.info(
        "Dataset size: %d samples (train %d | val %d)",
        len(full_dataset),
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    encoder = FrameEncoder(
        base_channels=config.base_channels,
        latent_channels=config.latent_channels,
    ).to(device)
    decoder = LatentDecoder(
        latent_channels=config.latent_channels,
        base_channels=config.base_channels,
    ).to(device)
    model = LatentCFMField(
        context_len=config.context_len,
        latent_channels=config.latent_channels,
        hidden_channels=config.hidden_channels,
    ).to(device)
    params: List[nn.Parameter] = list(model.parameters()) + list(encoder.parameters())
    if config.lambda_align > 0.0 or config.lambda_pixel > 0.0:
        params += list(decoder.parameters())
    else:
        decoder.requires_grad_(False)
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    decoder_optimizer: Optional[torch.optim.Optimizer] = None
    if config.lambda_pixel > 0.0:
        decoder_optimizer = torch.optim.AdamW(
            decoder.parameters(),
            lr=config.decoder_lr,
            weight_decay=config.decoder_weight_decay,
        )

    start_epoch = 0
    global_step = 0
    best_val = math.inf
    history: List[Dict[str, float]] = []
    if config.resume:
        last_ckpt = dirs["checkpoints"] / "last.pt"
        if last_ckpt.exists():
            logger.info("Resuming from %s", last_ckpt)
            start_epoch, global_step, best_val, history = load_checkpoint(
                last_ckpt,
                model,
                encoder,
                decoder,
                optimizer,
            )
            start_epoch += 1
            model.to(device)
            encoder.to(device)
            decoder.to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)

    for epoch in range(start_epoch, config.epochs):
        model.train()
        encoder.train()
        if config.lambda_align > 0.0 or config.lambda_pixel > 0.0:
            decoder.train()
        running_loss = 0.0
        running_flow = 0.0
        running_align = 0.0
        running_pixel = 0.0
        for batch_idx, batch in enumerate(train_loader):
            context = batch["context"].to(device)
            future = batch["future"].to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=config.mixed_precision):
                context_latents, context_skips = encode_sequence(
                    encoder, context, return_skips=True
                )
                context_latents_detached = context_latents.detach()
                context_skips_detached: Tuple[torch.Tensor, ...] = tuple(
                    skip.detach() for skip in context_skips
                )
                future_latents = encode_sequence(encoder, future)
                latent_frames = torch.cat([context_latents, future_latents], dim=1)
                times = torch.rand(context.shape[0], device=device, dtype=latent_frames.dtype)
                x_t, velocity_target, segment_idx = sample_path_state(latent_frames, times)
                window = gather_window_frames(latent_frames, segment_idx, config.context_len)
                latent_channels = latent_frames.shape[2]
                cond_flat = window.view(
                    context.shape[0],
                    config.context_len * latent_channels,
                    window.shape[-2],
                    window.shape[-1],
                )
                pred = model(x_t, times, cond_flat)
                flow_loss = F.mse_loss(pred, velocity_target)
                align_loss = flow_loss.new_zeros(())
                if config.lambda_align > 0.0:
                    recon_frames = decode_sequence(
                        decoder,
                        context_latents,
                        image_hw=config.image_hw,
                        skip_pyramids=context_skips,
                    )
                    reencoded_latents = encode_sequence(encoder, recon_frames)
                    align_loss = F.mse_loss(reencoded_latents, context_latents.detach())
                loss = flow_loss + config.lambda_align * align_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pixel_loss = flow_loss.new_zeros(())
            if config.lambda_pixel > 0.0 and decoder_optimizer is not None:
                decoder_optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=config.mixed_precision):
                    recon_frames_pixel = decode_sequence(
                        decoder,
                        context_latents_detached,
                        image_hw=config.image_hw,
                        skip_pyramids=context_skips_detached,
                    )
                    pixel_loss = F.mse_loss(recon_frames_pixel, context)
                    pixel_total = config.lambda_pixel * pixel_loss
                pixel_total.backward()
                decoder_optimizer.step()

            running_loss += float(loss.item())
            running_flow += float(flow_loss.item())
            running_align += float(align_loss.item())
            running_pixel += float(pixel_loss.item())
            current_step = global_step + 1
            history.append(
                {
                    "step": current_step,
                    "train_loss": float(loss.item()),
                    "val_loss": float("nan"),
                    "flow_loss": float(flow_loss.item()),
                    "align_loss": float(align_loss.item()),
                    "pixel_loss": float(pixel_loss.item()),
                }
            )
            if config.log_every > 0 and current_step % config.log_every == 0:
                logger.info(
                    "Epoch %d Step %d | total %.6f flow %.6f align %.6f pixel %.6f",
                    epoch,
                    current_step,
                    loss.item(),
                    flow_loss.item(),
                    align_loss.item(),
                    pixel_loss.item(),
                )

            if config.checkpoint_every > 0 and current_step % config.checkpoint_every == 0:
                save_checkpoint(
                    dirs["checkpoints"] / "last.pt",
                    model,
                    encoder,
                    decoder,
                    optimizer,
                    epoch,
                    current_step,
                    best_val,
                    history,
                )
            if config.vis_every > 0 and current_step % config.vis_every == 0:
                vis_path = dirs["visualizations"] / f"step_{current_step:06d}.png"
                render_visualization_grid(
                    model,
                    encoder,
                    decoder,
                    batch,
                    device,
                    vis_path,
                    integration_steps=config.flow_steps,
                    image_hw=config.image_hw,
                )
                plot_loss_curves(history, dirs["plots"])
                write_loss_csv(history, dirs["plots"] / "loss_history.csv")
                model.train()

            global_step = current_step

        steps_in_epoch = max(1, len(train_loader))
        avg_train_loss = running_loss / steps_in_epoch
        avg_flow_loss = running_flow / steps_in_epoch
        avg_align_loss = running_align / steps_in_epoch
        avg_pixel_loss = running_pixel / steps_in_epoch
        val_loss = evaluate(
            model,
            encoder,
            val_loader,
            device,
            context_len=config.context_len,
        )
        history.append(
            {
                "step": global_step,
                "train_loss": float("nan"),
                "val_loss": float(val_loss),
                "flow_loss": float("nan"),
                "align_loss": float("nan"),
                "pixel_loss": float("nan"),
            }
        )
        logger.info(
            "Epoch %d complete | avg_total %.6f avg_flow %.6f avg_align %.6f avg_pixel %.6f | val_flow %.6f",
            epoch,
            avg_train_loss,
            avg_flow_loss,
            avg_align_loss,
            avg_pixel_loss,
            val_loss,
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                dirs["checkpoints"] / "best.pt",
                model,
                encoder,
                decoder,
                optimizer,
                epoch,
                global_step,
                best_val,
                history,
            )
            logger.info("Saved new best checkpoint (val_loss %.6f)", best_val)

        save_checkpoint(
            dirs["checkpoints"] / "last.pt",
            model,
            encoder,
            decoder,
            optimizer,
            epoch,
            global_step,
            best_val,
            history,
        )

    save_checkpoint(
        dirs["checkpoints"] / "final.pt",
        model,
        encoder,
        decoder,
        optimizer,
        config.epochs - 1,
        global_step,
        best_val,
        history,
    )
    plot_loss_curves(history, dirs["plots"])
    write_loss_csv(history, dirs["plots"] / "loss_history.csv")
    payload = {key: str(value) if isinstance(value, Path) else value for key, value in vars(config).items()}
    payload["run_dir"] = str(run_dir)
    with (run_dir / "config.json").open("w") as fp:
        json.dump(payload, fp, indent=2)
    logger.info("Training complete.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = tyro.cli(TrainConfig)
    train(config)


if __name__ == "__main__":
    main()
