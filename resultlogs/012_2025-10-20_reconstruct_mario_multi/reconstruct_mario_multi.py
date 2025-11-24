#!/usr/bin/env python3
"""Slot-based Mario frame reconstruction with learned object masks.

This script trains a slot attention autoencoder on NES Mario frames. The
encoder observes a sequence of frames to stabilise slot discovery, but the
slot-conditioned decoder reconstructs each frame independently. Each slot
produces an RGBA canvas that is alpha-composited to obtain the final image,
which naturally yields a segmentation mask per slot.
"""
from __future__ import annotations

import csv
import logging
import math
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
    mean = INV_MEAN.tolist()
    std = INV_STD.tolist()
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


class MarioSequentialDataset(Dataset):
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
        return torch.stack(frames, dim=0)


# -----------------------------------------------------------------------------
# Slot Attention modules
# -----------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_size: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, hidden_size, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_size),
            nn.SiLU(),
            nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1),
            nn.GroupNorm(8, hidden_size),
            nn.SiLU(),
        )
        self.pos_embed = PositionalEmbedding(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        feat = self.pos_embed(feat)
        B, C, H, W = feat.shape
        return feat.view(B, C, H * W).permute(0, 2, 1)  # (B, N, C)


class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(4, dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        ys = torch.linspace(-1, 1, steps=H, device=feat.device, dtype=feat.dtype)
        xs = torch.linspace(-1, 1, steps=W, device=feat.device, dtype=feat.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx, yy, xx**2, yy**2], dim=0)
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)
        coords = self.linear(coords.permute(0, 2, 3, 1))
        return feat + coords.permute(0, 3, 1, 2)


class SlotAttention(nn.Module):
    def __init__(self, dim: int, num_slots: int, iters: int = 3,
                 slot_dim: int = 128, mlp_hidden: int = 256) -> None:
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.iters = iters
        self.scale = (slot_dim) ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, slot_dim))

        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(dim, slot_dim, bias=False)
        self.to_v = nn.Linear(dim, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, slot_dim),
        )
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        B, N, D = inputs.shape
        inputs = self.norm_inputs(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = F.softplus(self.slots_sigma) + 1e-6
        slots = mu + sigma * torch.randn_like(mu)

        for _ in range(self.iters):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots))
            attn_logits = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = attn_logits.softmax(dim=1)  # slots attend to inputs
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            updates = torch.einsum('bjd,bij->bid', v, attn)
            slots = self.gru(updates.reshape(-1, updates.shape[-1]),
                             slots_prev.reshape(-1, slots_prev.shape[-1]))
            slots = slots.reshape(B, self.num_slots, -1)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots


class SpatialBroadcastDecoder(nn.Module):
    def __init__(self, slot_dim: int, hidden_channels: int = 128,
                 out_hw: Tuple[int, int] = (224, 240),
                 base_hw: Optional[Tuple[int, int]] = None) -> None:
        super().__init__()
        self.out_hw = out_hw
        if base_hw is None:
            self.base_hw = (max(8, out_hw[0] // 8), max(8, out_hw[1] // 8))
        else:
            self.base_hw = base_hw

        in_ch = slot_dim + 2
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.GroupNorm(8, hidden_channels // 2),
            nn.SiLU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(hidden_channels // 2, hidden_channels // 2, 3, padding=1),
            nn.GroupNorm(8, hidden_channels // 2),
            nn.SiLU(),
        )
        self.final_conv = nn.Conv2d(hidden_channels // 2, 4, 1)

    def forward(self, slots: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, D = slots.shape
        H, W = self.out_hw
        h0, w0 = self.base_hw
        yy = torch.linspace(-1, 1, h0, device=slots.device, dtype=slots.dtype)
        xx = torch.linspace(-1, 1, w0, device=slots.device, dtype=slots.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        coord = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).unsqueeze(0)
        coord = coord.expand(B, S, -1, -1, -1)

        slots = slots.view(B, S, D, 1, 1).expand(-1, -1, -1, h0, w0)
        x = torch.cat([slots, coord], dim=2).view(B * S, D + 2, h0, w0)
        x = self.input_proj(x)
        x = self.block1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.block2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.block3(x)
        if x.shape[-2] < H or x.shape[-1] < W:
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        else:
            x = x[:, :, :H, :W]
        decoded = self.final_conv(x).view(B, S, 4, H, W)
        rgb = torch.tanh(decoded[:, :, :3])
        alpha = decoded[:, :, 3:4]
        attn = (alpha / max(temperature, 1e-3)).softmax(dim=1)
        recon = (rgb * attn).sum(dim=1)
        return recon, attn, rgb


class MarioSlotReconstructor(nn.Module):
    def __init__(self, num_slots: int = 6, slot_dim: int = 128,
                 encoder_hidden: int = 128, attn_iters: int = 2,
                 decoder_channels: int = 64) -> None:
        super().__init__()
        self.encoder = Encoder(3, encoder_hidden)
        self.slot_attention = SlotAttention(encoder_hidden, num_slots,
                                            iters=attn_iters, slot_dim=slot_dim,
                                            mlp_hidden=slot_dim * 2)
        self.decoder = SpatialBroadcastDecoder(slot_dim, decoder_channels)
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.img_hw = (224, 240)

    def forward(self, frames: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if frames.dim() != 5:
            raise ValueError("Expected frames with shape (B,T,3,H,W)")
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)
        feats = self.encoder(flat)
        slots = self.slot_attention(feats)
        recon, attn, rgb = self.decoder(slots, temperature=temperature)
        recon = recon.view(B, T, 3, H, W)
        attn = attn.view(B, T, self.num_slots, H, W)
        rgb = rgb.view(B, T, self.num_slots, 3, H, W)
        slots = slots.view(B, T, self.num_slots, self.slot_dim)
        return recon, attn, rgb, slots

    def decode_from_slots(self, slots: torch.Tensor) -> torch.Tensor:
        B, T, S, D = slots.shape
        recon, _, _ = self.decoder(slots.view(B * T, S, D))
        return recon.view(B, T, 3, *self.img_hw)


# -----------------------------------------------------------------------------
# Visualisation helpers
# -----------------------------------------------------------------------------

def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    if img.dim() != 3:
        raise ValueError("Expected (3,H,W) tensor")
    img = img.unsqueeze(0)
    img = unnormalize(img)[0].clamp(0, 1)
    arr = (img.mul(255).round().byte().permute(1, 2, 0).cpu().numpy())
    return Image.fromarray(arr)


def segmentation_to_pil(weights: torch.Tensor) -> Image.Image:
    if weights.dim() != 3:
        raise ValueError("Expected (S,H,W)")
    seg_idx = weights.argmax(dim=0).cpu().numpy().astype(np.int32)
    cmap = plt.get_cmap("tab20")
    colors = (np.array([cmap(i % cmap.N)[:3] for i in range(weights.shape[0])]) * 255).astype(np.uint8)
    return Image.fromarray(colors[seg_idx])


def slot_rgb_to_pil(rgb: torch.Tensor) -> Image.Image:
    arr = rgb.detach().cpu().clamp(-1, 1)
    arr = ((arr + 1) * 0.5 * 255).permute(1, 2, 0).byte().numpy()
    return Image.fromarray(arr)


def save_samples(frames: torch.Tensor, recon: torch.Tensor, attn: torch.Tensor,
                 slot_rgb: torch.Tensor, out_dir: Path, step: int,
                 max_items: int = 3, slots_per_row: int = 4) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    B, T = frames.shape[0], frames.shape[1]
    rows = min(B, max_items)
    for i in range(rows):
        panes: List[Image.Image] = []
        for t in range(T):
            orig = tensor_to_pil(frames[i, t])
            rec = tensor_to_pil(recon[i, t])
            mask = segmentation_to_pil(attn[i, t])
            panes.extend([orig, rec, mask])
            slot_weights = attn[i, t].view(attn.shape[2], -1).mean(dim=1)
            topk = torch.topk(slot_weights, k=min(slots_per_row, attn.shape[2])).indices
            for idx in topk:
                panes.append(slot_rgb_to_pil(slot_rgb[i, t, idx]))
        w, h = panes[0].size
        cols = 3 + min(slots_per_row, attn.shape[2])
        canvas = Image.new("RGB", (cols * w, T * h))
        for idx, pane in enumerate(panes):
            col = idx % cols
            row = idx // cols
            canvas.paste(pane, (col * w, row * h))
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
# Training
# -----------------------------------------------------------------------------

@dataclass
class Args:
    traj_dir: str = "data.image_distance.train_levels_1_2"
    out_dir: str = "out.reconstruct_mario_slots"
    batch_size: int = 4
    lr: float = 1e-4
    epochs: int = 1000
    steps_per_epoch: int = 100
    num_frames: int = 2
    max_trajs: Optional[int] = None
    save_every: int = 50
    log_every: int = 10
    log_debug_every: int = 50
    num_workers: int = 0
    device: Optional[str] = None
    ms_weight: float = 1.0
    l1_weight: float = 0.1
    slot_count: int = 6
    slot_dim: int = 128
    slot_iters: int = 2
    decoder_channels: int = 64
    slot_entropy_weight: float = 0.0
    slot_temperature: float = 1.0
    slot_temperature_min: float = 0.3
    slot_temperature_decay: float = 5e-4


def latent_summary(slots: torch.Tensor) -> dict[str, float]:
    lat = slots.detach()
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

    model = MarioSlotReconstructor(args.slot_count, args.slot_dim,
                                   encoder_hidden=args.slot_dim,
                                   attn_iters=args.slot_iters,
                                   decoder_channels=args.decoder_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    run_dir = Path(args.out_dir) / f"run__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    loss_hist: List[Tuple[int, float]] = []
    global_step = 0
    start_time = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb in loader:
            xb = xb.to(device)
            current_temp = max(
                args.slot_temperature * math.exp(-args.slot_temperature_decay * max(global_step, 0)),
                args.slot_temperature_min,
            )
            recon, attn, slot_rgb, slots = model(xb, temperature=current_temp)

            ms_terms = [ms_ssim_loss(recon[:, t], xb[:, t]) for t in range(xb.shape[1])]
            l1_terms = [F.l1_loss(recon[:, t], xb[:, t]) for t in range(xb.shape[1])]
            ms_loss = torch.stack(ms_terms).mean()
            l1_loss = torch.stack(l1_terms).mean()
            loss = args.ms_weight * ms_loss + args.l1_weight * l1_loss

            slot_entropy = torch.tensor(0.0, device=device)
            if args.slot_entropy_weight > 0.0:
                pixel_entropy = -(attn.clamp_min(1e-8).log() * attn).sum(dim=2)
                slot_entropy = pixel_entropy.mean()
                loss = loss + args.slot_entropy_weight * slot_entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            proto_grad = float(0.0)
            decoder_grads: List[torch.Tensor] = []
            for param in model.decoder.parameters():
                if param.grad is not None:
                    decoder_grads.append(param.grad.detach().abs().mean())
            if decoder_grads:
                proto_grad = float(torch.stack(decoder_grads).mean().item())
            optimizer.step()

            global_step += 1
            loss_hist.append((global_step, float(loss.item())))

            if args.log_every > 0 and global_step % args.log_every == 0:
                elapsed = (time.monotonic() - start_time) / 60
                logger.info(
                    "[ep %03d] step %06d | loss=%.4f | ms=%.4f | l1=%.4f | slot_ent=%.4f | temp=%.3f | grad=%.6e | elapsed=%.2f min",
                    epoch,
                    global_step,
                    loss.item(),
                    ms_loss.item(),
                    l1_loss.item(),
                    slot_entropy.item() if isinstance(slot_entropy, torch.Tensor) else float(slot_entropy),
                    current_temp,
                    proto_grad,
                    elapsed,
                )

            if args.log_debug_every > 0 and global_step % args.log_debug_every == 0:
                stats = latent_summary(slots)
                attn_mass = attn.mean(dim=(0, 1, 3, 4))
                logger.info(
                    "[step %06d] slots | mean=%.4f std=%.4f min=%.4f max=%.4f | attn_mass=%s",
                    global_step,
                    stats["mean"],
                    stats["std"],
                    stats["min"],
                    stats["max"],
                    ",".join(f"{m:.3f}" for m in attn_mass.tolist()),
                )

            if args.save_every > 0 and global_step % args.save_every == 0:
                with torch.no_grad():
                    save_samples(xb.cpu(), recon.cpu(), attn.cpu(), slot_rgb.cpu(),
                                 samples_dir, global_step)
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
