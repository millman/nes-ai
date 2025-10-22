#!/usr/bin/env python3
"""Temporal frame predictor with shared latent space.

This script trains a model that:
1. Encodes 4 sequential frames into a spatial latent grid
2. Reconstructs those frames from the latent grid
3. Predicts the next (5th) frame by learning latent dynamics

The decoder uses alpha-compositing to reveal pixel groupings, enabling
visualization of which pixels belong together (e.g., static objects).
The latent space supports PCA traversal and arbitrary latent manipulation.
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


class MarioTemporalDataset(Dataset):
    """Dataset that returns 5 sequential frames for temporal prediction."""

    def __init__(self, root_dir: str, sequence_len: int = 5,
                 transform: Optional[T.Compose] = None,
                 max_trajs: Optional[int] = None) -> None:
        if sequence_len < 5:
            raise ValueError("sequence_len must be >= 5 for temporal prediction")
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
        return torch.stack(frames, dim=0)  # (T, 3, H, W)


# -----------------------------------------------------------------------------
# Model Components
# -----------------------------------------------------------------------------

class TemporalEncoder(nn.Module):
    """Encodes 4 sequential frames into a spatial latent grid.

    Architecture: Frame-wise encoding followed by temporal fusion.
    Input: (B, 4, 3, H, W)
    Output: (B, D_lat, H_lat, W_lat)
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 128,
                 downsample_factor: int = 8) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.downsample_factor = downsample_factor

        # Frame-wise encoder (processes each frame independently)
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, stride=2, padding=2),  # /2
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),  # /4
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 5, stride=2, padding=2),  # /8
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )

        # Temporal fusion (combine 4 frames)
        # Input: (B, 4*256, H/8, W/8) → Output: (B, latent_dim, H/8, W/8)
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(256 * 4, 512, 3, padding=1),
            nn.GroupNorm(8, 512),
            nn.SiLU(),
            nn.Conv2d(512, latent_dim, 3, padding=1),
            nn.GroupNorm(8, latent_dim),
            nn.SiLU(),
        )

        # Positional embedding
        self.pos_embed = PositionalEmbedding(latent_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, 3, H, W) where T=4
        Returns:
            latents: (B, D_lat, H_lat, W_lat)
        """
        B, T, C, H, W = frames.shape
        if T != 4:
            raise ValueError(f"Expected 4 frames, got {T}")

        # Encode each frame independently
        frame_feats = []
        for t in range(T):
            feat = self.frame_encoder(frames[:, t])  # (B, 256, H/8, W/8)
            frame_feats.append(feat)

        # Concatenate along channel dimension
        concat_feats = torch.cat(frame_feats, dim=1)  # (B, 256*4, H/8, W/8)

        # Temporal fusion
        latents = self.temporal_fusion(concat_feats)  # (B, D_lat, H/8, W/8)

        # Add positional embedding
        latents = self.pos_embed(latents)

        return latents


class PositionalEmbedding(nn.Module):
    """Adds learned positional embeddings to a spatial feature map."""

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


class LatentToImageDecoder(nn.Module):
    """Decodes latent grid to image using slot-based alpha-compositing.

    Uses a fixed number of learned slots that attend to the upsampled latent
    features. Each slot produces an RGBA canvas, and the canvases are
    alpha-composited to form the final image.

    Input: (B, D_lat, H_lat, W_lat)
    Output: RGB image (B, 3, H, W) + alpha masks (B, num_slots, H, W)
    """

    def __init__(self, latent_dim: int = 128, hidden_channels: int = 128,
                 out_hw: Tuple[int, int] = (224, 240), num_slots: int = 8) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.out_hw = out_hw
        self.num_slots = num_slots

        # Add positional encoding to latents
        self.pos_proj = nn.Conv2d(2, latent_dim, 1)

        # Spatial upsampling of the entire latent grid
        # (B, D, H_lat, W_lat) → (B, D, H, W)
        self.up1 = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.GroupNorm(8, hidden_channels // 2),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        # Convert to shared features
        self.to_features = nn.Sequential(
            nn.Conv2d(hidden_channels // 2, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )

        # Per-slot RGBA prediction
        # Each slot gets a separate conv layer to produce RGBA
        # This is memory-efficient because we only have num_slots (e.g., 8) canvases
        self.slot_decoders = nn.ModuleList([
            nn.Conv2d(hidden_channels, 4, 1) for _ in range(num_slots)
        ])

    def forward(self, latents: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            latents: (B, D_lat, H_lat, W_lat)
            temperature: Temperature for alpha softmax
        Returns:
            recon: (B, 3, H, W) - reconstructed image
            alpha_masks: (B, num_slots, H, W) - alpha compositing weights
            rgb_canvases: (B, num_slots, 3, H, W) - per-slot RGB canvases
        """
        B, D, H_lat, W_lat = latents.shape
        H, W = self.out_hw

        # Add positional encoding
        yy = torch.linspace(-1, 1, H_lat, device=latents.device, dtype=latents.dtype)
        xx = torch.linspace(-1, 1, W_lat, device=latents.device, dtype=latents.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        pos_embed = self.pos_proj(coords)
        latents = latents + pos_embed

        # Upsample the entire latent grid
        x = self.up1(latents)  # (B, hidden, H_lat*2, W_lat*2)
        x = self.up2(x)        # (B, hidden, H_lat*4, W_lat*4)
        x = self.up3(x)        # (B, hidden/2, H_lat*8, W_lat*8)

        # Resize to target if needed
        if x.shape[-2] != H or x.shape[-1] != W:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        # Convert to shared features
        features = self.to_features(x)  # (B, hidden, H, W)

        # Generate per-slot RGBA canvases
        slot_outputs = []
        for slot_decoder in self.slot_decoders:
            rgba = slot_decoder(features)  # (B, 4, H, W)
            slot_outputs.append(rgba)

        # Stack slot outputs: (num_slots, B, 4, H, W) → (B, num_slots, 4, H, W)
        slot_rgba = torch.stack(slot_outputs, dim=1)

        # Split into RGB and alpha
        rgb = torch.tanh(slot_rgba[:, :, :3])  # (B, num_slots, 3, H, W)
        alpha_logits = slot_rgba[:, :, 3]      # (B, num_slots, H, W)

        # Softmax over slots (with temperature)
        alpha_masks = (alpha_logits / max(temperature, 1e-3)).softmax(dim=1)

        # Alpha-composite
        recon = (rgb * alpha_masks.unsqueeze(2)).sum(dim=1)  # (B, 3, H, W)

        return recon, alpha_masks, rgb


class LatentDynamics(nn.Module):
    """Predicts next latent grid from current latent grid.

    Uses spatial convolutions to learn local dynamics and object motion.
    Input: (B, D_lat, H_lat, W_lat)
    Output: (B, D_lat, H_lat, W_lat)
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()

        # ResNet-style blocks for stability
        self.blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(latent_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, latent_dim, 3, padding=1),
            nn.GroupNorm(8, latent_dim),
        )

        self.blocks2 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(latent_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, latent_dim, 3, padding=1),
            nn.GroupNorm(8, latent_dim),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: (B, D_lat, H_lat, W_lat)
        Returns:
            next_latents: (B, D_lat, H_lat, W_lat)
        """
        # Residual connections
        x = latents + self.blocks(latents)
        x = x + self.blocks2(x)
        return x


class TemporalFramePredictor(nn.Module):
    """Complete model: encode, reconstruct, and predict."""

    def __init__(self, latent_dim: int = 128, decoder_hidden: int = 128,
                 predictor_hidden: int = 256, num_slots: int = 8) -> None:
        super().__init__()
        self.encoder = TemporalEncoder(in_channels=3, latent_dim=latent_dim)
        self.decoder = LatentToImageDecoder(latent_dim=latent_dim,
                                           hidden_channels=decoder_hidden,
                                           num_slots=num_slots)
        self.predictor = LatentDynamics(latent_dim=latent_dim,
                                       hidden_dim=predictor_hidden)
        self.latent_dim = latent_dim
        self.num_slots = num_slots
        self.img_hw = (224, 240)

    def forward(self, frames: torch.Tensor, temperature: float = 1.0) -> dict:
        """
        Args:
            frames: (B, 5, 3, H, W) - 5 sequential frames
            temperature: Temperature for alpha softmax
        Returns:
            dict with keys:
                - latents: (B, D, H_lat, W_lat)
                - recon_frames: (B, 4, 3, H, W) - reconstructed frames 0-3
                - recon_alpha: (B, 4, N, H, W) - alpha masks for frames 0-3
                - recon_rgb: (B, 4, N, 3, H, W) - RGB canvases for frames 0-3
                - pred_latents: (B, D, H_lat, W_lat) - predicted next latents
                - pred_frame: (B, 3, H, W) - predicted frame 4
                - pred_alpha: (B, N, H, W) - alpha masks for predicted frame
                - pred_rgb: (B, N, 3, H, W) - RGB canvases for predicted frame
        """
        if frames.dim() != 5 or frames.shape[1] != 5:
            raise ValueError(f"Expected (B, 5, 3, H, W), got {frames.shape}")

        # Encode first 4 frames
        input_frames = frames[:, :4]  # (B, 4, 3, H, W)
        latents = self.encoder(input_frames)  # (B, D, H_lat, W_lat)

        # Reconstruct input frames
        # We decode once from the fused latents (representing all 4 frames)
        # The latents encode temporal information from all 4 frames
        recon, alpha, rgb = self.decoder(latents, temperature=temperature)

        # Expand to match 4 frames (broadcasting same reconstruction)
        # NOTE: We use shared latents from all 4 frames, so same reconstruction
        recon_frames = recon.unsqueeze(1).expand(-1, 4, -1, -1, -1)  # (B, 4, 3, H, W)
        recon_alpha = alpha.unsqueeze(1).expand(-1, 4, -1, -1, -1)  # (B, 4, num_slots, H, W)
        recon_rgb = rgb.unsqueeze(1).expand(-1, 4, -1, -1, -1, -1)  # (B, 4, num_slots, 3, H, W)

        # Predict next latents
        pred_latents = self.predictor(latents)  # (B, D, H_lat, W_lat)

        # Decode predicted latents to get next frame
        pred_frame, pred_alpha, pred_rgb = self.decoder(pred_latents, temperature=temperature)

        return {
            'latents': latents,
            'recon_frames': recon_frames,
            'recon_alpha': recon_alpha,
            'recon_rgb': recon_rgb,
            'pred_latents': pred_latents,
            'pred_frame': pred_frame,
            'pred_alpha': pred_alpha,
            'pred_rgb': pred_rgb,
        }

    def decode_latents(self, latents: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode arbitrary latent modifications."""
        return self.decoder(latents, temperature=temperature)


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------

def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    """Convert normalized tensor to PIL image."""
    if img.dim() != 3:
        raise ValueError("Expected (3,H,W) tensor")
    img = img.unsqueeze(0)
    img = unnormalize(img)[0].clamp(0, 1)
    arr = (img.mul(255).round().byte().permute(1, 2, 0).cpu().numpy())
    return Image.fromarray(arr)


def alpha_mask_to_pil(alpha: torch.Tensor) -> Image.Image:
    """Convert alpha mask to colored segmentation image."""
    if alpha.dim() != 3:
        raise ValueError("Expected (N,H,W)")
    seg_idx = alpha.argmax(dim=0).cpu().numpy().astype(np.int32)
    cmap = plt.get_cmap("tab20")
    colors = (np.array([cmap(i % cmap.N)[:3] for i in range(alpha.shape[0])]) * 255).astype(np.uint8)
    return Image.fromarray(colors[seg_idx])


def save_training_samples(frames: torch.Tensor, outputs: dict, out_dir: Path,
                          step: int, max_items: int = 2) -> None:
    """Save visualization of reconstruction and prediction.

    Layout (3 rows x 5 columns):
    Row 1: Original frames 0, 1, 2, 3, 4
    Row 2: [empty], [empty], [empty], Reconstructed frame 3, Predicted frame 4
    Row 3: [empty], [empty], [empty], Alpha mask (recon), Alpha mask (pred)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    B = frames.shape[0]
    rows = min(B, max_items)

    for i in range(rows):
        # Get dimensions from first frame
        w, h = tensor_to_pil(frames[i, 0]).size

        # Create canvas: 3 rows x 5 columns
        canvas = Image.new("RGB", (5 * w, 3 * h))

        # Row 1: Original frames 0-4
        for t in range(5):
            pane = tensor_to_pil(frames[i, t])
            canvas.paste(pane, (t * w, 0))

        # Row 2, Column 3: Reconstructed frame 3
        recon_pane = tensor_to_pil(outputs['recon_frames'][i, 3])
        canvas.paste(recon_pane, (3 * w, h))

        # Row 2, Column 4: Predicted frame 4
        pred_pane = tensor_to_pil(outputs['pred_frame'][i])
        canvas.paste(pred_pane, (4 * w, h))

        # Row 3, Column 3: Alpha mask for reconstruction
        recon_alpha_pane = alpha_mask_to_pil(outputs['recon_alpha'][i, 3])
        canvas.paste(recon_alpha_pane, (3 * w, 2 * h))

        # Row 3, Column 4: Alpha mask for prediction
        pred_alpha_pane = alpha_mask_to_pil(outputs['pred_alpha'][i])
        canvas.paste(pred_alpha_pane, (4 * w, 2 * h))

        canvas.save(out_dir / f"sample_step_{step:06d}_idx_{i}.png")


def save_pca_traversal(model: TemporalFramePredictor, frames: torch.Tensor,
                       out_dir: Path, step: int, device: torch.device,
                       n_components: int = 3, n_steps: int = 7) -> None:
    """Visualize PCA axis traversal in latent space."""
    from sklearn.decomposition import PCA

    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    with torch.no_grad():
        # Move frames to device if needed
        frames = frames.to(device)

        # Encode frames
        latents = model.encoder(frames[:, :4])  # (B, D, H_lat, W_lat)
        B, D, H_lat, W_lat = latents.shape

        # Flatten for PCA
        latents_flat = latents.view(B, -1).cpu().numpy()  # (B, D*H_lat*W_lat)

        # Compute PCA
        pca = PCA(n_components=n_components)
        pca.fit(latents_flat)

        # Traverse along each principal component
        alphas = np.linspace(-3, 3, n_steps)

        for pc_idx in range(n_components):
            panes: List[Image.Image] = []

            for alpha in alphas:
                # Modify latents along PC
                modified = latents_flat[0:1] + alpha * pca.components_[pc_idx:pc_idx+1]
                modified_latents = torch.from_numpy(modified).float().to(device)
                modified_latents = modified_latents.view(1, D, H_lat, W_lat)

                # Decode
                recon, _, _ = model.decode_latents(modified_latents)
                panes.append(tensor_to_pil(recon[0]))

            # Arrange in grid
            w, h = panes[0].size
            canvas = Image.new("RGB", (len(panes) * w, h))
            for idx, pane in enumerate(panes):
                canvas.paste(pane, (idx * w, 0))

            canvas.save(out_dir / f"pca_pc{pc_idx}_step_{step:06d}.png")


def plot_loss(hist: List[Tuple[int, float, float]], out_dir: Path, step: int) -> None:
    """Plot reconstruction and prediction losses."""
    if not hist:
        return
    steps, recon_losses, pred_losses = zip(*hist)
    plt.figure(figsize=(10, 5))
    plt.semilogy(steps, recon_losses, label='Reconstruction')
    plt.semilogy(steps, pred_losses, label='Prediction')
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"loss_step_{step:06d}.png")
    plt.close()


def write_loss_csv(hist: List[Tuple[int, float, float]], out_dir: Path) -> None:
    """Write loss history to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "loss_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "recon_loss", "pred_loss"])
        writer.writerows(hist)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

@dataclass
class Args:
    traj_dir: str = "data.image_distance.train_levels_1_2"
    out_dir: str = "out.temporal_frame_predictor"
    batch_size: int = 4
    lr: float = 1e-4
    epochs: int = 1000
    steps_per_epoch: int = 100
    num_frames: int = 5
    max_trajs: Optional[int] = None
    save_every: int = 50
    log_every: int = 10
    num_workers: int = 0
    device: Optional[str] = None

    # Model architecture
    latent_dim: int = 128
    decoder_hidden: int = 128
    predictor_hidden: int = 256
    num_slots: int = 8

    # Loss weights
    recon_ms_weight: float = 1.0
    recon_l1_weight: float = 0.1
    pred_ms_weight: float = 1.0
    pred_l1_weight: float = 0.1
    slot_diversity_weight: float = 0.1  # Encourage using all slots

    # Training
    alpha_temperature: float = 1.0
    alpha_temperature_min: float = 0.3
    alpha_temperature_decay: float = 5e-4

    # Staged training
    recon_only_steps: int = 500  # Train only reconstruction for first N steps

    # Visualization
    save_pca_every: int = 200


def main() -> None:
    args = tyro.cli(Args)
    logging.basicConfig(level=logging.INFO,
                       format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    device = pick_device(args.device)
    logger.info("Using device: %s", device)

    dataset = MarioTemporalDataset(args.traj_dir, sequence_len=args.num_frames,
                                   max_trajs=args.max_trajs)
    logger.info("Dataset size: %d", len(dataset))
    sampler = RandomSampler(dataset, replacement=False,
                           num_samples=args.steps_per_epoch * args.batch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                       num_workers=args.num_workers)

    model = TemporalFramePredictor(
        latent_dim=args.latent_dim,
        decoder_hidden=args.decoder_hidden,
        predictor_hidden=args.predictor_hidden,
        num_slots=args.num_slots,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    run_dir = Path(args.out_dir) / f"run__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    samples_dir = run_dir / "samples"
    pca_dir = run_dir / "pca"
    samples_dir.mkdir(parents=True, exist_ok=True)
    pca_dir.mkdir(parents=True, exist_ok=True)

    loss_hist: List[Tuple[int, float, float]] = []
    global_step = 0
    start_time = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb in loader:
            xb = xb.to(device)  # (B, 5, 3, H, W)

            # Temperature annealing for alpha masks
            current_temp = max(
                args.alpha_temperature * math.exp(-args.alpha_temperature_decay * global_step),
                args.alpha_temperature_min,
            )

            # Forward pass
            outputs = model(xb, temperature=current_temp)

            # Reconstruction loss (frames 0-3)
            # NOTE: Currently we only reconstruct once from shared latents
            # So we compare the single reconstruction to all 4 input frames
            recon_loss = torch.tensor(0.0, device=device)
            for t in range(4):
                ms_loss = ms_ssim_loss(outputs['recon_frames'][:, t], xb[:, t])
                l1_loss = F.l1_loss(outputs['recon_frames'][:, t], xb[:, t])
                recon_loss = recon_loss + args.recon_ms_weight * ms_loss + args.recon_l1_weight * l1_loss
            recon_loss = recon_loss / 4  # Average over frames

            # Prediction loss (frame 4)
            pred_ms_loss = ms_ssim_loss(outputs['pred_frame'], xb[:, 4])
            pred_l1_loss = F.l1_loss(outputs['pred_frame'], xb[:, 4])
            pred_loss = args.pred_ms_weight * pred_ms_loss + args.pred_l1_weight * pred_l1_loss

            # Slot diversity loss (encourage balanced slot usage)
            slot_diversity = torch.tensor(0.0, device=device)
            if args.slot_diversity_weight > 0.0:
                # Average alpha across spatial dimensions: (B, num_slots, H, W) → (B, num_slots)
                alpha = outputs['recon_alpha'][:, 0]  # (B, num_slots, H, W)
                slot_usage = alpha.mean(dim=[2, 3])  # (B, num_slots)

                # We want uniform usage across slots (1/num_slots for each)
                # Use KL divergence from uniform distribution
                uniform_target = torch.ones_like(slot_usage) / slot_usage.shape[1]
                # Add small epsilon for numerical stability
                slot_usage_safe = slot_usage.clamp_min(1e-8)
                # KL(uniform || slot_usage) = sum(uniform * log(uniform / slot_usage))
                kl_div = (uniform_target * (uniform_target / slot_usage_safe).log()).sum(dim=1).mean()
                slot_diversity = kl_div

            # Total loss (staged training)
            if global_step < args.recon_only_steps:
                # Stage 1: Only reconstruction
                loss = recon_loss + args.slot_diversity_weight * slot_diversity
                pred_loss_val = torch.tensor(0.0)  # Don't backprop prediction
            else:
                # Stage 2: Reconstruction + Prediction
                loss = recon_loss + pred_loss + args.slot_diversity_weight * slot_diversity
                pred_loss_val = pred_loss

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            loss_hist.append((global_step, float(recon_loss.item()), float(pred_loss_val.item())))

            # Logging
            if args.log_every > 0 and global_step % args.log_every == 0:
                elapsed = (time.monotonic() - start_time) / 60
                stage = "RECON" if global_step < args.recon_only_steps else "FULL"
                logger.info(
                    "[%s] [ep %03d] step %06d | total=%.4f | recon=%.4f | pred=%.4f | slot_div=%.4f | temp=%.3f | elapsed=%.2f min",
                    stage,
                    epoch,
                    global_step,
                    loss.item(),
                    recon_loss.item(),
                    pred_loss_val.item(),
                    slot_diversity.item(),
                    current_temp,
                    elapsed,
                )

            # Save samples
            if args.save_every > 0 and global_step % args.save_every == 0:
                with torch.no_grad():
                    save_training_samples(xb.cpu(),
                                         {k: v.cpu() for k, v in outputs.items()},
                                         samples_dir, global_step)
                plot_loss(loss_hist, run_dir, global_step)
                torch.save({
                    "epoch": epoch,
                    "step": global_step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, run_dir / "checkpoint.pt")

            # Save PCA traversal
            if args.save_pca_every > 0 and global_step % args.save_pca_every == 0:
                with torch.no_grad():
                    save_pca_traversal(model, xb, pca_dir, global_step, device)

        logger.info("[ep %03d] done.", epoch)

    write_loss_csv(loss_hist, run_dir)
    torch.save({
        "epoch": epoch,
        "step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, run_dir / "final.pt")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
