#!/usr/bin/env python3
"""Simple next-frame predictor for Mario.

Given 4 context frames + actions, predict the next frame directly in pixel space.
Straightforward architecture: encode → aggregate → decode.
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
from recon.data import list_trajectories, load_frame_as_tensor

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    data_root: Path = Path("data.image_distance.train_levels_1_2")
    output_dir: Path = Path("out.predict_mario_simple_claude")
    epochs: int = 100
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    val_fraction: float = 0.1
    seed: int = 42
    device: str = "auto"
    latent_dim: int = 256
    vis_every: int = 50
    checkpoint_every: int = 50
    resume: bool = False
    max_trajs: Optional[int] = None
    log_every: int = 50
    mixed_precision: bool = False
    image_hw: Tuple[int, int] = (224, 224)

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.output_dir = Path(self.output_dir)


class MarioSequenceDataset(Dataset):
    """Return 4 context frames + actions, and 4 future frames."""

    def __init__(
        self,
        root_dir: Path,
        *,
        image_hw: Tuple[int, int],
        normalize: Optional[Normalize] = None,
        max_trajs: Optional[int] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_hw = image_hw
        self.normalize = normalize
        self.entries: List[Tuple[List[Path], np.ndarray, int]] = []
        self.action_dim: Optional[int] = None

        traj_map = list_trajectories(self.root_dir)
        traj_items = list(traj_map.items())
        if max_trajs is not None:
            traj_items = traj_items[:max_trajs]

        for traj_name, frames in traj_items:
            actions_path = (self.root_dir / traj_name) / "actions.npz"
            assert actions_path.is_file(), f"Not a file: {actions_path}"

            if len(frames) < 8:  # Need 4 context + 4 future
                continue

            with np.load(actions_path) as data:
                if "actions" in data.files:
                    actions = data["actions"]
                elif len(data.files) == 1:
                    actions = data[data.files[0]]
                else:
                    raise ValueError(f"Missing 'actions' in {actions_path}: {data.files}")

            if actions.ndim == 1:
                actions = actions[:, None]
            if actions.shape[0] == len(frames) + 1:
                actions = actions[1:]
            if actions.shape[0] != len(frames):
                raise ValueError(
                    f"Action count {actions.shape[0]} mismatch with frames {len(frames)}"
                )

            actions = actions.astype(np.float32)
            if self.action_dim is None:
                self.action_dim = actions.shape[1]
            elif actions.shape[1] != self.action_dim:
                raise ValueError(f"Inconsistent action dimensions")

            # Sliding window: 4 context + 4 future = 8 total
            for start in range(len(frames) - 7):
                self.entries.append((frames, actions, start))

        if not self.entries:
            raise RuntimeError(f"No valid trajectories under {self.root_dir}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        frame_paths, actions, start = self.entries[idx]

        # Load 4 context frames
        context_frames = []
        for i in range(4):
            frame = load_frame_as_tensor(
                frame_paths[start + i],
                size=self.image_hw,
                normalize=None,
            )
            if self.normalize is not None:
                frame = self.normalize(frame)
            context_frames.append(frame)

        # Load 4 future frames
        future_frames = []
        for i in range(4):
            frame = load_frame_as_tensor(
                frame_paths[start + 4 + i],
                size=self.image_hw,
                normalize=None,
            )
            if self.normalize is not None:
                frame = self.normalize(frame)
            future_frames.append(frame)

        context_actions = actions[start:start + 4]

        return {
            "context_frames": torch.stack(context_frames, dim=0),
            "context_actions": torch.from_numpy(context_actions),
            "future_frames": torch.stack(future_frames, dim=0),
        }


class SimpleFrameEncoder(nn.Module):
    """Simple CNN encoder: image → latent vector."""

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim

        # Input: (3, 224, 224)
        self.net = nn.Sequential(
            # → (64, 112, 112)
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # → (128, 56, 56)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # → (256, 28, 28)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # → (512, 14, 14)
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Global average pooling → (512,)
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            # → (latent_dim,)
            nn.Linear(512, latent_dim),
        )

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        # frame: (B, 3, H, W) → (B, latent_dim)
        return self.net(frame)


class SimpleActionEncoder(nn.Module):
    """Simple MLP: action → embedding."""

    def __init__(self, action_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        # actions: (B, action_dim) → (B, embed_dim)
        return self.net(actions)


class SimpleTemporalAggregator(nn.Module):
    """Combine frame+action pairs across time."""

    def __init__(self, latent_dim: int):
        super().__init__()
        # Simple: just average the combined embeddings
        self.combine = nn.Linear(latent_dim * 2, latent_dim)

    def forward(
        self,
        frame_latents: torch.Tensor,
        action_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        # frame_latents: (B, 4, latent_dim)
        # action_embeddings: (B, 4, latent_dim)

        # Combine frame + action for each timestep
        combined = torch.cat([frame_latents, action_embeddings], dim=-1)  # (B, 4, 2*latent_dim)
        combined = self.combine(combined)  # (B, 4, latent_dim)

        # Average across time
        context = combined.mean(dim=1)  # (B, latent_dim)

        return context


class SimpleFrameDecoder(nn.Module):
    """Simple transposed CNN decoder: latent → image."""

    def __init__(self, latent_dim: int = 256, image_hw: Tuple[int, int] = (224, 224)):
        super().__init__()
        self.image_hw = image_hw

        # Start from latent vector
        self.fc = nn.Linear(latent_dim, 512 * 14 * 14)

        # Upsample with transposed convolutions
        self.net = nn.Sequential(
            # (512, 14, 14) → (256, 28, 28)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # (256, 28, 28) → (128, 56, 56)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # (128, 56, 56) → (64, 112, 112)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # (64, 112, 112) → (3, 224, 224)
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # latent: (B, latent_dim) → (B, 3, H, W)
        h = self.fc(latent)  # (B, 512*14*14)
        h = h.view(-1, 512, 14, 14)  # (B, 512, 14, 14)
        frame = self.net(h)  # (B, 3, 224, 224)

        # Ensure correct size
        if frame.shape[-2:] != self.image_hw:
            frame = F.interpolate(frame, size=self.image_hw, mode="bilinear", align_corners=False)

        return frame


class SimpleNextFramePredictor(nn.Module):
    """Simple next-frame predictor."""

    def __init__(
        self,
        action_dim: int,
        latent_dim: int = 256,
        image_hw: Tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.image_hw = image_hw
        self.latent_dim = latent_dim

        self.frame_encoder = SimpleFrameEncoder(latent_dim)
        self.action_encoder = SimpleActionEncoder(action_dim, latent_dim)
        self.aggregator = SimpleTemporalAggregator(latent_dim)
        self.decoder = SimpleFrameDecoder(latent_dim, image_hw)

    def forward(
        self,
        context_frames: torch.Tensor,
        context_actions: torch.Tensor,
    ) -> torch.Tensor:
        # context_frames: (B, 4, 3, H, W)
        # context_actions: (B, 4, action_dim)

        batch_size, num_frames = context_frames.shape[:2]

        # Encode each frame
        frame_latents = []
        for t in range(num_frames):
            latent = self.frame_encoder(context_frames[:, t])  # (B, latent_dim)
            frame_latents.append(latent)
        frame_latents = torch.stack(frame_latents, dim=1)  # (B, 4, latent_dim)

        # Encode each action
        action_embeddings = []
        for t in range(num_frames):
            emb = self.action_encoder(context_actions[:, t])  # (B, latent_dim)
            action_embeddings.append(emb)
        action_embeddings = torch.stack(action_embeddings, dim=1)  # (B, 4, latent_dim)

        # Aggregate
        context = self.aggregator(frame_latents, action_embeddings)  # (B, latent_dim)

        # Decode
        next_frame = self.decoder(context)  # (B, 3, H, W)

        return next_frame


def _ensure_dirs(config: TrainConfig) -> Dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir: Path
    if config.resume:
        existing_runs = sorted(
            [p for p in config.output_dir.iterdir() if p.is_dir() and p.name[0].isdigit()],
            reverse=True,
        )
        if existing_runs:
            run_dir = existing_runs[0]
        else:
            run_dir = config.output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

    return {
        "run": run_dir,
        "checkpoints": ckpt_dir,
        "plots": plots_dir,
        "visualizations": vis_dir,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    best_loss: float,
    history: List[Dict[str, float]],
) -> None:
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_loss": best_loss,
        "history": history,
    }, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[int, int, float, List[Dict[str, float]]]:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return (
        int(payload["epoch"]),
        int(payload["step"]),
        float(payload.get("best_loss", math.inf)),
        payload.get("history", []),
    )


def _tensor_to_image(img: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) tensor to (H, W, C) numpy array."""
    if img.dim() != 3:
        raise ValueError(f"Expected (C,H,W) tensor, got shape {tuple(img.shape)}")
    with torch.no_grad():
        data = unnormalize(img.cpu()).clamp(0.0, 1.0)
    # unnormalize returns (1, C, H, W)
    assert data.dim() == 4, f"Expected dim size 4, got {data.dim()}"
    assert data.shape[0] == 1, f"Expected batch size 1, got {data.shape[0]}"
    data = data.squeeze(0)  # (C, H, W)
    return data.permute(1, 2, 0).numpy()


def _blank_image(hw: Tuple[int, int]) -> np.ndarray:
    h, w = hw
    return np.zeros((h, w, 3), dtype=np.float32)


def render_visualization_grid(
    model: SimpleNextFramePredictor,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    output_path: Path,
) -> None:
    """
    Visualization with 2 rows × 8 columns:
    - Row 0: context frames (0-3) + ground truth future frames (4-7)
    - Row 1: blanks (0-3) + autoregressive predictions (4-7)
    """
    model.eval()

    # Take first sample from batch
    context_frames = batch["context_frames"][0].unsqueeze(0).to(device)  # (1, 4, 3, H, W)
    context_actions = batch["context_actions"][0].unsqueeze(0).to(device)  # (1, 4, action_dim)
    future_frames = batch["future_frames"][0].unsqueeze(0).to(device)  # (1, 4, 3, H, W)

    with torch.no_grad():
        # Autoregressive rollout for 4 steps
        # Start with context frames [0, 1, 2, 3]
        window_frames = context_frames.clone()  # (1, 4, 3, H, W)
        predictions = []

        for step in range(4):
            # Predict next frame from current window
            # Use zero action for future (since we don't have ground truth actions)
            window_actions = torch.zeros(
                (1, 4, context_actions.shape[-1]),
                dtype=context_actions.dtype,
                device=device
            )
            # For the last 4 actions, use context if available
            if step == 0:
                window_actions = context_actions

            pred_frame = model(window_frames, window_actions)  # (1, 3, H, W)
            predictions.append(pred_frame[0])  # (3, H, W)

            # Shift window: drop oldest frame, add prediction
            window_frames = torch.cat([
                window_frames[:, 1:],  # (1, 3, 3, H, W)
                pred_frame.unsqueeze(1)  # (1, 1, 3, H, W)
            ], dim=1)  # (1, 4, 3, H, W)

    # Create visualization
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    # Row 0: context frames (cols 0-3) + ground truth future (cols 4-7)
    for i in range(4):
        axes[0, i].imshow(_tensor_to_image(context_frames[0, i]))
        axes[0, i].set_title(f"Context {i}", fontsize=8)

    for i in range(4):
        axes[0, 4 + i].imshow(_tensor_to_image(future_frames[0, i]))
        axes[0, 4 + i].set_title(f"GT Future {i}", fontsize=8)

    # Row 1: blanks (cols 0-3) + predictions (cols 4-7)
    for i in range(4):
        axes[1, i].imshow(_blank_image(model.image_hw))

    for i in range(4):
        axes[1, 4 + i].imshow(_tensor_to_image(predictions[i]))
        axes[1, 4 + i].set_title(f"Pred {i}", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_loss_csv(history: List[Dict[str, float]], path: Path) -> None:
    """Write loss history to CSV file."""
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        for entry in history:
            writer.writerow([entry["step"], entry["loss"]])


def plot_losses(history: List[Dict[str, float]], out_dir: Path) -> None:
    """Plot loss curves and save to PNG."""
    if not history:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = [entry["step"] for entry in history]
    losses = [entry["loss"] for entry in history]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, label="MSE Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=200)
    plt.close()


def prepare_dataloaders(
    config: TrainConfig
) -> Tuple[DataLoader, Optional[DataLoader], MarioSequenceDataset]:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = Normalize(mean=mean, std=std)

    dataset = MarioSequenceDataset(
        config.data_root,
        image_hw=config.image_hw,
        normalize=normalize,
        max_trajs=config.max_trajs,
    )

    generator = torch.Generator().manual_seed(config.seed)
    if config.val_fraction <= 0.0 or len(dataset) < 10:
        train_dataset = dataset
        val_dataset = None
    else:
        val_count = max(1, int(len(dataset) * config.val_fraction))
        train_count = len(dataset) - val_count
        train_dataset, val_dataset = random_split(
            dataset, [train_count, val_count], generator=generator
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    return train_loader, val_loader, dataset


def train(config: TrainConfig) -> None:
    set_seed(config.seed)
    dirs = _ensure_dirs(config)
    run_dir = dirs["run"]
    logger.info("Writing outputs to %s", run_dir)

    train_loader, val_loader, full_dataset = prepare_dataloaders(config)

    device_pref = None
    if config.device is not None and str(config.device).lower() != "auto":
        device_pref = config.device
    device = pick_device(device_pref)
    logger.info("Using device %s", device)

    action_dim = full_dataset.action_dim or 1

    model = SimpleNextFramePredictor(
        action_dim=action_dim,
        latent_dim=config.latent_dim,
        image_hw=config.image_hw,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    loss_fn = nn.MSELoss()

    start_epoch = 0
    global_step = 0
    best_loss = math.inf
    loss_history: List[Dict[str, float]] = []

    if config.resume:
        last_ckpt = dirs["checkpoints"] / "last.pt"
        if last_ckpt.exists():
            logger.info("Resuming from %s", last_ckpt)
            start_epoch, global_step, best_loss, loss_history = load_checkpoint(
                last_ckpt, model, optimizer
            )
            start_epoch += 1
            model.to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)

    logger.info(
        "Dataset size: %d samples (action_dim=%d)",
        len(train_loader.dataset),
        action_dim,
    )

    for epoch in range(start_epoch, config.epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            context_frames = batch["context_frames"].to(device)  # (B, 4, 3, H, W)
            context_actions = batch["context_actions"].to(device)  # (B, 4, action_dim)
            future_frames = batch["future_frames"].to(device)  # (B, 4, 3, H, W)

            # Target: first future frame
            target_frame = future_frames[:, 0]  # (B, 3, H, W)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=config.mixed_precision):
                pred_frame = model(context_frames, context_actions)  # (B, 3, H, W)
                loss = loss_fn(pred_frame, target_frame)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            current_step = global_step + 1

            if config.log_every > 0 and current_step % config.log_every == 0:
                logger.info(
                    "Epoch %d Step %d | loss %.4f",
                    epoch,
                    current_step,
                    loss.item(),
                )

            loss_history.append({
                "step": current_step,
                "loss": float(loss.item()),
            })

            if config.checkpoint_every > 0 and current_step % config.checkpoint_every == 0:
                save_checkpoint(
                    dirs["checkpoints"] / "last.pt",
                    model,
                    optimizer,
                    epoch,
                    current_step,
                    best_loss,
                    loss_history,
                )

            if config.vis_every > 0 and current_step % config.vis_every == 0:
                plot_losses(loss_history, dirs["plots"])
                write_loss_csv(loss_history, dirs["plots"] / "loss_history.csv")

                vis_path = dirs["visualizations"] / f"step_{current_step:06d}.png"
                render_visualization_grid(model, batch, device, vis_path)
                model.train()

            global_step = current_step

        # End of epoch
        steps_in_epoch = len(train_loader)
        avg_loss = running_loss / steps_in_epoch
        logger.info(
            "Epoch %d complete | avg_loss %.4f",
            epoch,
            avg_loss,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                dirs["checkpoints"] / "best.pt",
                model,
                optimizer,
                epoch,
                global_step,
                best_loss,
                loss_history,
            )
            logger.info("Saved new best checkpoint (loss %.4f)", best_loss)

        save_checkpoint(
            dirs["checkpoints"] / "last.pt",
            model,
            optimizer,
            epoch,
            global_step,
            best_loss,
            loss_history,
        )

    # Final checkpoint
    save_checkpoint(
        dirs["checkpoints"] / "final.pt",
        model,
        optimizer,
        config.epochs - 1,
        global_step,
        best_loss,
        loss_history,
    )

    plot_losses(loss_history, dirs["plots"])
    write_loss_csv(loss_history, dirs["plots"] / "loss_history.csv")

    config_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(config).items()
    }
    config_payload["run_dir"] = str(run_dir)
    with (run_dir / "config.json").open("w") as fp:
        json.dump(config_payload, fp, indent=2)

    logger.info("Training complete.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = tyro.cli(TrainConfig)
    train(config)


if __name__ == "__main__":
    main()
