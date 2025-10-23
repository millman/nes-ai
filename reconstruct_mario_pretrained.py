#!/usr/bin/env python3
"""Reconstruct NES Mario frames with frozen ImageNet encoders and learned decoders.

Each pretrained backbone keeps a dedicated decoder that trains in lock-step with
the others. Checkpoints track the last, best, and final decoder weights and can
be used to resume training runs.
"""
from __future__ import annotations

import random
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.models import (
    ResNet50_Weights,
    ConvNeXt_Base_Weights,
    convnext_base,
    resnet50,
)
import tyro
from PIL import Image

from predict_mario_ms_ssim import default_transform, pick_device, unnormalize
from trajectory_utils import list_state_frames, list_traj_dirs


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class MarioFrameDataset(Dataset):
    """Flat dataset of Mario frames with ImageNet preprocessing."""

    def __init__(
        self,
        root_dir: Path,
        *,
        transform: Optional[T.Compose] = None,
        max_trajs: Optional[int] = None,
    ) -> None:
        self.transform = transform or default_transform()
        self.paths: List[Path] = []
        traj_count = 0
        for traj_dir in list_traj_dirs(root_dir):
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


def load_image_batch(paths: Sequence[str], transform: T.Compose) -> torch.Tensor:
    tensors = []
    for path in paths:
        with Image.open(path).convert("RGB") as img:
            tensors.append(transform(img))
    if not tensors:
        raise RuntimeError("No images provided for visualisation batch.")
    return torch.stack(tensors)


def sample_random_batch(dataset: MarioFrameDataset, count: int) -> torch.Tensor:
    if count <= 0:
        raise ValueError("count must be positive.")
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty; cannot sample frames.")
    indices = random.sample(range(len(dataset)), k=min(count, len(dataset)))
    tensors = [dataset[idx][0] for idx in indices]
    return torch.stack(tensors)


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8 if out_ch % 8 == 0 else 1, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8 if out_ch % 8 == 0 else 1, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """Shared decoder architecture parameterised by encoder channel width."""

    def __init__(self, in_channels: int, *, base_channels: int = 512) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=1),
            nn.SiLU(inplace=True),
        )
        self.up1 = UpBlock(base_channels, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 96)
        self.up4 = UpBlock(96, 64)
        self.up5 = UpBlock(64, 48)
        self.head = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = self.proj(feat)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        h = self.up5(h)
        out = self.head(h)
        if out.shape[-2:] != (224, 224):
            out = F.interpolate(out, size=(224, 224), mode="bilinear", align_corners=False)
        return out


@torch.no_grad()
def _resnet_encoder(weights: ResNet50_Weights) -> nn.Module:
    model = resnet50(weights=weights)
    layers = list(model.children())[:-2]  # drop avgpool + fc
    encoder = nn.Sequential(*layers)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


@torch.no_grad()
def _convnext_encoder(weights: ConvNeXt_Base_Weights) -> nn.Module:
    model = convnext_base(weights=weights)
    encoder = model.features  # (B,1024,7,7)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


# ---------------------------------------------------------------------------
# Trainer wrapper
# ---------------------------------------------------------------------------


class ReconstructionTrainer:
    """Wraps a frozen encoder and trainable decoder with a unified step API."""

    def __init__(
        self,
        name: str,
        encoder: nn.Module,
        decoder: nn.Module,
        *,
        device: torch.device,
        lr: float,
    ) -> None:
        self.name = name
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        self.loss_fn = nn.L1Loss()
        self.history: List[Tuple[int, float]] = []
        self.global_step = 0
        self.best_loss: Optional[float] = None

    def step(self, batch: torch.Tensor) -> Tuple[float, bool]:
        self.decoder.train()
        with torch.no_grad():
            feats = self.encoder(batch)
        recon = self.decoder(feats)
        loss = self.loss_fn(recon, batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.global_step += 1
        loss_val = float(loss.detach().item())
        self.history.append((self.global_step, loss_val))
        improved = self.best_loss is None or loss_val < self.best_loss
        if improved:
            self.best_loss = loss_val
        return loss_val, improved

    @torch.no_grad()
    def reconstruct(self, batch: torch.Tensor) -> torch.Tensor:
        was_training = self.decoder.training
        self.encoder.eval()
        self.decoder.eval()
        feats = self.encoder(batch.to(self.device))
        recon = self.decoder(feats)
        if was_training:
            self.decoder.train()
        return recon.cpu()

    def state_dict(self) -> dict:
        return {
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "history": self.history,
            "global_step": self.global_step,
            "name": self.name,
            "best_loss": self.best_loss,
        }

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_state_dict(self, state: dict, *, lr: Optional[float] = None) -> None:
        self.decoder.load_state_dict(state["decoder"])
        self.optimizer.load_state_dict(state["optimizer"])
        if lr is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = lr
        self.history = state.get("history", [])
        self.global_step = state.get("global_step", 0)
        self.best_loss = state.get("best_loss")


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def _tensor_to_numpy(img: torch.Tensor) -> torch.Tensor:
    return img.permute(1, 2, 0).clamp(0.0, 1.0).cpu()


def save_recon_grid(
    inputs: torch.Tensor,
    reconstructions: Sequence[Tuple[str, torch.Tensor]],
    *,
    out_path: Path,
) -> None:
    rows = inputs.shape[0]
    cols = 1 + len(reconstructions)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1:
        axes = axes[None, :]
    unnorm_inputs = unnormalize(inputs)
    unnorm_recons = [(name, unnormalize(tensor)) for name, tensor in reconstructions]
    col_titles = ["Input"] + [name for name, _ in unnorm_recons]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title)
    for row in range(rows):
        axes[row, 0].imshow(_tensor_to_numpy(unnorm_inputs[row]))
        axes[row, 0].axis("off")
        for col, (_, tensor) in enumerate(unnorm_recons, start=1):
            axes[row, col].imshow(_tensor_to_numpy(tensor[row]))
            axes[row, col].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loss_histories(trainers: Sequence[ReconstructionTrainer], out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for trainer in trainers:
        if not trainer.history:
            continue
        steps, losses = zip(*trainer.history)
        plt.plot(steps, losses, label=trainer.name)
    plt.xlabel("Step")
    plt.ylabel("L1 loss")
    plt.title("Decoder training losses")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_loss_histories(trainers: Sequence[ReconstructionTrainer], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for trainer in trainers:
        history_path = out_dir / f"{trainer.name}_loss.csv"
        with history_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss"])
            for step, loss in trainer.history:
                writer.writerow([step, loss])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class Config:
    traj_root: Path = Path("data.image_distance.train_levels_1_2")
    out_dir: Path = Path("out.reconstruct_mario_pretrained")
    max_trajs: Optional[int] = None
    batch_size: int = 16
    num_workers: int = 0
    train_steps: int = 1_000
    log_every: int = 20
    vis_every: int = 100
    vis_rows: int = 6
    lr: float = 1e-4
    device: Optional[str] = None
    seed: int = 0
    resume_dir: Optional[Path] = None
    resume_tag: str = "last"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_trainers(device: torch.device, lr: float) -> List[ReconstructionTrainer]:
    resnet_enc = _resnet_encoder(ResNet50_Weights.IMAGENET1K_V2)
    convnext_enc = _convnext_encoder(ConvNeXt_Base_Weights.IMAGENET1K_V1)
    resnet_dec = Decoder(2048)
    convnext_dec = Decoder(1024)
    return [
        ReconstructionTrainer("resnet50", resnet_enc, resnet_dec, device=device, lr=lr),
        ReconstructionTrainer("convnext_base", convnext_enc, convnext_dec, device=device, lr=lr),
    ]


def main() -> None:
    cfg = tyro.cli(Config)
    if cfg.vis_rows <= 0:
        raise ValueError("vis_rows must be positive.")
    if cfg.vis_every <= 0:
        raise ValueError("vis_every must be positive.")
    if cfg.resume_tag not in {"last", "best", "final"}:
        raise ValueError("resume_tag must be one of {'last', 'best', 'final'}.")

    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    dataset = MarioFrameDataset(Path(cfg.traj_root), max_trajs=cfg.max_trajs)

    if cfg.resume_dir is not None:
        run_dir = cfg.resume_dir
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {run_dir}")
    else:
        run_dir = cfg.out_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_root = run_dir / "checkpoints"
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    samples_root = run_dir / "samples"
    fixed_samples_dir = samples_root / "fixed"
    rolling_samples_dir = samples_root / "rolling"
    fixed_samples_dir.mkdir(parents=True, exist_ok=True)
    rolling_samples_dir.mkdir(parents=True, exist_ok=True)

    trainers = build_trainers(device, cfg.lr)
    checkpoint_paths: dict[str, dict[str, Path]] = {}

    for trainer in trainers:
        trainer_dir = checkpoints_root / trainer.name
        trainer_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_paths[trainer.name] = {
            "last": trainer_dir / "last.pt",
            "best": trainer_dir / "best.pt",
            "final": trainer_dir / "final.pt",
        }
        if cfg.resume_dir is not None:
            resume_path = checkpoint_paths[trainer.name][cfg.resume_tag]
            if not resume_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint for {trainer.name!r} not found at {resume_path}"
                )
            state = torch.load(resume_path, map_location=device)
            trainer.load_state_dict(state, lr=cfg.lr)

    if cfg.resume_dir is not None:
        step_set = {trainer.global_step for trainer in trainers}
        if len(step_set) != 1:
            raise RuntimeError("Loaded checkpoints have mismatched global steps.")
        start_step = step_set.pop()
    else:
        start_step = 0

    vis_paths_file = run_dir / "vis_paths.txt"
    if vis_paths_file.exists():
        vis_paths = [
            line.strip() for line in vis_paths_file.read_text().splitlines() if line.strip()
        ]
    else:
        vis_count = min(cfg.vis_rows, len(dataset))
        if vis_count <= 0:
            raise RuntimeError("Not enough frames available for visualisation.")
        indices = random.sample(range(len(dataset)), vis_count)
        vis_paths = [str(dataset.paths[idx]) for idx in indices]
        vis_paths_file.write_text("\n".join(vis_paths) + "\n")
    vis_batch = load_image_batch(vis_paths, dataset.transform)
    vis_batch_device = vis_batch.to(device)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    target_step = start_step + cfg.train_steps
    if cfg.train_steps > 0:
        data_iterator = iter(loader)
        for current_step in range(start_step + 1, target_step + 1):
            try:
                batch, _ = next(data_iterator)
            except StopIteration:
                data_iterator = iter(loader)
                batch, _ = next(data_iterator)
            batch = batch.to(device, non_blocking=True)
            losses: dict[str, float] = {}
            for trainer in trainers:
                loss, improved = trainer.step(batch)
                losses[trainer.name] = loss
                trainer.save_checkpoint(checkpoint_paths[trainer.name]["last"])
                if improved:
                    trainer.save_checkpoint(checkpoint_paths[trainer.name]["best"])
            if cfg.log_every > 0 and current_step % cfg.log_every == 0:
                loss_str = ", ".join(f"{name}: {losses[name]:.4f}" for name in losses)
                print(f"[step {current_step:05d}] {loss_str}")
            if current_step % cfg.vis_every == 0 or current_step == target_step:
                step_tag = f"step_{current_step:05d}"
                fixed_recons = [
                    (trainer.name, trainer.reconstruct(vis_batch_device)) for trainer in trainers
                ]
                save_recon_grid(
                    vis_batch,
                    fixed_recons,
                    out_path=fixed_samples_dir / f"{step_tag}.png",
                )
                rolling_batch = sample_random_batch(dataset, cfg.vis_rows)
                rolling_batch_device = rolling_batch.to(device)
                rolling_recons = [
                    (trainer.name, trainer.reconstruct(rolling_batch_device)) for trainer in trainers
                ]
                save_recon_grid(
                    rolling_batch,
                    rolling_recons,
                    out_path=rolling_samples_dir / f"{step_tag}.png",
                )
    else:
        print("train_steps is 0; skipping decoder optimisation.")
        step_tag = f"step_{target_step:05d}"
        fixed_recons = [
            (trainer.name, trainer.reconstruct(vis_batch_device)) for trainer in trainers
        ]
        save_recon_grid(
            vis_batch,
            fixed_recons,
            out_path=fixed_samples_dir / f"{step_tag}.png",
        )
        rolling_batch = sample_random_batch(dataset, cfg.vis_rows)
        rolling_batch_device = rolling_batch.to(device)
        rolling_recons = [
            (trainer.name, trainer.reconstruct(rolling_batch_device)) for trainer in trainers
        ]
        save_recon_grid(
            rolling_batch,
            rolling_recons,
            out_path=rolling_samples_dir / f"{step_tag}.png",
        )

    plot_loss_histories(trainers, metrics_dir / "decoder_losses.png")
    write_loss_histories(trainers, metrics_dir)
    for trainer in trainers:
        paths = checkpoint_paths[trainer.name]
        trainer.save_checkpoint(paths["last"])
        if trainer.best_loss is not None and not paths["best"].exists():
            trainer.save_checkpoint(paths["best"])
        trainer.save_checkpoint(paths["final"])


if __name__ == "__main__":
    main()
