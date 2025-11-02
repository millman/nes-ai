#!/usr/bin/env python3
"""Simplified training loop for a 4-frame Mario predictor with action modulation.

The model embeds four context frames and their corresponding controller actions,
builds a latent via an interleaved GRU, and decodes back to pixels. Future
latents are produced by sliding a 4-frame window that swaps in predicted frames
while shifting in the next action embedding, enforcing the modulation loss.
"""

from __future__ import annotations

import csv
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import tyro
from recon.data import list_trajectories, load_frame_as_tensor
from predict_mario_ms_ssim import pick_device


plt.switch_backend("Agg")
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
    output_dir: Path = Path("out.predict_mario3")
    epochs: int = 1000
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_workers: int = 4
    device: str = "mps"
    seed: int = 0
    context_frames: int = 4
    prediction_frames: int = 4
    image_hw: Tuple[int, int] = (224, 224)
    log_every_steps: int = 10
    vis_every_steps: int = 100
    checkpoint_every_steps: int = 1000
    max_trajectories: Optional[int] = None


class MarioSequenceDataset(Dataset):
    """Return sliding windows of four context frames and four future frames plus actions."""

    def __init__(
        self,
        *,
        root: Path,
        context: int,
        prediction: int,
        image_hw: Tuple[int, int],
        max_traj: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.context = context
        self.prediction = prediction
        self.image_hw = image_hw
        trajectories = list_trajectories(self.root)
        items = list(trajectories.items())
        if max_traj is not None:
            items = items[:max_traj]
        self._samples: List[Tuple[List[Path], np.ndarray, int]] = []
        self.action_dim: Optional[int] = None

        for traj_name, frame_paths in items:
            actions_path = (self.root / traj_name) / "actions.npz"
            if not actions_path.is_file():
                logger.warning("Missing actions.npz for %s; skipping.", traj_name)
                continue
            try:
                with np.load(actions_path) as data:
                    if "actions" in data:
                        actions = data["actions"]
                    else:
                        actions = data[list(data.files)[0]]
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to read %s: %s", actions_path, exc)
                continue

            if actions.ndim == 1:
                actions = actions[:, None]
            if actions.shape[0] == len(frame_paths) + 1:
                actions = actions[1:]
            if actions.shape[0] < len(frame_paths):
                logger.warning(
                    "Action count %d shorter than frames %d in %s; skipping.",
                    actions.shape[0],
                    len(frame_paths),
                    traj_name,
                )
                continue

            actions = actions.astype(np.float32)
            if self.action_dim is None:
                self.action_dim = actions.shape[1]
            elif self.action_dim != actions.shape[1]:
                logger.warning(
                    "Inconsistent action dim in %s (expected %d, found %d); skipping.",
                    traj_name,
                    self.action_dim,
                    actions.shape[1],
                )
                continue

            minimum_frames = context + prediction
            if len(frame_paths) < minimum_frames:
                continue

            max_start = len(frame_paths) - minimum_frames
            for start in range(max_start + 1):
                self._samples.append((frame_paths, actions, start))

        if not self._samples:
            raise RuntimeError(f"No usable sequences found under {self.root}")
        if self.action_dim is None:
            raise RuntimeError("Unable to infer action dimensionality.")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        frame_paths, actions, start = self._samples[index]
        context_frames: List[torch.Tensor] = []
        future_frames: List[torch.Tensor] = []
        for offset in range(self.context):
            frame = load_frame_as_tensor(
                frame_paths[start + offset],
                size=self.image_hw,
                normalize=None,
            )
            context_frames.append(frame)
        for offset in range(self.prediction):
            frame = load_frame_as_tensor(
                frame_paths[start + self.context + offset],
                size=self.image_hw,
                normalize=None,
            )
            future_frames.append(frame)

        context_actions = actions[start : start + self.context]
        future_actions = actions[start + self.context : start + self.context + self.prediction]

        return {
            "context_frames": torch.stack(context_frames, dim=0),
            "context_actions": torch.from_numpy(context_actions),
            "future_frames": torch.stack(future_frames, dim=0),
            "future_actions": torch.from_numpy(future_actions),
        }


class FrameEncoder(nn.Module):
    """Shared CNN encoder that returns latent vectors."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        channels = [3, 32, 64, 128, 256]
        layers: List[nn.Module] = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(channels[-1], latent_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        x = self.features(frames)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class ActionEncoder(nn.Module):
    """Embed multi-hot controller vectors."""

    def __init__(self, action_dim: int, latent_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.net(actions)


class InterleavedAggregator(nn.Module):
    """Combine frame and action embeddings using an interleaved GRU."""

    def __init__(self, frame_dim: int, action_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.frame_proj = nn.Linear(frame_dim, latent_dim)
        self.transition_proj = nn.Linear(frame_dim + action_dim, latent_dim)
        self.gru = nn.GRUCell(latent_dim, latent_dim)

    def forward(
        self,
        frame_embeds: torch.Tensor,
        action_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if frame_embeds.shape != action_embeds.shape:
            raise ValueError(
                f"Frame embeds shape {frame_embeds.shape} expected to match action embeds {action_embeds.shape}"
            )
        bsz, steps, _ = frame_embeds.shape
        if steps < 1:
            raise ValueError("Expected at least one frame to aggregate.")
        hidden = self.frame_proj(frame_embeds[:, 0, :])
        if steps > 1:
            for idx in range(steps - 1):
                transition_input = torch.cat(
                    [frame_embeds[:, idx, :], action_embeds[:, idx, :]],
                    dim=-1,
                )
                transition_vec = self.transition_proj(transition_input)
                hidden = self.gru(transition_vec, hidden)
                next_frame_latent = self.frame_proj(frame_embeds[:, idx + 1, :])
                hidden = 0.5 * (hidden + next_frame_latent)
        final_transition = torch.cat(
            [frame_embeds[:, -1, :], action_embeds[:, -1, :]],
            dim=-1,
        )
        final_vec = self.transition_proj(final_transition)
        hidden = self.gru(final_vec, hidden)
        return hidden


class FrameDecoder(nn.Module):
    """Upsample latent vectors back into 224x224 RGB frames."""

    def __init__(self, latent_dim: int, base: int = 32) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, base * 8 * 14 * 14)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base * 8, base * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 4, base * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 2, base, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.fc(latent)
        x = x.view(latent.size(0), -1, 14, 14)
        x = self.net(x)
        return torch.sigmoid(x)


class MarioPredictor(nn.Module):
    """End-to-end context encoder, action modulator, and decoder."""

    def __init__(self, *, action_dim: int, latent_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.frame_encoder = FrameEncoder(latent_dim)
        self.action_encoder = ActionEncoder(action_dim, latent_dim)
        self.aggregator = InterleavedAggregator(latent_dim, latent_dim, latent_dim)
        self.decoder = FrameDecoder(latent_dim)

    def _build_latent(self, frames: Sequence[torch.Tensor], actions: Sequence[torch.Tensor]) -> torch.Tensor:
        frame_tensor = torch.stack(frames, dim=1)
        action_tensor = torch.stack(actions, dim=1)
        return self.aggregator(frame_tensor, action_tensor)

    def forward(
        self,
        context_frames: torch.Tensor,
        context_actions: torch.Tensor,
        *,
        future_actions: Optional[torch.Tensor] = None,
        future_frames: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        bsz, context_steps, _, _, _ = context_frames.shape
        device = context_frames.device
        flat_context = context_frames.view(-1, *context_frames.shape[2:])
        context_embeds = self.frame_encoder(flat_context).view(bsz, context_steps, self.latent_dim)
        context_action_embeds = self.action_encoder(context_actions)

        frame_window: List[torch.Tensor] = [
            context_embeds[:, idx, :] for idx in range(context_steps)
        ]
        action_window: List[torch.Tensor] = [
            context_action_embeds[:, idx, :] for idx in range(context_steps)
        ]

        aggregated_latent = self._build_latent(frame_window, action_window)
        context_latent = aggregated_latent
        reconstructed = self.decoder(context_embeds[:, -1, :])

        rollout_steps = 0
        future_action_embeds = None
        if future_actions is not None:
            rollout_steps = future_actions.shape[1]
            future_action_embeds = self.action_encoder(future_actions)
        if future_frames is not None:
            rollout_steps = future_frames.shape[1]
        if future_actions is not None and future_frames is not None:
            rollout_steps = min(future_actions.shape[1], future_frames.shape[1])

        predicted_frames: List[torch.Tensor] = []
        predicted_latents: List[torch.Tensor] = []
        target_latents: List[torch.Tensor] = []

        future_frame_embeds = None
        if future_frames is not None:
            flat_future = future_frames.view(-1, *future_frames.shape[2:])
            future_frame_embeds = self.frame_encoder(flat_future).view(
                bsz, future_frames.shape[1], self.latent_dim
            )

        for step in range(rollout_steps):
            latent = self._build_latent(frame_window, action_window)
            predicted_latents.append(latent)
            predicted_frames.append(self.decoder(latent))

            if future_frame_embeds is not None:
                direct_latent = future_frame_embeds[:, step, :]
                target_latents.append(direct_latent)
            if future_action_embeds is not None:
                next_action = future_action_embeds[:, step, :]
            elif action_window:
                next_action = action_window[-1]
            else:
                next_action = torch.zeros(bsz, self.latent_dim, device=device, dtype=context_frames.dtype)

            frame_window = list(frame_window[1:]) + [latent]
            action_window = list(action_window[1:]) + [next_action]

        outputs: Dict[str, torch.Tensor] = {
            "context_latent": context_latent,
            "reconstruction": reconstructed,
        }
        if predicted_frames:
            outputs["predicted_frames"] = torch.stack(predicted_frames, dim=1)
            outputs["predicted_latents"] = torch.stack(predicted_latents, dim=1)
        if target_latents:
            outputs["target_latents"] = torch.stack(target_latents, dim=1)
        if future_frames is not None:
            outputs["target_frames"] = future_frames[:, :rollout_steps]
        return outputs


@dataclass
class LossHistory:
    steps: List[float]
    total: List[float]
    recon: List[float]
    action: List[float]
    pixel: List[float]

    @classmethod
    def empty(cls) -> "LossHistory":
        return cls([], [], [], [], [])

    def append(
        self,
        *,
        step: float,
        total: float,
        recon: float,
        action: float,
        pixel: float,
    ) -> None:
        self.steps.append(step)
        self.total.append(total)
        self.recon.append(recon)
        self.action.append(action)
        self.pixel.append(pixel)

    def __len__(self) -> int:
        return len(self.steps)

    def rows(self) -> Dict[str, List[float]]:
        return {
            "step": self.steps,
            "total": self.total,
            "recon": self.recon,
            "action": self.action,
            "pixel": self.pixel,
        }


LOSS_COLUMNS = ["step", "total", "recon", "action", "pixel"]


def write_loss_csv(history: LossHistory, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LOSS_COLUMNS)
        for row in zip(
            history.steps,
            history.total,
            history.recon,
            history.action,
            history.pixel,
        ):
            writer.writerow(row)


def plot_loss_curves(history: LossHistory, out_dir: Path) -> None:
    if len(history) == 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = history.steps

    plt.figure(figsize=(8, 5))
    plt.plot(steps, history.total, label="total")
    plt.plot(steps, history.recon, label="recon")
    plt.plot(steps, history.action, label="action")
    plt.plot(steps, history.pixel, label="pixel")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=200)
    plt.close()


def make_dataloader(cfg: TrainConfig) -> DataLoader:
    dataset = MarioSequenceDataset(
        root=cfg.data_root,
        context=cfg.context_frames,
        prediction=cfg.prediction_frames,
        image_hw=cfg.image_hw,
        max_traj=cfg.max_trajectories,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )


def visualize_rollout(
    *,
    out_path: Path,
    context_frames: torch.Tensor,
    target_frames: torch.Tensor,
    predicted_frames: torch.Tensor,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_ctx = context_frames[0].detach().cpu()
    sample_target = target_frames[0].detach().cpu()
    sample_pred = predicted_frames[0].detach().cpu()

    cols = sample_ctx.shape[0] + sample_target.shape[0]
    fig, axes = plt.subplots(2, cols, figsize=(cols * 2, 4))
    blank = torch.ones_like(sample_ctx[0])

    for idx in range(sample_ctx.shape[0]):
        axes[0, idx].imshow(sample_ctx[idx].permute(1, 2, 0).numpy())
        axes[0, idx].axis("off")
        axes[1, idx].imshow(blank.permute(1, 2, 0).numpy())
        axes[1, idx].axis("off")

    for offset in range(sample_target.shape[0]):
        col = sample_ctx.shape[0] + offset
        axes[0, col].imshow(sample_target[offset].permute(1, 2, 0).numpy())
        axes[0, col].axis("off")

        pred_img = sample_pred[offset]
        pred_img = torch.clamp(pred_img, 0.0, 1.0)
        axes[1, col].imshow(pred_img.permute(1, 2, 0).numpy())
        axes[1, col].axis("off")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_checkpoint(
    *,
    output_dir: Path,
    name: str,
    step: int,
    epoch: int,
    metric: float,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Path:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "epoch": epoch,
        "metric": metric,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    path = ckpt_dir / name
    torch.save(state, path)
    return path


def train(cfg: TrainConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger.info("Configuration: %s", cfg)
    set_seed(cfg.seed)

    device_pref = None if cfg.device == "auto" else cfg.device
    device = pick_device(device_pref)
    logger.info("Using device: %s", device)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = cfg.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", run_dir)

    dataloader = make_dataloader(cfg)
    model = MarioPredictor(action_dim=dataloader.dataset.action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    mse_loss = nn.MSELoss()
    global_step = 0
    metrics_dir = run_dir / "metrics"
    samples_root = run_dir / "samples"
    fixed_samples_dir = samples_root / "fixed"
    rolling_samples_dir = samples_root / "rolling"
    checkpoints_dir = run_dir / "checkpoints"
    for directory in (metrics_dir, fixed_samples_dir, rolling_samples_dir, checkpoints_dir):
        directory.mkdir(parents=True, exist_ok=True)

    loss_history = LossHistory.empty()
    fixed_batch_cpu: Optional[Dict[str, torch.Tensor]] = None
    best_metric = float("inf")
    last_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_recon = 0.0
        running_action = 0.0
        running_pixel = 0.0
        running_total = 0.0
        window_data_time = 0.0
        window_forward_time = 0.0
        window_backward_time = 0.0
        epoch_start = perf_counter()
        for batch_idx, batch in enumerate(dataloader, start=1):
            batch_start = perf_counter()
            if fixed_batch_cpu is None:
                fixed_batch_cpu = {
                    "context_frames": batch["context_frames"][:1].detach().cpu(),
                    "context_actions": batch["context_actions"][:1].detach().cpu(),
                    "future_frames": batch["future_frames"][:1].detach().cpu(),
                    "future_actions": batch["future_actions"][:1].detach().cpu(),
                }
            context_frames = batch["context_frames"].to(device)
            context_actions = batch["context_actions"].to(device)
            future_frames = batch["future_frames"].to(device)
            future_actions = batch["future_actions"].to(device)
            transfer_end = perf_counter()

            outputs = model(
                context_frames,
                context_actions,
                future_actions=future_actions,
                future_frames=future_frames,
            )
            forward_end = perf_counter()

            recon_loss = mse_loss(outputs["reconstruction"], context_frames[:, -1])
            action_loss = mse_loss(outputs["predicted_latents"], outputs["target_latents"])
            pixel_loss = mse_loss(outputs["predicted_frames"], outputs["target_frames"])
            loss = recon_loss + action_loss + pixel_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            backward_end = perf_counter()
            current_loss = float(loss.item())
            last_loss = current_loss

            running_recon += recon_loss.item()
            running_action += action_loss.item()
            running_pixel += pixel_loss.item()
            running_total += loss.item()
            global_step += 1

            window_data_time += transfer_end - batch_start
            window_forward_time += forward_end - transfer_end
            window_backward_time += backward_end - forward_end

            if global_step % cfg.log_every_steps == 0:
                avg_recon = running_recon / cfg.log_every_steps
                avg_action = running_action / cfg.log_every_steps
                avg_pixel = running_pixel / cfg.log_every_steps
                avg_total = running_total / cfg.log_every_steps
                avg_data = window_data_time / cfg.log_every_steps
                avg_forward = window_forward_time / cfg.log_every_steps
                avg_backward = window_backward_time / cfg.log_every_steps
                logger.info(
                    (
                        "[step %d] recon %.4f | action %.4f | pixel %.4f | total %.4f"
                        " | data %.4f s | forward %.4f s | backward %.4f s"
                    ),
                    global_step,
                    avg_recon,
                    avg_action,
                    avg_pixel,
                    avg_total,
                    avg_data,
                    avg_forward,
                    avg_backward,
                )
                loss_history.append(
                    step=float(global_step),
                    total=avg_total,
                    recon=avg_recon,
                    action=avg_action,
                    pixel=avg_pixel,
                )
                write_loss_csv(loss_history, metrics_dir / "loss.csv")
                plot_loss_curves(loss_history, metrics_dir)
                running_recon = running_action = running_pixel = running_total = 0.0
                window_data_time = window_forward_time = window_backward_time = 0.0

            if cfg.vis_every_steps > 0 and global_step % cfg.vis_every_steps == 0:
                model.eval()
                with torch.no_grad():
                    if fixed_batch_cpu is not None:
                        fixed_inputs = {
                            "context_frames": fixed_batch_cpu["context_frames"].to(device),
                            "context_actions": fixed_batch_cpu["context_actions"].to(device),
                            "future_frames": fixed_batch_cpu["future_frames"].to(device),
                            "future_actions": fixed_batch_cpu["future_actions"].to(device),
                        }
                        fixed_outputs = model(
                            fixed_inputs["context_frames"],
                            fixed_inputs["context_actions"],
                            future_actions=fixed_inputs["future_actions"],
                            future_frames=fixed_inputs["future_frames"],
                        )
                        visualize_rollout(
                            out_path=fixed_samples_dir / f"step_{global_step:07d}.png",
                            context_frames=fixed_batch_cpu["context_frames"],
                            target_frames=fixed_outputs["target_frames"].detach().cpu(),
                            predicted_frames=fixed_outputs["predicted_frames"].detach().cpu(),
                        )
                    visualize_rollout(
                        out_path=rolling_samples_dir / f"step_{global_step:07d}.png",
                        context_frames=context_frames.detach().cpu(),
                        target_frames=outputs["target_frames"].detach().cpu(),
                        predicted_frames=outputs["predicted_frames"].detach().cpu(),
                    )
                model.train()

            if cfg.checkpoint_every_steps > 0 and global_step % cfg.checkpoint_every_steps == 0:
                save_checkpoint(
                    output_dir=run_dir,
                    name="checkpoint_last.pt",
                    step=global_step,
                    epoch=epoch,
                    metric=current_loss,
                    model=model,
                    optimizer=optimizer,
                )
                if current_loss < best_metric:
                    best_metric = current_loss
                    save_checkpoint(
                        output_dir=run_dir,
                        name="checkpoint_best.pt",
                        step=global_step,
                        epoch=epoch,
                        metric=current_loss,
                        model=model,
                        optimizer=optimizer,
                    )

        epoch_duration = perf_counter() - epoch_start
        logger.info("Epoch %d finished in %.2f s (global_step=%d)", epoch, epoch_duration, global_step)
    write_loss_csv(loss_history, metrics_dir / "loss.csv")
    plot_loss_curves(loss_history, metrics_dir)
    save_checkpoint(
        output_dir=run_dir,
        name="checkpoint_last.pt",
        step=global_step,
        epoch=epoch,
        metric=last_loss,
        model=model,
        optimizer=optimizer,
    )
    if last_loss < best_metric:
        best_metric = last_loss
        save_checkpoint(
            output_dir=run_dir,
            name="checkpoint_best.pt",
            step=global_step,
            epoch=epoch,
            metric=last_loss,
            model=model,
            optimizer=optimizer,
        )
    save_checkpoint(
        output_dir=run_dir,
        name="checkpoint_final.pt",
        step=global_step,
        epoch=epoch,
        metric=last_loss,
        model=model,
        optimizer=optimizer,
    )


def main() -> None:
    cfg = tyro.cli(TrainConfig)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    main()
