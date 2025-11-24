#!/usr/bin/env python3
"""Action-conditioned Mario frame predictor with simple ConvLSTM core.

Given four context frames and their actions, the model predicts the next four
frames while maintaining an internal latent state suitable for rolling out
future frames autoregressively.
"""
from __future__ import annotations

import csv
import logging
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torchvision.transforms as T
import tyro
from torch.optim import Optimizer

from predict_mario_ms_ssim import default_transform, pick_device, unnormalize
from trajectory_utils import list_state_frames, list_traj_dirs

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

class MarioActionFutureDataset(Dataset):
    """Return context frames/actions and future rollouts from Mario trajectories."""

    def __init__(
        self,
        root_dir: str,
        *,
        context: int = 4,
        future: int = 4,
        transform: Optional[T.Compose] = None,
        max_trajs: Optional[int] = None,
    ) -> None:
        if context < 1 or future < 1:
            raise ValueError("context and future must be positive.")
        self.root_dir = Path(root_dir)
        self.context = context
        self.future = future
        self.transform = transform or default_transform()
        self.entries: List[Tuple[List[Path], np.ndarray, int]] = []
        self.traj_count = 0
        self.action_dim: Optional[int] = None

        trajs = list(list_traj_dirs(self.root_dir))
        if max_trajs is not None:
            trajs = trajs[:max_trajs]

        for traj in trajs:
            if not traj.is_dir():
                continue
            states_dir = traj / "states"
            actions_path = traj / "actions.npz"
            if not states_dir.is_dir() or not actions_path.is_file():
                continue
            frames = list_state_frames(states_dir)
            min_len = context + future
            if len(frames) < min_len:
                continue
            with np.load(actions_path) as data:
                if "actions" in data.files:
                    actions = data["actions"]
                elif len(data.files) == 1:
                    actions = data[data.files[0]]
                else:
                    raise ValueError(f"Missing 'actions' array in {actions_path}")
            if actions.ndim == 1:
                actions = actions[:, None]
            if actions.shape[0] == len(frames) + 1:
                actions = actions[1:]
            if actions.shape[0] < len(frames):
                raise ValueError(
                    f"Trajectory {traj} has {len(frames)} frames but only {actions.shape[0]} actions."
                )
            actions = actions.astype(np.float32)
            if self.action_dim is None:
                self.action_dim = actions.shape[1]
            elif actions.shape[1] != self.action_dim:
                raise ValueError(
                    f"Inconsistent action dimensionality in {traj}: expected {self.action_dim}, got {actions.shape[1]}"
                )
            limit = len(frames) - (min_len - 1)
            for start in range(limit):
                self.entries.append((frames, actions, start))
            self.traj_count += 1

        if not self.entries:
            raise RuntimeError(f"No usable trajectories found under {self.root_dir}")
        if self.action_dim is None:
            raise RuntimeError("Failed to infer action dimensionality.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        frame_paths, actions, offset = self.entries[idx]
        ctx_frames: List[torch.Tensor] = []
        for i in range(self.context):
            with Image.open(frame_paths[offset + i]).convert("RGB") as img:
                ctx_frames.append(self.transform(img))
        context = torch.stack(ctx_frames, dim=0)  # (context, 3, H, W)

        fut_frames: List[torch.Tensor] = []
        for j in range(self.future):
            with Image.open(frame_paths[offset + self.context + j]).convert("RGB") as img:
                fut_frames.append(self.transform(img))
        targets = torch.stack(fut_frames, dim=0)  # (future, 3, H, W)

        ctx_actions = actions[offset : offset + self.context]
        fut_actions = actions[offset + self.context - 1 : offset + self.context - 1 + self.future]

        ctx_actions_t = torch.from_numpy(ctx_actions)
        fut_actions_t = torch.from_numpy(fut_actions)
        return context, ctx_actions_t, fut_actions_t, targets


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class ConvLSTMCell(nn.Module):
    """Minimal ConvLSTM cell for spatial hidden states."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            h = torch.zeros(
                x.size(0), self.hidden_dim, x.size(-2), x.size(-1), device=x.device, dtype=x.dtype
            )
            c = torch.zeros_like(h)
        else:
            h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class FrameEncoder(nn.Module):
    def __init__(self, latent_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        return self.net(frame)


class FrameDecoder(nn.Module):
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels // 2, hidden_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 4, 3, kernel_size=3, padding=1),
        )

    def forward(self, state: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        out = self.net(state)
        if out.shape[-2:] != target_hw:
            out = F.interpolate(out, size=target_hw, mode="bilinear", align_corners=False)
        return out


class ActionConditionedPredictor(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        latent_channels: int = 96,
        hidden_channels: int = 128,
        action_channels: int = 32,
    ) -> None:
        super().__init__()
        self.encoder = FrameEncoder(latent_channels)
        self.action_embed = nn.Linear(action_dim, action_channels)
        self.cell = ConvLSTMCell(latent_channels + action_channels, hidden_channels)
        self.decoder = FrameDecoder(hidden_channels)

    def _step(
        self,
        frame: torch.Tensor,
        action: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        encoded = self.encoder(frame)
        act_vec = self.action_embed(action)
        act_map = act_vec.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, encoded.shape[-2], encoded.shape[-1])
        cell_input = torch.cat([encoded, act_map], dim=1)
        next_state = self.cell(cell_input, state)
        return next_state, encoded

    def forward(
        self,
        context_frames: torch.Tensor,
        context_actions: torch.Tensor,
        future_actions: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        *,
        teacher_forcing: bool = True,
    ) -> torch.Tensor:
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        current = context_frames[:, 0]
        for idx in range(context_frames.shape[1] - 1):
            action = context_actions[:, idx]
            state, _ = self._step(current, action, state)
            current = context_frames[:, idx + 1]

        preds: List[torch.Tensor] = []
        target_hw = context_frames.shape[-2:]
        for step in range(future_actions.shape[1]):
            action = future_actions[:, step]
            state, _ = self._step(current, action, state)
            pred = self.decoder(state[0], target_hw)
            preds.append(pred)
            if teacher_forcing and targets is not None:
                current = targets[:, step]
            else:
                current = pred.detach()
        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def rollout(
        self,
        context_frames: torch.Tensor,
        context_actions: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(
            context_frames,
            context_actions,
            future_actions,
            targets=None,
            teacher_forcing=False,
        )


# -----------------------------------------------------------------------------
# Logging utilities
# -----------------------------------------------------------------------------

METRIC_COLUMNS = ["step", "l1_loss"]


def write_metrics_csv(history: Sequence[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(METRIC_COLUMNS)
        for record in history:
            writer.writerow([record.get("step", float("nan")), record.get("l1_loss", float("nan"))])


def plot_metrics(history: Sequence[Dict[str, float]], out_dir: Path) -> None:
    if not history:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = [record["step"] for record in history]
    losses = [record["l1_loss"] for record in history]
    plt.figure(figsize=(6, 4))
    plt.semilogy(steps, losses, label="L1 loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=200)
    plt.close()


def ensure_run_dirs(base: Path, resume: bool) -> Dict[str, Path]:
    base.mkdir(parents=True, exist_ok=True)
    if resume:
        existing = sorted(
            [p for p in base.iterdir() if p.is_dir() and p.name[:4].isdigit()],
            reverse=True,
        )
        run_dir = existing[0] if existing else base / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        run_dir = base / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    return {
        "run": run_dir,
        "checkpoints": ckpt_dir,
        "metrics": metrics_dir,
        "samples": samples_dir,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    step: int,
    best_loss: float,
    history: Sequence[Dict[str, float]],
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_loss": best_loss,
        "history": list(history),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
) -> Tuple[int, int, float, List[Dict[str, float]]]:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return (
        int(payload.get("epoch", 0)),
        int(payload.get("step", 0)),
        float(payload.get("best_loss", float("inf"))),
        list(payload.get("history", [])),
    )


def save_rollout_grid(
    context: torch.Tensor,
    targets: torch.Tensor,
    preds: torch.Tensor,
    out_dir: Path,
    step: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    to_pil = T.ToPILImage()
    def _to_image(frame: torch.Tensor) -> Image.Image:
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        if frame.dim() != 4:
            raise ValueError(f"Expected 3D or 4D tensor, got {frame.shape}")
        return to_pil(unnormalize(frame)[0].cpu()).convert("RGB")
    rows = min(context.shape[0], 4)
    for idx in range(rows):
        ctx_frames = [_to_image(context[idx, j]) for j in range(context.shape[1])]
        tgt_frames = [_to_image(targets[idx, j]) for j in range(targets.shape[1])]
        pred_frames = [_to_image(preds[idx, j]) for j in range(preds.shape[1])]
        tile_w, tile_h = ctx_frames[0].size
        blank = Image.new("RGB", (tile_w, tile_h), (0, 0, 0))
        canvas = Image.new("RGB", (tile_w * (context.shape[1] + targets.shape[1]), tile_h * 2))
        for col in range(context.shape[1]):
            canvas.paste(ctx_frames[col], (col * tile_w, 0))
        for col in range(targets.shape[1]):
            canvas.paste(tgt_frames[col], ((context.shape[1] + col) * tile_w, 0))
        for col in range(context.shape[1]):
            canvas.paste(blank, (col * tile_w, tile_h))
        for col in range(preds.shape[1]):
            canvas.paste(pred_frames[col], ((context.shape[1] + col) * tile_w, tile_h))
        canvas.save(out_dir / f"rollout_step_{step:06d}_idx_{idx}.png")


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

@dataclass
class Args:
    traj_dir: str = "data.image_distance.train_levels_1_2"
    out_dir: str = "out.predict_mario_simple_codex"
    batch_size: int = 16
    lr: float = 1e-4
    epochs: int = 200
    steps_per_epoch: int = 100
    num_workers: int = 0
    device: Optional[str] = None
    max_trajs: Optional[int] = None
    context: int = 4
    future: int = 4
    latent_channels: int = 96
    hidden_channels: int = 128
    action_channels: int = 32
    checkpoint_every: int = 200
    sample_every: int = 200
    metrics_every: int = 100
    log_every: int = 20
    resume: bool = False


def main() -> None:
    args = tyro.cli(Args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    base_out = Path(args.out_dir)
    dirs = ensure_run_dirs(base_out, args.resume)
    run_dir = dirs["run"]
    logger.info("Writing outputs to %s", run_dir)

    device = pick_device(args.device)
    logger.info("Using device: %s", device)

    dataset = MarioActionFutureDataset(
        args.traj_dir,
        context=args.context,
        future=args.future,
        max_trajs=args.max_trajs,
    )
    logger.info("Loaded dataset with %d samples from %d trajectories.", len(dataset), dataset.traj_count)

    sampler = RandomSampler(
        dataset,
        replacement=False,
        num_samples=min(len(dataset), args.steps_per_epoch * args.batch_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = ActionConditionedPredictor(
        action_dim=dataset.action_dim,  # type: ignore[arg-type]
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        action_channels=args.action_channels,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    metrics_history: List[Dict[str, float]] = []
    if args.resume:
        last_ckpt = dirs["checkpoints"] / "last.pt"
        if last_ckpt.exists():
            ckpt_epoch, ckpt_step, ckpt_best, ckpt_history = load_checkpoint(last_ckpt, model, optimizer)
            start_epoch = ckpt_epoch + 1
            global_step = ckpt_step
            best_loss = ckpt_best
            metrics_history = ckpt_history
            logger.info(
                "Resumed from %s (epoch %d, step %d, best_loss %.4f)",
                last_ckpt,
                ckpt_epoch,
                ckpt_step,
                best_loss,
            )
        else:
            logger.info("Resume requested but no checkpoint found at %s", last_ckpt)

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0
        last_epoch = epoch
        for batch_idx, batch in enumerate(loader, start=1):
            ctx_frames, ctx_actions, fut_actions, targets = batch
            ctx_frames = ctx_frames.to(device)
            ctx_actions = ctx_actions.to(device)
            fut_actions = fut_actions.to(device)
            targets = targets.to(device)

            preds = model(
                ctx_frames,
                ctx_actions,
                fut_actions,
                targets,
                teacher_forcing=True,
            )
            loss = F.l1_loss(preds, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            current_step = global_step + 1
            loss_value = float(loss.detach().cpu().item())
            metrics_history.append({"step": current_step, "l1_loss": loss_value})
            epoch_loss += loss_value
            batches += 1

            if args.log_every > 0 and current_step % args.log_every == 0:
                logger.info(
                    "Epoch %d step %d/%d - L1 loss: %.4f",
                    epoch + 1,
                    batch_idx,
                    args.steps_per_epoch,
                    loss_value,
                )

            if args.metrics_every > 0 and current_step % args.metrics_every == 0:
                write_metrics_csv(metrics_history, dirs["metrics"] / "training_metrics.csv")
                plot_metrics(metrics_history, dirs["metrics"])

            if args.sample_every > 0 and current_step % args.sample_every == 0:
                model.eval()
                with torch.no_grad():
                    rollout_preds = model.rollout(ctx_frames[:4], ctx_actions[:4], fut_actions[:4])
                save_rollout_grid(
                    ctx_frames[:4].detach().cpu(),
                    targets[:4].detach().cpu(),
                    rollout_preds.cpu(),
                    dirs["samples"],
                    current_step,
                )
                model.train()

            if args.checkpoint_every > 0 and current_step % args.checkpoint_every == 0:
                save_checkpoint(
                    dirs["checkpoints"] / "last.pt",
                    model,
                    optimizer,
                    epoch,
                    current_step,
                    best_loss,
                    metrics_history,
                )

            global_step = current_step

            if batch_idx >= args.steps_per_epoch:
                break

        if batches == 0:
            continue
        avg_loss = epoch_loss / batches
        logger.info("Epoch %d complete | avg L1 loss %.4f", epoch + 1, avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                dirs["checkpoints"] / "best.pt",
                model,
                optimizer,
                epoch,
                global_step,
                best_loss,
                metrics_history,
            )
            logger.info("Saved new best checkpoint (avg loss %.4f)", best_loss)
        save_checkpoint(
            dirs["checkpoints"] / "last.pt",
            model,
            optimizer,
            epoch,
            global_step,
            best_loss,
            metrics_history,
        )

    final_epoch = max(last_epoch, 0)
    save_checkpoint(
        dirs["checkpoints"] / "final.pt",
        model,
        optimizer,
        final_epoch,
        global_step,
        best_loss,
        metrics_history,
    )
    write_metrics_csv(metrics_history, dirs["metrics"] / "training_metrics.csv")
    plot_metrics(metrics_history, dirs["metrics"])
    torch.save(model.state_dict(), run_dir / "model_final.pt")
    config_payload = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    config_payload["run_dir"] = str(run_dir)
    with (run_dir / "config.json").open("w") as fp:
        json.dump(config_payload, fp, indent=2)
    logger.info("Training complete; artifacts stored in %s", run_dir)


if __name__ == "__main__":
    main()
