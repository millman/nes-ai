#!/usr/bin/env python3
"""Pure Conditional Flow Matching (CFM) on Mario frame sequences.

This training script learns a conditional vector field that maps Gaussian noise
to the next four frames of gameplay given the preceding four frames as context.
Unlike the autoencoder-based variants, the flow operates directly in pixel
space, so only the conditional field is trained—no encoder/decoder pair.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from contextlib import nullcontext
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torchvision.transforms as T
import torchvision.utils as vutils
import tyro

from predict_mario_ms_ssim import (
    default_transform,
    pick_device,
    unnormalize,
)
from trajectory_utils import list_state_frames, list_traj_dirs

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class MarioSequenceCFMDataset(Dataset):
    """Returns (context, target) frame sequences for pure CFM training.

    Context contains the past `context_len` frames, while target contains the
    next `target_len` frames. All frames are normalised with ImageNet stats.
    """

    def __init__(
        self,
        root_dir: str,
        *,
        context_len: int = 4,
        target_len: int = 4,
        transform: Optional[T.Compose] = None,
        max_trajs: Optional[int] = None,
    ) -> None:
        if context_len <= 0 or target_len <= 0:
            raise ValueError("context_len and target_len must be positive.")
        self.context_len = context_len
        self.target_len = target_len
        self.total_len = context_len + target_len
        self.transform = transform or default_transform()

        self.index: List[Tuple[List[Path], int]] = []
        traj_count = 0
        for traj_dir in list_traj_dirs(Path(root_dir)):
            if not traj_dir.is_dir():
                continue
            states_dir = traj_dir / "states"
            if not states_dir.is_dir():
                continue
            frames = list_state_frames(states_dir)
            if len(frames) < self.total_len:
                continue
            for start in range(len(frames) - self.total_len + 1):
                self.index.append((frames, start))
            traj_count += 1
            if max_trajs is not None and traj_count >= max_trajs:
                break
        if not self.index:
            raise RuntimeError(f"No frame sequences found under {root_dir}.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        files, start = self.index[idx]
        seq: List[torch.Tensor] = []
        for offset in range(self.total_len):
            with Image.open(files[start + offset]).convert("RGB") as img:
                seq.append(self.transform(img))
        stack = torch.stack(seq, dim=0)  # (total_len, 3, H, W)
        context = stack[: self.context_len]
        target = stack[self.context_len :]
        last_path = str(files[start + self.total_len - 1])
        return context, target, last_path


# -----------------------------------------------------------------------------
# Model components
# -----------------------------------------------------------------------------


def _make_group_norm(num_channels: int) -> nn.GroupNorm:
    groups = 8 if num_channels % 8 == 0 else 1
    return nn.GroupNorm(groups, num_channels)


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            _make_group_norm(c_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            _make_group_norm(c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.block = ConvBlock(c_in, c_out)
        self.down = nn.Conv2d(c_out, c_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.block(x)
        return h, self.down(h)


class UpBlock(nn.Module):
    def __init__(self, c_in: int, skip_c: int, c_out: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1)
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_c, c_out, kernel_size=1),
            _make_group_norm(c_out),
        )
        self.block = ConvBlock(c_out * 2, c_out)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        skip_h = self.skip_proj(skip)
        if h.shape[-2:] != skip_h.shape[-2:]:
            skip_h = F.interpolate(skip_h, size=h.shape[-2:], mode="bilinear", align_corners=False)
        h = torch.cat([h, skip_h], dim=1)
        return self.block(h)


class ConditionalVectorField(nn.Module):
    """U-Net style conditional field operating directly in pixel space."""

    def __init__(
        self,
        *,
        context_len: int,
        target_len: int,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        if base_channels % 2 != 0:
            raise ValueError("base_channels must be even.")
        self.target_channels = target_len * 3
        in_channels = context_len * 3 + self.target_channels + 1  # +1 for time conditioning

        self.enc1 = ConvBlock(in_channels, base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)

        self.mid = ConvBlock(base_channels * 4, base_channels * 4)

        self.up2 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels * 2, base_channels)

        self.head = nn.Conv2d(base_channels, self.target_channels, kernel_size=1)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if x_t.shape[0] != context.shape[0]:
            raise RuntimeError("Batch size mismatch between x_t and context.")
        if x_t.shape[-2:] != context.shape[-2:]:
            raise RuntimeError("Spatial size mismatch between x_t and context.")
        B, _, H, W = x_t.shape
        t_img = t[:, None, None, None].expand(B, 1, H, W)
        cond = torch.cat([context, x_t, t_img], dim=1)

        h1 = self.enc1(cond)
        skip1, h = self.down1(h1)
        skip2, h = self.down2(h)
        h = self.mid(h)
        h = self.up2(h, skip2)
        h = self.up1(h, skip1)
        return self.head(h)


# -----------------------------------------------------------------------------
# Visualisation helpers
# -----------------------------------------------------------------------------


def frames_to_grid(
    frames: torch.Tensor,
    *,
    num_rows: int,
    normalize: bool = False,
) -> torch.Tensor:
    """Convert a (B, T, 3, H, W) tensor to a grid for logging."""
    B, T, C, H, W = frames.shape
    flat = frames.view(B * T, C, H, W)
    kwargs = {"nrow": num_rows, "normalize": normalize}
    if normalize:
        kwargs["value_range"] = (0.0, 1.0)
    return vutils.make_grid(flat, **kwargs)


def save_sequence_grid(
    context: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    out_path: Path,
) -> None:
    """Save a comparison grid showing context, ground truth, and prediction."""
    ctx = unnormalize(context).clamp(0.0, 1.0).unsqueeze(0)
    tgt = unnormalize(target).clamp(0.0, 1.0).unsqueeze(0)
    pred = unnormalize(prediction).clamp(0.0, 1.0).unsqueeze(0)

    ctx_grid = frames_to_grid(ctx, num_rows=context.shape[0])
    tgt_grid = frames_to_grid(tgt, num_rows=target.shape[0])
    pred_grid = frames_to_grid(pred, num_rows=prediction.shape[0])

    stacked = torch.cat([ctx_grid, tgt_grid, pred_grid], dim=1)
    T.ToPILImage()(stacked.cpu()).save(out_path)


# -----------------------------------------------------------------------------
# Training CLI
# -----------------------------------------------------------------------------


@dataclass
class Args:
    traj_dir: str = "data.image_distance.train_levels_1_2"
    out_dir: str = "out.reconstruct_mario_pure_cfm"
    batch_size: int = 12
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 1000
    steps_per_epoch: int = 100
    context_len: int = 4
    target_len: int = 4
    lambda_cfm: float = 1.0
    grad_clip: Optional[float] = 1.0
    max_trajs: Optional[int] = None
    base_channels: int = 64
    num_workers: int = 0
    device: Optional[str] = None
    log_every: int = 10
    viz_every: int = 50
    viz_samples: int = 2
    flow_steps: int = 64
    seed: int = 42
    resume_checkpoint: Optional[str] = None
    checkpoint_every: int = 50


def sample_flow(
    model: ConditionalVectorField,
    context_flat: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    """Integrate the learned field from noise to data."""
    device = context_flat.device
    B, _, H, W = context_flat.shape
    target_channels = model.target_channels
    x = torch.randn(B, target_channels, H, W, device=device)
    times = torch.linspace(0.0, 1.0, steps + 1, device=device)
    for t0, t1 in zip(times[:-1], times[1:]):
        t = torch.full((B,), t0.item(), device=device)
        v = model(x, t, context_flat)
        dt = (t1 - t0).item()
        x = x + dt * v
    return x


def main() -> None:
    args = tyro.cli(Args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if args.context_len <= 0 or args.target_len <= 0:
        raise ValueError("context_len and target_len must be positive.")
    device = pick_device(args.device)
    logger.info("Using device: %s", device)

    dataset = MarioSequenceCFMDataset(
        args.traj_dir,
        context_len=args.context_len,
        target_len=args.target_len,
        max_trajs=args.max_trajs,
    )
    logger.info("Dataset loaded: %d sequences", len(dataset))

    num_samples = args.steps_per_epoch * args.batch_size
    sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    sample_ctx, sample_tgt, _ = dataset[0]
    sample_hw = sample_ctx.shape[-2:]
    logger.info("Frame size: H=%d, W=%d", sample_hw[0], sample_hw[1])

    model = ConditionalVectorField(
        context_len=args.context_len,
        target_len=args.target_len,
        base_channels=args.base_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
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
            "Start epoch %d is >= target epochs %d; nothing to train.",
            start_epoch,
            args.epochs,
        )
        torch.save(
            {
                "epoch": start_epoch,
                "step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss_hist": loss_hist,
                "best_metric": best_metric,
            },
            run_dir / "final.pt",
        )
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

    device_type = device.type
    use_amp = device_type == "cuda" and hasattr(torch.amp, "GradScaler")
    scaler = torch.amp.GradScaler() if use_amp else None

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        for context, target, _ in loader:
            context = context.to(device)  # (B, T_ctx, 3, H, W)
            target = target.to(device)  # (B, T_tgt, 3, H, W)
            B, T_ctx, _, H, W = context.shape
            T_tgt = target.shape[1]

            context_flat = context.view(B, T_ctx * 3, H, W)
            target_flat = target.view(B, T_tgt * 3, H, W)
            base = torch.randn_like(target_flat)
            t = torch.rand(B, device=device)
            x_t = (1.0 - t)[:, None, None, None] * base + t[:, None, None, None] * target_flat

            autocast_ctx = (
                torch.amp.autocast(device_type="cuda") if use_amp else nullcontext()
            )
            with autocast_ctx:
                v_pred = model(x_t, t, context_flat)
                v_target = target_flat - base
                cfm_loss = F.mse_loss(v_pred, v_target)
                loss = args.lambda_cfm * cfm_loss
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.grad_clip is not None and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if args.grad_clip is not None and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            total_value = float(loss.item())
            loss_hist.append((global_step, total_value))

            if global_step % args.log_every == 0:
                v_pred_std = float(v_pred.detach().std(unbiased=False).item())
                logger.info(
                    "[epoch %03d | step %06d] loss=%.4f cfm=%.4f v_pred_std=%.3f",
                    epoch,
                    global_step,
                    total_value,
                    float(cfm_loss.item()),
                    v_pred_std,
                )

            if args.viz_every > 0 and global_step % args.viz_every == 0:
                model.eval()
                with torch.no_grad():
                    sample_idx = random.sample(
                        range(len(dataset)),
                        k=min(args.viz_samples, len(dataset)),
                    )
                    ctx_list: List[torch.Tensor] = []
                    tgt_list: List[torch.Tensor] = []
                    for idx in sample_idx:
                        ctx, tgt, _ = dataset[idx]
                        ctx_list.append(ctx)
                        tgt_list.append(tgt)
                    ctx_batch = torch.stack(ctx_list, dim=0).to(device)
                    tgt_batch = torch.stack(tgt_list, dim=0).to(device)
                    ctx_flat = ctx_batch.view(ctx_batch.size(0), -1, H, W)
                    pred_flat = sample_flow(
                        model,
                        ctx_flat,
                        steps=args.flow_steps,
                    )
                    pred = pred_flat.view(ctx_batch.size(0), T_tgt, 3, H, W)
                for i in range(ctx_batch.size(0)):
                    out_path = samples_dir / f"sample_step_{global_step:06d}_idx_{i}.png"
                    save_sequence_grid(
                        ctx_batch[i].cpu(),
                        tgt_batch[i].cpu(),
                        pred[i].cpu(),
                        out_path,
                    )
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
        sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=False,
        )

    torch.save(
        {
            "epoch": final_epoch,
            "step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss_hist": loss_hist,
            "best_metric": best_metric,
        },
        run_dir / "final.pt",
    )
    logger.info("Training finished. Artifacts written to %s", run_dir)


if __name__ == "__main__":
    main()
