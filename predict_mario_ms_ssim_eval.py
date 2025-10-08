#!/usr/bin/env python3
"""Autoregressive rollout visualizer for predict_mario_ms_ssim models."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
import tyro

from predict_mario_ms_ssim import (
    Mario4to1Dataset,
    UNetPredictor,
    pick_device,
    unnormalize,
)


@dataclass
class Args:
    traj_dir: str = "data.image_distance.train_levels_1_2"
    checkpoint: str = "out.predict_mario_ms_ssim/run__2025-10-08_11-45-38/last.pt"
    out_dir: str = "out.predict_mario_ms_ssim_eval"
    num_samples: int = 10
    max_trajs: Optional[int] = None
    seed: int = 0
    device: Optional[str] = None
    save_name: Optional[str] = None


@torch.no_grad()
def rollout(model: UNetPredictor, context: torch.Tensor, steps: int) -> List[torch.Tensor]:
    """Runs autoregressive prediction starting from 4-frame context.

    Args:
        model: trained predictor in eval mode.
        context: tensor shaped (4, 3, H, W), normalized like the training data.
        steps: number of future frames to predict.
    Returns:
        List of predicted frames (each a tensor (3, H, W), normalized).
    """
    preds: List[torch.Tensor] = []
    window = context.clone()  # working buffer on same device
    height, width = context.shape[-2:]
    for _ in range(steps):
        model_input = window.reshape(1, 12, height, width)
        pred = model(model_input)[0]
        preds.append(pred)
        window = torch.cat([window[1:], pred.unsqueeze(0)], dim=0)
    return preds


def main() -> None:
    args = tyro.cli(Args)

    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive")

    device = pick_device(args.device)
    print(f"[device] using {device}")

    ds = Mario4to1Dataset(args.traj_dir, max_trajs=args.max_trajs)
    if len(ds) == 0:
        raise RuntimeError(f"No samples found in trajectory directory: {args.traj_dir}")
    eval_n = min(args.num_samples, len(ds))
    print(f"Dataset size: {len(ds)} | evaluating {eval_n} samples")

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(ds), generator=generator)[:eval_n].tolist()

    model = UNetPredictor().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    all_frames: List[torch.Tensor] = []

    for row_idx, ds_idx in enumerate(indices):
        x_stack, _ = ds[ds_idx]
        frames4 = x_stack.view(4, 3, *x_stack.shape[-2:]).to(device)
        preds = rollout(model, frames4, steps=12)
        row_frames = list(frames4.cpu()) + [p.cpu() for p in preds]
        row_tensor = torch.stack(row_frames)  # (16, 3, H, W)
        row_tensor = unnormalize(row_tensor).clamp(0.0, 1.0)
        all_frames.append(row_tensor)
        print(f"Sample {row_idx+1}/{eval_n}: dataset idx {ds_idx} -> generated {len(preds)} frames")

    stacked = torch.cat(all_frames, dim=0)
    grid = make_grid(stacked, nrow=16, padding=2)
    image = TF.to_pil_image(grid)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fname = args.save_name or f"rollouts_{timestamp}.png"
    out_path = out_dir / fname
    image.save(out_path)
    print(f"Saved rollout grid to {out_path}")


if __name__ == "__main__":
    main()
