#!/usr/bin/env python3
"""Autoregressive rollout visualizer for predict_mario_ms_ssim models."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
import torchvision.transforms.functional as TF
import tyro
from PIL import Image

from predict_mario_ms_ssim import (
    Mario4to1Dataset,
    UNetPredictor,
    pick_device,
    unnormalize,
)


@dataclass
class Args:
    traj_dir: str = "data.image_distance.train_levels_1_2"
    out_dir: str = "out.predict_mario_ms_ssim_eval"
    num_samples: int = 10
    max_trajs: Optional[int] = None
    seed: int = 0
    device: Optional[str] = None
    save_name: Optional[str] = None
    rollout_steps: int = 12

    # Trained models:
    #   out.predict_mario_ms_ssim/run__2025-10-08_11-45-38/last.pt
    #     - trained on: data.image_distance.train_levels_1_2
    #     - uses MS-SSIM loss only
    #
    #   out.predict_mario_ms_ssim/run__2025-10-08_16-11-41/last.pt
    #     - trained on: data.image_distance.train_levels_1_2
    #     - uses MS-SSIM loss and L1 loss together
    #
    #   out.predict_mario_ms_ssim/run__2025-10-08_16-44-53
    #     - trained on: data.image_distance.train_levels_1_2
    #     - uses MS-SSIM loss and L1 loss together
    #     - feeds predicted outputs as inputs during training
    #
    checkpoint: str = "out.predict_mario_ms_ssim/run__2025-10-08_16-44-53/last.pt"


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
    if args.rollout_steps < 1:
        raise ValueError("rollout_steps must be >= 1")

    device = pick_device(args.device)
    print(f"[device] using {device}")

    ds = Mario4to1Dataset(args.traj_dir, max_trajs=args.max_trajs, rollout=args.rollout_steps)
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

    rows: List[Image.Image] = []
    ctx_len = 4

    for row_idx, ds_idx in enumerate(indices):
        x_stack, targets = ds[ds_idx]
        # Context frames (4,3,H,W)
        frames4 = x_stack.view(ctx_len, 3, *x_stack.shape[-2:]).to(device)
        # Ground-truth rollout (S,3,H,W)
        targets = targets.to(device)

        preds = rollout(model, frames4, steps=args.rollout_steps)
        pred_tensor = torch.stack(preds, dim=0)

        frames4_cpu = frames4.cpu()
        targets_cpu = targets.cpu()
        preds_cpu = pred_tensor.cpu()

        def to_pil(frame: torch.Tensor) -> Image.Image:
            vis = unnormalize(frame.unsqueeze(0)).clamp(0.0, 1.0)[0]
            return TF.to_pil_image(vis)

        ctx_imgs = [to_pil(frames4_cpu[i]) for i in range(ctx_len)]
        tgt_imgs = [to_pil(targets_cpu[i]) for i in range(targets_cpu.shape[0])]
        pred_imgs = [to_pil(preds_cpu[i]) for i in range(preds_cpu.shape[0])]

        blank_tile = Image.new("RGB", ctx_imgs[0].size, (0, 0, 0))
        top_row = ctx_imgs + tgt_imgs
        bottom_row = [blank_tile] * ctx_len + pred_imgs
        cols = len(top_row)
        tile_w, tile_h = ctx_imgs[0].size
        canvas = Image.new("RGB", (tile_w * cols, tile_h * 2))
        for col, img in enumerate(top_row):
            canvas.paste(img, (col * tile_w, 0))
        for col, img in enumerate(bottom_row):
            canvas.paste(img, (col * tile_w, tile_h))

        rows.append(canvas)
        print(
            f"Sample {row_idx+1}/{eval_n}: dataset idx {ds_idx} -> predicted {len(preds)} frames"
        )

    if not rows:
        raise RuntimeError("No rows generated for visualization")

    row_gap = 4
    width = rows[0].width
    total_height = sum(row.height for row in rows) + row_gap * (len(rows) - 1)
    image = Image.new("RGB", (width, total_height), (0, 0, 0))
    y = 0
    for idx, row in enumerate(rows):
        image.paste(row, (0, y))
        y += row.height
        if idx < len(rows) - 1:
            y += row_gap

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fname = args.save_name or f"rollouts_{timestamp}.png"
    out_path = out_dir / fname
    image.save(out_path)
    print(f"Saved rollout grid to {out_path}")


if __name__ == "__main__":
    main()
