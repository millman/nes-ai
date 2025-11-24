#!/usr/bin/env python3
"""Quick diagnostics for ``predict_mario_multi`` checkpoints."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tyro

from predict_mario_multi import (
    MultiHeadPredictor,
    save_multi_samples,
    compute_high_freq_energy,
    latent_summary,
)
from predict_mario_ms_ssim import (
    pick_device,
    Mario4to1Dataset,
    ms_ssim_loss,
)


@dataclass
class Args:
    traj_dir: str = "data.image_distance.train_levels_1_2"
    checkpoint: Optional[str] = None
    batch_size: int = 16
    rollout_steps: int = 4
    max_batches: int = 4
    max_trajs: Optional[int] = 256
    num_workers: int = 0
    device: Optional[str] = None
    latent_dim: int = 256
    out_dir: str = "out.predict_mario_multi_debug"
    save_visuals: bool = True
    visual_batches: int = 1
    low_freq_ratio: float = 0.25
    compute_frequency_energy: bool = True


def main() -> None:
    args = tyro.cli(Args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("predict_mario_multi.debug")
    device = pick_device(args.device)
    logger.info("Using device: %s", device)

    dataset = Mario4to1Dataset(args.traj_dir, max_trajs=args.max_trajs, rollout=args.rollout_steps)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty; check traj_dir/max_trajs settings")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = MultiHeadPredictor(args.rollout_steps, args.latent_dim).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys in loaded state_dict: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys in loaded state_dict: %s", unexpected)
        logger.info("Loaded checkpoint: %s", args.checkpoint)
    model.eval()

    out_dir = Path(args.out_dir)
    if args.save_visuals:
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    totals: Dict[str, float] = {
        "recon_ms": 0.0,
        "recon_l1": 0.0,
        "pred_ms": 0.0,
        "pred_l1": 0.0,
    }
    freq_metrics: List[Dict[str, float]] = []
    latent_tracks: Dict[str, List[float]] = {key: [] for key in ("mean", "std", "min", "max")}
    batches_evaluated = 0

    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(loader):
            if batch_idx >= args.max_batches:
                break
            xb = xb.to(device)
            yb = yb.to(device)
            B = xb.shape[0]
            context = xb.view(B, 4, 3, xb.shape[-2], xb.shape[-1])

            recon_pred, preds, latent = model(xb)
            preds = preds[:, : yb.shape[1]]
            recon_target = context[:, -1]

            recon_ms = ms_ssim_loss(recon_pred, recon_target).item()
            recon_l1 = F.l1_loss(recon_pred, recon_target).item()
            totals["recon_ms"] += recon_ms
            totals["recon_l1"] += recon_l1

            pred_ms_terms = []
            pred_l1_terms = []
            for step_idx in range(yb.shape[1]):
                pred_ms_terms.append(ms_ssim_loss(preds[:, step_idx], yb[:, step_idx]).item())
                pred_l1_terms.append(F.l1_loss(preds[:, step_idx], yb[:, step_idx]).item())
            pred_ms_avg = sum(pred_ms_terms) / len(pred_ms_terms)
            pred_l1_avg = sum(pred_l1_terms) / len(pred_l1_terms)
            totals["pred_ms"] += pred_ms_avg
            totals["pred_l1"] += pred_l1_avg

            stats = latent_summary(latent)
            for key, value in stats.items():
                latent_tracks[key].append(value)

            if args.compute_frequency_energy:
                recon_hf = compute_high_freq_energy(recon_pred, args.low_freq_ratio)
                recon_target_hf = compute_high_freq_energy(recon_target, args.low_freq_ratio)
                preds_view = preds.reshape(-1, preds.size(-3), preds.size(-2), preds.size(-1))
                targets_view = yb.reshape(-1, yb.size(-3), yb.size(-2), yb.size(-1))
                pred_hf = compute_high_freq_energy(preds_view, args.low_freq_ratio)
                rollout_hf = compute_high_freq_energy(targets_view, args.low_freq_ratio)
                freq_metrics.append(
                    {
                        "recon": recon_hf,
                        "recon_target": recon_target_hf,
                        "pred": pred_hf,
                        "target": rollout_hf,
                    }
                )

            if args.save_visuals and batch_idx < args.visual_batches:
                save_multi_samples(
                    context.cpu(),
                    recon_pred.cpu(),
                    preds.cpu(),
                    yb.cpu(),
                    out_dir / "samples",
                    batch_idx,
                )

            batches_evaluated += 1

    if batches_evaluated == 0:
        logger.warning("No batches evaluated; increase max_batches or check dataset.")
        return

    for key in totals:
        totals[key] /= batches_evaluated

    logger.info(
        "Averages over %d batches | recon(ms=%.4f, l1=%.4f) | pred(ms=%.4f, l1=%.4f)",
        batches_evaluated,
        totals["recon_ms"],
        totals["recon_l1"],
        totals["pred_ms"],
        totals["pred_l1"],
    )

    latent_means = {key: (mean(vals) if vals else float("nan")) for key, vals in latent_tracks.items()}
    logger.info("Latent stats averages | %s", ", ".join(f"{k}={v:.4f}" for k, v in latent_means.items()))

    if freq_metrics:
        recon_hf_avg = mean(item["recon"] for item in freq_metrics)
        target_hf_avg = mean(item["recon_target"] for item in freq_metrics)
        pred_hf_avg = mean(item["pred"] for item in freq_metrics)
        rollout_hf_avg = mean(item["target"] for item in freq_metrics)
        logger.info(
            "High-frequency energy | recon=%.4f target=%.4f | pred=%.4f rollout=%.4f",
            recon_hf_avg,
            target_hf_avg,
            pred_hf_avg,
            rollout_hf_avg,
        )


if __name__ == "__main__":
    main()
