#!/usr/bin/env python3
"""CLI for running JEPA encoder layer distance scan diagnostic.

This diagnostic measures how much each encoder layer's activation changes
between consecutive frames that differ by small motion (1-2 pixels).
It helps identify the layer at which small differences vanish.

Usage:
    python -m jepa_world_model.diagnostics.run_layer_scan --data_root data.gridworldkey_wander_to_key
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import tyro

from jepa_world_model.conv_encoder_decoder import Encoder
from jepa_world_model.diagnostics.layer_distance_scan import run_layer_distance_scan
from jepa_world_model_trainer import (
    ModelConfig,
    TrajectorySequenceDataset,
    collate_batch,
)
from torch.utils.data import DataLoader
from utils.device_utils import pick_device


@dataclass
class LayerScanConfig:
    """Configuration for layer distance scan diagnostic."""

    # Data
    data_root: Path = Path("data.gridworldkey_wander_to_key")
    max_trajectories: Optional[int] = None
    seq_len: int = 8
    batch_size: int = 4

    # Scan parameters
    max_pairs: int = 100

    # Output
    output_dir: Path = Path("diagnostics_output")
    save_csv: bool = True
    save_json: bool = True

    # Device
    device: Optional[str] = None

    # Model config (subset needed for encoder)
    image_size: int = 128
    embedding_dim: int = 256
    num_downsample_layers: int = 4


def main() -> None:
    cfg = tyro.cli(LayerScanConfig)

    device = pick_device(cfg.device)
    print(f"[layer_scan] Using device: {device}")

    # Build model config
    model_cfg = ModelConfig(
        image_size=cfg.image_size,
        embedding_dim=cfg.embedding_dim,
        num_downsample_layers=cfg.num_downsample_layers,
    )

    # Initialize encoder
    encoder = Encoder(
        model_cfg.in_channels,
        model_cfg.channel_schedule,
        model_cfg.image_size,
    ).to(device)
    encoder.eval()

    print(f"[layer_scan] Encoder channel schedule: {model_cfg.channel_schedule}")

    # Load dataset
    print(f"[layer_scan] Loading dataset from {cfg.data_root}")
    dataset = TrajectorySequenceDataset(
        root=cfg.data_root,
        seq_len=cfg.seq_len,
        image_hw=(cfg.image_size, cfg.image_size),
        max_traj=cfg.max_trajectories,
    )
    print(f"[layer_scan] Dataset has {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    # Run the scan
    print(f"[layer_scan] Running layer distance scan on up to {cfg.max_pairs} frame pairs...")
    result = run_layer_distance_scan(
        encoder=encoder,
        dataloader=dataloader,
        device=device,
        max_pairs=cfg.max_pairs,
    )

    # Print results
    result.print_summary()

    # Save results
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.save_csv:
        csv_path = cfg.output_dir / "layer_distance_scan.csv"
        result.save_csv(csv_path)
        print(f"[layer_scan] Saved CSV to {csv_path}")

    if cfg.save_json:
        json_path = cfg.output_dir / "layer_distance_scan.json"
        result.save_json(json_path)
        print(f"[layer_scan] Saved JSON to {json_path}")


if __name__ == "__main__":
    main()
