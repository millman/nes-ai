#!/usr/bin/env python3
"""Train encoder/decoder only and emit rollouts + self-distance diagnostics."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import tyro
from torch.utils.data import DataLoader

from jepa_world_model_trainer import (
    Encoder,
    LossMultiScaleBoxReconConfig,
    VisualizationDecoder,
    _prepare_self_distance_inputs,
    multi_scale_recon_loss_box,
)
from jepa_world_model.data import TrajectorySequenceDataset, collate_batch
from jepa_world_model.model_config import ModelConfig
from jepa_world_model.vis_rollout import VisualizationSequence
from jepa_world_model.vis_rollout_batch import save_rollout_sequence_batch
from jepa_world_model.vis_self_distance import write_self_distance_outputs
from recon.data import list_trajectories, short_traj_state_label
from utils.device_utils import pick_device


RECON_LOSS = nn.MSELoss()


@dataclass
class AutoencoderTrainConfig:
    data_root: Path = Path("data.gridworldkey_no_obs_loops")
    output_dir: Path = Path("out.encoder_decoder_only")
    steps: int = 100_000
    batch_size: int = 8
    seq_len: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.03
    device: Optional[str] = "mps"
    seed: Optional[int] = 0
    max_trajectories: Optional[int] = None

    recon_loss: str = "multi_box"  # "mse" or "multi_box"
    recon_multi_box: LossMultiScaleBoxReconConfig = field(default_factory=LossMultiScaleBoxReconConfig)

    vis_every_steps: int = 50
    vis_rows: int = 4
    vis_columns: int = 6
    rollout_steps: int = 6

    model: ModelConfig = field(default_factory=ModelConfig)


def _serialize(value):
    if is_dataclass(value):
        return {k: _serialize(v) for k, v in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)


def _cycle(loader: Iterable):
    while True:
        for item in loader:
            yield item


def _build_sequences(
    frames: torch.Tensor,
    frame_paths: Sequence[Sequence[str]],
    *,
    encoder: nn.Module,
    decoder: nn.Module,
    rows: int,
    steps: int,
) -> List[VisualizationSequence]:
    bsz, total_steps, channels, height, width = frames.shape
    window = min(steps, total_steps)
    rows = min(rows, bsz)
    flat = frames[:, :window].reshape(-1, channels, height, width)
    with torch.no_grad():
        recon = decoder(encoder(flat)).view(bsz, window, channels, height, width).clamp(0, 1)
    sequences: List[VisualizationSequence] = []
    for row in range(rows):
        gt = frames[row, :window].detach().cpu()
        recon_row = recon[row].detach().cpu()
        rollout: List[Optional[torch.Tensor]] = []
        reencoded: List[Optional[torch.Tensor]] = []
        current = gt[0].to(frames.device)
        for _ in range(window):
            with torch.no_grad():
                pred = decoder(encoder(current.unsqueeze(0)))[0].clamp(0, 1)
            rollout.append(pred.detach().cpu())
            reencoded.append(None)
            current = pred.detach()
        labels = [
            short_traj_state_label(frame_paths[row][idx])
            if idx < len(frame_paths[row])
            else f"t={idx}"
            for idx in range(window)
        ]
        sequences.append(
            VisualizationSequence(
                ground_truth=gt,
                rollout=rollout,
                gradients=[None for _ in range(window)],
                reconstructions=recon_row,
                reencoded=reencoded,
                labels=labels,
                actions=[],
            )
        )
    return sequences


def _compute_recon_loss(
    recon: torch.Tensor,
    targets: torch.Tensor,
    cfg: AutoencoderTrainConfig,
) -> torch.Tensor:
    if cfg.recon_loss == "mse":
        return RECON_LOSS(recon, targets)
    if cfg.recon_loss == "multi_box":
        return multi_scale_recon_loss_box(recon, targets, cfg.recon_multi_box)
    raise ValueError(f"Unsupported recon_loss={cfg.recon_loss}")


def main(cfg: AutoencoderTrainConfig) -> None:
    _set_seed(cfg.seed)
    device = pick_device(cfg.device)
    run_dir = cfg.output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(_serialize(cfg), indent=2))

    image_hw = (cfg.model.image_size, cfg.model.image_size)
    dataset = TrajectorySequenceDataset(
        root=cfg.data_root,
        seq_len=cfg.seq_len,
        image_hw=image_hw,
        max_traj=cfg.max_trajectories,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_batch,
    )
    iterator = _cycle(loader)

    decoder_schedule = cfg.model.decoder_schedule or cfg.model.encoder_schedule
    encoder = Encoder(cfg.model.in_channels, cfg.model.encoder_schedule, cfg.model.image_size).to(device)
    decoder = VisualizationDecoder(
        encoder.latent_dim,
        cfg.model.in_channels,
        cfg.model.image_size,
        decoder_schedule,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    loss_csv = run_dir / "loss.csv"
    loss_csv.write_text("step,loss_recon\n")

    traj_map = list_trajectories(cfg.data_root)
    train_trajs = list(traj_map.keys())
    if cfg.max_trajectories is not None:
        train_trajs = train_trajs[: cfg.max_trajectories]
    self_distance_inputs = _prepare_self_distance_inputs(
        cfg.data_root,
        train_trajs,
        image_hw,
        run_dir,
    )
    self_distance_z_dir = run_dir / "self_distance_z"
    vis_self_distance_z_dir = run_dir / "vis_self_distance_z"
    rollout_dir = run_dir / "vis_encoder_decoder_rollout"

    for step in range(cfg.steps):
        frames, _, frame_paths, _ = next(iterator)
        frames = frames.to(device)
        bsz, seq_len, channels, height, width = frames.shape
        flat = frames.reshape(-1, channels, height, width)
        z = encoder(flat)
        recon = decoder(z).view(bsz, seq_len, channels, height, width)
        loss = _compute_recon_loss(recon, frames, cfg)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with loss_csv.open("a") as handle:
            handle.write(f"{step},{loss.item()}\n")

        if cfg.vis_every_steps > 0 and step % cfg.vis_every_steps == 0:
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                sequences = _build_sequences(
                    frames,
                    frame_paths,
                    encoder=encoder,
                    decoder=decoder,
                    rows=cfg.vis_rows,
                    steps=cfg.rollout_steps,
                )
                save_rollout_sequence_batch(
                    rollout_dir,
                    sequences,
                    grad_label="Pixel Diff",
                    global_step=step,
                    include_pixel_delta=False,
                )
                self_frames = self_distance_inputs.frames.to(device)
                flat_self = self_frames.reshape(-1, channels, height, width)
                self_embeddings = encoder(flat_self).view(1, self_frames.shape[1], -1)
                write_self_distance_outputs(
                    self_embeddings,
                    self_distance_inputs,
                    self_distance_z_dir,
                    vis_self_distance_z_dir,
                    step,
                    embedding_label="z",
                    title_prefix="Self-distance (Z)",
                    file_prefix="self_distance_z",
                    cosine_prefix="self_distance_z_cosine",
                )
            encoder.train()
            decoder.train()

        if step % 100 == 0:
            print(f"[step {step:06d}] recon_loss={loss.item():.6f}")


if __name__ == "__main__":
    main(tyro.cli(AutoencoderTrainConfig))
