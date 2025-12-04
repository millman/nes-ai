#!/usr/bin/env python3
"""Training loop for the JEPA world model with local/global specialization and SIGReg."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import math
import random

import csv
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from datetime import datetime

from recon.data import list_trajectories, load_frame_as_tensor
from nes_controller import _describe_controller_vector, _describe_controller_vector_compact
from utils.device_utils import pick_device
from jepa_world_model.loss import HardnessWeightedL1Loss, HardnessWeightedMSELoss, HardnessWeightedMedianLoss
from jepa_world_model.metadata import write_run_metadata
from jepa_world_model.vis import (
    describe_action_tensor,
    save_embedding_projection,
    save_input_batch_visualization,
    save_rollout_sequence_batch,
    save_rollout_visualization,
    save_temporal_pair_visualization,
)
from jepa_world_model.vis_hard_samples import save_hard_example_grid


# ------------------------------------------------------------
# Model components
# ------------------------------------------------------------


def _norm_groups(out_ch: int) -> int:
    return max(1, out_ch // 8)


class DownBlock(nn.Module):
    """Conv block with stride-2 contraction followed by local refinement."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        groups = _norm_groups(out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    """ConvTranspose block that mirrors DownBlock with stride-2 expansion."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        groups = _norm_groups(out_ch)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, schedule: Tuple[int, ...], input_hw: int):
        super().__init__()
        blocks: List[nn.Module] = []
        ch_prev = in_channels
        for ch in schedule:
            blocks.append(DownBlock(ch_prev, ch))
            ch_prev = ch
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.in_channels = in_channels
        self.channel_schedule = schedule
        self.input_hw = input_hw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.blocks(x)
        return self.pool(feats).flatten(1)

    def shape_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "module": "Encoder",
            "input": (self.input_hw, self.input_hw, self.in_channels),
            "stages": [],
        }
        h = self.input_hw
        w = self.input_hw
        c = self.in_channels
        for idx, out_ch in enumerate(self.channel_schedule, start=1):
            next_h = max(1, h // 2)
            next_w = max(1, w // 2)
            stage = {
                "stage": idx,
                "in": (h, w, c),
                "out": (next_h, next_w, out_ch),
            }
            info["stages"].append(stage)
            h, w, c = next_h, next_w, out_ch
        info["latent_dim"] = self.channel_schedule[-1] if self.channel_schedule else c
        return info


class PredictorNetwork(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, action_dim: int, film_layers: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embedding_dim, hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, embedding_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.activation = nn.SiLU(inplace=True)
        self.action_embed = ActionEmbedding(action_dim, hidden_dim)
        self.action_embed_dim = hidden_dim
        self.film = ActionFiLM(hidden_dim, hidden_dim, film_layers)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.film_layers = film_layers

    def forward(self, embeddings: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if embeddings.shape[:-1] != actions.shape[:-1]:
            raise ValueError("Embeddings and actions must share leading dimensions for predictor conditioning.")
        original_shape = embeddings.shape[:-1]
        emb_flat = embeddings.reshape(-1, embeddings.shape[-1])
        act_embed = self.action_embed(actions).reshape(-1, self.action_embed_dim)
        hidden = self.activation(self.in_proj(emb_flat))
        hidden = self.film(hidden, act_embed)
        hidden = self.activation(self.hidden_proj(hidden))
        hidden = self.film(hidden, act_embed)
        delta = self.out_proj(hidden)
        action_logits = self.action_head(hidden)
        return delta.view(*original_shape, delta.shape[-1]), action_logits.view(*original_shape, action_logits.shape[-1])

    def shape_info(self) -> Dict[str, Any]:
        return {
            "module": "Predictor",
            "latent_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "action_dim": self.action_dim,
            "film_layers": self.film_layers,
        }


class ActionEmbedding(nn.Module):
    """Embed controller actions for FiLM modulation."""

    def __init__(self, action_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        original_shape = actions.shape[:-1]
        flat = actions.reshape(-1, actions.shape[-1])
        embedded = self.net(flat)
        return embedded.view(*original_shape, embedded.shape[-1])


class FiLMLayer(nn.Module):
    """Single FiLM modulation layer."""

    def __init__(self, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(action_dim, hidden_dim)
        self.beta = nn.Linear(action_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: torch.Tensor, action_embed: torch.Tensor) -> torch.Tensor:
        conditioned = self.norm(hidden)
        gamma = self.gamma(action_embed)
        beta = self.beta(action_embed)
        return conditioned * (1.0 + gamma) + beta


class ActionFiLM(nn.Module):
    """Stack multiple FiLM layers for action conditioning."""

    def __init__(self, hidden_dim: int, action_dim: int, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FiLMLayer(hidden_dim, action_dim) for _ in range(layers)])
        self.act = nn.GELU()

    def forward(self, hidden: torch.Tensor, action_embed: torch.Tensor) -> torch.Tensor:
        out = hidden
        for layer in self.layers:
            out = layer(out, action_embed)
            out = self.act(out)
        return out


class VisualizationDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        image_channels: int,
        image_size: int,
        channel_schedule: Tuple[int, ...],
        latent_hw: int,
    ) -> None:
        super().__init__()
        if not channel_schedule:
            raise ValueError("channel_schedule must be non-empty for decoder construction.")
        stages = max(len(channel_schedule) - 1, 0)
        self.start_hw = max(1, latent_hw)
        self.start_ch = channel_schedule[-1]
        self.image_channels = image_channels
        self.image_size = image_size

        self.project = nn.Sequential(
            nn.Linear(latent_dim, self.start_ch * self.start_hw * self.start_hw),
            nn.SiLU(inplace=True),
        )
        up_blocks: List[nn.Module] = []
        in_ch = self.start_ch
        for out_ch in reversed(channel_schedule[:-1]):
            up_blocks.append(UpBlock(in_ch, out_ch))
            in_ch = out_ch
        self.up_blocks = nn.Sequential(*up_blocks)
        head_hidden = max(in_ch // 2, 1)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_hidden, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(head_hidden, image_channels, kernel_size=1),
        )
        self.channel_schedule = channel_schedule
        self.latent_hw = latent_hw

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        original_shape = latent.shape[:-1]
        latent = latent.reshape(-1, latent.shape[-1])
        projected = self.project(latent)
        decoded = projected.view(-1, self.start_ch, self.start_hw, self.start_hw)
        decoded = self.up_blocks(decoded)
        frame = self.head(decoded)
        if frame.shape[-1] != self.image_size:
            frame = F.interpolate(frame, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        frame = frame.view(*original_shape, self.image_channels, self.image_size, self.image_size)
        return frame

    def shape_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "module": "Decoder",
            "projection": (self.start_hw, self.start_hw, self.start_ch),
            "upsample": [],
            "final_target": (self.image_size, self.image_size, self.image_channels),
        }
        hw = self.start_hw
        ch = self.start_ch
        for idx, out_ch in enumerate(reversed(self.channel_schedule[:-1]), start=1):
            prev_hw = hw
            hw = max(1, hw * 2)
            stage = {
                "stage": idx,
                "in": (prev_hw, prev_hw, ch),
                "out": (hw, hw, out_ch),
            }
            info["upsample"].append(stage)
            ch = out_ch
        info["pre_resize"] = (hw, hw, self.image_channels)
        info["needs_resize"] = hw != self.image_size
        return info


RECON_LOSS = HardnessWeightedL1Loss()
JEPA_LOSS = nn.MSELoss()
SIGREG_LOSS = nn.MSELoss()
ACTION_RECON_LOSS = nn.BCEWithLogitsLoss()
EMA_CONSISTENCY_LOSS = nn.MSELoss()


# ------------------------------------------------------------
# Configs and containers
# ------------------------------------------------------------


def _derive_channel_schedule(embedding_dim: int, num_layers: int) -> Tuple[int, ...]:
    if num_layers < 1:
        raise ValueError("num_downsample_layers must be positive.")
    factor = 2 ** (num_layers - 1)
    if embedding_dim % factor != 0:
        raise ValueError("embedding_dim must be divisible by 2^(num_downsample_layers - 1) for automatic schedule.")
    base_channels = max(1, embedding_dim // factor)
    schedule: List[int] = []
    current = base_channels
    for _ in range(num_layers):
        schedule.append(current)
        current *= 2
    schedule[-1] = embedding_dim
    return tuple(schedule)


@dataclass
class ModelConfig:
    """Dimensional flow (bottlenecks marked with *):

        image (image_size^2·in_channels)
            └─ Encoder(channel_schedule → embedding_dim*)
            ▼
        embedding_dim*  ────────────────┐
                        ├─ FiLM(action_dim) → Predictor(hidden_dim*) → embedding_dim
        action_dim ──────┘
            │
            └→ VisualizationDecoder(latent_dim* → image_size^2·in_channels)

    • image_size controls the spatial resolution feeding the encoder.
    • embedding_dim (≈ channel_schedule[-1]) is the latent bottleneck the predictor must match.
    • action_dim defines the controller space that modulates the predictor through FiLM layers.
    • hidden_dim is the predictor’s internal width.
    • latent_dim governs the visualization decoder’s compression when turning embeddings back into images.
    """

    in_channels: int = 3
    image_size: int = 224
    latent_dim: int = 256
    hidden_dim: int = 512
    embedding_dim: int = 256
    num_downsample_layers: int = 4
    latent_hw: int = 16
    action_dim: int = 8
    predictor_film_layers: int = 2
    channel_schedule: Tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.channel_schedule = _derive_channel_schedule(self.embedding_dim, self.num_downsample_layers)


@dataclass
class LossWeights:
    jepa: float = 1.0
    sigreg: float = 0.5
    rollout: float = 0.0
    consistency: float = 0.0
    ema_consistency: float = 1.0
    action_recon: float = 1.0


@dataclass
class DebugVisualization:
    input_vis_every_steps: int = 0
    input_vis_rows: int = 4
    pair_vis_every_steps: int = 0
    pair_vis_rows: int = 4


@dataclass
class TrainConfig:
    seq_len: int = 8
    batch_size: int = 8
    lr: float = 1e-4
    sigreg_projections: int = 64
    recon_weight: float = 1.0
    device: Optional[str] = "mps"
    total_steps: int = 100_000
    log_every_steps: int = 10
    vis_every_steps: int = 50
    vis_rows: int = 2
    vis_rollout: int = 4
    vis_columns: int = 8
    vis_gradient_norms: bool = True
    vis_log_deltas: bool = False
    rollout_horizon: int = 8
    ema_momentum: float = 0.99
    embedding_projection_samples: int = 256
    output_dir: Path = Path("out.jepa_world_model_trainer")
    data_root: Path = Path("data.gridworldkey_wander_to_key")
    max_trajectories: Optional[int] = None
    hard_example_reservoir: int = 256
    hard_example_mix_ratio: float = 0.5
    hard_example_vis_rows: int = 6
    hard_example_vis_columns: int = 8

    loss_weights: LossWeights = field(default_factory=LossWeights)
    debug_visualization: DebugVisualization = field(default_factory=DebugVisualization)


@dataclass
class VisualizationSelection:
    row_indices: torch.Tensor
    time_indices: torch.Tensor


@dataclass
class VisualizationSequence:
    ground_truth: torch.Tensor
    rollout: List[Optional[torch.Tensor]]
    gradients: List[Optional[np.ndarray]]
    reconstructions: torch.Tensor
    labels: List[str]
    actions: List[str] = field(default_factory=list)


class JEPAWorldModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.in_channels, cfg.channel_schedule, cfg.image_size)
        enc_dim = cfg.channel_schedule[-1]
        self.embedding_dim = enc_dim
        self.predictor = PredictorNetwork(
            self.embedding_dim,
            cfg.hidden_dim,
            cfg.action_dim,
            cfg.predictor_film_layers,
        )

    def encode_sequence(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, t, _, _, _ = images.shape
        embeddings: List[torch.Tensor] = []
        for step in range(t):
            current = images[:, step]
            feats = self.encoder(current)
            embeddings.append(feats)
        return {
            "embeddings": torch.stack(embeddings, dim=1),
        }


# ------------------------------------------------------------
# Loss utilities
# ------------------------------------------------------------


def jepa_loss(
    model: JEPAWorldModel, outputs: Dict[str, torch.Tensor], actions: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard one-step JEPA loss plus action logits from the predictor."""
    embeddings = outputs["embeddings"]
    if embeddings.shape[1] < 2:
        zero = embeddings.new_tensor(0.0)
        logits = embeddings.new_zeros((*embeddings.shape[:2], actions.shape[-1]))
        return zero, logits
    prev_embed = embeddings[:, :-1]
    prev_actions = actions[:, :-1]
    deltas, action_logits = model.predictor(prev_embed, prev_actions)
    preds = prev_embed + deltas
    target = embeddings[:, 1:].detach()
    return JEPA_LOSS(preds, target), action_logits


def rollout_loss(
    model: JEPAWorldModel,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    rollout_horizon: int,
) -> torch.Tensor:
    if rollout_horizon <= 1:
        return embeddings.new_tensor(0.0)
    b, t, d = embeddings.shape
    if t < 2:
        return embeddings.new_tensor(0.0)
    total = embeddings.new_tensor(0.0)
    steps = 0
    for start in range(t - 1):
        current = embeddings[:, start]
        max_h = min(rollout_horizon, t - start - 1)
        for offset in range(max_h):
            act = actions[:, start + offset]
            delta, _ = model.predictor(current, act)
            pred = current + delta
            target_step = embeddings[:, start + offset + 1].detach()
            total = total + JEPA_LOSS(pred, target_step)
            steps += 1
            current = pred
    return total / steps if steps > 0 else embeddings.new_tensor(0.0)


def latent_consistency_loss(embeddings: torch.Tensor) -> torch.Tensor:
    if embeddings.shape[1] < 2:
        return embeddings.new_tensor(0.0)
    diffs = embeddings[:, 1:] - embeddings[:, :-1]
    return diffs.abs().mean()


def ema_latent_consistency_loss(
    embeddings: torch.Tensor, ema_embeddings: torch.Tensor
) -> torch.Tensor:
    if embeddings.shape != ema_embeddings.shape:
        raise ValueError("EMA and online embeddings must share shape for consistency loss.")
    return EMA_CONSISTENCY_LOSS(embeddings, ema_embeddings.detach())


def sigreg_loss(embeddings: torch.Tensor, num_projections: int) -> torch.Tensor:
    b, t, d = embeddings.shape
    flat = embeddings.reshape(b * t, d)
    device = embeddings.device
    directions = torch.randn(num_projections, d, device=device)
    directions = F.normalize(directions, dim=-1)
    projected = flat @ directions.t()
    projected = projected.t()
    sorted_proj, _ = torch.sort(projected, dim=1)
    normal_samples = torch.randn_like(projected)
    sorted_normal, _ = torch.sort(normal_samples, dim=1)
    return SIGREG_LOSS(sorted_proj, sorted_normal)


def build_ema_model(model: JEPAWorldModel) -> JEPAWorldModel:
    device = next(model.parameters()).device
    ema_model = JEPAWorldModel(model.cfg).to(device)
    ema_model.load_state_dict(model.state_dict())
    for param in ema_model.parameters():
        param.requires_grad_(False)
    ema_model.eval()
    return ema_model


def update_ema_model(source: JEPAWorldModel, target: JEPAWorldModel, momentum: float) -> None:
    if not (0.0 <= momentum <= 1.0):
        raise ValueError("EMA momentum must be between 0 and 1.")
    if momentum == 1.0:
        return
    with torch.no_grad():
        for tgt_param, src_param in zip(target.parameters(), source.parameters()):
            tgt_param.data.mul_(momentum).add_(src_param.data, alpha=1.0 - momentum)


# ------------------------------------------------------------
# Training loop utilities
# ------------------------------------------------------------


@dataclass
class BatchDifficultyInfo:
    indices: List[int]
    paths: List[List[str]]
    scores: List[float]
    frame_indices: List[int]


def training_step(
    model: JEPAWorldModel,
    decoder: VisualizationDecoder,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor, List[List[str]], torch.Tensor],
    cfg: TrainConfig,
    weights: LossWeights,
    ema_model: Optional[JEPAWorldModel] = None,
    ema_momentum: float = 0.0,
) -> Tuple[Dict[str, float], Optional[BatchDifficultyInfo]]:
    images, actions = batch[0], batch[1]
    batch_paths = batch[2] if len(batch) > 2 else None
    batch_indices = batch[3] if len(batch) > 3 else None
    track_hard_examples = cfg.hard_example_reservoir > 0
    device = next(model.parameters()).device
    images = images.to(device)
    actions = actions.to(device)
    outputs = model.encode_sequence(images)
    need_recon = cfg.recon_weight > 0 or track_hard_examples
    recon: Optional[torch.Tensor] = None
    if need_recon:
        recon = decoder(outputs["embeddings"])
    loss_jepa_raw, action_logits = jepa_loss(model, outputs, actions)
    loss_jepa = loss_jepa_raw if weights.jepa > 0 else images.new_tensor(0.0)
    prev_actions = actions[:, :-1]
    if weights.action_recon > 0 and prev_actions.numel() > 0:
        loss_action = ACTION_RECON_LOSS(action_logits, prev_actions)
    else:
        loss_action = images.new_tensor(0.0)
    loss_rollout = (
        rollout_loss(model, outputs["embeddings"], actions, cfg.rollout_horizon)
        if weights.rollout > 0
        else images.new_tensor(0.0)
    )
    loss_consistency = (
        latent_consistency_loss(outputs["embeddings"])
        if weights.consistency > 0
        else images.new_tensor(0.0)
    )
    loss_sigreg = (
        sigreg_loss(outputs["embeddings"], cfg.sigreg_projections)
        if weights.sigreg > 0
        else images.new_tensor(0.0)
    )
    loss_ema_consistency = images.new_tensor(0.0)
    if ema_model is not None and weights.ema_consistency > 0:
        with torch.no_grad():
            ema_outputs = ema_model.encode_sequence(images)
        loss_ema_consistency = ema_latent_consistency_loss(outputs["embeddings"], ema_outputs["embeddings"])
    if cfg.recon_weight > 0 and recon is not None:
        loss_recon = RECON_LOSS(recon, images)
    else:
        loss_recon = images.new_tensor(0.0)

    world_loss = (
        weights.jepa * loss_jepa
        + weights.sigreg * loss_sigreg
        + weights.rollout * loss_rollout
        + weights.consistency * loss_consistency
        + weights.ema_consistency * loss_ema_consistency
        + weights.action_recon * loss_action
        + cfg.recon_weight * loss_recon
    )
    optimizer.zero_grad()
    world_loss.backward()
    world_grad_norm = grad_norm(model.parameters())
    decoder_grad_norm = grad_norm(decoder.parameters())
    optimizer.step()
    if ema_model is not None:
        update_ema_model(model, ema_model, ema_momentum)

    difficulty_info: Optional[BatchDifficultyInfo] = None
    if (
        track_hard_examples
        and recon is not None
        and batch_paths is not None
        and batch_indices is not None
        and len(batch_paths) == images.shape[0]
    ):
        per_frame_errors = (recon.detach() - images.detach()).abs().mean(dim=(2, 3, 4))
        sample_scores, hardest_frames = torch.max(per_frame_errors, dim=1)
        difficulty_info = BatchDifficultyInfo(
            indices=batch_indices.detach().cpu().tolist(),
            paths=[list(paths) for paths in batch_paths],
            scores=sample_scores.detach().cpu().tolist(),
            frame_indices=hardest_frames.detach().cpu().tolist(),
        )

    metrics = {
        "loss_jepa": loss_jepa.item(),
        "loss_sigreg": loss_sigreg.item(),
        "loss_rollout": loss_rollout.item(),
        "loss_consistency": loss_consistency.item(),
        "loss_ema_consistency": loss_ema_consistency.item(),
        "loss_recon": loss_recon.item(),
        "loss_action": loss_action.item(),
        "loss_world": world_loss.item(),
        "grad_world": world_grad_norm,
        "grad_decoder": decoder_grad_norm,
    }
    return metrics, difficulty_info


# ------------------------------------------------------------
# Example dataset + dataloader
# ------------------------------------------------------------


def collate_batch(
    batch: Iterable[Tuple[torch.Tensor, torch.Tensor, List[str], int]]
) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]], torch.Tensor]:
    obs, actions, paths, indices = zip(*batch)
    obs_tensor = torch.stack(obs, dim=0)
    act_tensor = torch.stack(actions, dim=0)
    path_batch = [list(seq) for seq in paths]
    idx_tensor = torch.tensor(indices, dtype=torch.long)
    return obs_tensor, act_tensor, path_batch, idx_tensor


class TrajectorySequenceDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, List[str], int]]):
    """Load contiguous frame/action sequences from recorded trajectories."""

    def __init__(
        self,
        root: Path,
        seq_len: int,
        image_hw: Tuple[int, int],
        max_traj: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.seq_len = seq_len
        self.image_hw = image_hw
        if self.seq_len < 1:
            raise ValueError("seq_len must be positive.")
        trajectories = list_trajectories(self.root)
        items = list(trajectories.items())
        if max_traj is not None:
            items = items[:max_traj]
        self.samples: List[Tuple[List[Path], np.ndarray, int]] = []
        self.action_dim: Optional[int] = None
        for traj_name, frame_paths in items:
            actions_path = (self.root / traj_name) / "actions.npz"
            if not actions_path.is_file():
                raise FileNotFoundError(f"Missing actions.npz for {traj_name}")
            with np.load(actions_path) as data:
                action_arr = data["actions"] if "actions" in data else data[list(data.files)[0]]
            if action_arr.ndim == 1:
                action_arr = action_arr[:, None]
            assert action_arr.shape[0] == len(frame_paths), (
                f"Action count {action_arr.shape[0]} does not match frame count {len(frame_paths)} in {traj_name}"
            )
            action_arr = action_arr.astype(np.float32, copy=False)
            if self.action_dim is None:
                self.action_dim = action_arr.shape[1]
            elif self.action_dim != action_arr.shape[1]:
                raise ValueError(
                    f"Inconsistent action dimension for {traj_name}: expected {self.action_dim}, got {action_arr.shape[1]}"
                )
            if len(frame_paths) < self.seq_len:
                raise ValueError(f"Trajectory {traj_name} shorter than seq_len {self.seq_len}")
            max_start = len(frame_paths) - self.seq_len
            for start in range(max_start + 1):
                self.samples.append((frame_paths, action_arr, start))
        if not self.samples:
            raise RuntimeError(f"No usable sequences found under {self.root}")
        if self.action_dim is None:
            raise RuntimeError("Failed to infer action dimensionality.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        frame_paths, actions, start = self.samples[index]
        frames: List[torch.Tensor] = []
        path_slice: List[str] = []
        for offset in range(self.seq_len):
            path = frame_paths[start + offset]
            frame = load_frame_as_tensor(path, size=self.image_hw)
            frames.append(frame)
            path_slice.append(str(path))
        action_slice = actions[start : start + self.seq_len]
        # Each frame/action pair must stay aligned so the predictor knows which action follows each observation.
        assert len(frames) == self.seq_len, f"Expected {self.seq_len} frames, got {len(frames)}"
        assert action_slice.shape[0] == self.seq_len, (
            f"Expected {self.seq_len} actions, got {action_slice.shape[0]}"
        )
        assert action_slice.shape[1] == actions.shape[1], "Action dimensionality changed unexpectedly."
        return torch.stack(frames, dim=0), torch.from_numpy(action_slice), path_slice, index


# ------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------


def log_metrics(step: int, metrics: Dict[str, float]) -> None:
    pretty = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    print(f"[step {step}] {pretty}")


def _filtered_metrics_for_logging(
    metrics: Dict[str, float],
    weights: LossWeights,
    recon_weight: float,
) -> Dict[str, float]:
    filtered = dict(metrics)
    if weights.jepa <= 0:
        filtered.pop("loss_jepa", None)
    if weights.sigreg <= 0:
        filtered.pop("loss_sigreg", None)
    if weights.rollout <= 0:
        filtered.pop("loss_rollout", None)
    if weights.consistency <= 0:
        filtered.pop("loss_consistency", None)
    if weights.ema_consistency <= 0:
        filtered.pop("loss_ema_consistency", None)
    if recon_weight <= 0:
        filtered.pop("loss_recon", None)
    if weights.action_recon <= 0:
        filtered.pop("loss_action", None)
    return filtered


LOSS_COLUMNS = [
    "step",
    "loss_world",
    "loss_jepa",
    "loss_sigreg",
    "loss_rollout",
    "loss_consistency",
    "loss_ema_consistency",
    "loss_recon",
    "loss_action",
    "grad_world",
    "grad_decoder",
]


@dataclass
class LossHistory:
    steps: List[float] = field(default_factory=list)
    world: List[float] = field(default_factory=list)
    jepa: List[float] = field(default_factory=list)
    sigreg: List[float] = field(default_factory=list)
    rollout: List[float] = field(default_factory=list)
    consistency: List[float] = field(default_factory=list)
    ema_consistency: List[float] = field(default_factory=list)
    recon: List[float] = field(default_factory=list)
    action: List[float] = field(default_factory=list)
    grad_world: List[float] = field(default_factory=list)
    grad_decoder: List[float] = field(default_factory=list)

    def append(self, step: float, metrics: Dict[str, float]) -> None:
        self.steps.append(step)
        self.world.append(metrics["loss_world"])
        self.jepa.append(metrics["loss_jepa"])
        self.sigreg.append(metrics["loss_sigreg"])
        self.rollout.append(metrics["loss_rollout"])
        self.consistency.append(metrics["loss_consistency"])
        self.ema_consistency.append(metrics["loss_ema_consistency"])
        self.recon.append(metrics["loss_recon"])
        self.action.append(metrics["loss_action"])
        self.grad_world.append(metrics["grad_world"])
        self.grad_decoder.append(metrics["grad_decoder"])

    def __len__(self) -> int:
        return len(self.steps)


def write_loss_csv(history: LossHistory, path: Path) -> None:
    if len(history) == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LOSS_COLUMNS)
        for row in zip(
            history.steps,
            history.world,
            history.jepa,
            history.sigreg,
            history.rollout,
            history.consistency,
            history.ema_consistency,
            history.recon,
            history.action,
            history.grad_world,
            history.grad_decoder,
        ):
            writer.writerow(row)


def plot_loss_curves(history: LossHistory, out_dir: Path) -> None:
    if len(history) == 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    default_cycle = plt.rcParams.get("axes.prop_cycle")
    color_cycle = default_cycle.by_key().get("color", []) if default_cycle is not None else []

    def _color(idx: int) -> str:
        return color_cycle[idx % len(color_cycle)]

    color_map = {
        "world": _color(0),
        "jepa": _color(1),
        "sigreg": _color(2),
        "recon": _color(3),
        "rollout": _color(4),
        "consistency": _color(5),
        "action": _color(6),
        "ema_consistency": _color(7),
    }
    plt.plot(history.steps, history.world, label="world", color=color_map["world"])
    plt.plot(history.steps, history.jepa, label="jepa", color=color_map["jepa"])
    plt.plot(history.steps, history.sigreg, label="sigreg", color=color_map["sigreg"])
    plt.plot(history.steps, history.recon, label="recon", color=color_map["recon"])
    if any(val != 0.0 for val in history.rollout):
        plt.plot(history.steps, history.rollout, label="rollout", color=color_map["rollout"])
    if any(val != 0.0 for val in history.consistency):
        plt.plot(history.steps, history.consistency, label="consistency", color=color_map["consistency"])
    if any(val != 0.0 for val in history.ema_consistency):
        plt.plot(history.steps, history.ema_consistency, label="ema_consistency", color=color_map["ema_consistency"])
    if any(val != 0.0 for val in history.action):
        plt.plot(history.steps, history.action, label="action_recon", color=color_map["action"])
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training Losses")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=200)
    plt.close()


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2).item()
        total += param_norm * param_norm
    return float(total**0.5)


def _short_traj_state_label(frame_path: str) -> str:
    path = Path(frame_path)
    traj = next((part for part in path.parts if part.startswith("traj_")), path.parent.name)
    return f"{traj}/{path.stem}"


@dataclass
class HardSampleRecord:
    dataset_index: int
    score: float
    frame_path: str
    label: str
    sequence_paths: List[str]
    frame_index: int


class HardSampleReservoir:
    def __init__(self, capacity: int, sample_decay: float = 0.9) -> None:
        self.capacity = max(0, capacity)
        self.sample_decay = sample_decay
        self._samples: Dict[int, HardSampleRecord] = {}

    def __len__(self) -> int:
        return len(self._samples)

    def update(
        self,
        indices: List[int],
        paths: List[List[str]],
        scores: List[float],
        frame_indices: List[int],
    ) -> None:
        if self.capacity <= 0:
            return
        for idx, seq_paths, score, frame_idx in zip(indices, paths, scores, frame_indices):
            if seq_paths is None or not seq_paths:
                continue
            score_val = float(score)
            if not math.isfinite(score_val):
                continue
            idx_int = int(idx)
            frame_list = list(seq_paths)
            frame_pos = max(0, min(int(frame_idx), len(frame_list) - 1))
            frame_path = frame_list[frame_pos]
            label = _short_traj_state_label(frame_path)
            record = self._samples.get(idx_int)
            if record is None:
                self._samples[idx_int] = HardSampleRecord(idx_int, score_val, frame_path, label, frame_list, frame_pos)
            else:
                if score_val >= record.score:
                    record.score = score_val
                    record.frame_path = frame_path
                    record.label = label
                    record.sequence_paths = frame_list
                    record.frame_index = frame_pos
                else:
                    record.score = (record.score * 0.75) + (score_val * 0.25)
        self._prune()

    def sample_records(self, count: int) -> List[HardSampleRecord]:
        if count <= 0 or not self._samples:
            return []
        population = list(self._samples.values())
        count = min(count, len(population))
        weights = [max(record.score, 1e-6) for record in population]
        chosen = random.choices(population=population, weights=weights, k=count)
        return chosen

    def mark_sampled(self, dataset_index: int) -> None:
        record = self._samples.get(dataset_index)
        if record is None:
            return
        record.score *= self.sample_decay
        if record.score <= 1e-6:
            self._samples.pop(dataset_index, None)

    def topk(self, limit: int) -> List[HardSampleRecord]:
        if limit <= 0 or not self._samples:
            return []
        limit = min(limit, len(self._samples))
        return sorted(self._samples.values(), key=lambda rec: rec.score, reverse=True)[:limit]

    def _prune(self) -> None:
        if self.capacity <= 0 or len(self._samples) <= self.capacity:
            return
        ordered = sorted(self._samples.items(), key=lambda item: item[1].score, reverse=True)
        self._samples = dict(ordered[: self.capacity])


def inject_hard_examples_into_batch(
    batch: Tuple[torch.Tensor, torch.Tensor, List[List[str]], torch.Tensor],
    dataset: TrajectorySequenceDataset,
    reservoir: Optional[HardSampleReservoir],
    mix_ratio: float,
) -> None:
    if reservoir is None or mix_ratio <= 0:
        return
    images, actions, paths, indices = batch
    batch_size = images.shape[0]
    if batch_size == 0 or len(reservoir) == 0:
        return
    ratio = max(0.0, min(1.0, mix_ratio))
    desired = min(int(round(batch_size * ratio)), len(reservoir))
    if desired <= 0:
        return
    hard_records = reservoir.sample_records(desired)
    for slot, record in enumerate(hard_records):
        hard_obs, hard_actions, hard_paths, hard_index = dataset[record.dataset_index]
        images[slot].copy_(hard_obs)
        actions[slot].copy_(hard_actions)
        paths[slot] = list(hard_paths)
        indices[slot] = hard_index
        reservoir.mark_sampled(record.dataset_index)


def _infer_raw_frame_shape(dataset: TrajectorySequenceDataset) -> Optional[Tuple[int, int, int]]:
    if not dataset.samples:
        return None
    frame_paths, _, start = dataset.samples[0]
    if not frame_paths:
        return None
    index = min(max(start, 0), len(frame_paths) - 1)
    path = frame_paths[index]
    try:
        with Image.open(path) as img:
            width, height = img.size
            channels = len(img.getbands())
    except (FileNotFoundError, OSError):
        return None
    return height, width, channels


def _format_hwc(height: int, width: int, channels: int) -> str:
    return f"{height}×{width}×{channels}"


def format_shape_summary(
    raw_shape: Optional[Tuple[int, int, int]],
    encoder_info: Dict[str, Any],
    predictor_info: Dict[str, Any],
    decoder_info: Dict[str, Any],
    cfg: ModelConfig,
) -> str:
    lines: List[str] = []
    lines.append("Model Shape Summary (H×W×C)")
    if raw_shape is not None:
        lines.append(f"Raw frame {_format_hwc(*raw_shape)}")
    input_shape = encoder_info.get("input")
    if input_shape is not None:
        line = f"Input to encoder {_format_hwc(*input_shape)}"
        if raw_shape is not None:
            line = f"  └─ Data loader resize → {_format_hwc(*input_shape)}"
        lines.append(line)
    lines.append("")
    auto_schedule = _derive_channel_schedule(cfg.embedding_dim, cfg.num_downsample_layers)
    lines.append(f"Channel schedule: {auto_schedule}")
    lines.append("Encoder:")
    for stage in encoder_info.get("stages", []):
        lines.append(
            f"  • Stage {stage['stage']}: {_format_hwc(*stage['in'])} → {_format_hwc(*stage['out'])}"
        )
    latent_dim = encoder_info.get("latent_dim")
    if latent_dim is not None:
        lines.append(f"AdaptiveAvgPool → Latent vector 1×1×{latent_dim}")
    lines.append("")
    lines.append("Predictor:")
    lines.append(
        f"  latent {predictor_info.get('latent_dim')} → hidden {predictor_info.get('hidden_dim')} "
        f"(action_dim={predictor_info.get('action_dim')}, FiLM layers={predictor_info.get('film_layers')})"
    )
    lines.append("")
    lines.append("Decoder:")
    proj_shape = decoder_info.get("projection")
    if proj_shape is not None:
        lines.append(f"  Projection reshape → {_format_hwc(*proj_shape)}")
    for stage in decoder_info.get("upsample", []):
        lines.append(
            f"  • UpStage {stage['stage']}: {_format_hwc(*stage['in'])} → {_format_hwc(*stage['out'])}"
        )
    pre_resize = decoder_info.get("pre_resize")
    target = decoder_info.get("final_target")
    needs_resize = decoder_info.get("needs_resize")
    if pre_resize is not None and target is not None:
        if needs_resize:
            lines.append(
                f"  Final conv output {_format_hwc(*pre_resize)} → bilinear resize → {_format_hwc(*target)}"
            )
        else:
            lines.append(f"  Final output {_format_hwc(*pre_resize)}")
    return "\n".join(lines)


def _extract_frame_labels(
    batch_paths: Optional[List[List[str]]],
    sample_idx: int,
    start_idx: int,
    length: int,
) -> List[str]:
    if batch_paths is None or sample_idx >= len(batch_paths):
        return [f"t={start_idx + offset}" for offset in range(length)]
    sample_paths = batch_paths[sample_idx]
    end = min(start_idx + length, len(sample_paths))
    slice_paths = sample_paths[start_idx:end]
    if len(slice_paths) < length:
        slice_paths = sample_paths[-length:]
    return [_short_traj_state_label(path) for path in slice_paths]


LOSS_CMAP = plt.get_cmap("coolwarm")
GRAD_CMAP = plt.get_cmap("magma")


def _loss_to_heatmap(frame: torch.Tensor, recon: torch.Tensor) -> np.ndarray:
    diff = (recon.detach() - frame.detach()).mean(dim=0)
    diff_np = diff.cpu().numpy()
    max_val = np.max(np.abs(diff_np))
    if max_val <= 0:
        norm = diff_np
    else:
        norm = diff_np / max_val
    heat = LOSS_CMAP((norm + 1) / 2)[..., :3]
    return (heat * 255.0).astype(np.uint8)


def _gradient_norm_to_heatmap(grad_map: torch.Tensor) -> np.ndarray:
    grad_np = grad_map.detach().cpu().numpy()
    max_val = np.max(grad_np)
    if not np.isfinite(max_val) or max_val <= 0:
        norm = np.zeros_like(grad_np)
    else:
        norm = grad_np / max_val
    heat = GRAD_CMAP(norm)[..., :3]
    return (heat * 255.0).astype(np.uint8)


def _per_pixel_gradient_heatmap(delta_frame: torch.Tensor, target_delta: torch.Tensor) -> np.ndarray:
    with torch.enable_grad():
        delta_leaf = delta_frame.detach().unsqueeze(0).clone().requires_grad_(True)
        target = target_delta.detach().unsqueeze(0)
        loss = RECON_LOSS(delta_leaf, target)
        grad = torch.autograd.grad(loss, delta_leaf, retain_graph=False, create_graph=False)[0]
    grad_norm = grad.squeeze(0).pow(2).sum(dim=0).sqrt()
    return _gradient_norm_to_heatmap(grad_norm)


def _prediction_gradient_heatmap(pred_frame: torch.Tensor, target_frame: torch.Tensor) -> np.ndarray:
    with torch.enable_grad():
        pred_leaf = pred_frame.detach().unsqueeze(0).clone().requires_grad_(True)
        target = target_frame.detach().unsqueeze(0)
        loss = RECON_LOSS(pred_leaf, target)
        grad = torch.autograd.grad(loss, pred_leaf, retain_graph=False, create_graph=False)[0]
    grad_norm = grad.squeeze(0).pow(2).sum(dim=0).sqrt()
    return _gradient_norm_to_heatmap(grad_norm)


def _make_fixed_selection(frames: torch.Tensor, vis_rows: int) -> Optional[VisualizationSelection]:
    if frames is None:
        return None
    batch_size, seq_len = frames.shape[0], frames.shape[1]
    if batch_size == 0 or seq_len < 2:
        return None
    num_rows = min(vis_rows, batch_size)
    row_indices = torch.arange(num_rows, dtype=torch.long)
    time_indices = (torch.arange(num_rows, dtype=torch.long) % (seq_len - 1)) + 1
    return VisualizationSelection(row_indices=row_indices, time_indices=time_indices)


def _render_visualization_batch(
    model: JEPAWorldModel,
    decoder: VisualizationDecoder,
    batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[List[List[str]]]]],
    rows: int,
    rollout_steps: int,
    max_columns: Optional[int],
    device: torch.device,
    selection: Optional[VisualizationSelection] = None,
    show_gradients: bool = False,
    log_deltas: bool = False,
) -> Optional[Tuple[List[VisualizationSequence], str]]:
    if batch_cpu is None:
        return None
    vis_frames = batch_cpu[0].to(device)
    vis_actions = batch_cpu[1].to(device)
    frame_paths = batch_cpu[2] if len(batch_cpu) > 2 else None
    vis_outputs = model.encode_sequence(vis_frames)
    vis_embeddings = vis_outputs["embeddings"]
    if vis_frames.shape[1] < 1:
        return None
    decoded_frames = decoder(vis_embeddings)
    batch_size = vis_frames.shape[0]
    if batch_size == 0:
        return None
    min_start = 0
    target_window = max(2, rollout_steps + 1)
    if max_columns is not None:
        target_window = max(target_window, max(2, max_columns))
    max_window = min(target_window, vis_frames.shape[1] - min_start)
    if max_window < 2:
        return None
    max_start = max(min_start, vis_frames.shape[1] - max_window)
    if selection is not None and selection.row_indices.numel() > 0:
        num_rows = min(rows, selection.row_indices.numel())
        row_indices = selection.row_indices[:num_rows].to(device=device)
        base_starts = selection.time_indices[:num_rows].to(device=device)
    else:
        num_rows = min(rows, batch_size)
        row_indices = torch.randperm(batch_size, device=device)[:num_rows]
        base_starts = torch.randint(min_start, max_start + 1, (num_rows,), device=device)
    sequences: List[VisualizationSequence] = []
    debug_lines: List[str] = []
    for row_offset, idx in enumerate(row_indices):
        start_idx = int(base_starts[row_offset].item()) if base_starts is not None else min_start
        start_idx = max(min_start, min(start_idx, max_start))
        gt_slice = vis_frames[idx, start_idx : start_idx + max_window]
        if gt_slice.shape[0] < max_window:
            continue
        action_texts: List[str] = []
        for offset in range(max_window):
            action_idx = min(start_idx + offset, vis_actions.shape[1] - 1)
            action_texts.append(describe_action_tensor(vis_actions[idx, action_idx]))
        recon_tensor = decoded_frames[idx, start_idx : start_idx + max_window].clamp(0, 1)
        rollout_frames: List[Optional[torch.Tensor]] = [None for _ in range(max_window)]
        gradient_maps: List[Optional[np.ndarray]] = [None for _ in range(max_window)]
        current_embed = vis_embeddings[idx, start_idx].unsqueeze(0)
        prev_pred_frame = decoded_frames[idx, start_idx].detach()
        current_frame = prev_pred_frame
        for step in range(1, max_window):
            action = vis_actions[idx, start_idx + step - 1].unsqueeze(0)
            delta_embed, _ = model.predictor(current_embed, action)
            next_embed = current_embed + delta_embed
            decoded_next = decoder(next_embed)[0]
            current_frame = decoded_next.clamp(0, 1)
            if show_gradients:
                gradient_maps[step] = _prediction_gradient_heatmap(current_frame, gt_slice[step])
            else:
                gradient_maps[step] = _loss_to_heatmap(gt_slice[step], current_frame)
            rollout_frames[step] = current_frame.detach().cpu()
            current_embed = next_embed
            if log_deltas and row_offset < 2:
                latent_norm = delta_embed.norm().item()
                pixel_delta = (current_frame - prev_pred_frame).abs().mean().item()
                frame_mse = F.mse_loss(current_frame, gt_slice[step]).item()
                debug_lines.append(
                    (
                        f"[viz] row={int(idx)} step={step} "
                        f"latent_norm={latent_norm:.4f} pixel_delta={pixel_delta:.4f} "
                        f"frame_mse={frame_mse:.4f}"
                    )
                )
            prev_pred_frame = current_frame.detach()
        labels = _extract_frame_labels(frame_paths, int(idx.item()), start_idx, max_window)
        sequences.append(
            VisualizationSequence(
                ground_truth=gt_slice.detach().cpu(),
                rollout=rollout_frames,
                gradients=gradient_maps,
                reconstructions=recon_tensor.detach().cpu(),
                labels=labels,
                actions=action_texts,
            )
        )
    if not sequences:
        return None
    if debug_lines:
        print("\n".join(debug_lines))
    grad_label = "Gradient Norm" if show_gradients else "Error Heatmap"
    return sequences, grad_label


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------


def run_training(cfg: TrainConfig, model_cfg: ModelConfig, weights: LossWeights, demo: bool = True) -> None:
    device = pick_device(cfg.device)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = cfg.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = run_dir / "metrics"
    (metrics_dir).mkdir(parents=True, exist_ok=True)
    fixed_vis_dir = run_dir / "vis_fixed"
    rolling_vis_dir = run_dir / "vis_rolling"
    embeddings_vis_dir = run_dir / "embeddings"
    embeddings_vis_dir.mkdir(parents=True, exist_ok=True)
    samples_hard_dir = run_dir / "samples_hard"
    samples_hard_dir.mkdir(parents=True, exist_ok=True)
    debug_vis = cfg.debug_visualization
    inputs_vis_dir: Optional[Path]
    pair_vis_dir: Optional[Path]
    if debug_vis.input_vis_every_steps > 0:
        inputs_vis_dir = run_dir / "vis_inputs"
        inputs_vis_dir.mkdir(parents=True, exist_ok=True)
    else:
        inputs_vis_dir = None
    if debug_vis.pair_vis_every_steps > 0:
        pair_vis_dir = run_dir / "vis_pairs"
        pair_vis_dir.mkdir(parents=True, exist_ok=True)
    else:
        pair_vis_dir = None
    loss_history = LossHistory()
    write_run_metadata(run_dir, cfg, model_cfg)
    dataset = TrajectorySequenceDataset(
        root=cfg.data_root,
        seq_len=cfg.seq_len,
        image_hw=(model_cfg.image_size, model_cfg.image_size),
        max_traj=cfg.max_trajectories,
    )
    raw_shape = _infer_raw_frame_shape(dataset)
    hard_reservoir = HardSampleReservoir(cfg.hard_example_reservoir) if cfg.hard_example_reservoir > 0 else None
    dataset_action_dim = getattr(dataset, "action_dim", model_cfg.action_dim)
    assert dataset_action_dim == 8, f"Expected action_dim 8, got {dataset_action_dim}"
    if model_cfg.action_dim != dataset_action_dim:
        model_cfg = replace(model_cfg, action_dim=dataset_action_dim)
    model = JEPAWorldModel(model_cfg).to(device)
    ema_model: Optional[JEPAWorldModel] = None
    if cfg.ema_momentum >= 0.0:
        ema_model = build_ema_model(model)
    decoder = VisualizationDecoder(
        model.embedding_dim,
        model_cfg.in_channels,
        model_cfg.image_size,
        model_cfg.channel_schedule,
        model_cfg.latent_hw,
    ).to(device)
    summary = format_shape_summary(
        raw_shape,
        model.encoder.shape_info(),
        model.predictor.shape_info(),
        decoder.shape_info(),
        model_cfg,
    )
    print(summary)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(decoder.parameters()),
        lr=cfg.lr,
    )
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_batch)

    fixed_batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[List[List[str]]]]] = None
    fixed_selection: Optional[VisualizationSelection] = None
    try:
        sample_batch = next(iter(dataloader))
        fixed_paths = sample_batch[2] if len(sample_batch) > 2 else None
        fixed_batch_cpu = (
            sample_batch[0].clone(),
            sample_batch[1].clone(),
            [list(paths) for paths in fixed_paths] if fixed_paths is not None else None,
        )
        fixed_selection = _make_fixed_selection(fixed_batch_cpu[0], cfg.vis_rows)
    except StopIteration:
        fixed_batch_cpu = None
    rolling_batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[List[List[str]]]]] = None
    embedding_batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    if cfg.embedding_projection_samples > 0:
        embed_loader = DataLoader(
            dataset,
            batch_size=min(cfg.embedding_projection_samples, len(dataset)),
            shuffle=True,
            collate_fn=collate_batch,
        )
        try:
            embed_batch = next(iter(embed_loader))
            embedding_batch_cpu = (embed_batch[0].clone(), embed_batch[1].clone())
        except StopIteration:
            embedding_batch_cpu = None

    data_iter = iter(dataloader)
    for global_step in range(cfg.total_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        inject_hard_examples_into_batch(batch, dataset, hard_reservoir, cfg.hard_example_mix_ratio)
        metrics, difficulty_info = training_step(
            model, decoder, optimizer, batch, cfg, weights, ema_model, cfg.ema_momentum
        )
        if hard_reservoir is not None and difficulty_info is not None:
            hard_reservoir.update(
                difficulty_info.indices, difficulty_info.paths, difficulty_info.scores, difficulty_info.frame_indices
            )
        if cfg.log_every_steps > 0 and global_step % cfg.log_every_steps == 0:
            loggable = _filtered_metrics_for_logging(metrics, weights, cfg.recon_weight)
            log_metrics(global_step, loggable)
            loss_history.append(global_step, metrics)
            write_loss_csv(loss_history, metrics_dir / "loss.csv")
            plot_loss_curves(loss_history, metrics_dir)
        batch_paths = batch[2] if len(batch) > 2 else None
        rolling_batch_cpu = (
            batch[0].clone(),
            batch[1].clone(),
            [list(paths) for paths in batch_paths] if batch_paths is not None else None,
        )
        if (
            inputs_vis_dir is not None
            and debug_vis.input_vis_every_steps > 0
            and global_step % debug_vis.input_vis_every_steps == 0
            and rolling_batch_cpu is not None
        ):
            save_input_batch_visualization(
                inputs_vis_dir / f"inputs_{global_step:07d}.png",
                rolling_batch_cpu[0],
                rolling_batch_cpu[1],
                debug_vis.input_vis_rows,
            )
        if (
            pair_vis_dir is not None
            and debug_vis.pair_vis_every_steps > 0
            and global_step % debug_vis.pair_vis_every_steps == 0
            and rolling_batch_cpu is not None
        ):
            save_temporal_pair_visualization(
                pair_vis_dir / f"pairs_{global_step:07d}.png",
                rolling_batch_cpu[0],
                rolling_batch_cpu[1],
                debug_vis.pair_vis_rows,
            )
        if (
            cfg.vis_every_steps > 0
            and global_step % cfg.vis_every_steps == 0
        ):
            model.eval()
            with torch.no_grad():
                rendered = _render_visualization_batch(
                    model,
                    decoder,
                    fixed_batch_cpu,
                    cfg.vis_rows,
                    cfg.vis_rollout,
                    cfg.vis_columns,
                    device,
                    fixed_selection,
                    cfg.vis_gradient_norms,
                    cfg.vis_log_deltas,
                )
                if rendered is not None:
                    sequences, grad_label = rendered
                    save_rollout_sequence_batch(
                        fixed_vis_dir,
                        sequences,
                        grad_label,
                        global_step,
                    )
                rendered = _render_visualization_batch(
                    model,
                    decoder,
                    rolling_batch_cpu,
                    cfg.vis_rows,
                    cfg.vis_rollout,
                    cfg.vis_columns,
                    device,
                    None,
                    cfg.vis_gradient_norms,
                    cfg.vis_log_deltas,
                )
                if rendered is not None:
                    sequences, grad_label = rendered
                    save_rollout_sequence_batch(
                        rolling_vis_dir,
                        sequences,
                        grad_label,
                        global_step,
                    )
                if embedding_batch_cpu is not None:
                    embed_frames = embedding_batch_cpu[0].to(device)
                    embed_outputs = model.encode_sequence(embed_frames)
                    save_embedding_projection(
                        embed_outputs["embeddings"],
                        embeddings_vis_dir / f"embeddings_{global_step:07d}.png",
                    )
                if hard_reservoir is not None:
                    hard_samples = hard_reservoir.topk(cfg.hard_example_vis_rows * cfg.hard_example_vis_columns)
                    save_hard_example_grid(
                        samples_hard_dir / f"hard_{global_step:07d}.png",
                        hard_samples,
                        cfg.hard_example_vis_columns,
                        cfg.hard_example_vis_rows,
                        dataset.image_hw,
                    )
            model.train()
    if len(loss_history):
        write_loss_csv(loss_history, metrics_dir / "loss.csv")
        plot_loss_curves(loss_history, metrics_dir)


def main() -> None:
    cfg = tyro.cli(TrainConfig)
    model_cfg = ModelConfig()
    run_training(cfg, model_cfg, cfg.loss_weights, demo=False)


if __name__ == "__main__":
    main()
