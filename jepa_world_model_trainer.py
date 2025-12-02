#!/usr/bin/env python3
"""Training loop for the JEPA world model with local/global specialization and SIGReg."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import csv
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from datetime import datetime

from recon.data import list_trajectories, load_frame_as_tensor
from super_mario_env import _describe_controller_vector, _describe_controller_vector_compact
from utils.device_utils import pick_device


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


class HardnessWeightedL1Loss(nn.Module):
    """L1 loss with per-sample hardness weighting derived from mean error."""

    def __init__(self, beta: float = 2.0, max_weight: float = 100.0, eps: float = 1e-6) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive.")
        if max_weight <= 0:
            raise ValueError("max_weight must be positive.")
        self.beta = beta
        self.max_weight = max_weight
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = torch.abs(input - target)
        dims = tuple(range(1, l1.dim()))
        norm = l1.detach().mean(dim=dims, keepdim=True).clamp_min(self.eps)
        rel_error = l1.detach() / norm
        weight = rel_error.pow(self.beta).clamp(max=self.max_weight)
        return (weight * l1).mean()


class HardnessWeightedMSELoss(nn.Module):
    """Simple hardness-weighted MSE mirroring the L1 variant."""

    def __init__(self, beta: float = 2.0, max_weight: float = 100.0, eps: float = 1e-6) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive.")
        if max_weight <= 0:
            raise ValueError("max_weight must be positive.")
        self.beta = beta
        self.max_weight = max_weight
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.shape != target.shape:
            raise ValueError("input and target must share shape")
        mse = (input - target).pow(2)
        dims = tuple(range(1, mse.dim()))
        norm = mse.detach().mean(dim=dims, keepdim=True).clamp_min(self.eps)
        rel_error = mse.detach() / norm
        weight = rel_error.pow(self.beta).clamp(max=self.max_weight)
        return (weight * mse).mean()


class HardnessWeightedMedianLoss(nn.Module):
    """Twin of :class:`HardnessWeightedL1Loss` using a median baseline on residual deltas."""

    def __init__(self, beta: float = 2.0, max_weight: float = 100.0, eps: float = 1e-6) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive.")
        if max_weight <= 0:
            raise ValueError("max_weight must be positive.")
        self.beta = beta
        self.max_weight = max_weight
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.shape != target.shape:
            raise ValueError("input and target must share shape")
        residual = (input - target).abs()
        b = residual.shape[0]
        flat = residual.detach().flatten(start_dim=1)
        median = flat.median(dim=1).values.view(b, *((1,) * (residual.dim() - 1))).clamp_min(self.eps)
        rel_error = residual.detach() / median
        weight = rel_error.pow(self.beta).clamp(max=self.max_weight)
        return (weight * residual).mean()


class Encoder(nn.Module):
    def __init__(self, in_channels: int, schedule: Tuple[int, ...]):
        super().__init__()
        blocks: List[nn.Module] = []
        ch_prev = in_channels
        for ch in schedule:
            blocks.append(DownBlock(ch_prev, ch))
            ch_prev = ch
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.blocks(x)
        return self.pool(feats).flatten(1)


class PredictorNetwork(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, action_dim: int, film_layers: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embedding_dim, hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, embedding_dim)
        self.activation = nn.SiLU(inplace=True)
        self.action_embed = ActionEmbedding(action_dim, hidden_dim)
        self.action_embed_dim = hidden_dim
        self.film = ActionFiLM(hidden_dim, hidden_dim, film_layers)

    def forward(self, embeddings: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
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
        return delta.view(*original_shape, delta.shape[-1])


class FocalL1Loss(nn.Module):
    """Focal-style L1 reconstruction loss that upweights harder pixels."""

    def __init__(self, gamma: float = 2.0, max_weight: float = 100.0, eps: float = 1e-6) -> None:
        super().__init__()
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        if max_weight <= 0:
            raise ValueError("max_weight must be positive.")
        self.gamma = gamma
        self.max_weight = max_weight
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = torch.abs(input - target)
        dims = tuple(range(1, l1.dim()))
        norm = l1.detach().mean(dim=dims, keepdim=True).clamp_min(self.eps)
        rel_error = l1 / norm
        weight = rel_error.pow(self.gamma).clamp(max=self.max_weight)
        return (weight * l1).mean()


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
    ) -> None:
        super().__init__()
        if not channel_schedule:
            raise ValueError("channel_schedule must be non-empty for decoder construction.")
        stages = max(len(channel_schedule) - 1, 0)
        downscale = 2 ** stages if stages > 0 else 1
        if image_size % downscale != 0:
            raise ValueError(
                f"image_size {image_size} is incompatible with channel schedule length {len(channel_schedule)}"
            )
        self.start_hw = image_size // downscale
        self.start_ch = channel_schedule[-1]
        self.image_channels = image_channels
        self.image_size = image_size

        self.project = nn.Sequential(
            nn.Linear(latent_dim, self.start_ch * self.start_hw * self.start_hw),
            nn.SiLU(inplace=True),
        )
        self.prev_adapter = nn.Sequential(
            nn.Conv2d(image_channels, self.start_ch, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        up_blocks: List[nn.Module] = []
        in_ch = self.start_ch * 2
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

    def forward(self, latent: torch.Tensor, prev_frame: Optional[torch.Tensor] = None) -> torch.Tensor:
        original_shape = latent.shape[:-1]
        latent = latent.reshape(-1, latent.shape[-1])
        projected = self.project(latent)
        decoded = projected.view(-1, self.start_ch, self.start_hw, self.start_hw)
        if prev_frame is not None:
            prev = prev_frame.reshape(-1, self.image_channels, self.image_size, self.image_size)
            prev = F.interpolate(prev, size=(self.start_hw, self.start_hw), mode="bilinear", align_corners=False)
            prev_feat = self.prev_adapter(prev)
            decoded = torch.cat([decoded, prev_feat], dim=1)
        else:
            decoded = torch.cat([decoded, torch.zeros_like(decoded)], dim=1)
        decoded = self.up_blocks(decoded)
        delta = self.head(decoded)
        delta = delta.view(*original_shape, self.image_channels, self.image_size, self.image_size)
        return delta


RECON_LOSS = HardnessWeightedMedianLoss()
JEPA_LOSS = nn.MSELoss()
SIGREG_LOSS = nn.MSELoss()


# ------------------------------------------------------------
# Configs and containers
# ------------------------------------------------------------


@dataclass
class ModelConfig:
    """Dimensional flow (bottlenecks marked with *):

        image (image_size^2·in_channels)
            ├─ Encoder(channel_schedule → appearance_dim)
            ├─ MotionEncoder(motion_channel_schedule → motion_dim)
            ▼
        (appearance_dim + motion_dim)*  ─┐
                        ├─ FiLM(action_dim) → Predictor(hidden_dim*) → embedding_dim
        action_dim ──────┘
            │
            └→ VisualizationDecoder(latent_dim* → image_size^2·in_channels)

    • image_size controls the spatial resolution feeding the encoder.
    • embedding_dim (≈ channel_schedule[-1] + motion_channel_schedule[-1]) is the latent bottleneck the predictor must match.
    • action_dim defines the controller space that modulates the predictor through FiLM layers.
    • hidden_dim is the predictor’s internal width.
    • latent_dim governs the visualization decoder’s compression when turning embeddings back into images.
    """

    in_channels: int = 3
    image_size: int = 128
    channel_schedule: Tuple[int, ...] = (32, 64, 128, 256)
    latent_dim: int = 256
    hidden_dim: int = 256
    embedding_dim: int = 256
    action_dim: int = 8
    predictor_film_layers: int = 2
    motion_channel_schedule: Optional[Tuple[int, ...]] = (32, 64)


@dataclass
class LossWeights:
    jepa: float = 1.0
    sigreg: float = 0.5
    rollout: float = 0.0
    consistency: float = 0.0


@dataclass
class TrainConfig:
    seq_len: int = 8
    batch_size: int = 8
    lr: float = 1e-4
    decoder_lr: float = 1e-4
    grad_clip: float = 0.0
    sigreg_projections: int = 64
    recon_weight: float = 1.0
    device: Optional[str] = "mps"
    total_steps: int = 10_000
    log_every_steps: int = 10
    vis_every_steps: int = 50
    vis_rows: int = 2
    vis_rollout: int = 4
    vis_gradient_norms: bool = False
    input_vis_every_steps: int = 0
    input_vis_rows: int = 4
    rollout_horizon: int = 8
    pair_vis_every_steps: int = 0
    pair_vis_rows: int = 4
    embedding_projection_samples: int = 256
    output_dir: Path = Path("out.jepa_world_model_trainer")
    data_root: Path = Path("data.gridworldkey")
    max_trajectories: Optional[int] = None

    loss_weights: LossWeights = field(default_factory=LossWeights)


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
        self.encoder = Encoder(cfg.in_channels, cfg.channel_schedule)
        enc_dim = cfg.channel_schedule[-1]
        if cfg.motion_channel_schedule:
            self.motion_encoder: Optional[Encoder] = Encoder(cfg.in_channels, cfg.motion_channel_schedule)
            motion_dim = cfg.motion_channel_schedule[-1]
        else:
            self.motion_encoder = None
            motion_dim = 0
        self.embedding_dim = enc_dim + motion_dim
        self.predictor = PredictorNetwork(
            self.embedding_dim,
            cfg.hidden_dim,
            cfg.action_dim,
            cfg.predictor_film_layers,
        )

    def encode_sequence(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, t, _, _, _ = images.shape
        embeddings: List[torch.Tensor] = []
        prev_frame: Optional[torch.Tensor] = None
        for step in range(t):
            current = images[:, step]
            feats = self.encoder(current)
            if self.motion_encoder is not None:
                if prev_frame is None:
                    delta = torch.zeros_like(current)
                else:
                    delta = current - prev_frame
                motion_feats = self.motion_encoder(delta)
                feats = torch.cat([feats, motion_feats], dim=1)
            embeddings.append(feats)
            prev_frame = current
        return {
            "embeddings": torch.stack(embeddings, dim=1),
        }


# ------------------------------------------------------------
# Loss utilities
# ------------------------------------------------------------


def jepa_loss(model: JEPAWorldModel, outputs: Dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
    """Standard one-step JEPA loss."""
    embeddings = outputs["embeddings"]
    prev_embed = embeddings[:, :-1]
    prev_actions = actions[:, :-1]
    deltas = model.predictor(prev_embed, prev_actions)
    preds = prev_embed + deltas
    target = embeddings[:, 1:].detach()
    return JEPA_LOSS(preds, target)


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
            delta = model.predictor(current, act)
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


def reconstruction_loss(decoder: VisualizationDecoder, embeddings: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    if embeddings.shape[1] < 2:
        return embeddings.new_tensor(0.0)
    delta_pred = decoder(embeddings[:, 1:], images[:, :-1])
    target_delta = images[:, 1:] - images[:, :-1]
    return RECON_LOSS(delta_pred, target_delta)


# ------------------------------------------------------------
# Training loop utilities
# ------------------------------------------------------------


def training_step(
    model: JEPAWorldModel,
    decoder: VisualizationDecoder,
    optim_world: torch.optim.Optimizer,
    optim_decoder: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor, List[List[str]]],
    cfg: TrainConfig,
    weights: LossWeights,
) -> Dict[str, float]:
    images, actions = batch[:2]
    world_params = [p for p in model.parameters()]
    device = next(model.parameters()).device
    images = images.to(device)
    actions = actions.to(device)
    outputs = model.encode_sequence(images)
    loss_jepa = jepa_loss(model, outputs, actions) if weights.jepa > 0 else images.new_tensor(0.0)
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
    loss_recon = (
        reconstruction_loss(decoder, outputs["embeddings"].detach(), images)
        if cfg.recon_weight > 0
        else images.new_tensor(0.0)
    )

    world_loss = (
        weights.jepa * loss_jepa
        + weights.sigreg * loss_sigreg
        + weights.rollout * loss_rollout
        + weights.consistency * loss_consistency
    )
    decoder_loss = cfg.recon_weight * loss_recon

    optim_world.zero_grad()
    world_loss.backward()
    if cfg.grad_clip > 0:
        world_grad_norm = float(nn.utils.clip_grad_norm_(world_params, cfg.grad_clip).item())
    else:
        world_grad_norm = grad_norm(world_params)
    optim_world.step()

    optim_decoder.zero_grad()
    decoder_loss.backward()
    decoder_grad_norm = grad_norm(decoder.parameters())
    optim_decoder.step()

    return {
        "loss_jepa": loss_jepa.item(),
        "loss_sigreg": loss_sigreg.item(),
        "loss_rollout": loss_rollout.item(),
        "loss_consistency": loss_consistency.item(),
        "loss_recon": loss_recon.item(),
        "loss_world": world_loss.item(),
        "grad_world": world_grad_norm,
        "grad_decoder": decoder_grad_norm,
    }


# ------------------------------------------------------------
# Example dataset + dataloader
# ------------------------------------------------------------


def collate_batch(
    batch: Iterable[Tuple[torch.Tensor, torch.Tensor, List[str]]]
) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
    obs, actions, paths = zip(*batch)
    obs_tensor = torch.stack(obs, dim=0)
    act_tensor = torch.stack(actions, dim=0)
    path_batch = [list(seq) for seq in paths]
    return obs_tensor, act_tensor, path_batch


class TrajectorySequenceDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, List[str]]]):
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
        return torch.stack(frames, dim=0), torch.from_numpy(action_slice), path_slice


# ------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------


def log_metrics(step: int, metrics: Dict[str, float]) -> None:
    pretty = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    print(f"[step {step}] {pretty}")


LOSS_COLUMNS = [
    "step",
    "loss_world",
    "loss_jepa",
    "loss_sigreg",
    "loss_rollout",
    "loss_consistency",
    "loss_recon",
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
    recon: List[float] = field(default_factory=list)
    grad_world: List[float] = field(default_factory=list)
    grad_decoder: List[float] = field(default_factory=list)

    def append(self, step: float, metrics: Dict[str, float]) -> None:
        self.steps.append(step)
        self.world.append(metrics["loss_world"])
        self.jepa.append(metrics["loss_jepa"])
        self.sigreg.append(metrics["loss_sigreg"])
        self.rollout.append(metrics["loss_rollout"])
        self.consistency.append(metrics["loss_consistency"])
        self.recon.append(metrics["loss_recon"])
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
            history.recon,
            history.grad_world,
            history.grad_decoder,
        ):
            writer.writerow(row)


def plot_loss_curves(history: LossHistory, out_dir: Path) -> None:
    if len(history) == 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history.steps, history.world, label="world")
    plt.plot(history.steps, history.jepa, label="jepa")
    plt.plot(history.steps, history.sigreg, label="sigreg")
    if any(val != 0.0 for val in history.rollout):
        plt.plot(history.steps, history.rollout, label="rollout")
    if any(val != 0.0 for val in history.consistency):
        plt.plot(history.steps, history.consistency, label="consistency")
    plt.plot(history.steps, history.recon, label="recon")
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


def save_embedding_projection(embeddings: torch.Tensor, path: Path) -> None:
    flat = embeddings.detach().cpu().numpy()
    b, t, d = flat.shape
    flat = flat.reshape(b * t, d)
    flat = flat - flat.mean(axis=0, keepdims=True)
    if flat.shape[0] < 2:
        return
    try:
        u, s, vt = np.linalg.svd(flat, full_matrices=False)
    except np.linalg.LinAlgError:
        jitter = np.random.normal(scale=1e-6, size=flat.shape)
        try:
            u, s, vt = np.linalg.svd(flat + jitter, full_matrices=False)
        except np.linalg.LinAlgError:
            print("Warning: embedding SVD failed to converge; skipping projection.")
            return
    coords = flat @ vt[:2].T
    colors = np.tile(np.arange(t), b)
    colors = colors / max(1, t - 1)
    plt.figure(figsize=(6, 5))
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, cmap="viridis", s=10, alpha=0.7)
    plt.title("Embedding Projection (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="Time step")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def tensor_to_uint8_image(frame: torch.Tensor) -> np.ndarray:
    array = frame.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (array * 255.0).round().astype(np.uint8)


def _short_traj_state_label(frame_path: str) -> str:
    path = Path(frame_path)
    traj = next((part for part in path.parts if part.startswith("traj_")), path.parent.name)
    return f"{traj}/{path.stem}"


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


TEXT_FONT = ImageFont.load_default()
LOSS_CMAP = plt.get_cmap("coolwarm")
GRAD_CMAP = plt.get_cmap("magma")


def _describe_action_tensor(action: torch.Tensor) -> str:
    vector = action.detach().cpu().numpy().reshape(-1)
    binary = (vector > 0.5).astype(np.uint8)
    return _describe_controller_vector_compact(binary)


def _annotate_with_text(image: np.ndarray, text: str) -> np.ndarray:
    if not text:
        return image
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    padding = 2
    text = text.strip()
    bbox = draw.textbbox((padding, padding), text, font=TEXT_FONT)
    rect = (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding)
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.text((padding, padding), text, fill=(255, 255, 255), font=TEXT_FONT)
    return np.array(pil_image)


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


def save_temporal_pair_visualization(
    out_path: Path,
    frames: torch.Tensor,
    actions: torch.Tensor,
    rows: int,
) -> None:
    if frames.shape[1] < 2:
        return
    frames = frames.detach().cpu()
    actions = actions.detach().cpu()
    batch_size = frames.shape[0]
    num_rows = min(rows, batch_size)
    order = torch.randperm(batch_size)[:num_rows]
    pairs = []
    for idx in order:
        time_idx = torch.randint(1, frames.shape[1], ()).item()
        prev_frame = tensor_to_uint8_image(frames[idx, time_idx - 1])
        next_frame = tensor_to_uint8_image(frames[idx, time_idx])
        prev_frame = _annotate_with_text(prev_frame, _describe_action_tensor(actions[idx, time_idx - 1]))
        next_frame = _annotate_with_text(next_frame, _describe_action_tensor(actions[idx, time_idx]))
        pairs.append(np.concatenate([prev_frame, next_frame], axis=1))
    grid = np.concatenate(pairs, axis=0) if pairs else np.zeros((1, 1, 3), dtype=np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path)


def save_input_batch_visualization(
    out_path: Path,
    frames: torch.Tensor,
    actions: torch.Tensor,
    rows: int,
) -> None:
    frames = frames.detach().cpu()
    actions = actions.detach().cpu()
    batch_size, seq_len = frames.shape[0], frames.shape[1]
    num_rows = min(rows, batch_size)
    if num_rows <= 0:
        return
    grid_rows: List[np.ndarray] = []
    for row_idx in range(num_rows):
        columns: List[np.ndarray] = []
        for step in range(seq_len):
            frame_img = tensor_to_uint8_image(frames[row_idx, step])
            desc = _describe_action_tensor(actions[row_idx, step])
            frame_img = _annotate_with_text(frame_img, desc)
            columns.append(frame_img)
        row_image = np.concatenate(columns, axis=1)
        grid_rows.append(row_image)
    grid = np.concatenate(grid_rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path)


def _run_git_command(args: List[str]) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return "git executable not found."
    if result.returncode != 0:
        stderr = result.stderr.strip()
        return f"Command {' '.join(args)} failed (code {result.returncode}): {stderr}"
    return result.stdout.strip()


def _serialize_for_json(value):
    if isinstance(value, dict):
        return {k: _serialize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_for_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_run_metadata(run_dir: Path, cfg: TrainConfig, model_cfg: ModelConfig) -> None:
    commit_sha = _run_git_command(["git", "rev-parse", "HEAD"])
    diff_output = _run_git_command(["git", "diff", "--patch"])
    if not diff_output.strip():
        diff_output = "No uncommitted changes."
    git_metadata = "\n".join(
        [
            "Git commit:",
            commit_sha or "Unavailable",
            "",
            "Git diff (uncommitted changes):",
            diff_output,
            "",
        ]
    )
    (run_dir / "metadata_git.txt").write_text(git_metadata)
    model_metadata = {
        "train_config": _serialize_for_json(asdict(cfg)),
        "model_config": _serialize_for_json(asdict(model_cfg)),
        "data_root": str(cfg.data_root),
    }
    toml_lines = []
    toml_lines.append("[train_config]")
    for k, v in model_metadata["train_config"].items():
        toml_lines.append(f"{k} = {json.dumps(v)}")
    toml_lines.append("")
    toml_lines.append("[model_config]")
    for k, v in model_metadata["model_config"].items():
        toml_lines.append(f"{k} = {json.dumps(v)}")
    toml_lines.append("")
    toml_lines.append(f'data_root = "{model_metadata["data_root"]}"')
    (run_dir / "metadata.txt").write_text("\n".join(toml_lines))


def _render_visualization_batch(
    model: JEPAWorldModel,
    decoder: VisualizationDecoder,
    batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[List[List[str]]]]],
    rows: int,
    rollout_steps: int,
    device: torch.device,
    selection: Optional[VisualizationSelection] = None,
    show_gradients: bool = False,
) -> Optional[Tuple[List[VisualizationSequence], str]]:
    if batch_cpu is None:
        return None
    vis_frames = batch_cpu[0].to(device)
    vis_actions = batch_cpu[1].to(device)
    frame_paths = batch_cpu[2] if len(batch_cpu) > 2 else None
    vis_outputs = model.encode_sequence(vis_frames)
    vis_embeddings = vis_outputs["embeddings"]
    if vis_frames.shape[1] < 2:
        return None
    delta_frames = decoder(vis_embeddings[:, 1:], vis_frames[:, :-1])
    batch_size = vis_frames.shape[0]
    if batch_size == 0:
        return None
    min_start = 1
    target_window = max(2, rollout_steps + 1)
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
    for row_offset, idx in enumerate(row_indices):
        start_idx = int(base_starts[row_offset].item()) if base_starts is not None else min_start
        start_idx = max(min_start, min(start_idx, max_start))
        gt_slice = vis_frames[idx, start_idx : start_idx + max_window]
        if gt_slice.shape[0] < max_window:
            continue
        action_texts: List[str] = []
        for offset in range(max_window):
            action_idx = min(start_idx + offset, vis_actions.shape[1] - 1)
            action_texts.append(_describe_action_tensor(vis_actions[idx, action_idx]))
        recon_seq: List[torch.Tensor] = []
        for offset in range(max_window):
            delta_idx = start_idx + offset - 1
            prev_frame = vis_frames[idx, delta_idx]
            delta_frame = delta_frames[idx, delta_idx]
            recon_seq.append((prev_frame + delta_frame).clamp(0, 1))
        recon_tensor = torch.stack(recon_seq, dim=0)
        rollout_frames: List[Optional[torch.Tensor]] = [None for _ in range(max_window)]
        gradient_maps: List[Optional[np.ndarray]] = [None for _ in range(max_window)]
        current_frame = gt_slice[0]
        current_embed = vis_embeddings[idx, start_idx].unsqueeze(0)
        for step in range(1, max_window):
            action = vis_actions[idx, start_idx + step - 1].unsqueeze(0)
            delta_embed = model.predictor(current_embed, action)
            next_embed = current_embed + delta_embed
            delta_next = decoder(next_embed, current_frame.unsqueeze(0))[0]
            current_frame = (current_frame + delta_next).clamp(0, 1)
            if show_gradients:
                try:
                    gradient_maps[step] = _prediction_gradient_heatmap(current_frame, gt_slice[step])
                except RuntimeError:
                    gradient_maps[step] = _loss_to_heatmap(gt_slice[step], current_frame)
            else:
                gradient_maps[step] = _loss_to_heatmap(gt_slice[step], current_frame)
            rollout_frames[step] = current_frame.detach().cpu()
            current_embed = next_embed
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
    grad_label = "Gradient Norm" if show_gradients else "Error Heatmap"
    return sequences, grad_label


def save_rollout_visualization(
    out_path: Path,
    sequence: VisualizationSequence,
    grad_label: str,
) -> None:
    seq_cols = len(sequence.labels)
    if seq_cols == 0:
        return
    row_block = 4
    fig, axes = plt.subplots(
        row_block,
        seq_cols,
        figsize=(seq_cols * 2, row_block * 1.5),
    )
    axes = np.atleast_2d(axes)

    def _imshow_tensor(ax: plt.Axes, tensor: torch.Tensor) -> None:
        ax.imshow(tensor.clamp(0, 1).permute(1, 2, 0).numpy())
        ax.axis("off")

    def _imshow_array(ax: plt.Axes, array: np.ndarray) -> None:
        if array.dtype != np.float32 and array.dtype != np.float64:
            data = array.astype(np.float32) / 255.0
        else:
            data = array
        ax.imshow(data)
        ax.axis("off")

    sample_tensor = sequence.ground_truth[0]
    height, width = sample_tensor.shape[1], sample_tensor.shape[2]
    blank = np.full((height, width, 3), 0.5, dtype=np.float32)
    row_labels = [
        "Ground Truth",
        "Rollout Prediction",
        grad_label,
        "Direct Reconstruction",
    ]

    for row_idx in range(row_block):
        axes[row_idx, 0].text(
            -0.12,
            0.5,
            row_labels[row_idx],
            transform=axes[row_idx, 0].transAxes,
            va="center",
            ha="right",
            fontsize=8,
            rotation=90,
        )
    for col in range(seq_cols):
        gt_ax = axes[0, col]
        rollout_ax = axes[1, col]
        grad_ax = axes[2, col]
        recon_ax = axes[3, col]
        gt_img = tensor_to_uint8_image(sequence.ground_truth[col])
        if sequence.actions and col < len(sequence.actions):
            gt_img = _annotate_with_text(gt_img, sequence.actions[col])
        gt_ax.imshow(gt_img)
        gt_ax.axis("off")
        gt_ax.set_title(sequence.labels[col], fontsize=8)
        recon_tensor = sequence.reconstructions[col]
        _imshow_tensor(recon_ax, recon_tensor)
        rollout_frame = sequence.rollout[col]
        if rollout_frame is None:
            _imshow_array(rollout_ax, blank)
        else:
            _imshow_tensor(rollout_ax, rollout_frame)
        grad_map = sequence.gradients[col]
        if grad_map is None:
            _imshow_array(grad_ax, blank)
        else:
            _imshow_array(grad_ax, grad_map)
    fig.suptitle("JEPA Rollout Visualization", fontsize=12)
    fig.tight_layout(rect=(0.08, 0.02, 1.0, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def save_rollout_sequence_batch(
    template_dir: Path,
    sequences: List[VisualizationSequence],
    grad_label: str,
    global_step: int,
) -> None:
    if not sequences:
        return
    base_parent = template_dir.parent
    base_name = template_dir.name
    for idx, sequence in enumerate(sequences):
        sample_dir = base_parent / f"{base_name}_{idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        out_path = sample_dir / f"rollout_{global_step:07d}.png"
        save_rollout_visualization(out_path, sequence, grad_label)


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
    inputs_vis_dir: Optional[Path]
    pair_vis_dir: Optional[Path]
    if cfg.input_vis_every_steps > 0:
        inputs_vis_dir = run_dir / "vis_inputs"
        inputs_vis_dir.mkdir(parents=True, exist_ok=True)
    else:
        inputs_vis_dir = None
    if cfg.pair_vis_every_steps > 0:
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
    dataset_action_dim = getattr(dataset, "action_dim", model_cfg.action_dim)
    assert dataset_action_dim == 8, f"Expected action_dim 8, got {dataset_action_dim}"
    if model_cfg.action_dim != dataset_action_dim:
        model_cfg = replace(model_cfg, action_dim=dataset_action_dim)
    model = JEPAWorldModel(model_cfg).to(device)
    decoder = VisualizationDecoder(
        model.embedding_dim,
        model_cfg.in_channels,
        model_cfg.image_size,
        model_cfg.channel_schedule,
    ).to(device)
    optim_world = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    optim_decoder = torch.optim.Adam(decoder.parameters(), lr=cfg.decoder_lr)
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
        metrics = training_step(model, decoder, optim_world, optim_decoder, batch, cfg, weights)
        if cfg.log_every_steps > 0 and global_step % cfg.log_every_steps == 0:
            log_metrics(global_step, metrics)
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
            and cfg.input_vis_every_steps > 0
            and global_step % cfg.input_vis_every_steps == 0
            and rolling_batch_cpu is not None
        ):
            save_input_batch_visualization(
                inputs_vis_dir / f"inputs_{global_step:07d}.png",
                rolling_batch_cpu[0],
                rolling_batch_cpu[1],
                cfg.input_vis_rows,
            )
        if (
            pair_vis_dir is not None
            and cfg.pair_vis_every_steps > 0
            and global_step % cfg.pair_vis_every_steps == 0
            and rolling_batch_cpu is not None
        ):
            save_temporal_pair_visualization(
                pair_vis_dir / f"pairs_{global_step:07d}.png",
                rolling_batch_cpu[0],
                rolling_batch_cpu[1],
                cfg.pair_vis_rows,
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
                    device,
                    fixed_selection,
                    cfg.vis_gradient_norms,
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
                    device,
                    None,
                    cfg.vis_gradient_norms,
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
