#!/usr/bin/env python3
"""Training loop for the JEPA world model with local/global specialization and SIGReg."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Annotated, Any, Dict, Iterable, List, Optional, Sequence, Tuple

import math
import random

import csv
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import tomli_w
import tyro
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from datetime import datetime
from time import perf_counter

from recon.data import list_trajectories, load_frame_as_tensor
from utils.device_utils import pick_device
from jepa_world_model.conv_encoder_decoder import Encoder as ConvEncoder, VisualizationDecoder as ConvVisualizationDecoder
from jepa_world_model.loss import FocalL1Loss, HardnessWeightedL1Loss, HardnessWeightedMSELoss, HardnessWeightedMedianLoss
from jepa_world_model.metadata import write_run_metadata
from jepa_world_model.vis import (
    describe_action_tensor,
    save_embedding_projection,
    save_input_batch_visualization,
    save_rollout_sequence_batch,
    save_temporal_pair_visualization,
)
from jepa_world_model.vis_hard_samples import save_hard_example_grid



# ------------------------------------------------------------
# Model components
# ------------------------------------------------------------

Encoder = ConvEncoder
VisualizationDecoder = ConvVisualizationDecoder
LegacyEncoder = ConvEncoder
LegacyVisualizationDecoder = ConvVisualizationDecoder



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
        self.delta_scale = 1.0
        self.use_delta_squash = False

    def forward(self, embeddings: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if embeddings.shape[:-1] != actions.shape[:-1]:
            raise ValueError("Embeddings and actions must share leading dimensions for predictor conditioning.")
        original_shape = embeddings.shape[:-1]
        emb_flat = embeddings.reshape(-1, embeddings.shape[-1])
        act_embed = self.action_embed(actions).reshape(-1, self.action_embed_dim)
        hidden = self.activation(self.in_proj(emb_flat))
        hidden = self.film(hidden, act_embed)
        hidden = self.activation(self.hidden_proj(hidden))
        hidden = self.film(hidden, act_embed)
        raw_delta = self.out_proj(hidden)
        if self.use_delta_squash:
            delta = torch.tanh(raw_delta) * self.delta_scale
        else:
            delta = raw_delta * self.delta_scale
        action_logits = self.action_head(hidden)
        pred = emb_flat + delta
        return (
            pred.view(*original_shape, pred.shape[-1]),
            delta.view(*original_shape, delta.shape[-1]),
            action_logits.view(*original_shape, action_logits.shape[-1]),
        )

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


RECON_LOSS = HardnessWeightedL1Loss()
JEPA_LOSS = nn.MSELoss()
SIGREG_LOSS = nn.MSELoss()
ACTION_RECON_LOSS = nn.BCEWithLogitsLoss()
EMA_CONSISTENCY_LOSS = nn.MSELoss()


# ------------------------------------------------------------
# Configs and containers
# ------------------------------------------------------------


def _derive_encoder_schedule(embedding_dim: int, num_layers: int) -> Tuple[int, ...]:
    """Derive a channel schedule that doubles each layer and ends at embedding_dim."""
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


def _suggest_encoder_schedule(embedding_dim: int, num_layers: int) -> str:
    """Generate a suggested encoder_schedule for error messages."""
    try:
        suggested = _derive_encoder_schedule(embedding_dim, num_layers)
        return f"encoder_schedule={suggested}"
    except ValueError:
        # If we can't derive, suggest a simple pattern
        return f"encoder_schedule with {num_layers} layers ending in {embedding_dim}"


@dataclass
class ModelConfig:
    """Dimensional flow (bottlenecks marked with *):

        image (image_size^2·in_channels)
            └─ Encoder(encoder_schedule → pool → encoder_schedule[-1]*)
            ▼
        encoder_schedule[-1]*  ─────────────┐
                        ├─ FiLM(action_dim) → Predictor(hidden_dim*) → encoder_schedule[-1]
        action_dim ──────┘
            │
            └→ VisualizationDecoder(decoder_schedule → image_size^2·in_channels)

    • image_size controls the spatial resolution feeding the encoder.
    • encoder_schedule defines the encoder conv layer widths; encoder_schedule[-1] is the embedding dim.
    • decoder_schedule defines the decoder conv layer widths (defaults to encoder_schedule if not set).
    • action_dim defines the controller space that modulates the predictor through FiLM layers.
    • hidden_dim is the predictor's internal width.
    • image_size must be divisible by 2**len(encoder_schedule) for the encoder.
    """

    in_channels: int = 3
    image_size: int = 64
    hidden_dim: int = 512
    encoder_schedule: Tuple[int, ...] = (32, 64, 128, 256)
    decoder_schedule: Optional[Tuple[int, ...]] = (32, 64, 64, 128)
    action_dim: int = 8
    predictor_film_layers: int = 2

    @property
    def embedding_dim(self) -> int:
        """The embedding dimension is encoder_schedule[-1]."""
        return self.encoder_schedule[-1]

    def __post_init__(self) -> None:
        if not self.encoder_schedule:
            raise ValueError("encoder_schedule must be non-empty.")

        # Validate image_size is divisible by encoder stride
        num_layers = len(self.encoder_schedule)
        stride = 2 ** num_layers
        if self.image_size % stride != 0:
            raise ValueError(
                f"image_size={self.image_size} must be divisible by 2**len(encoder_schedule)={stride}."
            )

        # Validate decoder_schedule if provided
        if self.decoder_schedule is not None:
            decoder_stride = 2 ** len(self.decoder_schedule)
            if self.image_size % decoder_stride != 0:
                raise ValueError(
                    f"image_size={self.image_size} must be divisible by 2**len(decoder_schedule)={decoder_stride}."
                )


@dataclass
class LossWeights:
    jepa: float = 1.0
    sigreg: float = 1.0
    recon: float = 0.0
    recon_patch: float = 0.0
    recon_multi_gauss: float = 0.0
    recon_multi_box: float = 1.0
    action_recon: float = 0.0
    delta: float = 0.0
    rollout: float = 0.0
    consistency: float = 0.0
    ema_consistency: float = 0.0

@dataclass
class LossEMAConfig:
    momentum: float = 0.99

@dataclass
class LossRolloutConfig:
    horizon: int = 8

@dataclass
class LossSigRegConfig:
    projections: int = 64

@dataclass
class LossReconPatchConfig:
    patch_sizes: Tuple[int, ...] = (32,)

@dataclass
class LossMultiScaleGaussReconConfig:
    # kernel_sizes: spatial support per scale (analogous to patch sizes); length = number of pyramid scales (each level is 2× downsampled).
    kernel_sizes: Tuple[int, ...] = (32,)
    # sigmas: Gaussian blur stddev per scale; larger sigma smooths hardness over a wider area.
    sigmas: Tuple[float, ...] = (16.0,)
    # betas: hardness exponents per scale; >0 upweights harder regions.
    betas: Tuple[float, ...] = (2.0,)
    # lambdas: per-scale weights to balance contributions across scales.
    lambdas: Tuple[float, ...] = (1.0,)
    # max_weight: optional clamp for hardness weights to avoid extreme scaling.
    max_weight: float = 100.0
    # strides: optional stride for the blur (reduces compute, may downsample before reprojecting).
    strides: Tuple[int, ...] = (1,)


@dataclass
class LossMultiScaleBoxReconConfig:
    # kernel_sizes: spatial support per scale (analogous to patch sizes); length = number of pyramid scales (each level is 2× downsampled).
    kernel_sizes: Tuple[int, ...] = (8, 16, 16,)
    # betas: hardness exponents per scale; >0 upweights harder regions.
    betas: Tuple[float, ...] = (2.0, 2.0, 2.0,)
    # lambdas: per-scale weights to balance contributions across scales.
    lambdas: Tuple[float, ...] = (0.333, 0.333, 0.333,)
    # max_weight: optional clamp for hardness weights to avoid extreme scaling.
    max_weight: float = 100.0
    # strides: optional stride for the blur (reduces compute, may downsample before reprojecting).
    strides: Tuple[int, ...] = (4, 8, 8,)

@dataclass
class VisConfig:
    rows: int = 2
    rollout: int = 4
    columns: int = 8
    gradient_norms: bool = True
    log_deltas: bool = False
    embedding_projection_samples: int = 256

@dataclass
class HardExampleConfig:
    reservoir: int = 0
    mix_ratio: float = 0.5
    vis_rows: int = 4
    vis_columns: int = 6

@dataclass
class DebugVisualization:
    input_vis_every_steps: int = 0
    input_vis_rows: int = 4
    pair_vis_every_steps: int = 0
    pair_vis_rows: int = 4


@dataclass
class TrainConfig:
    data_root: Path = Path("data.gridworldkey_wander_to_key")
    output_dir: Path = Path("out.jepa_world_model_trainer")
    log_every_steps: int = 10
    vis_every_steps: int = 50
    steps: int = 100_000
    show_timing_breakdown: bool = True
    seed: Optional[int] = 0

    # A validation split only materializes when multiple trajectories exist; with a single traj, keep val_fraction at 0.
    val_fraction: float = 0
    val_split_seed: int = 0

    # Dataset & batching
    max_trajectories: Optional[int] = None
    seq_len: int = 8
    batch_size: int = 8

    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.03
    device: Optional[str] = "mps"

    # Loss configuration
    loss_weights: LossWeights = field(default_factory=LossWeights)

    # Specific losses
    ema: LossEMAConfig = field(default_factory=LossEMAConfig)
    rollout: LossRolloutConfig = field(default_factory=LossRolloutConfig)
    sigreg: LossSigRegConfig = field(default_factory=LossSigRegConfig)
    patch_recon: LossReconPatchConfig = field(default_factory=LossReconPatchConfig)
    recon_multi_gauss: LossMultiScaleGaussReconConfig = field(default_factory=LossMultiScaleGaussReconConfig)
    recon_multi_box: LossMultiScaleBoxReconConfig = field(default_factory=LossMultiScaleBoxReconConfig)

    # Visualization
    vis: VisConfig = field(default_factory=VisConfig)
    hard_example: HardExampleConfig = field(default_factory=HardExampleConfig)
    debug_visualization: DebugVisualization = field(default_factory=DebugVisualization)

    # CLI-only field (not part of training config, used for experiment metadata)
    title: Annotated[Optional[str], tyro.conf.arg(aliases=["-m"])] = None


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
        self.encoder = Encoder(
            cfg.in_channels,
            cfg.encoder_schedule,
            cfg.image_size,
        )
        # latent_dim is encoder_schedule[-1]
        self.embedding_dim = self.encoder.latent_dim
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
            pooled = self.encoder(current)
            embeddings.append(pooled)
        return {
            "embeddings": torch.stack(embeddings, dim=1),
        }


# ------------------------------------------------------------
# Loss utilities
# ------------------------------------------------------------


def jepa_loss(
    model: JEPAWorldModel, outputs: Dict[str, torch.Tensor], actions: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard one-step JEPA loss plus action logits from the predictor."""
    embeddings = outputs["embeddings"]
    if embeddings.shape[1] < 2:
        zero = embeddings.new_tensor(0.0)
        logits = embeddings.new_zeros((*embeddings.shape[:2], actions.shape[-1]))
        delta = embeddings.new_zeros(embeddings[:, :-1].shape)
        return zero, logits, delta
    prev_embed = embeddings[:, :-1]
    prev_actions = actions[:, :-1]
    preds, delta_pred, action_logits = model.predictor(prev_embed, prev_actions)
    target = embeddings[:, 1:].detach()
    return JEPA_LOSS(preds, target), action_logits, delta_pred


def delta_prediction_loss(
    delta_pred: torch.Tensor, embeddings: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Auxiliary loss comparing predicted and target latent deltas."""
    if embeddings.shape[1] < 2:
        zero = embeddings.new_tensor(0.0)
        return zero, zero, zero
    delta_target = (embeddings[:, 1:] - embeddings[:, :-1]).detach()
    loss = F.mse_loss(delta_pred, delta_target)
    pred_norm = delta_pred.detach().norm(dim=-1).mean()
    target_norm = delta_target.norm(dim=-1).mean()
    return loss, pred_norm, target_norm


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
            pred, _, _ = model.predictor(current, act)
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


def gaussian_kernel_1d(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if size <= 0:
        raise ValueError("Gaussian kernel size must be positive.")
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    kernel = torch.exp(-0.5 * (coords / max(sigma, 1e-6)) ** 2)
    kernel = kernel / kernel.sum().clamp_min(1e-6)
    return kernel


def gaussian_blur_separable_2d(x: torch.Tensor, kernel_1d: torch.Tensor, stride: int = 1) -> torch.Tensor:
    """Apply separable Gaussian blur to a single-channel map."""
    if stride <= 0:
        raise ValueError("Gaussian blur stride must be positive.")
    k = kernel_1d.numel()
    pad = k // 2
    vert = F.conv2d(x, kernel_1d.view(1, 1, k, 1), padding=pad, stride=stride)
    horiz = F.conv2d(vert, kernel_1d.view(1, 1, 1, k), padding=pad, stride=stride)
    return horiz


def box_kernel_2d(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Normalized box filter over an NxN window."""
    if size <= 0:
        raise ValueError("Box kernel size must be positive.")
    kernel = torch.ones((1, 1, size, size), device=device, dtype=dtype)
    return kernel / kernel.sum().clamp_min(1e-6)


def _build_feature_pyramid(
    pred: torch.Tensor, target: torch.Tensor, num_scales: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    preds = [pred]
    targets = [target]
    for _ in range(1, num_scales):
        pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
        target = F.avg_pool2d(target, kernel_size=2, stride=2)
        preds.append(pred)
        targets.append(target)
    return preds, targets


def multi_scale_hardness_loss_gaussian(
    preds: List[torch.Tensor],
    targets: List[torch.Tensor],
    kernel_sizes: Sequence[int],
    sigmas: Sequence[float],
    betas: Sequence[float],
    lambdas: Sequence[float],
    strides: Sequence[int],
    max_weight: float = 100.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute hardness-weighted loss over multiple scales.

    • kernel_sizes: spatial support per scale (similar to patch size).
    • sigmas: Gaussian blur stddev per scale (controls how far hardness spreads).
    • betas: hardness exponents; higher beta emphasizes harder regions.
    • lambdas: per-scale weights to balance contributions.
    • strides: blur stride per scale (reduces compute; resampled back to original size if needed).
    """
    if max_weight <= 0:
        raise ValueError("max_weight must be positive.")
    if not (
        len(preds)
        == len(targets)
        == len(kernel_sizes)
        == len(sigmas)
        == len(betas)
        == len(lambdas)
        == len(strides)
    ):
        raise ValueError("preds, targets, kernel_sizes, sigmas, betas, lambdas, and strides must share length.")
    if not preds:
        return torch.tensor(0.0, device="cpu")
    total = preds[0].new_tensor(0.0)
    for idx, (p, t) in enumerate(zip(preds, targets)):
        if p.shape != t.shape:
            raise ValueError(f"Pred/target shape mismatch at scale {idx}: {p.shape} vs {t.shape}")
        k = int(kernel_sizes[idx])
        sigma = float(sigmas[idx])
        beta = float(betas[idx])
        lam = float(lambdas[idx])
        stride = int(strides[idx])
        if stride <= 0:
            raise ValueError("Gaussian hardness stride must be positive.")
        per_pixel = ((p - t) ** 2).mean(dim=1, keepdim=True)
        per_pixel_detached = per_pixel.detach()
        g1d = gaussian_kernel_1d(k, sigma, device=per_pixel.device, dtype=per_pixel.dtype)
        blurred_weight = gaussian_blur_separable_2d(per_pixel_detached, g1d, stride=stride)
        if blurred_weight.shape[-2:] != per_pixel.shape[-2:]:
            blurred_weight = F.interpolate(blurred_weight, size=per_pixel.shape[-2:], mode="nearest")
        weight = (blurred_weight + eps).pow(beta).clamp(max=max_weight)
        scale_loss = (weight * (per_pixel + eps)).mean()
        total = total + lam * scale_loss
    return total


def multi_scale_hardness_loss_box(
    preds: List[torch.Tensor],
    targets: List[torch.Tensor],
    kernel_sizes: Sequence[int],
    betas: Sequence[float],
    lambdas: Sequence[float],
    strides: Sequence[int],
    max_weight: float = 100.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute hardness-weighted loss over multiple scales with a convolutional approximation of patch-style hardness.
    """
    if max_weight <= 0:
        raise ValueError("max_weight must be positive.")
    if not (
        len(preds)
        == len(targets)
        == len(kernel_sizes)
        == len(betas)
        == len(lambdas)
        == len(strides)
    ):
        raise ValueError("preds, targets, kernel_sizes, betas, lambdas, and strides must share length.")
    if not preds:
        return torch.tensor(0.0, device="cpu")
    total = preds[0].new_tensor(0.0)
    for idx, (p, t) in enumerate(zip(preds, targets)):
        if p.shape != t.shape:
            raise ValueError(f"Pred/target shape mismatch at scale {idx}: {p.shape} vs {t.shape}")
        k = int(kernel_sizes[idx])
        beta = float(betas[idx])
        lam = float(lambdas[idx])
        stride = int(strides[idx])
        if stride <= 0:
            raise ValueError("Box hardness stride must be positive.")
        _, _, h, w = p.shape
        k_eff = min(k, h, w)
        # Clamp kernel/stride to valid spatial support so deeper pyramid levels still contribute.
        stride_eff = max(1, min(stride, k_eff))
        per_pixel_l1 = (p - t).abs().mean(dim=1, keepdim=True)  # Bx1xHxW
        per_pixel_detached = per_pixel_l1.detach()
        # Valid pooling (no padding) to avoid border bleed; stride can mimic patch overlap (e.g., k//2).
        norm = F.avg_pool2d(per_pixel_detached, kernel_size=k_eff, stride=stride_eff, padding=0)
        coverage = F.avg_pool2d(torch.ones_like(per_pixel_detached), kernel_size=k_eff, stride=stride_eff, padding=0)
        # Upsample pooled maps back to full resolution for per-pixel weighting.
        if norm.shape[-2:] != per_pixel_l1.shape[-2:]:
            norm = F.interpolate(norm, size=per_pixel_l1.shape[-2:], mode="bilinear", align_corners=False)
            coverage = F.interpolate(coverage, size=per_pixel_l1.shape[-2:], mode="bilinear", align_corners=False)
        norm = norm.clamp_min(eps)
        coverage = coverage.clamp_min(eps)
        weight = (per_pixel_detached / norm).pow(beta).clamp(max=max_weight)
        weighted_sum = (weight * per_pixel_l1).sum()
        scale_loss = weighted_sum / coverage.sum()
        total = total + lam * scale_loss
    return total


def patch_recon_loss(
    recon: torch.Tensor, target: torch.Tensor, patch_sizes: Sequence[int]
) -> torch.Tensor:
    """
    Compute reconstruction loss over a grid of overlapping patches for multiple sizes.

    Rationale: keep supervision in image space without adding feature taps or extra
    forward passes—cheap to bolt on and works with the existing decoder output.
    A more traditional multi-scale hardness term could sample patches from intermediate
    CNN layers (feature pyramids, perceptual losses) with size-aware weights, but that
    would require exposing/retaining feature maps and increase memory/compute.
    """
    if not patch_sizes:
        raise ValueError("patch_recon_loss requires at least one patch size.")
    h, w = recon.shape[-2], recon.shape[-1]
    total = recon.new_tensor(0.0)
    count = 0

    def _grid_indices(limit: int, size: int) -> Iterable[int]:
        step = max(1, size // 2)  # 50% overlap by default
        positions = list(range(0, limit - size + 1, step))
        if positions and positions[-1] != limit - size:
            positions.append(limit - size)
        elif not positions:
            positions = [0]
        return positions

    for patch_size in patch_sizes:
        if patch_size <= 0:
            raise ValueError("patch_recon_loss requires all patch sizes to be > 0.")
        if patch_size > h or patch_size > w:
            raise ValueError(f"patch_size={patch_size} exceeds recon dimensions {(h, w)}.")
        row_starts = _grid_indices(h, patch_size)
        col_starts = _grid_indices(w, patch_size)
        for rs in row_starts:
            for cs in col_starts:
                recon_patch = recon[..., rs : rs + patch_size, cs : cs + patch_size]
                target_patch = target[..., rs : rs + patch_size, cs : cs + patch_size]
                total = total + RECON_LOSS(recon_patch, target_patch)
                count += 1
    return total / count if count > 0 else recon.new_tensor(0.0)


def multi_scale_recon_loss_gauss(
    recon: torch.Tensor,
    target: torch.Tensor,
    cfg: LossMultiScaleGaussReconConfig,
) -> torch.Tensor:
    """Multi-scale hardness-weighted reconstruction over an image pyramid."""
    # Flatten batch/time to apply pyramid on spatial dims only.
    recon_bt = recon.reshape(-1, *recon.shape[2:])
    target_bt = target.reshape(-1, *target.shape[2:])
    num_scales = len(cfg.kernel_sizes)
    preds_scales, targets_scales = _build_feature_pyramid(recon_bt, target_bt, num_scales)
    loss_raw = multi_scale_hardness_loss_gaussian(
        preds_scales,
        targets_scales,
        cfg.kernel_sizes,
        cfg.sigmas,
        cfg.betas,
        cfg.lambdas,
        cfg.strides,
        cfg.max_weight,
    )
    return loss_raw


def multi_scale_recon_loss_box(
    recon: torch.Tensor,
    target: torch.Tensor,
    cfg: LossMultiScaleBoxReconConfig,
) -> torch.Tensor:
    """Multi-scale hardness-weighted reconstruction over an image pyramid using box filters."""
    recon_bt = recon.reshape(-1, *recon.shape[2:])
    target_bt = target.reshape(-1, *target.shape[2:])
    num_scales = len(cfg.kernel_sizes)
    preds_scales, targets_scales = _build_feature_pyramid(recon_bt, target_bt, num_scales)
    loss_raw = multi_scale_hardness_loss_box(
        preds_scales,
        targets_scales,
        cfg.kernel_sizes,
        cfg.betas,
        cfg.lambdas,
        cfg.strides,
        cfg.max_weight,
    )
    return loss_raw


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
    metrics, difficulty_info, world_loss, grads = _compute_losses_and_metrics(
        model,
        decoder,
        batch,
        cfg,
        weights,
        ema_model=ema_model,
        ema_momentum=ema_momentum,
        track_hard_examples=True,
        for_training=True,
        optimizer=optimizer,
    )
    return metrics, difficulty_info


@torch.no_grad()
def validation_step(
    model: JEPAWorldModel,
    decoder: VisualizationDecoder,
    batch: Tuple[torch.Tensor, torch.Tensor, List[List[str]], torch.Tensor],
    cfg: TrainConfig,
    weights: LossWeights,
) -> Tuple[Dict[str, float], Optional[BatchDifficultyInfo]]:
    metrics, difficulty_info, _, _ = _compute_losses_and_metrics(
        model,
        decoder,
        batch,
        cfg,
        weights,
        ema_model=None,
        ema_momentum=0.0,
        track_hard_examples=True,
        for_training=False,
        optimizer=None,
    )
    return metrics, difficulty_info


def _compute_losses_and_metrics(
    model: JEPAWorldModel,
    decoder: VisualizationDecoder,
    batch: Tuple[torch.Tensor, torch.Tensor, List[List[str]], torch.Tensor],
    cfg: TrainConfig,
    weights: LossWeights,
    ema_model: Optional[JEPAWorldModel],
    ema_momentum: float,
    track_hard_examples: bool,
    for_training: bool,
    optimizer: Optional[torch.optim.Optimizer],
) -> Tuple[Dict[str, float], Optional[BatchDifficultyInfo], torch.Tensor, Tuple[float, float]]:
    images, actions = batch[0], batch[1]
    batch_paths = batch[2] if len(batch) > 2 else None
    batch_indices = batch[3] if len(batch) > 3 else None
    device = next(model.parameters()).device
    images = images.to(device)
    actions = actions.to(device)

    outputs = model.encode_sequence(images)

    need_recon = (
        weights.recon > 0
        or track_hard_examples
        or weights.recon_multi_gauss > 0
        or weights.recon_multi_box > 0
        or weights.recon_patch > 0
    )
    recon: Optional[torch.Tensor] = None
    if need_recon:
        recon = decoder(outputs["embeddings"])

    loss_jepa_raw, action_logits, delta_pred = jepa_loss(model, outputs, actions)
    loss_jepa = loss_jepa_raw if weights.jepa > 0 else images.new_tensor(0.0)
    loss_delta_raw, delta_pred_norm, delta_target_norm = delta_prediction_loss(delta_pred, outputs["embeddings"])
    loss_delta = loss_delta_raw if weights.delta > 0 else images.new_tensor(0.0)
    prev_actions = actions[:, :-1]
    if weights.action_recon > 0 and prev_actions.numel() > 0:
        loss_action = ACTION_RECON_LOSS(action_logits, prev_actions)
    else:
        loss_action = images.new_tensor(0.0)

    loss_rollout = (
        rollout_loss(model, outputs["embeddings"], actions, cfg.rollout.horizon)
        if weights.rollout > 0
        else images.new_tensor(0.0)
    )
    loss_consistency = (
        latent_consistency_loss(outputs["embeddings"])
        if weights.consistency > 0
        else images.new_tensor(0.0)
    )
    loss_sigreg = (
        sigreg_loss(outputs["embeddings"], cfg.sigreg.projections)
        if weights.sigreg > 0
        else images.new_tensor(0.0)
    )
    loss_ema_consistency = images.new_tensor(0.0)
    if ema_model is not None and weights.ema_consistency > 0:
        with torch.no_grad():
            ema_outputs = ema_model.encode_sequence(images)
        loss_ema_consistency = ema_latent_consistency_loss(outputs["embeddings"], ema_outputs["embeddings"])

    if weights.recon > 0 and recon is not None:
        loss_recon = RECON_LOSS(recon, images)
    else:
        loss_recon = images.new_tensor(0.0)

    if weights.recon_multi_gauss > 0 and recon is not None:
        loss_recon_multi_gauss = multi_scale_recon_loss_gauss(recon, images, cfg.recon_multi_gauss)
    else:
        loss_recon_multi_gauss = images.new_tensor(0.0)

    if weights.recon_multi_box > 0 and recon is not None:
        loss_recon_multi_box = multi_scale_recon_loss_box(recon, images, cfg.recon_multi_box)
    else:
        loss_recon_multi_box = images.new_tensor(0.0)

    if weights.recon_patch > 0 and recon is not None:
        loss_recon_patch = patch_recon_loss(recon, images, cfg.patch_recon.patch_sizes)
    else:
        loss_recon_patch = images.new_tensor(0.0)

    world_loss = (
        weights.jepa * loss_jepa
        + weights.delta * loss_delta
        + weights.sigreg * loss_sigreg
        + weights.rollout * loss_rollout
        + weights.consistency * loss_consistency
        + weights.ema_consistency * loss_ema_consistency
        + weights.action_recon * loss_action
        + weights.recon * loss_recon
        + weights.recon_multi_gauss * loss_recon_multi_gauss
        + weights.recon_multi_box * loss_recon_multi_box
        + weights.recon_patch * loss_recon_patch
    )

    world_grad_norm = 0.0
    decoder_grad_norm = 0.0
    if for_training and optimizer is not None:
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
        "loss_recon_multi_gauss": loss_recon_multi_gauss.item(),
        "loss_recon_multi_box": loss_recon_multi_box.item(),
        "loss_recon_patch": loss_recon_patch.item(),
        "loss_action": loss_action.item(),
        "loss_delta": loss_delta.item(),
        "delta_pred_norm": delta_pred_norm.item(),
        "delta_target_norm": delta_target_norm.item(),
        "loss_world": world_loss.item(),
    }
    if for_training:
        metrics["grad_world"] = world_grad_norm
        metrics["grad_decoder"] = decoder_grad_norm

    return metrics, difficulty_info, world_loss, (world_grad_norm, decoder_grad_norm)


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
        included_trajectories: Optional[Sequence[str]] = None,
    ) -> None:
        self.root = Path(root)
        self.seq_len = seq_len
        self.image_hw = image_hw
        if self.seq_len < 1:
            raise ValueError("seq_len must be positive.")
        trajectories = list_trajectories(self.root)
        if included_trajectories is not None:
            include_set = set(included_trajectories)
            items = [(name, frames) for name, frames in trajectories.items() if name in include_set]
        else:
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
            raise AssertionError(f"No usable sequences found under {self.root}")
        if self.action_dim is None:
            raise AssertionError("Failed to infer action dimensionality.")

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


def _split_trajectories(
    root: Path, max_traj: Optional[int], val_fraction: float, seed: int
) -> Tuple[List[str], List[str]]:
    traj_names = sorted(list(list_trajectories(root).keys()))
    if max_traj is not None:
        traj_names = traj_names[:max_traj]
    if not traj_names:
        return [], []
    rng = random.Random(seed)
    rng.shuffle(traj_names)
    if val_fraction <= 0:
        return traj_names, []
    if val_fraction >= 1.0:
        return [], traj_names
    val_count = max(1, int(len(traj_names) * val_fraction)) if len(traj_names) > 1 else 0
    val_count = min(val_count, max(0, len(traj_names) - 1))
    val_names = traj_names[:val_count]
    train_names = traj_names[val_count:]
    return train_names, val_names


def _seed_everything(seed: Optional[int]) -> Tuple[int, random.Random]:
    """Seed Python, NumPy, and torch RNGs and return a dedicated Python RNG."""
    seed_value = 0 if seed is None else int(seed)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    return seed_value, random.Random(seed_value)


# ------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------


def _format_elapsed_time(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def log_metrics(
    step: int,
    metrics: Dict[str, float],
    weights: LossWeights,
    samples_per_sec: Optional[float] = None,
    elapsed_seconds: Optional[float] = None,
) -> None:
    filtered = dict(metrics)
    if weights.jepa <= 0:
        filtered.pop("loss_jepa", None)
    if weights.delta <= 0:
        filtered.pop("loss_delta", None)
    if weights.sigreg <= 0:
        filtered.pop("loss_sigreg", None)
    if weights.rollout <= 0:
        filtered.pop("loss_rollout", None)
    if weights.consistency <= 0:
        filtered.pop("loss_consistency", None)
    if weights.ema_consistency <= 0:
        filtered.pop("loss_ema_consistency", None)
    if weights.recon <= 0:
        filtered.pop("loss_recon", None)
    if weights.recon_multi_gauss <= 0:
        filtered.pop("loss_recon_multi_gauss", None)
    if weights.recon_multi_box <= 0:
        filtered.pop("loss_recon_multi_box", None)
    if weights.recon_patch <= 0:
        filtered.pop("loss_recon_patch", None)
    # Always show val loss if present
    if "loss_val_world" in metrics:
        filtered["loss_val_world"] = metrics["loss_val_world"]
    if "loss_val_recon" in metrics:
        filtered["loss_val_recon"] = metrics["loss_val_recon"]
    if "loss_val_recon_multi_gauss" in metrics:
        filtered["loss_val_recon_multi_gauss"] = metrics["loss_val_recon_multi_gauss"]
    if "loss_val_recon_multi_box" in metrics:
        filtered["loss_val_recon_multi_box"] = metrics["loss_val_recon_multi_box"]
    if "loss_val_recon_patch" in metrics:
        filtered["loss_val_recon_patch"] = metrics["loss_val_recon_patch"]
    if weights.action_recon <= 0:
        filtered.pop("loss_action", None)
    pretty = ", ".join(f"{k}: {v:.4f}" for k, v in filtered.items())
    summary_parts: List[str] = []
    if pretty:
        summary_parts.append(pretty)
    if samples_per_sec is not None:
        summary_parts.append(f"{samples_per_sec:.1f} samples/s")
    if elapsed_seconds is not None and elapsed_seconds >= 0:
        summary_parts.append(f"elapsed {_format_elapsed_time(elapsed_seconds)}")
    summary = " | ".join(summary_parts)
    if summary:
        print(f"[step {step}] {summary}")
    else:
        print(f"[step {step}]")


LOSS_COLUMNS = [
    "step",
    "elapsed_seconds",
    "cumulative_flops",
    "loss_world",
    "loss_val_world",
    "loss_val_recon",
    "loss_val_recon_multi_gauss",
    "loss_val_recon_multi_box",
    "loss_val_recon_patch",
    "loss_jepa",
    "loss_sigreg",
    "loss_rollout",
    "loss_consistency",
    "loss_ema_consistency",
    "loss_recon",
    "loss_recon_multi_gauss",
    "loss_recon_multi_box",
    "loss_recon_patch",
    "loss_action",
    "loss_delta",
    "delta_pred_norm",
    "delta_target_norm",
    "grad_world",
    "grad_decoder",
]


@dataclass
class LossHistory:
    steps: List[float] = field(default_factory=list)
    elapsed_seconds: List[float] = field(default_factory=list)
    cumulative_flops: List[float] = field(default_factory=list)
    world: List[float] = field(default_factory=list)
    val_world: List[float] = field(default_factory=list)
    val_recon: List[float] = field(default_factory=list)
    val_recon_multi_gauss: List[float] = field(default_factory=list)
    val_recon_multi_box: List[float] = field(default_factory=list)
    val_recon_patch: List[float] = field(default_factory=list)
    jepa: List[float] = field(default_factory=list)
    sigreg: List[float] = field(default_factory=list)
    rollout: List[float] = field(default_factory=list)
    consistency: List[float] = field(default_factory=list)
    ema_consistency: List[float] = field(default_factory=list)
    recon: List[float] = field(default_factory=list)
    recon_multi_gauss: List[float] = field(default_factory=list)
    recon_multi_box: List[float] = field(default_factory=list)
    recon_patch: List[float] = field(default_factory=list)
    action: List[float] = field(default_factory=list)
    delta: List[float] = field(default_factory=list)
    delta_pred_norm: List[float] = field(default_factory=list)
    delta_target_norm: List[float] = field(default_factory=list)
    grad_world: List[float] = field(default_factory=list)
    grad_decoder: List[float] = field(default_factory=list)

    def append(self, step: float, elapsed: float, cumulative_flops: float, metrics: Dict[str, float]) -> None:
        self.steps.append(step)
        self.elapsed_seconds.append(elapsed)
        self.cumulative_flops.append(cumulative_flops)
        self.world.append(metrics["loss_world"])
        self.val_world.append(metrics.get("loss_val_world", 0.0))
        self.val_recon.append(metrics.get("loss_val_recon", 0.0))
        self.val_recon_multi_gauss.append(metrics.get("loss_val_recon_multi_gauss", 0.0))
        self.val_recon_multi_box.append(metrics.get("loss_val_recon_multi_box", 0.0))
        self.val_recon_patch.append(metrics.get("loss_val_recon_patch", 0.0))
        self.jepa.append(metrics["loss_jepa"])
        self.sigreg.append(metrics["loss_sigreg"])
        self.rollout.append(metrics["loss_rollout"])
        self.consistency.append(metrics["loss_consistency"])
        self.ema_consistency.append(metrics["loss_ema_consistency"])
        self.recon.append(metrics["loss_recon"])
        self.recon_multi_gauss.append(metrics["loss_recon_multi_gauss"])
        self.recon_multi_box.append(metrics["loss_recon_multi_box"])
        self.recon_patch.append(metrics["loss_recon_patch"])
        self.action.append(metrics["loss_action"])
        self.delta.append(metrics["loss_delta"])
        self.delta_pred_norm.append(metrics["delta_pred_norm"])
        self.delta_target_norm.append(metrics["delta_target_norm"])
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
            history.elapsed_seconds,
            history.cumulative_flops,
            history.world,
            history.val_world,
            history.val_recon,
            history.val_recon_multi_gauss,
            history.val_recon_multi_box,
            history.val_recon_patch,
            history.jepa,
            history.sigreg,
            history.rollout,
            history.consistency,
            history.ema_consistency,
            history.recon,
            history.recon_multi_gauss,
            history.recon_multi_box,
            history.recon_patch,
            history.action,
            history.delta,
            history.delta_pred_norm,
            history.delta_target_norm,
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
    if any(val != 0.0 for val in history.val_world):
        plt.plot(history.steps, history.val_world, label="val_world", color=_color(11))
    if any(val != 0.0 for val in history.val_recon):
        plt.plot(history.steps, history.val_recon, label="val_recon", color=_color(12))
    if any(val != 0.0 for val in history.val_recon_multi_gauss):
        plt.plot(history.steps, history.val_recon_multi_gauss, label="val_recon_multi_gauss", color=_color(13))
    if any(val != 0.0 for val in history.val_recon_multi_box):
        plt.plot(history.steps, history.val_recon_multi_box, label="val_recon_multi_box", color=_color(14))
    if any(val != 0.0 for val in history.val_recon_patch):
        plt.plot(history.steps, history.val_recon_patch, label="val_recon_patch", color=_color(15))
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
    if any(val != 0.0 for val in history.recon_patch):
        plt.plot(history.steps, history.recon_patch, label="recon_patch", color=_color(8))
    if any(val != 0.0 for val in history.recon_multi_gauss):
        plt.plot(history.steps, history.recon_multi_gauss, label="recon_multi_gauss", color=_color(9))
    if any(val != 0.0 for val in history.recon_multi_box):
        plt.plot(history.steps, history.recon_multi_box, label="recon_multi_box", color=_color(10))
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
    def __init__(self, capacity: int, sample_decay: float = 0.9, rng: random.Random = None) -> None:
        self.capacity = max(0, capacity)
        self.sample_decay = sample_decay
        self._samples: Dict[int, HardSampleRecord] = {}
        if rng is None:
            raise ValueError("HardSampleReservoir requires an explicit RNG; got None.")
        self.rng = rng

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
        chosen = self.rng.choices(population=population, weights=weights, k=count)
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


def _infer_raw_frame_shape(dataset: TrajectorySequenceDataset) -> Tuple[int, int, int]:
    if not dataset.samples:
        raise ValueError("Cannot infer frame shape without dataset samples.")
    frame_paths, _, start = dataset.samples[0]
    if not frame_paths:
        raise ValueError("Dataset sample contains no frame paths for shape inference.")
    index = min(max(start, 0), len(frame_paths) - 1)
    path = frame_paths[index]
    with Image.open(path) as img:
        width, height = img.size
        channels = len(img.getbands())
    return height, width, channels


def _format_hwc(height: int, width: int, channels: int) -> str:
    return f"{height}×{width}×{channels}"


def _format_param_count(count: int) -> str:
    if count < 0:
        raise ValueError("Parameter count cannot be negative.")
    if count < 1_000:
        return str(count)
    for divisor, suffix in (
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "k"),
    ):
        if count >= divisor:
            value = count / divisor
            if value >= 100:
                formatted = f"{value:.0f}"
            elif value >= 10:
                formatted = f"{value:.1f}".rstrip("0").rstrip(".")
            else:
                formatted = f"{value:.2f}".rstrip("0").rstrip(".")
            return f"{formatted}{suffix}"
    return str(count)


def _count_parameters(modules: Iterable[nn.Module]) -> int:
    total = 0
    for module in modules:
        total += sum(p.numel() for p in module.parameters())
    return total


def _conv2d_flops(in_ch: int, out_ch: int, kernel_size: int, h: int, w: int, stride: int = 1) -> Tuple[int, int, int]:
    """Calculate FLOPs for Conv2d (multiply-adds counted as 2 ops). Returns (flops, out_h, out_w)."""
    padding = (kernel_size - 1) // 2
    out_h = (h + 2 * padding - kernel_size) // stride + 1
    out_w = (w + 2 * padding - kernel_size) // stride + 1
    flops_per_pixel = kernel_size * kernel_size * in_ch * 2  # *2 for multiply-add
    total_flops = flops_per_pixel * out_ch * out_h * out_w
    return total_flops, out_h, out_w


def _conv_transpose2d_flops(in_ch: int, out_ch: int, kernel_size: int, h: int, w: int, stride: int = 2) -> Tuple[int, int, int]:
    """Calculate FLOPs for ConvTranspose2d. Returns (flops, out_h, out_w)."""
    out_h = (h - 1) * stride + kernel_size
    out_w = (w - 1) * stride + kernel_size
    flops_per_pixel = kernel_size * kernel_size * in_ch * 2
    total_flops = flops_per_pixel * out_ch * out_h * out_w
    return total_flops, out_h, out_w


def _linear_flops(in_features: int, out_features: int) -> int:
    """Calculate FLOPs for Linear layer."""
    return in_features * out_features * 2  # multiply-add


def calculate_flops_per_step(cfg: ModelConfig, batch_size: int, seq_len: int) -> int:
    """Calculate estimated FLOPs per training step (forward + backward).

    Returns total FLOPs including forward pass and backward pass (estimated as 2x forward).
    """
    h, w = cfg.image_size, cfg.image_size

    # --- Encoder FLOPs (per frame) ---
    encoder_flops = 0
    curr_h, curr_w = h, w
    in_ch = cfg.in_channels

    for i, out_ch in enumerate(cfg.encoder_schedule):
        # First conv (stride 2) - first layer has CoordConv (+2 channels)
        actual_in_ch = in_ch + 2 if i == 0 else in_ch
        flops1, curr_h, curr_w = _conv2d_flops(actual_in_ch, out_ch, 3, curr_h, curr_w, stride=2)
        # Second conv (stride 1)
        flops2, _, _ = _conv2d_flops(out_ch, out_ch, 3, curr_h, curr_w, stride=1)
        encoder_flops += flops1 + flops2
        in_ch = out_ch

    encoder_total = encoder_flops * batch_size * seq_len

    # --- Predictor FLOPs (per prediction) ---
    predictor_flops = 0
    emb_dim = cfg.embedding_dim
    hidden_dim = cfg.hidden_dim
    action_dim = cfg.action_dim

    # in_proj: emb_dim -> hidden_dim
    predictor_flops += _linear_flops(emb_dim, hidden_dim)
    # action_embed: action_dim -> hidden_dim (2 layers)
    predictor_flops += _linear_flops(action_dim, hidden_dim)
    predictor_flops += _linear_flops(hidden_dim, hidden_dim)
    # FiLM layers (applied twice, each has gamma/beta projections)
    for _ in range(cfg.predictor_film_layers * 2):
        predictor_flops += _linear_flops(hidden_dim, hidden_dim) * 2  # gamma + beta
    # hidden_proj
    predictor_flops += _linear_flops(hidden_dim, hidden_dim)
    # out_proj
    predictor_flops += _linear_flops(hidden_dim, emb_dim)
    # action_head
    predictor_flops += _linear_flops(hidden_dim, action_dim)

    num_predictions = batch_size * (seq_len - 1)
    predictor_total = predictor_flops * num_predictions

    # --- Decoder FLOPs (per frame) ---
    decoder_schedule = cfg.decoder_schedule if cfg.decoder_schedule is not None else cfg.encoder_schedule
    num_layers = len(decoder_schedule)
    start_hw = cfg.image_size // (2 ** num_layers)
    start_ch = decoder_schedule[-1]

    decoder_flops = 0
    # Linear projection
    decoder_flops += _linear_flops(emb_dim, start_ch * start_hw * start_hw)

    curr_h, curr_w = start_hw, start_hw
    in_ch = start_ch

    for out_ch in reversed(decoder_schedule):
        # Upsample (ConvTranspose2d kernel=2, stride=2)
        flops1, curr_h, curr_w = _conv_transpose2d_flops(in_ch, out_ch, 2, curr_h, curr_w, stride=2)
        # Conv refinement
        flops2, _, _ = _conv2d_flops(out_ch, out_ch, 3, curr_h, curr_w, stride=1)
        decoder_flops += flops1 + flops2
        in_ch = out_ch

    # Head convs
    head_hidden = max(in_ch // 2, 1)
    flops_head1, _, _ = _conv2d_flops(in_ch, head_hidden, 3, curr_h, curr_w)
    flops_head2, _, _ = _conv2d_flops(head_hidden, cfg.in_channels, 1, curr_h, curr_w)
    decoder_flops += flops_head1 + flops_head2

    decoder_total = decoder_flops * batch_size * seq_len

    # --- Total ---
    forward_total = encoder_total + predictor_total + decoder_total
    backward_total = forward_total * 2  # Backward is roughly 2x forward
    total_per_step = forward_total + backward_total

    return total_per_step


def _format_flops(flops: int) -> str:
    """Format FLOPs count in human-readable form."""
    if flops < 0:
        raise ValueError("FLOPs count cannot be negative.")
    if flops >= 1_000_000_000_000:
        value = flops / 1_000_000_000_000
        suffix = "TFLOPs"
    elif flops >= 1_000_000_000:
        value = flops / 1_000_000_000
        suffix = "GFLOPs"
    elif flops >= 1_000_000:
        value = flops / 1_000_000
        suffix = "MFLOPs"
    elif flops >= 1_000:
        value = flops / 1_000
        suffix = "KFLOPs"
    else:
        return f"{flops} FLOPs"

    if value >= 100:
        formatted = f"{value:.0f}"
    elif value >= 10:
        formatted = f"{value:.1f}".rstrip("0").rstrip(".")
    else:
        formatted = f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{formatted} {suffix}"


def format_shape_summary(
    raw_shape: Tuple[int, int, int],
    encoder_info: Dict[str, Any],
    predictor_info: Dict[str, Any],
    decoder_info: Dict[str, Any],
    cfg: ModelConfig,
    total_param_text: Optional[str] = None,
    flops_per_step_text: Optional[str] = None,
) -> str:
    lines: List[str] = []
    lines.append("Model Shape Summary (H×W×C)")
    lines.append(f"Raw frame {_format_hwc(*raw_shape)}")
    lines.append(f"  └─ Data loader resize → {_format_hwc(*encoder_info['input'])}")
    lines.append("")
    lines.append(f"Encoder schedule: {cfg.encoder_schedule}")
    lines.append("Encoder:")
    for stage in encoder_info["stages"]:
        lines.append(
            f"  • Stage {stage['stage']}: {_format_hwc(*stage['in'])} → {_format_hwc(*stage['out'])}"
        )
    # Show pooling
    latent_dim = encoder_info["latent_dim"]
    lines.append(f"  AdaptiveAvgPool → 1×1×{latent_dim} (latent)")
    lines.append("")
    lines.append("Predictor:")
    lines.append(
        f"  latent {predictor_info['latent_dim']} → hidden {predictor_info['hidden_dim']} "
        f"(action_dim={predictor_info['action_dim']}, FiLM layers={predictor_info['film_layers']})"
    )
    lines.append("")
    lines.append("Decoder:")
    lines.append(f"  latent_dim={decoder_info.get('latent_dim', 'N/A')}")
    lines.append(f"  Projection reshape → {_format_hwc(*decoder_info['projection'])}")
    for stage in decoder_info["upsample"]:
        lines.append(
            f"  • UpStage {stage['stage']}: {_format_hwc(*stage['in'])} → {_format_hwc(*stage['out'])}"
        )
    # No more detail_skip in decoder
    pre_resize = decoder_info["pre_resize"]
    target = decoder_info["final_target"]
    if decoder_info["needs_resize"]:
        lines.append(
            f"  Final conv output {_format_hwc(*pre_resize)} → bilinear resize → {_format_hwc(*target)}"
        )
    else:
        lines.append(f"  Final output {_format_hwc(*pre_resize)}")
    if total_param_text or flops_per_step_text:
        lines.append("")
    if total_param_text:
        lines.append(f"Total parameters: {total_param_text}")
    if flops_per_step_text:
        lines.append(f"FLOPs per step: {flops_per_step_text}")
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


def _make_fixed_selection(frames: torch.Tensor, vis_rows: int) -> VisualizationSelection:
    if frames is None:
        raise ValueError("Frames tensor must not be None for visualization selection.")
    batch_size, seq_len = frames.shape[0], frames.shape[1]
    if batch_size == 0:
        raise ValueError("Need at least one sequence to build visualization selection.")
    if seq_len < 2:
        raise ValueError("Visualization selection requires sequences with at least two frames.")
    num_rows = min(vis_rows, batch_size)
    if num_rows <= 0:
        raise ValueError("vis_rows must be positive to build a selection.")
    row_indices = torch.arange(num_rows, dtype=torch.long)
    time_indices = (torch.arange(num_rows, dtype=torch.long) % (seq_len - 1)) + 1
    return VisualizationSelection(row_indices=row_indices, time_indices=time_indices)


def _build_fixed_vis_batch(
    dataloader: DataLoader,
    vis_rows: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, Optional[List[List[str]]]], VisualizationSelection]:
    sample_batch = next(iter(dataloader))
    frames_cpu, actions_cpu = sample_batch[0], sample_batch[1]
    if frames_cpu.shape[0] == 0:
        raise AssertionError("Visualization requires at least one sequence in the dataset.")
    if frames_cpu.shape[1] < 2:
        raise AssertionError("Visualization requires sequences with at least two frames.")
    fixed_paths = sample_batch[2] if len(sample_batch) > 2 else None
    fixed_batch_cpu: Tuple[torch.Tensor, torch.Tensor, Optional[List[List[str]]]] = (
        frames_cpu.clone(),
        actions_cpu.clone(),
        [list(paths) for paths in fixed_paths] if fixed_paths is not None else None,
    )
    return fixed_batch_cpu, _make_fixed_selection(fixed_batch_cpu[0], vis_rows)


def _build_embedding_batch(
    dataset: TrajectorySequenceDataset,
    sample_count: int,
    generator: torch.Generator = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive for embedding projection batches.")
    if generator is None:
        raise ValueError("Embedding batch construction requires an explicit torch.Generator.")
    embed_loader = DataLoader(
        dataset,
        batch_size=min(sample_count, len(dataset)),
        shuffle=True,
        collate_fn=collate_batch,
        generator=generator,
    )
    embed_batch = next(iter(embed_loader))
    return (embed_batch[0].clone(), embed_batch[1].clone())


def _render_visualization_batch(
    model: JEPAWorldModel,
    decoder: VisualizationDecoder,
    batch_cpu: Tuple[torch.Tensor, torch.Tensor, Optional[List[List[str]]]],
    rows: int,
    rollout_steps: int,
    max_columns: Optional[int],
    device: torch.device,
    selection: Optional[VisualizationSelection],
    show_gradients: bool,
    log_deltas: bool,
) -> Tuple[List[VisualizationSequence], str]:
    vis_frames = batch_cpu[0].to(device)
    vis_actions = batch_cpu[1].to(device)
    frame_paths = batch_cpu[2]
    if vis_frames.shape[0] == 0:
        raise ValueError("Visualization batch must include at least one sequence.")
    if vis_frames.shape[1] < 2:
        raise ValueError("Visualization batch must include at least two frames.")
    vis_outputs = model.encode_sequence(vis_frames)
    vis_embeddings = vis_outputs["embeddings"]
    # Decoder no longer uses detail_skip - all spatial info is in the latent
    decoded_frames = decoder(vis_embeddings)
    batch_size = vis_frames.shape[0]
    min_start = 0
    target_window = max(2, rollout_steps + 1)
    if max_columns is not None:
        target_window = max(target_window, max(2, max_columns))
    max_window = min(target_window, vis_frames.shape[1] - min_start)
    if max_window < 2:
        raise ValueError("Visualization window must be at least two frames wide.")
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
            prev_embed = current_embed
            next_embed, _, _ = model.predictor(current_embed, action)
            decoded_next = decoder(next_embed)[0]
            current_frame = decoded_next.clamp(0, 1)
            if show_gradients:
                gradient_maps[step] = _prediction_gradient_heatmap(current_frame, gt_slice[step])
            else:
                gradient_maps[step] = _loss_to_heatmap(gt_slice[step], current_frame)
            rollout_frames[step] = current_frame.detach().cpu()
            if log_deltas and row_offset < 2:
                latent_norm = (next_embed - prev_embed).norm().item()
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
        raise AssertionError("Failed to build any visualization sequences.")
    if debug_lines:
        print("\n".join(debug_lines))
    grad_label = "Gradient Norm" if show_gradients else "Error Heatmap"
    return sequences, grad_label


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------


def run_training(cfg: TrainConfig, model_cfg: ModelConfig, weights: LossWeights, title: Optional[str] = None) -> None:
    # --- Filesystem + metadata setup ---
    device = pick_device(cfg.device)
    seed_value, python_rng = _seed_everything(cfg.seed)
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(seed_value)
    val_dataloader_generator = torch.Generator()
    val_dataloader_generator.manual_seed(python_rng.randint(0, 2**32 - 1))
    embedding_generator = torch.Generator()
    embedding_generator.manual_seed(python_rng.randint(0, 2**32 - 1))
    hard_reservoir_rng = random.Random(python_rng.randint(0, 2**32 - 1))
    hard_reservoir_val_rng = random.Random(python_rng.randint(0, 2**32 - 1))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = cfg.output_dir / timestamp
    metrics_dir = run_dir / "metrics"
    fixed_vis_dir = run_dir / "vis_fixed"
    rolling_vis_dir = run_dir / "vis_rolling"
    embeddings_vis_dir = run_dir / "embeddings"
    samples_hard_dir = run_dir / "samples_hard"
    samples_hard_val_dir = run_dir / "samples_hard_val"
    inputs_vis_dir = run_dir / "vis_inputs"
    pair_vis_dir = run_dir / "vis_pairs"

    print(f"[run] Writing outputs to {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    embeddings_vis_dir.mkdir(parents=True, exist_ok=True)
    samples_hard_dir.mkdir(parents=True, exist_ok=True)
    samples_hard_val_dir.mkdir(parents=True, exist_ok=True)

    debug_vis = cfg.debug_visualization
    if debug_vis.input_vis_every_steps > 0:
        inputs_vis_dir.mkdir(parents=True, exist_ok=True)
    if debug_vis.pair_vis_every_steps > 0:
        pair_vis_dir.mkdir(parents=True, exist_ok=True)

    loss_history = LossHistory()

    write_run_metadata(run_dir, cfg, model_cfg, exclude_fields={"title"})

    # Write experiment title to experiment_metadata.txt only if provided
    if title is not None:
        experiment_metadata_path = run_dir / "experiment_metadata.txt"
        experiment_metadata_path.write_text(tomli_w.dumps({"title": title}))

    # --- Dataset initialization ---
    train_trajs, val_trajs = _split_trajectories(cfg.data_root, cfg.max_trajectories, cfg.val_fraction, cfg.val_split_seed)
    dataset = TrajectorySequenceDataset(
        root=cfg.data_root,
        seq_len=cfg.seq_len,
        image_hw=(model_cfg.image_size, model_cfg.image_size),
        max_traj=None,
        included_trajectories=train_trajs,
    )
    val_dataset: Optional[TrajectorySequenceDataset] = None
    if val_trajs:
        val_dataset = TrajectorySequenceDataset(
            root=cfg.data_root,
            seq_len=cfg.seq_len,
            image_hw=(model_cfg.image_size, model_cfg.image_size),
            max_traj=None,
            included_trajectories=val_trajs,
        )
    if cfg.val_fraction > 0 and (val_dataset is None or len(val_dataset) == 0):
        raise AssertionError(
            "val_fraction > 0 but no validation samples are available; check dataset size, val_fraction, and max_traj."
        )

    dataset_action_dim = getattr(dataset, "action_dim", model_cfg.action_dim)
    if val_dataset is not None:
        val_action_dim = getattr(val_dataset, "action_dim", dataset_action_dim)
        if val_action_dim != dataset_action_dim:
            raise AssertionError(f"Validation action_dim {val_action_dim} does not match train action_dim {dataset_action_dim}")
    assert dataset_action_dim == 8, f"Expected action_dim 8, got {dataset_action_dim}"
    if model_cfg.action_dim != dataset_action_dim:
        model_cfg = replace(model_cfg, action_dim=dataset_action_dim)

    hard_reservoir = (
        HardSampleReservoir(cfg.hard_example.reservoir, rng=hard_reservoir_rng) if cfg.hard_example.reservoir > 0 else None
    )
    hard_reservoir_val = (
        HardSampleReservoir(cfg.hard_example.reservoir, rng=hard_reservoir_val_rng) if cfg.hard_example.reservoir > 0 else None
    )

    if len(dataset) == 0:
        raise AssertionError(f"No training samples available in dataset at {cfg.data_root}")
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        generator=dataloader_generator,
    )
    val_dataloader = (
        DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_batch,
            generator=val_dataloader_generator,
        )
        if val_dataset
        else None
    )

    # --- Model initialization ---
    model = JEPAWorldModel(model_cfg).to(device)

    ema_model: Optional[JEPAWorldModel] = None
    if cfg.loss_weights.ema_consistency > 0 and cfg.ema.momentum >= 0.0:
        ema_model = build_ema_model(model)

    decoder_schedule = model_cfg.decoder_schedule if model_cfg.decoder_schedule is not None else model_cfg.encoder_schedule
    decoder = VisualizationDecoder(
        model.embedding_dim,
        model_cfg.in_channels,
        model_cfg.image_size,
        decoder_schedule,
    ).to(device)

    raw_shape = _infer_raw_frame_shape(dataset)
    total_params = _count_parameters((model, decoder))
    flops_per_step = calculate_flops_per_step(model_cfg, cfg.batch_size, cfg.seq_len)
    summary = format_shape_summary(
        raw_shape,
        model.encoder.shape_info(),
        model.predictor.shape_info(),
        decoder.shape_info(),
        model_cfg,
        total_param_text=_format_param_count(total_params),
        flops_per_step_text=_format_flops(flops_per_step),
    )
    print(summary)

    # Write model_shape.txt
    (run_dir / "model_shape.txt").write_text(summary)

    # Write metadata_model.txt (TOML format)
    model_metadata: Dict[str, Any] = {
        "parameters": {
            "total": total_params,
            "total_formatted": _format_param_count(total_params),
        },
        "flops": {
            "per_step": flops_per_step,
            "per_step_formatted": _format_flops(flops_per_step),
        },
    }
    (run_dir / "metadata_model.txt").write_text(tomli_w.dumps(model_metadata))

    # --- Optimizer initialization ---
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(decoder.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # --- Fixed visualization batch (required later) ---
    fixed_batch_cpu, fixed_selection = _build_fixed_vis_batch(dataloader, cfg.vis.rows)
    rolling_batch_cpu: Tuple[torch.Tensor, torch.Tensor, Optional[List[List[str]]]] = fixed_batch_cpu
    embedding_batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor]]
    if cfg.vis.embedding_projection_samples > 0:
        embedding_batch_cpu = _build_embedding_batch(
            dataset,
            cfg.vis.embedding_projection_samples,
            generator=embedding_generator,
        )
    else:
        embedding_batch_cpu = None

    def _print_timing_summary(step: int, totals: Dict[str, float]) -> None:
        total_time = sum(totals.values())
        if total_time <= 0:
            return
        parts = []
        for key, label in (
            ("train", "train"),
            ("log", "log"),
            ("vis", "vis"),
        ):
            value = totals.get(key, 0.0)
            fraction = (value / total_time) if total_time > 0 else 0.0
            parts.append(f"{label}: {value:.2f}s ({fraction:.1%})")
        print(f"[timing up to step {step}] " + ", ".join(parts))

    timing_totals: Dict[str, float] = {"train": 0.0, "log": 0.0, "vis": 0.0}
    total_samples_processed = 0
    run_start_time = perf_counter()

    # --- Main optimization loop ---
    data_iter = iter(dataloader)
    val_iter = iter(val_dataloader) if val_dataloader is not None else None
    for global_step in range(cfg.steps):
        # Get next batch of inputs.
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Update batch with hard examples.
        inject_hard_examples_into_batch(batch, dataset, hard_reservoir, cfg.hard_example.mix_ratio)

        batch_size = int(batch[0].shape[0]) if hasattr(batch[0], "shape") else cfg.batch_size
        total_samples_processed += batch_size

        # Take a training step.
        train_start = perf_counter()
        metrics, difficulty_info = training_step(
            model, decoder, optimizer, batch, cfg, weights, ema_model, cfg.ema.momentum
        )
        timing_totals["train"] += perf_counter() - train_start

        # Update hard examples.
        if hard_reservoir is not None and difficulty_info is not None:
            hard_reservoir.update(
                difficulty_info.indices, difficulty_info.paths, difficulty_info.scores, difficulty_info.frame_indices
            )

        # Log outputs.
        if cfg.log_every_steps > 0 and global_step % cfg.log_every_steps == 0:
            log_start = perf_counter()
            val_metrics: Optional[Dict[str, float]] = None
            val_difficulty: Optional[BatchDifficultyInfo] = None
            if val_iter is not None:
                try:
                    val_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_dataloader)
                    val_batch = next(val_iter)
                val_metrics, val_difficulty = validation_step(model, decoder, val_batch, cfg, weights)
                if hard_reservoir_val is not None and val_difficulty is not None:
                    hard_reservoir_val.update(
                        val_difficulty.indices,
                        val_difficulty.paths,
                        val_difficulty.scores,
                        val_difficulty.frame_indices,
                    )
            elapsed_seconds = max(log_start - run_start_time, 0.0)
            samples_per_sec: Optional[float]
            if elapsed_seconds > 0:
                samples_per_sec = total_samples_processed / elapsed_seconds
            else:
                samples_per_sec = None
            metrics_for_log = dict(metrics)
            if val_metrics is not None:
                metrics_for_log["loss_val_world"] = val_metrics["loss_world"]
                metrics_for_log["loss_val_recon"] = val_metrics["loss_recon"]
                metrics_for_log["loss_val_recon_multi_gauss"] = val_metrics["loss_recon_multi_gauss"]
                metrics_for_log["loss_val_recon_multi_box"] = val_metrics["loss_recon_multi_box"]
                metrics_for_log["loss_val_recon_patch"] = val_metrics["loss_recon_patch"]
            log_metrics(
                global_step,
                metrics_for_log,
                weights,
                samples_per_sec=samples_per_sec,
                elapsed_seconds=elapsed_seconds,
            )

            cumulative_flops = (global_step + 1) * flops_per_step
            loss_history.append(global_step, elapsed_seconds, cumulative_flops, metrics_for_log)
            write_loss_csv(loss_history, metrics_dir / "loss.csv")

            plot_loss_curves(loss_history, metrics_dir)
            timing_totals["log"] += perf_counter() - log_start
            if cfg.show_timing_breakdown:
                _print_timing_summary(global_step, timing_totals)
            model.train()

        # --- Visualization of raw inputs/pairs ---
        batch_paths = batch[2] if len(batch) > 2 else None
        rolling_batch_cpu = (
            batch[0].clone(),
            batch[1].clone(),
            [list(paths) for paths in batch_paths] if batch_paths is not None else None,
        )

        if (
            debug_vis.input_vis_every_steps > 0
            and global_step % debug_vis.input_vis_every_steps == 0
        ):
            save_input_batch_visualization(
                inputs_vis_dir / f"inputs_{global_step:07d}.png",
                rolling_batch_cpu[0],
                rolling_batch_cpu[1],
                debug_vis.input_vis_rows,
            )
        if (
            debug_vis.pair_vis_every_steps > 0
            and global_step % debug_vis.pair_vis_every_steps == 0
        ):
            save_temporal_pair_visualization(
                pair_vis_dir / f"pairs_{global_step:07d}.png",
                rolling_batch_cpu[0],
                rolling_batch_cpu[1],
                debug_vis.pair_vis_rows,
            )

        # --- Rollout/embedding/hard-sample visualizations ---
        if (
            cfg.vis_every_steps > 0
            and global_step % cfg.vis_every_steps == 0
        ):
            vis_start = perf_counter()
            model.eval()
            with torch.no_grad():
                # Render fixed batch.
                sequences, grad_label = _render_visualization_batch(
                    model=model,
                    decoder=decoder,
                    batch_cpu=fixed_batch_cpu,
                    rows=cfg.vis.rows,
                    rollout_steps=cfg.vis.rollout,
                    max_columns=cfg.vis.columns,
                    device=device,
                    selection=fixed_selection,
                    show_gradients=cfg.vis.gradient_norms,
                    log_deltas=cfg.vis.log_deltas,
                )
                save_rollout_sequence_batch(
                    fixed_vis_dir,
                    sequences,
                    grad_label,
                    global_step,
                )

                # Render rolling batch.
                sequences, grad_label = _render_visualization_batch(
                    model=model,
                    decoder=decoder,
                    batch_cpu=rolling_batch_cpu,
                    rows=cfg.vis.rows,
                    rollout_steps=cfg.vis.rollout,
                    max_columns=cfg.vis.columns,
                    device=device,
                    selection=None,
                    show_gradients=cfg.vis.gradient_norms,
                    log_deltas=cfg.vis.log_deltas,
                )
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
                    hard_samples = hard_reservoir.topk(cfg.hard_example.vis_rows * cfg.hard_example.vis_columns)
                    save_hard_example_grid(
                        samples_hard_dir / f"hard_{global_step:07d}.png",
                        hard_samples,
                        cfg.hard_example.vis_columns,
                        cfg.hard_example.vis_rows,
                        dataset.image_hw,
                    )
                if hard_reservoir_val is not None:
                    hard_samples_val = hard_reservoir_val.topk(cfg.hard_example.vis_rows * cfg.hard_example.vis_columns)
                    save_hard_example_grid(
                        samples_hard_val_dir / f"hard_{global_step:07d}.png",
                        hard_samples_val,
                        cfg.hard_example.vis_columns,
                        cfg.hard_example.vis_rows,
                        dataset.image_hw,
                    )

            # --- Train ---
            model.train()
            timing_totals["vis"] += perf_counter() - vis_start

    # --- Final metrics export ---
    if len(loss_history):
        write_loss_csv(loss_history, metrics_dir / "loss.csv")
        plot_loss_curves(loss_history, metrics_dir)


def main() -> None:
    cfg = tyro.cli(
        TrainConfig,
        config=(tyro.conf.HelptextFromCommentsOff,),
    )
    model_cfg = ModelConfig()
    run_training(cfg, model_cfg, cfg.loss_weights, title=cfg.title)


if __name__ == "__main__":
    main()
