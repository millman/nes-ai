#!/usr/bin/env python3
"""Training loop for the JEPA world model with local/global specialization and SIGReg."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections import defaultdict
import os
from pathlib import Path
from typing import Annotated, Any, Dict, Iterable, List, Optional, Sequence, Tuple

import math
import random
import shutil
import re

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
from matplotlib import colors as mcolors
from datetime import datetime
from time import perf_counter
import warnings

try:
    from pytorch_optimizer import SOAP
except ImportError as _soap_import_error:
    SOAP = None
else:
    _soap_import_error = None

from recon.data import list_trajectories, load_frame_as_tensor, short_traj_state_label
from utils.device_utils import pick_device
from jepa_world_model.conv_encoder_decoder import Encoder as ConvEncoder, VisualizationDecoder as ConvVisualizationDecoder
from jepa_world_model.loss import FocalL1Loss, HardnessWeightedL1Loss, HardnessWeightedMSELoss, HardnessWeightedMedianLoss
from jepa_world_model.loss_adjacency import AdjacencyConfig, adjacency_losses, gaussian_augment
from jepa_world_model.metadata import write_run_metadata, write_git_metadata
from jepa_world_model.vis import (
    describe_action_tensor,
    save_embedding_projection,
    save_input_batch_visualization,
    save_adjacency_input_visualization,
    save_rollout_sequence_batch,
    save_temporal_pair_visualization,
    tensor_to_uint8_image,
)
from jepa_world_model.actions import compress_actions_to_ids, decode_action_id
from jepa_world_model.vis_action_alignment import save_action_alignment_detail_plot
from jepa_world_model.vis_graph_diagnostics import save_graph_diagnostics
from jepa_world_model.loss_multi_scale_hardness import (
    build_feature_pyramid,
    multi_scale_hardness_loss_box,
    multi_scale_hardness_loss_gaussian,
)
from jepa_world_model.vis_self_distance import write_self_distance_outputs
from jepa_world_model.vis_state_embedding import write_state_embedding_outputs
from jepa_world_model.vis_cycle_error import compute_cycle_errors, save_cycle_error_plot
from jepa_world_model.vis_hard_samples import save_hard_example_grid



# ------------------------------------------------------------
# Model components
# ------------------------------------------------------------

Encoder = ConvEncoder
VisualizationDecoder = ConvVisualizationDecoder
LegacyEncoder = ConvEncoder
LegacyVisualizationDecoder = ConvVisualizationDecoder


class StateEmbeddingProjector(nn.Module):
    """Project hidden state to a planning/state embedding."""

    def __init__(self, h_dim: int, s_dim: int, hidden_dim: int, unit_norm: bool) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, s_dim),
        )
        self.h_dim = h_dim
        self.s_dim = s_dim
        self.hidden_dim = hidden_dim
        self.unit_norm = unit_norm

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        original_shape = h.shape[:-1]
        h_flat = h.reshape(-1, h.shape[-1])
        s = self.net(h_flat)
        if self.unit_norm:
            s = F.normalize(s, dim=-1)
        return s.view(*original_shape, s.shape[-1])


class PredictorNetwork(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        action_dim: int,
        film_layers: int,
        state_dim: int,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embedding_dim + action_dim + state_dim, hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, embedding_dim)
        self.h_out = nn.Linear(hidden_dim, state_dim)
        self.activation = nn.SiLU(inplace=True)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.delta_scale = 1.0
        self.use_delta_squash = False

    def forward(
        self, embeddings: torch.Tensor, hidden_state: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not (embeddings.shape[:-1] == actions.shape[:-1] == hidden_state.shape[:-1]):
            raise ValueError("Embeddings, hidden state, and actions must share leading dimensions for predictor conditioning.")
        original_shape = embeddings.shape[:-1]
        emb_flat = embeddings.reshape(-1, embeddings.shape[-1])
        act_flat = actions.reshape(-1, actions.shape[-1])
        h_flat = hidden_state.reshape(-1, hidden_state.shape[-1])
        hidden = self.activation(self.in_proj(torch.cat([emb_flat, h_flat, act_flat], dim=-1)))
        hidden = self.activation(self.hidden_proj(hidden))
        raw_delta = self.out_proj(hidden)
        h_next = self.h_out(hidden)
        if self.use_delta_squash:
            delta = torch.tanh(raw_delta) * self.delta_scale
        else:
            delta = raw_delta * self.delta_scale
        pred = emb_flat + delta
        h_next = h_next.view(*original_shape, h_next.shape[-1])
        return (
            pred.view(*original_shape, pred.shape[-1]),
            delta.view(*original_shape, delta.shape[-1]),
            h_next,
        )

    def shape_info(self) -> Dict[str, Any]:
        return {
            "module": "Predictor",
            "latent_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "conditioning": "concat(z,h,action)",
        }


class HiddenToZProjector(nn.Module):
    """Project hidden state to an image-anchored embedding prediction."""

    def __init__(self, h_dim: int, z_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, z_dim),
        )
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        original_shape = h.shape[:-1]
        h_flat = h.reshape(-1, h.shape[-1])
        z_hat = self.net(h_flat)
        return z_hat.view(*original_shape, z_hat.shape[-1])


class ActionDeltaHead(nn.Module):
    """Predict controller actions from latent deltas to avoid action-conditioning leakage."""

    def __init__(self, latent_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        original_shape = delta.shape[:-1]
        flat = delta.reshape(-1, delta.shape[-1])
        logits = self.net(flat)
        return logits.view(*original_shape, logits.shape[-1])


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
    """Dimensional flow:

        image (image_size^2·in_channels)
            └─ Encoder(encoder_schedule → pool → z_dim = encoder_schedule[-1])
            ▼
        z_t ────────────────────────────────┐
                        ├─ Predictor([z_t, h_t, action_t]) → ẑ_{t+1}, ĥ_{t+1}, ŝ_{t+1}
        h_t (state_dim) ─────────────────────┘
            │
            ├→ StateHead(h_t) → s_t (planning embedding)
            └→ Decoder(decoder_schedule → image reconstruction from z)

    Notes:
    • image_size must be divisible by 2**len(encoder_schedule) and 2**len(decoder_schedule) (if provided).
    • encoder_schedule[-1] defines embedding_dim (z_dim); state_dim defines h dimensionality.
    • hidden_dim is the predictor’s internal width; action_dim is the controller space size.
    • decoder_schedule defaults to encoder_schedule if not set.
    """

    in_channels: int = 3
    image_size: int = 64
    hidden_dim: int = 512
    encoder_schedule: Tuple[int, ...] = (32, 64, 128, 256)
    decoder_schedule: Optional[Tuple[int, ...]] = (32, 64, 64, 128)
    action_dim: int = 8
    predictor_film_layers: int = 2
    state_dim: int = 256
    state_embed_dim: Optional[int] = None
    state_embed_unit_norm: bool = False
    state_warmup_frames: int = 8

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
        if self.state_embed_dim is None:
            self.state_embed_dim = self.state_dim


@dataclass
class LossWeights:
    """
    Loss weight semantics (who gets supervised and where gradients flow):
    - loss_jepa (latent transition):
        * What: predictor takes [z_t, h_t, action_t] -> z_hat_{t+1}; MSE vs detached z_{t+1}.
        * Grads: into predictor; into encoder via z_t (and h_t path if attached); target path is stop-grad.
        * Purpose: advance the observable latent consistent with the next encoded frame.
    - loss_h2z: hidden->z projection; z_hat_from_h vs z (detached); shapes hidden path without moving encoder targets.
    Other losses (recon, rollout, etc.) behave as before.
    """
    # Latent transition supervision: ẑ_{t+1} (from predictor) vs detached z_{t+1}; shapes encoder+predictor via z_t.
    jepa: float = 1.0
    sigreg: float = 0.01

    # Image/pixel reconstruction
    recon: float = 0.0
    recon_patch: float = 0.0
    recon_multi_gauss: float = 0.0
    recon_multi_box: float = 0.3
    delta: float = 0.0

    # Action controller input reconstruction
    action_recon: float = 1.0
    # Action prediction from s-deltas (state embedding).
    action_s: float = 0.0

    # Project hidden→z: ẑ_from_h vs z (detached); shapes hidden path without pushing encoder targets.
    h2z: float = 1.0

    rollout: float = 0.0

    consistency: float = 0.0
    ema_consistency: float = 0.0

    # Adjacency: adj0 enforces invariance between clean/noisy state embeddings (s) at same timestep.
    adj0: float = 0.0
    # Adjacency: adj1 aligns predicted next s with next targets via transport.
    adj1: float = 0.0
    # Adjacency: adj2 enforces two-hop composition in s-space toward s_{t+2}.
    adj2: float = 0.0

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
class NormalizeLossesConfig:
    decay: float = 0.99
    epsilon: float = 1e-4


@dataclass
class DiagnosticsConfig:
    enabled: bool = True
    sample_sequences: int = 128
    top_k_components: int = 4
    min_action_count: int = 5
    max_actions_to_plot: int = 12
    cosine_high_threshold: float = 0.7
    synthesize_cycle_samples: bool = False


@dataclass
class GraphDiagnosticsConfig:
    enabled: bool = True
    sample_chunks: int = 32
    chunk_len: int = 12
    k_neighbors: int = 10
    temp: float = 0.1
    eps: float = 1e-8
    long_gap_window: int = 50
    top_m_candidates: int = 0
    block_size: int = 256
    normalize_latents: bool = True
    use_predictor_scores: bool = True
    use_ema_targets: bool = False
    mask_self_edges: bool = True
    include_edge_consistency: bool = True
    edge_consistency_samples: int = 1024


@dataclass
class TrainConfig:
    data_root: Path = Path("data.gridworldkey_wander_to_key")
    output_dir: Path = Path("out.jepa_world_model_trainer")
    log_every_steps: int = 10
    vis_every_steps: int = 50
    checkpoint_every_steps: int = 100
    steps: int = 100_000
    show_timing_breakdown: bool = True
    seed: Optional[int] = 0

    # A validation split only materializes when multiple trajectories exist; with a single traj, keep val_fraction at 0.
    val_fraction: float = 0
    val_split_seed: int = 0

    # Dataset & batching
    max_trajectories: Optional[int] = None
    seq_len: int = 16
    batch_size: int = 8

    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.03
    device: Optional[str] = "mps"
    use_soap: bool = False

    # Loss configuration
    loss_weights: LossWeights = field(default_factory=LossWeights)
    loss_normalization_enabled: bool = False
    normalize_losses: NormalizeLossesConfig = field(default_factory=NormalizeLossesConfig)

    # Specific losses
    adjacency: AdjacencyConfig = field(default_factory=AdjacencyConfig)
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
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    graph_diagnostics: GraphDiagnosticsConfig = field(default_factory=GraphDiagnosticsConfig)

    # CLI-only field (not part of training config, used for experiment metadata)
    title: Annotated[Optional[str], tyro.conf.arg(aliases=["-m"])] = None


@dataclass
class VisualizationSelection:
    row_indices: torch.Tensor
    time_indices: torch.Tensor


@dataclass
class SelfDistanceInputs:
    frames: torch.Tensor  # [1, T, C, H, W] on CPU
    frame_paths: List[Path]  # relative to run directory
    frame_labels: List[str]
    trajectory_label: str
    actions: np.ndarray  # [T, action_dim]
    action_labels: List[str]
    action_dim: int


@dataclass
class VisualizationSequence:
    ground_truth: torch.Tensor
    rollout: List[Optional[torch.Tensor]]
    gradients: List[Optional[np.ndarray]]
    reconstructions: torch.Tensor
    labels: List[str]
    actions: List[str] = field(default_factory=list)


def _assert_adjacency_requirements(cfg: TrainConfig, model_cfg: ModelConfig, weights: LossWeights) -> None:
    warmup_frames = max(getattr(model_cfg, "state_warmup_frames", 0), 0)
    warmup = max(min(warmup_frames, cfg.seq_len - 1), 0)
    remaining = cfg.seq_len - warmup
    if weights.adj0 > 0 and remaining < 2:
        raise AssertionError(
            "adj0 requires at least 2 frames after warmup, "
            f"got seq_len={cfg.seq_len} with state_warmup_frames={warmup_frames} "
            f"(remaining={remaining})."
        )
    if weights.adj1 > 0 and remaining < 2:
        raise AssertionError(
            "adj1 requires at least 2 frames after warmup, "
            f"got seq_len={cfg.seq_len} with state_warmup_frames={warmup_frames} "
            f"(remaining={remaining})."
        )
    if weights.adj2 > 0 and remaining < 3:
        raise AssertionError(
            "adj2 requires at least 3 frames after warmup, "
            f"got seq_len={cfg.seq_len} with state_warmup_frames={warmup_frames} "
            f"(remaining={remaining})."
        )


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
            cfg.action_dim * 2,
            cfg.predictor_film_layers,
            cfg.state_dim,
        )
        self.state_dim = cfg.state_dim
        self.state_head = StateEmbeddingProjector(
            cfg.state_dim,
            cfg.state_embed_dim if cfg.state_embed_dim is not None else cfg.state_dim,
            cfg.hidden_dim,
            cfg.state_embed_unit_norm,
        )
        self.h_to_z = HiddenToZProjector(
            cfg.state_dim,
            self.embedding_dim,
            cfg.hidden_dim,
        )
        self.action_delta_head = ActionDeltaHead(
            self.embedding_dim,
            cfg.hidden_dim,
            cfg.action_dim,
        )
        self.action_delta_head_s = ActionDeltaHead(
            cfg.state_embed_dim if cfg.state_embed_dim is not None else cfg.state_dim,
            cfg.hidden_dim,
            cfg.action_dim,
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


def _pair_actions(actions: torch.Tensor) -> torch.Tensor:
    """Concatenate current and prior actions for predictor conditioning."""
    if actions.ndim != 3:
        raise ValueError("Actions must have shape [B, T, action_dim].")
    if actions.shape[1] == 0:
        return actions.new_zeros((actions.shape[0], 0, actions.shape[2] * 2))
    zeros = actions.new_zeros((actions.shape[0], 1, actions.shape[2]))
    prev = torch.cat([zeros, actions[:, :-1]], dim=1)
    return torch.cat([actions, prev], dim=-1)


def _predictor_rollout(
    model: JEPAWorldModel, embeddings: torch.Tensor, actions: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Roll predictor across sequence to produce z_hat, delta, and h states."""
    b, t, _ = embeddings.shape
    if t < 2:
        zero = embeddings.new_tensor(0.0)
        return (
            zero,
            zero,
            zero,
            embeddings.new_zeros((b, t, model.state_dim)),
        )
    preds = []
    deltas = []
    h_preds = []
    h_states = [embeddings.new_zeros(b, model.state_dim, device=embeddings.device)]
    paired_actions = _pair_actions(actions)
    for step in range(t - 1):
        z_t = embeddings[:, step]
        h_t = h_states[-1]
        act_t = paired_actions[:, step]
        pred, delta, h_next = model.predictor(z_t, h_t, act_t)
        preds.append(pred)
        deltas.append(delta)
        h_preds.append(h_next)
        h_states.append(h_next)
    return (
        torch.stack(preds, dim=1),
        torch.stack(deltas, dim=1),
        torch.stack(h_preds, dim=1),
        torch.stack(h_states, dim=1),
    )


def jepa_loss(
    model: JEPAWorldModel, outputs: Dict[str, torch.Tensor], actions: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """JEPA loss plus action logits, using predictor conditioned on z, h, and action."""
    embeddings = outputs["embeddings"]
    preds, delta_pred, h_preds, h_states = _predictor_rollout(model, embeddings, actions)
    if embeddings.shape[1] < 2:
        zero = embeddings.new_tensor(0.0)
        logits = embeddings.new_zeros((*embeddings.shape[:2], actions.shape[-1]))
        delta = embeddings.new_zeros(embeddings[:, :-1].shape)
        return zero, logits, delta, embeddings.new_zeros(embeddings[:, :-1].shape), h_preds, h_states
    target = embeddings[:, 1:].detach()
    delta_true = embeddings[:, 1:] - embeddings[:, :-1]
    action_logits = model.action_delta_head(delta_true)
    return JEPA_LOSS(preds, target), action_logits, delta_pred, preds, h_preds, h_states


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
    h_states: torch.Tensor,
    rollout_horizon: int,
    warmup_frames: int,
) -> torch.Tensor:
    if rollout_horizon <= 1:
        return embeddings.new_tensor(0.0)
    b, t, d = embeddings.shape
    if t < 2:
        return embeddings.new_tensor(0.0)
    total = embeddings.new_tensor(0.0)
    steps = 0
    warmup = max(min(warmup_frames, t - 1), 0)
    paired_actions = _pair_actions(actions)
    for start in range(warmup, t - 1):
        current = embeddings[:, start]
        max_h = min(rollout_horizon, t - start - 1)
        h_current = h_states[:, start]
        for offset in range(max_h):
            act = paired_actions[:, start + offset]
            pred, _, h_next = model.predictor(current, h_current, act)
            target_step = embeddings[:, start + offset + 1].detach()
            total = total + JEPA_LOSS(pred, target_step)
            steps += 1
            current = pred
            h_current = h_next
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
    preds_scales, targets_scales = build_feature_pyramid(recon_bt, target_bt, num_scales)
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
    preds_scales, targets_scales = build_feature_pyramid(recon_bt, target_bt, num_scales)
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
    loss_norm_ema: Optional[Dict[str, float]] = None,
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
        loss_norm_ema=loss_norm_ema,
        loss_norm_enabled=cfg.loss_normalization_enabled,
        loss_norm_decay=cfg.normalize_losses.decay,
        update_loss_norm=True,
    )
    return metrics, difficulty_info


@torch.no_grad()
def validation_step(
    model: JEPAWorldModel,
    decoder: VisualizationDecoder,
    batch: Tuple[torch.Tensor, torch.Tensor, List[List[str]], torch.Tensor],
    cfg: TrainConfig,
    weights: LossWeights,
    loss_norm_ema: Optional[Dict[str, float]],
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
        loss_norm_ema=loss_norm_ema,
        loss_norm_enabled=cfg.loss_normalization_enabled,
        loss_norm_decay=cfg.normalize_losses.decay,
        update_loss_norm=False,
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
    loss_norm_ema: Optional[Dict[str, float]] = None,
    loss_norm_enabled: bool = True,
    loss_norm_decay: float = 0.99,
    update_loss_norm: bool = True,
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

    loss_jepa_raw, action_logits, delta_pred, z_hat_next, h_preds, h_states = jepa_loss(model, outputs, actions)
    loss_jepa = loss_jepa_raw if weights.jepa > 0 else images.new_tensor(0.0)
    loss_delta_raw, delta_pred_norm, delta_target_norm = delta_prediction_loss(delta_pred, outputs["embeddings"])
    loss_delta = loss_delta_raw if weights.delta > 0 else images.new_tensor(0.0)
    prev_actions = actions[:, :-1]
    if weights.action_recon > 0 and prev_actions.numel() > 0:
        loss_action = ACTION_RECON_LOSS(action_logits, prev_actions)
    else:
        loss_action = images.new_tensor(0.0)
    loss_action_s = images.new_tensor(0.0)
    if weights.action_s > 0 and prev_actions.numel() > 0:
        s_states = model.state_head(h_states)
        s_delta_true = s_states[:, 1:] - s_states[:, :-1]
        action_logits_s = model.action_delta_head_s(s_delta_true)
        loss_action_s = ACTION_RECON_LOSS(action_logits_s, prev_actions)

    with torch.no_grad():
        if prev_actions.numel() > 0:
            action_probs = torch.sigmoid(action_logits)
            action_pred = (action_probs >= 0.5).float()
            action_target = prev_actions.float()
            action_accuracy_bit = (action_pred == action_target).float().mean()
            action_accuracy_all = (action_pred == action_target).all(dim=-1).float().mean()
        else:
            action_accuracy_bit = images.new_tensor(0.0)
            action_accuracy_all = images.new_tensor(0.0)

    warmup_frames = max(getattr(model.cfg, "state_warmup_frames", 0), 0)
    loss_rollout = (
        rollout_loss(
            model,
            outputs["embeddings"],
            actions,
            h_states,
            cfg.rollout.horizon,
            warmup_frames,
        )
        if weights.rollout > 0
        else images.new_tensor(0.0)
    )
    loss_consistency = (
        latent_consistency_loss(outputs["embeddings"])
        if weights.consistency > 0
        else images.new_tensor(0.0)
    )
    combined_latent = torch.cat([outputs["embeddings"], h_states], dim=-1)
    loss_sigreg = (
        sigreg_loss(combined_latent, cfg.sigreg.projections)
        if weights.sigreg > 0
        else images.new_tensor(0.0)
    )
    loss_ema_consistency = images.new_tensor(0.0)
    if ema_model is not None and weights.ema_consistency > 0:
        with torch.no_grad():
            ema_outputs = ema_model.encode_sequence(images)
        loss_ema_consistency = ema_latent_consistency_loss(outputs["embeddings"], ema_outputs["embeddings"])

    # Hidden-state dynamics and cross-projection losses
    loss_h2z = images.new_tensor(0.0)
    need_hidden = weights.h2z > 0
    if need_hidden:
        seq_len = outputs["embeddings"].shape[1]
        if weights.h2z > 0 and h_states.numel() > 0 and seq_len > 0:
            start = max(min(warmup_frames, seq_len - 1), 0)
            if seq_len - start > 0:
                h_stack = h_states[:, start:]
                z_hat_from_h = model.h_to_z(h_stack)
                loss_h2z = F.mse_loss(z_hat_from_h, outputs["embeddings"][:, start:].detach())

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

    loss_adj0, loss_adj1, loss_adj2, adj_entropy, adj_hit, adj2_hit = adjacency_losses(
        model=model,
        embeddings=outputs["embeddings"],
        actions=actions,
        images=images,
        weights=weights,
        cfg=cfg.adjacency,
    )

    def _scaled(name: str, loss_tensor: torch.Tensor) -> torch.Tensor:
        if not loss_norm_enabled:
            return loss_tensor
        if loss_norm_ema is None:
            return loss_tensor
        val = float(loss_tensor.detach().mean().item())
        prev = loss_norm_ema.get(name)
        if prev is None:
            ema_val = val
        else:
            ema_val = loss_norm_decay * prev + (1.0 - loss_norm_decay) * val
        if update_loss_norm:
            loss_norm_ema[name] = ema_val
        denom = max(ema_val, cfg.normalize_losses.epsilon)
        return loss_tensor / denom

    world_loss = (
        weights.jepa * _scaled("loss_jepa", loss_jepa)
        + weights.delta * _scaled("loss_delta", loss_delta)
        + weights.sigreg * _scaled("loss_sigreg", loss_sigreg)
        + weights.rollout * _scaled("loss_rollout", loss_rollout)
        + weights.consistency * _scaled("loss_consistency", loss_consistency)
        + weights.ema_consistency * _scaled("loss_ema_consistency", loss_ema_consistency)
        + weights.action_recon * _scaled("loss_action", loss_action)
        + weights.action_s * _scaled("loss_action_s", loss_action_s)
        + weights.h2z * _scaled("loss_h2z", loss_h2z)
        + weights.recon * _scaled("loss_recon", loss_recon)
        + weights.recon_multi_gauss * _scaled("loss_recon_multi_gauss", loss_recon_multi_gauss)
        + weights.recon_multi_box * _scaled("loss_recon_multi_box", loss_recon_multi_box)
        + weights.recon_patch * _scaled("loss_recon_patch", loss_recon_patch)
        + weights.adj0 * _scaled("loss_adj0", loss_adj0)
        + weights.adj1 * _scaled("loss_adj1", loss_adj1)
        + weights.adj2 * _scaled("loss_adj2", loss_adj2)
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
        "loss_action_s": loss_action_s.item(),
        "loss_h2z": loss_h2z.item(),
        "loss_delta": loss_delta.item(),
        "delta_pred_norm": delta_pred_norm.item(),
        "delta_target_norm": delta_target_norm.item(),
        "loss_world": world_loss.item(),
        "action_accuracy_bit": action_accuracy_bit.item(),
        "action_accuracy_all": action_accuracy_all.item(),
        "loss_adj0": loss_adj0.item(),
        "loss_adj1": loss_adj1.item(),
        "loss_adj2": loss_adj2.item(),
        "adj_entropy": adj_entropy.item(),
        "adj_hit": adj_hit.item(),
        "adj2_hit": adj2_hit.item(),
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


def _load_actions_for_trajectory(traj_dir: Path, expected_length: Optional[int] = None) -> np.ndarray:
    """Load actions.npz for a trajectory and ensure alignment."""
    actions_path = Path(traj_dir) / "actions.npz"
    if not actions_path.is_file():
        raise FileNotFoundError(f"Missing actions.npz for {traj_dir}")
    with np.load(actions_path) as data:
        action_arr = data["actions"] if "actions" in data else data[list(data.files)[0]]
    if action_arr.ndim == 1:
        action_arr = action_arr[:, None]
    action_arr = action_arr.astype(np.float32, copy=False)
    if expected_length is not None and action_arr.shape[0] != expected_length:
        raise ValueError(
            f"Action count {action_arr.shape[0]} does not match frame count {expected_length} in {traj_dir}"
        )
    return action_arr


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
            if len(frame_paths) < self.seq_len:
                warnings.warn(
                    f"Skipping trajectory {traj_name} shorter than seq_len {self.seq_len}",
                    RuntimeWarning,
                )
                continue
            action_arr = _load_actions_for_trajectory(self.root / traj_name, expected_length=len(frame_paths))
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
    if weights.adj0 <= 0:
        filtered.pop("loss_adj0", None)
    if weights.adj1 <= 0:
        filtered.pop("loss_adj1", None)
    if weights.adj2 <= 0:
        filtered.pop("loss_adj2", None)
        filtered.pop("adj2_hit", None)
    if weights.adj1 <= 0 and weights.adj2 <= 0:
        filtered.pop("adj_hit", None)
        filtered.pop("adj_entropy", None)
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
    if weights.action_s <= 0:
        filtered.pop("loss_action_s", None)
    if weights.h2z <= 0:
        filtered.pop("loss_h2z", None)
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
    "loss_action_s",
    "loss_h2z",
    "loss_delta",
    "delta_pred_norm",
    "delta_target_norm",
    "loss_adj0",
    "loss_adj1",
    "loss_adj2",
    "adj_hit",
    "adj2_hit",
    "adj_entropy",
    "action_accuracy_bit",
    "action_accuracy_all",
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
    action_s: List[float] = field(default_factory=list)
    h2z: List[float] = field(default_factory=list)
    delta: List[float] = field(default_factory=list)
    delta_pred_norm: List[float] = field(default_factory=list)
    delta_target_norm: List[float] = field(default_factory=list)
    action_acc_bit: List[float] = field(default_factory=list)
    action_acc_all: List[float] = field(default_factory=list)
    grad_world: List[float] = field(default_factory=list)
    grad_decoder: List[float] = field(default_factory=list)
    adj0: List[float] = field(default_factory=list)
    adj1: List[float] = field(default_factory=list)
    adj2: List[float] = field(default_factory=list)
    adj_hit: List[float] = field(default_factory=list)
    adj2_hit: List[float] = field(default_factory=list)
    adj_entropy: List[float] = field(default_factory=list)

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
        self.action_s.append(metrics["loss_action_s"])
        self.h2z.append(metrics["loss_h2z"])
        self.delta.append(metrics["loss_delta"])
        self.delta_pred_norm.append(metrics["delta_pred_norm"])
        self.delta_target_norm.append(metrics["delta_target_norm"])
        self.adj0.append(metrics.get("loss_adj0", 0.0))
        self.adj1.append(metrics.get("loss_adj1", 0.0))
        self.adj2.append(metrics.get("loss_adj2", 0.0))
        self.adj_hit.append(metrics.get("adj_hit", 0.0))
        self.adj2_hit.append(metrics.get("adj2_hit", 0.0))
        self.adj_entropy.append(metrics.get("adj_entropy", 0.0))
        self.action_acc_bit.append(metrics.get("action_accuracy_bit", 0.0))
        self.action_acc_all.append(metrics.get("action_accuracy_all", 0.0))
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
            history.action_s,
            history.h2z,
            history.delta,
            history.delta_pred_norm,
            history.delta_target_norm,
            history.adj0,
            history.adj1,
            history.adj2,
            history.adj_hit,
            history.adj2_hit,
            history.adj_entropy,
            history.action_acc_bit,
            history.action_acc_all,
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
        "action_s": _color(7),
        "ema_consistency": _color(8),
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
    if any(val != 0.0 for val in history.action_s):
        plt.plot(history.steps, history.action_s, label="action_s", color=color_map["action_s"])
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
    action_dim = cfg.action_dim * 2

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

    num_predictions = batch_size * (seq_len - 1)
    predictor_total = predictor_flops * num_predictions

    # --- Action-from-delta head (per transition) ---
    delta_head_flops = 0
    delta_head_flops += _linear_flops(emb_dim, hidden_dim)
    delta_head_flops += _linear_flops(hidden_dim, hidden_dim)
    delta_head_flops += _linear_flops(hidden_dim, action_dim)
    delta_head_total = delta_head_flops * num_predictions

    # --- Action-from-s-delta head (per transition) ---
    s_dim = cfg.state_embed_dim if cfg.state_embed_dim is not None else cfg.state_dim
    s_delta_head_flops = 0
    s_delta_head_flops += _linear_flops(s_dim, hidden_dim)
    s_delta_head_flops += _linear_flops(hidden_dim, hidden_dim)
    s_delta_head_flops += _linear_flops(hidden_dim, action_dim)
    s_delta_head_total = s_delta_head_flops * num_predictions

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
    forward_total = encoder_total + predictor_total + decoder_total + delta_head_total + s_delta_head_total
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
    conditioning = predictor_info.get("conditioning")
    film_layers = predictor_info.get("film_layers")
    cond_text = (
        f"conditioning={conditioning}"
        if conditioning is not None
        else f"FiLM layers={film_layers}" if film_layers is not None else "conditioning=unknown"
    )
    lines.append(
        f"  latent {predictor_info['latent_dim']} → hidden {predictor_info['hidden_dim']} "
        f"(action_dim={predictor_info['action_dim']}, {cond_text})"
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
) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[List[str]]]]:
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
    paths = [list(p) for p in embed_batch[2]] if len(embed_batch) > 2 else None
    return (embed_batch[0].clone(), embed_batch[1].clone(), paths)


def _build_inverse_action_map(action_dim: int, observed_ids: Iterable[int]) -> Dict[int, int]:
    """Approximate inverse mapping by swapping up/down and left/right bits."""
    weights = (1 << np.arange(action_dim, dtype=np.int64))

    def _invert_bits(action_id: int) -> int:
        bits = np.array([(action_id >> idx) & 1 for idx in range(action_dim)], dtype=np.int64)
        if action_dim > 5:
            bits[4], bits[5] = bits[5], bits[4]
        if action_dim > 7:
            bits[6], bits[7] = bits[7], bits[6]
        return int((bits * weights).sum())

    mapping: Dict[int, int] = {}
    for aid in observed_ids:
        mapping[int(aid)] = _invert_bits(int(aid))
    return mapping


def _compute_pca(delta_z_centered: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if delta_z_centered.shape[0] < 2:
        raise ValueError("Need at least two delta steps to compute PCA.")
    try:
        _, s, vt = np.linalg.svd(delta_z_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        jitter = np.random.normal(scale=1e-6, size=delta_z_centered.shape)
        _, s, vt = np.linalg.svd(delta_z_centered + jitter, full_matrices=False)
    eigvals = (s ** 2) / max(1, delta_z_centered.shape[0] - 1)
    total_var = float(eigvals.sum()) if eigvals.size else 0.0
    if total_var <= 0:
        var_ratio = np.zeros_like(eigvals)
    else:
        var_ratio = eigvals / total_var
    return vt, var_ratio


def _compute_motion_subspace(
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    top_k: int,
    paths: Optional[List[List[str]]] = None,
) -> Optional[Dict[str, Any]]:
    if embeddings.shape[1] < 2:
        return None
    embed_np = embeddings.detach().cpu().numpy()
    action_np = actions.detach().cpu().numpy()
    batch, seq_len, latent_dim = embed_np.shape
    delta_list: List[np.ndarray] = []
    action_vecs: List[np.ndarray] = []
    for b in range(batch):
        delta_list.append(embed_np[b, 1:] - embed_np[b, :-1])
        action_vecs.append(action_np[b, :-1])
    delta_embed = np.concatenate(delta_list, axis=0)
    if delta_embed.shape[0] < 2:
        return None
    actions_flat = np.concatenate(action_vecs, axis=0)
    action_ids = compress_actions_to_ids(actions_flat)
    delta_mean = delta_embed.mean(axis=0, keepdims=True)
    delta_centered = delta_embed - delta_mean
    components, variance_ratio = _compute_pca(delta_centered)
    use_k = max(1, min(top_k, components.shape[0]))
    projection = components[:use_k].T
    flat_embed = embed_np.reshape(-1, latent_dim)
    embed_centered = flat_embed - flat_embed.mean(axis=0, keepdims=True)
    delta_proj = delta_centered @ projection
    proj_flat = embed_centered @ projection
    proj_sequences: List[np.ndarray] = []
    offset = 0
    for _ in range(batch):
        proj_sequences.append(proj_flat[offset : offset + seq_len])
        offset += seq_len
    return {
        "delta_proj": delta_proj,
        "proj_flat": proj_flat,
        "proj_sequences": proj_sequences,
        "variance_ratio": variance_ratio,
        "components": components,
        "action_ids": action_ids,
        "action_dim": action_np.shape[-1],
        "actions_seq": action_np,
        "paths": paths,
    }


def _save_delta_pca_plot(
    out_path: Path,
    variance_ratio: np.ndarray,
    delta_proj: np.ndarray,
    proj_flat: np.ndarray,
    action_ids: np.ndarray,
    action_dim: int,
    embedding_label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    num_var = min(10, variance_ratio.shape[0])
    axes[0, 0].bar(np.arange(num_var), variance_ratio[:num_var], color="tab:blue")
    axes[0, 0].set_title(f"Delta-{embedding_label} PCA variance ratio")
    axes[0, 0].set_xlabel("component")
    axes[0, 0].set_ylabel("explained variance")

    if delta_proj.shape[1] >= 2:
        unique_actions = sorted({int(a) for a in np.asarray(action_ids).reshape(-1)})
        action_to_index = {aid: idx for idx, aid in enumerate(unique_actions)}
        color_indices = (
            np.array([action_to_index.get(int(a), 0) for a in np.asarray(action_ids).reshape(-1)], dtype=np.float32)
            if unique_actions
            else np.asarray(action_ids, dtype=np.float32)
        )
        palette = plt.get_cmap("tab20").colors
        color_count = max(1, min(len(palette), len(unique_actions) if unique_actions else 1))
        color_list = list(palette[:color_count])
        cmap = mcolors.ListedColormap(color_list)
        bounds = np.arange(color_count + 1) - 0.5
        color_indices_mapped = (
            np.mod(color_indices, color_count) if color_count else color_indices
        )
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        scatter = axes[0, 1].scatter(
            delta_proj[:, 0],
            delta_proj[:, 1],
            c=color_indices_mapped,
            cmap=cmap,
            norm=norm,
            s=8,
            alpha=0.7,
        )
        axes[0, 1].set_xlabel(f"PC1 (delta {embedding_label})")
        axes[0, 1].set_ylabel(f"PC2 (delta {embedding_label})")
        cbar = fig.colorbar(scatter, ax=axes[0, 1], fraction=0.046, pad=0.04, boundaries=bounds)
        ticks = list(range(color_count))
        cbar.set_ticks(ticks)
        tick_labels = [decode_action_id(aid, action_dim) for aid in unique_actions[:color_count]] if unique_actions else ["NOOP"]
        cbar.set_ticklabels(tick_labels)
        cbar.set_label("action")
    else:
        axes[0, 1].plot(delta_proj[:, 0], np.zeros_like(delta_proj[:, 0]), ".", alpha=0.6)
        axes[0, 1].set_xlabel(f"PC1 (delta {embedding_label})")
        axes[0, 1].set_ylabel("density")
    axes[0, 1].set_title(f"Delta-{embedding_label} projections")

    cumulative = np.cumsum(variance_ratio)
    axes[1, 0].plot(np.arange(len(cumulative)), cumulative, marker="o", color="tab:green")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_xlabel("component")
    axes[1, 0].set_ylabel("cumulative variance")
    axes[1, 0].set_title("Cumulative explained variance")

    if proj_flat.shape[1] >= 2:
        t = np.linspace(0, 1, num=proj_flat.shape[0])
        sc2 = axes[1, 1].scatter(proj_flat[:, 0], proj_flat[:, 1], c=t, cmap="viridis", s=6, alpha=0.6)
        axes[1, 1].set_xlabel(f"PC1 ({embedding_label})")
        axes[1, 1].set_ylabel(f"PC2 ({embedding_label})")
        fig.colorbar(sc2, ax=axes[1, 1], fraction=0.046, pad=0.04, label="time (normalized)")
    else:
        axes[1, 1].plot(proj_flat[:, 0], np.zeros_like(proj_flat[:, 0]), ".", alpha=0.6)
        axes[1, 1].set_xlabel(f"PC1 ({embedding_label})")
        axes[1, 1].set_ylabel("density")
    axes[1, 1].set_title(f"Embedding projections ({embedding_label})")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_variance_spectrum_plot(out_path: Path, variance_ratio: np.ndarray, max_bars: int = 32) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    k = min(len(variance_ratio), max_bars)
    if k == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(k)
    ax.bar(x, variance_ratio[:k], color="tab:blue", alpha=0.8, label="variance ratio")
    cumulative = np.cumsum(variance_ratio[:k])
    ax.plot(x, cumulative, color="tab:red", marker="o", linewidth=2, label="cumulative")
    ax.set_xlabel("component")
    ax.set_ylabel("variance ratio")
    ax.set_ylim(0, max(1.05, float(cumulative[-1]) + 0.05))
    ax.set_title("Motion PCA spectrum")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_variance_report(delta_dir: Path, global_step: int, variance_ratio: np.ndarray, embedding_label: str) -> None:
    delta_dir.mkdir(parents=True, exist_ok=True)
    report_path = delta_dir / f"delta_{embedding_label}_pca_report_{global_step:07d}.txt"
    with report_path.open("w") as handle:
        if variance_ratio.size == 0:
            handle.write("No variance ratios available.\n")
            return
        cumulative = np.cumsum(variance_ratio)
        targets = [1, 2, 4, 8, 16, 32]
        handle.write("Explained variance coverage by component count:\n")
        for t in targets:
            if t <= len(cumulative):
                handle.write(f"top_{t:02d}: {cumulative[t-1]:.4f}\n")
        handle.write("\nTop variance ratios:\n")
        top_k = min(10, len(variance_ratio))
        for i in range(top_k):
            handle.write(f"comp_{i:02d}: {variance_ratio[i]:.6f}\n")
        handle.write(f"\nTotal components: {len(variance_ratio)}\n")


def _compute_action_alignment_stats(
    delta_proj: np.ndarray,
    action_ids: np.ndarray,
    cfg: DiagnosticsConfig,
    max_actions: Optional[int] = None,
    include_mean_vectors: bool = False,
    include_norm_stats: bool = False,
) -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    unique_actions, counts = np.unique(action_ids, return_counts=True)
    if unique_actions.size == 0:
        return stats
    order = np.argsort(counts)[::-1]
    for idx in order:
        aid = int(unique_actions[idx])
        mask = action_ids == aid
        delta_a = delta_proj[mask]
        if delta_a.shape[0] < cfg.min_action_count:
            continue
        mean_dir = delta_a.mean(axis=0)
        norm = np.linalg.norm(mean_dir)
        if norm < 1e-8:
            continue
        v_unit = mean_dir / norm
        cosines: List[float] = []
        for vec in delta_a:
            denom = np.linalg.norm(vec)
            if denom < 1e-8:
                continue
            cosines.append(float(np.dot(vec / denom, v_unit)))
        if not cosines:
            continue
        cos_np = np.array(cosines, dtype=np.float32)
        entry: Dict[str, Any] = {
            "action_id": aid,
            "count": len(cos_np),
            "mean": float(cos_np.mean()),
            "median": float(np.median(cos_np)),
            "std": float(cos_np.std()),
            "pct_high": float((cos_np > cfg.cosine_high_threshold).mean()),
            "frac_neg": float((cos_np < 0).mean()),
            "cosines": cos_np,
            "mean_dir_norm": float(norm),
        }
        if include_norm_stats:
            delta_norms = np.linalg.norm(delta_a, axis=1)
            entry.update(
                {
                    "delta_norm_mean": float(delta_norms.mean()),
                    "delta_norm_median": float(np.median(delta_norms)),
                    "delta_norm_p10": float(np.percentile(delta_norms, 10)),
                    "delta_norm_p90": float(np.percentile(delta_norms, 90)),
                    "frac_low_delta_norm": float((delta_norms < 1e-8).mean()),
                }
            )
        if include_mean_vectors:
            entry["mean_dir"] = mean_dir
        stats.append(entry)
        if max_actions is not None and len(stats) >= max_actions:
            break
    return stats


def _build_action_alignment_debug(
    alignment_stats: List[Dict[str, Any]],
    delta_proj: np.ndarray,
    action_ids: np.ndarray,
) -> Dict[str, Any]:
    """Build auxiliary tensors for debugging alignment drift/degeneracy."""
    mean_units: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    for stat in alignment_stats:
        aid = int(stat.get("action_id", -1))
        mean_vec = stat.get("mean_dir")
        norm = stat.get("mean_dir_norm", 0.0) or 0.0
        if mean_vec is not None and norm >= 1e-8:
            mean_units[aid] = np.asarray(mean_vec, dtype=np.float32) / float(norm)
        counts[aid] = int(stat.get("count", 0))

    per_action_norms: Dict[int, np.ndarray] = {}
    per_action_cos: Dict[int, np.ndarray] = {}
    overall_cos_list: List[np.ndarray] = []
    overall_norm_list: List[np.ndarray] = []

    for aid, mean_unit in mean_units.items():
        mask = action_ids == aid
        vecs = delta_proj[mask]
        if vecs.shape[0] == 0:
            continue
        norms = np.linalg.norm(vecs, axis=1)
        valid = norms >= 1e-8
        if not np.any(valid):
            continue
        vec_unit = vecs[valid] / norms[valid, None]
        cos = vec_unit @ mean_unit
        per_action_norms[aid] = norms[valid]
        per_action_cos[aid] = cos
        overall_cos_list.append(cos)
        overall_norm_list.append(norms[valid])

    overall_cos = np.concatenate(overall_cos_list) if overall_cos_list else np.asarray([], dtype=np.float32)
    overall_norms = np.concatenate(overall_norm_list) if overall_norm_list else np.asarray([], dtype=np.float32)

    actions_sorted = sorted(mean_units.keys())
    pairwise = np.full((len(actions_sorted), len(actions_sorted)), np.nan, dtype=np.float32)
    for i, ai in enumerate(actions_sorted):
        for j, aj in enumerate(actions_sorted):
            a_vec = mean_units.get(ai)
            b_vec = mean_units.get(aj)
            if a_vec is None or b_vec is None:
                continue
            denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
            if denom < 1e-8:
                continue
            pairwise[i, j] = float(np.dot(a_vec, b_vec) / denom)

    return {
        "actions_sorted": actions_sorted,
        "counts": counts,
        "overall_cos": overall_cos,
        "overall_norms": overall_norms,
        "per_action_norms": per_action_norms,
        "per_action_cos": per_action_cos,
        "pairwise": pairwise,
    }


def _write_action_alignment_report(
    alignment_dir: Path,
    global_step: int,
    stats: List[Dict[str, Any]],
    action_dim: int,
    inverse_map: Dict[int, int],
) -> None:
    alignment_dir.mkdir(parents=True, exist_ok=True)
    report_path = alignment_dir / f"action_alignment_report_{global_step:07d}.txt"
    with report_path.open("w") as handle:
        handle.write("Action alignment diagnostics (per action)\n")
        if not stats:
            handle.write("No actions met alignment criteria.\n")
            return
        handle.write(
            "action_id\tlabel\tcount\tmean\tmedian\tstd\tfrac_neg\tpct_gt_thr\tv_norm"
            "\tdelta_norm_median\tdelta_norm_p90\tfrac_low_norm\tinverse_alignment\tnotes\n"
        )
        mean_vecs: Dict[int, np.ndarray] = {}
        for stat in stats:
            if "mean_dir" in stat:
                mean_vecs[int(stat["action_id"])] = stat["mean_dir"]
        for stat in stats:
            aid = int(stat["action_id"])
            label = decode_action_id(aid, action_dim)
            inv_align = ""
            inv_id = inverse_map.get(aid)
            if inv_id is not None:
                inv_vec = mean_vecs.get(inv_id)
                this_vec = mean_vecs.get(aid)
                if inv_vec is not None and this_vec is not None:
                    inv_norm = float(np.linalg.norm(inv_vec))
                    this_norm = float(np.linalg.norm(this_vec))
                    if inv_norm > 1e-8 and this_norm > 1e-8:
                        inv_align = float(np.dot(this_vec, -inv_vec) / (inv_norm * this_norm))
            note = ""
            if stat.get("mean", 0.0) < 0:
                if stat.get("mean_dir_norm", 0.0) < 1e-6 or stat.get("delta_norm_p90", 0.0) < 1e-6:
                    note = "degenerate mean/blocked"
                elif stat.get("frac_neg", 0.0) > 0.4:
                    note = "bimodal/aliasing suspected"
                else:
                    note = "check action mapping/PCA"
            elif stat.get("mean_dir_norm", 0.0) < 1e-6:
                note = "mean direction near zero"
            handle.write(
                f"{aid}\t{label}\t{stat.get('count', 0)}\t{stat.get('mean', float('nan')):.4f}"
                f"\t{stat.get('median', float('nan')):.4f}\t{stat.get('std', float('nan')):.4f}"
                f"\t{stat.get('frac_neg', float('nan')):.3f}\t{stat.get('pct_high', float('nan')):.3f}"
                f"\t{stat.get('mean_dir_norm', float('nan')):.4f}"
                f"\t{stat.get('delta_norm_median', float('nan')):.4f}\t{stat.get('delta_norm_p90', float('nan')):.4f}"
                f"\t{stat.get('frac_low_delta_norm', float('nan')):.3f}\t{inv_align}\t{note}\n"
            )


def _write_action_alignment_strength(
    alignment_dir: Path,
    global_step: int,
    stats: List[Dict[str, Any]],
    action_dim: int,
) -> None:
    """Summarize per-action directional strength relative to step magnitude."""
    path = alignment_dir / f"action_alignment_strength_{global_step:07d}.txt"
    with path.open("w") as handle:
        if not stats:
            handle.write("No actions met alignment criteria.\n")
            return
        handle.write(
            "Per-action directional strength (mean_dir_norm / delta_norm_median)\n"
            "Lower ratios imply the average direction is weak relative to per-step magnitude (possible aliasing/sign flips).\n\n"
        )
        handle.write(
            "action_id\tlabel\tcount\tmean_cos\tstd\tfrac_neg\tmean_dir_norm\tdelta_norm_median\tstrength_ratio\tnote\n"
        )
        for stat in stats:
            delta_med = float(stat.get("delta_norm_median", float("nan")))
            mean_norm = float(stat.get("mean_dir_norm", float("nan")))
            strength = float("nan")
            if np.isfinite(delta_med) and delta_med > 0 and np.isfinite(mean_norm):
                strength = mean_norm / delta_med
            note = ""
            if not np.isfinite(strength) or strength < 0.05:
                note = "weak mean vs magnitude"
            elif strength < 0.15:
                note = "moderate mean vs magnitude"
            handle.write(
                f"{stat.get('action_id')}\t{decode_action_id(stat.get('action_id', -1), action_dim)}"
                f"\t{stat.get('count', 0)}\t{stat.get('mean', float('nan')):.4f}"
                f"\t{stat.get('std', float('nan')):.4f}\t{stat.get('frac_neg', float('nan')):.3f}"
                f"\t{mean_norm:.4f}\t{delta_med:.4f}\t{strength:.4f}\t{note}\n"
            )


def _write_action_alignment_crosscheck(
    alignment_dir: Path,
    global_step: int,
    stats: List[Dict[str, Any]],
    action_dim: int,
    action_ids: np.ndarray,
    delta_proj: np.ndarray,
) -> None:
    alignment_dir.mkdir(parents=True, exist_ok=True)
    path = alignment_dir / f"action_alignment_crosscheck_{global_step:07d}.txt"
    mean_units: Dict[int, np.ndarray] = {}
    for stat in stats:
        mean_vec = stat.get("mean_dir")
        norm = stat.get("mean_dir_norm", 0.0)
        aid = int(stat["action_id"])
        if mean_vec is None or norm is None or norm < 1e-8:
            continue
        mean_units[aid] = mean_vec / norm
    if not mean_units:
        with path.open("w") as handle:
            handle.write("No usable mean directions for crosscheck.\n")
        return
    with path.open("w") as handle:
        handle.write(
            "Cross-check: sample cosines against own vs other mean directions\n"
            "action_id\tlabel\tcount_valid\tself_mean\tbest_other_id\tbest_other_label"
            "\tbest_other_mean\tgap_self_minus_best_other\tnote\n"
        )
        for aid, mean_unit in mean_units.items():
            mask = action_ids == aid
            vecs = delta_proj[mask]
            if vecs.shape[0] == 0:
                continue
            norms = np.linalg.norm(vecs, axis=1)
            valid_mask = norms >= 1e-8
            if not np.any(valid_mask):
                continue
            vecs_unit = vecs[valid_mask] / norms[valid_mask, None]
            self_mean = float(np.dot(vecs_unit, mean_unit).mean())
            best_other_id: Optional[int] = None
            best_other_mean = -float("inf")
            for bid, other_unit in mean_units.items():
                if bid == aid:
                    continue
                other_mean = float(np.dot(vecs_unit, other_unit).mean())
                if other_mean > best_other_mean:
                    best_other_mean = other_mean
                    best_other_id = bid
            gap = self_mean - best_other_mean if best_other_id is not None else float("nan")
            note = ""
            if best_other_id is not None and gap < 0.05:
                note = "samples align similarly to another action"
            handle.write(
                f"{aid}\t{decode_action_id(aid, action_dim)}\t{vecs_unit.shape[0]}"
                f"\t{self_mean:.4f}\t{best_other_id}"
                f"\t{decode_action_id(best_other_id, action_dim) if best_other_id is not None else ''}"
                f"\t{best_other_mean:.4f}\t{gap:.4f}\t{note}\n"
            )


def _write_diagnostics_csvs(
    delta_dir: Path,
    alignment_dir: Path,
    cycle_dir: Path,
    global_step: int,
    motion: Dict[str, Any],
    cfg: DiagnosticsConfig,
    alignment_stats: List[Dict[str, Any]],
    alignment_debug: Optional[Dict[str, Any]],
    cycle_errors: List[Tuple[int, float]],
    cycle_per_action: Dict[int, List[float]],
    embedding_label: str,
) -> None:
    delta_dir.mkdir(parents=True, exist_ok=True)
    alignment_dir.mkdir(parents=True, exist_ok=True)
    cycle_dir.mkdir(parents=True, exist_ok=True)

    delta_var_csv = delta_dir / f"delta_{embedding_label}_pca_variance_{global_step:07d}.csv"
    with delta_var_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["component", "variance_ratio"])
        for idx, val in enumerate(motion["variance_ratio"][:64]):  # cap rows
            writer.writerow([idx, float(val)])

    delta_samples_csv = delta_dir / f"delta_{embedding_label}_pca_samples_{global_step:07d}.csv"
    with delta_samples_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "frame_index", "frame_path"])
        paths = motion.get("paths") or []
        if paths:
            for sample_idx, frame_list in enumerate(paths):
                if not frame_list:
                    continue
                writer.writerow([sample_idx, 0, frame_list[0]])

    align_csv = alignment_dir / f"action_alignment_{global_step:07d}.csv"
    with align_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["action_id", "action_label", "count", "mean_cos", "std_cos", "pct_high"])
        for stat in alignment_stats:
            writer.writerow(
                [
                    stat["action_id"],
                    decode_action_id(stat["action_id"], motion["action_dim"]),
                    stat["count"],
                    stat["mean"],
                    stat["std"],
                    stat["pct_high"],
                ]
            )

    align_full_csv = alignment_dir / f"action_alignment_full_{global_step:07d}.csv"
    with align_full_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "action_id",
                "action_label",
                "count",
                "mean_cos",
                "median_cos",
                "std_cos",
                "pct_high",
                "frac_negative",
                "mean_dir_norm",
                "delta_norm_mean",
                "delta_norm_median",
                "delta_norm_p10",
                "delta_norm_p90",
                "frac_low_delta_norm",
            ]
        )
        for stat in alignment_stats:
            writer.writerow(
                [
                    stat.get("action_id"),
                    decode_action_id(stat.get("action_id", -1), motion["action_dim"]),
                    stat.get("count", 0),
                    stat.get("mean", float("nan")),
                    stat.get("median", float("nan")),
                    stat.get("std", float("nan")),
                    stat.get("pct_high", float("nan")),
                    stat.get("frac_neg", float("nan")),
                    stat.get("mean_dir_norm", float("nan")),
                    stat.get("delta_norm_mean", float("nan")),
                    stat.get("delta_norm_median", float("nan")),
                    stat.get("delta_norm_p10", float("nan")),
                    stat.get("delta_norm_p90", float("nan")),
                    stat.get("frac_low_delta_norm", float("nan")),
                ]
            )

    if alignment_debug is not None:
        pairwise_csv = alignment_dir / f"action_alignment_pairwise_{global_step:07d}.csv"
        actions_sorted: List[int] = list(alignment_debug.get("actions_sorted") or [])
        pairwise_raw = alignment_debug.get("pairwise")
        pairwise = np.asarray([] if pairwise_raw is None else pairwise_raw, dtype=np.float32)
        with pairwise_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["action_id_a", "label_a", "action_id_b", "label_b", "cosine"])
            for i, aid in enumerate(actions_sorted):
                for j, bid in enumerate(actions_sorted):
                    if pairwise.size == 0 or i >= pairwise.shape[0] or j >= pairwise.shape[1]:
                        continue
                    writer.writerow(
                        [
                            aid,
                            decode_action_id(aid, motion["action_dim"]),
                            bid,
                            decode_action_id(bid, motion["action_dim"]),
                            float(pairwise[i, j]),
                        ]
                    )

        overview_txt = alignment_dir / f"action_alignment_overview_{global_step:07d}.txt"
        overall_cos_raw = alignment_debug.get("overall_cos")
        overall_norms_raw = alignment_debug.get("overall_norms")
        overall_cos = np.asarray([] if overall_cos_raw is None else overall_cos_raw, dtype=np.float32)
        overall_norms = np.asarray([] if overall_norms_raw is None else overall_norms_raw, dtype=np.float32)
        with overview_txt.open("w") as handle:
            handle.write("Global alignment summary (cosine vs per-action mean)\n")
            if overall_cos.size == 0:
                handle.write("No valid cosine samples.\n")
            else:
                handle.write(f"samples: {overall_cos.size}\n")
                handle.write(f"mean: {float(overall_cos.mean()):.4f}\n")
                handle.write(f"median: {float(np.median(overall_cos)):.4f}\n")
                handle.write(f"std: {float(overall_cos.std()):.4f}\n")
                handle.write(f"pct_gt_thr({cfg.cosine_high_threshold}): {float((overall_cos > cfg.cosine_high_threshold).mean()):.4f}\n")
                handle.write(f"frac_negative: {float((overall_cos < 0).mean()):.4f}\n")
                if overall_norms.size:
                    handle.write("\nDelta norm stats (all actions):\n")
                    handle.write(f"median: {float(np.median(overall_norms)):.6f}\n")
                    handle.write(f"p10: {float(np.percentile(overall_norms, 10)):.6f}\n")
                    handle.write(f"p90: {float(np.percentile(overall_norms, 90)):.6f}\n")

    cycle_values_csv = cycle_dir / f"cycle_error_values_{global_step:07d}.csv"
    with cycle_values_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["action_id", "action_label", "cycle_error"])
        for aid, val in cycle_errors:
            writer.writerow([aid, decode_action_id(aid, motion["action_dim"]), val])

    cycle_summary_csv = cycle_dir / f"cycle_error_summary_{global_step:07d}.csv"
    with cycle_summary_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["action_id", "action_label", "count", "mean_cycle_error"])
        for aid, vals in sorted(cycle_per_action.items(), key=lambda kv: len(kv[1]), reverse=True):
            if not vals:
                continue
            writer.writerow([aid, decode_action_id(aid, motion["action_dim"]), len(vals), float(np.mean(vals))])


def _save_diagnostics_frames(
    frames: torch.Tensor,
    paths: Optional[List[List[str]]],
    actions: Optional[torch.Tensor],
    frames_dir: Path,
    global_step: int,
) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    csv_path = frames_dir / f"frames_{global_step:07d}.csv"
    max_save = frames.shape[0]
    entries: List[Tuple[str, int]] = []

    def _natural_path_key(path_str: str) -> Tuple:
        parts = re.split(r"(\d+)", path_str)
        key: List = []
        for part in parts:
            if not part:
                continue
            key.append(int(part) if part.isdigit() else part.lower())
        return tuple(key)

    for idx in range(max_save):
        src_path = paths[idx][0] if paths and idx < len(paths) and paths[idx] else ""
        entries.append((src_path, idx))
    entries.sort(key=lambda t: _natural_path_key(t[0]))
    new_sources_sorted = [src for src, _ in entries]

    # Try to reuse existing frames if the sources match exactly.
    reuse_image_lookup: Optional[Dict[str, str]] = None
    for existing_csv in sorted(frames_dir.glob("frames_*.csv")):
        try:
            with existing_csv.open("r", newline="") as handle:
                reader = csv.DictReader(handle)
                existing_records = list(reader)
        except (OSError, csv.Error):
            continue
        if not existing_records:
            continue
        existing_sources = [row.get("source_path", "") for row in existing_records]
        existing_sources_sorted = sorted(existing_sources, key=_natural_path_key)
        if len(existing_sources_sorted) != len(new_sources_sorted):
            continue
        if existing_sources_sorted == new_sources_sorted:
            reuse_image_lookup = {}
            for row in existing_records:
                src = row.get("source_path", "")
                img_rel = row.get("image_path", "")
                if src and img_rel:
                    reuse_image_lookup[src] = img_rel
            break

    records: List[Tuple[int, str, str, Optional[int], str]] = []
    if reuse_image_lookup:
        for out_idx, (src, orig_idx) in enumerate(entries):
            img_rel = reuse_image_lookup.get(src, "")
            if not img_rel:
                reuse_image_lookup = None
                records.clear()
                break
            action_id: Optional[int] = None
            action_label = ""
            if actions is not None and actions.ndim >= 2 and orig_idx < actions.shape[0]:
                action_vec = actions[orig_idx, 0].detach().cpu().numpy()
                action_id = int(compress_actions_to_ids(action_vec[None, ...])[0])
                action_label = decode_action_id(action_id, actions.shape[-1])
            records.append((out_idx, img_rel, src, action_id, action_label))

    if not records:
        step_dir = frames_dir / f"frames_{global_step:07d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        for out_idx, (src, orig_idx) in enumerate(entries):
            frame_img = tensor_to_uint8_image(frames[orig_idx, 0])
            out_path = step_dir / f"frame_{out_idx:04d}.png"
            Image.fromarray(frame_img).save(out_path)
            action_id: Optional[int] = None
            action_label = ""
            if actions is not None and actions.ndim >= 2 and orig_idx < actions.shape[0]:
                action_vec = actions[orig_idx, 0].detach().cpu().numpy()
                action_id = int(compress_actions_to_ids(action_vec[None, ...])[0])
                action_label = decode_action_id(action_id, actions.shape[-1])
            records.append(
                (
                    out_idx,
                    out_path.relative_to(step_dir.parent).as_posix(),
                    src,
                    action_id,
                    action_label,
                )
            )

    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "image_path", "source_path", "action_id", "action_label"])
        writer.writerows(records)


def _write_alignment_debug_csv(
    frames: torch.Tensor,
    actions: torch.Tensor,
    paths: Optional[List[List[str]]],
    out_dir: Path,
    global_step: int,
) -> None:
    """Log per-frame checksums and action context to spot indexing issues."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bsz, seq_len = frames.shape[0], frames.shape[1]
    action_ids = compress_actions_to_ids(actions.cpu().numpy().reshape(-1, actions.shape[-1])).reshape(bsz, seq_len)
    csv_path = out_dir / f"alignment_debug_{global_step:07d}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "batch_index",
                "time_index",
                "frame_mean",
                "frame_std",
                "same_as_prev_frame",
                "action_to_this_id",
                "action_to_this_label",
                "action_from_this_id",
                "action_from_this_label",
                "frame_path",
            ]
        )
        for b in range(bsz):
            for t in range(seq_len):
                frame = frames[b, t]
                mean = float(frame.mean().item())
                std = float(frame.std().item())
                same_prev = False
                if t > 0:
                    same_prev = bool(torch.equal(frame, frames[b, t - 1]))
                action_to_this_id = action_ids[b, t - 1] if t > 0 else None
                action_from_this_id = action_ids[b, t] if t < seq_len - 1 else None
                frame_path = paths[b][t] if paths and b < len(paths) and t < len(paths[b]) else ""
                writer.writerow(
                    [
                        b,
                        t,
                        mean,
                        std,
                        int(same_prev),
                        "" if action_to_this_id is None else int(action_to_this_id),
                        "" if action_to_this_id is None else decode_action_id(int(action_to_this_id), actions.shape[-1]),
                        "" if action_from_this_id is None else int(action_from_this_id),
                        "" if action_from_this_id is None else decode_action_id(int(action_from_this_id), actions.shape[-1]),
                        frame_path,
                    ]
                )

def _save_motion_diagnostics_outputs(
    motion: Dict[str, Any],
    cfg: DiagnosticsConfig,
    delta_dir: Path,
    alignment_dir: Path,
    cycle_dir: Path,
    global_step: int,
    embedding_label: str,
    inverse_map: Dict[int, int],
) -> None:
    delta_path = delta_dir / f"delta_{embedding_label}_pca_{global_step:07d}.png"
    _save_delta_pca_plot(
        delta_path,
        motion["variance_ratio"],
        motion["delta_proj"],
        motion["proj_flat"],
        motion["action_ids"],
        motion["action_dim"],
        embedding_label,
    )
    _save_variance_spectrum_plot(
        delta_dir / f"delta_{embedding_label}_variance_spectrum_{global_step:07d}.png",
        motion["variance_ratio"],
    )
    _write_variance_report(delta_dir, global_step, motion["variance_ratio"], embedding_label)
    alignment_stats_full = _compute_action_alignment_stats(
        motion["delta_proj"],
        motion["action_ids"],
        cfg,
        max_actions=None,
        include_mean_vectors=True,
        include_norm_stats=True,
    )
    alignment_debug = _build_action_alignment_debug(alignment_stats_full, motion["delta_proj"], motion["action_ids"])
    save_action_alignment_detail_plot(
        alignment_dir / f"action_alignment_detail_{global_step:07d}.png",
        alignment_debug,
        cfg.cosine_high_threshold,
        motion["action_dim"],
    )
    _write_action_alignment_report(alignment_dir, global_step, alignment_stats_full, motion["action_dim"], inverse_map)
    _write_action_alignment_strength(alignment_dir, global_step, alignment_stats_full, motion["action_dim"])
    _write_action_alignment_crosscheck(
        alignment_dir,
        global_step,
        alignment_stats_full,
        motion["action_dim"],
        motion["action_ids"],
        motion["delta_proj"],
    )
    cycle_path = cycle_dir / f"cycle_error_{global_step:07d}.png"
    errors, per_action = compute_cycle_errors(
        motion["proj_sequences"],
        motion["actions_seq"],
        inverse_map,
        include_synthetic=cfg.synthesize_cycle_samples,
    )
    save_cycle_error_plot(cycle_path, [e[1] for e in errors], per_action, motion["action_dim"])
    _write_diagnostics_csvs(
        delta_dir,
        alignment_dir,
        cycle_dir,
        global_step,
        motion,
        cfg,
        alignment_stats_full,
        alignment_debug,
        errors,
        per_action,
        embedding_label,
    )


def _save_diagnostics_outputs(
    model: JEPAWorldModel,
    frames_cpu: torch.Tensor,
    actions_cpu: torch.Tensor,
    paths: Optional[List[List[str]]],
    device: torch.device,
    cfg: DiagnosticsConfig,
    delta_dir: Path,
    alignment_dir: Path,
    cycle_dir: Path,
    frames_dir: Path,
    global_step: int,
    delta_s_dir: Optional[Path] = None,
    alignment_s_dir: Optional[Path] = None,
    cycle_s_dir: Optional[Path] = None,
) -> None:
    if frames_cpu.shape[0] == 0 or frames_cpu.shape[1] < 2:
        return
    with torch.no_grad():
        frames = frames_cpu.to(device)
        actions = actions_cpu.to(device)
        embeddings = model.encode_sequence(frames)["embeddings"]
    motion = _compute_motion_subspace(embeddings, actions_cpu, cfg.top_k_components, paths)
    if motion is None:
        return
    inverse_map = _build_inverse_action_map(
        motion["action_dim"],
        np.unique(compress_actions_to_ids(motion["actions_seq"].reshape(-1, motion["actions_seq"].shape[-1]))),
    )
    _save_motion_diagnostics_outputs(
        motion,
        cfg,
        delta_dir,
        alignment_dir,
        cycle_dir,
        global_step,
        "z",
        inverse_map,
    )

    if delta_s_dir is not None and alignment_s_dir is not None and cycle_s_dir is not None:
        with torch.no_grad():
            _, _, _, h_states = _predictor_rollout(model, embeddings, actions)
            s_embeddings = model.state_head(h_states)
        motion_s = _compute_motion_subspace(s_embeddings, actions_cpu, cfg.top_k_components, paths)
        if motion_s is not None:
            _save_motion_diagnostics_outputs(
                motion_s,
                cfg,
                delta_s_dir,
                alignment_s_dir,
                cycle_s_dir,
                global_step,
                "s",
                inverse_map,
            )

    _write_alignment_debug_csv(frames_cpu, actions_cpu, paths, frames_dir, global_step)
    _save_diagnostics_frames(frames_cpu, paths, actions_cpu, frames_dir, global_step)


def _prepare_self_distance_inputs(
    data_root: Path,
    train_trajs: List[str],
    image_hw: Tuple[int, int],
    run_dir: Path,
) -> Optional[SelfDistanceInputs]:
    traj_map = list_trajectories(data_root)
    if not traj_map:
        return None
    ordered = train_trajs if train_trajs else list(traj_map.keys())
    chosen: Optional[str] = None
    for name in ordered:
        if name in traj_map and len(traj_map[name]) >= 2:
            chosen = name
            break
    if chosen is None:
        return None
    src_paths = traj_map[chosen]
    frames: List[torch.Tensor] = []
    rel_paths: List[Path] = []
    labels: List[str] = []
    frames_dir = run_dir / "self_distance_frames"
    for src in src_paths:
        dst = frames_dir / chosen / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        frames.append(load_frame_as_tensor(src, size=image_hw))
        rel_paths.append(dst.relative_to(run_dir))
        labels.append(short_traj_state_label(str(src)))
    if not frames:
        return None
    actions = _load_actions_for_trajectory(data_root / chosen, expected_length=len(frames))
    action_dim = actions.shape[1]
    action_ids = compress_actions_to_ids(actions)
    action_labels = [decode_action_id(int(aid), action_dim) for aid in action_ids]
    stacked = torch.stack(frames, dim=0).unsqueeze(0)
    traj_label = f"{data_root.name}/{chosen}" if data_root.name else chosen
    return SelfDistanceInputs(
        frames=stacked,
        frame_paths=rel_paths,
        frame_labels=labels,
        trajectory_label=traj_label,
        actions=actions,
        action_labels=action_labels,
        action_dim=action_dim,
    )


def _save_checkpoint(
    checkpoints_dir: Path,
    model: JEPAWorldModel,
    decoder: VisualizationDecoder,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    tag: str,
) -> None:
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "global_step": global_step,
        "model_state": model.state_dict(),
        "decoder_state": decoder.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    target = checkpoints_dir / f"{tag}.pt"
    torch.save(payload, target)


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
    rng: Optional[torch.Generator] = None,
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
        row_indices = torch.randperm(batch_size, generator=rng, device=device)[:num_rows]
        base_starts = torch.randint(min_start, max_start + 1, (num_rows,), device=device, generator=rng)
    sequences: List[VisualizationSequence] = []
    debug_lines: List[str] = []
    paired_actions = _pair_actions(vis_actions)
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
        current_hidden = current_embed.new_zeros(1, model.state_dim)
        prev_pred_frame = decoded_frames[idx, start_idx].detach()
        current_frame = prev_pred_frame
        for step in range(1, max_window):
            action = paired_actions[idx, start_idx + step - 1].unsqueeze(0)
            prev_embed = current_embed
            next_embed, _, next_hidden = model.predictor(current_embed, current_hidden, action)
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
            current_hidden = next_hidden
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
    diagnostics_generator = torch.Generator()
    diagnostics_generator.manual_seed(python_rng.randint(0, 2**32 - 1))
    hard_reservoir_rng = random.Random(python_rng.randint(0, 2**32 - 1))
    hard_reservoir_val_rng = random.Random(python_rng.randint(0, 2**32 - 1))
    # Dedicated RNGs keep visualization sampling consistent across experiments.
    vis_selection_generator = torch.Generator(device=device)
    vis_selection_generator.manual_seed(python_rng.randint(0, 2**32 - 1))
    training_vis_generator = torch.Generator()
    training_vis_generator.manual_seed(python_rng.randint(0, 2**32 - 1))

    _assert_adjacency_requirements(cfg, model_cfg, weights)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = cfg.output_dir / timestamp
    metrics_dir = run_dir / "metrics"
    fixed_vis_dir = run_dir / "vis_fixed"
    rolling_vis_dir = run_dir / "vis_rolling"
    embeddings_vis_dir = run_dir / "embeddings"
    diagnostics_delta_dir = run_dir / "vis_delta_z_pca"
    diagnostics_delta_s_dir = run_dir / "vis_delta_s_pca"
    diagnostics_alignment_dir = run_dir / "vis_action_alignment_z"
    diagnostics_alignment_s_dir = run_dir / "vis_action_alignment_s"
    diagnostics_cycle_dir = run_dir / "vis_cycle_error_z"
    diagnostics_cycle_s_dir = run_dir / "vis_cycle_error_s"
    diagnostics_frames_dir = run_dir / "vis_diagnostics_frames"
    graph_diagnostics_dir = run_dir / "graph_diagnostics_z"
    graph_diagnostics_s_dir = run_dir / "graph_diagnostics_s"
    adjacency_vis_dir = run_dir / "vis_adjacency"
    samples_hard_dir = run_dir / "samples_hard"
    samples_hard_val_dir = run_dir / "samples_hard_val"
    inputs_vis_dir = run_dir / "vis_inputs"
    pair_vis_dir = run_dir / "vis_pairs"
    vis_self_distance_z_dir = run_dir / "vis_self_distance_z"
    vis_self_distance_s_dir = run_dir / "vis_self_distance_s"
    vis_state_embedding_dir = run_dir / "vis_state_embedding"
    self_distance_z_dir = run_dir / "self_distance_z"
    self_distance_s_dir = run_dir / "self_distance_s"
    checkpoints_dir = run_dir / "checkpoints"

    print(f"[run] Writing outputs to {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    embeddings_vis_dir.mkdir(parents=True, exist_ok=True)
    if cfg.diagnostics.enabled:
        diagnostics_delta_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_delta_s_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_alignment_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_alignment_s_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_cycle_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_cycle_s_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_frames_dir.mkdir(parents=True, exist_ok=True)
    if cfg.graph_diagnostics.enabled:
        graph_diagnostics_dir.mkdir(parents=True, exist_ok=True)
        graph_diagnostics_s_dir.mkdir(parents=True, exist_ok=True)
    adjacency_vis_dir.mkdir(parents=True, exist_ok=True)
    samples_hard_dir.mkdir(parents=True, exist_ok=True)
    samples_hard_val_dir.mkdir(parents=True, exist_ok=True)
    vis_self_distance_z_dir.mkdir(parents=True, exist_ok=True)
    vis_self_distance_s_dir.mkdir(parents=True, exist_ok=True)
    vis_state_embedding_dir.mkdir(parents=True, exist_ok=True)
    self_distance_z_dir.mkdir(parents=True, exist_ok=True)
    self_distance_s_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    debug_vis = cfg.debug_visualization
    if debug_vis.input_vis_every_steps > 0:
        inputs_vis_dir.mkdir(parents=True, exist_ok=True)
    if debug_vis.pair_vis_every_steps > 0:
        pair_vis_dir.mkdir(parents=True, exist_ok=True)

    loss_history = LossHistory()

    write_run_metadata(run_dir, cfg, model_cfg, exclude_fields={"title"})
    write_git_metadata(run_dir)

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

    self_distance_inputs = _prepare_self_distance_inputs(
        cfg.data_root,
        train_trajs,
        (model_cfg.image_size, model_cfg.image_size),
        run_dir,
    )

    dataset_action_dim = getattr(dataset, "action_dim", model_cfg.action_dim)
    if val_dataset is not None:
        val_action_dim = getattr(val_dataset, "action_dim", dataset_action_dim)
        if val_action_dim != dataset_action_dim:
            raise AssertionError(f"Validation action_dim {val_action_dim} does not match train action_dim {dataset_action_dim}")
    assert dataset_action_dim == 8, f"Expected action_dim 8, got {dataset_action_dim}"
    if model_cfg.action_dim != dataset_action_dim:
        model_cfg = replace(model_cfg, action_dim=dataset_action_dim)

    graph_diag_dataset: Optional[TrajectorySequenceDataset] = None
    graph_diag_batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[List[List[str]]]]] = None
    if cfg.graph_diagnostics.enabled:
        graph_seq_len = max(cfg.graph_diagnostics.chunk_len, 3)
        graph_diag_dataset = TrajectorySequenceDataset(
            root=cfg.data_root,
            seq_len=graph_seq_len,
            image_hw=(model_cfg.image_size, model_cfg.image_size),
            max_traj=None,
            included_trajectories=train_trajs,
        )
        graph_action_dim = getattr(graph_diag_dataset, "action_dim", dataset_action_dim)
        if graph_action_dim != dataset_action_dim:
            raise AssertionError(
                f"Graph diagnostics action_dim {graph_action_dim} does not match train action_dim {dataset_action_dim}"
            )
        graph_diag_batch_cpu = _build_embedding_batch(
            graph_diag_dataset,
            cfg.graph_diagnostics.sample_chunks,
            generator=diagnostics_generator,
        )

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
    params = list(model.parameters()) + list(decoder.parameters())
    if cfg.use_soap:
        if SOAP is None:
            raise ImportError(
                "SOAP optimizer requires the pytorch-optimizer package. Install with `pip install pytorch-optimizer`."
            ) from _soap_import_error
        if device.type == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in {"1", "true", "TRUE"}:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            print(
                "[warning] Enabled PYTORCH_ENABLE_MPS_FALLBACK=1 for SOAP on MPS to allow CPU fallback for missing ops."
            )
        optimizer = SOAP(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    # --- Fixed visualization batch (required later) ---
    fixed_batch_cpu, fixed_selection = _build_fixed_vis_batch(dataloader, cfg.vis.rows)
    rolling_batch_cpu: Tuple[torch.Tensor, torch.Tensor, Optional[List[List[str]]]] = fixed_batch_cpu
    embedding_batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[List[List[str]]]]]
    if cfg.vis.embedding_projection_samples > 0:
        embedding_batch_cpu = _build_embedding_batch(
            dataset,
            cfg.vis.embedding_projection_samples,
            generator=embedding_generator,
        )
    else:
        embedding_batch_cpu = None

    diagnostics_batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[List[List[str]]]]] = None
    if cfg.diagnostics.enabled and cfg.diagnostics.sample_sequences > 0:
        diagnostics_batch_cpu = _build_embedding_batch(
            dataset,
            cfg.diagnostics.sample_sequences,
            generator=diagnostics_generator,
        )

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
    loss_norm_ema: Dict[str, float] = {}

    # --- Main optimization loop ---
    data_iter = iter(dataloader)
    val_iter = iter(val_dataloader) if val_dataloader is not None else None
    last_step = -1
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
            model, decoder, optimizer, batch, cfg, weights, ema_model, cfg.ema.momentum, loss_norm_ema
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
                val_metrics, val_difficulty = validation_step(model, decoder, val_batch, cfg, weights, loss_norm_ema)
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
                generator=training_vis_generator,
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
                    rng=vis_selection_generator,
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
                    rng=vis_selection_generator,
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
                if self_distance_inputs is not None:
                    write_self_distance_outputs(
                        model,
                        self_distance_inputs,
                        device,
                        self_distance_z_dir,
                        vis_self_distance_z_dir,
                        global_step,
                        embedding_label="z",
                        title_prefix="Self-distance (Z)",
                        file_prefix="self_distance_z",
                    )
                if self_distance_inputs is not None:
                    hist_frames = rolling_batch_cpu[0] if rolling_batch_cpu is not None else None
                    hist_actions = rolling_batch_cpu[1] if rolling_batch_cpu is not None else None
                    write_state_embedding_outputs(
                        model,
                        self_distance_inputs,
                        device,
                        self_distance_s_dir,
                        vis_self_distance_s_dir,
                        vis_state_embedding_dir,
                        global_step,
                        hist_frames_cpu=hist_frames,
                        hist_actions_cpu=hist_actions,
                    )
                if (weights.adj0 > 0) or (weights.adj1 > 0) or (weights.adj2 > 0):
                    sigma_adj = max(cfg.adjacency.sigma_aug, 0.0)
                    noisy_batch = gaussian_augment(rolling_batch_cpu[0], sigma_adj)
                    save_adjacency_input_visualization(
                        adjacency_vis_dir / f"adjacency_{global_step:07d}.png",
                        rolling_batch_cpu[0],
                        noisy_batch,
                        rows=cfg.vis.rows,
                        max_steps=cfg.vis.columns,
                    )
                if cfg.diagnostics.enabled:
                    diag_frames = diagnostics_batch_cpu[0] if diagnostics_batch_cpu is not None else rolling_batch_cpu[0]
                    diag_actions = diagnostics_batch_cpu[1] if diagnostics_batch_cpu is not None else rolling_batch_cpu[1]
                    diag_paths = diagnostics_batch_cpu[2] if diagnostics_batch_cpu is not None and len(diagnostics_batch_cpu) > 2 else rolling_batch_cpu[2]
                    _save_diagnostics_outputs(
                        model,
                        diag_frames,
                        diag_actions,
                        diag_paths,
                        device,
                        cfg.diagnostics,
                        diagnostics_delta_dir,
                        diagnostics_alignment_dir,
                        diagnostics_cycle_dir,
                        diagnostics_frames_dir,
                        global_step,
                        delta_s_dir=diagnostics_delta_s_dir,
                        alignment_s_dir=diagnostics_alignment_s_dir,
                        cycle_s_dir=diagnostics_cycle_s_dir,
                    )
                if cfg.graph_diagnostics.enabled:
                    graph_frames = graph_diag_batch_cpu[0] if graph_diag_batch_cpu is not None else rolling_batch_cpu[0]
                    graph_actions = graph_diag_batch_cpu[1] if graph_diag_batch_cpu is not None else rolling_batch_cpu[1]
                    save_graph_diagnostics(
                        model,
                        ema_model,
                        graph_frames,
                        graph_actions,
                        device,
                        cfg.graph_diagnostics,
                        graph_diagnostics_dir,
                        global_step,
                        embedding_kind="z",
                        history_csv_path=metrics_dir / "graph_diagnostics_z.csv",
                    )
                    save_graph_diagnostics(
                        model,
                        ema_model,
                        graph_frames,
                        graph_actions,
                        device,
                        cfg.graph_diagnostics,
                        graph_diagnostics_s_dir,
                        global_step,
                        embedding_kind="s",
                        history_csv_path=metrics_dir / "graph_diagnostics_s.csv",
                    )

            # --- Train ---
            model.train()
            timing_totals["vis"] += perf_counter() - vis_start

        if (
            cfg.checkpoint_every_steps > 0
            and global_step % cfg.checkpoint_every_steps == 0
        ):
            _save_checkpoint(
                checkpoints_dir,
                model,
                decoder,
                optimizer,
                global_step,
                tag="last",
            )
        last_step = global_step

    # --- Final metrics export ---
    last_step = global_step if cfg.steps > 0 else last_step
    if self_distance_inputs is not None:
        model.eval()
        with torch.no_grad():
            write_self_distance_outputs(
                model,
                self_distance_inputs,
                device,
                self_distance_z_dir,
                vis_self_distance_z_dir,
                last_step if last_step >= 0 else 0,
                embedding_label="z",
                title_prefix="Self-distance (Z)",
                file_prefix="self_distance_z",
            )
    if last_step >= 0:
        _save_checkpoint(
            checkpoints_dir,
            model,
            decoder,
            optimizer,
            last_step,
            tag="last",
        )
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
