#!/usr/bin/env python3
"""Training loop for the JEPA world model with local/global specialization and SIGReg."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections import defaultdict
import os
from pathlib import Path
from typing import Annotated, Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import math
import random
import shutil
import csv
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import tomli_w
import tyro
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from time import perf_counter

try:
    from pytorch_optimizer import SOAP
except ImportError as _soap_import_error:
    SOAP = None
else:
    _soap_import_error = None

from recon.data import list_trajectories, load_frame_as_tensor, short_traj_state_label
from utils.device_utils import pick_device
from jepa_world_model.conv_encoder_decoder import Encoder as ConvEncoder, VisualizationDecoder as ConvVisualizationDecoder
from jepa_world_model.loss_recon import (
    FocalL1Loss,
    HardnessWeightedL1Loss,
    HardnessWeightedMSELoss,
    HardnessWeightedMedianLoss,
    build_feature_pyramid,
    multi_scale_hardness_loss_box,
    multi_scale_hardness_loss_gaussian,
    patch_recon_loss,
)
from jepa_world_model.loss_sigreg import sigreg_loss
from jepa_world_model.loss_geometry import GeometryLossConfig, geometry_ranking_loss
from jepa_world_model.format import (
    _format_elapsed_time,
    _format_flops,
    _format_param_count,
    format_shape_summary,
)
from jepa_world_model.flops import calculate_flops_per_step
from jepa_world_model.metadata import write_run_metadata, write_git_metadata
from jepa_world_model.model_config import ModelConfig
from jepa_world_model.vis import (
    describe_action_tensor,
    save_embedding_projection,
    save_input_batch_visualization,
    save_rollout_sequence_batch,
    save_temporal_pair_visualization,
)
from jepa_world_model.actions import compress_actions_to_ids, decode_action_id
from jepa_world_model.vis_diagnostics import DiagnosticsConfig, save_diagnostics_outputs
from jepa_world_model.vis_graph_diagnostics import save_graph_diagnostics
from jepa_world_model.vis_self_distance import write_self_distance_outputs
from jepa_world_model.vis_state_embedding import write_state_embedding_outputs, _rollout_hidden_states
from jepa_world_model.vis_hard_samples import save_hard_example_grid
from jepa_world_model.vis_vis_ctrl import (
    compute_vis_ctrl_metrics,
    save_composition_error_plot,
    save_smoothness_plot,
    save_stability_plot,
    write_vis_ctrl_metrics_csv,
)
from jepa_world_model.encoder_schedule import _derive_encoder_schedule, _suggest_encoder_schedule
from jepa_world_model.step_schedule import _parse_schedule, _should_run_schedule
from jepa_world_model.data import TrajectorySequenceDataset, collate_batch, load_actions_for_trajectory
from jepa_world_model.hard_sample_reservoir import HardSampleReservoir, inject_hard_examples_into_batch
from jepa_world_model.vis_rollout import (
    VisualizationSelection,
    VisualizationSequence,
    render_rollout_batch,
)



# ------------------------------------------------------------
# Model components
# ------------------------------------------------------------

Encoder = ConvEncoder
VisualizationDecoder = ConvVisualizationDecoder
LegacyEncoder = ConvEncoder
LegacyVisualizationDecoder = ConvVisualizationDecoder


class HiddenToSProjector(nn.Module):
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


class BeliefUpdate(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        action_dim: int,
        film_layers: int,
        state_dim: int,
        predictor_layers: int,
    ) -> None:
        super().__init__()
        if predictor_layers < 1:
            raise ValueError("predictor_layers must be >= 1.")
        self.in_proj = nn.Linear(embedding_dim + action_dim + state_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(predictor_layers - 1)]
        )
        self.out_proj = nn.Linear(hidden_dim, embedding_dim)
        self.h_out = nn.Linear(hidden_dim, state_dim)
        self.activation = nn.SiLU(inplace=True)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.predictor_layers = predictor_layers

    def forward(
        self, embeddings: torch.Tensor, hidden_state: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if not (embeddings.shape[:-1] == actions.shape[:-1] == hidden_state.shape[:-1]):
            raise ValueError("Embeddings, hidden state, and actions must share leading dimensions for predictor conditioning.")
        original_shape = embeddings.shape[:-1]
        emb_flat = embeddings.reshape(-1, embeddings.shape[-1])
        act_flat = actions.reshape(-1, actions.shape[-1])
        h_flat = hidden_state.reshape(-1, hidden_state.shape[-1])
        hidden = self.activation(self.in_proj(torch.cat([emb_flat, h_flat, act_flat], dim=-1)))
        for layer in self.hidden_layers:
            hidden = self.activation(layer(hidden))
        h_delta = self.h_out(hidden)
        h_next = h_delta + h_flat
        h_next = h_next.view(*original_shape, h_next.shape[-1])
        return h_next

    def shape_info(self) -> Dict[str, Any]:
        return {
            "module": "BeliefUpdate",
            "latent_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "conditioning": "concat(z,h,action)",
            "layers": self.predictor_layers,
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


class HiddenToDeltaProjector(nn.Module):
    """Project hidden state to a delta in the target latent space."""

    def __init__(self, h_dim: int, target_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )
        self.h_dim = h_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        original_shape = h.shape[:-1]
        h_flat = h.reshape(-1, h.shape[-1])
        delta = self.net(h_flat)
        return delta.view(*original_shape, delta.shape[-1])


class HiddenActionDeltaProjector(nn.Module):
    """Predict state delta from hidden state and action."""

    def __init__(self, h_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(h_dim + action_dim),
            nn.Linear(h_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, h_dim),
        )
        self.h_dim = h_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

    def forward(self, h: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if h.shape[:-1] != actions.shape[:-1]:
            raise ValueError("Hidden state and actions must share leading dimensions.")
        original_shape = h.shape[:-1]
        h_flat = h.reshape(-1, h.shape[-1])
        act_flat = actions.reshape(-1, actions.shape[-1])
        delta = self.net(torch.cat([h_flat, act_flat], dim=-1))
        return delta.view(*original_shape, delta.shape[-1])


class InverseDynamicsHead(nn.Module):
    """Predict action from consecutive observation embeddings."""

    def __init__(self, z_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(z_dim * 2),
            nn.Linear(z_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        if z_t.shape != z_next.shape:
            raise ValueError("z_t and z_next must have matching shapes.")
        original_shape = z_t.shape[:-1]
        flat = torch.cat([z_t, z_next], dim=-1).reshape(-1, z_t.shape[-1] * 2)
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
INVERSE_DYNAMICS_LOSS = nn.BCEWithLogitsLoss()


# ------------------------------------------------------------
# Configs and containers
# ------------------------------------------------------------


@dataclass
class LossWeights:
    """
    Loss weight semantics (who gets supervised and where gradients flow):
    - loss_jepa (latent transition):
        * What: predictor takes [z_t, h_t, action_t] -> z_hat_{t+1}; MSE vs detached z_{t+1}.
        * Grads: into predictor; into encoder via z_t (and h_t path if attached); target path is stop-grad.
        * Purpose: advance the observable latent consistent with the next encoded frame.
    - loss_h2z: hidden->z projection; z_hat_from_h vs z (detached); shapes hidden path without moving encoder targets.
    - loss_h2z_2hop: 2-hop composability (arbitrary action sequence) via predicted h-delta composition.
    - loss_pixel_delta: pixel delta reconstruction on decoded frames.
    Other losses (recon, sigreg, inverse dynamics, etc.) behave as before.
    """
    # Latent transition supervision: ẑ_{t+1} (from predictor) vs detached z_{t+1}; shapes encoder+predictor via z_t.
    jepa: float = 1.0
    sigreg: float = 0.01

    # Image/pixel reconstruction
    recon: float = 0.0
    recon_patch: float = 0.0
    recon_multi_gauss: float = 0.0
    recon_multi_box: float = 0.3

    # Pixel delta reconstruction loss: recon(z_{t+1}) - recon(z_t) vs x_{t+1} - x_t.
    pixel_delta: float = 0.0

    # Project hidden→z: ẑ_from_h vs z (detached); shapes hidden path without pushing encoder targets.
    h2z: float = 0.0
    # 2-hop composability (arbitrary actions): enforce predicted h-delta composition across two steps.
    h2z_2hop: float = 0.0

    # Auxiliary delta prediction from hidden state to z.
    delta_z: float = 0.0

    # Inverse dynamics from consecutive z pairs (z_t, z_{t+1}).
    inverse_dynamics_z: float = 0.0
    # Inverse dynamics from consecutive h pairs (h_t, h_{t+1}).
    inverse_dynamics_h: float = 0.1

    # Goal-conditioned ranking loss on s = g(stopgrad(h)).
    geometry_rank: float = 0.0

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
class VisCtrlConfig:
    enabled: bool = True
    sample_sequences: int = 128
    knn_k_values: Tuple[int, ...] = (1, 2, 5, 10)
    knn_chunk_size: int = 512
    min_action_count: int = 5
    stability_delta: int = 1


@dataclass
class TrainConfig:
    data_root: Path = Path("data.gridworldkey_wander_to_key")
    output_dir: Path = Path("out.jepa_world_model_trainer")
    log_schedule: Annotated[
        Union[str, Tuple[Tuple[int, Optional[int]], ...]],
        tyro.conf.arg(
            help=(
                "Logging schedule entries use every_steps:max_step (or None for no cap). "
                "Example: '10:None' or '10:100, 50:1000'. Commas or spaces separate entries."
            )
        ),
    ] = "10:None"
    vis_schedule: Annotated[
        Union[str, Tuple[Tuple[int, Optional[int]], ...]],
        tyro.conf.arg(
            help=(
                "Visualization schedule entries use every_steps:max_step (or None for no cap). "
                "Example: '10:100 50:1000 100:10000 200:None'. Commas or spaces separate entries."
            )
        ),
    ] = "10:100 50:1000 100:10000 200:None"
    checkpoint_every_steps: int = 100
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
    use_soap: bool = False

    # Loss configuration
    loss_weights: LossWeights = field(default_factory=LossWeights)
    loss_normalization_enabled: bool = False
    normalize_losses: NormalizeLossesConfig = field(default_factory=NormalizeLossesConfig)
    detach_decoder: bool = False

    # Specific losses
    sigreg: LossSigRegConfig = field(default_factory=LossSigRegConfig)
    geometry: GeometryLossConfig = field(default_factory=GeometryLossConfig)
    patch_recon: LossReconPatchConfig = field(default_factory=LossReconPatchConfig)
    recon_multi_gauss: LossMultiScaleGaussReconConfig = field(default_factory=LossMultiScaleGaussReconConfig)
    recon_multi_box: LossMultiScaleBoxReconConfig = field(default_factory=LossMultiScaleBoxReconConfig)

    # Visualization
    vis: VisConfig = field(default_factory=VisConfig)
    hard_example: HardExampleConfig = field(default_factory=HardExampleConfig)
    debug_visualization: DebugVisualization = field(default_factory=DebugVisualization)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    graph_diagnostics: GraphDiagnosticsConfig = field(default_factory=GraphDiagnosticsConfig)
    vis_ctrl: VisCtrlConfig = field(default_factory=VisCtrlConfig)

    # CLI-only field (not part of training config, used for experiment metadata)
    title: Annotated[Optional[str], tyro.conf.arg(aliases=["-m"])] = None


@dataclass
class SelfDistanceInputs:
    frames: torch.Tensor  # [1, T, C, H, W] on CPU
    frame_paths: List[Path]  # relative to run directory
    frame_labels: List[str]
    trajectory_label: str
    actions: np.ndarray  # [T, action_dim]
    action_labels: List[str]
    action_dim: int


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
        self.predictor = BeliefUpdate(
            self.embedding_dim,
            cfg.hidden_dim,
            cfg.action_dim * 2,
            cfg.predictor_film_layers,
            cfg.state_dim,
            cfg.predictor_layers,
        )
        self.state_dim = cfg.state_dim
        self.h2s = HiddenToSProjector(
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
        self.h_to_delta_z = HiddenToDeltaProjector(
            cfg.state_dim,
            self.embedding_dim,
            cfg.hidden_dim,
        )
        self.h_to_delta_h = HiddenActionDeltaProjector(
            cfg.state_dim,
            cfg.action_dim,
            cfg.hidden_dim,
        )
        self.inverse_dynamics_z = InverseDynamicsHead(
            self.embedding_dim,
            cfg.hidden_dim,
            cfg.action_dim,
        )
        self.inverse_dynamics_h = InverseDynamicsHead(
            cfg.state_dim,
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Roll belief update across sequence to produce z_hat and h states."""
    b, t, _ = embeddings.shape
    if t < 2:
        zero = embeddings.new_tensor(0.0)
        return (
            zero,
            zero,
            embeddings.new_zeros((b, t, model.state_dim)),
        )
    preds = []
    h_preds = []
    h_states = [embeddings.new_zeros(b, model.state_dim, device=embeddings.device)]
    paired_actions = _pair_actions(actions)
    for step in range(t - 1):
        z_t = embeddings[:, step]
        h_t = h_states[-1]
        act_t = paired_actions[:, step]
        h_next = model.predictor(z_t, h_t, act_t)
        z_pred = model.h_to_z(h_next)
        preds.append(z_pred)
        h_preds.append(h_next)
        h_states.append(h_next)
    return (
        torch.stack(preds, dim=1),
        torch.stack(h_preds, dim=1),
        torch.stack(h_states, dim=1),
    )


def jepa_loss(
    model: JEPAWorldModel, outputs: Dict[str, torch.Tensor], actions: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """JEPA loss using predictor conditioned on z, h, and action."""
    embeddings = outputs["embeddings"]
    preds, h_preds, h_states = _predictor_rollout(model, embeddings, actions)
    if embeddings.shape[1] < 2:
        zero = embeddings.new_tensor(0.0)
        return zero, embeddings.new_zeros(embeddings[:, :-1].shape), h_preds, h_states
    target = embeddings[:, 1:].detach()
    return JEPA_LOSS(preds, target), preds, h_preds, h_states


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


def h2z_loss(
    model: JEPAWorldModel,
    embeddings: torch.Tensor,
    h_states: torch.Tensor,
    start: int,
) -> torch.Tensor:
    if h_states.numel() == 0 or embeddings.shape[1] - start <= 0:
        return embeddings.new_tensor(0.0)
    h_stack = h_states[:, start:]
    z_hat_from_h = model.h_to_z(h_stack)
    return F.mse_loss(z_hat_from_h, embeddings[:, start:].detach())


def h2z_2hop_composability_loss(
    model: JEPAWorldModel,
    h_states: torch.Tensor,
    actions: torch.Tensor,
    start: int,
) -> torch.Tensor:
    if h_states.numel() == 0 or h_states.shape[1] - start <= 2:
        return h_states.new_tensor(0.0)
    # Compare sequential predictor rollout vs composed deltas from a separate delta head.
    h0 = h_states[:, start:-2]
    h1 = h_states[:, start + 1 : -1]
    h2 = h_states[:, start + 2 :]
    a0 = actions[:, start:-2]
    a1 = actions[:, start + 1 : -1]
    delta0 = model.h_to_delta_h(h0, a0)
    h1_alt = h0 + delta0
    delta1 = model.h_to_delta_h(h1_alt, a1)
    h2_alt = h1_alt + delta1
    return F.mse_loss(h2_alt, h2)


def delta_z_loss(
    model: JEPAWorldModel,
    embeddings: torch.Tensor,
    h_states: torch.Tensor,
    start: int,
) -> torch.Tensor:
    if h_states.numel() == 0 or embeddings.shape[1] - start <= 1:
        return embeddings.new_tensor(0.0)
    h_stack_delta = h_states[:, start:-1]
    delta_hat = model.h_to_delta_z(h_stack_delta)
    delta_target = (embeddings[:, start + 1 :] - embeddings[:, start:-1]).detach()
    return F.mse_loss(delta_hat, delta_target)


def geometry_rank_loss(
    model: JEPAWorldModel,
    h_states: torch.Tensor,
    cfg: LossGeometryConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if h_states.numel() == 0:
        zero = h_states.new_tensor(0.0)
        return zero, zero
    s_geom = model.h2s(h_states.detach())
    return geometry_ranking_loss(s_geom, cfg)


def pixel_delta_loss(recon: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    if frames.shape[1] <= 1:
        return frames.new_tensor(0.0)
    delta_target = frames[:, 1:] - frames[:, :-1]
    delta_pred = recon[:, 1:] - recon[:, :-1]
    return RECON_LOSS(delta_pred, delta_target)


def inverse_dynamics_z_loss(
    model: JEPAWorldModel,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    if embeddings.shape[1] <= 1:
        raise AssertionError("inverse_dynamics_z requires seq_len > 1.")
    z_t = embeddings[:, :-1]
    z_next = embeddings[:, 1:]
    logits_z = model.inverse_dynamics_z(z_t, z_next)
    return INVERSE_DYNAMICS_LOSS(logits_z, actions[:, :-1].float())


def inverse_dynamics_h_loss(
    model: JEPAWorldModel,
    h_states: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    if h_states.shape[1] <= 1:
        raise AssertionError("inverse_dynamics_h requires seq_len > 1.")
    h_t = h_states[:, :-1]
    h_next = h_states[:, 1:]
    logits_h = model.inverse_dynamics_h(h_t, h_next)
    return INVERSE_DYNAMICS_LOSS(logits_h, actions[:, :-1].float())


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
    loss_norm_ema: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], Optional[BatchDifficultyInfo]]:
    metrics, difficulty_info, world_loss, grads = _compute_losses_and_metrics(
        model,
        decoder,
        batch,
        cfg,
        weights,
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
    track_hard_examples: bool,
    for_training: bool,
    optimizer: Optional[torch.optim.Optimizer],
    loss_norm_ema: Optional[Dict[str, float]] = None,
    loss_norm_enabled: bool = True,
    loss_norm_decay: float = 0.99,
    update_loss_norm: bool = True,
) -> Tuple[Dict[str, float], Optional[BatchDifficultyInfo], torch.Tensor, Tuple[float, float]]:
    # Losses below assume at least one timestep; guard early for clarity.
    if batch[0].shape[1] <= 0:
        raise AssertionError("Sequence length must be positive.")
    if batch[0].shape[0] <= 0:
        raise AssertionError("Batch size must be positive.")

    # Naming guide:
    # x: raw pixel frames (B, T, C, H, W) used for recon and pixel-space losses.
    # z: per-frame encoder embeddings used for JEPA and z-space auxiliaries.
    # h: hidden/belief state produced by the dynamics predictor.
    # s: geometry/planning head derived from h for ranking/geometry losses.
    x_frames, a_seq = batch[0], batch[1]
    batch_paths = batch[2] if len(batch) > 2 else None
    batch_indices = batch[3] if len(batch) > 3 else None
    device = next(model.parameters()).device
    x_frames = x_frames.to(device)
    a_seq = a_seq.to(device)

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    # Core JEPA losses
    loss_jepa = x_frames.new_tensor(0.0)
    loss_sigreg = x_frames.new_tensor(0.0)

    # z (Recon) losses
    loss_recon = x_frames.new_tensor(0.0)
    loss_recon_multi_gauss = x_frames.new_tensor(0.0)
    loss_recon_multi_box = x_frames.new_tensor(0.0)
    loss_recon_patch = x_frames.new_tensor(0.0)

    # z (Auxiliary) losses
    loss_pixel_delta = x_frames.new_tensor(0.0)
    loss_delta_z = x_frames.new_tensor(0.0)
    loss_inverse_dynamics_z = x_frames.new_tensor(0.0)

    # h (Hidden) losses
    loss_h2z = x_frames.new_tensor(0.0)
    loss_h2z_2hop = x_frames.new_tensor(0.0)
    loss_inverse_dynamics_h = x_frames.new_tensor(0.0)

    # s (Geometry) losses
    loss_geometry_rank = x_frames.new_tensor(0.0)
    geometry_rank_accuracy = x_frames.new_tensor(0.0)

    # -------------------------------------------------------------------------
    # Calculate required inputs
    # -------------------------------------------------------------------------

    # NOTE: We may not actually need each of these inputs, but it majorly simplifies the
    #   conditionals, since some inputs are needed in multiple branches.

    encode_outputs = model.encode_sequence(x_frames)
    z_embeddings = encode_outputs["embeddings"]
    loss_jepa_raw, _, h_preds, h_states = jepa_loss(model, encode_outputs, a_seq)

    z_for_decoder = z_embeddings.detach() if cfg.detach_decoder else z_embeddings
    x_recon = decoder(z_for_decoder)

    seq_len = z_embeddings.shape[1]
    warmup_frames = max(getattr(model.cfg, "state_warmup_frames", 0), 0)

    # Warmup before applying hidden-state losses to avoid cold-start transients.
    start_frame = max(min(warmup_frames, seq_len - 1), 0)

    # -------------------------------------------------------------------------
    # Losses
    # -------------------------------------------------------------------------

    # Core JEPA losses
    if weights.jepa > 0:
        loss_jepa = loss_jepa_raw

    if weights.sigreg > 0:
        loss_sigreg = sigreg_loss(z_embeddings, cfg.sigreg.projections)

    # z (Recon) Reconstruction and pixel-space losses
    if weights.recon > 0:
        loss_recon = RECON_LOSS(x_recon, x_frames)

    if weights.recon_multi_gauss > 0:
        loss_recon_multi_gauss = multi_scale_recon_loss_gauss(x_recon, x_frames, cfg.recon_multi_gauss)

    if weights.recon_multi_box > 0:
        loss_recon_multi_box = multi_scale_recon_loss_box(x_recon, x_frames, cfg.recon_multi_box)

    if weights.recon_patch > 0:
        loss_recon_patch = patch_recon_loss(
            x_recon,
            x_frames,
            cfg.patch_recon.patch_sizes,
            loss_fn=RECON_LOSS,
        )

    if weights.pixel_delta > 0:
        loss_pixel_delta = pixel_delta_loss(x_recon, x_frames)

    # z (Auxiliary) losses
    if weights.inverse_dynamics_z > 0:
        loss_inverse_dynamics_z = inverse_dynamics_z_loss(model, z_embeddings, a_seq)

    if weights.delta_z > 0:
        loss_delta_z = delta_z_loss(model, z_embeddings, h_states, start_frame)

    # h (Hidden) Hidden-state dynamics and cross-projection losses
    if weights.h2z > 0:
        loss_h2z = h2z_loss(model, z_embeddings, h_states, start_frame)

    if weights.h2z_2hop > 0:
        loss_h2z_2hop = h2z_2hop_composability_loss(model, h_states, a_seq, start_frame)

    # h (Auxiliary)
    if weights.inverse_dynamics_h > 0:
        loss_inverse_dynamics_h = inverse_dynamics_h_loss(model, h_states, a_seq)

    # s (Geometry)
    if weights.geometry_rank > 0:
        loss_geometry_rank, geometry_rank_accuracy = geometry_rank_loss(model, h_states, cfg.geometry)

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
        + weights.sigreg * _scaled("loss_sigreg", loss_sigreg)
        + weights.h2z * _scaled("loss_h2z", loss_h2z)
        + weights.h2z_2hop * _scaled("loss_h2z_2hop", loss_h2z_2hop)
        + weights.delta_z * _scaled("loss_delta_z", loss_delta_z)
        + weights.geometry_rank * _scaled("loss_geometry_rank", loss_geometry_rank)
        + weights.inverse_dynamics_z * _scaled("loss_inverse_dynamics_z", loss_inverse_dynamics_z)
        + weights.inverse_dynamics_h * _scaled("loss_inverse_dynamics_h", loss_inverse_dynamics_h)
        + weights.recon * _scaled("loss_recon", loss_recon)
        + weights.recon_multi_gauss * _scaled("loss_recon_multi_gauss", loss_recon_multi_gauss)
        + weights.recon_multi_box * _scaled("loss_recon_multi_box", loss_recon_multi_box)
        + weights.recon_patch * _scaled("loss_recon_patch", loss_recon_patch)
        + weights.pixel_delta * _scaled("loss_pixel_delta", loss_pixel_delta)
    )

    world_grad_norm = 0.0
    decoder_grad_norm = 0.0
    if for_training and optimizer is not None:
        optimizer.zero_grad()
        world_loss.backward()
        world_grad_norm = grad_norm(model.parameters())
        decoder_grad_norm = grad_norm(decoder.parameters())
        optimizer.step()

    difficulty_info: Optional[BatchDifficultyInfo] = None
    if (
        track_hard_examples
        and x_recon is not None
        and batch_paths is not None
        and batch_indices is not None
        and len(batch_paths) == x_frames.shape[0]
    ):
        per_frame_errors = (x_recon.detach() - x_frames.detach()).abs().mean(dim=(2, 3, 4))
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
        "loss_recon": loss_recon.item(),
        "loss_recon_multi_gauss": loss_recon_multi_gauss.item(),
        "loss_recon_multi_box": loss_recon_multi_box.item(),
        "loss_recon_patch": loss_recon_patch.item(),
        "loss_pixel_delta": loss_pixel_delta.item(),
        "loss_h2z": loss_h2z.item(),
        "loss_h2z_2hop": loss_h2z_2hop.item(),
        "loss_delta_z": loss_delta_z.item(),
        "loss_geometry_rank": loss_geometry_rank.item(),
        "geometry_rank_accuracy": geometry_rank_accuracy.item(),
        "loss_inverse_dynamics_z": loss_inverse_dynamics_z.item(),
        "loss_inverse_dynamics_h": loss_inverse_dynamics_h.item(),
        "loss_world": world_loss.item(),
    }
    if for_training:
        metrics["grad_world"] = world_grad_norm
        metrics["grad_decoder"] = decoder_grad_norm

    return metrics, difficulty_info, world_loss, (world_grad_norm, decoder_grad_norm)


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
    if weights.sigreg <= 0:
        filtered.pop("loss_sigreg", None)
    if weights.recon <= 0:
        filtered.pop("loss_recon", None)
    if weights.recon_multi_gauss <= 0:
        filtered.pop("loss_recon_multi_gauss", None)
    if weights.recon_multi_box <= 0:
        filtered.pop("loss_recon_multi_box", None)
    if weights.recon_patch <= 0:
        filtered.pop("loss_recon_patch", None)
    if weights.pixel_delta <= 0:
        filtered.pop("loss_pixel_delta", None)
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
    if weights.h2z <= 0:
        filtered.pop("loss_h2z", None)
    if weights.h2z_2hop <= 0:
        filtered.pop("loss_h2z_2hop", None)
    if weights.delta_z <= 0:
        filtered.pop("loss_delta_z", None)
    if weights.inverse_dynamics_z <= 0:
        filtered.pop("loss_inverse_dynamics_z", None)
    if weights.inverse_dynamics_h <= 0:
        filtered.pop("loss_inverse_dynamics_h", None)
    if weights.geometry_rank <= 0:
        filtered.pop("loss_geometry_rank", None)
        filtered.pop("geometry_rank_accuracy", None)
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
    "loss_recon",
    "loss_recon_multi_gauss",
    "loss_recon_multi_box",
    "loss_recon_patch",
    "loss_pixel_delta",
    "loss_h2z",
    "loss_h2z_2hop",
    "loss_delta_z",
    "loss_geometry_rank",
    "geometry_rank_accuracy",
    "loss_inverse_dynamics_z",
    "loss_inverse_dynamics_h",
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
    recon: List[float] = field(default_factory=list)
    recon_multi_gauss: List[float] = field(default_factory=list)
    recon_multi_box: List[float] = field(default_factory=list)
    recon_patch: List[float] = field(default_factory=list)
    pixel_delta: List[float] = field(default_factory=list)
    h2z: List[float] = field(default_factory=list)
    h2z_2hop: List[float] = field(default_factory=list)
    delta_z: List[float] = field(default_factory=list)
    geometry_rank: List[float] = field(default_factory=list)
    geometry_rank_accuracy: List[float] = field(default_factory=list)
    inverse_dynamics_z: List[float] = field(default_factory=list)
    inverse_dynamics_h: List[float] = field(default_factory=list)
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
        self.recon.append(metrics["loss_recon"])
        self.recon_multi_gauss.append(metrics["loss_recon_multi_gauss"])
        self.recon_multi_box.append(metrics["loss_recon_multi_box"])
        self.recon_patch.append(metrics["loss_recon_patch"])
        self.pixel_delta.append(metrics.get("loss_pixel_delta", 0.0))
        self.h2z.append(metrics["loss_h2z"])
        self.h2z_2hop.append(metrics["loss_h2z_2hop"])
        self.delta_z.append(metrics.get("loss_delta_z", 0.0))
        self.geometry_rank.append(metrics.get("loss_geometry_rank", 0.0))
        self.geometry_rank_accuracy.append(metrics.get("geometry_rank_accuracy", 0.0))
        self.inverse_dynamics_z.append(metrics.get("loss_inverse_dynamics_z", 0.0))
        self.inverse_dynamics_h.append(metrics.get("loss_inverse_dynamics_h", 0.0))
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
            history.recon,
            history.recon_multi_gauss,
            history.recon_multi_box,
            history.recon_patch,
            history.pixel_delta,
            history.h2z,
            history.h2z_2hop,
            history.delta_z,
            history.geometry_rank,
            history.geometry_rank_accuracy,
            history.inverse_dynamics_z,
            history.inverse_dynamics_h,
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
        "h2z_2hop": _color(16),
        "geometry_rank_accuracy": _color(17),
        "inverse_dynamics_z": _color(18),
        "inverse_dynamics_h": _color(19),
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
    if any(val != 0.0 for val in history.h2z_2hop):
        plt.plot(history.steps, history.h2z_2hop, label="h2z_2hop", color=color_map["h2z_2hop"])
    if any(val != 0.0 for val in history.geometry_rank_accuracy):
        plt.plot(
            history.steps,
            history.geometry_rank_accuracy,
            label="geometry_rank_accuracy",
            color=color_map["geometry_rank_accuracy"],
        )
    if any(val != 0.0 for val in history.inverse_dynamics_z):
        plt.plot(history.steps, history.inverse_dynamics_z, label="inverse_dynamics_z", color=color_map["inverse_dynamics_z"])
    if any(val != 0.0 for val in history.inverse_dynamics_h):
        plt.plot(history.steps, history.inverse_dynamics_h, label="inverse_dynamics_h", color=color_map["inverse_dynamics_h"])
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


def _count_parameters(modules: Iterable[nn.Module]) -> int:
    total = 0
    for module in modules:
        total += sum(p.numel() for p in module.parameters())
    return total


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
    return [short_traj_state_label(path) for path in slice_paths]


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
    actions = load_actions_for_trajectory(data_root / chosen, expected_length=len(frames))
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
    def _predict_z_and_h(
        current_embed: torch.Tensor, current_hidden: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_next = model.predictor(current_embed, current_hidden, action)
        z_pred = model.h_to_z(h_next)
        return z_pred, h_next

    vis_frames = batch_cpu[0].to(device)
    vis_actions = batch_cpu[1].to(device)
    frame_paths = batch_cpu[2]
    vis_outputs = model.encode_sequence(vis_frames)
    vis_embeddings = vis_outputs["embeddings"]
    # Decoder no longer uses detail_skip - all spatial info is in the latent
    decoded_frames = decoder(vis_embeddings)
    paired_actions = _pair_actions(vis_actions)
    items = render_rollout_batch(
        vis_frames=vis_frames,
        vis_actions=vis_actions,
        embeddings=vis_embeddings,
        decoded_frames=decoded_frames,
        rows=rows,
        rollout_steps=rollout_steps,
        max_columns=max_columns,
        selection=selection,
        log_deltas=log_deltas,
        predictor=_predict_z_and_h,
        decode_fn=decoder,
        paired_actions=paired_actions,
        state_dim=model.state_dim,
        action_text_fn=describe_action_tensor,
        rng=rng,
    )
    sequences: List[VisualizationSequence] = []
    grad_label = "Gradient Norm" if show_gradients else "Error Heatmap"
    for item in items:
        labels = _extract_frame_labels(frame_paths, item.row_index, item.start_idx, len(item.ground_truth))
        gradients: List[Optional[np.ndarray]] = [None for _ in range(len(item.rollout))]
        for step in range(1, len(item.rollout)):
            current_frame = item.rollout[step]
            if current_frame is None:
                continue
            target_frame = item.ground_truth[step]
            if show_gradients:
                gradients[step] = _prediction_gradient_heatmap(current_frame, target_frame)
            else:
                gradients[step] = _loss_to_heatmap(target_frame, current_frame)
        sequences.append(
            VisualizationSequence(
                ground_truth=item.ground_truth,
                rollout=item.rollout,
                gradients=gradients,
                reconstructions=item.reconstructions,
                labels=labels,
                actions=item.actions,
            )
        )
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
    vis_ctrl_generator = torch.Generator()
    vis_ctrl_generator.manual_seed(python_rng.randint(0, 2**32 - 1))
    hard_reservoir_rng = random.Random(python_rng.randint(0, 2**32 - 1))
    hard_reservoir_val_rng = random.Random(python_rng.randint(0, 2**32 - 1))
    # Dedicated RNGs keep visualization sampling consistent across experiments.
    vis_selection_generator = torch.Generator(device=device)
    vis_selection_generator.manual_seed(python_rng.randint(0, 2**32 - 1))
    training_vis_generator = torch.Generator()
    training_vis_generator.manual_seed(python_rng.randint(0, 2**32 - 1))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = cfg.output_dir / timestamp
    metrics_dir = run_dir / "metrics"
    fixed_vis_dir = run_dir / "vis_fixed"
    rolling_vis_dir = run_dir / "vis_rolling"
    pca_z_dir = run_dir / "pca_z"
    pca_s_dir = run_dir / "pca_s"
    diagnostics_delta_dir = run_dir / "vis_delta_z_pca"
    diagnostics_delta_s_dir = run_dir / "vis_delta_s_pca"
    diagnostics_alignment_dir = run_dir / "vis_action_alignment_z"
    diagnostics_alignment_s_dir = run_dir / "vis_action_alignment_s"
    diagnostics_cycle_dir = run_dir / "vis_cycle_error_z"
    diagnostics_cycle_s_dir = run_dir / "vis_cycle_error_s"
    diagnostics_frames_dir = run_dir / "vis_diagnostics_frames"
    graph_diagnostics_dir = run_dir / "graph_diagnostics_z"
    graph_diagnostics_s_dir = run_dir / "graph_diagnostics_s"
    vis_ctrl_dir = run_dir / "vis_vis_ctrl"
    samples_hard_dir = run_dir / "samples_hard"
    samples_hard_val_dir = run_dir / "samples_hard_val"
    inputs_vis_dir = run_dir / "vis_inputs"
    pair_vis_dir = run_dir / "vis_pairs"
    vis_self_distance_z_dir = run_dir / "vis_self_distance_z"
    vis_self_distance_s_dir = run_dir / "vis_self_distance_s"
    vis_state_embedding_dir = run_dir / "vis_state_embedding"
    vis_odometry_dir = run_dir / "vis_odometry"
    self_distance_z_dir = run_dir / "self_distance_z"
    self_distance_s_dir = run_dir / "self_distance_s"
    checkpoints_dir = run_dir / "checkpoints"

    print(f"[run] Writing outputs to {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pca_z_dir.mkdir(parents=True, exist_ok=True)
    pca_s_dir.mkdir(parents=True, exist_ok=True)
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
    if cfg.vis_ctrl.enabled:
        vis_ctrl_dir.mkdir(parents=True, exist_ok=True)
    samples_hard_dir.mkdir(parents=True, exist_ok=True)
    samples_hard_val_dir.mkdir(parents=True, exist_ok=True)
    vis_self_distance_z_dir.mkdir(parents=True, exist_ok=True)
    vis_self_distance_s_dir.mkdir(parents=True, exist_ok=True)
    vis_state_embedding_dir.mkdir(parents=True, exist_ok=True)
    vis_odometry_dir.mkdir(parents=True, exist_ok=True)
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
        model_cfg.encoder_schedule,
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
    vis_ctrl_batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[List[List[str]]]]] = None
    if cfg.vis_ctrl.enabled:
        if diagnostics_batch_cpu is not None:
            vis_ctrl_batch_cpu = diagnostics_batch_cpu
        elif cfg.vis_ctrl.sample_sequences > 0:
            vis_ctrl_batch_cpu = _build_embedding_batch(
                dataset,
                cfg.vis_ctrl.sample_sequences,
                generator=vis_ctrl_generator,
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
        metrics, difficulty_info = training_step(model, decoder, optimizer, batch, cfg, weights, loss_norm_ema)
        timing_totals["train"] += perf_counter() - train_start

        # Update hard examples.
        if hard_reservoir is not None and difficulty_info is not None:
            hard_reservoir.update(
                difficulty_info.indices, difficulty_info.paths, difficulty_info.scores, difficulty_info.frame_indices
            )

        # Log outputs.
        if _should_run_schedule(global_step, cfg.log_schedule):
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
            recon_vis: Optional[torch.Tensor] = None
            with torch.no_grad():
                was_training = model.training
                decoder_was_training = decoder.training
                model.eval()
                decoder.eval()
                vis_frames = rolling_batch_cpu[0].to(device)
                vis_embeddings = model.encode_sequence(vis_frames)["embeddings"]
                recon_vis = decoder(vis_embeddings).cpu()
                if was_training:
                    model.train()
                if decoder_was_training:
                    decoder.train()
            save_input_batch_visualization(
                inputs_vis_dir / f"inputs_{global_step:07d}.png",
                rolling_batch_cpu[0],
                rolling_batch_cpu[1],
                debug_vis.input_vis_rows,
                recon=recon_vis,
                include_deltas=True,
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
        if _should_run_schedule(global_step, cfg.vis_schedule):
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
                    embed_actions = embedding_batch_cpu[1].to(device)
                    embed_outputs = model.encode_sequence(embed_frames)
                    save_embedding_projection(
                        embed_outputs["embeddings"],
                        pca_z_dir / f"pca_z_{global_step:07d}.png",
                        "PCA z",
                    )
                    h_states = _rollout_hidden_states(model, embed_outputs["embeddings"], embed_actions)
                    s_embeddings = model.h2s(h_states)
                    save_embedding_projection(
                        s_embeddings,
                        pca_s_dir / f"pca_s_{global_step:07d}.png",
                        "PCA s",
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
                    with torch.no_grad():
                        self_dist_frames = self_distance_inputs.frames.to(device)
                        self_dist_embeddings = model.encode_sequence(self_dist_frames)["embeddings"][0]
                    write_self_distance_outputs(
                        self_dist_embeddings,
                        self_distance_inputs,
                        self_distance_z_dir,
                        vis_self_distance_z_dir,
                        global_step,
                        embedding_label="z",
                        title_prefix="Self-distance (Z)",
                        file_prefix="self_distance_z",
                        cosine_prefix="self_distance_z_cosine",
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
                        vis_odometry_dir,
                        global_step,
                        hist_frames_cpu=hist_frames,
                        hist_actions_cpu=hist_actions,
                    )
                if cfg.diagnostics.enabled:
                    diag_frames = diagnostics_batch_cpu[0] if diagnostics_batch_cpu is not None else rolling_batch_cpu[0]
                    diag_actions = diagnostics_batch_cpu[1] if diagnostics_batch_cpu is not None else rolling_batch_cpu[1]
                    diag_paths = diagnostics_batch_cpu[2] if diagnostics_batch_cpu is not None and len(diagnostics_batch_cpu) > 2 else rolling_batch_cpu[2]
                    with torch.no_grad():
                        diag_frames_device = diag_frames.to(device)
                        diag_actions_device = diag_actions.to(device)
                        diag_embeddings = model.encode_sequence(diag_frames_device)["embeddings"]
                        diag_s_embeddings = None
                        if (
                            diagnostics_delta_s_dir is not None
                            and diagnostics_alignment_s_dir is not None
                            and diagnostics_cycle_s_dir is not None
                        ):
                            _, _, diag_h_states = _predictor_rollout(model, diag_embeddings, diag_actions_device)
                            diag_s_embeddings = model.h2s(diag_h_states)
                    save_diagnostics_outputs(
                        diag_embeddings,
                        diag_frames,
                        diag_actions,
                        diag_paths,
                        cfg.diagnostics,
                        diagnostics_delta_dir,
                        diagnostics_alignment_dir,
                        diagnostics_cycle_dir,
                        diagnostics_frames_dir,
                        global_step,
                        s_embeddings=diag_s_embeddings,
                        delta_s_dir=diagnostics_delta_s_dir,
                        alignment_s_dir=diagnostics_alignment_s_dir,
                        cycle_s_dir=diagnostics_cycle_s_dir,
                    )
                if cfg.vis_ctrl.enabled and vis_ctrl_batch_cpu is not None:
                    vis_frames = vis_ctrl_batch_cpu[0].to(device)
                    vis_actions = vis_ctrl_batch_cpu[1].to(device)
                    vis_embeddings = model.encode_sequence(vis_frames)["embeddings"]
                    _, _, vis_h_states = _predictor_rollout(model, vis_embeddings, vis_actions)
                    vis_s_embeddings = model.h2s(vis_h_states)
                    warmup_frames = max(getattr(model.cfg, "state_warmup_frames", 0), 0)
                    metrics_z = compute_vis_ctrl_metrics(
                        vis_embeddings,
                        vis_actions,
                        cfg.vis_ctrl.knn_k_values,
                        warmup_frames,
                        cfg.vis_ctrl.min_action_count,
                        cfg.vis_ctrl.stability_delta,
                        cfg.vis_ctrl.knn_chunk_size,
                    )
                    metrics_s = compute_vis_ctrl_metrics(
                        vis_s_embeddings,
                        vis_actions,
                        cfg.vis_ctrl.knn_k_values,
                        warmup_frames,
                        cfg.vis_ctrl.min_action_count,
                        cfg.vis_ctrl.stability_delta,
                        cfg.vis_ctrl.knn_chunk_size,
                    )
                    metrics_h = compute_vis_ctrl_metrics(
                        vis_h_states,
                        vis_actions,
                        cfg.vis_ctrl.knn_k_values,
                        warmup_frames,
                        cfg.vis_ctrl.min_action_count,
                        cfg.vis_ctrl.stability_delta,
                        cfg.vis_ctrl.knn_chunk_size,
                    )
                    save_smoothness_plot(
                        vis_ctrl_dir / f"smoothness_z_{global_step:07d}.png",
                        metrics_z,
                        "z",
                    )
                    save_smoothness_plot(
                        vis_ctrl_dir / f"smoothness_s_{global_step:07d}.png",
                        metrics_s,
                        "s",
                    )
                    save_smoothness_plot(
                        vis_ctrl_dir / f"smoothness_h_{global_step:07d}.png",
                        metrics_h,
                        "h",
                    )
                    save_composition_error_plot(
                        vis_ctrl_dir / f"composition_error_z_{global_step:07d}.png",
                        metrics_z,
                        "z",
                    )
                    save_composition_error_plot(
                        vis_ctrl_dir / f"composition_error_s_{global_step:07d}.png",
                        metrics_s,
                        "s",
                    )
                    save_composition_error_plot(
                        vis_ctrl_dir / f"composition_error_h_{global_step:07d}.png",
                        metrics_h,
                        "h",
                    )
                    save_stability_plot(
                        vis_ctrl_dir / f"stability_z_{global_step:07d}.png",
                        metrics_z,
                        "z",
                    )
                    save_stability_plot(
                        vis_ctrl_dir / f"stability_s_{global_step:07d}.png",
                        metrics_s,
                        "s",
                    )
                    save_stability_plot(
                        vis_ctrl_dir / f"stability_h_{global_step:07d}.png",
                        metrics_h,
                        "h",
                    )
                    write_vis_ctrl_metrics_csv(
                        metrics_dir / "vis_ctrl_metrics.csv",
                        global_step,
                        metrics_z,
                        metrics_s,
                        metrics_h,
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
            self_dist_frames = self_distance_inputs.frames.to(device)
            self_dist_embeddings = model.encode_sequence(self_dist_frames)["embeddings"][0]
        write_self_distance_outputs(
            self_dist_embeddings,
            self_distance_inputs,
            self_distance_z_dir,
            vis_self_distance_z_dir,
            last_step if last_step >= 0 else 0,
            embedding_label="z",
            title_prefix="Self-distance (Z)",
            file_prefix="self_distance_z",
            cosine_prefix="self_distance_z_cosine",
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
    cfg = replace(
        cfg,
        log_schedule=_parse_schedule(cfg.log_schedule),
        vis_schedule=_parse_schedule(cfg.vis_schedule),
    )
    model_cfg = ModelConfig()
    run_training(cfg, model_cfg, cfg.loss_weights, title=cfg.title)


if __name__ == "__main__":
    main()
