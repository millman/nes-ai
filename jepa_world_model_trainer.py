#!/usr/bin/env python3
"""Training loop for the JEPA world model with local/global specialization and SIGReg."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections import defaultdict
import json
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
from jepa_world_model.loss_geometry import LossGeometryConfig, geometry_ranking_loss
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
)
from jepa_world_model.vis_embedding_projection import save_embedding_projection
from jepa_world_model.vis_input_batch import save_input_batch_visualization
from jepa_world_model.vis_rollout_batch import save_rollout_sequence_batch
from jepa_world_model.vis_temporal_pairs import save_temporal_pair_visualization
from jepa_world_model.actions import compress_actions_to_ids, decode_action_id
from jepa_world_model.config_diagnostics import DiagnosticsConfig, SpikeDiagnosticsConfig
from jepa_world_model.plots.write_action_alignment_crosscheck import (
    write_action_alignment_crosscheck,
)
from jepa_world_model.plots.plot_action_alignment_debug import (
    build_action_alignment_debug,
)
from jepa_world_model.vis_action_alignment import save_action_alignment_detail_plot
from jepa_world_model.rollout import rollout_teacher_forced
from jepa_world_model.plots.write_action_alignment_report import (
    write_action_alignment_report,
)
from jepa_world_model.plots.plot_action_alignment_stats import (
    compute_action_alignment_stats,
)
from jepa_world_model.plots.write_action_alignment_strength import (
    write_action_alignment_strength,
)
from jepa_world_model.plots.write_alignment_debug_csv import (
    write_alignment_debug_csv,
)
from jepa_world_model.plots.write_action_alignment_csv import write_action_alignment_csv
from jepa_world_model.plots.write_action_alignment_full_csv import (
    write_action_alignment_full_csv,
)
from jepa_world_model.plots.write_action_alignment_overview_txt import (
    write_action_alignment_overview_txt,
)
from jepa_world_model.plots.write_action_alignment_pairwise_csv import (
    write_action_alignment_pairwise_csv,
)
from jepa_world_model.plots.write_cycle_error_summary_csv import (
    write_cycle_error_summary_csv,
)
from jepa_world_model.plots.write_cycle_error_values_csv import (
    write_cycle_error_values_csv,
)
from jepa_world_model.plots.write_delta_samples_csv import write_delta_samples_csv
from jepa_world_model.plots.write_delta_variance_csv import write_delta_variance_csv
from jepa_world_model.vis_cycle_error import save_cycle_error_plot
from jepa_world_model.vis_cycle_error import compute_cycle_errors
from jepa_world_model.plots.plot_delta_pca import save_delta_pca_plot
from jepa_world_model.plots.plot_diagnostics_frames import save_diagnostics_frames
from jepa_world_model.plots.plot_action_inverse_map import build_action_inverse_map
from jepa_world_model.plots.build_motion_subspace import build_motion_subspace
from jepa_world_model.plots.plot_variance_report import write_variance_report
from jepa_world_model.plots.plot_variance_spectrum import (
    save_variance_spectrum_plot,
)
from jepa_world_model.plots.plot_diagnostics_extra import (
    StraightLineTrajectory,
    save_ablation_divergence_plot,
    save_drift_by_action_plot,
    save_monotonicity_plot,
    save_norm_timeseries_plot,
    save_path_independence_plot,
    save_rollout_divergence_plot,
    save_straightline_plot,
    save_z_consistency_plot,
)
from jepa_world_model.plots.plot_layout import DEFAULT_DPI, figsize_for_grid
from jepa_world_model.plots.plot_graph_diagnostics import (
    GraphDiagnosticsConfig,
    build_graph_diag_indices,
    compute_graph_diagnostics_stats,
    update_graph_diagnostics_history,
)
from jepa_world_model.plots.plot_edge_consistency_hist import (
    save_edge_consistency_hist_plot,
)
from jepa_world_model.plots.plot_in_degree_hist import save_in_degree_hist_plot
from jepa_world_model.plots.plot_neff_violin import save_neff_violin_plot
from jepa_world_model.plots.plot_rank_cdf import save_rank_cdf_plot
from jepa_world_model.vis_self_distance import write_self_distance_outputs
from jepa_world_model.vis_state_embedding import write_state_embedding_outputs
from jepa_world_model.vis_hard_samples import save_hard_example_grid
from jepa_world_model.vis_visualization_batch import _render_visualization_batch
from jepa_world_model.vis_composability import compute_composability_series, save_composability_plot
from jepa_world_model.plots.plot_two_step_composition_error import (
    save_two_step_composition_error_plot,
)
from jepa_world_model.plots.plot_smoothness_knn_distance_eigenvalue_spectrum import (
    save_smoothness_knn_distance_eigenvalue_spectrum_plot,
)
from jepa_world_model.plots.plot_neighborhood_stability import save_neighborhood_stability_plot
from jepa_world_model.vis_vis_ctrl_metrics import (
    compute_vis_ctrl_metrics,
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


class HiddenToStateProjector(nn.Module):
    """Project hidden state to a planning (p) or feature (f) embedding."""

    def __init__(
        self,
        h_dim: int,
        s_dim: int,
        hidden_dim: int,
        unit_norm: bool,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(h_dim))
        layers.extend(
            [
                nn.Linear(h_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, s_dim),
            ]
        )
        self.net = nn.Sequential(*layers)
        self.h_dim = h_dim
        self.s_dim = s_dim
        self.hidden_dim = hidden_dim
        self.unit_norm = unit_norm
        self.use_layer_norm = use_layer_norm

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        original_shape = h.shape[:-1]
        h_flat = h.reshape(-1, h.shape[-1])
        state = self.net(h_flat)
        if self.unit_norm:
            state = F.normalize(state, dim=-1)
        return state.view(*original_shape, state.shape[-1])


class PredictorNetwork(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        action_dim: int,
        state_dim: int,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embedding_dim + action_dim + state_dim, hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.h_out = nn.Linear(hidden_dim, state_dim)
        self.h_norm = nn.LayerNorm(state_dim) if use_layer_norm else None
        self.activation = nn.SiLU(inplace=True)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.use_layer_norm = use_layer_norm

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
        hidden = self.activation(self.hidden_proj(hidden))
        h_next = self.h_out(hidden)
        if self.h_norm is not None:
            h_next = self.h_norm(h_next)
        h_next = h_next.view(*original_shape, h_next.shape[-1])
        return h_next

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

    def __init__(self, h_dim: int, z_dim: int, hidden_dim: int, use_layer_norm: bool = True) -> None:
        super().__init__()
        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(h_dim))
        layers.extend(
            [
                nn.Linear(h_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, z_dim),
            ]
        )
        self.net = nn.Sequential(*layers)
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        original_shape = h.shape[:-1]
        h_flat = h.reshape(-1, h.shape[-1])
        z_hat = self.net(h_flat)
        return z_hat.view(*original_shape, z_hat.shape[-1])


class ZToHProjector(nn.Module):
    """Project image-anchored embedding to a hidden state seed."""

    def __init__(self, z_dim: int, h_dim: int, hidden_dim: int, use_layer_norm: bool = True) -> None:
        super().__init__()
        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(z_dim))
        layers.extend(
            [
                nn.Linear(z_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, h_dim),
            ]
        )
        self.net = nn.Sequential(*layers)
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        original_shape = z.shape[:-1]
        z_flat = z.reshape(-1, z.shape[-1])
        h_seed = self.net(z_flat)
        return h_seed.view(*original_shape, h_seed.shape[-1])


class HiddenToDeltaProjector(nn.Module):
    """Project hidden state to a delta in the target latent space."""

    def __init__(
        self,
        h_dim: int,
        target_dim: int,
        hidden_dim: int,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(h_dim))
        layers.extend(
            [
                nn.Linear(h_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, target_dim),
            ]
        )
        self.net = nn.Sequential(*layers)
        self.h_dim = h_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        original_shape = h.shape[:-1]
        h_flat = h.reshape(-1, h.shape[-1])
        delta = self.net(h_flat)
        return delta.view(*original_shape, delta.shape[-1])


class HiddenActionDeltaProjector(nn.Module):
    """Predict state delta from hidden state and action."""

    def __init__(
        self,
        h_dim: int,
        action_dim: int,
        hidden_dim: int,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(h_dim + action_dim))
        layers.extend(
            [
                nn.Linear(h_dim + action_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, h_dim),
            ]
        )
        self.net = nn.Sequential(*layers)
        self.h_dim = h_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm

    def forward(self, h: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if h.shape[:-1] != actions.shape[:-1]:
            raise ValueError("Hidden state and actions must share leading dimensions.")
        original_shape = h.shape[:-1]
        h_flat = h.reshape(-1, h.shape[-1])
        act_flat = actions.reshape(-1, actions.shape[-1])
        delta = self.net(torch.cat([h_flat, act_flat], dim=-1))
        return delta.view(*original_shape, delta.shape[-1])


class ActionDeltaProjector(nn.Module):
    """Project action vectors into latent delta prototypes."""

    def __init__(self, action_dim: int, target_dim: int, use_layer_norm: bool = True) -> None:
        super().__init__()
        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(action_dim))
        layers.append(nn.Linear(action_dim, target_dim, bias=False))
        self.net = nn.Sequential(*layers)
        self.action_dim = action_dim
        self.target_dim = target_dim
        self.use_layer_norm = use_layer_norm

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        original_shape = actions.shape[:-1]
        flat = actions.reshape(-1, actions.shape[-1])
        projected = self.net(flat)
        return projected.view(*original_shape, projected.shape[-1])


class InverseDynamicsHead(nn.Module):
    """Predict action from consecutive observation embeddings."""

    def __init__(self, z_dim: int, hidden_dim: int, action_dim: int, use_layer_norm: bool = True) -> None:
        super().__init__()
        layers = []
        input_dim = z_dim
        if use_layer_norm:
            layers.append(nn.LayerNorm(input_dim))
        layers.extend(
            [
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, action_dim),
            ]
        )
        self.net = nn.Sequential(*layers)
        self.use_layer_norm = use_layer_norm

    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        if z_t.shape != z_next.shape:
            raise ValueError("z_t and z_next must have matching shapes.")
        delta = z_next - z_t
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

    def __init__(self, hidden_dim: int, action_dim: int, use_layer_norm: bool = True) -> None:
        super().__init__()
        self.gamma = nn.Linear(action_dim, hidden_dim)
        self.beta = nn.Linear(action_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim) if use_layer_norm else None
        self.use_layer_norm = use_layer_norm

    def forward(self, hidden: torch.Tensor, action_embed: torch.Tensor) -> torch.Tensor:
        conditioned = self.norm(hidden) if self.norm is not None else hidden
        gamma = self.gamma(action_embed)
        beta = self.beta(action_embed)
        return conditioned * (1.0 + gamma) + beta


class ActionFiLM(nn.Module):
    """Stack multiple FiLM layers for action conditioning."""

    def __init__(self, hidden_dim: int, action_dim: int, layers: int, use_layer_norm: bool = True) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [FiLMLayer(hidden_dim, action_dim, use_layer_norm=use_layer_norm) for _ in range(layers)]
        )
        self.act = nn.GELU()

    def forward(self, hidden: torch.Tensor, action_embed: torch.Tensor) -> torch.Tensor:
        out = hidden
        for layer in self.layers:
            out = layer(out, action_embed)
            out = self.act(out)
        return out


RECON_LOSS = nn.MSELoss()
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
    - loss_z2h: z->hidden init projection; h_hat_from_z0 vs h0 (detached); z inputs are also detached so this only trains the z->h projector.
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
    recon_multi_box: float = 1.0

    # Pixel delta reconstruction loss: recon(z_{t+1}) - recon(z_t) vs x_{t+1} - x_t.
    pixel_delta: float = 1.0

    # Project hidden→z: ẑ_from_h vs z (detached); shapes hidden path without pushing encoder targets.
    h2z: float = 1.0
    # Project z→hidden: ĥ_from_z vs h (detached); trains z->h projector.
    z2h: float = 1.0

    # Goal-conditioned ranking loss on p = g(stopgrad(h)).
    # Keeps p useful for planning geometry while avoiding action algebra constraints.
    geometry_rank: float = 1.0

    # Inverse dynamics from consecutive z pairs (z_t, z_{t+1}).
    # Keep at 0: forces z to reveal actions, breaking loop-closure invariance.
    inverse_dynamics_z: float = 0.0
    # Inverse dynamics from consecutive h pairs (h_t, h_{t+1}).
    inverse_dynamics_h: float = 1.0
    # Inverse dynamics from consecutive h pairs (p_t, p_{t+1}).
    inverse_dynamics_p: float = 1.0

    # -------------------------------------------------------------------------
    # Algebra losses for Mario: keep z light, push h to local translation + short composition, and make p algebraic for planning.
    # -------------------------------------------------------------------------
    # z: Level 1 fixed translation; weakly anchor action directions without overfitting perception.
    # h: Level 2 + light Level 3; state-conditioned deltas with short-horizon composition.
    # p: Level 3; a smoother planning space with consistent multi-step structure.

    # --- Z ---
    # Action delta alignment: z_{t+1} - z_t vs learned action prototype.
    # Keep at 0: makes z compose under actions (path-dependent), violating loop closure.
    action_delta_z: float = 0.0

    # k-step rollout consistency in z-space.
    # Encourages short-horizon compositionality without forcing long-horizon rigidity.
    # Keep at 0: pushes z toward action-compositional dynamics rather than place invariance.
    rollout_kstep_z: float = 0.0

    # Pixel reconstruction on predicted z rollouts (decode z_roll vs x_{t+k}).
    # Keep at 0: ties z rollouts to action effects, which should live in h/p instead.
    rollout_recon_z: float = 0.0

    # Projection consistency on predicted z rollouts (enc(dec(z_roll)) vs z_roll).
    # Keep at 0: reinforces action-conditioned z trajectories, hurting loop closure.
    rollout_project_z: float = 0.0

    # --- H ---
    # State-conditioned delta alignment: h_{t+1} - h_t vs E(h_t, a_t).
    # Makes action effects locally predictable (supports momentum, contacts, and walls).
    action_delta_h: float = 1.0

    # Explicit additivity of state deltas across steps.
    # Promotes near-linear multi-step effects; keep light for Mario's nonlinearity.
    additivity_h: float = 0.0

    # k-step rollout consistency in h-space.
    # Encourages short-horizon compositionality in the dynamics state.
    rollout_kstep_h: float = 0.0

    # --- P ---
    # State-conditioned delta alignment: p_{t+1} - p_t vs E(p_t, a_t).
    action_delta_p: float = 1.0

    # Explicit additivity of p deltas across steps.
    additivity_p: float = 1.0

    # k-step rollout consistency in p-space.
    rollout_kstep_p: float = 1.0

    # Pose delta magnitude regularizer.
    mag_p: float = 0.0


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
    rollout_horizon: int = 4

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
    delta_z_detach_target: bool = False

    # Specific losses
    sigreg: LossSigRegConfig = field(default_factory=LossSigRegConfig)
    geometry: LossGeometryConfig = field(default_factory=LossGeometryConfig)
    patch_recon: LossReconPatchConfig = field(default_factory=LossReconPatchConfig)
    recon_multi_gauss: LossMultiScaleGaussReconConfig = field(default_factory=LossMultiScaleGaussReconConfig)
    recon_multi_box: LossMultiScaleBoxReconConfig = field(default_factory=LossMultiScaleBoxReconConfig)

    # Visualization
    vis: VisConfig = field(default_factory=VisConfig)
    hard_example: HardExampleConfig = field(default_factory=HardExampleConfig)
    debug_visualization: DebugVisualization = field(default_factory=DebugVisualization)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    spike_diagnostics: SpikeDiagnosticsConfig = field(default_factory=SpikeDiagnosticsConfig)
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
        self.predictor = PredictorNetwork(
            self.embedding_dim,
            cfg.hidden_dim,
            cfg.action_dim,
            cfg.state_dim,
            use_layer_norm=cfg.layer_norms.h_next,
        )
        self.state_dim = cfg.state_dim
        pose_dim = cfg.pose_dim if cfg.pose_dim is not None else cfg.state_embed_dim
        desc_dim = cfg.descriptor_dim if cfg.descriptor_dim is not None else pose_dim
        self.h2p = HiddenToStateProjector(
            cfg.state_dim,
            pose_dim if pose_dim is not None else cfg.state_dim,
            cfg.hidden_dim,
            cfg.state_embed_unit_norm,
            use_layer_norm=cfg.layer_norms.h2s_projector,
        )
        self.h2f = HiddenToStateProjector(
            cfg.state_dim,
            desc_dim if desc_dim is not None else cfg.state_dim,
            cfg.hidden_dim,
            cfg.descriptor_unit_norm,
            use_layer_norm=cfg.layer_norms.h2s_projector,
        )
        # Backwards-compatible aliases for legacy names.
        self.h_to_z = HiddenToZProjector(
            cfg.state_dim,
            self.embedding_dim,
            cfg.hidden_dim,
            use_layer_norm=cfg.layer_norms.h2z_projector,
        )
        self.z_to_h = ZToHProjector(
            self.embedding_dim,
            cfg.state_dim,
            cfg.hidden_dim,
            use_layer_norm=cfg.layer_norms.z2h_projector,
        )
        self.inverse_dynamics_z = InverseDynamicsHead(
            self.embedding_dim,
            cfg.hidden_dim,
            cfg.action_dim,
            use_layer_norm=cfg.layer_norms.inverse_dynamics,
        )
        self.inverse_dynamics_h = InverseDynamicsHead(
            cfg.state_dim,
            cfg.hidden_dim,
            cfg.action_dim,
            use_layer_norm=cfg.layer_norms.inverse_dynamics,
        )
        self.inverse_dynamics_p = InverseDynamicsHead(
            pose_dim if pose_dim is not None else cfg.state_dim,
            cfg.hidden_dim,
            cfg.action_dim,
            use_layer_norm=cfg.layer_norms.inverse_dynamics,
        )
        self.h_action_delta_projector = HiddenActionDeltaProjector(
            cfg.state_dim,
            cfg.action_dim,
            cfg.hidden_dim,
            use_layer_norm=cfg.layer_norms.action_delta_projector,
        )
        self.p_action_delta_projector = HiddenActionDeltaProjector(
            pose_dim if pose_dim is not None else cfg.state_dim,
            cfg.action_dim,
            cfg.hidden_dim,
            use_layer_norm=cfg.layer_norms.action_delta_projector,
        )
        self.s_action_delta_projector = self.p_action_delta_projector
        self.z_action_delta_projector = ActionDeltaProjector(
            cfg.action_dim,
            self.embedding_dim,
            use_layer_norm=cfg.layer_norms.action_delta_projector,
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
    model: JEPAWorldModel,
    outputs: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    use_z2h_init: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """JEPA loss using predictor conditioned on z, h, and action."""
    embeddings = outputs["embeddings"]
    if embeddings.shape[1] < 2:
        raise AssertionError("JEPA loss requires at least two timesteps.")
    preds, h_preds, h_states = rollout_teacher_forced(
        model,
        embeddings,
        actions,
        use_z2h_init=use_z2h_init,
    )
    target = embeddings[:, 1:].detach()
    return JEPA_LOSS(preds, target), preds, h_preds, h_states


def _rollout_pose(
    model: JEPAWorldModel,
    h_states: torch.Tensor,
    actions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build pose observations from h and integrate pose deltas over actions."""
    pose_obs = model.h2p(h_states)
    bsz, steps, dim = pose_obs.shape
    if steps <= 1 or actions.shape[1] < 1:
        zero_delta = pose_obs.new_zeros((bsz, 0, dim))
        return pose_obs, pose_obs, zero_delta
    max_steps = min(actions.shape[1], steps - 1)
    pose_pred: List[torch.Tensor] = [pose_obs[:, 0]]
    pose_deltas: List[torch.Tensor] = []
    for idx in range(max_steps):
        delta = model.p_action_delta_projector(pose_pred[-1], actions[:, idx])
        pose_next = pose_pred[-1] + delta
        pose_deltas.append(delta)
        pose_pred.append(pose_next)
    pose_pred_tensor = torch.stack(pose_pred, dim=1)
    if pose_pred_tensor.shape[1] < steps:
        pad = pose_pred_tensor[:, -1:].repeat(1, steps - pose_pred_tensor.shape[1], 1)
        pose_pred_tensor = torch.cat([pose_pred_tensor, pad], dim=1)
    return pose_obs, pose_pred_tensor, torch.stack(pose_deltas, dim=1)


def rollout_z_loss(
    model: JEPAWorldModel,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    h_states: torch.Tensor,
    rollout_horizon: int,
    warmup_frames: int,
) -> torch.Tensor:
    if rollout_horizon <= 1:
        raise AssertionError("rollout_horizon must be > 1 for rollout_z_loss.")
    b, t, d = embeddings.shape
    if t < 2:
        raise AssertionError("rollout_z_loss requires at least two timesteps.")
    total = embeddings.new_tensor(0.0)
    steps = 0
    warmup = max(min(warmup_frames, t - 1), 0)
    if warmup >= t - 1:
        raise AssertionError("warmup_frames leaves no rollout steps for rollout_z_loss.")
    for start in range(warmup, t - 1):
        current = embeddings[:, start]
        max_h = min(rollout_horizon, t - start - 1)
        h_current = h_states[:, start]
        for offset in range(max_h):
            act = actions[:, start + offset]
            h_next = model.predictor(current, h_current, act)
            pred = model.h_to_z(h_next)
            target_step = embeddings[:, start + offset + 1].detach()
            total = total + JEPA_LOSS(pred, target_step)
            steps += 1
            current = pred
            h_current = h_next
    if steps <= 0:
        raise AssertionError("rollout_z_loss produced no steps; check rollout_horizon or warmup_frames.")
    return total / steps


def rollout_recon_z_loss(
    model: JEPAWorldModel,
    decoder: VisualizationDecoder,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    frames: torch.Tensor,
    h_states: torch.Tensor,
    rollout_horizon: int,
    warmup_frames: int,
) -> torch.Tensor:
    if rollout_horizon <= 0:
        raise AssertionError("rollout_horizon must be > 0 for rollout_recon_z_loss.")
    b, t, _ = embeddings.shape
    if t < 2:
        raise AssertionError("rollout_recon_z_loss requires at least two timesteps.")
    total = embeddings.new_tensor(0.0)
    steps = 0
    warmup = max(min(warmup_frames, t - 1), 0)
    if warmup >= t - 1:
        raise AssertionError("warmup_frames leaves no rollout steps for rollout_recon_z_loss.")
    for start in range(warmup, t - 1):
        current = embeddings[:, start]
        h_current = h_states[:, start]
        max_h = min(rollout_horizon, t - start - 1)
        for offset in range(max_h):
            act = actions[:, start + offset]
            h_next = model.predictor(current, h_current, act)
            pred = model.h_to_z(h_next)
            decoded = decoder(pred)
            target_frame = frames[:, start + offset + 1]
            total = total + RECON_LOSS(decoded, target_frame)
            steps += 1
            current = pred
            h_current = h_next
    if steps <= 0:
        raise AssertionError("rollout_recon_z_loss produced no steps; check rollout_horizon or warmup_frames.")
    return total / steps


def rollout_project_z_loss(
    model: JEPAWorldModel,
    decoder: VisualizationDecoder,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    h_states: torch.Tensor,
    rollout_horizon: int,
    warmup_frames: int,
) -> torch.Tensor:
    if rollout_horizon <= 0:
        raise AssertionError("rollout_horizon must be > 0 for rollout_project_z_loss.")
    b, t, _ = embeddings.shape
    if t < 2:
        raise AssertionError("rollout_project_z_loss requires at least two timesteps.")
    total = embeddings.new_tensor(0.0)
    steps = 0
    warmup = max(min(warmup_frames, t - 1), 0)
    if warmup >= t - 1:
        raise AssertionError("warmup_frames leaves no rollout steps for rollout_project_z_loss.")
    for start in range(warmup, t - 1):
        current = embeddings[:, start]
        h_current = h_states[:, start]
        max_h = min(rollout_horizon, t - start - 1)
        for offset in range(max_h):
            act = actions[:, start + offset]
            h_next = model.predictor(current, h_current, act)
            pred = model.h_to_z(h_next)
            decoded = decoder(pred)
            z_back = model.encoder(decoded)
            total = total + F.mse_loss(z_back, pred)
            steps += 1
            current = pred
            h_current = h_next
    if steps <= 0:
        raise AssertionError("rollout_project_z_loss produced no steps; check rollout_horizon or warmup_frames.")
    return total / steps


def rollout_h_loss(
    model: JEPAWorldModel,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    h_states: torch.Tensor,
    rollout_horizon: int,
    warmup_frames: int,
) -> torch.Tensor:
    if rollout_horizon <= 1:
        raise AssertionError("rollout_horizon must be > 1 for rollout_h_loss.")
    b, t, _ = embeddings.shape
    if t < 2:
        raise AssertionError("rollout_h_loss requires at least two timesteps.")
    total = embeddings.new_tensor(0.0)
    steps = 0
    warmup = max(min(warmup_frames, t - 1), 0)
    if warmup >= t - 1:
        raise AssertionError("warmup_frames leaves no rollout steps for rollout_h_loss.")
    for start in range(warmup, t - 1):
        current = embeddings[:, start]
        max_h = min(rollout_horizon, t - start - 1)
        h_current = h_states[:, start]
        for offset in range(max_h):
            act = actions[:, start + offset]
            h_next = model.predictor(current, h_current, act)
            pred = model.h_to_z(h_next)
            target_h = h_states[:, start + offset + 1].detach()
            total = total + F.mse_loss(h_next, target_h)
            steps += 1
            current = pred
            h_current = h_next
    if steps <= 0:
        raise AssertionError("rollout_h_loss produced no steps; check rollout_horizon or warmup_frames.")
    return total / steps


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
    if h_states.numel() == 0:
        raise AssertionError("h2z_loss requires non-empty h_states.")
    if embeddings.shape[1] - start <= 0:
        raise AssertionError("h2z_loss requires at least one timestep after start.")
    h_stack = h_states[:, start:]
    z_hat_from_h = model.h_to_z(h_stack)
    return F.mse_loss(z_hat_from_h, embeddings[:, start:].detach())


def z2h_loss(
    model: JEPAWorldModel,
    embeddings: torch.Tensor,
    h_states: torch.Tensor,
    start: int,
) -> torch.Tensor:
    if h_states.numel() == 0:
        raise AssertionError("z2h_loss requires non-empty h_states.")
    if embeddings.shape[1] < 1:
        raise AssertionError("z2h_loss requires at least one timestep.")
    z0 = embeddings[:, :1].detach()
    h0_target = h_states[:, :1].detach()
    h_hat_from_z = model.z_to_h(z0)
    return F.mse_loss(h_hat_from_z, h0_target)




def geometry_rank_loss(
    model: JEPAWorldModel,
    h_states: torch.Tensor,
    cfg: LossGeometryConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if h_states.numel() == 0:
        raise AssertionError("geometry_rank_loss requires non-empty h_states.")
    p_geom = model.h2p(h_states.detach())
    return geometry_ranking_loss(p_geom, cfg)


def pixel_delta_loss(recon: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    if frames.shape[1] <= 1:
        raise AssertionError("pixel_delta_loss requires at least two frames.")
    delta_target = frames[:, 1:] - frames[:, :-1]
    delta_pred = recon[:, 1:] - recon[:, :-1]
    return RECON_LOSS(delta_pred, delta_target)




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
    # p: geometry/planning head derived from h for ranking/geometry losses.
    x_frames, a_seq = batch[0], batch[1]
    batch_paths = batch[2]
    batch_indices = batch[3]
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

    # h (Hidden) losses
    loss_h2z = x_frames.new_tensor(0.0)
    loss_z2h = x_frames.new_tensor(0.0)

    # p (Geometry) losses
    loss_geometry_rank = x_frames.new_tensor(0.0)
    geometry_rank_accuracy = x_frames.new_tensor(0.0)
    geometry_rank_pairs = x_frames.new_tensor(0.0)

    # Inverse dynamics losses
    loss_inverse_dynamics_z = x_frames.new_tensor(0.0)
    loss_inverse_dynamics_h = x_frames.new_tensor(0.0)
    loss_inverse_dynamics_p = x_frames.new_tensor(0.0)
    loss_action_delta_z = x_frames.new_tensor(0.0)
    loss_action_delta_h = x_frames.new_tensor(0.0)
    loss_rollout_kstep_z = x_frames.new_tensor(0.0)
    loss_rollout_kstep_h = x_frames.new_tensor(0.0)
    loss_rollout_kstep_p = x_frames.new_tensor(0.0)
    loss_rollout_recon_z = x_frames.new_tensor(0.0)
    loss_rollout_project_z = x_frames.new_tensor(0.0)
    loss_additivity_h = x_frames.new_tensor(0.0)
    loss_action_delta_p = x_frames.new_tensor(0.0)
    loss_additivity_p = x_frames.new_tensor(0.0)
    loss_mag_p = x_frames.new_tensor(0.0)

    # -------------------------------------------------------------------------
    # Calculate required inputs
    # -------------------------------------------------------------------------

    # NOTE: We may not actually need each of these inputs, but it majorly simplifies the
    #   conditionals, since some inputs are needed in multiple branches.

    encode_outputs = model.encode_sequence(x_frames)
    z_embeddings = encode_outputs["embeddings"]
    loss_jepa_raw, preds, h_preds, h_states = jepa_loss(
        model,
        encode_outputs,
        a_seq,
        use_z2h_init=weights.z2h > 0,
    )

    z_for_decoder = z_embeddings.detach() if cfg.detach_decoder else z_embeddings
    x_recon = decoder(z_for_decoder)

    seq_len = z_embeddings.shape[1]
    warmup_frames = max(model.cfg.warmup_frames_h, 0)

    # Warmup before applying hidden-state losses to avoid cold-start transients.
    start_frame = max(min(warmup_frames, seq_len - 1), 0)

    pose_obs, pose_pred, pose_deltas = _rollout_pose(model, h_states, a_seq)

    def _norm_stats(tensor: torch.Tensor) -> Tuple[float, float, float]:
        if tensor.numel() == 0:
            return 0.0, 0.0, 0.0
        norms = tensor.norm(dim=-1)
        return (
            norms.mean().item(),
            norms.std(unbiased=False).item(),
            norms.max().item(),
        )

    with torch.no_grad():
        z_norm_mean, z_norm_std, z_norm_max = _norm_stats(z_embeddings)
        h_norm_mean, h_norm_std, h_norm_max = _norm_stats(h_states)
        p_embeddings = pose_obs.detach()
        p_norm_mean, p_norm_std, p_norm_max = _norm_stats(p_embeddings)

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

    # h (Hidden) Hidden-state dynamics and cross-projection losses
    if weights.h2z > 0:
        loss_h2z = h2z_loss(model, z_embeddings, h_states, start_frame)
    if weights.z2h > 0:
        loss_z2h = z2h_loss(model, z_embeddings, h_states, start_frame)

    # h (Auxiliary)
    # p (Geometry)
    if weights.geometry_rank > 0:
        loss_geometry_rank, geometry_rank_accuracy, geometry_rank_pairs = geometry_rank_loss(model, h_states, cfg.geometry)

    dh_pred: Optional[torch.Tensor] = None
    if weights.action_delta_h > 0 or weights.additivity_h > 0:
        if h_states.shape[1] >= 2:
            h_curr = h_states[:, start_frame:-1]
            h_next = h_states[:, start_frame + 1 :]
            a_curr = a_seq[:, start_frame:-1]
            if h_curr.numel() > 0:
                dh_pred = model.h_action_delta_projector(h_curr, a_curr)
                dh_target = h_next - h_curr
                if weights.action_delta_h > 0:
                    loss_action_delta_h = F.mse_loss(dh_pred, dh_target)

    if (
        weights.action_delta_p > 0
        or weights.additivity_p > 0
        or weights.rollout_kstep_p > 0
        or weights.mag_p > 0
    ):
        if pose_obs.shape[1] >= 2:
            pose_curr = pose_pred[:, start_frame:-1]
            pose_next_obs = pose_obs[:, start_frame + 1 :]
            if pose_curr.numel() > 0:
                if weights.action_delta_p > 0:
                    loss_action_delta_p = F.mse_loss(
                        pose_pred[:, start_frame + 1 :],
                        pose_next_obs.detach(),
                    )
            if weights.additivity_p > 0 and pose_deltas.shape[1] >= start_frame + 2:
                delta_sum = pose_deltas[:, start_frame:-1] + pose_deltas[:, start_frame + 1 :]
                pose_target = pose_obs[:, start_frame + 2 :] - pose_obs[:, start_frame:-2]
                if delta_sum.numel() > 0 and pose_target.numel() > 0:
                    loss_additivity_p = F.mse_loss(delta_sum, pose_target.detach())
            if weights.rollout_kstep_p > 0 and pose_deltas.shape[1] >= 2:
                max_k = min(cfg.rollout_horizon, pose_obs.shape[1] - 1)
                if max_k >= 2:
                    delta_cumsum = pose_deltas.cumsum(dim=1)
                    k_losses: List[torch.Tensor] = []
                    for k in range(2, max_k + 1):
                        end = pose_obs.shape[1] - k
                        if end <= start_frame:
                            continue
                        delta_end = delta_cumsum[:, start_frame + k - 1 : end + k - 1]
                        if start_frame > 0:
                            delta_start = delta_cumsum[:, start_frame - 1 : end - 1]
                            delta_sum_k = delta_end - delta_start
                        else:
                            delta_sum_k = delta_end
                        pose_hat = pose_obs[:, start_frame:end] + delta_sum_k
                        pose_target_k = pose_obs[:, start_frame + k : end + k]
                        if pose_hat.numel() > 0:
                            k_losses.append(F.mse_loss(pose_hat, pose_target_k.detach()))
                    if k_losses:
                        loss_rollout_kstep_p = torch.stack(k_losses).mean()
            if weights.mag_p > 0 and pose_deltas.numel() > 0:
                loss_mag_p = pose_deltas.norm(dim=-1).mean()

    # Inverse dynamics
    if weights.inverse_dynamics_z > 0:
        assert z_embeddings.shape[1] >= 2
        action_logits_z = model.inverse_dynamics_z(z_embeddings[:, :-1], z_embeddings[:, 1:])
        loss_inverse_dynamics_z = INVERSE_DYNAMICS_LOSS(action_logits_z, a_seq[:, :-1])

    if weights.inverse_dynamics_h > 0:
        assert h_states.shape[1] >= 2
        action_logits_h = model.inverse_dynamics_h(h_states[:, :-1], h_states[:, 1:])
        loss_inverse_dynamics_h = INVERSE_DYNAMICS_LOSS(action_logits_h, a_seq[:, :-1])

    if weights.inverse_dynamics_p > 0:
        assert h_states.shape[1] >= 2
        # NOTE: inverse dynamics on s should not backprop into h/dynamics.
        p_for_inverse = pose_obs.detach()
        action_logits_p = model.inverse_dynamics_p(p_for_inverse[:, :-1], p_for_inverse[:, 1:])
        loss_inverse_dynamics_p = INVERSE_DYNAMICS_LOSS(action_logits_p, a_seq[:, :-1])

    if weights.action_delta_z > 0:
        assert z_embeddings.shape[1] >= 2
        delta_target = z_embeddings[:, 1:] - z_embeddings[:, :-1]
        delta_proto = model.z_action_delta_projector(a_seq[:, :-1])
        loss_action_delta_z = F.mse_loss(delta_target, delta_proto)

    if weights.rollout_kstep_z > 0:
        loss_rollout_kstep_z = rollout_z_loss(
            model,
            z_embeddings,
            a_seq,
            h_states,
            cfg.rollout_horizon,
            warmup_frames,
        )
    if weights.rollout_recon_z > 0:
        loss_rollout_recon_z = rollout_recon_z_loss(
            model,
            decoder,
            z_embeddings,
            a_seq,
            x_frames,
            h_states,
            cfg.rollout_horizon,
            warmup_frames,
        )
    if weights.rollout_project_z > 0:
        loss_rollout_project_z = rollout_project_z_loss(
            model,
            decoder,
            z_embeddings,
            a_seq,
            h_states,
            cfg.rollout_horizon,
            warmup_frames,
        )

    if weights.rollout_kstep_h > 0:
        loss_rollout_kstep_h = rollout_h_loss(
            model,
            z_embeddings,
            a_seq,
            h_states,
            cfg.rollout_horizon,
            warmup_frames,
        )

    if weights.additivity_h > 0 and dh_pred is not None:
        if dh_pred.shape[1] >= 2:
            dh_add_pred = dh_pred[:, :-1] + dh_pred[:, 1:]
            dh_add_target = h_states[:, start_frame + 2 :] - h_states[:, start_frame:-2]
            if dh_add_pred.numel() > 0:
                loss_additivity_h = F.mse_loss(dh_add_pred, dh_add_target)

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
        + weights.z2h * _scaled("loss_z2h", loss_z2h)
        + weights.geometry_rank * _scaled("loss_geometry_rank", loss_geometry_rank)
        + weights.recon * _scaled("loss_recon", loss_recon)
        + weights.recon_multi_gauss * _scaled("loss_recon_multi_gauss", loss_recon_multi_gauss)
        + weights.recon_multi_box * _scaled("loss_recon_multi_box", loss_recon_multi_box)
        + weights.recon_patch * _scaled("loss_recon_patch", loss_recon_patch)
        + weights.pixel_delta * _scaled("loss_pixel_delta", loss_pixel_delta)
        + weights.inverse_dynamics_z * _scaled("loss_inverse_dynamics_z", loss_inverse_dynamics_z)
        + weights.inverse_dynamics_h * _scaled("loss_inverse_dynamics_h", loss_inverse_dynamics_h)
        + weights.inverse_dynamics_p * _scaled("loss_inverse_dynamics_p", loss_inverse_dynamics_p)
        + weights.action_delta_z * _scaled("loss_action_delta_z", loss_action_delta_z)
        + weights.action_delta_h * _scaled("loss_action_delta_h", loss_action_delta_h)
        + weights.rollout_kstep_z * _scaled("loss_rollout_kstep_z", loss_rollout_kstep_z)
        + weights.rollout_recon_z * _scaled("loss_rollout_recon_z", loss_rollout_recon_z)
        + weights.rollout_project_z * _scaled("loss_rollout_project_z", loss_rollout_project_z)
        + weights.rollout_kstep_h * _scaled("loss_rollout_kstep_h", loss_rollout_kstep_h)
        + weights.rollout_kstep_p * _scaled("loss_rollout_kstep_p", loss_rollout_kstep_p)
        + weights.additivity_h * _scaled("loss_additivity_h", loss_additivity_h)
        + weights.mag_p * _scaled("loss_mag_p", loss_mag_p)
        + weights.action_delta_p * _scaled("loss_action_delta_p", loss_action_delta_p)
        + weights.additivity_p * _scaled("loss_additivity_p", loss_additivity_p)
    )

    world_grad_norm = 0.0
    decoder_grad_norm = 0.0
    if for_training and optimizer is not None:
        optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
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
        "loss_z2h": loss_z2h.item(),
        "loss_geometry_rank": loss_geometry_rank.item(),
        "geometry_rank_accuracy": geometry_rank_accuracy.item(),
        "geometry_rank_pairs": geometry_rank_pairs.item(),
        "loss_inverse_dynamics_z": loss_inverse_dynamics_z.item(),
        "loss_inverse_dynamics_h": loss_inverse_dynamics_h.item(),
        "loss_inverse_dynamics_p": loss_inverse_dynamics_p.item(),
        "loss_action_delta_z": loss_action_delta_z.item(),
        "loss_action_delta_h": loss_action_delta_h.item(),
        "loss_rollout_kstep_z": loss_rollout_kstep_z.item(),
        "loss_rollout_recon_z": loss_rollout_recon_z.item(),
        "loss_rollout_project_z": loss_rollout_project_z.item(),
        "loss_rollout_kstep_h": loss_rollout_kstep_h.item(),
        "loss_rollout_kstep_p": loss_rollout_kstep_p.item(),
        "loss_additivity_h": loss_additivity_h.item(),
        "loss_mag_p": loss_mag_p.item(),
        "loss_action_delta_p": loss_action_delta_p.item(),
        "loss_additivity_p": loss_additivity_p.item(),
        "z_norm_mean": z_norm_mean,
        "z_norm_std": z_norm_std,
        "z_norm_max": z_norm_max,
        "h_norm_mean": h_norm_mean,
        "h_norm_std": h_norm_std,
        "h_norm_max": h_norm_max,
        "p_norm_mean": p_norm_mean,
        "p_norm_std": p_norm_std,
        "p_norm_max": p_norm_max,
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


def _seed_everything(seed: int) -> Tuple[int, random.Random]:
    """Seed Python, NumPy, and torch RNGs and return a dedicated Python RNG."""
    seed_value = int(seed)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    return seed_value, random.Random(seed_value)


# ------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------


@dataclass
class SpikeMetricState:
    mean: float = 0.0
    var: float = 0.0
    count: int = 0


class SpikeTracker:
    def __init__(self, cfg: SpikeDiagnosticsConfig) -> None:
        self.cfg = cfg
        self.states: Dict[str, SpikeMetricState] = {}
        self.spike_count = 0

    def detect(self, metrics: Dict[str, float]) -> List[Dict[str, float]]:
        events: List[Dict[str, float]] = []
        for metric in self.cfg.metrics:
            if metric not in metrics:
                continue
            value = float(metrics[metric])
            state = self.states.setdefault(metric, SpikeMetricState())
            if state.count >= self.cfg.warmup_steps:
                std = math.sqrt(max(state.var, 1e-12))
                z = 0.0 if std <= 0 else (value - state.mean) / std
                ratio = value / state.mean if state.mean > self.cfg.min_reference else float("inf")
                is_spike = (
                    value > state.mean + self.cfg.z_threshold * std
                    or (state.mean > self.cfg.min_reference and value > state.mean * self.cfg.ratio_threshold)
                )
                if is_spike and self.spike_count < self.cfg.max_spikes:
                    events.append(
                        {
                            "metric": metric,
                            "value": value,
                            "mean": state.mean,
                            "std": std,
                            "z": z,
                            "ratio": ratio,
                        }
                    )
            self._update_state(state, value)
        if events:
            self.spike_count += len(events)
        return events

    def _update_state(self, state: SpikeMetricState, value: float) -> None:
        if state.count == 0:
            state.mean = value
            state.var = 0.0
            state.count = 1
            return
        decay = self.cfg.ema_decay
        prev_mean = state.mean
        state.mean = decay * state.mean + (1.0 - decay) * value
        delta = value - prev_mean
        state.var = decay * state.var + (1.0 - decay) * delta * delta
        state.count += 1


def _write_spike_event_row(path: Path, step: int, event: Dict[str, float]) -> None:
    header = ["step", "metric", "value", "mean", "std", "z", "ratio"]
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(header)
        writer.writerow(
            [
                step,
                event["metric"],
                f"{event['value']:.6f}",
                f"{event['mean']:.6f}",
                f"{event['std']:.6f}",
                f"{event['z']:.6f}",
                f"{event['ratio']:.6f}",
            ]
        )


def _compute_spike_batch_stats(frames: torch.Tensor, actions: torch.Tensor) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    frames_f = frames.float()
    stats["frame_mean"] = float(frames_f.mean().item())
    stats["frame_std"] = float(frames_f.std(unbiased=False).item())
    stats["frame_min"] = float(frames_f.min().item())
    stats["frame_max"] = float(frames_f.max().item())

    if frames_f.shape[1] >= 2:
        deltas = (frames_f[:, 1:] - frames_f[:, :-1]).abs().mean(dim=(2, 3, 4))
        deltas_flat = deltas.reshape(-1)
        stats["delta_mean"] = float(deltas_flat.mean().item())
        stats["delta_std"] = float(deltas_flat.std(unbiased=False).item())
        stats["delta_max"] = float(deltas_flat.max().item())
        stats["delta_p90"] = float(torch.quantile(deltas_flat, 0.9).item())
        stats["delta_zero_frac"] = float((deltas_flat < 1e-3).float().mean().item())
    else:
        stats["delta_mean"] = 0.0
        stats["delta_std"] = 0.0
        stats["delta_max"] = 0.0
        stats["delta_p90"] = 0.0
        stats["delta_zero_frac"] = 0.0

    action_ids = compress_actions_to_ids(actions).detach().cpu().numpy().reshape(-1)
    action_dim = int(actions.shape[-1])
    counts: Dict[int, int] = {}
    for aid in action_ids.tolist():
        counts[int(aid)] = counts.get(int(aid), 0) + 1
    total = max(len(action_ids), 1)
    stats["action_distribution"] = [
        {
            "action_id": aid,
            "label": decode_action_id(aid, action_dim),
            "count": count,
            "fraction": count / total,
        }
        for aid, count in sorted(counts.items())
    ]

    per_seq_stats: List[Dict[str, Any]] = []
    action_ids_seq = compress_actions_to_ids(actions).detach().cpu().numpy()
    for seq_idx, seq_actions in enumerate(action_ids_seq):
        seq_list = seq_actions.tolist()
        if isinstance(seq_list, int):
            seq_list = [seq_list]
        seq_actions = [int(a) for a in seq_list]
        unique_actions = len(set(seq_actions))
        repeats = sum(1 for i in range(1, len(seq_actions)) if seq_actions[i] == seq_actions[i - 1])
        per_seq_stats.append(
            {
                "batch_index": seq_idx,
                "unique_actions": unique_actions,
                "repeat_fraction": repeats / max(len(seq_actions) - 1, 1),
                "noop_fraction": seq_actions.count(0) / max(len(seq_actions), 1),
            }
        )
    stats["per_sequence"] = per_seq_stats
    return stats


def _summarize_action_distribution(actions: torch.Tensor) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    action_ids = compress_actions_to_ids(actions).detach().cpu().numpy().reshape(-1)
    action_dim = int(actions.shape[-1])
    counts: Dict[int, int] = {}
    for aid in action_ids.tolist():
        counts[int(aid)] = counts.get(int(aid), 0) + 1
    total = max(len(action_ids), 1)
    distribution = [
        {
            "action_id": aid,
            "label": decode_action_id(aid, action_dim),
            "count": count,
            "fraction": count / total,
        }
        for aid, count in sorted(counts.items())
    ]
    action_ids_seq = compress_actions_to_ids(actions).detach().cpu().numpy()
    if action_ids_seq.ndim == 1:
        action_ids_seq = action_ids_seq.reshape(actions.shape[0], actions.shape[1])
    unique_actions: List[int] = []
    repeat_fracs: List[float] = []
    noop_fracs: List[float] = []
    for seq_actions in action_ids_seq:
        seq_list = [int(a) for a in seq_actions.tolist()]
        unique_actions.append(len(set(seq_list)))
        repeats = sum(1 for i in range(1, len(seq_list)) if seq_list[i] == seq_list[i - 1])
        repeat_fracs.append(repeats / max(len(seq_list) - 1, 1))
        noop_fracs.append(seq_list.count(0) / max(len(seq_list), 1))
    summary = {
        "mean_unique_actions": float(np.mean(unique_actions)) if unique_actions else 0.0,
        "mean_repeat_fraction": float(np.mean(repeat_fracs)) if repeat_fracs else 0.0,
        "mean_noop_fraction": float(np.mean(noop_fracs)) if noop_fracs else 0.0,
    }
    return distribution, summary


def _write_recon_spike_action_overlay_row(
    path: Path,
    step: int,
    actions: torch.Tensor,
    recon_value: float,
    recon_event: Optional[Dict[str, float]],
) -> None:
    distribution, summary = _summarize_action_distribution(actions)
    top_action = max(distribution, key=lambda item: item["fraction"]) if distribution else None
    header = [
        "step",
        "recon_spike",
        "recon_value",
        "recon_mean",
        "recon_std",
        "recon_z",
        "recon_ratio",
        "top_action_id",
        "top_action_label",
        "top_action_fraction",
        "mean_unique_actions",
        "mean_repeat_fraction",
        "mean_noop_fraction",
        "action_distribution_json",
    ]
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(header)
        writer.writerow(
            [
                step,
                1 if recon_event is not None else 0,
                f"{recon_value:.6f}",
                f"{recon_event['mean']:.6f}" if recon_event is not None else "",
                f"{recon_event['std']:.6f}" if recon_event is not None else "",
                f"{recon_event['z']:.6f}" if recon_event is not None else "",
                f"{recon_event['ratio']:.6f}" if recon_event is not None else "",
                top_action["action_id"] if top_action is not None else "",
                top_action["label"] if top_action is not None else "",
                f"{top_action['fraction']:.6f}" if top_action is not None else "",
                f"{summary['mean_unique_actions']:.6f}",
                f"{summary['mean_repeat_fraction']:.6f}",
                f"{summary['mean_noop_fraction']:.6f}",
                json.dumps(distribution, sort_keys=True),
            ]
        )


def _write_spike_batch(
    spike_dir: Path,
    global_step: int,
    event: Dict[str, float],
    batch: Tuple[torch.Tensor, torch.Tensor, List[List[str]], torch.Tensor],
    metrics: Dict[str, float],
    cfg: SpikeDiagnosticsConfig,
) -> None:
    frames = batch[0].detach().cpu()
    actions = batch[1].detach().cpu()
    paths = batch[2]
    indices = batch[3].detach().cpu().tolist()

    event_dir = spike_dir / f"spike_{global_step:07d}_{event['metric']}"
    event_dir.mkdir(parents=True, exist_ok=True)

    stats = _compute_spike_batch_stats(frames, actions)
    payload = {
        "step": global_step,
        "metric": event["metric"],
        "value": event["value"],
        "mean": event["mean"],
        "std": event["std"],
        "z": event["z"],
        "ratio": event["ratio"],
        "metrics": metrics,
        "batch_indices": indices,
    }

    (event_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (event_dir / "batch_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))

    with (event_dir / "batch_frames.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["batch_index", "time_index", "frame_path"])
        for b_idx, seq_paths in enumerate(paths):
            for t_idx, frame_path in enumerate(seq_paths):
                writer.writerow([b_idx, t_idx, frame_path])

    with (event_dir / "batch_actions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["batch_index", "time_index", "action_id", "action_label"]
        header.extend([f"action_{i}" for i in range(actions.shape[-1])])
        writer.writerow(header)
        action_ids = compress_actions_to_ids(actions).detach().cpu().numpy()
        if action_ids.ndim == 1:
            action_ids = action_ids.reshape(actions.shape[0], actions.shape[1])
        for b_idx in range(actions.shape[0]):
            for t_idx in range(actions.shape[1]):
                aid = int(action_ids[b_idx, t_idx])
                label = decode_action_id(aid, actions.shape[-1])
                writer.writerow([b_idx, t_idx, aid, label, *actions[b_idx, t_idx].tolist()])

    if cfg.save_visuals:
        rows = min(4, frames.shape[0])
        save_input_batch_visualization(
            event_dir / "batch_inputs.png",
            frames,
            actions,
            rows,
            recon=None,
            include_deltas=True,
        )


def log_metrics(
    step: int,
    metrics: Dict[str, float],
    weights: LossWeights,
    samples_per_sec: float,
    elapsed_seconds: float,
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
    if weights.z2h <= 0:
        filtered.pop("loss_z2h", None)
    if weights.geometry_rank <= 0:
        filtered.pop("loss_geometry_rank", None)
        filtered.pop("geometry_rank_accuracy", None)
        filtered.pop("geometry_rank_pairs", None)
    if weights.inverse_dynamics_z <= 0:
        filtered.pop("loss_inverse_dynamics_z", None)
    if weights.inverse_dynamics_h <= 0:
        filtered.pop("loss_inverse_dynamics_h", None)
    if weights.inverse_dynamics_p <= 0:
        filtered.pop("loss_inverse_dynamics_p", None)
    if weights.action_delta_z <= 0:
        filtered.pop("loss_action_delta_z", None)
    if weights.action_delta_h <= 0:
        filtered.pop("loss_action_delta_h", None)
    if weights.rollout_kstep_z <= 0:
        filtered.pop("loss_rollout_kstep_z", None)
    if weights.rollout_recon_z <= 0:
        filtered.pop("loss_rollout_recon_z", None)
    if weights.rollout_project_z <= 0:
        filtered.pop("loss_rollout_project_z", None)
    if weights.rollout_kstep_h <= 0:
        filtered.pop("loss_rollout_kstep_h", None)
    if weights.rollout_kstep_p <= 0:
        filtered.pop("loss_rollout_kstep_p", None)
    if weights.additivity_h <= 0:
        filtered.pop("loss_additivity_h", None)
    if weights.mag_p <= 0:
        filtered.pop("loss_mag_p", None)
    if weights.action_delta_p <= 0:
        filtered.pop("loss_action_delta_p", None)
    if weights.additivity_p <= 0:
        filtered.pop("loss_additivity_p", None)
    pretty = ", ".join(f"{k}: {v:.4f}" for k, v in filtered.items())
    summary_parts: List[str] = []
    if pretty:
        summary_parts.append(pretty)
    summary_parts.append(f"{samples_per_sec:.1f} samples/s")
    if elapsed_seconds >= 0:
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
    "loss_z2h",
    "loss_geometry_rank",
    "geometry_rank_accuracy",
    "geometry_rank_pairs",
    "loss_inverse_dynamics_z",
    "loss_inverse_dynamics_h",
    "loss_inverse_dynamics_p",
    "loss_action_delta_z",
    "loss_action_delta_h",
    "loss_rollout_kstep_z",
    "loss_rollout_recon_z",
    "loss_rollout_project_z",
    "loss_rollout_kstep_h",
    "loss_rollout_kstep_p",
    "loss_additivity_h",
    "loss_mag_p",
    "loss_action_delta_p",
    "loss_additivity_p",
    "z_norm_mean",
    "z_norm_std",
    "z_norm_max",
    "h_norm_mean",
    "h_norm_std",
    "h_norm_max",
    "p_norm_mean",
    "p_norm_std",
    "p_norm_max",
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
    z2h: List[float] = field(default_factory=list)
    geometry_rank: List[float] = field(default_factory=list)
    geometry_rank_accuracy: List[float] = field(default_factory=list)
    geometry_rank_pairs: List[float] = field(default_factory=list)
    inverse_dynamics_z: List[float] = field(default_factory=list)
    inverse_dynamics_h: List[float] = field(default_factory=list)
    inverse_dynamics_p: List[float] = field(default_factory=list)
    action_delta_z: List[float] = field(default_factory=list)
    action_delta_h: List[float] = field(default_factory=list)
    rollout_kstep_z: List[float] = field(default_factory=list)
    rollout_recon_z: List[float] = field(default_factory=list)
    rollout_project_z: List[float] = field(default_factory=list)
    rollout_kstep_h: List[float] = field(default_factory=list)
    rollout_kstep_p: List[float] = field(default_factory=list)
    additivity_h: List[float] = field(default_factory=list)
    mag_p: List[float] = field(default_factory=list)
    action_delta_p: List[float] = field(default_factory=list)
    additivity_p: List[float] = field(default_factory=list)
    z_norm_mean: List[float] = field(default_factory=list)
    z_norm_std: List[float] = field(default_factory=list)
    z_norm_max: List[float] = field(default_factory=list)
    h_norm_mean: List[float] = field(default_factory=list)
    h_norm_std: List[float] = field(default_factory=list)
    h_norm_max: List[float] = field(default_factory=list)
    p_norm_mean: List[float] = field(default_factory=list)
    p_norm_std: List[float] = field(default_factory=list)
    p_norm_max: List[float] = field(default_factory=list)
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
        self.z2h.append(metrics["loss_z2h"])
        self.geometry_rank.append(metrics.get("loss_geometry_rank", 0.0))
        self.geometry_rank_accuracy.append(metrics.get("geometry_rank_accuracy", 0.0))
        self.geometry_rank_pairs.append(metrics.get("geometry_rank_pairs", 0.0))
        self.inverse_dynamics_z.append(metrics.get("loss_inverse_dynamics_z", 0.0))
        self.inverse_dynamics_h.append(metrics.get("loss_inverse_dynamics_h", 0.0))
        self.inverse_dynamics_p.append(metrics.get("loss_inverse_dynamics_p", 0.0))
        self.action_delta_z.append(metrics.get("loss_action_delta_z", 0.0))
        self.action_delta_h.append(metrics.get("loss_action_delta_h", 0.0))
        self.rollout_kstep_z.append(metrics.get("loss_rollout_kstep_z", 0.0))
        self.rollout_recon_z.append(metrics.get("loss_rollout_recon_z", 0.0))
        self.rollout_project_z.append(metrics.get("loss_rollout_project_z", 0.0))
        self.rollout_kstep_h.append(metrics.get("loss_rollout_kstep_h", 0.0))
        self.rollout_kstep_p.append(metrics.get("loss_rollout_kstep_p", 0.0))
        self.additivity_h.append(metrics.get("loss_additivity_h", 0.0))
        self.mag_p.append(metrics.get("loss_mag_p", 0.0))
        self.action_delta_p.append(metrics.get("loss_action_delta_p", 0.0))
        self.additivity_p.append(metrics.get("loss_additivity_p", 0.0))
        self.z_norm_mean.append(metrics.get("z_norm_mean", 0.0))
        self.z_norm_std.append(metrics.get("z_norm_std", 0.0))
        self.z_norm_max.append(metrics.get("z_norm_max", 0.0))
        self.h_norm_mean.append(metrics.get("h_norm_mean", 0.0))
        self.h_norm_std.append(metrics.get("h_norm_std", 0.0))
        self.h_norm_max.append(metrics.get("h_norm_max", 0.0))
        self.p_norm_mean.append(metrics.get("p_norm_mean", 0.0))
        self.p_norm_std.append(metrics.get("p_norm_std", 0.0))
        self.p_norm_max.append(metrics.get("p_norm_max", 0.0))
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
            history.z2h,
            history.geometry_rank,
            history.geometry_rank_accuracy,
            history.geometry_rank_pairs,
            history.inverse_dynamics_z,
            history.inverse_dynamics_h,
            history.inverse_dynamics_p,
            history.action_delta_z,
            history.action_delta_h,
            history.rollout_kstep_z,
            history.rollout_recon_z,
            history.rollout_project_z,
            history.rollout_kstep_h,
            history.rollout_kstep_p,
            history.additivity_h,
            history.mag_p,
            history.action_delta_p,
            history.additivity_p,
            history.z_norm_mean,
            history.z_norm_std,
            history.z_norm_max,
            history.h_norm_mean,
            history.h_norm_std,
            history.h_norm_max,
            history.p_norm_mean,
            history.p_norm_std,
            history.p_norm_max,
            history.grad_world,
            history.grad_decoder,
        ):
            writer.writerow(row)


def plot_loss_curves(history: LossHistory, out_dir: Path) -> None:
    if len(history) == 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=figsize_for_grid(2, 2), constrained_layout=True)
    default_cycle = plt.rcParams.get("axes.prop_cycle")
    color_cycle = default_cycle.by_key().get("color", []) if default_cycle is not None else []

    def _color(idx: int) -> str:
        return color_cycle[idx % len(color_cycle)]

    color_map = {
        "world": _color(0),
        "jepa": _color(1),
        "sigreg": _color(2),
        "recon": _color(3),
        "geometry_rank_accuracy": _color(17),
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
    if any(val != 0.0 for val in history.geometry_rank_accuracy):
        plt.plot(
            history.steps,
            history.geometry_rank_accuracy,
            label="geometry_rank_accuracy",
            color=color_map["geometry_rank_accuracy"],
        )
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
    plt.savefig(out_dir / "loss_curves.png", dpi=DEFAULT_DPI)
    plt.close()

    has_rank_acc = any(val != 0.0 for val in history.geometry_rank_accuracy)
    has_rank_loss = any(val != 0.0 for val in history.geometry_rank)
    if has_rank_acc or has_rank_loss:
        fig, ax1 = plt.subplots(figsize=figsize_for_grid(1, 1), constrained_layout=True)
        if has_rank_acc:
            ax1.plot(history.steps, history.geometry_rank_accuracy, label="geometry_rank_accuracy", color=_color(3))
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Ranking accuracy")
        ax1.set_ylim(0.0, 1.0)
        ax1.grid(True, alpha=0.3)
        if has_rank_loss:
            ax2 = ax1.twinx()
            ax2.plot(history.steps, history.geometry_rank, label="loss_geometry_rank", color=_color(4))
            ax2.set_ylabel("Ranking loss")
            ax2.set_yscale("log")
        pair_values = [val for val in history.geometry_rank_pairs if val > 0]
        pair_count = int(round(pair_values[-1])) if pair_values else 0
        title = f"Geometry Ranking Accuracy (pairs/batch: {pair_count})" if pair_count else "Geometry Ranking Accuracy"
        fig.suptitle(title, fontsize=11)
        fig.savefig(out_dir / "ranking_accuracy.png", dpi=DEFAULT_DPI)
        plt.close(fig)


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
    batch_paths: List[List[str]],
    sample_idx: int,
    start_idx: int,
    length: int,
) -> List[str]:
    if sample_idx >= len(batch_paths):
        raise AssertionError("Frame paths are required for visualization labels.")
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


def _build_fixed_vis_batch(
    dataloader: DataLoader,
    vis_rows: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, List[List[str]]], VisualizationSelection]:
    sample_batch = next(iter(dataloader))
    frames_cpu, actions_cpu = sample_batch[0], sample_batch[1]
    if vis_rows <= 0:
        raise AssertionError("vis.rows must be positive to build a visualization selection.")
    if frames_cpu.shape[0] == 0:
        raise AssertionError("Visualization requires at least one sequence in the dataset.")
    if frames_cpu.shape[1] < 2:
        raise AssertionError("Visualization requires sequences with at least two frames.")
    if len(sample_batch) <= 2:
        raise AssertionError("Visualization requires frame paths for labeling.")
    batch_size, seq_len = frames_cpu.shape[0], frames_cpu.shape[1]
    num_rows = min(vis_rows, batch_size)
    row_indices = torch.arange(num_rows, dtype=torch.long)
    time_indices = (torch.arange(num_rows, dtype=torch.long) % (seq_len - 1)) + 1
    fixed_paths = sample_batch[2]
    fixed_batch_cpu: Tuple[torch.Tensor, torch.Tensor, List[List[str]]] = (
        frames_cpu.clone(),
        actions_cpu.clone(),
        [list(paths) for paths in fixed_paths],
    )
    return fixed_batch_cpu, VisualizationSelection(row_indices=row_indices, time_indices=time_indices)


def _build_embedding_batch(
    dataset: TrajectorySequenceDataset,
    sample_count: int,
    generator: torch.Generator = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
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
    if len(embed_batch) <= 2:
        raise AssertionError("Embedding batch must include trajectory paths for downstream diagnostics.")
    paths = [list(p) for p in embed_batch[2]]
    return (embed_batch[0].clone(), embed_batch[1].clone(), paths)


@dataclass
class GraphDiagnosticsBatch:
    graph_embeddings: torch.Tensor
    graph_preds: torch.Tensor
    graph_h_preds: torch.Tensor
    graph_h_states: torch.Tensor
    ema_embeddings: Optional[torch.Tensor]
    ema_h_states: Optional[torch.Tensor]
    next_index: torch.Tensor
    next2_index: torch.Tensor
    chunk_ids: torch.Tensor


def _prepare_graph_diagnostics(
    *,
    graph_frames: torch.Tensor,
    graph_actions: torch.Tensor,
    model: JEPAWorldModel,
    ema_model: Optional[JEPAWorldModel],
    graph_cfg: GraphDiagnosticsConfig,
    device: torch.device,
    use_z2h_init: bool,
) -> "GraphDiagnosticsBatch":
    if graph_frames.shape[1] < 3:
        raise AssertionError("Graph diagnostics require sequences with at least three frames.")
    assert torch.is_grad_enabled()
    with torch.no_grad():
        graph_frames_device = graph_frames.to(device)
        graph_actions_device = graph_actions.to(device)
        graph_embeddings = model.encode_sequence(graph_frames_device)["embeddings"]
        graph_preds, graph_h_preds, graph_h_states = rollout_teacher_forced(
            model,
            graph_embeddings,
            graph_actions_device,
            use_z2h_init=use_z2h_init,
        )
        ema_embeddings: Optional[torch.Tensor] = None
        ema_h_states: Optional[torch.Tensor] = None
        if graph_cfg.use_ema_targets and ema_model is not None:
            ema_embeddings = ema_model.encode_sequence(graph_frames_device)["embeddings"]
            _, _, ema_h_states = rollout_teacher_forced(
                ema_model,
                ema_embeddings,
                graph_actions_device,
                use_z2h_init=use_z2h_init,
            )
    assert torch.is_grad_enabled()
    next_index, next2_index, chunk_ids = build_graph_diag_indices(graph_frames_device)
    return GraphDiagnosticsBatch(
        graph_embeddings=graph_embeddings,
        graph_preds=graph_preds,
        graph_h_preds=graph_h_preds,
        graph_h_states=graph_h_states,
        ema_embeddings=ema_embeddings,
        ema_h_states=ema_h_states,
        next_index=next_index,
        next2_index=next2_index,
        chunk_ids=chunk_ids,
    )


def _run_graph_diag(
    *,
    embedding_kind: str,
    graph_cfg: GraphDiagnosticsConfig,
    model: JEPAWorldModel,
    ema_model: Optional[JEPAWorldModel],
    graph_embeddings: torch.Tensor,
    graph_preds: torch.Tensor,
    graph_h_preds: torch.Tensor,
    graph_h_states: torch.Tensor,
    ema_embeddings: Optional[torch.Tensor],
    ema_h_states: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if embedding_kind == "p":
        p_targets = model.h2p(graph_h_states)
        p_hat_full = torch.cat(
            [model.h2p(graph_h_preds), model.h2p(graph_h_states[:, -1:, :])],
            dim=1,
        )
        if graph_cfg.use_ema_targets and ema_model is not None and ema_h_states is not None:
            targets = ema_model.h2p(ema_h_states)
        else:
            targets = p_targets
        z_flat = p_targets.reshape(-1, p_targets.shape[-1])
        target_flat = targets.reshape(-1, targets.shape[-1])
        zhat_full = p_hat_full.reshape(-1, p_hat_full.shape[-1])
    elif embedding_kind == "h":
        targets = ema_h_states if graph_cfg.use_ema_targets and ema_h_states is not None else graph_h_states
        z_flat = graph_h_states.reshape(-1, graph_h_states.shape[-1])
        target_flat = targets.reshape(-1, targets.shape[-1])
        zhat_full = torch.cat(
            [graph_h_preds, graph_h_states[:, -1:, :]],
            dim=1,
        ).reshape(-1, graph_h_states.shape[-1])
    else:
        targets = ema_embeddings if graph_cfg.use_ema_targets and ema_embeddings is not None else graph_embeddings
        z_flat = graph_embeddings.reshape(-1, graph_embeddings.shape[-1])
        target_flat = targets.reshape(-1, targets.shape[-1])
        zhat_full = torch.cat(
            [graph_preds, graph_embeddings[:, -1:, :]],
            dim=1,
        ).reshape(-1, graph_embeddings.shape[-1])

    if graph_cfg.normalize_latents:
        z_flat = F.normalize(z_flat, dim=-1)
        target_flat = F.normalize(target_flat, dim=-1)
        zhat_full = F.normalize(zhat_full, dim=-1)

    queries = zhat_full if graph_cfg.use_predictor_scores else z_flat
    return queries, target_flat, zhat_full


def _prepare_graph_diag_dataset(
    *,
    cfg: TrainConfig,
    model_cfg: ModelConfig,
    dataset_action_dim: int,
    train_trajs: List[str],
    generator: torch.Generator,
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor, List[List[str]]],
]:
    if not cfg.graph_diagnostics.enabled:
        raise AssertionError("Graph diagnostics requested but graph_diagnostics.enabled is false.")
    graph_seq_len = max(cfg.graph_diagnostics.chunk_len, 3)
    graph_diag_dataset = TrajectorySequenceDataset(
        root=cfg.data_root,
        seq_len=graph_seq_len,
        image_hw=(model_cfg.image_size, model_cfg.image_size),
        max_traj=None,
        included_trajectories=train_trajs,
    )
    graph_action_dim = graph_diag_dataset.action_dim
    if graph_action_dim != dataset_action_dim:
        raise AssertionError(
            f"Graph diagnostics action_dim {graph_action_dim} does not match train action_dim {dataset_action_dim}"
        )
    graph_diag_batch_cpu = _build_embedding_batch(
        graph_diag_dataset,
        cfg.graph_diagnostics.sample_chunks,
        generator=generator,
    )
    return graph_diag_batch_cpu


def _build_visualization_sequences(
    *,
    batch_cpu: Tuple[torch.Tensor, torch.Tensor, List[List[str]]],
    selection: Optional[VisualizationSelection],
    model: JEPAWorldModel,
    decoder: VisualizationDecoder,
    device: torch.device,
    vis_cfg: VisConfig,
    vis_selection_generator: torch.Generator,
    use_z2h_init: bool,
) -> Tuple[List[VisualizationSequence], str]:
    vis_frames = batch_cpu[0].to(device)
    vis_actions = batch_cpu[1].to(device)
    frame_paths = batch_cpu[2]
    assert torch.is_grad_enabled()
    with torch.no_grad():
        vis_embeddings = model.encode_sequence(vis_frames)["embeddings"]
        decoded_frames = decoder(vis_embeddings)
    assert torch.is_grad_enabled()
    if vis_frames.shape[0] == 0:
        raise ValueError("Visualization batch must include at least one sequence.")
    if vis_frames.shape[1] < 2:
        raise ValueError("Visualization batch must include at least two frames.")
    batch_size = vis_frames.shape[0]
    min_start = 0
    target_window = max(2, vis_cfg.rollout + 1)
    if vis_cfg.columns is not None:
        target_window = max(target_window, max(2, vis_cfg.columns))
    max_window = min(target_window, vis_frames.shape[1] - min_start)
    if max_window < 2:
        raise ValueError("Visualization window must be at least two frames wide.")
    max_start = max(min_start, vis_frames.shape[1] - max_window)
    if selection is not None and selection.row_indices.numel() > 0:
        num_rows = min(vis_cfg.rows, selection.row_indices.numel())
        row_indices = selection.row_indices[:num_rows].to(device=vis_frames.device)
        base_starts = selection.time_indices[:num_rows].to(device=vis_frames.device)
    else:
        num_rows = min(vis_cfg.rows, batch_size)
        row_indices = torch.randperm(batch_size, generator=vis_selection_generator, device=vis_frames.device)[:num_rows]
        base_starts = torch.randint(
            min_start,
            max_start + 1,
            (num_rows,),
            device=vis_frames.device,
            generator=vis_selection_generator,
        )
    action_texts: List[List[str]] = []
    rollout_frames: List[List[Optional[torch.Tensor]]] = []
    reencoded_frames: List[List[Optional[torch.Tensor]]] = []
    kept_row_indices: List[torch.Tensor] = []
    kept_start_indices: List[int] = []
    debug_lines: List[str] = []
    for row_offset, idx in enumerate(row_indices):
        start_idx = int(base_starts[row_offset].item())
        start_idx = max(min_start, min(start_idx, max_start))
        gt_slice = vis_frames[idx, start_idx : start_idx + max_window]
        if gt_slice.shape[0] < max_window:
            continue
        row_actions: List[str] = []
        for offset in range(max_window):
            action_idx = min(start_idx + offset, vis_actions.shape[1] - 1)
            row_actions.append(describe_action_tensor(vis_actions[idx, action_idx]))
        row_rollout: List[Optional[torch.Tensor]] = [None for _ in range(max_window)]
        row_reencoded: List[Optional[torch.Tensor]] = [None for _ in range(max_window)]
        assert torch.is_grad_enabled()
        with torch.no_grad():
            current_embed = vis_embeddings[idx, start_idx].unsqueeze(0)
            if use_z2h_init:
                current_hidden = model.z_to_h(current_embed.detach())
            else:
                current_hidden = current_embed.new_zeros(1, model.state_dim)
            prev_pred_frame = decoded_frames[idx, start_idx].detach()
            current_frame = prev_pred_frame
            for step in range(1, max_window):
                action = vis_actions[idx, start_idx + step - 1].unsqueeze(0)
                prev_embed = current_embed
                h_next = model.predictor(current_embed, current_hidden, action)
                next_embed = model.h_to_z(h_next)
                decoded_next = decoder(next_embed)[0]
                current_frame = decoded_next.clamp(0, 1)
                row_rollout[step] = current_frame.detach().cpu()
                reenc_embed = model.encoder(current_frame.unsqueeze(0))
                row_reencoded[step] = decoder(reenc_embed)[0].clamp(0, 1).detach().cpu()
                if vis_cfg.log_deltas and row_offset < 2:
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
                current_hidden = h_next
        assert torch.is_grad_enabled()
        action_texts.append(row_actions)
        rollout_frames.append(row_rollout)
        reencoded_frames.append(row_reencoded)
        kept_row_indices.append(idx)
        kept_start_indices.append(start_idx)
    if not kept_row_indices:
        raise AssertionError("Failed to build any visualization sequences.")
    kept_rows = torch.stack(kept_row_indices)
    kept_starts = torch.tensor(kept_start_indices, device=vis_frames.device)
    items = render_rollout_batch(
        vis_frames=vis_frames,
        decoded_frames=decoded_frames,
        row_indices=kept_rows,
        start_indices=kept_starts,
        max_window=max_window,
        action_texts=action_texts,
        rollout_frames=rollout_frames,
        reencoded_frames=reencoded_frames,
    )
    if debug_lines:
        print("\n".join(debug_lines))
    labels: List[List[str]] = []
    gradients: List[List[Optional[np.ndarray]]] = []
    for item in items:
        labels.append(
            _extract_frame_labels(
                frame_paths,
                item.row_index,
                item.start_idx,
                len(item.ground_truth),
            )
        )
        item_gradients: List[Optional[np.ndarray]] = [None for _ in range(len(item.rollout))]
        for step in range(1, len(item.rollout)):
            current_frame = item.rollout[step]
            if current_frame is None:
                continue
            target_frame = item.ground_truth[step]
            if vis_cfg.gradient_norms:
                item_gradients[step] = _prediction_gradient_heatmap(current_frame, target_frame)
            else:
                item_gradients[step] = _loss_to_heatmap(target_frame, current_frame)
        gradients.append(item_gradients)
    return _render_visualization_batch(
        items=items,
        labels=labels,
        gradients=gradients,
        show_gradients=vis_cfg.gradient_norms,
    )


def _build_off_manifold_batch(
    *,
    data_root: Path,
    train_trajs: Sequence[str],
    seq_len: int,
    image_hw: Tuple[int, int],
    sample_count: int,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
    off_dataset = TrajectorySequenceDataset(
        root=data_root,
        seq_len=seq_len,
        image_hw=image_hw,
        max_traj=None,
        included_trajectories=train_trajs,
    )
    return _build_embedding_batch(off_dataset, sample_count, generator=generator)


def _compute_off_manifold_errors(
    *,
    model: JEPAWorldModel,
    decoder: VisualizationDecoder,
    batch_cpu: Tuple[torch.Tensor, torch.Tensor, List[List[str]]],
    device: torch.device,
    rollout_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    frames = batch_cpu[0].to(device)
    actions = batch_cpu[1].to(device)
    if frames.shape[0] == 0:
        raise AssertionError("Off-manifold batch requires at least one sequence.")
    if frames.shape[1] < rollout_steps + 1:
        raise AssertionError("Off-manifold batch does not include enough frames for the requested rollout.")
    if actions.shape[1] < rollout_steps:
        raise AssertionError("Off-manifold batch does not include enough actions for the requested rollout.")
    start_frames = frames[:, 0]
    action_seq = actions[:, :rollout_steps]
    z_t = model.encoder(start_frames)
    h_t = z_t.new_zeros(z_t.shape[0], model.state_dim)
    step_indices: List[np.ndarray] = []
    errors: List[np.ndarray] = []
    for step in range(rollout_steps + 1):
        decoded = decoder(z_t).clamp(0, 1)
        z_back = model.encoder(decoded)
        err = (z_back - z_t).norm(dim=-1)
        errors.append(err.detach().cpu().numpy())
        step_indices.append(np.full(err.shape, step, dtype=np.int64))
        if step < rollout_steps:
            act = action_seq[:, step]
            h_next = model.predictor(z_t, h_t, act)
            z_t = model.h_to_z(h_next)
            h_t = h_next
    return np.concatenate(step_indices), np.concatenate(errors)


def save_off_manifold_visualization(
    out_dir: Path,
    step_indices: np.ndarray,
    errors: np.ndarray,
    global_step: int,
) -> None:
    if step_indices.size == 0 or errors.size == 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=figsize_for_grid(1, 2), constrained_layout=True)
    rng = np.random.default_rng(0)
    jitter = rng.normal(scale=0.06, size=step_indices.shape)
    axes[0].scatter(step_indices + jitter, errors, s=10, alpha=0.6, color="tab:blue")
    axes[0].set_title("Off-manifold error by rollout index")
    axes[0].set_xlabel("rollout index")
    axes[0].set_ylabel("||enc(dec(z_roll)) - z_roll||")
    axes[1].hist(errors, bins=30, color="tab:orange", alpha=0.8)
    axes[1].set_title("Off-manifold error histogram")
    axes[1].set_xlabel("error")
    axes[1].set_ylabel("count")
    out_path = out_dir / f"off_manifold_{global_step:07d}.png"
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)


def _append_csv_row(path: Path, header: Sequence[str], row: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(header)
        writer.writerow(row)


def _write_step_csv(out_dir: Path, name: str, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _compute_norm_stats(tensor: torch.Tensor, quantile: float = 0.95) -> Tuple[float, float]:
    if tensor.numel() == 0:
        return 0.0, 0.0
    norms = tensor.detach().flatten(0, -2).norm(dim=-1)
    if norms.numel() == 0:
        return 0.0, 0.0
    mean = float(norms.mean().item())
    try:
        p95 = float(torch.quantile(norms, quantile).item())
    except RuntimeError:
        p95 = float(norms.max().item())
    return mean, p95


def _shift_frame(frame: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    _, h, w = frame.shape
    shifted = torch.zeros_like(frame)
    x0_src = max(0, -dx)
    x1_src = min(w, w - dx) if dx < 0 else min(w, w - dx)
    y0_src = max(0, -dy)
    y1_src = min(h, h - dy) if dy < 0 else min(h, h - dy)
    x0_dst = max(0, dx)
    x1_dst = min(w, w + dx)
    y0_dst = max(0, dy)
    y1_dst = min(h, h + dy)
    if x0_src < x1_src and y0_src < y1_src:
        shifted[:, y0_dst:y1_dst, x0_dst:x1_dst] = frame[:, y0_src:y1_src, x0_src:x1_src]
    return shifted


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


def _write_model_shape_summary(
    run_dir: Path,
    dataset: TrajectorySequenceDataset,
    model: JEPAWorldModel,
    decoder: VisualizationDecoder,
    model_cfg: ModelConfig,
    flops_per_step: Optional[int],
) -> None:
    raw_shape = _infer_raw_frame_shape(dataset)
    total_params = _count_parameters((model, decoder))
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
    (run_dir / "model_shape.txt").write_text(summary)


def _write_model_metadata(
    run_dir: Path,
    model: JEPAWorldModel,
    decoder: VisualizationDecoder,
    flops_per_step: Optional[int],
) -> None:
    total_params = _count_parameters((model, decoder))
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


def _init_optimizer(
    *,
    params: List[torch.nn.Parameter],
    use_soap: bool,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> torch.optim.Optimizer:
    if use_soap:
        if SOAP is None:
            raise ImportError(
                "SOAP optimizer requires the pytorch-optimizer package. Install with `pip install pytorch-optimizer`."
            ) from _soap_import_error
        if device.type == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in {"1", "true", "TRUE"}:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            print(
                "[warning] Enabled PYTORCH_ENABLE_MPS_FALLBACK=1 for SOAP on MPS to allow CPU fallback for missing ops."
            )
        return SOAP(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )
    return torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
    )


def _prepare_self_distance_inputs(
    data_root: Path,
    train_trajs: List[str],
    image_hw: Tuple[int, int],
    run_dir: Path,
) -> SelfDistanceInputs:
    traj_map = list_trajectories(data_root)
    if not traj_map:
        raise AssertionError("Self-distance requested but no trajectories were found.")
    ordered = train_trajs if train_trajs else list(traj_map.keys())
    chosen: Optional[str] = None
    for name in ordered:
        if name in traj_map and len(traj_map[name]) >= 2:
            chosen = name
            break
    if chosen is None:
        raise AssertionError("Self-distance requested but no trajectory had enough frames.")
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
        raise AssertionError("Self-distance requested but no frames were loaded.")
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
    off_manifold_generator = torch.Generator()
    off_manifold_generator.manual_seed(python_rng.randint(0, 2**32 - 1))
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
    pca_p_dir = run_dir / "pca_p"
    pca_h_dir = run_dir / "pca_h"
    diagnostics_delta_dir = run_dir / "vis_delta_z_pca"
    diagnostics_delta_p_dir = run_dir / "vis_delta_p_pca"
    diagnostics_delta_h_dir = run_dir / "vis_delta_h_pca"
    diagnostics_alignment_dir = run_dir / "vis_action_alignment_z"
    diagnostics_alignment_raw_dir = run_dir / "vis_action_alignment_z_raw"
    diagnostics_alignment_centered_dir = run_dir / "vis_action_alignment_z_centered"
    diagnostics_alignment_p_dir = run_dir / "vis_action_alignment_p"
    diagnostics_alignment_p_raw_dir = run_dir / "vis_action_alignment_p_raw"
    diagnostics_alignment_p_centered_dir = run_dir / "vis_action_alignment_p_centered"
    diagnostics_alignment_h_dir = run_dir / "vis_action_alignment_h"
    diagnostics_alignment_h_raw_dir = run_dir / "vis_action_alignment_h_raw"
    diagnostics_alignment_h_centered_dir = run_dir / "vis_action_alignment_h_centered"
    diagnostics_cycle_dir = run_dir / "vis_cycle_error_z"
    diagnostics_cycle_p_dir = run_dir / "vis_cycle_error_p"
    diagnostics_cycle_h_dir = run_dir / "vis_cycle_error_h"
    diagnostics_frames_dir = run_dir / "vis_diagnostics_frames"
    diagnostics_rollout_divergence_dir = run_dir / "vis_rollout_divergence"
    diagnostics_straightline_p_dir = run_dir / "vis_straightline_p"
    diagnostics_z_consistency_dir = run_dir / "vis_z_consistency"
    diagnostics_z_monotonicity_dir = run_dir / "vis_z_monotonicity"
    diagnostics_path_independence_dir = run_dir / "vis_path_independence"
    diagnostics_h_ablation_dir = run_dir / "vis_h_ablation"
    diagnostics_h_drift_dir = run_dir / "vis_h_drift_by_action"
    diagnostics_norm_timeseries_dir = run_dir / "vis_norm_timeseries"
    spike_diagnostics_dir = run_dir / "spike_diagnostics"
    graph_diagnostics_dir = run_dir / "graph_diagnostics_z"
    graph_diagnostics_p_dir = run_dir / "graph_diagnostics_p"
    graph_diagnostics_h_dir = run_dir / "graph_diagnostics_h"
    vis_ctrl_dir = run_dir / "vis_vis_ctrl"
    vis_off_manifold_dir = run_dir / "vis_off_manifold"
    samples_hard_dir = run_dir / "samples_hard"
    samples_hard_val_dir = run_dir / "samples_hard_val"
    inputs_vis_dir = run_dir / "vis_inputs"
    pair_vis_dir = run_dir / "vis_pairs"
    vis_self_distance_z_dir = run_dir / "vis_self_distance_z"
    vis_self_distance_p_dir = run_dir / "vis_self_distance_p"
    vis_self_distance_h_dir = run_dir / "vis_self_distance_h"
    vis_composability_z_dir = run_dir / "vis_composability_z"
    vis_composability_p_dir = run_dir / "vis_composability_p"
    vis_composability_h_dir = run_dir / "vis_composability_h"
    vis_state_embedding_dir = run_dir / "vis_state_embedding"
    vis_odometry_dir = run_dir / "vis_odometry"
    self_distance_z_dir = run_dir / "self_distance_z"
    self_distance_p_dir = run_dir / "self_distance_p"
    self_distance_h_dir = run_dir / "self_distance_h"
    checkpoints_dir = run_dir / "checkpoints"

    # Legacy aliases: keep _s variable names for backward-compat while writing to new _p outputs.
    pca_s_dir = pca_p_dir
    diagnostics_delta_s_dir = diagnostics_delta_p_dir
    diagnostics_alignment_s_dir = diagnostics_alignment_p_dir
    diagnostics_alignment_s_raw_dir = diagnostics_alignment_p_raw_dir
    diagnostics_alignment_s_centered_dir = diagnostics_alignment_p_centered_dir
    diagnostics_cycle_s_dir = diagnostics_cycle_p_dir
    diagnostics_straightline_s_dir = diagnostics_straightline_p_dir
    vis_composability_s_dir = vis_composability_p_dir
    vis_self_distance_s_dir = vis_self_distance_p_dir
    self_distance_s_dir = self_distance_p_dir
    graph_diagnostics_s_dir = graph_diagnostics_p_dir

    print(f"[run] Writing outputs to {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pca_z_dir.mkdir(parents=True, exist_ok=True)
    pca_p_dir.mkdir(parents=True, exist_ok=True)
    pca_h_dir.mkdir(parents=True, exist_ok=True)
    if cfg.diagnostics.enabled:
        diagnostics_delta_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_delta_p_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_delta_h_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_alignment_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_alignment_raw_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_alignment_centered_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_alignment_p_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_alignment_p_raw_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_alignment_p_centered_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_alignment_h_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_alignment_h_raw_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_alignment_h_centered_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_cycle_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_cycle_p_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_cycle_h_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_frames_dir.mkdir(parents=True, exist_ok=True)
        vis_composability_z_dir.mkdir(parents=True, exist_ok=True)
        vis_composability_p_dir.mkdir(parents=True, exist_ok=True)
        vis_composability_h_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_rollout_divergence_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_straightline_p_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_z_consistency_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_z_monotonicity_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_path_independence_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_h_ablation_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_h_drift_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_norm_timeseries_dir.mkdir(parents=True, exist_ok=True)
    if cfg.spike_diagnostics.enabled:
        spike_diagnostics_dir.mkdir(parents=True, exist_ok=True)
    if cfg.graph_diagnostics.enabled:
        graph_diagnostics_dir.mkdir(parents=True, exist_ok=True)
        graph_diagnostics_p_dir.mkdir(parents=True, exist_ok=True)
        graph_diagnostics_h_dir.mkdir(parents=True, exist_ok=True)
    if cfg.vis_ctrl.enabled:
        vis_ctrl_dir.mkdir(parents=True, exist_ok=True)
    vis_off_manifold_dir.mkdir(parents=True, exist_ok=True)
    samples_hard_dir.mkdir(parents=True, exist_ok=True)
    samples_hard_val_dir.mkdir(parents=True, exist_ok=True)
    vis_self_distance_z_dir.mkdir(parents=True, exist_ok=True)
    vis_self_distance_p_dir.mkdir(parents=True, exist_ok=True)
    vis_self_distance_h_dir.mkdir(parents=True, exist_ok=True)
    vis_state_embedding_dir.mkdir(parents=True, exist_ok=True)
    vis_odometry_dir.mkdir(parents=True, exist_ok=True)
    self_distance_z_dir.mkdir(parents=True, exist_ok=True)
    self_distance_p_dir.mkdir(parents=True, exist_ok=True)
    self_distance_h_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    debug_vis = cfg.debug_visualization
    if debug_vis.input_vis_every_steps > 0:
        inputs_vis_dir.mkdir(parents=True, exist_ok=True)
    if debug_vis.pair_vis_every_steps > 0:
        pair_vis_dir.mkdir(parents=True, exist_ok=True)

    loss_history = LossHistory()
    spike_tracker = SpikeTracker(cfg.spike_diagnostics) if cfg.spike_diagnostics.enabled else None
    spike_events_path = metrics_dir / "spike_events.csv"
    recon_overlay_path = metrics_dir / "recon_spike_action_overlay.csv"

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

    dataset_action_dim = dataset.action_dim
    if val_dataset is not None:
        val_action_dim = val_dataset.action_dim
        if val_action_dim != dataset_action_dim:
            raise AssertionError(f"Validation action_dim {val_action_dim} does not match train action_dim {dataset_action_dim}")
    assert dataset_action_dim == 8, f"Expected action_dim 8, got {dataset_action_dim}"
    if model_cfg.action_dim != dataset_action_dim:
        model_cfg = replace(model_cfg, action_dim=dataset_action_dim)

    if cfg.graph_diagnostics.enabled:
        graph_diag_batch_cpu = _prepare_graph_diag_dataset(
            cfg=cfg,
            model_cfg=model_cfg,
            dataset_action_dim=dataset_action_dim,
            train_trajs=train_trajs,
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

    flops_per_step = calculate_flops_per_step(model_cfg, cfg.batch_size, cfg.seq_len)

    # Write model_shape.txt
    _write_model_shape_summary(run_dir, dataset, model, decoder, model_cfg, flops_per_step)

    # Write metadata_model.txt (TOML format)
    _write_model_metadata(run_dir, model, decoder, flops_per_step)

    # --- Optimizer initialization ---
    params = list(model.parameters()) + list(decoder.parameters())
    optimizer = _init_optimizer(
        params=params,
        use_soap=cfg.use_soap,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        device=device,
    )

    # --- Fixed visualization batch (required later) ---
    fixed_batch_cpu, fixed_selection = _build_fixed_vis_batch(dataloader, cfg.vis.rows)
    off_manifold_batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor, List[List[str]]]] = None
    off_manifold_steps = max(1, cfg.rollout_horizon * 2)
    off_points_per_rollout = off_manifold_steps + 1
    off_sample_count = max(1, int(round(256 / off_points_per_rollout)))
    try:
        off_manifold_batch_cpu = _build_off_manifold_batch(
            data_root=cfg.data_root,
            train_trajs=train_trajs,
            seq_len=off_manifold_steps + 1,
            image_hw=(model_cfg.image_size, model_cfg.image_size),
            sample_count=off_sample_count,
            generator=off_manifold_generator,
        )
    except Exception as exc:
        print(f"[warn] off-manifold batch unavailable: {exc}")

    if cfg.vis.embedding_projection_samples <= 0:
        raise AssertionError(
            "Embedding projection requested but vis.embedding_projection_samples is not positive."
        )
    embedding_batch_cpu = _build_embedding_batch(
        dataset,
        cfg.vis.embedding_projection_samples,
        generator=embedding_generator,
    )

    if cfg.diagnostics.enabled:
        if cfg.diagnostics.sample_sequences <= 0:
            raise AssertionError(
                "Diagnostics requested but diagnostics.sample_sequences is not positive."
            )
        diagnostics_batch_cpu = _build_embedding_batch(
            dataset,
            cfg.diagnostics.sample_sequences,
            generator=diagnostics_generator,
        )

    if cfg.vis_ctrl.enabled:
        if cfg.vis_ctrl.sample_sequences <= 0:
            raise AssertionError(
                "Vis-ctrl requested but vis_ctrl.sample_sequences is not positive."
            )
        vis_ctrl_batch_cpu = _build_embedding_batch(
            dataset,
            cfg.vis_ctrl.sample_sequences,
            generator=vis_ctrl_generator,
        )

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

        if spike_tracker is not None:
            spike_events = spike_tracker.detect(metrics)
            for event in spike_events:
                _write_spike_event_row(spike_events_path, global_step, event)
                _write_spike_batch(
                    spike_diagnostics_dir,
                    global_step,
                    event,
                    batch,
                    metrics,
                    cfg.spike_diagnostics,
                )
                print(
                    f"[spike] step {global_step} {event['metric']}={event['value']:.4f} "
                    f"(mean {event['mean']:.4f}, std {event['std']:.4f}, z {event['z']:.2f})"
                )
        if cfg.spike_diagnostics.enabled:
            recon_event = None
            if spike_tracker is not None:
                recon_event = next(
                    (event for event in spike_events if event["metric"] == "loss_recon_multi_box"),
                    None,
                )
            recon_value = float(metrics.get("loss_recon_multi_box", 0.0))
            _write_recon_spike_action_overlay_row(
                recon_overlay_path,
                global_step,
                batch[1],
                recon_value,
                recon_event,
            )

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
            samples_per_sec: float
            if elapsed_seconds > 0:
                samples_per_sec = total_samples_processed / elapsed_seconds
            else:
                samples_per_sec = 0.0
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
        if len(batch) <= 2 or batch[2] is None:
            raise AssertionError("Rolling visualization batch requires frame paths.")
        rolling_batch_cpu = (
            batch[0].clone(),
            batch[1].clone(),
            [list(paths) for paths in batch[2]],
        )

        if (
            debug_vis.input_vis_every_steps > 0
            and global_step % debug_vis.input_vis_every_steps == 0
        ):
            recon_vis: torch.Tensor
            assert torch.is_grad_enabled()
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
            assert torch.is_grad_enabled()
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

            sequences, grad_label = _build_visualization_sequences(
                batch_cpu=fixed_batch_cpu,
                selection=fixed_selection,
                model=model,
                decoder=decoder,
                device=device,
                vis_cfg=cfg.vis,
                vis_selection_generator=vis_selection_generator,
                use_z2h_init=weights.z2h > 0,
            )
            save_rollout_sequence_batch(
                fixed_vis_dir,
                sequences,
                grad_label,
                global_step,
                include_pixel_delta=weights.pixel_delta > 0,
            )

            sequences, grad_label = _build_visualization_sequences(
                batch_cpu=rolling_batch_cpu,
                selection=None,
                model=model,
                decoder=decoder,
                device=device,
                vis_cfg=cfg.vis,
                vis_selection_generator=vis_selection_generator,
                use_z2h_init=weights.z2h > 0,
            )
            save_rollout_sequence_batch(
                rolling_vis_dir,
                sequences,
                grad_label,
                global_step,
                include_pixel_delta=weights.pixel_delta > 0,
            )
            if off_manifold_batch_cpu is not None:
                assert torch.is_grad_enabled()
                with torch.no_grad():
                    step_indices, errors = _compute_off_manifold_errors(
                        model=model,
                        decoder=decoder,
                        batch_cpu=off_manifold_batch_cpu,
                        device=device,
                        rollout_steps=off_manifold_steps,
                    )
                assert torch.is_grad_enabled()
                save_off_manifold_visualization(
                    vis_off_manifold_dir,
                    step_indices,
                    errors,
                    global_step,
                )

            assert torch.is_grad_enabled()
            with torch.no_grad():
                embed_frames = embedding_batch_cpu[0].to(device)
                embed_actions = embedding_batch_cpu[1].to(device)
                embed_outputs = model.encode_sequence(embed_frames)
                _, _, h_states = rollout_teacher_forced(
                    model,
                    embed_outputs["embeddings"],
                    embed_actions,
                    use_z2h_init=weights.z2h > 0,
                )
                p_embeddings = model.h2p(h_states)
            assert torch.is_grad_enabled()
            save_embedding_projection(
                embed_outputs["embeddings"],
                pca_z_dir / f"pca_z_{global_step:07d}.png",
                "PCA z",
            )
            save_embedding_projection(
                p_embeddings,
                pca_p_dir / f"pca_p_{global_step:07d}.png",
                "PCA p",
            )
            save_embedding_projection(
                h_states,
                pca_h_dir / f"pca_h_{global_step:07d}.png",
                "PCA h",
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
            assert torch.is_grad_enabled()
            with torch.no_grad():
                self_dist_frames = self_distance_inputs.frames.to(device)
                self_dist_actions = torch.from_numpy(self_distance_inputs.actions).unsqueeze(0).to(device)
                self_dist_embeddings_full = model.encode_sequence(self_dist_frames)["embeddings"]
                self_dist_embeddings = self_dist_embeddings_full[0]
                _, _, self_dist_h_states_batch = rollout_teacher_forced(
                    model,
                    self_dist_embeddings_full,
                    self_dist_actions,
                    use_z2h_init=weights.z2h > 0,
                )
                self_dist_h_states = self_dist_h_states_batch[0]
                self_dist_p = model.h2p(self_dist_h_states.unsqueeze(0))[0]
            assert torch.is_grad_enabled()
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
            write_self_distance_outputs(
                self_dist_h_states,
                self_distance_inputs,
                self_distance_h_dir,
                vis_self_distance_h_dir,
                global_step,
                embedding_label="h",
                title_prefix="Self-distance (H)",
                file_prefix="self_distance_h",
                cosine_prefix="self_distance_h_cosine",
            )
            write_self_distance_outputs(
                self_dist_p,
                self_distance_inputs,
                self_distance_p_dir,
                vis_self_distance_p_dir,
                global_step,
                embedding_label="p",
                title_prefix="Self-distance (P)",
                file_prefix="self_distance_p",
                cosine_prefix="self_distance_p_cosine",
            )
            hist_frames = rolling_batch_cpu[0]
            hist_actions = rolling_batch_cpu[1]
            write_state_embedding_outputs(
                model,
                self_distance_inputs,
                device,
                self_distance_p_dir,
                vis_self_distance_p_dir,
                vis_state_embedding_dir,
                vis_odometry_dir,
                global_step,
                use_z2h_init=weights.z2h > 0,
                hist_frames_cpu=hist_frames,
                hist_actions_cpu=hist_actions,
            )
            if cfg.diagnostics.enabled:
                diag_frames = diagnostics_batch_cpu[0]
                diag_actions = diagnostics_batch_cpu[1]
                diag_paths = diagnostics_batch_cpu[2]
                if not (diag_frames.shape[0] > 0 and diag_frames.shape[1] >= 2):
                    raise AssertionError("Diagnostics require at least one sequence with two frames.")

                warmup_frames = max(model.cfg.warmup_frames_h, 0)
                assert torch.is_grad_enabled()
                with torch.no_grad():
                    diag_frames_device = diag_frames.to(device)
                    diag_actions_device = diag_actions.to(device)
                    diag_embeddings = model.encode_sequence(diag_frames_device)["embeddings"]
                    _, _, diag_h_states = rollout_teacher_forced(
                        model,
                        diag_embeddings,
                        diag_actions_device,
                        use_z2h_init=weights.z2h > 0,
                    )
                    diag_p_embeddings = model.h2p(diag_h_states)
                diag_composability = compute_composability_series(
                    model,
                    diag_embeddings,
                    diag_h_states,
                    diag_actions_device,
                    warmup_frames,
                    cfg.diagnostics.min_action_count,
                )
                assert torch.is_grad_enabled()
                diag_p_series = diag_composability.get("p") or diag_composability.get("s")

                def _write_motion_bundle(
                    name: str,
                    motion: MotionSubspace,
                    delta_dir: Path,
                    alignment_dir: Path,
                    alignment_raw_dir: Path,
                    alignment_centered_dir: Path,
                    cycle_dir: Path,
                ) -> Tuple[
                    Dict[str, Dict[int, Dict[str, float]]],
                    Dict[str, Dict[int, Dict[str, float]]],
                    Tuple[List[Tuple[int, float]], Dict[int, float]],
                ]:
                    save_delta_pca_plot(
                        delta_dir / f"delta_{name}_pca_{global_step:07d}.png",
                        motion.variance_ratio,
                        motion.delta_proj,
                        motion.proj_flat,
                        motion.action_ids,
                        motion.action_dim,
                        name,
                    )
                    save_variance_spectrum_plot(
                        motion.variance_ratio,
                        delta_dir,
                        global_step,
                        name,
                    )
                    write_variance_report(
                        motion.variance_ratio,
                        delta_dir,
                        global_step,
                        name,
                    )
                    stats = compute_action_alignment_stats(
                        motion.delta_proj,
                        motion.action_ids,
                        cfg.diagnostics.min_action_count,
                        cfg.diagnostics.cosine_high_threshold,
                    )
                    debug = build_action_alignment_debug(
                        stats,
                        motion.delta_proj,
                        motion.action_ids,
                    )
                    save_action_alignment_detail_plot(
                        alignment_dir / f"action_alignment_detail_{global_step:07d}.png",
                        debug,
                        cfg.diagnostics.cosine_high_threshold,
                        motion.action_dim,
                        alignment_label="PCA",
                    )
                    write_action_alignment_report(
                        stats,
                        motion.action_dim,
                        inverse_map,
                        alignment_dir,
                        global_step,
                    )
                    write_action_alignment_strength(
                        stats,
                        motion.action_dim,
                        alignment_dir,
                        global_step,
                    )
                    write_action_alignment_crosscheck(
                        stats,
                        motion,
                        alignment_dir,
                        global_step,
                    )
                    motion_raw = replace(motion, delta_proj=motion.delta_embed)
                    stats_raw = compute_action_alignment_stats(
                        motion_raw.delta_proj,
                        motion_raw.action_ids,
                        cfg.diagnostics.min_action_count,
                        cfg.diagnostics.cosine_high_threshold,
                    )
                    debug_raw = build_action_alignment_debug(
                        stats_raw,
                        motion_raw.delta_proj,
                        motion_raw.action_ids,
                    )
                    save_action_alignment_detail_plot(
                        alignment_raw_dir / f"action_alignment_detail_{global_step:07d}.png",
                        debug_raw,
                        cfg.diagnostics.cosine_high_threshold,
                        motion_raw.action_dim,
                        alignment_label="raw delta",
                    )
                    write_action_alignment_crosscheck(
                        stats_raw,
                        motion_raw,
                        alignment_raw_dir,
                        global_step,
                    )
                    motion_centered = replace(motion, delta_proj=motion.delta_centered)
                    stats_centered = compute_action_alignment_stats(
                        motion_centered.delta_proj,
                        motion_centered.action_ids,
                        cfg.diagnostics.min_action_count,
                        cfg.diagnostics.cosine_high_threshold,
                    )
                    debug_centered = build_action_alignment_debug(
                        stats_centered,
                        motion_centered.delta_proj,
                        motion_centered.action_ids,
                    )
                    save_action_alignment_detail_plot(
                        alignment_centered_dir / f"action_alignment_detail_{global_step:07d}.png",
                        debug_centered,
                        cfg.diagnostics.cosine_high_threshold,
                        motion_centered.action_dim,
                        alignment_label="centered delta",
                    )
                    write_action_alignment_crosscheck(
                        stats_centered,
                        motion_centered,
                        alignment_centered_dir,
                        global_step,
                    )
                    for dir_entry, stats_entry, debug_entry in [
                        (alignment_dir, stats, debug),
                        (alignment_raw_dir, stats_raw, debug_raw),
                        (alignment_centered_dir, stats_centered, debug_centered),
                    ]:
                        write_action_alignment_csv(
                            dir_entry,
                            global_step,
                            motion.action_dim,
                            stats_entry,
                        )
                        write_action_alignment_full_csv(
                            dir_entry,
                            global_step,
                            motion.action_dim,
                            stats_entry,
                        )
                        write_action_alignment_pairwise_csv(
                            dir_entry,
                            global_step,
                            motion.action_dim,
                            debug_entry,
                        )
                        write_action_alignment_overview_txt(
                            dir_entry,
                            global_step,
                            cfg.diagnostics.cosine_high_threshold,
                            debug_entry,
                        )
                    cycle_errors, cycle_per_action = compute_cycle_errors(
                        motion.proj_sequences,
                        motion.actions_seq,
                        inverse_map,
                        include_synthetic=cfg.diagnostics.synthesize_cycle_samples,
                    )
                    save_cycle_error_plot(
                        cycle_dir / f"cycle_error_{global_step:07d}.png",
                        [e[1] for e in cycle_errors],
                        cycle_per_action,
                        motion.action_dim,
                    )
                    write_delta_variance_csv(
                        delta_dir,
                        global_step,
                        motion.variance_ratio,
                        name,
                    )
                    write_delta_samples_csv(
                        delta_dir,
                        global_step,
                        motion.paths,
                        name,
                    )
                    write_cycle_error_values_csv(
                        cycle_dir,
                        global_step,
                        motion.action_dim,
                        cycle_errors,
                    )
                    write_cycle_error_summary_csv(
                        cycle_dir,
                        global_step,
                        motion.action_dim,
                        cycle_per_action,
                    )
                    return (
                        {"pca": stats, "raw": stats_raw, "centered": stats_centered},
                        {"pca": debug, "raw": debug_raw, "centered": debug_centered},
                        (cycle_errors, cycle_per_action),
                    )

                inverse_map = build_action_inverse_map(diag_actions.detach().cpu().numpy())
                motion_z = build_motion_subspace(
                    diag_embeddings,
                    diag_actions,
                    cfg.diagnostics.top_k_components,
                    diag_paths,
                )
                motion_h = build_motion_subspace(
                    diag_h_states,
                    diag_actions,
                    cfg.diagnostics.top_k_components,
                    diag_paths,
                )
                motion_p = build_motion_subspace(
                    diag_p_embeddings,
                    diag_actions,
                    cfg.diagnostics.top_k_components,
                    diag_paths,
                )
                save_composability_plot(
                    vis_composability_z_dir / f"composability_z_{global_step:07d}.png",
                    diag_composability["z"],
                    "z",
                )
                save_composability_plot(
                    vis_composability_h_dir / f"composability_h_{global_step:07d}.png",
                    diag_composability["h"],
                    "h",
                )
                save_composability_plot(
                    vis_composability_p_dir / f"composability_p_{global_step:07d}.png",
                    diag_p_series,
                    "p",
                )
                alignment_stats_z, alignment_debug_z, (cycle_errors_z, cycle_per_action_z) = _write_motion_bundle(
                    "z",
                    motion_z,
                    diagnostics_delta_dir,
                    diagnostics_alignment_dir,
                    diagnostics_alignment_raw_dir,
                    diagnostics_alignment_centered_dir,
                    diagnostics_cycle_dir,
                )
                alignment_stats_h, alignment_debug_h, (cycle_errors_h, cycle_per_action_h) = _write_motion_bundle(
                    "h",
                    motion_h,
                    diagnostics_delta_h_dir,
                    diagnostics_alignment_h_dir,
                    diagnostics_alignment_h_raw_dir,
                    diagnostics_alignment_h_centered_dir,
                    diagnostics_cycle_h_dir,
                )
                alignment_stats_p, alignment_debug_p, (cycle_errors_p, cycle_per_action_p) = _write_motion_bundle(
                    "p",
                    motion_p,
                    diagnostics_delta_p_dir,
                    diagnostics_alignment_p_dir,
                    diagnostics_alignment_p_raw_dir,
                    diagnostics_alignment_p_centered_dir,
                    diagnostics_cycle_p_dir,
                )
                write_alignment_debug_csv(
                    diag_frames,
                    diag_actions,
                    diag_paths,
                    diagnostics_frames_dir,
                    global_step,
                )
                save_diagnostics_frames(
                    diag_frames,
                    diag_paths,
                    diag_actions,
                    diagnostics_frames_dir,
                    global_step,
                )

                diagnostics_scalars_path = metrics_dir / "diagnostics_scalars.csv"
                z_norm_mean, z_norm_p95 = _compute_norm_stats(diag_embeddings)
                h_norm_mean, h_norm_p95 = _compute_norm_stats(diag_h_states)
                p_norm_mean, p_norm_p95 = _compute_norm_stats(diag_p_embeddings)
                if diag_h_states.shape[1] >= 2:
                    h_drift = diag_h_states[:, 1:] - diag_h_states[:, :-1]
                    p_drift = diag_p_embeddings[:, 1:] - diag_p_embeddings[:, :-1]
                    h_drift_mean = float(h_drift.norm(dim=-1).mean().item())
                    p_drift_mean = float(p_drift.norm(dim=-1).mean().item())
                else:
                    h_drift_mean = 0.0
                    p_drift_mean = 0.0
                id_acc_p = 0.0
                if diag_p_embeddings.shape[1] >= 2:
                    action_logits_p = model.inverse_dynamics_p(
                        diag_p_embeddings[:, :-1],
                        diag_p_embeddings[:, 1:],
                    )
                    action_preds = (torch.sigmoid(action_logits_p) > 0.5).to(diag_actions_device.dtype)
                    id_acc_p = float((action_preds == diag_actions_device[:, :-1]).float().mean().item())
                _append_csv_row(
                    diagnostics_scalars_path,
                    [
                        "step",
                        "z_norm_mean",
                        "z_norm_p95",
                        "h_norm_mean",
                        "h_norm_p95",
                        "p_norm_mean",
                        "p_norm_p95",
                        "h_drift_mean",
                        "p_drift_mean",
                        "id_acc_p",
                    ],
                    [
                        global_step,
                        z_norm_mean,
                        z_norm_p95,
                        h_norm_mean,
                        h_norm_p95,
                        p_norm_mean,
                        p_norm_p95,
                        h_drift_mean,
                        p_drift_mean,
                        id_acc_p,
                    ],
                )
                try:
                    with diagnostics_scalars_path.open("r", newline="") as handle:
                        reader = csv.DictReader(handle)
                        steps_list: List[int] = []
                        z_mean_list: List[float] = []
                        z_p95_list: List[float] = []
                        h_mean_list: List[float] = []
                        h_p95_list: List[float] = []
                        p_mean_list: List[float] = []
                        p_p95_list: List[float] = []
                        for row in reader:
                            steps_list.append(int(float(row["step"])))
                            z_mean_list.append(float(row["z_norm_mean"]))
                            z_p95_list.append(float(row["z_norm_p95"]))
                            h_mean_list.append(float(row["h_norm_mean"]))
                            h_p95_list.append(float(row["h_norm_p95"]))
                            p_mean_list.append(float(row.get("p_norm_mean", row.get("s_norm_mean", 0.0))))
                            p_p95_list.append(float(row.get("p_norm_p95", row.get("s_norm_p95", 0.0))))
                    if steps_list:
                        save_norm_timeseries_plot(
                            diagnostics_norm_timeseries_dir / f"norm_timeseries_{global_step:07d}.png",
                            steps_list,
                            z_mean_list,
                            z_p95_list,
                            h_mean_list,
                            h_p95_list,
                            p_mean_list,
                            p_p95_list,
                        )
                except OSError:
                    pass

                action_ids_np = compress_actions_to_ids(diag_actions.detach().cpu().numpy())
                action_ids_flat = action_ids_np.reshape(-1)
                unique_actions, action_counts = np.unique(action_ids_flat, return_counts=True)
                action_labels = {int(aid): decode_action_id(int(aid), motion_z.action_dim) for aid in unique_actions}
                action_vectors: Dict[int, torch.Tensor] = {}
                flat_actions = diag_actions.detach().cpu().reshape(-1, diag_actions.shape[-1])
                for idx, aid in enumerate(action_ids_flat):
                    if int(aid) not in action_vectors:
                        action_vectors[int(aid)] = flat_actions[idx].to(device)

                def _find_action_id(keyword: str) -> Optional[int]:
                    for aid, label in action_labels.items():
                        if keyword in label:
                            return aid
                    return None

                right_id = _find_action_id("RIGHT")
                left_id = _find_action_id("LEFT")
                up_id = _find_action_id("UP")
                down_id = _find_action_id("DOWN")
                noop_id = _find_action_id("NOOP")
                sorted_ids = [
                    int(aid)
                    for aid, _ in sorted(
                        zip(unique_actions.tolist(), action_counts.tolist()),
                        key=lambda pair: pair[1],
                        reverse=True,
                    )
                ]
                straightline_ids: List[int] = []
                if right_id is not None and left_id is not None:
                    straightline_ids = [right_id, left_id]
                elif up_id is not None and down_id is not None:
                    straightline_ids = [up_id, down_id]
                else:
                    straightline_ids = sorted_ids[:2]

                if straightline_ids and diag_p_embeddings.shape[1] >= 2:
                    p_flat = diag_p_embeddings.detach().cpu().reshape(-1, diag_p_embeddings.shape[-1]).numpy()
                    p_center = p_flat.mean(axis=0, keepdims=True)
                    centered = p_flat - p_center
                    try:
                        _, _, vt = np.linalg.svd(centered, full_matrices=False)
                    except np.linalg.LinAlgError:
                        jitter = np.random.normal(scale=1e-6, size=centered.shape)
                        _, _, vt = np.linalg.svd(centered + jitter, full_matrices=False)
                    projection = vt[:2].T if vt.shape[0] >= 2 else None
                    if projection is not None:
                        seq_len = diag_p_embeddings.shape[1]
                        start_frame = max(min(warmup_frames, seq_len - 2), 0)
                        max_starts = min(cfg.diagnostics.straightline_starts, diag_p_embeddings.shape[0])
                        if max_starts > 0:
                            perm = torch.randperm(diag_p_embeddings.shape[0], generator=diagnostics_generator)[:max_starts]
                            trajectories: List[StraightLineTrajectory] = []
                            palette = ["#4c72b0", "#55a868", "#c44e52", "#8172b3"]
                            for action_idx, action_id in enumerate(straightline_ids):
                                action_vec = action_vectors.get(action_id)
                                if action_vec is None:
                                    continue
                                label = action_labels.get(action_id, f"action {action_id}")
                                color = palette[action_idx % len(palette)]
                                for row_offset, b_idx in enumerate(perm):
                                    b = int(b_idx.item())
                                    z_t = diag_embeddings[b, start_frame]
                                    h_t = diag_h_states[b, start_frame]
                                    p_points = [diag_p_embeddings[b, start_frame].detach().cpu().numpy()]
                                    for _ in range(cfg.diagnostics.straightline_steps):
                                        h_next = model.predictor(
                                            z_t.unsqueeze(0),
                                            h_t.unsqueeze(0),
                                            action_vec.unsqueeze(0),
                                        )
                                        z_t = model.h_to_z(h_next).squeeze(0)
                                        h_t = h_next.squeeze(0)
                                        p_points.append(model.h2p(h_t.unsqueeze(0)).squeeze(0).detach().cpu().numpy())
                                    p_points_np = np.stack(p_points, axis=0)
                                    proj = (p_points_np - p_center) @ projection
                                    traj_label = f"{label} (start {row_offset + 1})"
                                    trajectories.append(StraightLineTrajectory(points=proj, label=traj_label, color=color))
                            if trajectories:
                                save_straightline_plot(
                                    diagnostics_straightline_p_dir / f"straightline_p_{global_step:07d}.png",
                                    trajectories,
                                )

                rollout_horizon = min(cfg.diagnostics.rollout_divergence_horizon, diag_frames.shape[1] - 1)
                if rollout_horizon > 0:
                    start_span = diag_frames.shape[1] - 1 - warmup_frames
                    if start_span > 0:
                        total_positions = diag_frames.shape[0] * start_span
                        sample_count = min(cfg.diagnostics.rollout_divergence_samples, total_positions)
                        perm = torch.randperm(total_positions, generator=diagnostics_generator)[:sample_count]
                        pixel_errors = torch.zeros(rollout_horizon, device=device)
                        latent_errors = torch.zeros(rollout_horizon, device=device)
                        counts = torch.zeros(rollout_horizon, device=device)
                        for flat_idx in perm.tolist():
                            b = flat_idx // start_span
                            t0 = (flat_idx % start_span) + warmup_frames
                            z_t = diag_embeddings[b, t0]
                            h_t = diag_h_states[b, t0]
                            for k in range(rollout_horizon):
                                if t0 + k >= diag_frames.shape[1] - 1:
                                    break
                                act = diag_actions_device[b, t0 + k]
                                h_next = model.predictor(
                                    z_t.unsqueeze(0),
                                    h_t.unsqueeze(0),
                                    act.unsqueeze(0),
                                )
                                z_t = model.h_to_z(h_next).squeeze(0)
                                h_t = h_next.squeeze(0)
                                decoded = decoder(z_t.unsqueeze(0))
                                pixel_errors[k] += RECON_LOSS(decoded, diag_frames_device[b, t0 + k + 1].unsqueeze(0))
                                p_pred = model.h2p(h_t.unsqueeze(0)).squeeze(0)
                                p_gt = diag_p_embeddings[b, t0 + k + 1]
                                latent_errors[k] += (p_pred - p_gt).norm()
                                counts[k] += 1
                        counts = torch.clamp(counts, min=1.0)
                        pixel_mean = (pixel_errors / counts).detach().cpu().numpy().tolist()
                        latent_mean = (latent_errors / counts).detach().cpu().numpy().tolist()
                        horizons = list(range(1, rollout_horizon + 1))
                        save_rollout_divergence_plot(
                            diagnostics_rollout_divergence_dir / f"rollout_divergence_{global_step:07d}.png",
                            horizons,
                            pixel_mean,
                            latent_mean,
                        )
                        _write_step_csv(
                            diagnostics_rollout_divergence_dir,
                            f"rollout_divergence_{global_step:07d}.csv",
                            ["k", "pixel_error", "latent_error"],
                            zip(horizons, pixel_mean, latent_mean),
                        )

                z_consistency_samples = min(
                    cfg.diagnostics.z_consistency_samples,
                    diag_frames.shape[0] * diag_frames.shape[1],
                )
                if z_consistency_samples > 0:
                    frame_count = diag_frames.shape[0] * diag_frames.shape[1]
                    perm = torch.randperm(frame_count, generator=diagnostics_generator)[:z_consistency_samples]
                    distances: List[float] = []
                    cosines: List[float] = []
                    for sample_idx, flat_idx in enumerate(perm.tolist()):
                        b = flat_idx // diag_frames.shape[1]
                        t0 = flat_idx % diag_frames.shape[1]
                        frame = diag_frames_device[b, t0]
                        repeats = cfg.diagnostics.z_consistency_repeats
                        noise = torch.randn(
                            (repeats, *frame.shape),
                            device=frame.device,
                        ) * cfg.diagnostics.z_consistency_noise_std
                        noisy = (frame.unsqueeze(0) + noise).clamp(0, 1)
                        z_samples = model.encoder(noisy)
                        z_mean = z_samples.mean(dim=0, keepdim=True)
                        dist = (z_samples - z_mean).norm(dim=-1)
                        cos = F.cosine_similarity(z_samples, z_mean, dim=-1)
                        distances.extend(dist.detach().cpu().numpy().tolist())
                        cosines.extend(cos.detach().cpu().numpy().tolist())
                    if distances and cosines:
                        save_z_consistency_plot(
                            diagnostics_z_consistency_dir / f"z_consistency_{global_step:07d}.png",
                            distances,
                            cosines,
                        )
                        _write_step_csv(
                            diagnostics_z_consistency_dir,
                            f"z_consistency_{global_step:07d}.csv",
                            ["idx", "distance", "cosine"],
                            [(idx, d, c) for idx, (d, c) in enumerate(zip(distances, cosines))],
                        )

                max_shift = max(1, cfg.diagnostics.z_monotonicity_max_shift)
                monotonicity_samples = min(
                    cfg.diagnostics.z_monotonicity_samples,
                    diag_frames.shape[0] * diag_frames.shape[1],
                )
                if monotonicity_samples > 0:
                    frame_count = diag_frames.shape[0] * diag_frames.shape[1]
                    perm = torch.randperm(frame_count, generator=diagnostics_generator)[:monotonicity_samples]
                    z_refs: List[torch.Tensor] = []
                    frames_sel: List[torch.Tensor] = []
                    for flat_idx in perm.tolist():
                        b = flat_idx // diag_frames.shape[1]
                        t0 = flat_idx % diag_frames.shape[1]
                        frame = diag_frames_device[b, t0]
                        z_ref = model.encoder(frame.unsqueeze(0)).squeeze(0)
                        z_refs.append(z_ref)
                        frames_sel.append(frame)
                    shifts = list(range(0, max_shift + 1))
                    distances = []
                    for shift in shifts:
                        total = 0.0
                        count = 0
                        for frame, z_ref in zip(frames_sel, z_refs):
                            shifted = _shift_frame(frame, shift, 0)
                            z_shift = model.encoder(shifted.unsqueeze(0)).squeeze(0)
                            total += float((z_shift - z_ref).norm().item())
                            count += 1
                        distances.append(total / max(1, count))
                    save_monotonicity_plot(
                        diagnostics_z_monotonicity_dir / f"z_monotonicity_{global_step:07d}.png",
                        shifts,
                        distances,
                    )
                    _write_step_csv(
                        diagnostics_z_monotonicity_dir,
                        f"z_monotonicity_{global_step:07d}.csv",
                        ["shift", "distance"],
                        zip(shifts, distances),
                    )

                if diag_embeddings.shape[1] >= 2 and straightline_ids:
                    a_id = straightline_ids[0]
                    b_id = straightline_ids[1] if len(straightline_ids) > 1 else straightline_ids[0]
                    action_a = action_vectors.get(a_id)
                    action_b = action_vectors.get(b_id)
                    if action_a is not None and action_b is not None:
                        seq_len = diag_embeddings.shape[1]
                        start_frame = max(min(warmup_frames, seq_len - 2), 0)
                        max_starts = min(cfg.diagnostics.path_independence_samples, diag_embeddings.shape[0])
                        if max_starts > 0:
                            perm = torch.randperm(diag_embeddings.shape[0], generator=diagnostics_generator)[:max_starts]
                            z_diffs = []
                            p_diffs = []
                            use_noop = noop_id is not None and noop_id in action_vectors
                            action_b_first = action_vectors.get(noop_id) if use_noop else action_b
                            action_b_second = action_vectors.get(noop_id) if use_noop else action_a
                            for b_idx in perm.tolist():
                                z_start = diag_embeddings[b_idx, start_frame]
                                h_start = diag_h_states[b_idx, start_frame]
                                z_a = z_start
                                h_a = h_start
                                z_b = z_start
                                h_b = h_start
                                for _ in range(cfg.diagnostics.path_independence_steps):
                                    h_a_next = model.predictor(
                                        z_a.unsqueeze(0),
                                        h_a.unsqueeze(0),
                                        action_a.unsqueeze(0),
                                    )
                                    h_b_next = model.predictor(
                                        z_b.unsqueeze(0),
                                        h_b.unsqueeze(0),
                                        action_b_first.unsqueeze(0),
                                    )
                                    z_a = model.h_to_z(h_a_next).squeeze(0)
                                    h_a = h_a_next.squeeze(0)
                                    z_b = model.h_to_z(h_b_next).squeeze(0)
                                    h_b = h_b_next.squeeze(0)
                                for _ in range(cfg.diagnostics.path_independence_steps):
                                    h_a_next = model.predictor(
                                        z_a.unsqueeze(0),
                                        h_a.unsqueeze(0),
                                        action_b.unsqueeze(0),
                                    )
                                    h_b_next = model.predictor(
                                        z_b.unsqueeze(0),
                                        h_b.unsqueeze(0),
                                        action_b_second.unsqueeze(0),
                                    )
                                    z_a = model.h_to_z(h_a_next).squeeze(0)
                                    h_a = h_a_next.squeeze(0)
                                    z_b = model.h_to_z(h_b_next).squeeze(0)
                                    h_b = h_b_next.squeeze(0)
                                z_diffs.append(float((z_a - z_b).norm().item()))
                                p_a = model.h2p(h_a.unsqueeze(0)).squeeze(0)
                                p_b = model.h2p(h_b.unsqueeze(0)).squeeze(0)
                                p_diffs.append(float((p_a - p_b).norm().item()))
                            if z_diffs and p_diffs:
                                label_a = action_labels.get(a_id, f"action {a_id}")
                                label_b = action_labels.get(b_id, f"action {b_id}")
                                if use_noop:
                                    label_b_path = action_labels.get(noop_id, "NOOP")
                                else:
                                    label_b_path = f"{label_b}+{label_a}"
                                labels = [f"{label_a}+{label_b} vs {label_b_path}"]
                                save_path_independence_plot(
                                    diagnostics_path_independence_dir / f"path_independence_{global_step:07d}.png",
                                    labels,
                                    [float(np.mean(z_diffs))],
                                    [float(np.mean(p_diffs))],
                                )
                                _write_step_csv(
                                    diagnostics_path_independence_dir,
                                    f"path_independence_{global_step:07d}.csv",
                                    ["label", "z_distance", "p_distance"],
                                    [(labels[0], float(np.mean(z_diffs)), float(np.mean(p_diffs)))],
                                )

                if rollout_horizon > 0 and diag_embeddings.shape[1] >= 2:
                    start_span = diag_embeddings.shape[1] - 1 - warmup_frames
                    if start_span > 0:
                        total_positions = diag_embeddings.shape[0] * start_span
                        sample_count = min(cfg.diagnostics.rollout_divergence_samples, total_positions)
                        perm = torch.randperm(total_positions, generator=diagnostics_generator)[:sample_count]
                        pixel_errors = torch.zeros(rollout_horizon, device=device)
                        pixel_errors_zero = torch.zeros(rollout_horizon, device=device)
                        latent_errors = torch.zeros(rollout_horizon, device=device)
                        latent_errors_zero = torch.zeros(rollout_horizon, device=device)
                        counts = torch.zeros(rollout_horizon, device=device)
                        for flat_idx in perm.tolist():
                            b = flat_idx // start_span
                            t0 = (flat_idx % start_span) + warmup_frames
                            z_norm = diag_embeddings[b, t0]
                            h_norm = diag_h_states[b, t0]
                            z_zero = z_norm.clone()
                            for k in range(rollout_horizon):
                                if t0 + k >= diag_embeddings.shape[1] - 1:
                                    break
                                act = diag_actions_device[b, t0 + k]
                                h_next_norm = model.predictor(
                                    z_norm.unsqueeze(0),
                                    h_norm.unsqueeze(0),
                                    act.unsqueeze(0),
                                )
                                h_next_zero = model.predictor(
                                    z_zero.unsqueeze(0),
                                    torch.zeros_like(h_norm).unsqueeze(0),
                                    act.unsqueeze(0),
                                )
                                z_norm = model.h_to_z(h_next_norm).squeeze(0)
                                h_norm = h_next_norm.squeeze(0)
                                z_zero = model.h_to_z(h_next_zero).squeeze(0)
                                decoded_norm = decoder(z_norm.unsqueeze(0))
                                decoded_zero = decoder(z_zero.unsqueeze(0))
                                target = diag_frames_device[b, t0 + k + 1].unsqueeze(0)
                                pixel_errors[k] += RECON_LOSS(decoded_norm, target)
                                pixel_errors_zero[k] += RECON_LOSS(decoded_zero, target)
                                p_norm = model.h2p(h_norm.unsqueeze(0)).squeeze(0)
                                p_zero = model.h2p(h_next_zero).squeeze(0)
                                p_gt = diag_p_embeddings[b, t0 + k + 1]
                                latent_errors[k] += (p_norm - p_gt).norm()
                                latent_errors_zero[k] += (p_zero - p_gt).norm()
                                counts[k] += 1
                        counts = torch.clamp(counts, min=1.0)
                        pixel_mean = (pixel_errors / counts).detach().cpu().numpy().tolist()
                        pixel_zero_mean = (pixel_errors_zero / counts).detach().cpu().numpy().tolist()
                        latent_mean = (latent_errors / counts).detach().cpu().numpy().tolist()
                        latent_zero_mean = (latent_errors_zero / counts).detach().cpu().numpy().tolist()
                        horizons = list(range(1, rollout_horizon + 1))
                        save_ablation_divergence_plot(
                            diagnostics_h_ablation_dir / f"h_ablation_{global_step:07d}.png",
                            horizons,
                            pixel_mean,
                            pixel_zero_mean,
                            latent_mean,
                            latent_zero_mean,
                        )
                        _write_step_csv(
                            diagnostics_h_ablation_dir,
                            f"h_ablation_{global_step:07d}.csv",
                            ["k", "pixel_error", "pixel_error_zero", "latent_error", "latent_error_zero"],
                            zip(horizons, pixel_mean, pixel_zero_mean, latent_mean, latent_zero_mean),
                        )

                if diag_h_states.shape[1] >= 2:
                    h_deltas = diag_h_states[:, 1:] - diag_h_states[:, :-1]
                    action_ids = compress_actions_to_ids(diag_actions[:, :-1].detach().cpu().numpy())
                    action_ids_flat = action_ids.reshape(-1)
                    h_drift_flat = h_deltas.detach().cpu().reshape(-1, h_deltas.shape[-1]).norm(dim=-1).numpy()
                    drift_stats = []
                    for aid in np.unique(action_ids_flat):
                        mask = action_ids_flat == aid
                        count = int(mask.sum())
                        if count == 0:
                            continue
                        mean_drift = float(h_drift_flat[mask].mean())
                        drift_stats.append((int(aid), count, mean_drift))
                    drift_stats.sort(key=lambda row: row[1], reverse=True)
                    drift_stats = drift_stats[: cfg.diagnostics.h_drift_max_actions]
                    if drift_stats:
                        labels = [action_labels.get(aid, f"action {aid}") for aid, _, _ in drift_stats]
                        counts = [count for _, count, _ in drift_stats]
                        drifts = [drift for _, _, drift in drift_stats]
                        save_drift_by_action_plot(
                            diagnostics_h_drift_dir / f"h_drift_by_action_{global_step:07d}.png",
                            labels,
                            drifts,
                            counts,
                        )
                        _write_step_csv(
                            diagnostics_h_drift_dir,
                            f"h_drift_by_action_{global_step:07d}.csv",
                            ["action_id", "label", "count", "mean_drift"],
                            [
                                (aid, action_labels.get(aid, f"action {aid}"), count, drift)
                                for aid, count, drift in drift_stats
                            ],
                        )

            if cfg.vis_ctrl.enabled:
                vis_frames = vis_ctrl_batch_cpu[0].to(device)
                vis_actions = vis_ctrl_batch_cpu[1].to(device)
                vis_embeddings = model.encode_sequence(vis_frames)["embeddings"]
                _, _, vis_h_states = rollout_teacher_forced(
                    model,
                    vis_embeddings,
                    vis_actions,
                    use_z2h_init=weights.z2h > 0,
                )
                vis_p_embeddings = model.h2p(vis_h_states)
                warmup_frames = max(model.cfg.warmup_frames_h, 0)
                metrics_z = compute_vis_ctrl_metrics(
                    vis_embeddings,
                    vis_actions,
                    cfg.vis_ctrl.knn_k_values,
                    warmup_frames,
                    cfg.vis_ctrl.min_action_count,
                    cfg.vis_ctrl.stability_delta,
                    cfg.vis_ctrl.knn_chunk_size,
                )
                metrics_p = compute_vis_ctrl_metrics(
                    vis_p_embeddings,
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
                save_smoothness_knn_distance_eigenvalue_spectrum_plot(
                    vis_ctrl_dir / f"smoothness_z_{global_step:07d}.png",
                    metrics_z,
                    "z",
                )
                save_smoothness_knn_distance_eigenvalue_spectrum_plot(
                    vis_ctrl_dir / f"smoothness_p_{global_step:07d}.png",
                    metrics_p,
                    "p",
                )
                save_smoothness_knn_distance_eigenvalue_spectrum_plot(
                    vis_ctrl_dir / f"smoothness_h_{global_step:07d}.png",
                    metrics_h,
                    "h",
                )
                save_two_step_composition_error_plot(
                    vis_ctrl_dir / f"composition_error_z_{global_step:07d}.png",
                    metrics_z,
                    "z",
                )
                save_two_step_composition_error_plot(
                    vis_ctrl_dir / f"composition_error_p_{global_step:07d}.png",
                    metrics_p,
                    "p",
                )
                save_two_step_composition_error_plot(
                    vis_ctrl_dir / f"composition_error_h_{global_step:07d}.png",
                    metrics_h,
                    "h",
                )
                save_neighborhood_stability_plot(
                    vis_ctrl_dir / f"stability_z_{global_step:07d}.png",
                    metrics_z,
                    "z",
                )
                save_neighborhood_stability_plot(
                    vis_ctrl_dir / f"stability_p_{global_step:07d}.png",
                    metrics_p,
                    "p",
                )
                save_neighborhood_stability_plot(
                    vis_ctrl_dir / f"stability_h_{global_step:07d}.png",
                    metrics_h,
                    "h",
                )
                write_vis_ctrl_metrics_csv(
                    metrics_dir / "vis_ctrl_metrics.csv",
                    global_step,
                    metrics_z,
                    metrics_p,
                    metrics_h,
                )

            if cfg.graph_diagnostics.enabled:
                graph_frames = graph_diag_batch_cpu[0]
                graph_actions = graph_diag_batch_cpu[1]
                graph_diag = _prepare_graph_diagnostics(
                    graph_frames=graph_frames,
                    graph_actions=graph_actions,
                    model=model,
                    ema_model=ema_model,
                    graph_cfg=cfg.graph_diagnostics,
                    device=device,
                    use_z2h_init=weights.z2h > 0,
                )
                z_queries, z_targets, z_predictions = _run_graph_diag(
                    embedding_kind="z",
                    graph_cfg=cfg.graph_diagnostics,
                    model=model,
                    ema_model=ema_model,
                    graph_embeddings=graph_diag.graph_embeddings,
                    graph_preds=graph_diag.graph_preds,
                    graph_h_preds=graph_diag.graph_h_preds,
                    graph_h_states=graph_diag.graph_h_states,
                    ema_embeddings=graph_diag.ema_embeddings,
                    ema_h_states=graph_diag.ema_h_states,
                )
                stats_z = compute_graph_diagnostics_stats(
                    z_queries,
                    z_targets,
                    z_predictions,
                    graph_diag.next_index,
                    graph_diag.next2_index,
                    graph_diag.chunk_ids,
                    cfg.graph_diagnostics,
                    global_step,
                )
                save_rank_cdf_plot(
                    graph_diagnostics_dir / f"rank1_cdf_{global_step:07d}.png",
                    stats_z.ranks1,
                    stats_z.k,
                    "1-step rank CDF",
                )
                save_rank_cdf_plot(
                    graph_diagnostics_dir / f"rank2_cdf_{global_step:07d}.png",
                    stats_z.ranks2,
                    stats_z.k,
                    "2-hop rank CDF",
                )
                save_neff_violin_plot(
                    graph_diagnostics_dir / f"neff_violin_{global_step:07d}.png",
                    stats_z.neff1,
                    stats_z.neff2,
                )
                save_in_degree_hist_plot(
                    graph_diagnostics_dir / f"in_degree_hist_{global_step:07d}.png",
                    stats_z.in_degree,
                )
                save_edge_consistency_hist_plot(
                    graph_diagnostics_dir / f"edge_consistency_{global_step:07d}.png",
                    stats_z.edge_errors,
                    embedding_label="z",
                )
                update_graph_diagnostics_history(
                    graph_diagnostics_dir,
                    stats_z,
                    global_step,
                    metrics_dir / "graph_diagnostics_z.csv",
                )
                h_queries, h_targets, h_predictions = _run_graph_diag(
                    embedding_kind="h",
                    graph_cfg=cfg.graph_diagnostics,
                    model=model,
                    ema_model=ema_model,
                    graph_embeddings=graph_diag.graph_embeddings,
                    graph_preds=graph_diag.graph_preds,
                    graph_h_preds=graph_diag.graph_h_preds,
                    graph_h_states=graph_diag.graph_h_states,
                    ema_embeddings=graph_diag.ema_embeddings,
                    ema_h_states=graph_diag.ema_h_states,
                )
                stats_h = compute_graph_diagnostics_stats(
                    h_queries,
                    h_targets,
                    h_predictions,
                    graph_diag.next_index,
                    graph_diag.next2_index,
                    graph_diag.chunk_ids,
                    cfg.graph_diagnostics,
                    global_step,
                )
                save_rank_cdf_plot(
                    graph_diagnostics_h_dir / f"rank1_cdf_{global_step:07d}.png",
                    stats_h.ranks1,
                    stats_h.k,
                    "1-step rank CDF",
                )
                save_rank_cdf_plot(
                    graph_diagnostics_h_dir / f"rank2_cdf_{global_step:07d}.png",
                    stats_h.ranks2,
                    stats_h.k,
                    "2-hop rank CDF",
                )
                save_neff_violin_plot(
                    graph_diagnostics_h_dir / f"neff_violin_{global_step:07d}.png",
                    stats_h.neff1,
                    stats_h.neff2,
                )
                save_in_degree_hist_plot(
                    graph_diagnostics_h_dir / f"in_degree_hist_{global_step:07d}.png",
                    stats_h.in_degree,
                )
                save_edge_consistency_hist_plot(
                    graph_diagnostics_h_dir / f"edge_consistency_{global_step:07d}.png",
                    stats_h.edge_errors,
                    embedding_label="h",
                )
                update_graph_diagnostics_history(
                    graph_diagnostics_h_dir,
                    stats_h,
                    global_step,
                    metrics_dir / "graph_diagnostics_h.csv",
                )
                p_queries, p_targets, p_predictions = _run_graph_diag(
                    embedding_kind="p",
                    graph_cfg=cfg.graph_diagnostics,
                    model=model,
                    ema_model=ema_model,
                    graph_embeddings=graph_diag.graph_embeddings,
                    graph_preds=graph_diag.graph_preds,
                    graph_h_preds=graph_diag.graph_h_preds,
                    graph_h_states=graph_diag.graph_h_states,
                    ema_embeddings=graph_diag.ema_embeddings,
                    ema_h_states=graph_diag.ema_h_states,
                )
                stats_p = compute_graph_diagnostics_stats(
                    p_queries,
                    p_targets,
                    p_predictions,
                    graph_diag.next_index,
                    graph_diag.next2_index,
                    graph_diag.chunk_ids,
                    cfg.graph_diagnostics,
                    global_step,
                )
                save_rank_cdf_plot(
                    graph_diagnostics_p_dir / f"rank1_cdf_{global_step:07d}.png",
                    stats_p.ranks1,
                    stats_p.k,
                    "1-step rank CDF",
                )
                save_rank_cdf_plot(
                    graph_diagnostics_p_dir / f"rank2_cdf_{global_step:07d}.png",
                    stats_p.ranks2,
                    stats_p.k,
                    "2-hop rank CDF",
                )
                save_neff_violin_plot(
                    graph_diagnostics_p_dir / f"neff_violin_{global_step:07d}.png",
                    stats_p.neff1,
                    stats_p.neff2,
                )
                save_in_degree_hist_plot(
                    graph_diagnostics_p_dir / f"in_degree_hist_{global_step:07d}.png",
                    stats_p.in_degree,
                )
                save_edge_consistency_hist_plot(
                    graph_diagnostics_p_dir / f"edge_consistency_{global_step:07d}.png",
                    stats_p.edge_errors,
                    embedding_label="p",
                )
                update_graph_diagnostics_history(
                    graph_diagnostics_p_dir,
                    stats_p,
                    global_step,
                    metrics_dir / "graph_diagnostics_p.csv",
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
    model.eval()

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
