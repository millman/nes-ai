#!/usr/bin/env python3
"""Training loop for the JEPA world model with local/global specialization and SIGReg."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
from typing import Annotated, Any, Dict, Iterable, Iterator, List, Literal, Optional, Sequence, TextIO, Tuple, Union

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
from matplotlib.figure import Figure
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
from jepa_world_model.conv_encoder_decoder import VisualizationDecoder as ConvVisualizationDecoder
from jepa_world_model.loss_recon import (
    FocalL1Loss,
    HardnessWeightedL1Loss,
    HardnessWeightedMSELoss,
    HardnessWeightedMedianLoss,
    build_feature_pyramid,
    multi_scale_hardness_loss_box,
    multi_scale_hardness_loss_gaussian,
    multi_scale_recon_loss_box_mse,
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
from jepa_world_model.model import JEPAWorldModel
from jepa_world_model.vis import (
    describe_action_tensor,
)
from jepa_world_model.vis_embedding_projection import save_embedding_projection
from jepa_world_model.vis_input_batch import save_input_batch_visualization
from jepa_world_model.vis_rollout_batch import save_rollout_sequence_batch
from jepa_world_model.vis_temporal_pairs import save_temporal_pair_visualization
from jepa_world_model.actions import compress_actions_to_ids, decode_action_id
from jepa_world_model.config_diagnostics import DiagnosticsConfig, SpikeDiagnosticsConfig
from jepa_world_model.config_planning import PlanningDiagnosticsConfig
from jepa_world_model.pose_rollout import rollout_pose_sequence
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
from jepa_world_model.plots.plot_reachable_fraction_hist import (
    save_reachable_fraction_hist_plot,
)
from jepa_world_model.plots.plot_planning_graph import save_planning_graph_plot
from jepa_world_model.plots.plot_action_vector_field import (
    save_action_time_slice_plot,
    save_action_vector_field_plot,
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
from jepa_world_model.diagnostics_runner import (
    planning_outputs_enabled,
    run_diagnostics_step,
    run_planning_diagnostics_step,
)
from jepa_world_model.diagnostics.grid_overlay import (
    build_grid_overlay_frames,
    save_grid_overlay_frame_grid,
)
from jepa_world_model.diagnostics_utils import append_csv_row, should_use_z2h_init
from jepa_world_model.planning.planning_eval import (
    DIRECTION_ORDER,
    ActionDeltaStats,
    DatasetGraph,
    action_labels_from_vectors,
    bfs_plan,
    build_dataset_graph,
    cluster_latents,
    compute_action_delta_stats,
    delta_lattice_astar,
    PlanningTestResult,
    plot_action_stats,
    plot_action_strip,
    plot_grid_trace,
    plot_pca_path,
    reachable_fractions,
    run_plan_in_env,
)
from jepa_world_model.encoder_schedule import _derive_encoder_schedule, _suggest_encoder_schedule
from jepa_world_model.step_schedule import _parse_schedule, _should_run_schedule
from jepa_world_model.data import (
    PreloadedTrajectorySequenceDataset,
    TrajectorySequenceDataset,
    collate_batch,
    load_actions_for_trajectory,
)
from jepa_world_model.hard_sample_reservoir import HardSampleReservoir, inject_hard_examples_into_batch
from jepa_world_model.vis_rollout import (
    VisualizationSelection,
    VisualizationSequence,
    render_rollout_batch,
)
from gridworldkey_env import GridworldKeyEnv, create_env_with_theme



# ------------------------------------------------------------
# Model components
# ------------------------------------------------------------

VisualizationDecoder = ConvVisualizationDecoder
RenderMode = Literal["anchor_delta", "direct"]

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
    - loss_jepa_rep (representation alignment):
        * What: encoder z_{t+1} vs detached predictor z_hat_{t+1}.
        * Grads: into encoder only; predictor path is stop-grad.
        * Purpose: gently align perception with belief without collapsing dynamics into perception.
    - loss_h2z: hidden->z projection; z_hat_from_h vs z (detached); shapes hidden path without moving encoder targets.
    - loss_z2h: z->hidden init projection; h_hat_from_z0 vs h0 (detached); z inputs are also detached so this only trains the z->h projector.
    - loss_z2h_init_zero: z->hidden init projection; h_hat_from_z0 vs h1 from h0=0; trains z->h to match the
      dynamics "cold start" state.
    - loss_z2h_match_h: z->hidden projection; h_hat_from_z_t vs h_t (detached) for t>=1; aligns z->h with predictor h.
    - loss_h2z_delta: h->(z_{t+1}-z_t) prediction against detached z deltas; trains a delta head for anchored rollout rendering.
    - loss_pixel_delta: pixel delta reconstruction on decoded frames.
    Other losses (recon, sigreg, inverse dynamics, etc.) behave as before.
    """
    # Latent transition supervision: ẑ_{t+1} (from predictor) vs detached z_{t+1}; shapes encoder+predictor via z_t.
    jepa: float = 0.5
    # Representation alignment: z_{t+1} (encoder) vs detached ẑ_{t+1} (predictor).
    jepa_rep: float = 0.0
    # Multi-step open-loop JEPA rollout loss (self-fed z predictions).
    jepa_open_loop: float = 0.5

    sigreg: float = 0.05

    # Image/pixel reconstruction
    recon: float = 0.0
    recon_patch: float = 0.0
    recon_multi_gauss: float = 0.0
    recon_multi_box: float = 0.1
    recon_multi_box_mse: float = 0.0

    # Image/pixel reconstruction for ẑ: recon(ẑ_{t+1}) vs x_{t+1} (detached from encoder)
    recon_zhat_multi_box: float = 0.05

    # Pixel delta reconstruction loss: recon(z_{t+1}) - recon(z_t) vs x_{t+1} - x_t.
    pixel_delta: float = 0.0
    # Pixel delta reconstruction loss using multi-scale box weighting.
    pixel_delta_multi_box: float = 0.0

    # Project hidden→z: ẑ_from_h vs z (detached); shapes hidden path without pushing encoder targets.
    h2z: float = 0.0
    # Project z→hidden: ĥ_from_z vs h (detached); trains z->h projector.
    z2h: float = 0.0
    # Project z→hidden to match the 1-step dynamics target from h0=0.
    z2h_init_zero: float = 0.1
    # Project z→hidden to match predictor h_t for t>=1.
    z2h_match_h: float = 0.1
    # Predict z deltas from h (for z-anchor rollout rendering); trains h->Δz head.
    h2z_delta: float = 0.0

    # Inverse dynamics from consecutive z pairs (z_t, z_{t+1}).
    # Keep at 0: forces z to reveal actions, breaking loop-closure invariance.
    inverse_dynamics_z: float = 0.0
    # Inverse dynamics from consecutive h pairs (h_t, h_{t+1}).
    inverse_dynamics_h: float = 0.1
    # Inverse dynamics from consecutive p pairs (p_t, p_{t+1}).
    # Keep at 0: pushes p to encode action identity/scale, which conflicts with geometric invariance
    # (p should encode place/pose, allow no-motion steps, and stay bounded for planning).
    inverse_dynamics_p: float = 0.0
    # Inverse dynamics from consecutive Δp (pose deltas).
    # Encodes action identity in Δp without forcing pose to be action-conditioned.
    inverse_dynamics_dp: float = 0.0

    # Robust temporal smoothness on z (Huber on consecutive z distances).
    z_smooth: float = 0.1
    # Robust temporal smoothness on h (Huber on consecutive h distances).
    h_smooth: float = 0.0

    # -------------------------------------------------------------------------
    # Algebra losses for Mario: keep z light, push h to local translation + short composition, and make p algebraic for planning.
    # -------------------------------------------------------------------------
    # z: Level 1 fixed translation; weakly anchor action directions without overfitting perception.
    # h: Level 2 + light Level 3; state-conditioned deltas with short-horizon composition.
    # p: Level 3; a smoother planning space with consistent multi-step structure.
    # p_odometry note: in the odometry writeup, "p" refers to Δp (increment), while here p is pose.

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

    # Multi-scale box reconstruction on predicted z rollouts (decode z_roll vs x_{t+k}).
    # Keep at 0: ties z rollouts to action effects, which should live in h/p instead.
    rollout_recon_multi_box_z: float = 0.0

    # Pixel delta reconstruction on predicted z rollouts.
    # Keep at 0: ties z rollouts to action effects, which should live in h/p instead.
    rollout_recon_delta_z: float = 0.0

    # Pixel delta reconstruction with multi-scale box weighting on predicted z rollouts.
    # Keep at 0: ties z rollouts to action effects, which should live in h/p instead.
    rollout_recon_multi_box_delta_z: float = 0.0

    # Projection consistency on predicted z rollouts (enc(dec(z_roll)) vs z_roll).
    # Keep at 0: reinforces action-conditioned z trajectories, hurting loop closure.
    rollout_project_z: float = 0.0

    # --- H ---
    # State-conditioned delta alignment: h_{t+1} - h_t vs E(h_t, a_t).
    # Makes action effects locally predictable (supports momentum, contacts, and walls).
    action_delta_h: float = 0.1

    # Explicit additivity of state deltas across steps.
    # Promotes near-linear multi-step effects; keep light for Mario's nonlinearity.
    additivity_h: float = 0.1

    # k-step rollout consistency in h-space.
    # Encourages short-horizon compositionality in the dynamics state.
    rollout_kstep_h: float = 0.1

    # k-step rollout delta consistency in h-space.
    # Encourages predicted h changes to match teacher deltas over short horizons.
    rollout_kstep_delta_h: float = 0.0

    # Pixel reconstruction on predicted h rollouts (decode h->z_roll vs x_{t+k}) with detached z inputs.
    rollout_recon_h: float = 0.0

    # Multi-scale box reconstruction on predicted h rollouts with detached z inputs.
    rollout_recon_multi_box_h: float = 0.0

    # NOOP should produce near-zero ΔH.
    noop_residual_dh: float = 0.1
    # Same-frame identity on h: ||h_{t+1} - h_t|| when consecutive frames are pixel-identical.
    same_frame_h_identity: float = 0.1
    # Orthogonality leak penalty: discourage NOOP mean delta from aligning with move-action mean deltas.
    noop_move_orth_h: float = 0.1

    # --- P ---
    # State-conditioned delta alignment (ΔP): Δp_t vs observed pose delta.
    # p_odometry: 1-step odometry consistency on ΔP (action algebra lives in ΔP).
    action_delta_dp: float = 0.0

    # Explicit additivity of ΔP across steps (action algebra lives in ΔP).
    # p_odometry: short-horizon composition/additivity of increments.
    additivity_dp: float = 0.0

    # k-step rollout consistency in P-space (pose accumulation).
    # p_odometry: k-step odometry consistency (multi-step integration of ΔP).
    rollout_kstep_p: float = 0.0

    # Pose scale anchoring (distribution-level).
    # p_odometry: scale/magnitude anchor to limit drift.
    scale_dp: float = 0.0

    # Goal-conditioned ranking loss on the pose rollout.
    # Keeps pose useful for planning geometry while avoiding action algebra constraints.
    # p_odometry: planning geometry / ranking head over the pose space.
    geometry_rank_p: float = 0.0

    # Anchor ΔP using ΔZ (project Δz -> Δp).
    # Provides a direct perceptual anchor for pose increments.
    dz_anchor_dp: float = 0.0

    # Soft loop closure: nearby z should imply nearby p.
    loop_closure_p: float = 0.0

    # NOOP should produce near-zero ΔP.
    noop_residual_dp: float = 0.0

    # Distance correlation between z and p (repulsive + attractive).
    distance_corr_p: float = 0.0

    # --- Inverse-cycle ---
    # Learn inverse actions in h (cycle in h), then enforce inverse/cancel in ΔP.
    inverse_cycle_h: float = 0.0
    inverse_cycle_dp: float = 0.0


@dataclass
class LossSigRegConfig:
    projections: int = 64


@dataclass
class LossLoopClosureConfig:
    samples_per_seq: int = 6
    min_gap: int = 2
    tau_scale: float = 1.0


@dataclass
class LossDistanceCorrelationConfig:
    samples_per_seq: int = 32
    min_gap: int = 2
    eps: float = 1e-6
    tau_scale: float = 1.0


@dataclass
class LossZTemporalSmoothConfig:
    delta: Annotated[
        float,
        tyro.conf.arg(
            help=(
                "Huber threshold for z temporal smoothness; with cosine distance in [0,2], 0.3 treats small motions as smooth."
            )
        ),
    ] = 0.3


@dataclass
class LossHTemporalSmoothConfig:
    delta: Annotated[
        float,
        tyro.conf.arg(
            help="Huber threshold for h temporal smoothness (L2 distance); smaller values push h to be more stable."
        ),
    ] = 0.3


@dataclass
class LossJEPAOpenLoopConfig:
    weighting: Annotated[
        Literal["hyperbolic", "uniform"],
        tyro.conf.arg(help="Step weighting for JEPA open-loop loss (hyperbolic = k/K, uniform = 1)."),
    ] = "uniform"


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
class ScaleDPConfig:
    mode: str = "median"
    target: float = 1.0
    trim: float = 0.2

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
    sample_sequences: int = 128
    knn_k_values: Tuple[int, ...] = (1, 2, 5, 10)
    knn_chunk_size: int = 512
    min_action_count: int = 5
    stability_delta: int = 1


@dataclass
class TrainConfig:
    data_root: Path = Path("data.gwbasic_rand_corner_loops2")
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
    ] = "10:10 50:1000 100:10000 200:None"
    plan_schedule: Annotated[
        Union[str, Tuple[Tuple[int, Optional[int]], ...]],
        tyro.conf.arg(
            help=(
                "Planning schedule entries use every_steps:max_step (or None for no cap). "
                "Example: '10:100 50:1000 100:10000 200:None'. Commas or spaces separate entries."
            )
        ),
    ] = "10:10 50:1000 100:10000 200:None"
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
    rollout_horizon: int = 8

    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.03
    device: Optional[str] = "mps"
    use_soap: bool = False
    detect_anomaly: bool = False

    # Loss configuration
    loss_weights: LossWeights = field(default_factory=LossWeights)
    loss_normalization_enabled: bool = False
    normalize_losses: NormalizeLossesConfig = field(default_factory=NormalizeLossesConfig)
    z_norm: Annotated[
        bool,
        tyro.conf.arg(help="Normalize encoder/predicted z to unit norm globally."),
    ] = False
    detach_decoder: bool = False
    detach_z_from_h_and_p: bool = True
    force_h_zero: Annotated[
        bool,
        tyro.conf.arg(help="Force hidden state to zero in rollouts (z-only dynamics)."),
    ] = False
    render_mode: Annotated[
        RenderMode,
        tyro.conf.arg(help="Rendering mode for rollout decode losses: anchor_delta or direct."),
    ] = "direct"

    # Specific losses
    sigreg: LossSigRegConfig = field(default_factory=LossSigRegConfig)
    z_smooth: LossZTemporalSmoothConfig = field(default_factory=LossZTemporalSmoothConfig)
    h_smooth: LossHTemporalSmoothConfig = field(default_factory=LossHTemporalSmoothConfig)
    loop_closure_p: LossLoopClosureConfig = field(default_factory=LossLoopClosureConfig)
    distance_corr_p: LossDistanceCorrelationConfig = field(default_factory=LossDistanceCorrelationConfig)
    jepa_open_loop: LossJEPAOpenLoopConfig = field(default_factory=LossJEPAOpenLoopConfig)
    geometry: LossGeometryConfig = field(default_factory=LossGeometryConfig)
    patch_recon: LossReconPatchConfig = field(default_factory=LossReconPatchConfig)
    recon_multi_gauss: LossMultiScaleGaussReconConfig = field(default_factory=LossMultiScaleGaussReconConfig)
    recon_multi_box: LossMultiScaleBoxReconConfig = field(default_factory=LossMultiScaleBoxReconConfig)
    recon_zhat_multi_box: LossMultiScaleBoxReconConfig = field(default_factory=LossMultiScaleBoxReconConfig)
    recon_multi_box_mse: LossMultiScaleBoxReconConfig = field(default_factory=LossMultiScaleBoxReconConfig)
    scale_dp: ScaleDPConfig = field(default_factory=ScaleDPConfig)

    # Visualization
    vis: VisConfig = field(default_factory=VisConfig)
    hard_example: HardExampleConfig = field(default_factory=HardExampleConfig)
    debug_visualization: DebugVisualization = field(default_factory=DebugVisualization)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    planning_diagnostics: PlanningDiagnosticsConfig = field(default_factory=PlanningDiagnosticsConfig)
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


# ------------------------------------------------------------
# Loss utilities
# ------------------------------------------------------------

RECON_LOSS = nn.MSELoss()
JEPA_LOSS = nn.MSELoss()
INVERSE_DYNAMICS_LOSS = nn.BCEWithLogitsLoss()


def z_vector_loss(pred: torch.Tensor, target: torch.Tensor, use_cosine: bool) -> torch.Tensor:
    if pred.shape != target.shape:
        raise AssertionError("z_vector_loss requires matching shapes.")
    if use_cosine:
        return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
    return F.mse_loss(pred, target)


def _rollout_render_latent(
    model: JEPAWorldModel,
    z_anchor: torch.Tensor,
    h_next: torch.Tensor,
    render_mode: RenderMode,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if render_mode == "direct":
        return model.h_to_z(h_next), None
    if render_mode == "anchor_delta":
        if model.h2z_delta is None:
            raise AssertionError("render_mode anchor_delta requires model.h2z_delta.")
        delta = model.h2z_delta(h_next)
        return z_anchor + delta, delta
    raise AssertionError(f"Unknown render_mode: {render_mode}")


def jepa_loss(
    model: JEPAWorldModel,
    outputs: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    use_z2h_init: bool = False,
    detach_z_inputs: bool = False,
    force_h_zero: bool = False,
    use_cosine_z: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """JEPA loss using predictor conditioned on z, h, and action."""
    embeddings = outputs["embeddings"]
    if embeddings.shape[1] < 2:
        raise AssertionError("JEPA loss requires at least two timesteps.")
    embeddings_for_rollout = embeddings.detach() if detach_z_inputs else embeddings
    z_preds, h_preds, h_states = rollout_teacher_forced(
        model,
        embeddings_for_rollout,
        actions,
        use_z2h_init=use_z2h_init,
        force_h_zero=force_h_zero,
    )
    target = embeddings[:, 1:].detach()
    return z_vector_loss(z_preds, target, use_cosine_z), z_preds, h_preds, h_states


def jepa_open_loop_loss(
    model: JEPAWorldModel,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    rollout_horizon: int,
    warmup_frames: int,
    use_z2h_init: bool = False,
    detach_z_inputs: bool = False,
    force_h_zero: bool = False,
    use_cosine_z: bool = False,
    weighting: Literal["hyperbolic", "uniform"] = "hyperbolic",
) -> torch.Tensor:
    """Open-loop multi-step JEPA loss using self-fed predictions."""
    if rollout_horizon <= 0:
        raise AssertionError("rollout_horizon must be > 0 for jepa_open_loop_loss.")
    b, t, _ = embeddings.shape
    if t < 2:
        raise AssertionError("jepa_open_loop_loss requires at least two timesteps.")
    warmup = max(min(warmup_frames, t - 1), 0)
    if warmup >= t - 1:
        raise AssertionError("warmup_frames leaves no rollout steps for jepa_open_loop_loss.")
    start = warmup
    embeddings_for_rollout = embeddings.detach() if detach_z_inputs else embeddings
    z_current = embeddings_for_rollout[:, start]
    if force_h_zero:
        h_current = embeddings.new_zeros((b, model.state_dim))
    else:
        h_current = (
            model.z_to_h(embeddings[:, start].detach())
            if use_z2h_init
            else embeddings.new_zeros((b, model.state_dim))
        )
    max_h = min(rollout_horizon, t - start - 1)
    if max_h <= 0:
        raise AssertionError("jepa_open_loop_loss produced no steps; check rollout_horizon or warmup_frames.")
    total = embeddings.new_tensor(0.0)
    weight_total = embeddings.new_tensor(0.0)
    steps = 0
    for offset in range(max_h):
        act = actions[:, start + offset]
        h_next = model.predictor(z_current, h_current, act)
        z_next = model.h_to_z(h_next)
        target = embeddings[:, start + offset + 1].detach()
        if weighting == "uniform":
            step_weight = embeddings.new_tensor(1.0)
        elif weighting == "hyperbolic":
            step_weight = embeddings.new_tensor((offset + 1) / max_h)
        else:
            raise AssertionError(f"Unknown JEPA open-loop weighting: {weighting}")
        total = total + step_weight * z_vector_loss(z_next, target, use_cosine_z)
        weight_total = weight_total + step_weight
        steps += 1
        z_current = z_next
        if force_h_zero:
            h_current = embeddings.new_zeros((b, model.state_dim))
        else:
            h_current = h_next
    if steps <= 0:
        raise AssertionError("jepa_open_loop_loss produced no steps; check rollout_horizon or warmup_frames.")
    if weight_total.item() <= 0:
        raise AssertionError("jepa_open_loop_loss produced non-positive weight sum; check rollout_horizon.")
    return total / weight_total


def _rollout_pose(
    model: JEPAWorldModel,
    h_states: torch.Tensor,
    actions: torch.Tensor,
    z_embeddings: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate Δp via the pose delta model to produce a pose rollout."""
    if z_embeddings.ndim != 3:
        raise AssertionError("z_embeddings must have shape [B, T, D].")
    if z_embeddings.shape[0] != h_states.shape[0]:
        raise AssertionError("z_embeddings and h_states must share the batch dimension.")
    if z_embeddings.shape[1] != h_states.shape[1]:
        raise AssertionError("z_embeddings and h_states must share the time dimension.")
    pose_pred, pose_deltas = rollout_pose_sequence(model, h_states, actions)
    pose_obs = pose_pred
    return pose_obs, pose_pred, pose_deltas


def rollout_z_loss(
    model: JEPAWorldModel,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    h_states: torch.Tensor,
    rollout_horizon: int,
    warmup_frames: int,
    use_z2h_init: bool,
    force_h_zero: bool = False,
    use_cosine_z: bool = False,
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
    start = warmup
    current = embeddings[:, start]
    max_h = min(rollout_horizon, t - start - 1)
    if max_h <= 0:
        raise AssertionError("rollout_z_loss produced no steps; check rollout_horizon or warmup_frames.")
    if force_h_zero:
        h_current = embeddings.new_zeros((b, model.state_dim))
    else:
        h_current = model.z_to_h(embeddings[:, start].detach()) if use_z2h_init else h_states[:, start]
    for offset in range(max_h):
        act = actions[:, start + offset]
        h_next = model.predictor(current, h_current, act)
        pred = model.h_to_z(h_next)
        target_step = embeddings[:, start + offset + 1].detach()
        total = total + z_vector_loss(pred, target_step, use_cosine_z)
        steps += 1
        current = pred
        if force_h_zero:
            h_current = embeddings.new_zeros((b, model.state_dim))
        else:
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
    render_mode: RenderMode,
    use_z2h_init: bool,
    force_h_zero: bool = False,
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
    start = warmup
    z_anchor = embeddings[:, start].detach()
    current = z_anchor if render_mode == "anchor_delta" else embeddings[:, start]
    if force_h_zero:
        h_current = embeddings.new_zeros((b, model.state_dim))
    else:
        h_current = model.z_to_h(embeddings[:, start].detach()) if use_z2h_init else h_states[:, start]
    max_h = min(rollout_horizon, t - start - 1)
    if max_h <= 0:
        raise AssertionError("rollout_recon_z_loss produced no steps; check rollout_horizon or warmup_frames.")
    for offset in range(max_h):
        act = actions[:, start + offset]
        h_next = model.predictor(current, h_current, act)
        pred = model.h_to_z(h_next)
        z_render, _ = _rollout_render_latent(model, z_anchor, h_next, render_mode)
        decoded = decoder(z_render)
        target_frame = frames[:, start + offset + 1]
        total = total + RECON_LOSS(decoded, target_frame)
        steps += 1
        current = pred
        if force_h_zero:
            h_current = embeddings.new_zeros((b, model.state_dim))
        else:
            h_current = h_next
    if steps <= 0:
        raise AssertionError("rollout_recon_z_loss produced no steps; check rollout_horizon or warmup_frames.")
    return total / steps


def rollout_recon_h_loss(
    model: JEPAWorldModel,
    decoder: VisualizationDecoder,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    frames: torch.Tensor,
    h_states: torch.Tensor,
    rollout_horizon: int,
    warmup_frames: int,
    render_mode: RenderMode,
    use_z2h_init: bool,
) -> torch.Tensor:
    if rollout_horizon <= 0:
        raise AssertionError("rollout_horizon must be > 0 for rollout_recon_h_loss.")
    b, t, _ = embeddings.shape
    if t < 2:
        raise AssertionError("rollout_recon_h_loss requires at least two timesteps.")
    total = embeddings.new_tensor(0.0)
    steps = 0
    warmup = max(min(warmup_frames, t - 1), 0)
    if warmup >= t - 1:
        raise AssertionError("warmup_frames leaves no rollout steps for rollout_recon_h_loss.")
    start = warmup
    z_anchor = embeddings[:, start].detach()
    current = z_anchor
    h_current = model.z_to_h(embeddings[:, start].detach()) if use_z2h_init else h_states[:, start]
    max_h = min(rollout_horizon, t - start - 1)
    if max_h <= 0:
        raise AssertionError("rollout_recon_h_loss produced no steps; check rollout_horizon or warmup_frames.")
    for offset in range(max_h):
        act = actions[:, start + offset]
        h_next = model.predictor(current, h_current, act)
        pred = model.h_to_z(h_next)
        z_render, delta = _rollout_render_latent(model, z_anchor, h_next, render_mode)
        decoded = decoder(z_render)
        target_frame = frames[:, start + offset + 1]
        total = total + RECON_LOSS(decoded, target_frame)
        if render_mode == "anchor_delta":
            assert delta is not None, "anchor_delta render_mode requires delta output."
            delta_target = embeddings[:, start + offset + 1] - z_anchor
            total = total + F.mse_loss(delta, delta_target.detach())
        steps += 1
        current = pred
        h_current = h_next
    if steps <= 0:
        raise AssertionError("rollout_recon_h_loss produced no steps; check rollout_horizon or warmup_frames.")
    return total / steps


def rollout_recon_multi_box_z_loss(
    model: JEPAWorldModel,
    decoder: VisualizationDecoder,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    frames: torch.Tensor,
    h_states: torch.Tensor,
    rollout_horizon: int,
    warmup_frames: int,
    cfg: LossMultiScaleBoxReconConfig,
    render_mode: RenderMode,
    use_z2h_init: bool,
    force_h_zero: bool = False,
) -> torch.Tensor:
    if rollout_horizon <= 0:
        raise AssertionError("rollout_horizon must be > 0 for rollout_recon_multi_box_z_loss.")
    b, t, _ = embeddings.shape
    if t < 2:
        raise AssertionError("rollout_recon_multi_box_z_loss requires at least two timesteps.")
    total = embeddings.new_tensor(0.0)
    steps = 0
    warmup = max(min(warmup_frames, t - 1), 0)
    if warmup >= t - 1:
        raise AssertionError("warmup_frames leaves no rollout steps for rollout_recon_multi_box_z_loss.")
    start = warmup
    z_anchor = embeddings[:, start].detach()
    current = z_anchor
    if force_h_zero:
        h_current = embeddings.new_zeros((b, model.state_dim))
    else:
        h_current = model.z_to_h(z_anchor) if use_z2h_init else h_states[:, start]
    max_h = min(rollout_horizon, t - start - 1)
    if max_h <= 0:
        raise AssertionError(
            "rollout_recon_multi_box_z_loss produced no steps; check rollout_horizon or warmup_frames."
        )
    for offset in range(max_h):
        act = actions[:, start + offset]
        h_next = model.predictor(current, h_current, act)
        pred = model.h_to_z(h_next)
        z_render, _ = _rollout_render_latent(model, z_anchor, h_next, render_mode)
        decoded = decoder(z_render)
        target_frame = frames[:, start + offset + 1]
        total = total + multi_scale_recon_loss_box(decoded, target_frame, cfg)
        steps += 1
        current = pred
        if force_h_zero:
            h_current = embeddings.new_zeros((b, model.state_dim))
        else:
            h_current = h_next
    if steps <= 0:
        raise AssertionError(
            "rollout_recon_multi_box_z_loss produced no steps; check rollout_horizon or warmup_frames."
        )
    return total / steps


def rollout_recon_multi_box_h_loss(
    model: JEPAWorldModel,
    decoder: VisualizationDecoder,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    frames: torch.Tensor,
    h_states: torch.Tensor,
    rollout_horizon: int,
    warmup_frames: int,
    cfg: LossMultiScaleBoxReconConfig,
    render_mode: RenderMode,
    use_z2h_init: bool,
) -> torch.Tensor:
    if rollout_horizon <= 0:
        raise AssertionError("rollout_horizon must be > 0 for rollout_recon_multi_box_h_loss.")
    b, t, _ = embeddings.shape
    if t < 2:
        raise AssertionError("rollout_recon_multi_box_h_loss requires at least two timesteps.")
    total = embeddings.new_tensor(0.0)
    steps = 0
    warmup = max(min(warmup_frames, t - 1), 0)
    if warmup >= t - 1:
        raise AssertionError("warmup_frames leaves no rollout steps for rollout_recon_multi_box_h_loss.")
    start = warmup
    z_anchor = embeddings[:, start].detach()
    current = z_anchor
    h_current = model.z_to_h(z_anchor) if use_z2h_init else h_states[:, start]
    max_h = min(rollout_horizon, t - start - 1)
    if max_h <= 0:
        raise AssertionError(
            "rollout_recon_multi_box_h_loss produced no steps; check rollout_horizon or warmup_frames."
        )
    for offset in range(max_h):
        act = actions[:, start + offset]
        h_next = model.predictor(current, h_current, act)
        pred = model.h_to_z(h_next)
        z_render, delta = _rollout_render_latent(model, z_anchor, h_next, render_mode)
        decoded = decoder(z_render)
        target_frame = frames[:, start + offset + 1]
        total = total + multi_scale_recon_loss_box(decoded, target_frame, cfg)
        if render_mode == "anchor_delta":
            assert delta is not None, "anchor_delta render_mode requires delta output."
            delta_target = embeddings[:, start + offset + 1] - z_anchor
            total = total + F.mse_loss(delta, delta_target.detach())
        steps += 1
        current = pred
        h_current = h_next
    return total / steps


def rollout_recon_delta_z_loss(
    model: JEPAWorldModel,
    decoder: VisualizationDecoder,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    frames: torch.Tensor,
    h_states: torch.Tensor,
    rollout_horizon: int,
    warmup_frames: int,
    render_mode: RenderMode,
    use_z2h_init: bool,
    force_h_zero: bool = False,
) -> torch.Tensor:
    if rollout_horizon <= 0:
        raise AssertionError("rollout_horizon must be > 0 for rollout_recon_delta_z_loss.")
    b, t, _ = embeddings.shape
    if t < 2:
        raise AssertionError("rollout_recon_delta_z_loss requires at least two timesteps.")
    total = embeddings.new_tensor(0.0)
    steps = 0
    warmup = max(min(warmup_frames, t - 1), 0)
    if warmup >= t - 1:
        raise AssertionError("warmup_frames leaves no rollout steps for rollout_recon_delta_z_loss.")
    start = warmup
    z_anchor = embeddings[:, start].detach()
    current = z_anchor if render_mode == "anchor_delta" else embeddings[:, start]
    if force_h_zero:
        h_current = embeddings.new_zeros((b, model.state_dim))
    else:
        h_current = model.z_to_h(z_anchor) if use_z2h_init else h_states[:, start]
    prev_decoded = decoder(z_anchor if render_mode == "anchor_delta" else embeddings[:, start])
    max_h = min(rollout_horizon, t - start - 1)
    if max_h <= 0:
        raise AssertionError("rollout_recon_delta_z_loss produced no steps; check rollout_horizon or warmup_frames.")
    for offset in range(max_h):
        act = actions[:, start + offset]
        h_next = model.predictor(current, h_current, act)
        pred = model.h_to_z(h_next)
        z_render, _ = _rollout_render_latent(model, z_anchor, h_next, render_mode)
        decoded = decoder(z_render)
        target_delta = frames[:, start + offset + 1] - frames[:, start + offset]
        pred_delta = decoded - prev_decoded
        total = total + RECON_LOSS(pred_delta, target_delta)
        steps += 1
        current = pred
        if force_h_zero:
            h_current = embeddings.new_zeros((b, model.state_dim))
        else:
            h_current = h_next
        prev_decoded = decoded
    if steps <= 0:
        raise AssertionError("rollout_recon_delta_z_loss produced no steps; check rollout_horizon or warmup_frames.")
    return total / steps


def rollout_recon_multi_box_delta_z_loss(
    model: JEPAWorldModel,
    decoder: VisualizationDecoder,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    frames: torch.Tensor,
    h_states: torch.Tensor,
    rollout_horizon: int,
    warmup_frames: int,
    cfg: LossMultiScaleBoxReconConfig,
    render_mode: RenderMode,
    use_z2h_init: bool,
    force_h_zero: bool = False,
) -> torch.Tensor:
    if rollout_horizon <= 0:
        raise AssertionError("rollout_horizon must be > 0 for rollout_recon_multi_box_delta_z_loss.")
    b, t, _ = embeddings.shape
    if t < 2:
        raise AssertionError("rollout_recon_multi_box_delta_z_loss requires at least two timesteps.")
    total = embeddings.new_tensor(0.0)
    steps = 0
    warmup = max(min(warmup_frames, t - 1), 0)
    if warmup >= t - 1:
        raise AssertionError("warmup_frames leaves no rollout steps for rollout_recon_multi_box_delta_z_loss.")
    start = warmup
    z_anchor = embeddings[:, start].detach()
    current = z_anchor if render_mode == "anchor_delta" else embeddings[:, start]
    if force_h_zero:
        h_current = embeddings.new_zeros((b, model.state_dim))
    else:
        h_current = model.z_to_h(z_anchor) if use_z2h_init else h_states[:, start]
    prev_decoded = decoder(z_anchor if render_mode == "anchor_delta" else embeddings[:, start])
    max_h = min(rollout_horizon, t - start - 1)
    if max_h <= 0:
        raise AssertionError(
            "rollout_recon_multi_box_delta_z_loss produced no steps; check rollout_horizon or warmup_frames."
        )
    for offset in range(max_h):
        act = actions[:, start + offset]
        h_next = model.predictor(current, h_current, act)
        pred = model.h_to_z(h_next)
        z_render, _ = _rollout_render_latent(model, z_anchor, h_next, render_mode)
        decoded = decoder(z_render)
        target_delta = frames[:, start + offset + 1] - frames[:, start + offset]
        pred_delta = decoded - prev_decoded
        total = total + multi_scale_recon_loss_box(pred_delta, target_delta, cfg)
        steps += 1
        current = pred
        if force_h_zero:
            h_current = embeddings.new_zeros((b, model.state_dim))
        else:
            h_current = h_next
        prev_decoded = decoded
    if steps <= 0:
        raise AssertionError(
            "rollout_recon_multi_box_delta_z_loss produced no steps; check rollout_horizon or warmup_frames."
        )
    return total / steps


def rollout_project_z_loss(
    model: JEPAWorldModel,
    decoder: VisualizationDecoder,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    h_states: torch.Tensor,
    rollout_horizon: int,
    warmup_frames: int,
    render_mode: RenderMode,
    use_z2h_init: bool,
    force_h_zero: bool = False,
    use_cosine_z: bool = False,
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
    start = warmup
    z_anchor = embeddings[:, start].detach()
    current = z_anchor if render_mode == "anchor_delta" else embeddings[:, start]
    if force_h_zero:
        h_current = embeddings.new_zeros((b, model.state_dim))
    else:
        h_current = model.z_to_h(z_anchor) if use_z2h_init else h_states[:, start]
    max_h = min(rollout_horizon, t - start - 1)
    if max_h <= 0:
        raise AssertionError("rollout_project_z_loss produced no steps; check rollout_horizon or warmup_frames.")
    for offset in range(max_h):
        act = actions[:, start + offset]
        h_next = model.predictor(current, h_current, act)
        pred = model.h_to_z(h_next)
        z_render, _ = _rollout_render_latent(model, z_anchor, h_next, render_mode)
        decoded = decoder(z_render)
        z_back = model.encoder(decoded)
        total = total + z_vector_loss(z_back, z_render, use_cosine_z)
        steps += 1
        current = pred
        if force_h_zero:
            h_current = embeddings.new_zeros((b, model.state_dim))
        else:
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
    use_z2h_init: bool,
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
    start = warmup
    current = embeddings[:, start]
    max_h = min(rollout_horizon, t - start - 1)
    if max_h <= 0:
        raise AssertionError("rollout_h_loss produced no steps; check rollout_horizon or warmup_frames.")
    z_anchor = current
    h_current = model.z_to_h(z_anchor) if use_z2h_init else h_states[:, start]
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


def rollout_h_delta_loss(
    model: JEPAWorldModel,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    h_states: torch.Tensor,
    rollout_horizon: int,
    warmup_frames: int,
    use_z2h_init: bool,
) -> torch.Tensor:
    if rollout_horizon <= 1:
        raise AssertionError("rollout_horizon must be > 1 for rollout_h_delta_loss.")
    b, t, _ = embeddings.shape
    if t < 2:
        raise AssertionError("rollout_h_delta_loss requires at least two timesteps.")
    total = embeddings.new_tensor(0.0)
    steps = 0
    warmup = max(min(warmup_frames, t - 1), 0)
    if warmup >= t - 1:
        raise AssertionError("warmup_frames leaves no rollout steps for rollout_h_delta_loss.")
    start = warmup
    current = embeddings[:, start]
    max_h = min(rollout_horizon, t - start - 1)
    if max_h <= 0:
        raise AssertionError("rollout_h_delta_loss produced no steps; check rollout_horizon or warmup_frames.")
    h_current = model.z_to_h(embeddings[:, start].detach()) if use_z2h_init else h_states[:, start]
    for offset in range(max_h):
        act = actions[:, start + offset]
        h_next = model.predictor(current, h_current, act)
        pred = model.h_to_z(h_next)
        target_delta = h_states[:, start + offset + 1].detach() - h_states[:, start + offset].detach()
        total = total + F.mse_loss(h_next - h_current, target_delta)
        steps += 1
        current = pred
        h_current = h_next
    if steps <= 0:
        raise AssertionError("rollout_h_delta_loss produced no steps; check rollout_horizon or warmup_frames.")
    return total / steps


def rollout_p_loss(
    pose_pred: torch.Tensor,
    pose_obs: torch.Tensor,
    pose_deltas: torch.Tensor,
    rollout_horizon: int,
    start_frame: int,
) -> torch.Tensor:
    if rollout_horizon <= 1:
        raise AssertionError("rollout_horizon must be > 1 for rollout_p_loss.")
    if pose_pred.shape[1] < 2:
        raise AssertionError("rollout_p_loss requires at least two timesteps.")
    if pose_deltas.shape[1] < 1:
        raise AssertionError("rollout_p_loss requires at least one delta.")
    if pose_obs.shape[1] < 2:
        raise AssertionError("rollout_p_loss requires at least two observation timesteps.")
    max_k = min(rollout_horizon, pose_pred.shape[1] - 1, pose_obs.shape[1] - 1)
    if max_k < 2:
        raise AssertionError("rollout_horizon leaves no rollout steps for rollout_p_loss.")
    total = pose_pred.new_tensor(0.0)
    steps = 0
    delta_cumsum = pose_deltas.cumsum(dim=1)
    for k in range(2, max_k + 1):
        end = min(pose_pred.shape[1], pose_obs.shape[1]) - k
        if end <= start_frame:
            continue
        delta_end = delta_cumsum[:, start_frame + k - 1 : end + k - 1]
        if start_frame > 0:
            delta_start = delta_cumsum[:, start_frame - 1 : end - 1]
            delta_sum_k = delta_end - delta_start
        else:
            delta_sum_k = delta_end
        pose_hat = pose_pred[:, start_frame:end] + delta_sum_k
        pose_target_k = pose_obs[:, start_frame + k : end + k]
        if pose_hat.numel() > 0:
            total = total + F.mse_loss(pose_hat, pose_target_k.detach())
            steps += 1
    if steps <= 0:
        raise AssertionError("rollout_p_loss produced no steps; check rollout_horizon or start_frame.")
    return total / steps


def multi_scale_recon_loss_gauss(
    recon: torch.Tensor,
    target: torch.Tensor,
    cfg: LossMultiScaleGaussReconConfig,
) -> torch.Tensor:
    """Multi-scale hardness-weighted reconstruction over an image pyramid."""
    if recon.ndim == 4:
        recon = recon.unsqueeze(1)
    if target.ndim == 4:
        target = target.unsqueeze(1)
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
    if recon.ndim == 4:
        recon = recon.unsqueeze(1)
    if target.ndim == 4:
        target = target.unsqueeze(1)
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
    use_cosine_z: bool,
) -> torch.Tensor:
    if h_states.numel() == 0:
        raise AssertionError("h2z_loss requires non-empty h_states.")
    if embeddings.shape[1] - start <= 0:
        raise AssertionError("h2z_loss requires at least one timestep after start.")
    h_stack = h_states[:, start:]
    z_hat_from_h = model.h_to_z(h_stack)
    return z_vector_loss(z_hat_from_h, embeddings[:, start:].detach(), use_cosine_z)


def h2z_delta_loss(
    model: JEPAWorldModel,
    embeddings: torch.Tensor,
    h_states: torch.Tensor,
    start: int,
) -> torch.Tensor:
    if model.h2z_delta is None:
        raise AssertionError("h2z_delta_loss requires model.h2z_delta.")
    if embeddings.shape[1] < start + 2:
        raise AssertionError("h2z_delta_loss requires at least two timesteps after start.")
    h_stack = h_states[:, start + 1 :].detach()
    if h_stack.numel() == 0:
        raise AssertionError("h2z_delta_loss requires non-empty h states after start.")
    delta_target = embeddings[:, start + 1 :] - embeddings[:, start:-1]
    delta_pred = model.h2z_delta(h_stack)
    return F.mse_loss(delta_pred, delta_target.detach())


def z2h_loss(
    model: JEPAWorldModel,
    embeddings: torch.Tensor,
    start: int,
    use_cosine_z: bool,
) -> torch.Tensor:
    """Aux z->h->z consistency loss that trains z_to_h only.

    Targets are detached to keep the encoder fixed for this aux term.
    h_to_z is temporarily frozen so the only trainable path is z_to_h.
    This avoids circular "teacher moves to match student" behavior.
    """
    if embeddings.shape[1] < 1:
        raise AssertionError("z2h_loss requires at least one timestep.")
    if embeddings.shape[1] <= start:
        raise AssertionError("z2h_loss requires a valid start index.")
    # Use post-warmup z_t as fixed targets for the aux consistency loss.
    z_stack = embeddings[:, start:].detach()
    # Trainable student path: z -> h.
    h_hat_from_z = model.z_to_h(z_stack)
    # Freeze h_to_z so gradients only update z_to_h for this aux term.
    h2z_requires_grad = [param.requires_grad for param in model.h_to_z.parameters()]
    for param in model.h_to_z.parameters():
        param.requires_grad_(False)
    z_hat = model.h_to_z(h_hat_from_z)
    for param, requires_grad in zip(model.h_to_z.parameters(), h2z_requires_grad):
        param.requires_grad_(requires_grad)
    # Compare predicted z_hat to the detached target z_t.
    return z_vector_loss(z_hat, z_stack, use_cosine_z)


def z2h_init_zero_loss(
    model: JEPAWorldModel,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    start: int,
) -> torch.Tensor:
    """Train z->h to match the 1-step h state from a zero-initialized rollout."""
    if embeddings.shape[1] <= start:
        raise AssertionError("z2h_init_zero_loss requires a valid start index.")
    if actions.shape[1] <= start:
        raise AssertionError("z2h_init_zero_loss requires actions for the start index.")
    z0 = embeddings[:, start].detach()
    a0 = actions[:, start]
    h0 = embeddings.new_zeros((embeddings.shape[0], model.state_dim))
    h1_target = model.predictor(z0, h0, a0).detach()
    h0_hat = model.z_to_h(z0)
    return F.mse_loss(h0_hat, h1_target)


def z2h_match_h_loss(
    model: JEPAWorldModel,
    embeddings: torch.Tensor,
    h_states: torch.Tensor,
    start: int,
) -> torch.Tensor:
    """Align z->h with predictor h_t for t>=1 using detached targets."""
    if embeddings.shape[1] < 2:
        raise AssertionError("z2h_match_h_loss requires at least two timesteps.")
    if embeddings.shape[1] != h_states.shape[1]:
        raise AssertionError("z2h_match_h_loss requires matching z/h timesteps.")
    start_idx = max(1, start)
    if embeddings.shape[1] <= start_idx:
        raise AssertionError("z2h_match_h_loss requires a valid start index.")
    z_stack = embeddings[:, start_idx:].detach()
    h_target = h_states[:, start_idx:].detach()
    h_hat = model.z_to_h(z_stack)
    return F.mse_loss(h_hat, h_target)


def geometry_rank_loss(
    pose: torch.Tensor,
    cfg: LossGeometryConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pose.numel() == 0:
        raise AssertionError("geometry_rank_loss requires non-empty pose.")
    return geometry_ranking_loss(pose, cfg)


def pixel_delta_loss(recon: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    if frames.shape[1] <= 1:
        raise AssertionError("pixel_delta_loss requires at least two frames.")
    delta_target = frames[:, 1:] - frames[:, :-1]
    delta_pred = recon[:, 1:] - recon[:, :-1]
    return RECON_LOSS(delta_pred, delta_target)


def pixel_delta_multi_box_loss(
    recon: torch.Tensor,
    frames: torch.Tensor,
    cfg: LossMultiScaleBoxReconConfig,
) -> torch.Tensor:
    if frames.shape[1] <= 1:
        raise AssertionError("pixel_delta_multi_box_loss requires at least two frames.")
    delta_target = frames[:, 1:] - frames[:, :-1]
    delta_pred = recon[:, 1:] - recon[:, :-1]
    return multi_scale_recon_loss_box(delta_pred, delta_target, cfg)


def z_smooth_huber_loss(
    z_embeddings: torch.Tensor,
    delta: float,
    use_cosine: bool,
) -> torch.Tensor:
    if z_embeddings.shape[1] < 2:
        raise AssertionError("z_smooth_huber_loss requires at least two timesteps.")
    z_t = z_embeddings[:, :-1]
    z_tp1 = z_embeddings[:, 1:]
    if use_cosine:
        distances = 1.0 - F.cosine_similarity(z_t, z_tp1, dim=-1)
    else:
        distances = torch.norm(z_tp1 - z_t, dim=-1)
    delta_tensor = z_embeddings.new_tensor(delta)
    loss = torch.where(
        distances <= delta_tensor,
        0.5 * distances * distances,
        delta_tensor * (distances - 0.5 * delta_tensor),
    )
    return loss.mean()


def h_smooth_huber_loss(
    h_states: torch.Tensor,
    delta: float,
    *,
    start: int,
) -> torch.Tensor:
    if h_states.shape[1] < start + 2:
        raise AssertionError("h_smooth_huber_loss requires at least two timesteps after start.")
    h_t = h_states[:, start:-1]
    h_tp1 = h_states[:, start + 1 :]
    distances = torch.norm(h_tp1 - h_t, dim=-1)
    delta_tensor = h_states.new_tensor(delta)
    loss = torch.where(
        distances <= delta_tensor,
        0.5 * distances * distances,
        delta_tensor * (distances - 0.5 * delta_tensor),
    )
    return loss.mean()


def scale_dp_loss(pose: torch.Tensor, start: int, cfg: ScaleDPConfig) -> torch.Tensor:
    if pose.shape[1] <= start + 1:
        return pose.new_tensor(0.0)
    deltas = pose[:, start + 1 :] - pose[:, start:-1]
    norms = deltas.norm(dim=-1).reshape(-1)
    if norms.numel() == 0:
        return pose.new_tensor(0.0)
    mode = cfg.mode.lower()
    if mode == "median":
        scale = norms.median()
    elif mode == "trimmed_mean":
        trim = min(max(cfg.trim, 0.0), 0.49)
        if trim > 0.0:
            sorted_norms, _ = norms.sort()
            n = sorted_norms.numel()
            k = int(math.floor(trim * n))
            if k * 2 >= n:
                scale = sorted_norms.mean()
            else:
                scale = sorted_norms[k : n - k].mean()
        else:
            scale = norms.mean()
    else:
        raise ValueError(f"scale_dp.mode must be 'median' or 'trimmed_mean', got {cfg.mode!r}.")
    return (scale - cfg.target) ** 2


def loop_closure_p_loss(
    pose: torch.Tensor,
    z_embeddings: torch.Tensor,
    start: int,
    cfg: LossLoopClosureConfig,
) -> torch.Tensor:
    if pose.shape[1] <= start + 1:
        return pose.new_tensor(0.0)
    if z_embeddings.shape[1] <= start + 1:
        return pose.new_tensor(0.0)
    seq_len = pose.shape[1] - start
    if seq_len <= cfg.min_gap:
        return pose.new_tensor(0.0)
    samples = max(int(cfg.samples_per_seq), 1)
    idx_i = []
    idx_j = []
    for _ in range(samples):
        for _ in range(64):
            i = int(torch.randint(0, seq_len, (1,), device=pose.device).item())
            j = int(torch.randint(0, seq_len, (1,), device=pose.device).item())
            if abs(i - j) >= cfg.min_gap:
                idx_i.append(i)
                idx_j.append(j)
                break
        else:
            raise AssertionError("loop_closure_p_loss could not sample valid index pairs.")
    idx_i_t = torch.tensor(idx_i, device=pose.device, dtype=torch.long)
    idx_j_t = torch.tensor(idx_j, device=pose.device, dtype=torch.long)

    pose_seq = pose[:, start:]
    z_seq = z_embeddings[:, start:]
    z_seq_det = z_seq.detach()
    dz = z_seq_det[:, 1:] - z_seq_det[:, :-1]
    dz_norm = dz.norm(dim=-1).reshape(-1)
    if dz_norm.numel() == 0:
        return pose.new_tensor(0.0)
    tau = dz_norm.median() * float(cfg.tau_scale)
    if tau.item() <= 1e-6:
        tau = pose.new_tensor(1.0)

    p_i = pose_seq[:, idx_i_t]
    p_j = pose_seq[:, idx_j_t]
    z_i = z_seq_det[:, idx_i_t]
    z_j = z_seq_det[:, idx_j_t]
    z_dist = (z_i - z_j).norm(dim=-1)
    p_dist = (p_i - p_j).norm(dim=-1)
    weights = torch.exp(-z_dist / (tau + 1e-8))
    return (weights * p_dist.pow(2)).mean()


def distance_corr_p_loss(
    pose: torch.Tensor,
    z_embeddings: torch.Tensor,
    start: int,
    cfg: LossDistanceCorrelationConfig,
) -> torch.Tensor:
    if pose.shape[1] <= start + 1:
        return pose.new_tensor(0.0)
    if z_embeddings.shape[1] <= start + 1:
        return pose.new_tensor(0.0)
    seq_len = pose.shape[1] - start
    if seq_len <= cfg.min_gap:
        return pose.new_tensor(0.0)
    samples = max(int(cfg.samples_per_seq), 1)
    idx_i = []
    idx_j = []
    for _ in range(samples):
        for _ in range(64):
            i = int(torch.randint(0, seq_len, (1,), device=pose.device).item())
            j = int(torch.randint(0, seq_len, (1,), device=pose.device).item())
            if abs(i - j) >= cfg.min_gap:
                idx_i.append(i)
                idx_j.append(j)
                break
        else:
            raise AssertionError("distance_corr_p_loss could not sample valid index pairs.")
    idx_i_t = torch.tensor(idx_i, device=pose.device, dtype=torch.long)
    idx_j_t = torch.tensor(idx_j, device=pose.device, dtype=torch.long)

    pose_seq = pose[:, start:]
    z_seq = z_embeddings[:, start:].detach()

    p_i = pose_seq[:, idx_i_t]
    p_j = pose_seq[:, idx_j_t]
    z_i = z_seq[:, idx_i_t]
    z_j = z_seq[:, idx_j_t]

    p_dist = (p_i - p_j).norm(dim=-1).reshape(-1)
    z_dist = (z_i - z_j).norm(dim=-1).reshape(-1)

    p_centered = p_dist - p_dist.mean()
    z_centered = z_dist - z_dist.mean()
    cov = (p_centered * z_centered).mean()
    denom = (p_centered.std(unbiased=False) * z_centered.std(unbiased=False)).clamp_min(cfg.eps)
    corr = cov / denom
    return 1.0 - corr




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
    loss_jepa_rep = x_frames.new_tensor(0.0)
    loss_jepa_open_loop = x_frames.new_tensor(0.0)
    loss_sigreg = x_frames.new_tensor(0.0)

    # z (Recon) losses
    loss_recon = x_frames.new_tensor(0.0)
    loss_recon_multi_gauss = x_frames.new_tensor(0.0)
    loss_recon_multi_box = x_frames.new_tensor(0.0)
    loss_recon_zhat_multi_box = x_frames.new_tensor(0.0)
    loss_recon_multi_box_mse = x_frames.new_tensor(0.0)
    loss_recon_patch = x_frames.new_tensor(0.0)

    # z (Auxiliary) losses
    loss_pixel_delta = x_frames.new_tensor(0.0)
    loss_pixel_delta_multi_box = x_frames.new_tensor(0.0)
    loss_z_smooth = x_frames.new_tensor(0.0)
    loss_h_smooth = x_frames.new_tensor(0.0)

    # h (Hidden) losses
    loss_h2z = x_frames.new_tensor(0.0)
    loss_z2h = x_frames.new_tensor(0.0)
    loss_z2h_init_zero = x_frames.new_tensor(0.0)
    loss_z2h_match_h = x_frames.new_tensor(0.0)
    loss_h2z_delta = x_frames.new_tensor(0.0)

    # p (Geometry) losses
    loss_geometry_rank_p = x_frames.new_tensor(0.0)
    geometry_rank_p_accuracy = x_frames.new_tensor(0.0)
    geometry_rank_p_pairs = x_frames.new_tensor(0.0)

    # Inverse dynamics losses
    loss_inverse_dynamics_z = x_frames.new_tensor(0.0)
    loss_inverse_dynamics_h = x_frames.new_tensor(0.0)
    loss_inverse_dynamics_p = x_frames.new_tensor(0.0)
    loss_inverse_dynamics_dp = x_frames.new_tensor(0.0)
    loss_action_delta_z = x_frames.new_tensor(0.0)
    loss_action_delta_h = x_frames.new_tensor(0.0)
    loss_rollout_kstep_z = x_frames.new_tensor(0.0)
    loss_rollout_kstep_h = x_frames.new_tensor(0.0)
    loss_rollout_kstep_p = x_frames.new_tensor(0.0)
    loss_rollout_recon_z = x_frames.new_tensor(0.0)
    loss_rollout_recon_multi_box_z = x_frames.new_tensor(0.0)
    loss_rollout_recon_delta_z = x_frames.new_tensor(0.0)
    loss_rollout_recon_multi_box_delta_z = x_frames.new_tensor(0.0)
    loss_rollout_project_z = x_frames.new_tensor(0.0)
    loss_rollout_recon_h = x_frames.new_tensor(0.0)
    loss_rollout_recon_multi_box_h = x_frames.new_tensor(0.0)
    loss_rollout_kstep_delta_h = x_frames.new_tensor(0.0)
    loss_inverse_cycle_h = x_frames.new_tensor(0.0)
    loss_inverse_cycle_dp = x_frames.new_tensor(0.0)
    loss_additivity_h = x_frames.new_tensor(0.0)
    loss_action_delta_dp = x_frames.new_tensor(0.0)
    loss_dz_anchor_dp = x_frames.new_tensor(0.0)
    loss_loop_closure_p = x_frames.new_tensor(0.0)
    loss_noop_residual_dp = x_frames.new_tensor(0.0)
    loss_noop_residual_dh = x_frames.new_tensor(0.0)
    loss_same_frame_h_identity = x_frames.new_tensor(0.0)
    loss_noop_move_orth_h = x_frames.new_tensor(0.0)
    loss_additivity_dp = x_frames.new_tensor(0.0)
    loss_scale_dp = x_frames.new_tensor(0.0)
    loss_distance_corr_p = x_frames.new_tensor(0.0)

    # -------------------------------------------------------------------------
    # Calculate required inputs
    # -------------------------------------------------------------------------

    # NOTE: We may not actually need each of these inputs, but it majorly simplifies the
    #   conditionals, since some inputs are needed in multiple branches.

    encode_outputs = model.encode_sequence(x_frames)
    z_embeddings = encode_outputs["embeddings"]
    z_embeddings_raw = encode_outputs["embeddings_raw"]
    use_cosine_z = cfg.z_norm
    loss_jepa_raw, z_preds, h_preds, h_states = jepa_loss(
        model,
        encode_outputs,
        a_seq,
        use_z2h_init=should_use_z2h_init(weights),
        detach_z_inputs=cfg.detach_z_from_h_and_p,
        force_h_zero=cfg.force_h_zero,
        use_cosine_z=use_cosine_z,
    )

    z_for_decoder = z_embeddings.detach() if cfg.detach_decoder else z_embeddings
    x_recon = decoder(z_for_decoder)

    seq_len = z_embeddings.shape[1]
    warmup_frames = max(model.cfg.warmup_frames_h, 0)

    # Warmup before applying hidden-state losses to avoid cold-start transients.
    start_frame = max(min(warmup_frames, seq_len - 1), 0)

    use_action_delta_dp = (
        weights.action_delta_dp > 0
        or weights.additivity_dp > 0
        or weights.rollout_kstep_p > 0
        or weights.dz_anchor_dp > 0
    )
    use_pose_rollout = (
        use_action_delta_dp
        or weights.geometry_rank_p > 0
        or weights.inverse_dynamics_p > 0
        or weights.inverse_dynamics_dp > 0
        or weights.scale_dp > 0
        or weights.dz_anchor_dp > 0
        or weights.loop_closure_p > 0
        or weights.distance_corr_p > 0
    )
    pose_obs: Optional[torch.Tensor] = None
    pose_pred: Optional[torch.Tensor] = None
    pose_deltas: Optional[torch.Tensor] = None
    pose_pred_rollout: Optional[torch.Tensor] = None
    if use_pose_rollout:
        pose_obs, pose_pred, pose_deltas = _rollout_pose(
            model,
            h_states,
            a_seq,
            z_embeddings=z_embeddings,
        )
    if use_action_delta_dp:
        h_pred_states = h_preds
        if h_pred_states.shape[1] + 1 == h_states.shape[1]:
            h_pred_states = torch.cat([h_states[:, :1], h_pred_states], dim=1)
        pose_pred_rollout, _, _ = _rollout_pose(
            model,
            h_pred_states,
            a_seq,
            z_embeddings=z_embeddings,
        )

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
        p_norm_mean = x_frames.new_tensor(0.0)
        p_norm_std = x_frames.new_tensor(0.0)
        p_norm_max = x_frames.new_tensor(0.0)
        if pose_pred is not None:
            p_embeddings = pose_pred.detach()
            p_norm_mean, p_norm_std, p_norm_max = _norm_stats(p_embeddings)
        elif pose_obs is not None:
            p_embeddings = pose_obs.detach()
            p_norm_mean, p_norm_std, p_norm_max = _norm_stats(p_embeddings)

    # -------------------------------------------------------------------------
    # Losses
    # -------------------------------------------------------------------------

    # Core JEPA losses
    if weights.jepa > 0:
        loss_jepa = loss_jepa_raw
    if weights.jepa_rep > 0:
        loss_jepa_rep = z_vector_loss(z_embeddings[:, 1:], z_preds.detach(), use_cosine_z)
    if weights.jepa_open_loop > 0:
        loss_jepa_open_loop = jepa_open_loop_loss(
            model,
            z_embeddings,
            a_seq,
            cfg.rollout_horizon,
            warmup_frames,
            use_z2h_init=should_use_z2h_init(weights),
            detach_z_inputs=cfg.detach_z_from_h_and_p,
            force_h_zero=cfg.force_h_zero,
            use_cosine_z=use_cosine_z,
            weighting=cfg.jepa_open_loop.weighting,
        )

    if weights.sigreg > 0:
        loss_sigreg = sigreg_loss(z_embeddings_raw, cfg.sigreg.projections)
    if weights.z_smooth > 0:
        loss_z_smooth = z_smooth_huber_loss(
            z_embeddings,
            cfg.z_smooth.delta,
            cfg.z_norm,
        )
    if weights.h_smooth > 0:
        loss_h_smooth = h_smooth_huber_loss(
            h_states,
            cfg.h_smooth.delta,
            start=start_frame,
        )

    # z (Recon) Reconstruction and pixel-space losses
    if weights.recon > 0:
        loss_recon = RECON_LOSS(x_recon, x_frames)

    if weights.recon_multi_gauss > 0:
        loss_recon_multi_gauss = multi_scale_recon_loss_gauss(x_recon, x_frames, cfg.recon_multi_gauss)

    if weights.recon_multi_box > 0:
        loss_recon_multi_box = multi_scale_recon_loss_box(x_recon, x_frames, cfg.recon_multi_box)

    if weights.recon_zhat_multi_box > 0:
        z_preds_hat, _, _ = rollout_teacher_forced(
            model,
            z_embeddings.detach(),
            a_seq,
            use_z2h_init=should_use_z2h_init(weights),
            force_h_zero=cfg.force_h_zero,
        )
        x_recon_hat = decoder(z_preds_hat)
        loss_recon_zhat_multi_box = multi_scale_recon_loss_box(
            x_recon_hat,
            x_frames[:, 1:],
            cfg.recon_zhat_multi_box,
        )

    if weights.recon_multi_box_mse > 0:
        loss_recon_multi_box_mse = multi_scale_recon_loss_box_mse(
            x_recon,
            x_frames,
            cfg.recon_multi_box_mse.kernel_sizes,
            cfg.recon_multi_box_mse.lambdas,
            cfg.recon_multi_box_mse.strides,
        )

    if weights.recon_patch > 0:
        loss_recon_patch = patch_recon_loss(
            x_recon,
            x_frames,
            cfg.patch_recon.patch_sizes,
            loss_fn=RECON_LOSS,
        )

    if weights.pixel_delta > 0:
        loss_pixel_delta = pixel_delta_loss(x_recon, x_frames)
    if weights.pixel_delta_multi_box > 0:
        loss_pixel_delta_multi_box = pixel_delta_multi_box_loss(x_recon, x_frames, cfg.recon_multi_box)

    # z (Auxiliary) losses

    # h (Hidden) Hidden-state dynamics and cross-projection losses
    if weights.h2z > 0:
        loss_h2z = h2z_loss(model, z_embeddings, h_states, start_frame, use_cosine_z)
    if weights.z2h > 0:
        loss_z2h = z2h_loss(model, z_embeddings, start_frame, use_cosine_z)
    if weights.z2h_init_zero > 0:
        loss_z2h_init_zero = z2h_init_zero_loss(model, z_embeddings, a_seq, start_frame)
    if weights.z2h_match_h > 0:
        loss_z2h_match_h = z2h_match_h_loss(model, z_embeddings, h_states, start_frame)
    if weights.h2z_delta > 0:
        loss_h2z_delta = h2z_delta_loss(model, z_embeddings, h_states, start_frame)

    # h (Auxiliary)
    # p (Geometry)
    if weights.geometry_rank_p > 0:
        # p_odometry: planning geometry/ranking head (goal-conditioned ordering in pose space).
        if pose_pred is None:
            raise AssertionError("geometry_rank_p requires pose rollouts to be computed.")
        loss_geometry_rank_p, geometry_rank_p_accuracy, geometry_rank_p_pairs = geometry_rank_loss(
            pose_pred,
            cfg.geometry,
        )

    dh_pred: Optional[torch.Tensor] = None
    if weights.action_delta_h > 0 or weights.additivity_h > 0:
        if model.h_action_delta_projector is None:
            raise AssertionError("h_action_delta_projector is disabled but action_delta_h/additivity_h is enabled.")
        assert h_states.shape[1] >= 2, "action_delta_h/additivity_h requires at least two h timesteps."
        h_curr = h_states[:, start_frame:-1]
        h_next = h_states[:, start_frame + 1 :]
        a_curr = a_seq[:, start_frame:-1]
        assert h_curr.numel() > 0, "action_delta_h/additivity_h requires non-empty h_curr."
        dh_pred = model.h_action_delta_projector(h_curr, a_curr)

    if weights.action_delta_h > 0:
        assert dh_pred is not None, "action_delta_h requires dh_pred."
        dh_target = h_next - h_curr
        loss_action_delta_h = F.mse_loss(dh_pred, dh_target)

    if weights.action_delta_dp > 0:
        # p_odometry: action-prototype alignment in ΔP.
        if model.dp_action_delta_projector is None:
            raise AssertionError("dp_action_delta_projector is disabled but action_delta_dp is enabled.")
        if pose_deltas is None:
            raise AssertionError("action_delta_dp requires pose deltas to be computed.")
        dp_pred = pose_deltas[:, start_frame:]
        assert dp_pred.shape[1] >= 1, "action_delta_dp requires at least one delta."
        actions_dp = a_seq[:, start_frame : start_frame + dp_pred.shape[1]]
        delta_target = model.dp_action_delta_projector(actions_dp)
        loss_action_delta_dp = F.mse_loss(dp_pred, delta_target.detach())

    if weights.dz_anchor_dp > 0:
        # p_odometry: anchor ΔP using ΔZ projected into pose delta space.
        if model.dz_to_dp_projector is None:
            raise AssertionError("dz_to_dp_projector is disabled but dz_anchor_dp is enabled.")
        if pose_deltas is None:
            raise AssertionError("dz_anchor_dp requires pose deltas to be computed.")
        assert z_embeddings.shape[1] >= start_frame + 2, "dz_anchor_dp requires at least two z timesteps."
        dz = z_embeddings[:, start_frame + 1 :] - z_embeddings[:, start_frame:-1]
        dp_target = model.dz_to_dp_projector(dz)
        dp_pred = pose_deltas[:, start_frame:]
        assert dp_pred.shape == dp_target.shape, "dz_anchor_dp requires matching ΔP/ΔZ shapes."
        loss_dz_anchor_dp = F.mse_loss(dp_pred, dp_target.detach())

    if weights.loop_closure_p > 0:
        # p_odometry: soft loop closure using z similarity.
        if pose_pred is None:
            raise AssertionError("loop_closure_p requires pose rollouts to be computed.")
        loss_loop_closure_p = loop_closure_p_loss(
            pose_pred,
            z_embeddings,
            start_frame,
            cfg.loop_closure_p,
        )

    if weights.distance_corr_p > 0:
        # p_odometry: distance correlation between z and p.
        if pose_pred is None:
            raise AssertionError("distance_corr_p requires pose rollouts to be computed.")
        loss_distance_corr_p = distance_corr_p_loss(
            pose_pred,
            z_embeddings,
            start_frame,
            cfg.distance_corr_p,
        )

    if weights.noop_residual_dp > 0:
        # p_odometry: NOOP should not change pose (ΔP ≈ 0).
        if pose_deltas is None:
            raise AssertionError("noop_residual_dp requires pose deltas to be computed.")
        action_ids = compress_actions_to_ids(a_seq.detach().cpu().numpy())
        if action_ids.ndim == 1:
            action_ids = action_ids.reshape(a_seq.shape[0], a_seq.shape[1])
        if action_ids.shape[1] == pose_deltas.shape[1] + 1:
            action_ids = action_ids[:, :-1]
        if action_ids.shape[1] != pose_deltas.shape[1]:
            raise AssertionError("noop_residual_dp requires action/p_delta shape alignment.")
        noop_mask = torch.as_tensor(action_ids == 0, device=pose_deltas.device)
        dp_slice = pose_deltas[:, start_frame:]
        mask_slice = noop_mask[:, start_frame : start_frame + dp_slice.shape[1]]
        if mask_slice.any():
            dp_noop = dp_slice[mask_slice]
            loss_noop_residual_dp = (dp_noop.norm(dim=-1) ** 2).mean()
        else:
            loss_noop_residual_dp = pose_deltas.new_tensor(0.0)

    if weights.noop_residual_dh > 0:
        # NOOP should not change hidden state (ΔH ≈ 0).
        action_ids = compress_actions_to_ids(a_seq.detach().cpu().numpy())
        if action_ids.ndim == 1:
            action_ids = action_ids.reshape(a_seq.shape[0], a_seq.shape[1])
        if action_ids.shape[1] == h_states.shape[1]:
            action_ids = action_ids[:, :-1]
        if action_ids.shape[1] != h_states.shape[1] - 1:
            raise AssertionError("noop_residual_dh requires action/h_state shape alignment.")
        noop_mask = torch.as_tensor(action_ids == 0, device=h_states.device)
        dh_slice = h_states[:, start_frame + 1 :] - h_states[:, start_frame:-1]
        mask_slice = noop_mask[:, start_frame : start_frame + dh_slice.shape[1]]
        if mask_slice.any():
            dh_noop = dh_slice[mask_slice]
            loss_noop_residual_dh = (dh_noop.norm(dim=-1) ** 2).mean()
        else:
            loss_noop_residual_dh = h_states.new_tensor(0.0)

    if weights.same_frame_h_identity > 0 or weights.noop_move_orth_h > 0:
        action_ids = compress_actions_to_ids(a_seq.detach().cpu().numpy())
        if action_ids.ndim == 1:
            action_ids = action_ids.reshape(a_seq.shape[0], a_seq.shape[1])
        if action_ids.shape[1] == h_states.shape[1]:
            action_ids = action_ids[:, :-1]
        if action_ids.shape[1] != h_states.shape[1] - 1:
            raise AssertionError("same_frame_h_identity/noop_move_orth_h requires action/h_state shape alignment.")
        dh_slice = h_states[:, start_frame + 1 :] - h_states[:, start_frame:-1]
        action_ids_t = torch.as_tensor(action_ids, device=h_states.device)
        action_ids_slice = action_ids_t[:, start_frame : start_frame + dh_slice.shape[1]]

        if weights.same_frame_h_identity > 0:
            frame_delta = (x_frames[:, 1:] - x_frames[:, :-1]).abs().amax(dim=(2, 3, 4))
            same_frame_mask = frame_delta <= 1e-6
            same_frame_slice = same_frame_mask[:, start_frame : start_frame + dh_slice.shape[1]]
            if same_frame_slice.any():
                loss_same_frame_h_identity = dh_slice[same_frame_slice].norm(dim=-1).mean()
            else:
                loss_same_frame_h_identity = h_states.new_tensor(0.0)

        if weights.noop_move_orth_h > 0:
            noop_mask = action_ids_slice == 0
            if noop_mask.any():
                mu_noop = dh_slice[noop_mask].mean(dim=0)
                move_ids = torch.unique(action_ids_slice[action_ids_slice != 0])
                penalties: List[torch.Tensor] = []
                margin = 0.1
                for aid in move_ids.tolist():
                    move_mask = action_ids_slice == aid
                    if not move_mask.any():
                        continue
                    mu_move = dh_slice[move_mask].mean(dim=0)
                    cos = F.cosine_similarity(mu_noop.unsqueeze(0), mu_move.unsqueeze(0), dim=-1).squeeze(0)
                    penalties.append(F.relu(cos.abs() - margin))
                if penalties:
                    loss_noop_move_orth_h = torch.stack(penalties).mean()
                else:
                    loss_noop_move_orth_h = h_states.new_tensor(0.0)
            else:
                loss_noop_move_orth_h = h_states.new_tensor(0.0)

    if weights.additivity_dp > 0:
        # p_odometry: additivity of ΔP increments (Δp_t + Δp_{t+1} ≈ Δp_{t:t+2}).
        if pose_pred is None or pose_deltas is None or pose_obs is None:
            raise AssertionError("additivity_dp requires pose rollouts to be computed.")
        assert pose_pred.shape[1] >= start_frame + 3, "additivity_dp requires at least three pose timesteps."
        assert pose_deltas.shape[1] >= start_frame + 2, "additivity_dp requires at least two deltas."
        delta_sum = pose_deltas[:, start_frame:-1] + pose_deltas[:, start_frame + 1 :]
        delta_target = pose_obs[:, start_frame + 2 :] - pose_obs[:, start_frame:-2]
        assert delta_sum.numel() > 0 and delta_target.numel() > 0, "additivity_dp requires non-empty tensors."
        loss_additivity_dp = F.mse_loss(delta_sum, delta_target.detach())

    if weights.rollout_kstep_p > 0:
        # p_odometry: k-step integration consistency.
        if pose_obs is None or pose_pred is None or pose_deltas is None:
            raise AssertionError("rollout_kstep_p requires pose rollouts to be computed.")
        assert pose_pred.shape[1] >= 2, "rollout_kstep_p requires at least two pose timesteps."
        assert pose_deltas.shape[1] >= 2, "rollout_kstep_p requires at least one delta."
        loss_rollout_kstep_p = rollout_p_loss(
            pose_pred,
            pose_obs,
            pose_deltas,
            cfg.rollout_horizon,
            start_frame,
        )

    if weights.scale_dp > 0:
        # p_odometry: scale anchor for pose drift control.
        if pose_pred is None:
            raise AssertionError("scale_dp requires pose observations to be computed.")
        assert pose_pred.shape[1] >= start_frame + 2, "scale_dp requires at least two pose timesteps after start."
        loss_scale_dp = scale_dp_loss(pose_pred, start_frame, cfg.scale_dp)

    # Inverse dynamics
    if weights.inverse_dynamics_z > 0:
        if model.inverse_dynamics_z is None:
            raise AssertionError("inverse_dynamics_z head is disabled but inverse_dynamics_z loss is enabled.")
        assert z_embeddings.shape[1] >= 2
        action_logits_z = model.inverse_dynamics_z(z_embeddings[:, :-1], z_embeddings[:, 1:])
        loss_inverse_dynamics_z = INVERSE_DYNAMICS_LOSS(action_logits_z, a_seq[:, :-1])

    if weights.inverse_dynamics_h > 0:
        if model.inverse_dynamics_h is None:
            raise AssertionError("inverse_dynamics_h head is disabled but inverse_dynamics_h loss is enabled.")
        assert h_states.shape[1] >= 2
        action_logits_h = model.inverse_dynamics_h(h_states[:, :-1], h_states[:, 1:])
        loss_inverse_dynamics_h = INVERSE_DYNAMICS_LOSS(action_logits_h, a_seq[:, :-1])

    if weights.inverse_dynamics_p > 0:
        if model.inverse_dynamics_p is None:
            raise AssertionError("inverse_dynamics_p head is disabled but inverse_dynamics_p loss is enabled.")
        if pose_pred is None:
            _, pose_pred, _ = _rollout_pose(model, h_states, a_seq, z_embeddings=z_embeddings)
        assert pose_pred.shape[1] >= 2
        p_for_inverse = pose_pred
        action_logits_p = model.inverse_dynamics_p(p_for_inverse[:, :-1], p_for_inverse[:, 1:])
        loss_inverse_dynamics_p = INVERSE_DYNAMICS_LOSS(action_logits_p, a_seq[:, :-1])

    if weights.inverse_dynamics_dp > 0:
        if model.inverse_dynamics_dp is None:
            raise AssertionError("inverse_dynamics_dp head is disabled but inverse_dynamics_dp loss is enabled.")
        if pose_deltas is None:
            raise AssertionError("inverse_dynamics_dp requires pose deltas to be computed.")
        dp_for_inverse = pose_deltas[:, start_frame:]
        assert dp_for_inverse.shape[1] >= 1, "inverse_dynamics_dp requires at least one delta."
        actions_dp = a_seq[:, start_frame : start_frame + dp_for_inverse.shape[1]]
        action_logits_dp = model.inverse_dynamics_dp(dp_for_inverse)
        loss_inverse_dynamics_dp = INVERSE_DYNAMICS_LOSS(action_logits_dp, actions_dp)

    if weights.action_delta_z > 0:
        if model.z_action_delta_projector is None:
            raise AssertionError("z_action_delta_projector is disabled but action_delta_z loss is enabled.")
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
            should_use_z2h_init(weights),
            force_h_zero=cfg.force_h_zero,
            use_cosine_z=use_cosine_z,
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
            cfg.render_mode,
            should_use_z2h_init(weights),
            force_h_zero=cfg.force_h_zero,
        )
    if weights.rollout_recon_multi_box_z > 0:
        loss_rollout_recon_multi_box_z = rollout_recon_multi_box_z_loss(
            model,
            decoder,
            z_embeddings,
            a_seq,
            x_frames,
            h_states,
            cfg.rollout_horizon,
            warmup_frames,
            cfg.recon_multi_box,
            cfg.render_mode,
            should_use_z2h_init(weights),
            force_h_zero=cfg.force_h_zero,
        )
    if weights.rollout_recon_delta_z > 0:
        loss_rollout_recon_delta_z = rollout_recon_delta_z_loss(
            model,
            decoder,
            z_embeddings,
            a_seq,
            x_frames,
            h_states,
            cfg.rollout_horizon,
            warmup_frames,
            cfg.render_mode,
            should_use_z2h_init(weights),
            force_h_zero=cfg.force_h_zero,
        )
    if weights.rollout_recon_multi_box_delta_z > 0:
        loss_rollout_recon_multi_box_delta_z = rollout_recon_multi_box_delta_z_loss(
            model,
            decoder,
            z_embeddings,
            a_seq,
            x_frames,
            h_states,
            cfg.rollout_horizon,
            warmup_frames,
            cfg.recon_multi_box,
            cfg.render_mode,
            should_use_z2h_init(weights),
            force_h_zero=cfg.force_h_zero,
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
            cfg.render_mode,
            should_use_z2h_init(weights),
            force_h_zero=cfg.force_h_zero,
            use_cosine_z=use_cosine_z,
        )
    if weights.rollout_recon_h > 0:
        loss_rollout_recon_h = rollout_recon_h_loss(
            model,
            decoder,
            z_embeddings,
            a_seq,
            x_frames,
            h_states,
            cfg.rollout_horizon,
            warmup_frames,
            cfg.render_mode,
            should_use_z2h_init(weights),
        )
    if weights.rollout_recon_multi_box_h > 0:
        loss_rollout_recon_multi_box_h = rollout_recon_multi_box_h_loss(
            model,
            decoder,
            z_embeddings,
            a_seq,
            x_frames,
            h_states,
            cfg.rollout_horizon,
            warmup_frames,
            cfg.recon_multi_box,
            cfg.render_mode,
            should_use_z2h_init(weights),
        )

    z_for_rollout_h: Optional[torch.Tensor] = None
    if weights.rollout_kstep_h > 0 or weights.rollout_kstep_delta_h > 0:
        z_for_rollout_h = z_embeddings.detach() if cfg.detach_z_from_h_and_p else z_embeddings
    if weights.rollout_kstep_h > 0:
        loss_rollout_kstep_h = rollout_h_loss(
            model,
            z_for_rollout_h,
            a_seq,
            h_states,
            cfg.rollout_horizon,
            warmup_frames,
            should_use_z2h_init(weights),
        )
    if weights.rollout_kstep_delta_h > 0:
        loss_rollout_kstep_delta_h = rollout_h_delta_loss(
            model,
            z_for_rollout_h,
            a_seq,
            h_states,
            cfg.rollout_horizon,
            warmup_frames,
            should_use_z2h_init(weights),
        )

    if weights.inverse_cycle_h > 0 or weights.inverse_cycle_dp > 0:
        if model.inverse_action_head is None:
            raise AssertionError("inverse_action_head is missing for inverse cycle losses.")
        assert h_states.shape[1] >= start_frame + 2, "inverse_cycle losses require at least two timesteps."
        z_for_inv = z_embeddings.detach() if cfg.detach_z_from_h_and_p else z_embeddings
        h_curr = h_states[:, start_frame:-1]
        z_curr = z_for_inv[:, start_frame:-1]
        a_curr = a_seq[:, start_frame:-1]
        if h_curr.numel() == 0:
            raise AssertionError("inverse_cycle losses require non-empty h_curr.")
        inv_logits = model.inverse_action_head(h_curr.detach(), a_curr.detach())
        a_inv_probs = F.softmax(inv_logits, dim=-1)
        action_basis = torch.eye(a_curr.shape[-1], device=a_curr.device, dtype=a_curr.dtype)
        a_inv = a_inv_probs @ action_basis
        h_next = model.predictor(z_curr, h_curr, a_curr)
        z_pred = model.h_to_z(h_next)
        h_back = model.predictor(z_pred, h_next, a_inv)
        if weights.inverse_cycle_h > 0:
            loss_inverse_cycle_h = F.mse_loss(h_back, h_curr.detach())
        if weights.inverse_cycle_dp > 0:
            if model.p_action_delta_projector is None:
                raise AssertionError("inverse_cycle_dp requires p_action_delta_projector.")
            if pose_pred is None:
                _, pose_pred, _ = _rollout_pose(
                    model,
                    h_states,
                    a_seq,
                    z_embeddings=z_embeddings,
                )
            p_curr = pose_pred[:, start_frame:-1]
            if model.cfg.pose_delta_detach_h:
                h_curr_for_p = h_curr.detach()
                h_next_for_p = h_next.detach()
            else:
                h_curr_for_p = h_curr
                h_next_for_p = h_next
            delta1 = model.p_action_delta_projector(p_curr, h_curr_for_p, a_curr)
            p1 = p_curr + delta1
            delta2 = model.p_action_delta_projector(p1, h_next_for_p, a_inv.detach())
            delta_sum = delta1 + delta2
            loss_inverse_cycle_dp = F.mse_loss(delta_sum, torch.zeros_like(delta_sum))

    if weights.additivity_h > 0:
        assert dh_pred is not None, "additivity_h requires dh_pred to be computed."
        assert dh_pred.shape[1] >= 2, "additivity_h requires at least two predicted deltas."
        dh_add_pred = dh_pred[:, :-1] + dh_pred[:, 1:]
        dh_add_target = h_states[:, start_frame + 2 :] - h_states[:, start_frame:-2]
        assert dh_add_pred.numel() > 0, "additivity_h requires non-empty delta tensors."
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
        + weights.jepa_rep * _scaled("loss_jepa_rep", loss_jepa_rep)
        + weights.jepa_open_loop * _scaled("loss_jepa_open_loop", loss_jepa_open_loop)
        + weights.sigreg * _scaled("loss_sigreg", loss_sigreg)
        + weights.z_smooth * _scaled("loss_z_smooth", loss_z_smooth)
        + weights.h2z * _scaled("loss_h2z", loss_h2z)
        + weights.z2h * _scaled("loss_z2h", loss_z2h)
        + weights.z2h_init_zero * _scaled("loss_z2h_init_zero", loss_z2h_init_zero)
        + weights.z2h_match_h * _scaled("loss_z2h_match_h", loss_z2h_match_h)
        + weights.h2z_delta * _scaled("loss_h2z_delta", loss_h2z_delta)
        + weights.geometry_rank_p * _scaled("loss_geometry_rank_p", loss_geometry_rank_p)
        + weights.recon * _scaled("loss_recon", loss_recon)
        + weights.recon_multi_gauss * _scaled("loss_recon_multi_gauss", loss_recon_multi_gauss)
        + weights.recon_multi_box * _scaled("loss_recon_multi_box", loss_recon_multi_box)
        + weights.recon_zhat_multi_box * _scaled("loss_recon_zhat_multi_box", loss_recon_zhat_multi_box)
        + weights.recon_multi_box_mse * _scaled("loss_recon_multi_box_mse", loss_recon_multi_box_mse)
        + weights.recon_patch * _scaled("loss_recon_patch", loss_recon_patch)
        + weights.pixel_delta * _scaled("loss_pixel_delta", loss_pixel_delta)
        + weights.pixel_delta_multi_box * _scaled("loss_pixel_delta_multi_box", loss_pixel_delta_multi_box)
        + weights.inverse_dynamics_z * _scaled("loss_inverse_dynamics_z", loss_inverse_dynamics_z)
        + weights.inverse_dynamics_h * _scaled("loss_inverse_dynamics_h", loss_inverse_dynamics_h)
        + weights.inverse_dynamics_p * _scaled("loss_inverse_dynamics_p", loss_inverse_dynamics_p)
        + weights.inverse_dynamics_dp * _scaled("loss_inverse_dynamics_dp", loss_inverse_dynamics_dp)
        + weights.action_delta_z * _scaled("loss_action_delta_z", loss_action_delta_z)
        + weights.action_delta_h * _scaled("loss_action_delta_h", loss_action_delta_h)
        + weights.rollout_kstep_z * _scaled("loss_rollout_kstep_z", loss_rollout_kstep_z)
        + weights.rollout_recon_z * _scaled("loss_rollout_recon_z", loss_rollout_recon_z)
        + weights.rollout_recon_multi_box_z
        * _scaled("loss_rollout_recon_multi_box_z", loss_rollout_recon_multi_box_z)
        + weights.rollout_recon_delta_z * _scaled("loss_rollout_recon_delta_z", loss_rollout_recon_delta_z)
        + weights.rollout_recon_multi_box_delta_z
        * _scaled("loss_rollout_recon_multi_box_delta_z", loss_rollout_recon_multi_box_delta_z)
        + weights.rollout_project_z * _scaled("loss_rollout_project_z", loss_rollout_project_z)
        + weights.rollout_recon_h * _scaled("loss_rollout_recon_h", loss_rollout_recon_h)
        + weights.rollout_recon_multi_box_h
        * _scaled("loss_rollout_recon_multi_box_h", loss_rollout_recon_multi_box_h)
        + weights.rollout_kstep_h * _scaled("loss_rollout_kstep_h", loss_rollout_kstep_h)
        + weights.rollout_kstep_delta_h * _scaled("loss_rollout_kstep_delta_h", loss_rollout_kstep_delta_h)
        + weights.inverse_cycle_h * _scaled("loss_inverse_cycle_h", loss_inverse_cycle_h)
        + weights.rollout_kstep_p * _scaled("loss_rollout_kstep_p", loss_rollout_kstep_p)
        + weights.additivity_h * _scaled("loss_additivity_h", loss_additivity_h)
        + weights.h_smooth * _scaled("loss_h_smooth", loss_h_smooth)
        + weights.scale_dp * _scaled("loss_scale_dp", loss_scale_dp)
        + weights.action_delta_dp * _scaled("loss_action_delta_dp", loss_action_delta_dp)
        + weights.dz_anchor_dp * _scaled("loss_dz_anchor_dp", loss_dz_anchor_dp)
        + weights.loop_closure_p * _scaled("loss_loop_closure_p", loss_loop_closure_p)
        + weights.distance_corr_p * _scaled("loss_distance_corr_p", loss_distance_corr_p)
        + weights.noop_residual_dp * _scaled("loss_noop_residual_dp", loss_noop_residual_dp)
        + weights.noop_residual_dh * _scaled("loss_noop_residual_dh", loss_noop_residual_dh)
        + weights.same_frame_h_identity * _scaled("loss_same_frame_h_identity", loss_same_frame_h_identity)
        + weights.noop_move_orth_h * _scaled("loss_noop_move_orth_h", loss_noop_move_orth_h)
        + weights.additivity_dp * _scaled("loss_additivity_dp", loss_additivity_dp)
        + weights.inverse_cycle_dp * _scaled("loss_inverse_cycle_dp", loss_inverse_cycle_dp)
    )

    world_grad_norm = 0.0
    decoder_grad_norm = 0.0
    if for_training and optimizer is not None:
        optimizer.zero_grad()
        anomaly_ctx = torch.autograd.set_detect_anomaly(True) if cfg.detect_anomaly else nullcontext()
        with anomaly_ctx:
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

    dp_norm_median = 0.0
    dz_to_dp_norm_median = 0.0
    with torch.no_grad():
        if pose_deltas is not None and pose_deltas.numel() > 0:
            dp_norms = pose_deltas.norm(dim=-1).reshape(-1)
            dp_norm_median = float(dp_norms.median().item())
        if model.dz_to_dp_projector is not None and z_embeddings.shape[1] >= start_frame + 2:
            dz = z_embeddings[:, start_frame + 1 :] - z_embeddings[:, start_frame:-1]
            if dz.numel() > 0:
                dp_from_dz = model.dz_to_dp_projector(dz)
                dz_norms = dp_from_dz.norm(dim=-1).reshape(-1)
                dz_to_dp_norm_median = float(dz_norms.median().item())

    metrics = {
        "loss_jepa": loss_jepa.item(),
        "loss_jepa_rep": loss_jepa_rep.item(),
        "loss_jepa_open_loop": loss_jepa_open_loop.item(),
        "loss_sigreg": loss_sigreg.item(),
        "loss_z_smooth": loss_z_smooth.item(),
        "loss_recon": loss_recon.item(),
        "loss_recon_multi_gauss": loss_recon_multi_gauss.item(),
        "loss_recon_multi_box": loss_recon_multi_box.item(),
        "loss_recon_zhat_multi_box": loss_recon_zhat_multi_box.item(),
        "loss_recon_multi_box_mse": loss_recon_multi_box_mse.item(),
        "loss_recon_patch": loss_recon_patch.item(),
        "loss_pixel_delta": loss_pixel_delta.item(),
        "loss_pixel_delta_multi_box": loss_pixel_delta_multi_box.item(),
        "loss_h2z": loss_h2z.item(),
        "loss_z2h": loss_z2h.item(),
        "loss_z2h_init_zero": loss_z2h_init_zero.item(),
        "loss_z2h_match_h": loss_z2h_match_h.item(),
        "loss_h2z_delta": loss_h2z_delta.item(),
        "loss_geometry_rank_p": loss_geometry_rank_p.item(),
        "geometry_rank_p_accuracy": geometry_rank_p_accuracy.item(),
        "geometry_rank_p_pairs": geometry_rank_p_pairs.item(),
        "loss_inverse_dynamics_z": loss_inverse_dynamics_z.item(),
        "loss_inverse_dynamics_h": loss_inverse_dynamics_h.item(),
        "loss_inverse_dynamics_p": loss_inverse_dynamics_p.item(),
        "loss_inverse_dynamics_dp": loss_inverse_dynamics_dp.item(),
        "loss_action_delta_z": loss_action_delta_z.item(),
        "loss_action_delta_h": loss_action_delta_h.item(),
        "loss_rollout_kstep_z": loss_rollout_kstep_z.item(),
        "loss_rollout_recon_z": loss_rollout_recon_z.item(),
        "loss_rollout_recon_multi_box_z": loss_rollout_recon_multi_box_z.item(),
        "loss_rollout_recon_delta_z": loss_rollout_recon_delta_z.item(),
        "loss_rollout_recon_multi_box_delta_z": loss_rollout_recon_multi_box_delta_z.item(),
        "loss_rollout_project_z": loss_rollout_project_z.item(),
        "loss_rollout_recon_h": loss_rollout_recon_h.item(),
        "loss_rollout_recon_multi_box_h": loss_rollout_recon_multi_box_h.item(),
        "loss_rollout_kstep_h": loss_rollout_kstep_h.item(),
        "loss_rollout_kstep_delta_h": loss_rollout_kstep_delta_h.item(),
        "loss_inverse_cycle_h": loss_inverse_cycle_h.item(),
        "loss_rollout_kstep_p": loss_rollout_kstep_p.item(),
        "loss_additivity_h": loss_additivity_h.item(),
        "loss_h_smooth": loss_h_smooth.item(),
        "loss_scale_dp": loss_scale_dp.item(),
        "loss_action_delta_dp": loss_action_delta_dp.item(),
        "loss_dz_anchor_dp": loss_dz_anchor_dp.item(),
        "loss_loop_closure_p": loss_loop_closure_p.item(),
        "loss_distance_corr_p": loss_distance_corr_p.item(),
        "loss_noop_residual_dp": loss_noop_residual_dp.item(),
        "loss_noop_residual_dh": loss_noop_residual_dh.item(),
        "loss_same_frame_h_identity": loss_same_frame_h_identity.item(),
        "loss_noop_move_orth_h": loss_noop_move_orth_h.item(),
        "loss_additivity_dp": loss_additivity_dp.item(),
        "loss_inverse_cycle_dp": loss_inverse_cycle_dp.item(),
        "z_norm_mean": z_norm_mean,
        "z_norm_std": z_norm_std,
        "z_norm_max": z_norm_max,
        "h_norm_mean": h_norm_mean,
        "h_norm_std": h_norm_std,
        "h_norm_max": h_norm_max,
        "p_norm_mean": p_norm_mean,
        "p_norm_std": p_norm_std,
        "p_norm_max": p_norm_max,
        "dp_norm_median": dp_norm_median,
        "dz_to_dp_norm_median": dz_to_dp_norm_median,
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


def _json_safe_value(value: object) -> object:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.item())
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


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
        "metrics": {key: _json_safe_value(val) for key, val in metrics.items()},
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
    if weights.jepa_rep <= 0:
        filtered.pop("loss_jepa_rep", None)
    if weights.jepa_open_loop <= 0:
        filtered.pop("loss_jepa_open_loop", None)
    if weights.sigreg <= 0:
        filtered.pop("loss_sigreg", None)
    if weights.z_smooth <= 0:
        filtered.pop("loss_z_smooth", None)
    if weights.h_smooth <= 0:
        filtered.pop("loss_h_smooth", None)
    if weights.recon <= 0:
        filtered.pop("loss_recon", None)
    if weights.recon_multi_gauss <= 0:
        filtered.pop("loss_recon_multi_gauss", None)
    if weights.recon_multi_box <= 0:
        filtered.pop("loss_recon_multi_box", None)
    if weights.recon_zhat_multi_box <= 0:
        filtered.pop("loss_recon_zhat_multi_box", None)
    if weights.recon_multi_box_mse <= 0:
        filtered.pop("loss_recon_multi_box_mse", None)
    if weights.recon_patch <= 0:
        filtered.pop("loss_recon_patch", None)
    if weights.pixel_delta <= 0:
        filtered.pop("loss_pixel_delta", None)
    if weights.pixel_delta_multi_box <= 0:
        filtered.pop("loss_pixel_delta_multi_box", None)
    # Always show val loss if present
    if "loss_val_world" in metrics:
        filtered["loss_val_world"] = metrics["loss_val_world"]
    if "loss_val_recon" in metrics:
        filtered["loss_val_recon"] = metrics["loss_val_recon"]
    if "loss_val_recon_multi_gauss" in metrics:
        filtered["loss_val_recon_multi_gauss"] = metrics["loss_val_recon_multi_gauss"]
    if "loss_val_recon_multi_box" in metrics:
        filtered["loss_val_recon_multi_box"] = metrics["loss_val_recon_multi_box"]
    if "loss_val_recon_zhat_multi_box" in metrics:
        filtered["loss_val_recon_zhat_multi_box"] = metrics["loss_val_recon_zhat_multi_box"]
    if "loss_val_recon_multi_box_mse" in metrics:
        filtered["loss_val_recon_multi_box_mse"] = metrics["loss_val_recon_multi_box_mse"]
    if "loss_val_recon_patch" in metrics:
        filtered["loss_val_recon_patch"] = metrics["loss_val_recon_patch"]
    if weights.h2z <= 0:
        filtered.pop("loss_h2z", None)
    if weights.z2h <= 0:
        filtered.pop("loss_z2h", None)
    if weights.z2h_init_zero <= 0:
        filtered.pop("loss_z2h_init_zero", None)
    if weights.z2h_match_h <= 0:
        filtered.pop("loss_z2h_match_h", None)
    if weights.h2z_delta <= 0:
        filtered.pop("loss_h2z_delta", None)
    if weights.geometry_rank_p <= 0:
        filtered.pop("loss_geometry_rank_p", None)
        filtered.pop("geometry_rank_p_accuracy", None)
        filtered.pop("geometry_rank_p_pairs", None)
    if weights.inverse_dynamics_z <= 0:
        filtered.pop("loss_inverse_dynamics_z", None)
    if weights.inverse_dynamics_h <= 0:
        filtered.pop("loss_inverse_dynamics_h", None)
    if weights.inverse_dynamics_p <= 0:
        filtered.pop("loss_inverse_dynamics_p", None)
    if weights.inverse_dynamics_dp <= 0:
        filtered.pop("loss_inverse_dynamics_dp", None)
    if weights.action_delta_z <= 0:
        filtered.pop("loss_action_delta_z", None)
    if weights.action_delta_h <= 0:
        filtered.pop("loss_action_delta_h", None)
    if weights.rollout_kstep_z <= 0:
        filtered.pop("loss_rollout_kstep_z", None)
    if weights.rollout_recon_z <= 0:
        filtered.pop("loss_rollout_recon_z", None)
    if weights.rollout_recon_multi_box_z <= 0:
        filtered.pop("loss_rollout_recon_multi_box_z", None)
    if weights.rollout_recon_delta_z <= 0:
        filtered.pop("loss_rollout_recon_delta_z", None)
    if weights.rollout_recon_multi_box_delta_z <= 0:
        filtered.pop("loss_rollout_recon_multi_box_delta_z", None)
    if weights.rollout_project_z <= 0:
        filtered.pop("loss_rollout_project_z", None)
    if weights.rollout_recon_h <= 0:
        filtered.pop("loss_rollout_recon_h", None)
    if weights.rollout_recon_multi_box_h <= 0:
        filtered.pop("loss_rollout_recon_multi_box_h", None)
    if weights.rollout_kstep_h <= 0:
        filtered.pop("loss_rollout_kstep_h", None)
    if weights.rollout_kstep_delta_h <= 0:
        filtered.pop("loss_rollout_kstep_delta_h", None)
    if weights.inverse_cycle_h <= 0:
        filtered.pop("loss_inverse_cycle_h", None)
    if weights.rollout_kstep_p <= 0:
        filtered.pop("loss_rollout_kstep_p", None)
    if weights.additivity_h <= 0:
        filtered.pop("loss_additivity_h", None)
    if weights.scale_dp <= 0:
        filtered.pop("loss_scale_dp", None)
    if weights.action_delta_dp <= 0:
        filtered.pop("loss_action_delta_dp", None)
    if weights.dz_anchor_dp <= 0:
        filtered.pop("loss_dz_anchor_dp", None)
    if weights.loop_closure_p <= 0:
        filtered.pop("loss_loop_closure_p", None)
    if weights.distance_corr_p <= 0:
        filtered.pop("loss_distance_corr_p", None)
    if weights.noop_residual_dp <= 0:
        filtered.pop("loss_noop_residual_dp", None)
    if weights.noop_residual_dh <= 0:
        filtered.pop("loss_noop_residual_dh", None)
    if weights.same_frame_h_identity <= 0:
        filtered.pop("loss_same_frame_h_identity", None)
    if weights.noop_move_orth_h <= 0:
        filtered.pop("loss_noop_move_orth_h", None)
    if weights.additivity_dp <= 0:
        filtered.pop("loss_additivity_dp", None)
    if weights.inverse_cycle_dp <= 0:
        filtered.pop("loss_inverse_cycle_dp", None)
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
    "loss_val_recon_zhat_multi_box",
    "loss_val_recon_multi_box_mse",
    "loss_val_recon_patch",
    "loss_jepa",
    "loss_jepa_rep",
    "loss_jepa_open_loop",
    "loss_sigreg",
    "loss_z_smooth",
    "loss_h_smooth",
    "loss_recon",
    "loss_recon_multi_gauss",
    "loss_recon_multi_box",
    "loss_recon_zhat_multi_box",
    "loss_recon_multi_box_mse",
    "loss_recon_patch",
    "loss_pixel_delta",
    "loss_pixel_delta_multi_box",
    "loss_h2z",
    "loss_z2h",
    "loss_z2h_init_zero",
    "loss_z2h_match_h",
    "loss_h2z_delta",
    "loss_geometry_rank_p",
    "geometry_rank_p_accuracy",
    "geometry_rank_p_pairs",
    "loss_inverse_dynamics_z",
    "loss_inverse_dynamics_h",
    "loss_inverse_dynamics_p",
    "loss_inverse_dynamics_dp",
    "loss_action_delta_z",
    "loss_action_delta_h",
    "loss_rollout_kstep_z",
    "loss_rollout_recon_z",
    "loss_rollout_recon_multi_box_z",
    "loss_rollout_recon_delta_z",
    "loss_rollout_recon_multi_box_delta_z",
    "loss_rollout_project_z",
    "loss_rollout_recon_h",
    "loss_rollout_recon_multi_box_h",
    "loss_rollout_kstep_h",
    "loss_rollout_kstep_delta_h",
    "loss_inverse_cycle_h",
    "loss_rollout_kstep_p",
    "loss_additivity_h",
    "loss_scale_dp",
    "loss_action_delta_dp",
    "loss_dz_anchor_dp",
    "loss_loop_closure_p",
    "loss_distance_corr_p",
    "loss_noop_residual_dp",
    "loss_noop_residual_dh",
    "loss_same_frame_h_identity",
    "loss_noop_move_orth_h",
    "loss_additivity_dp",
    "loss_inverse_cycle_dp",
    "z_norm_mean",
    "z_norm_std",
    "z_norm_max",
    "h_norm_mean",
    "h_norm_std",
    "h_norm_max",
    "p_norm_mean",
    "p_norm_std",
    "p_norm_max",
    "dp_norm_median",
    "dz_to_dp_norm_median",
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
    val_recon_zhat_multi_box: List[float] = field(default_factory=list)
    val_recon_multi_box_mse: List[float] = field(default_factory=list)
    val_recon_patch: List[float] = field(default_factory=list)
    jepa: List[float] = field(default_factory=list)
    jepa_rep: List[float] = field(default_factory=list)
    jepa_open_loop: List[float] = field(default_factory=list)
    sigreg: List[float] = field(default_factory=list)
    z_smooth: List[float] = field(default_factory=list)
    h_smooth: List[float] = field(default_factory=list)
    recon: List[float] = field(default_factory=list)
    recon_multi_gauss: List[float] = field(default_factory=list)
    recon_multi_box: List[float] = field(default_factory=list)
    recon_zhat_multi_box: List[float] = field(default_factory=list)
    recon_multi_box_mse: List[float] = field(default_factory=list)
    recon_patch: List[float] = field(default_factory=list)
    pixel_delta: List[float] = field(default_factory=list)
    pixel_delta_multi_box: List[float] = field(default_factory=list)
    h2z: List[float] = field(default_factory=list)
    z2h: List[float] = field(default_factory=list)
    z2h_init_zero: List[float] = field(default_factory=list)
    z2h_match_h: List[float] = field(default_factory=list)
    h2z_delta: List[float] = field(default_factory=list)
    geometry_rank_p: List[float] = field(default_factory=list)
    geometry_rank_p_accuracy: List[float] = field(default_factory=list)
    geometry_rank_p_pairs: List[float] = field(default_factory=list)
    inverse_dynamics_z: List[float] = field(default_factory=list)
    inverse_dynamics_h: List[float] = field(default_factory=list)
    inverse_dynamics_p: List[float] = field(default_factory=list)
    inverse_dynamics_dp: List[float] = field(default_factory=list)
    action_delta_z: List[float] = field(default_factory=list)
    action_delta_h: List[float] = field(default_factory=list)
    rollout_kstep_z: List[float] = field(default_factory=list)
    rollout_recon_z: List[float] = field(default_factory=list)
    rollout_recon_multi_box_z: List[float] = field(default_factory=list)
    rollout_recon_delta_z: List[float] = field(default_factory=list)
    rollout_recon_multi_box_delta_z: List[float] = field(default_factory=list)
    rollout_project_z: List[float] = field(default_factory=list)
    rollout_recon_h: List[float] = field(default_factory=list)
    rollout_recon_multi_box_h: List[float] = field(default_factory=list)
    rollout_kstep_h: List[float] = field(default_factory=list)
    rollout_kstep_delta_h: List[float] = field(default_factory=list)
    inverse_cycle_h: List[float] = field(default_factory=list)
    rollout_kstep_p: List[float] = field(default_factory=list)
    additivity_h: List[float] = field(default_factory=list)
    scale_dp: List[float] = field(default_factory=list)
    action_delta_dp: List[float] = field(default_factory=list)
    dz_anchor_dp: List[float] = field(default_factory=list)
    loop_closure_p: List[float] = field(default_factory=list)
    distance_corr_p: List[float] = field(default_factory=list)
    noop_residual_dp: List[float] = field(default_factory=list)
    noop_residual_dh: List[float] = field(default_factory=list)
    same_frame_h_identity: List[float] = field(default_factory=list)
    noop_move_orth_h: List[float] = field(default_factory=list)
    additivity_dp: List[float] = field(default_factory=list)
    inverse_cycle_dp: List[float] = field(default_factory=list)
    z_norm_mean: List[float] = field(default_factory=list)
    z_norm_std: List[float] = field(default_factory=list)
    z_norm_max: List[float] = field(default_factory=list)
    h_norm_mean: List[float] = field(default_factory=list)
    h_norm_std: List[float] = field(default_factory=list)
    h_norm_max: List[float] = field(default_factory=list)
    p_norm_mean: List[float] = field(default_factory=list)
    p_norm_std: List[float] = field(default_factory=list)
    p_norm_max: List[float] = field(default_factory=list)
    dp_norm_median: List[float] = field(default_factory=list)
    dz_to_dp_norm_median: List[float] = field(default_factory=list)
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
        self.val_recon_zhat_multi_box.append(metrics.get("loss_val_recon_zhat_multi_box", 0.0))
        self.val_recon_multi_box_mse.append(metrics.get("loss_val_recon_multi_box_mse", 0.0))
        self.val_recon_patch.append(metrics.get("loss_val_recon_patch", 0.0))
        self.jepa.append(metrics["loss_jepa"])
        self.jepa_rep.append(metrics.get("loss_jepa_rep", 0.0))
        self.jepa_open_loop.append(metrics.get("loss_jepa_open_loop", 0.0))
        self.sigreg.append(metrics["loss_sigreg"])
        self.z_smooth.append(metrics.get("loss_z_smooth", 0.0))
        self.h_smooth.append(metrics.get("loss_h_smooth", 0.0))
        self.recon.append(metrics["loss_recon"])
        self.recon_multi_gauss.append(metrics["loss_recon_multi_gauss"])
        self.recon_multi_box.append(metrics["loss_recon_multi_box"])
        self.recon_zhat_multi_box.append(metrics.get("loss_recon_zhat_multi_box", 0.0))
        self.recon_multi_box_mse.append(metrics.get("loss_recon_multi_box_mse", 0.0))
        self.recon_patch.append(metrics["loss_recon_patch"])
        self.pixel_delta.append(metrics.get("loss_pixel_delta", 0.0))
        self.pixel_delta_multi_box.append(metrics.get("loss_pixel_delta_multi_box", 0.0))
        self.h2z.append(metrics["loss_h2z"])
        self.z2h.append(metrics["loss_z2h"])
        self.z2h_init_zero.append(metrics.get("loss_z2h_init_zero", 0.0))
        self.z2h_match_h.append(metrics.get("loss_z2h_match_h", 0.0))
        self.h2z_delta.append(metrics.get("loss_h2z_delta", 0.0))
        self.geometry_rank_p.append(metrics.get("loss_geometry_rank_p", 0.0))
        self.geometry_rank_p_accuracy.append(metrics.get("geometry_rank_p_accuracy", 0.0))
        self.geometry_rank_p_pairs.append(metrics.get("geometry_rank_p_pairs", 0.0))
        self.inverse_dynamics_z.append(metrics.get("loss_inverse_dynamics_z", 0.0))
        self.inverse_dynamics_h.append(metrics.get("loss_inverse_dynamics_h", 0.0))
        self.inverse_dynamics_p.append(metrics.get("loss_inverse_dynamics_p", 0.0))
        self.inverse_dynamics_dp.append(metrics.get("loss_inverse_dynamics_dp", 0.0))
        self.action_delta_z.append(metrics.get("loss_action_delta_z", 0.0))
        self.action_delta_h.append(metrics.get("loss_action_delta_h", 0.0))
        self.rollout_kstep_z.append(metrics.get("loss_rollout_kstep_z", 0.0))
        self.rollout_recon_z.append(metrics.get("loss_rollout_recon_z", 0.0))
        self.rollout_recon_multi_box_z.append(metrics.get("loss_rollout_recon_multi_box_z", 0.0))
        self.rollout_recon_delta_z.append(metrics.get("loss_rollout_recon_delta_z", 0.0))
        self.rollout_recon_multi_box_delta_z.append(metrics.get("loss_rollout_recon_multi_box_delta_z", 0.0))
        self.rollout_project_z.append(metrics.get("loss_rollout_project_z", 0.0))
        self.rollout_recon_h.append(metrics.get("loss_rollout_recon_h", 0.0))
        self.rollout_recon_multi_box_h.append(metrics.get("loss_rollout_recon_multi_box_h", 0.0))
        self.rollout_kstep_h.append(metrics.get("loss_rollout_kstep_h", 0.0))
        self.rollout_kstep_delta_h.append(metrics.get("loss_rollout_kstep_delta_h", 0.0))
        self.inverse_cycle_h.append(metrics.get("loss_inverse_cycle_h", 0.0))
        self.rollout_kstep_p.append(metrics.get("loss_rollout_kstep_p", 0.0))
        self.additivity_h.append(metrics.get("loss_additivity_h", 0.0))
        self.scale_dp.append(metrics.get("loss_scale_dp", 0.0))
        self.action_delta_dp.append(metrics.get("loss_action_delta_dp", 0.0))
        self.dz_anchor_dp.append(metrics.get("loss_dz_anchor_dp", 0.0))
        self.loop_closure_p.append(metrics.get("loss_loop_closure_p", 0.0))
        self.distance_corr_p.append(metrics.get("loss_distance_corr_p", 0.0))
        self.noop_residual_dp.append(metrics.get("loss_noop_residual_dp", 0.0))
        self.noop_residual_dh.append(metrics.get("loss_noop_residual_dh", 0.0))
        self.same_frame_h_identity.append(metrics.get("loss_same_frame_h_identity", 0.0))
        self.noop_move_orth_h.append(metrics.get("loss_noop_move_orth_h", 0.0))
        self.additivity_dp.append(metrics.get("loss_additivity_dp", 0.0))
        self.inverse_cycle_dp.append(metrics.get("loss_inverse_cycle_dp", 0.0))
        self.z_norm_mean.append(metrics.get("z_norm_mean", 0.0))
        self.z_norm_std.append(metrics.get("z_norm_std", 0.0))
        self.z_norm_max.append(metrics.get("z_norm_max", 0.0))
        self.h_norm_mean.append(metrics.get("h_norm_mean", 0.0))
        self.h_norm_std.append(metrics.get("h_norm_std", 0.0))
        self.h_norm_max.append(metrics.get("h_norm_max", 0.0))
        self.p_norm_mean.append(metrics.get("p_norm_mean", 0.0))
        self.p_norm_std.append(metrics.get("p_norm_std", 0.0))
        self.p_norm_max.append(metrics.get("p_norm_max", 0.0))
        self.dp_norm_median.append(metrics.get("dp_norm_median", 0.0))
        self.dz_to_dp_norm_median.append(metrics.get("dz_to_dp_norm_median", 0.0))
        self.grad_world.append(metrics["grad_world"])
        self.grad_decoder.append(metrics["grad_decoder"])

    def __len__(self) -> int:
        return len(self.steps)


def _loss_history_row(history: LossHistory, index: int) -> Tuple[object, ...]:
    return (
        history.steps[index],
        history.elapsed_seconds[index],
        history.cumulative_flops[index],
        history.world[index],
        history.val_world[index],
        history.val_recon[index],
        history.val_recon_multi_gauss[index],
        history.val_recon_multi_box[index],
        history.val_recon_zhat_multi_box[index],
        history.val_recon_multi_box_mse[index],
        history.val_recon_patch[index],
        history.jepa[index],
        history.jepa_rep[index],
        history.jepa_open_loop[index],
        history.sigreg[index],
        history.z_smooth[index],
        history.h_smooth[index],
        history.recon[index],
        history.recon_multi_gauss[index],
        history.recon_multi_box[index],
        history.recon_zhat_multi_box[index],
        history.recon_multi_box_mse[index],
        history.recon_patch[index],
        history.pixel_delta[index],
        history.pixel_delta_multi_box[index],
        history.h2z[index],
        history.z2h[index],
        history.z2h_init_zero[index],
        history.z2h_match_h[index],
        history.h2z_delta[index],
        history.geometry_rank_p[index],
        history.geometry_rank_p_accuracy[index],
        history.geometry_rank_p_pairs[index],
        history.inverse_dynamics_z[index],
        history.inverse_dynamics_h[index],
        history.inverse_dynamics_p[index],
        history.inverse_dynamics_dp[index],
        history.action_delta_z[index],
        history.action_delta_h[index],
        history.rollout_kstep_z[index],
        history.rollout_recon_z[index],
        history.rollout_recon_multi_box_z[index],
        history.rollout_recon_delta_z[index],
        history.rollout_recon_multi_box_delta_z[index],
        history.rollout_project_z[index],
        history.rollout_recon_h[index],
        history.rollout_recon_multi_box_h[index],
        history.rollout_kstep_h[index],
        history.rollout_kstep_delta_h[index],
        history.inverse_cycle_h[index],
        history.rollout_kstep_p[index],
        history.additivity_h[index],
        history.scale_dp[index],
        history.action_delta_dp[index],
        history.dz_anchor_dp[index],
        history.loop_closure_p[index],
        history.distance_corr_p[index],
        history.noop_residual_dp[index],
        history.noop_residual_dh[index],
        history.same_frame_h_identity[index],
        history.noop_move_orth_h[index],
        history.additivity_dp[index],
        history.inverse_cycle_dp[index],
        history.z_norm_mean[index],
        history.z_norm_std[index],
        history.z_norm_max[index],
        history.h_norm_mean[index],
        history.h_norm_std[index],
        history.h_norm_max[index],
        history.p_norm_mean[index],
        history.p_norm_std[index],
        history.p_norm_max[index],
        history.dp_norm_median[index],
        history.dz_to_dp_norm_median[index],
        history.grad_world[index],
        history.grad_decoder[index],
    )


class LossCsvWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Optional[TextIO] = None
        self._writer: Optional[csv.writer] = None
        self._last_step: Optional[float] = None

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = True
        if self.path.exists():
            try:
                if self.path.stat().st_size > 0:
                    needs_header = False
            except OSError:
                needs_header = True
        self._handle = self.path.open("a", newline="")
        self._writer = csv.writer(self._handle)
        if needs_header:
            self._writer.writerow(LOSS_COLUMNS)
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._writer = None
        self._last_step = None

    def append_latest(self, history: LossHistory) -> None:
        if len(history) == 0:
            return
        if self._writer is None:
            raise AssertionError("LossCsvWriter is not open.")
        step = history.steps[-1]
        if self._last_step is not None and step == self._last_step:
            return
        self._writer.writerow(_loss_history_row(history, -1))
        assert self._handle is not None
        self._handle.flush()
        self._last_step = step


def plot_loss_curves(history: LossHistory, out_dir: Path) -> None:
    if len(history) == 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=figsize_for_grid(2, 2), constrained_layout=True)
    default_cycle = plt.rcParams.get("axes.prop_cycle")
    color_cycle = default_cycle.by_key().get("color", []) if default_cycle is not None else []

    def _color(idx: int) -> str:
        if not color_cycle:
            return "C0"
        return color_cycle[idx % len(color_cycle)]

    loss_series = [
        ("world", history.world),
        ("val_world", history.val_world),
        ("val_recon", history.val_recon),
        ("val_recon_multi_gauss", history.val_recon_multi_gauss),
        ("val_recon_multi_box", history.val_recon_multi_box),
        ("val_recon_zhat_multi_box", history.val_recon_zhat_multi_box),
        ("val_recon_multi_box_mse", history.val_recon_multi_box_mse),
        ("val_recon_patch", history.val_recon_patch),
        ("jepa", history.jepa),
        ("jepa_rep", history.jepa_rep),
        ("jepa_open_loop", history.jepa_open_loop),
        ("sigreg", history.sigreg),
        ("z_smooth", history.z_smooth),
        ("recon", history.recon),
        ("recon_multi_gauss", history.recon_multi_gauss),
        ("recon_multi_box", history.recon_multi_box),
        ("recon_zhat_multi_box", history.recon_zhat_multi_box),
        ("recon_multi_box_mse", history.recon_multi_box_mse),
        ("recon_patch", history.recon_patch),
        ("pixel_delta", history.pixel_delta),
        ("pixel_delta_multi_box", history.pixel_delta_multi_box),
        ("h2z", history.h2z),
        ("z2h", history.z2h),
        ("z2h_init_zero", history.z2h_init_zero),
        ("h2z_delta", history.h2z_delta),
        ("loss_geometry_rank_p", history.geometry_rank_p),
        ("loss_inverse_dynamics_z", history.inverse_dynamics_z),
        ("loss_inverse_dynamics_h", history.inverse_dynamics_h),
        ("loss_inverse_dynamics_p", history.inverse_dynamics_p),
        ("loss_inverse_dynamics_dp", history.inverse_dynamics_dp),
        ("loss_action_delta_z", history.action_delta_z),
        ("loss_action_delta_h", history.action_delta_h),
        ("loss_action_delta_dp", history.action_delta_dp),
        ("loss_dz_anchor_dp", history.dz_anchor_dp),
        ("loss_loop_closure_p", history.loop_closure_p),
        ("loss_distance_corr_p", history.distance_corr_p),
        ("loss_noop_residual_dp", history.noop_residual_dp),
        ("loss_noop_residual_dh", history.noop_residual_dh),
        ("loss_same_frame_h_identity", history.same_frame_h_identity),
        ("loss_noop_move_orth_h", history.noop_move_orth_h),
        ("loss_h_smooth", history.h_smooth),
        ("loss_rollout_kstep_z", history.rollout_kstep_z),
        ("loss_rollout_recon_z", history.rollout_recon_z),
        ("loss_rollout_recon_multi_box_z", history.rollout_recon_multi_box_z),
        ("loss_rollout_recon_delta_z", history.rollout_recon_delta_z),
        ("loss_rollout_recon_multi_box_delta_z", history.rollout_recon_multi_box_delta_z),
        ("loss_rollout_project_z", history.rollout_project_z),
        ("loss_rollout_recon_h", history.rollout_recon_h),
        ("loss_rollout_recon_multi_box_h", history.rollout_recon_multi_box_h),
        ("loss_rollout_kstep_h", history.rollout_kstep_h),
        ("loss_rollout_kstep_delta_h", history.rollout_kstep_delta_h),
        ("loss_inverse_cycle_h", history.inverse_cycle_h),
        ("loss_rollout_kstep_p", history.rollout_kstep_p),
        ("loss_additivity_h", history.additivity_h),
        ("loss_additivity_dp", history.additivity_dp),
        ("loss_scale_dp", history.scale_dp),
        ("loss_inverse_cycle_dp", history.inverse_cycle_dp),
    ]
    for idx, (label, series) in enumerate(loss_series):
        if any(val != 0.0 for val in series):
            plt.plot(history.steps, series, label=label, color=_color(idx))
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training Losses")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.savefig(out_dir / "loss_curves.png", dpi=DEFAULT_DPI)
    plt.close()

    has_rank_acc = any(val != 0.0 for val in history.geometry_rank_p_accuracy)
    has_rank_loss = any(val != 0.0 for val in history.geometry_rank_p)
    if has_rank_acc or has_rank_loss:
        fig, ax1 = plt.subplots(figsize=figsize_for_grid(1, 1), constrained_layout=True)
        if has_rank_acc:
            ax1.plot(history.steps, history.geometry_rank_p_accuracy, label="geometry_rank_p_accuracy", color=_color(3))
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Ranking accuracy")
        ax1.set_ylim(0.0, 1.0)
        ax1.grid(True, alpha=0.3)
        if has_rank_loss:
            ax2 = ax1.twinx()
            ax2.plot(history.steps, history.geometry_rank_p, label="loss_geometry_rank_p", color=_color(4))
            ax2.set_ylabel("Ranking loss")
            ax2.set_yscale("log")
        pair_values = [val for val in history.geometry_rank_p_pairs if val > 0]
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


TrajectoryDataset = Union[TrajectorySequenceDataset, PreloadedTrajectorySequenceDataset]


def _infer_raw_frame_shape(dataset: TrajectoryDataset) -> Tuple[int, int, int]:
    if not dataset.samples:
        raise ValueError("Cannot infer frame shape without dataset samples.")
    if isinstance(dataset, PreloadedTrajectorySequenceDataset):
        traj_idx, start = dataset.samples[0]
        if traj_idx < 0 or traj_idx >= len(dataset.trajs):
            raise ValueError("Dataset sample refers to missing preloaded trajectory.")
        _, _, traj_paths = dataset.trajs[traj_idx]
        if not traj_paths:
            raise ValueError("Preloaded trajectory contains no frame paths for shape inference.")
        index = min(max(start, 0), len(traj_paths) - 1)
        path = traj_paths[index]
    else:
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
    dataset: TrajectoryDataset,
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
    graph_diag_dataset = PreloadedTrajectorySequenceDataset(
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


def _build_off_manifold_batch(
    *,
    data_root: Path,
    train_trajs: Sequence[str],
    seq_len: int,
    image_hw: Tuple[int, int],
    sample_count: int,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
    off_dataset = PreloadedTrajectorySequenceDataset(
        root=data_root,
        seq_len=seq_len,
        image_hw=image_hw,
        max_traj=None,
        included_trajectories=train_trajs,
    )
    return _build_embedding_batch(off_dataset, sample_count, generator=generator)


def _frame_to_tensor(frame: np.ndarray, image_size: int) -> torch.Tensor:
    pil = Image.fromarray(frame)
    pil = pil.resize((image_size, image_size), Image.BILINEAR)
    arr = np.array(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _pose_from_frames(
    frames: Sequence[np.ndarray],
    model: JEPAWorldModel,
    model_cfg: ModelConfig,
    device: torch.device,
    *,
    use_z2h_init: bool,
    action_dim: int,
    force_h_zero: bool = False,
) -> np.ndarray:
    if len(frames) < 2:
        raise AssertionError("Pose extraction requires at least two frames.")
    frames_tensor = torch.stack(
        [_frame_to_tensor(f, model_cfg.image_size) for f in frames],
        dim=0,
    ).unsqueeze(0)
    frames_tensor = frames_tensor.to(device)
    actions_zero = torch.zeros((1, frames_tensor.shape[1] - 1, action_dim), device=device)
    with torch.no_grad():
        embeds = model.encode_sequence(frames_tensor)["embeddings"]
        _, _, h_states = rollout_teacher_forced(
            model,
            embeds,
            actions_zero,
            use_z2h_init=use_z2h_init,
            force_h_zero=force_h_zero,
        )
        pose_obs, _, _ = _rollout_pose(
            model,
            h_states,
            actions_zero,
            z_embeddings=embeds,
        )
    return pose_obs[0].detach().cpu().numpy()


def _extract_planning_latents(
    model: JEPAWorldModel,
    plan_frames: torch.Tensor,
    plan_actions: torch.Tensor,
    device: torch.device,
    *,
    use_z2h_init: bool,
    force_h_zero: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Optional[str]], np.ndarray, torch.Tensor]:
    assert torch.is_grad_enabled()
    with torch.no_grad():
        plan_frames_device = plan_frames.to(device)
        plan_actions_device = plan_actions.to(device)
        plan_embeddings = model.encode_sequence(plan_frames_device)["embeddings"]
        _, _, plan_h_states = rollout_teacher_forced(
            model,
            plan_embeddings,
            plan_actions_device,
            use_z2h_init=use_z2h_init,
            force_h_zero=force_h_zero,
        )
        _, plan_p_embeddings, _ = _rollout_pose(
            model,
            plan_h_states,
            plan_actions_device,
            z_embeddings=plan_embeddings,
        )
    p_t = plan_p_embeddings[:, :-1].detach().cpu().reshape(-1, plan_p_embeddings.shape[-1]).numpy()
    p_tp1 = plan_p_embeddings[:, 1:].detach().cpu().reshape(-1, plan_p_embeddings.shape[-1]).numpy()
    h_t = plan_h_states[:, :-1].detach().cpu().reshape(-1, plan_h_states.shape[-1]).numpy()
    h_tp1 = plan_h_states[:, 1:].detach().cpu().reshape(-1, plan_h_states.shape[-1]).numpy()
    actions_np = plan_actions[:, :-1].detach().cpu().reshape(-1, plan_actions.shape[-1]).numpy()
    action_labels = action_labels_from_vectors(actions_np)
    deltas = p_tp1 - p_t
    return p_t, p_tp1, h_t, h_tp1, actions_np, action_labels, deltas, plan_h_states


def _compute_planning_graphs(
    p_t: np.ndarray,
    p_tp1: np.ndarray,
    h_t: np.ndarray,
    h_tp1: np.ndarray,
    actions_np: np.ndarray,
    action_labels: Sequence[Optional[str]],
    *,
    min_action_count: int,
) -> Tuple[ActionDeltaStats, DatasetGraph, DatasetGraph, float]:
    stats = compute_action_delta_stats(
        p_t,
        p_tp1,
        actions_np,
        min_action_count=min_action_count,
    )
    non_noop = np.array([lbl in DIRECTION_ORDER for lbl in action_labels], dtype=bool)
    if not np.any(non_noop):
        raise AssertionError("Planning diagnostics require non-noop actions for h graph thresholds.")
    h_dot = (h_t * h_tp1).sum(axis=1)
    h_norms = np.maximum(np.linalg.norm(h_t, axis=1) * np.linalg.norm(h_tp1, axis=1), 1e-8)
    h_cos = h_dot / h_norms
    d_nn = float(np.median(1.0 - h_cos[non_noop]))
    tau_h_merge = min(max(3.0 * d_nn, 0.02), 0.08)
    graph_h = build_dataset_graph(
        h_t,
        h_tp1,
        actions_np,
        radius=tau_h_merge,
        metric="cosine",
    )
    graph_p = build_dataset_graph(
        p_t,
        p_tp1,
        actions_np,
        radius=stats.r_cluster_p,
        metric="l2",
    )
    return stats, graph_h, graph_p, tau_h_merge


def _run_h_local_sanity(
    graph_h: DatasetGraph,
    plan_h_states: torch.Tensor,
    tau_h_merge: float,
    cfg: PlanningDiagnosticsConfig,
    rng: random.Random,
) -> bool:
    seq_len = plan_h_states.shape[1]
    if seq_len <= cfg.local_k_min:
        raise AssertionError("Planning diagnostics require seq_len > local_k_min.")
    h_all = plan_h_states.detach().cpu().numpy().reshape(-1, plan_h_states.shape[-1])
    _, h_nodes = cluster_latents(h_all, radius=tau_h_merge, metric="cosine")
    h_nodes = h_nodes.reshape(plan_h_states.shape[0], seq_len)
    b_idx = rng.randrange(plan_h_states.shape[0])
    max_start = seq_len - cfg.local_k_min - 1
    if max_start <= 0:
        raise AssertionError("Planning diagnostics require room for local k-step test.")
    t0 = rng.randrange(max_start)
    k_max = min(cfg.local_k_max, seq_len - 1 - t0)
    k = rng.randint(cfg.local_k_min, k_max)
    h_local_actions = bfs_plan(graph_h, int(h_nodes[b_idx, t0]), int(h_nodes[b_idx, t0 + k]))
    return h_local_actions is not None


def _build_planning_tests(
    env: GridworldKeyEnv,
    cfg: PlanningDiagnosticsConfig,
) -> List[Tuple[str, Tuple[int, int], Tuple[int, int]]]:
    start_loop = (env.grid_rows - 2, 1)
    goal_center = (env.grid_rows // 2, env.grid_cols // 2)
    goal_mid = (
        env.grid_rows // 2,
        min(env.grid_cols - 2, env.grid_cols // 2 + cfg.interior_goal_col_offset),
    )
    return [
        ("test1", start_loop, goal_center),
        ("test2", goal_center, goal_mid),
    ]


def _write_planning_metrics_row(
    metrics_dir: Path,
    stats: ActionDeltaStats,
    graph_h: DatasetGraph,
    graph_p: DatasetGraph,
    h_reach: np.ndarray,
    p_reach: np.ndarray,
    h_local_success: bool,
    test_results: Dict[str, PlanningTestResult],
    *,
    global_step: int,
) -> None:
    def _pct(values: np.ndarray, q: float) -> float:
        if values.size == 0:
            return float("nan")
        return float(np.quantile(values, q))

    planning_metrics_path = metrics_dir / "planning_metrics.csv"
    header = [
        "step",
        "L",
        "inv_lr",
        "inv_ud",
        "noop_ratio",
        "q_L",
        "q_R",
        "q_U",
        "q_D",
        "num_nodes_h",
        "num_nodes_p",
        "reach_h_median",
        "reach_h_p10",
        "reach_h_p90",
        "reach_p_median",
        "reach_p_p10",
        "reach_p_p90",
        "h_local_success",
        "test1_success",
        "test1_steps",
        "test1_final_p_dist",
        "test1_goal_dist",
        "test2_success",
        "test2_steps",
        "test2_final_p_dist",
        "test2_goal_dist",
    ]
    row = [
        global_step,
        stats.L_scale,
        stats.inv_lr,
        stats.inv_ud,
        stats.noop_ratio,
        stats.q.get("L", float("nan")),
        stats.q.get("R", float("nan")),
        stats.q.get("U", float("nan")),
        stats.q.get("D", float("nan")),
        graph_h.centers.shape[0],
        graph_p.centers.shape[0],
        float(np.median(h_reach)) if h_reach.size else float("nan"),
        _pct(h_reach, 0.10),
        _pct(h_reach, 0.90),
        float(np.median(p_reach)) if p_reach.size else float("nan"),
        _pct(p_reach, 0.10),
        _pct(p_reach, 0.90),
        float(h_local_success),
        float(test_results["test1"].success),
        test_results["test1"].steps,
        test_results["test1"].final_p_distance,
        test_results["test1"].goal_distance,
        float(test_results["test2"].success),
        test_results["test2"].steps,
        test_results["test2"].final_p_distance,
        test_results["test2"].goal_distance,
    ]
    append_csv_row(planning_metrics_path, header, row)


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
        ("plan", "plan"),
    ):
        value = totals.get(key, 0.0)
        fraction = (value / total_time) if total_time > 0 else 0.0
        parts.append(f"{label}: {value:.2f}s ({fraction:.1%})")
    print(f"[timing up to step {step}] " + ", ".join(parts))


@dataclass(frozen=True)
class ImageWriteTiming:
    path: str
    seconds: float
    writer: str


@contextmanager
def _capture_image_write_timings() -> Iterator[List[ImageWriteTiming]]:
    timings: List[ImageWriteTiming] = []
    original_savefig = Figure.savefig
    original_image_save = Image.Image.save
    in_matplotlib_save = 0

    def _timed_savefig(fig: Figure, fname: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal in_matplotlib_save
        start = perf_counter()
        in_matplotlib_save += 1
        try:
            return original_savefig(fig, fname, *args, **kwargs)
        finally:
            in_matplotlib_save = max(in_matplotlib_save - 1, 0)
            timings.append(
                ImageWriteTiming(
                    path=str(fname),
                    seconds=max(perf_counter() - start, 0.0),
                    writer="matplotlib",
                )
            )

    def _timed_image_save(image: Image.Image, fp: Any, *args: Any, **kwargs: Any) -> Any:
        if in_matplotlib_save > 0:
            return original_image_save(image, fp, *args, **kwargs)
        start = perf_counter()
        try:
            return original_image_save(image, fp, *args, **kwargs)
        finally:
            timings.append(
                ImageWriteTiming(
                    path=str(fp),
                    seconds=max(perf_counter() - start, 0.0),
                    writer="pillow",
                )
            )

    Figure.savefig = _timed_savefig  # type: ignore[assignment]
    Image.Image.save = _timed_image_save  # type: ignore[assignment]
    try:
        yield timings
    finally:
        Figure.savefig = original_savefig  # type: ignore[assignment]
        Image.Image.save = original_image_save  # type: ignore[assignment]


def _write_output_timing_rows(
    csv_path: Path,
    *,
    section: str,
    step: int,
    image_timings: Sequence[ImageWriteTiming],
    preprocess_seconds: float,
    total_seconds: float,
) -> None:
    header = [
        "step",
        "section",
        "output_path",
        "writer",
        "image_seconds",
        "section_preprocess_seconds",
        "section_total_seconds",
    ]
    for timing in image_timings:
        append_csv_row(
            csv_path,
            header,
            [
                step,
                section,
                timing.path,
                timing.writer,
                timing.seconds,
                preprocess_seconds,
                total_seconds,
            ],
        )


def _print_output_timing_summary(
    *,
    step: int,
    section: str,
    image_timings: Sequence[ImageWriteTiming],
    preprocess_seconds: float,
    total_seconds: float,
    top_k: int = 8,
) -> None:
    image_total = max(total_seconds - preprocess_seconds, 0.0)
    if not image_timings:
        print(
            f"[{section} timing step {step}] preprocess={preprocess_seconds:.2f}s, "
            f"image_save={image_total:.2f}s, outputs=0"
        )
        return
    output_totals: Dict[str, float] = defaultdict(float)
    for timing in image_timings:
        output_totals[timing.path] += timing.seconds
    ranked = sorted(output_totals.items(), key=lambda item: item[1], reverse=True)
    summary = ", ".join(
        f"{Path(path).name}:{seconds:.2f}s"
        for path, seconds in ranked[: max(top_k, 1)]
    )
    print(
        f"[{section} timing step {step}] preprocess={preprocess_seconds:.2f}s, "
        f"image_save={image_total:.2f}s, outputs={len(output_totals)} | {summary}"
    )


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
    if asdict(weights) != asdict(cfg.loss_weights):
        mismatched = [
            name
            for name, expected in asdict(cfg.loss_weights).items()
            if asdict(weights)[name] != expected
        ]
        raise AssertionError(
            "run_training received weights that differ from cfg.loss_weights; "
            f"mismatched fields: {', '.join(sorted(mismatched))}"
        )
    # --- Filesystem + metadata setup ---
    device = pick_device(cfg.device)
    seed_value, python_rng = _seed_everything(cfg.seed)
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(seed_value)
    val_dataloader_generator = torch.Generator()
    val_dataloader_generator.manual_seed(python_rng.randint(0, 2**32 - 1))
    embedding_generator = torch.Generator()
    embedding_generator.manual_seed(python_rng.randint(0, 2**32 - 1))
    for name in (
        "enable_inverse_dynamics_z",
        "enable_inverse_dynamics_h",
        "enable_inverse_dynamics_p",
        "enable_inverse_dynamics_dp",
        "enable_action_delta_z",
        "enable_action_delta_h",
        "enable_action_delta_p",
        "enable_dz_to_dp_projector",
        "enable_h2z_delta",
    ):
        if getattr(model_cfg, name) is not None:
            raise AssertionError(f"{name} must be None and derived from loss weights during training.")
    if cfg.force_h_zero:
        invalid_losses = [
            name
            for name in (
                "h2z",
                "h2z_delta",
                "z2h",
                "z2h_init_zero",
                "z2h_match_h",
                "inverse_dynamics_h",
                "action_delta_h",
                "additivity_h",
                "rollout_kstep_h",
                "rollout_kstep_delta_h",
                "rollout_recon_h",
                "rollout_recon_multi_box_h",
                "inverse_cycle_h",
                "inverse_dynamics_p",
                "inverse_dynamics_dp",
                "action_delta_dp",
                "dz_anchor_dp",
                "loop_closure_p",
                "noop_residual_dp",
                "same_frame_h_identity",
                "noop_move_orth_h",
                "additivity_dp",
                "rollout_kstep_p",
                "scale_dp",
                "geometry_rank_p",
                "inverse_cycle_dp",
            )
            if getattr(weights, name, 0.0) > 0
        ]
        if invalid_losses:
            raise AssertionError(
                "force_h_zero requires h/p losses to be disabled: "
                + ", ".join(sorted(invalid_losses))
            )
    model_cfg_runtime = replace(
        model_cfg,
        z_norm=cfg.z_norm,
        enable_inverse_dynamics_z=weights.inverse_dynamics_z > 0,
        enable_inverse_dynamics_h=weights.inverse_dynamics_h > 0,
        enable_inverse_dynamics_p=weights.inverse_dynamics_p > 0,
        enable_inverse_dynamics_dp=weights.inverse_dynamics_dp > 0,
        enable_action_delta_z=weights.action_delta_z > 0,
        enable_action_delta_h=(
            weights.action_delta_h > 0
            or weights.additivity_h > 0
        ),
        enable_action_delta_p=(
            weights.action_delta_dp > 0
            or weights.additivity_dp > 0
            or weights.rollout_kstep_p > 0
            or weights.dz_anchor_dp > 0
            or weights.loop_closure_p > 0
            or weights.distance_corr_p > 0
            or weights.noop_residual_dp > 0
            or weights.noop_residual_dh > 0
            or weights.h_smooth > 0
            or weights.scale_dp > 0
            or weights.geometry_rank_p > 0
            or weights.inverse_cycle_dp > 0
            or weights.inverse_dynamics_p > 0
            or weights.inverse_dynamics_dp > 0
        ),
        enable_dz_to_dp_projector=weights.dz_anchor_dp > 0,
        enable_h2z_delta=weights.h2z_delta > 0,
    )
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
    inputs_vis_dir = run_dir / "vis_inputs"
    pair_vis_dir = run_dir / "vis_pairs"
    planning_action_stats_dir = run_dir / "vis_planning_action_stats"
    planning_pca_dir = run_dir / "vis_planning_pca"
    planning_exec_dir = run_dir / "vis_planning_exec"
    planning_reachable_dir = run_dir / "vis_planning_reachable"
    planning_graph_dir = run_dir / "vis_planning_graph"
    spike_diagnostics_dir = run_dir / "spike_diagnostics"
    checkpoints_dir = run_dir / "checkpoints"

    print(f"[run] Writing outputs to {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if cfg.spike_diagnostics.enabled:
        spike_diagnostics_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    debug_vis = cfg.debug_visualization
    if debug_vis.input_vis_every_steps > 0:
        inputs_vis_dir.mkdir(parents=True, exist_ok=True)
    if debug_vis.pair_vis_every_steps > 0:
        pair_vis_dir.mkdir(parents=True, exist_ok=True)

    loss_history = LossHistory()
    loss_csv_writer = LossCsvWriter(metrics_dir / "loss.csv")
    loss_csv_writer.open()
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
    dataset = PreloadedTrajectorySequenceDataset(
        root=cfg.data_root,
        seq_len=cfg.seq_len,
        image_hw=(model_cfg.image_size, model_cfg.image_size),
        max_traj=None,
        included_trajectories=train_trajs,
    )
    val_dataset: Optional[TrajectoryDataset] = None
    if val_trajs:
        val_dataset = PreloadedTrajectorySequenceDataset(
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

    graph_diag_batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor, List[List[str]]]] = None
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
    model = JEPAWorldModel(model_cfg_runtime).to(device)

    ema_model: Optional[JEPAWorldModel] = None

    decoder_schedule = (
        model_cfg_runtime.decoder_schedule
        if model_cfg_runtime.decoder_schedule is not None
        else model_cfg_runtime.encoder_schedule
    )
    decoder = VisualizationDecoder(
        model.embedding_dim,
        model_cfg_runtime.in_channels,
        model_cfg_runtime.image_size,
        decoder_schedule,
    ).to(device)

    flops_per_step = calculate_flops_per_step(model_cfg_runtime, cfg.batch_size, cfg.seq_len)

    # Write model_shape.txt
    _write_model_shape_summary(run_dir, dataset, model, decoder, model_cfg_runtime, flops_per_step)

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

    planning_enabled = planning_outputs_enabled(
        weights=weights,
        model=model,
        hard_example_cfg=cfg.hard_example,
        graph_cfg=cfg.graph_diagnostics,
        planning_cfg=cfg.planning_diagnostics,
    )
    if planning_enabled:
        planning_action_stats_dir.mkdir(parents=True, exist_ok=True)
        planning_pca_dir.mkdir(parents=True, exist_ok=True)
        planning_exec_dir.mkdir(parents=True, exist_ok=True)
        planning_reachable_dir.mkdir(parents=True, exist_ok=True)
        planning_graph_dir.mkdir(parents=True, exist_ok=True)

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

    diagnostics_batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor, List[List[str]]]] = None
    if cfg.diagnostics.sample_sequences <= 0:
        raise AssertionError(
            "Diagnostics requested but diagnostics.sample_sequences is not positive."
        )
    diagnostics_batch_cpu = _build_embedding_batch(
        dataset,
        cfg.diagnostics.sample_sequences,
        generator=diagnostics_generator,
    )

    planning_batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor, List[List[str]]]] = None
    planning_env: Optional[GridworldKeyEnv] = None
    grid_overlay_frames = None
    if planning_enabled:
        if cfg.planning_diagnostics.sample_sequences <= 0:
            raise AssertionError(
                "Planning diagnostics requested but planning_diagnostics.sample_sequences is not positive."
            )
        planning_batch_cpu = _build_embedding_batch(
            dataset,
            cfg.planning_diagnostics.sample_sequences,
            generator=diagnostics_generator,
        )
        planning_env = create_env_with_theme(
            theme=cfg.planning_diagnostics.env_theme,
            render_mode="rgb_array",
            keyboard_override=False,
            start_manual_control=False,
        )
        grid_overlay_frames = build_grid_overlay_frames(theme=cfg.planning_diagnostics.env_theme)
        grid_overlay_dir = run_dir / "vis_grid_overlay"
        grid_overlay_dir.mkdir(parents=True, exist_ok=True)
        save_grid_overlay_frame_grid(
            image_size=model_cfg.image_size,
            out_path=grid_overlay_dir / "grid_frames.png",
            frames_data=grid_overlay_frames,
        )

    vis_ctrl_batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor, List[List[str]]]] = None
    if cfg.vis_ctrl.sample_sequences <= 0:
        raise AssertionError(
            "Vis-ctrl requested but vis_ctrl.sample_sequences is not positive."
        )
    vis_ctrl_batch_cpu = _build_embedding_batch(
        dataset,
        cfg.vis_ctrl.sample_sequences,
        generator=vis_ctrl_generator,
    )

    timing_totals: Dict[str, float] = {"train": 0.0, "log": 0.0, "vis": 0.0, "plan": 0.0}
    total_samples_processed = 0
    run_start_time = perf_counter()
    loss_norm_ema: Dict[str, float] = {}

    try:
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
                    val_metrics, val_difficulty = validation_step(
                        model, decoder, val_batch, cfg, weights, loss_norm_ema
                    )
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
                    metrics_for_log["loss_val_recon_zhat_multi_box"] = val_metrics["loss_recon_zhat_multi_box"]
                    metrics_for_log["loss_val_recon_multi_box_mse"] = val_metrics["loss_recon_multi_box_mse"]
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
                loss_csv_writer.append_latest(loss_history)

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
                with _capture_image_write_timings() as vis_image_timings:
                    run_diagnostics_step(
                        diagnostics_cfg=cfg.diagnostics,
                        vis_cfg=cfg.vis,
                        hard_example_cfg=cfg.hard_example,
                        vis_ctrl_cfg=cfg.vis_ctrl,
                        graph_cfg=cfg.graph_diagnostics,
                        planning_cfg=cfg.planning_diagnostics,
                        planning_env=planning_env,
                        grid_overlay_frames=grid_overlay_frames,
                        model=model,
                        decoder=decoder,
                        device=device,
                        weights=weights,
                        global_step=global_step,
                        fixed_batch_cpu=fixed_batch_cpu,
                        fixed_selection=fixed_selection,
                        rolling_batch_cpu=rolling_batch_cpu,
                        off_manifold_batch_cpu=off_manifold_batch_cpu,
                        off_manifold_steps=off_manifold_steps,
                        embedding_batch_cpu=embedding_batch_cpu,
                        diagnostics_batch_cpu=diagnostics_batch_cpu,
                        vis_ctrl_batch_cpu=vis_ctrl_batch_cpu,
                        graph_diag_batch_cpu=graph_diag_batch_cpu,
                        hard_reservoir=hard_reservoir,
                        hard_reservoir_val=hard_reservoir_val,
                        dataset=dataset,
                        self_distance_inputs=self_distance_inputs,
                        diagnostics_generator=diagnostics_generator,
                        vis_selection_generator=vis_selection_generator,
                        run_dir=run_dir,
                        render_mode=cfg.render_mode,
                        force_h_zero=cfg.force_h_zero,
                    )
                vis_total = max(perf_counter() - vis_start, 0.0)
                vis_image_total = sum(item.seconds for item in vis_image_timings)
                vis_preprocess = max(vis_total - vis_image_total, 0.0)
                timing_totals["vis"] += vis_total
                _write_output_timing_rows(
                    metrics_dir / "visualization_output_timing.csv",
                    section="visualization",
                    step=global_step,
                    image_timings=vis_image_timings,
                    preprocess_seconds=vis_preprocess,
                    total_seconds=vis_total,
                )
                if cfg.show_timing_breakdown:
                    _print_output_timing_summary(
                        step=global_step,
                        section="visualization",
                        image_timings=vis_image_timings,
                        preprocess_seconds=vis_preprocess,
                        total_seconds=vis_total,
                    )

            if planning_batch_cpu is not None and _should_run_schedule(global_step, cfg.plan_schedule):
                plan_start = perf_counter()
                with _capture_image_write_timings() as planning_image_timings:
                    run_planning_diagnostics_step(
                        planning_cfg=cfg.planning_diagnostics,
                        hard_example_cfg=cfg.hard_example,
                        graph_cfg=cfg.graph_diagnostics,
                        model_cfg=model_cfg,
                        model=model,
                        device=device,
                        weights=weights,
                        global_step=global_step,
                        planning_batch_cpu=planning_batch_cpu,
                        planning_env=planning_env,
                        grid_overlay_frames=grid_overlay_frames,
                        run_dir=run_dir,
                        force_h_zero=cfg.force_h_zero,
                    )
                plan_total = max(perf_counter() - plan_start, 0.0)
                plan_image_total = sum(item.seconds for item in planning_image_timings)
                plan_preprocess = max(plan_total - plan_image_total, 0.0)
                timing_totals["plan"] += plan_total
                _write_output_timing_rows(
                    metrics_dir / "planning_output_timing.csv",
                    section="planning",
                    step=global_step,
                    image_timings=planning_image_timings,
                    preprocess_seconds=plan_preprocess,
                    total_seconds=plan_total,
                )
                if cfg.show_timing_breakdown:
                    _print_output_timing_summary(
                        step=global_step,
                        section="planning",
                        image_timings=planning_image_timings,
                        preprocess_seconds=plan_preprocess,
                        total_seconds=plan_total,
                    )

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

        if planning_env is not None:
            planning_env.close()

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
            loss_csv_writer.append_latest(loss_history)
            plot_loss_curves(loss_history, metrics_dir)
    finally:
        loss_csv_writer.close()


def main() -> None:
    cfg = tyro.cli(
        TrainConfig,
        config=(tyro.conf.HelptextFromCommentsOff,),
    )
    cfg = replace(
        cfg,
        log_schedule=_parse_schedule(cfg.log_schedule),
        vis_schedule=_parse_schedule(cfg.vis_schedule),
        plan_schedule=_parse_schedule(cfg.plan_schedule),
    )
    h_metric = cfg.planning_diagnostics.h_distance_metric
    if isinstance(h_metric, str) and h_metric.lower() == "null":
        h_metric = "l2"
    if h_metric not in ("l2", "cosine"):
        raise AssertionError(f"planning_diagnostics.h_distance_metric must be 'l2' or 'cosine', got {h_metric!r}.")
    if h_metric != cfg.planning_diagnostics.h_distance_metric:
        cfg = replace(
            cfg,
            planning_diagnostics=replace(cfg.planning_diagnostics, h_distance_metric=h_metric),
        )
    model_cfg = ModelConfig()
    run_training(cfg, model_cfg, cfg.loss_weights, title=cfg.title)


if __name__ == "__main__":
    main()
