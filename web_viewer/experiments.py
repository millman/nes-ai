from __future__ import annotations

import fnmatch
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import csv
from itertools import chain
import re

from .csv_utils import get_max_step
from .experiments_action_alignment import extract_alignment_summary
from .first_matches import (
    _first_existing_steps,
    _first_matching_csv_candidate,
    _first_matching_file,
)
from .experiments_git import _extract_git_commit
from .experiments_last_modified import _get_last_modified, _quick_last_modified
from .experiments_metadata import (
    _extract_data_root_from_metadata,
    _read_metadata,
    _read_model_metadata,
    _read_or_create_notes,
    write_archived,
    write_notes,
    write_starred,
    write_tags,
    write_title,
)
from .experiments_model_diff import _ensure_model_diff, _parse_model_diff_items, _render_model_diff

ALL_ZERO_EPS = 1e-12
PROFILE_ENABLED = os.environ.get("VIEWER_PROFILE", "").lower() in {"1", "true", "yes", "on"}
PROFILE_LOGGER = logging.getLogger("web_viewer.profile")
PROFILE_LOGGER.setLevel(logging.INFO)


def _profile(label: str, start_time: float, path: Optional[Path] = None, **fields) -> None:
    """Log lightweight profiling info when VIEWER_PROFILE is enabled."""
    if not PROFILE_ENABLED:
        return
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    parts = [f"{label} {elapsed_ms:.1f}ms"]
    if path is not None:
        parts.append(f"path={path}")
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    PROFILE_LOGGER.info(" ".join(parts))


@dataclass(frozen=True)
class VisSpec:
    label: Optional[str]
    candidates: List[Tuple[str, str] | Tuple[str, str, str]]
    prefix: Optional[str] = None

    def __post_init__(self) -> None:
        if self.prefix is not None:
            return
        updated: List[Tuple[str, str] | Tuple[str, str, str]] = []
        changed = False
        for candidate in self.candidates:
            if len(candidate) == 2:
                folder, pattern = candidate
                updated.append((folder, pattern, _pattern_prefix(pattern)))
                changed = True
            else:
                updated.append(candidate)
        if changed:
            object.__setattr__(self, "candidates", updated)


def _pattern_prefix(pattern: str) -> str:
    for idx, ch in enumerate(pattern):
        if ch in "*?[":
            return pattern[:idx]
    return pattern


DIAGNOSTICS_SUFFIXES = (".png", ".csv", ".txt")
DIAGNOSTICS_Z_DIRS = [
    "vis_delta_z_pca",
    "vis_delta_p_pca",
    "vis_action_alignment_z",
    "vis_action_alignment",
    "vis_action_alignment_p",
    "vis_cycle_error_z",
    "vis_cycle_error",
    "vis_cycle_error_p",
    "vis_diagnostics_frames",
    "vis_rollout_divergence",
    "vis_rollout_divergence_z",
    "vis_rollout_divergence_h",
    "vis_rollout_divergence_p",
    "vis_z_consistency",
    "vis_z_monotonicity",
    "vis_path_independence",
]
DIAGNOSTICS_P_DIRS = [
    "vis_delta_p_pca",
    "vis_action_alignment_p",
    "vis_cycle_error_p",
    "vis_straightline_p",
    "vis_delta_s_pca",
    "vis_action_alignment_s",
    "vis_cycle_error_s",
    "vis_straightline_s",
    "vis_rollout_divergence_p",
    "vis_rollout_divergence_s",
]
DIAGNOSTICS_H_PATTERNS = [
    ("vis_delta_h_pca", "delta_h_pca_*.png"),
    ("vis_action_alignment_h", "action_alignment_detail_*.png"),
    ("vis_cycle_error_h", "cycle_error_*.png"),
    ("vis_self_distance_h", "self_distance_h_*.png"),
    ("vis_h_ablation", "h_ablation_*.png"),
    ("vis_h_drift_by_action", "h_drift_by_action_*.png"),
    ("vis_norm_timeseries", "norm_timeseries_*.png"),
]
GRAPH_DIAGNOSTICS_Z_FOLDER_CANDIDATES = ["graph_diagnostics_z", "graph_diagnostics"]
GRAPH_DIAGNOSTICS_H_FOLDER_CANDIDATES = ["graph_diagnostics_h"]
GRAPH_DIAGNOSTICS_P_FOLDER_CANDIDATES = ["graph_diagnostics_p", "graph_diagnostics_s"]
GRAPH_DIAGNOSTICS_Z_RANK1_CDF_SPEC = VisSpec(
    label="Graph Diagnostics:Rank-1 CDF (Z)",
    candidates=[
        ("graph_diagnostics_z", "rank1_cdf_*.png"),
        ("graph_diagnostics", "rank1_cdf_*.png"),
    ],
)
GRAPH_DIAGNOSTICS_Z_RANK2_CDF_SPEC = VisSpec(
    label="Graph Diagnostics:Rank-2 CDF (Z)",
    candidates=[
        ("graph_diagnostics_z", "rank2_cdf_*.png"),
        ("graph_diagnostics", "rank2_cdf_*.png"),
    ],
)
GRAPH_DIAGNOSTICS_Z_NEFF_VIOLIN_SPEC = VisSpec(
    label="Graph Diagnostics:Neighborhood size (Z)",
    candidates=[
        ("graph_diagnostics_z", "neff_violin_*.png"),
        ("graph_diagnostics", "neff_violin_*.png"),
    ],
)
GRAPH_DIAGNOSTICS_Z_IN_DEGREE_HIST_SPEC = VisSpec(
    label="Graph Diagnostics:In-degree (Z)",
    candidates=[
        ("graph_diagnostics_z", "in_degree_hist_*.png"),
        ("graph_diagnostics", "in_degree_hist_*.png"),
    ],
)
GRAPH_DIAGNOSTICS_Z_EDGE_CONSISTENCY_SPEC = VisSpec(
    label="Graph Diagnostics:Edge consistency (Z)",
    candidates=[
        ("graph_diagnostics_z", "edge_consistency_*.png"),
        ("graph_diagnostics", "edge_consistency_*.png"),
    ],
)
GRAPH_DIAGNOSTICS_Z_METRICS_HISTORY_SPEC = VisSpec(
    label="Graph Diagnostics:Metrics history (Z)",
    candidates=[
        ("graph_diagnostics_z", "metrics_history_*.png"),
        ("graph_diagnostics", "metrics_history_*.png"),
    ],
)
GRAPH_DIAGNOSTICS_Z_METRICS_HISTORY_LATEST_SPEC = VisSpec(
    label="graph diagnostics metrics history latest z images",
    candidates=[
        ("graph_diagnostics_z", "metrics_history_latest.png"),
        ("graph_diagnostics", "metrics_history_latest.png"),
    ],
)
GRAPH_DIAGNOSTICS_Z_METRICS_HISTORY_CSV_SPEC = VisSpec(
    label="graph diagnostics metrics history z csvs",
    candidates=[
        ("graph_diagnostics_z", "metrics_history*.csv"),
        ("graph_diagnostics", "metrics_history*.csv"),
    ],
)
GRAPH_DIAGNOSTICS_H_RANK1_CDF_SPEC = VisSpec(
    label="Graph Diagnostics:Rank-1 CDF (H)",
    candidates=[
        ("graph_diagnostics_h", "rank1_cdf_*.png"),
    ],
)
GRAPH_DIAGNOSTICS_H_RANK2_CDF_SPEC = VisSpec(
    label="Graph Diagnostics:Rank-2 CDF (H)",
    candidates=[
        ("graph_diagnostics_h", "rank2_cdf_*.png"),
    ],
)
GRAPH_DIAGNOSTICS_H_NEFF_VIOLIN_SPEC = VisSpec(
    label="Graph Diagnostics:Neighborhood size (H)",
    candidates=[
        ("graph_diagnostics_h", "neff_violin_*.png"),
    ],
)
GRAPH_DIAGNOSTICS_H_IN_DEGREE_HIST_SPEC = VisSpec(
    label="Graph Diagnostics:In-degree (H)",
    candidates=[
        ("graph_diagnostics_h", "in_degree_hist_*.png"),
    ],
)
GRAPH_DIAGNOSTICS_H_EDGE_CONSISTENCY_SPEC = VisSpec(
    label="Graph Diagnostics:Edge consistency (H)",
    candidates=[
        ("graph_diagnostics_h", "edge_consistency_*.png"),
    ],
)
GRAPH_DIAGNOSTICS_H_METRICS_HISTORY_SPEC = VisSpec(
    label="Graph Diagnostics:Metrics history (H)",
    candidates=[
        ("graph_diagnostics_h", "metrics_history_*.png"),
    ],
)
GRAPH_DIAGNOSTICS_H_METRICS_HISTORY_LATEST_SPEC = VisSpec(
    label="graph diagnostics metrics history latest h images",
    candidates=[
        ("graph_diagnostics_h", "metrics_history_latest.png"),
    ],
)
GRAPH_DIAGNOSTICS_H_METRICS_HISTORY_CSV_SPEC = VisSpec(
    label="graph diagnostics metrics history h csvs",
    candidates=[
        ("graph_diagnostics_h", "metrics_history*.csv"),
    ],
)
GRAPH_DIAGNOSTICS_P_RANK1_CDF_SPEC = VisSpec(
    label="Graph Diagnostics:Rank-1 CDF (P)",
    candidates=[
        ("graph_diagnostics_p", "rank1_cdf_*.png"),
        ("graph_diagnostics_s", "rank1_cdf_*.png"),
    ],
)
GRAPH_DIAGNOSTICS_P_RANK2_CDF_SPEC = VisSpec(
    label="Graph Diagnostics:Rank-2 CDF (P)",
    candidates=[
        ("graph_diagnostics_p", "rank2_cdf_*.png"),
        ("graph_diagnostics_s", "rank2_cdf_*.png"),
    ],
)
GRAPH_DIAGNOSTICS_P_NEFF_VIOLIN_SPEC = VisSpec(
    label="Graph Diagnostics:Neighborhood size (P)",
    candidates=[
        ("graph_diagnostics_p", "neff_violin_*.png"),
        ("graph_diagnostics_s", "neff_violin_*.png"),
    ],
)
GRAPH_DIAGNOSTICS_P_IN_DEGREE_HIST_SPEC = VisSpec(
    label="Graph Diagnostics:In-degree (P)",
    candidates=[
        ("graph_diagnostics_p", "in_degree_hist_*.png"),
        ("graph_diagnostics_s", "in_degree_hist_*.png"),
    ],
)
GRAPH_DIAGNOSTICS_P_EDGE_CONSISTENCY_SPEC = VisSpec(
    label="Graph Diagnostics:Edge consistency (P)",
    candidates=[
        ("graph_diagnostics_p", "edge_consistency_*.png"),
        ("graph_diagnostics_s", "edge_consistency_*.png"),
    ],
)
GRAPH_DIAGNOSTICS_P_METRICS_HISTORY_SPEC = VisSpec(
    label="Graph Diagnostics:Metrics history (P)",
    candidates=[
        ("graph_diagnostics_p", "metrics_history_*.png"),
        ("graph_diagnostics_s", "metrics_history_*.png"),
    ],
)
GRAPH_DIAGNOSTICS_P_METRICS_HISTORY_LATEST_SPEC = VisSpec(
    label="graph diagnostics metrics history latest p images",
    candidates=[
        ("graph_diagnostics_p", "metrics_history_latest.png"),
        ("graph_diagnostics_s", "metrics_history_latest.png"),
    ],
)
GRAPH_DIAGNOSTICS_P_METRICS_HISTORY_CSV_SPEC = VisSpec(
    label="graph diagnostics metrics history p csvs",
    candidates=[
        ("graph_diagnostics_p", "metrics_history*.csv"),
        ("graph_diagnostics_s", "metrics_history*.csv"),
    ],
)
VIS_CTRL_SMOOTHNESS_Z_SPEC = VisSpec(
    label="Vis v Ctrl:Local smoothness (Z)",
    candidates=[
        ("vis_vis_ctrl", "smoothness_z_*.png"),
    ],
)
VIS_CTRL_SMOOTHNESS_P_SPEC = VisSpec(
    label="Vis v Ctrl:Local smoothness (P)",
    candidates=[
        ("vis_vis_ctrl", "smoothness_p_*.png"),
        ("vis_vis_ctrl", "smoothness_s_*.png"),
    ],
)
VIS_CTRL_SMOOTHNESS_H_SPEC = VisSpec(
    label="Vis v Ctrl:Local smoothness (H)",
    candidates=[
        ("vis_vis_ctrl", "smoothness_h_*.png"),
    ],
)
VIS_CTRL_COMPOSITION_Z_SPEC = VisSpec(
    label="Vis v Ctrl:Two-step composition error (Z)",
    candidates=[
        ("vis_vis_ctrl", "composition_error_z_*.png"),
    ],
)
VIS_CTRL_COMPOSITION_P_SPEC = VisSpec(
    label="Vis v Ctrl:Two-step composition error (P)",
    candidates=[
        ("vis_vis_ctrl", "composition_error_p_*.png"),
        ("vis_vis_ctrl", "composition_error_s_*.png"),
    ],
)
VIS_CTRL_COMPOSITION_H_SPEC = VisSpec(
    label="Vis v Ctrl:Two-step composition error (H)",
    candidates=[
        ("vis_vis_ctrl", "composition_error_h_*.png"),
    ],
)
VIS_CTRL_STABILITY_Z_SPEC = VisSpec(
    label="Vis v Ctrl:Neighborhood stability (Z)",
    candidates=[
        ("vis_vis_ctrl", "stability_z_*.png"),
    ],
)
VIS_CTRL_STABILITY_P_SPEC = VisSpec(
    label="Vis v Ctrl:Neighborhood stability (P)",
    candidates=[
        ("vis_vis_ctrl", "stability_p_*.png"),
        ("vis_vis_ctrl", "stability_s_*.png"),
    ],
)
VIS_CTRL_STABILITY_H_SPEC = VisSpec(
    label="Vis v Ctrl:Neighborhood stability (H)",
    candidates=[
        ("vis_vis_ctrl", "stability_h_*.png"),
    ],
)
VIS_CTRL_ALIGNMENT_Z_SPEC = VisSpec(
    label="vis_ctrl alignment_z images",
    candidates=[
        ("vis_action_alignment_z", "action_alignment_detail_*.png"),
    ],
)
VIS_CTRL_ALIGNMENT_P_SPEC = VisSpec(
    label="Diagnostics:Action alignment of PCA (P)",
    candidates=[
        ("vis_action_alignment_p", "action_alignment_detail_*.png"),
        ("vis_action_alignment_s", "action_alignment_detail_*.png"),
    ],
)
VIS_CTRL_ALIGNMENT_H_SPEC = VisSpec(
    label="Diagnostics:Action alignment of PCA (H)",
    candidates=[
        ("vis_action_alignment_h", "action_alignment_detail_*.png"),
    ],
)
SELF_DISTANCE_Z_IMAGES_SPEC = VisSpec(
    label="Self-distance:Distance (Z)",
    candidates=[
        ("vis_self_distance_z", "self_distance_z_*.png"),
        ("vis_self_distance", "self_distance_*.png"),
    ],
)
SELF_DISTANCE_P_IMAGES_SPEC = VisSpec(
    label="Self-distance:Distance (P)",
    candidates=[
        ("vis_self_distance_p", "self_distance_p_*.png"),
        ("vis_self_distance_s", "self_distance_s_*.png"),
        ("vis_state_embedding", "state_embedding_[0-9]*.png"),
    ],
)
SELF_DISTANCE_H_IMAGES_SPEC = VisSpec(
    label="Self-distance:Distance (H)",
    candidates=[
        ("vis_self_distance_h", "self_distance_h_*.png"),
    ],
)
STATE_EMBEDDING_CSVS_SPEC = VisSpec(
    label="state embedding CSV folders",
    candidates=[
        ("self_distance_s", "self_distance_s_*.csv"),
        ("state_embedding", "state_embedding_*.csv"),
    ],
)
SELF_DISTANCE_Z_CSVS_SPEC = VisSpec(
    label="self-distance CSV folders",
    candidates=[
        ("self_distance_z", "self_distance_z_*.csv"),
        ("self_distance", "self_distance_*.csv"),
    ],
)
SELF_DISTANCE_P_CSVS_SPEC = VisSpec(
    label="pose self-distance CSV folders",
    candidates=[
        ("self_distance_p", "self_distance_p_*.csv"),
        ("self_distance_s", "self_distance_s_*.csv"),
        ("state_embedding", "state_embedding_*.csv"),
    ],
)
SELF_DISTANCE_H_CSVS_SPEC = VisSpec(
    label="self-distance H CSV folders",
    candidates=[
        ("self_distance_h", "self_distance_h_*.csv"),
    ],
)
VIS_CTRL_METRICS_CSV_SPEC = VisSpec(
    label="vis ctrl CSV folders",
    candidates=[
        ("metrics", "vis_ctrl_metrics.csv"),
    ],
)
VIS_CTRL_VIS_CTRL_CSV_SPEC = VisSpec(
    label="vis ctrl CSV folders",
    candidates=[
        ("vis_ctrl", "vis_ctrl_metrics_*.csv"),
    ],
)
DIAGNOSTICS_DELTA_Z_PCA_CSV_FOLDERS = ["vis_delta_z_pca"]
DIAGNOSTICS_ACTION_ALIGNMENT_CSV_FOLDERS = ["vis_action_alignment_z", "vis_action_alignment"]
DIAGNOSTICS_CYCLE_ERROR_CSV_FOLDERS = ["vis_cycle_error_z", "vis_cycle_error"]
DIAGNOSTICS_FRAME_ALIGNMENT_CSV_FOLDERS = ["vis_diagnostics_frames"]
DIAGNOSTICS_ROLLOUT_DIVERGENCE_CSV_FOLDERS = ["vis_rollout_divergence_z", "vis_rollout_divergence"]
DIAGNOSTICS_ROLLOUT_DIVERGENCE_Z_CSV_FOLDERS = ["vis_rollout_divergence_z", "vis_rollout_divergence"]
DIAGNOSTICS_Z_CONSISTENCY_CSV_FOLDERS = ["vis_z_consistency"]
DIAGNOSTICS_Z_MONOTONICITY_CSV_FOLDERS = ["vis_z_monotonicity"]
DIAGNOSTICS_PATH_INDEPENDENCE_CSV_FOLDERS = ["vis_path_independence"]
DIAGNOSTICS_P_DELTA_P_PCA_CSV_FOLDERS = ["vis_delta_p_pca", "vis_delta_s_pca"]
DIAGNOSTICS_P_ACTION_ALIGNMENT_CSV_FOLDERS = ["vis_action_alignment_p", "vis_action_alignment_s"]
DIAGNOSTICS_P_CYCLE_ERROR_CSV_FOLDERS = ["vis_cycle_error_p", "vis_cycle_error_s"]
DIAGNOSTICS_P_ROLLOUT_DIVERGENCE_CSV_FOLDERS = ["vis_rollout_divergence_p", "vis_rollout_divergence_s"]

STATE_EMBEDDING_HIST_IMAGE_SPEC = VisSpec(
    label="state embedding hist images",
    candidates=[("vis_state_embedding", "state_embedding_hist_*.png")],
)
STATE_EMBEDDING_COSINE_IMAGE_SPEC = VisSpec(
    label="state embedding cosine images",
    candidates=[
        ("vis_self_distance_s", "self_distance_cosine_*.png"),
        ("vis_state_embedding", "state_embedding_cosine_*.png"),
    ],
)
STATE_EMBEDDING_DISTANCE_IMAGE_SPEC = VisSpec(
    label="state embedding distance images",
    candidates=[
        ("vis_self_distance_s", "self_distance_s_*.png"),
        ("vis_state_embedding", "state_embedding_[0-9]*.png"),
    ],
)
DIAGNOSTICS_P_DELTA_P_PCA_IMAGE_SPEC = VisSpec(
    label="Diagnostics:Delta-p PCA",
    candidates=[
        ("vis_delta_p_pca", "delta_p_pca_*.png"),
        ("vis_delta_s_pca", "delta_s_pca_*.png"),
    ],
)
DIAGNOSTICS_P_VARIANCE_SPECTRUM_IMAGE_SPEC = VisSpec(
    label="variance_spectrum_p image folders",
    candidates=[
        ("vis_delta_p_pca", "delta_p_variance_spectrum_*.png"),
        ("vis_delta_s_pca", "delta_s_variance_spectrum_*.png"),
    ],
)
DIAGNOSTICS_P_ACTION_ALIGNMENT_DETAIL_IMAGE_SPEC = VisSpec(
    label="action_alignment_detail_p image folders",
    candidates=[
        ("vis_action_alignment_p", "action_alignment_detail_*.png"),
        ("vis_action_alignment_s", "action_alignment_detail_*.png"),
    ],
)
DIAGNOSTICS_P_CYCLE_ERROR_IMAGE_SPEC = VisSpec(
    label="Diagnostics:Cycle error (P)",
    candidates=[
        ("vis_cycle_error_p", "*.png"),
        ("vis_cycle_error_s", "*.png"),
    ],
)
DIAGNOSTICS_P_STRAIGHTLINE_IMAGE_SPEC = VisSpec(
    label="Diagnostics:Straight-line P",
    candidates=[
        ("vis_straightline_p", "straightline_p_*.png"),
        ("vis_straightline_s", "straightline_s_*.png"),
    ],
)
DIAGNOSTICS_P_ROLLOUT_DIVERGENCE_IMAGE_SPEC = VisSpec(
    label="Diagnostics:Rollout divergence (P)",
    candidates=[
        ("vis_rollout_divergence_p", "rollout_divergence_p_*.png"),
        ("vis_rollout_divergence_s", "rollout_divergence_s_*.png"),
    ],
)
DIAGNOSTICS_DELTA_Z_PCA_IMAGE_SPEC = VisSpec(
    label="Diagnostics:Delta-z PCA",
    candidates=[("vis_delta_z_pca", "delta_z_pca_*.png")],
)
DIAGNOSTICS_VARIANCE_SPECTRUM_IMAGE_SPEC = VisSpec(
    label="variance_spectrum image folders",
    candidates=[("vis_delta_z_pca", "delta_z_variance_spectrum_*.png")],
)
DIAGNOSTICS_ACTION_ALIGNMENT_DETAIL_IMAGE_SPEC = VisSpec(
    label="Diagnostics:Action alignment of PCA (Z)",
    candidates=[
        ("vis_action_alignment_z", "action_alignment_detail_*.png"),
        ("vis_action_alignment", "action_alignment_detail_*.png"),
    ],
)
DIAGNOSTICS_CYCLE_ERROR_IMAGE_SPEC = VisSpec(
    label="Diagnostics:Cycle error (Z)",
    candidates=[
        ("vis_cycle_error_z", "*.png"),
        ("vis_cycle_error", "*.png"),
    ],
)
DIAGNOSTICS_ROLLOUT_DIVERGENCE_IMAGE_SPEC = VisSpec(
    label="Diagnostics:Rollout divergence",
    candidates=[
        ("vis_rollout_divergence_z", "rollout_divergence_z_*.png"),
        ("vis_rollout_divergence", "rollout_divergence_*.png"),
    ],
)
DIAGNOSTICS_Z_CONSISTENCY_IMAGE_SPEC = VisSpec(
    label="Diagnostics:Z consistency",
    candidates=[("vis_z_consistency", "z_consistency_*.png")],
)
DIAGNOSTICS_Z_MONOTONICITY_IMAGE_SPEC = VisSpec(
    label="Diagnostics:Z monotonicity",
    candidates=[("vis_z_monotonicity", "z_monotonicity_*.png")],
)
DIAGNOSTICS_PATH_INDEPENDENCE_IMAGE_SPEC = VisSpec(
    label="Diagnostics:Path independence",
    candidates=[("vis_path_independence", "path_independence_*.png")],
)
VIS_ODOMETRY_IMAGES_SPEC = VisSpec(
    label="vis_odometry images",
    candidates=[("vis_odometry", "*.png")],
)
QUICK_SELF_DISTANCE_Z_CSV_CANDIDATES = [
    ("self_distance_z", "self_distance_z", "self_distance_z_0000000.csv", "self_distance_z_*.csv"),
    ("self_distance", "self_distance", "self_distance_0000000.csv", "self_distance_*.csv"),
]
QUICK_SELF_DISTANCE_P_CSV_CANDIDATES = [
    ("self_distance_p", "self_distance_p", "self_distance_p_0000000.csv", "self_distance_p_*.csv"),
    ("self_distance_s", "self_distance_s", "self_distance_s_0000000.csv", "self_distance_s_*.csv"),
    ("state_embedding", "state_embedding", "state_embedding_0000000.csv", "state_embedding_*.csv"),
]
QUICK_STATE_EMBEDDING_CSV_CANDIDATES = [
    ("self_distance_s", "self_distance_s", "self_distance_s_0000000.csv", "self_distance_s_*.csv"),
    ("state_embedding", "state_embedding", "state_embedding_0000000.csv", "state_embedding_*.csv"),
]


VIS_STEP_SPECS: Dict[str, VisSpec] = {
    'vis_fixed_0': VisSpec(
        label='Rollouts:Fixed 0',
        candidates=[('vis_fixed_0', 'rollout_*.png', 'rollout_')],
    ),
    'vis_fixed_1': VisSpec(
        label='Rollouts:Fixed 1',
        candidates=[('vis_fixed_1', 'rollout_*.png', 'rollout_')],
    ),
    'vis_rolling_0': VisSpec(
        label='Rollouts:Rolling 0',
        candidates=[('vis_rolling_0', 'rollout_*.png', 'rollout_')],
    ),
    'vis_rolling_1': VisSpec(
        label='Rollouts:Rolling 1',
        candidates=[('vis_rolling_1', 'rollout_*.png', 'rollout_')],
    ),
    'embeddings': VisSpec(
        label=None,
        candidates=[('embeddings', 'embeddings_*.png', 'embeddings_')],
    ),
    'pca_z': VisSpec(
        label='Diagnostics:PCA (Z)',
        candidates=[
            ('pca_z', 'pca_z_*.png', 'pca_z_'),
            ('embeddings', 'embeddings_*.png', 'embeddings_'),
        ],
    ),
    'pca_p': VisSpec(
        label='Diagnostics:PCA (P)',
        candidates=[
            ('pca_p', 'pca_p_*.png', 'pca_p_'),
            ('pca_s', 'pca_s_*.png', 'pca_s_'),
        ],
    ),
    'pca_s': VisSpec(
        label=None,
        candidates=[('pca_s', 'pca_s_*.png', 'pca_s_')],
    ),
    'pca_h': VisSpec(
        label='Diagnostics:PCA (H)',
        candidates=[('pca_h', 'pca_h_*.png', 'pca_h_')],
    ),
    'samples_hard': VisSpec(
        label='Samples:Hard',
        candidates=[('samples_hard', 'hard_*.png', 'hard_')],
    ),
    'vis_self_distance_z': SELF_DISTANCE_Z_IMAGES_SPEC,
    'vis_self_distance_p': SELF_DISTANCE_P_IMAGES_SPEC,
    'vis_self_distance_s': VisSpec(
        label=None,
        candidates=[
            ('vis_self_distance_s', 'self_distance_s_*.png', 'self_distance_s_'),
            ('vis_state_embedding', 'state_embedding_[0-9]*.png', 'state_embedding_'),
        ],
    ),
    'vis_self_distance_h': SELF_DISTANCE_H_IMAGES_SPEC,
    'vis_delta_z_pca': DIAGNOSTICS_DELTA_Z_PCA_IMAGE_SPEC,
    'vis_delta_p_pca': DIAGNOSTICS_P_DELTA_P_PCA_IMAGE_SPEC,
    'vis_delta_s_pca': VisSpec(
        label=None,
        candidates=[('vis_delta_s_pca', 'delta_s_pca_*.png', 'delta_s_pca_')],
    ),
    'vis_delta_h_pca': VisSpec(
        label='Diagnostics:Delta-h PCA',
        candidates=[('vis_delta_h_pca', 'delta_h_pca_*.png', 'delta_h_pca_')],
    ),
    'vis_odometry_current_z': VisSpec(
        label='Odometry:Cumulative sum of Δz PCA/ICA/t-SNE',
        candidates=[('vis_odometry', 'odometry_z_*.png', 'odometry_z_')],
    ),
    'vis_odometry_current_p': VisSpec(
        label='Odometry:Cumulative sum of Δp PCA/ICA/t-SNE',
        candidates=[
            ('vis_odometry', 'odometry_p_*.png', 'odometry_p_'),
            ('vis_odometry', 'odometry_s_*.png', 'odometry_s_'),
        ],
    ),
    'vis_odometry_current_s': VisSpec(
        label=None,
        candidates=[('vis_odometry', 'odometry_s_*.png', 'odometry_s_')],
    ),
    'vis_odometry_current_h': VisSpec(
        label='Odometry:Cumulative sum of Δh PCA/ICA/t-SNE',
        candidates=[('vis_odometry', 'odometry_h_*.png', 'odometry_h_')],
    ),
    'vis_odometry_z_vs_z_hat': VisSpec(
        label='Odometry:||z - z_hat|| + scatter',
        candidates=[('vis_odometry', 'z_vs_z_hat_*.png', 'z_vs_z_hat_')],
    ),
    'vis_odometry_p_vs_p_hat': VisSpec(
        label='Odometry:||p - p_hat|| + scatter',
        candidates=[
            ('vis_odometry', 'p_vs_p_hat_*.png', 'p_vs_p_hat_'),
            ('vis_odometry', 's_vs_s_hat_*.png', 's_vs_s_hat_'),
        ],
    ),
    'vis_odometry_s_vs_s_hat': VisSpec(
        label=None,
        candidates=[('vis_odometry', 's_vs_s_hat_*.png', 's_vs_s_hat_')],
    ),
    'vis_odometry_h_vs_h_hat': VisSpec(
        label='Odometry:||h - h_hat|| + scatter',
        candidates=[('vis_odometry', 'h_vs_h_hat_*.png', 'h_vs_h_hat_')],
    ),
    'vis_action_alignment_detail_z': DIAGNOSTICS_ACTION_ALIGNMENT_DETAIL_IMAGE_SPEC,
    'vis_action_alignment_detail_raw_z': VisSpec(
        label='Diagnostics:Action alignment of raw delta (Z)',
        candidates=[('vis_action_alignment_z_raw', 'action_alignment_detail_*.png', 'action_alignment_detail_')],
    ),
    'vis_action_alignment_detail_centered_z': VisSpec(
        label='Diagnostics:Action alignment of centered delta (Z)',
        candidates=[('vis_action_alignment_z_centered', 'action_alignment_detail_*.png', 'action_alignment_detail_')],
    ),
    'vis_action_alignment_detail_p': VIS_CTRL_ALIGNMENT_P_SPEC,
    'vis_action_alignment_detail_raw_p': VisSpec(
        label='Diagnostics:Action alignment of raw delta (P)',
        candidates=[
            ('vis_action_alignment_p_raw', 'action_alignment_detail_*.png', 'action_alignment_detail_'),
            ('vis_action_alignment_s_raw', 'action_alignment_detail_*.png', 'action_alignment_detail_'),
        ],
    ),
    'vis_action_alignment_detail_centered_p': VisSpec(
        label='Diagnostics:Action alignment of centered delta (P)',
        candidates=[
            ('vis_action_alignment_p_centered', 'action_alignment_detail_*.png', 'action_alignment_detail_'),
            ('vis_action_alignment_s_centered', 'action_alignment_detail_*.png', 'action_alignment_detail_'),
        ],
    ),
    'vis_action_alignment_detail_s': VisSpec(
        label=None,
        candidates=[('vis_action_alignment_s', 'action_alignment_detail_*.png', 'action_alignment_detail_')],
    ),
    'vis_action_alignment_detail_raw_s': VisSpec(
        label=None,
        candidates=[('vis_action_alignment_s_raw', 'action_alignment_detail_*.png', 'action_alignment_detail_')],
    ),
    'vis_action_alignment_detail_centered_s': VisSpec(
        label=None,
        candidates=[('vis_action_alignment_s_centered', 'action_alignment_detail_*.png', 'action_alignment_detail_')],
    ),
    'vis_action_alignment_detail_h': VIS_CTRL_ALIGNMENT_H_SPEC,
    'vis_action_alignment_detail_raw_h': VisSpec(
        label='Diagnostics:Action alignment of raw delta (H)',
        candidates=[('vis_action_alignment_h_raw', 'action_alignment_detail_*.png', 'action_alignment_detail_')],
    ),
    'vis_action_alignment_detail_centered_h': VisSpec(
        label='Diagnostics:Action alignment of centered delta (H)',
        candidates=[('vis_action_alignment_h_centered', 'action_alignment_detail_*.png', 'action_alignment_detail_')],
    ),
    'vis_cycle_error': DIAGNOSTICS_CYCLE_ERROR_IMAGE_SPEC,
    'vis_cycle_error_p': DIAGNOSTICS_P_CYCLE_ERROR_IMAGE_SPEC,
    'vis_cycle_error_s': VisSpec(
        label=None,
        candidates=[('vis_cycle_error_s', 'cycle_error_*.png', 'cycle_error_')],
    ),
    'vis_cycle_error_h': VisSpec(
        label='Diagnostics:Cycle error (H)',
        candidates=[('vis_cycle_error_h', 'cycle_error_*.png', 'cycle_error_')],
    ),
    'vis_rollout_divergence': DIAGNOSTICS_ROLLOUT_DIVERGENCE_IMAGE_SPEC,
    'vis_rollout_divergence_z': VisSpec(
        label='Diagnostics:Rollout divergence (Z)',
        candidates=[('vis_rollout_divergence_z', 'rollout_divergence_z_*.png', 'rollout_divergence_z_')],
    ),
    'vis_rollout_divergence_h': VisSpec(
        label='Diagnostics:Rollout divergence (H)',
        candidates=[('vis_rollout_divergence_h', 'rollout_divergence_h_*.png', 'rollout_divergence_h_')],
    ),
    'vis_rollout_divergence_p': DIAGNOSTICS_P_ROLLOUT_DIVERGENCE_IMAGE_SPEC,
    'vis_z_consistency': DIAGNOSTICS_Z_CONSISTENCY_IMAGE_SPEC,
    'vis_z_monotonicity': DIAGNOSTICS_Z_MONOTONICITY_IMAGE_SPEC,
    'vis_path_independence': DIAGNOSTICS_PATH_INDEPENDENCE_IMAGE_SPEC,
    'vis_straightline_p': DIAGNOSTICS_P_STRAIGHTLINE_IMAGE_SPEC,
    'vis_straightline_s': VisSpec(
        label=None,
        candidates=[('vis_straightline_s', 'straightline_s_*.png', 'straightline_s_')],
    ),
    'vis_h_ablation': VisSpec(
        label='Diagnostics:H ablation divergence',
        candidates=[('vis_h_ablation', 'h_ablation_*.png', 'h_ablation_')],
    ),
    'vis_h_drift_by_action': VisSpec(
        label='Diagnostics:H drift by action',
        candidates=[('vis_h_drift_by_action', 'h_drift_by_action_*.png', 'h_drift_by_action_')],
    ),
    'vis_norm_timeseries': VisSpec(
        label='Diagnostics:Norm stability',
        candidates=[('vis_norm_timeseries', 'norm_timeseries_*.png', 'norm_timeseries_')],
    ),
    'vis_graph_rank1_cdf_z': GRAPH_DIAGNOSTICS_Z_RANK1_CDF_SPEC,
    'vis_graph_rank2_cdf_z': GRAPH_DIAGNOSTICS_Z_RANK2_CDF_SPEC,
    'vis_graph_neff_violin_z': GRAPH_DIAGNOSTICS_Z_NEFF_VIOLIN_SPEC,
    'vis_graph_in_degree_hist_z': GRAPH_DIAGNOSTICS_Z_IN_DEGREE_HIST_SPEC,
    'vis_graph_edge_consistency_z': GRAPH_DIAGNOSTICS_Z_EDGE_CONSISTENCY_SPEC,
    'vis_graph_metrics_history_z': GRAPH_DIAGNOSTICS_Z_METRICS_HISTORY_SPEC,
    'vis_graph_rank1_cdf_h': GRAPH_DIAGNOSTICS_H_RANK1_CDF_SPEC,
    'vis_graph_rank2_cdf_h': GRAPH_DIAGNOSTICS_H_RANK2_CDF_SPEC,
    'vis_graph_neff_violin_h': GRAPH_DIAGNOSTICS_H_NEFF_VIOLIN_SPEC,
    'vis_graph_in_degree_hist_h': GRAPH_DIAGNOSTICS_H_IN_DEGREE_HIST_SPEC,
    'vis_graph_edge_consistency_h': GRAPH_DIAGNOSTICS_H_EDGE_CONSISTENCY_SPEC,
    'vis_graph_metrics_history_h': GRAPH_DIAGNOSTICS_H_METRICS_HISTORY_SPEC,
    'vis_graph_rank1_cdf_p': GRAPH_DIAGNOSTICS_P_RANK1_CDF_SPEC,
    'vis_graph_rank2_cdf_p': GRAPH_DIAGNOSTICS_P_RANK2_CDF_SPEC,
    'vis_graph_neff_violin_p': GRAPH_DIAGNOSTICS_P_NEFF_VIOLIN_SPEC,
    'vis_graph_in_degree_hist_p': GRAPH_DIAGNOSTICS_P_IN_DEGREE_HIST_SPEC,
    'vis_graph_edge_consistency_p': GRAPH_DIAGNOSTICS_P_EDGE_CONSISTENCY_SPEC,
    'vis_graph_metrics_history_p': GRAPH_DIAGNOSTICS_P_METRICS_HISTORY_SPEC,
    'vis_graph_rank1_cdf_s': VisSpec(
        label=None,
        candidates=[('graph_diagnostics_s', 'rank1_cdf_*.png', 'rank1_cdf_')],
    ),
    'vis_graph_rank2_cdf_s': VisSpec(
        label=None,
        candidates=[('graph_diagnostics_s', 'rank2_cdf_*.png', 'rank2_cdf_')],
    ),
    'vis_graph_neff_violin_s': VisSpec(
        label=None,
        candidates=[('graph_diagnostics_s', 'neff_violin_*.png', 'neff_violin_')],
    ),
    'vis_graph_in_degree_hist_s': VisSpec(
        label=None,
        candidates=[('graph_diagnostics_s', 'in_degree_hist_*.png', 'in_degree_hist_')],
    ),
    'vis_graph_edge_consistency_s': VisSpec(
        label=None,
        candidates=[('graph_diagnostics_s', 'edge_consistency_*.png', 'edge_consistency_')],
    ),
    'vis_graph_metrics_history_s': VisSpec(
        label=None,
        candidates=[('graph_diagnostics_s', 'metrics_history_*.png', 'metrics_history_')],
    ),
    'vis_ctrl_smoothness_z': VIS_CTRL_SMOOTHNESS_Z_SPEC,
    'vis_ctrl_smoothness_p': VIS_CTRL_SMOOTHNESS_P_SPEC,
    'vis_ctrl_smoothness_s': VisSpec(
        label=None,
        candidates=[('vis_vis_ctrl', 'smoothness_s_*.png', 'smoothness_s_')],
    ),
    'vis_ctrl_smoothness_h': VIS_CTRL_SMOOTHNESS_H_SPEC,
    'vis_ctrl_composition_z': VIS_CTRL_COMPOSITION_Z_SPEC,
    'vis_ctrl_composition_p': VIS_CTRL_COMPOSITION_P_SPEC,
    'vis_ctrl_composition_s': VisSpec(
        label=None,
        candidates=[('vis_vis_ctrl', 'composition_error_s_*.png', 'composition_error_s_')],
    ),
    'vis_ctrl_composition_h': VIS_CTRL_COMPOSITION_H_SPEC,
    'vis_composability_z': VisSpec(
        label='Composability:Two-step (Z)',
        candidates=[('vis_composability_z', 'composability_z_*.png', 'composability_z_')],
    ),
    'vis_composability_p': VisSpec(
        label='Composability:Two-step (P)',
        candidates=[
            ('vis_composability_p', 'composability_p_*.png', 'composability_p_'),
            ('vis_composability_s', 'composability_s_*.png', 'composability_s_'),
        ],
    ),
    'vis_composability_s': VisSpec(
        label=None,
        candidates=[('vis_composability_s', 'composability_s_*.png', 'composability_s_')],
    ),
    'vis_composability_h': VisSpec(
        label='Composability:Two-step (H)',
        candidates=[('vis_composability_h', 'composability_h_*.png', 'composability_h_')],
    ),
    'vis_ctrl_stability_z': VIS_CTRL_STABILITY_Z_SPEC,
    'vis_ctrl_stability_p': VIS_CTRL_STABILITY_P_SPEC,
    'vis_ctrl_stability_s': VisSpec(
        label=None,
        candidates=[('vis_vis_ctrl', 'stability_s_*.png', 'stability_s_')],
    ),
    'vis_ctrl_stability_h': VIS_CTRL_STABILITY_H_SPEC,
}


def _image_folder_sort_key(option: Dict[str, object]) -> Tuple[str, str, str]:
    label = str(option.get("label", ""))
    group = label.split(":")[0].strip() if label else ""
    return group, label, str(option.get("value", ""))


def get_image_folder_specs(root: Path) -> List[Dict[str, object]]:
    options = [
        {"value": value, "label": spec.label}
        for value, spec in VIS_STEP_SPECS.items()
        if spec.label
    ]
    resolved_options: List[Dict[str, object]] = []
    for option in sorted(options, key=_image_folder_sort_key):
        value = str(option.get("value", ""))
        if not value:
            continue
        spec = VIS_STEP_SPECS.get(value)
        if spec is None:
            continue
        candidates = spec.candidates
        spec_prefix = spec.prefix
        selected = None
        for candidate in candidates:
            if len(candidate) == 3:
                folder_name, pattern, candidate_prefix = candidate
            else:
                folder_name, pattern = candidate
                candidate_prefix = ""
            folder = root / folder_name
            if _first_matching_file(folder, exact_name=None, pattern=pattern) is not None:
                selected = (folder_name, candidate_prefix)
                break
        if selected is None and candidates:
            candidate = candidates[0]
            if len(candidate) == 3:
                folder_name, _, candidate_prefix = candidate
            else:
                folder_name, _ = candidate
                candidate_prefix = ""
            selected = (folder_name, candidate_prefix)
        if selected is None:
            continue
        folder_name, prefix = selected
        if spec_prefix is not None:
            prefix = spec_prefix
        resolved_options.append(
            {
                "value": value,
                "label": option.get("label"),
                "folder": folder_name,
                "prefix": prefix,
            }
        )
    return resolved_options


@dataclass
class Experiment:
    """Metadata bundled for rendering experiment summaries."""

    # Core metadata and filesystem references.
    id: str
    name: str
    path: Path
    metadata_text: str
    data_root: Optional[str]
    model_diff_text: str
    model_diff_items: List[Tuple[str, str, bool]]
    git_metadata_text: str
    git_commit: str
    notes_text: str
    title: str
    tags: str
    starred: bool
    archived: bool

    # Global training curves and rollout steps.
    loss_image: Optional[Path]
    loss_csv: Optional[Path]
    rollout_steps: List[int]
    max_step: Optional[Union[int, str]]
    last_modified: Optional[datetime]
    total_params: Optional[int]
    flops_per_step: Optional[int]

    # Self-distance + state-embedding assets (Z/P/S).
    self_distance_z_csv: Optional[Path]
    self_distance_z_images: List[Path]
    self_distance_z_csvs: List[Path]
    self_distance_h_csv: Optional[Path]
    self_distance_h_images: List[Path]
    self_distance_h_csvs: List[Path]
    self_distance_p_csv: Optional[Path]
    self_distance_p_images: List[Path]
    self_distance_p_csvs: List[Path]
    state_embedding_csv: Optional[Path]
    state_embedding_images: List[Path]
    state_embedding_hist_images: List[Path]
    state_embedding_csvs: List[Path]

    # Odometry visuals.
    odometry_images: List[Path]
    has_odometry_images: bool

    # Diagnostics images (Z).
    diagnostics_delta_z_pca_images: List[Path]
    diagnostics_variance_spectrum_images: List[Path]
    diagnostics_action_alignment_detail_images: List[Path]
    diagnostics_cycle_error_images: List[Path]
    diagnostics_rollout_divergence_images: List[Path]
    diagnostics_rollout_divergence_z_images: List[Path]
    diagnostics_z_consistency_images: List[Path]
    diagnostics_z_monotonicity_images: List[Path]
    diagnostics_path_independence_images: List[Path]
    diagnostics_z_steps: List[int]
    has_diagnostics_z_steps: bool

    # Diagnostics CSVs (Z).
    diagnostics_delta_z_pca_csvs: List[Path]
    diagnostics_action_alignment_csvs: List[Path]
    diagnostics_cycle_error_csvs: List[Path]
    diagnostics_frame_alignment_csvs: List[Path]
    diagnostics_rollout_divergence_csvs: List[Path]
    diagnostics_rollout_divergence_z_csvs: List[Path]
    diagnostics_z_consistency_csvs: List[Path]
    diagnostics_z_monotonicity_csvs: List[Path]
    diagnostics_path_independence_csvs: List[Path]

    # Diagnostics frames (parallel arrays: steps -> entries).
    diagnostics_frame_steps: List[int]
    diagnostics_frame_entries: List[List[Tuple[Path, str, str, Optional[int]]]]

    # Diagnostics images (P).
    diagnostics_p_delta_p_pca_images: List[Path]
    diagnostics_p_variance_spectrum_images: List[Path]
    diagnostics_p_action_alignment_detail_images: List[Path]
    diagnostics_p_cycle_error_images: List[Path]
    diagnostics_p_straightline_images: List[Path]
    diagnostics_p_rollout_divergence_images: List[Path]
    diagnostics_p_steps: List[int]
    diagnostics_h_steps: List[int]
    has_diagnostics_p_steps: bool
    has_diagnostics_h_steps: bool

    # Diagnostics CSVs (P).
    diagnostics_p_delta_p_pca_csvs: List[Path]
    diagnostics_p_action_alignment_csvs: List[Path]
    diagnostics_p_cycle_error_csvs: List[Path]
    diagnostics_p_rollout_divergence_csvs: List[Path]

    # Graph diagnostics images (Z/H/P).
    graph_diagnostics_rank1_cdf_z_images: List[Path]
    graph_diagnostics_rank1_cdf_h_images: List[Path]
    graph_diagnostics_rank1_cdf_p_images: List[Path]
    graph_diagnostics_rank2_cdf_z_images: List[Path]
    graph_diagnostics_rank2_cdf_h_images: List[Path]
    graph_diagnostics_rank2_cdf_p_images: List[Path]
    graph_diagnostics_neff_violin_z_images: List[Path]
    graph_diagnostics_neff_violin_h_images: List[Path]
    graph_diagnostics_neff_violin_p_images: List[Path]
    graph_diagnostics_in_degree_hist_z_images: List[Path]
    graph_diagnostics_in_degree_hist_h_images: List[Path]
    graph_diagnostics_in_degree_hist_p_images: List[Path]
    graph_diagnostics_edge_consistency_z_images: List[Path]
    graph_diagnostics_edge_consistency_h_images: List[Path]
    graph_diagnostics_edge_consistency_p_images: List[Path]
    graph_diagnostics_metrics_history_z_images: List[Path]
    graph_diagnostics_metrics_history_h_images: List[Path]
    graph_diagnostics_metrics_history_p_images: List[Path]
    graph_diagnostics_metrics_history_latest_z_images: List[Path]
    graph_diagnostics_metrics_history_latest_h_images: List[Path]
    graph_diagnostics_metrics_history_latest_p_images: List[Path]

    # Graph diagnostics steps + CSVs (parallel per axis).
    graph_diagnostics_z_steps: List[int]
    graph_diagnostics_metrics_history_z_csvs: List[Path]
    graph_diagnostics_h_steps: List[int]
    graph_diagnostics_metrics_history_h_csvs: List[Path]
    graph_diagnostics_p_steps: List[int]
    graph_diagnostics_metrics_history_p_csvs: List[Path]

    # Vis ctrl images (Z/P/H) + steps.
    vis_ctrl_smoothness_z_images: List[Path]
    vis_ctrl_smoothness_p_images: List[Path]
    vis_ctrl_smoothness_h_images: List[Path]
    vis_ctrl_composition_z_images: List[Path]
    vis_ctrl_composition_p_images: List[Path]
    vis_ctrl_composition_h_images: List[Path]
    vis_ctrl_stability_z_images: List[Path]
    vis_ctrl_stability_p_images: List[Path]
    vis_ctrl_stability_h_images: List[Path]
    vis_ctrl_alignment_z_images: List[Path]
    vis_ctrl_alignment_p_images: List[Path]
    vis_ctrl_alignment_h_images: List[Path]
    vis_ctrl_steps: List[int]
    vis_ctrl_csvs: List[Path]

    # Convenience flags for dashboard gating.
    has_self_distance_z_csv: bool
    has_self_distance_h_csv: bool
    has_self_distance_p_csv: bool
    has_model_diff: bool

    def asset_exists(self, relative: str) -> bool:
        return (self.path / relative).exists()


@dataclass
class ExperimentIndex:
    """Lightweight index row for pagination."""

    id: str
    path: Path
    last_modified: Optional[datetime]
    starred: bool
    archived: bool


@dataclass
class LossCurveData:
    steps: List[float]
    cumulative_flops: List[float]
    elapsed_seconds: List[float]
    series: Dict[str, List[float]]


def list_experiments(output_dir: Path) -> List[Experiment]:
    experiments: List[Experiment] = []
    if not output_dir.exists():
        return experiments
    for subdir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        exp = load_experiment(subdir)
        if exp is not None:
            experiments.append(exp)
    experiments.sort(key=lambda e: e.name, reverse=True)
    return experiments


def build_experiment_index(output_dir: Path) -> List[ExperimentIndex]:
    """Build a lightweight listing without reading full metadata."""
    index: List[ExperimentIndex] = []
    if not output_dir.exists():
        return index
    for subdir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        last_modified = _quick_last_modified(subdir)
        metadata_custom_path = subdir / "experiment_metadata.txt"
        _, _, starred, archived = _read_metadata(metadata_custom_path)
        index.append(
            ExperimentIndex(
                id=subdir.name,
                path=subdir,
                last_modified=last_modified,
                starred=starred,
                archived=archived,
            )
        )
    index.sort(key=lambda e: e.id, reverse=True)
    return index


def _load_experiment_graph_diagnostics_z(
    experiment: Experiment,
    path: Path,
    *,
    include_graph_diagnostics: bool,
) -> None:
    if include_graph_diagnostics:
        experiment.graph_diagnostics_rank1_cdf_z_images = _collect_from_spec(path, GRAPH_DIAGNOSTICS_Z_RANK1_CDF_SPEC)
        experiment.graph_diagnostics_rank2_cdf_z_images = _collect_from_spec(path, GRAPH_DIAGNOSTICS_Z_RANK2_CDF_SPEC)
        experiment.graph_diagnostics_neff_violin_z_images = _collect_from_spec(path, GRAPH_DIAGNOSTICS_Z_NEFF_VIOLIN_SPEC)
        experiment.graph_diagnostics_in_degree_hist_z_images = _collect_from_spec(
            path, GRAPH_DIAGNOSTICS_Z_IN_DEGREE_HIST_SPEC
        )
        experiment.graph_diagnostics_edge_consistency_z_images = _collect_from_spec(
            path, GRAPH_DIAGNOSTICS_Z_EDGE_CONSISTENCY_SPEC
        )
        experiment.graph_diagnostics_metrics_history_z_images = _collect_from_spec(
            path, GRAPH_DIAGNOSTICS_Z_METRICS_HISTORY_SPEC
        )
        experiment.graph_diagnostics_metrics_history_latest_z_images = _collect_from_spec(
            path, GRAPH_DIAGNOSTICS_Z_METRICS_HISTORY_LATEST_SPEC
        )
        experiment.graph_diagnostics_metrics_history_z_csvs = _collect_from_spec(
            path, GRAPH_DIAGNOSTICS_Z_METRICS_HISTORY_CSV_SPEC, collect_all=False
        )
        experiment.graph_diagnostics_z_steps = _merge_steps(
            _collect_steps_from_path_list(experiment.graph_diagnostics_rank1_cdf_z_images),
            _collect_steps_from_path_list(experiment.graph_diagnostics_rank2_cdf_z_images),
            _collect_steps_from_path_list(experiment.graph_diagnostics_neff_violin_z_images),
            _collect_steps_from_path_list(experiment.graph_diagnostics_in_degree_hist_z_images),
            _collect_steps_from_path_list(experiment.graph_diagnostics_edge_consistency_z_images),
            _collect_steps_from_path_list(experiment.graph_diagnostics_metrics_history_z_images),
            _collect_steps_from_path_list(experiment.graph_diagnostics_metrics_history_latest_z_images),
        )
    else:
        graph_folder = _resolve_first_existing_folder(path, GRAPH_DIAGNOSTICS_Z_FOLDER_CANDIDATES)
        experiment.graph_diagnostics_z_steps = (
            [0] if graph_folder and _folder_has_any_file(graph_folder, (".png", ".csv")) else []
        )


def _load_experiment_graph_diagnostics_h(
    experiment: Experiment,
    path: Path,
    *,
    include_graph_diagnostics_h: bool,
) -> None:
    if include_graph_diagnostics_h:
        experiment.graph_diagnostics_rank1_cdf_h_images = _collect_from_spec(path, GRAPH_DIAGNOSTICS_H_RANK1_CDF_SPEC)
        experiment.graph_diagnostics_rank2_cdf_h_images = _collect_from_spec(path, GRAPH_DIAGNOSTICS_H_RANK2_CDF_SPEC)
        experiment.graph_diagnostics_neff_violin_h_images = _collect_from_spec(path, GRAPH_DIAGNOSTICS_H_NEFF_VIOLIN_SPEC)
        experiment.graph_diagnostics_in_degree_hist_h_images = _collect_from_spec(
            path, GRAPH_DIAGNOSTICS_H_IN_DEGREE_HIST_SPEC
        )
        experiment.graph_diagnostics_edge_consistency_h_images = _collect_from_spec(
            path, GRAPH_DIAGNOSTICS_H_EDGE_CONSISTENCY_SPEC
        )
        experiment.graph_diagnostics_metrics_history_h_images = _collect_from_spec(
            path, GRAPH_DIAGNOSTICS_H_METRICS_HISTORY_SPEC
        )
        experiment.graph_diagnostics_metrics_history_latest_h_images = _collect_from_spec(
            path, GRAPH_DIAGNOSTICS_H_METRICS_HISTORY_LATEST_SPEC
        )
        experiment.graph_diagnostics_metrics_history_h_csvs = _collect_from_spec(
            path, GRAPH_DIAGNOSTICS_H_METRICS_HISTORY_CSV_SPEC, collect_all=False
        )
        experiment.graph_diagnostics_h_steps = _merge_steps(
            _collect_steps_from_path_list(experiment.graph_diagnostics_rank1_cdf_h_images),
            _collect_steps_from_path_list(experiment.graph_diagnostics_rank2_cdf_h_images),
            _collect_steps_from_path_list(experiment.graph_diagnostics_neff_violin_h_images),
            _collect_steps_from_path_list(experiment.graph_diagnostics_in_degree_hist_h_images),
            _collect_steps_from_path_list(experiment.graph_diagnostics_edge_consistency_h_images),
            _collect_steps_from_path_list(experiment.graph_diagnostics_metrics_history_h_images),
            _collect_steps_from_path_list(experiment.graph_diagnostics_metrics_history_latest_h_images),
        )
    else:
        graph_h_folder = _resolve_first_existing_folder(path, GRAPH_DIAGNOSTICS_H_FOLDER_CANDIDATES)
        experiment.graph_diagnostics_h_steps = (
            [0] if graph_h_folder and _folder_has_any_file(graph_h_folder, (".png", ".csv")) else []
        )


def _load_experiment_graph_diagnostics_p(
    experiment: Experiment,
    path: Path,
    *,
    include_graph_diagnostics_p: bool,
) -> None:
    if include_graph_diagnostics_p:
        experiment.graph_diagnostics_rank1_cdf_p_images = _collect_from_spec(path, GRAPH_DIAGNOSTICS_P_RANK1_CDF_SPEC)
        experiment.graph_diagnostics_rank2_cdf_p_images = _collect_from_spec(path, GRAPH_DIAGNOSTICS_P_RANK2_CDF_SPEC)
        experiment.graph_diagnostics_neff_violin_p_images = _collect_from_spec(path, GRAPH_DIAGNOSTICS_P_NEFF_VIOLIN_SPEC)
        experiment.graph_diagnostics_in_degree_hist_p_images = _collect_from_spec(
            path, GRAPH_DIAGNOSTICS_P_IN_DEGREE_HIST_SPEC
        )
        experiment.graph_diagnostics_edge_consistency_p_images = _collect_from_spec(
            path, GRAPH_DIAGNOSTICS_P_EDGE_CONSISTENCY_SPEC
        )
        experiment.graph_diagnostics_metrics_history_p_images = _collect_from_spec(
            path, GRAPH_DIAGNOSTICS_P_METRICS_HISTORY_SPEC
        )
        experiment.graph_diagnostics_metrics_history_latest_p_images = _collect_from_spec(
            path, GRAPH_DIAGNOSTICS_P_METRICS_HISTORY_LATEST_SPEC
        )
        experiment.graph_diagnostics_metrics_history_p_csvs = _collect_from_spec(
            path, GRAPH_DIAGNOSTICS_P_METRICS_HISTORY_CSV_SPEC, collect_all=False
        )
        experiment.graph_diagnostics_p_steps = _merge_steps(
            _collect_steps_from_path_list(experiment.graph_diagnostics_rank1_cdf_p_images),
            _collect_steps_from_path_list(experiment.graph_diagnostics_rank2_cdf_p_images),
            _collect_steps_from_path_list(experiment.graph_diagnostics_neff_violin_p_images),
            _collect_steps_from_path_list(experiment.graph_diagnostics_in_degree_hist_p_images),
            _collect_steps_from_path_list(experiment.graph_diagnostics_edge_consistency_p_images),
            _collect_steps_from_path_list(experiment.graph_diagnostics_metrics_history_p_images),
            _collect_steps_from_path_list(experiment.graph_diagnostics_metrics_history_latest_p_images),
        )
    else:
        graph_p_folder = _resolve_first_existing_folder(path, GRAPH_DIAGNOSTICS_P_FOLDER_CANDIDATES)
        experiment.graph_diagnostics_p_steps = (
            [0] if graph_p_folder and _folder_has_any_file(graph_p_folder, (".png", ".csv")) else []
        )


def _load_experiment_self_distance(
    experiment: Experiment,
    path: Path,
    *,
    include_self_distance: bool,
) -> None:
    section_start = time.perf_counter()
    if include_self_distance:
        csvs = _collect_from_spec(path, SELF_DISTANCE_Z_CSVS_SPEC, collect_all=False)
        experiment.self_distance_z_csvs = csvs
        experiment.self_distance_z_csv = csvs[-1] if csvs else None
        experiment.self_distance_z_images = _collect_from_spec(path, SELF_DISTANCE_Z_IMAGES_SPEC)
    else:
        experiment.self_distance_z_csv = _first_matching_csv_candidate(
            path,
            QUICK_SELF_DISTANCE_Z_CSV_CANDIDATES,
            conflict_label="self-distance",
        )
    _profile(
        "load_experiment.self_distance",
        section_start,
        path,
        csvs=len(experiment.self_distance_z_csvs),
        images=len(experiment.self_distance_z_images),
        included=include_self_distance,
    )


def _load_experiment_self_distance_p(
    experiment: Experiment,
    path: Path,
    *,
    include_self_distance_p: bool,
) -> None:
    section_start = time.perf_counter()
    if include_self_distance_p:
        csvs = _collect_from_spec(path, SELF_DISTANCE_P_CSVS_SPEC, collect_all=False)
        experiment.self_distance_p_csvs = csvs
        experiment.self_distance_p_csv = csvs[-1] if csvs else None
        experiment.self_distance_p_images = _collect_from_spec(path, SELF_DISTANCE_P_IMAGES_SPEC)
    else:
        experiment.self_distance_p_csv = _first_matching_csv_candidate(
            path,
            QUICK_SELF_DISTANCE_P_CSV_CANDIDATES,
            conflict_label="self-distance (P)",
        )
    _profile(
        "load_experiment.self_distance_p",
        section_start,
        path,
        csvs=len(experiment.self_distance_p_csvs),
        images=len(experiment.self_distance_p_images),
        included=include_self_distance_p,
    )


def _load_experiment_state_embedding(
    experiment: Experiment,
    path: Path,
    *,
    include_state_embedding: bool,
) -> None:
    section_start = time.perf_counter()
    if include_state_embedding:
        experiment.state_embedding_hist_images = _collect_from_spec(path, STATE_EMBEDDING_HIST_IMAGE_SPEC)
        experiment.state_embedding_images = _collect_state_embedding_distance_images(path)
        csvs = _collect_from_spec(path, STATE_EMBEDDING_CSVS_SPEC, collect_all=False)
        experiment.state_embedding_csvs = csvs
        experiment.state_embedding_csv = csvs[-1] if csvs else None
    else:
        experiment.state_embedding_csv = _first_matching_csv_candidate(
            path,
            QUICK_STATE_EMBEDDING_CSV_CANDIDATES,
            conflict_label="state embedding",
        )
    _profile(
        "load_experiment.state_embedding",
        section_start,
        path,
        images=len(experiment.state_embedding_images) + len(experiment.state_embedding_hist_images),
        csvs=len(experiment.state_embedding_csvs),
        included=include_state_embedding,
    )


def _load_experiment_odometry(
    experiment: Experiment,
    path: Path,
    *,
    include_odometry: bool,
) -> None:
    section_start = time.perf_counter()
    if include_odometry:
        experiment.odometry_images = _collect_from_spec(path, VIS_ODOMETRY_IMAGES_SPEC)
        probe = experiment.odometry_images
    else:
        probe = _collect_from_spec(path, VIS_ODOMETRY_IMAGES_SPEC, collect_all=False)
    experiment.has_odometry_images = bool(probe)
    _profile(
        "load_experiment.odometry",
        section_start,
        path,
        images=len(experiment.odometry_images),
        included=include_odometry,
    )


def _load_experiment_diagnostics(
    experiment: Experiment,
    path: Path,
    *,
    include_diagnostics_images: bool,
    include_diagnostics_frames: bool,
) -> int:
    if include_diagnostics_images:
        experiment.diagnostics_delta_z_pca_images = _collect_from_spec(path, DIAGNOSTICS_DELTA_Z_PCA_IMAGE_SPEC)
        experiment.diagnostics_variance_spectrum_images = _collect_from_spec(path, DIAGNOSTICS_VARIANCE_SPECTRUM_IMAGE_SPEC)
        experiment.diagnostics_action_alignment_detail_images = _collect_from_spec(
            path, DIAGNOSTICS_ACTION_ALIGNMENT_DETAIL_IMAGE_SPEC
        )
        experiment.diagnostics_cycle_error_images = _collect_from_spec(path, DIAGNOSTICS_CYCLE_ERROR_IMAGE_SPEC)
        experiment.diagnostics_rollout_divergence_images = _collect_from_spec(path, DIAGNOSTICS_ROLLOUT_DIVERGENCE_IMAGE_SPEC)
        experiment.diagnostics_rollout_divergence_z_images = experiment.diagnostics_rollout_divergence_images
        experiment.diagnostics_z_consistency_images = _collect_from_spec(path, DIAGNOSTICS_Z_CONSISTENCY_IMAGE_SPEC)
        experiment.diagnostics_z_monotonicity_images = _collect_from_spec(path, DIAGNOSTICS_Z_MONOTONICITY_IMAGE_SPEC)
        experiment.diagnostics_path_independence_images = _collect_from_spec(path, DIAGNOSTICS_PATH_INDEPENDENCE_IMAGE_SPEC)
        experiment.diagnostics_z_steps = _merge_steps(
            _collect_steps_from_path_list(experiment.diagnostics_delta_z_pca_images),
            _collect_steps_from_path_list(experiment.diagnostics_variance_spectrum_images),
            _collect_steps_from_path_list(experiment.diagnostics_action_alignment_detail_images),
            _collect_steps_from_path_list(experiment.diagnostics_cycle_error_images),
            _collect_steps_from_path_list(experiment.diagnostics_rollout_divergence_images),
            _collect_steps_from_path_list(experiment.diagnostics_rollout_divergence_z_images),
            _collect_steps_from_path_list(experiment.diagnostics_z_consistency_images),
            _collect_steps_from_path_list(experiment.diagnostics_z_monotonicity_images),
            _collect_steps_from_path_list(experiment.diagnostics_path_independence_images),
        )
        experiment.has_diagnostics_z_steps = bool(experiment.diagnostics_z_steps)
        experiment.diagnostics_delta_z_pca_csvs = _collect_csvs_by_folders(path, DIAGNOSTICS_DELTA_Z_PCA_CSV_FOLDERS)
        experiment.diagnostics_action_alignment_csvs = _collect_csvs_by_folders(
            path, DIAGNOSTICS_ACTION_ALIGNMENT_CSV_FOLDERS
        )
        experiment.diagnostics_cycle_error_csvs = _collect_csvs_by_folders(path, DIAGNOSTICS_CYCLE_ERROR_CSV_FOLDERS)
        experiment.diagnostics_frame_alignment_csvs = _collect_csvs_by_folders(path, DIAGNOSTICS_FRAME_ALIGNMENT_CSV_FOLDERS)
        experiment.diagnostics_rollout_divergence_csvs = _collect_csvs_by_folders(
            path, DIAGNOSTICS_ROLLOUT_DIVERGENCE_CSV_FOLDERS
        )
        experiment.diagnostics_rollout_divergence_z_csvs = _collect_csvs_by_folders(
            path, DIAGNOSTICS_ROLLOUT_DIVERGENCE_Z_CSV_FOLDERS
        )
        experiment.diagnostics_z_consistency_csvs = _collect_csvs_by_folders(path, DIAGNOSTICS_Z_CONSISTENCY_CSV_FOLDERS)
        experiment.diagnostics_z_monotonicity_csvs = _collect_csvs_by_folders(path, DIAGNOSTICS_Z_MONOTONICITY_CSV_FOLDERS)
        experiment.diagnostics_path_independence_csvs = _collect_csvs_by_folders(
            path, DIAGNOSTICS_PATH_INDEPENDENCE_CSV_FOLDERS
        )
    else:
        experiment.diagnostics_z_steps = [0] if _diagnostics_suffix_exists(path, DIAGNOSTICS_Z_DIRS) else []
        experiment.has_diagnostics_z_steps = bool(experiment.diagnostics_z_steps)

    diagnostics_frames = _collect_diagnostics_frames(path) if include_diagnostics_frames else {}
    if diagnostics_frames:
        experiment.diagnostics_frame_steps = sorted(diagnostics_frames)
        experiment.diagnostics_frame_entries = [diagnostics_frames[step] for step in experiment.diagnostics_frame_steps]
    _load_experiment_diagnostics_h(experiment, path)
    return (
        len(experiment.diagnostics_delta_z_pca_csvs)
        + len(experiment.diagnostics_action_alignment_csvs)
        + len(experiment.diagnostics_cycle_error_csvs)
        + len(experiment.diagnostics_frame_alignment_csvs)
        + len(experiment.diagnostics_rollout_divergence_csvs)
        + len(experiment.diagnostics_rollout_divergence_z_csvs)
        + len(experiment.diagnostics_z_consistency_csvs)
        + len(experiment.diagnostics_z_monotonicity_csvs)
        + len(experiment.diagnostics_path_independence_csvs)
    )


def _load_experiment_diagnostics_h(experiment: Experiment, path: Path) -> None:
    experiment.diagnostics_h_steps = _merge_steps(
        _collect_steps_from_path_list(
            list((path / "vis_delta_h_pca").glob("delta_h_pca_*.png")),
            prefix="delta_h_pca_",
        ),
        _collect_steps_from_path_list(
            list((path / "vis_action_alignment_h").glob("action_alignment_detail_*.png")),
            prefix="action_alignment_detail_",
        ),
        _collect_steps_from_path_list(
            list((path / "vis_cycle_error_h").glob("cycle_error_*.png")),
            prefix="cycle_error_",
        ),
        _collect_steps_from_path_list(
            list((path / "vis_self_distance_h").glob("self_distance_h_*.png")),
            prefix="self_distance_h_",
        ),
    )
    experiment.has_diagnostics_h_steps = bool(experiment.diagnostics_h_steps)


def _load_experiment_diagnostics_p(
    experiment: Experiment,
    path: Path,
    *,
    include_diagnostics_p: bool,
) -> None:
    if include_diagnostics_p:
        experiment.diagnostics_p_delta_p_pca_images = _collect_from_spec(path, DIAGNOSTICS_P_DELTA_P_PCA_IMAGE_SPEC)
        experiment.diagnostics_p_variance_spectrum_images = _collect_from_spec(path, DIAGNOSTICS_P_VARIANCE_SPECTRUM_IMAGE_SPEC)
        experiment.diagnostics_p_action_alignment_detail_images = _collect_from_spec(
            path, DIAGNOSTICS_P_ACTION_ALIGNMENT_DETAIL_IMAGE_SPEC
        )
        experiment.diagnostics_p_cycle_error_images = _collect_from_spec(path, DIAGNOSTICS_P_CYCLE_ERROR_IMAGE_SPEC)
        experiment.diagnostics_p_straightline_images = _collect_from_spec(path, DIAGNOSTICS_P_STRAIGHTLINE_IMAGE_SPEC)
        experiment.diagnostics_p_rollout_divergence_images = _collect_from_spec(
            path, DIAGNOSTICS_P_ROLLOUT_DIVERGENCE_IMAGE_SPEC
        )
        experiment.diagnostics_p_steps = _merge_steps(
            _collect_steps_from_path_list(experiment.diagnostics_p_delta_p_pca_images),
            _collect_steps_from_path_list(experiment.diagnostics_p_variance_spectrum_images),
            _collect_steps_from_path_list(experiment.diagnostics_p_action_alignment_detail_images),
            _collect_steps_from_path_list(experiment.diagnostics_p_cycle_error_images),
            _collect_steps_from_path_list(experiment.diagnostics_p_straightline_images),
            _collect_steps_from_path_list(experiment.diagnostics_p_rollout_divergence_images),
        )
        experiment.has_diagnostics_p_steps = bool(experiment.diagnostics_p_steps)
        experiment.diagnostics_p_delta_p_pca_csvs = _collect_csvs_by_folders(path, DIAGNOSTICS_P_DELTA_P_PCA_CSV_FOLDERS)
        experiment.diagnostics_p_action_alignment_csvs = _collect_csvs_by_folders(
            path, DIAGNOSTICS_P_ACTION_ALIGNMENT_CSV_FOLDERS
        )
        experiment.diagnostics_p_cycle_error_csvs = _collect_csvs_by_folders(path, DIAGNOSTICS_P_CYCLE_ERROR_CSV_FOLDERS)
        experiment.diagnostics_p_rollout_divergence_csvs = _collect_csvs_by_folders(
            path, DIAGNOSTICS_P_ROLLOUT_DIVERGENCE_CSV_FOLDERS
        )
    else:
        experiment.diagnostics_p_steps = [0] if _diagnostics_suffix_exists(path, DIAGNOSTICS_P_DIRS) else []
        experiment.has_diagnostics_p_steps = bool(experiment.diagnostics_p_steps)


def _load_experiment_vis_ctrl(
    experiment: Experiment,
    path: Path,
    *,
    include_vis_ctrl: bool,
) -> None:
    section_start = time.perf_counter()
    if include_vis_ctrl:
        experiment.vis_ctrl_smoothness_z_images = _collect_from_spec(path, VIS_CTRL_SMOOTHNESS_Z_SPEC)
        experiment.vis_ctrl_smoothness_p_images = _collect_from_spec(path, VIS_CTRL_SMOOTHNESS_P_SPEC)
        experiment.vis_ctrl_smoothness_h_images = _collect_from_spec(path, VIS_CTRL_SMOOTHNESS_H_SPEC)
        experiment.vis_ctrl_composition_z_images = _collect_from_spec(path, VIS_CTRL_COMPOSITION_Z_SPEC)
        experiment.vis_ctrl_composition_p_images = _collect_from_spec(path, VIS_CTRL_COMPOSITION_P_SPEC)
        experiment.vis_ctrl_composition_h_images = _collect_from_spec(path, VIS_CTRL_COMPOSITION_H_SPEC)
        experiment.vis_ctrl_stability_z_images = _collect_from_spec(path, VIS_CTRL_STABILITY_Z_SPEC)
        experiment.vis_ctrl_stability_p_images = _collect_from_spec(path, VIS_CTRL_STABILITY_P_SPEC)
        experiment.vis_ctrl_stability_h_images = _collect_from_spec(path, VIS_CTRL_STABILITY_H_SPEC)
        experiment.vis_ctrl_alignment_z_images = _collect_from_spec(path, VIS_CTRL_ALIGNMENT_Z_SPEC)
        experiment.vis_ctrl_alignment_p_images = _collect_from_spec(path, VIS_CTRL_ALIGNMENT_P_SPEC)
        experiment.vis_ctrl_alignment_h_images = _collect_from_spec(path, VIS_CTRL_ALIGNMENT_H_SPEC)
        experiment.vis_ctrl_steps = _merge_steps(
            _collect_steps_from_path_list(experiment.vis_ctrl_smoothness_z_images),
            _collect_steps_from_path_list(experiment.vis_ctrl_smoothness_p_images),
            _collect_steps_from_path_list(experiment.vis_ctrl_smoothness_h_images),
            _collect_steps_from_path_list(experiment.vis_ctrl_composition_z_images),
            _collect_steps_from_path_list(experiment.vis_ctrl_composition_p_images),
            _collect_steps_from_path_list(experiment.vis_ctrl_composition_h_images),
            _collect_steps_from_path_list(experiment.vis_ctrl_stability_z_images),
            _collect_steps_from_path_list(experiment.vis_ctrl_stability_p_images),
            _collect_steps_from_path_list(experiment.vis_ctrl_stability_h_images),
            _collect_steps_from_path_list(experiment.vis_ctrl_alignment_z_images),
            _collect_steps_from_path_list(experiment.vis_ctrl_alignment_p_images),
            _collect_steps_from_path_list(experiment.vis_ctrl_alignment_h_images),
        )
        experiment.vis_ctrl_csvs = _collect_from_spec(path, VIS_CTRL_METRICS_CSV_SPEC, collect_all=False) + _collect_from_spec(
            path, VIS_CTRL_VIS_CTRL_CSV_SPEC, collect_all=False
        )
    else:
        experiment.vis_ctrl_steps = (
            [0]
            if (
                _any_folder_pattern_exists(
                    path,
                    [
                        ("vis_vis_ctrl", "smoothness_z_0000000.png", "smoothness_z_*.png"),
                    ],
                )
                or _any_existing_paths(path, ["metrics/vis_ctrl_metrics.csv"])
            )
            else []
        )
    _profile(
        "load_experiment.vis_ctrl",
        section_start,
        path,
        steps=len(experiment.vis_ctrl_steps),
        csvs=len(experiment.vis_ctrl_csvs),
        included=include_vis_ctrl,
    )


def _load_experiment_self_distance_flags(experiment: Experiment, path: Path) -> None:
    experiment.has_self_distance_z_csv = experiment.self_distance_z_csv is not None
    experiment.has_self_distance_p_csv = experiment.self_distance_p_csv is not None
    _load_experiment_self_distance_h(experiment, path)


def _load_experiment_self_distance_h(experiment: Experiment, path: Path) -> None:
    csvs = _collect_from_spec(path, SELF_DISTANCE_H_CSVS_SPEC, collect_all=False)
    experiment.self_distance_h_csvs = csvs
    experiment.self_distance_h_csv = csvs[-1] if csvs else None
    experiment.has_self_distance_h_csv = experiment.self_distance_h_csv is not None
    experiment.self_distance_h_images = _collect_from_spec(path, SELF_DISTANCE_H_IMAGES_SPEC)


def _load_experiment_metadata(
    experiment: Experiment,
    path: Path,
    *,
    metadata_path: Path,
    metadata_git_path: Path,
    metadata_model_diff_path: Path,
    include_model_diff: bool,
    include_model_diff_generation: bool,
) -> None:
    section_start = time.perf_counter()
    notes_path = path / "notes.txt"
    metadata_custom_path = path / "experiment_metadata.txt"
    meta_start = time.perf_counter()
    title, tags, starred, archived = _read_metadata(metadata_custom_path)
    metadata_exists = metadata_path.exists()
    metadata_text = metadata_path.read_text() if metadata_exists else "metadata.txt missing."
    metadata_data_root = _extract_data_root_from_metadata(metadata_text) if metadata_exists else None
    _profile("load_experiment.text_meta.base", meta_start, path)
    meta_start = time.perf_counter()
    metadata_model_diff_text_raw = "—"
    metadata_model_diff_items: List[Tuple[str, str, bool]] = []
    metadata_model_diff_text = "—"
    if include_model_diff:
        if metadata_model_diff_path.exists():
            metadata_model_diff_text_raw = metadata_model_diff_path.read_text()
        elif include_model_diff_generation:
            metadata_model_diff_text_raw = _ensure_model_diff(path)
        else:
            metadata_model_diff_text_raw = "model_diff.txt missing."
        metadata_model_diff_items = _parse_model_diff_items(metadata_model_diff_text_raw)
        metadata_model_diff_text = _render_model_diff(metadata_model_diff_text_raw)
    _profile("load_experiment.text_meta.model_diff", meta_start, path)
    meta_start = time.perf_counter()
    git_metadata_text = metadata_git_path.read_text() if metadata_git_path.exists() else "metadata_git.txt missing."
    git_commit = _extract_git_commit(git_metadata_text)
    notes_text = _read_or_create_notes(notes_path)
    _profile("load_experiment.text_meta.notes", meta_start, path)
    _profile("load_experiment.text_meta", section_start, path)
    experiment.title = title
    experiment.tags = tags
    experiment.starred = starred
    experiment.archived = archived
    experiment.metadata_text = metadata_text
    experiment.data_root = metadata_data_root
    experiment.model_diff_text = metadata_model_diff_text
    experiment.model_diff_items = metadata_model_diff_items
    experiment.has_model_diff = bool(metadata_model_diff_items)
    experiment.git_metadata_text = git_metadata_text
    experiment.git_commit = git_commit or "Unknown commit"
    experiment.notes_text = notes_text


def _load_experiment_metrics(
    experiment: Experiment,
    path: Path,
    *,
    loss_png: Path,
    loss_csv: Optional[Path],
    include_rollout_steps: bool,
    include_max_step: bool,
    include_last_modified: bool,
) -> None:
    section_start = time.perf_counter()
    experiment.loss_image = loss_png if loss_png.exists() else None
    experiment.loss_csv = loss_csv if loss_csv and loss_csv.exists() else None
    experiment.rollout_steps = (
        _collect_steps_from_path_list(list(path.glob("vis_fixed_*/rollout_*.png")), prefix="rollout_")
        if include_rollout_steps
        else []
    )
    experiment.max_step = (
        get_max_step(experiment.loss_csv)
        if include_max_step and experiment.loss_csv and experiment.loss_csv.exists()
        else None
    )
    experiment.last_modified = (
        _get_last_modified(path, profile=_profile) if include_last_modified else _quick_last_modified(path)
    )
    experiment.total_params, experiment.flops_per_step = _read_model_metadata(path / "metadata_model.txt")
    _profile(
        "load_experiment.metrics",
        section_start,
        path,
        rollout_steps=len(experiment.rollout_steps),
        max_step=experiment.max_step if experiment.max_step is not None else "",
        loss_csv=bool(experiment.loss_csv and experiment.loss_csv.exists()),
        last_modified_mode="deep" if include_last_modified else "quick",
    )


def load_experiment(
    path: Path,
    include_self_distance: bool = False,
    include_self_distance_p: bool = False,
    include_diagnostics_images: bool = False,
    include_diagnostics_frames: bool = False,
    include_diagnostics_p: bool = False,
    include_graph_diagnostics: bool = False,
    include_vis_ctrl: bool = False,
    include_state_embedding: bool = False,
    include_odometry: bool = False,
    include_graph_diagnostics_p: bool = False,
    include_graph_diagnostics_h: bool = False,
    include_last_modified: bool = False,
    include_rollout_steps: bool = False,
    include_max_step: bool = False,
    include_model_diff: bool = False,
    include_model_diff_generation: bool = False,
) -> Optional[Experiment]:
    if not path.is_dir():
        return None
    start_time = time.perf_counter()
    section_start = start_time
    metadata_path = path / "metadata.txt"
    metadata_git_path = path / "metadata_git.txt"
    metadata_model_diff_path = path / "server_cache" / "model_diff.txt"
    metrics_dir = path / "metrics"
    loss_png = metrics_dir / "loss_curves.png"
    loss_csv = (
        _resolve_first_existing_folder(metrics_dir, ["loss_curves.csv", "loss.csv"])
        if metrics_dir.exists()
        else None
    )
    _profile("load_experiment.paths", start_time, path, metrics_dir=metrics_dir)
    section_start = time.perf_counter()

    experiment = Experiment(
        id=path.name,
        name=path.name,
        path=path,
        metadata_text="",
        data_root=None,
        model_diff_text="—",
        model_diff_items=[],
        git_metadata_text="",
        git_commit="Unknown commit",
        notes_text="",
        title="",
        tags="",
        starred=False,
        archived=False,
        loss_image=None,
        loss_csv=None,
        rollout_steps=[],
        max_step=None,
        last_modified=None,
        total_params=None,
        flops_per_step=None,
        self_distance_z_csv=None,
        self_distance_z_images=[],
        self_distance_z_csvs=[],
        self_distance_h_csv=None,
        self_distance_h_images=[],
        self_distance_h_csvs=[],
        self_distance_p_csv=None,
        self_distance_p_images=[],
        self_distance_p_csvs=[],
        state_embedding_csv=None,
        state_embedding_images=[],
        state_embedding_hist_images=[],
        state_embedding_csvs=[],
        odometry_images=[],
        has_odometry_images=False,
        diagnostics_delta_z_pca_images=[],
        diagnostics_variance_spectrum_images=[],
        diagnostics_action_alignment_detail_images=[],
        diagnostics_cycle_error_images=[],
        diagnostics_rollout_divergence_images=[],
        diagnostics_rollout_divergence_z_images=[],
        diagnostics_z_consistency_images=[],
        diagnostics_z_monotonicity_images=[],
        diagnostics_path_independence_images=[],
        diagnostics_z_steps=[],
        has_diagnostics_z_steps=False,
        diagnostics_delta_z_pca_csvs=[],
        diagnostics_action_alignment_csvs=[],
        diagnostics_cycle_error_csvs=[],
        diagnostics_frame_alignment_csvs=[],
        diagnostics_rollout_divergence_csvs=[],
        diagnostics_rollout_divergence_z_csvs=[],
        diagnostics_z_consistency_csvs=[],
        diagnostics_z_monotonicity_csvs=[],
        diagnostics_path_independence_csvs=[],
        diagnostics_frame_steps=[],
        diagnostics_frame_entries=[],
        diagnostics_p_delta_p_pca_images=[],
        diagnostics_p_variance_spectrum_images=[],
        diagnostics_p_action_alignment_detail_images=[],
        diagnostics_p_cycle_error_images=[],
        diagnostics_p_straightline_images=[],
        diagnostics_p_rollout_divergence_images=[],
        diagnostics_p_steps=[],
        diagnostics_h_steps=[],
        has_diagnostics_p_steps=False,
        has_diagnostics_h_steps=False,
        diagnostics_p_delta_p_pca_csvs=[],
        diagnostics_p_action_alignment_csvs=[],
        diagnostics_p_cycle_error_csvs=[],
        diagnostics_p_rollout_divergence_csvs=[],
        graph_diagnostics_rank1_cdf_z_images=[],
        graph_diagnostics_rank1_cdf_h_images=[],
        graph_diagnostics_rank1_cdf_p_images=[],
        graph_diagnostics_rank2_cdf_z_images=[],
        graph_diagnostics_rank2_cdf_h_images=[],
        graph_diagnostics_rank2_cdf_p_images=[],
        graph_diagnostics_neff_violin_z_images=[],
        graph_diagnostics_neff_violin_h_images=[],
        graph_diagnostics_neff_violin_p_images=[],
        graph_diagnostics_in_degree_hist_z_images=[],
        graph_diagnostics_in_degree_hist_h_images=[],
        graph_diagnostics_in_degree_hist_p_images=[],
        graph_diagnostics_edge_consistency_z_images=[],
        graph_diagnostics_edge_consistency_h_images=[],
        graph_diagnostics_edge_consistency_p_images=[],
        graph_diagnostics_metrics_history_z_images=[],
        graph_diagnostics_metrics_history_h_images=[],
        graph_diagnostics_metrics_history_p_images=[],
        graph_diagnostics_metrics_history_latest_z_images=[],
        graph_diagnostics_metrics_history_latest_h_images=[],
        graph_diagnostics_metrics_history_latest_p_images=[],
        graph_diagnostics_z_steps=[],
        graph_diagnostics_metrics_history_z_csvs=[],
        graph_diagnostics_h_steps=[],
        graph_diagnostics_metrics_history_h_csvs=[],
        graph_diagnostics_p_steps=[],
        graph_diagnostics_metrics_history_p_csvs=[],
        vis_ctrl_smoothness_z_images=[],
        vis_ctrl_smoothness_p_images=[],
        vis_ctrl_smoothness_h_images=[],
        vis_ctrl_composition_z_images=[],
        vis_ctrl_composition_p_images=[],
        vis_ctrl_composition_h_images=[],
        vis_ctrl_stability_z_images=[],
        vis_ctrl_stability_p_images=[],
        vis_ctrl_stability_h_images=[],
        vis_ctrl_alignment_z_images=[],
        vis_ctrl_alignment_p_images=[],
        vis_ctrl_alignment_h_images=[],
        vis_ctrl_steps=[],
        vis_ctrl_csvs=[],
        has_self_distance_z_csv=False,
        has_self_distance_h_csv=False,
        has_self_distance_p_csv=False,
        has_model_diff=False,
    )
    _load_experiment_metadata(
        experiment,
        path,
        metadata_path=metadata_path,
        metadata_git_path=metadata_git_path,
        metadata_model_diff_path=metadata_model_diff_path,
        include_model_diff=include_model_diff,
        include_model_diff_generation=include_model_diff_generation,
    )
    _load_experiment_metrics(
        experiment,
        path,
        loss_png=loss_png,
        loss_csv=loss_csv,
        include_rollout_steps=include_rollout_steps,
        include_max_step=include_max_step,
        include_last_modified=include_last_modified,
    )
    _load_experiment_self_distance(experiment, path, include_self_distance=include_self_distance)
    _load_experiment_self_distance_p(experiment, path, include_self_distance_p=include_self_distance_p)
    _load_experiment_state_embedding(experiment, path, include_state_embedding=include_state_embedding)
    _load_experiment_odometry(experiment, path, include_odometry=include_odometry)
    diagnostics_start = time.perf_counter()
    diagnostics_csv_count = _load_experiment_diagnostics(
        experiment,
        path,
        include_diagnostics_images=include_diagnostics_images,
        include_diagnostics_frames=include_diagnostics_frames,
    )
    _load_experiment_diagnostics_p(experiment, path, include_diagnostics_p=include_diagnostics_p)
    _load_experiment_graph_diagnostics_z(
        experiment,
        path,
        include_graph_diagnostics=include_graph_diagnostics,
    )
    _load_experiment_graph_diagnostics_h(
        experiment,
        path,
        include_graph_diagnostics_h=include_graph_diagnostics_h,
    )
    _load_experiment_graph_diagnostics_p(
        experiment,
        path,
        include_graph_diagnostics_p=include_graph_diagnostics_p,
    )
    _load_experiment_vis_ctrl(experiment, path, include_vis_ctrl=include_vis_ctrl)
    _load_experiment_self_distance_flags(experiment, path)
    _profile(
        "load_experiment.diagnostics",
        diagnostics_start,
        path,
        images=(
            len(experiment.diagnostics_delta_z_pca_images)
            + len(experiment.diagnostics_variance_spectrum_images)
            + len(experiment.diagnostics_action_alignment_detail_images)
            + len(experiment.diagnostics_cycle_error_images)
            + len(experiment.diagnostics_rollout_divergence_images)
            + len(experiment.diagnostics_rollout_divergence_z_images)
            + len(experiment.diagnostics_z_consistency_images)
            + len(experiment.diagnostics_z_monotonicity_images)
            + len(experiment.diagnostics_path_independence_images)
        ),
        steps=len(experiment.diagnostics_z_steps),
        csvs=diagnostics_csv_count,
        frames=sum(len(entries) for entries in experiment.diagnostics_frame_entries),
        include_frames=include_diagnostics_frames,
        graph_images=(
            len(experiment.graph_diagnostics_rank1_cdf_z_images)
            + len(experiment.graph_diagnostics_rank2_cdf_z_images)
            + len(experiment.graph_diagnostics_neff_violin_z_images)
            + len(experiment.graph_diagnostics_in_degree_hist_z_images)
            + len(experiment.graph_diagnostics_edge_consistency_z_images)
            + len(experiment.graph_diagnostics_metrics_history_z_images)
            + len(experiment.graph_diagnostics_metrics_history_latest_z_images)
        ),
        graph_steps=len(experiment.graph_diagnostics_z_steps),
        graph_csvs=len(experiment.graph_diagnostics_metrics_history_z_csvs),
        graph_h_images=(
            len(experiment.graph_diagnostics_rank1_cdf_h_images)
            + len(experiment.graph_diagnostics_rank2_cdf_h_images)
            + len(experiment.graph_diagnostics_neff_violin_h_images)
            + len(experiment.graph_diagnostics_in_degree_hist_h_images)
            + len(experiment.graph_diagnostics_edge_consistency_h_images)
            + len(experiment.graph_diagnostics_metrics_history_h_images)
            + len(experiment.graph_diagnostics_metrics_history_latest_h_images)
        ),
        graph_h_steps=len(experiment.graph_diagnostics_h_steps),
        graph_h_csvs=len(experiment.graph_diagnostics_metrics_history_h_csvs),
        graph_p_images=(
            len(experiment.graph_diagnostics_rank1_cdf_p_images)
            + len(experiment.graph_diagnostics_rank2_cdf_p_images)
            + len(experiment.graph_diagnostics_neff_violin_p_images)
            + len(experiment.graph_diagnostics_in_degree_hist_p_images)
            + len(experiment.graph_diagnostics_edge_consistency_p_images)
            + len(experiment.graph_diagnostics_metrics_history_p_images)
            + len(experiment.graph_diagnostics_metrics_history_latest_p_images)
        ),
        graph_p_steps=len(experiment.graph_diagnostics_p_steps),
        graph_p_csvs=len(experiment.graph_diagnostics_metrics_history_p_csvs),
        include_graph=include_graph_diagnostics,
        include_images=include_diagnostics_images,
    )
    _profile(
        "load_experiment",
        start_time,
        path,
        diagnostics_images=(
            len(experiment.diagnostics_delta_z_pca_images)
            + len(experiment.diagnostics_variance_spectrum_images)
            + len(experiment.diagnostics_action_alignment_detail_images)
            + len(experiment.diagnostics_cycle_error_images)
            + len(experiment.diagnostics_rollout_divergence_images)
            + len(experiment.diagnostics_rollout_divergence_z_images)
            + len(experiment.diagnostics_z_consistency_images)
            + len(experiment.diagnostics_z_monotonicity_images)
            + len(experiment.diagnostics_path_independence_images)
        ),
        diagnostics_frames=sum(len(entries) for entries in experiment.diagnostics_frame_entries),
        diagnostics_csvs=diagnostics_csv_count,
        include_frames=include_diagnostics_frames,
    )
    return experiment


def load_loss_curves(csv_path: Path) -> Optional[LossCurveData]:
    if not csv_path.exists():
        return None
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "step" not in reader.fieldnames:
            return None
        # Exclude step, cumulative_flops, and elapsed_seconds from series
        excluded_fields = {"step", "cumulative_flops", "elapsed_seconds"}
        other_fields = [f for f in reader.fieldnames if f not in excluded_fields]
        has_cumulative_flops = "cumulative_flops" in reader.fieldnames
        has_elapsed_seconds = "elapsed_seconds" in reader.fieldnames
        rows = list(reader)
    if not rows:
        return None
    steps: List[float] = []
    cumulative_flops: List[float] = []
    elapsed_seconds: List[float] = []
    series: Dict[str, List[float]] = {field: [] for field in other_fields}
    for row in rows:
        steps.append(float(row.get("step", 0.0)))
        if has_cumulative_flops:
            cumulative_flops.append(float(row.get("cumulative_flops", "0") or 0.0))
        if has_elapsed_seconds:
            elapsed_seconds.append(float(row.get("elapsed_seconds", "0") or 0.0))
        for field in other_fields:
            try:
                series[field].append(float(row.get(field, "0") or 0.0))
            except ValueError:
                series[field].append(0.0)
    # If no cumulative_flops column, generate placeholder (step-based)
    if not cumulative_flops:
        cumulative_flops = [s for s in steps]
    filtered_series = {
        field: values
        for field, values in series.items()
        if not _is_all_zero(values)
    }
    if not filtered_series:
        return None
    return LossCurveData(
        steps=steps,
        cumulative_flops=cumulative_flops,
        elapsed_seconds=elapsed_seconds,
        series=filtered_series,
    )




def _parse_step_from_stem(stem: str, prefix: str) -> Optional[int]:
    if prefix and stem.startswith(prefix):
        suffix = stem[len(prefix) :]
    else:
        parts = stem.split("_")
        suffix = parts[-1] if parts else stem
    try:
        return int(suffix)
    except ValueError:
        return None


def _collect_steps_from_path_list(paths: Sequence[Path], *, prefix: Optional[str] = None) -> List[int]:
    steps: set[int] = set()
    for path in paths:
        stem = path.stem
        if prefix is not None and not stem.startswith(prefix):
            continue
        step = _parse_step_from_stem(stem, prefix or "")
        if step is None:
            continue
        steps.add(step)
    return sorted(steps)


def _merge_steps(*steps: Sequence[int]) -> List[int]:
    merged: set[int] = set()
    for values in steps:
        merged.update(values)
    return sorted(merged)


def _latest_matching_file(folder: Path, pattern: str) -> Optional[Path]:
    best_name = ""
    best_path: Optional[Path] = None
    try:
        with os.scandir(folder) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                name = entry.name
                if not fnmatch.fnmatch(name, pattern):
                    continue
                if name > best_name:
                    best_name = name
                    best_path = Path(entry.path)
    except OSError:
        return None
    return best_path


def _collect_from_spec(root: Path, spec: VisSpec, *, collect_all: bool = True) -> List[Path]:
    for folder_name, pattern, *_ in spec.candidates:
        folder = root / folder_name
        if not folder.exists():
            continue
        if collect_all:
            if _first_matching_file(folder, exact_name=None, pattern=pattern) is None:
                continue
            files = sorted(folder.glob(pattern))
            if files:
                return files
        else:
            match = _latest_matching_file(folder, pattern)
            if match is not None:
                return [match]
    return []


def _collect_named_paths_in_first_existing_folder(
    root: Path,
    candidates: Sequence[str],
    specs: Sequence[Tuple[str, str]],
    *,
    latest: Optional[Tuple[str, str]] = None,
) -> Dict[str, List[Path]]:
    """Collect named file lists from the first existing folder; optionally append a single latest file.

    Example (graph diagnostics):
    - root/
      - graph_diagnostics_z/
        - metrics_history_0000100.png
        - metrics_history_0000200.png
        - metrics_history_latest.png
    Usage:
    _collect_named_paths_in_first_existing_folder(
        root,
        ["graph_diagnostics_z", "graph_diagnostics"],
        [("metrics_history", "metrics_history_*.png")],
        latest=("metrics_history_latest", "metrics_history_latest.png"),
    )
    """
    folder = _resolve_first_existing_folder(root, candidates)
    if folder is None:
        return {}
    paths: Dict[str, List[Path]] = {}
    for name, pattern in specs:
        files = sorted(folder.glob(pattern))
        if files:
            paths[name] = files
    if latest:
        latest_name, latest_filename = latest
        latest_path = folder / latest_filename
        if latest_path.exists():
            paths.setdefault(latest_name, []).append(latest_path)
    return paths


def _diagnostics_suffix_exists(root: Path, folder_names: Sequence[str]) -> bool:
    return any(
        _folder_has_any_file(root / folder_name, DIAGNOSTICS_SUFFIXES)
        for folder_name in folder_names
    )


def _resolve_first_existing_folder(root: Path, candidates: Sequence[str]) -> Optional[Path]:
    for folder_name in candidates:
        folder = root / folder_name
        if folder.exists():
            return folder
    return None


def _any_existing_paths(root: Path, relative_paths: Sequence[str]) -> bool:
    return any((root / rel_path).exists() for rel_path in relative_paths)


def _any_folder_pattern_exists(
    root: Path,
    checks: Sequence[Tuple[str, Optional[str], str]],
) -> bool:
    for folder_name, exact_name, pattern in checks:
        folder = root / folder_name
        if _first_matching_file(folder, exact_name=exact_name, pattern=pattern) is not None:
            return True
    return False


def _collect_csvs_by_folders(
    root: Path,
    spec: Sequence[str],
) -> List[Path]:
    if not spec:
        return []
    for folder_name in spec:
        folder = root / folder_name
        if not folder.exists():
            continue
        files: List[Path] = []
        files.extend(sorted(folder.glob("*.csv")))
        files.extend(sorted(folder.glob("*.txt")))
        if files:
            return files
    return []


def _collect_visualization_steps(root: Path) -> Dict[str, List[int]]:
    """Gather per-visualization step lists used for comparison previews."""
    step_map: Dict[str, List[int]] = {}
    for key, spec in VIS_STEP_SPECS.items():
        for candidate in spec.candidates:
            if len(candidate) == 3:
                folder_name, pattern, candidate_prefix = candidate
            else:
                folder_name, pattern = candidate
                candidate_prefix = ""
            prefix = spec.prefix if spec.prefix is not None else candidate_prefix
            folder = root / folder_name
            if not folder.exists():
                continue
            try:
                steps = _collect_steps_from_path_list(list(folder.glob(pattern)), prefix=prefix)
            except OSError:
                steps = []
            if steps:
                step_map[key] = steps
                break
    return step_map


def _collect_state_embedding_distance_images(root: Path) -> List[Path]:
    distance_files = _collect_from_spec(root, STATE_EMBEDDING_DISTANCE_IMAGE_SPEC)
    cosine_files = _collect_from_spec(root, STATE_EMBEDDING_COSINE_IMAGE_SPEC)
    if cosine_files:
        cosine_folder = cosine_files[0].parent.name
        filtered_distance = [path for path in distance_files if path.parent.name == cosine_folder]
        return filtered_distance or distance_files
    return distance_files


def _collect_diagnostics_frames(root: Path) -> Dict[int, List[Tuple[Path, str, str, Optional[int]]]]:
    frames_root = root / "vis_diagnostics_frames"
    if not frames_root.exists():
        return {}
    start_time = time.perf_counter()
    frame_map: Dict[int, List[Tuple[Path, str, str, Optional[int]]]] = {}
    csv_count = 0
    frame_count = 0

    def _natural_sort_key(path_str: str) -> Tuple:
        """Split a path into text/int chunks so state_2 comes before state_10."""
        parts = re.split(r"(\d+)", path_str)
        key: List = []
        for part in parts:
            if not part:
                continue
            if part.isdigit():
                key.append(int(part))
            else:
                key.append(part.lower())
        return tuple(key)

    for csv_path in sorted(frames_root.glob("frames_*.csv")):
        stem = csv_path.stem
        suffix = stem.split("_")[-1]
        try:
            step = int(suffix)
        except ValueError:
            continue
        try:
            with csv_path.open("r", newline="") as handle:
                reader = csv.DictReader(handle)
                sorted_entries: List[Tuple[Tuple, Path, str, str, Optional[int]]] = []
                csv_count += 1
                for idx, row in enumerate(reader):
                    rel = row.get("image_path")
                    src = (row.get("source_path") or "").strip()
                    action_label = (row.get("action_label") or "").strip()
                    aid_raw = row.get("action_id")
                    try:
                        action_id: Optional[int] = int(aid_raw) if aid_raw not in (None, "", "None") else None
                    except (TypeError, ValueError):
                        action_id = None
                    if not rel:
                        continue
                    sort_basis = src if src else rel
                    sort_key = (_natural_sort_key(sort_basis), idx)
                    sorted_entries.append((sort_key, frames_root / rel, src, action_label, action_id))
                if sorted_entries:
                    sorted_entries.sort(key=lambda t: t[0])
                    frame_count += len(sorted_entries)
                    frame_map[step] = [(p, src, action_label, action_id) for _, p, src, action_label, action_id in sorted_entries]
        except (OSError, csv.Error):
            continue
    _profile("diagnostics.frames", start_time, root, csv_files=csv_count, frames=frame_count)
    return frame_map


def _folder_has_any_file(folder: Path, suffixes: Tuple[str, ...]) -> bool:
    """Return True if folder contains any file with a matching suffix."""
    try:
        with os.scandir(folder) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                if entry.name.lower().endswith(suffixes):
                    return True
    except OSError:
        return False
    return False


def _is_all_zero(values: Iterable[float]) -> bool:
    return all(abs(v) <= ALL_ZERO_EPS for v in values)
