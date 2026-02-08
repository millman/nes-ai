#!/usr/bin/env python3
"""Configuration for planning diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class PlanningDiagnosticsConfig:
    sample_sequences: int = 128
    min_action_count: int = 5
    astar_max_nodes: int = 3000
    reachable_fraction_samples: int = 128
    local_k_min: int = 3
    local_k_max: int = 10
    interior_goal_col_offset: int = 2
    pca_samples: int = 4096
    # Which latent to use for planning diagnostics: "p", "h", or "auto" (use p when available).
    latent_kind: str = "h"
    # Theme for planning env + grid overlay (None uses default theme).
    env_theme: Optional[Literal["basic", "zelda"]] = "basic"
    # Distance metric for H planning graph clustering.
    h_distance_metric: Literal["l2", "cosine"] = "l2"
    # Fixed H-planning radii derived from one non-NOOP move-step scale d_step.
    # r_add = h_move_step_radius_scale * d_step
    # r_merge = h_move_step_merge_scale * d_step
    # r_edge = h_move_step_edge_scale * d_step
    # r_goal = h_move_step_goal_scale * d_step
    h_move_step_radius_scale: float = 1.2
    h_move_step_merge_scale: float = 1.0
    h_move_step_edge_scale: float = 2.0
    h_move_step_goal_scale: float = 1.0
    # Stop A* when candidate lattice expansion exceeds known-grid bounds by this fraction.
    astar_grid_bounds_expand_fraction: float = 0.5
    # PCA components used for position action vector field projections.
    position_vector_pca_components: int = 8
