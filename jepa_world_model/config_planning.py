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
    # H graph merge heuristic: radius = h_merge_multiplier * d_nn.
    h_merge_multiplier: float = 1.5
    # Optional clamp range for h merge radius (None disables clamp).
    h_merge_min: Optional[float] = None
    h_merge_max: Optional[float] = None
    # Distance metric for H planning graph clustering.
    h_distance_metric: Literal["l2", "cosine"] = "l2"
    # H planning node distance mode.
    # - "legacy": use fixed multiplier over empirical non-noop step distance.
    # - "anchor_h": derive radii from empirical 1-step h displacement scale.
    h_distance_mode: Literal["legacy", "anchor_h"] = "anchor_h"
    # Anchor-mode base multipliers (r = c * anchor_h).
    h_anchor_c_merge: float = 1.1
    h_anchor_c_add: float = 1.35
    h_anchor_c_edge: float = 2.0
    h_anchor_c_goal: float = 1.0
    # Anchor-mode safety floor: merge radius >= h_anchor_noise_multiplier * noise_floor.
    h_anchor_noise_multiplier: float = 1.1
    # Controller target range for h graph node count.
    h_anchor_target_nodes_min: int = 16
    h_anchor_target_nodes_max: int = 28
    # Controller step size for c_add adaptation.
    h_anchor_c_add_adjust: float = 0.05
    # Bounds for adaptive c_add.
    h_anchor_c_add_min: float = 1.0
    h_anchor_c_add_max: float = 2.0
    # If true, adapt c_add online using node-count and reachability feedback.
    h_anchor_adapt_c_add: bool = True
    # Reachability threshold used by the c_add controller.
    h_anchor_reachability_target: float = 0.95
    # PCA components used for position action vector field projections.
    position_vector_pca_components: int = 8
