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
    # PCA components used for position action vector field projections.
    position_vector_pca_components: int = 8
