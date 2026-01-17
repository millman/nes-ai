#!/usr/bin/env python3
"""Configuration for planning diagnostics."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlanningDiagnosticsConfig:
    enabled: bool = True
    sample_sequences: int = 128
    min_action_count: int = 5
    astar_max_nodes: int = 3000
    reachable_fraction_samples: int = 128
    local_k_min: int = 3
    local_k_max: int = 10
    interior_goal_col_offset: int = 2
    pca_samples: int = 4096
