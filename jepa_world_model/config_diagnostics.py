#!/usr/bin/env python3
"""Motion/action diagnostics visualization and export helpers."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DiagnosticsConfig:
    enabled: bool = True
    sample_sequences: int = 128
    top_k_components: int = 4
    min_action_count: int = 5
    max_actions_to_plot: int = 12
    cosine_high_threshold: float = 0.7
    synthesize_cycle_samples: bool = False
