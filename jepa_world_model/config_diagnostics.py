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


@dataclass
class SpikeDiagnosticsConfig:
    enabled: bool = True
    metrics: tuple[str, ...] = (
        "loss_world",
        "loss_inverse_dynamics_h",
        "loss_inverse_dynamics_s",
        "grad_world",
    )
    warmup_steps: int = 100
    ema_decay: float = 0.98
    z_threshold: float = 4.0
    ratio_threshold: float = 3.0
    min_reference: float = 1e-3
    max_spikes: int = 50
    save_visuals: bool = True
