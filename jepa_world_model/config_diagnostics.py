#!/usr/bin/env python3
"""Motion/action diagnostics visualization and export helpers."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DiagnosticsConfig:
    sample_sequences: int = 128
    top_k_components: int = 64
    min_action_count: int = 5
    max_actions_to_plot: int = 12
    cosine_high_threshold: float = 0.7
    synthesize_cycle_samples: bool = False
    rollout_divergence_horizon: int = 8
    rollout_divergence_samples: int = 64
    straightline_steps: int = 8
    straightline_starts: int = 3
    z_consistency_samples: int = 6
    z_consistency_repeats: int = 8
    z_consistency_noise_std: float = 0.02
    z_monotonicity_max_shift: int = 6
    z_monotonicity_samples: int = 12
    path_independence_steps: int = 4
    path_independence_samples: int = 16
    h_drift_max_actions: int = 12
    zp_distance_pairs: int = 512
    zp_distance_min_gap: int = 2
    position_grid_rows: int = 14
    position_grid_cols: int = 16
    position_agent_color: tuple[int, int, int] = (66, 167, 70)
    position_inventory_height: int | None = None
    position_vector_scale: float = 1.0


@dataclass
class SpikeDiagnosticsConfig:
    enabled: bool = True
    metrics: tuple[str, ...] = (
        "loss_world",
        "loss_recon_multi_box",
        "loss_inverse_dynamics_h",
        "loss_inverse_dynamics_p",
        "loss_inverse_dynamics_dp",
        "grad_world",
    )
    warmup_steps: int = 100
    ema_decay: float = 0.98
    z_threshold: float = 4.0
    ratio_threshold: float = 3.0
    min_reference: float = 1e-3
    max_spikes: int = 50
    save_visuals: bool = True
