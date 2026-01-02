"""Action-alignment strength helper for diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from jepa_world_model.actions import decode_action_id


def write_action_alignment_strength(
    alignment_stats: List[Dict[str, Any]],
    motion: Dict[str, Any],
    out_dir: Path,
    global_step: int,
) -> None:
    """Summarize per-action directional strength relative to step magnitude."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"action_alignment_strength_{global_step:07d}.txt"
    action_dim = motion["action_dim"]
    with path.open("w") as handle:
        if not alignment_stats:
            handle.write("No actions met alignment criteria.\n")
            return
        handle.write(
            "Per-action directional strength (mean_dir_norm / delta_norm_median)\n"
            "Lower ratios imply the average direction is weak relative to per-step magnitude (possible aliasing/sign flips).\n\n"
        )
        handle.write(
            "action_id\tlabel\tcount\tmean_cos\tstd\tfrac_neg\tmean_dir_norm\t"
            "delta_norm_median\tstrength_ratio\tnote\n"
        )
        for stat in alignment_stats:
            delta_med = float(stat.get("delta_norm_median", float("nan")))
            mean_norm = float(stat.get("mean_dir_norm", float("nan")))
            strength = float("nan")
            if np.isfinite(delta_med) and delta_med > 0 and np.isfinite(mean_norm):
                strength = mean_norm / delta_med
            note = ""
            if not np.isfinite(strength) or strength < 0.05:
                note = "weak mean vs magnitude"
            elif strength < 0.15:
                note = "moderate mean vs magnitude"
            handle.write(
                f"{stat.get('action_id')}\t{decode_action_id(stat.get('action_id', -1), action_dim)}"
                f"\t{stat.get('count', 0)}\t{stat.get('mean', float('nan')):.4f}"
                f"\t{stat.get('std', float('nan')):.4f}\t{stat.get('frac_neg', float('nan')):.3f}"
                f"\t{mean_norm:.4f}\t{delta_med:.4f}\t{strength:.4f}\t{note}\n"
            )
