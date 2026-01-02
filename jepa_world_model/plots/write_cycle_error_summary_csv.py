"""Cycle error summary CSV writer."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from jepa_world_model.actions import decode_action_id


def write_cycle_error_summary_csv(
    cycle_dir: Path,
    global_step: int,
    action_dim: int,
    cycle_per_action: Dict[int, List[float]],
) -> None:
    cycle_dir.mkdir(parents=True, exist_ok=True)
    cycle_summary_csv = cycle_dir / f"cycle_error_summary_{global_step:07d}.csv"
    with cycle_summary_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["action_id", "action_label", "count", "mean_cycle_error"])
        for aid, vals in sorted(cycle_per_action.items(), key=lambda kv: len(kv[1]), reverse=True):
            if not vals:
                continue
            writer.writerow([aid, decode_action_id(aid, action_dim), len(vals), float(np.mean(vals))])
