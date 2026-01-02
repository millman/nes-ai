"""Cycle error per-sample CSV writer."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

from jepa_world_model.actions import decode_action_id


def write_cycle_error_values_csv(
    cycle_dir: Path,
    global_step: int,
    action_dim: int,
    cycle_errors: List[Tuple[int, float]],
) -> None:
    cycle_dir.mkdir(parents=True, exist_ok=True)
    cycle_values_csv = cycle_dir / f"cycle_error_values_{global_step:07d}.csv"
    with cycle_values_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["action_id", "action_label", "cycle_error"])
        for aid, val in cycle_errors:
            writer.writerow([aid, decode_action_id(aid, action_dim), val])
