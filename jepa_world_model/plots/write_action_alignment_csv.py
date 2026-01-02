"""Action alignment summary CSV writer."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from jepa_world_model.actions import decode_action_id


def write_action_alignment_csv(
    alignment_dir: Path,
    global_step: int,
    action_dim: int,
    alignment_stats: List[Dict[str, Any]],
) -> None:
    alignment_dir.mkdir(parents=True, exist_ok=True)
    align_csv = alignment_dir / f"action_alignment_{global_step:07d}.csv"
    with align_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["action_id", "action_label", "count", "mean_cos", "std_cos", "pct_high"])
        for stat in alignment_stats:
            writer.writerow(
                [
                    stat["action_id"],
                    decode_action_id(stat["action_id"], action_dim),
                    stat["count"],
                    stat["mean"],
                    stat["std"],
                    stat["pct_high"],
                ]
            )
