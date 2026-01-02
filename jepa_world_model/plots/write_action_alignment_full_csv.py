"""Action alignment full CSV writer."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from jepa_world_model.actions import decode_action_id


def write_action_alignment_full_csv(
    alignment_dir: Path,
    global_step: int,
    motion: Dict[str, Any],
    alignment_stats: List[Dict[str, Any]],
) -> None:
    alignment_dir.mkdir(parents=True, exist_ok=True)
    align_full_csv = alignment_dir / f"action_alignment_full_{global_step:07d}.csv"
    with align_full_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "action_id",
                "action_label",
                "count",
                "mean_cos",
                "median_cos",
                "std_cos",
                "pct_high",
                "frac_negative",
                "mean_dir_norm",
                "delta_norm_mean",
                "delta_norm_median",
                "delta_norm_p10",
                "delta_norm_p90",
                "frac_low_delta_norm",
            ]
        )
        for stat in alignment_stats:
            writer.writerow(
                [
                    stat.get("action_id"),
                    decode_action_id(stat.get("action_id", -1), motion["action_dim"]),
                    stat.get("count", 0),
                    stat.get("mean", float("nan")),
                    stat.get("median", float("nan")),
                    stat.get("std", float("nan")),
                    stat.get("pct_high", float("nan")),
                    stat.get("frac_neg", float("nan")),
                    stat.get("mean_dir_norm", float("nan")),
                    stat.get("delta_norm_mean", float("nan")),
                    stat.get("delta_norm_median", float("nan")),
                    stat.get("delta_norm_p10", float("nan")),
                    stat.get("delta_norm_p90", float("nan")),
                    stat.get("frac_low_delta_norm", float("nan")),
                ]
            )
