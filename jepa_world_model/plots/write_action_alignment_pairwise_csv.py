"""Action alignment pairwise CSV writer."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from jepa_world_model.actions import decode_action_id


def write_action_alignment_pairwise_csv(
    alignment_dir: Path,
    global_step: int,
    action_dim: int,
    alignment_debug: Dict[str, Any],
) -> None:
    alignment_dir.mkdir(parents=True, exist_ok=True)
    pairwise_csv = alignment_dir / f"action_alignment_pairwise_{global_step:07d}.csv"
    actions_sorted: List[int] = list(alignment_debug.get("actions_sorted") or [])
    pairwise_raw = alignment_debug.get("pairwise")
    pairwise = np.asarray([] if pairwise_raw is None else pairwise_raw, dtype=np.float32)
    with pairwise_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["action_id_a", "label_a", "action_id_b", "label_b", "cosine"])
        for i, aid in enumerate(actions_sorted):
            for j, bid in enumerate(actions_sorted):
                if pairwise.size == 0 or i >= pairwise.shape[0] or j >= pairwise.shape[1]:
                    continue
                writer.writerow(
                    [
                        aid,
                        decode_action_id(aid, action_dim),
                        bid,
                        decode_action_id(bid, action_dim),
                        float(pairwise[i, j]),
                    ]
                )
