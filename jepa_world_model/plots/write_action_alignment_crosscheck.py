"""Action-alignment crosscheck helper for diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from jepa_world_model.actions import decode_action_id
from jepa_world_model.plots.build_motion_subspace import MotionSubspace


def write_action_alignment_crosscheck(
    alignment_stats: List[Dict[str, Any]],
    motion: MotionSubspace,
    out_dir: Path,
    global_step: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"action_alignment_crosscheck_{global_step:07d}.txt"
    action_dim = motion.action_dim
    action_ids = motion.action_ids
    delta_proj = motion.delta_proj
    mean_units: Dict[int, np.ndarray] = {}
    for stat in alignment_stats:
        mean_vec = stat.get("mean_dir")
        norm = stat.get("mean_dir_norm", 0.0)
        aid = int(stat["action_id"])
        if mean_vec is None or norm is None or norm < 1e-8:
            continue
        mean_units[aid] = mean_vec / norm
    if not mean_units:
        with path.open("w") as handle:
            handle.write("No usable mean directions for crosscheck.\n")
        return
    with path.open("w") as handle:
        handle.write(
            "Cross-check: sample cosines against own vs other mean directions\n"
            "action_id\tlabel\tcount_valid\tself_mean\tbest_other_id\tbest_other_label"
            "\tbest_other_mean\tgap_self_minus_best_other\tnote\n"
        )
        for aid, mean_unit in mean_units.items():
            mask = action_ids == aid
            vecs = delta_proj[mask]
            if vecs.shape[0] == 0:
                continue
            norms = np.linalg.norm(vecs, axis=1)
            valid_mask = norms >= 1e-8
            if not np.any(valid_mask):
                continue
            vecs_unit = vecs[valid_mask] / norms[valid_mask, None]
            self_mean = float(np.dot(vecs_unit, mean_unit).mean())
            best_other_id: Optional[int] = None
            best_other_mean = -float("inf")
            for bid, other_unit in mean_units.items():
                if bid == aid:
                    continue
                other_mean = float(np.dot(vecs_unit, other_unit).mean())
                if other_mean > best_other_mean:
                    best_other_mean = other_mean
                    best_other_id = bid
            gap = self_mean - best_other_mean if best_other_id is not None else float("nan")
            note = ""
            if best_other_id is not None and gap < 0.05:
                note = "samples align similarly to another action"
            handle.write(
                f"{aid}\t{decode_action_id(aid, action_dim)}\t{vecs_unit.shape[0]}"
                f"\t{self_mean:.4f}\t{best_other_id}"
                f"\t{decode_action_id(best_other_id, action_dim) if best_other_id is not None else ''}"
                f"\t{best_other_mean:.4f}\t{gap:.4f}\t{note}\n"
            )
