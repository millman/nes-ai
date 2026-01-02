"""Action-alignment report helper for diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from jepa_world_model.actions import decode_action_id


def write_action_alignment_report(
    alignment_stats: List[Dict[str, Any]],
    action_dim: int,
    inverse_map: Dict[int, int],
    out_dir: Path,
    global_step: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"action_alignment_report_{global_step:07d}.txt"
    with report_path.open("w") as handle:
        handle.write("Action alignment diagnostics (per action)\n")
        if not alignment_stats:
            handle.write("No actions met alignment criteria.\n")
            return
        handle.write(
            "action_id\tlabel\tcount\tmean\tmedian\tstd\tfrac_neg\tpct_gt_thr\tv_norm"
            "\tdelta_norm_median\tdelta_norm_p90\tfrac_low_norm\tinverse_alignment\tnotes\n"
        )
        mean_vecs: Dict[int, np.ndarray] = {}
        for stat in alignment_stats:
            if "mean_dir" in stat:
                mean_vecs[int(stat["action_id"])] = stat["mean_dir"]
        for stat in alignment_stats:
            aid = int(stat["action_id"])
            label = decode_action_id(aid, action_dim)
            inv_align = ""
            inv_id = inverse_map.get(aid)
            if inv_id is not None:
                inv_vec = mean_vecs.get(inv_id)
                this_vec = mean_vecs.get(aid)
                if inv_vec is not None and this_vec is not None:
                    inv_norm = float(np.linalg.norm(inv_vec))
                    this_norm = float(np.linalg.norm(this_vec))
                    if inv_norm > 1e-8 and this_norm > 1e-8:
                        inv_align = float(np.dot(this_vec, -inv_vec) / (inv_norm * this_norm))
            note = ""
            if stat.get("mean", 0.0) < 0:
                if stat.get("mean_dir_norm", 0.0) < 1e-6 or stat.get("delta_norm_p90", 0.0) < 1e-6:
                    note = "degenerate mean/blocked"
                elif stat.get("frac_neg", 0.0) > 0.4:
                    note = "bimodal/aliasing suspected"
                else:
                    note = "check action mapping/PCA"
            elif stat.get("mean_dir_norm", 0.0) < 1e-6:
                note = "mean direction near zero"
            handle.write(
                f"{aid}\t{label}\t{stat.get('count', 0)}\t{stat.get('mean', float('nan')):.4f}"
                f"\t{stat.get('median', float('nan')):.4f}\t{stat.get('std', float('nan')):.4f}"
                f"\t{stat.get('frac_neg', float('nan')):.3f}\t{stat.get('pct_high', float('nan')):.3f}"
                f"\t{stat.get('mean_dir_norm', float('nan')):.4f}"
                f"\t{stat.get('delta_norm_median', float('nan')):.4f}\t{stat.get('delta_norm_p90', float('nan')):.4f}"
                f"\t{stat.get('frac_low_delta_norm', float('nan')):.3f}\t{inv_align}\t{note}\n"
            )
