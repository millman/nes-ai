"""Action-alignment stats helper for diagnostics."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

def compute_action_alignment_stats(
    delta_proj: np.ndarray,
    action_ids: np.ndarray,
    min_action_count: int,
    cosine_high_threshold: float,
) -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    unique_actions, counts = np.unique(action_ids, return_counts=True)
    if unique_actions.size == 0:
        return stats
    order = np.argsort(counts)[::-1]
    for idx in order:
        aid = int(unique_actions[idx])
        mask = action_ids == aid
        delta_a = delta_proj[mask]
        if delta_a.shape[0] < min_action_count:
            continue
        mean_dir = delta_a.mean(axis=0)
        norm = np.linalg.norm(mean_dir)
        if norm < 1e-8:
            continue
        v_unit = mean_dir / norm
        cosines: List[float] = []
        for vec in delta_a:
            denom = np.linalg.norm(vec)
            if denom < 1e-8:
                continue
            cosines.append(float(np.dot(vec / denom, v_unit)))
        if not cosines:
            continue
        cos_np = np.array(cosines, dtype=np.float32)
        entry: Dict[str, Any] = {
            "action_id": aid,
            "count": len(cos_np),
            "mean": float(cos_np.mean()),
            "median": float(np.median(cos_np)),
            "std": float(cos_np.std()),
            "pct_high": float((cos_np > cosine_high_threshold).mean()),
            "frac_neg": float((cos_np < 0).mean()),
            "cosines": cos_np,
            "mean_dir_norm": float(norm),
        }
        delta_norms = np.linalg.norm(delta_a, axis=1)
        entry.update(
            {
                "delta_norm_mean": float(delta_norms.mean()),
                "delta_norm_median": float(np.median(delta_norms)),
                "delta_norm_p10": float(np.percentile(delta_norms, 10)),
                "delta_norm_p90": float(np.percentile(delta_norms, 90)),
                "frac_low_delta_norm": float((delta_norms < 1e-8).mean()),
                "mean_dir": mean_dir,
            }
        )
        stats.append(entry)
    return stats
