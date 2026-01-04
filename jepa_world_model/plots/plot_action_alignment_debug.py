"""Action-alignment debug helper for diagnostics."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def build_action_alignment_debug(
    alignment_stats: List[Dict[str, Any]],
    delta_proj: np.ndarray,
    action_ids: np.ndarray,
) -> Dict[str, Any]:
    """Build auxiliary tensors for debugging alignment drift/degeneracy."""
    mean_units: Dict[int, np.ndarray] = {}
    zero_delta_actions: set[int] = set()
    counts: Dict[int, int] = {}
    for stat in alignment_stats:
        aid = int(stat.get("action_id", -1))
        mean_vec = stat.get("mean_dir")
        norm = stat.get("mean_dir_norm", 0.0) or 0.0
        if mean_vec is not None and norm >= 1e-8:
            mean_units[aid] = np.asarray(mean_vec, dtype=np.float32) / float(norm)
        if stat.get("zero_delta"):
            zero_delta_actions.add(aid)
        counts[aid] = int(stat.get("count", 0))

    per_action_norms: Dict[int, np.ndarray] = {}
    per_action_cos: Dict[int, np.ndarray] = {}
    overall_cos_list: List[np.ndarray] = []
    overall_norm_list: List[np.ndarray] = []

    for aid, mean_unit in mean_units.items():
        mask = action_ids == aid
        vecs = delta_proj[mask]
        if vecs.shape[0] == 0:
            continue
        norms = np.linalg.norm(vecs, axis=1)
        valid = norms >= 1e-8
        if not np.any(valid):
            continue
        vec_unit = vecs[valid] / norms[valid, None]
        cos = vec_unit @ mean_unit
        per_action_norms[aid] = norms[valid]
        per_action_cos[aid] = cos
        overall_cos_list.append(cos)
        overall_norm_list.append(norms[valid])

    for aid in zero_delta_actions:
        if aid in per_action_cos:
            continue
        mask = action_ids == aid
        vecs = delta_proj[mask]
        if vecs.shape[0] == 0:
            continue
        norms = np.linalg.norm(vecs, axis=1)
        per_action_norms[aid] = norms
        cos = np.ones_like(norms, dtype=np.float32)
        per_action_cos[aid] = cos
        overall_cos_list.append(cos)
        overall_norm_list.append(norms)

    overall_cos = np.concatenate(overall_cos_list) if overall_cos_list else np.asarray([], dtype=np.float32)
    overall_norms = np.concatenate(overall_norm_list) if overall_norm_list else np.asarray([], dtype=np.float32)

    actions_sorted = sorted(set(mean_units.keys()) | zero_delta_actions)
    pairwise = np.full((len(actions_sorted), len(actions_sorted)), np.nan, dtype=np.float32)
    for i, ai in enumerate(actions_sorted):
        for j, aj in enumerate(actions_sorted):
            a_vec = mean_units.get(ai)
            b_vec = mean_units.get(aj)
            if a_vec is None or b_vec is None:
                continue
            denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
            if denom < 1e-8:
                continue
            pairwise[i, j] = float(np.dot(a_vec, b_vec) / denom)

    return {
        "actions_sorted": actions_sorted,
        "counts": counts,
        "overall_cos": overall_cos,
        "overall_norms": overall_norms,
        "per_action_norms": per_action_norms,
        "per_action_cos": per_action_cos,
        "pairwise": pairwise,
    }
