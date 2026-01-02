"""Action inverse-map helpers for diagnostics."""
from __future__ import annotations

from typing import Dict

import numpy as np

from jepa_world_model.actions import compress_actions_to_ids


def build_action_inverse_map(actions_np: np.ndarray) -> Dict[int, int]:
    """Approximate inverse mapping by swapping up/down and left/right bits."""
    action_dim = actions_np.shape[-1]
    action_ids = compress_actions_to_ids(actions_np.reshape(-1, action_dim))
    observed_ids = np.unique(action_ids)
    weights = (1 << np.arange(action_dim, dtype=np.int64))

    mapping: Dict[int, int] = {}
    for aid in observed_ids:
        bits = np.array([(int(aid) >> idx) & 1 for idx in range(action_dim)], dtype=np.int64)
        if action_dim > 5:
            bits[4], bits[5] = bits[5], bits[4]
        if action_dim > 7:
            bits[6], bits[7] = bits[7], bits[6]
        mapping[int(aid)] = int((bits * weights).sum())
    return mapping
