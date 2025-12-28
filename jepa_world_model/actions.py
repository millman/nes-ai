#!/usr/bin/env python3
"""Action utilities shared across diagnostics and visualization."""
from __future__ import annotations

from typing import Union

import numpy as np
import torch

from nes_controller import CONTROLLER_STATE_DESC


def decode_action_id(action_id: int, action_dim: int) -> str:
    names = CONTROLLER_STATE_DESC[:action_dim]
    parts = [names[idx] for idx in range(len(names)) if action_id & (1 << idx)]
    return "+".join(parts) if parts else "NOOP"


def compress_actions_to_ids(actions: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Convert multi-hot controller vectors to integer action ids."""
    if isinstance(actions, torch.Tensor):
        if actions.ndim < 2:
            raise ValueError("Expected at least 2D actions tensor for compression.")
        flat = actions.reshape(-1, actions.shape[-1])
        binary = (flat > 0.5).to(dtype=torch.int64)
        weights = (1 << torch.arange(binary.shape[1], dtype=torch.int64, device=binary.device))
        return (binary * weights).sum(dim=1)
    if not isinstance(actions, np.ndarray):
        raise TypeError("actions must be a numpy array or torch tensor.")
    if actions.ndim < 2:
        raise ValueError("Expected at least 2D actions array for compression.")
    flat = actions.reshape(-1, actions.shape[-1])
    binary = (flat > 0.5).astype(np.int64)
    weights = (1 << np.arange(binary.shape[1], dtype=np.int64))
    return (binary * weights).sum(axis=1)
