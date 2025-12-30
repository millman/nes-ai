#!/usr/bin/env python3
"""Encoder schedule helpers."""
from __future__ import annotations

from typing import List, Tuple


def _derive_encoder_schedule(embedding_dim: int, num_layers: int) -> Tuple[int, ...]:
    """Derive a channel schedule that doubles each layer and ends at embedding_dim."""
    if num_layers < 1:
        raise ValueError("num_downsample_layers must be positive.")
    factor = 2 ** (num_layers - 1)
    if embedding_dim % factor != 0:
        raise ValueError("embedding_dim must be divisible by 2^(num_downsample_layers - 1) for automatic schedule.")
    base_channels = max(1, embedding_dim // factor)
    schedule: List[int] = []
    current = base_channels
    for _ in range(num_layers):
        schedule.append(current)
        current *= 2
    schedule[-1] = embedding_dim
    return tuple(schedule)


def _suggest_encoder_schedule(embedding_dim: int, num_layers: int) -> str:
    """Generate a suggested encoder_schedule for error messages."""
    try:
        suggested = _derive_encoder_schedule(embedding_dim, num_layers)
        return f"encoder_schedule={suggested}"
    except ValueError:
        # If we can't derive, suggest a simple pattern
        return f"encoder_schedule with {num_layers} layers ending in {embedding_dim}"


__all__ = ["_derive_encoder_schedule", "_suggest_encoder_schedule"]
