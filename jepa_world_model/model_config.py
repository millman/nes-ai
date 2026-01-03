#!/usr/bin/env python3
"""Model configuration dataclass for the JEPA world model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class LayerNormConfig:
    """LayerNorm toggles for debugging latent stability."""

    h2z_projector: bool = True
    h2s_projector: bool = True
    delta_projector: bool = True
    action_delta_projector: bool = True
    inverse_dynamics: bool = True
    h_next: bool = False


@dataclass
class ModelConfig:
    """Dimensional flow:

        image (image_size^2·in_channels)
            └─ Encoder(encoder_schedule → pool → z_dim = encoder_schedule[-1])
            ▼
        z_t ────────────────────────────────┐
                        ├─ Predictor([z_t, h_t, action_t]) → ẑ_{t+1}, ĥ_{t+1}, ŝ_{t+1}
        h_t (state_dim) ─────────────────────┘
            │
            ├→ StateHead(h_t) → s_t (planning embedding)
            └→ Decoder(decoder_schedule → image reconstruction from z)

    Notes:
    • image_size must be divisible by 2**len(encoder_schedule) and 2**len(decoder_schedule) (if provided).
    • encoder_schedule[-1] defines embedding_dim (z_dim); state_dim defines h dimensionality.
    • hidden_dim is the predictor's internal width; action_dim is the controller space size.
    • decoder_schedule defaults to encoder_schedule if not set.
    """

    in_channels: int = 3
    image_size: int = 64
    hidden_dim: int = 512
    encoder_schedule: Tuple[int, ...] = (32, 64, 128, 256)
    decoder_schedule: Optional[Tuple[int, ...]] = (32, 64, 64, 128)
    action_dim: int = 8
    predictor_film_layers: int = 2
    state_dim: int = 256
    # Optional separate dimension for the projected planning embedding s (h2s).
    # If None, defaults to state_dim so h and s share dimensionality.
    state_embed_dim: Optional[int] = None
    state_embed_unit_norm: bool = False
    # LayerNorm toggles for debugging.
    layer_norms: LayerNormConfig = field(default_factory=LayerNormConfig)
    state_warmup_frames: int = 4

    @property
    def embedding_dim(self) -> int:
        """The embedding dimension is encoder_schedule[-1]."""
        return self.encoder_schedule[-1]

    def __post_init__(self) -> None:
        if not self.encoder_schedule:
            raise ValueError("encoder_schedule must be non-empty.")

        # Validate image_size is divisible by encoder stride
        num_layers = len(self.encoder_schedule)
        stride = 2 ** num_layers
        if self.image_size % stride != 0:
            raise ValueError(
                f"image_size={self.image_size} must be divisible by 2**len(encoder_schedule)={stride}."
            )

        # Validate decoder_schedule if provided
        if self.decoder_schedule is not None:
            decoder_stride = 2 ** len(self.decoder_schedule)
            if self.image_size % decoder_stride != 0:
                raise ValueError(
                    f"image_size={self.image_size} must be divisible by 2**len(decoder_schedule)={decoder_stride}."
                )
        if self.state_embed_dim is None:
            self.state_embed_dim = self.state_dim
