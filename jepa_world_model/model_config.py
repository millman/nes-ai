#!/usr/bin/env python3
"""Model configuration dataclass for the JEPA world model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class LayerNormConfig:
    """LayerNorm toggles for debugging latent stability."""

    h2z_projector: bool = False
    z2h_projector: bool = False
    h2z_delta_projector: bool = False

    # Action deltas are learned residual updates applied repeatedly over rollout time,
    # so their scale directly compounds. Poorly scaled deltas can explode gradients
    # because each step adds another unnormalized update to h/p, amplifying errors and
    # destabilizing training. LayerNorm is especially suitable here because it
    # normalizes the additive update itself, keeping step-to-step residual magnitude
    # consistent without constraining the base representation. In contrast, LayerNorm
    # on core encoders/decoders or main state embeddings can wash out absolute scale or
    # interfere with downstream losses that rely on raw feature magnitudes. The delta
    # projector is the most "safe" place to normalize because it governs residual
    # corrections rather than the primary representational content.
    action_delta_projector_z: Optional[bool] = True
    action_delta_projector_h: Optional[bool] = True
    action_delta_projector_p: Optional[bool] = True

    inverse_dynamics_z: bool = False
    inverse_dynamics_h: bool = False
    inverse_dynamics_p: bool = False

    h_next: bool = False
    pose_correction_projector: bool = False

@dataclass
class ModelConfig:
    """Dimensional flow:

        image (image_size^2·in_channels)
            └─ Encoder(encoder_schedule → pool → z_dim = encoder_schedule[-1])
            ▼
        z_t ────────────────────────────────┐
                        ├─ Predictor([z_t, h_t, action_t]) → ẑ_{t+1}, ĥ_{t+1}
        h_t (state_dim) ─────────────────────┘
            │
            ├→ PoseDelta(p_t, h_t, a_t) → Δp_t (pose increment)
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
    state_dim: int = 256

    # Number of initial seq_len frames excluded from h/pose losses only.
    warmup_frames_h: int = 0

    # Planning pose dimensionality (defaults to state_dim).
    pose_dim: Optional[int] = None
    # Stop-grad h when conditioning pose deltas to keep p losses from shaping dynamics.
    pose_delta_detach_h: bool = True
    # Apply observation-conditioned pose correction using z_{t+1}.
    pose_correction_use_z: bool = False
    # Stop-grad z when applying pose correction.
    pose_correction_detach_z: bool = True

    # LayerNorm toggles for debugging.
    layer_norms: LayerNormConfig = field(default_factory=LayerNormConfig)

    # Apply spectral norm to predictor linear layers for h stability.
    predictor_spectral_norm: bool = True

    # Normalize encoder/predicted z to unit norm.
    z_norm: bool = False

    # Optional head toggles (resolved from loss weights at runtime when training).
    enable_inverse_dynamics_z: Optional[bool] = None
    enable_inverse_dynamics_h: Optional[bool] = None
    enable_inverse_dynamics_p: Optional[bool] = None
    enable_action_delta_z: Optional[bool] = None
    enable_action_delta_h: Optional[bool] = None
    enable_action_delta_p: Optional[bool] = None
    enable_h2z_delta: Optional[bool] = None

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
        if self.pose_dim is None:
            self.pose_dim = self.state_dim
