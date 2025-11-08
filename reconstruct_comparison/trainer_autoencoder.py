from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseAutoencoderTrainer


class AutoencoderTrainer(BaseAutoencoderTrainer):
    """Trainable encoder/decoder pair using an explicit reconstruction loss."""

    def __init__(
        self,
        name: str,
        model: nn.Module,
        *,
        device: torch.device,
        lr: float,
        loss_fn: nn.Module,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__(
            name,
            model,
            device=device,
            lr=lr,
            loss_fn=loss_fn,
            weight_decay=weight_decay,
        )


__all__ = ["AutoencoderTrainer"]
