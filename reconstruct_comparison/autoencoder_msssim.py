from __future__ import annotations

import torch
import torch.nn as nn

from .autoencoder_lightweight_unet import LightweightAutoencoderUNet
from .autoencoder_texture_unet import TextureAwareAutoencoderUNet


class MSSSIMAutoencoderUNet(LightweightAutoencoderUNet):
    """Variant tuned for pure MS-SSIM optimisation."""

    def __init__(self) -> None:
        super().__init__(base_channels=40, latent_channels=120)


class FocalMSSSIMAutoencoderUNet(TextureAwareAutoencoderUNet):
    """Higher-capacity variant for focal MS-SSIM training."""

    def __init__(self) -> None:
        super().__init__(base_channels=72, latent_channels=224)


class MSSSIMLoss(nn.Module):
    """Wrapper that returns the mean MS-SSIM reconstruction loss."""

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        from predict_mario_ms_ssim import ms_ssim_loss

        return ms_ssim_loss(input, target)


class FocalMSSSIMLoss(nn.Module):
    """Apply focal reweighting to per-sample MS-SSIM losses."""

    def __init__(self, gamma: float = 2.0, max_weight: float = 5.0, eps: float = 1e-6) -> None:
        super().__init__()
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        if max_weight <= 0:
            raise ValueError("max_weight must be positive.")
        self.gamma = gamma
        self.max_weight = max_weight
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        from reconstruct_comparison.metrics import ms_ssim_per_sample

        scores = ms_ssim_per_sample(input, target)
        losses = 1.0 - scores
        norm = losses.detach().mean().clamp_min(self.eps)
        weight = torch.pow(losses / norm, self.gamma).clamp(max=self.max_weight)
        return (weight * losses).mean()


__all__ = [
    "MSSSIMAutoencoderUNet",
    "FocalMSSSIMAutoencoderUNet",
    "MSSSIMLoss",
    "FocalMSSSIMLoss",
]
