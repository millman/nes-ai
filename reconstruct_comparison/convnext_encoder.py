from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ConvNeXt_Base_Weights, convnext_base


@torch.no_grad()
def _convnext_encoder(weights: ConvNeXt_Base_Weights) -> nn.Module:
    """Frozen ConvNeXt-Base encoder that outputs the final feature stage."""
    model = convnext_base(weights=weights)
    encoder = model.features  # last block yields (B,1024,7,7)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


__all__ = ["_convnext_encoder"]
