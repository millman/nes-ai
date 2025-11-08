from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


@torch.no_grad()
def _resnet_encoder(weights: ResNet50_Weights) -> nn.Module:
    """Frozen ResNet-50 encoder returning the final spatial feature map."""
    model = resnet50(weights=weights)
    layers = list(model.children())[:-2]  # drop avgpool + fc
    encoder = nn.Sequential(*layers)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


__all__ = ["_resnet_encoder"]
