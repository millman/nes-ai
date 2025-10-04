"""Common tensor/image utilities for reconstruction scripts."""

from __future__ import annotations

import math
import random
from typing import Callable, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn

TensorFn = Callable[[torch.Tensor], torch.Tensor]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_float01(t: torch.Tensor, device: torch.device, non_blocking: bool = True) -> torch.Tensor:
    if t.dtype != torch.uint8:
        return t.to(device=device, non_blocking=non_blocking)
    return t.to(device=device, non_blocking=non_blocking, dtype=torch.float32) / 255.0


def tensor_to_pil(t: torch.Tensor, *, denormalize: Optional[TensorFn] = None) -> Image.Image:
    tensor = t.detach().cpu()
    if tensor.dtype == torch.uint8:
        arr = tensor.permute(1, 2, 0).contiguous().numpy()
    else:
        tensor = tensor.float()
        if denormalize is not None:
            tensor = denormalize(tensor)
        tensor = tensor.clamp(0.0, 1.0)
        arr = (tensor.permute(1, 2, 0).contiguous().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def psnr_01(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.dtype != torch.float32:
        x01 = x.float().div(255.0)
    else:
        x01 = x
    if y.dtype != torch.float32:
        y01 = y.float().div(255.0)
    else:
        y01 = y
    x01 = x01.clamp(0.0, 1.0)
    y01 = y01.clamp(0.0, 1.0)
    mse = F.mse_loss(x01, y01, reduction="mean").clamp_min(eps)
    return 10.0 * torch.log10(1.0 / mse)


def grad_norm(module: nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().data.pow(2).sum().cpu())
    return math.sqrt(total) if total > 0 else 0.0


__all__ = [
    "set_seed",
    "to_float01",
    "tensor_to_pil",
    "psnr_01",
    "grad_norm",
]
