from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .model_utils import _norm_groups


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        groups = _norm_groups(out_ch)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    """Strided contraction block that preserves channel locality."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        groups = _norm_groups(out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    """Residual block with optional down-sampling via stride."""

    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(_norm_groups(out_ch), out_ch)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_norm_groups(out_ch), out_ch)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + identity
        return self.act(out)


class ResidualUpBlock(nn.Module):
    """Upsampling residual block using transposed convolutions."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.norm = nn.GroupNorm(_norm_groups(out_ch), out_ch)
        self.act = nn.SiLU(inplace=True)
        self.block = ResidualBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.norm(x)
        x = self.act(x)
        return self.block(x)


class PreActResidualBlock(nn.Module):
    """Pre-activation residual block following ResNet v2 conventions."""

    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_norm_groups(in_ch), in_ch)
        self.act1 = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(_norm_groups(out_ch), out_ch)
        self.act2 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
            if stride != 1 or in_ch != out_ch
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm1(x)
        out = self.act1(out)
        identity = self.skip(out) if self.skip is not None else x
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.conv2(out)
        return out + identity


class PreActResidualUpBlock(nn.Module):
    """Pre-activation upsampling block mirroring ResNet v2 style."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_norm_groups(in_ch), in_ch)
        self.act = nn.SiLU(inplace=True)
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.block = PreActResidualBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.act(x)
        x = self.up(x)
        return self.block(x)


__all__ = [
    "UpBlock",
    "DownBlock",
    "ResidualBlock",
    "ResidualUpBlock",
    "PreActResidualBlock",
    "PreActResidualUpBlock",
]
