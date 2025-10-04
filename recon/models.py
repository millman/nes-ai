"""Model building blocks shared by reconstruction scripts."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class DownBlock(nn.Module):
    """Strided conv block with SiLU activations for encoder down-sampling."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=2, padding=padding),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    """Transpose-conv block with SiLU activations for decoder up-sampling."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """Lightweight CNN decoder shared by lightweight and MobileNet pipelines."""

    def __init__(
        self,
        z_dim: int = 128,
        *,
        h_shape: tuple[int, int, int] = (256, 15, 14),
        clamp_output: bool = True,
    ):
        super().__init__()
        self.h_shape = h_shape
        self.clamp_output = clamp_output
        self.fc = nn.Linear(z_dim, int(np.prod(self.h_shape)))
        self.pre = nn.SiLU(inplace=True)
        self.up1 = UpBlock(self.h_shape[0], 128)
        self.up2 = UpBlock(128, 64)
        self.up3 = UpBlock(64, 32)
        self.up4 = UpBlock(32, 16)
        self.head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1),
        )

        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, *self.h_shape)
        h = self.pre(h)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        x = self.head(h)
        return torch.clamp(x, 0.0, 1.0) if self.clamp_output else x


__all__ = ["DownBlock", "UpBlock", "Decoder"]
