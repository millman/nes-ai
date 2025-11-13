from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelsLastLayerNorm(nn.Module):
    """LayerNorm wrapper that keeps tensors in NCHW form internally."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.permute(0, 2, 3, 1)
        h = self.norm(h)
        return h.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block as described by Liu et al. (2022, arXiv:2201.03545)."""

    def __init__(
        self,
        dim: int,
        *,
        expansion: int = 4,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = ChannelsLastLayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * expansion, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + x


def _make_blocks(
    dim: int,
    depth: int,
    *,
    layer_scale_init_value: float,
) -> nn.Sequential:
    if depth <= 0:
        return nn.Sequential()
    blocks = [
        ConvNeXtBlock(dim, layer_scale_init_value=layer_scale_init_value) for _ in range(depth)
    ]
    return nn.Sequential(*blocks)


class ConvNeXtDownsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.norm = ChannelsLastLayerNorm(in_ch)
        self.reduction = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.reduction(x)


class ConvNeXtUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.norm = ChannelsLastLayerNorm(in_ch)
        self.expand = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.expand(x)


@dataclass(frozen=True)
class ConvNeXtAutoencoderConfig:
    base_channels: int = 48
    latent_channels: int = 128
    encoder_depths: Sequence[int] = (3, 4)
    decoder_depths: Sequence[int] = (3, 2)
    latent_depth: int = 2
    layer_scale_init_value: float = 1e-6
    input_hw: Tuple[int, int] = (224, 224)


class ConvNeXtEncoder(nn.Module):
    """Two-stage ConvNeXt encoder producing a 28×28 latent grid."""

    def __init__(self, cfg: ConvNeXtAutoencoderConfig) -> None:
        super().__init__()
        height, width = cfg.input_hw
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("input_hw must be divisible by 8 to match ConvNeXt strides.")
        self.input_hw = cfg.input_hw
        self.latent_hw = (height // 8, width // 8)
        stem_out = cfg.base_channels
        hidden = cfg.base_channels * 2
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_out, kernel_size=4, stride=4),
            ChannelsLastLayerNorm(stem_out),
        )
        enc_depths = list(cfg.encoder_depths)
        if len(enc_depths) != 2:
            raise ValueError("encoder_depths must contain two stage depths.")
        self.stage1 = _make_blocks(
            stem_out, enc_depths[0], layer_scale_init_value=cfg.layer_scale_init_value
        )
        self.down = ConvNeXtDownsample(stem_out, hidden)
        self.stage2 = _make_blocks(
            hidden, enc_depths[1], layer_scale_init_value=cfg.layer_scale_init_value
        )
        self.to_latent = nn.Conv2d(hidden, cfg.latent_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != self.input_hw:
            raise RuntimeError(
                f"Expected input spatial size {self.input_hw}, got {tuple(x.shape[-2:])}"
            )
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down(x)
        x = self.stage2(x)
        return self.to_latent(x)


class ConvNeXtDecoder(nn.Module):
    """Upsampling decoder that mirrors :class:`ConvNeXtEncoder`."""

    def __init__(self, cfg: ConvNeXtAutoencoderConfig) -> None:
        super().__init__()
        stem_out = cfg.base_channels
        hidden = cfg.base_channels * 2
        self.latent_hw = (cfg.input_hw[0] // 8, cfg.input_hw[1] // 8)
        self.output_hw = cfg.input_hw
        self.latent_channels = cfg.latent_channels
        dec_depths = list(cfg.decoder_depths)
        if len(dec_depths) != 2:
            raise ValueError("decoder_depths must contain two stage depths.")
        self.latent_blocks = _make_blocks(
            cfg.latent_channels,
            cfg.latent_depth,
            layer_scale_init_value=cfg.layer_scale_init_value,
        )
        self.up1 = ConvNeXtUpsample(cfg.latent_channels, hidden)
        self.stage_mid = _make_blocks(
            hidden, dec_depths[0], layer_scale_init_value=cfg.layer_scale_init_value
        )
        self.up2 = ConvNeXtUpsample(hidden, stem_out)
        self.stage_low = _make_blocks(
            stem_out, dec_depths[1], layer_scale_init_value=cfg.layer_scale_init_value
        )
        head_hidden = max(16, stem_out // 2)
        self.head = nn.Sequential(
            ChannelsLastLayerNorm(stem_out),
            nn.Conv2d(stem_out, head_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(head_hidden, 3, kernel_size=1),
        )

    def forward(
        self,
        latent: torch.Tensor,
        *,
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if latent.shape[-3] != self.latent_channels:
            raise RuntimeError(
                f"Expected {self.latent_channels} latent channels, got {latent.shape[-3]}"
            )
        if latent.shape[-2:] != self.latent_hw:
            raise RuntimeError(
                f"Expected latent spatial size {self.latent_hw}, got {tuple(latent.shape[-2:])}"
            )
        x = self.latent_blocks(latent)
        x = self.up1(x)
        x = self.stage_mid(x)
        x = self.up2(x)
        x = self.stage_low(x)
        x = self.head(x)
        final_hw = target_hw or self.output_hw
        if x.shape[-2:] != final_hw:
            x = F.interpolate(x, size=final_hw, mode="bilinear", align_corners=False)
        return x


class ConvNeXtAutoencoder(nn.Module):
    """Lightweight ConvNeXt-style autoencoder without pretrained weights.

    The architecture mirrors ConvNeXt-Tiny (Liu et al., 2022) but downsizes the
    widths/depths so the learnable parameter count (~1.9M) stays close to the
    existing :class:`LightweightAutoencoder` while keeping the latent tensor at
    28×28×128. All convolutions are trained from scratch to avoid ImageNet
    supervision while retaining the depthwise + pointwise structure that makes
    ConvNeXt blocks effective for high-resolution inputs.
    """

    def __init__(self, cfg: Optional[ConvNeXtAutoencoderConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ConvNeXtAutoencoderConfig()
        self.encoder = ConvNeXtEncoder(self.cfg)
        self.decoder = ConvNeXtDecoder(self.cfg)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(
        self,
        latent: torch.Tensor,
        *,
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        return self.decoder(latent, target_hw=target_hw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        return self.decode(latent, target_hw=x.shape[-2:])


__all__ = [
    "ChannelsLastLayerNorm",
    "ConvNeXtAutoencoder",
    "ConvNeXtAutoencoderConfig",
    "ConvNeXtDecoder",
    "ConvNeXtEncoder",
]
