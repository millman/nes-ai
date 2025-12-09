"""Baseline convolutional encoder/decoder preserved from the original trainer."""
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def _norm_groups(out_ch: int) -> int:
    return max(1, out_ch // 8)


class DownBlock(nn.Module):
    """Conv block with stride-2 contraction followed by local refinement."""

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


class UpBlock(nn.Module):
    """ConvTranspose block that mirrors DownBlock with stride-2 expansion."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        groups = _norm_groups(out_ch)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder(nn.Module):
    """Convolutional encoder that maps images to fixed-size latent vectors.

    Architecture overview:
        Input image (input_hw × input_hw × in_channels)
            ↓
        Conv blocks following channel_schedule (each halves spatial resolution)
            ↓
        Final feature map: (input_hw / 2^num_blocks) × same × channel_schedule[-1]
            ↓
        AdaptiveAvgPool2d(1) → 1×1×channel_schedule[-1]
            ↓
        Optional projection: Linear(channel_schedule[-1] → latent_dim)
            ↓
        Output: latent vector of size latent_dim

    The latent_dim parameter allows the output embedding dimension to differ from
    the final channel count in the schedule.
    """

    def __init__(
        self,
        in_channels: int,
        schedule: Tuple[int, ...],
        input_hw: int,
        *,
        latent_dim: Optional[int] = None,
    ):
        super().__init__()
        if not schedule:
            raise ValueError("Channel schedule must be non-empty")

        blocks: List[nn.Module] = []
        ch_prev = in_channels

        for ch in schedule:
            blocks.append(DownBlock(ch_prev, ch))
            ch_prev = ch

        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Output latent dimension
        conv_out_dim = schedule[-1]
        self.latent_dim = latent_dim if latent_dim is not None else conv_out_dim

        # Add projection layer if latent_dim differs from conv output
        if self.latent_dim != conv_out_dim:
            self.latent_proj: Optional[nn.Module] = nn.Sequential(
                nn.Linear(conv_out_dim, self.latent_dim),
                nn.SiLU(inplace=True),
            )
        else:
            self.latent_proj = None

        self.in_channels = in_channels
        self.channel_schedule = schedule
        self.input_hw = input_hw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = x
        for block in self.blocks:
            feats = block(feats)

        # Pool and optionally project
        pooled = self.pool(feats).flatten(1)
        if self.latent_proj is not None:
            pooled = self.latent_proj(pooled)

        return pooled

    def shape_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "module": "Encoder",
            "input": (self.input_hw, self.input_hw, self.in_channels),
            "stages": [],
        }

        h = self.input_hw
        w = self.input_hw
        c = self.in_channels
        for idx, out_ch in enumerate(self.channel_schedule, start=1):
            next_h = (h + 1) // 2
            next_w = (w + 1) // 2
            stage = {
                "stage": idx,
                "in": (h, w, c),
                "out": (next_h, next_w, out_ch),
            }
            info["stages"].append(stage)
            h, w, c = next_h, next_w, out_ch
        # conv_out_dim is the channel count after the last conv block (before projection)
        conv_out_dim = self.channel_schedule[-1] if self.channel_schedule else c
        info["conv_out_dim"] = conv_out_dim
        # latent_dim is the final output dimension (after optional projection)
        info["latent_dim"] = self.latent_dim
        info["has_projection"] = self.latent_proj is not None
        return info


class VisualizationDecoder(nn.Module):
    """Decoder that reconstructs images from latent vectors.

    Architecture overview:
        Input: latent vector of size latent_dim
            ↓
        Linear projection: latent_dim → (start_ch × start_hw × start_hw)
            ↓
        Reshape to spatial: start_hw × start_hw × start_ch
            ↓
        UpBlocks (each doubles spatial resolution, mirrors encoder's channel_schedule)
            ↓
        Output: image_size × image_size × image_channels

    The decoder reconstructs images purely from the latent vector. There is no
    skip connection from the encoder - all spatial information must be encoded
    in the latent. This forces the encoder to capture fine details in the latent
    representation rather than relying on shortcuts.

    The channel_schedule should match the encoder's schedule so that the
    up-convolutions mirror the down-convolutions.
    """

    def __init__(
        self,
        latent_dim: int,
        image_channels: int,
        image_size: int,
        channel_schedule: Tuple[int, ...],
        latent_hw: int,
    ) -> None:
        super().__init__()
        if not channel_schedule:
            raise ValueError("channel_schedule must be non-empty for decoder construction.")
        scale = 2 ** len(channel_schedule)
        if image_size % scale != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by 2**len(channel_schedule)={scale} "
                "so the decoder can mirror the encoder."
            )
        expected_hw = image_size // scale
        if latent_hw != expected_hw:
            raise ValueError(
                f"latent_hw must equal image_size // 2**len(channel_schedule) ({expected_hw}), "
                f"got {latent_hw}. Adjust ModelConfig.latent_hw or image_size."
            )
        self.start_hw = expected_hw
        self.start_ch = channel_schedule[-1]
        self.image_channels = image_channels
        self.image_size = image_size
        self.channel_schedule = channel_schedule
        self.latent_hw = latent_hw
        self.latent_dim = latent_dim

        self.project = nn.Sequential(
            nn.Linear(latent_dim, self.start_ch * self.start_hw * self.start_hw),
            nn.SiLU(inplace=True),
        )
        up_blocks: List[nn.Module] = []
        in_ch = self.start_ch
        for out_ch in reversed(channel_schedule):
            up_blocks.append(UpBlock(in_ch, out_ch))
            in_ch = out_ch
        self.up_blocks = nn.ModuleList(up_blocks)

        head_hidden = max(in_ch // 2, 1)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_hidden, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(head_hidden, image_channels, kernel_size=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        original_shape = latent.shape[:-1]
        latent = latent.reshape(-1, latent.shape[-1])
        projected = self.project(latent)
        decoded = projected.view(-1, self.start_ch, self.start_hw, self.start_hw)

        # Upsample through blocks (no skip connection)
        for block in self.up_blocks:
            decoded = block(decoded)

        frame = self.head(decoded)
        frame = frame.view(*original_shape, self.image_channels, self.image_size, self.image_size)
        return frame

    def shape_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "module": "Decoder",
            "latent_dim": self.latent_dim,
            "projection": (self.start_hw, self.start_hw, self.start_ch),
            "upsample": [],
            "final_target": (self.image_size, self.image_size, self.image_channels),
        }
        hw = self.start_hw
        ch = self.start_ch
        for idx, out_ch in enumerate(reversed(self.channel_schedule), start=1):
            prev_hw = hw
            hw = max(1, hw * 2)
            stage = {
                "stage": idx,
                "in": (prev_hw, prev_hw, ch),
                "out": (hw, hw, out_ch),
            }
            info["upsample"].append(stage)
            ch = out_ch
        info["pre_resize"] = (hw, hw, self.image_channels)
        info["needs_resize"] = hw != self.image_size
        return info
