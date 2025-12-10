"""Baseline convolutional encoder/decoder preserved from the original trainer."""
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


def _norm_groups(out_ch: int) -> int:
    return max(1, out_ch // 8)


class DownBlock(nn.Module):
    """Conv block with stride-2 contraction."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        groups = _norm_groups(out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm1(self.conv1(x)))
        out = self.act(self.norm2(self.conv2(out)))
        return out


class UpBlock(nn.Module):
    """Upsample block with stride-2 expansion."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        groups = _norm_groups(out_ch)
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm1(self.upsample(x)))
        out = self.act(self.norm2(self.conv(out)))
        return out


class Encoder(nn.Module):
    """Convolutional encoder that maps images to fixed-size latent vectors.

    Architecture overview:
        Input image (input_hw × input_hw × in_channels)
            ↓
        Conv blocks following schedule (each halves spatial resolution)
            ↓
        Final feature map: (input_hw / 2^num_blocks) × same × schedule[-1]
            ↓
        AdaptiveAvgPool2d(1) → 1×1×schedule[-1]
            ↓
        Output: latent vector of size schedule[-1]
    """

    def __init__(
        self,
        in_channels: int,
        schedule: Tuple[int, ...],
        input_hw: int,
    ):
        super().__init__()
        if not schedule:
            raise ValueError("Schedule must be non-empty")

        blocks: List[nn.Module] = []
        ch_prev = in_channels

        for ch in schedule:
            blocks.append(DownBlock(ch_prev, ch))
            ch_prev = ch

        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.latent_dim = schedule[-1]

        self.in_channels = in_channels
        self.schedule = schedule
        self.input_hw = input_hw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = x
        for block in self.blocks:
            feats = block(feats)

        return self.pool(feats).flatten(1)

    def shape_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "module": "Encoder",
            "input": (self.input_hw, self.input_hw, self.in_channels),
            "stages": [],
        }

        h = self.input_hw
        w = self.input_hw
        c = self.in_channels
        for idx, out_ch in enumerate(self.schedule, start=1):
            next_h = (h + 1) // 2
            next_w = (w + 1) // 2
            stage = {
                "stage": idx,
                "in": (h, w, c),
                "out": (next_h, next_w, out_ch),
            }
            info["stages"].append(stage)
            h, w, c = next_h, next_w, out_ch
        info["latent_dim"] = self.latent_dim
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
        UpBlocks (each doubles spatial resolution)
            ↓
        Output: image_size × image_size × image_channels

    The decoder reconstructs images purely from the latent vector. There is no
    skip connection from the encoder - all spatial information must be encoded
    in the latent. This forces the encoder to capture fine details in the latent
    representation rather than relying on shortcuts.

    The schedule can differ from the encoder's schedule. The decoder
    starts at start_ch (schedule[-1]) and upsamples through the
    reversed schedule.
    """

    def __init__(
        self,
        latent_dim: int,
        image_channels: int,
        image_size: int,
        schedule: Tuple[int, ...],
    ) -> None:
        super().__init__()
        if not schedule:
            raise ValueError("Schedule must be non-empty for decoder construction.")
        num_layers = len(schedule)
        scale = 2 ** num_layers
        if image_size % scale != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by 2**len(schedule)={scale}."
            )
        start_hw = image_size // scale
        self.start_hw = start_hw
        self.start_ch = schedule[-1]
        self.image_channels = image_channels
        self.image_size = image_size
        self.schedule = schedule
        self.latent_dim = latent_dim

        self.project = nn.Sequential(
            nn.Linear(latent_dim, self.start_ch * self.start_hw * self.start_hw),
            nn.SiLU(inplace=True),
        )
        up_blocks: List[nn.Module] = []
        in_ch = self.start_ch
        for out_ch in reversed(schedule):
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
        for idx, out_ch in enumerate(reversed(self.schedule), start=1):
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
