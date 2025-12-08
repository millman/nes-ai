"""Baseline convolutional encoder/decoder preserved from the original trainer."""
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_groups(out_ch: int) -> int:
    return max(1, out_ch // 8)


def _make_blur_kernel() -> torch.Tensor:
    kernel = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32)
    kernel = kernel / kernel.sum()
    return kernel


class Blur(nn.Module):
    """Depthwise blur filter to anti-alias stride-2 downsamples."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        kernel = _make_blur_kernel().unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel.expand(self.channels, 1, 3, 3)
        return F.conv2d(x, kernel, stride=1, padding=1, groups=self.channels)


class MotionSensitiveStem(nn.Module):
    """Depthwise-separable stem that preserves spatial detail before downsampling."""

    def __init__(self, in_ch: int, out_ch: int, use_blur: bool = True) -> None:
        super().__init__()
        groups = _norm_groups(out_ch)
        layers: List[nn.Module] = []

        # Depthwise: captures local patterns per-channel at full resolution
        layers.extend([
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch),
            nn.SiLU(inplace=True),
        ])

        # Pointwise: learns channel mixing without downsampling
        layers.extend([
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        ])

        # Apply blur before downsampling for anti-aliasing
        if use_blur:
            layers.append(Blur(out_ch))

        # Only NOW apply stride-2 downsampling
        layers.extend([
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    """Conv block with optional stride-2 contraction followed by local refinement."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        downsample: bool = True,
        use_blur: bool = True,
    ) -> None:
        super().__init__()
        groups = _norm_groups(out_ch)
        stride = 2 if downsample else 1
        layers: List[nn.Module] = []
        if downsample and use_blur:
            layers.append(Blur(in_ch))
        layers.extend(
            [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
                nn.GroupNorm(groups, out_ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(groups, out_ch),
                nn.SiLU(inplace=True),
            ]
        )
        self.net = nn.Sequential(*layers)

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
    def __init__(
        self,
        in_channels: int,
        schedule: Tuple[int, ...],
        input_hw: int,
        *,
        first_block_no_downsample: bool = True,
        use_blur: bool = True,
        use_motion_stem: bool = True,
    ):
        super().__init__()
        if not schedule:
            raise ValueError("Channel schedule must be non-empty")

        blocks: List[nn.Module] = []
        ch_prev = in_channels

        for idx, ch in enumerate(schedule):
            if idx == 0:
                # First block: use motion-sensitive stem if enabled
                if use_motion_stem:
                    blocks.append(MotionSensitiveStem(ch_prev, ch, use_blur=use_blur))
                else:
                    downsample = not first_block_no_downsample
                    blocks.append(DownBlock(ch_prev, ch, downsample=downsample, use_blur=use_blur))
            else:
                # Remaining blocks: standard downsampling
                blocks.append(DownBlock(ch_prev, ch, downsample=True, use_blur=use_blur))
            ch_prev = ch

        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.in_channels = in_channels
        self.channel_schedule = schedule
        self.input_hw = input_hw
        self.first_block_no_downsample = first_block_no_downsample
        self.use_motion_stem = use_motion_stem

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        detail_skip = None
        feats = x
        for idx, block in enumerate(self.blocks):
            feats = block(feats)
            if idx == 0:
                detail_skip = feats
        pooled = self.pool(feats).flatten(1)
        if detail_skip is None:
            raise RuntimeError("detail_skip capture failed in Encoder forward.")
        return pooled, detail_skip

    def shape_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "module": "Encoder",
            "input": (self.input_hw, self.input_hw, self.in_channels),
            "stages": [],
        }
        # First stage always downsamples by 2 (either via MotionSensitiveStem or DownBlock with downsample=True)
        # unless first_block_no_downsample is True for DownBlock
        first_stage_hw = self.input_hw // 2 if (self.use_motion_stem or not self.first_block_no_downsample) else self.input_hw
        info["detail_skip"] = (
            first_stage_hw,
            first_stage_hw,
            self.channel_schedule[0] if self.channel_schedule else self.in_channels,
        )

        h = self.input_hw
        w = self.input_hw
        c = self.in_channels
        for idx, out_ch in enumerate(self.channel_schedule, start=1):
            if idx == 1:
                stride = 2 if (self.use_motion_stem or not self.first_block_no_downsample) else 1
            else:
                stride = 2
            next_h = h if stride == 1 else (h + 1) // 2
            next_w = w if stride == 1 else (w + 1) // 2
            stage = {
                "stage": idx,
                "in": (h, w, c),
                "out": (next_h, next_w, out_ch),
            }
            info["stages"].append(stage)
            h, w, c = next_h, next_w, out_ch
        info["latent_dim"] = self.channel_schedule[-1] if self.channel_schedule else c
        return info


class VisualizationDecoder(nn.Module):
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

        # Skip connection adapter for detail_skip
        # detail_skip comes from first encoder stage at image_size // 2 resolution with channel_schedule[0] channels
        # It needs to be projected to match the decoder's channel count at the skip insertion point
        # At skip_insert_idx (second-to-last block), decoder has reversed(channel_schedule)[skip_insert_idx] channels
        skip_insert_idx = len(channel_schedule) - 2
        decoded_channels_at_skip = list(reversed(channel_schedule))[skip_insert_idx]
        self.skip_adapter = nn.Sequential(
            nn.GroupNorm(max(1, channel_schedule[0] // 8), channel_schedule[0]),
            nn.Conv2d(channel_schedule[0], decoded_channels_at_skip, kernel_size=1),
        )
        self.skip_gate = nn.Parameter(torch.zeros(1))

        head_hidden = max(in_ch // 2, 1)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_hidden, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(head_hidden, image_channels, kernel_size=1),
        )

    def forward(self, latent: torch.Tensor, detail_skip: torch.Tensor | None = None) -> torch.Tensor:
        original_shape = latent.shape[:-1]
        latent = latent.reshape(-1, latent.shape[-1])
        projected = self.project(latent)
        decoded = projected.view(-1, self.start_ch, self.start_hw, self.start_hw)

        # Prepare skip connection if available
        skip = None
        if detail_skip is not None:
            # Reshape detail_skip from (..., C, H, W) to (batch, C, H, W)
            skip = detail_skip.reshape(-1, detail_skip.shape[-3], detail_skip.shape[-2], detail_skip.shape[-1])
            skip = self.skip_adapter(skip)

        # Upsample through blocks
        # detail_skip is at image_size // 2 = 64x64, with channel_schedule[0] channels
        # We need to add it when decoded reaches 64x64, which is after the second-to-last block
        skip_insert_idx = len(self.up_blocks) - 2
        for idx, block in enumerate(self.up_blocks):
            decoded = block(decoded)

            # Add detail_skip when we reach the skip's resolution (64x64 for 128x128 images)
            if idx == skip_insert_idx and skip is not None:
                decoded = decoded + torch.tanh(self.skip_gate) * skip

        frame = self.head(decoded)
        frame = frame.view(*original_shape, self.image_channels, self.image_size, self.image_size)
        return frame

    def shape_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "module": "Decoder",
            "projection": (self.start_hw, self.start_hw, self.start_ch),
            "upsample": [],
            "final_target": (self.image_size, self.image_size, self.image_channels),
            "detail_skip": (self.image_size // 2, self.image_size // 2, self.channel_schedule[0]),
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
