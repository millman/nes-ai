#!/usr/bin/env python3
"""Compare NES Mario frame reconstruction across multiple encoder/decoder strategies.

Frozen ImageNet encoders pair with learned decoders while several lightweight
autoencoders explore focal L1, pure MS-SSIM, focal MS-SSIM, and style/contrastive
objectives; each branch can be enabled or disabled individually for ablations.
"""
from __future__ import annotations

import logging
import random
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import time
import textwrap

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.models import (
    ResNet50_Weights,
    ConvNeXt_Base_Weights,
    VGG16_Weights,
    convnext_base,
    resnet50,
    vgg16,
)
import tyro
from PIL import Image

from predict_mario_ms_ssim import default_transform, ms_ssim_loss, pick_device, unnormalize
from predict_mario4 import (
    ImageEncoder as Mario4ImageEncoder,
    ImageDecoder as Mario4ImageDecoder,
    ImageDecoderMirrored as Mario4ImageDecoderMirrored,
)
from trajectory_utils import list_state_frames, list_traj_dirs


SCRIPT_START_TIME = time.time()

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "resnet50": "ResNet-50 (MSE)",
    "convnext_base": "ConvNeXt-Base (MSE)",
    "l1_autoencoder": "Autoencoder (L1)",
    "mse_autoencoder": "Autoencoder (MSE)",
    "focal_autoencoder": "Autoencoder (Focal L1)",
    "smoothl1_autoencoder": "Autoencoder (Smooth L1)",
    "cauchy_autoencoder": "Autoencoder (Cauchy)",
    "style_contrast_autoencoder": "Autoencoder (Style + PatchNCE)",
    "msssim_autoencoder": "Autoencoder (MS-SSIM)",
    "focal_msssim_autoencoder": "Autoencoder (Focal MS-SSIM)",
    "no_skip_autoencoder": "Autoencoder (Spatial Latent)",
    "no_skip_patch_autoencoder": "Autoencoder (No Skip Patch)",
    "skip_train_autoencoder": "Autoencoder (Train Skip, Eval Zero)",
    "resnet_autoencoder": "Autoencoder (ResNet Blocks)",
    "resnetv2_autoencoder": "Autoencoder (ResNet v2)",
    "modern_attn_autoencoder": "Autoencoder (Modern ResNet + Attn)",
    "mario4_autoencoder": "Autoencoder (Mario4)",
    "mario4_mirrored_autoencoder": "Mario4 Mirrored Decoder",
    "mario4_spatial_autoencoder": "Mario4 Spatial Softmax 192",
    "mario4_large_autoencoder": "Mario4 Latent 1024",
    "mario4_spatial_large_autoencoder": "Mario4 Spatial Softmax 1024",
}


def _display_name(name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(name, name)


def _flatten_named_parameters(module: nn.Module) -> List[Tuple[str, int]]:
    params: List[Tuple[str, int]] = []
    for name, param in module.named_parameters():
        params.append((name, param.numel()))
    return params


def _summarize_parameters(name: str, module: nn.Module, *, logger: logging.Logger) -> None:
    entries = _flatten_named_parameters(module)
    total = sum(count for _, count in entries)
    logger.info("%s parameters: %d", _display_name(name), total)
    for entry_name, count in entries:
        logger.info("    %s: %d", entry_name, count)


class _ElapsedTimeFormatter(logging.Formatter):
    """Inject elapsed wall-clock time since process start into log records."""

    def __init__(self, *args: object, start_time: float, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._start_time = start_time

    def format(self, record: logging.LogRecord) -> str:
        elapsed_seconds = record.created - self._start_time
        record.elapsed = f"{elapsed_seconds:9.2f}s"
        try:
            return super().format(record)
        finally:
            # Clean up to avoid leaking the custom attribute outside this formatter.
            del record.elapsed


def _norm_groups(channels: int) -> int:
    return 8 if channels % 8 == 0 else 1


def default(val: Optional[int], d: int) -> int:
    return d if val is None else val


def _group_count(channels: int, max_groups: int) -> int:
    """Pick the largest group count ≤ max_groups that divides channels."""
    limit = min(max_groups, channels)
    for groups in range(limit, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("reconstruct_mario_comparison")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    formatter = _ElapsedTimeFormatter(
        "%(asctime)s [Δ%(elapsed)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        start_time=SCRIPT_START_TIME,
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class MarioFrameDataset(Dataset):
    """Flat dataset of Mario frames with ImageNet preprocessing."""

    def __init__(
        self,
        root_dir: Path,
        *,
        transform: Optional[T.Compose] = None,
        max_trajs: Optional[int] = None,
    ) -> None:
        self.transform = transform or default_transform()
        self.paths: List[Path] = []
        traj_count = 0
        for traj_dir in list_traj_dirs(root_dir):
            if not traj_dir.is_dir():
                continue
            states_dir = traj_dir / "states"
            if not states_dir.is_dir():
                continue
            for frame_path in list_state_frames(states_dir):
                self.paths.append(frame_path)
            traj_count += 1
            if max_trajs is not None and traj_count >= max_trajs:
                break
        if not self.paths:
            raise RuntimeError(f"No frames found under {root_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.paths[idx]
        with Image.open(path).convert("RGB") as img:
            tensor = self.transform(img)
        return tensor, str(path)


def load_image_batch(paths: Sequence[str], transform: T.Compose) -> torch.Tensor:
    tensors = []
    for path in paths:
        with Image.open(path).convert("RGB") as img:
            tensors.append(transform(img))
    if not tensors:
        raise RuntimeError("No images provided for visualisation batch.")
    return torch.stack(tensors)


def sample_random_batch(dataset: MarioFrameDataset, count: int) -> torch.Tensor:
    if count <= 0:
        raise ValueError("count must be positive.")
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty; cannot sample frames.")
    indices = random.sample(range(len(dataset)), k=min(count, len(dataset)))
    tensors = [dataset[idx][0] for idx in indices]
    return torch.stack(tensors)


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8 if out_ch % 8 == 0 else 1, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8 if out_ch % 8 == 0 else 1, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """Shared decoder architecture parameterised by encoder channel width."""

    def __init__(self, in_channels: int, *, base_channels: int = 512) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=1),
            nn.SiLU(inplace=True),
        )
        self.up1 = UpBlock(base_channels, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 96)
        self.up4 = UpBlock(96, 64)
        self.up5 = UpBlock(64, 48)
        self.head = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = self.proj(feat)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        h = self.up5(h)
        out = self.head(h)
        if out.shape[-2:] != (224, 224):
            out = F.interpolate(out, size=(224, 224), mode="bilinear", align_corners=False)
        return out


class DownBlock(nn.Module):
    """Strided contraction block that preserves channel locality."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        groups = 8 if out_ch % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )
        # Shape summary for DownBlock:
        #   input  -> (B, in_ch,  H,  W)
        #   output -> (B, out_ch, H/2, W/2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    """Residual block with optional downsampling."""

    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(_norm_groups(out_ch), out_ch)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_norm_groups(out_ch), out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)
        else:
            self.skip = nn.Identity()

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
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_norm_groups(out_ch), out_ch)
        self.act2 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
        else:
            self.skip = None

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


class Residual(nn.Module):
    """Wrap a module with a residual connection, optionally learning the skip."""

    def __init__(
        self,
        fn: nn.Module,
        *,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        learned_skip = (
            in_channels is not None
            and out_channels is not None
            and in_channels != out_channels
        )
        self.fn = fn
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if learned_skip
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self.skip(x) + self.fn(x, **kwargs)


class ResNetBlock2d(nn.Module):
    """Pre-activation residual block with SiLU activations and GroupNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        *,
        groups: int = 32,
    ) -> None:
        super().__init__()
        out_channels = default(out_channels, in_channels)
        self.norm1 = nn.GroupNorm(_group_count(in_channels, groups), in_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels, groups), out_channels)
        self.act2 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Multi-head self-attention operating over spatial tokens."""

    def __init__(
        self,
        channels: int,
        *,
        heads: int = 4,
        dim_head: int = 64,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__()
        self.heads = heads
        inner = heads * dim_head
        self.norm = nn.GroupNorm(1, channels)
        self.to_qkv = nn.Conv2d(channels, inner * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(inner, channels, kernel_size=1)
        self.scale = dim_head ** -0.5
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        self.max_tokens = max_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_n = self.norm(x)
        tokens = h * w
        if tokens > self.max_tokens:
            pool_factor = int(math.ceil(math.sqrt(tokens / self.max_tokens)))
            pooled = F.avg_pool2d(x_n, kernel_size=pool_factor, stride=pool_factor, ceil_mode=True)
            target_size = pooled.shape[-2:]
        else:
            pooled = x_n
            target_size = (h, w)

        qkv = self.to_qkv(pooled)
        q, k, v = qkv.chunk(3, dim=1)

        def reshape_heads(t: torch.Tensor) -> torch.Tensor:
            bsz, ch, height, width = t.shape
            t = t.view(bsz, self.heads, ch // self.heads, height * width)
            return t

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)
        attn = torch.einsum("bhdi,bhdj->bhij", q * self.scale, k)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhij,bhdj->bhdi", attn, v)
        out = out.contiguous().view(b, -1, target_size[0] * target_size[1]).view(
            b, -1, target_size[0], target_size[1]
        )
        out = self.proj(out)
        if target_size != (h, w):
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out + x


class ModernResNetAttnBlock(nn.Module):
    """Residual block that combines modern conv block with spatial attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        *,
        heads: int = 4,
        dim_head: int = 64,
        groups: int = 32,
    ) -> None:
        super().__init__()
        out_channels = default(out_channels, in_channels)
        self.residual_block = ResNetBlock2d(
            in_channels, out_channels, groups=groups
        )
        self.attn = Residual(
            SelfAttention2d(out_channels, heads=heads, dim_head=dim_head),
            in_channels=out_channels,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual_block(x)
        x = self.attn(x)
        return x


class ModernResNetAttnDown(nn.Module):
    """Downsampling block built around ModernResNetAttnBlock."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        heads: int = 4,
        dim_head: int = 64,
        groups: int = 32,
    ) -> None:
        super().__init__()
        self.block = ModernResNetAttnBlock(
            in_channels,
            out_channels,
            heads=heads,
            dim_head=dim_head,
            groups=groups,
        )
        self.down = nn.Sequential(
            nn.GroupNorm(_group_count(out_channels, groups), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return self.down(x)


class ModernResNetAttnUp(nn.Module):
    """Upsampling block for the modern attention autoencoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        heads: int = 4,
        dim_head: int = 64,
        groups: int = 32,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ModernResNetAttnBlock(
            out_channels,
            out_channels,
            heads=heads,
            dim_head=dim_head,
            groups=groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.block(x)


class LightweightAutoencoder(nn.Module):
    """Compact encoder/decoder that trains quickly on a single GPU."""

    def __init__(self, base_channels: int = 48, latent_channels: int = 128) -> None:
        super().__init__()
        if base_channels % 8 != 0:
            raise ValueError("base_channels must be divisible by 8 for GroupNorm stability.")
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(inplace=True),
        )  # -> (B, base_channels, 224, 224)
        self.down1 = DownBlock(base_channels, base_channels * 2)    # -> (B, base_channels*2, 112, 112)
        self.down2 = DownBlock(base_channels * 2, base_channels * 3)  # -> (B, base_channels*3, 56, 56)
        self.down3 = DownBlock(base_channels * 3, latent_channels)  # -> (B, latent_channels, 28, 28)
        groups = 8 if latent_channels % 8 == 0 else 1
        self.bottleneck = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, latent_channels),
            nn.SiLU(inplace=True),
        )
        self.up1 = UpBlock(latent_channels, base_channels * 3)
        self.up2 = UpBlock(base_channels * 3, base_channels * 2)
        self.up3 = UpBlock(base_channels * 2, base_channels)
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels // 2, 3, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.stem(x)
        h1 = self.down1(h0)  # (B, base_channels*2, 112, 112)
        h2 = self.down2(h1)  # (B, base_channels*3, 56, 56)
        h3 = self.down3(h2)  # (B, latent_channels, 28, 28)
        b = self.bottleneck(h3)
        u1 = self.up1(b) + h2
        u2 = self.up2(u1) + h1
        u3 = self.up3(u2) + h0
        out = self.head(u3)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


class LightweightEncoder(nn.Module):
    """Encoder counterpart without exposing skip connections."""

    def __init__(self, base_channels: int = 48, latent_channels: int = 128) -> None:
        super().__init__()
        if base_channels % 8 != 0:
            raise ValueError("base_channels must be divisible by 8 for GroupNorm stability.")
        self.input_hw = (224, 224)
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(inplace=True),
        )
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 3)
        self.down3 = DownBlock(base_channels * 3, latent_channels)
        groups = 8 if latent_channels % 8 == 0 else 1
        self.bottleneck = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, latent_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != self.input_hw:
            raise RuntimeError(f"Expected input spatial size {self.input_hw}, got {tuple(x.shape[-2:])}")
        h = self.stem(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        return self.bottleneck(h)


class LightweightDecoder(nn.Module):
    """Decoder that mirrors LightweightEncoder without skip inputs."""

    def __init__(
        self,
        base_channels: int = 48,
        latent_channels: int = 128,
        output_hw: Tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        self.output_hw = output_hw
        self.up1 = UpBlock(latent_channels, base_channels * 3)
        self.up2 = UpBlock(base_channels * 3, base_channels * 2)
        self.up3 = UpBlock(base_channels * 2, base_channels)
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels // 2, 3, kernel_size=1),
        )

    def forward(
        self,
        latent: torch.Tensor,
        *,
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        h = self.up1(latent)
        h = self.up2(h)
        h = self.up3(h)
        out = self.head(h)
        final_hw = target_hw or self.output_hw
        if out.shape[-2:] != final_hw:
            out = F.interpolate(out, size=final_hw, mode="bilinear", align_corners=False)
        return out


class LightweightAutoencoderNoSkip(nn.Module):
    """Variant without skip connections; exposes encoder/decoder components."""

    def __init__(self, base_channels: int = 48, latent_channels: int = 128) -> None:
        super().__init__()
        self.encoder = LightweightEncoder(base_channels, latent_channels)
        self.decoder = LightweightDecoder(base_channels, latent_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent, target_hw=x.shape[-2:])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(
        self,
        latent: torch.Tensor,
        *,
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        return self.decoder(latent, target_hw=target_hw)


class LightweightAutoencoderNoSkipPatch(LightweightAutoencoderNoSkip):
    """No-skip autoencoder trained with multi-scale patch reconstruction loss."""

    def __init__(self, base_channels: int = 48, latent_channels: int = 128) -> None:
        super().__init__(base_channels, latent_channels)


class LightweightAutoencoderSkipTrain(nn.Module):
    """Trains with skip connections but removes them during evaluation/inference."""

    def __init__(self, base_channels: int = 48, latent_channels: int = 128) -> None:
        super().__init__()
        if base_channels % 8 != 0:
            raise ValueError("base_channels must be divisible by 8 for GroupNorm stability.")
        self.output_hw = (224, 224)
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(inplace=True),
        )
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 3)
        self.down3 = DownBlock(base_channels * 3, latent_channels)
        groups = 8 if latent_channels % 8 == 0 else 1
        self.bottleneck = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, latent_channels),
            nn.SiLU(inplace=True),
        )
        self.up1 = UpBlock(latent_channels, base_channels * 3)
        self.up2 = UpBlock(base_channels * 3, base_channels * 2)
        self.up3 = UpBlock(base_channels * 2, base_channels)
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels // 2, 3, kernel_size=1),
        )

    def _encode_with_skips(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if x.shape[-2:] != self.output_hw:
            raise RuntimeError(f"Expected input spatial size {self.output_hw}, got {tuple(x.shape[-2:])}")
        h0 = self.stem(x)
        h1 = self.down1(h0)
        h2 = self.down2(h1)
        h3 = self.down3(h2)
        latent = self.bottleneck(h3)
        return latent, (h0, h1, h2)

    def _decode(
        self,
        latent: torch.Tensor,
        *,
        skips: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        h0 = h1 = h2 = None
        if skips is not None:
            h0, h1, h2 = skips
        u1 = self.up1(latent)
        if h2 is not None:
            u1 = u1 + h2
        u2 = self.up2(u1)
        if h1 is not None:
            u2 = u2 + h1
        u3 = self.up3(u2)
        if h0 is not None:
            u3 = u3 + h0
        out = self.head(u3)
        final_hw = target_hw or self.output_hw
        if out.shape[-2:] != final_hw:
            out = F.interpolate(out, size=final_hw, mode="bilinear", align_corners=False)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, skips = self._encode_with_skips(x)
        use_skips = self.training
        return self._decode(latent, skips=skips if use_skips else None, target_hw=x.shape[-2:])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        latent, _ = self._encode_with_skips(x)
        return latent

    def decode(
        self,
        latent: torch.Tensor,
        *,
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        return self._decode(latent, skips=None, target_hw=target_hw)


class ResNetAutoencoder(nn.Module):
    """Residual autoencoder without skip connections for baseline comparison."""

    def __init__(self, base_channels: int = 48, latent_channels: int = 128) -> None:
        super().__init__()
        if base_channels <= 0 or latent_channels <= 0:
            raise ValueError("base_channels and latent_channels must be positive.")
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(base_channels), base_channels),
            nn.SiLU(inplace=True),
        )
        self.enc0 = ResidualBlock(base_channels, base_channels)
        self.down1 = ResidualBlock(base_channels, base_channels * 2, stride=2)
        self.down2 = ResidualBlock(base_channels * 2, base_channels * 3, stride=2)
        self.down3 = ResidualBlock(base_channels * 3, latent_channels, stride=2)
        self.bottleneck = ResidualBlock(latent_channels, latent_channels)
        self.up1 = ResidualUpBlock(latent_channels, base_channels * 3)
        self.up2 = ResidualUpBlock(base_channels * 3, base_channels * 2)
        self.up3 = ResidualUpBlock(base_channels * 2, base_channels)
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels // 2, 3, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_hw = x.shape[-2:]
        h = self.stem(x)
        h = self.enc0(h)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.bottleneck(h)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        out = self.head(h)
        if out.shape[-2:] != target_hw:
            out = F.interpolate(out, size=target_hw, mode="bilinear", align_corners=False)
        return out


class ResNetV2Autoencoder(nn.Module):
    """Pre-activation residual autoencoder inspired by ResNet v2."""

    def __init__(self, base_channels: int = 48, latent_channels: int = 128) -> None:
        super().__init__()
        if base_channels <= 0 or latent_channels <= 0:
            raise ValueError("base_channels and latent_channels must be positive.")
        self.stem = nn.Conv2d(3, base_channels, kernel_size=3, padding=1, bias=False)
        self.block0 = PreActResidualBlock(base_channels, base_channels)
        self.down1 = PreActResidualBlock(base_channels, base_channels * 2, stride=2)
        self.down2 = PreActResidualBlock(base_channels * 2, base_channels * 3, stride=2)
        self.down3 = PreActResidualBlock(base_channels * 3, latent_channels, stride=2)
        self.bottleneck = PreActResidualBlock(latent_channels, latent_channels)
        self.up1 = PreActResidualUpBlock(latent_channels, base_channels * 3)
        self.up2 = PreActResidualUpBlock(base_channels * 3, base_channels * 2)
        self.up3 = PreActResidualUpBlock(base_channels * 2, base_channels)
        self.head = nn.Sequential(
            nn.GroupNorm(_norm_groups(base_channels), base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels // 2, 3, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_hw = x.shape[-2:]
        h = self.stem(x)
        h = self.block0(h)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.bottleneck(h)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        out = self.head(h)
        if out.shape[-2:] != target_hw:
            out = F.interpolate(out, size=target_hw, mode="bilinear", align_corners=False)
        return out


class ModernResNetAttnAutoencoder(nn.Module):
    """Modern ResNet + attention blocks forming a skip-free autoencoder."""

    def __init__(
        self,
        base_channels: int = 48,
        latent_channels: int = 128,
        *,
        heads: int = 4,
        dim_head: int = 64,
        groups: int = 32,
    ) -> None:
        super().__init__()
        if base_channels <= 0 or latent_channels <= 0:
            raise ValueError("base_channels and latent_channels must be positive.")
        self.stem = nn.Conv2d(3, base_channels, kernel_size=3, padding=1)
        self.block0 = ModernResNetAttnBlock(
            base_channels,
            base_channels,
            heads=heads,
            dim_head=dim_head,
            groups=groups,
        )
        self.down1 = ModernResNetAttnDown(
            base_channels,
            base_channels * 2,
            heads=heads,
            dim_head=dim_head,
            groups=groups,
        )
        self.down2 = ModernResNetAttnDown(
            base_channels * 2,
            base_channels * 3,
            heads=heads,
            dim_head=dim_head,
            groups=groups,
        )
        self.down3 = ModernResNetAttnDown(
            base_channels * 3,
            latent_channels,
            heads=heads,
            dim_head=dim_head,
            groups=groups,
        )
        self.bottleneck = ModernResNetAttnBlock(
            latent_channels,
            latent_channels,
            heads=heads,
            dim_head=dim_head,
            groups=groups,
        )
        self.up1 = ModernResNetAttnUp(
            latent_channels,
            base_channels * 3,
            heads=heads,
            dim_head=dim_head,
            groups=groups,
        )
        self.up2 = ModernResNetAttnUp(
            base_channels * 3,
            base_channels * 2,
            heads=heads,
            dim_head=dim_head,
            groups=groups,
        )
        self.up3 = ModernResNetAttnUp(
            base_channels * 2,
            base_channels,
            heads=heads,
            dim_head=dim_head,
            groups=groups,
        )
        self.head = nn.Sequential(
            nn.GroupNorm(_group_count(base_channels, groups), base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels // 2, 3, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_hw = x.shape[-2:]
        h = self.stem(x)
        h = self.block0(h)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.bottleneck(h)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        out = self.head(h)
        if out.shape[-2:] != target_hw:
            out = F.interpolate(out, size=target_hw, mode="bilinear", align_corners=False)
        return out


class _Mario4AutoencoderBase(nn.Module):
    """Shared normalisation wrapper for Mario4-derived autoencoders."""

    def __init__(
        self,
        *,
        decoder: nn.Module,
        latent_dim: int = 192,
        encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else Mario4ImageEncoder(latent_dim)
        self.decoder = decoder
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        self.register_buffer("_mean", mean.view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", std.view(1, 3, 1, 1), persistent=False)

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._std.to(dtype=x.dtype, device=x.device) + self._mean.to(
            dtype=x.dtype, device=x.device
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean.to(dtype=x.dtype, device=x.device)) / self._std.to(
            dtype=x.dtype, device=x.device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self._denormalize(x)
        latent = self.encoder(raw)
        recon_raw = self.decoder(latent)
        return self._normalize(recon_raw)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raw = self._denormalize(x)
        return self.encoder(raw)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        recon_raw = self.decoder(latent)
        return self._normalize(recon_raw)


class Mario4Autoencoder(_Mario4AutoencoderBase):
    """Autoencoder built from predict_mario4's baseline decoder."""

    def __init__(self, latent_dim: int = 192, base_channels: int = 32) -> None:
        decoder = Mario4ImageDecoder(latent_dim, base=base_channels)
        super().__init__(decoder=decoder, latent_dim=latent_dim)


class Mario4MirroredAutoencoder(_Mario4AutoencoderBase):
    """Autoencoder using the mirrored ImageDecoder for spatial upsampling."""

    def __init__(self, latent_dim: int = 192, initial_hw: int = 14) -> None:
        decoder = Mario4ImageDecoderMirrored(latent_dim, initial_hw=initial_hw)
        super().__init__(decoder=decoder, latent_dim=latent_dim)


class Mario4ImageDecoderMirroredLarge(Mario4ImageDecoderMirrored):
    """Mario4 mirrored decoder defaulting to a 1024-dimensional latent."""

    def __init__(self, latent_dim: int = 1024, initial_hw: int = 14) -> None:
        super().__init__(latent_dim=latent_dim, initial_hw=initial_hw)


class SpatialSoftmax(nn.Module):
    """Compute per-channel spatial softmax expectations."""

    def __init__(self, channels: int, *, normalize: bool = True) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        self.channels = channels
        self.normalize = normalize

    @property
    def output_dim(self) -> int:
        return self.channels * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W); got {x.shape}.")
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(f"Expected {self.channels} channels, received {c}.")
        flat = x.view(b, c, h * w)
        weights = torch.softmax(flat, dim=-1)
        if self.normalize:
            ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
            xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
        else:
            ys = torch.linspace(0.0, float(h - 1), h, device=x.device, dtype=x.dtype)
            xs = torch.linspace(0.0, float(w - 1), w, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_x = grid_x.reshape(1, 1, -1)
        grid_y = grid_y.reshape(1, 1, -1)
        exp_x = torch.sum(weights * grid_x, dim=-1)
        exp_y = torch.sum(weights * grid_y, dim=-1)
        return torch.cat([exp_x, exp_y], dim=-1)


class Mario4SpatialSoftmaxEncoder(nn.Module):
    """Mario4 encoder variant that applies spatial softmax before projection."""

    def __init__(self, latent_dim: int = 192) -> None:
        super().__init__()
        base = Mario4ImageEncoder(latent_dim)
        self.features = base.features
        feature_dim = base.proj.in_features
        self.spatial_pool = SpatialSoftmax(feature_dim)
        self.proj = nn.Linear(self.spatial_pool.output_dim, latent_dim)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        x = self.features(frame)
        x = self.spatial_pool(x)
        return self.proj(x)


class Mario4SpatialSoftmaxAutoencoder(_Mario4AutoencoderBase):
    """Autoencoder pairing the Mario4 decoder with a spatial-softmax encoder head."""

    def __init__(self, latent_dim: int = 192, initial_hw: int = 14) -> None:
        encoder = Mario4SpatialSoftmaxEncoder(latent_dim)
        decoder = Mario4ImageDecoderMirrored(latent_dim, initial_hw=initial_hw)
        super().__init__(decoder=decoder, latent_dim=latent_dim, encoder=encoder)


class Mario4SpatialSoftmaxLargeAutoencoder(_Mario4AutoencoderBase):
    """Spatial-softmax Mario4 autoencoder variant with a 1024-D latent."""

    def __init__(self, latent_dim: int = 1024, initial_hw: int = 14) -> None:
        encoder = Mario4SpatialSoftmaxEncoder(latent_dim)
        decoder = Mario4ImageDecoderMirroredLarge(latent_dim=latent_dim, initial_hw=initial_hw)
        super().__init__(decoder=decoder, latent_dim=latent_dim, encoder=encoder)


class Mario4LargeAutoencoder(_Mario4AutoencoderBase):
    """Mario4 autoencoder variant with enlarged latent dimensionality."""

    def __init__(self, latent_dim: int = 1024, base_channels: int = 32) -> None:
        decoder = Mario4ImageDecoder(latent_dim, base=base_channels)
        super().__init__(decoder=decoder, latent_dim=latent_dim)


class TextureAwareAutoencoder(nn.Module):
    """Higher-capacity autoencoder tuned for style and patch contrastive training."""

    def __init__(self, base_channels: int = 64, latent_channels: int = 192) -> None:
        super().__init__()
        if base_channels % 8 != 0:
            raise ValueError("base_channels must be divisible by 8 for GroupNorm stability.")
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(inplace=True),
        )  # -> (B, base_channels, 224, 224)
        self.down1 = DownBlock(base_channels, base_channels * 2)      # -> (B, base_channels*2, 112, 112)
        self.down2 = DownBlock(base_channels * 2, base_channels * 3)  # -> (B, base_channels*3, 56, 56)
        self.down3 = DownBlock(base_channels * 3, base_channels * 4)  # -> (B, base_channels*4, 28, 28)
        self.down4 = DownBlock(base_channels * 4, latent_channels)    # -> (B, latent_channels, 14, 14)
        groups = 8 if latent_channels % 8 == 0 else 1
        self.bottleneck = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, latent_channels),
            nn.SiLU(inplace=True),
        )
        self.up1 = UpBlock(latent_channels, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 3)
        self.up3 = UpBlock(base_channels * 3, base_channels * 2)
        self.up4 = UpBlock(base_channels * 2, base_channels)
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels // 2, 3, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.stem(x)
        h1 = self.down1(h0)  # (B, base_channels*2, 112, 112)
        h2 = self.down2(h1)  # (B, base_channels*3, 56, 56)
        h3 = self.down3(h2)  # (B, base_channels*4, 28, 28)
        h4 = self.down4(h3)  # (B, latent_channels, 14, 14)
        b = self.bottleneck(h4)
        u1 = self.up1(b) + h3
        u2 = self.up2(u1) + h2
        u3 = self.up3(u2) + h1
        u4 = self.up4(u3) + h0
        out = self.head(u4)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


class MSSSIMAutoencoder(LightweightAutoencoder):
    """Variant tuned for pure MS-SSIM optimisation."""

    def __init__(self) -> None:
        super().__init__(base_channels=40, latent_channels=120)


class FocalMSSSIMAutoencoder(TextureAwareAutoencoder):
    """Higher-capacity variant for focal MS-SSIM training."""

    def __init__(self) -> None:
        super().__init__(base_channels=72, latent_channels=224)


class FocalL1Loss(nn.Module):
    """Pixel-wise focal weighting applied to an L1 reconstruction objective."""

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
        l1 = torch.abs(input - target)
        norm = l1.detach().mean(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps)
        weight = torch.pow(l1 / norm, self.gamma).clamp(max=self.max_weight)
        loss = weight * l1
        return loss.mean()


def _build_gaussian_window(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = (gauss / gauss.sum()).unsqueeze(0)
    window_2d = (gauss.t() @ gauss).unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def _ssim_components_full(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    *,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tuple[torch.Tensor, torch.Tensor]:
    padding = window.shape[-1] // 2
    mu_x = F.conv2d(x, window, padding=padding, groups=x.shape[1])
    mu_y = F.conv2d(y, window, padding=padding, groups=y.shape[1])
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x_sq = F.conv2d(x * x, window, padding=padding, groups=x.shape[1]) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=padding, groups=y.shape[1]) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=x.shape[1]) - mu_xy
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / denominator
    cs_map = (2 * sigma_xy + c2) / (sigma_x_sq + sigma_y_sq + c2)
    ssim_val = ssim_map.mean(dim=(1, 2, 3))
    cs_val = cs_map.mean(dim=(1, 2, 3))
    return ssim_val, cs_val


def ms_ssim_per_sample(
    x_hat_norm: torch.Tensor,
    x_true_norm: torch.Tensor,
    *,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if weights is None:
        weights = torch.tensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
            device=x_hat_norm.device,
            dtype=x_hat_norm.dtype,
        )
    xh = unnormalize(x_hat_norm)
    xt = unnormalize(x_true_norm)
    window_size = 11
    sigma = 1.5
    channels = xh.shape[1]
    window = _build_gaussian_window(window_size, sigma, channels, xh.device, xh.dtype)
    levels = weights.shape[0]
    mssim: List[torch.Tensor] = []
    mcs: List[torch.Tensor] = []
    x_scaled = xh
    y_scaled = xt
    for _ in range(levels):
        ssim_val, cs_val = _ssim_components_full(x_scaled, y_scaled, window)
        mssim.append(ssim_val)
        mcs.append(cs_val)
        x_scaled = F.avg_pool2d(x_scaled, kernel_size=2, stride=2)
        y_scaled = F.avg_pool2d(y_scaled, kernel_size=2, stride=2)
    mssim_tensor = torch.stack(mssim, dim=0)
    mcs_tensor = torch.stack(mcs[:-1], dim=0)
    eps = torch.finfo(mssim_tensor.dtype).eps
    mssim_tensor = mssim_tensor.clamp(min=eps, max=1.0)
    mcs_tensor = mcs_tensor.clamp(min=eps, max=1.0)
    pow1 = weights[:-1].unsqueeze(1)
    pow2 = weights[-1]
    ms_prod = torch.prod(mcs_tensor ** pow1, dim=0) * (mssim_tensor[-1] ** pow2)
    return ms_prod


class MSSSIMLoss(nn.Module):
    """Wrapper that returns the mean MS-SSIM reconstruction loss."""

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
        scores = ms_ssim_per_sample(input, target)
        losses = 1.0 - scores
        norm = losses.detach().mean().clamp_min(self.eps)
        weight = torch.pow(losses / norm, self.gamma).clamp(max=self.max_weight)
        return (weight * losses).mean()


class CauchyLoss(nn.Module):
    """Robust loss based on the negative log-likelihood of the Cauchy distribution."""

    def __init__(self, sigma: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.sigma = sigma
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = (input - target) / self.sigma
        return torch.log1p(diff.pow(2)).mean().clamp_min(self.eps)


class MultiScalePatchLoss(nn.Module):
    """Aggregate MSE over local patches across a pyramid of spatial scales."""

    def __init__(
        self,
        patch_sizes: Sequence[int] = (7, 11, 15),
        pool_scales: Sequence[int] = (1, 2, 4),
    ) -> None:
        super().__init__()
        if not patch_sizes:
            raise ValueError("patch_sizes must be non-empty.")
        if not pool_scales:
            raise ValueError("pool_scales must be non-empty.")
        if len(patch_sizes) != len(pool_scales):
            raise ValueError("patch_sizes and pool_scales must be the same length.")
        if any(size <= 0 for size in patch_sizes):
            raise ValueError("patch_sizes must contain positive integers.")
        if any(scale <= 0 for scale in pool_scales):
            raise ValueError("pool_scales must contain positive integers.")
        self.patch_sizes = tuple(int(size) for size in patch_sizes)
        self.pool_scales = tuple(int(scale) for scale in pool_scales)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.shape != target.shape:
            raise ValueError("input and target must share the same shape.")
        total_loss = input.new_tensor(0.0)
        total_weight = 0
        for patch_size, pool_scale in zip(self.patch_sizes, self.pool_scales):
            if pool_scale > 1:
                pooled_input = F.avg_pool2d(input, kernel_size=pool_scale, stride=pool_scale)
                pooled_target = F.avg_pool2d(target, kernel_size=pool_scale, stride=pool_scale)
            else:
                pooled_input = input
                pooled_target = target
            k = min(patch_size, pooled_input.shape[-2], pooled_input.shape[-1])
            if k <= 0:
                raise RuntimeError("Patch size became non-positive after clamping.")
            stride = max(1, k // 2)
            padding = k // 2
            unfolded_input = F.unfold(pooled_input, kernel_size=k, stride=stride, padding=padding)
            unfolded_target = F.unfold(pooled_target, kernel_size=k, stride=stride, padding=padding)
            if unfolded_input.shape[-1] == 0:
                patch_loss = F.mse_loss(pooled_input, pooled_target)
            else:
                diff = unfolded_input - unfolded_target
                patch_loss = diff.pow(2).mean()
            total_loss = total_loss + patch_loss
            total_weight += 1
        if total_weight == 0:
            raise RuntimeError("No scales contributed to the loss computation.")
        return total_loss / total_weight


def _compute_shared_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        l1 = F.l1_loss(pred, target).item()
        ms = float(ms_ssim_per_sample(pred, target).mean().item())
    return {"l1": l1, "ms_ssim": ms}


def _gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    b, c, h, w = feat.shape
    feature = feat.view(b, c, h * w)
    gram = torch.bmm(feature, feature.transpose(1, 2))
    return gram / (c * h * w)


def _style_loss(
    pred_feats: Sequence[torch.Tensor],
    target_feats: Sequence[torch.Tensor],
) -> torch.Tensor:
    losses = []
    for pred, target in zip(pred_feats, target_feats):
        gram_pred = _gram_matrix(pred)
        gram_target = _gram_matrix(target)
        losses.append(F.l1_loss(gram_pred, gram_target))
    if not losses:
        raise ValueError("No feature maps provided to style loss.")
    return torch.stack(losses).mean()


def _patch_contrastive_loss(
    pred_feat: torch.Tensor,
    target_feat: torch.Tensor,
    *,
    temperature: float,
    max_patches: int,
) -> torch.Tensor:
    b, c, h, w = pred_feat.shape
    pred_flat = pred_feat.permute(0, 2, 3, 1).reshape(-1, c)
    target_flat = target_feat.permute(0, 2, 3, 1).reshape(-1, c)
    total_patches = pred_flat.shape[0]
    if total_patches == 0:
        raise ValueError("Feature map contains no patches for contrastive loss.")
    if max_patches <= 0:
        raise ValueError("max_patches must be positive.")
    if total_patches <= max_patches:
        indices = torch.arange(total_patches, device=pred_feat.device)
    else:
        indices = torch.randperm(total_patches, device=pred_feat.device)[:max_patches]
    pred_emb = F.normalize(pred_flat[indices], dim=-1)
    target_emb = F.normalize(target_flat[indices], dim=-1)
    logits = pred_emb @ target_emb.t()
    logits = logits / temperature
    labels = torch.arange(pred_emb.shape[0], device=pred_feat.device)
    loss_fw = F.cross_entropy(logits, labels)
    loss_bw = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_fw + loss_bw)


class StyleFeatureExtractor(nn.Module):
    """Frozen VGG16 feature pyramid used for style and contrastive objectives."""

    def __init__(self, layers: Sequence[int]) -> None:
        super().__init__()
        if not layers:
            raise ValueError("At least one layer index must be specified for feature extraction.")
        max_idx = max(layers)
        backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features
        self.layers = nn.ModuleList(backbone[: max_idx + 1])
        # Representative VGG16 feature map sizes (B, C, H, W):
        #   0 conv1_1 -> (B, 64, 224, 224)
        #   1 relu    -> (B, 64, 224, 224)
        #   3 conv1_2 -> (B, 64, 224, 224)
        #   4 relu    -> (B, 64, 224, 224)
        #   5 pool1   -> (B, 64, 112, 112)
        #   8 conv2_2 -> (B, 128,112,112)
        #  15 conv3_3 -> (B, 256,56,56)
        #  22 conv4_3 -> (B, 512,28,28)
        #  29 conv5_3 -> (B, 512,14,14)
        self.selected_layers = set(layers)
        for param in self.layers.parameters():
            param.requires_grad_(False)
        self.eval()

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        features: Dict[int, torch.Tensor] = {}
        out = x
        for idx, layer in enumerate(self.layers):
            out = layer(out)
            if idx in self.selected_layers:
                features[idx] = out
            if len(features) == len(self.selected_layers):
                break
        return features


@torch.no_grad()
def _resnet_encoder(weights: ResNet50_Weights) -> nn.Module:
    model = resnet50(weights=weights)
    layers = list(model.children())[:-2]  # drop avgpool + fc
    # Spatial/channel progression:
    #  input      -> (B, 3,   224, 224)
    #  conv1      -> (B, 64,  112, 112)
    #  maxpool    -> (B, 64,   56,  56)
    #  layer1     -> (B, 256,  56,  56)
    #  layer2     -> (B, 512,  28,  28)
    #  layer3     -> (B, 1024, 14,  14)
    #  layer4     -> (B, 2048,  7,   7)  <-- returned feature map
    encoder = nn.Sequential(*layers)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


@torch.no_grad()
def _convnext_encoder(weights: ConvNeXt_Base_Weights) -> nn.Module:
    model = convnext_base(weights=weights)
    encoder = model.features  # last block yields (B,1024,7,7)
    # Spatial/channel progression of ConvNeXt-Base stages:
    #  input      -> (B,   3, 224, 224)
    #  stem       -> (B, 128, 112, 112)
    #  stage1     -> (B, 256, 112, 112)
    #  stage2     -> (B, 512,  56,  56)
    #  stage3     -> (B,1024,  28,  28)
    #  stage4     -> (B,1024,   7,   7)  <-- returned feature map
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


# ---------------------------------------------------------------------------
# Trainer wrapper
# ---------------------------------------------------------------------------


class ReconstructionTrainer:
    """Wraps a frozen encoder and trainable decoder with a unified step API."""

    def __init__(
        self,
        name: str,
        encoder: nn.Module,
        decoder: nn.Module,
        *,
        device: torch.device,
        lr: float,
    ) -> None:
        self.name = name
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.history: List[Tuple[int, float]] = []
        self.shared_history: List[Tuple[int, Dict[str, float]]] = []
        self.global_step = 0
        self.best_loss: Optional[float] = None

    def step(self, batch: torch.Tensor) -> Tuple[float, bool, Dict[str, float]]:
        self.decoder.train()
        with torch.no_grad():
            feats = self.encoder(batch)
        recon = self.decoder(feats)
        loss = self.loss_fn(recon, batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.global_step += 1
        loss_val = float(loss.detach().item())
        self.history.append((self.global_step, loss_val))
        metrics = _compute_shared_metrics(recon.detach(), batch)
        self.shared_history.append((self.global_step, metrics))
        improved = self.best_loss is None or loss_val < self.best_loss
        if improved:
            self.best_loss = loss_val
        return loss_val, improved, metrics

    @torch.no_grad()
    def reconstruct(self, batch: torch.Tensor) -> torch.Tensor:
        was_training = self.decoder.training
        self.encoder.eval()
        self.decoder.eval()
        feats = self.encoder(batch.to(self.device))
        recon = self.decoder(feats)
        if was_training:
            self.decoder.train()
        return recon.cpu()

    def state_dict(self) -> dict:
        return {
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "history": self.history,
            "shared_history": self.shared_history,
            "global_step": self.global_step,
            "name": self.name,
            "best_loss": self.best_loss,
        }

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_state_dict(self, state: dict, *, lr: Optional[float] = None) -> None:
        self.decoder.load_state_dict(state["decoder"])
        self.optimizer.load_state_dict(state["optimizer"])
        if lr is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = lr
        self.history = state.get("history", [])
        self.shared_history = state.get("shared_history", [])
        self.global_step = state.get("global_step", 0)
        self.best_loss = state.get("best_loss")


class AutoencoderTrainer:
    """Trainable encoder/decoder pair driven by focal L1 loss."""

    def __init__(
        self,
        name: str,
        model: nn.Module,
        *,
        device: torch.device,
        lr: float,
        weight_decay: float = 1e-4,
        loss_fn: Optional[nn.Module] = None,
    ) -> None:
        self.name = name
        self.device = device
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.loss_fn = loss_fn or FocalL1Loss()
        self.history: List[Tuple[int, float]] = []
        self.shared_history: List[Tuple[int, Dict[str, float]]] = []
        self.global_step = 0
        self.best_loss: Optional[float] = None

    def step(self, batch: torch.Tensor) -> Tuple[float, bool, Dict[str, float]]:
        self.model.train()
        recon = self.model(batch)
        loss = self.loss_fn(recon, batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.global_step += 1
        loss_val = float(loss.detach().item())
        self.history.append((self.global_step, loss_val))
        metrics = _compute_shared_metrics(recon.detach(), batch)
        self.shared_history.append((self.global_step, metrics))
        improved = self.best_loss is None or loss_val < self.best_loss
        if improved:
            self.best_loss = loss_val
        return loss_val, improved, metrics

    @torch.no_grad()
    def reconstruct(self, batch: torch.Tensor) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        recon = self.model(batch.to(self.device))
        if was_training:
            self.model.train()
        return recon.cpu()

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "history": self.history,
            "shared_history": self.shared_history,
            "global_step": self.global_step,
            "name": self.name,
            "best_loss": self.best_loss,
        }

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_state_dict(self, state: dict, *, lr: Optional[float] = None) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if lr is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = lr
        self.history = state.get("history", [])
        self.shared_history = state.get("shared_history", [])
        self.global_step = state.get("global_step", 0)
        self.best_loss = state.get("best_loss")


class StyleContrastTrainer:
    """Autoencoder trained with Gram style loss and patchwise contrastive loss."""

    def __init__(
        self,
        name: str,
        model: nn.Module,
        feature_extractor: StyleFeatureExtractor,
        *,
        device: torch.device,
        lr: float,
        style_layers: Sequence[int],
        patch_layer: int,
        style_weight: float,
        contrast_weight: float,
        contrast_temperature: float,
        contrast_patches: int,
        reconstruction_weight: float = 0.0,
        reconstruction_loss: Optional[nn.Module] = None,
    ) -> None:
        self.name = name
        self.device = device
        self.model = model.to(device)
        self.feature_extractor = feature_extractor.to(device)
        self.style_layers = list(style_layers)
        self.patch_layer = patch_layer
        missing = set(self.style_layers + [self.patch_layer]) - feature_extractor.selected_layers
        if missing:
            raise ValueError(f"Feature extractor does not provide layers: {sorted(missing)}")
        self.feature_extractor.eval()
        self.style_weight = style_weight
        self.contrast_weight = contrast_weight
        self.contrast_temperature = contrast_temperature
        self.contrast_patches = contrast_patches
        if self.contrast_temperature <= 0:
            raise ValueError("contrast_temperature must be positive.")
        if self.contrast_patches <= 0:
            raise ValueError("contrast_patches must be positive.")
        self.reconstruction_weight = reconstruction_weight
        self.reconstruction_loss = reconstruction_loss or nn.L1Loss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.history: List[Tuple[int, float]] = []
        self.shared_history: List[Tuple[int, Dict[str, float]]] = []
        self.global_step = 0
        self.best_loss: Optional[float] = None

    def step(self, batch: torch.Tensor) -> Tuple[float, bool, Dict[str, float]]:
        self.model.train()
        recon = self.model(batch)
        pred_feats = self.feature_extractor(recon)
        with torch.no_grad():
            target_feats = self.feature_extractor(batch)
        style_pred = [pred_feats[idx] for idx in self.style_layers]
        style_target = [target_feats[idx] for idx in self.style_layers]
        style_loss = _style_loss(style_pred, style_target) * self.style_weight
        contrast_loss = _patch_contrastive_loss(
            pred_feats[self.patch_layer],
            target_feats[self.patch_layer],
            temperature=self.contrast_temperature,
            max_patches=self.contrast_patches,
        ) * self.contrast_weight
        total_loss = style_loss + contrast_loss
        if self.reconstruction_weight > 0.0:
            recon_loss = self.reconstruction_loss(recon, batch) * self.reconstruction_weight
            total_loss = total_loss + recon_loss
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()
        self.global_step += 1
        loss_val = float(total_loss.detach().item())
        self.history.append((self.global_step, loss_val))
        metrics = _compute_shared_metrics(recon.detach(), batch)
        self.shared_history.append((self.global_step, metrics))
        improved = self.best_loss is None or loss_val < self.best_loss
        if improved:
            self.best_loss = loss_val
        return loss_val, improved, metrics

    @torch.no_grad()
    def reconstruct(self, batch: torch.Tensor) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        recon = self.model(batch.to(self.device))
        if was_training:
            self.model.train()
        return recon.cpu()

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "history": self.history,
            "shared_history": self.shared_history,
            "global_step": self.global_step,
            "name": self.name,
            "best_loss": self.best_loss,
        }

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_state_dict(self, state: dict, *, lr: Optional[float] = None) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if lr is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = lr
        self.history = state.get("history", [])
        self.shared_history = state.get("shared_history", [])
        self.global_step = state.get("global_step", 0)
        self.best_loss = state.get("best_loss")


Trainer = ReconstructionTrainer | AutoencoderTrainer | StyleContrastTrainer


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def _tensor_to_numpy(img: torch.Tensor) -> torch.Tensor:
    return img.permute(1, 2, 0).clamp(0.0, 1.0).cpu()


def save_recon_grid(
    inputs: torch.Tensor,
    reconstructions: Sequence[Tuple[str, torch.Tensor]],
    *,
    out_path: Path,
) -> None:
    rows = inputs.shape[0]
    cols = 1 + len(reconstructions)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1:
        axes = axes[None, :]
    unnorm_inputs = unnormalize(inputs)
    unnorm_recons = [(name, unnormalize(tensor)) for name, tensor in reconstructions]
    col_titles = ["Input"] + [_display_name(name) for name, _ in unnorm_recons]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title)
    for row in range(rows):
        axes[row, 0].imshow(_tensor_to_numpy(unnorm_inputs[row]))
        axes[row, 0].axis("off")
        for col, (_, tensor) in enumerate(unnorm_recons, start=1):
            axes[row, col].imshow(_tensor_to_numpy(tensor[row]))
            axes[row, col].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loss_histories(trainers: Sequence[Trainer], out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for trainer in trainers:
        if not trainer.history:
            continue
        steps, losses = zip(*trainer.history)
        plt.plot(steps, losses, label=_display_name(trainer.name))
    plt.xlabel("Step")
    plt.ylabel("Reconstruction loss")
    plt.title("Model comparison losses (log scale)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_loss_histories(trainers: Sequence[Trainer], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for trainer in trainers:
        history_path = out_dir / f"{trainer.name}_loss.csv"
        with history_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss"])
            for step, loss in trainer.history:
                writer.writerow([step, loss])


def write_shared_metric_histories(trainers: Sequence[Trainer], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for trainer in trainers:
        history = getattr(trainer, "shared_history", [])
        if not history:
            continue
        history_path = out_dir / f"{trainer.name}_shared_metrics.csv"
        with history_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "l1", "ms_ssim"])
            for step, metrics in history:
                writer.writerow(
                    [
                        step,
                        metrics.get("l1"),
                        metrics.get("ms_ssim"),
                    ]
                )


def plot_shared_metric_histories(trainers: Sequence[Trainer], out_dir: Path) -> None:
    has_data = any(getattr(trainer, "shared_history", []) for trainer in trainers)
    if not has_data:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot L1
    plt.figure(figsize=(8, 5))
    for trainer in trainers:
        history = getattr(trainer, "shared_history", [])
        if not history:
            continue
        steps = [item[0] for item in history]
        l1_values = [item[1]["l1"] for item in history]
        plt.plot(steps, l1_values, label=_display_name(trainer.name))
    plt.xlabel("Step")
    plt.ylabel("L1 (shared metric)")
    plt.title("Shared L1 metric (log scale)")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out_dir / "shared_l1.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot MS-SSIM
    plt.figure(figsize=(8, 5))
    for trainer in trainers:
        history = getattr(trainer, "shared_history", [])
        if not history:
            continue
        steps = [item[0] for item in history]
        ms_values = [item[1]["ms_ssim"] for item in history]
        plt.plot(steps, ms_values, label=_display_name(trainer.name))
    plt.xlabel("Step")
    plt.ylabel("MS-SSIM (shared metric)")
    plt.title("Shared MS-SSIM metric")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out_dir / "shared_ms_ssim.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class Config:
    traj_root: Path = Path("data.image_distance.train_levels_1_2")
    out_dir: Path = Path("out.reconstruct_mario_comparison")
    max_trajs: Optional[int] = None
    batch_size: int = 16
    num_workers: int = 0
    train_steps: int = 10_000
    log_every: int = 10
    vis_every: int = 50
    vis_rows: int = 6
    lr: float = 1e-4
    device: Optional[str] = None
    seed: int = 0
    resume_dir: Optional[Path] = None
    resume_tag: str = "last"
    style_weight: float = 1.0
    contrast_weight: float = 1.0
    contrast_temperature: float = 0.07
    contrast_patches: int = 256
    reconstruction_weight: float = 0.0
    style_layers: Tuple[int, ...] = (3, 8, 15)
    patch_layer: int = 22
    enable_resnet50: bool = False
    enable_convnext_base: bool = False
    enable_l1_autoencoder: bool = False
    enable_smoothl1_autoencoder: bool = False
    enable_mse_autoencoder: bool = False
    enable_focal_autoencoder: bool = False
    enable_style_contrast_autoencoder: bool = False
    enable_cauchy_autoencoder: bool = False
    enable_msssim_autoencoder: bool = False
    enable_focal_msssim_autoencoder: bool = False
    enable_no_skip_autoencoder: bool = False
    enable_no_skip_patch_autoencoder: bool = False
    enable_skip_train_autoencoder: bool = False
    enable_resnet_autoencoder: bool = False
    enable_resnetv2_autoencoder: bool = False
    enable_modern_attn_autoencoder: bool = False
    enable_mario4_autoencoder: bool = False
    enable_mario4_mirrored_autoencoder: bool = False
    enable_mario4_spatial_autoencoder: bool = False
    enable_mario4_large_autoencoder: bool = False
    enable_mario4_spatial_large_autoencoder: bool = False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_trainers(cfg: Config, device: torch.device) -> List[Trainer]:
    trainers: List[Trainer] = []
    if cfg.enable_resnet50:
        resnet_enc = _resnet_encoder(ResNet50_Weights.IMAGENET1K_V2)
        resnet_dec = Decoder(2048)
        trainers.append(
            ReconstructionTrainer("resnet50", resnet_enc, resnet_dec, device=device, lr=cfg.lr)
        )
    if cfg.enable_convnext_base:
        convnext_enc = _convnext_encoder(ConvNeXt_Base_Weights.IMAGENET1K_V1)
        convnext_dec = Decoder(1024)
        trainers.append(
            ReconstructionTrainer(
                "convnext_base", convnext_enc, convnext_dec, device=device, lr=cfg.lr
            )
        )
    if cfg.enable_mse_autoencoder:
        mse_model = LightweightAutoencoder()
        trainers.append(
            AutoencoderTrainer(
                "mse_autoencoder",
                mse_model,
                device=device,
                lr=cfg.lr,
                loss_fn=nn.MSELoss(),
            )
        )
    if cfg.enable_l1_autoencoder:
        l1_model = LightweightAutoencoder()
        trainers.append(
            AutoencoderTrainer(
                "l1_autoencoder",
                l1_model,
                device=device,
                lr=cfg.lr,
                loss_fn=nn.L1Loss(),
            )
        )
    if cfg.enable_smoothl1_autoencoder:
        smooth_model = LightweightAutoencoder()
        trainers.append(
            AutoencoderTrainer(
                "smoothl1_autoencoder",
                smooth_model,
                device=device,
                lr=cfg.lr,
                loss_fn=nn.SmoothL1Loss(),
            )
        )
    if cfg.enable_focal_autoencoder:
        autoencoder = LightweightAutoencoder()
        trainers.append(AutoencoderTrainer("focal_autoencoder", autoencoder, device=device, lr=cfg.lr))
    if cfg.enable_style_contrast_autoencoder:
        texture_autoencoder = TextureAwareAutoencoder()
        feature_layers = sorted(set(cfg.style_layers + (cfg.patch_layer,)))
        feature_extractor = StyleFeatureExtractor(feature_layers)
        trainers.append(
            StyleContrastTrainer(
                "style_contrast_autoencoder",
                texture_autoencoder,
                feature_extractor,
                device=device,
                lr=cfg.lr,
                style_layers=cfg.style_layers,
                patch_layer=cfg.patch_layer,
                style_weight=cfg.style_weight,
                contrast_weight=cfg.contrast_weight,
                contrast_temperature=cfg.contrast_temperature,
                contrast_patches=cfg.contrast_patches,
                reconstruction_weight=cfg.reconstruction_weight,
            )
        )
    if cfg.enable_cauchy_autoencoder:
        cauchy_model = LightweightAutoencoder()
        trainers.append(
            AutoencoderTrainer(
                "cauchy_autoencoder",
                cauchy_model,
                device=device,
                lr=cfg.lr,
                loss_fn=CauchyLoss(),
            )
        )
    if cfg.enable_no_skip_autoencoder:
        no_skip_model = LightweightAutoencoderNoSkip()
        trainers.append(
            AutoencoderTrainer(
                "no_skip_autoencoder",
                no_skip_model,
                device=device,
                lr=cfg.lr,
            )
        )
    if cfg.enable_no_skip_patch_autoencoder:
        no_skip_patch_model = LightweightAutoencoderNoSkipPatch()
        trainers.append(
            AutoencoderTrainer(
                "no_skip_patch_autoencoder",
                no_skip_patch_model,
                device=device,
                lr=cfg.lr,
                loss_fn=MultiScalePatchLoss(),
            )
        )
    if cfg.enable_skip_train_autoencoder:
        skip_train_model = LightweightAutoencoderSkipTrain()
        trainers.append(
            AutoencoderTrainer(
                "skip_train_autoencoder",
                skip_train_model,
                device=device,
                lr=cfg.lr,
            )
        )
    if cfg.enable_resnet_autoencoder:
        resnet_model = ResNetAutoencoder()
        trainers.append(
            AutoencoderTrainer(
                "resnet_autoencoder",
                resnet_model,
                device=device,
                lr=cfg.lr,
                loss_fn=nn.SmoothL1Loss(),
            )
        )
    if cfg.enable_resnetv2_autoencoder:
        resnet_v2_model = ResNetV2Autoencoder()
        trainers.append(
            AutoencoderTrainer(
                "resnetv2_autoencoder",
                resnet_v2_model,
                device=device,
                lr=cfg.lr,
                loss_fn=nn.SmoothL1Loss(),
            )
        )
    if cfg.enable_modern_attn_autoencoder:
        modern_attn_model = ModernResNetAttnAutoencoder()
        trainers.append(
            AutoencoderTrainer(
                "modern_attn_autoencoder",
                modern_attn_model,
                device=device,
                lr=cfg.lr,
                loss_fn=nn.SmoothL1Loss(),
            )
        )
    if cfg.enable_mario4_autoencoder:
        mario4_model = Mario4Autoencoder()
        trainers.append(
            AutoencoderTrainer(
                "mario4_autoencoder",
                mario4_model,
                device=device,
                lr=cfg.lr,
                loss_fn=nn.SmoothL1Loss(),
            )
        )
    if cfg.enable_mario4_mirrored_autoencoder:
        mario4_mirror_model = Mario4MirroredAutoencoder()
        trainers.append(
            AutoencoderTrainer(
                "mario4_mirrored_autoencoder",
                mario4_mirror_model,
                device=device,
                lr=cfg.lr,
                loss_fn=nn.SmoothL1Loss(),
            )
        )
    if cfg.enable_mario4_spatial_autoencoder:
        mario4_spatial_model = Mario4SpatialSoftmaxAutoencoder()
        trainers.append(
            AutoencoderTrainer(
                "mario4_spatial_autoencoder",
                mario4_spatial_model,
                device=device,
                lr=cfg.lr,
                loss_fn=nn.SmoothL1Loss(),
            )
        )
    if cfg.enable_mario4_large_autoencoder:
        mario4_large_model = Mario4LargeAutoencoder()
        trainers.append(
            AutoencoderTrainer(
                "mario4_large_autoencoder",
                mario4_large_model,
                device=device,
                lr=cfg.lr,
                loss_fn=nn.SmoothL1Loss(),
            )
        )
    if cfg.enable_mario4_spatial_large_autoencoder:
        mario4_spatial_large_model = Mario4SpatialSoftmaxLargeAutoencoder()
        trainers.append(
            AutoencoderTrainer(
                "mario4_spatial_large_autoencoder",
                mario4_spatial_large_model,
                device=device,
                lr=cfg.lr,
                loss_fn=nn.SmoothL1Loss(),
            )
        )
    if cfg.enable_msssim_autoencoder:
        msssim_model = MSSSIMAutoencoder()
        trainers.append(
            AutoencoderTrainer(
                "msssim_autoencoder",
                msssim_model,
                device=device,
                lr=cfg.lr,
                loss_fn=MSSSIMLoss(),
            )
        )
    if cfg.enable_focal_msssim_autoencoder:
        focal_msssim_model = FocalMSSSIMAutoencoder()
        trainers.append(
            AutoencoderTrainer(
                "focal_msssim_autoencoder",
                focal_msssim_model,
                device=device,
                lr=cfg.lr,
                loss_fn=FocalMSSSIMLoss(),
            )
        )
    if not trainers:
        raise ValueError("No trainers enabled. Enable at least one model to proceed.")
    return trainers


def main() -> None:
    cfg = tyro.cli(Config)
    if cfg.vis_rows <= 0:
        raise ValueError("vis_rows must be positive.")
    if cfg.vis_every <= 0:
        raise ValueError("vis_every must be positive.")
    if cfg.resume_tag not in {"last", "best", "final"}:
        raise ValueError("resume_tag must be one of {'last', 'best', 'final'}.")

    logger = _get_logger()
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    dataset = MarioFrameDataset(Path(cfg.traj_root), max_trajs=cfg.max_trajs)

    if cfg.resume_dir is not None:
        run_dir = cfg.resume_dir
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {run_dir}")
    else:
        run_dir = cfg.out_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_root = run_dir / "checkpoints"
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    samples_root = run_dir / "samples"
    fixed_samples_dir = samples_root / "fixed"
    rolling_samples_dir = samples_root / "rolling"
    fixed_samples_dir.mkdir(parents=True, exist_ok=True)
    rolling_samples_dir.mkdir(parents=True, exist_ok=True)

    trainers = build_trainers(cfg, device)
    logger.info("Parameter summary:")
    for trainer in trainers:
        if isinstance(trainer, ReconstructionTrainer):
            module = trainer.decoder
        elif isinstance(trainer, AutoencoderTrainer):
            module = trainer.model
        elif isinstance(trainer, StyleContrastTrainer):
            module = trainer.model
        else:
            continue
        _summarize_parameters(trainer.name, module, logger=logger)
    checkpoint_paths: dict[str, dict[str, Path]] = {}

    for trainer in trainers:
        trainer_dir = checkpoints_root / trainer.name
        trainer_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_paths[trainer.name] = {
            "last": trainer_dir / "last.pt",
            "best": trainer_dir / "best.pt",
            "final": trainer_dir / "final.pt",
        }
        if cfg.resume_dir is not None:
            resume_path = checkpoint_paths[trainer.name][cfg.resume_tag]
            if not resume_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint for {trainer.name!r} not found at {resume_path}"
                )
            state = torch.load(resume_path, map_location=device)
            trainer.load_state_dict(state, lr=cfg.lr)

    if cfg.resume_dir is not None:
        step_set = {trainer.global_step for trainer in trainers}
        if len(step_set) != 1:
            raise RuntimeError("Loaded checkpoints have mismatched global steps.")
        start_step = step_set.pop()
    else:
        start_step = 0

    vis_paths_file = run_dir / "vis_paths.txt"
    if vis_paths_file.exists():
        vis_paths = [
            line.strip() for line in vis_paths_file.read_text().splitlines() if line.strip()
        ]
    else:
        vis_count = min(cfg.vis_rows, len(dataset))
        if vis_count <= 0:
            raise RuntimeError("Not enough frames available for visualisation.")
        indices = random.sample(range(len(dataset)), vis_count)
        vis_paths = [str(dataset.paths[idx]) for idx in indices]
        vis_paths_file.write_text("\n".join(vis_paths) + "\n")
    vis_batch = load_image_batch(vis_paths, dataset.transform)
    vis_batch_device = vis_batch.to(device)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    target_step = start_step + cfg.train_steps
    if cfg.train_steps > 0:
        data_iterator = iter(loader)
        for current_step in range(start_step + 1, target_step + 1):
            try:
                batch, _ = next(data_iterator)
            except StopIteration:
                data_iterator = iter(loader)
                batch, _ = next(data_iterator)
            batch = batch.to(device, non_blocking=True)
            losses: dict[str, float] = {}
            shared_metrics_step: dict[str, Dict[str, float]] = {}
            timing: dict[str, float] = {}
            for trainer in trainers:
                step_start = time.perf_counter()
                loss, improved, shared = trainer.step(batch)
                timing[trainer.name] = time.perf_counter() - step_start
                losses[trainer.name] = loss
                shared_metrics_step[trainer.name] = shared
                trainer.save_checkpoint(checkpoint_paths[trainer.name]["last"])
                if improved:
                    trainer.save_checkpoint(checkpoint_paths[trainer.name]["best"])
            if cfg.log_every > 0 and current_step % cfg.log_every == 0:
                loss_str = ", ".join(f"{name}: {losses[name]:.4f}" for name in losses)
                metric_str = ", ".join(
                    f"{name}: L1 {shared_metrics_step[name]['l1']:.4f}, "
                    f"MS {shared_metrics_step[name]['ms_ssim']:.4f}"
                    for name in shared_metrics_step
                )
                timing_str = ", ".join(
                    f"{_display_name(name)}: {timing[name]*1000:.1f}ms" for name in timing
                )
                logger.info(
                    "[step %05d] %s | Shared %s",
                    current_step,
                    loss_str,
                    metric_str,
                )
                logger.info("[step %05d] Timing %s", current_step, timing_str)
                plot_loss_histories(trainers, metrics_dir / "decoder_losses.png")
                write_loss_histories(trainers, metrics_dir)
                write_shared_metric_histories(trainers, metrics_dir)
                plot_shared_metric_histories(trainers, metrics_dir)
            if current_step % cfg.vis_every == 0 or current_step == target_step:
                step_tag = f"step_{current_step:05d}"
                fixed_recons = [
                    (trainer.name, trainer.reconstruct(vis_batch_device)) for trainer in trainers
                ]
                save_recon_grid(
                    vis_batch,
                    fixed_recons,
                    out_path=fixed_samples_dir / f"{step_tag}.png",
                )
                rolling_batch = sample_random_batch(dataset, cfg.vis_rows)
                rolling_batch_device = rolling_batch.to(device)
                rolling_recons = [
                    (trainer.name, trainer.reconstruct(rolling_batch_device)) for trainer in trainers
                ]
                save_recon_grid(
                    rolling_batch,
                    rolling_recons,
                    out_path=rolling_samples_dir / f"{step_tag}.png",
                )
    else:
        logger.info("train_steps is 0; skipping decoder optimisation.")
        step_tag = f"step_{target_step:05d}"
        fixed_recons = [
            (trainer.name, trainer.reconstruct(vis_batch_device)) for trainer in trainers
        ]
        save_recon_grid(
            vis_batch,
            fixed_recons,
            out_path=fixed_samples_dir / f"{step_tag}.png",
        )
        rolling_batch = sample_random_batch(dataset, cfg.vis_rows)
        rolling_batch_device = rolling_batch.to(device)
        rolling_recons = [
            (trainer.name, trainer.reconstruct(rolling_batch_device)) for trainer in trainers
        ]
        save_recon_grid(
            rolling_batch,
            rolling_recons,
            out_path=rolling_samples_dir / f"{step_tag}.png",
        )

    plot_loss_histories(trainers, metrics_dir / "decoder_losses.png")
    write_loss_histories(trainers, metrics_dir)
    write_shared_metric_histories(trainers, metrics_dir)
    plot_shared_metric_histories(trainers, metrics_dir)
    for trainer in trainers:
        paths = checkpoint_paths[trainer.name]
        trainer.save_checkpoint(paths["last"])
        if trainer.best_loss is not None and not paths["best"].exists():
            trainer.save_checkpoint(paths["best"])
        trainer.save_checkpoint(paths["final"])


if __name__ == "__main__":
    main()
