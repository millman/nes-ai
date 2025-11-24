#!/usr/bin/env python3
"""Motion-aware multi-frame reconstruction for NES Mario trajectories.

This script explores whether emphasising moving sprites during training
improves reconstruction fidelity. Key ideas:

  • the encoder observes 4 sequential frames and distils them into a single
    latent vector regularised with Conditional Flow Matching (CFM)
  • the network predicts per-pixel motion saliency for both "what moved" and
    "what is likely to move", enabling motion-aware reconstruction weighting
  • reconstruction loss up-weights errors on pixels that actually moved or are
    predicted to move, encouraging the decoder to focus on dynamic objects
  • inference can operate on a single frame by internally repeating it to form
    a temporal context, while still producing motion saliency outputs

Outputs include reconstruction comparisons, motion saliency overlays, PCA
traversals from the latent space, and training curves analogous to the CFM
baseline script.
"""
from __future__ import annotations

import csv
import logging
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torchvision.transforms as T
import tyro

from predict_mario_ms_ssim import (
    INV_MEAN,
    INV_STD,
    default_transform,
    ms_ssim_loss,
    pick_device,
    unnormalize,
)
from trajectory_utils import list_state_frames, list_traj_dirs

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


def _raw_transform() -> T.Compose:
    return T.Compose(
        [
            T.ToTensor(),  # returns float in [0, 1]
        ]
    )


class MarioMotionDataset(Dataset):
    """Returns 4-frame sequences with motion saliency heuristics.

    Each item contains:
      • frames_4: Tensor with shape (4, 3, H, W) in ImageNet-normalised space
      • target: final frame (3, H, W) to reconstruct
      • motion_target: soft mask (1, H, W) highlighting pixels that moved
      • last_path: path to the final frame (string)
    """

    def __init__(
        self,
        root_dir: str,
        *,
        sequence_len: int = 4,
        transform: Optional[T.Compose] = None,
        motion_smoothing: int = 3,
        motion_scale: float = 6.0,
        max_trajs: Optional[int] = None,
    ) -> None:
        if sequence_len < 2:
            raise ValueError("sequence_len must be ≥ 2")
        self.sequence_len = sequence_len
        self.transform = transform or default_transform()
        self.raw_transform = _raw_transform()
        self.motion_smoothing = motion_smoothing
        self.motion_scale = motion_scale
        self.index: List[Tuple[List[Path], int]] = []

        traj_count = 0
        for traj_dir in list_traj_dirs(Path(root_dir)):
            if not traj_dir.is_dir():
                continue
            states_dir = traj_dir / "states"
            if not states_dir.is_dir():
                continue
            frames = list_state_frames(states_dir)
            if len(frames) < sequence_len:
                continue
            for start in range(len(frames) - sequence_len + 1):
                self.index.append((frames, start))
            traj_count += 1
            if max_trajs is not None and traj_count >= max_trajs:
                break
        if not self.index:
            raise RuntimeError(f"No sequential frames found under {root_dir}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        files, start = self.index[idx]
        seq_raw: List[torch.Tensor] = []
        seq_norm: List[torch.Tensor] = []
        for offset in range(self.sequence_len):
            with Image.open(files[start + offset]).convert("RGB") as img:
                raw = self.raw_transform(img)
                norm = self.transform(img)
            seq_raw.append(raw)
            seq_norm.append(norm)

        raw_stack = torch.stack(seq_raw, dim=0)  # (T, 3, H, W)
        norm_stack = torch.stack(seq_norm, dim=0)
        target = norm_stack[-1]

        motion = self._compute_motion_mask(raw_stack)
        target_hw = target.shape[-2:]
        if motion.shape[-2:] != target_hw:
            motion = F.interpolate(
                motion.unsqueeze(0),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            motion = motion.clamp(0.0, 1.0)
        last_path = str(files[start + self.sequence_len - 1])
        return norm_stack, target, motion, last_path

    def _compute_motion_mask(self, raw_stack: torch.Tensor) -> torch.Tensor:
        """Heuristic saliency map: emphasise max channel change across frames."""
        diffs = raw_stack[1:] - raw_stack[:-1]  # (T-1, 3, H, W)
        motion = diffs.abs().mean(dim=1)  # (T-1, H, W)
        motion = motion.max(dim=0).values  # (H, W)
        motion = motion.unsqueeze(0)  # (1, H, W)
        if self.motion_smoothing > 1:
            motion = F.avg_pool2d(
                motion,
                kernel_size=self.motion_smoothing,
                stride=1,
                padding=self.motion_smoothing // 2,
            )
        motion = motion * self.motion_scale
        motion = motion.clamp(0.0, 1.0)
        return motion


# -----------------------------------------------------------------------------
# Model components
# -----------------------------------------------------------------------------


def _make_group_norm(num_channels: int) -> nn.GroupNorm:
    groups = 8 if num_channels % 8 == 0 else 1
    return nn.GroupNorm(groups, num_channels)


class FeedForward(nn.Module):
    """Two-layer MLP with SiLU activation."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalSelfAttention(nn.Module):
    """Per-pixel temporal attention across the input frame sequence."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ff = FeedForward(channels)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: (B, T, C, H, W)
        Returns:
            Tensor with same shape incorporating temporal attention.
        """
        B, T, C, H, W = feats.shape
        tokens = feats.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        residual = tokens
        tokens_norm = self.norm1(tokens)
        attn_out, _ = self.attn(tokens_norm, tokens_norm, tokens_norm, need_weights=False)
        tokens = residual + attn_out
        tokens = tokens + self.ff(self.norm2(tokens))
        tokens = tokens.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2)
        return tokens


class SpatialSelfAttention(nn.Module):
    """Attention over spatial positions of a feature map."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ff = FeedForward(channels)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, C, H, W)
        Returns:
            Tensor with same shape after spatial attention.
        """
        B, C, H, W = feat.shape
        tokens = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        residual = tokens
        tokens_norm = self.norm1(tokens)
        attn_out, _ = self.attn(tokens_norm, tokens_norm, tokens_norm, need_weights=False)
        tokens = residual + attn_out
        tokens = tokens + self.ff(self.norm2(tokens))
        tokens = tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return tokens


class CrossAttentionBlock(nn.Module):
    """Cross-attention where queries attend to external key/value tokens."""

    def __init__(self, channels: int, num_heads: int = 8) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.norm_ff = nn.LayerNorm(channels)
        self.ff = FeedForward(channels)

    def forward(self, queries: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: (B, N, C)
            kv: (B, M, C)
        Returns:
            (B, N, C) enriched with cross-attention context.
        """
        q_norm = self.norm_q(queries)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.attn(q_norm, kv_norm, kv_norm, need_weights=False)
        out = queries + attn_out
        out = out + self.ff(self.norm_ff(out))
        return out


class ConvBlock(nn.Module):
    """Conv-Norm-Activation block."""

    def __init__(self, c_in: int, c_out: int, *, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1),
            _make_group_norm(c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    """Residual block with optional down-sampling."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1)
        self.norm1 = _make_group_norm(c_out)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1)
        self.norm2 = _make_group_norm(c_out)
        if stride != 1 or c_in != c_out:
            self.skip = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride),
                _make_group_norm(c_out),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.skip(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return self.act(out + shortcut)


class FrameBackbone(nn.Module):
    """Shared CNN backbone that encodes a single frame to a spatial map."""

    def __init__(
        self,
        *,
        base_channels: int = 64,
        input_hw: Tuple[int, int],
    ) -> None:
        super().__init__()
        if input_hw[0] % 16 != 0 or input_hw[1] % 16 != 0:
            raise ValueError("input_hw must be divisible by 16 in both dimensions.")
        if base_channels % 2 != 0:
            raise ValueError("base_channels must be even.")
        self.input_hw = input_hw

        stem_width = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_width, kernel_size=3, stride=1, padding=1),
            _make_group_norm(stem_width),
            nn.SiLU(inplace=True),
        )
        stage_channels = [
            stem_width,
            stem_width * 3 // 2,
            stem_width * 2,
            stem_width * 5 // 2,
        ]
        stages: List[nn.Module] = []
        in_ch = stem_width
        for out_ch in stage_channels:
            stages.append(
                nn.Sequential(
                    ResidualBlock(in_ch, out_ch, stride=2),
                    ResidualBlock(out_ch, out_ch, stride=1),
                )
            )
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)
        self.post = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            _make_group_norm(in_ch),
            nn.SiLU(inplace=True),
        )
        self.out_channels = in_ch
        self.latent_hw = (input_hw[0] // 16, input_hw[1] // 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != self.input_hw:
            raise RuntimeError(
                f"Expected frame spatial size {self.input_hw}, got {tuple(x.shape[-2:])}"
            )
        h = self.stem(x)
        for stage in self.stages:
            h = stage(h)
        h = self.post(h)
        return h


class UpBlock(nn.Module):
    """Conv-transpose up-sampling block."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
            _make_group_norm(c_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            _make_group_norm(c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MotionAwareEncoder(nn.Module):
    """Aggregates multi-frame features and predicts motion saliency maps."""

    def __init__(
        self,
        latent_dim: int,
        *,
        input_hw: Tuple[int, int],
        base_channels: int = 64,
        sequence_len: int = 4,
    ) -> None:
        super().__init__()
        if sequence_len < 2:
            raise ValueError("sequence_len must be ≥ 2")
        self.sequence_len = sequence_len
        self.input_hw = input_hw
        self.backbone = FrameBackbone(
            base_channels=base_channels,
            input_hw=input_hw,
        )
        self.latent_hw = self.backbone.latent_hw
        backbone_out = self.backbone.out_channels
        self.temporal_attn = TemporalSelfAttention(backbone_out)

        fusion_in = backbone_out * 4  # last, mean, std, motion magnitude
        self.fusion_proj = nn.Sequential(
            ConvBlock(fusion_in, backbone_out),
        )
        self.spatial_attn = SpatialSelfAttention(backbone_out)
        self.post_attn = ResidualBlock(backbone_out, backbone_out)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(backbone_out, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

        motion_present_in = backbone_out * 2  # fused + motion magnitude
        motion_prior_in = backbone_out * 2  # fused + last frame feat
        self.motion_present_head = nn.Sequential(
            ConvBlock(motion_present_in, backbone_out),
            nn.Conv2d(backbone_out, 1, kernel_size=1),
        )
        self.motion_prior_head = nn.Sequential(
            ConvBlock(motion_prior_in, backbone_out),
            nn.Conv2d(backbone_out, 1, kernel_size=1),
        )

        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(
        self, frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            frames: Tensor with shape (B, T, 3, H, W)
        Returns:
            latent: (B, latent_dim)
            motion_present_logits: (B, 1, H, W)
            motion_prior_logits: (B, 1, H, W)
        """
        B, T, C, H, W = frames.shape
        if T != self.sequence_len:
            raise RuntimeError(
                f"Expected {self.sequence_len} frames, received {T}."
            )
        feats: List[torch.Tensor] = []
        for t in range(T):
            feat = self.backbone(frames[:, t])
            feats.append(feat)

        feat_stack = torch.stack(feats, dim=1)  # (B, T, C, h, w)
        attn_stack = self.temporal_attn(feat_stack)
        last_feat = attn_stack[:, -1]
        mean_feat = attn_stack.mean(dim=1)
        std_feat = torch.sqrt(
            torch.clamp(attn_stack.var(dim=1, unbiased=False), min=1e-6)
        )
        motion_mag = attn_stack[:, 1:] - attn_stack[:, :-1]
        motion_mag = motion_mag.abs().mean(dim=1)

        fusion_input = torch.cat([last_feat, mean_feat, std_feat, motion_mag], dim=1)
        fused = self.fusion_proj(fusion_input)
        fused = self.spatial_attn(fused)
        latent_map = self.post_attn(fused)
        pooled = self.global_pool(latent_map).flatten(1)
        latent = self.norm(self.fc(pooled))

        motion_present_inp = torch.cat([fused, motion_mag], dim=1)
        motion_prior_inp = torch.cat([fused, last_feat], dim=1)

        motion_present_logits = self.motion_present_head(motion_present_inp)
        motion_prior_logits = self.motion_prior_head(motion_prior_inp)

        motion_present_logits = F.interpolate(
            motion_present_logits,
            size=self.input_hw,
            mode="bilinear",
            align_corners=False,
        )
        motion_prior_logits = F.interpolate(
            motion_prior_logits,
            size=self.input_hw,
            mode="bilinear",
            align_corners=False,
        )
        return latent, motion_present_logits, motion_prior_logits

    def encode_single(self, frame: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convenience helper for single-frame inference."""
        if frame.dim() != 4 or frame.shape[1] != 3:
            raise ValueError("Expected frame tensor with shape (B, 3, H, W)")
        repeated = frame.unsqueeze(1).repeat(1, self.sequence_len, 1, 1, 1)
        return self.forward(repeated)


class MotionAwareDecoder(nn.Module):
    """Latent decoder conditioned on motion saliency statistics."""

    def __init__(self, latent_dim: int, *, out_hw: Tuple[int, int]) -> None:
        super().__init__()
        self.out_hw = out_hw
        base_hw = (7, 7)
        base_ch = 512
        self.base_hw = base_hw
        self.base_ch = base_ch
        stats_dim = 4  # mean/max for present + prior
        self.motion_mlp = nn.Sequential(
            nn.Linear(stats_dim, latent_dim),
            nn.SiLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )
        nn.init.kaiming_normal_(self.motion_mlp[0].weight, nonlinearity="relu")
        if self.motion_mlp[0].bias is not None:
            nn.init.zeros_(self.motion_mlp[0].bias)
        nn.init.zeros_(self.motion_mlp[-1].bias)

        self.fc = nn.Linear(latent_dim * 2, base_ch * base_hw[0] * base_hw[1])
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

        self.token_pool = nn.AdaptiveAvgPool2d(base_hw)
        self.motion_token_proj = nn.Linear(2, base_ch)
        self.cross_attn = CrossAttentionBlock(base_ch, num_heads=8)

        self.pre = nn.SiLU(inplace=True)
        self.up1 = UpBlock(base_ch, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up4 = UpBlock(64, 32)
        self.up5 = UpBlock(32, 16)
        self.head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1),
        )

    def forward(
        self,
        latent: torch.Tensor,
        motion_present: Optional[torch.Tensor] = None,
        motion_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if motion_present is None or motion_prior is None:
            device = latent.device
            batch = latent.shape[0]
            motion_embed = torch.zeros(batch, latent.shape[1], device=device)
            motion_tokens = torch.zeros(
                batch,
                self.base_hw[0] * self.base_hw[1],
                self.base_ch,
                device=device,
            )
        else:
            stats = self._motion_stats(motion_present, motion_prior)
            motion_embed = self.motion_mlp(stats)
            motion_tokens = self._build_motion_tokens(motion_present, motion_prior)
        full_latent = torch.cat([latent, motion_embed], dim=1)
        h = self.fc(full_latent)
        h = self.pre(h).view(-1, self.base_ch, *self.base_hw)
        queries = h.view(h.shape[0], self.base_ch, -1).permute(0, 2, 1)
        queries = self.cross_attn(queries, motion_tokens)
        h = queries.permute(0, 2, 1).view(-1, self.base_ch, *self.base_hw)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        h = self.up5(h)
        out = self.head(h)
        if out.shape[-2:] != self.out_hw:
            out = F.interpolate(out, size=self.out_hw, mode="bilinear", align_corners=False)
        return out

    @staticmethod
    def _motion_stats(
        motion_present: torch.Tensor, motion_prior: torch.Tensor
    ) -> torch.Tensor:
        def _stats(map_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            mean = map_tensor.mean(dim=[1, 2, 3], keepdim=False)
            maxv = map_tensor.amax(dim=[1, 2, 3])
            return mean, maxv

        present_mean, present_max = _stats(motion_present)
        prior_mean, prior_max = _stats(motion_prior)
        stats = torch.stack(
            [present_mean, present_max, prior_mean, prior_max], dim=1
        )
        return stats

    def _build_motion_tokens(self, motion_present: torch.Tensor, motion_prior: torch.Tensor) -> torch.Tensor:
        batch = motion_present.shape[0]
        device = motion_present.device
        motion_maps = torch.cat([motion_present, motion_prior], dim=1)
        pooled = self.token_pool(motion_maps)
        tokens = pooled.permute(0, 2, 3, 1).reshape(batch, -1, 2)
        tokens = self.motion_token_proj(tokens)
        return tokens


class ConditionalVectorField(nn.Module):
    """Conditional velocity field for flow matching."""

    def __init__(self, latent_dim: int, hidden: int = 512) -> None:
        super().__init__()
        in_dim = latent_dim * 3 + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        z_data: torch.Tensor,
        z_base: torch.Tensor,
    ) -> torch.Tensor:
        delta = z_data - z_base
        midpoint = 0.5 * (z_data + z_base)
        feat = torch.cat([z_t, delta, midpoint, t[:, None]], dim=1)
        return self.net(feat)


class MotionCFMAutoencoder(nn.Module):
    """Autoencoder with motion-aware conditioning and CFM regularisation."""

    def __init__(
        self,
        latent_dim: int,
        *,
        encoder_base_channels: int = 64,
        input_hw: Tuple[int, int],
        sequence_len: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = MotionAwareEncoder(
            latent_dim,
            input_hw=input_hw,
            base_channels=encoder_base_channels,
            sequence_len=sequence_len,
        )
        self.decoder = MotionAwareDecoder(latent_dim, out_hw=input_hw)
        self.vector_field = ConditionalVectorField(latent_dim)
        self.sequence_len = sequence_len

    def forward(
        self, frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, motion_present_logits, motion_prior_logits = self.encoder(frames)
        motion_present = torch.sigmoid(motion_present_logits)
        motion_prior = torch.sigmoid(motion_prior_logits)
        recon = self.decoder(latent, motion_present, motion_prior)
        return (
            recon,
            latent,
            motion_present_logits,
            motion_prior_logits,
            motion_present,
            motion_prior,
        )

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        latent, _, _ = self.encoder(frames)
        return latent

    def encode_single(self, frame: torch.Tensor) -> torch.Tensor:
        latent, _, _ = self.encoder.encode_single(frame)
        return latent

    def decode(
        self,
        latent: torch.Tensor,
        motion_present: Optional[torch.Tensor] = None,
        motion_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.decoder(latent, motion_present, motion_prior)


# -----------------------------------------------------------------------------
# Visualisation helpers
# -----------------------------------------------------------------------------


def latent_summary(latent: torch.Tensor) -> dict[str, float]:
    data = latent.detach()
    return {
        "mean": float(data.mean().item()),
        "std": float(data.std(unbiased=False).item()),
        "min": float(data.min().item()),
        "max": float(data.max().item()),
    }


@torch.no_grad()
def collect_latents(
    model: MotionCFMAutoencoder,
    dataset: Dataset,
    device: torch.device,
    *,
    max_samples: int,
    batch_size: int,
) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    latents: List[torch.Tensor] = []
    total = 0
    for seq, _, _, _ in loader:
        seq = seq.to(device)
        lat = model.encode(seq).cpu()
        latents.append(lat)
        total += lat.shape[0]
        if total >= max_samples:
            break
    if not latents:
        raise RuntimeError("Failed to collect latents for PCA.")
    return torch.cat(latents, dim=0)[:max_samples]


def compute_pca_components(
    latents: torch.Tensor, n_components: int
) -> Tuple[List[torch.Tensor], List[float], List[torch.Tensor]]:
    if latents.dim() != 2:
        raise ValueError("Expected latent matrix with shape (N, D)")
    centered = latents - latents.mean(dim=0, keepdim=True)
    max_components = min(n_components, centered.shape[1])
    if max_components == 0:
        raise ValueError("No components available for PCA traversal")
    _, _, V = torch.pca_lowrank(centered, q=max_components)
    projections = centered @ V[:, :max_components]
    directions: List[torch.Tensor] = []
    stds: List[float] = []
    projection_samples: List[torch.Tensor] = []
    for idx in range(max_components):
        directions.append(V[:, idx])
        comp_proj = projections[:, idx].detach().cpu()
        std = float(comp_proj.std(unbiased=False).item())
        stds.append(1.0 if std == 0.0 else std)
        projection_samples.append(comp_proj)
    return directions, stds, projection_samples


def to_image(tensor: torch.Tensor) -> Image.Image:
    if tensor.dim() != 3:
        raise ValueError("Expected tensor with shape (C, H, W)")
    img = unnormalize(tensor.unsqueeze(0)).squeeze(0).clip(0, 1)
    to_pil = T.ToPILImage()
    return to_pil(img.cpu())


def motion_to_image(mask: torch.Tensor, *, cmap: str = "magma") -> Image.Image:
    mask_np = mask.squeeze(0).detach().cpu().numpy()
    mask_np = np.clip(mask_np, 0.0, 1.0)
    cmap_fn = plt.get_cmap(cmap)
    rgba = cmap_fn(mask_np)
    rgba_img = (rgba * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(rgba_img, "RGBA")
    return pil_img.convert("RGB")


def overlay_motion(base: Image.Image, motion_img: Image.Image, alpha: float = 0.45) -> Image.Image:
    return Image.blend(base.convert("RGB"), motion_img.convert("RGB"), alpha=alpha)


def save_sequence_visualisation(
    seq: torch.Tensor,
    target: torch.Tensor,
    recon: torch.Tensor,
    motion_target: torch.Tensor,
    motion_present: torch.Tensor,
    motion_prior: torch.Tensor,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [to_image(seq[i]) for i in range(seq.shape[0])]
    target_img = to_image(target)
    recon_img = to_image(recon)
    motion_target_img = overlay_motion(target_img, motion_to_image(motion_target))
    motion_present_img = overlay_motion(target_img, motion_to_image(motion_present))
    motion_prior_img = overlay_motion(target_img, motion_to_image(motion_prior))

    tiles = frames + [target_img, recon_img, motion_target_img, motion_present_img, motion_prior_img]
    titles = [f"t-{len(frames)-idx}" for idx in range(len(frames), 0, -1)] + [
        "target",
        "recon",
        "motion_gt",
        "motion_pred",
        "motion_prior",
    ]
    tile_w, tile_h = target_img.size
    cols = 3
    rows = math.ceil(len(tiles) / cols)
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for idx, (tile, title) in enumerate(zip(tiles, titles)):
        row = idx // cols
        col = idx % cols
        x = col * tile_w
        y = row * tile_h
        canvas.paste(tile, (x, y))
        draw.text((x + 6, y + 6), title, fill=(255, 255, 0))
    canvas.save(out_path)


@torch.no_grad()
def save_pca_traversal_grid(
    model: MotionCFMAutoencoder,
    sequence: torch.Tensor,
    target_frame: torch.Tensor,
    motion_present: torch.Tensor,
    motion_prior: torch.Tensor,
    latent: torch.Tensor,
    directions: Sequence[torch.Tensor],
    stds: Sequence[float],
    percent_levels: Sequence[float],
    projections: Sequence[torch.Tensor],
    out_path: Path,
    device: torch.device,
) -> None:
    if not directions:
        raise ValueError("No PCA directions provided")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_img = to_image(target_frame)
    tile_w, tile_h = base_img.size
    num_rows = len(directions)
    num_cols = len(percent_levels) + 1
    canvas = Image.new("RGB", (num_cols * tile_w, num_rows * tile_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    latent_device = latent.to(device)
    motion_present_device = motion_present.unsqueeze(0).to(device)
    motion_prior_device = motion_prior.unsqueeze(0).to(device)
    percent_list = [float(p) for p in percent_levels]
    label_height = 24

    for row_idx, (direction, std, comp_proj) in enumerate(
        zip(directions, stds, projections)
    ):
        y = row_idx * tile_h
        canvas.paste(base_img, (0, y))
        draw.rectangle([4, y + 4, tile_w - 4, y + 28], outline=(255, 255, 0))
        draw.text((8, y + 8), f"PC{row_idx + 1}", fill=(255, 255, 0))
        draw.rectangle(
            [0, y + tile_h - label_height, tile_w, y + tile_h],
            fill=(0, 0, 0),
        )
        draw.text(
            (6, y + tile_h - label_height + 4),
            "+0.000 (+0%)",
            fill=(255, 255, 0),
        )

        direction_device = direction.to(device)
        comp_proj_float = comp_proj.to(dtype=torch.float32)
        for col_idx, percent in enumerate(percent_list, start=1):
            x = col_idx * tile_w
            if abs(percent) < 1e-6:
                delta = 0.0
                decoded = model.decode(
                    latent_device.unsqueeze(0),
                    motion_present=motion_present_device,
                    motion_prior=motion_prior_device,
                ).cpu()[0]
            else:
                q = 0.5 + 0.5 * percent
                q = max(0.0, min(1.0, q))
                delta = float(torch.quantile(comp_proj_float, q))
                shifted = latent_device + delta * direction_device
                decoded = model.decode(
                    shifted.unsqueeze(0),
                    motion_present=motion_present_device,
                    motion_prior=motion_prior_device,
                ).cpu()[0]
            tile = to_image(decoded)
            canvas.paste(tile, (x, y))
            draw.rectangle(
                [x, y + tile_h - label_height, x + tile_w, y + tile_h],
                fill=(0, 0, 0),
            )
            draw.text(
                (x + 6, y + tile_h - label_height + 4),
                f"{delta:+.3f} ({percent*100:+.0f}%)",
                fill=(255, 255, 0),
            )
    canvas.save(out_path)


def write_loss_csv(hist: List[Tuple[int, float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "loss_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        writer.writerows(hist)


def plot_loss(hist: List[Tuple[int, float]], out_dir: Path, step: int) -> None:
    if not hist:
        return
    steps, losses = zip(*hist)
    plt.figure()
    plt.semilogy(steps, losses)
    plt.xlabel("Step")
    plt.ylabel("Total loss")
    plt.title("Training loss")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"loss_step_{step:06d}.png")
    plt.close()


# -----------------------------------------------------------------------------
# Training CLI
# -----------------------------------------------------------------------------


@dataclass
class Args:
    traj_dir: str = "data.image_distance.train_levels_1_2"
    out_dir: str = "out.reconstruct_mario_motion"
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 200
    steps_per_epoch: int = 100
    latent_dim: int = 1024
    sequence_len: int = 4
    lambda_recon: float = 1.0
    lambda_cfm: float = 0.1
    lambda_latent_l2: float = 0.0
    lambda_motion_bce: float = 1.0
    lambda_prior_consistency: float = 0.1
    lambda_motion_sparsity: float = 1e-3
    ms_weight: float = 1.0
    l1_weight: float = 0.1
    motion_weight: float = 0.5
    motion_target_weight: float = 3.0
    motion_pred_weight: float = 1.0
    motion_prior_weight: float = 0.5
    grad_clip: Optional[float] = 1.0
    max_trajs: Optional[int] = None
    encoder_base_channels: int = 64
    num_workers: int = 0
    device: Optional[str] = None
    log_every: int = 10
    viz_every: int = 50
    viz_samples: int = 4
    pca_batch_size: int = 32
    pca_sample_size: int = 1024
    pca_std_multiplier: float = 2.0
    pca_traverse_steps: int = 5
    seed: int = 42
    resume_checkpoint: Optional[str] = None
    checkpoint_every: int = 50  # 0 disables periodic last-checkpoint updates


def main() -> None:
    args = tyro.cli(Args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if args.pca_traverse_steps < 2:
        raise ValueError("pca_traverse_steps must be ≥ 2")
    device = pick_device(args.device)
    logger.info("Using device: %s", device)

    dataset = MarioMotionDataset(
        args.traj_dir,
        sequence_len=args.sequence_len,
        max_trajs=args.max_trajs,
    )
    logger.info("Dataset loaded: %d sequences", len(dataset))

    num_samples = args.steps_per_epoch * args.batch_size
    sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    sample_seq, sample_target, sample_motion, _ = dataset[0]
    input_hw = sample_target.shape[-2:]

    model = MotionCFMAutoencoder(
        args.latent_dim,
        encoder_base_channels=args.encoder_base_channels,
        input_hw=input_hw,
        sequence_len=args.sequence_len,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_run_dir = Path(args.out_dir) / f"run__{timestamp}"
    run_dir = default_run_dir

    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint).resolve()
        if resume_path.parent.name == "checkpoints":
            resume_run_dir = resume_path.parent.parent
        else:
            resume_run_dir = resume_path.parent
        if resume_run_dir.is_dir():
            run_dir = resume_run_dir

    metrics_dir = run_dir / "metrics"
    samples_dir = run_dir / "samples"
    checkpoints_dir = run_dir / "checkpoints"

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.txt").write_text(str(args))

    global_step = 0
    start_epoch = 0
    loss_hist: List[Tuple[int, float]] = []
    best_metric = float("inf")

    checkpoint_last_path = checkpoints_dir / "checkpoint_last.pt"
    checkpoint_best_path = checkpoints_dir / "checkpoint_best.pt"

    def save_checkpoint(path: Path, epoch_val: int, step_val: int, best_val: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "args": args,
            "epoch": epoch_val,
            "step": step_val,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss_hist": loss_hist,
            "best_metric": best_val,
        }
        torch.save(payload, path)

    if args.resume_checkpoint:
        ckpt_path = Path(args.resume_checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        loss_hist = payload.get("loss_hist", [])
        start_epoch = payload.get("epoch", 0)
        global_step = payload.get("step", 0)
        best_metric = float(payload.get("best_metric", best_metric))
        logger.info("Resumed from %s (epoch %d, step %d)", ckpt_path, start_epoch, global_step)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        for seq, target, motion_target, _ in loader:
            seq = seq.to(device)
            target = target.to(device)
            motion_target = motion_target.to(device)

            (
                recon,
                latent,
                motion_present_logits,
                motion_prior_logits,
                motion_present,
                motion_prior,
            ) = model(seq)

            ms = ms_ssim_loss(recon, target)
            l1 = F.l1_loss(recon, target)
            recon_loss = args.ms_weight * ms + args.l1_weight * l1

            weight_map = 1.0
            if args.motion_weight > 0:
                expanded_motion_target = motion_target.expand_as(recon)
                expanded_motion_pred = motion_present.detach().expand_as(recon)
                expanded_motion_prior = motion_prior.detach().expand_as(recon)
                weight_map = (
                    1.0
                    + args.motion_target_weight * expanded_motion_target
                    + args.motion_pred_weight * expanded_motion_pred
                    + args.motion_prior_weight * expanded_motion_prior
                )
                norm = weight_map.mean(dim=[1, 2, 3], keepdim=True).clamp_min(1e-6)
                weighted_l1 = (torch.abs(recon - target) * weight_map / norm).mean()
            else:
                weighted_l1 = torch.tensor(0.0, device=recon.device)

            motion_bce = F.binary_cross_entropy_with_logits(
                motion_present_logits, motion_target
            )
            motion_prior_prob = torch.sigmoid(motion_prior_logits)
            prior_consistency = F.mse_loss(
                motion_prior_prob, motion_present.detach()
            )
            motion_sparsity = motion_present.mean()

            z_base = torch.randn_like(latent)
            t = torch.rand(latent.shape[0], device=device)
            z_t = (1.0 - t)[:, None] * z_base + t[:, None] * latent
            v_pred = model.vector_field(z_t, t, latent, z_base)
            v_target = latent - z_base
            cfm_loss = F.mse_loss(v_pred, v_target)

            latent_reg = latent.pow(2).mean()
            total_loss = (
                args.lambda_recon * recon_loss
                + args.motion_weight * weighted_l1
                + args.lambda_motion_bce * motion_bce
                + args.lambda_prior_consistency * prior_consistency
                + args.lambda_motion_sparsity * motion_sparsity
                + args.lambda_cfm * cfm_loss
                + args.lambda_latent_l2 * latent_reg
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            global_step += 1
            total_value = float(total_loss.item())
            loss_hist.append((global_step, total_value))

            if global_step % args.log_every == 0:
                stats = latent_summary(latent)
                logger.info(
                    "[epoch %03d | step %06d] loss=%.4f recon=%.4f weighted_l1=%.4f "
                    "motion_bce=%.4f prior=%.4f cfm=%.4f ms=%.4f l1=%.4f latent_std=%.3f",
                    epoch,
                    global_step,
                    total_value,
                    float(recon_loss.item()),
                    float(weighted_l1.item()),
                    float(motion_bce.item()),
                    float(prior_consistency.item()),
                    float(cfm_loss.item()),
                    float(ms.item()),
                    float(l1.item()),
                    stats["std"],
                )

            if global_step % args.viz_every == 0:
                model.eval()
                with torch.no_grad():
                    sample_indices = random.sample(
                        range(len(dataset)), k=min(args.viz_samples, len(dataset))
                    )
                    batch_seq: List[torch.Tensor] = []
                    batch_target: List[torch.Tensor] = []
                    batch_motion: List[torch.Tensor] = []
                    for idx_vis in sample_indices:
                        seq_vis, target_vis, motion_vis, _ = dataset[idx_vis]
                        batch_seq.append(seq_vis)
                        batch_target.append(target_vis)
                        batch_motion.append(motion_vis)
                    batch_seq_tensor = torch.stack(batch_seq, dim=0).to(device)
                    (
                        recon_vis,
                        latent_vis,
                        motion_present_logits_vis,
                        motion_prior_logits_vis,
                        motion_present_vis,
                        motion_prior_vis,
                    ) = model(batch_seq_tensor)

                recon_vis = recon_vis.cpu()
                latent_vis = latent_vis.cpu()
                motion_present_vis = motion_present_vis.cpu()
                motion_prior_vis = motion_prior_vis.cpu()

                latents_for_pca = collect_latents(
                    model,
                    dataset,
                    device,
                    max_samples=args.pca_sample_size,
                    batch_size=args.pca_batch_size,
                )
                directions, stds, proj_samples = compute_pca_components(
                    latents_for_pca, n_components=5
                )
                steps = args.pca_traverse_steps
                if steps % 2 == 0:
                    steps += 1
                max_percent = math.erf(
                    abs(float(args.pca_std_multiplier)) / math.sqrt(2.0)
                )
                if max_percent <= 0.0:
                    max_percent = 1.0
                max_percent = min(max_percent, 1.0)
                percent_levels = torch.linspace(
                    -max_percent,
                    max_percent,
                    steps=steps,
                )

                for idx_vis, idx_dataset in enumerate(sample_indices):
                    seq_vis, target_vis, motion_vis, _ = dataset[idx_dataset]
                    out_path = samples_dir / f"sample_step_{global_step:06d}_idx_{idx_vis}.png"
                    save_sequence_visualisation(
                        seq_vis,
                        target_vis,
                        recon_vis[idx_vis],
                        motion_vis,
                        motion_present_vis[idx_vis],
                        motion_prior_vis[idx_vis],
                        out_path,
                    )
                    if directions:
                        pca_path = samples_dir / (
                            f"pca_step_{global_step:06d}_idx_{idx_vis}.png"
                        )
                        save_pca_traversal_grid(
                            model,
                            seq_vis,
                            target_vis,
                            motion_present_vis[idx_vis],
                            motion_prior_vis[idx_vis],
                            latent_vis[idx_vis],
                            directions,
                            stds,
                            percent_levels.tolist(),
                            proj_samples,
                            pca_path,
                            device,
                        )
                plot_loss(loss_hist, metrics_dir, global_step)
                model.train()

            updated_best = False
            if total_value < best_metric:
                best_metric = total_value
                save_checkpoint(checkpoint_best_path, epoch, global_step, best_metric)
                updated_best = True

            save_last = updated_best or (
                args.checkpoint_every > 0
                and global_step % args.checkpoint_every == 0
            )
            if save_last:
                save_checkpoint(checkpoint_last_path, epoch, global_step, best_metric)

        sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=False,
        )

    write_loss_csv(loss_hist, metrics_dir)
    save_checkpoint(run_dir / "final.pt", args.epochs, global_step, best_metric)
    logger.info("Training finished. Artifacts written to %s", run_dir)


if __name__ == "__main__":
    main()
