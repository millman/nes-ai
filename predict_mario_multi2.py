#!/usr/bin/env python3
"""Train a Mario frame predictor with shared autoencoder and temporal attention.

The model ingests four consecutive RGB frames plus their actions, aggregates
them with attention, and predicts the next-frame latent representation. A shared
decoder reconstructs both the predicted latent and individual encoded frames,
which keeps the decoder usable outside the sequence context. Training minimises
SmoothL1 reconstruction over the decoded inputs and a SmoothL1 latent loss
against the ground-truth next-frame latent. The script also mirrors the
convenience features from ``reconstruct_mario_comparison.py``: resumable
best/last/final checkpoints, scalar loss plotting, and visual grid summaries.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import tyro
from torchvision.models import ResNet18_Weights
from torchvision.transforms import Normalize

from predict_mario_ms_ssim import pick_device, unnormalize
from recon.data import list_trajectories, load_frame_as_tensor, short_traj_state_label
from super_mario_env import _describe_controller_vector


logger = logging.getLogger(__name__)


class _NullContext:
    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, *_: object) -> bool:
        return False


def _normalize_latent_tensor(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalise over spatial/channel axes while preserving batch/sequence dims."""
    if tensor.dim() < 3:
        raise ValueError(f"Expected at least 3 dims for latent tensor, got {tuple(tensor.shape)}")
    flat = tensor.flatten(start_dim=-3)
    norms = flat.norm(dim=-1, keepdim=True).clamp_min(eps)
    normalized = flat / norms
    return normalized.view_as(tensor)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    data_root: Path = Path("data.image_distance.train_levels_1_2")
    output_dir: Path = Path("out.predict_mario_multi2")
    epochs: int = 1000
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    val_fraction: float = 0.1
    seed: int = 42
    device: str = "auto"
    lambda_recon: float = 1.0
    lambda_latent: float = 1.0
    lambda_pred: float = 1.0
    lambda_residual: float = 1.0
    lambda_latent_cosine: float = 0.5
    lambda_skipless: float = 0.5
    lambda_skip: float = 0.5
    context_steps: int = 8
    future_steps: int = 8
    encode_chunk: int = 4
    decode_chunk: int = 4
    reason_steps: int = 2
    reason_inner_steps: int = 2
    base_channels: int = 32
    latent_channels: int = 96
    vis_every: int = 50
    checkpoint_every: int = 50
    resume: bool = False
    max_trajs: Optional[int] = None
    log_every: int = 50
    mixed_precision: bool = False
    image_hw: Tuple[int, int] = (224, 224)

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.output_dir = Path(self.output_dir)
        if self.base_channels % 8 != 0:
            raise ValueError("base_channels must be divisible by 8 for GroupNorm compatibility.")
        if self.latent_channels % 8 != 0:
            raise ValueError("latent_channels must be divisible by 8 for GroupNorm compatibility.")
        if self.context_steps <= 0 or self.future_steps <= 0:
            raise ValueError("context_steps and future_steps must be positive.")
        if self.encode_chunk <= 0 or self.decode_chunk <= 0:
            raise ValueError("encode_chunk and decode_chunk must be positive.")
        if self.encode_chunk > self.context_steps:
            self.encode_chunk = self.context_steps
        if self.decode_chunk > self.future_steps:
            self.decode_chunk = self.future_steps
        if self.reason_steps < 0:
            raise ValueError("reason_steps must be non-negative.")
        if self.reason_inner_steps <= 0:
            raise ValueError("reason_inner_steps must be positive.")


class MarioSequenceDataset(Dataset):
    """Return sliding windows of context frames plus actions and future frames."""

    def __init__(
        self,
        root_dir: Path,
        *,
        image_hw: Tuple[int, int],
        context_steps: int,
        future_steps: int,
        normalize: Optional[Normalize] = None,
        max_trajs: Optional[int] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_hw = image_hw
        self.context_steps = int(context_steps)
        self.future_steps = int(future_steps)
        self.normalize = normalize
        self.entries: List[Tuple[List[Path], np.ndarray, int]] = []
        self.action_dim: Optional[int] = None
        traj_map = list_trajectories(self.root_dir)
        traj_items = list(traj_map.items())
        if max_trajs is not None:
            traj_items = traj_items[:max_trajs]
        for traj_name, frames in traj_items:
            actions_path = (self.root_dir / traj_name) / "actions.npz"

            assert actions_path.is_file(), f"Not a file: {actions_path}"

            min_frames = self.context_steps + self.future_steps
            if len(frames) < min_frames:
                continue

            with np.load(actions_path) as data:
                if "actions" in data.files:
                    actions = data["actions"]
                elif len(data.files) == 1:
                    actions = data[data.files[0]]
                else:
                    raise ValueError(f"Missing 'actions' in {actions_path}: {data.files}")
            if actions.ndim == 1:
                actions = actions[:, None]
            if actions.shape[0] == len(frames) + 1:
                actions = actions[1:]
            if actions.shape[0] != len(frames):
                raise ValueError(
                    f"Action count {actions.shape[0]} mismatch with frames {len(frames)} in {actions_path.parent}"
                )
            actions = actions.astype(np.float32)
            if self.action_dim is None:
                self.action_dim = actions.shape[1]
            elif actions.shape[1] != self.action_dim:
                raise ValueError(
                    f"Inconsistent action dimensions: expected {self.action_dim}, got {actions.shape[1]} in {actions_path.parent}"
                )
            for start in range(len(frames) - (min_frames - 1)):
                self.entries.append((frames, actions, start))
        if not self.entries:
            raise RuntimeError(f"No valid trajectories under {self.root_dir}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        frame_paths, actions, start = self.entries[idx]
        inputs = []
        futures = []
        for i in range(self.context_steps):
            frame = load_frame_as_tensor(
                frame_paths[start + i],
                size=self.image_hw,
                normalize=None,
            )
            if self.normalize is not None:
                frame = self.normalize(frame)
            inputs.append(frame)
        for i in range(self.future_steps):
            frame = load_frame_as_tensor(
                frame_paths[start + self.context_steps + i],
                size=self.image_hw,
                normalize=None,
            )
            if self.normalize is not None:
                frame = self.normalize(frame)
            futures.append(frame)
        input_actions = actions[start : start + self.context_steps]
        future_actions = actions[start + self.context_steps : start + self.context_steps + self.future_steps]
        input_labels = [
            short_traj_state_label(str(frame_paths[start + i])) for i in range(self.context_steps)
        ]
        future_labels = [
            short_traj_state_label(str(frame_paths[start + self.context_steps + i])) for i in range(self.future_steps)
        ]

        return {
            "inputs": torch.stack(inputs, dim=0),
            "future_frames": torch.stack(futures, dim=0),
            "actions": torch.from_numpy(input_actions),
            "future_actions": torch.from_numpy(future_actions),
            "input_labels": input_labels,
            "future_labels": future_labels,
        }


class DownBlock(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _group_norm(num_channels: int) -> nn.GroupNorm:
    groups = 8 if num_channels % 8 == 0 else 1
    return nn.GroupNorm(groups, num_channels)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.skip_ch = skip_ch
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        fuse_in = out_ch + skip_ch
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, out_ch, kernel_size=3, padding=1),
            _group_norm(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            _group_norm(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.up(x)
        if skip is None:
            skip = x.new_zeros(x.shape[0], self.skip_ch, *x.shape[-2:])
        else:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class FrameEncoder(nn.Module):
    """Shared encoder inspired by LightweightAutoencoder."""

    def __init__(self, base_channels: int = 32, latent_channels: int = 96) -> None:
        super().__init__()
        if base_channels % 8 != 0:
            raise ValueError("base_channels must be divisible by 8")
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
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(
        self, frame: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        h0 = self.stem(frame)
        h1 = self.down1(h0)
        h2 = self.down2(h1)
        h3 = self.down3(h2)
        latent = self.bottleneck(h3)
        pooled = self.pool(_normalize_latent_tensor(latent)).flatten(1)
        # h0, h1, h2 correspond to resolutions H, H/2, and H/4 respectively.
        skips = (h0, h1, h2)
        return latent, pooled, skips


class LatentDecoder(nn.Module):
    def __init__(self, latent_channels: int = 96, base_channels: int = 32) -> None:
        super().__init__()
        self.up_bottom = UpBlock(latent_channels, base_channels * 3, base_channels * 3)
        self.up_mid = UpBlock(base_channels * 3, base_channels * 2, base_channels * 2)
        self.up_top = UpBlock(base_channels * 2, base_channels, base_channels)
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels // 2, 3, kernel_size=1),
        )

    def forward(
        self,
        latents: torch.Tensor,
        skip_pyramids: Sequence[Optional[torch.Tensor]],
        *,
        target_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        latents: (B,C,H,W) or (B,T,C,H,W)
        skip_pyramids: iterable matching encoder skip order (h0, h1, h2)
                       shaped like latents (broadcastable along batch/temporal dims).
        """
        if latents.dim() == 5:
            b, t, c, h, w = latents.shape
            latents_flat = latents.reshape(b * t, c, h, w)
            skips_flat: List[Optional[torch.Tensor]] = []
            for skip in skip_pyramids:
                if skip is None:
                    skips_flat.append(None)
                else:
                    skips_flat.append(skip.reshape(b * t, skip.shape[2], skip.shape[3], skip.shape[4]))
        else:
            latents_flat = latents
            skips_flat = []
            for skip in skip_pyramids:
                skips_flat.append(skip)

        # Expecting skip order [h0, h1, h2] (shallow -> deep)
        skip_h0 = skips_flat[0] if len(skips_flat) > 0 else None
        skip_h1 = skips_flat[1] if len(skips_flat) > 1 else None
        skip_h2 = skips_flat[2] if len(skips_flat) > 2 else None

        h = self.up_bottom(latents_flat, skip_h2)
        h = self.up_mid(h, skip_h1)
        h = self.up_top(h, skip_h0)
        frame = self.head(h)
        if frame.shape[-2:] != target_hw:
            frame = F.interpolate(frame, size=target_hw, mode="bilinear", align_corners=False)
        if latents.dim() == 5:
            frame = frame.reshape(b, t, 3, *target_hw)
        return frame


class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.net(actions)


class TemporalSpatialAggregator(nn.Module):
    def __init__(self, latent_dim: int, action_embed_dim: int) -> None:
        super().__init__()
        if action_embed_dim != latent_dim:
            raise ValueError("Action embedding dimension must match latent dimension.")
        hidden = max(1, latent_dim // 2)
        self.summary_pool = nn.AdaptiveAvgPool2d(1)
        self.weight_net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        latent_maps: torch.Tensor,
        skip_pyramids: Sequence[torch.Tensor],
        action_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor]:
        if latent_maps.dim() != 5:
            raise ValueError(f"Expected 5D latent tensor, got shape {tuple(latent_maps.shape)}")
        batch, steps, channels, height, width = latent_maps.shape
        if action_embeddings.shape[:2] != (batch, steps):
            raise ValueError(
                f"Action embeddings shape {tuple(action_embeddings.shape)} does not match latent sequence {(batch, steps)}"
            )
        if action_embeddings.shape[-1] != channels:
            raise ValueError(
                f"Expected action embedding dimension {channels}, got {action_embeddings.shape[-1]}"
            )

        normalized = _normalize_latent_tensor(latent_maps)
        pooled = self.summary_pool(
            normalized.reshape(batch * steps, channels, height, width)
        ).reshape(batch, steps, channels)
        step_features = pooled
        weight_inputs = torch.cat([step_features, action_embeddings], dim=-1)
        logits = self.weight_net(weight_inputs).squeeze(-1)
        attn_weights = torch.softmax(logits, dim=1)
        weight_view = attn_weights.reshape(batch, steps, 1, 1, 1)
        fused = (latent_maps * weight_view).sum(dim=1)
        fused_skips: List[torch.Tensor] = []
        for level_feats in skip_pyramids:
            if level_feats.dim() != 5:
                raise ValueError(
                    f"Expected skip pyramid tensor with 5 dimensions, got {tuple(level_feats.shape)}"
                )
            fused_skips.append((level_feats * weight_view).sum(dim=1))
        return fused, tuple(fused_skips), attn_weights


class MultiScaleResidualPredictor(nn.Module):
    def __init__(self, latent_dim: int, skip_dims: Sequence[int]) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.skip_dims = tuple(int(d) for d in skip_dims)

        self.action_proj = nn.Linear(latent_dim, latent_dim)
        self.delta_net = nn.Sequential(
            nn.Conv2d(latent_dim * 2, latent_dim, kernel_size=3, padding=1),
            _group_norm(latent_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
        )
        self.skip_projections = nn.ModuleList(
            nn.Conv2d(latent_dim, skip_dim, kernel_size=1) for skip_dim in self.skip_dims
        )

    def forward(
        self,
        initial_latent: torch.Tensor,
        initial_skips: Sequence[torch.Tensor],
        future_action_embeddings: Optional[torch.Tensor],
        *,
        steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        if initial_latent.dim() != 4:
            raise ValueError(
                f"Expected initial latent of shape (B,C,H,W), got {tuple(initial_latent.shape)}"
            )
        if len(initial_skips) != len(self.skip_dims):
            raise ValueError("Initial skip pyramid count mismatch.")
        batch, channels, height, width = initial_latent.shape
        if channels != self.latent_dim:
            raise ValueError(
                f"Initial latent channel count {channels} does not match configured {self.latent_dim}"
            )

        if future_action_embeddings is not None:
            if future_action_embeddings.dim() != 3:
                raise ValueError(
                    f"Expected future action embeddings (B,T,C), got {tuple(future_action_embeddings.shape)}"
                )
            if future_action_embeddings.shape[0] != batch:
                raise ValueError("Future action embedding batch dimension mismatch.")
            if future_action_embeddings.shape[2] != self.latent_dim:
                raise ValueError("Future action embedding dimension must equal latent dimension.")
            rollout_steps = future_action_embeddings.shape[1]
            action_embeddings = future_action_embeddings
        else:
            if steps is None:
                raise ValueError("Number of rollout steps must be specified when action embeddings are None.")
            rollout_steps = steps
            action_embeddings = initial_latent.new_zeros(batch, rollout_steps, self.latent_dim)

        if rollout_steps <= 0:
            raise ValueError("Rollout steps must be positive.")

        current_latent = initial_latent
        current_norm = _normalize_latent_tensor(current_latent)

        latents: List[torch.Tensor] = []
        residuals: List[torch.Tensor] = []
        skip_sequences: List[List[torch.Tensor]] = [[] for _ in self.skip_dims]
        skip_sizes = [skip.shape[-2:] for skip in initial_skips]

        for t in range(rollout_steps):
            action_vec = action_embeddings[:, t]
            projected_action = self.action_proj(action_vec).unsqueeze(-1).unsqueeze(-1)
            action_map = projected_action.expand(-1, -1, height, width)
            fused = torch.cat([current_latent, action_map], dim=1)
            delta = self.delta_net(fused)
            updated_latent = current_latent + delta
            updated_norm = _normalize_latent_tensor(updated_latent)

            latents.append(updated_latent)
            residuals.append(updated_norm - current_norm)

            current_latent = updated_latent
            current_norm = updated_norm
            for idx, proj in enumerate(self.skip_projections):
                skip_h, skip_w = skip_sizes[idx]
                skip_map = proj(
                    F.interpolate(updated_latent, size=(skip_h, skip_w), mode="bilinear", align_corners=False)
                )
                skip_sequences[idx].append(skip_map)

        stacked_skips = tuple(torch.stack(level_seq, dim=1) for level_seq in skip_sequences)
        return torch.stack(latents, dim=1), torch.stack(residuals, dim=1), stacked_skips


class LatentReasoner(nn.Module):
    def __init__(self, latent_dim: int, skip_dims: Sequence[int]) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.skip_dims = tuple(int(d) for d in skip_dims)

        combined_channels = latent_dim * 2 + latent_dim  # latent + state + action
        groups = 8 if latent_dim % 8 == 0 else 1
        self.latent_update = nn.Sequential(
            nn.Conv2d(combined_channels, latent_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, latent_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, latent_dim),
        )
        self.state_update = nn.Sequential(
            nn.Conv2d(latent_dim * 2, latent_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, latent_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, latent_dim),
        )
        self.action_proj = nn.Linear(latent_dim, latent_dim)

        self.skip_refine = nn.ModuleList()
        for skip_dim in self.skip_dims:
            in_channels = skip_dim + latent_dim * 3
            skip_groups = 8 if skip_dim % 8 == 0 else 1
            self.skip_refine.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, skip_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(skip_groups, skip_dim),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(skip_dim, skip_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(skip_groups, skip_dim),
                )
            )

    def forward(
        self,
        latents: torch.Tensor,
        skip_pyramids: Tuple[torch.Tensor, ...],
        reason_state: torch.Tensor,
        action_embeddings: Optional[torch.Tensor],
        *,
        inner_steps: int,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor]:
        if inner_steps <= 0:
            return latents, skip_pyramids, reason_state
        if action_embeddings is None:
            action_embeddings = latents.new_zeros(latents.shape[0], latents.shape[1], self.latent_dim)
        else:
            if action_embeddings.shape[-1] != self.latent_dim:
                raise ValueError("Action embeddings must match latent dimension for reasoning.")
        state = reason_state
        latent_sequence = latents
        skip_sequence = skip_pyramids
        for _ in range(inner_steps):
            latent_sequence, skip_sequence, state = self._single_iteration(latent_sequence, skip_sequence, state, action_embeddings)
        return latent_sequence, skip_sequence, state

    def _single_iteration(
        self,
        latents: torch.Tensor,
        skip_pyramids: Tuple[torch.Tensor, ...],
        reason_state: torch.Tensor,
        action_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor]:
        batch, steps, _, height, width = latents.shape
        state = reason_state
        updated_latents: List[torch.Tensor] = []
        updated_skip_levels: List[List[torch.Tensor]] = [[] for _ in self.skip_dims]

        for t in range(steps):
            action_vec = self.action_proj(action_embeddings[:, t])
            action_map = action_vec.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
            latent_input = torch.cat([latents[:, t], state, action_map], dim=1)
            latent_delta = self.latent_update(latent_input)
            latent_next = latents[:, t] + latent_delta
            latent_next = _normalize_latent_tensor(latent_next)

            state_input = torch.cat([state, latent_next], dim=1)
            state_delta = self.state_update(state_input)
            state = _normalize_latent_tensor(state + state_delta)

            updated_latents.append(latent_next)

            for level, module in enumerate(self.skip_refine):
                skip_tensor = skip_pyramids[level][:, t]
                target_hw = skip_tensor.shape[-2:]
                latent_resized = F.interpolate(latent_next, size=target_hw, mode="bilinear", align_corners=False)
                state_resized = F.interpolate(state, size=target_hw, mode="bilinear", align_corners=False)
                action_resized = F.interpolate(action_map, size=target_hw, mode="bilinear", align_corners=False)
                skip_input = torch.cat([skip_tensor, latent_resized, state_resized, action_resized], dim=1)
                skip_delta = module(skip_input)
                updated_skip_levels[level].append(skip_tensor + skip_delta)

        updated_latent_tensor = torch.stack(updated_latents, dim=1)
        updated_skip_tensors: Tuple[torch.Tensor, ...]
        if self.skip_dims:
            updated_skip_tensors = tuple(torch.stack(level_seq, dim=1) for level_seq in updated_skip_levels)
        else:
            updated_skip_tensors = tuple()
        return updated_latent_tensor, updated_skip_tensors, state


@dataclass
class ModelOutputs:
    predicted_frames: torch.Tensor
    predicted_latents: torch.Tensor
    predicted_residuals: torch.Tensor
    predicted_skips: Tuple[torch.Tensor, ...]
    context_latent: torch.Tensor
    context_skips: Tuple[torch.Tensor, ...]
    input_latents: torch.Tensor
    input_reconstructions: Optional[torch.Tensor]
    encoded_future_latents: Optional[torch.Tensor]
    encoded_future_skips: Optional[Tuple[torch.Tensor, ...]]
    reason_state: torch.Tensor
    reason_iterations: int
    attention_weights: Optional[torch.Tensor]


class MarioSequencePredictor(nn.Module):
    def __init__(
        self,
        action_dim: int,
        *,
        base_channels: int = 32,
        latent_channels: int = 96,
        image_hw: Tuple[int, int] = (224, 224),
        decode_chunk: int = 4,
        reason_steps: int = 0,
        reason_inner_steps: int = 1,
    ) -> None:
        super().__init__()
        self.image_hw = image_hw
        self.decode_chunk = max(1, int(decode_chunk))
        self.reason_steps = max(0, int(reason_steps))
        self.reason_inner_steps = max(1, int(reason_inner_steps))
        self.encoder = FrameEncoder(base_channels=base_channels, latent_channels=latent_channels)
        self.decoder = LatentDecoder(latent_channels=latent_channels, base_channels=base_channels)
        self.action_encoder = ActionEncoder(action_dim, embed_dim=latent_channels)
        self.aggregator = TemporalSpatialAggregator(
            latent_dim=latent_channels,
            action_embed_dim=latent_channels,
        )
        self.skip_dims = (base_channels, base_channels * 2, base_channels * 3)
        self.predictor = MultiScaleResidualPredictor(latent_dim=latent_channels, skip_dims=self.skip_dims)
        self.reason_state_init = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        self.reasoner = LatentReasoner(latent_dim=latent_channels, skip_dims=self.skip_dims)
        self.recon_loss_fn = nn.SmoothL1Loss()
        self.latent_loss_fn = nn.MSELoss()
        self.residual_loss_fn = nn.MSELoss()

    def _decode_sequence(
        self,
        latents: torch.Tensor,
        skip_pyramids: Sequence[Optional[torch.Tensor]],
        *,
        chunk_size: int,
    ) -> torch.Tensor:
        if latents.dim() != 5:
            return self.decoder(latents, skip_pyramids, target_hw=self.image_hw)
        steps = latents.shape[1]
        if chunk_size <= 0 or chunk_size >= steps:
            return self.decoder(latents, skip_pyramids, target_hw=self.image_hw)
        outputs: List[torch.Tensor] = []
        for start in range(0, steps, chunk_size):
            end = min(start + chunk_size, steps)
            chunk_latents = latents[:, start:end]
            chunk_skips: List[Optional[torch.Tensor]] = []
            for skip in skip_pyramids:
                if skip is None:
                    chunk_skips.append(None)
                else:
                    chunk_skips.append(skip[:, start:end])
            outputs.append(
                self.decoder(
                    chunk_latents,
                    chunk_skips,
                    target_hw=self.image_hw,
                )
            )
        return torch.cat(outputs, dim=1)

    def encode_frames(
        self, frames: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        batch, steps, _, _, _ = frames.shape
        latents: List[torch.Tensor] = []
        skip_levels: Optional[List[List[torch.Tensor]]] = None
        for t in range(steps):
            latent, _, skips = self.encoder(frames[:, t])
            latents.append(latent)
            if skip_levels is None:
                skip_levels = [[] for _ in range(len(skips))]
            for level, feat in enumerate(skips):
                skip_levels[level].append(feat)
        stacked_latents = torch.stack(latents, dim=1)
        skip_pyramids: Tuple[torch.Tensor, ...]
        if skip_levels is None:
            skip_pyramids = tuple()
        else:
            skip_pyramids = tuple(torch.stack(level_feats, dim=1) for level_feats in skip_levels)
        return stacked_latents, skip_pyramids

    def forward(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        *,
        future_frames: Optional[torch.Tensor] = None,
        future_actions: Optional[torch.Tensor] = None,
        decode_recon: bool = True,
    ) -> ModelOutputs:
        latent_maps, skip_pyramids = self.encode_frames(frames)
        action_embeddings = self.action_encoder(actions)
        context_map, context_skips, attn_weights = self.aggregator(
            latent_maps, skip_pyramids, action_embeddings
        )
        rollout_steps: Optional[int] = None
        future_action_embeddings = None
        if future_actions is not None:
            future_action_embeddings = self.action_encoder(future_actions)
            rollout_steps = future_action_embeddings.shape[1]
        if rollout_steps is None and future_frames is not None:
            rollout_steps = future_frames.shape[1]
        if rollout_steps is None:
            rollout_steps = frames.shape[1]
        predicted_latents, predicted_residuals, predicted_skips = self.predictor(
            context_map,
            context_skips,
            future_action_embeddings,
            steps=rollout_steps,
        )
        reason_state = _normalize_latent_tensor(self.reason_state_init(context_map))
        latents_for_reason = predicted_latents
        skips_for_reason = predicted_skips
        executed_reason_iterations = 0
        if self.reason_steps > 0 and latents_for_reason.dim() == 5:
            for step in range(self.reason_steps):
                context_manager = torch.no_grad() if step < self.reason_steps - 1 else _NullContext()
                with context_manager:
                    latents_for_reason, skips_for_reason, reason_state = self.reasoner(
                        latents_for_reason,
                        skips_for_reason,
                        reason_state,
                        future_action_embeddings,
                        inner_steps=self.reason_inner_steps,
                    )
                executed_reason_iterations += self.reason_inner_steps
                if step < self.reason_steps - 1:
                    latents_for_reason = latents_for_reason.detach()
                    skips_for_reason = tuple(skip.detach() for skip in skips_for_reason)
                    reason_state = reason_state.detach()
        predicted_latents = latents_for_reason
        predicted_skips = skips_for_reason
        predicted_frames = self._decode_sequence(
            predicted_latents,
            predicted_skips,
            chunk_size=self.decode_chunk,
        )
        recon = None
        if decode_recon:
            recon = self._decode_sequence(
                latent_maps,
                skip_pyramids,
                chunk_size=self.decode_chunk,
            )
        encoded_future_latents = None
        encoded_future_skips: Optional[Tuple[torch.Tensor, ...]] = None
        if future_frames is not None:
            future_latent_maps, future_skip_pyramids = self.encode_frames(future_frames)
            encoded_future_latents = future_latent_maps
            encoded_future_skips = future_skip_pyramids
        return ModelOutputs(
            predicted_frames=predicted_frames,
            predicted_latents=predicted_latents,
            predicted_residuals=predicted_residuals,
            predicted_skips=predicted_skips,
            context_latent=context_map,
            context_skips=context_skips,
            input_latents=latent_maps,
            input_reconstructions=recon,
            encoded_future_latents=encoded_future_latents,
            encoded_future_skips=encoded_future_skips,
             reason_state=reason_state,
             reason_iterations=executed_reason_iterations,
            attention_weights=attn_weights,
        )

    def compute_losses(
        self,
        outputs: ModelOutputs,
        *,
        input_frames: torch.Tensor,
        target_frames: torch.Tensor,
        lambda_recon: float = 1.0,
        lambda_latent: float = 1.0,
        lambda_pred: float = 1.0,
        lambda_residual: float = 1.0,
        lambda_latent_cosine: float = 1.0,
        lambda_skipless: float = 0.0,
        lambda_skip: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        if outputs.input_reconstructions is None:
            raise ValueError("Reconstruction logits not computed.")
        recon_loss = self.recon_loss_fn(outputs.input_reconstructions, input_frames)
        skipless_loss = torch.zeros_like(recon_loss)
        if lambda_skipless > 0.0:
            if outputs.input_latents is None:
                raise ValueError("Input latents required for skipless loss.")
            b, steps, _, _, _ = outputs.input_latents.shape
            skipless_terms = []
            for t in range(steps):
                decoded = self.decoder(
                    outputs.input_latents[:, t],
                    (),
                    target_hw=self.image_hw,
                ).contiguous()
                target = input_frames[:, t].contiguous()
                skipless_terms.append(self.recon_loss_fn(decoded, target))
            skipless_loss = torch.stack(skipless_terms).mean()
        if outputs.encoded_future_latents is None:
            raise ValueError("Encoded future latents required for sequence losses.")
        target_latents = _normalize_latent_tensor(outputs.encoded_future_latents.detach())
        predicted_latents_norm = _normalize_latent_tensor(outputs.predicted_latents)
        if predicted_latents_norm.shape != target_latents.shape:
            raise ValueError(
                f"Predicted latents shape {tuple(predicted_latents_norm.shape)} does not match targets {tuple(target_latents.shape)}"
            )
        latent_loss = self.latent_loss_fn(
            predicted_latents_norm.reshape(-1, *predicted_latents_norm.shape[2:]),
            target_latents.reshape(-1, *target_latents.shape[2:]),
        )
        target_frames = target_frames.contiguous()
        if outputs.predicted_frames.shape != target_frames.shape:
            raise ValueError(
                f"Predicted frames shape {tuple(outputs.predicted_frames.shape)} does not match targets {tuple(target_frames.shape)}"
            )
        pred_loss = self.recon_loss_fn(
            outputs.predicted_frames.reshape(-1, *outputs.predicted_frames.shape[2:]),
            target_frames.reshape(-1, *target_frames.shape[2:]),
        )
        prev_context = _normalize_latent_tensor(outputs.context_latent.detach().unsqueeze(1))
        prev_latents = torch.cat([prev_context, target_latents[:, :-1]], dim=1)
        target_residuals = target_latents - prev_latents
        predicted_residuals = outputs.predicted_residuals
        if predicted_residuals.shape != target_residuals.shape:
            raise ValueError(
                f"Predicted residuals shape {tuple(predicted_residuals.shape)} does not match targets {tuple(target_residuals.shape)}"
            )
        residual_loss = self.residual_loss_fn(
            predicted_residuals.reshape(-1, *predicted_residuals.shape[2:]),
            target_residuals.reshape(-1, *target_residuals.shape[2:]),
        )
        cosine_loss = predicted_latents_norm.new_zeros(())
        if lambda_latent_cosine > 0.0:
            pred_flat = predicted_latents_norm.reshape(predicted_latents_norm.shape[0], predicted_latents_norm.shape[1], -1)
            target_flat = target_latents.reshape(target_latents.shape[0], target_latents.shape[1], -1)
            cosine = F.cosine_similarity(pred_flat, target_flat, dim=-1, eps=1e-6)
            cosine_loss = (1.0 - cosine).mean()
        skip_loss = predicted_latents_norm.new_zeros(())
        if lambda_skip > 0.0:
            if outputs.predicted_skips is None or outputs.encoded_future_skips is None:
                raise ValueError("Skip supervision requested but skip tensors are unavailable.")
            if len(outputs.predicted_skips) != len(outputs.encoded_future_skips):
                raise ValueError("Predicted skip pyramid count does not match targets.")
            skip_terms = []
            for pred_skip, target_skip in zip(outputs.predicted_skips, outputs.encoded_future_skips):
                if pred_skip.shape != target_skip.shape:
                    raise ValueError(
                        f"Predicted skip shape {tuple(pred_skip.shape)} does not match targets {tuple(target_skip.shape)}"
                    )
                skip_terms.append(
                    self.recon_loss_fn(
                        pred_skip.reshape(-1, *pred_skip.shape[2:]),
                        target_skip.detach().reshape(-1, *target_skip.shape[2:]),
                    )
                )
            skip_loss = torch.stack(skip_terms).mean()
        total = (
            lambda_recon * recon_loss
            + lambda_latent * latent_loss
            + lambda_pred * pred_loss
            + lambda_residual * residual_loss
            + lambda_latent_cosine * cosine_loss
            + lambda_skipless * skipless_loss
            + lambda_skip * skip_loss
        )
        return {
            "total": total,
            "recon": recon_loss,
            "latent": latent_loss,
            "pred": pred_loss,
            "residual": residual_loss,
            "latent_cosine": cosine_loss,
            "skipless": skipless_loss,
            "skip": skip_loss,
        }


LOSS_COLUMNS = ["step", "total", "recon", "latent", "pred", "residual", "latent_cosine", "skipless", "skip"]


def write_loss_csv(history: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LOSS_COLUMNS)
        for entry in history:
            writer.writerow([entry.get(col, float("nan")) for col in LOSS_COLUMNS])


def plot_loss_curves(history: List[Dict[str, float]], out_dir: Path) -> None:
    if not history:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = [entry["step"] for entry in history]

    plt.figure(figsize=(8, 5))
    for label in ("total", "recon", "latent", "pred", "residual", "latent_cosine", "skipless", "skip"):
        plt.plot(steps, [entry.get(label, float("nan")) for entry in history], label=label)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=200)
    plt.close()


def _ensure_dirs(config: TrainConfig) -> Dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir: Path
    if config.resume:
        existing_runs = sorted(
            [
                p
                for p in config.output_dir.iterdir()
                if p.is_dir() and p.name[0].isdigit()
            ],
            reverse=True,
        )
        if existing_runs:
            run_dir = existing_runs[0]
        else:
            run_dir = config.output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = config.output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    vis_dir = run_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    return {
        "run": run_dir,
        "checkpoints": ckpt_dir,
        "plots": plots_dir,
        "visualizations": vis_dir,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    best_loss: float,
    history: List[Dict[str, float]],
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_loss": best_loss,
        "history": history,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[int, int, float, List[Dict[str, float]]]:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return (
        int(payload["epoch"]),
        int(payload["step"]),
        float(payload.get("best_loss", math.inf)),
        payload.get("history", []),
    )


def _tensor_to_image(img: torch.Tensor) -> np.ndarray:
    """Convert a (C, H, W) tensor to (H, W, C) numpy array for matplotlib."""
    if img.dim() != 3:
        raise ValueError(f"Expected (C,H,W) tensor, got shape {tuple(img.shape)}")
    # img: (C, H, W)
    with torch.no_grad():
        data = unnormalize(img.cpu()).clamp(0.0, 1.0)
    # unnormalize returns (1, C, H, W), squeeze the batch dimension
    assert data.dim() == 4, f"Expected dim size 4, got {data.dim()}"
    assert data.shape[0] == 1, f"Expected batch size 1, got {data.shape[0]}"
    data = data.squeeze(0)  # (C, H, W)
    # data: (C, H, W) -> (H, W, C)
    return data.permute(1, 2, 0).numpy()


def _blank_image(hw: Tuple[int, int]) -> np.ndarray:
    h, w = hw
    return np.zeros((h, w, 3), dtype=np.float32)


def render_visualization_grid(
    model: MarioSequencePredictor,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    output_path: Path,
) -> None:
    model.eval()
    inputs = batch["inputs"][0].unsqueeze(0).to(device)
    actions = batch["actions"][0].unsqueeze(0).to(device)
    future_frames = batch["future_frames"][0].unsqueeze(0).to(device)
    raw_input_actions = batch["actions"][0].cpu().numpy()
    future_actions_tensor = batch.get("future_actions")
    if future_actions_tensor is not None:
        future_actions = future_actions_tensor[0].unsqueeze(0).to(device)
        raw_future_actions = future_actions_tensor[0].cpu().numpy()
    else:
        steps = future_frames.shape[1]
        future_actions = torch.zeros(
            (1, steps, actions.shape[-1]),
            dtype=actions.dtype,
            device=device,
        )
        raw_future_actions = None
    input_labels_batch = batch.get("input_labels")
    future_labels_batch = batch.get("future_labels")

    def _labels_for_sample(label_batch: Optional[List[List[str]]], sample_idx: int) -> Optional[List[str]]:
        """Return labels for the given sample.

        `label_batch` is expected to be a list (length M) of per-frame label sequences,
        each sequence containing N string labels; after PyTorch collation this corresponds
        to shape M×N (e.g., `[('traj_2/state_4', ...), ('traj_2/state_5', ...), ...]`).
        We select the sample at `sample_idx` by reading the same position across each
        sequence. Assertions guard against unexpected schema drift.
        """

        if label_batch is None:
            return None
        assert isinstance(label_batch, (list, tuple)) and label_batch, "Expected non-empty label batch"
        first = label_batch[0]
        assert isinstance(first, (list, tuple)), "Label batch should contain per-frame sequences"
        sequence_length = len(first)
        for seq in label_batch:
            assert isinstance(seq, (list, tuple)) and len(seq) == sequence_length, "Label sequences misaligned"
        if sample_idx >= sequence_length:
            return None
        return [str(seq[sample_idx]) for seq in label_batch]

    input_labels = _labels_for_sample(input_labels_batch, 0)
    future_labels = _labels_for_sample(future_labels_batch, 0)
    inputs = inputs.clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    outputs = model(
        inputs,
        actions,
        future_frames=future_frames,
        future_actions=future_actions,
    )
    predicted_frames = outputs.predicted_frames
    recon = outputs.input_reconstructions
    prediction_loss = F.smooth_l1_loss(predicted_frames, future_frames, reduction="sum")
    prediction_loss.backward()
    saliency = inputs.grad.detach().abs().sum(dim=2)
    model.zero_grad(set_to_none=True)

    with torch.no_grad():
        predicted_frames = predicted_frames.detach()
        predicted_skipless = model._decode_sequence(
            outputs.predicted_latents.detach(),
            (),
            chunk_size=model.decode_chunk,
        ).detach()
        if recon is not None:
            recon = recon.detach()
        saliency_flat = saliency.flatten(start_dim=2)
        max_vals = saliency_flat.max(dim=2, keepdim=True).values
        saliency_normalized = saliency_flat / (max_vals + 1e-8)
        saliency_maps = saliency_normalized.reshape_as(saliency).squeeze(0)
        input_imgs = inputs.detach()

    context_steps = inputs.shape[1]
    target_steps = future_frames.shape[1]
    rollout_steps = predicted_frames.shape[1]
    prediction_columns = max(target_steps, rollout_steps, 1)
    columns = max(context_steps + prediction_columns, context_steps + 1)
    rows = 5
    fig_width = max(16, columns * 2)
    fig, axes = plt.subplots(rows, columns, figsize=(fig_width, 11))
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    # Row 1: sequential frames (context + ground-truth futures)
    def _action_desc(action_vec: np.ndarray) -> str:
        binarized = (action_vec > 0.5).astype(np.uint8)
        return _describe_controller_vector(binarized)

    for i in range(min(context_steps, columns)):
        axes[0, i].imshow(_tensor_to_image(input_imgs[0, i]))
        labels = []
        if input_labels is not None and i < len(input_labels):
            labels.append(str(input_labels[i]))
        if i < raw_input_actions.shape[0]:
            labels.append(_action_desc(raw_input_actions[i]))
        if labels:
            axes[0, i].set_title("\n".join(labels), fontsize=8)

    prediction_offset = context_steps
    slot_capacity = columns - prediction_offset
    for i in range(min(target_steps, slot_capacity)):
        target_idx = i
        axes[0, prediction_offset + i].imshow(_tensor_to_image(future_frames[0, target_idx]))
        labels = []
        if future_labels is not None and i < len(future_labels):
            labels.append(str(future_labels[i]))
        if raw_future_actions is not None and i < raw_future_actions.shape[0]:
            labels.append(_action_desc(raw_future_actions[i]))
        if labels:
            axes[0, prediction_offset + i].set_title("\n".join(labels), fontsize=8)
    for col in range(prediction_offset + min(target_steps, slot_capacity), columns):
        axes[0, col].imshow(_blank_image(model.image_hw))

    # Row 2: reconstructed context frames, followed by blanks
    for i in range(min(context_steps, columns)):
        if recon is not None and i < recon.shape[1]:
            axes[1, i].imshow(_tensor_to_image(recon[0, i]))
        else:
            axes[1, i].imshow(_blank_image(model.image_hw))
    for i in range(columns - context_steps):
        axes[1, context_steps + i].imshow(_blank_image(model.image_hw))

    # Row 3: predictions occupy slots after context
    for i in range(min(context_steps, columns)):
        axes[2, i].imshow(_blank_image(model.image_hw))
    for i in range(min(rollout_steps, slot_capacity)):
        axes[2, prediction_offset + i].imshow(_tensor_to_image(predicted_frames[0, i]))

    # Row 4: saliency overlays on context frames, then blanks
    cmap = plt.get_cmap("magma")
    for i in range(min(context_steps, saliency_maps.shape[0], columns)):
        axes[3, i].imshow(_tensor_to_image(input_imgs[0, i]))
        if saliency_maps.shape[0] > i:
            axes[3, i].imshow(
                saliency_maps[i].cpu().numpy(),
                cmap=cmap,
                alpha=0.6,
                vmin=0.0,
                vmax=1.0,
            )
    for i in range(columns - context_steps):
        axes[3, context_steps + i].imshow(_blank_image(model.image_hw))

    # Row 5: predictions decoded without skip connections
    for i in range(min(context_steps, columns)):
        axes[4, i].imshow(_blank_image(model.image_hw))
    skipless_count = min(rollout_steps, columns - context_steps)
    for i in range(skipless_count):
        axes[4, context_steps + i].imshow(_tensor_to_image(predicted_skipless[0, i]))
    for col in range(context_steps + skipless_count, columns):
        axes[4, col].imshow(_blank_image(model.image_hw))

    fig.suptitle(
        "Inputs, targets, predictions, saliency, and skipless decodes (rows 1-5)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def prepare_dataloaders(config: TrainConfig) -> Tuple[DataLoader, Optional[DataLoader], MarioSequenceDataset]:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = Normalize(mean=mean, std=std)
    dataset = MarioSequenceDataset(
        config.data_root,
        image_hw=config.image_hw,
        context_steps=config.context_steps,
        future_steps=config.future_steps,
        normalize=normalize,
        max_trajs=config.max_trajs,
    )
    generator = torch.Generator().manual_seed(config.seed)
    if config.val_fraction <= 0.0 or len(dataset) < 10:
        train_dataset = dataset
        val_dataset = None
    else:
        val_count = max(1, int(len(dataset) * config.val_fraction))
        train_count = len(dataset) - val_count
        train_dataset, val_dataset = random_split(dataset, [train_count, val_count], generator=generator)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    return train_loader, val_loader, dataset


def train(config: TrainConfig) -> None:
    set_seed(config.seed)
    dirs = _ensure_dirs(config)
    run_dir = dirs["run"]
    logger.info("Writing outputs to %s", run_dir)
    train_loader, val_loader, full_dataset = prepare_dataloaders(config)
    device_pref = None
    if config.device is not None and str(config.device).lower() != "auto":
        device_pref = config.device
    device = pick_device(device_pref)
    logger.info("Using device %s", device)
    image_hw = full_dataset.image_hw or config.image_hw
    action_dim = full_dataset.action_dim or 1
    model = MarioSequencePredictor(
        action_dim=action_dim,
        base_channels=config.base_channels,
        latent_channels=config.latent_channels,
        image_hw=image_hw,
        decode_chunk=config.decode_chunk,
        reason_steps=config.reason_steps,
        reason_inner_steps=config.reason_inner_steps,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    start_epoch = 0
    global_step = 0
    best_loss = math.inf
    loss_history: List[Dict[str, float]] = []
    if config.resume:
        last_ckpt = dirs["checkpoints"] / "last.pt"
        if last_ckpt.exists():
            logger.info("Resuming from %s", last_ckpt)
            start_epoch, global_step, best_loss, loss_history = load_checkpoint(last_ckpt, model, optimizer)
            start_epoch += 1
            model.to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)

    logger.info(
        "Dataset size: %d samples (action_dim=%d, image_hw=%s)",
        len(train_loader.dataset),
        action_dim,
        image_hw,
    )
    for epoch in range(start_epoch, config.epochs):
        model.train()
        running = {
            "total": 0.0,
            "recon": 0.0,
            "latent": 0.0,
            "pred": 0.0,
            "residual": 0.0,
            "latent_cosine": 0.0,
            "skipless": 0.0,
            "skip": 0.0,
        }
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["inputs"].to(device)
            actions = batch["actions"].to(device)
            future = batch["future_frames"].to(device)
            future_actions = batch["future_actions"].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=config.mixed_precision):
                outputs = model(
                    inputs,
                    actions,
                    future_frames=future,
                    future_actions=future_actions,
                )
                losses = model.compute_losses(
                    outputs,
                    input_frames=inputs,
                    target_frames=future,
                    lambda_recon=config.lambda_recon,
                    lambda_latent=config.lambda_latent,
                    lambda_pred=config.lambda_pred,
                    lambda_residual=config.lambda_residual,
                    lambda_latent_cosine=config.lambda_latent_cosine,
                    lambda_skipless=config.lambda_skipless,
                    lambda_skip=config.lambda_skip,
                )
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()

            running["total"] += losses["total"].item()
            running["recon"] += losses["recon"].item()
            running["latent"] += losses["latent"].item()
            running["pred"] += losses["pred"].item()
            running["residual"] += losses["residual"].item()
            running["latent_cosine"] += losses["latent_cosine"].item()
            running["skipless"] += losses["skipless"].item()
            running["skip"] += losses["skip"].item()

            current_step = global_step + 1
            if config.log_every > 0 and current_step % config.log_every == 0:
                logger.info(
                    "Epoch %d Step %d | total %.4f recon %.4f latent %.4f pred %.4f residual %.4f latent_cos %.4f skipless %.4f skip %.4f",
                    epoch,
                    current_step,
                    losses["total"].item(),
                    losses["recon"].item(),
                    losses["latent"].item(),
                    losses["pred"].item(),
                    losses["residual"].item(),
                    losses["latent_cosine"].item(),
                    losses["skipless"].item(),
                    losses["skip"].item(),
                )
            loss_history.append(
                {
                    "step": current_step,
                    "total": float(losses["total"].item()),
                    "recon": float(losses["recon"].item()),
                    "latent": float(losses["latent"].item()),
                    "pred": float(losses["pred"].item()),
                    "residual": float(losses["residual"].item()),
                    "latent_cosine": float(losses["latent_cosine"].item()),
                    "skipless": float(losses["skipless"].item()),
                    "skip": float(losses["skip"].item()),
                }
            )
            if config.checkpoint_every > 0 and current_step % config.checkpoint_every == 0:
                save_checkpoint(
                    dirs["checkpoints"] / "last.pt",
                    model,
                    optimizer,
                    epoch,
                    current_step,
                    best_loss,
                    loss_history,
                )
            if config.vis_every > 0 and current_step % config.vis_every == 0:
                plot_loss_curves(loss_history, dirs["plots"])
                write_loss_csv(loss_history, dirs["plots"] / "loss_history.csv")

                vis_batch = batch
                vis_path = dirs["visualizations"] / f"step_{current_step:06d}.png"
                render_visualization_grid(model, vis_batch, device, vis_path)
                model.train()
            global_step = current_step

        steps_in_epoch = len(train_loader)
        avg_total = running["total"] / steps_in_epoch
        logger.info(
            "Epoch %d complete | avg_total %.4f avg_recon %.4f avg_latent %.4f avg_pred %.4f avg_residual %.4f avg_latent_cos %.4f avg_skipless %.4f avg_skip %.4f",
            epoch,
            avg_total,
            running["recon"] / steps_in_epoch,
            running["latent"] / steps_in_epoch,
            running["pred"] / steps_in_epoch,
            running["residual"] / steps_in_epoch,
            running["latent_cosine"] / steps_in_epoch,
            running["skipless"] / steps_in_epoch,
            running["skip"] / steps_in_epoch,
        )

        if avg_total < best_loss:
            best_loss = avg_total
            save_checkpoint(
                dirs["checkpoints"] / "best.pt",
                model,
                optimizer,
                epoch,
                global_step,
                best_loss,
                loss_history,
            )
            logger.info("Saved new best checkpoint (loss %.4f)", best_loss)

        save_checkpoint(
            dirs["checkpoints"] / "last.pt",
            model,
            optimizer,
            epoch,
            global_step,
            best_loss,
            loss_history,
        )

    save_checkpoint(
        dirs["checkpoints"] / "final.pt",
        model,
        optimizer,
        config.epochs - 1,
        global_step,
        best_loss,
        loss_history,
    )
    plot_loss_curves(loss_history, dirs["plots"])
    write_loss_csv(loss_history, dirs["plots"] / "loss_history.csv")
    config_payload = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(config).items()
    }
    config_payload["run_dir"] = str(run_dir)
    with (run_dir / "config.json").open("w") as fp:
        json.dump(config_payload, fp, indent=2)
    logger.info("Training complete.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = tyro.cli(TrainConfig)
    train(config)


if __name__ == "__main__":
    main()
