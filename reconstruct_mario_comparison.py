#!/usr/bin/env python3
"""Compare NES Mario frame reconstruction across pretrained and lightweight models.

Frozen ImageNet encoders pair with learned decoders while a fast autoencoder
trains end-to-end using a focal L1 objective to highlight small, hard examples.
"""
from __future__ import annotations

import logging
import random
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import time

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

from predict_mario_ms_ssim import default_transform, pick_device, unnormalize
from trajectory_utils import list_state_frames, list_traj_dirs


SCRIPT_START_TIME = time.time()


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.stem(x)
        h1 = self.down1(h0)
        h2 = self.down2(h1)
        h3 = self.down3(h2)
        b = self.bottleneck(h3)
        u1 = self.up1(b) + h2
        u2 = self.up2(u1) + h1
        u3 = self.up3(u2) + h0
        out = self.head(u3)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


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
        )
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 3)
        self.down3 = DownBlock(base_channels * 3, base_channels * 4)
        self.down4 = DownBlock(base_channels * 4, latent_channels)
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
        h1 = self.down1(h0)
        h2 = self.down2(h1)
        h3 = self.down3(h2)
        h4 = self.down4(h3)
        b = self.bottleneck(h4)
        u1 = self.up1(b) + h3
        u2 = self.up2(u1) + h2
        u3 = self.up3(u2) + h1
        u4 = self.up4(u3) + h0
        out = self.head(u4)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


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
    encoder = nn.Sequential(*layers)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


@torch.no_grad()
def _convnext_encoder(weights: ConvNeXt_Base_Weights) -> nn.Module:
    model = convnext_base(weights=weights)
    encoder = model.features  # (B,1024,7,7)
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
        self.loss_fn = nn.L1Loss()
        self.history: List[Tuple[int, float]] = []
        self.global_step = 0
        self.best_loss: Optional[float] = None

    def step(self, batch: torch.Tensor) -> Tuple[float, bool]:
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
        improved = self.best_loss is None or loss_val < self.best_loss
        if improved:
            self.best_loss = loss_val
        return loss_val, improved

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
        self.global_step = 0
        self.best_loss: Optional[float] = None

    def step(self, batch: torch.Tensor) -> Tuple[float, bool]:
        self.model.train()
        self.feature_extractor.eval()
        recon = self.model(batch)
        loss = self.loss_fn(recon, batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.global_step += 1
        loss_val = float(loss.detach().item())
        self.history.append((self.global_step, loss_val))
        improved = self.best_loss is None or loss_val < self.best_loss
        if improved:
            self.best_loss = loss_val
        return loss_val, improved

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
        self.global_step = 0
        self.best_loss: Optional[float] = None

    def step(self, batch: torch.Tensor) -> Tuple[float, bool]:
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
        improved = self.best_loss is None or loss_val < self.best_loss
        if improved:
            self.best_loss = loss_val
        return loss_val, improved

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
    col_titles = ["Input"] + [name for name, _ in unnorm_recons]
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
        plt.plot(steps, losses, label=trainer.name)
    plt.xlabel("Step")
    plt.ylabel("Reconstruction loss")
    plt.title("Model comparison losses")
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
    log_every: int = 20
    vis_every: int = 100
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_trainers(cfg: Config, device: torch.device) -> List[Trainer]:
    resnet_enc = _resnet_encoder(ResNet50_Weights.IMAGENET1K_V2)
    convnext_enc = _convnext_encoder(ConvNeXt_Base_Weights.IMAGENET1K_V1)
    resnet_dec = Decoder(2048)
    convnext_dec = Decoder(1024)
    autoencoder = LightweightAutoencoder()
    texture_autoencoder = TextureAwareAutoencoder()
    feature_layers = sorted(set(cfg.style_layers + (cfg.patch_layer,)))
    feature_extractor = StyleFeatureExtractor(feature_layers)
    return [
        ReconstructionTrainer("resnet50", resnet_enc, resnet_dec, device=device, lr=cfg.lr),
        ReconstructionTrainer("convnext_base", convnext_enc, convnext_dec, device=device, lr=cfg.lr),
        AutoencoderTrainer("focal_autoencoder", autoencoder, device=device, lr=cfg.lr),
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
        ),
    ]


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
            for trainer in trainers:
                loss, improved = trainer.step(batch)
                losses[trainer.name] = loss
                trainer.save_checkpoint(checkpoint_paths[trainer.name]["last"])
                if improved:
                    trainer.save_checkpoint(checkpoint_paths[trainer.name]["best"])
            if cfg.log_every > 0 and current_step % cfg.log_every == 0:
                loss_str = ", ".join(f"{name}: {losses[name]:.4f}" for name in losses)
                logger.info("[step %05d] %s", current_step, loss_str)
                plot_loss_histories(trainers, metrics_dir / "decoder_losses.png")
                write_loss_histories(trainers, metrics_dir)
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
    for trainer in trainers:
        paths = checkpoint_paths[trainer.name]
        trainer.save_checkpoint(paths["last"])
        if trainer.best_loss is not None and not paths["best"].exists():
            trainer.save_checkpoint(paths["best"])
        trainer.save_checkpoint(paths["final"])


if __name__ == "__main__":
    main()
