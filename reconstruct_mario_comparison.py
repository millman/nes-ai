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
)
import tyro
from PIL import Image

from predict_mario_ms_ssim import default_transform, pick_device, unnormalize
from reconstruct_comparison import (
    AutoencoderTrainer,
    BasicAutoencoderTrainer,
    BasicVectorAutoencoderTrainer,
    BestPracticeAutoencoderTrainer,
    BestPracticeVectorAutoencoderTrainer,
    Decoder,
    FocalMSSSIMAutoencoderUNetUNet,
    FocalMSSSIMLoss,
    FocalL1Loss,
    HardnessWeightedL1Loss,
    LightweightAutoencoder,
    LightweightAutoencoderPatch,
    LightweightAutoencoderUNet,
    LightweightAutoencoderUNetSkipTrain,
    LightweightFlatLatentAutoencoder,
    Mario4Autoencoder,
    Mario4LargeAutoencoder,
    Mario4MirroredAutoencoder,
    Mario4SpatialSoftmaxAutoencoder,
    Mario4SpatialSoftmaxLargeAutoencoder,
    ModernResNetAttnAutoencoder,
    MSSSIMAutoencoderUNet,
    MSSSIMLoss,
    ReconstructionTrainer,
    ResNetAutoencoder,
    ResNetV2Autoencoder,
    StyleContrastTrainer,
    StyleFeatureExtractor,
    TextureAwareAutoencoderUNet,
    compute_shared_metrics,
    ms_ssim_per_sample,
)
from reconstruct_comparison.convnext_encoder import _convnext_encoder
from reconstruct_comparison.resnet_encoder import _resnet_encoder
from trajectory_utils import list_state_frames, list_traj_dirs


SCRIPT_START_TIME = time.time()

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "resnet50": "ResNet-50 (MSE)",
    "convnext_base": "ConvNeXt-Base (MSE)",
    "autoencoder_l1": "Autoencoder (L1)",
    "autoencoder_mse": "Autoencoder (MSE)",
    "autoencoder_focal": "Autoencoder (Focal L1)",
    "autoencoder_smooth_l1": "Autoencoder (Smooth L1)",
    "autoencoder_cauchy": "Autoencoder (Cauchy)",
    "autoencoder_style_contrast": "Autoencoder (Style + PatchNCE)",
    "msssim_autoencoder": "Autoencoder (MS-SSIM)",
    "autoencoder_focal_msssim": "Autoencoder (Focal MS-SSIM)",
    "autoencoder_spatial_latent": "Autoencoder (Spatial Latent)",
    "autoencoder_patch": "Autoencoder (No Skip Patch)",
    "autoencoder_skip_train": "Autoencoder (Train Skip, Eval Zero)",
    "autoencoder_resnet": "Autoencoder (ResNet Blocks)",
    "autoencoder_resnetv2": "Autoencoder (ResNet v2)",
    "autoencoder_modern_attn": "Autoencoder (Modern ResNet + Attn)",
    "autoencoder_mario4": "Mario4 Autoencoder",
    "mario4_mirrored_decoder": "Mario4 Mirrored Decoder",
    "mario4_spatial_softmax_192": "Mario4 Spatial Softmax 192",
    "mario4_latent_1024": "Mario4 Latent 1024",
    "mario4_spatial_softmax_1024": "Mario4 Spatial Softmax 1024",
    "autoencoder_lightweight_flat_latent": "Lightweight Flat Latent",
    "basic_autoencoder": "Autoencoder (Basic)",
    "basic_vector_autoencoder": "Autoencoder (Basic Vector)",
    "basic_autoencoder_mse": "Basic Autoencoder (MSE)",
    "basic_autoencoder_l1": "Basic Autoencoder (L1)",
    "basic_autoencoder_focal": "Basic Autoencoder (Focal L1)",
    "basic_autoencoder_hardness": "Basic Autoencoder (Hardness Weighted)",
    "best_practice_autoencoder": "Autoencoder (Best Practice)",
    "best_practice_vector_autoencoder": "Autoencoder (Best Practice Vector)",
}

@dataclass(frozen=True)
class TrainerInfo:
    flag: str
    model_key: str
    loss: str


TRAINER_INFOS: Tuple[TrainerInfo, ...] = (
    TrainerInfo("enable_pretrained_resnet50", "resnet50", "MSELoss"),
    TrainerInfo("enable_pretrained_convnext", "convnext_base", "MSELoss"),
    TrainerInfo("enable_unet_mse", "autoencoder_mse", "MSELoss"),
    TrainerInfo("enable_unet_l1", "autoencoder_l1", "L1Loss"),
    TrainerInfo("enable_unet_smoothl1", "autoencoder_smooth_l1", "SmoothL1Loss"),
    TrainerInfo("enable_unet_focal", "autoencoder_focal", "FocalL1Loss"),
    TrainerInfo("enable_unet_style_contrast", "autoencoder_style_contrast", "Style + PatchNCE"),
    TrainerInfo("enable_unet_cauchy", "autoencoder_cauchy", "CauchyLoss"),
    TrainerInfo("enable_ae_focal", "autoencoder_spatial_latent", "FocalL1Loss"),
    TrainerInfo("enable_ae_patch_mse", "autoencoder_patch", "MultiScalePatchLoss"),
    TrainerInfo("enable_ae_skip_train", "autoencoder_skip_train", "FocalL1Loss"),
    TrainerInfo("enable_resnet", "autoencoder_resnet", "SmoothL1Loss"),
    TrainerInfo("enable_resnetv2", "autoencoder_resnetv2", "SmoothL1Loss"),
    TrainerInfo("enable_modern_attn", "autoencoder_modern_attn", "SmoothL1Loss"),
    TrainerInfo("enable_ae_flat_l1", "autoencoder_lightweight_flat_latent", "SmoothL1Loss"),
    TrainerInfo("enable_mario4", "autoencoder_mario4", "SmoothL1Loss"),
    TrainerInfo("enable_mario4_mirrored", "mario4_mirrored_decoder", "SmoothL1Loss"),
    TrainerInfo("enable_mario4_spatial_softmax_192", "mario4_spatial_softmax_192", "SmoothL1Loss"),
    TrainerInfo("enable_mario4_1024", "mario4_latent_1024", "SmoothL1Loss"),
    TrainerInfo("enable_mario4_spatial_softmax_1024", "mario4_spatial_softmax_1024", "SmoothL1Loss"),
    TrainerInfo("enable_unet_msssim", "msssim_autoencoder", "MSSSIMLoss"),
    TrainerInfo("enable_unet_focal_msssim", "autoencoder_focal_msssim", "FocalMSSSIMLoss"),
    TrainerInfo("enable_basic_mse", "basic_autoencoder_mse", "MSELoss"),
    TrainerInfo("enable_basic_l1", "basic_autoencoder_l1", "L1Loss"),
    TrainerInfo("enable_basic_focal", "basic_autoencoder_focal", "FocalL1Loss"),
    TrainerInfo(
        "enable_basic_hardness",
        "basic_autoencoder_hardness",
        "HardnessWeightedL1Loss",
    ),
    TrainerInfo("enable_basic_flat_mse", "basic_vector_autoencoder", "MSELoss"),
    TrainerInfo("enable_best_focal", "best_practice_autoencoder", "FocalL1Loss"),
    TrainerInfo(
        "enable_best_flat_focal",
        "best_practice_vector_autoencoder",
        "FocalL1Loss",
    ),
)


def _display_name(name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(name, name)


def _cli_flag_name(flag: str) -> str:
    return f"--{flag.replace('_', '-')}"


def _print_encoder_table(cfg: "Config") -> None:
    headers = ("Enabled?", "Name", "cli arg", "loss function")
    rows: List[Tuple[str, str, str, str]] = []
    for info in TRAINER_INFOS:
        enabled = getattr(cfg, info.flag)
        rows.append(
            (
                "Yes" if enabled else "No",
                _display_name(info.model_key),
                _cli_flag_name(info.flag),
                info.loss,
            )
        )
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    print("Available encoders:")
    header_line = " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    separator = "-+-".join("-" * widths[idx] for idx in range(len(headers)))
    print(header_line)
    print(separator)
    for row in rows:
        print(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))
    print()


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

# ---------------------------------------------------------------------------
# Trainer wrapper
# ---------------------------------------------------------------------------


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

    # Pretrained
    enable_pretrained_resnet50: bool = False
    enable_pretrained_convnext: bool = False

    # UNets
    enable_unet_l1: bool = False
    enable_unet_smoothl1: bool = False
    enable_unet_mse: bool = False
    enable_unet_focal: bool = False
    enable_unet_style_contrast: bool = False
    enable_unet_cauchy: bool = False
    enable_unet_msssim: bool = False
    enable_unet_focal_msssim: bool = False

    # Basic autoencoders
    enable_basic_mse: bool = False
    enable_basic_l1: bool = False
    enable_basic_focal: bool = False
    enable_basic_hardness: bool = False
    enable_basic_flat_mse: bool = False

    # Standard/Lightweight autoencoders
    enable_ae_focal: bool = False
    enable_ae_flat_l1: bool = False
    enable_ae_patch_mse: bool = False
    enable_ae_skip_train: bool = False

    # Heavier "best-practice" autoencoders
    enable_best_focal: bool = False
    enable_best_flat_focal: bool = False

    # Experimental other autoencoders
    enable_resnet: bool = False
    enable_resnetv2: bool = False
    enable_modern_attn: bool = False

    # Autoencoders based on the current predict_mario4 configuration
    enable_mario4: bool = False
    enable_mario4_mirrored: bool = False
    enable_mario4_spatial_softmax_192: bool = False
    enable_mario4_1024: bool = False
    enable_mario4_spatial_softmax_1024: bool = False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_trainers(cfg: Config, device: torch.device) -> List[Trainer]:
    trainers: List[Trainer] = []
    if cfg.enable_pretrained_resnet50:
        resnet_enc = _resnet_encoder(ResNet50_Weights.IMAGENET1K_V2)
        resnet_dec = Decoder(2048)
        trainers.append(
            ReconstructionTrainer("resnet50", resnet_enc, resnet_dec, device=device, lr=cfg.lr)
        )
    if cfg.enable_pretrained_convnext:
        convnext_enc = _convnext_encoder(ConvNeXt_Base_Weights.IMAGENET1K_V1)
        convnext_dec = Decoder(1024)
        trainers.append(
            ReconstructionTrainer(
                "convnext_base", convnext_enc, convnext_dec, device=device, lr=cfg.lr
            )
        )
    if cfg.enable_unet_mse:
        trainers.append(
            AutoencoderTrainer(
                name="autoencoder_mse",
                model=LightweightAutoencoderUNet(),
                device=device,
                lr=cfg.lr,
                loss_fn=nn.MSELoss(),
            )
        )
    if cfg.enable_unet_l1:
        trainers.append(
            AutoencoderTrainer(
                name="autoencoder_l1",
                model=LightweightAutoencoderUNet(),
                device=device,
                lr=cfg.lr,
                loss_fn=nn.L1Loss(),
            )
        )
    if cfg.enable_unet_smoothl1:
        trainers.append(
            AutoencoderTrainer(
                name="autoencoder_smooth_l1",
                model=LightweightAutoencoderUNet(),
                device=device,
                lr=cfg.lr,
                loss_fn=nn.SmoothL1Loss(),
            )
        )
    if cfg.enable_unet_focal:
        trainers.append(
            AutoencoderTrainer(
                name="autoencoder_focal",
                model=LightweightAutoencoderUNet(),
                device=device,
                lr=cfg.lr,
                loss_fn=FocalL1Loss(),
            )
        )
    if cfg.enable_unet_style_contrast:
        texture_autoencoder = TextureAwareAutoencoderUNet()
        feature_layers = sorted(set(cfg.style_layers + (cfg.patch_layer,)))
        feature_extractor = StyleFeatureExtractor(feature_layers)
        trainers.append(
            StyleContrastTrainer(
                "autoencoder_style_contrast",
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
    if cfg.enable_unet_cauchy:
        trainers.append(
            AutoencoderTrainer(
                name="autoencoder_cauchy",
                model=LightweightAutoencoderUNet(),
                device=device,
                lr=cfg.lr,
                loss_fn=CauchyLoss(),
            )
        )
    if cfg.enable_ae_focal:
        trainers.append(
            AutoencoderTrainer(
                name="autoencoder_spatial_latent",
                model=LightweightAutoencoder(),
                device=device,
                lr=cfg.lr,
                loss_fn=FocalL1Loss(),
            )
        )
    if cfg.enable_ae_patch_mse:
        trainers.append(
            AutoencoderTrainer(
                name="autoencoder_patch",
                model=LightweightAutoencoderPatch(),
                device=device,
                lr=cfg.lr,
                loss_fn=MultiScalePatchLoss(),
            )
        )
    if cfg.enable_ae_skip_train:
        trainers.append(
            AutoencoderTrainer(
                name="autoencoder_skip_train",
                model=LightweightAutoencoderUNetSkipTrain(),
                device=device,
                lr=cfg.lr,
                loss_fn=FocalL1Loss(),
            )
        )
    if cfg.enable_resnet:
        trainers.append(
            AutoencoderTrainer(
                name="autoencoder_resnet",
                model=ResNetAutoencoder(),
                device=device,
                lr=cfg.lr,
                loss_fn=nn.SmoothL1Loss(),
            )
        )
    if cfg.enable_resnetv2:
        trainers.append(
            AutoencoderTrainer(
                name="autoencoder_resnetv2",
                model=ResNetV2Autoencoder(),
                device=device,
                lr=cfg.lr,
                loss_fn=nn.SmoothL1Loss(),
            )
        )
    if cfg.enable_modern_attn:
        trainers.append(
            AutoencoderTrainer(
                name="autoencoder_modern_attn",
                model=ModernResNetAttnAutoencoder(),
                device=device,
                lr=cfg.lr,
                loss_fn=nn.SmoothL1Loss(),
            )
        )
    if cfg.enable_ae_flat_l1:
        trainers.append(
            AutoencoderTrainer(
                name="autoencoder_lightweight_flat_latent",
                model=LightweightFlatLatentAutoencoder(),
                device=device,
                lr=cfg.lr,
                loss_fn=nn.SmoothL1Loss(),
            )
        )
    if cfg.enable_mario4:
        trainers.append(
            AutoencoderTrainer(
                name="autoencoder_mario4",
                model=Mario4Autoencoder(),
                device=device,
                lr=cfg.lr,
                loss_fn=nn.SmoothL1Loss(),
            )
        )
    if cfg.enable_mario4_mirrored:
        trainers.append(
            AutoencoderTrainer(
                name="mario4_mirrored_decoder",
                model=Mario4MirroredAutoencoder(),
                device=device,
                lr=cfg.lr,
                loss_fn=nn.SmoothL1Loss(),
            )
        )
    if cfg.enable_mario4_spatial_softmax_192:
        trainers.append(
            AutoencoderTrainer(
                name="mario4_spatial_softmax_192",
                model=Mario4SpatialSoftmaxAutoencoder(),
                device=device,
                lr=cfg.lr,
                loss_fn=nn.SmoothL1Loss(),
            )
        )
    if cfg.enable_mario4_1024:
        trainers.append(
            AutoencoderTrainer(
                name="mario4_latent_1024",
                model=Mario4LargeAutoencoder(),
                device=device,
                lr=cfg.lr,
                loss_fn=nn.SmoothL1Loss(),
            )
        )
    if cfg.enable_mario4_spatial_softmax_1024:
        trainers.append(
            AutoencoderTrainer(
                name="mario4_spatial_softmax_1024",
                model=Mario4SpatialSoftmaxLargeAutoencoder(),
                device=device,
                lr=cfg.lr,
                loss_fn=nn.SmoothL1Loss(),
            )
        )
    if cfg.enable_unet_msssim:
        trainers.append(
            AutoencoderTrainer(
                name="msssim_autoencoder",
                model=MSSSIMAutoencoderUNet(),
                device=device,
                lr=cfg.lr,
                loss_fn=MSSSIMLoss(),
            )
        )
    if cfg.enable_unet_focal_msssim:
        trainers.append(
            AutoencoderTrainer(
                name="autoencoder_focal_msssim",
                model=FocalMSSSIMAutoencoderUNetUNet(),
                device=device,
                lr=cfg.lr,
                loss_fn=FocalMSSSIMLoss(),
            )
        )
    basic_variants: Tuple[Tuple[bool, str, type[nn.Module]], ...] = (
        (cfg.enable_basic_mse, "basic_autoencoder_mse", nn.MSELoss),
        (cfg.enable_basic_l1, "basic_autoencoder_l1", nn.L1Loss),
        (cfg.enable_basic_focal, "basic_autoencoder_focal", FocalL1Loss),
        (
            cfg.enable_basic_hardness,
            "basic_autoencoder_hardness",
            HardnessWeightedL1Loss,
        ),
    )
    for enabled, trainer_name, loss_ctor in basic_variants:
        if not enabled:
            continue
        trainers.append(
            BasicAutoencoderTrainer(
                name=trainer_name,
                device=device,
                lr=cfg.lr,
                loss_fn=loss_ctor(),
                weight_decay=0.0,
            )
        )
    if cfg.enable_basic_flat_mse:
        trainers.append(
            BasicVectorAutoencoderTrainer(
                device=device,
                lr=cfg.lr,
                loss_fn=nn.MSELoss(),
                weight_decay=0.0,
            )
        )
    if cfg.enable_best_focal:
        trainers.append(
            BestPracticeAutoencoderTrainer(
                device=device,
                lr=cfg.lr,
                loss_fn=FocalL1Loss(),
            )
        )
    if cfg.enable_best_flat_focal:
        trainers.append(
            BestPracticeVectorAutoencoderTrainer(
                device=device,
                lr=cfg.lr,
                loss_fn=FocalL1Loss(),
            )
        )
    if not trainers:
        raise ValueError("No trainers enabled. Enable at least one model to proceed.")
    return trainers


def main() -> None:
    cfg = tyro.cli(Config)
    _print_encoder_table(cfg)
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
        module = getattr(trainer, "model", None)
        if module is None:
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
