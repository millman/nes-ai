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
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, Mapping
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
    BasicAutoencoder,
    BasicFlatAutoencoder,
    BestPracticeAutoencoderTrainer,
    BestPracticeVectorAutoencoderTrainer,
    Decoder,
    FocalMSSSIMAutoencoderUNet,
    FocalMSSSIMLoss,
    FocalL1Loss,
    FocalSpatialSoftmaxLoss,
    HardnessWeightedL1Loss,
    CauchyLoss,
    MultiScalePatchLoss,
    LightweightAutoencoder,
    LightweightAutoencoderPatch,
    LightweightAutoencoderUNet,
    LightweightAutoencoderUNetSkipTrain,
    LightweightFlatAutoencoder,
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

LossSpec = Union[str, type[nn.Module]]


@dataclass(frozen=True)
class TrainerInfo:
    model_key: str
    loss: LossSpec
    description: str


def _trainer_flag(model_key: str) -> str:
    return f"enable_{model_key}"


TRAINER_INFOS: Tuple[TrainerInfo, ...] = (
    TrainerInfo(model_key="ae_flat_l1", loss=nn.SmoothL1Loss, description="Autoencoder Flat (L1)",),
    TrainerInfo(model_key="ae_flat_focal", loss=FocalL1Loss, description="Autoencoder Flat (Focal)"),
    TrainerInfo(model_key="ae_flat_hardness", loss=HardnessWeightedL1Loss, description="Autoencoder Flat (Hardness)"),
    TrainerInfo(model_key="ae_focal", loss=FocalL1Loss, description="Autoencoder (Focal)"),
    TrainerInfo(model_key="ae_l1", loss=nn.SmoothL1Loss, description="Autoencoder (L1)"),
    TrainerInfo(model_key="ae_hardness", loss=HardnessWeightedL1Loss, description="Autoencoder (Hardness)"),
    TrainerInfo(model_key="ae_patch_mse", loss=MultiScalePatchLoss, description="Autoencoder (No Skip Patch)"),
    TrainerInfo(model_key="ae_skip_train", loss=FocalL1Loss, description="Autoencoder (Train Skip, Eval Zero)"),
    TrainerInfo(model_key="basic_flat_mse", loss=nn.MSELoss, description="Basic Flat (MSE)"),
    TrainerInfo(model_key="basic_flat_l1", loss=nn.SmoothL1Loss, description="Basic Flat (L1)"),
    TrainerInfo(model_key="basic_flat_focal", loss=FocalL1Loss, description="Basic Flat (Focal)"),
    TrainerInfo(model_key="basic_flat_hardness", loss=HardnessWeightedL1Loss, description="Basic Flat (Hardness)"),
    TrainerInfo(model_key="basic_flat_focal_spatial", loss=FocalSpatialSoftmaxLoss, description="Basic Flat (Focal + Spatial)",),
    TrainerInfo(model_key="basic_focal", loss=FocalL1Loss, description="Basic (Focal)"),
    TrainerInfo(model_key="basic_hardness", loss=HardnessWeightedL1Loss, description="Basic (Hardness)"),
    TrainerInfo(model_key="basic_l1", loss=nn.SmoothL1Loss, description="Basic Autoencoder (L1)"),
    TrainerInfo(model_key="basic_mse", loss=nn.MSELoss, description="Basic Autoencoder (MSE)"),
    TrainerInfo(model_key="best_flat_focal", loss=FocalL1Loss, description="Autoencoder (Best Practice Flat)"),
    TrainerInfo(model_key="best_focal", loss=FocalL1Loss, description="Autoencoder (Best Practice)"),
    TrainerInfo(model_key="mario4_1024", loss=nn.SmoothL1Loss, description="Mario4 Latent 1024"),
    TrainerInfo(model_key="mario4_mirrored", loss=nn.SmoothL1Loss, description="Mario4 Mirrored Decoder"),
    TrainerInfo(model_key="mario4_spatial_softmax_1024", loss=nn.SmoothL1Loss, description="Mario4 Spatial Softmax 1024"),
    TrainerInfo(model_key="mario4_spatial_softmax_192", loss=nn.SmoothL1Loss, description="Mario4 Spatial Softmax 192"),
    TrainerInfo(model_key="mario4", loss=nn.SmoothL1Loss, description="Mario4 Autoencoder"),
    TrainerInfo(model_key="modern_attn", loss=nn.SmoothL1Loss, description="Autoencoder (Modern ResNet + Attn)"),
    TrainerInfo(model_key="pretrained_convnext", loss=nn.MSELoss, description="ConvNeXt-Base (MSE)"),
    TrainerInfo(model_key="pretrained_resnet50", loss=nn.MSELoss, description="ResNet-50 (MSE)"),
    TrainerInfo(model_key="resnet", loss=nn.SmoothL1Loss, description="Autoencoder (ResNet Blocks)"),
    TrainerInfo(model_key="resnetv2", loss=nn.SmoothL1Loss, description="Autoencoder (ResNet v2)"),
    TrainerInfo(model_key="unet_cauchy", loss=CauchyLoss, description="Autoencoder (Cauchy)"),
    TrainerInfo(model_key="unet_focal_msssim", loss=FocalMSSSIMLoss, description="Autoencoder (Focal MS-SSIM)"),
    TrainerInfo(model_key="unet_focal", loss=FocalL1Loss, description="Autoencoder (Focal L1)"),
    TrainerInfo(model_key="unet_l1", loss=nn.SmoothL1Loss, description="Autoencoder (L1)"),
    TrainerInfo(model_key="unet_mse", loss=nn.MSELoss, description="Autoencoder (MSE)"),
    TrainerInfo(model_key="unet_msssim", loss=MSSSIMLoss, description="Autoencoder (MS-SSIM)"),
    TrainerInfo(model_key="unet_smoothl1", loss=nn.SmoothL1Loss, description="Autoencoder (Smooth L1)"),
    TrainerInfo(model_key="unet_style_contrast", loss="Style + PatchNCE", description="Autoencoder (Style + PatchNCE)"),
)


TRAINER_INFO_MAP: Dict[str, TrainerInfo] = {info.model_key: info for info in TRAINER_INFOS}


def _display_name(name: str) -> str:
    info = TRAINER_INFO_MAP.get(name)
    if info is None:
        return name
    return info.description


def _cli_flag_name(flag: str) -> str:
    return f"--{flag.replace('_', '-')}"


def _print_encoder_table(cfg: "Config") -> None:
    headers = ("Enabled?", "Name", "cli arg", "loss function")
    rows: List[Tuple[str, str, str, str]] = []
    for info in TRAINER_INFOS:
        flag = _trainer_flag(info.model_key)
        enabled = getattr(cfg, flag)
        rows.append(
            (
                "Yes" if enabled else "No",
                _display_name(info.model_key),
                _cli_flag_name(flag),
                info.loss if isinstance(info.loss, str) else info.loss.__name__,
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
    inputs_cpu = inputs.detach().cpu()
    recon_cpu = [(name, tensor.detach().cpu()) for name, tensor in reconstructions]
    image_count = inputs.shape[0]
    rows = max(1, image_count * 2)
    cols = 1 + len(recon_cpu)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1:
        axes = axes[None, :]
    unnorm_inputs = unnormalize(inputs_cpu)
    unnorm_recons = [(name, unnormalize(tensor)) for name, tensor in recon_cpu]
    col_titles = ["Input"] + [_display_name(name) for name, _ in unnorm_recons]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title)

    loss_maps: Dict[Tuple[int, int], torch.Tensor] = {}
    max_loss_value = 0.0
    for img_idx in range(image_count):
        target = unnorm_inputs[img_idx]
        for col_idx, (_, tensor) in enumerate(unnorm_recons):
            recon = tensor[img_idx]
            diff = torch.abs(target - recon).mean(dim=0)
            diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
            loss_maps[(img_idx, col_idx)] = diff
            max_loss_value = max(max_loss_value, float(diff.max().item()))
    if max_loss_value == 0.0:
        max_loss_value = 1.0

    for img_idx in range(image_count):
        recon_row = img_idx * 2
        loss_row = min(rows - 1, recon_row + 1)
        axes[recon_row, 0].imshow(_tensor_to_numpy(unnorm_inputs[img_idx]))
        axes[recon_row, 0].axis("off")
        axes[loss_row, 0].axis("off")
        axes[loss_row, 0].text(
            0.5,
            0.5,
            "Loss",
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
        )
        for col, (_, tensor) in enumerate(unnorm_recons, start=1):
            axes[recon_row, col].imshow(_tensor_to_numpy(tensor[img_idx]))
            axes[recon_row, col].axis("off")
            loss_map = loss_maps[(img_idx, col - 1)] / max_loss_value
            axes[loss_row, col].imshow(loss_map.cpu(), cmap="inferno")
            axes[loss_row, col].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loss_histories(trainers: Dict[str, Trainer], out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for key, trainer in trainers.items():
        if not trainer.history:
            continue
        steps, losses = zip(*trainer.history)
        plt.plot(steps, losses, label=_display_name(key))
    plt.xlabel("Step")
    plt.ylabel("Reconstruction loss")
    plt.title("Model comparison losses (log scale)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_loss_histories(trainers: Dict[str, Trainer], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, trainer in trainers.items():
        history_path = out_dir / f"{key}_loss.csv"
        with history_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss"])
            for step, loss in trainer.history:
                writer.writerow([step, loss])


def write_shared_metric_histories(trainers: Dict[str, Trainer], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, trainer in trainers.items():
        history = getattr(trainer, "shared_history", [])
        if not history:
            continue
        history_path = out_dir / f"{key}_shared_metrics.csv"
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


def plot_shared_metric_histories(trainers: Dict[str, Trainer], out_dir: Path) -> None:
    has_data = any(getattr(trainer, "shared_history", []) for trainer in trainers.values())
    if not has_data:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot L1
    plt.figure(figsize=(8, 5))
    for key, trainer in trainers.items():
        history = getattr(trainer, "shared_history", [])
        if not history:
            continue
        steps = [item[0] for item in history]
        l1_values = [item[1]["l1"] for item in history]
        plt.plot(steps, l1_values, label=_display_name(key))
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
    for key, trainer in trainers.items():
        history = getattr(trainer, "shared_history", [])
        if not history:
            continue
        steps = [item[0] for item in history]
        ms_values = [item[1]["ms_ssim"] for item in history]
        plt.plot(steps, ms_values, label=_display_name(key))
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

    # Basic autoencoders
    enable_basic_mse: bool = False
    enable_basic_l1: bool = False
    enable_basic_focal: bool = False
    enable_basic_hardness: bool = False
    enable_basic_flat_mse: bool = False
    enable_basic_flat_l1: bool = False
    enable_basic_flat_focal: bool = False
    enable_basic_flat_focal_spatial: bool = False
    enable_basic_flat_hardness: bool = False

    # Standard/Lightweight autoencoders
    enable_ae_focal: bool = False
    enable_ae_l1: bool = False
    enable_ae_hardness: bool = False
    enable_ae_flat_l1: bool = False
    enable_ae_flat_focal: bool = False
    enable_ae_flat_hardness: bool = False
    enable_ae_patch_mse: bool = False
    enable_ae_skip_train: bool = False

    # Heavier "best-practice" autoencoders
    enable_best_focal: bool = False
    enable_best_flat_focal: bool = False

    # Autoencoders based on the current predict_mario4 configuration
    enable_mario4: bool = False
    enable_mario4_mirrored: bool = False
    enable_mario4_spatial_softmax_192: bool = False
    enable_mario4_1024: bool = False
    enable_mario4_spatial_softmax_1024: bool = False

    # Experimental other autoencoders
    enable_resnet: bool = False
    enable_resnetv2: bool = False
    enable_modern_attn: bool = False

    # UNets
    enable_unet_l1: bool = False
    enable_unet_smoothl1: bool = False
    enable_unet_mse: bool = False
    enable_unet_focal: bool = False
    enable_unet_style_contrast: bool = False
    enable_unet_cauchy: bool = False
    enable_unet_msssim: bool = False
    enable_unet_focal_msssim: bool = False


def _verify_trainer_config_alignment() -> None:
    config_flags = {
        name[len("enable_") :]
        for name in Config.__dataclass_fields__
        if name.startswith("enable_")
    }
    info_keys = {info.model_key for info in TRAINER_INFOS}
    missing = info_keys - config_flags
    if missing:
        raise RuntimeError(
            "TrainerInfo keys missing matching Config enable_ flags: "
            + ", ".join(sorted(missing))
        )


_verify_trainer_config_alignment()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_trainers(
    cfg: Config,
    device: torch.device,
    trainer_infos: Mapping[str, TrainerInfo],
) -> Dict[str, Trainer]:
    trainers: Dict[str, Trainer] = {}
    if cfg.enable_pretrained_resnet50:
        resnet_enc = _resnet_encoder(ResNet50_Weights.IMAGENET1K_V2)
        resnet_dec = Decoder(2048)
        trainer = ReconstructionTrainer(
            resnet_enc,
            resnet_dec,
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["pretrained_resnet50"].loss(),
        )
        trainers["pretrained_resnet50"] = trainer
    if cfg.enable_pretrained_convnext:
        convnext_enc = _convnext_encoder(ConvNeXt_Base_Weights.IMAGENET1K_V1)
        convnext_dec = Decoder(1024)
        trainer = ReconstructionTrainer(
            convnext_enc,
            convnext_dec,
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["pretrained_convnext"].loss(),
        )
        trainers["pretrained_convnext"] = trainer
    if cfg.enable_unet_mse:
        trainer = AutoencoderTrainer(
            model=LightweightAutoencoderUNet(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["unet_mse"].loss(),
        )
        trainers["unet_mse"] = trainer
    if cfg.enable_unet_l1:
        trainer = AutoencoderTrainer(
            model=LightweightAutoencoderUNet(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["unet_l1"].loss(),
        )
        trainers["unet_l1"] = trainer
    if cfg.enable_unet_smoothl1:
        trainer = AutoencoderTrainer(
            model=LightweightAutoencoderUNet(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["unet_smoothl1"].loss(),
        )
        trainers["unet_smoothl1"] = trainer
    if cfg.enable_unet_focal:
        trainer = AutoencoderTrainer(
            model=LightweightAutoencoderUNet(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["unet_focal"].loss(),
        )
        trainers["unet_focal"] = trainer
    if cfg.enable_unet_style_contrast:
        texture_autoencoder = TextureAwareAutoencoderUNet()
        feature_layers = sorted(set(cfg.style_layers + (cfg.patch_layer,)))
        feature_extractor = StyleFeatureExtractor(feature_layers)
        trainer = StyleContrastTrainer(
            "unet_style_contrast",
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
        trainers["unet_style_contrast"] = trainer
    if cfg.enable_unet_cauchy:
        trainer = AutoencoderTrainer(
            model=LightweightAutoencoderUNet(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["unet_cauchy"].loss(),
        )
        trainers["unet_cauchy"] = trainer
    if cfg.enable_ae_focal:
        trainer = AutoencoderTrainer(
            model=LightweightAutoencoder(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["ae_focal"].loss(),
        )
        trainers["ae_focal"] = trainer
    if cfg.enable_ae_l1:
        trainer = AutoencoderTrainer(
            model=LightweightAutoencoder(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["ae_l1"].loss(),
        )
        trainers["ae_l1"] = trainer
    if cfg.enable_ae_hardness:
        trainer = AutoencoderTrainer(
            model=LightweightAutoencoder(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["ae_hardness"].loss(),
        )
        trainers["ae_hardness"] = trainer
    if cfg.enable_ae_patch_mse:
        trainer = AutoencoderTrainer(
            model=LightweightAutoencoderPatch(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["ae_patch_mse"].loss(),
        )
        trainers["ae_patch_mse"] = trainer
    if cfg.enable_ae_skip_train:
        trainer = AutoencoderTrainer(
            model=LightweightAutoencoderUNetSkipTrain(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["ae_skip_train"].loss(),
        )
        trainers["ae_skip_train"] = trainer
    if cfg.enable_resnet:
        trainer = AutoencoderTrainer(
            model=ResNetAutoencoder(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["resnet"].loss(),
        )
        trainers["resnet"] = trainer
    if cfg.enable_resnetv2:
        trainer = AutoencoderTrainer(
            model=ResNetV2Autoencoder(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["resnetv2"].loss(),
        )
        trainers["resnetv2"] = trainer
    if cfg.enable_modern_attn:
        trainer = AutoencoderTrainer(
            model=ModernResNetAttnAutoencoder(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["modern_attn"].loss(),
        )
        trainers["modern_attn"] = trainer
    if cfg.enable_ae_flat_l1:
        trainer = AutoencoderTrainer(
            model=LightweightFlatAutoencoder(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["ae_flat_l1"].loss(),
        )
        trainers["ae_flat_l1"] = trainer
    if cfg.enable_ae_flat_focal:
        trainer = AutoencoderTrainer(
            model=LightweightFlatAutoencoder(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["ae_flat_focal"].loss(),
        )
        trainers["ae_flat_focal"] = trainer
    if cfg.enable_ae_flat_hardness:
        trainer = AutoencoderTrainer(
            model=LightweightFlatAutoencoder(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["ae_flat_hardness"].loss(),
        )
        trainers["ae_flat_hardness"] = trainer
    if cfg.enable_mario4:
        trainer = AutoencoderTrainer(
            model=Mario4Autoencoder(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["mario4"].loss(),
        )
        trainers["mario4"] = trainer
    if cfg.enable_mario4_mirrored:
        trainer = AutoencoderTrainer(
            model=Mario4MirroredAutoencoder(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["mario4_mirrored"].loss(),
        )
        trainers["mario4_mirrored"] = trainer
    if cfg.enable_mario4_spatial_softmax_192:
        trainer = AutoencoderTrainer(
            model=Mario4SpatialSoftmaxAutoencoder(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["mario4_spatial_softmax_192"].loss(),
        )
        trainers["mario4_spatial_softmax_192"] = trainer
    if cfg.enable_mario4_1024:
        trainer = AutoencoderTrainer(
            model=Mario4LargeAutoencoder(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["mario4_1024"].loss(),
        )
        trainers["mario4_1024"] = trainer
    if cfg.enable_mario4_spatial_softmax_1024:
        trainer = AutoencoderTrainer(
            model=Mario4SpatialSoftmaxLargeAutoencoder(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["mario4_spatial_softmax_1024"].loss(),
        )
        trainers["mario4_spatial_softmax_1024"] = trainer
    if cfg.enable_unet_msssim:
        trainer = AutoencoderTrainer(
            model=MSSSIMAutoencoderUNet(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["unet_msssim"].loss(),
        )
        trainers["unet_msssim"] = trainer
    if cfg.enable_unet_focal_msssim:
        trainer = AutoencoderTrainer(
            model=FocalMSSSIMAutoencoderUNet(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["unet_focal_msssim"].loss(),
        )
        trainers["unet_focal_msssim"] = trainer
    basic_variants: Tuple[Tuple[bool, str], ...] = (
        (cfg.enable_basic_mse, "basic_mse"),
        (cfg.enable_basic_l1, "basic_l1"),
        (cfg.enable_basic_focal, "basic_focal"),
        (cfg.enable_basic_hardness, "basic_hardness"),
    )
    for enabled, model_key in basic_variants:
        if not enabled:
            continue
        trainer = AutoencoderTrainer(
            model=BasicAutoencoder(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos[model_key].loss(),
            weight_decay=0.0,
        )
        trainers[model_key] = trainer
    if cfg.enable_basic_flat_mse:
        trainer = AutoencoderTrainer(
            model=BasicFlatAutoencoder(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["basic_flat_mse"].loss(),
            weight_decay=0.0,
        )
        trainers["basic_flat_mse"] = trainer
    if cfg.enable_basic_flat_l1:
        trainer = AutoencoderTrainer(
            model=BasicFlatAutoencoder(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["basic_flat_l1"].loss(),
            weight_decay=0.0,
        )
        trainers["basic_flat_l1"] = trainer
    if cfg.enable_basic_flat_focal:
        trainer = AutoencoderTrainer(
            model=BasicFlatAutoencoder(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["basic_flat_focal"].loss(),
            weight_decay=0.0,
        )
        trainers["basic_flat_focal"] = trainer
    if cfg.enable_basic_flat_focal_spatial:
        trainer = AutoencoderTrainer(
            model=BasicFlatAutoencoder(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["basic_flat_focal_spatial"].loss(),
            weight_decay=0.0,
        )
        trainers["basic_flat_focal_spatial"] = trainer
    if cfg.enable_basic_flat_hardness:
        trainer = AutoencoderTrainer(
            model=BasicFlatAutoencoder(),
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["basic_flat_hardness"].loss(),
            weight_decay=0.0,
        )
        trainers["basic_flat_hardness"] = trainer
    if cfg.enable_best_focal:
        trainer = BestPracticeAutoencoderTrainer(
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["best_focal"].loss(),
        )
        trainers["best_focal"] = trainer
    if cfg.enable_best_flat_focal:
        trainer = BestPracticeVectorAutoencoderTrainer(
            device=device,
            lr=cfg.lr,
            loss_fn=trainer_infos["best_flat_focal"].loss(),
        )
        trainers["best_flat_focal"] = trainer
    if not trainers:
        raise ValueError("No trainers enabled. Enable at least one model to proceed.")
    return trainers


def main() -> None:
    cfg = tyro.cli(Config)
    _print_encoder_table(cfg)
    if cfg.vis_rows <= 0:
        raise ValueError("vis_rows must be positive.")
    if cfg.vis_rows % 2 != 0:
        raise ValueError("vis_rows must be even so each image can include a loss heatmap.")
    if cfg.vis_every <= 0:
        raise ValueError("vis_every must be positive.")
    if cfg.resume_tag not in {"last", "best", "final"}:
        raise ValueError("resume_tag must be one of {'last', 'best', 'final'}.")

    logger = _get_logger()
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    dataset = MarioFrameDataset(Path(cfg.traj_root), max_trajs=cfg.max_trajs)
    vis_image_rows = max(1, cfg.vis_rows // 2)
    spike_window = 50
    spike_min_baseline = 10
    spike_std_multiplier = 3.0
    spike_ratio_threshold = 1.75

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
    spike_samples_dir = samples_root / "spikes"
    fixed_samples_dir.mkdir(parents=True, exist_ok=True)
    rolling_samples_dir.mkdir(parents=True, exist_ok=True)
    spike_samples_dir.mkdir(parents=True, exist_ok=True)

    trainers = build_trainers(cfg, device, TRAINER_INFO_MAP)
    spike_windows: Dict[str, deque[float]] = {
        key: deque(maxlen=spike_window) for key in trainers
    }
    logger.info("Parameter summary:")
    for key, trainer in trainers.items():
        module = getattr(trainer, "model", None)
        if module is None:
            continue
        _summarize_parameters(key, module, logger=logger)
    checkpoint_paths: dict[str, dict[str, Path]] = {}

    for key, trainer in trainers.items():
        trainer_dir = checkpoints_root / key
        trainer_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_paths[key] = {
            "last": trainer_dir / "last.pt",
            "best": trainer_dir / "best.pt",
            "final": trainer_dir / "final.pt",
        }
        if cfg.resume_dir is not None:
            resume_path = checkpoint_paths[key][cfg.resume_tag]
            if not resume_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint for {key!r} not found at {resume_path}"
                )
            state = torch.load(resume_path, map_location=device)
            trainer.load_state_dict(state, lr=cfg.lr)

    if cfg.resume_dir is not None:
        step_set = {trainer.global_step for trainer in trainers.values()}
        if len(step_set) != 1:
            raise RuntimeError("Loaded checkpoints have mismatched global steps.")
        start_step = step_set.pop()
    else:
        start_step = 0

    vis_paths_file = run_dir / "vis_paths.txt"
    expected_vis = min(vis_image_rows, len(dataset))
    if expected_vis <= 0:
        raise RuntimeError("Not enough frames available for visualisation.")

    def _select_vis_paths() -> List[str]:
        indices = random.sample(range(len(dataset)), expected_vis)
        return [str(dataset.paths[idx]) for idx in indices]

    if vis_paths_file.exists():
        vis_paths = [
            line.strip() for line in vis_paths_file.read_text().splitlines() if line.strip()
        ]
    else:
        vis_paths = []
    if len(vis_paths) != expected_vis:
        vis_paths = _select_vis_paths()
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
            batch_cpu = batch
            batch = batch_cpu.to(device, non_blocking=True)
            losses: dict[str, float] = {}
            shared_metrics_step: dict[str, Dict[str, float]] = {}
            timing: dict[str, float] = {}
            step_tag = f"step_{current_step:05d}"
            for key, trainer in trainers.items():
                step_start = time.perf_counter()
                loss, improved, shared = trainer.step(batch)
                timing[key] = time.perf_counter() - step_start
                losses[key] = loss
                shared_metrics_step[key] = shared
                prev_losses = spike_windows[key]
                is_spike = False
                baseline: Optional[float] = None
                if len(prev_losses) >= spike_min_baseline:
                    baseline = sum(prev_losses) / len(prev_losses)
                    variance = (
                        sum((value - baseline) ** 2 for value in prev_losses) / len(prev_losses)
                    )
                    stddev = math.sqrt(variance)
                    ratio_trigger = baseline > 0 and loss >= baseline * spike_ratio_threshold
                    std_trigger = stddev > 0 and loss >= baseline + spike_std_multiplier * stddev
                    is_spike = ratio_trigger or std_trigger
                prev_losses.append(float(loss))
                if is_spike:
                    spike_rows = min(vis_image_rows, batch_cpu.shape[0])
                    if spike_rows > 0:
                        spike_dir = spike_samples_dir / key
                        spike_dir.mkdir(parents=True, exist_ok=True)
                        spike_inputs = batch_cpu[:spike_rows].detach().cpu()
                        with torch.no_grad():
                            spike_recon = trainer.reconstruct(batch)[:spike_rows]
                        spike_recon = spike_recon.detach().cpu()
                        spike_path = spike_dir / f"{step_tag}_loss_{loss:.4f}.png"
                        save_recon_grid(spike_inputs, [(key, spike_recon)], out_path=spike_path)
                        logger.warning(
                            "Loss spike detected for %s (%.4f vs baseline %.4f)",
                            _display_name(key),
                            loss,
                            baseline if baseline is not None else float("nan"),
                        )
                trainer.save_checkpoint(checkpoint_paths[key]["last"])
                if improved:
                    trainer.save_checkpoint(checkpoint_paths[key]["best"])
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
                fixed_recons = [
                    (key, trainer.reconstruct(vis_batch_device)) for key, trainer in trainers.items()
                ]
                save_recon_grid(
                    vis_batch,
                    fixed_recons,
                    out_path=fixed_samples_dir / f"{step_tag}.png",
                )
                rolling_batch = sample_random_batch(dataset, vis_image_rows)
                rolling_batch_device = rolling_batch.to(device)
                rolling_recons = [
                    (key, trainer.reconstruct(rolling_batch_device))
                    for key, trainer in trainers.items()
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
            (key, trainer.reconstruct(vis_batch_device)) for key, trainer in trainers.items()
        ]
        save_recon_grid(
            vis_batch,
            fixed_recons,
            out_path=fixed_samples_dir / f"{step_tag}.png",
        )
        rolling_batch = sample_random_batch(dataset, vis_image_rows)
        rolling_batch_device = rolling_batch.to(device)
        rolling_recons = [
            (key, trainer.reconstruct(rolling_batch_device))
            for key, trainer in trainers.items()
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
    for key, trainer in trainers.items():
        paths = checkpoint_paths[key]
        trainer.save_checkpoint(paths["last"])
        if trainer.best_loss is not None and not paths["best"].exists():
            trainer.save_checkpoint(paths["best"])
        trainer.save_checkpoint(paths["final"])


if __name__ == "__main__":
    main()
