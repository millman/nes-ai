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
from typing import Dict, List, Optional, Tuple

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


class MarioSequenceDataset(Dataset):
    """Return sliding windows of four frames + actions and four future frames."""

    def __init__(
        self,
        root_dir: Path,
        *,
        image_hw: Tuple[int, int],
        normalize: Optional[Normalize] = None,
        max_trajs: Optional[int] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_hw = image_hw
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

            if len(frames) < 8:
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
            for start in range(len(frames) - 7):
                self.entries.append((frames, actions, start))
        if not self.entries:
            raise RuntimeError(f"No valid trajectories under {self.root_dir}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        frame_paths, actions, start = self.entries[idx]
        inputs = []
        futures = []
        for i in range(4):
            frame = load_frame_as_tensor(
                frame_paths[start + i],
                size=self.image_hw,
                normalize=None,
            )
            if self.normalize is not None:
                frame = self.normalize(frame)
            inputs.append(frame)
        for i in range(4):
            frame = load_frame_as_tensor(
                frame_paths[start + 4 + i],
                size=self.image_hw,
                normalize=None,
            )
            if self.normalize is not None:
                frame = self.normalize(frame)
            futures.append(frame)
        input_actions = actions[start : start + 4]
        future_actions = actions[start + 4 : start + 8]
        input_labels = [
            short_traj_state_label(str(frame_paths[start + i])) for i in range(4)
        ]
        future_labels = [
            short_traj_state_label(str(frame_paths[start + 4 + i])) for i in range(4)
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


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        groups = 8 if out_ch % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrameEncoder(nn.Module):
    """Shared encoder inspired by LightweightAutoencoder."""

    def __init__(self, base_channels: int = 48, latent_channels: int = 128) -> None:
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

    def forward(self, frame: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = self.stem(frame)
        h1 = self.down1(h0)
        h2 = self.down2(h1)
        h3 = self.down3(h2)
        latent = self.bottleneck(h3)
        pooled = self.pool(latent).flatten(1)
        return latent, pooled


class LatentDecoder(nn.Module):
    def __init__(self, latent_channels: int = 128, base_channels: int = 48) -> None:
        super().__init__()
        self.up1 = UpBlock(latent_channels, base_channels * 3)
        self.up2 = UpBlock(base_channels * 3, base_channels * 2)
        self.up3 = UpBlock(base_channels * 2, base_channels)
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels // 2, 3, kernel_size=1),
        )

    def forward(self, latent: torch.Tensor, *, target_hw: Tuple[int, int]) -> torch.Tensor:
        h = self.up1(latent)
        h = self.up2(h)
        h = self.up3(h)
        frame = self.head(h)
        if frame.shape[-2:] != target_hw:
            frame = F.interpolate(frame, size=target_hw, mode="bilinear", align_corners=False)
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


class TemporalAttentionAggregator(nn.Module):
    def __init__(self, latent_dim: int, action_embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.token_proj = nn.Linear(latent_dim + action_embed_dim, latent_dim)
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.key_proj = nn.Linear(latent_dim, latent_dim)
        self.value_proj = nn.Linear(latent_dim, latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, num_heads=num_heads, batch_first=True)
        self.context_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.post = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(
        self,
        latent_vectors: torch.Tensor,
        action_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.cat([latent_vectors, action_embeddings], dim=-1)
        tokens = self.token_proj(tokens)
        query = self.context_token.expand(tokens.shape[0], -1, -1)
        q = self.query_proj(query)
        k = self.key_proj(tokens)
        v = self.value_proj(tokens)
        context, weights = self.attn(q, k, v)
        fused = self.post(context.squeeze(1))
        return fused, weights


class NextLatentPredictor(nn.Module):
    def __init__(self, latent_dim: int, latent_hw: Tuple[int, int]) -> None:
        super().__init__()
        h, w = latent_hw
        hidden = latent_dim * 2
        self.latent_dim = latent_dim
        self.latent_hw = latent_hw
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, latent_dim * h * w),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        flat = self.net(context)
        return flat.view(context.shape[0], self.latent_dim, *self.latent_hw)


@dataclass
class ModelOutputs:
    predicted_frame: torch.Tensor
    predicted_latent: torch.Tensor
    encoded_next_latent: Optional[torch.Tensor]
    attention_weights: Optional[torch.Tensor]
    input_reconstructions: Optional[torch.Tensor]


class MarioSequencePredictor(nn.Module):
    def __init__(
        self,
        action_dim: int,
        *,
        base_channels: int = 48,
        latent_channels: int = 128,
        image_hw: Tuple[int, int] = (224, 224),
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.image_hw = image_hw
        self.encoder = FrameEncoder(base_channels=base_channels, latent_channels=latent_channels)
        self.decoder = LatentDecoder(latent_channels=latent_channels, base_channels=base_channels)
        self.action_encoder = ActionEncoder(action_dim, embed_dim=latent_channels)
        latent_hw = (image_hw[0] // 8, image_hw[1] // 8)
        self.aggregator = TemporalAttentionAggregator(
            latent_dim=latent_channels,
            action_embed_dim=latent_channels,
            num_heads=num_heads,
        )
        self.predictor = NextLatentPredictor(latent_dim=latent_channels, latent_hw=latent_hw)
        self.recon_loss_fn = nn.SmoothL1Loss()
        self.latent_loss_fn = nn.SmoothL1Loss()

    def encode_frames(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _, _, _ = frames.shape
        spatial, pooled = [], []
        for t in range(steps):
            latent, vec = self.encoder(frames[:, t])
            spatial.append(latent)
            pooled.append(vec)
        spatial_tensor = torch.stack(spatial, dim=1)
        pooled_tensor = torch.stack(pooled, dim=1)
        return spatial_tensor, pooled_tensor

    def forward(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        *,
        next_frame: Optional[torch.Tensor] = None,
        decode_recon: bool = True,
    ) -> ModelOutputs:
        latent_maps, latent_vecs = self.encode_frames(frames)
        action_embeddings = self.action_encoder(actions)
        context_vec, attn_weights = self.aggregator(latent_vecs, action_embeddings)
        predicted_latent = self.predictor(context_vec)
        predicted_frame = self.decoder(predicted_latent, target_hw=self.image_hw)
        recon = None
        if decode_recon:
            b, steps, c, h, w = latent_maps.shape
            flat = latent_maps.view(b * steps, c, h, w)
            decoded = self.decoder(flat, target_hw=self.image_hw)
            recon = decoded.view(b, steps, 3, *self.image_hw)
        encoded_next_latent = None
        if next_frame is not None:
            encoded_next_latent, _ = self.encoder(next_frame)
        return ModelOutputs(
            predicted_frame=predicted_frame,
            predicted_latent=predicted_latent,
            encoded_next_latent=encoded_next_latent,
            attention_weights=attn_weights,
            input_reconstructions=recon,
        )

    def compute_losses(
        self,
        outputs: ModelOutputs,
        *,
        input_frames: torch.Tensor,
        target_frame: torch.Tensor,
        lambda_recon: float = 1.0,
        lambda_latent: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if outputs.input_reconstructions is None:
            raise ValueError("Reconstruction logits not computed.")
        recon_loss = self.recon_loss_fn(outputs.input_reconstructions, input_frames)
        if outputs.encoded_next_latent is None:
            raise ValueError("Encoded next-frame latent required for latent loss.")
        latent_loss = self.latent_loss_fn(
            outputs.predicted_latent, outputs.encoded_next_latent.detach()
        )
        total = lambda_recon * recon_loss + lambda_latent * latent_loss
        return {
            "total": total,
            "recon": recon_loss,
            "latent": latent_loss,
        }


LOSS_COLUMNS = ["step", "total", "recon", "latent"]


def write_loss_csv(history: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LOSS_COLUMNS)
        for entry in history:
            writer.writerow([entry[col] for col in LOSS_COLUMNS])


def plot_loss_curves(history: List[Dict[str, float]], out_dir: Path) -> None:
    if not history:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = [entry["step"] for entry in history]

    plt.figure(figsize=(8, 5))
    for label in ("total", "recon", "latent"):
        plt.plot(steps, [entry[label] for entry in history], label=label)
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
    raw_future_actions_tensor = batch.get("future_actions")
    raw_future_actions = (
        raw_future_actions_tensor[0].cpu().numpy()
        if raw_future_actions_tensor is not None
        else None
    )
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
    action_template = torch.zeros_like(actions[:, :1])
    with torch.no_grad():
        outputs = model(inputs, actions, next_frame=future_frames[:, 0])
        predicted_frame = outputs.predicted_frame
        recon = outputs.input_reconstructions

    hw = model.image_hw
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    # Row 1: ground-truth inputs and targets
    def _action_desc(action_vec: np.ndarray) -> str:
        binarized = (action_vec > 0.5).astype(np.uint8)
        return _describe_controller_vector(binarized)

    for i in range(4):
        axes[0, i].imshow(_tensor_to_image(inputs[0, i]))
        labels = []
        if input_labels is not None and i < len(input_labels):
            labels.append(str(input_labels[i]))
        if i < raw_input_actions.shape[0]:
            labels.append(_action_desc(raw_input_actions[i]))
        if labels:
            axes[0, i].set_title("\n".join(labels), fontsize=8)
    for i in range(4):
        axes[0, 4 + i].imshow(_tensor_to_image(future_frames[0, i]))
        labels = []
        if future_labels is not None and i < len(future_labels):
            labels.append(str(future_labels[i]))
        if raw_future_actions is not None and i < raw_future_actions.shape[0]:
            labels.append(_action_desc(raw_future_actions[i]))
        if labels:
            axes[0, 4 + i].set_title("\n".join(labels), fontsize=8)

    # Row 2: reconstructed inputs
    if recon is not None:
        for i in range(4):
            axes[1, i].imshow(_tensor_to_image(recon[0, i]))
    for i in range(4, 8):
        axes[1, i].imshow(_blank_image(hw))

    # Row 3: autoregressive predictions
    window_frames = inputs.clone()
    window_actions = actions.clone()
    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for step in range(4):
            roll = model(window_frames, window_actions, decode_recon=False)
            preds.append(roll.predicted_frame[0])
            window_frames = torch.cat([window_frames[:, 1:], roll.predicted_frame.unsqueeze(1)], dim=1)
            zero_action = torch.zeros_like(action_template)
            window_actions = torch.cat([window_actions[:, 1:], zero_action], dim=1)
    for i in range(4):
        axes[2, i].imshow(_blank_image(hw))
    for i, pred in enumerate(preds):
        axes[2, 4 + i].imshow(_tensor_to_image(pred))

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
        base_channels=48,
        latent_channels=128,
        image_hw=image_hw,
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
        running = {"total": 0.0, "recon": 0.0, "latent": 0.0}
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["inputs"].to(device)
            actions = batch["actions"].to(device)
            future = batch["future_frames"].to(device)
            target_frame = future[:, 0]
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=config.mixed_precision):
                outputs = model(inputs, actions, next_frame=target_frame)
                losses = model.compute_losses(
                    outputs,
                    input_frames=inputs,
                    target_frame=target_frame,
                    lambda_recon=config.lambda_recon,
                    lambda_latent=config.lambda_latent,
                )
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()

            running["total"] += losses["total"].item()
            running["recon"] += losses["recon"].item()
            running["latent"] += losses["latent"].item()

            current_step = global_step + 1
            if config.log_every > 0 and current_step % config.log_every == 0:
                logger.info(
                    "Epoch %d Step %d | total %.4f recon %.4f latent %.4f",
                    epoch,
                    current_step,
                    losses["total"].item(),
                    losses["recon"].item(),
                    losses["latent"].item(),
                )
            loss_history.append(
                {
                    "step": current_step,
                    "total": float(losses["total"].item()),
                    "recon": float(losses["recon"].item()),
                    "latent": float(losses["latent"].item()),
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
            "Epoch %d complete | avg_total %.4f avg_recon %.4f avg_latent %.4f",
            epoch,
            avg_total,
            running["recon"] / steps_in_epoch,
            running["latent"] / steps_in_epoch,
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
