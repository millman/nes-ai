#!/usr/bin/env python3
"""Training loop for the JEPA world model with local/global specialization and SIGReg."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import csv
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from datetime import datetime

from recon.data import list_trajectories, load_frame_as_tensor
from utils.device_utils import pick_device


# ------------------------------------------------------------
# Model components
# ------------------------------------------------------------


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


class Encoder(nn.Module):
    def __init__(self, in_channels: int, schedule: Tuple[int, ...]):
        super().__init__()
        blocks: List[nn.Module] = []
        ch_prev = in_channels
        for ch in schedule:
            blocks.append(DownBlock(ch_prev, ch))
            ch_prev = ch
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.blocks(x)
        return self.pool(feats).flatten(1)


class PredictorNetwork(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VisualizationAdapter(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


class HardnessWeightedL1Loss(nn.Module):
    """Standalone copy of the hardness-weighted L1 used by reconstruction scripts."""

    def __init__(self, beta: float = 2.0, max_weight: float = 100.0, eps: float = 1e-6) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive.")
        if max_weight <= 0:
            raise ValueError("max_weight must be positive.")
        self.beta = beta
        self.max_weight = max_weight
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = torch.abs(input - target)
        dims = tuple(range(1, l1.dim()))
        norm = l1.detach().mean(dim=dims, keepdim=True).clamp_min(self.eps)
        rel_error = l1.detach() / norm
        weight = rel_error.pow(self.beta).clamp(max=self.max_weight)
        return (weight * l1).mean()


class VisualizationDecoder(nn.Module):
    def __init__(self, latent_dim: int, image_channels: int, image_size: int) -> None:
        super().__init__()
        out_dim = image_channels * image_size * image_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.SiLU(inplace=True),
            nn.Linear(512, 512),
            nn.SiLU(inplace=True),
            nn.Linear(512, out_dim),
        )
        self.image_channels = image_channels
        self.image_size = image_size

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        original_shape = latent.shape[:-1]
        latent = latent.reshape(-1, latent.shape[-1])
        decoded = torch.sigmoid(self.net(latent))
        decoded = decoded.view(*original_shape, self.image_channels, self.image_size, self.image_size)
        return decoded


RECON_LOSS = HardnessWeightedL1Loss()


# ------------------------------------------------------------
# Configs and containers
# ------------------------------------------------------------


@dataclass
class ModelConfig:
    in_channels: int = 3
    image_size: int = 64
    channel_schedule: Tuple[int, ...] = (32, 64, 128, 256)
    latent_dim: int = 128
    hidden_dim: int = 256
    h_local_dim: int = 128
    h_global_dim: int = 128
    embedding_dim: int = 192
    action_dim: int = 4


@dataclass
class LossWeights:
    jepa: float = 1.0
    local: float = 1.0
    global_pred: float = 0.5
    global_slow: float = 0.1
    sigreg: float = 0.5
    latent_match: float = 1.0


@dataclass
class TrainConfig:
    seq_len: int = 8
    batch_size: int = 4
    lr: float = 1e-4
    decoder_lr: float = 1e-4
    grad_clip: float = 1.0
    sigreg_projections: int = 32
    recon_weight: float = 1.0
    device: Optional[str] = "mps"
    total_steps: int = 10_000
    log_every_steps: int = 10
    vis_every_steps: int = 50
    vis_rows: int = 4
    vis_rollout: int = 4
    output_dir: Path = Path("out.jepa_world_model_trainer")
    data_root: Path = Path("data.gridworldkey")
    max_trajectories: Optional[int] = None
    dummy_dataset_size: int = 256


class JEPAWorldModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.h_local_dim + cfg.h_global_dim != cfg.hidden_dim:
            raise ValueError("h_local_dim + h_global_dim must equal hidden_dim")
        self.cfg = cfg
        self.encoder = Encoder(cfg.in_channels, cfg.channel_schedule)
        enc_dim = cfg.channel_schedule[-1]
        self.obs_proj = nn.Sequential(
            nn.Linear(enc_dim, cfg.latent_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
        self.prior_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
        self.gru = nn.GRUCell(cfg.latent_dim + cfg.action_dim, cfg.hidden_dim)
        emb_in = cfg.latent_dim + cfg.hidden_dim
        self.proj_e = nn.Sequential(
            nn.Linear(emb_in, cfg.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.embedding_dim),
        )
        self.target_proj = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.embedding_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cfg.embedding_dim, cfg.embedding_dim),
        )
        pred_in = cfg.latent_dim + cfg.hidden_dim + cfg.action_dim
        self.predictor = PredictorNetwork(pred_in, cfg.hidden_dim, cfg.embedding_dim)
        loc_in = cfg.latent_dim + cfg.h_local_dim + cfg.action_dim
        glob_in = cfg.latent_dim + cfg.h_global_dim + cfg.action_dim
        self.local_predictor = PredictorNetwork(loc_in, cfg.hidden_dim, cfg.h_local_dim)
        self.global_predictor = PredictorNetwork(glob_in, cfg.hidden_dim, cfg.h_global_dim)

    @property
    def hidden_dim(self) -> int:
        return self.cfg.hidden_dim

    def split_hidden(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_local = h[..., : self.cfg.h_local_dim]
        h_global = h[..., self.cfg.h_local_dim :]
        return h_local, h_global

    def transition(self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([z, action], dim=-1)
        return self.gru(inp, h)

    def build_embedding(self, z: torch.Tensor, h_local: torch.Tensor, h_global: torch.Tensor) -> torch.Tensor:
        return self.proj_e(torch.cat([z, h_local, h_global], dim=-1))

    def rollout(self, images: torch.Tensor, actions: torch.Tensor, h0: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        b, t, _, _, _ = images.shape
        device = images.device
        if h0 is None:
            h = torch.zeros(b, self.hidden_dim, device=device)
        else:
            h = h0
        h_history: List[torch.Tensor] = []
        z_obs_list: List[torch.Tensor] = []
        z_prior_list: List[torch.Tensor] = []
        embeddings: List[torch.Tensor] = []
        for step in range(t):
            h_history.append(h)
            h_local, h_global = self.split_hidden(h)
            feats = self.encoder(images[:, step])
            z_prior = self.prior_proj(h)
            z_obs = self.obs_proj(feats)
            embedding = self.build_embedding(z_obs, h_local, h_global)
            z_obs_list.append(z_obs)
            z_prior_list.append(z_prior)
            embeddings.append(embedding)
            h = self.transition(h, z_obs, actions[:, step])
        h_history.append(h)
        h_hist = torch.stack(h_history, dim=1)  # [B, T+1, hidden]
        return {
            "z_obs": torch.stack(z_obs_list, dim=1),
            "z_prior": torch.stack(z_prior_list, dim=1),
            "embeddings": torch.stack(embeddings, dim=1),
            "h_history": h_hist,
        }


# ------------------------------------------------------------
# Loss utilities
# ------------------------------------------------------------


def jepa_loss(model: JEPAWorldModel, outputs: Dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
    z = outputs["z_obs"]
    embeddings = outputs["embeddings"]
    h_t = outputs["h_history"][:, :-1]
    preds = model.predictor(torch.cat([z[:, :-1], h_t[:, :-1], actions[:, :-1]], dim=-1))
    target = model.target_proj(z[:, 1:]).detach()
    return F.mse_loss(preds, target)


def specialization_losses(model: JEPAWorldModel, outputs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h_t = outputs["h_history"][:, :-1]
    h_t1 = outputs["h_history"][:, 1:]
    z = outputs["z_obs"]
    local_now = h_t[:, :-1, : model.cfg.h_local_dim]
    local_target = h_t1[:, :-1, : model.cfg.h_local_dim].detach()
    global_now = h_t[:, :-1, model.cfg.h_local_dim :]
    global_target = h_t1[:, :-1, model.cfg.h_local_dim :].detach()
    local_in = torch.cat([z[:, :-1], local_now, actions[:, :-1]], dim=-1)
    global_in = torch.cat([z[:, :-1], global_now, actions[:, :-1]], dim=-1)
    local_pred = model.local_predictor(local_in)
    global_pred = model.global_predictor(global_in)
    local_loss = F.mse_loss(local_pred, local_target)
    global_pred_loss = F.mse_loss(global_pred, global_target)
    global_slow_loss = F.mse_loss(global_target, global_now)
    return local_loss, global_pred_loss, global_slow_loss


def sigreg_loss(embeddings: torch.Tensor, num_projections: int) -> torch.Tensor:
    b, t, d = embeddings.shape
    flat = embeddings.reshape(b * t, d)
    device = embeddings.device
    directions = torch.randn(num_projections, d, device=device)
    directions = F.normalize(directions, dim=-1)
    projected = flat @ directions.t()
    projected = projected.t()
    sorted_proj, _ = torch.sort(projected, dim=1)
    normal_samples = torch.randn_like(projected)
    sorted_normal, _ = torch.sort(normal_samples, dim=1)
    return F.mse_loss(sorted_proj, sorted_normal)


def reconstruction_loss(vis_adapter: VisualizationAdapter, decoder: VisualizationDecoder, embeddings: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    vis_latent = vis_adapter(embeddings)
    recons = decoder(vis_latent)
    return RECON_LOSS(recons, images)


# ------------------------------------------------------------
# Training loop utilities
# ------------------------------------------------------------


def training_step(
    model: JEPAWorldModel,
    vis_adapter: VisualizationAdapter,
    decoder: VisualizationDecoder,
    optim_world: torch.optim.Optimizer,
    optim_decoder: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    cfg: TrainConfig,
    weights: LossWeights,
) -> Dict[str, float]:
    images, actions = batch
    world_params = [p for p in model.parameters()]
    device = next(model.parameters()).device
    images = images.to(device)
    actions = actions.to(device)
    outputs = model.rollout(images, actions)
    loss_jepa = jepa_loss(model, outputs, actions)
    loss_local, loss_global_pred, loss_global_slow = specialization_losses(model, outputs, actions)
    loss_sigreg = sigreg_loss(outputs["embeddings"], cfg.sigreg_projections)
    loss_recon = reconstruction_loss(vis_adapter, decoder, outputs["embeddings"], images)
    loss_prior = F.mse_loss(outputs["z_prior"], outputs["z_obs"].detach())

    world_loss = (
        weights.jepa * loss_jepa
        + weights.local * loss_local
        + weights.global_pred * loss_global_pred
        + weights.global_slow * loss_global_slow
        + weights.sigreg * loss_sigreg
        + weights.latent_match * loss_prior
        + cfg.recon_weight * loss_recon
    )

    optim_world.zero_grad()
    optim_decoder.zero_grad()
    world_loss.backward()
    if cfg.grad_clip > 0:
        world_grad_norm = float(nn.utils.clip_grad_norm_(world_params, cfg.grad_clip).item())
    else:
        world_grad_norm = grad_norm(world_params)
    optim_world.step()

    decoder_params = list(vis_adapter.parameters()) + list(decoder.parameters())
    decoder_grad_norm = grad_norm(decoder_params)
    optim_decoder.step()

    z_obs = outputs["z_obs"]
    z_mean = float(z_obs.mean().item())
    z_std = float(z_obs.std().item())
    h_hist = outputs["h_history"]
    h_local = h_hist[..., : model.cfg.h_local_dim]
    h_global = h_hist[..., model.cfg.h_local_dim :]
    h_local_std = float(h_local.std().item())
    h_global_std = float(h_global.std().item())

    return {
        "loss_jepa": loss_jepa.item(),
        "loss_local": loss_local.item(),
        "loss_global_pred": loss_global_pred.item(),
        "loss_global_slow": loss_global_slow.item(),
        "loss_sigreg": loss_sigreg.item(),
        "loss_recon": loss_recon.item(),
        "loss_prior": loss_prior.item(),
        "loss_world": world_loss.item(),
        "grad_world": world_grad_norm,
        "grad_decoder": decoder_grad_norm,
        "z_mean": z_mean,
        "z_std": z_std,
        "h_local_std": h_local_std,
        "h_global_std": h_global_std,
    }


# ------------------------------------------------------------
# Example dataset + dataloader
# ------------------------------------------------------------


class DummyTrajectoryDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Synthetic dataset to exercise the training loop."""

    def __init__(self, length: int, seq_len: int, image_shape: Tuple[int, int, int], action_dim: int) -> None:
        self.length = length
        self.seq_len = seq_len
        self.image_shape = image_shape
        self.action_dim = action_dim

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        c, h, w = self.image_shape
        obs = torch.rand(self.seq_len, c, h, w)
        actions = torch.randn(self.seq_len, self.action_dim)
        return obs, actions


def collate_batch(batch: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    obs, actions = zip(*batch)
    obs_tensor = torch.stack(obs, dim=0)
    act_tensor = torch.stack(actions, dim=0)
    return obs_tensor, act_tensor


class TrajectorySequenceDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Load contiguous frame/action sequences from recorded trajectories."""

    def __init__(
        self,
        root: Path,
        seq_len: int,
        image_hw: Tuple[int, int],
        max_traj: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.seq_len = seq_len
        self.image_hw = image_hw
        if self.seq_len < 1:
            raise ValueError("seq_len must be positive.")
        trajectories = list_trajectories(self.root)
        items = list(trajectories.items())
        if max_traj is not None:
            items = items[:max_traj]
        self.samples: List[Tuple[List[Path], np.ndarray, int]] = []
        self.action_dim: Optional[int] = None
        for traj_name, frame_paths in items:
            actions_path = (self.root / traj_name) / "actions.npz"
            if not actions_path.is_file():
                continue
            try:
                with np.load(actions_path) as data:
                    action_arr = data["actions"] if "actions" in data else data[list(data.files)[0]]
            except Exception:
                continue
            if action_arr.ndim == 1:
                action_arr = action_arr[:, None]
            if action_arr.shape[0] == len(frame_paths) + 1:
                action_arr = action_arr[1:]
            if action_arr.shape[0] < len(frame_paths):
                continue
            action_arr = action_arr.astype(np.float32, copy=False)
            if self.action_dim is None:
                self.action_dim = action_arr.shape[1]
            elif self.action_dim != action_arr.shape[1]:
                continue
            if len(frame_paths) < self.seq_len:
                continue
            max_start = len(frame_paths) - self.seq_len
            for start in range(max_start + 1):
                self.samples.append((frame_paths, action_arr, start))
        if not self.samples:
            raise RuntimeError(f"No usable sequences found under {self.root}")
        if self.action_dim is None:
            raise RuntimeError("Failed to infer action dimensionality.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_paths, actions, start = self.samples[index]
        frames: List[torch.Tensor] = []
        for offset in range(self.seq_len):
            path = frame_paths[start + offset]
            frame = load_frame_as_tensor(path, size=self.image_hw)
            frames.append(frame)
        action_slice = actions[start : start + self.seq_len]
        return torch.stack(frames, dim=0), torch.from_numpy(action_slice)


# ------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------


def log_metrics(step: int, metrics: Dict[str, float]) -> None:
    pretty = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    print(f"[step {step}] {pretty}")


LOSS_COLUMNS = [
    "step",
    "loss_world",
    "loss_jepa",
    "loss_local",
    "loss_global_pred",
    "loss_global_slow",
    "loss_sigreg",
    "loss_recon",
    "loss_prior_match",
    "grad_world",
    "grad_decoder",
    "z_mean",
    "z_std",
    "h_local_std",
    "h_global_std",
]


@dataclass
class LossHistory:
    steps: List[float] = field(default_factory=list)
    world: List[float] = field(default_factory=list)
    jepa: List[float] = field(default_factory=list)
    local: List[float] = field(default_factory=list)
    global_pred: List[float] = field(default_factory=list)
    global_slow: List[float] = field(default_factory=list)
    sigreg: List[float] = field(default_factory=list)
    recon: List[float] = field(default_factory=list)
    prior_match: List[float] = field(default_factory=list)
    grad_world: List[float] = field(default_factory=list)
    grad_decoder: List[float] = field(default_factory=list)
    z_mean_vals: List[float] = field(default_factory=list)
    z_std_vals: List[float] = field(default_factory=list)
    h_local_std_vals: List[float] = field(default_factory=list)
    h_global_std_vals: List[float] = field(default_factory=list)

    def append(self, step: float, metrics: Dict[str, float]) -> None:
        self.steps.append(step)
        self.world.append(metrics["loss_world"])
        self.jepa.append(metrics["loss_jepa"])
        self.local.append(metrics["loss_local"])
        self.global_pred.append(metrics["loss_global_pred"])
        self.global_slow.append(metrics["loss_global_slow"])
        self.sigreg.append(metrics["loss_sigreg"])
        self.recon.append(metrics["loss_recon"])
        self.prior_match.append(metrics["loss_prior"])
        self.grad_world.append(metrics["grad_world"])
        self.grad_decoder.append(metrics["grad_decoder"])
        self.z_mean_vals.append(metrics["z_mean"])
        self.z_std_vals.append(metrics["z_std"])
        self.h_local_std_vals.append(metrics["h_local_std"])
        self.h_global_std_vals.append(metrics["h_global_std"])

    def __len__(self) -> int:
        return len(self.steps)


def write_loss_csv(history: LossHistory, path: Path) -> None:
    if len(history) == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LOSS_COLUMNS)
        for row in zip(
            history.steps,
            history.world,
            history.jepa,
            history.local,
            history.global_pred,
            history.global_slow,
            history.sigreg,
            history.recon,
            history.prior_match,
            history.grad_world,
            history.grad_decoder,
            history.z_mean_vals,
            history.z_std_vals,
            history.h_local_std_vals,
            history.h_global_std_vals,
        ):
            writer.writerow(row)


def plot_loss_curves(history: LossHistory, out_dir: Path) -> None:
    if len(history) == 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history.steps, history.world, label="world")
    plt.plot(history.steps, history.jepa, label="jepa")
    plt.plot(history.steps, history.local, label="local")
    plt.plot(history.steps, history.global_pred, label="global_pred")
    plt.plot(history.steps, history.global_slow, label="global_slow")
    plt.plot(history.steps, history.sigreg, label="sigreg")
    plt.plot(history.steps, history.recon, label="recon")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training Losses")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=200)
    plt.close()


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2).item()
        total += param_norm * param_norm
    return float(total**0.5)


def save_embedding_projection(embeddings: torch.Tensor, path: Path) -> None:
    flat = embeddings.detach().cpu().numpy()
    b, t, d = flat.shape
    flat = flat.reshape(b * t, d)
    flat = flat - flat.mean(axis=0, keepdims=True)
    if flat.shape[0] < 2:
        return
    u, s, vt = np.linalg.svd(flat, full_matrices=False)
    coords = flat @ vt[:2].T
    colors = np.tile(np.arange(t), b)
    colors = colors / max(1, t - 1)
    plt.figure(figsize=(6, 5))
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, cmap="viridis", s=10, alpha=0.7)
    plt.title("Embedding Projection (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="Time step")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def tensor_to_uint8_image(frame: torch.Tensor) -> np.ndarray:
    array = frame.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (array * 255.0).round().astype(np.uint8)


def save_rollout_visualization(
    out_path: Path,
    frames: torch.Tensor,
    recon_frames: torch.Tensor,
    pred_frames: torch.Tensor,
    rows: int,
    rollout_steps: int,
) -> None:
    frames = frames.detach().cpu()
    recon_frames = recon_frames.detach().cpu()
    pred_frames = pred_frames.detach().cpu()
    total_frames = frames.shape[0]
    num_rows = min(rows, total_frames)
    pred_len = pred_frames.shape[0]
    grid_rows: List[np.ndarray] = []
    for row_idx in range(num_rows):
        columns = [
            tensor_to_uint8_image(frames[row_idx]),
            tensor_to_uint8_image(recon_frames[row_idx]),
        ]
        for horizon in range(rollout_steps):
            if pred_len == 0:
                columns.append(columns[-1])
                continue
            pred_idx = min(pred_len - 1, row_idx + horizon)
            columns.append(tensor_to_uint8_image(pred_frames[pred_idx]))
        row_image = np.concatenate(columns, axis=1)
        grid_rows.append(row_image)
    grid = np.concatenate(grid_rows, axis=0) if grid_rows else np.zeros((1, 1, 3), dtype=np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path)


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------


def run_training(cfg: TrainConfig, model_cfg: ModelConfig, weights: LossWeights, demo: bool = True) -> None:
    device = pick_device(cfg.device)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = cfg.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = run_dir / "metrics"
    vis_dir = run_dir / "visualizations"
    (metrics_dir).mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    loss_history = LossHistory()
    if demo:
        dataset = DummyTrajectoryDataset(
            length=cfg.dummy_dataset_size,
            seq_len=cfg.seq_len,
            image_shape=(model_cfg.in_channels, model_cfg.image_size, model_cfg.image_size),
            action_dim=model_cfg.action_dim,
        )
    else:
        dataset = TrajectorySequenceDataset(
            root=cfg.data_root,
            seq_len=cfg.seq_len,
            image_hw=(model_cfg.image_size, model_cfg.image_size),
            max_traj=cfg.max_trajectories,
        )
    dataset_action_dim = getattr(dataset, "action_dim", model_cfg.action_dim)
    if model_cfg.action_dim != dataset_action_dim:
        model_cfg = replace(model_cfg, action_dim=dataset_action_dim)
    model = JEPAWorldModel(model_cfg).to(device)
    vis_adapter = VisualizationAdapter(model_cfg.embedding_dim, model_cfg.latent_dim).to(device)
    decoder = VisualizationDecoder(model_cfg.latent_dim, model_cfg.in_channels, model_cfg.image_size).to(device)
    optim_world = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    optim_decoder = torch.optim.Adam(list(vis_adapter.parameters()) + list(decoder.parameters()), lr=cfg.decoder_lr)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_batch)

    vis_batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    try:
        sample_batch = next(iter(dataloader))
        vis_batch_cpu = (sample_batch[0].clone(), sample_batch[1].clone())
    except StopIteration:
        vis_batch_cpu = None

    data_iter = iter(dataloader)
    for global_step in range(cfg.total_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        metrics = training_step(
            model, vis_adapter, decoder, optim_world, optim_decoder, batch, cfg, weights
        )
        if cfg.log_every_steps > 0 and global_step % cfg.log_every_steps == 0:
            log_metrics(global_step, metrics)
            loss_history.append(global_step, metrics)
            write_loss_csv(loss_history, metrics_dir / "loss.csv")
            plot_loss_curves(loss_history, metrics_dir)
        if (
            cfg.vis_every_steps > 0
            and global_step % cfg.vis_every_steps == 0
            and vis_batch_cpu is not None
        ):
            model.eval()
            with torch.no_grad():
                vis_frames = vis_batch_cpu[0].to(device)
                vis_actions = vis_batch_cpu[1].to(device)
                vis_outputs = model.rollout(vis_frames, vis_actions)
                vis_embeddings = vis_outputs["embeddings"]
                recon_frames = decoder(vis_adapter(vis_embeddings)).clamp(0, 1)
                h_t = vis_outputs["h_history"][:, :-1]
                predictor_in = torch.cat(
                    [
                        vis_outputs["z_obs"][:, :-1],
                        h_t[:, :-1],
                        vis_actions[:, :-1],
                    ],
                    dim=-1,
                )
                pred_embeddings = model.predictor(predictor_in)
                pred_frames = decoder(vis_adapter(pred_embeddings)).clamp(0, 1)
                seq_index = 0
                save_rollout_visualization(
                    vis_dir / f"rollout_{global_step:07d}.png",
                    vis_frames[seq_index],
                    recon_frames[seq_index],
                    pred_frames[seq_index],
                    cfg.vis_rows,
                    cfg.vis_rollout,
                )
                save_embedding_projection(
                    vis_outputs["embeddings"],
                    vis_dir / f"embeddings_{global_step:07d}.png",
                )
            model.train()
    if len(loss_history):
        write_loss_csv(loss_history, metrics_dir / "loss.csv")
        plot_loss_curves(loss_history, metrics_dir)


def main() -> None:
    cfg = tyro.cli(TrainConfig)
    model_cfg = ModelConfig()
    weights = LossWeights()
    run_training(cfg, model_cfg, weights, demo=False)


if __name__ == "__main__":
    main()
