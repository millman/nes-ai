#!/usr/bin/env python3
"""Training loop for the JEPA world model with local/global specialization and SIGReg."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import csv
import json
import numpy as np
from PIL import Image
import subprocess
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


class FocalL1Loss(nn.Module):
    """Focal-style L1 reconstruction loss that upweights harder pixels."""

    def __init__(self, gamma: float = 2.0, max_weight: float = 100.0, eps: float = 1e-6) -> None:
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
        dims = tuple(range(1, l1.dim()))
        norm = l1.detach().mean(dim=dims, keepdim=True).clamp_min(self.eps)
        rel_error = l1 / norm
        weight = rel_error.pow(self.gamma).clamp(max=self.max_weight)
        return (weight * l1).mean()


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
JEPA_LOSS = HardnessWeightedL1Loss()
SIGREG_LOSS = nn.MSELoss()


# ------------------------------------------------------------
# Configs and containers
# ------------------------------------------------------------


@dataclass
class ModelConfig:
    """Dimensional flow (bottlenecks marked with *):

        image (image_size^2·in_channels)
            │  Encoder(channel_schedule → embedding_dim)
            ▼
        embedding_dim*  ─┐
                        ├─ concat action_dim → Predictor(hidden_dim*) → embedding_dim
        action_dim ──────┘
            │
            └→ VisualizationDecoder(latent_dim* → image_size^2·in_channels)

    • image_size controls the spatial resolution feeding the encoder.
    • embedding_dim (≈ channel_schedule[-1]) is the latent bottleneck the predictor must match.
    • action_dim is concatenated to embeddings before the predictor MLP.
    • hidden_dim is the predictor’s internal width.
    • latent_dim governs the visualization decoder’s compression when turning embeddings back into images.
    """

    in_channels: int = 3
    image_size: int = 128
    channel_schedule: Tuple[int, ...] = (32, 64, 128, 256)
    latent_dim: int = 256
    hidden_dim: int = 256
    embedding_dim: int = 256
    action_dim: int = 8


@dataclass
class LossWeights:
    jepa: float = 1.0
    sigreg: float = 1.0


@dataclass
class TrainConfig:
    seq_len: int = 8
    batch_size: int = 8
    lr: float = 1e-4
    decoder_lr: float = 1e-4
    grad_clip: float = 0.0
    sigreg_projections: int = 64
    recon_weight: float = 1.0
    device: Optional[str] = "mps"
    total_steps: int = 10_000
    log_every_steps: int = 10
    vis_every_steps: int = 50
    vis_rows: int = 4
    vis_rollout: int = 4
    embedding_projection_samples: int = 256
    output_dir: Path = Path("out.jepa_world_model_trainer")
    data_root: Path = Path("data.gridworldkey")
    max_trajectories: Optional[int] = None

    loss_weights: LossWeights = field(default_factory=LossWeights)


class JEPAWorldModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.in_channels, cfg.channel_schedule)
        enc_dim = cfg.channel_schedule[-1]
        self.embedding_dim = enc_dim
        pred_in = self.embedding_dim + cfg.action_dim
        self.predictor = PredictorNetwork(pred_in, cfg.hidden_dim, self.embedding_dim)

    def rollout(self, images: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, t, _, _, _ = images.shape
        embeddings: List[torch.Tensor] = []
        for step in range(t):
            feats = self.encoder(images[:, step])
            embeddings.append(feats)
        return {
            "embeddings": torch.stack(embeddings, dim=1),
        }


# ------------------------------------------------------------
# Loss utilities
# ------------------------------------------------------------


def _temporal_weights(errors: torch.Tensor, eps: float = 1e-6, clamp: float = 2.0) -> torch.Tensor:
    """Compute temporal hardness weights from per-step error magnitudes.

    Args:
        errors: Per-sample error for each transition, shape [B, T-1].
        eps: Small constant to avoid division by zero.
        clamp: Symmetric clamp value for ratios (>= 1).
    """
    if errors.dim() != 2:
        raise ValueError("errors must be 2D with shape [B, T-1]")
    prev = torch.cat([errors[:, :1], errors[:, :-1]], dim=1)
    ratios = (errors + eps) / (prev + eps)
    if clamp is not None and clamp > 1.0:
        ratios = ratios.clamp(1.0 / clamp, clamp)
    weights = ratios / ratios.mean().clamp_min(eps)
    return weights.detach()


def jepa_loss(model: JEPAWorldModel, outputs: Dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
    embeddings = outputs["embeddings"]
    preds = model.predictor(torch.cat([embeddings[:, :-1], actions[:, :-1]], dim=-1))
    future = embeddings[:, 1:]
    diff_for_stats = torch.abs(preds.detach() - future.detach()).mean(dim=-1)
    weights = _temporal_weights(diff_for_stats)
    per_step_loss = torch.abs(preds - future.detach()).mean(dim=-1)
    per_step_loss_enc = torch.abs(future - preds.detach()).mean(dim=-1)
    # Mirror Dreamer's balanced loss: each branch sees its counterpart as a fixed target.
    loss_predictor = (weights * per_step_loss).mean()
    loss_encoder = (weights * per_step_loss_enc).mean()
    return 0.5 * (loss_predictor + loss_encoder)


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
    return SIGREG_LOSS(sorted_proj, sorted_normal)


def reconstruction_loss(decoder: VisualizationDecoder, embeddings: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    recons = decoder(embeddings)
    return RECON_LOSS(recons, images)


# ------------------------------------------------------------
# Training loop utilities
# ------------------------------------------------------------


def training_step(
    model: JEPAWorldModel,
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
    loss_sigreg = sigreg_loss(outputs["embeddings"], cfg.sigreg_projections)
    loss_recon = reconstruction_loss(decoder, outputs["embeddings"].detach(), images)

    world_loss = (
        weights.jepa * loss_jepa
        + weights.sigreg * loss_sigreg
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

    decoder_grad_norm = grad_norm(decoder.parameters())
    optim_decoder.step()

    return {
        "loss_jepa": loss_jepa.item(),
        "loss_sigreg": loss_sigreg.item(),
        "loss_recon": loss_recon.item(),
        "loss_world": world_loss.item(),
        "grad_world": world_grad_norm,
        "grad_decoder": decoder_grad_norm,
    }


# ------------------------------------------------------------
# Example dataset + dataloader
# ------------------------------------------------------------


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
                raise FileNotFoundError(f"Missing actions.npz for {traj_name}")
            with np.load(actions_path) as data:
                action_arr = data["actions"] if "actions" in data else data[list(data.files)[0]]
            if action_arr.ndim == 1:
                action_arr = action_arr[:, None]
            assert action_arr.shape[0] == len(frame_paths), (
                f"Action count {action_arr.shape[0]} does not match frame count {len(frame_paths)} in {traj_name}"
            )
            action_arr = action_arr.astype(np.float32, copy=False)
            if self.action_dim is None:
                self.action_dim = action_arr.shape[1]
            elif self.action_dim != action_arr.shape[1]:
                raise ValueError(
                    f"Inconsistent action dimension for {traj_name}: expected {self.action_dim}, got {action_arr.shape[1]}"
                )
            if len(frame_paths) < self.seq_len:
                raise ValueError(f"Trajectory {traj_name} shorter than seq_len {self.seq_len}")
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
        # Each frame/action pair must stay aligned so the predictor knows which action follows each observation.
        assert len(frames) == self.seq_len, f"Expected {self.seq_len} frames, got {len(frames)}"
        assert action_slice.shape[0] == self.seq_len, (
            f"Expected {self.seq_len} actions, got {action_slice.shape[0]}"
        )
        assert action_slice.shape[1] == actions.shape[1], "Action dimensionality changed unexpectedly."
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
    "loss_sigreg",
    "loss_recon",
    "grad_world",
    "grad_decoder",
]


@dataclass
class LossHistory:
    steps: List[float] = field(default_factory=list)
    world: List[float] = field(default_factory=list)
    jepa: List[float] = field(default_factory=list)
    sigreg: List[float] = field(default_factory=list)
    recon: List[float] = field(default_factory=list)
    grad_world: List[float] = field(default_factory=list)
    grad_decoder: List[float] = field(default_factory=list)

    def append(self, step: float, metrics: Dict[str, float]) -> None:
        self.steps.append(step)
        self.world.append(metrics["loss_world"])
        self.jepa.append(metrics["loss_jepa"])
        self.sigreg.append(metrics["loss_sigreg"])
        self.recon.append(metrics["loss_recon"])
        self.grad_world.append(metrics["grad_world"])
        self.grad_decoder.append(metrics["grad_decoder"])

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
            history.sigreg,
            history.recon,
            history.grad_world,
            history.grad_decoder,
        ):
            writer.writerow(row)


def plot_loss_curves(history: LossHistory, out_dir: Path) -> None:
    if len(history) == 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history.steps, history.world, label="world")
    plt.plot(history.steps, history.jepa, label="jepa")
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
    try:
        u, s, vt = np.linalg.svd(flat, full_matrices=False)
    except np.linalg.LinAlgError:
        jitter = np.random.normal(scale=1e-6, size=flat.shape)
        try:
            u, s, vt = np.linalg.svd(flat + jitter, full_matrices=False)
        except np.linalg.LinAlgError:
            print("Warning: embedding SVD failed to converge; skipping projection.")
            return
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


def _run_git_command(args: List[str]) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return "git executable not found."
    if result.returncode != 0:
        stderr = result.stderr.strip()
        return f"Command {' '.join(args)} failed (code {result.returncode}): {stderr}"
    return result.stdout.strip()


def _serialize_for_json(value):
    if isinstance(value, dict):
        return {k: _serialize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_for_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_run_metadata(run_dir: Path, cfg: TrainConfig, model_cfg: ModelConfig) -> None:
    commit_sha = _run_git_command(["git", "rev-parse", "HEAD"])
    diff_output = _run_git_command(["git", "diff", "--patch"])
    if not diff_output.strip():
        diff_output = "No uncommitted changes."
    git_metadata = "\n".join(
        [
            "Git commit:",
            commit_sha or "Unavailable",
            "",
            "Git diff (uncommitted changes):",
            diff_output,
            "",
        ]
    )
    (run_dir / "metadata_git.txt").write_text(git_metadata)
    model_metadata = {
        "train_config": _serialize_for_json(asdict(cfg)),
        "model_config": _serialize_for_json(asdict(model_cfg)),
        "data_root": str(cfg.data_root),
    }
    toml_lines = []
    toml_lines.append("[train_config]")
    for k, v in model_metadata["train_config"].items():
        toml_lines.append(f"{k} = {json.dumps(v)}")
    toml_lines.append("")
    toml_lines.append("[model_config]")
    for k, v in model_metadata["model_config"].items():
        toml_lines.append(f"{k} = {json.dumps(v)}")
    toml_lines.append("")
    toml_lines.append(f'data_root = "{model_metadata["data_root"]}"')
    (run_dir / "metadata.txt").write_text("\n".join(toml_lines))


def save_rollout_visualization(
    out_path: Path,
    frames: torch.Tensor,
    recon_frames: torch.Tensor,
    pred_rollouts: Optional[torch.Tensor],
    rows: int,
    rollout_steps: int,
) -> None:
    frames = frames.detach().cpu()
    recon_frames = recon_frames.detach().cpu()
    pred_frames = pred_rollouts.detach().cpu() if pred_rollouts is not None else None
    total_rows = frames.shape[0]
    num_rows = min(rows, total_rows)
    pred_horizons = 0 if pred_frames is None else pred_frames.shape[1]
    grid_rows: List[np.ndarray] = []
    for row_idx in range(num_rows):
        columns = [
            tensor_to_uint8_image(frames[row_idx]),
            tensor_to_uint8_image(recon_frames[row_idx]),
        ]
        for horizon in range(rollout_steps):
            if (
                pred_frames is None
                or horizon >= pred_horizons
            ):
                columns.append(columns[-1])
                continue
            columns.append(tensor_to_uint8_image(pred_frames[row_idx, horizon]))
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
    (metrics_dir).mkdir(parents=True, exist_ok=True)
    fixed_vis_dir = run_dir / "vis_fixed"
    rolling_vis_dir = run_dir / "vis_rolling"
    embeddings_vis_dir = run_dir / "embeddings"
    fixed_vis_dir.mkdir(parents=True, exist_ok=True)
    rolling_vis_dir.mkdir(parents=True, exist_ok=True)
    embeddings_vis_dir.mkdir(parents=True, exist_ok=True)
    loss_history = LossHistory()
    write_run_metadata(run_dir, cfg, model_cfg)
    dataset = TrajectorySequenceDataset(
        root=cfg.data_root,
        seq_len=cfg.seq_len,
        image_hw=(model_cfg.image_size, model_cfg.image_size),
        max_traj=cfg.max_trajectories,
    )
    dataset_action_dim = getattr(dataset, "action_dim", model_cfg.action_dim)
    assert dataset_action_dim == 8, f"Expected action_dim 8, got {dataset_action_dim}"
    if model_cfg.action_dim != dataset_action_dim:
        model_cfg = replace(model_cfg, action_dim=dataset_action_dim)
    model = JEPAWorldModel(model_cfg).to(device)
    decoder = VisualizationDecoder(model.embedding_dim, model_cfg.in_channels, model_cfg.image_size).to(device)
    optim_world = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    optim_decoder = torch.optim.Adam(decoder.parameters(), lr=cfg.decoder_lr)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_batch)

    fixed_batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    try:
        sample_batch = next(iter(dataloader))
        fixed_batch_cpu = (sample_batch[0].clone(), sample_batch[1].clone())
    except StopIteration:
        fixed_batch_cpu = None
    rolling_batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    embedding_batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    if cfg.embedding_projection_samples > 0:
        embed_loader = DataLoader(
            dataset,
            batch_size=min(cfg.embedding_projection_samples, len(dataset)),
            shuffle=True,
            collate_fn=collate_batch,
        )
        try:
            embed_batch = next(iter(embed_loader))
            embedding_batch_cpu = (embed_batch[0].clone(), embed_batch[1].clone())
        except StopIteration:
            embedding_batch_cpu = None

    data_iter = iter(dataloader)
    for global_step in range(cfg.total_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        metrics = training_step(model, decoder, optim_world, optim_decoder, batch, cfg, weights)
        if cfg.log_every_steps > 0 and global_step % cfg.log_every_steps == 0:
            log_metrics(global_step, metrics)
            loss_history.append(global_step, metrics)
            write_loss_csv(loss_history, metrics_dir / "loss.csv")
            plot_loss_curves(loss_history, metrics_dir)
        rolling_batch_cpu = (batch[0].clone(), batch[1].clone())
        if (
            cfg.vis_every_steps > 0
            and global_step % cfg.vis_every_steps == 0
        ):
            model.eval()
            with torch.no_grad():
                def render_sample(
                    batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor]],
                    out_dir: Path,
                ) -> None:
                    if batch_cpu is None:
                        return
                    vis_frames = batch_cpu[0].to(device)
                    vis_actions = batch_cpu[1].to(device)
                    vis_outputs = model.rollout(vis_frames, vis_actions)
                    vis_embeddings = vis_outputs["embeddings"]
                    recon_frames = decoder(vis_embeddings).clamp(0, 1)
                    batch_size = vis_frames.shape[0]
                    if batch_size == 0:
                        return
                    num_rows = min(cfg.vis_rows, batch_size)
                    row_indices = torch.randperm(batch_size, device=device)[:num_rows]
                    sampled_frames: List[torch.Tensor] = []
                    sampled_recons: List[torch.Tensor] = []
                    sampled_preds: List[torch.Tensor] = []
                    for idx in row_indices:
                        time_idx = torch.randint(0, vis_frames.shape[1], ()).item()
                        sampled_frames.append(vis_frames[idx, time_idx])
                        sampled_recons.append(recon_frames[idx, time_idx])
                        if cfg.vis_rollout > 0:
                            seq_preds: List[torch.Tensor] = []
                            current_embed = vis_embeddings[idx, time_idx].unsqueeze(0)
                            current_action = vis_actions[idx, time_idx].unsqueeze(0)
                            for _ in range(cfg.vis_rollout):
                                pred_input = torch.cat([current_embed, current_action], dim=-1)
                                pred_embed = model.predictor(pred_input)
                                decoded = decoder(pred_embed).clamp(0, 1)[0]
                                seq_preds.append(decoded)
                                current_embed = pred_embed
                            sampled_preds.append(torch.stack(seq_preds))
                    frames_tensor = torch.stack(sampled_frames, dim=0)
                    recon_tensor = torch.stack(sampled_recons, dim=0)
                    pred_tensor: Optional[torch.Tensor]
                    if cfg.vis_rollout > 0 and len(sampled_preds) == len(sampled_frames):
                        pred_tensor = torch.stack(sampled_preds, dim=0)
                    else:
                        pred_tensor = None
                    save_rollout_visualization(
                        out_dir / f"rollout_{global_step:07d}.png",
                        frames_tensor,
                        recon_tensor,
                        pred_tensor,
                        cfg.vis_rows,
                        cfg.vis_rollout,
                    )

                render_sample(fixed_batch_cpu, fixed_vis_dir)
                render_sample(rolling_batch_cpu, rolling_vis_dir)
                if embedding_batch_cpu is not None:
                    embed_frames = embedding_batch_cpu[0].to(device)
                    embed_actions = embedding_batch_cpu[1].to(device)
                    embed_outputs = model.rollout(embed_frames, embed_actions)
                    save_embedding_projection(
                        embed_outputs["embeddings"],
                        embeddings_vis_dir / f"embeddings_{global_step:07d}.png",
                    )
            model.train()
    if len(loss_history):
        write_loss_csv(loss_history, metrics_dir / "loss.csv")
        plot_loss_curves(loss_history, metrics_dir)


def main() -> None:
    cfg = tyro.cli(TrainConfig)
    model_cfg = ModelConfig()
    run_training(cfg, model_cfg, cfg.loss_weights, demo=False)


if __name__ == "__main__":
    main()
