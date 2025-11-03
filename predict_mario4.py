#!/usr/bin/env python3
"""Training loop for a single-frame Mario predictor with action-conditioned hidden state.

This variant encodes individual frames, reconstructs them, and propagates a latent
hidden state that is updated via stacked FiLM modulation, a selective SSM cell,
and a GRU transition conditioned on the current action.
"""

from __future__ import annotations

import csv
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from torch import nn
from torch.utils.data import DataLoader, Dataset

from predict_mario_ms_ssim import pick_device
from recon.data import list_trajectories, load_frame_as_tensor


plt.switch_backend("Agg")
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log transform used to damp outliers."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symlog_loss(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Symlog regression loss."""
    diff = pred - target
    transformed = symlog(diff)
    loss = transformed.pow(2)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unsupported reduction: {reduction}")


def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Sample using the reparameterization trick for a diagonal Gaussian."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std


def kl_divergence_diag_gaussians(
    mean_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mean_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    """KL divergence between two diagonal Gaussians."""
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    term1 = logvar_p - logvar_q
    term2 = (var_q + (mean_q - mean_p).pow(2)) / var_p
    return 0.5 * (term1 + term2 - 1.0)


@dataclass
class TrainConfig:
    data_root: Path = Path("data.image_distance.train_levels_1_2")
    output_dir: Path = Path("out.predict_mario4")
    epochs: int = 1000
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_workers: int = 4
    device: str = "mps"
    seed: int = 0
    warmup_steps: int = 4
    rollout_steps: int = 4
    image_hw: Tuple[int, int] = (224, 224)
    hidden_dim: int = 256
    latent_dim: int = 192
    film_layers: int = 2
    action_loss_weight: float = 1.0
    beta_pred: float = 1.0
    beta_dyn: float = 0.95
    beta_rep: float = 0.05
    log_every_steps: int = 10
    vis_every_steps: int = 100
    checkpoint_every_steps: int = 1000
    max_trajectories: Optional[int] = None


class MarioWarmupDataset(Dataset):
    """Return contiguous sequences of frames/actions for warmup + rollout training."""

    def __init__(
        self,
        *,
        root: Path,
        warmup_steps: int,
        rollout_steps: int,
        image_hw: Tuple[int, int],
        max_traj: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.warmup_steps = warmup_steps
        self.rollout_steps = rollout_steps
        self.total_steps = warmup_steps + rollout_steps

        if self.total_steps < 1:
            raise ValueError("Expected warmup_steps + rollout_steps >= 1.")
        self.image_hw = image_hw

        trajectories = list_trajectories(self.root)
        items = list(trajectories.items())
        if max_traj is not None:
            items = items[:max_traj]

        self._samples: List[Tuple[List[Path], np.ndarray, int]] = []
        self.action_dim: Optional[int] = None

        for traj_name, frame_paths in items:
            actions_path = (self.root / traj_name) / "actions.npz"
            if not actions_path.is_file():
                logger.warning("Missing actions.npz for %s; skipping.", traj_name)
                continue
            try:
                with np.load(actions_path) as data:
                    if "actions" in data:
                        actions = data["actions"]
                    else:
                        actions = data[list(data.files)[0]]
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to read %s: %s", actions_path, exc)
                continue

            if actions.ndim == 1:
                actions = actions[:, None]
            if actions.shape[0] == len(frame_paths) + 1:
                actions = actions[1:]
            if actions.shape[0] < len(frame_paths):
                logger.warning(
                    "Action count %d shorter than frames %d in %s; skipping.",
                    actions.shape[0],
                    len(frame_paths),
                    traj_name,
                )
                continue

            actions = actions.astype(np.float32)
            if self.action_dim is None:
                self.action_dim = actions.shape[1]
            elif self.action_dim != actions.shape[1]:
                logger.warning(
                    "Inconsistent action dim in %s (expected %d, found %d); skipping.",
                    traj_name,
                    self.action_dim,
                    actions.shape[1],
                )
                continue

            if len(frame_paths) < self.total_steps:
                continue

            max_start = len(frame_paths) - self.total_steps
            for start in range(max_start + 1):
                self._samples.append((frame_paths, actions, start))

        if not self._samples:
            raise RuntimeError(f"No usable sequences found under {self.root}")
        if self.action_dim is None:
            raise RuntimeError("Unable to infer action dimensionality.")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        frame_paths, actions, start = self._samples[index]
        frames: List[torch.Tensor] = []
        path_slice: List[str] = []
        for offset in range(self.total_steps):
            frame = load_frame_as_tensor(
                frame_paths[start + offset],
                size=self.image_hw,
                normalize=None,
            )
            frames.append(frame)
            path_slice.append(str(frame_paths[start + offset]))
        action_slice = actions[start : start + self.total_steps]
        return {
            "frames": torch.stack(frames, dim=0),
            "actions": torch.from_numpy(action_slice),
            "frame_paths": path_slice,
        }


def extract_traj_state_label(frame_path: str) -> str:
    path = Path(frame_path)
    traj = next((part for part in path.parts if part.startswith("traj_")), "traj_unknown")
    state = path.stem
    return f"{traj}/{state}"


def build_frame_labels(frame_paths: List[str], warmup_steps: int) -> Tuple[List[str], List[str]]:
    if warmup_steps > len(frame_paths):
        raise ValueError("warmup_steps exceeds number of frame paths.")
    labels = [extract_traj_state_label(p) for p in frame_paths]
    warmup_labels = [f"{label} | warmup" for label in labels[:warmup_steps]]
    predicted_labels = [f"{label} | predicted" for label in labels[warmup_steps:]]
    expected_predicted = len(labels) - warmup_steps
    assert expected_predicted >= 0, "Expected non-negative predicted frame count."
    assert len(predicted_labels) >= expected_predicted, (
        f"Expected at least {expected_predicted} predicted labels, received {len(predicted_labels)}."
    )
    predicted_labels = predicted_labels[:expected_predicted]
    assert len(warmup_labels) == warmup_steps, (
        f"Expected {warmup_steps} warmup labels, received {len(warmup_labels)}."
    )
    assert len(predicted_labels) == expected_predicted, (
        f"Expected {expected_predicted} predicted labels, received {len(predicted_labels)}."
    )
    return warmup_labels, predicted_labels


def extract_paths_for_sample(batch_paths: Sequence[Any], sample_idx: int) -> List[str]:
    assert isinstance(batch_paths, (list, tuple)), (
        f"Unexpected frame_paths type: {type(batch_paths)!r}"
    )
    if not batch_paths:
        return []
    first = batch_paths[0]
    if isinstance(first, (list, tuple)):
        paths = [seq[sample_idx] for seq in batch_paths]
    elif isinstance(first, (str, Path)):
        assert sample_idx == 0, "String-based frame_paths only support sample_idx == 0."
        paths = batch_paths
    else:
        raise TypeError(f"Unsupported frame_paths element type: {type(first)!r}")
    return [str(path) for path in paths]


class ImageEncoder(nn.Module):
    """Encode a single RGB frame into a latent vector."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        channels = [3, 32, 64, 128, 192]
        layers: List[nn.Module] = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.GELU())
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(channels[-1], latent_dim)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        x = self.features(frame)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class ImageDecoder(nn.Module):
    """Project a latent vector back to an RGB frame."""

    def __init__(self, latent_dim: int, base: int = 32) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, base * 8 * 14 * 14)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base * 8, base * 4, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(base * 4, base * 2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(base * 2, base, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(base, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.fc(latent)
        x = x.view(latent.size(0), -1, 14, 14)
        x = self.net(x)
        return torch.sigmoid(x)


class ActionEmbedding(nn.Module):
    """Embed controller actions for FiLM modulation."""

    def __init__(self, action_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.net(actions)


class DiagonalGaussianMLP(nn.Module):
    """Produce mean and log-variance for a diagonal Gaussian latent."""

    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.net(inputs)
        mean, logvar = torch.chunk(params, 2, dim=-1)
        return mean, logvar


class RepresentationModel(nn.Module):
    """Posterior over latent state conditioned on deterministic state and encoded features."""

    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        in_dim = hidden_dim + latent_dim
        self.posterior = DiagonalGaussianMLP(in_dim, hidden_dim, latent_dim)

    def forward(self, hidden: torch.Tensor, encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat([hidden, encoded], dim=-1)
        return self.posterior(inputs)


class DynamicsModel(nn.Module):
    """Prior over latent state conditioned on deterministic hidden state."""

    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.prior = DiagonalGaussianMLP(hidden_dim, hidden_dim, latent_dim)

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.prior(hidden)


class FiLMLayer(nn.Module):
    """Single FiLM modulation layer."""

    def __init__(self, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(action_dim, hidden_dim)
        self.beta = nn.Linear(action_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: torch.Tensor, action_embed: torch.Tensor) -> torch.Tensor:
        conditioned = self.norm(hidden)
        gamma = self.gamma(action_embed)
        beta = self.beta(action_embed)
        return conditioned * (1.0 + gamma) + beta


class ActionFiLM(nn.Module):
    """Stack multiple FiLM layers for action conditioning."""

    def __init__(self, hidden_dim: int, action_dim: int, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FiLMLayer(hidden_dim, action_dim) for _ in range(layers)])
        self.act = nn.GELU()

    def forward(self, hidden: torch.Tensor, action_embed: torch.Tensor) -> torch.Tensor:
        out = hidden
        for layer in self.layers:
            out = layer(out, action_embed)
            out = self.act(out)
        return out


class SelectiveSSMCell(nn.Module):
    """
    h_t = (A ⊙ g_t + (1 - g_t) ⊙ I) h_{t-1} + B(x_t_mod)
    y_t = LN(W_o h_t)   # optional head

    - A is diagonal (stable) for speed/stability; B, W_o are linear maps
    - g_t is a gate in [0,1]^D, computed from [x_t_mod, h_{t-1}]
    - x_t_mod optionally includes action conditioning (FiLM)
    """

    def __init__(self, state_dim: int, x_dim: int, y_dim: Optional[int] = None) -> None:
        super().__init__()
        D = state_dim
        self.D = D

        self.a_diag = nn.Parameter(torch.zeros(D))
        self.B = nn.Linear(x_dim, D)
        self.gate = nn.Sequential(
            nn.Linear(D + x_dim, 2 * D),
            nn.SiLU(),
            nn.Linear(2 * D, D),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(D)
        self.out_proj = nn.Linear(D, y_dim) if y_dim is not None else None

    def forward(self, x_t_mod: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        A = -F.softplus(self.a_diag)
        g = self.gate(torch.cat([x_t_mod, h_prev], dim=-1))
        A_eff = A.unsqueeze(0) * g + (1.0 - g)
        h_t = A_eff * h_prev + self.B(x_t_mod)

        if self.out_proj is not None:
            y_t = self.out_proj(self.out_norm(h_t))
            return h_t, y_t
        return h_t, h_t


class HiddenUpdateCell(nn.Module):
    """Update hidden state with FiLM + selective SSM + GRU."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        latent_dim: int,
        action_dim: int,
        film_layers: int,
    ) -> None:
        super().__init__()
        self.preprocess = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.film = ActionFiLM(hidden_dim, action_dim, film_layers)
        self.ssm = SelectiveSSMCell(state_dim=hidden_dim, x_dim=hidden_dim, y_dim=None)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.post_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        latent: torch.Tensor,
        action_embed: torch.Tensor,
    ) -> torch.Tensor:
        x_t = torch.cat([hidden, latent], dim=-1)
        x_t = self.preprocess(x_t)
        x_mod = self.film(x_t, action_embed)
        s_t, _ = self.ssm(x_mod, hidden)
        h_t = self.gru(s_t, hidden)
        return self.post_norm(h_t)


class Mario4Model(nn.Module):
    """End-to-end model for single-frame conditioning and hidden state propagation."""

    def __init__(
        self,
        *,
        action_dim: int,
        hidden_dim: int,
        latent_dim: int,
        film_layers: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder = ImageEncoder(latent_dim)
        self.decoder = ImageDecoder(latent_dim)
        self.representation = RepresentationModel(hidden_dim, latent_dim)
        self.dynamics = DynamicsModel(hidden_dim, latent_dim)
        self.action_embed = ActionEmbedding(action_dim, hidden_dim)
        self.hidden_cell = HiddenUpdateCell(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            action_dim=hidden_dim,
            film_layers=film_layers,
        )
        self.policy_head = nn.Linear(hidden_dim + latent_dim, action_dim)

    def forward(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        *,
        warmup_steps: int,
    ) -> Dict[str, torch.Tensor]:
        if frames.shape[0] != actions.shape[0] or frames.shape[1] != actions.shape[1]:
            raise ValueError("Frames and actions must share leading batch/time dimensions.")

        batch_size, total_steps = frames.shape[0], frames.shape[1]
        if total_steps <= warmup_steps:
            raise ValueError("Total steps must exceed warmup_steps to compute training losses.")

        device = frames.device
        hidden = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=frames.dtype)

        # Warmup hidden state without tracking gradients.
        with torch.no_grad():
            for step in range(min(warmup_steps, total_steps)):
                frame = frames[:, step]
                action = actions[:, step]
                encoded = self.encoder(frame)
                post_mean, post_logvar = self.representation(hidden, encoded)
                latent = reparameterize(post_mean, post_logvar)
                action_embed = self.action_embed(action)
                hidden = self.hidden_cell(hidden, latent, action_embed)
        hidden = hidden.detach()

        reconstructions: List[torch.Tensor] = []
        action_logits: List[torch.Tensor] = []
        hidden_states: List[torch.Tensor] = []
        posterior_means: List[torch.Tensor] = []
        posterior_logvars: List[torch.Tensor] = []
        prior_means: List[torch.Tensor] = []
        prior_logvars: List[torch.Tensor] = []

        for step in range(warmup_steps, total_steps):
            frame = frames[:, step]
            action = actions[:, step]
            encoded = self.encoder(frame)
            prior_mean, prior_logvar = self.dynamics(hidden)
            post_mean, post_logvar = self.representation(hidden, encoded)
            latent = reparameterize(post_mean, post_logvar)
            action_embed = self.action_embed(action)
            hidden = self.hidden_cell(hidden, latent, action_embed)
            recon = self.decoder(latent)
            logits = self.policy_head(torch.cat([hidden, latent], dim=-1))
            reconstructions.append(recon)
            action_logits.append(logits)
            hidden_states.append(hidden)
            posterior_means.append(post_mean)
            posterior_logvars.append(post_logvar)
            prior_means.append(prior_mean)
            prior_logvars.append(prior_logvar)

        return {
            "reconstructions": torch.stack(reconstructions, dim=1),
            "action_logits": torch.stack(action_logits, dim=1),
            "hidden_states": torch.stack(hidden_states, dim=1),
            "posterior_mean": torch.stack(posterior_means, dim=1),
            "posterior_logvar": torch.stack(posterior_logvars, dim=1),
            "prior_mean": torch.stack(prior_means, dim=1),
            "prior_logvar": torch.stack(prior_logvars, dim=1),
        }


@dataclass
class LossHistory:
    steps: List[float]
    total: List[float]
    pred: List[float]
    recon: List[float]
    action: List[float]
    dyn: List[float]
    rep: List[float]

    @classmethod
    def empty(cls) -> "LossHistory":
        return cls([], [], [], [], [], [], [])

    def append(
        self,
        *,
        step: float,
        total: float,
        pred: float,
        recon: float,
        action: float,
        dyn: float,
        rep: float,
    ) -> None:
        self.steps.append(step)
        self.total.append(total)
        self.pred.append(pred)
        self.recon.append(recon)
        self.action.append(action)
        self.dyn.append(dyn)
        self.rep.append(rep)

    def __len__(self) -> int:
        return len(self.steps)

    def rows(self) -> Dict[str, List[float]]:
        return {
            "step": self.steps,
            "total": self.total,
            "pred": self.pred,
            "recon": self.recon,
            "action": self.action,
            "dyn": self.dyn,
            "rep": self.rep,
        }


LOSS_COLUMNS = ["step", "total", "pred", "recon", "action", "dyn", "rep"]


def write_loss_csv(history: LossHistory, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LOSS_COLUMNS)
        rows = history.rows()
        for idx in range(len(history)):
            writer.writerow([rows[column][idx] for column in LOSS_COLUMNS])


def plot_loss_curves(history: LossHistory, out_dir: Path) -> None:
    if len(history) == 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = history.steps

    plt.figure(figsize=(8, 5))
    plt.plot(steps, history.total, label="total")
    plt.plot(steps, history.pred, label="pred")
    plt.plot(steps, history.recon, label="recon")
    plt.plot(steps, history.action, label="action")
    plt.plot(steps, history.dyn, label="dyn")
    plt.plot(steps, history.rep, label="rep")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=200)
    plt.close()


def make_dataloader(cfg: TrainConfig) -> DataLoader:
    dataset = MarioWarmupDataset(
        root=cfg.data_root,
        warmup_steps=cfg.warmup_steps,
        rollout_steps=cfg.rollout_steps,
        image_hw=cfg.image_hw,
        max_traj=cfg.max_trajectories,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )


def visualize_reconstruction(
    *,
    out_path: Path,
    warmup_frames: torch.Tensor,
    predicted_frames: torch.Tensor,
    hidden_reconstructions: torch.Tensor,
    direct_warmup: torch.Tensor,
    direct_predicted: torch.Tensor,
    warmup_labels: Sequence[str],
    predicted_labels: Sequence[str],
    row_labels: Tuple[str, str, str] = (
        "Ground Truth",
        "Hidden-State Reconstruction",
        "Direct Encoder/Decoder",
    ),
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    warmup = torch.clamp(warmup_frames[0].detach().cpu(), 0.0, 1.0)
    predicted = torch.clamp(predicted_frames[0].detach().cpu(), 0.0, 1.0)
    hidden_pred = torch.clamp(hidden_reconstructions[0].detach().cpu(), 0.0, 1.0)
    direct_w = torch.clamp(direct_warmup[0].detach().cpu(), 0.0, 1.0)
    direct_p = torch.clamp(direct_predicted[0].detach().cpu(), 0.0, 1.0)

    warmup_cols = warmup.shape[0]
    predicted_cols = predicted.shape[0]
    warmup_labels = list(warmup_labels)
    predicted_labels = list(predicted_labels)
    assert len(warmup_labels) == warmup_cols, (
        f"Expected {warmup_cols} warmup labels, received {len(warmup_labels)}."
    )
    assert len(predicted_labels) == predicted_cols, (
        f"Expected {predicted_cols} predicted labels, received {len(predicted_labels)}."
    )
    total_cols = warmup_cols + predicted_cols
    all_labels = warmup_labels + predicted_labels

    fig, axes = plt.subplots(3, total_cols, figsize=(total_cols * 2, 6))
    blank = torch.ones_like(warmup[0]) * 0.5

    def _imshow(ax: plt.Axes, tensor: torch.Tensor) -> None:
        ax.imshow(tensor.permute(1, 2, 0).numpy())
        ax.axis("off")

    for idx, label in enumerate(all_labels):
        if idx < warmup_cols:
            _imshow(axes[0, idx], warmup[idx])
            _imshow(axes[1, idx], blank)
            _imshow(axes[2, idx], direct_w[idx])
        else:
            pred_idx = idx - warmup_cols
            _imshow(axes[0, idx], predicted[pred_idx])
            _imshow(axes[1, idx], hidden_pred[pred_idx])
            _imshow(axes[2, idx], direct_p[pred_idx])

        axes[0, idx].set_title(label, fontsize=8)

    fig.suptitle("Ground truth vs. hidden-path vs. direct reconstructions")
    fig.tight_layout(rect=(0.08, 0.0, 1.0, 0.95))

    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].text(
            -0.12,
            0.5,
            label,
            transform=axes[row_idx, 0].transAxes,
            va="center",
            ha="right",
            fontsize=8,
            rotation=90,
        )

    fig.savefig(out_path)
    plt.close(fig)


def save_checkpoint(
    *,
    output_dir: Path,
    name: str,
    step: int,
    epoch: int,
    metric: float,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Path:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "epoch": epoch,
        "metric": metric,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    path = ckpt_dir / name
    torch.save(state, path)
    return path


def train(cfg: TrainConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger.info("Configuration: %s", cfg)
    set_seed(cfg.seed)

    device_pref = None if cfg.device == "auto" else cfg.device
    device = pick_device(device_pref)
    logger.info("Using device: %s", device)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = cfg.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", run_dir)

    dataloader = make_dataloader(cfg)
    model = Mario4Model(
        action_dim=dataloader.dataset.action_dim,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        film_layers=cfg.film_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    metrics_dir = run_dir / "metrics"
    samples_dir = run_dir / "samples"
    checkpoints_dir = run_dir / "checkpoints"
    for directory in (metrics_dir, samples_dir, checkpoints_dir):
        directory.mkdir(parents=True, exist_ok=True)

    loss_history = LossHistory.empty()
    fixed_batch_cpu: Optional[Dict[str, Any]] = None
    best_metric = float("inf")
    last_loss = float("inf")
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_recon = 0.0
        running_action = 0.0
        running_pred = 0.0
        running_dyn = 0.0
        running_rep = 0.0
        running_total = 0.0
        window_data_time = 0.0
        window_forward_time = 0.0
        window_backward_time = 0.0
        epoch_start = perf_counter()

        for batch_idx, batch in enumerate(dataloader, start=1):
            batch_start = perf_counter()
            frames = batch["frames"].to(device)
            actions = batch["actions"].to(device)

            if fixed_batch_cpu is None:
                fixed_batch_cpu = {
                    "frames": batch["frames"][:1].detach().cpu(),
                    "actions": batch["actions"][:1].detach().cpu(),
                    "frame_paths": extract_paths_for_sample(batch["frame_paths"], 0),
                }

            transfer_end = perf_counter()

            outputs = model(frames, actions, warmup_steps=cfg.warmup_steps)
            target_frames = frames[:, cfg.warmup_steps :]
            target_actions = actions[:, cfg.warmup_steps :]
            forward_end = perf_counter()

            recon_loss = symlog_loss(outputs["reconstructions"], target_frames)
            action_probs = torch.sigmoid(outputs["action_logits"])
            action_loss = symlog_loss(action_probs, target_actions)
            prediction_loss = recon_loss + cfg.action_loss_weight * action_loss

            posterior_mean = outputs["posterior_mean"]
            posterior_logvar = outputs["posterior_logvar"]
            prior_mean = outputs["prior_mean"]
            prior_logvar = outputs["prior_logvar"]

            kl_dyn = kl_divergence_diag_gaussians(
                posterior_mean.detach(),
                posterior_logvar.detach(),
                prior_mean,
                prior_logvar,
            ).sum(dim=-1)
            kl_dyn = torch.maximum(kl_dyn, torch.ones_like(kl_dyn))
            dynamics_loss = kl_dyn.mean()

            kl_rep = kl_divergence_diag_gaussians(
                posterior_mean,
                posterior_logvar,
                prior_mean.detach(),
                prior_logvar.detach(),
            ).sum(dim=-1)
            kl_rep = torch.maximum(kl_rep, torch.ones_like(kl_rep))
            representation_loss = kl_rep.mean()

            loss = (
                cfg.beta_pred * prediction_loss
                + cfg.beta_dyn * dynamics_loss
                + cfg.beta_rep * representation_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            backward_end = perf_counter()

            current_loss = float(loss.item())
            last_loss = current_loss
            running_recon += recon_loss.item()
            running_action += action_loss.item()
            running_pred += prediction_loss.item()
            running_dyn += dynamics_loss.item()
            running_rep += representation_loss.item()
            running_total += loss.item()
            global_step += 1

            window_data_time += transfer_end - batch_start
            window_forward_time += forward_end - transfer_end
            window_backward_time += backward_end - forward_end

            if global_step % cfg.log_every_steps == 0:
                avg_recon = running_recon / cfg.log_every_steps
                avg_action = running_action / cfg.log_every_steps
                avg_pred = running_pred / cfg.log_every_steps
                avg_dyn = running_dyn / cfg.log_every_steps
                avg_rep = running_rep / cfg.log_every_steps
                avg_total = running_total / cfg.log_every_steps
                avg_data = window_data_time / cfg.log_every_steps
                avg_forward = window_forward_time / cfg.log_every_steps
                avg_backward = window_backward_time / cfg.log_every_steps
                logger.info(
                    (
                        "[step %d] total %.4f | pred %.4f (recon %.4f, action %.4f)"
                        " | dyn %.4f | rep %.4f | data %.4f s | forward %.4f s | backward %.4f s"
                    ),
                    global_step,
                    avg_total,
                    avg_pred,
                    avg_recon,
                    avg_action,
                    avg_dyn,
                    avg_rep,
                    avg_data,
                    avg_forward,
                    avg_backward,
                )
                loss_history.append(
                    step=float(global_step),
                    total=avg_total,
                    pred=avg_pred,
                    recon=avg_recon,
                    action=avg_action,
                    dyn=avg_dyn,
                    rep=avg_rep,
                )
                write_loss_csv(loss_history, metrics_dir / "loss.csv")
                plot_loss_curves(loss_history, metrics_dir)
                running_recon = running_action = running_pred = running_dyn = running_rep = running_total = 0.0
                window_data_time = window_forward_time = window_backward_time = 0.0

            if cfg.vis_every_steps > 0 and global_step % cfg.vis_every_steps == 0:
                model.eval()
                with torch.no_grad():
                    if fixed_batch_cpu is not None:
                        fixed_frames_dev = fixed_batch_cpu["frames"].to(device)
                        fixed_actions_dev = fixed_batch_cpu["actions"].to(device)
                        fixed_paths = fixed_batch_cpu["frame_paths"]
                        fixed_warm_labels, fixed_pred_labels = build_frame_labels(fixed_paths, cfg.warmup_steps)
                        fixed_outputs = model(
                            fixed_frames_dev,
                            fixed_actions_dev,
                            warmup_steps=cfg.warmup_steps,
                        )
                        fixed_warmup_dev = fixed_frames_dev[:, : cfg.warmup_steps]
                        fixed_targets_dev = fixed_frames_dev[:, cfg.warmup_steps :]
                        flat_fixed_warmup = fixed_warmup_dev.reshape(-1, *fixed_warmup_dev.shape[2:])
                        flat_fixed_pred = fixed_targets_dev.reshape(-1, *fixed_targets_dev.shape[2:])
                        fixed_direct_warmup = model.decoder(model.encoder(flat_fixed_warmup)).reshape_as(
                            fixed_warmup_dev
                        )
                        fixed_direct_pred = model.decoder(model.encoder(flat_fixed_pred)).reshape_as(
                            fixed_targets_dev
                        )
                        visualize_reconstruction(
                            out_path=samples_dir / f"fixed_step_{global_step:07d}.png",
                            warmup_frames=fixed_warmup_dev.detach().cpu(),
                            predicted_frames=fixed_targets_dev.detach().cpu(),
                            hidden_reconstructions=fixed_outputs["reconstructions"].detach().cpu(),
                            direct_warmup=fixed_direct_warmup.detach().cpu(),
                            direct_predicted=fixed_direct_pred.detach().cpu(),
                            warmup_labels=fixed_warm_labels,
                            predicted_labels=fixed_pred_labels,
                        )
                    rolling_outputs = model(frames, actions, warmup_steps=cfg.warmup_steps)
                    rolling_targets_dev = frames[:, cfg.warmup_steps :]
                    rolling_warmup_dev = frames[:, : cfg.warmup_steps]
                    rolling_paths = extract_paths_for_sample(batch["frame_paths"], 0)
                    rolling_warm_labels, rolling_pred_labels = build_frame_labels(rolling_paths, cfg.warmup_steps)
                    flat_rolling_warmup = rolling_warmup_dev.reshape(-1, *rolling_warmup_dev.shape[2:])
                    flat_rolling_pred = rolling_targets_dev.reshape(-1, *rolling_targets_dev.shape[2:])
                    rolling_direct_warm = model.decoder(model.encoder(flat_rolling_warmup)).reshape_as(
                        rolling_warmup_dev
                    )
                    rolling_direct_pred = model.decoder(model.encoder(flat_rolling_pred)).reshape_as(
                        rolling_targets_dev
                    )
                    visualize_reconstruction(
                        out_path=samples_dir / f"rolling_step_{global_step:07d}.png",
                        warmup_frames=rolling_warmup_dev.detach().cpu(),
                        predicted_frames=rolling_targets_dev.detach().cpu(),
                        hidden_reconstructions=rolling_outputs["reconstructions"].detach().cpu(),
                        direct_warmup=rolling_direct_warm.detach().cpu(),
                        direct_predicted=rolling_direct_pred.detach().cpu(),
                        warmup_labels=rolling_warm_labels,
                        predicted_labels=rolling_pred_labels,
                    )
                model.train()

            if cfg.checkpoint_every_steps > 0 and global_step % cfg.checkpoint_every_steps == 0:
                save_checkpoint(
                    output_dir=run_dir,
                    name="checkpoint_last.pt",
                    step=global_step,
                    epoch=epoch,
                    metric=current_loss,
                    model=model,
                    optimizer=optimizer,
                )
                if current_loss < best_metric:
                    best_metric = current_loss
                    save_checkpoint(
                        output_dir=run_dir,
                        name="checkpoint_best.pt",
                        step=global_step,
                        epoch=epoch,
                        metric=current_loss,
                        model=model,
                        optimizer=optimizer,
                    )

        epoch_duration = perf_counter() - epoch_start
        logger.info("Epoch %d finished in %.2f s (global_step=%d)", epoch, epoch_duration, global_step)

    write_loss_csv(loss_history, metrics_dir / "loss.csv")
    plot_loss_curves(loss_history, metrics_dir)
    final_epoch = cfg.epochs
    save_checkpoint(
        output_dir=run_dir,
        name="checkpoint_last.pt",
        step=global_step,
        epoch=final_epoch,
        metric=last_loss,
        model=model,
        optimizer=optimizer,
    )
    if last_loss < best_metric:
        best_metric = last_loss
        save_checkpoint(
            output_dir=run_dir,
            name="checkpoint_best.pt",
            step=global_step,
            epoch=final_epoch,
            metric=last_loss,
            model=model,
            optimizer=optimizer,
        )
    save_checkpoint(
        output_dir=run_dir,
        name="checkpoint_final.pt",
        step=global_step,
        epoch=final_epoch,
        metric=last_loss,
        model=model,
        optimizer=optimizer,
    )


def main() -> None:
    cfg = tyro.cli(TrainConfig)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    main()
