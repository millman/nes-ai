#!/usr/bin/env python3
"""
Latent action-influence discovery for NES Mario trajectories with multi-frame
context and prediction.

The model ingests a configurable window of past frames and predicts multiple
future frames by decomposing dynamics into (a) a state-driven head that models
smooth evolution and (b) an action head that explains residual, intervention-
like changes. Two saliency heads are trained so they highlight, respectively,
state-driven evidence and regions most altered by the inferred latent action.
A counterfactual rollout using the latent prior's mean provides an explicit
"no-action" baseline for both losses and visualizations.

Example:
    python latent_action_influence.py \
        --data-root data.image_distance.train_10_traj \
        --out-dir out.latent_action_influence \
        --input-len 4 --pred-len 3
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import os
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from latent_distance.shared import (
    TrajectoryIndex,
    attach_run_output_dir,
    format_elapsed,
    pick_device,
    set_seed,
)
from latent_distance.viz_utils import overlay_heatmap, plot_loss_grid, save_image_grid


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class ActionInfluenceDataset(Dataset):
    """Samples contiguous windows for latent action discovery."""

    def __init__(
        self,
        index: TrajectoryIndex,
        image_size: int = 128,
        seed: int = 0,
        input_len: int = 4,
        pred_len: int = 3,
    ):
        if input_len < 2:
            raise ValueError("input_len must be >= 2 so velocity can be estimated")
        if pred_len < 1:
            raise ValueError("pred_len must be >= 1")

        self.index = index
        self.image_size = image_size
        self.input_len = input_len
        self.pred_len = pred_len
        self.total_len = input_len + pred_len
        self.rng = random.Random(seed)
        self.tr = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.windows: List[Tuple[int, int]] = []
        for traj_idx in range(index.num_traj()):
            L = index.len_traj(traj_idx)
            if L < self.total_len:
                continue
            for start in range(L - self.total_len + 1):
                self.windows.append((traj_idx, start))
        if not self.windows:
            raise RuntimeError(
                "No windows available; check data_root, input_len, and pred_len settings."
            )

    def __len__(self) -> int:
        return len(self.windows)

    def _load(self, traj_idx: int, t: int) -> torch.Tensor:
        path = self.index.get_path(traj_idx, t)
        img = Image.open(path).convert('RGB').resize((self.image_size, self.image_size), Image.BILINEAR)
        return self.tr(img)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx, start = self.windows[idx]
        frames = []
        for offset in range(self.total_len):
            frames.append(self._load(traj_idx, start + offset))
        stack = torch.stack(frames, dim=0)  # [total_len, 3, H, W]
        context = stack[: self.input_len]
        future = stack[self.input_len :]
        return {
            'context': context,
            'future': future,
            'traj_idx': traj_idx,
            'start': start,
        }


# -----------------------------------------------------------------------------
# Model components
# -----------------------------------------------------------------------------


class TinyConvEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, width: int = 32, feat_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, width, 5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width * 2, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(width * 2)
        self.conv3 = nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(width * 4)
        self.conv4 = nn.Conv2d(width * 4, width * 4, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(width * 4)
        self.act = nn.SiLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(width * 4, feat_dim)
        self.out_channels = width * 4

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.act(self.bn1(self.conv1(x)))
        z = self.act(self.bn2(self.conv2(z)))
        z = self.act(self.bn3(self.conv3(z)))
        z = self.act(self.bn4(self.conv4(z)))
        pooled = self.pool(z).flatten(1)
        h = self.proj(pooled)
        return h, z


class ConvTransition(nn.Module):
    """Predicts next feature map from current state (no action)."""

    def __init__(self, in_ch: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionDecoder(nn.Module):
    """Injects latent action into spatial feature map."""

    def __init__(self, feat_ch: int, action_dim: int):
        super().__init__()
        self.fc = nn.Linear(action_dim, feat_ch)
        self.net = nn.Sequential(
            nn.Conv2d(feat_ch * 2, feat_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1),
        )

    def forward(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        action_proj = self.fc(action).unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)
        x = torch.cat([feat, action_proj], dim=1)
        return self.net(x)


class SaliencyHead(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_ch // 2, 1, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(feat))


class ActionSaliencyHead(nn.Module):
    """Action-conditioned saliency predictor."""

    def __init__(self, feat_ch: int, action_dim: int):
        super().__init__()
        self.action_fc = nn.Linear(action_dim, feat_ch)
        self.net = nn.Sequential(
            nn.Conv2d(feat_ch * 2, feat_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(feat_ch, 1, 1),
        )

    def forward(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        action_proj = self.action_fc(action).unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)
        fused = torch.cat([feat, action_proj], dim=1)
        return torch.sigmoid(self.net(fused))


def _kl_divergence(mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    """KL divergence between two diagonal-cov Gaussians (per-sample)."""
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    term = (var_q + (mu_q - mu_p).pow(2)) / (var_p + 1e-8)
    kld = 0.5 * (logvar_p - logvar_q + term - 1.0)
    return kld.sum(dim=1)


class ActionInfluenceModel(nn.Module):
    def __init__(self, in_ch: int = 3, feat_dim: int = 256, action_dim: int = 32):
        super().__init__()
        self.enc = TinyConvEncoder(in_ch=in_ch, feat_dim=feat_dim)
        state_dim = 256
        feat_ch = self.enc.out_channels
        self.state_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_ch, state_dim),
            nn.SiLU(),
        )
        self.prior_net = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, action_dim * 2),
        )
        self.post_net = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, action_dim * 2),
        )
        self.state_transition = ConvTransition(in_ch=feat_ch * 2, hidden=feat_ch)
        self.action_decoder = ActionDecoder(feat_ch=feat_ch, action_dim=action_dim)
        self.state_saliency_head = SaliencyHead(in_ch=feat_ch * 2)
        self.action_saliency_head = ActionSaliencyHead(feat_ch=feat_ch, action_dim=action_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.enc(x)

    def state_vector(self, feat: torch.Tensor) -> torch.Tensor:
        return self.state_proj(feat)

    def compute_prior(self, state_prev: torch.Tensor, state_curr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([state_prev, state_curr], dim=1)
        out = self.prior_net(inp)
        mu, logvar = out.chunk(2, dim=1)
        return mu, logvar

    def compute_posterior(
        self,
        state_prev: torch.Tensor,
        state_curr: torch.Tensor,
        state_next: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([state_prev, state_curr, state_next], dim=1)
        out = self.post_net(inp)
        mu, logvar = out.chunk(2, dim=1)
        return mu, logvar

    def sample_action(self, mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def transition(
        self,
        feat_prev: torch.Tensor,
        feat_curr: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        velocity = feat_curr - feat_prev
        base_in = torch.cat([feat_curr, velocity], dim=1)
        base = self.state_transition(base_in)
        delta = self.action_decoder(feat_curr, action)
        pred = base + delta
        return base, delta, pred

    def action_saliency(self, feat_curr: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.action_saliency_head(feat_curr, action)

    def state_saliency(self, feat_prev: torch.Tensor, feat_curr: torch.Tensor) -> torch.Tensor:
        velocity = feat_curr - feat_prev
        return self.state_saliency_head(torch.cat([feat_curr, velocity], dim=1))


# -----------------------------------------------------------------------------
# Sequence helpers
# -----------------------------------------------------------------------------


def encode_sequence(
    model: ActionInfluenceModel,
    frames: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a [B, L, 3, H, W] tensor into feature maps and state vectors."""
    B, L, C, H, W = frames.shape
    flat = frames.view(B * L, C, H, W)
    _, feats = model.encode(flat)
    feat_ch, h_feat, w_feat = feats.shape[1:]
    feats = feats.view(B, L, feat_ch, h_feat, w_feat)
    states = model.state_vector(feats.view(B * L, feat_ch, h_feat, w_feat))
    state_dim = states.shape[1]
    states = states.view(B, L, state_dim)
    return feats, states


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------


def _tensor_to_np_image(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()


def _tensor_to_np_heatmap(m: torch.Tensor, target_hw: torch.Size) -> np.ndarray:
    if m.ndim == 3:
        m = m.unsqueeze(0)
    heat = F.interpolate(m, size=target_hw[-2:], mode='bilinear', align_corners=False)
    return heat.detach().cpu().squeeze(0).squeeze(0).numpy()


def collect_eval_windows(
    index: TrajectoryIndex,
    input_len: int,
    pred_len: int,
) -> List[Tuple[int, int]]:
    windows = []
    total = input_len + pred_len
    for traj_idx in range(index.num_traj()):
        L = index.len_traj(traj_idx)
        if L < total:
            continue
        for start in range(L - total + 1):
            windows.append((traj_idx, start))
    return windows


def load_window(
    index: TrajectoryIndex,
    traj_idx: int,
    start: int,
    input_len: int,
    pred_len: int,
    image_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    total = input_len + pred_len
    frames = []
    paths = []
    tr = transforms.Compose([transforms.ToTensor()])
    for offset in range(total):
        path = index.get_path(traj_idx, start + offset)
        img = Image.open(path).convert('RGB').resize((image_size, image_size), Image.BILINEAR)
        frames.append(tr(img))
        paths.append(str(path))
    stack = torch.stack(frames, dim=0)
    context = stack[:input_len]
    future = stack[input_len:]
    return context, future, paths[:input_len], paths[input_len:]


def run_evals(
    cfg,
    model: ActionInfluenceModel,
    index: TrajectoryIndex,
    tag: str,
) -> Dict[str, float]:
    model.eval()
    out_dir = Path(cfg.out_dir) / f"eval_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = collect_eval_windows(index, cfg.input_len, cfg.pred_len)
    if cfg.eval_samples > 0:
        windows = windows[: cfg.eval_samples]
    vis_steps = min(cfg.pred_len, cfg.vis_steps)

    entries = []
    all_action_mag: List[float] = []
    all_effect_mag: List[float] = []

    for traj_idx, start in windows:
        context, future, ctx_paths, fut_paths = load_window(
            index, traj_idx, start, cfg.input_len, cfg.pred_len, cfg.image_size
        )
        frames = torch.cat([context, future], dim=0).unsqueeze(0).to(cfg.device)
        with torch.no_grad():
            feat_seq, state_seq = encode_sequence(model, frames)

        feat_seq = feat_seq.squeeze(0)
        state_seq = state_seq.squeeze(0)
        total_len = cfg.input_len + cfg.pred_len

        state_overlays: List[np.ndarray] = []
        action_overlays: List[np.ndarray] = []
        effect_overlays: List[np.ndarray] = []
        target_imgs: List[np.ndarray] = []
        ctx_imgs: List[np.ndarray] = []

        for step in range(cfg.pred_len):
            idx_target = cfg.input_len + step
            idx_curr = idx_target - 1
            idx_prev = idx_curr - 1

            feat_prev = feat_seq[idx_prev: idx_prev + 1]
            feat_curr = feat_seq[idx_curr: idx_curr + 1]
            feat_target = feat_seq[idx_target: idx_target + 1]

            state_prev = state_seq[idx_prev: idx_prev + 1]
            state_curr = state_seq[idx_curr: idx_curr + 1]
            state_next = state_seq[idx_target: idx_target + 1]

            mu_prior, logvar_prior = model.compute_prior(state_prev, state_curr)
            mu_post, _ = model.compute_posterior(state_prev, state_curr, state_next)
            action = mu_post
            base_feat, delta_feat, pred_feat = model.transition(feat_prev, feat_curr, action)
            residual_target = (feat_target - base_feat).abs()

            delta_mag = delta_feat.abs().mean().item()
            effect_mag = residual_target.mean().item()
            all_action_mag.append(delta_mag)
            all_effect_mag.append(effect_mag)

            if step < vis_steps:
                state_mask = model.state_saliency(feat_prev, feat_curr)
                action_mask = model.action_saliency(feat_curr, action)

                effect_map = residual_target.pow(2).sum(dim=1, keepdim=True).sqrt()
                denom = effect_map.amax(dim=(2, 3), keepdim=True) + 1e-6
                effect_map = effect_map / denom

                heat_state = _tensor_to_np_heatmap(state_mask, context.shape)
                heat_action = _tensor_to_np_heatmap(action_mask, context.shape)
                heat_effect = _tensor_to_np_heatmap(effect_map, context.shape)

                base_state_img = _tensor_to_np_image(context[-1])
                future_img = _tensor_to_np_image(future[step])
                state_overlays.append(overlay_heatmap(base_state_img, heat_state))
                action_overlays.append(overlay_heatmap(future_img, heat_action))
                effect_overlays.append(overlay_heatmap(future_img, heat_effect))
                target_imgs.append(future_img)

        for step in range(cfg.input_len):
            ctx_imgs.append(_tensor_to_np_image(context[step]))

        if state_overlays:
            total_cols = cfg.input_len + vis_steps
            grids: List[np.ndarray] = []
            titles: List[str] = []
            ylabels = ["groundtruth", "prediction", "state", "action", "cf"]

            # Column labels for time (used only on the top row)
            col_labels = []
            for idx in range(cfg.input_len):
                offset = cfg.input_len - idx - 1
                col_labels.append(f"t-{offset}" if offset > 0 else "t")
            for step in range(vis_steps):
                col_labels.append(f"t+{step+1}")

            blank = np.zeros_like(ctx_imgs[0]) if ctx_imgs else None

            row_contents = []

            # Groundtruth row
            row = []
            for col in range(total_cols):
                if col < len(ctx_imgs):
                    row.append(ctx_imgs[col])
                else:
                    row.append(target_imgs[col - len(ctx_imgs)])
            row_contents.append(row)

            # Prediction row (placeholder)
            row = []
            for col in range(total_cols):
                if col < len(ctx_imgs):
                    row.append(blank if blank is not None else ctx_imgs[col])
                else:
                    row.append(target_imgs[col - len(ctx_imgs)])
            row_contents.append(row)

            # State row
            row = []
            for col in range(total_cols):
                if col < len(ctx_imgs):
                    row.append(ctx_imgs[col])
                else:
                    idx_local = col - len(ctx_imgs)
                    row.append(state_overlays[idx_local])
            row_contents.append(row)

            # Action row
            row = []
            for col in range(total_cols):
                if col < len(ctx_imgs):
                    row.append(ctx_imgs[col])
                else:
                    idx_local = col - len(ctx_imgs)
                    row.append(action_overlays[idx_local])
            row_contents.append(row)

            # CF row
            row = []
            for col in range(total_cols):
                if col < len(ctx_imgs):
                    row.append(ctx_imgs[col])
                else:
                    idx_local = col - len(ctx_imgs)
                    row.append(effect_overlays[idx_local])
            row_contents.append(row)

            for row_idx, row in enumerate(row_contents):
                for col_idx, img in enumerate(row):
                    grids.append(img)
                    if row_idx == 0:
                        titles.append(col_labels[col_idx])
                    else:
                        titles.append("")

            chunk_name = f"traj_{traj_idx}_state_{start}_to_{start + cfg.input_len + cfg.pred_len - 1}"
            traj_dir = out_dir / f"traj_{traj_idx}"
            traj_dir.mkdir(parents=True, exist_ok=True)
            save_image_grid(
                grids,
                titles,
                str(traj_dir / f"{chunk_name}.png"),
                ncol=total_cols,
                ylabels=ylabels,
            )

        entries.append(1)

    stats = {
        'eval_samples': len(entries),
        'mean_action_mag': float(np.mean(all_action_mag)) if all_action_mag else 0.0,
        'median_action_mag': float(np.median(all_action_mag)) if all_action_mag else 0.0,
        'mean_effect_mag': float(np.mean(all_effect_mag)) if all_effect_mag else 0.0,
    }
    return stats


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


@dataclass
class Args:
    data_root: str = "data.image_distance.train_10_traj"
    out_dir: str = "out.latent_action_influence"
    image_size: int = 128
    batch_size: int = 8
    epochs: int = 40
    lr: float = 1e-3
    seed: int = 0
    device: Optional[str] = None

    input_len: int = 4
    pred_len: int = 3
    action_dim: int = 32

    lambda_recon: float = 1.0
    lambda_state_sal: float = 0.5
    lambda_action_sal: float = 0.5
    lambda_kl: float = 1e-3
    lambda_sparse: float = 1e-2

    eval_every: int = 1
    eval_samples: int = 10
    vis_steps: int = 3


def build_loaders(cfg: Args):
    index = TrajectoryIndex(Path(cfg.data_root), min_len=cfg.input_len + cfg.pred_len)
    train_ds = ActionInfluenceDataset(
        index,
        image_size=cfg.image_size,
        seed=cfg.seed,
        input_len=cfg.input_len,
        pred_len=cfg.pred_len,
    )
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=True)
    return index, train_ld


def train(cfg: Args):
    index, train_ld = build_loaders(cfg)
    model = ActionInfluenceModel(in_ch=3, feat_dim=256, action_dim=cfg.action_dim).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    run_start = time.time()
    loss_hist: Dict[str, List[Tuple[int, float]]] = {
        'total': [],
        'recon': [],
        'state_sal': [],
        'action_sal': [],
        'kl': [],
        'sparse': [],
    }

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        meter = {k: 0.0 for k in loss_hist.keys()}
        n_batches = 0
        t0 = time.time()

        for batch in train_ld:
            n_batches += 1
            global_step += 1
            context = batch['context'].to(cfg.device)
            future = batch['future'].to(cfg.device)

            frames = torch.cat([context, future], dim=1)
            feat_seq, state_seq = encode_sequence(model, frames)

            recon_terms = []
            state_terms = []
            action_terms = []
            kl_terms = []
            sparse_terms = []

            for step in range(cfg.pred_len):
                idx_target = cfg.input_len + step
                idx_curr = idx_target - 1
                idx_prev = idx_curr - 1

                feat_prev = feat_seq[:, idx_prev]
                feat_curr = feat_seq[:, idx_curr]
                feat_target = feat_seq[:, idx_target]

                state_prev = state_seq[:, idx_prev]
                state_curr = state_seq[:, idx_curr]
                state_next = state_seq[:, idx_target]

                mu_prior, logvar_prior = model.compute_prior(state_prev, state_curr)
                mu_post, logvar_post = model.compute_posterior(state_prev, state_curr, state_next)
                action_latent, _ = model.sample_action(mu_post, logvar_post)

                base_feat, delta_feat, pred_feat = model.transition(feat_prev, feat_curr, action_latent)
                loss_base_fit = F.mse_loss(base_feat, feat_target)
                residual_target = (feat_target - base_feat).detach()
                loss_residual_fit = F.mse_loss(delta_feat, residual_target)

                recon_terms.append(loss_base_fit + loss_residual_fit)
                kl_terms.append(_kl_divergence(mu_post, logvar_post, mu_prior, logvar_prior).mean())
                sparse_terms.append(delta_feat.abs().mean())

                state_grad = torch.autograd.grad(
                    loss_base_fit, feat_curr, retain_graph=True, create_graph=False
                )[0].detach()
                state_target = state_grad.pow(2).sum(dim=1, keepdim=True).sqrt()
                state_target = state_target / (state_target.amax(dim=(2, 3), keepdim=True) + 1e-6)
                state_mask = model.state_saliency(feat_prev, feat_curr)
                state_terms.append(F.mse_loss(state_mask, state_target))

                action_target = residual_target.pow(2).sum(dim=1, keepdim=True).sqrt()
                action_target = action_target / (action_target.amax(dim=(2, 3), keepdim=True) + 1e-6)
                action_mask = model.action_saliency(feat_curr, action_latent)
                action_terms.append(F.mse_loss(action_mask, action_target))

            avg_recon = torch.stack(recon_terms).mean()
            avg_state = torch.stack(state_terms).mean()
            avg_action = torch.stack(action_terms).mean()
            avg_kl = torch.stack(kl_terms).mean()
            avg_sparse = torch.stack(sparse_terms).mean()

            loss = (
                cfg.lambda_recon * avg_recon
                + cfg.lambda_state_sal * avg_state
                + cfg.lambda_action_sal * avg_action
                + cfg.lambda_kl * avg_kl
                + cfg.lambda_sparse * avg_sparse
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            meter['total'] += float(loss.detach().cpu())
            meter['recon'] += float(avg_recon.detach().cpu())
            meter['state_sal'] += float(avg_state.detach().cpu())
            meter['action_sal'] += float(avg_action.detach().cpu())
            meter['kl'] += float(avg_kl.detach().cpu())
            meter['sparse'] += float(avg_sparse.detach().cpu())

        denom = max(1, n_batches)
        elapsed = time.time() - t0
        prefix = f"[{format_elapsed(run_start)} ep {epoch:03d}]"
        avg_total = meter['total'] / denom
        avg_recon = meter['recon'] / denom
        avg_state = meter['state_sal'] / denom
        avg_action = meter['action_sal'] / denom
        avg_kl = meter['kl'] / denom
        avg_sparse = meter['sparse'] / denom

        loss_hist['total'].append((epoch, avg_total))
        loss_hist['recon'].append((epoch, avg_recon))
        loss_hist['state_sal'].append((epoch, avg_state))
        loss_hist['action_sal'].append((epoch, avg_action))
        loss_hist['kl'].append((epoch, avg_kl))
        loss_hist['sparse'].append((epoch, avg_sparse))

        msg = (
            f"{prefix} loss {avg_total:.4f} | "
            f"recon {avg_recon:.3f} state_sal {avg_state:.3f} action_sal {avg_action:.3f} "
            f"kl {avg_kl:.4f} sparse {avg_sparse:.4f} | "
            f"train_time {elapsed:.2f}s"
        )
        print(msg, flush=True)

        ckpt = {
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'cfg': dataclasses.asdict(cfg),
            'epoch': epoch,
            'global_step': global_step,
        }
        torch.save(ckpt, os.path.join(cfg.out_dir, "last.pt"))

        plot_loss_grid(loss_hist, Path(cfg.out_dir) / "loss_plots", epoch)

        if epoch % cfg.eval_every == 0:
            eval_stats = run_evals(cfg, model, index, tag=f"ep{epoch:03d}")
            print(
                f"  eval[{epoch}] samples={eval_stats['eval_samples']} "
                f"mean|Δ|={eval_stats['mean_action_mag']:.4f} "
                f"median|Δ|={eval_stats['median_action_mag']:.4f} "
                f"mean|cf|={eval_stats['mean_effect_mag']:.4f}",
                flush=True,
            )


def main():
    cfg = attach_run_output_dir(tyro.cli(Args))

    set_seed(cfg.seed)
    cfg.device = pick_device(cfg.device)

    print(f"[Device] {cfg.device} | [Data] {cfg.data_root} | [Out] {cfg.out_dir}", flush=True)

    os.makedirs(cfg.out_dir, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    main()
