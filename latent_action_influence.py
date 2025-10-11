#!/usr/bin/env python3
"""
Latent action-influence discovery for NES Mario trajectories.

This script mirrors the training/eval pattern used by `latent_distance_causal.py`.
It learns a state encoder and a latent "action" variable directly from image
sequences (no ground-truth actions). The model decomposes the next-frame
prediction into (a) what can already be explained by the current visual state
and (b) what requires an inferred action impulse. We supervise two saliency
heads so they highlight, respectively, state-driven evidence and regions most
altered by the latent action.

Outputs include:
  * latent action embeddings and their sparsity statistics
  * per-frame heatmaps for action-driven vs. state-driven influence
  * diagnostic loss plots saved alongside checkpoints

Usage example (mirrors other scripts):
    python latent_action_influence.py --data-root data.image_distance.train_10_traj \
        --out-dir out.latent_action_influence --epochs 50
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
    """Samples 3-frame windows (t-1, t, t+1) for latent action discovery."""

    def __init__(
        self,
        index: TrajectoryIndex,
        image_size: int = 128,
        seed: int = 0,
    ):
        self.index = index
        self.image_size = image_size
        self.rng = random.Random(seed)
        self.tr = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        # define a virtual epoch length proportional to total available windows
        total = 0
        for frames in self.index.frames:
            total += max(0, len(frames) - 2)
        return max(total, 1)

    def _load(self, traj_idx: int, t: int) -> Tuple[torch.Tensor, str]:
        path = self.index.get_path(traj_idx, t)
        img = Image.open(path).convert('RGB').resize((self.image_size, self.image_size), Image.BILINEAR)
        return self.tr(img), str(path)

    def __getitem__(self, _: int) -> Dict[str, object]:
        traj_idx = self.rng.randrange(self.index.num_traj())
        L = self.index.len_traj(traj_idx)
        if L < 3:
            raise RuntimeError("Trajectory too short for 3-frame window")
        center = self.rng.randrange(1, L - 1)
        x_prev, path_prev = self._load(traj_idx, center - 1)
        x_curr, path_curr = self._load(traj_idx, center)
        x_next, path_next = self._load(traj_idx, center + 1)
        return {
            'x_prev': x_prev,
            'x_curr': x_curr,
            'x_next': x_next,
            'traj_idx': traj_idx,
            't_prev': center - 1,
            't_curr': center,
            't_next': center + 1,
            'path_prev': path_prev,
            'path_curr': path_curr,
            'path_next': path_next,
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
        # state transition takes current feat and velocity-like difference
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

    def transition(self, feat_prev: torch.Tensor, feat_curr: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
# Evaluation helpers
# -----------------------------------------------------------------------------


def _tensor_to_np_image(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()


def _tensor_to_np_heatmap(m: torch.Tensor, target_hw: torch.Size) -> np.ndarray:
    if m.ndim == 3:
        m = m.unsqueeze(0)
    heat = F.interpolate(m, size=target_hw[-2:], mode='bilinear', align_corners=False)
    return heat.detach().cpu().squeeze(0).squeeze(0).numpy()


def collect_eval_windows(index: TrajectoryIndex) -> List[Tuple[int, int]]:
    windows = []
    for traj_idx in range(index.num_traj()):
        L = index.len_traj(traj_idx)
        for center in range(1, L - 1):
            windows.append((traj_idx, center))
    return windows


def load_window(index: TrajectoryIndex, traj_idx: int, center: int, image_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    paths = []
    tr = transforms.Compose([transforms.ToTensor()])
    frames = []
    for offset in (-1, 0, 1):
        path = index.get_path(traj_idx, center + offset)
        img = Image.open(path).convert('RGB').resize((image_size, image_size), Image.BILINEAR)
        frames.append(tr(img))
        paths.append(str(path))
    return frames[0], frames[1], frames[2], paths


def run_evals(cfg, model: ActionInfluenceModel, index: TrajectoryIndex, tag: str) -> Dict[str, float]:
    model.eval()
    out_dir = Path(cfg.out_dir) / f"eval_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(cfg.seed + 1337)
    windows = collect_eval_windows(index)
    rng.shuffle(windows)
    windows = windows[: cfg.eval_samples]

    entries = []
    all_action_mag = []
    with torch.no_grad():
        for traj_idx, center in windows:
            x_prev, x_curr, x_next, paths = load_window(index, traj_idx, center, cfg.image_size)
            xb_prev = x_prev.unsqueeze(0).to(cfg.device)
            xb_curr = x_curr.unsqueeze(0).to(cfg.device)
            xb_next = x_next.unsqueeze(0).to(cfg.device)

            _, feat_prev = model.encode(xb_prev)
            _, feat_curr = model.encode(xb_curr)
            _, feat_next = model.encode(xb_next)

            state_prev = model.state_vector(feat_prev)
            state_curr = model.state_vector(feat_curr)
            state_next = model.state_vector(feat_next)

            mu_prior, logvar_prior = model.compute_prior(state_prev, state_curr)
            mu_post, _ = model.compute_posterior(state_prev, state_curr, state_next)
            action = mu_post  # deterministic for viz
            base_feat, delta_feat, pred_feat = model.transition(feat_prev, feat_curr, action)

            action_mask = model.action_saliency(feat_curr, action)
            state_mask = model.state_saliency(feat_prev, feat_curr)

            delta_mag = delta_feat.abs().mean(dim=(1, 2, 3))
            all_action_mag.append(float(delta_mag.item()))

            heat_action = _tensor_to_np_heatmap(action_mask, x_curr.shape)
            heat_state = _tensor_to_np_heatmap(state_mask, x_curr.shape)
            img_curr = _tensor_to_np_image(x_curr)
            img_next = _tensor_to_np_image(x_next)

            overlay_action = overlay_heatmap(img_curr, heat_action)
            overlay_state = overlay_heatmap(img_curr, heat_state)

            entries.append({
                'traj': traj_idx,
                'center': center,
                'path_curr': paths[1],
                'path_next': paths[2],
                'img_curr': img_curr,
                'img_next': img_next,
                'overlay_action': overlay_action,
                'overlay_state': overlay_state,
                'action_mag': float(delta_mag.item()),
                'prior_std': float((0.5 * logvar_prior).exp().mean().item()),
            })

    if entries:
        grids = []
        titles = []
        for ent in entries:
            grids.extend([ent['img_curr'], ent['overlay_state'], ent['overlay_action'], ent['img_next']])
            titles.extend([
                f"traj{ent['traj']}@t{ent['center']}",
                "state influence",
                f"action infl (|Δ|={ent['action_mag']:.3f})",
                "next frame",
            ])
        save_image_grid(grids, titles, str(out_dir / "windows.png"), ncol=4)

    stats = {
        'eval_samples': len(entries),
        'mean_action_mag': float(np.mean(all_action_mag)) if all_action_mag else 0.0,
        'median_action_mag': float(np.median(all_action_mag)) if all_action_mag else 0.0,
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
    batch_size: int = 16
    epochs: int = 1000
    lr: float = 1e-3
    seed: int = 0
    device: Optional[str] = None

    action_dim: int = 32

    lambda_recon: float = 1.0
    lambda_state_sal: float = 0.5
    lambda_action_sal: float = 0.5
    lambda_kl: float = 1e-3
    lambda_sparse: float = 1e-2

    eval_every: int = 2
    eval_samples: int = 12


def build_loaders(cfg: Args):
    index = TrajectoryIndex(Path(cfg.data_root), min_len=3)
    train_ds = ActionInfluenceDataset(index, image_size=cfg.image_size, seed=cfg.seed)
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=True)
    return index, train_ld


def train(cfg: Args):
    set_seed(cfg.seed)
    cfg.device = pick_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)

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
            x_prev = batch['x_prev'].to(cfg.device)
            x_curr = batch['x_curr'].to(cfg.device)
            x_next = batch['x_next'].to(cfg.device)

            opt.zero_grad(set_to_none=True)

            _, feat_prev = model.encode(x_prev)
            _, feat_curr = model.encode(x_curr)
            _, feat_next = model.encode(x_next)

            state_prev = model.state_vector(feat_prev)
            state_curr = model.state_vector(feat_curr)
            state_next = model.state_vector(feat_next)

            mu_prior, logvar_prior = model.compute_prior(state_prev, state_curr)
            mu_post, logvar_post = model.compute_posterior(state_prev, state_curr, state_next)
            action_latent, _ = model.sample_action(mu_post, logvar_post)

            base_feat, delta_feat, pred_feat = model.transition(feat_prev, feat_curr, action_latent)

            loss_recon = F.mse_loss(pred_feat, feat_next)
            loss_base = F.mse_loss(base_feat, feat_next)
            kl = _kl_divergence(mu_post, logvar_post, mu_prior, logvar_prior).mean()
            loss_sparse = delta_feat.abs().mean()

            state_grad = torch.autograd.grad(loss_base, feat_curr, retain_graph=True)[0].detach()
            state_target = state_grad.pow(2).sum(dim=1, keepdim=True).sqrt()
            state_target = state_target / (state_target.amax(dim=(2, 3), keepdim=True) + 1e-6)
            state_mask = model.state_saliency(feat_prev, feat_curr)
            loss_state_sal = F.mse_loss(state_mask, state_target)

            action_target = delta_feat.pow(2).sum(dim=1, keepdim=True).sqrt()
            action_target = action_target / (action_target.amax(dim=(2, 3), keepdim=True) + 1e-6)
            action_mask = model.action_saliency(feat_curr, action_latent)
            loss_action_sal = F.mse_loss(action_mask, action_target)

            loss = (
                cfg.lambda_recon * loss_recon
                + cfg.lambda_state_sal * loss_state_sal
                + cfg.lambda_action_sal * loss_action_sal
                + cfg.lambda_kl * kl
                + cfg.lambda_sparse * loss_sparse
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            meter['total'] += float(loss.detach().cpu())
            meter['recon'] += float(loss_recon.detach().cpu())
            meter['state_sal'] += float(loss_state_sal.detach().cpu())
            meter['action_sal'] += float(loss_action_sal.detach().cpu())
            meter['kl'] += float(kl.detach().cpu())
            meter['sparse'] += float(loss_sparse.detach().cpu())

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
                f"median|Δ|={eval_stats['median_action_mag']:.4f}",
                flush=True,
            )


def main():
    cfg = attach_run_output_dir(tyro.cli(Args))
    train(cfg)


if __name__ == "__main__":
    main()
