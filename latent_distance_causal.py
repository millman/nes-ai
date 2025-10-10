#!/usr/bin/env python3
"""
Causal-visual distance learning for NES Mario trajectories.

This script follows the same data handling and evaluation structure as
`latent_distance.py`, but augments the training objective with a
forward-dynamics model that serves as a proxy for "causal" influence.

Key components:
  * TinyConvEncoder that produces latent embeddings and spatial feature maps.
  * ForwardModel that predicts the next embedding; its gradients define
    what visual regions the dynamics depend on.
  * SaliencyHead trained to approximate those gradient-based influence maps.
  * Causal distance computed by weighting feature differences with the
    learned saliency, so purely cosmetic changes (e.g., HUD flashes) are
    down-weighted while dynamics-relevant changes stay prominent.

The resulting calibrated distance aims to answer: "How different are these
frames in terms of how they influence the future?"
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import os
from pathlib import Path
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from latent_distance.shared import (
    FramesDataset,
    TrajectoryIndex,
    attach_run_output_dir,
    format_elapsed,
    pick_device,
    set_seed,
)
from latent_distance.viz_utils import overlay_heatmap, save_image_grid, plot_causal_distance


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------


class CausalPairDataset(Dataset):
    """Samples anchor pairs for causal distance learning.

    For each item we return:
        x_t: anchor frame at time t
        x_t1: next frame (t+1) for dynamics supervision
        x_j: comparison frame within the same trajectory
        delta_t: |j - t|
        traj_idx, i, j for bookkeeping
    """

    def __init__(
        self,
        index: TrajectoryIndex,
        image_size: int = 128,
        B: int = 16,
        seed: int = 0,
        p_short: float = 0.4,
        p_medium: float = 0.4,
    ):
        self.index = index
        self.image_size = image_size
        self.B = B
        self.rng = random.Random(seed)
        self.p_short = p_short
        self.p_medium = p_medium
        self.p_long = max(0.0, 1.0 - p_short - p_medium)

        self.tr_img = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return sum(len(f) for f in self.index.frames)

    def _load(self, traj_idx: int, t: int):
        p = self.index.get_path(traj_idx, t)
        img = Image.open(p).convert('RGB').resize((self.image_size, self.image_size), Image.BILINEAR)
        return self.tr_img(img), str(p)

    def __getitem__(self, _):
        traj_idx = self.rng.randrange(self.index.num_traj())
        L = self.index.len_traj(traj_idx)
        assert L >= 3

        i = self.rng.randrange(0, L - 1)  # ensure t+1 exists
        x_t, p_i = self._load(traj_idx, i)
        x_t1, p_ip1 = self._load(traj_idx, min(i + 1, L - 1))

        bucket = self.rng.random()
        if bucket < self.p_short:
            dt_max = max(1, min(3, self.B))
            delta_t = self.rng.randint(1, dt_max)
        elif bucket < self.p_short + self.p_medium:
            delta_t = self.rng.randint(max(4, 1), max(self.B, 4))
        else:
            max_forward = max(1, L - 1 - i)
            far_lo = max(self.B + 1, 2)
            far_hi = min(max_forward, 150)
            if far_hi >= far_lo:
                delta_t = self.rng.randint(far_lo, far_hi)
            else:
                delta_t = self.rng.randint(1, max_forward)

        j = min(L - 1, i + delta_t)
        x_j, p_j = self._load(traj_idx, j)

        sample = {
            'x_t': x_t,
            'x_t1': x_t1,
            'x_j': x_j,
            'traj_idx': traj_idx,
            'i': i,
            'j': j,
            'delta_t': torch.tensor(delta_t, dtype=torch.float32),
            'path_t': p_i,
            'path_t1': p_ip1,
            'path_j': p_j,
        }
        return sample


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class TinyConvEncoder(nn.Module):
    def __init__(self, in_ch=3, width=32, feat_dim=256):
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
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(width * 4, feat_dim)
        self.out_channels = width * 4

    def forward(self, x):
        z = self.act(self.bn1(self.conv1(x)))
        z = self.act(self.bn2(self.conv2(z)))
        z = self.act(self.bn3(self.conv3(z)))
        z = self.act(self.bn4(self.conv4(z)))
        pooled = self.pool(z).flatten(1)
        h = self.proj(pooled)
        return h, z  # embedding, feature map


class ForwardModel(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.SiLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, h):
        return self.net(h)


class SaliencyHead(nn.Module):
    """Predicts per-location influence mask from encoder feature maps."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1), nn.SiLU(),
            nn.Conv2d(in_ch // 2, 1, 1),
        )

    def forward(self, feat):
        mask = self.net(feat)
        return torch.sigmoid(mask)


class HeadCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.offset = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        scale = torch.exp(self.log_scale)
        return scale * x + self.offset


class CausalDistanceModel(nn.Module):
    def __init__(self, in_ch=3, feat_dim=256):
        super().__init__()
        self.enc = TinyConvEncoder(in_ch=in_ch, feat_dim=feat_dim)
        self.fwd = ForwardModel(feat_dim=feat_dim)
        self.sal_head = SaliencyHead(in_ch=self.enc.out_channels)
        self.calibrator = HeadCalibrator()

    def encode(self, x):
        return self.enc(x)

    def forward_model(self, h):
        return self.fwd(h)

    def saliency(self, feat):
        return self.sal_head(feat)

    def causal_distance(self, feat_a, feat_b, saliency):
        # saliency: [B,1,H,W], feat: [B,C,H,W]
        diff = torch.abs(feat_a - feat_b)
        weighted = saliency * diff
        dist = weighted.flatten(1).sum(dim=1) / (saliency.flatten(1).sum(dim=1) + 1e-6)
        return self.calibrator(dist)


# -----------------------------------------------------------------------------
# Loss helpers
# -----------------------------------------------------------------------------


def temporal_ranking_loss(dists: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    B = dists.shape[0]
    if B < 3:
        return torch.tensor(0.0, device=dists.device)
    idx = torch.randperm(B, device=dists.device)
    a = idx[: B // 2]
    b = idx[B // 2 : B]
    mask = deltas[a] < deltas[b]
    if mask.sum() == 0:
        return torch.tensor(0.0, device=dists.device)
    return F.softplus(dists[a[mask]] - dists[b[mask]]).mean()


# -----------------------------------------------------------------------------
# Evaluation helpers (adapted from latent_distance.py)
# -----------------------------------------------------------------------------


def _tensor_to_np_image(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()


def _tensor_to_np_heatmap(m: torch.Tensor, target_hw: torch.Size) -> np.ndarray:
    if m.ndim == 3:
        m = m.unsqueeze(0)
    heat = F.interpolate(m, size=target_hw[-2:], mode='bilinear', align_corners=False)
    return heat.detach().cpu().squeeze(0).squeeze(0).numpy()


def run_evals(cfg, model: CausalDistanceModel, frame_ld: DataLoader, index: TrajectoryIndex, tag: str):
    model.eval()
    out_dir = Path(cfg.out_dir) / f"eval_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_h = []
    all_x = []
    all_meta = []
    with torch.no_grad():
        for batch in frame_ld:
            x = batch['x'].to(cfg.device)
            h, feat = model.encode(x)
            all_h.append(h.cpu())
            all_x.append(x.cpu())
            for ti, tt, pp in zip(batch['traj_idx'], batch['t'], batch['path']):
                all_meta.append((int(ti), int(tt), pp))

    H = torch.cat(all_h, dim=0).cpu().numpy()
    X = torch.cat(all_x, dim=0)

    traj_to_indices: Dict[int, List[int]] = {}
    for idx_global, (ti, tt, _) in enumerate(all_meta):
        traj_to_indices.setdefault(ti, []).append(idx_global)
    for ti, lst in traj_to_indices.items():
        lst.sort(key=lambda gidx: all_meta[gidx][1])
    traj_names = {
        ti: (index.traj_paths[ti].name if ti < len(index.traj_paths) else str(ti))
        for ti in traj_to_indices.keys()
    }

    num_traj_plot = min(4, len(traj_to_indices))
    chosen_traj = sorted(traj_to_indices.keys())[:num_traj_plot]
    for ti in chosen_traj:
        idxs = traj_to_indices[ti]
        if len(idxs) < 2:
            continue
        x0_idx = idxs[0]
        x0 = X[x0_idx:x0_idx + 1].to(cfg.device)
        xs = X[idxs].to(cfg.device)
        with torch.no_grad():
            _, feat0 = model.encode(x0)
            _, feat_s = model.encode(xs)
            sal0 = model.saliency(feat0)
            causal = model.causal_distance(
                feat0.repeat(feat_s.shape[0], 1, 1, 1),
                feat_s,
                sal0.repeat(feat_s.shape[0], 1, 1, 1)
            )
        t_steps = np.arange(len(idxs))
        plot_causal_distance(
            t_steps,
            causal.detach().cpu().numpy(),
            title=f'Traj {traj_names[ti]}: causal self-distance',
            out_path=str(out_dir / f"traj{ti:03d}_causal_distance.png"),
        )

    M = 6
    heat_imgs: List[np.ndarray] = []
    heat_titles: List[str] = []
    rng = random.Random(0)
    keys = list(traj_to_indices.keys())
    for _ in range(M):
        ti = rng.choice(keys)
        idxs = traj_to_indices[ti]
        if len(idxs) < 3:
            continue
        i_local = rng.randrange(0, len(idxs) - 1)
        j_local = rng.randrange(i_local + 1, len(idxs))
        gi = idxs[i_local]
        gj = idxs[j_local]
        xi = X[gi:gi + 1].to(cfg.device)
        xj = X[gj:gj + 1].to(cfg.device)
        with torch.no_grad():
            _, feat_i = model.encode(xi)
            _, feat_j = model.encode(xj)
            sal = model.saliency(feat_i)
        xi_np = _tensor_to_np_image(xi[0])
        heat_imgs.extend([
            xi_np,
            _tensor_to_np_image(xj[0]),
            overlay_heatmap(xi_np, _tensor_to_np_heatmap(sal, xi.shape))
        ])
        heat_titles.extend([
            f"{traj_names[ti]} t={all_meta[gi][1]} (from)",
            f"{traj_names[ti]} t={all_meta[gj][1]} (to)",
            "saliency"
        ])
    if heat_imgs:
        save_image_grid(heat_imgs, heat_titles, str(out_dir / "saliency_examples.png"), ncol=3)

    return len(all_meta)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


@dataclass
class Args:
    data_root: str = "data.image_distance.train_10_traj"
    out_dir: str = "out.latent_distance_causal"
    seed: int = 0
    image_size: int = 128
    batch_size: int = 64
    epochs: int = 1000
    lr: float = 1e-3
    device: Optional[str] = None
    B: int = 16  # positive window for NCE

    # loss weights
    lambda_dyn: float = 1.0
    lambda_sal: float = 0.1
    lambda_rank: float = 0.2
    lambda_dist: float = 1.0

    # logging/viz
    eval_every: int = 5


def build_loaders(cfg: Args):
    index = TrajectoryIndex(Path(cfg.data_root), min_len=3)
    pair_ds = CausalPairDataset(index, image_size=cfg.image_size, B=cfg.B, seed=cfg.seed)
    pair_ld = DataLoader(pair_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=True)

    frame_ds = FramesDataset(index, image_size=cfg.image_size)
    frame_ld = DataLoader(frame_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    return index, pair_ld, frame_ld


def train(cfg: Args):
    set_seed(cfg.seed)
    cfg.device = pick_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    index, pair_ld, frame_ld = build_loaders(cfg)
    model = CausalDistanceModel(in_ch=3, feat_dim=256).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    run_start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        meter = {
            'dyn': 0.0,
            'sal': 0.0,
            'dist': 0.0,
            'rank': 0.0,
            'loss': 0.0,
        }
        n_batches = 0
        t0 = time.time()

        for batch in pair_ld:
            n_batches += 1
            x_t = batch['x_t'].to(cfg.device)
            x_t1 = batch['x_t1'].to(cfg.device)
            x_j = batch['x_j'].to(cfg.device)
            delta_t = batch['delta_t'].to(cfg.device)

            opt.zero_grad(set_to_none=True)

            h_t, feat_t = model.encode(x_t)
            h_t1, _ = model.encode(x_t1)
            h_j, feat_j = model.encode(x_j)

            h_pred = model.forward_model(h_t)
            loss_dyn = F.mse_loss(h_pred, h_t1)

            grads = torch.autograd.grad(loss_dyn, feat_t, retain_graph=True, create_graph=False)[0].detach()
            grad_target = grads.pow(2).sum(dim=1, keepdim=True).sqrt()
            grad_target = grad_target / (grad_target.amax(dim=(2, 3), keepdim=True) + 1e-6)
            saliency = model.saliency(feat_t)
            loss_sal = F.mse_loss(saliency, grad_target)

            causal_dist = model.causal_distance(feat_t, feat_j, saliency)
            loss_dist = F.mse_loss(causal_dist, delta_t)
            loss_rank = temporal_ranking_loss(causal_dist, delta_t)

            loss = (
                cfg.lambda_dyn * loss_dyn +
                cfg.lambda_sal * loss_sal +
                cfg.lambda_dist * loss_dist +
                cfg.lambda_rank * loss_rank
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            meter['dyn'] += float(loss_dyn.detach().cpu())
            meter['sal'] += float(loss_sal.detach().cpu())
            meter['dist'] += float(loss_dist.detach().cpu())
            meter['rank'] += float(loss_rank.detach().cpu())
            meter['loss'] += float(loss.detach().cpu())

        denom = max(1, n_batches)
        train_elapsed = time.time() - t0
        samples = n_batches * cfg.batch_size
        prefix = f"[{format_elapsed(run_start_time)}, ep {epoch:03d}]"
        msg = (
            f"{prefix} loss {meter['loss']/denom:.4f} | "
            f"dyn {meter['dyn']/denom:.3f} sal {meter['sal']/denom:.3f} "
            f"dist {meter['dist']/denom:.3f} rank {meter['rank']/denom:.3f} | "
            f"train_time {train_elapsed:.2f}s ({samples/max(train_elapsed,1e-6):.1f} samp/s)"
        )
        print(msg, flush=True)

        ckpt = {
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'cfg': dataclasses.asdict(cfg),
            'epoch': epoch,
        }
        torch.save(ckpt, os.path.join(cfg.out_dir, "last.pt"))

        if epoch % cfg.eval_every == 0:
            with torch.no_grad():
                eval_t0 = time.time()
                n_eval = run_evals(cfg, model, frame_ld, index, tag=f"ep{epoch:03d}")
                eval_elapsed = time.time() - eval_t0
            eval_msg = (
                f"{prefix} eval_time {eval_elapsed:.2f}s "
                f"({n_eval/max(eval_elapsed,1e-6):.1f} samp/s)"
            )
            print(eval_msg, flush=True)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    cfg = attach_run_output_dir(tyro.cli(Args))
    cfg.device = pick_device(cfg.device)
    print(f"[Device] {cfg.device} | [Data] {cfg.data_root} | [Out] {cfg.out_dir}", flush=True)
    train(cfg)


if __name__ == '__main__':
    main()
