#!/usr/bin/env python3
"""
Latent Distance Learning (2-head: visual & non-visual)
Style: predict_mario_ms_ssim.py (single-script training + viz)

What this script does
---------------------
1) Loads short image-only trajectories (<=150 frames each)
   Expected layout (customize as needed):
       data_root/
         traj_000/
           states/*.png
           actions.npz   (ignored here; we are in the 'no actions' setting)
         traj_001/
           ...

2) Trains an encoder + two-head distance predictor that, given a pair (x_i, x_j),
   outputs d_vis, d_hid, and d_total = d_vis + d_hid.

3) Losses (all self-supervised):
   - Temporal Ranking (monotonic w.r.t. |Δt|)
   - Time Regression on d_hid to log(1+|Δt|)
   - Reachability NCE (positives within small step window, negatives outside / other trajs)
   - Triangle Inequality (soft)
   - Augmentation Invariance (identity pairs with jitter/noise/blur)
   - VICReg-style variance/cov regularization on embeddings

4) Evaluation / Visualization:
   (a) Head-specific heatmaps via Grad-CAM-like gradients onto the last conv map
   (b) Trajectory self-distance 2D plot: x-axis = true time/action-steps (t),
       y-axis = d_vis(x0, xt), color (viridis) = d_hid(x0, xt).
   (c) Cross-trajectory nearest/farthest sampling by cosine distance in latent h.
       For random anchors, show topK nearest/farthest frames from OTHER trajectories
       and report (d_vis, d_hid, d_total) for sanity.

NOTE
----
- This is a reference training skeleton—clean, modular, and hackable.
- It avoids external deps (e.g., faiss). Cosine search is numpy-based.
- Heatmaps are approximate (Grad-CAM-like) and work for conv backbones.
- Replace/upgrade encoder as you like; default is a tiny CNN.

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
import tyro
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from PIL import Image
from latent_distance.viz_utils import overlay_heatmap, plot_self_distance, save_image_grid


# ------------------------------
# Dataset
# ------------------------------

@dataclass
class PairSample:
    traj_idx: int
    i: int
    j: int
    delta_t: int
    # aug flags
    is_augident: bool = False


class PairDataset(Dataset):
    """Samples pairs within trajectories (Δt within a configurable range),
    plus augmentation-identity pairs.

    For NCE, the batch will contain positives and provide negatives implicitly.
    """
    def __init__(
        self,
        index: TrajectoryIndex,
        image_size: int = 128,
        B: int = 16,
        p_augident: float = 0.2,
        p_medium: float = 0.4,
        p_short: float = 0.4,
        p_cross_traj_negs: float = 0.0,  # not used for pairs; negatives are implicit in-batch
        seed: int = 0,
    ):
        self.index = index
        self.rng = random.Random(seed)
        self.image_size = image_size
        self.B = B
        self.p_augident = p_augident
        self.p_medium = p_medium
        self.p_short = p_short
        self.p_far = 1.0 - p_augident - p_medium - p_short
        assert self.p_far >= 0.0

        self.tr_img = transforms.Compose([
            transforms.ToTensor(),
        ])
        # augmentations for A0
        self.tr_aug = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        # define arbitrary large epoch length as total frames; each __getitem__ samples fresh pair
        return sum(len(f) for f in self.index.frames)

    def _load(self, traj_idx: int, t: int, aug: bool=False):
        p = self.index.get_path(traj_idx, t)
        img = Image.open(p).convert('RGB').resize((self.image_size, self.image_size), Image.BILINEAR)
        x = self.tr_aug(img) if aug else self.tr_img(img)
        return x, str(p)

    def __getitem__(self, _):
        # choose a trajectory
        traj_idx = self.rng.randrange(self.index.num_traj())
        L = self.index.len_traj(traj_idx)
        assert L >= 2

        r = self.rng.random()
        is_augident = False

        if r < self.p_augident:
            # augmentation identity pair
            i = self.rng.randrange(L)
            x_i, p_i = self._load(traj_idx, i, aug=False)
            x_ip, _ = self._load(traj_idx, i, aug=True)
            j = i
            delta_t = 0
            is_augident = True
            x_j = x_ip
            p_j = p_i + "|aug"
        else:
            # choose Δt bucket
            i = self.rng.randrange(L-1)
            bucket = self.rng.random()
            if bucket < self.p_short:
                # short: Δt in [1, min(3,B)]
                dt_max = max(1, min(3, self.B))
                delta_t = self.rng.randint(1, dt_max)
            elif bucket < self.p_short + self.p_medium:
                # medium: Δt in [4, B]
                delta_t = self.rng.randint(max(4, 1), max(self.B, 4))
            else:
                # far within trajectory (up to limit 150); fall back gracefully if window is small
                max_forward = max(1, L - 1 - i)
                far_lo = max(self.B + 1, 2)
                far_hi = min(max_forward, 150)
                if far_hi >= far_lo:
                    delta_t = self.rng.randint(far_lo, far_hi)
                else:
                    delta_t = self.rng.randint(1, max_forward)
            j = min(L-1, i + delta_t)
            delta_t = j - i
            x_i, p_i = self._load(traj_idx, i, aug=False)
            x_j, p_j = self._load(traj_idx, j, aug=False)

        sample = {
            'x_i': x_i, 'x_j': x_j,
            'traj_idx': traj_idx,
            'i': i, 'j': j,
            'delta_t': delta_t,
            'is_augident': is_augident,
            'path_i': p_i, 'path_j': p_j,
        }
        return sample


# ------------------------------
# Model
# ------------------------------

class TinyConvEncoder(nn.Module):
    def __init__(self, in_ch=3, width=32, feat_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, width, 5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width*2, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(width*2)
        self.conv3 = nn.Conv2d(width*2, width*4, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(width*4)
        self.conv4 = nn.Conv2d(width*4, width*4, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(width*4)
        self.act = nn.SiLU()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.proj = nn.Linear(width*4, feat_dim)
        self.feat_map: Optional[torch.Tensor] = None  # for CAM

    def forward(self, x):
        z = self.act(self.bn1(self.conv1(x)))
        z = self.act(self.bn2(self.conv2(z)))
        z = self.act(self.bn3(self.conv3(z)))
        z = self.act(self.bn4(self.conv4(z)))
        self.feat_map = z  # [B,C,H,W]
        pooled = self.pool(z).flatten(1)
        h = self.proj(pooled)
        return h  # [B, feat_dim]


class TwoHeadDistance(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        in_dim = feat_dim*4
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
        )
        self.out_vis = nn.Linear(128, 1)
        self.out_hid = nn.Linear(128, 1)

    def pair_features(self, h_i, h_j):
        return torch.cat([h_i, h_j, torch.abs(h_i - h_j), h_i * h_j], dim=-1)

    def forward(self, h_i, h_j):
        p = self.mlp(self.pair_features(h_i, h_j))
        d_vis = F.softplus(self.out_vis(p))  # [B,1]
        d_hid = F.softplus(self.out_hid(p))  # [B,1]
        d = d_vis + d_hid
        return d_vis.squeeze(-1), d_hid.squeeze(-1), d.squeeze(-1)


class HeadCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1))  # scale = 1
        self.offset = nn.Parameter(torch.zeros(1))     # offset = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.exp(self.log_scale)
        return scale * x + self.offset


class LatentDistanceModel(nn.Module):
    def __init__(self, in_ch=3, feat_dim=256):
        super().__init__()
        self.enc = TinyConvEncoder(in_ch=in_ch, feat_dim=feat_dim)
        self.head = TwoHeadDistance(feat_dim=feat_dim)
        self.calib_vis = HeadCalibrator()
        self.calib_hid = HeadCalibrator()

    def forward(self, x_i, x_j):
        h_i = self.enc(x_i)
        h_j = self.enc(x_j)
        d_vis_raw, d_hid_raw, d_raw = self.head(h_i, h_j)
        d_vis = self.calib_vis(d_vis_raw)
        d_hid = self.calib_hid(d_hid_raw)
        d = d_vis + d_hid
        return h_i, h_j, d_vis, d_hid, d

    # ---- Grad-CAM-ish heatmaps for each head wrt x_i ----
    def head_heatmaps(self, x_i, x_j, which_head: str = 'vis'):
        was_training = self.training
        self.eval()
        with torch.enable_grad():
            self.zero_grad(set_to_none=True)
            h_i = self.enc(x_i.detach())
            h_j = self.enc(x_j.detach())
            d_vis, d_hid, _ = self.head(h_i, h_j)
            score = d_vis if which_head == 'vis' else d_hid
            score_sum = score.sum()
            feat = self.enc.feat_map  # [B,C,H,W]
            assert feat is not None
            grads = torch.autograd.grad(score_sum, feat, retain_graph=False, create_graph=False)[0]
            feat = feat.detach()
        self.zero_grad(set_to_none=True)
        if was_training:
            self.train()
        weights = grads.mean(dim=(2,3), keepdim=True)  # GAP over H,W
        cam = (weights * feat).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x_i.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam / (cam.amax(dim=(2,3), keepdim=True) + 1e-6)
        return cam.detach()  # [B,1,H,W] in [0,1]


# ------------------------------
# Losses
# ------------------------------

def temporal_ranking_loss(dists: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Pairwise ranking within a batch: sample pairs of pairs s.t. delta1<delta2.
    dists: [B] scalar distance for each pair
    deltas: [B] integer Δt for each pair
    """
    B = dists.shape[0]
    if B < 3:
        return torch.tensor(0.0, device=dists.device)
    # quick-and-dirty: sample K pairs
    idx = torch.randperm(B, device=dists.device)
    half = B//2
    a = idx[:half]
    b = idx[half:half*2]
    mask = deltas[a] < deltas[b]
    if mask.sum() == 0:
        return torch.tensor(0.0, device=dists.device)
    loss = F.softplus(dists[a[mask]] - dists[b[mask]]).mean()
    return loss


def nce_loss(dists_pos: torch.Tensor, dists_neg: torch.Tensor) -> torch.Tensor:
    """InfoNCE with distances (smaller is better). Input:
    dists_pos: [B]
    dists_neg: [B, K]
    """
    # logits = -dist
    logits_pos = -dists_pos.unsqueeze(1)  # [B,1]
    logits_neg = -dists_neg  # [B,K]
    logits = torch.cat([logits_pos, logits_neg], dim=1)
    labels = torch.zeros(dists_pos.shape[0], dtype=torch.long, device=dists_pos.device)
    return F.cross_entropy(logits, labels)


def triangle_loss(d_ij: torch.Tensor, d_ik: torch.Tensor, d_kj: torch.Tensor) -> torch.Tensor:
    viol = torch.relu(d_ij - (d_ik + d_kj))
    return viol.mean()


def vicreg_loss(h: torch.Tensor, sim_coeff=25.0, var_coeff=25.0, cov_coeff=1.0) -> torch.Tensor:
    # Invariance not used (no pairs here), just variance+covariance on batch features
    h = h - h.mean(dim=0, keepdim=True)
    std = torch.sqrt(h.var(dim=0) + 1e-04)
    var_loss = torch.mean(F.relu(1.0 - std))
    h_norm = h / (std + 1e-6)
    cov = (h_norm.T @ h_norm) / (h_norm.shape[0] - 1)
    I = torch.eye(cov.shape[0], device=h.device)
    cov_loss = ((cov - I)**2).sum() / cov.shape[0]
    return var_coeff * var_loss + cov_coeff * cov_loss


# ------------------------------
# Training & Eval
# ------------------------------

@dataclass
class Args:
    data_root: str = "data.image_distance.train_10_traj"
    out_dir: str = "out.latent_distance"
    seed: int = 0
    image_size: int = 128
    batch_size: int = 64
    epochs: int = 1000
    lr: float = 1e-3
    device: Optional[str] = None
    B: int = 16  # positive window for NCE

    # loss weights
    lambda_rank: float = 0.5
    lambda_time: float = 0.5
    lambda_nce: float = 1.0
    lambda_tri: float = 0.3
    lambda_aug: float = 0.5
    lambda_vic: float = 1.0

    # logging/viz
    eval_every: int = 5


def build_loaders(cfg: Args):
    index = TrajectoryIndex(Path(cfg.data_root))
    pair_ds = PairDataset(index, image_size=cfg.image_size, B=cfg.B, seed=cfg.seed)
    pair_ld = DataLoader(pair_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    # for retrieval/eval
    frame_ds = FramesDataset(index, image_size=cfg.image_size)
    frame_ld = DataLoader(frame_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    return index, pair_ld, frame_ld


def batch_to_device(batch: dict, device: str):
    out = {}
    for k,v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def sample_negatives_in_batch(d_mat: torch.Tensor, pos_idx: torch.Tensor, K: int = 8) -> torch.Tensor:
    """Given pairwise distances d_mat [B,B], and pos_idx (index of each sample's positive),
    return K negatives per anchor (exclude self and the pos)."""
    B = d_mat.shape[0]
    negs = []
    for b in range(B):
        cand = [i for i in range(B) if i != b and i != int(pos_idx[b])]
        if len(cand) == 0:
            cand = [i for i in range(B) if i != b]
        # choose K farthest among candidates for stability (hard negatives)
        dd = d_mat[b, cand]
        order = torch.argsort(dd, descending=True)
        pick = [cand[int(i)] for i in order[:K]]
        negs.append(torch.stack([d_mat[b, p] for p in pick]))
    return torch.stack(negs, dim=0)  # [B,K]


def train(cfg: Args):
    set_seed(cfg.seed)
    cfg.device = pick_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    index, pair_ld, frame_ld = build_loaders(cfg)
    model = LatentDistanceModel(in_ch=3, feat_dim=256).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    global_step = 0
    run_start_time = time.time()

    for epoch in range(1, cfg.epochs+1):
        model.train()
        t0 = time.time()
        meter = {
            'loss':0.0,
            'rank':0.0,
            'time':0.0,
            'nce':0.0,
            'tri':0.0,
            'aug':0.0,
            'vic':0.0,
        }
        n_batches = 0

        for batch in pair_ld:
            n_batches += 1
            # build tensors
            x_i = batch['x_i'].to(cfg.device)
            x_j = batch['x_j'].to(cfg.device)
            delta_t = batch['delta_t'].to(cfg.device)
            is_augident = batch['is_augident'].to(cfg.device)

            opt.zero_grad()
            h_i, h_j, d_vis, d_hid, d = model(x_i, x_j)

            # --- losses ---
            target_total = delta_t.float()
            L_rank = temporal_ranking_loss(d, delta_t)

            # Vision head learns to explain the full Δt, hidden head covers the residual.
            L_time_vis = F.mse_loss(d_vis, target_total)
            residual_target = torch.clamp(target_total - d_vis.detach(), min=0.0)
            L_time_hid = F.mse_loss(d_hid, residual_target)
            L_time = L_time_vis + L_time_hid

            # NCE: need positives per anchor and negatives from others in batch
            # For simplicity, treat each (i,j) in batch as anchor-pos; compute distance matrix among all anchors to all others' positives
            with torch.no_grad():
                # Build a [B,B] matrix of distances d(anchor b, positive k)
                # Using our computed d between (x_i[b], x_j[k])
                # We'll recompute head on cross pairs for negatives; that's heavier but clearer.
                pass
            # Efficient approximation: build in-batch negatives using d among anchor pairs only
            # Compute full pairwise d between anchors' h_i and h_j within batch for negatives
            D_anchor_anchor = torch.cdist(h_i, h_i, p=2)  # [B,B] (not perfect but workable)
            negs = sample_negatives_in_batch(D_anchor_anchor, pos_idx=torch.arange(h_i.shape[0], device=h_i.device), K=8)
            L_nce = nce_loss(dists_pos=d, dists_neg=negs)

            # Triangle: sample random k in same batch
            if h_i.shape[0] >= 3:
                idx = torch.randperm(h_i.shape[0], device=h_i.device)
                n_triplets = idx.shape[0] // 3
                if n_triplets > 0:
                    triplets = idx[:n_triplets * 3].view(n_triplets, 3)
                    a, b, c = triplets[:, 0], triplets[:, 1], triplets[:, 2]
                    _, _, _, _, d_ab = model(x_i[a], x_j[b])
                    _, _, _, _, d_ac = model(x_i[a], x_j[c])
                    _, _, _, _, d_cb = model(x_i[c], x_j[b])
                    L_tri = triangle_loss(d_ab, d_ac, d_cb)
                else:
                    L_tri = torch.tensor(0.0, device=cfg.device)
            else:
                L_tri = torch.tensor(0.0, device=cfg.device)

            # Augmentation invariance for A0 pairs
            if is_augident.any():
                L_aug = (d_hid[is_augident].abs().mean() + 0.2 * d_vis[is_augident].abs().mean())
            else:
                L_aug = torch.tensor(0.0, device=cfg.device)

            # VICReg on embeddings
            H = torch.cat([h_i, h_j], dim=0)
            L_vic = vicreg_loss(H)

            loss = (
                cfg.lambda_rank * L_rank +
                cfg.lambda_time * L_time +
                cfg.lambda_nce  * L_nce +
                cfg.lambda_tri  * L_tri +
                cfg.lambda_aug  * L_aug +
                cfg.lambda_vic  * L_vic
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            meter['loss'] += float(loss.detach().cpu())
            meter['rank'] += float(L_rank.detach().cpu())
            meter['time'] += float(L_time.detach().cpu())
            meter['nce']  += float(L_nce.detach().cpu())
            meter['tri']  += float(L_tri.detach().cpu())
            meter['aug']  += float(L_aug.detach().cpu())
            meter['vic']  += float(L_vic.detach().cpu())

            global_step += 1

        # epoch summary
        train_elapsed = time.time() - t0
        denom = max(1, n_batches)
        samples = n_batches * cfg.batch_size
        samp_per_sec = samples / max(train_elapsed, 1e-6)
        prefix = f"[{format_elapsed(run_start_time)}, ep {epoch:03d}]"
        msg = (
            f"{prefix} loss {meter['loss']/denom:.4f} | "
            f"rank {meter['rank']/denom:.3f} time {meter['time']/denom:.3f} "
            f"nce {meter['nce']/denom:.3f} tri {meter['tri']/denom:.3f} "
            f"aug {meter['aug']/denom:.3f} vic {meter['vic']/denom:.3f} | "
            f"train_time {train_elapsed:.2f}s ({samp_per_sec:.1f} samp/s) "
            f"step_time {train_elapsed/denom:.2f}s"
        )
        print(msg, flush=True)

        # save checkpoint
        ckpt = {
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'cfg': dataclasses.asdict(cfg),
            'epoch': epoch,
        }
        torch.save(ckpt, os.path.join(cfg.out_dir, "last.pt"))

        if (epoch % cfg.eval_every) == 0:
            with torch.no_grad():
                eval_t0 = time.time()
                eval_stats = run_evals(cfg, model, frame_ld, index, tag=f"ep{epoch:03d}")
                eval_elapsed = time.time() - eval_t0
                n_eval = eval_stats.get('eval_samples', 0)
                avg_num = eval_stats.get('mean_num')
                avg_den = eval_stats.get('mean_den')
            eval_msg = (
                f"{prefix} eval_time {eval_elapsed:.2f}s "
                f"({n_eval/max(eval_elapsed, 1e-6):.1f} samp/s)"
            )
            if avg_num is not None and avg_den is not None:
                eval_msg += f" | sal_num {avg_num:.4f} sal_den {avg_den:.4f}"
            print(eval_msg, flush=True)


# ------------------------------
# Evaluations & Visualizations
# ------------------------------


def _tensor_to_np_image(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()


def _tensor_to_np_heatmap(m: torch.Tensor) -> np.ndarray:
    return m.detach().cpu().squeeze(0).numpy()


def run_evals(cfg: Args, model: LatentDistanceModel, frame_ld: DataLoader, index: TrajectoryIndex, tag: str):
    model.eval()
    out_dir = Path(cfg.out_dir) / f"eval_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_h = []
    all_x = []
    all_meta: List[Tuple[int, int, str]] = []
    with torch.no_grad():
        for batch in frame_ld:
            x = batch['x'].to(cfg.device)
            h = model.enc(x)
            all_h.append(h.cpu())
            all_x.append(x.cpu())
            for ti, tt, pp in zip(batch['traj_idx'], batch['t'], batch['path']):
                all_meta.append((int(ti), int(tt), pp))

    H = torch.cat(all_h, dim=0).cpu().numpy()
    X = torch.cat(all_x, dim=0)
    N = H.shape[0]

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
            _, _, d_vis, d_hid, _ = model(x0.repeat(xs.shape[0], 1, 1, 1), xs)
        d_vis = d_vis.cpu().numpy()
        d_hid = d_hid.cpu().numpy()
        t_steps = np.arange(len(idxs))
        plot_self_distance(
            t_steps,
            d_vis,
            d_hid,
            title=f'Traj {traj_names[ti]}: self-distance (x0 to xt)',
            ylabel='d_vis(x0, xt)',
            color_label='d_hid(x0, xt)',
            out_path=str(out_dir / f"traj{ti:03d}_self_distance.png")
        )

    M = 8
    heat_imgs: List[np.ndarray] = []
    heat_titles: List[str] = []
    rng = random.Random(0)
    for _ in range(M):
        ti = rng.choice(list(traj_to_indices.keys()))
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
            _, _, dvis, dhid, _ = model(xi, xj)
            cam_vis = model.head_heatmaps(xi, xj, which_head='vis')
            cam_hid = model.head_heatmaps(xi, xj, which_head='hid')
        xi_np = _tensor_to_np_image(xi[0])
        heat_imgs.extend([
            xi_np,
            _tensor_to_np_image(xj[0]),
            overlay_heatmap(xi_np, _tensor_to_np_heatmap(cam_vis[0])),
            overlay_heatmap(xi_np, _tensor_to_np_heatmap(cam_hid[0]))
        ])
        heat_titles.extend([
            f"{traj_names[ti]} from t={all_meta[gi][1]}",
            f"{traj_names[ti]} to t={all_meta[gj][1]}",
            f"vis {float(dvis):.2f}",
            f"hid {float(dhid):.2f}"
        ])
    if heat_imgs:
        save_image_grid(heat_imgs, heat_titles, str(out_dir / "head_heatmaps.png"), ncol=4)

    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)

    def cosine_dist(a, B):
        sims = B @ a
        return 1.0 - sims

    K = 5
    sample_cnt = 6
    rng_idx = list(range(N))
    rng.shuffle(rng_idx)
    cards_imgs: List[np.ndarray] = []
    cards_titles: List[str] = []
    taken = 0
    for gidx in rng_idx:
        if taken >= sample_cnt:
            break
        ti, tt, _ = all_meta[gidx]
        mask = np.array([m_ti != ti for (m_ti, _, _) in all_meta])
        Hn_other = Hn[mask]
        other_idx = np.nonzero(mask)[0]
        if Hn_other.shape[0] < K + 1:
            continue
        dists = cosine_dist(Hn[gidx], Hn_other)
        order = np.argsort(dists)
        topk = order[:K]
        botk = order[-K:][::-1]

        anchor = X[gidx]
        cards_imgs.append(_tensor_to_np_image(anchor))
        cards_titles.append(f"anchor {traj_names[ti]} t{tt}")

        xi = anchor.unsqueeze(0).to(cfg.device)
        neigh_idxs = [int(other_idx[i]) for i in topk] + [int(other_idx[i]) for i in botk]
        xj = X[neigh_idxs].to(cfg.device)
        with torch.no_grad():
            _, _, dvis, dhid, d = model(xi.repeat(len(neigh_idxs), 1, 1, 1), xj)
        dvis = dvis.cpu().numpy()
        dhid = dhid.cpu().numpy()
        dtot = d.cpu().numpy()

        for rank, ni in enumerate(neigh_idxs[:K]):
            cards_imgs.append(_tensor_to_np_image(X[ni]))
            nti, ntt, _ = all_meta[ni]
            cards_titles.append(
                f"near{rank + 1}: {traj_names.get(nti, str(nti))} t{ntt}\n"
                f"(dv {dvis[rank]:.2f}, dh {dhid[rank]:.2f}, d {dtot[rank]:.2f})"
            )
        for rank, ni in enumerate(neigh_idxs[K:]):
            cards_imgs.append(_tensor_to_np_image(X[ni]))
            nti, ntt, _ = all_meta[ni]
            cards_titles.append(
                f"far{rank + 1}: {traj_names.get(nti, str(nti))} t{ntt}\n"
                f"(dv {dvis[K + rank]:.2f}, dh {dhid[K + rank]:.2f}, d {dtot[K + rank]:.2f})"
            )

        taken += 1

    if cards_imgs:
        save_image_grid(cards_imgs, cards_titles, str(out_dir / "cross_traj_topk.png"), ncol=1 + 2 * K)

    return N


# ------------------------------
# CLI
# ------------------------------


def main():
    cfg = attach_run_output_dir(tyro.cli(Args))
    cfg.device = pick_device(cfg.device)
    print(f"[Device] {cfg.device} | [Data] {cfg.data_root} | [Out] {cfg.out_dir}", flush=True)
    train(cfg)


if __name__ == '__main__':
    main()
