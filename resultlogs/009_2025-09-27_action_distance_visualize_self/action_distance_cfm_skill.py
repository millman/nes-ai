#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill-conditioned CFM trainer with ms-SSIM reconstruction loss."""

from __future__ import annotations

import argparse
import contextlib
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, get_worker_info

from PIL import Image

from recon import (
    H,
    W,
    load_frame_as_tensor as base_load_frame_as_tensor,
    set_seed,
    to_float01,
)
from recon.data import list_trajectories
# Decoder redefined locally to mirror action_distance_recon.py
from recon.utils import psnr_01, tensor_to_pil

# --------------------------------------------------------------------------------------
# Image similarity helpers (copied from action_distance_recon.py)
# --------------------------------------------------------------------------------------

def _normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    return t


def _denormalize_tensor(t: torch.Tensor) -> torch.Tensor:
    return t


def load_frame_as_tensor(path: Path) -> torch.Tensor:
    return base_load_frame_as_tensor(path, normalize=_normalize_tensor)


def _gaussian_window(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = (gauss / gauss.sum()).unsqueeze(0)
    window_2d = (gauss.t() @ gauss).unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def _ssim_components(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    padding = window.shape[-1] // 2
    mu_x = F.conv2d(x, window, padding=padding, groups=x.shape[1])
    mu_y = F.conv2d(y, window, padding=padding, groups=y.shape[1])

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=padding, groups=x.shape[1]) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=padding, groups=y.shape[1]) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=x.shape[1]) - mu_xy

    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2

    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    ssim_map = numerator / (denominator + 1e-8)
    cs_map = (2 * sigma_xy + C2) / (sigma_x_sq + sigma_y_sq + C2 + 1e-8)

    ssim_val = ssim_map.mean(dim=(1, 2, 3))
    cs_val = cs_map.mean(dim=(1, 2, 3))
    return ssim_val, cs_val


def ms_ssim(x: torch.Tensor, y: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weights is None:
        weights = torch.tensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
            device=x.device,
            dtype=x.dtype,
        )
    window_size = 11
    sigma = 1.5
    channels = x.shape[1]
    window = _gaussian_window(window_size, sigma, channels, x.device)

    levels = weights.shape[0]
    mssim: List[torch.Tensor] = []
    mcs: List[torch.Tensor] = []

    x_scaled, y_scaled = x, y
    for _ in range(levels):
        ssim_val, cs_val = _ssim_components(x_scaled, y_scaled, window)
        mssim.append(ssim_val)
        mcs.append(cs_val)
        x_scaled = F.avg_pool2d(x_scaled, kernel_size=2, stride=2, padding=0, ceil_mode=False)
        y_scaled = F.avg_pool2d(y_scaled, kernel_size=2, stride=2, padding=0, ceil_mode=False)

    mssim_tensor = torch.stack(mssim, dim=0)
    mcs_tensor = torch.stack(mcs[:-1], dim=0)

    pow1 = weights[:-1].unsqueeze(1)
    pow2 = weights[-1]
    ms_prod = torch.prod(mcs_tensor ** pow1, dim=0) * (mssim_tensor[-1] ** pow2)
    return ms_prod.mean()


def ms_ssim_loss(x: torch.Tensor, y: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    return 1.0 - ms_ssim(x, y, weights=weights)


# --------------------------------------------------------------------------------------
# Dataset with fixed-length skill chunks
# --------------------------------------------------------------------------------------

class SkillCFMDataset(Dataset):
    """Return fixed-length frame chunks for skill-conditioned CFM training."""

    def __init__(
        self,
        data_root: Path,
        *,
        split: str,
        train_frac: float,
        seed: int,
        chunk_len: int,
        delta_choices: Sequence[int],
        load_frame: Optional[callable] = None,
    ) -> None:
        super().__init__()
        if chunk_len < 2:
            raise ValueError("chunk_len must be >= 2")
        self.chunk_len = chunk_len
        self.delta_choices = sorted([d for d in delta_choices if 1 <= d < chunk_len])
        if not self.delta_choices:
            raise ValueError("delta_choices must contain values in [1, chunk_len-1]")

        self._load_frame = load_frame if load_frame is not None else load_frame_as_tensor
        traj_map = list_trajectories(data_root)
        traj_items = list(traj_map.items())

        rng = random.Random(seed)
        rng.shuffle(traj_items)
        split_idx = max(1, int(round(len(traj_items) * train_frac)))
        if split_idx >= len(traj_items):
            split_idx = max(1, len(traj_items) - 1)
        if split == "train":
            chosen = traj_items[:split_idx]
        elif split == "val":
            chosen = traj_items[split_idx:]
        else:
            raise ValueError("split must be 'train' or 'val'")
        if not chosen:
            chosen = traj_items

        self.samples: List[List[Path]] = []
        for _, paths in chosen:
            if len(paths) < chunk_len:
                continue
            for start in range(0, len(paths) - chunk_len + 1):
                chunk_paths = paths[start : start + chunk_len]
                self.samples.append(chunk_paths)
        if not self.samples:
            raise RuntimeError("No chunks available for the given configuration")

        self._main_rng = random.Random(seed)
        self._worker_rngs: dict[int, random.Random] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _get_worker_rng(self) -> random.Random:
        info = get_worker_info()
        if info is None:
            return self._main_rng
        rng = self._worker_rngs.get(info.id)
        if rng is None:
            rng = random.Random(info.seed)
            self._worker_rngs[info.id] = rng
        return rng

    def __getitem__(self, idx: int):  # type: ignore[override]
        paths = self.samples[idx]
        rng = self._get_worker_rng()
        delta = rng.choice(self.delta_choices)
        frames = [self._load_frame(p) for p in paths]
        chunk = torch.stack(frames, dim=0)
        return chunk, delta, [str(p) for p in paths]


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------

class Encoder(nn.Module):
    """Lightweight conv encoder matching action_distance_recon."""

    def __init__(self, z_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, z_dim)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.fc(h)


class Decoder(nn.Module):
    """Transpose-conv decoder mirrored from action_distance_recon."""

    def __init__(self, z_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256 * 15 * 14)
        self.pre = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
        )
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 256, 15, 14)
        h = self.pre(h)
        x = self.net(h)
        return torch.sigmoid(x)


class SkillEncoder(nn.Module):
    def __init__(self, z_dim: int = 128, skill_dim: int = 32, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim * 3, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, skill_dim),
        )

    def forward(self, z_seq: torch.Tensor, delta_idx: torch.Tensor) -> torch.Tensor:
        # z_seq: [B, L, D], delta_idx: [B]
        zA = z_seq[:, 0]
        batch = z_seq.shape[0]
        idx = delta_idx.view(-1, 1, 1).expand(-1, 1, z_seq.shape[2])
        zB = torch.gather(z_seq, 1, idx).squeeze(1)

        L = z_seq.shape[1]
        device = z_seq.device
        steps = torch.arange(L, device=device).unsqueeze(0)
        mask = (steps <= delta_idx.unsqueeze(1)).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        mean_future = (z_seq * mask).sum(dim=1) / denom

        delta = zB - zA
        drift = mean_future - zA
        feat = torch.cat([delta, drift, mean_future], dim=1)
        return self.net(feat)


class SkillDeltaHead(nn.Module):
    def __init__(self, skill_dim: int = 32, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(skill_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, omega: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(omega))


class VectorField(nn.Module):
    def __init__(
        self,
        z_dim: int = 128,
        skill_dim: int = 32,
        hidden: int = 512,
        delta_feat_dim: int = 3,
    ):
        super().__init__()
        self.delta_feat_dim = delta_feat_dim
        in_dim = z_dim * 3 + skill_dim + 1 + delta_feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, z_dim),
        )

    def forward(
        self,
        zt: torch.Tensor,
        t: torch.Tensor,
        zA: torch.Tensor,
        zB: torch.Tensor,
        omega: torch.Tensor,
        delta_feat: torch.Tensor,
    ) -> torch.Tensor:
        delta = zB - zA
        mid = 0.5 * (zA + zB)
        if delta_feat.shape[1] != self.delta_feat_dim:
            raise ValueError(
                f"delta_feat has dim {delta_feat.shape[1]} but expected {self.delta_feat_dim}"
            )
        feat = torch.cat([zt, delta, mid, omega, t[:, None], delta_feat], dim=1)
        return self.net(feat)


# --------------------------------------------------------------------------------------
# Training utilities
# --------------------------------------------------------------------------------------

def kaiming_init(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _psnr_01(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return psnr_01(_denormalize_tensor(x), _denormalize_tensor(y))


def _to_pil(t: torch.Tensor):
    return tensor_to_pil(t, denormalize=_denormalize_tensor)


@dataclass
class TrainConfig:
    data_root: Path
    out_dir: Path = Path("out.action_distance_cfm_skill")
    seed: int = 0
    device: Optional[str] = None
    epochs: int = 50
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 3e-4
    vf_lr_mult: float = 0.5
    chunk_len: int = 8
    delta_choices: Tuple[int, ...] = (1, 2, 4, 6)
    z_dim: int = 128
    skill_dim: int = 32
    lambda_cfm: float = 0.3
    lambda_rec_l1: float = 1.0
    lambda_rec_ms_ssim: float = 0.5
    lambda_latent_l2: float = 1e-3
    lambda_zcal: float = 1e-3
    lambda_skill: float = 0.1
    warmup_steps: int = 2_000
    log_every: int = 50
    viz_every: int = 50
    val_every: int = 1
    max_grad_norm: Optional[float] = 1.0
    use_amp: bool = False
    train_frac: float = 0.95


def make_loaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    ds_tr = SkillCFMDataset(
        cfg.data_root,
        split="train",
        train_frac=cfg.train_frac,
        seed=cfg.seed,
        chunk_len=cfg.chunk_len,
        delta_choices=cfg.delta_choices,
        load_frame=load_frame_as_tensor,
    )
    ds_va = SkillCFMDataset(
        cfg.data_root,
        split="val",
        train_frac=cfg.train_frac,
        seed=cfg.seed,
        chunk_len=cfg.chunk_len,
        delta_choices=cfg.delta_choices,
        load_frame=load_frame_as_tensor,
    )
    dl_tr = DataLoader(
        ds_tr,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return dl_tr, dl_va


@torch.no_grad()
def save_debug_grid(
    enc: Encoder,
    dec: Decoder,
    batch_chunk: torch.Tensor,
    batch_delta: torch.Tensor,
    out_path: Path,
    device: torch.device,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Bsz = batch_chunk.shape[0]
    chunk_dev = to_float01(batch_chunk.view(-1, 3, H, W), device)
    chunk_dev = chunk_dev.view(Bsz, -1, 3, H, W)

    idx = torch.arange(Bsz, device=device)
    z_seq = enc(chunk_dev.view(-1, 3, H, W)).view(Bsz, -1, enc.fc.out_features)
    zA = z_seq[:, 0]
    zB = z_seq[idx, batch_delta.to(device)]
    recon = dec(torch.cat([zA, zB], dim=0))
    xA_rec, xB_rec = recon.chunk(2, dim=0)

    rows = min(4, Bsz)
    cols = 4
    canvas = Image.new("RGB", (cols * W, rows * H))
    for r in range(rows):
        idx_b = int(batch_delta[r].item())
        images = [
            batch_chunk[r, 0].cpu(),
            _denormalize_tensor(xA_rec[r].cpu()),
            batch_chunk[r, idx_b].cpu(),
            _denormalize_tensor(xB_rec[r].cpu()),
        ]
        for c, img_t in enumerate(images):
            canvas.paste(_to_pil(img_t), (c * W, r * H))
    canvas.save(out_path)


def pick_device(pref: Optional[str]) -> torch.device:
    if pref:
        dev = torch.device(pref)
        if dev.type == "cuda":
            raise ValueError("CUDA is not supported by this trainer; use 'cpu' or 'mps'.")
        return dev
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(cfg: TrainConfig) -> None:
    device = pick_device(cfg.device)
    set_seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.monotonic()

    dl_tr, dl_va = make_loaders(cfg)
    print(
        f"[Device] {device} | "
        f"[Data] train chunks≈{len(dl_tr.dataset)} val≈{len(dl_va.dataset)} | "
        f"chunk_len={cfg.chunk_len} deltas={list(cfg.delta_choices)}"
    )

    enc = Encoder(cfg.z_dim).to(device)
    dec = Decoder(cfg.z_dim).to(device)
    vf = VectorField(cfg.z_dim, cfg.skill_dim).to(device)
    skill_enc = SkillEncoder(cfg.z_dim, cfg.skill_dim).to(device)
    skill_head = SkillDeltaHead(cfg.skill_dim).to(device)

    dec.apply(kaiming_init)
    vf.apply(kaiming_init)
    skill_enc.apply(kaiming_init)
    skill_head.apply(kaiming_init)

    opt = torch.optim.AdamW(
        [
            {"params": enc.parameters(), "lr": cfg.lr},
            {"params": dec.parameters(), "lr": cfg.lr},
            {"params": vf.parameters(), "lr": cfg.lr * cfg.vf_lr_mult},
            {"params": skill_enc.parameters(), "lr": cfg.lr},
            {"params": skill_head.parameters(), "lr": cfg.lr},
        ],
        weight_decay=1e-4,
    )

    scaler = GradScaler(enabled=cfg.use_amp and device.type == "cuda")
    autocast_ctx = autocast if scaler.is_enabled() else contextlib.nullcontext

    best_val = float("inf")
    global_step = 0
    win = 100
    q_loss_total = deque(maxlen=win)
    q_loss_cfm = deque(maxlen=win)
    q_loss_rec = deque(maxlen=win)
    q_loss_skill = deque(maxlen=win)
    q_psnr = deque(maxlen=win)

    for epoch in range(1, cfg.epochs + 1):
        enc.train()
        dec.train()
        vf.train()
        skill_enc.train()
        skill_head.train()

        run_loss = run_cfm = run_rec = run_skill = 0.0
        run_n = 0

        for chunk, delta_idx, _ in dl_tr:
            chunk = chunk.to(device, non_blocking=True)
            delta_idx = delta_idx.to(device, non_blocking=True)
            batch_size, seq_len = chunk.shape[:2]

            flat = chunk.view(-1, 3, H, W)
            flat = to_float01(flat, device)
            flat = flat.view(batch_size, seq_len, 3, H, W)

            idx_range = torch.arange(batch_size, device=device)
            A = flat[:, 0]
            B = flat[idx_range, delta_idx]

            with autocast_ctx():
                z_seq = enc(flat.view(-1, 3, H, W)).view(batch_size, seq_len, cfg.z_dim)
                zA = z_seq[:, 0]
                zB = z_seq[idx_range, delta_idx]

                omega = skill_enc(z_seq, delta_idx)
                delta_norm = delta_idx.float() / float(cfg.chunk_len - 1)
                delta_feat = torch.stack(
                    [delta_norm, torch.log1p(delta_idx.float()), delta_norm ** 2],
                    dim=1,
                )

                t = torch.rand(batch_size, device=device)
                zA_d = zA.detach()
                zB_d = zB.detach()
                zt = (1.0 - t[:, None]) * zA_d + t[:, None] * zB_d
                target = zB_d - zA_d

                v = vf(zt, t, zA_d, zB_d, omega, delta_feat)
                eps = 1e-6
                cos_term = 1.0 - F.cosine_similarity(v, target, dim=1).mean()
                mag_term = (
                    torch.linalg.norm(v - target, dim=1)
                    / (torch.linalg.norm(target, dim=1) + eps)
                ).mean()
                loss_cfm = cos_term + mag_term

                pair_latents = torch.cat([zA, zB], dim=0)
                recon = dec(pair_latents)
                xA_rec, xB_rec = recon.chunk(2, dim=0)

                loss_l1 = F.l1_loss(xA_rec, A) + F.l1_loss(xB_rec, B)
                if cfg.lambda_rec_ms_ssim > 0:
                    loss_ms = ms_ssim_loss(xA_rec, A) + ms_ssim_loss(xB_rec, B)
                else:
                    loss_ms = torch.tensor(0.0, device=device, dtype=xA_rec.dtype)
                loss_rec = cfg.lambda_rec_l1 * loss_l1 + cfg.lambda_rec_ms_ssim * loss_ms

                loss_skill = F.mse_loss(skill_head(omega).squeeze(-1), delta_norm)

                loss_lat = 0.5 * (zA.pow(2).mean() + zB.pow(2).mean())
                z_std = torch.cat([zA, zB], dim=0).std(dim=0)
                loss_zcal = (z_std - 1.0).abs().mean()

                cfm_w = 0.0 if global_step < cfg.warmup_steps else cfg.lambda_cfm
                lat_w = 0.0 if global_step < cfg.warmup_steps else cfg.lambda_latent_l2
                zcal_w = 0.0 if global_step < cfg.warmup_steps else cfg.lambda_zcal

                loss = (
                    cfm_w * loss_cfm
                    + loss_rec
                    + cfg.lambda_skill * loss_skill
                    + lat_w * loss_lat
                    + zcal_w * loss_zcal
                )

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.max_grad_norm is not None:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(vf.parameters(), cfg.max_grad_norm)
            scaler.step(opt)
            scaler.update()

            psnr = 0.5 * (_psnr_01(xA_rec, A) + _psnr_01(xB_rec, B))

            run_loss += float(loss.item()) * batch_size
            run_cfm += float(loss_cfm.item()) * batch_size
            run_rec += float(loss_rec.item()) * batch_size
            run_skill += float(loss_skill.item()) * batch_size
            run_n += batch_size

            q_loss_total.append(float(loss.item()))
            q_loss_cfm.append(float(loss_cfm.item()))
            q_loss_rec.append(float(loss_rec.item()))
            q_loss_skill.append(float(loss_skill.item()))
            q_psnr.append(float(psnr.item()))

            if cfg.log_every > 0 and global_step % cfg.log_every == 0:
                avg = lambda q: sum(q) / max(1, len(q))
                elapsed = int(time.monotonic() - start_time)
                h = elapsed // 3600
                m = (elapsed % 3600) // 60
                s = elapsed % 60
                print(
                    f"[{h:02d}:{m:02d}:{s:02d}] "
                    f"ep {epoch:02d} step {global_step:06d} | "
                    f"loss {avg(q_loss_total):.4f} | "
                    f"Lcfm {avg(q_loss_cfm):.4f} Lrec {avg(q_loss_rec):.4f} Lskill {avg(q_loss_skill):.4f} | "
                    f"PSNR {avg(q_psnr):.2f}dB"
                )

            if cfg.viz_every > 0 and global_step % cfg.viz_every == 0:
                enc.eval()
                dec.eval()
                save_debug_grid(
                    enc,
                    dec,
                    chunk[:4].cpu(),
                    delta_idx[:4].cpu(),
                    cfg.out_dir / f"viz_step_{global_step:06d}.png",
                    device,
                )
                enc.train()
                dec.train()

            global_step += 1

        tr_loss = run_loss / max(1, run_n)
        tr_cfm = run_cfm / max(1, run_n)
        tr_rec = run_rec / max(1, run_n)
        print(f"[ep {epoch:02d}] train: loss={tr_loss:.4f} Lcfm={tr_cfm:.4f} Lrec={tr_rec:.4f}")

        if epoch % cfg.val_every != 0:
            continue

        enc.eval()
        dec.eval()
        vf.eval()
        skill_enc.eval()
        skill_head.eval()

        va_loss = va_cfm = va_rec = va_skill = 0.0
        va_n = 0
        with torch.no_grad():
            for chunk, delta_idx, _ in dl_va:
                chunk = chunk.to(device, non_blocking=True)
                delta_idx = delta_idx.to(device, non_blocking=True)
                batch_size, seq_len = chunk.shape[:2]

                flat = chunk.view(-1, 3, H, W)
                flat = to_float01(flat, device)
                flat = flat.view(batch_size, seq_len, 3, H, W)

                idx_range = torch.arange(batch_size, device=device)
                A = flat[:, 0]
                B = flat[idx_range, delta_idx]

                z_seq = enc(flat.view(-1, 3, H, W)).view(batch_size, seq_len, cfg.z_dim)
                zA = z_seq[:, 0]
                zB = z_seq[idx_range, delta_idx]

                omega = skill_enc(z_seq, delta_idx)
                delta_norm = delta_idx.float() / float(cfg.chunk_len - 1)
                delta_feat = torch.stack(
                    [delta_norm, torch.log1p(delta_idx.float()), delta_norm ** 2],
                    dim=1,
                )

                t = torch.rand(batch_size, device=device)
                zA_d = zA
                zB_d = zB
                zt = (1.0 - t[:, None]) * zA_d + t[:, None] * zB_d
                target = zB_d - zA_d
                v = vf(zt, t, zA_d, zB_d, omega, delta_feat)
                eps = 1e-6
                cos_term = 1.0 - F.cosine_similarity(v, target, dim=1).mean()
                mag_term = (
                    torch.linalg.norm(v - target, dim=1)
                    / (torch.linalg.norm(target, dim=1) + eps)
                ).mean()
                loss_cfm = cos_term + mag_term

                pair_latents = torch.cat([zA, zB], dim=0)
                recon = dec(pair_latents)
                xA_rec, xB_rec = recon.chunk(2, dim=0)

                loss_l1 = F.l1_loss(xA_rec, A) + F.l1_loss(xB_rec, B)
                if cfg.lambda_rec_ms_ssim > 0:
                    loss_ms = ms_ssim_loss(xA_rec, A) + ms_ssim_loss(xB_rec, B)
                else:
                    loss_ms = torch.tensor(0.0, device=device, dtype=xA_rec.dtype)
                loss_rec = cfg.lambda_rec_l1 * loss_l1 + cfg.lambda_rec_ms_ssim * loss_ms

                loss_skill = F.mse_loss(skill_head(omega).squeeze(-1), delta_norm)

                loss = cfg.lambda_cfm * loss_cfm + loss_rec + cfg.lambda_skill * loss_skill

                va_loss += float(loss.item()) * batch_size
                va_cfm += float(loss_cfm.item()) * batch_size
                va_rec += float(loss_rec.item()) * batch_size
                va_skill += float(loss_skill.item()) * batch_size
                va_n += batch_size

        va_loss /= max(1, va_n)
        va_cfm /= max(1, va_n)
        va_rec /= max(1, va_n)
        va_skill /= max(1, va_n)
        print(
            f"[ep {epoch:02d}]   val: loss={va_loss:.4f} Lcfm={va_cfm:.4f} Lrec={va_rec:.4f} Lskill={va_skill:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "enc": enc.state_dict(),
            "dec": dec.state_dict(),
            "vf": vf.state_dict(),
            "skill_enc": skill_enc.state_dict(),
            "skill_head": skill_head.state_dict(),
            "cfg": cfg,
        }
        torch.save(ckpt, cfg.out_dir / "last.pt")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, cfg.out_dir / "best.pt")
            print(f"[ep {epoch:02d}] saved new best checkpoint (val loss {best_val:.4f})")

    elapsed = time.monotonic() - start_time
    print(f"Training finished in {elapsed/60:.2f} min ({elapsed:.1f} s)")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=TrainConfig.out_dir)
    ap.add_argument("--seed", type=int, default=TrainConfig.seed)
    ap.add_argument("--device", type=str, default=TrainConfig.device)
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    ap.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    ap.add_argument("--num_workers", type=int, default=TrainConfig.num_workers)
    ap.add_argument("--lr", type=float, default=TrainConfig.lr)
    ap.add_argument("--vf_lr_mult", type=float, default=TrainConfig.vf_lr_mult)
    ap.add_argument("--chunk_len", type=int, default=TrainConfig.chunk_len)
    ap.add_argument(
        "--delta_choices",
        type=int,
        nargs="+",
        default=list(TrainConfig.delta_choices),
    )
    ap.add_argument("--z_dim", type=int, default=TrainConfig.z_dim)
    ap.add_argument("--skill_dim", type=int, default=TrainConfig.skill_dim)
    ap.add_argument("--lambda_cfm", type=float, default=TrainConfig.lambda_cfm)
    ap.add_argument("--lambda_rec_l1", type=float, default=TrainConfig.lambda_rec_l1)
    ap.add_argument(
        "--lambda_rec_ms_ssim",
        type=float,
        default=TrainConfig.lambda_rec_ms_ssim,
    )
    ap.add_argument("--lambda_latent_l2", type=float, default=TrainConfig.lambda_latent_l2)
    ap.add_argument("--lambda_zcal", type=float, default=TrainConfig.lambda_zcal)
    ap.add_argument("--lambda_skill", type=float, default=TrainConfig.lambda_skill)
    ap.add_argument("--warmup_steps", type=int, default=TrainConfig.warmup_steps)
    ap.add_argument("--log_every", type=int, default=TrainConfig.log_every)
    ap.add_argument("--viz_every", type=int, default=TrainConfig.viz_every)
    ap.add_argument("--val_every", type=int, default=TrainConfig.val_every)
    ap.add_argument("--max_grad_norm", type=float, default=TrainConfig.max_grad_norm or 0.0)
    ap.add_argument("--use_amp", action="store_true", default=TrainConfig.use_amp)
    ap.add_argument("--train_frac", type=float, default=TrainConfig.train_frac)
    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()
    cfg = TrainConfig(
        data_root=args.data_root,
        out_dir=args.out_dir,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        vf_lr_mult=args.vf_lr_mult,
        chunk_len=args.chunk_len,
        delta_choices=tuple(args.delta_choices),
        z_dim=args.z_dim,
        skill_dim=args.skill_dim,
        lambda_cfm=args.lambda_cfm,
        lambda_rec_l1=args.lambda_rec_l1,
        lambda_rec_ms_ssim=args.lambda_rec_ms_ssim,
        lambda_latent_l2=args.lambda_latent_l2,
        lambda_zcal=args.lambda_zcal,
        lambda_skill=args.lambda_skill,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
        viz_every=args.viz_every,
        val_every=args.val_every,
        max_grad_norm=None if args.max_grad_norm == 0.0 else args.max_grad_norm,
        use_amp=args.use_amp,
        train_frac=args.train_frac,
    )
    train(cfg)


if __name__ == "__main__":
    main()
