#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Action-distance reconstruction trainer with a lightweight CNN autoencoder and
full A→B latent interpolation visualization with trajectory/state labels.

Run:
  python action_distance_recon.py --data_root traj_dumps --out_dir out.action_distance_recon
"""

from __future__ import annotations
import argparse, contextlib, math, os, random, time, warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, get_worker_info

# --------------------------------------------------------------------------------------
# Global image size (NES frames resized to this)
# --------------------------------------------------------------------------------------
H, W = 240, 224  # (height, width)

# --------------------------------------------------------------------------------------
# IO utils
# --------------------------------------------------------------------------------------
def list_trajectories(data_root: Path) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for traj_dir in sorted(data_root.glob("traj_*")):
        state_dir = traj_dir / "states"
        if not state_dir.is_dir():
            continue
        paths = sorted(state_dir.glob("state_*.png"), key=lambda p: int(p.stem.split("_")[1]))
        if paths:
            out[traj_dir.name] = paths
    if not out:
        raise FileNotFoundError(f"No trajectories under {data_root} (expected traj_*/states/state_*.png)")
    return out


def _normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    return t  # kept for compatibility with legacy checkpoints


def _denormalize_tensor(t: torch.Tensor) -> torch.Tensor:
    return t  # kept for compatibility with legacy checkpoints


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32, copy=True) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return _normalize_tensor(t)


def load_frame_as_tensor(p: Path) -> torch.Tensor:
    img = Image.open(p).convert("RGB").resize((W, H), resample=Image.NEAREST)
    return pil_to_tensor(img)


def short_traj_state_label(path_str: str) -> str:
    base = os.path.normpath(path_str)
    parts = base.split(os.sep)
    traj_idx = next((i for i, p in enumerate(parts) if p.startswith("traj_")), None)
    if (
        traj_idx is not None
        and traj_idx + 2 < len(parts)
        and parts[traj_idx + 1] == "states"
        and parts[traj_idx + 2].startswith("state_")
    ):
        return f"{parts[traj_idx]}/{os.path.splitext(parts[traj_idx + 2])[0]}"
    if len(parts) >= 2:
        return f"{parts[-2]}/{os.path.splitext(parts[-1])[0]}"
    return os.path.splitext(os.path.basename(base))[0]


# --------------------------------------------------------------------------------------
# Dataset: by default, sample A/B from the SAME trajectory
# --------------------------------------------------------------------------------------
class PairFromTrajDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str = "train",
        train_frac: float = 0.95,
        seed: int = 0,
        max_step_gap: int = 20,
        allow_cross_traj: bool = False,
        p_cross_traj: float = 0.0,
    ):
        super().__init__()
        self.trajs = list_trajectories(data_root)
        self.traj_items = list(self.trajs.items())
        all_paths = [p for lst in self.trajs.values() for p in lst]

        self._base_seed = seed
        self._main_rng = random.Random(seed)
        self._worker_rngs: Dict[int, random.Random] = {}

        self._main_rng.shuffle(all_paths)
        n_train = int(round(len(all_paths) * train_frac))
        self.pool = all_paths[:n_train] if split == "train" else all_paths[n_train:]
        self.max_step_gap = max_step_gap
        self.allow_cross_traj = allow_cross_traj
        self.p_cross_traj = p_cross_traj if allow_cross_traj else 0.0

        pool_set = set(map(str, self.pool))
        self.split_trajs: Dict[str, List[Path]] = {}
        for traj_name, paths in self.traj_items:
            kept = [p for p in paths if str(p) in pool_set]
            if len(kept) >= 2:
                self.split_trajs[traj_name] = kept
        if not self.split_trajs:
            raise RuntimeError(f"No trajectories with >=2 frames in split='{split}'")
        self.split_traj_items = list(self.split_trajs.items())

    def __len__(self):
        return sum(len(v) for v in self.split_trajs.values())

    def _get_worker_rng(self) -> random.Random:
        info = get_worker_info()
        if info is None:
            return self._main_rng
        wid = info.id
        rng = self._worker_rngs.get(wid)
        if rng is None:
            rng = random.Random(info.seed)
            self._worker_rngs[wid] = rng
        return rng

    def _sample_same_traj_pair(self, rng: random.Random) -> Tuple[Path, Path]:
        traj_idx = rng.randrange(len(self.split_traj_items))
        _, paths = self.split_traj_items[traj_idx]
        if len(paths) < 2:
            return paths[0], paths[-1]
        i0 = rng.randrange(0, len(paths) - 1)
        gap = rng.randint(1, min(max(1, self.max_step_gap), len(paths) - 1 - i0))
        j0 = i0 + gap
        return paths[i0], paths[j0]

    def _sample_cross_traj_pair(self, rng: random.Random) -> Tuple[Path, Path]:
        idx1 = rng.randrange(len(self.split_traj_items))
        idx2 = rng.randrange(len(self.split_traj_items))
        p1s = self.split_traj_items[idx1][1]
        p2s = self.split_traj_items[idx2][1]
        return p1s[rng.randrange(len(p1s))], p2s[rng.randrange(len(p2s))]

    def __getitem__(self, idx: int):
        rng = self._get_worker_rng()
        if self.allow_cross_traj and (rng.random() < self.p_cross_traj):
            p1, p2 = self._sample_cross_traj_pair(rng)
        else:
            p1, p2 = self._sample_same_traj_pair(rng)
        a = load_frame_as_tensor(p1)
        b = load_frame_as_tensor(p2)
        return a, b, str(p1), str(p2)


def to_float01(t: torch.Tensor, device: torch.device, non_blocking: bool = True) -> torch.Tensor:
    if t.dtype != torch.uint8:
        return t.to(device=device, non_blocking=non_blocking)
    return t.to(device=device, non_blocking=non_blocking, dtype=torch.float32) / 255.0


# --------------------------------------------------------------------------------------
# Lightweight CNN AE (no skips): Encoder/Decoder
# --------------------------------------------------------------------------------------
class DownBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=2, padding=padding),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, z_dim: int = 128, pretrained: bool = False, freeze_backbone: bool = False):
        super().__init__()
        if pretrained or freeze_backbone:
            warnings.warn(
                "Lightweight encoder does not support pretrained/freeze flags; ignoring.",
                RuntimeWarning,
            )

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.down1 = DownBlock(32, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, z_dim)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.pool(h).flatten(1)
        return self.fc(h)


class Decoder(nn.Module):
    def __init__(self, z_dim: int = 128):
        super().__init__()
        self.h0 = (256, 15, 14)
        self.fc = nn.Linear(z_dim, int(np.prod(self.h0)))
        self.pre = nn.SiLU(inplace=True)
        self.up1 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.up3 = UpBlock(64, 32)
        self.up4 = UpBlock(32, 16)
        self.head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, *self.h0)
        h = self.pre(h)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        x = self.head(h)
        return torch.clamp(x, 0.0, 1.0)


# --------------------------------------------------------------------------------------
# Utilities & debug helpers
# --------------------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu()
    if t.dtype == torch.uint8:
        arr = t.permute(1, 2, 0).contiguous().numpy()
    else:
        t = _denormalize_tensor(t)
        if t.dtype != torch.float32:
            t = t.float()
        t = t.clamp(0.0, 1.0)
        arr = (t.permute(1, 2, 0).contiguous().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _psnr_01(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.dtype != torch.float32:
        x01 = x.float().div(255.0)
    else:
        x01 = x
    if y.dtype != torch.float32:
        y01 = y.float().div(255.0)
    else:
        y01 = y
    x01 = _denormalize_tensor(x01).clamp(0.0, 1.0)
    y01 = _denormalize_tensor(y01).clamp(0.0, 1.0)
    mse = F.mse_loss(x01, y01, reduction="mean").clamp_min(eps)
    return 10.0 * torch.log10(1.0 / mse)


def _grad_norm(module: nn.Module) -> float:
    s = 0.0
    for p in module.parameters():
        if p.grad is not None:
            s += float(p.grad.detach().data.pow(2).sum().cpu())
    return math.sqrt(s) if s > 0 else 0.0


@torch.no_grad()
def save_full_interpolation_grid(
    enc: Encoder,
    dec: Decoder,
    A: torch.Tensor,
    B: torch.Tensor,
    pathsA: List[str],
    pathsB: List[str],
    out_path: Path,
    device: torch.device,
    interp_steps: int = 12,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Bsz = min(A.shape[0], 8)
    K = max(1, int(interp_steps))

    A = A[:Bsz].contiguous()
    B = B[:Bsz].contiguous()
    pA = pathsA[:Bsz]
    pB = pathsB[:Bsz]

    A_dev = to_float01(A, device, non_blocking=False)
    B_dev = to_float01(B, device, non_blocking=False)
    zA = enc(A_dev)
    zB = enc(B_dev)
    pair_recon = dec(torch.cat([zA, zB], dim=0))
    dec_zA, dec_zB = pair_recon.chunk(2, dim=0)

    t_vals = torch.linspace(0, 1, K + 2, device=device)[1:-1]
    if K > 0:
        z_interp = torch.stack([(1.0 - t) * zA + t * zB for t in t_vals], dim=0)
        interp_decoded = dec(z_interp.view(-1, zA.shape[1])).view(K, Bsz, 3, H, W)
    else:
        interp_decoded = torch.empty(0, Bsz, 3, H, W, device=device)

    tiles_per_row = 2 + K + 2  # A | dec(zA) | interps | dec(zB) | B
    tile_w, tile_h = W, H
    total_w = tile_w * tiles_per_row
    total_h = tile_h * Bsz
    canvas = Image.new("RGB", (total_w, total_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=18)
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    def annotate(draw_ctx, x, y, txt, fill=(255, 255, 255)):
        if font:
            bbox = draw_ctx.textbbox((0, 0), txt, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            tw, th = len(txt) * 6, 10
        pad = 4
        draw_ctx.rectangle([x + 2, y + 2, x + 2 + tw + pad * 2, y + 2 + th + pad * 2], fill=(0, 0, 0))
        draw_ctx.text((x + 2 + pad, y + 2 + pad), txt, font=font, fill=fill)

    for r in range(Bsz):
        x = 0
        y = r * tile_h
        labelA = short_traj_state_label(pA[r])
        labelB = short_traj_state_label(pB[r])

        canvas.paste(_to_pil(A[r]), (x, y))
        annotate(draw, x, y, labelA)
        x += tile_w
        canvas.paste(_to_pil(dec_zA[r]), (x, y))
        annotate(draw, x, y, labelA + " (dec)", (220, 220, 255))
        x += tile_w

        for k in range(K):
            canvas.paste(_to_pil(interp_decoded[k, r]), (x, y))
            annotate(draw, x, y, f"t={float(t_vals[k]):.2f}", (200, 255, 200))
            x += tile_w

        canvas.paste(_to_pil(dec_zB[r]), (x, y))
        annotate(draw, x, y, labelB + " (dec)", (220, 220, 255))
        x += tile_w
        canvas.paste(_to_pil(B[r]), (x, y))
        annotate(draw, x, y, labelB)

        txt = f"‖zB−zA‖={torch.linalg.norm(zB[r] - zA[r]).item():.2f}"
        annotate(draw, 2, y + tile_h - 22, txt, (255, 255, 0))

    canvas.save(out_path)


@torch.no_grad()
def save_simple_debug_grid(
    enc: Encoder,
    dec: Decoder,
    A: torch.Tensor,
    B: torch.Tensor,
    pathsA: List[str],
    pathsB: List[str],
    out_path: Path,
    device: torch.device,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Bsz = min(A.shape[0], 4)
    A = A[:Bsz].contiguous()
    B = B[:Bsz].contiguous()

    A_dev = to_float01(A, device, non_blocking=False)
    B_dev = to_float01(B, device, non_blocking=False)
    A_dev_back = A_dev.to("cpu")
    B_dev_back = B_dev.to("cpu")

    zA = enc(A_dev)
    zB = enc(B_dev)
    rec_pair = dec(torch.cat([zA, zB], dim=0)).to("cpu")
    A_rec, B_rec = rec_pair.chunk(2, dim=0)

    cols = [
        ("A raw", A),
        ("A recon", A_rec),
        ("A gpu→cpu", A_dev_back),
        ("B raw", B),
        ("B recon", B_rec),
        ("B gpu→cpu", B_dev_back),
    ]

    tile_w, tile_h = W, H
    canvas = Image.new("RGB", (tile_w * len(cols), tile_h * Bsz), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    for r in range(Bsz):
        y = r * tile_h
        for c, (label, stack) in enumerate(cols):
            x = c * tile_w
            canvas.paste(_to_pil(stack[r]), (x, y))
            draw.rectangle([x + 2, y + 2, x + tile_w - 2, y + 26], outline=(255, 255, 0))
            draw.text((x + 6, y + 6), label, fill=(255, 255, 0))

    canvas.save(out_path)


# --------------------------------------------------------------------------------------
# Training configuration
# --------------------------------------------------------------------------------------
@dataclass
class TrainCfg:
    data_root: Path
    out_dir: Path = Path("out.action_distance_recon")
    z_dim: int = 128
    batch_size: int = 32
    epochs: int = 5
    lr: float = 3e-4
    seed: int = 0
    num_workers: int = 4
    device: Optional[str] = None
    max_step_gap: int = 20
    allow_cross_traj: bool = False
    p_cross_traj: float = 0.0
    encoder_pretrained: bool = False
    freeze_backbone: bool = False
    use_foreach_optim: bool = True

    lambda_rec: float = 2.0
    lambda_latent_l2: float = 0.0
    lambda_zcal: float = 0.05
    reg_warmup_steps: int = 2000

    viz_every: int = 500
    interp_steps: int = 12
    log_every: int = 50
    simple_viz: bool = False


# --------------------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------------------
def pick_device(pref: Optional[str]) -> torch.device:
    if pref:
        dev = torch.device(pref)
        if dev.type == "cuda":
            raise ValueError("CUDA is not supported by this trainer; use 'cpu' or 'mps'.")
        return dev
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def kaiming_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu", mode="fan_in")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def train(cfg: TrainCfg):
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "checkpoints").mkdir(exist_ok=True, parents=True)
    (cfg.out_dir / "viz").mkdir(exist_ok=True, parents=True)

    ds_tr = PairFromTrajDataset(
        cfg.data_root,
        "train",
        0.95,
        cfg.seed,
        cfg.max_step_gap,
        cfg.allow_cross_traj,
        cfg.p_cross_traj,
    )
    ds_va = PairFromTrajDataset(
        cfg.data_root,
        "val",
        0.95,
        cfg.seed,
        cfg.max_step_gap,
        cfg.allow_cross_traj,
        cfg.p_cross_traj,
    )

    use_pin = False
    dl_tr = DataLoader(
        ds_tr,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=use_pin,
        drop_last=True,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=use_pin,
        drop_last=False,
    )

    print(f"[Device] {device} | [Data] train pairs≈{len(ds_tr)}  val pairs≈{len(ds_va)}")

    enc = Encoder(cfg.z_dim, pretrained=cfg.encoder_pretrained, freeze_backbone=cfg.freeze_backbone).to(device)
    dec = Decoder(cfg.z_dim).to(device)
    dec.apply(kaiming_init)

    optim_kwargs = {"lr": cfg.lr, "weight_decay": 1e-4}
    if cfg.use_foreach_optim:
        optim_kwargs["foreach"] = True
    try:
        opt = torch.optim.AdamW(
            [
                {"params": enc.parameters(), "lr": cfg.lr},
                {"params": dec.parameters(), "lr": cfg.lr},
            ],
            **optim_kwargs,
        )
    except TypeError:
        if optim_kwargs.pop("foreach", None) is not None:
            opt = torch.optim.AdamW(
                [
                    {"params": enc.parameters(), "lr": cfg.lr},
                    {"params": dec.parameters(), "lr": cfg.lr},
                ],
                **optim_kwargs,
            )
        else:
            raise

    amp_autocast = contextlib.nullcontext
    scaler = GradScaler(enabled=False)

    global_step = 0
    best_val = float("inf")

    win = 50
    q_lrec, q_llat, q_lzcal, q_ltot = deque(maxlen=win), deque(maxlen=win), deque(maxlen=win), deque(maxlen=win)
    q_psnrA, q_psnrB = deque(maxlen=win), deque(maxlen=win)
    q_genc, q_gdec = deque(maxlen=win), deque(maxlen=win)
    q_data_time, q_forward_time, q_backward_time = deque(maxlen=win), deque(maxlen=win), deque(maxlen=win)
    step_time_accum = 0.0
    step_time_count = 0

    for ep in range(1, cfg.epochs + 1):
        enc.train()
        dec.train()
        run_loss_rec = run_n = 0.0

        for A, B, pathsA, pathsB in dl_tr:
            step_start = time.perf_counter()
            need_viz = (cfg.viz_every > 0) and (global_step % cfg.viz_every == 0)
            A_cpu = A.detach().clone() if need_viz else None
            B_cpu = B.detach().clone() if need_viz else None

            data_start = time.perf_counter()
            A = to_float01(A, device)
            B = to_float01(B, device)
            data_time = time.perf_counter() - data_start

            forward_start = time.perf_counter()
            with amp_autocast():
                zA = enc(A)
                zB = enc(B)

                z_pair = torch.cat([zA, zB], dim=0)
                x_pair = dec(z_pair)
                xA_rec, xB_rec = x_pair.chunk(2, dim=0)

                loss_rec = F.l1_loss(xA_rec, A) + F.l1_loss(xB_rec, B)
                loss_lat = 0.5 * (zA.pow(2).mean() + zB.pow(2).mean())
                z_all = torch.cat([zA, zB], dim=0)
                loss_zcal = (z_all.std(dim=0) - 1.0).abs().mean()

                reg_on = 1.0 if global_step >= cfg.reg_warmup_steps else 0.0
                lat_w = reg_on * cfg.lambda_latent_l2
                zcl_w = reg_on * cfg.lambda_zcal

                loss = cfg.lambda_rec * loss_rec + lat_w * loss_lat + zcl_w * loss_zcal
            forward_time = time.perf_counter() - forward_start

            backward_start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            backward_time = time.perf_counter() - backward_start

            step_time = time.perf_counter() - step_start
            step_time_accum += step_time
            step_time_count += 1

            run_loss_rec += loss_rec.item() * A.shape[0]
            run_n += A.shape[0]

            with torch.no_grad():
                d = (zB - zA)
                dnorm = torch.linalg.norm(d, dim=1).mean().item()
                psnrA = _psnr_01(xA_rec, A).item()
                psnrB = _psnr_01(xB_rec, B).item()

                q_lrec.append(float(loss_rec.item()))
                q_llat.append(float(loss_lat.item()))
                q_lzcal.append(float(loss_zcal.item()))
                q_ltot.append(float(loss.item()))
                q_psnrA.append(psnrA)
                q_psnrB.append(psnrB)
                q_genc.append(_grad_norm(enc))
                q_gdec.append(_grad_norm(dec))
                q_data_time.append(data_time * 1000)
                q_forward_time.append(forward_time * 1000)
                q_backward_time.append(backward_time * 1000)

            if (cfg.log_every > 0) and (global_step % cfg.log_every == 0):
                avg = lambda q: (sum(q) / len(q)) if q else 0.0
                avg_step_time = (step_time_accum / step_time_count) if step_time_count else 0.0
                step_time_accum = 0.0
                step_time_count = 0
                throughput = (cfg.batch_size / avg_step_time) if avg_step_time > 0 else 0.0
                print(
                    f"ep {ep:02d} step {global_step:06d} | "
                    f"loss {avg(q_ltot):.4f} | "
                    f"Lrec {avg(q_lrec):.4f} Llat {avg(q_llat):.5f} Lzcal {avg(q_lzcal):.5f} | "
                    f"PSNR A {avg(q_psnrA):.2f}dB B {avg(q_psnrB):.2f}dB | "
                    f"∥g_enc∥ {avg(q_genc):.3e} ∥g_dec∥ {avg(q_gdec):.3e} | "
                    f"timing: data {avg(q_data_time):.1f}ms fwd {avg(q_forward_time):.1f}ms bwd {avg(q_backward_time):.1f}ms | "
                    f"step_time {avg_step_time * 1000:.1f} ms ({throughput:.1f} samples/s)"
                )

            if global_step % cfg.viz_every == 0:
                enc.eval()
                dec.eval()

                if cfg.simple_viz:
                    out_path = cfg.out_dir / "viz" / f"ep{ep:02d}_step{global_step:06d}_simple.png"
                    save_simple_debug_grid(
                        enc,
                        dec,
                        A_cpu if A_cpu is not None else A.detach().cpu(),
                        B_cpu if B_cpu is not None else B.detach().cpu(),
                        list(pathsA),
                        list(pathsB),
                        out_path,
                        device=device,
                    )

                out_path = cfg.out_dir / "viz" / f"ep{ep:02d}_step{global_step:06d}.png"
                save_full_interpolation_grid(
                    enc,
                    dec,
                    A_cpu if A_cpu is not None else A.detach().cpu(),
                    B_cpu if B_cpu is not None else B.detach().cpu(),
                    list(pathsA),
                    list(pathsB),
                    out_path,
                    device=device,
                    interp_steps=cfg.interp_steps,
                )
                enc.train()
                dec.train()

            global_step += 1

        tr_rec = run_loss_rec / max(1, run_n)
        print(f"[ep {ep:02d}] train: Lrec={tr_rec:.4f}")

        enc.eval()
        dec.eval()
        va_rec = va_n = 0.0
        with torch.no_grad():
            for A, B, _, _ in dl_va:
                A = to_float01(A, device, non_blocking=False)
                B = to_float01(B, device, non_blocking=False)
                zA = enc(A)
                zB = enc(B)
                x_pair = dec(torch.cat([zA, zB], dim=0))
                xA_rec, xB_rec = x_pair.chunk(2, dim=0)
                loss_rec = F.l1_loss(xA_rec, A, reduction="sum") + F.l1_loss(xB_rec, B, reduction="sum")
                va_rec += loss_rec.item()
                va_n += A.shape[0]
        va_rec /= max(1, va_n)
        print(f"[ep {ep:02d}]   val: Lrec={va_rec:.4f}")

        ckpt = {
            "epoch": ep,
            "enc": enc.state_dict(),
            "dec": dec.state_dict(),
            "val_rec": va_rec,
            "cfg": vars(cfg),
        }
        torch.save(ckpt, cfg.out_dir / "last.ckpt")
        if va_rec < best_val:
            best_val = va_rec
            torch.save(ckpt, cfg.out_dir / "best.ckpt")
            print(f"[ep {ep:02d}] saved best (val Lrec={best_val:.4f})")
        torch.save(ckpt, cfg.out_dir / "checkpoints" / f"ep{ep:02d}.ckpt")

    print("[done]")


# --------------------------------------------------------------------------------------
# Export (optional): encode pairs and write ||zB - zA|| to CSV
# --------------------------------------------------------------------------------------
@torch.no_grad()
def export_latent_distances(
    ckpt_path: Path,
    data_root: Path,
    n_pairs: int,
    out_csv: Path,
    device_str: Optional[str] = None,
):
    device = pick_device(device_str)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    z_dim = ckpt.get("cfg", {}).get("z_dim", 128)
    enc = Encoder(z_dim, pretrained=False).to(device)
    enc.load_state_dict(ckpt["enc"])
    enc.eval()
    ds = PairFromTrajDataset(data_root, "val", 0.95, 0, 20, False, 0.0)
    rng = random.Random(0)

    rows = [("path_a", "path_b", "latent_l2")]
    for _ in range(n_pairs):
        a, b, pa, pb = ds[rng.randrange(0, len(ds))]
        A = to_float01(a.unsqueeze(0), device, non_blocking=False)
        B = to_float01(b.unsqueeze(0), device, non_blocking=False)
        zA = enc(A)
        zB = enc(B)
        d = torch.linalg.norm(zB - zA, dim=1).item()
        rows.append((pa, pb, f"{d:.6f}"))

    import csv

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)
    print(f"[export] wrote {out_csv}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("out.action_distance_recon"))
    ap.add_argument("--z_dim", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--max_step_gap", type=int, default=20)
    ap.add_argument("--allow_cross_traj", action="store_true")
    ap.add_argument("--p_cross_traj", type=float, default=0.0)

    ap.add_argument("--lambda_rec", type=float, default=2.0)
    ap.add_argument("--lambda_latent_l2", type=float, default=0.0)
    ap.add_argument("--lambda_zcal", type=float, default=0.05)
    ap.add_argument("--reg_warmup_steps", type=int, default=2000)

    ap.add_argument("--viz_every", type=int, default=10)
    ap.add_argument("--interp_steps", type=int, default=12)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--simple_viz", action="store_true", help="emit minimal GPU debug grids instead of full interpolation viz")
    ap.add_argument(
        "--encoder_pretrained",
        dest="encoder_pretrained",
        action="store_true",
        help="(deprecated) no-op with lightweight encoder; kept for CLI compatibility",
    )
    ap.add_argument("--no_encoder_pretrained", dest="encoder_pretrained", action="store_false")
    ap.set_defaults(encoder_pretrained=False)
    ap.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="(deprecated) no-op with lightweight encoder; kept for CLI compatibility",
    )
    ap.add_argument(
        "--no_foreach_optim",
        dest="use_foreach_optim",
        action="store_false",
        help="disable foreach AdamW updates (use if PyTorch build lacks foreach support)",
    )
    ap.add_argument(
        "--foreach_optim",
        dest="use_foreach_optim",
        action="store_true",
        help="force-enable foreach AdamW updates",
    )
    ap.set_defaults(use_foreach_optim=True)

    ap.add_argument("--export_pairs", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = TrainCfg(
        data_root=args.data_root,
        out_dir=args.out_dir,
        z_dim=args.z_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        max_step_gap=args.max_step_gap,
        allow_cross_traj=args.allow_cross_traj,
        p_cross_traj=args.p_cross_traj,
        encoder_pretrained=args.encoder_pretrained,
        freeze_backbone=args.freeze_backbone,
        use_foreach_optim=args.use_foreach_optim,
        lambda_rec=args.lambda_rec,
        lambda_latent_l2=args.lambda_latent_l2,
        lambda_zcal=args.lambda_zcal,
        reg_warmup_steps=args.reg_warmup_steps,
        viz_every=args.viz_every,
        interp_steps=args.interp_steps,
        log_every=args.log_every,
        simple_viz=args.simple_viz,
    )
    train(cfg)

    if args.export_pairs > 0:
        export_latent_distances(
            ckpt_path=cfg.out_dir / "best.ckpt",
            data_root=cfg.data_root,
            n_pairs=args.export_pairs,
            out_csv=cfg.out_dir / "latent_distances.csv",
            device_str=cfg.device,
        )


if __name__ == "__main__":
    main()
