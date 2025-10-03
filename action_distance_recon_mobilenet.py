#!/usr/bin/env python3
"""
Autoencoder reconstruction script built around a pretrained MobileNetV3-Large
encoder. By default this script only runs the latent interpolation visualization
(no training), letting you inspect how a pretrained backbone embeds NES frames.

Optional fine-tuning of the decoder (and, if desired, the MobileNet backbone)
can be enabled via CLI flags, reusing the same trajectory pair dataset from
``action_distance_recon.py``. The visualization matches the original script's
A→B latent interpolation debug grid.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
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
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

# --------------------------------------------------------------------------------------
# Global image size (NES frames resized to this before encoding / decoding)
# --------------------------------------------------------------------------------------
H, W = 240, 224  # decoder target size (height, width)
MOBILENET_HW = 224  # mobilenet nominal input resolution


# --------------------------------------------------------------------------------------
# IO utils (copied from action_distance_recon with light tweaks)
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


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32, copy=True) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t


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

    def __len__(self) -> int:
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

    def __getitem__(self, idx: int):  # type: ignore[override]
        rng = self._get_worker_rng()
        if self.allow_cross_traj and (rng.random() < self.p_cross_traj):
            p1, p2 = self._sample_cross_traj_pair(rng)
        else:
            p1, p2 = self._sample_same_traj_pair(rng)
        a = load_frame_as_tensor(p1)
        b = load_frame_as_tensor(p2)
        return a, b, str(p1), str(p2)


# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_float01(t: torch.Tensor, device: torch.device, non_blocking: bool = True) -> torch.Tensor:
    if t.dtype != torch.uint8:
        return t.to(device=device, non_blocking=non_blocking)
    return t.to(device=device, non_blocking=non_blocking, dtype=torch.float32) / 255.0


def pick_device(pref: Optional[str]) -> torch.device:
    if pref:
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------------------
# Encoder: MobileNetV3-Large -> latent vector
# --------------------------------------------------------------------------------------
class MobileNetEncoder(nn.Module):
    def __init__(
        self,
        z_dim: int = 256,
        pretrained: bool = True,
        train_backbone: bool = False,
    ):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        feat_dim = backbone.classifier[0].in_features
        self.proj = nn.Linear(feat_dim, z_dim)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="relu")
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

        if weights is not None:
            mean = torch.tensor(weights.meta["mean"]).view(1, 3, 1, 1)
            std = torch.tensor(weights.meta["std"]).view(1, 3, 1, 1)
        else:
            mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        self.register_buffer("input_mean", mean, persistent=False)
        self.register_buffer("input_std", std, persistent=False)

        if not train_backbone:
            for p in self.features.parameters():
                p.requires_grad = False
            for p in self.avgpool.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (MOBILENET_HW, MOBILENET_HW):
            x = F.interpolate(x, size=(MOBILENET_HW, MOBILENET_HW), mode="bilinear", align_corners=False)
        x = (x - self.input_mean) / self.input_std
        feats = self.features(x)
        pooled = self.avgpool(feats)
        pooled = pooled.flatten(1)
        z = self.proj(pooled)
        return z


class Decoder(nn.Module):
    def __init__(self, z_dim: int = 256):
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


# --------------------------------------------------------------------------------------
# Visualization helpers
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu()
    if t.dtype == torch.uint8:
        arr = t.permute(1, 2, 0).contiguous().numpy()
    else:
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
    x01 = x01.clamp(0.0, 1.0)
    y01 = y01.clamp(0.0, 1.0)
    mse = F.mse_loss(x01, y01, reduction="mean").clamp_min(eps)
    return 10.0 * torch.log10(1.0 / mse)


@torch.no_grad()
def save_full_interpolation_grid(
    enc: nn.Module,
    dec: nn.Module,
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
        font = ImageFont.load_default()

    def paste(row: int, col: int, tensor: torch.Tensor):
        img = _to_pil(tensor)
        canvas.paste(img.resize((tile_w, tile_h), Image.NEAREST), (col * tile_w, row * tile_h))

    for r in range(Bsz):
        paste(r, 0, A[r])
        paste(r, 1, dec_zA[r])
        for k in range(K):
            paste(r, 2 + k, interp_decoded[k, r])
        paste(r, 2 + K, dec_zB[r])
        paste(r, 3 + K, B[r])

        y0 = r * tile_h + 4
        if font is not None:
            draw.text((4, y0), short_traj_state_label(pA[r]), (255, 255, 255), font=font)
            draw.text(((tiles_per_row - 1) * tile_w + 4, y0), short_traj_state_label(pB[r]), (255, 255, 255), font=font)

    canvas.save(out_path)


# --------------------------------------------------------------------------------------
# Training (optional)
# --------------------------------------------------------------------------------------
@dataclass
class TrainCfg:
    data_root: Path
    out_dir: Path
    z_dim: int
    batch_size: int
    epochs: int
    lr: float
    seed: int
    num_workers: int
    device: str
    max_step_gap: int
    allow_cross_traj: bool
    p_cross_traj: float
    pretrained_encoder: bool
    train_backbone: bool
    lambda_rec: float
    lambda_latent_l2: float
    lambda_zcal: float
    reg_warmup_steps: int
    viz_every: int
    interp_steps: int
    log_every: int
    simple_viz: bool


def build_dataloaders(cfg: TrainCfg) -> Tuple[DataLoader, DataLoader]:
    ds_tr = PairFromTrajDataset(
        cfg.data_root,
        split="train",
        train_frac=0.95,
        seed=cfg.seed,
        max_step_gap=cfg.max_step_gap,
        allow_cross_traj=cfg.allow_cross_traj,
        p_cross_traj=cfg.p_cross_traj,
    )
    ds_va = PairFromTrajDataset(
        cfg.data_root,
        split="val",
        train_frac=0.95,
        seed=cfg.seed + 1,
        max_step_gap=cfg.max_step_gap,
        allow_cross_traj=cfg.allow_cross_traj,
        p_cross_traj=cfg.p_cross_traj,
    )
    dl_tr = DataLoader(
        ds_tr,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=max(1, cfg.num_workers // 2),
        pin_memory=True,
        drop_last=False,
    )
    return dl_tr, dl_va


def train(
    cfg: TrainCfg,
    enc: Optional[MobileNetEncoder] = None,
    dec: Optional[Decoder] = None,
):
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    if enc is None:
        enc = MobileNetEncoder(
            z_dim=cfg.z_dim,
            pretrained=cfg.pretrained_encoder,
            train_backbone=cfg.train_backbone,
        )
    if dec is None:
        dec = Decoder(cfg.z_dim)
    enc = enc.to(device)
    dec = dec.to(device)

    params: List[nn.Parameter] = []
    for p in enc.parameters():
        if p.requires_grad:
            params.append(p)
    params += list(dec.parameters())

    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.95))
    scaler = GradScaler(enabled=(device.type == "cuda"))

    dl_tr, dl_va = build_dataloaders(cfg)

    amp_autocast = contextlib.nullcontext
    if device.type == "cuda":
        amp_autocast = torch.cuda.amp.autocast  # type: ignore[attr-defined]

    global_step = 0
    best_val = float("inf")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "viz").mkdir(exist_ok=True, parents=True)

    for ep in range(1, cfg.epochs + 1):
        enc.train()
        dec.train()
        run_loss_rec = 0.0
        run_n = 0

        q_lrec: List[float] = []
        q_llat: List[float] = []
        q_lzcal: List[float] = []
        q_ltot: List[float] = []
        q_psnrA: List[float] = []
        q_psnrB: List[float] = []

        for A, B, pathsA, pathsB in dl_tr:
            A_cpu = A.detach().clone() if cfg.viz_every and (global_step % cfg.viz_every == 0) else None
            B_cpu = B.detach().clone() if cfg.viz_every and (global_step % cfg.viz_every == 0) else None

            A = to_float01(A, device)
            B = to_float01(B, device)

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

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            run_loss_rec += loss_rec.item() * A.shape[0]
            run_n += A.shape[0]

            with torch.no_grad():
                q_lrec.append(float(loss_rec.item()))
                q_llat.append(float(loss_lat.item()))
                q_lzcal.append(float(loss_zcal.item()))
                q_ltot.append(float(loss.item()))
                q_psnrA.append(float(_psnr_01(xA_rec, A).item()))
                q_psnrB.append(float(_psnr_01(xB_rec, B).item()))

            if (cfg.log_every > 0) and (global_step % cfg.log_every == 0):
                avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
                print(
                    f"ep {ep:02d} step {global_step:06d} | "
                    f"loss {avg(q_ltot):.4f} | "
                    f"Lrec {avg(q_lrec):.4f} Llat {avg(q_llat):.5f} Lzcal {avg(q_lzcal):.5f} | "
                    f"PSNR A {avg(q_psnrA):.2f}dB B {avg(q_psnrB):.2f}dB"
                )

            if cfg.viz_every and (global_step % cfg.viz_every == 0):
                enc.eval()
                dec.eval()
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

    print("[done]")
    return enc, dec


# --------------------------------------------------------------------------------------
# Visualization-only entry point
# --------------------------------------------------------------------------------------
@torch.no_grad()
def run_visualization(
    enc: MobileNetEncoder,
    dec: Decoder,
    data_root: Path,
    out_dir: Path,
    device: torch.device,
    seed: int,
    interp_steps: int,
    batch_size: int,
    num_workers: int,
    allow_cross_traj: bool,
    p_cross_traj: float,
    max_step_gap: int,
):
    set_seed(seed)
    ds = PairFromTrajDataset(
        data_root,
        split="val",
        train_frac=0.95,
        seed=seed,
        max_step_gap=max_step_gap,
        allow_cross_traj=allow_cross_traj,
        p_cross_traj=p_cross_traj,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "viz").mkdir(parents=True, exist_ok=True)

    try:
        batch = next(iter(dl))
    except StopIteration:
        raise RuntimeError("Dataset yielded no samples – check data_root")

    A, B, pathsA, pathsB = batch
    enc.eval()
    dec.eval()
    out_path = out_dir / "viz" / "mobilenet_interpolation.png"
    save_full_interpolation_grid(
        enc,
        dec,
        A,
        B,
        list(pathsA),
        list(pathsB),
        out_path,
        device=device,
        interp_steps=interp_steps,
    )
    print(f"saved visualization to {out_path}")


def load_checkpoint(
    ckpt_path: Optional[Path],
    enc: MobileNetEncoder,
    dec: Decoder,
) -> None:
    if ckpt_path is None:
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing_enc, unexpected_enc = enc.load_state_dict(ckpt.get("enc", {}), strict=False)
    if missing_enc or unexpected_enc:
        print(f"[warn] encoder state mismatch. missing={missing_enc}, unexpected={unexpected_enc}")
    missing_dec, unexpected_dec = dec.load_state_dict(ckpt.get("dec", {}), strict=False)
    if missing_dec or unexpected_dec:
        print(f"[warn] decoder state mismatch. missing={missing_dec}, unexpected={unexpected_dec}")
    print(f"loaded checkpoint from {ckpt_path}")


# --------------------------------------------------------------------------------------
# Argument parsing & CLI entry
# --------------------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", type=Path, default=Path("traj_dumps"))
    ap.add_argument("--out_dir", type=Path, default=Path("out.action_distance_recon_mobilenet"))
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--interp_steps", type=int, default=12)
    ap.add_argument("--max_step_gap", type=int, default=20)
    ap.add_argument("--allow_cross_traj", action="store_true")
    ap.add_argument("--p_cross_traj", type=float, default=0.0)

    ap.add_argument("--z_dim", type=int, default=256)
    ap.add_argument("--no_pretrained", action="store_true", help="disable ImageNet weights for MobileNet encoder")
    ap.add_argument("--train_backbone", action="store_true", help="fine-tune MobileNet weights during training")

    ap.add_argument("--train_epochs", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_rec", type=float, default=2.0)
    ap.add_argument("--lambda_latent_l2", type=float, default=0.0)
    ap.add_argument("--lambda_zcal", type=float, default=0.05)
    ap.add_argument("--reg_warmup_steps", type=int, default=2000)
    ap.add_argument("--viz_every", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--simple_viz", action="store_true")  # kept for CLI parity, unused here but forwarded to TrainCfg

    ap.add_argument("--checkpoint", type=Path, default=None, help="optional checkpoint with encoder/decoder weights")

    return ap.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)

    enc = MobileNetEncoder(
        z_dim=args.z_dim,
        pretrained=not args.no_pretrained,
        train_backbone=args.train_backbone and (args.train_epochs > 0),
    ).to(device)
    dec = Decoder(args.z_dim).to(device)
    load_checkpoint(args.checkpoint, enc, dec)

    if args.train_epochs > 0:
        cfg = TrainCfg(
            data_root=args.data_root,
            out_dir=args.out_dir,
            z_dim=args.z_dim,
            batch_size=args.batch_size,
            epochs=args.train_epochs,
            lr=args.lr,
            seed=args.seed,
            num_workers=args.num_workers,
            device=device.type,
            max_step_gap=args.max_step_gap,
            allow_cross_traj=args.allow_cross_traj,
            p_cross_traj=args.p_cross_traj,
            pretrained_encoder=not args.no_pretrained,
            train_backbone=args.train_backbone,
            lambda_rec=args.lambda_rec,
            lambda_latent_l2=args.lambda_latent_l2,
            lambda_zcal=args.lambda_zcal,
            reg_warmup_steps=args.reg_warmup_steps,
            viz_every=args.viz_every,
            interp_steps=args.interp_steps,
            log_every=args.log_every,
            simple_viz=args.simple_viz,
        )
        enc, dec = train(cfg, enc, dec)

    run_visualization(
        enc,
        dec,
        data_root=args.data_root,
        out_dir=args.out_dir,
        device=device,
        seed=args.seed,
        interp_steps=args.interp_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        allow_cross_traj=args.allow_cross_traj,
        p_cross_traj=args.p_cross_traj,
        max_step_gap=args.max_step_gap,
    )


if __name__ == "__main__":
    main()
