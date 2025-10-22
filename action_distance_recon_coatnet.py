#!/usr/bin/env python3
"""CoAtNet-inspired action-distance reconstruction trainer."""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from recon import (
    H,
    W,
    PairFromTrajDataset,
    TileSpec,
    load_frame_as_tensor as base_load_frame_as_tensor,
    render_image_grid,
    set_seed,
    short_traj_state_label,
    to_float01,
)
from recon.utils import psnr_01, tensor_to_pil


def _normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    return t


def _denormalize_tensor(t: torch.Tensor) -> torch.Tensor:
    return t


def load_frame_as_tensor(path: Path) -> torch.Tensor:
    return base_load_frame_as_tensor(path, normalize=_normalize_tensor)


class MBConv(nn.Module):
    """Mobile inverted bottleneck convolution with squeeze-excitation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 4,
        se_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        hidden_dim = int(in_channels * expansion)
        self.use_residual = stride == 1 and in_channels == out_channels

        if expansion != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
            )
        else:
            hidden_dim = in_channels
            self.expand = None

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )

        squeeze_channels = max(1, int(hidden_dim * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, squeeze_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(squeeze_channels, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
        )

        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.expand is not None:
            x = self.expand(x)
        x = self.depthwise(x)
        x = x * self.se(x)
        x = self.project(x)
        if self.use_residual:
            x = x + identity
        return x


class MBConvStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int, stride: int) -> None:
        super().__init__()
        blocks = []
        for i in range(depth):
            blk_stride = stride if i == 0 else 1
            blk_in = in_channels if i == 0 else out_channels
            blocks.append(MBConv(blk_in, out_channels, stride=blk_stride))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class AttentionBlock2d(nn.Module):
    """Multi-head self-attention over spatial tokens with MLP."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        attn_in = self.norm1(seq)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        seq = seq + attn_out
        seq = seq + self.mlp(self.norm2(seq))
        x = seq.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x


class AttentionStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        heads: int,
        stride: int,
    ) -> None:
        super().__init__()
        if stride > 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            )
        else:
            self.downsample = None
        self.blocks = nn.ModuleList([AttentionBlock2d(out_channels, heads) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class CoAtNetEncoder(nn.Module):
    def __init__(self, z_dim: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )
        self.stage1 = MBConvStage(64, 64, depth=1, stride=1)
        self.stage2 = MBConvStage(64, 128, depth=2, stride=2)
        self.stage3 = AttentionStage(128, 192, depth=2, heads=4, stride=2)
        self.stage4 = AttentionStage(192, 256, depth=2, heads=8, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, z_dim)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class Decoder(nn.Module):
    """Lightweight decoder mirroring the encoder downsampling factor."""

    def __init__(self, z_dim: int = 64) -> None:
        super().__init__()
        self.fc = nn.Linear(z_dim, 256 * 15 * 14)
        self.pre = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 192, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 256, 15, 14)
        h = self.pre(h)
        x = self.net(h)
        return torch.sigmoid(x)


def _to_pil(t: torch.Tensor) -> Image.Image:
    return tensor_to_pil(t, denormalize=_denormalize_tensor)


def _psnr_01(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return psnr_01(_denormalize_tensor(x), _denormalize_tensor(y), eps)


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = (gauss / gauss.sum()).unsqueeze(0)
    window_2d = (gauss.t() @ gauss).unsqueeze(0).unsqueeze(0)
    return window_2d.to(dtype=dtype).expand(channels, 1, window_size, window_size).contiguous()


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

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / denominator
    cs_map = (2 * sigma_xy + c2) / (sigma_x_sq + sigma_y_sq + c2)

    ssim_val = ssim_map.mean(dim=(1, 2, 3))
    cs_val = cs_map.mean(dim=(1, 2, 3))
    return ssim_val, cs_val


def ms_ssim(x: torch.Tensor, y: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weights is None:
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=x.device, dtype=x.dtype)
    window_size = 11
    sigma = 1.5
    channels = x.shape[1]
    window = _gaussian_window(window_size, sigma, channels, x.device, x.dtype)
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


@torch.no_grad()
def save_full_interpolation_grid(
    enc: CoAtNetEncoder,
    dec: Decoder,
    A: torch.Tensor,
    B: torch.Tensor,
    pathsA: List[str],
    pathsB: List[str],
    out_path: Path,
    device: torch.device,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bsz = min(A.shape[0], 8)

    A = A[:bsz].contiguous()
    B = B[:bsz].contiguous()
    pA = pathsA[:bsz]
    pB = pathsB[:bsz]

    A_dev = to_float01(A, device, non_blocking=False)
    B_dev = to_float01(B, device, non_blocking=False)
    zA = enc(A_dev)
    zB = enc(B_dev)
    pair_recon = dec(torch.cat([zA, zB], dim=0))
    dec_zA, dec_zB = pair_recon.chunk(2, dim=0)

    t_vals = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], device=device)
    mid_t_vals = t_vals[1:-1]
    if mid_t_vals.numel() > 0:
        z_interp = torch.stack([(1.0 - t) * zA + t * zB for t in mid_t_vals], dim=0)
        interp_decoded = dec(z_interp.view(-1, zA.shape[1])).view(mid_t_vals.numel(), bsz, 3, H, W)
    else:
        interp_decoded = torch.empty(0, bsz, 3, H, W, device=device)

    z_norm = torch.linalg.norm(zB - zA, dim=1).cpu()
    mid_t_vals_cpu = mid_t_vals.detach().cpu().tolist()

    rows: List[List[TileSpec]] = []
    for idx in range(bsz):
        labelA = short_traj_state_label(pA[idx])
        labelB = short_traj_state_label(pB[idx])

        row: List[TileSpec] = [
            TileSpec(
                image=_to_pil(A[idx]),
                top_label=labelA,
                bottom_label=f"‖zB−zA‖={z_norm[idx]:.2f}",
                bottom_color=(255, 255, 0),
            ),
            TileSpec(
                image=_to_pil(dec_zA[idx]),
                top_label="t=0.0 (A)",
                top_color=(220, 220, 255),
            ),
        ]

        for interp_idx, t_val in enumerate(mid_t_vals_cpu):
            row.append(
                TileSpec(
                    image=_to_pil(interp_decoded[interp_idx, idx]),
                    top_label=f"t={t_val:.1f}",
                    top_color=(200, 255, 200),
                )
            )

        row.extend(
            [
                TileSpec(
                    image=_to_pil(dec_zB[idx]),
                    top_label="t=1.0 (B)",
                    top_color=(220, 220, 255),
                ),
                TileSpec(
                    image=_to_pil(B[idx]),
                    top_label=labelB,
                ),
            ]
        )

        rows.append(row)

    render_image_grid(rows, out_path, tile_size=(W, H))


@torch.no_grad()
def save_simple_debug_grid(
    enc: CoAtNetEncoder,
    dec: Decoder,
    A: torch.Tensor,
    B: torch.Tensor,
    pathsA: List[str],
    pathsB: List[str],
    out_path: Path,
    device: torch.device,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bsz = min(A.shape[0], 4)
    A = A[:bsz].contiguous()
    B = B[:bsz].contiguous()

    A_dev = to_float01(A, device, non_blocking=False)
    B_dev = to_float01(B, device, non_blocking=False)

    zA = enc(A_dev)
    zB = enc(B_dev)
    rec_pair = dec(torch.cat([zA, zB], dim=0)).to("cpu")
    A_rec, B_rec = rec_pair.chunk(2, dim=0)

    cols = [
        ("A raw", A),
        ("A recon", A_rec),
        ("A gpu→cpu", A_dev.to("cpu")),
        ("B raw", B),
        ("B recon", B_rec),
        ("B gpu→cpu", B_dev.to("cpu")),
    ]

    tile_w, tile_h = W, H
    canvas = Image.new("RGB", (tile_w * len(cols), tile_h * bsz), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    for r in range(bsz):
        y = r * tile_h
        for c, (label, stack) in enumerate(cols):
            x = c * tile_w
            canvas.paste(_to_pil(stack[r]), (x, y))
            draw.line([(x, y), (x, y + tile_h - 1)], fill=(255, 255, 255))
            draw.line([(x + tile_w - 1, y), (x + tile_w - 1, y + tile_h - 1)], fill=(0, 0, 0))
            draw.rectangle([x + 2, y + 2, x + tile_w - 2, y + 26], outline=(255, 255, 0))
            draw.text((x + 6, y + 6), label, fill=(255, 255, 0))

    canvas.save(out_path)


@dataclass
class TrainCfg:
    data_root: Path
    out_dir: Path = Path("out.action_distance_recon_coatnet")
    z_dim: int = 64
    batch_size: int = 8
    epochs: int = 20
    lr: float = 1e-3
    seed: int = 0
    num_workers: int = 2
    device: Optional[str] = None
    max_step_gap: int = 10
    viz_every: int = 200
    log_every: int = 50
    simple_viz: bool = False
    lambda_l1: float = 0.5
    lambda_ms_ssim: float = 0.5


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
        False,
        0.0,
        load_frame=load_frame_as_tensor,
    )
    ds_va = PairFromTrajDataset(
        cfg.data_root,
        "val",
        0.95,
        cfg.seed,
        cfg.max_step_gap,
        False,
        0.0,
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

    print(f"[Device] {device} | [Data] train pairs≈{len(ds_tr)}  val pairs≈{len(ds_va)}")

    enc = CoAtNetEncoder(cfg.z_dim).to(device)
    dec = Decoder(cfg.z_dim).to(device)
    enc.apply(kaiming_init)
    dec.apply(kaiming_init)

    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=cfg.lr)

    global_step = 0
    best_val = float("inf")

    win = 50
    q_loss = deque(maxlen=win)
    q_l1 = deque(maxlen=win)
    q_ms = deque(maxlen=win)
    q_psnrA, q_psnrB = deque(maxlen=win), deque(maxlen=win)
    q_step_ms = deque(maxlen=win)
    start_time = time.monotonic()

    for ep in range(1, cfg.epochs + 1):
        enc.train()
        dec.train()
        run_loss_rec = run_n = 0.0

        for A, B, pathsA, pathsB in dl_tr:
            step_start = time.perf_counter()
            need_viz = (cfg.viz_every > 0) and (global_step % cfg.viz_every == 0)
            A_cpu = A.detach().clone() if need_viz else None
            B_cpu = B.detach().clone() if need_viz else None

            A = to_float01(A, device)
            B = to_float01(B, device)
            zA = enc(A)
            zB = enc(B)

            xA_rec = dec(zA)
            xB_rec = dec(zB)
            loss_l1 = F.l1_loss(xA_rec, A) + F.l1_loss(xB_rec, B)
            loss_ms = ms_ssim_loss(xA_rec, A) + ms_ssim_loss(xB_rec, B)
            loss = cfg.lambda_l1 * loss_l1 + cfg.lambda_ms_ssim * loss_ms

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            step_time = time.perf_counter() - step_start

            with torch.no_grad():
                psnrA = _psnr_01(xA_rec, A).item()
                psnrB = _psnr_01(xB_rec, B).item()

            batch_mean = 0.5 * loss_l1.item()
            run_loss_rec += batch_mean * A.shape[0]
            run_n += A.shape[0]

            q_loss.append(float(loss.item()))
            q_l1.append(float(loss_l1.item()))
            q_ms.append(float(loss_ms.item()))
            q_psnrA.append(psnrA)
            q_psnrB.append(psnrB)
            q_step_ms.append(step_time * 1000.0)

            if (cfg.log_every > 0) and (global_step % cfg.log_every == 0):
                avg_loss = (sum(q_loss) / len(q_loss)) if q_loss else 0.0
                avg_l1 = (sum(q_l1) / len(q_l1)) if q_l1 else 0.0
                avg_ms = (sum(q_ms) / len(q_ms)) if q_ms else 0.0
                avg_psnrA = (sum(q_psnrA) / len(q_psnrA)) if q_psnrA else 0.0
                avg_psnrB = (sum(q_psnrB) / len(q_psnrB)) if q_psnrB else 0.0
                avg_step_ms = (sum(q_step_ms) / len(q_step_ms)) if q_step_ms else 0.0
                throughput = (cfg.batch_size / (avg_step_ms / 1000.0)) if avg_step_ms > 0 else 0.0
                elapsed = int(time.monotonic() - start_time)
                h = elapsed // 3600
                m = (elapsed % 3600) // 60
                s = elapsed % 60
                print(
                    f"ep {ep:02d} step {global_step:06d} | "
                    f"loss {avg_loss:.4f} | "
                    f"L1 {avg_l1:.4f} MS-SSIM {avg_ms:.4f} | "
                    f"PSNR A {avg_psnrA:.2f}dB B {avg_psnrB:.2f}dB | "
                    f"step {avg_step_ms:.1f} ms ({throughput:.1f} samples/s) | "
                    f"elapsed {h:02d}:{m:02d}:{s:02d}"
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
                xA_rec = dec(zA)
                xB_rec = dec(zB)
                batch_mean = 0.5 * (F.l1_loss(xA_rec, A).item() + F.l1_loss(xB_rec, B).item())
                va_rec += batch_mean * A.shape[0]
                va_n += A.shape[0]
        va_rec = va_rec / max(1, va_n)
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("out.action_distance_recon_coatnet"))
    ap.add_argument("--z_dim", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--max_step_gap", type=int, default=10)
    ap.add_argument("--viz_every", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument(
        "--simple_viz",
        action="store_true",
        help="emit a smaller debug grid instead of the full interpolation layout",
    )
    ap.add_argument("--lambda_l1", type=float, default=0.5)
    ap.add_argument("--lambda_ms_ssim", type=float, default=0.5)
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
        viz_every=args.viz_every,
        log_every=args.log_every,
        simple_viz=args.simple_viz,
        lambda_l1=args.lambda_l1,
        lambda_ms_ssim=args.lambda_ms_ssim,
    )
    train(cfg)


if __name__ == "__main__":
    main()
