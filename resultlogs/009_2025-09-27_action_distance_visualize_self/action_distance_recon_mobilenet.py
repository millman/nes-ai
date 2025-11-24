#!/usr/bin/env python3
"""
MobileNetV3-based latent interpolation demo for NES frame trajectories.
This script loads a pretrained MobileNet encoder and lightweight decoder, then
renders the A→B interpolation grid without any training steps.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, get_worker_info
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from recon.visualize import TileSpec, render_image_grid

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
    with Image.open(p) as img:
        img = img.convert("RGB").resize((W, H), resample=Image.NEAREST)
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
        dev = torch.device(pref)
        if dev.type == "cuda":
            raise ValueError("CUDA is not supported by this trainer; use 'cpu' or 'mps'.")
        return dev
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------------------
# Encoder: MobileNetV3-Large -> latent vector
# --------------------------------------------------------------------------------------


def _resolve_imagenet_norm(weights: Optional[MobileNet_V3_Large_Weights]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return broadcastable mean/std buffers even on old torchvision builds."""
    default_mean = (0.485, 0.456, 0.406)
    default_std = (0.229, 0.224, 0.225)

    mean: Optional[Tuple[float, float, float]] = None
    std: Optional[Tuple[float, float, float]] = None

    meta = getattr(weights, "meta", None)
    if isinstance(meta, dict):
        mean = tuple(meta.get("mean", ())) or None
        std = tuple(meta.get("std", ())) or None

    if (mean is None or std is None) and weights is not None:
        try:
            transforms = weights.transforms()  # type: ignore[operator]
        except Exception:
            transforms = None
        if transforms is not None:
            mean_attr = getattr(transforms, "mean", None)
            std_attr = getattr(transforms, "std", None)
            if isinstance(mean_attr, (list, tuple)) and len(mean_attr) == 3:
                mean = tuple(float(x) for x in mean_attr)
            if isinstance(std_attr, (list, tuple)) and len(std_attr) == 3:
                std = tuple(float(x) for x in std_attr)

    if mean is None:
        mean = default_mean
    if std is None:
        std = default_std

    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return mean_t, std_t


class MobileNetEncoder(nn.Module):
    def __init__(
        self,
        z_dim: int = 256,
        pretrained: bool = True,
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

        mean, std = _resolve_imagenet_norm(weights)
        self.register_buffer("input_mean", mean, persistent=False)
        self.register_buffer("input_std", std, persistent=False)

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

    steps = max(0, int(interp_steps))
    linspace = torch.linspace(0.0, 1.0, steps + 2, device=device)
    mid_t_vals = linspace[1:-1]
    if mid_t_vals.numel() > 0:
        z_interp = torch.stack([(1.0 - t) * zA + t * zB for t in mid_t_vals], dim=0)
        interp_decoded = dec(z_interp.view(-1, zA.shape[1])).view(mid_t_vals.numel(), Bsz, 3, H, W)
    else:
        interp_decoded = torch.empty(0, Bsz, 3, H, W, device=device)

    z_norm = torch.linalg.norm(zB - zA, dim=1).cpu()
    mid_t_vals_cpu = mid_t_vals.detach().cpu().tolist()

    rows: List[List[TileSpec]] = []
    for idx in range(Bsz):
        row: List[TileSpec] = [
            TileSpec(
                image=_to_pil(A[idx]),
                top_label=short_traj_state_label(pA[idx]),
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
                    top_label=f"t={t_val:.2f}",
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
                    top_label=short_traj_state_label(pB[idx]),
                    bottom_label=f"‖zB−zA‖={z_norm[idx]:.2f}",
                    bottom_color=(255, 255, 0),
                ),
            ]
        )

        rows.append(row)

    render_image_grid(rows, out_path, tile_size=(W, H))



# --------------------------------------------------------------------------------------
# Checkpoint loading
# --------------------------------------------------------------------------------------

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
# Visualization config & entry points
# --------------------------------------------------------------------------------------


@dataclass
class VizCfg:
    data_root: Path
    out_dir: Path
    batch_size: int
    num_workers: int
    device: Optional[str]
    seed: int
    max_step_gap: int
    allow_cross_traj: bool
    p_cross_traj: float
    interp_steps: int
    viz_batches: int
    z_dim: int
    pretrained: bool
    checkpoint: Optional[Path]


def run(cfg: VizCfg) -> None:
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "viz").mkdir(parents=True, exist_ok=True)

    enc = MobileNetEncoder(z_dim=cfg.z_dim, pretrained=cfg.pretrained).to(device)
    dec = Decoder(cfg.z_dim).to(device)
    load_checkpoint(cfg.checkpoint, enc, dec)
    enc.eval()
    dec.eval()

    ds = PairFromTrajDataset(
        cfg.data_root,
        split="val",
        train_frac=0.95,
        seed=cfg.seed,
        max_step_gap=cfg.max_step_gap,
        allow_cross_traj=cfg.allow_cross_traj,
        p_cross_traj=cfg.p_cross_traj,
    )
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    any_batch = False
    for batch_idx, (A, B, pathsA, pathsB) in enumerate(dl):
        if cfg.viz_batches > 0 and batch_idx >= cfg.viz_batches:
            break
        any_batch = True
        out_path = cfg.out_dir / "viz" / f"mobilenet_interpolation_{batch_idx:04d}.png"
        save_full_interpolation_grid(
            enc,
            dec,
            A,
            B,
            list(pathsA),
            list(pathsB),
            out_path,
            device=device,
            interp_steps=cfg.interp_steps,
        )
        print(f"[viz] saved {out_path}")

    if not any_batch:
        print("No batches visualized – check data_root or increase viz_batches.")


def parse_args() -> VizCfg:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("out.action_distance_recon_mobilenet"))
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_step_gap", type=int, default=20)
    ap.add_argument("--allow_cross_traj", action="store_true")
    ap.add_argument("--p_cross_traj", type=float, default=0.0)
    ap.add_argument("--interp_steps", type=int, default=12)
    ap.add_argument("--viz_batches", type=int, default=10)
    ap.add_argument("--z_dim", type=int, default=256)
    ap.add_argument(
        "--no_pretrained",
        action="store_true",
        help="disable ImageNet weights for the MobileNet encoder",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="optional checkpoint providing encoder/decoder weights",
    )
    args = ap.parse_args()
    return VizCfg(
        data_root=args.data_root,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        max_step_gap=args.max_step_gap,
        allow_cross_traj=args.allow_cross_traj,
        p_cross_traj=args.p_cross_traj,
        interp_steps=args.interp_steps,
        viz_batches=args.viz_batches,
        z_dim=args.z_dim,
        pretrained=not args.no_pretrained,
        checkpoint=args.checkpoint,
    )


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
