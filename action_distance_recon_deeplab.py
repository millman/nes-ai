#!/usr/bin/env python3
"""
Action-distance visualization using a pretrained DeepLabV3 segmentation
backbone (ResNet-50 by default). Instead of training a custom autoencoder,
this script reuses the encoder and decoder head from the ImageNet/COCO
pretrained model to embed NES frames and decode the latent features back into
semantic maps. The same latent interpolation grid from ``action_distance_recon``
is emitted to make it easy to compare trajectories.

Examples
--------
Visualize the first few batches without any training:

    python action_distance_recon_deeplab.py \
        --data_root traj_dumps \
        --out_dir out.deeplab_viz \
        --viz_batches 4

Switch to the lighter MobileNetV3 backbone (faster, still good quality):

    python action_distance_recon_deeplab.py \
        --data_root traj_dumps \
        --model deeplabv3_mobilenet_v3_large

The script only performs inference; no weights are updated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    DeepLabV3_ResNet50_Weights,
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
)

from recon import (
    H,
    W,
    PairFromTrajDataset,
    load_frame_as_tensor as base_load_frame_as_tensor,
    set_seed,
    short_traj_state_label,
    to_float01,
)
from recon.utils import tensor_to_pil


# --------------------------------------------------------------------------------------
# Data loading helpers (reuse default tensor loader but allow override)
# --------------------------------------------------------------------------------------
def load_frame_as_tensor(path: Path) -> torch.Tensor:
    return base_load_frame_as_tensor(path)


# --------------------------------------------------------------------------------------
# DeepLab feature autoencoder wrapper
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BackboneOption:
    builder: Callable[..., nn.Module]
    weights: object  # torchvision WeightsEnum (kept generic for typing simplicity)
    description: str


BACKBONES: Dict[str, BackboneOption] = {
    "deeplabv3_resnet50": BackboneOption(
        builder=deeplabv3_resnet50,
        weights=DeepLabV3_ResNet50_Weights.DEFAULT,
        description="ResNet-50 encoder (higher accuracy)",
    ),
    "deeplabv3_mobilenet_v3_large": BackboneOption(
        builder=deeplabv3_mobilenet_v3_large,
        weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
        description="MobileNetV3 encoder (lighter, still solid quality)",
    ),
}


def _build_color_palette(num_classes: int) -> torch.Tensor:
    palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1], dtype=torch.long)
    colors = torch.stack([(i * palette) % 255 for i in range(num_classes)], dim=0)
    return colors.to(torch.float32).div(255.0)


class DeepLabAutoencoder(nn.Module):
    """Freeze a pretrained DeepLab network and expose encode/decode helpers."""

    def __init__(
        self,
        *,
        output_size: tuple[int, int] = (H, W),
        model_factory: Callable[..., nn.Module] = deeplabv3_resnet50,
        weights: Optional[object] = None,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        if weights is None:
            weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = model_factory(weights=weights)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.backbone = model.backbone
        self.classifier = model.classifier
        meta = getattr(weights, "meta", {})
        mean = torch.tensor(meta.get("mean", (0.485, 0.456, 0.406)), dtype=torch.float32)
        std = torch.tensor(meta.get("std", (0.229, 0.224, 0.225)), dtype=torch.float32)
        self.register_buffer("mean", mean.view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", std.view(1, 3, 1, 1), persistent=False)
        categories = meta.get("categories")
        if categories:
            num_classes = len(categories)
        else:
            num_classes = getattr(self.classifier[-1], "out_channels", 21)
        num_classes = int(num_classes)
        palette = _build_color_palette(num_classes)
        self.register_buffer("palette", palette, persistent=False)

    def encode(self, x01: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Return flattened latent and its spatial shape for decoding."""
        x = (x01 - self.mean) / self.std
        feats = self.backbone(x)["out"]
        latent = feats.flatten(1)
        spatial_shape = feats.shape[1:]
        return latent, (spatial_shape[0], spatial_shape[1], spatial_shape[2])

    def decode(self, latent: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
        """Map latent back to RGB semantic visualization (0-1 range)."""
        if latent.dim() == 2:
            feats = latent.view(-1, *spatial_shape)
        elif latent.dim() == 4:
            feats = latent
        else:
            raise ValueError(f"Unexpected latent shape: {latent.shape}")
        logits = self.classifier(feats)
        logits = F.interpolate(logits, size=self.output_size, mode="bilinear", align_corners=False)
        mask = logits.argmax(dim=1)
        palette = self.palette.to(mask.device)
        rgb = palette[mask].permute(0, 3, 1, 2)
        return rgb.clamp(0.0, 1.0)


# --------------------------------------------------------------------------------------
# Visualization (copied from action_distance_recon with minor tweaks)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def save_full_interpolation_grid(
    autoenc: DeepLabAutoencoder,
    A: torch.Tensor,
    B: torch.Tensor,
    pathsA: List[str],
    pathsB: List[str],
    out_path: Path,
    device: torch.device,
    interp_steps: int = 6,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    batch = min(A.shape[0], 8)
    if batch == 0:
        return

    A = A[:batch].contiguous()
    B = B[:batch].contiguous()
    pA = pathsA[:batch]
    pB = pathsB[:batch]

    A_dev = to_float01(A, device, non_blocking=False)
    B_dev = to_float01(B, device, non_blocking=False)
    zA, spatial = autoenc.encode(A_dev)
    zB, _ = autoenc.encode(B_dev)

    decA = autoenc.decode(zA, spatial).cpu()
    decB = autoenc.decode(zB, spatial).cpu()

    ts = torch.linspace(0.0, 1.0, interp_steps + 2, device=device)[1:-1]
    if ts.numel() > 0:
        interp = torch.stack([(1.0 - t) * zA + t * zB for t in ts], dim=0)
        interp_dec = autoenc.decode(interp.view(-1, zA.shape[1]), spatial)
        interp_dec = interp_dec.view(ts.numel(), batch, 3, H, W).cpu()
    else:
        interp_dec = torch.empty(0, batch, 3, H, W)

    tiles_per_row = 4 + ts.numel()  # A raw | dec(A) | interps | dec(B) | B raw
    tile_w, tile_h = W, H
    canvas = Image.new("RGB", (tile_w * tiles_per_row, tile_h * batch), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=18)
    except Exception:
        font = ImageFont.load_default()

    def paste(col: int, row: int, tensor: torch.Tensor) -> None:
        img = tensor_to_pil(tensor)
        canvas.paste(img, (col * tile_w, row * tile_h))

    for r in range(batch):
        paste(0, r, A[r])
        paste(1, r, decA[r])
        for k in range(ts.numel()):
            paste(2 + k, r, interp_dec[k, r])
        paste(2 + ts.numel(), r, decB[r])
        paste(3 + ts.numel(), r, B[r])

        label_y = r * tile_h + 6
        draw.text((4, label_y), short_traj_state_label(pA[r]), fill=(255, 255, 255), font=font)
        draw.text(((tiles_per_row - 1) * tile_w + 4, label_y), short_traj_state_label(pB[r]), fill=(255, 255, 255), font=font)

    canvas.save(out_path)


# --------------------------------------------------------------------------------------
# Config & main loop
# --------------------------------------------------------------------------------------
@dataclass
class VizCfg:
    data_root: Path
    out_dir: Path
    model: str = "deeplabv3_resnet50"
    batch_size: int = 8
    num_workers: int = 2
    device: Optional[str] = None
    seed: int = 0
    max_step_gap: int = 10
    viz_batches: int = 1
    interp_steps: int = 6


def pick_device(pref: Optional[str]) -> torch.device:
    if pref:
        dev = torch.device(pref)
        if dev.type == "cuda":
            raise ValueError("CUDA is not supported; please use 'cpu' or 'mps'.")
        return dev
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run(cfg: VizCfg) -> None:
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "viz").mkdir(parents=True, exist_ok=True)

    if cfg.model not in BACKBONES:
        available = ", ".join(sorted(BACKBONES))
        raise ValueError(f"Unknown model '{cfg.model}'. Available options: {available}")
    backbone = BACKBONES[cfg.model]
    print(f"[model] Using {cfg.model}: {backbone.description}")

    ds = PairFromTrajDataset(
        cfg.data_root,
        split="val",
        train_frac=0.95,
        seed=cfg.seed,
        max_step_gap=cfg.max_step_gap,
        allow_cross_traj=False,
        load_frame=load_frame_as_tensor,
    )
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    autoenc = DeepLabAutoencoder(
        output_size=(H, W),
        model_factory=backbone.builder,
        weights=backbone.weights,
    ).to(device)
    autoenc.eval()

    any_batch = False
    for batch_idx, (A, B, pathsA, pathsB) in enumerate(dl):
        if cfg.viz_batches > 0 and batch_idx >= cfg.viz_batches:
            break
        any_batch = True
        out_path = cfg.out_dir / "viz" / f"deeplab_interpolation_{batch_idx:04d}.png"
        save_full_interpolation_grid(autoenc, A, B, list(pathsA), list(pathsB), out_path, device, cfg.interp_steps)
        print(f"[viz] saved {out_path}")

    if not any_batch:
        print("No batches visualized – check data_root or increase viz_batches.")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args() -> VizCfg:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("out.action_distance_recon_deeplab"))
    ap.add_argument("--model", type=str, choices=sorted(BACKBONES.keys()), default="deeplabv3_resnet50",
                    help="Which pretrained DeepLab variant to use (lighter MobileNet vs higher quality ResNet).")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_step_gap", type=int, default=10)
    ap.add_argument("--viz_batches", type=int, default=1)
    ap.add_argument("--interp_steps", type=int, default=6)
    return VizCfg(**vars(ap.parse_args()))


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
