#!/usr/bin/env python3
"""
Action-distance visualization using pretrained autoencoders.
The script streams NES frame pairs through a selectable pretrained autoencoder,
interpolates in latent space, and renders the same A→B debug grid as
``action_distance_recon`` without any additional training.

Examples
--------
Visualize four batches with the default ViT-MAE Large weights:

    python action_distance_recon_deeplab.py \
        --data_root traj_dumps \
        --out_dir out.action_distance_recon_mae \
        --viz_batches 4

Switch to the lighter ViT-MAE Small variant:

    python action_distance_recon_deeplab.py \
        --data_root traj_dumps \
        --model_name facebook/vit-mae-small

Try a diffusion VAE (AutoencoderKL) fine-tuned for reconstruction:

    python action_distance_recon_deeplab.py \
        --data_root traj_dumps \
        --model_family autoencoderkl \
        --model_name stabilityai/sd-vae-ft-ema

Use the OpenCLIP VAE refinement:

    python action_distance_recon_deeplab.py \
        --data_root traj_dumps \
        --model_family openclip-vae \
        --out_dir out.action_distance_recon_openclip

The script only runs inference; no weights are updated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import ViTMAEForPreTraining

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
from recon.utils import tensor_to_pil


MODEL_DEFAULT = "facebook/vit-mae-large"
MODEL_DEFAULTS = {
    "vit-mae": MODEL_DEFAULT,
    "autoencoderkl": "stabilityai/sd-vae-ft-ema",
    "openclip-vae": "stabilityai/sd-vae-ft-mse",
}


def load_frame_as_tensor(path: Path) -> torch.Tensor:
    return base_load_frame_as_tensor(path)


@dataclass
class EncodedLatents:
    latent: torch.Tensor
    context: Dict[str, Any]


class BaseAutoencoder(nn.Module):
    """Abstract autoencoder interface providing encode/decode helpers."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_dummy", torch.empty(0), persistent=False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters(), self._dummy).device

    @torch.no_grad()
    def encode(self, x01: torch.Tensor) -> EncodedLatents:
        raise NotImplementedError

    @torch.no_grad()
    def decode(self, latent_flat: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    def expand_context(self, context: Dict[str, Any], batch_size: int) -> Dict[str, Any]:
        base_batch = context.get("base_batch")
        if base_batch is None or base_batch == batch_size:
            return context
        if batch_size % base_batch != 0:
            raise ValueError(f"Cannot expand context of batch {base_batch} to {batch_size}")
        repeat = batch_size // base_batch
        expanded: Dict[str, Any] = dict(context)
        for key, value in context.items():
            if isinstance(value, torch.Tensor) and value.shape and value.shape[0] == base_batch:
                shape_tail = value.shape[1:]
                tiled = value.unsqueeze(0).repeat(repeat, *([1] * value.dim()))
                expanded[key] = tiled.reshape(batch_size, *shape_tail)
        expanded["base_batch"] = batch_size
        return expanded


class ViTMAEAutoencoder(BaseAutoencoder):
    """Wrapper that exposes encode/decode for pretrained ViT-MAE checkpoints."""

    def __init__(self, model_name: str, device: torch.device) -> None:
        super().__init__()
        self.model = ViTMAEForPreTraining.from_pretrained(model_name)
        self.model.config.mask_ratio = 0.0  # keep all tokens during inference
        self.image_size = (
            self.model.config.image_size
            if isinstance(self.model.config.image_size, int)
            else self.model.config.image_size[0]
        )
        self.num_patches = self.model.vit.embeddings.num_patches
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)
        self.to(device)
        self.eval()

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return x

    @torch.no_grad()
    def encode(self, x01: torch.Tensor) -> EncodedLatents:
        x = self._resize(x01)
        x = (x - self.mean) / self.std
        noise = torch.zeros(x.size(0), self.num_patches, device=x.device)
        outputs = self.model.vit(pixel_values=x, noise=noise, interpolate_pos_encoding=True)
        latent = outputs.last_hidden_state  # (B, seq_len, hidden)
        ids_restore = outputs.ids_restore  # (B, seq_len_without_cls)
        seq_shape = (latent.shape[1], latent.shape[2])
        context: Dict[str, Any] = {
            "ids_restore": ids_restore,
            "seq_shape": seq_shape,
            "base_batch": latent.size(0),
        }
        return EncodedLatents(latent.view(latent.size(0), -1), context)

    @torch.no_grad()
    def decode(self, latent_flat: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        batch = latent_flat.shape[0]
        seq_shape = context["seq_shape"]
        seq_len, hidden_dim = seq_shape
        ids_restore = context["ids_restore"]
        if ids_restore.shape[0] != batch:
            raise ValueError(f"ids_restore batch {ids_restore.shape[0]} != latent batch {batch}")
        latent = latent_flat.view(batch, seq_len, hidden_dim)
        decoder_outputs = self.model.decoder(latent, ids_restore, interpolate_pos_encoding=True)
        logits = decoder_outputs.logits
        recon = self.model.unpatchify(logits, (self.image_size, self.image_size))
        recon = recon * self.std + self.mean
        return recon.clamp(0.0, 1.0)


class DiffusersAutoencoderKL(BaseAutoencoder):
    """Wrapper exposing encode/decode for diffusion VAEs (AutoencoderKL)."""

    def __init__(self, model_name: str, device: torch.device, *, subfolder: Optional[str] = None) -> None:
        super().__init__()
        try:
            from diffusers import AutoencoderKL
        except ImportError as exc:  # pragma: no cover - informative error for missing dep
            raise ImportError(
                "The diffusers package is required for AutoencoderKL models. Install via 'pip install diffusers'."
            ) from exc

        load_kwargs = {"subfolder": subfolder} if subfolder else {}
        try:
            self.model = AutoencoderKL.from_pretrained(model_name, **load_kwargs)
        except OSError as err:
            if subfolder is None:
                try:
                    self.model = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
                except Exception:
                    raise err
            else:
                raise
        self.scaling_factor = getattr(self.model.config, "scaling_factor", getattr(self.model, "scaling_factor", 0.18215))
        self.to(device)
        self.eval()

    def _maybe_resize(self, x: torch.Tensor) -> torch.Tensor:
        # AutoencoderKL expects spatial dims divisible by 8; inputs already satisfy this.
        return x

    @torch.no_grad()
    def encode(self, x01: torch.Tensor) -> EncodedLatents:
        x = self._maybe_resize(x01)
        x = x * 2.0 - 1.0
        posterior = self.model.encode(x)
        latents = posterior.latent_dist.mode() * self.scaling_factor
        context: Dict[str, Any] = {
            "latent_shape": latents.shape[1:],
            "scaling_factor": self.scaling_factor,
            "base_batch": latents.size(0),
        }
        return EncodedLatents(latents.view(latents.size(0), -1), context)

    @torch.no_grad()
    def decode(self, latent_flat: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        latent_shape = context["latent_shape"]
        scaling = context["scaling_factor"]
        latents = latent_flat.view(-1, *latent_shape) / scaling
        recon = self.model.decode(latents).sample
        return recon.add(1.0).div(2.0).clamp(0.0, 1.0)

@torch.no_grad()
def save_full_interpolation_grid(
    autoenc: ViTMAEAutoencoder,
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
    encA = autoenc.encode(A_dev)
    encB = autoenc.encode(B_dev)

    decA = autoenc.decode(encA.latent, encA.context).cpu()
    decB = autoenc.decode(encB.latent, encB.context).cpu()
    dec_h, dec_w = decA.shape[-2], decA.shape[-1]

    ts = torch.linspace(0.0, 1.0, interp_steps + 2, device=device)[1:-1]
    if ts.numel() > 0:
        interp = torch.stack([(1.0 - t) * encA.latent + t * encB.latent for t in ts], dim=0)
        interp = interp.view(-1, encA.latent.shape[1])
        interp_ctx = autoenc.expand_context(encA.context, interp.shape[0])
        interp_dec = autoenc.decode(interp, interp_ctx)
        interp_dec = interp_dec.view(ts.numel(), batch, 3, dec_h, dec_w).cpu()
    else:
        interp_dec = torch.empty(0, batch, 3, dec_h, dec_w)

    def resize_for_display(t: torch.Tensor) -> torch.Tensor:
        if t.shape[-2:] != (H, W):
            t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
        return t

    decA = resize_for_display(decA)
    decB = resize_for_display(decB)
    if interp_dec.numel() > 0:
        interp_dec = resize_for_display(interp_dec.view(-1, 3, dec_h, dec_w)).view(ts.numel(), batch, 3, H, W)

    z_norm = torch.linalg.norm(encB.latent - encA.latent, dim=1).cpu()
    ts_cpu = ts.detach().cpu().tolist()

    rows: List[List[TileSpec]] = []
    for idx in range(batch):
        row: List[TileSpec] = [
            TileSpec(
                image=tensor_to_pil(A[idx]),
                top_label=short_traj_state_label(pA[idx]),
            ),
            TileSpec(
                image=tensor_to_pil(decA[idx]),
                top_label="t=0.0 (A)",
                top_color=(220, 220, 255),
            ),
        ]

        for interp_idx, t_val in enumerate(ts_cpu):
            row.append(
                TileSpec(
                    image=tensor_to_pil(interp_dec[interp_idx, idx]),
                    top_label=f"t={t_val:.2f}",
                    top_color=(200, 255, 200),
                )
            )

        row.extend(
            [
                TileSpec(
                    image=tensor_to_pil(decB[idx]),
                    top_label="t=1.0 (B)",
                    top_color=(220, 220, 255),
                ),
                TileSpec(
                    image=tensor_to_pil(B[idx]),
                    top_label=short_traj_state_label(pB[idx]),
                    bottom_label=f"‖Δz‖={z_norm[idx]:.2f}",
                    bottom_color=(255, 255, 0),
                ),
            ]
        )

        rows.append(row)

    render_image_grid(rows, out_path, tile_size=(W, H))


@dataclass
class VizCfg:
    data_root: Path
    out_dir: Path
    model_family: str = "vit-mae"
    model_name: str = MODEL_DEFAULT
    model_subfolder: Optional[str] = None
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


def build_autoencoder(cfg: VizCfg, device: torch.device) -> BaseAutoencoder:
    family = cfg.model_family
    if family == "vit-mae":
        return ViTMAEAutoencoder(cfg.model_name, device)
    if family in {"autoencoderkl", "openclip-vae"}:
        return DiffusersAutoencoderKL(cfg.model_name, device, subfolder=cfg.model_subfolder)
    raise ValueError(f"Unsupported model_family '{family}'")


def run(cfg: VizCfg) -> None:
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "viz").mkdir(parents=True, exist_ok=True)

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

    print(f"[model] Loading {cfg.model_family}: {cfg.model_name}")
    autoenc = build_autoencoder(cfg, device)

    any_batch = False
    for batch_idx, (A, B, pathsA, pathsB) in enumerate(dl):
        if cfg.viz_batches > 0 and batch_idx >= cfg.viz_batches:
            break
        any_batch = True
        prefix = cfg.model_family.replace("-", "_")
        out_path = cfg.out_dir / "viz" / f"{prefix}_interpolation_{batch_idx:04d}.png"
        save_full_interpolation_grid(autoenc, A, B, list(pathsA), list(pathsB), out_path, device, cfg.interp_steps)
        print(f"[viz] saved {out_path}")

    if not any_batch:
        print("No batches visualized – check data_root or increase viz_batches.")


def parse_args() -> VizCfg:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("out.action_distance_recon_mae"))
    ap.add_argument(
        "--model_family",
        type=str,
        default="vit-mae",
        choices=tuple(sorted(MODEL_DEFAULTS.keys())),
        help="Which pretrained autoencoder to use (vit-mae, autoencoderkl, openclip-vae).",
    )
    ap.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name or path of the pretrained checkpoint for the selected family.",
    )
    ap.add_argument(
        "--model_subfolder",
        type=str,
        default=None,
        help="Optional subfolder when loading diffusers AutoencoderKL checkpoints (e.g. 'vae').",
    )
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_step_gap", type=int, default=10)
    ap.add_argument("--viz_batches", type=int, default=10)
    ap.add_argument("--interp_steps", type=int, default=6)
    args = ap.parse_args()
    if args.model_name is None:
        args.model_name = MODEL_DEFAULTS[args.model_family]
    return VizCfg(**vars(args))


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
