#!/usr/bin/env python3
"""
Quick sanity check for action_distance_cfm image IO.
Loads a handful of raw frames, runs them through the encoder/decoder,
then writes side-by-side grids comparing:
  1) original raw PNG (nearest resized to 224x240)
  2) tensor after preprocessing (converted back to [0,1])
  3) reconstructed output from current checkpoint (if provided)

Usage:
  python check_load_vs_recon.py --data_root <path> [--ckpt <checkpoint>] [--out out_dir]
"""

import argparse
from pathlib import Path

import torch
from torchvision.utils import make_grid
import numpy as np
from PIL import Image

from action_distance_cfm import Encoder, Decoder, load_frame_as_tensor, _to_pil


def tensor_to_image(t: torch.Tensor) -> Image.Image:
    return _to_pil(t.cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("out.debug_images"))
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--count", type=int, default=8, help="number of frames to sample")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cpu")

    pngs = sorted(args.data_root.rglob("state_*.png"))
    if not pngs:
        raise SystemExit(f"No state_*.png found under {args.data_root}")
    pngs = pngs[: args.count]
    args.out.mkdir(parents=True, exist_ok=True)

    tensors = torch.stack([load_frame_as_tensor(p) for p in pngs])  # [-1,1]
    grid_loaded = make_grid(tensors * 0.5 + 0.5, nrow=min(len(tensors), 4))
    tensor_to_image(grid_loaded).save(args.out / "loaded_frames.png")

    pil_imgs = []
    for p in pngs:
        with Image.open(p) as img:
            img = img.convert("RGB").resize((224, 240), Image.NEAREST)
            pil_imgs.append(torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1))
    pil_stack = torch.stack(pil_imgs).float() / 255.0
    grid_orig = make_grid(pil_stack, nrow=min(len(pil_stack), 4))
    Image.fromarray((grid_orig.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8")).save(
        args.out / "original_frames.png"
    )

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        z_dim = ckpt.get("cfg", {}).get("z_dim", 128)
        enc = Encoder(z_dim).to(device)
        dec = Decoder(z_dim).to(device)
        enc.load_state_dict(ckpt["enc"])
        dec.load_state_dict(ckpt["dec"])
        enc.eval(); dec.eval()

        with torch.no_grad():
            z = enc(tensors.to(device))
            rec = dec(z).cpu()

        grid_rec = make_grid(rec * 0.5 + 0.5, nrow=min(len(rec), 4))
        tensor_to_image(grid_rec).save(args.out / "recon_frames.png")

    print(f"Saved comparison grids to {args.out}")


if __name__ == "__main__":
    main()
