import sys
from pathlib import Path
import numpy as np
from PIL import Image

def convert_pngs_to_npz(input_dir, output_npz):
    input_dir = Path(input_dir)
    png_paths = sorted(input_dir.glob('*.png'))
    if not png_paths:
        print("No PNG files found.")
        return

    imgs = []
    for path in png_paths:
        with Image.open(path) as img:
            imgs.append(np.array(img))
    imgs = np.stack(imgs)
    np.savez_compressed(output_npz, imgs=imgs)
    print(f"Saved {len(imgs)} images to {output_npz}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pngs_to_npz.py <input_png_dir> <output_file.npz>")
        sys.exit(1)
    convert_pngs_to_npz(sys.argv[1], sys.argv[2])