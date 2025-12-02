import time
from pathlib import Path
import numpy as np
from PIL import Image
import tqdm

def convert_pngs_to_individual_npz(directory, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    paths = sorted(Path(directory).glob("*.png"))
    if not paths:
        print("No PNG files found!")
        return

    for path in tqdm.tqdm(paths):
        with Image.open(path) as img:
            arr = np.array(img)
        out_path = out_dir / (path.stem + ".npz")
        np.savez_compressed(out_path, img=arr)
    print(f"Converted {len(paths)} images to individual npz files in {out_dir}")

def time_individual_npz_read(npz_dir):
    npz_files = sorted(Path(npz_dir).glob("*.npz"))
    if not npz_files:
        print("No npz files found!")
        return

    start = time.perf_counter()
    imgs = []
    for f in npz_files:
        with np.load(f) as data:
            imgs.append(data["img"])
    elapsed = time.perf_counter() - start
    n = len(imgs)
    print(f"Loaded {n} images from {npz_dir} in {elapsed:.3f} seconds")
    print(f"Rate: {n / elapsed:.2f} images per second")
    return imgs

if __name__ == "__main__":
    import sys
    # Usage: python script.py input_png_dir output_npz_dir
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_png_dir> <output_npz_dir>")
        exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    convert_pngs_to_individual_npz(input_dir, output_dir)
    time_individual_npz_read(output_dir)
