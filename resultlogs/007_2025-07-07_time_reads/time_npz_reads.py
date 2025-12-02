import time
from pathlib import Path
import numpy as np

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
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_npz_dir>")
        exit(1)

    input_dir = sys.argv[1]

    time_individual_npz_read(input_dir)
