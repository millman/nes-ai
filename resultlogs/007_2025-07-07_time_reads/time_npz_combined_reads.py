import sys
import time
import numpy as np

def time_npz_read(npz_file):
    start = time.perf_counter()
    with np.load(npz_file) as data:
        imgs = data['imgs']
    elapsed = time.perf_counter() - start
    n = imgs.shape[0]
    print(f"Loaded {n} images from {npz_file} in {elapsed:.3f} seconds")
    print(f"Rate: {n / elapsed:.2f} images per second")
    return imgs

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python time_npz_load.py <images.npz>")
        sys.exit(1)
    time_npz_read(sys.argv[1])
