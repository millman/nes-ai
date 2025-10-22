import time
from pathlib import Path
import cv2

def time_cv2_png_load(directory):
    png_files = list(Path(directory).glob("*.png"))
    n = len(png_files)
    if n == 0:
        print("No PNG files found.")
        return

    start = time.perf_counter()
    for file in png_files:
        img = cv2.imread(str(file))  # str(file) ensures compatibility
        if img is None:
            print(f"Warning: {file} could not be read.")
    elapsed = time.perf_counter() - start

    print(f"Read {n} PNGs in {elapsed:.3f} seconds")
    print(f"Rate: {n / elapsed:.2f} PNGs per second")

if __name__ == "__main__":
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    time_cv2_png_load(directory)
