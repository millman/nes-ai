import time
from pathlib import Path
from PIL import Image

def main(directory):
    png_files = list(Path(directory).glob('*.png'))
    n = len(png_files)
    if n == 0:
        print("No PNG files found.")
        return

    start = time.perf_counter()
    for file in png_files:
        with Image.open(file) as img:
            img.load()  # Actually load the image data
    elapsed = time.perf_counter() - start

    print(f"Read {n} PNGs in {elapsed:.3f} seconds")
    print(f"Rate: {n / elapsed:.2f} PNGs per second")

if __name__ == "__main__":
    import sys
    # directory = sys.argv[1] if len(sys.argv) > 1 else "."

    directory = "/Users/dave/rl/nes-ai/traj_dumps/smb-search-v0__search_mario__1__2025-06-25_18-03-35/level_1-1_2563_end/states"

    main(directory)