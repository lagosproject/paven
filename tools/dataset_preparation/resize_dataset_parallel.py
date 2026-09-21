"""
Parallel Dataset Resizing Utility
Resizes large image directories using multi-threading with persistent resume logs.
"""

import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm


def resize_image(input_path: str, target_size: tuple, log_file: str = None) -> bool:
    """Resize a single image in-place using high-quality Lanczos resampling."""
    try:
        with Image.open(input_path) as img:
            if img.size != target_size:
                resized = img.resize(target_size, Image.Resampling.LANCZOS)
                resized.save(input_path)
        if log_file:
            with open(log_file, "a") as f:
                f.write(f"{input_path}\n")
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def load_processed_set(log_file: str) -> set:
    if log_file and os.path.exists(log_file):
        with open(log_file, "r") as f:
            return set(f.read().splitlines())
    return set()


def process_dataset(directory: str, width: int, height: int, log_file: str = None, workers: int = None):
    target_size = (width, height)
    processed = load_processed_set(log_file)
    workers = workers or os.cpu_count() or 4

    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(root, file)
                if full_path not in processed:
                    image_paths.append(full_path)

    print(f"Found {len(image_paths)} images to resize to {width}x{height} using {workers} threads.")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(resize_image, p, target_size, log_file) for p in image_paths]
        for _ in tqdm(futures, total=len(futures), desc="Resizing"):
            pass

    print("Dataset resizing finished.")


def main():
    parser = argparse.ArgumentParser(description="Multithreaded image resizing utility.")
    parser.add_argument("--directory", "-d", type=str, required=True, help="Root directory containing images.")
    parser.add_argument("--width", type=int, default=384, help="Target width (default: 384).")
    parser.add_argument("--height", type=int, default=224, help="Target height (default: 224).")
    parser.add_argument("--workers", "-w", type=int, default=None, help="Number of worker threads.")
    parser.add_argument("--log", type=str, default="processed_images.log", help="Resume log file.")
    args = parser.parse_args()

    process_dataset(args.directory, args.width, args.height, args.log, args.workers)


if __name__ == "__main__":
    main()
