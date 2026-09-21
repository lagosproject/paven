"""
PAVEN: Continuous Saliency Ground Truth Map Generator via 2D Gaussian Kernel Density Estimation (KDE)
Paper: "PAVEN: A Perceptual Algorithm for Versatile video Encoding using Neural networks"
       (Elsevier EAAI, 2025, DOI: 10.1016/j.engappai.2025.111664)
Author: Pablo Fernández Lagos (UPM Master Thesis, Chapter 4, 2025)

Converts discrete eye-tracking fixation matrices into continuous, smoothed visual saliency maps
using parallel multiprocessing and adaptive Gaussian standard deviation according to visual angle degrees.
"""

import os
import argparse
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def generate_saliency_map_from_fixations(
    fixation_coords: np.ndarray,
    shape: tuple = (1080, 1920),
    sigma_pixels: float = 30.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Generate a 2D continuous saliency density map from discrete gaze fixation coordinates.

    Args:
        fixation_coords: (N, 2) array of (x, y) fixation locations on the image frame.
        shape: Output resolution (height, width).
        sigma_pixels: Standard deviation for Gaussian kernel (approx. 1 degree of visual angle).
        normalize: If True, normalizes peak density to 1.0.

    Returns:
        saliency_map: 2D float32 numpy array with continuous saliency distribution.
    """
    h, w = shape
    fixation_map = np.zeros((h, w), dtype=np.float32)

    for pt in fixation_coords:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h:
            fixation_map[y, x] += 1.0

    # Apply 2D Gaussian filtering
    saliency_map = gaussian_filter(fixation_map, sigma=sigma_pixels)

    if normalize and saliency_map.max() > 0:
        saliency_map = saliency_map / saliency_map.max()

    return saliency_map


def process_video_fixation_folder(args_tuple):
    video_dir, out_dir, shape, sigma = args_tuple
    os.makedirs(out_dir, exist_ok=True)
    fixation_files = sorted([f for f in os.listdir(video_dir) if f.endswith(('.png', '.npy', '.mat'))])

    for f_name in fixation_files:
        f_path = os.path.join(video_dir, f_name)
        out_path = os.path.join(out_dir, os.path.splitext(f_name)[0] + '.png')
        if os.path.exists(out_path):
            continue

        if f_name.endswith('.npy'):
            coords = np.load(f_path)
            s_map = generate_saliency_map_from_fixations(coords, shape=shape, sigma_pixels=sigma)
        else:
            # Binary fixation image
            f_img = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
            if f_img is None:
                continue
            s_map = gaussian_filter(f_img.astype(np.float32), sigma=sigma)
            if s_map.max() > 0:
                s_map = s_map / s_map.max()

        s_map_uint8 = (s_map * 255.0).astype(np.uint8)
        cv2.imwrite(out_path, s_map_uint8)


def main():
    parser = argparse.ArgumentParser(description="PAVEN: Multiprocessing Eye-Fixation to Saliency Map Generator")
    parser.add_argument("--input-dir", "-i", type=str, required=True, help="Directory containing raw fixation records per video")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="Destination directory for continuous saliency maps")
    parser.add_argument("--width", "-w", type=int, default=1920, help="Frame width in pixels (default: 1920)")
    parser.add_argument("--height", "-H", type=int, default=1080, help="Frame height in pixels (default: 1080)")
    parser.add_argument("--sigma", "-s", type=float, default=30.0, help="Gaussian sigma in pixels (default: 30.0, ~1 deg visual angle)")
    parser.add_argument("--workers", "-j", type=int, default=min(8, cpu_count()), help="Number of parallel worker processes")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    video_dirs = [os.path.join(args.input_dir, d) for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]

    if not video_dirs:
        print(f"[PAVEN] No video subdirectories found in: {args.input_dir}")
        return

    tasks = [
        (v_dir, os.path.join(args.output_dir, os.path.basename(v_dir)), (args.height, args.width), args.sigma)
        for v_dir in video_dirs
    ]

    print(f"[PAVEN] Processing {len(tasks)} video sequences with {args.workers} worker processes...")
    with Pool(processes=args.workers) as pool:
        list(tqdm(pool.imap_unordered(process_video_fixation_folder, tasks), total=len(tasks), desc="Generating Saliency Maps"))

    print("[PAVEN] Saliency ground-truth generation complete.")


if __name__ == "__main__":
    main()
