"""
Multi-Panel Video Comparison Visualizer
Generates a synchronized comparison video showing:
- Original Video Frame
- Ground-Truth Saliency Map & Overlay
- PAVEN Predicted Saliency Map & Overlay
- Coincidence / Overlap Map
- Delta-QP Grid Matrix Visualization
Based on the visualization suite developed for the EAAI 2025 paper and Master Thesis.
"""

import os
import re
import argparse
import cv2
import numpy as np


def numerical_sort(value: str) -> int:
    nums = re.findall(r"\d+", value)
    return int(nums[0]) if nums else 0


def load_folder_images(folder: str, size: tuple) -> list:
    if not os.path.exists(folder):
        return []
    files = sorted(os.listdir(folder), key=numerical_sort)
    images = []
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            img = cv2.imread(os.path.join(folder, f))
            if img is not None:
                images.append(cv2.resize(img, size))
    return images


def apply_gradual_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    norm_mask = cv2.normalize(mask.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
    norm_frame = frame.astype(np.float32) / 255.0
    result = norm_frame * norm_mask[:, :, np.newaxis]
    return (result * 255).astype(np.uint8)


def create_coincidence_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY) if len(gt.shape) == 3 else gt
    pred_gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY) if len(pred.shape) == 3 else pred

    _, gt_bin = cv2.threshold(gt_gray, 127, 255, cv2.THRESH_BINARY)
    _, pred_bin = cv2.threshold(pred_gray, 127, 255, cv2.THRESH_BINARY)

    cmap = np.zeros((*gt_gray.shape, 3), dtype=np.uint8)
    cmap[(gt_bin == 255) & (pred_bin == 0)] = [255, 0, 255]    # Magenta: False Negative
    cmap[(gt_bin == 0) & (pred_bin == 255)] = [0, 255, 255]    # Yellow: False Positive
    cmap[(gt_bin == 255) & (pred_bin == 255)] = [0, 255, 0]    # Green: True Positive
    return cmap


def generate_comparison_video(
    rgb_folder: str,
    gt_folder: str,
    pred_folder: str,
    grid_folder: str,
    output_video: str,
    sequence_title: str = "PAVEN Comparison",
    fps: float = 25.0,
):
    tile_w, tile_h = 1280, 720
    canvas_w, canvas_h = tile_w * 3, tile_h * 3  # 3840 x 2160 (4K UHD)

    print(f"Loading video frames from: {rgb_folder}")
    frames = load_folder_images(rgb_folder, (tile_w, tile_h))
    print(f"Loading ground truth from: {gt_folder}")
    gt_saliency = load_folder_images(gt_folder, (tile_w, tile_h))
    print(f"Loading predictions from: {pred_folder}")
    pred_saliency = load_folder_images(pred_folder, (tile_w, tile_h))
    print(f"Loading QP grids from: {grid_folder}")
    grid_images = load_folder_images(grid_folder, (tile_w, tile_h))

    available = [len(x) for x in [frames, gt_saliency, pred_saliency, grid_images] if len(x) > 0]
    if not available:
        raise ValueError("No images found across provided folders.")
    num_frames = min(available)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (canvas_w, canvas_h))
    print(f"Generating {num_frames} frames into {output_video}...")

    for i in range(num_frames):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        f = frames[i] if i < len(frames) else np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        gt = gt_saliency[i] if i < len(gt_saliency) else np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        pred = pred_saliency[i] if i < len(pred_saliency) else np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        grid = grid_images[i] if i < len(grid_images) else np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

        gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        pred_gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)

        masked_gt = apply_gradual_mask(f, gt_gray)
        masked_pred = apply_gradual_mask(f, pred_gray)
        coincidence = create_coincidence_map(gt, pred)

        # Row 0: Original | Masked GT | Masked Prediction
        canvas[0:tile_h, 0:tile_w] = f
        canvas[0:tile_h, tile_w:tile_w*2] = masked_gt
        canvas[0:tile_h, tile_w*2:canvas_w] = masked_pred

        # Row 1: Coincidence Map | GT Heatmap | Predicted Heatmap
        canvas[tile_h:tile_h*2, 0:tile_w] = coincidence
        canvas[tile_h:tile_h*2, tile_w:tile_w*2] = gt
        canvas[tile_h:tile_h*2, tile_w*2:canvas_w] = pred

        # Row 2: QP Grid | Empty / Overlay | Legend
        canvas[tile_h*2:canvas_h, 0:tile_w] = grid

        # Frame overlay text
        label = f"{sequence_title} - Frame: {i:04d}"
        cv2.putText(canvas, label, (30, canvas_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3, cv2.LINE_AA)

        out.write(canvas)

    out.release()
    print(f"Comparison video generated successfully: {output_video}")


def main():
    parser = argparse.ArgumentParser(description="Multi-panel visualizer for PAVEN.")
    parser.add_argument("--rgb", type=str, required=True, help="Folder containing RGB frames.")
    parser.add_argument("--gt", type=str, required=True, help="Folder containing ground truth saliency maps.")
    parser.add_argument("--pred", type=str, required=True, help="Folder containing PAVEN predicted saliency maps.")
    parser.add_argument("--grid", type=str, required=True, help="Folder containing visualized QP grids.")
    parser.add_argument("--output", "-o", type=str, default="paven_comparison.mp4", help="Output video path.")
    parser.add_argument("--title", type=str, default="PAVEN Perceptual Analysis", help="Title label.")
    parser.add_argument("--fps", type=float, default=25.0, help="Framerate.")
    args = parser.parse_args()

    generate_comparison_video(args.rgb, args.gt, args.pred, args.grid, args.output, args.title, args.fps)


if __name__ == "__main__":
    main()
