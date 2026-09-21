"""
Saliency Heatmap Video Overlay Utility
Superimposes predicted or ground-truth saliency heatmaps on video frames using a JET colormap.
"""

import os
import re
import argparse
import cv2
import numpy as np


def numerical_sort(value: str) -> int:
    numbers = re.findall(r"\d+", value)
    return int(numbers[0]) if numbers else 0


def create_heatmap_overlay(
    frames_dir: str,
    saliency_dir: str,
    output_video: str,
    fps: float = 30.0,
    alpha: float = 0.5,
):
    """Overlay saliency heatmaps (JET colormap) onto corresponding video frames."""
    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".png"))],
        key=numerical_sort,
    )
    sal_files = sorted(
        [f for f in os.listdir(saliency_dir) if f.lower().endswith((".jpg", ".png"))],
        key=numerical_sort,
    )

    if not frame_files or not sal_files:
        raise ValueError("No frames or saliency maps found in provided directories.")

    num_frames = min(len(frame_files), len(sal_files))
    sample_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    h, w, _ = sample_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    print(f"Creating overlay video: {output_video} ({num_frames} frames @ {fps} fps, resolution: {w}x{h})...")

    for i in range(num_frames):
        frame = cv2.imread(os.path.join(frames_dir, frame_files[i]))
        sal_map = cv2.imread(os.path.join(saliency_dir, sal_files[i]), cv2.IMREAD_GRAYSCALE)

        if sal_map.shape[:2] != (h, w):
            sal_map = cv2.resize(sal_map, (w, h))

        heatmap = cv2.applyColorMap(sal_map, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(heatmap, alpha, frame, 1.0 - alpha, 0)
        out.write(overlay)

    out.release()
    print(f"Overlay video saved successfully: {output_video}")


def main():
    parser = argparse.ArgumentParser(description="Create saliency overlay video.")
    parser.add_argument("--frames", "-f", type=str, required=True, help="Directory with RGB frame images.")
    parser.add_argument("--saliency", "-s", type=str, required=True, help="Directory with grayscale saliency images.")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output video file (.mp4).")
    parser.add_argument("--fps", type=float, default=30.0, help="Framerate of output video.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Saliency blend weight (0.0 to 1.0).")
    args = parser.parse_args()

    create_heatmap_overlay(args.frames, args.saliency, args.output, args.fps, args.alpha)


if __name__ == "__main__":
    main()
