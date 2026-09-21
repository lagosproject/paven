"""
PAVEN: Planar YUV Channel Splitter & YYY Replicator
Paper: "PAVEN: A Perceptual Algorithm for Versatile video Encoding using Neural networks"
       (Elsevier EAAI, 2025, DOI: 10.1016/j.engappai.2025.111664)

Extracts individual Y, U, V channels from raw planar YUV420 video streams
and produces replicated YYY streams for convolutional neural network compatibility.
"""

import os
import argparse
import numpy as np


def split_channels(yuv_path: str, width: int, height: int, out_dir: str, num_frames: int = -1):
    os.makedirs(out_dir, exist_ok=True)
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    frame_size = y_size + 2 * uv_size

    total_frames = os.path.getsize(yuv_path) // frame_size
    if num_frames > 0:
        total_frames = min(total_frames, num_frames)

    base = os.path.splitext(os.path.basename(yuv_path))[0]
    y_out = os.path.join(out_dir, f"{base}_Y_only.yuv")
    yyy_out = os.path.join(out_dir, f"{base}_YYY_444.yuv")

    print(f"[PAVEN] Processing {total_frames} frames from: {yuv_path}")
    with open(yuv_path, "rb") as fin, open(y_out, "wb") as fy, open(yyy_out, "wb") as fyyy:
        for _ in range(total_frames):
            y = fin.read(y_size)
            fin.seek(2 * uv_size, os.SEEK_CUR)  # Skip U and V
            fy.write(y)
            # Replicate Y as Y, Y, Y (4:4:4 planar)
            fyyy.write(y)
            fyyy.write(y)
            fyyy.write(y)

    print(f"[PAVEN] Extracted Y stream: {y_out}")
    print(f"[PAVEN] Generated YYY stream: {yyy_out}")


def main():
    parser = argparse.ArgumentParser(description="PAVEN: YUV Channel Separator and YYY Generator")
    parser.add_argument("--input", "-i", required=True, type=str, help="Input raw YUV420p file")
    parser.add_argument("--width", "-w", required=True, type=int, help="Frame width in pixels")
    parser.add_argument("--height", "-H", required=True, type=int, help="Frame height in pixels")
    parser.add_argument("--out-dir", "-o", default="./output_channels", type=str, help="Output folder")
    parser.add_argument("--frames", "-n", default=-1, type=int, help="Number of frames to process (-1 for all)")
    args = parser.parse_args()

    split_channels(args.input, args.width, args.height, args.out_dir, args.frames)


if __name__ == "__main__":
    main()
