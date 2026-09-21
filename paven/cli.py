"""
PAVEN Command Line Interface (CLI): 'paven-generate'
Paper: "PAVEN: A Perceptual Algorithm for Versatile video Encoding using Neural networks"
       (Elsevier EAAI, 2025, DOI: 10.1016/j.engappai.2025.111664)
Author: Pablo Fernández Lagos (UPM, 2025)

Generates standardized CTU-level (128x128) Delta-QP configuration files (.qp)
from raw YUV video streams using the PAVEN perceptual neural network.
"""

import sys
import os
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm

from .model import PavenModel
from .qp_grid import QpGrid
from .yuv_loader import YUVReader


def main():
    parser = argparse.ArgumentParser(
        description="PAVEN: Perceptual Video Saliency & Delta-QP Matrix Generator"
    )
    parser.add_argument("--video", "-i", required=True, type=str, help="Path to input raw YUV420p video file")
    parser.add_argument("--output", "-o", required=True, type=str, help="Path to output .qp matrix file")
    parser.add_argument("--width", "-w", required=True, type=int, help="Video width in pixels (e.g. 1920)")
    parser.add_argument("--height", "-H", required=True, type=int, help="Video height in pixels (e.g. 1080)")
    parser.add_argument("--fps", type=float, default=30.0, help="Video frame rate (default: 30.0)")
    parser.add_argument("--weights", "-m", type=str, default=None, help="Path to local weights (.pt). If None, downloads from Hugging Face Hub (lagosproject/paven)")
    parser.add_argument("--is-10bit", action="store_true", help="Set flag if input is 10-bit YUV420p")
    parser.add_argument("--cell-size", type=int, default=128, help="CTU block size in pixels (default: 128)")
    parser.add_argument("--clip-len", type=int, default=32, help="Temporal window size in frames (default: 32)")
    parser.add_argument("--vis-dir", type=str, default=None, help="Optional directory to save visual grid overlays")
    parser.add_argument("--max-frames", type=int, default=-1, help="Maximum frames to process (-1 for entire video)")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda', 'cpu')")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase verbosity level")

    args = parser.parse_args()

    # Determine execution device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[PAVEN] Initializing on device: {device}")
    print(f"[PAVEN] Input Video: {args.video} ({args.width}x{args.height})")

    # Load YUV stream
    reader = YUVReader(args.video, width=args.width, height=args.height, is_10bit=args.is_10bit)
    total_frames = len(reader)
    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)
    print(f"[PAVEN] Total frames to process: {total_frames}")

    # Load Neural Network Model
    if args.weights and os.path.isfile(args.weights):
        print(f"[PAVEN] Loading local weights: {args.weights}")
        model = PavenModel.from_file(args.weights, device=device)
    else:
        print("[PAVEN] No local weights provided. Loading from Hugging Face Hub ('lagosproject/paven')...")
        model = PavenModel.from_pretrained(repo_id="lagosproject/paven", device=device)

    model.eval()

    # Initialize QP Grid Generator
    qp_gen = QpGrid(
        output_path=args.output,
        qp_grid_vis_path=args.vis_dir,
        img_size=(args.width, args.height),
        frame_count=total_frames,
        cell_size=args.cell_size,
        verbose=args.verbose,
    )

    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)

    print("[PAVEN] Generating perceptual saliency and Delta-QP matrices...")
    clip_len = args.clip_len

    with torch.no_grad():
        for frame_idx in tqdm(range(total_frames), desc="Encoding CTU Grids"):
            # Use a sliding window of clip_len frames
            start_idx = max(0, frame_idx - clip_len + 1)
            clip_tensor = reader.get_clip_tensor(
                start_frame=start_idx,
                clip_len=clip_len,
                target_size=(384, 224),
                to_yyy=True
            ).to(device)

            # Saliency prediction
            saliency_pred = model(clip_tensor)  # (1, 224, 384)
            saliency_2d = saliency_pred.squeeze().cpu().numpy()

            # Process QP grid for current frame
            qp_gen.process_image(saliency_2d, frame_idx)

    # Save to file
    out_file = qp_gen.save_qp_to_file()
    print(f"[PAVEN] Delta-QP matrix successfully saved to: {out_file}")
    print(f"[PAVEN] Ready for ingestion by VTM, HM, or other compatible video encoders.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
