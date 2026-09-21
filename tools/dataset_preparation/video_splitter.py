"""
Video Splitter Utility
Extracts sequential image frames (.jpg) from video files (e.g., MP4, AVI).
"""

import os
import argparse
import cv2
from pathlib import Path


def split_video_to_frames(video_path: str, output_dir: str, ext: str = "jpg") -> int:
    """Extract frames from a video file and save them as individual images."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame_filename = f"{frame_count:04d}.{ext}"
        cv2.imwrite(os.path.join(output_dir, frame_filename), frame)

    cap.release()
    return frame_count


def main():
    parser = argparse.ArgumentParser(description="Split video files into individual frames.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to video file or directory of videos.")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory to save frames.")
    parser.add_argument("--ext", default="jpg", choices=["jpg", "png"], help="Output image format.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_file():
        count = split_video_to_frames(str(input_path), args.output, args.ext)
        print(f"Extracted {count} frames from {input_path.name} to {args.output}")
    elif input_path.is_dir():
        for vid in sorted(input_path.glob("*.mp4")) + sorted(input_path.glob("*.avi")):
            vid_out = os.path.join(args.output, vid.stem, "frames")
            count = split_video_to_frames(str(vid), vid_out, args.ext)
            print(f"Extracted {count} frames from {vid.name} -> {vid_out}")


if __name__ == "__main__":
    main()
