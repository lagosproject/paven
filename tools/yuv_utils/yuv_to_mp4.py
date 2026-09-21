"""
PAVEN: YUV <-> MP4 Video Format Conversion Utility
Paper: "PAVEN: A Perceptual Algorithm for Versatile video Encoding using Neural networks"
       (Elsevier EAAI, 2025, DOI: 10.1016/j.engappai.2025.111664)

Automates high-quality conversions between raw planar YUV and compressed MP4 formats using FFmpeg.
"""

import os
import subprocess
import argparse


def yuv_to_mp4(yuv_path: str, mp4_path: str, width: int, height: int, fps: float = 30.0, is_10bit: bool = False):
    pix_fmt = "yuv420p10le" if is_10bit else "yuv420p"
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", pix_fmt,
        "-r", str(fps),
        "-i", yuv_path,
        "-c:v", "libx264",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        mp4_path
    ]
    print(f"[PAVEN] Running FFmpeg: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[PAVEN] Conversion successful: {mp4_path}")


def mp4_to_yuv(mp4_path: str, yuv_path: str, is_10bit: bool = False):
    pix_fmt = "yuv420p10le" if is_10bit else "yuv420p"
    cmd = [
        "ffmpeg", "-y",
        "-i", mp4_path,
        "-pix_fmt", pix_fmt,
        yuv_path
    ]
    print(f"[PAVEN] Running FFmpeg: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[PAVEN] Conversion successful: {yuv_path}")


def main():
    parser = argparse.ArgumentParser(description="PAVEN: YUV and MP4 Video Converter via FFmpeg")
    parser.add_argument("--mode", choices=["yuv2mp4", "mp42yuv"], required=True, help="Conversion direction")
    parser.add_argument("--input", "-i", required=True, help="Input video file path")
    parser.add_argument("--output", "-o", required=True, help="Output video file path")
    parser.add_argument("--width", "-w", type=int, help="Width in pixels (required for yuv2mp4)")
    parser.add_argument("--height", "-H", type=int, help="Height in pixels (required for yuv2mp4)")
    parser.add_argument("--fps", type=float, default=30.0, help="Frame rate (default: 30.0)")
    parser.add_argument("--is-10bit", action="store_true", help="Use 10-bit depth (yuv420p10le)")
    args = parser.parse_args()

    if args.mode == "yuv2mp4":
        if not args.width or not args.height:
            raise ValueError("--width and --height are required for yuv2mp4 conversion.")
        yuv_to_mp4(args.input, args.output, args.width, args.height, args.fps, args.is_10bit)
    else:
        mp4_to_yuv(args.input, args.output, args.is_10bit)


if __name__ == "__main__":
    main()
