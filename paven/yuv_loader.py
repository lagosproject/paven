"""
PAVEN: High-Performance Raw YUV Video Loader & Streamer
Paper: "PAVEN: A Perceptual Algorithm for Versatile video Encoding using Neural networks"
       (Elsevier EAAI, 2025, DOI: 10.1016/j.engappai.2025.111664)
Author: Pablo Fernández Lagos (UPM, 2025)

Supports raw planar YUV420p streams in both 8-bit and 10-bit color depths.
Optimized for direct luminance extraction and YYY replication for the PAVEN model.
"""

import os
import cv2
import torch
import numpy as np
from typing import Tuple, Optional, Generator, Union


class YUVReader:
    """
    Streaming reader and tensor extractor for raw YUV 4:2:0 video sequences.
    """

    def __init__(
        self,
        file_path: str,
        width: int,
        height: int,
        is_10bit: bool = False,
    ):
        self.file_path = file_path
        self.width = width
        self.height = height
        self.is_10bit = is_10bit

        # Calculate bytes per frame for planar 4:2:0
        bytes_per_sample = 2 if is_10bit else 1
        self.y_size = width * height * bytes_per_sample
        self.uv_size = (width // 2) * (height // 2) * bytes_per_sample
        self.frame_bytes = self.y_size + 2 * self.uv_size

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"YUV video file not found: {file_path}")

        file_size = os.path.getsize(file_path)
        self.num_frames = file_size // self.frame_bytes
        self.dtype = np.uint16 if is_10bit else np.uint8

    def __len__(self) -> int:
        return self.num_frames

    def read_frame_y(self, frame_idx: int) -> np.ndarray:
        """
        Extract only the Luminance (Y) channel for a specific frame index.
        Extremely fast and memory-efficient for YYY mode.
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise IndexError(f"Frame index {frame_idx} out of range (0..{self.num_frames - 1})")

        with open(self.file_path, "rb") as f:
            f.seek(frame_idx * self.frame_bytes)
            raw = f.read(self.y_size)
            y_plane = np.frombuffer(raw, dtype=self.dtype).reshape((self.height, self.width))

            if self.is_10bit:
                # Bit-shift from 10-bit (0..1023) down to 8-bit (0..255)
                y_plane = (y_plane >> 2).astype(np.uint8)

            return y_plane

    def read_frame_yuv(self, frame_idx: int) -> np.ndarray:
        """
        Read complete YUV frame, upsampling U and V planes to full resolution.
        Returns: (height, width, 3) uint8 array.
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise IndexError(f"Frame index {frame_idx} out of range (0..{self.num_frames - 1})")

        with open(self.file_path, "rb") as f:
            f.seek(frame_idx * self.frame_bytes)
            y_raw = f.read(self.y_size)
            u_raw = f.read(self.uv_size)
            v_raw = f.read(self.uv_size)

            y = np.frombuffer(y_raw, dtype=self.dtype).reshape((self.height, self.width))
            u = np.frombuffer(u_raw, dtype=self.dtype).reshape((self.height // 2, self.width // 2))
            v = np.frombuffer(v_raw, dtype=self.dtype).reshape((self.height // 2, self.width // 2))

            if self.is_10bit:
                y = (y >> 2).astype(np.uint8)
                u = (u >> 2).astype(np.uint8)
                v = (v >> 2).astype(np.uint8)

            u_up = cv2.resize(u, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            v_up = cv2.resize(v, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            return np.dstack((y, u_up, v_up))

    def get_clip_tensor(
        self,
        start_frame: int,
        clip_len: int = 32,
        target_size: Tuple[int, int] = (384, 224),
        to_yyy: bool = True,
    ) -> torch.Tensor:
        """
        Extract a normalized PyTorch clip tensor formatted for PAVEN model inference.

        Args:
            start_frame: First frame index of temporal window.
            clip_len: Number of consecutive frames (default: 32).
            target_size: Model input spatial resolution as (width, height) (default: 384x224).
            to_yyy: If True, replicates Y channel across 3 channels (YYY mode).

        Returns:
            torch.Tensor: Shape (1, 3, clip_len, height, width) with float values in [-1, 1].
        """
        frames = []
        for i in range(start_frame, start_frame + clip_len):
            idx = min(i, self.num_frames - 1)
            if to_yyy:
                y = self.read_frame_y(idx)
                y_resized = cv2.resize(y, target_size, interpolation=cv2.INTER_LINEAR)
                # Replicate luminance Y -> YYY
                frame_3ch = np.dstack((y_resized, y_resized, y_resized))
            else:
                yuv = self.read_frame_yuv(idx)
                frame_3ch = cv2.resize(yuv, target_size, interpolation=cv2.INTER_LINEAR)

            # Normalize to [-1.0, 1.0] (matching ViNet dataloader)
            frame_norm = (frame_3ch.astype(np.float32) / 127.5) - 1.0
            frames.append(frame_norm)

        # Stack into (Time, Height, Width, Channels) -> (1, Channels, Time, Height, Width)
        clip_np = np.stack(frames, axis=0)  # (T, H, W, C)
        clip_tensor = torch.from_numpy(clip_np).permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, T, H, W)
        return clip_tensor.float()


class YUVFileLoader:
    """
    Backwards-compatible wrapper matching original YUVFileLoader interface.
    """

    @staticmethod
    def load_yuv_file(file_path: str, width: int, height: int, is_10bit: bool = False) -> np.ndarray:
        reader = YUVReader(file_path, width, height, is_10bit)
        video = np.empty((len(reader), height, width, 3), dtype=np.uint8)
        for i in range(len(reader)):
            video[i] = reader.read_frame_yuv(i)
        return video
