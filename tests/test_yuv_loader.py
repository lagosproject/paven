import pytest
import os
import tempfile
import numpy as np
from paven.yuv_loader import YUVReader, YUVFileLoader


def test_yuv_reader_synthetic_stream():
    """Create a temporary raw YUV420p video and test frame-by-frame and clip reading."""
    width, height = 320, 240
    num_frames = 6
    frame_bytes = width * height * 3 // 2

    with tempfile.NamedTemporaryFile(suffix=".yuv", delete=False) as tmp:
        tmp_path = tmp.name
        for idx in range(num_frames):
            y = np.full((height, width), fill_value=idx * 40, dtype=np.uint8)
            u = np.full((height // 2, width // 2), fill_value=128, dtype=np.uint8)
            v = np.full((height // 2, width // 2), fill_value=128, dtype=np.uint8)
            tmp.write(y.tobytes())
            tmp.write(u.tobytes())
            tmp.write(v.tobytes())

    try:
        reader = YUVReader(tmp_path, width=width, height=height)
        assert len(reader) == num_frames

        # Test Y extraction
        y0 = reader.read_frame_y(0)
        assert y0.shape == (height, width)
        assert np.all(y0 == 0)

        y3 = reader.read_frame_y(3)
        assert np.all(y3 == 120)

        # Test full YUV extraction
        yuv_frame = reader.read_frame_yuv(1)
        assert yuv_frame.shape == (height, width, 3)

        # Test PyTorch clip tensor generation
        clip = reader.get_clip_tensor(start_frame=0, clip_len=4, target_size=(384, 224), to_yyy=True)
        assert clip.shape == (1, 3, 4, 224, 384)
        assert clip.dtype.is_floating_point
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
