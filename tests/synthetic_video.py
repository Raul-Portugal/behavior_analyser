"""A tiny synthetic video whose frame index is encoded in the pixel values."""
from pathlib import Path

import cv2
import numpy as np

SIZE = (160, 120)   # width, height
STEP = 3            # brightness of the marker patch is STEP * index + 10
FPS = 25.0


def write_indexed_video(path, frame_count=80, fps=FPS, size=SIZE) -> Path:
    """Write an MJPG AVI; frame ``i`` has a flat patch of ``STEP * i + 10``."""
    path = Path(path)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), float(fps), tuple(size)
    )
    rng = np.random.default_rng(3)
    width, height = size
    for index in range(frame_count):
        frame = rng.integers(0, 40, (height, width, 3), dtype=np.uint8)
        frame[65:115, 5:75] = min(255, STEP * index + 10)
        writer.write(frame)
    writer.release()
    return path


def decode_index(frame) -> int:
    """The frame index encoded by :func:`write_indexed_video`."""
    return int(round((float(frame[70:110, 10:70].mean()) - 10) / STEP))
