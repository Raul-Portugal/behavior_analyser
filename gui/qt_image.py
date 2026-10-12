"""Shared OpenCV (numpy) to Qt image conversion helpers."""
import cv2
import numpy as np
from PyQt6.QtGui import QImage, QPixmap


def gray_or_bgr_to_qimage(image: np.ndarray) -> QImage:
    """Convert a uint8 gray (H, W) / (H, W, 1) or BGR (H, W, 3) array to a QImage.

    The returned QImage owns its pixel data (it is ``copy()``-ed), so it stays
    valid after the numpy buffer, or the temporary RGB conversion made here, is
    garbage collected. Wrapping ``ndarray.data`` directly would leave the image
    pointing at freed memory.
    """
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 2:
        array = np.ascontiguousarray(array)
        h, w = array.shape
        return QImage(
            array.data, w, h, w, QImage.Format.Format_Grayscale8
        ).copy()

    if array.ndim == 3 and array.shape[2] == 3:
        rgb = np.ascontiguousarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
        h, w = rgb.shape[:2]
        return QImage(
            rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888
        ).copy()

    raise ValueError(
        f"Unsupported image shape for Qt conversion: {array.shape}"
    )


def bgr_to_qpixmap(frame: np.ndarray) -> QPixmap:
    """Convert a BGR (or gray) OpenCV frame to a QPixmap."""
    return QPixmap.fromImage(gray_or_bgr_to_qimage(frame))
