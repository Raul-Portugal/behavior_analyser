import gc
import unittest

import numpy as np

try:
    from PyQt6.QtGui import QImage
    from PyQt6.QtWidgets import QApplication
    from gui.qt_image import bgr_to_qpixmap, gray_or_bgr_to_qimage
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class QtImageConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_bgr_frame_gives_matching_pixmap_size(self):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        pixmap = bgr_to_qpixmap(frame)
        self.assertEqual((pixmap.width(), pixmap.height()), (64, 48))

    def test_gray_frame_gives_matching_pixmap_size(self):
        frame = np.zeros((30, 50), dtype=np.uint8)
        pixmap = bgr_to_qpixmap(frame)
        self.assertEqual((pixmap.width(), pixmap.height()), (50, 30))

    def test_single_channel_3d_input_is_treated_as_gray(self):
        frame = np.zeros((20, 40, 1), dtype=np.uint8)
        pixmap = bgr_to_qpixmap(frame)
        self.assertEqual((pixmap.width(), pixmap.height()), (40, 20))

    def test_bgr_channel_order_is_converted_to_rgb(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[:, :] = (255, 0, 0)  # pure blue in BGR
        image = gray_or_bgr_to_qimage(frame)
        color = image.pixelColor(1, 1)
        self.assertEqual((color.red(), color.green(), color.blue()), (0, 0, 255))

    def test_image_survives_release_of_source_buffer(self):
        frame = np.full((16, 16, 3), 200, dtype=np.uint8)
        image = gray_or_bgr_to_qimage(frame)
        del frame
        gc.collect()
        self.assertIsInstance(image, QImage)
        self.assertEqual(image.pixelColor(5, 5).red(), 200)

    def test_non_contiguous_input_is_supported(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)[::2, ::2]
        pixmap = bgr_to_qpixmap(frame)
        self.assertEqual((pixmap.width(), pixmap.height()), (30, 20))

    def test_unsupported_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            gray_or_bgr_to_qimage(np.zeros((8, 8, 4), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
