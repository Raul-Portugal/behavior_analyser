import unittest
from unittest.mock import patch

import numpy as np

try:
    from PyQt6.QtCore import QPointF
    from PyQt6.QtWidgets import QApplication
    from gui.roi_selector import RoiSelectorDialog
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


class _Maze:
    def get_roi_definitions(self):
        return [("mouse_1", "Mouse 1")]


class _Capture:
    def get(self, _property):
        return 20

    def set(self, _property, _value):
        return True

    def read(self):
        return True, np.zeros((1080, 1920, 3), dtype=np.uint8)

    def release(self):
        pass


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class RoiSelectorCoordinateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @patch("gui.roi_selector.cv2.VideoCapture", return_value=_Capture())
    def test_scrollable_canvas_preserves_click_coordinates(self, _capture):
        dialog = RoiSelectorDialog("example.mp4", _Maze())
        try:
            dialog.resize(500, 360)
            dialog.show()
            self.app.processEvents()
            self.assertEqual(dialog.image_label.size(), dialog.base_pixmap.size())
            self.assertIs(dialog.image_scroll.widget(), dialog.image_label)
            self.assertFalse(dialog.image_scroll.isAncestorOf(dialog.next_button))
            self.assertTrue(dialog.next_button.isVisible())

            dialog.current_polygon_points = [
                QPointF(100, 100), QPointF(200, 100), QPointF(200, 200)
            ]
            dialog.finalize_polygon()
            points = dialog.roi_manager.rois["mouse_1"][0].points
            np.testing.assert_array_equal(
                points,
                np.array([[150, 150], [300, 150], [300, 300]]),
            )
        finally:
            dialog.close()


if __name__ == "__main__":
    unittest.main()
