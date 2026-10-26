import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402

try:
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
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
    def test_scrolled_zoomed_canvas_preserves_click_coordinates(self, _capture):
        # The editor's scene is in source pixels, so a click lands on the
        # pixel under the cursor at any zoom and scroll position; the dialog
        # buttons stay reachable in a small window.
        dialog = RoiSelectorDialog("example.mp4", _Maze())
        try:
            dialog.resize(500, 360)
            dialog.show()
            self.app.processEvents()
            canvas = dialog.editor.canvas
            self.assertEqual(canvas.scene().sceneRect().size().width(), 1920.0)
            self.assertEqual(canvas.scene().sceneRect().size().height(), 1080.0)
            self.assertTrue(dialog.ok_button.isVisible())
            self.assertTrue(dialog.cancel_button.isVisible())
            self.assertFalse(canvas.isAncestorOf(dialog.ok_button))

            canvas.set_zoom(1.0)
            canvas.horizontalScrollBar().setValue(700)
            canvas.verticalScrollBar().setValue(300)
            self.app.processEvents()
            expected = []
            for view_point in ((60, 50), (160, 50), (160, 150)):
                point = QPoint(*view_point)
                scene = canvas.mapToScene(point)
                expected.append((int(scene.x()), int(scene.y())))
                QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton,
                                 Qt.KeyboardModifier.NoModifier, point)
            self.assertEqual(dialog.editor.pending_points(), expected)

            QTest.keyClick(dialog, Qt.Key.Key_Return)
            points = dialog.roi_manager.rois["mouse_1"][0].points
            np.testing.assert_array_equal(points, np.array(expected))
            self.assertEqual(points.dtype, np.int32)
            # Scrolled into the middle of the 1920x1080 frame, not the corner.
            self.assertGreater(points[0][0], 700)
            self.assertGreater(points[0][1], 300)
        finally:
            # Closing a dialog with a drawn polygon asks first; allow it.
            with patch("gui.ux_helpers.confirm_discard", return_value=True):
                dialog.close()


if __name__ == "__main__":
    unittest.main()
