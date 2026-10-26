"""RoiEditorDialog / RoiSelectorDialog: finishing, calibration, preloading."""
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from gui import ux_helpers
    from gui.roi_editor import RoiEditorDialog, RoiEditorWidget
    from gui.roi_selector import (ReferenceLineDialog, RoiSelectorDialog,
                                  apply_reference_line)
    from mazes import create_maze
    from roi_manager import ROIManager
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


class _Capture:
    def __init__(self, width=320, height=240, readable=True):
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.readable = readable

    def get(self, _property):
        return 20

    def set(self, _property, _value):
        return True

    def read(self):
        return self.readable, (self.frame if self.readable else None)

    def release(self):
        pass


def _square(x, y, size=60):
    return [(x, y), (x + size, y), (x + size, y + size), (x, y + size)]


class _DialogCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def dialog(self, maze=None, cls=None, **kwargs):
        cls = cls or RoiEditorDialog
        with patch("gui.roi_selector.cv2.VideoCapture", return_value=_Capture()):
            dialog = cls("example.mp4", maze or create_maze("Y-Maze"), **kwargs)
        self.addCleanup(self._dispose, dialog)
        return dialog

    @staticmethod
    def _dispose(dialog):
        with patch.object(ux_helpers, "confirm_discard", return_value=True):
            dialog.close()
        dialog.deleteLater()

    @staticmethod
    def draw(dialog, points):
        for x, y in points:
            dialog.editor.add_point(x, y)
        return dialog.editor.close_shape()

    def fill_y_maze(self, dialog):
        for index, category in enumerate(("arm_a", "arm_b", "arm_c", "center")):
            dialog.editor.set_current_zone(category)
            self.assertTrue(self.draw(dialog, _square(10 + 70 * index, 10)))


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class FinishingTests(_DialogCase):
    def test_unreadable_video_raises_like_the_old_dialog(self):
        with patch("gui.roi_selector.cv2.VideoCapture",
                   return_value=_Capture(readable=False)):
            with self.assertRaises(IOError):
                RoiEditorDialog("missing.mp4", create_maze("Y-Maze"))

    def test_ok_lists_the_zones_that_still_break_a_rule(self):
        dialog = self.dialog()
        self.draw(dialog, _square(10, 10))
        with patch.object(QMessageBox, "warning") as warning:
            dialog.accept()
        warning.assert_called_once()
        message = warning.call_args[0][2]
        self.assertIn("Arm B", message)
        self.assertIn("Center", message)
        self.assertNotIn("Arm A", message)
        self.assertEqual(dialog.result(), 0)

    def test_ok_asks_for_calibration_once_then_returns_the_rois(self):
        dialog = self.dialog()
        self.fill_y_maze(dialog)
        with patch.object(QMessageBox, "question",
                          return_value=QMessageBox.StandardButton.No) as question:
            dialog.accept()
        question.assert_called_once()
        self.assertEqual(dialog.result(), dialog.DialogCode.Accepted)
        manager = dialog.get_roi_manager()
        self.assertEqual(list(manager.rois), ["arm_a", "arm_b", "arm_c", "center"])
        self.assertFalse(manager.has_reference)
        self.assertEqual(manager.source_dimensions, (320, 240))
        self.assertIs(dialog.get_roi_manager(), manager)   # stable after accept

    def test_declining_calibration_is_remembered_if_ok_is_pressed_again(self):
        dialog = self.dialog()
        self.fill_y_maze(dialog)
        with patch.object(QMessageBox, "question",
                          return_value=QMessageBox.StandardButton.Yes):
            dialog.accept()
        self.assertEqual(dialog.editor.tool(), "line")
        self.assertEqual(dialog.result(), 0)
        with patch.object(QMessageBox, "question",
                          return_value=QMessageBox.StandardButton.No) as question:
            dialog.accept()
            question.assert_called_once()
        self.assertEqual(dialog.result(), dialog.DialogCode.Accepted)

    def test_tst_never_asks_for_calibration(self):
        maze = create_maze("Tail Suspension Test")
        maze.configure_mice(1)
        dialog = self.dialog(maze)
        self.draw(dialog, _square(10, 10))
        with patch.object(QMessageBox, "question") as question:
            dialog.accept()
        question.assert_not_called()
        self.assertEqual(dialog.result(), dialog.DialogCode.Accepted)

    def test_a_drawn_line_is_not_asked_about_again(self):
        dialog = self.dialog()
        self.fill_y_maze(dialog)
        dialog.editor.set_line((10, 100), (110, 100))
        with patch.object(QMessageBox, "question") as question:
            dialog.accept()
        question.assert_not_called()
        manager = dialog.get_roi_manager()
        self.assertEqual(manager.reference_line, ((10.0, 100.0), (110.0, 100.0)))

    def test_ok_closes_a_pending_shape_and_refuses_a_stub(self):
        dialog = self.dialog(create_maze("Tail Suspension Test"))
        dialog.editor.add_point(10, 10)
        dialog.editor.add_point(20, 10)
        with patch.object(QMessageBox, "warning") as warning:
            dialog.accept()
        self.assertIn("three points", warning.call_args[0][2])
        self.assertEqual(dialog.result(), 0)

    def test_enter_never_presses_ok(self):
        dialog = self.dialog()
        dialog.show()
        QTest.keyClick(dialog.ok_button, Qt.Key.Key_Return)
        self.assertEqual(dialog.result(), 0)
        self.assertTrue(dialog.isVisible())

    def test_shortcuts_reach_the_editor_from_the_dialog(self):
        dialog = self.dialog()
        dialog.show()
        for x, y in _square(10, 10)[:3]:
            dialog.editor.add_point(x, y)
        QTest.keyClick(dialog, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(len(dialog.editor.pending_points()), 2)

    def test_cancel_with_changes_asks_first(self):
        dialog = self.dialog()
        self.draw(dialog, _square(10, 10))
        with patch.object(ux_helpers, "confirm_discard", return_value=False) as confirm:
            dialog.cancel_button.click()
        confirm.assert_called_once()
        self.assertEqual(dialog.result(), 0)


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class CompatibilityTests(_DialogCase):
    def test_selector_is_an_editor_dialog_with_the_3x_surface(self):
        dialog = self.dialog(cls=RoiSelectorDialog)
        self.assertIsInstance(dialog, RoiEditorDialog)
        self.assertEqual(dialog.windowTitle(), "ROI Selection")
        self.assertEqual(
            [category for category, _ in dialog.roi_definitions],
            ["arm_a", "arm_b", "arm_c", "center"],
        )
        self.assertFalse(dialog.arena_step_enabled)
        self.assertEqual(dialog.zone_definitions, dialog.roi_definitions)
        self.assertEqual(dialog.video_path, "example.mp4")

    def test_selector_accepts_the_new_optional_initial_manager(self):
        manager = ROIManager((320, 240))
        manager.add_roi("arm_a", np.array(_square(10, 10)))
        manager.set_reference_line((0, 0), (30, 40), "User-drawn reference line")
        dialog = self.dialog(cls=RoiSelectorDialog, initial_roi_manager=manager)
        self.assertEqual(len(dialog.roi_manager.rois["arm_a"]), 1)
        self.assertEqual(dialog.editor.line_points(), ((0.0, 0.0), (30.0, 40.0)))
        self.assertFalse(dialog.has_unsaved_work())
        # Positional use stays valid: (video, maze, parent).
        plain = self.dialog(cls=RoiSelectorDialog)
        self.assertEqual(plain.roi_manager.rois, {})

    def test_redraw_preloads_edits_and_keeps_untouched_zones(self):
        manager = ROIManager((320, 240))
        for index, category in enumerate(("arm_a", "arm_b", "arm_c", "center")):
            manager.add_roi(category, np.array(_square(10 + 70 * index, 10)))
        manager.set_reference_line((10, 150), (110, 150), "User-drawn reference line")
        dialog = self.dialog(cls=RoiSelectorDialog, initial_roi_manager=manager)
        self.assertTrue(dialog.editor.is_valid())
        shape = dialog.editor.session.document.shapes_in("arm_b")[0]
        dialog.editor.set_tool("edit")
        dialog.editor.session.select(shape.id, 2)
        dialog.editor.session.nudge(5, 5)
        with patch.object(QMessageBox, "question") as question:
            dialog.accept()
        question.assert_not_called()
        result = dialog.get_roi_manager()
        np.testing.assert_array_equal(
            result.rois["arm_a"][0].points, manager.rois["arm_a"][0].points)
        self.assertEqual(result.rois["arm_b"][0].points[2].tolist(), [145, 75])
        self.assertEqual(result.reference_line, manager.reference_line)


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class ReferenceLineDialogCanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_line_is_reported_in_source_pixels_and_can_be_adjusted(self):
        from PyQt6.QtGui import QPixmap
        pixmap = QPixmap(320, 200)
        pixmap.fill()
        dialog = ReferenceLineDialog(pixmap, source_scale=(2.0, 3.0))
        self.addCleanup(dialog.deleteLater)
        self.assertIsInstance(dialog.editor, RoiEditorWidget)
        # The scene spans the SOURCE frame (640 x 600), not the pixmap.
        self.assertEqual(dialog.editor.source_size, (640, 600))

        dialog.editor.add_line_point(20, 60)
        dialog.editor.add_line_point(80, 180)
        manager = ROIManager()
        self.assertTrue(apply_reference_line(manager, dialog))
        self.assertEqual(manager.reference_line, ((20.0, 60.0), (80.0, 180.0)))
        dialog.reset()
        self.assertIsNone(dialog.get_line_points())
        self.assertEqual(dialog.get_line_length(), 0.0)

    def test_accept_needs_a_complete_line(self):
        from PyQt6.QtGui import QPixmap
        pixmap = QPixmap(100, 100)
        pixmap.fill()
        dialog = ReferenceLineDialog(pixmap)
        self.addCleanup(dialog.deleteLater)
        with patch.object(QMessageBox, "warning") as warning:
            dialog.accept_line()
        warning.assert_called_once()
        self.assertEqual(dialog.result(), 0)


if __name__ == "__main__":
    unittest.main()
