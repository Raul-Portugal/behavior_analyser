import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from roi_manager import ROIManager

try:
    from PyQt6.QtCore import QPoint
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtWidgets import QApplication
    from gui.roi_selector import (REFERENCE_LINE_NAME, ReferenceLineDialog,
                                  apply_reference_line, offer_reference_line)
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


def _manager():
    manager = ROIManager(source_dimensions=(640, 480))
    manager.add_roi("arm_a", np.array([[10, 10], [200, 10], [200, 100], [10, 100]]))
    return manager


def _click(x, y):
    class Event:
        def pos(self):
            return QPoint(x, y)
    return Event()


class ReferenceLineStorageTests(unittest.TestCase):
    def test_length_only_manager_serializes_without_new_keys(self):
        manager = _manager()
        manager.set_reference_length(120.5, "User-drawn reference line")
        data = manager.to_dict()
        self.assertNotIn("reference_line", data)
        self.assertEqual(data["schema_version"], 2)
        self.assertIsNone(manager.reference_line)

    def test_set_reference_line_keeps_endpoints_and_length_consistent(self):
        manager = _manager()
        manager.set_reference_line((10, 20), (13, 24), "line")
        self.assertEqual(manager.reference_line, ((10.0, 20.0), (13.0, 24.0)))
        self.assertAlmostEqual(manager.reference_length_pixels, 5.0)
        data = manager.to_dict()
        self.assertEqual(data["reference_line"], [[10.0, 20.0], [13.0, 24.0]])
        self.assertEqual(data["_reference_length_pixels"], 5.0)

    def test_round_trip_through_json(self):
        manager = _manager()
        manager.set_reference_line((10.25, 20.5), (310.75, 60.125), "line")
        restored = ROIManager.from_dict(json.loads(json.dumps(manager.to_dict())))
        self.assertEqual(restored.reference_line, manager.reference_line)
        self.assertEqual(
            restored.reference_length_pixels, manager.reference_length_pixels
        )
        self.assertEqual(restored.to_dict(), manager.to_dict())

    def test_old_config_with_only_a_length_still_loads(self):
        data = _manager().to_dict()
        data["_reference_length_pixels"] = 88.0
        data["_reference_name"] = "User-drawn reference line"
        restored = ROIManager.from_dict(data)
        self.assertIsNone(restored.reference_line)
        self.assertEqual(restored.reference_length_pixels, 88.0)
        self.assertEqual(restored.to_dict(), data)

    def test_setting_a_bare_length_drops_stale_endpoints(self):
        manager = _manager()
        manager.set_reference_line((0, 0), (30, 40))
        manager.set_reference_length(99.0)
        self.assertIsNone(manager.reference_line)
        self.assertNotIn("reference_line", manager.to_dict())

    def test_inconsistent_endpoints_are_dropped_on_load(self):
        manager = _manager()
        manager.set_reference_line((0, 0), (30, 40))
        data = manager.to_dict()
        data["reference_line"] = [[0, 0], [300, 400]]
        restored = ROIManager.from_dict(data)
        self.assertIsNone(restored.reference_line)
        self.assertEqual(restored.reference_length_pixels, 50.0)

    def test_malformed_endpoints_do_not_break_loading(self):
        manager = _manager()
        manager.set_reference_line((0, 0), (30, 40))
        for bad in ([[1, 2]], "xx", [[0, 0], [0, 0]], [[0, 0], [None, 1]]):
            data = manager.to_dict()
            data["reference_line"] = bad
            restored = ROIManager.from_dict(data)
            self.assertIsNone(restored.reference_line)
            self.assertEqual(restored.reference_length_pixels, 50.0)

    def test_degenerate_line_is_rejected(self):
        manager = _manager()
        with self.assertRaises(ValueError):
            manager.set_reference_line((5, 5), (5, 5))
        self.assertFalse(manager.has_reference)

    def test_deepcopy_preserves_line(self):
        manager = _manager()
        manager.set_reference_line((1, 2), (4, 6))
        self.assertEqual(copy.deepcopy(manager).reference_line, manager.reference_line)


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class ReferenceLineDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _dialog(self, scale=(2.0, 3.0)):
        pixmap = QPixmap(320, 200)
        pixmap.fill()
        return ReferenceLineDialog(pixmap, source_scale=scale)

    def test_endpoints_are_reported_in_source_pixels(self):
        dialog = self._dialog()
        try:
            self.assertIsNone(dialog.get_line_points())
            dialog.add_point(_click(10, 20))
            dialog.add_point(_click(40, 60))
            self.assertEqual(dialog.get_line_points(), ((20.0, 60.0), (80.0, 180.0)))
            self.assertAlmostEqual(dialog.get_line_length(), np.hypot(60, 120))
        finally:
            dialog.deleteLater()

    def test_apply_reference_line_stores_endpoints_and_matching_length(self):
        dialog = self._dialog()
        try:
            dialog.add_point(_click(10, 20))
            dialog.add_point(_click(40, 60))
            manager = ROIManager()
            self.assertTrue(apply_reference_line(manager, dialog))
            self.assertEqual(manager.reference_line, ((20.0, 60.0), (80.0, 180.0)))
            self.assertAlmostEqual(
                manager.reference_length_pixels, dialog.get_line_length()
            )
            self.assertEqual(manager.reference_name, REFERENCE_LINE_NAME)
        finally:
            dialog.deleteLater()

    def test_incomplete_line_is_not_applied(self):
        dialog = self._dialog()
        try:
            manager = ROIManager()
            self.assertFalse(apply_reference_line(manager, dialog))
            self.assertFalse(manager.has_reference)
        finally:
            dialog.deleteLater()

    def test_offer_reference_line_records_endpoints(self):
        frame = np.zeros((480, 1920, 3), dtype=np.uint8)

        class Capture:
            def get(self, _p):
                return 20

            def set(self, *_a):
                return True

            def read(self):
                return True, frame

            def release(self):
                pass

        def fake_exec(dialog):
            # The canvas shows the frame at source resolution (it zooms
            # instead of downscaling), so image pixels are source pixels.
            dialog.add_point(_click(150, 75))
            dialog.add_point(_click(450, 75))
            return 1

        manager = ROIManager()
        with patch("gui.roi_selector.cv2.VideoCapture", return_value=Capture()), \
                patch.object(ReferenceLineDialog, "exec", fake_exec):
            self.assertTrue(offer_reference_line(None, "v.mp4", manager))
        self.assertEqual(manager.source_dimensions, (1920, 480))
        self.assertEqual(manager.reference_line, ((150.0, 75.0), (450.0, 75.0)))
        self.assertAlmostEqual(manager.reference_length_pixels, 300.0)


if __name__ == "__main__":
    unittest.main()
