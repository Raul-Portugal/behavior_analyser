import unittest

import numpy as np

from core.calibration import (real_world_cm_from_scale,
                              rescale_for_new_reference,
                              scale_factor_from_cm)
from roi_manager import ROIManager


def _tilted_arm() -> np.ndarray:
    # A long thin arm drawn at roughly 120 degrees: bounding-box width is
    # far smaller than the arm length.
    return np.array([[200, 50], [220, 50], [100, 250], [80, 250]])


class ReferenceLengthTests(unittest.TestCase):
    def test_no_bounding_box_fallback(self):
        manager = ROIManager()
        manager.add_roi("arm_a", _tilted_arm())
        self.assertEqual(manager.calculate_reference_length(), 0.0)
        self.assertEqual(manager.reference_length_pixels, 0.0)
        self.assertFalse(manager.has_reference)
        self.assertEqual(manager.reference_name, "N/A")

    def test_user_line_is_kept(self):
        manager = ROIManager()
        manager.add_roi("arm_a", _tilted_arm())
        manager.set_reference_length(150.0, "User-drawn reference line")
        self.assertEqual(manager.calculate_reference_length(), 150.0)
        self.assertTrue(manager.has_reference)
        self.assertFalse(manager.is_legacy_bbox_reference)

    def test_legacy_bounding_box_reference_is_detected_on_load(self):
        manager = ROIManager()
        manager.add_roi("arm_a", _tilted_arm())
        data = manager.to_dict()
        data["_reference_length_pixels"] = 120.0
        data["_reference_name"] = "Width of first Arm A ROI"
        loaded = ROIManager.from_dict(data)
        self.assertEqual(loaded.reference_length_pixels, 120.0)
        self.assertTrue(loaded.is_legacy_bbox_reference)

    def test_legacy_detection_requires_a_length(self):
        manager = ROIManager()
        manager._reference_name = "Width of first Center ROI"
        self.assertFalse(manager.is_legacy_bbox_reference)


class ScaleHelperTests(unittest.TestCase):
    def test_scale_from_cm(self):
        self.assertAlmostEqual(scale_factor_from_cm(30.0, 300.0), 0.1)

    def test_unusable_inputs_give_pixel_units(self):
        self.assertEqual(scale_factor_from_cm(0.0, 300.0), 0.0)
        self.assertEqual(scale_factor_from_cm(30.0, 0.0), 0.0)
        self.assertEqual(scale_factor_from_cm(-1.0, 300.0), 0.0)

    def test_cm_round_trip(self):
        scale = scale_factor_from_cm(30.0, 300.0)
        self.assertAlmostEqual(real_world_cm_from_scale(scale, 300.0), 30.0)
        self.assertIsNone(real_world_cm_from_scale(0.0, 300.0))
        self.assertIsNone(real_world_cm_from_scale(0.1, 0.0))

    def test_redraw_recomputes_from_cm_not_old_scale(self):
        old_scale = scale_factor_from_cm(30.0, 300.0)
        new_scale = rescale_for_new_reference(30.0, 150.0)
        self.assertAlmostEqual(new_scale, 0.2)
        self.assertNotAlmostEqual(new_scale, old_scale)
        # A redrawn video without a new line falls back to pixels.
        self.assertEqual(rescale_for_new_reference(30.0, 0.0), 0.0)


class SettingsAfterRedrawTests(unittest.TestCase):
    def _settings(self, scale):
        from core.models import AppConfig, BatchSettings
        manager = ROIManager()
        manager.add_roi("arm_a", _tilted_arm())
        manager.set_reference_length(300.0, "User-drawn reference line")
        return BatchSettings(manager, AppConfig().detection, scale, 5.0, 60.0)

    def test_cm_is_carried_and_scale_recomputed(self):
        from gui.main_window import MainWindow
        settings = self._settings(0.1)
        redrawn = ROIManager()
        redrawn.add_roi("arm_a", _tilted_arm())
        redrawn.set_reference_length(150.0, "User-drawn reference line")
        result = MainWindow.settings_after_redraw(settings, redrawn, 30.0)
        self.assertAlmostEqual(result.scale_factor, 0.2)
        self.assertIs(result.roi_manager, redrawn)
        self.assertEqual((result.start_time, result.end_time), (5.0, 60.0))
        # The original is untouched.
        self.assertAlmostEqual(settings.scale_factor, 0.1)

    def test_redraw_without_line_falls_back_to_pixels(self):
        from gui.main_window import MainWindow
        redrawn = ROIManager()
        redrawn.add_roi("arm_a", _tilted_arm())
        result = MainWindow.settings_after_redraw(
            self._settings(0.1), redrawn, 30.0
        )
        self.assertEqual(result.scale_factor, 0.0)


if __name__ == "__main__":
    unittest.main()
