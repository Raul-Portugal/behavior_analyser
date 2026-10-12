import unittest
from unittest.mock import patch

import cv2
import numpy as np

from core import roi_alignment as ra
from roi_manager import ROIManager
from tests.alignment_support import (TEMPLATE_DIMS, arena_image, camera_move,
                                     max_grid_error, rescale, warp)

try:
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication
    from gui.roi_alignment_dialog import (METHOD_AS_IS, METHOD_AUTO,
                                          METHOD_LANDMARKS, METHOD_SCALE,
                                          RoiAlignmentDialog,
                                          align_template_with_dialog)
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False

CORNERS = [(60.0, 50.0), (580.0, 60.0), (570.0, 430.0), (70.0, 420.0)]


def _manager():
    manager = ROIManager(source_dimensions=TEMPLATE_DIMS)
    manager.add_roi("arm_a", np.array([[280, 60], [360, 60], [360, 200], [280, 200]]))
    manager.add_roi("arm_b", np.array([[80, 200], [260, 200], [260, 280], [80, 280]]))
    manager.set_reference_line((80, 240), (260, 240), "User-drawn reference line")
    return manager


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class RoiAlignmentDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.template = arena_image()

    def _dialog(self, truth, dims=TEMPLATE_DIMS, auto=True):
        target = warp(self.template, truth, dims)
        auto_align = None
        if auto:
            def auto_align(should_cancel):
                return ra.auto_register(self.template, target)
        dialog = RoiAlignmentDialog(
            _manager(), self.template, target, auto_align,
            template_name="template.mp4", target_name="mouse2.mp4",
        )
        self.addCleanup(dialog.deleteLater)
        return dialog

    def test_same_size_starts_unchanged(self):
        dialog = self._dialog(ra.identity())
        self.assertTrue(ra.is_identity(dialog.get_matrix()))
        self.assertEqual(dialog.get_method(), METHOD_SCALE)
        self.assertEqual(dialog.get_roi_manager().to_dict(), _manager().to_dict())

    def test_different_size_starts_scaled_to_the_new_resolution(self):
        dialog = self._dialog(rescale(ra.identity(), 0.5), dims=(320, 240))
        np.testing.assert_allclose(dialog.get_matrix(), np.diag([0.5, 0.5, 1.0]))
        aligned = dialog.get_roi_manager()
        self.assertEqual(aligned.source_dimensions, (320, 240))
        self.assertAlmostEqual(aligned.reference_length_pixels, 90.0)

    def test_auto_align_button_finds_the_move_and_reports_quality(self):
        truth = camera_move(shift=(18.0, -12.0), rotation_deg=3.0)
        dialog = self._dialog(truth)
        self.assertTrue(dialog.auto_button.isEnabled())
        dialog.auto_button.click()
        self.assertEqual(dialog.get_method(), METHOD_AUTO)
        self.assertLess(max_grid_error(dialog.get_matrix(), truth), 1.0)
        self.assertEqual(dialog.quality_level, "good")
        self.assertIn("good:", dialog.quality_text)
        self.assertIn(" px", dialog.quality_text)
        moved = dialog.get_roi_manager()
        expected = ra.transform_roi_manager(_manager(), truth, TEMPLATE_DIMS)
        diff = moved.rois["arm_a"][0].points - expected.rois["arm_a"][0].points
        self.assertLessEqual(int(np.max(np.abs(diff))), 1)

    def test_auto_align_disabled_without_a_provider(self):
        dialog = self._dialog(ra.identity(), auto=False)
        self.assertFalse(dialog.auto_button.isEnabled())

    def test_failed_auto_align_keeps_the_current_alignment(self):
        dialog = self._dialog(ra.identity())
        before = dialog.get_matrix()
        blank = np.full((480, 640), 128, dtype=np.uint8)
        dialog.apply_registration(ra.auto_register(blank, blank))
        np.testing.assert_array_equal(dialog.get_matrix(), before)
        self.assertEqual(dialog.quality_level, "poor")
        self.assertIn("Auto-align failed", dialog.quality_text)

    def test_landmarks_recover_the_move_and_show_residual(self):
        truth = camera_move(shift=(25.0, 10.0), rotation_deg=-4.0, scale=1.05)
        dialog = self._dialog(truth)
        dialog.start_landmarks()
        self.assertTrue(dialog.landmark_mode)
        moved = ra.transform_points(truth, CORNERS)
        for src, dst in zip(CORNERS, moved):
            dialog.add_template_landmark(*src)
            dialog.add_target_landmark(*dst)
        self.assertEqual(dialog.get_method(), METHOD_LANDMARKS)
        self.assertLess(max_grid_error(dialog.get_matrix(), truth), 0.01)
        self.assertLess(dialog.get_residual(), 0.01)
        self.assertEqual(dialog.quality_level, "good")

    def test_exactly_enough_landmarks_warn_that_the_fit_is_unverified(self):
        truth = camera_move(shift=(5.0, 5.0))
        dialog = self._dialog(truth)
        dialog.start_landmarks()
        for src, dst in zip(CORNERS[:2], ra.transform_points(truth, CORNERS[:2])):
            dialog.add_template_landmark(*src)
            dialog.add_target_landmark(*dst)
        self.assertIn("add one more", dialog.quality_text)

    def test_noisy_landmarks_report_a_poor_residual(self):
        dialog = self._dialog(ra.identity())
        dialog.start_landmarks()
        offsets = [(0, 0), (9, -8), (-10, 7), (8, 9)]
        for src, (dx, dy) in zip(CORNERS, offsets):
            dialog.add_template_landmark(*src)
            dialog.add_target_landmark(src[0] + dx, src[1] + dy)
        self.assertGreater(dialog.get_residual(), 4.0)
        self.assertEqual(dialog.quality_level, "poor")
        self.assertIn("check carefully", dialog.quality_text)

    def test_target_click_without_template_click_is_ignored(self):
        dialog = self._dialog(ra.identity())
        dialog.start_landmarks()
        dialog.add_target_landmark(10, 10)
        self.assertEqual(dialog.landmarks, [])

    def test_mirrored_landmarks_are_reported_not_applied(self):
        dialog = self._dialog(ra.identity())
        dialog.mode_combo.setCurrentIndex(1)  # affine
        dialog.start_landmarks()
        for src in CORNERS[:3]:
            dialog.add_template_landmark(*src)
            dialog.add_target_landmark(640 - src[0], src[1])
        self.assertEqual(dialog.quality_level, "poor")
        self.assertNotEqual(dialog.get_method(), METHOD_LANDMARKS)

    def test_clicks_on_scaled_down_frames_map_to_source_pixels(self):
        big = cv2.resize(self.template, (1280, 960))
        manager = ROIManager(source_dimensions=(1280, 960))
        manager.add_roi("z", np.array([[100, 100], [400, 100], [400, 400]]))
        dialog = RoiAlignmentDialog(manager, big, big)
        self.addCleanup(dialog.deleteLater)
        dialog.start_landmarks()
        # 1280 px shown at 620 px: factor 1280/620.
        QTest.mouseClick(
            dialog._template.label, Qt.MouseButton.LeftButton, pos=QPoint(310, 155)
        )
        QTest.mouseClick(
            dialog._target.label, Qt.MouseButton.LeftButton, pos=QPoint(62, 31)
        )
        (tx, ty), (dx, dy) = dialog.landmarks[0]
        scale = 1280 / 620
        self.assertAlmostEqual(tx, 310 * scale, places=3)
        self.assertAlmostEqual(ty, 155 * (960 / 465), places=3)
        self.assertAlmostEqual(dx, 62 * scale, places=3)

    def test_clicks_are_ignored_outside_landmark_mode(self):
        dialog = self._dialog(ra.identity())
        QTest.mouseClick(
            dialog._template.label, Qt.MouseButton.LeftButton, pos=QPoint(20, 20)
        )
        self.assertEqual(dialog.landmarks, [])
        self.assertIsNone(dialog._pending_template_point)

    def test_use_as_is_keeps_pixels_but_adopts_target_dimensions(self):
        dialog = self._dialog(rescale(ra.identity(), 2.0), dims=(1280, 960))
        dialog.use_as_is()
        self.assertEqual(dialog.get_method(), METHOD_AS_IS)
        aligned = dialog.get_roi_manager()
        self.assertEqual(aligned.source_dimensions, (1280, 960))
        np.testing.assert_array_equal(
            aligned.rois["arm_a"][0].points, _manager().rois["arm_a"][0].points
        )

    def test_clear_landmarks(self):
        dialog = self._dialog(ra.identity())
        dialog.start_landmarks()
        dialog.add_template_landmark(10, 10)
        dialog.add_target_landmark(12, 12)
        dialog.clear_button.click()
        self.assertEqual(dialog.landmarks, [])

    def test_helper_returns_choice_on_accept_and_none_on_cancel(self):
        target = warp(self.template, camera_move(shift=(4, 4)), TEMPLATE_DIMS)

        def accept(dialog):
            dialog.use_as_is()
            return 1

        with patch.object(RoiAlignmentDialog, "exec", accept):
            choice = align_template_with_dialog(
                None, _manager(), self.template, target
            )
        self.assertEqual(choice.method, METHOD_AS_IS)
        self.assertTrue(ra.is_identity(choice.matrix))

        with patch.object(RoiAlignmentDialog, "exec", lambda dialog: 0):
            self.assertIsNone(align_template_with_dialog(
                None, _manager(), self.template, target
            ))


if __name__ == "__main__":
    unittest.main()
