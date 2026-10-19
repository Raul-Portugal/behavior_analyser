import copy
import json
import unittest

import cv2
import numpy as np

from core import roi_alignment as ra
from roi_manager import ROIManager
from tests.alignment_support import (TEMPLATE_DIMS, arena_image, camera_move,
                                     max_grid_error, rescale, warp)


def _template_manager(with_line=True, with_arena=False):
    manager = ROIManager(source_dimensions=TEMPLATE_DIMS)
    manager.add_roi("arm_a", np.array([[280, 60], [360, 60], [360, 200], [280, 200]]))
    manager.add_roi("arm_b", np.array([[80, 200], [260, 200], [260, 280], [80, 280]]))
    if with_arena:
        manager.set_arena(np.array([[60, 40], [580, 40], [580, 440], [60, 440]]))
    if with_line:
        manager.set_reference_line((80, 240), (260, 240), "User-drawn reference line")
    else:
        manager.set_reference_length(180.0, "User-drawn reference line")
    return manager


def _roi_dict_bytes(manager):
    return json.dumps(manager.to_dict(), sort_keys=True).encode()


class EstimateTransformTests(unittest.TestCase):
    PTS = np.array([[100, 80], [500, 90], [480, 400], [120, 380]], dtype=float)

    def _moved(self, matrix):
        return ra.transform_points(matrix, self.PTS)

    def test_similarity_recovers_exact_move(self):
        truth = camera_move(shift=(12.5, -7.0), rotation_deg=4.0, scale=1.1)
        matrix, rms = ra.estimate_transform(self.PTS, self._moved(truth), "similarity")
        np.testing.assert_allclose(matrix, truth, atol=1e-6)
        self.assertLess(rms, 1e-6)

    def test_similarity_works_with_two_points(self):
        truth = camera_move(shift=(3, 4), rotation_deg=-8.0, scale=0.9)
        matrix, rms = ra.estimate_transform(
            self.PTS[:2], ra.transform_points(truth, self.PTS[:2]), "similarity"
        )
        np.testing.assert_allclose(matrix, truth, atol=1e-6)
        self.assertLess(rms, 1e-6)

    def test_similarity_residual_reports_click_noise(self):
        truth = camera_move(shift=(10, 10), rotation_deg=2.0)
        rng = np.random.default_rng(3)
        noisy = self._moved(truth) + rng.normal(0, 1.5, self.PTS.shape)
        matrix, rms = ra.estimate_transform(self.PTS, noisy, "similarity")
        self.assertGreater(rms, 0.3)
        self.assertLess(rms, 3.0)
        self.assertLess(max_grid_error(matrix, truth), 6.0)

    def test_affine_with_three_and_more_points(self):
        truth = np.array([[1.1, 0.15, 20.0], [-0.05, 0.9, -10.0], [0, 0, 1.0]])
        for count in (3, 4):
            matrix, rms = ra.estimate_transform(
                self.PTS[:count], ra.transform_points(truth, self.PTS[:count]), "affine"
            )
            np.testing.assert_allclose(matrix, truth, atol=1e-3)
            self.assertLess(rms, 1e-3)

    def test_homography_with_four_points(self):
        truth = np.array([[1.0, 0.02, 15.0], [0.01, 1.0, -9.0], [1e-5, 2e-5, 1.0]])
        matrix, rms = ra.estimate_transform(
            self.PTS, ra.transform_points(truth, self.PTS), "homography"
        )
        self.assertLess(rms, 1e-3)
        self.assertLess(max_grid_error(matrix, truth), 1e-2)

    def test_too_few_points_is_an_error(self):
        for mode, count in (("similarity", 1), ("affine", 2), ("homography", 3)):
            with self.assertRaises(ValueError):
                ra.estimate_transform(self.PTS[:count], self.PTS[:count], mode)

    def test_unknown_mode_and_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            ra.estimate_transform(self.PTS, self.PTS, "perspective")
        with self.assertRaises(ValueError):
            ra.estimate_transform(self.PTS, self.PTS[:3], "similarity")

    def test_collinear_and_coincident_points_are_refused(self):
        line = np.array([[0, 0], [10, 10], [20, 20], [30, 30]], dtype=float)
        with self.assertRaises(ValueError):
            ra.estimate_transform(line, line, "affine")
        with self.assertRaises(ValueError):
            ra.estimate_transform(np.ones((3, 2)), self.PTS[:3], "similarity")

    def test_mirrored_landmarks_are_refused(self):
        mirrored = self.PTS * np.array([-1.0, 1.0]) + [700, 0]
        with self.assertRaises(ValueError):
            ra.estimate_transform(self.PTS, mirrored, "affine")

    def test_describe_residual_bands(self):
        self.assertEqual(ra.describe_residual(0.8), ("good", "good: 0.8 px"))
        self.assertEqual(ra.describe_residual(3.0)[0], "check")
        level, text = ra.describe_residual(6.0)
        self.assertEqual(level, "poor")
        self.assertEqual(text, "check carefully: 6.0 px")

    def test_matrix_helpers(self):
        truth = camera_move(shift=(5, 6), rotation_deg=30.0, scale=2.0)
        scale, angle, tx, ty = ra.decompose_similarity(truth)
        self.assertAlmostEqual(scale, 2.0)
        self.assertAlmostEqual(angle, -30.0)  # OpenCV rotates counter-clockwise
        self.assertTrue(ra.is_similarity(truth))
        self.assertFalse(ra.is_similarity(np.diag([1.0, 2.0, 1.0])))
        self.assertTrue(ra.is_identity(ra.identity()))
        self.assertEqual(ra.displacement_px(ra.identity(), (640, 480)), 0.0)
        self.assertAlmostEqual(
            ra.displacement_px(camera_move(shift=(3, 4)), (640, 480)), 5.0
        )


class TransformRoiManagerTests(unittest.TestCase):
    def test_zones_arena_line_and_dimensions_follow_the_transform(self):
        manager = _template_manager(with_arena=True)
        # Camera moved, and the target video has twice the resolution.
        truth = rescale(
            camera_move(shift=(14.0, -9.0), rotation_deg=3.0, scale=1.05), 2.0
        )
        moved = ra.transform_roi_manager(manager, truth, (1280, 960))

        self.assertEqual(moved.source_dimensions, (1280, 960))
        for category, rois in manager.rois.items():
            for original, new in zip(rois, moved.rois[category]):
                expected = ra.transform_points(truth, original.points)
                self.assertLessEqual(float(np.max(np.abs(new.points - expected))), 0.5)
                self.assertEqual(new.points.dtype, np.int32)
        expected_arena = ra.transform_points(truth, manager.arena.points)
        self.assertLessEqual(
            float(np.max(np.abs(moved.arena.points - expected_arena))), 0.5
        )

        p1, p2 = ra.transform_points(truth, manager.reference_line)
        self.assertEqual(moved.reference_line, (tuple(p1), tuple(p2)))
        self.assertAlmostEqual(
            moved.reference_length_pixels, float(np.linalg.norm(p2 - p1))
        )
        # 180 px line, scale 2.1 -> 378 px.
        self.assertAlmostEqual(moved.reference_length_pixels, 378.0, places=6)
        self.assertFalse(moved.reference_is_approximate)
        self.assertEqual(moved.reference_name, manager.reference_name)

    def test_calibration_agrees_with_ground_truth(self):
        manager = _template_manager()
        truth = rescale(camera_move(scale=1.0), 0.5)
        moved = ra.transform_roi_manager(manager, truth, (320, 240))
        scale = ra.scale_factor_after_alignment(0.02, manager, moved)
        # The same physical line is now half as many pixels, so each pixel
        # spans twice the distance: 0.02 cm/px -> 0.04 cm/px.
        self.assertAlmostEqual(scale, 0.04)
        self.assertAlmostEqual(180.0 * 0.02, moved.reference_length_pixels * scale)

    def test_length_only_reference_is_scaled_and_flagged(self):
        manager = _template_manager(with_line=False)
        moved = ra.transform_roi_manager(
            manager, rescale(camera_move(shift=(5, 5)), 0.5), (320, 240)
        )
        self.assertIsNone(moved.reference_line)
        self.assertAlmostEqual(moved.reference_length_pixels, 90.0)
        self.assertTrue(moved.reference_is_approximate)
        self.assertIn("approximate", moved.reference_name)
        # Survives serialization.
        restored = ROIManager.from_dict(json.loads(json.dumps(moved.to_dict())))
        self.assertTrue(restored.reference_is_approximate)

    def test_length_only_reference_is_exact_for_pure_shift(self):
        manager = _template_manager(with_line=False)
        moved = ra.transform_roi_manager(
            manager, camera_move(shift=(8, -6), rotation_deg=2.0), TEMPLATE_DIMS
        )
        self.assertAlmostEqual(moved.reference_length_pixels, 180.0)
        self.assertFalse(moved.reference_is_approximate)

    def test_identity_alignment_is_byte_identical(self):
        for factory in (
            lambda: _template_manager(with_line=True),
            lambda: _template_manager(with_line=False),
            lambda: _template_manager(with_arena=True),
            lambda: ROIManager(source_dimensions=TEMPLATE_DIMS),
        ):
            manager = factory()
            moved = ra.transform_roi_manager(manager, ra.identity(), TEMPLATE_DIMS)
            self.assertEqual(_roi_dict_bytes(moved), _roi_dict_bytes(manager))

    def test_legacy_manager_without_new_keys_stays_without_them(self):
        manager = _template_manager(with_line=False)
        moved = ra.transform_roi_manager(
            manager, camera_move(shift=(4, 4)), TEMPLATE_DIMS
        )
        self.assertNotIn("reference_line", moved.to_dict())

    def test_input_manager_is_not_modified(self):
        manager = _template_manager(with_arena=True)
        before = _roi_dict_bytes(manager)
        snapshot = copy.deepcopy(manager)
        ra.transform_roi_manager(manager, camera_move(shift=(30, 30), scale=1.3), (900, 700))
        self.assertEqual(_roi_dict_bytes(manager), before)
        self.assertEqual(manager.source_dimensions, snapshot.source_dimensions)

    def test_manager_without_calibration_stays_uncalibrated(self):
        manager = ROIManager(source_dimensions=TEMPLATE_DIMS)
        manager.add_roi("z", np.array([[10, 10], [200, 10], [200, 200]]))
        moved = ra.transform_roi_manager(manager, camera_move(shift=(5, 5)), TEMPLATE_DIMS)
        self.assertFalse(moved.has_reference)
        self.assertEqual(moved.reference_name, manager.reference_name)

    def test_scale_factor_without_calibration_is_unchanged(self):
        manager = ROIManager(source_dimensions=TEMPLATE_DIMS)
        self.assertEqual(ra.scale_factor_after_alignment(0.0, manager, manager), 0.0)
        self.assertEqual(ra.scale_factor_after_alignment(0.05, manager, manager), 0.05)


class AutoRegisterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = arena_image()

    def _register(self, truth, dims=TEMPLATE_DIMS):
        target = warp(self.template, truth, dims)
        return ra.auto_register(
            cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(target, cv2.COLOR_BGR2GRAY),
        )

    def test_identical_frames_give_identity(self):
        result = self._register(ra.identity())
        self.assertTrue(result.usable)
        self.assertEqual(result.verdict, "good")
        self.assertLess(max_grid_error(result.matrix, ra.identity()), 0.5)

    def test_recovers_a_shift(self):
        truth = camera_move(shift=(22.0, -15.0))
        result = self._register(truth)
        self.assertEqual(result.verdict, "good")
        self.assertLess(max_grid_error(result.matrix, truth), 1.0)
        self.assertLess(result.rms, 1.5)

    def test_recovers_a_rotation(self):
        truth = camera_move(shift=(-6.0, 9.0), rotation_deg=4.5)
        result = self._register(truth)
        self.assertTrue(result.usable)
        self.assertLess(max_grid_error(result.matrix, truth), 1.0)

    def test_recovers_scale_and_shift(self):
        truth = camera_move(shift=(10.0, 5.0), rotation_deg=-3.0, scale=1.08)
        result = self._register(truth)
        self.assertTrue(result.usable)
        self.assertLess(max_grid_error(result.matrix, truth), 1.0)

    def test_recovers_a_resolution_change(self):
        for factor, dims in ((0.5, (320, 240)), (1.5, (960, 720))):
            truth = rescale(camera_move(shift=(8.0, -4.0), rotation_deg=2.0), factor)
            result = self._register(truth, dims)
            self.assertTrue(result.usable, f"factor {factor}")
            self.assertLess(
                max_grid_error(result.matrix, truth), 1.0, f"factor {factor}"
            )

    def test_result_drives_roi_transform_within_a_pixel(self):
        truth = camera_move(shift=(18.0, 11.0), rotation_deg=-2.5)
        result = self._register(truth)
        manager = _template_manager()
        moved = ra.transform_roi_manager(manager, result.matrix, TEMPLATE_DIMS)
        expected = ra.transform_roi_manager(manager, truth, TEMPLATE_DIMS)
        for category in manager.rois:
            diff = moved.rois[category][0].points - expected.rois[category][0].points
            self.assertLessEqual(float(np.max(np.abs(diff))), 1.0)
        self.assertAlmostEqual(
            moved.reference_length_pixels, expected.reference_length_pixels, delta=1.0
        )

    def test_featureless_images_fail_cleanly(self):
        blank = np.full((480, 640), 128, dtype=np.uint8)
        result = ra.auto_register(blank, blank)
        self.assertFalse(result.usable)
        self.assertEqual(result.verdict, "failed")
        self.assertIsNone(result.matrix)

    def test_unrelated_images_are_not_called_good(self):
        other = arena_image(seed=99)
        result = ra.auto_register(
            cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(other, cv2.COLOR_BGR2GRAY),
        )
        self.assertNotEqual(result.verdict, "good")

    def test_accepts_color_frames(self):
        target = warp(self.template, camera_move(shift=(5, 5)), TEMPLATE_DIMS)
        result = ra.auto_register(self.template, target)
        self.assertEqual(result.verdict, "good")


if __name__ == "__main__":
    unittest.main()
