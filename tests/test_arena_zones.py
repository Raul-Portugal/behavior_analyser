import math
import unittest

import numpy as np

from core.analysis_geometry import AnalysisGeometry
from core.arena_zones import (
    CENTER,
    OUTSIDE,
    PERIPHERY,
    UNDETECTED,
    analysis_mask,
    center_polygon,
    center_polygons,
    center_threshold_for_area_fraction,
    classify_center_periphery,
    max_wall_distance,
    signed_wall_distance,
    valid_step_distances,
    wall_distances,
)

# 200 x 100 rectangle with its top-left corner at (100, 50).
RECT = np.array([[100, 50], [300, 50], [300, 150], [100, 150]])
# 200 x 200 square.
SQUARE = np.array([[10, 10], [210, 10], [210, 210], [10, 210]])


def _circle(cx, cy, radius, n=360):
    angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.stack(
        [cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1
    )


def _sampled_center_fraction(arena, threshold, count=20000, seed=3):
    """Fraction of uniformly sampled arena points that classify as center."""
    rng = np.random.default_rng(seed)
    low, high = np.min(arena, axis=0), np.max(arena, axis=0)
    points = rng.uniform(low, high, size=(count, 2))
    labels = classify_center_periphery(
        [tuple(p) for p in points], arena, threshold
    )
    inside = [label for label in labels if label != OUTSIDE]
    return sum(label == CENTER for label in inside) / len(inside)


class SignedWallDistanceTests(unittest.TestCase):
    def test_inside_is_positive_distance_to_nearest_wall(self):
        self.assertAlmostEqual(signed_wall_distance(RECT, (200, 100)), 50.0)
        self.assertAlmostEqual(signed_wall_distance(RECT, (120, 100)), 20.0)
        self.assertAlmostEqual(signed_wall_distance(RECT, (200, 60)), 10.0)

    def test_outside_is_negative_and_wall_is_zero(self):
        self.assertAlmostEqual(signed_wall_distance(RECT, (90, 100)), -10.0)
        self.assertAlmostEqual(signed_wall_distance(RECT, (100, 100)), 0.0)

    def test_none_position_is_refused(self):
        with self.assertRaises(ValueError):
            signed_wall_distance(RECT, None)

    def test_invalid_arenas_are_refused(self):
        for arena in (
            np.array([[0, 0], [10, 10]]),
            np.array([[0, 0], [10, 0], [20, 0]]),
            np.array([[0, 0], [10, np.nan], [20, 5]]),
        ):
            with self.assertRaises(ValueError):
                signed_wall_distance(arena, (1, 1))

    def test_wall_distances_keep_none_placeholders(self):
        result = wall_distances([(200, 100), None, (90, 100)], RECT)
        self.assertAlmostEqual(result[0], 50.0)
        self.assertIsNone(result[1])
        self.assertAlmostEqual(result[2], -10.0)


class CenterThresholdTests(unittest.TestCase):
    def test_square_matches_analytic_inset(self):
        side = 200.0
        for fraction in (0.1, 0.25, 0.5, 0.75):
            expected = side * (1.0 - math.sqrt(fraction)) / 2.0
            actual = center_threshold_for_area_fraction(SQUARE, fraction)
            self.assertAlmostEqual(actual, expected, delta=1.0, msg=fraction)

    def test_rectangle_matches_analytic_inset(self):
        # Inset d leaves (200 - 2d) * (100 - 2d); solve for the fraction.
        fraction = 0.25
        a, b = 200.0, 100.0
        # 4 d^2 - 2 (a + b) d + a b (1 - fraction) = 0, smaller root.
        qa, qb, qc = 4.0, -2.0 * (a + b), a * b * (1.0 - fraction)
        expected = (-qb - math.sqrt(qb * qb - 4 * qa * qc)) / (2 * qa)
        actual = center_threshold_for_area_fraction(RECT, fraction)
        self.assertAlmostEqual(actual, expected, delta=1.0)

    def test_circle_approximation_matches_analytic_inset(self):
        radius = 150.0
        arena = _circle(400.0, 300.0, radius)
        for fraction in (0.25, 0.5):
            expected = radius * (1.0 - math.sqrt(fraction))
            actual = center_threshold_for_area_fraction(arena, fraction)
            self.assertAlmostEqual(actual, expected, delta=1.5, msg=fraction)

    def test_threshold_agrees_with_point_classification(self):
        for arena in (SQUARE, RECT, _circle(200.0, 200.0, 120.0)):
            threshold = center_threshold_for_area_fraction(arena, 0.25)
            self.assertAlmostEqual(
                _sampled_center_fraction(arena, threshold), 0.25, delta=0.02
            )

    def test_larger_fraction_needs_smaller_threshold(self):
        thresholds = [
            center_threshold_for_area_fraction(SQUARE, f)
            for f in (0.1, 0.3, 0.6, 0.9)
        ]
        self.assertEqual(thresholds, sorted(thresholds, reverse=True))

    def test_translation_does_not_change_threshold(self):
        shifted = SQUARE + np.array([517, 233])
        self.assertAlmostEqual(
            center_threshold_for_area_fraction(SQUARE, 0.4),
            center_threshold_for_area_fraction(shifted, 0.4),
            delta=0.1,
        )

    def test_fraction_must_be_strictly_between_zero_and_one(self):
        for fraction in (0.0, 1.0, -0.1, 1.5, float("nan")):
            with self.assertRaises(ValueError):
                center_threshold_for_area_fraction(SQUARE, fraction)

    def test_max_wall_distance_is_inscribed_radius(self):
        self.assertAlmostEqual(max_wall_distance(SQUARE), 100.0, delta=1.0)
        self.assertAlmostEqual(max_wall_distance(RECT), 50.0, delta=1.0)
        self.assertAlmostEqual(
            max_wall_distance(_circle(0.0, 0.0, 80.0)), 80.0, delta=1.0
        )


class ClassificationTests(unittest.TestCase):
    def test_labels_for_square_arena(self):
        positions = [
            (110, 110),   # 100 from every wall
            (60, 60),     # 50 from the nearest wall
            (30, 110),    # 20 from the left wall
            (5, 110),     # outside
            None,         # undetected
            (10, 110),    # on the wall
        ]
        labels = classify_center_periphery(positions, SQUARE, 40.0)
        self.assertEqual(
            labels,
            [CENTER, CENTER, PERIPHERY, OUTSIDE, UNDETECTED, PERIPHERY],
        )

    def test_boundary_belongs_to_center_at_exact_threshold(self):
        # (60, 110) is exactly 50 from the left wall.
        self.assertEqual(
            classify_center_periphery([(60, 110)], SQUARE, 50.0), [CENTER]
        )
        self.assertEqual(
            classify_center_periphery([(60, 110)], SQUARE, 50.5), [PERIPHERY]
        )

    def test_zero_threshold_makes_every_inside_point_center(self):
        labels = classify_center_periphery([(10, 10), (100, 100)], SQUARE, 0.0)
        self.assertEqual(labels, [CENTER, CENTER])

    def test_threshold_beyond_inscribed_radius_leaves_no_center(self):
        labels = classify_center_periphery([(110, 110)], SQUARE, 500.0)
        self.assertEqual(labels, [PERIPHERY])

    def test_invalid_threshold_is_refused(self):
        for threshold in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                classify_center_periphery([(1, 1)], SQUARE, threshold)

    def test_empty_positions(self):
        self.assertEqual(classify_center_periphery([], SQUARE, 10.0), [])

    def test_concave_arena(self):
        # An L shape: the notch is outside.
        arena = np.array(
            [[0, 0], [200, 0], [200, 100], [100, 100], [100, 200], [0, 200]]
        )
        labels = classify_center_periphery(
            [(50, 50), (150, 150), (150, 50)], arena, 20.0
        )
        self.assertEqual(labels, [CENTER, OUTSIDE, CENTER])


class CenterPolygonTests(unittest.TestCase):
    def test_square_center_is_the_inset_square(self):
        outline = center_polygon(SQUARE, 50.0)
        self.assertIsNotNone(outline)
        self.assertEqual(outline.dtype, np.int32)
        low, high = outline.min(axis=0), outline.max(axis=0)
        np.testing.assert_allclose(low, [60, 60], atol=1.5)
        np.testing.assert_allclose(high, [160, 160], atol=1.5)

    def test_outline_points_lie_at_the_threshold_distance(self):
        arena = _circle(300.0, 300.0, 150.0)
        outline = center_polygon(arena, 60.0)
        distances = [signed_wall_distance(arena, tuple(p)) for p in outline]
        self.assertTrue(all(abs(d - 60.0) < 2.0 for d in distances), distances)

    def test_zero_threshold_covers_the_arena(self):
        outline = center_polygon(SQUARE, 0.0)
        np.testing.assert_allclose(outline.min(axis=0), [10, 10], atol=1.5)
        np.testing.assert_allclose(outline.max(axis=0), [210, 210], atol=1.5)

    def test_empty_center_gives_none(self):
        self.assertIsNone(center_polygon(SQUARE, 500.0))
        self.assertEqual(center_polygons(SQUARE, 500.0), [])

    def test_center_area_matches_requested_fraction(self):
        import cv2

        threshold = center_threshold_for_area_fraction(SQUARE, 0.25)
        outline = center_polygon(SQUARE, threshold)
        area = cv2.contourArea(outline.astype(np.float32))
        self.assertAlmostEqual(area / 40000.0, 0.25, delta=0.02)

    def test_concave_arena_can_split_the_center(self):
        # A dumbbell: two squares joined by a thin neck.
        arena = np.array([
            [0, 0], [100, 0], [100, 40], [200, 40], [200, 0], [300, 0],
            [300, 100], [200, 100], [200, 60], [100, 60], [100, 100], [0, 100],
        ])
        outlines = center_polygons(arena, 30.0)
        self.assertEqual(len(outlines), 2)
        self.assertIsNone(center_polygon(arena, 80.0))


class AnalysisMaskTests(unittest.TestCase):
    def test_unscaled_mask_matches_polygon(self):
        geometry = AnalysisGeometry(320, 240, 0)
        mask = analysis_mask(SQUARE, geometry)
        self.assertEqual(mask.shape, (240, 320))
        self.assertEqual(set(np.unique(mask)), {0, 255})
        self.assertEqual(mask[100, 100], 255)
        self.assertEqual(mask[5, 5], 0)
        self.assertEqual(mask[100, 215], 0)

    def test_scaled_mask_covers_the_same_region(self):
        geometry = AnalysisGeometry(640, 480, 320)
        mask = analysis_mask(SQUARE, geometry)
        self.assertEqual(mask.shape, (240, 320))
        # The square (10..210) maps to roughly (5..105) at half resolution.
        self.assertEqual(mask[50, 50], 255)
        self.assertEqual(mask[50, 120], 0)
        self.assertEqual(mask[110, 50], 0)
        inside_fraction = np.count_nonzero(mask) / (100.0 * 100.0)
        self.assertAlmostEqual(inside_fraction, 1.0, delta=0.06)


class ValidStepDistanceTests(unittest.TestCase):
    def test_gaps_break_the_trajectory(self):
        positions = [(0, 0), (3, 4), None, (10, 10), (13, 14)]
        valid = [True, True, False, True, True]
        steps = valid_step_distances(positions, valid, None)
        self.assertEqual(steps, [None, 5.0, None, None, 5.0])

    def test_jumps_over_the_limit_are_excluded(self):
        positions = [(0, 0), (3, 4), (300, 400), (303, 404)]
        steps = valid_step_distances(positions, [True] * 4, 100.0)
        self.assertEqual(steps, [None, 5.0, None, 5.0])

    def test_invalid_flag_breaks_even_with_a_position(self):
        positions = [(0, 0), (3, 4), (6, 8)]
        steps = valid_step_distances(positions, [True, False, True], None)
        self.assertEqual(steps, [None, None, None])


if __name__ == "__main__":
    unittest.main()
