"""ROI point tests hand OpenCV plain floats (OpenCV 5 rejects numpy scalars)."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from roi_manager import ROI, ROIManager


def square(x, y, size=40):
    return np.array([[x, y], [x + size, y], [x + size, y + size], [x, y + size]])


def strict_point_in_polygon(contour, point, measure_dist):
    """What OpenCV 5 does: the point must be a pair of Python floats."""
    if not all(type(value) is float for value in point):
        raise cv2.error("Error in point-in-polygon test: not Python floats")
    return real_point_polygon_test(contour, point, measure_dist)


real_point_polygon_test = cv2.pointPolygonTest


class OverlapTests(unittest.TestCase):
    def test_overlapping_integer_polygons_are_found(self):
        a, b = ROI(square(0, 0), "a"), ROI(square(20, 20), "b")
        with patch.object(cv2, "pointPolygonTest", strict_point_in_polygon):
            self.assertTrue(a.overlaps_with(b))
            self.assertTrue(b.overlaps_with(a))

    def test_separate_polygons_do_not_overlap(self):
        a, b = ROI(square(0, 0), "a"), ROI(square(100, 100), "b")
        with patch.object(cv2, "pointPolygonTest", strict_point_in_polygon):
            self.assertFalse(a.overlaps_with(b))

    def test_numpy_points_are_accepted_by_contains_point(self):
        roi = ROI(square(0, 0), "a")
        with patch.object(cv2, "pointPolygonTest", strict_point_in_polygon):
            self.assertTrue(roi.contains_point((np.int64(10), np.int32(10))))
            self.assertFalse(roi.contains_point((np.float32(90.5), np.int64(10))))
            self.assertTrue(roi.contains_point((10, 10)))
            self.assertTrue(roi.contains_point(np.array([10, 10])))

    def test_a_warning_is_not_swallowing_the_overlap_any_more(self):
        a, b = ROI(square(0, 0), "a"), ROI(square(20, 20), "b")
        with patch.object(cv2, "pointPolygonTest", strict_point_in_polygon), \
                self.assertNoLogs("roi_manager", level="WARNING"):
            a.overlaps_with(b)

    def test_the_manager_validation_warns_about_overlapping_zones_of_a_category(self):
        manager = ROIManager()
        manager.add_roi("arm_a", square(0, 0))
        manager.add_roi("arm_a", square(20, 20))
        with patch.object(cv2, "pointPolygonTest", strict_point_in_polygon), \
                self.assertLogs("roi_manager", level="WARNING") as logs:
            manager.validate_rois()
        self.assertTrue(any("overlap" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
