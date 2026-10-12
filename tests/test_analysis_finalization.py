import tempfile
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np

    from core.analysis_engine import Analyzer, apply_scale_to_result
    from core.detection import DetectionEngine
    from core.models import DetectionConfig
    from mazes.base_maze import BaseAnalysisResult
    from roi_manager import ROIManager
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


def _write_video(path: Path, width: int = 100, height: int = 100):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (width, height)
    )
    for _ in range(5):
        writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class FinalizeAndScaleTests(unittest.TestCase):
    def _finalized_result(self):
        # Jump limit is 0.2 * hypot(100, 100) = 28.28 px.
        result = BaseAnalysisResult(
            positions=[(0.0, 0.0), (3.0, 4.0), (80.0, 4.0), None, (10.0, 10.0)],
            roi_labels=["center", "center", "arm", "undetected", "center"],
            detection_confidences=[0.9, 0.9, 0.9, 0.0, 0.9],
        )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            video = Path(tmp) / "tiny.avi"
            _write_video(video)
            engine = DetectionEngine(
                np.zeros((100, 100), dtype=np.uint8), DetectionConfig()
            )
            analyzer = Analyzer(video, engine, ROIManager((100, 100)))
            analyzer.finalize_result(result)
        return result

    def test_finalize_result_pixel_metrics(self):
        result = self._finalized_result()

        self.assertAlmostEqual(result.total_distance, 5.0)
        self.assertAlmostEqual(result.distance_in_roi["center"], 5.0)
        self.assertNotIn("arm", result.distance_in_roi)
        self.assertAlmostEqual(result.time_in_roi["center"], 0.3)
        self.assertAlmostEqual(result.time_in_roi["arm"], 0.1)
        self.assertEqual(result.undetected_frames, 1)
        self.assertAlmostEqual(result.undetected_time_s, 0.1)
        self.assertEqual(result.excluded_jump_count, 1)
        self.assertAlmostEqual(result.excluded_jump_distance, 77.0)
        self.assertEqual(result.distance_unit, "pixels")

    def test_apply_scale_converts_every_distance_but_not_counts_or_time(self):
        result = self._finalized_result()

        apply_scale_to_result(result, 0.5)

        self.assertAlmostEqual(result.total_distance, 2.5)
        self.assertAlmostEqual(result.distance_in_roi["center"], 2.5)
        self.assertAlmostEqual(result.excluded_jump_distance, 38.5)
        self.assertEqual(result.distance_unit, "cm")
        self.assertEqual(result.excluded_jump_count, 1)
        self.assertEqual(result.undetected_frames, 1)
        self.assertAlmostEqual(result.undetected_time_s, 0.1)
        self.assertAlmostEqual(result.time_in_roi["center"], 0.3)
        self.assertAlmostEqual(result.time_in_roi["arm"], 0.1)

    def test_non_positive_scale_leaves_pixels_untouched(self):
        result = self._finalized_result()

        apply_scale_to_result(result, 0.0)

        self.assertAlmostEqual(result.excluded_jump_distance, 77.0)
        self.assertEqual(result.distance_unit, "pixels")


if __name__ == "__main__":
    unittest.main()
