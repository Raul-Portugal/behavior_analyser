import csv
import json
import tempfile
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np

    from core.analysis_engine import Analyzer
    from core.detection import DetectionEngine
    from core.models import DetectionConfig
    from mazes.base_maze import BaseAnalysisResult
    from output import DataExporter
    from roi_manager import ROIManager
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


def _config(**overrides) -> "DetectionConfig":
    values = dict(
        threshold_percentile=95.0,
        use_weighting=False,
        min_component_area=20,
        morphology_kernel_size=3,
        min_detection_confidence=0.1,
    )
    values.update(overrides)
    return DetectionConfig(**values)


def _write_moving_blob_video(path: Path, width: int, height: int, count: int):
    """Blob moving 1 source pixel per frame; returns true centers."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (width, height)
    )
    centers = []
    for index in range(count):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        cx, cy = 600 + index, 500
        cv2.rectangle(
            frame, (cx - 60, cy - 60), (cx + 60, cy + 60), (255, 255, 255), -1
        )
        writer.write(frame)
        centers.append((float(cx), float(cy)))
    writer.release()
    return centers


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class SubPixelDetectionTests(unittest.TestCase):
    def _blob_scene(self):
        reference = np.zeros((100, 100), dtype=np.uint8)
        frame = reference.copy()
        # 20 px wide: the true centroid sits on a half pixel.
        cv2.rectangle(frame, (40, 45), (59, 64), 255, -1)
        return reference, frame

    def test_centroid_is_not_rounded_to_integers(self):
        reference, frame = self._blob_scene()

        position = DetectionEngine(reference, _config()).detect(frame).position

        self.assertIsInstance(position[0], float)
        self.assertAlmostEqual(position[0], 49.5, delta=0.01)
        self.assertAlmostEqual(position[1], 54.5, delta=0.01)

    def test_weighting_window_accepts_float_last_position(self):
        reference, frame = self._blob_scene()
        engine = DetectionEngine(reference, _config(use_weighting=True))

        first = engine.detect(frame)
        second = engine.detect(frame)

        self.assertTrue(first.detected and second.detected)
        self.assertEqual(engine.last_position, second.position)

    def test_state_round_trip_preserves_float_position_through_json(self):
        reference, frame = self._blob_scene()
        engine = DetectionEngine(reference, _config())
        engine.detect(frame)
        state = json.loads(json.dumps(engine.export_state()))

        restored = DetectionEngine(reference, _config())
        restored.restore_state(state)

        self.assertEqual(restored.last_position, engine.last_position)
        self.assertEqual(
            restored.consecutive_failures, engine.consecutive_failures
        )

    def test_restore_state_accepts_legacy_integer_positions(self):
        reference, _ = self._blob_scene()
        engine = DetectionEngine(reference, _config())

        engine.restore_state(
            {"last_position": [10, 20], "consecutive_failures": 2}
        )

        self.assertEqual(engine.last_position, (10.0, 20.0))
        self.assertEqual(engine.consecutive_failures, 2)

    def test_jump_check_uses_float_distance(self):
        reference, frame = self._blob_scene()
        # Limit is 0.1 * hypot(100, 100) = 14.14 px. The candidate centroid
        # (49.5, 54.5) is 14.0 px from the first last_position, 15.5 px from
        # the second.
        config = _config(max_expected_jump_ratio=0.1)
        engine = DetectionEngine(reference, config)
        engine.last_position = (35.5, 54.5)

        self.assertTrue(engine.detect(frame).detected)
        engine.last_position = (34.0, 54.5)
        self.assertFalse(engine.detect(frame).detected)


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class SubPixelAnalyzerTests(unittest.TestCase):
    def test_1080p_positions_are_float_and_not_quantized_to_ratio_steps(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            video = Path(tmp) / "moving.avi"
            centers = _write_moving_blob_video(video, 1920, 1080, 6)
            roi_manager = ROIManager(source_dimensions=(1920, 1080))
            roi_manager.add_roi(
                "center",
                np.array([[400, 300], [1000, 300], [1000, 700], [400, 700]]),
            )
            reference = np.full((1080, 1920), 40, dtype=np.uint8)
            engine = DetectionEngine(reference, _config(min_component_area=50))
            analyzer = Analyzer(video, engine, roi_manager, 0, 6)

            rows = list(analyzer.process_frames())

        positions = [pos for _, pos, _ in rows]
        self.assertTrue(all(pos is not None for pos in positions))
        for (x, y), (true_x, true_y) in zip(positions, centers):
            self.assertIsInstance(x, float)
            self.assertAlmostEqual(x, true_x, delta=1.5)
            self.assertAlmostEqual(y, true_y, delta=1.5)
        # With int centroids * 3 every x would be on a 3 px lattice.
        self.assertTrue(any(abs(x - round(x)) > 1e-6 for x, _ in positions))
        # Float positions still resolve ROI membership.
        self.assertTrue(all(zones == ["center"] for _, _, zones in rows))


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class SubPixelExportTests(unittest.TestCase):
    def test_tracking_csv_writes_two_decimal_coordinates(self):
        result = BaseAnalysisResult(
            positions=[(10.126, 20.5), None],
            timestamps=[0.0, 0.1],
            fps=10.0,
            roi_labels=["center", "undetected"],
            detection_confidences=[0.9, 0.0],
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.csv"
            DataExporter.export_to_csv(result, path)
            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual((rows[0]["x"], rows[0]["y"]), ("10.13", "20.50"))
        self.assertEqual((rows[1]["x"], rows[1]["y"]), ("", ""))


if __name__ == "__main__":
    unittest.main()
