import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import cv2
    import numpy as np
    from PyQt6.QtWidgets import QApplication

    from core.analysis_engine import Analyzer
    from core.analysis_geometry import AnalysisGeometry
    from core.detection import DetectionEngine
    from core.models import DetectionConfig
    from gui.tuner_dialog import TunerDialog
    from roi_manager import ROIManager
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


def _write_static_video(path: Path, width: int, height: int, frames: int):
    """Write identical intra-coded frames so every decode is identical."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (width, height)
    )
    frame = np.full((height, width, 3), 40, dtype=np.uint8)
    cv2.rectangle(
        frame,
        (width // 2 - 60, height // 2 - 40),
        (width // 2 + 60, height // 2 + 40),
        (255, 255, 255),
        -1,
    )
    for _ in range(frames):
        writer.write(frame)
    writer.release()


def _make_detect_spy(calls):
    """Wrap DetectionEngine.detect, recording the frame and reference."""
    original = DetectionEngine.detect

    def spy(engine, frame, use_last_pos=True):
        calls.append((frame.copy(), engine.ref_frame.copy()))
        return original(engine, frame, use_last_pos)

    return spy


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class AnalysisGeometryTests(unittest.TestCase):
    def test_downscale_dims_ratio_and_point_mapping(self):
        geometry = AnalysisGeometry(1920, 1080, 640)

        self.assertTrue(geometry.is_scaled)
        self.assertEqual(geometry.analysis_dims, (640, 360))
        self.assertAlmostEqual(geometry.scale_ratio, 3.0)
        self.assertEqual(geometry.to_source((0, 0)), (1.0, 1.0))
        x, y = geometry.to_source((10.5, 20.25))
        self.assertAlmostEqual(x, 11.0 * 3.0 - 0.5)
        self.assertAlmostEqual(y, 20.75 * 3.0 - 0.5)
        self.assertIsNone(geometry.to_source(None))
        back = geometry.to_analysis_point((x, y))
        self.assertAlmostEqual(back[0], 10.5)
        self.assertAlmostEqual(back[1], 20.25)

    def test_narrow_video_is_a_no_op(self):
        geometry = AnalysisGeometry(640, 480, 640)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ref = np.zeros((480, 640), dtype=np.uint8)

        self.assertFalse(geometry.is_scaled)
        self.assertEqual(geometry.analysis_dims, (640, 480))
        self.assertEqual(geometry.scale_ratio, 1.0)
        self.assertIs(geometry.to_analysis(frame), frame)
        self.assertIs(geometry.reference_to_analysis(ref), ref)
        self.assertEqual(geometry.to_source((12.5, 3.0)), (12.5, 3.0))

    def test_reference_already_at_analysis_size_is_untouched(self):
        geometry = AnalysisGeometry(1920, 1080, 640)
        ref = np.zeros((360, 640), dtype=np.uint8)

        self.assertIs(geometry.reference_to_analysis(ref), ref)
        full = np.zeros((1080, 1920), dtype=np.uint8)
        self.assertEqual(
            geometry.reference_to_analysis(full).shape, (360, 640)
        )


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class TunerProductionParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _run_both_paths(self, width: int, height: int):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "static.avi"
            _write_static_video(video, width, height, 4)
            reference = np.full((height, width), 40, dtype=np.uint8)
            roi_manager = ROIManager(source_dimensions=(width, height))
            config = DetectionConfig(
                use_weighting=False, min_detection_confidence=0.0
            )

            calls = []
            with patch.object(
                DetectionEngine, "detect", _make_detect_spy(calls)
            ):
                engine = DetectionEngine(reference.copy(), config)
                analyzer = Analyzer(video, engine, roi_manager, 0, 2)
                list(analyzer.process_frames())
                production = calls[-1]

                calls.clear()
                dialog = TunerDialog(
                    str(video), reference.copy(), roi_manager
                )
                try:
                    dialog.update_detection()
                    tuner = calls[-1]
                finally:
                    dialog.done(0)
        return production, tuner

    def test_tuner_and_production_feed_identical_inputs_1080p(self):
        production, tuner = self._run_both_paths(1920, 1080)

        self.assertEqual(production[0].shape[:2], (360, 640))
        self.assertEqual(production[1].shape, (360, 640))
        np.testing.assert_array_equal(production[0], tuner[0])
        np.testing.assert_array_equal(production[1], tuner[1])

    def test_640_wide_video_is_not_resized(self):
        production, tuner = self._run_both_paths(640, 360)

        self.assertEqual(production[0].shape[:2], (360, 640))
        np.testing.assert_array_equal(production[0], tuner[0])
        np.testing.assert_array_equal(production[1], tuner[1])


if __name__ == "__main__":
    unittest.main()
