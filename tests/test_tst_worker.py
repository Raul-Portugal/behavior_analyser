import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

try:
    import cv2

    from core.models import BatchSettings, DetectionConfig
    from core.video import VideoReadError
    from gui.worker import AnalysisWorker
    from tests import worker_support as support
    from mazes.tst import TST
    from roi_manager import ROIManager
    from tests import tst_synthetic as synthetic
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


def _roi_manager():
    manager = ROIManager(source_dimensions=synthetic.BASE_SIZE)
    manager.add_roi("mouse_1", synthetic.clip_roi_points())
    return manager


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class TSTWorkerTests(unittest.TestCase):
    def test_reader_is_released_when_the_first_read_raises(self):
        reader = Mock()
        reader.read.side_effect = VideoReadError("decode stalled")
        handler = Mock(fps=30.0)
        settings = BatchSettings(
            _roi_manager(), DetectionConfig(), 0.0, per_roi_times={}
        )
        with tempfile.TemporaryDirectory() as directory:
            pipeline, _ = support.make_pipeline(
                Path(directory), TST(), settings
            )
            with patch("core.pipeline.BufferedVideoReader", return_value=reader):
                with self.assertRaises(VideoReadError):
                    pipeline._run_tst_analysis(
                        Path("unused.avi"), settings, 0, 100, handler,
                    )
        reader.release.assert_called_once()

    def test_short_video_reports_early_end_and_truncated_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "short_tst.avi"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0,
                synthetic.BASE_SIZE,
            )
            for frame in synthetic.render_clip(30.0, duration=3.0):
                writer.write(frame)
            writer.release()

            settings = BatchSettings(
                _roi_manager(), DetectionConfig(), 0.0,
                start_time=0.0, end_time=20.0,
                per_roi_times={"mouse_1": (0.0, 20.0)},
                paradigm_id="Tail Suspension Test",
                algorithm_parameters=TST().get_algorithm_parameters(),
            )
            worker = AnalysisWorker(
                {video_path: settings}, root, TST()
            )
            outcomes = []
            worker.finished.connect(outcomes.append)
            worker.run()

            self.assertEqual(len(outcomes), 1)
            self.assertEqual(outcomes[0].failures, [])
            messages = [message for _, message in outcomes[0].quality_warnings]
            self.assertTrue(
                any("analysis ended early at frame" in m for m in messages),
                messages,
            )
            self.assertTrue(
                any(
                    m.startswith("mouse_1: requested window 0-20 s but video "
                                 "covers 0-")
                    for m in messages
                ),
                messages,
            )


if __name__ == "__main__":
    unittest.main()
