"""Cancellation must be honoured outside the frame loops too."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np

    from core.video import ReferenceFrameGenerator, SafeVideoWriter
    from gui.worker import AnalysisWorker
    from mazes.freestyle import Freestyle
    from mazes.tst import TST
    from output import DataExporter, Visualizer
    from tests import worker_support as support
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class ReferenceFrameCancellationTests(unittest.TestCase):
    def test_generate_stops_between_samples(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            video = support.write_tracking_video(root, frames=40)
            polls = []

            def should_cancel():
                polls.append(1)
                return len(polls) > 3

            with patch.object(
                ReferenceFrameGenerator, "get_cache_dir",
                return_value=root / "cache",
            ):
                with self.assertRaises(InterruptedError):
                    ReferenceFrameGenerator.generate(
                        video, num_samples=20, should_cancel=should_cancel
                    )
                self.assertEqual(len(polls), 4)
                self.assertFalse(
                    ReferenceFrameGenerator.get_cache_path(video).exists()
                )

    def test_never_cancelling_callback_does_not_change_the_result(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            video = support.write_tracking_video(root, frames=20)
            with patch.object(
                ReferenceFrameGenerator, "get_cache_dir",
                return_value=root / "cache",
            ):
                plain = ReferenceFrameGenerator.generate(
                    video, num_samples=10, use_cache=False
                )
                polled = ReferenceFrameGenerator.generate(
                    video, num_samples=10, use_cache=False,
                    should_cancel=lambda: False,
                )
            np.testing.assert_array_equal(plain, polled)


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class WorkerCancellationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        cache = patch.object(
            ReferenceFrameGenerator, "get_cache_dir",
            return_value=self.root / "cache",
        )
        cache.start()
        self.addCleanup(cache.stop)

    def _make(self, maze, settings, video):
        out = self.root / "out"
        out.mkdir(exist_ok=True)
        worker = AnalysisWorker({video: settings}, out, maze)
        outcomes, logs = [], []
        worker.finished.connect(outcomes.append)
        worker.log.connect(logs.append)
        return worker, outcomes, logs

    def _assert_cancelled_cleanly(self, outcomes):
        self.assertEqual(len(outcomes), 1)
        self.assertTrue(outcomes[0].cancelled)
        self.assertEqual(outcomes[0].failures, [])
        self.assertEqual(outcomes[0].results, [])

    def test_cancel_during_reference_generation(self):
        video = support.write_tracking_video(self.root)
        worker, outcomes, _ = self._make(
            Freestyle(), support.tracking_settings(), video
        )

        def on_progress(_pct, _main, sub):
            if "reference frame" in sub:
                worker.stop()

        worker.detailed_progress.connect(on_progress)
        worker.run()

        self._assert_cancelled_cleanly(outcomes)
        self.assertFalse(
            ReferenceFrameGenerator.get_cache_path(video).exists()
        )
        # No tracking happened, so there is nothing to checkpoint.
        self.assertEqual(
            worker.checkpoint_mgr.find_all_checkpoints(), []
        )

    def test_cancel_during_validation_video_write(self):
        video = support.write_tst_video(self.root)
        worker, outcomes, _ = self._make(
            TST(), support.tst_settings(create_timelapse=True), video
        )
        writes = []

        class StoppingWriter(SafeVideoWriter):
            def write(self, frame):
                super().write(frame)
                writes.append(1)
                if len(writes) == 3:
                    worker.stop()

        with patch("gui.worker.SafeVideoWriter", StoppingWriter):
            worker.run()

        self._assert_cancelled_cleanly(outcomes)
        self.assertEqual(len(writes), 3, "writer kept going after cancel")
        self.assertEqual(
            list(worker.output_dir.rglob("*_validation.mp4")), [],
            "partial validation video should be removed",
        )

    def test_cancel_during_timelapse_write(self):
        video = support.write_tracking_video(self.root)
        worker, outcomes, _ = self._make(
            Freestyle(), support.tracking_settings(create_timelapse=True),
            video,
        )
        writes = []

        class StoppingWriter(SafeVideoWriter):
            def write(self, frame):
                super().write(frame)
                writes.append(1)
                if len(writes) == 2:
                    worker.stop()

        with patch("gui.worker.SafeVideoWriter", StoppingWriter):
            worker.run()

        self._assert_cancelled_cleanly(outcomes)
        self.assertEqual(len(writes), 2)
        self.assertEqual(list(worker.output_dir.rglob("*_timelapse.mp4")), [])

    def test_cancel_between_plot_jobs_skips_the_remaining_plots(self):
        video = support.write_tracking_video(self.root)
        worker, outcomes, _ = self._make(
            Freestyle(), support.tracking_settings(), video
        )

        def heatmap_then_stop(*args, **kwargs):
            worker.stop()

        with patch.object(
            Visualizer, "generate_heatmap", side_effect=heatmap_then_stop
        ), patch.object(Visualizer, "generate_trajectory_plot") as trajectory:
            worker.run()

        self._assert_cancelled_cleanly(outcomes)
        trajectory.assert_not_called()

    def test_cancel_between_export_steps_skips_the_remaining_exports(self):
        video = support.write_tracking_video(self.root)
        worker, outcomes, _ = self._make(
            Freestyle(), support.tracking_settings(), video
        )
        real_export = DataExporter.export_to_csv

        def export_then_stop(*args, **kwargs):
            real_export(*args, **kwargs)
            worker.stop()

        with patch.object(
            DataExporter, "export_to_csv", side_effect=export_then_stop
        ), patch.object(DataExporter, "export_summary_csv") as summary:
            worker.run()

        self._assert_cancelled_cleanly(outcomes)
        summary.assert_not_called()


if __name__ == "__main__":
    unittest.main()
