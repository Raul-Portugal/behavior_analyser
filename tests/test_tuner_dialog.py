import tempfile
import time
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np
    from PyQt6.QtWidgets import QApplication

    from core.analysis_geometry import AnalysisGeometry
    from core.models import DetectionConfig
    from gui.tuner_dialog import (
        QuickScanWorker,
        TunerDialog,
        quick_scan_windows,
    )
    from roi_manager import ROIManager
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


def _write_video(path: Path, frames: int, width: int = 64, height: int = 48):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (width, height)
    )
    for index in range(frames):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        x = 10 + index % 40
        cv2.rectangle(frame, (x, 15), (x + 12, 30), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


class _CountingCapture:
    """Capture stand-in that serves solid frames and logs seeks/reads."""

    def __init__(self, frame_count=100):
        self.frame_count = frame_count
        self.position = 0
        self.seeks = []
        self.reads = 0
        self.released = False

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.seeks.append(int(value))
            self.position = int(value)
        return True

    def read(self):
        if self.position >= self.frame_count:
            return False, None
        self.position += 1
        self.reads += 1
        return True, np.full((48, 64, 3), 40, dtype=np.uint8)

    def release(self):
        self.released = True


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class TunerDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _dialog(self, directory: str, frames: int):
        video = Path(directory) / "clip.avi"
        _write_video(video, frames)
        return TunerDialog(
            str(video),
            np.full((48, 64), 40, dtype=np.uint8),
            ROIManager(source_dimensions=(64, 48)),
        )

    def test_initial_frame_is_rendered_on_open(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            dialog = self._dialog(tmp, 21)
            try:
                self.assertEqual(dialog.frame_slider.value(), 10)
                self.assertEqual(dialog.frame_label.text(), "Frame: 10 / 20")
                self.assertGreaterEqual(
                    dialog.quality_monitor.frames_processed, 1
                )
                self.assertFalse(dialog.main_view.pixmap().isNull())
            finally:
                dialog.done(0)

    def test_single_frame_video_still_renders(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            dialog = self._dialog(tmp, 1)
            try:
                self.assertEqual(dialog.frame_label.text(), "Frame: 0 / 0")
            finally:
                dialog.done(0)

    def test_accept_and_reject_release_capture_and_stop_timer(self):
        for finish in ("accept", "reject"):
            with self.subTest(finish=finish):
                with tempfile.TemporaryDirectory(
                    ignore_cleanup_errors=True
                ) as tmp:
                    dialog = self._dialog(tmp, 5)
                    self.assertTrue(dialog.cap.isOpened())
                    self.assertTrue(dialog.stats_timer.isActive())

                    getattr(dialog, finish)()

                    self.assertFalse(dialog.cap.isOpened())
                    self.assertFalse(dialog.stats_timer.isActive())

    def test_quick_scan_runs_in_background_and_disables_button(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            dialog = self._dialog(tmp, 40)
            try:
                dialog.run_quick_scan()
                self.assertFalse(dialog.test_scan_button.isEnabled())
                self.assertEqual(dialog.test_scan_button.text(), "Scanning...")

                deadline = time.monotonic() + 20.0
                while (
                    not dialog.test_scan_button.isEnabled()
                    and time.monotonic() < deadline
                ):
                    self.app.processEvents()
                    time.sleep(0.01)

                self.assertTrue(dialog.test_scan_button.isEnabled())
                self.assertEqual(dialog.quality_monitor.frames_processed, 45)
            finally:
                dialog.done(0)


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class QuickScanWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_windows_cover_start_middle_and_end(self):
        self.assertEqual(
            quick_scan_windows(100), [(0, 15), (43, 15), (85, 15)]
        )
        self.assertEqual(quick_scan_windows(9), [(0, 9)])

    def test_one_seek_per_window_then_sequential_reads(self):
        capture = _CountingCapture(100)
        worker = QuickScanWorker(
            "unused.mp4",
            np.full((48, 64), 40, dtype=np.uint8),
            AnalysisGeometry(64, 48, 640),
            DetectionConfig(),
            100,
            capture_factory=lambda _path: capture,
        )
        results = []
        worker.scan_complete.connect(results.append)

        worker.run()  # synchronous: exercise the body without a thread

        self.assertEqual(capture.seeks, [0, 43, 85])
        self.assertEqual(capture.reads, 45)
        self.assertEqual(len(results[0]), 45)
        self.assertTrue(capture.released)

    def test_capture_errors_are_reported_not_raised(self):
        def broken_factory(_path):
            raise OSError("cannot open")

        worker = QuickScanWorker(
            "unused.mp4",
            np.zeros((48, 64), dtype=np.uint8),
            AnalysisGeometry(64, 48, 640),
            DetectionConfig(),
            10,
            capture_factory=broken_factory,
        )
        failures = []
        worker.scan_failed.connect(failures.append)

        worker.run()

        self.assertEqual(failures, ["cannot open"])


if __name__ == "__main__":
    unittest.main()
