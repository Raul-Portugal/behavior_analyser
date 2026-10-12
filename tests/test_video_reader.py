import tempfile
import threading
import time
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np

    from core.video import BufferedVideoReader, VideoReadError
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False

try:
    from gui.worker import AnalysisWorker
    HAS_WORKER = True
except ModuleNotFoundError:
    HAS_WORKER = False


class FakeCapture:
    """cv2.VideoCapture stand-in driven by a list of read() outcomes."""

    def __init__(self, outcomes, opened=True):
        self._outcomes = list(outcomes)
        self._opened = opened
        self.released = threading.Event()

    def isOpened(self):
        return self._opened

    def set(self, prop, value):
        return True

    def read(self):
        if not self._outcomes:
            return False, None
        outcome = self._outcomes.pop(0)
        if callable(outcome):
            return outcome()
        return outcome

    def release(self):
        self.released.set()


def _frame(value=0):
    return True, np.full((4, 4, 3), value, dtype=np.uint8)


def _reader(capture, **kwargs):
    return BufferedVideoReader(
        Path("fake.mp4"), capture_factory=lambda _path: capture, **kwargs
    )


@unittest.skipUnless(HAS_RUNTIME, "Video runtime is not installed")
class BufferedVideoReaderTests(unittest.TestCase):
    def test_eof_returns_false_none_and_stays_at_eof(self):
        capture = FakeCapture([_frame(1), _frame(2)])
        reader = _reader(capture)
        try:
            self.assertTrue(reader.read()[0])
            self.assertTrue(reader.read()[0])
            self.assertEqual(reader.read(), (False, None))
            self.assertEqual(reader.read(), (False, None))
        finally:
            reader.release()
        self.assertTrue(capture.released.wait(2.0))

    def test_real_video_reads_every_frame_then_clean_eof(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "tiny.avi"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 32)
            )
            for _ in range(7):
                writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
            writer.release()

            reader = BufferedVideoReader(path)
            try:
                count = 0
                while True:
                    ret, frame = reader.read()
                    if not ret:
                        break
                    count += 1
            finally:
                reader.release()

        self.assertEqual(count, 7)

    def test_stall_raises_video_read_error(self):
        unblock = threading.Event()
        capture = FakeCapture([_frame(), lambda: unblock.wait(10) and _frame()])
        reader = _reader(capture, stall_timeout_s=0.4)
        try:
            self.assertTrue(reader.read()[0])
            started = time.monotonic()
            with self.assertRaises(VideoReadError):
                reader.read()
            self.assertLess(time.monotonic() - started, 3.0)
            # The failure is sticky instead of looking like EOF later.
            with self.assertRaises(VideoReadError):
                reader.read()
        finally:
            unblock.set()
            reader.release()

    def test_video_read_error_is_an_ioerror(self):
        self.assertTrue(issubclass(VideoReadError, IOError))

    def test_reader_thread_exception_propagates_after_queued_frames(self):
        def explode():
            raise RuntimeError("decoder crashed")

        capture = FakeCapture([_frame(), _frame(), explode])
        reader = _reader(capture)
        try:
            self.assertTrue(reader.read()[0])
            self.assertTrue(reader.read()[0])
            with self.assertRaisesRegex(RuntimeError, "decoder crashed"):
                reader.read()
            with self.assertRaisesRegex(RuntimeError, "decoder crashed"):
                reader.read()
        finally:
            reader.release()
        self.assertTrue(capture.released.wait(2.0))

    def test_unopenable_capture_raises_instead_of_reporting_eof(self):
        reader = _reader(FakeCapture([], opened=False), stall_timeout_s=5.0)
        try:
            with self.assertRaises(VideoReadError):
                reader.read()
        finally:
            reader.release()

    def test_release_stops_thread_promptly_with_full_queue(self):
        capture = FakeCapture([_frame() for _ in range(50)])
        reader = _reader(capture, buffer_size=2)
        time.sleep(0.2)  # let the producer fill the queue and block

        started = time.monotonic()
        reader.release()

        self.assertLess(time.monotonic() - started, 1.0)
        self.assertFalse(reader.reader_thread.is_alive())
        self.assertTrue(capture.released.is_set())


@unittest.skipUnless(HAS_WORKER, "GUI/scientific runtime is not installed")
class EarlyEndWarningTests(unittest.TestCase):
    def test_complete_and_near_complete_runs_do_not_warn(self):
        warn = AnalysisWorker._early_end_warning
        self.assertIsNone(warn(1000, 1000, 0, 1000))
        self.assertIsNone(warn(1000, 990, 0, 1000))  # exactly 1%
        self.assertIsNone(warn(50, 48, 0, 50))  # within the 2 frame floor

    def test_shortfall_reports_stop_frame_and_expected_end(self):
        message = AnalysisWorker._early_end_warning(1000, 400, 100, 1100)

        self.assertIn("analysis ended early at frame 500 of 1100", message)


if __name__ == "__main__":
    unittest.main()
