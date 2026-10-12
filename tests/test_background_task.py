"""run_with_progress and the off-thread reference frame in the setup wizard."""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

try:
    import numpy as np
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication, QProgressDialog, QPushButton
    from core.video import ReferenceFrameGenerator
    from gui.background_task import TaskCancelled, run_with_progress
    from gui.main_window import MainWindow
    from mazes import create_maze
    from tests import worker_support as support
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


def _click_cancel_soon(delay_ms=50):
    def click():
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QProgressDialog):
                widget.findChild(QPushButton).click()

    QTimer.singleShot(delay_ms, click)


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class RunWithProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_returns_the_result_computed_off_the_ui_thread(self):
        seen = {}

        def work(should_cancel):
            seen["thread"] = threading.get_ident()
            return 42

        self.assertEqual(run_with_progress(None, "Working...", work), 42)
        self.assertNotEqual(seen["thread"], threading.get_ident())

    def test_ui_event_loop_keeps_running_during_the_task(self):
        ticks = []
        timer = QTimer()
        timer.timeout.connect(lambda: ticks.append(1))
        timer.start(10)
        self.addCleanup(timer.stop)

        run_with_progress(None, "Working...", lambda c: time.sleep(0.3))

        self.assertGreater(len(ticks), 5)

    def test_errors_are_reraised_on_the_calling_thread(self):
        def work(should_cancel):
            raise ValueError("boom")

        with self.assertRaisesRegex(ValueError, "boom"):
            run_with_progress(None, "Working...", work)

    def test_cancel_button_stops_the_task_and_raises_task_cancelled(self):
        stopped = []

        def work(should_cancel):
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if should_cancel():
                    stopped.append(1)
                    raise InterruptedError("stopped")
                time.sleep(0.01)
            return "finished anyway"

        _click_cancel_soon()
        started = time.monotonic()
        with self.assertRaises(TaskCancelled):
            run_with_progress(None, "Working...", work)

        self.assertEqual(stopped, [1])
        self.assertLess(time.monotonic() - started, 3)

    def test_cancel_wins_even_if_the_task_finishes_first(self):
        def work(should_cancel):
            while not should_cancel():
                time.sleep(0.01)
            return "late result"

        _click_cancel_soon()
        with self.assertRaises(TaskCancelled):
            run_with_progress(None, "Working...", work)

    def test_dialog_is_closed_afterwards(self):
        run_with_progress(None, "Working...", lambda c: None)
        QApplication.processEvents()
        visible = [
            w for w in QApplication.topLevelWidgets()
            if isinstance(w, QProgressDialog) and w.isVisible()
        ]
        self.assertEqual(visible, [])


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class WizardReferenceFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.window = MainWindow()
        self.addCleanup(self.window.deleteLater)
        self.addCleanup(self.window._detach_log_handler)
        self.window.maze = create_maze("Y-Maze")
        cache = patch.object(
            ReferenceFrameGenerator, "get_cache_dir",
            return_value=self.root / "cache",
        )
        cache.start()
        self.addCleanup(cache.stop)

    def test_reference_frame_is_built_on_a_worker_thread(self):
        video = support.write_tracking_video(self.root, frames=30)
        real_generate = ReferenceFrameGenerator.generate
        threads = []

        def spy(*args, **kwargs):
            threads.append(threading.get_ident())
            self.assertIn("should_cancel", kwargs)
            return real_generate(*args, **kwargs)

        with patch.object(ReferenceFrameGenerator, "generate", spy):
            frame = self.window._generate_reference_frame(video)

        self.assertEqual(frame.shape, (120, 160))
        self.assertNotEqual(threads, [threading.get_ident()])

    def test_cancelling_reference_generation_aborts_the_wizard(self):
        roi_dialog = Mock()
        roi_dialog.exec.return_value = True
        arm_dialog = Mock()
        arm_dialog.exec.return_value = True
        arm_dialog.get_parameters.return_value = (
            self.window.maze.get_algorithm_parameters()
        )
        with patch("gui.main_window.RoiSelectorDialog", return_value=roi_dialog), \
                patch("gui.main_window.ArmEntryParametersDialog",
                      return_value=arm_dialog), \
                patch.object(
                    self.window, "_generate_reference_frame",
                    side_effect=TaskCancelled(),
                ), \
                patch("gui.main_window.TunerDialog") as tuner:
            result = self.window._run_setup_wizard(
                self.root / "video.avi", "Y-Maze"
            )

        self.assertIsNone(result)
        tuner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
