"""Closing the window and menu state while an analysis is running."""
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

try:
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from gui.main_window import MainWindow
    from gui.worker import BatchRunOutcome
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


class FakeThread:
    """Stands in for the QThread of a running analysis."""

    def __init__(self, stops_on_wait=True):
        self.running = True
        self.stops_on_wait = stops_on_wait
        self.calls = []

    def isRunning(self):
        return self.running

    def quit(self):
        self.calls.append("quit")
        if self.stops_on_wait:
            self.running = False

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))
        if self.stops_on_wait:
            self.running = False
        return not self.running

    def terminate(self):
        self.calls.append("terminate")
        self.running = False


class FakeWorker:
    def __init__(self):
        self.stopped = 0

    def stop(self):
        self.stopped += 1


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class RunLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.addCleanup(self.window.deleteLater)
        self.addCleanup(self.window._detach_log_handler)
        self.window.FORCE_QUIT_DELAY_MS = 60
        self.window.show()
        self.thread = FakeThread()
        self.worker = FakeWorker()
        self.window.analysis_thread = self.thread
        self.window.analysis_worker = self.worker
        self.window.set_ui_state(is_running=True)

    def _close_confirming(self):
        with patch.object(
            QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes
        ):
            started = time.monotonic()
            closed = self.window.close()
            elapsed = time.monotonic() - started
        return closed, elapsed

    def test_close_during_run_requests_cancel_and_does_not_block(self):
        closed, elapsed = self._close_confirming()

        self.assertFalse(closed)
        self.assertTrue(self.window.isVisible())
        self.assertLess(elapsed, 1.0)
        self.assertEqual(self.worker.stopped, 1)
        self.assertTrue(self.window._close_requested)
        self.assertEqual(self.window.status_label.text(), "Stopping analysis…")
        self.assertNotIn(("wait", None), self.thread.calls)
        self.assertEqual(self.thread.calls, [])

    def test_declining_the_prompt_keeps_the_run_going(self):
        with patch.object(
            QMessageBox, "question", return_value=QMessageBox.StandardButton.No
        ):
            self.assertFalse(self.window.close())

        self.assertEqual(self.worker.stopped, 0)
        self.assertFalse(self.window._close_requested)

    def test_window_closes_when_the_worker_finishes(self):
        self._close_confirming()
        with patch.object(QMessageBox, "warning") as warning, \
                patch.object(QMessageBox, "information") as information:
            self.window.analysis_finished(
                BatchRunOutcome(cancelled=True, output_dir=Path("."))
            )
            QTest.qWait(50)

        self.assertFalse(self.window.isVisible())
        warning.assert_not_called()  # no "Analysis Cancelled" box
        information.assert_not_called()
        self.assertIsNone(self.window.analysis_thread)
        self.assertIsNone(self.window.analysis_worker)
        self.assertEqual(self.thread.calls[0], "quit")

    def test_repeated_close_requests_only_cancel_once(self):
        self._close_confirming()
        self.assertFalse(self.window.close())
        self.assertEqual(self.worker.stopped, 1)

    def test_force_quit_after_the_timeout_closes_the_window(self):
        self._close_confirming()
        with patch.object(
            self.window, "_confirm_force_quit", return_value=True
        ) as confirm:
            QTest.qWait(300)

        confirm.assert_called_once()
        self.assertFalse(self.window.isVisible())
        self.assertIn("quit", self.thread.calls)
        self.assertNotIn("terminate", self.thread.calls)

    def test_force_quit_terminates_a_thread_that_will_not_stop(self):
        self.thread.stops_on_wait = False
        self._close_confirming()
        with patch.object(
            self.window, "_confirm_force_quit", return_value=True
        ):
            QTest.qWait(300)

        self.assertIn("terminate", self.thread.calls)
        self.assertFalse(self.window.isVisible())

    def test_keep_waiting_leaves_the_window_open_and_asks_again(self):
        self._close_confirming()
        with patch.object(
            self.window, "_confirm_force_quit", return_value=False
        ) as confirm:
            QTest.qWait(250)

        self.assertGreaterEqual(confirm.call_count, 2)
        self.assertTrue(self.window.isVisible())
        self.window._close_requested = False  # let cleanup proceed

    def test_force_quit_offer_is_skipped_if_the_run_already_ended(self):
        self._close_confirming()
        self.window.analysis_thread = None
        with patch.object(self.window, "_confirm_force_quit") as confirm:
            QTest.qWait(150)
        confirm.assert_not_called()

    def test_close_without_a_run_accepts_immediately(self):
        self.window.analysis_thread = None
        self.window.analysis_worker = None
        self.assertTrue(self.window.close())

    def test_configuration_and_output_menus_are_disabled_while_running(self):
        self.window.current_settings = Mock()
        self.window.set_ui_state(is_running=True)
        self.assertFalse(self.window.save_config_action.isEnabled())
        self.assertFalse(self.window.load_config_action.isEnabled())
        self.assertFalse(self.window.set_output_action.isEnabled())

        self.window.set_ui_state(is_running=False)
        self.assertTrue(self.window.save_config_action.isEnabled())
        self.assertTrue(self.window.load_config_action.isEnabled())
        self.assertTrue(self.window.set_output_action.isEnabled())

    def test_save_action_stays_disabled_without_settings_after_a_run(self):
        self.window.current_settings = None
        self.window.set_ui_state(is_running=False)
        self.assertFalse(self.window.save_config_action.isEnabled())


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class RealThreadLifecycleTests(unittest.TestCase):
    """A real QThread must be torn down without touching freed objects."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_worker_and_thread_are_released_after_finish(self):
        import tempfile

        window = MainWindow()
        self.addCleanup(window.deleteLater)
        self.addCleanup(window._detach_log_handler)
        from mazes import create_maze
        window.maze = create_maze("Elevated Plus Maze")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            window.output_dir = Path(tmp)
            with patch.object(QMessageBox, "information") as information, \
                    patch.object(QMessageBox, "warning"):
                with patch.object(
                    window, "_validate_analysis_plan", return_value=True
                ):
                    window.run_worker({})
                deadline = time.monotonic() + 10
                while window.analysis_thread is not None:
                    QTest.qWait(20)
                    self.assertLess(time.monotonic(), deadline, "run hung")
                QTest.qWait(50)  # let the deleteLater events run

            information.assert_called_once()
            self.assertIsNone(window.analysis_worker)
            # Safe even though Qt has freed the worker and thread.
            window.cancel_analysis()
            self.assertTrue(window.start_button.isEnabled())
            self.assertTrue(window.load_config_action.isEnabled())


if __name__ == "__main__":
    unittest.main()
