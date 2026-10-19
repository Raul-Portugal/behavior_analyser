import time
import unittest

try:
    from PyQt6 import sip
    from PyQt6.QtCore import QThread
    from PyQt6.QtWidgets import QApplication, QWidget

    from gui.tuners import lifecycle
    from gui.tuners.lifecycle import TunerLifecycle, stop_thread
    HAS_QT = True
except ModuleNotFoundError:
    HAS_QT = False


if HAS_QT:
    class _LoopThread(QThread):
        """Runs until interrupted (optionally ignoring it for a while)."""

        def __init__(self, ignore_seconds: float = 0.0):
            super().__init__()
            self.ignore_seconds = ignore_seconds
            self.cancelled = False

        def cancel(self):
            self.cancelled = True
            self.requestInterruption()

        def run(self):
            deadline = time.monotonic() + self.ignore_seconds
            while time.monotonic() < deadline:
                time.sleep(0.005)
            while not self.isInterruptionRequested():
                time.sleep(0.005)

    class _FakeCapture:
        def __init__(self, path):
            self.path = path
            self.open = True
            self.releases = 0

        def isOpened(self):
            return self.open

        def release(self):
            self.open = False
            self.releases += 1


@unittest.skipUnless(HAS_QT, "Qt runtime is not installed")
class StopThreadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_none_and_idle_threads_count_as_stopped(self):
        self.assertTrue(stop_thread(None))
        self.assertTrue(stop_thread(_LoopThread()))

    def test_running_thread_is_interrupted_and_joined(self):
        thread = _LoopThread()
        thread.start()
        self.assertTrue(stop_thread(thread))
        self.assertFalse(thread.isRunning())

    def test_cancel_callback_replaces_plain_interruption(self):
        thread = _LoopThread()
        thread.start()
        self.assertTrue(stop_thread(thread, cancel=thread.cancel, timeout_ms=5000))
        self.assertTrue(thread.cancelled)
        self.assertFalse(thread.isRunning())

    def test_stubborn_thread_is_retained_until_it_finishes(self):
        thread = _LoopThread(ignore_seconds=0.4)
        thread.start()
        self.assertFalse(stop_thread(thread, timeout_ms=20))
        self.assertIn(thread, lifecycle._STRAGGLERS)
        thread.wait()
        deadline = time.monotonic() + 5
        while thread in lifecycle._STRAGGLERS and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertNotIn(thread, lifecycle._STRAGGLERS)


@unittest.skipUnless(HAS_QT, "Qt runtime is not installed")
class CaptureHolderTests(unittest.TestCase):
    def test_reopens_lazily_after_release_until_shut_down(self):
        created = []

        def factory(path):
            created.append(_FakeCapture(path))
            return created[-1]

        tuner = TunerLifecycle("test")
        holder = tuner.open_capture("clip.mp4", factory)
        self.assertEqual(len(created), 1)
        self.assertTrue(holder.is_open)

        tuner.pause()
        self.assertFalse(holder.is_open)
        self.assertTrue(holder.get().isOpened())  # reopened on demand
        self.assertEqual(len(created), 2)

        tuner.shutdown()
        self.assertFalse(created[-1].isOpened())
        self.assertFalse(holder.get().isOpened())
        self.assertEqual(len(created), 2)  # never reopened after shutdown

    def test_lazy_holder_shut_down_before_use_hands_out_a_dead_capture(self):
        tuner = TunerLifecycle("test")
        holder = tuner.open_capture("clip.mp4", _FakeCapture, open_now=False)
        tuner.shutdown()
        self.assertFalse(holder.get().isOpened())


@unittest.skipUnless(HAS_QT, "Qt runtime is not installed")
class TunerLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_pause_joins_threads_and_allows_a_restart(self):
        tuner = TunerLifecycle("test")
        first = _LoopThread()
        self.assertTrue(tuner.start_thread(first))
        self.assertTrue(tuner.has_running_thread())

        tuner.pause()
        self.assertFalse(first.isRunning())
        self.assertFalse(tuner.has_running_thread())
        self.assertFalse(tuner.is_shut_down)

        second = _LoopThread()
        self.assertTrue(tuner.start_thread(second))
        tuner.shutdown()
        self.assertFalse(second.isRunning())

    def test_shutdown_is_idempotent_and_refuses_new_threads(self):
        tuner = TunerLifecycle("test")
        thread = _LoopThread()
        tuner.start_thread(thread, cancel=thread.cancel)
        tuner.shutdown()
        tuner.shutdown()
        self.assertTrue(tuner.is_shut_down)
        self.assertTrue(thread.cancelled)
        late = _LoopThread()
        self.assertFalse(tuner.start_thread(late))
        self.assertFalse(late.isRunning())

    def test_deleting_the_attached_widget_joins_and_releases(self):
        tuner = TunerLifecycle("test")
        holder = tuner.open_capture("clip.mp4", _FakeCapture)
        thread = _LoopThread()
        host = QWidget()
        tuner.attach(host)
        tuner.start_thread(thread)

        sip.delete(host)

        self.assertFalse(thread.isRunning())
        self.assertFalse(holder.is_open)
        self.assertTrue(tuner.is_shut_down)


if __name__ == "__main__":
    unittest.main()
