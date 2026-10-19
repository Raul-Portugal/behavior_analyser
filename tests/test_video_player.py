import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import cv2
    from PyQt6.QtCore import QCoreApplication, QEvent, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

    from gui.video_player import (PLAYBACK_SPEEDS, VideoPlayerWidget,
                                  format_timestamp)
    from tests.synthetic_video import FPS, decode_index, write_indexed_video
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False

FRAME_COUNT = 80
_RealCapture = cv2.VideoCapture if HAS_RUNTIME else None


class _RecordingCapture:
    """cv2.VideoCapture wrapper that counts releases and can seek badly."""

    instances = []
    seek_error = 0

    def __init__(self, *args):
        self._cap = _RealCapture(*args)
        self.released = False
        self.seeks = []
        _RecordingCapture.instances.append(self)

    def isOpened(self):
        return self._cap.isOpened()

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.seeks.append(int(value))
            if value > 0:
                value = max(0, value - _RecordingCapture.seek_error)
        return self._cap.set(prop, value)

    def get(self, prop):
        return self._cap.get(prop)

    def grab(self):
        return self._cap.grab()

    def read(self):
        return self._cap.read()

    def release(self):
        self.released = True
        self._cap.release()


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class VideoPlayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.video = write_indexed_video(
            Path(self.tmp.name) / "indexed.avi", FRAME_COUNT
        )
        self.player = VideoPlayerWidget()
        self.addCleanup(self._dispose)
        self.frames = []
        self.player.frame_changed.connect(
            lambda index, ts: self.frames.append((index, ts))
        )

    def _dispose(self):
        self.player.close_video()
        self.player.deleteLater()

    def _open(self):
        self.assertTrue(self.player.open_video(self.video))
        self.assertTrue(self.player.wait_idle())

    def _go(self, frame):
        self.player.seek(frame)
        self.assertTrue(self.player.wait_idle())

    def _wait_until(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                return False
            QCoreApplication.processEvents()
            time.sleep(0.002)
        return True

    def test_first_frame_is_shown_on_open(self):
        self._open()
        self.assertEqual(self.player.current_frame, 0)
        self.assertEqual(self.player.total_frames, FRAME_COUNT)
        self.assertEqual(self.player.fps, FPS)
        self.assertEqual(decode_index(self.player.last_frame), 0)
        self.assertEqual(self.frames[0], (0, 0.0))

    def test_seeks_land_on_the_exact_frame(self):
        self._open()
        for target in (40, 5, 79, 0, 33, 34, 12, 70):
            with self.subTest(target=target):
                self._go(target)
                self.assertEqual(self.player.current_frame, target)
                self.assertEqual(decode_index(self.player.last_frame), target)
                index, timestamp = self.frames[-1]
                self.assertEqual(index, target)
                self.assertAlmostEqual(timestamp, target / FPS)

    def test_seek_is_clamped_to_the_video(self):
        self._open()
        self._go(10_000)
        self.assertEqual(self.player.current_frame, FRAME_COUNT - 1)
        self._go(-5)
        self.assertEqual(self.player.current_frame, 0)

    def test_stepping_and_jumping(self):
        self._open()
        self._go(20)
        for delta, expected in ((1, 21), (-1, 20), (-1, 19), (10, 29),
                                (-10, 19), (-100, 0), (1000, FRAME_COUNT - 1)):
            with self.subTest(delta=delta):
                self.player.step(delta)
                self.assertTrue(self.player.wait_idle())
                self.assertEqual(self.player.current_frame, expected)
                self.assertEqual(
                    decode_index(self.player.last_frame), expected
                )

    def test_scrub_slider_seeks(self):
        self._open()
        self.assertEqual(self.player.slider.maximum(), FRAME_COUNT - 1)
        self.player.slider.setValue(47)
        self.assertTrue(self.player.wait_idle())
        self.assertEqual(self.player.current_frame, 47)

    def test_keyboard_navigation(self):
        self._open()
        self.player.show()
        self.player.setFocus()
        self._go(30)

        def press(key, modifier=Qt.KeyboardModifier.NoModifier):
            QTest.keyClick(self.player, key, modifier)
            self.assertTrue(self.player.wait_idle())

        press(Qt.Key.Key_Right)
        self.assertEqual(self.player.current_frame, 31)
        press(Qt.Key.Key_Left)
        press(Qt.Key.Key_Left)
        self.assertEqual(self.player.current_frame, 29)
        press(Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(self.player.current_frame, 39)
        press(Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(self.player.current_frame, 29)
        press(Qt.Key.Key_End)
        self.assertEqual(self.player.current_frame, FRAME_COUNT - 1)
        press(Qt.Key.Key_Home)
        self.assertEqual(self.player.current_frame, 0)

        QTest.keyClick(self.player, Qt.Key.Key_Space)
        self.assertTrue(self.player.is_playing)
        QTest.keyClick(self.player, Qt.Key.Key_Space)
        self.assertFalse(self.player.is_playing)

    def test_stepping_pauses_playback(self):
        self._open()
        self.player.play()
        self.assertTrue(self.player.is_playing)
        self.player.step(1)
        self.assertFalse(self.player.is_playing)

    def test_playback_advances_with_correct_frames_and_stops_at_the_end(self):
        self._open()
        self._go(FRAME_COUNT - 20)
        self.frames.clear()
        self.player.set_speed(2.0)
        self.player.play()
        self.assertTrue(self._wait_until(lambda: not self.player.is_playing))
        self.assertTrue(self.player.wait_idle())
        self.assertEqual(self.player.current_frame, FRAME_COUNT - 1)
        indices = [index for index, _ in self.frames]
        self.assertGreater(len(indices), 3)
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(indices[-1], FRAME_COUNT - 1)
        self.assertEqual(decode_index(self.player.last_frame), FRAME_COUNT - 1)

    def test_play_from_the_end_restarts(self):
        self._open()
        self._go(FRAME_COUNT - 1)
        self.player.play()
        self.assertTrue(self._wait_until(lambda: self.player.current_frame < 10
                                         and self.player.is_playing))
        self.player.pause()

    def test_playback_speed_choices(self):
        self.assertEqual(PLAYBACK_SPEEDS, (0.25, 0.5, 1.0, 2.0))
        self._open()
        for speed in PLAYBACK_SPEEDS:
            self.player.set_speed(speed)
            self.assertEqual(self.player.speed, speed)
        with self.assertRaises(ValueError):
            self.player.set_speed(3.0)

    def test_slower_speed_plays_fewer_frames(self):
        self._open()
        counts = {}
        for speed in (0.25, 2.0):
            self._go(0)
            self.player.set_speed(speed)
            self.player.play()
            started = time.monotonic()
            self._wait_until(lambda: time.monotonic() - started > 0.6)
            self.player.pause()
            self.player.wait_idle()
            counts[speed] = self.player.current_frame
        self.assertLess(counts[0.25], counts[2.0])

    def test_time_label_shows_time_and_frame(self):
        self._open()
        self._go(37)
        text = self.player.time_label.text()
        self.assertIn(format_timestamp(37 / FPS), text)
        self.assertIn("frame 37", text)
        self.assertEqual(format_timestamp(75.5), "01:15.500")
        self.assertEqual(format_timestamp(0), "00:00.000")
        self.assertEqual(format_timestamp(3725.1234), "62:05.123")

    def test_overlay_callback_receives_a_copy_and_the_frame_index(self):
        self._open()
        seen = []

        def overlay(frame, index):
            seen.append(index)
            frame[:] = 255
            return frame

        self.player.set_overlay(overlay)
        self._go(15)
        self.assertEqual(seen[-1], 15)
        # The raw frame stays free of the overlay.
        self.assertEqual(decode_index(self.player.last_frame), 15)
        self.player.set_overlay(None)

    def test_failing_overlay_does_not_break_display(self):
        self._open()

        def broken(frame, index):
            raise RuntimeError("boom")

        self.player.set_overlay(broken)
        with self.assertLogs("gui.video_player", level="ERROR"):
            self._go(9)
        self.assertEqual(self.player.current_frame, 9)

    def test_unreadable_video_reports_failure(self):
        failures = []
        self.player.load_failed.connect(failures.append)
        self.assertFalse(self.player.open_video(Path(self.tmp.name) / "no.avi"))
        self.assertEqual(len(failures), 1)
        self.assertFalse(self.player.slider.isEnabled())
        self.player.seek(3)   # must be a harmless no-op
        self.assertEqual(self.player.current_frame, -1)

    def test_capture_is_released_on_close_video_and_close_event(self):
        for how in ("close_video", "close"):
            with self.subTest(how=how):
                _RecordingCapture.instances.clear()
                with patch("gui.video_player.cv2.VideoCapture",
                           _RecordingCapture):
                    player = VideoPlayerWidget()
                    try:
                        self.assertTrue(player.open_video(self.video))
                        self.assertTrue(player.wait_idle())
                        self.assertTrue(player.capture_open)
                        player.play()
                        getattr(player, how)()
                    finally:
                        player.close_video()
                        player.deleteLater()
                self.assertFalse(player.capture_open)
                self.assertFalse(player.is_playing)
                self.assertTrue(_RecordingCapture.instances)
                self.assertTrue(
                    all(cap.released for cap in _RecordingCapture.instances)
                )

    def test_reopening_releases_the_previous_capture(self):
        _RecordingCapture.instances.clear()
        with patch("gui.video_player.cv2.VideoCapture", _RecordingCapture):
            self.assertTrue(self.player.open_video(self.video))
            self.assertTrue(self.player.wait_idle())
            first = list(_RecordingCapture.instances)
            self.assertTrue(self.player.open_video(self.video))
            self.assertTrue(self.player.wait_idle())
            self.assertTrue(all(cap.released for cap in first))
            self.player.close_video()

    def test_inexact_codec_seeks_are_corrected(self):
        _RecordingCapture.seek_error = 2
        self.addCleanup(setattr, _RecordingCapture, "seek_error", 0)
        with patch("gui.video_player.cv2.VideoCapture", _RecordingCapture):
            self.assertTrue(self.player.open_video(self.video))
            self.assertTrue(self.player.wait_idle())
            for target in (50, 10, 70, 3):
                self._go(target)
                self.assertEqual(self.player.current_frame, target)
                self.assertEqual(decode_index(self.player.last_frame), target)
            self.player.close_video()

    def test_deleting_the_widget_without_closing_stops_the_decoder(self):
        player = VideoPlayerWidget()
        self.assertTrue(player.open_video(self.video))
        self.assertTrue(player.wait_idle())
        thread = player._thread
        player.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.assertTrue(thread.wait(3000))
        self.assertFalse(thread.capture_open)


if __name__ == "__main__":
    unittest.main()
