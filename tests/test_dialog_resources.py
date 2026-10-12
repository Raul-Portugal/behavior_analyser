"""Dialogs must release their video capture however they are dismissed."""
import tempfile
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np
    from PyQt6.QtWidgets import QApplication
    from gui.roi_preview import ROIPreviewDialog
    from gui.settings_dialog import SettingsDialog
    from core.models import AppConfig, BatchSettings
    from mazes import create_maze
    from roi_manager import ROIManager
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


def _write_video(path: Path, frames: int = 30) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (32, 32)
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    for _ in range(frames):
        writer.write(frame)
    writer.release()


def _roi_manager() -> "ROIManager":
    manager = ROIManager()
    manager.add_roi("arm_a", np.array([[0, 0], [30, 0], [30, 30], [0, 30]]))
    return manager


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class DialogCaptureReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.video = Path(cls._tmp.name) / "clip.avi"
        _write_video(cls.video)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _settings_dialog(self):
        settings = BatchSettings(
            _roi_manager(), AppConfig().detection, 0.0
        )
        dialog = SettingsDialog(str(self.video), settings)
        self.addCleanup(dialog.deleteLater)
        return dialog

    def _preview_dialog(self):
        return ROIPreviewDialog(
            self.video, _roi_manager(), create_maze("Y-Maze")
        )

    def test_settings_dialog_releases_capture_on_accept(self):
        dialog = self._settings_dialog()
        cap = dialog.cap
        self.assertTrue(cap.isOpened())

        dialog.on_accept()

        self.assertFalse(cap.isOpened())
        # The widgets must outlive accept() so callers can read the result.
        self.assertIsNotNone(dialog.get_settings())

    def test_settings_dialog_releases_capture_on_reject(self):
        dialog = self._settings_dialog()
        cap = dialog.cap

        dialog.reject()

        self.assertFalse(cap.isOpened())

    def test_settings_dialog_releases_capture_on_skip_and_redraw(self):
        for action in ("on_skip", "on_redraw"):
            dialog = self._settings_dialog()
            cap = dialog.cap
            getattr(dialog, action)()
            self.assertFalse(cap.isOpened(), action)

    def test_settings_dialog_releases_capture_when_window_is_closed(self):
        dialog = self._settings_dialog()
        cap = dialog.cap
        dialog.show()

        dialog.close()

        self.assertFalse(cap.isOpened())

    def test_preview_dialog_releases_capture_on_accept(self):
        dialog = self._preview_dialog()
        cap = dialog.cap
        self.assertTrue(cap.isOpened())

        dialog.accept()

        self.assertFalse(cap.isOpened())

    def test_preview_dialog_releases_capture_on_reject(self):
        dialog = self._preview_dialog()
        cap = dialog.cap

        dialog.reject()

        self.assertFalse(cap.isOpened())

    def test_preview_dialog_is_deleted_after_it_closes(self):
        from PyQt6 import sip
        from PyQt6.QtCore import QCoreApplication, QEvent

        dialog = self._preview_dialog()
        dialog.show()
        dialog.reject()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

        self.assertTrue(sip.isdeleted(dialog))

    def test_unopenable_video_does_not_leak_a_capture(self):
        with self.assertRaises(IOError):
            ROIPreviewDialog(
                Path(self._tmp.name) / "missing.avi",
                _roi_manager(), create_maze("Y-Maze"),
            )


if __name__ == "__main__":
    unittest.main()
