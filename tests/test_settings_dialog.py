import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

try:
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from gui.settings_dialog import SettingsDialog, is_tst_mode
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False

from core.calibration import LEGACY_REFERENCE_WARNING, NO_CALIBRATION_NOTE
from core.models import AppConfig, BatchSettings
from roi_manager import ROIManager

FPS = 30.0
FRAMES = 8999  # 299.97 s: not a whole number of seconds


def _write_video(path: Path, frames: int = FRAMES) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (32, 32)
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    for _ in range(frames):
        writer.write(frame)
    writer.release()


def _settings(zone: str = "arm_a", paradigm_id=None, **kwargs) -> BatchSettings:
    manager = ROIManager()
    manager.add_roi(zone, np.array([[0, 0], [30, 0], [30, 30], [0, 30]]))
    return BatchSettings(
        manager, AppConfig().detection, 0.0, paradigm_id=paradigm_id, **kwargs
    )


class TstModeDetectionTests(unittest.TestCase):
    @unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
    def test_detection_prefers_paradigm_id(self):
        self.assertTrue(is_tst_mode("Tail Suspension Test", "Y-Maze"))
        self.assertFalse(is_tst_mode("Freestyle / Open Field", "Tail Suspension Test"))
        self.assertTrue(is_tst_mode(None, "Tail Suspension Test"))
        self.assertFalse(is_tst_mode(None, "Y-Maze"))
        self.assertFalse(is_tst_mode(None, None))


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class SettingsDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls._tmp = tempfile.TemporaryDirectory()
        cls.video = Path(cls._tmp.name) / "clip.avi"
        _write_video(cls.video)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _dialog(self, settings, **kwargs):
        dialog = SettingsDialog(str(self.video), settings, **kwargs)
        self.addCleanup(dialog.close)
        return dialog

    def test_start_slider_leaves_end_time_unset(self):
        dialog = self._dialog(_settings())
        self.assertAlmostEqual(dialog.video_info.duration, 299.9667, places=3)
        dialog.start_slider.setValue(1234)  # 123.4 s
        self.assertAlmostEqual(dialog.start_spinbox.value(), 123.4)
        self.assertAlmostEqual(dialog.end_spinbox.value(), 299.97)
        settings = dialog.get_settings()
        self.assertIsNone(settings.end_time)
        self.assertAlmostEqual(settings.start_time, 123.4)

    def test_precise_spinbox_value_survives_slider_sync(self):
        dialog = self._dialog(_settings())
        dialog.end_spinbox.setValue(150.55)
        dialog.start_slider.setValue(200)
        self.assertAlmostEqual(dialog.end_spinbox.value(), 150.55)
        self.assertAlmostEqual(dialog.get_settings().end_time, 150.55)

    def test_slider_cannot_cross_the_other_bound(self):
        dialog = self._dialog(_settings())
        dialog.end_spinbox.setValue(10.0)
        dialog.start_slider.setValue(500)  # 50 s, beyond the 10 s end
        self.assertAlmostEqual(dialog.start_spinbox.value(), 10.0)

    def test_accept_rejects_start_not_before_end(self):
        dialog = self._dialog(_settings())
        dialog.start_spinbox.setValue(20.0)
        dialog.end_spinbox.setValue(20.0)
        with patch.object(QMessageBox, "warning") as warning:
            dialog.on_accept()
        warning.assert_called_once()
        self.assertEqual(dialog.result_action, "cancel")

    def test_accept_allows_valid_range(self):
        dialog = self._dialog(_settings())
        dialog.start_spinbox.setValue(5.0)
        dialog.end_spinbox.setValue(20.0)
        with patch.object(QMessageBox, "warning") as warning:
            dialog.on_accept()
        warning.assert_not_called()
        self.assertEqual(dialog.result_action, "accept")

    def test_freestyle_zone_named_mouse_is_not_tst(self):
        settings = _settings("mouse_1", paradigm_id="Freestyle / Open Field")
        dialog = self._dialog(settings)
        self.assertFalse(dialog.is_tst)
        self.assertTrue(hasattr(dialog, "start_slider"))

    def test_tst_paradigm_enables_tst_mode(self):
        settings = _settings("mouse_1", paradigm_id="Tail Suspension Test")
        self.assertTrue(self._dialog(settings).is_tst)

    def test_missing_paradigm_falls_back_to_selected_maze(self):
        settings = _settings("mouse_1")
        self.assertFalse(self._dialog(settings).is_tst)
        self.assertTrue(
            self._dialog(settings, maze_name="Tail Suspension Test").is_tst
        )

    def test_no_calibration_note_and_pixel_scale(self):
        dialog = self._dialog(_settings())
        texts = [label.text() for label in dialog.findChildren(type(dialog.preview_label))]
        self.assertIn(NO_CALIBRATION_NOTE, texts)
        self.assertFalse(dialog.scale_spin.isEnabled())
        self.assertEqual(dialog.get_settings().scale_factor, 0.0)

    def test_reference_line_label_and_scale(self):
        settings = _settings()
        settings.roi_manager.set_reference_length(30.0, "User-drawn reference line")
        dialog = self._dialog(settings)
        texts = [label.text() for label in dialog.findChildren(type(dialog.preview_label))]
        self.assertIn("User-drawn reference line (30.00 px)", texts)
        dialog.scale_spin.setValue(15.0)
        self.assertAlmostEqual(dialog.get_settings().scale_factor, 0.5)

    def test_legacy_bounding_box_reference_shows_warning(self):
        settings = _settings()
        manager = settings.roi_manager
        manager._reference_length_pixels = 30.0
        manager._reference_name = "Width of first Arm A ROI"
        dialog = self._dialog(settings)
        texts = [label.text() for label in dialog.findChildren(type(dialog.preview_label))]
        self.assertIn(LEGACY_REFERENCE_WARNING, texts)
        self.assertFalse(any(t.startswith("User-drawn") for t in texts))

    def test_explicit_cm_prefill_is_used_after_redraw(self):
        settings = _settings()
        settings.roi_manager.set_reference_length(40.0, "User-drawn reference line")
        dialog = self._dialog(settings, real_world_cm=12.5)
        self.assertAlmostEqual(dialog.scale_spin.value(), 12.5)
        self.assertAlmostEqual(dialog.get_real_world_cm(), 12.5)


if __name__ == "__main__":
    unittest.main()
