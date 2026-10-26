"""Wording, shortcuts, ETA and the log panel."""
import logging
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    from PyQt6.QtGui import QKeySequence
    from PyQt6.QtWidgets import QApplication, QInputDialog, QMessageBox
    from core.models import AppConfig, BatchSettings
    from gui import ux_helpers
    from gui.main_window import MainWindow, QTextEditLogger
    from gui.settings_dialog import SettingsDialog
    from gui.tst_tuner import TstTunerDialog
    from gui.tuners.tst_tuner import TstTunerWidget
    from gui.tuner_dialog import TunerDialog
    from mazes import create_maze
    from roi_manager import ROIManager
    from tests import worker_support as support
    from tests.gui_support import temp_app_settings
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


def _square(offset=0):
    return np.array([[offset, 0], [offset + 40, 0], [offset + 40, 40], [offset, 40]])


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class ProgressHelperTests(unittest.TestCase):
    def test_batch_eta_is_taken_from_the_worker_text(self):
        text = "Motion Analysis | Video ETA: 5s | Batch ETA: 2m 15s"
        self.assertEqual(ux_helpers.extract_batch_eta(text), "2m 15s")
        self.assertIsNone(ux_helpers.extract_batch_eta("Frame 30 | ETA: 5s"))
        self.assertIsNone(ux_helpers.extract_batch_eta("Batch ETA: Calculating..."))
        self.assertIsNone(ux_helpers.extract_batch_eta(""))

    def test_duration_formatting(self):
        self.assertEqual(ux_helpers.format_duration(0), "0s")
        self.assertEqual(ux_helpers.format_duration(45.4), "45s")
        self.assertEqual(ux_helpers.format_duration(185), "3m 05s")
        self.assertEqual(ux_helpers.format_duration(7440), "2h 04m")
        self.assertEqual(ux_helpers.format_duration(-3), "0s")

    def test_estimate_waits_for_enough_progress(self):
        self.assertIsNone(ux_helpers.estimate_batch_eta(1.0, 50))
        self.assertIsNone(ux_helpers.estimate_batch_eta(60.0, 1))
        self.assertIsNone(ux_helpers.estimate_batch_eta(60.0, 100))
        self.assertAlmostEqual(ux_helpers.estimate_batch_eta(60.0, 25), 180.0)

    def test_worker_symbols_map_to_log_levels(self):
        self.assertEqual(ux_helpers.level_for_worker_message("✗ Error x"), logging.ERROR)
        self.assertEqual(ux_helpers.level_for_worker_message("  ⚠ low quality"), logging.WARNING)
        self.assertEqual(ux_helpers.level_for_worker_message("✓ done"), logging.INFO)
        self.assertEqual(ux_helpers.level_for_worker_message("Preparing"), logging.INFO)


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class LogPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _handler(self):
        handler = QTextEditLogger(None)
        self.addCleanup(handler.widget.deleteLater)
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        for level, text in ((logging.INFO, "hello"), (logging.WARNING, "careful <b>"),
                            (logging.ERROR, "broken & done")):
            handler.emit(logging.LogRecord("t", level, "f", 1, text, None, None))
        return handler

    def test_filter_shows_only_the_chosen_level_and_above(self):
        handler = self._handler()
        shown = lambda: handler.widget.toPlainText()  # noqa: E731
        self.assertIn("INFO hello", shown())

        handler.set_min_level(logging.WARNING)
        self.assertNotIn("hello", shown())
        self.assertIn("careful <b>", shown())  # text is escaped, not interpreted
        self.assertIn("broken & done", shown())

        handler.set_min_level(logging.ERROR)
        self.assertNotIn("careful", shown())
        self.assertIn("broken & done", shown())

        handler.set_min_level(logging.INFO)
        self.assertIn("hello", shown())

    def test_new_records_respect_the_current_filter(self):
        handler = self._handler()
        handler.set_min_level(logging.ERROR)
        handler.emit(logging.LogRecord("t", logging.INFO, "f", 1, "later info", None, None))
        handler.emit(logging.LogRecord("t", logging.ERROR, "f", 1, "later error", None, None))
        text = handler.widget.toPlainText()
        self.assertNotIn("later info", text)
        self.assertIn("later error", text)

    def test_full_text_ignores_the_filter_and_the_record_list_is_capped(self):
        handler = self._handler()
        handler.set_min_level(logging.ERROR)
        full = handler.full_text()
        self.assertIn("INFO hello", full)
        self.assertIn("WARNING careful <b>", full)
        handler.MAX_RECORDS = 3
        handler.emit(logging.LogRecord("t", logging.INFO, "f", 1, "x", None, None))
        self.assertEqual(len(handler.records), 3)


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class MainWindowShortcutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        # gui_main raises the root level to INFO; the tests need the same.
        root_logger = logging.getLogger()
        previous_level = root_logger.level
        root_logger.setLevel(logging.INFO)
        self.addCleanup(root_logger.setLevel, previous_level)
        self.window = MainWindow(temp_app_settings(self))
        self.addCleanup(self.window.deleteLater)
        self.addCleanup(self.window._detach_log_handler)

    def test_shortcuts_are_bound(self):
        w = self.window
        expected = {
            w.add_videos_action: "Ctrl+O",
            w.add_folder_action: "Ctrl+Shift+O",
            w.save_config_action: "Ctrl+S",
            w.load_config_action: "Ctrl+L",
            w.start_action: "F5",
            w.cancel_action: "Esc",
        }
        for action, keys in expected.items():
            with self.subTest(action=action.text()):
                self.assertEqual(action.shortcut(), QKeySequence(keys))

    def test_actions_trigger_the_same_slots_as_the_buttons(self):
        with patch("gui.main_window.QFileDialog.getOpenFileNames",
                   return_value=([], "")) as picker:
            self.window.add_videos_action.trigger()
        picker.assert_called_once()

        self.window.video_paths = [Path("a.mp4")]
        self.window.update_button_states()
        with patch.object(QInputDialog, "getItem", return_value=("", False)) as asked:
            self.window.start_action.trigger()
        asked.assert_called_once()

    def test_start_and_cancel_actions_follow_the_state(self):
        w = self.window
        self.assertFalse(w.start_action.isEnabled())
        self.assertFalse(w.cancel_action.isEnabled())
        w.video_paths = [Path("a.mp4")]
        w.update_button_states()
        self.assertTrue(w.start_action.isEnabled())

        w.set_ui_state(is_running=True)
        self.assertFalse(w.start_action.isEnabled())
        self.assertFalse(w.add_videos_action.isEnabled())
        self.assertTrue(w.cancel_action.isEnabled())
        w.set_ui_state(is_running=False)
        self.assertTrue(w.add_videos_action.isEnabled())
        self.assertFalse(w.cancel_action.isEnabled())

    def test_stopping_a_run_asks_first(self):
        class Thread:
            def isRunning(self):
                return True

        class Worker:
            stopped = 0

            def stop(self):
                Worker.stopped += 1

        self.window.analysis_thread = Thread()
        self.window.analysis_worker = Worker()
        self.window.set_ui_state(is_running=True)

        for button in (self.window.cancel_button.click, self.window.cancel_action.trigger):
            Worker.stopped = 0
            with patch.object(ux_helpers, "ask_choice", return_value="keep") as ask:
                button()
            ask.assert_called_once()
            self.assertEqual(Worker.stopped, 0)
            with patch.object(ux_helpers, "ask_choice", return_value="stop"):
                button()
            self.assertEqual(Worker.stopped, 1)
            self.window.cancel_button.setEnabled(True)

    def test_a_forced_cancel_does_not_prompt(self):
        class Worker:
            stopped = 0

            def stop(self):
                Worker.stopped += 1

        self.window.analysis_worker = Worker()
        with patch.object(ux_helpers, "ask_choice") as ask:
            self.window.cancel_analysis()
        ask.assert_not_called()
        self.assertEqual(Worker.stopped, 1)

    def test_prompts_use_real_button_labels(self):
        self.window.video_paths = [Path("a.mp4")]
        self.window.current_settings = object()
        seen = []

        def fake_ask(parent, title, text, choices, **_kwargs):
            seen.append((title, [c.label for c in choices]))
            return "cancel"

        with patch.object(QInputDialog, "getItem", return_value=("Y-Maze", True)), \
                patch.object(ux_helpers, "ask_choice", side_effect=fake_ask), \
                patch.object(self.window, "_offer_resume_setup", return_value=None):
            self.window.start_analysis_workflow()
        self.assertEqual(seen[0][0], "Reuse previous configuration?")
        self.assertEqual(
            seen[0][1], ["Reuse it for all videos", "Set up again", "Cancel"]
        )

    def test_batch_question_uses_review_and_run_buttons(self):
        w = self.window
        w.video_paths = [Path("a.mp4"), Path("b.mp4")]
        manager = ROIManager(source_dimensions=(160, 120))
        manager.add_roi("arm_a", _square())
        settings = BatchSettings(manager, AppConfig().detection, 0.0)
        w.current_settings = settings
        seen = []

        def fake_ask(parent, title, text, choices, **_kwargs):
            seen.append((title, [c.label for c in choices]))
            return "reuse" if title.startswith("Reuse") else "run"

        with patch.object(QInputDialog, "getItem", return_value=("Y-Maze", True)), \
                patch.object(ux_helpers, "ask_choice", side_effect=fake_ask), \
                patch.object(w, "_review_settings", return_value=settings), \
                patch.object(w, "run_batch_analysis") as run_all:
            w.start_analysis_workflow()
        run_all.assert_called_once()
        self.assertEqual(seen[1][0], "Batch Processing")
        self.assertEqual(seen[1][1], ["Review each video…", "Run all now", "Cancel"])

    def test_review_each_video_passes_the_reviewed_video_on(self):
        w = self.window
        w.video_paths = [Path("a.mp4"), Path("b.mp4")]
        manager = ROIManager(source_dimensions=(160, 120))
        manager.add_roi("arm_a", _square())
        settings = BatchSettings(manager, AppConfig().detection, 0.0)
        w.current_settings = settings

        def fake_ask(parent, title, text, choices, **_kwargs):
            return "reuse" if title.startswith("Reuse") else "review"

        with patch.object(QInputDialog, "getItem", return_value=("Y-Maze", True)), \
                patch.object(ux_helpers, "ask_choice", side_effect=fake_ask), \
                patch.object(w, "_review_settings", return_value=settings), \
                patch.object(w, "run_batch_with_verification") as verify:
            w.start_analysis_workflow()
        verify.assert_called_once_with(settings, reviewed_video=Path("a.mp4"))

    def test_eta_label_prefers_the_workers_figure_then_estimates(self):
        w = self.window
        self.assertFalse(w.eta_label.isVisibleTo(w))
        w._run_started = time.monotonic() - 60
        w.eta_label.setVisible(True)

        w.update_progress_detailed(25, "Video: a", "Frame 30 | ETA: 5s")
        self.assertIn("~3m 00s", w.eta_label.text())
        w.update_progress_detailed(26, "Video: a", "Motion | Batch ETA: 9m 10s")
        self.assertIn("9m 10s", w.eta_label.text())
        self.assertIn("elapsed 1m", w.eta_label.text())

    def test_setup_uses_a_busy_bar_instead_of_made_up_percentages(self):
        w = self.window
        w._show_setup_status("Configuration", "Tuning detection parameters...")
        self.assertEqual((w.progress_bar.minimum(), w.progress_bar.maximum()), (0, 0))
        w.update_progress(40, "Video: a")
        self.assertEqual(w.progress_bar.maximum(), 100)
        self.assertEqual(w.progress_bar.value(), 40)
        w._clear_setup_status("Ready")
        self.assertEqual(w.progress_bar.value(), 0)

    def test_worker_messages_are_logged_at_their_severity(self):
        w = self.window
        w._on_worker_log("⚠ tracking quality review")
        w._on_worker_log("✗ Error processing a.mp4: boom")
        w._on_worker_log("✓ Successfully processed: a.mp4")
        levels = [level for level, text in w.log_handler.records
                  if "tracking quality" in text or "boom" in text or "Successfully" in text]
        self.assertEqual(levels, [logging.WARNING, logging.ERROR, logging.INFO])

    def test_log_level_combo_filters_the_view_and_save_log_writes_everything(self):
        w = self.window
        logging.getLogger().info("visible info line")
        logging.getLogger().error("visible error line")
        w.log_level_combo.setCurrentIndex(2)  # Error
        shown = w.log_handler.widget.toPlainText()
        self.assertNotIn("visible info line", shown)
        self.assertIn("visible error line", shown)

        target = self.root / "log.txt"
        with patch("gui.main_window.QFileDialog.getSaveFileName",
                   return_value=(str(target), "")):
            w.save_log()
        saved = target.read_text(encoding="utf-8")
        self.assertIn("visible info line", saved)
        self.assertIn("visible error line", saved)

    def test_save_log_reports_a_write_failure(self):
        target = self.root / "missing_folder" / "log.txt"
        with patch("gui.main_window.QFileDialog.getSaveFileName",
                   return_value=(str(target), "")), \
                patch.object(QMessageBox, "critical") as critical:
            self.window.save_log()
        critical.assert_called_once()


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class DialogWordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _tuner(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        video = support.write_tracking_video(Path(tmp.name), frames=12)
        ref = np.full((120, 160), 40, dtype=np.uint8)
        dialog = TunerDialog(str(video), ref, ROIManager(source_dimensions=(160, 120)))
        self.addCleanup(dialog.deleteLater)
        return dialog

    def test_slider_labels_show_real_values_and_units(self):
        dialog = self._tuner()
        labels = {name: label.text() for name, label in dialog.slider_labels.items()}
        self.assertEqual(labels["Threshold percentile"], "Threshold percentile: 99.0")
        self.assertEqual(labels["Weight omega"], "Weight omega: 0.90")
        self.assertEqual(labels["Search window"], "Search window: 100 px")
        self.assertEqual(labels["Min component area"], "Min component area: 20 px²")
        self.assertEqual(labels["Morphology kernel"], "Morphology kernel: 3 px")
        self.assertEqual(labels["Min confidence"], "Min confidence: 0.15")

    def test_moving_a_slider_updates_label_and_config_in_real_units(self):
        dialog = self._tuner()
        dialog.sliders["Threshold percentile"].setValue(985)
        dialog.sliders["Min confidence"].setValue(30)
        self.assertEqual(
            dialog.slider_labels["Threshold percentile"].text(),
            "Threshold percentile: 98.5",
        )
        config = dialog.get_detection_config()
        self.assertAlmostEqual(config.threshold_percentile, 98.5)
        self.assertAlmostEqual(config.min_detection_confidence, 0.30)

    def test_default_slider_values_still_equal_the_default_config(self):
        dialog = self._tuner()
        default = AppConfig().detection
        config = dialog.get_detection_config()
        self.assertEqual(config.threshold_percentile, default.threshold_percentile)
        self.assertEqual(config.weight_omega, default.weight_omega)
        self.assertEqual(config.window_size, default.window_size)
        self.assertEqual(config.min_detection_confidence, default.min_detection_confidence)

    def test_window_titles_are_neutral(self):
        self.assertEqual(self._tuner().windowTitle(), "Detection Tuner")
        roi_manager = ROIManager(source_dimensions=(100, 100))
        roi_manager.add_roi("mouse_1", _square())
        with patch.object(TstTunerWidget, "start_scan"):
            tst = TstTunerDialog(Path("unopened.mp4"), roi_manager, create_maze("Tail Suspension Test"))
        self.addCleanup(tst.deleteLater)
        try:
            self.assertEqual(tst.windowTitle(), "TST Tuner")
        finally:
            tst.cleanup()
        for title in (self._tuner().windowTitle(), tst.windowTitle()):
            self.assertNotIn("Enhanced", title)
            self.assertNotIn("High Performance", title)

    def test_validation_video_checkbox_has_one_name_for_every_paradigm(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        video = support.write_tracking_video(Path(tmp.name), frames=12)
        manager = ROIManager(source_dimensions=(160, 120))
        manager.add_roi("arm_a", _square())
        manager.add_roi("mouse_1", _square(50))
        for paradigm in ("Y-Maze", "Tail Suspension Test"):
            settings = BatchSettings(manager, AppConfig().detection, 0.0)
            dialog = SettingsDialog(str(video), settings, maze_name=paradigm)
            self.addCleanup(dialog.deleteLater)
            self.assertEqual(
                dialog.timelapse_check.text(), "Create validation video (annotated)"
            )


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class TstTunerShortcutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _dialog(self):
        roi_manager = ROIManager(source_dimensions=(100, 100))
        roi_manager.add_roi("mouse_1", _square())
        with patch.object(TstTunerWidget, "start_scan"):
            dialog = TstTunerDialog(
                Path("unopened.mp4"), roi_manager, create_maze("Tail Suspension Test")
            )
        self.addCleanup(dialog.deleteLater)
        self.addCleanup(dialog.cleanup)
        return dialog

    def test_shortcuts_are_registered(self):
        dialog = self._dialog()
        keys = {shortcut.key().toString() for shortcut in dialog.playback_shortcuts}
        self.assertEqual(keys, {"Space", "Left", "Right", "Ctrl+Left", "Ctrl+Right"})

    def test_keys_do_nothing_before_the_scan_finished(self):
        dialog = self._dialog()
        dialog.toggle_playback_from_key()
        self.assertFalse(dialog.is_playing)
        dialog.step_frames(1)
        self.assertEqual(dialog.scrub_slider.value(), 0)

    def test_space_toggles_playback_and_arrows_step_frames(self):
        dialog = self._dialog()
        dialog.play_button.setEnabled(True)
        dialog.scrub_slider.setEnabled(True)
        dialog.scrub_slider.setRange(0, 100)
        dialog.scrub_slider.setValue(50)
        dialog.frame_buffer = list(range(101))
        with patch.object(dialog.widget, "update_frame"), \
                patch.object(dialog.widget, "update_cursor_blit"):
            dialog.step_frames(1)
            self.assertEqual(dialog.scrub_slider.value(), 51)
            dialog.step_frames(-10)
            self.assertEqual(dialog.scrub_slider.value(), 41)
            dialog.step_frames(-1000)
            self.assertEqual(dialog.scrub_slider.value(), 0)

            dialog.toggle_playback_from_key()
            self.assertTrue(dialog.is_playing)
            dialog.step_frames(1)  # stepping pauses playback first
            self.assertFalse(dialog.is_playing)
            self.assertEqual(dialog.scrub_slider.value(), 1)
            dialog.toggle_playback_from_key()
            dialog.toggle_playback_from_key()
            self.assertFalse(dialog.is_playing)


if __name__ == "__main__":
    unittest.main()
