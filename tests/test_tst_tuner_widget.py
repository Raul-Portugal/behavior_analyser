import copy
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
    from PyQt6 import sip
    from PyQt6.QtCore import QCoreApplication, QEvent, Qt
    from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget

    from core.tst_motion_engine import TSTFramePipeline, TSTMotionConfig
    from gui.tst_tuner import TstTunerDialog
    from gui.tuners import tst_tuner
    from gui.tuners.tst_tuner import TstTunerWidget
    from mazes import create_maze
    from roi_manager import ROIManager
    from tests import worker_support as support
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


def _slow_pipeline_class():
    """A TSTFramePipeline that is slow and records create/close calls."""

    class SlowPipeline(TSTFramePipeline):
        created = []
        closed = []

        def __init__(self, engine):
            super().__init__(engine)
            type(self).created.append(self)

        def pop(self):
            time.sleep(0.02)
            return super().pop()

        def close(self):
            type(self).closed.append(self)
            super().close()

    return SlowPipeline


def _wait_until(app, predicate, seconds=30.0):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    return predicate()


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class TstTunerWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.video = support.write_tracking_video(Path(tmp.name), frames=240)
        self.roi_manager = ROIManager(source_dimensions=(160, 120))
        self.roi_manager.add_roi(
            "mouse_1", np.array([[0, 0], [159, 0], [159, 119], [0, 119]])
        )

    def _maze(self):
        return create_maze("Tail Suspension Test")

    def _widget(self, maze=None, initial_motion_config=None, parent=None,
                video=None):
        widget = TstTunerWidget(
            video or self.video, self.roi_manager, maze or self._maze(),
            initial_motion_config=initial_motion_config, parent=parent,
        )
        self.addCleanup(widget.shutdown)
        return widget

    @staticmethod
    def _fake_scan(widget):
        """Put the widget in the state a finished scan leaves it in."""
        widget.fps = 10.0
        widget.analysis_fps = 10.0
        widget.scan_len = 40
        widget.frame_buffer = list(range(40))
        widget.play_button.setEnabled(True)
        widget.scrub_slider.setEnabled(True)
        widget.scrub_slider.setRange(0, 39)

    # -- initial configuration ---------------------------------------------
    def test_defaults_come_from_the_maze(self):
        maze = self._maze()
        maze.set_parameters(1.7)
        widget = self._widget(maze)
        self.assertEqual(widget.time_slider.value(), 17)
        self.assertEqual(widget.min_bout_seconds(), 1.7)
        self.assertEqual(widget.motion_config(), maze.motion_config)

    def test_initial_motion_config_round_trips_through_the_controls(self):
        initial = TSTMotionConfig(
            active_residual_threshold=0.02,
            segmentation_contrast=33.0,
            temporal_window_seconds=2.5,
        )
        maze = self._maze()
        widget = self._widget(maze, initial_motion_config=initial)

        self.assertEqual(widget.motion_config(), initial)
        self.assertAlmostEqual(
            widget.advanced_controls["active_residual_threshold"].value(),
            0.02,
        )
        self.assertAlmostEqual(
            widget.advanced_controls["segmentation_contrast"].value(), 33.0
        )
        # A copy in, a copy out; the maze is untouched until applied.
        initial.segmentation_contrast = 1.0
        self.assertEqual(widget.motion_config().segmentation_contrast, 33.0)
        widget.motion_config().segmentation_contrast = 2.0
        self.assertEqual(widget.motion_config().segmentation_contrast, 33.0)
        self.assertNotEqual(maze.motion_config.segmentation_contrast, 33.0)

    def test_apply_to_maze_writes_config_and_min_bout(self):
        widget = self._widget(
            initial_motion_config=TSTMotionConfig(
                passive_pendulum_score_threshold=0.4
            )
        )
        widget.time_slider.setValue(23)
        target = self._maze()
        widget.apply_to_maze(target)
        self.assertEqual(target.min_time_seconds, 2.3)
        self.assertEqual(target.motion_config, widget.motion_config())
        self.assertIsNot(target.motion_config, widget.preview_motion_config)

    def test_apply_to_maze_rejects_invalid_settings_without_side_effects(self):
        widget = self._widget()
        widget.advanced_controls["temporal_window_seconds"].setValue(0.5)
        widget.preview_motion_config.analysis_width = -5
        target = self._maze()
        before = copy.deepcopy(target.get_algorithm_parameters())
        with self.assertRaises(ValueError):
            widget.apply_to_maze(target)
        self.assertEqual(target.get_algorithm_parameters(), before)

    def test_config_changed_fires_for_every_kind_of_edit(self):
        widget = self._widget()
        fired = []
        widget.config_changed.connect(lambda: fired.append(1))
        widget.time_slider.setValue(widget.time_slider.value() + 1)
        self.assertEqual(len(fired), 1)
        widget.advanced_controls["active_residual_threshold"].setValue(0.03)
        self.assertEqual(len(fired), 2)

    # -- scan start and playback --------------------------------------------
    def test_scan_starts_on_first_show_not_at_construction(self):
        with patch.object(TstTunerWidget, "start_scan") as start:
            widget = self._widget()
            start.assert_not_called()
            widget.show()
            self.app.processEvents()
            start.assert_called_once()

    def test_a_widget_without_rois_reports_the_error_and_never_scans(self):
        widget = TstTunerWidget(
            self.video, ROIManager(source_dimensions=(160, 120)), self._maze()
        )
        self.addCleanup(widget.shutdown)
        with patch.object(TstTunerWidget, "start_scan") as start:
            widget.show()
            start.assert_not_called()
        self.assertEqual(widget.image_label.text(), "Error: No ROI defined")

    def test_hiding_stops_playback_and_showing_does_not_resume_it(self):
        with patch.object(TstTunerWidget, "start_scan"):
            widget = self._widget()
            widget.show()
        self._fake_scan(widget)
        widget.play_button.click()
        self.assertTrue(widget.is_playing)
        self.assertTrue(widget.play_timer.isActive())

        widget.hide()

        self.assertFalse(widget.is_playing)
        self.assertFalse(widget.play_timer.isActive())
        self.assertFalse(widget.play_button.isChecked())
        self.assertEqual(widget.play_button.text(), "▶ Play")
        with patch.object(TstTunerWidget, "start_scan"):
            widget.show()
        self.assertFalse(widget.play_timer.isActive())

    def test_shortcuts_are_scoped_to_the_widget_and_children(self):
        widget = self._widget()
        keys = {s.key().toString() for s in widget.playback_shortcuts}
        self.assertEqual(
            keys, {"Space", "Left", "Right", "Ctrl+Left", "Ctrl+Right"}
        )
        for shortcut in widget.playback_shortcuts:
            self.assertIs(shortcut.parent(), widget)
            self.assertEqual(
                shortcut.context(),
                Qt.ShortcutContext.WidgetWithChildrenShortcut,
            )

    # -- lifecycle ----------------------------------------------------------
    def test_shutdown_joins_the_scanner_and_closes_the_pipeline(self):
        pipeline = _slow_pipeline_class()
        with patch.object(tst_tuner, "TSTFramePipeline", pipeline):
            widget = self._widget()
            widget.show()
            self.assertTrue(_wait_until(
                self.app, lambda: bool(pipeline.created)
            ))
            scanner = widget.scanner
            self.assertTrue(scanner.isRunning())

            widget.shutdown()

            self.assertFalse(scanner.isRunning())
            self.assertGreaterEqual(len(pipeline.created), 1)
            self.assertEqual(len(pipeline.closed), len(pipeline.created))
            self.assertFalse(widget._preview_capture.is_open)
            self.assertIsNone(widget.get_decoded_frame(0))
            widget.start_scan()  # refused once shut down
            self.assertIs(widget.scanner, scanner)
            widget.shutdown()  # idempotent

    def test_hiding_mid_scan_stops_it_and_showing_runs_it_again(self):
        pipeline = _slow_pipeline_class()
        with patch.object(tst_tuner, "TSTFramePipeline", pipeline):
            widget = self._widget()
            widget.show()
            self.assertTrue(_wait_until(
                self.app, lambda: bool(pipeline.created)
            ))
            first = widget.scanner

            widget.hide()

            self.assertFalse(first.isRunning())
            self.assertEqual(len(pipeline.closed), len(pipeline.created))
            self.assertFalse(widget.has_unsaved_changes())

            widget.show()
            self.assertIsNot(widget.scanner, first)
            self.assertTrue(widget.scanner.isRunning())
            widget.shutdown()
            self.assertFalse(widget.scanner.isRunning())

    def test_deleting_a_parent_mid_scan_is_safe(self):
        # Repeated: the original crash was a race on object destruction.
        pipeline = _slow_pipeline_class()
        with patch.object(tst_tuner, "TSTFramePipeline", pipeline):
            for attempt in range(10):
                with self.subTest(attempt=attempt):
                    parent = QWidget()
                    layout = QVBoxLayout(parent)
                    widget = TstTunerWidget(
                        self.video, self.roi_manager, self._maze(),
                        parent=parent,
                    )
                    layout.addWidget(widget)
                    parent.show()
                    self.assertTrue(_wait_until(
                        self.app, lambda: widget.scanner is not None
                        and widget.scanner.isRunning(),
                    ))
                    scanner = widget.scanner

                    if attempt % 2:
                        parent.deleteLater()
                        QCoreApplication.sendPostedEvents(
                            None, QEvent.Type.DeferredDelete.value
                        )
                    else:
                        sip.delete(parent)
                    self.app.processEvents()

                    self.assertFalse(scanner.isRunning())
            self.assertEqual(len(pipeline.closed), len(pipeline.created))

    # -- dialog wrapper -----------------------------------------------------
    def test_dialog_wrapper_matches_the_legacy_maze_configuration(self):
        """Scripted tuning through the dialog gives the pre-refactor result."""
        maze = self._maze()
        dialog = TstTunerDialog(self.video, self.roi_manager, maze)
        self.addCleanup(dialog.deleteLater)
        try:
            self.assertFalse(dialog.accept_button.isEnabled())
            dialog.show()
            self.assertTrue(_wait_until(
                self.app, dialog.accept_button.isEnabled
            ))

            dialog.time_slider.setValue(17)
            dialog.advanced_controls["active_residual_threshold"].setValue(
                0.02
            )
            dialog.accept_button.click()
            self.assertEqual(dialog.result(), 1)
        finally:
            dialog.done(dialog.result())

        # What the old dialog's accept() did, written out independently.
        expected = self._maze()
        values = dict(TSTMotionConfig().__dict__)
        values["active_residual_threshold"] = 0.02
        expected.motion_config = TSTMotionConfig(**values)
        expected.set_parameters(17 / 10.0)
        self.assertEqual(
            maze.get_algorithm_parameters(), expected.get_algorithm_parameters()
        )

    def test_dialog_wrapper_warns_and_stays_open_on_invalid_settings(self):
        maze = self._maze()
        before = copy.deepcopy(maze.get_algorithm_parameters())
        with patch.object(TstTunerWidget, "start_scan"):
            dialog = TstTunerDialog(self.video, self.roi_manager, maze)
        self.addCleanup(dialog.deleteLater)
        try:
            dialog.preview_motion_config.analysis_width = -1
            with patch("gui.tst_tuner.QMessageBox.warning") as warning:
                dialog.accept()
            warning.assert_called_once()
            self.assertEqual(dialog.result(), 0)
            self.assertEqual(maze.get_algorithm_parameters(), before)
        finally:
            dialog.cleanup()


if __name__ == "__main__":
    unittest.main()
