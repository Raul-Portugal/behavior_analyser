import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
    from PyQt6 import sip
    from PyQt6.QtCore import QCoreApplication, QEvent, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget

    from core.models import DetectionConfig
    from gui.tuner_dialog import TunerDialog
    from gui.tuners import detection_tuner
    from gui.tuners.detection_tuner import DetectionTunerWidget
    from roi_manager import ROIManager
    from tests.test_tuner_dialog import _write_video
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


if HAS_RUNTIME:
    class _SlowWorker(detection_tuner.QuickScanWorker):
        """A quick scan that keeps running until it is interrupted."""

        def run(self):
            while not self.isInterruptionRequested():
                time.sleep(0.005)


def _wait_until(app, predicate, seconds=20.0):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    return predicate()


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class DetectionTunerWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.video = Path(tmp.name) / "clip.avi"
        _write_video(self.video, 40)
        self.reference = np.full((48, 64), 40, dtype=np.uint8)

    def _widget(self, initial_config=None, roi_manager=None, parent=None,
                embedded=False):
        widget = DetectionTunerWidget(
            str(self.video), self.reference,
            roi_manager or ROIManager(source_dimensions=(64, 48)),
            initial_config=initial_config, parent=parent, embedded=embedded,
        )
        self.addCleanup(widget.shutdown)
        return widget

    # -- the result of the detector ------------------------------------------
    def test_the_result_is_a_label_not_text_drawn_into_the_frame(self):
        for embedded in (False, True):
            with self.subTest(embedded=embedded):
                widget = self._widget(embedded=embedded)
                with patch.object(detection_tuner.cv2, "putText") as put_text:
                    widget.update_detection()
                put_text.assert_not_called()
                text = widget.detect_label.text()
                self.assertTrue(text.startswith(("Confidence:", "Not detected:")), text)

    def test_the_label_sits_in_the_row_under_the_preview(self):
        widget = self._widget(embedded=True)
        self.assertGreaterEqual(widget.footer_layout.indexOf(widget.detect_label), 0)
        self.assertLess(widget.footer_layout.indexOf(widget.frame_label),
                        widget.footer_layout.indexOf(widget.detect_label))

    # -- initial configuration -------------------------------------------
    def test_without_initial_config_the_defaults_are_used(self):
        widget = self._widget()
        self.assertEqual(widget.detection_config(), DetectionConfig())

    def test_initial_config_is_loaded_into_the_sliders_exactly(self):
        initial = DetectionConfig(
            threshold_percentile=98.5,
            weight_omega=0.55,
            window_size=150,
            min_component_area=60,
            morphology_kernel_size=5,
            min_detection_confidence=0.3,
            min_absolute_contrast=12.0,
            use_weighting=False,
            max_expected_jump_ratio=0.35,
        )
        widget = self._widget(initial)

        self.assertEqual(
            {name: slider.value() for name, slider in widget.sliders.items()},
            {
                "Threshold percentile": 985,
                "Weight omega": 55,
                "Search window": 150,
                "Min component area": 60,
                "Morphology kernel": 5,
                "Min confidence": 30,
            },
        )
        self.assertEqual(
            widget.slider_labels["Threshold percentile"].text(),
            "Threshold percentile: 98.5",
        )
        self.assertEqual(
            widget.slider_labels["Weight omega"].text(), "Weight omega: 0.55"
        )
        # Fields without a slider survive untouched, and so does the rest.
        self.assertEqual(widget.detection_config(), initial)
        self.assertEqual(widget.detection_config().min_absolute_contrast, 12.0)
        self.assertFalse(widget.has_unsaved_changes())

    def test_initial_config_is_copied_not_aliased(self):
        initial = DetectionConfig(window_size=120)
        widget = self._widget(initial)
        widget.sliders["Search window"].setValue(80)
        self.assertEqual(initial.window_size, 120)
        self.assertEqual(widget.detection_config().window_size, 80)

    def test_values_the_sliders_cannot_represent_survive_until_moved(self):
        initial = DetectionConfig(threshold_percentile=98.55, window_size=10)
        widget = self._widget(initial)
        config = widget.detection_config()
        self.assertEqual(config.threshold_percentile, 98.55)
        self.assertEqual(config.window_size, 10)
        # The slider shows the nearest step in range ...
        self.assertEqual(widget.sliders["Search window"].value(), 20)
        # ... and moving it makes the slider the source of truth.
        widget.sliders["Search window"].setValue(40)
        self.assertEqual(widget.detection_config().window_size, 40)
        self.assertEqual(
            widget.detection_config().threshold_percentile, 98.55
        )

    def test_arena_mask_is_built_from_the_roi_manager(self):
        manager = ROIManager(source_dimensions=(64, 48))
        manager.set_arena(np.array([[8, 6], [56, 6], [56, 42], [8, 42]]))
        widget = self._widget(
            DetectionConfig(min_absolute_contrast=9.0), roi_manager=manager
        )
        self.assertIsNotNone(widget.arena_mask)
        self.assertEqual(widget.arena_mask.shape, widget.analysis_ref.shape)
        self.assertGreater(int(np.count_nonzero(widget.arena_mask)), 0)
        self.assertEqual(widget.detection_config().min_absolute_contrast, 9.0)
        # No arena -> no restriction.
        self.assertIsNone(self._widget().arena_mask)

    # -- signals and state -------------------------------------------------
    def test_config_changed_carries_the_new_config(self):
        widget = self._widget(DetectionConfig(min_absolute_contrast=7.0))
        received = []
        widget.config_changed.connect(received.append)

        widget.sliders["Min confidence"].setValue(40)

        self.assertEqual(len(received), 1)
        self.assertIsInstance(received[0], DetectionConfig)
        self.assertAlmostEqual(received[0].min_detection_confidence, 0.4)
        self.assertEqual(received[0].min_absolute_contrast, 7.0)
        self.assertTrue(widget.has_unsaved_changes())

    def test_frame_navigation_keys_work_on_the_focused_widget(self):
        widget = self._widget()
        start = widget.frame_slider.value()
        QTest.keyClick(widget, Qt.Key.Key_N)
        self.assertEqual(widget.frame_slider.value(), start + 1)
        QTest.keyClick(widget, Qt.Key.Key_P)
        self.assertEqual(widget.frame_slider.value(), start)

    # -- lifecycle ---------------------------------------------------------
    def test_shutdown_joins_the_quick_scan_and_releases_the_capture(self):
        widget = self._widget()
        with patch.object(detection_tuner, "QuickScanWorker", _SlowWorker):
            widget.run_quick_scan()
        worker = widget.scan_worker
        self.assertTrue(worker.isRunning())
        self.assertTrue(widget.cap.isOpened())

        widget.shutdown()

        self.assertFalse(worker.isRunning())
        self.assertFalse(widget.cap.isOpened())
        self.assertFalse(widget.stats_timer.isActive())
        widget.shutdown()  # idempotent
        widget.run_quick_scan()  # refused after shutdown
        self.assertIs(widget.scan_worker, worker)

    def test_hiding_pauses_work_and_showing_resumes_it(self):
        widget = self._widget()
        widget.show()
        self.app.processEvents()
        self.assertTrue(widget.stats_timer.isActive())
        with patch.object(detection_tuner, "QuickScanWorker", _SlowWorker):
            widget.run_quick_scan()
        worker = widget.scan_worker

        widget.hide()

        self.assertFalse(worker.isRunning())
        self.assertFalse(widget.stats_timer.isActive())
        self.assertFalse(widget._capture.is_open)

        widget.show()
        self.assertTrue(widget.stats_timer.isActive())
        widget.update_detection()  # the capture reopens on demand
        self.assertTrue(widget._capture.is_open)
        self.assertTrue(_wait_until(
            self.app, widget.test_scan_button.isEnabled
        ))

    def test_close_event_shuts_the_widget_down(self):
        widget = self._widget()
        widget.show()
        widget.close()
        self.assertFalse(widget.cap.isOpened())
        self.assertFalse(widget.stats_timer.isActive())

    def test_deleting_a_parent_mid_scan_is_safe(self):
        # Repeated: the original crash was a race on object destruction.
        for attempt in range(12):
            with self.subTest(attempt=attempt):
                parent = QWidget()
                layout = QVBoxLayout(parent)
                widget = DetectionTunerWidget(
                    str(self.video), self.reference,
                    ROIManager(source_dimensions=(64, 48)), parent=parent,
                )
                layout.addWidget(widget)
                parent.show()
                worker_class = (
                    _SlowWorker if attempt % 2 == 0
                    else detection_tuner.QuickScanWorker
                )
                with patch.object(
                    detection_tuner, "QuickScanWorker", worker_class
                ):
                    widget.run_quick_scan()
                worker = widget.scan_worker
                self.assertTrue(worker.isRunning() or attempt % 2)

                if attempt % 3 == 0:
                    parent.deleteLater()
                    QCoreApplication.sendPostedEvents(
                        None, QEvent.Type.DeferredDelete.value
                    )
                else:
                    sip.delete(parent)
                self.app.processEvents()

                self.assertFalse(worker.isRunning())

    def test_dialog_wrapper_matches_the_widget_configuration(self):
        initial = DetectionConfig(window_size=130, min_absolute_contrast=5.0)
        dialog = TunerDialog(
            str(self.video), self.reference,
            ROIManager(source_dimensions=(64, 48)), initial_config=initial,
        )
        self.addCleanup(dialog.deleteLater)
        try:
            self.assertEqual(dialog.get_detection_config(), initial)
            self.assertFalse(dialog.has_unsaved_work())
            dialog.sliders["Morphology kernel"].setValue(1)
            self.assertTrue(dialog.has_unsaved_work())
            self.assertEqual(
                dialog.get_detection_config(), dialog.widget.detection_config()
            )
            self.assertEqual(
                dialog.get_detection_config().morphology_kernel_size, 1
            )
        finally:
            dialog.done(0)


if __name__ == "__main__":
    unittest.main()
