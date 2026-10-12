import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    import numpy as np
    from PyQt6.QtCore import QPoint
    from PyQt6.QtWidgets import QApplication

    from core.tst_display import tst_display_style
    from core.tst_motion_engine import TSTMotionConfig, TSTMotionEngine
    from gui.tst_tuner import EnergyScanner, TstTunerDialog
    from mazes.tst import TST
    from roi_manager import ROIManager
    HAS_GUI = True
except ModuleNotFoundError:
    HAS_GUI = False


@unittest.skipUnless(HAS_GUI, "Qt/scientific runtime is not installed")
class TSTTunerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_multi_mouse_selector_and_saved_duration_are_initialized(self):
        roi_manager = ROIManager(source_dimensions=(100, 100))
        roi_manager.add_roi(
            "mouse_1",
            np.array([[5, 5], [45, 5], [45, 95], [5, 95]]),
        )
        roi_manager.add_roi(
            "mouse_2",
            np.array([[55, 5], [95, 5], [95, 95], [55, 95]]),
        )
        tst = TST()
        tst.configure_mice(2)
        tst.set_parameters(1.7)

        with patch.object(TstTunerDialog, "start_scan"):
            dialog = TstTunerDialog(
                Path("not-opened-during-this-test.mp4"),
                roi_manager,
                tst,
            )
        try:
            self.assertEqual(dialog.mouse_selector.count(), 2)
            self.assertEqual(dialog.mouse_selector.currentData(), "mouse_1")
            self.assertEqual(dialog.time_slider.value(), 17)
            self.assertFalse(dialog.energy_slider.isEnabled())
            dialog.resize(640, 480)
            dialog.show()
            self.app.processEvents()
            self.assertTrue(dialog.accept_button.isVisible())
            self.assertTrue(dialog.cancel_button.isVisible())
            self.assertFalse(
                dialog.content_scroll.isAncestorOf(dialog.accept_button)
            )
            self.assertGreater(
                dialog.content_scroll.verticalScrollBar().maximum(), 0
            )
            button_bottom = (
                dialog.accept_button.mapTo(dialog, QPoint(0, 0)).y()
                + dialog.accept_button.height()
            )
            self.assertLessEqual(button_bottom, dialog.contentsRect().bottom())
        finally:
            dialog.cleanup()
            dialog.close()

    def test_switching_between_populated_mouse_traces_does_not_raise(self):
        roi_manager = ROIManager(source_dimensions=(100, 100))
        roi_manager.add_roi(
            "mouse_1",
            np.array([[5, 5], [45, 5], [45, 95], [5, 95]]),
        )
        roi_manager.add_roi(
            "mouse_2",
            np.array([[55, 5], [95, 5], [95, 95], [55, 95]]),
        )
        tst = TST()
        tst.configure_mice(2)
        with patch.object(TstTunerDialog, "start_scan"):
            dialog = TstTunerDialog(Path("not-opened.mp4"), roi_manager, tst)
        try:
            dialog.fps = 10.0
            dialog.scan_len = 4
            dialog.frame_buffer = [0, 1, 2, 3]
            dialog.energy_data = {
                "mouse_1": [2.0, 3.0, 4.0, 5.0],
                "mouse_2": [8.0, 7.0, 6.0, 5.0],
            }
            dialog.state_data = {
                "mouse_1": ["immobile_static"] * 4,
                "mouse_2": ["active_escape"] * 4,
            }
            dialog.init_plot_optimized(dialog.energy_data["mouse_1"])
            dialog.on_param_change()

            dialog.mouse_selector.setCurrentIndex(1)
            self.app.processEvents()

            self.assertEqual(dialog.selected_mouse, "mouse_2")
            self.assertIs(dialog.fill_immobile.axes, dialog.ax)
            self.assertIs(dialog.fill_pause.axes, dialog.ax)
        finally:
            dialog.cleanup()
            dialog.close()

    def _dialog(self):
        roi_manager = ROIManager(source_dimensions=(100, 100))
        roi_manager.add_roi(
            "mouse_1",
            np.array([[5, 5], [45, 5], [45, 95], [5, 95]]),
        )
        with patch.object(TstTunerDialog, "start_scan"):
            return TstTunerDialog(Path("not-opened.mp4"), roi_manager, TST())

    def test_zoom_and_reset_before_a_scan_are_no_ops(self):
        dialog = self._dialog()
        try:
            self.assertEqual(dialog.timestamps.size, 0)
            dialog.zoom_in()
            dialog.zoom_out()
            dialog.reset_zoom()
        finally:
            dialog.cleanup()
            dialog.close()

    def _load_profile_text(self, dialog, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(text, encoding="utf-8")
            with patch(
                "gui.tst_tuner.QFileDialog.getOpenFileName",
                return_value=(str(path), ""),
            ), patch("gui.tst_tuner.QMessageBox.warning") as warning:
                dialog._load_profile()
        return warning

    def test_invalid_profiles_are_reported_not_raised(self):
        dialog = self._dialog()
        engine_id = TSTMotionEngine.ENGINE_ID
        valid_config = dict(TSTMotionConfig().__dict__)
        try:
            for text in (
                "[1, 2, 3]",
                "not json at all",
                json.dumps({"engine_id": "tst_motion_decomposition_v2"}),
                json.dumps({"engine_id": engine_id, "motion_config": []}),
                json.dumps({
                    "engine_id": engine_id,
                    "motion_config": {**valid_config, "surprise": 1.0},
                }),
                json.dumps({
                    "engine_id": engine_id,
                    "motion_config": valid_config,
                    "min_immobility_seconds": "soon",
                }),
                json.dumps({
                    "engine_id": engine_id,
                    "motion_config": valid_config,
                    "min_immobility_seconds": None,
                }),
            ):
                warning = self._load_profile_text(dialog, text)
                self.assertEqual(warning.call_count, 1, text)
        finally:
            dialog.cleanup()
            dialog.close()

    def test_valid_profile_is_loaded(self):
        dialog = self._dialog()
        try:
            config = dict(TSTMotionConfig().__dict__)
            config["active_residual_threshold"] = 0.02
            warning = self._load_profile_text(dialog, json.dumps({
                "engine_id": TSTMotionEngine.ENGINE_ID,
                "motion_config": config,
                "min_immobility_seconds": 2.5,
            }))
            warning.assert_not_called()
            self.assertEqual(dialog.time_slider.value(), 25)
            self.assertAlmostEqual(
                dialog.preview_motion_config.active_residual_threshold, 0.02
            )
        finally:
            dialog.cleanup()
            dialog.close()

    def test_saved_profile_carries_the_engine_identifier(self):
        dialog = self._dialog()
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "profile.json"
                with patch(
                    "gui.tst_tuner.QFileDialog.getSaveFileName",
                    return_value=(str(path), ""),
                ):
                    dialog._save_profile()
                payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["engine_id"], TSTMotionEngine.ENGINE_ID
            )
            self.assertEqual(payload["motion_config"]["analysis_width"], 480)
        finally:
            dialog.cleanup()
            dialog.close()

    @staticmethod
    def _scanner(path):
        roi = ROIManager(source_dimensions=(100, 100))
        roi.add_roi(
            "mouse_1", np.array([[5, 5], [45, 5], [45, 95], [5, 95]])
        )
        rois = {name: items[0] for name, items in roi.rois.items()}
        return EnergyScanner(path, rois, TSTMotionConfig(), [(0, 10)])

    def test_scanner_emits_scan_complete_for_unreadable_video(self):
        scanner = self._scanner(Path("definitely-missing-video.mp4"))
        completed = []
        scanner.scan_complete.connect(lambda *args: completed.append(args))
        scanner.run()
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0][0], {})
        self.assertEqual(completed[0][1], [])

    def test_scanner_emits_empty_result_when_the_scan_raises(self):
        scanner = self._scanner(Path("unused.mp4"))
        completed = []
        scanner.scan_complete.connect(lambda *args: completed.append(args))
        with patch.object(
            EnergyScanner, "_scan", side_effect=RuntimeError("boom")
        ):
            scanner.run()
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0][0], {})

    def test_scanner_releases_capture_on_its_own_thread_only(self):
        scanner = self._scanner(Path("unused.mp4"))
        capture = MagicMock()
        scanner._capture = capture
        with patch.object(scanner, "requestInterruption") as request:
            scanner.cancel()
        request.assert_called_once()
        capture.release.assert_not_called()
        with patch.object(EnergyScanner, "_scan"):
            scanner.run()
        capture.release.assert_called_once()
        self.assertIsNone(scanner._capture)

    def test_display_style_mapping_is_shared_by_all_states(self):
        self.assertEqual(tst_display_style("uncertain", False)[1], "UNCERTAIN")
        self.assertEqual(
            tst_display_style("immobile_static", None)[1], "UNCERTAIN"
        )
        self.assertEqual(
            tst_display_style("passive_swing", True)[1], "PASSIVE SWING"
        )
        self.assertEqual(
            tst_display_style("immobile_static", True)[1], "IMMOBILE STATIC"
        )
        self.assertEqual(
            tst_display_style("passive_swing", False)[1], "SHORT PAUSE"
        )
        self.assertEqual(
            tst_display_style("active_escape", False)[1], "ACTIVE ESCAPE"
        )


if __name__ == "__main__":
    unittest.main()
