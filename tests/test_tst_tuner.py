import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
    from PyQt6.QtCore import QPoint
    from PyQt6.QtWidgets import QApplication

    from gui.tst_tuner import TstTunerDialog
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


if __name__ == "__main__":
    unittest.main()
