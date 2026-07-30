import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
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
        finally:
            dialog.cleanup()
            dialog.close()


if __name__ == "__main__":
    unittest.main()
