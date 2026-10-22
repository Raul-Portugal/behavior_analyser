"""Redraw in verification opens the ROI editor preloaded, not blank."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402

try:
    from PyQt6.QtWidgets import QApplication
    from core.models import AppConfig, BatchSettings
    from gui.main_window import MainWindow
    from gui.roi_selector import RoiSelectorDialog
    from mazes import create_maze
    from roi_manager import ROIManager
    from tests import worker_support as support
    from tests.gui_support import temp_app_settings
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


def _square(offset=0, size=40):
    return np.array(
        [[offset, 0], [offset + size, 0], [offset + size, size], [offset, size]]
    )


class _VerificationDialog:
    """Stands in for SettingsDialog: asks to redraw once, then accepts."""
    instances = []

    def __init__(self, video_path, settings, is_verification=False, parent=None,
                 maze_name="", real_world_cm=None):
        self.settings = settings
        self.index = len(self.instances)
        self.instances.append(self)
        self.result_action = "redraw" if self.index == 0 else "accept"

    def setWindowTitle(self, _title):
        pass

    def exec(self):
        return True

    def get_settings(self):
        return self.settings

    def get_real_world_cm(self):
        return 30.0

    def deleteLater(self):
        pass


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class RedrawOpensPreloadedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.video = support.write_tracking_video(Path(tmp.name), frames=20)
        self.window = MainWindow(temp_app_settings(self))
        self.addCleanup(self.window.deleteLater)
        self.addCleanup(self.window._detach_log_handler)
        self.window.maze = create_maze("Y-Maze")
        self.window.video_paths = [self.video]
        _VerificationDialog.instances = []

    def _settings(self):
        manager = ROIManager(source_dimensions=(160, 120))
        manager.add_roi("arm_a", _square(0))
        manager.add_roi("arm_b", _square(50))
        manager.set_reference_line((0, 100), (50, 100), "User-drawn reference line")
        return BatchSettings(manager, AppConfig().detection, 0.0)

    def test_redraw_passes_the_current_rois_to_the_editor(self):
        base = self._settings()
        calls = []

        class FakeRoiDialog:
            def __init__(self, video, maze, parent=None, initial_roi_manager=None):
                calls.append(initial_roi_manager)
                self.manager = ROIManager(source_dimensions=(160, 120))
                self.manager.add_roi("arm_a", _square(5))
                self.manager.set_reference_line((0, 100), (100, 100), "User-drawn reference line")

            def exec(self):
                return True

            def get_roi_manager(self):
                return self.manager

            def deleteLater(self):
                pass

        with patch("gui.main_window.SettingsDialog", _VerificationDialog), \
                patch("gui.main_window.RoiSelectorDialog", FakeRoiDialog), \
                patch.object(self.window, "_ask_partial_verification"), \
                patch("gui.main_window.ux_helpers.ask_choice", return_value="run"):
            plan = self.window.build_analysis_plan(base, verify_each=True)

        self.assertEqual(len(calls), 1)
        preloaded = calls[0]
        self.assertIsNotNone(preloaded, "Redraw must open the editor preloaded")
        self.assertEqual(sorted(preloaded.rois), ["arm_a", "arm_b"])
        np.testing.assert_array_equal(preloaded.rois["arm_b"][0].points, _square(50))
        self.assertEqual(preloaded.reference_line, ((0.0, 100.0), (50.0, 100.0)))
        # The redrawn ROIs and a scale from the typed 30 cm replace the old ones.
        settings = plan[self.video]
        self.assertEqual(list(settings.roi_manager.rois), ["arm_a"])
        self.assertAlmostEqual(settings.scale_factor, 30.0 / 100.0)

    def test_the_real_dialog_opens_with_those_rois_and_returns_the_edit(self):
        base = self._settings()
        dialog = RoiSelectorDialog(
            str(self.video), self.window.maze, initial_roi_manager=base.roi_manager
        )
        self.addCleanup(dialog.deleteLater)
        self.assertEqual(dialog.roi_manager.rois.keys(), base.roi_manager.rois.keys())
        self.assertEqual(dialog.editor.line_points(), base.roi_manager.reference_line)
        self.assertFalse(dialog.has_unsaved_work())
        # Editing one polygon and accepting keeps the rest as drawn.
        manager = dialog.editor.roi_manager()
        np.testing.assert_array_equal(
            manager.rois["arm_a"][0].points, base.roi_manager.rois["arm_a"][0].points
        )
        self.assertEqual(manager.source_dimensions, (160, 120))


if __name__ == "__main__":
    unittest.main()
