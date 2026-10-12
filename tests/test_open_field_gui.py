import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402

try:
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from gui.freestyle_config_dialog import (
        FreestyleConfigDialog,
        is_reserved_zone_name,
    )
    from gui.roi_selector import RoiSelectorDialog
    from mazes.freestyle import Freestyle
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False

from core.models import BatchSettings, DetectionConfig  # noqa: E402
from roi_manager import ROIManager  # noqa: E402


class _Capture:
    """A 1920 x 1080 video, so clicks are scaled by 1.5 into source pixels."""

    def get(self, _property):
        return 20

    def set(self, _property, _value):
        return True

    def read(self):
        return True, np.zeros((1080, 1920, 3), dtype=np.uint8)

    def release(self):
        pass


def _click_polygon(dialog, points):
    dialog.current_polygon_points = [QPointF(x, y) for x, y in points]
    return dialog.finalize_polygon()


SQUARE = [(100, 100), (500, 100), (500, 500), (100, 500)]


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class FreestyleArenaDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _dialog(self, maze=None):
        dialog = FreestyleConfigDialog(maze=maze)
        self.addCleanup(dialog.deleteLater)
        return dialog

    def _accept(self, dialog):
        with patch.object(QMessageBox, "warning") as warning:
            dialog.validate_and_accept()
        return [call.args[1] for call in warning.call_args_list]

    def test_arena_is_off_by_default_and_changes_no_parameters(self):
        maze = Freestyle()
        before = maze.get_algorithm_parameters()
        dialog = self._dialog(maze)

        self.assertFalse(dialog.arena_checkbox.isChecked())
        self.assertFalse(dialog.arena_options.isEnabled())
        self.assertFalse(dialog.is_arena_enabled())
        self.assertNotIn("arena_mode", dialog.get_analysis_parameters())
        self.assertEqual(self._accept(dialog), [])

        self.assertFalse(maze.arena_mode)
        self.assertEqual(maze.get_algorithm_parameters(), before)

    def test_center_controls_default_and_enable_with_the_checkbox(self):
        dialog = self._dialog()
        dialog.arena_checkbox.setChecked(True)

        self.assertTrue(dialog.arena_options.isEnabled())
        self.assertTrue(dialog.center_distance_radio.isChecked())
        self.assertFalse(dialog.center_area_radio.isChecked())
        self.assertEqual(dialog.center_distance_spin.value(), 10.0)
        self.assertEqual(dialog.center_area_spin.value(), 25.0)
        self.assertEqual(dialog.center_mode(), "distance_cm")
        self.assertEqual(dialog.center_area_fraction(), 0.25)
        self.assertEqual(
            dialog.get_analysis_parameters()["arena_center_mode"], "distance_cm"
        )

        dialog.arena_checkbox.setChecked(False)
        self.assertFalse(dialog.arena_options.isEnabled())

    def test_center_radios_are_exclusive(self):
        dialog = self._dialog()
        dialog.arena_checkbox.setChecked(True)
        dialog.center_area_radio.click()
        self.assertTrue(dialog.center_area_radio.isChecked())
        self.assertFalse(dialog.center_distance_radio.isChecked())
        self.assertEqual(dialog.center_mode(), "area_fraction")
        dialog.center_distance_radio.click()
        self.assertEqual(dialog.center_mode(), "distance_cm")

    def test_choices_are_applied_to_the_maze_and_reloaded(self):
        maze = Freestyle()
        dialog = self._dialog(maze)
        dialog.arena_checkbox.setChecked(True)
        dialog.center_area_radio.click()
        dialog.center_area_spin.setValue(40.0)
        dialog.center_distance_spin.setValue(6.5)
        self.assertEqual(self._accept(dialog), [])

        self.assertTrue(maze.arena_mode)
        self.assertEqual(maze.arena_center_mode, "area_fraction")
        self.assertAlmostEqual(maze.arena_center_area_fraction, 0.4)
        self.assertEqual(maze.arena_center_distance_cm, 6.5)
        saved = maze.get_algorithm_parameters()
        self.assertTrue(saved["arena_mode"])

        reopened = self._dialog(maze)
        self.assertTrue(reopened.arena_checkbox.isChecked())
        self.assertTrue(reopened.center_area_radio.isChecked())
        self.assertEqual(reopened.center_area_spin.value(), 40.0)
        self.assertEqual(reopened.center_distance_spin.value(), 6.5)

    def test_unchecking_turns_the_arena_off_again(self):
        maze = Freestyle()
        maze.configure_arena(True)
        dialog = self._dialog(maze)
        self.assertTrue(dialog.arena_checkbox.isChecked())
        dialog.arena_checkbox.setChecked(False)
        self.assertEqual(self._accept(dialog), [])
        self.assertFalse(maze.arena_mode)
        self.assertNotIn("arena_mode", maze.get_algorithm_parameters())

    def test_arena_works_in_zone_free_mode(self):
        maze = Freestyle()
        dialog = self._dialog(maze)
        dialog.zone_free_radio.click()
        dialog.arena_checkbox.setChecked(True)
        self.assertEqual(self._accept(dialog), [])
        self.assertTrue(dialog.is_zone_free_mode())
        self.assertEqual(dialog.get_zone_definitions(), [])
        self.assertTrue(maze.arena_mode)
        self.assertIn("arena", dialog.info_label.text().lower())

    def test_arena_is_a_reserved_zone_name(self):
        for name in ("arena", "Arena", "ARENA"):
            self.assertTrue(is_reserved_zone_name(name), name)
        self.assertFalse(is_reserved_zone_name("arena floor"))

        dialog = self._dialog()
        dialog.zone_name_inputs[0].setText("Arena")
        self.assertEqual(self._accept(dialog), ["Reserved Zone Name"])
        self.assertEqual(dialog.get_zone_definitions(), [])

    def test_toggling_the_arena_counts_as_unsaved_work(self):
        dialog = self._dialog(Freestyle())
        self.assertFalse(dialog.has_unsaved_work())
        dialog.arena_checkbox.setChecked(True)
        self.assertTrue(dialog.has_unsaved_work())
        dialog.arena_checkbox.setChecked(False)
        self.assertFalse(dialog.has_unsaved_work())
        dialog.arena_checkbox.setChecked(True)
        dialog.center_distance_spin.setValue(7.0)
        dialog.arena_checkbox.setChecked(False)
        self.assertTrue(dialog.has_unsaved_work())


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class ArenaRoiSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _maze(self, zones=(("zone_a", "Zone A"),), arena=True, mode="distance_cm"):
        maze = Freestyle()
        maze.configure_zones(list(zones))
        maze.configure_arena(arena, mode)
        return maze

    def _dialog(self, maze):
        with patch("gui.roi_selector.cv2.VideoCapture", return_value=_Capture()):
            dialog = RoiSelectorDialog("example.mp4", maze)
        self.addCleanup(self._close, dialog)
        return dialog

    @staticmethod
    def _close(dialog):
        with patch("gui.ux_helpers.confirm_discard", return_value=True):
            dialog.close()
        dialog.deleteLater()

    def test_no_arena_step_unless_enabled(self):
        dialog = self._dialog(self._maze(arena=False))
        self.assertEqual(
            [category for category, _ in dialog.roi_definitions], ["zone_a"]
        )
        self.assertFalse(dialog._is_arena_step())

    def test_arena_is_asked_first_and_stored_apart_from_zones(self):
        dialog = self._dialog(self._maze())
        self.assertEqual(
            [category for category, _ in dialog.roi_definitions],
            ["arena", "zone_a"],
        )
        self.assertTrue(dialog._is_arena_step())
        self.assertIn("arena", dialog.instruction_label.text().lower())
        self.assertIn("Step 1/2", dialog.instruction_label.text())
        self.assertFalse(dialog.next_button.isEnabled())

        self.assertTrue(_click_polygon(dialog, SQUARE))
        manager = dialog.roi_manager
        self.assertIsNotNone(manager.arena)
        self.assertNotIn("arena", manager.rois)
        # Display pixels are scaled to the 1920 px source (x1.5).
        np.testing.assert_array_equal(
            manager.arena.points,
            np.array([[150, 150], [750, 150], [750, 750], [150, 750]]),
        )

        self.assertTrue(dialog.next_button.isEnabled())
        dialog.next_roi()
        self.assertFalse(dialog._is_arena_step())
        self.assertIn("Step 2/2", dialog.instruction_label.text())
        self.assertTrue(_click_polygon(
            dialog, [(200, 200), (300, 200), (300, 300), (200, 300)]
        ))
        self.assertEqual(list(manager.rois), ["zone_a"])
        self.assertEqual(manager.get_overlapping_zones((375, 375)), ["zone_a"])

    def test_arena_takes_exactly_one_polygon(self):
        dialog = self._dialog(self._maze())
        self.assertTrue(_click_polygon(dialog, SQUARE))
        state = dialog.step_state()
        self.assertFalse(state.can_add_point)
        self.assertTrue(state.can_advance)
        # A further click cannot start a second arena polygon.
        class _Click:
            def button(self):
                return Qt.MouseButton.LeftButton

            def pos(self):
                return QPoint(30, 30)

        dialog.add_point(_Click())
        self.assertEqual(dialog.current_polygon_points, [])
        self.assertEqual(
            dialog.roi_manager.arena.points[0].tolist(), [150, 150]
        )

    def test_too_small_arena_is_refused(self):
        dialog = self._dialog(self._maze())
        self.assertFalse(_click_polygon(
            dialog, [(10, 10), (14, 10), (14, 14), (10, 14)]
        ))
        self.assertIsNone(dialog.roi_manager.arena)

    def test_undo_reopens_the_arena(self):
        dialog = self._dialog(self._maze())
        _click_polygon(dialog, SQUARE)
        self.assertTrue(dialog.has_unsaved_work())
        dialog.undo_point()
        self.assertIsNone(dialog.roi_manager.arena)
        self.assertFalse(dialog.next_button.isEnabled())
        self.assertFalse(dialog.has_unsaved_work())

    def test_zone_free_arena_goes_straight_to_calibration(self):
        dialog = self._dialog(self._maze(zones=()))
        self.assertEqual(
            [category for category, _ in dialog.roi_definitions], ["arena"]
        )
        _click_polygon(dialog, SQUARE)
        with patch.object(
            QMessageBox, "question", return_value=QMessageBox.StandardButton.No
        ) as question:
            dialog.next_roi()
        question.assert_called_once()
        self.assertEqual(dialog.result(), dialog.DialogCode.Accepted)
        manager = dialog.get_roi_manager()
        self.assertIsNotNone(manager.arena)
        self.assertEqual(manager.rois, {})
        self.assertEqual(manager.source_dimensions, (1920, 1080))

    def test_area_fraction_center_is_previewed_and_carried_on_the_manager(self):
        dialog = self._dialog(self._maze(mode="area_fraction"))
        _click_polygon(dialog, SQUARE)
        overlay = dialog.roi_manager.center_overlay
        self.assertIsNotNone(overlay)
        # A quarter of a 600 px square: the central 300 px square.
        np.testing.assert_allclose(overlay.min(axis=0), [300, 300], atol=2)
        np.testing.assert_allclose(overlay.max(axis=0), [600, 600], atol=2)
        dialog.undo_point()
        self.assertIsNone(dialog.roi_manager.center_overlay)

    def test_distance_center_needs_calibration_so_no_preview_yet(self):
        dialog = self._dialog(self._maze(mode="distance_cm"))
        _click_polygon(dialog, SQUARE)
        self.assertIsNone(dialog.roi_manager.center_overlay)

    def test_dialog_paints_arena_and_center_preview(self):
        dialog = self._dialog(self._maze(mode="area_fraction"))
        _click_polygon(dialog, SQUARE)
        dialog.show()
        self.app.processEvents()
        image = dialog.image_label.pixmap().toImage()
        # Orange arena outline pixels exist along the polygon edge (x = 100).
        orange = [
            image.pixelColor(100, y) for y in range(150, 450, 5)
            if image.pixelColor(100, y).red() > 200
            and 100 < image.pixelColor(100, y).green() < 220
            and image.pixelColor(100, y).blue() < 80
        ]
        self.assertTrue(orange)


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class ArenaMainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from gui.main_window import MainWindow
        from tests.gui_support import temp_app_settings

        self.window = MainWindow(temp_app_settings(self))
        self.addCleanup(self.window.deleteLater)
        self.addCleanup(self.window._detach_log_handler)

    def _settings(self, **maze_arena):
        manager = ROIManager(source_dimensions=(1920, 1080))
        manager.set_arena(np.array([[0, 0], [400, 0], [400, 400], [0, 400]]))
        return BatchSettings(manager, DetectionConfig(), 0.05)

    def test_overlay_uses_calibration_for_the_distance_rule(self):
        from mazes import create_maze

        self.window.maze = create_maze("Freestyle / Open Field")
        self.window.maze.configure_arena(True, "distance_cm", 4.0)
        settings = self._settings()

        self.window._refresh_center_overlay(settings)
        outline = settings.roi_manager.center_overlay
        # 4 cm at 0.05 cm/px is an 80 px inset.
        np.testing.assert_allclose(outline.min(axis=0), [80, 80], atol=2)

        settings.scale_factor = 0.0   # uncalibrated: area-fraction fallback
        self.window._refresh_center_overlay(settings)
        np.testing.assert_allclose(
            settings.roi_manager.center_overlay.min(axis=0), [100, 100], atol=2
        )

    def test_overlay_is_cleared_without_arena_mode_and_ignored_by_other_mazes(self):
        from mazes import create_maze

        settings = self._settings()
        settings.roi_manager.set_center_overlay(np.array([[1, 1], [9, 1], [9, 9]]))
        self.window.maze = create_maze("Freestyle / Open Field")
        self.window._refresh_center_overlay(settings)
        self.assertIsNone(settings.roi_manager.center_overlay)

        self.window.maze = create_maze("Y-Maze")
        self.window._refresh_center_overlay(settings)   # no attribute: no-op

    def test_loaded_arena_configuration_restores_the_maze_without_a_zone(self):
        maze = Freestyle()
        maze.configure_zones([("zone_a", "Zone A")])
        maze.configure_arena(True, "area_fraction", 8.0, 0.3)
        settings = self._settings()
        settings.roi_manager.add_roi(
            "zone_a", np.array([[10, 10], [60, 10], [60, 60], [10, 60]])
        )
        settings.paradigm_id = maze.name
        settings.algorithm_parameters = maze.get_algorithm_parameters()

        self.window.maze = None
        self.window._infer_maze_from_settings(settings)

        restored = self.window.maze
        self.assertTrue(restored.arena_mode)
        self.assertEqual(restored.arena_center_mode, "area_fraction")
        self.assertEqual(restored.arena_center_area_fraction, 0.3)
        self.assertEqual(restored.get_roi_definitions(), [("zone_a", "Zone A")])

        # Legacy fallback (no paradigm id): the arena never becomes a zone.
        settings.paradigm_id = None
        self.window.maze = None
        self.window._infer_maze_from_settings(settings)
        zones = [name for name, _ in self.window.maze.get_roi_definitions()]
        self.assertEqual(zones, ["zone_a"])


if __name__ == "__main__":
    unittest.main()
