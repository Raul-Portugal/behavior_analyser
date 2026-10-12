import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from gui.freestyle_config_dialog import (
        FreestyleConfigDialog,
        is_reserved_zone_name,
    )
    from mazes.freestyle import Freestyle
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class FreestyleConfigDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _dialog(self, maze=None):
        dialog = FreestyleConfigDialog(maze=maze)
        self.addCleanup(dialog.deleteLater)
        return dialog

    def _accept(self, dialog):
        """Run validation, returning the warning titles that were shown."""
        with patch.object(QMessageBox, "warning") as warning:
            dialog.validate_and_accept()
        return [call.args[1] for call in warning.call_args_list]

    def test_mode_radios_are_exclusive_and_never_both_unchecked(self):
        dialog = self._dialog()
        based, free = dialog.zone_based_radio, dialog.zone_free_radio

        self.assertTrue(dialog.mode_button_group.exclusive())
        self.assertTrue(based.isChecked() and not free.isChecked())

        # Clicking the already-selected option must not deselect anything.
        based.click()
        self.assertTrue(based.isChecked() and not free.isChecked())
        self.assertFalse(dialog.zone_free_mode)

        free.click()
        self.assertTrue(free.isChecked() and not based.isChecked())
        self.assertTrue(dialog.zone_free_mode)
        self.assertFalse(dialog.zone_config_group.isEnabled())

        free.click()
        self.assertTrue(free.isChecked())
        free.setChecked(False)  # programmatic uncheck is ignored too
        self.assertTrue(free.isChecked() and not based.isChecked())

        based.click()
        self.assertTrue(dialog.zone_config_group.isEnabled())
        self.assertFalse(dialog.zone_free_mode)

    def test_reserved_zone_names_are_detected(self):
        for name in ("outside", "Outside", "undetected", "UNDETECTED",
                     "mouse_1", "Mouse 2", "mouse_anything"):
            self.assertTrue(is_reserved_zone_name(name), name)
        for name in ("Zone A", "mouse", "mousetrap", "outside area", "North"):
            self.assertFalse(is_reserved_zone_name(name), name)

    def test_reserved_names_keep_the_dialog_open(self):
        for reserved in ("Outside", "undetected", "Mouse 1", "MOUSE_7"):
            with self.subTest(name=reserved):
                dialog = self._dialog()
                dialog.zone_name_inputs[1].setText(reserved)

                titles = self._accept(dialog)

                self.assertEqual(titles, ["Reserved Zone Name"])
                self.assertEqual(dialog.result(), 0)
                self.assertEqual(dialog.get_zone_definitions(), [])

    def test_valid_names_are_accepted_as_internal_display_pairs(self):
        dialog = self._dialog()
        dialog.num_zones_spin.setValue(2)
        dialog.zone_name_inputs[0].setText("Open Arm")
        dialog.zone_name_inputs[1].setText("Centre")

        self.assertEqual(self._accept(dialog), [])

        self.assertEqual(
            dialog.get_zone_definitions(),
            [("open_arm", "Open Arm"), ("centre", "Centre")],
        )
        self.assertEqual(dialog.result(), 1)

    def test_changing_zone_count_keeps_typed_names(self):
        dialog = self._dialog()
        dialog.zone_name_inputs[0].setText("Home")
        dialog.zone_name_inputs[1].setText("Feeder")
        dialog.zone_name_inputs[2].setText("Novel Object")

        dialog.num_zones_spin.setValue(5)
        self.assertEqual(
            [edit.text() for edit in dialog.zone_name_inputs],
            ["Home", "Feeder", "Novel Object", "", ""],
        )

        dialog.num_zones_spin.setValue(2)
        self.assertEqual(
            [edit.text() for edit in dialog.zone_name_inputs],
            ["Home", "Feeder"],
        )

        dialog.num_zones_spin.setValue(3)
        self.assertEqual(
            [edit.text() for edit in dialog.zone_name_inputs],
            ["Home", "Feeder", ""],
        )
        self.assertEqual(dialog.zone_list_widget.count(), 3)
        self.assertEqual(dialog.zone_list_widget.item(0).text(), "1. Home")

    def test_parameters_round_trip_into_the_maze(self):
        maze = Freestyle()
        dialog = self._dialog(maze)
        self.assertEqual(dialog.immobility_threshold_spin.value(), 2.0)
        self.assertEqual(dialog.speed_window_spin.value(), 0.2)
        self.assertEqual(dialog.min_immobility_spin.value(), 1.0)
        self.assertAlmostEqual(dialog.min_dwell_spin.value(), 0.10)

        dialog.immobility_threshold_spin.setValue(1.5)
        dialog.speed_window_spin.setValue(0.35)
        dialog.min_immobility_spin.setValue(2.5)
        dialog.min_dwell_spin.setValue(0.25)
        self.assertEqual(self._accept(dialog), [])

        self.assertEqual(maze.immobility_threshold_cm_per_s, 1.5)
        self.assertEqual(maze.speed_window_seconds, 0.35)
        self.assertEqual(maze.min_immobility_seconds, 2.5)
        self.assertEqual(maze.min_dwell_seconds, 0.25)
        saved = maze.get_algorithm_parameters()
        self.assertEqual(saved["immobility_threshold_cm_per_s"], 1.5)

        # A fresh dialog reflects the maze's stored values.
        reopened = self._dialog(maze)
        self.assertEqual(reopened.get_analysis_parameters(), {
            "immobility_threshold_cm_per_s": 1.5,
            "speed_window_seconds": 0.35,
            "min_immobility_seconds": 2.5,
            "min_dwell_seconds": 0.25,
        })

    def test_maze_is_untouched_when_validation_fails(self):
        maze = Freestyle()
        dialog = self._dialog(maze)
        dialog.immobility_threshold_spin.setValue(9.0)
        dialog.zone_name_inputs[0].setText("outside")

        self._accept(dialog)

        self.assertEqual(maze.immobility_threshold_cm_per_s, 2.0)

    def test_zone_free_mode_still_applies_movement_parameters(self):
        maze = Freestyle()
        dialog = self._dialog(maze)
        dialog.zone_free_radio.click()
        dialog.min_immobility_spin.setValue(3.0)

        self.assertEqual(self._accept(dialog), [])

        self.assertTrue(dialog.is_zone_free_mode())
        self.assertEqual(dialog.get_zone_definitions(), [])
        self.assertEqual(maze.min_immobility_seconds, 3.0)


if __name__ == "__main__":
    unittest.main()
