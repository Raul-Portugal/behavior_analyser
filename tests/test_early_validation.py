"""Problems are caught while drawing / loading, and Start is confirmed once."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from core.models import AppConfig, BatchSettings
    from gui import ux_helpers
    from gui.config_checks import (DimensionMismatch, dimension_mismatches,
                                   format_dimension_mismatches,
                                   probe_dimensions)
    from gui.main_window import MainWindow
    from gui.roi_selector import (MIN_ROI_AREA_PX, RoiRule, RoiSelectorDialog,
                                  evaluate_roi_step, roi_rule_for_maze)
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


# ----------------------------------------------------------------------
# ROI rules
# ----------------------------------------------------------------------
@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class RoiRuleTests(unittest.TestCase):
    def test_rules_per_paradigm(self):
        self.assertEqual(
            roi_rule_for_maze(create_maze("Tail Suspension Test")), RoiRule(1, 1)
        )
        self.assertEqual(roi_rule_for_maze(create_maze("Y-Maze")), RoiRule(1, None))
        self.assertEqual(
            roi_rule_for_maze(create_maze("Freestyle / Open Field")), RoiRule(1, None)
        )

    def test_nothing_drawn_blocks_next_and_says_what_is_needed(self):
        state = evaluate_roi_step(RoiRule(1, 1), 0, 0, "Mouse 1")
        self.assertFalse(state.can_advance)
        self.assertTrue(state.can_add_point)
        self.assertIn("exactly 1 polygon", state.hint)
        self.assertIn("Mouse 1", state.hint)

    def test_unfinished_shape_blocks_next(self):
        state = evaluate_roi_step(RoiRule(1, None), 1, 2, "Arm A")
        self.assertFalse(state.can_advance)
        self.assertIn("Add 1 more point", state.hint)

    def test_three_points_can_be_closed_by_next(self):
        state = evaluate_roi_step(RoiRule(1, None), 0, 3, "Arm A")
        self.assertTrue(state.can_advance)

    def test_exactly_one_polygon_stops_further_drawing(self):
        state = evaluate_roi_step(RoiRule(1, 1), 1, 0, "Mouse 1")
        self.assertTrue(state.can_advance)
        self.assertFalse(state.can_add_point)
        self.assertIn("undo", state.hint)

    def test_open_ended_categories_accept_more_shapes(self):
        state = evaluate_roi_step(RoiRule(1, None), 2, 0, "Arm A")
        self.assertTrue(state.can_advance and state.can_add_point)


class _Capture:
    def get(self, _property):
        return 20

    def set(self, _property, _value):
        return True

    def read(self):
        return True, np.zeros((120, 160, 3), dtype=np.uint8)

    def release(self):
        pass


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class RoiSelectorEnforcementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _dialog(self, maze):
        with patch("gui.roi_selector.cv2.VideoCapture", return_value=_Capture()):
            dialog = RoiSelectorDialog("example.mp4", maze)
        self.addCleanup(dialog.deleteLater)
        return dialog

    def _tst(self, mice=1):
        maze = create_maze("Tail Suspension Test")
        maze.configure_mice(mice)
        return maze

    @staticmethod
    def _points(dialog, *coords):
        for x, y in coords:
            dialog.editor.add_point(x, y)

    def _finish(self, dialog):
        """Press OK; returns the warning boxes that were shown."""
        with patch.object(QMessageBox, "warning") as warning:
            dialog.accept()
        return warning

    def test_ok_is_refused_until_a_tst_mouse_has_its_polygon(self):
        dialog = self._dialog(self._tst())
        self.assertFalse(dialog.editor.is_valid())
        self.assertIn("exactly 1 polygon", dialog.editor.hint_label.text())
        self.assertIn("Mouse 1", dialog.editor.hint_label.text())
        self.assertTrue(self._finish(dialog).called)       # nothing drawn
        self.assertEqual(dialog.result(), 0)

        self._points(dialog, (10, 10), (60, 10))
        self.assertIn("Add 1 more point", dialog.editor.hint_label.text())
        self.assertTrue(self._finish(dialog).called)       # unfinished shape
        self.assertEqual(dialog.result(), 0)

        self._points(dialog, (60, 60))
        self.assertTrue(dialog.editor.pending_points())
        self.assertFalse(self._finish(dialog).called)      # closed by OK
        self.assertEqual(dialog.result(), dialog.DialogCode.Accepted)
        self.assertEqual(len(dialog.get_roi_manager().rois["mouse_1"]), 1)

    def test_a_second_polygon_for_one_mouse_is_refused(self):
        dialog = self._dialog(self._tst(2))
        self._points(dialog, (10, 10), (60, 10), (60, 60))
        self.assertTrue(dialog.editor.close_shape())
        dialog.editor.set_current_zone("mouse_1")
        self.assertEqual(dialog.editor.zone_status("mouse_1").count, 1)

        dialog.editor.set_tool("draw")
        self.assertFalse(dialog.editor.add_point(5, 5))
        self.assertEqual(dialog.editor.pending_points(), [])
        self.assertIn("undo", dialog.editor.hint_label.text())

    def test_undo_removes_a_closed_polygon_so_it_can_be_redrawn(self):
        dialog = self._dialog(self._tst())
        self._points(dialog, (10, 10), (60, 10), (60, 60))
        dialog.editor.close_shape()
        self.assertTrue(dialog.editor.is_valid())

        dialog.editor.undo()
        self.assertEqual(dialog.editor.zone_status("mouse_1").count, 0)
        self.assertNotIn("mouse_1", dialog.roi_manager.rois)
        self.assertFalse(dialog.editor.is_valid())

    def test_closing_a_shape_moves_to_the_next_mouse_and_blocks_again(self):
        dialog = self._dialog(self._tst(2))
        self._points(dialog, (10, 10), (60, 10), (60, 60))
        QTest.keyClick(dialog, Qt.Key.Key_Return)   # closes the shape
        self.assertEqual(dialog.editor.current_zone(), "mouse_2")
        self.assertFalse(dialog.editor.is_valid())
        self.assertIn("Mouse 2", dialog.editor.hint_label.text())
        self.assertTrue(self._finish(dialog).called)

    def test_too_small_polygon_is_refused_with_a_hint(self):
        dialog = self._dialog(create_maze("Y-Maze"))
        self._points(dialog, (10, 10), (15, 10), (15, 15))
        self.assertFalse(dialog.editor.close_shape())
        self.assertEqual(dialog.editor.zone_status("arm_a").count, 0)
        self.assertIn("too small", dialog.editor.hint_label.text())
        self.assertGreater(MIN_ROI_AREA_PX, 12.5)

    def test_enter_closes_and_backspace_or_ctrl_z_undo(self):
        dialog = self._dialog(create_maze("Y-Maze"))
        dialog.show()
        self._points(dialog, (10, 10), (60, 10), (60, 60))

        QTest.keyClick(dialog, Qt.Key.Key_Backspace)
        self.assertEqual(len(dialog.editor.pending_points()), 2)
        self._points(dialog, (60, 60))
        QTest.keyClick(dialog, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(len(dialog.editor.pending_points()), 2)

        self._points(dialog, (60, 60))
        QTest.keyClick(dialog, Qt.Key.Key_Return)
        self.assertEqual(len(dialog.roi_manager.rois["arm_a"]), 1)
        self.assertEqual(dialog.editor.current_zone(), "arm_a")   # open-ended zone
        # Enter again: the zone is done, so on to the next missing one.
        QTest.keyClick(dialog, Qt.Key.Key_Return)
        self.assertEqual(dialog.editor.current_zone(), "arm_b")
        with patch.object(ux_helpers, "confirm_discard", return_value=True):
            dialog.close()

    def test_enter_does_nothing_while_the_zone_is_still_missing(self):
        dialog = self._dialog(create_maze("Y-Maze"))
        dialog.show()
        QTest.keyClick(dialog, Qt.Key.Key_Return)
        self.assertEqual(dialog.editor.current_zone(), "arm_a")
        self.assertEqual(dialog.result(), 0)
        dialog.close()


# ----------------------------------------------------------------------
# Dimension check when a configuration is chosen
# ----------------------------------------------------------------------
@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class DimensionCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_mismatches_are_found_and_unknown_sizes_are_skipped(self):
        sizes = {Path("a.mp4"): (160, 120), Path("b.mp4"): (640, 480), Path("c.mp4"): None}
        expected = {Path("a.mp4"): (160, 120), Path("b.mp4"): (160, 120),
                    Path("c.mp4"): (160, 120), Path("d.mp4"): None}
        found = dimension_mismatches(
            list(expected), expected.get, prober=lambda p: sizes.get(p)
        )
        self.assertEqual(
            found,
            [
                DimensionMismatch(Path("b.mp4"), (160, 120), (640, 480)),
                DimensionMismatch(Path("c.mp4"), (160, 120), None),
            ],
        )
        text = format_dimension_mismatches(found)
        self.assertIn("b.mp4: ROIs drawn for 160x120, video is 640x480", text)
        self.assertIn("c.mp4", text)
        self.assertIn("could not be read", text)
        self.assertEqual(format_dimension_mismatches([]), "")

    def test_probe_reads_real_video_size(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            video = support.write_tracking_video(Path(tmp), frames=5)
            self.assertEqual(probe_dimensions(video), (160, 120))
            self.assertIsNone(probe_dimensions(Path(tmp) / "missing.avi"))

    def test_loading_a_template_lists_videos_with_the_wrong_size(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        video = support.write_tracking_video(root, frames=5)  # 160x120

        for dims, expect_warning in (((200, 100), True), ((160, 120), False)):
            manager = ROIManager(source_dimensions=dims)
            manager.add_roi("arm_a", _square())
            template = root / f"template_{dims[0]}.json"
            BatchSettings(manager, AppConfig().detection, 0.0).save_to_file(template)

            window = MainWindow(temp_app_settings(self))
            self.addCleanup(window.deleteLater)
            self.addCleanup(window._detach_log_handler)
            window._add_video_paths([video])
            with patch.object(QMessageBox, "warning") as warning, \
                    patch.object(QMessageBox, "information") as info:
                window.load_configuration_file(template)
            if expect_warning:
                info.assert_not_called()
                text = warning.call_args.args[2]
                self.assertIn("blob.avi", text)
                self.assertIn("200x100", text)
                self.assertIn("160x120", text)
            else:
                warning.assert_not_called()
                info.assert_called_once()


# ----------------------------------------------------------------------
# One confirmation for Start
# ----------------------------------------------------------------------
@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class StartSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow(temp_app_settings(self))
        self.addCleanup(self.window.deleteLater)
        self.addCleanup(self.window._detach_log_handler)
        self.window.maze = create_maze("Y-Maze")
        manager = ROIManager(source_dimensions=(160, 120))
        for category, _ in self.window.maze.get_roi_definitions():
            manager.add_roi(category, _square())
        self.plan = {
            Path("a.mp4"): BatchSettings(manager, AppConfig().detection, 0.0)
        }

    def _validate(self, preflight, answer="start", confirm_start=True):
        with patch.object(self.window, "_run_preflight", return_value=preflight), \
                patch.object(ux_helpers, "ask_choice", return_value=answer) as ask, \
                patch.object(QMessageBox, "critical") as critical:
            result = self.window._validate_analysis_plan(self.plan, confirm_start)
        return result, ask, critical

    def test_summary_without_warnings_is_a_single_start_box(self):
        result, ask, _ = self._validate(([], []))
        self.assertTrue(result)
        ask.assert_called_once()
        title, text = ask.call_args.args[1], ask.call_args.args[2]
        self.assertEqual(title, "Start Analysis")
        self.assertIn("1 video (Y-Maze)", text)
        labels = [c.label for c in ask.call_args.args[3]]
        self.assertEqual(labels, ["Start analysis", "Cancel"])

    def test_warnings_are_listed_in_the_same_box(self):
        result, ask, _ = self._validate(([], ["Low disk space", "Odd frame rate"]))
        self.assertTrue(result)
        ask.assert_called_once()
        self.assertEqual(ask.call_args.args[1], "Ready with Warnings")
        informative = ask.call_args.kwargs["informative"]
        self.assertIn("Low disk space", informative)
        self.assertIn("Odd frame rate", informative)

    def test_cancel_stops_the_run(self):
        result, _, _ = self._validate(([], []), answer="cancel")
        self.assertFalse(result)

    def test_blocking_issues_are_reported_without_a_start_box(self):
        result, ask, critical = self._validate((["Video cannot be opened"], []))
        self.assertFalse(result)
        ask.assert_not_called()
        critical.assert_called_once()

    def test_no_second_question_after_verification_unless_there_are_warnings(self):
        result, ask, _ = self._validate(([], []), confirm_start=False)
        self.assertTrue(result)
        ask.assert_not_called()
        result, ask, _ = self._validate(([], ["Low disk space"]), confirm_start=False)
        self.assertTrue(result)
        ask.assert_called_once()

    def test_run_all_now_has_no_extra_confirmation_of_its_own(self):
        self.window.video_paths = [Path("a.mp4")]
        with patch.object(ux_helpers, "ask_choice") as ask, \
                patch.object(QMessageBox, "question") as question, \
                patch.object(self.window, "run_worker") as run_worker:
            self.window.run_batch_analysis(next(iter(self.plan.values())))
        ask.assert_not_called()
        question.assert_not_called()
        run_worker.assert_called_once()

    def test_summary_formatting(self):
        title, text, informative = ux_helpers.format_start_summary(
            3, "EPM", Path("out"), []
        )
        self.assertEqual(title, "Start Analysis")
        self.assertIn("3 videos (EPM)", text)
        self.assertIn("out", informative)
        _, _, informative = ux_helpers.format_start_summary(
            3, "EPM", None, [f"w{i}" for i in range(12)]
        )
        self.assertIn("12 warnings", informative)
        self.assertIn("... and 4 more", informative)
        self.assertIn("first video", informative)


# ----------------------------------------------------------------------
# Video 1 is reviewed once
# ----------------------------------------------------------------------
class _ScriptedVerifyDialog:
    script = []
    seen = []

    def __init__(self, video_path, settings, **_kwargs):
        self.settings = settings
        _ScriptedVerifyDialog.seen.append(Path(video_path).name)
        self.result_action = "accept"

    def setWindowTitle(self, _title):
        pass

    def exec(self):
        return True

    def get_settings(self):
        return self.settings

    def get_real_world_cm(self):
        return 0.0

    def deleteLater(self):
        pass


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class FirstVideoReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow(temp_app_settings(self))
        self.addCleanup(self.window.deleteLater)
        self.addCleanup(self.window._detach_log_handler)
        self.window.maze = create_maze("Y-Maze")
        self.videos = [Path(f"v{i}.mp4") for i in range(3)]
        self.window.video_paths = list(self.videos)
        manager = ROIManager(source_dimensions=(160, 120))
        manager.add_roi("arm_a", _square())
        self.base = BatchSettings(manager, AppConfig().detection, 0.0)
        _ScriptedVerifyDialog.seen = []
        patcher = patch("gui.main_window.SettingsDialog", _ScriptedVerifyDialog)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _build(self, **kwargs):
        with patch.object(ux_helpers, "ask_choice", return_value="run"):
            return self.window.build_analysis_plan(self.base, True, **kwargs)

    def test_the_video_reviewed_in_setup_is_not_shown_again(self):
        plan = self._build(reviewed_video=self.videos[0])
        self.assertEqual(_ScriptedVerifyDialog.seen, ["v1.mp4", "v2.mp4"])
        self.assertEqual(list(plan), self.videos)
        self.assertEqual(
            plan[self.videos[0]].to_dict(), self.base.to_dict()
        )

    def test_without_a_reviewed_video_every_video_is_shown(self):
        self._build()
        self.assertEqual(_ScriptedVerifyDialog.seen, ["v0.mp4", "v1.mp4", "v2.mp4"])

    def test_a_reviewed_video_that_is_not_first_is_still_skipped_once(self):
        plan = self._build(reviewed_video=self.videos[1])
        self.assertEqual(_ScriptedVerifyDialog.seen, ["v0.mp4", "v2.mp4"])
        self.assertEqual(list(plan), self.videos)


if __name__ == "__main__":
    unittest.main()
