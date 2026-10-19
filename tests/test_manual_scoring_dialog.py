import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    from PyQt6.QtCore import QCoreApplication, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QMessageBox

    from core.manual_scoring import (center_time_s, load_epm_totals,
                                     load_zone_intervals,
                                     zone_frame_labels)
    from core.models import AppConfig, BatchSettings
    from core.positional_validation import (compare_y_maze,
                                            load_entry_annotations)
    from core.tst_validation import load_annotations
    from gui.manual_scoring_dialog import (
        MODE_HOLD, ManualScoringDialog, ManualScoringSetupDialog,
        ScoringSetup, hotkeys_for, load_overlay_rois, parse_mouse_names,
        draw_zone_outlines)
    from roi_manager import ROIManager
    from tests.synthetic_video import FPS, write_indexed_video
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False

FRAME_COUNT = 80
CTRL = Qt.KeyboardModifier.ControlModifier if HAS_RUNTIME else None


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class ScoringDialogTestCase(unittest.TestCase):
    paradigm = "y_maze"

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.video = write_indexed_video(self.root / "video1.avi", FRAME_COUNT)
        self.out = self.root / "scores"
        self.dialogs = []

    def open_dialog(self, paradigm=None, initials="ab", **kwargs):
        setup = ScoringSetup(
            paradigm=paradigm or self.paradigm, video_path=self.video,
            scorer_initials=initials, output_dir=self.out, **kwargs)
        dialog = ManualScoringDialog(setup)
        self.dialogs.append(dialog)
        self.addCleanup(self._dispose, dialog)
        self.assertTrue(dialog.player.wait_idle())
        return dialog

    def _dispose(self, dialog):
        dialog.player.close_video()
        dialog.deleteLater()

    def go(self, dialog, frame):
        dialog.player.seek(frame)
        self.assertTrue(dialog.player.wait_idle())
        self.assertEqual(dialog.player.current_frame, frame)

    def press(self, dialog, key, modifier=Qt.KeyboardModifier.NoModifier):
        QTest.keyClick(dialog, key, modifier)
        QCoreApplication.processEvents()

    def go_press(self, dialog, frame, key):
        self.go(dialog, frame)
        self.press(dialog, key)

    def wait_until(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                return False
            QCoreApplication.processEvents()
            time.sleep(0.002)
        return True


class YMazeScoringTests(ScoringDialogTestCase):
    def test_a_b_c_log_arm_entries_and_round_trip_through_the_loader(self):
        dialog = self.open_dialog()
        self.go_press(dialog, 10, Qt.Key.Key_A)
        self.go_press(dialog, 20, Qt.Key.Key_B)
        self.go_press(dialog, 33, Qt.Key.Key_C)
        self.go_press(dialog, 40, Qt.Key.Key_0)      # center marker
        self.go_press(dialog, 50, Qt.Key.Key_1)      # alias of A
        self.assertEqual(dialog.event_list.count(), 5)
        self.assertTrue(dialog.save())
        path = self.out / "video1_manual_arm_entries.csv"
        entries = load_entry_annotations(path, "y_maze")
        self.assertEqual(
            [arm for arm, _ in entries],
            ["arm_a", "arm_b", "arm_c", "arm_a"])
        for (_, stamp), frame in zip(entries, (10, 20, 33, 50)):
            self.assertAlmostEqual(stamp, frame / FPS, places=3)
        header = path.read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(header, "entry_number,arm,timestamp_s")
        # The file is directly usable for a comparison with automated entries.
        auto = [("arm_a", 0.4), ("arm_b", 0.8), ("arm_c", 1.32),
                ("arm_a", 2.0)]
        self.assertEqual(
            compare_y_maze(auto, entries, 0.1).matching.f1, 1.0)

    def test_key_layout_is_listed_and_center_does_not_clash_with_arm_c(self):
        keys = {key: hotkey.action
                for hotkey in hotkeys_for("y_maze") for key in hotkey.keys}
        self.assertEqual(keys["A"], "arm_a")
        self.assertEqual(keys["C"], "arm_c")
        self.assertEqual(keys["0"], "center")
        dialog = self.open_dialog()
        legend = dialog.legend_label.text()
        for token in ("A / 1", "B / 2", "C / 3", "<b>0</b>"):
            self.assertIn(token, legend)

    def test_ctrl_z_undoes_and_delete_removes_the_selected_row(self):
        dialog = self.open_dialog()
        self.go_press(dialog, 10, Qt.Key.Key_A)
        self.go_press(dialog, 20, Qt.Key.Key_B)
        self.go_press(dialog, 30, Qt.Key.Key_C)
        self.press(dialog, Qt.Key.Key_Z, CTRL)
        self.assertEqual([e.label for e in dialog.session.events],
                         ["arm_a", "arm_b"])
        dialog.event_list.item(0).setSelected(True)
        self.press(dialog, Qt.Key.Key_Delete)
        self.assertEqual([e.label for e in dialog.session.events], ["arm_b"])
        self.assertEqual(dialog.event_list.count(), 1)
        self.press(dialog, Qt.Key.Key_Z, CTRL)
        self.assertEqual(dialog.event_list.count(), 2)

    def test_double_click_jumps_to_the_event_frame(self):
        dialog = self.open_dialog()
        self.go_press(dialog, 12, Qt.Key.Key_A)
        self.go_press(dialog, 44, Qt.Key.Key_B)
        self.go(dialog, 70)
        dialog.event_list.itemDoubleClicked.emit(dialog.event_list.item(0))
        self.assertTrue(dialog.player.wait_idle())
        self.assertEqual(dialog.player.current_frame, 12)

    def test_player_keys_pass_through(self):
        dialog = self.open_dialog()
        self.go(dialog, 30)
        self.press(dialog, Qt.Key.Key_Right)
        self.assertTrue(dialog.player.wait_idle())
        self.assertEqual(dialog.player.current_frame, 31)
        self.assertEqual(dialog.session.events, [])

    def test_autosave_resume_and_provenance(self):
        dialog = self.open_dialog()
        self.go_press(dialog, 10, Qt.Key.Key_A)
        self.go_press(dialog, 20, Qt.Key.Key_B)
        sidecar = dialog.sidecar_path
        self.assertTrue(sidecar.is_file())
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        self.assertEqual(data["scorer_initials"], "AB")
        self.assertRegex(data["date"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertEqual(len(data["events"]), 2)
        self.assertTrue(data["blind"])
        # The dialog was never saved, so closing asks; discard it.
        with patch.object(ManualScoringDialog, "_ask_close_action",
                          return_value="discard") as ask:
            dialog.reject()
        ask.assert_called_once()
        self.assertFalse(dialog.player.capture_open)
        self.assertFalse((self.out / "video1_manual_arm_entries.csv").exists())

        again = self.open_dialog()
        self.assertEqual([(e.label, e.frame) for e in again.session.events],
                         [("arm_a", 10), ("arm_b", 20)])
        self.assertIn("Resumed", again.status_label.text())
        self.assertEqual(again.event_list.count(), 2)

        other = self.open_dialog(initials="zz")
        self.assertEqual(other.session.events, [])

    def test_save_asks_before_replacing_files_it_did_not_write(self):
        dialog = self.open_dialog()
        self.go_press(dialog, 10, Qt.Key.Key_A)
        self.out.mkdir(parents=True, exist_ok=True)
        target = self.out / "video1_manual_arm_entries.csv"
        target.write_text("entry_number,arm,timestamp_s\n1,arm_b,9.0\n",
                          encoding="utf-8")
        with patch.object(ManualScoringDialog, "_confirm_overwrite",
                          return_value=False) as ask:
            self.assertFalse(dialog.save())
        ask.assert_called_once()
        self.assertIn("arm_b", target.read_text(encoding="utf-8"))
        with patch.object(ManualScoringDialog, "_confirm_overwrite",
                          return_value=True):
            self.assertTrue(dialog.save())
        self.assertIn("arm_a", target.read_text(encoding="utf-8"))
        # Its own earlier file is replaced without asking.
        with patch.object(ManualScoringDialog, "_confirm_overwrite") as ask:
            self.assertTrue(dialog.save())
        ask.assert_not_called()

    def test_closing_with_unsaved_scores_can_cancel_or_save(self):
        dialog = self.open_dialog()
        self.go_press(dialog, 10, Qt.Key.Key_A)
        with patch.object(ManualScoringDialog, "_ask_close_action",
                          return_value="cancel"):
            dialog.reject()
        self.assertTrue(dialog.player.capture_open)
        self.assertFalse((self.out / "video1_manual_arm_entries.csv").exists())
        with patch.object(ManualScoringDialog, "_ask_close_action",
                          return_value="save"):
            dialog.reject()
        self.assertFalse(dialog.player.capture_open)
        self.assertTrue((self.out / "video1_manual_arm_entries.csv").exists())

    def test_saved_session_closes_without_asking(self):
        dialog = self.open_dialog()
        self.go_press(dialog, 10, Qt.Key.Key_A)
        self.assertTrue(dialog.save())
        with patch.object(ManualScoringDialog, "_ask_close_action") as ask:
            dialog.reject()
        ask.assert_not_called()
        self.assertFalse(dialog.player.capture_open)

    def test_clear_all_asks_and_can_be_undone(self):
        dialog = self.open_dialog()
        self.go_press(dialog, 10, Qt.Key.Key_A)
        with patch("gui.manual_scoring_dialog.QMessageBox.question",
                   return_value=QMessageBox.StandardButton.No):
            dialog.clear_all()
        self.assertEqual(len(dialog.session.events), 1)
        with patch("gui.manual_scoring_dialog.QMessageBox.question",
                   return_value=QMessageBox.StandardButton.Yes):
            dialog.clear_all()
        self.assertEqual(dialog.session.events, [])
        dialog.undo()
        self.assertEqual(len(dialog.session.events), 1)

    def test_done_releases_the_video_and_saved_signal_lists_files(self):
        dialog = self.open_dialog()
        saved = []
        dialog.files_saved.connect(saved.extend)
        self.go_press(dialog, 10, Qt.Key.Key_A)
        self.assertTrue(dialog.player.capture_open)
        dialog.save_and_close()
        self.assertFalse(dialog.player.capture_open)
        self.assertEqual([Path(p).name for p in saved],
                         ["video1_manual_arm_entries.csv"])

    def test_unreadable_video_raises(self):
        setup = ScoringSetup("y_maze", self.root / "missing.avi", "ab", self.out)
        with self.assertRaises(IOError):
            ManualScoringDialog(setup)


class EpmScoringTests(ScoringDialogTestCase):
    paradigm = "epm"

    def test_five_keys_and_round_trip(self):
        dialog = self.open_dialog()
        self.go_press(dialog, 5, Qt.Key.Key_1)
        self.go_press(dialog, 15, Qt.Key.Key_0)
        self.go_press(dialog, 25, Qt.Key.Key_3)
        self.go_press(dialog, 35, Qt.Key.Key_0)
        self.go_press(dialog, 45, Qt.Key.Key_2)
        self.go_press(dialog, 55, Qt.Key.Key_0)
        self.go_press(dialog, 65, Qt.Key.Key_4)
        self.assertTrue(dialog.save())
        entries = load_entry_annotations(
            self.out / "video1_manual_arm_entries.csv", "epm")
        self.assertEqual([arm for arm, _ in entries],
                         ["open_arm_1", "closed_arm_1", "open_arm_2",
                          "closed_arm_2"])
        totals = load_epm_totals(self.out / "video1_manual_epm_totals.csv")
        self.assertAlmostEqual(totals["time_in_open_arms_s"], 20 / FPS, 3)
        self.assertAlmostEqual(
            totals["time_in_closed_arms_s"], (10 + 15) / FPS, 3)
        self.assertAlmostEqual(
            totals["percent_time_in_open"], 20 / 45 * 100, 2)

    def test_window_end_limits_the_derived_time(self):
        dialog = self.open_dialog()
        self.go_press(dialog, 0, Qt.Key.Key_0)
        self.go_press(dialog, 10, Qt.Key.Key_1)
        dialog.window_end_spin.setValue(1.0)          # 25 frames
        dialog.window_end_spin.editingFinished.emit()
        self.assertEqual(dialog.session.window_end_s, 1.0)
        dialog.save()
        totals = load_epm_totals(self.out / "video1_manual_epm_totals.csv")
        self.assertAlmostEqual(totals["time_in_open_arms_s"], 15 / FPS, 3)


class OpenFieldScoringTests(ScoringDialogTestCase):
    paradigm = "open_field"

    def test_toggle_mode_writes_intervals(self):
        dialog = self.open_dialog()
        self.go_press(dialog, 10, Qt.Key.Key_C)
        self.go_press(dialog, 30, Qt.Key.Key_C)
        self.go_press(dialog, 50, Qt.Key.Key_C)
        self.assertIn("center", dialog.state_label.text())
        self.assertTrue(dialog.save())
        intervals = load_zone_intervals(
            self.out / "video1_manual_zone_intervals.csv")
        self.assertAlmostEqual(center_time_s(intervals), 50 / FPS, places=3)
        labels = zone_frame_labels(intervals)
        self.assertEqual(len(labels), FRAME_COUNT)
        self.assertEqual(labels[9], "periphery")
        self.assertEqual(labels[29], "center")
        self.assertEqual(labels[79], "center")

    def test_hold_mode_marks_center_while_the_key_is_down(self):
        dialog = self.open_dialog()
        dialog.set_zone_mode(MODE_HOLD)
        self.go(dialog, 10)
        QTest.keyPress(dialog, Qt.Key.Key_C)
        self.go(dialog, 30)
        QTest.keyRelease(dialog, Qt.Key.Key_C)
        self.assertEqual([(e.label, e.frame) for e in dialog.session.events],
                         [("center", 10), ("periphery", 30)])
        self.go_press(dialog, 50, Qt.Key.Key_C)       # a tap: center to 50
        QTest.keyRelease(dialog, Qt.Key.Key_C)
        self.assertEqual(dialog.session.zone_now(50), "periphery")

    def test_initial_zone_and_window_are_honoured(self):
        dialog = self.open_dialog()
        dialog.initial_zone_combo.setCurrentIndex(1)           # center
        self.go_press(dialog, 20, Qt.Key.Key_C)                 # -> periphery
        dialog.window_start_spin.setValue(0.4)                  # frame 10
        dialog.window_start_spin.editingFinished.emit()
        dialog.save()
        intervals = load_zone_intervals(
            self.out / "video1_manual_zone_intervals.csv")
        self.assertEqual(intervals[0].zone, "center")
        self.assertEqual(intervals[0].start_frame, 10)
        self.assertEqual(intervals[0].end_frame, 19)

    def test_resume_keeps_zone_changes(self):
        dialog = self.open_dialog()
        self.go_press(dialog, 10, Qt.Key.Key_C)
        dialog.done(0)
        again = self.open_dialog()
        self.assertEqual([(e.label, e.frame) for e in again.session.events],
                         [("center", 10)])


class TstScoringTests(ScoringDialogTestCase):
    paradigm = "tst"

    def test_held_key_labels_every_frame_including_skipped_ones(self):
        dialog = self.open_dialog()
        self.go(dialog, 5)
        QTest.keyPress(dialog, Qt.Key.Key_I)
        self.assertIn("immobile_static", dialog.state_label.text())
        for _ in range(4):
            dialog.player.step(1)
            self.assertTrue(dialog.player.wait_idle())
        self.go(dialog, 25)                      # a 16-frame jump while held
        QTest.keyRelease(dialog, Qt.Key.Key_I)
        [bout] = dialog.session.bouts
        self.assertEqual((bout.state, bout.start, bout.end),
                         ("immobile_static", 5, 25))
        self.assertTrue(dialog.save())
        annotations = load_annotations(
            self.out / "video1_manual_tst_annotations.csv")
        self.assertEqual(
            sorted(frame for (_, frame) in annotations), list(range(5, 26)))
        self.assertTrue(all(state == "immobile_static"
                            for state in annotations.values()))
        self.assertEqual({mouse for mouse, _ in annotations}, {"mouse_1"})

    def test_playing_while_holding_fills_a_contiguous_bout(self):
        dialog = self.open_dialog()
        self.go(dialog, 30)
        QTest.keyPress(dialog, Qt.Key.Key_P)
        dialog.player.set_speed(2.0)
        dialog.player.play()
        self.assertTrue(self.wait_until(lambda: not dialog.player.is_playing))
        self.assertTrue(dialog.player.wait_idle())
        QTest.keyRelease(dialog, Qt.Key.Key_P)
        [bout] = dialog.session.bouts
        self.assertEqual(bout.state, "passive_swing")
        self.assertEqual(bout.start, 30)
        self.assertEqual(bout.end, FRAME_COUNT - 1)

    def test_switching_keys_ends_one_bout_and_starts_the_next(self):
        dialog = self.open_dialog()
        self.go(dialog, 0)
        QTest.keyPress(dialog, Qt.Key.Key_A)
        self.go(dialog, 9)
        QTest.keyRelease(dialog, Qt.Key.Key_A)
        self.go(dialog, 12)
        QTest.keyPress(dialog, Qt.Key.Key_U)
        self.go(dialog, 20)
        QTest.keyRelease(dialog, Qt.Key.Key_U)
        self.assertEqual(
            [(b.state, b.start, b.end) for b in dialog.session.bouts],
            [("active_escape", 0, 9), ("uncertain", 12, 20)])
        self.assertEqual(dialog.event_list.count(), 2)

    def test_mouse_selection_and_fill_option(self):
        dialog = self.open_dialog(mice=("mouse_1", "mouse_2"))
        self.assertEqual(dialog.mouse_combo.count(), 2)
        self.go(dialog, 2)
        QTest.keyPress(dialog, Qt.Key.Key_I)
        self.go(dialog, 4)
        QTest.keyRelease(dialog, Qt.Key.Key_I)
        dialog.mouse_combo.setCurrentIndex(1)
        self.assertEqual(dialog.event_list.count(), 0)   # other mouse's list
        self.go(dialog, 10)
        QTest.keyPress(dialog, Qt.Key.Key_A)
        self.go(dialog, 11)
        QTest.keyRelease(dialog, Qt.Key.Key_A)
        dialog.fill_combo.setCurrentIndex(1)             # fill with active
        dialog.save()
        annotations = load_annotations(
            self.out / "video1_manual_tst_annotations.csv")
        self.assertEqual(annotations[("mouse_1", 3)], "immobile_static")
        self.assertEqual(annotations[("mouse_1", 40)], "active_escape")
        self.assertEqual(annotations[("mouse_2", 10)], "active_escape")
        self.assertEqual(len(annotations), 2 * FRAME_COUNT)

    def test_undo_removes_the_last_bout(self):
        dialog = self.open_dialog()
        self.go(dialog, 3)
        QTest.keyPress(dialog, Qt.Key.Key_I)
        self.go(dialog, 6)
        QTest.keyRelease(dialog, Qt.Key.Key_I)
        self.press(dialog, Qt.Key.Key_Z, CTRL)
        self.assertEqual(dialog.session.bouts, [])
        self.assertEqual(dialog.event_list.count(), 0)

    def test_delete_removes_a_bout_and_jump_goes_to_its_start(self):
        dialog = self.open_dialog()
        for start, end, key in ((3, 6, Qt.Key.Key_I), (20, 30, Qt.Key.Key_U)):
            self.go(dialog, start)
            QTest.keyPress(dialog, key)
            self.go(dialog, end)
            QTest.keyRelease(dialog, key)
        self.go(dialog, 70)
        dialog.event_list.itemDoubleClicked.emit(dialog.event_list.item(1))
        self.assertTrue(dialog.player.wait_idle())
        self.assertEqual(dialog.player.current_frame, 20)
        dialog.event_list.item(0).setSelected(True)
        dialog.delete_selected()
        self.assertEqual([b.state for b in dialog.session.bouts], ["uncertain"])

    def test_losing_focus_ends_the_held_bout(self):
        dialog = self.open_dialog()
        self.go(dialog, 3)
        QTest.keyPress(dialog, Qt.Key.Key_I)
        self.assertIsNotNone(dialog.session.open_bout)
        dialog.release_hold()
        self.assertIsNone(dialog.session.open_bout)
        self.assertEqual(len(dialog.session.bouts), 1)

    def test_closing_while_holding_keeps_the_bout(self):
        dialog = self.open_dialog()
        self.go(dialog, 3)
        QTest.keyPress(dialog, Qt.Key.Key_I)
        self.go(dialog, 8)
        dialog.done(0)
        data = json.loads(dialog.sidecar_path.read_text(encoding="utf-8"))
        self.assertEqual(data["bouts"][0]["end"], 8)


class OverlayAndSetupTests(ScoringDialogTestCase):
    def _template(self, dimensions=(160, 120)):
        manager = ROIManager(dimensions)
        manager.add_roi("arm_a", np.array([[10, 10], [80, 10], [80, 60], [10, 60]]))
        manager.add_roi("center", np.array([[90, 70], [140, 70], [140, 110]]))
        path = self.root / "template.json"
        BatchSettings(manager, AppConfig().detection, 0.0).save_to_file(path)
        return path

    def test_template_rois_are_drawn_over_the_frame(self):
        path = self._template()
        manager, note = load_overlay_rois(path, self.video)
        self.assertEqual(note, "")
        self.assertEqual(set(manager.rois), {"arm_a", "center"})
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        drawn = draw_zone_outlines(frame, manager)
        self.assertGreater(int(drawn.sum()), 0)
        self.assertEqual(int(frame.sum()), 0)

        dialog = self.open_dialog(roi_file=path)
        self.assertTrue(dialog.zones_check.isEnabled())
        self.assertFalse(dialog.zones_check.isHidden())
        shown = dialog._draw_overlay(frame.copy(), 0)
        self.assertGreater(int(shown.sum()), 0)
        dialog.zones_check.setChecked(False)
        self.assertEqual(int(dialog._draw_overlay(frame.copy(), 0).sum()), 0)

    def test_dialog_without_rois_never_draws_anything(self):
        dialog = self.open_dialog()
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        self.assertEqual(int(dialog._draw_overlay(frame, 0).sum()), 0)
        self.assertTrue(dialog.session.to_dict()["blind"])
        self.assertIn("Blind", dialog.blind_label.text())

    def test_mismatched_resolution_draws_no_zones(self):
        path = self._template(dimensions=(320, 240))
        dialog = self.open_dialog(roi_file=path)
        self.assertIsNone(dialog._roi_manager)
        self.assertIn("320x240", dialog.roi_note_label.text())

    def test_batch_plan_entry_is_selected_by_video(self):
        plan_file = self.root / "plan.json"
        first = ROIManager((160, 120))
        first.add_roi("arm_a", np.array([[10, 10], [80, 10], [80, 60]]))
        second = ROIManager((160, 120))
        second.add_roi("arm_b", np.array([[10, 10], [80, 10], [80, 60]]))
        BatchSettings.save_batch_plan({
            "video1.avi": BatchSettings(first, AppConfig().detection, 0.0),
            "other.avi": BatchSettings(second, AppConfig().detection, 0.0),
        }, plan_file)
        manager, note = load_overlay_rois(plan_file, self.video)
        self.assertEqual(note, "")
        self.assertEqual(set(manager.rois), {"arm_a"})
        manager, note = load_overlay_rois(plan_file, self.root / "nope.avi")
        self.assertIsNone(manager)
        self.assertIn("no unambiguous entry", note)

    def test_unreadable_roi_file_is_reported(self):
        bad = self.root / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        manager, note = load_overlay_rois(bad, self.video)
        self.assertIsNone(manager)
        self.assertTrue(note)


class SetupDialogTests(ScoringDialogTestCase):
    def test_validation_and_defaults(self):
        dialog = ManualScoringSetupDialog()
        self.addCleanup(dialog.deleteLater)
        self.assertIn("video", dialog.validation_problem().lower())
        dialog.video_edit.setText(str(self.video))
        self.assertIn("initials", dialog.validation_problem())
        dialog.initials_edit.setText("a b")
        self.assertEqual(dialog.validation_problem(), "")
        setup = dialog.setup()
        self.assertEqual(setup.scorer_initials, "AB")
        self.assertEqual(setup.video_path, self.video)
        self.assertEqual(setup.output_dir, self.video.parent)
        self.assertEqual(setup.paradigm, "y_maze")
        self.assertIsNone(setup.roi_file)

    def test_paradigm_choice_and_mice(self):
        dialog = ManualScoringSetupDialog()
        self.addCleanup(dialog.deleteLater)
        dialog.set_paradigm("tst")
        self.assertFalse(dialog.mice_edit.isHidden() and False)
        dialog.mice_edit.setText("mouse_1, mouse_2;mouse_1 , left mouse")
        dialog.video_edit.setText(str(self.video))
        dialog.initials_edit.setText("zz")
        self.assertEqual(dialog.setup().mice,
                         ("mouse_1", "mouse_2", "left_mouse"))
        self.assertEqual(parse_mouse_names(""), ("mouse_1",))

    def test_output_follows_video_until_edited(self):
        dialog = ManualScoringSetupDialog()
        self.addCleanup(dialog.deleteLater)
        dialog.video_edit.setText(str(self.video))
        self.assertEqual(dialog.output_edit.text(), str(self.root))
        dialog.output_edit.textEdited.emit("x")
        dialog.output_edit.setText(str(self.out))
        dialog.video_edit.setText(str(self.root / "other" / "v.avi"))
        self.assertEqual(dialog.output_edit.text(), str(self.out))

    def test_tst_template_supplies_mouse_names(self):
        manager = ROIManager((160, 120))
        for name in ("mouse_2", "mouse_1"):
            manager.add_roi(name, np.array([[10, 10], [80, 10], [80, 60]]))
        path = self.root / "tst.json"
        manager_dict = {"type": "single_template", "data": {
            "roi_manager": manager.to_dict()}}
        path.write_text(json.dumps(manager_dict), encoding="utf-8")
        dialog = ManualScoringSetupDialog()
        self.addCleanup(dialog.deleteLater)
        dialog.set_paradigm("tst")
        dialog.video_edit.setText(str(self.video))
        dialog.roi_edit.setText(str(path))
        self.assertEqual(dialog.mice_edit.text(), "mouse_1, mouse_2")


if __name__ == "__main__":
    unittest.main()
