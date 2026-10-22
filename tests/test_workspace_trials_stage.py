"""The Trials stage: per-video start and end times, the protocol card, the
player with "Set start here" / "Set end here", and the keyboard flow."""
import unittest
from pathlib import Path

import numpy as np

from tests.synthetic_video import FPS, write_indexed_video
from tests.test_workspace_stage_status import settings_for
from tests.workspace_support import HAS_RUNTIME, WorkspaceFixture

if HAS_RUNTIME:
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QLabel, QRadioButton
    from core.models import AppConfig, BatchSettings
    from core.project import (START_PER_VIDEO, START_VIDEO, TRIAL_INVALID,
                              TRIAL_NEEDS_START, TRIAL_OK)
    from gui.workspace.stages import trials_widgets as tw
    from gui.workspace.stages.trials_stage import PAGE_EMPTY, PAGE_MAIN
    from gui.workspace.video_panel import ChipRole
    from mazes import create_maze
    from roi_manager import ROIManager

TST = "Tail Suspension Test"


def tst_settings(mice: int = 3) -> "BatchSettings":
    """Tail suspension settings with one zone per mouse."""
    maze = create_maze(TST)
    maze.configure_mice(mice)
    manager = ROIManager(source_dimensions=(640, 480))
    for index, (name, _label) in enumerate(maze.get_roi_definitions()):
        x = index * 100
        manager.add_roi(name, np.array([[x, 0], [x + 40, 0], [x + 40, 40], [x, 40]]))
    return BatchSettings(
        manager, AppConfig().detection, 0.0, paradigm_id=TST,
        algorithm_parameters=maze.get_algorithm_parameters(),
    )


# ----------------------------------------------------------------------
# Qt-free: the time text
# ----------------------------------------------------------------------
@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class TimeTextTests(unittest.TestCase):
    def test_seconds_and_minutes_are_understood(self):
        for text, seconds in (
            ("90", 90.0), ("75.5", 75.5), (" 12 ", 12.0), ("0", 0.0),
            ("1:15", 75.0), ("1:15.5", 75.5), ("01:05", 65.0), ("0:00.5", 0.5),
            ("12:00", 720.0), ("1:02:03", 3723.0), ("90,5", 90.5), (".5", 0.5),
        ):
            with self.subTest(text):
                self.assertAlmostEqual(tw.parse_time(text), seconds)

    def test_text_that_is_not_a_time_is_refused_with_a_reason(self):
        for text in ("", "  ", "abc", "1:2:3:4", "1::30", ":30", "1:75", "-5",
                     "1:60", "1.5:30", "2:61:00", "1e3", "5s", "nan"):
            with self.subTest(text):
                with self.assertRaises(ValueError) as caught:
                    tw.parse_time(text)
                self.assertTrue(str(caught.exception))

    def test_times_are_written_as_minutes_seconds_tenths(self):
        self.assertEqual(tw.format_time(0), "0:00.0")
        self.assertEqual(tw.format_time(75.5), "1:15.5")
        self.assertEqual(tw.format_time(300), "5:00.0")
        self.assertEqual(tw.format_time(3725.04), "62:05.0")
        self.assertEqual(tw.format_time(59.96), "1:00.0")
        self.assertEqual(tw.format_time(90, 0), "1:30")

    def test_exact_text_round_trips(self):
        for seconds in (0.0, 12.3, 75.533, 300.0, 4.04):
            self.assertAlmostEqual(
                tw.parse_time(tw.format_time_exact(seconds)), seconds, places=3)
        self.assertEqual(tw.format_time_exact(75.5), "1:15.5")
        self.assertEqual(tw.format_time_exact(300), "5:00.0")

    def test_times_from_the_player_are_rounded_to_milliseconds(self):
        self.assertEqual(tw.round_time(10 / 30), 0.333)

    def test_ticks_leave_room_for_their_labels(self):
        self.assertEqual(tw.tick_step(300, 700), 30.0)
        self.assertEqual(tw.tick_step(15, 700), 2.0)
        self.assertEqual(tw.tick_step(0, 700), 60.0)


# ----------------------------------------------------------------------
# The stage
# ----------------------------------------------------------------------
class TrialsFixture(WorkspaceFixture):
    """A saved Y-maze project of three (fake) videos, on the Trials stage."""

    paradigm = "Y-Maze"
    template = True

    def setUp(self):
        super().setUp()
        self.new_project_at()
        self.window.change_paradigm(self.paradigm)
        self.project = self.window.project
        if self.template:
            self.project.set_template(self.make_template())
        self.add_videos(self.make_videos())
        self.stage = self.window.stage("trials")
        self.addCleanup(self.stage.shutdown)
        self.keys = self.project.keys()
        self.prompts.queues.clear()

    def make_template(self):
        return settings_for(self.paradigm)

    def make_videos(self):
        return self.videos

    def go(self):
        self.assertTrue(self.window.go_to_stage("trials"))
        return self.stage

    def cell(self, key, column, subject=None):
        model = self.stage.model
        row = model.row_of(key, subject)
        self.assertGreaterEqual(row, 0, (key, subject))
        return model.index(row, column)

    def text(self, key, column, subject=None):
        return self.stage.model.data(self.cell(key, column, subject))

    def edit(self, key, column, text, subject=None):
        """Type ``text`` into a cell of the timing table."""
        return self.stage.model.setData(self.cell(key, column, subject), text)

    def key_click(self, widget, key, modifier=Qt.KeyboardModifier.NoModifier):
        QTest.keyClick(widget, key, modifier)
        self.app.processEvents()

    def trial(self, index=0):
        return self.project.entry(self.keys[index]).trial


class RealVideoFixture(TrialsFixture):
    """The same, with real (tiny) videos so the player has frames."""

    def make_videos(self):
        folder = self.root / "real"
        folder.mkdir()
        return [
            write_indexed_video(folder / f"mouse_{n}.avi", frame_count=80).resolve()
            for n in (1, 2, 3)
        ]

    def show(self, index=0, subject=None):
        self.go()
        self.assertTrue(self.stage.select_row(self.keys[index], subject))
        self.assertTrue(self.stage.player.wait_idle())
        return self.stage.player

    def seek(self, frame):
        player = self.stage.player
        player.seek(frame)
        self.assertTrue(player.wait_idle())
        self.assertEqual(player.current_frame, frame)


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class TableTests(TrialsFixture):
    def test_it_is_the_trials_stage(self):
        self.assertEqual(type(self.stage).__name__, "TrialsStage")
        self.assertEqual(self.stage.stage_id, "trials")
        self.assertEqual(self.stage.title, "Trials")

    def test_the_start_column_has_room_for_the_call_to_action(self):
        self.project.set_trial_protocol(300.0, START_PER_VIDEO)
        self.go()
        table = self.stage.table
        key = self.keys[0]
        self.assertEqual(self.text(key, tw.COL_START), "Set start…")
        # Wide enough for the text at the theme's 13 px (the offscreen test
        # font is much wider, so the exact width is checked by eye in the
        # gallery); the video name column keeps its room.
        self.assertGreaterEqual(table.columnWidth(tw.COL_START), 88)
        self.assertGreater(table.columnWidth(tw.COL_VIDEO), 80)

    def test_one_row_per_video_with_the_six_columns(self):
        self.go()
        model = self.stage.model
        self.assertEqual(model.rowCount(), 3)
        self.assertEqual(
            [model.headerData(c, Qt.Orientation.Horizontal) for c in range(6)],
            ["Status", "Video", "Start", "End", "Duration", "Source"],
        )
        self.assertEqual(
            [self.text(k, tw.COL_VIDEO) for k in self.keys],
            ["m1.mp4", "m2.mp4", "m3.mp4"],
        )

    def test_a_video_that_follows_the_protocol_shows_its_end_greyed_and_tagged(self):
        self.project.set_trial_protocol(300.0, START_VIDEO)
        self.go()
        key = self.keys[0]
        self.assertEqual(self.text(key, tw.COL_START), "0:00.0")
        self.assertEqual(self.text(key, tw.COL_END), "5:00.0  protocol")
        self.assertEqual(self.text(key, tw.COL_DURATION), "5:00.0")
        self.assertEqual(self.text(key, tw.COL_SOURCE), "protocol")
        self.assertEqual(self.text(key, tw.COL_STATUS), "OK")
        end = self.cell(key, tw.COL_END)
        self.assertTrue(self.stage.model.data(end, Qt.ItemDataRole.FontRole).italic())
        self.assertIsNotNone(self.stage.model.data(end, Qt.ItemDataRole.ForegroundRole))
        self.assertIn("protocol", self.stage.model.data(end, Qt.ItemDataRole.ToolTipRole))

    def test_an_end_set_by_the_user_is_plain_and_manual(self):
        self.project.set_trial_protocol(300.0, START_VIDEO)
        self.project.set_trial(self.keys[0], start=10, end=250)
        self.go()
        key = self.keys[0]
        self.assertEqual(self.text(key, tw.COL_START), "0:10.0")
        self.assertEqual(self.text(key, tw.COL_END), "4:10.0")
        self.assertEqual(self.text(key, tw.COL_DURATION), "4:00.0")
        self.assertEqual(self.text(key, tw.COL_SOURCE), "manual")
        self.assertIsNone(
            self.stage.model.data(self.cell(key, tw.COL_END), Qt.ItemDataRole.FontRole))

    def test_without_a_duration_the_trial_runs_until_the_video_ends(self):
        self.go()
        self.assertEqual(self.text(self.keys[0], tw.COL_END), "video end")
        self.assertEqual(self.text(self.keys[0], tw.COL_DURATION), "—")

    def test_a_video_that_needs_its_start_says_so(self):
        self.project.set_trial_protocol(300.0, START_PER_VIDEO)
        self.project.set_trial(self.keys[1], start=7.5)
        self.go()
        chips = [self.stage.model.data(self.cell(k, tw.COL_STATUS), ChipRole)
                 for k in self.keys]
        self.assertEqual(chips, [("warning", "Needs start"), ("ok", "OK"),
                                 ("warning", "Needs start")])
        self.assertEqual(self.text(self.keys[0], tw.COL_START), "Set start…")
        self.assertEqual(self.text(self.keys[0], tw.COL_END), "start + 5:00")
        self.assertEqual(self.text(self.keys[1], tw.COL_START), "0:07.5")
        self.assertEqual(self.text(self.keys[1], tw.COL_END), "5:07.5  protocol")

    def test_an_invalid_trial_is_red_with_the_reason_in_the_tooltip(self):
        self.project.set_trial(self.keys[0], start=50, end=20)
        self.go()
        key = self.keys[0]
        self.assertEqual(self.stage.model.data(self.cell(key, tw.COL_STATUS), ChipRole),
                         ("error", "Invalid"))
        end = self.cell(key, tw.COL_END)
        self.assertIn("after the start",
                      self.stage.model.data(end, Qt.ItemDataRole.ToolTipRole))
        self.assertIsNotNone(self.stage.model.data(end, Qt.ItemDataRole.BackgroundRole))
        # The start itself is fine.
        self.assertIsNone(self.stage.model.data(
            self.cell(key, tw.COL_START), Qt.ItemDataRole.BackgroundRole))

    def test_the_table_follows_changes_made_elsewhere(self):
        self.go()
        self.project.set_trial(self.keys[2], start=33)
        self.assertEqual(self.text(self.keys[2], tw.COL_START), "0:33.0")
        self.project.clear_trial(self.keys[2])
        self.assertEqual(self.text(self.keys[2], tw.COL_START), "0:00.0")

    def test_video_rows_are_selectable_and_editable_headers_are_not(self):
        self.go()
        flags = self.stage.model.flags(self.cell(self.keys[0], tw.COL_START))
        self.assertTrue(flags & Qt.ItemFlag.ItemIsEditable)
        self.assertTrue(flags & Qt.ItemFlag.ItemIsSelectable)
        for column in (tw.COL_STATUS, tw.COL_VIDEO, tw.COL_SOURCE):
            self.assertFalse(
                self.stage.model.flags(self.cell(self.keys[0], column))
                & Qt.ItemFlag.ItemIsEditable)


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class EditingTests(TrialsFixture):
    def test_seconds_and_minutes_both_write_the_project(self):
        self.go()
        key = self.keys[0]
        for text, seconds in (("90", 90.0), ("1:15.5", 75.5), ("2:00", 120.0)):
            with self.subTest(text):
                self.assertTrue(self.edit(key, tw.COL_START, text))
                self.assertEqual(self.trial().start_time, seconds)
                self.assertTrue(self.trial().start_set)
        self.assertTrue(self.edit(key, tw.COL_END, "3:30"))
        self.assertEqual(self.trial().end_time, 210.0)

    def test_edits_reach_the_project_at_once_and_mark_it_modified(self):
        self.go()
        self.window._unsaved = False
        self.edit(self.keys[1], tw.COL_START, "12")
        self.assertTrue(self.window.has_unsaved_changes)
        self.assertEqual(self.project.entry(self.keys[1]).trial.start_time, 12.0)
        self.assertEqual(self.text(self.keys[1], tw.COL_START), "0:12.0")

    def test_text_that_is_not_a_time_is_kept_in_the_cell_and_not_written(self):
        self.go()
        key = self.keys[0]
        self.assertTrue(self.edit(key, tw.COL_START, "abc"))
        self.assertIsNone(self.trial())
        self.assertEqual(self.text(key, tw.COL_START), "abc")
        cell = self.cell(key, tw.COL_START)
        self.assertIn("not a time", self.stage.model.data(cell, Qt.ItemDataRole.ToolTipRole))
        self.assertIsNotNone(self.stage.model.data(cell, Qt.ItemDataRole.ForegroundRole))
        self.assertIsNotNone(self.stage.model.data(cell, Qt.ItemDataRole.BackgroundRole))
        self.assertEqual(list(self.stage.model.bad_cells().values())[0][0], "abc")
        # It survives a refresh, and a good value replaces it.
        self.stage.refresh()
        self.assertEqual(self.text(key, tw.COL_START), "abc")
        self.edit(key, tw.COL_START, "45")
        self.assertEqual(self.text(key, tw.COL_START), "0:45.0")
        self.assertEqual(self.stage.model.bad_cells(), {})

    def test_negative_times_are_refused(self):
        self.go()
        self.edit(self.keys[0], tw.COL_END, "-4")
        self.assertIsNone(self.trial())
        self.assertIn("negative", list(self.stage.model.bad_cells().values())[0][1])

    def test_an_unchanged_cell_writes_nothing(self):
        self.project.set_trial_protocol(300.0, START_VIDEO)
        self.go()
        seen = []
        self.window.project_modified.connect(lambda: seen.append(1))
        # Accepting the protocol's end as it is must not turn it into a manual end.
        self.assertFalse(self.edit(self.keys[0], tw.COL_END, "5:00.0"))
        self.assertFalse(self.edit(self.keys[0], tw.COL_START, "0:00.0"))
        self.assertEqual(seen, [])
        self.assertIsNone(self.trial())

    def test_an_end_before_the_start_is_stored_and_flagged(self):
        self.go()
        key = self.keys[0]
        self.edit(key, tw.COL_START, "60")
        self.edit(key, tw.COL_END, "30")
        self.assertEqual(self.project.trial_status(key), TRIAL_INVALID)
        self.assertEqual(self.text(key, tw.COL_STATUS), "Invalid")
        self.assertEqual(self.trial().end_time, 30.0)
        self.edit(key, tw.COL_END, "90")
        self.assertEqual(self.project.trial_status(key), TRIAL_OK)

    def test_clearing_a_cell_goes_back_to_the_protocol(self):
        self.project.set_trial_protocol(300.0, START_VIDEO)
        self.project.set_trial(self.keys[0], start=20, end=100)
        self.go()
        key = self.keys[0]
        self.assertTrue(self.edit(key, tw.COL_END, ""))
        self.assertIsNone(self.trial().end_time)
        self.assertEqual(self.text(key, tw.COL_END), "5:20.0  protocol")
        self.assertTrue(self.edit(key, tw.COL_START, ""))
        self.assertIsNone(self.trial())
        self.assertEqual(self.text(key, tw.COL_START), "0:00.0")

    def test_the_delete_key_clears_the_current_cell(self):
        self.project.set_trial(self.keys[0], start=20, end=100)
        self.go()
        self.stage.select_row(self.keys[0])
        self.stage.table.setCurrentIndex(self.cell(self.keys[0], tw.COL_END))
        self.key_click(self.stage.table, Qt.Key.Key_Delete)
        self.assertIsNone(self.trial().end_time)
        self.assertEqual(self.trial().start_time, 20)

    def test_typing_a_duration_sets_the_end(self):
        self.go()
        key = self.keys[0]
        self.edit(key, tw.COL_START, "30")
        self.assertTrue(self.edit(key, tw.COL_DURATION, "5:00"))
        self.assertEqual(self.trial().end_time, 330.0)
        self.assertEqual(self.text(key, tw.COL_DURATION), "5:00.0")

    def test_a_duration_needs_a_start_in_per_video_mode(self):
        self.project.set_trial_protocol(300.0, START_PER_VIDEO)
        self.go()
        self.edit(self.keys[0], tw.COL_DURATION, "2:00")
        self.assertIsNone(self.trial())
        self.assertIn("start first", list(self.stage.model.bad_cells().values())[0][1])

    def test_the_in_cell_editor_commits_through_the_delegate(self):
        self.go()
        key = self.keys[0]
        index = self.cell(key, tw.COL_START)
        self.stage.table.edit(index)
        editor = self.stage.delegate._editor
        self.assertIsNotNone(editor)
        editor.setText("1:30")
        self.assertTrue(self.stage.delegate.has_open_editor())
        self.stage.delegate.commit_open_editor()
        self.app.processEvents()
        self.assertEqual(self.trial().start_time, 90.0)

    def test_the_editor_turns_red_while_the_text_is_not_a_time(self):
        self.go()
        self.stage.table.edit(self.cell(self.keys[0], tw.COL_END))
        editor = self.stage.delegate._editor
        editor.setText("12x")
        self.assertIn("border", editor.styleSheet())
        editor.setText("12")
        self.assertEqual(editor.styleSheet(), "")
        self.stage.delegate.commit_open_editor()

    def test_changes_made_while_typing_wait_for_the_editor_to_close(self):
        self.go()
        self.stage.table.edit(self.cell(self.keys[0], tw.COL_START))
        editor = self.stage.delegate._editor
        editor.setText("77")
        self.project.set_trial(self.keys[2], start=5)       # someone else
        self.assertEqual(editor.text(), "77")
        self.stage.delegate.commit_open_editor()
        self.app.processEvents()
        self.assertEqual(self.text(self.keys[2], tw.COL_START), "0:05.0")
        self.assertEqual(self.trial().start_time, 77.0)

    def test_leaving_the_stage_commits_what_is_being_typed(self):
        self.go()
        self.stage.table.edit(self.cell(self.keys[0], tw.COL_START))
        self.stage.delegate._editor.setText("2:10")
        self.assertTrue(self.window.go_to_stage("review"))
        self.assertEqual(self.trial().start_time, 130.0)

    def test_nothing_is_edited_while_a_run_is_in_progress(self):
        self.go()
        self.window.set_busy(True, "")
        self.stage.refresh()
        flags = self.stage.model.flags(self.cell(self.keys[0], tw.COL_START))
        self.assertFalse(flags & Qt.ItemFlag.ItemIsEditable)
        self.assertFalse(self.edit(self.keys[0], tw.COL_START, "5"))
        self.assertIsNone(self.trial())
        self.assertFalse(self.stage.protocol.apply_button.isEnabled())
        self.window.set_busy(False)
        self.stage.refresh()
        self.assertTrue(self.edit(self.keys[0], tw.COL_START, "5"))


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class EffectiveSettingsTests(TrialsFixture):
    def test_an_edited_video_runs_with_the_times_that_were_typed(self):
        self.project.set_trial_protocol(120.0, START_PER_VIDEO)
        self.go()
        a, b, c = self.keys
        self.edit(a, tw.COL_START, "0:20")
        self.edit(b, tw.COL_START, "35.5")
        self.edit(b, tw.COL_END, "4:00")
        settings = {k: self.project.effective_settings(k) for k in self.keys}
        self.assertEqual((settings[a].start_time, settings[a].end_time), (20.0, 140.0))
        self.assertEqual((settings[b].start_time, settings[b].end_time), (35.5, 240.0))
        # The one still without a start keeps the template's time (and says so).
        self.assertEqual(settings[c].start_time, 0.0)
        self.assertEqual(self.project.trial_status(c), TRIAL_NEEDS_START)

    def test_changing_the_protocol_moves_the_end_of_videos_that_follow_it(self):
        self.go()
        self.edit(self.keys[0], tw.COL_START, "10")
        self.stage.set_protocol(180.0, START_VIDEO)
        self.assertEqual(self.project.effective_settings(self.keys[0]).end_time, 190.0)
        self.assertEqual(self.text(self.keys[0], tw.COL_END), "3:10.0  protocol")
        self.stage.set_protocol(None, START_VIDEO)
        self.assertIsNone(self.project.effective_settings(self.keys[0]).end_time)
        self.assertEqual(self.text(self.keys[0], tw.COL_END), "video end")


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class PlayerTests(RealVideoFixture):
    def test_selecting_a_row_loads_its_video_at_its_start(self):
        self.project.set_trial(self.keys[1], start=1.0)
        player = self.show(1)
        self.assertEqual(player.video_path, self.project.entry(self.keys[1]).path)
        self.assertEqual(player.current_frame, int(round(1.0 * FPS)))
        self.assertTrue(self.stage.bar.duration() > 3.0)
        self.stage.select_row(self.keys[0])
        self.assertTrue(player.wait_idle())
        self.assertEqual(player.video_path, self.project.entry(self.keys[0]).path)
        self.assertEqual(player.current_frame, 0)

    def test_set_start_here_takes_the_time_of_the_frame_on_screen(self):
        self.show(0)
        self.seek(10)
        self.assertTrue(self.stage.set_start_here())
        self.assertEqual(self.trial().start_time, round(10 / FPS, 3))
        self.assertTrue(self.trial().start_set)
        self.assertEqual(self.text(self.keys[0], tw.COL_START), "0:00.4")

    def test_set_end_here_takes_the_time_of_the_frame_on_screen(self):
        self.show(0)
        self.seek(60)
        self.assertTrue(self.stage.set_end_here())
        self.assertEqual(self.trial().end_time, round(60 / FPS, 3))
        self.assertEqual(self.text(self.keys[0], tw.COL_SOURCE), "manual")

    def test_the_buttons_do_the_same(self):
        self.show(0)
        self.seek(20)
        self.stage.pane.set_start_button.click()
        self.seek(50)
        self.stage.pane.set_end_button.click()
        self.assertEqual(self.trial().start_time, 0.8)
        self.assertEqual(self.trial().end_time, 2.0)

    def test_go_to_start_returns_to_the_trial_start(self):
        self.project.set_trial(self.keys[0], start=2.0)
        self.show(0)
        self.seek(70)
        self.stage.pane.to_start_button.click()
        self.assertTrue(self.stage.player.wait_idle())
        self.assertEqual(self.stage.player.current_frame, 50)

    def test_the_buttons_and_hints_exist_with_their_keys(self):
        self.go()
        pane = self.stage.pane
        self.assertIn("Set start here", pane.set_start_button.text())
        self.assertIn("Set end here", pane.set_end_button.text())
        self.assertEqual(pane.start_hint.text(), "S")
        self.assertEqual(pane.end_hint.text(), "E")

    def test_s_and_e_keys_work_from_the_table_and_from_the_player_pane(self):
        self.show(0)
        self.seek(8)
        self.key_click(self.stage.table, Qt.Key.Key_S)
        self.assertEqual(self.trial().start_time, round(8 / FPS, 3))
        self.seek(40)
        self.key_click(self.stage.pane, Qt.Key.Key_E)
        self.assertEqual(self.trial().end_time, round(40 / FPS, 3))

    def test_arrow_keys_step_frames_and_shift_steps_ten(self):
        player = self.show(0)
        self.seek(20)
        self.key_click(self.stage.table, Qt.Key.Key_Right)
        self.assertTrue(player.wait_idle())
        self.assertEqual(player.current_frame, 21)
        self.key_click(self.stage.table, Qt.Key.Key_Left)
        self.assertTrue(player.wait_idle())
        self.assertEqual(player.current_frame, 20)
        self.key_click(self.stage.pane, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
        self.assertTrue(player.wait_idle())
        self.assertEqual(player.current_frame, 30)
        self.key_click(self.stage.pane, Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier)
        self.assertTrue(player.wait_idle())
        self.assertEqual(player.current_frame, 20)

    def test_the_scrub_bar_moves_the_player_and_marks_the_window(self):
        self.project.set_trial_protocol(2.0, START_PER_VIDEO)
        self.project.set_trial(self.keys[0], start=1.0)
        player = self.show(0)
        start, end = self.stage.bar.window()
        self.assertEqual((start, end), (1.0, 3.0))
        self.stage.bar.seek_requested.emit(2.0)
        self.assertTrue(player.wait_idle())
        self.assertEqual(player.current_frame, 50)
        self.assertAlmostEqual(self.stage.bar.playhead(), 2.0, places=2)
        # Clicking the bar asks for the time under the mouse.
        asked = []
        self.stage.bar.seek_requested.connect(asked.append)
        QTest.mouseClick(self.stage.bar, Qt.MouseButton.LeftButton,
                         pos=self.stage.bar.rect().center())
        self.assertAlmostEqual(asked[0], self.stage.bar.duration() / 2, delta=0.2)

    def test_editing_a_start_shows_that_frame(self):
        player = self.show(0)
        self.edit(self.keys[0], tw.COL_START, "2")
        self.assertTrue(player.wait_idle())
        self.assertEqual(player.current_frame, 50)

    def test_nothing_is_taken_from_a_video_that_cannot_be_opened(self):
        Path(self.project.entry(self.keys[2]).path).write_bytes(b"not a video")
        self.go()
        self.stage.select_row(self.keys[2])
        self.assertFalse(self.stage.pane.set_start_button.isEnabled())
        self.assertIn("could not be opened", self.stage.pane.set_start_button.toolTip())
        self.assertFalse(self.stage.set_start_here())
        self.assertIsNone(self.project.entry(self.keys[2]).trial)

    def test_leaving_the_stage_releases_the_video(self):
        player = self.show(0)
        self.assertTrue(player.capture_open)
        self.assertTrue(self.window.go_to_stage("review"))
        self.assertFalse(player.capture_open)
        self.go()
        self.assertTrue(self.stage.player.wait_idle())
        self.assertTrue(self.stage.player.capture_open)

    def test_closing_the_project_releases_the_video(self):
        player = self.show(0)
        self.prompts.say("ask_save_changes", "discard")
        self.window.close_project()
        self.assertFalse(player.capture_open)
        self.assertEqual(self.stage.pages.currentIndex(), PAGE_EMPTY)


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class KeyboardFlowTests(RealVideoFixture):
    def test_tagging_a_cohort_with_s_and_enter(self):
        self.project.set_trial_protocol(2.0, START_PER_VIDEO)
        self.go()
        a, b, c = self.keys
        # Starts on the first video that needs a start.
        self.assertEqual(self.stage.selected_row().key, a)
        for frame, key in ((5, a), (12, b), (30, c)):
            self.assertTrue(self.stage.player.wait_idle())
            self.seek(frame)
            self.key_click(self.stage.table, Qt.Key.Key_S)
            self.assertEqual(self.project.entry(key).trial.start_time, round(frame / FPS, 3))
            self.key_click(self.stage.table, Qt.Key.Key_Return)
        self.assertTrue(all(self.project.trial_status(k) == TRIAL_OK for k in self.keys))
        self.assertEqual(self.stage.can_advance(), (True, ""))

    def test_enter_goes_to_the_next_row_needing_a_start_and_wraps(self):
        self.project.set_trial_protocol(300.0, START_PER_VIDEO)
        self.project.set_trial(self.keys[1], start=2.0)
        self.go()
        a, _b, c = self.keys
        self.stage.select_row(a)
        self.key_click(self.stage.table, Qt.Key.Key_Return)
        self.assertEqual(self.stage.selected_row().key, c)       # skips the one with a start
        self.key_click(self.stage.table, Qt.Key.Key_N)
        self.assertEqual(self.stage.selected_row().key, a)       # wraps around

    def test_next_needing_start_says_when_there_is_none(self):
        self.go()
        self.assertFalse(self.stage.next_needing_start())
        self.assertFalse(self.stage.next_button.isEnabled())
        self.assertIn("start", self.stage.next_button.toolTip())
        self.project.set_trial_protocol(300.0, START_PER_VIDEO)
        self.assertTrue(self.stage.next_button.isEnabled())

    def test_the_button_and_the_footer_link_do_the_same(self):
        self.project.set_trial_protocol(300.0, START_PER_VIDEO)
        self.project.set_trial(self.keys[0], start=1.0)
        self.go()
        self.stage.select_row(self.keys[0])
        self.stage.next_button.click()
        self.assertEqual(self.stage.selected_row().key, self.keys[1])
        self.stage.select_row(self.keys[0])
        self.assertTrue(self.stage.show_next_button.isVisibleTo(self.window.footer))
        self.stage.show_next_button.click()
        self.assertEqual(self.stage.selected_row().key, self.keys[1])

    def test_the_video_picked_on_the_videos_page_is_selected_first(self):
        self.project.set_trial_protocol(300.0, START_PER_VIDEO)
        self.window.video_panel.select_keys([self.keys[2]])
        self.window.video_panel.flush_selection()
        self.go()
        self.assertEqual(self.stage.selected_row().key, self.keys[2])

    def test_keys_do_nothing_without_a_project(self):
        self.go()
        self.prompts.say("ask_save_changes", "discard")
        self.window.close_project()
        self.key_click(self.stage.table, Qt.Key.Key_S)      # must not raise


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class ProtocolTests(TrialsFixture):
    def test_the_card_shows_the_project_protocol(self):
        self.project.set_trial_protocol(480.0, START_PER_VIDEO)
        self.go()
        card = self.stage.protocol
        self.assertTrue(card.fixed_radio.isChecked())
        self.assertEqual(card.duration_edit.text(), "8:00.0")
        self.assertTrue(card.start_per_video_radio.isChecked())
        self.project.set_trial_protocol(None, START_VIDEO)
        self.assertTrue(card.until_end_radio.isChecked())
        self.assertTrue(card.start_video_radio.isChecked())

    def test_the_card_explains_how_the_end_is_decided(self):
        self.assertEqual(self.stage.protocol.help_label.text(),
                         "End = start + duration unless you set an end yourself.")
        card = self.stage.protocol
        texts = {w.text() for w in card.findChildren(QLabel)}
        texts |= {w.text() for w in card.findChildren(QRadioButton)}
        for text in ("Trial duration", "Trial start", "Fixed length",
                     "Until the video ends", "At the start of each video",
                     "Set per video (when the mouse enters)"):
            self.assertIn(text, texts)
        self.assertEqual(self.stage.protocol.apply_button.text(), "Apply to all videos")

    def test_typing_a_duration_writes_the_protocol(self):
        self.go()
        card = self.stage.protocol
        card.fixed_radio.click()
        card.duration_edit.setText("5:00")
        card.duration_edit.editingFinished.emit()
        self.assertEqual(self.project.trial_protocol.duration_s, 300.0)
        card.duration_edit.setText("90")
        card.duration_edit.editingFinished.emit()
        self.assertEqual(self.project.trial_protocol.duration_s, 90.0)

    def test_until_the_video_ends_clears_the_duration(self):
        self.project.set_trial_protocol(300.0, START_VIDEO)
        self.go()
        self.stage.protocol.until_end_radio.click()
        self.assertIsNone(self.project.trial_protocol.duration_s)
        self.assertFalse(self.stage.protocol.duration_edit.isEnabled())

    def test_an_invalid_duration_is_not_written_and_says_why(self):
        self.project.set_trial_protocol(300.0, START_VIDEO)
        self.go()
        card = self.stage.protocol
        card.duration_edit.setText("nope")
        card.duration_edit.editingFinished.emit()
        self.assertEqual(self.project.trial_protocol.duration_s, 300.0)
        self.assertFalse(card.duration_error.isHidden())
        self.assertIn("not a time", card.duration_error.text())
        card.duration_edit.setText("0")
        card.duration_edit.editingFinished.emit()
        self.assertIn("above 0", card.duration_error.text())
        self.assertEqual(self.project.trial_protocol.duration_s, 300.0)

    def test_the_start_choice_writes_the_protocol(self):
        self.go()
        card = self.stage.protocol
        card.start_per_video_radio.click()
        self.assertEqual(self.project.trial_protocol.start_mode, START_PER_VIDEO)
        self.assertEqual(self.stage.counts()[2], 3)
        card.start_video_radio.click()
        self.assertEqual(self.project.trial_protocol.start_mode, START_VIDEO)

    def test_the_collapsed_card_summarises_the_protocol(self):
        self.project.set_trial_protocol(300.0, START_PER_VIDEO)
        self.go()
        card = self.stage.protocol
        card.section.set_expanded(False)
        self.assertEqual(card.section._summary.text(), "5:00 per trial · start set per video")
        self.assertEqual(card.summary_text(), "5:00 per trial · start set per video")

    def test_apply_to_all_asks_before_dropping_manual_ends(self):
        self.project.set_trial_protocol(300.0, START_PER_VIDEO)
        self.project.set_trial(self.keys[0], start=10, end=100)
        self.project.set_trial(self.keys[1], start=20)
        self.go()
        self.prompts.say("confirm", False)
        self.assertIsNone(self.stage.apply_protocol_to_all())
        self.assertEqual(self.trial(0).end_time, 100)
        (title, text), = self.prompts.called("confirm")
        self.assertIn("1 video", text)
        self.assertIn("ends", text)
        self.prompts.say("confirm", True)
        self.assertEqual(self.stage.apply_protocol_to_all(), [self.keys[0]])
        self.assertIsNone(self.trial(0).end_time)
        self.assertEqual(self.trial(0).start_time, 10)           # starts are kept per video
        self.assertEqual(self.trial(1).start_time, 20)

    def test_apply_to_all_also_names_the_starts_it_resets(self):
        self.project.set_trial_protocol(300.0, START_VIDEO)
        self.project.set_trial(self.keys[0], start=10)
        self.go()
        self.prompts.say("confirm", True)
        self.stage.protocol.apply_button.click()
        (title, text), = self.prompts.called("confirm")
        self.assertIn("ends and starts", text)
        self.assertIsNone(self.trial(0))

    def test_apply_to_all_does_not_ask_when_nothing_is_overwritten(self):
        self.project.set_trial_protocol(300.0, START_PER_VIDEO)
        self.project.set_trial(self.keys[0], start=10)           # a start only
        self.go()
        self.assertEqual(self.stage.apply_protocol_to_all(), [])
        self.assertEqual(self.prompts.called("confirm"), [])
        self.assertEqual(self.trial(0).start_time, 10)

    def test_a_duration_being_typed_is_written_when_leaving(self):
        self.go()
        card = self.stage.protocol
        card.fixed_radio.click()
        card.duration_edit.setText("8:00")
        self.window.go_to_stage("review")
        self.assertEqual(self.project.trial_protocol.duration_s, 480.0)


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class NavigationTests(TrialsFixture):
    def test_the_next_button_reads_review_and_run(self):
        self.assertEqual(self.stage.next_label(), "Review & run")
        self.go()
        self.assertEqual(self.window.footer.next_text(), "Next: Review & run →")

    def test_it_is_ready_once_every_trial_is_valid(self):
        self.go()
        self.assertEqual(self.stage.can_advance(), (True, ""))
        self.assertEqual(self.stage.status()[0], "done")

    def test_videos_without_a_start_block_the_next_stage_and_say_how_many(self):
        self.project.set_trial_protocol(300.0, START_PER_VIDEO)
        self.project.set_trial(self.keys[0], start=4)
        self.go()
        allowed, reason = self.stage.can_advance()
        self.assertFalse(allowed)
        self.assertEqual(reason, "2 of 3 videos need a start time")
        self.assertFalse(self.window.footer.next_button.isEnabled())
        self.assertEqual(self.window.footer.next_button.toolTip(), reason)
        self.assertEqual(self.stage.status()[0], "attention")

    def test_invalid_times_block_it_too(self):
        self.project.set_trial(self.keys[0], start=9, end=3)
        self.go()
        allowed, reason = self.stage.can_advance()
        self.assertFalse(allowed)
        self.assertEqual(reason, "1 of 3 videos has times that cannot be used")

    def test_the_footer_counts_ready_and_waiting_videos(self):
        self.project.set_trial_protocol(300.0, START_PER_VIDEO)
        self.project.set_trial(self.keys[0], start=4)
        self.go()
        text, chip = self.stage.footer_text()
        self.assertEqual(text, "1 ready · 2 need a start")
        self.assertEqual(chip, ("warning", "Needs attention"))
        self.window._message_done()          # the "Added 3 videos" note is over
        self.assertEqual(self.window.footer.status_label.text(), text)
        self.project.set_trial(self.keys[1], start=5)
        self.assertEqual(self.stage.footer_text()[0], "2 ready · 1 needs a start")
        self.project.set_trial(self.keys[2], start=6)
        text, chip = self.stage.footer_text()
        self.assertEqual(text, "All 3 videos are ready")
        self.assertEqual(chip, ("ok", "Done"))
        self.assertEqual(self.window.footer.status_label.text(), text)
        self.assertTrue(self.window.footer.next_button.isEnabled())

    def test_the_footer_shows_invalid_videos_too(self):
        self.project.set_trial(self.keys[0], start=9, end=3)
        self.go()
        self.assertEqual(self.stage.footer_text()[0], "2 ready · 1 with invalid times")

    def test_show_next_is_in_the_footer_only_while_something_is_waiting(self):
        link = self.stage.show_next_button
        self.assertEqual(link.text(), "Show next")
        self.go()
        self.assertFalse(link.isVisibleTo(self.window.footer))
        self.project.set_trial_protocol(300.0, START_PER_VIDEO)
        self.assertTrue(link.isVisibleTo(self.window.footer))
        self.window.go_to_stage("review")
        self.assertFalse(link.isVisibleTo(self.window.footer))
        self.go()
        self.assertTrue(link.isVisibleTo(self.window.footer))

    def test_the_stepper_shows_the_trials_status(self):
        self.project.set_trial_protocol(300.0, START_PER_VIDEO)
        self.window.refresh_navigation()
        self.window.go_to_stage("review")
        self.assertEqual(self.window.stepper.status("trials"), "attention")
        self.project.set_trial_protocol(300.0, START_VIDEO)
        self.window.refresh_navigation()
        self.assertEqual(self.window.stepper.status("trials"), "done")


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class EmptyStateTests(WorkspaceFixture):
    def setUp(self):
        super().setUp()
        self.stage = self.window.stage("trials")
        self.addCleanup(self.stage.shutdown)

    def test_without_a_project(self):
        self.assertEqual(self.stage.pages.currentIndex(), PAGE_EMPTY)
        self.assertEqual(self.stage.empty.title(), "Open a study first")
        self.assertEqual(self.stage.can_advance()[0], False)

    def test_without_a_paradigm(self):
        self.new_project_at()
        self.assertEqual(self.stage.pages.currentIndex(), PAGE_EMPTY)
        self.assertEqual(self.stage.empty.title(), "Choose the test first")

    def test_without_videos_the_button_goes_to_the_videos_page(self):
        self.new_project_at()
        self.window.change_paradigm("Y-Maze")
        self.assertEqual(self.stage.pages.currentIndex(), PAGE_EMPTY)
        self.assertEqual(self.stage.empty.title(), "Add your videos first")
        self.assertEqual(self.stage.can_advance(), (False, "Add a video first"))
        self.assertTrue(self.window.go_to_stage("trials", force=True))
        self.stage.empty.buttons[0].click()
        self.assertEqual(self.window.current_stage_id, "videos")

    def test_the_window_does_not_open_it_without_videos(self):
        self.new_project_at()
        self.window.change_paradigm("Y-Maze")
        self.assertFalse(self.window.go_to_stage("trials"))

    def test_tail_suspension_needs_its_zones_first(self):
        self.new_project_at()
        self.window.change_paradigm(TST)
        self.add_videos()
        self.assertEqual(self.stage.pages.currentIndex(), PAGE_EMPTY)
        self.assertEqual(self.stage.empty.title(), "Draw the mouse zones first")
        self.assertEqual(self.stage.empty.buttons[0].text(), "Go to Arena & zones")
        self.window.project.set_template(tst_settings())
        self.assertEqual(self.stage.pages.currentIndex(), PAGE_MAIN)

    def test_without_the_arena_the_times_can_still_be_entered(self):
        self.new_project_at()
        self.window.change_paradigm("Y-Maze")
        self.add_videos()
        self.assertEqual(self.stage.pages.currentIndex(), PAGE_MAIN)
        self.assertTrue(self.stage.notice.isVisibleTo(self.stage))
        key = self.window.project.keys()[0]
        row = self.stage.model.row_of(key)
        self.assertTrue(self.stage.model.setData(self.stage.model.index(row, tw.COL_START), "30"))
        self.assertEqual(self.window.project.entry(key).trial.start_time, 30.0)
        self.assertEqual(self.stage.model.data(self.stage.model.index(row, tw.COL_START)), "0:30.0")


class TstFixture(TrialsFixture):
    paradigm = TST

    def make_template(self):
        return tst_settings(3)


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class TailSuspensionTests(TstFixture):
    MICE = ("mouse_1", "mouse_2", "mouse_3")

    def test_one_row_per_mouse_grouped_under_its_video(self):
        self.go()
        rows = self.stage.model.rows()
        self.assertEqual(len(rows), 3 * (1 + 3))
        kinds = [r.kind for r in rows]
        self.assertEqual(kinds, ["header", "mouse", "mouse", "mouse"] * 3)
        self.assertEqual([r.subject for r in rows[:4]], [None, *self.MICE])
        self.assertEqual(self.text(self.keys[0], tw.COL_VIDEO, "mouse_2"), "    › mouse_2")
        self.assertEqual(rows[1].label, "m1.mp4 › mouse_1")

    def test_the_video_rows_are_not_selectable(self):
        self.go()
        header = self.stage.model.index(0, tw.COL_VIDEO)
        self.assertFalse(self.stage.model.flags(header) & Qt.ItemFlag.ItemIsSelectable)
        self.assertEqual(self.stage.model.data(header), "m1.mp4")

    def test_a_mouse_without_a_window_uses_the_whole_video(self):
        self.go()
        self.assertEqual(self.text(self.keys[0], tw.COL_END, "mouse_1"), "video end")
        self.assertEqual(self.text(self.keys[0], tw.COL_SOURCE, "mouse_1"), "whole")

    def test_each_mouse_has_its_own_window_through_per_roi_times(self):
        self.go()
        key = self.keys[0]
        self.assertTrue(self.edit(key, tw.COL_END, "4:00", subject="mouse_1"))
        self.assertEqual(self.trial().per_roi_times, {"mouse_1": (0.0, 240.0)})
        self.edit(key, tw.COL_START, "30", subject="mouse_1")
        self.edit(key, tw.COL_END, "5:00", subject="mouse_2")
        self.edit(key, tw.COL_START, "1:00", subject="mouse_2")
        self.assertEqual(self.trial().per_roi_times,
                         {"mouse_1": (30.0, 240.0), "mouse_2": (60.0, 300.0)})
        windows = self.project.effective_settings(key).per_roi_times
        self.assertEqual(windows, {"mouse_1": (30.0, 240.0), "mouse_2": (60.0, 300.0)})
        self.assertEqual(self.text(key, tw.COL_START, "mouse_2"), "1:00.0")
        self.assertEqual(self.text(key, tw.COL_DURATION, "mouse_2"), "4:00.0")
        self.assertEqual(self.text(key, tw.COL_SOURCE, "mouse_2"), "manual")
        # Other videos are untouched.
        self.assertIsNone(self.project.entry(self.keys[1]).trial)

    def test_a_duration_gives_a_mouse_its_end(self):
        self.go()
        key = self.keys[0]
        self.edit(key, tw.COL_END, "9:00", subject="mouse_3")
        self.edit(key, tw.COL_START, "20", subject="mouse_3")
        self.edit(key, tw.COL_DURATION, "6:00", subject="mouse_3")
        self.assertEqual(self.trial().per_roi_times["mouse_3"], (20.0, 380.0))

    def test_a_start_alone_needs_to_know_where_the_video_ends(self):
        self.go()
        self.edit(self.keys[0], tw.COL_START, "10", subject="mouse_1")
        self.assertIsNone(self.trial())
        self.assertIn("end", list(self.stage.model.bad_cells().values())[0][1])

    def test_clearing_a_mouse_removes_its_window(self):
        self.project.set_trial(self.keys[0], per_roi_times={
            "mouse_1": (1.0, 5.0), "mouse_2": (2.0, 6.0)})
        self.go()
        self.assertTrue(self.edit(self.keys[0], tw.COL_START, "", subject="mouse_1"))
        self.assertEqual(self.trial().per_roi_times, {"mouse_2": (2.0, 6.0)})
        self.assertTrue(self.edit(self.keys[0], tw.COL_START, "", subject="mouse_2"))
        self.assertIsNone(self.trial())
        self.assertEqual(self.text(self.keys[0], tw.COL_SOURCE, "mouse_2"), "whole")

    def test_template_windows_are_shown_and_become_manual_when_edited(self):
        template = tst_settings(3)
        template.per_roi_times = {"mouse_1": (0.0, 300.0), "mouse_2": (10.0, 310.0)}
        self.project.set_template(template)
        self.go()
        key = self.keys[0]
        self.assertEqual(self.text(key, tw.COL_SOURCE, "mouse_2"), "template")
        self.assertEqual(self.text(key, tw.COL_END, "mouse_2"), "5:10.0  template")
        self.edit(key, tw.COL_START, "20", subject="mouse_2")
        self.assertEqual(self.trial().per_roi_times,
                         {"mouse_1": (0.0, 300.0), "mouse_2": (20.0, 310.0)})
        self.assertEqual(self.text(key, tw.COL_SOURCE, "mouse_1"), "manual")

    def test_a_window_that_ends_before_it_starts_is_invalid(self):
        self.go()
        key = self.keys[0]
        self.edit(key, tw.COL_END, "20", subject="mouse_1")
        self.edit(key, tw.COL_START, "50", subject="mouse_1")
        self.assertEqual(self.project.trial_status(key), TRIAL_INVALID)
        self.assertEqual(self.text(key, tw.COL_STATUS, "mouse_1"), "Invalid")
        self.assertEqual(self.text(key, tw.COL_STATUS), "Invalid")        # the video row
        self.assertFalse(self.stage.can_advance()[0])

    def test_the_protocol_card_is_hidden_and_stays_at_the_video_start(self):
        self.project.set_trial_protocol(None, START_PER_VIDEO)
        self.go()
        self.assertFalse(self.stage.protocol.isVisibleTo(self.stage))
        self.assertEqual(self.project.trial_protocol.start_mode, START_VIDEO)
        self.assertEqual(self.stage.subtitle, "Choose the time window of each mouse in each video.")
        self.assertTrue(self.stage.can_advance()[0])

    def test_the_other_mice_are_drawn_on_the_scrub_bar(self):
        self.project.set_trial(self.keys[0], per_roi_times={
            "mouse_1": (1.0, 5.0), "mouse_2": (2.0, 6.0), "mouse_3": (3.0, 7.0)})
        self.go()
        self.stage.select_row(self.keys[0], "mouse_2")
        self.assertEqual(self.stage.bar.window(), (2.0, 6.0))
        self.assertEqual(sorted(self.stage.bar._others), [(1.0, 5.0), (3.0, 7.0)])

    def test_the_next_needing_button_is_hidden_when_nothing_can_wait(self):
        self.go()
        self.assertTrue(self.stage.next_button.isHidden())
        self.edit(self.keys[0], tw.COL_END, "20", subject="mouse_1")
        self.edit(self.keys[0], tw.COL_START, "50", subject="mouse_1")
        self.assertFalse(self.stage.next_button.isHidden())


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class TailSuspensionPlayerTests(TstFixture, RealVideoFixture):
    def make_template(self):
        return tst_settings(2)

    def test_set_start_and_end_here_for_the_selected_mouse(self):
        self.show(0, "mouse_2")
        self.seek(25)
        self.assertTrue(self.stage.set_end_here())
        self.assertEqual(self.trial().per_roi_times, {"mouse_2": (0.0, 1.0)})
        self.seek(5)
        self.assertTrue(self.stage.set_start_here())
        self.assertEqual(self.trial().per_roi_times, {"mouse_2": (0.2, 1.0)})
        self.assertEqual(self.stage.bar.window(), (0.2, 1.0))

    def test_selecting_a_mouse_goes_to_its_start(self):
        self.project.set_trial(self.keys[0], per_roi_times={
            "mouse_1": (0.4, 2.0), "mouse_2": (1.2, 3.0)})
        player = self.show(0, "mouse_2")
        self.assertEqual(player.current_frame, 30)
        self.stage.select_row(self.keys[0], "mouse_1")
        self.assertTrue(player.wait_idle())
        self.assertEqual(player.current_frame, 10)


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class TimelineBarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_geometry_maps_times_both_ways(self):
        bar = tw.TimelineBar()
        bar.resize(520, 46)
        bar.set_video(100.0)
        self.assertAlmostEqual(bar.time_at(bar.x_of(40.0)), 40.0)
        self.assertEqual(bar.time_at(-50), 0.0)
        self.assertEqual(bar.time_at(10_000), 100.0)

    def test_it_paints_with_and_without_a_window(self):
        bar = tw.TimelineBar()
        bar.resize(520, 46)
        bar.grab()                                   # no video yet
        bar.set_video(300.0)
        bar.set_window(30.0, 330.0, end_is_default=True, others=[(10.0, 20.0)])
        bar.set_playhead(120.0)
        self.assertFalse(bar.grab().isNull())
        bar.set_window(None, None)
        self.assertFalse(bar.grab().isNull())


if __name__ == "__main__":
    unittest.main()
