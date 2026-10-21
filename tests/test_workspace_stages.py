"""The stage shell: navigation, stepper, footer, drawer, the Videos stage."""
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.workspace_support import HAS_RUNTIME, WorkspaceFixture

if HAS_RUNTIME:
    from PyQt6.QtCore import QCoreApplication, QEvent, QMimeData, QPoint, QPointF, Qt, QUrl
    from PyQt6.QtGui import QDropEvent, QKeySequence
    from PyQt6.QtTest import QTest
    from gui.project_table_model import COL_CONFIGURED, KeyRole
    from gui.workspace.stages import status as st
    from gui.workspace.stages.base import Stage
    from gui.workspace.stages.hosted import HostedSetupStage
    from gui.workspace.stages.videos_stage import (PAGE_PARADIGM, PAGE_VIDEOS,
                                                   PAGE_WELCOME)
    from gui.workspace.ui_kit import ParadigmCard, ParadigmPicker
    from tests.test_workspace_stage_status import settings_for


class StageFixture(WorkspaceFixture):
    def ready(self, paradigm="Y-Maze", template=True):
        """A saved project with a paradigm, three videos and a template."""
        self.new_project_at()
        self.window.change_paradigm(paradigm)
        if template:
            self.window.project.set_template(settings_for(paradigm))
        self.add_videos()
        self.window.refresh_navigation()
        return self.window.project

    def step_status(self, stage_id):
        return self.window.stepper.status(stage_id)


class FrameworkTests(StageFixture):
    def test_every_stage_is_a_stage_with_an_id_title_and_icon(self):
        for stage in self.window.stages.values():
            self.assertIsInstance(stage, Stage)
            self.assertTrue(stage.stage_id and stage.title and stage.icon_name)
            self.assertEqual(stage.header.title(), stage.title)

    def test_the_default_hooks_do_nothing_harmful(self):
        stage = Stage()
        self.addCleanup(stage.deleteLater)
        stage.on_project_changed(None)
        stage.on_project_modified()
        stage.on_selection_changed([])
        stage.on_enter()
        self.assertTrue(stage.on_leave())
        self.assertEqual(stage.can_advance(), (True, ""))
        self.assertIsNone(stage.next_label())
        self.assertEqual(stage.status(), (st.STATUS_TODO, ""))
        self.assertEqual(stage.footer_text(), ("", None))

    def test_content_can_be_set_and_replaced(self):
        from PyQt6.QtWidgets import QLabel
        stage = Stage()
        self.addCleanup(stage.deleteLater)
        first, second = QLabel("a"), QLabel("b")
        stage.set_content(first)
        stage.set_content(second)
        self.assertIs(first.parentWidget(), None)
        self.assertIs(second.parentWidget(), stage.content)

    def test_a_stage_without_the_window_has_no_project(self):
        stage = Stage()
        self.addCleanup(stage.deleteLater)
        self.assertIsNone(stage.project)
        stage.refresh_footer()      # must not raise

    def test_stage_status_follows_the_project_rules(self):
        project = self.ready()
        for stage in self.window.stages.values():
            found = stage.status()
            expected = st.stage_status(stage.stage_id, project)
            self.assertEqual(found, (expected.status, expected.reason))


class NavigationTests(StageFixture):
    def test_starts_on_the_videos_stage(self):
        self.assertEqual(self.window.current_stage_id, "videos")
        self.assertIs(self.window.stack.currentWidget(), self.window.videos_stage)

    def test_without_a_project_only_videos_is_reachable(self):
        for stage_id in self.window.stage_ids()[1:]:
            with self.subTest(stage_id):
                self.assertFalse(self.window.go_to_stage(stage_id))
        self.assertEqual(self.window.current_stage_id, "videos")

    def test_stages_open_in_any_order(self):
        self.ready()
        window = self.window
        for stage_id in ("review", "trials", "results", "arena", "videos"):
            self.assertTrue(window.go_to_stage(stage_id), stage_id)
            self.assertEqual(window.current_stage_id, stage_id)
            self.assertIs(window.stack.currentWidget(), window.stage(stage_id))

    def test_the_stage_changed_signal(self):
        self.ready()
        seen = []
        self.window.stage_changed.connect(seen.append)
        self.window.go_to_stage("arena")
        self.window.go_to_stage("arena")        # already there: no signal
        self.window.go_to_stage("trials")
        self.assertEqual(seen, ["arena", "trials"])

    def test_unknown_stage_is_a_key_error(self):
        with self.assertRaises(KeyError):
            self.window.go_to_stage("nope")

    def test_next_and_previous(self):
        self.ready()
        window = self.window
        self.assertTrue(window.next_stage())
        self.assertEqual(window.current_stage_id, "arena")
        self.assertTrue(window.previous_stage())
        self.assertEqual(window.current_stage_id, "videos")
        self.assertFalse(window.previous_stage())
        window.go_to_stage("results")
        self.assertFalse(window.next_stage())

    def test_next_is_refused_with_the_reason_when_the_stage_cannot_advance(self):
        self.new_project_at()
        window = self.window
        self.assertFalse(window.next_stage())
        self.assertEqual(window.current_stage_id, "videos")
        self.assertIn("test", window.statusBar().currentMessage())

    def test_a_stage_can_veto_leaving(self):
        self.ready()
        window = self.window
        window.go_to_stage("arena")
        with patch.object(window.stage("arena"), "on_leave", return_value=False) as veto:
            self.assertFalse(window.go_to_stage("trials"))
            self.assertFalse(window.next_stage())
            self.assertFalse(window.previous_stage())
        self.assertEqual(veto.call_count, 3)
        self.assertEqual(window.current_stage_id, "arena")
        self.assertTrue(window.go_to_stage("trials"))

    def test_a_forced_move_ignores_the_veto(self):
        self.ready()
        window = self.window
        window.go_to_stage("arena")
        with patch.object(window.stage("arena"), "on_leave", return_value=False):
            self.assertTrue(window.go_to_stage("videos", force=True))

    def test_the_hooks_are_called_on_enter_and_leave(self):
        self.ready()
        window = self.window
        calls = []
        arena = window.stage("arena")
        with patch.object(arena, "on_enter", side_effect=lambda: calls.append("enter")), \
                patch.object(arena, "on_leave", side_effect=lambda: calls.append("leave") or True):
            window.go_to_stage("arena")
            window.go_to_stage("videos")
        self.assertEqual(calls, ["enter", "leave"])

    def test_live_setup_stages_never_ask_before_leaving(self):
        # Arena, Detection and Trials write their edits live, so leaving
        # never needs an Apply / Discard prompt.
        self.ready()
        window = self.window
        for stage_id in ("arena", "detection", "trials"):
            window.go_to_stage(stage_id)
            self.assertTrue(window.go_to_stage("videos"))
            self.assertEqual(window.current_stage_id, "videos")
        self.assertEqual(self.prompts.called("ask_save_changes"), [])

    def test_closing_the_project_returns_to_the_first_stage(self):
        self.ready()
        self.window.go_to_stage("trials")
        self.window.save()
        self.assertTrue(self.window.close_project())
        self.assertEqual(self.window.current_stage_id, "videos")
        self.assertEqual(self.window.videos_stage.page, PAGE_WELCOME)

    def test_the_stage_is_remembered_per_project(self):
        path = self.new_project_at("a.bmaproj")
        window = self.window
        window.change_paradigm("Y-Maze")
        window.project.set_template(settings_for("Y-Maze"))
        self.add_videos()
        window.go_to_stage("trials")
        self.assertEqual(self.settings.workspace_stage(path), "trials")
        window.save()
        window.close_project()
        self.assertEqual(window.current_stage_id, "videos")
        self.assertTrue(window.open_project(path))
        self.assertEqual(window.current_stage_id, "trials")
        other = self.new_project_at("b.bmaproj")
        self.assertEqual(window.current_stage_id, "videos")
        self.assertNotEqual(self.settings.workspace_stage(other), "trials")

    def test_a_remembered_stage_that_is_no_longer_available_falls_back(self):
        path = self.new_project_at("a.bmaproj")
        self.window.close_project()
        self.settings.set_workspace_stage(path, "review")     # no videos yet
        self.assertTrue(self.window.open_project(path))
        self.assertEqual(self.window.current_stage_id, "videos")

    def test_show_tab_maps_the_old_tabs_to_stages(self):
        self.ready()
        window = self.window
        for tab, stage_id in (("setup", "arena"), ("run", "review"),
                              ("results", "results")):
            window.show_tab(tab)
            self.assertEqual(window.current_stage_id, stage_id)
            self.assertEqual(window.current_tab_name(), tab)


class StepperTests(StageFixture):
    def test_without_a_project(self):
        window = self.window
        self.assertEqual(self.step_status("videos"), "current")
        for stage_id in window.stage_ids()[1:]:
            self.assertEqual(self.step_status(stage_id), "disabled", stage_id)
        self.assertEqual(window.stepper.step_ids(), window.stage_ids())

    def test_statuses_follow_the_project(self):
        self.new_project_at()
        window = self.window
        self.assertEqual(self.step_status("arena"), "disabled")
        window.change_paradigm("Y-Maze")
        window.refresh_navigation()
        self.assertEqual(self.step_status("arena"), "disabled")     # no video yet
        self.add_videos()
        window.refresh_navigation()
        self.assertEqual(self.step_status("arena"), "todo")
        self.assertEqual(self.step_status("review"), "todo")
        window.project.set_template(settings_for("Y-Maze"))
        window.refresh_navigation()
        self.assertEqual(self.step_status("arena"), "done")
        self.assertEqual(self.step_status("detection"), "todo")   # not reviewed yet
        window.project.set_detection_reviewed()
        window.refresh_navigation()
        self.assertEqual(self.step_status("detection"), "done")
        self.assertEqual(self.step_status("trials"), "done")
        self.assertEqual(self.step_status("review"), "done")
        self.assertEqual(self.step_status("results"), "todo")

    def test_the_current_stage_is_marked_current(self):
        self.ready()
        window = self.window
        window.go_to_stage("arena")
        self.assertEqual(self.step_status("arena"), "current")
        self.assertEqual(self.step_status("videos"), "done")
        self.assertEqual(window.stepper.current_id(), "arena")

    def test_attention_for_a_missing_video(self):
        project = self.ready()
        project.videos[0].path.unlink()
        self.window.refresh_navigation()
        self.assertEqual(self.step_status("review"), "attention")
        self.assertIn("1 video", self.window.stepper.reason("review"))

    def test_the_stepper_is_refreshed_after_edits(self):
        project = self.ready()
        self.assertEqual(self.step_status("trials"), "done")
        project.set_trial_protocol(60.0, "per_video")
        QCoreApplication.processEvents()        # the coalesced refresh
        self.assertEqual(self.step_status("trials"), "attention")

    def test_clicking_a_step_goes_there(self):
        self.ready()
        self.window.stepper.step_clicked.emit("trials")
        self.assertEqual(self.window.current_stage_id, "trials")

    def test_a_disabled_step_cannot_be_reached_from_the_stepper(self):
        self.window.stepper.step_clicked.emit("arena")
        self.assertEqual(self.window.current_stage_id, "videos")

    def test_the_tst_detection_stage_is_called_motion_tuning(self):
        self.new_project_at()
        window = self.window
        self.assertEqual(window.stepper.label("detection"), "Detection")
        window.change_paradigm("Tail Suspension Test")
        window.refresh_navigation()
        self.assertEqual(window.stepper.label("detection"), "Motion tuning")
        self.assertEqual(window.stage("detection").header.title(), "Motion tuning")
        window.change_paradigm("Y-Maze")
        window.refresh_navigation()
        self.assertEqual(window.stepper.label("detection"), "Detection")

    def test_validation_is_shared_between_the_stages(self):
        project = self.ready()
        with patch.object(type(project), "validate", return_value=[]) as validate:
            self.window.invalidate_issues()
            self.window.refresh_navigation()
        self.assertEqual(validate.call_count, 1)


class FooterTests(StageFixture):
    def footer(self):
        return self.window.footer

    def test_the_first_stage_has_no_back_and_a_disabled_next(self):
        footer = self.footer()
        self.assertTrue(footer.back_button.isHidden())
        self.assertEqual(footer.next_text(), "Next: Arena & zones →")
        self.assertFalse(footer.next_button.isEnabled())
        self.assertIn("project", footer.next_button.toolTip())

    def test_next_says_why_it_is_disabled(self):
        self.new_project_at()
        footer = self.footer()
        self.assertFalse(footer.next_button.isEnabled())
        self.assertIn("test", footer.next_button.toolTip())
        self.window.change_paradigm("Y-Maze")
        self.window.refresh_navigation()
        self.assertFalse(footer.next_button.isEnabled())
        self.assertIn("video", footer.next_button.toolTip())
        self.add_videos()
        self.window.refresh_navigation()
        self.assertTrue(footer.next_button.isEnabled())

    def test_next_and_back_navigate(self):
        self.ready()
        footer = self.footer()
        footer.next_button.click()
        self.assertEqual(self.window.current_stage_id, "arena")
        self.assertFalse(footer.back_button.isHidden())
        self.assertEqual(footer.next_text(), "Next: Detection →")
        footer.back_button.click()
        self.assertEqual(self.window.current_stage_id, "videos")

    def test_the_last_stage_has_no_next(self):
        self.ready()
        self.window.go_to_stage("results")
        self.assertTrue(self.footer().next_button.isHidden())

    def test_the_status_line_and_chip_come_from_the_stage(self):
        self.ready()
        footer = self.footer()
        self.window._message_done()
        self.assertIn("3 videos", footer.status_label.text())
        self.assertFalse(footer.chip.isHidden())
        self.assertEqual(footer.chip.kind(), "ok")

    def test_messages_replace_the_status_for_a_while(self):
        self.ready()
        window = self.window
        window.show_message("Hello there")
        self.assertEqual(window.footer.status_label.text(), "Hello there")
        self.assertEqual(window.statusBar().currentMessage(), "Hello there")
        window.refresh_footer()
        self.assertEqual(window.footer.status_label.text(), "Hello there")
        window._message_done()
        self.assertNotEqual(window.footer.status_label.text(), "Hello there")

    def test_next_stays_disabled_while_the_next_stage_is_not_available(self):
        self.new_project_at()
        self.window.change_paradigm("Y-Maze")
        self.window.go_to_stage("videos")
        self.assertFalse(self.footer().next_button.isEnabled())


class KeyboardTests(StageFixture):
    def test_the_shortcuts_are_the_documented_ones(self):
        window = self.window
        for action, keys in ((window.next_stage_action, "Ctrl+PgDown"),
                             (window.previous_stage_action, "Ctrl+PgUp"),
                             (window.review_action, "F5"),
                             (window.details_action, "Ctrl+I")):
            self.assertEqual(action.shortcut(), QKeySequence(keys))
            self.assertIn(action, window.actions())

    def test_the_actions_navigate(self):
        self.ready()
        window = self.window
        window.next_stage_action.trigger()
        self.assertEqual(window.current_stage_id, "arena")
        window.next_stage_action.trigger()
        self.assertEqual(window.current_stage_id, "detection")
        window.previous_stage_action.trigger()
        self.assertEqual(window.current_stage_id, "arena")
        window.review_action.trigger()
        self.assertEqual(window.current_stage_id, "review")

    def test_f5_no_longer_starts_a_run(self):
        self.ready()
        fired = []
        self.window.run_requested.connect(lambda: fired.append(1))
        self.assertTrue(self.window.run_action.shortcut().isEmpty())
        self.window.review_action.trigger()
        self.assertEqual(fired, [])

    def test_the_keys_work_in_a_shown_window(self):
        self.ready()
        window = self.window
        window.show()
        window.activateWindow()
        self.app.processEvents()
        QTest.keyClick(window, Qt.Key.Key_PageDown, Qt.KeyboardModifier.ControlModifier)
        self.app.processEvents()
        if window.current_stage_id == "videos":
            self.skipTest("this platform plugin does not deliver shortcuts")
        self.assertEqual(window.current_stage_id, "arena")
        QTest.keyClick(window, Qt.Key.Key_PageUp, Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(window.current_stage_id, "videos")
        QTest.keyClick(window, Qt.Key.Key_F5)
        self.assertEqual(window.current_stage_id, "review")

    def test_ctrl_i_toggles_the_drawer(self):
        self.ready()
        window = self.window
        window.drawer.animation_ms = 0
        window.details_action.trigger()
        self.assertTrue(window.drawer.is_open())
        window.details_action.trigger()
        self.assertFalse(window.drawer.is_open())


class VideosStageTests(StageFixture):
    def stage(self):
        return self.window.videos_stage

    def test_no_project_shows_the_welcome_page(self):
        stage = self.stage()
        self.assertEqual(stage.page, PAGE_WELCOME)
        self.assertFalse(stage.header.isVisibleTo(stage))
        self.assertEqual(stage.welcome.hero.title(), "Start a new study")
        self.assertEqual(stage.can_advance()[0], False)

    def test_a_project_without_a_paradigm_shows_the_picker_first(self):
        self.new_project_at()
        stage = self.stage()
        self.assertEqual(stage.page, PAGE_PARADIGM)
        self.assertTrue(stage.header.isVisibleTo(stage))
        self.assertEqual(stage.header.title(), "What test are these videos from?")
        from mazes import AVAILABLE_MAZES
        self.assertEqual(stage.picker.names(), list(AVAILABLE_MAZES))
        self.assertIsNone(stage.picker.selected())
        self.assertTrue(stage.keep_button.isHidden())
        # The stepper label never changes with the page.
        self.assertEqual(self.window.stepper.label("videos"), "Videos")

    def test_an_unknown_paradigm_keeps_the_picker_without_a_selection(self):
        self.new_project_at()
        self.window.project.set_paradigm("Future Maze")
        stage = self.stage()
        self.assertEqual(stage.page, PAGE_PARADIGM)
        self.assertIsNone(stage.picker.selected())
        self.assertFalse(stage.can_advance()[0])

    def test_choosing_the_paradigm_moves_on_to_the_videos(self):
        self.new_project_at()
        stage = self.stage()
        stage.picker.card("Elevated Plus Maze").click()
        self.assertEqual(stage.page, PAGE_VIDEOS)
        self.assertEqual(self.window.project.paradigm, "Elevated Plus Maze")
        self.assertEqual(stage.header.title(), "Videos")
        self.assertEqual(stage.summary.name_label.text(), "Elevated Plus Maze")

    def test_an_empty_project_shows_the_drop_zone(self):
        self.new_project_at()
        self.window.change_paradigm("Y-Maze")
        panel = self.window.video_panel
        self.assertEqual(self.stage().page, PAGE_VIDEOS)
        self.assertTrue(panel.empty_state_visible)
        self.assertFalse(panel.view.isVisibleTo(panel))
        ok, reason = self.stage().can_advance()
        self.assertFalse(ok)
        self.assertIn("video", reason)

    def test_with_videos_the_table_and_the_compact_add_bar(self):
        self.ready()
        panel = self.window.video_panel
        self.assertFalse(panel.empty_state_visible)
        self.assertTrue(panel.view.isVisibleTo(panel))
        self.assertTrue(panel.add_bar.isVisibleTo(panel))
        self.assertEqual(self.stage().can_advance(), (True, ""))
        self.assertEqual(self.stage().next_label(), "Arena & zones")

    def test_the_table_columns(self):
        self.ready()
        panel = self.window.video_panel
        view, model = panel.view, panel.model
        headers = model.headers()
        header = view.horizontalHeader()
        visible = sorted(
            (header.visualIndex(c), c) for c in range(model.columnCount())
            if not view.isColumnHidden(c)
        )
        shown = [
            panel.proxy.headerData(c, Qt.Orientation.Horizontal) for _v, c in visible
        ]
        self.assertEqual(shown, ["Status", "Video", "Duration", "Trial", "Zones"])
        for hidden in ("Last run", "Notes", "Issues"):
            self.assertTrue(view.isColumnHidden(headers.index(hidden)), hidden)

    def test_metadata_columns_are_shown(self):
        self.ready()
        panel = self.window.video_panel
        panel.model.add_metadata_column("Group")
        column = panel.model.column_for_metadata("Group")
        self.assertFalse(panel.view.isColumnHidden(column))

    def test_the_header_menu_shows_and_hides_columns(self):
        self.ready()
        panel = self.window.video_panel
        menu = panel.build_header_menu()
        actions = {a.text(): a for a in menu.actions()}
        self.assertNotIn("Video", actions)           # the name always stays
        self.assertFalse(actions["Notes"].isChecked())
        actions["Notes"].toggle()
        notes = panel.model.headers().index("Notes")
        self.assertFalse(panel.view.isColumnHidden(notes))
        from gui.project_table_model import COL_SETTINGS
        actions["Duration"].toggle()
        self.assertTrue(panel.view.isColumnHidden(COL_SETTINGS))

    def test_the_status_chip_says_where_each_video_stands(self):
        project = self.ready()
        proxy = self.window.video_panel.proxy
        keys = project.keys()
        self.assertEqual(proxy.chip_for(keys[0]), ("info", "Set up"))
        project.set_verified(keys[:1], True)
        self.assertEqual(proxy.chip_for(keys[0]), ("ok", "Verified"))
        project.videos[1].last_status = "done"
        self.assertEqual(proxy.chip_for(keys[1]), ("ok", "Analysed"))
        project.videos[2].path.unlink()
        self.window.video_panel.model.refresh()
        self.assertEqual(proxy.chip_for(keys[2]), ("error", "Missing file"))
        index = proxy.index(0, COL_CONFIGURED)
        self.assertEqual(index.data(Qt.ItemDataRole.DisplayRole), "Verified")
        self.assertEqual(
            proxy.headerData(COL_CONFIGURED, Qt.Orientation.Horizontal), "Status"
        )

    def test_unconfigured_and_trial_problems_show_in_the_chip(self):
        project = self.ready(template=False)
        proxy = self.window.video_panel.proxy
        self.assertEqual(proxy.chip_for(project.keys()[0]), ("neutral", "Not set up"))
        project.set_template(settings_for("Y-Maze"))
        project.set_trial_protocol(60.0, "per_video")
        self.assertEqual(proxy.chip_for(project.keys()[0]), ("warning", "Needs start"))

    def test_the_duration_column_sorts_by_seconds(self):
        project = self.ready()
        panel = self.window.video_panel
        durations = {k: d for k, d in zip(project.keys(), (30.0, 5.0, 100.0))}
        for key, seconds in durations.items():
            project.entry(key).duration_s = seconds
        panel.model.refresh()
        from gui.project_table_model import COL_SETTINGS
        self.assertEqual(
            [panel.proxy.index(r, COL_SETTINGS).data() for r in range(3)],
            ["30s", "5s", "1m 40s"],
        )
        panel.view.sortByColumn(COL_SETTINGS, Qt.SortOrder.AscendingOrder)
        order = [panel.proxy.index(r, 0).data(KeyRole) for r in range(3)]
        self.assertEqual([durations[k] for k in order], [5.0, 30.0, 100.0])
        self.assertEqual(
            panel.proxy.headerData(COL_SETTINGS, Qt.Orientation.Horizontal), "Duration"
        )

    def test_unknown_durations_are_a_dash(self):
        self.ready()
        from gui.project_table_model import COL_SETTINGS
        self.assertEqual(self.window.video_panel.proxy.index(0, COL_SETTINGS).data(), "—")

    def test_durations_are_read_in_the_background(self):
        from tests.synthetic_video import write_indexed_video
        project = self.ready()
        real = write_indexed_video(self.root / "real.avi")
        panel = self.window.video_panel
        panel.add_paths([real])
        self.assertGreaterEqual(panel.request_durations(), 1)
        self.window.thumbnails.wait_idle()
        self.app.processEvents()
        from gui.project_table_model import COL_SETTINGS
        row = project.keys().index(project.find(real).key)
        self.assertEqual(panel.proxy.index(row, COL_SETTINGS).data(), "3s")

    def test_change_paradigm_and_keeping_it(self):
        self.ready()
        stage = self.stage()
        stage.summary.change_button.click()
        self.assertEqual(stage.page, PAGE_PARADIGM)
        self.assertFalse(stage.keep_button.isHidden())
        stage.keep_button.click()
        self.assertEqual(stage.page, PAGE_VIDEOS)

    def test_study_details_edit_the_name_and_notes(self):
        project = self.ready()
        stage = self.stage()
        self.assertFalse(stage.details.is_expanded())
        stage.details.set_expanded(True)
        stage.name_edit.setText("Renamed study")
        stage.notes_edit.setPlainText("Cohort of 12 animals")
        stage.commit_pending()
        self.assertEqual(project.name, "Renamed study")
        self.assertEqual(project.notes, "Cohort of 12 animals")
        self.assertIn("Renamed study", self.window.windowTitle())
        self.assertEqual(self.window.app_bar.project_label.text(), "Renamed study")

    def test_a_blank_name_is_refused(self):
        project = self.ready()
        stage = self.stage()
        stage.name_edit.setText("   ")
        stage.commit_pending()
        self.assertEqual(project.name, "study")
        self.assertEqual(stage.name_edit.text(), "study")

    def test_saving_commits_the_study_details(self):
        project = self.ready()
        self.stage().notes_edit.setPlainText("saved with the project")
        self.window.save()
        self.assertEqual(project.notes, "saved with the project")

    def test_the_app_bar_shows_the_save_state(self):
        bar = self.window.app_bar
        self.assertTrue(bar.project_label.isHidden())
        self.ready()
        self.assertEqual(bar.project_label.text(), "study")
        self.assertEqual(bar.save_chip.text(), "Unsaved changes")
        self.window.save()
        self.assertEqual(bar.save_chip.text(), "Saved")

    def test_busy_locks_the_study(self):
        self.ready()
        stage = self.stage()
        self.window.set_busy(True, "Running")
        self.assertTrue(stage.name_edit.isReadOnly())
        self.assertFalse(stage.picker.isEnabled())
        self.window.set_busy(False)
        self.assertFalse(stage.name_edit.isReadOnly())

    def test_the_footer_counts_the_selection(self):
        project = self.ready()
        panel = self.window.video_panel
        panel.select_keys(project.keys()[:2])
        panel.flush_selection()
        self.window._message_done()
        self.assertIn("2 selected", self.window.footer.status_label.text())


class DropAndAddTests(StageFixture):
    def drop(self, widget, paths):
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
        event = QDropEvent(
            QPointF(QPoint(5, 5)), Qt.DropAction.CopyAction, mime,
            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
        )
        widget.dropEvent(event)
        return event

    def test_dropping_on_the_empty_state_adds_the_videos(self):
        self.new_project_at()
        self.window.change_paradigm("Y-Maze")
        self.drop(self.window.video_panel, self.videos[:2])
        self.assertEqual(len(self.window.project.videos), 2)
        self.assertFalse(self.window.video_panel.empty_state_visible)

    def test_dropping_a_folder_adds_its_videos(self):
        self.new_project_at()
        self.window.change_paradigm("Y-Maze")
        self.drop(self.window.video_panel, [self.root / "videos"])
        self.assertEqual(len(self.window.project.videos), 3)

    def test_the_zone_buttons_open_the_dialogs(self):
        self.new_project_at()
        self.window.change_paradigm("Y-Maze")
        zone = self.window.video_panel.drop_zone
        self.prompts.say("open_files", self.videos[:1])
        zone.browse_button.click()
        self.assertEqual(len(self.window.project.videos), 1)
        self.window.video_panel.remove_keys(self.window.project.keys(), confirm=False)
        self.prompts.say("choose_folder", self.root / "videos")
        zone.folder_button.click()
        self.assertEqual(len(self.window.project.videos), 3)

    def test_the_add_bar_buttons_open_the_dialogs(self):
        self.ready()
        panel = self.window.video_panel
        extra = self.write_video("more/x.mp4", b"xx")
        self.prompts.say("open_files", [extra])
        panel.add_videos_button.click()
        self.assertEqual(len(self.window.project.videos), 4)
        folder = self.root / "folder2"
        folder.mkdir()
        self.write_video("folder2/y.mp4", b"yyy")
        self.prompts.say("choose_folder", folder)
        panel.add_folder_button.click()
        self.assertEqual(len(self.window.project.videos), 5)

    def test_the_subfolder_choice_reaches_the_drop_zone(self):
        self.new_project_at()
        self.window.change_paradigm("Y-Maze")
        panel = self.window.video_panel
        panel.recursive_check.setChecked(True)
        self.assertTrue(panel.drop_zone.recursive())
        panel.recursive_check.setChecked(False)
        self.assertFalse(panel.drop_zone.recursive())

    def test_a_drag_over_the_panel_lights_the_zone(self):
        from PyQt6.QtGui import QDragEnterEvent
        self.new_project_at()
        self.window.change_paradigm("Y-Maze")
        panel = self.window.video_panel
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(self.videos[0]))])
        event = QDragEnterEvent(
            QPoint(5, 5), Qt.DropAction.CopyAction, mime,
            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
        )
        panel.dragEnterEvent(event)
        self.assertTrue(panel.drop_zone.is_dragging())
        panel.dragLeaveEvent(QEvent(QEvent.Type.DragLeave))
        self.assertFalse(panel.drop_zone.is_dragging())

    def test_buttons_are_disabled_while_a_run_is_in_progress(self):
        self.ready()
        panel = self.window.video_panel
        self.window.set_busy(True)
        self.assertFalse(panel.add_videos_button.isEnabled())
        self.assertFalse(panel.add_folder_button.isEnabled())
        self.window.set_busy(False)
        self.assertTrue(panel.add_videos_button.isEnabled())


class RowActionTests(StageFixture):
    def setUp(self):
        super().setUp()
        self.ready()
        self.panel = self.window.video_panel
        self.view = self.panel.view

    def hover(self, row):
        self.view.show_row_actions(self.panel.proxy.index(row, 0))

    def test_buttons_follow_the_hovered_row(self):
        self.hover(1)
        bar = self.view.row_actions
        self.assertFalse(bar.isHidden())
        row_rect = self.view.visualRect(self.panel.proxy.index(1, 1))
        self.assertTrue(row_rect.top() <= bar.y() <= row_rect.bottom())
        self.assertEqual(set(bar.buttons), {"details", "reveal", "relink", "remove"})
        self.view.hide_row_actions()
        self.assertTrue(bar.isHidden())

    def test_relink_only_for_a_missing_file(self):
        self.hover(0)
        self.assertTrue(self.view.row_actions.buttons["relink"].isHidden())
        self.project_video(0).unlink()
        self.panel.model.refresh()
        self.hover(0)
        self.assertFalse(self.view.row_actions.buttons["relink"].isHidden())

    def project_video(self, row):
        return self.window.project.videos[row].path

    def test_details_selects_the_row_and_opens_the_drawer(self):
        self.window.drawer.animation_ms = 0
        self.hover(2)
        self.view.row_actions.buttons["details"].click()
        key = self.window.project.keys()[2]
        self.assertEqual(self.window.selected_keys(), [key])
        self.assertTrue(self.window.drawer.is_open())
        self.assertEqual(self.window.context_panel.selection(), [key])

    def test_reveal_and_remove(self):
        self.hover(0)
        self.view.row_actions.buttons["reveal"].click()
        self.assertEqual(self.prompts.called("reveal"), [(self.videos[0],)])
        self.hover(0)
        self.prompts.say("confirm", True)
        self.view.row_actions.buttons["remove"].click()
        self.assertEqual(len(self.window.project.videos), 2)

    def test_remove_and_relink_are_locked_while_busy(self):
        self.window.set_busy(True)
        self.hover(0)
        self.assertFalse(self.view.row_actions.buttons["remove"].isEnabled())
        self.window.set_busy(False)

    def test_relink_from_the_row_asks_for_the_folder(self):
        self.window.project.videos[0].path.unlink()
        self.panel.model.refresh()
        self.hover(0)
        self.prompts.say("choose_folder", None)
        self.view.row_actions.buttons["relink"].click()
        self.assertEqual(len(self.prompts.called("choose_folder")), 1)

    def test_a_double_click_on_a_name_opens_the_details(self):
        self.window.drawer.animation_ms = 0
        index = self.panel.proxy.index(1, 1)
        self.view.doubleClicked.emit(index)
        self.assertTrue(self.window.drawer.is_open())
        self.assertEqual(self.window.selected_keys(), [self.window.project.keys()[1]])

    def test_a_double_click_on_an_editable_cell_edits_instead(self):
        self.window.drawer.animation_ms = 0
        self.panel.model.add_metadata_column("Group")
        column = self.panel.model.column_for_metadata("Group")
        self.view.doubleClicked.emit(self.panel.proxy.index(0, column))
        self.assertFalse(self.window.drawer.is_open())

    def test_the_details_button_follows_the_selection(self):
        self.panel.clear_selection()
        self.panel.flush_selection()
        self.assertFalse(self.panel.details_button.isEnabled())
        self.panel.select_keys(self.window.project.keys()[:1])
        self.panel.flush_selection()
        self.assertTrue(self.panel.details_button.isEnabled())
        opened = []
        self.panel.details_requested.connect(opened.append)
        self.panel.details_button.click()
        self.assertEqual(opened, [self.window.project.keys()[:1]])

    def test_the_context_menu_has_details(self):
        self.panel.select_keys(self.window.project.keys()[:1])
        self.panel.flush_selection()
        menu = self.panel.build_context_menu(self.panel.proxy.index(0, 1))
        opened = []
        self.panel.details_requested.connect(opened.append)
        next(a for a in menu.actions() if a.text() == "Details").trigger()
        self.assertEqual(len(opened), 1)


class DrawerTests(StageFixture):
    def setUp(self):
        super().setUp()
        self.project = self.ready()
        self.window.drawer.animation_ms = 0

    def test_hidden_by_default(self):
        self.assertFalse(self.window.drawer.is_open())
        self.assertFalse(self.window.drawer.isVisibleTo(self.window.body))

    def test_show_details_selects_and_opens(self):
        keys = self.project.keys()
        self.window.show_details(keys[1:2])
        drawer = self.window.drawer
        self.assertTrue(drawer.is_open())
        self.assertTrue(drawer.isVisibleTo(self.window.body))
        self.assertEqual(self.window.selected_keys(), keys[1:2])
        self.assertEqual(
            self.window.context_panel.name_label.text().replace("​", ""),
            self.project.videos[1].name,
        )

    def test_it_overlays_the_stage_instead_of_squeezing_it(self):
        window = self.window
        window.resize(1300, 700)
        window.show()
        self.app.processEvents()
        before = window.stack.width()
        window.drawer.open()
        self.app.processEvents()
        self.assertEqual(window.stack.width(), before)
        rect = window.drawer.geometry()
        self.assertEqual(rect.right() + 1, window.body.width())
        self.assertEqual(rect.height(), window.body.height())
        self.assertLessEqual(rect.width(), window.drawer.WIDTH)

    def test_it_follows_the_window_size(self):
        window = self.window
        window.resize(1300, 700)
        window.show()
        window.drawer.open()
        self.app.processEvents()
        window.resize(1100, 640)
        self.app.processEvents()
        self.assertEqual(window.drawer.geometry().right() + 1, window.body.width())

    def test_close_button_escape_and_toggle(self):
        drawer = self.window.drawer
        drawer.open()
        drawer.close_button.click()
        self.assertFalse(drawer.is_open())
        self.assertFalse(drawer.isVisibleTo(self.window.body))
        drawer.open()
        QTest.keyClick(drawer, Qt.Key.Key_Escape)
        self.assertFalse(drawer.is_open())
        drawer.toggle()
        self.assertTrue(drawer.is_open())
        drawer.toggle()
        self.assertFalse(drawer.is_open())

    def test_it_slides_in_and_out_when_animated(self):
        from PyQt6.QtTest import QTest as T
        window = self.window
        window.resize(1300, 700)
        window.show()
        self.app.processEvents()
        drawer = window.drawer
        drawer.animation_ms = 60
        drawer.open()
        self.assertTrue(drawer.is_open())
        T.qWait(150)
        self.assertEqual(drawer.geometry().right() + 1, window.body.width())
        drawer.close_drawer()
        T.qWait(150)
        self.assertFalse(drawer.isVisible())

    def test_a_closed_drawer_does_not_redraw_on_every_selection(self):
        window = self.window
        panel = window.context_panel
        keys = self.project.keys()
        with patch.object(type(panel), "_show_single") as show:
            window.video_panel.select_keys(keys[:1])
            window.video_panel.flush_selection()
        show.assert_not_called()
        with patch.object(type(panel), "_show_single") as show:
            window.drawer.open()
        show.assert_called_once()

    def test_closing_the_project_closes_the_drawer(self):
        self.window.show_details(self.project.keys()[:1])
        self.window.save()
        self.assertTrue(self.window.close_project())
        self.assertFalse(self.window.drawer.is_open())

    def test_closing_commits_the_notes(self):
        self.window.show_details(self.project.keys()[:1])
        self.window.context_panel.notes_edit.setPlainText("note from the drawer")
        self.window.drawer.close_drawer()
        self.assertEqual(self.project.videos[0].notes, "note from the drawer")

    def test_the_actions_of_the_drawer_reach_the_project(self):
        self.window.show_details(self.project.keys()[:1])
        panel = self.window.context_panel
        panel.verify_button.click()
        self.assertTrue(self.project.videos[0].verified)
        self.prompts.say("confirm", True)
        panel.remove_button.click()
        self.assertEqual(len(self.project.videos), 2)

    def test_it_is_available_from_every_stage(self):
        self.window.go_to_stage("trials")
        self.window.show_details(self.project.keys()[:1])
        self.assertTrue(self.window.drawer.is_open())
        self.assertEqual(self.window.current_stage_id, "trials")


class HostedStageTests(StageFixture):
    def test_setup_stages_are_real_stages(self):
        self.ready()
        window = self.window
        for stage_id in ("arena", "detection", "trials"):
            window.go_to_stage(stage_id)
            self.assertNotIsInstance(window.stage(stage_id), HostedSetupStage)
            self.assertIsNot(window.setup_tab.parentWidget(), window.stage(stage_id).content)

    def test_no_setup_stage_is_hosted_any_more(self):
        for stage_id in ("arena", "detection", "trials"):
            self.assertNotIsInstance(self.window.stage(stage_id), HostedSetupStage)

    def test_run_and_results_host_their_tabs(self):
        self.ready()
        window = self.window
        window.go_to_stage("review")
        self.assertIs(window.run_tab.parentWidget(), window.stage("review").content)
        window.go_to_stage("results")
        self.assertIs(window.results_tab.parentWidget(), window.stage("results").content)

    def test_a_replaced_tab_is_hosted_again(self):
        from PyQt6.QtWidgets import QWidget
        self.ready()
        window = self.window
        window.go_to_stage("results")
        replacement = QWidget()
        old = window.install_tab("results", replacement)
        self.assertIsNot(old, replacement)
        self.assertIs(replacement.parentWidget(), window.stage("results").content)
        self.assertTrue(replacement.isVisibleTo(window.stage("results")))

    def test_parked_tabs_belong_to_the_window(self):
        window = self.window
        for name in ("setup", "run", "results"):
            tab = window.tab_widget(name)
            self.assertIsNotNone(tab.parentWidget(), name)

    def test_the_old_tabs_still_hear_the_hooks(self):
        project = self.ready()
        tab = self.window.setup_tab
        self.assertIs(tab.project, project)
        self.assertIs(tab.workspace, self.window)


class StartupAndTimerTests(StageFixture):
    def test_the_startup_check_builds_and_closes_the_window(self):
        import os
        import subprocess
        import sys
        root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, "-B", "gui_main.py", "--startup-check"], cwd=root,
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen",
                 "BEHAVIORAL_MAZE_ANALYZER_SETTINGS_FILE": str(self.root / "s.ini")},
            capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_no_timer_of_the_shell_fires_after_the_window_is_deleted(self):
        window = self.window
        project = self.ready()
        project.set_notes("changed")      # schedules a navigation refresh
        window.show_message("a message")  # starts the message timer
        window.stage("videos").notes_edit.setPlainText("typing")
        calls = []
        with patch.object(type(window), "refresh_navigation",
                          side_effect=lambda *a: calls.append("nav")), \
                patch.object(type(window), "_message_done",
                             side_effect=lambda *a: calls.append("msg")):
            window._skip_close_prompt = True
            window.close()
            window.deleteLater()
            self.window = None
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            QTest.qWait(80)
        self.assertEqual(calls, [])

    def test_closing_stops_every_timer_of_the_shell(self):
        window = self.window
        self.ready()
        window.show_message("x")
        window.schedule_navigation_refresh()
        window._skip_close_prompt = True
        window.close()
        for timer in (window._autosave_timer, window._start_timer,
                      window._nav_timer, window._message_timer,
                      window.videos_stage._notes_timer,
                      window.context_panel._notes_timer):
            self.assertFalse(timer.isActive())

    def test_a_late_refresh_after_closing_is_harmless(self):
        window = self.window
        self.ready()
        window._skip_close_prompt = True
        window.close()
        window.refresh_navigation()
        window.refresh_footer()
        window.show_message("late")
        window.schedule_navigation_refresh()


class ParadigmCardTests(WorkspaceFixture):
    def test_long_descriptions_wrap_and_the_card_grows(self):
        long = "Spontaneous alternation measured as the share of arm entries " * 2
        card = ParadigmCard("Y-Maze", long, "maze-y")
        self.addCleanup(card.deleteLater)
        wide = card.heightForWidth(600)
        narrow = card.heightForWidth(160)
        self.assertTrue(card.hasHeightForWidth())
        self.assertGreater(narrow, wide)
        self.assertGreaterEqual(wide, card.MIN_HEIGHT)

    def test_short_descriptions_keep_the_minimum_height(self):
        card = ParadigmCard("Y-Maze", "Short", "maze-y")
        self.addCleanup(card.deleteLater)
        self.assertEqual(card.heightForWidth(300), card.MIN_HEIGHT)

    def test_the_picker_grows_when_it_is_narrow(self):
        picker = ParadigmPicker()
        self.addCleanup(picker.deleteLater)
        picker.resize(1200, 10)
        picker.show()
        self.app.processEvents()
        wide = picker.layout().totalHeightForWidth(1200)
        picker.resize(260, 10)
        self.app.processEvents()
        narrow = picker.layout().totalHeightForWidth(260)
        self.assertGreater(narrow, wide)
        for name in picker.names():
            card = picker.card(name)
            self.assertGreaterEqual(card.height(), card.heightForWidth(card.width()) - 1)


class GalleryTests(WorkspaceFixture):
    def test_every_workspace_page_renders(self):
        import sys
        import tempfile
        root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(root / "tools"))
        self.addCleanup(sys.path.remove, str(root / "tools"))
        import ui_gallery
        from gui.workspace import theme
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as out:
            try:
                paths = ui_gallery.render_workspace(
                    out, dark=True, sizes=[(1100, 700)], app=self.app)
            finally:
                theme.reset_theme(self.app)
            self.assertEqual(len(paths), len(ui_gallery.WORKSPACE_PAGES))
            for path in paths:
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 1000)
            self.assertTrue(any(p.name == "ws_videos_drawer_1100x700_dark.png" for p in paths))


if __name__ == "__main__":
    unittest.main()
