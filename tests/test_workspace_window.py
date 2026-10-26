"""The workspace window: layout, project session, import routing, closing."""
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from core.project import Project
from core.project_io import ProjectAutosaver, autosave_path_for, load_project
from core.version import APP_NAME
from tests.project_support import make_settings
from tests.workspace_support import HAS_RUNTIME, WorkspaceFixture

if HAS_RUNTIME:
    from PyQt6.QtCore import QCoreApplication, QEvent, Qt
    from PyQt6.QtWidgets import QMainWindow
    from gui.workspace.stages.base import Stage


class LayoutTests(WorkspaceFixture):
    def test_six_real_stages(self):
        window = self.window
        self.assertEqual(
            window.stage_ids(),
            ["videos", "arena", "detection", "trials", "review", "results"],
        )
        for stage in window.stages.values():
            self.assertIsInstance(stage, Stage)
            self.assertIs(stage.workspace, window)
            self.assertIs(window.stack.indexOf(stage) >= 0, True)
        self.assertIs(window.video_panel, window.videos_stage.video_panel)
        self.assertEqual(
            [type(stage).__name__ for stage in window.stages.values()],
            ["VideosStage", "ArenaStage", "DetectionStage", "TrialsStage",
             "ReviewStage", "ResultsStage"],
        )
        # The old Setup / Run / Results tabs are gone: no stage hosts a tab.
        for attribute in ("setup_tab", "run_tab", "results_tab", "install_tab",
                          "show_tab", "tab_widget", "current_tab_name"):
            self.assertFalse(hasattr(window, attribute), attribute)

    def test_the_old_chrome_is_gone(self):
        window = self.window
        for attribute in ("splitter", "toolbar", "tabs", "welcome_card"):
            self.assertFalse(hasattr(window, attribute), attribute)
        self.assertEqual(window.drawer.parentWidget(), window.body)
        self.assertFalse(window.drawer.isVisibleTo(window))

    def test_menus_are_lean_and_complete(self):
        window = self.window
        menus = {a.text(): a.menu() for a in window.menuBar().actions()}
        self.assertEqual(
            list(menus), ["&File", "&Tools", "&Validation", "&Help"]
        )
        file_texts = [a.text() for a in menus["&File"].actions() if a.text()]
        for expected in (
            "&New project...", "&Open project...", "Open &recent", "&Save",
            "Save &as...", "&Import", "&Export batch plan...",
            "&Close project", "Switch to &classic interface", "&Quit",
        ):
            self.assertIn(expected, file_texts)
        import_texts = [a.text() for a in window.import_menu.actions()]
        self.assertEqual(
            import_texts,
            ["&Template...", "&Batch plan...", "&Previous run folder..."],
        )
        self.assertEqual(
            [a.text() for a in window.validation_menu.actions()],
            ["&Score a video manually...", "Tracking s&pot-check...",
             "&Build validation report..."],
        )
        tools = [a.text() for a in window.tools_menu.actions() if a.text()]
        for expected in ("&Align template to videos...", "Go to Review && &run",
                         "&Next stage", "&Previous stage", "Video &details",
                         "&Run analysis", "&Cancel run"):
            self.assertIn(expected, tools)

    def test_the_actions_the_toolbar_had_still_exist(self):
        window = self.window
        for action in ("new_action", "open_action", "save_action",
                       "add_videos_action", "add_folder_action",
                       "remove_action", "run_action", "cancel_action"):
            self.assertTrue(hasattr(window, action), action)
        self.assertTrue(window.new_action.isEnabled())
        self.assertTrue(window.open_action.isEnabled())
        self.assertFalse(window.add_videos_action.isEnabled())

    def test_the_menu_button_of_the_app_bar_lists_the_everyday_actions(self):
        texts = [a.text() for a in self.window.more_menu.actions() if a.text()]
        for expected in ("&New project...", "&Open project...", "&Save",
                         "Save &as...", "Video &details",
                         "Switch to &classic interface"):
            self.assertIn(expected, texts)

    def test_title_without_a_project(self):
        self.assertEqual(self.window.windowTitle(), APP_NAME)
        self.assertFalse(self.window.save_action.isEnabled())
        self.assertFalse(self.window.add_videos_action.isEnabled())

    def test_validation_actions_open_their_dialogs(self):
        cases = (
            (self.window.manual_scoring_action,
             "gui.manual_scoring_dialog.open_manual_scoring"),
            (self.window.position_spotcheck_action,
             "gui.position_spotcheck_dialog.open_position_spotcheck"),
            (self.window.validation_report_action,
             "gui.validation_report_dialog.open_validation_report"),
        )
        for action, target in cases:
            with self.subTest(target=target):
                with patch(target) as opener:
                    action.trigger()
                opener.assert_called_once_with(self.window, self.settings)

    def test_help_actions(self):
        self.window.quickstart_action.trigger()
        self.assertEqual(len(self.prompts.called("open_url")), 1)
        self.window.about_action.trigger()
        (title, text), = self.prompts.called("about")
        self.assertIn(APP_NAME, title)
        self.assertIn("Version", text)

    def test_run_and_cancel_are_placeholders(self):
        window = self.window
        self.new_project_at()
        self.assertFalse(window.run_action.isEnabled())
        self.assertFalse(window.cancel_action.isEnabled())
        fired = []
        window.run_requested.connect(lambda: fired.append("run"))
        window.cancel_requested.connect(lambda: fired.append("cancel"))
        window.set_run_enabled(True)
        self.assertTrue(window.run_action.isEnabled())
        window.run_action.trigger()
        window.set_busy(True, "Running")
        self.assertFalse(window.run_action.isEnabled())
        self.assertTrue(window.cancel_action.isEnabled())
        window.cancel_action.trigger()
        window.set_busy(False)
        self.assertEqual(fired, ["run", "cancel"])


class GeometryTests(WorkspaceFixture):
    def test_the_geometry_is_saved_on_close_and_restored(self):
        window = self.window
        window.resize(1180, 700)
        window.show()
        self.app.processEvents()
        window.close()
        saved = self.settings.workspace_geometry()
        self.assertIsNotNone(saved)
        restored = []
        with patch.object(
            QMainWindow, "restoreGeometry",
            new=lambda widget, geometry: restored.append(geometry) or True,
        ):
            other = type(window)(self.settings, self.prompts)
        self.addCleanup(other.deleteLater)
        self.assertEqual(restored, [saved])

    def test_a_window_that_was_never_shown_saves_nothing(self):
        self.window.close()
        self.assertIsNone(self.settings.workspace_geometry())

    def test_the_default_size_fits_a_1366_by_768_screen(self):
        self.assertLessEqual(self.window.width(), 1366)
        self.assertLessEqual(self.window.height(), 768)


class NewOpenSaveTests(WorkspaceFixture):
    def test_new_project_creates_a_file_and_opens_it(self):
        path = self.new_project_at("my study.bmaproj")
        window = self.window
        self.assertTrue(path.is_file())
        self.assertEqual(window.project.name, "my study")
        self.assertEqual(window.project.file_path, path)
        self.assertEqual(window.windowTitle(), f"my study[*] — {APP_NAME}")
        self.assertFalse(window.isWindowModified())
        self.assertTrue(window.save_action.isEnabled())
        self.assertTrue(window.add_videos_action.isEnabled())
        self.assertEqual(self.settings.last_project(), path)
        self.assertEqual(self.settings.recent_projects(), [path])

    def test_new_project_adds_the_extension(self):
        self.prompts.say("save_file", self.root / "plain")
        self.assertTrue(self.window.new_project())
        self.assertEqual(self.window.project.file_path.suffix, ".bmaproj")

    def test_cancelling_new_project_changes_nothing(self):
        self.prompts.say("save_file", None)
        self.assertFalse(self.window.new_project())
        self.assertIsNone(self.window.project)

    def test_editing_marks_modified_and_save_clears_it(self):
        self.new_project_at()
        window = self.window
        self.add_videos()
        self.assertTrue(window.isWindowModified())
        self.assertTrue(window.has_unsaved_changes)
        self.assertTrue(window.save())
        self.assertFalse(window.isWindowModified())
        saved = load_project(window.project.file_path)
        self.assertEqual(len(saved.videos), 3)
        self.assertFalse(autosave_path_for(window.project.file_path).exists())

    def test_open_project_roundtrip(self):
        path = self.new_project_at()
        self.add_videos()
        self.window.save()
        self.window.close_project()
        self.assertIsNone(self.window.project)
        self.assertTrue(self.window.open_project(path))
        self.assertEqual(
            [v.name for v in self.window.project.videos],
            ["m1.mp4", "m2.mp4", "m3.mp4"],
        )
        self.assertEqual(self.window.video_panel.model.rowCount(), 3)
        self.assertFalse(self.window.isWindowModified())

    def test_open_through_the_dialog(self):
        path = self.new_project_at()
        self.window.close_project()
        self.prompts.say("open_file", path)
        self.assertTrue(self.window.open_project())
        self.assertEqual(self.window.project.file_path, path)

    def test_open_a_missing_file_reports_and_forgets_it(self):
        path = self.new_project_at()
        self.window.close_project()
        path.unlink()
        self.assertFalse(self.window.open_path(path))
        self.assertEqual(len(self.prompts.called("error")), 1)
        self.assertEqual(self.settings.recent_projects(existing_only=False), [])

    def test_open_a_corrupt_project_reports_the_error(self):
        bad = self.root / "bad.bmaproj"
        bad.write_text("{ not json", encoding="utf-8")
        self.assertFalse(self.window.open_path(bad))
        (title, text), = self.prompts.called("error")
        self.assertIn("bad.bmaproj", text)
        self.assertIsNone(self.window.project)

    def test_open_a_newer_schema_is_refused(self):
        path = self.new_project_at()
        data = json.loads(path.read_text(encoding="utf-8"))
        data["schema_version"] = 99
        path.write_text(json.dumps(data), encoding="utf-8")
        self.window.close_project()
        self.assertFalse(self.window.open_path(path))
        self.assertIn("newer", self.prompts.called("error")[0][1])

    def test_save_as_moves_the_project_and_its_autosave(self):
        first = self.new_project_at("a.bmaproj")
        self.add_videos()
        window = self.window
        second = self.root / "sub" / "b.bmaproj"
        self.prompts.say("save_file", second)
        self.assertTrue(window.save_as())
        self.assertEqual(window.project.file_path, second)
        self.assertTrue(second.is_file())
        self.assertEqual(window.autosaver.path, second)
        self.assertEqual(self.settings.last_project(), second)
        self.assertEqual(self.settings.recent_projects(), [second, first])
        # the stored video paths still resolve from the new folder
        reopened = load_project(second)
        self.assertEqual([v.path for v in reopened.videos], self.videos)

    def test_save_failure_is_reported_and_stays_unsaved(self):
        self.new_project_at()
        self.add_videos()
        with patch("gui.workspace.window.save_project",
                   side_effect=OSError("disk full")):
            self.assertFalse(self.window.save())
        self.assertIn("disk full", self.prompts.called("error")[0][1])
        self.assertTrue(self.window.has_unsaved_changes)

    def test_open_recent_menu_lists_projects_and_opens_one(self):
        first = self.new_project_at("a.bmaproj")
        self.window.close_project()
        self.new_project_at("b.bmaproj")
        window = self.window
        window._rebuild_recent_menu()
        names = [a.text() for a in window.recent_menu.actions() if a.text()]
        self.assertEqual(names[:2], ["b.bmaproj", "a.bmaproj"])
        self.assertIn("Clear list", names)
        window.close_project()
        window.recent_menu.actions()[1].trigger()    # a.bmaproj
        self.assertEqual(window.project.file_path, first)
        window.recent_menu.actions()[-1].trigger()   # Clear list
        self.assertEqual(self.settings.recent_projects(), [])

    def test_close_project_forgets_the_last_project_but_keeps_recent(self):
        path = self.new_project_at()
        self.assertTrue(self.window.close_project())
        self.assertIsNone(self.window.project)
        self.assertIsNone(self.settings.last_project())
        self.assertEqual(self.settings.recent_projects(), [path])
        self.assertEqual(self.window.windowTitle(), APP_NAME)
        self.assertFalse(self.window.save_action.isEnabled())

    def test_project_changed_signal(self):
        seen = []
        self.window.project_changed.connect(seen.append)
        self.new_project_at()
        self.window.close_project()
        self.assertEqual([type(p) for p in seen], [Project, type(None)])


class RecoveryTests(WorkspaceFixture):
    def make_newer(self, path):
        stamp = path.stat().st_mtime + 5
        os.utime(autosave_path_for(path), (stamp, stamp))

    def test_recover_opens_the_autosave_as_unsaved(self):
        path = self.new_project_at()
        self.add_videos()
        self.window.autosaver.flush()
        self.make_newer(path)
        self.prompts.say("ask_save_changes", "discard")
        self.assertTrue(self.window.close_project())
        self.assertTrue(autosave_path_for(path).is_file())
        self.prompts.say("ask_recovery", "recover")
        self.assertTrue(self.window.open_path(path))
        self.assertEqual(len(self.window.project.videos), 3)
        self.assertTrue(self.window.has_unsaved_changes)
        self.assertTrue(self.window.isWindowModified())
        # the saved file still has no videos until the user saves
        self.assertEqual(len(load_project(path).videos), 0)
        self.assertTrue(self.window.save())
        self.assertEqual(len(load_project(path).videos), 3)
        self.assertFalse(autosave_path_for(path).exists())

    def test_declining_recovery_opens_the_saved_file(self):
        path = self.new_project_at()
        self.add_videos()
        self.window.autosaver.flush()
        self.make_newer(path)
        self.prompts.say("ask_save_changes", "discard")
        self.window.close_project()
        self.prompts.say("ask_recovery", "saved")
        self.assertTrue(self.window.open_path(path))
        self.assertEqual(len(self.window.project.videos), 0)
        self.assertFalse(self.window.has_unsaved_changes)

    def test_cancelling_the_recovery_question_opens_nothing(self):
        path = self.new_project_at()
        self.add_videos()
        self.window.autosaver.flush()
        self.make_newer(path)
        self.prompts.say("ask_save_changes", "discard")
        self.window.close_project()
        self.prompts.say("ask_recovery", "cancel")
        self.assertFalse(self.window.open_path(path))
        self.assertIsNone(self.window.project)

    def test_the_autosave_timer_writes_the_autosave(self):
        path = self.new_project_at()
        window = self.window
        window.autosaver.debounce_s = 0.0
        self.add_videos()
        window._poll_autosave()
        self.assertTrue(autosave_path_for(path).is_file())


class StartupTests(WorkspaceFixture):
    def test_offers_to_reopen_the_last_project(self):
        path = self.new_project_at()
        self.window.close_project()
        self.settings.add_recent_project(path)     # last_project again
        self.prompts.say("confirm", True)
        self.assertTrue(self.window.start())
        self.assertEqual(self.window.project.file_path, path)
        (title, text), = self.prompts.called("confirm")
        self.assertIn("Reopen", title)

    def test_declining_leaves_the_window_empty(self):
        path = self.new_project_at()
        self.window.close_project()
        self.settings.add_recent_project(path)
        self.prompts.say("confirm", False)
        self.assertFalse(self.window.start())
        self.assertIsNone(self.window.project)

    def test_no_last_project_asks_nothing(self):
        self.assertFalse(self.window.start())
        self.assertEqual(self.prompts.calls, [])

    def test_a_path_on_the_command_line_opens(self):
        path = self.new_project_at()
        self.window.close_project()
        self.assertTrue(self.window.start(path))
        self.assertEqual(self.window.project.file_path, path)

    def test_schedule_start_runs_from_the_event_loop(self):
        path = self.new_project_at()
        self.window.close_project()
        self.window.schedule_start(path)
        self.assertIsNone(self.window.project)
        self.app.processEvents()
        deadline = 50
        while self.window.project is None and deadline:
            QCoreApplication.processEvents()
            deadline -= 1
        self.assertEqual(self.window.project.file_path, path)


class CloseTests(WorkspaceFixture):
    def test_close_with_no_changes_does_not_ask(self):
        self.new_project_at()
        self.window.close()
        self.assertEqual(self.prompts.called("ask_save_changes"), [])

    def test_close_asks_to_save_unsaved_changes_and_saves(self):
        path = self.new_project_at()
        self.add_videos()
        self.prompts.say("ask_save_changes", "save")
        self.window._skip_close_prompt = False
        self.window.close()
        self.assertEqual(len(self.prompts.called("ask_save_changes")), 1)
        self.assertEqual(len(load_project(path).videos), 3)
        self.assertTrue(self.window._closed)

    def test_cancel_keeps_the_window_open(self):
        self.new_project_at()
        self.add_videos()
        self.prompts.say("ask_save_changes", "cancel")
        self.window.close()
        self.assertFalse(self.window._closed)
        self.assertTrue(self.window.has_unsaved_changes)

    def test_discard_closes_without_saving_but_keeps_a_recoverable_copy(self):
        path = self.new_project_at()
        self.add_videos()
        self.prompts.say("ask_save_changes", "discard")
        self.window.close()
        self.assertTrue(self.window._closed)
        self.assertEqual(len(load_project(path).videos), 0)
        self.assertTrue(autosave_path_for(path).is_file())

    def test_a_failed_save_cancels_the_close(self):
        self.new_project_at()
        self.add_videos()
        self.prompts.say("ask_save_changes", "save")
        with patch("gui.workspace.window.save_project",
                   side_effect=OSError("nope")):
            self.window.close()
        self.assertFalse(self.window._closed)

    def test_opening_another_project_asks_too(self):
        first = self.new_project_at("a.bmaproj")
        self.window.close_project()
        self.new_project_at("b.bmaproj")
        self.add_videos()
        self.prompts.say("ask_save_changes", "cancel")
        self.assertFalse(self.window.open_path(first))
        self.assertEqual(self.window.project.file_path, self.root / "b.bmaproj")

    def test_close_flushes_the_autosave(self):
        path = self.new_project_at()
        self.add_videos()
        self.prompts.say("ask_save_changes", "discard")
        with patch.object(ProjectAutosaver, "flush", autospec=True,
                          side_effect=ProjectAutosaver.flush) as flush:
            self.window.close()
        flush.assert_called()
        self.assertTrue(autosave_path_for(path).is_file())

    def test_a_close_guard_can_veto(self):
        self.new_project_at()
        allow = []
        self.window.register_close_guard(lambda: bool(allow))
        self.window.close()
        self.assertFalse(self.window._closed)
        allow.append(1)
        self.window.close()
        self.assertTrue(self.window._closed)

    def test_close_project_is_blocked_while_busy(self):
        self.new_project_at()
        self.window.set_busy(True)
        self.assertFalse(self.window.close_project())
        self.assertIsNotNone(self.window.project)
        self.window.set_busy(False)


class ImportRoutingTests(WorkspaceFixture):
    def write_template(self, name="tpl.json"):
        path = self.root / name
        make_settings().save_to_file(path)
        return path

    def write_plan(self, videos, name="plan.json"):
        from core.batch_plan_keys import build_plan_by_key
        path = self.root / name
        plan = {v: make_settings() for v in videos}
        path.write_text(json.dumps({
            "type": "batch_plan",
            "plan": {k: s.to_dict()
                     for k, s in build_plan_by_key(plan, path).items()},
        }), encoding="utf-8")
        return path

    def test_open_routes_a_template_to_an_import_offer(self):
        template = self.write_template()
        target = self.root / "from_template.bmaproj"
        self.prompts.say("confirm", True)
        self.prompts.say("save_file", target)
        self.assertTrue(self.window.open_path(template))
        (title, text), = self.prompts.called("confirm")
        self.assertIn("a saved template", text)
        project = self.window.project
        self.assertEqual(project.file_path, target)
        self.assertIsNotNone(project.template_settings)
        self.assertEqual(project.paradigm, "Y-Maze")
        self.assertTrue(target.is_file())

    def test_open_routes_a_batch_plan_to_an_import_offer(self):
        plan = self.write_plan(self.videos)
        target = self.root / "from_plan.bmaproj"
        self.prompts.say("confirm", True)
        self.prompts.say("save_file", target)
        self.assertTrue(self.window.open_path(plan))
        self.assertIn("a batch plan", self.prompts.called("confirm")[0][1])
        self.assertEqual(len(self.window.project.videos), 3)
        self.assertTrue(all(v.settings for v in self.window.project.videos))

    def test_declining_the_import_offer_opens_nothing(self):
        template = self.write_template()
        self.prompts.say("confirm", False)
        self.assertFalse(self.window.open_path(template))
        self.assertIsNone(self.window.project)
        self.assertEqual(self.prompts.called("save_file"), [])

    def test_a_project_file_with_a_json_name_opens_as_project(self):
        path = self.new_project_at()
        renamed = path.with_suffix(".json")
        path.rename(renamed)
        self.window.close_project()
        self.assertTrue(self.window.open_path(renamed))
        self.assertEqual(self.window.project.file_path, renamed)

    def test_unknown_file_is_reported(self):
        junk = self.root / "junk.txt"
        junk.write_text("hello", encoding="utf-8")
        self.assertFalse(self.window.open_path(junk))
        self.assertIn("not a project", self.prompts.called("error")[0][1])

    def test_import_menu_template(self):
        template = self.write_template()
        target = self.root / "t.bmaproj"
        self.prompts.say("open_file", template)
        self.prompts.say("save_file", target)
        self.window.import_template_action.trigger()
        self.assertEqual(self.window.project.file_path, target)
        self.assertIn(template, self.settings.recent_files())

    def test_import_menu_batch_plan(self):
        plan = self.write_plan(self.videos[:2])
        target = self.root / "p.bmaproj"
        self.prompts.say("open_file", plan)
        self.prompts.say("save_file", target)
        self.window.import_plan_action.trigger()
        self.assertEqual(len(self.window.project.videos), 2)

    def test_import_run_folder(self):
        from core.run_outputs import write_run_plan
        run_dir = self.root / "analysis_run_1"
        run_dir.mkdir()
        plan = {self.videos[0]: make_settings(), self.videos[1]: make_settings()}
        write_run_plan(plan, run_dir)
        target = self.root / "r.bmaproj"
        self.prompts.say("choose_folder", run_dir)
        self.prompts.say("save_file", target)
        self.window.import_run_action.trigger()
        self.assertEqual(len(self.window.project.videos), 2)

    def test_a_broken_template_is_reported_and_nothing_opens(self):
        bad = self.root / "bad.json"
        bad.write_text(json.dumps({"detection": 1}), encoding="utf-8")
        self.prompts.say("open_file", bad)
        self.prompts.say("save_file", self.root / "x.bmaproj")
        self.window.import_template_action.trigger()
        self.assertEqual(len(self.prompts.called("error")), 1)
        self.assertIsNone(self.window.project)

    def test_import_asks_to_save_the_open_project_first(self):
        self.new_project_at()
        self.add_videos()
        template = self.write_template()
        self.prompts.say("open_file", template)
        self.prompts.say("save_file", self.root / "t.bmaproj")
        self.prompts.say("ask_save_changes", "cancel")
        self.window.import_template_action.trigger()
        self.assertEqual(self.window.project.file_path, self.root / "study.bmaproj")

    def test_export_batch_plan(self):
        self.new_project_at()
        self.add_videos()
        window = self.window
        window.project.set_template(make_settings())
        target = self.root / "out_plan.json"
        self.prompts.say("save_file", target)
        self.assertTrue(window.export_plan())
        data = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(data["type"], "batch_plan")
        self.assertEqual(len(data["plan"]), 3)

    def test_export_with_no_configured_video_warns(self):
        self.new_project_at()
        self.add_videos()
        self.prompts.say("save_file", self.root / "empty_plan.json")
        self.assertFalse(self.window.export_plan())
        self.assertEqual(len(self.prompts.called("warn")), 1)


class DropTests(WorkspaceFixture):
    def test_dropping_a_project_opens_it(self):
        path = self.new_project_at()
        self.window.close_project()
        self.window.handle_dropped_paths([path])
        self.assertEqual(self.window.project.file_path, path)

    def test_dropping_videos_adds_them(self):
        self.new_project_at()
        self.window.handle_dropped_paths(self.videos[:2])
        self.assertEqual(len(self.window.project.videos), 2)

    def test_dropping_a_folder_adds_its_videos(self):
        self.new_project_at()
        self.window.handle_dropped_paths([self.root / "videos"])
        self.assertEqual(len(self.window.project.videos), 3)

    def test_dropping_a_template_offers_an_import(self):
        template = self.root / "tpl.json"
        make_settings().save_to_file(template)
        self.prompts.say("confirm", False)
        self.window.handle_dropped_paths([template])
        self.assertEqual(len(self.prompts.called("confirm")), 1)

    def test_videos_without_a_project_offer_a_new_one(self):
        self.prompts.say("confirm", True)
        self.prompts.say("save_file", self.root / "dropped.bmaproj")
        self.window.handle_dropped_paths(self.videos)
        self.assertEqual(len(self.window.project.videos), 3)

    def test_videos_without_a_project_can_be_declined(self):
        self.prompts.say("confirm", False)
        self.window.handle_dropped_paths(self.videos)
        self.assertIsNone(self.window.project)

    def test_unusable_drop_is_explained(self):
        self.new_project_at()
        junk = self.root / "notes.txt"
        junk.write_text("x", encoding="utf-8")
        self.window.handle_dropped_paths([junk])
        self.assertEqual(len(self.prompts.called("info")), 1)

    def test_drops_are_ignored_while_busy(self):
        self.new_project_at()
        self.window.set_busy(True)
        self.window.handle_dropped_paths(self.videos)
        self.assertEqual(self.window.project.videos, [])
        self.window.set_busy(False)

    def test_a_real_drop_event_reaches_the_window(self):
        from PyQt6.QtCore import QMimeData, QPoint, QPointF, QUrl
        from PyQt6.QtGui import QDropEvent
        self.new_project_at()
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(self.videos[0]))])
        event = QDropEvent(
            QPointF(QPoint(5, 5)), Qt.DropAction.CopyAction, mime,
            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
        )
        self.window.dropEvent(event)
        self.assertEqual(len(self.window.project.videos), 1)


class ParadigmTests(WorkspaceFixture):
    def picker(self):
        return self.window.videos_stage.picker

    def test_the_picker_comes_first_until_a_paradigm_is_chosen(self):
        from gui.workspace.stages.videos_stage import PAGE_PARADIGM
        self.new_project_at()
        stage = self.window.videos_stage
        self.assertEqual(stage.page, PAGE_PARADIGM)
        self.assertIsNone(self.picker().selected())
        self.assertEqual(stage.header.title(), "What test are these videos from?")
        self.assertTrue(self.picker().isEnabled())

    def test_choosing_a_paradigm_sets_it_without_asking(self):
        from gui.workspace.stages.videos_stage import PAGE_VIDEOS
        self.new_project_at()
        self.add_videos()
        self.assertTrue(self.window.change_paradigm("Elevated Plus Maze"))
        self.assertEqual(self.window.project.paradigm, "Elevated Plus Maze")
        self.assertEqual(self.prompts.called("confirm"), [])
        self.assertEqual(self.window.videos_stage.page, PAGE_VIDEOS)
        self.assertEqual(
            self.window.videos_stage.summary.name_label.text(),
            "Elevated Plus Maze",
        )
        self.assertEqual(self.settings.last_paradigm(), "Elevated Plus Maze")

    def test_clicking_a_card_triggers_the_change(self):
        self.new_project_at()
        self.picker().card("Y-Maze").click()
        self.assertEqual(self.window.project.paradigm, "Y-Maze")

    def test_changing_with_configured_videos_confirms_and_clears(self):
        self.new_project_at()
        window = self.window
        window.project.set_template(make_settings())
        self.add_videos()
        self.assertTrue(all(v.settings for v in window.project.videos))
        self.prompts.say("confirm", True)
        self.assertTrue(window.change_paradigm("Elevated Plus Maze"))
        project = window.project
        self.assertEqual(project.paradigm, "Elevated Plus Maze")
        self.assertIsNone(project.template_settings)
        self.assertTrue(all(v.settings is None for v in project.videos))
        text = self.prompts.called("confirm")[0][1]
        self.assertIn("Elevated Plus Maze", text)

    def test_declining_keeps_everything_and_resets_the_picker(self):
        self.new_project_at()
        window = self.window
        window.project.set_template(make_settings())
        self.add_videos()
        self.prompts.say("confirm", False)
        self.assertFalse(window.change_paradigm("Elevated Plus Maze"))
        self.assertEqual(window.project.paradigm, "Y-Maze")
        self.assertTrue(all(v.settings for v in window.project.videos))
        self.assertEqual(self.picker().selected(), "Y-Maze")

    def test_choosing_the_same_paradigm_asks_nothing(self):
        self.new_project_at()
        window = self.window
        window.project.set_template(make_settings())
        self.assertTrue(window.change_paradigm("Y-Maze"))
        self.assertEqual(self.prompts.called("confirm"), [])
        self.assertIsNotNone(window.project.template_settings)

    def test_the_change_button_shows_the_picker_again(self):
        from gui.workspace.stages.videos_stage import PAGE_PARADIGM, PAGE_VIDEOS
        self.new_project_at()
        window = self.window
        window.change_paradigm("Y-Maze")
        stage = window.videos_stage
        self.assertEqual(stage.page, PAGE_VIDEOS)
        stage.summary.change_button.click()
        self.assertEqual(stage.page, PAGE_PARADIGM)
        self.assertEqual(self.picker().selected(), "Y-Maze")
        stage.keep_button.click()
        self.assertEqual(stage.page, PAGE_VIDEOS)

    def test_the_change_button_is_locked_while_a_run_is_in_progress(self):
        self.new_project_at()
        window = self.window
        window.change_paradigm("Y-Maze")
        window.set_busy(True, "Running")
        self.assertFalse(window.videos_stage.summary.change_button.isEnabled())
        self.assertFalse(window.change_paradigm("Elevated Plus Maze"))
        window.set_busy(False)
        self.assertTrue(window.videos_stage.summary.change_button.isEnabled())


class ProjectSetParadigmTests(unittest.TestCase):
    def test_first_choice_clears_nothing(self):
        project = Project()
        self.assertEqual(project.set_paradigm("Y-Maze"), [])
        self.assertEqual(project.paradigm, "Y-Maze")

    def test_changing_clears_settings_and_verification(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "a.mp4"
            video.write_bytes(b"x")
            project = Project()
            project.set_template(make_settings())
            project.add_videos([video])
            project.set_verified(project.keys(), True)
            self.assertEqual(
                project.set_paradigm("Freestyle / Open Field"), project.keys()
            )
            entry = project.videos[0]
            self.assertIsNone(entry.settings)
            self.assertFalse(entry.verified)
            self.assertIsNone(entry.fingerprint)
            self.assertEqual(entry.last_status, "new")
            self.assertIsNone(project.template_settings)
            self.assertEqual(project.algorithm_parameters, {})

    def test_notifies_listeners(self):
        project = Project()
        calls = []
        project.add_listener(lambda: calls.append(1))
        project.set_paradigm("Y-Maze")
        project.set_paradigm("Y-Maze")
        self.assertEqual(calls, [1])


class StageHookTests(WorkspaceFixture):
    """The window tells every stage about the project and the selection."""

    def watch(self, stage, name):
        calls = []
        original = getattr(stage, name)

        def spy(*args):
            calls.append(args)
            return original(*args)

        patcher = patch.object(stage, name, spy)
        patcher.start()
        self.addCleanup(patcher.stop)
        return calls

    def test_the_stages_hear_about_the_project_and_the_selection(self):
        window = self.window
        opened = self.watch(window.stage("review"), "on_project_changed")
        modified = self.watch(window.stage("results"), "on_project_modified")
        selected = self.watch(window.stage("results"), "on_selection_changed")
        self.new_project_at()
        self.add_videos()
        window.change_paradigm("Y-Maze")
        panel = window.video_panel
        key = panel.project().keys()[0]
        panel.select_keys([key])
        panel.flush_selection()
        self.assertEqual([call[0].name for call in opened], ["study"])
        self.assertTrue(modified)
        self.assertIn(([key],), selected)

    def test_a_stage_that_needs_closing_is_shut_down_with_the_window(self):
        window = self.window
        closed = []
        for stage_id in ("review", "results", "trials", "detection", "arena"):
            stage = window.stage(stage_id)
            self.assertTrue(hasattr(stage, "shutdown"), stage_id)
            patcher = patch.object(
                stage, "shutdown", lambda stage_id=stage_id: closed.append(stage_id)
            )
            patcher.start()
            self.addCleanup(patcher.stop)
        window.close()
        self.assertEqual(
            sorted(closed), ["arena", "detection", "results", "review", "trials"]
        )


class ClassicSwitchTests(WorkspaceFixture):
    def test_switch_opens_the_classic_window_and_closes_the_workspace(self):
        from gui import main_window
        from gui.workspace import window as window_module
        created = []

        class FakeClassic:
            def __init__(self, settings):
                created.append(self)
                self.shown = False

            def setAttribute(self, *args):
                pass

            def show(self):
                self.shown = True

            class _Signal:
                def connect(self, slot):
                    pass
            destroyed = _Signal()

        with patch.object(main_window, "MainWindow", FakeClassic):
            self.assertTrue(self.window.switch_to_classic())
        self.assertEqual(len(created), 1)
        self.assertTrue(created[0].shown)
        self.assertTrue(self.window._closed)
        window_module._open_classic_windows.clear()

    def test_switch_is_cancelled_by_the_unsaved_changes_prompt(self):
        from gui import main_window
        self.new_project_at()
        self.add_videos()
        self.prompts.say("ask_save_changes", "cancel")
        with patch.object(main_window, "MainWindow") as classic:
            self.assertFalse(self.window.switch_to_classic())
        classic.assert_not_called()
        self.assertFalse(self.window._closed)


class NoTimersAfterDeletionTests(WorkspaceFixture):
    def test_nothing_fires_after_the_window_is_deleted(self):
        window = self.window
        window.AUTOSAVE_INTERVAL_MS = 5
        self.new_project_at()
        calls = []
        with patch.object(ProjectAutosaver, "poll",
                          side_effect=lambda self=None: calls.append(1)):
            window._autosave_timer.setInterval(5)
            self.add_videos()
            window.schedule_start(self.root / "never.bmaproj")
            window._skip_close_prompt = True
            window.close()
            window.deleteLater()
            self.window = None
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            before = len(calls)
            from PyQt6.QtTest import QTest
            QTest.qWait(80)
        self.assertEqual(len(calls), before)

    def test_closing_stops_the_timers(self):
        window = self.window
        self.assertTrue(window._autosave_timer.isActive())
        window.close()
        self.assertFalse(window._autosave_timer.isActive())
        self.assertFalse(window.context_panel._notes_timer.isActive())
        self.assertFalse(window.videos_stage._notes_timer.isActive())
        self.assertFalse(window._nav_timer.isActive())
        self.assertFalse(window._message_timer.isActive())

    def test_late_autosave_poll_after_close_is_harmless(self):
        self.new_project_at()
        window = self.window
        window.close()
        window._poll_autosave()    # must not raise


class SaveRoundTripOfStateTests(WorkspaceFixture):
    def test_unsaved_flag_survives_the_autosave(self):
        self.new_project_at()
        self.add_videos()
        self.window.autosaver.flush()
        self.assertFalse(self.window.autosaver.is_dirty)
        self.assertTrue(self.window.has_unsaved_changes)

    def test_save_project_helper_is_used_for_keys(self):
        path = self.new_project_at()
        self.add_videos()
        self.window.save()
        project = self.window.project
        self.assertEqual(project.keys(), load_project(path).keys())


if __name__ == "__main__":
    unittest.main()
