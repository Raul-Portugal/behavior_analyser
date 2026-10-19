"""Project files: save/load, importers, batch-plan export and autosave."""
import json
import os
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from checkpoint_manager import CheckpointManager
from core.batch_plan_keys import build_plan_by_key
from core.models import BatchSettings
from core.project import (SCHEMA_VERSION, STATUS_DONE, STATUS_FAILED,
                          STATUS_INTERRUPTED, Project, ProjectError)
from core.project_io import (ProjectAutosaver, autosave_path_for,
                             discard_autosave, export_batch_plan,
                             is_project_file, load_project,
                             project_from_batch_plan, project_from_run_folder,
                             project_from_template, recovery_available,
                             save_project, with_project_extension)
from core.run_outputs import (output_base_names, write_run_manifest,
                              write_run_plan)
from core.sample_sheet import SampleMetadata
from tests.project_support import ProjectFixture, make_settings


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _project_with_videos(fixture) -> Project:
    project = Project(name="Study")
    project.set_template(make_settings())
    project.add_videos([
        fixture.write_video("videos/a.mp4"),
        fixture.write_video("videos/b.mp4", b"bb"),
    ])
    project.customize(project.videos[1].key, make_settings(offset=4))
    project.add_metadata_column("group")
    project.set_metadata_value(project.videos[0].key, "group", "ctl")
    project.notes = "notes"
    return project


class SaveLoadTests(ProjectFixture):
    def test_extension_is_added_and_final_path_returned(self):
        project = _project_with_videos(self)
        target = save_project(project, self.root / "proj" / "study")
        self.assertEqual(target, (self.root / "proj" / "study.bmaproj").resolve())
        self.assertTrue(target.is_file())
        self.assertEqual(with_project_extension(target), target)
        self.assertEqual(
            with_project_extension(Path("x.json")), Path("x.json.bmaproj")
        )

    def test_file_is_atomic_json_with_relative_paths(self):
        project = _project_with_videos(self)
        target = save_project(project, self.root / "proj" / "study.bmaproj")
        self.assertEqual(os.listdir(target.parent), ["study.bmaproj"])
        data = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(data["type"], "project")
        self.assertEqual(data["schema_version"], SCHEMA_VERSION)
        self.assertEqual(
            [v["path"] for v in data["videos"]],
            ["../videos/a.mp4", "../videos/b.mp4"],
        )
        self.assertEqual(
            [v["key"] for v in data["videos"]],
            ["../videos/a.mp4", "../videos/b.mp4"],
        )
        self.assertNotIn(str(self.root).replace("\\", "/"), json.dumps(data))

    def test_round_trip(self):
        project = _project_with_videos(self)
        target = save_project(project, self.root / "study.bmaproj")
        loaded = load_project(target)
        self.assertEqual(loaded.file_path, target)
        self.assertEqual(loaded.name, "Study")
        self.assertEqual(loaded.notes, "notes")
        self.assertEqual(loaded.keys(), project.keys())
        self.assertEqual(
            [v.path for v in loaded.videos], [v.path for v in project.videos]
        )
        self.assertEqual(loaded.videos[0].metadata, {"group": "ctl"})
        self.assertEqual(loaded.videos[1].settings_source, "custom")
        self.assertEqual(
            loaded.videos[1].settings.to_dict(),
            project.videos[1].settings.to_dict(),
        )
        self.assertEqual(loaded.template_settings.to_dict(),
                         project.template_settings.to_dict())
        self.assertEqual(loaded.validate(), [])

    def test_first_save_makes_keys_relative_and_notifies(self):
        project = _project_with_videos(self)
        self.assertEqual(project.videos[0].key, project.videos[0].path.as_posix())
        calls = []
        project.add_listener(lambda: calls.append(1))
        save_project(project, self.root / "p" / "s.bmaproj")
        self.assertEqual(project.videos[0].key, "../videos/a.mp4")
        self.assertEqual(len(calls), 1)

    def test_save_as_elsewhere_keeps_keys_and_stores_new_relative_paths(self):
        project = _project_with_videos(self)
        save_project(project, self.root / "p" / "s.bmaproj")
        keys = project.keys()
        second = save_project(project, self.root / "deeper" / "x" / "s.bmaproj")
        self.assertEqual(project.keys(), keys)
        data = json.loads(second.read_text(encoding="utf-8"))
        self.assertEqual(data["videos"][0]["path"], "../../videos/a.mp4")
        self.assertEqual(
            [v.path for v in load_project(second).videos],
            [v.path for v in project.videos],
        )

    def test_whole_study_folder_can_be_moved(self):
        project = _project_with_videos(self)
        save_project(project, self.root / "videos" / "s.bmaproj")
        moved = self.root.parent / (self.root.name + "_moved")
        self.addCleanup(shutil.rmtree, moved, True)
        shutil.copytree(self.root, moved)
        loaded = load_project(moved / "videos" / "s.bmaproj")
        self.assertEqual(
            [v.path for v in loaded.videos],
            [(moved / "videos" / "a.mp4").resolve(),
             (moved / "videos" / "b.mp4").resolve()],
        )
        self.assertEqual(loaded.missing_files(), [])

    def test_saving_a_project_restamps_the_app_version(self):
        project = _project_with_videos(self)
        project.app_version = "0.0.1"
        target = save_project(project, self.root / "s.bmaproj")
        self.assertNotEqual(load_project(target).app_version, "0.0.1")

    def test_newer_schema_is_rejected(self):
        project = _project_with_videos(self)
        target = save_project(project, self.root / "s.bmaproj")
        data = json.loads(target.read_text(encoding="utf-8"))
        data["schema_version"] = SCHEMA_VERSION + 1
        target.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ProjectError) as raised:
            load_project(target)
        self.assertIn("newer", str(raised.exception))

    def test_other_files_are_not_projects(self):
        template = self.root / "template.json"
        make_settings().save_to_file(template)
        with self.assertRaises(ProjectError):
            load_project(template)
        broken = self.root / "broken.bmaproj"
        broken.write_text("{nope", encoding="utf-8")
        with self.assertRaises(ProjectError):
            load_project(broken)
        listing = self.root / "list.bmaproj"
        listing.write_text("[1]", encoding="utf-8")
        with self.assertRaises(ProjectError):
            load_project(listing)
        with self.assertRaises(OSError):
            load_project(self.root / "missing.bmaproj")
        self.assertFalse(is_project_file(template))
        self.assertFalse(is_project_file(broken))

    def test_detect_file_type_knows_projects_and_keeps_the_old_answers(self):
        project = _project_with_videos(self)
        target = save_project(project, self.root / "s.bmaproj")
        template = self.root / "template.json"
        make_settings().save_to_file(template)
        plan = self.root / "plan.json"
        BatchSettings.save_batch_plan({"a.mp4": make_settings()}, plan)
        garbage = self.root / "garbage.json"
        garbage.write_text("not json", encoding="utf-8")
        self.assertEqual(BatchSettings.detect_file_type(target), "project")
        self.assertEqual(BatchSettings.detect_file_type(template), "template")
        self.assertEqual(BatchSettings.detect_file_type(plan), "batch_plan")
        self.assertEqual(BatchSettings.detect_file_type(garbage), "unknown")
        self.assertTrue(is_project_file(target))

    def test_template_loader_explains_a_project_file(self):
        target = save_project(_project_with_videos(self), self.root / "s.bmaproj")
        with self.assertRaises(ValueError) as raised:
            BatchSettings.load_from_file(target)
        self.assertIn("project", str(raised.exception).lower())
        with self.assertRaises(ValueError):
            BatchSettings.load_batch_plan(target)


class ImporterTests(ProjectFixture):
    def test_from_template_gives_every_video_the_template(self):
        template = self.root / "tpl.json"
        make_settings(paradigm_id="EPM").save_to_file(template)
        videos = [self.write_video("v/a.mp4"), self.write_video("v/b.mp4")]
        project = project_from_template(
            template, videos, project_file=self.root / "p" / "x"
        )
        self.assertEqual(project.name, "tpl")
        self.assertEqual(project.paradigm, "EPM")
        self.assertEqual(project.keys(), ["../v/a.mp4", "../v/b.mp4"])
        self.assertTrue(all(
            v.settings_source == "template" and v.settings is not None
            for v in project.videos
        ))
        self.assertEqual(project.file_path, (self.root / "p" / "x.bmaproj").resolve())

    def test_from_template_without_videos_and_bad_files(self):
        template = self.root / "tpl.json"
        make_settings().save_to_file(template)
        self.assertEqual(project_from_template(template).videos, [])
        plan = self.root / "plan.json"
        BatchSettings.save_batch_plan({"a.mp4": make_settings()}, plan)
        with self.assertRaises(ProjectError):
            project_from_template(plan)
        with self.assertRaises(ProjectError):
            project_from_template(self.root / "missing.json")

    def _plan(self, entries):
        plan_file = self.root / "plans" / "plan.json"
        plan_file.parent.mkdir(exist_ok=True)
        BatchSettings.save_batch_plan(
            build_plan_by_key(entries, plan_file), plan_file
        )
        return plan_file

    def test_from_batch_plan_uses_its_entries_and_finds_the_template(self):
        a = self.write_video("v/a.mp4")
        b = self.write_video("v/b.mp4")
        c = self.write_video("v/c.mp4")
        plan_file = self._plan({
            a: make_settings(), b: make_settings(offset=6), c: make_settings(),
        })
        project = project_from_batch_plan(plan_file)
        self.assertEqual([v.path for v in project.videos], [a, b, c])
        self.assertEqual(
            [v.settings_source for v in project.videos],
            ["template", "custom", "template"],
        )
        self.assertIsNotNone(project.template_settings)
        self.assertEqual(project.paradigm, "Y-Maze")
        self.assertTrue(all(v.fingerprint for v in project.videos))
        self.assertEqual(project.validate(), [])

    def test_from_batch_plan_with_one_entry_has_no_template(self):
        a = self.write_video("v/a.mp4")
        project = project_from_batch_plan(self._plan({a: make_settings()}))
        self.assertIsNone(project.template_settings)
        self.assertEqual(project.videos[0].settings_source, "custom")

    def test_from_batch_plan_matches_given_videos_like_the_cli(self):
        a = self.write_video("v/a.mp4")
        b = self.write_video("v/b.mp4")
        other = self.write_video("w/zzz.mp4")
        plan_file = self._plan({a: make_settings(), b: make_settings(offset=2)})
        project = project_from_batch_plan(plan_file, [b, other, a])
        self.assertEqual([v.name for v in project.videos],
                         ["b.mp4", "zzz.mp4", "a.mp4"])
        self.assertEqual(project.videos[0].settings.to_dict(),
                         make_settings(offset=2).to_dict())
        self.assertIsNone(project.videos[1].settings)
        self.assertEqual(
            [i.code for i in project.validate() if i.key == project.videos[1].key],
            ["unconfigured"],
        )

    def test_from_old_name_keyed_plan(self):
        a = self.write_video("v/a.mp4")
        plan_file = self.root / "old.json"
        BatchSettings.save_batch_plan({"a.mp4": make_settings()}, plan_file)
        project = project_from_batch_plan(plan_file, [a])
        self.assertIsNotNone(project.videos[0].settings)

    def test_from_batch_plan_rejects_a_template(self):
        template = self.root / "tpl.json"
        make_settings().save_to_file(template)
        with self.assertRaises(ProjectError):
            project_from_batch_plan(template)


class RunFolderTests(ProjectFixture):
    def setUp(self):
        super().setUp()
        self.a = self.write_video("v/a.mp4")
        self.b = self.write_video("v/b.mp4", b"bb")
        self.c = self.write_video("v/c.mp4", b"ccc")
        self.run_dir = self.root / "out" / "analysis_run_1"
        self.run_dir.mkdir(parents=True)
        self.plan = {path: make_settings() for path in (self.a, self.b, self.c)}
        self.maze = SimpleNamespace(
            name="Y-Maze", get_algorithm_parameters=lambda: {"alpha": 3}
        )

    def _write(self, with_plan=True, with_manifest=True):
        if with_plan:
            write_run_plan(self.plan, self.run_dir)
        if with_manifest:
            sheet = self.root / "sheet.csv"
            metadata = SampleMetadata(
                sheet, ("group",),
                {self.a: {"group": "ctl"}, self.b: {"group": "trt"}},
            )
            write_run_manifest(
                self.run_dir, self.maze, "2026-02-01T09:00:00+0000", 3,
                output_base_names(list(self.plan)),
                [(self.a, None), (self.b, None)], [(self.c, "boom")],
                [(self.b, "low coverage")], False,
                sample_metadata=metadata, analysis_plan=self.plan,
                hasher=CheckpointManager(self.root, create_dir=False),
                bin_seconds=30.0,
            )

    def test_a_3x_run_opens_as_a_project(self):
        self._write()
        project = project_from_run_folder(self.run_dir)
        self.assertEqual(project.name, "analysis_run_1")
        self.assertEqual([v.path for v in project.videos], [self.a, self.b, self.c])
        self.assertEqual(
            [v.last_status for v in project.videos],
            [STATUS_DONE, "warning", STATUS_FAILED],
        )
        self.assertEqual(project.paradigm, "Y-Maze")
        self.assertEqual(project.algorithm_parameters, {"alpha": 3})
        self.assertEqual(project.run_options.time_bin_seconds, 30.0)
        self.assertEqual(project.run_options.sample_sheet, self.root / "sheet.csv")
        self.assertEqual(project.videos[0].metadata, {"group": "ctl"})
        self.assertEqual(project.videos[1].metadata, {"group": "trt"})
        self.assertEqual(project.videos[2].metadata, {"group": ""})
        # One shared configuration across the run -> the project template.
        self.assertIsNotNone(project.template_settings)
        self.assertTrue(all(v.settings_source == "template" for v in project.videos))
        self.assertEqual(len(project.runs), 1)
        self.assertEqual(project.videos[0].last_run_dir, self.run_dir)
        # The fingerprint is the one recorded by the run.
        self.assertEqual(
            project.videos[0].fingerprint,
            CheckpointManager.get_video_fingerprint(self.a),
        )

    def test_a_video_edited_after_the_run_is_flagged(self):
        self._write()
        self.a.write_bytes(b"re-encoded")
        project = project_from_run_folder(self.run_dir)
        self.assertIn(
            ("video_changed", project.videos[0].key),
            {(i.code, i.key) for i in project.validate()},
        )

    def test_the_plan_alone_is_enough(self):
        self._write(with_manifest=False)
        project = project_from_run_folder(self.run_dir)
        self.assertEqual(len(project.videos), 3)
        self.assertTrue(all(
            v.last_status == STATUS_INTERRUPTED for v in project.videos
        ))
        self.assertIsNotNone(project.videos[0].settings)

    def test_the_manifest_alone_is_enough(self):
        self._write(with_plan=False)
        project = project_from_run_folder(self.run_dir)
        self.assertEqual(len(project.videos), 3)
        self.assertIsNone(project.videos[0].settings)
        self.assertEqual(project.videos[0].last_status, STATUS_DONE)

    def test_an_empty_folder_is_rejected(self):
        with self.assertRaises(ProjectError):
            project_from_run_folder(self.run_dir)

    def test_the_project_can_be_saved_and_reloaded(self):
        self._write()
        project = project_from_run_folder(
            self.run_dir, project_file=self.run_dir / "run.bmaproj"
        )
        target = save_project(project, self.run_dir / "run.bmaproj")
        loaded = load_project(target)
        self.assertEqual(
            [v.last_status for v in loaded.videos],
            [v.last_status for v in project.videos],
        )
        self.assertEqual(loaded.runs[0].run_dir, self.run_dir)
        self.assertEqual(loaded.videos[0].key, "../../v/a.mp4")
        plan = loaded.analysis_plan()
        self.assertEqual(list(plan.plan), [self.a, self.b, self.c])


class ExportTests(ProjectFixture):
    def test_export_is_a_3x_batch_plan_with_relative_keys(self):
        project = _project_with_videos(self)
        extra = self.write_video("videos/none.mp4")
        project.add_videos([extra])
        project.videos[2].settings = None
        out = self.root / "exports" / "plan.json"
        out.parent.mkdir()
        skipped = export_batch_plan(project, out)
        self.assertEqual(skipped, [extra])
        self.assertEqual(BatchSettings.detect_file_type(out), "batch_plan")
        plan = BatchSettings.load_batch_plan(out)
        self.assertEqual(list(plan), ["../videos/a.mp4", "../videos/b.mp4"])
        self.assertEqual(
            plan["../videos/b.mp4"].to_dict(),
            project.videos[1].settings.to_dict(),
        )

    def test_exported_plan_is_read_by_the_cli(self):
        from core.cli import load_analysis_plan
        project = _project_with_videos(self)
        out = self.root / "plan.json"
        export_batch_plan(project, out)
        loaded = load_analysis_plan(out)
        self.assertEqual(set(loaded), {v.path for v in project.videos})

    def test_export_includes_videos_whose_file_is_gone(self):
        project = _project_with_videos(self)
        project.videos[0].path.unlink()
        out = self.root / "plan.json"
        export_batch_plan(project, out)
        self.assertEqual(len(BatchSettings.load_batch_plan(out)), 2)

    def test_nothing_configured_is_an_error(self):
        project = Project()
        project.add_videos([self.write_video("a.mp4")])
        with self.assertRaises(ProjectError):
            export_batch_plan(project, self.root / "plan.json")
        self.assertFalse((self.root / "plan.json").exists())

    def test_round_trip_through_a_batch_plan(self):
        project = _project_with_videos(self)
        out = self.root / "plan.json"
        export_batch_plan(project, out)
        again = project_from_batch_plan(out)
        self.assertEqual(
            [v.settings.to_dict() for v in again.videos],
            [v.settings.to_dict() for v in project.videos],
        )


class AutosaveTests(ProjectFixture):
    def setUp(self):
        super().setUp()
        self.project = _project_with_videos(self)
        self.path = save_project(self.project, self.root / "s.bmaproj")
        self.clock = FakeClock()
        self.saver = ProjectAutosaver(
            self.project, self.path, debounce_s=2, clock=self.clock
        )
        self.addCleanup(self.saver.close)

    def test_autosave_name(self):
        self.assertEqual(self.saver.autosave_path, self.root / "s.bmaproj.autosave")
        self.assertEqual(autosave_path_for(self.path), self.saver.autosave_path)

    def test_nothing_is_written_when_nothing_changed(self):
        self.clock.advance(60)
        self.assertFalse(self.saver.poll())
        self.assertFalse(self.saver.autosave_path.exists())

    def test_a_change_is_written_after_the_debounce(self):
        self.project.notes = "edited"
        self.project.set_video_notes(self.project.videos[0].key, "n")
        self.assertTrue(self.saver.is_dirty)
        self.clock.advance(1.9)
        self.assertFalse(self.saver.poll())
        self.clock.advance(0.2)
        self.assertTrue(self.saver.poll())
        self.assertFalse(self.saver.is_dirty)
        self.assertTrue(self.saver.autosave_path.is_file())
        self.assertFalse(self.saver.poll())

    def test_each_change_restarts_the_debounce(self):
        key = self.project.videos[0].key
        self.project.set_video_notes(key, "1")
        self.clock.advance(1.5)
        self.project.set_video_notes(key, "2")
        self.clock.advance(1.5)
        self.assertFalse(self.saver.poll())
        self.clock.advance(1.0)
        self.assertTrue(self.saver.poll())

    def test_continuous_editing_cannot_postpone_the_autosave_forever(self):
        saver = ProjectAutosaver(
            self.project, self.path, debounce_s=2, max_delay_s=10,
            clock=self.clock,
        )
        self.addCleanup(saver.close)
        key = self.project.videos[0].key
        for step in range(7):
            self.project.set_video_notes(key, str(step))
            self.clock.advance(1.5)
        self.assertTrue(saver.poll())

    def test_the_autosave_is_a_project_that_can_be_recovered(self):
        self.project.set_video_notes(self.project.videos[0].key, "unsaved work")
        self.assertTrue(self.saver.flush())
        os.utime(self.saver.autosave_path, (
            self.path.stat().st_mtime + 10, self.path.stat().st_mtime + 10
        ))
        self.assertTrue(recovery_available(self.path))
        self.assertTrue(ProjectAutosaver.recovery_available(self.path))
        recovered = load_project(self.path, recover=True)
        self.assertEqual(recovered.videos[0].notes, "unsaved work")
        self.assertEqual(recovered.file_path, self.path)
        self.assertEqual(load_project(self.path).videos[0].notes, "")

    def test_an_older_autosave_is_not_offered(self):
        self.project.set_video_notes(self.project.videos[0].key, "x")
        self.saver.flush()
        old = self.path.stat().st_mtime - 100
        os.utime(self.saver.autosave_path, (old, old))
        self.assertFalse(recovery_available(self.path))
        self.assertFalse(recovery_available(self.root / "none.bmaproj"))

    def test_an_autosave_without_its_project_file_is_offered(self):
        self.project.set_video_notes(self.project.videos[0].key, "x")
        self.saver.flush()
        self.path.unlink()
        self.assertTrue(recovery_available(self.path))

    def test_saving_for_real_removes_the_autosave(self):
        self.project.set_video_notes(self.project.videos[0].key, "x")
        self.saver.flush()
        self.assertTrue(self.saver.autosave_path.exists())
        save_project(self.project, self.path)
        self.assertFalse(self.saver.autosave_path.exists())
        self.assertFalse(recovery_available(self.path))

    def test_saved_clears_pending_changes_and_the_file(self):
        self.project.set_video_notes(self.project.videos[0].key, "x")
        self.saver.flush()
        self.project.set_video_notes(self.project.videos[0].key, "y")
        self.saver.saved()
        self.assertFalse(self.saver.is_dirty)
        self.assertFalse(self.saver.autosave_path.exists())
        self.clock.advance(60)
        self.assertFalse(self.saver.poll())

    def test_a_failed_write_is_retried(self):
        self.project.set_video_notes(self.project.videos[0].key, "x")
        self.clock.advance(5)
        with self.assertLogs("core.project_io", "WARNING"), patch(
            "core.project_io._write_json_atomic",
            side_effect=OSError("disk full"),
        ):
            self.assertFalse(self.saver.poll())
        self.assertIn("disk full", self.saver.last_error)
        self.assertTrue(self.saver.is_dirty)
        self.assertTrue(self.saver.poll())
        self.assertIsNone(self.saver.last_error)

    def test_closed_autosaver_stops_listening(self):
        self.saver.close()
        self.project.set_video_notes(self.project.videos[0].key, "x")
        self.assertFalse(self.saver.is_dirty)

    def test_discard_is_harmless_without_an_autosave(self):
        discard_autosave(self.path)
        discard_autosave(self.root / "nothing.bmaproj")


if __name__ == "__main__":
    unittest.main()
