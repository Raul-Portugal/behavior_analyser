"""Project operations: videos, settings, metadata, plan, validation, runs."""
import json
import os
import unittest
from types import SimpleNamespace

from checkpoint_manager import CheckpointManager
from core.project import (SEVERITY_ERROR, STATUS_CONFIGURED, STATUS_DONE,
                          STATUS_FAILED, STATUS_INTERRUPTED, STATUS_NEW,
                          STATUS_WARNING, Project, ProjectError)
from core.run_outputs import (MANIFEST_FILENAME, RUN_PLAN_FILENAME,
                              output_base_names, write_run_manifest,
                              write_run_plan)
from core.sample_sheet import SampleSheetError
from tests.project_support import ProjectFixture, make_settings


class OpsFixture(ProjectFixture):
    def setUp(self):
        super().setUp()
        self.project = Project(name="P")
        self.project.file_path = self.root / "study" / "p.bmaproj"
        self.a = self.write_video("day1/a.mp4")
        self.b = self.write_video("day1/b.mp4", b"bbb")
        self.c = self.write_video("day2/c.mp4", b"cc")

    def keys(self):
        return [video.name for video in self.project.videos]


class AddRemoveMoveTests(OpsFixture):
    def test_add_keeps_order_and_deduplicates(self):
        result = self.project.add_videos([self.b, self.a, self.b])
        self.assertEqual(self.keys(), ["b.mp4", "a.mp4"])
        self.assertEqual(result.duplicates, [self.b])
        again = self.project.add_videos([self.a, self.c])
        self.assertEqual([v.name for v in again.added], ["c.mp4"])
        self.assertEqual(again.duplicates, [self.a])
        self.assertEqual(self.keys(), ["b.mp4", "a.mp4", "c.mp4"])

    def test_duplicates_are_found_through_other_spellings(self):
        self.project.add_videos([self.a])
        spelled = self.a.parent / ".." / "day1" / "a.mp4"
        self.assertEqual(self.project.add_videos([spelled]).added, [])
        self.assertIs(self.project.find(spelled), self.project.videos[0])

    def test_keys_are_relative_to_the_project_file(self):
        self.project.add_videos([self.a])
        self.assertEqual(self.project.videos[0].key, "../day1/a.mp4")

    def test_new_videos_without_template_are_unconfigured(self):
        self.project.add_videos([self.a])
        entry = self.project.videos[0]
        self.assertIsNone(entry.settings)
        self.assertEqual(entry.last_status, STATUS_NEW)

    def test_new_videos_inherit_the_template(self):
        self.project.set_template(make_settings())
        self.project.add_videos([self.a])
        entry = self.project.videos[0]
        self.assertIsNotNone(entry.settings)
        self.assertIsNot(entry.settings, self.project.template_settings)
        self.assertEqual(entry.settings_source, "template")
        self.assertEqual(entry.last_status, STATUS_CONFIGURED)
        self.assertIsNotNone(entry.fingerprint)
        self.assertFalse(entry.verified)

    def test_remove_videos(self):
        self.project.add_videos([self.a, self.b, self.c])
        removed = self.project.remove_videos(
            [self.project.videos[0].key, self.project.videos[2].key, "zz"]
        )
        self.assertEqual([v.name for v in removed], ["a.mp4", "c.mp4"])
        self.assertEqual(self.keys(), ["b.mp4"])

    def test_move_single_and_edges(self):
        self.project.add_videos([self.a, self.b, self.c])
        keys = self.project.keys()
        self.assertTrue(self.project.move_videos([keys[2]], -1))
        self.assertEqual(self.keys(), ["a.mp4", "c.mp4", "b.mp4"])
        self.assertTrue(self.project.move_videos([keys[2]], -5))
        self.assertEqual(self.keys(), ["c.mp4", "a.mp4", "b.mp4"])
        self.assertFalse(self.project.move_videos([keys[2]], -1))
        self.assertFalse(self.project.move_videos([keys[0]], 0))

    def test_move_block_together(self):
        self.project.add_videos([self.a, self.b, self.c])
        keys = self.project.keys()
        self.assertTrue(self.project.move_videos([keys[0], keys[1]], 1))
        self.assertEqual(self.keys(), ["c.mp4", "a.mp4", "b.mp4"])
        self.assertFalse(self.project.move_videos([keys[0], keys[1]], 1))

    def test_move_non_adjacent_selection_keeps_relative_order(self):
        d = self.write_video("day2/d.mp4")
        self.project.add_videos([self.a, self.b, self.c, d])
        keys = self.project.keys()
        self.project.move_videos([keys[1], keys[3]], -1)
        self.assertEqual(self.keys(), ["b.mp4", "a.mp4", "d.mp4", "c.mp4"])

    def test_listeners_hear_changes(self):
        calls = []
        self.project.add_listener(lambda: calls.append(1))
        self.project.add_videos([self.a])
        self.project.add_videos([self.a])  # nothing new: no notification
        self.project.remove_videos(self.project.keys())
        self.assertEqual(len(calls), 2)


class SettingsTests(OpsFixture):
    def setUp(self):
        super().setUp()
        self.project.add_videos([self.a, self.b])
        self.keys_ = self.project.keys()

    def test_set_template_updates_inheriting_and_unconfigured_videos(self):
        self.project.customize(self.keys_[1], make_settings(offset=9))
        updated = self.project.set_template(make_settings())
        self.assertEqual(updated, [self.keys_[0]])
        self.assertEqual(self.project.videos[0].last_status, STATUS_CONFIGURED)
        self.assertEqual(self.project.videos[1].settings_source, "custom")
        self.assertEqual(self.project.paradigm, "Y-Maze")

    def test_set_template_can_leave_videos_alone(self):
        self.assertEqual(
            self.project.set_template(make_settings(), update_inheriting=False),
            [],
        )
        self.assertIsNone(self.project.videos[0].settings)

    def test_apply_template_requires_a_template(self):
        with self.assertRaises(ProjectError):
            self.project.apply_template_to([self.keys_[0]])

    def test_apply_template_resets_custom_videos(self):
        self.project.set_template(make_settings())
        self.project.customize(self.keys_[0], make_settings(offset=5), "aligned",
                               verified=True)
        entry = self.project.videos[0]
        self.assertEqual(entry.settings_source, "aligned")
        self.assertTrue(entry.verified)
        self.assertEqual(self.project.apply_template_to([self.keys_[0]]),
                         [self.keys_[0]])
        self.assertEqual(entry.settings_source, "template")
        self.assertFalse(entry.verified)
        self.assertEqual(
            entry.settings.to_dict(), self.project.template_settings.to_dict()
        )

    def test_customize_rejects_template_source_and_unknown_keys(self):
        with self.assertRaises(ProjectError):
            self.project.customize(self.keys_[0], make_settings(), "template")
        with self.assertRaises(KeyError):
            self.project.customize("nope", make_settings())

    def test_changed_settings_reset_status_and_verification(self):
        self.project.set_template(make_settings())
        entry = self.project.videos[0]
        entry.last_status = STATUS_DONE
        entry.verified = True
        self.project.customize(self.keys_[0], make_settings(offset=2))
        self.assertEqual(entry.last_status, STATUS_CONFIGURED)
        self.assertFalse(entry.verified)

    def test_identical_settings_keep_status_and_verification(self):
        self.project.set_template(make_settings())
        entry = self.project.videos[0]
        entry.last_status = STATUS_DONE
        entry.verified = True
        self.project.apply_template_to([self.keys_[0]])
        self.assertEqual(entry.last_status, STATUS_DONE)
        self.assertTrue(entry.verified)

    def test_set_verified_and_accept_changes(self):
        self.project.set_template(make_settings())
        self.project.set_verified([self.keys_[0]])
        self.assertTrue(self.project.videos[0].verified)
        self.a.write_bytes(b"different size")
        self.assertEqual(self.project.accept_video_changes(), [self.keys_[0]])
        self.assertEqual(self.project.accept_video_changes(), [])


class MetadataTests(OpsFixture):
    def setUp(self):
        super().setUp()
        self.project.add_videos([self.a, self.b, self.c])

    def _sheet(self, text):
        path = self.root / "sheet.csv"
        path.write_text(text, encoding="utf-8")
        return path

    def test_sheet_fills_matched_videos_and_reports_the_rest(self):
        sheet = self._sheet(
            "video,animal_id,group\n"
            "day1/a.mp4,A1,ctl\n"
            "day1/b.mp4,B1,trt\n"
            "day1/zzz.mp4,Z,x\n"
        )
        self.project.videos[0].metadata["custom"] = "keep"
        match = self.project.set_metadata_from_sheet(sheet)
        self.assertEqual(
            self.project.videos[0].metadata,
            {"custom": "keep", "animal_id": "A1", "group": "ctl"},
        )
        self.assertEqual(self.project.videos[1].metadata["group"], "trt")
        self.assertEqual(self.project.videos[2].metadata, {})
        self.assertEqual([v for v, _ in match.unmatched], [self.c])
        self.assertEqual(match.unused_rows, ["day1/zzz.mp4"])
        self.assertEqual(self.project.run_options.sample_sheet, sheet.resolve())

    def test_clearing_the_sheet_forgets_it_and_every_metadata_column(self):
        sheet = self._sheet("video,group\nday1/a.mp4,ctl\nday1/b.mp4,trt\n")
        self.project.set_metadata_from_sheet(sheet)
        self.project.add_metadata_column("note")
        calls = []
        self.project.add_listener(lambda: calls.append(1))
        self.assertTrue(self.project.clear_sample_sheet())
        self.assertIsNone(self.project.run_options.sample_sheet)
        self.assertEqual(self.project.metadata_columns(), [])
        self.assertIsNone(self.project.sample_metadata())
        self.assertEqual(len(calls), 1)
        # Nothing left to clear: no change and no notification.
        self.assertFalse(self.project.clear_sample_sheet())
        self.assertEqual(len(calls), 1)

    def test_ambiguous_videos_are_reported_not_guessed(self):
        other = self.write_video("day3/a.mp4")
        self.project.add_videos([other])
        sheet = self._sheet("video,group\na.mp4,ctl\nb.mp4,trt\nc.mp4,trt\n")
        match = self.project.set_metadata_from_sheet(sheet)
        self.assertEqual({v for v, _ in match.ambiguous}, {self.a, other})
        self.assertEqual(self.project.videos[0].metadata, {})
        self.assertEqual(self.project.videos[1].metadata, {"group": "trt"})

    def test_bad_sheet_raises(self):
        with self.assertRaises(SampleSheetError):
            self.project.set_metadata_from_sheet(self._sheet("nothing\n"))

    def test_manual_columns_values_and_notes(self):
        self.assertTrue(self.project.add_metadata_column(" group "))
        self.assertFalse(self.project.add_metadata_column("group"))
        key = self.project.videos[0].key
        self.project.set_metadata_value(key, "group", "ctl")
        self.project.set_video_notes(key, "n")
        self.assertEqual(self.project.metadata_columns(), ["group"])
        self.assertEqual(self.project.videos[0].metadata["group"], "ctl")
        self.assertEqual(self.project.videos[0].notes, "n")
        self.assertTrue(self.project.remove_metadata_column("group"))
        self.assertEqual(self.project.metadata_columns(), [])
        with self.assertRaises(ProjectError):
            self.project.add_metadata_column("  ")

    def test_sample_metadata_for_a_run(self):
        self.assertIsNone(self.project.sample_metadata())
        key = self.project.videos[0].key
        self.project.add_metadata_column("group")
        self.project.set_metadata_value(key, "group", "ctl")
        metadata = self.project.sample_metadata()
        self.assertEqual(metadata.columns, ("group",))
        self.assertEqual(metadata.values_for(self.a), {"group": "ctl"})
        self.assertEqual(metadata.values_for(self.b), {"group": ""})


class PlanAndValidationTests(OpsFixture):
    def test_analysis_plan_splits_configured_unconfigured_and_missing(self):
        self.project.add_videos([self.a, self.b, self.c])
        keys = self.project.keys()
        self.project.customize(keys[0], make_settings(paradigm_id=None))
        self.project.customize(keys[2], make_settings(offset=3))
        self.c.unlink()
        self.project.paradigm = "EPM"
        self.project.algorithm_parameters = {"alpha": 2}
        plan = self.project.analysis_plan()
        self.assertEqual(list(plan.plan), [self.a])
        self.assertEqual(plan.unconfigured, [self.b])
        self.assertEqual(plan.missing, [self.c])
        self.assertEqual(plan.keys, {self.a: keys[0]})
        settings = plan.plan[self.a]
        self.assertIsNot(settings, self.project.videos[0].settings)
        self.assertEqual(settings.paradigm_id, "EPM")
        self.assertEqual(settings.algorithm_parameters, {"alpha": 2})
        self.assertIsNone(self.project.videos[0].settings.paradigm_id)

    def test_analysis_plan_for_selected_keys_keeps_project_order(self):
        self.project.set_template(make_settings())
        self.project.add_videos([self.a, self.b, self.c])
        keys = self.project.keys()
        plan = self.project.analysis_plan([keys[2], keys[0]])
        self.assertEqual(list(plan.plan), [self.a, self.c])

    def test_validate_reports_every_kind_of_problem(self):
        self.project.set_template(make_settings())
        self.project.add_videos([self.a, self.b, self.c])
        keys = self.project.keys()
        self.project.customize(keys[1], make_settings(paradigm_id="EPM"))
        self.project.videos[2].settings = None
        self.a.write_bytes(b"changed on disk")
        self.c.unlink()
        self.project.run_options.sample_sheet = self.root / "none.csv"
        found = {(i.code, i.key) for i in self.project.validate()}
        self.assertIn(("video_changed", keys[0]), found)
        self.assertIn(("paradigm_mismatch", keys[1]), found)
        self.assertIn(("unconfigured", keys[2]), found)
        self.assertIn(("missing_file", keys[2]), found)
        self.assertIn(("sample_sheet_missing", None), found)
        errors = {i.code for i in self.project.validate()
                  if i.severity == SEVERITY_ERROR}
        self.assertEqual(errors, {"paradigm_mismatch", "missing_file"})

    def test_validate_template_paradigm_mismatch(self):
        self.project.paradigm = "EPM"
        self.project.template_settings = make_settings(paradigm_id="Y-Maze")
        self.assertEqual(
            [(i.code, i.key) for i in self.project.validate()],
            [("paradigm_mismatch", None)],
        )

    def test_clean_project_has_no_issues(self):
        self.project.set_template(make_settings())
        self.project.add_videos([self.a, self.b])
        self.assertEqual(self.project.validate(), [])


class RunHistoryTests(OpsFixture):
    def setUp(self):
        super().setUp()
        self.project.set_template(make_settings())
        self.project.add_videos([self.a, self.b, self.c])
        self.run_dir = self.root / "out" / "analysis_run_1"
        self.run_dir.mkdir(parents=True)
        self.plan = {
            video.path: make_settings() for video in self.project.videos
        }
        self.maze = SimpleNamespace(
            name="Y-Maze", get_algorithm_parameters=lambda: {}
        )

    def _write_run(self, results, failures, warnings, cancelled=False):
        names = output_base_names(list(self.plan))
        write_run_plan(self.plan, self.run_dir)
        write_run_manifest(
            self.run_dir, self.maze, "2026-01-01T10:00:00+0000", len(self.plan),
            names, [(p, None) for p in results], failures, warnings,
            cancelled, analysis_plan=self.plan,
            hasher=CheckpointManager(self.root, create_dir=False),
        )

    def test_manifest_updates_statuses_and_history(self):
        self._write_run(
            results=[self.a, self.b], failures=[(self.c, "boom")],
            warnings=[(self.b, "low coverage")],
        )
        record = self.project.record_run(self.run_dir)
        statuses = {v.name: v.last_status for v in self.project.videos}
        self.assertEqual(statuses, {
            "a.mp4": STATUS_DONE, "b.mp4": STATUS_WARNING, "c.mp4": STATUS_FAILED,
        })
        self.assertEqual(self.project.videos[0].last_run_dir, self.run_dir)
        self.assertEqual((record.completed, record.failed), (2, 1))
        self.assertFalse(record.cancelled)
        self.assertEqual(record.manifest, self.run_dir / MANIFEST_FILENAME)
        self.assertEqual(record.started_at, "2026-01-01T10:00:00+0000")
        self.assertEqual(set(record.settings_hashes), set(self.project.keys()))
        self.assertEqual(self.project.runs, [record])

    def test_videos_in_the_plan_without_an_outcome_are_interrupted(self):
        self._write_run(results=[self.a], failures=[], warnings=[],
                        cancelled=True)
        # The manifest of a cancelled run lists only videos with an outcome.
        record = self.project.record_run(self.run_dir)
        statuses = {v.name: v.last_status for v in self.project.videos}
        self.assertEqual(statuses["a.mp4"], STATUS_DONE)
        self.assertEqual(statuses["b.mp4"], STATUS_INTERRUPTED)
        self.assertEqual(statuses["c.mp4"], STATUS_INTERRUPTED)
        self.assertTrue(record.cancelled)

    def test_run_plan_alone_marks_everything_interrupted(self):
        write_run_plan(self.plan, self.run_dir)
        record = self.project.record_run(self.run_dir)
        self.assertTrue(all(
            v.last_status == STATUS_INTERRUPTED for v in self.project.videos
        ))
        self.assertIsNone(record.manifest)
        self.assertTrue(record.cancelled)

    def test_recording_again_replaces_the_record(self):
        self._write_run([self.a, self.b, self.c], [], [])
        self.project.record_run(self.run_dir)
        self.project.record_run(self.run_dir)
        self.assertEqual(len(self.project.runs), 1)

    def test_videos_not_in_the_project_are_ignored(self):
        self._write_run([self.a, self.b, self.c], [], [])
        self.project.remove_videos([self.project.videos[2].key])
        record = self.project.record_run(self.run_dir)
        self.assertEqual(record.completed, 3)
        self.assertEqual(len(record.settings_hashes), 2)

    def test_folder_without_manifest_or_plan_and_newer_schema(self):
        with self.assertRaises(ProjectError):
            self.project.record_run(self.run_dir)
        (self.run_dir / MANIFEST_FILENAME).write_text(
            json.dumps({"schema_version": 99})
        )
        with self.assertRaises(ProjectError):
            self.project.record_run(self.run_dir)

    def test_unreadable_manifest_raises_project_error(self):
        (self.run_dir / MANIFEST_FILENAME).write_text("{not json")
        with self.assertRaises(ProjectError):
            self.project.record_run(self.run_dir)

    def test_plan_file_name_constants_match_the_writer(self):
        write_run_plan(self.plan, self.run_dir)
        self.assertTrue((self.run_dir / RUN_PLAN_FILENAME).is_file())


class MovedDataTests(OpsFixture):
    def test_missing_files_and_relink(self):
        self.project.set_template(make_settings())
        self.project.add_videos([self.a, self.c])
        keys = self.project.keys()
        old_fingerprint = self.project.videos[0].fingerprint
        new_root = self.root / "moved"
        new_root.mkdir()
        os.replace(self.root / "day1", new_root / "day1")
        self.assertEqual([v.name for v in self.project.missing_files()],
                         ["a.mp4"])
        relinked = self.project.relink(self.root / "day1", new_root / "day1")
        self.assertEqual(relinked, [keys[0]])
        entry = self.project.videos[0]
        self.assertEqual(entry.path, (new_root / "day1" / "a.mp4").resolve())
        self.assertEqual(entry.key, keys[0])
        self.assertNotEqual(entry.fingerprint, old_fingerprint)
        self.assertEqual(self.project.missing_files(), [])
        self.assertEqual(self.project.videos[1].path, self.c)

    def test_relink_follows_runs_and_sample_sheet(self):
        old = self.root / "old"
        new = self.root / "new"
        self.project.run_options.sample_sheet = old / "s.csv"
        from core.project import RunRecord
        self.project.runs.append(RunRecord(
            run_dir=(old / "out" / "r").resolve(),
            manifest=(old / "out" / "r" / "manifest.json").resolve(),
        ))
        self.project.relink(old, new)
        self.assertEqual(self.project.run_options.sample_sheet,
                         (new / "s.csv").resolve())
        self.assertEqual(self.project.runs[0].run_dir,
                         (new / "out" / "r").resolve())
        self.assertEqual(self.project.runs[0].manifest,
                         (new / "out" / "r" / "manifest.json").resolve())

    def test_first_bind_makes_keys_relative_and_remaps_history(self):
        project = Project()
        project.add_videos([self.a])
        self.assertEqual(project.videos[0].key, self.a.as_posix())
        from core.project import RunRecord
        project.runs.append(
            RunRecord(run_dir=self.root, settings_hashes={self.a.as_posix(): "h"})
        )
        project.bind_to_file(self.root / "p" / "x.bmaproj")
        self.assertEqual(project.videos[0].key, "../day1/a.mp4")
        self.assertEqual(project.runs[0].settings_hashes, {"../day1/a.mp4": "h"})
        # Binding to another place later keeps the keys.
        project.bind_to_file(self.root / "elsewhere" / "x.bmaproj")
        self.assertEqual(project.videos[0].key, "../day1/a.mp4")


if __name__ == "__main__":
    unittest.main()
