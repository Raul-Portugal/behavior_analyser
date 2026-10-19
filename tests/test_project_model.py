"""Project data model: dataclasses, paths and (de)serialization."""
import json
import unittest

from core.project import (SCHEMA_VERSION, STATUS_DONE, STATUS_NEW, Project,
                          ProjectError, RunOptions, RunRecord, VideoEntry,
                          rebase_path, resolve_stored_path, store_path)
from tests.project_support import ProjectFixture, make_settings


class PathHelperTests(ProjectFixture):
    def test_paths_are_stored_relative_with_posix_separators(self):
        video = self.write_video("day1/mouse1.mp4")
        base = self.root / "proj"
        base.mkdir()
        self.assertEqual(store_path(video, base), "../day1/mouse1.mp4")
        self.assertEqual(resolve_stored_path("../day1/mouse1.mp4", base), video)

    def test_absolute_fallback_and_none(self):
        video = self.write_video("a.mp4")
        self.assertEqual(store_path(video, None), video.as_posix())
        self.assertEqual(resolve_stored_path(video.as_posix(), self.root), video)
        self.assertIsNone(store_path(None, self.root))
        self.assertIsNone(resolve_stored_path("", self.root))

    def test_rebase_only_inside_old_root(self):
        old = self.root / "old"
        new = self.root / "new"
        inside = old / "x" / "a.mp4"
        self.assertEqual(
            rebase_path(inside, old, new), (new / "x" / "a.mp4").resolve()
        )
        self.assertIsNone(rebase_path(self.root / "other" / "a.mp4", old, new))


class RunOptionsTests(ProjectFixture):
    def test_round_trip_with_relative_sample_sheet(self):
        sheet = self.write_video("sheets/s.csv")
        options = RunOptions(4, 30.0, sheet, True)
        data = options.to_dict(self.root)
        self.assertEqual(data["sample_sheet"], "sheets/s.csv")
        self.assertEqual(RunOptions.from_dict(data, self.root), options)

    def test_defaults_and_validation(self):
        self.assertEqual(RunOptions.from_dict(None), RunOptions())
        self.assertEqual(RunOptions().parallel_workers, 0)
        with self.assertRaises(ProjectError):
            RunOptions.from_dict({"parallel_workers": -1})
        with self.assertRaises(ProjectError):
            RunOptions.from_dict({"time_bin_seconds": -5})


class VideoEntryTests(ProjectFixture):
    def test_round_trip_keeps_every_field(self):
        video = self.write_video("m1.mp4")
        entry = VideoEntry(
            key="m1.mp4", path=video, metadata={"group": "ctl"},
            settings=make_settings(), settings_source="aligned", verified=True,
            fingerprint="abc", last_status=STATUS_DONE,
            last_run_dir=self.root / "out" / "run1", notes="note",
        )
        data = json.loads(json.dumps(entry.to_dict(self.root)))
        self.assertEqual(data["path"], "m1.mp4")
        self.assertEqual(data["last_run_dir"], "out/run1")
        loaded = VideoEntry.from_dict(data, self.root)
        self.assertEqual(loaded.path, video)
        self.assertEqual(loaded.metadata, {"group": "ctl"})
        self.assertEqual(loaded.settings.to_dict(), entry.settings.to_dict())
        self.assertEqual(
            (loaded.settings_source, loaded.verified, loaded.fingerprint,
             loaded.last_status, loaded.notes),
            ("aligned", True, "abc", STATUS_DONE, "note"),
        )
        self.assertEqual(
            loaded.last_run_dir, (self.root / "out" / "run1").resolve()
        )

    def test_unconfigured_entry_defaults(self):
        entry = VideoEntry(key="k", path=self.root / "k.mp4")
        self.assertFalse(entry.is_configured)
        self.assertEqual(entry.last_status, STATUS_NEW)
        self.assertEqual(entry.name, "k.mp4")
        loaded = VideoEntry.from_dict(entry.to_dict(self.root), self.root)
        self.assertIsNone(loaded.settings)

    def test_unknown_status_and_source_are_normalised(self):
        data = VideoEntry(key="k", path=self.root / "k.mp4").to_dict(self.root)
        data["last_status"] = "weird"
        data["settings_source"] = "weird"
        loaded = VideoEntry.from_dict(data, self.root)
        self.assertEqual(loaded.last_status, STATUS_NEW)
        self.assertEqual(loaded.settings_source, "custom")

    def test_missing_fields_are_rejected(self):
        with self.assertRaises(ProjectError):
            VideoEntry.from_dict({"key": "k"}, self.root)

    def test_unreadable_settings_are_kept_verbatim(self):
        data = VideoEntry(key="k", path=self.root / "k.mp4").to_dict(self.root)
        data["settings"] = dict(make_settings().to_dict(), schema_version=99)
        warnings = []
        loaded = VideoEntry.from_dict(data, self.root, warnings=warnings)
        self.assertIsNone(loaded.settings)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(
            loaded.to_dict(self.root)["settings"], data["settings"]
        )


class RunRecordTests(ProjectFixture):
    def test_round_trip(self):
        record = RunRecord(
            run_dir=(self.root / "out" / "analysis_run_1").resolve(),
            started_at="2026-01-01T10:00:00+0000",
            finished_at="2026-01-01T10:05:00+0000",
            cancelled=True, completed=3, failed=1,
            manifest=(self.root / "out" / "analysis_run_1"
                      / "manifest.json").resolve(),
            settings_hashes={"a.mp4": "h1"},
        )
        data = record.to_dict(self.root)
        self.assertEqual(data["run_dir"], "out/analysis_run_1")
        self.assertEqual(data["manifest"], "out/analysis_run_1/manifest.json")
        self.assertEqual(RunRecord.from_dict(data, self.root), record)

    def test_run_dir_is_required(self):
        with self.assertRaises(ProjectError):
            RunRecord.from_dict({}, self.root)


class ProjectSerializationTests(ProjectFixture):
    def _project(self) -> Project:
        project = Project(name="Study", paradigm="Y-Maze")
        project.template_settings = make_settings()
        project.file_path = self.root / "study.bmaproj"
        project.notes = "hello"
        project.algorithm_parameters = {"alpha": 1}
        for name in ("a.mp4", "sub/b.mp4"):
            video = self.write_video(name)
            project.videos.append(VideoEntry(
                key=project.key_for(video), path=video,
                settings=project.template_settings.copy(),
            ))
        project.videos[1].settings = make_settings(offset=7)
        project.videos[1].settings_source = "custom"
        project.runs.append(RunRecord(run_dir=(self.root / "out" / "r1").resolve()))
        return project

    def test_round_trip(self):
        project = self._project()
        data = json.loads(json.dumps(project.to_dict(self.root)))
        loaded = Project.from_dict(data, self.root)
        self.assertEqual(loaded.name, "Study")
        self.assertEqual(loaded.paradigm, "Y-Maze")
        self.assertEqual(loaded.notes, "hello")
        self.assertEqual(loaded.algorithm_parameters, {"alpha": 1})
        self.assertEqual(loaded.keys(), ["a.mp4", "sub/b.mp4"])
        self.assertEqual(loaded.videos[1].path, project.videos[1].path)
        self.assertEqual(
            loaded.videos[0].settings.to_dict(),
            project.videos[0].settings.to_dict(),
        )
        self.assertEqual(
            loaded.videos[1].settings.to_dict(),
            project.videos[1].settings.to_dict(),
        )
        self.assertEqual(loaded.runs[0].run_dir, project.runs[0].run_dir)
        self.assertEqual(loaded.load_warnings, [])

    def test_header_fields(self):
        data = self._project().to_dict(self.root)
        self.assertEqual(data["type"], "project")
        self.assertEqual(data["schema_version"], SCHEMA_VERSION)
        self.assertEqual(Project.schema_version, 1)

    def test_template_identical_settings_are_written_once(self):
        data = self._project().to_dict(self.root)
        self.assertEqual(data["videos"][0]["settings"], "template")
        self.assertIsInstance(data["videos"][1]["settings"], dict)

    def test_template_settings_changed_by_hand_are_not_shared(self):
        project = self._project()
        project.videos[0].settings = make_settings(offset=3)  # still "template"
        data = project.to_dict(self.root)
        self.assertIsInstance(data["videos"][0]["settings"], dict)

    def test_newer_schema_and_wrong_type_are_rejected(self):
        data = self._project().to_dict(self.root)
        with self.assertRaises(ProjectError):
            Project.from_dict(dict(data, schema_version=SCHEMA_VERSION + 1))
        with self.assertRaises(ProjectError):
            Project.from_dict(dict(data, type="batch_plan"))

    def test_duplicate_keys_are_rejected(self):
        data = self._project().to_dict(self.root)
        data["videos"][1]["key"] = data["videos"][0]["key"]
        with self.assertRaises(ProjectError):
            Project.from_dict(data, self.root)

    def test_template_reference_without_template_is_rejected(self):
        data = self._project().to_dict(self.root)
        data["template_settings"] = None
        with self.assertRaises(ProjectError):
            Project.from_dict(data, self.root)

    def test_unreadable_template_is_an_error(self):
        data = self._project().to_dict(self.root)
        data["template_settings"]["schema_version"] = 99
        with self.assertRaises(ProjectError):
            Project.from_dict(data, self.root)

    def test_runtime_attributes_are_not_serialized(self):
        data = self._project().to_dict(self.root)
        for name in ("file_path", "load_warnings", "_listeners"):
            self.assertNotIn(name, data)

    def test_key_for_is_relative_to_project_file_or_absolute(self):
        video = self.write_video("day/m.mp4")
        project = Project()
        self.assertEqual(project.key_for(video), video.as_posix())
        project.file_path = self.root / "p" / "x.bmaproj"
        self.assertEqual(project.key_for(video), "../day/m.mp4")

    def test_metadata_columns_in_first_seen_order(self):
        project = Project()
        project.videos = [
            VideoEntry(key="a", path=self.root / "a",
                       metadata={"g": "1", "s": "2"}),
            VideoEntry(key="b", path=self.root / "b",
                       metadata={"x": "1", "g": "2"}),
        ]
        self.assertEqual(project.metadata_columns(), ["g", "s", "x"])


if __name__ == "__main__":
    unittest.main()
