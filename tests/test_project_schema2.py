"""Schema 2 of the project file: overrides on disk, migration from schema 1,
and the importers producing overrides instead of full copies."""
import json
import unittest

import numpy as np

from core.batch_plan_keys import build_plan_by_key
from core.models import BatchSettings
from core.project import (SCHEMA_VERSION, START_PER_VIDEO, ZONE_ADJUSTED,
                          ZONE_ALIGNED, Project, ProjectError, TrialWindow,
                          canonical_json, decompose_settings, settings_hash)
from core.project_io import (export_batch_plan, load_project,
                             project_from_batch_plan, project_from_run_folder,
                             save_project)
from core.run_outputs import write_run_plan
from tests.project_support import ProjectFixture, make_settings
from tests.test_project_overrides import (adjusted_zones, hash_of,
                                          paradigm_settings)

NEW_VIDEO_KEYS = ("trial", "zones", "duration_s", "frame_size")


def schema1_payload(project: Project, root) -> dict:
    """What a schema-1 build wrote for ``project`` (full copies, no layers)."""
    data = json.loads(json.dumps(project.to_dict(root)))
    data["schema_version"] = 1
    data.pop("trial_protocol", None)
    for video in data["videos"]:
        for key in NEW_VIDEO_KEYS:
            video.pop(key, None)
    return data


class SchemaTwoFileTests(ProjectFixture):
    def _project(self):
        project = Project(name="Study")
        project.set_template(paradigm_settings("Elevated Plus Maze"))
        project.add_videos([
            self.write_video("a.mp4"), self.write_video("b.mp4", b"bb"),
            self.write_video("c.mp4", b"ccc"),
        ])
        a, b, c = project.keys()
        project.set_trial_protocol(duration_s=300.0, start_mode=START_PER_VIDEO)
        project.set_trial(a, start=12.5)
        project.set_trial(b, start=30, end=130)
        project.set_zone_override(
            b, adjusted_zones("Elevated Plus Maze"), source=ZONE_ADJUSTED,
            scale_factor=0.5,
        )
        project.set_video_info(a, duration_s=900.0, frame_size=(640, 480))
        return project

    def test_header_and_new_fields_are_written(self):
        project = self._project()
        target = save_project(project, self.root / "p" / "study.bmaproj")
        data = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], SCHEMA_VERSION)
        self.assertEqual(SCHEMA_VERSION, 2)
        self.assertEqual(
            data["trial_protocol"],
            {"duration_s": 300.0, "start_mode": "per_video"},
        )
        a, b, c = data["videos"]
        self.assertEqual(a["trial"]["start_time"], 12.5)
        self.assertTrue(a["trial"]["start_set"])
        self.assertEqual(a["duration_s"], 900.0)
        self.assertEqual(a["frame_size"], [640, 480])
        self.assertEqual(b["zones"]["source"], "adjusted")
        self.assertEqual(b["zones"]["scale_factor"], 0.5)
        self.assertIsNone(c["trial"])
        self.assertIsNone(c["zones"])
        # The followers of the template still write just a reference to it.
        self.assertEqual([v["settings"] for v in data["videos"]], ["template"] * 3)

    def test_round_trip_keeps_the_effective_settings_and_their_hash(self):
        project = self._project()
        target = save_project(project, self.root / "p" / "study.bmaproj")
        loaded = load_project(target)
        self.assertEqual(loaded.trial_protocol, project.trial_protocol)
        self.assertEqual(loaded.load_warnings, [])
        for before, after in zip(project.videos, loaded.videos):
            self.assertEqual(before.trial, after.trial)
            self.assertEqual(before.duration_s, after.duration_s)
            self.assertEqual(before.frame_size, after.frame_size)
            self.assertEqual(
                (before.zones is None), (after.zones is None)
            )
            mine = project.effective_settings(before)
            theirs = loaded.effective_settings(after)
            self.assertEqual(canonical_json(mine.to_dict()),
                             canonical_json(theirs.to_dict()))
            self.assertEqual(settings_hash(mine), settings_hash(theirs))
        self.assertEqual(
            loaded.entry(loaded.keys()[1]).zones.to_dict(),
            project.entry(project.keys()[1]).zones.to_dict(),
        )

    def test_saving_a_loaded_project_changes_nothing(self):
        project = self._project()
        target = save_project(project, self.root / "p" / "study.bmaproj")
        first = json.loads(target.read_text(encoding="utf-8"))
        save_project(load_project(target), target)
        again = json.loads(target.read_text(encoding="utf-8"))
        for data in (first, again):
            data.pop("modified_at")
        self.assertEqual(first, again)

    def test_a_newer_schema_is_still_rejected(self):
        data = self._project().to_dict(self.root)
        with self.assertRaises(ProjectError):
            Project.from_dict(dict(data, schema_version=3), self.root)

    def test_unreadable_layers_are_kept_and_reported(self):
        project = self._project()
        data = json.loads(json.dumps(project.to_dict(self.root)))
        data["videos"][0]["trial"] = {"start_time": "soon"}
        data["videos"][1]["zones"] = {"source": "adjusted"}
        loaded = Project.from_dict(data, self.root)
        self.assertEqual(len(loaded.load_warnings), 2)
        self.assertIsNone(loaded.videos[0].trial)
        self.assertIsNone(loaded.videos[1].zones)
        again = loaded.to_dict(self.root)
        self.assertEqual(again["videos"][0]["trial"], {"start_time": "soon"})
        self.assertEqual(again["videos"][1]["zones"], {"source": "adjusted"})
        loaded.clear_trial(loaded.keys()[0])
        self.assertIsNone(loaded.to_dict(self.root)["videos"][0]["trial"])

    def test_a_bad_protocol_is_an_error(self):
        data = self._project().to_dict(self.root)
        data["trial_protocol"] = {"duration_s": -1, "start_mode": "per_video"}
        with self.assertRaises(ProjectError):
            Project.from_dict(data, self.root)


class MigrationTests(ProjectFixture):
    """Schema-1 files: full copies decompose only when that is exact."""

    def _legacy(self, customs, paradigm="Y-Maze", template_kwargs=None):
        """A schema-1 payload: ``customs`` maps video name -> (settings, source).

        The first video follows the template.
        """
        project = Project(name="Old")
        project.set_template(
            paradigm_settings(paradigm, **(template_kwargs or {}))
        )
        project.add_videos([self.write_video("plain.mp4")] + [
            self.write_video(name, name.encode()) for name in customs
        ])
        for name, (settings, source) in customs.items():
            entry = next(v for v in project.videos if v.name == name)
            project.customize(entry.key, settings, source=source)
        return project, schema1_payload(project, self.root)

    def assertUnchanged(self, old_project, loaded, name):
        """The migrated video composes to exactly what the full copy was."""
        before = old_project.effective_settings(
            next(v for v in old_project.videos if v.name == name)
        )
        after = loaded.effective_settings(
            next(v for v in loaded.videos if v.name == name)
        )
        self.assertEqual(before.to_dict(), after.to_dict())
        self.assertEqual(canonical_json(before.to_dict()),
                         canonical_json(after.to_dict()))
        self.assertEqual(hash_of(before), hash_of(after))

    def test_time_only_custom_becomes_a_trial(self):
        project, data = self._legacy({
            "m1.mp4": (paradigm_settings("Y-Maze", start_time=12.5, end_time=312.5),
                       "custom"),
        })
        loaded = Project.from_dict(data, self.root)
        entry = next(v for v in loaded.videos if v.name == "m1.mp4")
        self.assertEqual(entry.settings_source, "template")
        self.assertIsNone(entry.zones)
        self.assertEqual(
            entry.trial, TrialWindow(12.5, 312.5, {}, start_set=True)
        )
        self.assertEqual(loaded.zone_status(entry), "template")
        self.assertEqual(loaded.trial_status(entry), "ok")
        self.assertUnchanged(project, loaded, "m1.mp4")
        self.assertUnchanged(project, loaded, "plain.mp4")

    def test_roi_only_custom_becomes_a_zone_override(self):
        project, data = self._legacy({
            "m1.mp4": (paradigm_settings("Y-Maze", offset=9), "custom"),
        })
        loaded = Project.from_dict(data, self.root)
        entry = next(v for v in loaded.videos if v.name == "m1.mp4")
        self.assertEqual(entry.settings_source, "template")
        self.assertIsNone(entry.trial)
        self.assertEqual(entry.zones.source, "adjusted")
        self.assertIsNone(entry.zones.scale_factor)
        self.assertEqual(entry.zones.reference, loaded.zones_reference())
        self.assertEqual(loaded.zone_status(entry), "adjusted")
        self.assertUnchanged(project, loaded, "m1.mp4")

    def test_an_aligned_copy_becomes_aligned_zones_with_its_calibration(self):
        template = paradigm_settings("Y-Maze", scale=0.5)
        template.roi_manager.set_reference_line((0, 0), (100, 0), "ref")
        from core import roi_alignment as ra
        matrix = np.array([[2.0, 0.0, 3.0], [0.0, 2.0, 4.0], [0.0, 0.0, 1.0]])
        aligned = ra.apply_alignment(template, matrix, (1280, 960))
        aligned.start_time = 20.0
        project = Project(name="Old")
        project.set_template(template)
        project.add_videos([self.write_video("a.mp4"), self.write_video("b.mp4", b"b")])
        project.customize(project.keys()[1], aligned, source="aligned")
        loaded = Project.from_dict(schema1_payload(project, self.root), self.root)
        entry = loaded.videos[1]
        self.assertEqual(entry.zones.source, ZONE_ALIGNED)
        self.assertEqual(entry.zones.scale_factor, aligned.scale_factor)
        self.assertEqual(entry.trial.start_time, 20.0)
        self.assertEqual(loaded.zone_status(entry), "aligned")
        self.assertUnchanged(project, loaded, "b.mp4")

    def test_times_and_regions_together(self):
        project, data = self._legacy({
            "m1.mp4": (paradigm_settings("Y-Maze", offset=9, start_time=5.0,
                                         end_time=65.0), "custom"),
        })
        loaded = Project.from_dict(data, self.root)
        entry = next(v for v in loaded.videos if v.name == "m1.mp4")
        self.assertIsNotNone(entry.trial)
        self.assertIsNotNone(entry.zones)
        self.assertUnchanged(project, loaded, "m1.mp4")

    def test_tst_windows_migrate_to_a_trial(self):
        windows = {"mouse_1": (10.0, 250.0), "mouse_2": (25.0, 265.0)}
        project, data = self._legacy({
            "m1.mp4": (paradigm_settings("Tail Suspension Test",
                                         per_roi_times=windows), "custom"),
        }, paradigm="Tail Suspension Test")
        loaded = Project.from_dict(data, self.root)
        entry = next(v for v in loaded.videos if v.name == "m1.mp4")
        self.assertEqual(entry.trial.per_roi_times, windows)
        self.assertUnchanged(project, loaded, "m1.mp4")

    def test_other_differences_stay_a_full_copy(self):
        different = paradigm_settings("Y-Maze", offset=9, start_time=5.0)
        different.create_timelapse = True
        project, data = self._legacy({"m1.mp4": (different, "custom")})
        loaded = Project.from_dict(data, self.root)
        entry = next(v for v in loaded.videos if v.name == "m1.mp4")
        self.assertEqual(entry.settings_source, "custom")
        self.assertIsNone(entry.trial)
        self.assertIsNone(entry.zones)
        self.assertEqual(loaded.zone_status(entry), "custom")
        self.assertEqual(loaded.load_warnings, [])
        self.assertUnchanged(project, loaded, "m1.mp4")

    def test_what_a_layer_cannot_express_stays_a_full_copy(self):
        cases = {
            # The template ends at 600 s; "until the video ends" is not a trial.
            "open_end.mp4": paradigm_settings("Y-Maze", start_time=5.0),
            # A changed calibration without changed regions.
            "scale.mp4": paradigm_settings("Y-Maze", scale=0.3, end_time=600.0),
        }
        project, data = self._legacy(
            {name: (settings, "custom") for name, settings in cases.items()},
            template_kwargs={"end_time": 600.0},
        )
        loaded = Project.from_dict(data, self.root)
        for name in cases:
            entry = next(v for v in loaded.videos if v.name == name)
            self.assertEqual(entry.settings_source, "custom", name)
            self.assertUnchanged(project, loaded, name)

    def test_a_copy_identical_to_the_template_stays_detached(self):
        project, data = self._legacy({
            "m1.mp4": (paradigm_settings("Y-Maze"), "custom"),
        })
        loaded = Project.from_dict(data, self.root)
        entry = next(v for v in loaded.videos if v.name == "m1.mp4")
        self.assertEqual(entry.settings_source, "custom")
        self.assertEqual(loaded.load_warnings, [])

    def test_an_aligned_copy_with_unmoved_regions_stays_aligned(self):
        project, data = self._legacy({
            "m1.mp4": (paradigm_settings("Y-Maze", start_time=4.0), "aligned"),
        })
        loaded = Project.from_dict(data, self.root)
        entry = next(v for v in loaded.videos if v.name == "m1.mp4")
        self.assertEqual(entry.settings_source, "aligned")
        self.assertIsNone(entry.trial)
        self.assertEqual(loaded.zone_status(entry), "aligned")

    def test_without_a_template_nothing_migrates(self):
        project = Project(name="Old")
        project.add_videos([self.write_video("a.mp4")])
        project.customize(project.keys()[0], paradigm_settings("Y-Maze"))
        loaded = Project.from_dict(schema1_payload(project, self.root), self.root)
        self.assertEqual(loaded.videos[0].settings_source, "custom")
        self.assertEqual(loaded.load_warnings, [])

    def test_the_migration_is_reported_once_and_is_informational(self):
        customs = {
            f"m{i}.mp4": (paradigm_settings("Y-Maze", start_time=float(i)), "custom")
            for i in range(1, 8)
        }
        project, data = self._legacy(customs)
        loaded = Project.from_dict(data, self.root)
        self.assertEqual(len(loaded.load_warnings), 1)
        message = loaded.load_warnings[0]
        self.assertIn("7 video(s)", message)
        self.assertIn("m1.mp4", message)
        self.assertIn("2 more", message)
        self.assertIn("unchanged", message)

    def test_a_schema_2_file_is_not_migrated_again(self):
        project = Project(name="New")
        project.set_template(paradigm_settings("Y-Maze"))
        project.add_videos([self.write_video("a.mp4")])
        project.customize(
            project.keys()[0], paradigm_settings("Y-Maze", start_time=3.0)
        )
        loaded = Project.from_dict(
            json.loads(json.dumps(project.to_dict(self.root))), self.root
        )
        self.assertEqual(loaded.videos[0].settings_source, "custom")
        self.assertIsNone(loaded.videos[0].trial)
        self.assertEqual(loaded.load_warnings, [])

    def test_existing_files_open_and_resave_as_schema_2(self):
        project, data = self._legacy({
            "m1.mp4": (paradigm_settings("Y-Maze", offset=4), "custom"),
        })
        path = self.root / "old.bmaproj"
        path.write_text(json.dumps(data), encoding="utf-8")
        loaded = load_project(path)
        self.assertEqual(len(loaded.load_warnings), 1)
        save_project(loaded, path)
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["schema_version"], 2)
        self.assertEqual(
            [v["settings"] for v in saved["videos"]], ["template", "template"]
        )
        self.assertEqual(load_project(path).load_warnings, [])
        self.assertUnchanged(project, load_project(path), "m1.mp4")

    def test_the_project_make_settings_fixture_migrates_exactly(self):
        # The fixtures of the older tests: make_settings(offset=...) customs.
        project = Project(name="Old")
        project.set_template(make_settings())
        project.add_videos([
            self.write_video("a.mp4"), self.write_video("b.mp4", b"bb"),
        ])
        project.customize(project.keys()[1], make_settings(offset=4))
        loaded = Project.from_dict(schema1_payload(project, self.root), self.root)
        self.assertEqual(loaded.zone_status(loaded.videos[1]), "adjusted")
        self.assertEqual(
            loaded.effective_settings(loaded.videos[1]).to_dict(),
            make_settings(offset=4).to_dict(),
        )


class DecomposeTests(unittest.TestCase):
    def test_identical_and_unrelated_settings(self):
        template = paradigm_settings("Y-Maze")
        self.assertEqual(
            decompose_settings(template, template.copy()), (None, None)
        )
        other = paradigm_settings("Elevated Plus Maze")
        self.assertIsNone(decompose_settings(template, other))

    def test_returns_copies(self):
        template = paradigm_settings("Y-Maze")
        moved = paradigm_settings("Y-Maze", offset=5)
        trial, zones = decompose_settings(template, moved)
        self.assertIsNone(trial)
        zones.roi_manager.rois.clear()
        self.assertTrue(moved.roi_manager.rois)


class ImporterTests(ProjectFixture):
    def _plan(self, entries):
        plan_file = self.root / "plans" / "plan.json"
        plan_file.parent.mkdir(exist_ok=True)
        BatchSettings.save_batch_plan(
            build_plan_by_key(entries, plan_file), plan_file
        )
        return plan_file

    def assertPlanReproduced(self, project, expected):
        for path, settings in expected.items():
            entry = project.find(path)
            self.assertEqual(
                project.effective_settings(entry).to_dict(), settings.to_dict()
            )
            self.assertEqual(
                settings_hash(project.effective_settings(entry)),
                settings_hash(settings),
            )

    def test_a_plan_that_differs_only_in_times_is_a_template_plus_trials(self):
        videos = [self.write_video(f"v/m{i}.mp4", bytes([i])) for i in range(4)]
        starts = [10.0, 22.5, 0.0, 41.0]
        expected = {
            path: paradigm_settings(
                "Y-Maze", start_time=start, end_time=start + 300.0
            )
            for path, start in zip(videos, starts)
        }
        project = project_from_batch_plan(self._plan(expected))
        self.assertIsNotNone(project.template_settings)
        self.assertEqual(project.template_settings.start_time, 0.0)
        self.assertIsNone(project.template_settings.end_time)
        self.assertTrue(
            all(v.settings_source == "template" for v in project.videos)
        )
        self.assertTrue(all(v.trial is not None for v in project.videos))
        self.assertTrue(all(v.zones is None for v in project.videos))
        self.assertEqual(project.validate(), [])
        self.assertPlanReproduced(project, expected)

    def test_shared_times_stay_in_the_template(self):
        videos = [self.write_video(f"v/m{i}.mp4", bytes([i])) for i in range(3)]
        expected = {
            path: paradigm_settings("Y-Maze", start_time=15.0, end_time=315.0)
            for path in videos
        }
        project = project_from_batch_plan(self._plan(expected))
        self.assertEqual(project.template_settings.start_time, 15.0)
        self.assertTrue(all(v.trial is None for v in project.videos))
        self.assertPlanReproduced(project, expected)

    def test_times_regions_and_a_real_outlier(self):
        a, b, c, d = (self.write_video(f"v/{n}.mp4", n.encode()) for n in "abcd")
        outlier = paradigm_settings("Y-Maze", start_time=3.0)
        outlier.create_timelapse = True
        expected = {
            a: paradigm_settings("Y-Maze", start_time=1.0),
            b: paradigm_settings("Y-Maze", start_time=2.0),
            c: paradigm_settings("Y-Maze", offset=8, start_time=2.0),
            d: outlier,
        }
        project = project_from_batch_plan(self._plan(expected))
        self.assertEqual(
            [project.zone_status(v) for v in project.videos],
            ["template", "template", "adjusted", "custom"],
        )
        self.assertEqual(project.entry(project.find(d).key).settings_source, "custom")
        self.assertPlanReproduced(project, expected)

    def test_a_run_folder_with_different_starts_opens_as_trials(self):
        videos = [self.write_video(f"v/m{i}.mp4", bytes([i])) for i in range(3)]
        plan = {
            path: paradigm_settings(
                "Y-Maze", start_time=float(5 * (i + 1)), end_time=5.0 * (i + 1) + 60
            )
            for i, path in enumerate(videos)
        }
        run_dir = self.root / "out" / "analysis_run_1"
        run_dir.mkdir(parents=True)
        write_run_plan(plan, run_dir)
        project = project_from_run_folder(run_dir)
        self.assertEqual(len(project.videos), 3)
        self.assertTrue(all(v.trial is not None for v in project.videos))
        self.assertPlanReproduced(project, plan)

    def test_export_writes_the_effective_settings(self):
        project = Project(name="Study")
        project.set_template(paradigm_settings("Y-Maze"))
        a, b = self.write_video("v/a.mp4"), self.write_video("v/b.mp4", b"bb")
        project.add_videos([a, b])
        project.set_trial(project.find(a).key, start=12.0, end=72.0)
        project.set_zone_override(
            project.find(b).key, adjusted_zones("Y-Maze"), scale_factor=0.25
        )
        target = self.root / "export" / "plan.json"
        self.assertEqual(export_batch_plan(project, target), [])
        loaded = BatchSettings.load_batch_plan(target)
        by_name = {
            path.split("/")[-1]: settings for path, settings in loaded.items()
        }
        self.assertEqual(
            by_name["a.mp4"].to_dict(),
            paradigm_settings("Y-Maze", start_time=12.0, end_time=72.0).to_dict(),
        )
        self.assertEqual(
            by_name["b.mp4"].to_dict(),
            paradigm_settings("Y-Maze", offset=7, scale=0.25).to_dict(),
        )


if __name__ == "__main__":
    unittest.main()
