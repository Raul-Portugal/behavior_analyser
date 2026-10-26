"""A workspace run with trial and zone overrides equals the CLI run of the
equivalent full copies, byte for byte (outputs, run plan, settings hashes)."""
import contextlib
import copy
import io
import json
import unittest

import numpy as np

from tests.worker_support import data_files
from tests.workspace_run_support import RunFixture, freestyle_settings
from tests.workspace_support import HAS_RUNTIME

if HAS_RUNTIME:
    from core import cli
    from core.project import START_PER_VIDEO, ZONE_ADJUSTED
    from core.project_io import (export_batch_plan, load_project,
                                 save_project)
    from roi_manager import ROIManager

# Three videos of 70 / 90 / 110 frames at 30 fps (2.3, 3.0 and 3.7 s).
START_A, END_A = 0.3, 2.0       # explicit start and end
START_B, DURATION = 0.5, 2.0    # start only: the protocol gives the end
START_C, END_C = 0.2, 3.0       # also has its own regions


def smaller_arena() -> "ROIManager":
    manager = ROIManager(source_dimensions=(160, 120))
    manager.add_roi(
        "arena", np.array([[8, 6], [150, 6], [150, 112], [8, 112]])
    )
    return manager


def full_copy(name: str):
    """What a 3.x batch plan would hold for the video: a full copy."""
    settings = freestyle_settings()
    if name == "mouse_a.avi":
        settings.start_time, settings.end_time = START_A, END_A
    elif name == "mouse_b.avi":
        settings.start_time, settings.end_time = START_B, START_B + DURATION
    else:
        settings.start_time, settings.end_time = START_C, END_C
        settings.roi_manager = smaller_arena()
    return settings


def manifest_hashes(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    return {
        item["source"].replace("\\", "/").rsplit("/", 1)[-1]:
            item["settings_hash"]
        for item in manifest["videos"]
    }


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class TrialRunTests(RunFixture):
    def setUp(self):
        super().setUp()
        project = self.project
        project.set_trial_protocol(DURATION, START_PER_VIDEO)
        a, b, c = (project.find(path).key for path in self.real_videos)
        project.set_trial(a, start=START_A, end=END_A)
        project.set_trial(b, start=START_B)
        project.set_trial(c, start=START_C, end=END_C)
        project.set_zone_override(c, smaller_arena(), source=ZONE_ADJUSTED)

    def _by_name(self, project=None):
        project = project or self.project
        return {
            entry.name: project.effective_settings(entry)
            for entry in project.videos
        }

    def test_the_composed_settings_are_the_full_copies(self):
        for name, settings in self._by_name().items():
            expected = full_copy(name)
            self.assertEqual(settings.to_dict(), expected.to_dict(), name)

    def test_every_video_follows_the_template_for_the_rest(self):
        self.assertTrue(
            all(v.settings_source == "template" for v in self.project.videos)
        )
        self.assertEqual(
            self.project.to_dict(self.root)["videos"][0]["settings"], "template"
        )

    def test_the_workspace_run_equals_the_cli_run_of_the_full_copies(self):
        with self.isolated_environment("ws"):
            outcome = self.start_and_wait()
        self.assertEqual(outcome.failures, [])
        ws_dir = outcome.output_dir
        cli_dir = self.run_cli(
            "cli", "--workers", "1",
            settings_for=lambda video: full_copy(video.name),
        )
        expected, actual = data_files(cli_dir), data_files(ws_dir)
        self.assertEqual(sorted(actual), sorted(expected))
        self.assertIn("batch_summary.csv", expected)
        self.assertIn("run_plan.json", expected)
        for name in expected:
            self.assertEqual(actual[name], expected[name], name)
        # The settings the videos were analysed with hash the same, so
        # checkpoints written by one are accepted by the other.
        self.assertEqual(manifest_hashes(ws_dir), manifest_hashes(cli_dir))
        self.assertEqual(len(manifest_hashes(ws_dir)), 3)

    def test_an_exported_plan_run_by_the_cli_equals_the_workspace_run(self):
        with self.isolated_environment("ws"):
            outcome = self.start_and_wait()
        plan = self.root / "exported_plan.json"
        self.assertEqual(export_batch_plan(self.project, plan), [])
        out = self.root / "exported_out"
        with self.isolated_environment("exported"), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = cli.main([
                "run", "--plan", str(plan), "--out", str(out), "--quiet",
                "--workers", "1",
            ])
        self.assertEqual(code, cli.EXIT_OK)
        (cli_dir,) = sorted(out.glob("analysis_run_*"))
        expected, actual = data_files(cli_dir), data_files(outcome.output_dir)
        self.assertEqual(sorted(actual), sorted(expected))
        for name in expected:
            self.assertEqual(actual[name], expected[name], name)
        self.assertEqual(
            manifest_hashes(cli_dir), manifest_hashes(outcome.output_dir)
        )

    def test_the_overrides_change_what_is_analysed(self):
        # Guard against a vacuous comparison: the same run without them
        # produces different numbers.
        with self.isolated_environment("ws"):
            outcome = self.start_and_wait()
        plain_cli = self.run_cli("plain", "--workers", "1")
        self.assertNotEqual(
            data_files(plain_cli)["batch_summary.csv"],
            data_files(outcome.output_dir)["batch_summary.csv"],
        )

    def test_a_saved_and_reopened_project_runs_the_same(self):
        target = save_project(self.project, self.root / "study.bmaproj")
        reopened = load_project(target)
        for name, settings in self._by_name(reopened).items():
            self.assertEqual(
                settings.to_dict(), self._by_name()[name].to_dict(), name
            )
        self.assertEqual(
            {name: manifest_hash_of(s) for name, s in self._by_name(reopened).items()},
            {name: manifest_hash_of(s) for name, s in self._by_name().items()},
        )

    def test_the_run_plan_the_workspace_wrote_holds_the_composed_settings(self):
        with self.isolated_environment("ws"):
            outcome = self.start_and_wait()
        plan = json.loads(
            (outcome.output_dir / "run_plan.json").read_text("utf-8")
        )["plan"]
        by_name = {
            key.replace("\\", "/").rsplit("/", 1)[-1]: value
            for key, value in plan.items()
        }
        wanted = copy.deepcopy(by_name["mouse_c.avi"])
        self.assertEqual(wanted["start_time"], START_C)
        self.assertEqual(wanted["end_time"], END_C)
        self.assertEqual(
            wanted["roi_manager"], smaller_arena().to_dict()
        )
        self.assertEqual(by_name["mouse_b.avi"]["end_time"], START_B + DURATION)


def manifest_hash_of(settings) -> str:
    from core.project import settings_hash
    return settings_hash(settings)


if __name__ == "__main__":
    unittest.main()
