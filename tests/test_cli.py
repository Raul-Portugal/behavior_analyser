"""The headless CLI must match what the GUI worker writes, without Qt."""
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from core import cli
    from core.models import BatchSettings
    from gui.worker import AnalysisWorker
    from mazes.freestyle import Freestyle
    from tests import worker_support as support
    from tests.worker_support import FRAME_COUNTS, data_files
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FREESTYLE = "Freestyle / Open Field"


def freestyle_settings(paradigm_id=FREESTYLE):
    settings = support.tracking_settings(create_timelapse=True)
    settings.paradigm_id = paradigm_id
    settings.algorithm_parameters = Freestyle().get_algorithm_parameters()
    return settings


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        self.video_dir = self.root / "videos"
        self.video_dir.mkdir()
        self.videos = [
            support.write_tracking_video(self.video_dir, frames, name)
            for name, frames in FRAME_COUNTS.items()
        ]
        self.plan_file = self.root / "plan.json"
        BatchSettings.save_batch_plan(
            {f"videos/{video.name}": freestyle_settings()
             for video in self.videos},
            self.plan_file,
        )

    def _environment(self, label):
        cache = str(self.root / f"cache_{label}")
        return {"LOCALAPPDATA": cache, "XDG_CACHE_HOME": cache}

    def _run_cli(self, label, *extra, plan=None):
        out = self.root / label
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, self._environment(label)), \
                contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(stderr):
            code = cli.main([
                "run", "--plan", str(plan or self.plan_file),
                "--out", str(out), "--quiet", *extra,
            ])
        return code, out, stdout.getvalue(), stderr.getvalue()

    def _only_run_dir(self, out):
        (run_dir,) = sorted(out.glob("analysis_run_*"))
        return run_dir

    def _gui_worker_run_dir(self, label):
        out = self.root / label
        out.mkdir()
        plan = {
            video.resolve(): freestyle_settings() for video in self.videos
        }
        with patch.dict(os.environ, self._environment(label)):
            worker = AnalysisWorker(plan, out, Freestyle(), workers=1)
            outcomes = []
            worker.finished.connect(outcomes.append)
            worker.run()
        self.assertEqual(outcomes[0].failures, [])
        return worker.output_dir

    def test_cli_output_matches_the_gui_worker(self):
        gui_dir = self._gui_worker_run_dir("gui")
        code, out, stdout, _ = self._run_cli("cli_sequential", "--workers", "1")
        self.assertEqual(code, cli.EXIT_OK, stdout)
        cli_dir = self._only_run_dir(out)

        expected = data_files(gui_dir)
        actual = data_files(cli_dir)
        self.assertEqual(sorted(actual), sorted(expected))
        for required in ("batch_summary.csv", "run_plan.json"):
            self.assertIn(required, expected)
        for name in expected:
            self.assertEqual(actual[name], expected[name], name)
        self.assertTrue((cli_dir / "manifest.json").exists())
        self.assertIn(f"Run folder: {cli_dir}", stdout)

    def test_parallel_cli_matches_the_gui_worker(self):
        gui_dir = self._gui_worker_run_dir("gui")
        code, out, stdout, _ = self._run_cli("cli_parallel", "--workers", "3")
        self.assertEqual(code, cli.EXIT_OK, stdout)
        expected = data_files(gui_dir)
        actual = data_files(self._only_run_dir(out))
        self.assertEqual(sorted(actual), sorted(expected))
        for name in expected:
            self.assertEqual(actual[name], expected[name], name)

    def test_template_needs_videos_and_paradigm(self):
        template = self.root / "template.json"
        freestyle_settings(paradigm_id=None).save_to_file(template)

        code, _, _, stderr = self._run_cli("no_videos", plan=template)
        self.assertEqual(code, cli.EXIT_BAD_INPUT)
        self.assertIn("--videos", stderr)

        code, _, _, stderr = self._run_cli(
            "no_paradigm", "--videos", str(self.video_dir), plan=template
        )
        self.assertEqual(code, cli.EXIT_BAD_INPUT)
        self.assertIn("--paradigm", stderr)

        code, out, stdout, _ = self._run_cli(
            "template", "--videos", str(self.video_dir),
            "--paradigm", "freestyle", "--workers", "1", plan=template,
        )
        self.assertEqual(code, cli.EXIT_OK, stdout)
        files = data_files(self._only_run_dir(out))
        self.assertIn("batch_summary.csv", files)
        for video in self.videos:
            self.assertIn(f"{video.stem}/data/{video.stem}_summary.json", files)

    def test_paradigm_must_match_the_plan(self):
        code, _, _, stderr = self._run_cli(
            "mismatch", "--paradigm", "Y-Maze"
        )
        self.assertEqual(code, cli.EXIT_BAD_INPUT)
        self.assertIn("does not match", stderr)

    def test_bad_inputs_exit_with_status_2(self):
        code, _, _, stderr = self._run_cli(
            "missing", plan=self.root / "absent.json"
        )
        self.assertEqual(code, cli.EXIT_BAD_INPUT)
        self.assertIn("Cannot read", stderr)

        self.videos[1].unlink()
        code, _, _, stderr = self._run_cli("gone")
        self.assertEqual(code, cli.EXIT_BAD_INPUT)
        self.assertIn(self.videos[1].name, stderr)

    def test_a_failing_video_gives_exit_status_1_and_keeps_the_rest(self):
        self.videos[1].write_bytes(b"not a video")
        # The corrupt file passes plan loading but not the pre-flight check,
        # so run the plan directly to exercise the per-video failure path.
        plan = cli.load_analysis_plan(self.plan_file)
        maze = cli.resolve_maze(plan, None)
        out = self.root / "failing"
        with patch.dict(os.environ, self._environment("failing")):
            outcome, run_dir = cli.run_plan(
                plan, maze, out, 3, cli.ConsoleReporter(quiet=True),
                threading.Event(),
            )
        self.assertEqual(
            [path.name for path, _ in outcome.failures], [self.videos[1].name]
        )
        self.assertEqual(len(outcome.results), 2)
        self.assertTrue((run_dir / "batch_summary.csv").exists())

    def test_core_cli_does_not_import_qt(self):
        probe = (
            "import sys, core.cli, core.pipeline, core.batch_runner;"
            "bad = sorted(m for m in sys.modules if m.startswith('PyQt'));"
            "print(bad); sys.exit(1 if bad else 0)"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", probe], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_module_entry_point_runs_a_parallel_batch(self):
        out = self.root / "module"
        environment = {**os.environ, **self._environment("module")}
        result = subprocess.run(
            [sys.executable, "-B", "-m", "core.cli", "run",
             "--plan", str(self.plan_file), "--out", str(out),
             "--workers", "2", "--quiet"],
            cwd=PROJECT_ROOT, env=environment,
            capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        run_dir = self._only_run_dir(out)
        self.assertIn(f"Run folder: {run_dir}", result.stdout)
        self.assertTrue((run_dir / "batch_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
