"""Parallel batches must produce exactly what a sequential run produces."""
import json
import os
import pickle
import threading
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import psutil
    from PyQt6.QtWidgets import QApplication

    from core.batch_runner import default_worker_count, maze_rebuild_is_exact
    from core.pipeline import (
        NullReporter, VideoJob, VideoOutcome, build_maze, run_video,
    )
    from gui.main_window import MainWindow
    from gui.worker import AnalysisWorker
    from mazes.freestyle import Freestyle
    from mazes.tst import TST
    from tests import worker_support as support
    from tests.gui_support import temp_app_settings
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False

FRAME_COUNTS = {"mouse_a.avi": 70, "mouse_b.avi": 90, "mouse_c.avi": 110}


def data_files(run_dir: Path):
    """Every deterministic output of a run: all files except the manifest."""
    return {
        path.relative_to(run_dir).as_posix(): path.read_bytes()
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class ParallelBatchTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.video_dir = self.root / "videos"
        self.video_dir.mkdir()
        self.videos = [
            support.write_tracking_video(self.video_dir, frames, name)
            for name, frames in FRAME_COUNTS.items()
        ]

    def _run(self, workers, label, videos=None, on_progress=None):
        """Run the batch through the real worker; returns what it emitted."""
        out = self.root / label
        out.mkdir()
        cache = self.root / f"cache_{label}"
        plan = {
            video: support.tracking_settings(create_timelapse=True)
            for video in (videos or self.videos)
        }
        # Child processes inherit the environment, so each run gets its own
        # reference-frame cache and none touches the user's profile.
        environment = {"LOCALAPPDATA": str(cache), "XDG_CACHE_HOME": str(cache)}
        with patch.dict(os.environ, environment):
            worker = AnalysisWorker(plan, out, Freestyle(), workers=workers)
            outcomes, statuses, logs = [], [], []
            worker.finished.connect(outcomes.append)
            worker.video_status.connect(
                lambda video, state, message: statuses.append(
                    (Path(video).name, state, message)
                )
            )
            worker.log.connect(logs.append)
            if on_progress is not None:
                worker.detailed_progress.connect(
                    lambda *args: on_progress(worker, *args)
                )
            worker.run()
        return worker, outcomes, statuses, logs

    def _long_videos(self):
        """Videos long enough that a cancel always lands mid-tracking."""
        return [
            support.write_tracking_video(self.video_dir, 2500, f"long_{n}.avi")
            for n in "abc"
        ]

    def _assert_no_child_processes(self):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            children = psutil.Process().children(recursive=True)
            if not children:
                return
            time.sleep(0.2)
        self.fail(f"child processes still alive: {children}")

    def test_parallel_outputs_are_byte_identical_to_sequential(self):
        sequential, seq_outcomes, _, _ = self._run(1, "sequential")
        parallel, par_outcomes, _, par_logs = self._run(3, "parallel")

        for outcomes in (seq_outcomes, par_outcomes):
            self.assertEqual(len(outcomes), 1)
            self.assertEqual(outcomes[0].failures, [])
            self.assertFalse(outcomes[0].cancelled)
        self.assertTrue(
            any("3 parallel workers" in line for line in par_logs), par_logs
        )

        expected = data_files(sequential.output_dir)
        actual = data_files(parallel.output_dir)
        self.assertEqual(sorted(actual), sorted(expected))
        self.assertIn("batch_summary.csv", expected)
        self.assertTrue(any(name.endswith("_timelapse.mp4") for name in expected))
        for name in expected:
            self.assertEqual(actual[name], expected[name], name)

        # Results come back in the original video order.
        self.assertEqual(
            [path for path, _ in par_outcomes[0].results], self.videos
        )
        self.assertEqual(
            [path for path, _ in seq_outcomes[0].results], self.videos
        )
        manifests = [
            json.loads(
                (run.output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            for run in (sequential, parallel)
        ]
        for manifest in manifests:
            manifest.pop("started_at")
            manifest.pop("finished_at")
        self.assertEqual(manifests[0], manifests[1])
        self._assert_no_child_processes()

    def test_parallel_run_reports_status_for_every_video(self):
        _, outcomes, statuses, _ = self._run(3, "status")
        self.assertEqual(outcomes[0].failures, [])
        for name in FRAME_COUNTS:
            states = [state for video, state, _ in statuses if video == name]
            self.assertEqual(states[0], "running", name)
            self.assertEqual(states[-1], "done", name)

    def test_one_failing_video_does_not_stop_the_others(self):
        broken = self.video_dir / "broken.avi"
        broken.write_bytes(b"this is not a video")
        videos = [self.videos[0], broken, self.videos[1]]
        worker, outcomes, statuses, _ = self._run(3, "failure", videos)

        self.assertEqual(len(outcomes), 1)
        outcome = outcomes[0]
        self.assertFalse(outcome.cancelled)
        self.assertEqual([path for path, _ in outcome.failures], [broken])
        self.assertEqual(
            [path for path, _ in outcome.results], [self.videos[0], self.videos[1]]
        )
        self.assertEqual(
            [state for video, state, _ in statuses if video == "broken.avi"][-1],
            "failed",
        )
        run_dir = worker.output_dir
        self.assertTrue((run_dir / "mouse_a" / "data").is_dir())
        self.assertTrue((run_dir / "mouse_b" / "data").is_dir())
        self.assertTrue((run_dir / "batch_summary.csv").exists())
        self._assert_no_child_processes()

    def test_cancelling_a_parallel_run_stops_every_child(self):
        def stop_once_tracking(worker, _pct, _main, sub):
            if "Frame " in sub and not worker.is_cancelled:
                worker.stop()

        _, outcomes, statuses, _ = self._run(
            3, "cancel", self._long_videos(), on_progress=stop_once_tracking
        )

        self.assertEqual(len(outcomes), 1, "finished must be emitted once")
        self.assertTrue(outcomes[0].cancelled)
        self.assertEqual(outcomes[0].failures, [])
        self.assertTrue(
            any(state == "queued" and "interrupted" in message
                for _, state, message in statuses)
        )
        self._assert_no_child_processes()

    def test_cancelled_parallel_run_leaves_resumable_checkpoints(self):
        def stop_once_tracking(worker, _pct, _main, sub):
            if "Frame " in sub and not worker.is_cancelled:
                worker.stop()

        worker, outcomes, _, _ = self._run(
            3, "resume", self._long_videos(), on_progress=stop_once_tracking
        )
        self.assertTrue(outcomes[0].cancelled)
        checkpoints = worker.checkpoint_mgr.find_all_checkpoints()
        self.assertEqual(len(checkpoints), 3)


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class PicklingAndSizingTests(unittest.TestCase):
    def _job(self, root, maze, settings, **extra):
        return VideoJob(
            video_path=Path(root) / "video.avi",
            settings=settings,
            maze_name=maze.name,
            algorithm_parameters=maze.get_algorithm_parameters(),
            output_dir=Path(root) / "out" / "video",
            base_name="video",
            checkpoint_root=Path(root) / "out",
            **extra,
        )

    def test_jobs_pickle_for_every_paradigm_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            for maze, settings in (
                (Freestyle(), support.tracking_settings()),
                (TST(), support.tst_settings()),
            ):
                job = self._job(directory, maze, settings)
                clone = pickle.loads(pickle.dumps(job))
                self.assertEqual(
                    clone.settings.to_dict(), settings.to_dict()
                )
                self.assertEqual(clone.algorithm_parameters,
                                 maze.get_algorithm_parameters())

    def test_results_pickle_exactly(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            video = support.write_tst_video(root)
            for maze, settings, source in (
                (TST(), support.tst_settings(), video),
                (
                    Freestyle(), support.tracking_settings(),
                    support.write_tracking_video(root),
                ),
            ):
                job = self._job(root, maze, settings)
                job.video_path = source
                outcome = run_video(job, NullReporter(), threading.Event())
                self.assertTrue(outcome.succeeded, outcome.failure)
                clone = pickle.loads(pickle.dumps(outcome))
                self.assertIsInstance(clone, VideoOutcome)
                self.assertEqual(clone.result, outcome.result)
                self.assertEqual(clone.warnings, outcome.warnings)

    def test_maze_is_rebuilt_from_name_and_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            maze = TST()
            maze.configure_mice(2)
            maze.set_parameters(2.5, 0.4)
            job = self._job(directory, maze, support.tst_settings())
            rebuilt = build_maze(job)
            self.assertIsNot(rebuilt, maze)
            self.assertEqual(
                rebuilt.get_algorithm_parameters(),
                maze.get_algorithm_parameters(),
            )
            self.assertTrue(
                maze_rebuild_is_exact(maze, maze.get_algorithm_parameters())
            )

    def test_state_outside_the_parameters_forces_a_snapshot(self):
        maze = Freestyle()
        maze._legacy_threshold_px_per_s = 12.0
        self.assertFalse(
            maze_rebuild_is_exact(maze, maze.get_algorithm_parameters())
        )

    def test_default_worker_count(self):
        self.assertEqual(default_worker_count(10, cpu_count=16, ram_gb=64), 4)
        self.assertEqual(default_worker_count(2, cpu_count=16, ram_gb=64), 2)
        self.assertEqual(default_worker_count(10, cpu_count=16, ram_gb=7), 2)
        self.assertEqual(default_worker_count(10, cpu_count=2, ram_gb=64), 1)
        self.assertEqual(default_worker_count(0, cpu_count=16, ram_gb=64), 1)


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class WorkerControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _window(self):
        window = MainWindow(temp_app_settings(self))
        self.addCleanup(window.deleteLater)
        self.addCleanup(window._detach_log_handler)
        return window

    def test_setting_defaults_to_auto_and_round_trips(self):
        settings = temp_app_settings(self)
        self.assertEqual(settings.parallel_workers(), 0)
        settings.set_parallel_workers(3)
        self.assertEqual(settings.parallel_workers(), 3)
        settings.set_parallel_workers(-4)
        self.assertEqual(settings.parallel_workers(), 0)

    def test_spinbox_choice_is_remembered_and_clamped_to_the_batch(self):
        window = self._window()
        self.assertEqual(window.workers_spin.value(), 0)
        with patch("gui.main_window.default_worker_count", return_value=2) as auto:
            self.assertEqual(window._worker_count(5), 2)
            auto.assert_called_once_with(5)
        window.workers_spin.setMaximum(8)
        window.workers_spin.setValue(4)
        self.assertEqual(window.app_settings.parallel_workers(), 4)
        self.assertEqual(window._worker_count(10), 4)
        self.assertEqual(window._worker_count(3), 3)
        self.assertEqual(window._worker_count(1), 1)

    def test_spinbox_is_locked_while_running(self):
        window = self._window()
        window.set_ui_state(is_running=True)
        self.assertFalse(window.workers_spin.isEnabled())
        window.set_ui_state(is_running=False)
        self.assertTrue(window.workers_spin.isEnabled())


if __name__ == "__main__":
    unittest.main()
