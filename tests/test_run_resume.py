"""run_plan.json and resuming an interrupted run after a restart."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from checkpoint_manager import CheckpointManager

try:
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from core.models import BatchSettings
    from core.video import ReferenceFrameGenerator
    from gui.main_window import MainWindow
    from gui.run_resume import (RUN_PLAN_FILENAME, describe_scan,
                                plan_key_to_path, scan_for_resumable_runs)
    from gui.worker import AnalysisWorker
    from mazes import AVAILABLE_MAZES, create_maze
    from mazes.freestyle import Freestyle
    from mazes.tst import TST
    from tests import worker_support as support
    from tests.gui_support import temp_app_settings
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


def _cancel_at_first_progress(worker):
    state = {"fired": False}

    def on_progress(_pct, _main, sub):
        if not state["fired"] and ("Frame " in sub or "Motion Analysis" in sub):
            state["fired"] = True
            worker.stop()

    worker.detailed_progress.connect(on_progress)


def _hash(output_dir: Path, settings) -> str:
    return CheckpointManager(output_dir, create_dir=False)._get_settings_hash(
        settings
    )


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class RunPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        self.out = self.root / "out"
        self.out.mkdir()
        cache = patch.object(
            ReferenceFrameGenerator, "get_cache_dir",
            return_value=self.root / "cache",
        )
        cache.start()
        self.addCleanup(cache.stop)

    def _run(self, plan, maze, cancel=False):
        worker = AnalysisWorker(plan, self.out, maze)
        logs, outcomes = [], []
        worker.log.connect(logs.append)
        worker.finished.connect(outcomes.append)
        if cancel:
            _cancel_at_first_progress(worker)
        worker.run()
        return worker, outcomes[0], logs

    def _interrupted_tracking_run(self, videos=None):
        video = support.write_tracking_video(self.root)
        videos = videos or [video]
        plan = {path: support.tracking_settings() for path in videos}
        worker, outcome, _ = self._run(plan, Freestyle(), cancel=True)
        self.assertTrue(outcome.cancelled)
        return video, worker

    def test_every_run_folder_gets_a_plan_with_paradigm_parameters(self):
        video, worker = self._interrupted_tracking_run()
        plan_file = worker.output_dir / RUN_PLAN_FILENAME
        self.assertTrue(plan_file.is_file())

        data = json.loads(plan_file.read_text(encoding="utf-8"))
        self.assertEqual(data["type"], "batch_plan")
        (key, entry), = data["plan"].items()
        self.assertEqual(Path(key).as_posix(), key)
        self.assertEqual(plan_key_to_path(key, plan_file), video)
        self.assertEqual(entry["paradigm_id"], "Freestyle / Open Field")
        self.assertIn("min_dwell_seconds", entry["algorithm_parameters"])

    def test_plan_round_trip_keeps_the_checkpoint_settings_hash(self):
        video, worker = self._interrupted_tracking_run()
        plan = BatchSettings.load_batch_plan(
            worker.output_dir / RUN_PLAN_FILENAME
        )
        (loaded,) = plan.values()
        self.assertEqual(
            _hash(self.out, loaded), _hash(self.out, worker.analysis_plan[video])
        )
        (checkpoint,) = worker.checkpoint_mgr.find_all_checkpoints()
        header = json.loads(checkpoint.read_text(encoding="utf-8"))
        self.assertEqual(header["settings_hash"], _hash(self.out, loaded))

    def test_tst_plan_round_trip_keeps_the_settings_hash(self):
        video = support.write_tst_video(self.root)
        worker = AnalysisWorker({video: support.tst_settings()}, self.out, TST())
        worker._write_run_plan()
        (loaded,) = BatchSettings.load_batch_plan(
            worker.output_dir / RUN_PLAN_FILENAME
        ).values()
        self.assertEqual(
            _hash(self.out, loaded), _hash(self.out, worker.analysis_plan[video])
        )

    def test_scan_finds_the_run_and_a_new_worker_resumes_from_its_plan(self):
        video, first = self._interrupted_tracking_run()

        scan = scan_for_resumable_runs(self.out)
        self.assertEqual(scan.unmatched, [])
        (run,) = scan.runs
        self.assertEqual(run.run_dir, first.output_dir)
        self.assertEqual(run.paradigm_id, "Freestyle / Open Field")
        self.assertEqual(run.pending_videos, [video])
        self.assertEqual([item.video for item in run.interrupted], [video])
        self.assertIn("1 interrupted video", describe_scan(scan))

        # What "Resume interrupted run" does: plan + parameters, nothing else.
        plan = BatchSettings.load_batch_plan(run.plan_path)
        (settings,) = plan.values()
        maze = create_maze(run.paradigm_id)
        maze.configure_algorithm_parameters(settings.algorithm_parameters)
        _, resumed, logs = self._run({video: settings}, maze)
        self.assertTrue(any("Resuming from checkpoint" in m for m in logs), logs)
        self.assertEqual(resumed.failures, [])
        self.assertEqual(scan_for_resumable_runs(self.out).interrupted_count, 0)

    def test_tst_run_resumes_from_its_plan(self):
        video = support.write_tst_video(self.root)
        _, outcome, _ = self._run(
            {video: support.tst_settings()}, TST(), cancel=True
        )
        self.assertTrue(outcome.cancelled)

        (run,) = scan_for_resumable_runs(self.out).runs
        (settings,) = BatchSettings.load_batch_plan(run.plan_path).values()
        maze = create_maze(run.paradigm_id)
        maze.configure_algorithm_parameters(settings.algorithm_parameters)
        _, resumed, logs = self._run({video: settings}, maze)
        self.assertTrue(any("Resuming from checkpoint" in m for m in logs), logs)

    def test_parameters_are_stable_when_reapplied_to_a_fresh_maze(self):
        for name in AVAILABLE_MAZES:
            with self.subTest(maze=name):
                maze = create_maze(name)
                saved = maze.get_algorithm_parameters()
                again = create_maze(name)
                again.configure_algorithm_parameters(json.loads(json.dumps(saved)))
                self.assertEqual(again.get_algorithm_parameters(), saved)

    def test_plan_that_no_longer_matches_the_checkpoint_is_not_offered(self):
        _, worker = self._interrupted_tracking_run()
        plan_file = worker.output_dir / RUN_PLAN_FILENAME
        plan = BatchSettings.load_batch_plan(plan_file)
        for settings in plan.values():
            settings.start_time = 1.5
        BatchSettings.save_batch_plan(plan, plan_file)

        scan = scan_for_resumable_runs(self.out)
        self.assertEqual(scan.runs, [])
        self.assertEqual(len(scan.unmatched), 1)
        self.assertIn("older run", describe_scan(scan))

    def test_checkpoint_without_a_plan_is_reported_as_unmatched(self):
        _, worker = self._interrupted_tracking_run()
        (worker.output_dir / RUN_PLAN_FILENAME).unlink()
        scan = scan_for_resumable_runs(self.out)
        self.assertEqual(scan.runs, [])
        self.assertEqual(len(scan.unmatched), 1)

    def test_unusable_checkpoints_are_not_offered(self):
        _, worker = self._interrupted_tracking_run()
        (checkpoint,) = worker.checkpoint_mgr.find_all_checkpoints()
        header = json.loads(checkpoint.read_text(encoding="utf-8"))

        stale = dict(header, timestamp=header["timestamp"] - 8 * 86400)
        checkpoint.write_text(json.dumps(stale), encoding="utf-8")
        self.assertFalse(scan_for_resumable_runs(self.out))

        old_version = dict(header, version=3)
        checkpoint.write_text(json.dumps(old_version), encoding="utf-8")
        self.assertFalse(scan_for_resumable_runs(self.out))

        changed = dict(header, video_fingerprint="0" * 24)
        checkpoint.write_text(json.dumps(changed), encoding="utf-8")
        self.assertFalse(scan_for_resumable_runs(self.out))

        checkpoint.write_text("{broken", encoding="utf-8")
        self.assertFalse(scan_for_resumable_runs(self.out))

        checkpoint.write_text(json.dumps(header), encoding="utf-8")
        self.assertTrue(scan_for_resumable_runs(self.out))

    def test_pending_videos_include_unstarted_but_not_finished_ones(self):
        first_video = support.write_tracking_video(self.root)
        second = self.root / "second.avi"
        third = self.root / "third.avi"
        shutil.copy(first_video, second)
        shutil.copy(first_video, third)
        _, worker = self._interrupted_tracking_run([first_video, second, third])

        # Pretend the second video had already been exported.
        finished = worker.output_dir / "second" / "data"
        finished.mkdir(parents=True)
        (finished / "second_results.csv").write_text("x")

        (run,) = scan_for_resumable_runs(self.out).runs
        self.assertEqual([item.video for item in run.interrupted], [first_video])
        self.assertEqual(
            sorted(run.pending_videos, key=str), sorted([first_video, third], key=str)
        )

    def test_newest_matching_run_wins_over_older_runs(self):
        video, first = self._interrupted_tracking_run()
        # A later run of the same video with identical settings.
        plan = {video: support.tracking_settings()}
        second, outcome, _ = self._run(plan, Freestyle(), cancel=True)
        self.assertTrue(outcome.cancelled)
        (run,) = scan_for_resumable_runs(self.out).runs
        self.assertEqual(run.run_dir, second.output_dir)
        self.assertNotEqual(run.run_dir, first.output_dir)


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class ResumeActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        self.out = self.root / "out"
        self.out.mkdir()
        cache = patch.object(
            ReferenceFrameGenerator, "get_cache_dir",
            return_value=self.root / "cache",
        )
        cache.start()
        self.addCleanup(cache.stop)

        self.video = support.write_tracking_video(self.root)
        worker = AnalysisWorker(
            {self.video: support.tracking_settings()}, self.out, Freestyle()
        )
        _cancel_at_first_progress(worker)
        worker.run()
        (checkpoint,) = worker.checkpoint_mgr.find_all_checkpoints()
        self.checkpoint_hash = json.loads(
            checkpoint.read_text(encoding="utf-8")
        )["settings_hash"]
        self.run_dir = worker.output_dir

        self.window = MainWindow(temp_app_settings(self))
        self.addCleanup(self.window.deleteLater)
        self.addCleanup(self.window._detach_log_handler)

    def test_banner_and_menu_action_appear_for_an_interrupted_run(self):
        self.assertFalse(self.window.resume_bar.isVisibleTo(self.window))
        self.window._set_output_dir(self.out, explicit=True, persist=False)
        self.window.check_for_resumable_analyses()
        self.assertTrue(self.window.resume_bar.isVisibleTo(self.window))
        self.assertTrue(self.window.resume_button.isEnabled())
        self.assertTrue(self.window.resume_action.isEnabled())
        self.assertIn(self.run_dir.name, self.window.resume_label.text())

    def test_resume_loads_plan_and_builds_settings_with_the_checkpoint_hash(self):
        self.window._set_output_dir(self.out, explicit=True, persist=False)
        self.window.check_for_resumable_analyses()
        captured = {}

        def spy(plan, output_dir, maze, **kwargs):
            worker = AnalysisWorker(plan, output_dir, maze, **kwargs)
            captured["plan"] = worker.analysis_plan
            raise RuntimeError("stop before starting a thread")

        with patch("gui.main_window.AnalysisWorker", side_effect=spy), \
                patch.object(self.window, "_validate_analysis_plan", return_value=True), \
                patch.object(QMessageBox, "critical"):
            self.window.resume_interrupted_run()

        self.assertEqual(self.window.video_paths, [self.video])
        self.assertEqual(self.window.maze.name, "Freestyle / Open Field")
        self.assertEqual(
            self.window.batch_plan_path, self.run_dir / RUN_PLAN_FILENAME
        )
        (settings,) = captured["plan"].values()
        self.assertEqual(_hash(self.out, settings), self.checkpoint_hash)

    def test_load_settings_from_previous_run_loads_its_plan(self):
        with patch(
            "gui.main_window.QFileDialog.getExistingDirectory",
            return_value=str(self.run_dir),
        ), patch.object(QMessageBox, "information") as info:
            self.window.load_settings_from_previous_run()
        self.assertEqual(
            self.window.batch_plan_path, self.run_dir / RUN_PLAN_FILENAME
        )
        self.assertIsNotNone(self.window.current_settings)
        info.assert_called_once()

    def test_folder_without_a_plan_gets_an_explanation(self):
        empty = self.root / "old_run"
        empty.mkdir()
        with patch(
            "gui.main_window.QFileDialog.getExistingDirectory",
            return_value=str(empty),
        ), patch.object(QMessageBox, "information") as info:
            self.window.load_settings_from_previous_run()
        self.assertIsNone(self.window.batch_plan_path)
        self.assertIn(RUN_PLAN_FILENAME, info.call_args.args[2])


if __name__ == "__main__":
    unittest.main()
