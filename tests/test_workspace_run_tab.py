"""The workspace Run tab: planning, running, cancelling, closing, resuming."""
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tests import worker_support as support
from tests.workspace_run_support import (RunFixture, freestyle_settings,
                                         wait_for)
from tests.worker_support import data_files
from tests.workspace_support import HAS_RUNTIME

if HAS_RUNTIME:
    from PyQt6.QtCore import Qt
    from core.batch_runner import BatchRunOutcome
    from core.project import STATUS_DONE
    from gui.workspace import run_check
    from gui.workspace.run_check import (SCOPE_ALL, SCOPE_PENDING,
                                         SCOPE_SELECTED)
    from gui.workspace.run_tab import RunProgressModel, RunTab
    from gui.workspace.results_tab import ResultsTab
    from gui.video_list import STATUS_QUEUED, STATUS_RUNNING


class InstallTests(RunFixture):
    def test_the_tabs_are_installed(self):
        self.assertIsInstance(self.window.run_tab, RunTab)
        self.assertIsInstance(self.window.results_tab, ResultsTab)
        self.assertIs(self.run_tab.workspace, self.window)

    def test_run_is_enabled_when_a_video_is_configured(self):
        self.assertTrue(self.window.run_action.isEnabled())
        self.assertTrue(self.run_tab.run_button.isEnabled())
        self.assertFalse(self.window.cancel_action.isEnabled())

    def test_run_is_disabled_without_configured_videos(self):
        for video in list(self.project.videos):
            self.project.entry(video.key).settings = None
        self.project.set_template(None)
        self.run_tab.on_project_modified()
        self.assertFalse(self.run_tab.run_button.isEnabled())
        self.assertFalse(self.window.run_action.isEnabled())

    def test_the_default_output_folder_is_runs_beside_the_project(self):
        self.run_tab.set_output_dir(None)
        self.assertEqual(
            self.run_tab.output_dir(), self.project.base_dir / "runs"
        )

    def test_the_chosen_output_folder_is_remembered_per_project(self):
        chosen = self.root / "elsewhere"
        self.prompts.say("choose_folder", chosen)
        self.run_tab.choose_output_dir()
        self.assertEqual(self.run_tab.output_dir(), chosen)
        self.assertEqual(self.run_tab.output_edit.text(), str(chosen))
        self.assertEqual(
            self.settings.run_output_dir(self.project.file_path), chosen
        )
        # Another project does not inherit it, and reopening restores it.
        self.run_tab.on_project_changed(self.project)
        self.assertEqual(self.run_tab.output_dir(), chosen)

    def test_header_choices_are_kept_in_the_project(self):
        self.run_tab.workers_spin.setValue(3)
        self.run_tab.bins_spin.setValue(60.0)
        self.assertEqual(self.project.run_options.parallel_workers, 3)
        self.assertEqual(self.project.run_options.time_bin_seconds, 60.0)
        self.assertTrue(self.window.has_unsaved_changes)
        self.run_tab.on_project_changed(self.project)
        self.assertEqual(self.run_tab.workers_spin.value(), 3)
        self.assertEqual(self.run_tab.bins_spin.value(), 60.0)

    def test_the_sample_sheet_status_names_the_metadata(self):
        self.assertIn("No metadata", self.run_tab.sheet_label.text())
        self.project.add_metadata_column("group")
        key = self.project.keys()[0]
        self.project.set_metadata_value(key, "group", "ctl")
        self.assertIn("group", self.run_tab.sheet_label.text())
        self.assertIn("1 of 3", self.run_tab.sheet_label.text())

    def test_validation_checkbox_states(self):
        tab = self.run_tab
        self.assertIsNone(tab.validation_override())
        tab.set_validation_override(True)
        self.assertTrue(tab.validation_override())
        tab.set_validation_override(False)
        self.assertFalse(tab.validation_override())
        self.assertEqual(
            tab.validation_check.checkState(), Qt.CheckState.Unchecked
        )


class PlanningTests(RunFixture):
    """The pre-run check, without starting anything."""

    def add_unconfigured(self):
        extra = support.write_tracking_video(
            self.real_dir, 40, "mouse_new.avi"
        ).resolve()
        self.window.video_panel.add_paths([extra])
        entry = self.project.find(extra)
        self.project.entry(entry.key).settings = None
        return entry

    def test_scope_keys(self):
        project, keys = self.project, self.project.keys()
        self.assertEqual(run_check.scope_keys(project, SCOPE_ALL), keys)
        self.assertEqual(
            run_check.scope_keys(project, SCOPE_SELECTED, [keys[2], keys[0]]),
            [keys[0], keys[2]],
        )
        project.entry(keys[1]).last_status = STATUS_DONE
        self.assertEqual(
            run_check.scope_keys(project, SCOPE_PENDING), [keys[0], keys[2]]
        )

    def test_unconfigured_selected_video_blocks_the_run(self):
        entry = self.add_unconfigured()
        self.select(entry.name)
        self.run_tab.set_scope(SCOPE_SELECTED)
        self.assertFalse(self.run_tab.start_run())
        (error,) = self.prompts.called("error")
        self.assertIn("Cannot start", error[0])
        self.assertIn("no settings", error[1])
        self.assertIn("mouse_new.avi", error[1])
        self.assertFalse(self.run_tab.is_running)
        self.assertFalse(self.window.is_busy)

    def test_all_scope_leaves_unconfigured_videos_out_with_a_warning(self):
        self.add_unconfigured()
        report = self.run_tab.prepare(self.project.keys(), strict=False)
        self.assertTrue(report.ok, report.issues)
        self.assertEqual(len(report.plan), 3)
        self.assertEqual([p.name for p in report.skipped], ["mouse_new.avi"])
        self.assertTrue(any("left out" in w for w in report.warnings))

    def test_no_configured_video_blocks_the_run(self):
        for video in list(self.project.videos):
            self.project.entry(video.key).settings = None
        self.project.set_template(None)
        report = run_check.prepare_run(
            self.project, self.project.keys(), self.root / "o", preflight=False
        )
        self.assertFalse(report.ok)
        self.assertIn("settings", report.issues[0])

    def test_a_missing_video_file_blocks_the_run(self):
        self.real_videos[1].unlink()
        report = run_check.prepare_run(
            self.project, self.project.keys(), self.root / "o", preflight=False
        )
        self.assertFalse(report.ok)
        self.assertTrue(any("not found" in i for i in report.issues))

    def test_reserved_metadata_columns_block_the_run(self):
        self.project.add_metadata_column("total_distance")
        report = run_check.prepare_run(
            self.project, self.project.keys(), self.root / "o", preflight=False
        )
        self.assertFalse(report.ok)
        self.assertTrue(
            any("total_distance" in i and "clash" in i for i in report.issues)
        )

    def test_a_paradigm_mismatch_blocks_the_run(self):
        entry = self.project.entry(self.project.keys()[0])
        entry.settings = freestyle_settings()
        entry.settings.paradigm_id = "Y-Maze"
        report = run_check.prepare_run(
            self.project, self.project.keys(), self.root / "o", preflight=False
        )
        self.assertFalse(report.ok)
        self.assertTrue(any("Y-Maze" in i for i in report.issues))

    def test_maze_validation_problems_block_the_run(self):
        from mazes.freestyle import Freestyle
        with patch.object(
            Freestyle, "validate_settings", return_value=["no arena drawn"]
        ):
            report = run_check.prepare_run(
                self.project, self.project.keys(), self.root / "o",
                preflight=False,
            )
        self.assertFalse(report.ok)
        self.assertEqual(
            sum("mouse_a.avi: no arena drawn" in i for i in report.issues), 1
        )

    def test_the_validation_checkbox_applies_to_this_run_only(self):
        keys = self.project.keys()
        self.project.entry(keys[0]).settings.create_timelapse = False
        for choice, expected in ((None, [False, True, True]),
                                 (True, [True, True, True]),
                                 (False, [False, False, False])):
            report = run_check.prepare_run(
                self.project, keys, self.root / "o", preflight=False,
                create_validation_video=choice,
            )
            self.assertEqual(
                [s.create_timelapse for s in report.plan.values()], expected
            )
        self.assertFalse(self.project.entry(keys[0]).settings.create_timelapse)
        self.assertTrue(self.project.entry(keys[1]).settings.create_timelapse)

    def test_preflight_warnings_are_listed_and_need_confirmation(self):
        self.prompts.say("confirm", False)
        self.assertFalse(self.run_tab.start_run())
        (call,) = self.prompts.called("confirm")
        self.assertEqual(call[0], "Ready with warnings")
        self.assertIn("calibration", call[1])
        self.assertFalse(self.run_tab.is_running)

    def test_nothing_selected_is_reported(self):
        self.run_tab.set_scope(SCOPE_SELECTED)
        self.window.video_panel.clear_selection()
        self.assertFalse(self.run_tab.start_run())
        self.assertEqual(len(self.prompts.called("info")), 1)


class RunTests(RunFixture):
    def test_a_run_produces_the_same_files_as_the_cli(self):
        with self.isolated_environment("ws"):
            outcome = self.start_and_wait()
        self.assertEqual(outcome.failures, [])
        ws_dir = outcome.output_dir
        cli_dir = self.run_cli("cli", "--workers", "1")

        expected, actual = data_files(cli_dir), data_files(ws_dir)
        self.assertEqual(sorted(actual), sorted(expected))
        self.assertIn("batch_summary.csv", expected)
        self.assertIn("run_plan.json", expected)
        for name in expected:
            self.assertEqual(actual[name], expected[name], name)
        self.assertTrue((ws_dir / "manifest.json").is_file())

    def test_a_parallel_run_produces_the_same_files_as_the_cli(self):
        self.project.set_run_options(parallel_workers=3)
        with self.isolated_environment("ws"):
            outcome = self.start_and_wait(timeout=180.0)
        cli_dir = self.run_cli("cli", "--workers", "3")
        expected, actual = data_files(cli_dir), data_files(outcome.output_dir)
        self.assertEqual(sorted(actual), sorted(expected))
        for name in expected:
            self.assertEqual(actual[name], expected[name], name)

    def test_the_run_is_recorded_in_the_project_and_the_table(self):
        outcome = self.start_and_wait()
        project = self.project
        self.assertEqual(len(project.runs), 1)
        self.assertEqual(project.runs[0].run_dir, outcome.output_dir.resolve())
        self.assertEqual(project.runs[0].completed, 3)
        self.assertEqual(
            {v.last_status for v in project.videos}, {STATUS_DONE}
        )
        model = self.window.video_panel.model
        shown = {model.index(r, 2).data() for r in range(model.rowCount())}
        self.assertTrue(shown, "the table has rows")
        self.assertTrue(self.window.has_unsaved_changes)

    def test_busy_while_running_and_idle_afterwards(self):
        self.prompts.say("confirm", True)
        busy_states = []
        self.assertTrue(self.run_tab.start_run())
        busy_states.append(
            (self.window.is_busy, self.window.cancel_action.isEnabled(),
             self.window.video_panel.is_read_only(),
             self.run_tab.run_button.isEnabled(),
             self.run_tab.cancel_button.isEnabled())
        )
        self.assertEqual(busy_states, [(True, True, True, False, True)])
        wait_for(lambda: not self.run_tab.is_running, 120.0, "the run to end")
        self.assertFalse(self.window.is_busy)
        self.assertFalse(self.window.cancel_action.isEnabled())
        self.assertFalse(self.window.video_panel.is_read_only())
        self.assertTrue(self.run_tab.run_button.isEnabled())

    def test_the_progress_table_follows_the_videos(self):
        seen = {}
        self.prompts.say("confirm", True)
        model = self.run_tab.progress_model
        self.assertTrue(self.run_tab.start_run())
        self.assertEqual(
            [model.index(r, 1).data() for r in range(3)], ["Queued"] * 3
        )

        def poll():
            for row in range(model.rowCount()):
                state = model.state(model.paths()[row])
                seen.setdefault(model.paths()[row].name, set()).add(
                    state["status"]
                )
            return not self.run_tab.is_running

        wait_for(poll, 120.0, "the run to end")
        for statuses in seen.values():
            self.assertIn("done", statuses)
        for row in range(3):
            self.assertEqual(model.index(row, 1).data(), "Done")
            self.assertEqual(model.index(row, 2).data(), "100%")
        self.assertEqual(self.run_tab.progress_bar.value(), 100)
        self.assertEqual(self.run_tab.status_label.text(), "Finished")

    def test_the_log_is_filterable(self):
        self.start_and_wait()
        tab = self.run_tab
        self.assertIn("Batch Analysis Complete", tab.log_text())
        tab.log_filter_edit.setText("zzz-no-such-text")
        self.assertEqual(tab.log_text(), "")
        tab.log_filter_edit.setText("processed")
        self.assertIn("Processed: 3/3", tab.log_text())
        tab.log_filter_edit.setText("")
        tab.log_level_combo.setCurrentIndex(
            tab.log_level_combo.findText("Error")
        )
        self.assertNotIn("Batch Analysis Complete", tab.log_text())

    def test_the_completion_summary_opens_results_and_folder(self):
        outcome = self.start_and_wait()
        tab = self.run_tab
        self.assertFalse(tab.summary_frame.isHidden())
        self.assertIn("Processed: 3", tab.summary_label.text())
        self.assertIn(str(outcome.output_dir), tab.summary_label.text())
        self.assertTrue(tab.open_folder_button.isEnabled())
        self.assertTrue(tab.open_run_folder())
        self.assertEqual(self.prompts.called("open_folder"), [(outcome.output_dir,)])
        self.assertTrue(tab.open_results())
        self.assertEqual(self.window.current_tab_name(), "results")
        self.assertEqual(
            self.results_tab.run.run_dir.resolve(), outcome.output_dir.resolve()
        )

    def test_the_scope_selection_is_honoured(self):
        keys = self.project.keys()
        self.select("mouse_b.avi")
        self.run_tab.set_scope(SCOPE_SELECTED)
        outcome = self.start_and_wait()
        self.assertEqual(
            [path.name for path, _ in outcome.results], ["mouse_b.avi"]
        )
        statuses = {v.name: v.last_status for v in self.project.videos}
        self.assertEqual(statuses["mouse_b.avi"], STATUS_DONE)
        self.assertNotEqual(statuses["mouse_a.avi"], STATUS_DONE)

        # "Not yet done" now runs the other two.
        self.run_tab.set_scope(SCOPE_PENDING)
        outcome = self.start_and_wait()
        self.assertEqual(
            sorted(path.name for path, _ in outcome.results),
            ["mouse_a.avi", "mouse_c.avi"],
        )
        self.assertEqual(
            {v.last_status for v in self.project.videos}, {STATUS_DONE}
        )
        self.assertEqual(len(self.project.runs), 2)
        self.assertEqual(keys, self.project.keys())

    def test_all_scope_runs_every_configured_video(self):
        self.run_tab.set_scope(SCOPE_ALL)
        outcome = self.start_and_wait()
        self.assertEqual(len(outcome.results), 3)

    def test_the_time_bins_and_metadata_reach_the_outputs(self):
        self.project.add_metadata_column("group")
        for key, group in zip(self.project.keys(), ("ctl", "ctl", "trt")):
            self.project.set_metadata_value(key, "group", group)
        self.project.set_run_options(time_bin_seconds=1.0)
        outcome = self.start_and_wait()
        run_dir = outcome.output_dir
        summary = (run_dir / "batch_summary.csv").read_text(encoding="utf-8")
        self.assertIn("group", summary.splitlines()[0])
        self.assertTrue((run_dir / "batch_time_bins.csv").is_file())

    def test_the_validation_checkbox_switches_the_validation_videos(self):
        self.run_tab.set_validation_override(False)
        outcome = self.start_and_wait()
        self.assertEqual(
            list(outcome.output_dir.rglob("*.mp4")), []
        )
        self.run_tab.set_validation_override(True)
        outcome = self.start_and_wait()
        self.assertEqual(len(list(outcome.output_dir.rglob("*.mp4"))), 3)


class LongRunFixture(RunFixture):
    """A run slow enough that a cancel always lands mid-tracking.

    The synthetic videos are tracked in well under a second, so every frame
    is slowed a little (the estimator is updated once per tracked frame).
    """

    FRAMES = {"long_a.avi": 1000, "long_b.avi": 1000}
    FRAME_DELAY_S = 0.003

    def setUp(self):
        super().setUp()
        from checkpoint_manager import ProgressEstimator
        original = ProgressEstimator.update
        delay = self.FRAME_DELAY_S

        def slow_update(estimator, current_frame):
            time.sleep(delay)
            return original(estimator, current_frame)

        patcher = patch.object(ProgressEstimator, "update", slow_update)
        patcher.start()
        self.addCleanup(patcher.stop)

    def wait_until_running(self):
        model = self.run_tab.progress_model
        wait_for(
            lambda: any(
                (model.state(p) or {}).get("status") == STATUS_RUNNING
                and "Frame" in (model.state(p) or {}).get("message", "")
                for p in model.paths()
            ),
            90.0, "a video to be tracked",
        )


class CancelTests(LongRunFixture):
    def test_declining_the_confirmation_keeps_the_run_going(self):
        self.prompts.say("confirm", True)          # pre-flight warnings
        self.assertTrue(self.run_tab.start_run())
        self.wait_until_running()
        self.prompts.say("confirm", False)
        self.assertFalse(self.run_tab.request_cancel())
        self.assertTrue(self.run_tab.is_running)
        self.prompts.say("confirm", True)
        self.assertTrue(self.run_tab.request_cancel())
        wait_for(lambda: not self.run_tab.is_running, 120.0, "the run to stop")

    def test_cancel_stops_the_run_and_keeps_a_checkpoint(self):
        finished = []
        self.run_tab.run_finished.connect(finished.append)
        self.prompts.say("confirm", True)
        self.assertTrue(self.run_tab.start_run())
        self.wait_until_running()
        self.prompts.say("confirm", True)
        self.window.cancel_action.trigger()        # the toolbar action
        wait_for(lambda: bool(finished), 120.0, "the run to stop")
        outcome = finished[0]
        self.assertTrue(outcome.cancelled)
        self.assertFalse(self.window.is_busy)
        self.assertEqual(self.run_tab.status_label.text(),
                         "Cancelled - checkpoint saved")
        self.assertIn("cancelled", self.run_tab.summary_label.text())
        # The cancelled run is still recorded; its videos are interrupted.
        self.assertEqual(len(self.project.runs), 1)
        self.assertTrue(self.project.runs[0].cancelled)
        self.assertEqual(
            {v.last_status for v in self.project.videos}, {"interrupted"}
        )
        checkpoints = list(
            (self.root / "ws_out" / ".checkpoints").glob("*_checkpoint.json")
        )
        self.assertTrue(checkpoints)

    def test_an_interrupted_run_can_be_resumed(self):
        finished = []
        self.run_tab.run_finished.connect(finished.append)
        self.prompts.say("confirm", True)
        self.assertTrue(self.run_tab.start_run())
        self.wait_until_running()
        self.prompts.say("confirm", True)
        self.assertTrue(self.run_tab.request_cancel())
        wait_for(lambda: bool(finished), 120.0, "the run to stop")
        self.assertTrue(finished[0].cancelled)

        offer = self.run_tab.resume_offer
        self.assertTrue(offer)
        self.assertFalse(self.run_tab.resume_button.isHidden())
        self.assertIn("interrupted", self.run_tab.resume_label.text())
        # Changing what the next run would use makes the checkpoint stale.
        self.run_tab.set_validation_override(False)
        self.assertFalse(self.run_tab.resume_offer)
        self.run_tab.set_validation_override(None)
        self.assertTrue(self.run_tab.resume_offer)

        finished.clear()
        self.prompts.say("confirm", True)
        self.assertTrue(self.run_tab.start_resume())
        wait_for(lambda: bool(finished), 240.0, "the resumed run to finish")
        outcome = finished[0]
        self.assertFalse(outcome.cancelled)
        self.assertEqual(outcome.failures, [])
        self.assertGreaterEqual(len(outcome.results), 1)
        # The interrupted video was finished from its checkpoint.
        self.assertIn("long_a.avi", [p.name for p, _ in outcome.results])
        self.assertIn(
            self.project.entry(self.project.keys()[0]).last_status,
            (STATUS_DONE, "warning"),    # the synthetic video leaves the frame
        )
        self.assertFalse(self.run_tab.resume_offer)


class CloseTests(LongRunFixture):
    def test_closing_during_a_run_cancels_first_and_closes_afterwards(self):
        finished = []
        self.run_tab.run_finished.connect(finished.append)
        self.prompts.say("confirm", True)
        self.assertTrue(self.run_tab.start_run())
        self.wait_until_running()

        self.prompts.say("confirm", True)          # Quit and cancel?
        self.assertFalse(self.window.close())
        self.assertFalse(self.window._closed)
        self.assertTrue(self.run_tab._close_requested)
        self.assertTrue(self.run_tab._force_timer.isActive())
        self.assertIs(self.run_tab._force_timer.parent(), self.run_tab)
        self.assertEqual(len(self.prompts.called("confirm")), 2)

        # A second close request while stopping is ignored, not re-asked.
        self.assertFalse(self.window.close())
        self.assertEqual(len(self.prompts.called("confirm")), 2)

        self.prompts.say("ask_save_changes", "discard")
        wait_for(lambda: self.window._closed, 120.0, "the window to close")
        self.assertTrue(finished[0].cancelled)
        self.assertFalse(self.run_tab._force_timer.isActive())
        self.assertEqual(len(self.project.runs), 1)

    def test_declining_keeps_the_run_and_the_window(self):
        self.prompts.say("confirm", True)
        self.assertTrue(self.run_tab.start_run())
        self.wait_until_running()
        self.prompts.say("confirm", False)
        self.assertFalse(self.window.close())
        self.assertFalse(self.run_tab._close_requested)
        self.assertTrue(self.run_tab.is_running)
        self.assertFalse(self.window._closed)
        self.prompts.say("confirm", True)
        self.run_tab.request_cancel()
        wait_for(lambda: not self.run_tab.is_running, 120.0, "the run to stop")

    def test_the_project_cannot_be_closed_during_a_run(self):
        self.prompts.say("confirm", True)
        self.assertTrue(self.run_tab.start_run())
        self.assertFalse(self.window.close_project())
        self.assertFalse(self.window.new_project())
        self.prompts.say("confirm", True)
        self.run_tab.request_cancel()
        wait_for(lambda: not self.run_tab.is_running, 120.0, "the run to stop")


class FakeThread:
    def __init__(self, stops_on_wait=True):
        self.running = True
        self.stops_on_wait = stops_on_wait
        self.calls = []

    def isRunning(self):
        return self.running

    def quit(self):
        self.calls.append("quit")
        if self.stops_on_wait:
            self.running = False

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))
        return not self.running

    def terminate(self):
        self.calls.append("terminate")
        self.running = False


class ForceQuitTests(RunFixture):
    """The close and force-quit logic, with a stand-in thread."""

    def setUp(self):
        super().setUp()
        self.thread = FakeThread()
        self.worker = Mock()
        self.run_tab._thread = self.thread
        self.run_tab._worker = self.worker
        self.addCleanup(self.release_fakes)
        self.run_tab.FORCE_QUIT_DELAY_MS = 40

    def release_fakes(self):
        self.run_tab._force_timer.stop()
        self.run_tab._thread = None
        self.run_tab._worker = None

    def test_closing_asks_cancels_once_and_arms_the_force_quit_timer(self):
        self.prompts.say("confirm", True)
        self.assertFalse(self.window.close())
        self.worker.stop.assert_called_once()
        self.assertTrue(self.run_tab._force_timer.isActive())
        self.assertEqual(self.thread.calls, [])        # never blocks the GUI

    def test_finishing_while_closing_closes_the_window(self):
        self.prompts.say("confirm", True)
        self.window.close()
        self.prompts.say("ask_save_changes", "discard")
        run_dir = self.root / "ws_out" / "analysis_run_fake"
        run_dir.mkdir(parents=True)
        (run_dir / "run_plan.json").write_text('{"plan": {}}')
        self.run_tab._on_finished(
            BatchRunOutcome(cancelled=True, output_dir=run_dir)
        )
        wait_for(lambda: self.window._closed, 10.0, "the window to close")
        self.assertEqual(self.thread.calls[0], "quit")
        self.assertFalse(self.run_tab._force_timer.isActive())

    def test_force_quit_after_the_timeout_closes_the_window(self):
        self.prompts.say("confirm", True)       # quit and cancel
        self.window.close()
        self.prompts.say("confirm", True)       # force quit
        self.prompts.say("ask_save_changes", "discard")
        wait_for(lambda: self.window._closed, 10.0, "the force quit")
        self.assertIn("quit", self.thread.calls)
        self.assertNotIn("terminate", self.thread.calls)

    def test_force_quit_terminates_a_thread_that_will_not_stop(self):
        self.thread.stops_on_wait = False
        self.prompts.say("confirm", True)
        self.window.close()
        self.prompts.say("confirm", True)
        self.prompts.say("ask_save_changes", "discard")
        wait_for(lambda: self.window._closed, 10.0, "the force quit")
        self.assertIn("terminate", self.thread.calls)

    def test_keep_waiting_asks_again(self):
        self.prompts.say("confirm", True)
        self.window.close()
        self.prompts.say("confirm", False, False)
        wait_for(
            lambda: len(self.prompts.called("confirm")) >= 3, 10.0,
            "a second force-quit offer",
        )
        self.assertFalse(self.window._closed)

    def test_the_force_quit_offer_is_skipped_once_the_run_ended(self):
        self.prompts.say("confirm", True)
        self.window.close()
        self.thread.running = False
        from PyQt6.QtTest import QTest
        QTest.qWait(150)
        self.assertEqual(len(self.prompts.called("confirm")), 1)

    def test_the_timers_belong_to_the_tab(self):
        self.assertIs(self.run_tab._close_timer.parent(), self.run_tab)
        self.assertIs(self.run_tab._force_timer.parent(), self.run_tab)


class ProgressModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not HAS_RUNTIME:
            raise unittest.SkipTest("GUI runtime is not installed")
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.model = RunProgressModel()
        self.paths = [Path("a.avi"), Path("b.avi")]
        self.model.reset_rows(self.paths)

    def test_rows_start_queued(self):
        self.assertEqual(self.model.rowCount(), 2)
        self.assertEqual(self.model.state(self.paths[0])["status"], STATUS_QUEUED)
        self.assertEqual(self.model.index(0, 1).data(), "Queued")

    def test_progress_marks_the_video_running_and_never_goes_back(self):
        self.model.set_progress(self.paths[0], 40, "1m 02s", "Tracking")
        self.model.set_progress(self.paths[0], 25, "", "Still tracking")
        state = self.model.state(self.paths[0])
        self.assertEqual(
            (state["status"], state["pct"], state["eta"], state["message"]),
            (STATUS_RUNNING, 40, "1m 02s", "Still tracking"),
        )
        self.assertEqual(self.model.index(0, 2).data(), "40%")
        self.assertEqual(self.model.index(0, 3).data(), "1m 02s")

    def test_unknown_percentage_is_indeterminate(self):
        self.model.set_progress(self.paths[1], None, "", "Working")
        self.assertEqual(self.model.index(1, 2).data(), "...")

    def test_done_is_final(self):
        self.model.set_status(self.paths[0], "done")
        self.model.set_progress(self.paths[0], 10, "5s", "late")
        state = self.model.state(self.paths[0])
        self.assertEqual((state["status"], state["pct"], state["eta"]),
                         ("done", 100, ""))

    def test_failed_keeps_its_message(self):
        self.model.set_status(self.paths[0], "failed", "cannot open the video")
        self.assertEqual(self.model.index(0, 4).data(), "cannot open the video")


class RunOptionTests(unittest.TestCase):
    """Project.set_run_options and the remembered output folders (no Qt UI)."""

    def test_set_run_options_changes_and_notifies_once(self):
        from core.project import Project, ProjectError
        project = Project()
        calls = []
        project.add_listener(lambda: calls.append(1))
        self.assertTrue(project.set_run_options(
            parallel_workers=2, time_bin_seconds=30.0,
            create_validation_video=True,
        ))
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            (project.run_options.parallel_workers,
             project.run_options.time_bin_seconds,
             project.run_options.create_validation_video),
            (2, 30.0, True),
        )
        self.assertFalse(project.set_run_options(parallel_workers=2))
        self.assertEqual(len(calls), 1)
        with self.assertRaises(ProjectError):
            project.set_run_options(parallel_workers=-1)
        with self.assertRaises(ProjectError):
            project.set_run_options(time_bin_seconds=-5)
        self.assertEqual(project.run_options.parallel_workers, 2)

    @unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
    def test_run_output_dirs_are_remembered_per_project(self):
        import tempfile
        from gui.app_settings import AppSettings
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            settings = AppSettings.for_file(Path(tmp) / "s.ini")
            first, second = Path(tmp) / "a.bmaproj", Path(tmp) / "b.bmaproj"
            self.assertIsNone(settings.run_output_dir(first))
            settings.set_run_output_dir(first, Path(tmp) / "out1")
            settings.set_run_output_dir(second, Path(tmp) / "out2")
            self.assertEqual(settings.run_output_dir(first), Path(tmp) / "out1")
            self.assertEqual(settings.run_output_dir(second), Path(tmp) / "out2")
            settings.set_run_output_dir(first, None)
            self.assertIsNone(settings.run_output_dir(first))
            self.assertEqual(settings.run_output_dir(second), Path(tmp) / "out2")


if __name__ == "__main__":
    unittest.main()
