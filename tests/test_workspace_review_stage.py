"""The Review & run stage: the checklist, the options, blocked runs, the
progress view, cancel, resume and the finished-run summary."""
import json
import unittest
from pathlib import Path

from tests.test_workspace_run_tab import LongRunFixture
from tests.workspace_run_support import RunFixture, wait_for
from tests.workspace_support import HAS_RUNTIME

if HAS_RUNTIME:
    from core.batch_runner import BatchRunOutcome
    from core.project import START_PER_VIDEO
    from gui.workspace.run_check import SCOPE_PENDING, SCOPE_SELECTED
    from gui.workspace.stages.review_stage import (PAGE_EMPTY, PAGE_RUN,
                                                   PAGE_SETUP)


def fake_run_folder(root: Path) -> Path:
    run_dir = root / "ws_out" / "analysis_run_fake"
    run_dir.mkdir(parents=True)
    (run_dir / "run_plan.json").write_text('{"plan": {}}')
    return run_dir


class ChecklistTests(RunFixture):
    def test_one_row_per_stage(self):
        rows = self.review.check_rows()
        self.assertEqual(
            [(r.stage_id, r.title_label.text()) for r in rows],
            [("videos", "Videos"), ("arena", "Arena & zones"),
             ("detection", "Detection"), ("trials", "Trials")],
        )

    def test_a_done_row_shows_its_reason_and_no_fix(self):
        row = self.review.check_row("videos")
        self.assertEqual(row.chip.kind(), "ok")
        self.assertIn("3 videos", row.reason_label.text())
        self.assertTrue(row.fix_button.isHidden())

    def test_the_unreviewed_detection_offers_a_fix_without_blocking(self):
        row = self.review.check_row("detection")
        self.assertEqual(row.chip.text(), "Not reviewed")
        self.assertFalse(row.fix_button.isHidden())
        self.assertTrue(self.review.run_button.isEnabled())

    def test_the_fix_link_opens_the_stage(self):
        row = self.review.check_row("detection")
        row.fix_button.click()
        self.assertEqual(self.window.current_stage_id, "detection")

    def test_the_fix_links_of_every_stage_go_to_that_stage(self):
        self.project.set_trial_protocol(300.0, START_PER_VIDEO)
        self.project.set_video_info(self.project.keys()[0], 10.0, (1280, 720))
        self.window.refresh_navigation()
        for stage_id in ("arena", "trials"):
            with self.subTest(stage_id):
                self.window.go_to_stage("review")
                row = self.review.check_row(stage_id)
                self.assertEqual(row.chip.kind(), "error", stage_id)
                self.assertFalse(row.fix_button.isHidden())
                row.fix_button.click()
                self.assertEqual(self.window.current_stage_id, stage_id)

    def test_the_checklist_follows_the_project(self):
        self.window.go_to_stage("review")
        self.assertEqual(self.review.check_row("trials").chip.kind(), "ok")
        self.project.set_trial_protocol(300.0, START_PER_VIDEO)
        self.assertEqual(self.review.check_row("trials").chip.kind(), "error")
        self.assertIn("need a start time", self.review.check_row("trials").reason_label.text())


class BlockedRunTests(RunFixture):
    def need_starts(self):
        self.project.set_trial_protocol(300.0, START_PER_VIDEO)

    def test_problems_are_listed_by_kind_with_their_videos(self):
        self.need_starts()
        (row,) = self.review.problem_rows()
        self.assertEqual(row.title_label.text(), "Start time needed")
        self.assertEqual(row.chip.text(), "3 videos")
        self.assertIn("mouse_a.avi", row.names_label.text())
        self.assertFalse(self.review.problems.isHidden())

    def test_the_run_button_is_disabled_and_says_why(self):
        self.need_starts()
        button = self.review.run_button
        self.assertFalse(button.isEnabled())
        self.assertIn("Start time needed: 3 videos", button.toolTip())
        self.assertIn("Not ready", self.review.ready_label.text())
        self.assertIn("Start time needed", self.review.detail_label.text())

    def test_the_problem_fix_link_goes_where_it_is_fixed(self):
        self.need_starts()
        self.review.problem_rows()[0].fix_button.click()
        self.assertEqual(self.window.current_stage_id, "trials")

    def test_starting_anyway_lists_the_problems_and_runs_nothing(self):
        self.need_starts()
        self.assertFalse(self.review.start_run())
        (error,) = self.prompts.called("error")
        self.assertIn("Cannot start", error[0])
        self.assertIn("needs its start time", error[1])
        self.assertFalse(self.review.is_running)
        self.assertFalse(self.window.is_busy)

    def test_clicking_a_disabled_run_button_does_nothing(self):
        self.need_starts()
        self.review.run_button.click()
        self.assertEqual(self.prompts.called("error"), [])
        self.assertFalse(self.review.is_running)

    def test_the_shortcut_is_ignored_while_blocked(self):
        self.need_starts()
        self.review._on_shortcut()
        self.assertFalse(self.review.is_running)

    def test_fixing_the_problem_enables_the_button_again(self):
        self.need_starts()
        for key in self.project.keys():
            self.project.set_trial(key, start=1.0)
        self.assertTrue(self.review.run_button.isEnabled())
        self.assertEqual(self.review.run_button.toolTip().split(";")[0],
                         "Analyse 3 videos")
        self.assertEqual(self.review.problem_rows(), [])
        self.assertTrue(self.review.problems.isHidden())

    def test_a_missing_video_blocks_and_points_to_the_videos(self):
        self.real_videos[1].unlink()
        self.window.refresh_navigation()
        self.review.refresh()
        self.assertFalse(self.review.run_button.isEnabled())
        row = self.review.problem_rows()[0]
        self.assertEqual(row.title_label.text(), "Video files not found")
        row.fix_button.click()
        self.assertEqual(self.window.current_stage_id, "videos")

    def test_a_video_without_settings_is_only_a_warning_for_all_videos(self):
        entry = self.project.entry(self.project.keys()[2])
        entry.settings = None
        self.window.refresh_navigation()
        self.review.refresh()
        self.assertTrue(self.review.run_button.isEnabled())
        (row,) = self.review.problem_rows()
        self.assertEqual(row.chip.kind(), "warning")
        self.assertIn("2 videos will be analysed", self.review.scope_label.text())
        self.assertIn("1 without settings", self.review.scope_label.text())
        self.assertEqual(self.review.problems.title(), "Things to know")

    def test_selected_videos_without_settings_block(self):
        entry = self.project.entry(self.project.keys()[2])
        entry.settings = None
        self.window.refresh_navigation()
        self.select(entry.name)
        self.review.set_scope(SCOPE_SELECTED)
        self.assertFalse(self.review.run_button.isEnabled())
        self.assertEqual(self.review.problems.title(), "Problems to fix")

    def test_nothing_selected_says_so_next_to_the_scope(self):
        self.window.video_panel.clear_selection()
        self.review.set_scope(SCOPE_SELECTED)
        self.assertFalse(self.review.run_button.isEnabled())
        self.assertIn("Select videos in the Videos stage", self.review.scope_label.text())

    def test_the_run_action_while_blocked_goes_to_the_stage_and_explains(self):
        self.need_starts()
        self.window.go_to_stage("videos")
        self.window.run_action.trigger()
        self.assertEqual(self.window.current_stage_id, "review")
        self.assertFalse(self.review.is_running)
        self.assertEqual(self.prompts.called("error"), [])
        self.assertIn("Start time needed", self.window.footer.status_label.text())

    def test_the_footer_tells_what_is_missing(self):
        self.need_starts()
        text, chip = self.review.footer_text()
        self.assertIn("Start time needed", text)
        self.assertEqual(chip[0], "warning")


class EmptyStateTests(RunFixture):
    def test_without_a_project_the_stage_says_so(self):
        self.prompts.say("ask_save_changes", "discard")
        self.assertTrue(self.window.close_project())
        self.window.go_to_stage("review", force=True)
        self.assertEqual(self.review.pages.currentIndex(), PAGE_EMPTY)
        self.assertIn("No project", self.review.empty.title())
        self.assertFalse(self.window.run_action.isEnabled())

    def test_the_empty_state_leads_to_the_videos(self):
        self.prompts.say("ask_save_changes", "discard")
        self.window.close_project()
        self.window.go_to_stage("review", force=True)
        self.review.empty.buttons[0].click()
        self.assertEqual(self.window.current_stage_id, "videos")


class OptionTests(RunFixture):
    def test_the_options_are_on_the_setup_page(self):
        self.assertEqual(self.review.pages.currentIndex(), PAGE_SETUP)
        review = self.review
        for widget in (review.output_edit, review.output_button, review.workers_spin,
                       review.bins_spin, review.scope_combo, review.sheet_button,
                       review.sheet_clear_button, review.validation_check,
                       review.run_button):
            self.assertTrue(widget.isEnabled() or widget is review.sheet_clear_button)

    def test_auto_and_off_are_spelled_out(self):
        self.assertEqual(self.review.workers_spin.specialValueText(), "Auto")
        self.assertEqual(self.review.bins_spin.specialValueText(), "Off")
        self.assertEqual(self.review.bins_spin.suffix(), " min")
        self.assertEqual(self.review.workers_spin.value(), 1)    # the fixture's choice

    def test_time_bins_are_minutes_in_the_screen_and_seconds_in_the_project(self):
        self.review.bins_spin.setValue(2.0)
        self.assertEqual(self.project.run_options.time_bin_seconds, 120.0)
        self.review.bins_spin.setValue(0.0)
        self.assertEqual(self.project.run_options.time_bin_seconds, 0.0)
        self.project.set_run_options(time_bin_seconds=30.0)
        self.assertEqual(self.review.bins_spin.value(), 0.5)

    def test_the_scope_choices_are_all_pending_and_selected(self):
        combo = self.review.scope_combo
        self.assertEqual(
            [combo.itemData(i) for i in range(combo.count())],
            ["all", "selected", "pending"],
        )
        self.review.set_scope(SCOPE_PENDING)
        self.assertEqual(self.review.scope(), SCOPE_PENDING)

    def test_the_sample_sheet_can_be_loaded_and_cleared(self):
        sheet = self.root / "sheet.csv"
        sheet.write_text("video,group\nmouse_a.avi,ctl\nmouse_b.avi,ctl\nmouse_c.avi,trt\n")
        self.prompts.say("open_file", sheet)
        self.review.sheet_button.click()
        self.assertIn("group", self.review.sheet_label.text())
        self.assertTrue(self.review.sheet_clear_button.isEnabled())
        self.assertEqual(self.project.run_options.sample_sheet, sheet.resolve())
        self.prompts.say("confirm", False)
        self.review.sheet_clear_button.click()
        self.assertIn("group", self.review.sheet_label.text())
        self.prompts.say("confirm", True)
        self.review.sheet_clear_button.click()
        self.assertIn("No metadata", self.review.sheet_label.text())
        self.assertIsNone(self.project.run_options.sample_sheet)
        self.assertEqual(self.project.metadata_columns(), [])
        self.assertFalse(self.review.sheet_clear_button.isEnabled())

    def test_the_output_folder_button_asks_for_a_folder(self):
        chosen = self.root / "elsewhere"
        self.prompts.say("choose_folder", chosen)
        self.review.output_button.click()
        self.assertEqual(self.review.output_edit.text(), str(chosen))
        self.prompts.say("choose_folder", None)
        self.review.output_button.click()
        self.assertEqual(self.review.output_dir(), chosen)

    def test_every_disabled_option_says_why_while_running(self):
        controller = self.review.controller
        controller._checking = True          # the pre-run check is "busy" too
        self.review._update_controls()
        for widget in (self.review.scope_combo, self.review.workers_spin,
                       self.review.bins_spin, self.review.validation_check,
                       self.review.output_button, self.review.sheet_button):
            self.assertFalse(widget.isEnabled())
            self.assertIn("Locked", widget.toolTip())
        self.assertFalse(self.review.run_button.isEnabled())
        self.assertIn("already in progress", self.review.run_button.toolTip())
        controller._checking = False
        self.review._update_controls()
        self.assertTrue(self.review.run_button.isEnabled())

    def test_the_footer_says_what_will_be_analysed(self):
        text, chip = self.review.footer_text()
        self.assertEqual(text, "3 videos will be analysed")
        self.assertEqual(chip, ("ok", "Ready"))
        self.assertEqual(self.review.can_advance(), (True, ""))


class ProgressViewTests(RunFixture):
    def test_the_same_stage_switches_to_the_progress_view_and_back(self):
        review = self.review
        self.prompts.say("confirm", True)
        self.assertEqual(review.pages.currentIndex(), PAGE_SETUP)
        self.assertTrue(review.run_button.isEnabled())
        review.run_button.click()
        self.assertEqual(review.pages.currentIndex(), PAGE_RUN)
        self.assertFalse(review.cancel_button.isHidden())
        self.assertTrue(review.cancel_button.isEnabled())
        self.assertTrue(review.summary_card.isHidden())
        self.assertEqual(self.window.current_stage_id, "review")
        wait_for(lambda: not review.is_running, 120.0, "the run to end")
        # Finished: the summary card, no Cancel, the table still there.
        self.assertEqual(review.pages.currentIndex(), PAGE_RUN)
        self.assertFalse(review.summary_card.isHidden())
        self.assertTrue(review.cancel_button.isHidden())
        self.assertEqual(review.progress_model.rowCount(), 3)
        review.back_button.click()
        self.assertEqual(review.pages.currentIndex(), PAGE_SETUP)
        self.assertTrue(review.summary_card.isHidden())

    def test_the_summary_card_counts_completed_failed_and_warnings(self):
        review = self.review
        paths = [v.path for v in self.project.videos]
        run_dir = fake_run_folder(self.root)
        outcome = BatchRunOutcome(
            results=[(paths[0], None), (paths[1], None)],
            failures=[(paths[2], "boom")],
            quality_warnings=[(paths[1], "lost for 2 s")],
            output_dir=run_dir,
        )
        review.progress_model.reset_rows(paths)
        review.controller._on_finished(outcome)
        self.assertEqual(review.completed_chip.text(), "2 completed")
        self.assertEqual(review.failed_chip.text(), "1 failed")
        self.assertFalse(review.failed_chip.isHidden())
        self.assertEqual(review.warnings_chip.text(), "1 warning")
        self.assertFalse(review.warnings_chip.isHidden())
        self.assertTrue(review.cancelled_chip.isHidden())
        self.assertEqual(review.summary_callout.kind(), "error")
        self.assertIn("mouse_c.avi: boom", review.summary_label.text())
        self.assertIn(str(run_dir), review.summary_label.text())
        self.assertFalse(self.window.is_busy)
        text, chip = review.footer_text()
        self.assertEqual(chip[0], "error")

    def test_a_clean_run_has_a_success_summary_without_failure_chips(self):
        review = self.review
        paths = [v.path for v in self.project.videos]
        review.controller._on_finished(BatchRunOutcome(
            results=[(p, None) for p in paths], output_dir=fake_run_folder(self.root)))
        self.assertEqual(review.summary_callout.kind(), "success")
        self.assertTrue(review.failed_chip.isHidden())
        self.assertTrue(review.warnings_chip.isHidden())
        self.assertEqual(review.footer_text()[1], ("ok", "Finished"))

    def test_view_results_opens_the_run_in_the_results_stage(self):
        review = self.review
        run_dir = fake_run_folder(self.root)
        (run_dir / "manifest.json").write_text(json.dumps({"schema_version": 3}))
        review.controller._on_finished(BatchRunOutcome(
            results=[(self.project.videos[0].path, None)], output_dir=run_dir))
        self.assertTrue(review.open_results_button.isEnabled())
        review.open_results_button.click()
        self.assertEqual(self.window.current_stage_id, "results")

    def test_view_results_is_disabled_when_the_run_wrote_nothing(self):
        review = self.review
        review.controller._on_finished(BatchRunOutcome(
            cancelled=True, output_dir=fake_run_folder(self.root)))
        self.assertFalse(review.open_results_button.isEnabled())
        self.assertIn("did not write", review.open_results_button.toolTip())

    def test_open_folder_opens_the_run_folder(self):
        review = self.review
        run_dir = fake_run_folder(self.root)
        review.controller._on_finished(BatchRunOutcome(output_dir=run_dir))
        review.open_folder_button.click()
        self.assertEqual(self.prompts.called("open_folder"), [(run_dir,)])

    def test_the_log_section_is_collapsed_until_asked(self):
        self.assertFalse(self.review.log_section.is_expanded())
        self.review.log_section.set_expanded(True)
        self.assertTrue(self.review.log_section.is_expanded())

    def test_the_log_is_saved_complete(self):
        self.prompts.say("confirm", True)
        self.review.start_run()
        wait_for(lambda: not self.review.is_running, 120.0, "the run to end")
        target = self.root / "log.txt"
        self.prompts.say("save_file", target)
        self.review.save_log_button.click()
        self.assertIn("Batch Analysis Complete", target.read_text(encoding="utf-8"))


class CancelTests(LongRunFixture):
    def test_cancel_asks_first_and_the_run_page_stays(self):
        review = self.review
        self.prompts.say("confirm", True)
        review.run_button.click()
        self.wait_until_running()
        self.prompts.say("confirm", False)
        review.cancel_button.click()
        self.assertTrue(review.is_running)
        self.assertEqual(self.prompts.called("confirm")[-1][0], "Stop the analysis?")
        self.assertTrue(review.cancel_button.isEnabled())
        self.prompts.say("confirm", True)
        review.cancel_button.click()
        self.assertFalse(review.cancel_button.isEnabled())
        self.assertEqual(review.status_label.text(), "Cancelling...")
        wait_for(lambda: not review.is_running, 120.0, "the run to stop")

    def test_a_cancelled_run_says_so_and_offers_to_resume(self):
        review = self.review
        finished = []
        review.run_finished.connect(finished.append)
        self.prompts.say("confirm", True)
        review.run_button.click()
        self.wait_until_running()
        self.prompts.say("confirm", True)
        review.cancel_button.click()
        wait_for(lambda: bool(finished), 120.0, "the run to stop")
        self.assertTrue(finished[0].cancelled)
        self.assertFalse(review.cancelled_chip.isHidden())
        self.assertEqual(review.summary_callout.kind(), "warning")
        self.assertIn("cancelled", review.summary_label.text())
        self.assertFalse(review.resume_run_button.isHidden())
        self.assertTrue(review.resume_run_button.isEnabled())
        self.assertEqual(review.status_label.text(), "Cancelled - checkpoint saved")
        # The same offer is on the setup page after going back.
        review.back_button.click()
        self.assertFalse(review.resume_box.isHidden())
        self.assertIn("interrupted", review.resume_label.text())
        self.assertTrue(review.resume_button.isEnabled())

    def test_resuming_from_the_button_finishes_the_interrupted_video(self):
        review = self.review
        finished = []
        review.run_finished.connect(finished.append)
        self.prompts.say("confirm", True)
        review.run_button.click()
        self.wait_until_running()
        self.prompts.say("confirm", True)
        review.cancel_button.click()
        wait_for(lambda: bool(finished), 120.0, "the run to stop")
        finished.clear()
        self.prompts.say("confirm", True)
        review.resume_run_button.click()
        self.assertEqual(review.pages.currentIndex(), PAGE_RUN)
        wait_for(lambda: bool(finished), 240.0, "the resumed run to finish")
        self.assertFalse(finished[0].cancelled)
        self.assertEqual(finished[0].failures, [])
        self.assertTrue(review.resume_run_button.isHidden())
        self.assertFalse(review.summary_card.isHidden())

    def test_the_close_guard_keeps_the_stage_on_a_stopping_message(self):
        review = self.review
        self.prompts.say("confirm", True)
        review.run_button.click()
        self.wait_until_running()
        self.prompts.say("confirm", True)
        self.assertFalse(self.window.close())
        self.assertIn("window closes automatically", review.summary_label.text())
        self.assertFalse(review.open_results_button.isEnabled())
        self.prompts.say("ask_save_changes", "discard")
        wait_for(lambda: self.window._closed, 120.0, "the window to close")


class ByteIdentityTests(RunFixture):
    def test_the_button_runs_the_same_files_as_the_cli(self):
        from tests.worker_support import data_files
        review = self.review
        finished = []
        review.run_finished.connect(finished.append)
        self.prompts.say("confirm", True)
        with self.isolated_environment("ws"):
            review.run_button.click()
            wait_for(lambda: bool(finished), 120.0, "the run to finish")
        outcome = finished[0]
        self.assertEqual(outcome.failures, [])
        cli_dir = self.run_cli("cli", "--workers", "1")
        expected, actual = data_files(cli_dir), data_files(outcome.output_dir)
        self.assertEqual(sorted(actual), sorted(expected))
        for name in expected:
            self.assertEqual(actual[name], expected[name], name)


class ShutdownTests(RunFixture):
    def test_shutdown_stops_the_timers(self):
        controller = self.review.controller
        controller._force_timer.start(60_000)
        controller._close_timer.start(60_000)
        self.review.shutdown()
        self.assertFalse(controller._force_timer.isActive())
        self.assertFalse(controller._close_timer.isActive())

    def test_the_window_calls_it_when_it_closes(self):
        controller = self.review.controller
        controller._force_timer.start(60_000)
        self.prompts.say("ask_save_changes", "discard")
        self.window.close()
        self.assertFalse(controller._force_timer.isActive())


class SummaryTextTests(unittest.TestCase):
    def test_the_headline_for_each_outcome(self):
        from gui.workspace.run_controller import RunSummary
        self.assertEqual(RunSummary(completed=3).kind, "success")
        self.assertIn("Processed: 3", RunSummary(completed=3).headline())
        self.assertEqual(RunSummary(completed=2, failed=1).kind, "error")
        self.assertIn("failed: 1", RunSummary(completed=2, failed=1).headline())
        self.assertEqual(RunSummary(completed=1, warnings=2).kind, "warning")
        self.assertEqual(RunSummary(cancelled=True).kind, "warning")
        self.assertIn("cancelled", RunSummary(cancelled=True).headline())
        self.assertEqual(RunSummary(problem="x").kind, "error")

    def test_the_details_list_failures_before_warnings(self):
        from gui.workspace.run_controller import RunSummary
        summary = RunSummary(
            completed=1, failed=1, failures=[("a.avi", "boom")],
            quality_warnings=[("b.avi", "lost")], run_dir=Path("out"),
            problem="not recorded",
        )
        lines = summary.detail_lines()
        self.assertEqual(lines[0], "- a.avi: boom")
        self.assertNotIn("- b.avi: lost", lines)
        self.assertIn("Saved to: out", lines)
        self.assertIn("not recorded", lines)


if __name__ == "__main__":
    unittest.main()
