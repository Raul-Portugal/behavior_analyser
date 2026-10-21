"""A run started from the Review & run stage is the same run as the CLI's."""
import unittest

from tests.worker_support import data_files
from tests.workspace_run_support import RunFixture, wait_for
from tests.workspace_support import HAS_RUNTIME

if HAS_RUNTIME:
    from core.project import STATUS_DONE


class RunThroughTheStagesTests(RunFixture):
    def test_review_and_run_produces_the_files_of_the_cli(self):
        window = self.window
        self.assertTrue(window.go_to_stage("review"))
        self.assertIs(self.run_tab.parentWidget(), window.stage("review").content)
        self.assertTrue(window.run_action.isEnabled())
        self.prompts.say("confirm", True)
        finished = []
        self.run_tab.run_finished.connect(finished.append)
        with self.isolated_environment("ws"):
            window.run_action.trigger()                 # Tools > Run analysis
            wait_for(lambda: bool(finished), 120.0, "the run to finish")
        outcome = finished[0]
        self.assertEqual(outcome.failures, [])
        cli_dir = self.run_cli("cli", "--workers", "1")
        expected, actual = data_files(cli_dir), data_files(outcome.output_dir)
        self.assertEqual(sorted(actual), sorted(expected))
        self.assertIn("batch_summary.csv", expected)
        for name in expected:
            self.assertEqual(actual[name], expected[name], name)

    def test_open_results_goes_to_the_results_stage(self):
        self.window.go_to_stage("review")
        self.start_and_wait()
        self.assertEqual(self.window.current_stage_id, "review")
        self.assertTrue(self.run_tab.open_results())
        self.assertEqual(self.window.current_stage_id, "results")
        self.assertEqual(self.window.stepper.status("results"), "current")
        self.assertIs(
            self.results_tab.parentWidget(), self.window.stage("results").content
        )

    def test_a_finished_run_marks_the_stages(self):
        self.start_and_wait()
        window = self.window
        window.go_to_stage("videos")
        window.refresh_navigation()
        self.assertEqual(window.stepper.status("results"), "done")
        self.assertEqual(
            {v.last_status for v in self.project.videos}, {STATUS_DONE}
        )
        chips = {
            self.window.video_panel.proxy.chip_for(key)
            for key in self.project.keys()
        }
        self.assertEqual(chips, {("ok", "Analysed")})

    def test_the_study_cannot_be_edited_while_the_run_is_busy(self):
        window = self.window
        window.go_to_stage("review")
        window.set_busy(True, "Running")
        self.assertTrue(window.video_panel.is_read_only())
        self.assertTrue(window.go_to_stage("results"))     # looking is allowed
        window.set_busy(False)


if __name__ == "__main__":
    unittest.main()
