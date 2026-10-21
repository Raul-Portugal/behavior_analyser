"""Trial times entered in the Trials stage (typed, or taken from the player)
give a workspace run whose outputs equal, byte for byte, the CLI run of the
equivalent full copies (see ``test_workspace_trial_run`` for the model-level
version of the same guarantee)."""
import unittest

from tests.test_workspace_trial_run import (DURATION, END_A, END_C, START_A,
                                            START_B, START_C, full_copy,
                                            manifest_hashes, smaller_arena)
from tests.worker_support import data_files
from tests.workspace_run_support import RunFixture
from tests.workspace_support import HAS_RUNTIME

if HAS_RUNTIME:
    from core.project import START_PER_VIDEO, ZONE_ADJUSTED
    from gui.workspace.stages import trials_widgets as tw

FPS = 30        # the synthetic tracking videos


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class TrialsStageRunTests(RunFixture):
    def setUp(self):
        super().setUp()
        project = self.project
        self.stage = self.window.stage("trials")
        self.addCleanup(self.stage.shutdown)
        self.a, self.b, self.c = (project.find(p).key for p in self.real_videos)
        self.assertTrue(self.window.go_to_stage("trials"))
        self.tag_the_cohort()

    def tag_the_cohort(self):
        """What a person does on the stage: protocol, then each video."""
        stage, model = self.stage, self.stage.model
        card = stage.protocol
        card.section.set_expanded(True)
        card.fixed_radio.click()
        card.duration_edit.setText("2")
        card.duration_edit.editingFinished.emit()
        card.start_per_video_radio.click()
        self.assertEqual(self.project.trial_protocol.duration_s, DURATION)
        self.assertEqual(self.project.trial_protocol.start_mode, START_PER_VIDEO)
        self.assertEqual(stage.counts()[:3], (3, 0, 3))

        def cell(key, column):
            return model.index(model.row_of(key), column)

        # Video A: start and end taken from the frame on screen.
        stage.select_row(self.a)
        player = stage.player
        self.assertTrue(player.wait_idle())
        player.seek(round(START_A * FPS))
        self.assertTrue(player.wait_idle())
        self.assertTrue(stage.set_start_here())
        player.seek(round(END_A * FPS))
        self.assertTrue(player.wait_idle())
        self.assertTrue(stage.set_end_here())
        # Video B: only a typed start; the protocol gives the end.
        self.assertTrue(model.setData(cell(self.b, tw.COL_START), "0.5"))
        # Video C: typed start and end, with its own zones.
        self.assertTrue(model.setData(cell(self.c, tw.COL_START), "0:00.2"))
        self.assertTrue(model.setData(cell(self.c, tw.COL_END), "3"))
        self.project.set_zone_override(self.c, smaller_arena(), source=ZONE_ADJUSTED)
        self.assertEqual(stage.counts()[:3], (3, 3, 0))
        self.assertTrue(stage.can_advance()[0])
        # Leave the stage: the video is released before the run starts.
        self.assertTrue(self.window.go_to_stage("review"))
        self.assertFalse(player.capture_open)

    def test_the_stage_wrote_the_times_of_the_full_copies(self):
        for entry in self.project.videos:
            settings = self.project.effective_settings(entry)
            self.assertEqual(settings.to_dict(), full_copy(entry.name).to_dict(),
                             entry.name)
        self.assertEqual(self.project.entry(self.a).trial.start_time, START_A)
        self.assertEqual(self.project.entry(self.a).trial.end_time, END_A)
        self.assertEqual(self.project.entry(self.b).trial.start_time, START_B)
        self.assertIsNone(self.project.entry(self.b).trial.end_time)
        self.assertEqual(self.project.entry(self.c).trial.end_time, END_C)
        self.assertEqual(self.project.entry(self.c).trial.start_time, START_C)

    def test_the_workspace_run_equals_the_cli_run_of_the_full_copies(self):
        with self.isolated_environment("ws"):
            outcome = self.start_and_wait()
        self.assertEqual(outcome.failures, [])
        cli_dir = self.run_cli(
            "cli", "--workers", "1",
            settings_for=lambda video: full_copy(video.name),
        )
        expected, actual = data_files(cli_dir), data_files(outcome.output_dir)
        self.assertEqual(sorted(actual), sorted(expected))
        self.assertIn("batch_summary.csv", expected)
        self.assertIn("run_plan.json", expected)
        for name in expected:
            self.assertEqual(actual[name], expected[name], name)
        self.assertEqual(manifest_hashes(outcome.output_dir), manifest_hashes(cli_dir))
        self.assertEqual(len(manifest_hashes(cli_dir)), 3)

    def test_the_times_change_what_is_analysed(self):
        # Guard against a vacuous comparison.
        with self.isolated_environment("ws"):
            outcome = self.start_and_wait()
        plain = self.run_cli("plain", "--workers", "1")
        self.assertNotEqual(
            data_files(plain)["batch_summary.csv"],
            data_files(outcome.output_dir)["batch_summary.csv"],
        )


if __name__ == "__main__":
    unittest.main()
