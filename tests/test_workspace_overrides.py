"""The workspace still works on videos that have a trial or a zone override."""
import unittest

from core.project import SOURCE_CUSTOM, SOURCE_TEMPLATE
from mazes import create_maze
from mazes.base_maze import STEP_TIME_WINDOW
from tests.setup_tab_support import Y, SetupTabFixture
from tests.test_workspace_setup_tab import valid_manager, valid_settings
from tests.workspace_support import HAS_RUNTIME

if HAS_RUNTIME:
    from gui.workspace.setup_tab import SCOPE_VIDEOS


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class SetupTabWithOverridesTests(SetupTabFixture):
    def setUp(self):
        super().setUp()
        self.project.set_template(valid_settings(Y))
        self.zones = valid_manager(create_maze(Y), shift=5)
        self.project.set_trial(self.keys[0], start=0.5)
        self.project.set_zone_override(self.keys[0], self.zones)
        self.app.processEvents()

    def _edit_first_video(self):
        self.select([self.keys[0]])
        self.tab.set_scope(SCOPE_VIDEOS)

    def test_the_tab_shows_the_effective_settings_of_the_video(self):
        self._edit_first_video()
        shown = self.tab.state.to_batch_settings(finalize=True)
        self.assertEqual(shown.start_time, 0.5)
        self.assertIsNone(shown.end_time)
        self.assertEqual(
            shown.roi_manager.to_dict(), self.zones.to_dict()
        )

    def test_applying_an_unchanged_step_keeps_the_layers(self):
        self._edit_first_video()
        self.tab.go_to_step(STEP_TIME_WINDOW)
        self.assertTrue(self.tab.apply_step(), self.tab._apply_issues)
        entry = self.project.entry(self.keys[0])
        self.assertEqual(entry.settings_source, SOURCE_TEMPLATE)
        self.assertEqual(entry.trial.start_time, 0.5)
        self.assertIsNone(entry.trial.end_time)
        self.assertIsNotNone(entry.zones)
        self.assertEqual(self.project.zone_status(entry), "adjusted")

    def test_editing_the_times_keeps_the_regions_and_the_trial_marker(self):
        self._edit_first_video()
        self.tab.go_to_step(STEP_TIME_WINDOW)
        self.tab.host.panel.start_spinbox.setValue(0.75)
        self.assertTrue(self.tab.apply_step(), self.tab._apply_issues)
        entry = self.project.entry(self.keys[0])
        self.assertEqual(entry.settings_source, SOURCE_CUSTOM)
        self.assertAlmostEqual(entry.trial.start_time, 0.75, places=2)
        self.assertTrue(entry.trial.start_set)
        effective = self.project.effective_settings(entry)
        self.assertAlmostEqual(effective.start_time, 0.75, places=2)
        self.assertIsNone(effective.end_time)
        self.assertEqual(
            effective.roi_manager.to_dict(), self.zones.to_dict()
        )
        # The other video is untouched.
        other = self.project.entry(self.keys[1])
        self.assertEqual(other.settings_source, SOURCE_TEMPLATE)
        self.assertIsNone(other.trial)

    def test_apply_template_asks_before_dropping_a_zone_override(self):
        panel = self.window.video_panel
        self.prompts.say("confirm", False, True)
        self.assertEqual(panel.apply_template([self.keys[0]]), [])
        self.assertIsNotNone(self.project.entry(self.keys[0]).zones)
        self.assertEqual(panel.apply_template([self.keys[0]]), [self.keys[0]])
        entry = self.project.entry(self.keys[0])
        self.assertIsNone(entry.zones)
        self.assertEqual(entry.trial.start_time, 0.5)     # the times stay
        self.assertEqual(len(self.prompts.called("confirm")), 2)


if __name__ == "__main__":
    unittest.main()
