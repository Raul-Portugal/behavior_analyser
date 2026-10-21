"""Template alignment from the Setup tab: the align action and the banner."""
import unittest
from unittest.mock import patch

import numpy as np

from core.project import SOURCE_ALIGNED, SOURCE_TEMPLATE
from mazes.base_maze import STEP_TIME_WINDOW
from tests.setup_tab_support import Y, SetupTabFixture
from tests.synthetic_video import write_indexed_video
from tests.test_workspace_setup_tab import valid_settings
from tests.workspace_support import HAS_RUNTIME

if HAS_RUNTIME:
    from gui.roi_alignment_dialog import AlignmentChoice
    from gui.workspace.prompts import CANCEL
    from gui.workspace.setup_tab import SCOPE_VIDEOS

DIALOG = "gui.workspace.setup_tab.align_template_with_dialog"
BIG = (200, 150)


def scaling(factor=1.25) -> "AlignmentChoice":
    return AlignmentChoice(
        np.diag([factor, factor, 1.0]), 0.5, "manual landmarks"
    )


class AlignFixture(SetupTabFixture):
    PARADIGM = Y

    def setUp(self):
        super().setUp()
        folder = self.root / "big"
        folder.mkdir()
        self.big = [
            write_indexed_video(folder / f"big{n}.avi", 40, size=BIG)
            for n in (1, 2)
        ]
        self.project.add_videos(self.big)
        self.app.processEvents()
        self.big_keys = self.project.keys()[2:]
        self.project.set_template(valid_settings(Y))
        self.app.processEvents()
        self.window.video_panel.clear_selection()

    def align(self):
        # The Tools action now belongs to the Arena & zones stage (see
        # test_workspace_arena_stage); the tab's own alignment still works.
        return self.tab.align_template()


class AlignActionTests(AlignFixture):
    def test_the_action_follows_the_template(self):
        self.assertTrue(self.window.align_action.isEnabled())
        self.project.set_template(None)
        self.app.processEvents()
        self.assertFalse(self.window.align_action.isEnabled())
        self.project.set_template(valid_settings(Y))
        self.app.processEvents()
        self.assertTrue(self.window.align_action.isEnabled())

    def test_the_action_is_off_without_a_project(self):
        self.prompts.say("ask_save_changes", "discard")
        self.window.close_project()
        self.assertFalse(self.window.align_action.isEnabled())

    def test_the_action_aligns_the_videos_that_do_not_fit(self):
        with patch(DIALOG, return_value=scaling()) as dialog:
            self.prompts.say("confirm", False)
            self.align()
        # Two big videos share a size; the answer "no" asks for the second.
        self.assertEqual(dialog.call_count, 2)
        for key in self.big_keys:
            entry = self.project.entry(key)
            self.assertEqual(entry.settings_source, SOURCE_ALIGNED)
            self.assertEqual(entry.settings.roi_manager.source_dimensions, BIG)
        for key in self.keys:
            self.assertEqual(
                self.project.entry(key).settings_source, SOURCE_TEMPLATE)

    def test_the_dialog_gets_the_template_and_the_two_frames(self):
        with patch(DIALOG, return_value=scaling()) as dialog:
            self.prompts.say("confirm", True)
            self.align()
        self.assertEqual(dialog.call_count, 1)
        args, kwargs = dialog.call_args
        _parent, manager, template_frame, target_frame = args[:4]
        self.assertEqual(
            manager.to_dict(), self.project.template_settings.roi_manager.to_dict())
        self.assertEqual(template_frame.shape[:2], (120, 160))
        self.assertEqual(target_frame.shape[:2], (150, 200))
        self.assertEqual(kwargs["template_name"], "clip1.avi")
        self.assertEqual(kwargs["target_name"], "big1.avi")

    def test_one_alignment_for_the_same_camera(self):
        with patch(DIALOG, return_value=scaling(1.25)) as dialog:
            self.prompts.say("confirm", True)
            self.align()
        self.assertEqual(dialog.call_count, 1)
        (title, text), = self.prompts.called("confirm")
        self.assertIn("Same camera", title)
        for key in self.big_keys:
            self.assertEqual(
                self.project.entry(key).settings_source, SOURCE_ALIGNED)

    def test_the_regions_are_carried_over_to_the_new_size(self):
        template = self.project.template_settings
        with patch(DIALOG, return_value=scaling(1.25)):
            self.prompts.say("confirm", True)
            self.align()
        aligned = self.project.entry(self.big_keys[0]).settings
        source = template.roi_manager.rois["arm_a"][0].points
        moved = aligned.roi_manager.rois["arm_a"][0].points
        np.testing.assert_allclose(moved, source * 1.25, atol=1.0)
        self.assertEqual(
            self.project.template_settings.to_dict(), template.to_dict())

    def test_cancelling_the_dialog_changes_nothing(self):
        with patch(DIALOG, return_value=None) as dialog:
            self.align()
        self.assertEqual(dialog.call_count, 1)
        for key in self.big_keys:
            self.assertEqual(
                self.project.entry(key).settings_source, SOURCE_TEMPLATE)

    def test_selected_videos_are_aligned_even_if_they_fit(self):
        self.select([self.keys[1]])
        with patch(DIALOG, return_value=scaling(1.0)) as dialog:
            self.align()
        self.assertEqual(dialog.call_count, 1)
        self.assertEqual(
            self.project.entry(self.keys[1]).settings_source, SOURCE_ALIGNED)
        self.assertEqual(
            self.project.entry(self.big_keys[0]).settings_source,
            SOURCE_TEMPLATE)

    def test_everything_fits_says_so(self):
        self.project.remove_videos(self.big_keys)
        self.app.processEvents()
        with patch(DIALOG) as dialog:
            self.align()
        dialog.assert_not_called()
        self.assertEqual(len(self.prompts.called("info")), 1)

    def test_without_a_template_it_explains(self):
        self.project.set_template(None)
        self.app.processEvents()
        self.assertEqual(self.tab.align_template(), 0)
        (title, text), = self.prompts.called("info")
        self.assertIn("template", text)

    def test_unapplied_edits_are_asked_about_first(self):
        self.tab.go_to_step(STEP_TIME_WINDOW)
        self.tab.host.panel.start_spinbox.setValue(0.3)
        self.prompts.say("ask_save_changes", CANCEL)
        with patch(DIALOG) as dialog:
            self.assertEqual(self.tab.align_template(), 0)
        dialog.assert_not_called()

    def test_the_template_video_is_asked_for_when_none_fits(self):
        self.project.remove_videos(self.keys)
        self.app.processEvents()
        self.prompts.say("open_file", self.clips[0])
        self.prompts.say("confirm", True)
        with patch(DIALOG, return_value=scaling()) as dialog:
            self.align()
        self.assertEqual(dialog.call_count, 1)
        (title, _start), = self.prompts.called("open_file")
        self.assertIn("template", title)
        self.assertEqual(
            self.project.entry(self.big_keys[0]).settings_source,
            SOURCE_ALIGNED)

    def test_a_wrongly_sized_template_video_is_refused(self):
        self.project.remove_videos(self.keys)
        self.app.processEvents()
        self.prompts.say("open_file", self.big[0])
        with patch(DIALOG) as dialog:
            self.align()
        dialog.assert_not_called()
        self.assertEqual(len(self.prompts.called("warn")), 1)

    def test_declining_the_template_video_prompt_stops(self):
        self.project.remove_videos(self.keys)
        self.app.processEvents()
        self.prompts.say("open_file", None)
        with patch(DIALOG) as dialog:
            self.align()
        dialog.assert_not_called()


class AlignBannerTests(AlignFixture):
    def test_a_video_that_does_not_fit_shows_the_banner(self):
        self.select([self.big_keys[0]])
        self.tab.set_scope(SCOPE_VIDEOS)
        self.assertFalse(self.tab.banner.isHidden())
        text = self.tab.banner_label.text()
        self.assertIn("200x150", text)
        self.assertIn("160x120", text)
        self.assertEqual(text, self.tab.mismatch_text())

    def test_a_video_that_fits_has_no_banner(self):
        self.select([self.keys[0]])
        self.tab.set_scope(SCOPE_VIDEOS)
        self.assertTrue(self.tab.banner.isHidden())

    def test_the_template_scope_has_no_banner(self):
        self.select([self.big_keys[0]])
        self.assertTrue(self.tab.banner.isHidden())

    def test_the_banner_button_routes_to_the_same_alignment(self):
        self.select([self.big_keys[0]])
        self.tab.set_scope(SCOPE_VIDEOS)
        with patch(DIALOG, return_value=scaling()) as dialog:
            self.prompts.say("confirm", False)
            self.tab.banner_align_button.click()
        self.assertEqual(dialog.call_count, 1)
        self.assertEqual(
            self.project.entry(self.big_keys[0]).settings_source,
            SOURCE_ALIGNED)
        # The tab picked the aligned settings up, and the banner is gone.
        self.assertEqual(
            self.tab.state.roi_manager.source_dimensions, BIG)
        self.assertTrue(self.tab.banner.isHidden())

    def test_the_roi_step_shows_the_aligned_regions(self):
        self.select([self.big_keys[0]])
        self.tab.set_scope(SCOPE_VIDEOS)
        with patch(DIALOG, return_value=scaling()):
            self.prompts.say("confirm", False)
            self.tab.align_template()
        self.tab.go_to_step("rois")
        self.assertEqual(self.tab.host.editor.source_size, BIG)


if __name__ == "__main__":
    unittest.main()
