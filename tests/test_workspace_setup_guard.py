"""The Setup tab's unapplied-step guard: Apply / Discard / Cancel everywhere."""
import unittest

from core.project import SOURCE_CUSTOM
from mazes.base_maze import (STEP_DETECTION_TUNING, STEP_ROIS,
                             STEP_TIME_WINDOW)
from tests.setup_tab_support import Y, SetupTabFixture, draw_zones, square
from tests.test_workspace_setup_tab import valid_settings
from tests.workspace_support import HAS_RUNTIME

if HAS_RUNTIME:
    from PyQt6.QtCore import QCoreApplication, QEvent
    from gui.workspace.prompts import CANCEL, DISCARD, SAVE
    from gui.workspace.setup_tab import SCOPE_TEMPLATE, SCOPE_VIDEOS


class GuardFixture(SetupTabFixture):
    def setUp(self):
        super().setUp()
        self.project.set_template(valid_settings(Y))
        self.app.processEvents()

    def edit_time_window(self, value=0.7):
        self.tab.go_to_step(STEP_TIME_WINDOW)
        self.tab.host.panel.start_spinbox.setValue(value)
        self.assertTrue(self.tab.has_unapplied_edits())

    def template_start(self):
        return self.project.template_settings.start_time

    def asked(self):
        return self.prompts.called("ask_save_changes")


class LeavingAStepTests(GuardFixture):
    def test_nothing_is_asked_without_edits(self):
        self.tab.go_to_step(STEP_TIME_WINDOW)
        self.tab.go_to_step(STEP_ROIS)
        self.assertEqual(self.asked(), [])

    def test_apply_writes_the_edits_and_moves_on(self):
        self.edit_time_window()
        self.prompts.say("ask_save_changes", SAVE)
        self.assertTrue(self.tab.go_to_step(STEP_ROIS))
        self.assertEqual(self.tab.step, STEP_ROIS)
        self.assertAlmostEqual(self.template_start(), 0.7, places=2)
        (name,), = self.asked()
        self.assertIn("Time window", name)
        self.assertIn("project template", name)

    def test_discard_moves_on_without_writing(self):
        self.edit_time_window()
        self.prompts.say("ask_save_changes", DISCARD)
        self.assertTrue(self.tab.go_to_step(STEP_ROIS))
        self.assertEqual(self.tab.step, STEP_ROIS)
        self.assertEqual(self.template_start(), 0.0)
        self.tab.go_to_step(STEP_TIME_WINDOW)
        self.assertEqual(self.tab.host.panel.start_spinbox.value(), 0.0)

    def test_cancel_stays_and_keeps_the_edits(self):
        self.edit_time_window()
        self.prompts.say("ask_save_changes", CANCEL)
        self.assertFalse(self.tab.go_to_step(STEP_ROIS))
        self.assertEqual(self.tab.step, STEP_TIME_WINDOW)
        self.assertEqual(self.template_start(), 0.0)
        self.assertAlmostEqual(
            self.tab.host.panel.start_spinbox.value(), 0.7, places=2)

    def test_cancel_puts_the_list_selection_back(self):
        self.edit_time_window()
        self.prompts.say("ask_save_changes", CANCEL)
        row = self.tab.steps.index(STEP_ROIS)
        self.tab.step_list.setCurrentRow(row)
        self.assertEqual(self.tab.step, STEP_TIME_WINDOW)
        self.assertEqual(
            self.tab.step_list.currentRow(),
            self.tab.steps.index(STEP_TIME_WINDOW),
        )

    def test_apply_that_cannot_finish_keeps_the_user_on_the_step(self):
        self.tab.go_to_step(STEP_ROIS)
        self.tab.host.editor.clear_current_zone()
        self.tab.host.editor.set_current_zone("arm_a")
        self.tab.host.editor.clear_current_zone()
        self.assertTrue(self.tab.has_unapplied_edits())
        self.prompts.say("ask_save_changes", SAVE)
        self.assertFalse(self.tab.go_to_step(STEP_TIME_WINDOW))
        self.assertEqual(self.tab.step, STEP_ROIS)
        self.assertIn("Arm A", self.tab.issues_label.text())

    def test_unsaved_roi_work_counts(self):
        self.tab.go_to_step(STEP_ROIS)
        editor = self.tab.host.editor
        self.assertFalse(self.tab.has_unapplied_edits())
        editor.clear_current_zone()
        draw_zones(editor, ["arm_a"], first_index=8)
        self.assertTrue(editor.has_unsaved_work())
        self.assertTrue(self.tab.has_unapplied_edits())
        self.prompts.say("ask_save_changes", SAVE)
        self.assertTrue(self.tab.go_to_step(STEP_TIME_WINDOW))
        points = self.project.template_settings.roi_manager.rois["arm_a"][0]
        self.assertEqual(points.points.tolist(), [list(p) for p in square(8)])
        self.tab.go_to_step(STEP_ROIS)
        self.assertFalse(self.tab.has_unapplied_edits())

    def test_a_moved_tuner_slider_counts_and_discard_shuts_it_down(self):
        self.tab.go_to_step(STEP_DETECTION_TUNING)
        tuner = self.tab.host.tuner
        self.assertFalse(self.tab.has_unapplied_edits())
        slider = tuner.sliders["Min component area"]
        original = self.project.template_settings.detection_config             .min_component_area
        slider.setValue(slider.value() + 7)
        self.assertTrue(self.tab.has_unapplied_edits())
        self.prompts.say("ask_save_changes", DISCARD)
        self.assertTrue(self.tab.go_to_step(STEP_TIME_WINDOW))
        self.assertTrue(tuner._lifecycle.is_shut_down)
        self.assertEqual(
            self.project.template_settings.detection_config
            .min_component_area, original)

    def test_apply_from_the_guard_stores_the_tuning(self):
        self.tab.go_to_step(STEP_DETECTION_TUNING)
        slider = self.tab.host.tuner.sliders["Min component area"]
        slider.setValue(slider.value() + 7)
        wanted = slider.value()
        self.prompts.say("ask_save_changes", SAVE)
        self.assertTrue(self.tab.go_to_step(STEP_TIME_WINDOW))
        self.assertEqual(
            self.project.template_settings.detection_config
            .min_component_area, wanted)

    def test_changing_the_reference_video_is_guarded(self):
        self.tab.go_to_step(STEP_TIME_WINDOW)
        self.tab.host.panel.start_spinbox.setValue(0.4)
        self.prompts.say("ask_save_changes", CANCEL)
        self.assertFalse(self.tab.set_reference_video(self.keys[1]))
        self.assertEqual(self.tab.reference_combo.currentData(), self.keys[0])
        self.prompts.say("ask_save_changes", DISCARD)
        self.assertTrue(self.tab.set_reference_video(self.keys[1]))

    def test_marking_verified_asks_first(self):
        self.edit_time_window()
        self.select([self.keys[0]])
        self.prompts.say("ask_save_changes", CANCEL)
        self.assertFalse(self.tab.mark_selected_verified())
        self.assertFalse(self.project.entry(self.keys[0]).verified)
        self.prompts.say("ask_save_changes", SAVE)
        self.assertTrue(self.tab.mark_selected_verified())
        self.assertTrue(self.project.entry(self.keys[0]).verified)
        self.assertAlmostEqual(self.template_start(), 0.7, places=2)


class SwitchingScopeTests(GuardFixture):
    def test_switching_the_scope_with_edits_asks(self):
        self.select([self.keys[0]])
        self.edit_time_window()
        self.prompts.say("ask_save_changes", CANCEL)
        self.assertFalse(self.tab.set_scope(SCOPE_VIDEOS))
        self.assertEqual(self.tab.scope, SCOPE_TEMPLATE)
        self.assertEqual(
            self.tab.scope_combo.currentData(), SCOPE_TEMPLATE)
        self.prompts.say("ask_save_changes", DISCARD)
        self.assertTrue(self.tab.set_scope(SCOPE_VIDEOS))
        self.assertEqual(self.tab.scope, SCOPE_VIDEOS)
        self.assertEqual(self.template_start(), 0.0)

    def test_applying_while_switching_writes_to_the_old_scope(self):
        self.select([self.keys[0]])
        self.edit_time_window()
        self.prompts.say("ask_save_changes", SAVE)
        self.assertTrue(self.tab.set_scope(SCOPE_VIDEOS))
        self.assertAlmostEqual(self.template_start(), 0.7, places=2)
        # The video still follows the template, so it got the edit too.
        self.assertAlmostEqual(
            self.project.entry(self.keys[0]).settings.start_time, 0.7, places=2)

    def test_changing_the_selection_with_edits_asks(self):
        self.select([self.keys[0]])
        self.tab.set_scope(SCOPE_VIDEOS)
        self.edit_time_window(0.3)
        self.prompts.say("ask_save_changes", CANCEL)
        self.select([self.keys[1]])
        # The selection snapped back to the video being edited.
        self.assertEqual(self.window.selected_keys(), [self.keys[0]])
        self.assertEqual(self.tab.targets, [self.keys[0]])
        self.assertAlmostEqual(
            self.tab.host.panel.start_spinbox.value(), 0.3, places=2)

        self.prompts.say("ask_save_changes", SAVE)
        self.select([self.keys[1]])
        self.assertEqual(self.tab.targets, [self.keys[1]])
        first = self.project.entry(self.keys[0])
        self.assertEqual(first.settings_source, SOURCE_CUSTOM)
        self.assertAlmostEqual(first.settings.start_time, 0.3, places=2)
        self.assertEqual(
            self.project.entry(self.keys[1]).settings.start_time, 0.0)

    def test_discarding_on_a_selection_change_loads_the_new_video(self):
        self.select([self.keys[0]])
        self.tab.set_scope(SCOPE_VIDEOS)
        self.edit_time_window(0.3)
        self.prompts.say("ask_save_changes", DISCARD)
        self.select([self.keys[1]])
        self.assertEqual(self.tab.targets, [self.keys[1]])
        self.assertEqual(self.tab.step, STEP_TIME_WINDOW)
        self.assertEqual(self.tab.host.panel.start_spinbox.value(), 0.0)

    def test_the_selection_is_free_while_editing_the_template(self):
        self.edit_time_window()
        self.select([self.keys[1]])
        self.assertEqual(self.asked(), [])
        self.assertTrue(self.tab.has_unapplied_edits())


class ClosingTests(GuardFixture):
    def test_closing_the_project_asks_about_the_step_first(self):
        self.edit_time_window()
        self.prompts.say("ask_save_changes", CANCEL)
        self.assertFalse(self.window.close_project())
        self.assertIsNotNone(self.window.project)
        self.assertEqual(len(self.asked()), 1)

    def test_discarding_the_step_then_the_project(self):
        self.edit_time_window()
        self.prompts.say("ask_save_changes", DISCARD, DISCARD)
        self.assertTrue(self.window.close_project())
        self.assertIsNone(self.window.project)
        self.assertEqual(len(self.asked()), 2)
        self.assertIn("Time window", self.asked()[0][0])

    def test_applying_the_step_before_closing_saves_it_in_the_project(self):
        self.edit_time_window()
        project = self.project
        self.prompts.say("ask_save_changes", SAVE, DISCARD)
        self.assertTrue(self.window.close_project())
        self.assertAlmostEqual(
            project.template_settings.start_time, 0.7, places=2)

    def test_closing_the_window_asks_and_can_be_cancelled(self):
        self.edit_time_window()
        self.window._skip_close_prompt = False
        self.prompts.say("ask_save_changes", CANCEL)
        self.window.close()
        self.assertFalse(self.window._closed)
        self.assertEqual(len(self.asked()), 1)
        self.window._skip_close_prompt = True    # let the fixture clean up

    def test_a_clean_step_does_not_block_closing(self):
        self.tab.go_to_step(STEP_TIME_WINDOW)
        self.prompts.say("ask_save_changes", DISCARD)       # project-level only
        self.assertTrue(self.window.close_project())
        self.assertEqual(len(self.asked()), 1)
        self.assertEqual(self.asked()[0][0], "study")

    def test_closing_the_project_tears_the_step_down(self):
        self.tab.go_to_step(STEP_DETECTION_TUNING)
        tuner = self.tab.host.tuner
        self.prompts.say("ask_save_changes", DISCARD)
        self.assertTrue(self.window.close_project())
        self.assertTrue(tuner._lifecycle.is_shut_down)
        self.assertIsNone(self.tab.host)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


class QuittingTests(GuardFixture):
    def test_quitting_the_application_joins_a_running_tuner(self):
        from PyQt6.QtWidgets import QApplication
        self.tab.go_to_step(STEP_DETECTION_TUNING)
        tuner = self.tab.host.tuner
        QApplication.instance().aboutToQuit.emit()
        self.assertTrue(tuner._lifecycle.is_shut_down)
        self.assertIsNone(self.tab.host)


class BusyTests(GuardFixture):
    def test_nothing_is_applied_while_a_run_is_in_progress(self):
        self.edit_time_window()
        self.window.set_busy(True, "running")
        self.addCleanup(self.window.set_busy, False)
        self.tab.on_tab_activated()     # what returning to the tab does
        self.assertFalse(self.tab.apply_step())
        self.assertFalse(self.tab.apply_button.isEnabled())
        self.assertEqual(len(self.prompts.called("warn")), 1)
        self.assertEqual(self.template_start(), 0.0)


if __name__ == "__main__":
    unittest.main()
