"""The workspace Setup tab: layout, scopes, step hosting and teardown."""
import unittest
from pathlib import Path

import numpy as np

from core.project import (SOURCE_CUSTOM, SOURCE_TEMPLATE)
from core.setup_state import (STATUS_DONE, STATUS_INVALID, STATUS_TODO,
                              SetupState)
from mazes import create_maze
from mazes.base_maze import (STEP_CALIBRATION, STEP_DETECTION_TUNING,
                             STEP_PARADIGM_PARAMS, STEP_REVIEW, STEP_ROIS,
                             STEP_TIME_WINDOW, STEP_TST_TUNING)
from roi_manager import ROIManager
from tests.project_support import make_settings
from tests.setup_tab_support import (E, F, T, Y, SetupTabFixture, draw_zones,
                                     square, wait_until)
from tests.workspace_support import HAS_RUNTIME

if HAS_RUNTIME:
    from PyQt6 import sip
    from PyQt6.QtCore import QCoreApplication, QEvent
    from gui.roi_editor import TOOL_LINE
    from gui.workspace.prompts import DISCARD
    from gui.workspace.setup_steps import (CalibrationHost, DetectionHost,
                                           MessageHost, PanelHost, ReviewHost,
                                           RoiHost, TstHost, status_icon)
    from gui.workspace.setup_tab import (SCOPE_TEMPLATE, SCOPE_VIDEOS,
                                         SetupTab)

SIZE = (160, 120)


def valid_manager(maze, size=SIZE, reference=True, shift=0) -> ROIManager:
    manager = ROIManager(source_dimensions=size)
    for index, (name, _label) in enumerate(maze.get_roi_definitions()):
        points = np.array(square(index + shift))
        manager.add_roi(name, points)
    if reference:
        manager.set_reference_line((10, 100), (110, 100), "User-defined")
    return manager


def valid_settings(paradigm=Y, **overrides):
    """Complete, valid settings for ``paradigm`` (built through SetupState)."""
    state = SetupState.new(paradigm)
    maze = state.build_maze()
    if paradigm == T:
        maze.configure_mice(2)
    state.set_maze(maze)
    state.roi_manager = valid_manager(maze, reference=paradigm != T)
    for key, value in overrides.items():
        setattr(state, key, value)
    return state.to_batch_settings(finalize=True)


def dispose(widget) -> None:
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


class LayoutTests(SetupTabFixture):
    def test_the_window_installs_the_setup_tab(self):
        self.assertIsInstance(self.window.setup_tab, SetupTab)
        self.assertIs(self.window.tab_widget("setup"), self.window.setup_tab)
        self.assertIs(self.tab.workspace, self.window)

    def test_no_stage_hosts_the_setup_tab_any_more(self):
        self.choose_paradigm(T)
        window = self.window
        for stage_id in ("arena", "detection", "trials"):
            with self.subTest(stage_id):
                self.assertTrue(window.go_to_stage(stage_id))
                self.assertIsNot(self.tab.parentWidget(), window.stage(stage_id).content)

    def test_empty_states_point_to_what_is_missing(self):
        window = self.window
        self.prompts.say("ask_save_changes", DISCARD)
        window.close_project()
        self.assertIn("project", self.tab.empty_label.text().lower())
        self.assertIs(self.tab.stack.currentWidget(), self.tab.empty_label)
        self.assertEqual(self.tab.step_list.count(), 0)
        for button in (self.tab.apply_button, self.tab.next_button,
                       self.tab.previous_button, self.tab.verify_button):
            self.assertFalse(button.isEnabled())

        self.new_project_at("second.bmaproj")
        self.assertIn("paradigm", self.tab.empty_label.text().lower())
        self.assertIn("right", self.tab.empty_label.text().lower())
        self.assertIs(self.tab.stack.currentWidget(), self.tab.empty_label)

        self.choose_paradigm(Y)
        self.assertIsNot(self.tab.stack.currentWidget(), self.tab.empty_label)
        self.assertEqual(self.tab.steps[0], STEP_PARADIGM_PARAMS)

    def test_steps_follow_the_paradigm_hooks(self):
        for name in (Y, E, F, T):
            with self.subTest(paradigm=name):
                self.choose_paradigm(name)
                expected = create_maze(name).setup_steps()
                self.assertEqual(self.tab.steps, expected)
                self.assertEqual(
                    [self.tab.step_list.item(i).data(0x0100)
                     for i in range(self.tab.step_list.count())],
                    expected,
                )

    def test_a_fresh_template_starts_at_the_first_open_step(self):
        self.assertEqual(self.tab.step, STEP_PARADIGM_PARAMS)
        self.assertIsInstance(self.tab.host, PanelHost)
        self.assertEqual(self.tab.stack.count(), 2)

    def test_status_icons_follow_the_step_report(self):
        self.project.set_template(make_settings(Y))    # only arm_a drawn
        self.app.processEvents()
        tab = self.tab
        statuses = {s: tab.step_status(s) for s in tab.steps}
        self.assertEqual(statuses[STEP_ROIS], STATUS_INVALID)
        self.assertEqual(statuses[STEP_PARADIGM_PARAMS], STATUS_DONE)
        self.assertEqual(statuses[STEP_REVIEW], STATUS_INVALID)
        for row, step in enumerate(tab.steps):
            item = tab.step_list.item(row)
            self.assertEqual(
                item.icon().cacheKey(),
                status_icon(statuses[step]).cacheKey(), step,
            )
        keys = {status_icon(s).cacheKey()
                for s in (STATUS_DONE, STATUS_TODO, STATUS_INVALID)}
        self.assertEqual(len(keys), 3)

    def test_an_invalid_step_shows_its_issues(self):
        self.project.set_template(make_settings(Y))
        self.app.processEvents()
        self.assertTrue(self.tab.go_to_step(STEP_ROIS))
        text = self.tab.issues_label.text()
        self.assertIn("arm_b", text)
        row = self.tab.steps.index(STEP_ROIS)
        self.assertIn("arm_b", self.tab.step_list.item(row).toolTip())

    def test_steps_can_be_visited_in_any_order(self):
        for step in (STEP_REVIEW, STEP_TIME_WINDOW, STEP_PARADIGM_PARAMS,
                     STEP_DETECTION_TUNING, STEP_ROIS, STEP_CALIBRATION):
            with self.subTest(step=step):
                self.assertTrue(self.tab.go_to_step(step))
                self.assertEqual(self.tab.step, step)
                self.assertEqual(self.tab.stack.count(), 2)
        self.assertFalse(self.tab.go_to_step(STEP_TST_TUNING))   # not a Y-maze step

    def test_previous_and_next_walk_the_list(self):
        tab = self.tab
        tab.go_to_step(tab.steps[0])
        self.assertFalse(tab.previous_button.isEnabled())
        self.assertTrue(tab.next_button.isEnabled())
        tab.next_button.click()
        self.assertEqual(tab.step, tab.steps[1])
        tab.previous_button.click()
        self.assertEqual(tab.step, tab.steps[0])
        tab.go_to_step(tab.steps[-1])
        self.assertFalse(tab.next_button.isEnabled())

    def test_clicking_a_step_in_the_list_opens_it(self):
        tab = self.tab
        tab.step_list.setCurrentRow(tab.steps.index(STEP_TIME_WINDOW))
        self.assertEqual(tab.step, STEP_TIME_WINDOW)


class ScopeTests(SetupTabFixture):
    def setUp(self):
        super().setUp()
        self.project.set_template(valid_settings(Y))
        self.app.processEvents()

    def test_template_header_counts_the_videos_using_it(self):
        self.assertEqual(self.tab.scope, SCOPE_TEMPLATE)
        self.assertEqual(
            self.tab.header_text(),
            "Project template (applies to 2 videos using the template)",
        )
        self.project.customize(self.keys[0], valid_settings(
            Y, start_time=1.0))
        self.app.processEvents()
        self.assertIn("applies to 1 video using", self.tab.header_text())
        self.assertIn("applies to 1 video using", self.tab.header_label.text())

    def test_selected_scope_edits_the_selected_video(self):
        self.select([self.keys[1]])
        self.assertTrue(self.tab.set_scope(SCOPE_VIDEOS))
        self.assertEqual(self.tab.targets, [self.keys[1]])
        self.assertTrue(self.tab.header_text().startswith(
            "Custom settings for clip2.avi"))
        self.assertIn("clip2.avi", self.tab.header_label.text())
        self.assertTrue(self.tab.reference_row.isHidden())

    def test_selected_scope_without_a_selection_says_so(self):
        self.tab.set_scope(SCOPE_VIDEOS)
        self.assertIs(self.tab.stack.currentWidget(), self.tab.empty_label)
        self.assertIn("Select", self.tab.empty_label.text())
        self.select([self.keys[0]])
        self.assertIsNot(self.tab.stack.currentWidget(), self.tab.empty_label)

    def test_switching_to_custom_copies_the_template_values(self):
        self.select([self.keys[0]])
        self.tab.set_scope(SCOPE_VIDEOS)
        template = self.project.template_settings.to_dict()
        self.assertEqual(
            self.tab.state.to_batch_settings(finalize=True).to_dict(), template
        )
        entry = self.project.entry(self.keys[0])
        self.assertEqual(entry.settings_source, SOURCE_TEMPLATE)

    def test_applying_an_unchanged_step_keeps_following_the_template(self):
        self.select([self.keys[0]])
        self.tab.set_scope(SCOPE_VIDEOS)
        self.tab.go_to_step(STEP_TIME_WINDOW)
        self.assertTrue(self.tab.apply_step())
        self.assertEqual(
            self.project.entry(self.keys[0]).settings_source, SOURCE_TEMPLATE
        )

    def test_editing_a_selected_video_writes_custom_settings(self):
        self.select([self.keys[0]])
        self.tab.set_scope(SCOPE_VIDEOS)
        self.tab.go_to_step(STEP_TIME_WINDOW)
        panel = self.tab.host.panel
        panel.start_spinbox.setValue(0.5)
        self.assertTrue(self.tab.has_unapplied_edits())
        self.assertTrue(self.tab.apply_step())
        edited = self.project.entry(self.keys[0])
        other = self.project.entry(self.keys[1])
        self.assertEqual(edited.settings_source, SOURCE_CUSTOM)
        self.assertAlmostEqual(edited.settings.start_time, 0.5, places=2)
        self.assertEqual(other.settings_source, SOURCE_TEMPLATE)
        self.assertEqual(other.settings.start_time, 0.0)
        self.assertEqual(self.project.template_settings.start_time, 0.0)
        self.assertFalse(self.tab.has_unapplied_edits())

    def test_template_edits_reach_only_the_videos_using_the_template(self):
        self.project.customize(self.keys[0], valid_settings(
            Y, start_time=1.0))
        self.tab.go_to_step(STEP_TIME_WINDOW)
        self.tab.host.panel.start_spinbox.setValue(0.4)
        self.assertTrue(self.tab.apply_step())
        template = self.project.template_settings
        self.assertAlmostEqual(template.start_time, 0.4, places=2)
        self.assertAlmostEqual(
            self.project.entry(self.keys[1]).settings.start_time, 0.4, places=2
        )
        self.assertEqual(self.project.entry(self.keys[0]).settings.start_time, 1.0)
        self.assertEqual(
            self.project.entry(self.keys[1]).settings_source, SOURCE_TEMPLATE
        )

    def test_multiple_videos_get_only_the_applied_step(self):
        # Give the second video different regions, then edit both together.
        maze = create_maze(Y)
        other = valid_settings(Y, roi_manager=valid_manager(maze, shift=5))
        self.project.customize(self.keys[1], other)
        self.select(self.keys)
        self.tab.set_scope(SCOPE_VIDEOS)
        self.assertEqual(len(self.tab.targets), 2)
        self.assertIn("2 selected videos", self.tab.header_text())
        self.tab.go_to_step(STEP_TIME_WINDOW)
        self.tab.host.panel.start_spinbox.setValue(0.25)
        self.assertTrue(self.tab.apply_step())
        first = self.project.entry(self.keys[0]).settings
        second = self.project.entry(self.keys[1]).settings
        self.assertAlmostEqual(first.start_time, 0.25, places=2)
        self.assertAlmostEqual(second.start_time, 0.25, places=2)
        self.assertEqual(
            second.roi_manager.rois["arm_a"][0].points.tolist(),
            [list(p) for p in square(5)],
        )

    def test_mark_selected_verified(self):
        self.assertFalse(self.tab.verify_button.isEnabled())
        self.select([self.keys[0]])
        self.assertTrue(self.tab.verify_button.isEnabled())
        self.tab.verify_button.click()
        self.assertTrue(self.project.entry(self.keys[0]).verified)
        self.assertFalse(self.project.entry(self.keys[1]).verified)

    def test_changing_the_paradigm_resets_the_tab(self):
        self.prompts.say("confirm", True)
        self.choose_paradigm(E)
        self.assertEqual(self.tab.steps, create_maze(E).setup_steps())
        self.assertIsNone(self.project.template_settings)
        self.assertEqual(self.tab.step, STEP_PARADIGM_PARAMS)


class StepHostingTests(SetupTabFixture):
    def test_each_step_gets_its_own_widget(self):
        self.project.set_template(valid_settings(Y))
        expected = {
            STEP_PARADIGM_PARAMS: PanelHost,
            STEP_ROIS: RoiHost,
            STEP_CALIBRATION: CalibrationHost,
            STEP_DETECTION_TUNING: DetectionHost,
            STEP_TIME_WINDOW: PanelHost,
            STEP_REVIEW: ReviewHost,
        }
        for step, kind in expected.items():
            with self.subTest(step=step):
                self.assertTrue(self.tab.go_to_step(step))
                self.assertIs(type(self.tab.host), kind)
                self.assertEqual(self.tab.stack.count(), 2)

    def test_the_tst_step_hosts_the_tst_tuner(self):
        self.choose_paradigm(T)
        self.project.set_template(valid_settings(T))
        self.assertTrue(self.tab.go_to_step(STEP_TST_TUNING))
        self.assertIsInstance(self.tab.host, TstHost)

    def test_leaving_a_step_shuts_its_tuner_down(self):
        self.project.set_template(valid_settings(Y))
        self.tab.go_to_step(STEP_DETECTION_TUNING)
        host = self.tab.host
        tuner = host.tuner
        self.assertFalse(tuner._lifecycle.is_shut_down)
        self.tab.go_to_step(STEP_TIME_WINDOW)
        self.assertTrue(tuner._lifecycle.is_shut_down)
        self.assertFalse(tuner._lifecycle.has_running_thread())
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.assertTrue(sip.isdeleted(host))
        self.assertEqual(self.tab.stack.count(), 2)

    def test_leaving_the_tst_step_joins_the_scan_thread(self):
        self.choose_paradigm(T)
        self.project.set_template(valid_settings(T))
        # No stage hosts the setup tab any more: show it on its own so the
        # TST tuner (which scans on first show) starts.
        self.tab.setParent(None)
        self.addCleanup(self.tab.hide)
        self.tab.show()
        self.assertTrue(self.tab.go_to_step(STEP_TST_TUNING))
        tuner = self.tab.host.tuner
        self.assertTrue(wait_until(
            self.app, lambda: tuner.scanner is not None
        ))
        scanner = tuner.scanner
        self.tab.go_to_step(STEP_ROIS)
        self.assertTrue(tuner._lifecycle.is_shut_down)
        self.assertFalse(scanner.isRunning())

    def test_the_roi_editor_shows_the_reference_video(self):
        self.tab.go_to_step(STEP_ROIS)
        editor = self.tab.host.editor
        self.assertEqual(editor.source_size, SIZE)
        self.assertEqual(self.tab.reference_combo.count(), 2)
        self.assertEqual(self.tab.reference_combo.currentData(), self.keys[0])

    def test_the_reference_video_can_be_chosen(self):
        self.tab.go_to_step(STEP_ROIS)
        first = self.tab.host
        self.assertTrue(self.tab.set_reference_video(self.keys[1]))
        self.assertIsNot(self.tab.host, first)
        self.assertEqual(
            Path(self.tab.state.video_path).name, "clip2.avi"
        )

    def test_a_selected_video_is_its_own_reference(self):
        self.project.set_template(valid_settings(Y))
        self.select([self.keys[1]])
        self.tab.set_scope(SCOPE_VIDEOS)
        self.assertEqual(Path(self.tab.state.video_path).name, "clip2.avi")

    def test_an_unreadable_video_gives_a_message_not_a_crash(self):
        fake = self.write_video("fake/broken.avi", b"not a video")
        self.project.add_videos([fake])
        self.app.processEvents()
        self.assertTrue(self.tab.go_to_step(STEP_ROIS))
        self.tab.set_reference_video(self.project.keys()[-1])
        self.assertIsInstance(self.tab.host, MessageHost)
        self.assertIn("broken.avi", self.tab.host.text)
        self.assertFalse(self.tab.apply_button.isEnabled())

    def test_tuning_needs_regions_first(self):
        self.assertTrue(self.tab.go_to_step(STEP_DETECTION_TUNING))
        self.assertIsInstance(self.tab.host, MessageHost)
        self.assertIn("regions", self.tab.host.text)

    def test_tuning_unlocks_once_the_regions_exist(self):
        self.tab.go_to_step(STEP_DETECTION_TUNING)
        self.assertIsInstance(self.tab.host, MessageHost)
        self.tab.go_to_step(STEP_ROIS)
        draw_zones(self.tab.host.editor,
                   ["arm_a", "arm_b", "arm_c", "center"])
        self.assertTrue(self.tab.apply_step())
        self.tab.go_to_step(STEP_DETECTION_TUNING)
        self.assertIsInstance(self.tab.host, DetectionHost)

    def test_the_reference_frame_is_generated_off_the_gui_thread(self):
        from unittest.mock import patch
        self.project.set_template(valid_settings(Y))
        with patch(
            "gui.workspace.setup_tab.run_with_progress",
            side_effect=lambda parent, label, fn, title="": fn(lambda: False),
        ) as runner:
            self.tab.go_to_step(STEP_DETECTION_TUNING)
        self.assertEqual(runner.call_count, 1)
        self.assertIsInstance(self.tab.host, DetectionHost)

    def test_a_cancelled_reference_frame_offers_a_retry(self):
        from gui.background_task import TaskCancelled
        from unittest.mock import patch
        self.project.set_template(valid_settings(Y))
        with patch("gui.workspace.setup_tab.run_with_progress",
                   side_effect=TaskCancelled()):
            self.tab.go_to_step(STEP_DETECTION_TUNING)
        self.assertIsInstance(self.tab.host, MessageHost)
        self.assertFalse(self.tab.host.retry_button.isHidden())
        self.tab.host.retry_button.click()
        self.assertIsInstance(self.tab.host, DetectionHost)

    def test_calibration_can_jump_to_the_line_tool(self):
        self.project.set_template(valid_settings(Y))
        self.tab.go_to_step(STEP_CALIBRATION)
        self.tab.host.draw_line_button.click()
        self.assertEqual(self.tab.step, STEP_ROIS)
        self.assertEqual(self.tab.host.editor.tool(), TOOL_LINE)

    def test_the_review_lists_problems_of_the_videos(self):
        self.project.set_template(valid_settings(Y))
        fake = self.write_video("fake/gone.avi", b"x")
        self.project.add_videos([fake])
        fake.unlink()
        self.app.processEvents()
        self.tab.go_to_step(STEP_REVIEW)
        lines = self.tab.host.project_issue_lines()
        self.assertTrue(any("gone.avi" in line for line in lines))
        self.assertIn("gone.avi", self.tab.host.project_label.text())

    def test_the_review_lists_the_mazes_own_findings(self):
        self.project.set_template(valid_settings(Y))
        self.project.customize(self.keys[1], make_settings(Y))   # arms missing
        self.app.processEvents()
        self.tab.go_to_step(STEP_REVIEW)
        lines = self.tab.host.project_issue_lines()
        self.assertTrue(any("clip2.avi" in line and "arm_b" in line
                            for line in lines), lines)

    def test_applying_before_the_regions_keeps_a_draft(self):
        self.tab.go_to_step(STEP_PARADIGM_PARAMS)
        spin = self.tab.host.panel.active_panel.dwell_spin
        spin.setValue(0.4)
        self.assertTrue(self.tab.apply_step())
        self.assertIsNone(self.project.template_settings)
        self.tab.go_to_step(STEP_ROIS)
        self.tab.go_to_step(STEP_PARADIGM_PARAMS)
        self.assertAlmostEqual(
            self.tab.host.panel.active_panel.dwell_spin.value(), 0.4)
        self.tab.go_to_step(STEP_ROIS)
        draw_zones(self.tab.host.editor,
                   ["arm_a", "arm_b", "arm_c", "center"])
        self.assertTrue(self.tab.apply_step())
        params = self.project.template_settings.algorithm_parameters
        self.assertAlmostEqual(params["min_dwell_seconds"], 0.4)

    def test_blocking_issues_keep_a_step_from_applying(self):
        self.tab.go_to_step(STEP_ROIS)
        draw_zones(self.tab.host.editor, ["arm_a"])
        self.assertFalse(self.tab.apply_step())
        self.assertIn("Arm B", self.tab.issues_label.text())
        self.assertIsNone(self.project.template_settings)

    def test_revert_discards_the_unapplied_edits(self):
        self.project.set_template(valid_settings(Y))
        self.tab.go_to_step(STEP_TIME_WINDOW)
        self.tab.host.panel.start_spinbox.setValue(0.8)
        self.assertTrue(self.tab.revert_button.isEnabled())
        self.tab.revert_button.click()
        self.assertEqual(self.tab.host.panel.start_spinbox.value(), 0.0)
        self.assertFalse(self.tab.has_unapplied_edits())


class CompletionTests(SetupTabFixture):
    def test_a_loaded_template_counts_as_confirmed(self):
        self.project.set_template(valid_settings(Y))
        self.app.processEvents()
        tab = self.tab
        for step in tab.steps:
            expected = STATUS_TODO if step == STEP_REVIEW else STATUS_DONE
            self.assertEqual(tab.step_status(step), expected, step)
        self.assertEqual(tab.step, STEP_PARADIGM_PARAMS)   # stays put

    def test_the_review_completes_when_everything_is_done(self):
        self.project.set_template(valid_settings(Y))
        self.tab.go_to_step(STEP_REVIEW)
        self.tab.host.panel.timelapse_check.setChecked(True)
        self.assertTrue(self.tab.apply_step())
        self.assertEqual(self.tab.step_status(STEP_REVIEW), STATUS_DONE)
        self.assertTrue(self.project.template_settings.create_timelapse)

    def test_a_new_project_forgets_the_old_progress(self):
        self.project.set_template(valid_settings(Y))
        self.assertEqual(self.tab.step_status(STEP_ROIS), STATUS_DONE)
        self.prompts.say("ask_save_changes", DISCARD)
        self.assertTrue(self.window.close_project())
        self.new_project_at("another.bmaproj")
        self.assertIs(self.tab.stack.currentWidget(), self.tab.empty_label)


if __name__ == "__main__":
    unittest.main()
