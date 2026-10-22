"""The Detection stage (Motion tuning for the Tail Suspension Test).

Covers the simple-mode mappings, the live writes to the project template and
their equivalence with the classic tuner, the video list and the quick check
of every video, and the tuner lifecycle (threads joined, no leaks).
"""
import copy
import threading
import time
import unittest
from unittest.mock import patch

from core.models import AppConfig, DetectionConfig
from core.project import Project, canonical_json
from core.setup_state import SetupState
from core.tracking_quality import summarize_tracking_quality
from core.video import ReferenceFrameGenerator
from mazes.base_maze import STEP_DETECTION_TUNING
from tests.setup_tab_support import (T, Y, SetupTabFixture, wait_until)
from tests.test_workspace_setup_tab import valid_settings
from tests.workspace_support import HAS_RUNTIME

if HAS_RUNTIME:
    from PyQt6 import sip
    from PyQt6.QtCore import QCoreApplication, QEvent
    from core.models import DetectionConfig  # noqa: F811
    from gui.tuners import detection_tuner as dt
    from gui.workspace.stages import detection_stage as ds
    from gui.workspace.stages import status as st
    from gui.workspace.stages.detection_stage import (PAGE_EMPTY, PAGE_WORK,
                                                      QuickCheckJob,
                                                      QuickCheckResult,
                                                      check_video)
    from gui.tuner_dialog import TunerDialog

GOOD = DetectionConfig(
    threshold_percentile=95.0, use_weighting=False, min_component_area=20,
    morphology_kernel_size=3, min_detection_confidence=0.1,
)
TUNING = {"Min component area": 55, "Threshold percentile": 980}


def deliver_deletes() -> None:
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


# ----------------------------------------------------------------------
# Qt-free: the simple-mode mappings and the quality grading
# ----------------------------------------------------------------------
class SimpleModeMappingTests(unittest.TestCase):
    def setUp(self):
        if not HAS_RUNTIME:
            self.skipTest("GUI runtime is not installed")

    def test_sensitivity_is_monotonic_and_inside_the_slider_range(self):
        levels = dt.SENSITIVITY_LEVELS
        low, high = (spec for spec in dt._SLIDER_SPECS
                     if spec[1] == "threshold_percentile").__next__()[2:4]
        for before, after in zip(levels, levels[1:]):
            self.assertGreater(before, after, "more sensitive = lower percentile")
        for percentile in levels:
            self.assertGreaterEqual(percentile * 10, low)
            self.assertLessEqual(percentile * 10, high)

    def test_the_defaults_are_the_middle_positions(self):
        state = dt.simple_state_of(DetectionConfig())
        self.assertEqual(state.sensitivity, dt.DEFAULT_SENSITIVITY_LEVEL)
        self.assertEqual(state.size, "medium")

    def test_every_simple_position_is_a_valid_config(self):
        specs = {spec[1]: spec for spec in dt._SLIDER_SPECS}
        for level in range(len(dt.SENSITIVITY_LEVELS)):
            for size in dt.SIZE_ORDER:
                config = DetectionConfig()
                config.threshold_percentile = dt.sensitivity_percentile(level)
                area, window = dt.animal_size_fields(size)
                config.min_component_area, config.window_size = area, window
                config.validate()                       # does not raise
                for field in ("min_component_area", "window_size"):
                    _n, _f, low, high, scale, *_ = specs[field]
                    self.assertTrue(low <= getattr(config, field) * scale <= high)
                self.assertEqual(dt.sensitivity_level_of(config), level)
                self.assertEqual(dt.animal_size_of(config), size)

    def test_sizes_grow_from_small_to_large(self):
        areas = [dt.animal_size_fields(s) for s in dt.SIZE_ORDER]
        self.assertEqual(areas, sorted(areas))
        self.assertEqual(len(set(areas)), 3)

    def test_a_value_between_positions_is_custom_but_has_a_nearest(self):
        config = DetectionConfig(
            threshold_percentile=98.65, min_component_area=33, window_size=117)
        state = dt.simple_state_of(config)
        self.assertIsNone(state.sensitivity)
        self.assertIsNone(state.size)
        self.assertEqual(dt.sensitivity_percentile(state.nearest_sensitivity), 99.0)
        self.assertEqual(state.nearest_size, "medium")

    def test_quality_levels_and_the_sentence(self):
        windows = [[(True, 0.9, (10 + i, 10)) for i in range(15)]
                   for _ in range(3)]
        summary = dt.summarize_windows(windows, 200.0)
        self.assertEqual((summary.level, summary.label), ("good", "Good"))
        self.assertEqual(summary.explanation,
                         "Animal found in 100% of sampled frames")
        nothing = [[(False, 0.0, None)] * 15] * 3
        poor = dt.summarize_windows(nothing, 200.0)
        self.assertEqual(poor.level, "poor")
        self.assertIn("0%", poor.explanation)
        self.assertIsNone(dt.summarize_windows([[(True, 1.0, (1, 1))]], 100.0))

    def test_a_jump_between_windows_is_not_a_tracking_error(self):
        near = [(True, 0.9, (10 + i, 10)) for i in range(15)]
        far = [(True, 0.9, (150 + i, 140)) for i in range(15)]
        flat = dt.summarize_windows([near + far], 200.0)         # one run
        split = dt.summarize_windows([near, far], 200.0)         # two windows
        self.assertEqual(flat.level, "check")
        self.assertEqual(split.level, "good")

    def test_break_before_defaults_to_the_old_behaviour(self):
        positions = [(0, 0), (100, 100), (101, 100)]
        confidences = [0.9] * 3
        plain = summarize_tracking_quality(
            positions, confidences, frame_diagonal=200.0)
        same = summarize_tracking_quality(
            positions, confidences, frame_diagonal=200.0, break_before=())
        cut = summarize_tracking_quality(
            positions, confidences, frame_diagonal=200.0, break_before=(1,))
        self.assertEqual(plain, same)
        self.assertGreater(plain["implausible_jump_rate"], 0)
        self.assertEqual(cut["implausible_jump_rate"], 0)


# ----------------------------------------------------------------------
# The project field
# ----------------------------------------------------------------------
class DetectionReviewedTests(unittest.TestCase):
    def test_defaults_to_false_and_is_not_written_then(self):
        project = Project(name="p", paradigm="Y-Maze")
        self.assertFalse(project.detection_reviewed)
        self.assertNotIn("detection_reviewed", project.to_dict())
        self.assertFalse(Project.from_dict(project.to_dict()).detection_reviewed)

    def test_round_trips_when_true(self):
        project = Project(name="p", paradigm="Y-Maze")
        self.assertTrue(project.set_detection_reviewed())
        self.assertFalse(project.set_detection_reviewed())     # no change
        data = project.to_dict()
        self.assertIs(data["detection_reviewed"], True)
        self.assertTrue(Project.from_dict(data).detection_reviewed)

    def test_older_files_and_odd_values_read_as_false(self):
        data = Project(name="p").to_dict()
        for value in ("yes", 1, None, [True]):
            data["detection_reviewed"] = value
            self.assertFalse(Project.from_dict(data).detection_reviewed)

    def test_listeners_hear_about_it(self):
        project = Project(name="p")
        heard = []
        project.add_listener(lambda: heard.append(1))
        project.set_detection_reviewed()
        project.set_detection_reviewed()
        self.assertEqual(len(heard), 1)

    def test_changing_the_paradigm_clears_it(self):
        from tests.project_support import make_settings
        project = Project(name="p", paradigm="Y-Maze")
        project.set_template(make_settings("Y-Maze"))
        project.set_detection_reviewed()
        project.set_paradigm("Elevated Plus Maze")
        self.assertFalse(project.detection_reviewed)

    def test_the_status_needs_the_review(self):
        from tests.test_workspace_stage_status import settings_for
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            video = Path(tmp) / "v.mp4"
            video.write_bytes(b"x")
            project = Project(name="p", paradigm="Y-Maze")
            project.set_template(settings_for("Y-Maze"))
            project.add_videos([video])
            found = st.stage_status(st.STAGE_DETECTION, project)
            self.assertEqual(found.status, st.STATUS_TODO)
            project.set_detection_reviewed()
            found = st.stage_status(st.STAGE_DETECTION, project)
            self.assertEqual(found.status, st.STATUS_DONE)
            project.template_settings.detection_config.threshold_percentile = 150.0
            found = st.stage_status(st.STAGE_DETECTION, project)
            self.assertEqual(found.status, st.STATUS_ATTENTION)


# ----------------------------------------------------------------------
# The embedded tuner's simple mode
# ----------------------------------------------------------------------
class EmbeddedTunerTests(SetupTabFixture):
    def make(self, config=None):
        template = valid_settings(Y)
        tuner = dt.DetectionTunerWidget(
            str(self.clips[0]),
            ReferenceFrameGenerator.generate(self.clips[0], 10, use_cache=False),
            template.roi_manager, initial_config=config, embedded=True)
        self.addCleanup(tuner.shutdown)
        return tuner

    def test_exact_values_survive_loading_and_reading_back(self):
        config = DetectionConfig(
            threshold_percentile=98.65, min_component_area=33, window_size=117)
        tuner = self.make(config)
        self.assertEqual(tuner.detection_config(), config)
        state = tuner.simple_state()
        self.assertIsNone(state.sensitivity)
        self.assertIsNone(state.size)
        self.assertEqual(tuner.detection_config(), config)

    def test_touching_one_control_leaves_the_other_values_alone(self):
        config = DetectionConfig(
            threshold_percentile=98.65, min_component_area=33, window_size=117)
        tuner = self.make(config)
        tuner.set_sensitivity(3)
        got = tuner.detection_config()
        self.assertEqual(got.threshold_percentile, 99.0)
        self.assertEqual((got.min_component_area, got.window_size), (33, 117))
        tuner.set_animal_size("small")
        got = tuner.detection_config()
        self.assertEqual(got.threshold_percentile, 99.0)
        self.assertEqual((got.min_component_area, got.window_size), (10, 60))
        self.assertEqual(tuner.simple_state().size, "small")

    def test_a_position_the_slider_already_shows_still_applies_exactly(self):
        # 99.04 rounds to the slider step 990, so a plain slider move would
        # not fire; the simple control must still write 99.0.
        tuner = self.make(DetectionConfig(threshold_percentile=99.04))
        self.assertIsNone(tuner.simple_state().sensitivity)
        tuner.set_sensitivity(3)
        self.assertEqual(tuner.detection_config().threshold_percentile, 99.0)

    def test_simple_and_advanced_stay_in_step(self):
        tuner = self.make()
        tuner.set_sensitivity(5)
        self.assertEqual(tuner.sliders["Threshold percentile"].value(), 970)
        self.assertIn("97.0", tuner.slider_labels["Threshold percentile"].text())
        tuner.sliders["Threshold percentile"].setValue(980)
        self.assertEqual(tuner.simple_state().sensitivity, 4)
        tuner.sliders["Threshold percentile"].setValue(983)
        self.assertIsNone(tuner.simple_state().sensitivity)

    def test_config_changed_fires_for_simple_edits(self):
        tuner = self.make()
        heard = []
        tuner.config_changed.connect(heard.append)
        tuner.set_animal_size("large")
        self.assertEqual(len(heard), 1)
        self.assertEqual(heard[0].min_component_area, 60)

    def test_the_mask_view_swaps_the_hero_image(self):
        tuner = self.make()
        before = tuner.main_view._source.toImage()
        tuner.set_view_mode(dt.VIEW_MASK)
        after = tuner.main_view._source.toImage()
        self.assertNotEqual(before, after)
        tuner.set_view_mode(dt.VIEW_VIDEO)
        self.assertEqual(tuner.main_view._source.toImage(), before)
        with self.assertRaises(ValueError):
            tuner.set_view_mode("nope")

    def test_the_embedded_layout_has_one_preview_and_hands_out_the_sliders(self):
        tuner = self.make()
        self.assertIsNone(tuner.diff_view)
        self.assertIsNone(tuner.control_scroll)
        self.assertEqual(len(tuner.sliders), 6)
        self.assertIsNone(tuner.advanced_panel.parentWidget())

    def test_the_classic_layout_is_unchanged(self):
        tuner = dt.DetectionTunerWidget(
            str(self.clips[0]),
            ReferenceFrameGenerator.generate(self.clips[0], 10, use_cache=False),
            valid_settings(Y).roi_manager)
        self.addCleanup(tuner.shutdown)
        self.assertFalse(tuner.embedded)
        self.assertIsNotNone(tuner.diff_view)
        self.assertIsNotNone(tuner.control_scroll)


# ----------------------------------------------------------------------
# The stage
# ----------------------------------------------------------------------
class DetectionStageFixture(SetupTabFixture):
    TEMPLATE = True
    DETECTION = GOOD

    def setUp(self):
        super().setUp()
        self.window.resize(1366, 768)
        self.window.show()
        self.stage = self.window.stage("detection")
        if self.TEMPLATE:
            self.project.set_template(valid_settings(
                self.PARADIGM, detection_config=copy.copy(self.DETECTION)))
        self.app.processEvents()

    def tearDown(self):
        # Leave the stage so no thread outlives the test.
        if self.window is not None and self.window.current_stage_id == "detection":
            self.window.go_to_stage("videos", force=True)
        super().tearDown()

    def enter(self):
        self.assertTrue(self.window.go_to_stage("detection"))
        self.app.processEvents()
        return self.stage

    @property
    def template(self):
        return self.project.template_settings

    def written(self, predicate, seconds=5.0):
        return wait_until(self.app, predicate, seconds)

    def quality_ready(self):
        stage = self.stage
        return self.written(
            lambda: stage.simple.quality_chip.kind() != "running", 20.0)


class StageLayoutTests(DetectionStageFixture):
    def test_the_preview_is_large(self):
        stage = self.enter()
        self.assertEqual(stage.pages.currentIndex(), PAGE_WORK)
        tuner = stage.tuner
        self.assertIsNotNone(tuner)
        self.assertGreaterEqual(tuner.width(), 0.65 * self.window.width())
        self.assertLessEqual(stage.side_scroll.width(), 360)
        self.assertGreater(tuner.main_view.height(), 200)

    def test_header_texts(self):
        stage = self.enter()
        self.assertEqual(stage.header.title(), "Detection")
        self.assertEqual(self.window.stepper.label("detection"), "Detection")

    def test_the_first_video_is_previewed_and_listed(self):
        stage = self.enter()
        self.assertEqual(stage.previewed_key, self.keys[0])
        self.assertEqual(stage.filmstrip.keys(), self.keys)
        self.assertEqual(stage.filmstrip.selected_key(), self.keys[0])
        self.assertTrue(stage.tuner.video_path.endswith("clip1.avi"))

    def test_the_selected_video_is_previewed(self):
        self.select([self.keys[1]])
        stage = self.enter()
        self.assertEqual(stage.previewed_key, self.keys[1])

    def test_advanced_holds_the_sliders_and_starts_collapsed(self):
        stage = self.enter()
        self.assertFalse(stage.advanced.is_expanded())
        panel = stage.tuner.advanced_panel
        self.assertTrue(stage.advanced.isAncestorOf(panel))
        self.assertEqual(len(stage.tuner.sliders), 6)

    def test_the_mask_toggle_switches_the_hero(self):
        stage = self.enter()
        self.assertEqual(stage.tuner.view_mode, dt.VIEW_VIDEO)
        stage.mask_button.click()
        self.assertEqual(stage.tuner.view_mode, dt.VIEW_MASK)
        # the choice survives a change of video
        stage.show_video(self.keys[1])
        self.assertEqual(stage.tuner.view_mode, dt.VIEW_MASK)
        self.assertTrue(stage.mask_button.isChecked())
        stage.mask_button.click()
        self.assertEqual(stage.tuner.view_mode, dt.VIEW_VIDEO)

    def test_the_next_button_leads_to_trials(self):
        stage = self.enter()
        self.assertEqual(stage.next_label(), "Trials")
        self.assertEqual(self.window.footer.next_text(), "Next: Trials →")


class StageBlockedTests(DetectionStageFixture):
    TEMPLATE = False

    def test_without_zones_it_points_at_arena_and_zones(self):
        stage = self.enter()
        self.assertEqual(stage.pages.currentIndex(), PAGE_EMPTY)
        self.assertIsNone(stage.tuner)
        self.assertIn("zones", stage.empty.title())
        button = stage.empty.buttons[0]
        self.assertEqual(button.text(), "Go to Arena && zones")
        allowed, reason = stage.can_advance()
        self.assertFalse(allowed)
        self.assertIn("zones", reason)
        button.click()
        self.assertEqual(self.window.current_stage_id, "arena")

    def test_zones_appearing_later_show_the_preview(self):
        stage = self.enter()
        self.assertIsNone(stage.tuner)
        self.window.go_to_stage("videos", force=True)
        self.project.set_template(valid_settings(Y))
        stage = self.enter()
        self.assertIsNotNone(stage.tuner)


class StageLiveWriteTests(DetectionStageFixture):
    def test_a_slider_edit_reaches_the_template_without_apply(self):
        stage = self.enter()
        stage.tuner.sliders["Min component area"].setValue(55)
        self.assertTrue(self.written(
            lambda: self.template.detection_config.min_component_area == 55))
        for key in self.keys:       # the videos follow the template
            self.assertEqual(
                self.project.effective_settings(key).detection_config
                .min_component_area, 55)

    def test_drags_are_debounced_into_one_write(self):
        stage = self.enter()
        heard = []
        self.project.add_listener(lambda: heard.append(1))
        slider = stage.tuner.sliders["Min component area"]
        for value in range(30, 60, 3):
            slider.setValue(value)
        self.assertEqual(heard, [], "nothing is written while dragging")
        self.assertTrue(self.written(
            lambda: self.template.detection_config.min_component_area == 57))
        self.assertTrue(self.written(lambda: not stage._write_timer.isActive()))
        self.assertLessEqual(len(heard), 2)      # the template (+ "reviewed")

    def test_leaving_flushes_a_pending_edit(self):
        stage = self.enter()
        stage.tuner.sliders["Min component area"].setValue(77)
        self.assertNotEqual(self.template.detection_config.min_component_area, 77)
        self.window.go_to_stage("trials")
        self.assertEqual(self.template.detection_config.min_component_area, 77)

    def test_the_simple_controls_write_valid_values(self):
        stage = self.enter()
        card = stage.simple
        for level in range(len(dt.SENSITIVITY_LEVELS)):
            card.sensitivity_slider.setValue(level)
            stage.commit_pending()
            config = self.template.detection_config
            config.validate()
            self.assertEqual(config.threshold_percentile,
                             dt.SENSITIVITY_LEVELS[level])
        for size in dt.SIZE_ORDER:
            card.size_buttons[size].click()
            stage.commit_pending()
            config = self.template.detection_config
            config.validate()
            self.assertEqual(
                (config.min_component_area, config.window_size),
                dt.ANIMAL_SIZES[size])
            self.assertEqual(
                [b.isChecked() for b in card.size_buttons.values()],
                [name == size for name in dt.SIZE_ORDER])

    def test_advanced_edits_update_the_simple_controls(self):
        stage = self.enter()
        stage.tuner.sliders["Threshold percentile"].setValue(970)
        self.assertEqual(stage.simple.sensitivity_slider.value(), 5)
        self.assertEqual(stage.simple.sensitivity_value.text(), "High")
        stage.tuner.sliders["Threshold percentile"].setValue(973)
        self.assertEqual(stage.simple.sensitivity_value.text(), ds.CUSTOM_TEXT)

    def test_simple_edits_update_the_advanced_sliders(self):
        stage = self.enter()
        stage.simple.size_buttons["large"].click()
        self.assertEqual(stage.tuner.sliders["Min component area"].value(), 60)
        self.assertEqual(stage.tuner.sliders["Search window"].value(), 160)

    def test_custom_values_are_kept_until_a_simple_control_is_touched(self):
        custom = DetectionConfig(
            threshold_percentile=98.65, min_component_area=33, window_size=117,
            use_weighting=False, min_detection_confidence=0.1)
        self.project.set_template(valid_settings(Y, detection_config=custom))
        stage = self.enter()
        self.assertEqual(stage.simple.sensitivity_value.text(), ds.CUSTOM_TEXT)
        self.assertEqual(stage.simple.size_value.text(), ds.CUSTOM_TEXT)
        self.assertFalse(any(b.isChecked() for b in stage.simple.size_buttons.values()))
        self.window.go_to_stage("trials")        # leaving rewrites nothing
        self.assertEqual(self.template.detection_config, custom)
        stage = self.enter()
        stage.simple.size_buttons["small"].click()
        stage.commit_pending()
        got = self.template.detection_config
        self.assertEqual((got.min_component_area, got.window_size), (10, 60))
        self.assertEqual(got.threshold_percentile, 98.65)
        self.assertEqual(stage.simple.sensitivity_value.text(), ds.CUSTOM_TEXT)

    def test_the_review_is_recorded(self):
        self.assertFalse(self.project.detection_reviewed)
        self.enter()
        self.window.go_to_stage("trials")
        self.assertTrue(self.project.detection_reviewed)
        self.window.refresh_navigation()
        self.assertEqual(self.window.stepper.status("detection"), "done")

    def test_a_project_edit_made_elsewhere_rebuilds_the_preview(self):
        stage = self.enter()
        before = stage.tuner
        changed = valid_settings(Y, detection_config=copy.copy(GOOD))
        changed.detection_config.min_component_area = 91
        self.project.set_template(changed)
        self.app.processEvents()
        self.assertIsNot(stage.tuner, before)
        self.assertEqual(
            stage.tuner.sliders["Min component area"].value(), 91)


class EquivalenceTests(DetectionStageFixture):
    def classic_config(self, sliders):
        frame = ReferenceFrameGenerator.generate(
            self.clips[0], AppConfig().video.ref_frame_samples, use_cache=True)
        dialog = TunerDialog(
            str(self.clips[0]), frame, self.template.roi_manager,
            initial_config=copy.copy(GOOD))
        self.addCleanup(dialog.deleteLater)
        for name, value in sliders.items():
            dialog.sliders[name].setValue(value)
        config = dialog.get_detection_config()
        dialog.widget.shutdown()
        return config

    def classic_settings(self, config):
        state = SetupState.from_batch_settings(
            self.template.copy(), maze_name=None)
        state.detection_config = config
        state.mark_done(STEP_DETECTION_TUNING)
        return state.to_batch_settings(finalize=True)

    def test_the_same_sliders_give_the_same_template_as_the_classic_tuner(self):
        classic = self.classic_settings(self.classic_config(TUNING))
        stage = self.enter()
        for name, value in TUNING.items():
            stage.tuner.sliders[name].setValue(value)
        stage.commit_pending()
        self.assertEqual(self.template.to_dict(), classic.to_dict())
        for key in self.keys:
            self.assertEqual(
                self.project.entry(key).settings.to_dict(), classic.to_dict())

    def test_the_simple_controls_match_the_classic_tuner_at_their_positions(self):
        sliders = {"Threshold percentile": 970, "Min component area": 60,
                   "Search window": 160}
        classic = self.classic_settings(self.classic_config(sliders))
        stage = self.enter()
        stage.simple.sensitivity_slider.setValue(5)         # 97.0
        stage.simple.size_buttons["large"].click()          # 60 / 160
        stage.commit_pending()
        self.assertEqual(self.template.to_dict(), classic.to_dict())

    def test_nothing_else_in_the_template_changes(self):
        before = self.template.copy()
        stage = self.enter()
        stage.tuner.sliders["Search window"].setValue(150)
        stage.commit_pending()
        after = self.template.to_dict()
        expected = before.to_dict()
        expected["detection_config"]["window_size"] = 150
        self.assertEqual(after, expected)

    def test_an_untouched_visit_does_not_rewrite_the_template(self):
        before = canonical_json(self.template.to_dict())
        self.enter()
        self.window.go_to_stage("trials")
        self.assertEqual(canonical_json(self.template.to_dict()), before)


class StageVideoTests(DetectionStageFixture):
    def test_picking_a_video_rebuilds_the_tuner_on_its_frame(self):
        stage = self.enter()
        first = stage.tuner
        stage.filmstrip.video_selected.emit(self.keys[1])
        self.assertIsNot(stage.tuner, first)
        self.assertTrue(first._lifecycle.is_shut_down)
        self.assertEqual(stage.previewed_key, self.keys[1])
        self.assertTrue(stage.tuner.video_path.endswith("clip2.avi"))
        self.assertEqual(stage.tuner.detection_config(), GOOD)

    def test_the_config_stays_the_shared_template(self):
        stage = self.enter()
        stage.tuner.sliders["Min component area"].setValue(66)
        stage.show_video(self.keys[1])         # flushes first
        self.assertEqual(self.template.detection_config.min_component_area, 66)
        self.assertEqual(
            stage.tuner.sliders["Min component area"].value(), 66)

    def test_the_reference_frame_is_made_through_run_with_progress_and_cached(self):
        calls = []
        real = ds.run_with_progress

        def spy(parent, label, fn, title="Please Wait"):
            calls.append((label, title))
            return real(parent, label, fn, title=title)

        with patch.object(ds, "run_with_progress", spy):
            stage = self.enter()
            self.assertEqual(len(calls), 1)
            self.assertIn("clip1.avi", calls[0][0])
            stage.show_video(self.keys[1])
            self.assertEqual(len(calls), 2)
            stage.show_video(self.keys[0])          # cached in memory
            self.assertEqual(len(calls), 2)

    def test_a_cancelled_frame_keeps_the_current_preview(self):
        stage = self.enter()
        first = stage.tuner

        def cancel(*_args, **_kwargs):
            raise ds.TaskCancelled()

        with patch.object(ds, "run_with_progress", cancel):
            self.assertFalse(stage.show_video(self.keys[1]))
        self.assertIs(stage.tuner, first)
        self.assertEqual(stage.filmstrip.selected_key(), self.keys[0])

    def test_an_unreadable_video_shows_a_message_and_a_retry(self):
        self.clips[0].write_bytes(b"not a video")
        stage = self.enter()
        self.assertIsNone(stage.tuner)
        self.assertFalse(stage.problem.isHidden())
        self.assertIn("could not be opened", stage.problem.title().lower())
        self.assertFalse(stage.simple.sensitivity_slider.isEnabled())
        self.assertTrue(stage.simple.sensitivity_slider.toolTip())
        # another video still works
        self.assertTrue(stage.show_video(self.keys[1]))
        self.assertTrue(stage.problem.isHidden())
        self.assertTrue(stage.simple.sensitivity_slider.isEnabled())

    def test_a_video_of_another_size_than_the_zones_is_flagged(self):
        stage = self.enter()
        self.assertTrue(stage.zones_note.isHidden())
        template = self.template.copy()
        template.roi_manager.set_source_dimensions((320, 240))
        self.project.set_template(template)
        stage = self.enter()
        self.assertFalse(stage.zones_note.isHidden())


class StageQualityTests(DetectionStageFixture):
    def test_the_chip_shows_good_check_or_poor_with_a_sentence(self):
        stage = self.enter()
        self.assertTrue(self.quality_ready())
        chip = stage.simple.quality_chip
        self.assertIn(chip.text(), ("Good", "Check", "Poor"))
        self.assertRegex(stage.simple.quality_text.text(),
                         r"^Animal found in \d+% of sampled frames")
        self.assertIn(stage.footer_text()[0][:6], "Animal")

    def test_a_good_config_is_good(self):
        stage = self.enter()
        self.assertTrue(self.quality_ready())
        self.assertEqual(stage.simple.quality_chip.text(), "Good")
        self.assertEqual(stage.simple.quality_chip.kind(), "ok")

    def test_a_hopeless_config_is_poor(self):
        stage = self.enter()
        self.assertTrue(self.quality_ready())
        stage.simple.size_buttons["large"].click()
        stage.tuner.sliders["Min component area"].setValue(2000)
        self.assertEqual(stage.simple.quality_chip.kind(), "running")
        self.assertTrue(self.quality_ready())
        self.assertEqual(stage.simple.quality_chip.text(), "Poor")
        self.assertEqual(stage.simple.quality_chip.kind(), "error")
        self.assertIn("0%", stage.simple.quality_text.text())


class QuickCheckTests(DetectionStageFixture):
    def finish_check(self, stage):
        self.assertTrue(stage.start_quick_check())
        self.assertTrue(stage.checking)
        self.assertFalse(stage.check_row.isHidden())
        self.assertTrue(self.written(lambda: not stage.checking, 60.0))

    def test_every_video_gets_a_chip_and_the_summary(self):
        stage = self.enter()
        self.finish_check(stage)
        items = stage.filmstrip.items()
        self.assertEqual(len(items), 2)
        for item in items:
            self.assertEqual((item.badge_text, item.badge_kind), ("Good", "ok"))
        counts = stage.check_counts()
        self.assertEqual(counts, (2, 0, 0))
        self.assertFalse(stage.summary.isHidden())
        self.assertEqual(stage.summary.kind(), "success")
        self.assertEqual(stage.footer_text()[0], "2 good · 0 to check")
        self.assertTrue(stage.check_row.isHidden())
        self.assertTrue(stage.check_button.isEnabled())
        self.assertTrue(self.project.detection_reviewed)

    def test_problem_videos_are_listed_and_counted(self):
        # An empty, unreadable file and a clip the detector cannot find.
        broken = self.root / "real" / "broken.avi"
        broken.write_bytes(b"nothing")
        self.project.add_videos([broken])
        self.app.processEvents()
        self.project.set_template(valid_settings(
            Y, detection_config=copy.copy(GOOD)))
        stage = self.enter()
        self.finish_check(stage)
        broken_key = self.project.keys()[-1]
        result = stage._results[broken_key]
        self.assertEqual(result.level, ds.LEVEL_ERROR)
        item = {i.key: i for i in stage.filmstrip.items()}[broken_key]
        self.assertEqual(item.badge_text, "Unreadable")
        self.assertEqual(stage.summary.kind(), "warning")
        self.assertIn("broken.avi", stage.summary.text())
        self.assertIn("1 of 3", stage.summary.text())
        good, to_check, unreadable = stage.check_counts()
        self.assertEqual((good, unreadable), (2, 1))
        self.assertRegex(stage.footer_text()[0],
                         r"^2 good · 0 to check · 1 unreadable$")

    def test_a_video_without_the_animal_is_poor(self):
        config = copy.copy(GOOD)
        config.min_component_area = 2000     # nothing is big enough
        self.project.set_template(valid_settings(Y, detection_config=config))
        stage = self.enter()
        self.finish_check(stage)
        levels = {r.level for r in stage._results.values()}
        self.assertEqual(levels, {ds.LEVEL_POOR})
        self.assertEqual(
            {i.badge_text for i in stage.filmstrip.items()}, {"Poor"})
        self.assertEqual(stage.check_counts(), (0, 2, 0))
        self.assertEqual(stage.footer_text()[0], "0 good · 2 to check")

    def test_check_video_alone_returns_a_result_or_none(self):
        job = QuickCheckJob(
            "k", self.clips[0], "clip1.avi", self.template.roi_manager)
        result = check_video(job, GOOD)
        self.assertIsInstance(result, QuickCheckResult)
        self.assertEqual(result.level, "good")
        self.assertGreater(result.detection_rate, 90)
        self.assertIsNone(check_video(job, GOOD, should_stop=lambda: True))

    def test_the_settings_changing_makes_the_chips_stale(self):
        stage = self.enter()
        self.finish_check(stage)
        stage.simple.sensitivity_slider.setValue(5)
        self.assertIsNone(stage.check_counts())
        self.assertEqual(
            {i.badge_text for i in stage.filmstrip.items()}, {""})
        self.assertFalse(stage.summary.isHidden())
        self.assertIn("changed", stage.summary.text())
        self.assertNotIn("good", stage.footer_text()[0])

    def test_cancel_stops_the_thread_and_keeps_what_was_checked(self):
        gate = threading.Event()
        seen = []

        def slow(job, config, should_stop=None):
            seen.append(job.key)
            if len(seen) == 1:
                return QuickCheckResult(job.key, ds.LEVEL_GOOD, 100.0)
            while not (should_stop and should_stop()):
                time.sleep(0.005)
                gate.set()
            return None

        stage = self.enter()
        with patch.object(ds, "check_video", slow):
            self.assertTrue(stage.start_quick_check())
            self.assertTrue(self.written(lambda: len(seen) == 2, 10.0))
            self.assertTrue(self.written(lambda: self.keys[0] in stage._results))
            stage.cancel_quick_check()
            self.assertTrue(self.written(lambda: not stage.checking, 10.0))
        self.assertFalse(stage._checks.has_running_thread())
        self.assertEqual(list(stage._results), [self.keys[0]])
        self.assertTrue(stage.check_row.isHidden())

    def test_leaving_joins_the_checking_thread(self):
        def slow(job, config, should_stop=None):
            while not (should_stop and should_stop()):
                time.sleep(0.005)
            return None

        stage = self.enter()
        with patch.object(ds, "check_video", slow):
            self.assertTrue(stage.start_quick_check())
            worker = stage._check_worker
            self.assertTrue(self.written(lambda: worker.isRunning()))
            self.window.go_to_stage("trials")
            self.assertFalse(worker.isRunning())
        self.assertFalse(stage._checks.has_running_thread())

    def test_a_second_check_cannot_start_while_one_runs(self):
        def slow(job, config, should_stop=None):
            while not (should_stop and should_stop()):
                time.sleep(0.005)
            return None

        stage = self.enter()
        with patch.object(ds, "check_video", slow):
            self.assertTrue(stage.start_quick_check())
            self.assertFalse(stage.start_quick_check())
            self.assertFalse(stage.check_button.isEnabled())
            stage.cancel_quick_check()
            self.assertTrue(self.written(lambda: not stage.checking, 10.0))

    def test_the_check_uses_each_videos_own_zones(self):
        stage = self.enter()
        jobs = stage._jobs()
        self.assertEqual([j.key for j in jobs], self.keys)
        self.assertEqual(jobs[0].roi_manager.rois.keys(),
                         self.template.roi_manager.rois.keys())


class StageLifecycleTests(DetectionStageFixture):
    def test_leaving_shuts_the_tuner_down_and_joins_its_thread(self):
        stage = self.enter()
        tuner = stage.tuner
        self.assertTrue(self.quality_ready())
        self.window.go_to_stage("trials")
        self.assertIsNone(stage.tuner)
        self.assertTrue(tuner._lifecycle.is_shut_down)
        self.assertFalse(tuner._lifecycle.has_running_thread())
        self.assertFalse(tuner.cap.isOpened())
        deliver_deletes()
        self.assertTrue(sip.isdeleted(tuner))

    def test_leaving_in_the_middle_of_a_scan_joins_it(self):
        stage = self.enter()
        tuner = stage.tuner
        tuner.run_quick_scan()
        worker = tuner.scan_worker
        self.window.go_to_stage("trials")
        self.assertFalse(worker.isRunning())

    def test_coming_back_rebuilds_lazily(self):
        stage = self.enter()
        first = stage.tuner
        self.window.go_to_stage("trials")
        self.assertIsNone(stage.tuner)
        stage = self.enter()
        self.assertIsNotNone(stage.tuner)
        self.assertIsNot(stage.tuner, first)
        self.assertFalse(stage.tuner._lifecycle.is_shut_down)

    def test_the_preview_video_is_remembered(self):
        stage = self.enter()
        stage.show_video(self.keys[1])
        self.window.go_to_stage("trials")
        stage = self.enter()
        self.assertEqual(stage.previewed_key, self.keys[1])

    def test_closing_the_project_flushes_and_stops_everything(self):
        stage = self.enter()
        tuner = stage.tuner
        stage.tuner.sliders["Min component area"].setValue(88)
        project = self.project
        self.prompts.say("ask_save_changes", "discard")
        self.window._skip_close_prompt = True
        self.assertTrue(self.window.close_project())
        self.assertEqual(project.template_settings.detection_config
                         .min_component_area, 88)
        self.assertTrue(tuner._lifecycle.is_shut_down)
        self.assertIsNone(stage.tuner)

    def test_a_refused_close_brings_the_preview_back(self):
        stage = self.enter()
        self.window.register_close_guard(lambda: False)
        self.assertFalse(self.window.close_project())
        self.app.processEvents()
        self.assertIsNotNone(stage.tuner)
        self.assertFalse(stage.tuner._lifecycle.is_shut_down)

    def test_a_pending_write_never_lands_in_another_project(self):
        stage = self.enter()
        stage.tuner.sliders["Min component area"].setValue(99)
        other = Project(name="other", paradigm=Y)
        other.set_template(valid_settings(Y))
        untouched = canonical_json(other.template_settings.to_dict())
        self.window._skip_close_prompt = True
        self.window._install_project(other)
        self.app.processEvents()
        self.assertEqual(canonical_json(other.template_settings.to_dict()),
                         untouched)

    def test_entering_and_leaving_in_a_loop_leaks_nothing(self):
        stage = self.enter()
        self.window.go_to_stage("trials")
        baseline_threads = threading.active_count()
        tuners = []
        for round_number in range(12):
            stage = self.enter()
            tuner = stage.tuner
            tuners.append(tuner)
            if round_number % 3 == 0:
                tuner.sliders["Search window"].setValue(60 + round_number)
                tuner.run_quick_scan()
            if round_number % 4 == 0:
                stage.show_video(self.keys[1])
                tuners.append(stage.tuner)
            if round_number % 5 == 0:
                stage.start_quick_check()
            self.window.go_to_stage("trials")
            self.assertIsNone(stage.tuner)
            self.assertFalse(stage._checks.has_running_thread())
        for tuner in tuners:
            self.assertTrue(tuner._lifecycle.is_shut_down)
            self.assertFalse(tuner._lifecycle.has_running_thread())
            self.assertFalse(tuner.cap.isOpened())
        deliver_deletes()
        self.app.processEvents()
        deliver_deletes()
        for tuner in tuners:
            self.assertTrue(sip.isdeleted(tuner))
        self.assertTrue(self.written(
            lambda: threading.active_count() <= baseline_threads, 10.0))
        self.assertEqual(stage.hero_layout.count(), 1)       # only the message
        self.assertEqual(stage.advanced.body.count(), 1)     # only the hint


class StageFooterTests(DetectionStageFixture):
    def test_can_advance_with_a_valid_config(self):
        stage = self.enter()
        self.assertEqual(stage.can_advance(), (True, ""))

    def test_an_invalid_config_blocks_next_with_the_reason(self):
        stage = self.enter()
        self.template.detection_config.threshold_percentile = 150.0
        allowed, reason = stage.can_advance()
        self.assertFalse(allowed)
        self.assertIn("threshold_percentile", reason)
        self.window.refresh_footer()
        self.assertFalse(self.window.footer.next_button.isEnabled())

    def test_the_footer_reports_the_quick_check(self):
        stage = self.enter()
        stage.start_quick_check()
        self.assertTrue(self.written(lambda: not stage.checking, 60.0))
        text = stage.footer_text()[0]
        self.assertRegex(text, r"^\d+ good · \d+ to check")
        self.assertEqual(self.window.footer.status_text(), text) \
            if hasattr(self.window.footer, "status_text") else None

    def test_progress_is_in_the_footer_while_checking(self):
        def slow(job, config, should_stop=None):
            while not (should_stop and should_stop()):
                time.sleep(0.005)
            return None

        stage = self.enter()
        with patch.object(ds, "check_video", slow):
            stage.start_quick_check()
            self.assertRegex(stage.footer_text()[0], r"^Checking videos")
            stage.cancel_quick_check()
            self.assertTrue(self.written(lambda: not stage.checking, 10.0))


class MotionStageTests(DetectionStageFixture):
    PARADIGM = T

    def setUp(self):
        super().setUp()
        self.window.resize(1366, 768)

    def test_it_is_called_motion_tuning_and_hosts_the_tst_tuner(self):
        stage = self.enter()
        self.assertEqual(stage.header.title(), "Motion tuning")
        self.assertTrue(stage.is_motion)
        self.assertIsNotNone(stage.tuner)
        self.assertTrue(stage.tuner.embedded)
        self.assertFalse(stage.simple.isVisibleTo(stage))
        self.assertFalse(stage.motion_card.isHidden())
        self.assertFalse(stage.check_button.isVisibleTo(stage))
        self.assertGreaterEqual(stage.tuner.width(), 0.6 * self.window.width())

    def test_the_minimum_immobility_slider_writes_live(self):
        stage = self.enter()
        slider = stage.tuner.time_slider
        slider.setValue(23)
        self.assertTrue(self.written(
            lambda: self.project.template_settings.algorithm_parameters
            .get("min_time_seconds") == 2.3))

    def test_it_matches_the_classic_motion_tuner(self):
        stage = self.enter()
        stage.tuner.time_slider.setValue(17)
        stage.tuner.advanced_controls["static_rigid_threshold"].setValue(0.01)
        stage.commit_pending()

        state = SetupState.from_batch_settings(
            valid_settings(T, detection_config=copy.copy(GOOD)), maze_name=None)
        maze = state.build_maze()
        from gui.tuners.tst_tuner import TstTunerWidget
        tuner = TstTunerWidget(
            str(self.clips[0]), state.roi_manager, maze,
            initial_motion_config=maze.motion_config)
        self.addCleanup(tuner.shutdown)
        tuner.time_slider.setValue(17)
        tuner.advanced_controls["static_rigid_threshold"].setValue(0.01)
        tuner.apply_to_maze(maze)
        state.set_maze(maze)
        classic = state.to_batch_settings(finalize=True)
        self.assertEqual(self.project.template_settings.to_dict(),
                         classic.to_dict())

    def test_an_untouched_slider_keeps_the_exact_bout_length(self):
        settings = valid_settings(T)
        params = copy.deepcopy(settings.algorithm_parameters)
        params["min_time_seconds"] = 1.37
        settings.algorithm_parameters = params
        self.project.set_template(settings)
        stage = self.enter()
        stage.tuner.advanced_controls["segmentation_contrast"].setValue(30.0)
        stage.commit_pending()
        self.assertEqual(self.project.template_settings.algorithm_parameters
                         ["min_time_seconds"], 1.37)

    def test_an_invalid_classifier_is_refused_with_a_reason(self):
        stage = self.enter()
        controls = stage.tuner.advanced_controls
        controls["min_mouse_area_ratio"].setValue(0.4)
        controls["max_mouse_area_ratio"].setValue(0.3)
        before = canonical_json(self.template.to_dict())
        stage.commit_pending()
        self.assertEqual(canonical_json(self.template.to_dict()), before)
        allowed, reason = stage.can_advance()
        self.assertFalse(allowed)
        self.assertTrue(reason)

    def test_leaving_joins_the_scan_thread(self):
        stage = self.enter()
        tuner = stage.tuner
        self.assertTrue(self.written(lambda: tuner.scanner is not None))
        scanner = tuner.scanner
        self.window.go_to_stage("trials")
        self.assertTrue(tuner._lifecycle.is_shut_down)
        self.assertFalse(scanner.isRunning())
        self.assertIsNone(stage.tuner)

    def test_switching_video_rebuilds_and_rescans(self):
        stage = self.enter()
        first = stage.tuner
        stage.show_video(self.keys[1])
        self.assertIsNot(stage.tuner, first)
        self.assertTrue(first._lifecycle.is_shut_down)
        self.assertTrue(stage.tuner.video_path.endswith("clip2.avi"))
        self.assertTrue(self.written(lambda: stage.tuner._scan_complete, 60.0))
        self.assertEqual(stage.motion_chip.text(), "Preview ready")

    def test_without_mouse_zones_it_asks_for_them(self):
        template = valid_settings(T)
        for name in list(template.roi_manager.rois):
            template.roi_manager.rois[name] = []
        self.project.set_template(template)
        stage = self.enter()
        self.assertEqual(stage.pages.currentIndex(), PAGE_EMPTY)
        self.assertIn("mouse", stage.empty.body().lower())

    def test_loop_enter_and_leave(self):
        tuners = []
        for _ in range(6):
            stage = self.enter()
            tuners.append(stage.tuner)
            self.window.go_to_stage("trials")
        for tuner in tuners:
            self.assertTrue(tuner._lifecycle.is_shut_down)
            self.assertFalse(tuner._lifecycle.has_running_thread())
        deliver_deletes()
        self.assertTrue(all(sip.isdeleted(t) for t in tuners))


if __name__ == "__main__":
    unittest.main()
