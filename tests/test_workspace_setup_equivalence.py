"""The Setup tab must produce exactly the settings the classic wizard does.

Each test scripts the tab through the widgets' test hooks (ROI ``add_point`` /
``close_shape``, panel setters, tuner sliders), then builds the settings for
the same choices through the classic path (the classic dialogs and the steps
``MainWindow._run_setup_wizard`` takes) and compares the two ``to_dict()``s.
"""
import unittest

from core.models import AppConfig, BatchSettings
from core.project import SOURCE_CUSTOM
from core.video import ReferenceFrameGenerator
from mazes import create_maze
from mazes.base_maze import (STEP_CALIBRATION, STEP_DETECTION_TUNING,
                             STEP_PARADIGM_PARAMS, STEP_REVIEW, STEP_ROIS,
                             STEP_TIME_WINDOW, STEP_TST_TUNING)
from tests.setup_tab_support import (E, F, T, Y, SetupTabFixture, draw_zones,
                                     draw_polygon)
from tests.workspace_support import HAS_RUNTIME

if HAS_RUNTIME:
    from gui.arm_entry_dialog import ArmEntryParametersDialog
    from gui.freestyle_config_dialog import FreestyleConfigDialog
    from gui.roi_selector import RoiSelectorDialog
    from gui.settings_dialog import SettingsDialog
    from gui.tst_tuner import TstTunerDialog
    from gui.tuner_dialog import TunerDialog
    from gui.workspace.setup_tab import SCOPE_VIDEOS

LINE = ((10, 100), (110, 100))
ARENA = [(2, 2), (158, 2), (158, 100), (2, 100)]
TUNING = {"Min component area": 55, "Threshold percentile": 980}


class EquivalenceFixture(SetupTabFixture):
    """Helpers that drive the tab and build the classic reference result."""

    # -- the tab ----------------------------------------------------------
    def tab_params(self, edit) -> None:
        self.assertTrue(self.tab.go_to_step(STEP_PARADIGM_PARAMS))
        edit(self.tab.host.panel.active_panel)
        self.assertTrue(self.tab.apply_step(), self.tab.displayed_issues())

    def tab_rois(self, draw) -> None:
        self.assertTrue(self.tab.go_to_step(STEP_ROIS))
        draw(self.tab.host.editor)
        self.assertTrue(self.tab.apply_step(), self.tab.displayed_issues())

    def tab_calibration(self, cm: float) -> None:
        self.assertTrue(self.tab.go_to_step(STEP_CALIBRATION))
        self.tab.host.panel.scale_spin.setValue(cm)
        self.assertTrue(self.tab.apply_step(), self.tab.displayed_issues())

    def tab_detection(self, sliders) -> None:
        self.assertTrue(self.tab.go_to_step(STEP_DETECTION_TUNING))
        for name, value in sliders.items():
            self.tab.host.tuner.sliders[name].setValue(value)
        self.assertTrue(self.tab.apply_step(), self.tab.displayed_issues())

    def tab_tst(self, bout_slider) -> None:
        self.assertTrue(self.tab.go_to_step(STEP_TST_TUNING))
        self.tab.host.tuner.time_slider.setValue(bout_slider)
        self.assertTrue(self.tab.apply_step(), self.tab.displayed_issues())

    def tab_time(self, edit) -> None:
        self.assertTrue(self.tab.go_to_step(STEP_TIME_WINDOW))
        edit(self.tab.host.panel)
        self.assertTrue(self.tab.apply_step(), self.tab.displayed_issues())

    def tab_review(self, timelapse: bool) -> None:
        self.assertTrue(self.tab.go_to_step(STEP_REVIEW))
        self.tab.host.panel.timelapse_check.setChecked(timelapse)
        self.assertTrue(self.tab.apply_step(), self.tab.displayed_issues())

    # -- the classic path ---------------------------------------------------
    @property
    def video(self) -> str:
        return str(self.clips[0])

    def classic_rois(self, maze, draw):
        dialog = RoiSelectorDialog(self.video, maze)
        self.addCleanup(dialog.deleteLater)
        draw(dialog.editor)
        return dialog.get_roi_manager()

    def classic_detection(self, roi_manager, sliders):
        frame = ReferenceFrameGenerator.generate(
            self.clips[0], AppConfig().video.ref_frame_samples, use_cache=True
        )
        dialog = TunerDialog(self.video, frame, roi_manager)
        self.addCleanup(dialog.deleteLater)
        for name, value in sliders.items():
            dialog.sliders[name].setValue(value)
        config = dialog.get_detection_config()
        dialog.widget.shutdown()
        return config

    def classic_review(self, base: BatchSettings, maze, edit,
                       video=None, verification=False) -> BatchSettings:
        """The settings dialog (times, calibration, validation video)."""
        dialog = SettingsDialog(
            video or self.video, base, is_verification=verification,
            maze_name=maze.name,
        )
        self.addCleanup(dialog.deleteLater)
        edit(dialog)
        settings = dialog.get_settings()
        settings.paradigm_id = maze.name
        settings.algorithm_parameters = maze.get_algorithm_parameters()
        return settings

    def assert_same(self, classic: BatchSettings) -> None:
        template = self.project.template_settings
        self.assertIsNotNone(template)
        self.assertEqual(template.to_dict(), classic.to_dict())
        for key in self.keys:
            entry = self.project.entry(key)
            self.assertEqual(entry.settings.to_dict(), classic.to_dict())


def arm_zones(maze):
    return [name for name, _label in maze.get_roi_definitions()]


class ArmMazeEquivalenceTests(EquivalenceFixture):
    PARADIGM = None

    def check(self, paradigm):
        self.choose_paradigm(paradigm)
        zones = arm_zones(create_maze(paradigm))

        # -- the tab
        self.tab_params(lambda panel: (
            panel.dwell_spin.setValue(0.3), panel.initial_check.setChecked(False)
        ))

        def draw(editor):
            draw_zones(editor, zones)
            self.assertTrue(editor.set_line(*LINE))
        self.tab_rois(draw)
        self.tab_calibration(25.0)
        self.tab_detection(TUNING)

        def window(panel):
            panel.start_spinbox.setValue(0.5)
            panel.end_spinbox.setValue(1.5)
        self.tab_time(window)
        self.tab_review(True)

        # -- the classic wizard
        maze = create_maze(paradigm)
        roi_manager = self.classic_rois(maze, draw)
        arm = ArmEntryParametersDialog(maze)
        self.addCleanup(arm.deleteLater)
        arm.dwell_spin.setValue(0.3)
        arm.initial_check.setChecked(False)
        maze.configure_algorithm_parameters(arm.get_parameters())
        detection = self.classic_detection(roi_manager, TUNING)
        base = BatchSettings(roi_manager, detection, 0.0)

        def review(dialog):
            dialog.start_spinbox.setValue(0.5)
            dialog.end_spinbox.setValue(1.5)
            dialog.scale_spin.setValue(25.0)
            dialog.timelapse_check.setChecked(True)
        classic = self.classic_review(base, maze, review)

        self.assert_same(classic)
        self.assertEqual(self.project.template_settings.paradigm_id, paradigm)
        self.assertTrue(self.project.template_settings.create_timelapse)
        self.assertGreater(self.project.template_settings.scale_factor, 0)

    def test_y_maze(self):
        self.check(Y)

    def test_elevated_plus_maze(self):
        self.check(E)


class FreestyleEquivalenceTests(EquivalenceFixture):
    PARADIGM = F

    def test_zones_with_an_arena(self):
        def params(panel):
            panel.num_zones_spin.setValue(2)
            panel.quick_fill("letters")
            panel.arena_checkbox.setChecked(True)
            panel.immobility_threshold_spin.setValue(3.5)
        self.tab_params(params)

        def draw(editor):
            draw_polygon(editor, "arena", ARENA)
            draw_zones(editor, ["zone_a", "zone_b"], first_index=1)
            self.assertTrue(editor.set_line(*LINE))
        self.tab_rois(draw)
        self.tab_calibration(30.0)
        self.tab_detection(TUNING)
        self.tab_time(lambda panel: panel.start_spinbox.setValue(0.25))
        self.tab_review(False)

        maze = create_maze(F)
        dialog = FreestyleConfigDialog(maze=maze)
        self.addCleanup(dialog.deleteLater)
        params(dialog)
        dialog.validate_and_accept()
        self.assertEqual(dialog.result(), dialog.DialogCode.Accepted)
        maze.configure_zones(dialog.get_zone_definitions())
        roi_manager = self.classic_rois(maze, draw)
        detection = self.classic_detection(roi_manager, TUNING)
        base = BatchSettings(roi_manager, detection, 0.0)

        def review(settings_dialog):
            settings_dialog.start_spinbox.setValue(0.25)
            settings_dialog.scale_spin.setValue(30.0)
        classic = self.classic_review(base, maze, review)

        self.assert_same(classic)
        parameters = self.project.template_settings.algorithm_parameters
        self.assertTrue(parameters["arena_mode"])
        self.assertEqual(len(parameters["zones"]), 2)
        self.assertIsNotNone(self.project.template_settings.roi_manager.arena)

    def test_zone_free_tracking(self):
        self.tab_params(lambda panel: panel.zone_free_radio.setChecked(True))

        def draw(editor):
            self.assertTrue(editor.set_line(*LINE))
        self.tab_rois(draw)
        self.tab_calibration(12.5)
        self.tab_detection({})
        self.tab_time(lambda panel: None)
        self.tab_review(False)

        maze = create_maze(F)
        dialog = FreestyleConfigDialog(maze=maze)
        self.addCleanup(dialog.deleteLater)
        dialog.zone_free_radio.setChecked(True)
        dialog.validate_and_accept()
        maze.configure_zones(dialog.get_zone_definitions())
        roi_manager = self.classic_rois(maze, draw)
        detection = self.classic_detection(roi_manager, {})
        base = BatchSettings(roi_manager, detection, 0.0)
        classic = self.classic_review(
            base, maze, lambda settings_dialog: settings_dialog.scale_spin
            .setValue(12.5))
        self.assert_same(classic)


class TstEquivalenceTests(EquivalenceFixture):
    PARADIGM = T

    def test_two_mice(self):
        self.tab_params(lambda panel: panel.mice_spin.setValue(2))
        self.tab_rois(lambda editor: draw_zones(editor, ["mouse_1", "mouse_2"]))
        self.tab_tst(8)

        def windows(panel):
            self.assertEqual(sorted(panel.mouse_time_widgets),
                             ["mouse_1", "mouse_2"])
            start, end = panel.mouse_time_widgets["mouse_1"]
            start.setValue(0.2)
            end.setValue(1.4)
            start, end = panel.mouse_time_widgets["mouse_2"]
            start.setValue(0.5)
            end.setValue(1.8)
        self.tab_time(windows)
        self.tab_review(True)

        maze = create_maze(T)
        maze.configure_mice(2)
        roi_manager = self.classic_rois(
            maze, lambda editor: draw_zones(editor, ["mouse_1", "mouse_2"]))
        tuner = TstTunerDialog(self.video, roi_manager, maze)
        self.addCleanup(tuner.deleteLater)
        tuner.widget.time_slider.setValue(8)
        tuner.widget.apply_to_maze()
        tuner.cleanup()
        base = BatchSettings(roi_manager, AppConfig().detection, 0.0)

        def review(dialog):
            start, end = dialog.mouse_time_widgets["mouse_1"]
            start.setValue(0.2)
            end.setValue(1.4)
            start, end = dialog.mouse_time_widgets["mouse_2"]
            start.setValue(0.5)
            end.setValue(1.8)
            dialog.timelapse_check.setChecked(True)
        classic = self.classic_review(base, maze, review)

        self.assert_same(classic)
        settings = self.project.template_settings
        self.assertEqual(settings.algorithm_parameters["num_mice"], 2)
        self.assertEqual(settings.scale_factor, 0.0)
        self.assertEqual(
            set(settings.per_roi_times), {"mouse_1", "mouse_2"})


class CustomVideoEquivalenceTests(EquivalenceFixture):
    def test_a_verified_video_matches_the_classic_verification(self):
        zones = arm_zones(create_maze(Y))

        def draw(editor):
            draw_zones(editor, zones)
            self.assertTrue(editor.set_line(*LINE))
        self.tab_params(lambda panel: None)
        self.tab_rois(draw)
        self.tab_calibration(25.0)
        self.tab_detection(TUNING)
        self.tab_time(lambda panel: None)
        self.tab_review(False)
        template = self.project.template_settings.copy()

        # Custom times and a calibration for the second video only.
        self.select([self.keys[1]])
        self.assertTrue(self.tab.set_scope(SCOPE_VIDEOS))
        self.tab_time(lambda panel: panel.start_spinbox.setValue(0.75))
        self.tab_calibration(40.0)
        self.tab_review(True)

        maze = create_maze(Y)
        maze.configure_algorithm_parameters(template.algorithm_parameters)

        def verify(dialog):
            dialog.start_spinbox.setValue(0.75)
            dialog.scale_spin.setValue(40.0)
            dialog.timelapse_check.setChecked(True)
        classic = self.classic_review(
            template.copy(), maze, verify, video=str(self.clips[1]),
            verification=True,
        )
        entry = self.project.entry(self.keys[1])
        self.assertEqual(entry.settings_source, SOURCE_CUSTOM)
        self.assertEqual(entry.settings.to_dict(), classic.to_dict())
        # The template and the other video are untouched.
        self.assertEqual(
            self.project.template_settings.to_dict(), template.to_dict())
        self.assertEqual(
            self.project.entry(self.keys[0]).settings.to_dict(),
            template.to_dict())


if __name__ == "__main__":
    unittest.main()
