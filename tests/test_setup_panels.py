"""The embeddable setup panels: round trips, issues and the changed signal."""
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

try:
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from gui.setup_panels import (ArmEntryParamsPanel, CalibrationPanel,
                                  FreestyleParamsPanel, ParadigmParamsPanel,
                                  ReviewPanel, TimeWindowPanel,
                                  TstParamsPanel)
    HAS_QT = True
except ModuleNotFoundError:
    HAS_QT = False

from core.calibration import (REFERENCE_APPROXIMATE, REFERENCE_DRAWN,
                              REFERENCE_LEGACY, REFERENCE_NONE,
                              describe_reference)
from core.models import AppConfig, BatchSettings
from core.setup_state import SetupState
from mazes import AVAILABLE_MAZES, create_maze
from roi_manager import ROIManager

SQUARE = np.array([[10, 10], [60, 10], [60, 60], [10, 60]])
Y, E, F, T = (
    "Y-Maze", "Elevated Plus Maze", "Freestyle / Open Field",
    "Tail Suspension Test",
)
FPS = 20.0
FRAMES = 400  # 20 s


def write_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (64, 64)
    )
    for _ in range(FRAMES):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()


def configured(name):
    maze = create_maze(name)
    if name == F:
        maze.configure_zones([("zone_a", "Zone A"), ("zone_b", "Zone B")])
    if name == T:
        maze.configure_mice(2)
    return maze


def rois_for(maze, reference=True, arena=False):
    manager = ROIManager(source_dimensions=(100, 100))
    for index, (name, _) in enumerate(maze.get_roi_definitions()):
        manager.add_roi(name, SQUARE + 100 * index)
    if reference:
        manager.set_reference_line((0, 0), (30, 40), "User-defined")
    if arena:
        manager.set_arena(SQUARE)
    return manager


def state_for(name, maze=None, video=None, **overrides) -> SetupState:
    maze = maze or configured(name)
    state = SetupState.new(name, video)
    state.set_maze(maze)
    state.roi_manager = rois_for(maze, reference=name != T)
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def dump(state: SetupState) -> str:
    return json.dumps(state.to_batch_settings().to_dict(), sort_keys=True)


class CountingSlot:
    def __init__(self, signal):
        self.count = 0
        signal.connect(self._hit)

    def _hit(self, *_args):
        self.count += 1


class QtCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make(self, panel):
        self.addCleanup(panel.deleteLater)
        return panel


@unittest.skipUnless(HAS_QT, "GUI runtime is not installed")
class ParadigmParamsPanelTests(QtCase):
    def test_dispatches_to_the_panel_of_each_paradigm(self):
        panel = self.make(ParadigmParamsPanel())
        expected = {Y: ArmEntryParamsPanel, E: ArmEntryParamsPanel,
                    F: FreestyleParamsPanel, T: TstParamsPanel}
        for name, panel_class in expected.items():
            with self.subTest(maze=name):
                panel.load(SetupState.new(name))
                self.assertIsInstance(panel.active_panel, panel_class)
                self.assertIs(
                    panel._stack.currentWidget(), panel.active_panel
                )

    def test_round_trip_leaves_the_parameters_identical(self):
        panel = self.make(ParadigmParamsPanel())
        for name in AVAILABLE_MAZES:
            with self.subTest(maze=name):
                state = state_for(name)
                before = dump(state)
                params = dict(state.algorithm_parameters)
                panel.load(state)
                self.assertTrue(panel.apply(state))
                self.assertEqual(state.algorithm_parameters, params)
                self.assertEqual(dump(state), before)
                self.assertEqual(state.step_status("paradigm_params"), "done")

    def test_round_trip_keeps_values_the_boxes_would_round(self):
        # 0.125 s dwell and a 33 % center would be shown rounded; untouched
        # they must come back exactly.
        maze = configured(F)
        maze.min_dwell_seconds = 0.125
        maze.speed_window_seconds = 0.333
        maze.configure_arena(True, "area_fraction", 3.33, 0.335)
        state = state_for(F, maze)
        before = state.algorithm_parameters
        panel = self.make(ParadigmParamsPanel())
        panel.load(state)
        self.assertTrue(panel.apply(state))
        self.assertEqual(state.algorithm_parameters, before)

        arm = YMazeWithDwell(0.125)
        state = state_for(Y, arm)
        before = state.algorithm_parameters
        panel.load(state)
        panel.apply(state)
        self.assertEqual(state.algorithm_parameters, before)

        tst = configured(T)
        tst.set_parameters(1.234, 0.123)
        state = state_for(T, tst)
        before = state.algorithm_parameters
        panel.load(state)
        panel.apply(state)
        self.assertEqual(state.algorithm_parameters, before)

    def test_edits_reach_the_state(self):
        panel = self.make(ParadigmParamsPanel())
        state = state_for(Y)
        panel.load(state)
        panel.active_panel.dwell_spin.setValue(0.35)
        panel.active_panel.initial_check.setChecked(False)
        panel.apply(state)
        self.assertEqual(state.algorithm_parameters["min_dwell_seconds"], 0.35)
        self.assertFalse(state.algorithm_parameters["count_initial_arm"])

        state = state_for(T)
        panel.load(state)
        tst = panel.active_panel
        tst.mice_spin.setValue(3)
        tst.min_bout_spin.setValue(2.5)
        tst.max_gap_spin.setValue(0.4)
        panel.apply(state)
        self.assertEqual(state.algorithm_parameters["num_mice"], 3)
        self.assertEqual(state.algorithm_parameters["min_time_seconds"], 2.5)
        self.assertEqual(
            state.algorithm_parameters["max_motion_gap_seconds"], 0.4
        )
        self.assertEqual(
            [n for n, _ in state.build_maze().get_roi_definitions()],
            ["mouse_1", "mouse_2", "mouse_3"],
        )

    def test_changed_is_emitted_by_edits_and_not_by_load(self):
        panel = self.make(ParadigmParamsPanel())
        slot = CountingSlot(panel.changed)
        for name, edit in (
            (Y, lambda p: p.dwell_spin.setValue(0.5)),
            (E, lambda p: p.initial_check.toggle()),
            (T, lambda p: p.mice_spin.setValue(4)),
            (T, lambda p: p.min_bout_spin.setValue(3.0)),
            (T, lambda p: p.max_gap_spin.setValue(0.5)),
            (F, lambda p: p.immobility_threshold_spin.setValue(3.0)),
            (F, lambda p: p.arena_checkbox.setChecked(True)),
            (F, lambda p: p.num_zones_spin.setValue(4)),
            (F, lambda p: p.zone_name_inputs[0].setText("Home")),
            (F, lambda p: p.zone_free_radio.setChecked(True)),
        ):
            panel.load(SetupState.new(name))
            before = slot.count
            self.assertEqual(before, slot.count)
            edit(panel.active_panel)
            self.assertGreater(slot.count, before, name)
        # Loading never counts as a user change.
        slot.count = 0
        panel.load(state_for(F))
        panel.load(state_for(T))
        panel.load(state_for(Y))
        self.assertEqual(slot.count, 0)

    def test_tst_loads_into_the_mouse_count_range(self):
        panel = self.make(TstParamsPanel())
        tst = configured(T)
        tst.configure_mice(9)
        panel.load_from_maze(tst)
        self.assertEqual(panel.mice_spin.value(), 6)


def YMazeWithDwell(dwell):
    maze = create_maze(Y)
    maze.min_dwell_seconds = dwell
    return maze


@unittest.skipUnless(HAS_QT, "GUI runtime is not installed")
class FreestylePanelTests(QtCase):
    def test_zones_round_trip_with_their_internal_names(self):
        maze = create_maze(F)
        maze.configure_zones([("zoneA", "Zonea"), ("my_zone", "My Zone")])
        state = state_for(F, maze)
        panel = self.make(FreestyleParamsPanel())
        panel.load(state)
        self.assertEqual(panel.num_zones_spin.value(), 2)
        self.assertEqual(
            [f.text() for f in panel.zone_name_inputs], ["Zonea", "My Zone"]
        )
        self.assertTrue(panel.apply(state))
        self.assertEqual(
            state.algorithm_parameters["zones"],
            [["zoneA", "Zonea"], ["my_zone", "My Zone"]],
        )
        # Renaming a zone derives the internal name from the new text.
        panel.zone_name_inputs[0].setText("Home Base")
        panel.apply(state)
        self.assertEqual(
            state.algorithm_parameters["zones"][0], ["home_base", "Home Base"]
        )

    def test_zone_free_and_unconfigured_states(self):
        panel = self.make(FreestyleParamsPanel())
        zone_free = create_maze(F)
        zone_free.configure_zones([])
        panel.load_from_maze(zone_free)
        self.assertTrue(panel.zone_free_radio.isChecked())
        self.assertTrue(panel.zone_free_mode)
        self.assertEqual(panel.zone_definitions(), [])
        self.assertEqual(panel.issues(), [])
        maze = create_maze(F)
        panel.apply_to_maze(maze)
        self.assertTrue(maze.zones_configured)
        self.assertEqual(maze.user_defined_zones, [])

        # A maze with no zones chosen yet shows the defaults again.
        panel.load_from_maze(create_maze(F))
        self.assertTrue(panel.zone_based_radio.isChecked())
        self.assertEqual(panel.num_zones_spin.value(), 3)
        self.assertEqual(
            [f.text() for f in panel.zone_name_inputs],
            ["Zone A", "Zone B", "Zone C"],
        )

    def test_issues_list_every_bad_zone_name(self):
        panel = self.make(FreestyleParamsPanel())
        self.assertEqual(panel.issues(), [])
        panel.zone_name_inputs[0].setText("")
        panel.zone_name_inputs[1].setText("Mouse 1")
        panel.zone_name_inputs[2].setText("Zone A")
        panel.num_zones_spin.setValue(4)
        panel.zone_name_inputs[3].setText("zone a")
        titles = [title for title, _msg, _field in panel.zone_problems()]
        self.assertEqual(
            titles,
            ["Empty Zone Name", "Reserved Zone Name", "Duplicate Zone Name"],
        )
        self.assertEqual(len(panel.issues()), 3)
        self.assertIs(
            panel.first_zone_problem()[2], panel.zone_name_inputs[0]
        )
        state = state_for(F)
        before = dict(state.algorithm_parameters)
        self.assertFalse(panel.apply(state))
        self.assertEqual(state.algorithm_parameters, before)

    def test_apply_does_not_touch_other_state(self):
        panel = self.make(FreestyleParamsPanel())
        state = state_for(F)
        panel.load(state)
        panel.immobility_threshold_spin.setValue(4.0)
        panel.apply(state)
        self.assertEqual(
            state.algorithm_parameters["immobility_threshold_cm_per_s"], 4.0
        )
        self.assertIn("rois", state.steps)
        self.assertIsNotNone(state.roi_manager)


@unittest.skipUnless(HAS_QT, "GUI runtime is not installed")
class CalibrationPanelTests(QtCase):
    def test_reference_status_of_every_kind(self):
        maze = create_maze(Y)
        manager = rois_for(maze, reference=False)
        status = describe_reference(manager)
        self.assertEqual(status.kind, REFERENCE_NONE)
        manager.set_reference_line((0, 0), (30, 40))
        status = describe_reference(manager)
        self.assertEqual(status.kind, REFERENCE_DRAWN)
        self.assertEqual(status.text, "User-drawn reference line (50.00 px)")
        manager.set_reference_length(30.0, "Width of first Arm A ROI")
        self.assertEqual(describe_reference(manager).kind, REFERENCE_LEGACY)
        self.assertTrue(describe_reference(manager).warning)
        manager.set_reference_length(
            30.0, "User-defined (approximate: scaled by alignment)"
        )
        self.assertEqual(
            describe_reference(manager).kind, REFERENCE_APPROXIMATE
        )
        self.assertEqual(describe_reference(None).kind, REFERENCE_NONE)

    def test_panel_shows_each_status(self):
        panel = self.make(CalibrationPanel())
        maze = create_maze(Y)

        state = state_for(Y)
        state.roi_manager = rois_for(maze, reference=False)
        panel.load(state)
        self.assertEqual(panel.status.kind, REFERENCE_NONE)
        self.assertFalse(panel.scale_spin.isEnabled())
        self.assertIn("No calibration line", panel.note_label.text())
        self.assertEqual(panel.scale_label.text(), "Pixels (no calibration)")

        state.roi_manager = rois_for(maze)
        state.scale_cm = 10.0
        panel.load(state)
        self.assertEqual(panel.status.kind, REFERENCE_DRAWN)
        self.assertTrue(panel.scale_spin.isEnabled())
        self.assertEqual(panel.scale_spin.value(), 10.0)
        self.assertIn("User-drawn reference line (50.00 px)",
                      panel.reference_label.text())
        self.assertTrue(panel.note_label.isHidden())
        self.assertAlmostEqual(panel.scale_factor(), 0.2)
        self.assertIn("0.20000 cm/px", panel.scale_label.text())
        self.assertIn("5.00 px/cm", panel.scale_label.text())

        manager = rois_for(maze, reference=False)
        manager.set_reference_length(30.0, "Width of first Arm A ROI")
        state.roi_manager = manager
        panel.load(state)
        self.assertEqual(panel.status.kind, REFERENCE_LEGACY)
        self.assertFalse(panel.note_label.isHidden())
        self.assertIn("bounding box", panel.note_label.text())
        self.assertIn("Legacy", panel.reference_label.text())

        manager.set_reference_length(
            30.0, "User-defined (approximate: scaled by alignment)"
        )
        panel.load(state)
        self.assertEqual(panel.status.kind, REFERENCE_APPROXIMATE)
        self.assertIn("approximate", panel.note_label.text())

    def test_round_trip_and_edit(self):
        panel = self.make(CalibrationPanel())
        state = state_for(Y)
        state.scale_cm = 12.5
        before = dump(state)
        panel.load(state)
        self.assertTrue(panel.apply(state))
        self.assertEqual(dump(state), before)
        panel.scale_spin.setValue(20.0)
        panel.apply(state)
        self.assertEqual(state.scale_cm, 20.0)
        self.assertAlmostEqual(state.scale_factor, 0.4)
        self.assertEqual(state.step_status("calibration"), "done")

    def test_untouched_calibration_round_trips_exactly(self):
        settings = BatchSettings(
            rois_for(create_maze(Y)), AppConfig().detection,
            0.123456789012345, paradigm_id=Y,
        )
        state = SetupState.from_batch_settings(settings)
        panel = self.make(CalibrationPanel())
        panel.load(state)
        panel.apply(state)
        self.assertEqual(state.to_batch_settings().scale_factor,
                         settings.scale_factor)
        # The classic dialogs read back what the box shows instead.
        rounded = self.make(CalibrationPanel(preserve_exact=False))
        rounded.load(state)
        self.assertEqual(
            rounded.real_world_cm(), round(state.scale_cm, 4)
        )

    def test_explicit_cm_overrides_the_state(self):
        panel = self.make(CalibrationPanel(real_world_cm=7.5))
        state = state_for(Y)
        state.scale_cm = 12.5
        panel.load(state)
        self.assertEqual(panel.scale_spin.value(), 7.5)

    def test_changed_and_issues(self):
        panel = self.make(CalibrationPanel())
        slot = CountingSlot(panel.changed)
        state = state_for(Y)
        panel.load(state)
        self.assertEqual(slot.count, 0)
        panel.scale_spin.setValue(5.0)
        self.assertEqual(slot.count, 1)
        self.assertEqual(panel.issues(), [])

        # A length without a line is reported and not applied.
        state.roi_manager = rois_for(create_maze(Y), reference=False)
        panel.load(state)
        panel.scale_spin.setEnabled(True)
        panel.scale_spin.setValue(5.0)
        self.assertEqual(len(panel.issues()), 1)
        self.assertFalse(panel.apply(state))
        self.assertEqual(state.scale_cm, 0.0)

    def test_load_without_rois(self):
        panel = self.make(CalibrationPanel())
        state = SetupState.new(Y)
        panel.load(state)
        self.assertEqual(panel.status.kind, REFERENCE_NONE)
        self.assertEqual(panel.issues(), [])


@unittest.skipUnless(HAS_QT, "GUI runtime is not installed")
class TimeWindowPanelTests(QtCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.video = Path(cls._tmp.name) / "clip.avi"
        write_video(cls.video)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def panel(self, **kwargs):
        panel = self.make(TimeWindowPanel(**kwargs))
        self.addCleanup(panel.release_capture)
        return panel

    def test_global_window_round_trip(self):
        panel = self.panel()
        state = state_for(Y, video=self.video, start_time=2.0, end_time=12.0)
        before = dump(state)
        panel.load(state)
        self.assertFalse(panel.is_per_roi)
        self.assertEqual(panel.start_spinbox.value(), 2.0)
        self.assertEqual(panel.end_spinbox.value(), 12.0)
        self.assertEqual(panel.start_slider.value(), 20)
        self.assertEqual(panel.end_slider.value(), 120)
        self.assertTrue(panel.apply(state))
        self.assertEqual(dump(state), before)
        self.assertEqual(state.step_status("time_window"), "done")

    def test_end_of_video_is_none_and_loads_back_to_the_top(self):
        panel = self.panel()
        state = state_for(Y, video=self.video, end_time=None)
        panel.load(state)
        self.assertAlmostEqual(panel.end_spinbox.value(), 20.0)
        self.assertEqual(panel.end_slider.value(), 200)
        self.assertIsNone(panel.end_time())
        panel.apply(state)
        self.assertIsNone(state.end_time)
        # Within a tenth of a second of the end counts as the end too.
        panel.end_spinbox.setValue(19.95)
        self.assertIsNone(panel.end_time())
        panel.end_spinbox.setValue(15.0)
        self.assertEqual(panel.end_time(), 15.0)

    def test_slider_steps_and_clamping(self):
        panel = self.panel()
        state = state_for(Y, video=self.video)
        panel.load(state)
        panel.start_slider.setValue(123)
        self.assertAlmostEqual(panel.start_spinbox.value(), 12.3)
        self.assertAlmostEqual(panel.end_spinbox.value(), 20.0)
        # The end cannot be dragged before the start ...
        panel.end_spinbox.setValue(5.0)
        self.assertAlmostEqual(panel.end_spinbox.value(), 12.3)
        # ... and the start cannot pass the end.
        panel.load(state_for(Y, video=self.video))
        panel.end_spinbox.setValue(10.0)
        panel.start_slider.setValue(150)
        self.assertAlmostEqual(panel.start_spinbox.value(), 10.0)
        panel.end_spinbox.setValue(10.55)
        panel.start_slider.setValue(5)
        self.assertAlmostEqual(panel.end_spinbox.value(), 10.55)
        panel.apply(state)
        self.assertAlmostEqual(state.start_time, 0.5)
        self.assertAlmostEqual(state.end_time, 10.55)

    def test_issues_for_an_empty_window(self):
        panel = self.panel()
        state = state_for(Y, video=self.video)
        panel.load(state)
        self.assertEqual(panel.issues(), [])
        panel.start_spinbox.setValue(8.0)
        panel.end_spinbox.setValue(8.0)
        self.assertEqual(
            panel.issues(), ["The end time must be greater than the start time."]
        )
        before = state.start_time
        self.assertFalse(panel.apply(state))
        self.assertEqual(state.start_time, before)
        self.assertNotIn("time_window", state.completed_steps)

    def test_per_mouse_windows_for_the_tst(self):
        panel = self.panel()
        state = state_for(T, video=self.video)
        state.set_per_roi_times({"mouse_1": (1.0, 9.0), "mouse_2": (2.5, 15.0)})
        before = dump(state)
        panel.load(state)
        self.assertTrue(panel.is_per_roi)
        self.assertEqual(sorted(panel.mouse_time_widgets),
                         ["mouse_1", "mouse_2"])
        self.assertEqual(panel.mouse_time_widgets["mouse_2"][0].value(), 2.5)
        self.assertTrue(panel.apply(state))
        self.assertEqual(dump(state), before)
        panel.mouse_time_widgets["mouse_1"][1].setValue(5.0)
        panel.apply(state)
        self.assertEqual(state.per_roi_times["mouse_1"], (1.0, 5.0))
        self.assertEqual((state.start_time, state.end_time), (1.0, 15.0))

    def test_per_mouse_defaults_and_issues(self):
        panel = self.panel()
        state = state_for(T, video=self.video)
        panel.load(state)
        start, end = panel.mouse_time_widgets["mouse_1"]
        self.assertEqual((start.value(), end.value()), (0.0, 20.0))
        start.setValue(9.0)
        end.setValue(4.0)
        self.assertEqual(panel.issues(), [
            "Mouse 1 must have an end time greater than its start time."
        ])
        self.assertFalse(panel.apply(state))
        self.assertEqual(state.per_roi_times, {})

    def test_per_mouse_rows_follow_the_loaded_rois(self):
        panel = self.panel()
        state = state_for(T, video=self.video)
        panel.load(state)
        self.assertEqual(len(panel.mouse_time_widgets), 2)
        maze = configured(T)
        maze.configure_mice(1)
        state.set_maze(maze)
        state.roi_manager = rois_for(maze, reference=False)
        panel.load(state)
        self.assertEqual(sorted(panel.mouse_time_widgets), ["mouse_1"])
        panel.load(state_for(Y, video=self.video))
        self.assertFalse(panel.is_per_roi)

    def test_changed_is_emitted_by_edits_and_not_by_load(self):
        panel = self.panel()
        slot = CountingSlot(panel.changed)
        panel.load(state_for(Y, video=self.video, start_time=3.0))
        panel.load(state_for(T, video=self.video))
        self.assertEqual(slot.count, 0)
        panel.mouse_time_widgets["mouse_1"][0].setValue(1.0)
        self.assertEqual(slot.count, 1)
        panel.load(state_for(Y, video=self.video))
        for edit in (
            lambda: panel.start_slider.setValue(10),
            lambda: panel.end_spinbox.setValue(10.0),
            lambda: panel.start_spinbox.setValue(2.0),
        ):
            before = slot.count
            edit()
            self.assertGreater(slot.count, before)

    def test_preview_shows_the_frame_with_the_rois(self):
        panel = self.panel()
        state = state_for(Y, video=self.video)
        panel.load(state)
        self.assertFalse(panel.preview_label.pixmap().isNull())
        panel.release_capture()
        self.assertIsNone(panel._cap)
        panel.update_preview(5.0)  # reopens on demand
        self.assertTrue(panel.cap.isOpened())

    def test_capture_is_released_when_hidden_and_closed(self):
        panel = self.panel()
        panel.load(state_for(Y, video=self.video))
        cap = panel.cap
        self.assertTrue(cap.isOpened())
        panel.show()
        panel.hide()
        self.assertFalse(cap.isOpened())
        cap = panel.cap
        panel.close()
        self.assertFalse(cap.isOpened())

    def test_without_a_video_nothing_breaks(self):
        panel = self.panel()
        state = state_for(Y, start_time=1.0, end_time=30.0)
        panel.load(state)
        self.assertIsNone(panel.cap)
        self.assertEqual(panel.issues(), [])
        panel.apply(state)
        self.assertEqual((state.start_time, state.end_time), (1.0, 30.0))
        # No video to measure: the end stays as typed, never "end of video".
        state.end_time = None
        panel.load(state)
        panel.apply(state)
        self.assertEqual(state.start_time, 1.0)

    def test_an_unreadable_video_is_an_issue(self):
        panel = self.panel()
        panel.load(state_for(Y, video=Path(self._tmp.name) / "missing.avi"))
        self.assertEqual(len(panel.issues()), 1)
        self.assertIn("Cannot read the video", panel.issues()[0])

    def test_values_the_box_rounds_survive_unless_edited(self):
        state = state_for(Y, video=self.video, start_time=1.234, end_time=9.876)
        panel = self.panel()
        panel.load(state)
        panel.apply(state)
        self.assertEqual((state.start_time, state.end_time), (1.234, 9.876))
        classic = self.panel(preserve_exact=False)
        classic.load(state)
        classic.apply(state)
        self.assertEqual((state.start_time, state.end_time), (1.23, 9.88))


@unittest.skipUnless(HAS_QT, "GUI runtime is not installed")
class ReviewPanelTests(QtCase):
    def finished(self, name):
        state = state_for(name)
        state.set_maze(configured(name))
        for step in state.steps:
            state.mark_done(step)
        state.mark_todo("review")
        return state

    def test_summary_lists_the_configuration(self):
        panel = self.make(ReviewPanel())
        state = self.finished(Y)
        state.scale_cm = 10.0
        state.start_time, state.end_time = 3.0, None
        panel.load(state)
        text = panel.summary_label.text()
        self.assertIn("Y-Maze", text)
        self.assertIn("arm_a (1)", text)
        self.assertIn("0.20000 cm per pixel", text)
        self.assertIn("3 s to end of video", text)
        self.assertEqual(panel.issues(), [])
        self.assertEqual(panel.issues_label.text(), "")

    def test_issues_cover_blocking_problems_and_unfinished_steps(self):
        panel = self.make(ReviewPanel())
        state = self.finished(Y)
        del state.roi_manager.rois["arm_c"]
        state.mark_todo("detection_tuning")
        panel.load(state)
        issues = panel.issues()
        self.assertTrue(any("Regions of interest" in i for i in issues))
        self.assertIn("Detection tuning is not done yet", issues)
        self.assertIn("To fix before running", panel.issues_label.text())
        before = set(state.completed_steps)
        self.assertFalse(panel.apply(state))
        self.assertEqual(state.completed_steps, before)

    def test_validation_video_round_trip(self):
        panel = self.make(ReviewPanel())
        state = self.finished(F)
        panel.load(state)
        self.assertFalse(panel.timelapse_check.isChecked())
        self.assertEqual(
            panel.timelapse_check.text(), "Create validation video (annotated)"
        )
        panel.timelapse_check.setChecked(True)
        self.assertTrue(panel.apply(state))
        self.assertTrue(state.create_timelapse)
        self.assertEqual(state.step_status("review"), "done")
        panel.load(state)
        self.assertTrue(panel.timelapse_check.isChecked())
        panel.timelapse_check.setChecked(False)
        panel.apply(state)
        self.assertFalse(state.create_timelapse)

    def test_every_paradigm_reviews_cleanly_when_finished(self):
        panel = self.make(ReviewPanel())
        for name in AVAILABLE_MAZES:
            with self.subTest(maze=name):
                state = self.finished(name)
                before = dump(state)
                panel.load(state)
                self.assertEqual(panel.issues(), [])
                self.assertTrue(panel.apply(state))
                self.assertEqual(dump(state), before)

    def test_changed_and_summary_less_mode(self):
        panel = self.make(ReviewPanel())
        slot = CountingSlot(panel.changed)
        panel.load(self.finished(Y))
        self.assertEqual(slot.count, 0)
        panel.timelapse_check.setChecked(True)
        self.assertEqual(slot.count, 1)

        bare = self.make(ReviewPanel(show_summary=False))
        state = self.finished(Y)
        bare.load(state)
        self.assertEqual(bare.issues(), [])
        bare.timelapse_check.setChecked(True)
        self.assertTrue(bare.apply(state))
        self.assertTrue(state.create_timelapse)
        self.assertFalse(bare.summary_label.isVisibleTo(bare))


@unittest.skipUnless(HAS_QT, "GUI runtime is not installed")
class SharedImplementationTests(QtCase):
    """Classic dialogs and the workspace panels are one implementation."""

    def test_dialogs_host_the_panels(self):
        from gui.arm_entry_dialog import ArmEntryParametersDialog
        from gui.freestyle_config_dialog import FreestyleConfigDialog
        from gui.settings_dialog import SettingsDialog

        arm = ArmEntryParametersDialog(create_maze(Y))
        self.addCleanup(arm.deleteLater)
        self.assertIsInstance(arm.panel, ArmEntryParamsPanel)
        self.assertIs(arm.dwell_spin, arm.panel.dwell_spin)

        free = FreestyleConfigDialog(maze=create_maze(F))
        self.addCleanup(free.deleteLater)
        self.assertIsInstance(free.panel, FreestyleParamsPanel)
        self.assertIs(free.zone_name_inputs, free.panel.zone_name_inputs)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            video = Path(tmp) / "clip.avi"
            write_video(video)
            for name, calibrated in ((Y, True), (T, False)):
                settings = state_for(name).to_batch_settings()
                dialog = SettingsDialog(str(video), settings, maze_name=name)
                self.addCleanup(dialog.deleteLater)
                self.assertIsInstance(dialog.time_panel, TimeWindowPanel)
                self.assertIsInstance(dialog.review_panel, ReviewPanel)
                # Calibration is only offered where distances are reported.
                self.assertEqual(
                    dialog.calibration_panel.parent() is not None, calibrated
                )
                self.assertIs(
                    dialog.timelapse_check, dialog.review_panel.timelapse_check
                )
                dialog._release_capture()

    def test_a_dialog_and_a_panel_agree_on_the_same_edit(self):
        from gui.arm_entry_dialog import ArmEntryParametersDialog

        for name in (Y, E):
            maze = create_maze(name)
            dialog = ArmEntryParametersDialog(maze)
            self.addCleanup(dialog.deleteLater)
            dialog.dwell_spin.setValue(0.35)
            dialog.initial_check.setChecked(False)
            via_dialog = create_maze(name)
            via_dialog.configure_algorithm_parameters(dialog.get_parameters())

            state = SetupState.new(name)
            panel = self.make(ParadigmParamsPanel())
            panel.load(state)
            panel.active_panel.dwell_spin.setValue(0.35)
            panel.active_panel.initial_check.setChecked(False)
            panel.apply(state)
            self.assertEqual(
                via_dialog.get_algorithm_parameters(),
                state.algorithm_parameters,
            )

    def test_freestyle_dialog_rejects_bad_names_through_the_panel(self):
        from gui.freestyle_config_dialog import FreestyleConfigDialog
        from unittest.mock import patch

        dialog = FreestyleConfigDialog()
        self.addCleanup(dialog.deleteLater)
        dialog.zone_name_inputs[0].setText("outside")
        with patch.object(QMessageBox, "warning") as warning:
            dialog.validate_and_accept()
        self.assertEqual(warning.call_args.args[1], "Reserved Zone Name")
        self.assertEqual(dialog.result(), 0)


if __name__ == "__main__":
    unittest.main()
