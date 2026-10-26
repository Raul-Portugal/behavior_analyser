"""SetupState: lossless BatchSettings round trip and hook-driven step status."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.models import AppConfig, BatchSettings
from core.setup_state import (STATUS_DONE, STATUS_INVALID, STATUS_TODO,
                              SetupState)
from core.tst_motion_engine import TSTMotionConfig
from mazes import AVAILABLE_MAZES, create_maze
from roi_manager import ROIManager

SQUARE = np.array([[10, 10], [60, 10], [60, 60], [10, 60]])
Y, E, F, T = (
    "Y-Maze", "Elevated Plus Maze", "Freestyle / Open Field",
    "Tail Suspension Test",
)


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


def settings_for(name, **overrides):
    maze = configured(name)
    kwargs = dict(
        roi_manager=rois_for(maze, reference=name != T),
        detection_config=AppConfig().detection,
        scale_factor=0.0 if name == T else 0.1,
        start_time=2.0, end_time=40.0, create_timelapse=True,
        per_roi_times={}, paradigm_id=name,
        algorithm_parameters=maze.get_algorithm_parameters(),
    )
    if name == T:
        kwargs["per_roi_times"] = {
            "mouse_1": (1.0, 10.0), "mouse_2": (2.5, 30.0),
        }
        kwargs["start_time"], kwargs["end_time"] = 1.0, 30.0
    kwargs.update(overrides)
    return BatchSettings(**kwargs)


def as_json(settings):
    return json.dumps(settings.to_dict(), sort_keys=True)


class RoundTripTests(unittest.TestCase):
    def test_round_trip_is_lossless_for_every_paradigm(self):
        for name in AVAILABLE_MAZES:
            with self.subTest(maze=name):
                settings = settings_for(name)
                state = SetupState.from_batch_settings(settings)
                self.assertEqual(state.maze_name, name)
                self.assertEqual(
                    as_json(state.to_batch_settings()), as_json(settings)
                )

    def test_round_trip_keeps_an_exact_scale_factor(self):
        # 0.1 cm/px over a 50 px line is not exactly 5.0 / 50 in floats.
        settings = settings_for(Y, scale_factor=0.07000000000000001)
        state = SetupState.from_batch_settings(settings)
        self.assertEqual(
            state.to_batch_settings().scale_factor, settings.scale_factor
        )

    def test_round_trip_keeps_arena_and_untagged_legacy_settings(self):
        maze = configured(F)
        maze.configure_arena(True, "area_fraction", 8.0, 0.3)
        settings = settings_for(
            F, roi_manager=rois_for(maze, arena=True),
            algorithm_parameters=maze.get_algorithm_parameters(),
        )
        state = SetupState.from_batch_settings(settings)
        self.assertEqual(as_json(state.to_batch_settings()), as_json(settings))

        untagged = settings_for(Y, paradigm_id=None)
        state = SetupState.from_batch_settings(untagged)
        self.assertEqual(state.maze_name, Y)
        self.assertTrue(state.legacy_untagged)
        self.assertIsNone(state.to_batch_settings().paradigm_id)
        self.assertEqual(state.to_batch_settings(finalize=True).paradigm_id, Y)

    def test_round_trip_survives_saved_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in AVAILABLE_MAZES:
                path = Path(tmp) / f"{abs(hash(name))}.json"
                settings_for(name).save_to_file(path)
                loaded = BatchSettings.load_from_file(path)
                with self.subTest(maze=name):
                    state = SetupState.from_batch_settings(loaded)
                    self.assertEqual(
                        as_json(state.to_batch_settings()), as_json(loaded)
                    )

    def test_states_are_independent_of_their_settings(self):
        settings = settings_for(Y)
        state = SetupState.from_batch_settings(settings)
        state.roi_manager.rois.clear()
        state.detection_config.threshold_percentile = 50.0
        self.assertTrue(settings.roi_manager.rois)
        self.assertEqual(settings.detection_config.threshold_percentile, 99.0)
        out = state.to_batch_settings()
        out.roi_manager.rois.clear()
        self.assertEqual(state.roi_manager.rois, {})

    def test_finalize_writes_paradigm_id_and_normalised_parameters(self):
        settings = settings_for(Y, algorithm_parameters={})
        state = SetupState.from_batch_settings(settings)
        self.assertEqual(state.to_batch_settings().algorithm_parameters, {})
        final = state.to_batch_settings(finalize=True)
        self.assertEqual(
            final.algorithm_parameters, create_maze(Y).get_algorithm_parameters()
        )

    def test_missing_rois_cannot_become_settings(self):
        with self.assertRaises(ValueError):
            SetupState.new(Y).to_batch_settings()

    def test_scale_follows_the_cm_length_and_reference(self):
        state = SetupState.from_batch_settings(settings_for(Y))
        state.scale_cm = 10.0
        self.assertAlmostEqual(state.scale_factor, 10.0 / 50.0)
        state.scale_cm = 0.0
        self.assertEqual(state.scale_factor, 0.0)

    def test_tst_never_reports_a_scale(self):
        state = SetupState.from_batch_settings(settings_for(T))
        state.scale_cm = 25.0
        self.assertEqual(state.scale_factor, 0.0)
        self.assertEqual(state.to_batch_settings().scale_factor, 0.0)

    def test_per_roi_times_set_the_global_window(self):
        state = SetupState.new(T)
        state.set_per_roi_times({"mouse_1": (5, 20), "mouse_2": (1.0, 30)})
        self.assertEqual((state.start_time, state.end_time), (1.0, 30.0))
        self.assertEqual(state.per_roi_times["mouse_1"], (5.0, 20.0))

    def test_new_state_has_default_parameters_per_paradigm(self):
        for name in AVAILABLE_MAZES:
            state = SetupState.new(name)
            self.assertEqual(
                state.algorithm_parameters,
                create_maze(name).get_algorithm_parameters(),
            )
            self.assertEqual(state.build_maze().name, name)

    def test_tst_motion_config_lives_in_the_parameters(self):
        state = SetupState.new(T)
        self.assertIsInstance(state.tst_motion_config, TSTMotionConfig)
        config = TSTMotionConfig(**{
            **state.tst_motion_config.__dict__,
            "min_mouse_area_ratio": 0.01,
        })
        state.tst_motion_config = config
        self.assertEqual(
            state.algorithm_parameters["motion_config"]["min_mouse_area_ratio"],
            0.01,
        )
        self.assertEqual(state.tst_motion_config.min_mouse_area_ratio, 0.01)
        self.assertIsNone(SetupState.new(Y).tst_motion_config)
        with self.assertRaises(ValueError):
            SetupState.new(Y).tst_motion_config = config

    def test_set_maze_adopts_parameters_and_tags_the_state(self):
        state = SetupState.from_batch_settings(
            settings_for(Y, paradigm_id=None))
        maze = state.build_maze()
        maze.min_dwell_seconds = 0.4
        state.set_maze(maze)
        self.assertFalse(state.legacy_untagged)
        self.assertEqual(
            state.algorithm_parameters["min_dwell_seconds"], 0.4
        )

    def test_legacy_untagged_freestyle_takes_zones_from_the_rois(self):
        maze = configured(F)
        settings = settings_for(F, paradigm_id=None, algorithm_parameters={})
        state = SetupState.from_batch_settings(settings, maze_name=F)
        self.assertEqual(
            [n for n, _ in state.build_maze().get_roi_definitions()],
            [n for n, _ in maze.get_roi_definitions()],
        )


class StepStatusTests(unittest.TestCase):
    def test_steps_come_from_the_maze(self):
        for name in AVAILABLE_MAZES:
            state = SetupState.new(name)
            self.assertEqual(state.steps, create_maze(name).setup_steps())
            self.assertEqual(list(state.step_reports()), state.steps)

    def test_a_blank_state_is_all_todo_except_review(self):
        state = SetupState.new(Y)
        for step in state.steps:
            self.assertEqual(
                state.step_status(step), STATUS_TODO, step
            )
        self.assertFalse(state.is_ready())

    def test_params_step(self):
        state = SetupState.new(Y)
        self.assertEqual(state.step_status("paradigm_params"), STATUS_TODO)
        state.mark_done("paradigm_params")
        self.assertEqual(state.step_status("paradigm_params"), STATUS_DONE)
        state.algorithm_parameters = {"min_dwell_seconds": 99.0}
        self.assertEqual(state.step_status("paradigm_params"), STATUS_INVALID)
        self.assertTrue(state.step_issues("paradigm_params"))
        state.mark_todo("paradigm_params")
        self.assertEqual(state.step_status("paradigm_params"), STATUS_INVALID)

    def test_rois_step(self):
        state = SetupState.new(Y)
        self.assertEqual(state.step_status("rois"), STATUS_TODO)
        state.roi_manager = ROIManager()
        self.assertEqual(state.step_status("rois"), STATUS_TODO)
        manager = rois_for(create_maze(Y))
        del manager.rois["arm_a"]
        state.roi_manager = manager
        report = state.step_report("rois")
        self.assertEqual(report.status, STATUS_INVALID)
        self.assertIn("Required category 'arm_a' is missing", report.issues)
        state.roi_manager = rois_for(create_maze(Y))
        self.assertEqual(state.step_status("rois"), STATUS_DONE)

    def test_tst_rois_follow_the_mouse_count(self):
        state = SetupState.from_batch_settings(settings_for(T))
        self.assertEqual(state.step_status("rois"), STATUS_DONE)
        maze = state.build_maze()
        maze.configure_mice(3)
        state.set_maze(maze)
        self.assertEqual(state.step_status("rois"), STATUS_INVALID)
        manager = rois_for(maze, reference=False)
        manager.add_roi("mouse_1", SQUARE + 300)
        state.roi_manager = manager
        self.assertIn(
            "mouse_1 must contain exactly one ROI, found 2",
            state.step_issues("rois"),
        )

    def test_freestyle_arena_is_needed_when_arena_mode_is_on(self):
        maze = configured(F)
        maze.configure_arena(True)
        state = SetupState.new(F)
        state.set_maze(maze)
        state.roi_manager = rois_for(maze)
        report = state.step_report("rois")
        self.assertEqual(report.status, STATUS_INVALID)
        self.assertIn("arena is enabled", report.issues[0])
        state.roi_manager = rois_for(maze, arena=True)
        self.assertEqual(state.step_status("rois"), STATUS_DONE)

    def test_zone_free_freestyle_needs_confirmation_only(self):
        maze = create_maze(F)
        maze.configure_zones([])
        state = SetupState.new(F)
        state.set_maze(maze)
        state.roi_manager = ROIManager()
        self.assertEqual(state.step_status("rois"), STATUS_TODO)
        state.mark_done("rois")
        self.assertEqual(state.step_status("rois"), STATUS_DONE)

    def test_calibration_step(self):
        state = SetupState.new(Y)
        self.assertEqual(state.step_status("calibration"), STATUS_TODO)
        state.roi_manager = rois_for(create_maze(Y), reference=False)
        self.assertEqual(state.step_status("calibration"), STATUS_TODO)
        state.mark_done("calibration")  # chose pixel units
        self.assertEqual(state.step_status("calibration"), STATUS_DONE)
        state.scale_cm = 12.0  # but there is no line to measure
        self.assertEqual(state.step_status("calibration"), STATUS_INVALID)
        state.roi_manager = rois_for(create_maze(Y), reference=True)
        self.assertEqual(state.step_status("calibration"), STATUS_DONE)
        state.scale_cm = -1.0
        self.assertEqual(state.step_status("calibration"), STATUS_INVALID)

    def test_calibration_warnings_for_legacy_and_approximate_references(self):
        state = SetupState.new(Y)
        manager = rois_for(create_maze(Y), reference=False)
        manager.set_reference_length(30.0, "Width of first Arm A ROI")
        state.roi_manager = manager
        state.scale_cm = 10.0
        report = state.step_report("calibration")
        self.assertEqual(report.status, STATUS_DONE)
        self.assertEqual(len(report.warnings), 1)
        manager.set_reference_length(
            30.0, "User-defined (approximate: scaled by alignment)"
        )
        self.assertEqual(len(state.step_report("calibration").warnings), 1)

    def test_tst_has_no_calibration_or_detection_step(self):
        state = SetupState.new(T)
        self.assertNotIn("calibration", state.steps)
        self.assertEqual(state.step_status("calibration"), STATUS_DONE)
        self.assertEqual(state.step_status("detection_tuning"), STATUS_DONE)
        self.assertEqual(state.step_status("tst_tuning"), STATUS_TODO)
        state.mark_done("tst_tuning")
        self.assertEqual(state.step_status("tst_tuning"), STATUS_DONE)

    def test_detection_step(self):
        state = SetupState.new(Y)
        self.assertEqual(state.step_status("detection_tuning"), STATUS_TODO)
        state.mark_done("detection_tuning")
        self.assertEqual(state.step_status("detection_tuning"), STATUS_DONE)
        state.detection_config.threshold_percentile = 120.0
        self.assertEqual(
            state.step_status("detection_tuning"), STATUS_INVALID
        )

    def test_time_window_step(self):
        state = SetupState.from_batch_settings(settings_for(Y))
        self.assertEqual(state.step_status("time_window"), STATUS_TODO)
        state.mark_done("time_window")
        self.assertEqual(state.step_status("time_window"), STATUS_DONE)
        state.end_time = state.start_time
        self.assertEqual(state.step_status("time_window"), STATUS_INVALID)
        state.end_time = None  # to the end of the video
        self.assertEqual(state.step_status("time_window"), STATUS_DONE)

    def test_tst_time_windows_per_mouse(self):
        state = SetupState.from_batch_settings(settings_for(T))
        state.mark_done("time_window")
        self.assertEqual(state.step_status("time_window"), STATUS_DONE)
        state.per_roi_times["mouse_2"] = (9.0, 3.0)
        report = state.step_report("time_window")
        self.assertEqual(report.status, STATUS_INVALID)
        self.assertIn("Mouse 2", report.issues[0])
        state.per_roi_times["mouse_2"] = (3.0, 9.0)
        state.per_roi_times["mouse_9"] = (3.0, 9.0)
        self.assertEqual(state.step_status("time_window"), STATUS_INVALID)

    def test_review_aggregates_every_step(self):
        for name in AVAILABLE_MAZES:
            with self.subTest(maze=name):
                state = SetupState.from_batch_settings(settings_for(name))
                self.assertEqual(state.step_status("review"), STATUS_TODO)
                for step in state.steps:
                    state.mark_done(step)
                self.assertEqual(state.step_status("review"), STATUS_DONE)
                self.assertTrue(state.is_ready())

    def test_review_is_invalid_when_a_step_is_and_lists_why(self):
        state = SetupState.from_batch_settings(settings_for(Y))
        for step in state.steps:
            state.mark_done(step)
        del state.roi_manager.rois["arm_c"]
        report = state.step_report("review")
        self.assertEqual(report.status, STATUS_INVALID)
        self.assertTrue(any("Regions of interest" in i for i in report.issues))
        self.assertFalse(state.is_ready())

    def test_review_catches_a_paradigm_mismatch_in_the_settings(self):
        state = SetupState.from_batch_settings(settings_for(Y))
        for step in state.steps:
            state.mark_done(step)
        state.maze_name = E  # parameters and ROIs belong to the Y-maze
        self.assertEqual(state.step_status("review"), STATUS_INVALID)

    def test_summary_lines_describe_every_paradigm(self):
        for name in AVAILABLE_MAZES:
            with self.subTest(maze=name):
                state = SetupState.from_batch_settings(settings_for(name))
                rows = dict(state.summary_lines())
                self.assertEqual(rows["Paradigm"], name)
                self.assertIn("Regions", rows)
                self.assertEqual(rows["Validation video"], "yes")
                self.assertEqual(
                    "Calibration" in rows, create_maze(name).reports_distance
                )
                self.assertEqual(
                    "Time windows" in rows, name == T
                )
        rows = dict(SetupState.from_batch_settings(
            settings_for(Y)).summary_lines())
        self.assertIn("0.10000 cm per pixel", rows["Calibration"])
        self.assertEqual(rows["Time window"], "2 s to 40 s")
        state = SetupState.new(Y)
        self.assertEqual(dict(state.summary_lines())["Regions"],
                         "not drawn yet")

    def test_steps_a_paradigm_lacks_are_done_and_unknown_ids_rejected(self):
        state = SetupState.new(Y)
        self.assertEqual(state.step_status("tst_tuning"), STATUS_DONE)
        with self.assertRaises(ValueError):
            state.step_report("nonsense")


if __name__ == "__main__":
    unittest.main()
