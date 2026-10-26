"""Setup hooks on every maze: steps, flags, ROI rules and inference."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.models import AppConfig, BatchSettings
from mazes import (AVAILABLE_MAZES, create_maze, infer_maze_from_settings,
                   maze_class_for_settings, paradigm_analysis_kind)
from mazes.base_maze import (ALL_SETUP_STEPS, ANALYSIS_MOTION,
                             ANALYSIS_TRACKING, KIND_ARMS, KIND_FREESTYLE,
                             KIND_STANDARD, KIND_TST)
from mazes.epm import EPM
from mazes.freestyle import Freestyle
from mazes.tst import TST
from mazes.y_maze import YMaze
from roi_manager import ROIManager

SQUARE = np.array([[10, 10], [60, 10], [60, 60], [10, 60]])
Y, E, F, T = (
    "Y-Maze", "Elevated Plus Maze", "Freestyle / Open Field",
    "Tail Suspension Test",
)


def manager_for(maze, *, polygons=1) -> ROIManager:
    manager = ROIManager(source_dimensions=(100, 100))
    for name, _ in maze.get_roi_definitions():
        for index in range(polygons):
            shifted = SQUARE + 100 * index
            manager.add_roi(name, shifted)
    return manager


def settings_for(maze, manager=None, paradigm_id="maze", **kwargs):
    manager = manager or manager_for(maze)
    return BatchSettings(
        manager, AppConfig().detection, 0.0,
        paradigm_id=maze.name if paradigm_id == "maze" else paradigm_id,
        algorithm_parameters=maze.get_algorithm_parameters(), **kwargs,
    )


def configured(name: str):
    maze = create_maze(name)
    if name == F:
        maze.configure_zones([("zone_a", "Zone A"), ("zone_b", "Zone B")])
    if name == T:
        maze.configure_mice(2)
    return maze


class HookValuesTests(unittest.TestCase):
    """Every hook on every maze."""

    EXPECTED = {
        Y: dict(
            steps=["paradigm_params", "rois", "calibration",
                   "detection_tuning", "time_window", "review"],
            kind=ANALYSIS_TRACKING, wizard=KIND_ARMS, detection=True,
            motion=False, per_roi=False, distance=True, max_polygons=None,
        ),
        E: dict(
            steps=["paradigm_params", "rois", "calibration",
                   "detection_tuning", "time_window", "review"],
            kind=ANALYSIS_TRACKING, wizard=KIND_ARMS, detection=True,
            motion=False, per_roi=False, distance=True, max_polygons=None,
        ),
        F: dict(
            steps=["paradigm_params", "rois", "calibration",
                   "detection_tuning", "time_window", "review"],
            kind=ANALYSIS_TRACKING, wizard=KIND_FREESTYLE, detection=True,
            motion=False, per_roi=False, distance=True, max_polygons=None,
        ),
        T: dict(
            steps=["paradigm_params", "rois", "tst_tuning", "time_window",
                   "review"],
            kind=ANALYSIS_MOTION, wizard=KIND_TST, detection=False,
            motion=True, per_roi=True, distance=False, max_polygons=1,
        ),
    }

    def test_hooks_on_every_maze(self):
        self.assertEqual(set(self.EXPECTED), set(AVAILABLE_MAZES))
        for name, expected in self.EXPECTED.items():
            with self.subTest(maze=name):
                maze = create_maze(name)
                self.assertEqual(maze.setup_steps(), expected["steps"])
                self.assertEqual(maze.analysis_kind, expected["kind"])
                self.assertEqual(maze.wizard_kind, expected["wizard"])
                self.assertIs(maze.uses_detection_tuner, expected["detection"])
                self.assertIs(maze.uses_motion_tuner, expected["motion"])
                self.assertIs(maze.has_per_roi_times, expected["per_roi"])
                self.assertIs(maze.reports_distance, expected["distance"])
                self.assertIs(maze.needs_reference_line(), expected["distance"])
                self.assertEqual(
                    maze.max_polygons_per_roi, expected["max_polygons"]
                )

    def test_setup_steps_are_known_unique_and_end_in_review(self):
        for name in AVAILABLE_MAZES:
            with self.subTest(maze=name):
                steps = create_maze(name).setup_steps()
                self.assertTrue(set(steps) <= set(ALL_SETUP_STEPS))
                self.assertEqual(len(steps), len(set(steps)))
                self.assertEqual(steps[-1], "review")
                # Calibration and tuners follow the maze's own flags.
                maze = create_maze(name)
                self.assertEqual(
                    "calibration" in steps, maze.reports_distance
                )
                self.assertEqual(
                    "detection_tuning" in steps, maze.uses_detection_tuner
                )
                self.assertEqual(
                    "tst_tuning" in steps, maze.uses_motion_tuner
                )
                self.assertEqual(steps[0] == "paradigm_params",
                                 maze.wizard_kind != KIND_STANDARD)

    def test_wizard_kinds_match_the_wizard_constants(self):
        from gui import setup_progress
        self.assertEqual(KIND_TST, setup_progress.KIND_TST)
        self.assertEqual(KIND_ARMS, setup_progress.KIND_ARMS)
        self.assertEqual(KIND_FREESTYLE, setup_progress.KIND_FREESTYLE)
        self.assertEqual(KIND_STANDARD, setup_progress.KIND_STANDARD)

    def test_time_window_subjects(self):
        tst = configured(T)
        manager = manager_for(tst)
        self.assertEqual(
            tst.time_window_subjects(manager), ["mouse_1", "mouse_2"]
        )
        for name in (Y, E, F):
            maze = configured(name)
            self.assertEqual(
                maze.time_window_subjects(manager_for(maze)), []
            )

    def test_paradigm_analysis_kind(self):
        self.assertEqual(paradigm_analysis_kind(T), ANALYSIS_MOTION)
        self.assertEqual(paradigm_analysis_kind(Y), ANALYSIS_TRACKING)
        self.assertEqual(paradigm_analysis_kind("EPM"), ANALYSIS_TRACKING)
        self.assertEqual(paradigm_analysis_kind(None), ANALYSIS_TRACKING)


class ParameterValidationTests(unittest.TestCase):
    def test_defaults_are_valid_everywhere(self):
        for name in AVAILABLE_MAZES:
            with self.subTest(maze=name):
                self.assertEqual(configured(name).validate_parameters(), [])

    def test_arm_dwell_out_of_range(self):
        for maze in (YMaze(), EPM()):
            maze.min_dwell_seconds = 11.0
            self.assertEqual(len(maze.validate_parameters()), 1)

    def test_tst_parameters(self):
        tst = TST()
        tst.configure_mice(9)
        self.assertEqual(len(tst.validate_parameters()), 1)
        tst.configure_mice(2)
        tst.min_time_seconds = -1.0
        self.assertIn("negative", tst.validate_parameters()[0])

    def test_freestyle_parameters(self):
        maze = Freestyle()
        maze.min_dwell_seconds = -0.1
        maze.user_defined_zones = [("a", "A"), ("a", "A again")]
        self.assertEqual(len(maze.validate_parameters()), 2)


class ValidateRoisTests(unittest.TestCase):
    def test_complete_rois_are_valid(self):
        for name in AVAILABLE_MAZES:
            with self.subTest(maze=name):
                maze = configured(name)
                self.assertEqual(maze.validate_rois(manager_for(maze)), [])

    def test_missing_required_categories(self):
        for name in (Y, E, F, T):
            with self.subTest(maze=name):
                maze = configured(name)
                issues = maze.validate_rois(ROIManager())
                self.assertTrue(issues)
        maze = configured(Y)
        manager = manager_for(maze)
        del manager.rois["arm_c"]
        self.assertIn(
            "Required category 'arm_c' is missing", maze.validate_rois(manager)
        )

    def test_no_roi_manager_is_an_issue(self):
        for name in AVAILABLE_MAZES:
            self.assertTrue(configured(name).validate_rois(None))

    def test_tst_one_polygon_per_mouse(self):
        tst = configured(T)
        manager = manager_for(tst, polygons=2)
        issues = tst.validate_rois(manager)
        self.assertIn("mouse_1 must contain exactly one ROI, found 2", issues)
        self.assertIn("mouse_2 must contain exactly one ROI, found 2", issues)

    def test_tst_unexpected_categories(self):
        tst = configured(T)
        manager = manager_for(tst)
        manager.add_roi("mouse_7", SQUARE)
        self.assertIn(
            "unexpected TST ROI categories: mouse_7",
            tst.validate_rois(manager),
        )

    def test_other_mazes_allow_several_polygons(self):
        for name in (Y, E):
            maze = configured(name)
            self.assertEqual(
                maze.validate_rois(manager_for(maze, polygons=2)), []
            )

    def test_freestyle_arena_required_when_arena_mode_is_on(self):
        maze = configured(F)
        manager = manager_for(maze)
        self.assertEqual(maze.validate_rois(manager), [])
        maze.configure_arena(True)
        issues = maze.validate_rois(manager)
        self.assertEqual(len(issues), 1)
        self.assertIn("arena is enabled but no arena was drawn", issues[0])
        manager.set_arena(SQUARE)
        self.assertEqual(maze.validate_rois(manager), [])

    def test_zone_free_freestyle_needs_no_rois(self):
        maze = Freestyle()
        maze.configure_zones([])
        self.assertEqual(maze.validate_rois(ROIManager()), [])

    def test_validate_settings_adds_time_window_and_paradigm_checks(self):
        tst = configured(T)
        manager = manager_for(tst)
        settings = settings_for(tst, manager,
                                per_roi_times={"mouse_9": (0.0, 5.0)})
        self.assertEqual(
            tst.validate_settings(settings),
            ["time ranges reference unknown mice: mouse_9"],
        )
        wrong = settings_for(tst, manager, paradigm_id=Y)
        self.assertEqual(
            tst.validate_settings(wrong),
            [f"configuration is for '{Y}', not '{T}'"],
        )
        untagged = settings_for(tst, manager, paradigm_id=None)
        self.assertEqual(tst.validate_settings(untagged), [])


class ReferenceValidationTests(unittest.TestCase):
    """The hooks give the same issues the inlined branches used to."""

    @staticmethod
    def old_issues(maze, settings):
        issues = []
        if (
            getattr(maze, "arena_enabled", False)
            and settings.roi_manager.arena is None
        ):
            issues.append(
                "the open-field arena is enabled but no arena was drawn. "
                "Redraw the regions for this video."
            )
        required = [name for name, _ in maze.get_roi_definitions()]
        if required:
            _, roi_issues = settings.roi_manager.validate_rois(required)
            issues.extend(roi_issues)
        if isinstance(maze, TST):
            extra = set(settings.roi_manager.rois) - set(required)
            if extra:
                issues.append(
                    f"unexpected TST ROI categories: {', '.join(sorted(extra))}"
                )
            for category in set(required):
                count = len(settings.roi_manager.rois.get(category, []))
                if count != 1:
                    issues.append(
                        f"{category} must contain exactly one ROI, "
                        f"found {count}"
                    )
            unknown = set(settings.per_roi_times) - set(required)
            if unknown:
                issues.append(
                    "time ranges reference unknown mice: "
                    f"{', '.join(sorted(unknown))}"
                )
        if settings.paradigm_id and settings.paradigm_id != maze.name:
            issues.append(
                f"configuration is for '{settings.paradigm_id}', "
                f"not '{maze.name}'"
            )
        return issues

    def test_matches_the_old_inline_rules(self):
        for name in AVAILABLE_MAZES:
            maze = configured(name)
            for polygons in (1, 2):
                for paradigm_id in (name, None, Y if name != Y else E):
                    manager = manager_for(maze, polygons=polygons)
                    manager.add_roi("stray", SQUARE)
                    settings = settings_for(
                        maze, manager, paradigm_id=paradigm_id,
                        per_roi_times={"ghost": (0.0, 1.0)},
                    )
                    with self.subTest(maze=name, polygons=polygons,
                                      paradigm=paradigm_id):
                        expected = self.old_issues(maze, settings)
                        got = maze.validate_settings(settings)
                        # Order of the per-mouse count messages follows set
                        # iteration in the old code.
                        self.assertEqual(sorted(got), sorted(expected))


class InferenceTests(unittest.TestCase):
    @staticmethod
    def old_infer(settings):
        """The inference ``MainWindow`` used before the hooks existed."""
        if settings.paradigm_id in AVAILABLE_MAZES:
            maze = create_maze(settings.paradigm_id)
            maze.configure_algorithm_parameters(settings.algorithm_parameters)
            return maze
        keys = set(settings.roi_manager.rois.keys())
        if "arm_a" in keys and "arm_b" in keys:
            return create_maze(Y)
        if "open_arm_1" in keys and "closed_arm_1" in keys:
            return create_maze(E)
        if any(k.startswith("mouse_") for k in keys):
            return create_maze(T)
        maze = create_maze(F)
        maze.configure_zones(
            [(k, k.replace("_", " ").title()) for k in keys if k != "outside"]
        )
        return maze

    @staticmethod
    def old_configure(maze, settings, maze_choice):
        """``MainWindow._configure_maze_from_settings`` before the hooks."""
        if settings.algorithm_parameters:
            maze.configure_algorithm_parameters(settings.algorithm_parameters)
        rois = settings.roi_manager.rois
        if maze_choice == F:
            maze.configure_zones([
                (k, k.replace("_", " ").title())
                for k in rois.keys() if k != "outside"
            ])
        elif isinstance(maze, TST):
            if settings.paradigm_id == maze.name:
                count = sum(1 for k in rois.keys() if k != "outside")
            else:
                count = sum(1 for k in rois.keys() if k.startswith("mouse_"))
            maze.configure_mice(count)

    @staticmethod
    def state_of(maze):
        return (
            maze.name, maze.get_algorithm_parameters(),
            [name for name, _ in maze.get_roi_definitions()],
        )

    def fixtures(self):
        """Saved configs: each paradigm, tagged and untagged, round-tripped."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for name in AVAILABLE_MAZES:
            maze = configured(name)
            if name == F:
                maze.configure_arena(True, "area_fraction", 8.0, 0.3)
                maze.min_dwell_seconds = 0.4
            if name == Y:
                maze.min_dwell_seconds = 0.35
                maze.count_initial_arm = False
            for tagged in (True, False):
                manager = manager_for(maze)
                if name == F:
                    manager.set_arena(SQUARE)
                settings = settings_for(
                    maze, manager, paradigm_id=name if tagged else None
                )
                path = Path(tmp.name) / f"{len(list(Path(tmp.name).iterdir()))}.json"
                settings.save_to_file(path)
                yield f"{name}/{'tagged' if tagged else 'legacy'}", \
                    BatchSettings.load_from_file(path)
        # A project-style file that carries a free-form paradigm id.
        manager = manager_for(configured(E))
        yield "unknown-id", BatchSettings(
            manager, AppConfig().detection, 0.0, paradigm_id="EPM"
        )

    def test_old_and_new_inference_agree_on_saved_configs(self):
        seen = 0
        for label, settings in self.fixtures():
            with self.subTest(config=label):
                old_maze = self.old_infer(settings)
                new_maze = infer_maze_from_settings(settings)
                self.assertIs(type(old_maze), type(new_maze))
                if settings.paradigm_id in AVAILABLE_MAZES:
                    self.assertEqual(
                        self.state_of(old_maze), self.state_of(new_maze)
                    )
                else:
                    # The old legacy branch ignored saved parameters: it
                    # listed Freestyle zones from a set (arbitrary order)
                    # and never counted TST mice (always 1). The new hook
                    # gives the zones / mice the ROIs actually contain.
                    roi_names = sorted(
                        k for k in settings.roi_manager.rois if k != "outside"
                    )
                    new_names = sorted(
                        n for n, _ in new_maze.get_roi_definitions()
                    )
                    if isinstance(new_maze, (Freestyle, TST)):
                        self.assertEqual(new_names, roi_names)
                    if isinstance(old_maze, Freestyle):
                        self.assertEqual(sorted(
                            n for n, _ in old_maze.get_roi_definitions()
                        ), new_names)
                self.assertTrue(type(new_maze).matches_settings(settings))
                for other in AVAILABLE_MAZES.values():
                    if other is not type(new_maze):
                        self.assertFalse(other.matches_settings(settings))
                seen += 1
        self.assertEqual(seen, 9)

    def test_old_and_new_configure_agree(self):
        for label, settings in self.fixtures():
            for choice in (settings.paradigm_id, F, T, Y):
                if choice not in AVAILABLE_MAZES:
                    continue
                with self.subTest(config=label, maze=choice):
                    old_maze, new_maze = create_maze(choice), create_maze(choice)
                    try:
                        self.old_configure(old_maze, settings, choice)
                    except ValueError:
                        with self.assertRaises(ValueError):
                            new_maze.infer_from_settings(settings)
                        continue
                    new_maze.infer_from_settings(settings)
                    old_state = self.state_of(old_maze)
                    new_state = self.state_of(new_maze)
                    # Zone display names are cosmetic; compare internal ones.
                    self.assertEqual(old_state[2], new_state[2])
                    if choice == F:
                        old_params = dict(old_state[1])
                        new_params = dict(new_state[1])
                        for params in (old_params, new_params):
                            params["zones"] = [z[0] for z in params["zones"]]
                        self.assertEqual(old_params, new_params)
                    else:
                        self.assertEqual(old_state[1], new_state[1])

    def test_legacy_heuristics_and_priority(self):
        def legacy(keys):
            manager = ROIManager()
            for key in keys:
                manager.add_roi(key, SQUARE)
            return BatchSettings(manager, AppConfig().detection, 0.0)

        cases = [
            (["arm_a", "arm_b", "arm_c", "center"], YMaze),
            (["open_arm_1", "closed_arm_1", "center"], EPM),
            (["mouse_1"], TST),
            (["zone_a", "zone_b"], Freestyle),
            (["arm_a", "arm_b", "mouse_1"], YMaze),
            (["arm_a", "mouse_1"], TST),
            (["center"], Freestyle),
        ]
        for keys, expected in cases:
            with self.subTest(keys=keys):
                settings = legacy(keys)
                self.assertIs(maze_class_for_settings(settings), expected)
                self.assertIs(self.old_infer(settings).__class__, expected)
                self.assertTrue(expected.matches_settings(settings))

    def test_paradigm_id_beats_roi_names(self):
        manager = ROIManager()
        manager.add_roi("mouse_1", SQUARE)
        settings = BatchSettings(
            manager, AppConfig().detection, 0.0, paradigm_id=F
        )
        self.assertIs(maze_class_for_settings(settings), Freestyle)
        self.assertFalse(TST.matches_settings(settings))
        maze = infer_maze_from_settings(settings)
        self.assertEqual(
            [name for name, _ in maze.get_roi_definitions()], ["mouse_1"]
        )

    def test_tst_infer_counts_mice(self):
        tst = configured(T)
        settings = settings_for(tst)
        restored = TST()
        restored.infer_from_settings(settings)
        self.assertEqual(restored.num_mice, 2)
        legacy = settings_for(tst, paradigm_id=None)
        legacy.algorithm_parameters = {}
        restored = TST()
        restored.infer_from_settings(legacy)
        self.assertEqual(restored.num_mice, 2)

    def test_unknown_names_fall_back_to_freestyle_zones(self):
        manager = ROIManager()
        manager.add_roi("left_side", SQUARE)
        manager.add_roi("outside", SQUARE)
        settings = BatchSettings(manager, AppConfig().detection, 0.0)
        maze = infer_maze_from_settings(settings)
        self.assertEqual(
            maze.get_roi_definitions(), [("left_side", "Left Side")]
        )


class RoiEditorRuleTests(unittest.TestCase):
    def test_editor_rule_comes_from_the_maze(self):
        from gui.roi_editor.rules import roi_rule_for_maze
        self.assertEqual(roi_rule_for_maze(TST()).maximum, 1)
        for maze in (YMaze(), EPM(), Freestyle()):
            self.assertIsNone(roi_rule_for_maze(maze).maximum)
            self.assertEqual(roi_rule_for_maze(maze).minimum, 1)


if __name__ == "__main__":
    unittest.main()
