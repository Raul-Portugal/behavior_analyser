import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.arena_zones import center_threshold_for_area_fraction
from mazes.freestyle import (
    ARENA_METRIC_HEADERS,
    CENTER_BY_AREA_FRACTION,
    CENTER_BY_DISTANCE,
    Freestyle,
    FreestyleAnalysisResult,
)
from output import DataExporter

FPS = 30.0
# 400 x 400 px arena; at 0.05 cm/px that is 20 x 20 cm.
ARENA = [[0, 0], [400, 0], [400, 400], [0, 400]]
SCALE = 0.05
BASELINE_HEADERS = [
    'average_speed', 'peak_speed', 'speed_unit',
    'immobility_threshold', 'immobility_threshold_unit',
    'immobility_time_s', 'immobility_pct', 'immobility_bouts',
    'smoothed_total_distance', 'movement_efficiency',
]


def _walk(start, end, frames):
    """Evenly spaced positions from start (inclusive) to end (exclusive)."""
    return [
        (
            start[0] + (end[0] - start[0]) * i / frames,
            start[1] + (end[1] - start[1]) * i / frames,
        )
        for i in range(frames)
    ]


def _make_result(positions, scale_factor=SCALE, arena=ARENA, jump_limit=None,
                 roi_labels=None):
    return FreestyleAnalysisResult(
        positions=list(positions),
        timestamps=[i / FPS for i in range(len(positions))],
        fps=FPS,
        scale_factor=scale_factor,
        arena_polygon=arena,
        jump_limit_px=jump_limit,
        roi_labels=list(roi_labels or []),
    )


def _arena_maze(mode=CENTER_BY_DISTANCE, distance_cm=4.0, fraction=0.25,
                dwell=0.1):
    maze = Freestyle()
    maze.configure_arena(True, mode, distance_cm, fraction)
    maze.min_dwell_seconds = dwell
    return maze


def _metrics(maze, result):
    maze.calculate_metrics(result)
    return result._arena_metrics


class ThigmotaxisTrackTests(unittest.TestCase):
    def test_track_along_the_wall_is_all_periphery(self):
        # 20 px from the top wall for 3 s.
        positions = [(20.0 + 3.0 * i, 20.0) for i in range(90)]
        metrics = _metrics(_arena_maze(), _make_result(positions))

        self.assertAlmostEqual(metrics['periphery_time_s'], 3.0)
        self.assertEqual(metrics['center_time_s'], 0.0)
        self.assertEqual(metrics['center_entries'], 0)
        self.assertIsNone(metrics['latency_to_center_s'])
        self.assertAlmostEqual(metrics['thigmotaxis_time_pct'], 100.0)
        self.assertAlmostEqual(metrics['thigmotaxis_distance_pct'], 100.0)
        self.assertAlmostEqual(metrics['mean_wall_distance'], 20 * SCALE)
        self.assertIsNone(metrics['center_speed_mean'])
        self.assertGreater(metrics['periphery_speed_mean'], 0.0)

    def test_track_through_the_center_has_no_thigmotaxis(self):
        positions = [(100.0 + 2.0 * i, 200.0) for i in range(100)]
        metrics = _metrics(_arena_maze(), _make_result(positions))

        self.assertAlmostEqual(metrics['center_time_s'], 100 / FPS)
        self.assertEqual(metrics['periphery_time_s'], 0.0)
        self.assertAlmostEqual(metrics['thigmotaxis_time_pct'], 0.0)
        self.assertAlmostEqual(metrics['thigmotaxis_distance_pct'], 0.0)
        # Distance to the nearest wall: min(x, 400 - x, 200).
        expected = np.mean([
            min(x, 400 - x, 200.0)
            for x in (100.0 + 2.0 * i for i in range(100))
        ]) * SCALE
        self.assertAlmostEqual(metrics['mean_wall_distance'], expected)
        self.assertEqual(metrics['center_entries'], 0)
        self.assertAlmostEqual(metrics['latency_to_center_s'], 0.0)

    def test_mixed_track_times_entries_latency_and_distance(self):
        # 1 s on the wall, 2 s through the center, 1 s back on the wall.
        wall_a = _walk((20, 20), (140, 20), 30)           # 4 px/frame
        center = _walk((140, 200), (260, 200), 60)        # 2 px/frame
        wall_b = _walk((380, 380), (380, 260), 30)        # 4 px/frame
        result = _make_result(wall_a + center + wall_b)
        metrics = _metrics(_arena_maze(), result)

        self.assertAlmostEqual(metrics['periphery_time_s'], 2.0)
        self.assertAlmostEqual(metrics['center_time_s'], 2.0)
        self.assertEqual(metrics['center_entries'], 1)
        self.assertAlmostEqual(metrics['latency_to_center_s'], 1.0)
        self.assertAlmostEqual(metrics['thigmotaxis_time_pct'], 50.0)

        # Distance per step ending on a wall / center frame. The first frame
        # of each segment also carries the (large) step from the previous
        # segment, so compare with the exact step lengths.
        positions = result.positions
        steps = [
            float(np.hypot(b[0] - a[0], b[1] - a[1]))
            for a, b in zip(positions, positions[1:])
        ]
        labels = result._arena_labels
        periphery = sum(
            step for step, label in zip(steps, labels[1:]) if label == 'periphery'
        )
        center = sum(
            step for step, label in zip(steps, labels[1:]) if label == 'center'
        )
        self.assertAlmostEqual(
            metrics['thigmotaxis_distance_pct'],
            100.0 * periphery / (periphery + center),
        )
        self.assertGreater(metrics['thigmotaxis_distance_pct'], 50.0)

    def test_brief_center_excursion_is_debounced(self):
        # Two frames in the center (< 0.1 s dwell = 3 frames) are suppressed.
        positions = (
            [(20.0, 20.0)] * 30 + [(200.0, 200.0)] * 2 + [(20.0, 20.0)] * 30
        )
        metrics = _metrics(_arena_maze(), _make_result(positions))
        self.assertEqual(metrics['center_entries'], 0)
        self.assertEqual(metrics['center_time_s'], 0.0)

        metrics = _metrics(_arena_maze(dwell=0.0), _make_result(positions))
        self.assertEqual(metrics['center_entries'], 1)
        self.assertAlmostEqual(metrics['center_time_s'], 2 / FPS)

    def test_undetected_frames_are_not_arena_time(self):
        positions = (
            [(20.0, 20.0)] * 30 + [None] * 30 + [(200.0, 200.0)] * 30
        )
        result = _make_result(positions)
        metrics = _metrics(_arena_maze(), result)
        self.assertAlmostEqual(metrics['periphery_time_s'], 1.0)
        self.assertAlmostEqual(metrics['center_time_s'], 1.0)
        # The gap hides the transition, so it is not counted as an entry.
        self.assertEqual(metrics['center_entries'], 0)
        self.assertEqual(result._arena_labels[30:60], ['undetected'] * 30)
        self.assertIsNone(result._arena_wall_distances[40])

    def test_speeds_are_reported_per_zone_in_cm_per_s(self):
        # 6 px/frame on the wall, 1.5 px/frame in the center.
        wall = [(10.0 + 6.0 * i, 20.0) for i in range(60)]
        center = [(100.0 + 1.5 * i, 200.0) for i in range(60)]
        # A tracking gap keeps the wall -> center hop out of the speeds.
        metrics = _metrics(
            _arena_maze(), _make_result(wall + [None] * 5 + center)
        )
        self.assertAlmostEqual(
            metrics['periphery_speed_mean'], 6.0 * FPS * SCALE, delta=0.3
        )
        self.assertAlmostEqual(
            metrics['center_speed_mean'], 1.5 * FPS * SCALE, delta=0.3
        )

    def test_excluded_jumps_and_gaps_follow_the_analyzer_rule(self):
        # A 300 px jump (over the 100 px limit) must not count as distance.
        positions = [(20.0, 20.0), (23.0, 20.0), (323.0, 20.0), (326.0, 20.0)]
        positions += [(200.0, 200.0), (203.0, 200.0), (206.0, 200.0)]
        result = _make_result(
            positions, jump_limit=100.0
        )
        metrics = _metrics(_arena_maze(dwell=0.0), result)
        # Periphery steps: 3 (20->23) and 3 (323->326); the jump is excluded.
        # The wall -> center hop (326,20)->(200,200) is 219.7 px, excluded too.
        # Center steps: 3 and 3.
        self.assertAlmostEqual(metrics['thigmotaxis_distance_pct'], 50.0)

    def test_outside_positions_are_neither_center_nor_periphery(self):
        positions = [(200.0, 200.0)] * 30 + [(450.0, 200.0)] * 30
        result = _make_result(positions)
        metrics = _metrics(_arena_maze(), result)
        self.assertAlmostEqual(metrics['center_time_s'], 1.0)
        self.assertEqual(metrics['periphery_time_s'], 0.0)
        self.assertEqual(result._arena_labels[40], 'outside')
        self.assertAlmostEqual(metrics['thigmotaxis_time_pct'], 0.0)


class CenterDefinitionTests(unittest.TestCase):
    def _rim_and_core(self):
        # Core points at 100 px from every wall; rim points 30 px in.
        return [(200.0, 200.0)] * 30 + [(30.0, 200.0)] * 30

    def test_distance_rule_is_used_when_calibrated(self):
        result = _make_result(self._rim_and_core())
        metrics = _metrics(_arena_maze(distance_cm=4.0), result)
        self.assertEqual(metrics['center_definition'], CENTER_BY_DISTANCE)
        self.assertAlmostEqual(metrics['center_threshold_px'], 80.0)
        self.assertAlmostEqual(metrics['center_threshold_cm'], 4.0)
        self.assertEqual(metrics['distance_unit'], 'cm')
        self.assertFalse(any('area fraction' in n for n in result.notes))

    def test_distance_rule_can_make_a_point_periphery(self):
        # 30 px = 1.5 cm from the wall: periphery at 4 cm, center at 1 cm.
        positions = [(30.0, 200.0)] * 30
        wide = _metrics(_arena_maze(distance_cm=1.0), _make_result(positions))
        narrow = _metrics(_arena_maze(distance_cm=4.0), _make_result(positions))
        self.assertAlmostEqual(wide['center_time_s'], 1.0)
        self.assertAlmostEqual(narrow['periphery_time_s'], 1.0)

    def test_area_fraction_rule_when_chosen(self):
        result = _make_result(self._rim_and_core())
        metrics = _metrics(
            _arena_maze(mode=CENTER_BY_AREA_FRACTION, fraction=0.25), result
        )
        # A quarter of a 400 px square is the central 200 px square.
        self.assertEqual(metrics['center_definition'], CENTER_BY_AREA_FRACTION)
        self.assertAlmostEqual(metrics['center_threshold_px'], 100.0, delta=1.5)
        self.assertAlmostEqual(
            metrics['center_threshold_px'],
            center_threshold_for_area_fraction(ARENA, 0.25),
        )
        self.assertAlmostEqual(metrics['center_time_s'], 1.0)
        self.assertAlmostEqual(metrics['periphery_time_s'], 1.0)

    def test_uncalibrated_distance_rule_falls_back_to_area_fraction(self):
        result = _make_result(self._rim_and_core(), scale_factor=0.0)
        metrics = _metrics(_arena_maze(fraction=0.25), result)
        self.assertEqual(metrics['center_definition'], CENTER_BY_AREA_FRACTION)
        self.assertIsNone(metrics['center_threshold_cm'])
        self.assertEqual(metrics['distance_unit'], 'pixels')
        self.assertTrue(any('area fraction' in n for n in result.notes))
        # Wall distance stays in pixels: min distance of the two blocks.
        self.assertAlmostEqual(metrics['mean_wall_distance'], (200 + 30) / 2)

    def test_calibrated_and_uncalibrated_agree_on_equivalent_thresholds(self):
        positions = self._rim_and_core()
        calibrated = _metrics(
            _arena_maze(distance_cm=4.0), _make_result(positions)
        )
        pixels = _metrics(
            _arena_maze(mode=CENTER_BY_AREA_FRACTION, fraction=0.36),
            _make_result(positions, scale_factor=0.0),
        )
        # 80 px inset leaves (240 / 400)^2 = 36% of the arena.
        self.assertAlmostEqual(pixels['center_threshold_px'], 80.0, delta=1.5)
        self.assertEqual(calibrated['center_time_s'], pixels['center_time_s'])
        self.assertAlmostEqual(
            calibrated['mean_wall_distance'] / SCALE,
            pixels['mean_wall_distance'],
        )

    def test_empty_center_is_reported(self):
        result = _make_result([(200.0, 200.0)] * 30)
        metrics = _metrics(_arena_maze(distance_cm=50.0), result)
        self.assertEqual(metrics['center_time_s'], 0.0)
        self.assertAlmostEqual(metrics['periphery_time_s'], 1.0)
        self.assertTrue(any('center region is empty' in n for n in result.notes))

    def test_invalid_center_parameters_are_refused(self):
        maze = Freestyle()
        for kwargs in (
            {'center_mode': 'diameter'},
            {'distance_cm': 0.0},
            {'distance_cm': float('nan')},
            {'area_fraction': 0.0},
            {'area_fraction': 1.0},
        ):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                maze.configure_arena(True, **kwargs)
        self.assertFalse(maze.arena_mode)


class NoArenaIdentityTests(unittest.TestCase):
    def test_default_maze_has_no_arena_columns_or_parameters(self):
        maze = Freestyle()
        self.assertFalse(maze.arena_mode)
        self.assertEqual(maze.get_batch_summary_headers(), BASELINE_HEADERS)
        self.assertEqual(
            set(maze.get_algorithm_parameters()),
            {'zones', 'zones_configured', 'immobility_threshold_cm_per_s',
             'speed_window_seconds', 'min_immobility_seconds',
             'min_dwell_seconds'},
        )

    def test_zone_headers_are_unchanged_without_arena(self):
        maze = Freestyle()
        maze.configure_zones([('zone_a', 'Zone A'), ('zone_b', 'Zone B')])
        self.assertEqual(
            maze.get_batch_summary_headers(),
            BASELINE_HEADERS + [
                'total_transitions', 'num_zones_visited',
                'zone_a_entries', 'zone_b_entries',
            ],
        )

    def test_arena_polygon_is_ignored_when_arena_mode_is_off(self):
        positions = [(20.0 + 3.0 * i, 20.0) for i in range(90)]
        with_polygon = _make_result(positions)
        without_polygon = _make_result(positions, arena=None)
        maze = Freestyle()
        maze.calculate_metrics(with_polygon)
        maze.calculate_metrics(without_polygon)
        self.assertIsNone(with_polygon._arena_metrics)
        self.assertEqual(
            maze.get_batch_summary_row(with_polygon),
            maze.get_batch_summary_row(without_polygon),
        )
        self.assertEqual(with_polygon.to_dict(), without_polygon.to_dict())
        self.assertNotIn('arena', with_polygon.to_dict()['freestyle_metrics'])
        self.assertEqual(with_polygon.notes, without_polygon.notes)

    def test_summary_json_has_no_arena_keys_without_arena_mode(self):
        result = _make_result([(20.0 + 3.0 * i, 20.0) for i in range(90)])
        Freestyle().calculate_metrics(result)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "summary.json"
            DataExporter.export_summary_json(result, path)
            text = path.read_text(encoding='utf-8')
        self.assertNotIn('arena', text)
        self.assertNotIn('jump_limit', text)

    def test_arena_parameters_do_not_leak_when_mode_is_off(self):
        maze = Freestyle()
        maze.configure_arena(False, CENTER_BY_AREA_FRACTION, 7.0, 0.4)
        self.assertNotIn('arena_mode', maze.get_algorithm_parameters())
        self.assertNotIn(
            'arena_center_mode', maze.get_algorithm_parameters()
        )


class ArenaParameterPersistenceTests(unittest.TestCase):
    def test_parameters_round_trip(self):
        maze = _arena_maze(CENTER_BY_AREA_FRACTION, 6.0, 0.4)
        parameters = json.loads(json.dumps(maze.get_algorithm_parameters()))
        self.assertTrue(parameters['arena_mode'])

        restored = Freestyle()
        restored.configure_algorithm_parameters(parameters)
        self.assertTrue(restored.arena_mode)
        self.assertEqual(restored.arena_center_mode, CENTER_BY_AREA_FRACTION)
        self.assertEqual(restored.arena_center_distance_cm, 6.0)
        self.assertEqual(restored.arena_center_area_fraction, 0.4)
        self.assertEqual(restored.get_algorithm_parameters(), parameters)

    def test_defaults_are_ten_cm_and_a_quarter(self):
        maze = Freestyle()
        self.assertEqual(maze.arena_center_mode, CENTER_BY_DISTANCE)
        self.assertEqual(maze.arena_center_distance_cm, 10.0)
        self.assertEqual(maze.arena_center_area_fraction, 0.25)

    def test_configuration_without_arena_turns_the_mode_off(self):
        maze = _arena_maze()
        plain = Freestyle().get_algorithm_parameters()
        maze.configure_algorithm_parameters(plain)
        self.assertFalse(maze.arena_mode)

    def test_legacy_parameters_without_arena_keys_load(self):
        maze = Freestyle()
        maze.configure_algorithm_parameters({
            'zones': [['a', 'A']], 'zones_configured': True,
            'immobility_threshold_cm_per_s': 3.0,
        })
        self.assertFalse(maze.arena_mode)
        self.assertEqual(maze.user_defined_zones, [('a', 'A')])


class ArenaOutputTests(unittest.TestCase):
    def _result(self, **kwargs):
        positions = _walk((20, 20), (140, 20), 30) + _walk(
            (140, 200), (260, 200), 60
        )
        return _make_result(positions, **kwargs)

    def test_rows_are_rectangular_with_and_without_zones(self):
        for zones in ([], [('zone_a', 'Zone A')]):
            for scale in (SCALE, 0.0):
                with self.subTest(zones=len(zones), scale=scale):
                    maze = _arena_maze()
                    maze.configure_zones(zones)
                    result = self._result(scale_factor=scale)
                    maze.calculate_metrics(result)
                    headers = maze.get_batch_summary_headers()
                    row = maze.get_batch_summary_row(result)
                    self.assertEqual(len(headers), len(row))
                    self.assertEqual(
                        headers[-len(ARENA_METRIC_HEADERS):],
                        ARENA_METRIC_HEADERS,
                    )
                    self.assertTrue(
                        headers[:len(BASELINE_HEADERS)] == BASELINE_HEADERS
                    )

    def test_arena_cells_are_formatted_and_na_when_undefined(self):
        maze = _arena_maze()
        result = self._result()
        maze.calculate_metrics(result)
        cells = dict(zip(
            maze.get_batch_summary_headers(),
            maze.get_batch_summary_row(result),
        ))
        self.assertEqual(cells['center_time_s'], '2.00')
        self.assertEqual(cells['periphery_time_s'], '1.00')
        self.assertEqual(cells['latency_to_center_s'], '1.00')
        self.assertEqual(cells['center_entries'], 1)

        wall_only = _make_result([(20.0 + 3.0 * i, 20.0) for i in range(90)])
        maze.calculate_metrics(wall_only)
        cells = dict(zip(
            maze.get_batch_summary_headers(),
            maze.get_batch_summary_row(wall_only),
        ))
        self.assertEqual(cells['latency_to_center_s'], 'N/A')
        self.assertEqual(cells['center_speed_mean'], 'N/A')

    def test_missing_arena_polygon_gives_na_cells_and_a_note(self):
        maze = _arena_maze()
        result = self._result(arena=None)
        maze.calculate_metrics(result)
        headers = maze.get_batch_summary_headers()
        row = maze.get_batch_summary_row(result)
        self.assertEqual(len(headers), len(row))
        self.assertEqual(row[-len(ARENA_METRIC_HEADERS):],
                         ['N/A'] * len(ARENA_METRIC_HEADERS))
        self.assertTrue(any('no arena' in note for note in result.notes))

    def test_json_summary_and_csv_outputs(self):
        maze = _arena_maze()
        result = self._result()
        maze.calculate_metrics(result)

        data = result.to_dict()['freestyle_metrics']['arena']
        self.assertEqual(data['center_entries'], 1)
        self.assertEqual(data['center_definition'], CENTER_BY_DISTANCE)

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            DataExporter.export_summary_json(result, folder / "s.json")
            saved = json.loads((folder / "s.json").read_text(encoding='utf-8'))
            self.assertEqual(
                saved['freestyle_metrics']['arena']['center_entries'], 1
            )
            self.assertNotIn('_arena_labels', json.dumps(saved))

            DataExporter.export_summary_csv(result, folder / "a.csv", maze)
            with (folder / "a.csv").open(newline='', encoding='utf-8') as f:
                rows = {r[0]: r for r in csv.reader(f)}
            self.assertEqual(rows['center_time_s'][2], '2.00')
            self.assertEqual(rows['center_time_s'][3], 'seconds')
            self.assertEqual(rows['thigmotaxis_distance_pct'][3], 'percent')
            self.assertEqual(rows['thigmotaxis_time_pct'][3], 'percent')
            self.assertEqual(rows['center_entries'][3], 'count')
            self.assertEqual(rows['mean_wall_distance'][3], 'cm')
            self.assertEqual(rows['center_speed_mean'][3], 'cm/s')

            maze.export_specific_data(result, folder, "clip")
            with (folder / "clip_arena_zones.csv").open(
                newline='', encoding='utf-8'
            ) as f:
                table = list(csv.reader(f))
            self.assertEqual(
                table[0],
                ['frame', 'timestamp_s', 'arena_zone',
                 'wall_distance_cm', 'distance_unit'],
            )
            self.assertEqual(len(table) - 1, len(result.positions))
            self.assertEqual(table[1][2], 'periphery')
            self.assertEqual(table[-1][2], 'center')

    def test_no_arena_csv_is_written_without_arena_mode(self):
        maze = Freestyle()
        result = self._result()
        maze.calculate_metrics(result)
        with tempfile.TemporaryDirectory() as folder:
            maze.export_specific_data(result, Path(folder), "clip")
            names = sorted(p.name for p in Path(folder).iterdir())
        self.assertEqual(names, ["clip_speed_data.csv"])

    def test_overlay_is_cached_on_the_roi_manager(self):
        from roi_manager import ROIManager

        manager = ROIManager(source_dimensions=(500, 500))
        manager.set_arena(np.array(ARENA))
        maze = _arena_maze(distance_cm=4.0)
        maze.update_roi_overlay(manager, SCALE)
        outline = manager.center_overlay
        self.assertIsNotNone(outline)
        np.testing.assert_allclose(outline.min(axis=0), [80, 80], atol=1.5)
        np.testing.assert_allclose(outline.max(axis=0), [320, 320], atol=1.5)

        # Uncalibrated distance rule falls back to the area fraction.
        maze.update_roi_overlay(manager, 0.0)
        np.testing.assert_allclose(
            manager.center_overlay.min(axis=0), [100, 100], atol=1.5
        )

        Freestyle().update_roi_overlay(manager, SCALE)
        self.assertIsNone(manager.center_overlay)


if __name__ == "__main__":
    unittest.main()
