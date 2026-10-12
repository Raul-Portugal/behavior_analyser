import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

try:
    from mazes.freestyle import Freestyle, FreestyleAnalysisResult
    from output import DataExporter
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


def _jittery_still_positions(count=300, seed=11):
    rng = random.Random(seed)
    return [
        (200.0 + rng.randint(-1, 1), 150.0 + rng.randint(-1, 1))
        for _ in range(count)
    ]


def _moving_then_paused_positions(fps=30.0):
    """4 s at 200 px/s, a 3 s pause, then 3 s at 200 px/s."""
    positions = []
    x = 0.0
    for frame in range(int(10 * fps)):
        t = frame / fps
        if not 4.0 <= t < 7.0:
            x += 200.0 / fps
        positions.append((x, 50.0))
    return positions


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class FreestyleMetricTests(unittest.TestCase):
    FPS = 30.0

    def _result(self, positions, scale_factor):
        raw = sum(
            ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
            for a, b in zip(positions, positions[1:])
        )
        return FreestyleAnalysisResult(
            positions=positions,
            timestamps=[i / self.FPS for i in range(len(positions))],
            fps=self.FPS,
            total_distance=raw * (scale_factor if scale_factor > 0 else 1.0),
            scale_factor=scale_factor,
        )

    def test_calibrated_jitter_is_immobile_and_not_counted_as_distance(self):
        result = self._result(_jittery_still_positions(), 0.05)

        Freestyle().calculate_metrics(result)

        self.assertGreaterEqual(result.immobility_percentage, 99.0)
        self.assertEqual(result.immobility_bouts, 1)
        self.assertEqual(result.immobility_threshold, 2.0)
        self.assertEqual(result.immobility_threshold_unit, "cm/s")
        self.assertLess(result.average_speed, 2.0)
        self.assertLess(
            result.smoothed_total_distance, result.total_distance * 0.25
        )
        self.assertEqual(result.notes, [])

    def test_calibrated_pause_and_motion_are_separated(self):
        result = self._result(_moving_then_paused_positions(), 0.05)

        Freestyle().calculate_metrics(result)

        self.assertAlmostEqual(result.immobility_time, 3.0, delta=0.5)
        self.assertEqual(result.immobility_bouts, 1)
        self.assertAlmostEqual(result.average_speed, 10.0 * 7 / 10, delta=0.6)
        self.assertAlmostEqual(result.peak_speed, 10.0, delta=0.1)
        self.assertAlmostEqual(
            result.smoothed_total_distance, result.total_distance, delta=0.1
        )
        self.assertAlmostEqual(result.movement_efficiency, 1.0, delta=0.01)
        self.assertIn(True, result.immobile_states)

    def test_uncalibrated_reports_pixel_speeds_and_no_immobility(self):
        result = self._result(_moving_then_paused_positions(), 0.0)

        Freestyle().calculate_metrics(result)

        self.assertEqual(result.distance_unit, "pixels")
        self.assertAlmostEqual(result.peak_speed, 200.0, delta=2.0)
        self.assertIsNone(result.immobility_time)
        self.assertIsNone(result.immobility_percentage)
        self.assertIsNone(result.immobility_bouts)
        self.assertIsNone(result.immobility_threshold)
        self.assertEqual(
            result.immobility_threshold_unit, "N/A (uncalibrated)"
        )
        self.assertTrue(any("calibrat" in note for note in result.notes))

    def test_batch_row_is_rectangular_and_renders_na(self):
        for scale in (0.05, 0.0):
            for zones in ([], [("zone_a", "Zone A"), ("zone_b", "Zone B")]):
                with self.subTest(scale=scale, zones=len(zones)):
                    maze = Freestyle()
                    maze.configure_zones(zones)
                    result = self._result(
                        _moving_then_paused_positions(), scale
                    )
                    maze.calculate_metrics(result)

                    headers = maze.get_batch_summary_headers()
                    row = maze.get_batch_summary_row(result)

                    self.assertEqual(len(headers), len(row))
                    cells = dict(zip(headers, row))
                    self.assertIn("smoothed_total_distance", cells)
                    if scale > 0:
                        self.assertEqual(cells["immobility_threshold"], "2.00")
                        self.assertNotIn("N/A", row)
                    else:
                        for key in (
                            "immobility_threshold", "immobility_time_s",
                            "immobility_pct", "immobility_bouts",
                        ):
                            self.assertEqual(cells[key], "N/A")

    def test_non_freestyle_result_row_has_header_length(self):
        maze = Freestyle()
        from mazes.base_maze import BaseAnalysisResult
        self.assertEqual(
            len(maze.get_batch_summary_row(BaseAnalysisResult())),
            len(maze.get_batch_summary_headers()),
        )

    def test_json_has_no_per_frame_speeds_and_handles_none(self):
        for scale in (0.05, 0.0):
            result = self._result(_moving_then_paused_positions(), scale)
            Freestyle().calculate_metrics(result)

            data = json.loads(json.dumps(result.to_dict()))

            movement = data["freestyle_metrics"]["movement"]
            self.assertNotIn("instantaneous_speeds", data["freestyle_metrics"])
            self.assertNotIn("smoothed_speeds", data["freestyle_metrics"])
            self.assertIn("smoothed_total_distance", movement)
            self.assertIn("immobility_bouts", movement)

    def test_speed_csv_and_plot_for_both_calibration_states(self):
        for scale in (0.05, 0.0):
            with self.subTest(scale=scale), tempfile.TemporaryDirectory() as d:
                maze = Freestyle()
                result = self._result(_moving_then_paused_positions(), scale)
                maze.calculate_metrics(result)

                maze.export_specific_data(result, Path(d), "clip")
                maze.generate_specific_plots(result, Path(d), "clip")

                self.assertTrue((Path(d) / "clip_speed.png").is_file())
                with (Path(d) / "clip_speed_data.csv").open(
                    newline="", encoding="utf-8"
                ) as file:
                    rows = list(csv.DictReader(file))
                unit = "cm" if scale > 0 else "pixels"
                self.assertEqual(len(rows), len(result.positions))
                self.assertIn(f"smoothed_speed_{unit}_per_s", rows[0])
                flags = {row["is_immobile"] for row in rows}
                if scale > 0:
                    self.assertEqual(flags, {"", "0", "1"})
                else:
                    self.assertEqual(flags, {""})

    def test_summary_csv_renders_na_values(self):
        with tempfile.TemporaryDirectory() as d:
            maze = Freestyle()
            result = self._result(_moving_then_paused_positions(), 0.0)
            maze.calculate_metrics(result)
            path = Path(d) / "summary.csv"

            DataExporter.export_summary_csv(result, path, maze)

            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.reader(file))
            metrics = {row[0]: row[2] for row in rows if len(row) > 2}
            self.assertEqual(metrics["immobility_pct"], "N/A")
            self.assertEqual(
                metrics["immobility_threshold_unit"], "N/A (uncalibrated)"
            )


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class FreestyleParameterTests(unittest.TestCase):
    def test_defaults_and_round_trip(self):
        maze = Freestyle()
        self.assertEqual(maze.immobility_threshold_cm_per_s, 2.0)
        self.assertEqual(maze.speed_window_seconds, 0.2)
        self.assertEqual(maze.min_immobility_seconds, 1.0)

        maze.configure_zones([("a", "A")])
        maze.set_immobility_threshold(1.5)
        maze.speed_window_seconds = 0.3
        maze.min_immobility_seconds = 2.0
        maze.min_dwell_seconds = 0.4
        restored = Freestyle()
        restored.configure_algorithm_parameters(
            json.loads(json.dumps(maze.get_algorithm_parameters()))
        )

        self.assertEqual(
            restored.get_algorithm_parameters(),
            maze.get_algorithm_parameters(),
        )
        self.assertEqual(restored.immobility_threshold_cm_per_s, 1.5)
        self.assertEqual(restored.user_defined_zones, [("a", "A")])

    def test_legacy_px_threshold_is_reported_but_not_applied(self):
        maze = Freestyle()
        maze.configure_algorithm_parameters({
            "zones": [],
            "zones_configured": False,
            "immobility_threshold_px_per_s": 7.0,
            "min_dwell_seconds": 0.25,
        })
        positions = _jittery_still_positions()
        calibrated = FreestyleAnalysisResult(
            positions=positions, fps=30.0, scale_factor=0.05,
            total_distance=1.0,
        )
        uncalibrated = FreestyleAnalysisResult(
            positions=positions, fps=30.0, total_distance=1.0,
        )

        maze.calculate_metrics(calibrated)
        maze.calculate_metrics(uncalibrated)

        self.assertEqual(maze.immobility_threshold_cm_per_s, 2.0)
        self.assertEqual(maze.min_dwell_seconds, 0.25)
        self.assertEqual(calibrated.immobility_threshold, 2.0)
        self.assertTrue(any("7 px/s" in note for note in calibrated.notes))
        self.assertIsNone(uncalibrated.immobility_threshold)
        self.assertNotIn(
            "immobility_threshold_px_per_s", maze.get_algorithm_parameters()
        )

    def test_new_threshold_key_takes_precedence_over_legacy(self):
        maze = Freestyle()
        maze.configure_algorithm_parameters({
            "immobility_threshold_px_per_s": 7.0,
            "immobility_threshold_cm_per_s": 3.0,
        })

        self.assertEqual(maze.immobility_threshold_cm_per_s, 3.0)

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            Freestyle().set_immobility_threshold(-1.0)


if __name__ == "__main__":
    unittest.main()
