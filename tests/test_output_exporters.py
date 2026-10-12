import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from mazes.epm import EPM, EPMAnalysisResult
    from mazes.freestyle import Freestyle, FreestyleAnalysisResult
    from mazes.tst import TST, TSTResult
    from mazes.y_maze import YMaze, YMazeAnalysisResult
    from output import DataExporter, Visualizer
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class OutputExporterTests(unittest.TestCase):
    def _assert_rectangular_csv(self, path):
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.reader(file))
        self.assertTrue(rows)
        self.assertEqual(len({len(row) for row in rows}), 1)

    def test_positional_summaries_are_rectangular_and_json_is_valid(self):
        result = EPMAnalysisResult(
            positions=[(0, 0), (1, 0)],
            timestamps=[0.0, 0.1],
            fps=10.0,
            total_distance=1.0,
            detection_rate=1.0,
            tracking_quality_grade="Excellent",
            tracking_quality={"mean_confidence": 0.9},
            roi_labels=["center", "open_arm_1"],
        )
        result.time_in_roi = {"center": 0.1, "open_arm_1": 0.1}
        result.distance_in_roi = {"open_arm_1": 1.0}
        maze = EPM()
        maze.calculate_metrics(result)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            DataExporter.export_summary_csv(
                result, root / "summary.csv", maze
            )
            DataExporter.export_summary_json(
                result, root / "summary.json"
            )
            DataExporter.export_batch_summary_csv(
                [(Path("epm.mp4"), result)], root, maze
            )

            self._assert_rectangular_csv(root / "summary.csv")
            self._assert_rectangular_csv(root / "batch_summary.csv")
            json.loads((root / "summary.json").read_text(encoding="utf-8"))

    def test_tst_frame_and_summary_tables_are_rectangular(self):
        result = TSTResult(
            timestamps=[0.1],
            fps=10.0,
            activity_scores={"mouse_1": [25.0]},
            motion_evidence={
                "mouse_1": [{
                    "state": "uncertain",
                    "confidence": 0.25,
                    "pendulum_score": 0.3,
                }]
            },
            immobile_states={"mouse_1": [None]},
            total_immobility_time={"mouse_1": 0.0},
            immobility_percentage={"mouse_1": 0.0},
            valid_time={"mouse_1": 0.0},
            uncertain_time={"mouse_1": 0.1},
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            DataExporter.export_to_csv(result, root / "frames.csv")
            DataExporter.export_summary_csv(
                result, root / "summary.csv", TST()
            )

            self._assert_rectangular_csv(root / "frames.csv")
            self._assert_rectangular_csv(root / "summary.csv")

    def test_all_positional_paradigms_have_complete_batch_contract(self):
        cases = []

        y_result = YMazeAnalysisResult(
            timestamps=[0.0, 0.1, 0.2],
            fps=10.0,
            total_distance=2.0,
            scale_factor=0.5,
            detection_rate=1.0,
            tracking_quality_grade="Excellent",
            tracking_quality={"mean_confidence": 0.9},
            roi_labels=["arm_a", "center", "arm_b"],
        )
        YMaze().calculate_metrics(y_result)
        cases.append(("y_maze", YMaze(), y_result))

        freestyle = Freestyle()
        freestyle.configure_zones([("zone_a", "Zone A")])
        freestyle_result = FreestyleAnalysisResult(
            positions=[(0, 0), (1, 0), (2, 0)],
            timestamps=[0.0, 0.1, 0.2],
            fps=10.0,
            total_distance=2.0,
            scale_factor=0.5,
            detection_rate=1.0,
            tracking_quality_grade="Excellent",
            tracking_quality={"mean_confidence": 0.9},
            roi_labels=["zone_a", "zone_a", "zone_a"],
        )
        freestyle.calculate_metrics(freestyle_result)
        cases.append(("freestyle", freestyle, freestyle_result))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, maze, result in cases:
                output_dir = root / name
                DataExporter.export_batch_summary_csv(
                    [(Path(f"{name}.mp4"), result)],
                    output_dir,
                    maze,
                )
                DataExporter.export_summary_csv(
                    result,
                    output_dir / "summary.csv",
                    maze,
                )
                self._assert_rectangular_csv(
                    output_dir / "batch_summary.csv"
                )
                self._assert_rectangular_csv(output_dir / "summary.csv")

            with (
                root / "freestyle" / "batch_summary.csv"
            ).open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["speed_unit"], "cm/s")
            self.assertEqual(
                rows[0]["immobility_threshold_unit"], "cm/s"
            )
            self.assertEqual(rows[0]["total_distance"], "2.00")


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class HeatmapRoundingTests(unittest.TestCase):
    def _heatmap_array(self, positions, dimensions):
        import numpy as np
        from matplotlib.axes import Axes

        captured = {}
        original = Axes.imshow

        def spy(axes, data, *args, **kwargs):
            captured["data"] = np.array(data)
            return original(axes, data, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(Axes, "imshow", spy):
                Visualizer.generate_heatmap(
                    positions,
                    dimensions,
                    Path(temp_dir) / "heatmap.png",
                    blur_sigma=0,
                )
        return captured["data"]

    def test_sub_pixel_positions_are_rounded_not_truncated(self):
        data = self._heatmap_array([(9.6, 4.6)], (20, 10))

        self.assertEqual(data[5, 10], 1)
        self.assertEqual(data.sum(), 1)

    def test_positions_rounding_past_the_edge_are_dropped_not_raised(self):
        data = self._heatmap_array([(19.7, 3.0), (5.0, 5.0)], (20, 10))

        self.assertEqual(data.sum(), 1)
        self.assertEqual(data[5, 5], 1)

if __name__ == "__main__":
    unittest.main()
