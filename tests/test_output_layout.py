import csv
import json
import tempfile
import unittest
from pathlib import Path

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
class OutputLayoutTests(unittest.TestCase):
    def _positional_result(self, result_type, labels):
        count = len(labels)
        return result_type(
            positions=[(10 + index * 2, 10 + index) for index in range(count)],
            timestamps=[index / 10.0 for index in range(count)],
            raw_roi_labels=list(labels),
            roi_labels=list(labels),
            visual_labels=list(labels),
            fps=10.0,
            total_distance=float(count - 1),
            scale_factor=0.5,
            detection_rate=1.0,
            detection_confidences=[0.9] * count,
            tracking_quality={"mean_confidence": 0.9},
            tracking_quality_grade="Excellent",
        )

    def _export_run(self, root, name, maze, result):
        run = root / name
        data = run / "video" / "data"
        plots = run / "video" / "plots"
        data.mkdir(parents=True)
        plots.mkdir(parents=True)
        base = "video"
        is_tst = isinstance(maze, TST)
        frame_name = (
            f"{base}_tst_motion_evidence.csv"
            if is_tst else f"{base}_tracking_data.csv"
        )
        DataExporter.export_to_csv(result, data / frame_name)
        DataExporter.export_summary_csv(
            result, data / f"{base}_analysis_summary.csv", maze
        )
        DataExporter.export_summary_json(
            result,
            data / f"{base}_summary.json",
            run_configuration={"paradigm": maze.name},
        )
        maze.export_specific_data(result, data, base)
        if is_tst:
            maze.generate_specific_plots(result, plots, base)
        else:
            Visualizer.generate_heatmap(
                result.positions, (80, 60), plots / f"{base}_heatmap.png"
            )
            Visualizer.generate_trajectory_plot(
                result,
                (80, 60),
                plots / f"{base}_trajectory.png",
            )
            Visualizer.generate_time_series(
                result, plots / f"{base}_time_series.png"
            )
            maze.generate_specific_plots(result, plots, base)
        DataExporter.export_batch_summary_csv(
            [(Path("video.mp4"), result)], run, maze
        )
        return run

    def test_each_paradigm_uses_the_compact_run_layout(self):
        epm = EPM()
        epm_result = self._positional_result(
            EPMAnalysisResult,
            [
                "open_arm_1", "open_arm_1", "center",
                "closed_arm_1", "closed_arm_1",
            ],
        )
        epm_result.time_in_roi = {
            "open_arm_1": 0.2,
            "center": 0.1,
            "closed_arm_1": 0.2,
        }
        epm_result.distance_in_roi = {
            "open_arm_1": 2.0,
            "center": 1.0,
            "closed_arm_1": 1.0,
        }
        epm.calculate_metrics(epm_result)

        y_maze = YMaze()
        y_result = self._positional_result(
            YMazeAnalysisResult,
            ["arm_a", "center", "arm_b", "center", "arm_c"],
        )
        y_maze.calculate_metrics(y_result)

        freestyle = Freestyle()
        freestyle.configure_zones([
            ("zone_a", "Zone A"),
            ("zone_b", "Zone B"),
        ])
        freestyle_result = self._positional_result(
            FreestyleAnalysisResult,
            ["zone_a", "zone_a", "zone_b", "zone_b", "zone_a"],
        )
        freestyle.calculate_metrics(freestyle_result)

        tst = TST()
        tst.set_parameters(0.1)
        states = [
            "uncertain",
            "immobile_static",
            "immobile_static",
            "passive_swing",
            "active_escape",
        ]
        tst_result = TSTResult(
            timestamps=[index / 10.0 for index in range(len(states))],
            fps=10.0,
            activity_scores={"mouse_1": [0.0, 2.0, 2.0, 8.0, 80.0]},
            motion_evidence={
                "mouse_1": [
                    {
                        "state": state,
                        "confidence": 0.8 if state != "uncertain" else 0.0,
                    }
                    for state in states
                ]
            },
        )
        tst.calculate_metrics(tst_result)

        expected = {
            "epm": {
                "data": {
                    "video_tracking_data.csv",
                    "video_analysis_summary.csv",
                    "video_summary.json",
                },
                "plots": {
                    "video_heatmap.png",
                    "video_trajectory.png",
                    "video_time_series.png",
                },
            },
            "y_maze": {
                "data": {
                    "video_tracking_data.csv",
                    "video_analysis_summary.csv",
                    "video_summary.json",
                    "video_arm_entries.csv",
                },
                "plots": {
                    "video_heatmap.png",
                    "video_trajectory.png",
                    "video_time_series.png",
                    "video_sequence_plot.png",
                    "video_transition_matrix.png",
                },
            },
            "freestyle": {
                "data": {
                    "video_tracking_data.csv",
                    "video_analysis_summary.csv",
                    "video_summary.json",
                    "video_speed_data.csv",
                },
                "plots": {
                    "video_heatmap.png",
                    "video_trajectory.png",
                    "video_time_series.png",
                    "video_speed.png",
                    "video_transition_matrix.png",
                    "video_zone_entries.png",
                },
            },
            "tst": {
                "data": {
                    "video_tst_motion_evidence.csv",
                    "video_analysis_summary.csv",
                    "video_summary.json",
                },
                "plots": {"video_tst_ethogram.png"},
            },
        }
        cases = {
            "epm": (epm, epm_result),
            "y_maze": (y_maze, y_result),
            "freestyle": (freestyle, freestyle_result),
            "tst": (tst, tst_result),
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, (maze, result) in cases.items():
                run = self._export_run(root, name, maze, result)
                self.assertEqual(
                    {path.name for path in (run / "video" / "data").iterdir()},
                    expected[name]["data"],
                )
                self.assertEqual(
                    {
                        path.name
                        for path in (run / "video" / "plots").iterdir()
                    },
                    expected[name]["plots"],
                )
                self.assertFalse((run / "video" / "media").exists())
                self.assertEqual(
                    {
                        path.name
                        for path in (run / "video").iterdir()
                        if path.is_file()
                    },
                    set(),
                )
                self.assertTrue((run / "batch_summary.csv").is_file())
                json.loads(
                    (run / "video" / "data" / "video_summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                for csv_path in (run / "video" / "data").glob("*.csv"):
                    with csv_path.open(
                        newline="", encoding="utf-8"
                    ) as file:
                        rows = list(csv.reader(file))
                    self.assertTrue(rows, csv_path)
                    self.assertEqual(
                        len({len(row) for row in rows}),
                        1,
                        csv_path,
                    )


if __name__ == "__main__":
    unittest.main()
