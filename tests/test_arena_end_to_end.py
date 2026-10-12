"""End-to-end open-field arena run through the real worker on a scripted video.

A dark animal walks along the top wall of a square arena, crosses the center
and returns to a wall. For the first frames a larger dark blob appears just
outside the arena (a distractor). With an arena drawn, detection ignores the
distractor and the center/periphery metrics follow the scripted path; without
an arena the run is the plain Freestyle analysis.
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import cv2
    import numpy as np

    from core.arena_zones import classify_center_periphery
    from core.models import BatchSettings, DetectionConfig
    from core.video import ReferenceFrameGenerator
    from gui.worker import AnalysisWorker
    from mazes.freestyle import Freestyle
    from roi_manager import ROIManager
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False

WIDTH, HEIGHT, FPS = 320, 240, 30.0
SCALE_CM_PER_PX = 0.05
STEP_PX = 5
ARENA = [[60, 20], [260, 20], [260, 220], [60, 220]]
CENTER_DISTANCE_CM = 2.0                       # 40 px
DISTRACTOR_CENTER, DISTRACTOR_RADIUS = (40, 60), 14
DISTRACTOR_FRAMES = 40

# (x, y) waypoints and the frames the animal rests on each.
SCRIPT = [
    ((80, 35), 15),     # near the top wall (15 px from it)
    ((240, 35), 15),    # along the wall
    ((160, 120), 45),   # center
    ((80, 205), 30),    # bottom-left wall region
]


def _positions():
    positions = []
    previous = None
    for (x, y), pause in SCRIPT:
        if previous is not None:
            px, py = previous
            steps = max(1, int(np.ceil(max(abs(x - px), abs(y - py)) / STEP_PX)))
            for step in range(1, steps + 1):
                positions.append((
                    int(round(px + (x - px) * step / steps)),
                    int(round(py + (y - py) * step / steps)),
                ))
        else:
            positions.append((x, y))
        positions.extend([(x, y)] * pause)
        previous = (x, y)
    return positions


def _write_video(path: Path):
    positions = _positions()
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT)
    )
    for index, center in enumerate(positions):
        frame = np.full((HEIGHT, WIDTH, 3), 200, dtype=np.uint8)
        cv2.circle(frame, center, 8, (20, 20, 20), -1)
        if index < DISTRACTOR_FRAMES:
            cv2.circle(
                frame, DISTRACTOR_CENTER, DISTRACTOR_RADIUS, (10, 10, 10), -1
            )
        writer.write(frame)
    writer.release()
    return positions


def _settings(with_arena: bool) -> "BatchSettings":
    manager = ROIManager(source_dimensions=(WIDTH, HEIGHT))
    if with_arena:
        manager.set_arena(np.array(ARENA))
    return BatchSettings(
        manager,
        DetectionConfig(
            threshold_percentile=97.0, use_weighting=False,
            min_component_area=20, morphology_kernel_size=3,
            min_detection_confidence=0.1,
        ),
        SCALE_CM_PER_PX,
    )


def _run(video: Path, root: Path, name: str, settings, maze):
    out_dir = root / name
    out_dir.mkdir()
    worker = AnalysisWorker({video: settings}, out_dir, maze)
    outcomes = []
    worker.finished.connect(outcomes.append)
    worker.run()
    return outcomes[0]


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class ArenaEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._tmp.name)
        video = cls.root / "openfield.avi"
        cls.script = _write_video(video)

        with patch.object(
            ReferenceFrameGenerator, "get_cache_dir",
            return_value=cls.root / "cache",
        ):
            arena_maze = Freestyle()
            arena_maze.configure_zones([])
            arena_maze.configure_arena(True, "distance_cm", CENTER_DISTANCE_CM)
            cls.arena_outcome = _run(
                video, cls.root, "arena", _settings(True), arena_maze
            )
            plain_maze = Freestyle()
            plain_maze.configure_zones([])
            cls.plain_outcome = _run(
                video, cls.root, "plain", _settings(False), plain_maze
            )
        cls.arena_result = (
            cls.arena_outcome.results[0][1]
            if cls.arena_outcome.results else None
        )
        cls.plain_result = (
            cls.plain_outcome.results[0][1]
            if cls.plain_outcome.results else None
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_both_runs_complete(self):
        for outcome in (self.arena_outcome, self.plain_outcome):
            self.assertEqual(outcome.failures, [])
            self.assertFalse(outcome.cancelled)
        self.assertEqual(len(self.arena_result.positions), len(self.script))

    def test_distractor_is_tracked_without_but_not_with_the_arena(self):
        def left_of_arena(result):
            return sum(
                1 for p in result.positions if p is not None and p[0] < 52
            )
        self.assertGreater(left_of_arena(self.plain_result), 0)
        self.assertEqual(left_of_arena(self.arena_result), 0)

    def test_arena_polygon_reaches_the_result(self):
        self.assertEqual(self.arena_result.arena_polygon, ARENA)
        self.assertIsNone(self.plain_result.arena_polygon)
        self.assertGreater(self.arena_result.jump_limit_px, 0)

    def test_metrics_follow_the_scripted_path(self):
        metrics = self.arena_result._arena_metrics
        threshold_px = CENTER_DISTANCE_CM / SCALE_CM_PER_PX
        expected = classify_center_periphery(self.script, ARENA, threshold_px)
        expected_center = expected.count("center") / FPS
        expected_periphery = expected.count("periphery") / FPS

        self.assertAlmostEqual(metrics["center_time_s"], expected_center, delta=0.5)
        self.assertAlmostEqual(
            metrics["periphery_time_s"], expected_periphery, delta=0.5
        )
        self.assertEqual(metrics["center_entries"], 1)
        self.assertEqual(metrics["center_definition"], "distance_cm")
        self.assertAlmostEqual(metrics["center_threshold_px"], threshold_px)
        # Starts on the wall, so the center is reached after the wall walk.
        first_center = expected.index("center") / FPS
        self.assertAlmostEqual(
            metrics["latency_to_center_s"], first_center, delta=0.5
        )
        self.assertGreater(metrics["thigmotaxis_time_pct"], 30.0)
        self.assertLess(metrics["thigmotaxis_time_pct"], 90.0)
        self.assertGreater(metrics["mean_wall_distance"], 0.0)

    def test_batch_summary_has_arena_columns_only_for_the_arena_run(self):
        def rows(outcome):
            with open(
                outcome.output_dir / "batch_summary.csv",
                newline="", encoding="utf-8",
            ) as handle:
                return list(csv.DictReader(handle))

        arena_row = rows(self.arena_outcome)[0]
        plain_row = rows(self.plain_outcome)[0]
        self.assertEqual(arena_row["center_entries"], "1")
        self.assertNotEqual(arena_row["thigmotaxis_time_pct"], "N/A")
        self.assertNotIn("center_entries", plain_row)
        self.assertNotIn("thigmotaxis_time_pct", plain_row)

    def test_summary_json_arena_section_and_files(self):
        folder = self.arena_outcome.output_dir / "openfield"
        data = json.loads(
            (folder / "data" / "openfield_summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            data["freestyle_metrics"]["arena"]["center_entries"], 1
        )
        self.assertTrue((folder / "data" / "openfield_arena_zones.csv").is_file())

        plain = self.plain_outcome.output_dir / "openfield" / "data"
        plain_data = json.loads(
            (plain / "openfield_summary.json").read_text(encoding="utf-8")
        )
        self.assertNotIn("arena", plain_data["freestyle_metrics"])
        self.assertNotIn("arena", json.dumps(plain_data["freestyle_metrics"]))
        self.assertFalse((plain / "openfield_arena_zones.csv").exists())

    def test_run_plan_keeps_the_arena(self):
        plan = json.loads(
            (self.arena_outcome.output_dir / "run_plan.json").read_text(
                encoding="utf-8"
            )
        )
        entry = next(iter(plan["plan"].values()))
        self.assertEqual(entry["roi_manager"]["arena"]["points"], ARENA)
        self.assertTrue(entry["algorithm_parameters"]["arena_mode"])
        plain_plan = json.loads(
            (self.plain_outcome.output_dir / "run_plan.json").read_text(
                encoding="utf-8"
            )
        )
        plain_entry = next(iter(plain_plan["plan"].values()))
        self.assertNotIn("arena", plain_entry["roi_manager"])
        self.assertNotIn("arena_mode", plain_entry["algorithm_parameters"])
        self.assertEqual(plain_entry["roi_manager"]["schema_version"], 2)


if __name__ == "__main__":
    unittest.main()
