"""End-to-end check of the real AnalysisWorker on a scripted Y-maze video.

A dark blob walks A -> center -> B -> center -> C -> center -> A with pauses on
a light arena. The test drives the unmodified worker headlessly and checks the
scored arm sequence, the alternation result, the total distance against the
scripted path length and the run-level output files.
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

    from core.models import BatchSettings, DetectionConfig
    from core.video import ReferenceFrameGenerator
    from gui.worker import AnalysisWorker
    from mazes.y_maze import YMaze
    from roi_manager import ROIManager
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False

WIDTH, HEIGHT, FPS = 320, 240, 30.0
STEP_PX = 5  # integer steps keep the drawn blob free of rounding jitter
SCALE_CM_PER_PX = 0.1

# Rectangles are (x0, y0, x1, y1); they tile a plus-shaped arena.
ARM_A = (140, 20, 180, 100)
CENTER = (140, 100, 180, 140)
ARM_B = (40, 100, 140, 140)
ARM_C = (180, 100, 280, 140)

# Waypoints and how many frames the animal rests on each.
SCRIPT = [
    ((160, 60), 15),    # arm A
    ((160, 120), 6),    # center
    ((100, 120), 15),   # arm B
    ((160, 120), 6),    # center
    ((220, 120), 15),   # arm C
    ((160, 120), 6),    # center
    ((160, 60), 15),    # arm A
]


def _rect_polygon(rect):
    x0, y0, x1, y1 = rect
    return np.array([[x0, y0], [x1 - 1, y0], [x1 - 1, y1 - 1], [x0, y1 - 1]])


def _positions():
    """Per-frame blob centers and the scripted path length in pixels."""
    positions, path_length = [], 0.0
    previous = None
    for (x, y), pause in SCRIPT:
        if previous is not None:
            px, py = previous
            distance = max(abs(x - px), abs(y - py))
            steps = distance // STEP_PX
            for step in range(1, steps + 1):
                positions.append((
                    px + (x - px) * step // steps,
                    py + (y - py) * step // steps,
                ))
            path_length += float(np.hypot(x - px, y - py))
        else:
            positions.append((x, y))
        positions.extend([(x, y)] * pause)
        previous = (x, y)
    return positions, path_length


def _write_video(path: Path):
    positions, path_length = _positions()
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT)
    )
    for center in positions:
        frame = np.full((HEIGHT, WIDTH, 3), 120, dtype=np.uint8)
        for rect in (ARM_A, CENTER, ARM_B, ARM_C):
            x0, y0, x1, y1 = rect
            cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), (235, 235, 235), -1)
        cv2.circle(frame, center, 8, (20, 20, 20), -1)
        writer.write(frame)
    writer.release()
    return len(positions), path_length


def _settings() -> "BatchSettings":
    manager = ROIManager(source_dimensions=(WIDTH, HEIGHT))
    for name, rect in (
        ("arm_a", ARM_A), ("arm_b", ARM_B), ("arm_c", ARM_C),
        ("center", CENTER),
    ):
        manager.add_roi(name, _rect_polygon(rect))
    return BatchSettings(
        manager,
        DetectionConfig(
            threshold_percentile=97.0, use_weighting=False,
            min_component_area=20, morphology_kernel_size=3,
            min_detection_confidence=0.1,
        ),
        SCALE_CM_PER_PX,
    )


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class YMazeEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._tmp.name)
        video = cls.root / "ymaze.avi"
        cls.frame_count, cls.path_px = _write_video(video)

        with patch.object(
            ReferenceFrameGenerator, "get_cache_dir",
            return_value=cls.root / "cache",
        ):
            out_dir = cls.root / "out"
            out_dir.mkdir()
            worker = AnalysisWorker({video: _settings()}, out_dir, YMaze())
            outcomes = []
            worker.finished.connect(outcomes.append)
            worker.run()
        cls.outcome = outcomes[0]
        cls.result = cls.outcome.results[0][1] if cls.outcome.results else None

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_run_completes_without_failures(self):
        self.assertEqual(self.outcome.failures, [])
        self.assertFalse(self.outcome.cancelled)
        self.assertIsNotNone(self.result)
        self.assertEqual(len(self.result.positions), self.frame_count)

    def test_arm_sequence_and_alternation(self):
        self.assertEqual(self.result.arm_sequence, "ABCA")
        self.assertEqual(self.result.total_arm_entries, 4)
        self.assertEqual(self.result.possible_alternations, 2)
        self.assertEqual(self.result.correct_alternations, 2)
        self.assertAlmostEqual(self.result.alternation_score, 100.0)
        self.assertEqual(self.result.same_arm_return_count, 0)
        self.assertEqual(self.result.entries_possibly_missed, 0)

    def test_distance_matches_scripted_path_within_five_percent(self):
        expected = self.path_px * SCALE_CM_PER_PX
        self.assertEqual(self.result.distance_unit, "cm")
        self.assertAlmostEqual(
            self.result.total_distance, expected, delta=0.05 * expected
        )
        self.assertEqual(self.result.excluded_jump_count, 0)

    def test_batch_summary_and_manifest_are_written(self):
        run_dir = self.outcome.output_dir
        summary = run_dir / "batch_summary.csv"
        manifest = run_dir / "manifest.json"
        self.assertTrue(summary.is_file())
        self.assertTrue(manifest.is_file())

        with open(summary, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["arm_sequence"], "ABCA")
        self.assertEqual(rows[0]["correct_alternations"], "2")

        data = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(data["failures"], [])


if __name__ == "__main__":
    unittest.main()
