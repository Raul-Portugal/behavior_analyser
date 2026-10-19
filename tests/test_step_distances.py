"""``compute_step_distances`` must reproduce the old inline distance loop."""
import random
import tempfile
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np

    from core.analysis_engine import Analyzer, compute_step_distances
    from core.detection import DetectionEngine
    from core.models import DetectionConfig
    from mazes.base_maze import BaseAnalysisResult
    from roi_manager import ROIManager
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


def _legacy_finalize_distances(positions, roi_labels, fps, jump_limit):
    """The loop ``Analyzer.finalize_result`` used before the refactor."""
    out = {
        "total": 0.0, "distance_in_roi": {}, "time_in_roi": {},
        "undetected_frames": 0, "undetected_time_s": 0.0,
        "excluded_jump_count": 0, "excluded_jump_distance": 0.0,
    }
    total_dist = 0.0
    prev = None
    prev_idx = None
    for i, pos in enumerate(positions):
        zone = roi_labels[i] if i < len(roi_labels) else 'undetected'
        if pos is None or zone == 'undetected':
            out["undetected_frames"] += 1
            out["undetected_time_s"] += (1.0 / fps)
            prev = None
            prev_idx = None
            continue
        out["time_in_roi"][zone] = out["time_in_roi"].get(zone, 0.0) + (1.0/fps)
        if prev is not None and prev_idx == i - 1:
            dist = np.sqrt((pos[0]-prev[0])**2 + (pos[1]-prev[1])**2)
            if dist <= jump_limit:
                total_dist += dist
                out["distance_in_roi"][zone] = (
                    out["distance_in_roi"].get(zone, 0.0) + dist
                )
            else:
                out["excluded_jump_count"] += 1
                out["excluded_jump_distance"] += float(dist)
        prev = pos
        prev_idx = i
    out["total"] = total_dist
    return out


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class StepDistanceTests(unittest.TestCase):
    def test_none_for_invalid_frames_and_gaps(self):
        steps = compute_step_distances(
            [(0, 0), (3, 4), None, (6, 8), (9, 12), (9, 12)],
            ["a", "a", "a", "a", "undetected", "a"],
            jump_limit=100.0,
        )
        self.assertEqual(steps.steps, [None, 5.0, None, None, None, None])
        self.assertEqual(steps.excluded_jumps, [])

    def test_jumps_are_reported_not_counted(self):
        steps = compute_step_distances(
            [(0, 0), (3, 4), (80, 4), (83, 8)], ["z"] * 4, jump_limit=28.0
        )
        self.assertEqual(steps.steps, [None, 5.0, None, 5.0])
        self.assertEqual(steps.excluded_jumps, [(2, 77.0)])

    def test_no_limit_counts_every_step_and_short_labels_mean_undetected(self):
        steps = compute_step_distances([(0, 0), (300, 400)], ["z", "z"], None)
        self.assertEqual(steps.steps, [None, 500.0])
        short = compute_step_distances([(0, 0), (3, 4)], ["z"], None)
        self.assertEqual(short.steps, [None, None])

    def test_finalize_result_matches_the_old_loop_exactly(self):
        rng = random.Random(7)
        labels_pool = ["arm_a", "arm_b", "center", "outside", "undetected"]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            video = Path(tmp) / "blank.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (100, 100)
            )
            for _ in range(5):
                writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
            writer.release()
            engine = DetectionEngine(
                np.zeros((100, 100), dtype=np.uint8), DetectionConfig()
            )
            analyzer = Analyzer(video, engine, ROIManager((100, 100)))

            for trial in range(40):
                count = rng.randint(1, 160)
                positions, labels = [], []
                x, y = 50.0, 50.0
                for _ in range(count):
                    if rng.random() < 0.08:
                        positions.append(None)
                    else:
                        step = 60.0 if rng.random() < 0.05 else 2.5
                        x += rng.uniform(-step, step)
                        y += rng.uniform(-step, step)
                        positions.append((x, y))
                    labels.append(rng.choice(labels_pool))
                # A label list shorter than the positions is allowed.
                if trial % 5 == 0:
                    labels = labels[:max(0, count - 3)]

                result = BaseAnalysisResult(
                    positions=list(positions), roi_labels=list(labels),
                    detection_confidences=[0.9] * count,
                )
                analyzer.finalize_result(result)
                expected = _legacy_finalize_distances(
                    positions, labels, analyzer.fps, result.jump_limit_px
                )
                self.assertEqual(result.total_distance, expected["total"])
                self.assertEqual(
                    result.distance_in_roi, expected["distance_in_roi"]
                )
                self.assertEqual(
                    list(result.distance_in_roi),
                    list(expected["distance_in_roi"]),
                )
                self.assertEqual(result.time_in_roi, expected["time_in_roi"])
                self.assertEqual(
                    result.undetected_frames, expected["undetected_frames"]
                )
                self.assertEqual(
                    result.undetected_time_s, expected["undetected_time_s"]
                )
                self.assertEqual(
                    result.excluded_jump_count, expected["excluded_jump_count"]
                )
                self.assertEqual(
                    result.excluded_jump_distance,
                    expected["excluded_jump_distance"],
                )


if __name__ == "__main__":
    unittest.main()
