"""Hour-long plots must render quickly and still show every peak and run."""

import os
import statistics
import tempfile
import time
import unittest
from pathlib import Path

try:
    import numpy as np

    from core.plot_utils import decimate_min_max, run_spans, true_runs
    from core.tracking_quality import summarize_tracking_quality
    from mazes.tst import TST, TSTResult
    from output import Visualizer
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False

FPS = 30.0
HOUR_SAMPLES = int(3600 * FPS)
# Shared CI runners are much slower and noisier than a workstation.
TIME_LIMIT_S = 10.0 if os.environ.get("CI") else 5.0


def _hour_tst_result(mice=4):
    rng = np.random.default_rng(8)
    timestamps = (np.arange(HOUR_SAMPLES) / FPS).tolist()
    result = TSTResult(timestamps=timestamps, fps=FPS)
    for index in range(mice):
        name = f"mouse_{index + 1}"
        activity = np.clip(
            50 + 40 * np.sin(np.arange(HOUR_SAMPLES) / 300.0)
            + rng.normal(0, 12, HOUR_SAMPLES), 0, 100,
        )
        result.activity_scores[name] = activity.tolist()
        # Bouts of 1-20 s, like real immobility/mobility alternation.
        state = []
        while len(state) < HOUR_SAMPLES:
            length = int(rng.integers(30, 600))
            state.extend([[True, False, None][int(rng.integers(0, 3))]] * length)
        state = state[:HOUR_SAMPLES]
        result.immobile_states[name] = state
        result.per_roi_times = {name: (60.0, 3500.0)}
    return result


def _hour_positional_result():
    rng = np.random.default_rng(9)
    steps = rng.normal(0, 1.5, (HOUR_SAMPLES, 2)).cumsum(axis=0)
    xy = np.clip(40 + steps, 0, 79)
    labels = np.array(["arm_a", "arm_b", "arm_c", "center"])[
        (np.arange(HOUR_SAMPLES) // 900) % 4
    ].tolist()
    positions = [None if i % 211 == 0 else (float(x), float(y))
                 for i, (x, y) in enumerate(xy)]
    return TSTResult(
        positions=positions,
        timestamps=(np.arange(HOUR_SAMPLES) / FPS).tolist(),
        roi_labels=labels, visual_labels=labels, fps=FPS,
    )


@unittest.skipUnless(HAS_RUNTIME, "Plot runtime is not installed")
class PlotHelperTests(unittest.TestCase):
    def test_decimation_keeps_extremes_and_order_and_bounds_size(self):
        rng = np.random.default_rng(1)
        for count in (4999, 5000, 5001, 12345, 108000, 100003):
            y = rng.normal(0, 1, count)
            y[count // 3] = 50.0
            y[2 * count // 3] = -50.0
            x = np.arange(count, dtype=float)
            dx, dy = decimate_min_max(x, y, 5000)
            self.assertLessEqual(len(dx), 5000, msg=count)
            self.assertEqual(dy.max(), 50.0)
            self.assertEqual(dy.min(), -50.0)
            self.assertTrue(np.all(np.diff(dx) > 0))
            np.testing.assert_array_equal(dy, y[dx.astype(int)])
            self.assertEqual(dx[0] // 1, dx[0])

    def test_short_traces_are_returned_unchanged(self):
        x = np.arange(100.0)
        y = np.arange(100.0) ** 2
        dx, dy = decimate_min_max(x, y, 5000)
        np.testing.assert_array_equal(dx, x)
        np.testing.assert_array_equal(dy, y)

    def test_bucket_envelope_matches_full_trace(self):
        rng = np.random.default_rng(4)
        y = rng.normal(0, 1, 40000)
        x = np.arange(40000.0)
        dx, dy = decimate_min_max(x, y, 2000)
        # Every decimated bucket contains the true min and max of its range.
        edges = np.searchsorted(dx, np.linspace(0, 40000, 11))
        for left, right in zip(edges[:-1], edges[1:]):
            lo_x, hi_x = dx[left], dx[right - 1]
            window = y[int(lo_x):int(hi_x) + 1]
            self.assertAlmostEqual(dy[left:right].max(), window.max())
            self.assertAlmostEqual(dy[left:right].min(), window.min())

    def test_true_runs_edge_cases(self):
        starts, stops = true_runs([])
        self.assertEqual(len(starts), 0)
        starts, stops = true_runs([True, True, False, True, False, False, True])
        self.assertEqual(starts.tolist(), [0, 3, 6])
        self.assertEqual(stops.tolist(), [2, 4, 7])
        starts, stops = true_runs([False, False])
        self.assertEqual(len(starts), 0)
        starts, stops = true_runs([True, True])
        self.assertEqual((starts.tolist(), stops.tolist()), ([0], [2]))

    def test_run_spans_cover_sample_durations(self):
        times = [0.0, 0.5, 1.0, 1.5, 2.0]
        spans = run_spans(times, [True, True, False, False, True])
        self.assertEqual(spans, [(0.0, 1.0), (2.0, 0.5)])
        self.assertEqual(run_spans(times, [False] * 5), [])

    def test_runs_reconstruct_the_flag_vector(self):
        rng = np.random.default_rng(6)
        flags = rng.random(5000) < 0.4
        times = np.arange(5000) * 0.1
        rebuilt = np.zeros(5000, dtype=bool)
        for start, width in run_spans(times, flags):
            first = int(round(start / 0.1))
            rebuilt[first:first + int(round(width / 0.1))] = True
        np.testing.assert_array_equal(rebuilt, flags)


@unittest.skipUnless(HAS_RUNTIME, "Plot runtime is not installed")
class HourLongPlotTests(unittest.TestCase):
    def _timed(self, function):
        started = time.perf_counter()
        function()
        return time.perf_counter() - started

    def test_tst_ethogram_for_one_hour_four_mice(self):
        result = _hour_tst_result()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            elapsed = self._timed(
                lambda: TST().generate_specific_plots(result, Path(tmp), "x")
            )
            output = Path(tmp) / "x_tst_ethogram.png"
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1000)
        self.assertLess(elapsed, TIME_LIMIT_S)

    def test_trajectory_plot_for_one_hour(self):
        result = _hour_positional_result()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            output = Path(tmp) / "t.png"
            elapsed = self._timed(
                lambda: Visualizer.generate_trajectory_plot(
                    result, (80, 80), output
                )
            )
            self.assertGreater(output.stat().st_size, 1000)
        self.assertLess(elapsed, TIME_LIMIT_S)

    def test_trajectory_draws_one_collection_per_label(self):
        from matplotlib.figure import Figure
        captured = []
        original = Figure.savefig

        def spy(figure, *args, **kwargs):
            captured.append(figure)
            return original(figure, *args, **kwargs)

        result = _hour_positional_result()
        result.positions = result.positions[:4000]
        result.roi_labels = result.roi_labels[:4000]
        Figure.savefig = spy
        try:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                Visualizer.generate_trajectory_plot(
                    result, (80, 80), Path(tmp) / "t.png"
                )
        finally:
            Figure.savefig = original
        axis = captured[0].axes[0]
        sizes = sorted(len(c.get_offsets()) for c in axis.collections)
        valid = [p for p in result.positions if p is not None]
        self.assertEqual(len(axis.collections), 4)
        self.assertEqual(sum(sizes), len(valid))


@unittest.skipUnless(HAS_RUNTIME, "Plot runtime is not installed")
class TrackingQualityMeanTests(unittest.TestCase):
    def test_means_equal_exact_statistics_mean(self):
        """summarize_tracking_quality reports the correctly rounded mean."""
        rng = np.random.default_rng(10)
        confidences = rng.uniform(0.1, 0.99, 5000).tolist()
        positions = [
            None if i % 17 == 0 else (float(i % 50), float((i * 7) % 40))
            for i in range(5000)
        ]
        summary = summarize_tracking_quality(
            positions, confidences, frame_diagonal=100.0
        )
        detected = [
            confidences[i] for i, p in enumerate(positions) if p is not None
        ]
        self.assertEqual(
            summary["mean_confidence"], statistics.mean(detected)
        )
        jumps = [
            ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
            for a, b in zip(positions, positions[1:])
            if a is not None and b is not None
        ]
        self.assertAlmostEqual(
            summary["mean_consecutive_jump_px"], statistics.mean(jumps),
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
