"""Time bins: the pure computation, the CSVs, and the run options."""
import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import cv2
    import numpy as np

    from core import cli
    from core.analysis_engine import Analyzer, apply_scale_to_result
    from core.detection import DetectionEngine
    from core.models import BatchSettings, DetectionConfig
    from core.sample_sheet import load_sample_sheet, metadata_for_videos
    from core.time_bins import TIME_BIN_HEADERS, compute_time_bins
    from gui.worker import AnalysisWorker
    from mazes.epm import EPM, EPMAnalysisResult
    from mazes.freestyle import Freestyle, FreestyleAnalysisResult
    from mazes.tst import TST, TSTResult
    from mazes.y_maze import YMaze, YMazeAnalysisResult
    from roi_manager import ROIManager
    from tests import worker_support as support
    from tests.worker_support import FRAME_COUNTS, data_files
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False

FPS = 10.0
FREESTYLE = "Freestyle / Open Field"


def _segments(*parts):
    labels = []
    for label, frames in parts:
        labels.extend([label] * frames)
    return labels


def _write_blank_video(path: Path, frames: int):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (100, 100)
    )
    for _ in range(frames):
        writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class BinFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()

    def finalize(self, result, scale=0.0):
        """Run the real ``finalize_result`` (10 fps) on ``result``."""
        video = self.root / "blank.avi"
        _write_blank_video(video, len(result.positions))
        engine = DetectionEngine(
            np.zeros((100, 100), dtype=np.uint8), DetectionConfig()
        )
        Analyzer(video, engine, ROIManager((100, 100))).finalize_result(result)
        if scale > 0:
            apply_scale_to_result(result, scale)
        return result

    @staticmethod
    def positions(frames, jump_at=None):
        """One pixel per frame, with one oversized jump (excluded) if asked."""
        return [
            (float(i + (40 if jump_at is not None and i >= jump_at else 0)),
             10.0)
            for i in range(frames)
        ]


class PositionalTotalsMixin:
    """Every positional metric summed over the bins equals the video total."""

    def assert_totals(self, table, result):
        distance = table.metric("distance_total")
        self.assertAlmostEqual(distance.total(), result.total_distance, places=6)
        self.assertEqual(distance.unit, result.distance_unit)
        self.assertAlmostEqual(
            table.metric("undetected_time_s").total(),
            result.undetected_time_s, places=6,
        )
        for zone, seconds in result.time_in_roi.items():
            self.assertAlmostEqual(
                table.metric(f"time_in_{zone}_s").total(), seconds, places=6,
                msg=zone,
            )
            self.assertAlmostEqual(
                table.metric(f"distance_in_{zone}").total(),
                result.distance_in_roi.get(zone, 0.0), places=6, msg=zone,
            )


class YMazeBinTests(BinFixture, PositionalTotalsMixin):
    def _result(self, shift=0.0, scale=0.0):
        labels = _segments(
            ("arm_a", 10), ("center", 5), ("arm_b", 15), ("center", 5),
            ("arm_c", 5), ("undetected", 3), ("arm_c", 12), ("center", 10),
            ("arm_a", 25), ("center", 2), ("arm_c", 8),
        )
        positions = self.positions(len(labels), jump_at=70)
        for index, label in enumerate(labels):
            if label == "undetected":
                positions[index] = None
        result = YMazeAnalysisResult(
            positions=positions,
            timestamps=[shift + i / FPS for i in range(len(labels))],
            roi_labels=labels,
            raw_roi_labels=list(labels),
            detection_confidences=[0.9] * len(labels),
        )
        self.finalize(result, scale)
        YMaze().calculate_metrics(result)
        return result

    def test_bins_sum_to_the_video_totals(self):
        result = self._result()
        table = compute_time_bins(result, 3.0)

        self.assertEqual(len(table.bins), 4)
        self.assert_totals(table, result)
        self.assertEqual(
            table.metric("entries_total").total(), result.total_arm_entries
        )
        self.assertEqual(
            table.metric("alternations_correct").total(),
            result.correct_alternations,
        )
        self.assertEqual(
            table.metric("alternations_possible").total(),
            result.possible_alternations,
        )
        for arm in ("arm_a", "arm_b", "arm_c"):
            self.assertEqual(
                table.metric(f"entries_{arm}").total(),
                sum(1 for name, _ in result.arm_entries if name == arm),
            )
        # The jump was excluded from the distance and so from the bins.
        self.assertEqual(result.excluded_jump_count, 1)

    def test_a_triplet_is_counted_in_the_bin_of_its_third_entry(self):
        result = self._result()
        self.assertEqual(result.arm_sequence, "ABCAC")
        table = compute_time_bins(result, 3.0)

        # Entries at 0.0 and 1.5 s (bin 1), 3.5 s (bin 2), 6.5 s (bin 3),
        # 9.2 s (bin 4). A-B-C straddles bins 1/2 and is scored in bin 2.
        self.assertEqual(table.metric("entries_total").values, [2, 1, 1, 1])
        self.assertEqual(
            table.metric("alternations_possible").values, [0, 1, 1, 1]
        )
        self.assertEqual(
            table.metric("alternations_correct").values, [0, 1, 1, 0]
        )
        self.assertEqual(
            table.metric("alternation_pct").values, [None, 100.0, 100.0, 0.0]
        )

    def test_last_bin_is_partial_and_bins_are_relative_to_the_start(self):
        result = self._result(shift=5.0)
        table = compute_time_bins(result, 3.0)

        self.assertEqual([b.partial for b in table.bins],
                         [False, False, False, True])
        last = table.bins[-1]
        self.assertAlmostEqual(last.start_s, 9.0)
        self.assertAlmostEqual(last.end_s, 10.0)
        self.assertAlmostEqual(last.duration_s, 1.0)
        self.assertAlmostEqual(last.abs_start_s, 14.0)
        self.assertAlmostEqual(last.abs_end_s, 15.0)
        self.assertEqual(last.frame_count, 10)
        first = table.bins[0]
        self.assertAlmostEqual(first.start_s, 0.0)
        self.assertAlmostEqual(first.abs_start_s, 5.0)
        self.assertAlmostEqual(first.duration_s, 3.0)
        # Entries are still binned from the analysis start.
        self.assertEqual(table.metric("entries_total").values, [2, 1, 1, 1])

    def test_a_bin_length_that_divides_the_video_has_no_partial_bin(self):
        table = compute_time_bins(self._result(), 5.0)
        self.assertEqual([b.partial for b in table.bins], [False, False])
        table = compute_time_bins(self._result(), 10.0)
        self.assertEqual([b.partial for b in table.bins], [False])
        table = compute_time_bins(self._result(), 60.0)
        self.assertEqual(len(table.bins), 1)
        self.assertTrue(table.bins[0].partial)
        self.assertAlmostEqual(table.bins[0].duration_s, 10.0)

    def test_calibrated_distances_are_scaled(self):
        result = self._result(scale=0.5)
        table = compute_time_bins(result, 3.0)
        self.assertEqual(result.distance_unit, "cm")
        self.assert_totals(table, result)
        self.assertEqual(table.metric("distance_total").unit, "cm")

    def test_off_or_empty_gives_no_table(self):
        result = self._result()
        self.assertIsNone(compute_time_bins(result, 0))
        self.assertIsNone(compute_time_bins(result, -1))
        self.assertIsNone(compute_time_bins(YMazeAnalysisResult(), 3.0))

    def test_rows_follow_the_documented_columns(self):
        table = compute_time_bins(self._result(), 3.0)
        rows = list(table.rows())
        self.assertTrue(rows)
        self.assertTrue(all(len(row) == len(TIME_BIN_HEADERS) for row in rows))
        self.assertEqual(
            len(rows), len(table.bins) * len(table.metrics)
        )
        pct = [r for r in rows if r[TIME_BIN_HEADERS.index("metric")]
               == "alternation_pct"]
        self.assertEqual(pct[0][TIME_BIN_HEADERS.index("value")], "")


class EPMBinTests(BinFixture, PositionalTotalsMixin):
    def test_entries_per_arm_type_sum_to_the_totals(self):
        labels = _segments(
            ("center", 5), ("open_arm_1", 20), ("center", 5),
            ("closed_arm_1", 20), ("center", 5), ("open_arm_2", 20),
            ("center", 5), ("closed_arm_2", 20),
        )
        result = EPMAnalysisResult(
            positions=self.positions(len(labels)),
            timestamps=[i / FPS for i in range(len(labels))],
            roi_labels=labels, raw_roi_labels=list(labels),
            detection_confidences=[0.9] * len(labels),
        )
        self.finalize(result)
        EPM().calculate_metrics(result)

        table = compute_time_bins(result, 4.0)
        self.assert_totals(table, result)
        self.assertEqual(
            table.metric("entries_open_arms").total(),
            result.entries_to_open_arms,
        )
        self.assertEqual(
            table.metric("entries_closed_arms").total(),
            result.entries_to_closed_arms,
        )
        self.assertEqual(
            table.metric("entries_total").total(), result.total_arm_entries
        )
        self.assertEqual(table.metric("entries_total").values, [2, 1, 1])
        self.assertIsNone(table.metric("alternation_pct"))


class FreestyleBinTests(BinFixture, PositionalTotalsMixin):
    def _result(self):
        labels = _segments(
            ("z1", 50), ("outside", 10), ("z2", 20), ("undetected", 5),
            ("z2", 5), ("z1", 10),
        )
        positions = [(20.0, 50.0)] * 40 + [
            (20.0 + 1.5 * i, 50.0) for i in range(len(labels) - 40)
        ]
        for index, label in enumerate(labels):
            if label == "undetected":
                positions[index] = None
        result = FreestyleAnalysisResult(
            positions=positions,
            timestamps=[i / FPS for i in range(len(labels))],
            roi_labels=labels, raw_roi_labels=list(labels),
            detection_confidences=[0.9] * len(labels),
        )
        self.finalize(result, scale=0.5)
        maze = Freestyle()
        maze.configure_zones([("z1", "Z1"), ("z2", "Z2")])
        maze.calculate_metrics(result)
        return result

    def test_zone_entries_and_immobility_sum_to_the_totals(self):
        result = self._result()
        table = compute_time_bins(result, 3.0)

        self.assert_totals(table, result)
        self.assertGreater(result.immobility_time, 0.0)
        self.assertAlmostEqual(
            table.metric("immobility_time_s").total(), result.immobility_time,
            places=6,
        )
        self.assertEqual(
            table.metric("entries_total").total(),
            sum(result.zone_entry_counts.values()),
        )
        for zone, count in result.zone_entry_counts.items():
            self.assertEqual(table.metric(f"entries_{zone}").total(), count)
        # Entries are in the bins where they happen: z1 first seen at 0 s.
        self.assertEqual(table.metric("entries_z1").values[0], 1)
        self.assertIsNone(table.metric("center_time_s"))

    def test_uncalibrated_runs_have_no_immobility_rows(self):
        labels = _segments(("z1", 30), ("z2", 30))
        result = FreestyleAnalysisResult(
            positions=self.positions(len(labels)),
            timestamps=[i / FPS for i in range(len(labels))],
            roi_labels=labels, raw_roi_labels=list(labels),
            detection_confidences=[0.9] * len(labels),
        )
        self.finalize(result)
        maze = Freestyle()
        maze.configure_zones([("z1", "Z1"), ("z2", "Z2")])
        maze.calculate_metrics(result)

        table = compute_time_bins(result, 2.0)
        self.assertIsNone(table.metric("immobility_time_s"))
        self.assertEqual(table.metric("entries_total").total(), 2)

    def test_arena_center_time_and_thigmotaxis_per_bin(self):
        frames = 100
        # Toward the top wall: the center for ~9 s, then the periphery.
        positions = [(50.0, 50.0 - 0.45 * i) for i in range(frames)]
        result = FreestyleAnalysisResult(
            positions=positions,
            timestamps=[i / FPS for i in range(frames)],
            roi_labels=["outside"] * frames,
            raw_roi_labels=["outside"] * frames,
            detection_confidences=[0.9] * frames,
        )
        self.finalize(result, scale=0.5)
        result.arena_polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
        maze = Freestyle()
        maze.configure_arena(True, "distance_cm", 5.0, 0.25)
        maze.calculate_metrics(result)
        metrics = result._arena_metrics
        self.assertGreater(metrics["center_time_s"], 0.0)
        self.assertGreater(metrics["periphery_time_s"], 0.0)

        table = compute_time_bins(result, 3.0)
        self.assertAlmostEqual(
            table.metric("center_time_s").total(), metrics["center_time_s"],
            places=6,
        )
        self.assertAlmostEqual(
            table.metric("periphery_time_s").total(),
            metrics["periphery_time_s"], places=6,
        )
        thigmotaxis = table.metric("thigmotaxis_time_pct").values
        self.assertEqual(thigmotaxis[0], 0.0)
        self.assertGreater(thigmotaxis[-1], 0.0)
        self.assertTrue(table.bins[-1].partial)


class TSTBinTests(unittest.TestCase):
    def _result(self):
        states = (
            ["immobile_static"] * 30 + ["active_escape"] * 10
            + ["uncertain"] * 5 + ["immobile_static"] * 35
            + ["active_escape"] * 20
        )
        evidence = [{"state": s, "confidence": 0.9} for s in states]
        result = TSTResult(
            fps=FPS,
            timestamps=[i / FPS for i in range(len(states))],
            per_roi_times={"mouse_1": (1.0, 9.0), "mouse_2": (0.0, 20.0)},
            motion_evidence={
                "mouse_1": list(evidence), "mouse_2": list(evidence),
            },
            activity_scores={"mouse_1": [0.0] * 100, "mouse_2": [0.0] * 100},
        )
        TST().calculate_metrics(result)
        return result

    def test_state_seconds_per_mouse_sum_to_the_window_totals(self):
        result = self._result()
        table = compute_time_bins(result, 3.0)

        self.assertEqual(len(table.bins), 4)
        self.assertTrue(table.bins[-1].partial)
        for mouse in ("mouse_1", "mouse_2"):
            self.assertAlmostEqual(
                table.metric("immobile_s", mouse).total(),
                result.total_immobility_time[mouse], places=6,
            )
            self.assertAlmostEqual(
                table.metric("mobile_s", mouse).total(),
                result.mobile_time[mouse], places=6,
            )
            self.assertAlmostEqual(
                table.metric("uncertain_s", mouse).total(),
                result.uncertain_time[mouse], places=6,
            )
        self.assertGreater(result.uncertain_time["mouse_1"], 0.0)

    def test_frames_outside_the_mouse_window_are_not_binned(self):
        result = self._result()
        table = compute_time_bins(result, 3.0)
        # mouse_1 is scored from 1.0 s to 9.0 s: the first bin has 2 s of it,
        # and the partial last bin (9-10 s) none.
        scored = [
            sum(table.metric(name, "mouse_1").values[i]
                for name in ("immobile_s", "mobile_s", "uncertain_s"))
            for i in range(4)
        ]
        self.assertEqual([round(v, 6) for v in scored], [2.0, 3.0, 3.0, 0.0])
        self.assertIsNone(table.metric("immobility_pct", "mouse_1").values[3])
        self.assertEqual(table.metric("distance_total"), None)


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class BinRunTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        self.video_dir = self.root / "videos"
        self.video_dir.mkdir()
        self.videos = [
            support.write_tracking_video(self.video_dir, frames, name)
            for name, frames in FRAME_COUNTS.items()
        ]

    def _environment(self, label):
        cache = str(self.root / f"cache_{label}")
        return {"LOCALAPPDATA": cache, "XDG_CACHE_HOME": cache}

    def _worker_run(self, label, workers=1, **options):
        out = self.root / label
        out.mkdir()
        plan = {}
        for video in self.videos:
            settings = support.tracking_settings(create_timelapse=False)
            settings.paradigm_id = FREESTYLE
            settings.algorithm_parameters = Freestyle().get_algorithm_parameters()
            plan[video.resolve()] = settings
        with patch.dict(os.environ, self._environment(label)):
            worker = AnalysisWorker(
                plan, out, Freestyle(), workers=workers, **options
            )
            outcomes = []
            worker.finished.connect(outcomes.append)
            worker.run()
        self.assertEqual(outcomes[0].failures, [])
        return worker.output_dir

    @staticmethod
    def _read(path):
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_bins_off_changes_nothing(self):
        default = self._worker_run("default")
        off = self._worker_run("off", bin_seconds=0.0)
        self.assertEqual(data_files(default), data_files(off))
        self.assertFalse(list(default.rglob("*time_bins*")))

    def test_per_video_and_batch_bin_files(self):
        run_dir = self._worker_run("bins", bin_seconds=1.0)

        batch = self._read(run_dir / "batch_time_bins.csv")
        self.assertEqual(
            list(batch[0]), ["video_filename", "paradigm"] + TIME_BIN_HEADERS
        )
        per_video_rows = 0
        summary = {
            row["video_filename"]: row
            for row in self._read(run_dir / "batch_summary.csv")
        }
        for video in self.videos:
            path = run_dir / video.stem / "data" / f"{video.stem}_time_bins.csv"
            rows = self._read(path)
            self.assertEqual(list(rows[0]), TIME_BIN_HEADERS)
            per_video_rows += len(rows)
            total = sum(
                float(row["value"]) for row in rows
                if row["metric"] == "distance_total"
            )
            self.assertAlmostEqual(
                total, float(summary[video.name]["total_distance"]), delta=0.05
            )
            partial = {row["bin_index"]: row["partial_bin"] for row in rows}
            self.assertEqual(partial["1"], "0")
            # 30 fps: only a video that is not a whole number of seconds long
            # ends in a partial bin (mouse_b is exactly 3 s).
            self.assertEqual(
                max(partial.values()),
                "1" if FRAME_COUNTS[video.name] % 30 else "0",
            )
        self.assertEqual(len(batch), per_video_rows)
        self.assertEqual(
            [row["video_filename"] for row in batch][:1], ["mouse_a.avi"]
        )

    def test_batch_bin_file_carries_the_metadata_columns(self):
        sheet_path = self.root / "sheet.csv"
        sheet_path.write_text(
            "video,animal_id,group\n"
            + "".join(
                f"videos/{v.name},A{n},g{n}\n"
                for n, v in enumerate(self.videos, 1)
            ),
            encoding="utf-8",
        )
        sheet = load_sample_sheet(sheet_path)
        metadata = metadata_for_videos(
            sheet, sheet_path, [v.resolve() for v in self.videos]
        ).metadata
        run_dir = self._worker_run(
            "meta", bin_seconds=2.0, sample_metadata=metadata
        )
        batch = self._read(run_dir / "batch_time_bins.csv")
        self.assertEqual(
            list(batch[0]),
            ["video_filename", "paradigm", "animal_id", "group"]
            + TIME_BIN_HEADERS,
        )
        by_video = {row["video_filename"]: row["animal_id"] for row in batch}
        self.assertEqual(
            by_video,
            {"mouse_a.avi": "A1", "mouse_b.avi": "A2", "mouse_c.avi": "A3"},
        )
        # The per-video file has no metadata columns.
        per_video = self._read(
            run_dir / "mouse_a" / "data" / "mouse_a_time_bins.csv"
        )
        self.assertEqual(list(per_video[0]), TIME_BIN_HEADERS)

    def test_parallel_and_sequential_runs_write_identical_bin_files(self):
        sequential = self._worker_run("sequential", workers=1, bin_seconds=1.0)
        parallel = self._worker_run("parallel", workers=3, bin_seconds=1.0)
        expected = data_files(sequential)
        actual = data_files(parallel)
        self.assertEqual(sorted(actual), sorted(expected))
        bin_files = [name for name in expected if "time_bins" in name]
        self.assertEqual(len(bin_files), len(self.videos) + 1)
        for name in expected:
            self.assertEqual(actual[name], expected[name], name)

    def test_bin_seconds_is_not_part_of_the_checkpoint_hash(self):
        from checkpoint_manager import CheckpointManager
        manager = CheckpointManager(self.root / "ckpt", create_dir=False)
        (self.root / "o").mkdir()
        hashes = []
        for bin_seconds in (0.0, 60.0):
            settings = support.tracking_settings()
            AnalysisWorker(
                {self.videos[0].resolve(): settings}, self.root / "o",
                Freestyle(), bin_seconds=bin_seconds,
            )
            hashes.append(manager._get_settings_hash(settings))
            self.assertNotIn("bin", json.dumps(settings.to_dict()).lower())
        self.assertEqual(hashes[0], hashes[1])

    def _cli(self, label, *extra):
        out = self.root / label
        plan_file = self.root / "plan.json"
        if not plan_file.exists():
            settings = support.tracking_settings(create_timelapse=False)
            settings.paradigm_id = FREESTYLE
            settings.algorithm_parameters = Freestyle().get_algorithm_parameters()
            BatchSettings.save_batch_plan(
                {f"videos/{v.name}": settings.copy() for v in self.videos},
                plan_file,
            )
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, self._environment(label)), \
                contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(stderr):
            code = cli.main([
                "run", "--plan", str(plan_file), "--out", str(out),
                "--quiet", "--workers", "1", *extra,
            ])
        return code, out, stderr.getvalue()

    def test_cli_bin_seconds_flag(self):
        code, out, stderr = self._cli("cli", "--bin-seconds", "1")
        self.assertEqual(code, cli.EXIT_OK, stderr)
        (run_dir,) = sorted(out.glob("analysis_run_*"))
        self.assertTrue((run_dir / "batch_time_bins.csv").exists())
        self.assertTrue(
            (run_dir / "mouse_a" / "data" / "mouse_a_time_bins.csv").exists()
        )

    def test_cli_rejects_a_negative_bin_length(self):
        code, out, stderr = self._cli("cli_bad", "--bin-seconds", "-5")
        self.assertEqual(code, cli.EXIT_BAD_INPUT)
        self.assertIn("--bin-seconds", stderr)
        self.assertFalse(out.exists())


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class TSTBinRunTests(unittest.TestCase):
    def test_tst_run_writes_bins_per_mouse(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp).resolve()
            video = support.write_tst_video(root)
            out = root / "out"
            out.mkdir()
            worker = AnalysisWorker(
                {video: support.tst_settings()}, out, TST(), bin_seconds=1.0
            )
            outcomes = []
            worker.finished.connect(outcomes.append)
            with patch.dict(os.environ, {
                "LOCALAPPDATA": str(root / "c"), "XDG_CACHE_HOME": str(root / "c"),
            }):
                worker.run()
            self.assertEqual(outcomes[0].failures, [])
            run_dir = worker.output_dir
            path = run_dir / "tst" / "data" / "tst_time_bins.csv"
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual({row["subject"] for row in rows}, {"mouse_1"})
            result = outcomes[0].results[0][1]
            for metric, total in (
                ("immobile_s", result.total_immobility_time["mouse_1"]),
                ("mobile_s", result.mobile_time["mouse_1"]),
                ("uncertain_s", result.uncertain_time["mouse_1"]),
            ):
                value = sum(
                    float(r["value"]) for r in rows if r["metric"] == metric
                )
                self.assertAlmostEqual(value, total, delta=0.01, msg=metric)
            self.assertTrue((run_dir / "batch_time_bins.csv").exists())


if __name__ == "__main__":
    unittest.main()
