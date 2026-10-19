import math
import tempfile
import unittest
from pathlib import Path

try:
    from core import tst_validation as tv
    from core.agreement_stats import summarize_agreement
    from mazes.tst import TST, TSTResult
    from output import DataExporter
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


FPS = 10.0
START_FRAME = 100
FRAMES = 40


def _evidence(states, start=START_FRAME):
    rows = []
    for offset, state in enumerate(states):
        rows.append({
            "source_frame": start + offset,
            "state": state,
            "confidence": 0.9,
            "rigid_motion": 0.25 + offset * 0.001,
            "pendulum_score": 0.5,
            "temporal_ready": True,
            "camera_stabilized": offset % 2 == 0,
            "qc_flag": "ok" if offset % 3 else "",
        })
    return rows


def _result(per_roi_times=None):
    """Two mice over frames 100-139: one run in the middle is immobile."""
    first = (
        ["active_escape"] * 5 + ["immobile_static"] * 20
        + ["passive_swing"] * 5 + ["active_escape"] * 10
    )
    second = ["active_escape"] * 15 + ["passive_swing"] * 25
    result = TSTResult(
        timestamps=[(START_FRAME + i) / FPS for i in range(FRAMES)],
        fps=FPS,
        activity_scores={
            "mouse_1": [float(i) for i in range(FRAMES)],
            # A name that begins with another mouse's name.
            "mouse_1_b": [1.5] * FRAMES,
        },
        motion_evidence={
            "mouse_1": _evidence(first),
            "mouse_1_b": _evidence(second),
        },
        per_roi_times=per_roi_times,
    )
    TST().calculate_metrics(result)
    return result


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class EvidenceCsvRoundTripTests(unittest.TestCase):
    def _round_trip(self, result):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video_tst_motion_evidence.csv"
            DataExporter.export_to_csv(result, path)
            return tv.load_tst_evidence_csv(path)

    def test_frame_column_becomes_source_frame(self):
        result = _result()
        loaded = self._round_trip(result)
        self.assertEqual(set(loaded), {"mouse_1", "mouse_1_b"})
        for mouse, rows in loaded.items():
            self.assertEqual(len(rows), FRAMES)
            self.assertEqual(
                [row["source_frame"] for row in rows],
                [item["source_frame"] for item in result.motion_evidence[mouse]],
            )
            self.assertEqual(
                [row["state"] for row in rows],
                [item["state"] for item in result.motion_evidence[mouse]],
            )

    def test_features_flags_and_immobility_survive(self):
        result = _result()
        loaded = self._round_trip(result)
        row = loaded["mouse_1"][7]
        source = result.motion_evidence["mouse_1"][7]
        self.assertAlmostEqual(row["rigid_motion"], source["rigid_motion"], places=6)
        self.assertAlmostEqual(row["confidence"], 0.9, places=4)
        self.assertEqual(row["temporal_ready"], True)
        self.assertEqual(row["camera_stabilized"], False)
        self.assertEqual(loaded["mouse_1"][8]["camera_stabilized"], True)
        self.assertEqual(row["qc_flag"], source["qc_flag"])
        self.assertAlmostEqual(row["timestamp_s"], 10.7)
        self.assertEqual(
            [item["is_immobile"] for item in loaded["mouse_1"]],
            result.immobile_states["mouse_1"],
        )
        self.assertAlmostEqual(loaded["mouse_1"][3]["active_evidence"], 3.0)

    def test_evaluate_annotations_gives_the_same_report_as_in_memory(self):
        result = _result()
        loaded = self._round_trip(result)
        annotations = {
            ("mouse_1", START_FRAME + i): state
            for i, state in enumerate(
                ["active_escape"] * 4 + ["immobile_static"] * 22
                + ["active_escape"] * 14)
        }
        in_memory = tv.evaluate_annotations(result.motion_evidence, annotations)
        from_csv = tv.evaluate_annotations(loaded, annotations)
        self.assertEqual(in_memory, from_csv)
        self.assertEqual(from_csv["mouse_1"]["annotated_frames"], 40.0)

    def test_loaded_evidence_without_the_frame_column_would_misalign(self):
        # The point of source_frame: annotations use absolute frames, not row
        # positions, so a run that starts at frame 100 still lines up.
        loaded = self._round_trip(_result())
        annotations = {("mouse_1", START_FRAME): "active_escape"}
        report = tv.evaluate_annotations(loaded, annotations)
        self.assertEqual(report["mouse_1"]["annotated_frames"], 1.0)

    def test_window_outside_scored_range_is_none(self):
        result = _result(per_roi_times={"mouse_1": (11.0, 13.0)})
        loaded = self._round_trip(result)
        flags = [row["is_immobile"] for row in loaded["mouse_1"]]
        self.assertIsNone(flags[0])
        self.assertIsNotNone(flags[10])
        self.assertIsNone(flags[-1])

    def test_rejects_other_csv_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracking.csv"
            path.write_text(
                "frame,timestamp,x,y,zone\n1,0.0,1,2,center\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "motion evidence"):
                tv.load_tst_evidence_csv(path)
            path.write_text("timestamp_s,mouse_1_x\n0.1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                tv.load_tst_evidence_csv(path)

    def test_bad_cell_names_the_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.csv"
            path.write_text(
                "frame,timestamp_s,mouse_1_behavior_state\n"
                "10,0.5,active_escape\nten,0.6,active_escape\n",
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 3"):
                tv.load_tst_evidence_csv(path)


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class ImmobilityTimeTests(unittest.TestCase):
    def test_auto_seconds_match_the_reported_total_after_round_trip(self):
        result = _result(per_roi_times={"mouse_1": (10.5, 13.5)})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "e.csv"
            DataExporter.export_to_csv(result, path)
            loaded = tv.load_tst_evidence_csv(path)
        for mouse in loaded:
            self.assertAlmostEqual(
                tv.auto_immobility_seconds(loaded[mouse]),
                result.total_immobility_time[mouse], places=6)
            self.assertAlmostEqual(
                tv.auto_immobility_seconds(loaded[mouse], fps=FPS),
                result.total_immobility_time[mouse], places=6)
        self.assertGreater(result.total_immobility_time["mouse_1"], 0.0)

    def test_per_mouse_agreement_with_frame_annotations(self):
        evidence = {"m": [
            {"source_frame": 10 + i, "state": state, "timestamp_s": (10 + i) / 10}
            for i, state in enumerate(
                ["immobile_static", "immobile_static", "active_escape",
                 "passive_swing", "active_escape", "immobile_static"])
        ]}
        manual = dict(zip(
            [("m", 10 + i) for i in range(6)],
            ["immobile_static", "active_escape", "active_escape",
             "passive_swing", "uncertain", "passive_swing"]))
        report = tv.immobility_time_agreement(evidence, manual)["m"]
        # Frame 14 is uncertain and excluded: 5 frames at 10 fps.
        self.assertEqual(report.annotated_frames, 5)
        self.assertAlmostEqual(report.fps, 10.0)
        self.assertAlmostEqual(report.auto_s, 0.4)     # frames 10, 11, 13, 15
        self.assertAlmostEqual(report.manual_s, 0.3)   # frames 10, 13, 15
        self.assertAlmostEqual(report.difference_s, 0.1)

    def test_difference_is_auto_minus_manual_and_unannotated_mice_are_omitted(self):
        evidence = {
            "a": [{"source_frame": i, "state": "immobile_static"}
                  for i in range(4)],
            "b": [{"source_frame": i, "state": "active_escape"}
                  for i in range(4)],
        }
        manual = {("a", i): "active_escape" for i in range(4)}
        report = tv.immobility_time_agreement(evidence, manual, fps=2.0)
        self.assertEqual(set(report), {"a"})
        self.assertAlmostEqual(report["a"].difference_s, 2.0)
        self.assertEqual(tv.agreement_pairs(report), [(2.0, 0.0)])

    def test_smoothed_flag_is_preferred_and_none_is_outside_the_window(self):
        evidence = {"m": [
            {"source_frame": 0, "state": "immobile_static", "is_immobile": False},
            {"source_frame": 1, "state": "immobile_static", "is_immobile": True},
            {"source_frame": 2, "state": "immobile_static", "is_immobile": None},
        ]}
        manual = {("m", i): "immobile_static" for i in range(3)}
        smoothed = tv.immobility_time_agreement(evidence, manual, fps=1.0)["m"]
        self.assertEqual(smoothed.annotated_frames, 2)
        self.assertEqual(smoothed.auto_s, 1.0)
        raw = tv.immobility_time_agreement(
            evidence, manual, fps=1.0, smoothed=False)["m"]
        self.assertEqual(raw.annotated_frames, 3)
        self.assertEqual(raw.auto_s, 3.0)

    def test_fps_is_required_when_it_cannot_be_inferred(self):
        evidence = {"m": [{"source_frame": 1, "state": "immobile_static"}]}
        with self.assertRaises(ValueError):
            tv.immobility_time_agreement(evidence, {("m", 1): "active_escape"})
        with self.assertRaises(ValueError):
            tv.auto_immobility_seconds(evidence["m"])

    def test_totals_pairs_feed_summarize_agreement(self):
        auto = {f"v{i}/mouse_1": 60.0 + 10 * i for i in range(8)}
        manual = {f"v{i}/mouse_1": 58.0 + 10 * i + (i % 3) for i in range(8)}
        manual["v99/mouse_1"] = 5.0   # no automated counterpart
        pairs = tv.immobility_time_pairs(auto, manual)
        self.assertEqual(len(pairs), 8)
        summary = summarize_agreement(pairs)
        self.assertEqual(summary.n, 8)
        self.assertGreater(summary.icc.icc, 0.99)
        self.assertTrue(math.isfinite(summary.bland_altman.bias))


if __name__ == "__main__":
    unittest.main()
