import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from core import positional_validation as pv
    from mazes.epm import EPM, EPMAnalysisResult
    from mazes.y_maze import YMaze, YMazeAnalysisResult
    from output import DataExporter
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


def _write(directory, name, text):
    path = Path(directory) / name
    path.write_text(text, encoding="utf-8")
    return path


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class ArmNameTests(unittest.TestCase):
    def test_y_maze_names_are_normalized(self):
        for name in ("arm_a", "A", "a", " Arm A ", "ARM-A", "arm a", "arma"):
            self.assertEqual(pv.normalize_y_maze_arm(name), "arm_a", name)
        self.assertEqual(pv.normalize_y_maze_arm("c"), "arm_c")

    def test_epm_names_are_normalized(self):
        self.assertEqual(pv.normalize_epm_arm("open_arm_1"), "open_arm_1")
        self.assertEqual(pv.normalize_epm_arm("Closed Arm 2"), "closed_arm_2")
        self.assertEqual(pv.normalize_epm_arm("OA1"), "open_arm_1")
        self.assertEqual(pv.normalize_epm_arm("ca2"), "closed_arm_2")
        self.assertEqual(pv.normalize_epm_arm("open"), "open")
        self.assertEqual(pv.normalize_epm_arm("Closed"), "closed")
        self.assertEqual(pv.normalize_epm_arm("open-arm"), "open")
        self.assertEqual(pv.epm_arm_class("closed_arm_2"), "closed")
        self.assertEqual(pv.epm_arm_class("Open"), "open")

    def test_unknown_names_raise(self):
        with self.assertRaises(ValueError):
            pv.normalize_y_maze_arm("arm_d")
        with self.assertRaises(ValueError):
            pv.normalize_epm_arm("center")
        with self.assertRaises(ValueError):
            pv.normalize_arm("open", "y_maze")
        with self.assertRaises(ValueError):
            pv.normalize_arm("a", "no_such_maze")

    def test_auto_detection_tries_y_maze_then_epm(self):
        self.assertEqual(pv.normalize_arm("B"), "arm_b")
        self.assertEqual(pv.normalize_arm("open"), "open")


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class AnnotationLoadingTests(unittest.TestCase):
    def test_loads_app_format_and_flexible_arm_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write(
                directory, "manual.csv",
                "entry_number,arm,timestamp_s\n1,A,1.5\n2,arm_b,3.25\n3,c,7\n",
            )
            entries = pv.load_entry_annotations(path, "y_maze")
        self.assertEqual(
            entries, [("arm_a", 1.5), ("arm_b", 3.25), ("arm_c", 7.0)])

    def test_loads_epm_names_and_sorts_by_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write(
                directory, "epm.csv",
                "entry_number,arm,timestamp_s\n"
                "2,open,9.0\n1,closed_arm_2,2.0\n3,Open Arm 1,12.0\n",
            )
            entries = pv.load_entry_annotations(path)
        self.assertEqual(
            entries,
            [("closed_arm_2", 2.0), ("open", 9.0), ("open_arm_1", 12.0)])

    def test_accepts_bom_blank_lines_missing_entry_number_and_time_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manual.csv"
            path.write_bytes(
                b"\xef\xbb\xbfArm,Time_S\r\nA,1.0\r\n\r\nB,2.0\r\n")
            entries = pv.load_entry_annotations(path)
        self.assertEqual(entries, [("arm_a", 1.0), ("arm_b", 2.0)])

    def test_round_trips_the_apps_own_arm_entries_file(self):
        result = YMazeAnalysisResult(
            positions=[(0, 0)] * 6, timestamps=[0, .1, .2, .3, .4, .5],
            fps=10.0,
            roi_labels=["arm_a", "center", "arm_b", "center", "arm_c", "arm_c"],
        )
        maze = YMaze()
        maze.calculate_metrics(result)
        with tempfile.TemporaryDirectory() as directory:
            maze.export_specific_data(result, Path(directory), "video")
            loaded = pv.load_entry_annotations(
                Path(directory) / "video_arm_entries.csv", "y_maze")
        self.assertEqual(
            [arm for arm, _ in loaded], [arm for arm, _ in result.arm_entries])
        for (_, got), (_, want) in zip(loaded, result.arm_entries):
            self.assertAlmostEqual(got, want, places=3)

    def test_bad_files_raise_with_context(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = {
                "no_arm.csv": "entry_number,time\n1,2\n",
                "bad_time.csv": "arm,timestamp_s\nA,soon\n",
                "negative.csv": "arm,timestamp_s\nA,-1\n",
                "bad_arm.csv": "arm,timestamp_s\nQ,1\n",
            }
            for name, text in cases.items():
                path = _write(directory, name, text)
                with self.assertRaises(ValueError, msg=name):
                    pv.load_entry_annotations(path, "y_maze")
            path = _write(directory, "bad_time.csv", cases["bad_time.csv"])
            with self.assertRaisesRegex(ValueError, "line 2"):
                pv.load_entry_annotations(path)

    def test_empty_file_gives_no_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write(directory, "empty.csv", "arm,timestamp_s\n")
            self.assertEqual(pv.load_entry_annotations(path), [])


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class EventMatchingTests(unittest.TestCase):
    def test_perfect_match_with_timing_offset(self):
        auto = [("arm_a", 1.1), ("arm_b", 3.1), ("arm_c", 5.1)]
        manual = [("arm_a", 1.0), ("arm_b", 3.0), ("arm_c", 5.0)]
        result = pv.match_events(auto, manual, 0.5)
        self.assertEqual(
            (result.true_positives, result.false_positives,
             result.false_negatives), (3, 0, 0))
        self.assertEqual((result.precision, result.recall, result.f1),
                         (1.0, 1.0, 1.0))
        self.assertAlmostEqual(result.mean_offset_s, 0.1)
        self.assertAlmostEqual(result.mean_abs_offset_s, 0.1)

    def test_missed_and_spurious_entries(self):
        auto = [("arm_a", 1.0), ("arm_b", 3.0), ("arm_a", 9.0)]
        manual = [("arm_a", 1.0), ("arm_b", 3.0), ("arm_c", 6.0)]
        result = pv.match_events(auto, manual, 0.5)
        self.assertEqual(
            (result.true_positives, result.false_positives,
             result.false_negatives), (2, 1, 1))
        self.assertAlmostEqual(result.precision, 2 / 3)
        self.assertAlmostEqual(result.recall, 2 / 3)
        self.assertAlmostEqual(result.f1, 2 / 3)
        self.assertEqual(result.unmatched_auto, [2])
        self.assertEqual(result.unmatched_manual, [2])

    def test_arm_must_match(self):
        result = pv.match_events([("arm_a", 1.0)], [("arm_b", 1.0)], 5.0)
        self.assertEqual(result.true_positives, 0)
        self.assertEqual(result.false_positives, 1)
        self.assertEqual(result.false_negatives, 1)

    def test_tolerance_boundary_is_inclusive(self):
        auto, manual = [("arm_a", 1.3)], [("arm_a", 0.3)]
        self.assertEqual(pv.match_events(auto, manual, 1.0).true_positives, 1)
        self.assertEqual(pv.match_events(auto, manual, 0.999).true_positives, 0)
        self.assertEqual(pv.match_events(auto, manual, 0.0).true_positives, 0)
        same = pv.match_events([("arm_a", 2.0)], [("arm_a", 2.0)], 0.0)
        self.assertEqual(same.true_positives, 1)
        with self.assertRaises(ValueError):
            pv.match_events(auto, manual, -0.1)

    def test_matching_is_one_to_one_and_closest_pair_wins(self):
        # One manual entry, two automated candidates: the closer one matches.
        result = pv.match_events(
            [("arm_a", 1.4), ("arm_a", 1.1)], [("arm_a", 1.0)], 0.5)
        self.assertEqual(result.matches[0][:2], (1, 0))
        self.assertEqual(result.false_positives, 1)
        # Two manual entries competing for the same automated one.
        result = pv.match_events(
            [("arm_a", 1.0)], [("arm_a", 0.7), ("arm_a", 1.2)], 0.5)
        self.assertEqual(result.matches[0][:2], (0, 1))
        self.assertEqual(result.false_negatives, 1)

    def test_greedy_choice_does_not_steal_a_better_partner(self):
        # a1 is close to m1 only; a2 is close to both. Closest-first gives
        # a1-m1 (0.1) and a2-m2 (0.2) instead of a2-m1.
        auto = [("arm_a", 1.0), ("arm_a", 1.4)]
        manual = [("arm_a", 1.1), ("arm_a", 1.6)]
        result = pv.match_events(auto, manual, 0.5)
        self.assertEqual(result.true_positives, 2)

    def test_empty_inputs_leave_metrics_undefined_not_crashing(self):
        both = pv.match_events([], [], 1.0)
        self.assertTrue(math.isnan(both.precision))
        self.assertTrue(math.isnan(both.recall))
        self.assertTrue(math.isnan(both.f1))
        no_auto = pv.match_events([], [("arm_a", 1.0)], 1.0)
        self.assertTrue(math.isnan(no_auto.precision))
        self.assertEqual(no_auto.recall, 0.0)
        self.assertEqual(no_auto.f1, 0.0)
        no_manual = pv.match_events([("arm_a", 1.0)], [], 1.0)
        self.assertEqual(no_manual.precision, 0.0)
        self.assertTrue(math.isnan(no_manual.recall))
        self.assertTrue(math.isnan(no_manual.mean_offset_s))

    def test_arm_key_allows_type_level_epm_matching(self):
        auto = [("open_arm_1", 2.0), ("closed_arm_2", 5.0)]
        manual = [("open", 2.1), ("closed", 5.2)]
        strict = pv.match_events(auto, manual, 0.5)
        self.assertEqual(strict.true_positives, 0)
        loose = pv.match_events(auto, manual, 0.5, arm_key=pv.epm_arm_class)
        self.assertEqual(loose.true_positives, 2)

    def test_signed_offset_is_auto_minus_manual(self):
        result = pv.match_events([("arm_a", 0.8)], [("arm_a", 1.0)], 0.5)
        self.assertAlmostEqual(result.mean_offset_s, -0.2)


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class SequenceTests(unittest.TestCase):
    def test_edit_distance_known_values(self):
        self.assertEqual(pv.sequence_edit_distance("kitten", "sitting").distance, 3)
        self.assertEqual(pv.sequence_edit_distance("", "abc").distance, 3)
        self.assertEqual(pv.sequence_edit_distance("abc", "abc").distance, 0)
        self.assertEqual(pv.sequence_edit_distance(["A", "B"], ["B", "A"]).distance, 2)

    def test_similarity_is_normalized(self):
        result = pv.sequence_edit_distance("ABCA", "ABCB")
        self.assertEqual(result.distance, 1)
        self.assertAlmostEqual(result.similarity, 0.75)
        self.assertEqual((result.length_a, result.length_b), (4, 4))
        self.assertEqual(pv.sequence_edit_distance("", "").similarity, 1.0)
        self.assertEqual(pv.sequence_edit_distance("", "AB").similarity, 0.0)
        self.assertAlmostEqual(
            pv.sequence_edit_distance("AB", "ABC").similarity, 2 / 3)


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class YMazeComparisonTests(unittest.TestCase):
    def test_identical_scoring_agrees_completely(self):
        entries = [("arm_a", 1.0), ("arm_b", 2.0), ("arm_c", 3.0),
                   ("arm_a", 4.0)]
        result = pv.compare_y_maze(entries, entries)
        self.assertEqual(result.auto_sequence, "ABCA")
        self.assertEqual(result.edit_distance.distance, 0)
        self.assertEqual(result.total_entries_difference, 0)
        self.assertEqual(result.alternation_difference_pct, 0.0)
        self.assertEqual(result.matching.f1, 1.0)

    def test_known_errors_are_quantified(self):
        manual = [("A", 1.0), ("B", 2.0), ("C", 3.0), ("A", 4.0), ("B", 5.0)]
        # Auto misses the C entry at 3 s and adds a spurious B at 8 s.
        auto = [("arm_a", 1.05), ("arm_b", 2.0), ("arm_a", 4.0),
                ("arm_b", 5.1), ("arm_b", 8.0)]
        result = pv.compare_y_maze(auto, manual)
        self.assertEqual(result.manual_sequence, "ABCAB")
        self.assertEqual(result.auto_sequence, "ABABB")
        # ABCAB -> ABABB: substitute C->A, A->B: 2 edits
        self.assertEqual(result.edit_distance.distance, 2)
        self.assertEqual(result.total_entries_difference, 0)
        # manual: ABC, BCA, CAB all alternate = 100 %; auto: ABA, BAB, ABB = 0 %
        self.assertAlmostEqual(result.manual_alternation_pct, 100.0)
        self.assertAlmostEqual(result.auto_alternation_pct, 0.0)
        self.assertAlmostEqual(result.alternation_difference_pct, -100.0)
        self.assertEqual(result.matching.true_positives, 4)
        self.assertEqual(result.matching.false_positives, 1)
        self.assertEqual(result.matching.false_negatives, 1)

    def test_total_entry_difference_is_auto_minus_manual(self):
        manual = [("A", 1.0), ("B", 2.0)]
        auto = [("A", 1.0), ("B", 2.0), ("C", 3.0)]
        result = pv.compare_y_maze(auto, manual)
        self.assertEqual(result.total_entries_difference, 1)
        self.assertTrue(math.isnan(result.manual_alternation_pct))
        self.assertAlmostEqual(result.auto_alternation_pct, 100.0)
        self.assertTrue(math.isnan(result.alternation_difference_pct))

    def test_alternation_matches_the_analysis_rule(self):
        # Drive the real analysis and compare with the validation score.
        labels = ["arm_a", "center", "arm_b", "center", "arm_c", "center",
                  "arm_c", "center", "arm_a"]
        result = YMazeAnalysisResult(
            positions=[(0, 0)] * len(labels),
            timestamps=[i * 0.5 for i in range(len(labels))],
            fps=2.0, roi_labels=labels,
        )
        YMaze().calculate_metrics(result)
        comparison = pv.compare_y_maze(result.arm_entries, result.arm_entries)
        self.assertAlmostEqual(
            comparison.auto_alternation_pct, result.alternation_score)

    def test_empty_inputs(self):
        result = pv.compare_y_maze([], [])
        self.assertEqual(result.total_entries_difference, 0)
        self.assertEqual(result.edit_distance.similarity, 1.0)
        self.assertTrue(math.isnan(result.auto_alternation_pct))


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class EPMComparisonTests(unittest.TestCase):
    def test_differences_from_result_and_manual_totals(self):
        result = EPMAnalysisResult(
            positions=[(0, 0)] * 6,
            timestamps=[i * 0.5 for i in range(6)], fps=2.0,
            roi_labels=["center", "open_arm_1", "center", "closed_arm_1",
                        "center", "open_arm_2"],
        )
        result.time_in_roi = {"open_arm_1": 6.0, "open_arm_2": 4.0,
                              "closed_arm_1": 20.0}
        EPM().calculate_metrics(result)
        manual = {"entries_to_open_arms": 3, "entries_to_closed_arms": 1,
                  "time_in_open_arms_s": 12.0}
        comparison = pv.compare_epm(result, manual)
        self.assertEqual(comparison["entries_to_open_arms"].auto, 2.0)
        self.assertEqual(comparison["entries_to_open_arms"].difference, -1.0)
        self.assertEqual(comparison["entries_to_closed_arms"].difference, 0.0)
        self.assertEqual(comparison["total_arm_entries"].manual, 4.0)
        self.assertEqual(comparison["total_arm_entries"].difference, -1.0)
        self.assertAlmostEqual(comparison["time_in_open_arms_s"].difference, -2.0)
        self.assertAlmostEqual(
            comparison["time_in_open_arms_s"].pct_difference, -2.0 / 12.0 * 100)
        self.assertAlmostEqual(
            comparison["percent_entries_to_open"].difference, 2 / 3 * 100 - 75.0)
        # Not provided by the manual scorer: not compared.
        self.assertNotIn("time_in_closed_arms_s", comparison)

    def test_mapping_inputs_and_zero_manual_value(self):
        comparison = pv.compare_epm(
            {"time_in_open_arms_s": 5.0}, {"time_in_open_arms_s": 0.0})
        item = comparison["time_in_open_arms_s"]
        self.assertEqual(item.abs_difference, 5.0)
        self.assertTrue(math.isnan(item.pct_difference))

    def test_manual_totals_from_entries(self):
        totals = pv.epm_totals_from_entries(
            [("open", 1.0), ("closed_arm_1", 2.0), ("open_arm_2", 3.0)])
        self.assertEqual(totals["entries_to_open_arms"], 2.0)
        self.assertEqual(totals["entries_to_closed_arms"], 1.0)
        self.assertEqual(totals["total_arm_entries"], 3.0)
        self.assertAlmostEqual(totals["percent_entries_to_open"], 200 / 3)
        self.assertEqual(pv.epm_totals_from_entries([])["total_arm_entries"], 0.0)


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class FrameLabelTests(unittest.TestCase):
    def test_kappa_and_percent_agreement(self):
        auto = ["a", "a", "b", "b", "c", "c"]
        manual = ["a", "a", "b", "c", "c", "c"]
        result = pv.frame_label_agreement(auto, manual)
        self.assertEqual(result.n_frames, 6)
        self.assertEqual(result.n_ignored, 0)
        self.assertAlmostEqual(result.percent_agreement, 5 / 6 * 100)
        # po = 5/6; pe = (2*2 + 2*1 + 2*3) / 36 = 12/36
        self.assertAlmostEqual(result.kappa.kappa, (5 / 6 - 1 / 3) / (1 - 1 / 3))

    def test_undetected_and_missing_frames_are_ignored(self):
        auto = ["a", "undetected", "b", "a", "b"]
        manual = ["a", "a", "undetected", None, "b"]
        result = pv.frame_label_agreement(auto, manual)
        self.assertEqual(result.n_frames, 2)
        self.assertEqual(result.n_ignored, 3)
        self.assertEqual(result.percent_agreement, 100.0)

    def test_custom_ignore_set_and_length_check(self):
        result = pv.frame_label_agreement(
            ["a", "outside"], ["a", "b"], ignore={"outside"})
        self.assertEqual(result.n_frames, 1)
        with self.assertRaises(ValueError):
            pv.frame_label_agreement(["a"], ["a", "b"])

    def test_all_frames_ignored(self):
        result = pv.frame_label_agreement(["undetected"], ["a"])
        self.assertEqual(result.n_frames, 0)
        self.assertTrue(math.isnan(result.kappa.kappa))
        self.assertTrue(math.isnan(result.percent_agreement))


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class PositionErrorTests(unittest.TestCase):
    def test_errors_misses_and_false_detections(self):
        auto = [(0, 0), (3, 4), None, (10, 10), (5, 5)]
        manual = [(0, 0), (0, 0), (1, 1), None, None]
        result = pv.position_errors(auto, manual, scale_factor=0.1)
        self.assertEqual(result.n_manual, 3)
        self.assertEqual(result.n_matched, 2)
        self.assertEqual(result.n_missed, 1)
        self.assertAlmostEqual(result.miss_rate, 1 / 3)
        self.assertEqual(result.false_detections, 2)
        self.assertEqual(result.errors_px.tolist(), [0.0, 5.0])
        self.assertAlmostEqual(result.mean_px, 2.5)
        self.assertAlmostEqual(result.median_px, 2.5)
        self.assertAlmostEqual(result.mean_cm, 0.25)
        self.assertAlmostEqual(result.errors_cm[1], 0.5)

    def test_percentile_uses_linear_interpolation(self):
        manual = [(0, 0)] * 21
        auto = [(float(i), 0.0) for i in range(21)]   # errors 0..20
        result = pv.position_errors(auto, manual, 0.0)
        self.assertAlmostEqual(result.p95_px, 19.0)
        self.assertAlmostEqual(result.median_px, 10.0)

    def test_uncalibrated_has_no_cm(self):
        result = pv.position_errors([(3, 4)], [(0, 0)], scale_factor=0.0)
        self.assertEqual(result.mean_px, 5.0)
        self.assertTrue(math.isnan(result.mean_cm))
        self.assertTrue(math.isnan(result.p95_cm))

    def test_nan_coordinates_count_as_missing(self):
        result = pv.position_errors(
            [(float("nan"), 1.0)], [(1.0, 1.0)], 0.0)
        self.assertEqual(result.n_missed, 1)
        self.assertEqual(result.n_matched, 0)
        self.assertTrue(math.isnan(result.mean_px))

    def test_no_manual_points(self):
        result = pv.position_errors([(1, 1)], [None], 0.1)
        self.assertEqual(result.n_manual, 0)
        self.assertTrue(math.isnan(result.miss_rate))
        self.assertEqual(result.false_detections, 1)

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            pv.position_errors([(0, 0)], [], 0.1)
        with self.assertRaises(ValueError):
            pv.position_errors([], [], -1.0)


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class TrackingCsvTests(unittest.TestCase):
    def _export(self, directory):
        labels = ["arm_a", "arm_a", "center", "undetected", "arm_b", "arm_b"]
        result = YMazeAnalysisResult(
            positions=[(10.0, 20.0), (11.5, 21.25), (30.0, 40.0), None,
                       (50.0, 60.0), (51.0, 61.0)],
            timestamps=[10 / 30.0 + i / 30.0 for i in range(6)],
            fps=30.0,
            raw_roi_labels=list(labels),
            roi_labels=list(labels),
            visual_labels=list(labels),
            detection_confidences=[0.9, 0.8, 0.7, 0.0, 0.6, 0.5],
            detection_failure_reasons=["", "", "", "no_blob", "", ""],
        )
        path = Path(directory) / "video_tracking_data.csv"
        DataExporter.export_to_csv(result, path)
        return result, path

    def test_round_trip_through_export_to_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            result, path = self._export(directory)
            data = pv.load_tracking_csv(path)
        self.assertEqual(len(data), 6)
        self.assertEqual(data.frames, [10, 11, 12, 13, 14, 15])
        self.assertAlmostEqual(data.timestamps[0], 0.333, places=3)
        self.assertEqual(data.positions[0], (10.0, 20.0))
        self.assertEqual(data.positions[1], (11.5, 21.25))
        self.assertIsNone(data.positions[3])
        self.assertEqual(data.zones, result.roi_labels)
        self.assertEqual(data.raw_zones, result.raw_roi_labels)
        self.assertEqual(data.visual_zones, result.visual_labels)
        self.assertEqual(data.detected, [True, True, True, False, True, True])
        self.assertEqual(data.confidences[1], 0.8)
        self.assertEqual(data.failure_reasons[3], "no_blob")
        self.assertAlmostEqual(data.fps, 30.0, delta=0.5)  # timestamps are rounded to ms

    def test_loaded_data_feeds_the_validation_functions(self):
        with tempfile.TemporaryDirectory() as directory:
            _, path = self._export(directory)
            data = pv.load_tracking_csv(path)
        manual_xy = [(10.0, 20.0), (11.0, 21.0), (30.0, 40.0), (0.0, 0.0),
                     (50.0, 60.0), (51.0, 61.0)]
        errors = pv.position_errors(data.positions, manual_xy, 0.0)
        self.assertEqual(errors.n_missed, 1)
        self.assertEqual(errors.n_matched, 5)
        agreement = pv.frame_label_agreement(
            data.zones, ["arm_a", "arm_a", "center", "arm_b", "arm_b", "arm_a"])
        self.assertEqual(agreement.n_frames, 5)
        self.assertEqual(agreement.n_ignored, 1)
        self.assertAlmostEqual(agreement.percent_agreement, 80.0)

    def test_recounts_entries_from_the_exported_zones(self):
        labels = ["arm_a", "center", "arm_b", "center", "arm_c"]
        result = YMazeAnalysisResult(
            positions=[(0.0, 0.0)] * 5,
            timestamps=[i * 0.5 for i in range(5)], fps=2.0,
            raw_roi_labels=list(labels), roi_labels=list(labels),
            visual_labels=list(labels),
        )
        YMaze().calculate_metrics(result)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.csv"
            DataExporter.export_to_csv(result, path)
            data = pv.load_tracking_csv(path)
        entries = data.entries({"arm_a", "arm_b", "arm_c"})
        self.assertEqual([arm for arm, _ in entries],
                         [arm for arm, _ in result.arm_entries])

    def test_rejects_other_csv_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write(
                directory, "evidence.csv",
                "frame,timestamp_s,mouse_1_behavior_state\n1,0.03,active_escape\n")
            with self.assertRaisesRegex(ValueError, "tracking data"):
                pv.load_tracking_csv(path)


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class BatchSummaryTests(unittest.TestCase):
    def test_summary_over_per_video_pairs(self):
        rng = np.random.default_rng(5)
        manual = rng.uniform(30, 90, 15)
        auto = manual + rng.normal(1.0, 2.0, 15)
        summary = pv.summarize_agreement(list(zip(auto, manual)))
        self.assertEqual(summary.n, 15)
        self.assertEqual(summary.n_excluded, 0)
        self.assertGreater(summary.icc.icc, 0.95)
        self.assertAlmostEqual(summary.bland_altman.bias, float(np.mean(auto - manual)))
        self.assertGreater(summary.pearson.r, 0.95)
        self.assertGreater(summary.spearman.r, 0.9)
        self.assertAlmostEqual(
            summary.rmse, float(np.sqrt(np.mean((auto - manual) ** 2))))
        self.assertAlmostEqual(summary.mean_manual, float(manual.mean()))

    def test_missing_pairs_are_excluded_and_counted(self):
        pairs = [(1.0, 1.0), (2.0, 2.5), (3.0, 3.0), (None, 4.0),
                 (float("nan"), 5.0), (6.0, 7.0)]
        summary = pv.summarize_agreement(pairs)
        self.assertEqual(summary.n, 4)
        self.assertEqual(summary.n_excluded, 2)

    def test_too_few_pairs(self):
        summary = pv.summarize_agreement([(1.0, 2.0)])
        self.assertIsNone(summary.icc)
        self.assertTrue(math.isnan(summary.pearson.r))
        empty = pv.summarize_agreement([])
        self.assertEqual(empty.n, 0)
        self.assertTrue(math.isnan(empty.mean_auto))

    def test_summarize_metrics_keeps_metric_names(self):
        result = pv.summarize_metrics({
            "alternation_pct": [(50.0, 55.0), (60.0, 58.0), (70.0, 72.0)],
            "entries": [(10, 10), (12, 13), (20, 19)],
        })
        self.assertEqual(set(result), {"alternation_pct", "entries"})
        self.assertEqual(result["entries"].n, 3)


if __name__ == "__main__":
    unittest.main()
