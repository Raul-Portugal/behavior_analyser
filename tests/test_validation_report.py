import base64
import csv
import hashlib
import json
import math
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core import validation_report as vr
from core.agreement_stats import cohens_kappa, summarize_agreement
from core.manual_scoring import (ScoreEvent, ZoneInterval, load_epm_totals,
                                 write_arm_entries_csv, write_epm_totals_csv,
                                 write_tst_annotations_csv,
                                 write_zone_intervals_csv, center_time_s)
from core.positional_validation import (compare_epm, compare_y_maze,
                                        epm_totals_from_entries,
                                        load_entry_annotations, match_events,
                                        position_errors)
from core.tst_validation import (evaluate_annotations,
                                 immobility_time_agreement, load_annotations,
                                 load_tst_evidence_csv)
from core.version import APP_VERSION

FPS = 25.0


def write_entries(path, entries):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["entry_number", "arm", "timestamp_s"])
        for number, (arm, stamp) in enumerate(entries, start=1):
            writer.writerow([number, arm, f"{stamp:.3f}"])


def y_entries(seed):
    """A seeded entry list: arms follow a noisy rotation."""
    rng = np.random.default_rng(seed)
    arms, entries, current = ["arm_a", "arm_b", "arm_c"], [], 0
    stamp = 3.0
    for _ in range(int(rng.integers(12, 20))):
        current = (current + int(rng.choice([1, 1, 2, 0]))) % 3
        stamp += float(rng.uniform(2.0, 8.0))
        entries.append((arms[current], round(stamp, 3)))
    return entries


def perturb(entries, seed):
    """A 'manual' version: small time shifts, one drop, one wrong arm."""
    rng = np.random.default_rng(seed)
    manual = [(arm, round(t + float(rng.uniform(-0.3, 0.3)), 3))
              for arm, t in entries]
    if len(manual) > 6 and seed % 2 == 0:
        del manual[3]
    if seed % 3 == 0:
        arm, t = manual[5]
        manual[5] = ("arm_a" if arm != "arm_a" else "arm_b", t)
    return manual


def write_summary(path, specific=None, paradigm="Y-Maze", **extra):
    data = {"metadata": {"scale_factor": 0.05, "fps": FPS},
            "run_configuration": {"paradigm_id": paradigm,
                                  "scale_factor": 0.05}}
    if specific is not None:
        data["specific_metrics"] = specific
    data.update(extra)
    Path(path).write_text(json.dumps(data), encoding="utf-8")


class ReportTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.auto = self.root / "run" / "analysis_run_1"
        self.manual = self.root / "manual"
        self.auto.mkdir(parents=True)
        self.manual.mkdir()

    # ----- builders -----
    def make_y_maze(self, count=6):
        self.y_auto, self.y_manual = {}, {}
        for i in range(1, count + 1):
            stem = f"ym{i}"
            auto = y_entries(i)
            manual = perturb(auto, i)
            folder = self.auto / stem / "data"
            folder.mkdir(parents=True)
            write_entries(folder / f"{stem}_arm_entries.csv", auto)
            write_summary(folder / f"{stem}_summary.json",
                          {"arm_sequence": "ABC", "arm_entries": auto})
            write_entries(self.manual / f"{stem}_manual_arm_entries.csv",
                          manual)
            self.y_auto[stem], self.y_manual[stem] = auto, manual

    def make_epm(self, count=5):
        self.epm = {}
        rng = np.random.default_rng(7)
        names = ["open_arm_1", "open_arm_2", "closed_arm_1", "closed_arm_2"]
        for i in range(1, count + 1):
            stem = f"epm{i}"
            auto = [(names[int(rng.integers(0, 4))], 5.0 + 4.0 * k)
                    for k in range(10 + i)]
            manual = [(a, t + 0.2) for a, t in auto[:-1]]
            opens = sum(a.startswith("open") for a, _ in auto)
            closed = len(auto) - opens
            open_s = float(rng.uniform(30, 120))
            closed_s = float(rng.uniform(60, 200))
            specific = {
                "time_in_open_arms_s": open_s,
                "time_in_closed_arms_s": closed_s,
                "entries_to_open_arms": opens,
                "entries_to_closed_arms": closed,
                "total_arm_entries": len(auto),
                "percent_time_in_open": open_s / (open_s + closed_s) * 100,
                "percent_entries_to_open": opens / len(auto) * 100,
                "arm_entries": auto,
            }
            folder = self.auto / stem
            folder.mkdir(parents=True)
            write_entries(folder / f"{stem}_arm_entries.csv", auto)
            write_summary(folder / f"{stem}_summary.json", specific,
                          "Elevated Plus Maze")
            write_entries(self.manual / f"{stem}_manual_arm_entries.csv",
                          manual)
            manual_open = open_s + float(rng.uniform(-6, 6))
            write_epm_totals_csv(
                self.manual / f"{stem}_manual_epm_totals.csv",
                {"time_in_open_arms_s": manual_open,
                 "time_in_closed_arms_s": closed_s,
                 "percent_time_in_open":
                     manual_open / (manual_open + closed_s) * 100})
            self.epm[stem] = (specific, manual, manual_open, closed_s)

    def make_tst(self, count=4):
        self.tst_truth, self.tst_dirs = {}, {}
        states = ["immobile_static", "passive_swing", "active_escape",
                  "uncertain"]
        for i in range(1, count + 1):
            stem = f"tst{i}"
            rng = np.random.default_rng(100 + i)
            folder = self.auto / stem
            folder.mkdir(parents=True)
            frames = list(range(0, 300))
            auto_states = {m: [states[int(rng.choice([0, 0, 1, 2, 2, 3]))]
                               for _ in frames] for m in ("mouse_1", "mouse_2")}
            header = ["frame", "timestamp_s"]
            for mouse in auto_states:
                header += [f"{mouse}_behavior_state", f"{mouse}_is_immobile"]
            with (folder / f"{stem}_tst_motion_evidence.csv").open(
                    "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(header)
                for index, frame in enumerate(frames):
                    row = [frame, f"{frame / FPS:.3f}"]
                    for mouse, sequence in auto_states.items():
                        state = sequence[index]
                        row += [state, int(state in ("immobile_static",
                                                     "passive_swing"))]
                    writer.writerow(row)
            write_summary(folder / f"{stem}_summary.json", None,
                          "Tail Suspension Test",
                          tst_metrics={"total_immobility_time_s": {}})
            labels = {}
            for mouse, sequence in auto_states.items():
                for index, frame in enumerate(frames[:200]):
                    state = sequence[index]
                    if rng.random() < 0.2:
                        state = states[int(rng.integers(0, 4))]
                    labels[(mouse, frame)] = state
            write_tst_annotations_csv(
                self.manual / f"{stem}_manual_tst_annotations.csv", labels)
            self.tst_truth[stem] = labels
            self.tst_dirs[stem] = folder

    def make_open_field(self, count=4):
        self.of = {}
        for i in range(1, count + 1):
            stem = f"of{i}"
            folder = self.auto / stem
            folder.mkdir(parents=True)
            center_frames = 20 * i + 10
            auto_center = center_frames / FPS
            write_summary(
                folder / f"{stem}_summary.json", None, "Freestyle / Open Field",
                freestyle_metrics={"arena": {"center_time_s": auto_center}})
            with (folder / f"{stem}_arena_zones.csv").open(
                    "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["frame", "timestamp_s", "arena_zone",
                                 "wall_distance_cm", "distance_unit"])
                for frame in range(200):
                    zone = ("center" if 50 <= frame < 50 + center_frames
                            else "periphery")
                    if frame == 3:
                        zone = "undetected"
                    writer.writerow([frame, f"{frame / FPS:.3f}", zone, "", "cm"])
            intervals = [
                ZoneInterval("periphery", 0, 54, 0.0, 55 / FPS),
                ZoneInterval("center", 55, 55 + center_frames - 1, 55 / FPS,
                             (55 + center_frames) / FPS),
                ZoneInterval("periphery", 55 + center_frames, 199,
                             (55 + center_frames) / FPS, 200 / FPS),
            ]
            write_zone_intervals_csv(
                self.manual / f"{stem}_manual_zone_intervals.csv", intervals)
            self.of[stem] = (auto_center, intervals)

    def make_spotchecks(self):
        self.spot = {}
        rng = np.random.default_rng(5)
        for stem, scale in (("ym1", 0.05), ("epm1", 0.1)):
            rows = []
            for n in range(1, 21):
                if n % 7 == 0:
                    rows.append((n, None, (100.0, 100.0)))      # not visible
                elif n % 5 == 0:
                    rows.append((n, (200.0, 50.0), None))       # tracker miss
                else:
                    manual = (float(rng.uniform(0, 500)),
                              float(rng.uniform(0, 500)))
                    auto = (manual[0] + float(rng.normal(0, 3)),
                            manual[1] + float(rng.normal(0, 3)))
                    rows.append((n, manual, auto))
            path = self.manual / f"{stem}_position_spotcheck.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["sample_number", "frame", "timestamp_s",
                                 "visible", "manual_x", "manual_y", "auto_x",
                                 "auto_y", "error_px", "error_cm", "outcome",
                                 "scale_factor"])
                for n, manual, auto in rows:
                    writer.writerow([
                        n, n, f"{n / FPS:.3f}", 0 if manual is None else 1,
                        "" if manual is None else manual[0],
                        "" if manual is None else manual[1],
                        "" if auto is None else auto[0],
                        "" if auto is None else auto[1], "", "", "matched",
                        scale])
            self.spot[stem] = (rows, scale)

    def inputs(self):
        automated = vr.discover_automated(self.auto)
        manual = vr.discover_manual(self.manual)
        return vr.combine_inputs(automated, manual)

    def config(self, **kwargs):
        kwargs.setdefault("tolerance_s", 1.0)
        kwargs.setdefault("criteria", vr.default_criteria())
        return vr.ReportConfig(**kwargs)


class DiscoveryTests(ReportTestCase):
    def test_automated_runs_are_found_with_paradigm_and_siblings(self):
        self.make_y_maze(2)
        self.make_epm(1)
        self.make_tst(1)
        self.make_open_field(1)
        found = vr.discover_automated(self.auto)
        self.assertEqual(sorted(found), ["epm1", "of1", "tst1", "ym1", "ym2"])
        self.assertEqual(found["ym1"].paradigm, "y_maze")
        self.assertEqual(found["epm1"].paradigm, "epm")
        self.assertEqual(found["tst1"].paradigm, "tst")
        self.assertEqual(found["of1"].paradigm, "open_field")
        self.assertIsNotNone(found["ym1"].arm_entries)
        self.assertIsNotNone(found["tst1"].tst_evidence)
        self.assertIsNotNone(found["of1"].arena_zones)
        self.assertIsNone(found["ym1"].tst_evidence)

    def test_spotcheck_summary_and_duplicates(self):
        self.make_y_maze(1)
        (self.auto / "ym1" / "ym1_position_spotcheck_summary.json").parent \
            .mkdir(parents=True, exist_ok=True)
        (self.auto / "ym1" / "ym1_position_spotcheck_summary.json") \
            .write_text("{}", encoding="utf-8")
        other = self.auto.parent / "analysis_run_2"
        (other / "ym1").mkdir(parents=True)
        write_summary(other / "ym1" / "ym1_summary.json", {"arm_sequence": ""})
        found = vr.discover_automated([self.auto, other])
        self.assertEqual(sorted(found), ["ym1", "ym1 #2"])

    def test_manual_files_are_classified_by_name_and_header(self):
        self.make_y_maze(1)
        self.make_tst(1)
        self.make_open_field(1)
        self.make_epm(1)
        self.make_spotchecks()
        # A scorer's own file, not named by the dialog.
        write_entries(self.manual / "scorer_clip7.csv", [("arm_a", 1.0)])
        # Automated CSVs in the manual folder must not be picked up.
        write_entries(self.manual / "ym9_arm_entries.csv", [("arm_a", 1.0)])
        found = vr.discover_manual(self.manual)
        self.assertEqual(found["ym1"].manual_arm_entries.name,
                         "ym1_manual_arm_entries.csv")
        self.assertEqual(found["ym1"].spotcheck.name,
                         "ym1_position_spotcheck.csv")
        self.assertEqual(found["tst1"].manual_tst_annotations.name,
                         "tst1_manual_tst_annotations.csv")
        self.assertEqual(found["of1"].manual_zone_intervals.name,
                         "of1_manual_zone_intervals.csv")
        self.assertEqual(found["epm1"].manual_epm_totals.name,
                         "epm1_manual_epm_totals.csv")
        self.assertIn("scorer_clip7", found)
        self.assertIsNotNone(found["scorer_clip7"].manual_arm_entries)
        self.assertNotIn("ym9", found)
        self.assertIsNone(vr.classify_manual_file(self.manual / "nothing.txt"))

    def test_combine_matches_by_stem_ignoring_case_and_keeps_strays(self):
        self.make_y_maze(2)
        write_entries(self.manual / "ym1_manual_arm_entries.csv",
                      [("arm_a", 1.0)])
        lonely = self.manual / "stray_manual_arm_entries.csv"
        write_entries(lonely, [("arm_a", 1.0)])
        upper = self.manual / "YM2_manual_arm_entries.csv"
        (self.manual / "ym2_manual_arm_entries.csv").replace(upper)
        combined = {v.stem: v for v in self.inputs()}
        self.assertEqual(sorted(combined), ["stray", "ym1", "ym2"])
        self.assertIsNotNone(combined["ym1"].summary)
        self.assertIsNotNone(combined["ym2"].manual_arm_entries)
        self.assertIsNone(combined["stray"].summary)
        self.assertIsNone(combined["stray"].paradigm)

    def test_paradigm_can_be_inferred_from_manual_files(self):
        self.make_tst(1)
        manual_only = vr.combine_inputs({}, vr.discover_manual(self.manual))
        self.assertEqual(manual_only[0].paradigm, "tst")


class YMazeTests(ReportTestCase):
    def test_per_video_and_pooled_numbers_match_core_calls(self):
        self.make_y_maze(6)
        results = vr.build_results(self.inputs(), self.config())
        result = results.paradigms["y_maze"]
        self.assertEqual(result.n_videos, 6)

        direct = {stem: compare_y_maze(
            self.y_auto[stem], load_entry_annotations(
                self.manual / f"{stem}_manual_arm_entries.csv", "y_maze"), 1.0)
            for stem in self.y_auto}
        columns = result.columns
        for row in result.rows:
            comparison = direct[row[0]]
            record = dict(zip(columns, row))
            self.assertEqual(record["auto_entries"],
                             comparison.auto_total_entries)
            self.assertEqual(record["manual_entries"],
                             comparison.manual_total_entries)
            self.assertEqual(record["true_positives"],
                             comparison.matching.true_positives)
            self.assertEqual(record["auto_sequence"], comparison.auto_sequence)
            if math.isfinite(comparison.auto_alternation_pct):
                self.assertAlmostEqual(record["auto_alternation_pct"],
                                       comparison.auto_alternation_pct)

        pairs = [(c.auto_alternation_pct, c.manual_alternation_pct)
                 for c in direct.values()]
        reference = summarize_agreement(pairs)
        pooled = result.agreements["alternation_pct"].summary
        self.assertEqual(pooled.n, reference.n)
        self.assertAlmostEqual(pooled.icc.icc, reference.icc.icc)
        self.assertAlmostEqual(pooled.icc.ci_low, reference.icc.ci_low)
        self.assertAlmostEqual(pooled.bland_altman.bias,
                               reference.bland_altman.bias)
        self.assertAlmostEqual(pooled.pearson.r, reference.pearson.r)
        totals = summarize_agreement(
            [(c.auto_total_entries, c.manual_total_entries)
             for c in direct.values()])
        self.assertAlmostEqual(
            result.agreements["total_entries"].summary.icc.icc,
            totals.icc.icc)

        events = result.events
        tp = sum(c.matching.true_positives for c in direct.values())
        fp = sum(c.matching.false_positives for c in direct.values())
        fn = sum(c.matching.false_negatives for c in direct.values())
        self.assertEqual((events.true_positives, events.false_positives,
                          events.false_negatives), (tp, fp, fn))
        self.assertAlmostEqual(events.precision, tp / (tp + fp))
        self.assertAlmostEqual(events.recall, tp / (tp + fn))
        self.assertAlmostEqual(events.f1, 2 * tp / (2 * tp + fp + fn))
        self.assertEqual(events.tolerance_s, 1.0)

    def test_tolerance_changes_the_event_counts(self):
        self.make_y_maze(4)
        tight = vr.build_results(self.inputs(), self.config(tolerance_s=0.05))
        loose = vr.build_results(self.inputs(), self.config(tolerance_s=2.0))
        self.assertLess(tight.paradigms["y_maze"].events.true_positives,
                        loose.paradigms["y_maze"].events.true_positives)

    def test_automated_entries_fall_back_to_the_summary_json(self):
        self.make_y_maze(3)
        for path in self.auto.rglob("*_arm_entries.csv"):
            path.unlink()
        results = vr.build_results(self.inputs(), self.config())
        self.assertEqual(results.paradigms["y_maze"].n_videos, 3)


class EpmTests(ReportTestCase):
    def test_numbers_match_core_calls(self):
        self.make_epm(5)
        results = vr.build_results(self.inputs(), self.config())
        result = results.paradigms["epm"]
        self.assertEqual(result.n_videos, 5)

        time_pairs, entry_pairs, total_pairs = [], [], []
        matchings = []
        for stem, (specific, manual, manual_open, closed_s) in self.epm.items():
            entries = load_entry_annotations(
                self.manual / f"{stem}_manual_arm_entries.csv", "epm")
            totals = dict(epm_totals_from_entries(entries))
            totals.update(load_epm_totals(
                self.manual / f"{stem}_manual_epm_totals.csv"))
            diffs = compare_epm(specific, totals)
            time_pairs.append((diffs["percent_time_in_open"].auto,
                               diffs["percent_time_in_open"].manual))
            entry_pairs.append((diffs["percent_entries_to_open"].auto,
                                diffs["percent_entries_to_open"].manual))
            total_pairs.append((diffs["total_arm_entries"].auto,
                                diffs["total_arm_entries"].manual))
            matchings.append(match_events(
                specific["arm_entries"], entries, 1.0))
        for key, pairs in (("percent_time_in_open", time_pairs),
                           ("percent_entries_to_open", entry_pairs),
                           ("total_arm_entries", total_pairs)):
            reference = summarize_agreement(pairs)
            pooled = result.agreements[key].summary
            self.assertEqual(pooled.n, reference.n)
            self.assertAlmostEqual(pooled.icc.icc, reference.icc.icc)
            self.assertAlmostEqual(pooled.bland_altman.bias,
                                   reference.bland_altman.bias)
        self.assertAlmostEqual(
            result.events.true_positives,
            sum(m.true_positives for m in matchings))
        self.assertEqual(result.events.false_negatives,
                         sum(m.false_negatives for m in matchings))

    def test_coarse_manual_arm_names_match_on_open_vs_closed(self):
        self.make_epm(2)
        path = self.manual / "epm1_manual_arm_entries.csv"
        specific = self.epm["epm1"][0]
        coarse = [("open" if a.startswith("open") else "closed", t)
                  for a, t in specific["arm_entries"]]
        write_entries(path, coarse)
        results = vr.build_results(self.inputs(), self.config())
        row = dict(zip(results.paradigms["epm"].columns,
                       next(r for r in results.paradigms["epm"].rows
                            if r[0] == "epm1")))
        self.assertEqual(row["event_false_negatives"], 0)
        self.assertEqual(row["event_f1"], 1.0)


class TstTests(ReportTestCase):
    def test_numbers_match_core_calls(self):
        self.make_tst(4)
        results = vr.build_results(self.inputs(), self.config())
        result = results.paradigms["tst"]
        pairs, state_pairs = [], []
        for stem, folder in self.tst_dirs.items():
            evidence = load_tst_evidence_csv(
                folder / f"{stem}_tst_motion_evidence.csv")
            annotations = load_annotations(
                self.manual / f"{stem}_manual_tst_annotations.csv")
            per_mouse = evaluate_annotations(evidence, annotations)
            times = immobility_time_agreement(evidence, annotations)
            for mouse, stats in per_mouse.items():
                row = next(r for r in result.rows
                           if r[0] == stem and r[1] == mouse)
                record = dict(zip(result.columns, row))
                self.assertEqual(record["annotated_frames"],
                                 int(stats["annotated_frames"]))
                self.assertAlmostEqual(record["exact_state_agreement_pct"],
                                       stats["exact_state_agreement_pct"])
                self.assertAlmostEqual(record["immobility_f1"],
                                       stats["immobility_f1"])
                self.assertAlmostEqual(record["auto_immobility_s"],
                                       times[mouse].auto_s)
                pairs.append((times[mouse].auto_s, times[mouse].manual_s))
            for mouse, rows in evidence.items():
                for item in rows:
                    truth = annotations.get((mouse, item["source_frame"]))
                    if truth is not None:
                        state_pairs.append((item["state"], truth))
        reference = summarize_agreement(pairs)
        pooled = result.agreements["immobility_s"].summary
        self.assertEqual(pooled.n, 8)
        self.assertAlmostEqual(pooled.icc.icc, reference.icc.icc)
        self.assertAlmostEqual(pooled.bland_altman.bias,
                               reference.bland_altman.bias)

        confusion = result.confusions[0]
        self.assertEqual(int(confusion.matrix.sum()), len(state_pairs))
        kappa = cohens_kappa([a for a, _ in state_pairs],
                             [m for _, m in state_pairs],
                             categories=confusion.categories)
        self.assertAlmostEqual(confusion.kappa.kappa, kappa.kappa)
        np.testing.assert_array_equal(confusion.matrix, kappa.confusion)
        self.assertEqual(confusion.categories[:4],
                         ["immobile_static", "passive_swing", "active_escape",
                          "uncertain"])
        # The immobility counts follow from the confusion matrix.
        immobile = {"immobile_static", "passive_swing"}
        tp = sum(a in immobile and m in immobile for a, m in state_pairs)
        fp = sum(a in immobile and m not in immobile for a, m in state_pairs)
        self.assertAlmostEqual(result.frame_stats["immobility_precision"],
                               tp / (tp + fp))

    def test_annotation_for_unknown_mouse_is_reported(self):
        self.make_tst(1)
        path = self.manual / "tst1_manual_tst_annotations.csv"
        labels = load_annotations(path)
        labels[("ghost", 5)] = "uncertain"
        write_tst_annotations_csv(path, labels)
        results = vr.build_results(self.inputs(), self.config())
        self.assertTrue(any("ghost" in w for w in results.warnings))
        self.assertIn("tst", results.paradigms)


class OpenFieldTests(ReportTestCase):
    def test_center_time_and_zone_labels(self):
        self.make_open_field(4)
        results = vr.build_results(self.inputs(), self.config())
        result = results.paradigms["open_field"]
        pairs = [(auto, center_time_s(intervals))
                 for auto, intervals in self.of.values()]
        reference = summarize_agreement(pairs)
        pooled = result.agreements["center_time_s"].summary
        self.assertEqual(pooled.n, 4)
        self.assertAlmostEqual(pooled.mean_auto, reference.mean_auto)
        self.assertAlmostEqual(pooled.bland_altman.bias,
                               reference.bland_altman.bias)
        # Manual center starts 5 frames later than the automated label.
        for row in result.rows:
            record = dict(zip(result.columns, row))
            self.assertAlmostEqual(record["difference_s"], 0.0, places=6)
            self.assertGreater(record["frames_compared"], 190)
            self.assertLess(record["frame_kappa"], 1.0)
            self.assertGreater(record["frame_kappa"], 0.8)
        self.assertEqual(result.confusions[0].categories,
                         ["center", "periphery"])
        self.assertIn("kappa", result.frame_stats)

    def test_run_without_arena_metrics_is_skipped_with_a_warning(self):
        self.make_open_field(2)
        write_summary(self.auto / "of1" / "of1_summary.json", None,
                      "Freestyle / Open Field", freestyle_metrics={})
        results = vr.build_results(self.inputs(), self.config())
        self.assertEqual(results.paradigms["open_field"].n_videos, 1)
        self.assertTrue(any("of1" in w and "arena" in w
                            for w in results.warnings))


class SpotcheckTests(ReportTestCase):
    def test_pooled_position_statistics(self):
        self.make_y_maze(1)
        self.make_epm(1)
        self.make_spotchecks()
        results = vr.build_results(self.inputs(), self.config())
        pooled = results.spotcheck
        errors_px, errors_cm = [], []
        missed = visible = false = reviewed = 0
        for rows, scale in self.spot.values():
            auto = [a for _, _, a in rows]
            manual = [m for _, m, _ in rows]
            direct = position_errors(auto, manual, scale)
            errors_px.append(direct.errors_px)
            errors_cm.append(direct.errors_cm)
            missed += direct.n_missed
            visible += direct.n_manual
            false += direct.false_detections
            reviewed += len(rows)
        px = np.concatenate(errors_px)
        cm = np.concatenate(errors_cm)
        self.assertEqual(pooled.n_files, 2)
        self.assertEqual(pooled.n_reviewed, reviewed)
        self.assertEqual(pooled.n_missed, missed)
        self.assertAlmostEqual(pooled.miss_rate, missed / visible)
        self.assertEqual(pooled.false_detections, false)
        self.assertAlmostEqual(pooled.median_px, float(np.median(px)))
        self.assertAlmostEqual(pooled.p95_px, float(np.percentile(px, 95)))
        self.assertAlmostEqual(pooled.median_cm, float(np.median(cm)))
        self.assertEqual(len(results.spotcheck_rows), 2)

    def test_no_spotchecks_means_no_section(self):
        self.make_y_maze(2)
        results = vr.build_results(self.inputs(), self.config())
        self.assertIsNone(results.spotcheck)
        tables = {t.name for t in vr.build_tables(results)}
        self.assertNotIn("position_spotcheck_pooled", tables)


class CriteriaTests(ReportTestCase):
    def test_pass_fail_and_not_evaluable(self):
        self.make_y_maze(6)
        results = vr.build_results(self.inputs(), self.config(criteria=[]))
        statistics = vr.flat_statistics(results.paradigms, results.spotcheck)
        icc = statistics["y_maze.alternation_pct.icc"]
        criteria = [
            vr.Criterion("y_maze.alternation_pct.icc", ">=", icc - 0.01),
            vr.Criterion("y_maze.alternation_pct.icc", ">=", icc + 0.01),
            vr.Criterion("y_maze.alternation_pct.abs_bias", "<=", 1e9),
            vr.Criterion("tst.immobility_s.icc", ">=", 0.9),
            vr.Criterion("position.spotcheck.miss_rate", "<=", 0.1),
        ]
        evaluated = vr.evaluate_criteria(criteria, statistics)
        self.assertEqual([r.status for r in evaluated],
                         ["pass", "fail", "pass", "not evaluable",
                          "not evaluable"])
        self.assertEqual(evaluated[0].value, icc)
        self.assertIn("Alternation", evaluated[0].label)

    def test_comparison_operators(self):
        for op, value, expected in ((">=", 0.9, "pass"), (">", 0.9, "fail"),
                                    ("<=", 0.9, "pass"), ("<", 0.9, "fail")):
            self.assertEqual(vr.Criterion("k", op, 0.9).check(value), expected)
        with self.assertRaises(ValueError):
            vr.Criterion("k", "==", 1)
        self.assertEqual(vr.Criterion("k", ">=", 1).check(float("nan")),
                         "not evaluable")

    def test_defaults_refer_to_known_keys(self):
        known = {key for key, _ in vr.criteria_catalog()}
        for criterion in vr.default_criteria():
            self.assertIn(criterion.key, known)
        self.assertEqual(len(known), len(vr.criteria_catalog()))

    def test_every_computed_statistic_has_a_catalog_key(self):
        self.make_y_maze(4)
        self.make_epm(3)
        self.make_tst(2)
        self.make_open_field(3)
        self.make_spotchecks()
        results = vr.build_results(self.inputs(), self.config())
        known = {key for key, _ in vr.criteria_catalog()}
        for key in vr.flat_statistics(results.paradigms, results.spotcheck):
            if key.endswith(".n"):
                continue
            self.assertIn(key, known)


class ReportOutputTests(ReportTestCase):
    def build_full(self):
        self.make_y_maze(6)
        self.make_epm(5)
        self.make_tst(3)
        self.make_open_field(4)
        self.make_spotchecks()
        out = self.root / "report"
        outcome = vr.build_report(
            self.inputs(),
            self.config(title="Lab <validation> & test", notes="Setup A"),
            out)
        return out, outcome

    def test_html_is_self_contained(self):
        out, outcome = self.build_full()
        html = outcome["html"].read_text(encoding="utf-8")
        self.assertEqual(outcome["html"], out / "validation_report.html")
        self.assertNotIn("http://", html.lower())
        self.assertNotIn("https://", html.lower())
        self.assertNotRegex(html, r"""(?:src|href)\s*=\s*['"](?!data:)""")
        self.assertNotIn("<script", html.lower())
        self.assertNotIn("@import", html)
        self.assertNotRegex(html, r"url\((?!['\"]?data:)")
        images = re.findall(r"src='data:image/png;base64,([A-Za-z0-9+/=]+)'",
                            html)
        # 2 plots per pooled endpoint + confusion matrices + error histogram.
        self.assertGreaterEqual(len(images), 12)
        for encoded in images:
            self.assertEqual(base64.b64decode(encoded)[:8],
                             b"\x89PNG\r\n\x1a\n")

    def test_html_content(self):
        _, outcome = self.build_full()
        results = outcome["results"]
        html = outcome["html"].read_text(encoding="utf-8")
        self.assertIn("<title>Lab &lt;validation&gt; &amp; test</title>", html)
        self.assertIn("Setup A", html)
        self.assertIn(APP_VERSION, html)
        self.assertIn(results.generated, html)
        for heading in ("Acceptance criteria", "Pooled agreement",
                        "Y-Maze", "Elevated Plus Maze",
                        "Freestyle / Open Field", "Tail Suspension Test",
                        "Tracking position spot-check", "Reproducibility",
                        "confusion matrix"):
            self.assertIn(heading, html)
        self.assertRegex(html, r"badge (pass|fail)")
        icc = results.paradigms["y_maze"].agreements["alternation_pct"] \
            .summary.icc.icc
        self.assertIn(f"{icc:.3f}", html)
        # The input files are listed with hashes for reproducibility.
        digest = hashlib.sha256(
            (self.manual / "ym1_manual_arm_entries.csv").read_bytes()
        ).hexdigest()
        self.assertIn(digest, html)
        self.assertIn("ym1_manual_arm_entries.csv", html)

    def test_every_table_is_written_as_csv(self):
        out, outcome = self.build_full()
        tables = vr.build_tables(outcome["results"])
        names = {t.name for t in tables}
        for expected in ("y_maze_per_video", "epm_per_video",
                         "open_field_per_video", "tst_per_mouse",
                         "pooled_agreement", "pooled_events",
                         "frame_agreement", "tst_states_confusion",
                         "open_field_zones_confusion", "acceptance_criteria",
                         "input_files", "inputs",
                         "position_spotcheck_pooled",
                         "position_spotcheck_per_video"):
            self.assertIn(expected, names)
        self.assertEqual(len(outcome["csv"]), len(tables))
        for table in tables:
            path = out / f"validation_report_{table.name}.csv"
            self.assertTrue(path.is_file(), table.name)
            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.reader(file))
            self.assertEqual(rows[0], table.columns)
            self.assertEqual(len(rows) - 1, len(table.rows))

    def test_csv_values_equal_computed_values(self):
        out, outcome = self.build_full()
        results = outcome["results"]
        with (out / "validation_report_pooled_agreement.csv").open(
                newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        row = next(r for r in rows if r["paradigm"] == "Y-Maze"
                   and r["metric"] == "Alternation")
        summary = results.paradigms["y_maze"].agreements[
            "alternation_pct"].summary
        self.assertAlmostEqual(float(row["icc_2_1"]), summary.icc.icc, 6)
        self.assertAlmostEqual(float(row["bias"]),
                               summary.bland_altman.bias, 6)
        self.assertAlmostEqual(float(row["loa_upper"]),
                               summary.bland_altman.loa_upper, 6)
        self.assertEqual(int(row["n"]), summary.n)
        with (out / "validation_report_acceptance_criteria.csv").open(
                newline="", encoding="utf-8") as file:
            criteria = list(csv.DictReader(file))
        self.assertTrue({r["result"] for r in criteria}
                        <= {"pass", "fail", "not evaluable"})

    def test_file_inventory_has_sizes_and_hashes(self):
        _, outcome = self.build_full()
        files = {Path(f.path).name: f for f in outcome["results"].input_files}
        entry = files["ym1_manual_arm_entries.csv"]
        path = self.manual / "ym1_manual_arm_entries.csv"
        self.assertEqual(entry.size_bytes, path.stat().st_size)
        self.assertEqual(entry.sha256,
                         hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(entry.role, "manual_arm_entries")

    def test_a_bad_manual_file_becomes_a_warning_not_a_failure(self):
        self.make_y_maze(4)
        (self.manual / "ym2_manual_arm_entries.csv").write_text(
            "entry_number,arm,timestamp_s\n1,arm_z,1.0\n", encoding="utf-8")
        outcome = vr.build_report(self.inputs(), self.config(),
                                  self.root / "out")
        results = outcome["results"]
        self.assertEqual(results.paradigms["y_maze"].n_videos, 3)
        self.assertTrue(any("ym2" in w for w in results.warnings))
        html = outcome["html"].read_text(encoding="utf-8")
        self.assertIn("Warnings", html)

    def test_report_without_any_data_still_builds(self):
        outcome = vr.build_report([], self.config(), self.root / "empty")
        html = outcome["html"].read_text(encoding="utf-8")
        self.assertIn("Acceptance criteria", html)
        self.assertNotIn("http", html.lower())
        self.assertTrue(all(r.status == "not evaluable"
                            for r in outcome["results"].criteria))

    def test_fixing_the_matching_changes_the_report(self):
        self.make_y_maze(3)
        (self.manual / "ym3_manual_arm_entries.csv").unlink()
        stray = self.manual / "renamed.csv"
        write_entries(stray, self.y_manual["ym3"])
        inputs = self.inputs()
        before = vr.build_results(inputs, self.config())
        self.assertEqual(before.paradigms["y_maze"].n_videos, 2)
        by_stem = {v.stem: v for v in inputs}
        by_stem["ym3"].manual_arm_entries = stray       # the user's fix
        after = vr.build_results(
            [v for v in inputs if v.stem != "renamed"], self.config())
        self.assertEqual(after.paradigms["y_maze"].n_videos, 3)

    def test_manual_zone_intervals_and_totals_are_listed_in_inputs_table(self):
        self.make_open_field(1)
        results = vr.build_results(self.inputs(), self.config())
        table = next(t for t in vr.build_tables(results) if t.name == "inputs")
        self.assertIn("manual_zone_intervals", table.columns)
        self.assertEqual(table.rows[0][0], "of1")
        self.assertEqual(table.rows[0][1], "open_field")
        self.assertTrue(table.rows[0][table.columns.index(
            "manual_zone_intervals")].endswith("of1_manual_zone_intervals.csv"))


class ScoringRoundTripTests(ReportTestCase):
    def test_files_from_the_scoring_dialog_flow_into_a_report(self):
        """Manual-scoring writers -> discovery -> report."""
        self.make_y_maze(3)
        for stem, entries in self.y_manual.items():
            write_arm_entries_csv(
                self.manual / f"{stem}_manual_arm_entries.csv",
                [ScoreEvent(arm, int(round(t * FPS))) for arm, t in entries],
                FPS)
        results = vr.build_results(self.inputs(), self.config())
        self.assertEqual(results.paradigms["y_maze"].n_videos, 3)


if __name__ == "__main__":
    unittest.main()
