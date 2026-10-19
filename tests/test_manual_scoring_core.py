import json
import tempfile
import unittest
from pathlib import Path

from core.manual_scoring import (
    Bout, CENTER, ManualScoreSession, PARADIGM_EPM, PARADIGM_OPEN_FIELD,
    PARADIGM_TST, PARADIGM_Y_MAZE, PERIPHERY, ScoreEvent, ZoneInterval,
    allowed_event_labels, center_time_s, epm_time_totals, load_epm_totals,
    load_zone_intervals, output_names, sanitize_initials, sidecar_name,
    tst_frame_labels, write_arm_entries_csv, write_tst_annotations_csv,
    write_zone_intervals_csv, zone_at, zone_entries, zone_frame_labels,
    zone_intervals,
)
from core.positional_validation import load_entry_annotations
from core.tst_validation import load_annotations

FPS = 25.0


def make_session(paradigm, frames=100, **kwargs):
    return ManualScoreSession(
        paradigm, "video1.mp4", FPS, frames, kwargs.pop("initials", "ab"),
        **kwargs
    )


class ZoneIntervalTests(unittest.TestCase):
    def test_intervals_tile_the_window(self):
        events = [ScoreEvent(CENTER, 10), ScoreEvent(PERIPHERY, 30),
                  ScoreEvent(CENTER, 50)]
        spans = zone_intervals(events, PERIPHERY, FPS, 0, 80)
        self.assertEqual(
            [(s.zone, s.start_frame, s.end_frame) for s in spans],
            [(PERIPHERY, 0, 9), (CENTER, 10, 29),
             (PERIPHERY, 30, 49), (CENTER, 50, 79)],
        )
        self.assertAlmostEqual(sum(s.duration_s for s in spans), 80 / FPS)
        self.assertAlmostEqual(center_time_s(spans), 50 / FPS)
        self.assertEqual(spans[1].start_s, 10 / FPS)
        self.assertEqual(spans[1].end_s, 30 / FPS)

    def test_initial_zone_and_redundant_changes(self):
        events = [ScoreEvent(CENTER, 5), ScoreEvent(CENTER, 8)]
        spans = zone_intervals(events, CENTER, FPS, 0, 20)
        self.assertEqual([(s.zone, s.start_frame, s.end_frame) for s in spans],
                         [(CENTER, 0, 19)])

    def test_two_changes_on_one_frame_leave_no_empty_interval(self):
        events = [ScoreEvent(CENTER, 10), ScoreEvent(PERIPHERY, 10)]
        spans = zone_intervals(events, PERIPHERY, FPS, 0, 30)
        self.assertEqual([(s.zone, s.start_frame, s.end_frame) for s in spans],
                         [(PERIPHERY, 0, 29)])

    def test_window_clips_and_uses_earlier_state(self):
        events = [ScoreEvent(CENTER, 10), ScoreEvent(PERIPHERY, 40)]
        spans = zone_intervals(events, PERIPHERY, FPS, 20, 60)
        self.assertEqual([(s.zone, s.start_frame, s.end_frame) for s in spans],
                         [(CENTER, 20, 39), (PERIPHERY, 40, 59)])
        self.assertEqual(zone_intervals(events, PERIPHERY, FPS, 20, 20), [])

    def test_entries_exclude_starting_in_center(self):
        events = [ScoreEvent(PERIPHERY, 10), ScoreEvent(CENTER, 20)]
        self.assertEqual(zone_entries(events, CENTER, 0, 50), 1)
        self.assertEqual(zone_entries([], CENTER, 0, 50), 0)

    def test_zone_at(self):
        events = [ScoreEvent(CENTER, 10), ScoreEvent(PERIPHERY, 20)]
        self.assertEqual(zone_at(events, 5), PERIPHERY)
        self.assertEqual(zone_at(events, 10), CENTER)
        self.assertEqual(zone_at(events, 19), CENTER)
        self.assertEqual(zone_at(events, 20), PERIPHERY)

    def test_interval_csv_round_trip(self):
        spans = zone_intervals(
            [ScoreEvent(CENTER, 10), ScoreEvent(PERIPHERY, 33)],
            PERIPHERY, FPS, 0, 80)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "zones.csv"
            write_zone_intervals_csv(path, spans)
            header = path.read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(
                header, "interval_number,zone,start_frame,end_frame,"
                        "start_s,end_s,duration_s")
            loaded = load_zone_intervals(path)
        self.assertEqual(len(loaded), len(spans))
        for a, b in zip(spans, loaded):
            self.assertEqual((a.zone, a.start_frame, a.end_frame),
                             (b.zone, b.start_frame, b.end_frame))
            self.assertAlmostEqual(a.start_s, b.start_s, places=4)
            self.assertAlmostEqual(a.end_s, b.end_s, places=4)
        self.assertAlmostEqual(center_time_s(loaded), 23 / FPS, places=3)
        labels = zone_frame_labels(loaded)
        self.assertEqual(len(labels), 80)
        self.assertEqual(labels[9], PERIPHERY)
        self.assertEqual(labels[10], CENTER)
        self.assertEqual(labels[33], PERIPHERY)

    def test_interval_loader_rejects_other_csv(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "x.csv"
            path.write_text("a,b\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_zone_intervals(path)


class EpmTotalsTests(unittest.TestCase):
    def test_times_follow_the_marked_sequence(self):
        events = [ScoreEvent("open_arm_1", 10), ScoreEvent(CENTER, 20),
                  ScoreEvent("closed_arm_1", 30), ScoreEvent(CENTER, 40)]
        totals = epm_time_totals(events, 80, FPS)
        self.assertAlmostEqual(totals["time_in_open_arms_s"], 10 / FPS)
        self.assertAlmostEqual(totals["time_in_closed_arms_s"], 10 / FPS)
        self.assertAlmostEqual(totals["time_in_center_s"], 50 / FPS)
        self.assertAlmostEqual(totals["percent_time_in_open"], 50.0)

    def test_without_center_markers_no_time_is_derived(self):
        events = [ScoreEvent("open_arm_1", 10), ScoreEvent("closed_arm_1", 30)]
        self.assertIsNone(epm_time_totals(events, 80, FPS))

    def test_totals_csv_round_trip(self):
        from core.manual_scoring import write_epm_totals_csv
        totals = {"time_in_open_arms_s": 1.5, "percent_time_in_open": 33.3333}
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "t.csv"
            write_epm_totals_csv(path, totals)
            loaded = load_epm_totals(path)
        self.assertAlmostEqual(loaded["time_in_open_arms_s"], 1.5)
        self.assertAlmostEqual(loaded["percent_time_in_open"], 33.3333, 4)


class TstLabelTests(unittest.TestCase):
    def test_bouts_expand_and_later_overrides_earlier(self):
        bouts = [Bout("m1", "immobile_static", 2, 6),
                 Bout("m1", "uncertain", 5, 8)]
        labels = tst_frame_labels(bouts, ["m1"], 0, 10)
        self.assertEqual(labels[("m1", 4)], "immobile_static")
        self.assertEqual(labels[("m1", 5)], "uncertain")
        self.assertEqual(labels[("m1", 8)], "uncertain")
        self.assertEqual(len(labels), 7)

    def test_fill_marks_unlabelled_window_frames_active(self):
        bouts = [Bout("m1", "immobile_static", 2, 4)]
        labels = tst_frame_labels(bouts, ["m1", "m2"], 0, 6, "active_escape")
        self.assertEqual(labels[("m1", 0)], "active_escape")
        self.assertEqual(labels[("m1", 3)], "immobile_static")
        self.assertEqual(labels[("m2", 5)], "active_escape")
        self.assertEqual(len(labels), 12)

    def test_annotation_csv_is_read_by_the_validation_loader(self):
        labels = {("m1", 3): "immobile_static", ("m1", 1): "passive_swing",
                  ("m2", 2): "uncertain"}
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "a.csv"
            self.assertEqual(write_tst_annotations_csv(path, labels), 3)
            self.assertEqual(path.read_text(encoding="utf-8").splitlines()[0],
                             "mouse_name,frame,state")
            self.assertEqual(load_annotations(path), labels)

    def test_unknown_state_is_refused(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            with self.assertRaises(ValueError):
                write_tst_annotations_csv(Path(tmp) / "a.csv", {("m", 1): "x"})


class SessionTests(unittest.TestCase):
    def test_events_stay_sorted_and_validate(self):
        session = make_session(PARADIGM_Y_MAZE)
        session.add_event("arm_b", 20)
        session.add_event("arm_a", 10)
        session.add_event("arm_c", 20)
        self.assertEqual([(e.label, e.frame) for e in session.events],
                         [("arm_a", 10), ("arm_b", 20), ("arm_c", 20)])
        with self.assertRaises(ValueError):
            session.add_event("open_arm_1", 5)
        with self.assertRaises(ValueError):
            session.add_event("arm_a", 100)
        with self.assertRaises(ValueError):
            session.add_event("arm_a", -1)

    def test_undo_restores_each_step(self):
        session = make_session(PARADIGM_Y_MAZE)
        self.assertIsNone(session.undo())
        session.add_event("arm_a", 1)
        session.add_event("arm_b", 2)
        session.delete_events([0])
        self.assertEqual([e.label for e in session.events], ["arm_b"])
        self.assertIn("delete", session.undo())
        self.assertEqual([e.label for e in session.events], ["arm_a", "arm_b"])
        session.undo()
        session.undo()
        self.assertEqual(session.events, [])
        self.assertFalse(session.can_undo())

    def test_bout_extension_and_undo(self):
        session = make_session(PARADIGM_TST)
        session.begin_bout("mouse_1", "immobile_static", 5)
        session.extend_bout(9)
        session.extend_bout(7)       # going back never shrinks the bout
        session.end_bout()
        self.assertEqual((session.bouts[0].start, session.bouts[0].end), (5, 9))
        session.extend_bout(20)      # no open bout: ignored
        self.assertEqual(session.bouts[0].end, 9)
        session.undo()
        self.assertEqual(session.bouts, [])

    def test_starting_a_new_bout_closes_the_previous_one(self):
        session = make_session(PARADIGM_TST)
        session.begin_bout("mouse_1", "immobile_static", 5)
        session.extend_bout(8)
        session.begin_bout("mouse_1", "active_escape", 9)
        self.assertEqual(session.bouts[0].end, 8)
        self.assertEqual(session.open_bout.state, "active_escape")

    def test_toggle_zone_alternates(self):
        session = make_session(PARADIGM_OPEN_FIELD)
        self.assertEqual(session.toggle_zone(10).label, CENTER)
        self.assertEqual(session.toggle_zone(30).label, PERIPHERY)
        self.assertEqual(session.toggle_zone(50).label, CENTER)
        session.initial_zone = CENTER
        self.assertEqual(session.zone_now(0), CENTER)

    def test_window_frames(self):
        session = make_session(PARADIGM_OPEN_FIELD)
        self.assertEqual(session.window_frames(), (0, 100))
        session.set_window(1.0, 3.0)
        self.assertEqual(session.window_frames(), (25, 75))
        with self.assertRaises(ValueError):
            session.set_window(2.0, 1.0)

    def test_arm_entry_outputs_round_trip(self):
        session = make_session(PARADIGM_Y_MAZE)
        for label, frame in (("arm_a", 10), (CENTER, 15), ("arm_b", 20),
                             ("arm_c", 33)):
            session.add_event(label, frame)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            written = session.write_outputs(tmp)
            self.assertEqual([p.name for p in written],
                             ["video1_manual_arm_entries.csv"])
            lines = written[0].read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "entry_number,arm,timestamp_s")
            self.assertEqual(len(lines), 4)          # center marker excluded
            loaded = load_entry_annotations(written[0], "y_maze")
        self.assertEqual(loaded, [("arm_a", 0.4), ("arm_b", 0.8),
                                  ("arm_c", 1.32)])
        self.assertFalse(session.dirty)

    def test_epm_outputs(self):
        session = make_session(PARADIGM_EPM)
        for label, frame in (("open_arm_1", 10), (CENTER, 20),
                             ("closed_arm_2", 30)):
            session.add_event(label, frame)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            written = session.write_outputs(tmp)
            names = sorted(p.name for p in written)
            self.assertEqual(names, ["video1_manual_arm_entries.csv",
                                     "video1_manual_epm_totals.csv"])
            entries = load_entry_annotations(
                Path(tmp) / "video1_manual_arm_entries.csv", "epm")
            self.assertEqual([arm for arm, _ in entries],
                             ["open_arm_1", "closed_arm_2"])
            totals = load_epm_totals(Path(tmp) / "video1_manual_epm_totals.csv")
            self.assertAlmostEqual(totals["time_in_open_arms_s"], 10 / FPS,
                                   places=3)

    def test_epm_without_center_removes_a_stale_totals_file(self):
        session = make_session(PARADIGM_EPM)
        session.add_event("open_arm_1", 10)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            stale = Path(tmp) / "video1_manual_epm_totals.csv"
            stale.write_text("metric,value\n", encoding="utf-8")
            session.write_outputs(tmp)
            self.assertFalse(stale.exists())

    def test_open_field_and_tst_outputs(self):
        of = make_session(PARADIGM_OPEN_FIELD)
        of.toggle_zone(10)
        of.toggle_zone(30)
        tst = make_session(PARADIGM_TST, mice=["a", "b"])
        tst.begin_bout("a", "immobile_static", 3)
        tst.extend_bout(5)
        tst.end_bout()
        tst.begin_bout("b", "active_escape", 0)
        tst.extend_bout(1)
        tst.end_bout()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            [zones] = of.write_outputs(tmp)
            self.assertEqual(zones.name, "video1_manual_zone_intervals.csv")
            self.assertAlmostEqual(
                center_time_s(load_zone_intervals(zones)), 20 / FPS, places=3)
            [annotations] = tst.write_outputs(tmp)
            self.assertEqual(annotations.name,
                             "video1_manual_tst_annotations.csv")
            self.assertEqual(
                load_annotations(annotations),
                {("a", 3): "immobile_static", ("a", 4): "immobile_static",
                 ("a", 5): "immobile_static", ("b", 0): "active_escape",
                 ("b", 1): "active_escape"})

    def test_sidecar_round_trip_and_provenance(self):
        session = make_session(PARADIGM_OPEN_FIELD, initials="a.b")
        session.toggle_zone(12)
        session.set_window(0.4, 3.0)
        session.initial_zone = CENTER
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / sidecar_name("video1", PARADIGM_OPEN_FIELD, "a.b")
            self.assertEqual(path.name,
                             "video1_manual_scoring_open_field_AB.json")
            session.save_sidecar(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["scorer_initials"], "AB")
            self.assertRegex(data["date"], r"^\d{4}-\d{2}-\d{2}$")
            self.assertTrue(data["blind"])
            self.assertEqual(data["paradigm"], PARADIGM_OPEN_FIELD)
            self.assertEqual(data["app_version"], __import__(
                "core.version", fromlist=["x"]).APP_VERSION)

            restored = make_session(PARADIGM_OPEN_FIELD, initials="AB")
            self.assertTrue(restored.load_sidecar(path))
            self.assertEqual([(e.label, e.frame) for e in restored.events],
                             [("center", 12)])
            self.assertEqual(restored.initial_zone, CENTER)
            self.assertEqual(restored.window_start_s, 0.4)
            self.assertEqual(restored.window_end_s, 3.0)
            self.assertTrue(restored.dirty)

    def test_sidecar_is_not_loaded_for_another_scorer_video_or_paradigm(self):
        session = make_session(PARADIGM_Y_MAZE)
        session.add_event("arm_a", 3)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "s.json"
            session.save_sidecar(path)
            other_scorer = make_session(PARADIGM_Y_MAZE, initials="zz")
            self.assertFalse(other_scorer.load_sidecar(path))
            self.assertEqual(other_scorer.events, [])
            other_paradigm = make_session(PARADIGM_EPM)
            self.assertFalse(other_paradigm.load_sidecar(path))
            other_length = make_session(PARADIGM_Y_MAZE, frames=50)
            self.assertFalse(other_length.load_sidecar(path))

    def test_misc_helpers(self):
        self.assertEqual(sanitize_initials(" a-b 1"), "AB1")
        self.assertEqual(sanitize_initials(""), "")
        self.assertEqual(allowed_event_labels(PARADIGM_TST), ())
        self.assertIn("epm_totals", output_names("v", PARADIGM_EPM))
        with self.assertRaises(ValueError):
            output_names("v", "nope")
        with self.assertRaises(ValueError):
            ManualScoreSession("nope", "v.mp4", 25, 10, "ab")
        with self.assertRaises(ValueError):
            ManualScoreSession(PARADIGM_Y_MAZE, "v.mp4", 0, 10, "ab")

    def test_write_arm_entries_counts_without_center(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            count = write_arm_entries_csv(
                Path(tmp) / "e.csv",
                [ScoreEvent(CENTER, 1), ScoreEvent("arm_a", 2)], FPS)
        self.assertEqual(count, 1)
        # ZoneInterval is a plain value object
        self.assertEqual(ZoneInterval("center", 0, 24, 0.0, 1.0).duration_s, 1.0)


if __name__ == "__main__":
    unittest.main()
