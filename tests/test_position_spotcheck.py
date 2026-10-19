import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.position_spotcheck import (
    SpotcheckSession, base_name_of, load_spotcheck_csv, read_scale_factor,
    sample_frames, scan_run_folder,
)
from core.positional_validation import (load_tracking_csv, position_errors)

try:
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtGui import QImage
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

    from gui.position_spotcheck_dialog import (PAGE_RESULTS, PAGE_REVIEW,
                                               PAGE_SETUP,
                                               PositionSpotcheckDialog)
    from tests.synthetic_video import write_indexed_video
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False

SCALE = 0.05
FRAMES = 30


def write_tracking_csv(path, frames, positions):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "timestamp", "x", "y", "raw_zone", "zone",
                         "visual_zone", "detected", "confidence",
                         "failure_reason"])
        for frame in frames:
            position = positions.get(frame)
            writer.writerow([
                frame, f"{frame / 25:.3f}",
                "" if position is None else f"{position[0]:.2f}",
                "" if position is None else f"{position[1]:.2f}",
                "center", "center", "center", 0 if position is None else 1,
                "0.9", ""])


def auto_positions(frames):
    """Frames divisible by 5 are undetected; the rest track a moving dot."""
    return {f: (None if f % 5 == 0 else (200.0 + 4 * f, 300.0 + f))
            for f in frames}


class SamplingTests(unittest.TestCase):
    def test_same_seed_same_frames_and_bounds(self):
        pool = range(500)
        a = sample_frames(pool, 50, 123)
        self.assertEqual(a, sample_frames(pool, 50, 123))
        self.assertNotEqual(a, sample_frames(pool, 50, 124))
        self.assertEqual(len(set(a)), 50)
        self.assertTrue(set(a) <= set(pool))
        self.assertEqual(sorted(sample_frames(range(7), 50, 1)), list(range(7)))
        self.assertEqual(sample_frames([], 5, 1), [])
        self.assertEqual(sample_frames([3, 3, 3], 5, 1), [3])

    def test_scale_factor_is_read_from_the_summary(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "s.json"
            path.write_text(json.dumps({"metadata": {"scale_factor": 0.2}}),
                            encoding="utf-8")
            self.assertEqual(read_scale_factor(path), 0.2)
            path.write_text(json.dumps(
                {"metadata": {}, "run_configuration": {"scale_factor": 0.3}}),
                encoding="utf-8")
            self.assertEqual(read_scale_factor(path), 0.3)
            path.write_text("{", encoding="utf-8")
            self.assertEqual(read_scale_factor(path), 0.0)
            self.assertEqual(read_scale_factor(Path(tmp) / "none.json"), 0.0)

    def test_base_name(self):
        self.assertEqual(base_name_of("x/a_b_tracking_data.csv"), "a_b")
        self.assertEqual(base_name_of("x/other.csv", "v/clip.mp4"), "clip")


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.positions = auto_positions(range(FRAMES))
        write_tracking_csv(self.root / "t.csv", range(FRAMES), self.positions)
        self.tracking = load_tracking_csv(self.root / "t.csv")

    def test_statistics_match_position_errors(self):
        session = SpotcheckSession(self.tracking, 12, 5, SCALE)
        manual = {}
        for index, sample in enumerate(session.samples):
            if index % 4 == 3:
                session.answer_not_visible(index)
                manual[sample.frame] = None
            else:
                auto = sample.auto_xy or (50.0, 60.0)
                point = (auto[0] + 3.0, auto[1] + 4.0)
                session.answer_click(index, *point)
                manual[sample.frame] = point
        self.assertTrue(session.is_complete)
        errors = session.errors()
        reference = position_errors(
            [self.positions[s.frame] for s in session.samples],
            [manual[s.frame] for s in session.samples], SCALE)
        self.assertEqual(errors.n_manual, reference.n_manual)
        self.assertEqual(errors.n_missed, reference.n_missed)
        self.assertEqual(errors.false_detections, reference.false_detections)
        self.assertAlmostEqual(errors.median_px, 5.0)
        self.assertAlmostEqual(errors.median_cm, 5.0 * SCALE)
        data = session.summary()
        self.assertEqual(data["seed"], 5)
        self.assertEqual(data["sampled_frames"], session.frames)
        self.assertEqual(data["reviewed"], 12)
        self.assertEqual(data["false_detections"], errors.false_detections)
        self.assertAlmostEqual(data["miss_rate"], errors.miss_rate)
        self.assertTrue(data["calibrated"])

    def test_outputs_round_trip(self):
        session = SpotcheckSession(self.tracking, 10, 9, SCALE)
        for index, sample in enumerate(session.samples):
            if index == 0:
                session.answer_not_visible(index)
            else:
                session.answer_click(index, 100.0 + index, 50.0)
        csv_path, json_path = session.write_outputs(
            self.root / "out", "clip", {"scorer_initials": "AB"})
        self.assertEqual(csv_path.name, "clip_position_spotcheck.csv")
        self.assertEqual(json_path.name, "clip_position_spotcheck_summary.json")
        with csv_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0]["visible"], "0")
        self.assertEqual(rows[0]["manual_x"], "")
        for row, sample in zip(rows, session.samples):
            self.assertEqual(int(row["frame"]), sample.frame)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(data["scorer_initials"], "AB")
        self.assertEqual(data["results"]["seed"], 9)
        self.assertIn("app_version", data)

        loaded = load_spotcheck_csv(csv_path)
        self.assertEqual(loaded.scale_factor, SCALE)
        reference = position_errors(loaded.auto_xy, loaded.manual_xy, SCALE)
        self.assertEqual(reference.n_manual, session.errors().n_manual)
        self.assertEqual(reference.n_matched, session.errors().n_matched)
        np.testing.assert_allclose(reference.errors_px, session.errors().errors_px,
                                   atol=0.01)

    def test_outcomes_and_unreadable_frames_are_excluded(self):
        session = SpotcheckSession(self.tracking, FRAMES, 1, SCALE)
        by_frame = {s.frame: i for i, s in enumerate(session.samples)}
        i_missed, i_false = by_frame[5], by_frame[6]
        i_absent, i_ok = by_frame[10], by_frame[7]
        session.answer_click(i_missed, 10, 10)          # tracker had nothing
        session.answer_not_visible(i_false)             # tracker had a point
        session.answer_not_visible(i_absent)            # both absent
        session.answer_click(i_ok, 1, 1)
        session.mark_unreadable(by_frame[8])
        self.assertEqual(session.samples[i_missed].outcome, "missed")
        self.assertEqual(session.samples[i_false].outcome, "false_detection")
        self.assertEqual(session.samples[i_absent].outcome, "correct_absent")
        self.assertEqual(session.samples[i_ok].outcome, "matched")
        self.assertEqual(session.samples[by_frame[8]].outcome, "unreadable")
        errors = session.errors()
        self.assertEqual(errors.n_manual, 2)
        self.assertEqual(errors.n_missed, 1)
        self.assertEqual(errors.false_detections, 1)
        self.assertFalse(session.is_complete)
        self.assertEqual(session.summary()["unreadable"], 1)

    def test_loader_rejects_other_files(self):
        path = self.root / "bad.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_spotcheck_csv(path)

    def test_uncalibrated_run_has_no_cm_errors(self):
        session = SpotcheckSession(self.tracking, 5, 1, 0.0)
        for index, sample in enumerate(session.samples):
            session.answer_click(index, 1.0, 1.0)
        data = session.summary()
        self.assertFalse(data["calibrated"])
        self.assertIsNone(data["median_error_cm"])

    def test_empty_tracking_is_refused(self):
        with self.assertRaises(ValueError):
            SpotcheckSession(type(self.tracking)(), 5, 1)

    def test_scan_run_folder_finds_videos_sources_and_summaries(self):
        run = self.root / "analysis_run_1"
        data_dir = run / "clip1" / "data"
        data_dir.mkdir(parents=True)
        write_tracking_csv(data_dir / "clip1_tracking_data.csv", range(3), {})
        (data_dir / "clip1_summary.json").write_text("{}", encoding="utf-8")
        (run / "clip2").mkdir()
        write_tracking_csv(run / "clip2" / "clip2_tracking_data.csv",
                           range(3), {})
        source = self.root / "clip1.avi"
        source.write_bytes(b"x")
        (run / "manifest.json").write_text(json.dumps({"videos": [
            {"source": str(source), "output_folder": "clip1"},
            {"source": str(self.root / "gone.avi"), "output_folder": "clip2"},
        ]}), encoding="utf-8")
        found = scan_run_folder(run)
        self.assertEqual([f.base_name for f in found], ["clip1", "clip2"])
        self.assertEqual(found[0].source_video, source)
        self.assertIsNotNone(found[0].summary_json)
        self.assertIsNone(found[1].source_video)       # file does not exist
        self.assertIsNone(found[1].summary_json)
        self.assertEqual(scan_run_folder(self.root / "nothing"), [])


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class SpotcheckDialogTests(unittest.TestCase):
    WIDTH, HEIGHT = 1600, 900

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.video = write_indexed_video(
            self.root / "clip1.avi", FRAMES, size=(self.WIDTH, self.HEIGHT))
        self.positions = auto_positions(range(FRAMES + 10))
        # Tracking rows extend past the end of the video on purpose.
        self.tracking = self.root / "clip1_tracking_data.csv"
        write_tracking_csv(self.tracking, range(FRAMES + 10), self.positions)
        self.summary = self.root / "clip1_summary.json"
        self.summary.write_text(
            json.dumps({"metadata": {"scale_factor": SCALE}}),
            encoding="utf-8")
        self.out = self.root / "results"

    def use_clean_tracking(self):
        """Tracking rows only for frames the video really has."""
        write_tracking_csv(self.tracking, range(FRAMES), self.positions)

    def make_dialog(self, count=8, seed=11):
        dialog = PositionSpotcheckDialog()
        self.addCleanup(dialog.deleteLater)
        self.addCleanup(dialog.done, 0)
        dialog.video_edit.setText(str(self.video))
        dialog.tracking_edit.setText(str(self.tracking))
        dialog.count_spin.setValue(count)
        dialog.seed_spin.setValue(seed)
        dialog.initials_edit.setText("ab")
        dialog.output_edit.setText(str(self.out))
        return dialog

    def click_source(self, dialog, x, y):
        """Click the display position that maps to source (x, y)."""
        label = dialog.frame_label
        pos = QPoint(round(x / label.display_to_source_x),
                     round(y / label.display_to_source_y))
        QTest.mouseClick(label, Qt.MouseButton.LeftButton, pos=pos)
        return (pos.x() * label.display_to_source_x,
                pos.y() * label.display_to_source_y)

    def test_summary_json_is_picked_up_next_to_the_tracking_csv(self):
        dialog = self.make_dialog()
        self.assertEqual(dialog.summary_edit.text(), str(self.summary))
        self.assertEqual(dialog.pages.currentIndex(), PAGE_SETUP)
        self.assertEqual(dialog.seed_spin.value(), 11)

    def test_full_review_matches_direct_core_calls_and_saves_files(self):
        dialog = self.make_dialog(count=8, seed=11)
        self.assertTrue(dialog.start())
        self.assertEqual(dialog.pages.currentIndex(), PAGE_REVIEW)
        session = dialog.session
        expected_frames = sample_frames(range(FRAMES + 10), 8, 11)
        expected_readable = [f for f in expected_frames if f < FRAMES]
        self.assertEqual(session.frames, expected_frames)
        self.assertEqual(session.scale_factor, SCALE)

        manual = {}
        step = 0
        while dialog.pages.currentIndex() == PAGE_REVIEW:
            sample = session.samples[dialog._index]
            if step % 3 == 2:
                dialog.not_visible()
                manual[sample.frame] = None
            else:
                auto = sample.auto_xy or (400.0, 300.0)
                manual[sample.frame] = self.click_source(
                    dialog, auto[0] + 5, auto[1] + 10)
            step += 1
        self.assertEqual(dialog.pages.currentIndex(), PAGE_RESULTS)
        self.assertEqual(step, len(expected_readable))
        # Unreadable frames (beyond the end of the video) were skipped.
        self.assertEqual(
            [s.frame for s in session.samples if s.status != "unreadable"],
            expected_readable)

        csv_path, json_path = dialog.saved_files
        self.assertEqual(csv_path, self.out / "clip1_position_spotcheck.csv")
        loaded = load_spotcheck_csv(csv_path)
        reference = position_errors(
            [self.positions[f] for f in manual],
            [manual[f] for f in manual], SCALE)
        result = json.loads(json_path.read_text(encoding="utf-8"))["results"]
        self.assertEqual(result["visible"], reference.n_manual)
        self.assertEqual(result["missed"], reference.n_missed)
        self.assertEqual(result["false_detections"], reference.false_detections)
        self.assertAlmostEqual(result["median_error_px"], reference.median_px,
                               places=2)
        self.assertAlmostEqual(result["median_error_cm"], reference.median_cm,
                               places=3)
        pooled = position_errors(loaded.auto_xy, loaded.manual_xy, SCALE)
        self.assertEqual(pooled.n_matched, reference.n_matched)
        self.assertIn("Spot-check results", dialog.results_label.text())
        self.assertIn("seed", dialog.results_label.text().lower())
        provenance = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(provenance["scorer_initials"], "AB")

    def test_clicks_are_mapped_to_source_pixels(self):
        self.use_clean_tracking()
        dialog = self.make_dialog(count=3, seed=2)
        dialog.count_spin.setValue(3)
        self.assertTrue(dialog.start())
        label = dialog.frame_label
        self.assertEqual(label.width(), 1280)
        self.assertAlmostEqual(label.display_to_source_x, 1600 / 1280)
        self.assertAlmostEqual(label.display_to_source_y, 900 / 720)
        point = self.click_source(dialog, 800, 450)
        self.assertEqual(point, (800.0, 450.0))
        first = dialog.session.samples[0]
        self.assertEqual(first.manual_xy, (800.0, 450.0))

    def test_frame_is_shown_without_the_automated_dot(self):
        self.use_clean_tracking()
        dialog = self.make_dialog(count=3, seed=2)
        self.assertTrue(dialog.start())
        sample = dialog.session.samples[dialog._index]
        import cv2
        from gui.qt_image import gray_or_bgr_to_qimage
        cap = cv2.VideoCapture(str(self.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample.frame)
        ok, raw = cap.read()
        cap.release()
        self.assertTrue(ok)
        expected = gray_or_bgr_to_qimage(cv2.resize(raw, (1280, 720)))
        shown = dialog.frame_label.pixmap().toImage()
        rgb = QImage.Format.Format_RGB888
        self.assertEqual(shown.convertToFormat(rgb), expected.convertToFormat(rgb))

    def test_back_replaces_an_answer_and_keys_work(self):
        self.use_clean_tracking()
        dialog = self.make_dialog(count=4, seed=3)
        self.assertTrue(dialog.start())
        self.click_source(dialog, 100, 100)
        self.assertEqual(dialog._index, 1)
        QTest.keyClick(dialog, Qt.Key.Key_Backspace)
        self.assertEqual(dialog._index, 0)
        self.click_source(dialog, 200, 250)
        self.assertEqual(dialog.session.samples[0].manual_xy, (200.0, 250.0))
        QTest.keyClick(dialog, Qt.Key.Key_N)
        self.assertEqual(dialog.session.samples[1].status, "not_visible")
        self.assertEqual(dialog._index, 2)

    def test_going_back_to_answer_the_rest_finishes_when_complete(self):
        self.use_clean_tracking()
        dialog = self.make_dialog(count=3, seed=4)
        dialog.count_spin.setValue(3)
        self.assertTrue(dialog.start())
        total = len([s for s in dialog.session.samples
                     if s.status != "unreadable"])
        for _ in range(total):
            dialog.not_visible()
        self.assertEqual(dialog.pages.currentIndex(), PAGE_RESULTS)

    def test_start_validates_inputs(self):
        dialog = PositionSpotcheckDialog()
        self.addCleanup(dialog.deleteLater)
        self.assertFalse(dialog.start())
        self.assertIn("video", dialog.message_label.text().lower())
        dialog.video_edit.setText(str(self.video))
        self.assertFalse(dialog.start())
        self.assertIn("tracking", dialog.message_label.text().lower())
        dialog.tracking_edit.setText(str(self.root / "wrong.csv"))
        self.assertFalse(dialog.start())
        bad = self.root / "bad_tracking_data.csv"
        bad.write_text("a,b\n1,2\n", encoding="utf-8")
        dialog.tracking_edit.setText(str(bad))
        self.assertFalse(dialog.start())
        self.assertIn("Cannot use the tracking data",
                      dialog.message_label.text())
        self.assertEqual(dialog.pages.currentIndex(), PAGE_SETUP)

    def test_run_folder_fills_the_inputs(self):
        run = self.root / "analysis_run_1"
        data_dir = run / "clip1" / "data"
        data_dir.mkdir(parents=True)
        write_tracking_csv(data_dir / "clip1_tracking_data.csv",
                           range(FRAMES), self.positions)
        (data_dir / "clip1_summary.json").write_text(
            json.dumps({"metadata": {"scale_factor": 0.1}}), encoding="utf-8")
        (run / "manifest.json").write_text(json.dumps({"videos": [
            {"source": str(self.video), "output_folder": "clip1"}]}),
            encoding="utf-8")
        dialog = PositionSpotcheckDialog()
        self.addCleanup(dialog.deleteLater)
        self.assertEqual(dialog.load_run_folder(str(run)), 1)
        self.assertEqual(dialog.video_edit.text(), str(self.video))
        self.assertTrue(dialog.tracking_edit.text().endswith(
            "clip1_tracking_data.csv"))
        self.assertTrue(dialog.summary_edit.text().endswith("clip1_summary.json"))
        self.assertEqual(dialog.validation_problem(), "")
        self.assertEqual(dialog.load_run_folder(str(self.root / "none")), 0)

    def test_capture_is_released_on_close(self):
        self.use_clean_tracking()
        dialog = self.make_dialog(count=3, seed=2)
        self.assertTrue(dialog.start())
        self.assertIsNotNone(dialog._cap)
        dialog.done(0)
        self.assertIsNone(dialog._cap)

    def test_same_seed_gives_the_same_sample_twice(self):
        first = self.make_dialog(count=6, seed=77)
        first.start()
        second = self.make_dialog(count=6, seed=77)
        second.start()
        self.assertEqual(first.session.frames, second.session.frames)
        other = self.make_dialog(count=6, seed=78)
        other.start()
        self.assertNotEqual(first.session.frames, other.session.frames)
        self.assertFalse(math.isnan(first.session.samples[0].timestamp_s))


if __name__ == "__main__":
    unittest.main()
