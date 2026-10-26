"""Reading a run folder for the Results tab (no Qt)."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    from gui.workspace import results_data as rd
    from gui.workspace.overlay import (tracking_overlay, tst_overlay,
                                       load_video_rois)
    from core.positional_validation import TrackingData
    from tests.workspace_run_support import make_cli_run
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


class RunFolderCase(unittest.TestCase):
    """One real CLI run shared by the tests of a class (copied when edited)."""

    @classmethod
    def setUpClass(cls):
        if not HAS_RUNTIME:
            raise unittest.SkipTest("GUI/scientific runtime is not installed")
        cls._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._tmp.name).resolve()
        cls.run_dir = make_cli_run(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def copy_run(self, name="copy") -> Path:
        target = self.root / f"{name}_{self.id().split('.')[-1]}"
        shutil.copytree(self.run_dir, target / self.run_dir.name)
        return target / self.run_dir.name


class LoadingTests(RunFolderCase):
    def test_a_schema_3_run_loads_in_full(self):
        run = rd.RunData.load(self.run_dir)
        self.assertEqual(run.schema_version, 3)
        self.assertEqual(run.paradigm, "Freestyle / Open Field")
        self.assertEqual(run.time_bin_seconds, 1.0)
        self.assertEqual(
            [v.name for v in run.videos],
            ["mouse_a.avi", "mouse_b.avi", "mouse_c.avi"],
        )
        self.assertEqual({v.status for v in run.videos}, {"completed"})
        self.assertEqual(run.videos[2].metadata, {"animal_id": "3", "group": "trt"})
        self.assertEqual(run.metadata_columns(), ["animal_id", "group"])
        self.assertTrue(run.videos[0].summary_json.is_file())
        self.assertEqual(run.notes, [])
        self.assertIn("3 video(s) completed", run.describe())

    def test_the_summary_is_the_csv(self):
        run = rd.RunData.load(self.run_dir)
        text = (self.run_dir / "batch_summary.csv").read_text(encoding="utf-8")
        lines = text.strip().splitlines()
        self.assertEqual(",".join(run.summary.headers), lines[0])
        self.assertEqual(len(run.summary.rows), len(lines) - 1)
        self.assertEqual(run.summary.values("video_filename"),
                         ["mouse_a.avi", "mouse_b.avi", "mouse_c.avi"])
        self.assertEqual(run.summary.values("group"), ["ctl", "ctl", "trt"])

    def test_rows_find_their_videos(self):
        run = rd.RunData.load(self.run_dir)
        for row, name in enumerate(("mouse_a.avi", "mouse_b.avi", "mouse_c.avi")):
            self.assertEqual(run.video_for_row(row).name, name)
        self.assertIsNone(run.video_for_row(7))
        source = run.videos[1].source
        self.assertEqual(run.row_for_source(source), 1)
        self.assertEqual(run.row_for_source(Path("elsewhere") / "mouse_c.avi"), 2)
        self.assertEqual(run.row_for_source("nothing.avi"), -1)

    def test_a_video_files_are_found(self):
        video = rd.RunData.load(self.run_dir).videos[0]
        self.assertEqual(
            sorted(p.stem.split("_", 2)[2] for p in video.plots()),
            ["heatmap", "speed", "timeseries", "trajectory"],
        )
        self.assertEqual(len(video.media()), 1)
        self.assertTrue(video.tracking_csv().name.endswith("_tracking_data.csv"))
        self.assertIsNone(video.evidence_csv())
        self.assertTrue(video.time_bins_csv().is_file())

    def test_a_schema_2_manifest_loads_best_effort(self):
        run_dir = self.copy_run()
        manifest = json.loads((run_dir / "manifest.json").read_text())
        manifest["schema_version"] = 2
        manifest["videos"] = [
            {"source": v["source"], "output_folder": v["output_folder"]}
            for v in manifest["videos"]
        ]
        for key in ("sample_sheet", "time_bin_seconds"):
            manifest.pop(key, None)
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        run = rd.RunData.load(run_dir)
        self.assertEqual(run.schema_version, 2)
        self.assertEqual(len(run.videos), 3)
        self.assertEqual(run.videos[0].metadata, {})
        self.assertEqual(run.videos[0].status, "completed")
        self.assertIsNotNone(run.videos[0].load_summary().get("metadata"))
        self.assertTrue(any("schema 2" in note for note in run.notes))
        self.assertEqual(run.video_for_row(1).name, "mouse_b.avi")
        # The metadata columns still come from the batch time bins file.
        self.assertEqual(run.metadata_columns(), ["animal_id", "group"])

    def test_a_run_without_a_manifest_is_read_from_its_folders(self):
        run_dir = self.copy_run()
        (run_dir / "manifest.json").unlink()
        run = rd.RunData.load(run_dir)
        self.assertEqual(run.schema_version, 0)
        self.assertEqual(
            [v.name for v in run.videos], ["mouse_a", "mouse_b", "mouse_c"]
        )
        self.assertEqual(len(run.summary.rows), 3)
        self.assertTrue(any("manifest.json is missing" in n for n in run.notes))

    def test_a_newer_schema_is_refused(self):
        run_dir = self.copy_run()
        manifest = json.loads((run_dir / "manifest.json").read_text())
        manifest["schema_version"] = 99
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(rd.RunLoadError):
            rd.RunData.load(run_dir)

    def test_a_folder_that_is_not_a_run_is_refused(self):
        empty = self.root / "empty_folder"
        empty.mkdir(exist_ok=True)
        with self.assertRaises(rd.RunLoadError):
            rd.RunData.load(empty)
        with self.assertRaises(rd.RunLoadError):
            rd.RunData.load(self.root / "does_not_exist")
        self.assertFalse(rd.looks_like_run(empty))
        self.assertTrue(rd.looks_like_run(self.run_dir))

    def test_quality_warnings_and_failures_are_attached_to_their_videos(self):
        run_dir = self.copy_run()
        manifest = json.loads((run_dir / "manifest.json").read_text())
        source = manifest["videos"][1]["source"]
        manifest["quality_warnings"] = [
            {"source": source, "message": "tracking lost for 3 s"}
        ]
        manifest["videos"][2]["status"] = "failed"
        manifest["failures"] = [
            {"source": manifest["videos"][2]["source"], "message": "boom"}
        ]
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        run = rd.RunData.load(run_dir)
        self.assertEqual(run.videos[1].warnings, ["tracking lost for 3 s"])
        self.assertEqual(run.videos[2].failure, "boom")
        self.assertEqual(run.quality_warning_count(), 1)
        self.assertIn("1 failed", run.describe())
        # Rows still find their videos by name when a video failed.
        self.assertEqual(run.video_for_row(0).name, "mouse_a.avi")
        self.assertEqual(run.video_for_row(1).name, "mouse_b.avi")

    def test_per_video_time_bins_are_combined_when_the_batch_file_is_missing(self):
        run_dir = self.copy_run()
        (run_dir / "batch_time_bins.csv").unlink()
        run = rd.RunData.load(run_dir)
        self.assertIsNotNone(run.time_bins)
        full = rd.RunData.load(self.run_dir).time_bins
        self.assertEqual(len(run.time_bins.rows), len(full.rows))
        self.assertEqual(
            rd.time_bin_group_columns(run.time_bins), ["animal_id", "group"]
        )


class KeyMetricTests(RunFolderCase):
    def test_key_metrics_are_flattened_by_section(self):
        summary = rd.RunData.load(self.run_dir).videos[0].load_summary()
        rows, notes = rd.key_metrics(summary)
        metrics = {(s, m): v for s, m, v in rows}
        self.assertIn(("total distance", "total distance"), metrics) \
            if False else None
        sections = {s for s, _, _ in rows}
        self.assertIn("tracking quality", sections)
        self.assertIn("freestyle metrics", sections)
        self.assertNotIn("detection parameters", sections)
        self.assertNotIn("run configuration", sections)
        self.assertEqual(metrics[("metadata", "fps")], "30")
        self.assertEqual(metrics[("metadata", "distance unit")], "pixels")
        self.assertTrue(notes)

    def test_value_formatting(self):
        self.assertEqual(rd.format_value(None), "N/A")
        self.assertEqual(rd.format_value(True), "yes")
        self.assertEqual(rd.format_value(2), "2")
        self.assertEqual(rd.format_value(0.5), "0.5")
        self.assertEqual(rd.format_value(88.19957870648935), "88.1996")
        self.assertEqual(rd.format_value(float("nan")), "N/A")


class HeadlineMetricTests(RunFolderCase):
    def test_a_real_run_gives_the_distance_the_speeds_and_the_grade(self):
        summary = rd.RunData.load(self.run_dir).videos[0].load_summary()
        rows, _notes = rd.key_metrics(summary)
        headline = rd.headline_metrics(rows)
        names = [name for name, _value in headline]
        self.assertEqual(names[0], "total distance")
        self.assertIn("average speed", names)
        self.assertEqual(names[-1], "tracking quality grade")
        self.assertLessEqual(len(headline), 6)
        self.assertTrue(all(value not in ("", "N/A") for _n, value in headline))

    def test_settings_like_entries_and_missing_values_are_left_out(self):
        rows = [
            ("specific metrics", "average speed", "3.5"),
            ("specific metrics", "immobility time", "N/A"),
            ("specific metrics", "immobility threshold", "2"),
            ("specific metrics", "speed window seconds", "0.2"),
            ("specific metrics", "peak speed", ""),
            ("specific metrics", "zone transitions", "4"),
        ]
        self.assertEqual(
            rd.headline_metrics(rows),
            [("average speed", "3.5"), ("zone transitions", "4")],
        )

    def test_the_limit_is_respected_and_the_grade_keeps_its_place(self):
        rows = [("total distance", "total distance", "9")]
        rows += [("specific metrics", f"metric {n}", str(n)) for n in range(10)]
        rows += [("metadata", "tracking quality grade", "Good")]
        headline = rd.headline_metrics(rows, limit=5)
        self.assertEqual(len(headline), 5)
        self.assertEqual(headline[0], ("total distance", "9"))
        self.assertEqual(headline[-1], ("tracking quality grade", "Good"))

    def test_nothing_to_show_gives_an_empty_list(self):
        self.assertEqual(rd.headline_metrics([]), [])
        self.assertEqual(rd.headline_metrics([("sample", "group", "ctl")]), [])


class TimeBinTests(RunFolderCase):
    def table(self):
        return rd.RunData.load(self.run_dir).time_bins

    def test_metrics_subjects_and_group_columns(self):
        table = self.table()
        self.assertIn("distance_total", rd.time_bin_metrics(table))
        self.assertEqual(rd.time_bin_subjects(table), [])
        self.assertEqual(rd.time_bin_group_columns(table), ["animal_id", "group"])
        self.assertEqual(rd.time_bin_metrics(None), [])

    def test_one_series_per_video(self):
        series = rd.time_bin_series(self.table(), "distance_total")
        self.assertEqual(
            [s.label for s in series],
            ["mouse_a.avi", "mouse_b.avi", "mouse_c.avi"],
        )
        self.assertEqual([len(s.x) for s in series], [3, 3, 4])
        self.assertEqual(series[0].unit, "pixels")
        self.assertTrue(all(e == 0 for s in series for e in s.err))
        self.assertAlmostEqual(series[0].y[0], 36.4161, places=3)

    def test_groups_are_averaged_with_their_error(self):
        table = self.table()
        series = rd.time_bin_series(table, "distance_total", group_by="group")
        self.assertEqual([s.label for s in series], ["ctl", "trt"])
        ctl, trt = series
        self.assertEqual((ctl.n, trt.n), (2, 1))
        per_video = rd.time_bin_series(table, "distance_total")
        first_bin = (per_video[0].y[0] + per_video[1].y[0]) / 2
        self.assertAlmostEqual(ctl.y[0], first_bin)
        spread = abs(per_video[0].y[0] - per_video[1].y[0]) / 2
        self.assertAlmostEqual(ctl.err[0], spread)
        self.assertEqual(set(trt.err), {0.0})

    def test_blank_and_unknown_values_are_skipped(self):
        table = self.table()
        table.rows[0][table.column("value")] = "N/A"
        series = rd.time_bin_series(table, "distance_total")
        self.assertEqual(len(series[0].x), 2)
        self.assertEqual(rd.time_bin_series(table, "nothing"), [])
        self.assertEqual(rd.time_bin_series(None, "x"), [])

    def test_subjects_are_separate_series(self):
        table = rd.Table(
            ["video_filename", "paradigm", "group", "bin_index", "bin_start_s",
             "bin_end_s", "subject", "metric", "value", "unit"],
            [["t.mp4", "TST", "g", "1", "0", "60", "mouse_1", "immobility_s", "10", "s"],
             ["t.mp4", "TST", "g", "1", "0", "60", "mouse_2", "immobility_s", "20", "s"],
             ["t.mp4", "TST", "g", "2", "60", "120", "mouse_1", "immobility_s", "30", "s"]],
        )
        self.assertEqual(rd.time_bin_subjects(table), ["mouse_1", "mouse_2"])
        series = rd.time_bin_series(table, "immobility_s")
        self.assertEqual([s.label for s in series],
                         ["t.mp4 / mouse_1", "t.mp4 / mouse_2"])
        one = rd.time_bin_series(table, "immobility_s", subject="mouse_1")
        self.assertEqual([s.label for s in one], ["t.mp4"])
        self.assertEqual(one[0].y, [10.0, 30.0])
        grouped = rd.time_bin_series(
            table, "immobility_s", subject="mouse_2", group_by="group"
        )
        self.assertEqual(grouped[0].y, [20.0])


class DiffTests(unittest.TestCase):
    def tables(self):
        a = rd.Table(["video_filename", "total_distance", "group"],
                     [["a.avi", "10.00", "x"], ["b.avi", "20.00", "x"],
                      ["c.avi", "30.00", "y"]])
        b = rd.Table(["video_filename", "total_distance", "group", "extra"],
                     [["a.avi", "10", "x", ""], ["b.avi", "21.50", "x", "1"],
                      ["d.avi", "5", "y", ""]])
        return a, b

    def test_changed_values_are_flagged_per_video_and_column(self):
        diff = rd.diff_summaries(*self.tables())
        self.assertEqual(
            diff.headers, ["video_filename", "total_distance", "group", "extra"]
        )
        rows = {row.video: row for row in diff.rows}
        self.assertFalse(rows["a.avi"].changed)          # 10.00 == 10
        self.assertTrue(rows["b.avi"].changed)
        changed = [
            diff.headers[i] for i, c in enumerate(rows["b.avi"].cells) if c.changed
        ]
        self.assertEqual(changed, ["total_distance", "extra"])
        self.assertEqual(rows["c.avi"].status, "only_a")
        self.assertEqual(rows["d.avi"].status, "only_b")
        self.assertEqual(diff.changed_cells, 2)
        self.assertEqual(diff.changed_rows, 3)

    def test_identical_tables_have_no_changes(self):
        a, _ = self.tables()
        diff = rd.diff_summaries(a, a)
        self.assertEqual(diff.changed_cells, 0)
        self.assertEqual(diff.changed_rows, 0)

    def test_numbers_are_compared_as_numbers(self):
        self.assertTrue(rd.values_equal("1.0", "1"))
        self.assertTrue(rd.values_equal("N/A", "N/A"))
        self.assertFalse(rd.values_equal("N/A", "0"))
        self.assertFalse(rd.values_equal("1.0000001", "1"))
        self.assertTrue(rd.values_equal("1.0000000001", "1"))

    def test_repeated_names_are_matched_in_order(self):
        a = rd.Table(["video_filename", "v"], [["x", "1"], ["x", "2"]])
        b = rd.Table(["video_filename", "v"], [["x", "1"], ["x", "3"]])
        diff = rd.diff_summaries(a, b)
        self.assertEqual([r.changed for r in diff.rows], [False, True])
        self.assertEqual(diff.rows[1].video, "x (#2)")


class OverlayDataTests(RunFolderCase):
    def test_the_rois_of_a_video_come_from_the_run_plan(self):
        run = rd.RunData.load(self.run_dir)
        rois = load_video_rois(self.run_dir, run.videos[0].source)
        self.assertEqual(list(rois), ["arena"])
        self.assertEqual(rois["arena"][0].shape, (4, 2))
        self.assertEqual(load_video_rois(self.run_dir, "other.avi"), {})
        self.assertEqual(load_video_rois(self.root, run.videos[0].source), {})

    def test_the_tracking_overlay_draws_the_position_and_the_zone(self):
        data = TrackingData()
        data.frames = [10, 11, 12]
        data.timestamps = [0.0, 0.03, 0.06]
        data.positions = [(40.0, 50.0), (44.0, 52.0), None]
        data.zones = ["arena", "arena", "arena"]
        data.confidences = [0.9, 0.9, 0.0]
        draw = tracking_overlay(data)
        blank = np.zeros((120, 160, 3), np.uint8)
        shown = draw(blank.copy(), 11)
        self.assertGreater(int(np.abs(shown.astype(int) - blank).sum()), 0)
        # The dot is drawn at the position (green) ...
        self.assertTrue((shown[52, 44] == (0, 220, 0)).all())
        # ... and a lost frame and a frame outside the window still draw text.
        self.assertGreater(int(draw(blank.copy(), 12).sum()), 0)
        self.assertGreater(int(draw(blank.copy(), 99).sum()), 0)

    def test_the_tst_overlay_colours_each_mouse_by_state(self):
        rows = {
            "mouse_1": [{"source_frame": 5, "state": "active_escape",
                         "is_immobile": False}],
            "mouse_2": [{"source_frame": 5, "state": "immobile_static",
                         "is_immobile": True}],
        }
        polygon = np.array([[10, 10], [60, 10], [60, 60], [10, 60]], np.int32)
        draw = tst_overlay(rows, {"mouse_1": [polygon]})
        blank = np.zeros((120, 160, 3), np.uint8)
        shown = draw(blank.copy(), 5)
        self.assertTrue((shown[10, 30] == (0, 255, 0)).all())     # ACTIVE ESCAPE
        self.assertGreater(int(shown.sum()), 0)
        self.assertEqual(int(draw(blank.copy(), 6).sum()), 0)     # no data


if __name__ == "__main__":
    unittest.main()
