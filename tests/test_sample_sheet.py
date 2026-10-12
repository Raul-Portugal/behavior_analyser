"""Sample sheet (animal metadata): parsing, matching and the three outputs."""
import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.sample_sheet import (
    SampleSheetError, load_sample_sheet, metadata_for_videos,
    reserved_column_conflicts,
)

try:
    from core import cli
    from core.models import BatchSettings
    from core.output_contract import COMMON_BATCH_HEADERS
    from gui.worker import AnalysisWorker
    from mazes.freestyle import Freestyle
    from mazes.y_maze import YMaze, YMazeAnalysisResult
    from output import DataExporter
    from tests import worker_support as support
    from tests.worker_support import FRAME_COUNTS, data_files
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


class SheetFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()

    def write_sheet(self, text, name="sheet.csv", encoding="utf-8"):
        path = self.root / name
        path.write_bytes(text.encode(encoding))
        return path


class ParsingTests(SheetFixture):
    def test_reads_columns_and_rows_in_file_order(self):
        path = self.write_sheet(
            "video,animal_id,group,sex\n"
            "mouse1.mp4,A01,control,F\n"
            "mouse2.mp4,A02,treated,M\n"
        )
        sheet = load_sample_sheet(path)
        self.assertEqual(sheet.columns, ("animal_id", "group", "sex"))
        self.assertEqual([row.video for row in sheet.rows],
                         ["mouse1.mp4", "mouse2.mp4"])
        self.assertEqual(
            sheet.rows[1].values,
            {"animal_id": "A02", "group": "treated", "sex": "M"},
        )

    def test_bom_whitespace_and_blank_rows_are_handled(self):
        path = self.write_sheet(
            "﻿ video , animal_id ,group\r\n"
            "\r\n"
            "  mouse1.mp4 ,  A01 , control \r\n"
            " , , \r\n"
            "mouse2.mp4,A02,\r\n",
            encoding="utf-8",
        )
        sheet = load_sample_sheet(path)
        self.assertEqual(sheet.columns, ("animal_id", "group"))
        self.assertEqual(sheet.rows[0].values,
                         {"animal_id": "A01", "group": "control"})
        self.assertEqual(sheet.rows[1].values,
                         {"animal_id": "A02", "group": ""})

    def test_semicolon_and_tab_delimiters(self):
        semicolon = self.write_sheet("video;animal_id\nm.mp4;A1\n", "s.csv")
        tab = self.write_sheet("video\tanimal_id\nm.mp4\tA1\n", "t.csv")
        for path in (semicolon, tab):
            sheet = load_sample_sheet(path)
            self.assertEqual(sheet.columns, ("animal_id",))
            self.assertEqual(sheet.rows[0].values, {"animal_id": "A1"})

    def test_quoted_cells_may_contain_the_delimiter(self):
        path = self.write_sheet('video,note\nm.mp4,"a, b"\n')
        self.assertEqual(
            load_sample_sheet(path).rows[0].values, {"note": "a, b"}
        )

    def test_a_video_only_sheet_has_no_metadata_columns(self):
        sheet = load_sample_sheet(self.write_sheet("video\nm.mp4\n"))
        self.assertEqual(sheet.columns, ())

    def test_invalid_sheets_are_refused_with_a_reason(self):
        cases = {
            "first column": "animal_id,video\nA1,m.mp4\n",
            "twice": "video,group,Group\nm.mp4,a,b\n",
            "no name": "video,\nm.mp4,x\n",
            "empty": "\n\n",
            "no rows": "video,group\n",
            "listed twice": "video,group\nm.mp4,a\nm.mp4,b\n",
            "video column is empty": "video,group\n,a\n",
            "more values": "video,group\nm.mp4,a,b\n",
        }
        for fragment, text in cases.items():
            with self.subTest(fragment):
                path = self.write_sheet(text, "bad.csv")
                with self.assertRaises(SampleSheetError) as caught:
                    load_sample_sheet(path)
                self.assertIn(fragment, str(caught.exception))

    def test_missing_file_is_a_sheet_error(self):
        with self.assertRaises(SampleSheetError):
            load_sample_sheet(self.root / "absent.csv")

    def test_reserved_column_conflicts_ignore_case(self):
        self.assertEqual(
            reserved_column_conflicts(
                ["Paradigm", "group"], ["paradigm", "total_distance"]
            ),
            ["Paradigm"],
        )


class MatchingTests(SheetFixture):
    def _sheet(self, *rows):
        lines = ["video,animal_id"] + [f"{video},{animal}" for video, animal in rows]
        path = self.write_sheet("\n".join(lines) + "\n")
        return load_sample_sheet(path), path

    def test_matches_bare_filenames(self):
        sheet, path = self._sheet(("a.mp4", "A1"), ("b.mp4", "A2"))
        videos = [self.root / "x" / "a.mp4", self.root / "y" / "b.mp4"]
        match = metadata_for_videos(sheet, path, videos)
        self.assertTrue(match.is_complete)
        self.assertEqual(match.metadata.values_for(videos[0]),
                         {"animal_id": "A1"})
        self.assertEqual(match.metadata.values_for(videos[1]),
                         {"animal_id": "A2"})

    def test_relative_paths_tell_duplicate_filenames_apart(self):
        sheet, path = self._sheet(
            ("day1/mouse1.mp4", "A1"), ("day2/mouse1.mp4", "A2")
        )
        day1 = self.root / "day1" / "mouse1.mp4"
        day2 = self.root / "day2" / "mouse1.mp4"
        match = metadata_for_videos(sheet, path, [day2, day1])
        self.assertTrue(match.is_complete, match.problems())
        self.assertEqual(match.metadata.values_for(day1)["animal_id"], "A1")
        self.assertEqual(match.metadata.values_for(day2)["animal_id"], "A2")

    def test_a_bare_name_shared_by_two_videos_is_ambiguous_not_guessed(self):
        sheet, path = self._sheet(("mouse1.mp4", "A1"))
        day1 = self.root / "day1" / "mouse1.mp4"
        day2 = self.root / "day2" / "mouse1.mp4"
        match = metadata_for_videos(sheet, path, [day1, day2])
        self.assertEqual({video for video, _ in match.ambiguous}, {day1, day2})
        self.assertEqual(match.unmatched, [])
        self.assertEqual(match.metadata.by_video, {})
        self.assertEqual(match.metadata.values_for(day1), {"animal_id": ""})

    def test_a_filename_fitting_two_rows_is_ambiguous(self):
        sheet, path = self._sheet(
            ("day1/mouse1.mp4", "A1"), ("day2/mouse1.mp4", "A2")
        )
        elsewhere = self.root / "day3" / "mouse1.mp4"
        match = metadata_for_videos(sheet, path, [elsewhere])
        self.assertEqual([video for video, _ in match.ambiguous], [elsewhere])

    def test_unmatched_videos_and_unused_rows_are_reported(self):
        sheet, path = self._sheet(("a.mp4", "A1"), ("ghost.mp4", "A9"))
        known = self.root / "a.mp4"
        stray = self.root / "c.mp4"
        match = metadata_for_videos(sheet, path, [known, stray])
        self.assertFalse(match.is_complete)
        self.assertEqual([video for video, _ in match.unmatched], [stray])
        self.assertEqual(match.unused_rows, ["ghost.mp4"])
        self.assertIn("c.mp4", match.problems()[0])

    def test_absolute_paths_match(self):
        video = self.root / "deep" / "a.mp4"
        sheet, path = self._sheet((str(video), "A1"))
        match = metadata_for_videos(sheet, path, [video])
        self.assertTrue(match.is_complete)


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class BatchSummaryColumnTests(SheetFixture):
    def _results(self):
        return [
            (self.root / "a.mp4", YMazeAnalysisResult()),
            (self.root / "b.mp4", YMazeAnalysisResult()),
        ]

    def _metadata(self):
        sheet = load_sample_sheet(self.write_sheet(
            "video,animal_id,group\na.mp4,A1,ctl\nb.mp4,A2,trt\n"
        ))
        videos = [self.root / "a.mp4", self.root / "b.mp4"]
        return metadata_for_videos(sheet, sheet.path, videos).metadata

    def _read(self, out):
        with open(out / "batch_summary.csv", newline="", encoding="utf-8") as f:
            return list(csv.reader(f))

    def test_metadata_columns_sit_between_common_and_maze_columns(self):
        out = self.root / "with"
        DataExporter.export_batch_summary_csv(
            self._results(), out, YMaze(), metadata=self._metadata()
        )
        header, first, second = self._read(out)
        common = len(COMMON_BATCH_HEADERS)
        self.assertEqual(header[:common], COMMON_BATCH_HEADERS)
        self.assertEqual(header[common:common + 2], ["animal_id", "group"])
        self.assertEqual(header[common + 2:], YMaze().get_batch_summary_headers())
        self.assertEqual(first[common:common + 2], ["A1", "ctl"])
        self.assertEqual(second[common:common + 2], ["A2", "trt"])
        self.assertEqual(len({len(row) for row in (header, first, second)}), 1)

    def test_without_metadata_the_file_is_unchanged(self):
        out = self.root / "without"
        DataExporter.export_batch_summary_csv(self._results(), out, YMaze())
        header = self._read(out)[0]
        self.assertEqual(
            header, COMMON_BATCH_HEADERS + YMaze().get_batch_summary_headers()
        )
        stripped = self.root / "stripped"
        DataExporter.export_batch_summary_csv(
            self._results(), stripped, YMaze(), metadata=None
        )
        self.assertEqual(
            (out / "batch_summary.csv").read_bytes(),
            (stripped / "batch_summary.csv").read_bytes(),
        )


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class SampleSheetRunTests(SheetFixture):
    def setUp(self):
        super().setUp()
        self.video_dir = self.root / "videos"
        self.video_dir.mkdir()
        self.videos = [
            support.write_tracking_video(self.video_dir, frames, name)
            for name, frames in FRAME_COUNTS.items()
        ]
        self.sheet = self.write_sheet(
            "video,animal_id,group\n"
            + "".join(
                f"videos/{video.name},A{n},{'ctl' if n % 2 else 'trt'}\n"
                for n, video in enumerate(self.videos, 1)
            )
        )

    def _environment(self, label):
        cache = str(self.root / f"cache_{label}")
        return {"LOCALAPPDATA": cache, "XDG_CACHE_HOME": cache}

    def _worker_run(self, label, sample_metadata=None, workers=1):
        out = self.root / label
        out.mkdir()
        plan = {
            video.resolve(): support.tracking_settings(create_timelapse=False)
            for video in self.videos
        }
        for settings in plan.values():
            settings.paradigm_id = "Freestyle / Open Field"
            settings.algorithm_parameters = Freestyle().get_algorithm_parameters()
        with patch.dict(os.environ, self._environment(label)):
            worker = AnalysisWorker(
                plan, out, Freestyle(), workers=workers,
                sample_metadata=sample_metadata,
            )
            outcomes = []
            worker.finished.connect(outcomes.append)
            worker.run()
        self.assertEqual(outcomes[0].failures, [])
        return worker.output_dir

    def _metadata(self):
        sheet = load_sample_sheet(self.sheet)
        videos = [video.resolve() for video in self.videos]
        match = metadata_for_videos(sheet, self.sheet, videos)
        self.assertTrue(match.is_complete, match.problems())
        return match.metadata

    def test_metadata_reaches_summary_csv_json_and_manifest(self):
        run_dir = self._worker_run("sheet", self._metadata())

        with open(run_dir / "batch_summary.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(
            [(row["video_filename"], row["animal_id"], row["group"])
             for row in rows],
            [("mouse_a.avi", "A1", "ctl"), ("mouse_b.avi", "A2", "trt"),
             ("mouse_c.avi", "A3", "ctl")],
        )
        header = list(rows[0])
        self.assertEqual(header[len(COMMON_BATCH_HEADERS):][:2],
                         ["animal_id", "group"])

        summary = json.loads(
            (run_dir / "mouse_b" / "data" / "mouse_b_summary.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(summary["animal_metadata"],
                         {"animal_id": "A2", "group": "trt"})

        manifest = json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            [video["metadata"] for video in manifest["videos"]],
            [{"animal_id": "A1", "group": "ctl"},
             {"animal_id": "A2", "group": "trt"},
             {"animal_id": "A3", "group": "ctl"}],
        )
        self.assertEqual(Path(manifest["sample_sheet"]), self.sheet)

    def test_without_a_sheet_every_output_is_unchanged(self):
        plain = self._worker_run("plain")
        with_sheet = self._worker_run("sheet", self._metadata())

        plain_files = data_files(plain)
        sheet_files = data_files(with_sheet)
        self.assertEqual(sorted(plain_files), sorted(sheet_files))
        # Only the files that carry the metadata may differ.
        differing = {
            name for name in plain_files if plain_files[name] != sheet_files[name]
        }
        self.assertEqual(
            differing,
            {"batch_summary.csv"}
            | {f"{v.stem}/data/{v.stem}_summary.json" for v in self.videos},
        )
        for name in differing:
            self.assertNotIn("animal_id", plain_files[name].decode("utf-8"))

        manifest = json.loads(
            (plain / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertIsNone(manifest["sample_sheet"])
        self.assertTrue(all(v["metadata"] == {} for v in manifest["videos"]))

    def test_metadata_does_not_change_the_checkpoint_settings_hash(self):
        from checkpoint_manager import CheckpointManager
        manager = CheckpointManager(self.root / "ckpt", create_dir=False)
        (self.root / "o").mkdir()
        hashes = {}
        for label, metadata in (("plain", None), ("sheet", self._metadata())):
            plan = {
                video.resolve(): support.tracking_settings()
                for video in self.videos
            }
            AnalysisWorker(
                plan, self.root / "o", Freestyle(), sample_metadata=metadata
            )
            hashes[label] = [manager._get_settings_hash(s) for s in plan.values()]
            for settings in plan.values():
                self.assertNotIn("animal_id", json.dumps(settings.to_dict()))
        self.assertEqual(hashes["plain"], hashes["sheet"])

    def _cli(self, label, *extra):
        out = self.root / label
        plan_file = self.root / "plan.json"
        if not plan_file.exists():
            settings = support.tracking_settings(create_timelapse=False)
            settings.paradigm_id = "Freestyle / Open Field"
            settings.algorithm_parameters = Freestyle().get_algorithm_parameters()
            BatchSettings.save_batch_plan(
                {f"videos/{video.name}": settings.copy()
                 for video in self.videos},
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

    def test_cli_sample_sheet_flag_writes_the_metadata(self):
        code, out, stderr = self._cli("cli", "--sample-sheet", str(self.sheet))
        self.assertEqual(code, cli.EXIT_OK, stderr)
        (run_dir,) = sorted(out.glob("analysis_run_*"))
        with open(run_dir / "batch_summary.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([row["animal_id"] for row in rows], ["A1", "A2", "A3"])

    def test_cli_refuses_unmatched_videos_and_clashing_columns(self):
        partial = self.write_sheet(
            "video,animal_id\nvideos/mouse_a.avi,A1\n", "partial.csv"
        )
        code, out, stderr = self._cli("cli_partial", "--sample-sheet", str(partial))
        self.assertEqual(code, cli.EXIT_BAD_INPUT)
        self.assertIn("mouse_b.avi", stderr)
        self.assertFalse(out.exists())

        clash = self.write_sheet(
            "video,paradigm\n"
            + "".join(f"videos/{v.name},x\n" for v in self.videos),
            "clash.csv",
        )
        code, _, stderr = self._cli("cli_clash", "--sample-sheet", str(clash))
        self.assertEqual(code, cli.EXIT_BAD_INPUT)
        self.assertIn("paradigm", stderr)

        code, _, stderr = self._cli(
            "cli_absent", "--sample-sheet", str(self.root / "nope.csv")
        )
        self.assertEqual(code, cli.EXIT_BAD_INPUT)
        self.assertIn("nope.csv", stderr)


try:
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from gui.run_options_panel import RunOptionsPanel
    from tests.gui_support import temp_app_settings
    HAS_GUI = True
except ModuleNotFoundError:
    HAS_GUI = False


@unittest.skipUnless(HAS_GUI, "GUI runtime is not installed")
class RunOptionsPanelTests(SheetFixture):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        self.settings = temp_app_settings(self)
        self.videos = [self.root / "a.mp4", self.root / "b.mp4"]
        self.sheet = self.write_sheet(
            "video,animal_id\na.mp4,A1\nb.mp4,A2\n"
        )
        from mazes import create_maze
        self.maze = create_maze("Y-Maze")

    def _panel(self):
        panel = RunOptionsPanel(
            self.settings, lambda: list(self.videos), lambda: self.maze
        )
        self.addCleanup(panel.deleteLater)
        return panel

    def _choose(self, panel, path):
        with patch("gui.run_options_panel.QFileDialog.getOpenFileName",
                   return_value=(str(path), "")),                 patch("gui.run_options_panel.QMessageBox.warning") as warn,                 patch("gui.run_options_panel.QMessageBox.information"):
            panel.choose_sheet()
        return warn

    def test_time_bin_spinbox_is_off_by_default_and_gives_seconds(self):
        panel = self._panel()
        self.assertEqual(panel.bin_seconds(), 0.0)
        self.assertEqual(panel.bins_spin.specialValueText(), "Off")
        panel.bins_spin.setValue(5)
        self.assertEqual(panel.bin_seconds(), 300.0)

    def test_no_sheet_means_no_metadata(self):
        panel = self._panel()
        self.assertEqual(panel.resolve_for_run(self.videos), (True, None))
        self.assertEqual(panel.sheet_label.text(), "No sample sheet")

    def test_choosing_a_sheet_validates_remembers_and_clears(self):
        panel = self._panel()
        warn = self._choose(panel, self.sheet)
        warn.assert_not_called()
        self.assertIn("all 2 video(s) matched", panel.sheet_label.text())
        self.assertEqual(self.settings.sample_sheet(), self.sheet)

        # A new panel (next session) restores the last sheet.
        self.assertIsNotNone(self._panel().sheet)

        proceed, metadata = panel.resolve_for_run(self.videos)
        self.assertTrue(proceed)
        self.assertEqual(metadata.values_for(self.videos[1]),
                         {"animal_id": "A2"})

        panel.clear_sheet()
        self.assertIsNone(self.settings.sample_sheet())
        self.assertEqual(panel.resolve_for_run(self.videos), (True, None))

    def test_unmatched_videos_are_listed_when_the_sheet_loads(self):
        self.videos.append(self.root / "c.mp4")
        panel = self._panel()
        warn = self._choose(panel, self.sheet)
        warn.assert_called_once()
        self.assertIn("c.mp4", warn.call_args.args[2])
        self.assertIn("1 of 3", panel.sheet_label.text())

    def test_starting_with_unmatched_videos_asks_and_never_guesses(self):
        self.videos.append(self.root / "c.mp4")
        panel = self._panel()
        self._choose(panel, self.sheet)
        with patch("gui.run_options_panel.QMessageBox.question",
                   return_value=QMessageBox.StandardButton.No):
            self.assertEqual(panel.resolve_for_run(self.videos), (False, None))
        with patch("gui.run_options_panel.QMessageBox.question",
                   return_value=QMessageBox.StandardButton.Yes):
            proceed, metadata = panel.resolve_for_run(self.videos)
        self.assertTrue(proceed)
        self.assertEqual(metadata.values_for(self.videos[2]), {"animal_id": ""})

    def test_invalid_and_clashing_sheets_are_rejected(self):
        panel = self._panel()
        bad = self.write_sheet("animal_id,video\nA1,a.mp4\n", "bad.csv")
        warn = self._choose(panel, bad)
        warn.assert_called_once()
        self.assertIsNone(panel.sheet)
        clash = self.write_sheet("video,paradigm\na.mp4,x\n", "clash.csv")
        warn = self._choose(panel, clash)
        warn.assert_called_once()
        self.assertIsNone(panel.sheet)


if __name__ == "__main__":
    unittest.main()
