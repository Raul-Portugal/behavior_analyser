"""The Results stage: run picker, table, per-video view, charts."""
import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.workspace_run_support import make_cli_run
from tests.workspace_support import HAS_RUNTIME, WorkspaceFixture

if HAS_RUNTIME:
    from PyQt6.QtCore import Qt
    from core.project import STATUS_DONE
    from gui.workspace import results_data as rd
    from gui.workspace.results_widgets import (ALL_SUBJECTS, NO_GROUP,
                                           SOURCE_OVERLAY, SOURCE_VALIDATION,
                                           PlotDialog, overlay_for_video)


class ResultsFixture(WorkspaceFixture):
    """A workspace plus one real CLI run (made once for the whole class)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._run_tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.run_root = Path(cls._run_tmp.name).resolve()
        cls.run_dir = make_cli_run(cls.run_root)

    @classmethod
    def tearDownClass(cls):
        cls._run_tmp.cleanup()

    def setUp(self):
        super().setUp()
        self.tab = self.window.stage("results")
        self.addCleanup(self.tab.shutdown)
        self.csv_rows = self.read_csv(self.run_dir / "batch_summary.csv")

    @staticmethod
    def read_csv(path):
        with open(path, newline="", encoding="utf-8-sig") as handle:
            return list(csv.reader(handle))

    def copy_run(self, name="copy"):
        target = self.root / name / self.run_dir.name
        shutil.copytree(self.run_dir, target)
        return target

    def open_project_with_videos(self):
        """A project over the run's real videos, with the run recorded."""
        self.new_project_at()
        self.window.video_panel.add_paths(
            sorted((self.run_root / "real").glob("*.avi"))
        )
        project = self.window.project
        project.record_run(self.run_dir)
        return project


class LoadingTests(ResultsFixture):
    def test_show_run_loads_the_cli_run(self):
        self.assertTrue(self.tab.show_run(self.run_dir))
        run = self.tab.run
        self.assertEqual(run.run_dir, self.run_dir)
        self.assertIn("3 video(s) completed", self.tab.info_label.text())
        self.assertEqual(self.tab.run_combo.currentData(), str(self.run_dir))

    def test_the_table_matches_the_csv(self):
        self.tab.show_run(self.run_dir)
        headers, rows = self.csv_rows[0], self.csv_rows[1:]
        self.assertEqual(self.tab.model.headers, headers)
        self.assertEqual(self.tab.proxy.columnCount(), len(headers))
        self.assertEqual(self.tab.table_rows(), rows)
        self.assertEqual(self.tab.rows_label.text(), "3 rows")
        self.assertIn("group", headers)             # the metadata columns

    def test_sorting_is_numeric_and_filtering_is_by_text(self):
        self.tab.show_run(self.run_dir)
        column = self.tab.model.headers.index("analyzed_frames")
        self.tab.table.sortByColumn(column, Qt.SortOrder.DescendingOrder)
        frames = [row[column] for row in self.tab.table_rows()]
        self.assertEqual(frames, ["110", "90", "70"])     # not "90" first
        self.tab.table.sortByColumn(column, Qt.SortOrder.AscendingOrder)
        self.assertEqual(
            [row[0] for row in self.tab.table_rows()],
            ["mouse_a.avi", "mouse_b.avi", "mouse_c.avi"],
        )
        self.tab.filter_edit.setText("trt")
        self.assertEqual([r[0] for r in self.tab.table_rows()], ["mouse_c.avi"])
        self.assertEqual(self.tab.rows_label.text(), "1 of 3 rows")
        self.tab.filter_edit.setText("")
        self.assertEqual(len(self.tab.table_rows()), 3)

    def test_export_writes_the_table_as_shown(self):
        self.tab.show_run(self.run_dir)
        self.tab.filter_edit.setText("ctl")
        target = self.root / "exported.csv"
        self.prompts.say("save_file", target)
        self.assertTrue(self.tab.export_csv())
        exported = self.read_csv(target)
        self.assertEqual(exported[0], self.csv_rows[0])
        self.assertEqual(exported[1:], self.csv_rows[1:3])

    def test_copying_cells_gives_tab_separated_text(self):
        self.tab.show_run(self.run_dir)
        self.tab.table.selectAll()
        text = self.tab.copy_selection()
        lines = text.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0].split("\t"), self.csv_rows[1])
        self.assertEqual(self.prompts.clipboard, [text])
        self.tab.copy_selection(with_headers=True)
        self.assertEqual(
            self.prompts.clipboard[-1].splitlines()[0].split("\t"),
            self.csv_rows[0],
        )

    def test_ctrl_c_copies(self):
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QKeyEvent
        self.tab.show_run(self.run_dir)
        self.tab.table.selectAll()
        event = QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_C,
            Qt.KeyboardModifier.ControlModifier,
        )
        self.tab.table.keyPressEvent(event)
        self.assertTrue(event.isAccepted())
        self.assertEqual(len(self.prompts.clipboard), 1)

    def test_open_run_folder_for_an_external_run(self):
        self.prompts.say("choose_folder", self.run_dir)
        self.assertTrue(self.tab.open_run_folder())
        self.assertEqual(self.tab.run.run_dir, self.run_dir)
        self.assertEqual(self.tab.run_paths(), [self.run_dir])
        self.assertIn("opened", self.tab.run_combo.itemText(0))

    def test_a_folder_that_is_not_a_run_is_reported(self):
        self.prompts.say("choose_folder", self.root)
        self.assertFalse(self.tab.open_run_folder())
        (error,) = self.prompts.called("error")
        self.assertIn("does not look like", error[1])
        self.assertIsNone(self.tab.run)

    def test_cancelling_the_folder_dialog_changes_nothing(self):
        self.tab.show_run(self.run_dir)
        self.prompts.say("choose_folder", None)
        self.assertFalse(self.tab.open_run_folder())
        self.assertIsNotNone(self.tab.run)

    def test_a_schema_2_run_loads(self):
        run_dir = self.copy_run("schema2")
        manifest = json.loads((run_dir / "manifest.json").read_text())
        manifest["schema_version"] = 2
        manifest["videos"] = [
            {"source": v["source"], "output_folder": v["output_folder"]}
            for v in manifest["videos"]
        ]
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        self.assertTrue(self.tab.show_run(run_dir))
        self.assertEqual(self.tab.run.schema_version, 2)
        self.assertEqual(len(self.tab.table_rows()), 3)
        self.assertIn("schema 2", self.tab.info_label.text())
        self.tab.select_row(0)
        self.assertEqual(self.tab.selected_video().name, "mouse_a.avi")

    def test_an_unreadable_manifest_is_reported_without_a_crash(self):
        run_dir = self.copy_run("broken")
        (run_dir / "manifest.json").write_text("{not json")
        self.assertFalse(self.tab.show_run(run_dir))
        self.assertIn("Cannot open", self.tab.info_label.text())
        self.assertEqual(len(self.prompts.called("error")), 1)
        self.assertIsNone(self.tab.run)
        self.assertEqual(self.tab.table_rows(), [])


class ProjectRunTests(ResultsFixture):
    def test_the_projects_runs_are_listed_newest_first(self):
        project = self.open_project_with_videos()
        older = self.copy_run("older")
        manifest = json.loads((older / "manifest.json").read_text())
        manifest["started_at"] = "2020-01-01T00:00:00+0000"
        (older / "manifest.json").write_text(json.dumps(manifest))
        project.record_run(older)
        self.assertEqual(self.tab.run_paths()[0], self.run_dir.resolve())
        self.assertEqual(len(self.tab.run_paths()), 2)

    def test_activating_the_tab_shows_the_newest_run(self):
        self.open_project_with_videos()
        self.assertIsNone(self.tab.run)
        self.window.go_to_stage("results")
        self.assertIsNotNone(self.tab.run)
        self.assertEqual(self.tab.run.run_dir, self.run_dir.resolve())

    def test_selecting_a_video_in_the_workspace_selects_its_row(self):
        project = self.open_project_with_videos()
        self.tab.show_run(self.run_dir)
        panel = self.window.video_panel
        key = next(v.key for v in project.videos if v.name == "mouse_c.avi")
        panel.select_keys([key])
        panel.flush_selection()
        self.assertEqual(self.tab.selected_video().name, "mouse_c.avi")

    def test_a_missing_run_folder_is_marked(self):
        project = self.open_project_with_videos()
        gone = self.root / "gone" / "analysis_run_20200101_000000"
        gone.mkdir(parents=True)
        (gone / "run_plan.json").write_text(json.dumps({"plan": {}}))
        project.record_run(gone)
        shutil.rmtree(gone.parent)
        self.tab._refresh_runs()
        labels = [self.tab.run_combo.itemText(i)
                  for i in range(self.tab.run_combo.count())]
        self.assertTrue(any("folder not found" in label for label in labels))
        self.window.go_to_stage("results")
        self.assertEqual(self.tab.run.run_dir, self.run_dir.resolve())

    def test_closing_the_project_clears_the_tab(self):
        self.open_project_with_videos()
        self.window.go_to_stage("results")
        self.assertIsNotNone(self.tab.run)
        self.prompts.say("ask_save_changes", "discard")
        self.assertTrue(self.window.close_project())
        self.assertIsNone(self.tab.run)
        self.assertEqual(self.tab.run_paths(), [])


class VideoViewTests(ResultsFixture):
    def setUp(self):
        super().setUp()
        self.tab.show_run(self.run_dir)

    def test_selecting_a_row_loads_the_per_video_view(self):
        self.assertIsNone(self.tab.selected_video())
        self.tab.select_row(1)
        video = self.tab.selected_video()
        self.assertEqual(video.name, "mouse_b.avi")
        detail = self.tab.detail
        self.assertEqual(detail.title_label.text(), "mouse_b.avi")
        metrics = {(s, m): v for s, m, v in detail.metric_rows()}
        self.assertEqual(metrics[("sample", "group")], "ctl")
        self.assertEqual(metrics[("metadata", "fps")], "30")
        self.assertEqual(metrics[("metadata", "total frames")], "90")
        self.assertEqual(
            [p.name for p in detail.plot_paths()],
            sorted(p.name for p in (self.run_dir / "mouse_b" / "plots").glob("*.png")),
        )
        self.assertFalse(detail.plot_list.item(0).icon().isNull())
        self.assertTrue(detail.warnings_label.isHidden())

    def test_a_row_of_a_sorted_table_loads_its_own_video(self):
        column = self.tab.model.headers.index("analyzed_frames")
        self.tab.table.sortByColumn(column, Qt.SortOrder.DescendingOrder)
        self.tab.table.selectRow(0)
        self.tab.table.setCurrentIndex(self.tab.proxy.index(0, 0))
        self.assertEqual(self.tab.selected_video().name, "mouse_c.avi")

    def test_quality_warnings_are_shown(self):
        run_dir = self.copy_run("warn")
        manifest = json.loads((run_dir / "manifest.json").read_text())
        manifest["quality_warnings"] = [
            {"source": manifest["videos"][0]["source"],
             "message": "tracking was lost for 2 s"}
        ]
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        self.tab.show_run(run_dir)
        self.tab.select_row(0)
        detail = self.tab.detail
        self.assertFalse(detail.warnings_label.isHidden())
        self.assertIn("tracking was lost for 2 s", detail.warnings_label.text())
        index = self.tab.proxy.index(0, 0)
        self.assertIn("tracking was lost", index.data(Qt.ItemDataRole.ToolTipRole))
        self.assertFalse(index.data(Qt.ItemDataRole.DecorationRole).isNull())

    def test_clicking_a_plot_opens_it_enlarged(self):
        self.tab.select_row(0)
        detail = self.tab.detail
        opened = []
        detail.plot_requested.connect(opened.append)
        detail._on_plot_clicked(detail.plot_list.item(0))
        self.assertEqual(opened, [detail.plot_paths()[0]])
        dialog = self.tab.open_plot(opened[0])
        self.assertIsInstance(dialog, PlotDialog)
        self.assertTrue(dialog.isVisible())
        self.assertFalse(dialog.shown_pixmap.isNull())
        self.assertEqual(dialog.windowTitle(), opened[0].name)
        dialog.close()

    def test_the_validation_video_is_the_default_source(self):
        self.tab.select_row(0)
        detail = self.tab.detail
        detail.show_video_tab()
        pane = detail.video_pane
        self.assertEqual(pane.current_source()[0], SOURCE_VALIDATION)
        media = self.tab.selected_video().media()[0]
        self.assertEqual(pane.player.video_path, media)
        self.assertTrue(pane.player.wait_idle())
        self.assertIsNotNone(pane.player.last_frame)
        # Switching video releases the previous capture and loads the next.
        self.tab.select_row(2)
        self.assertEqual(
            pane.player.video_path, self.tab.selected_video().media()[0]
        )

    def test_the_original_video_gets_the_tracking_overlay_without_a_validation_video(self):
        run_dir = self.copy_run("nomedia")
        shutil.rmtree(run_dir / "mouse_a" / "media")
        self.tab.show_run(run_dir)
        self.tab.select_row(0)
        detail = self.tab.detail
        detail.show_video_tab()
        pane = detail.video_pane
        self.assertEqual(pane.current_source()[0], SOURCE_OVERLAY)
        video = self.tab.selected_video()
        self.assertEqual(pane.player.video_path, video.source)
        self.assertIsNotNone(pane.player._overlay)
        self.assertTrue(pane.player.wait_idle())
        # What is on screen is the frame with the overlay on it.
        raw = pane.player.last_frame
        drawn = pane.player._overlay(raw.copy(), 10)
        self.assertGreater(int(np.abs(drawn.astype(int) - raw.astype(int)).sum()), 0)

    def test_overlay_for_video_explains_missing_data(self):
        run = rd.RunData.load(self.run_dir)
        video = run.videos[0]
        callback, message = overlay_for_video(run, video)
        self.assertIsNotNone(callback)
        self.assertEqual(message, "")
        copy = self.copy_run("nodata")
        for path in (copy / "mouse_a" / "data").glob("*_tracking_data.csv"):
            path.unlink()
        run2 = rd.RunData.load(copy)
        callback, message = overlay_for_video(run2, run2.videos[0])
        self.assertIsNone(callback)
        self.assertIn("no per-frame data", message)

    def test_a_missing_original_video_is_explained(self):
        run_dir = self.copy_run("nosource")
        shutil.rmtree(run_dir / "mouse_a" / "media")
        manifest = json.loads((run_dir / "manifest.json").read_text())
        manifest["videos"][0]["source"] = str(self.root / "moved" / "mouse_a.avi")
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        self.tab.show_run(run_dir)
        self.tab.select_row(0)
        self.tab.detail.show_video_tab()
        pane = self.tab.detail.video_pane
        self.assertIn("not found", pane.message_label.text())
        self.assertIsNone(pane.player.video_path)

    def test_the_decoder_is_released_when_the_tab_is_hidden_or_shut_down(self):
        self.tab.select_row(0)
        self.tab.detail.show_video_tab()
        player = self.tab.detail.video_pane.player
        self.assertTrue(player.wait_idle())
        self.assertTrue(player.capture_open)
        self.tab.shutdown()
        self.assertFalse(player.capture_open)


class TimeBinsTests(ResultsFixture):
    def setUp(self):
        super().setUp()
        self.tab.show_run(self.run_dir)
        self.panel = self.tab.bins_panel

    def test_the_chart_renders_one_line_per_video(self):
        panel = self.panel
        self.assertIn("distance_total", [
            panel.metric_combo.itemText(i) for i in range(panel.metric_combo.count())
        ])
        panel.set_metric("distance_total")
        panel.canvas.draw()
        self.assertEqual(len(panel.series), 3)
        self.assertEqual(len(panel.axes.lines), 3)
        self.assertEqual(panel.message_label.text(), "")
        self.assertIn("distance total", panel.axes.get_ylabel())
        self.assertIn("pixels", panel.axes.get_ylabel())

    def test_the_chart_renders_a_grouped_metric(self):
        panel = self.panel
        panel.set_metric("distance_total")
        panel.set_group_by("group")
        panel.canvas.draw()
        self.assertEqual([s.label for s in panel.series], ["ctl", "trt"])
        self.assertEqual([s.n for s in panel.series], [2, 1])
        legend = panel.axes.get_legend()
        self.assertEqual(
            [t.get_text() for t in legend.get_texts()],
            ["ctl (n=2)", "trt (n=1)"],
        )
        # The ctl line carries error bars (two videos), trt does not.
        self.assertTrue(any(s.err and max(s.err) > 0 for s in panel.series))
        self.assertIn("by group", panel.axes.get_title())
        panel.set_group_by(None)
        self.assertEqual(len(panel.series), 3)

    def test_the_choices_are_the_runs_columns(self):
        panel = self.panel
        groups = [panel.group_combo.itemText(i)
                  for i in range(panel.group_combo.count())]
        self.assertEqual(groups, [NO_GROUP, "animal_id", "group"])
        self.assertTrue(panel.subject_combo.isHidden())

    def test_a_run_without_time_bins_says_so(self):
        run_dir = self.copy_run("nobins")
        (run_dir / "batch_time_bins.csv").unlink()
        for path in run_dir.glob("*/data/*_time_bins.csv"):
            path.unlink()
        self.tab.show_run(run_dir)
        self.assertEqual(self.panel.series, [])
        self.assertIn("no time bins", self.panel.message_label.text())
        self.panel.canvas.draw()

    def test_a_subject_choice_filters_the_series(self):
        table = rd.Table(
            ["video_filename", "paradigm", "bin_index", "bin_start_s",
             "bin_end_s", "subject", "metric", "value", "unit"],
            [["t.mp4", "TST", "1", "0", "60", "mouse_1", "immobility_s", "10", "s"],
             ["t.mp4", "TST", "1", "0", "60", "mouse_2", "immobility_s", "20", "s"]],
        )
        run = rd.RunData.load(self.run_dir)
        run.time_bins = table
        self.panel.set_run(run)
        self.assertFalse(self.panel.subject_combo.isHidden())
        self.assertEqual(self.panel.subject_combo.itemText(0), ALL_SUBJECTS)
        self.assertEqual(len(self.panel.series), 2)
        self.panel.set_subject("mouse_2")
        self.assertEqual([s.y for s in self.panel.series], [[20.0]])


class CompareTests(ResultsFixture):
    def make_changed_run(self):
        run_dir = self.copy_run("changed")
        path = run_dir / "batch_summary.csv"
        rows = self.read_csv(path)
        column = rows[0].index("total_distance")
        rows[2][column] = "999.00"
        with open(path, "w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(rows)
        return run_dir

    def test_compare_runs_flags_the_changed_values(self):
        changed = self.make_changed_run()
        self.tab.show_run(self.run_dir)
        self.tab.show_run(changed)
        panel = self.tab.compare_panel
        panel.choose(self.run_dir, changed)
        diff = panel.compare()
        self.assertEqual(panel.changed_cells(), [("mouse_b.avi", "total_distance")])
        self.assertEqual(diff.changed_cells, 1)
        self.assertIn("1 value(s) changed in 1 of 3 video(s)", panel.summary_label.text())
        # The cell shows both values and is highlighted.
        column = diff.headers.index("total_distance")
        item = panel.table.item(1, column)
        self.assertIn("→", item.text())
        self.assertIn("999.00", item.text())
        self.assertNotEqual(
            panel.table.item(1, column).background().color(),
            panel.table.item(0, column).background().color(),
        )

    def test_only_changed_videos_can_be_shown(self):
        changed = self.make_changed_run()
        self.tab.show_run(self.run_dir)
        self.tab.show_run(changed)
        panel = self.tab.compare_panel
        panel.choose(self.run_dir, changed)
        panel.compare()
        self.assertEqual(panel.table.rowCount(), 3)
        panel.only_changed_check.setChecked(True)
        self.assertEqual(panel.table.rowCount(), 1)

    def test_identical_runs_report_no_changes(self):
        same = self.copy_run("same")
        self.tab.show_run(self.run_dir)
        self.tab.show_run(same)
        panel = self.tab.compare_panel
        panel.choose(self.run_dir, same)
        diff = panel.compare()
        self.assertEqual(diff.changed_cells, 0)
        self.assertIn("0 value(s) changed", panel.summary_label.text())

    def test_comparing_a_run_with_itself_is_refused(self):
        self.tab.show_run(self.run_dir)
        panel = self.tab.compare_panel
        panel.choose(self.run_dir, self.run_dir)
        self.assertIsNone(panel.compare())
        self.assertIn("two different runs", panel.summary_label.text())

    def test_the_compare_button_preselects_this_run_and_the_one_before(self):
        changed = self.make_changed_run()
        self.tab.show_run(self.run_dir)
        self.tab.show_run(changed)
        self.assertTrue(self.tab.compare_button.isEnabled())
        self.tab.show_compare()
        panel = self.tab.compare_panel
        self.assertIs(self.tab.sections.currentWidget(), panel)
        self.assertEqual(panel.combo_b.currentData(), str(changed))
        self.assertEqual(panel.combo_a.currentData(), str(self.run_dir))

    def test_compare_needs_two_runs(self):
        self.assertFalse(self.tab.compare_button.isEnabled())
        self.assertIsNone(self.tab.compare_panel.compare())


class ResultsStageTests(ResultsFixture):
    """The stage around the widgets: empty states, picker, header, cards."""

    def test_without_a_project_the_empty_state_says_so(self):
        self.window.go_to_stage("results", force=True)
        self.assertEqual(self.tab.pages.currentIndex(), 0)
        self.assertIn("No project open", self.tab.empty.title())
        self.assertTrue(self.tab.empty_run_button.isHidden())
        self.assertFalse(self.tab.folder_button.isEnabled())
        self.assertTrue(self.tab.folder_button.toolTip())

    def test_a_project_without_runs_leads_to_review_and_run(self):
        self.new_project_at()
        self.add_videos()
        self.window.change_paradigm("Y-Maze")
        self.window.go_to_stage("results", force=True)
        self.assertEqual(self.tab.pages.currentIndex(), 0)
        self.assertEqual(self.tab.empty.title(), "No runs yet")
        self.assertFalse(self.tab.empty_run_button.isHidden())
        self.assertEqual(self.tab.empty_run_button.text(), "Go to Review & run")
        self.tab.empty_run_button.click()
        self.assertEqual(self.window.current_stage_id, "review")

    def test_the_empty_state_can_still_open_a_run_folder(self):
        self.window.go_to_stage("results", force=True)
        self.prompts.say("choose_folder", self.run_dir)
        self.tab.empty_open_button.click()
        self.assertEqual(self.tab.pages.currentIndex(), 1)
        self.assertEqual(self.tab.run.run_dir, self.run_dir)

    def test_the_header_offers_the_folder_actions(self):
        self.tab.show_run(self.run_dir)
        actions = self.tab.header.actions_layout
        buttons = [actions.itemAt(i).widget() for i in range(actions.count())]
        self.assertIn(self.tab.open_button, buttons)
        self.assertIn(self.tab.folder_button, buttons)
        self.assertTrue(self.tab.folder_button.isEnabled())
        self.tab.folder_button.click()
        self.assertEqual(self.prompts.called("open_folder"), [(self.run_dir,)])

    def test_the_picker_lists_date_folder_and_counts(self):
        project = self.open_project_with_videos()
        self.tab.on_project_modified()
        text = self.tab.run_combo.itemText(0)
        self.assertIn(self.run_dir.name, text)
        self.assertIn("3 done", text)
        self.assertRegex(text, r"\d{2} [A-Z][a-z]{2} \d{4}, \d{2}:\d{2}")
        self.assertEqual(len(project.runs), 1)

    def test_the_run_is_summarised_in_chips(self):
        self.tab.show_run(self.run_dir)
        self.assertEqual(self.tab.completed_chip.text(), "3 completed")
        self.assertTrue(self.tab.failed_chip.isHidden())
        self.assertTrue(self.tab.warnings_chip.isHidden())
        self.assertFalse(self.tab.paradigm_chip.isHidden())
        self.assertFalse(self.tab.bins_chip.isHidden())
        self.assertIn("time bins of 1 s", self.tab.bins_chip.text())

    def test_the_sections_are_videos_time_bins_and_compare(self):
        self.assertEqual(
            [self.tab.sections.tabText(i) for i in range(self.tab.sections.count())],
            ["Videos", "Time bins", "Compare runs"],
        )

    def test_the_footer_names_the_run_and_the_last_stage_has_no_next(self):
        self.assertIsNone(self.tab.next_label())
        self.tab.show_run(self.run_dir)
        text, chip = self.tab.footer_text()
        self.assertIn(self.run_dir.name, text)
        self.assertEqual(chip, ("ok", "Run loaded"))

    def test_a_video_is_shown_as_key_numbers_a_closed_table_and_plots(self):
        self.tab.show_run(self.run_dir)
        self.tab.select_row(0)
        detail = self.tab.detail
        tiles = dict(detail.tile_texts())
        self.assertIn("Total distance", tiles)
        self.assertIn("Average speed", tiles)
        self.assertFalse(detail.tiles_card.isHidden())
        self.assertFalse(detail.metrics_section.is_expanded())
        self.assertGreater(detail.plot_list.count(), 0)
        self.assertTrue(detail.no_plots_label.isHidden())
        self.assertEqual(
            [detail.tabs.tabText(i) for i in range(detail.tabs.count())],
            ["Overview", "Plots", "Video"],
        )

    def test_a_video_without_plots_says_so(self):
        run_dir = self.copy_run("noplots")
        shutil.rmtree(run_dir / "mouse_a" / "plots")
        self.tab.show_run(run_dir)
        self.tab.select_row(0)
        detail = self.tab.detail
        self.assertEqual(detail.plot_list.count(), 0)
        self.assertFalse(detail.no_plots_label.isHidden())
        self.assertTrue(detail.plot_list.isHidden())

    def test_a_quality_warning_is_a_warning_callout_with_a_title(self):
        run_dir = self.copy_run("warncallout")
        manifest = json.loads((run_dir / "manifest.json").read_text())
        manifest["quality_warnings"] = [
            {"source": manifest["videos"][0]["source"], "message": "lost <1 s>"}
        ]
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        self.tab.show_run(run_dir)
        self.tab.select_row(0)
        callout = self.tab.detail.warnings_label
        self.assertEqual(callout.kind(), "warning")
        self.assertEqual(callout.title(), "Quality warnings")
        self.assertIn("lost &lt;1 s&gt;", callout.text())    # never read as markup

    def test_the_chart_follows_the_theme(self):
        from matplotlib.colors import to_hex
        from gui.workspace import theme
        self.addCleanup(theme.apply_theme, self.app, "light")
        self.tab.show_run(self.run_dir)
        theme.apply_theme(self.app, "dark")
        self.tab.bins_panel.render()
        self.assertEqual(to_hex(self.tab.bins_panel.figure.get_facecolor()).lower(),
                         theme.DARK.surface.lower())
        theme.apply_theme(self.app, "light")
        self.tab.bins_panel.render()
        self.assertEqual(to_hex(self.tab.bins_panel.figure.get_facecolor()).lower(),
                         theme.LIGHT.surface.lower())

    def test_shutdown_releases_the_decoder_and_the_plot_windows(self):
        self.tab.show_run(self.run_dir)
        self.tab.select_row(0)
        self.tab.open_plot(self.tab.detail.plot_paths()[0])
        self.assertEqual(len(self.tab._plot_dialogs), 1)
        self.tab.shutdown()
        self.assertEqual(self.tab._plot_dialogs, [])

    def test_the_window_shuts_the_stage_down_when_it_closes(self):
        self.tab.show_run(self.run_dir)
        self.tab.select_row(0)
        self.tab.open_plot(self.tab.detail.plot_paths()[0])
        self.prompts.say("ask_save_changes", "discard")
        self.window.close()
        self.assertEqual(self.tab._plot_dialogs, [])


class RunLabelTests(unittest.TestCase):
    def test_the_label_names_date_folder_and_counts(self):
        from gui.workspace.stages.results_stage import format_started, run_label
        label = run_label(Path("analysis_run_1"), 3, 1, True, "2020-01-02T03:04:05+0000")
        self.assertEqual(
            label,
            "02 Jan 2020, 03:04  ·  analysis_run_1  ·  3 done, 1 failed, cancelled",
        )
        self.assertEqual(run_label(Path("analysis_run_1")), "analysis_run_1")
        self.assertEqual(format_started("20201231_235959"), "31 Dec 2020, 23:59")
        self.assertEqual(format_started("whenever"), "whenever")
        self.assertEqual(format_started(""), "")


class FinishedRunHandOffTests(ResultsFixture):
    def test_the_review_stage_hands_a_finished_run_to_the_results_stage(self):
        self.new_project_at()
        self.window.video_panel.add_paths(
            sorted((self.run_root / "real").glob("*.avi"))
        )
        project = self.window.project
        review = self.window.stage("review")
        review.controller._last_run_dir = self.run_dir
        self.assertTrue(review.open_results())
        self.assertEqual(self.window.current_stage_id, "results")
        self.assertEqual(self.tab.run.run_dir, self.run_dir)
        project.record_run(self.run_dir)
        self.assertEqual(
            {v.last_status for v in project.videos}, {STATUS_DONE}
        )


if __name__ == "__main__":
    unittest.main()
