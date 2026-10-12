import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from PyQt6.QtCore import Qt, QUrl
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from core.models import AppConfig, BatchSettings
    from core.video import ReferenceFrameGenerator
    from gui import ux_helpers
    from gui.main_window import MainWindow
    from gui.video_list import (STATUS_DONE, STATUS_FAILED, STATUS_QUEUED,
                                STATUS_RUNNING, STATUS_WARNING,
                                VideoListWidget, clear_list_choices,
                                collect_video_files, find_videos_in_folder,
                                move_selected, outcome_statuses, remove_at)
    from gui.worker import AnalysisWorker
    from mazes.freestyle import Freestyle
    from roi_manager import ROIManager
    from tests import worker_support as support
    from tests.gui_support import temp_app_settings
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


class VideoListLogicTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()

    @unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
    def test_folder_scan_is_flat_unless_recursive(self):
        top = _touch(self.root / "b.MP4")
        _touch(self.root / "notes.txt")
        nested = _touch(self.root / "day2" / "a.avi")
        self.assertEqual(find_videos_in_folder(self.root, recursive=False), [top])
        self.assertEqual(
            find_videos_in_folder(self.root, recursive=True), [top, nested]
        )

    @unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
    def test_recursive_scan_skips_result_and_hidden_folders(self):
        wanted = _touch(self.root / "videos" / "m1.mp4")
        _touch(self.root / "analysis_run_20260101_000000" / "m1" / "media" / "m1_timelapse.mp4")
        _touch(self.root / ".checkpoints" / "x.mov")
        self.assertEqual(find_videos_in_folder(self.root, recursive=True), [wanted])

    @unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
    def test_collect_accepts_files_and_folders_and_reports_the_rest(self):
        single = _touch(self.root / "one.mkv")
        folder = self.root / "set"
        in_folder = _touch(folder / "two.mov")
        document = _touch(self.root / "readme.txt")
        missing = self.root / "gone.mp4"

        videos, ignored = collect_video_files(
            [single, folder, single, document, missing], recursive=False
        )
        self.assertEqual(videos, [single, in_folder])
        self.assertEqual(ignored, [document, missing])

    @unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
    def test_move_selected_moves_blocks_and_stops_at_the_edges(self):
        items = list("abcd")
        self.assertEqual(move_selected(items, [1], -1), (list("bacd"), [0]))
        self.assertEqual(move_selected(items, [0], -1), (items, [0]))
        self.assertEqual(move_selected(items, [3], 1), (items, [3]))
        self.assertEqual(move_selected(items, [1, 2], 1), (list("adbc"), [2, 3]))
        self.assertEqual(move_selected(items, [0, 1], -1), (items, [0, 1]))
        self.assertEqual(move_selected(items, [0, 2], 1), (list("badc"), [1, 3]))
        self.assertEqual(move_selected(items, [], 1), (items, []))
        self.assertEqual(items, list("abcd"), "input must not be modified")

    @unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
    def test_remove_at(self):
        self.assertEqual(remove_at(list("abcd"), [0, 2]), ["b", "d"])
        self.assertEqual(remove_at(list("ab"), []), ["a", "b"])

    @unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
    def test_clear_list_only_offers_to_drop_a_configuration_that_exists(self):
        self.assertEqual([k for k, _ in clear_list_choices(False)], ["videos"])
        self.assertEqual([k for k, _ in clear_list_choices(True)], ["videos", "all"])

    @unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
    def test_outcome_statuses(self):
        a, b, c = Path("a.mp4"), Path("b.mp4"), Path("c.mp4")
        statuses = outcome_statuses(
            [a, b], [(c, "unreadable")], [(b, "Fair: 80% coverage")]
        )
        self.assertEqual(statuses[a], (STATUS_DONE, ""))
        self.assertEqual(statuses[b], (STATUS_WARNING, "Fair: 80% coverage"))
        self.assertEqual(statuses[c], (STATUS_FAILED, "unreadable"))


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class VideoListWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_rows_carry_status_icons_and_tooltips(self):
        widget = VideoListWidget()
        self.addCleanup(widget.deleteLater)
        a, b = Path("a.mp4"), Path("b.mp4")
        widget.set_videos([a, b], {b: (STATUS_FAILED, "bad codec")}, selected=[1])
        self.assertEqual(widget.count(), 2)
        self.assertFalse(widget.item(0).icon().isNull())
        self.assertIn("Queued", widget.item(0).toolTip())
        self.assertEqual(widget.item(1).toolTip(), "Failed: bad codec")
        self.assertEqual(widget.selected_rows(), [1])

    def test_delete_and_alt_arrows_emit_requests(self):
        widget = VideoListWidget()
        self.addCleanup(widget.deleteLater)
        widget.set_videos([Path("a.mp4"), Path("b.mp4")])
        removed, moved = [], []
        widget.remove_requested.connect(lambda: removed.append(1))
        widget.move_requested.connect(moved.append)
        QTest.keyClick(widget, Qt.Key.Key_Delete)
        QTest.keyClick(widget, Qt.Key.Key_Up, Qt.KeyboardModifier.AltModifier)
        QTest.keyClick(widget, Qt.Key.Key_Down, Qt.KeyboardModifier.AltModifier)
        self.assertEqual((removed, moved), ([1], [-1, 1]))


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class MainWindowVideoListTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.window = MainWindow(temp_app_settings(self))
        self.addCleanup(self.window.deleteLater)
        self.addCleanup(self.window._detach_log_handler)

    def _texts(self):
        return [
            self.window.video_list_widget.item(i).text()
            for i in range(self.window.video_list_widget.count())
        ]

    def _select(self, *rows):
        self.window.video_list_widget.clearSelection()
        for row in rows:
            self.window.video_list_widget.item(row).setSelected(True)

    def test_dropped_files_and_folders_are_added(self):
        a = _touch(self.root / "a.mp4")
        folder = self.root / "set"
        b = _touch(folder / "b.avi")
        _touch(folder / "readme.txt")
        self.window.add_dropped_paths([a, folder])
        self.assertEqual(self.window.video_paths, [a, b])
        self.assertEqual(self._texts(), [str(a), str(b)])
        # Dropping the same paths again adds nothing.
        self.window.add_dropped_paths([a, folder])
        self.assertEqual(len(self.window.video_paths), 2)

    def test_dropped_folder_honours_the_subfolder_checkbox(self):
        deep = _touch(self.root / "set" / "day1" / "deep.mp4")
        self.window.recursive_check.setChecked(False)
        with patch.object(QMessageBox, "information"):
            self.window.add_dropped_paths([self.root / "set"])
        self.assertEqual(self.window.video_paths, [])
        self.window.recursive_check.setChecked(True)
        self.window.add_dropped_paths([self.root / "set"])
        self.assertEqual(self.window.video_paths, [deep])

    def test_dropping_only_unsupported_files_explains_what_works(self):
        note = _touch(self.root / "note.txt")
        with patch.object(QMessageBox, "information") as info:
            self.window.add_dropped_paths([note])
        info.assert_called_once()
        self.assertEqual(self.window.video_paths, [])

    def test_dropping_a_saved_configuration_loads_it(self):
        manager = ROIManager()
        import numpy as np
        manager.add_roi("arm_a", np.array([[0, 0], [9, 0], [9, 9]]))
        template = self.root / "template.json"
        BatchSettings(manager, AppConfig().detection, 0.0).save_to_file(template)
        with patch.object(QMessageBox, "information"):
            self.window.add_dropped_paths([template])
        self.assertIsNotNone(self.window.current_settings)

    def test_delete_key_removes_the_selected_videos(self):
        videos = [_touch(self.root / f"v{i}.mp4") for i in range(4)]
        self.window._add_video_paths(videos)
        self._select(1, 2)
        QTest.keyClick(self.window.video_list_widget, Qt.Key.Key_Delete)
        self.assertEqual(self.window.video_paths, [videos[0], videos[3]])
        self.assertEqual(self._texts(), [str(videos[0]), str(videos[3])])

    def test_move_up_and_down_reorder_and_keep_the_selection(self):
        videos = [_touch(self.root / f"v{i}.mp4") for i in range(3)]
        self.window._add_video_paths(videos)
        self._select(2)
        self.window.move_up_button.click()
        self.assertEqual(self.window.video_paths, [videos[0], videos[2], videos[1]])
        self.assertEqual(self.window.video_list_widget.selected_rows(), [1])
        self.window.move_down_button.click()
        self.assertEqual(self.window.video_paths, videos)
        self.assertEqual(self.window.video_list_widget.selected_rows(), [2])

    def test_buttons_follow_the_selection(self):
        self.window._add_video_paths([_touch(self.root / "v.mp4")])
        self.assertFalse(self.window.remove_selected_button.isEnabled())
        self._select(0)
        self.assertTrue(self.window.remove_selected_button.isEnabled())
        self.assertTrue(self.window.move_up_button.isEnabled())

    def test_clear_list_asks_and_can_keep_the_configuration(self):
        videos = [_touch(self.root / "v.mp4")]
        self.window._add_video_paths(videos)
        self.window.current_settings = object()

        with patch.object(ux_helpers, "ask_choice", return_value="cancel"):
            self.window.clear_list()
        self.assertEqual(self.window.video_paths, videos)

        with patch.object(ux_helpers, "ask_choice", return_value="videos") as ask:
            self.window.clear_list()
        self.assertEqual(self.window.video_paths, [])
        self.assertIsNotNone(self.window.current_settings)
        labels = [choice.label for choice in ask.call_args.args[3]]
        self.assertIn("Clear videos, keep configuration", labels)

        self.window._add_video_paths(videos)
        with patch.object(ux_helpers, "ask_choice", return_value="all"):
            self.window.clear_list()
        self.assertIsNone(self.window.current_settings)
        self.assertEqual(self.window.batch_plan_cache, {})

    def test_clearing_without_a_configuration_offers_only_the_plain_option(self):
        self.window._add_video_paths([_touch(self.root / "v.mp4")])
        with patch.object(ux_helpers, "ask_choice", return_value="videos") as ask:
            self.window.clear_list()
        labels = [choice.label for choice in ask.call_args.args[3]]
        self.assertEqual(labels, ["Clear videos", "Cancel"])

    def test_automatic_output_folder_follows_the_first_video(self):
        a = _touch(self.root / "one" / "a.mp4")
        b = _touch(self.root / "two" / "b.mp4")
        self.window._add_video_paths([a, b])
        self.assertEqual(self.window.output_dir, a.parent)
        self._select(0)
        self.window.remove_selected_videos()
        self.assertEqual(self.window.output_dir, b.parent)
        self._select(0)
        self.window.remove_selected_videos()
        self.assertIsNone(self.window.output_dir)

    def test_status_signals_update_the_row_icon_and_tooltip(self):
        a = _touch(self.root / "a.mp4")
        self.window._add_video_paths([a])
        self.window._on_video_status(str(a), STATUS_RUNNING, "")
        self.assertIn("Running", self.window.video_list_widget.item(0).toolTip())
        self.window._on_video_status(str(a), STATUS_FAILED, "bad codec")
        self.assertEqual(
            self.window.video_list_widget.item(0).toolTip(), "Failed: bad codec"
        )

    def test_edits_keep_each_videos_status(self):
        videos = [_touch(self.root / f"v{i}.mp4") for i in range(2)]
        self.window._add_video_paths(videos)
        self.window._on_video_status(str(videos[1]), STATUS_FAILED, "x")
        self._select(1)
        self.window.move_up_button.click()
        self.assertIn("Failed", self.window.video_list_widget.item(0).toolTip())


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class RunStatusAndResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        cache = patch.object(
            ReferenceFrameGenerator, "get_cache_dir",
            return_value=self.root / "cache",
        )
        cache.start()
        self.addCleanup(cache.stop)
        self.window = MainWindow(temp_app_settings(self))
        self.addCleanup(self.window.deleteLater)
        self.addCleanup(self.window._detach_log_handler)

    def test_worker_reports_running_then_final_status_per_video(self):
        good = support.write_tracking_video(self.root, frames=40)
        bad = self.root / "missing.avi"
        out = self.root / "out"
        out.mkdir()
        plan = {
            good: support.tracking_settings(),
            bad: support.tracking_settings(),
        }
        worker = AnalysisWorker(plan, out, Freestyle())
        events = []
        worker.video_status.connect(lambda v, s, m: events.append((Path(v), s)))
        outcomes = []
        worker.finished.connect(outcomes.append)
        worker.run()

        self.assertEqual(
            events[:2], [(good, "running"), (good, "done")]
            if not outcomes[0].quality_warnings
            else [(good, "running"), (good, "warning")],
        )
        self.assertEqual(events[2:], [(bad, "running"), (bad, "failed")])

    def test_finished_run_updates_statuses_even_without_the_signals(self):
        a = _touch(self.root / "a.mp4")
        b = _touch(self.root / "b.mp4")
        self.window._add_video_paths([a, b])
        self.window._video_status = {a: (STATUS_QUEUED, ""), b: (STATUS_QUEUED, "")}

        from gui.worker import BatchRunOutcome
        outcome = BatchRunOutcome(
            results=[(a, object())], failures=[(b, "boom")], output_dir=self.root
        )
        with patch.object(ux_helpers, "ask_choice", return_value="ok"), \
                patch.object(QMessageBox, "warning"):
            self.window.analysis_finished(outcome)
        self.assertEqual(self.window._video_status[a][0], STATUS_DONE)
        self.assertEqual(self.window._video_status[b], (STATUS_FAILED, "boom"))
        self.assertEqual(self.window._last_run_dir, self.root)
        self.assertTrue(self.window.open_last_run_action.isEnabled())

    def _result_choices(self, run_dir):
        with patch.object(ux_helpers, "ask_choice", return_value="ok") as ask:
            self.window._show_run_result(
                QMessageBox.Icon.Information, "Complete", "text", run_dir
            )
        return [choice.key for choice in ask.call_args.args[3]]

    def test_result_dialog_offers_folder_and_summary_when_it_exists(self):
        self.assertEqual(self._result_choices(self.root), ["folder", "ok"])
        (self.root / "batch_summary.csv").write_text("a,b\n")
        self.assertEqual(
            self._result_choices(self.root), ["folder", "summary", "ok"]
        )
        self.assertEqual(self._result_choices(None), ["ok"])

    def test_result_buttons_open_the_folder_and_the_summary(self):
        summary = self.root / "batch_summary.csv"
        summary.write_text("a,b\n")
        answers = iter(["folder", "summary", "ok"])
        with patch.object(
            ux_helpers, "ask_choice", side_effect=lambda *a, **k: next(answers)
        ), patch("gui.main_window.QDesktopServices.openUrl", return_value=True) as opener:
            self.window._show_run_result(
                QMessageBox.Icon.Information, "Complete", "text", self.root
            )
        urls = [call.args[0] for call in opener.call_args_list]
        self.assertEqual(
            urls,
            [QUrl.fromLocalFile(str(self.root)), QUrl.fromLocalFile(str(summary))],
        )

    def test_open_output_folder_button(self):
        with patch.object(QMessageBox, "information") as info:
            self.window.open_output_button.click()
        info.assert_called_once()  # nothing chosen yet

        self.window._set_output_dir(self.root, explicit=True, persist=False)
        with patch("gui.main_window.QDesktopServices.openUrl", return_value=True) as opener:
            self.window.open_output_button.click()
        opener.assert_called_once_with(QUrl.fromLocalFile(str(self.root)))


if __name__ == "__main__":
    unittest.main()
