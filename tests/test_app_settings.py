import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from PyQt6.QtCore import QByteArray
    from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
    from gui.app_settings import (MAX_RECENT_FILES, SETTINGS_FILE_ENV_VAR,
                                  AppSettings)
    from gui.main_window import MainWindow
    from tests.gui_support import temp_app_settings
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class AppSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.ini = self.root / "settings.ini"
        self.settings = AppSettings.for_file(self.ini)

    def _reopen(self) -> AppSettings:
        self.settings.sync()
        return AppSettings.for_file(self.ini)

    def test_folders_round_trip_and_stale_folders_are_ignored(self):
        videos = self.root / "videos"
        videos.mkdir()
        self.settings.set_last_video_dir(videos)
        self.settings.set_output_dir(self.root)
        reopened = self._reopen()
        self.assertEqual(reopened.last_video_dir(), videos)
        self.assertEqual(reopened.output_dir(), self.root)

        videos.rmdir()
        self.assertIsNone(self._reopen().last_video_dir())
        self.assertEqual(self._reopen().dialog_dir("video"), "")

    def test_dialog_dir_falls_back_in_order(self):
        self.assertEqual(self.settings.dialog_dir("config"), "")
        self.assertEqual(self.settings.dialog_dir("config", self.root), str(self.root))
        output = self.root / "out"
        output.mkdir()
        self.settings.set_output_dir(output)
        self.assertEqual(self.settings.dialog_dir("config"), str(output))
        configs = self.root / "cfg"
        configs.mkdir()
        self.settings.set_last_config_dir(configs)
        self.assertEqual(self.settings.dialog_dir("config"), str(configs))

    def test_clearing_a_folder_removes_it(self):
        self.settings.set_output_dir(self.root)
        self.settings.set_output_dir(None)
        self.assertIsNone(self._reopen().output_dir())

    def test_window_geometry_round_trips(self):
        self.assertIsNone(self.settings.window_geometry())
        self.settings.save_window(QByteArray(b"geometry"), QByteArray(b"state"))
        reopened = self._reopen()
        self.assertEqual(bytes(reopened.window_geometry()), b"geometry")
        self.assertEqual(bytes(reopened.window_state()), b"state")

    def test_last_paradigm_round_trips(self):
        self.assertIsNone(self.settings.last_paradigm())
        self.settings.set_last_paradigm("Y-Maze")
        self.assertEqual(self._reopen().last_paradigm(), "Y-Maze")
        self.settings.set_last_paradigm(None)
        self.assertIsNone(self._reopen().last_paradigm())

    def test_recent_files_are_deduplicated_newest_first_and_capped(self):
        files = []
        for index in range(MAX_RECENT_FILES + 3):
            path = self.root / f"plan_{index}.json"
            path.write_text("{}")
            files.append(path)
            self.settings.add_recent_file(path)
        self.settings.add_recent_file(files[5])  # moves to the front

        recent = self._reopen().recent_files()
        self.assertEqual(len(recent), MAX_RECENT_FILES)
        self.assertEqual(recent[0], files[5])
        self.assertEqual(len(set(recent)), len(recent))

    def test_recent_files_skip_missing_files_and_can_be_cleared(self):
        kept = self.root / "kept.json"
        gone = self.root / "gone.json"
        kept.write_text("{}")
        gone.write_text("{}")
        self.settings.add_recent_file(gone)
        self.settings.add_recent_file(kept)
        gone.unlink()
        self.assertEqual(self._reopen().recent_files(), [kept])

        self.settings.remove_recent_file(kept)
        self.assertEqual(self._reopen().recent_files(), [])
        self.settings.add_recent_file(kept)
        self.settings.clear_recent_files()
        self.assertEqual(self._reopen().recent_files(), [])

    def test_corrupt_recent_list_is_treated_as_empty(self):
        self.settings._qs.setValue(AppSettings.KEY_RECENT, "{not json")
        self.assertEqual(self.settings.recent_files(), [])
        self.settings._qs.setValue(AppSettings.KEY_RECENT, '"a string"')
        self.assertEqual(self.settings.recent_files(), [])

    def test_environment_variable_selects_an_ini_file(self):
        with patch.dict(os.environ, {SETTINGS_FILE_ENV_VAR: str(self.ini)}):
            AppSettings().set_last_paradigm("EPM")
        self.assertEqual(self._reopen().last_paradigm(), "EPM")


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class MainWindowPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.settings = temp_app_settings(self)

    def _window(self) -> MainWindow:
        window = MainWindow(self.settings)
        self.addCleanup(window.deleteLater)
        self.addCleanup(window._detach_log_handler)
        return window

    def test_output_folder_is_shown_remembered_and_restored(self):
        window = self._window()
        self.assertIn("first video", window.output_label.text())

        out = self.root / "results"
        out.mkdir()
        with patch.object(QFileDialog, "getExistingDirectory", return_value=str(out)), \
                patch.object(QMessageBox, "information"):
            window.change_output_button.click()
        self.assertEqual(window.output_dir, out)
        self.assertIn(str(out), window.output_label.text())

        self.settings.sync()
        reopened = self._window()
        self.assertEqual(reopened.output_dir, out)
        self.assertIn(str(out), reopened.output_label.text())

    def test_automatic_output_folder_is_not_remembered(self):
        window = self._window()
        video = self.root / "a.mp4"
        window._add_video_paths([video])
        self.assertEqual(window.output_dir, self.root)
        self.assertIsNone(self.settings.output_dir())

    def test_add_videos_dialog_opens_in_the_last_video_folder(self):
        window = self._window()
        videos = self.root / "videos"
        videos.mkdir()
        picked = str(videos / "m1.mp4")
        with patch.object(
            QFileDialog, "getOpenFileNames", return_value=([picked], "")
        ) as dialog:
            window.add_videos()
            window.add_videos()
        self.assertEqual(dialog.call_args_list[0].args[2], "")
        self.assertEqual(dialog.call_args_list[1].args[2], str(videos))

    def test_loading_a_configuration_updates_the_recent_menu(self):
        window = self._window()
        path = self.root / "missing.json"
        path.write_text("not json")
        with patch.object(QMessageBox, "critical"):
            window.load_configuration_file(path)
        # A file that failed to load is not remembered.
        self.assertEqual(self.settings.recent_files(), [])

        self.settings.add_recent_file(path)
        window._rebuild_recent_menu()
        texts = [action.text() for action in window.recent_menu.actions()]
        self.assertIn("missing.json", texts)

    def test_closing_saves_the_window_geometry(self):
        window = self._window()
        window.show()
        window.close()
        self.assertIsNotNone(self.settings.window_geometry())


if __name__ == "__main__":
    unittest.main()
