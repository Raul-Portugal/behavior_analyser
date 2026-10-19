"""Helpers for the workspace tests: scripted dialogs and a window fixture."""
import sys
import tempfile
import traceback
import unittest
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List

from tests.gui_support import temp_app_settings
from tests.project_support import ProjectFixture

try:
    from PyQt6.QtCore import QCoreApplication, QEvent
    from PyQt6.QtWidgets import QApplication
    from gui.workspace.prompts import Prompts
    from gui.workspace.window import WorkspaceWindow
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False
    Prompts = object


class ScriptedPrompts(Prompts):
    """A ``Prompts`` that answers from a script and never opens a dialog.

    ``say("confirm", True, False)`` queues answers for a method (first call
    gets the first answer). A question with nothing queued fails the test, so
    no dialog is ever skipped by accident. Messages and OS actions are
    recorded in ``calls`` and answered with a harmless default.
    """

    QUESTIONS = (
        "confirm", "ask_save_changes", "ask_recovery", "ask_text", "ask_item",
        "open_file", "open_files", "save_file", "choose_folder",
    )

    def __init__(self):
        super().__init__(None)
        self.calls: List[tuple] = []
        self.queues: Dict[str, deque] = defaultdict(deque)
        self.clipboard: List[str] = []

    def say(self, method: str, *answers: Any) -> None:
        self.queues[method].extend(answers)

    def called(self, method: str) -> List[tuple]:
        return [call[1:] for call in self.calls if call[0] == method]

    def _answer(self, method: str, *args):
        self.calls.append((method,) + args)
        if not self.queues[method]:
            raise AssertionError(
                f"Unexpected prompt {method}{args!r}; script it with say()"
            )
        return self.queues[method].popleft()

    # -- messages --------------------------------------------------------
    def info(self, title, text):
        self.calls.append(("info", title, text))

    def warn(self, title, text):
        self.calls.append(("warn", title, text))

    def error(self, title, text):
        self.calls.append(("error", title, text))

    def about(self, title, text):
        self.calls.append(("about", title, text))

    # -- questions -------------------------------------------------------
    def confirm(self, title, text, accept="OK", informative="", destructive=False):
        return self._answer("confirm", title, text)

    def ask_save_changes(self, project_name):
        return self._answer("ask_save_changes", project_name)

    def ask_recovery(self, project_name, autosave):
        return self._answer("ask_recovery", project_name, autosave)

    def ask_text(self, title, label, text=""):
        return self._answer("ask_text", title, label, text)

    def ask_item(self, title, label, items):
        return self._answer("ask_item", title, label, tuple(items))

    def open_file(self, title, start_dir, filters):
        return self._answer("open_file", title, start_dir)

    def open_files(self, title, start_dir, filters):
        return self._answer("open_files", title, start_dir)

    def save_file(self, title, start_path, filters):
        return self._answer("save_file", title, start_path)

    def choose_folder(self, title, start_dir):
        return self._answer("choose_folder", title, start_dir)

    # -- the OS ----------------------------------------------------------
    def open_folder(self, path):
        self.calls.append(("open_folder", Path(path)))
        return True

    def reveal(self, path):
        self.calls.append(("reveal", Path(path)))
        return True

    def open_url(self, url):
        self.calls.append(("open_url", url))
        return True

    def copy_text(self, text):
        self.calls.append(("copy_text", text))
        self.clipboard.append(text)


class WorkspaceFixture(ProjectFixture):
    """A ``WorkspaceWindow`` with scripted prompts and a throw-away settings
    file, plus a folder of fake videos."""

    @classmethod
    def setUpClass(cls):
        if not HAS_RUNTIME:
            raise unittest.SkipTest("GUI runtime is not installed")
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        # PyQt6 aborts the whole process when an exception escapes a slot and
        # sys.excepthook is the default one; record it and fail the test.
        self.slot_errors: List[str] = []
        previous = sys.excepthook
        sys.excepthook = lambda kind, value, tb: self.slot_errors.append(
            "".join(traceback.format_exception(kind, value, tb))
        )
        self.addCleanup(self.check_slot_errors, previous)
        self.settings = temp_app_settings(self)
        self.prompts = ScriptedPrompts()
        self.window = WorkspaceWindow(self.settings, self.prompts)
        self.addCleanup(self.dispose_window)
        self.videos = [
            self.write_video(f"videos/m{n}.mp4", b"x" * n) for n in (1, 2, 3)
        ]

    def check_slot_errors(self, previous):
        sys.excepthook = previous
        self.assertEqual(self.slot_errors, [], "an exception escaped a slot")

    def dispose_window(self):
        window = self.window
        if window is None:
            return
        # Discard rather than prompt: tests that care script the prompt.
        window._skip_close_prompt = True
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.window = None

    # -- helpers ---------------------------------------------------------
    def new_project_at(self, name="study.bmaproj"):
        """Create a project through the window's New action."""
        path = self.root / name
        self.prompts.say("save_file", path)
        self.assertTrue(self.window.new_project())
        return path

    def add_videos(self, paths=None):
        result = self.window.video_panel.add_paths(paths or self.videos)
        self.app.processEvents()
        return result


def make_temp_dir(testcase) -> Path:
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    testcase.addCleanup(tmp.cleanup)
    return Path(tmp.name).resolve()
