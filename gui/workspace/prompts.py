"""
gui/workspace/prompts.py
Every question the workspace asks the user, and every "open this in the OS"
action, behind one small object.

The window and the panels never call ``QMessageBox`` / ``QFileDialog``
directly: they call a :class:`Prompts`. Real use gets the dialogs below; tests
(and any future automation) hand in a subclass that answers from a script, so
no test ever blocks on a modal dialog.

``Prompts.parent`` is the widget the dialogs hang from; the window sets it to
itself.
"""
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices, QGuiApplication
from PyQt6.QtWidgets import (QFileDialog, QInputDialog, QMessageBox, QWidget)

from gui import ux_helpers
from gui.ux_helpers import Choice

logger = logging.getLogger(__name__)

PROJECT_FILTER = "Projects (*.bmaproj)"
SOURCE_FILTER = "Templates and plans (*.json);;All Files (*.*)"
OPEN_FILTER = (
    "Projects, templates and plans (*.bmaproj *.json);;"
    "Projects (*.bmaproj);;All Files (*.*)"
)
VIDEO_FILTER = "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
SHEET_FILTER = "Sample sheets (*.csv *.tsv *.txt);;All Files (*.*)"

# ask_save_changes answers
SAVE = "save"
DISCARD = "discard"
CANCEL = "cancel"


# ----------------------------------------------------------------------
# Opening things in the operating system
# ----------------------------------------------------------------------
def open_folder(path: os.PathLike) -> bool:
    """Open a folder in the file manager; False if it does not exist."""
    path = Path(path)
    if not path.is_dir():
        return False
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def reveal_in_folder(path: os.PathLike) -> bool:
    """Show a file selected in the file manager (its folder elsewhere)."""
    path = Path(path)
    folder = path if path.is_dir() else path.parent
    if not folder.is_dir():
        return False
    if path.is_file():
        try:
            if sys.platform.startswith("win"):
                import subprocess
                subprocess.Popen(["explorer", "/select,", str(path)])
                return True
            if sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", "-R", str(path)])
                return True
        except OSError:
            logger.warning("Could not reveal %s", path, exc_info=True)
    return open_folder(folder)


def open_url(url: str) -> bool:
    return QDesktopServices.openUrl(QUrl(url))


def copy_text(text: str) -> None:
    QGuiApplication.clipboard().setText(text)


# ----------------------------------------------------------------------
# The dialogs
# ----------------------------------------------------------------------
class Prompts:
    """The real dialogs. Override any method to script the answers."""

    def __init__(self, parent: Optional[QWidget] = None):
        self.parent = parent

    # -- messages --------------------------------------------------------
    def info(self, title: str, text: str) -> None:
        QMessageBox.information(self.parent, title, text)

    def warn(self, title: str, text: str) -> None:
        QMessageBox.warning(self.parent, title, text)

    def error(self, title: str, text: str) -> None:
        QMessageBox.critical(self.parent, title, text)

    def about(self, title: str, text: str) -> None:
        QMessageBox.about(self.parent, title, text)

    # -- questions -------------------------------------------------------
    def confirm(
        self,
        title: str,
        text: str,
        accept: str = "OK",
        informative: str = "",
        destructive: bool = False,
    ) -> bool:
        """A yes/no question; True when the user takes the ``accept`` button."""
        answer = ux_helpers.ask_choice(
            self.parent, title, text,
            [
                Choice(
                    "yes", accept,
                    QMessageBox.ButtonRole.DestructiveRole if destructive
                    else QMessageBox.ButtonRole.AcceptRole,
                ),
                Choice("no", "Cancel", QMessageBox.ButtonRole.RejectRole),
            ],
            default_key="no" if destructive else "yes",
            informative=informative,
            icon=QMessageBox.Icon.Warning if destructive
            else QMessageBox.Icon.Question,
        )
        return answer == "yes"

    def ask_save_changes(self, project_name: str) -> str:
        """``SAVE``, ``DISCARD`` or ``CANCEL`` for a project with changes."""
        answer = ux_helpers.ask_choice(
            self.parent, "Unsaved changes",
            f"Save the changes to “{project_name}”?",
            [
                Choice(SAVE, "Save"),
                Choice(DISCARD, "Don't save", QMessageBox.ButtonRole.DestructiveRole),
                Choice(CANCEL, "Cancel", QMessageBox.ButtonRole.RejectRole),
            ],
            default_key=SAVE,
            icon=QMessageBox.Icon.Warning,
        )
        return answer or CANCEL

    def ask_recovery(self, project_name: str, autosave: Path) -> str:
        """``"recover"``, ``"saved"`` (open the file) or ``"cancel"``."""
        answer = ux_helpers.ask_choice(
            self.parent, "Recover unsaved work?",
            f"“{project_name}” has an autosave that is newer than "
            "the saved project, probably from a session that ended without "
            "saving.",
            [
                Choice("recover", "Recover the autosave"),
                Choice("saved", "Open the saved project"),
                Choice("cancel", "Cancel", QMessageBox.ButtonRole.RejectRole),
            ],
            default_key="recover",
            informative=str(autosave),
        )
        return answer or "cancel"

    def ask_text(
        self, title: str, label: str, text: str = ""
    ) -> Optional[str]:
        value, accepted = QInputDialog.getText(
            self.parent, title, label, text=text
        )
        return value if accepted else None

    def ask_item(
        self, title: str, label: str, items: Sequence[str]
    ) -> Optional[str]:
        value, accepted = QInputDialog.getItem(
            self.parent, title, label, list(items), 0, False
        )
        return value if accepted else None

    # -- files -----------------------------------------------------------
    def open_file(self, title: str, start_dir: str, filters: str) -> Optional[Path]:
        name, _ = QFileDialog.getOpenFileName(
            self.parent, title, start_dir, filters
        )
        return Path(name) if name else None

    def open_files(self, title: str, start_dir: str, filters: str) -> List[Path]:
        names, _ = QFileDialog.getOpenFileNames(
            self.parent, title, start_dir, filters
        )
        return [Path(name) for name in names]

    def save_file(self, title: str, start_path: str, filters: str) -> Optional[Path]:
        name, _ = QFileDialog.getSaveFileName(
            self.parent, title, start_path, filters
        )
        return Path(name) if name else None

    def choose_folder(self, title: str, start_dir: str) -> Optional[Path]:
        name = QFileDialog.getExistingDirectory(self.parent, title, start_dir)
        return Path(name) if name else None

    # -- the OS ----------------------------------------------------------
    def open_folder(self, path: os.PathLike) -> bool:
        return open_folder(path)

    def reveal(self, path: os.PathLike) -> bool:
        return reveal_in_folder(path)

    def open_url(self, url: str) -> bool:
        return open_url(url)

    def copy_text(self, text: str) -> None:
        copy_text(text)

