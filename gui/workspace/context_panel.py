"""
gui/workspace/context_panel.py
The right pane of the workspace: what is known about the selected video(s).

* The **project box** (always shown): name, file and the paradigm chooser.
* **No selection**: a hint.
* **One video**: first-frame thumbnail (read off the GUI thread, cached),
  path, resolution, fps, duration, configured / verified / settings source,
  its issues, the last run with an "Open run folder" button and a notes
  editor.
* **Several videos**: counts and the bulk actions.

The panel edits nothing itself except the notes (``Project.set_video_notes``).
Every other action is announced with ``action_requested(name)``; the window
runs it on the current selection. ``paradigm_chosen(name)`` is emitted when
the user picks a paradigm in the combo (never for programmatic changes).
"""
import logging
from typing import Callable, List, Optional, Sequence

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
                             QLabel, QListWidget, QListWidgetItem,
                             QPlainTextEdit, QPushButton, QScrollArea,
                             QSizePolicy, QStackedWidget, QVBoxLayout, QWidget)

from core.project import (SEVERITY_ERROR, SEVERITY_INFO, STATUS_NEW, Project,
                          ProjectIssue, VideoEntry)
from gui.ux_helpers import format_duration
from gui.workspace.prompts import Prompts
from gui.workspace.thumbnails import ThumbnailLoader, VideoPreview
from mazes import AVAILABLE_MAZES

logger = logging.getLogger(__name__)

ACTION_VERIFY = "verify"
ACTION_UNVERIFY = "unverify"
ACTION_APPLY_TEMPLATE = "apply_template"
ACTION_REMOVE = "remove"
ACTION_RELINK = "relink"

NOTES_COMMIT_MS = 600
PLACEHOLDER_PARADIGM = "(choose a paradigm)"
ERROR_STYLE = "color: #d1242f;"

_STATUS_TEXT = {
    "new": "New",
    "configured": "Configured",
    "done": "Done",
    "failed": "Failed",
    "warning": "Done, needs quality review",
    "interrupted": "Interrupted",
}
_SOURCE_TEXT = {
    "template": "Project template",
    "custom": "Customised for this video",
    "aligned": "Aligned from the template",
}

IssuesFor = Callable[[str], List[ProjectIssue]]


def _blank(label: QLabel) -> None:
    label.setText("—")


ZERO_WIDTH_SPACE = "​"


def wrappable(text: str) -> str:
    """``text`` with break opportunities after path separators.

    A label cannot wrap a long path (it has no spaces), which would force the
    whole panel wider than its pane. The invisible characters let it wrap;
    tooltips carry the exact text.
    """
    for separator in ("\\", "/", "_"):
        text = text.replace(separator, separator + ZERO_WIDTH_SPACE)
    return text


def _selectable(label: QLabel) -> QLabel:
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    label.setWordWrap(True)
    return label


class ContextPanel(QWidget):
    """Details, notes and bulk actions for the selected videos."""

    paradigm_chosen = pyqtSignal(str)
    action_requested = pyqtSignal(str)

    PAGE_EMPTY = 0
    PAGE_SINGLE = 1
    PAGE_MULTI = 2

    def __init__(
        self,
        prompts: Optional[Prompts] = None,
        issues_for: Optional[IssuesFor] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.prompts = prompts or Prompts(self)
        self._issues_for: IssuesFor = issues_for or (lambda key: [])
        self._project: Optional[Project] = None
        self._keys: List[str] = []
        self._notes_key: Optional[str] = None     # video the editor shows
        self._updating = False
        self.loader = ThumbnailLoader(self)
        self.loader.loaded.connect(self._on_preview_loaded)
        self._notes_timer = QTimer(self)
        self._notes_timer.setSingleShot(True)
        self._notes_timer.setInterval(NOTES_COMMIT_MS)
        self._notes_timer.timeout.connect(self.commit_notes)
        self._build_ui()
        self.set_project(None)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        layout = QVBoxLayout(body)

        # -- project ---------------------------------------------------
        self.project_box = QGroupBox("Project")
        form = QFormLayout(self.project_box)
        self.project_name_label = _selectable(QLabel())
        self.project_file_label = _selectable(QLabel())
        self.paradigm_combo = QComboBox()
        self.paradigm_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.paradigm_combo.setMinimumContentsLength(12)
        self.paradigm_combo.setToolTip(
            "The paradigm of the whole project. Changing it clears the "
            "settings of configured videos."
        )
        self.paradigm_combo.activated.connect(self._on_paradigm_activated)
        form.addRow("Name", self.project_name_label)
        form.addRow("File", self.project_file_label)
        form.addRow("Paradigm", self.paradigm_combo)
        layout.addWidget(self.project_box)

        # -- pages -----------------------------------------------------
        self.pages = QStackedWidget()
        layout.addWidget(self.pages, 1)
        self.pages.addWidget(self._build_empty_page())
        self.pages.addWidget(self._build_single_page())
        self.pages.addWidget(self._build_multi_page())

    def _build_empty_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        self.empty_label = QLabel("Select a video to see its details.")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setWordWrap(True)
        self.empty_label.setEnabled(False)
        box.addWidget(self.empty_label)
        box.addStretch(1)
        return page

    def _build_single_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(0, 0, 0, 0)
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setMinimumHeight(120)
        self.thumbnail_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        self.thumbnail_label.setFrameShape(QLabel.Shape.StyledPanel)
        box.addWidget(self.thumbnail_label)

        form = QFormLayout()
        self.name_label = _selectable(QLabel())
        self.path_label = _selectable(QLabel())
        self.resolution_label = QLabel()
        self.fps_label = QLabel()
        self.duration_label = QLabel()
        self.configured_label = QLabel()
        self.source_label = QLabel()
        form.addRow("Video", self.name_label)
        form.addRow("Path", self.path_label)
        form.addRow("Resolution", self.resolution_label)
        form.addRow("Frame rate", self.fps_label)
        form.addRow("Duration", self.duration_label)
        form.addRow("State", self.configured_label)
        form.addRow("Settings", self.source_label)
        box.addLayout(form)

        box.addWidget(QLabel("Issues"))
        self.issues_list = QListWidget()
        self.issues_list.setWordWrap(True)
        self.issues_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.issues_list.setMaximumHeight(110)
        box.addWidget(self.issues_list)
        self.relink_button = QPushButton("Relink...")
        self.relink_button.setToolTip(
            "Point the project at the folder where this video is now"
        )
        self.relink_button.clicked.connect(
            lambda _checked=False: self.action_requested.emit(ACTION_RELINK)
        )
        box.addWidget(self.relink_button)

        self.last_run_label = _selectable(QLabel())
        run_row = QHBoxLayout()
        run_row.addWidget(self.last_run_label, 1)
        self.open_run_button = QPushButton("Open run folder")
        self.open_run_button.clicked.connect(self.open_run_folder)
        run_row.addWidget(self.open_run_button)
        box.addWidget(QLabel("Last run"))
        box.addLayout(run_row)

        box.addWidget(QLabel("Notes"))
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Notes about this video...")
        self.notes_edit.setMaximumHeight(110)
        self.notes_edit.textChanged.connect(self._on_notes_edited)
        box.addWidget(self.notes_edit)
        box.addStretch(1)
        return page

    def _build_multi_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(0, 0, 0, 0)
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(Qt.TextFormat.PlainText)
        box.addWidget(self.summary_label)
        self.bulk_buttons = {}
        for action, text, tip in (
            (ACTION_VERIFY, "Mark verified",
             "Mark the selected configured videos as checked"),
            (ACTION_UNVERIFY, "Mark unverified", "Remove the verified mark"),
            (ACTION_APPLY_TEMPLATE, "Apply template settings",
             "Give the selected videos the project template"),
            (ACTION_RELINK, "Relink...",
             "Point the project at the folder where the missing videos are now"),
            (ACTION_REMOVE, "Remove from project", "Remove the selected videos"),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(
                lambda _checked=False, name=action: self.action_requested.emit(name)
            )
            box.addWidget(button)
            self.bulk_buttons[action] = button
        box.addStretch(1)
        return page

    def _show_page(self, index: int) -> None:
        """Switch page; hidden pages must not make the panel taller."""
        for number in range(self.pages.count()):
            self.pages.widget(number).setSizePolicy(
                QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Preferred if number == index
                else QSizePolicy.Policy.Ignored,
            )
        self.pages.setCurrentIndex(index)

    # ------------------------------------------------------------------
    # Project and selection
    # ------------------------------------------------------------------
    def set_issues_provider(self, issues_for: IssuesFor) -> None:
        self._issues_for = issues_for

    def set_project(self, project: Optional[Project]) -> None:
        self.commit_notes()
        self._project = project
        self._keys = []
        self._notes_key = None
        self.refresh()

    def set_selection(self, keys: Sequence[str]) -> None:
        self.commit_notes()
        self._keys = list(keys)
        self.refresh()

    def selection(self) -> List[str]:
        return list(self._keys)

    def refresh(self) -> None:
        """Redraw everything from the project (after any change)."""
        project = self._project
        self._refresh_project_box()
        if project is not None:
            known = {video.key for video in project.videos}
            self._keys = [key for key in self._keys if key in known]
        if project is None or not self._keys:
            self._notes_key = None
            self._show_page(self.PAGE_EMPTY)
            self.empty_label.setText(
                "No project open." if project is None
                else "Select a video to see its details."
            )
        elif len(self._keys) == 1:
            self._show_single(project, project.entry(self._keys[0]))
        else:
            self._show_multi(project, project.entries_for(self._keys))

    def _refresh_project_box(self) -> None:
        project = self._project
        self.project_box.setEnabled(project is not None)
        if project is None:
            self.project_name_label.setText("—")
            self.project_file_label.setText("—")
            self._set_paradigm_items("")
            return
        self.project_name_label.setText(project.name)
        file_text = str(project.file_path) if project.file_path else "(not saved yet)"
        self.project_file_label.setText(wrappable(file_text))
        self.project_file_label.setToolTip(file_text)
        self._set_paradigm_items(project.effective_paradigm())

    def _set_paradigm_items(self, current: str) -> None:
        names = list(AVAILABLE_MAZES)
        if current and current not in names:
            names.append(current)       # a paradigm this build does not know
        combo = self.paradigm_combo
        combo.blockSignals(True)
        try:
            combo.clear()
            if not current:
                combo.addItem(PLACEHOLDER_PARADIGM, "")
            for name in names:
                combo.addItem(name, name)
            combo.setCurrentIndex(max(0, combo.findData(current)))
        finally:
            combo.blockSignals(False)

    def _on_paradigm_activated(self, index: int) -> None:
        name = self.paradigm_combo.itemData(index)
        if name:
            self.paradigm_chosen.emit(name)

    # -- one video -------------------------------------------------------
    def _show_single(self, project: Project, entry: VideoEntry) -> None:
        self._show_page(self.PAGE_SINGLE)
        self.name_label.setText(wrappable(entry.name))
        self.path_label.setText(wrappable(str(entry.path)))
        self.path_label.setToolTip(str(entry.path))
        state = "Not configured"
        if entry.settings is not None:
            state = "Configured and verified" if entry.verified else "Configured"
        self.configured_label.setText(state)
        self.source_label.setText(
            _SOURCE_TEXT.get(entry.settings_source, entry.settings_source)
            if entry.settings is not None else "—"
        )
        self._show_issues(entry)
        self._show_last_run(entry)
        self._show_notes(entry)
        exists = entry.path.is_file()
        self.relink_button.setVisible(not exists)
        self._show_preview(entry, exists)

    def _show_issues(self, entry: VideoEntry) -> None:
        self.issues_list.clear()
        issues = self._issues_for(entry.key)
        for issue in issues:
            item = QListWidgetItem(issue.message)
            if issue.severity == SEVERITY_ERROR:
                item.setForeground(Qt.GlobalColor.red)
            elif issue.severity == SEVERITY_INFO:
                item.setForeground(Qt.GlobalColor.gray)
            self.issues_list.addItem(item)
        if not issues:
            item = QListWidgetItem("No issues")
            item.setForeground(Qt.GlobalColor.gray)
            self.issues_list.addItem(item)

    def has_issues(self) -> bool:
        """True if the shown video lists real issues (not the placeholder)."""
        return bool(
            self._keys and len(self._keys) == 1
            and self._issues_for(self._keys[0])
        )

    def _show_last_run(self, entry: VideoEntry) -> None:
        status = _STATUS_TEXT.get(entry.last_status, entry.last_status)
        if entry.last_run_dir is not None:
            text = f"{status}\n{entry.last_run_dir}"
        elif entry.last_status == STATUS_NEW or entry.settings is None:
            text = "Not analysed yet"
        else:
            text = status
        self.last_run_label.setText(text)
        self.last_run_label.setToolTip(
            str(entry.last_run_dir) if entry.last_run_dir else ""
        )
        self.open_run_button.setEnabled(
            entry.last_run_dir is not None and entry.last_run_dir.is_dir()
        )

    def _show_notes(self, entry: VideoEntry) -> None:
        if self._notes_key == entry.key and self.notes_edit.toPlainText() == entry.notes:
            return
        if (
            self._notes_key == entry.key
            and self._notes_timer.isActive()
        ):
            return      # the user is typing: their text wins until committed
        self._notes_key = entry.key
        self._updating = True
        try:
            self.notes_edit.setPlainText(entry.notes)
        finally:
            self._updating = False

    def _show_preview(self, entry: VideoEntry, exists: bool) -> None:
        if not exists:
            self.thumbnail_label.setPixmap(QPixmap())
            self.thumbnail_label.setText("File not found")
            self.thumbnail_label.setStyleSheet(ERROR_STYLE)
            for label in (self.resolution_label, self.fps_label,
                          self.duration_label):
                _blank(label)
            return
        self.thumbnail_label.setStyleSheet("")
        preview = self.loader.get(entry.path)
        if preview is None:
            self.thumbnail_label.setPixmap(QPixmap())
            self.thumbnail_label.setText("Loading preview...")
            for label in (self.resolution_label, self.fps_label,
                          self.duration_label):
                _blank(label)
            self.loader.request(entry.path)
            return
        self._apply_preview(preview)

    def _apply_preview(self, preview: VideoPreview) -> None:
        if preview.image is not None:
            pixmap = QPixmap.fromImage(preview.image)
            self.thumbnail_label.setText("")
            self.thumbnail_label.setPixmap(pixmap)
        else:
            self.thumbnail_label.setPixmap(QPixmap())
            self.thumbnail_label.setText(preview.error or "No preview")
        self.resolution_label.setText(
            f"{preview.width} × {preview.height}" if preview.width else "—"
        )
        self.fps_label.setText(f"{preview.fps:.2f} fps" if preview.fps > 0 else "—")
        self.duration_label.setText(
            format_duration(preview.duration) if preview.duration > 0 else "—"
        )

    def _on_preview_loaded(self, path_text: str) -> None:
        project = self._project
        if project is None or len(self._keys) != 1:
            return
        try:
            entry = project.entry(self._keys[0])
        except KeyError:
            return
        if str(entry.path) != path_text:
            return
        preview = self.loader.get(entry.path)
        if preview is not None:
            self._apply_preview(preview)

    # -- several videos --------------------------------------------------
    def _show_multi(self, project: Project, entries: List[VideoEntry]) -> None:
        self._show_page(self.PAGE_MULTI)
        configured = sum(1 for e in entries if e.settings is not None)
        verified = sum(1 for e in entries if e.verified)
        with_issues = sum(1 for e in entries if self._issues_for(e.key))
        missing = sum(1 for e in entries if not e.path.is_file())
        lines = [
            f"{len(entries)} videos selected",
            f"{configured} configured, {verified} verified",
            f"{with_issues} with issues",
        ]
        if missing:
            lines.append(f"{missing} with a missing file")
        self.summary_label.setText("\n".join(lines))
        self.bulk_buttons[ACTION_APPLY_TEMPLATE].setEnabled(
            project.template_settings is not None
        )
        self.bulk_buttons[ACTION_RELINK].setVisible(missing > 0)

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------
    def _on_notes_edited(self) -> None:
        if not self._updating:
            self._notes_timer.start()

    def commit_notes(self) -> None:
        """Write the notes editor into the project (no-op when unchanged)."""
        self._notes_timer.stop()
        project, key = self._project, self._notes_key
        if project is None or key is None:
            return
        try:
            entry = project.entry(key)
        except KeyError:
            return
        text = self.notes_edit.toPlainText()
        if text != entry.notes:
            project.set_video_notes(key, text)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def open_run_folder(self) -> bool:
        project = self._project
        if project is None or len(self._keys) != 1:
            return False
        folder = project.entry(self._keys[0]).last_run_dir
        if folder is None:
            return False
        if self.prompts.open_folder(folder):
            return True
        self.prompts.warn("Open run folder", f"The folder was not found:\n{folder}")
        return False

    def shutdown(self) -> None:
        """Stop the background reader; call before the window goes away."""
        self.commit_notes()
        self.loader.shutdown()


__all__ = [
    "ContextPanel", "wrappable", "ACTION_VERIFY", "ACTION_UNVERIFY", "ACTION_APPLY_TEMPLATE",
    "ACTION_REMOVE", "ACTION_RELINK",
]
