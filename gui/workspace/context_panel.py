"""
gui/workspace/context_panel.py
What is known about the selected video(s), shown in a slide-out drawer.

The drawer (:class:`DetailsDrawer`) sits on the right edge of the stage area
and is hidden until the user asks for it: the "Details" button of a video
row, Ctrl+I, or ``window.show_details()``. It overlays the stage instead of
squeezing it, so the stages keep the whole width.

Inside it, :class:`ContextPanel` shows

* **No selection**: a hint.
* **One video**: first-frame thumbnail (read off the GUI thread, cached),
  name, status chip, path, resolution, frame rate, duration, trial window,
  where its zones come from, its issues, the last run with an "Open run
  folder" button, notes and the actions (verify, reveal, remove).
* **Several videos**: counts and the bulk actions.

The panel edits nothing itself except the notes (``Project.set_video_notes``).
Every other action is announced with ``action_requested(name)``; the window
runs it on the current selection.

While the drawer is closed the panel does not redraw on every selection
change (``set_active``); it catches up when the drawer opens.
"""
import logging
from typing import Callable, List, Optional, Sequence, Tuple

from PyQt6.QtCore import (QEasingCurve, QEvent, QObject, QPoint,
                          QPropertyAnimation, QRect, Qt, QTimer, pyqtSignal)
from PyQt6.QtGui import QColor, QPainter, QPixmap
from PyQt6.QtWidgets import (QFormLayout, QFrame, QHBoxLayout, QLabel,
                             QListWidget, QListWidgetItem, QPlainTextEdit,
                             QScrollArea, QSizePolicy,
                             QStackedWidget, QVBoxLayout, QWidget)

from core.project import (SEVERITY_ERROR, SEVERITY_INFO, STATUS_NEW, Project,
                          ProjectIssue, TRIAL_OK, VideoEntry)
from gui.project_table_model import format_trial
from gui.ux_helpers import format_duration
from gui.workspace.prompts import Prompts
from gui.workspace.theme import Spacing, current_palette
from gui.workspace.thumbnails import ThumbnailLoader, VideoPreview
from gui.workspace.ui_kit import (DangerButton, GhostButton, Label,
                                  SecondaryButton, StatusChip)

logger = logging.getLogger(__name__)

ACTION_VERIFY = "verify"
ACTION_UNVERIFY = "unverify"
ACTION_APPLY_TEMPLATE = "apply_template"
ACTION_REMOVE = "remove"
ACTION_RELINK = "relink"
ACTION_REVEAL = "reveal"

NOTES_COMMIT_MS = 600
ANIMATION_MS = 170

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
_ZONES_TEXT = {
    "template": "The project template's",
    "adjusted": "Adjusted for this video",
    "aligned": "Aligned from the template",
    "custom": "This video's own copy",
}

IssuesFor = Callable[[str], List[ProjectIssue]]
ChipFor = Callable[[str], Tuple[str, str]]


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


def _caption(text: str) -> QLabel:
    return Label(text, "caption")


class ContextPanel(QWidget):
    """Details, notes and bulk actions for the selected videos."""

    action_requested = pyqtSignal(str)

    PAGE_EMPTY = 0
    PAGE_SINGLE = 1
    PAGE_MULTI = 2

    def __init__(
        self,
        prompts: Optional[Prompts] = None,
        issues_for: Optional[IssuesFor] = None,
        parent: Optional[QWidget] = None,
        loader: Optional[ThumbnailLoader] = None,
        chip_for: Optional[ChipFor] = None,
    ):
        super().__init__(parent)
        self.prompts = prompts or Prompts(self)
        self._issues_for: IssuesFor = issues_for or (lambda key: [])
        self._chip_for: Optional[ChipFor] = chip_for
        self._project: Optional[Project] = None
        self._keys: List[str] = []
        self._notes_key: Optional[str] = None     # video the editor shows
        self._updating = False
        self._active = True
        self._dirty = False
        # A shared loader belongs to the window; an own one is shut down here.
        self._owns_loader = loader is None
        self.loader = loader if loader is not None else ThumbnailLoader(self)
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
        self.pages = QStackedWidget()
        outer.addWidget(self.pages)
        self.pages.addWidget(self._build_empty_page())
        self.pages.addWidget(self._scrolling(self._build_single_page()))
        self.pages.addWidget(self._scrolling(self._build_multi_page()))

    @staticmethod
    def _scrolling(page: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(page)
        return scroll

    def _build_empty_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(Spacing.MD, Spacing.LG, Spacing.MD, Spacing.MD)
        self.empty_label = Label("Select a video to see its details.", "muted", wrap=True)
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box.addWidget(self.empty_label)
        box.addStretch(1)
        return page

    def _build_single_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(Spacing.MD, Spacing.XS, Spacing.MD, Spacing.MD)
        box.setSpacing(Spacing.XS)
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setMinimumHeight(150)
        self.thumbnail_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        self.thumbnail_label.setFrameShape(QLabel.Shape.StyledPanel)
        box.addWidget(self.thumbnail_label)

        self.name_label = _selectable(Label("", "subtitle"))
        box.addSpacing(Spacing.XXS)
        box.addWidget(self.name_label)
        chips = QHBoxLayout()
        chips.setSpacing(Spacing.XS)
        self.status_chip = StatusChip("neutral", "")
        chips.addWidget(self.status_chip)
        chips.addStretch(1)
        box.addLayout(chips)

        form = QFormLayout()
        form.setContentsMargins(0, Spacing.XS, 0, 0)
        form.setHorizontalSpacing(Spacing.MD)
        form.setVerticalSpacing(Spacing.XXS)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.path_label = _selectable(QLabel())
        self.path_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        self.path_label.setMinimumWidth(40)
        self.resolution_label = QLabel()
        self.fps_label = QLabel()
        self.duration_label = QLabel()
        self.trial_label = QLabel()
        self.zones_label = QLabel()
        self.configured_label = QLabel()
        self.source_label = QLabel()
        for text, label in (
            ("Resolution", self.resolution_label),
            ("Frame rate", self.fps_label),
            ("Duration", self.duration_label),
            ("Trial", self.trial_label),
            ("Zones", self.zones_label),
            ("State", self.configured_label),
            ("Settings", self.source_label),
        ):
            caption = Label(text, "muted")
            form.addRow(caption, label)
        box.addLayout(form)
        # The path is long: it gets the full width, under its caption.
        box.addSpacing(Spacing.XXS)
        box.addWidget(Label("Path", "muted"))
        box.addWidget(self.path_label)

        box.addSpacing(Spacing.XS)
        box.addWidget(_caption("Issues"))
        self.issues_list = QListWidget()
        self.issues_list.setWordWrap(True)
        self.issues_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.issues_list.setMaximumHeight(110)
        self.issues_list.setMinimumWidth(0)
        self.issues_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        box.addWidget(self.issues_list)
        self.relink_button = SecondaryButton("Relink...", "reset")
        self.relink_button.setToolTip(
            "Point the project at the folder where this video is now"
        )
        self.relink_button.clicked.connect(
            lambda _checked=False: self.action_requested.emit(ACTION_RELINK)
        )
        box.addWidget(self.relink_button)

        box.addSpacing(Spacing.XS)
        box.addWidget(_caption("Last run"))
        self.last_run_label = _selectable(QLabel())
        box.addWidget(self.last_run_label)
        self.open_run_button = SecondaryButton("Open run folder", "folder")
        self.open_run_button.clicked.connect(self.open_run_folder)
        run_row = QHBoxLayout()
        run_row.addWidget(self.open_run_button)
        run_row.addStretch(1)
        box.addLayout(run_row)

        box.addSpacing(Spacing.XS)
        box.addWidget(_caption("Notes"))
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Notes about this video...")
        self.notes_edit.setMaximumHeight(110)
        self.notes_edit.textChanged.connect(self._on_notes_edited)
        box.addWidget(self.notes_edit)

        box.addSpacing(Spacing.XS)
        actions = QHBoxLayout()
        actions.setSpacing(Spacing.XS)
        self.verify_button = SecondaryButton("Mark verified", "check")
        self.verify_button.setToolTip("Record that you checked this video's settings")
        self.verify_button.clicked.connect(self._on_verify_clicked)
        self.reveal_button = SecondaryButton("Show in folder", "folder")
        self.reveal_button.clicked.connect(
            lambda _checked=False: self.action_requested.emit(ACTION_REVEAL)
        )
        self.remove_button = DangerButton("Remove", "trash")
        self.remove_button.setToolTip("Remove this video from the project")
        self.remove_button.clicked.connect(
            lambda _checked=False: self.action_requested.emit(ACTION_REMOVE)
        )
        actions.addWidget(self.verify_button)
        actions.addWidget(self.reveal_button)
        actions.addStretch(1)
        actions.addWidget(self.remove_button)
        box.addLayout(actions)
        box.addStretch(1)
        return page

    def _build_multi_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(Spacing.MD, Spacing.XS, Spacing.MD, Spacing.MD)
        box.setSpacing(Spacing.XS)
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(Qt.TextFormat.PlainText)
        box.addWidget(self.summary_label)
        self.bulk_buttons = {}
        for action, text, tip, kind in (
            (ACTION_VERIFY, "Mark verified",
             "Mark the selected configured videos as checked", SecondaryButton),
            (ACTION_UNVERIFY, "Mark unverified", "Remove the verified mark",
             SecondaryButton),
            (ACTION_APPLY_TEMPLATE, "Apply template settings",
             "Give the selected videos the project template", SecondaryButton),
            (ACTION_RELINK, "Relink...",
             "Point the project at the folder where the missing videos are now",
             SecondaryButton),
            (ACTION_REMOVE, "Remove from project", "Remove the selected videos",
             DangerButton),
        ):
            button = kind(text)
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

    def set_chip_provider(self, chip_for: Optional[ChipFor]) -> None:
        self._chip_for = chip_for

    def set_active(self, active: bool) -> None:
        """An inactive panel (its drawer is closed) skips redrawing."""
        self._active = bool(active)
        if self._active and self._dirty:
            self.refresh()

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
        if not self._active:
            self._dirty = True
            return
        self._dirty = False
        project = self._project
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
        self._show_chip(entry)
        self._show_trial_and_zones(project, entry)
        self._show_issues(entry)
        self._show_last_run(entry)
        self._show_notes(entry)
        exists = entry.path.is_file()
        self.relink_button.setVisible(not exists)
        self.verify_button.setText("Mark unverified" if entry.verified else "Mark verified")
        self.verify_button.setEnabled(entry.settings is not None)
        self.reveal_button.setEnabled(exists)
        self._show_preview(entry, exists)

    def _show_chip(self, entry: VideoEntry) -> None:
        if self._chip_for is not None:
            try:
                kind, text = self._chip_for(entry.key)
            except (KeyError, AttributeError):
                kind, text = "neutral", ""
        else:
            kind, text = ("info", "Configured") if entry.settings else ("neutral", "Not set up")
        self.status_chip.set_status(kind, text)
        self.status_chip.setVisible(bool(text))

    def _show_trial_and_zones(self, project: Project, entry: VideoEntry) -> None:
        if entry.settings is None:
            self.trial_label.setText("—")
            self.zones_label.setText("—")
            return
        span = project.trial_span(entry)
        if project.trial_status(entry) != TRIAL_OK:
            self.trial_label.setText("Needs attention")
        elif span is not None:
            self.trial_label.setText(format_trial(*span))
        else:
            self.trial_label.setText("—")
        self.zones_label.setText(
            _ZONES_TEXT.get(project.zone_status(entry), project.zone_status(entry))
        )

    def _show_issues(self, entry: VideoEntry) -> None:
        self.issues_list.clear()
        issues = self._issues_for(entry.key)
        palette = current_palette()
        for issue in issues:
            item = QListWidgetItem(issue.message)
            if issue.severity == SEVERITY_ERROR:
                item.setForeground(QColor(palette.danger))
            elif issue.severity == SEVERITY_INFO:
                item.setForeground(QColor(palette.text_muted))
            self.issues_list.addItem(item)
        if not issues:
            item = QListWidgetItem("No issues")
            item.setForeground(QColor(palette.text_muted))
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
        palette = current_palette()
        if not exists:
            self.thumbnail_label.setPixmap(QPixmap())
            self.thumbnail_label.setText("File not found")
            self.thumbnail_label.setStyleSheet(f"color: {palette.danger};")
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
        if project is None or len(self._keys) != 1 or not self._active:
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
    def _on_verify_clicked(self) -> None:
        project = self._project
        verified = False
        if project is not None and len(self._keys) == 1:
            verified = project.entry(self._keys[0]).verified
        self.action_requested.emit(ACTION_UNVERIFY if verified else ACTION_VERIFY)

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
        """Stop the background reader (when the panel owns it); call before
        the window goes away."""
        self.commit_notes()
        if self._owns_loader:
            self.loader.shutdown()


class DetailsDrawer(QFrame):
    """A panel that slides in over the right edge of its parent.

    Hidden by default. ``open()`` / ``close_drawer()`` / ``toggle()`` move
    it (animated unless ``animation_ms`` is 0); ``toggled(bool)`` tells
    when it opened or closed. Esc or the close button closes it. The drawer
    is a child of the area it covers and follows that area's size.
    """

    toggled = pyqtSignal(bool)

    WIDTH = 392

    def __init__(self, panel: ContextPanel, parent: QWidget):
        super().__init__(parent)
        self.panel = panel
        self.animation_ms = ANIMATION_MS
        self._open = False
        self.setObjectName("details_drawer")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Video details")
        title = Label("Details", "subtitle")
        self.close_button = GhostButton("", "close")
        self.close_button.setFixedSize(32, 32)
        self.close_button.setToolTip("Close the details (Esc)")
        self.close_button.setAccessibleName("Close the details")
        self.close_button.clicked.connect(lambda _checked=False: self.close_drawer())
        header = QHBoxLayout()
        header.setContentsMargins(Spacing.MD, Spacing.SM, Spacing.XS, Spacing.XS)
        header.addWidget(title, 1)
        header.addWidget(self.close_button)
        column = QVBoxLayout(self)
        column.setContentsMargins(1, 0, 0, 0)    # the left border
        column.setSpacing(0)
        column.addLayout(header)
        column.addWidget(panel, 1)
        self._animation = QPropertyAnimation(self, b"pos", self)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.finished.connect(self._animation_done)
        self.hide()
        parent.installEventFilter(self)
        panel.set_active(False)

    # -- state ------------------------------------------------------------
    def is_open(self) -> bool:
        return self._open

    def animating(self) -> bool:
        return self._animation.state() == QPropertyAnimation.State.Running

    def drawer_width(self) -> int:
        parent = self.parentWidget()
        available = parent.width() if parent is not None else self.WIDTH
        return max(260, min(self.WIDTH, available - 48))

    def _shown_rect(self) -> QRect:
        parent = self.parentWidget()
        height = parent.height() if parent is not None else self.height()
        width = self.drawer_width()
        return QRect((parent.width() if parent else width) - width, 0, width, height)

    def _hidden_pos(self) -> QPoint:
        parent = self.parentWidget()
        return QPoint(parent.width() if parent is not None else self.WIDTH, 0)

    # -- opening and closing ------------------------------------------------
    def open(self, animated: Optional[bool] = None) -> None:
        if self._open:
            self.raise_()
            return
        self._open = True
        self.panel.set_active(True)
        rect = self._shown_rect()
        self._animation.stop()
        self.setGeometry(QRect(self._hidden_pos(), rect.size()))
        self.show()
        self.raise_()
        if (self.animation_ms > 0) if animated is None else animated:
            self._animation.setDuration(self.animation_ms)
            self._animation.setStartValue(self._hidden_pos())
            self._animation.setEndValue(rect.topLeft())
            self._animation.start()
        else:
            self.setGeometry(rect)
        self.toggled.emit(True)

    def close_drawer(self, animated: Optional[bool] = None) -> None:
        if not self._open:
            return
        self._open = False
        self.panel.commit_notes()
        self.panel.set_active(False)
        self._animation.stop()
        if self.isVisible() and ((self.animation_ms > 0) if animated is None else animated):
            self._animation.setDuration(self.animation_ms)
            self._animation.setStartValue(self.pos())
            self._animation.setEndValue(self._hidden_pos())
            self._animation.start()
        else:
            self.hide()
        self.toggled.emit(False)

    def toggle(self) -> None:
        if self._open:
            self.close_drawer()
        else:
            self.open()

    def _animation_done(self) -> None:
        if not self._open:
            self.hide()

    # -- geometry and painting ------------------------------------------------
    def eventFilter(self, watched: Optional[QObject], event: Optional[QEvent]) -> bool:
        if (watched is self.parentWidget() and event is not None
                and event.type() == QEvent.Type.Resize and self._open
                and not self.animating()):
            self.setGeometry(self._shown_rect())
        return False

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close_drawer()
            event.accept()
        else:
            super().keyPressEvent(event)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange):
            self.update()

    def paintEvent(self, event) -> None:
        palette = current_palette()
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(palette.surface))
        painter.fillRect(QRect(0, 0, 1, self.height()), QColor(palette.border_strong))


__all__ = [
    "ContextPanel", "DetailsDrawer", "wrappable", "ACTION_VERIFY",
    "ACTION_UNVERIFY", "ACTION_APPLY_TEMPLATE", "ACTION_REMOVE",
    "ACTION_RELINK", "ACTION_REVEAL",
]
