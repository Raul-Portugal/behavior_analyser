"""
gui/workspace/stages/videos_stage.py
Stage 1: the videos of the study.

What the stage shows depends on how far along the project is:

* **No project open**: a friendly empty state ("Start a new study" /
  "Open project"), the recent projects as cards and, when 3.x
  configurations are around, an import card (:class:`WelcomeView`);
* **A project without a paradigm** (or while the user is changing it): "What
  test are these videos from?" with a big :class:`ParadigmPicker`;
* **A project with a paradigm**: a compact summary of the test (with
  "Change"), the video table (:class:`VideoPanel`, whose drop zone stands in
  for an empty table) and a collapsible "Study details" card for the
  project's name and notes.

"Next: Arena & zones" is enabled once there is a paradigm and at least one
video.
"""
from typing import List, Optional, Tuple

from PyQt6.QtCore import QRectF, QTimer
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import (QHBoxLayout, QLineEdit, QPlainTextEdit,
                             QSizePolicy, QStackedWidget, QVBoxLayout, QWidget)

from core.project import Project
from gui.app_settings import AppSettings
from gui.workspace.prompts import Prompts
from gui.workspace.stages import status as stage_status
from gui.workspace.stages.base import Stage
from gui.workspace.theme import Radius, Spacing, current_palette
from gui.workspace.thumbnails import ThumbnailLoader
from gui.workspace.ui_kit import (PARADIGM_INFO, Callout, CollapsibleSection,
                                  GhostButton, Label, ParadigmPicker,
                                  _IconBadge, _rounded)
from gui.workspace.video_panel import VideoPanel
from gui.workspace.welcome import WelcomeView

PAGE_WELCOME = 0
PAGE_PARADIGM = 1
PAGE_VIDEOS = 2

PARADIGM_TITLE = "What test are these videos from?"
PARADIGM_SUBTITLE = (
    "This decides which zones, settings and results you get. "
    "You can change it later."
)
VIDEOS_SUBTITLE = (
    "Add the recordings you want to analyse. Each video can get its own "
    "trial times and zones later."
)
NOTES_COMMIT_MS = 600


class _ParadigmSummary(QWidget):
    """A compact pill: the test of the study and a "Change" button."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._badge = _IconBadge("maze-field", 30, shape="rounded")
        self.name_label = Label("", "subtitle")
        self.change_button = GhostButton("Change")
        self.change_button.setToolTip("Choose another test for this study")
        row = QHBoxLayout(self)
        row.setContentsMargins(Spacing.XS, Spacing.XXS, Spacing.XXS, Spacing.XXS)
        row.setSpacing(Spacing.XS)
        row.addWidget(self._badge)
        row.addWidget(self.name_label)
        row.addWidget(self.change_button)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set_paradigm(self, name: str) -> None:
        icon_name, description = PARADIGM_INFO.get(name, ("maze-field", ""))
        self._badge.set_icon_name(icon_name)
        self.name_label.setText(name)
        self.setToolTip(description)

    def paintEvent(self, event) -> None:
        palette = current_palette()
        painter = QPainter(self)
        _rounded(painter, QRectF(self.rect()), Radius.LG, palette.surface, palette.border)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        self.update()


class VideosStage(Stage):
    """Stage 1: videos and the paradigm they belong to."""

    stage_id = stage_status.STAGE_VIDEOS
    title = "Videos"
    subtitle = VIDEOS_SUBTITLE
    icon_name = "video"

    def __init__(self, app_settings: Optional[AppSettings] = None,
                 prompts: Optional[Prompts] = None,
                 loader: Optional[ThumbnailLoader] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.video_panel = VideoPanel(app_settings, prompts, loader=loader)
        self.welcome: Optional[WelcomeView] = None
        self._changing = False
        self._read_only = False
        self._build_pages()
        self._notes_timer = QTimer(self)
        self._notes_timer.setSingleShot(True)
        self._notes_timer.setInterval(NOTES_COMMIT_MS)
        self._notes_timer.timeout.connect(self.commit_pending)
        self.video_panel.status_message.connect(self._on_panel_message)
        self.refresh()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_pages(self) -> None:
        self.summary = _ParadigmSummary()
        self.summary.change_button.clicked.connect(lambda _checked=False: self.change_paradigm())
        self.header.add_action(self.summary)
        self.pages = QStackedWidget()
        self._welcome_slot = QWidget()
        QVBoxLayout(self._welcome_slot).setContentsMargins(0, 0, 0, 0)
        self.pages.addWidget(self._welcome_slot)
        self.pages.addWidget(self._build_paradigm_page())
        self.pages.addWidget(self._build_videos_page())
        self.set_content(self.pages)

    def _build_paradigm_page(self) -> QWidget:
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(0, Spacing.XS, 0, 0)
        column.setSpacing(Spacing.MD)
        self.picker = ParadigmPicker()
        self.picker.paradigm_chosen.connect(self._on_paradigm_chosen)
        column.addWidget(self.picker)
        self.paradigm_hint = Callout(
            "tip",
            "Not sure? Pick the test that matches how the videos were "
            "recorded. If some videos are already set up you will be asked "
            "before anything is cleared.",
        )
        column.addWidget(self.paradigm_hint)
        column.addStretch(1)
        bottom = QHBoxLayout()
        self.keep_button = GhostButton("Keep the current test", "arrow-left")
        self.keep_button.clicked.connect(lambda _checked=False: self.cancel_change())
        bottom.addWidget(self.keep_button)
        bottom.addStretch(1)
        column.addLayout(bottom)
        return page

    def _build_videos_page(self) -> QWidget:
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(Spacing.SM)
        column.addWidget(self.video_panel, 1)

        settings = self.video_panel.app_settings
        self.details = CollapsibleSection(
            "Study details", expanded=False, settings=settings,
            settings_key="videos_study_details", animated=False,
        )
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Name of the study")
        self.name_edit.editingFinished.connect(self.commit_pending)
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Notes about the study: the cohort, the setup, anything to remember...")
        self.notes_edit.setMaximumHeight(84)
        self.notes_edit.textChanged.connect(self._on_notes_edited)
        self.details.body.addWidget(Label("Name", "caption"))
        self.details.body.addWidget(self.name_edit)
        self.details.body.addWidget(Label("Notes", "caption"))
        self.details.body.addWidget(self.notes_edit)
        column.addWidget(self.details)
        return page

    # ------------------------------------------------------------------
    # The window's hooks
    # ------------------------------------------------------------------
    def attach(self, window) -> None:
        super().attach(window)
        self.welcome = WelcomeView(window)
        self._welcome_slot.layout().addWidget(self.welcome)
        self.refresh()

    def on_project_changed(self, project: Optional[Project]) -> None:
        self.commit_pending()
        self._changing = False
        if self.welcome is not None:
            self.welcome.on_project_changed(project)
        self.refresh()

    def on_project_modified(self) -> None:
        self.refresh()

    def on_enter(self) -> None:
        if self.welcome is not None and self.project is None:
            self.welcome.refresh()
        self.refresh()
        self.video_panel.request_durations()

    def on_leave(self) -> bool:
        self.commit_pending()
        return True

    def set_busy(self, busy: bool) -> None:
        """A run is in progress: the study cannot be edited."""
        self._read_only = bool(busy)
        self.summary.change_button.setEnabled(not busy)
        self.name_edit.setReadOnly(busy)
        self.notes_edit.setReadOnly(busy)
        self.picker.setEnabled(not busy)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def page(self) -> int:
        """Which of the three pages is showing."""
        return self.pages.currentIndex()

    def _paradigm(self) -> str:
        project = self.project
        return "" if project is None else project.effective_paradigm()

    def _target_page(self) -> int:
        project = self.project
        if project is None:
            return PAGE_WELCOME
        if self._changing or not stage_status.paradigm_known(project):
            return PAGE_PARADIGM
        return PAGE_VIDEOS

    def refresh(self) -> None:
        """Show the page and texts that fit the project's state."""
        page = self._target_page()
        self.pages.setCurrentIndex(page)
        self.header.setVisible(page != PAGE_WELCOME)
        self.summary.setVisible(page == PAGE_VIDEOS)
        paradigm = self._paradigm()
        self.picker.set_selected(paradigm if paradigm in self.picker.names() else None)
        if page == PAGE_PARADIGM:
            self.header.set_texts(PARADIGM_TITLE, PARADIGM_SUBTITLE)
            self.header.set_icon_name("maze-field")
            self.keep_button.setVisible(self._changing)
        elif page == PAGE_VIDEOS:
            self.header.set_texts(self.title, VIDEOS_SUBTITLE)
            self.header.set_icon_name(self.icon_name)
            self.summary.set_paradigm(self._paradigm())
            self._sync_details()
        self.refresh_footer()

    def _sync_details(self) -> None:
        """Show the project's name and notes (never over what is being typed)."""
        project = self.project
        if project is None:
            return
        if not self.name_edit.hasFocus() and self.name_edit.text() != project.name:
            self.name_edit.setText(project.name)
        typing = self.notes_edit.hasFocus() or self._notes_timer.isActive()
        if not typing and self.notes_edit.toPlainText() != project.notes:
            self.notes_edit.blockSignals(True)
            try:
                self.notes_edit.setPlainText(project.notes)
            finally:
                self.notes_edit.blockSignals(False)

    # ------------------------------------------------------------------
    # Paradigm
    # ------------------------------------------------------------------
    def change_paradigm(self) -> None:
        """Show the picker again so another test can be chosen."""
        if self._read_only or self.project is None:
            return
        self._changing = True
        self.refresh()

    def cancel_change(self) -> None:
        self._changing = False
        self.refresh()

    def _on_paradigm_chosen(self, name: str) -> None:
        window = self.workspace
        if window is None:
            return
        if window.change_paradigm(name):
            self._changing = False
        self.refresh()

    # ------------------------------------------------------------------
    # Study details
    # ------------------------------------------------------------------
    def _on_notes_edited(self) -> None:
        if not self._read_only:
            self._notes_timer.start()

    def commit_pending(self) -> None:
        """Write the name and notes being edited into the project."""
        self._notes_timer.stop()
        project = self.project
        if project is None or self._read_only:
            return
        if self.pages.currentIndex() != PAGE_VIDEOS:
            return
        # Read both first: the first change refreshes the page from the project.
        name = self.name_edit.text().strip()
        notes = self.notes_edit.toPlainText()
        if name and name != project.name:
            project.set_name(name)
        elif not name and self.name_edit.text() != project.name:
            self.name_edit.setText(project.name)   # a blank name is refused
        project.set_notes(notes)

    def _on_panel_message(self, text: str) -> None:
        if self.workspace is not None:
            self.workspace.show_message(text)

    # ------------------------------------------------------------------
    # The footer
    # ------------------------------------------------------------------
    def next_label(self) -> Optional[str]:
        return self.next_stage_title

    def can_advance(self) -> Tuple[bool, str]:
        project = self.project
        if project is None:
            return False, stage_status.NEEDS_PROJECT
        if not stage_status.paradigm_known(project):
            return False, "Choose the test these videos are from"
        if not project.videos:
            return False, "Add at least one video"
        return True, ""

    def footer_text(self):
        project = self.project
        if project is None:
            return "Open or create a study to begin", None
        text, chip = super().footer_text()
        keys: List[str] = self.video_panel.selected_keys()
        if keys and self.pages.currentIndex() == PAGE_VIDEOS:
            text = f"{text}  ·  {len(keys)} selected" if text else f"{len(keys)} selected"
        return text, chip

    def on_selection_changed(self, keys: List[str]) -> None:
        self.refresh_footer()

    def shutdown(self) -> None:
        """Stop the timers (the window is closing)."""
        self.commit_pending()
        self._notes_timer.stop()
        self.video_panel.shutdown()


__all__ = ["VideosStage", "PAGE_WELCOME", "PAGE_PARADIGM", "PAGE_VIDEOS"]
