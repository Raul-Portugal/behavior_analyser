"""
gui/workspace/welcome.py
The one-time welcome card of the 4.0 workspace.

When the workspace starts with no project to reopen, but the classic interface
remembers recent templates or batch plans, the card offers the ways forward:
create or open a project, import one of those 3.x configurations, or go back
to the classic interface. It covers the tab area, never blocks anything, and
is remembered as dismissed (``AppSettings.welcome_dismissed``) once the user
has dealt with it, so it appears at most once.

The card only calls the window's existing routing (``new_project``,
``open_project``, ``import_template``, ``import_batch_plan``,
``switch_to_classic``) and asks questions through ``window.prompts``.
"""
from pathlib import Path
from typing import List, Optional, Tuple

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton,
                             QVBoxLayout, QWidget)

from core.models import BatchSettings
from core.version import APP_NAME
from gui.app_settings import AppSettings
from gui.workspace.prompts import SOURCE_FILTER

MAX_LISTED_SOURCES = 3

KIND_TEMPLATE = "template"
KIND_PLAN = "batch_plan"
_KIND_LABELS = {KIND_TEMPLATE: "template", KIND_PLAN: "batch plan"}


def recent_import_sources(
    settings: AppSettings, limit: int = MAX_LISTED_SOURCES
) -> List[Tuple[Path, str]]:
    """Up to ``limit`` recent 3.x templates / plans as ``(path, kind)``.

    Most recent first; files that are gone or unreadable are skipped.
    """
    found: List[Tuple[Path, str]] = []
    for path in settings.recent_files():
        kind = BatchSettings.detect_file_type(path)
        if kind in _KIND_LABELS:
            found.append((path, kind))
        if len(found) >= limit:
            break
    return found


class WelcomeCard(QFrame):
    """Overlay on the workspace tabs, shown at most once at startup."""

    def __init__(self, window):
        super().__init__(window.tabs)
        self.window = window
        self.sources: List[Tuple[Path, str]] = []
        self.setObjectName("welcome_card")
        self.setAutoFillBackground(True)
        self._build_ui()
        window.tabs.installEventFilter(self)
        window.project_changed.connect(self._on_project_changed)
        self.hide()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        self.body = QWidget()
        self.body.setMaximumWidth(560)
        layout = QVBoxLayout(self.body)
        layout.setSpacing(10)

        title = QLabel(f"Welcome to {APP_NAME} 4.0")
        title.setObjectName("welcome_title")
        title.setStyleSheet("font-size: 18pt; font-weight: bold;")
        layout.addWidget(title)
        self.intro = QLabel(
            "Everything now lives in one window: a project holds your videos, "
            "their settings and the run history. Create a project, open one, "
            "or bring your 3.x configurations along."
        )
        self.intro.setWordWrap(True)
        layout.addWidget(self.intro)

        self.create_button = QPushButton("Create project")
        self.create_button.clicked.connect(self._create)
        self.open_button = QPushButton("Open project")
        self.open_button.clicked.connect(self._open)
        buttons = QHBoxLayout()
        buttons.addWidget(self.create_button)
        buttons.addWidget(self.open_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.import_button = QPushButton(
            "Import your recent 3.x configuration…"
        )
        self.import_button.setToolTip(
            "Start a project from a saved template or batch plan"
        )
        self.import_button.clicked.connect(self._import_chosen)
        layout.addWidget(self.import_button, 0, Qt.AlignmentFlag.AlignLeft)

        self.sources_label = QLabel("Recently used:")
        layout.addWidget(self.sources_label)
        self.source_box = QVBoxLayout()
        self.source_box.setContentsMargins(16, 0, 0, 0)
        layout.addLayout(self.source_box)
        self.source_buttons: List[QPushButton] = []

        bottom = QHBoxLayout()
        self.classic_button = QPushButton("Use classic interface")
        self.classic_button.clicked.connect(self._classic)
        self.dismiss_button = QPushButton("Dismiss")
        self.dismiss_button.setToolTip("Do not show this card again")
        self.dismiss_button.clicked.connect(self.dismiss)
        bottom.addWidget(self.classic_button)
        bottom.addStretch(1)
        bottom.addWidget(self.dismiss_button)
        layout.addSpacing(8)
        layout.addLayout(bottom)

        row.addWidget(self.body, 10)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(1)

    def _fill_sources(self) -> None:
        for button in self.source_buttons:
            self.source_box.removeWidget(button)
            button.deleteLater()
        self.source_buttons = []
        for path, kind in self.sources:
            button = QPushButton(f"{path.name}  ({_KIND_LABELS[kind]})")
            button.setFlat(True)
            button.setStyleSheet("text-align: left;")
            button.setToolTip(str(path))
            button.clicked.connect(
                lambda _checked=False, p=path, k=kind: self._import(p, k)
            )
            self.source_box.addWidget(button)
            self.source_buttons.append(button)
        self.sources_label.setVisible(bool(self.sources))

    # ------------------------------------------------------------------
    # When to show
    # ------------------------------------------------------------------
    def show_if_needed(self) -> bool:
        """Show the card if this is a first start that has 3.x history."""
        window = self.window
        settings = window.app_settings
        if (
            window.project is not None
            or settings.last_project() is not None
            or settings.welcome_dismissed()
        ):
            return False
        self.sources = recent_import_sources(settings)
        if not self.sources:
            return False
        self._fill_sources()
        self._cover()
        self.show()
        self.raise_()
        return True

    def _cover(self) -> None:
        self.setGeometry(self.window.tabs.rect())

    def eventFilter(self, obj: Optional[QObject], event: Optional[QEvent]) -> bool:
        if obj is self.window.tabs and event is not None \
                and event.type() == QEvent.Type.Resize and not self.isHidden():
            self._cover()
        return False

    # ------------------------------------------------------------------
    # Dismissal
    # ------------------------------------------------------------------
    def dismiss(self) -> None:
        """Hide the card for good."""
        self.window.app_settings.set_welcome_dismissed(True)
        self.window.app_settings.sync()
        self.hide()

    def _on_project_changed(self, project) -> None:
        # Opening anything means the user found their way: never show again.
        if project is not None and not self.isHidden():
            self.dismiss()

    # ------------------------------------------------------------------
    # Actions (all routed through the window)
    # ------------------------------------------------------------------
    def _create(self) -> None:
        self.window.new_project()

    def _open(self) -> None:
        self.window.open_project()

    def _classic(self) -> None:
        if self.window.switch_to_classic():
            self.dismiss()

    def _import(self, path: Path, kind: str) -> bool:
        if kind == KIND_PLAN:
            return self.window.import_batch_plan(path)
        return self.window.import_template(path)

    def _import_chosen(self) -> None:
        """Pick any template or plan with the file dialog and import it."""
        window = self.window
        path = window.prompts.open_file(
            "Select a template or batch plan",
            window.app_settings.dialog_dir("config"), SOURCE_FILTER,
        )
        if path is None:
            return
        kind = BatchSettings.detect_file_type(path)
        if kind not in _KIND_LABELS:
            window.prompts.error(
                "Import",
                f"{path.name} is not a template or a batch plan.",
            )
            return
        self._import(path, kind)
