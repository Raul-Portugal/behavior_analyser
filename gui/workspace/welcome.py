"""
gui/workspace/welcome.py
The page the Videos stage shows while no project is open.

A friendly empty state ("Start a new study" / "Open project"), the recent
projects as cards, and, when the classic interface remembers recent 3.x
templates or batch plans, a card that offers to import one of them. The
import card is remembered as hidden (``AppSettings.welcome_dismissed``) once
the user has dealt with it or opened a project, so it does not nag; the
File > Import menu stays available.

The page only calls the window's existing routing (``new_project``,
``open_project``, ``open_path``, ``import_template``, ``import_batch_plan``,
``switch_to_classic``) and asks questions through ``window.prompts``.
"""
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from PyQt6.QtCore import QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (QAbstractButton, QGridLayout, QHBoxLayout,
                             QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from core.models import BatchSettings
from core.version import APP_NAME
from gui.app_settings import AppSettings
from gui.workspace.prompts import SOURCE_FILTER
from gui.workspace.theme import (Radius, Spacing, TypeScale, current_palette)
from gui.workspace.ui_kit import (Card, EmptyState, GhostButton, Label,
                                  SecondaryButton, _draw_icon, _elide,
                                  _focus_ring, _font, _rounded)

MAX_LISTED_SOURCES = 3
MAX_RECENT_CARDS = 3

KIND_TEMPLATE = "template"
KIND_PLAN = "batch_plan"
_KIND_LABELS = {KIND_TEMPLATE: "template", KIND_PLAN: "batch plan"}

IMPORT_BUTTON_TEXT = "Import 3.x configuration…"


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


def _modified_text(path: Path) -> str:
    try:
        stamp = datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return ""
    return f"Modified {stamp.day} {stamp.strftime('%b %Y')}"


class RecentProjectCard(QAbstractButton):
    """A clickable card for one recent project: its name, folder and date."""

    HEIGHT = 72

    def __init__(self, path: Path, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.path = Path(path)
        self._hover = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(self.HEIGHT)
        self.setToolTip(str(self.path))
        self.setAccessibleName(f"Open {self.path.stem}")
        self._folder = str(self.path.parent)
        self._when = _modified_text(self.path)

    def sizeHint(self) -> QSize:
        return QSize(300, self.HEIGHT)

    def enterEvent(self, event) -> None:
        self._hover = True
        self.update()

    def leaveEvent(self, event) -> None:
        self._hover = False
        self.update()

    def focusInEvent(self, event) -> None:
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.click()
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event) -> None:
        palette = current_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        _rounded(painter, rect, Radius.LG,
                 palette.surface_alt if self._hover else palette.surface,
                 palette.border_strong if self._hover else palette.border)
        if self.hasFocus():
            _focus_ring(painter, rect, Radius.LG, palette)
        tile = QRectF(rect.left() + 14, rect.center().y() - 19, 38, 38)
        _rounded(painter, tile, 10, palette.tint("primary", 0.12))
        _draw_icon(painter, self, "folder", palette.primary, tile.adjusted(9, 9, -9, -9))
        left = tile.right() + 12
        width = rect.right() - 14 - left
        title_font = _font(self.font(), TypeScale.SUBTITLE, True)
        small_font = _font(self.font(), TypeScale.CAPTION)
        painter.setFont(title_font)
        painter.setPen(QColor(palette.text))
        painter.drawText(QRectF(left, rect.top() + 11, width, 22),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         _elide(self.path.stem, title_font, width))
        painter.setFont(small_font)
        painter.setPen(QColor(palette.text_muted))
        painter.drawText(QRectF(left, rect.top() + 33, width, 16),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         _elide(self._folder, small_font, width))
        if self._when:
            painter.drawText(QRectF(left, rect.top() + 49, width, 16),
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             self._when)


class WelcomeView(QWidget):
    """The Videos stage while no project is open."""

    COLUMNS = 1

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.sources: List[Tuple[Path, str]] = []
        self.recent_cards: List[RecentProjectCard] = []
        self.source_buttons: List[GhostButton] = []
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.viewport().setAutoFillBackground(False)
        outer.addWidget(scroll)
        holder = QWidget()
        holder.setAutoFillBackground(False)
        scroll.setWidget(holder)
        row = QHBoxLayout(holder)
        row.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        row.addStretch(1)
        self.body = QWidget()
        self.body.setMaximumWidth(1040)
        self.body.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        columns = QHBoxLayout(self.body)
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(Spacing.XL)
        row.addWidget(self.body, 100)
        row.addStretch(1)

        # -- left: the way in ------------------------------------------------------
        left = QVBoxLayout()
        left.setSpacing(Spacing.SM)
        self.hero = EmptyState(
            "maze-field", "Start a new study",
            "A study keeps your videos, their settings and every analysis run "
            "together in one file.",
        )
        self.create_button = self.hero.add_action("Start a new study", True, "plus")
        self.open_button = self.hero.add_action("Open project", False, "folder")
        self.create_button.clicked.connect(lambda _checked=False: self._create())
        self.open_button.clicked.connect(lambda _checked=False: self._open())
        left.addWidget(self.hero)
        bottom = QHBoxLayout()
        self.classic_button = GhostButton("Use classic interface")
        self.classic_button.setToolTip(f"Open the classic {APP_NAME} window instead")
        self.classic_button.clicked.connect(lambda _checked=False: self._classic())
        bottom.addStretch(1)
        bottom.addWidget(self.classic_button)
        bottom.addStretch(1)
        left.addLayout(bottom)
        left.addStretch(1)
        columns.addLayout(left, 5)

        # -- right: what was open before and the 3.x import ------------------------
        self.right = QWidget()
        right = QVBoxLayout(self.right)
        right.setContentsMargins(0, Spacing.LG, 0, 0)
        right.setSpacing(Spacing.SM)
        self.recent_header = Label("Recent projects", "subtitle")
        right.addWidget(self.recent_header)
        self.recent_grid_holder = QWidget()
        self.recent_grid = QGridLayout(self.recent_grid_holder)
        self.recent_grid.setContentsMargins(0, 0, 0, 0)
        self.recent_grid.setSpacing(Spacing.SM)
        self.recent_grid.setColumnStretch(0, 1)
        right.addWidget(self.recent_grid_holder)

        self.import_card = Card(
            "Coming from version 3?",
            "Bring a saved template or batch plan along as a new study.",
        )
        self.sources_box = QVBoxLayout()
        self.sources_box.setSpacing(Spacing.XXS)
        self.import_card.content_layout.addLayout(self.sources_box)
        buttons = QHBoxLayout()
        buttons.setSpacing(Spacing.XS)
        self.import_button = SecondaryButton(IMPORT_BUTTON_TEXT, "upload")
        self.import_button.setToolTip(
            "Start a project from a saved template or batch plan"
        )
        self.import_button.clicked.connect(lambda _checked=False: self._import_chosen())
        self.dismiss_button = GhostButton("Hide")
        self.dismiss_button.setToolTip("Do not show this card again")
        self.dismiss_button.clicked.connect(lambda _checked=False: self.dismiss())
        buttons.addWidget(self.import_button)
        buttons.addStretch(1)
        buttons.addWidget(self.dismiss_button)
        self.import_card.content_layout.addLayout(buttons)
        right.addWidget(self.import_card)
        right.addStretch(1)
        columns.addWidget(self.right, 5)

    # ------------------------------------------------------------------
    # Content
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Rebuild the recent projects and the import card from the settings."""
        settings = self.window.app_settings
        self._fill_recent(settings.recent_projects())
        # The offer is for a first start: not for someone with a project to
        # reopen, and not once it was dealt with.
        first_start = (
            self.window.project is None and settings.last_project() is None
            and not settings.welcome_dismissed()
        )
        self.sources = recent_import_sources(settings) if first_start else []
        self._fill_sources()

    def _fill_recent(self, projects: List[Path]) -> None:
        for card in self.recent_cards:
            self.recent_grid.removeWidget(card)
            card.deleteLater()
        self.recent_cards = []
        for index, path in enumerate(projects[:MAX_RECENT_CARDS]):
            card = RecentProjectCard(path)
            card.clicked.connect(lambda _checked=False, p=path: self.window.open_path(p))
            self.recent_grid.addWidget(card, index // self.COLUMNS, index % self.COLUMNS)
            self.recent_cards.append(card)
        has = bool(self.recent_cards)
        self.recent_header.setVisible(has)
        self.recent_grid_holder.setVisible(has)
        self._sync_right()

    def _fill_sources(self) -> None:
        for button in self.source_buttons:
            self.sources_box.removeWidget(button)
            button.deleteLater()
        self.source_buttons = []
        for path, kind in self.sources:
            button = GhostButton(f"{path.name}  ({_KIND_LABELS[kind]})")
            button.setToolTip(str(path))
            button.setStyleSheet("text-align: left;")
            button.clicked.connect(
                lambda _checked=False, p=path, k=kind: self._import(p, k)
            )
            self.sources_box.addWidget(button)
            self.source_buttons.append(button)
        self.import_card.setVisible(bool(self.sources))
        self._sync_right()

    def _sync_right(self) -> None:
        """The right column is only there when it has something to show."""
        self.right.setVisible(
            bool(self.recent_cards) or bool(self.sources)
        )

    # ------------------------------------------------------------------
    # Dismissal
    # ------------------------------------------------------------------
    def dismiss(self) -> None:
        """Hide the import card for good."""
        self.window.app_settings.set_welcome_dismissed(True)
        self.window.app_settings.sync()
        self.sources = []
        self._fill_sources()

    def on_project_changed(self, project) -> None:
        # Opening anything means the user found their way: stop offering.
        if project is not None:
            if self.sources:
                self.dismiss()
        else:
            self.refresh()

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


__all__ = ["WelcomeView", "RecentProjectCard", "recent_import_sources"]
