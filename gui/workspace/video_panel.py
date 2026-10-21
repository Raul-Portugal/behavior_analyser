"""
gui/workspace/video_panel.py
The video table of the Videos stage: a drop zone while the project is empty,
otherwise a compact add bar (Add videos, Add folder, search, status filter,
Details) above a full-width table, with a summary line below.

Everything the user does here goes through the Qt-free ``Project``:

* adding, removing and moving videos use ``ProjectTableModel`` (proper row
  signals, so the selection survives),
* metadata columns and notes are edited in place,
* the context menu and the buttons that appear on the hovered row offer the
  per-video actions (Details, Reveal, Relink, Remove).

The table shows a status chip, the video name, its duration, the trial
window, where its zones come from and the metadata columns. The other model
columns (last run, notes, issues, settings source) are hidden; the header's
context menu shows them again.

The panel never asks the operating system or the user anything itself; it
uses the shared :class:`~gui.workspace.prompts.Prompts` object, so the same
panel works under test with scripted answers.

Public surface for the rest of the workspace:

* ``selection_changed(list)`` - the keys of the selected videos, in table
  order, every time the selection (or the rows under it) changes;
* ``details_requested(list)`` - the user asked for the details of videos
  (the window opens the drawer);
* ``selected_keys()``, ``select_keys(keys)``, ``clear_selection()``;
* the actions as methods: ``add_videos_dialog``, ``add_folder_dialog``,
  ``add_paths``, ``remove_selected``, ``move_selected``, ``mark_verified``,
  ``apply_template``, ``edit_metadata_column``, ``load_sample_sheet``,
  ``relink``, ``add_column``.
"""
import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PyQt6.QtCore import (QItemSelectionModel, QModelIndex, QPoint, QRectF,
                          QSize, Qt, QTimer, pyqtSignal)
from PyQt6.QtGui import QAction, QBrush, QColor, QKeyEvent, QPainter
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox,
                             QHBoxLayout, QHeaderView, QLineEdit, QMenu,
                             QStyle, QStyledItemDelegate, QStyleOptionViewItem,
                             QTableView, QVBoxLayout, QWidget)

from core.project import (SEVERITY_ERROR, SEVERITY_INFO, STATUS_DONE,
                          STATUS_FAILED, STATUS_INTERRUPTED, STATUS_WARNING,
                          STATUSES, TRIAL_INVALID, AddVideosResult, Project,
                          ProjectError)
from core.sample_sheet import SampleSheetError, describe_match
from gui.app_settings import AppSettings
from gui.project_table_model import (COL_CONFIGURED, COL_SETTINGS, COL_STATUS,
                                     COL_TRIAL, COL_VIDEO, COL_ZONES,
                                     LEADING_COLUMNS, KeyRole,
                                     ProjectFilterProxyModel,
                                     ProjectTableModel)
from gui.ux_helpers import format_duration
from gui.video_list import collect_video_files
from gui.workspace.prompts import SHEET_FILTER, VIDEO_FILTER, Prompts
from gui.workspace.theme import (Radius, Spacing, current_palette,
                                 style_table_view)
from gui.workspace.thumbnails import ThumbnailLoader
from gui.workspace.ui_kit import (DropZone, GhostButton, Label,
                                  SecondaryButton, _chip_size, _font,
                                  _paint_chip, _rounded)

logger = logging.getLogger(__name__)

MISSING_FILE = "missing_file"

FILTER_ALL = "All statuses"
_FILTER_LABELS = {
    "new": "New",
    "configured": "Configured",
    "done": "Done",
    "failed": "Failed",
    "warning": "Warning",
    "interrupted": "Interrupted",
}

ChipRole = Qt.ItemDataRole.UserRole + 10      # (chip kind, chip text)
HEADER_STATUS_SHOWN = "Status"
HEADER_DURATION = "Duration"
NO_DURATION = "—"
# Model columns the header menu cannot hide.
_ALWAYS_SHOWN = {COL_VIDEO}
_COLUMN_WIDTHS = {COL_CONFIGURED: 150, COL_SETTINGS: 92, COL_TRIAL: 160,
                  COL_ZONES: 104, COL_STATUS: 100}
METADATA_WIDTH = 124

ACTION_DETAILS = "details"
ACTION_REVEAL = "reveal"
ACTION_RELINK = "relink"
ACTION_REMOVE = "remove"


def format_video_duration(seconds: Optional[float]) -> str:
    """``m:ss`` for a duration in seconds; a dash when it is not known."""
    if seconds is None or seconds <= 0:
        return NO_DURATION
    return format_duration(seconds)


class WorkspaceProxyModel(ProjectFilterProxyModel):
    """The filter proxy, as the workspace table shows it.

    On top of sorting and filtering it

    * draws videos whose file is missing in red,
    * answers ``ChipRole`` for the status column (``(kind, text)``, from
      the missing file, the last run, the setup and the trial) and calls
      that column "Status",
    * shows the duration of the video in the table model's "Settings"
      column (which the workspace does not need: "Zones" already says where
      the settings come from), sorted by the real number of seconds,
    * colours trial windows that need attention.

    ``duration_of(key)`` supplies the duration in seconds (None = unknown).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.duration_of: Callable[[str], Optional[float]] = lambda key: None

    # -- helpers ---------------------------------------------------------
    def _key(self, index) -> Optional[str]:
        model = self.sourceModel()
        if not isinstance(model, ProjectTableModel) or not index.isValid():
            return None
        source = self.mapToSource(index)
        return model.index(source.row(), 0).data(KeyRole)

    def _is_missing(self, key: Optional[str]) -> bool:
        model = self.sourceModel()
        return key is not None and isinstance(model, ProjectTableModel) and any(
            issue.code == MISSING_FILE for issue in model.issues_for(key)
        )

    def chip_for(self, key: str) -> Tuple[str, str]:
        """``(kind, text)`` of a video's status chip."""
        project = self.sourceModel().project()
        entry = project.entry(key)
        if self._is_missing(key):
            return "error", "Missing file"
        if entry.last_status == STATUS_FAILED:
            return "error", "Failed"
        if entry.last_status == STATUS_INTERRUPTED:
            return "warning", "Interrupted"
        if entry.settings is None:
            return "neutral", "Not set up"
        trial = project.trial_status(entry)
        if trial == TRIAL_INVALID:
            return "error", "Invalid trial"
        if trial != "ok":
            return "warning", "Needs start"
        if entry.last_status == STATUS_DONE:
            return "ok", "Analysed"
        if entry.last_status == STATUS_WARNING:
            return "warning", "Check results"
        if entry.verified:
            return "ok", "Verified"
        return "info", "Set up"

    # -- model interface -------------------------------------------------
    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if (role == Qt.ItemDataRole.DisplayRole
                and orientation == Qt.Orientation.Horizontal):
            if section == COL_CONFIGURED:
                return HEADER_STATUS_SHOWN
            if section == COL_SETTINGS:
                return HEADER_DURATION
        return super().headerData(section, orientation, role)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid():
            column = index.column()
            key = self._key(index)
            if role == Qt.ItemDataRole.ForegroundRole:
                if self._is_missing(key):
                    return QBrush(QColor(current_palette().danger))
                if column == COL_TRIAL:
                    text = super().data(index, Qt.ItemDataRole.DisplayRole) or ""
                    if str(text).startswith("⚠"):
                        palette = current_palette()
                        invalid = "invalid" in str(text)
                        return QBrush(QColor(palette.danger if invalid else palette.warning))
            elif column == COL_SETTINGS and key is not None:
                if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
                    return format_video_duration(self.duration_of(key))
                if role == Qt.ItemDataRole.ToolTipRole:
                    return None
            elif column == COL_CONFIGURED and key is not None:
                if role == ChipRole:
                    return self.chip_for(key)
                if role == Qt.ItemDataRole.ToolTipRole:
                    model = self.sourceModel()
                    lines = [issue.message for issue in model.issues_for(key)]
                    return "\n".join(lines) or None
        return super().data(index, role)

    def lessThan(self, left, right) -> bool:
        if left.column() == COL_SETTINGS:
            model = self.sourceModel()
            keys = [model.index(i.row(), 0).data(KeyRole) for i in (left, right)]
            values = [self.duration_of(k) or 0.0 for k in keys]
            return values[0] < values[1]
        return super().lessThan(left, right)


class ChipDelegate(QStyledItemDelegate):
    """Draws the status column as a pill (``ChipRole``) on a plain row."""

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        chip = index.data(ChipRole)
        base = QStyleOptionViewItem(option)
        self.initStyleOption(base, index)
        base.text = ""
        widget = option.widget
        style = widget.style() if widget is not None else None
        if style is not None:
            style.drawControl(QStyle.ControlElement.CE_ItemViewItem, base, painter, widget)
        if not chip:
            return
        kind, text = chip
        font = _font(option.font, 11, True)
        size = _chip_size(text, font, kind in ("ok", "warning", "error", "info"))
        rect = QRectF(option.rect.left() + 8,
                      option.rect.center().y() - size.height() / 2.0 + 0.5,
                      min(size.width(), option.rect.width() - 12), size.height())
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        _paint_chip(painter, widget, rect, text, kind, font)
        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        hint = super().sizeHint(option, index)
        return QSize(hint.width(), max(hint.height(), 32))


class _RowActions(QWidget):
    """Small icon buttons floating over the hovered row of the table."""

    action = pyqtSignal(str)

    _SPECS = (
        (ACTION_DETAILS, "eye", "Show the details of this video"),
        (ACTION_REVEAL, "folder", "Show the file in its folder"),
        (ACTION_RELINK, "reset", "Point the project at the folder where this video is now"),
        (ACTION_REMOVE, "trash", "Remove this video from the project"),
    )

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.buttons: Dict[str, GhostButton] = {}
        row = QHBoxLayout(self)
        row.setContentsMargins(4, 2, 4, 2)
        row.setSpacing(2)
        for name, icon_name, tip in self._SPECS:
            button = GhostButton("", icon_name)
            button.setFixedSize(28, 28)
            button.setToolTip(tip)
            button.setAccessibleName(tip)
            button.clicked.connect(lambda _c=False, n=name: self.action.emit(n))
            row.addWidget(button)
            self.buttons[name] = button
        self.hide()

    def paintEvent(self, event) -> None:
        palette = current_palette()
        painter = QPainter(self)
        _rounded(painter, QRectF(self.rect()), Radius.MD, palette.surface,
                 palette.border_strong)


class VideoTableView(QTableView):
    """The table, with Delete and Alt+Up/Down turned into signals and the
    row action buttons that follow the mouse."""

    remove_requested = pyqtSignal()
    move_requested = pyqtSignal(int)    # -1 up, +1 down
    row_action = pyqtSignal(str, str)   # (action, key of the row)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.missing_of: Callable[[str], bool] = lambda key: False
        self.can_edit: Callable[[], bool] = lambda: True
        self._actions = _RowActions(self.viewport())
        self._actions.action.connect(self._on_action)
        self._hover_key: Optional[str] = None
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.verticalScrollBar().valueChanged.connect(self.hide_row_actions)

    @property
    def row_actions(self) -> _RowActions:
        return self._actions

    def keyPressEvent(self, event: QKeyEvent) -> None:
        alt = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
        if event.key() == Qt.Key.Key_Delete and not alt:
            self.remove_requested.emit()
            event.accept()
        elif alt and event.key() == Qt.Key.Key_Up:
            self.move_requested.emit(-1)
            event.accept()
        elif alt and event.key() == Qt.Key.Key_Down:
            self.move_requested.emit(1)
            event.accept()
        else:
            super().keyPressEvent(event)

    # -- hover actions -----------------------------------------------------
    def hide_row_actions(self, *_args) -> None:
        self._hover_key = None
        self._actions.hide()

    def show_row_actions(self, index: QModelIndex) -> None:
        """Float the action buttons over the row of ``index``."""
        key = index.data(KeyRole) if index.isValid() else None
        if key is None:
            self.hide_row_actions()
            return
        self._hover_key = key
        editable = self.can_edit()
        self._actions.buttons[ACTION_RELINK].setVisible(self.missing_of(key))
        self._actions.buttons[ACTION_RELINK].setEnabled(editable)
        self._actions.buttons[ACTION_REMOVE].setEnabled(editable)
        self._actions.adjustSize()
        rect = self.visualRect(self.model().index(index.row(), COL_VIDEO))
        height = self._actions.height()
        x = max(0, self.viewport().width() - self._actions.width() - 8)
        y = rect.top() + (rect.height() - height) // 2
        self._actions.move(x, y)
        self._actions.show()
        self._actions.raise_()

    def mouseMoveEvent(self, event) -> None:
        super().mouseMoveEvent(event)
        index = self.indexAt(event.position().toPoint())
        if index.isValid():
            if index.data(KeyRole) != self._hover_key or not self._actions.isVisible():
                self.show_row_actions(index)
        elif not self._actions.underMouse():
            self.hide_row_actions()

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        if not self._actions.underMouse():
            self.hide_row_actions()

    def _on_action(self, name: str) -> None:
        key = self._hover_key
        self.hide_row_actions()
        if key is not None:
            self.row_action.emit(name, key)


class VideoPanel(QWidget):
    """Add bar, filters, table, footer and actions for the project's videos."""

    selection_changed = pyqtSignal(list)    # keys of the selected videos
    # The user asked for the details of these videos (a row's button, a
    # double click, the add bar's Details button, the context menu).
    details_requested = pyqtSignal(list)
    # Local paths dropped on the panel (files, folders, anything); whoever
    # owns the workspace decides what each one means.
    paths_dropped = pyqtSignal(list)
    # A short sentence about what the panel just did, for a status bar.
    status_message = pyqtSignal(str)

    def __init__(
        self,
        app_settings: Optional[AppSettings] = None,
        prompts: Optional[Prompts] = None,
        parent: Optional[QWidget] = None,
        loader: Optional[ThumbnailLoader] = None,
    ):
        super().__init__(parent)
        self.app_settings = app_settings or AppSettings()
        self.prompts = prompts or Prompts(self)
        self.loader = loader        # reads the durations (shared, not owned)
        self._project: Optional[Project] = None
        self._read_only = False
        self._last_keys: List[str] = []
        self._user_shown: set = set()
        self.model = ProjectTableModel(parent=self)
        self.proxy = WorkspaceProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.duration_of = self.duration_of
        self._build_ui()
        self.setAcceptDrops(True)
        # Coalesces the burst of model signals one edit produces.
        self._selection_timer = QTimer(self)
        self._selection_timer.setSingleShot(True)
        self._selection_timer.setInterval(0)
        self._selection_timer.timeout.connect(self._emit_selection)
        self.model.projectChanged.connect(self._on_model_changed)
        self.model.modelReset.connect(self._on_model_reset)
        self.model.rowsRemoved.connect(self._schedule_selection)
        self.model.rowsInserted.connect(self._on_rows_inserted)
        self.model.columnsInserted.connect(self._apply_view_layout)
        self.model.columnsRemoved.connect(self._apply_view_layout)
        self.view.selectionModel().selectionChanged.connect(
            self._schedule_selection
        )
        if loader is not None:
            loader.loaded.connect(self._on_preview_loaded)
        self.set_project(None)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.XS)

        # -- the add bar ----------------------------------------------------
        self.add_bar = QWidget()
        bar = QHBoxLayout(self.add_bar)
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(Spacing.XS)
        self.add_videos_button = SecondaryButton("Add videos", "video")
        self.add_videos_button.setToolTip("Add video files to the project")
        self.add_videos_button.clicked.connect(
            lambda _checked=False: self.add_videos_dialog()
        )
        self.add_folder_button = SecondaryButton("Add folder", "folder")
        self.add_folder_button.setToolTip("Add all the videos of a folder")
        self.add_folder_button.clicked.connect(
            lambda _checked=False: self.add_folder_dialog()
        )
        self.drop_hint = Label("or drop videos and folders anywhere", "muted")
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search videos, metadata, notes...")
        self.search_box.setClearButtonEnabled(True)
        self.search_box.setMinimumWidth(220)
        self.search_box.textChanged.connect(self._on_filter_changed)
        self.status_combo = QComboBox()
        self.status_combo.addItem(FILTER_ALL, "")
        for status in STATUSES:
            self.status_combo.addItem(_FILTER_LABELS.get(status, status), status)
        self.status_combo.currentIndexChanged.connect(self._on_filter_changed)
        self.details_button = GhostButton("Details", "eye")
        self.details_button.setToolTip(
            "Show the details of the selected video (Ctrl+I)"
        )
        self.details_button.clicked.connect(
            lambda _checked=False: self.request_details()
        )
        bar.addWidget(self.add_videos_button)
        bar.addWidget(self.add_folder_button)
        bar.addSpacing(Spacing.XS)
        bar.addWidget(self.drop_hint)
        bar.addStretch(1)
        bar.addWidget(self.search_box)
        bar.addWidget(self.status_combo)
        bar.addWidget(self.details_button)
        layout.addWidget(self.add_bar)

        # -- the empty state -------------------------------------------------
        self.drop_zone = DropZone()
        self.drop_zone.setAcceptDrops(False)    # the panel takes the drops
        self.drop_zone.browse_requested.connect(
            lambda: self.add_videos_dialog()
        )
        self.drop_zone.folder_requested.connect(
            lambda: self.add_folder_dialog()
        )
        layout.addWidget(self.drop_zone, 1)

        # -- the table --------------------------------------------------------
        self.view = VideoTableView()
        self.view.setModel(self.proxy)
        style_table_view(self.view)
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.view.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.view.setSortingEnabled(True)
        self.view.sortByColumn(-1, Qt.SortOrder.AscendingOrder)
        header = self.view.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setMinimumSectionSize(40)
        header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header.customContextMenuRequested.connect(self._show_header_menu)
        self.view.setItemDelegateForColumn(COL_CONFIGURED, ChipDelegate(self.view))
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._show_context_menu)
        self.view.remove_requested.connect(self.remove_selected)
        self.view.move_requested.connect(self.move_selected)
        self.view.row_action.connect(self._on_row_action)
        self.view.doubleClicked.connect(self._on_double_clicked)
        self.view.missing_of = lambda key: self.has_missing([key])
        self.view.can_edit = lambda: not self._blocked()
        layout.addWidget(self.view, 1)

        # -- the bottom row ----------------------------------------------------
        self.bottom_bar = QWidget()
        buttons = QHBoxLayout(self.bottom_bar)
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(Spacing.SM)
        self.add_column_button = GhostButton("Add column", "plus")
        self.add_column_button.setToolTip(
            "Add a metadata column (group, dose, session...) to every video"
        )
        self.add_column_button.clicked.connect(self.add_column)
        buttons.addWidget(self.add_column_button)
        self.recursive_check = QCheckBox("Include subfolders")
        self.recursive_check.setToolTip(
            "When adding or dropping a folder, also take the videos in the "
            "folders inside it"
        )
        self.recursive_check.setChecked(self.app_settings.add_folder_recursive())
        self.recursive_check.toggled.connect(self._on_recursive_toggled)
        self.drop_zone.set_recursive(self.recursive_check.isChecked())
        buttons.addWidget(self.recursive_check)
        buttons.addStretch(1)
        self.footer = Label("", "muted")
        self.footer.setTextFormat(Qt.TextFormat.RichText)
        self.footer.setWordWrap(True)
        buttons.addWidget(self.footer, 4)
        layout.addWidget(self.bottom_bar)

    @property
    def empty_state_visible(self) -> bool:
        """True while the drop zone stands in for the (empty) table."""
        return not self.drop_zone.isHidden()

    # -- table layout --------------------------------------------------------
    def _apply_view_layout(self, *_args) -> None:
        """Hide the secondary columns, order and size the rest."""
        view = self.view
        header = view.horizontalHeader()
        columns = self.model.columnCount()
        shown = {COL_CONFIGURED, COL_VIDEO, COL_SETTINGS, COL_TRIAL, COL_ZONES}
        shown.update(range(
            LEADING_COLUMNS, LEADING_COLUMNS + len(self.model.metadata_columns())
        ))
        for column in range(columns):
            visible = column in shown or column in self._user_shown
            view.setColumnHidden(column, not visible)
        # The status chip leads, then the name.
        if columns > max(COL_CONFIGURED, COL_VIDEO):
            header.moveSection(header.visualIndex(COL_CONFIGURED), 0)
            header.moveSection(header.visualIndex(COL_VIDEO), 1)
        for column in range(columns):
            if column == COL_VIDEO:
                header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
            else:
                header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
                header.resizeSection(column, _COLUMN_WIDTHS.get(column, METADATA_WIDTH))

    def _show_header_menu(self, position: QPoint) -> None:
        menu = self.build_header_menu()
        if menu.actions():
            menu.exec(self.view.horizontalHeader().mapToGlobal(position))
        menu.deleteLater()

    def build_header_menu(self) -> QMenu:
        """Checkable list of the columns (the name always stays)."""
        menu = QMenu(self)
        for column in range(self.model.columnCount()):
            if column in _ALWAYS_SHOWN:
                continue
            label = self.proxy.headerData(column, Qt.Orientation.Horizontal)
            action = QAction(str(label), menu)
            action.setCheckable(True)
            action.setChecked(not self.view.isColumnHidden(column))
            action.toggled.connect(
                lambda checked, c=column: self.set_column_visible(c, checked)
            )
            menu.addAction(action)
        return menu

    def set_column_visible(self, column: int, visible: bool) -> None:
        if visible:
            self._user_shown.add(column)
        else:
            self._user_shown.discard(column)
        self.view.setColumnHidden(column, not visible)
        if visible:
            self.view.horizontalHeader().resizeSection(
                column, _COLUMN_WIDTHS.get(column, METADATA_WIDTH)
            )

    # ------------------------------------------------------------------
    # Durations
    # ------------------------------------------------------------------
    def duration_of(self, key: str) -> Optional[float]:
        """Seconds: what the project knows, else what the loader has read."""
        project = self._project
        if project is None:
            return None
        try:
            entry = project.entry(key)
        except KeyError:
            return None
        if entry.duration_s:
            return entry.duration_s
        if self.loader is not None:
            preview = self.loader.get(entry.path)
            if preview is not None and preview.duration > 0:
                return preview.duration
        return None

    def request_durations(self) -> int:
        """Ask the loader to read every video not read yet; returns how many."""
        if self.loader is None or self._project is None:
            return 0
        started = 0
        for entry in self._project.videos:
            if not entry.duration_s and self.loader.get(entry.path) is None:
                started += int(self.loader.request(entry.path))
        return started

    def _on_preview_loaded(self, path_text: str) -> None:
        project = self._project
        if project is None:
            return
        for row, entry in enumerate(project.videos):
            if str(entry.path) == path_text and row < self.model.rowCount():
                index = self.model.index(row, COL_SETTINGS)
                self.model.dataChanged.emit(index, index)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.request_durations()

    def _on_rows_inserted(self, *_args) -> None:
        self._schedule_selection()
        if self.isVisible():
            self.request_durations()

    # ------------------------------------------------------------------
    # The project
    # ------------------------------------------------------------------
    def project(self) -> Optional[Project]:
        return self._project

    def set_read_only(self, read_only: bool) -> None:
        """Forbid every change to the project (while a run is in progress)."""
        self._read_only = bool(read_only)
        self.view.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers if read_only
            else QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.add_column_button.setEnabled(
            self._project is not None and not read_only
        )
        self._refresh_chrome()

    def is_read_only(self) -> bool:
        return self._read_only

    def _blocked(self) -> bool:
        """True when edits are not allowed (no project, or read-only)."""
        return self._project is None or self._read_only

    def set_project(self, project: Optional[Project]) -> None:
        self._project = project
        self.model.recursive_drop = self.recursive_check.isChecked()
        self.view.hide_row_actions()
        self._user_shown = set()
        self.model.set_project(project)
        enabled = project is not None
        self.search_box.setEnabled(enabled)
        self.status_combo.setEnabled(enabled)
        self.add_column_button.setEnabled(enabled and not self._read_only)
        self.view.setEnabled(enabled)
        self.proxy.clear_filters()
        self.search_box.clear()
        self.status_combo.setCurrentIndex(0)
        self._apply_view_layout()
        self._refresh_chrome()

    def _on_model_changed(self) -> None:
        self._refresh_chrome()

    def _on_model_reset(self) -> None:
        self.view.hide_row_actions()
        self._apply_view_layout()
        self._refresh_chrome()
        self._schedule_selection()
        if self.isVisible():
            self.request_durations()

    def _refresh_chrome(self) -> None:
        """Add bar, empty state and footer follow the project."""
        project = self._project
        empty = project is not None and not project.videos
        editable = not self._blocked()
        self.drop_zone.setVisible(empty)
        self.view.setVisible(not empty)
        self.add_bar.setVisible(not empty)
        self.add_videos_button.setEnabled(editable)
        self.add_folder_button.setEnabled(editable)
        self.drop_zone.setEnabled(editable)
        self.details_button.setEnabled(bool(self.selected_keys()))
        self.footer.setText(self.footer_text())

    def footer_text(self) -> str:
        """The summary line: videos, configured, verified and issues."""
        project = self._project
        if project is None:
            return "No project open"
        total = len(project.videos)
        configured = sum(1 for v in project.videos if v.settings is not None)
        verified = sum(1 for v in project.videos if v.verified)
        errors = warnings = 0
        counted = [
            issue
            for video in project.videos
            for issue in self.model.issues_for(video.key)
        ] + self.model.project_issues()
        for issue in counted:
            if issue.severity == SEVERITY_ERROR:
                errors += 1
            elif issue.severity != SEVERITY_INFO:
                warnings += 1
        shown = self.proxy.rowCount()
        count_text = f"{total} video{'s' if total != 1 else ''}"
        if shown != total:
            count_text += f" ({shown} shown)"
        if errors:
            issue_text = (
                f'<span style="color:{QColor(current_palette().danger).name()}">'
                f"{errors + warnings} issue{'s' if errors + warnings != 1 else ''}"
                "</span>"
            )
        else:
            issue_text = f"{warnings} issue{'s' if warnings != 1 else ''}"
        return (
            f"{count_text} · {configured} configured · "
            f"{verified} verified · {issue_text}"
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def selected_keys(self) -> List[str]:
        """Keys of the selected videos in the order shown."""
        rows = sorted(
            {index.row() for index in self.view.selectionModel().selectedRows()}
        )
        return self.proxy.keys_for_rows(rows)

    def select_keys(self, keys: Iterable[str]) -> None:
        wanted = set(keys)
        selection = self.view.selectionModel()
        selection.clearSelection()
        first: Optional[QModelIndex] = None
        for row in range(self.proxy.rowCount()):
            index = self.proxy.index(row, 0)
            if index.data(KeyRole) in wanted:
                selection.select(
                    index,
                    QItemSelectionModel.SelectionFlag.Select
                    | QItemSelectionModel.SelectionFlag.Rows,
                )
                first = first or index
        if first is not None:
            # Current item only: setCurrentIndex would collapse the selection.
            selection.setCurrentIndex(
                first, QItemSelectionModel.SelectionFlag.NoUpdate
            )

    def clear_selection(self) -> None:
        self.view.selectionModel().clearSelection()

    def _schedule_selection(self, *_args) -> None:
        self._selection_timer.start()

    def _emit_selection(self) -> None:
        keys = self.selected_keys()
        if keys != self._last_keys:
            self._last_keys = keys
            self.details_button.setEnabled(bool(keys))
            self.selection_changed.emit(keys)

    def flush_selection(self) -> None:
        """Deliver a pending ``selection_changed`` now (used by tests)."""
        self._selection_timer.stop()
        self._emit_selection()

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    def _on_filter_changed(self, *_args) -> None:
        self.proxy.set_text_filter(self.search_box.text())
        status = self.status_combo.currentData()
        self.proxy.set_status_filter([status] if status else [])
        self._refresh_chrome()
        self._schedule_selection()

    def _on_recursive_toggled(self, checked: bool) -> None:
        self.app_settings.set_add_folder_recursive(checked)
        self.model.recursive_drop = checked
        self.drop_zone.set_recursive(checked)

    # ------------------------------------------------------------------
    # Adding
    # ------------------------------------------------------------------
    def add_videos_dialog(self) -> Optional[AddVideosResult]:
        if self._blocked():
            return None
        files = self.prompts.open_files(
            "Select Video Files", self.app_settings.dialog_dir("video"),
            VIDEO_FILTER,
        )
        if not files:
            return None
        self.app_settings.set_last_video_dir(files[0].parent)
        return self.add_paths(files)

    def add_folder_dialog(self) -> Optional[AddVideosResult]:
        if self._blocked():
            return None
        folder = self.prompts.choose_folder(
            "Select Folder Containing Videos",
            self.app_settings.dialog_dir("video"),
        )
        if folder is None:
            return None
        self.app_settings.set_last_video_dir(folder)
        return self.add_paths([folder])

    def add_paths(self, paths: Sequence[Path]) -> Optional[AddVideosResult]:
        """Add videos, and the videos inside folders, to the project.

        Reports what was skipped. Returns None when there is no project or
        nothing usable was given.
        """
        if self._blocked():
            return None
        videos, ignored = collect_video_files(
            [Path(p) for p in paths], self.recursive_check.isChecked()
        )
        if not videos:
            hint = ""
            if any(Path(p).is_dir() for p in paths):
                hint = (
                    "\n\nTick 'Include subfolders' to search the folders "
                    "inside it too."
                ) if not self.recursive_check.isChecked() else ""
            names = ", ".join(Path(p).name for p in paths[:5])
            self.prompts.info(
                "No Videos Found",
                f"No video files (.mp4, .avi, .mov, .mkv) in: {names}{hint}",
            )
            return None
        self.app_settings.set_last_video_dir(videos[0].parent)
        result = self.model.add_videos(videos)
        parts = [f"Added {len(result.added)} video(s)"]
        if result.duplicates:
            parts.append(f"{len(result.duplicates)} already in the project")
        if ignored:
            parts.append(f"{len(ignored)} ignored")
        self.status_message.emit("; ".join(parts))
        if result.added:
            self.select_keys([entry.key for entry in result.added])
        return result

    # ------------------------------------------------------------------
    # Removing and moving
    # ------------------------------------------------------------------
    def remove_selected(self, confirm: bool = True) -> List[str]:
        """Remove the selected videos (after asking). Returns the keys removed."""
        keys = self.selected_keys()
        return self.remove_keys(keys, confirm)

    def remove_keys(self, keys: Sequence[str], confirm: bool = True) -> List[str]:
        if self._blocked() or not keys:
            return []
        if confirm:
            count = len(keys)
            what = (
                f"“{self._project.entry(keys[0]).name}”"
                if count == 1 else f"{count} videos"
            )
            if not self.prompts.confirm(
                "Remove videos",
                f"Remove {what} from the project?",
                accept="Remove",
                informative="The video files stay on disk. Their settings "
                            "in this project are forgotten.",
                destructive=True,
            ):
                return []
        row = min(
            (r.row() for r in self.view.selectionModel().selectedRows()),
            default=0,
        )
        removed = self.model.remove_keys(keys)
        self.status_message.emit(f"Removed {len(removed)} video(s)")
        # Keep a selection near where the user was working.
        if removed and self.proxy.rowCount():
            index = self.proxy.index(min(row, self.proxy.rowCount() - 1), 0)
            self.view.selectionModel().select(
                index,
                QItemSelectionModel.SelectionFlag.ClearAndSelect
                | QItemSelectionModel.SelectionFlag.Rows,
            )
            self.view.setCurrentIndex(index)
        return [entry.key for entry in removed]

    def move_selected(self, offset: int) -> bool:
        """Move the selection up (negative) or down in the project order."""
        keys = self.selected_keys()
        if self._blocked() or not keys:
            return False
        moved = self.model.move_keys(keys, offset)
        self.view.setFocus()
        return moved

    # ------------------------------------------------------------------
    # Per-video actions
    # ------------------------------------------------------------------
    def mark_verified(self, keys: Sequence[str], verified: bool = True) -> int:
        """Mark the *configured* videos among ``keys``; returns how many."""
        project = self._project
        if self._blocked():
            return 0
        entries = [e for e in project.entries_for(keys) if e.settings is not None]
        if not entries:
            self.prompts.info(
                "Nothing to mark",
                "Only videos that have settings can be marked as verified.",
            )
            return 0
        project.set_verified([e.key for e in entries], verified)
        return len(entries)

    def apply_template(self, keys: Sequence[str]) -> List[str]:
        """Reset the videos to the project template (asks before overwriting)."""
        project = self._project
        if self._blocked() or not keys:
            return []
        if project.template_settings is None:
            self.prompts.info(
                "No template",
                "The project has no template settings yet. Configure them "
                "in the Arena & zones stage first.",
            )
            return []
        custom = [
            e for e in project.entries_for(keys)
            if e.settings is not None
            and (e.settings_source != "template" or e.zones is not None)
        ]
        if custom and not self.prompts.confirm(
            "Apply template settings",
            f"{len(custom)} of the selected videos have their own settings "
            "(customised or aligned).",
            accept="Overwrite",
            informative="Their settings will be replaced by the project "
                        "template and their verification is cleared.",
            destructive=True,
        ):
            return []
        try:
            return project.apply_template_to(keys)
        except ProjectError as exc:
            self.prompts.warn("Apply template settings", str(exc))
            return []

    def reveal(self, key: str) -> bool:
        if self._project is None:
            return False
        return self.prompts.reveal(self._project.entry(key).path)

    def copy_paths(self, keys: Sequence[str]) -> None:
        if self._project is None:
            return
        entries = self._project.entries_for(keys)
        self.prompts.copy_text("\n".join(str(e.path) for e in entries))

    # -- metadata --------------------------------------------------------
    def add_column(self) -> Optional[str]:
        """Ask for a name and add a metadata column; returns the name."""
        if self._blocked():
            return None
        name = self.prompts.ask_text(
            "Add column", "Name of the new metadata column:"
        )
        if name is None:
            return None
        name = name.strip()
        if not name:
            return None
        try:
            added = self.model.add_metadata_column(name)
        except ProjectError as exc:
            self.prompts.warn("Add column", str(exc))
            return None
        if not added:
            self.prompts.info(
                "Add column", f"There is already a column named “{name}”."
            )
            return None
        return name

    def edit_metadata_column(
        self, keys: Sequence[str], column: Optional[str] = None
    ) -> int:
        """Set one metadata value on several videos at once.

        ``column`` is chosen from the project's columns when not given.
        Returns the number of videos changed.
        """
        project = self._project
        if self._blocked() or not keys:
            return 0
        if column is None:
            columns = project.metadata_columns()
            if not columns:
                column = self.add_column()
                if column is None:
                    return 0
            else:
                column = self.prompts.ask_item(
                    "Edit metadata column", "Column to set:", columns
                )
                if column is None:
                    return 0
        entries = project.entries_for(keys)
        values = {e.metadata.get(column, "") for e in entries}
        value = self.prompts.ask_text(
            "Edit metadata column",
            f"Value of “{column}” for {len(entries)} video(s):",
            next(iter(values)) if len(values) == 1 else "",
        )
        if value is None:
            return 0
        changed = 0
        for entry in entries:
            if entry.metadata.get(column) != value:
                project.set_metadata_value(entry.key, column, value)
                changed += 1
        return changed

    def remove_column(self, column: str) -> bool:
        if self._blocked():
            return False
        if not self.prompts.confirm(
            "Remove column",
            f"Remove the metadata column “{column}” from every video?",
            accept="Remove", destructive=True,
        ):
            return False
        return self.model.remove_metadata_column(column)

    def load_sample_sheet(self, path: Optional[Path] = None) -> Optional[object]:
        """Fill the metadata from a sample sheet CSV and report the match."""
        project = self._project
        if self._blocked():
            return None
        if path is None:
            start = project.run_options.sample_sheet
            path = self.prompts.open_file(
                "Load sample sheet",
                str(start.parent) if start is not None
                else self.app_settings.dialog_dir("config"),
                SHEET_FILTER,
            )
            if path is None:
                return None
        try:
            match = project.set_metadata_from_sheet(path)
        except (SampleSheetError, OSError, ValueError) as exc:
            self.prompts.error("Sample sheet", f"Cannot use {Path(path).name}:\n{exc}")
            return None
        self.prompts.info(
            "Sample sheet", describe_match(match, len(project.videos))
        )
        return match

    # -- missing files ---------------------------------------------------
    def relink(self, key: str) -> List[str]:
        """Point a moved data folder at its new place (``Project.relink``).

        The user picks the folder where the missing video is now; every
        file that lived under the old folder follows it.
        """
        project = self._project
        if self._blocked():
            return []
        entry = project.entry(key)
        folder = self.prompts.choose_folder(
            f"Where is {entry.name} now?", str(entry.path.parent.parent)
            if entry.path.parent.parent.is_dir() else
            self.app_settings.dialog_dir("video"),
        )
        if folder is None:
            return []
        if not (folder / entry.path.name).is_file():
            self.prompts.warn(
                "Relink",
                f"{entry.path.name} is not in:\n{folder}\n\nChoose the folder "
                "that now holds the video.",
            )
            return []
        relinked = project.relink(entry.path.parent, folder)
        self.status_message.emit(f"Relinked {len(relinked)} video(s)")
        self.model.refresh_issues()
        return relinked

    def has_missing(self, keys: Sequence[str]) -> bool:
        return any(
            issue.code == MISSING_FILE
            for key in keys for issue in self.model.issues_for(key)
        )

    # ------------------------------------------------------------------
    # Details and the hovered row's buttons
    # ------------------------------------------------------------------
    def request_details(self, keys: Optional[Sequence[str]] = None) -> None:
        """Ask the window to show the details of ``keys`` (default: selection)."""
        wanted = list(keys) if keys is not None else self.selected_keys()
        if wanted:
            self.details_requested.emit(wanted)

    def _on_row_action(self, action: str, key: str) -> None:
        if action == ACTION_DETAILS:
            if self.selected_keys() != [key]:
                self.select_keys([key])
                self.flush_selection()
            self.request_details()
        elif action == ACTION_REVEAL:
            self.reveal(key)
        elif action == ACTION_RELINK:
            self.relink(key)
        elif action == ACTION_REMOVE:
            self.remove_keys([key])

    def _on_double_clicked(self, index: QModelIndex) -> None:
        """A double click on a cell that cannot be edited opens the details."""
        if not index.isValid():
            return
        if self.proxy.flags(index) & Qt.ItemFlag.ItemIsEditable:
            return
        key = index.data(KeyRole)
        if key is not None:
            self.request_details([key])

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------
    def _show_context_menu(self, position: QPoint) -> None:
        index = self.view.indexAt(position)
        if index.isValid() and index.row() not in {
            i.row() for i in self.view.selectionModel().selectedRows()
        }:
            self.select_keys([index.data(KeyRole)])
        menu = self.build_context_menu(index)
        if menu.actions():
            menu.exec(self.view.viewport().mapToGlobal(position))
        menu.deleteLater()

    def build_context_menu(self, index: Optional[QModelIndex] = None) -> QMenu:
        """The menu for the current selection; ``index`` is the clicked cell."""
        menu = QMenu(self)
        keys = self.selected_keys()
        project = self._project
        if project is None:
            return menu
        has_selection = bool(keys)
        column_name = None
        if index is not None and index.isValid():
            source = self.proxy.mapToSource(index)
            header = self.model.headers()[source.column()]
            if source.column() in self.metadata_column_range():
                column_name = header

        def add(text: str, slot, enabled: bool = True) -> QAction:
            action = QAction(text, menu)
            action.setEnabled(enabled)
            action.triggered.connect(lambda _checked=False: slot())
            menu.addAction(action)
            return action

        add("Details", lambda: self.request_details(keys), has_selection)
        add("Remove", lambda: self.remove_keys(self.selected_keys()), has_selection)
        add("Reveal in folder", lambda: self.reveal(keys[0]), len(keys) == 1)
        add("Copy path", lambda: self.copy_paths(keys), has_selection)
        if has_selection and self.has_missing(keys):
            add("Relink...", lambda: self.relink(
                next(k for k in keys if self.has_missing([k]))
            ))
        menu.addSeparator()
        add("Mark verified", lambda: self.mark_verified(keys, True), has_selection)
        add("Mark unverified", lambda: self.mark_verified(keys, False), has_selection)
        add(
            "Apply template settings",
            lambda: self.apply_template(keys),
            has_selection and project.template_settings is not None,
        )
        menu.addSeparator()
        add(
            "Edit metadata column...",
            lambda: self.edit_metadata_column(keys, column_name),
            has_selection,
        )
        if column_name is not None:
            add(
                f"Remove column “{column_name}”...",
                lambda: self.remove_column(column_name),
            )
        add("Load sample sheet...", self.load_sample_sheet)
        return menu

    def metadata_column_range(self) -> range:
        return range(
            LEADING_COLUMNS, LEADING_COLUMNS + len(self.model.metadata_columns())
        )

    # ------------------------------------------------------------------
    # Drag and drop: the panel only reports what was dropped
    # ------------------------------------------------------------------
    def dragEnterEvent(self, event) -> None:
        if not self._blocked() and event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_zone.set_highlight(True)
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if not self._blocked() and event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self.drop_zone.set_highlight(False)
        event.accept()

    def dropEvent(self, event) -> None:
        self.drop_zone.set_highlight(False)
        paths = [
            Path(url.toLocalFile()) for url in event.mimeData().urls()
            if url.isLocalFile() and url.toLocalFile()
        ]
        if self._blocked() or not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        self.paths_dropped.emit(paths)

    def shutdown(self) -> None:
        """Stop what the panel started (the selection timer)."""
        self._selection_timer.stop()
        self.view.hide_row_actions()


__all__ = ["VideoPanel", "VideoTableView", "WorkspaceProxyModel", "ChipDelegate",
           "ChipRole"]
