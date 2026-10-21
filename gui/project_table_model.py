"""
gui/project_table_model.py
Qt table adapter over ``core.project.Project.videos``: the video table of the
workspace.

The model is a thin layer. All logic (adding, removing, moving, validating)
stays in the Qt-free ``Project``; the model only turns its data into table
cells, forwards the few edits a table allows (metadata and notes) and tells
views about row/column changes with proper signals, so selections survive.

Columns, left to right: Status, Video, Configured, Settings, Trial, Zones,
one column per metadata key of the project (dynamic), Last run, Notes,
Issues. "Configured" is the composed state of the video: its settings, its
trial (a missing start or invalid times show up there) and its verification.
"Trial" is when the analysed part starts and ends ("0:12 - 5:12", "whole
video" or "! set start"); "Zones" says where its regions come from (Template,
Adjusted, Aligned or Custom).

Use ``ProjectFilterProxyModel`` on top for sorting and for filtering by text
or status. Both models answer ``KeyRole`` with the video's stable project
key, so selections are read the same way through the proxy:
``selected_keys(view.selectionModel().selectedRows())``.
"""
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PyQt6.QtCore import (QAbstractItemModel, QAbstractTableModel,
                          QMimeData, QModelIndex, QSortFilterProxyModel, Qt,
                          pyqtSignal)
from PyQt6.QtGui import QIcon

from core.project import (STATUS_CONFIGURED, STATUS_DONE, STATUS_FAILED,
                          STATUS_INTERRUPTED, STATUS_NEW, STATUS_WARNING,
                          STATUSES, SEVERITY_ERROR, TRIAL_INVALID,
                          TRIAL_NEEDS_START, TRIAL_OK, AddVideosResult,
                          Project, ProjectIssue, VideoEntry)
from gui.video_list import collect_video_files, status_icon

KeyRole = Qt.ItemDataRole.UserRole           # the video's project key
SortRole = Qt.ItemDataRole.UserRole + 1      # value used for sorting
PathRole = Qt.ItemDataRole.UserRole + 2      # the video's absolute Path

COL_STATUS = 0
COL_VIDEO = 1
COL_CONFIGURED = 2
COL_SETTINGS = 3
COL_TRIAL = 4
COL_ZONES = 5
LEADING_COLUMNS = 6                          # metadata columns start here
HEADER_STATUS = "Status"
HEADER_VIDEO = "Video"
HEADER_CONFIGURED = "Configured"
HEADER_SETTINGS = "Settings"
HEADER_TRIAL = "Trial"
HEADER_ZONES = "Zones"
HEADER_LAST_RUN = "Last run"
HEADER_NOTES = "Notes"
HEADER_ISSUES = "Issues"

_STATUS_TEXT = {
    STATUS_NEW: "New",
    STATUS_CONFIGURED: "Configured",
    STATUS_DONE: "Done",
    STATUS_FAILED: "Failed",
    STATUS_WARNING: "Warning",
    STATUS_INTERRUPTED: "Interrupted",
}
# Statuses the shared badge icons know (video_list.status_icon).
_ICON_STATUS = {
    STATUS_NEW: "queued",
    STATUS_CONFIGURED: "queued",
    STATUS_DONE: "done",
    STATUS_FAILED: "failed",
    STATUS_WARNING: "warning",
    STATUS_INTERRUPTED: "warning",
}


TEXT_WHOLE_VIDEO = "whole video"
TEXT_NEEDS_START = "⚠ set start"
TEXT_INVALID_TRIAL = "⚠ invalid times"
_DASH = "–"


def format_clock(seconds) -> str:
    """``m:ss`` (``h:mm:ss`` from an hour), with decimals only if present."""
    try:
        total = round(abs(float(seconds)), 1)
    except (TypeError, ValueError, OverflowError):
        return str(seconds)
    sign = "-" if float(seconds) < 0 else ""
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    tail = f"{int(secs):02d}" if secs == int(secs) else f"{secs:04.1f}"
    if hours:
        return f"{sign}{int(hours)}:{int(minutes):02d}:{tail}"
    return f"{sign}{int(minutes)}:{tail}"


def format_trial(start, end) -> str:
    """``0:12 - 5:12``; ``whole video``; ``0:12 - end`` for an open end."""
    if end is None:
        if not start:
            return TEXT_WHOLE_VIDEO
        return f"{format_clock(start)} {_DASH} end"
    return f"{format_clock(start)} {_DASH} {format_clock(end)}"


def selected_keys(indexes: Iterable[QModelIndex]) -> List[str]:
    """Unique video keys of the rows of ``indexes``, in the given order.

    Works with indexes of the table model and of a proxy on top of it.
    """
    keys: List[str] = []
    seen = set()
    for index in indexes:
        key = index.data(KeyRole)
        if key is not None and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def paths_from_mime(
    mime: QMimeData, recursive: bool = False
) -> Tuple[List[Path], List[Path]]:
    """``(videos, ignored)`` for the local files and folders of a drop.

    Folders are expanded to the videos they hold (also in sub-folders when
    ``recursive``); anything that is not a video, or not a local file, is
    ignored.
    """
    if not mime.hasUrls():
        return [], []
    paths = [
        Path(url.toLocalFile()) for url in mime.urls()
        if url.isLocalFile() and url.toLocalFile()
    ]
    return collect_video_files(paths, recursive)


class ProjectTableModel(QAbstractTableModel):
    """The videos of a project as an editable table."""

    # A change was made to the project through this model or from outside.
    projectChanged = pyqtSignal()
    # Videos and ignored paths of a drop, for the shell to add (or report).
    pathsDropped = pyqtSignal(list)
    pathsIgnored = pyqtSignal(list)

    def __init__(self, project: Optional[Project] = None, parent=None):
        super().__init__(parent)
        self._project: Optional[Project] = None
        self._rows: List[VideoEntry] = []
        self._metadata_columns: List[str] = []
        self._issues: Dict[str, List[ProjectIssue]] = {}
        self._icons: Dict[str, QIcon] = {}
        self._busy = False          # our own edits: skip the listener
        self.recursive_drop = False  # expand dropped folders recursively
        if project is not None:
            self.set_project(project)

    # ------------------------------------------------------------------
    # The project
    # ------------------------------------------------------------------
    def project(self) -> Optional[Project]:
        return self._project

    def set_project(self, project: Optional[Project]) -> None:
        self.beginResetModel()
        if self._project is not None:
            self._project.remove_listener(self._on_project_changed)
        self._project = project
        if project is not None:
            project.add_listener(self._on_project_changed)
        self._take_snapshot()
        self.endResetModel()

    def _take_snapshot(self) -> None:
        project = self._project
        self._rows = list(project.videos) if project else []
        self._metadata_columns = project.metadata_columns() if project else []
        self._recompute_issues()

    def _recompute_issues(self) -> None:
        self._issues = {}
        if self._project is None:
            return
        for issue in self._project.validate():
            if issue.key is not None:
                self._issues.setdefault(issue.key, []).append(issue)

    def project_issues(self) -> List[ProjectIssue]:
        """Project-level issues (not tied to one video), freshly validated."""
        if self._project is None:
            return []
        return [i for i in self._project.validate() if i.key is None]

    def _on_project_changed(self) -> None:
        if self._busy:
            return
        try:
            self.refresh()
        except RuntimeError:
            # The Qt object was deleted without ``set_project(None)``.
            if self._project is not None:
                self._project.remove_listener(self._on_project_changed)

    def refresh(self) -> None:
        """Re-read the project (after changes made outside the model).

        Rows and columns are only reset when they changed; otherwise just the
        cells are reported as changed, so the selection stays.
        """
        project = self._project
        same_shape = (
            project is not None
            and [v.key for v in project.videos] == [v.key for v in self._rows]
            and project.metadata_columns() == self._metadata_columns
        )
        if same_shape:
            self._rows = list(project.videos)
            self._recompute_issues()
            self._emit_all_changed()
        else:
            self.beginResetModel()
            self._take_snapshot()
            self.endResetModel()
        self.projectChanged.emit()

    def refresh_issues(self) -> None:
        """Validate again (files may have appeared or changed)."""
        self._recompute_issues()
        self._emit_column_changed(self._issues_column())

    def _emit_all_changed(self) -> None:
        if self._rows and self.columnCount():
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(len(self._rows) - 1, self.columnCount() - 1),
            )

    def _emit_column_changed(self, column: int) -> None:
        if self._rows:
            self.dataChanged.emit(
                self.index(0, column), self.index(len(self._rows) - 1, column)
            )

    # ------------------------------------------------------------------
    # Columns
    # ------------------------------------------------------------------
    def metadata_columns(self) -> List[str]:
        return list(self._metadata_columns)

    def _last_run_column(self) -> int:
        return LEADING_COLUMNS + len(self._metadata_columns)

    def _notes_column(self) -> int:
        return self._last_run_column() + 1

    def _issues_column(self) -> int:
        return self._last_run_column() + 2

    def column_for_metadata(self, name: str) -> int:
        """Table column of a metadata key (-1 if the project has none)."""
        try:
            return LEADING_COLUMNS + self._metadata_columns.index(name)
        except ValueError:
            return -1

    def _metadata_name(self, column: int) -> Optional[str]:
        offset = column - LEADING_COLUMNS
        if 0 <= offset < len(self._metadata_columns):
            return self._metadata_columns[offset]
        return None

    def headers(self) -> List[str]:
        return (
            [HEADER_STATUS, HEADER_VIDEO, HEADER_CONFIGURED, HEADER_SETTINGS,
             HEADER_TRIAL, HEADER_ZONES]
            + list(self._metadata_columns)
            + [HEADER_LAST_RUN, HEADER_NOTES, HEADER_ISSUES]
        )

    def rowCount(self, parent: Optional[QModelIndex] = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self._rows)

    def columnCount(self, parent: Optional[QModelIndex] = None) -> int:
        if parent is not None and parent.isValid():
            return 0
        return LEADING_COLUMNS + len(self._metadata_columns) + 3

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            headers = self.headers()
            return headers[section] if 0 <= section < len(headers) else None
        return str(section + 1)

    # ------------------------------------------------------------------
    # Cells
    # ------------------------------------------------------------------
    def _trial_state(self, entry: VideoEntry) -> str:
        if self._project is None:
            return TRIAL_OK
        return self._project.trial_status(entry)

    def _configured_text(self, entry: VideoEntry) -> str:
        """The composed state: settings, then trial, then verification."""
        if entry.settings is None:
            return "Not configured"
        state = self._trial_state(entry)
        if state == TRIAL_NEEDS_START:
            return "Needs start"
        if state == TRIAL_INVALID:
            return "Invalid trial"
        return "Verified" if entry.verified else "Configured"

    def _configured_rank(self, entry: VideoEntry) -> int:
        if entry.settings is None:
            return 0
        if self._trial_state(entry) != TRIAL_OK:
            return 1
        return 3 if entry.verified else 2

    def _trial_text(self, entry: VideoEntry) -> str:
        project = self._project
        if project is None or entry.settings is None:
            return ""
        state = project.trial_status(entry)
        if state == TRIAL_NEEDS_START:
            return TEXT_NEEDS_START
        if state == TRIAL_INVALID:
            return TEXT_INVALID_TRIAL
        span = project.trial_span(entry)
        return "" if span is None else format_trial(*span)

    def _zones_text(self, entry: VideoEntry) -> str:
        if self._project is None or entry.settings is None:
            return ""
        return self._project.zone_status(entry).capitalize()

    def _trial_sort(self, entry: VideoEntry):
        """Invalid first, then videos without a start, then by start time."""
        project = self._project
        if project is None or entry.settings is None:
            return (3, 0.0)
        state = project.trial_status(entry)
        if state == TRIAL_INVALID:
            return (0, 0.0)
        if state == TRIAL_NEEDS_START:
            return (1, 0.0)
        span = project.trial_span(entry)
        return (2, float(span[0]) if span is not None else 0.0)

    def issues_for(self, key: str) -> List[ProjectIssue]:
        return list(self._issues.get(key, []))

    def _issues_text(self, key: str) -> str:
        return "; ".join(issue.message for issue in self._issues.get(key, []))

    def _text(self, entry: VideoEntry, column: int) -> str:
        if column == COL_STATUS:
            return _STATUS_TEXT.get(entry.last_status, entry.last_status)
        if column == COL_VIDEO:
            return entry.name
        if column == COL_CONFIGURED:
            return self._configured_text(entry)
        if column == COL_SETTINGS:
            return entry.settings_source.capitalize() if entry.settings else ""
        if column == COL_TRIAL:
            return self._trial_text(entry)
        if column == COL_ZONES:
            return self._zones_text(entry)
        name = self._metadata_name(column)
        if name is not None:
            return entry.metadata.get(name, "")
        if column == self._last_run_column():
            return entry.last_run_dir.name if entry.last_run_dir else ""
        if column == self._notes_column():
            return entry.notes
        if column == self._issues_column():
            return self._issues_text(entry.key)
        return ""

    def _sort_value(self, entry: VideoEntry, column: int):
        if column == COL_STATUS:
            return STATUSES.index(entry.last_status)
        if column == COL_VIDEO:
            return entry.name.casefold()
        if column == COL_CONFIGURED:
            return self._configured_rank(entry)
        if column == COL_TRIAL:
            return self._trial_sort(entry)
        if column == self._issues_column():
            issues = self._issues.get(entry.key, [])
            errors = sum(1 for i in issues if i.severity == SEVERITY_ERROR)
            return (errors, len(issues))
        return self._text(entry, column).casefold()

    def _icon(self, status: str) -> QIcon:
        badge = _ICON_STATUS.get(status, "queued")
        if badge not in self._icons:
            self._icons[badge] = status_icon(badge)
        return self._icons[badge]

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self._rows):
            return None
        entry = self._rows[index.row()]
        column = index.column()
        if role == KeyRole:
            return entry.key
        if role == PathRole:
            return entry.path
        if role == SortRole:
            return self._sort_value(entry, column)
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return self._text(entry, column)
        if role == Qt.ItemDataRole.DecorationRole and column == COL_STATUS:
            return self._icon(entry.last_status)
        if role == Qt.ItemDataRole.ToolTipRole:
            return self._tooltip(entry, column)
        return None

    def _tooltip(self, entry: VideoEntry, column: int) -> Optional[str]:
        if column == COL_VIDEO:
            return str(entry.path)
        project = self._project
        if column == COL_TRIAL and project is not None and entry.settings:
            problem = project.trial_problem(entry)
            if problem is not None:
                return problem[:1].upper() + problem[1:]
            if project.trial_status(entry) == TRIAL_NEEDS_START:
                return "This project gives every video its own start time"
            return "When the analysed part of the video starts and ends"
        if column == COL_ZONES and project is not None and entry.settings:
            return {
                "template": "The regions of the project template",
                "adjusted": "Regions adjusted for this video",
                "aligned": "The template regions moved onto this video",
                "custom": "This video has its own full copy of the settings",
            }.get(project.zone_status(entry))
        if column == self._issues_column():
            lines = [issue.message for issue in self._issues.get(entry.key, [])]
            return "\n".join(lines) or None
        if column == self._last_run_column() and entry.last_run_dir:
            return str(entry.last_run_dir)
        if column == self._notes_column():
            return entry.notes or None
        return None

    # ------------------------------------------------------------------
    # Editing: metadata and notes only
    # ------------------------------------------------------------------
    def _is_editable_column(self, column: int) -> bool:
        return (
            self._metadata_name(column) is not None
            or column == self._notes_column()
        )

    def flags(self, index: QModelIndex):
        if not index.isValid():
            return Qt.ItemFlag.ItemIsDropEnabled
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if self._is_editable_column(index.column()):
            flags |= Qt.ItemFlag.ItemIsEditable
        return flags

    def setData(self, index: QModelIndex, value, role=Qt.ItemDataRole.EditRole):
        if (
            role != Qt.ItemDataRole.EditRole
            or self._project is None
            or not index.isValid()
            or not self._is_editable_column(index.column())
        ):
            return False
        entry = self._rows[index.row()]
        text = "" if value is None else str(value)
        if text == self._text(entry, index.column()):
            return True
        name = self._metadata_name(index.column())
        self._busy = True
        try:
            if name is not None:
                self._project.set_metadata_value(entry.key, name, text)
            else:
                self._project.set_video_notes(entry.key, text)
        finally:
            self._busy = False
        self.dataChanged.emit(index, index)
        self.projectChanged.emit()
        return True

    def add_metadata_column(self, name: str) -> bool:
        """Add a metadata column to every video; False if it exists."""
        if self._project is None:
            return False
        position = self._last_run_column()
        self._busy = True
        try:
            added = self._project.add_metadata_column(name)
        finally:
            self._busy = False
        if not added:
            return False
        self.beginInsertColumns(QModelIndex(), position, position)
        self._metadata_columns = self._project.metadata_columns()
        self.endInsertColumns()
        self.projectChanged.emit()
        return True

    def remove_metadata_column(self, name: str) -> bool:
        column = self.column_for_metadata(name)
        if self._project is None or column < 0:
            return False
        self._busy = True
        try:
            self._project.remove_metadata_column(name)
        finally:
            self._busy = False
        self.beginRemoveColumns(QModelIndex(), column, column)
        self._metadata_columns = self._project.metadata_columns()
        self.endRemoveColumns()
        self.projectChanged.emit()
        return True

    # ------------------------------------------------------------------
    # Rows: add, remove, move (with proper signals)
    # ------------------------------------------------------------------
    def entry_at(self, row: int) -> VideoEntry:
        return self._rows[row]

    def key_at(self, row: int) -> str:
        return self._rows[row].key

    def row_for_key(self, key: str) -> int:
        for row, entry in enumerate(self._rows):
            if entry.key == key:
                return row
        return -1

    def keys_for_rows(self, rows: Iterable[int]) -> List[str]:
        """Keys of valid rows, in table order, without duplicates."""
        wanted = {row for row in rows if 0 <= row < len(self._rows)}
        return [self._rows[row].key for row in sorted(wanted)]

    def rows_for_keys(self, keys: Iterable[str]) -> List[int]:
        wanted = set(keys)
        return [r for r, entry in enumerate(self._rows) if entry.key in wanted]

    def keys_for_indexes(self, indexes: Iterable[QModelIndex]) -> List[str]:
        """Keys of the rows holding ``indexes`` (table model indexes)."""
        return self.keys_for_rows(i.row() for i in indexes if i.isValid())

    def entries_for_keys(self, keys: Iterable[str]) -> List[VideoEntry]:
        wanted = set(keys)
        return [entry for entry in self._rows if entry.key in wanted]

    def add_videos(self, paths: Iterable[Path]) -> AddVideosResult:
        """Add videos to the project (duplicates are reported, not added)."""
        if self._project is None:
            return AddVideosResult()
        self._busy = True
        try:
            result = self._project.add_videos(paths)
        finally:
            self._busy = False
        if result.added:
            first = len(self._rows)
            self.beginInsertRows(QModelIndex(), first, first + len(result.added) - 1)
            self._rows = list(self._project.videos)
            self.endInsertRows()
            self._sync_issues()
            self.projectChanged.emit()
        return result

    def remove_keys(self, keys: Iterable[str]) -> List[VideoEntry]:
        if self._project is None:
            return []
        rows = sorted(self.rows_for_keys(keys), reverse=True)
        if not rows:
            return []
        self._busy = True
        try:
            removed = self._project.remove_videos(self.keys_for_rows(rows))
        finally:
            self._busy = False
        # One signal pair per contiguous block, bottom-up so rows stay valid.
        block_end = rows[0]
        block_start = rows[0]
        blocks: List[Tuple[int, int]] = []
        for row in rows[1:]:
            if row == block_start - 1:
                block_start = row
            else:
                blocks.append((block_start, block_end))
                block_start = block_end = row
        blocks.append((block_start, block_end))
        for first, last in blocks:
            self.beginRemoveRows(QModelIndex(), first, last)
            del self._rows[first:last + 1]
            self.endRemoveRows()
        if self._project.metadata_columns() != self._metadata_columns:
            self.beginResetModel()
            self._take_snapshot()
            self.endResetModel()
        else:
            self._sync_issues()
        self.projectChanged.emit()
        return removed

    def move_keys(self, keys: Iterable[str], offset: int) -> bool:
        """Move videos up (negative) or down; persistent indexes follow."""
        if self._project is None:
            return False
        selected = list(keys)
        self.layoutAboutToBeChanged.emit(
            [], QAbstractItemModel.LayoutChangeHint.NoLayoutChangeHint
        )
        old_indexes = self.persistentIndexList()
        old_keys = [self._rows[i.row()].key for i in old_indexes]
        self._busy = True
        try:
            moved = self._project.move_videos(selected, offset)
        finally:
            self._busy = False
        self._rows = list(self._project.videos)
        position = {entry.key: row for row, entry in enumerate(self._rows)}
        for index, key in zip(old_indexes, old_keys):
            self.changePersistentIndex(
                index, self.index(position[key], index.column())
            )
        self.layoutChanged.emit(
            [], QAbstractItemModel.LayoutChangeHint.NoLayoutChangeHint
        )
        if moved:
            self.projectChanged.emit()
        return moved

    def _sync_issues(self) -> None:
        self._recompute_issues()
        self._emit_column_changed(self._issues_column())

    # ------------------------------------------------------------------
    # Drag and drop of files and folders
    # ------------------------------------------------------------------
    def supportedDropActions(self):
        return Qt.DropAction.CopyAction

    def mimeTypes(self) -> List[str]:
        return ["text/uri-list"]

    def canDropMimeData(self, data, action, row, column, parent) -> bool:
        return (
            action == Qt.DropAction.CopyAction
            and data is not None
            and data.hasUrls()
        )

    def dropMimeData(self, data, action, row, column, parent) -> bool:
        """Emit ``pathsDropped`` with the videos; the shell adds them.

        Folders are expanded to their videos (``recursive_drop``); other
        files are reported through ``pathsIgnored``.
        """
        if not self.canDropMimeData(data, action, row, column, parent):
            return False
        videos, ignored = paths_from_mime(data, self.recursive_drop)
        if ignored:
            self.pathsIgnored.emit(ignored)
        if not videos:
            return False
        self.pathsDropped.emit(videos)
        return True


class ProjectFilterProxyModel(QSortFilterProxyModel):
    """Sorting and filtering (text and status) of a ``ProjectTableModel``."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tokens: List[str] = []
        self._statuses: frozenset = frozenset()
        self.setSortRole(SortRole)
        self.setDynamicSortFilter(True)

    def set_text_filter(self, text: str) -> None:
        """Rows containing every word of ``text`` (case-insensitive).

        Searched: video name, status, settings source, metadata values, the
        last run folder, notes and the issue messages.
        """
        tokens = [token for token in text.casefold().split() if token]
        if tokens != self._tokens:
            self._tokens = tokens
            self.invalidateFilter()

    def set_status_filter(self, statuses: Iterable[str]) -> None:
        """Only rows whose last status is in ``statuses`` (empty = all)."""
        wanted = frozenset(statuses)
        if wanted != self._statuses:
            self._statuses = wanted
            self.invalidateFilter()

    def clear_filters(self) -> None:
        self._tokens = []
        self._statuses = frozenset()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent) -> bool:
        model = self.sourceModel()
        if not isinstance(model, ProjectTableModel):
            return True
        if not 0 <= source_row < model.rowCount():
            return False
        entry = model.entry_at(source_row)
        if self._statuses and entry.last_status not in self._statuses:
            return False
        if not self._tokens:
            return True
        parts = [
            model.data(model.index(source_row, column)) or ""
            for column in range(model.columnCount())
        ]
        haystack = " ".join(str(part) for part in parts).casefold()
        return all(token in haystack for token in self._tokens)

    # -- selection helpers ---------------------------------------------
    def keys_for_rows(self, rows: Iterable[int]) -> List[str]:
        """Keys of proxy rows, in the order shown."""
        return selected_keys(
            self.index(row, 0) for row in sorted(set(rows))
            if 0 <= row < self.rowCount()
        )

    def all_keys(self) -> List[str]:
        """Keys of every row that passes the filter, in the order shown."""
        return self.keys_for_rows(range(self.rowCount()))

    def row_for_key(self, key: str) -> int:
        for row in range(self.rowCount()):
            if self.index(row, 0).data(KeyRole) == key:
                return row
        return -1


__all__: Sequence[str] = (
    "KeyRole", "SortRole", "PathRole", "ProjectTableModel",
    "ProjectFilterProxyModel", "selected_keys", "paths_from_mime",
    "format_clock", "format_trial",
)
