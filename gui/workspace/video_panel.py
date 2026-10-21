"""
gui/workspace/video_panel.py
The left pane of the workspace: the videos of the project as a table with a
search box, a status filter, a summary footer and all the editing actions.

Everything the user does here goes through the Qt-free ``Project``:

* adding, removing and moving videos use ``ProjectTableModel`` (proper row
  signals, so the selection survives),
* metadata columns and notes are edited in place,
* the context menu offers the per-video actions.

The panel never asks the operating system or the user anything itself; it
uses the shared :class:`~gui.workspace.prompts.Prompts` object, so the same
panel works under test with scripted answers.

Public surface for the rest of the workspace:

* ``selection_changed(list)`` - the keys of the selected videos, in table
  order, every time the selection (or the rows under it) changes;
* ``selected_keys()``, ``select_keys(keys)``, ``clear_selection()``;
* the actions as methods: ``add_videos_dialog``, ``add_folder_dialog``,
  ``add_paths``, ``remove_selected``, ``move_selected``, ``mark_verified``,
  ``apply_template``, ``edit_metadata_column``, ``load_sample_sheet``,
  ``relink``, ``add_column``.
"""
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from PyQt6.QtCore import (QItemSelectionModel, QModelIndex, QPoint, Qt, QTimer,
                          pyqtSignal)
from PyQt6.QtGui import QAction, QBrush, QColor, QKeyEvent
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox,
                             QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QMenu, QPushButton, QTableView, QVBoxLayout,
                             QWidget)

from core.project import (SEVERITY_ERROR, SEVERITY_INFO, STATUSES,
                          AddVideosResult, Project, ProjectError)
from core.sample_sheet import SampleSheetError, describe_match
from gui.app_settings import AppSettings
from gui.project_table_model import (LEADING_COLUMNS, KeyRole,
                                     ProjectFilterProxyModel,
                                     ProjectTableModel)
from gui.video_list import collect_video_files
from gui.workspace.prompts import SHEET_FILTER, VIDEO_FILTER, Prompts

logger = logging.getLogger(__name__)

MISSING_FILE = "missing_file"
ERROR_COLOR = QColor("#d1242f")

FILTER_ALL = "All statuses"
_FILTER_LABELS = {
    "new": "New",
    "configured": "Configured",
    "done": "Done",
    "failed": "Failed",
    "warning": "Warning",
    "interrupted": "Interrupted",
}


class WorkspaceProxyModel(ProjectFilterProxyModel):
    """The filter proxy, plus red text for videos whose file is missing."""

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.ForegroundRole and index.isValid():
            model = self.sourceModel()
            if isinstance(model, ProjectTableModel):
                source = self.mapToSource(index)
                key = model.index(source.row(), 0).data(KeyRole)
                if key is not None and any(
                    issue.code == MISSING_FILE for issue in model.issues_for(key)
                ):
                    return QBrush(ERROR_COLOR)
        return super().data(index, role)


class VideoTableView(QTableView):
    """The table, with Delete and Alt+Up/Down turned into signals."""

    remove_requested = pyqtSignal()
    move_requested = pyqtSignal(int)    # -1 up, +1 down

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


class VideoPanel(QWidget):
    """Search, filter, table, footer and actions for the project's videos."""

    selection_changed = pyqtSignal(list)    # keys of the selected videos
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
    ):
        super().__init__(parent)
        self.app_settings = app_settings or AppSettings()
        self.prompts = prompts or Prompts(self)
        self._project: Optional[Project] = None
        self._read_only = False
        self._last_keys: List[str] = []
        self.model = ProjectTableModel(parent=self)
        self.proxy = WorkspaceProxyModel(self)
        self.proxy.setSourceModel(self.model)
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
        self.view.selectionModel().selectionChanged.connect(
            self._schedule_selection
        )
        self.set_project(None)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        top = QHBoxLayout()
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search videos, metadata, notes...")
        self.search_box.setClearButtonEnabled(True)
        self.search_box.textChanged.connect(self._on_filter_changed)
        top.addWidget(self.search_box, 1)
        self.status_combo = QComboBox()
        self.status_combo.addItem(FILTER_ALL, "")
        for status in STATUSES:
            self.status_combo.addItem(_FILTER_LABELS.get(status, status), status)
        self.status_combo.currentIndexChanged.connect(self._on_filter_changed)
        top.addWidget(self.status_combo)
        layout.addLayout(top)

        self.view = VideoTableView()
        self.view.setModel(self.proxy)
        self.view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.view.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.view.setAlternatingRowColors(True)
        self.view.setWordWrap(False)
        self.view.setSortingEnabled(True)
        self.view.sortByColumn(-1, Qt.SortOrder.AscendingOrder)
        self.view.verticalHeader().setVisible(False)
        header = self.view.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setMinimumSectionSize(40)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._show_context_menu)
        self.view.remove_requested.connect(self.remove_selected)
        self.view.move_requested.connect(self.move_selected)
        layout.addWidget(self.view, 1)

        buttons = QHBoxLayout()
        self.add_column_button = QPushButton("Add column")
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
        buttons.addWidget(self.recursive_check)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.empty_label = QLabel(
            "Drop videos or a folder here,\nor use Add videos / Add folder."
        )
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setEnabled(False)
        layout.addWidget(self.empty_label)

        self.footer = QLabel()
        self.footer.setTextFormat(Qt.TextFormat.RichText)
        self.footer.setWordWrap(True)
        layout.addWidget(self.footer)

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

    def is_read_only(self) -> bool:
        return self._read_only

    def _blocked(self) -> bool:
        """True when edits are not allowed (no project, or read-only)."""
        return self._project is None or self._read_only

    def set_project(self, project: Optional[Project]) -> None:
        self._project = project
        self.model.recursive_drop = self.recursive_check.isChecked()
        self.model.set_project(project)
        enabled = project is not None
        self.search_box.setEnabled(enabled)
        self.status_combo.setEnabled(enabled)
        self.add_column_button.setEnabled(enabled and not self._read_only)
        self.view.setEnabled(enabled)
        self.proxy.clear_filters()
        self.search_box.clear()
        self.status_combo.setCurrentIndex(0)
        self._refresh_chrome()

    def _on_model_changed(self) -> None:
        self._refresh_chrome()

    def _on_model_reset(self) -> None:
        self._refresh_chrome()
        self._schedule_selection()

    def _refresh_chrome(self) -> None:
        """Footer and empty-state label."""
        project = self._project
        self.empty_label.setVisible(project is not None and not project.videos)
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
                f'<span style="color:{ERROR_COLOR.name()}">'
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
                "in the Setup tab first.",
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
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if not self._blocked() and event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        paths = [
            Path(url.toLocalFile()) for url in event.mimeData().urls()
            if url.isLocalFile() and url.toLocalFile()
        ]
        if self._blocked() or not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        self.paths_dropped.emit(paths)


__all__ = ["VideoPanel", "VideoTableView", "WorkspaceProxyModel"]
