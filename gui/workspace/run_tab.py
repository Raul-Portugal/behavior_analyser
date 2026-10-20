"""
gui/workspace/run_tab.py
The Run tab of the workspace: start the analysis of the project's videos and
follow it.

::

    +- Run settings -----------------------------------------------------+
    | Output folder | Scope | Parallel videos | Time bins | Validation video |
    | Sample sheet status                                                |
    +--------------------------------------------------------------------+
    [Run analysis] [Cancel] [Resume interrupted run]       Batch ETA ...
    overall progress bar
    per-video table: status, progress, ETA, last message
    log (level + text filter)
    completion summary: [Open results] [Open folder]

The tab owns the analysis thread. It builds the plan from the project
(:mod:`gui.workspace.run_check` checks it first), runs ``AnalysisWorker`` on a
``QThread`` exactly like the classic window and records the finished run in
the project. Analysis itself is untouched: the same worker and ``BatchRunner``
produce the same files as the classic interface and the CLI.

Closing the window during a run asks, cancels, keeps the window open until
the worker has saved its checkpoint, then closes it. The timers belong to this
widget, so a pending force-quit offer or close never touches deleted objects.

Validation videos
-----------------
The checkbox is tri-state. *Each video's own setting* (the default) leaves
``BatchSettings.create_timelapse`` as configured per video; checked or
unchecked forces it on or off for every video of **this run only** (the run
works on copies of the settings, the project is not changed).
"""
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PyQt6.QtCore import (QAbstractTableModel, QModelIndex, QThread, QTimer,
                          Qt, pyqtSignal)
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                             QFrame, QGridLayout, QGroupBox, QHBoxLayout,
                             QHeaderView, QLabel, QLineEdit, QPlainTextEdit,
                             QProgressBar, QPushButton, QSpinBox, QSplitter,
                             QStyle, QStyledItemDelegate, QStyleOptionProgressBar,
                             QTableView, QVBoxLayout, QWidget)

from core.batch_runner import BatchRunOutcome, default_worker_count
from core.project import Project, ProjectError
from gui import ux_helpers
from gui.video_list import (STATUS_DONE, STATUS_FAILED, STATUS_LABELS,
                            STATUS_QUEUED, STATUS_RUNNING, STATUS_WARNING,
                            outcome_statuses, status_icon)
from gui.worker import AnalysisWorker
from gui.workspace.run_check import (SCOPE_ALL, SCOPE_LABELS, SCOPE_SELECTED, SCOPES, ResumeOffer,
                                     RunPreparation, find_resumable,
                                     prepare_run, scope_keys)
from gui.workspace.tabs import WorkspaceTab

logger = logging.getLogger(__name__)

FORCE_QUIT_DELAY_MS = 10_000
MAX_LOG_ENTRIES = 20_000
MAX_LISTED_PROBLEMS = 14
PROGRESS_ROLE = Qt.ItemDataRole.UserRole + 1

_VIDEO_MAIN = re.compile(r"^Video:\s*(.+)$")
_VIDEO_ETA = re.compile(r"(?:Video\s+)?ETA:\s*([^|]+?)(?:\s*\||$)")

# A running QThread must outlive a window that was destroyed under it (tests,
# force quit), or Qt aborts. Entries leave this set when the thread finishes.
_live_threads: Set[QThread] = set()


# ----------------------------------------------------------------------
# The per-video progress table
# ----------------------------------------------------------------------
class RunProgressModel(QAbstractTableModel):
    """One row per video of the run: status, percentage, ETA, last message."""

    COLUMNS = ("Video", "Status", "Progress", "ETA", "Message")
    VIDEO, STATUS, PROGRESS, ETA, MESSAGE = range(5)
    # Percentage that means "running, but no percentage is known".
    INDETERMINATE = -1

    def __init__(self, parent=None):
        super().__init__(parent)
        self._paths: List[Path] = []
        self._rows: Dict[Path, dict] = {}
        self._icons: Dict[str, object] = {}

    # -- content ---------------------------------------------------------
    def reset_rows(self, paths: List[Path]) -> None:
        self.beginResetModel()
        self._paths = list(paths)
        self._rows = {
            path: {"status": STATUS_QUEUED, "pct": 0, "eta": "", "message": ""}
            for path in self._paths
        }
        self.endResetModel()

    def paths(self) -> List[Path]:
        return list(self._paths)

    def row_of(self, path: Path) -> int:
        try:
            return self._paths.index(Path(path))
        except ValueError:
            return -1

    def state(self, path: Path) -> Optional[dict]:
        found = self._rows.get(Path(path))
        return dict(found) if found is not None else None

    def _touch(self, path: Path) -> None:
        row = self.row_of(path)
        if row >= 0:
            self.dataChanged.emit(
                self.index(row, 0), self.index(row, len(self.COLUMNS) - 1)
            )

    def set_status(self, path: Path, status: str, message: str = "") -> None:
        row = self._rows.get(Path(path))
        if row is None:
            return
        row["status"] = status
        if message or status in (STATUS_DONE, STATUS_WARNING, STATUS_FAILED):
            row["message"] = message
        if status in (STATUS_DONE, STATUS_WARNING):
            row["pct"], row["eta"] = 100, ""
        elif status == STATUS_FAILED:
            row["eta"] = ""
        self._touch(path)

    def set_progress(
        self, path: Path, pct: Optional[int], eta: str, message: str
    ) -> None:
        row = self._rows.get(Path(path))
        if row is None or row["status"] in (STATUS_DONE, STATUS_WARNING,
                                            STATUS_FAILED):
            return
        row["status"] = STATUS_RUNNING
        if pct is None:
            row["pct"] = self.INDETERMINATE if row["pct"] <= 0 else row["pct"]
        else:
            row["pct"] = max(int(pct), row["pct"], 0)
        if eta:
            row["eta"] = eta
        if message:
            row["message"] = message
        self._touch(path)

    # -- Qt model --------------------------------------------------------
    def rowCount(self, parent: Optional[QModelIndex] = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self._paths)

    def columnCount(self, parent: Optional[QModelIndex] = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if (
            role == Qt.ItemDataRole.DisplayRole
            and orientation == Qt.Orientation.Horizontal
        ):
            return self.COLUMNS[section]
        return None

    def _icon(self, status: str):
        if status not in self._icons:
            self._icons[status] = status_icon(status)
        return self._icons[status]

    @staticmethod
    def percent_text(pct: int) -> str:
        return "..." if pct == RunProgressModel.INDETERMINATE else f"{pct}%"

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._paths)):
            return None
        path = self._paths[index.row()]
        row = self._rows[path]
        column = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if column == self.VIDEO:
                return path.name
            if column == self.STATUS:
                return STATUS_LABELS.get(row["status"], row["status"])
            if column == self.PROGRESS:
                return self.percent_text(row["pct"])
            if column == self.ETA:
                return row["eta"]
            if column == self.MESSAGE:
                return row["message"]
        elif role == Qt.ItemDataRole.DecorationRole and column == self.STATUS:
            return self._icon(row["status"])
        elif role == Qt.ItemDataRole.ToolTipRole:
            if column == self.VIDEO:
                return str(path)
            if column == self.MESSAGE:
                return row["message"] or None
        elif role == PROGRESS_ROLE and column == self.PROGRESS:
            return row["pct"]
        return None


class ProgressBarDelegate(QStyledItemDelegate):
    """Paints the Progress column as a (possibly busy) progress bar."""

    def paint(self, painter, option, index) -> None:
        pct = index.data(PROGRESS_ROLE)
        if pct is None:
            super().paint(painter, option, index)
            return
        bar = QStyleOptionProgressBar()
        bar.rect = option.rect.adjusted(2, 3, -2, -3)
        bar.state = option.state
        bar.minimum = 0
        if pct == RunProgressModel.INDETERMINATE:
            bar.maximum = 0
            bar.progress = 0
            bar.text = ""
        else:
            bar.maximum = 100
            bar.progress = int(pct)
            bar.text = f"{int(pct)}%"
            bar.textVisible = True
        QApplication.style().drawControl(
            QStyle.ControlElement.CE_ProgressBar, bar, painter
        )


# ----------------------------------------------------------------------
# The tab
# ----------------------------------------------------------------------
def format_problems(issues: List[str], limit: int = MAX_LISTED_PROBLEMS) -> str:
    lines = [f"- {issue}" for issue in issues[:limit]]
    if len(issues) > limit:
        lines.append(f"... and {len(issues) - limit} more")
    return "\n".join(lines)


class RunTab(WorkspaceTab):
    """Start, follow and cancel the analysis of the project's videos."""

    run_started = pyqtSignal()
    run_finished = pyqtSignal(object)      # BatchRunOutcome

    FORCE_QUIT_DELAY_MS = FORCE_QUIT_DELAY_MS

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.title = "Run"
        self._loading = False
        self._checking = False
        self._thread: Optional[QThread] = None
        self._worker: Optional[AnalysisWorker] = None
        self._close_requested = False
        self._force_close = False
        self._run_started = 0.0
        self._worker_eta: Optional[str] = None
        self._sequential = True
        self._base_paths: Dict[str, Path] = {}
        self._current_path: Optional[Path] = None
        self._explicit_output: Optional[Path] = None
        self._last_run_dir: Optional[Path] = None
        self._last_outcome: Optional[BatchRunOutcome] = None
        self._resume = ResumeOffer()
        self._log_entries: List[Tuple[int, str]] = []
        self.last_report: Optional[RunPreparation] = None

        self._build_ui()
        # Owned by this widget: they die with it and never touch a deleted one.
        self._close_timer = QTimer(self)
        self._close_timer.setSingleShot(True)
        self._close_timer.setInterval(0)
        self._close_timer.timeout.connect(self._close_window)
        self._force_timer = QTimer(self)
        self._force_timer.setSingleShot(True)
        self._force_timer.timeout.connect(self._offer_force_quit)
        self._refresh()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.settings_box = QGroupBox("Run settings")
        grid = QGridLayout(self.settings_box)

        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setToolTip(
            "Each run creates an analysis_run_ folder inside this folder."
        )
        self.output_button = QPushButton("Change...")
        self.output_button.clicked.connect(self.choose_output_dir)
        grid.addWidget(QLabel("Output folder:"), 0, 0)
        grid.addWidget(self.output_edit, 0, 1, 1, 3)
        grid.addWidget(self.output_button, 0, 4)

        self.scope_combo = QComboBox()
        for scope in SCOPES:
            self.scope_combo.addItem(SCOPE_LABELS[scope], scope)
        self.scope_combo.setToolTip(
            "Which videos to analyse: every video, the ones selected in the "
            "table, or the ones that have no finished analysis yet."
        )
        self.scope_combo.currentIndexChanged.connect(self._on_scope_changed)
        self.scope_label = QLabel()
        grid.addWidget(QLabel("Run:"), 1, 0)
        grid.addWidget(self.scope_combo, 1, 1)
        grid.addWidget(self.scope_label, 1, 2, 1, 3)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 64)
        self.workers_spin.setSpecialValueText("Auto")
        self.workers_spin.setToolTip(
            "Videos analysed at once, each in its own process. Auto picks a "
            "number from the CPU cores and memory."
        )
        self.workers_spin.valueChanged.connect(self._on_workers_changed)
        self.bins_spin = QDoubleSpinBox()
        self.bins_spin.setRange(0.0, 14400.0)
        self.bins_spin.setDecimals(1)
        self.bins_spin.setSingleStep(10.0)
        self.bins_spin.setSpecialValueText("Off")
        self.bins_spin.setSuffix(" s")
        self.bins_spin.setToolTip(
            "Also report the metrics in consecutive bins of this many "
            "seconds (batch_time_bins.csv). Off leaves the outputs as they "
            "are without this option."
        )
        self.bins_spin.valueChanged.connect(self._on_bins_changed)
        grid.addWidget(QLabel("Parallel videos:"), 2, 0)
        grid.addWidget(self.workers_spin, 2, 1)
        grid.addWidget(QLabel("Time bins:"), 2, 2)
        grid.addWidget(self.bins_spin, 2, 3)

        self.validation_check = QCheckBox("Validation videos")
        self.validation_check.setTristate(True)
        self.validation_check.setCheckState(Qt.CheckState.PartiallyChecked)
        self.validation_check.setToolTip(
            "Partly filled: each video keeps its own setting. Checked or "
            "unchecked: create (or skip) the validation video for every "
            "video of this run. The project's settings are not changed."
        )
        self.validation_check.stateChanged.connect(self._on_validation_changed)
        self.validation_label = QLabel()
        grid.addWidget(self.validation_check, 3, 0, 1, 2)
        grid.addWidget(self.validation_label, 3, 2, 1, 3)

        self.sheet_label = QLabel()
        self.sheet_label.setWordWrap(True)
        self.sheet_button = QPushButton("Sample sheet...")
        self.sheet_button.setToolTip(
            "Load animal metadata (group, sex, ...) from a CSV into the video "
            "table. The metadata columns are added to the batch summary."
        )
        self.sheet_button.clicked.connect(self._load_sample_sheet)
        grid.addWidget(QLabel("Metadata:"), 4, 0)
        grid.addWidget(self.sheet_label, 4, 1, 1, 3)
        grid.addWidget(self.sheet_button, 4, 4)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        layout.addWidget(self.settings_box)

        actions = QHBoxLayout()
        self.run_button = QPushButton("Run analysis")
        self.run_button.setDefault(True)
        self.run_button.clicked.connect(self.start_run)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.request_cancel)
        self.resume_button = QPushButton("Resume interrupted run")
        self.resume_button.setVisible(False)
        self.resume_button.clicked.connect(self.start_resume)
        self.resume_label = QLabel()
        self.resume_label.setVisible(False)
        actions.addWidget(self.run_button)
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.resume_button)
        actions.addWidget(self.resume_label, 1)
        layout.addLayout(actions)

        self.status_label = QLabel("Ready")
        self.eta_label = QLabel()
        self.eta_label.setVisible(False)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_row = QHBoxLayout()
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.eta_label)
        layout.addLayout(status_row)
        layout.addWidget(self.progress_bar)

        self.summary_frame = QFrame()
        self.summary_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.summary_frame.setVisible(False)
        summary_layout = QHBoxLayout(self.summary_frame)
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.open_results_button = QPushButton("Open results")
        self.open_results_button.clicked.connect(self.open_results)
        self.open_folder_button = QPushButton("Open folder")
        self.open_folder_button.clicked.connect(self.open_run_folder)
        self.dismiss_button = QPushButton("Dismiss")
        self.dismiss_button.clicked.connect(
            lambda: self.summary_frame.setVisible(False)
        )
        summary_layout.addWidget(self.summary_label, 1)
        summary_layout.addWidget(self.open_results_button)
        summary_layout.addWidget(self.open_folder_button)
        summary_layout.addWidget(self.dismiss_button)
        layout.addWidget(self.summary_frame)

        self.progress_model = RunProgressModel(self)
        self.progress_table = QTableView()
        self.progress_table.setModel(self.progress_model)
        self.progress_table.setItemDelegateForColumn(
            RunProgressModel.PROGRESS, ProgressBarDelegate(self.progress_table)
        )
        self.progress_table.setSelectionBehavior(
            QTableView.SelectionBehavior.SelectRows
        )
        self.progress_table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.progress_table.verticalHeader().setVisible(False)
        header = self.progress_table.horizontalHeader()
        header.setSectionResizeMode(
            RunProgressModel.VIDEO, QHeaderView.ResizeMode.ResizeToContents
        )
        header.setSectionResizeMode(
            RunProgressModel.STATUS, QHeaderView.ResizeMode.ResizeToContents
        )
        header.setSectionResizeMode(
            RunProgressModel.PROGRESS, QHeaderView.ResizeMode.Interactive
        )
        header.resizeSection(RunProgressModel.PROGRESS, 140)
        header.setSectionResizeMode(
            RunProgressModel.ETA, QHeaderView.ResizeMode.ResizeToContents
        )
        header.setStretchLastSection(True)

        log_box = QWidget()
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_bar = QHBoxLayout()
        log_bar.addWidget(QLabel("Log:"))
        self.log_level_combo = QComboBox()
        for label, level in ux_helpers.LOG_LEVEL_CHOICES:
            self.log_level_combo.addItem(label, level)
        self.log_level_combo.setToolTip("Show messages of at least this level")
        self.log_level_combo.currentIndexChanged.connect(self._render_log)
        self.log_filter_edit = QLineEdit()
        self.log_filter_edit.setPlaceholderText("Filter the log...")
        self.log_filter_edit.setClearButtonEnabled(True)
        self.log_filter_edit.textChanged.connect(self._render_log)
        self.save_log_button = QPushButton("Save log...")
        self.save_log_button.clicked.connect(self.save_log)
        log_bar.addWidget(self.log_level_combo)
        log_bar.addWidget(self.log_filter_edit, 1)
        log_bar.addWidget(self.save_log_button)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(MAX_LOG_ENTRIES)
        log_layout.addLayout(log_bar)
        log_layout.addWidget(self.log_view, 1)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.progress_table)
        splitter.addWidget(log_box)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

    # ------------------------------------------------------------------
    # Workspace hooks
    # ------------------------------------------------------------------
    def attach_workspace(self, window) -> None:
        window.run_requested.connect(self.start_run)
        window.cancel_requested.connect(self.request_cancel)
        window.register_close_guard(self._close_guard)
        super().attach_workspace(window)

    @property
    def project(self) -> Optional[Project]:
        return None if self.workspace is None else self.workspace.project

    def on_project_changed(self, project: Optional[Project]) -> None:
        self._explicit_output = None
        if project is not None and project.file_path is not None:
            settings = self.workspace.app_settings
            self._explicit_output = settings.run_output_dir(project.file_path)
        if not self.is_running:
            self.summary_frame.setVisible(False)
            self.progress_model.reset_rows([])
            self._last_run_dir = None
            self._reset_progress()
        self._refresh()

    def on_project_modified(self) -> None:
        self._refresh(rescan=False)

    def on_selection_changed(self, keys: List[str]) -> None:
        self._refresh_scope_label()

    def on_tab_activated(self) -> None:
        self._refresh()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.isRunning()

    @property
    def run_dir(self) -> Optional[Path]:
        """The run folder of the last or current run."""
        return self._last_run_dir

    def scope(self) -> str:
        return self.scope_combo.currentData() or SCOPE_ALL

    def set_scope(self, scope: str) -> None:
        index = self.scope_combo.findData(scope)
        if index >= 0:
            self.scope_combo.setCurrentIndex(index)

    def validation_override(self) -> Optional[bool]:
        """None = each video's own setting; True / False = force for this run."""
        state = self.validation_check.checkState()
        if state == Qt.CheckState.Checked:
            return True
        if state == Qt.CheckState.Unchecked:
            return False
        return None

    def set_validation_override(self, value: Optional[bool]) -> None:
        self.validation_check.setCheckState(
            Qt.CheckState.PartiallyChecked if value is None
            else Qt.CheckState.Checked if value else Qt.CheckState.Unchecked
        )

    def default_output_dir(self) -> Optional[Path]:
        project = self.project
        if project is not None and project.base_dir is not None:
            return project.base_dir / "runs"
        if self.workspace is not None:
            return self.workspace.app_settings.output_dir()
        return None

    def output_dir(self) -> Optional[Path]:
        """Where run folders are created: the chosen folder, else ``runs/``."""
        return self._explicit_output or self.default_output_dir()

    def set_output_dir(self, folder: Optional[Path]) -> None:
        project = self.project
        self._explicit_output = Path(folder) if folder is not None else None
        if project is not None and project.file_path is not None:
            settings = self.workspace.app_settings
            settings.set_run_output_dir(project.file_path, folder)
            settings.sync()
        self._refresh()

    def choose_output_dir(self) -> None:
        if self.is_running or self.workspace is None:
            return
        start = self.output_dir()
        chosen = self.workspace.prompts.choose_folder(
            "Choose the output folder",
            str(start if start is not None else self.workspace.app_settings
                .dialog_dir("output")),
        )
        if chosen is not None:
            self.set_output_dir(chosen)

    def _configured_count(self, keys: List[str]) -> int:
        project = self.project
        return sum(
            1 for entry in project.entries_for(keys) if entry.is_configured
        ) if project is not None else 0

    def _scope_keys(self) -> List[str]:
        project = self.project
        if project is None:
            return []
        return scope_keys(
            project, self.scope(), self.workspace.selected_keys()
        )

    def _refresh_scope_label(self) -> None:
        project = self.project
        if project is None:
            self.scope_label.setText("No project open")
            return
        keys = self._scope_keys()
        configured = self._configured_count(keys)
        text = f"{len(keys)} video(s)"
        if configured != len(keys):
            text += f", {configured} configured"
        if self.scope() == SCOPE_SELECTED and not keys:
            text = "select videos in the table"
        self.scope_label.setText(text)

    def _refresh_sheet_label(self) -> None:
        project = self.project
        if project is None:
            self.sheet_label.setText("")
            return
        columns = project.metadata_columns()
        sheet = project.run_options.sample_sheet
        if not columns:
            self.sheet_label.setText("No metadata (no sample sheet loaded)")
            return
        filled = sum(
            1 for video in project.videos
            if any(video.metadata.get(c) for c in columns)
        )
        source = f"{sheet.name}: " if sheet is not None else ""
        text = (
            f"{source}{', '.join(columns)} "
            f"({filled} of {len(project.videos)} videos have values)"
        )
        if sheet is not None and not Path(sheet).is_file():
            text += " - the sample sheet file was not found"
        self.sheet_label.setText(text)
        self.sheet_label.setToolTip(str(sheet) if sheet is not None else "")

    def _refresh_validation_label(self) -> None:
        override = self.validation_override()
        self.validation_label.setText(
            "each video keeps its own setting" if override is None
            else "created for every video of this run" if override
            else "not created for any video of this run"
        )

    def _load_sample_sheet(self) -> None:
        if self.workspace is None or self.is_running:
            return
        self.workspace.video_panel.load_sample_sheet()

    def _refresh(self, rescan: bool = True) -> None:
        """Bring the header, buttons and window state up to date."""
        project = self.project
        self._loading = True
        try:
            options = project.run_options if project is not None else None
            self.workers_spin.setValue(
                options.parallel_workers if options is not None else 0
            )
            self.bins_spin.setValue(
                options.time_bin_seconds if options is not None else 0.0
            )
        finally:
            self._loading = False
        output = self.output_dir()
        self.output_edit.setText(str(output) if output is not None else "")
        self._refresh_scope_label()
        self._refresh_sheet_label()
        self._refresh_validation_label()
        if rescan:
            self._refresh_resume()
        self._update_controls()

    def _update_controls(self) -> None:
        project = self.project
        running = self.is_running
        busy = running or self._checking
        has_work = project is not None and any(
            entry.is_configured for entry in project.videos
        )
        can_run = has_work and not busy
        self.run_button.setEnabled(can_run)
        self.cancel_button.setEnabled(running and not self._close_requested)
        for widget in (self.scope_combo, self.workers_spin, self.bins_spin,
                       self.validation_check, self.output_button,
                       self.sheet_button):
            widget.setEnabled(not busy and project is not None)
        self.resume_button.setEnabled(not busy and bool(self._resume))
        if self.workspace is not None:
            self.workspace.set_run_enabled(has_work)

    # -- header edits ----------------------------------------------------
    def _on_scope_changed(self, _index: int = 0) -> None:
        self._refresh_scope_label()

    def _on_workers_changed(self, value: int) -> None:
        project = self.project
        if self._loading or project is None:
            return
        project.set_run_options(parallel_workers=value)

    def _on_bins_changed(self, value: float) -> None:
        project = self.project
        if self._loading or project is None:
            return
        project.set_run_options(time_bin_seconds=value)

    def _on_validation_changed(self, _state: int = 0) -> None:
        self._refresh_validation_label()
        self._refresh_resume()
        self._update_controls()

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    def _refresh_resume(self) -> None:
        project = self.project
        offer = ResumeOffer()
        if project is not None and not self.is_running:
            offer = find_resumable(
                project, self.output_dir(), self.validation_override()
            )
        self._resume = offer
        self.resume_button.setVisible(bool(offer))
        self.resume_label.setVisible(bool(offer) or bool(offer.changed))
        if offer:
            text = (
                f"{len(offer.resumable)} interrupted video(s) can continue "
                "from their checkpoint."
            )
            if offer.changed:
                text += (
                    f" {len(offer.changed)} more changed settings and would "
                    "start over."
                )
            self.resume_label.setText(text)
        elif offer.changed:
            self.resume_label.setText(
                f"{len(offer.changed)} interrupted video(s) have changed "
                "settings; running them starts over."
            )

    @property
    def resume_offer(self) -> ResumeOffer:
        return self._resume

    def start_resume(self) -> bool:
        """Run the videos of an interrupted run again, from their checkpoints."""
        if self.is_running or not self._resume:
            return False
        return self._start(list(self._resume.keys), strict=False)

    # ------------------------------------------------------------------
    # Starting
    # ------------------------------------------------------------------
    def start_run(self) -> bool:
        """Check the plan for the chosen scope and start the analysis."""
        project = self.project
        if project is None or self.is_running or self._checking:
            return False
        keys = self._scope_keys()
        if self.scope() == SCOPE_SELECTED and not keys:
            self.workspace.prompts.info(
                "Nothing selected", "Select the videos to analyse in the table."
            )
            return False
        return self._start(keys, strict=self.scope() == SCOPE_SELECTED)

    def prepare(self, keys: List[str], strict: bool) -> RunPreparation:
        """Build and check the plan (pre-flight included)."""
        project = self.project
        output = self.output_dir()
        self._checking = True
        self._update_controls()

        def on_progress(done: int, total: int, name: str) -> None:
            self.status_label.setText(
                f"Checking {name}..." if name else "Finishing checks..."
            )
            self.progress_bar.setMaximum(max(total, 1))
            self.progress_bar.setValue(done)
            QApplication.processEvents()

        try:
            self.status_label.setText("Checking the videos...")
            return prepare_run(
                project, keys, output, strict=strict,
                create_validation_video=self.validation_override(),
                progress=on_progress,
            )
        finally:
            self._checking = False
            self._reset_progress()
            self._update_controls()

    def _start(self, keys: List[str], strict: bool) -> bool:
        prompts = self.workspace.prompts
        output = self.output_dir()
        if output is None:
            chosen = prompts.choose_folder(
                "Choose the output folder",
                self.workspace.app_settings.dialog_dir("output"),
            )
            if chosen is None:
                return False
            self.set_output_dir(chosen)
            output = chosen
        report = self.prepare(keys, strict)
        self.last_report = report
        if report.issues:
            prompts.error(
                "Cannot start the analysis", format_problems(report.issues)
            )
            return False
        if not report.plan:
            prompts.info("Nothing to analyse", "No video of this scope is ready.")
            return False
        if report.warnings:
            count = len(report.plan)
            noun = "video" if count == 1 else "videos"
            if not prompts.confirm(
                "Ready with warnings",
                f"Ready to analyse {count} {noun} ({report.maze.name}).\n\n"
                + format_problems(report.warnings),
                accept="Start analysis",
                informative=f"Results go into a new folder inside {output}",
            ):
                return False
        return self._launch(report, output)

    def _worker_count(self, video_count: int) -> int:
        chosen = self.workers_spin.value()
        if chosen <= 0:
            return default_worker_count(video_count)
        return max(1, min(chosen, video_count))

    def _launch(self, report: RunPreparation, output: Path) -> bool:
        project = self.project
        prompts = self.workspace.prompts
        try:
            Path(output).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            prompts.error(
                "Cannot start the analysis",
                f"Cannot create the output folder:\n{exc}",
            )
            return False
        workers = self._worker_count(len(report.plan))
        thread = QThread()
        try:
            worker = AnalysisWorker(
                report.plan, Path(output), report.maze,
                workers=workers,
                sample_metadata=report.sample_metadata,
                bin_seconds=project.run_options.time_bin_seconds,
            )
        except Exception as exc:
            prompts.error(
                "Cannot start the analysis",
                f"Could not create the run output folder:\n{exc}",
            )
            return False
        self._sequential = workers <= 1 or len(report.plan) <= 1
        self._base_paths = {
            f"Video: {name}": path
            for path, name in worker.output_base_names.items()
        }
        self._current_path = None
        self._last_run_dir = worker.output_dir
        self._last_outcome = None
        self._worker_eta = None
        self._run_started = time.monotonic()
        self._close_requested = False
        self._force_close = False
        self._log_entries.clear()
        self.log_view.clear()
        self.summary_frame.setVisible(False)
        self.progress_model.reset_rows(list(report.plan))
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting the analysis...")
        self.eta_label.setText("Batch ETA: calculating...")
        self.eta_label.setVisible(True)

        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_finished)
        worker.detailed_progress.connect(self._on_detailed_progress)
        worker.log.connect(self._on_worker_log)
        worker.video_status.connect(self._on_video_status)
        # Qt frees both objects once the thread has stopped; _on_finished
        # drops our references first so nothing touches them afterwards.
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        _live_threads.add(thread)
        thread.finished.connect(lambda t=thread: _live_threads.discard(t))
        self._thread, self._worker = thread, worker

        self.workspace.set_busy(
            True, f"Analysing {len(report.plan)} video(s)..."
        )
        thread.start()
        self._update_controls()
        self.run_started.emit()
        return True

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------
    def _reset_progress(self) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")
        self.eta_label.setVisible(False)

    def _on_video_status(self, video: str, status: str, message: str) -> None:
        path = Path(video)
        self.progress_model.set_status(path, status, message)
        if status == STATUS_RUNNING:
            self._current_path = path
            row = self.progress_model.row_of(path)
            if row >= 0:
                self.progress_table.scrollTo(
                    self.progress_model.index(row, 0)
                )

    def _on_detailed_progress(self, value: int, main: str, sub: str) -> None:
        self.progress_bar.setValue(max(0, min(100, int(value))))
        self.status_label.setText(main)
        path = self._base_paths.get(main) or self._current_path
        if path is not None:
            self._update_video(path, int(value), sub)
        self._update_eta(int(value), sub)

    def _update_video(self, path: Path, batch_pct: int, sub: str) -> None:
        model = self.progress_model
        total = max(len(model.paths()), 1)
        pct: Optional[int] = None
        if self._sequential:
            row = model.row_of(path)
            if row >= 0:
                pct = int(max(0, min(100, batch_pct * total - row * 100)))
        match = _VIDEO_ETA.search(sub or "")
        eta = match.group(1).strip() if match else ""
        if eta.lower().startswith("calculating"):
            eta = ""
        message = _VIDEO_ETA.sub("", sub or "").strip(" |")
        model.set_progress(path, pct, eta, message or (sub or ""))

    def _update_eta(self, percent: int, sub_text: str = "") -> None:
        """The worker's own batch ETA, else one from the elapsed time."""
        reported = ux_helpers.extract_batch_eta(sub_text)
        if reported:
            self._worker_eta = reported
        elapsed = time.monotonic() - self._run_started
        eta_text = self._worker_eta
        if eta_text is None:
            remaining = ux_helpers.estimate_batch_eta(elapsed, percent)
            if remaining is not None:
                eta_text = "~" + ux_helpers.format_duration(remaining)
        self.eta_label.setText(
            f"Batch ETA: {eta_text or 'calculating...'}   "
            f"(elapsed {ux_helpers.format_duration(elapsed)})"
        )

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------
    def _on_worker_log(self, text: str) -> None:
        level = ux_helpers.level_for_worker_message(text)
        self._log_entries.append((level, text))
        if len(self._log_entries) > MAX_LOG_ENTRIES:
            del self._log_entries[: len(self._log_entries) - MAX_LOG_ENTRIES]
        if self._log_matches(level, text):
            self.log_view.appendPlainText(text)

    def _log_matches(self, level: int, text: str) -> bool:
        minimum = self.log_level_combo.currentData() or 0
        needle = self.log_filter_edit.text().strip().lower()
        return level >= minimum and (not needle or needle in text.lower())

    def _render_log(self, *_args) -> None:
        self.log_view.setPlainText("\n".join(
            text for level, text in self._log_entries
            if self._log_matches(level, text)
        ))

    def log_text(self) -> str:
        """The log lines currently shown (after the filters)."""
        return self.log_view.toPlainText()

    def save_log(self) -> bool:
        """Write the complete (unfiltered) log to a text file."""
        if self.workspace is None or not self._log_entries:
            return False
        start = self.output_dir() or Path(".")
        path = self.workspace.prompts.save_file(
            "Save log", str(Path(start) / "analysis_log.txt"),
            "Text Files (*.txt);;All Files (*.*)",
        )
        if path is None:
            return False
        try:
            Path(path).write_text(
                "\n".join(text for _, text in self._log_entries),
                encoding="utf-8",
            )
        except OSError as exc:
            self.workspace.prompts.error(
                "Save log", f"Could not save the log:\n{exc}"
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------
    def request_cancel(self) -> bool:
        """Ask for confirmation, then stop the run (progress is saved)."""
        if not self.is_running or self._close_requested:
            return False
        if not self.workspace.prompts.confirm(
            "Stop the analysis?",
            "Stop the running analysis?",
            accept="Stop analysis",
            informative=(
                "Progress is saved. The video in progress can be resumed "
                "from its checkpoint later."
            ),
            destructive=True,
        ):
            return False
        self._cancel_worker()
        return True

    def _cancel_worker(self) -> None:
        worker = self._worker
        if worker is None:
            return
        worker.stop()
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Cancelling...")

    # ------------------------------------------------------------------
    # Finishing
    # ------------------------------------------------------------------
    def _on_finished(self, outcome: BatchRunOutcome) -> None:
        thread = self._thread
        # Drop our references before the thread's finished signal lets Qt
        # delete the worker and thread.
        self._thread = None
        self._worker = None
        if thread is not None:
            thread.quit()
            thread.wait(5000)
        closing = self._close_requested
        self._force_timer.stop()
        self._last_outcome = outcome
        run_dir = outcome.output_dir or self._last_run_dir
        self._last_run_dir = run_dir

        for video, (status, message) in outcome_statuses(
            (path for path, _ in outcome.results),
            outcome.failures, outcome.quality_warnings,
        ).items():
            self.progress_model.set_status(video, status, message)
        problem = self._record_run(run_dir)

        self.eta_label.setVisible(False)
        if outcome.cancelled:
            self.status_label.setText("Cancelled - checkpoint saved")
        else:
            self.progress_bar.setValue(100)
            self.status_label.setText(
                "Finished with errors" if outcome.failures
                else "Finished - quality review required"
                if outcome.quality_warnings else "Finished"
            )
        self._show_summary(outcome, run_dir, problem)
        self.workspace.set_busy(False)
        self._refresh()
        self.run_finished.emit(outcome)
        if closing:
            self._close_timer.start()

    def _record_run(self, run_dir: Optional[Path]) -> str:
        """Add the run to the project; returns a problem text ('' when fine)."""
        project = self.project
        if project is None or run_dir is None:
            return ""
        try:
            project.record_run(run_dir)
        except (ProjectError, OSError, ValueError) as exc:
            logger.warning("Could not record the run", exc_info=True)
            return f"The run could not be added to the project: {exc}"
        return ""

    def _show_summary(
        self, outcome: BatchRunOutcome, run_dir: Optional[Path], problem: str
    ) -> None:
        done = len(outcome.results)
        lines: List[str]
        if outcome.cancelled:
            lines = [
                f"Analysis cancelled. Completed: {done} video(s).",
                "The interrupted video can be resumed from its checkpoint.",
            ]
        elif outcome.failures:
            lines = [
                f"Completed with errors. Completed: {done} video(s), "
                f"failed: {len(outcome.failures)}."
            ]
            lines += [
                f"- {path.name}: {message}"
                for path, message in outcome.failures[:5]
            ]
        elif outcome.quality_warnings:
            lines = [
                f"Completed: {done} video(s), with quality warnings. Review "
                "the per-frame evidence and the validation video before "
                "interpreting the results."
            ]
            lines += [
                f"- {path.name}: {message}"
                for path, message in outcome.quality_warnings[:5]
            ]
        else:
            lines = [f"Analysis finished. Processed: {done} video(s)."]
        if run_dir is not None:
            lines.append(f"Saved to: {run_dir}")
        if problem:
            lines.append(problem)
        self.summary_label.setText("\n".join(lines))
        has_run = run_dir is not None and Path(run_dir).is_dir()
        self.open_folder_button.setEnabled(has_run)
        self.open_results_button.setEnabled(
            has_run and (Path(run_dir) / "manifest.json").is_file()
        )
        self.summary_frame.setVisible(True)

    def open_results(self) -> bool:
        """Show the last run in the Results tab."""
        run_dir = self._last_run_dir
        window = self.workspace
        if run_dir is None or window is None:
            return False
        results = getattr(window, "results_tab", None)
        if results is not None and hasattr(results, "show_run"):
            results.show_run(run_dir)
        window.show_tab("results")
        return True

    def open_run_folder(self) -> bool:
        run_dir = self._last_run_dir
        if run_dir is None or self.workspace is None:
            return False
        return bool(self.workspace.prompts.open_folder(run_dir))

    # ------------------------------------------------------------------
    # Closing the window during a run
    # ------------------------------------------------------------------
    def _close_guard(self) -> bool:
        """Veto closing while a run is in progress: cancel first, close later."""
        if not self.is_running:
            return True
        if self._force_close:
            # Last resort after the worker failed to stop in time.
            self._stop_thread_forcefully()
            return True
        if self._close_requested:
            return False        # already stopping; the window closes by itself
        if not self.workspace.prompts.confirm(
            "Analysis in progress",
            "Quit and cancel the analysis?",
            accept="Cancel analysis and quit",
            informative=(
                "The analysis stops and its progress is saved; the window "
                "closes when it has stopped."
            ),
            destructive=True,
        ):
            return False
        # Never block the GUI thread waiting for the worker: ask it to stop,
        # keep the window open, and close from _on_finished.
        self._close_requested = True
        self._cancel_worker()
        self.status_label.setText("Stopping the analysis...")
        self.summary_label.setText(
            "Saving progress; the window closes automatically when done."
        )
        self.summary_frame.setVisible(True)
        self._force_timer.start(self.FORCE_QUIT_DELAY_MS)
        self._update_controls()
        return False

    def _close_window(self) -> None:
        if self.workspace is not None:
            self.workspace.close()

    def _offer_force_quit(self) -> None:
        """The worker did not stop in time; let the user stop waiting."""
        if not (self._close_requested and self.is_running):
            return
        if self.workspace.prompts.confirm(
            "Analysis still stopping",
            "The analysis has not stopped yet.\n\nForce quit now? The "
            "checkpoint of the video in progress may be stale, so the next "
            "run may redo some of its frames, and any video or plot being "
            "written will be incomplete.",
            accept="Force quit",
            destructive=True,
        ):
            self._force_close = True
            self.workspace.close()
        else:
            self._force_timer.start(self.FORCE_QUIT_DELAY_MS)

    def _stop_thread_forcefully(self) -> None:
        thread = self._thread
        if thread is None:
            return
        thread.quit()
        if not thread.wait(2000):
            logger.warning("Analysis thread did not stop; terminating it")
            thread.terminate()
            thread.wait(2000)


__all__ = ["RunTab", "RunProgressModel", "format_problems"]
