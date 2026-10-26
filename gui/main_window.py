"""
gui/main_window.py
Main GUI Window.
Updated to use unified Core modules (VideoHandler, Models).
"""
import copy
import html
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import QObject, QThread, QTimer, QUrl, Qt, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                             QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                             QProgressBar, QProgressDialog, QPushButton, QSpinBox,
                             QTextEdit, QVBoxLayout, QWidget, QInputDialog)
from PyQt6.QtGui import QAction, QDesktopServices, QKeySequence

# --- CORE IMPORTS ---
from core.batch_runner import default_worker_count
from core.batch_plan_keys import (PlanMatch, build_plan_by_key,
                                  match_videos_to_plan)
from core.calibration import rescale_for_new_reference
from core.models import AppConfig, BatchSettings
from core.video import VideoHandler, ReferenceFrameGenerator
from core.version import APP_NAME, APP_VERSION

# --- DIALOGS ---
from gui import ux_helpers
from gui.app_settings import AppSettings
from gui.run_options_panel import RunOptionsPanel
from gui.config_checks import dimension_mismatches, format_dimension_mismatches
from gui.template_alignment import TemplateAlignment
from gui.roi_selector import (REFERENCE_LINE_PROMPT, RoiSelectorDialog,
                              offer_reference_line)
from gui.arm_entry_dialog import ArmEntryParametersDialog
from gui.settings_dialog import SettingsDialog
from gui.setup_progress import (KIND_ARMS, KIND_FREESTYLE, KIND_TST,
                                STEP_ARM_RULES, STEP_DETECTION,
                                STEP_MICE, STEP_ROI, STEP_TST_TUNER,
                                STEP_ZONES, SetupProgress)
from gui.tuner_dialog import TunerDialog
from gui.ux_helpers import Choice
from gui.tst_tuner import TstTunerDialog
from gui.background_task import TaskCancelled, run_with_progress
from gui.run_resume import (RUN_PLAN_FILENAME, ResumableRun, ResumeScan,
                            describe_scan, scan_for_resumable_runs)
from gui.video_list import (STATUS_QUEUED, STATUS_RUNNING, VideoListWidget,
                            clear_list_choices, collect_video_files,
                            move_selected, outcome_statuses, remove_at)
from gui.worker import AnalysisWorker, BatchRunOutcome
from gui.roi_preview import ROIPreviewDialog
from gui.freestyle_config_dialog import FreestyleConfigDialog

# --- UTILS ---
from roi_manager import ROIManager
from mazes import AVAILABLE_MAZES, create_maze, infer_maze_from_settings
from mazes.base_maze import Maze
from preflight_checker import PreflightChecker, ValidationDialog

logger = logging.getLogger(__name__)

# Returned by _offer_resume_setup when the user backs out of starting at all.
_CANCEL = object()


@contextmanager
def _disposed(dialog):
    """Yield a modal dialog and schedule its deletion on leaving the block.

    Dialogs are created with ``parent=self`` and run via ``exec()``; without
    ``deleteLater()`` every one of them (and what it holds) stays alive as a
    child of the main window until the window itself closes.
    """
    try:
        yield dialog
    finally:
        dialog.deleteLater()


class QTextEditLogger(logging.Handler, QObject):
    """Logging handler that fills a QTextEdit and can filter it by level.

    Every record is kept (up to MAX_RECORDS) so the view can be re-filtered
    and the whole log saved, whatever level is currently shown.
    """
    appendRecord = pyqtSignal(int, str)
    MAX_RECORDS = 5000
    _COLORS = {logging.ERROR: "#d1242f", logging.WARNING: "#9a6700"}

    def __init__(self, parent):
        super().__init__()
        QObject.__init__(self)
        self.widget = QTextEdit(parent)
        self.widget.setReadOnly(True)
        self.records: List[Tuple[int, str]] = []
        self.min_level = logging.INFO
        self.appendRecord.connect(self._append)

    def emit(self, record):
        self.appendRecord.emit(record.levelno, self.format(record))

    def _html(self, levelno: int, text: str) -> str:
        color = self._COLORS.get(
            logging.ERROR if levelno >= logging.ERROR
            else logging.WARNING if levelno >= logging.WARNING else 0
        )
        body = html.escape(text).replace("\n", "<br>")
        # Always wrapped in a tag so QTextEdit.append() reads it as HTML.
        return f'<span style="color:{color}">{body}</span>' if color else f"<span>{body}</span>"

    def _append(self, levelno: int, text: str):
        self.records.append((levelno, text))
        if len(self.records) > self.MAX_RECORDS:
            del self.records[: len(self.records) - self.MAX_RECORDS]
        if levelno >= self.min_level:
            self.widget.append(self._html(levelno, text))

    def set_min_level(self, level: int):
        """Show only records at ``level`` or above."""
        self.min_level = level
        self.widget.clear()
        for levelno, text in self.records:
            if levelno >= level:
                self.widget.append(self._html(levelno, text))

    def full_text(self) -> str:
        """The complete log, regardless of the filter."""
        return "\n".join(text for _, text in self.records) + "\n"


class MainWindow(QMainWindow):
    """Main window: video list, setup workflow, run control and log."""

    # How long a close request waits for the worker to stop before offering
    # to force quit.
    FORCE_QUIT_DELAY_MS = 10_000

    def __init__(self, app_settings: Optional[AppSettings] = None):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setGeometry(100, 100, 1000, 750)
        # What the app remembers between sessions (folders, geometry, ...).
        self.app_settings = app_settings or AppSettings()
        
        # State variables
        self.video_paths: List[Path] = []
        self.output_dir: Optional[Path] = None
        # True once the user picked the output folder; otherwise it follows
        # the first video and is not remembered.
        self._output_dir_explicit = False
        self.analysis_thread: Optional[QThread] = None
        self.analysis_worker: Optional[AnalysisWorker] = None
        self.maze: Optional[Maze] = None
        
        # Holds the template (one configuration for all videos) if loaded/configured
        self.current_settings: Optional[BatchSettings] = None
        
        # Holds specific settings per video if loaded from a Batch Plan
        # Keyed by plan key (see core.batch_plan_keys); path is the plan file.
        self.batch_plan_cache: Dict[str, BatchSettings] = {}
        self.batch_plan_path: Optional[Path] = None

        # Carries the template's ROIs onto videos whose camera moved or
        # whose size differs (see gui.template_alignment).
        self._alignment = TemplateAlignment(self)
        
        # {video: (status, message)} shown as the icon of each list row.
        self._video_status: Dict[Path, Tuple[str, str]] = {}
        # Folder of the most recent run, for "Open last run folder".
        self._last_run_dir: Optional[Path] = None

        # What the setup wizard collected before it was cancelled; offered
        # again the next time Start is pressed (see setup_progress).
        self._pending_setup: Optional[SetupProgress] = None

        # Interrupted runs found in the output folder (see run_resume).
        self._resume_scan = ResumeScan()

        # Close-while-running handshake (see closeEvent)
        self._close_requested = False
        self._force_close = False

        self.setAcceptDrops(True)
        self.init_ui()
        self.create_menu_bar()
        self._restore_session()
        self.check_for_resumable_analyses()

    def _restore_session(self):
        """Re-apply the remembered window layout and output folder."""
        geometry = self.app_settings.window_geometry()
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = self.app_settings.window_state()
        if state is not None:
            self.restoreState(state)
        remembered = self.app_settings.output_dir()
        if remembered is not None:
            self._set_output_dir(remembered, explicit=True, persist=False)

    def _save_session(self):
        # A window that was never shown (e.g. --startup-check) has no layout
        # worth remembering and must not overwrite the user's saved one.
        if not self.isVisible():
            return
        self.app_settings.save_window(self.saveGeometry(), self.saveState())
        self.app_settings.sync()

    def _set_output_dir(self, folder: Optional[Path], explicit: bool,
                        persist: bool = True):
        """Change the output folder and keep the label (and settings) in step."""
        self.output_dir = folder
        self._output_dir_explicit = explicit and folder is not None
        if persist and self._output_dir_explicit:
            self.app_settings.set_output_dir(folder)
        self._refresh_output_label()

    def _refresh_output_label(self):
        if self.output_dir is None:
            text = "Output folder: same folder as the first video"
        else:
            suffix = "" if self._output_dir_explicit else " (from the first video)"
            text = f"Output folder: {self.output_dir}{suffix}"
        self.output_label.setText(text)
        self.output_label.setToolTip(str(self.output_dir or ""))
        
    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Top button layout
        top_layout = QHBoxLayout()
        self.add_videos_button = QPushButton("Add Videos")
        self.add_folder_button = QPushButton("Add Folder")
        self.recursive_check = QCheckBox("Include subfolders")
        self.recursive_check.setChecked(self.app_settings.add_folder_recursive())
        self.recursive_check.setToolTip(
            "Also add videos from the folders inside the chosen folder "
            "(analysis_run_* result folders are skipped)."
        )
        self.recursive_check.toggled.connect(
            self.app_settings.set_add_folder_recursive
        )
        self.remove_selected_button = QPushButton("Remove Selected")
        self.move_up_button = QPushButton("Move Up")
        self.move_down_button = QPushButton("Move Down")
        self.clear_list_button = QPushButton("Clear List")

        self.add_videos_button.clicked.connect(self.add_videos)
        self.add_folder_button.clicked.connect(self.add_folder)
        self.remove_selected_button.clicked.connect(self.remove_selected_videos)
        self.move_up_button.clicked.connect(lambda: self.move_selected_videos(-1))
        self.move_down_button.clicked.connect(lambda: self.move_selected_videos(1))
        self.clear_list_button.clicked.connect(self.clear_list)

        top_layout.addWidget(self.add_videos_button)
        top_layout.addWidget(self.add_folder_button)
        top_layout.addWidget(self.recursive_check)
        top_layout.addStretch()
        top_layout.addWidget(self.remove_selected_button)
        top_layout.addWidget(self.move_up_button)
        top_layout.addWidget(self.move_down_button)
        top_layout.addWidget(self.clear_list_button)
        main_layout.addLayout(top_layout)

        # Banner offering to resume an interrupted run (hidden when none)
        self.resume_bar = QWidget()
        resume_layout = QHBoxLayout(self.resume_bar)
        resume_layout.setContentsMargins(6, 4, 6, 4)
        self.resume_label = QLabel()
        self.resume_label.setWordWrap(True)
        self.resume_button = QPushButton("Resume interrupted run")
        self.resume_button.clicked.connect(self.resume_interrupted_run)
        resume_layout.addWidget(self.resume_label, 1)
        resume_layout.addWidget(self.resume_button)
        self.resume_bar.setStyleSheet(
            "background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px;"
        )
        self.resume_bar.setVisible(False)
        main_layout.addWidget(self.resume_bar)

        # Video list
        self.video_list_widget = VideoListWidget()
        self.video_list_widget.remove_requested.connect(self.remove_selected_videos)
        self.video_list_widget.move_requested.connect(self.move_selected_videos)
        self.video_list_widget.itemSelectionChanged.connect(self.update_button_states)
        main_layout.addWidget(self.video_list_widget)

        # Output folder
        output_layout = QHBoxLayout()
        self.output_label = QLabel()
        self.output_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.change_output_button = QPushButton("Change…")
        self.change_output_button.clicked.connect(self.set_output_directory)
        self.open_output_button = QPushButton("Open")
        self.open_output_button.setToolTip("Open the output folder in the file manager")
        self.open_output_button.clicked.connect(self.open_output_folder)
        output_layout.addWidget(self.output_label, 1)
        output_layout.addWidget(self.change_output_button)
        output_layout.addWidget(self.open_output_button)
        main_layout.addLayout(output_layout)
        self._refresh_output_label()
        
        # Action buttons layout
        action_layout = QHBoxLayout()
        self.start_button = QPushButton("Configure & Start Analysis")
        self.preview_button = QPushButton("Preview ROIs")
        self.cancel_button = QPushButton("Cancel Analysis")
        
        self.start_button.clicked.connect(self.start_analysis_workflow)
        self.preview_button.clicked.connect(self.preview_rois)
        self.cancel_button.clicked.connect(self.confirm_cancel_analysis)
        
        self.cancel_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        
        self.workers_label = QLabel("Parallel videos:")
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, max(1, os.cpu_count() or 1))
        self.workers_spin.setSpecialValueText("Auto")
        self.workers_spin.setValue(self.app_settings.parallel_workers())
        self.workers_spin.setToolTip(
            "How many videos are analysed at the same time. Auto uses "
            "min(CPU cores / 4, videos, RAM in GB / 3), at least 1; 1 runs "
            "the videos one after another. Results are identical either way."
        )
        self.workers_spin.valueChanged.connect(
            self.app_settings.set_parallel_workers
        )

        action_layout.addWidget(self.start_button)
        action_layout.addWidget(self.preview_button)
        action_layout.addWidget(self.cancel_button)
        action_layout.addWidget(self.workers_label)
        action_layout.addWidget(self.workers_spin)
        main_layout.addLayout(action_layout)

        # Run-level options (sample sheet, time bins): not part of any
        # video's settings, so they never change the checkpoint hash.
        self.run_options = RunOptionsPanel(
            self.app_settings,
            videos_provider=lambda: list(self.video_paths),
            maze_provider=lambda: self.maze,
        )
        main_layout.addWidget(self.run_options)
        
        # Progress indicators
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.status_label = QLabel("Ready. Please add videos to begin.")
        self.status_label.setStyleSheet("padding: 5px; font-weight: bold;")
        
        self.subtask_label = QLabel("")
        self.subtask_label.setStyleSheet("padding: 2px; font-size: 10pt; color: #666;")
        
        # Batch ETA: large and bold because it is what people wait for.
        self.eta_label = QLabel("")
        self.eta_label.setStyleSheet("padding: 2px 5px; font-size: 14pt; font-weight: bold;")
        self.eta_label.setVisible(False)
        self._run_started: Optional[float] = None
        self._worker_eta: Optional[str] = None

        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.eta_label)
        main_layout.addWidget(self.subtask_label)
        
        # Log output
        self.log_handler = QTextEditLogger(self)
        self.log_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        )
        logging.getLogger().addHandler(self.log_handler)

        log_bar = QHBoxLayout()
        log_bar.addWidget(QLabel("Log:"))
        self.log_level_combo = QComboBox()
        for label, level in ux_helpers.LOG_LEVEL_CHOICES:
            self.log_level_combo.addItem(label, level)
        self.log_level_combo.setToolTip("Show messages of this level and above")
        self.log_level_combo.currentIndexChanged.connect(self._on_log_level_changed)
        self.save_log_button = QPushButton("Save log…")
        self.save_log_button.setToolTip("Save the complete log (all levels) to a text file")
        self.save_log_button.clicked.connect(self.save_log)
        log_bar.addWidget(self.log_level_combo)
        log_bar.addStretch()
        log_bar.addWidget(self.save_log_button)
        main_layout.addLayout(log_bar)
        main_layout.addWidget(self.log_handler.widget)

        self.add_videos_button.setToolTip("Add videos (Ctrl+O)")
        self.add_folder_button.setToolTip("Add a folder of videos (Ctrl+Shift+O)")
        self.start_button.setToolTip("Configure and start the analysis (F5)")
        self.cancel_button.setToolTip("Stop the analysis (Esc)")

    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu('&File')
        
        add_videos_action = QAction('Add &Videos...', self)
        add_videos_action.setShortcut(QKeySequence("Ctrl+O"))
        add_videos_action.triggered.connect(self.add_videos)
        self.add_videos_action = add_videos_action

        add_folder_action = QAction('Add &Folder...', self)
        add_folder_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        add_folder_action.triggered.connect(self.add_folder)
        self.add_folder_action = add_folder_action

        save_config_action = QAction('&Save Template...', self)
        save_config_action.setShortcut(QKeySequence("Ctrl+S"))
        save_config_action.triggered.connect(self.save_configuration)
        save_config_action.setEnabled(False)
        self.save_config_action = save_config_action
        
        load_config_action = QAction('&Load Configuration / Plan...', self)
        load_config_action.setShortcut(QKeySequence("Ctrl+L"))
        load_config_action.triggered.connect(self.load_configuration)
        self.load_config_action = load_config_action
        
        load_run_action = QAction('Load settings from &previous run...', self)
        load_run_action.triggered.connect(self.load_settings_from_previous_run)
        self.load_run_action = load_run_action

        resume_action = QAction('Resume &interrupted run...', self)
        resume_action.triggered.connect(self.resume_interrupted_run)
        resume_action.setEnabled(False)
        self.resume_action = resume_action

        open_output_action = QAction('Open &output folder', self)
        open_output_action.triggered.connect(self.open_output_folder)
        self.open_output_action = open_output_action

        open_last_run_action = QAction('Open &last run folder', self)
        open_last_run_action.triggered.connect(self.open_last_run_folder)
        open_last_run_action.setEnabled(False)
        self.open_last_run_action = open_last_run_action

        file_menu.addAction(add_videos_action)
        file_menu.addAction(add_folder_action)
        file_menu.addSeparator()
        file_menu.addAction(save_config_action)
        file_menu.addAction(load_config_action)
        file_menu.addAction(load_run_action)
        file_menu.addAction(resume_action)
        file_menu.addSeparator()
        file_menu.addAction(open_output_action)
        file_menu.addAction(open_last_run_action)
        self.recent_menu = file_menu.addMenu('&Recent')
        self.recent_menu.aboutToShow.connect(self._rebuild_recent_menu)
        self._rebuild_recent_menu()
        file_menu.addSeparator()
        
        exit_action = QAction('E&xit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        run_menu = menubar.addMenu('&Run')
        start_action = QAction('&Configure and Start Analysis', self)
        start_action.setShortcut(QKeySequence("F5"))
        start_action.triggered.connect(self.start_analysis_workflow)
        start_action.setEnabled(False)
        self.start_action = start_action
        cancel_action = QAction('&Cancel Analysis', self)
        cancel_action.setShortcut(QKeySequence("Esc"))
        cancel_action.triggered.connect(self.confirm_cancel_analysis)
        cancel_action.setEnabled(False)
        self.cancel_action = cancel_action
        run_menu.addAction(start_action)
        run_menu.addAction(cancel_action)

        tools_menu = menubar.addMenu('&Tools')
        set_output_action = QAction('Set &Output Directory...', self)
        set_output_action.triggered.connect(self.set_output_directory)
        tools_menu.addAction(set_output_action)
        self.set_output_action = set_output_action

        align_action = QAction('&Align template to videos...', self)
        align_action.setToolTip(
            "Check that the loaded template's regions fit each video and "
            "align them when the camera moved or the resolution differs"
        )
        align_action.triggered.connect(self._alignment.check_all)
        tools_menu.addAction(align_action)
        self.align_template_action = align_action

        validation_menu = menubar.addMenu('&Validation')
        manual_action = QAction('&Score a video manually...', self)
        manual_action.setToolTip(
            "Blind manual scoring with hotkeys, saved as annotation files "
            "for the validation report"
        )
        manual_action.triggered.connect(self.open_manual_scoring)
        validation_menu.addAction(manual_action)
        self.manual_scoring_action = manual_action

        spotcheck_action = QAction('Tracking s&pot-check...', self)
        spotcheck_action.setToolTip(
            "Click the animal in randomly sampled frames to measure the "
            "tracker's position error"
        )
        spotcheck_action.triggered.connect(self.open_position_spotcheck)
        validation_menu.addAction(spotcheck_action)
        self.position_spotcheck_action = spotcheck_action

        report_action = QAction('&Build validation report...', self)
        report_action.setToolTip(
            "Compare automated results with manual scoring: agreement "
            "statistics, plots and acceptance criteria in one HTML report"
        )
        report_action.triggered.connect(self.open_validation_report)
        validation_menu.addAction(report_action)
        self.validation_report_action = report_action

    def open_manual_scoring(self):
        """Validation > Score a video manually (modal; imported on demand)."""
        from gui.manual_scoring_dialog import open_manual_scoring
        open_manual_scoring(self, self.app_settings)

    def open_position_spotcheck(self):
        """Validation > Tracking spot-check (modal; imported on demand)."""
        from gui.position_spotcheck_dialog import open_position_spotcheck
        open_position_spotcheck(self, self.app_settings)

    def open_validation_report(self):
        """Validation > Build validation report (modal; imported on demand)."""
        from gui.validation_report_dialog import open_validation_report
        open_validation_report(self, self.app_settings)

    def _rebuild_recent_menu(self):
        """List the most recently used templates and plans."""
        self.recent_menu.clear()
        recent = self.app_settings.recent_files()
        for path in recent:
            action = self.recent_menu.addAction(path.name)
            action.setToolTip(str(path))
            action.setStatusTip(str(path))
            action.triggered.connect(
                lambda _checked=False, p=path: self.load_configuration_file(p)
            )
        if not recent:
            empty = self.recent_menu.addAction("(none)")
            empty.setEnabled(False)
        else:
            self.recent_menu.addSeparator()
            clear = self.recent_menu.addAction("Clear list")
            clear.triggered.connect(self.app_settings.clear_recent_files)
        self.recent_menu.setEnabled(not self._analysis_running())

    def add_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files",
            self.app_settings.dialog_dir("video"),
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        if files:
            self.app_settings.set_last_video_dir(Path(files[0]).parent)
            added_count = self._add_video_paths([Path(file) for file in files])
            logger.info(f"Added {added_count} video(s)")

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing Videos",
            self.app_settings.dialog_dir("video")
        )
        if not folder: return
        folder_path = Path(folder)
        self.app_settings.set_last_video_dir(folder_path)

        found_videos, _ = collect_video_files(
            [folder_path], self.recursive_check.isChecked()
        )
        if not found_videos:
            hint = "" if self.recursive_check.isChecked() else (
                "\n\nTick 'Include subfolders' to search the folders inside it too."
            )
            QMessageBox.information(
                self, "No Videos Found",
                f"No video files found in:\n{folder_path}{hint}"
            )
            return

        answer = ux_helpers.ask_choice(
            self, "Add Folder",
            f"Found {len(found_videos)} video(s) in:\n{folder_path}",
            [
                Choice("add", f"Add {len(found_videos)} video(s)"),
                Choice("cancel", "Cancel", QMessageBox.ButtonRole.RejectRole),
            ],
            default_key="add",
        )
        if answer == "add":
            added_count = self._add_video_paths(found_videos)
            logger.info(f"Added {added_count} video(s) from folder: {folder_path.name}")

    def _add_video_paths(self, paths: List[Path]) -> int:
        """Append new videos to the list; returns how many were new."""
        output_was_unset = self.output_dir is None
        added_count = 0
        for path in paths:
            if path not in self.video_paths:
                self.video_paths.append(path)
                added_count += 1
        self._refresh_video_list()
        if self.output_dir is None and self.video_paths:
            self._set_output_dir(self.video_paths[0].parent, explicit=False)
        if output_was_unset and self.output_dir is not None:
            self.check_for_resumable_analyses()
        self.update_button_states()
        return added_count

    def _refresh_video_list(self, selected=()):
        """Rebuild the rows from ``video_paths`` with each video's status."""
        self.video_list_widget.set_videos(
            self.video_paths, self._video_status, selected
        )

    # --- drag and drop ---------------------------------------------------------
    def dragEnterEvent(self, event):
        if not self._analysis_running() and event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if self._analysis_running():
            event.ignore()
            return
        dropped = [
            Path(url.toLocalFile()) for url in event.mimeData().urls()
            if url.isLocalFile()
        ]
        event.acceptProposedAction()
        self.add_dropped_paths(dropped)

    def add_dropped_paths(self, dropped: List[Path]):
        """Handle files and folders dropped on the window."""
        configs = [p for p in dropped if p.suffix.lower() == ".json" and p.is_file()]
        videos, ignored = collect_video_files(
            [p for p in dropped if p not in configs],
            self.recursive_check.isChecked()
        )
        if videos:
            self.app_settings.set_last_video_dir(
                videos[0].parent if videos[0].parent.is_dir() else None
            )
            added = self._add_video_paths(videos)
            logger.info(f"Added {added} dropped video(s)")
        if len(configs) == 1 and not videos:
            self.load_configuration_file(configs[0])
        elif configs:
            ignored.extend(configs)
        if ignored:
            names = ", ".join(path.name for path in ignored[:5])
            more = f" and {len(ignored) - 5} more" if len(ignored) > 5 else ""
            logger.warning(f"Ignored dropped item(s): {names}{more}")
            if not videos:
                QMessageBox.information(
                    self, "Nothing To Add",
                    "Drop video files (.mp4, .avi, .mov, .mkv), a folder of "
                    "videos, or one saved configuration (.json).\n\n"
                    f"Ignored: {names}{more}"
                )

    # --- editing the list ---------------------------------------------------------
    def remove_selected_videos(self):
        if self._analysis_running():
            return
        rows = self.video_list_widget.selected_rows()
        if not rows:
            return
        for row in rows:
            self._video_status.pop(self.video_paths[row], None)
        self.video_paths = remove_at(self.video_paths, rows)
        self._refresh_video_list([min(rows[0], len(self.video_paths) - 1)])
        self._reset_auto_output_dir()
        self.update_button_states()
        logger.info(f"Removed {len(rows)} video(s)")

    def move_selected_videos(self, offset: int):
        if self._analysis_running():
            return
        rows = self.video_list_widget.selected_rows()
        if not rows:
            return
        self.video_paths, selected = move_selected(self.video_paths, rows, offset)
        self._refresh_video_list(selected)
        self.video_list_widget.setFocus()

    def _reset_auto_output_dir(self):
        """An output folder that only followed the first video follows it still."""
        if self._output_dir_explicit:
            return
        if self.video_paths:
            self._set_output_dir(self.video_paths[0].parent, explicit=False)
        else:
            self._set_output_dir(None, explicit=False)

    def clear_list(self):
        if not self.video_paths or self._analysis_running():
            return
        has_config = (
            self.current_settings is not None
            or bool(self.batch_plan_cache)
            or self._pending_setup is not None
        )
        options = clear_list_choices(has_config)
        choices = [
            Choice(key, label, QMessageBox.ButtonRole.DestructiveRole
                   if key == "all" else QMessageBox.ButtonRole.AcceptRole)
            for key, label in options
        ] + [Choice("cancel", "Cancel", QMessageBox.ButtonRole.RejectRole)]
        answer = ux_helpers.ask_choice(
            self, "Clear List",
            f"Remove all {len(self.video_paths)} video(s) from the list?",
            choices, default_key="cancel",
            informative=(
                "The loaded template, batch plan or unfinished setup can be "
                "kept for the next videos." if has_config else ""
            ),
        )
        if answer not in ("videos", "all"):
            return
        self.video_paths.clear()
        self._video_status.clear()
        self._refresh_video_list()
        if answer == "all":
            self.current_settings = None
            self.batch_plan_cache = {}
            self.batch_plan_path = None
            self._pending_setup = None
        self._reset_auto_output_dir()
        self.update_button_states()
        logger.info("Video list cleared" + (" with configuration" if answer == "all" else ""))

    def update_button_states(self):
        has_videos = len(self.video_paths) > 0
        has_settings = self.current_settings is not None or len(self.batch_plan_cache) > 0
        has_selection = bool(self.video_list_widget.selectedItems())
        idle = not self._analysis_running()

        self.start_button.setEnabled(has_videos and idle)
        self.preview_button.setEnabled(has_videos and has_settings and idle)
        self.save_config_action.setEnabled(self.current_settings is not None and idle)
        self.remove_selected_button.setEnabled(has_selection and idle)
        self.move_up_button.setEnabled(has_selection and idle)
        self.move_down_button.setEnabled(has_selection and idle)
        self.clear_list_button.setEnabled(has_videos and idle)
        run_options = getattr(self, "run_options", None)
        if run_options is not None:
            run_options.refresh()
        self._sync_actions(running=not idle)

    def _sync_actions(self, running: bool):
        """Menu actions (and their shortcuts) follow the buttons' state."""
        self.add_videos_action.setEnabled(not running)
        self.add_folder_action.setEnabled(not running)
        self.start_action.setEnabled(bool(self.video_paths) and not running)
        self.cancel_action.setEnabled(running)

    def preview_rois(self):
        if not self.video_paths: return
        first_vid = self.video_paths[0]
        if self.batch_plan_cache:
            settings_to_use = self._plan_settings_for(first_vid)
            if settings_to_use is None:
                QMessageBox.warning(
                    self, "Cannot Preview",
                    f"'{first_vid.name}' is not in the loaded Batch Plan "
                    "(or its filename is ambiguous)."
                )
                return
        else:
            settings_to_use = self.current_settings
        
        if not settings_to_use:
            QMessageBox.warning(self, "Cannot Preview", "Please configure analysis settings first.")
            return
        
        if self.maze is None:
            self._infer_maze_from_settings(settings_to_use)

        try:
            self._refresh_center_overlay(settings_to_use)
            preview_dialog = ROIPreviewDialog(first_vid, settings_to_use.roi_manager, self.maze, self)
            preview_dialog.exec()
        except Exception as e:
            logger.error(f"Error showing ROI preview: {e}", exc_info=True)
            QMessageBox.critical(self, "Preview Error", f"Failed to show ROI preview:\n{e}")

    def save_configuration(self):
        if not self.current_settings:
            QMessageBox.warning(self, "No Configuration", "Please configure analysis settings first.")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Template",
            str(Path(self.app_settings.dialog_dir("config", self.output_dir))
                / "template_config.json"),
            "JSON Files (*.json);;All Files (*.*)"
        )
        if filename:
            try:
                self.current_settings.save_to_file(Path(filename))
                self.app_settings.set_last_config_dir(Path(filename).parent)
                self.app_settings.add_recent_file(filename)
                logger.info(f"Template saved to: {filename}")
                QMessageBox.information(self, "Saved", f"Template saved to:\n{filename}")
            except Exception as e:
                logger.error(f"Failed to save configuration: {e}", exc_info=True)
                QMessageBox.critical(self, "Save Failed", f"Failed to save:\n{e}")

    def load_configuration(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration",
            self.app_settings.dialog_dir("config", self.output_dir),
            "JSON Files (*.json);;All Files (*.*)"
        )
        if not filename: return
        self.load_configuration_file(Path(filename))

    def load_configuration_file(self, file_path: Path):
        """Load a template or batch plan from a known path."""
        filename = str(file_path)
        try:
            file_type = BatchSettings.detect_file_type(file_path)
            
            if file_type == 'batch_plan':
                plan = BatchSettings.load_batch_plan(file_path)
                self._apply_batch_plan(plan, file_path)
                match = self._match_plan(self.video_paths)
                msg = (f"Loaded Batch Plan with {len(plan)} configurations.\n"
                       f"Matched to {len(match.matched)} of "
                       f"{len(self.video_paths)} videos currently in list.")
                if match.unmatched:
                    msg += (f"\n{len(match.unmatched)} video(s) are not in the "
                            "plan and will need their own configuration.")
            else:
                settings = BatchSettings.load_from_file(file_path)
                self.current_settings = settings
                self.batch_plan_cache = {} 
                self.batch_plan_path = None
                self._infer_maze_from_settings(settings)
                msg = ("Template loaded.\nThis configuration will be applied to ALL videos.")

            self.update_button_states()
            self.app_settings.set_last_config_dir(file_path.parent)
            self.app_settings.add_recent_file(file_path)
            self._rebuild_recent_menu()
            logger.info(f"Configuration loaded from: {filename}")
            mismatch_text = self._dimension_mismatch_text()
            if mismatch_text:
                QMessageBox.warning(
                    self, "Configuration Loaded - Check Video Sizes",
                    f"{msg}\n\n{mismatch_text}"
                )
            else:
                QMessageBox.information(self, "Configuration Loaded", msg)
            
        except Exception as e:
            self.app_settings.remove_recent_file(file_path)
            logger.error(f"Failed to load configuration: {e}", exc_info=True)
            QMessageBox.critical(self, "Load Failed", f"Failed to load configuration:\n{e}")

    def _expected_dimensions_for(self, video: Path):
        """Frame size the loaded ROIs for ``video`` were drawn on, if known."""
        if self.batch_plan_cache:
            settings = self._plan_settings_for(video)
            if settings is None:
                return None  # not covered by the plan: it will be configured anew
        else:
            settings = self.current_settings
        if settings is None or settings.roi_manager is None:
            return None
        return settings.roi_manager.source_dimensions

    def _dimension_mismatch_text(self) -> str:
        """List videos whose size differs from the loaded configuration's ROIs."""
        if not self.video_paths:
            return ""
        try:
            return format_dimension_mismatches(
                dimension_mismatches(self.video_paths, self._expected_dimensions_for)
            )
        except Exception:
            logger.warning("Could not compare video sizes", exc_info=True)
            return ""

    def _apply_batch_plan(self, plan: Dict[str, BatchSettings], plan_path: Path):
        """Make a loaded batch plan the active configuration."""
        self.batch_plan_cache = plan
        self.batch_plan_path = plan_path
        if plan:
            first_key = list(plan.keys())[0]
            self.current_settings = plan[first_key]
            self._infer_maze_from_settings(self.current_settings)

    def load_settings_from_previous_run(self):
        """Pick a run folder and load the plan it was analysed with."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select a Previous Run Folder",
            self.app_settings.dialog_dir("output", self.output_dir)
        )
        if not folder:
            return
        plan_path = Path(folder) / RUN_PLAN_FILENAME
        if not plan_path.is_file():
            QMessageBox.information(
                self, "No Saved Settings",
                f"'{Path(folder).name}' has no {RUN_PLAN_FILENAME}.\n\n"
                "Only runs made with this version or later save their "
                "settings. Use File > Load Configuration for older "
                "templates or plans."
            )
            return
        self.load_configuration_file(plan_path)

    def _match_plan(self, video_paths: List[Path]) -> PlanMatch:
        """Match videos to the loaded Batch Plan (never guessing on ambiguity)."""
        if not self.batch_plan_cache or self.batch_plan_path is None:
            return PlanMatch(
                unmatched=[(v, "no Batch Plan loaded") for v in video_paths]
            )
        return match_videos_to_plan(
            video_paths, self.batch_plan_cache.keys(), self.batch_plan_path
        )

    def _plan_settings_for(self, video_path: Path) -> Optional[BatchSettings]:
        """Return a copy of the Batch Plan entry for one video, if matched."""
        match = self._match_plan([video_path])
        key = match.matched.get(video_path)
        return self.batch_plan_cache[key].copy() if key is not None else None

    def _infer_maze_from_settings(self, settings):
        """Pick and configure the maze a saved configuration was made for."""
        if not settings or not settings.roi_manager: return
        self.maze = infer_maze_from_settings(settings)

    def set_output_directory(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Directory",
            str(self.output_dir) if self.output_dir
            else self.app_settings.dialog_dir("output")
        )
        if folder:
            self._set_output_dir(Path(folder), explicit=True)
            self.check_for_resumable_analyses()
            logger.info(f"Output directory set to: {self.output_dir}")
            QMessageBox.information(self, "Output Directory Set", f"Results will be saved to:\n{self.output_dir}")

    def check_for_resumable_analyses(self):
        """Look for interrupted runs in the output folder and show a banner."""
        scan = ResumeScan()
        if self.output_dir is not None:
            try:
                scan = scan_for_resumable_runs(self.output_dir)
            except Exception:
                logger.warning(
                    "Could not inspect resumable analyses",
                    exc_info=True,
                )
        self._resume_scan = scan
        self._refresh_resume_bar()

    def _refresh_resume_bar(self):
        scan = self._resume_scan
        self.resume_label.setText(describe_scan(scan))
        self.resume_bar.setVisible(bool(scan))
        resumable = bool(scan.runs) and not self._analysis_running()
        self.resume_button.setVisible(bool(scan.runs))
        self.resume_button.setEnabled(resumable)
        self.resume_action.setEnabled(resumable)

    def resume_interrupted_run(self):
        """Reload an interrupted run's plan and settings, then start it again."""
        runs = self._resume_scan.runs
        if not runs or self._analysis_running():
            return
        run = runs[0]
        if len(runs) > 1:
            names = [candidate.name for candidate in runs]
            chosen, ok = QInputDialog.getItem(
                self, "Resume Interrupted Run", "Which run?", names, 0, False
            )
            if not ok:
                return
            run = runs[names.index(chosen)]
        self._resume_run(run)

    def _resume_run(self, run: ResumableRun):
        try:
            plan = BatchSettings.load_batch_plan(run.plan_path)
        except Exception as exc:
            logger.error("Could not load run plan", exc_info=True)
            QMessageBox.critical(
                self, "Cannot Resume",
                f"Could not read the saved settings of '{run.name}':\n{exc}"
            )
            return
        if not run.pending_videos:
            QMessageBox.information(
                self, "Nothing To Resume",
                "None of the videos of that run are available any more."
            )
            return
        if run.paradigm_id not in AVAILABLE_MAZES:
            QMessageBox.critical(
                self, "Cannot Resume",
                f"The run's paradigm ({run.paradigm_id or 'unknown'}) is not available."
            )
            return
        if self.video_paths and set(self.video_paths) != set(run.pending_videos):
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Question)
            box.setWindowTitle("Resume Interrupted Run")
            box.setText(
                f"Resuming '{run.name}' replaces the current video list with "
                f"its {len(run.pending_videos)} unfinished video(s)."
            )
            replace = box.addButton("Replace list", QMessageBox.ButtonRole.AcceptRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            if box.clickedButton() is not replace:
                return

        self._apply_batch_plan(plan, run.plan_path)
        self._replace_video_list(run.pending_videos)
        self.maze = create_maze(run.paradigm_id)
        self.app_settings.set_last_paradigm(run.paradigm_id)
        logger.info("Resuming '%s' (%d video(s))", run.name, len(run.pending_videos))
        self._start_from_batch_plan(self._match_plan(self.video_paths), run.paradigm_id)

    def _replace_video_list(self, paths: List[Path]):
        self.video_paths = []
        self.video_list_widget.clear()
        self._add_video_paths(list(paths))

    def start_analysis_workflow(self):
        if not self.video_paths:
            QMessageBox.warning(self, "No Videos", "Please add one or more videos to the list.")
            return

        # --- MAZE SELECTION ---
        maze_names = list(AVAILABLE_MAZES.keys())
        default_idx = 0
        preferred = self.maze.name if self.maze else self.app_settings.last_paradigm()
        if preferred in maze_names:
            default_idx = maze_names.index(preferred)

        maze_choice, ok = QInputDialog.getItem(self, "Select Maze Type", "Which maze are you analyzing?", maze_names, default_idx, False)
        if not ok or not maze_choice: return

        self.maze = create_maze(maze_choice)
        self.app_settings.set_last_paradigm(maze_choice)
        logger.info(f"Selected maze type: {self.maze.name}")
        ref_video_path = self.video_paths[0]

        # --- CONFIGURATION LOGIC ---
        use_loaded_config = False
        plan_loaded = bool(self.batch_plan_cache)
        plan_match = self._match_plan(self.video_paths)

        try:
            resumed_setup = self._offer_resume_setup(ref_video_path, maze_choice)
            if resumed_setup is _CANCEL:
                return

            if resumed_setup is not None:
                pass  # the wizard continues where it stopped
            elif plan_match.matched:
                answer = ux_helpers.ask_choice(
                    self, "Batch Plan Detected",
                    f"A batch plan is loaded and covers {len(plan_match.matched)} "
                    f"of the {len(self.video_paths)} videos.",
                    [
                        Choice("use", "Use the batch plan"),
                        Choice("scratch", "Set up from scratch",
                               QMessageBox.ButtonRole.ActionRole),
                        Choice("cancel", "Cancel", QMessageBox.ButtonRole.RejectRole),
                    ],
                    default_key="use",
                )
                if answer == "use":
                    self._start_from_batch_plan(plan_match, maze_choice)
                    return
                if answer != "scratch":
                    return
            elif plan_loaded:
                QMessageBox.information(
                    self, "Batch Plan Not Applicable",
                    "None of the videos in the list match the loaded Batch Plan "
                    "(or their filenames are ambiguous).\n"
                    "You will be asked to configure them from scratch."
                )
            elif self.current_settings:
                answer = ux_helpers.ask_choice(
                    self, "Reuse previous configuration?",
                    "A configuration from an earlier setup or a loaded "
                    "template is available.",
                    [
                        Choice("reuse", "Reuse it for all videos"),
                        Choice("again", "Set up again",
                               QMessageBox.ButtonRole.ActionRole),
                        Choice("cancel", "Cancel", QMessageBox.ButtonRole.RejectRole),
                    ],
                    default_key="reuse",
                )
                if answer == "reuse":
                    use_loaded_config = True
                elif answer != "again":
                    return

            if use_loaded_config:
                base_settings = self.current_settings
                self._configure_maze_from_settings(base_settings, maze_choice)
            else:
                base_settings = self._run_setup_wizard(
                    ref_video_path, maze_choice, resumed_setup
                )
            if base_settings is None:
                return

            base_settings = self._review_settings(ref_video_path, base_settings)
            if base_settings is None:
                return

            # --- EXECUTION ---
            self._clear_setup_status("Configuration complete")
            if len(self.video_paths) > 1:
                answer = ux_helpers.ask_choice(
                    self, "Batch Processing",
                    f"Process {len(self.video_paths)} videos.",
                    [
                        Choice("review", "Review each video…"),
                        Choice("run", "Run all now",
                               QMessageBox.ButtonRole.ActionRole),
                        Choice("cancel", "Cancel", QMessageBox.ButtonRole.RejectRole),
                    ],
                    default_key="run",
                    informative=(
                        "Review each video lets you check the ROIs and times "
                        "per video before anything runs. Run all now applies "
                        "this configuration to every video."
                    ),
                )
                if answer == "review":
                    self.run_batch_with_verification(
                        base_settings, reviewed_video=ref_video_path
                    )
                elif answer == "run":
                    self.run_batch_analysis(base_settings)
                else:
                    self._clear_setup_status("Ready")
            else:
                self.run_batch_analysis(base_settings)

        except Exception as e:
            logger.error(f"Error in analysis workflow: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"An error occurred:\n{e}")
            self._clear_setup_status("Error")

    def _offer_resume_setup(self, ref_video_path: Path, maze_choice: str):
        """Offer to continue an unfinished setup.

        Returns the SetupProgress to continue, None to start from scratch, or
        _CANCEL if the user backs out.
        """
        pending = self._pending_setup
        if pending is None or not pending.is_resumable:
            return None
        try:
            dimensions = VideoHandler(ref_video_path).dimensions
        except Exception:
            dimensions = None
        if not pending.applies_to(maze_choice, dimensions):
            return None

        answer = ux_helpers.ask_choice(
            self, "Resume setup?",
            f"An unfinished {maze_choice} setup is still in memory.\n\n"
            f"{pending.describe()}",
            [
                Choice("resume", "Resume setup"),
                Choice("restart", "Start over",
                       QMessageBox.ButtonRole.DestructiveRole),
                Choice("cancel", "Cancel", QMessageBox.ButtonRole.RejectRole),
            ],
            default_key="resume",
        )
        if answer == "resume":
            try:
                if pending.algorithm_parameters:
                    self.maze.configure_algorithm_parameters(
                        copy.deepcopy(pending.algorithm_parameters)
                    )
            except (TypeError, ValueError) as exc:
                logger.warning("Could not restore setup parameters: %s", exc)
                self._pending_setup = None
                return None
            return pending
        if answer == "restart":
            self._pending_setup = None
            return None
        return _CANCEL

    def _wizard_kind(self, maze_choice: str) -> str:
        return self.maze.wizard_kind

    def _setup_cancelled(self, progress: SetupProgress) -> None:
        """Keep what was collected if it is worth resuming; always returns None."""
        self._pending_setup = progress if progress.is_resumable else None
        self._clear_setup_status("Ready")
        if self._pending_setup is not None:
            logger.info(
                "Setup cancelled; progress kept (%s)",
                ", ".join(progress.completed),
            )
        return None

    def _start_from_batch_plan(self, plan_match: PlanMatch, maze_choice: str):
        """Run a loaded Batch Plan; unmatched videos are never given another video's ROIs."""
        analysis_plan: Dict[Path, BatchSettings] = {}
        for video in self.video_paths:
            key = plan_match.matched.get(video)
            if key is not None:
                analysis_plan[video] = self.batch_plan_cache[key].copy()

        if plan_match.unmatched:
            choice = self._ask_unmatched_videos(plan_match.unmatched)
            if choice == "cancel":
                return
            if choice == "configure":
                unmatched_videos = [video for video, _ in plan_match.unmatched]
                # Default the new setup to the plan's paradigm parameters so
                # the batch stays parameter-consistent.
                first_plan = next(iter(analysis_plan.values()), None)
                if first_plan is not None and first_plan.algorithm_parameters:
                    self.maze.configure_algorithm_parameters(
                        first_plan.algorithm_parameters
                    )
                new_settings = self._run_setup_wizard(unmatched_videos[0], maze_choice)
                if new_settings is None:
                    return
                new_settings = self._review_settings(unmatched_videos[0], new_settings)
                if new_settings is None:
                    return
                for video in unmatched_videos:
                    analysis_plan[video] = new_settings.copy()

        ordered_plan = {
            video: analysis_plan[video]
            for video in self.video_paths if video in analysis_plan
        }
        if not ordered_plan:
            QMessageBox.warning(self, "No Videos", "No videos are left to analyze.")
            return
        self.run_worker(ordered_plan)

    def _ask_unmatched_videos(self, unmatched: List[Tuple[Path, str]]) -> str:
        """Ask what to do with videos missing from the plan: configure, skip or cancel."""
        listed = "\n".join(
            f"  - {video} ({reason})" for video, reason in unmatched[:15]
        )
        if len(unmatched) > 15:
            listed += f"\n  ... and {len(unmatched) - 15} more"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Videos Not In Batch Plan")
        box.setText(
            f"{len(unmatched)} video(s) have no matching entry in the loaded "
            "Batch Plan, so they have no ROIs of their own:\n\n" + listed
        )
        box.setInformativeText(
            "Configure them now, skip them, or cancel the whole run?"
        )
        configure_btn = box.addButton("Configure them", QMessageBox.ButtonRole.AcceptRole)
        skip_btn = box.addButton("Skip them", QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is configure_btn:
            return "configure"
        if clicked is skip_btn:
            return "skip"
        return "cancel"

    def _configure_maze_from_settings(self, settings: BatchSettings, maze_choice: str):
        """Restore maze state (parameters, zones, mouse count) from loaded settings."""
        self.maze.infer_from_settings(settings)

    def _generate_reference_frame(self, ref_video_path: Path):
        """Build the reference frame off the UI thread (raises TaskCancelled)."""
        samples = AppConfig().video.ref_frame_samples
        return run_with_progress(
            self,
            f"Generating reference frame for {ref_video_path.name}...",
            lambda should_cancel: ReferenceFrameGenerator.generate(
                ref_video_path, samples, use_cache=True,
                should_cancel=should_cancel
            ),
            title="Reference Frame",
        )

    def _run_setup_wizard(
        self, ref_video_path: Path, maze_choice: str,
        progress: Optional[SetupProgress] = None,
    ) -> Optional[BatchSettings]:
        """Interactive ROI / reference-line / detection setup; None if cancelled.

        Every finished step is recorded in a SetupProgress. If the wizard is
        cancelled after the ROI step the progress is kept, and passing it back
        in skips the steps that are already done.
        """
        logger.info(f"Using '{ref_video_path.name}' as reference for configuration.")
        kind = self._wizard_kind(maze_choice)
        if progress is None:
            progress = SetupProgress(maze_choice, ref_video_path, kind)
            self._pending_setup = None
        elif progress.base_settings is not None:
            return progress.base_settings
        video = str(ref_video_path)

        def params():
            return copy.deepcopy(self.maze.get_algorithm_parameters())

        # --- TST: number of mice -> ROIs -> motion tuning ---
        if kind == KIND_TST:
            if not progress.has(STEP_MICE):
                num, ok = QInputDialog.getInt(self, "TST Setup", "How many mice are in this video?", 1, 1, 6)
                if not ok:
                    return self._setup_cancelled(progress)
                self.maze.configure_mice(num)
                progress.mark(STEP_MICE, params())

            if not progress.has(STEP_ROI):
                with _disposed(RoiSelectorDialog(video, self.maze, self)) as roi_dialog:
                    if not roi_dialog.exec():
                        return self._setup_cancelled(progress)
                    progress.roi_manager = roi_dialog.get_roi_manager()
                progress.mark(STEP_ROI)

            if not progress.has(STEP_TST_TUNER):
                with _disposed(TstTunerDialog(video, progress.roi_manager, self.maze, self)) as tst_tuner:
                    if not tst_tuner.exec():
                        return self._setup_cancelled(progress)
                progress.mark(STEP_TST_TUNER, params())
            progress.detection_config = AppConfig().detection

        # --- Freestyle: zones -> ROIs (or reference line) -> detection ---
        elif kind == KIND_FREESTYLE:
            if not progress.has(STEP_ZONES):
                self._show_setup_status("Configuration", "Configuring freestyle zones...")
                with _disposed(FreestyleConfigDialog(self, maze=self.maze)) as freestyle_dialog:
                    if not freestyle_dialog.exec():
                        return self._setup_cancelled(progress)
                    zone_definitions = freestyle_dialog.get_zone_definitions()
                    progress.zone_free_mode = freestyle_dialog.is_zone_free_mode()
                self.maze.configure_zones(zone_definitions)
                progress.mark(STEP_ZONES, params())

            if not progress.has(STEP_ROI):
                self._show_setup_status("Configuration", "Defining regions of interest...")
                roi_manager = ROIManager()
                # Zone-free runs still need the selector for an open-field arena.
                if not progress.zone_free_mode or self.maze.arena_enabled:
                    with _disposed(RoiSelectorDialog(video, self.maze, self)) as roi_dialog:
                        if not roi_dialog.exec():
                            return self._setup_cancelled(progress)
                        roi_manager = roi_dialog.get_roi_manager()
                else:
                    if self.maze.needs_reference_line():
                        reply = QMessageBox.question(
                            self, 'Calibration',
                            REFERENCE_LINE_PROMPT,
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.Yes
                        )
                        if reply == QMessageBox.StandardButton.Yes:
                            offer_reference_line(
                                self, ref_video_path, roi_manager
                            )
                    if not roi_manager.source_dimensions:
                        roi_manager.set_source_dimensions(
                            VideoHandler(ref_video_path).dimensions
                        )
                progress.roi_manager = roi_manager
                progress.mark(STEP_ROI)

            if not self._tune_detection(progress, ref_video_path):
                return self._setup_cancelled(progress)

        # --- Y-maze / EPM / other: ROIs -> (arm rules) -> detection ---
        else:
            if not progress.has(STEP_ROI):
                self._show_setup_status("Configuration", "Defining regions of interest...")
                with _disposed(RoiSelectorDialog(video, self.maze, self)) as roi_dialog:
                    if not roi_dialog.exec():
                        return self._setup_cancelled(progress)
                    progress.roi_manager = roi_dialog.get_roi_manager()
                progress.mark(STEP_ROI)

            if kind == KIND_ARMS and not progress.has(STEP_ARM_RULES):
                with _disposed(ArmEntryParametersDialog(self.maze, self)) as arm_dialog:
                    if not arm_dialog.exec():
                        return self._setup_cancelled(progress)
                    self.maze.configure_algorithm_parameters(arm_dialog.get_parameters())
                progress.mark(STEP_ARM_RULES, params())

            if not self._tune_detection(progress, ref_video_path):
                return self._setup_cancelled(progress)

        base_settings = BatchSettings(
            progress.roi_manager, progress.detection_config, 0.0
        )
        # The template was drawn on this video: it is its reference view.
        self._alignment.remember_template_video(ref_video_path, progress.roi_manager)
        # Only the final review is left; keep everything until it is accepted.
        progress.base_settings = base_settings
        self._pending_setup = progress
        return base_settings

    def _tune_detection(self, progress: SetupProgress, ref_video_path: Path) -> bool:
        """Reference frame + detection tuner; False if the user backed out."""
        if progress.has(STEP_DETECTION):
            return True
        self._show_setup_status("Configuration", "Generating reference frame...")
        try:
            ref_frame = self._generate_reference_frame(ref_video_path)
        except TaskCancelled:
            return False
        self._show_setup_status("Configuration", "Tuning detection parameters...")
        with _disposed(TunerDialog(str(ref_video_path), ref_frame, progress.roi_manager, self)) as tuner_dialog:
            if not tuner_dialog.exec():
                return False
            progress.detection_config = tuner_dialog.get_detection_config()
        progress.mark(STEP_DETECTION)
        return True

    def _review_settings(self, ref_video_path: Path, base_settings: BatchSettings) -> Optional[BatchSettings]:
        """Final review of times and calibration; None if cancelled.

        Cancelling keeps the pending setup, so the next Start can resume
        straight at this review.
        """
        self._show_setup_status("Configuration", "Reviewing analysis parameters...")

        settings_dialog = SettingsDialog(
            str(ref_video_path), base_settings, is_verification=False,
            parent=self, maze_name=self.maze.name
        )

        if not settings_dialog.exec():
            settings_dialog.deleteLater()
            logger.info("Settings cancelled.")
            self._clear_setup_status("Ready")
            return None

        base_settings = settings_dialog.get_settings()
        settings_dialog.deleteLater()
        base_settings.paradigm_id = self.maze.name
        base_settings.algorithm_parameters = self.maze.get_algorithm_parameters()
        self.current_settings = base_settings
        self._pending_setup = None
        self.update_button_states()
        return base_settings

    def run_batch_analysis(self, base_settings: BatchSettings):
        analysis_plan = {video: base_settings.copy() for video in self.video_paths}
        # run_worker's pre-flight summary is the single "Start" confirmation.
        self.run_worker(analysis_plan)

    def run_batch_with_verification(
        self, base_settings: BatchSettings,
        reviewed_video: Optional[Path] = None,
    ):
        analysis_plan = self.build_analysis_plan(
            base_settings, verify_each=True, reviewed_video=reviewed_video
        )
        # Verification ended with its own Run now step, so only warnings ask again.
        if analysis_plan:
            self.run_worker(analysis_plan, confirm_start=False)

    def build_analysis_plan(
        self, base_settings: BatchSettings, verify_each: bool,
        reviewed_video: Optional[Path] = None,
    ) -> Dict[Path, BatchSettings]:
        """Per-video settings; ``reviewed_video`` was just reviewed in setup and
        is accepted as is instead of being shown a second time."""
        analysis_plan: Dict[Path, BatchSettings] = {}
        if not verify_each:
            for video_path in self.video_paths: analysis_plan[video_path] = base_settings.copy()
            return analysis_plan

        logger.info("Starting Verification Phase...")
        self._show_setup_status("Verification", "Reviewing each video...")
        QApplication.processEvents()

        current_settings = base_settings
        redraw_cm = None  # cm length to pre-fill right after a Redraw
        for i, video_path in enumerate(self.video_paths):
            logger.info(f"Verifying video {i+1}/{len(self.video_paths)}: {video_path.name}")
            if reviewed_video is not None and video_path == reviewed_video:
                logger.info(f"'{video_path.name}' was just reviewed; not shown again")
                analysis_plan[video_path] = base_settings.copy()
                reviewed_video = None
                continue
            self._show_setup_status(
                "Verification", f"Video {i + 1} of {len(self.video_paths)}: {video_path.name}"
            )
            try:
                current_settings = self._alignment.settings_for_review(
                    video_path, base_settings, current_settings
                )
                while True:
                    dialog = SettingsDialog(
                        str(video_path), current_settings.copy(),
                        is_verification=True, parent=self,
                        maze_name=self.maze.name,
                        real_world_cm=redraw_cm
                    )
                    redraw_cm = None
                    dialog.setWindowTitle(f"Verify: {video_path.name}")
                    if not dialog.exec():
                        dialog.deleteLater()
                        outcome = self._ask_partial_verification(
                            analysis_plan, len(self.video_paths)
                        )
                        if outcome.keep_verifying:
                            continue
                        return outcome.run_plan

                    action = dialog.result_action
                    if action == "accept":
                        final_settings = dialog.get_settings()
                        dialog.deleteLater()
                        analysis_plan[video_path] = final_settings.copy()
                        current_settings = final_settings
                        break
                    elif action == "skip":
                        dialog.deleteLater()
                        logger.info(f"Video '{video_path.name}' skipped")
                        break
                    elif action == "align":
                        # Re-align the ORIGINAL template (never a chained
                        # result); the times and calibration entered so far
                        # are kept.
                        carried = dialog.get_settings()
                        redraw_cm = dialog.get_real_world_cm() or None
                        dialog.deleteLater()
                        aligned = self._alignment.align_one(
                            video_path, base_settings, carried
                        )
                        current_settings = aligned if aligned is not None else carried
                    elif action == "redraw":
                        # Keep the times the user already set; the cm length
                        # (not the old scale factor) is what carries over.
                        carried = dialog.get_settings()
                        real_world_cm = dialog.get_real_world_cm()
                        dialog.deleteLater()
                        with _disposed(RoiSelectorDialog(
                                str(video_path), self.maze, self,
                                initial_roi_manager=carried.roi_manager)) as roi_dialog:
                            if roi_dialog.exec():
                                current_settings = self.settings_after_redraw(
                                    carried,
                                    roi_dialog.get_roi_manager(),
                                    real_world_cm,
                                )
                                redraw_cm = real_world_cm or None
            except Exception as e:
                logger.error(f"Could not verify {video_path.name}: {e}", exc_info=True)
                QMessageBox.warning(
                    self, "Verification Error",
                    f"Could not verify '{video_path.name}':\n{e}\n\n"
                    "This video is excluded from the run. "
                    "Verification continues with the next video."
                )
                continue

        if not analysis_plan:
            QMessageBox.warning(self, "No Videos", "All videos were skipped.")
            self._clear_setup_status("Ready")
            return {}

        answer = ux_helpers.ask_choice(
            self, "Verification Complete",
            f"{len(analysis_plan)} of {len(self.video_paths)} video(s) verified.",
            [
                Choice("run", "Run now"),
                Choice("save", "Save plan, then run…",
                       QMessageBox.ButtonRole.ActionRole),
                Choice("cancel", "Cancel", QMessageBox.ButtonRole.RejectRole),
            ],
            default_key="run",
            informative="A saved batch plan can be reloaded later with File > Load Configuration.",
        )
        if answer == "cancel":
            self._clear_setup_status("Ready")
            return {}
        if answer == "save":
            self._save_batch_plan_as(analysis_plan)
        return analysis_plan

    def _ask_partial_verification(self, analysis_plan, total: int):
        """Per-video verification was cancelled: keep, run or save what was accepted."""
        choices = ux_helpers.verification_cancel_choices(len(analysis_plan))
        if not choices:
            self._clear_setup_status("Ready")
            return ux_helpers.partial_verification_outcome(None, analysis_plan)
        choice = ux_helpers.ask_choice(
            self, "Verification Cancelled",
            f"{len(analysis_plan)} of {total} video(s) are already verified.",
            choices,
            default_key=ux_helpers.VERIFY_CONTINUE,
            informative="What should happen to the videos you already accepted?",
        )
        outcome = ux_helpers.partial_verification_outcome(choice, analysis_plan)
        if outcome.save_plan:
            self._save_batch_plan_as(analysis_plan)
        if not outcome.keep_verifying and not outcome.run_plan:
            self._clear_setup_status("Ready")
        return outcome

    def _save_batch_plan_as(self, analysis_plan: Dict[Path, BatchSettings]) -> Optional[Path]:
        """Ask where to save a batch plan; returns the file or None."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Batch Plan",
            str(Path(self.app_settings.dialog_dir("config", self.output_dir)) / "batch_plan.json"),
            "JSON Files (*.json)"
        )
        if not filename:
            return None
        try:
            BatchSettings.save_batch_plan(
                build_plan_by_key(analysis_plan, Path(filename)),
                Path(filename)
            )
        except Exception as e:
            logger.error(f"Failed to save batch plan: {e}", exc_info=True)
            QMessageBox.critical(self, "Save Failed", f"Failed to save the Batch Plan:\n{e}")
            return None
        self.app_settings.set_last_config_dir(Path(filename).parent)
        self.app_settings.add_recent_file(filename)
        self._rebuild_recent_menu()
        logger.info(f"Batch plan saved to: {filename}")
        return Path(filename)

    @staticmethod
    def settings_after_redraw(
        settings: BatchSettings,
        new_roi_manager: ROIManager,
        real_world_cm: float
    ) -> BatchSettings:
        """Rebuild settings around redrawn ROIs and a recomputed scale."""
        redrawn = settings.copy()
        redrawn.roi_manager = new_roi_manager
        redrawn.scale_factor = rescale_for_new_reference(
            real_world_cm, new_roi_manager.reference_length_pixels
        )
        return redrawn

    def _refresh_center_overlay(self, settings: BatchSettings) -> None:
        """Cache the open-field center outline so previews and the validation
        video can draw it (no-op for paradigms without an arena)."""
        update = getattr(self.maze, 'update_roi_overlay', None)
        if callable(update):
            update(settings.roi_manager, settings.scale_factor)

    def run_worker(self, analysis_plan: Dict[Path, BatchSettings],
                   confirm_start: bool = True):
        first_settings = next(iter(analysis_plan.values()), None)
        try:
            if first_settings and first_settings.algorithm_parameters:
                self.maze.configure_algorithm_parameters(
                    first_settings.algorithm_parameters
                )
        except (TypeError, ValueError) as exc:
            QMessageBox.critical(
                self,
                "Incompatible Configuration",
                str(exc),
            )
            return
        if not self._validate_analysis_plan(analysis_plan, confirm_start):
            return
        proceed, sample_metadata = self.run_options.resolve_for_run(
            list(analysis_plan)
        )
        if not proceed:
            return
        effective_parameters = self.maze.get_algorithm_parameters()
        for settings in analysis_plan.values():
            settings.paradigm_id = self.maze.name
            settings.algorithm_parameters = copy.deepcopy(
                effective_parameters
            )
            self._refresh_center_overlay(settings)

        self._run_started = time.monotonic()
        self._worker_eta = None
        self.eta_label.setText("Batch ETA: calculating…")
        self.eta_label.setVisible(True)
        self._video_status = {
            video: (STATUS_QUEUED, "") for video in analysis_plan
        }
        self._refresh_video_list(self.video_list_widget.selected_rows())
        self.set_ui_state(is_running=True)
        self.progress_bar.setValue(0)
        
        self.analysis_thread = QThread()
        try:
            self.analysis_worker = AnalysisWorker(
                analysis_plan, self.output_dir, self.maze,
                workers=self._worker_count(len(analysis_plan)),
                sample_metadata=sample_metadata,
                bin_seconds=self.run_options.bin_seconds(),
            )
        except Exception as exc:
            self.analysis_thread = None
            self.analysis_worker = None
            self.set_ui_state(is_running=False)
            QMessageBox.critical(
                self,
                "Cannot Start Analysis",
                f"Could not create the run output folder:\n{exc}",
            )
            return
        self.analysis_worker.moveToThread(self.analysis_thread)
        
        self.analysis_thread.started.connect(self.analysis_worker.run)
        self.analysis_worker.finished.connect(self.analysis_finished)
        self.analysis_worker.progress.connect(self.update_progress)
        self.analysis_worker.detailed_progress.connect(self.update_progress_detailed)
        self.analysis_worker.log.connect(self._on_worker_log)
        self.analysis_worker.video_status.connect(self._on_video_status)
        # Qt frees both objects once the thread has stopped; analysis_finished
        # drops our references first so nothing touches them afterwards.
        self.analysis_thread.finished.connect(self.analysis_worker.deleteLater)
        self.analysis_thread.finished.connect(self.analysis_thread.deleteLater)

        self.analysis_thread.start()

    def _worker_count(self, video_count: int) -> int:
        """Videos to analyse at once: the spinbox, or the automatic default."""
        chosen = self.workers_spin.value()
        if chosen <= 0:
            return default_worker_count(video_count)
        return max(1, min(chosen, video_count))

    def _run_preflight(
        self,
        analysis_plan: Dict[Path, BatchSettings]
    ) -> Tuple[List[str], List[str]]:
        """Run the batch pre-flight checks behind a small progress dialog."""
        dialog = QProgressDialog(
            "Checking videos...", "", 0, max(len(analysis_plan), 1), self
        )
        dialog.setWindowTitle("Pre-flight Checks")
        dialog.setCancelButton(None)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(400)

        def on_progress(done: int, total: int, name: str) -> None:
            dialog.setValue(done)
            dialog.setLabelText(f"Checking {name}..." if name else "Finishing checks...")
            QApplication.processEvents()

        try:
            return PreflightChecker().validate_batch(
                analysis_plan,
                self.output_dir,
                paradigm_name=self.maze.name,
                progress=on_progress,
            )
        finally:
            dialog.close()

    def _validate_analysis_plan(
        self,
        analysis_plan: Dict[Path, BatchSettings],
        confirm_start: bool = True,
    ) -> bool:
        """Validate every per-video configuration, including loaded plans.

        Problems that block the run are reported and stop here. Otherwise the
        user sees one summary (with any warnings) and chooses Start; with
        ``confirm_start`` False the box only appears when there are warnings.
        """
        # Apply stored alignments and align videos whose size differs from
        # their ROIs' (in place); declining ends here, as pre-flight would.
        if not self._alignment.prepare_plan(analysis_plan):
            return False
        # Video, settings, disk and system checks run once for the batch.
        all_issues, all_warnings = self._run_preflight(analysis_plan)
        expected_parameters = None

        for video_path, settings in analysis_plan.items():
            all_issues.extend(
                f"{video_path.name}: {issue}"
                for issue in self.maze.validate_settings(settings)
            )

            if expected_parameters is None:
                expected_parameters = settings.algorithm_parameters
            elif settings.algorithm_parameters != expected_parameters:
                all_issues.append(
                    f"{video_path.name}: paradigm parameters differ from the "
                    "rest of the batch; mixed-parameter batches are not yet supported"
                )

        if all_issues:
            title, message = ValidationDialog.format_for_messagebox(
                all_issues, all_warnings
            )
            QMessageBox.critical(self, title, message)
            return False

        if not all_warnings and not confirm_start:
            return True
        title, text, informative = ux_helpers.format_start_summary(
            len(analysis_plan), self.maze.name, self.output_dir, all_warnings
        )
        answer = ux_helpers.ask_choice(
            self, title, text,
            [
                Choice("start", "Start analysis"),
                Choice("cancel", "Cancel", QMessageBox.ButtonRole.RejectRole),
            ],
            default_key="start",
            informative=informative,
            icon=QMessageBox.Icon.Warning if all_warnings else QMessageBox.Icon.Question,
        )
        return answer == "start"

    def _on_video_status(self, video: str, status: str, message: str):
        """Update one row from the worker's per-video signal."""
        self._video_status[Path(video)] = (status, message)
        self._refresh_video_list(self.video_list_widget.selected_rows())
        if status == STATUS_RUNNING:
            row = next(
                (i for i, path in enumerate(self.video_paths) if path == Path(video)),
                None,
            )
            if row is not None:
                self.video_list_widget.scrollToItem(self.video_list_widget.item(row))

    def open_output_folder(self):
        folder = self.output_dir or (
            self.video_paths[0].parent if self.video_paths else None
        )
        if folder is None:
            QMessageBox.information(
                self, "No Output Folder",
                "Add videos or choose an output folder first."
            )
            return
        self._open_path(folder)

    def open_last_run_folder(self):
        if self._last_run_dir is not None:
            self._open_path(self._last_run_dir)

    def _open_path(self, path: Path) -> bool:
        """Open a file or folder with the system's default application."""
        if not Path(path).exists():
            QMessageBox.warning(self, "Not Found", f"{path}\n\ndoes not exist (any more).")
            return False
        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        if not opened:
            logger.warning("Could not open %s", path)
            QMessageBox.warning(self, "Cannot Open", f"Could not open:\n{path}")
        return opened

    def _show_run_result(self, icon, title: str, text: str,
                         run_output_dir: Optional[Path]):
        """Result box that can open the run folder and the batch summary."""
        summary = run_output_dir / "batch_summary.csv" if run_output_dir else None
        while True:
            choices = []
            if run_output_dir is not None:
                choices.append(Choice(
                    "folder", "Open output folder",
                    QMessageBox.ButtonRole.ActionRole,
                ))
            if summary is not None and summary.is_file():
                choices.append(Choice(
                    "summary", "Open batch summary",
                    QMessageBox.ButtonRole.ActionRole,
                ))
            choices.append(Choice("ok", "OK", QMessageBox.ButtonRole.RejectRole))
            answer = ux_helpers.ask_choice(
                self, title, text, choices, default_key="ok", icon=icon
            )
            if answer == "folder":
                self._open_path(run_output_dir)
            elif answer == "summary":
                self._open_path(summary)
            else:
                return

    def confirm_cancel_analysis(self):
        """Stop button and Esc: ask first, since a stop interrupts the video in progress."""
        if not self._analysis_running() or self._close_requested:
            return
        answer = ux_helpers.ask_choice(
            self, "Stop the analysis?",
            "Stop the running analysis?",
            [
                Choice("stop", "Stop analysis", QMessageBox.ButtonRole.DestructiveRole),
                Choice("keep", "Keep running", QMessageBox.ButtonRole.RejectRole),
            ],
            default_key="keep",
            informative=(
                "Progress is saved. The video in progress can be resumed "
                "from its checkpoint later."
            ),
        )
        if answer == "stop":
            self.cancel_analysis()

    def cancel_analysis(self):
        if self.analysis_worker:
            self.analysis_worker.stop()
            self.cancel_button.setEnabled(False)
            self.update_progress_detailed(self.progress_bar.value(), "Cancelling...", "Please wait...")

    def _show_setup_status(self, main_text: str, sub_text: str = ""):
        """Busy indicator for interactive setup; setup has no real percentage."""
        self.progress_bar.setRange(0, 0)
        self.status_label.setText(main_text)
        self.subtask_label.setText(sub_text)

    def _clear_setup_status(self, text: str = "Ready"):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText(text)
        self.subtask_label.setText("")

    def _ensure_determinate_progress(self):
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 100)

    def _on_worker_log(self, text: str):
        """Worker status lines go to the log at the severity their symbol carries."""
        logging.log(ux_helpers.level_for_worker_message(text), text)

    def _on_log_level_changed(self, _index: int):
        handler = self.log_handler
        if handler is not None:
            handler.set_min_level(self.log_level_combo.currentData())

    def save_log(self):
        """Save the complete log to a text file."""
        handler = self.log_handler
        if handler is None:
            return
        default = Path(self.app_settings.dialog_dir("output", self.output_dir)) / "analysis_log.txt"
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Log", str(default), "Text Files (*.txt);;All Files (*.*)"
        )
        if not filename:
            return
        try:
            Path(filename).write_text(handler.full_text(), encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Save Failed", f"Could not save the log:\n{exc}")
            return
        logger.info(f"Log saved to: {filename}")

    def update_progress(self, value: int, text: str):
        self._ensure_determinate_progress()
        self.progress_bar.setValue(value)
        self.status_label.setText(text)
        self._update_eta(value)

    def _update_eta(self, percent: int, sub_text: str = ""):
        """Show the batch ETA: the worker's own figure, else one from elapsed time."""
        if self._run_started is None:
            return
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
            f"Batch ETA: {eta_text or 'calculating…'}   "
            f"(elapsed {ux_helpers.format_duration(elapsed)})"
        )

    def update_progress_detailed(self, value: int, main_text: str, sub_text: str):
        self._ensure_determinate_progress()
        self.progress_bar.setValue(value)
        self.status_label.setText(main_text)
        self.subtask_label.setText(sub_text)
        self._update_eta(value, sub_text)

    def analysis_finished(self, outcome: BatchRunOutcome):
        logger.info("Analysis run finished.")
        all_results = outcome.results
        run_output_dir = outcome.output_dir or self.output_dir
        closing = self._close_requested

        # The per-video signals normally got there first; this makes the
        # list agree with the final outcome whatever their order was.
        for video, state in outcome_statuses(
            (path for path, _ in all_results),
            outcome.failures,
            outcome.quality_warnings,
        ).items():
            if video in self._video_status:
                self._video_status[video] = state
        self._refresh_video_list(self.video_list_widget.selected_rows())
        if outcome.output_dir is not None:
            self._last_run_dir = outcome.output_dir
            self.open_last_run_action.setEnabled(True)

        if closing:
            # The user is quitting: no result dialogs, just finish and close.
            pass
        elif outcome.cancelled:
            self._show_run_result(
                QMessageBox.Icon.Warning,
                "Analysis Cancelled",
                f"Analysis was cancelled.\n"
                f"Completed: {len(all_results)} video(s)\n"
                "The interrupted video can be resumed from its checkpoint.",
                run_output_dir,
            )
        elif outcome.failures:
            failure_lines = "\n".join(
                f"• {path.name}: {message}"
                for path, message in outcome.failures[:5]
            )
            self._show_run_result(
                QMessageBox.Icon.Warning,
                "Analysis Completed with Errors",
                f"Completed: {len(all_results)} video(s)\n"
                f"Failed: {len(outcome.failures)} video(s)\n\n"
                f"{failure_lines}\n\nSaved to: {run_output_dir}",
                run_output_dir,
            )
        elif outcome.quality_warnings:
            warning_lines = "\n".join(
                f"• {path.name}: {message}"
                for path, message in outcome.quality_warnings[:5]
            )
            self._show_run_result(
                QMessageBox.Icon.Warning,
                "Analysis Requires Quality Review",
                f"Completed: {len(all_results)} video(s)\n\n"
                f"{warning_lines}\n\n"
                "Review the per-frame evidence and, when generated, the "
                "validation video before "
                f"interpreting these results.\n\nSaved to: {run_output_dir}",
                run_output_dir,
            )
        else:
            self._show_run_result(
                QMessageBox.Icon.Information,
                "Complete",
                f"Analysis finished!\nProcessed: {len(all_results)} video(s)\n"
                f"Saved to: {run_output_dir}",
                run_output_dir,
            )
        self.set_ui_state(is_running=False)
        self._run_started = None
        self.eta_label.setVisible(False)
        if outcome.cancelled:
            self.update_progress_detailed(
                self.progress_bar.value(), "Cancelled", "Checkpoint saved"
            )
        else:
            self.progress_bar.setValue(100)
            if outcome.failures:
                status = "Finished with errors"
            elif outcome.quality_warnings:
                status = "Finished — quality review required"
            else:
                status = "Finished"
            self.update_progress_detailed(100, status, "")
        
        thread = self.analysis_thread
        # Drop our references before the thread's finished signal lets Qt
        # delete the worker and thread.
        self.analysis_thread = None
        self.analysis_worker = None
        if thread is not None:
            thread.quit()
            thread.wait(5000)
        self.check_for_resumable_analyses()

        if closing:
            QTimer.singleShot(0, self.close)

    def set_ui_state(self, is_running: bool):
        self.start_button.setEnabled(not is_running)
        self.add_videos_button.setEnabled(not is_running)
        self.add_folder_button.setEnabled(not is_running)
        self.clear_list_button.setEnabled(not is_running and bool(self.video_paths))
        self.recursive_check.setEnabled(not is_running)
        self.workers_spin.setEnabled(not is_running)
        self.run_options.setEnabled(not is_running)
        has_selection = bool(self.video_list_widget.selectedItems())
        for button in (self.remove_selected_button, self.move_up_button,
                       self.move_down_button):
            button.setEnabled(not is_running and has_selection)
        self.cancel_button.setEnabled(is_running)
        self._sync_actions(running=is_running)
        self.preview_button.setEnabled(not is_running and (self.current_settings is not None or len(self.batch_plan_cache) > 0))
        # Configuration and output-folder changes would race the running job.
        self.save_config_action.setEnabled(
            not is_running and self.current_settings is not None
        )
        self.load_config_action.setEnabled(not is_running)
        self.load_run_action.setEnabled(not is_running)
        self.set_output_action.setEnabled(not is_running)
        self.change_output_button.setEnabled(not is_running)
        self.recent_menu.setEnabled(not is_running)
        self._refresh_resume_bar()

    def _analysis_running(self) -> bool:
        thread = self.analysis_thread
        return thread is not None and thread.isRunning()

    def closeEvent(self, event):
        if not self._analysis_running():
            self._save_session()
            self._detach_log_handler()
            event.accept()
            return

        if self._force_close:
            # Last resort after the worker failed to stop in time.
            self._stop_thread_forcefully()
            self._save_session()
            self._detach_log_handler()
            event.accept()
            return

        if self._close_requested:
            # Already stopping; keep the window until the worker is done.
            event.ignore()
            return

        reply = QMessageBox.question(
            self, 'Analysis in Progress', "Quit and cancel analysis?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            event.ignore()
            return

        # Never block the GUI thread waiting for the worker: ask it to stop,
        # keep the window open, and close from analysis_finished.
        self._close_requested = True
        self.cancel_analysis()
        self.update_progress_detailed(
            self.progress_bar.value(), "Stopping analysis…",
            "Saving progress; the window closes automatically when done."
        )
        self._schedule_force_quit_offer()
        event.ignore()

    def _offer_force_quit(self):
        """The worker did not stop in time; let the user stop waiting."""
        if not (self._close_requested and self._analysis_running()):
            return
        if self._confirm_force_quit():
            self._force_close = True
            self.close()
        else:
            self._schedule_force_quit_offer()

    def _schedule_force_quit_offer(self):
        # A timer owned by the window dies with it, unlike a free singleShot
        # whose callback could otherwise run against a deleted window.
        timer = getattr(self, "_force_quit_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._offer_force_quit)
            self._force_quit_timer = timer
        timer.start(self.FORCE_QUIT_DELAY_MS)

    def _confirm_force_quit(self) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Analysis Still Stopping")
        box.setText(
            "The analysis has not stopped yet.\n\n"
            "Force quit now? The checkpoint of the video in progress may be "
            "stale, so the next run may redo some of its frames, and any "
            "video or plot being written will be incomplete."
        )
        force = box.addButton(
            "Force Quit", QMessageBox.ButtonRole.DestructiveRole
        )
        box.addButton("Keep Waiting", QMessageBox.ButtonRole.RejectRole)
        # Record the choice from the signal: the box (and even this window)
        # may already be deleted when the nested event loop returns.
        chosen = []
        force.clicked.connect(lambda: chosen.append(True))
        box.exec()
        return bool(chosen)

    def _stop_thread_forcefully(self):
        thread = self.analysis_thread
        if thread is None:
            return
        thread.quit()
        if not thread.wait(2000):
            logger.warning("Analysis thread did not stop; terminating it")
            thread.terminate()
            thread.wait(2000)

    def _detach_log_handler(self):
        """Remove the Qt-backed handler before its widget is destroyed."""
        handler = getattr(self, "log_handler", None)
        if handler is None:
            return
        logging.getLogger().removeHandler(handler)
        handler.close()
        self.log_handler = None
