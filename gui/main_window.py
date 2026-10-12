"""
gui/main_window.py
Main GUI Window.
Updated to use unified Core modules (VideoHandler, Models).
"""
import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                             QListWidget, QMainWindow, QMessageBox,
                             QProgressBar, QProgressDialog, QPushButton, QTextEdit, QVBoxLayout,
                             QWidget, QInputDialog)
from PyQt6.QtGui import QAction

# --- CORE IMPORTS ---
from core.batch_plan_keys import (PlanMatch, build_plan_by_key,
                                  match_videos_to_plan)
from core.calibration import rescale_for_new_reference
from core.models import AppConfig, BatchSettings
from core.video import VideoHandler, ReferenceFrameGenerator
from core.version import APP_NAME, APP_VERSION

# --- DIALOGS ---
from gui.roi_selector import (REFERENCE_LINE_PROMPT, RoiSelectorDialog,
                              offer_reference_line)
from gui.arm_entry_dialog import ArmEntryParametersDialog
from gui.settings_dialog import SettingsDialog
from gui.tuner_dialog import TunerDialog
from gui.tst_tuner import TstTunerDialog
from gui.background_task import TaskCancelled, run_with_progress
from gui.worker import AnalysisWorker, BatchRunOutcome
from gui.roi_preview import ROIPreviewDialog
from gui.freestyle_config_dialog import FreestyleConfigDialog

# --- UTILS ---
from roi_manager import ROIManager
from mazes import AVAILABLE_MAZES, create_maze
from mazes.base_maze import Maze
from mazes.epm import EPM
from mazes.tst import TST
from mazes.y_maze import YMaze
from preflight_checker import PreflightChecker, ValidationDialog
from checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


class QTextEditLogger(logging.Handler, QObject):
    """Custom logging handler that writes to a QTextEdit widget."""
    appendPlainText = pyqtSignal(str)
    
    def __init__(self, parent):
        super().__init__()
        QObject.__init__(self)
        self.widget = QTextEdit(parent)
        self.widget.setReadOnly(True)
        self.appendPlainText.connect(self.widget.append)
    
    def emit(self, record):
        self.appendPlainText.emit(self.format(record))


class MainWindow(QMainWindow):
    """Enhanced main window using Core architecture."""

    # How long a close request waits for the worker to stop before offering
    # to force quit.
    FORCE_QUIT_DELAY_MS = 10_000

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setGeometry(100, 100, 1000, 750)
        
        # State variables
        self.video_paths: List[Path] = []
        self.output_dir: Optional[Path] = None
        self.analysis_thread: Optional[QThread] = None
        self.analysis_worker: Optional[AnalysisWorker] = None
        self.maze: Optional[Maze] = None
        
        # Holds the 'Master Template' if loaded/configured
        self.current_settings: Optional[BatchSettings] = None
        
        # Holds specific settings per video if loaded from a Batch Plan
        # Keyed by plan key (see core.batch_plan_keys); path is the plan file.
        self.batch_plan_cache: Dict[str, BatchSettings] = {}
        self.batch_plan_path: Optional[Path] = None
        
        # Close-while-running handshake (see closeEvent)
        self._close_requested = False
        self._force_close = False

        self.init_ui()
        self.create_menu_bar()
        self.check_for_resumable_analyses()
        
    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Top button layout
        top_layout = QHBoxLayout()
        self.add_videos_button = QPushButton("Add Videos")
        self.add_folder_button = QPushButton("Add Folder")
        self.clear_list_button = QPushButton("Clear List")
        
        self.add_videos_button.clicked.connect(self.add_videos)
        self.add_folder_button.clicked.connect(self.add_folder)
        self.clear_list_button.clicked.connect(self.clear_list)
        
        top_layout.addWidget(self.add_videos_button)
        top_layout.addWidget(self.add_folder_button)
        top_layout.addWidget(self.clear_list_button)
        main_layout.addLayout(top_layout)
        
        # Video list
        self.video_list_widget = QListWidget()
        main_layout.addWidget(self.video_list_widget)
        
        # Action buttons layout
        action_layout = QHBoxLayout()
        self.start_button = QPushButton("Configure & Start Analysis")
        self.preview_button = QPushButton("Preview ROIs")
        self.cancel_button = QPushButton("Cancel Analysis")
        
        self.start_button.clicked.connect(self.start_analysis_workflow)
        self.preview_button.clicked.connect(self.preview_rois)
        self.cancel_button.clicked.connect(self.cancel_analysis)
        
        self.cancel_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        
        action_layout.addWidget(self.start_button)
        action_layout.addWidget(self.preview_button)
        action_layout.addWidget(self.cancel_button)
        main_layout.addLayout(action_layout)
        
        # Progress indicators
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.status_label = QLabel("Ready. Please add videos to begin.")
        self.status_label.setStyleSheet("padding: 5px; font-weight: bold;")
        
        self.subtask_label = QLabel("")
        self.subtask_label.setStyleSheet("padding: 2px; font-size: 10pt; color: #666;")
        
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.subtask_label)
        
        # Log output
        self.log_handler = QTextEditLogger(self)
        self.log_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        )
        logging.getLogger().addHandler(self.log_handler)
        main_layout.addWidget(self.log_handler.widget)

    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu('&File')
        
        save_config_action = QAction('&Save Master Template...', self)
        save_config_action.triggered.connect(self.save_configuration)
        save_config_action.setEnabled(False)
        self.save_config_action = save_config_action
        
        load_config_action = QAction('&Load Configuration / Plan...', self)
        load_config_action.triggered.connect(self.load_configuration)
        self.load_config_action = load_config_action
        
        file_menu.addAction(save_config_action)
        file_menu.addAction(load_config_action)
        file_menu.addSeparator()
        
        exit_action = QAction('E&xit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        tools_menu = menubar.addMenu('&Tools')
        set_output_action = QAction('Set &Output Directory...', self)
        set_output_action.triggered.connect(self.set_output_directory)
        tools_menu.addAction(set_output_action)
        self.set_output_action = set_output_action

    def add_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        if files:
            output_was_unset = self.output_dir is None
            added_count = 0
            for file in files:
                path = Path(file)
                if path not in self.video_paths:
                    self.video_paths.append(path)
                    self.video_list_widget.addItem(file)
                    added_count += 1
            
            if self.output_dir is None and self.video_paths:
                self.output_dir = self.video_paths[0].parent
            if output_was_unset and self.output_dir is not None:
                self.check_for_resumable_analyses()
            
            logger.info(f"Added {added_count} video(s)")
            self.update_button_states()

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing Videos")
        if not folder: return
        
        folder_path = Path(folder)
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
        
        found_videos = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]
        
        if not found_videos:
            QMessageBox.information(self, "No Videos Found", f"No video files found in:\n{folder_path}")
            return
        
        reply = QMessageBox.question(
            self, "Confirm Batch Import",
            f"Found {len(found_videos)} video(s) in folder.\nAdd all?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            output_was_unset = self.output_dir is None
            added_count = 0
            for video in sorted(found_videos):
                if video not in self.video_paths:
                    self.video_paths.append(video)
                    self.video_list_widget.addItem(str(video))
                    added_count += 1
            
            if self.output_dir is None and self.video_paths:
                self.output_dir = folder_path
            if output_was_unset and self.output_dir is not None:
                self.check_for_resumable_analyses()
            
            logger.info(f"Added {added_count} video(s) from folder: {folder_path.name}")
            self.update_button_states()

    def clear_list(self):
        if self.video_paths:
            reply = QMessageBox.question(self, "Clear List", "Remove all video(s)?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.video_paths.clear()
                self.video_list_widget.clear()
                self.current_settings = None
                self.batch_plan_cache = {}
                self.batch_plan_path = None
                self.update_button_states()
                logger.info("Video list cleared")

    def update_button_states(self):
        has_videos = len(self.video_paths) > 0
        has_settings = self.current_settings is not None or len(self.batch_plan_cache) > 0
        
        self.start_button.setEnabled(has_videos)
        self.preview_button.setEnabled(has_videos and has_settings)
        self.save_config_action.setEnabled(self.current_settings is not None)

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
            self, "Save Master Template",
            str(self.output_dir / "template_config.json") if self.output_dir else "template_config.json",
            "JSON Files (*.json);;All Files (*.*)"
        )
        if filename:
            try:
                self.current_settings.save_to_file(Path(filename))
                logger.info(f"Template saved to: {filename}")
                QMessageBox.information(self, "Saved", f"Master Template saved to:\n{filename}")
            except Exception as e:
                logger.error(f"Failed to save configuration: {e}", exc_info=True)
                QMessageBox.critical(self, "Save Failed", f"Failed to save:\n{e}")

    def load_configuration(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration",
            str(self.output_dir) if self.output_dir else "",
            "JSON Files (*.json);;All Files (*.*)"
        )
        if not filename: return
        
        file_path = Path(filename)
        try:
            file_type = BatchSettings.detect_file_type(file_path)
            
            if file_type == 'batch_plan':
                plan = BatchSettings.load_batch_plan(file_path)
                self.batch_plan_cache = plan
                self.batch_plan_path = file_path
                if plan:
                    first_key = list(plan.keys())[0]
                    self.current_settings = plan[first_key]
                    self._infer_maze_from_settings(self.current_settings)
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
                msg = ("Master Template loaded.\nThis configuration will be applied to ALL videos.")

            self.update_button_states()
            logger.info(f"Configuration loaded from: {filename}")
            QMessageBox.information(self, "Configuration Loaded", msg)
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}", exc_info=True)
            QMessageBox.critical(self, "Load Failed", f"Failed to load configuration:\n{e}")

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
        if not settings or not settings.roi_manager: return
        if settings.paradigm_id in AVAILABLE_MAZES:
            self.maze = create_maze(settings.paradigm_id)
            self.maze.configure_algorithm_parameters(settings.algorithm_parameters)
            return

        keys = set(settings.roi_manager.rois.keys())
        if 'arm_a' in keys and 'arm_b' in keys: self.maze = create_maze("Y-Maze")
        elif 'open_arm_1' in keys and 'closed_arm_1' in keys: self.maze = create_maze("Elevated Plus Maze")
        elif any(k.startswith('mouse_') for k in keys): self.maze = create_maze("Tail Suspension Test")
        else:
            self.maze = create_maze("Freestyle / Open Field")
            zones = [(k, k.replace('_', ' ').title()) for k in keys if k != 'outside']
            self.maze.configure_zones(zones)

    def set_output_directory(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory", str(self.output_dir) if self.output_dir else "")
        if folder:
            self.output_dir = Path(folder)
            self.check_for_resumable_analyses()
            logger.info(f"Output directory set to: {self.output_dir}")
            QMessageBox.information(self, "Output Directory Set", f"Results will be saved to:\n{self.output_dir}")

    def check_for_resumable_analyses(self):
        if self.output_dir is None: return
        try:
            checkpoints = CheckpointManager.find_checkpoints_readonly(
                self.output_dir
            )
            if checkpoints:
                reply = QMessageBox.question(self, 'Resume Analysis?', 
                                           f"Found {len(checkpoints)} interrupted analysis session(s). See them?",
                                           QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.Yes:
                    msg = "Resumable analyses:\n\n" + "\n".join(f"• {cp.stem.replace('_checkpoint', '')}" for cp in checkpoints)
                    QMessageBox.information(self, 'Resumable Analyses', msg)
        except Exception:
            logger.warning(
                "Could not inspect resumable analyses",
                exc_info=True,
            )

    def start_analysis_workflow(self):
        if not self.video_paths:
            QMessageBox.warning(self, "No Videos", "Please add one or more videos to the list.")
            return

        # --- MAZE SELECTION ---
        maze_names = list(AVAILABLE_MAZES.keys())
        default_idx = 0
        if self.maze:
            for i, name in enumerate(maze_names):
                if name == self.maze.name:
                    default_idx = i; break

        maze_choice, ok = QInputDialog.getItem(self, "Select Maze Type", "Which maze are you analyzing?", maze_names, default_idx, False)
        if not ok or not maze_choice: return

        self.maze = create_maze(maze_choice)
        logger.info(f"Selected maze type: {self.maze.name}")
        ref_video_path = self.video_paths[0]

        # --- CONFIGURATION LOGIC ---
        use_loaded_config = False
        plan_loaded = bool(self.batch_plan_cache)
        plan_match = self._match_plan(self.video_paths)

        try:
            if plan_match.matched:
                reply = QMessageBox.question(self, "Batch Plan Detected", "Use loaded Batch Plan settings?",
                                           QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
                if reply == QMessageBox.StandardButton.Yes:
                    self._start_from_batch_plan(plan_match, maze_choice)
                    return
            elif plan_loaded:
                QMessageBox.information(
                    self, "Batch Plan Not Applicable",
                    "None of the videos in the list match the loaded Batch Plan "
                    "(or their filenames are ambiguous).\n"
                    "You will be asked to configure them from scratch."
                )
            elif self.current_settings:
                reply = QMessageBox.question(self, "Master Template Loaded", "Use Master Template for all videos?",
                                           QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
                if reply == QMessageBox.StandardButton.Yes: use_loaded_config = True

            if use_loaded_config:
                base_settings = self.current_settings
                self._configure_maze_from_settings(base_settings, maze_choice)
            else:
                base_settings = self._run_setup_wizard(ref_video_path, maze_choice)
            if base_settings is None:
                return

            base_settings = self._review_settings(ref_video_path, base_settings)
            if base_settings is None:
                return

            # --- EXECUTION ---
            self.update_progress_detailed(0, "Configuration complete", "")
            if len(self.video_paths) > 1:
                reply = QMessageBox.question(self, "Batch Processing", "Process multiple videos?\nYes: Verify each\nNo: Run all now",
                                           QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.Yes:
                    self.run_batch_with_verification(base_settings)
                else:
                    self.run_batch_analysis(base_settings)
            else:
                self.run_batch_analysis(base_settings)

        except Exception as e:
            logger.error(f"Error in analysis workflow: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"An error occurred:\n{e}")
            self.update_progress_detailed(0, "Error", "")

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
        if settings.algorithm_parameters:
            self.maze.configure_algorithm_parameters(settings.algorithm_parameters)
        rois = settings.roi_manager.rois
        if maze_choice == "Freestyle / Open Field":
            zones = [(k, k.replace('_', ' ').title()) for k in rois.keys() if k != 'outside']
            self.maze.configure_zones(zones)
        elif isinstance(self.maze, TST):
            if settings.paradigm_id == self.maze.name:
                count = sum(1 for k in rois.keys() if k != 'outside')
            else:
                # Legacy configs without a paradigm id: infer from ROI names.
                count = sum(1 for k in rois.keys() if k.startswith('mouse_'))
            self.maze.configure_mice(count)

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

    def _run_setup_wizard(self, ref_video_path: Path, maze_choice: str) -> Optional[BatchSettings]:
        """Interactive ROI / reference-line / detection setup; None if cancelled."""
        # --- WIZARD SETUP ---
        logger.info(f"Using '{ref_video_path.name}' as reference for configuration.")
        
        # 1. TST CONFIGURATION
        if isinstance(self.maze, TST):
            num, ok = QInputDialog.getInt(self, "TST Setup", "How many mice are in this video?", 1, 1, 6)
            if not ok: return
            self.maze.configure_mice(num)
            
            roi_dialog = RoiSelectorDialog(str(ref_video_path), self.maze, self)
            if not roi_dialog.exec(): return
            roi_manager = roi_dialog.get_roi_manager()
            
            tst_tuner = TstTunerDialog(str(ref_video_path), roi_manager, self.maze, self)
            if not tst_tuner.exec(): return
            
            # Core: Create default settings
            base_settings = BatchSettings(roi_manager, AppConfig().detection, 0.0)
            
        # 2. FREESTYLE CONFIGURATION
        elif maze_choice == "Freestyle / Open Field":
            self.update_progress_detailed(5, "Configuration", "Configuring freestyle zones...")
            QApplication.processEvents()
            freestyle_dialog = FreestyleConfigDialog(self, maze=self.maze)
            if not freestyle_dialog.exec(): return
            
            zone_definitions = freestyle_dialog.get_zone_definitions()
            is_zone_free_mode = freestyle_dialog.is_zone_free_mode()
            self.maze.configure_zones(zone_definitions)
            
            self.update_progress_detailed(10, "Configuration", "Defining regions of interest...")
            roi_manager = ROIManager()
            
            if not is_zone_free_mode:
                roi_dialog = RoiSelectorDialog(str(ref_video_path), self.maze, self)
                if not roi_dialog.exec(): return
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
            
            try:
                ref_frame = self._generate_reference_frame(ref_video_path)
            except TaskCancelled:
                self.update_progress_detailed(0, "Ready", "")
                return
            tuner_dialog = TunerDialog(str(ref_video_path), ref_frame, roi_manager, self)
            if not tuner_dialog.exec(): return
            detection_config = tuner_dialog.get_detection_config()
            base_settings = BatchSettings(roi_manager, detection_config, 0.0)

        # 3. STANDARD CONFIGURATION
        else:
            self.update_progress_detailed(10, "Configuration", "Defining regions of interest...")
            roi_dialog = RoiSelectorDialog(str(ref_video_path), self.maze, self)
            if not roi_dialog.exec(): return
            roi_manager = roi_dialog.get_roi_manager()

            if isinstance(self.maze, (YMaze, EPM)):
                arm_dialog = ArmEntryParametersDialog(self.maze, self)
                if not arm_dialog.exec(): return
                self.maze.configure_algorithm_parameters(arm_dialog.get_parameters())
            
            self.update_progress_detailed(20, "Configuration", "Generating reference frame...")
            try:
                ref_frame = self._generate_reference_frame(ref_video_path)
            except TaskCancelled:
                self.update_progress_detailed(0, "Ready", "")
                return

            self.update_progress_detailed(35, "Configuration", "Tuning detection parameters...")
            tuner_dialog = TunerDialog(str(ref_video_path), ref_frame, roi_manager, self)
            if not tuner_dialog.exec(): return
            detection_config = tuner_dialog.get_detection_config()

            base_settings = BatchSettings(roi_manager, detection_config, 0.0)

        return base_settings

    def _review_settings(self, ref_video_path: Path, base_settings: BatchSettings) -> Optional[BatchSettings]:
        """Final review of times and calibration; None if cancelled."""
        self.update_progress_detailed(50, "Configuration", "Reviewing analysis parameters...")
        QApplication.processEvents()

        settings_dialog = SettingsDialog(
            str(ref_video_path), base_settings, is_verification=False,
            parent=self, maze_name=self.maze.name
        )

        if not settings_dialog.exec():
            settings_dialog.deleteLater()
            logger.info("Settings cancelled.")
            self.update_progress_detailed(0, "Ready", "")
            return None

        base_settings = settings_dialog.get_settings()
        settings_dialog.deleteLater()
        base_settings.paradigm_id = self.maze.name
        base_settings.algorithm_parameters = self.maze.get_algorithm_parameters()
        self.current_settings = base_settings
        self.update_button_states()
        return base_settings

    def run_batch_analysis(self, base_settings: BatchSettings):
        analysis_plan = {video: base_settings.copy() for video in self.video_paths}
        reply = QMessageBox.question(self, 'Start Analysis', f"Ready to analyze {len(self.video_paths)} video(s).\nProceed?", 
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
        if reply == QMessageBox.StandardButton.Yes: self.run_worker(analysis_plan)

    def run_batch_with_verification(self, base_settings: BatchSettings):
        analysis_plan = self.build_analysis_plan(base_settings, verify_each=True)
        if analysis_plan: self.run_worker(analysis_plan)

    def build_analysis_plan(self, base_settings: BatchSettings, verify_each: bool) -> Dict[Path, BatchSettings]:
        analysis_plan: Dict[Path, BatchSettings] = {}
        if not verify_each:
            for video_path in self.video_paths: analysis_plan[video_path] = base_settings.copy()
            return analysis_plan

        logger.info("Starting Verification Phase...")
        self.update_progress_detailed(0, "Verification Phase", "")
        QApplication.processEvents()
        
        current_settings = base_settings
        redraw_cm = None  # cm length to pre-fill right after a Redraw
        for i, video_path in enumerate(self.video_paths):
            logger.info(f"Verifying video {i+1}/{len(self.video_paths)}: {video_path.name}")
            try:
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
                        return {}

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
                    elif action == "redraw":
                        # Keep the times the user already set; the cm length
                        # (not the old scale factor) is what carries over.
                        carried = dialog.get_settings()
                        real_world_cm = dialog.get_real_world_cm()
                        dialog.deleteLater()
                        roi_dialog = RoiSelectorDialog(str(video_path), self.maze, self)
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
            return {}
        
        reply = QMessageBox.question(self, 'Verification Complete', "Save this verification plan?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            filename, _ = QFileDialog.getSaveFileName(self, "Save Batch Plan", str(self.output_dir / "batch_plan.json") if self.output_dir else "batch_plan.json", "JSON Files (*.json)")
            if filename:
                try:
                    BatchSettings.save_batch_plan(
                        build_plan_by_key(analysis_plan, Path(filename)),
                        Path(filename)
                    )
                except Exception as e:
                    logger.error(f"Failed to save batch plan: {e}", exc_info=True)
                    QMessageBox.critical(self, "Save Failed", f"Failed to save the Batch Plan:\n{e}")
        
        return analysis_plan

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

    def run_worker(self, analysis_plan: Dict[Path, BatchSettings]):
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
        if not self._validate_analysis_plan(analysis_plan):
            return
        effective_parameters = self.maze.get_algorithm_parameters()
        for settings in analysis_plan.values():
            settings.paradigm_id = self.maze.name
            settings.algorithm_parameters = copy.deepcopy(
                effective_parameters
            )

        self.set_ui_state(is_running=True)
        self.progress_bar.setValue(0)
        
        self.analysis_thread = QThread()
        try:
            self.analysis_worker = AnalysisWorker(
                analysis_plan, self.output_dir, self.maze
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
        self.analysis_worker.log.connect(logging.info)
        # Qt frees both objects once the thread has stopped; analysis_finished
        # drops our references first so nothing touches them afterwards.
        self.analysis_thread.finished.connect(self.analysis_worker.deleteLater)
        self.analysis_thread.finished.connect(self.analysis_thread.deleteLater)

        self.analysis_thread.start()

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
        analysis_plan: Dict[Path, BatchSettings]
    ) -> bool:
        """Validate every per-video configuration, including loaded plans."""
        # Video, settings, disk and system checks run once for the batch.
        all_issues, all_warnings = self._run_preflight(analysis_plan)
        expected_parameters = None

        for video_path, settings in analysis_plan.items():
            required_categories = [
                internal_name
                for internal_name, _ in self.maze.get_roi_definitions()
            ]
            if required_categories:
                _, roi_issues = settings.roi_manager.validate_rois(
                    required_categories
                )
                all_issues.extend(
                    f"{video_path.name}: {issue}" for issue in roi_issues
                )
            if isinstance(self.maze, TST):
                configured_categories = set(
                    settings.roi_manager.rois
                )
                expected_categories = set(required_categories)
                unexpected = configured_categories - expected_categories
                if unexpected:
                    all_issues.append(
                        f"{video_path.name}: unexpected TST ROI categories: "
                        f"{', '.join(sorted(unexpected))}"
                    )
                for category in expected_categories:
                    roi_count = len(
                        settings.roi_manager.rois.get(category, [])
                    )
                    if roi_count != 1:
                        all_issues.append(
                            f"{video_path.name}: {category} must contain "
                            f"exactly one ROI, found {roi_count}"
                        )
                interval_keys = set(settings.per_roi_times)
                invalid_interval_keys = interval_keys - expected_categories
                if invalid_interval_keys:
                    all_issues.append(
                        f"{video_path.name}: time ranges reference unknown "
                        "mice: "
                        f"{', '.join(sorted(invalid_interval_keys))}"
                    )

            if settings.paradigm_id and settings.paradigm_id != self.maze.name:
                all_issues.append(
                    f"{video_path.name}: configuration is for "
                    f"'{settings.paradigm_id}', not '{self.maze.name}'"
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

        if all_warnings:
            title, message = ValidationDialog.format_for_messagebox(
                [], all_warnings
            )
            reply = QMessageBox.question(
                self,
                title,
                message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return False

        return True

    def cancel_analysis(self):
        if self.analysis_worker:
            self.analysis_worker.stop()
            self.cancel_button.setEnabled(False)
            self.update_progress_detailed(self.progress_bar.value(), "Cancelling...", "Please wait...")

    def update_progress(self, value: int, text: str):
        self.progress_bar.setValue(value)
        self.status_label.setText(text)

    def update_progress_detailed(self, value: int, main_text: str, sub_text: str):
        self.progress_bar.setValue(value)
        self.status_label.setText(main_text)
        self.subtask_label.setText(sub_text)

    def analysis_finished(self, outcome: BatchRunOutcome):
        logger.info("Analysis run finished.")
        all_results = outcome.results
        run_output_dir = outcome.output_dir or self.output_dir
        closing = self._close_requested

        if closing:
            # The user is quitting: no result dialogs, just finish and close.
            pass
        elif outcome.cancelled:
            QMessageBox.warning(
                self,
                "Analysis Cancelled",
                f"Analysis was cancelled.\n"
                f"Completed: {len(all_results)} video(s)\n"
                "The interrupted video can be resumed from its checkpoint."
            )
        elif outcome.failures:
            failure_lines = "\n".join(
                f"• {path.name}: {message}"
                for path, message in outcome.failures[:5]
            )
            QMessageBox.warning(
                self,
                "Analysis Completed with Errors",
                f"Completed: {len(all_results)} video(s)\n"
                f"Failed: {len(outcome.failures)} video(s)\n\n"
                f"{failure_lines}\n\nSaved to: {run_output_dir}"
            )
        elif outcome.quality_warnings:
            warning_lines = "\n".join(
                f"• {path.name}: {message}"
                for path, message in outcome.quality_warnings[:5]
            )
            QMessageBox.warning(
                self,
                "Analysis Requires Quality Review",
                f"Completed: {len(all_results)} video(s)\n\n"
                f"{warning_lines}\n\n"
                "Review the per-frame evidence and, when generated, the "
                "validation video before "
                f"interpreting these results.\n\nSaved to: {run_output_dir}"
            )
        else:
            QMessageBox.information(
                self,
                "Complete",
                f"Analysis finished!\nProcessed: {len(all_results)} video(s)\n"
                f"Saved to: {run_output_dir}"
            )
        self.set_ui_state(is_running=False)
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

        if closing:
            QTimer.singleShot(0, self.close)

    def set_ui_state(self, is_running: bool):
        self.start_button.setEnabled(not is_running)
        self.add_videos_button.setEnabled(not is_running)
        self.add_folder_button.setEnabled(not is_running)
        self.clear_list_button.setEnabled(not is_running)
        self.cancel_button.setEnabled(is_running)
        self.preview_button.setEnabled(not is_running and (self.current_settings is not None or len(self.batch_plan_cache) > 0))
        # Configuration and output-folder changes would race the running job.
        self.save_config_action.setEnabled(
            not is_running and self.current_settings is not None
        )
        self.load_config_action.setEnabled(not is_running)
        self.set_output_action.setEnabled(not is_running)

    def _analysis_running(self) -> bool:
        thread = self.analysis_thread
        return thread is not None and thread.isRunning()

    def closeEvent(self, event):
        if not self._analysis_running():
            self._detach_log_handler()
            event.accept()
            return

        if self._force_close:
            # Last resort after the worker failed to stop in time.
            self._stop_thread_forcefully()
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
        QTimer.singleShot(self.FORCE_QUIT_DELAY_MS, self._offer_force_quit)
        event.ignore()

    def _offer_force_quit(self):
        """The worker did not stop in time; let the user stop waiting."""
        if not (self._close_requested and self._analysis_running()):
            return
        if self._confirm_force_quit():
            self._force_close = True
            self.close()
        else:
            QTimer.singleShot(self.FORCE_QUIT_DELAY_MS, self._offer_force_quit)

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
        box.exec()
        return box.clickedButton() is force

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
