"""
gui/main_window.py
Main GUI Window.
Updated to use unified Core modules (VideoHandler, Models).
"""
import copy
import logging
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                             QListWidget, QMainWindow, QMenu, QMessageBox,
                             QProgressBar, QPushButton, QTextEdit, QVBoxLayout,
                             QWidget, QInputDialog)
from PyQt6.QtGui import QAction, QImage, QPixmap

# --- CORE IMPORTS ---
from core.models import AppConfig, BatchSettings
from core.video import VideoHandler, ReferenceFrameGenerator
from core.version import APP_NAME, APP_VERSION

# --- DIALOGS ---
from gui.roi_selector import RoiSelectorDialog, ReferenceLineDialog
from gui.settings_dialog import SettingsDialog
from gui.tuner_dialog import TunerDialog
from gui.tst_tuner import TstTunerDialog
from gui.worker import AnalysisWorker, BatchRunOutcome
from gui.roi_preview import ROIPreviewDialog
from gui.freestyle_config_dialog import FreestyleConfigDialog

# --- UTILS ---
from roi_manager import ROIManager
from mazes import AVAILABLE_MAZES, create_maze
from mazes.base_maze import BaseAnalysisResult, Maze
from mazes.tst import TST
from preflight_checker import PreflightChecker, ValidationDialog
from checkpoint_manager import CheckpointManager, ResourceManager

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
        self.batch_plan_cache: Dict[str, BatchSettings] = {}
        
        self.checkpoint_mgr = None
        
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
        logging.getLogger().setLevel(logging.INFO)
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
        settings_to_use = self.batch_plan_cache.get(first_vid.name, self.current_settings)
        
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
                if plan:
                    first_key = list(plan.keys())[0]
                    self.current_settings = plan[first_key]
                    self._infer_maze_from_settings(self.current_settings)
                count_matched = sum(1 for v in self.video_paths if v.name in plan)
                msg = (f"Loaded Batch Plan with {len(plan)} configurations.\n"
                       f"Matched to {count_matched} videos currently in list.")
            else:
                settings = BatchSettings.load_from_file(file_path)
                self.current_settings = settings
                self.batch_plan_cache = {} 
                self._infer_maze_from_settings(settings)
                msg = ("Master Template loaded.\nThis configuration will be applied to ALL videos.")

            self.update_button_states()
            logger.info(f"Configuration loaded from: {filename}")
            QMessageBox.information(self, "Configuration Loaded", msg)
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}", exc_info=True)
            QMessageBox.critical(self, "Load Failed", f"Failed to load configuration:\n{e}")

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
            checkpoint_mgr = CheckpointManager(self.output_dir)
            checkpoints = checkpoint_mgr.find_all_checkpoints()
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
        has_batch_match = any(v.name in self.batch_plan_cache for v in self.video_paths)
        
        if has_batch_match:
            reply = QMessageBox.question(self, "Batch Plan Detected", "Use loaded Batch Plan settings?", 
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
            if reply == QMessageBox.StandardButton.Yes:
                analysis_plan = {}
                for v in self.video_paths:
                    if v.name in self.batch_plan_cache: analysis_plan[v] = self.batch_plan_cache[v.name].copy()
                    elif self.current_settings: analysis_plan[v] = self.current_settings.copy()
                if analysis_plan: self.run_worker(analysis_plan); return

        elif self.current_settings:
            reply = QMessageBox.question(self, "Master Template Loaded", "Use Master Template for all videos?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
            if reply == QMessageBox.StandardButton.Yes: use_loaded_config = True

        try:
            base_settings = None

            if use_loaded_config:
                base_settings = self.current_settings
                if base_settings.algorithm_parameters:
                    self.maze.configure_algorithm_parameters(
                        base_settings.algorithm_parameters
                    )
                # Re-configure maze state from settings if needed
                if maze_choice == "Freestyle / Open Field":
                    zones = [(k, k.replace('_', ' ').title()) for k in base_settings.roi_manager.rois.keys() if k != 'outside']
                    self.maze.configure_zones(zones)
                elif isinstance(self.maze, TST):
                    count = sum(1 for k in base_settings.roi_manager.rois.keys() if k.startswith('mouse_'))
                    self.maze.configure_mice(count)

            else:
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
                            reply = QMessageBox.question(self, 'Calibration', 'Draw reference line?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                            if reply == QMessageBox.StandardButton.Yes:
                                # Core: Use VideoHandler
                                handler = VideoHandler(ref_video_path)
                                roi_manager.set_source_dimensions(
                                    handler.dimensions
                                )
                                # For line dialog, we need a pixmap. Get a frame from handler.
                                frame = handler.get_frame(0)
                                if frame is not None:
                                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    h, w, c = rgb.shape
                                    q_img = QImage(rgb.data, w, h, c*w, QImage.Format.Format_RGB888)
                                    ref_dialog = ReferenceLineDialog(
                                        QPixmap.fromImage(q_img),
                                        parent=self
                                    )
                                    if ref_dialog.exec() and ref_dialog.get_line_length() > 0:
                                        roi_manager.set_reference_length(ref_dialog.get_line_length(), "User Reference")
                    
                    config = AppConfig()
                    # Core: Reference generation
                    ref_frame = ReferenceFrameGenerator.generate(
                        ref_video_path, config.video.ref_frame_samples, use_cache=True
                    )
                    tuner_dialog = TunerDialog(str(ref_video_path), ref_frame, roi_manager, self)
                    if not tuner_dialog.exec(): return
                    detection_config = tuner_dialog.get_detection_config()
                    if not is_zone_free_mode: roi_manager.calculate_reference_length()
                    base_settings = BatchSettings(roi_manager, detection_config, 0.0)

                # 3. STANDARD CONFIGURATION
                else:
                    self.update_progress_detailed(10, "Configuration", "Defining regions of interest...")
                    roi_dialog = RoiSelectorDialog(str(ref_video_path), self.maze, self)
                    if not roi_dialog.exec(): return
                    roi_manager = roi_dialog.get_roi_manager()
                    
                    config = AppConfig()
                    self.update_progress_detailed(20, "Configuration", "Generating reference frame...")
                    QApplication.processEvents()
                    
                    # Core: Reference generation
                    ref_frame = ReferenceFrameGenerator.generate(
                        ref_video_path, config.video.ref_frame_samples, use_cache=True
                    )
                    
                    self.update_progress_detailed(35, "Configuration", "Tuning detection parameters...")
                    tuner_dialog = TunerDialog(str(ref_video_path), ref_frame, roi_manager, self)
                    if not tuner_dialog.exec(): return
                    detection_config = tuner_dialog.get_detection_config()
                    
                    roi_manager.calculate_reference_length()
                    base_settings = BatchSettings(roi_manager, detection_config, 0.0)

            # --- FINAL REVIEW ---
            self.update_progress_detailed(50, "Configuration", "Reviewing analysis parameters...")
            QApplication.processEvents()
            
            settings_dialog = SettingsDialog(str(ref_video_path), base_settings, is_verification=False, parent=self)
            
            if not settings_dialog.exec():
                logger.info("Settings cancelled.")
                self.update_progress_detailed(0, "Ready", "")
                return
                
            base_settings = settings_dialog.get_settings()
            base_settings.paradigm_id = self.maze.name
            base_settings.algorithm_parameters = self.maze.get_algorithm_parameters()
            self.current_settings = base_settings
            self.update_button_states()
            
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
        for i, video_path in enumerate(self.video_paths):
            logger.info(f"Verifying video {i+1}/{len(self.video_paths)}: {video_path.name}")
            try:
                while True:
                    dialog = SettingsDialog(str(video_path), current_settings.copy(), is_verification=True, parent=self)
                    dialog.setWindowTitle(f"Verify: {video_path.name}")
                    if not dialog.exec(): return {}
                    
                    action = dialog.result_action
                    if action == "accept":
                        final_settings = dialog.get_settings()
                        analysis_plan[video_path] = final_settings.copy()
                        current_settings = final_settings
                        break
                    elif action == "skip":
                        logger.info(f"Video '{video_path.name}' skipped")
                        break
                    elif action == "redraw":
                        roi_dialog = RoiSelectorDialog(str(video_path), self.maze, self)
                        if roi_dialog.exec():
                            new_roi_manager = roi_dialog.get_roi_manager()
                            current_settings = BatchSettings(
                                new_roi_manager,
                                current_settings.detection_config,
                                current_settings.scale_factor,
                                current_settings.start_time,
                                current_settings.end_time,
                                current_settings.create_timelapse,
                                current_settings.per_roi_times,
                                current_settings.paradigm_id,
                                current_settings.algorithm_parameters
                            )
            except Exception as e:
                logger.error(f"Could not verify {video_path.name}: {e}")
                continue
        
        if not analysis_plan:
            QMessageBox.warning(self, "No Videos", "All videos were skipped.")
            return {}
        
        reply = QMessageBox.question(self, 'Verification Complete', "Save this verification plan?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            filename, _ = QFileDialog.getSaveFileName(self, "Save Batch Plan", str(self.output_dir / "batch_plan.json") if self.output_dir else "batch_plan.json", "JSON Files (*.json)")
            if filename:
                plan_by_name = {p.name: s for p, s in analysis_plan.items()}
                BatchSettings.save_batch_plan(plan_by_name, Path(filename))
        
        return analysis_plan

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

        optimal_workers = ResourceManager.get_optimal_workers()
        logger.info(f"System optimal parallel workers: {optimal_workers}")
        
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
        
        self.analysis_thread.start()

    def _validate_analysis_plan(
        self,
        analysis_plan: Dict[Path, BatchSettings]
    ) -> bool:
        """Validate every per-video configuration, including loaded plans."""
        all_issues = []
        all_warnings = []
        expected_parameters = None

        for video_path, settings in analysis_plan.items():
            checker = PreflightChecker()
            issues, warnings = checker.validate_all(
                [video_path], settings, self.output_dir
            )
            all_issues.extend(
                f"{video_path.name}: {issue}" for issue in issues
            )
            all_warnings.extend(
                f"{video_path.name}: {warning}" for warning in warnings
            )

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
        
        if outcome.cancelled:
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
        
        if self.analysis_thread:
            self.analysis_thread.quit()
            self.analysis_thread.wait()
        self.analysis_thread = None
        self.analysis_worker = None

    def set_ui_state(self, is_running: bool):
        self.start_button.setEnabled(not is_running)
        self.add_videos_button.setEnabled(not is_running)
        self.add_folder_button.setEnabled(not is_running)
        self.clear_list_button.setEnabled(not is_running)
        self.cancel_button.setEnabled(is_running)
        self.preview_button.setEnabled(not is_running and (self.current_settings is not None or len(self.batch_plan_cache) > 0))

    def closeEvent(self, event):
        if self.analysis_thread and self.analysis_thread.isRunning():
            reply = QMessageBox.question(self, 'Analysis in Progress', "Quit and cancel analysis?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.cancel_analysis()
                if self.analysis_thread: self.analysis_thread.quit(); self.analysis_thread.wait()
                self._detach_log_handler()
                event.accept()
            else: event.ignore()
        else:
            self._detach_log_handler()
            event.accept()

    def _detach_log_handler(self):
        """Remove the Qt-backed handler before its widget is destroyed."""
        handler = getattr(self, "log_handler", None)
        if handler is None:
            return
        logging.getLogger().removeHandler(handler)
        handler.close()
        self.log_handler = None
