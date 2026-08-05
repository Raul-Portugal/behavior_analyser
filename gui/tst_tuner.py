"""
High-Performance TST Tuner Dialog with Matplotlib Blitting.
Optimized for real-time video playback and graph synchronization.
"""
import copy
import json
import cv2
import logging
import numpy as np
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QElapsedTimer
from PyQt6.QtGui import QGuiApplication, QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QProgressBar, QSizePolicy, QSlider, QSpinBox, QVBoxLayout,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from core.tst_motion_engine import TSTMotionConfig, TSTMotionEngine
from core.tst_events import smooth_tst_immobility_states
from roi_manager import ROI

logger = logging.getLogger(__name__)


class EnergyScanner(QThread):
    """
    Background thread for a bounded evidence scan.
    Only source frame indices are retained; preview images are decoded on
    demand so the tuner remains memory-bounded for long recordings.
    """
    progress = pyqtSignal(int)
    scan_complete = pyqtSignal(dict, list, tuple, float)
    
    def __init__(
        self,
        video_path,
        rois,
        motion_config,
        review_segments,
        preview_width=480,
        target_fps=12.0,
    ):
        super().__init__()
        self.video_path = video_path
        self.rois = dict(rois)
        self.motion_config = TSTMotionConfig(**motion_config.__dict__)
        self.review_segments = list(review_segments)
        self.preview_width = max(160, int(preview_width))
        self.target_fps = max(1.0, float(target_fps))
        self.state_data = {name: [] for name in self.rois}
        self.evidence_data = {name: [] for name in self.rois}
        self._capture = None

    @staticmethod
    def _scale_rois(rois, scale_x, scale_y):
        scaled = {}
        for name, roi in rois.items():
            points = roi.points.astype(np.float32)
            points[:, 0] *= scale_x
            points[:, 1] *= scale_y
            scaled[name] = ROI(np.rint(points).astype(np.int32), name)
        return scaled

    def cancel(self):
        """Request cancellation and unblock a pending local video read."""
        self.requestInterruption()
        capture = self._capture
        if capture is not None:
            capture.release()
        
    def run(self):
        cap = cv2.VideoCapture(str(self.video_path))
        self._capture = cap
        if not cap.isOpened():
            self._capture = None
            self.scan_complete.emit({}, [], (1, 1), 30.0)
            return
        
        # Get video properties
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < 2:
            cap.release()
            self._capture = None
            self.scan_complete.emit({}, [], (orig_w, orig_h), fps)
            return
        source_step = max(1, int(round(fps / self.target_fps)))
        analysis_fps = fps / source_step
        preview_height = max(
            1, int(round(orig_h * min(1.0, self.preview_width / orig_w)))
        )
        preview_width = max(
            1, int(round(orig_w * min(1.0, self.preview_width / orig_w)))
        )
        scale_x = preview_width / max(1, orig_w)
        scale_y = preview_height / max(1, orig_h)
        scaled_rois = self._scale_rois(self.rois, scale_x, scale_y)

        segments = []
        for start_frame, end_frame in self.review_segments:
            start = max(0, int(start_frame))
            end = min(total_frames, int(end_frame))
            if end - start > source_step:
                segments.append((start, end))
        if not segments:
            segments = [(0, total_frames)]
        expected_intervals = sum(
            max(0, (end - start - 1) // source_step)
            for start, end in segments
        )

        energy_data = {name: [] for name in self.rois}
        frame_indices = []
        count = 0
        
        try:
            for segment_number, (start, end) in enumerate(segments):
                if self.isInterruptionRequested():
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                ret, previous_source = cap.read()
                if not ret or previous_source is None:
                    continue
                previous = cv2.resize(
                    previous_source, (preview_width, preview_height),
                    interpolation=cv2.INTER_AREA,
                )
                engine = TSTMotionEngine(
                    scaled_rois,
                    self.motion_config,
                    fps=analysis_fps,
                )
                current_source_frame = start
                first_interval = True
                while (
                    current_source_frame + source_step < end
                    and not self.isInterruptionRequested()
                ):
                    current_source = None
                    for _ in range(source_step):
                        ret, current_source = cap.read()
                        if not ret:
                            break
                    if not ret or current_source is None:
                        break
                    current_source_frame += source_step
                    current = cv2.resize(
                        current_source, (preview_width, preview_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    observations = engine.calculate_observations(
                        current, previous
                    )
                    for name, observation in observations.items():
                        evidence = observation.to_dict()
                        if segment_number and first_interval:
                            evidence.update({
                                "state": "uncertain",
                                "confidence": 0.0,
                                "qc_flag": "review_segment_boundary",
                            })
                        contour = engine.get_last_mask_contour(name)
                        if contour:
                            evidence["mask_contour"] = [
                                [
                                    int(round(point[0] / scale_x)),
                                    int(round(point[1] / scale_y)),
                                ]
                                for point in contour
                            ]
                        evidence["source_frame"] = current_source_frame
                        evidence["timestamp_s"] = current_source_frame / fps
                        energy_data[name].append(observation.activity_score)
                        self.state_data[name].append(evidence["state"])
                        self.evidence_data[name].append(evidence)
                    frame_indices.append(current_source_frame)
                    previous = current
                    first_interval = False
                    count += 1
                    if count % 25 == 0 or count == expected_intervals:
                        self.progress.emit(int(
                            count / max(1, expected_intervals) * 100
                        ))
        finally:
            cap.release()
            self._capture = None
        if not self.isInterruptionRequested():
            self.scan_complete.emit(
                energy_data,
                frame_indices,
                (orig_w, orig_h),
                fps,
            )
    
class OptimizedCanvas(FigureCanvasQTAgg):
    """
    Custom canvas with blitting support for high-performance updates.
    """
    def __init__(self, figure):
        super().__init__(figure)
        self.background = None
        self.cursor_background = None
        self._bg_cache = {}  # Cache multiple backgrounds for different states
        

class TstTunerDialog(QDialog):
    """
    High-performance tuner dialog with optimized video playback
    and Matplotlib blitting for real-time graph updates.
    """
    
    def __init__(self, video_path, roi_manager, tst_maze_instance, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TST Parameter Tuner - High Performance")
        self.setMinimumSize(820, 600)
        screen = QGuiApplication.primaryScreen()
        if screen:
            available = screen.availableGeometry()
            self.resize(
                min(1400, max(820, available.width() - 80)),
                min(900, max(600, available.height() - 80)),
            )
        else:
            self.resize(1400, 900)
        
        self.video_path = video_path
        self.roi_manager = roi_manager
        self.tst = tst_maze_instance
        self.preview_motion_config = copy.deepcopy(tst_maze_instance.motion_config)
        
        # Initialize ROI selection
        self.roi_dict = {cat: rois[0] for cat, rois in roi_manager.rois.items() if rois}
        
        # Focus on Mouse 1 or first available
        if "mouse_1" in self.roi_dict:
            self.selected_mouse = "mouse_1"
        elif self.roi_dict:
            self.selected_mouse = list(self.roi_dict.keys())[0]
        else:
            self.selected_mouse = None
        
        # Data storage
        self.energy_data = {}
        self.evidence_data = {}
        self.frame_buffer = []  # Source frame indices for preview decoding
        self.decoded_frame_cache = {}  # Cache for decoded frames
        self.max_decoded_cache = 200  # Keep more decoded frames
        self.orig_dims = (1, 1)
        self.fps = 30.0
        self.analysis_fps = 12.0
        self.scan_len = 0
        
        # Playback state
        self.is_playing = False
        self.current_frame = 0
        self.play_timer = QTimer()
        self.play_timer.setTimerType(Qt.TimerType.PreciseTimer)  # Use precise timer for accurate FPS
        self.play_timer.timeout.connect(self.on_playback_tick)
        
        # Performance timer for accurate FPS
        self.elapsed_timer = QElapsedTimer()
        self.frame_accumulator = 0.0
        
        # Matplotlib elements for blitting
        self.figure = None
        self.canvas = None
        self.ax = None
        self.line_trace = None
        self.line_thresh = None
        self.fill_immobile = None
        self.fill_pause = None
        self.cursor_line = None
        
        # Blitting backgrounds
        self.graph_background = None
        self.cursor_bbox = None
        
        # Pre-computed data for performance
        self.timestamps = []
        self.filtered_immobile = None
        self.raw_immobile = None
        self.state_data = {}
        
        # Frame cache for processed video frames
        self.processed_frame_cache = {}
        self.cache_size = 100  # Keep last N processed frames in cache
        self.preview_cap = cv2.VideoCapture(str(self.video_path))
        
        self.init_ui()
        
        if self.selected_mouse:
            self.start_scan()
        else:
            if hasattr(self, 'image_label'):
                self.image_label.setText("Error: No ROI defined")
    
    def init_ui(self):
        """Initialize the user interface with optimized layout."""
        main_layout = QVBoxLayout()
        
        # === TOP SECTION: VIDEO AND GRAPH ===
        top_layout = QHBoxLayout()
        
        # Video Preview Section
        video_group = QGroupBox("Visual Confirmation")
        video_layout = QVBoxLayout()
        
        self.image_label = QLabel("Initializing...")
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setMaximumSize(640, 480)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.image_label.setStyleSheet("""
            QLabel {
                background: #1a1a1a;
                color: #cccccc;
                border: 2px solid #444;
                border-radius: 4px;
            }
        """)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setScaledContents(False)
        
        video_layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Frame counter
        self.frame_label = QLabel("Frame: 0 / 0")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setStyleSheet("font-size: 12px; color: #888;")
        video_layout.addWidget(self.frame_label)

        self.mouse_selector = QComboBox()
        for mouse_name in sorted(self.roi_dict):
            self.mouse_selector.addItem(
                mouse_name.replace("_", " ").title(),
                mouse_name,
            )
        if self.selected_mouse:
            selected_index = self.mouse_selector.findData(
                self.selected_mouse
            )
            if selected_index >= 0:
                self.mouse_selector.setCurrentIndex(selected_index)
        self.mouse_selector.setEnabled(False)
        self.mouse_selector.currentIndexChanged.connect(
            self.on_mouse_change
        )
        video_layout.addWidget(self.mouse_selector)
        
        # FPS counter
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fps_label.setStyleSheet("font-size: 12px; color: #888;")
        video_layout.addWidget(self.fps_label)
        
        video_group.setLayout(video_layout)
        
        # Graph Section
        display_name = self.selected_mouse.replace('_', ' ').title() if self.selected_mouse else "None"
        self.graph_group = QGroupBox(
            f"Active-Motion Evidence: {display_name}"
        )
        graph_layout = QVBoxLayout()
        
        # Initialize Matplotlib with optimization settings
        self.figure = Figure(figsize=(10, 5), dpi=80)
        self.canvas = OptimizedCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Configure axes
        self.ax.set_xlabel("Time (s)", fontsize=10)
        self.ax.set_ylabel("Active-Motion Evidence", fontsize=10)
        self.ax.grid(True, alpha=0.3, linewidth=0.5)
        self.figure.tight_layout(pad=1.0)
        
        graph_layout.addWidget(self.canvas)
        
        # Add zoom controls
        zoom_layout = QHBoxLayout()
        zoom_label = QLabel("Graph Zoom:")
        zoom_label.setStyleSheet("font-size: 11px;")
        
        self.zoom_in_btn = QPushButton("🔍+")
        self.zoom_in_btn.setMaximumWidth(40)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        
        self.zoom_out_btn = QPushButton("🔍-")
        self.zoom_out_btn.setMaximumWidth(40)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        
        self.reset_zoom_btn = QPushButton("Reset")
        self.reset_zoom_btn.setMaximumWidth(60)
        self.reset_zoom_btn.clicked.connect(self.reset_zoom)
        
        zoom_layout.addWidget(zoom_label)
        zoom_layout.addWidget(self.zoom_in_btn)
        zoom_layout.addWidget(self.zoom_out_btn)
        zoom_layout.addWidget(self.reset_zoom_btn)
        zoom_layout.addStretch()
        
        graph_layout.addLayout(zoom_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444;
                border-radius: 3px;
                text-align: center;
                background-color: #222;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 2px;
            }
        """)
        graph_layout.addWidget(self.progress_bar)
        
        self.graph_group.setLayout(graph_layout)
        
        top_layout.addWidget(video_group)
        top_layout.addWidget(self.graph_group, 1)  # Graph takes more space
        main_layout.addLayout(top_layout)
        
        # === MIDDLE SECTION: CONTROLS ===
        controls_layout = QHBoxLayout()
        
        # Playback Controls
        playback_group = QGroupBox("Playback Controls")
        playback_layout = QHBoxLayout()
        
        self.play_button = QPushButton("▶ Play")
        self.play_button.setCheckable(True)
        self.play_button.setEnabled(False)
        self.play_button.setStyleSheet("""
            QPushButton {
                min-width: 80px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #d32f2f;
            }
        """)
        self.play_button.clicked.connect(self.toggle_playback)
        
        # Scrub slider with enhanced style
        self.scrub_slider = QSlider(Qt.Orientation.Horizontal)
        self.scrub_slider.setEnabled(False)
        self.scrub_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #333;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 18px;
                background: #4CAF50;
                border-radius: 9px;
                margin: -6px 0;
            }
            QSlider::handle:horizontal:hover {
                background: #66BB6A;
            }
        """)
        self.scrub_slider.valueChanged.connect(self.on_scrub)
        
        # Speed control
        self.speed_label = QLabel("Speed: 1.0x")
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(25, 200)  # 0.25x to 2.0x
        self.speed_slider.setValue(100)
        self.speed_slider.setFixedWidth(100)
        self.speed_slider.valueChanged.connect(self.on_speed_change)
        
        playback_layout.addWidget(self.play_button)
        playback_layout.addWidget(self.scrub_slider, 1)
        playback_layout.addWidget(self.speed_label)
        playback_layout.addWidget(self.speed_slider)
        
        playback_group.setLayout(playback_layout)
        controls_layout.addWidget(playback_group, 2)
        
        # Parameter Controls
        param_group = QGroupBox("Detection Parameters")
        param_layout = QHBoxLayout()
        
        # The classifier is automatic; this disabled indicator replaces the
        # legacy manually tuned pixel-difference threshold.
        energy_layout = QVBoxLayout()
        self.energy_label = QLabel(
            "Classifier: rigid/articulated motion decomposition"
        )
        self.energy_label.setStyleSheet("font-weight: bold;")
        self.energy_slider = QSlider(Qt.Orientation.Horizontal)
        self.energy_slider.setRange(0, 100)
        self.energy_slider.setValue(50)
        self.energy_slider.setEnabled(False)
        energy_layout.addWidget(self.energy_label)
        energy_layout.addWidget(self.energy_slider)
        
        # Time Threshold
        time_layout = QVBoxLayout()
        self.time_label = QLabel("Min Immobility: 0.5s")
        self.time_label.setStyleSheet("font-weight: bold;")
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(1, 50)  # 0.1s to 5.0s
        self.time_slider.setValue(int(round(
            np.clip(self.tst.min_time_seconds * 10.0, 1, 50)
        )))
        self.time_slider.valueChanged.connect(self.on_param_change)
        time_layout.addWidget(self.time_label)
        time_layout.addWidget(self.time_slider)
        
        param_layout.addLayout(energy_layout)
        param_layout.addLayout(time_layout)
        param_group.setLayout(param_layout)
        
        controls_layout.addWidget(param_group, 1)
        main_layout.addLayout(controls_layout)

        self._add_advanced_controls(main_layout)
        
        # === BOTTOM SECTION: ACTION BUTTONS ===
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.accept_button = QPushButton("Accept Parameters")
        self.accept_button.setEnabled(False)
        self.accept_button.setStyleSheet("""
            QPushButton {
                min-width: 120px;
                padding: 10px;
                font-weight: bold;
                background-color: #4CAF50;
                color: white;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #66BB6A;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
        """)
        self.accept_button.clicked.connect(self.accept)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                min-width: 120px;
                padding: 10px;
                border-radius: 4px;
            }
        """)
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.accept_button)
        button_layout.addWidget(self.cancel_button)
        
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def _add_advanced_controls(self, main_layout):
        """Keep expert classifier settings available without cluttering basics."""
        group = QGroupBox("Advanced classifier settings")
        group.setCheckable(True)
        group.setChecked(False)
        form = QFormLayout()
        self.advanced_controls = {}
        fields = (
            ("segmentation_contrast", "Segmentation contrast", 0.0, 255.0, 1.0, 1),
            ("min_mouse_area_ratio", "Minimum body area ratio", 0.0001, 0.5, 0.0001, 4),
            ("max_mouse_area_ratio", "Maximum body area ratio", 0.01, 1.0, 0.001, 3),
            ("static_rigid_threshold", "Static-motion threshold", 0.0, 0.1, 0.0005, 4),
            ("active_residual_threshold", "Articulated-motion threshold", 0.001, 0.2, 0.001, 3),
            ("passive_pendulum_score_threshold", "Passive-swing threshold", 0.0, 1.0, 0.01, 2),
            ("temporal_window_seconds", "Motion history window (s)", 0.25, 10.0, 0.25, 2),
        )
        for name, label, low, high, step, decimals in fields:
            control = QDoubleSpinBox()
            control.setRange(low, high)
            control.setSingleStep(step)
            control.setDecimals(decimals)
            control.setValue(float(getattr(self.preview_motion_config, name)))
            control.valueChanged.connect(self._on_advanced_changed)
            self.advanced_controls[name] = control
            form.addRow(label, control)
        buttons = QHBoxLayout()
        rescan = QPushButton("Rescan preview")
        rescan.clicked.connect(self._rescan_preview)
        save_profile = QPushButton("Save profile…")
        save_profile.clicked.connect(self._save_profile)
        load_profile = QPushButton("Load profile…")
        load_profile.clicked.connect(self._load_profile)
        buttons.addWidget(rescan)
        buttons.addWidget(save_profile)
        buttons.addWidget(load_profile)
        form.addRow(buttons)
        self.advanced_note = QLabel(
            "Advanced changes apply to the full analysis only after acceptance. "
            "Use Rescan preview to inspect their effect."
        )
        self.advanced_note.setWordWrap(True)
        form.addRow(self.advanced_note)
        group.setLayout(form)
        main_layout.addWidget(group)

    def _on_advanced_changed(self):
        for name, control in self.advanced_controls.items():
            setattr(self.preview_motion_config, name, float(control.value()))
        if self.preview_motion_config.min_mouse_area_ratio >= self.preview_motion_config.max_mouse_area_ratio:
            self.advanced_note.setText("Minimum body area must be lower than maximum body area.")
            return
        self.advanced_note.setText("Settings changed. Rescan preview to inspect them; accept to use them in analysis.")

    def _rescan_preview(self):
        if hasattr(self, "scanner") and self.scanner.isRunning():
            return
        try:
            self.preview_motion_config.validate()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid classifier settings", str(exc))
            return
        self.energy_data = {}
        self.evidence_data = {}
        self.processed_frame_cache.clear()
        self.start_scan()

    def _save_profile(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save TST profile", "", "JSON files (*.json)")
        if not filename:
            return
        payload = {
            "engine_id": TSTMotionEngine.ENGINE_ID,
            "min_immobility_seconds": self.time_slider.value() / 10.0,
            "motion_config": self.preview_motion_config.__dict__,
        }
        try:
            with open(filename, "w", encoding="utf-8") as profile_file:
                json.dump(payload, profile_file, indent=2)
        except OSError as exc:
            QMessageBox.warning(self, "Could not save profile", str(exc))

    def _load_profile(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load TST profile", "", "JSON files (*.json)")
        if not filename:
            return
        try:
            with open(filename, encoding="utf-8") as profile_file:
                payload = json.load(profile_file)
            if payload.get("engine_id") != TSTMotionEngine.ENGINE_ID:
                raise ValueError("This profile was made for a different TST engine.")
            config = TSTMotionConfig(**payload["motion_config"])
            config.validate()
            if config.min_mouse_area_ratio >= config.max_mouse_area_ratio:
                raise ValueError("Profile has an invalid body-area range.")
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Could not load profile", str(exc))
            return
        self.preview_motion_config = config
        for name, control in self.advanced_controls.items():
            control.blockSignals(True)
            control.setValue(float(getattr(config, name)))
            control.blockSignals(False)
        self.time_slider.setValue(int(round(float(payload.get("min_immobility_seconds", 0.5)) * 10)))
        self._on_advanced_changed()
    
    def start_scan(self):
        """Start background video scanning."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Get video info to determine scanning strategy
        cap = cv2.VideoCapture(str(self.video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        
        review_segments = self._build_review_segments(total_frames, fps)
        reviewed_seconds = sum(end - start for start, end in review_segments) / fps
        self.image_label.setText(
            f"Preparing a fast {reviewed_seconds:.0f}s representative review…"
        )
        
        self.scanner = EnergyScanner(
            self.video_path,
            self.roi_dict,
            self.preview_motion_config,
            review_segments,
        )
        self.scanner.progress.connect(self.on_scan_progress)
        self.scanner.scan_complete.connect(self.on_scan_complete)
        self.scanner.start()
    
    def on_scan_progress(self, value):
        """Update progress bar during scanning."""
        self.progress_bar.setValue(value)
    
    def on_scan_complete(self, energy_data, frame_buffer, orig_dims, fps):
        """Handle scan completion and initialize visualization."""
        if not energy_data or not frame_buffer:
            self.image_label.setText("Error: Failed to scan video")
            self.progress_bar.setVisible(False)
            return
        
        self.energy_data = energy_data
        self.state_data = dict(self.scanner.state_data)
        self.evidence_data = dict(self.scanner.evidence_data)
        self.frame_buffer = frame_buffer
        self.orig_dims = orig_dims
        self.fps = fps
        self.analysis_fps = fps / max(1, int(round(fps / self.scanner.target_fps)))
        self.scan_len = len(frame_buffer)
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Enable controls
        self.accept_button.setEnabled(True)
        self.play_button.setEnabled(True)
        self.scrub_slider.setEnabled(True)
        self.mouse_selector.setEnabled(len(self.roi_dict) > 1)
        
        # Set slider range based on actual frames
        self.scrub_slider.setRange(0, self.scan_len - 1)
        
        trace = energy_data[self.selected_mouse]
        
        # Show scan info
        duration = sum(
            end - start for start, end in self.scanner.review_segments
        ) / self.fps
        self.frame_label.setText(
            f"Fast review: {duration:.1f}s sampled at {self.analysis_fps:.1f} fps "
            f"({self.scan_len} motion intervals)"
        )
        
        # Initialize plot with blitting support
        self.init_plot_optimized(trace)
        
        # Compute initial parameters
        self.on_param_change()
        
        # Show first frame
        self.update_frame(0)

    @staticmethod
    def _build_review_segments(total_frames, fps):
        """Return short start/middle/end windows for a responsive tuner."""
        window = max(1, int(round(fps * 20.0)))
        if total_frames <= window * 3:
            return [(0, total_frames)]
        middle = max(0, total_frames // 2 - window // 2)
        return [
            (0, window),
            (middle, min(total_frames, middle + window)),
            (max(0, total_frames - window), total_frames),
        ]

    def on_mouse_change(self, _index):
        """Switch the evidence trace and overlay to another mouse."""
        mouse_name = self.mouse_selector.currentData()
        if not mouse_name or mouse_name not in self.energy_data:
            return
        try:
            self.selected_mouse = mouse_name
            self.graph_group.setTitle(
                "Active-Motion Evidence: "
                f"{mouse_name.replace('_', ' ').title()}"
            )
            self.processed_frame_cache.clear()
            self.init_plot_optimized(self.energy_data[mouse_name])
            self.on_param_change()
            self.update_frame(min(self.current_frame, self.scan_len - 1))
        except Exception as exc:
            logger.exception("Could not switch TST tuner mouse")
            self.image_label.setText(
                f"Could not display {mouse_name.replace('_', ' ')}: {exc}"
            )
    
    def init_plot_optimized(self, trace):
        """
        Initialize the plot with blitting optimization in mind.
        Sets up static elements and prepares for dynamic updates.
        """
        # Existing fill artists belong to the previous trace. Remove them
        # before clearing the axes; after ``Axes.clear`` they are already
        # detached and a second remove can raise inside the Qt callback.
        self._clear_fill_regions()
        self.ax.clear()
        self.fill_immobile = None
        self.fill_pause = None
        
        # Pre-compute timestamps
        self.timestamps = np.array([
            frame / self.fps for frame in self.frame_buffer[:len(trace)]
        ])
        
        # Plot active-motion evidence trace (static)
        self.line_trace, = self.ax.plot(
            self.timestamps, trace,
            'k-', linewidth=1, alpha=0.7,
            label="Active-motion evidence"
        )
        
        # Initialize threshold line (dynamic but less frequent)
        self.line_thresh = self.ax.axhline(
            y=0, color='blue', linestyle='--',
            linewidth=1.5, label="Reference level", animated=True
        )
        
        # Initialize cursor line (dynamic and frequent)
        self.cursor_line = self.ax.axvline(
            x=0, color='magenta', linewidth=2,
            alpha=0.8, label="Current Frame", animated=True  # Mark as animated
        )
        
        # Setup axes
        self.ax.set_xlim(0, self.timestamps[-1] if len(self.timestamps) > 0 else 1)
        y_max = max(100.0, max(trace) * 1.1) if trace else 100.0
        self.ax.set_ylim(0, y_max)
        
        # Add legend
        self.ax.legend(loc='upper right', fontsize=9, framealpha=0.8)
        
        # Draw canvas once to render everything
        self.canvas.draw()
        
        # CRITICAL: Store the background AFTER drawing but BEFORE any animations
        # This captures everything except the animated artists
        self.graph_background = self.canvas.copy_from_bbox(self.ax.bbox)
        
        # Store cursor bbox for efficient updates
        self.cursor_bbox = self.ax.bbox
        
        # Initialize the cursor at position 0
        self.update_cursor_blit(0)
    
    def on_param_change(self):
        """
        Handle parameter changes (thresholds).
        This requires updating the filled regions, which is more expensive.
        """
        if not self.energy_data or self.line_thresh is None:
            return
        
        time_thresh = self.time_slider.value() / 10.0
        
        # Update labels
        self.energy_label.setText(
            "Classifier: rigid/articulated motion decomposition"
        )
        self.time_label.setText(f"Min Immobility: {time_thresh:.1f}s")
        
        # Calculate immobility zones
        trace = self.energy_data[self.selected_mouse]
        min_frames = max(1, int(round(time_thresh * self.analysis_fps)))
        gap_frames = max(
            0, int(round(self.tst.max_motion_gap_seconds * self.analysis_fps))
        )
        
        states = self.state_data.get(self.selected_mouse, [])
        self.raw_immobile = np.array([
            state in {"immobile_static", "passive_swing"}
            for state in states
        ])
        smoothed = smooth_tst_immobility_states(
            states, min_frames, gap_frames
        )
        self.filtered_immobile = np.array([
            value is True for value in smoothed
        ])
        uncertain = np.array([value is None for value in smoothed])
        
        self._clear_fill_regions()
        
        # Create new fill regions
        y_max = max(100.0, max(trace) * 1.1) if trace else 100.0
        
        # Immobile zones (red)
        self.fill_immobile = self.ax.fill_between(
            self.timestamps, 0, y_max,
            where=self.filtered_immobile,
            color='red', alpha=0.25,
            interpolate=True, step='mid'
        )
        
        # Unscorable/ambiguous zones (yellow)
        self.fill_pause = self.ax.fill_between(
            self.timestamps, 0, y_max,
            where=uncertain,
            color='yellow', alpha=0.2,
            interpolate=True, step='mid'
        )
        
        # Update threshold line position (it's animated so won't be in background)
        self.line_thresh.set_ydata([50.0, 50.0])
        
        # Full redraw required for fill changes
        self.canvas.draw()
        
        # CRITICAL: Re-cache the background after the fills are drawn
        # but make sure animated artists (cursor, threshold) are not included
        # Temporarily hide animated artists
        cursor_visible = self.cursor_line.get_visible()
        thresh_visible = self.line_thresh.get_visible()
        self.cursor_line.set_visible(False)
        self.line_thresh.set_visible(False)
        
        # Draw and cache
        self.canvas.draw()
        self.graph_background = self.canvas.copy_from_bbox(self.ax.bbox)
        
        # Restore animated artists
        self.cursor_line.set_visible(cursor_visible)
        self.line_thresh.set_visible(thresh_visible)
        
        # Now draw the animated elements on top
        self.canvas.restore_region(self.graph_background)
        self.ax.draw_artist(self.line_thresh)
        self.ax.draw_artist(self.cursor_line)
        self.canvas.blit(self.ax.bbox)
        
        # Clear frame cache since parameters changed
        self.processed_frame_cache.clear()
        
        # Update current frame visualization
        self.update_frame(self.current_frame)

    def _clear_fill_regions(self):
        """Remove plot fills defensively across trace/axis changes."""
        for attribute in ("fill_immobile", "fill_pause"):
            artist = getattr(self, attribute, None)
            if artist is not None:
                try:
                    artist.remove()
                except (ValueError, NotImplementedError):
                    # Matplotlib already detached the artist when the axes
                    # were cleared. The replacement trace can proceed.
                    pass
            setattr(self, attribute, None)
    
    def update_cursor_blit(self, frame_idx):
        """
        Ultra-fast cursor update using blitting.
        Only redraws the cursor line without touching the rest of the graph.
        """
        if not self.graph_background or not self.cursor_line:
            return
        
        # Calculate time position
        if frame_idx >= len(self.timestamps):
            return
        time_pos = self.timestamps[frame_idx]
        
        # Restore background (removes old cursor)
        self.canvas.restore_region(self.graph_background)
        
        # Update cursor position
        self.cursor_line.set_xdata([time_pos, time_pos])
        
        # Redraw ONLY the artists that changed
        # This is the key optimization - we only redraw the cursor
        self.ax.draw_artist(self.cursor_line)
        
        # Blit only the axes bbox (not the whole figure)
        self.canvas.blit(self.ax.bbox)
        
        # No need for flush_events during playback - it adds overhead
        # Only flush during manual scrubbing
        if not self.is_playing:
            self.canvas.flush_events()
    
    def get_decoded_frame(self, frame_idx):
        """
        Decode a source frame on demand, using the preview cache when possible.
        """
        if frame_idx in self.decoded_frame_cache:
            return self.decoded_frame_cache[frame_idx]
        
        if frame_idx >= len(self.frame_buffer):
            return None
        
        source_frame_idx = int(self.frame_buffer[frame_idx])
        if not self.preview_cap.isOpened():
            self.preview_cap = cv2.VideoCapture(str(self.video_path))
        self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame_idx)
        ret, frame = self.preview_cap.read()
        if not ret or frame is None:
            return None
        height, width = frame.shape[:2]
        if height > 0 and width > 0:
            target_height = 480
            target_width = max(1, int(round(
                width * target_height / height
            )))
            frame = cv2.resize(
                frame,
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR,
            )
        
        # Cache decoded frame
        self.decoded_frame_cache[frame_idx] = frame
        
        # Limit cache size
        if len(self.decoded_frame_cache) > self.max_decoded_cache:
            # Remove frames furthest from current position
            keys = list(self.decoded_frame_cache.keys())
            keys.sort(key=lambda k: abs(k - frame_idx))
            for key in keys[self.max_decoded_cache:]:
                del self.decoded_frame_cache[key]
        
        return frame
    
    def update_frame(self, frame_idx):
        """
        Update video frame with ROI visualization.
        Uses on-demand decoding and bounded caches for smooth playback.
        """
        if not self.frame_buffer or frame_idx >= len(self.frame_buffer):
            return
        
        # Check processed frame cache first
        cache_key = (self.selected_mouse, frame_idx,
                    self.energy_slider.value() if hasattr(self, 'energy_slider') else 0,
                    self.time_slider.value() if hasattr(self, 'time_slider') else 0)
        
        if cache_key in self.processed_frame_cache:
            pixmap = self.processed_frame_cache[cache_key]
        else:
            # Get decoded frame
            frame = self.get_decoded_frame(frame_idx)
            if frame is None:
                return
            
            frame = frame.copy()
            
            # Get current state
            if self.energy_data and self.selected_mouse in self.energy_data:
                trace = self.energy_data[self.selected_mouse]
                if frame_idx < len(trace):
                    energy_val = trace[frame_idx]
                    
                    behavior_trace = self.state_data.get(
                        self.selected_mouse, []
                    )
                    behavior_state = (
                        behavior_trace[frame_idx]
                        if frame_idx < len(behavior_trace)
                        else "uncertain"
                    )
                    is_immobile = (
                        self.filtered_immobile[frame_idx]
                        if (
                            self.filtered_immobile is not None
                            and frame_idx < len(self.filtered_immobile)
                        )
                        else False
                    )

                    if behavior_state == "uncertain":
                        color = (0, 165, 255)
                        status = "UNCERTAIN"
                    elif is_immobile:
                        if behavior_state == "passive_swing":
                            color = (180, 0, 180)
                            status = "PASSIVE SWING"
                        else:
                            color = (0, 0, 255)
                            status = "IMMOBILE STATIC"
                    elif behavior_state in {
                        "immobile_static", "passive_swing"
                    }:
                        color = (0, 215, 255)
                        status = "SHORT PAUSE"
                    else:
                        color = (0, 255, 0)
                        status = "ACTIVE ESCAPE"
                    
                    # Draw ROI visualization
                    roi = self.roi_dict[self.selected_mouse]
                    orig_w, orig_h = self.orig_dims
                    preview_h, preview_w = frame.shape[:2]
                    
                    if orig_w > 0 and orig_h > 0:
                        # Scale ROI points to preview dimensions
                        scale_x = preview_w / orig_w
                        scale_y = preview_h / orig_h
                        scaled_pts = (roi.points * [scale_x, scale_y]).astype(np.int32)
                        
                        # Create semi-transparent overlay
                        overlay = frame.copy()
                        cv2.fillPoly(overlay, [scaled_pts], color)
                        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                        
                        # Draw ROI border
                        cv2.polylines(frame, [scaled_pts], True, color, 2, cv2.LINE_AA)

                        # The white outline is the actual segmented mouse body
                        # used by the classifier, not an inferred ROI fill.
                        evidence = self.evidence_data.get(
                            self.selected_mouse, []
                        )
                        if frame_idx < len(evidence):
                            contour = evidence[frame_idx].get("mask_contour", [])
                            if len(contour) >= 3:
                                body = np.asarray(contour, dtype=np.float32)
                                body *= [scale_x, scale_y]
                                body = np.rint(body).astype(np.int32)
                                body_overlay = frame.copy()
                                cv2.fillPoly(body_overlay, [body], color)
                                cv2.addWeighted(
                                    body_overlay, 0.35, frame, 0.65, 0, frame
                                )
                                cv2.polylines(
                                    frame, [body], True, (255, 255, 255), 2,
                                    cv2.LINE_AA,
                                )
                        
                        # Add status text with background
                        confidence = 0.0
                        qc_flag = ""
                        if frame_idx < len(self.evidence_data.get(self.selected_mouse, [])):
                            record = self.evidence_data[self.selected_mouse][frame_idx]
                            confidence = float(record.get("confidence", 0.0))
                            qc_flag = str(record.get("qc_flag", ""))
                        text = f"{status} | Evidence: {energy_val:.1f} | Confidence: {confidence:.2f}"
                        if qc_flag:
                            text += f" | QC: {qc_flag}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        thickness = 2
                        
                        (text_width, text_height), baseline = cv2.getTextSize(
                            text, font, font_scale, thickness
                        )
                        
                        text_x = max(10, min(scaled_pts[:, 0].min(), preview_w - text_width - 10))
                        text_y = max(30, scaled_pts[:, 1].min() - 10)
                        
                        cv2.rectangle(frame,
                                    (text_x - 5, text_y - text_height - 5),
                                    (text_x + text_width + 5, text_y + baseline),
                                    (0, 0, 0), -1)
                        
                        cv2.putText(frame, text, (text_x, text_y),
                                  font, font_scale, (255, 255, 255),
                                  thickness, cv2.LINE_AA)
            
            # Convert BGR to RGB for Qt
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            
            # Create QImage and pixmap
            qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            # Scale to fit label if needed
            if pixmap.width() > self.image_label.width() or pixmap.height() > self.image_label.height():
                pixmap = pixmap.scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation
                )
            
            # Cache the processed pixmap
            self.processed_frame_cache[cache_key] = pixmap
            
            # Limit cache size
            if len(self.processed_frame_cache) > self.cache_size:
                keys_to_remove = list(self.processed_frame_cache.keys())[:-self.cache_size]
                for key in keys_to_remove:
                    del self.processed_frame_cache[key]
        
        # Display the pixmap
        self.image_label.setPixmap(pixmap)
        
        # Update frame counter with time
        source_frame = int(self.frame_buffer[frame_idx])
        time_sec = source_frame / self.fps
        self.frame_label.setText(
            f"Frame: {source_frame} | Reviewed interval: "
            f"{frame_idx + 1} / {self.scan_len} | Time: {time_sec:.1f}s"
        )
        
        # Store current frame index
        self.current_frame = frame_idx
    
    def on_scrub(self, value):
        """
        Handle scrubbing with optimized updates.
        Uses blitting for cursor, full frame update for video.
        """
        if self.is_playing:
            # Don't update during playback (handled by timer)
            return
        
        # Update video
        self.update_frame(value)
        
        # Update cursor with blitting
        self.update_cursor_blit(value)
    
    def on_speed_change(self, value):
        """Handle playback speed changes."""
        speed = value / 100.0
        self.speed_label.setText(f"Speed: {speed:.2f}x")
        
        if self.is_playing:
            # Restart timer with new interval
            self.play_timer.stop()
            interval = int(1000 / (self.analysis_fps * speed))
            self.play_timer.start(max(1, interval))
    
    def toggle_playback(self):
        """Toggle video playback with accurate timing."""
        if self.play_button.isChecked():
            # Start playback
            self.is_playing = True
            self.play_button.setText("⏸ Pause")
            
            # Calculate timer interval based on speed
            speed = self.speed_slider.value() / 100.0
            interval = int(1000 / (self.analysis_fps * speed))
            
            # Start timer and elapsed timer
            self.elapsed_timer.start()
            self.frame_accumulator = 0.0
            self.play_timer.start(max(1, interval))
        else:
            # Stop playback
            self.is_playing = False
            self.play_button.setText("▶ Play")
            self.play_timer.stop()
            self.fps_label.setText("FPS: 0.0")
    
    def on_playback_tick(self):
        """
        Handle playback timer tick with frame-accurate timing.
        Optimized for minimal overhead during real-time playback.
        """
        if not self.is_playing:
            return
        
        # Use elapsed timer for frame-accurate playback
        elapsed_ms = self.elapsed_timer.elapsed()
        speed = self.speed_slider.value() / 100.0
        
        # Calculate which frame we should be showing based on elapsed time
        target_frame = int((elapsed_ms / 1000.0) * self.analysis_fps * speed)
        
        # Skip frames if we're behind (frame dropping for smooth playback)
        if target_frame > self.current_frame:
            new_frame = min(target_frame, self.scan_len - 1)
        else:
            # We're ahead or on time, just advance one frame
            new_frame = min(self.current_frame + 1, self.scan_len - 1)
        
        if new_frame >= self.scan_len - 1:
            # Reached end, stop playback
            self.play_button.setChecked(False)
            self.toggle_playback()
            return
        
        # Pre-decode next few frames for smoother playback
        if self.is_playing:
            for i in range(new_frame + 1, min(new_frame + 5, self.scan_len)):
                if i not in self.decoded_frame_cache:
                    self.get_decoded_frame(i)
        
        # Update slider without triggering on_scrub
        self.scrub_slider.blockSignals(True)
        self.scrub_slider.setValue(new_frame)
        self.scrub_slider.blockSignals(False)
        
        # Update video frame
        self.update_frame(new_frame)
        
        # Update cursor with blitting (ultra-fast)
        self.update_cursor_blit(new_frame)
        
        # Update FPS counter less frequently (every 10 frames) to reduce overhead
        if new_frame % 10 == 0 and elapsed_ms > 0:
            actual_fps = (new_frame * 1000.0) / elapsed_ms
            self.fps_label.setText(f"FPS: {actual_fps:.1f}")
    
    def reset_zoom(self):
        """Reset graph zoom to show full data range."""
        if not self.timestamps.size:
            return
        
        self.ax.set_xlim(0, self.timestamps[-1])
        
        # Redraw with new limits
        self.canvas.draw()
        self.graph_background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.update_cursor_blit(self.current_frame)
    
    def zoom_in(self):
        """Zoom in on the graph around current position."""
        if not self.timestamps.size:
            return
        
        current_time = self.timestamps[min(self.current_frame, len(self.timestamps) - 1)]
        xlim = self.ax.get_xlim()
        zoom_factor = 0.5
        
        # Calculate new limits
        span = (xlim[1] - xlim[0]) * zoom_factor
        new_left = max(0, current_time - span/2)
        new_right = min(self.timestamps[-1], current_time + span/2)
        
        # Ensure minimum span
        if new_right - new_left < 1.0:  # Minimum 1 second visible
            new_left = max(0, current_time - 0.5)
            new_right = min(self.timestamps[-1], current_time + 0.5)
        
        self.ax.set_xlim(new_left, new_right)
        
        # Redraw with new limits
        self.canvas.draw()
        self.graph_background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.update_cursor_blit(self.current_frame)
    
    def zoom_out(self):
        """Zoom out on the graph."""
        if not self.timestamps.size:
            return
        
        current_time = self.timestamps[min(self.current_frame, len(self.timestamps) - 1)]
        xlim = self.ax.get_xlim()
        zoom_factor = 2.0
        
        # Calculate new limits
        span = (xlim[1] - xlim[0]) * zoom_factor
        new_left = max(0, current_time - span/2)
        new_right = min(self.timestamps[-1], current_time + span/2)
        
        self.ax.set_xlim(new_left, new_right)
        
        # Redraw with new limits
        self.canvas.draw()
        self.graph_background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.update_cursor_blit(self.current_frame)
    
    def accept(self):
        """Save parameters and close dialog."""
        # Set parameters in the TST instance
        time_threshold = self.time_slider.value() / 10.0
        
        try:
            self.preview_motion_config.validate()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid classifier settings", str(exc))
            return
        self.tst.motion_config = copy.deepcopy(self.preview_motion_config)
        self.tst.set_parameters(time_threshold)
        
        # Clean up
        self.cleanup()
        super().accept()
    
    def reject(self):
        """Cancel and close dialog."""
        self.cleanup()
        super().reject()
    
    def cleanup(self):
        """Clean up resources before closing."""
        # Stop playback
        if self.is_playing:
            self.play_timer.stop()
            self.is_playing = False
        if hasattr(self, "scanner") and self.scanner.isRunning():
            self.scanner.cancel()
            if not self.scanner.wait(5000):
                logger.warning(
                    "TST review scanner did not stop within five seconds"
                )
        if self.preview_cap.isOpened():
            self.preview_cap.release()
        
        # Clear frame buffer and caches to free memory
        self.frame_buffer = []
        self.energy_data = {}
        self.decoded_frame_cache = {}
        self.processed_frame_cache = {}
    
    def closeEvent(self, event):
        """Handle window close event."""
        self.cleanup()
        super().closeEvent(event)
