"""
High-Performance TST Tuner Dialog with Matplotlib Blitting.
Optimized for real-time video playback and graph synchronization.
"""
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QElapsedTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QSlider, QPushButton, QGroupBox, QProgressBar,
                             QSizePolicy)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from core.analysis_engine import MotionEngine


class EnergyScanner(QThread):
    """
    Background thread for video scanning and frame buffering.
    Optimized for full video scanning with complete frame buffering.
    Uses compression to handle longer videos efficiently.
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict, list, tuple, float)
    
    def __init__(self, video_path, rois, target_mouse, max_frames=None):
        super().__init__()
        self.video_path = video_path
        self.rois = {target_mouse: rois[target_mouse]}
        self.max_frames = max_frames  # None means scan entire video
        
    def run(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            self.finished.emit({}, [], (1, 1), 30.0)
            return
        
        # Get video properties
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine how many frames to scan
        frames_to_scan = self.max_frames if self.max_frames else total_frames
        
        ret, prev = cap.read()
        if not ret:
            cap.release()
            self.finished.emit({}, [], (orig_w, orig_h), fps)
            return
        
        engine = MotionEngine(self.rois)
        energy_data = {name: [] for name in self.rois}
        frame_buffer = []
        
        # Buffer first frame with JPEG compression for memory efficiency
        self._buffer_frame_compressed(prev, frame_buffer)
        
        count = 0
        
        while count < frames_to_scan:
            ret, curr = cap.read()
            if not ret:
                break
            
            # Calculate energy for EVERY frame
            energies = engine.calculate_motion(curr, prev)
            for name, val in energies.items():
                energy_data[name].append(val)
            
            # Buffer ALL frames but with compression
            self._buffer_frame_compressed(curr, frame_buffer)
            
            prev = curr.copy()
            count += 1
            
            # Progress updates
            if count % 100 == 0:
                progress = int(count / frames_to_scan * 100)
                self.progress.emit(progress)
        
        cap.release()
        self.finished.emit(energy_data, frame_buffer, (orig_w, orig_h), fps)
    
    def _buffer_frame_compressed(self, frame, buffer):
        """
        Buffer frames with JPEG compression to reduce memory usage.
        This allows us to store more frames in memory for smooth playback.
        """
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return
        
        # Resize to reasonable preview size
        target_height = 480  # Higher res than before for better quality
        scale = target_height / h
        new_w = int(w * scale)
        
        # Use INTER_LINEAR for good quality
        resized = cv2.resize(frame, (new_w, target_height), 
                            interpolation=cv2.INTER_LINEAR)
        
        # Compress frame using JPEG encoding (reduces memory by ~10x)
        # Quality 85 provides good balance between size and quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, encoded = cv2.imencode('.jpg', resized, encode_param)
        
        # Store compressed data
        buffer.append(encoded)




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
        self.resize(1400, 900)
        
        self.video_path = video_path
        self.roi_manager = roi_manager
        self.tst = tst_maze_instance
        
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
        self.frame_buffer = []  # Stores compressed frames
        self.decoded_frame_cache = {}  # Cache for decoded frames
        self.max_decoded_cache = 200  # Keep more decoded frames
        self.orig_dims = (1, 1)
        self.fps = 30.0
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
        
        # Frame cache for processed video frames
        self.processed_frame_cache = {}
        self.cache_size = 100  # Keep last N processed frames in cache
        
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
        self.image_label.setFixedSize(640, 480)  # Increased size to match new resolution
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
        
        # FPS counter
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fps_label.setStyleSheet("font-size: 12px; color: #888;")
        video_layout.addWidget(self.fps_label)
        
        video_group.setLayout(video_layout)
        
        # Graph Section
        display_name = self.selected_mouse.replace('_', ' ').title() if self.selected_mouse else "None"
        graph_group = QGroupBox(f"Motion Energy Profile: {display_name}")
        graph_layout = QVBoxLayout()
        
        # Initialize Matplotlib with optimization settings
        self.figure = Figure(figsize=(10, 5), dpi=80)
        self.canvas = OptimizedCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Configure axes
        self.ax.set_xlabel("Time (s)", fontsize=10)
        self.ax.set_ylabel("Motion Energy", fontsize=10)
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
        
        graph_group.setLayout(graph_layout)
        
        top_layout.addWidget(video_group)
        top_layout.addWidget(graph_group, 1)  # Graph takes more space
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
        
        # Energy Threshold
        energy_layout = QVBoxLayout()
        self.energy_label = QLabel("Energy Threshold: 10.0")
        self.energy_label.setStyleSheet("font-weight: bold;")
        self.energy_slider = QSlider(Qt.Orientation.Horizontal)
        self.energy_slider.setRange(0, 100)
        self.energy_slider.setValue(10)
        self.energy_slider.valueChanged.connect(self.on_param_change)
        energy_layout.addWidget(self.energy_label)
        energy_layout.addWidget(self.energy_slider)
        
        # Time Threshold
        time_layout = QVBoxLayout()
        self.time_label = QLabel("Min Immobility: 0.5s")
        self.time_label.setStyleSheet("font-weight: bold;")
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(1, 50)  # 0.1s to 5.0s
        self.time_slider.setValue(5)
        self.time_slider.valueChanged.connect(self.on_param_change)
        time_layout.addWidget(self.time_label)
        time_layout.addWidget(self.time_slider)
        
        param_layout.addLayout(energy_layout)
        param_layout.addLayout(time_layout)
        param_group.setLayout(param_layout)
        
        controls_layout.addWidget(param_group, 1)
        main_layout.addLayout(controls_layout)
        
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
    
    def start_scan(self):
        """Start background video scanning."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Get video info to determine scanning strategy
        cap = cv2.VideoCapture(str(self.video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        
        # For TST, scan the entire video up to reasonable limits
        # With compression, we can handle longer videos
        max_frames_to_scan = min(total_frames, 54000)  # Cap at 30 minutes @ 30fps
        
        # Show info to user
        duration_sec = max_frames_to_scan / fps
        self.image_label.setText(f"Scanning {duration_sec:.1f} seconds of video...")
        
        self.scanner = EnergyScanner(
            self.video_path,
            self.roi_dict,
            self.selected_mouse,
            max_frames=max_frames_to_scan
        )
        self.scanner.progress.connect(self.on_scan_progress)
        self.scanner.finished.connect(self.on_scan_complete)
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
        self.frame_buffer = frame_buffer  # Compressed frames
        self.orig_dims = orig_dims
        self.fps = fps
        self.scan_len = len(frame_buffer)
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Enable controls
        self.accept_button.setEnabled(True)
        self.play_button.setEnabled(True)
        self.scrub_slider.setEnabled(True)
        
        # Set slider range based on actual frames
        self.scrub_slider.setRange(0, self.scan_len - 1)
        
        # Auto-scale energy threshold
        trace = energy_data[self.selected_mouse]
        if trace:
            max_energy = max(trace)
            self.energy_slider.setRange(0, int(max_energy * 1.2))
            self.energy_slider.setValue(int(max_energy * 0.3))
        
        # Show scan info
        duration = self.scan_len / self.fps
        memory_mb = sum(len(f) for f in frame_buffer) / (1024 * 1024)
        print(f"Scanned {duration:.1f}s, {self.scan_len} frames, ~{memory_mb:.1f}MB in memory")
        
        # Initialize plot with blitting support
        self.init_plot_optimized(trace)
        
        # Compute initial parameters
        self.on_param_change()
        
        # Show first frame
        self.update_frame(0)
    
    def init_plot_optimized(self, trace):
        """
        Initialize the plot with blitting optimization in mind.
        Sets up static elements and prepares for dynamic updates.
        """
        self.ax.clear()
        
        # Pre-compute timestamps
        self.timestamps = np.array([i / self.fps for i in range(len(trace))])
        
        # Plot motion energy trace (static)
        self.line_trace, = self.ax.plot(
            self.timestamps, trace,
            'k-', linewidth=1, alpha=0.7,
            label="Motion Energy"
        )
        
        # Initialize threshold line (dynamic but less frequent)
        self.line_thresh = self.ax.axhline(
            y=0, color='blue', linestyle='--',
            linewidth=1.5, label="Threshold", animated=True  # Mark as animated
        )
        
        # Initialize cursor line (dynamic and frequent)
        self.cursor_line = self.ax.axvline(
            x=0, color='magenta', linewidth=2,
            alpha=0.8, label="Current Frame", animated=True  # Mark as animated
        )
        
        # Setup axes
        self.ax.set_xlim(0, self.timestamps[-1] if len(self.timestamps) > 0 else 1)
        y_max = max(trace) * 1.1 if trace else 10
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
        
        energy_thresh = self.energy_slider.value()
        time_thresh = self.time_slider.value() / 10.0
        
        # Update labels
        self.energy_label.setText(f"Energy Threshold: {energy_thresh:.1f}")
        self.time_label.setText(f"Min Immobility: {time_thresh:.1f}s")
        
        # Calculate immobility zones
        trace = self.energy_data[self.selected_mouse]
        min_frames = int(time_thresh * self.fps)
        
        # Compute raw and filtered immobility
        self.raw_immobile = np.array(trace) < energy_thresh
        self.filtered_immobile = self.tst._apply_temporal_filter(
            self.raw_immobile, min_frames
        )
        
        # Remove old fill regions if they exist
        if self.fill_immobile:
            self.fill_immobile.remove()
            self.fill_immobile = None
        if self.fill_pause:
            self.fill_pause.remove()
            self.fill_pause = None
        
        # Create new fill regions
        y_max = max(trace) * 1.1 if trace else 10
        
        # Immobile zones (red)
        self.fill_immobile = self.ax.fill_between(
            self.timestamps, 0, y_max,
            where=self.filtered_immobile,
            color='red', alpha=0.25,
            interpolate=True, step='mid'
        )
        
        # Pause zones (yellow) - below threshold but not long enough
        potential_pause = self.raw_immobile & (~self.filtered_immobile)
        self.fill_pause = self.ax.fill_between(
            self.timestamps, 0, y_max,
            where=potential_pause,
            color='yellow', alpha=0.2,
            interpolate=True, step='mid'
        )
        
        # Update threshold line position (it's animated so won't be in background)
        self.line_thresh.set_ydata([energy_thresh, energy_thresh])
        
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
    
    def update_cursor_blit(self, frame_idx):
        """
        Ultra-fast cursor update using blitting.
        Only redraws the cursor line without touching the rest of the graph.
        """
        if not self.graph_background or not self.cursor_line:
            return
        
        # Calculate time position
        time_pos = frame_idx / self.fps
        
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
        Get a decoded frame from the buffer, using cache when possible.
        """
        if frame_idx in self.decoded_frame_cache:
            return self.decoded_frame_cache[frame_idx]
        
        if frame_idx >= len(self.frame_buffer):
            return None
        
        # Decode compressed frame
        compressed = self.frame_buffer[frame_idx]
        frame = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
        
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
        Handles compressed frames with caching for smooth playback.
        """
        if not self.frame_buffer or frame_idx >= len(self.frame_buffer):
            return
        
        # Check processed frame cache first
        cache_key = (frame_idx, 
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
                    
                    # Determine mobility state
                    is_below_thresh = self.raw_immobile[frame_idx] if (self.raw_immobile is not None and frame_idx < len(self.raw_immobile)) else False
                    is_immobile = self.filtered_immobile[frame_idx] if (self.filtered_immobile is not None and frame_idx < len(self.filtered_immobile)) else False
                    
                    if not is_below_thresh:
                        color = (0, 255, 0)  # Green - Mobile
                        status = "MOBILE"
                    elif is_below_thresh and not is_immobile:
                        color = (255, 200, 0)  # Yellow - Pause
                        status = "PAUSE"
                    else:
                        color = (255, 50, 50)  # Red - Immobile
                        status = "IMMOBILE"
                    
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
                        
                        # Add status text with background
                        text = f"{status} | Energy: {energy_val:.1f}"
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
        time_sec = frame_idx / self.fps
        self.frame_label.setText(f"Frame: {frame_idx + 1} / {self.scan_len} | Time: {time_sec:.1f}s")
        
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
            interval = int(1000 / (self.fps * speed))
            self.play_timer.start(max(1, interval))
    
    def toggle_playback(self):
        """Toggle video playback with accurate timing."""
        if self.play_button.isChecked():
            # Start playback
            self.is_playing = True
            self.play_button.setText("⏸ Pause")
            
            # Calculate timer interval based on speed
            speed = self.speed_slider.value() / 100.0
            interval = int(1000 / (self.fps * speed))
            
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
        target_frame = int((elapsed_ms / 1000.0) * self.fps * speed)
        
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
        
        current_time = self.current_frame / self.fps
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
        
        current_time = self.current_frame / self.fps
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
        energy_threshold = float(self.energy_slider.value())
        time_threshold = self.time_slider.value() / 10.0
        
        self.tst.set_parameters(energy_threshold, time_threshold)
        
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
        
        # Clear frame buffer and caches to free memory
        self.frame_buffer = []
        self.energy_data = {}
        self.decoded_frame_cache = {}
        self.processed_frame_cache = {}
    
    def closeEvent(self, event):
        """Handle window close event."""
        self.cleanup()
        super().closeEvent(event)