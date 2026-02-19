"""
gui/tuner_dialog.py
Enhanced dialog for tuning detection with live quality feedback.
Updated to fix parameter mismatch with Core modules.
"""
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QDialog, QGridLayout, QHBoxLayout, QLabel,
                             QPushButton, QSizePolicy, QSlider, QVBoxLayout,
                             QGroupBox)

from core.models import DetectionConfig
from core.detection import DetectionEngine, DetectionQualityMonitor
from core.video import VideoHandler
from roi_manager import ROIManager


class TunerDialog(QDialog):
    """Enhanced dialog for tuning detection with live quality feedback."""
    
    def __init__(self, video_path: str, ref_frame: np.ndarray, 
                 roi_manager: ROIManager, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.ref_frame = ref_frame
        self.roi_manager = roi_manager
        self.config = DetectionConfig()
        self.video_info = VideoHandler(video_path)
        self.cap = cv2.VideoCapture(video_path)
        self.current_frame_idx = self.video_info.total_frames // 2
        
        # Quality monitoring
        self.quality_monitor = DetectionQualityMonitor(window_size=30)
        
        self.init_ui()
        self.frame_slider.setValue(self.current_frame_idx)
        
        # Auto-update stats
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_quality_stats)
        self.stats_timer.start(500)

    def init_ui(self):
        self.setWindowTitle("Interactive Detection Tuner - Enhanced")
        self.setMinimumSize(1400, 900)

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)

        # Image Previews
        self.main_view = self._create_image_label("Main View")
        self.diff_view = self._create_image_label("Difference View")
        self.thresh_view = self._create_image_label("Threshold View")
        left_layout.addWidget(self.main_view, 2)
        left_layout.addWidget(self.diff_view, 1)
        left_layout.addWidget(self.thresh_view, 1)

        # Quality Stats Panel
        stats_group = QGroupBox("Detection Quality (Live)")
        stats_layout = QVBoxLayout()
        
        self.stats_label = QLabel("Analyzing...")
        self.stats_label.setStyleSheet("background: #f0f0f0; padding: 15px; font-size: 11pt; border-radius: 5px;")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        
        self.recommendation_label = QLabel("")
        self.recommendation_label.setStyleSheet("background: #fff3cd; padding: 10px; font-size: 10pt; border-radius: 5px; color: #856404;")
        self.recommendation_label.setWordWrap(True)
        self.recommendation_label.setVisible(False)
        stats_layout.addWidget(self.recommendation_label)
        
        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)

        # Parameter Controls
        param_grid = QGridLayout()
        self.sliders = {}
        self._add_slider(param_grid, "Threshold (x10)", 950, 999, int(self.config.threshold_percentile * 10))
        self._add_slider(param_grid, "Weight Omega (x100)", 0, 100, int(self.config.weight_omega * 100))
        self._add_slider(param_grid, "Window Size", 20, 300, self.config.window_size)
        right_layout.addLayout(param_grid)

        # Frame Navigation
        self.frame_label = QLabel()
        self.frame_slider = QSlider(Qt.Orientation.Horizontal, minimum=0, maximum=self.video_info.total_frames - 1)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_change)
        right_layout.addWidget(self.frame_label)
        right_layout.addWidget(self.frame_slider)

        help_label = QLabel("Keyboard: N/P (frame), +/- (jump 100 frames)\nTip: Scan through video to test different frames")
        help_label.setStyleSheet("color: #666; font-size: 9pt;")
        right_layout.addWidget(help_label)
        
        right_layout.addStretch()

        # Action buttons
        button_layout = QHBoxLayout()
        self.test_scan_button = QPushButton("Quick Scan Test")
        self.test_scan_button.clicked.connect(self.run_quick_scan)
        button_layout.addWidget(self.test_scan_button)
        
        self.accept_button = QPushButton("Accept Parameters")
        self.accept_button.clicked.connect(self.accept)
        button_layout.addWidget(self.accept_button)
        
        right_layout.addLayout(button_layout)
        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 1)
        self.setLayout(main_layout)

    def _create_image_label(self, title: str) -> QLabel:
        label = QLabel(title)
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("background-color: black; color: white; border: 1px solid grey;")
        return label

    def _add_slider(self, layout: QGridLayout, name: str, min_val: int, max_val: int, initial_val: int):
        row = layout.rowCount()
        label = QLabel(f"{name}: {initial_val}")
        slider = QSlider(Qt.Orientation.Horizontal, minimum=min_val, maximum=max_val, value=initial_val)
        slider.valueChanged.connect(lambda val, l=label, n=name: l.setText(f"{n}: {val}"))
        slider.valueChanged.connect(self.update_detection)
        layout.addWidget(label, row, 0)
        layout.addWidget(slider, row, 1)
        self.sliders[name] = slider

    def on_frame_slider_change(self, value: int):
        if self.current_frame_idx != value:
            self.current_frame_idx = value
            self.update_detection()

    def update_detection(self):
        # Update config
        self.config.threshold_percentile = self.sliders["Threshold (x10)"].value() / 10.0
        self.config.weight_omega = self.sliders["Weight Omega (x100)"].value() / 100.0
        self.config.window_size = self.sliders["Window Size"].value()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if not ret: return

        # Engine & Detection
        engine = DetectionEngine(self.ref_frame, self.config)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- FIX: Updated parameter name 'use_last_pos' ---
        position = engine.detect_position(gray, use_last_pos=False)
        
        # Monitor Update
        diff = cv2.absdiff(gray, self.ref_frame)
        confidence_map = self._calculate_confidence_map(diff)
        detected = position is not None
        confidence = 0.8 if detected else 0.0
        self.quality_monitor.update(detected, confidence, position)

        # Drawing
        display_frame = frame.copy()
        if position:
            cv2.circle(display_frame, position, 15, (0, 255, 0), 3)
            cv2.circle(display_frame, position, 3, (0, 0, 255), -1)
        
        for category, rois in self.roi_manager.rois.items():
            for roi in rois:
                pts = np.array(roi.points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], True, (255, 255, 0), 2)

        self._update_image(self.main_view, display_frame)
        
        diff_colored = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        self._update_image(self.diff_view, diff_colored)
        
        if confidence_map is not None:
            thresh_vis = (confidence_map * 255).astype(np.uint8)
            thresh_colored = cv2.applyColorMap(thresh_vis, cv2.COLORMAP_JET)
            if position: cv2.circle(thresh_colored, position, 15, (255, 255, 255), 2)
            self._update_image(self.thresh_view, thresh_colored)

        self.frame_label.setText(f"Frame: {self.current_frame_idx} / {self.video_info.total_frames - 1}")

    def update_quality_stats(self):
        if self.quality_monitor.frames_processed < 5: return
        stats = self.quality_monitor.get_statistics()
        status, color, recommendation = self.quality_monitor.get_quality_status()
        
        stats_text = (f"<b>Current Detection Quality:</b><br><br>"
                      f"Detection Rate: <b>{stats['detection_rate']:.1f}%</b><br>"
                      f"Confidence: <b>{stats['confidence']:.1f}%</b><br>"
                      f"Stability: <b>{stats['tracking_stability']:.1f}%</b><br><br>"
                      f"Status: <b>{status}</b>")
        
        self.stats_label.setText(stats_text)
        
        colors = {'green': '#d4edda', 'orange': '#fff3cd', 'red': '#f8d7da'}
        borders = {'green': '#28a745', 'orange': '#ffc107', 'red': '#dc3545'}
        self.stats_label.setStyleSheet(f"background: {colors[color]}; padding: 15px; font-size: 11pt; border: 2px solid {borders[color]}; border-radius: 5px;")
        
        if color != 'green':
            self.recommendation_label.setText(f"💡 {recommendation}")
            self.recommendation_label.setVisible(True)
        else:
            self.recommendation_label.setVisible(False)

    def run_quick_scan(self):
        self.test_scan_button.setEnabled(False)
        self.test_scan_button.setText("Scanning...")
        self.quality_monitor.reset()
        
        total_frames = self.video_info.total_frames
        test_indices = np.random.choice(total_frames, min(50, total_frames), replace=False)
        engine = DetectionEngine(self.ref_frame, self.config)
        
        for idx in test_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if not ret: continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # --- FIX: Updated parameter name 'use_last_pos' ---
            position = engine.detect_position(gray, use_last_pos=False)
            
            detected = position is not None
            self.quality_monitor.update(detected, 0.8 if detected else 0.0, position)
        
        self.update_quality_stats()
        self.test_scan_button.setEnabled(True)
        self.test_scan_button.setText("Quick Scan Test")

    def _calculate_confidence_map(self, diff_image: np.ndarray) -> np.ndarray:
        window_size = self.config.window_size
        if window_size % 2 == 0: window_size += 1
        window_size = max(3, window_size)
        blurred = cv2.GaussianBlur(diff_image, (window_size, window_size), 0)
        return blurred.astype(np.float32) / 255.0

    def _update_image(self, label: QLabel, image: np.ndarray):
        if image.ndim == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        q_image = QImage(image.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_N:
            self.frame_slider.setValue(min(self.current_frame_idx + 1, self.video_info.total_frames - 1))
        elif key == Qt.Key.Key_P:
            self.frame_slider.setValue(max(self.current_frame_idx - 1, 0))
        elif key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
            self.frame_slider.setValue(min(self.current_frame_idx + 100, self.video_info.total_frames - 1))
        elif key == Qt.Key.Key_Minus:
            self.frame_slider.setValue(max(self.current_frame_idx - 100, 0))
        else:
            super().keyPressEvent(event)

    def get_detection_config(self) -> DetectionConfig:
        return self.config

    def closeEvent(self, event):
        if self.cap is not None: self.cap.release()
        self.stats_timer.stop()
        event.accept()