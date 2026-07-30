"""
ROI Preview Dialog - Phase 2 Enhancement
Allows users to preview configured ROIs before starting analysis.
"""
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QDialog, QLabel, QPushButton, QSlider,
                             QVBoxLayout, QHBoxLayout, QGroupBox)

from roi_manager import ROIManager
from mazes.base_maze import Maze


class ROIPreviewDialog(QDialog):
    """
    Dialog to preview configured ROIs on video frames.
    Users can scrub through the video to verify ROI placement.
    """
    
    def __init__(self, video_path: Path, roi_manager: ROIManager,
                 maze: Maze, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.roi_manager = roi_manager
        self.maze = maze
        
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame_idx = 0
        
        self.setWindowTitle(f"ROI Preview - {video_path.name}")
        self.setMinimumSize(800, 700)
        self.init_ui()
        self.update_frame(0)

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        
        # Info label
        info_text = (
            f"<b>Video:</b> {self.video_path.name}<br>"
            f"<b>Maze:</b> {self.maze.name}<br>"
            f"<b>Duration:</b> {self.total_frames / self.fps:.1f}s "
            f"({self.total_frames} frames @ {self.fps:.1f} fps)"
        )
        info_label = QLabel(info_text)
        layout.addWidget(info_label)
        
        # Video preview
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("border: 2px solid #333; background: black;")
        layout.addWidget(self.preview_label, 1)
        
        # Frame info
        self.frame_info_label = QLabel()
        self.frame_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.frame_info_label)
        
        # Frame slider
        slider_group = QGroupBox("Frame Navigation")
        slider_layout = QVBoxLayout()
        
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.total_frames - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self.on_slider_changed)
        slider_layout.addWidget(self.frame_slider)
        
        # Jump buttons
        jump_layout = QHBoxLayout()
        
        btn_start = QPushButton("⏮ Start")
        btn_back_1s = QPushButton("⏪ -1s")
        btn_back_frame = QPushButton("◀ -1f")
        btn_forward_frame = QPushButton("▶ +1f")
        btn_forward_1s = QPushButton("⏩ +1s")
        btn_end = QPushButton("⏭ End")
        
        btn_start.clicked.connect(lambda: self.jump_to_frame(0))
        btn_back_1s.clicked.connect(lambda: self.jump_relative(-int(self.fps)))
        btn_back_frame.clicked.connect(lambda: self.jump_relative(-1))
        btn_forward_frame.clicked.connect(lambda: self.jump_relative(1))
        btn_forward_1s.clicked.connect(lambda: self.jump_relative(int(self.fps)))
        btn_end.clicked.connect(lambda: self.jump_to_frame(self.total_frames - 1))
        
        jump_layout.addWidget(btn_start)
        jump_layout.addWidget(btn_back_1s)
        jump_layout.addWidget(btn_back_frame)
        jump_layout.addWidget(btn_forward_frame)
        jump_layout.addWidget(btn_forward_1s)
        jump_layout.addWidget(btn_end)
        
        slider_layout.addLayout(jump_layout)
        slider_group.setLayout(slider_layout)
        layout.addWidget(slider_group)
        
        # ROI Statistics
        stats_group = QGroupBox("ROI Statistics")
        stats_layout = QVBoxLayout()
        
        stats = self.roi_manager.get_stats()
        stats_text = (
            f"<b>Total ROIs:</b> {stats['total_rois']}<br>"
            f"<b>Categories:</b> {stats['num_categories']}<br>"
        )
        for category, info in stats['categories'].items():
            stats_text += (
                f"<b>{category.replace('_', ' ').title()}:</b> "
                f"{info['count']} region(s), area={info['total_area']:.0f} px²<br>"
            )
        
        stats_label = QLabel(stats_text)
        stats_layout.addWidget(stats_label)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)
        
        self.setLayout(layout)

    def on_slider_changed(self, value):
        """Handle slider value change."""
        self.update_frame(value)

    def jump_to_frame(self, frame_idx):
        """Jump to a specific frame."""
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        self.frame_slider.setValue(frame_idx)

    def jump_relative(self, delta):
        """Jump relative to current frame."""
        new_idx = self.current_frame_idx + delta
        self.jump_to_frame(new_idx)

    def update_frame(self, frame_idx):
        """Update the displayed frame."""
        self.current_frame_idx = frame_idx
        
        # Read frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if not ret:
            return
        
        # Draw ROIs
        vis_frame = self.roi_manager.draw_on_frame(frame, thickness=2)
        
        # Convert to QPixmap
        rgb_frame = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale to fit
        scaled_pixmap = pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled_pixmap)
        
        # Update info
        timestamp = frame_idx / self.fps
        self.frame_info_label.setText(
            f"Frame {frame_idx}/{self.total_frames - 1} | "
            f"Time: {timestamp:.2f}s / {self.total_frames / self.fps:.2f}s"
        )

    def closeEvent(self, event):
        """Clean up resources on close."""
        if self.cap:
            self.cap.release()
        super().closeEvent(event)

    def __del__(self):
        """Ensure video capture is released."""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()