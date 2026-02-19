"""
PyQt dialogs for interactively defining ROIs and a reference line for scaling.
Updated to use the MIDDLE frame of the video to ensure animals are visible.
"""
import cv2
import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QVBoxLayout)

from roi_manager import ROIManager
from mazes.base_maze import Maze


class ReferenceLineDialog(QDialog):
    """A dialog to draw a single line for distance calibration."""
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Draw Reference Line")
        self.base_pixmap = pixmap
        self.points = []
        self.line_length = 0.0
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.instruction_label = QLabel("Click two points to define a line of known length (e.g., an arm width).")
        self.image_label = QLabel()
        self.image_label.setPixmap(self.base_pixmap)
        button_layout = QHBoxLayout()
        self.reset_button = QPushButton("Reset")
        self.accept_button = QPushButton("Accept")
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.accept_button)
        layout.addWidget(self.instruction_label)
        layout.addWidget(self.image_label)
        layout.addLayout(button_layout)
        self.setLayout(layout)
        self.image_label.mousePressEvent = self.add_point
        self.reset_button.clicked.connect(self.reset)
        self.accept_button.clicked.connect(self.accept_line)

    def add_point(self, event):
        if len(self.points) < 2: 
            self.points.append(event.pos())
            self.update()
            
    def reset(self): 
        self.points = []
        self.line_length = 0.0
        self.update()
        
    def accept_line(self):
        if len(self.points) == 2: 
            self.accept()
        else: 
            QMessageBox.warning(self, "Incomplete", "Please click two points to define a line.")

    def paintEvent(self, event):
        super().paintEvent(event)
        pixmap = self.base_pixmap.copy()
        painter = QPainter(pixmap)
        pen = QPen(Qt.GlobalColor.green, 2)
        painter.setPen(pen)
        
        if len(self.points) > 0: 
            painter.drawEllipse(self.points[0], 3, 3)
            
        if len(self.points) == 2:
            p1, p2 = self.points[0], self.points[1]
            painter.drawEllipse(p2, 3, 3)
            painter.drawLine(p1, p2)
            self.line_length = np.sqrt((p2.x() - p1.x())**2 + (p2.y() - p1.y())**2)
            mid_point = QPointF((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2 - 10)
            painter.drawText(mid_point, f"{self.line_length:.2f} pixels")
            
        painter.end()
        self.image_label.setPixmap(pixmap)
        
    def get_line_length(self): 
        return self.line_length


class RoiSelectorDialog(QDialog):
    """A dialog for interactively drawing polygonal ROIs on a video frame."""
    def __init__(self, video_path: str, maze: Maze, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.roi_manager = ROIManager()
        self.maze = maze
        self.roi_definitions = self.maze.get_roi_definitions()
        self.current_roi_index = 0
        self.current_polygon_points = []
        self.init_ui()
        self.start_next_roi_selection()

    def init_ui(self):
        cap = cv2.VideoCapture(str(self.video_path))
        
        # --- IMPROVEMENT: Seek to middle frame ---
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 10:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            
        ret, frame = cap.read()
        cap.release()
        
        if not ret: 
            raise IOError("Could not read frame from video.")
            
        self.base_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = self.base_frame.shape
        
        # Limit size if video is 4K to fit on screen
        if w > 1280:
            scale = 1280 / w
            w, h = int(w * scale), int(h * scale)
            self.base_frame = cv2.resize(self.base_frame, (w, h))
            
        self.base_pixmap = QPixmap.fromImage(QImage(self.base_frame.data, w, h, ch * w, QImage.Format.Format_RGB888))
        
        self.setWindowTitle("ROI Selection")
        self.setMinimumSize(w + 50, h + 150)
        
        layout = QVBoxLayout()
        self.instruction_label = QLabel()
        self.instruction_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.image_label = QLabel()
        self.image_label.setPixmap(self.base_pixmap)
        
        button_layout = QHBoxLayout()
        self.undo_button = QPushButton("Undo Last Point")
        self.clear_button = QPushButton("Clear Current Shape")
        self.next_button = QPushButton("Next Zone")
        
        button_layout.addWidget(self.undo_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.next_button)
        
        layout.addWidget(self.instruction_label)
        layout.addWidget(self.image_label)
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        self.next_button.clicked.connect(self.next_roi)
        self.undo_button.clicked.connect(self.undo_point)
        self.clear_button.clicked.connect(self.clear_current_polygon)
        self.image_label.mousePressEvent = self.add_point

    def start_next_roi_selection(self):
        if self.current_roi_index >= len(self.roi_definitions):
            self.prompt_for_reference_line()
            return
        _, desc = self.roi_definitions[self.current_roi_index]
        self.instruction_label.setText(
            f"Step {self.current_roi_index + 1}/{len(self.roi_definitions)}: Draw polygon(s) for {desc}.\n"
            "Left-click to add points. Right-click to finish a shape."
        )
        self.current_polygon_points = []
        if self.current_roi_index == len(self.roi_definitions) - 1:
            self.next_button.setText("Finish ROI Selection")
        self.update()

    def prompt_for_reference_line(self):
        reply = QMessageBox.question(self, 'Calibration', 'Would you like to draw a reference line for distance calibration?',
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                       QMessageBox.StandardButton.Yes)
        if reply == QMessageBox.StandardButton.Yes:
            ref_dialog = ReferenceLineDialog(self.image_label.pixmap(), self)
            if ref_dialog.exec():
                length = ref_dialog.get_line_length()
                if length > 0:
                    self.roi_manager.set_reference_length(length, "User-drawn reference line")
        self.accept()

    def next_roi(self):
        if len(self.current_polygon_points) >= 3: self.finalize_polygon()
        self.current_roi_index += 1
        self.start_next_roi_selection()

    def add_point(self, event):
        if event.button() == Qt.MouseButton.LeftButton: 
            self.current_polygon_points.append(QPointF(event.pos()))
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            if len(self.current_polygon_points) >= 3: 
                self.finalize_polygon()
                self.update()
    
    def undo_point(self):
        if self.current_polygon_points: 
            self.current_polygon_points.pop()
            self.update()
            
    def clear_current_polygon(self): 
        self.current_polygon_points = []
        self.update()

    def finalize_polygon(self):
        points = np.array([(p.x(), p.y()) for p in self.current_polygon_points], dtype=np.int32)
        category, _ = self.roi_definitions[self.current_roi_index]
        self.roi_manager.add_roi(category, points)
        self.current_polygon_points = []

    def paintEvent(self, event):
        super().paintEvent(event)
        pixmap = self.base_pixmap.copy()
        painter = QPainter(pixmap)
        
        base_colors = [QColor(255, 0, 0, 100), QColor(0, 255, 0, 100), QColor(0, 0, 255, 100), QColor(255, 0, 255, 100)]
        color_map = {cat: base_colors[i % len(base_colors)] for i, (cat, _) in enumerate(self.roi_definitions) if 'center' not in cat}
        color_map['center'] = QColor(255, 255, 0, 100)
        
        # Draw existing ROIs
        for category, roi_list in self.roi_manager.rois.items():
            color = color_map.get(category, QColor(200, 200, 200, 100))
            painter.setPen(QPen(color.darker(150), 2))
            painter.setBrush(color)
            for roi in roi_list:
                poly = QPolygonF([QPointF(p[0], p[1]) for p in roi.points])
                painter.drawPolygon(poly)
        
        # Draw current polygon being drawn
        if self.current_polygon_points:
            pen = QPen(Qt.GlobalColor.magenta, 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            
            poly_points = QPolygonF(self.current_polygon_points)
            painter.drawPolyline(poly_points)
            
            # Close the loop visually if > 2 points
            if len(self.current_polygon_points) > 2:
                painter.setPen(QPen(Qt.GlobalColor.magenta, 1, Qt.PenStyle.DashLine))
                painter.drawLine(self.current_polygon_points[-1], self.current_polygon_points[0])
                
            # Draw vertices
            painter.setBrush(Qt.GlobalColor.white)
            for point in self.current_polygon_points: 
                painter.drawEllipse(point, 3, 3)
                
        painter.end()
        self.image_label.setPixmap(pixmap)

    def get_roi_manager(self) -> ROIManager: 
        return self.roi_manager