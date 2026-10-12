"""
PyQt dialogs for interactively defining ROIs and a reference line for scaling.
Updated to use the MIDDLE frame of the video to ensure animals are visible.
"""
import cv2
import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QScrollArea, QVBoxLayout)

from roi_manager import ROIManager
from mazes.base_maze import Maze
from gui.dialog_layout import fit_dialog_to_available_screen


class ReferenceLineDialog(QDialog):
    """A dialog to draw a single line for distance calibration."""
    def __init__(self, pixmap, source_scale=(1.0, 1.0), parent=None):
        super().__init__(parent)
        self.setWindowTitle("Draw Reference Line")
        self.base_pixmap = pixmap
        self.source_scale_x = float(source_scale[0])
        self.source_scale_y = float(source_scale[1])
        self.points = []
        self.line_length = 0.0
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.instruction_label = QLabel("Click two points to define a line of known length (e.g., an arm width).")
        self.image_label = QLabel()
        self.image_label.setPixmap(self.base_pixmap)
        self.image_label.setFixedSize(self.base_pixmap.size())
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(False)
        self.image_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_scroll.setWidget(self.image_label)
        button_layout = QHBoxLayout()
        self.reset_button = QPushButton("Reset")
        self.accept_button = QPushButton("Accept")
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.accept_button)
        layout.addWidget(self.instruction_label)
        layout.addWidget(self.image_scroll, 1)
        layout.addLayout(button_layout)
        self.setLayout(layout)
        fit_dialog_to_available_screen(
            self,
            preferred_size=(
                self.base_pixmap.width() + 50,
                self.base_pixmap.height() + 130,
            ),
            minimum_size=(480, 360),
        )
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
            dx = (p2.x() - p1.x()) * self.source_scale_x
            dy = (p2.y() - p1.y()) * self.source_scale_y
            self.line_length = np.sqrt(dx**2 + dy**2)
            mid_point = QPointF((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2 - 10)
            painter.drawText(mid_point, f"{self.line_length:.2f} source pixels")
            
        painter.end()
        self.image_label.setPixmap(pixmap)
        
    def get_line_length(self): 
        return self.line_length


REFERENCE_LINE_NAME = "User-drawn reference line"
REFERENCE_LINE_PROMPT = (
    "Would you like to draw a reference line for distance calibration?\n\n"
    "Draw it over something of known length (e.g. an arm length). "
    "If you skip this, distances will be reported in pixels."
)


def offer_reference_line(parent, video_path, roi_manager: ROIManager) -> bool:
    """Ask for a reference line on a middle frame of the video.

    Sets the ROI manager's reference length (in source pixels) and returns
    True only if the user drew a valid line.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 10:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
    finally:
        cap.release()
    if not ret:
        return False

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    source_h, source_w = rgb.shape[:2]
    roi_manager.set_source_dimensions((source_w, source_h))
    if source_w > 1280:
        scale = 1280 / source_w
        rgb = cv2.resize(rgb, (int(source_w * scale), int(source_h * scale)))
    h, w, ch = rgb.shape
    pixmap = QPixmap.fromImage(
        QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    )
    dialog = ReferenceLineDialog(
        pixmap, source_scale=(source_w / w, source_h / h), parent=parent
    )
    if dialog.exec() and dialog.get_line_length() > 0:
        roi_manager.set_reference_length(
            dialog.get_line_length(), REFERENCE_LINE_NAME
        )
        return True
    return False


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
        self.source_width = w
        self.source_height = h
        self.display_width = w
        self.display_height = h
        
        # Bound very large source images once. The pixmap and click canvas
        # remain identical in size after this conversion.
        if w > 1280:
            scale = 1280 / w
            w, h = int(w * scale), int(h * scale)
            self.base_frame = cv2.resize(self.base_frame, (w, h))
            self.display_width = w
            self.display_height = h

        self.display_to_source_x = self.source_width / self.display_width
        self.display_to_source_y = self.source_height / self.display_height
        self.roi_manager.set_source_dimensions((self.source_width, self.source_height))
            
        self.base_pixmap = QPixmap.fromImage(QImage(self.base_frame.data, w, h, ch * w, QImage.Format.Format_RGB888))
        
        self.setWindowTitle("ROI Selection")
        
        layout = QVBoxLayout()
        self.instruction_label = QLabel()
        self.instruction_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.image_label = QLabel()
        self.image_label.setPixmap(self.base_pixmap)
        self.image_label.setFixedSize(self.base_pixmap.size())
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(False)
        self.image_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_scroll.setWidget(self.image_label)
        
        button_layout = QHBoxLayout()
        self.undo_button = QPushButton("Undo Last Point")
        self.clear_button = QPushButton("Clear Current Shape")
        self.next_button = QPushButton("Next Zone")
        
        button_layout.addWidget(self.undo_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.next_button)
        
        layout.addWidget(self.instruction_label)
        layout.addWidget(self.image_scroll, 1)
        layout.addLayout(button_layout)
        self.setLayout(layout)
        fit_dialog_to_available_screen(
            self,
            preferred_size=(
                self.base_pixmap.width() + 50,
                self.base_pixmap.height() + 130,
            ),
            minimum_size=(480, 360),
        )
        
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
        needs_line = getattr(self.maze, "needs_reference_line", None)
        if callable(needs_line) and not needs_line():
            # Paradigms without distance metrics (e.g. TST) skip calibration.
            self.accept()
            return
        reply = QMessageBox.question(self, 'Calibration', REFERENCE_LINE_PROMPT,
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                       QMessageBox.StandardButton.Yes)
        if reply == QMessageBox.StandardButton.Yes:
            ref_dialog = ReferenceLineDialog(
                self.image_label.pixmap(),
                source_scale=(self.display_to_source_x, self.display_to_source_y),
                parent=self
            )
            if ref_dialog.exec():
                length = ref_dialog.get_line_length()
                if length > 0:
                    self.roi_manager.set_reference_length(length, REFERENCE_LINE_NAME)
        self.accept()

    def next_roi(self):
        category, description = self.roi_definitions[self.current_roi_index]
        if 0 < len(self.current_polygon_points) < 3:
            QMessageBox.warning(
                self,
                "Incomplete ROI",
                f"{description} needs at least three points."
            )
            return
        if len(self.current_polygon_points) >= 3:
            self.finalize_polygon()
        if not self.roi_manager.rois.get(category):
            QMessageBox.warning(
                self,
                "Missing ROI",
                f"Draw at least one polygon for {description} before continuing."
            )
            return
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
        points = np.array([
            (p.x() * self.display_to_source_x, p.y() * self.display_to_source_y)
            for p in self.current_polygon_points
        ], dtype=np.int32)
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
                poly = QPolygonF([
                    QPointF(
                        p[0] / self.display_to_source_x,
                        p[1] / self.display_to_source_y
                    )
                    for p in roi.points
                ])
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
