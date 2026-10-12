"""
PyQt dialogs for interactively defining ROIs and a reference line for scaling.
Updated to use the MIDDLE frame of the video to ensure animals are visible.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QKeySequence, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QScrollArea, QVBoxLayout)

from roi_manager import ROIManager
from mazes.base_maze import Maze
from gui import ux_helpers
from gui.dialog_layout import fit_dialog_to_available_screen
from gui.qt_image import bgr_to_qpixmap


# Smallest polygon the pipeline accepts; ROIManager.validate_rois reports
# anything below this as an issue that blocks the run, so refuse it here.
MIN_ROI_AREA_PX = 100.0
MIN_POLYGON_POINTS = 3


@dataclass(frozen=True)
class RoiRule:
    """How many polygons one ROI category may hold."""
    minimum: int = 1
    maximum: Optional[int] = None


@dataclass(frozen=True)
class RoiStepState:
    """What the ROI selector allows right now, and why."""
    can_advance: bool
    can_add_point: bool
    hint: str


def roi_rule_for_maze(maze) -> RoiRule:
    """TST mice hold exactly one polygon each; every other category 1 or more."""
    from mazes.tst import TST
    if isinstance(maze, TST):
        return RoiRule(1, 1)
    return RoiRule(1, None)


def evaluate_roi_step(rule: RoiRule, finished: int, pending_points: int,
                      description: str) -> RoiStepState:
    """Decide Next / point-adding and the hint for the category being drawn.

    ``finished`` is the number of closed polygons in this category and
    ``pending_points`` the points of the shape still being drawn (a pending
    shape of three or more points is closed automatically by Next).
    """
    at_capacity = rule.maximum is not None and finished >= rule.maximum
    exact = rule.maximum is not None and rule.maximum == rule.minimum
    need = (
        f"exactly {rule.minimum} polygon" + ("" if rule.minimum == 1 else "s")
        if exact else f"at least {rule.minimum} polygon"
        + ("" if rule.minimum == 1 else "s")
    )

    if 0 < pending_points < MIN_POLYGON_POINTS:
        missing = MIN_POLYGON_POINTS - pending_points
        return RoiStepState(
            False, True,
            f"Add {missing} more point{'s' if missing != 1 else ''}, or clear "
            "the shape, to continue."
        )
    if pending_points >= MIN_POLYGON_POINTS and finished + 1 >= rule.minimum:
        return RoiStepState(
            True, True,
            "Right-click to close the shape, or continue to the next step."
        )
    if finished < rule.minimum:
        return RoiStepState(
            False, True,
            f"{description} needs {need}: click to add points, "
            "right-click to close the shape."
        )
    if at_capacity:
        return RoiStepState(
            True, False,
            f"{description} is done ({need}). Continue, or undo to redraw it."
        )
    return RoiStepState(
        True, True,
        f"{description} has {finished} polygon{'s' if finished != 1 else ''}. "
        "Draw another, or continue."
    )


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

    source_h, source_w = frame.shape[:2]
    roi_manager.set_source_dimensions((source_w, source_h))
    if source_w > 1280:
        scale = 1280 / source_w
        frame = cv2.resize(frame, (int(source_w * scale), int(source_h * scale)))
    h, w = frame.shape[:2]
    pixmap = bgr_to_qpixmap(frame)
    dialog = ReferenceLineDialog(
        pixmap, source_scale=(source_w / w, source_h / h), parent=parent
    )
    try:
        if dialog.exec() and dialog.get_line_length() > 0:
            roi_manager.set_reference_length(
                dialog.get_line_length(), REFERENCE_LINE_NAME
            )
            return True
        return False
    finally:
        dialog.deleteLater()


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
        self.roi_rule = roi_rule_for_maze(maze)
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
            
        self.base_frame = frame
        h, w = self.base_frame.shape[:2]
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
            
        self.base_pixmap = bgr_to_qpixmap(self.base_frame)
        
        self.setWindowTitle("ROI Selection")
        
        layout = QVBoxLayout()
        self.instruction_label = QLabel()
        self.instruction_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(
            "color: #7a4b00; background: #fff3cd; padding: 4px 8px; "
            "border-radius: 3px;"
        )
        
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
        
        # Enter goes to Next (see keyPressEvent), never to Undo.
        self.undo_button.setAutoDefault(False)
        self.clear_button.setAutoDefault(False)
        self.next_button.setDefault(True)
        self.undo_button.setToolTip("Undo Last Point (Backspace or Ctrl+Z)")
        self.next_button.setToolTip("Next (Enter)")

        layout.addWidget(self.instruction_label)
        layout.addWidget(self.hint_label)
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
        self._refresh_step_state()
        self.update()

    def _current_category(self):
        return self.roi_definitions[self.current_roi_index]

    def _finished_count(self) -> int:
        category, _ = self._current_category()
        return len(self.roi_manager.rois.get(category, []))

    def step_state(self) -> RoiStepState:
        _, description = self._current_category()
        return evaluate_roi_step(
            self.roi_rule, self._finished_count(),
            len(self.current_polygon_points), description
        )

    def _refresh_step_state(self):
        """Enable Next and show the hint for what is still missing."""
        if self.current_roi_index >= len(self.roi_definitions):
            return
        state = self.step_state()
        self.next_button.setEnabled(state.can_advance)
        self.hint_label.setText(state.hint)
        self.hint_label.setVisible(bool(state.hint))

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
        category, description = self._current_category()
        if 0 < len(self.current_polygon_points) < MIN_POLYGON_POINTS:
            QMessageBox.warning(
                self,
                "Incomplete ROI",
                f"{description} needs at least three points."
            )
            return
        if len(self.current_polygon_points) >= MIN_POLYGON_POINTS:
            if not self.finalize_polygon():
                return
        if not self.step_state().can_advance:
            return
        self.current_roi_index += 1
        self.start_next_roi_selection()

    def add_point(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.step_state().can_add_point:
                return
            self.current_polygon_points.append(QPointF(event.pos()))
            self._refresh_step_state()
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            if len(self.current_polygon_points) >= MIN_POLYGON_POINTS:
                self.finalize_polygon()
                self.update()

    def undo_point(self):
        """Remove the last point; with none pending, reopen the last closed shape."""
        if self.current_polygon_points:
            self.current_polygon_points.pop()
        else:
            category, _ = self._current_category()
            shapes = self.roi_manager.rois.get(category, [])
            if shapes:
                shapes.pop()
                if not shapes:
                    del self.roi_manager.rois[category]
        self._refresh_step_state()
        self.update()

    def clear_current_polygon(self):
        self.current_polygon_points = []
        self._refresh_step_state()
        self.update()

    def finalize_polygon(self) -> bool:
        """Close the shape being drawn; False if it was refused."""
        points = np.array([
            (p.x() * self.display_to_source_x, p.y() * self.display_to_source_y)
            for p in self.current_polygon_points
        ], dtype=np.int32)
        category, description = self._current_category()
        state = self.step_state()
        if not state.can_add_point:
            self.hint_label.setText(state.hint)
            return False
        area = float(cv2.contourArea(points.astype(np.float32)))
        if area < MIN_ROI_AREA_PX:
            self.hint_label.setText(
                f"That shape is too small ({area:.0f} px\u00b2; at least "
                f"{MIN_ROI_AREA_PX:.0f} px\u00b2 is needed). Clear it and draw "
                "it larger, or undo points."
            )
            self.hint_label.setVisible(True)
            return False
        self.roi_manager.add_roi(category, points)
        self.current_polygon_points = []
        self._refresh_step_state()
        return True

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.next_button.isEnabled():
                self.next_roi()
            event.accept()
        elif key == Qt.Key.Key_Backspace or event.matches(QKeySequence.StandardKey.Undo):
            self.undo_point()
            event.accept()
        else:
            super().keyPressEvent(event)

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

    def has_unsaved_work(self) -> bool:
        return ux_helpers.has_drawn_work(
            self.roi_manager, len(self.current_polygon_points)
        )

    def reject(self):
        """Esc, the window button and Cancel all end here: confirm first."""
        if self.has_unsaved_work() and not ux_helpers.confirm_discard(
            self, "the regions you have drawn",
            "Closing this window loses every polygon drawn so far."
        ):
            return
        super().reject()
