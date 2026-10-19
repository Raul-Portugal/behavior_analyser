"""
Compatibility layer for the 3.x ROI drawing dialogs.

The drawing itself now lives in :mod:`gui.roi_editor` (a zoomable editor with
undo, a zone list and a line tool). This module keeps the names and behaviour
the rest of the application and its tests use:

* ``RoiSelectorDialog(video_path, maze, parent=None, initial_roi_manager=None)``
  is a ``RoiEditorDialog``; ``get_roi_manager()`` returns the ROIs.
* ``ReferenceLineDialog`` / ``offer_reference_line`` / ``apply_reference_line``
  draw and store a calibration line, now on a zoomable canvas.
* The ROI rules (``RoiRule``, ``roi_rule_for_maze``, ``evaluate_roi_step``,
  ``MIN_ROI_AREA_PX``) are re-exported.
"""
from typing import List, Optional, Tuple

import cv2  # noqa: F401  (the tests patch ``gui.roi_selector.cv2.VideoCapture``)
from PyQt6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QVBoxLayout)

from roi_manager import ROIManager
from gui.dialog_layout import fit_dialog_to_available_screen
from gui.qt_image import bgr_to_qpixmap
from gui.roi_editor import RoiEditorWidget
from gui.roi_editor.dialog import (REFERENCE_LINE_PROMPT, RoiEditorDialog,
                                   read_middle_frame)
from gui.roi_editor.model import REFERENCE_LINE_NAME
from gui.roi_editor.rules import (ARENA_RULE, ARENA_STEP_DESCRIPTION,
                                  MIN_POLYGON_POINTS, MIN_ROI_AREA_PX,
                                  RoiRule, RoiStepState, evaluate_roi_step,
                                  roi_rule_for_maze)
from roi_manager import ARENA_CATEGORY

__all__ = [
    "ARENA_RULE", "ARENA_STEP_DESCRIPTION", "MIN_POLYGON_POINTS",
    "MIN_ROI_AREA_PX", "REFERENCE_LINE_NAME", "REFERENCE_LINE_PROMPT",
    "ReferenceLineDialog", "RoiRule", "RoiSelectorDialog", "RoiStepState",
    "apply_reference_line", "evaluate_roi_step", "offer_reference_line",
    "roi_rule_for_maze",
]


class ReferenceLineDialog(QDialog):
    """A dialog to draw a single line for distance calibration.

    ``pixmap`` is the image shown and ``source_scale`` the factor from its
    pixels to source pixels, so the reported endpoints are always in source
    pixels. The canvas zooms and pans like the ROI editor's.
    """
    def __init__(self, pixmap, source_scale=(1.0, 1.0), parent=None):
        super().__init__(parent)
        self.setWindowTitle("Draw Reference Line")
        self.base_pixmap = pixmap
        self.source_scale_x = float(source_scale[0])
        self.source_scale_y = float(source_scale[1])
        self.line_length = 0.0
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.instruction_label = QLabel(
            "Click two points (or drag) to define a line of known length "
            "(e.g., an arm width). Ctrl+wheel zooms; drag an end to adjust it."
        )
        self.instruction_label.setWordWrap(True)
        self.editor = RoiEditorWidget(
            None, self.base_pixmap, self,
            image_scale=(self.source_scale_x, self.source_scale_y),
            line_only=True,
        )
        button_layout = QHBoxLayout()
        self.reset_button = QPushButton("Reset")
        self.accept_button = QPushButton("Accept")
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.accept_button)
        layout.addWidget(self.instruction_label)
        layout.addWidget(self.editor, 1)
        layout.addLayout(button_layout)
        self.setLayout(layout)
        fit_dialog_to_available_screen(
            self,
            preferred_size=(
                int(self.base_pixmap.width() * self.source_scale_x) + 50,
                int(self.base_pixmap.height() * self.source_scale_y) + 190,
            ),
            minimum_size=(480, 360),
        )
        self.reset_button.setAutoDefault(False)
        self.accept_button.setAutoDefault(False)
        self.reset_button.clicked.connect(self.reset)
        self.accept_button.clicked.connect(self.accept_line)

    def add_point(self, event):
        """Place the next end from a click at ``event.pos()`` (image pixels)."""
        pos = event.pos()
        self.editor.add_line_point(pos.x() * self.source_scale_x,
                                   pos.y() * self.source_scale_y)

    def reset(self):
        session = self.editor.session
        session.cancel_draft()
        if session.document.line is not None:
            session.delete_line()
        self.line_length = 0.0

    def accept_line(self):
        if self.editor.line_points() is not None:
            self.accept()
        else:
            QMessageBox.warning(self, "Incomplete", "Please click two points to define a line.")

    def get_line_length(self):
        return self.editor.line_length()

    def get_line_points(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """The two endpoints in SOURCE pixels, or None until both exist."""
        return self.editor.line_points()


def apply_reference_line(roi_manager: ROIManager, dialog) -> bool:
    """Store the dialog's line (endpoints and length) on the ROI manager.

    Returns True only if the user drew a valid, non-degenerate line.
    """
    points = dialog.get_line_points()
    if points is None or dialog.get_line_length() <= 0:
        return False
    try:
        roi_manager.set_reference_line(points[0], points[1], REFERENCE_LINE_NAME)
    except ValueError:
        return False
    return True


def offer_reference_line(parent, video_path, roi_manager: ROIManager) -> bool:
    """Ask for a reference line on a middle frame of the video.

    Sets the ROI manager's reference length (in source pixels) and returns
    True only if the user drew a valid line.
    """
    try:
        frame = read_middle_frame(video_path)
    except IOError:
        return False

    source_h, source_w = frame.shape[:2]
    roi_manager.set_source_dimensions((source_w, source_h))
    pixmap = bgr_to_qpixmap(frame)
    dialog = ReferenceLineDialog(pixmap, source_scale=(1.0, 1.0), parent=parent)
    try:
        if dialog.exec():
            return apply_reference_line(roi_manager, dialog)
        return False
    finally:
        dialog.deleteLater()


class RoiSelectorDialog(RoiEditorDialog):
    """The 3.x name of the ROI dialog; see :class:`RoiEditorDialog`.

    Passing ``initial_roi_manager`` opens it preloaded for editing instead of
    on a blank canvas.
    """
    def __init__(self, video_path: str, maze, parent=None,
                 initial_roi_manager: Optional[ROIManager] = None):
        super().__init__(video_path, maze, parent,
                         initial_roi_manager=initial_roi_manager)
        self.setWindowTitle("ROI Selection")

    # -- 3.x attributes other code and tests read ---------------------------
    @property
    def zone_definitions(self) -> List[Tuple[str, str]]:
        return list(self.editor.session.zone_definitions)

    @property
    def arena_step_enabled(self) -> bool:
        return self.editor.session.arena_enabled

    @property
    def roi_definitions(self) -> List[Tuple[str, str]]:
        """Every drawable step: the arena (when enabled) then the zones."""
        steps = list(self.zone_definitions)
        if self.arena_step_enabled:
            steps.insert(0, (ARENA_CATEGORY, ARENA_STEP_DESCRIPTION))
        return steps

    @property
    def roi_manager(self) -> ROIManager:
        """The ROIs as currently drawn (a fresh manager on every access)."""
        return self.get_roi_manager()
