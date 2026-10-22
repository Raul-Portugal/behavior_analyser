"""``RoiEditorDialog``: the modal wrapper around the ROI editor widget.

It reads the middle frame of a video, adds OK / Cancel, refuses to finish
while a zone breaks its rule, offers the calibration line once, and asks
before discarding drawn work. ``gui.roi_selector.RoiSelectorDialog`` is the
3.x-compatible name for it.
"""
import logging
from typing import List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDialog, QHBoxLayout, QMessageBox, QPushButton,
                             QVBoxLayout)

from gui import ux_helpers
from gui.dialog_layout import fit_dialog_to_available_screen
from roi_manager import ROIManager

from .rules import MIN_POLYGON_POINTS
from .session import TOOL_LINE
from .widget import RoiEditorWidget

logger = logging.getLogger(__name__)

REFERENCE_LINE_PROMPT = (
    "Would you like to draw a reference line for distance calibration?\n\n"
    "Draw it over something of known length (e.g. an arm length). "
    "If you skip this, distances will be reported in pixels."
)


def read_middle_frame(video_path) -> np.ndarray:
    """The middle frame of a video (animals are visible there).

    Raises:
        IOError: if no frame can be read.
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
        raise IOError("Could not read frame from video.")
    return frame


class RoiEditorDialog(QDialog):
    """Edit the ROIs of one video in a dialog.

    ``initial_roi_manager`` opens the editor preloaded (the Redraw path); the
    caller reads the result with :meth:`get_roi_manager`.
    """

    def __init__(self, video_path, maze, parent=None,
                 initial_roi_manager: Optional[ROIManager] = None,
                 frame: Optional[np.ndarray] = None):
        super().__init__(parent)
        self.video_path = video_path
        self.maze = maze
        self._result_manager: Optional[ROIManager] = None
        self._calibration_declined = False
        if frame is None:
            frame = read_middle_frame(video_path)
        self.setWindowTitle("ROI Editor")

        self.editor = RoiEditorWidget(
            maze, frame, self, initial_roi_manager=initial_roi_manager
        )
        # Plain buttons, not a QDialogButtonBox: the box makes OK the default
        # button, and Enter belongs to the editor (close the shape, next zone).
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        for button in (self.ok_button, self.cancel_button):
            button.setAutoDefault(False)
            button.setDefault(False)
        self.ok_button.setToolTip("Finish and keep these regions")
        self.ok_button.clicked.connect(lambda _checked=False: self.accept())
        self.cancel_button.clicked.connect(lambda _checked=False: self.reject())
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.ok_button)
        buttons.addWidget(self.cancel_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.editor, 1)
        layout.addLayout(buttons)
        height, width = frame.shape[:2]
        fit_dialog_to_available_screen(
            self,
            preferred_size=(min(width, 1280) + 280, min(height, 960) + 190),
            minimum_size=(640, 480),
        )

    # ------------------------------------------------------------------
    def get_roi_manager(self) -> ROIManager:
        """The ROIs as accepted (or, before acceptance, as currently drawn)."""
        if self._result_manager is not None:
            return self._result_manager
        return self.editor.roi_manager()

    def has_unsaved_work(self) -> bool:
        return self.editor.has_unsaved_work()

    # ------------------------------------------------------------------
    def issues_before_finishing(self) -> List[str]:
        return self.editor.issues()

    def _may_finish(self) -> bool:
        editor = self.editor
        pending = editor.pending_points()
        if 0 < len(pending) < MIN_POLYGON_POINTS:
            QMessageBox.warning(
                self, "Incomplete ROI",
                "The shape being drawn needs at least three points. Add more "
                "points, or press Esc to cancel it."
            )
            return False
        if pending and not editor.close_shape():
            QMessageBox.warning(self, "ROI Selection", editor.session.hint())
            return False
        issues = self.issues_before_finishing()
        if issues:
            QMessageBox.warning(
                self, "ROI Selection",
                "Fix these before finishing:\n\n- " + "\n- ".join(issues)
            )
            return False
        return self._calibration_settled()

    def _calibration_settled(self) -> bool:
        """Offer the reference line once; True when the dialog may close."""
        needs_line = getattr(self.maze, "needs_reference_line", None)
        if callable(needs_line) and not needs_line():
            # Paradigms without distance metrics (e.g. TST) skip calibration.
            return True
        document = self.editor.session.document
        if document.line is not None or document.legacy_reference is not None:
            return True
        if self._calibration_declined:
            return True
        reply = QMessageBox.question(
            self, "Calibration", REFERENCE_LINE_PROMPT,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            # Stay open with the line tool; OK again once it is drawn.
            self.editor.set_tool(TOOL_LINE)
            return False
        self._calibration_declined = True
        return True

    def accept(self):
        """OK: close the pending shape, check the zones, offer calibration."""
        if not self._may_finish():
            return
        self._result_manager = self.editor.roi_manager()
        self.editor.mark_saved()
        super().accept()

    def reject(self):
        """Esc, the window button and Cancel all end here: confirm first."""
        if self.has_unsaved_work() and not ux_helpers.confirm_discard(
            self, "the regions you have drawn",
            "Closing this window loses every change made so far."
        ):
            return
        super().reject()

    def keyPressEvent(self, event):
        # Keys pressed while a button or the list has focus still reach the
        # editor's shortcuts (Enter, Ctrl+Z, Delete...); Esc falls through to
        # reject() when the editor has nothing to cancel.
        if self.editor.handle_key_event(event):
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.accept()
            return
        super().keyPressEvent(event)


__all__ = ["REFERENCE_LINE_PROMPT", "RoiEditorDialog", "read_middle_frame"]
