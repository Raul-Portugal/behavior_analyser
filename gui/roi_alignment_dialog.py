"""
gui/roi_alignment_dialog.py
Modal dialog that carries an ROI template onto another video.

Left: the template's frame with its ROIs. Right: the target video's frame
with the ROIs transformed by the current alignment. The alignment comes from
automatic registration of the two animal-free reference frames, from
landmarks the user clicks (the same arena corners on both frames, in order),
or is kept as is. The dialog returns the aligned ``ROIManager``.
"""
import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QDialog, QHBoxLayout, QLabel,
                             QMessageBox, QPushButton, QScrollArea,
                             QVBoxLayout)

from core import roi_alignment as ra
from gui.background_task import TaskCancelled, run_with_progress
from gui.dialog_layout import fit_dialog_to_available_screen
from gui.qt_image import bgr_to_qpixmap
from roi_manager import ROIManager

logger = logging.getLogger(__name__)

# Widest a frame is shown; larger frames are scaled down for display only.
PANE_MAX_WIDTH = 620
MAX_LANDMARKS = 8

METHOD_SCALE = "scale"
METHOD_AUTO = "auto"
METHOD_LANDMARKS = "landmarks"
METHOD_AS_IS = "as_is"

AutoAlign = Callable[[Callable[[], bool]], "ra.RegistrationResult"]

_LEVEL_STYLE = {
    ra.VERDICT_GOOD: "color: #1b5e20; font-weight: bold;",
    ra.VERDICT_CHECK: "color: #8a5a00; font-weight: bold;",
    ra.VERDICT_POOR: "color: #b00020; font-weight: bold;",
}
_LANDMARK_COLOR = (255, 200, 0)   # BGR
_PENDING_COLOR = (0, 215, 255)


@dataclass(frozen=True)
class AlignmentChoice:
    """What the user accepted."""
    matrix: np.ndarray
    rms: Optional[float]
    method: str


class _ClickLabel(QLabel):
    """A label reporting left clicks (in its own pixel coordinates)."""
    clicked = pyqtSignal(QPoint)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(event.pos())
        super().mousePressEvent(event)


class _Pane:
    """One displayed frame: the BGR source plus its display geometry."""

    def __init__(self, frame: np.ndarray, title: str):
        self.frame = frame
        self.height, self.width = frame.shape[:2]
        scale = min(1.0, PANE_MAX_WIDTH / float(self.width))
        self.display_width = max(1, int(round(self.width * scale)))
        self.display_height = max(1, int(round(self.height * scale)))
        # Same display-to-source mapping pattern as RoiSelectorDialog.
        self.display_to_source_x = self.width / self.display_width
        self.display_to_source_y = self.height / self.display_height
        self.title = QLabel(title)
        self.title.setStyleSheet("font-weight: bold;")
        self.label = _ClickLabel()
        self.label.setFixedSize(self.display_width, self.display_height)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.label)

    def to_source(self, point: QPoint) -> Tuple[float, float]:
        return (
            point.x() * self.display_to_source_x,
            point.y() * self.display_to_source_y,
        )

    def show(self, image: np.ndarray) -> None:
        if (self.display_width, self.display_height) != (self.width, self.height):
            image = cv2.resize(
                image, (self.display_width, self.display_height),
                interpolation=cv2.INTER_AREA,
            )
        self.label.setPixmap(bgr_to_qpixmap(image))


def _overlay_thickness(width: int) -> int:
    return max(2, int(round(width / 400.0)))


def _draw_landmarks(image: np.ndarray, points: List[Tuple[float, float]],
                    pending: Optional[Tuple[float, float]] = None) -> None:
    radius = max(5, image.shape[1] // 120)
    thickness = _overlay_thickness(image.shape[1])
    scale = max(0.6, image.shape[1] / 900.0)
    for index, (x, y) in enumerate(points, start=1):
        centre = (int(round(x)), int(round(y)))
        cv2.circle(image, centre, radius, _LANDMARK_COLOR, thickness, cv2.LINE_AA)
        cv2.putText(
            image, str(index), (centre[0] + radius + 2, centre[1] - radius),
            cv2.FONT_HERSHEY_SIMPLEX, scale, _LANDMARK_COLOR, thickness,
            cv2.LINE_AA,
        )
    if pending is not None:
        centre = (int(round(pending[0])), int(round(pending[1])))
        cv2.circle(image, centre, radius, _PENDING_COLOR, thickness, cv2.LINE_AA)
        cv2.putText(
            image, str(len(points) + 1),
            (centre[0] + radius + 2, centre[1] - radius),
            cv2.FONT_HERSHEY_SIMPLEX, scale, _PENDING_COLOR, thickness,
            cv2.LINE_AA,
        )


class RoiAlignmentDialog(QDialog):
    """Align a template's ROIs to a target video's frame."""

    def __init__(
        self,
        template_manager: ROIManager,
        template_frame: np.ndarray,
        target_frame: np.ndarray,
        auto_align: Optional[AutoAlign] = None,
        template_name: str = "Template",
        target_name: str = "Video",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Align template to {target_name}")
        self.template_manager = template_manager
        self._auto_align = auto_align
        self.landmarks: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        self._pending_template_point: Optional[Tuple[float, float]] = None
        self.landmark_mode = False

        self._template = _Pane(template_frame, f"{template_name} (template)")
        self._target = _Pane(target_frame, f"{target_name} (aligned preview)")
        self.target_dims = (self._target.width, self._target.height)

        self._matrix = self._initial_matrix()
        self._rms: Optional[float] = None
        self._method = METHOD_SCALE
        self._level = ra.VERDICT_CHECK
        self._build_ui()
        self._set_alignment(
            self._matrix, None, METHOD_SCALE,
            self._initial_note(),
        )

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        self.instruction_label = QLabel()
        self.instruction_label.setWordWrap(True)
        self.quality_label = QLabel()
        self.quality_label.setWordWrap(True)

        panes = QHBoxLayout()
        for pane in (self._template, self._target):
            column = QVBoxLayout()
            column.addWidget(pane.title)
            column.addWidget(pane.scroll, 1)
            panes.addLayout(column, 1)
        self._template.label.clicked.connect(self._on_template_clicked)
        self._target.label.clicked.connect(self._on_target_clicked)

        self.auto_button = QPushButton("Auto-align")
        self.auto_button.setToolTip(
            "Match the arena in the two animal-free background frames."
        )
        self.auto_button.setEnabled(self._auto_align is not None)
        self.landmark_button = QPushButton("Pick landmarks…")
        self.landmark_button.setCheckable(True)
        self.landmark_button.setToolTip(
            "Click the same 3-4 fixed points (e.g. arena corners) on both "
            "frames, in the same order."
        )
        self.clear_button = QPushButton("Clear landmarks")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Shift + rotate + scale", ra.MODE_SIMILARITY)
        self.mode_combo.addItem("Affine (needs 3+ points)", ra.MODE_AFFINE)
        self.mode_combo.addItem("Perspective (needs 4+ points)", ra.MODE_HOMOGRAPHY)
        self.mode_combo.setToolTip("How landmarks are turned into an alignment.")
        self.as_is_button = QPushButton("Use as is")
        self.as_is_button.setToolTip(
            "Keep the ROIs exactly where they are in pixels."
        )
        self.accept_button = QPushButton("Accept")
        self.accept_button.setDefault(True)
        self.cancel_button = QPushButton("Cancel")

        buttons = QHBoxLayout()
        for widget in (self.auto_button, self.landmark_button, self.clear_button,
                       self.mode_combo, self.as_is_button):
            buttons.addWidget(widget)
        buttons.addStretch(1)
        buttons.addWidget(self.accept_button)
        buttons.addWidget(self.cancel_button)

        layout.addWidget(self.instruction_label)
        layout.addLayout(panes, 1)
        layout.addWidget(self.quality_label)
        layout.addLayout(buttons)
        self.setLayout(layout)
        fit_dialog_to_available_screen(
            self,
            preferred_size=(
                self._template.display_width + self._target.display_width + 80,
                max(self._template.display_height, self._target.display_height) + 190,
            ),
            minimum_size=(640, 420),
        )

        self.auto_button.clicked.connect(self.run_auto_align)
        self.landmark_button.toggled.connect(self._on_landmark_toggled)
        self.clear_button.clicked.connect(self.clear_landmarks)
        self.mode_combo.currentIndexChanged.connect(self._refit_landmarks)
        self.as_is_button.clicked.connect(self.use_as_is)
        self.accept_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    # --------------------------------------------------------- alignment state

    def _initial_matrix(self) -> np.ndarray:
        """Scale the template to the target's resolution (identity if equal)."""
        source = self._template
        matrix = ra.identity()
        matrix[0, 0] = self._target.width / float(source.width)
        matrix[1, 1] = self._target.height / float(source.height)
        return matrix

    def _initial_note(self) -> str:
        if (self._template.width, self._template.height) == self.target_dims:
            return ("Same frame size: the ROIs are shown unchanged. Use "
                    "Auto-align or Pick landmarks if the camera moved.")
        return (
            f"Frame sizes differ ({self._template.width}x{self._template.height} "
            f"vs {self.target_dims[0]}x{self.target_dims[1]}): the ROIs are "
            "scaled to the new size. Align them if the camera also moved."
        )

    def _set_alignment(self, matrix: np.ndarray, rms: Optional[float],
                       method: str, note: str = "") -> None:
        self._matrix = np.asarray(matrix, dtype=np.float64)
        self._rms = rms
        self._method = method
        self._update_quality(note)
        self.render()

    def _show_message(self, text: str, level: Optional[str] = None) -> None:
        """Show a status line; ``level`` colours it (None: neutral)."""
        self.quality_text = text
        self._level = level or ra.VERDICT_CHECK
        self.quality_label.setText(text)
        self.quality_label.setStyleSheet(_LEVEL_STYLE.get(level, "") if level else "")

    def _update_quality(self, note: str) -> None:
        if self._rms is None:
            self._show_message(note)
            return
        level, text = ra.describe_residual(self._rms)
        self._show_message(
            f"Alignment residual {text}" + (f". {note}" if note else ""), level
        )

    @property
    def quality_level(self) -> str:
        return self._level

    def get_matrix(self) -> np.ndarray:
        return self._matrix.copy()

    def get_residual(self) -> Optional[float]:
        return self._rms

    def get_method(self) -> str:
        return self._method

    def get_choice(self) -> AlignmentChoice:
        return AlignmentChoice(self.get_matrix(), self._rms, self._method)

    def get_roi_manager(self) -> ROIManager:
        """The template ROIs moved by the accepted alignment (a new manager)."""
        return ra.transform_roi_manager(
            self.template_manager, self._matrix, self.target_dims
        )

    # ------------------------------------------------------------------ actions

    def use_as_is(self) -> None:
        """Keep every ROI at its template pixels."""
        self.clear_landmarks(render=False)
        self._set_alignment(
            ra.identity(), None, METHOD_AS_IS,
            "Using the ROIs as is, at their template pixel positions.",
        )

    def run_auto_align(self) -> None:
        if self._auto_align is None:
            return
        try:
            result = run_with_progress(
                self, "Aligning the template to this video...",
                self._auto_align, title="Auto-align",
            )
        except TaskCancelled:
            return
        except Exception as exc:
            logger.warning("Auto-align failed", exc_info=True)
            QMessageBox.warning(
                self, "Auto-align Failed",
                f"The reference frames could not be compared:\n{exc}",
            )
            return
        self.apply_registration(result)

    def apply_registration(self, result: "ra.RegistrationResult") -> None:
        """Show an automatic registration; unusable results change nothing."""
        if not result.usable:
            self._show_message(
                f"Auto-align failed: {result.message or 'no alignment found'}. "
                "Pick landmarks instead.", ra.VERDICT_POOR,
            )
            return
        self.clear_landmarks(render=False)
        self._set_alignment(
            result.matrix, result.rms, METHOD_AUTO,
            f"{result.inliers} matched features"
            + (" - review the overlay." if result.verdict != ra.VERDICT_GOOD else "."),
        )
        if result.verdict == ra.VERDICT_POOR:
            self._show_message(
                f"{self.quality_text} The match is weak: check carefully.",
                ra.VERDICT_POOR,
            )

    # -------------------------------------------------------------- landmarks

    def _selected_mode(self) -> str:
        return self.mode_combo.currentData()

    def _on_landmark_toggled(self, checked: bool) -> None:
        self.landmark_mode = bool(checked)
        self._pending_template_point = None
        self._refresh_instruction()
        self.render()

    def start_landmarks(self) -> None:
        self.landmark_button.setChecked(True)

    def clear_landmarks(self, render: bool = True) -> None:
        self.landmarks = []
        self._pending_template_point = None
        self._refresh_instruction()
        if render:
            self.render()

    def _refresh_instruction(self) -> None:
        if not self.landmark_mode:
            self.instruction_label.setText(
                "Auto-align from the background frames, or pick landmarks "
                "(the same fixed points on both frames). Accept when the "
                "ROIs on the right line up with the arena."
            )
            return
        number = len(self.landmarks) + 1
        if self._pending_template_point is None:
            self.instruction_label.setText(
                f"Landmark {number}: click the point on the TEMPLATE frame "
                f"(left). Use at least {ra.MIN_POINTS[self._selected_mode()] + 1} "
                "points so the fit can be checked."
            )
        else:
            self.instruction_label.setText(
                f"Landmark {number}: now click the SAME point on the "
                "video frame (right)."
            )

    def _on_template_clicked(self, point: QPoint) -> None:
        if self.landmark_mode:
            self.add_template_landmark(*self._template.to_source(point))

    def _on_target_clicked(self, point: QPoint) -> None:
        if self.landmark_mode:
            self.add_target_landmark(*self._target.to_source(point))

    def add_template_landmark(self, x: float, y: float) -> None:
        """Landmark step 1: a point on the template frame (source pixels)."""
        if len(self.landmarks) >= MAX_LANDMARKS:
            return
        self._pending_template_point = (float(x), float(y))
        self._refresh_instruction()
        self.render()

    def add_target_landmark(self, x: float, y: float) -> None:
        """Landmark step 2: the matching point on the video frame."""
        if self._pending_template_point is None:
            self.instruction_label.setText(
                "Click the point on the TEMPLATE frame (left) first."
            )
            return
        self.landmarks.append((self._pending_template_point, (float(x), float(y))))
        self._pending_template_point = None
        self._refresh_instruction()
        self._refit_landmarks()

    def _refit_landmarks(self, *_args) -> None:
        mode = self._selected_mode()
        count = len(self.landmarks)
        if count == 0:
            self.render()
            return
        needed = ra.MIN_POINTS[mode]
        if count < needed:
            self._show_message(
                f"{count} landmark{'s' if count != 1 else ''} picked; "
                f"{needed} are needed for this fit."
            )
            self.render()
            return
        src = [pair[0] for pair in self.landmarks]
        dst = [pair[1] for pair in self.landmarks]
        try:
            matrix, rms = ra.estimate_transform(src, dst, mode)
        except ValueError as exc:
            self._show_message(str(exc), ra.VERDICT_POOR)
            self.render()
            return
        note = ""
        if count == needed:
            note = "Exact fit; add one more landmark to verify it"
        self._set_alignment(matrix, rms, METHOD_LANDMARKS, note)

    # --------------------------------------------------------------- rendering

    def render(self) -> None:
        thickness_t = _overlay_thickness(self._template.width)
        template_image = self.template_manager.draw_on_frame(
            self._template.frame, thickness=thickness_t
        ).copy()
        _draw_landmarks(
            template_image, [pair[0] for pair in self.landmarks],
            self._pending_template_point,
        )
        self._template.show(template_image)

        aligned = self.get_roi_manager()
        target_image = aligned.draw_on_frame(
            self._target.frame, thickness=_overlay_thickness(self._target.width)
        ).copy()
        _draw_landmarks(target_image, [pair[1] for pair in self.landmarks])
        self._target.show(target_image)


def align_template_with_dialog(
    parent,
    template_manager: ROIManager,
    template_frame: np.ndarray,
    target_frame: np.ndarray,
    auto_align: Optional[AutoAlign] = None,
    template_name: str = "Template",
    target_name: str = "Video",
) -> Optional[AlignmentChoice]:
    """Run the dialog modally; the user's choice, or None if cancelled.

    ``choice.matrix`` maps template pixels to the video's pixels; apply it
    with ``core.roi_alignment.transform_roi_manager``.
    """
    dialog = RoiAlignmentDialog(
        template_manager, template_frame, target_frame, auto_align,
        template_name, target_name, parent,
    )
    try:
        if dialog.exec():
            return dialog.get_choice()
        return None
    finally:
        dialog.deleteLater()
