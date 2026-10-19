"""Calibration panel: reference-line status, real-world length, scale."""
from typing import List, Optional

from PyQt6.QtWidgets import QDoubleSpinBox, QFormLayout, QLabel

from core.calibration import (REFERENCE_APPROXIMATE, REFERENCE_DRAWN,
                              REFERENCE_LEGACY, REFERENCE_NONE,
                              ReferenceStatus, describe_reference,
                              scale_factor_from_cm)
from gui.setup_panels.base import SetupPanel
from mazes.base_maze import STEP_CALIBRATION

WARNING_STYLE = "color: #b00020; font-weight: bold;"


class CalibrationPanel(SetupPanel):
    """Converts the drawn reference line into cm per pixel.

    The line itself is drawn in the ROI editor; this panel shows whether one
    exists (drawn, none, a legacy bounding box, or approximate after an
    alignment), takes the real-world length of that line and shows the scale
    that results. Without a line the length cannot be set and distances are
    reported in pixels.
    """

    step_id = STEP_CALIBRATION

    def __init__(self, parent=None, real_world_cm: Optional[float] = None,
                 preserve_exact: bool = True):
        super().__init__(parent, preserve_exact)
        # Optional explicit cm length to show instead of deriving it from
        # the state (the verification flow re-fills it after a redraw).
        self.real_world_cm_override = real_world_cm
        self._reference_px = 0.0
        self._status: ReferenceStatus = describe_reference(None)

        self.reference_caption = QLabel("Reference:")
        self.reference_label = QLabel()
        self.scale_spin = QDoubleSpinBox(
            decimals=4, suffix=" cm", minimum=0, maximum=1000
        )
        self.scale_spin.setToolTip(
            "How long the drawn reference line is in the real world."
        )
        self.scale_label = QLabel()
        self.note_label = QLabel()
        self.note_label.setWordWrap(True)

        self.form = QFormLayout(self)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.addRow(self.reference_caption, self.reference_label)
        self.form.addRow("Real-world length:", self.scale_spin)
        self.form.addRow("Scale:", self.scale_label)
        self.form.addRow(self.note_label)

        self.scale_spin.valueChanged.connect(self._on_length_changed)

    # ------------------------------------------------------------------
    @property
    def status(self) -> ReferenceStatus:
        """What the loaded ROIs say about the reference line."""
        return self._status

    def real_world_cm(self) -> float:
        """The cm length typed for the reference line (0 if none)."""
        return float(self._exact.get(self.scale_spin))

    def scale_factor(self) -> float:
        """cm per pixel for the current length (0.0: pixel units)."""
        return scale_factor_from_cm(self.real_world_cm(), self._reference_px)

    def _on_length_changed(self, *_args) -> None:
        self._refresh_scale_label()
        self._emit_changed()

    def _refresh_scale_label(self) -> None:
        scale = self.scale_factor()
        if scale > 0:
            self.scale_label.setText(
                f"{scale:.5f} cm/px ({1.0 / scale:.2f} px/cm)"
            )
        else:
            self.scale_label.setText("Pixels (no calibration)")

    def load(self, state) -> None:
        manager = state.roi_manager
        self._status = describe_reference(manager)
        self._reference_px = self._status.length_px
        self._loading = True
        try:
            has_line = self._status.kind != REFERENCE_NONE
            self.scale_spin.setEnabled(has_line)
            self.reference_label.setText(self._status.text)
            self.note_label.setText(self._status.note)
            self.note_label.setStyleSheet(
                WARNING_STYLE if self._status.warning else ""
            )
            self.note_label.setVisible(bool(self._status.note))
            prefill = self.real_world_cm_override
            if prefill is None:
                prefill = state.scale_cm if has_line else 0.0
            self._exact.set(self.scale_spin, prefill if has_line else 0.0)
            self._refresh_scale_label()
        finally:
            self._loading = False

    def issues(self) -> List[str]:
        if self.scale_spin.value() > 0 and self._reference_px <= 0:
            return [
                "A real-world length needs a reference line: draw one in "
                "the regions step"
            ]
        return []

    def write(self, state) -> None:
        """Write the length unconditionally (``apply`` checks issues first)."""
        state.scale_cm = self.real_world_cm()

    def apply(self, state) -> bool:
        if self.issues():
            return False
        self.write(state)
        return self._finish_apply(state)


__all__ = [
    "CalibrationPanel", "REFERENCE_APPROXIMATE", "REFERENCE_DRAWN",
    "REFERENCE_LEGACY", "REFERENCE_NONE",
]
