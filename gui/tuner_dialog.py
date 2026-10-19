"""
gui/tuner_dialog.py
Dialog for tuning detection with live quality feedback.

The tuning UI lives in ``gui.tuners.detection_tuner.DetectionTunerWidget`` so
the workspace can embed it; ``TunerDialog`` is the thin modal wrapper that adds
OK/Cancel and the discard confirmation.
"""
from typing import Optional

import numpy as np
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QPushButton, QVBoxLayout

from core.models import DetectionConfig
from gui import ux_helpers
from gui.dialog_layout import fit_dialog_to_available_screen
from gui.tuners.detection_tuner import (  # noqa: F401  (re-exported API)
    DetectionTunerWidget,
    QuickScanWorker,
    ScanSample,
    quick_scan_windows,
)
from roi_manager import ROIManager


def _forwarded(name: str) -> property:
    """Read/write property that forwards ``name`` to the wrapped widget."""
    return property(
        lambda self: getattr(self.widget, name),
        lambda self, value: setattr(self.widget, name, value),
    )


class TunerDialog(QDialog):
    """Dialog for tuning detection with live quality feedback."""

    # Attributes of the embedded widget that callers and tests use directly.
    video_path = _forwarded("video_path")
    ref_frame = _forwarded("ref_frame")
    roi_manager = _forwarded("roi_manager")
    config = _forwarded("config")
    video_info = _forwarded("video_info")
    analysis_ref = _forwarded("analysis_ref")
    arena_mask = _forwarded("arena_mask")
    cap = _forwarded("cap")
    current_frame_idx = _forwarded("current_frame_idx")
    quality_monitor = _forwarded("quality_monitor")
    scan_worker = _forwarded("scan_worker")
    stats_timer = _forwarded("stats_timer")
    sliders = _forwarded("sliders")
    slider_labels = _forwarded("slider_labels")
    frame_slider = _forwarded("frame_slider")
    frame_label = _forwarded("frame_label")
    main_view = _forwarded("main_view")
    diff_view = _forwarded("diff_view")
    thresh_view = _forwarded("thresh_view")
    stats_label = _forwarded("stats_label")
    recommendation_label = _forwarded("recommendation_label")
    test_scan_button = _forwarded("test_scan_button")
    control_scroll = _forwarded("control_scroll")

    def __init__(self, video_path: str, ref_frame: np.ndarray,
                 roi_manager: ROIManager, parent=None,
                 initial_config: Optional[DetectionConfig] = None):
        super().__init__(parent)
        self.setWindowTitle("Detection Tuner")
        self.widget = DetectionTunerWidget(
            video_path, ref_frame, roi_manager,
            initial_config=initial_config, parent=self,
        )

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.accept_button = QPushButton("OK")
        self.accept_button.setDefault(True)
        self.accept_button.clicked.connect(self.accept)
        button_layout.addWidget(self.accept_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        root_layout = QVBoxLayout()
        root_layout.addWidget(self.widget, 1)
        root_layout.addLayout(button_layout)
        self.setLayout(root_layout)
        fit_dialog_to_available_screen(
            self,
            preferred_size=(1400, 900),
            minimum_size=(640, 480),
        )

    # -- forwarded behaviour -------------------------------------------------
    @staticmethod
    def format_slider_value(*args, **kwargs) -> str:
        return DetectionTunerWidget.format_slider_value(*args, **kwargs)

    def on_frame_slider_change(self, value: int):
        self.widget.on_frame_slider_change(value)

    def on_parameter_change(self, value: int = 0):
        self.widget.on_parameter_change(value)

    def update_detection(self):
        self.widget.update_detection()

    def update_quality_stats(self):
        self.widget.update_quality_stats()

    def run_quick_scan(self):
        self.widget.run_quick_scan()

    def keyPressEvent(self, event):
        # Navigation keys also work while an OK/Cancel button has focus.
        if not self.widget.handle_navigation_key(event):
            super().keyPressEvent(event)

    def get_detection_config(self) -> DetectionConfig:
        return self.widget.detection_config()

    def has_unsaved_work(self) -> bool:
        """True once a detection parameter was moved from its initial value."""
        return self.widget.has_unsaved_changes()

    def reject(self):
        if self.has_unsaved_work() and not ux_helpers.confirm_discard(
            self, "the detection tuning",
            "The detection parameters you adjusted will be lost."
        ):
            return
        super().reject()

    def _release_resources(self):
        """Stop background work and free the capture (safe to repeat)."""
        self.widget.shutdown()

    def done(self, result: int):
        # accept()/reject() end up here without going through closeEvent.
        self._release_resources()
        super().done(result)

    def closeEvent(self, event):
        self._release_resources()
        super().closeEvent(event)
