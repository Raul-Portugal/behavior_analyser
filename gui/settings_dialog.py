"""
PyQt dialog for setting final analysis parameters.

The controls live in the shared setup panels (``TimeWindowPanel``,
``CalibrationPanel`` and ``ReviewPanel``); this dialog hosts them with the
OK / Cancel (or Accept / Redraw / Check alignment / Skip) buttons and keeps
the historic attribute and method names.
"""
from typing import Optional

from PyQt6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout, QLabel,
                             QMessageBox, QScrollArea, QVBoxLayout, QWidget)

from core.calibration import scale_factor_from_cm
from core.setup_state import SetupState
from core.video import VideoHandler
from gui import ux_helpers
from gui.dialog_layout import fit_dialog_to_available_screen
from gui.setup_panels.calibration_panel import CalibrationPanel
from gui.setup_panels.review_panel import ReviewPanel
from gui.setup_panels.time_window_panel import (SLIDER_STEPS_PER_SECOND,
                                                TimeWindowPanel)
from mazes import AVAILABLE_MAZES, paradigm_analysis_kind
from mazes.base_maze import ANALYSIS_MOTION

__all__ = ["SettingsDialog", "SLIDER_STEPS_PER_SECOND", "is_tst_mode"]

# Used for the hooks only when neither the settings nor the caller name a
# paradigm: one global time window, calibration offered.
GENERIC_PARADIGM = "Freestyle / Open Field"

# Time-window behaviour the dialog exposes under its historic names.
_TIME_PANEL_METHODS = (
    "on_slider_change", "on_spinbox_change", "update_preview",
    "_seconds_to_slider", "_slider_to_seconds", "_set_quietly",
)
_TIME_PANEL_WIDGETS = (
    "preview_label", "start_slider", "start_spinbox", "end_slider",
    "end_spinbox", "mouse_time_widgets",
)


def is_tst_mode(paradigm_id: Optional[str], maze_name: Optional[str] = None) -> bool:
    """TST mode from the saved paradigm, falling back to the selected maze."""
    return paradigm_analysis_kind(paradigm_id or maze_name) == ANALYSIS_MOTION


class SettingsDialog(QDialog):
    """A comprehensive dialog for setting analysis parameters."""

    def __init__(self, video_path: str, settings, is_verification=False, parent=None,
                 real_world_cm: Optional[float] = None,
                 maze_name: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Analysis Settings")
        # Optional explicit cm length for the reference line; otherwise it is
        # derived from settings.scale_factor * reference length.
        self.real_world_cm = real_world_cm

        # Raises for a missing or unreadable video, as it always did.
        self.video_info = VideoHandler(video_path)
        self.settings = settings
        self.result_action = "cancel"

        # The paradigm comes from the settings or the selected maze, never
        # from ROI names: a Freestyle zone called "mouse_1" is not a TST
        # mouse. Only the maze's hooks are used here.
        paradigm = settings.paradigm_id or maze_name
        if paradigm not in AVAILABLE_MAZES:
            paradigm = GENERIC_PARADIGM
        self._state = SetupState.from_batch_settings(
            settings, maze_name=paradigm, video_path=video_path
        )
        # The dialog edits times, calibration and the validation video, not
        # the paradigm parameters, so those play no part here.
        self._state.algorithm_parameters = {}
        self._state.legacy_untagged = False
        self.maze = self._state.build_maze()
        self.is_tst = self.maze.has_per_roi_times

        self.time_panel = TimeWindowPanel(preserve_exact=False)
        self.calibration_panel = CalibrationPanel(
            real_world_cm=real_world_cm, preserve_exact=False
        )
        self.review_panel = ReviewPanel(show_summary=False)
        for panel in (self.time_panel, self.calibration_panel,
                      self.review_panel):
            panel.load(self._state)
        for name in _TIME_PANEL_WIDGETS:
            setattr(self, name, getattr(self.time_panel, name))
        for name in _TIME_PANEL_METHODS:
            setattr(self, name, getattr(self.time_panel, name))
        self.scale_spin = self.calibration_panel.scale_spin
        self.timelapse_check = self.review_panel.timelapse_check

        # Per-video verification has its own "what now?" prompt in the main
        # window, so only the one-off review confirms before discarding.
        self._confirm_discard = not is_verification
        self.init_ui(is_verification)
        self.update_preview(self.settings.start_time)
        self._initial_state = self._current_state()

    @property
    def cap(self):
        """The preview's video capture (opened on first use)."""
        return self.time_panel.cap

    def init_ui(self, is_verification: bool):
        main_layout = QVBoxLayout()

        # 1 + 2. Video preview and time configuration
        main_layout.addWidget(self.time_panel)

        # 3. Other Settings
        other_layout = QFormLayout()
        if self.maze.reports_distance:
            other_layout.addRow(self.calibration_panel)
        other_layout.addRow(self.review_panel)
        main_layout.addLayout(other_layout)

        # 4. Buttons
        self.button_box = QDialogButtonBox()
        if is_verification:
            self.accept_btn = self.button_box.addButton("Accept", QDialogButtonBox.ButtonRole.AcceptRole)
            self.redraw_btn = self.button_box.addButton("Redraw ROIs", QDialogButtonBox.ButtonRole.ActionRole)
            self.align_btn = self.button_box.addButton("Check alignment", QDialogButtonBox.ButtonRole.ActionRole)
            self.align_btn.setToolTip(
                "Align the template's regions to this video if the camera "
                "moved or the resolution differs"
            )
            self.skip_btn = self.button_box.addButton("Skip Video", QDialogButtonBox.ButtonRole.DestructiveRole)
        else:
            self.accept_btn = self.button_box.addButton("OK", QDialogButtonBox.ButtonRole.AcceptRole)
        self.cancel_btn = self.button_box.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        content_widget = QWidget()
        content_widget.setLayout(main_layout)
        self.content_scroll = QScrollArea()
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setWidget(content_widget)

        root_layout = QVBoxLayout()
        root_layout.addWidget(self.content_scroll, 1)
        root_layout.addWidget(self.button_box)
        self.setLayout(root_layout)
        fit_dialog_to_available_screen(
            self,
            preferred_size=(700, 600),
            minimum_size=(480, 360),
        )

        self.accept_btn.clicked.connect(self.on_accept)
        self.cancel_btn.clicked.connect(self.reject)
        if hasattr(self, 'redraw_btn'):
            self.redraw_btn.clicked.connect(self.on_redraw)
            self.align_btn.clicked.connect(self.on_align)
            self.skip_btn.clicked.connect(self.on_skip)

    @staticmethod
    def _note_label(text: str, warning: bool = False) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        if warning:
            label.setStyleSheet("color: #b00020; font-weight: bold;")
        return label

    def _current_state(self) -> tuple:
        return (
            self.time_panel.current_values(), self.get_real_world_cm(),
            self.timelapse_check.isChecked(),
        )

    def has_unsaved_work(self) -> bool:
        return ux_helpers.should_confirm_discard(
            self._initial_state, self._current_state()
        )

    def reject(self):
        if (
            self._confirm_discard
            and self.has_unsaved_work()
            and not ux_helpers.confirm_discard(self, "the changes to these settings")
        ):
            return
        super().reject()

    def get_real_world_cm(self) -> float:
        """Return the cm length typed for the reference line (0 if none)."""
        if not self.maze.reports_distance:
            return 0.0
        return self.calibration_panel.real_world_cm()

    def on_accept(self):
        issues = self.time_panel.issues()
        if issues:
            QMessageBox.warning(self, "Invalid Time Range", issues[0])
            return
        self.result_action = "accept"
        self.accept()
    def on_redraw(self):
        self.result_action = "redraw"
        self.accept()
    def on_align(self):
        self.result_action = "align"
        self.accept()
    def on_skip(self):
        self.result_action = "skip"
        self.accept()

    def get_settings(self):
        state = self._state
        self.time_panel.write(state)
        if self.maze.reports_distance:
            self.calibration_panel.write(state)
        state.create_timelapse = self.timelapse_check.isChecked()

        settings = self.settings
        settings.start_time = state.start_time
        settings.end_time = state.end_time
        if self.maze.has_per_roi_times:
            settings.per_roi_times = dict(state.per_roi_times)
        # Zero for paradigms that report no distance (the TST).
        settings.scale_factor = (
            scale_factor_from_cm(
                self.calibration_panel.real_world_cm(),
                settings.roi_manager.reference_length_pixels,
            ) if self.maze.reports_distance else 0.0
        )
        settings.create_timelapse = state.create_timelapse
        return settings

    def done(self, result):
        """Release the capture on accept, reject and window close alike.

        QDialog.accept()/reject() never reach closeEvent, so the capture
        (which keeps the video file locked on Windows) is freed here.  The
        widgets stay alive so callers can still read get_settings() after
        exec() returns.
        """
        self._release_capture()
        super().done(result)

    def _release_capture(self):
        self.time_panel.release_capture()

    def closeEvent(self, event):
        # QDialog only routes close through done() while visible.
        self._release_capture()
        super().closeEvent(event)
