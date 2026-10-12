"""
PyQt dialog for setting final analysis parameters.
Updated to ensure all TST spinboxes trigger video preview updates.
"""
from typing import Optional

import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox,
                             QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
                             QSlider, QVBoxLayout, QGroupBox, QWidget, QScrollArea,
                             QMessageBox)

from core.calibration import (LEGACY_REFERENCE_WARNING, NO_CALIBRATION_NOTE,
                              real_world_cm_from_scale, scale_factor_from_cm)
from core.video import VideoHandler
from gui import ux_helpers
from gui.dialog_layout import fit_dialog_to_available_screen
from gui.qt_image import bgr_to_qpixmap
from mazes.tst import TST

SLIDER_STEPS_PER_SECOND = 10  # sliders move in 0.1 s units


def is_tst_mode(paradigm_id: Optional[str], maze_name: Optional[str] = None) -> bool:
    """TST mode from the saved paradigm, falling back to the selected maze."""
    return (paradigm_id or maze_name) == TST.name


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
        
        self.video_info = VideoHandler(video_path)
        self.cap = cv2.VideoCapture(str(video_path))
        self.settings = settings
        self.result_action = "cancel"
        
        # TST mode comes from the paradigm, never from ROI names: a Freestyle
        # zone called "mouse_1" is not a TST mouse.
        self.is_tst = is_tst_mode(self.settings.paradigm_id, maze_name)
        
        self.mouse_time_widgets = {} # {name: (start_spin, end_spin)}
        
        # Per-video verification has its own "what now?" prompt in the main
        # window, so only the one-off review confirms before discarding.
        self._confirm_discard = not is_verification
        self.init_ui(is_verification)
        self.update_preview(self.settings.start_time)
        self._initial_state = self._current_state()

    def init_ui(self, is_verification: bool):
        main_layout = QVBoxLayout()

        # 1. Video Preview
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(640, 360)
        self.preview_label.setStyleSheet("background-color: black; border: 1px solid grey;")
        main_layout.addWidget(self.preview_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # 2. Time Configuration (Dynamic)
        time_group = QGroupBox("Time Range Configuration")
        time_layout = QVBoxLayout()
        
        max_duration = self.video_info.duration
        
        if self.is_tst:
            # --- TST MULTI-MOUSE MODE ---
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFixedHeight(150)
            
            container = QWidget()
            form = QFormLayout(container)
            
            # Get mouse names from ROIs
            mice = sorted([k for k in self.settings.roi_manager.rois.keys() if k != 'outside'])
            
            for mouse in mice:
                row_layout = QHBoxLayout()
                
                # Defaults
                s_val = 0.0
                e_val = max_duration
                
                if self.settings.per_roi_times and mouse in self.settings.per_roi_times:
                    s_val, e_val = self.settings.per_roi_times[mouse]
                
                start_spin = QDoubleSpinBox()
                start_spin.setRange(0, max_duration)
                start_spin.setValue(s_val)
                start_spin.setSuffix("s")
                
                end_spin = QDoubleSpinBox()
                end_spin.setRange(0, max_duration)
                end_spin.setValue(e_val)
                end_spin.setSuffix("s")
                
                row_layout.addWidget(QLabel("Start:"))
                row_layout.addWidget(start_spin)
                row_layout.addWidget(QLabel("End:"))
                row_layout.addWidget(end_spin)
                
                # --- FIX: Connect EVERY spinbox to update_preview ---
                # Use a closure (lambda v=None: ...) to safely capture loop variables if needed,
                # though sender() access inside update_preview isn't needed here, just the value.
                # We pass the value directly.
                start_spin.valueChanged.connect(lambda v: self.update_preview(v))
                end_spin.valueChanged.connect(lambda v: self.update_preview(v))
                
                self.mouse_time_widgets[mouse] = (start_spin, end_spin)
                form.addRow(f"{mouse.replace('_',' ').title()}:", row_layout)
            
            scroll.setWidget(container)
            time_layout.addWidget(QLabel("Set individual start/end times for each mouse:"))
            time_layout.addWidget(scroll)
            
        else:
            # --- STANDARD SINGLE MODE ---
            self._slider_max = int(max_duration * SLIDER_STEPS_PER_SECOND)

            self.start_slider = QSlider(Qt.Orientation.Horizontal, minimum=0, maximum=self._slider_max)
            self.start_spinbox = QDoubleSpinBox(decimals=2, minimum=0, maximum=max_duration, value=self.settings.start_time)
            self.start_slider.setValue(self._seconds_to_slider(self.settings.start_time))
            start_hbox = QHBoxLayout(); start_hbox.addWidget(self.start_slider); start_hbox.addWidget(self.start_spinbox)

            end_val = self.settings.end_time or max_duration
            self.end_slider = QSlider(Qt.Orientation.Horizontal, minimum=0, maximum=self._slider_max)
            self.end_spinbox = QDoubleSpinBox(decimals=2, minimum=0, maximum=max_duration, value=end_val)
            self.end_slider.setValue(self._seconds_to_slider(end_val))
            end_hbox = QHBoxLayout(); end_hbox.addWidget(self.end_slider); end_hbox.addWidget(self.end_spinbox)

            time_layout.addLayout(start_hbox)
            time_layout.addLayout(end_hbox)
            
            self.start_slider.valueChanged.connect(self.on_slider_change)
            self.end_slider.valueChanged.connect(self.on_slider_change)
            self.start_spinbox.valueChanged.connect(self.on_spinbox_change)
            self.end_spinbox.valueChanged.connect(self.on_spinbox_change)

        time_group.setLayout(time_layout)
        main_layout.addWidget(time_group)

        # 3. Other Settings
        other_layout = QFormLayout()
        
        if not self.is_tst:
            roi_manager = self.settings.roi_manager
            ref_len = roi_manager.reference_length_pixels
            self.scale_spin = QDoubleSpinBox(decimals=4, suffix=" cm", minimum=0, maximum=1000)

            if ref_len > 0:
                if roi_manager.is_legacy_bbox_reference:
                    ref_text = (
                        f"Legacy ROI bounding-box width ({ref_len:.2f} px)"
                    )
                else:
                    ref_text = f"User-drawn reference line ({ref_len:.2f} px)"
                ref_label = QLabel(ref_text)
                prefill = self.real_world_cm
                if prefill is None:
                    prefill = real_world_cm_from_scale(
                        self.settings.scale_factor, ref_len
                    )
                if prefill:
                    self.scale_spin.setValue(prefill)
                other_layout.addRow("Reference:", ref_label)
                other_layout.addRow("Real-world length:", self.scale_spin)
                if roi_manager.is_legacy_bbox_reference:
                    other_layout.addRow(self._note_label(
                        LEGACY_REFERENCE_WARNING, warning=True
                    ))
            else:
                self.scale_spin.setEnabled(False)
                other_layout.addRow(self._note_label(NO_CALIBRATION_NOTE))
        
        # One name everywhere: the annotated video is the "validation video".
        self.timelapse_check = QCheckBox("Create validation video (annotated)")
        self.timelapse_check.setToolTip(
            "Writes an annotated video so the tracking can be checked by eye. "
            "Slower, but recommended for new setups."
        )
        self.timelapse_check.setChecked(self.settings.create_timelapse)
        other_layout.addRow(self.timelapse_check)
        
        main_layout.addLayout(other_layout)

        # 4. Buttons
        self.button_box = QDialogButtonBox()
        if is_verification:
            self.accept_btn = self.button_box.addButton("Accept", QDialogButtonBox.ButtonRole.AcceptRole)
            self.redraw_btn = self.button_box.addButton("Redraw ROIs", QDialogButtonBox.ButtonRole.ActionRole)
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
            self.skip_btn.clicked.connect(self.on_skip)

    @staticmethod
    def _note_label(text: str, warning: bool = False) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        if warning:
            label.setStyleSheet("color: #b00020; font-weight: bold;")
        return label

    def _current_state(self) -> tuple:
        if self.is_tst:
            times = tuple(
                (name, start.value(), end.value())
                for name, (start, end) in sorted(self.mouse_time_widgets.items())
            )
        else:
            times = (self.start_spinbox.value(), self.end_spinbox.value())
        return (times, self.get_real_world_cm(), self.timelapse_check.isChecked())

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
        if self.is_tst or not hasattr(self, 'scale_spin'):
            return 0.0
        return float(self.scale_spin.value())

    def _seconds_to_slider(self, seconds: float) -> int:
        return min(self._slider_max, max(0, int(round(seconds * SLIDER_STEPS_PER_SECOND))))

    def _slider_to_seconds(self, value: int) -> float:
        # The top of the slider means "to the end of the video".
        if value >= self._slider_max:
            return self.video_info.duration
        return value / SLIDER_STEPS_PER_SECOND

    def on_slider_change(self):
        """Mirror a moved slider into its own spinbox only.

        The other spinbox keeps whatever precise value it holds, so touching
        one control never rewrites (and truncates) the other.
        """
        slider = self.sender()
        is_start = slider is self.start_slider
        own_spin = self.start_spinbox if is_start else self.end_spinbox
        other_spin = self.end_spinbox if is_start else self.start_spinbox
        seconds = self._slider_to_seconds(slider.value())
        # Keep start <= end by clamping the control the user moved.
        seconds = min(seconds, other_spin.value()) if is_start else max(seconds, other_spin.value())
        self._set_quietly(own_spin, seconds)
        self._set_quietly(slider, self._seconds_to_slider(own_spin.value()))
        self.update_preview(seconds)

    def on_spinbox_change(self):
        """Mirror an edited spinbox into its own slider only."""
        spin = self.sender()
        is_start = spin is self.start_spinbox
        other_spin = self.end_spinbox if is_start else self.start_spinbox
        value = spin.value()
        if is_start and value > other_spin.value():
            value = other_spin.value()
        elif not is_start and value < other_spin.value():
            value = other_spin.value()
        self._set_quietly(spin, value)
        slider = self.start_slider if is_start else self.end_slider
        self._set_quietly(slider, self._seconds_to_slider(value))
        self.update_preview(value)

    @staticmethod
    def _set_quietly(widget, value) -> None:
        widget.blockSignals(True)
        widget.setValue(value)
        widget.blockSignals(False)

    def on_accept(self):
        if self.is_tst:
            for name, (start_spin, end_spin) in self.mouse_time_widgets.items():
                if end_spin.value() <= start_spin.value():
                    QMessageBox.warning(
                        self,
                        "Invalid Time Range",
                        f"{name.replace('_', ' ').title()} must have an end "
                        "time greater than its start time."
                    )
                    return
        elif self.start_spinbox.value() >= self.end_spinbox.value():
            QMessageBox.warning(
                self,
                "Invalid Time Range",
                "The end time must be greater than the start time."
            )
            return
        self.result_action = "accept"
        self.accept()
    def on_redraw(self): 
        self.result_action = "redraw"
        self.accept()
    def on_skip(self): 
        self.result_action = "skip"
        self.accept()

    def update_preview(self, time_in_seconds: float):
        frame_idx = int(time_in_seconds * self.video_info.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            vis_frame = self.settings.roi_manager.draw_on_frame(frame)
            pixmap = bgr_to_qpixmap(vis_frame)
            self.preview_label.setPixmap(pixmap.scaled(
                self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def get_settings(self):
        if self.is_tst:
            per_roi_times = {}
            global_start = float('inf')
            global_end = 0.0
            
            for name, (s_spin, e_spin) in self.mouse_time_widgets.items():
                s = s_spin.value()
                e = e_spin.value()
                per_roi_times[name] = (s, e)
                global_start = min(global_start, s)
                global_end = max(global_end, e)
            
            self.settings.per_roi_times = per_roi_times
            self.settings.start_time = global_start
            self.settings.end_time = global_end
            self.settings.scale_factor = 0.0
            
        else:
            end_time = self.end_spinbox.value()
            if end_time >= self.video_info.duration - 0.1: end_time = None
            
            self.settings.start_time = self.start_spinbox.value()
            self.settings.end_time = end_time
            
            self.settings.scale_factor = scale_factor_from_cm(
                self.scale_spin.value(),
                self.settings.roi_manager.reference_length_pixels,
            )

        self.settings.create_timelapse = self.timelapse_check.isChecked()
        return self.settings

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
        cap = getattr(self, "cap", None)
        if cap is not None:
            cap.release()

    def closeEvent(self, event):
        # QDialog only routes close through done() while visible.
        self._release_capture()
        super().closeEvent(event)
