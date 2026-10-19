"""Time-window panel: one start/end for the video, or one per mouse (TST)."""
from typing import Dict, List, Optional, Tuple

import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDoubleSpinBox, QFormLayout, QGroupBox,
                             QHBoxLayout, QLabel, QScrollArea, QSlider,
                             QVBoxLayout, QWidget)

from core.setup_state import GLOBAL_WINDOW_ISSUE, per_roi_window_issue
from core.video import VideoHandler
from gui.qt_image import bgr_to_qpixmap
from gui.setup_panels.base import SetupPanel
from mazes.base_maze import STEP_TIME_WINDOW

SLIDER_STEPS_PER_SECOND = 10  # sliders move in 0.1 s units

# Longest window offered when there is no video to measure.
FALLBACK_DURATION_S = 24 * 3600.0



class TimeWindowPanel(SetupPanel):
    """Start/end sliders with a frame preview, or per-mouse windows.

    Which of the two is shown follows the paradigm
    (``Maze.has_per_roi_times``). Sliders move in 0.1 s steps and the very
    top of the end slider means "to the end of the video" (stored as
    ``None``). The preview shows the frame at the time last touched with the
    ROIs drawn on it; the video is opened lazily and released whenever the
    panel is hidden, so it never keeps the file locked.
    """

    step_id = STEP_TIME_WINDOW

    def __init__(self, parent=None, preview_size: Tuple[int, int] = (640, 360),
                 preserve_exact: bool = True):
        super().__init__(parent, preserve_exact)
        self.video_info: Optional[VideoHandler] = None
        # Why the video could not be opened (None: fine or no video).
        self.video_error: Optional[str] = None
        self._video_path: Optional[str] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._roi_manager = None
        self.is_per_roi = False
        self._duration = FALLBACK_DURATION_S
        self._slider_max = int(self._duration * SLIDER_STEPS_PER_SECOND)
        self.mouse_time_widgets: Dict[str, tuple] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Video preview
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(*preview_size)
        self.preview_label.setStyleSheet(
            "background-color: black; border: 1px solid grey;"
        )
        layout.addWidget(
            self.preview_label, alignment=Qt.AlignmentFlag.AlignCenter
        )

        self.time_group = QGroupBox("Time Range Configuration")
        time_layout = QVBoxLayout(self.time_group)

        # One window for the whole video
        self.global_widget = QWidget()
        global_layout = QVBoxLayout(self.global_widget)
        global_layout.setContentsMargins(0, 0, 0, 0)
        self.start_slider = QSlider(Qt.Orientation.Horizontal)
        self.start_spinbox = QDoubleSpinBox(decimals=2, minimum=0)
        self.end_slider = QSlider(Qt.Orientation.Horizontal)
        self.end_spinbox = QDoubleSpinBox(decimals=2, minimum=0)
        for slider, spin, caption in (
            (self.start_slider, self.start_spinbox, "Start:"),
            (self.end_slider, self.end_spinbox, "End:"),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(caption))
            row.addWidget(slider)
            row.addWidget(spin)
            global_layout.addLayout(row)
        self.start_slider.valueChanged.connect(
            lambda _value: self.on_slider_change(self.start_slider)
        )
        self.end_slider.valueChanged.connect(
            lambda _value: self.on_slider_change(self.end_slider)
        )
        self.start_spinbox.valueChanged.connect(
            lambda _value: self.on_spinbox_change(self.start_spinbox)
        )
        self.end_spinbox.valueChanged.connect(
            lambda _value: self.on_spinbox_change(self.end_spinbox)
        )
        time_layout.addWidget(self.global_widget)

        # One window per mouse
        self.roi_widget = QWidget()
        roi_layout = QVBoxLayout(self.roi_widget)
        roi_layout.setContentsMargins(0, 0, 0, 0)
        roi_layout.addWidget(
            QLabel("Set individual start/end times for each mouse:")
        )
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(150)
        self._roi_container = QWidget()
        self._roi_form = QFormLayout(self._roi_container)
        scroll.setWidget(self._roi_container)
        roi_layout.addWidget(scroll)
        time_layout.addWidget(self.roi_widget)

        layout.addWidget(self.time_group)
        self._show_mode(per_roi=False)
        self._configure_range(self._duration)

    # ------------------------------------------------------------------
    # Video
    # ------------------------------------------------------------------
    @property
    def cap(self) -> Optional[cv2.VideoCapture]:
        """The open capture (opened on first use), None without a video."""
        if self._cap is None and self._video_path is not None:
            self._cap = cv2.VideoCapture(self._video_path)
        return self._cap

    def release_capture(self) -> None:
        """Free the video file (it is reopened if a preview is needed)."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def hideEvent(self, event):
        self.release_capture()
        super().hideEvent(event)

    def closeEvent(self, event):
        self.release_capture()
        super().closeEvent(event)

    def _open_video(self, video_path) -> None:
        path = str(video_path) if video_path else None
        if path != self._video_path:
            self.release_capture()
            self._video_path = path
            self.video_info = None
            self.video_error = None
            if path is not None:
                try:
                    self.video_info = VideoHandler(path)
                except Exception as exc:
                    self.video_error = f"Cannot read the video: {exc}"

    @property
    def duration(self) -> float:
        return self._duration

    def _global_controls(self):
        return (self.start_slider, self.end_slider,
                self.start_spinbox, self.end_spinbox)

    def _configure_range(self, duration: float) -> None:
        self._duration = float(duration)
        self._slider_max = int(self._duration * SLIDER_STEPS_PER_SECOND)
        controls = self._global_controls()
        for control in controls:
            control.blockSignals(True)
        try:
            for slider in (self.start_slider, self.end_slider):
                slider.setRange(0, self._slider_max)
            for spin in (self.start_spinbox, self.end_spinbox):
                spin.setMaximum(self._duration)
        finally:
            for control in controls:
                control.blockSignals(False)

    def _show_mode(self, per_roi: bool) -> None:
        self.is_per_roi = per_roi
        self.global_widget.setVisible(not per_roi)
        self.roi_widget.setVisible(per_roi)

    # ------------------------------------------------------------------
    # Slider / spin box behaviour
    # ------------------------------------------------------------------
    def _seconds_to_slider(self, seconds: float) -> int:
        return min(
            self._slider_max,
            max(0, int(round(seconds * SLIDER_STEPS_PER_SECOND)))
        )

    def _slider_to_seconds(self, value: int) -> float:
        # The top of the slider means "to the end of the video".
        if value >= self._slider_max:
            return self._duration
        return value / SLIDER_STEPS_PER_SECOND

    def on_slider_change(self, slider):
        """Mirror a moved slider into its own spinbox only.

        The other spinbox keeps whatever precise value it holds, so touching
        one control never rewrites (and truncates) the other.
        """
        is_start = slider is self.start_slider
        own_spin = self.start_spinbox if is_start else self.end_spinbox
        other_spin = self.end_spinbox if is_start else self.start_spinbox
        seconds = self._slider_to_seconds(slider.value())
        # Keep start <= end by clamping the control the user moved.
        seconds = (
            min(seconds, other_spin.value()) if is_start
            else max(seconds, other_spin.value())
        )
        self._set_quietly(own_spin, seconds)
        self._set_quietly(slider, self._seconds_to_slider(own_spin.value()))
        self.update_preview(seconds)
        self._emit_changed()

    def on_spinbox_change(self, spin):
        """Mirror an edited spinbox into its own slider only."""
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
        self._emit_changed()

    @staticmethod
    def _set_quietly(widget, value) -> None:
        widget.blockSignals(True)
        widget.setValue(value)
        widget.blockSignals(False)

    def _on_roi_time_changed(self, value) -> None:
        self.update_preview(value)
        self._emit_changed()

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------
    def update_preview(self, time_in_seconds: float) -> None:
        if self._loading or self.video_info is None or self.cap is None:
            return
        frame_idx = int(time_in_seconds * self.video_info.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        if self._roi_manager is not None:
            frame = self._roi_manager.draw_on_frame(frame)
        pixmap = bgr_to_qpixmap(frame)
        self.preview_label.setPixmap(pixmap.scaled(
            self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    # ------------------------------------------------------------------
    # Contract
    # ------------------------------------------------------------------
    def load(self, state) -> None:
        maze = state.build_maze()
        self._roi_manager = state.roi_manager
        self._open_video(state.video_path)
        duration = (
            self.video_info.duration if self.video_info is not None
            else FALLBACK_DURATION_S
        )
        self._loading = True
        try:
            self._configure_range(duration)
            self._show_mode(maze.has_per_roi_times)
            if maze.has_per_roi_times:
                self._load_per_roi(state, maze)
            else:
                self._load_global(state)
        finally:
            self._loading = False
        self.update_preview_from_state(state)

    def _load_global(self, state) -> None:
        start = state.start_time
        end = state.end_time or self._duration
        controls = self._global_controls()
        for control in controls:
            control.blockSignals(True)
        try:
            self._exact.set(self.start_spinbox, start)
            self._exact.set(self.end_spinbox, end)
            self.start_slider.setValue(self._seconds_to_slider(start))
            self.end_slider.setValue(self._seconds_to_slider(end))
        finally:
            for control in controls:
                control.blockSignals(False)

    def _load_per_roi(self, state, maze) -> None:
        while self._roi_form.rowCount():
            self._roi_form.removeRow(0)
        self.mouse_time_widgets.clear()
        mice = (
            maze.time_window_subjects(state.roi_manager)
            if state.roi_manager is not None else []
        )
        for mouse in mice:
            start, end = state.per_roi_times.get(mouse, (0.0, self._duration))
            start_spin = QDoubleSpinBox()
            start_spin.setRange(0, self._duration)
            self._exact.set(start_spin, start)
            start_spin.setSuffix("s")
            end_spin = QDoubleSpinBox()
            end_spin.setRange(0, self._duration)
            self._exact.set(end_spin, end)
            end_spin.setSuffix("s")
            row_layout = QHBoxLayout()
            row_layout.addWidget(QLabel("Start:"))
            row_layout.addWidget(start_spin)
            row_layout.addWidget(QLabel("End:"))
            row_layout.addWidget(end_spin)
            start_spin.valueChanged.connect(self._on_roi_time_changed)
            end_spin.valueChanged.connect(self._on_roi_time_changed)
            self.mouse_time_widgets[mouse] = (start_spin, end_spin)
            self._roi_form.addRow(f"{mouse.replace('_', ' ').title()}:",
                                  row_layout)

    def update_preview_from_state(self, state) -> None:
        self.update_preview(state.start_time)

    # -- values -------------------------------------------------------------
    def start_time(self) -> float:
        return float(self._exact.get(self.start_spinbox))

    def end_time(self) -> Optional[float]:
        """The end of the window; None means "to the end of the video"."""
        end_time = float(self._exact.get(self.end_spinbox))
        if end_time >= self._duration - 0.1 and self.video_info is not None:
            return None
        return end_time

    def per_roi_times(self) -> Dict[str, Tuple[float, float]]:
        return {
            name: (float(self._exact.get(start)), float(self._exact.get(end)))
            for name, (start, end) in self.mouse_time_widgets.items()
        }

    def current_values(self) -> tuple:
        """Everything the user can edit, for unsaved-work detection."""
        if self.is_per_roi:
            return tuple(
                (name, start.value(), end.value())
                for name, (start, end) in sorted(self.mouse_time_widgets.items())
            )
        return (self.start_spinbox.value(), self.end_spinbox.value())

    def issues(self) -> List[str]:
        if self.video_error:
            return [self.video_error]
        if self.is_per_roi:
            return [
                per_roi_window_issue(name)
                for name, (start, end) in self.mouse_time_widgets.items()
                if end.value() <= start.value()
            ]
        if self.start_spinbox.value() >= self.end_spinbox.value():
            return [GLOBAL_WINDOW_ISSUE]
        return []

    def write(self, state) -> None:
        """Write the windows unconditionally (``apply`` checks issues first)."""
        if self.is_per_roi:
            times = self.per_roi_times()
            if times:
                state.set_per_roi_times(times)
        else:
            state.start_time = self.start_time()
            state.end_time = self.end_time()

    def apply(self, state) -> bool:
        if self.issues():
            return False
        self.write(state)
        return self._finish_apply(state)
