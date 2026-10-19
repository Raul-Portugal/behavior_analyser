"""
gui/tuners/detection_tuner.py
Embeddable detection tuner: live quality feedback while tuning detection.

``DetectionTunerWidget`` holds everything the modal ``TunerDialog`` shows
except its OK/Cancel buttons, so the workspace can host it as a setup step.
Lifecycle handling (thread, capture, timers) follows ``gui.tuners.lifecycle``.
"""
import copy
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                             QPushButton, QScrollArea, QSizePolicy, QSlider,
                             QVBoxLayout, QWidget)

from core.analysis_geometry import AnalysisGeometry
from core.arena_zones import analysis_mask
from core.detection import DetectionEngine, DetectionQualityMonitor
from core.models import AppConfig, DetectionConfig
from core.video import VideoHandler
from gui import ux_helpers
from gui.qt_image import bgr_to_qpixmap
from gui.tuners.lifecycle import TunerLifecycle
from roi_manager import ROIManager


# (detected, confidence, source-space position) for one scanned frame.
ScanSample = Tuple[bool, float, Optional[Tuple[float, float]]]


def quick_scan_windows(total_frames: int) -> List[Tuple[int, int]]:
    """Return (start, length) windows sampling the start, middle and end."""
    window_length = min(15, total_frames)
    starts = sorted({
        0,
        max(0, total_frames // 2 - window_length // 2),
        max(0, total_frames - window_length)
    })
    return [
        (start, min(window_length, total_frames - start)) for start in starts
    ]


class QuickScanWorker(QThread):
    """Run the quick-scan detection windows off the GUI thread.

    Opens its own capture and snapshots the config, so the tuner's capture
    and sliders stay free while it runs.
    """

    scan_complete = pyqtSignal(object)
    scan_failed = pyqtSignal(str)

    def __init__(self, video_path: str, analysis_ref: np.ndarray,
                 geometry: AnalysisGeometry, config: DetectionConfig,
                 total_frames: int,
                 capture_factory: Callable = cv2.VideoCapture, parent=None,
                 arena_mask: Optional[np.ndarray] = None):
        super().__init__(parent)
        self.video_path = video_path
        self.analysis_ref = analysis_ref
        self.arena_mask = arena_mask
        self.geometry = geometry
        self.config = copy.copy(config)
        self.total_frames = total_frames
        self._capture_factory = capture_factory

    def run(self):
        cap = None
        try:
            cap = self._capture_factory(self.video_path)
            samples: List[ScanSample] = []
            # Deterministic sequential windows exercise temporal weighting
            # while sampling the beginning, middle, and end of the recording.
            for start, length in quick_scan_windows(self.total_frames):
                engine = DetectionEngine(self.analysis_ref, self.config)
                engine.set_analysis_mask(self.arena_mask)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)  # one seek per window
                for _ in range(length):
                    if self.isInterruptionRequested():
                        return
                    ret, frame = cap.read()
                    if not ret:
                        break
                    detection = engine.detect(
                        self.geometry.to_analysis(frame), use_last_pos=True
                    )
                    samples.append((
                        detection.detected,
                        detection.confidence,
                        self.geometry.to_source(detection.position),
                    ))
            self.scan_complete.emit(samples)
        except Exception as exc:
            self.scan_failed.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()


# (slider name, DetectionConfig field, minimum, maximum, scale, decimals,
# unit).  Sliders move in integer steps; ``scale`` turns a step into the real
# value shown in the label (e.g. 990 -> 99.0).
_SLIDER_SPECS = (
    ("Threshold percentile", "threshold_percentile", 950, 999, 10, 1, ""),
    ("Weight omega", "weight_omega", 0, 100, 100, 2, ""),
    ("Search window", "window_size", 20, 300, 1, 0, " px"),
    ("Min component area", "min_component_area", 5, 2000, 1, 0, " px²"),
    ("Morphology kernel", "morphology_kernel_size", 0, 9, 1, 0, " px"),
    ("Min confidence", "min_detection_confidence", 0, 100, 100, 2, ""),
)


class DetectionTunerWidget(QWidget):
    """Tune detection with live quality feedback; embeddable in a workspace.

    API for hosts: ``detection_config()``, ``config_changed(DetectionConfig)``,
    ``has_unsaved_changes()`` and ``shutdown()`` (call it when the step is
    discarded; hiding the widget only pauses it).
    """

    config_changed = pyqtSignal(object)

    def __init__(self, video_path: str, ref_frame: np.ndarray,
                 roi_manager: ROIManager,
                 initial_config: Optional[DetectionConfig] = None,
                 parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.video_path = video_path
        self.ref_frame = ref_frame
        self.roi_manager = roi_manager
        self.config = (
            DetectionConfig() if initial_config is None
            else copy.copy(initial_config)
        )
        self.video_info = VideoHandler(video_path)
        # Run detection at exactly the resolution the production Analyzer
        # uses, so pixel-valued parameters mean the same thing in both.
        self.geometry = AnalysisGeometry(
            self.video_info.width,
            self.video_info.height,
            AppConfig().video.analysis_width,
        )
        self.analysis_ref = self.geometry.reference_to_analysis(ref_frame)
        # Tune against the same arena restriction the analysis will use.
        arena = getattr(roi_manager, 'arena', None)
        self.arena_mask = (
            None if arena is None
            else analysis_mask(arena.points, self.geometry)
        )
        self._lifecycle = TunerLifecycle("detection tuner")
        self._lifecycle.attach(self)
        self._capture = self._lifecycle.open_capture(video_path)
        self.current_frame_idx = self.video_info.total_frames // 2

        # Quality monitoring
        self.quality_monitor = DetectionQualityMonitor(
            window_size=50,
            frame_diagonal=float(
                np.hypot(self.video_info.width, self.video_info.height)
            )
        )

        self.scan_worker: Optional[QuickScanWorker] = None

        # Auto-update stats
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self.update_quality_stats)

        self.init_ui()
        self._initial_slider_state = self._slider_state()
        # Force the first render: the slider's valueChanged is skipped by
        # the equality guard if the index were already set.
        initial_frame = self.current_frame_idx
        self.current_frame_idx = -1
        self.frame_slider.setValue(initial_frame)
        if self.current_frame_idx != initial_frame:
            self.on_frame_slider_change(initial_frame)
        self.stats_timer.start(500)

    @property
    def cap(self):
        """The preview capture (reopens lazily unless shut down)."""
        return self._capture.get()

    # ------------------------------------------------------------------ UI
    def init_ui(self):
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)

        # Image Previews
        self.main_view = self._create_image_label("Main View")
        self.diff_view = self._create_image_label("Difference View")
        self.thresh_view = self._create_image_label("Threshold View")
        left_layout.addWidget(self.main_view, 2)
        left_layout.addWidget(self.diff_view, 1)
        left_layout.addWidget(self.thresh_view, 1)

        # Quality Stats Panel
        stats_group = QGroupBox("Detection Quality (Live)")
        stats_layout = QVBoxLayout()

        self.stats_label = QLabel("Analyzing...")
        self.stats_label.setStyleSheet("background: #f0f0f0; padding: 15px; font-size: 11pt; border-radius: 5px;")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)

        self.recommendation_label = QLabel("")
        self.recommendation_label.setStyleSheet("background: #fff3cd; padding: 10px; font-size: 10pt; border-radius: 5px; color: #856404;")
        self.recommendation_label.setWordWrap(True)
        self.recommendation_label.setVisible(False)
        stats_layout.addWidget(self.recommendation_label)

        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)

        # Parameter Controls
        param_grid = QGridLayout()
        self.sliders = {}
        self.slider_labels = {}
        for name, field, low, high, scale, decimals, unit in _SLIDER_SPECS:
            self._add_slider(
                param_grid, name, low, high,
                self._slider_position(field), scale=scale,
                decimals=decimals, unit=unit,
            )
        right_layout.addLayout(param_grid)

        # Frame Navigation
        self.frame_label = QLabel()
        self.frame_slider = QSlider(Qt.Orientation.Horizontal, minimum=0, maximum=self.video_info.total_frames - 1)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_change)
        right_layout.addWidget(self.frame_label)
        right_layout.addWidget(self.frame_slider)

        help_label = QLabel("Keyboard: N/P (frame), +/- (jump 100 frames)\nTip: Scan through video to test different frames")
        help_label.setStyleSheet("color: #666; font-size: 9pt;")
        right_layout.addWidget(help_label)

        right_layout.addStretch()

        # Action buttons (the host adds its own OK/Cancel)
        button_layout = QHBoxLayout()
        self.test_scan_button = QPushButton("Quick Scan Test")
        self.test_scan_button.clicked.connect(self.run_quick_scan)
        button_layout.addWidget(self.test_scan_button)
        button_layout.addStretch()

        right_panel = QWidget()
        right_panel.setLayout(right_layout)
        self.control_scroll = QScrollArea()
        self.control_scroll.setWidgetResizable(True)
        self.control_scroll.setWidget(right_panel)
        main_layout.addLayout(left_layout, 3)
        main_layout.addWidget(self.control_scroll, 1)

        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addLayout(main_layout, 1)
        root_layout.addLayout(button_layout)
        self.setLayout(root_layout)

    def _create_image_label(self, title: str) -> QLabel:
        label = QLabel(title)
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("background-color: black; color: white; border: 1px solid grey;")
        return label

    @staticmethod
    def format_slider_value(name: str, value: int, scale: int = 1,
                            decimals: int = 0, unit: str = "") -> str:
        """Label text with the real value and its unit, e.g. "Threshold percentile: 99.0"."""
        return f"{name}: {value / scale:.{decimals}f}{unit}"

    def _slider_position(self, field: str) -> int:
        """Slider step for a config field, clamped to the slider's range."""
        for _name, spec_field, low, high, scale, _dec, _unit in _SLIDER_SPECS:
            if spec_field == field:
                step = round(getattr(self.config, field) * scale)
                return max(low, min(high, step))
        raise KeyError(field)

    def _add_slider(self, layout: QGridLayout, name: str, min_val: int, max_val: int,
                    initial_val: int, scale: int = 1, decimals: int = 0, unit: str = ""):
        row = layout.rowCount()
        label = QLabel(self.format_slider_value(name, initial_val, scale, decimals, unit))
        slider = QSlider(Qt.Orientation.Horizontal, minimum=min_val, maximum=max_val, value=initial_val)
        slider.valueChanged.connect(
            lambda val, l=label: l.setText(
                self.format_slider_value(name, val, scale, decimals, unit)
            )
        )
        slider.valueChanged.connect(self.on_parameter_change)
        layout.addWidget(label, row, 0)
        layout.addWidget(slider, row, 1)
        self.sliders[name] = slider
        self.slider_labels[name] = label

    # ------------------------------------------------------------ behaviour
    def on_frame_slider_change(self, value: int):
        if self.current_frame_idx != value:
            self.current_frame_idx = value
            self.update_detection()

    def on_parameter_change(self, _value: int = 0):
        # A new parameter makes earlier quality samples incomparable.
        # (Avoid QObject.sender(): when update_detection is called directly
        # it can point at a deleted object and crash on comparison.)
        self.quality_monitor.reset()
        self.update_detection()
        self.config_changed.emit(copy.copy(self.config))

    def _sync_config_from_sliders(self):
        """Copy slider positions into the config.

        A field is only overwritten when its slider no longer sits at the
        step that represents the current value, so a value the sliders cannot
        represent exactly (finer than a step or outside the range) survives
        until the user actually moves that slider.
        """
        for name, field, _low, _high, scale, _dec, _unit in _SLIDER_SPECS:
            step = self.sliders[name].value()
            if step == self._slider_position(field):
                continue
            setattr(self.config, field, step / scale if scale != 1 else step)

    def update_detection(self):
        self._sync_config_from_sliders()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if not ret: return

        # Run the same segmentation and component scoring as production.
        engine = DetectionEngine(self.analysis_ref, self.config)
        engine.set_analysis_mask(self.arena_mask)
        detection = engine.detect(
            self.geometry.to_analysis(frame), use_last_pos=False
        )
        # Detector output is in analysis space; overlays and the quality
        # monitor use source-frame coordinates, as in production.
        position = self.geometry.to_source(detection.position)
        debug_images = engine.get_intermediate_images()
        detected = detection.detected
        self.quality_monitor.update(
            detected, detection.confidence, position
        )

        # Drawing
        display_frame = frame.copy()
        if position:
            center = (int(round(position[0])), int(round(position[1])))
            cv2.circle(display_frame, center, 15, (0, 255, 0), 3)
            cv2.circle(display_frame, center, 3, (0, 0, 255), -1)
            status_text = f"Confidence: {detection.confidence:.2f}"
            status_color = (0, 255, 0)
        else:
            status_text = f"Not detected: {detection.failure_reason}"
            status_color = (0, 0, 255)
        cv2.putText(
            display_frame,
            status_text,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
            cv2.LINE_AA
        )

        for category, rois in self.roi_manager.rois.items():
            for roi in rois:
                pts = np.array(roi.points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], True, (255, 255, 0), 2)
        arena = getattr(self.roi_manager, 'arena', None)
        if arena is not None:
            cv2.polylines(
                display_frame,
                [np.array(arena.points, np.int32).reshape((-1, 1, 2))],
                True, (0, 165, 255), 3,
            )

        self._update_image(self.main_view, display_frame)

        diff_colored = cv2.cvtColor(
            debug_images['diff'], cv2.COLOR_GRAY2BGR
        )
        self._update_image(self.diff_view, diff_colored)

        threshold_colored = cv2.cvtColor(
            debug_images['threshold'], cv2.COLOR_GRAY2BGR
        )
        selected_component = debug_images.get('selected_component')
        if selected_component is not None:
            threshold_colored[selected_component > 0] = (0, 255, 0)
        self._update_image(self.thresh_view, threshold_colored)

        self.frame_label.setText(f"Frame: {self.current_frame_idx} / {self.video_info.total_frames - 1}")

    def update_quality_stats(self):
        if self.quality_monitor.frames_processed < 5: return
        stats = self.quality_monitor.get_statistics()
        status, color, recommendation = self.quality_monitor.get_quality_status()

        stats_text = (f"<b>Current Detection Quality:</b><br><br>"
                      f"Detection Rate: <b>{stats['detection_rate']:.1f}%</b><br>"
                      f"Confidence: <b>{stats['confidence']:.1f}%</b><br>"
                      f"Stability: <b>{stats['tracking_stability']:.1f}%</b><br><br>"
                      f"Implausible jumps: <b>{stats['implausible_jump_rate']:.1f}%</b><br><br>"
                      f"Status: <b>{status}</b>")

        self.stats_label.setText(stats_text)

        colors = {'green': '#d4edda', 'orange': '#fff3cd', 'red': '#f8d7da'}
        borders = {'green': '#28a745', 'orange': '#ffc107', 'red': '#dc3545'}
        self.stats_label.setStyleSheet(f"background: {colors[color]}; padding: 15px; font-size: 11pt; border: 2px solid {borders[color]}; border-radius: 5px;")

        if color != 'green':
            self.recommendation_label.setText(f"💡 {recommendation}")
            self.recommendation_label.setVisible(True)
        else:
            self.recommendation_label.setVisible(False)

    def run_quick_scan(self):
        if self.scan_worker is not None and self.scan_worker.isRunning():
            return
        if self._lifecycle.is_shut_down:
            return
        self._sync_config_from_sliders()
        self.test_scan_button.setEnabled(False)
        self.test_scan_button.setText("Scanning...")
        self.quality_monitor.reset()

        # Parentless on purpose: the lifecycle owns and joins the thread (a
        # child thread would be destroyed, still running, with this widget).
        self.scan_worker = QuickScanWorker(
            self.video_path,
            self.analysis_ref,
            self.geometry,
            self.config,
            self.video_info.total_frames,
            arena_mask=self.arena_mask,
        )
        self.scan_worker.scan_complete.connect(self._on_quick_scan_complete)
        self.scan_worker.scan_failed.connect(self._on_quick_scan_failed)
        self.scan_worker.finished.connect(self._on_quick_scan_finished)
        self._lifecycle.start_thread(self.scan_worker)

    def _on_quick_scan_complete(self, samples):
        if self._lifecycle.is_shut_down:
            return
        self.quality_monitor.reset()
        for detected, confidence, position in samples:
            self.quality_monitor.update(detected, confidence, position)
        self.update_quality_stats()

    def _on_quick_scan_failed(self, message: str):
        self.recommendation_label.setText(f"Quick scan failed: {message}")
        self.recommendation_label.setVisible(True)

    def _on_quick_scan_finished(self):
        self.test_scan_button.setEnabled(True)
        self.test_scan_button.setText("Quick Scan Test")

    def _update_image(self, label: QLabel, image: np.ndarray):
        pixmap = bgr_to_qpixmap(image)
        label.setPixmap(pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def handle_navigation_key(self, event) -> bool:
        """Frame navigation keys; True when ``event`` was consumed."""
        key = event.key()
        last = self.video_info.total_frames - 1
        if key == Qt.Key.Key_N:
            self.frame_slider.setValue(min(self.current_frame_idx + 1, last))
        elif key == Qt.Key.Key_P:
            self.frame_slider.setValue(max(self.current_frame_idx - 1, 0))
        elif key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
            self.frame_slider.setValue(min(self.current_frame_idx + 100, last))
        elif key == Qt.Key.Key_Minus:
            self.frame_slider.setValue(max(self.current_frame_idx - 100, 0))
        else:
            return False
        return True

    def keyPressEvent(self, event):
        if not self.handle_navigation_key(event):
            super().keyPressEvent(event)

    # ----------------------------------------------------------- public API
    def detection_config(self) -> DetectionConfig:
        """The tuned configuration (a copy; later tuning does not alter it)."""
        self._sync_config_from_sliders()
        return copy.copy(self.config)

    def _slider_state(self):
        return tuple(slider.value() for slider in self.sliders.values())

    def has_unsaved_changes(self) -> bool:
        """True once a detection parameter was moved from its initial value."""
        return ux_helpers.should_confirm_discard(
            self._initial_slider_state, self._slider_state()
        )

    # ------------------------------------------------------------ lifecycle
    def shutdown(self):
        """Stop background work and free the capture (idempotent, final)."""
        self.stats_timer.stop()
        self._lifecycle.shutdown()

    def _pause(self):
        self.stats_timer.stop()
        self._lifecycle.pause()

    def hideEvent(self, event):
        super().hideEvent(event)
        # Minimising the window is a spontaneous hide: keep working then.
        if not event.spontaneous():
            self._pause()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._lifecycle.is_shut_down and not self.stats_timer.isActive():
            self.stats_timer.start(500)

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)
