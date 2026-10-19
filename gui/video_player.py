"""A reusable, frame-accurate video player widget.

``VideoPlayerWidget`` decodes frames with OpenCV on a background ``QThread``
(QtMultimedia is deliberately not used: it may not be packaged and its seeks
are not frame accurate). While playing, frames are read sequentially; when the
user scrubs or steps, the decoder repositions with
:func:`core.video.position_capture_at_frame`, which verifies the landing frame
and falls back to decoding forward when a codec seeks inexactly.

The widget knows nothing about scoring: it shows frames, reports which frame
is on screen through ``frame_changed(frame_index, timestamp_s)`` and offers an
optional overlay callback so a caller can draw ROIs. The manual-scoring dialog
uses it today and the v4 workspace is expected to embed it.

Keyboard (when the widget, or a dialog forwarding to
:meth:`VideoPlayerWidget.handle_key_event`, has focus): Space toggles play,
Left/Right step one frame, Shift+Left/Right jump ten frames, Home/End go to the
first/last frame.

A host dialog must call :meth:`VideoPlayerWidget.close_video` from its own
``done()`` / ``closeEvent`` so the capture is released: hiding or closing a
parent does not close its child widgets.
"""
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QCoreApplication, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QPixmap
from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton,
                             QSizePolicy, QSlider, QVBoxLayout, QWidget)

from core.video import VideoHandler, position_capture_at_frame
from gui.qt_image import bgr_to_qpixmap

logger = logging.getLogger(__name__)

PLAYBACK_SPEEDS = (0.25, 0.5, 1.0, 2.0)
DEFAULT_SPEED = 1.0
JUMP_FRAMES = 10
# A forward gap up to this many frames is reached by decoding (grab) instead
# of a seek, which is cheaper and always frame exact.
MAX_FORWARD_GAP = 30
# Playback tick: frames are chosen from the wall clock, so a slow decoder
# skips frames instead of slowing the video down.
PLAY_TICK_MS = 10

OverlayCallback = Callable[[np.ndarray, int], np.ndarray]


def format_timestamp(seconds: float) -> str:
    """``mm:ss.fff`` (minutes may exceed 59)."""
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000.0))
    minutes, remainder = divmod(total_ms, 60_000)
    return f"{minutes:02d}:{remainder // 1000:02d}.{remainder % 1000:03d}"


class _DecoderThread(QThread):
    """Owns the capture and answers "show frame N" requests, latest wins."""

    frame_ready = pyqtSignal(int, int, object)   # token, frame index, BGR array
    read_failed = pyqtSignal(int, int)           # token, requested index

    def __init__(self, path: str, total_frames: int, parent=None):
        super().__init__(parent)
        self._path = path
        self._total = int(total_frames)
        self._cond = threading.Condition()
        self._request = None            # (token, index) or None
        self._stop = False
        self._position: Optional[int] = None   # index the next read returns
        self.capture_open = False

    def request(self, token: int, index: int) -> None:
        with self._cond:
            self._request = (token, int(index))
            self._cond.notify()

    def request_stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()

    def _goto(self, cap, index: int) -> None:
        if self._position is not None:
            gap = index - self._position
            if gap == 0:
                return
            if 0 < gap <= MAX_FORWARD_GAP:
                for _ in range(gap):
                    if not cap.grab():
                        break
                else:
                    self._position = index
                    return
        position_capture_at_frame(cap, index)
        self._position = index

    def run(self) -> None:
        cap = cv2.VideoCapture(self._path)
        self.capture_open = bool(cap.isOpened())
        try:
            while True:
                with self._cond:
                    while self._request is None and not self._stop:
                        self._cond.wait()
                    if self._stop:
                        break
                    token, index = self._request
                    self._request = None
                if not self.capture_open:
                    self.read_failed.emit(token, index)
                    continue
                try:
                    if self._total > 0:
                        index = min(max(0, index), self._total - 1)
                    self._goto(cap, index)
                    ok, frame = cap.read()
                except Exception:
                    logger.exception("Video decoding failed at frame %d", index)
                    ok, frame = False, None
                if ok and frame is not None:
                    self._position = index + 1
                    self.frame_ready.emit(token, index, frame)
                else:
                    self._position = None
                    self.read_failed.emit(token, index)
        finally:
            cap.release()
            self.capture_open = False


class _FrameView(QLabel):
    """Shows a pixmap scaled to fit, keeping the aspect ratio."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(QSize(160, 120))
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #111; color: #ddd;")

    def set_frame(self, pixmap: Optional[QPixmap]) -> None:
        self._pixmap = pixmap
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            self.setPixmap(QPixmap())
            return
        self.setPixmap(self._pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def sizeHint(self) -> QSize:
        return QSize(640, 480)


class VideoPlayerWidget(QWidget):
    """Play, scrub and frame-step a video; see the module docstring."""

    frame_changed = pyqtSignal(int, float)   # frame index, timestamp in seconds
    video_loaded = pyqtSignal(int, float)    # total frames, fps
    load_failed = pyqtSignal(str)
    playback_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: Optional[_DecoderThread] = None
        self._path: Optional[Path] = None
        self._total = 0
        self._fps = 0.0
        self._frame = -1                      # index of the frame on screen
        self._last_frame: Optional[np.ndarray] = None
        self._overlay: Optional[OverlayCallback] = None
        self._token = 0                       # last request sent
        self._answered = 0                    # last request answered
        self._playing = False
        self._speed = DEFAULT_SPEED
        self._play_t0 = 0.0
        self._play_f0 = 0
        self._build_ui()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._timer = QTimer(self)
        self._timer.setInterval(PLAY_TICK_MS)
        self._timer.timeout.connect(self._on_tick)
        self._set_controls_enabled(False)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.view = _FrameView(self)
        self.view.setText("No video loaded")

        def button(text, tip, slot):
            widget = QPushButton(text, self)
            widget.setToolTip(tip)
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            widget.setAutoDefault(False)
            widget.clicked.connect(slot)
            return widget

        self.jump_back_button = button(
            f"-{JUMP_FRAMES}", f"Back {JUMP_FRAMES} frames (Shift+Left)",
            lambda: self.step(-JUMP_FRAMES))
        self.step_back_button = button(
            "<", "Previous frame (Left)", lambda: self.step(-1))
        self.play_button = button("Play", "Play / pause (Space)", self.toggle_play)
        self.step_forward_button = button(
            ">", "Next frame (Right)", lambda: self.step(1))
        self.jump_forward_button = button(
            f"+{JUMP_FRAMES}", f"Forward {JUMP_FRAMES} frames (Shift+Right)",
            lambda: self.step(JUMP_FRAMES))

        self.speed_combo = QComboBox(self)
        for speed in PLAYBACK_SPEEDS:
            self.speed_combo.addItem(f"{speed:g}x", speed)
        self.speed_combo.setCurrentIndex(PLAYBACK_SPEEDS.index(DEFAULT_SPEED))
        self.speed_combo.setToolTip("Playback speed")
        self.speed_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(0, 0)
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slider.setToolTip("Drag to scrub")
        self.slider.valueChanged.connect(self._on_slider_value)
        self.slider.sliderPressed.connect(self._on_slider_pressed)

        self.time_label = QLabel("00:00.000   frame - / -", self)
        self.time_label.setMinimumWidth(190)
        self.time_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.time_label.setStyleSheet("font-family: Consolas, monospace;")

        scrub = QHBoxLayout()
        scrub.addWidget(self.slider, 1)
        scrub.addWidget(self.time_label)

        controls = QHBoxLayout()
        for widget in (self.jump_back_button, self.step_back_button,
                       self.play_button, self.step_forward_button,
                       self.jump_forward_button, self.speed_combo):
            controls.addWidget(widget)
        controls.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view, 1)
        layout.addLayout(scrub)
        layout.addLayout(controls)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (self.jump_back_button, self.step_back_button,
                       self.play_button, self.step_forward_button,
                       self.jump_forward_button, self.speed_combo, self.slider):
            widget.setEnabled(enabled)

    # ------------------------------------------------------------ properties
    @property
    def video_path(self) -> Optional[Path]:
        return self._path

    @property
    def total_frames(self) -> int:
        return self._total

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def current_frame(self) -> int:
        """Index of the frame on screen (-1 before the first frame arrives)."""
        return self._frame

    @property
    def current_timestamp(self) -> float:
        return self._frame / self._fps if self._fps > 0 and self._frame >= 0 else 0.0

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def capture_open(self) -> bool:
        """True while the decoder thread holds an open capture."""
        return self._thread is not None and self._thread.capture_open

    @property
    def last_frame(self) -> Optional[np.ndarray]:
        """The raw (overlay-free) BGR frame on screen."""
        return self._last_frame

    def is_idle(self) -> bool:
        """True when every frame request has been answered."""
        return self._answered >= self._token

    # ---------------------------------------------------------------- loading
    def open_video(self, path) -> bool:
        """Open ``path`` and show its first frame. False if it cannot be read."""
        self.close_video()
        path = Path(path)
        try:
            handler = VideoHandler(path)
            total, fps = handler.total_frames, float(handler.fps)
        except (IOError, OSError) as error:
            self._fail(str(error))
            return False
        if total <= 0 or fps <= 0:
            self._fail(f"Cannot read frames or frame rate from {path.name}")
            return False

        self._path, self._total, self._fps = path, total, fps
        self._frame, self._last_frame = -1, None
        self._token = self._answered = 0
        thread = _DecoderThread(str(path), total)
        thread.frame_ready.connect(self._on_frame_ready)
        thread.read_failed.connect(self._on_read_failed)
        self._thread = thread
        # No reference to self: this must still stop the thread if the widget
        # is deleted without close_video() being called.
        self.destroyed.connect(thread.request_stop)
        thread.start()
        while not thread.isRunning():
            QThread.msleep(1)

        self.slider.blockSignals(True)
        self.slider.setRange(0, total - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._set_controls_enabled(True)
        self.view.setText("")
        self.video_loaded.emit(total, fps)
        self.seek(0)
        return True

    def _fail(self, message: str) -> None:
        logger.warning("Video player could not open the video: %s", message)
        self.view.set_frame(None)
        self.view.setText(message)
        self._set_controls_enabled(False)
        self.load_failed.emit(message)

    def close_video(self) -> None:
        """Stop playback and release the capture (waits for the decoder)."""
        self.pause()
        thread, self._thread = self._thread, None
        if thread is not None:
            try:
                self.destroyed.disconnect(thread.request_stop)
            except (TypeError, RuntimeError):
                pass
            thread.request_stop()
            if not thread.wait(5000):
                logger.warning("Video decoder thread did not stop in time")
            try:
                thread.frame_ready.disconnect(self._on_frame_ready)
                thread.read_failed.disconnect(self._on_read_failed)
            except (TypeError, RuntimeError):
                pass
        self._set_controls_enabled(False)

    def closeEvent(self, event):
        self.close_video()
        super().closeEvent(event)

    # --------------------------------------------------------------- overlay
    def set_overlay(self, callback: Optional[OverlayCallback]) -> None:
        """Draw on every shown frame: ``callback(frame_bgr, frame_index)``.

        The callback receives a copy it may modify and must return a BGR
        frame. It runs on the GUI thread, so keep it cheap.
        """
        self._overlay = callback
        self.refresh()

    def refresh(self) -> None:
        """Re-render the current frame (after the overlay's inputs changed)."""
        if self._last_frame is not None:
            self._render(self._last_frame, self._frame)

    def _render(self, frame: np.ndarray, index: int) -> None:
        shown = frame
        if self._overlay is not None:
            try:
                shown = self._overlay(frame.copy(), index)
            except Exception:
                logger.exception("Video overlay callback failed")
                shown = frame
        self.view.set_frame(bgr_to_qpixmap(shown))

    # ------------------------------------------------------------- navigation
    def seek(self, frame_index: int) -> None:
        """Show ``frame_index`` (clamped); the frame arrives asynchronously."""
        if self._thread is None or self._total <= 0:
            return
        index = min(max(0, int(frame_index)), self._total - 1)
        self._token += 1
        self._thread.request(self._token, index)

    def step(self, delta: int) -> None:
        """Move ``delta`` frames from the frame on screen (pauses playback)."""
        if self._thread is None:
            return
        self.pause()
        self.seek(max(self._frame, 0) + int(delta))

    def seek_to_start(self) -> None:
        self.pause()
        self.seek(0)

    def seek_to_end(self) -> None:
        self.pause()
        self.seek(self._total - 1)

    def seek_to_time(self, seconds: float) -> None:
        if self._fps > 0:
            self.seek(int(round(seconds * self._fps)))

    def wait_idle(self, timeout_ms: int = 5000) -> bool:
        """Process events until all requested frames are shown (for tests/hosts)."""
        deadline = time.monotonic() + timeout_ms / 1000.0
        while not self.is_idle():
            if time.monotonic() > deadline:
                return False
            QCoreApplication.processEvents()
            time.sleep(0.001)
        QCoreApplication.processEvents()
        return True

    # --------------------------------------------------------------- playback
    def play(self) -> None:
        if self._thread is None or self._playing:
            return
        if self._frame >= self._total - 1:
            self.seek(0)
            self._frame = 0
        self._playing = True
        self._play_t0 = time.monotonic()
        self._play_f0 = max(self._frame, 0)
        self.play_button.setText("Pause")
        self._timer.start()
        self.playback_changed.emit(True)

    def pause(self) -> None:
        if not self._playing:
            return
        self._playing = False
        self._timer.stop()
        self.play_button.setText("Play")
        self.playback_changed.emit(False)

    def toggle_play(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()

    def set_speed(self, speed: float) -> None:
        if speed not in PLAYBACK_SPEEDS:
            raise ValueError(f"speed must be one of {PLAYBACK_SPEEDS}")
        self.speed_combo.setCurrentIndex(PLAYBACK_SPEEDS.index(speed))

    def _on_speed_changed(self, _index: int) -> None:
        speed = self.speed_combo.currentData()
        if speed is None:
            return
        self._speed = float(speed)
        if self._playing:
            self._play_t0 = time.monotonic()
            self._play_f0 = max(self._frame, 0)

    def _on_tick(self) -> None:
        if self._thread is None or not self.is_idle():
            return   # still decoding the previous frame
        elapsed = time.monotonic() - self._play_t0
        target = self._play_f0 + int(elapsed * self._fps * self._speed)
        if target <= self._frame:
            return
        if target >= self._total - 1:
            if self._frame >= self._total - 1:
                self.pause()
                return
            target = self._total - 1
        self.seek(target)

    # ------------------------------------------------------------ frame intake
    def _on_frame_ready(self, token: int, index: int, frame) -> None:
        self._answered = max(self._answered, token)
        self._last_frame = frame
        self._frame = index
        self._render(frame, index)
        self._update_position_widgets()
        self.frame_changed.emit(index, index / self._fps if self._fps > 0 else 0.0)
        if self._playing and index >= self._total - 1:
            self.pause()

    def _on_read_failed(self, token: int, index: int) -> None:
        self._answered = max(self._answered, token)
        logger.warning("Could not read video frame %d", index)
        if self._playing:
            self.pause()

    def _update_position_widgets(self) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(max(self._frame, 0))
        self.slider.blockSignals(False)
        self.time_label.setText(
            f"{format_timestamp(self.current_timestamp)}   "
            f"frame {self._frame} / {max(self._total - 1, 0)}"
        )

    # ------------------------------------------------------------------ slider
    def _on_slider_pressed(self) -> None:
        self.pause()

    def _on_slider_value(self, value: int) -> None:
        # Fired by the user (programmatic updates block the signal).
        self.pause()
        self.seek(value)

    # --------------------------------------------------------------------- keys
    def handle_key_event(self, event: QKeyEvent) -> bool:
        """Apply a navigation key; True when the key was consumed."""
        if self._thread is None:
            return False
        key = event.key()
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if key == Qt.Key.Key_Space:
            if not event.isAutoRepeat():
                self.toggle_play()
            return True
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            sign = -1 if key == Qt.Key.Key_Left else 1
            self.step(sign * (JUMP_FRAMES if shift else 1))
            return True
        if key == Qt.Key.Key_Home:
            self.seek_to_start()
            return True
        if key == Qt.Key.Key_End:
            self.seek_to_end()
            return True
        return False

    def keyPressEvent(self, event):
        if self.handle_key_event(event):
            event.accept()
        else:
            super().keyPressEvent(event)
