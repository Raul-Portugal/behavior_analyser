"""
gui/workspace/setup_steps.py
The widgets the Setup tab hosts, one per setup step.

Every step widget is a :class:`StepHost` with the same small contract, so the
tab never branches on the step:

``load(state)``     show a ``SetupState`` (called once, before the host is
                    shown);
``apply(state)``    write the step into ``state`` and mark it done; False (and
                    no change) while the step has blocking issues;
``issues()``        what currently blocks ``apply``;
``is_dirty()``      the user edited the step since it was loaded / applied;
``mark_clean()``    ``apply`` was accepted by the project;
``shutdown()``      stop threads, release captures (idempotent, final).

The hosts wrap the existing building blocks (setup panels, ``RoiEditorWidget``
and the tuner widgets) and add nothing of their own to what a step computes,
so the settings the Setup tab produces are the ones the classic wizard
produces for the same choices.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QScrollArea,
                             QVBoxLayout, QWidget)

from core.setup_state import (STATUS_DONE, STATUS_INVALID, STATUS_TODO,
                              SetupState)
from gui.background_task import TaskCancelled
from gui.roi_editor import RoiEditorWidget
from gui.setup_panels import (CalibrationPanel, ParadigmParamsPanel,
                              ReviewPanel, SetupPanel, TimeWindowPanel)
from gui.tuners.detection_tuner import DetectionTunerWidget
from gui.tuners.tst_tuner import TstTunerWidget
from mazes.base_maze import (STEP_CALIBRATION, STEP_DETECTION_TUNING,
                             STEP_PARADIGM_PARAMS, STEP_REVIEW, STEP_ROIS,
                             STEP_TIME_WINDOW, STEP_TST_TUNING)

logger = logging.getLogger(__name__)

ICON_SIZE = 16
_STATUS_COLORS = {
    STATUS_DONE: "#2da44e",
    STATUS_TODO: "#8c959f",
    STATUS_INVALID: "#d1242f",
}
_icons = {}


def status_icon(status: str) -> QIcon:
    """A small round icon: green tick (done), grey ring (todo), red ! (invalid)."""
    icon = _icons.get(status)
    if icon is not None:
        return icon
    color = QColor(_STATUS_COLORS.get(status, _STATUS_COLORS[STATUS_TODO]))
    pixmap = QPixmap(ICON_SIZE, ICON_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    rect = pixmap.rect().adjusted(1, 1, -1, -1)
    if status == STATUS_TODO:
        painter.setPen(QPen(color, 2))
        painter.drawEllipse(rect)
    else:
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(rect)
        painter.setPen(QPen(QColor("white"), 2))
        if status == STATUS_DONE:
            painter.drawLine(4, 8, 7, 11)
            painter.drawLine(7, 11, 12, 5)
        else:
            painter.drawLine(8, 4, 8, 9)
            painter.drawPoint(8, 12)
    painter.end()
    icon = QIcon(pixmap)
    _icons[status] = icon
    return icon


# ----------------------------------------------------------------------
# Services the Setup tab lends to the hosts
# ----------------------------------------------------------------------
@dataclass
class StepServices:
    """What a host needs from the outside, so tests can replace it."""
    read_frame: Callable[[Path], Optional[np.ndarray]]
    reference_frame: Callable[[Path], np.ndarray]
    project_lines: Callable[[], List[str]]


# ----------------------------------------------------------------------
# The contract
# ----------------------------------------------------------------------
class StepHost(QWidget):
    """Base of the step widgets (see the module docstring)."""

    changed = pyqtSignal()
    step_id = ""
    # A host that wants the tab to jump to another step sets these.
    step_requested = pyqtSignal(str, str)       # (step id, tool)
    retry_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._dirty = False
        self._shut_down = False

    def load(self, state: SetupState) -> None:
        pass

    def apply(self, state: SetupState) -> bool:
        return True

    def issues(self) -> List[str]:
        return []

    def is_dirty(self) -> bool:
        return self._dirty

    def mark_clean(self) -> None:
        self._dirty = False

    def shutdown(self) -> None:
        self._shut_down = True

    def _touch(self, *_args) -> None:
        """Slot for control signals: the user edited the step."""
        self._dirty = True
        self.changed.emit()


class MessageHost(StepHost):
    """A step that cannot be edited right now, and why."""

    def __init__(self, step_id: str, text: str, retry: bool = False,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.step_id = step_id
        layout = QVBoxLayout(self)
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
        layout.addWidget(self.label)
        self.retry_button = QPushButton("Try again")
        self.retry_button.setVisible(retry)
        self.retry_button.clicked.connect(lambda _checked=False: self.retry_requested.emit())
        layout.addWidget(self.retry_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()

    @property
    def text(self) -> str:
        return self.label.text()

    def apply(self, state: SetupState) -> bool:
        return False

    def issues(self) -> List[str]:
        return [self.label.text()]


# ----------------------------------------------------------------------
# Setup panels
# ----------------------------------------------------------------------
class PanelHost(StepHost):
    """Hosts one ``SetupPanel`` (parameters, calibration, time window, review)."""

    def __init__(self, panel: SetupPanel, extra: Optional[QWidget] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.step_id = panel.step_id
        self.panel = panel
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(panel)
        if extra is not None:
            layout.addWidget(extra)
        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        panel.changed.connect(self._touch)

    def load(self, state: SetupState) -> None:
        self.panel.load(state)
        self._dirty = False

    def apply(self, state: SetupState) -> bool:
        return bool(self.panel.apply(state))

    def issues(self) -> List[str]:
        return self.panel.issues()


class CalibrationHost(PanelHost):
    """The calibration panel plus a shortcut to the reference-line tool."""

    def __init__(self, panel: CalibrationPanel, parent: Optional[QWidget] = None):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        self.draw_line_button = QPushButton("Draw reference line...")
        self.draw_line_button.setToolTip(
            "Open the regions step with the reference-line tool selected"
        )
        row_layout.addWidget(self.draw_line_button)
        row_layout.addStretch()
        super().__init__(panel, extra=row, parent=parent)
        self.draw_line_button.clicked.connect(
            lambda _checked=False: self.step_requested.emit(STEP_ROIS, "line")
        )


class ReviewHost(PanelHost):
    """The review panel plus the project's and the maze's per-video issues."""

    def __init__(self, panel: ReviewPanel, project_lines: Callable[[], List[str]],
                 parent: Optional[QWidget] = None):
        self._project_lines = project_lines
        self.project_label = QLabel()
        self.project_label.setWordWrap(True)
        self.project_label.setTextFormat(Qt.TextFormat.RichText)
        super().__init__(panel, extra=self.project_label, parent=parent)
        panel.changed.connect(self.refresh_project_issues)

    def load(self, state: SetupState) -> None:
        super().load(state)
        self.refresh_project_issues()

    def refresh_project_issues(self) -> None:
        lines = self._project_lines()
        if lines:
            body = "<br>".join(f"&bull; {line}" for line in lines)
            self.project_label.setText(f"<b>Videos of the project:</b><br>{body}")
        else:
            self.project_label.setText("")
        self.project_label.setVisible(bool(lines))

    def project_issue_lines(self) -> List[str]:
        return list(self._project_lines())


# ----------------------------------------------------------------------
# Regions of interest
# ----------------------------------------------------------------------
class RoiHost(StepHost):
    """The ROI editor on a frame of the reference video."""

    step_id = STEP_ROIS

    def __init__(self, maze, frame: np.ndarray, state: SetupState,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        manager = state.roi_manager
        has_shapes = manager is not None and (
            bool(manager.rois) or getattr(manager, "arena", None) is not None
            or manager.reference_length_pixels > 0
        )
        self.editor = RoiEditorWidget(
            maze, frame, parent=self,
            initial_roi_manager=manager if has_shapes else None,
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.editor)
        self.editor.changed.connect(self._touch)
        self.editor.validity_changed.connect(self._on_validity)

    def _on_validity(self, _valid, _issues) -> None:
        self.changed.emit()

    def set_tool(self, tool: str) -> bool:
        return self.editor.set_tool(tool)

    def is_dirty(self) -> bool:
        return self.editor.has_unsaved_work()

    def mark_clean(self) -> None:
        self.editor.mark_saved()

    def issues(self) -> List[str]:
        return self.editor.issues()

    def apply(self, state: SetupState) -> bool:
        if not self.editor.is_valid():
            return False
        state.roi_manager = self.editor.roi_manager()
        state.mark_done(STEP_ROIS)
        return True


# ----------------------------------------------------------------------
# Tuners
# ----------------------------------------------------------------------
class DetectionHost(StepHost):
    """The detection tuner; ``shutdown`` joins its threads."""

    step_id = STEP_DETECTION_TUNING

    def __init__(self, video_path: Path, ref_frame: np.ndarray,
                 state: SetupState, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.tuner = DetectionTunerWidget(
            str(video_path), ref_frame, state.roi_manager,
            initial_config=state.detection_config, parent=self,
        )
        self._baseline = self.tuner.detection_config()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tuner)
        self.tuner.config_changed.connect(self._touch)

    def is_dirty(self) -> bool:
        return self.tuner.detection_config() != self._baseline

    def mark_clean(self) -> None:
        self._baseline = self.tuner.detection_config()

    def apply(self, state: SetupState) -> bool:
        state.detection_config = self.tuner.detection_config()
        state.mark_done(STEP_DETECTION_TUNING)
        return True

    def shutdown(self) -> None:
        self.tuner.shutdown()
        super().shutdown()


class TstHost(StepHost):
    """The TST motion tuner; ``shutdown`` joins its scanner thread."""

    step_id = STEP_TST_TUNING

    def __init__(self, video_path: Path, state: SetupState,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._maze = state.build_maze()
        self.tuner = TstTunerWidget(
            str(video_path), state.roi_manager, self._maze,
            initial_motion_config=self._maze.motion_config, parent=self,
        )
        self._baseline = self._snapshot()
        self._last_error = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tuner)
        self.tuner.config_changed.connect(self._touch)

    def _snapshot(self):
        return (self.tuner.motion_config(), self.tuner.time_slider.value())

    def is_dirty(self) -> bool:
        return self._snapshot() != self._baseline

    def mark_clean(self) -> None:
        self._baseline = self._snapshot()

    def issues(self) -> List[str]:
        return [self._last_error] if self._last_error else []

    def apply(self, state: SetupState) -> bool:
        maze = state.build_maze()
        original_bout = maze.min_time_seconds
        try:
            self.tuner.apply_to_maze(maze)
        except ValueError as exc:
            self._last_error = str(exc)
            return False
        self._last_error = ""
        if self.tuner.time_slider.value() == self._baseline[1]:
            # The slider only shows a rounded bout length: keep the exact
            # value of the parameters step unless the user moved it.
            maze.min_time_seconds = original_bout
        state.set_maze(maze)
        state.mark_done(STEP_TST_TUNING)
        return True

    def shutdown(self) -> None:
        self.tuner.shutdown()
        super().shutdown()


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------
NO_VIDEO_TEXT = (
    "This step needs a video to show. Add a video to the project (or select "
    "one) and come back."
)
NO_ROIS_TEXT = (
    "Draw and apply the regions of interest first: this step tunes against "
    "them."
)


def create_host(
    step: str, state: SetupState, video: Optional[Path],
    services: StepServices, parent: Optional[QWidget] = None,
) -> StepHost:
    """The host for ``step``, loaded with ``state`` (never raises)."""
    try:
        return _create_host(step, state, video, services, parent)
    except TaskCancelled:
        return MessageHost(
            step, "Generating the reference frame was cancelled.",
            retry=True, parent=parent,
        )
    except Exception as exc:  # unreadable video, bad frame, ...
        logger.warning("Could not open the %s step", step, exc_info=True)
        return MessageHost(
            step, f"This step could not be opened:\n{exc}", retry=True,
            parent=parent,
        )


def _create_host(step, state, video, services, parent) -> StepHost:
    maze = state.build_maze()
    host: StepHost
    if step == STEP_PARADIGM_PARAMS:
        host = PanelHost(ParadigmParamsPanel(), parent=parent)
    elif step == STEP_CALIBRATION:
        host = CalibrationHost(CalibrationPanel(), parent=parent)
    elif step == STEP_TIME_WINDOW:
        host = PanelHost(TimeWindowPanel(), parent=parent)
    elif step == STEP_REVIEW:
        host = ReviewHost(ReviewPanel(), services.project_lines, parent=parent)
    elif step == STEP_ROIS:
        if video is None:
            return MessageHost(step, NO_VIDEO_TEXT, parent=parent)
        frame = services.read_frame(video)
        if frame is None:
            return MessageHost(
                step, f"A frame could not be read from {Path(video).name}.",
                retry=True, parent=parent,
            )
        return _loaded(RoiHost(maze, frame, state, parent=parent), state)
    elif step in (STEP_DETECTION_TUNING, STEP_TST_TUNING):
        if video is None:
            return MessageHost(step, NO_VIDEO_TEXT, parent=parent)
        if state.roi_manager is None or (
            step == STEP_TST_TUNING and not state.roi_manager.rois
        ):
            return MessageHost(step, NO_ROIS_TEXT, parent=parent)
        if step == STEP_TST_TUNING:
            host = TstHost(video, state, parent=parent)
        else:
            ref_frame = services.reference_frame(video)
            host = DetectionHost(video, ref_frame, state, parent=parent)
        return host
    else:
        return MessageHost(step, f"Unknown step: {step}", parent=parent)
    return _loaded(host, state)


def _loaded(host: StepHost, state: SetupState) -> StepHost:
    host.load(state)
    return host


__all__ = [
    "CalibrationHost", "DetectionHost", "MessageHost", "PanelHost",
    "ReviewHost", "RoiHost", "StepHost", "StepServices", "TstHost",
    "create_host", "status_icon",
]
