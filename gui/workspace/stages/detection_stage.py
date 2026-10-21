"""
gui/workspace/stages/detection_stage.py
Stage 3: Detection ("Motion tuning" for the Tail Suspension Test).

::

    +--------------------------------------------+------------------------+
    |                                            | Detection settings     |
    |   the preview, large: the video with the   |  Sensitivity  Low-High |
    |   animal marked (or what the detector      |  Animal size  S M L    |
    |   sees)                                    |  [Good] found in 98%   |
    |                                            +------------------------+
    |   frame 60 / 120     [What the detector    | Videos        (12)     |
    |   ======o=========    sees]  N P           |  list with chips       |
    |                                            |  [Quick check all]     |
    |                                            +------------------------+
    |                                            | > Advanced             |
    +--------------------------------------------+------------------------+

Tracking paradigms host the embedded ``DetectionTunerWidget`` (the preview and
the frame slider); the plain-language "Simple" card drives the same sliders as
the "Advanced" section, so both always agree. The Tail Suspension Test hosts
the embedded ``TstTunerWidget`` the same way, with its minimum-immobility
slider as the simple setting and the classifier settings under "Advanced".

How a change travels
--------------------
There is no Apply button. Every edit reaches the project's template within
``WRITE_DEBOUNCE_MS``: the stage builds a ``SetupState`` from the template,
puts the tuned detection config (or applies the tuner to the maze, for the
TST) and writes ``to_batch_settings(finalize=True)`` with ``set_template``, the
very conversion the classic wizard and the old Setup tab use, so the
resulting ``BatchSettings`` are identical for the same choices. Videos that
follow the template receive it; videos with their own full settings keep
them (a note says so). ``Project.detection_reviewed`` records that the
preview was looked at (it makes the stepper show "done").

The simple controls (documented in ``gui.tuners.detection_tuner``) never
rewrite values they did not move: a configuration that does not sit exactly
on a simple position shows "Custom (advanced)" until the control is touched.

Videos
------
The list picks which video is previewed. The tuner is rebuilt for that video's
frame (the reference frame is generated off the GUI thread with
``run_with_progress`` and kept in a small cache) and its own zones; the
configuration is always the shared template. "Quick check all videos" runs the
tuner's quick scan on every video in a background thread, with progress and
Cancel, and gives each video a Good / Check / Poor chip.

Lifecycle
---------
Tuner and quick-check threads are stopped in ``on_leave``, when the project
changes, and by the window's close guard; coming back rebuilds the tuner (the
reference frames are cached). Pending writes are flushed first, never lost.
"""
import copy
import html
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QHBoxLayout, QProgressBar, QPushButton,
                             QScrollArea, QSlider, QStackedWidget,
                             QVBoxLayout, QWidget)

from core.analysis_geometry import AnalysisGeometry
from core.arena_zones import analysis_mask
from core.models import AppConfig, DetectionConfig
from core.project import SOURCE_TEMPLATE, Project, canonical_json
from core.setup_state import SetupState
from core.video import ReferenceFrameGenerator, VideoHandler
from gui.background_task import TaskCancelled, run_with_progress
from gui.tuners.detection_tuner import (ANIMAL_SIZES, SENSITIVITY_LEVELS,
                                        SIZE_ORDER, VIEW_MASK, VIEW_VIDEO,
                                        DetectionTunerWidget, QualitySummary,
                                        SimpleState, scan_video_windows,
                                        simple_state_of, summarize_windows)
from gui.tuners.lifecycle import TunerLifecycle
from gui.tuners.tst_tuner import TstTunerWidget
from gui.workspace.stages import status as stage_status
from gui.workspace.stages.base import Stage
from gui.workspace.theme import Spacing
from gui.workspace.thumbnails import ThumbnailLoader
from gui.workspace.ui_kit import (Callout, Card, CollapsibleSection,
                                  EmptyState, Filmstrip, FilmstripItem,
                                  GhostButton, KeyHint, Label, SecondaryButton, StatusChip)
from mazes import AVAILABLE_MAZES

logger = logging.getLogger(__name__)

WRITE_DEBOUNCE_MS = 250
SCAN_DEBOUNCE_MS = 450
SIDE_WIDTH = 340

DETECTION_SUBTITLE = "Check that the animal is found in every frame."
MOTION_SUBTITLE = "Tune how movement is told apart from stillness for each mouse."

PAGE_EMPTY = 0
PAGE_WORK = 1

LEVEL_GOOD = "good"
LEVEL_CHECK = "check"
LEVEL_POOR = "poor"
LEVEL_ERROR = "error"
_BADGES = {
    LEVEL_GOOD: ("Good", "ok"),
    LEVEL_CHECK: ("Check", "warning"),
    LEVEL_POOR: ("Poor", "error"),
    LEVEL_ERROR: ("Unreadable", "error"),
}
SENSITIVITY_NAMES = (
    "Very low", "Low", "Slightly low", "Medium", "Slightly high", "High",
    "Very high",
)
SIZE_NAMES = {"small": "Small", "medium": "Medium", "large": "Large"}
CUSTOM_TEXT = "Custom (advanced)"
MAX_REFERENCE_FRAMES = 6

SENSITIVITY_TIP = (
    "How easily a pixel counts as part of the animal. Higher finds a pale "
    "animal but may also pick up shadows and noise."
)
SIZE_TIP = (
    "How big the animal looks in the video. It sets the smallest blob that "
    "counts as the animal and how far around its last position to look."
)


# ----------------------------------------------------------------------
# Quick check of every video (no Qt widgets: testable on its own)
# ----------------------------------------------------------------------
@dataclass
class QuickCheckJob:
    """One video to check: its zones decide where the animal may be."""
    key: str
    path: Path
    name: str
    roi_manager: object


@dataclass(frozen=True)
class QuickCheckResult:
    """The outcome for one video: ``level`` is good, check, poor or error."""
    key: str
    level: str
    detection_rate: float = 0.0
    message: str = ""


def check_video(job: QuickCheckJob, config: DetectionConfig,
                should_stop: Optional[Callable[[], bool]] = None
                ) -> Optional[QuickCheckResult]:
    """The tuner's quick scan on one video, as a Good / Check / Poor result.

    ``None`` when ``should_stop`` asked to stop. A video that cannot be read
    gives an ``error`` result instead of raising.
    """
    try:
        path = Path(job.path)
        info = VideoHandler(path)
        reference = ReferenceFrameGenerator.generate(
            path, AppConfig().video.ref_frame_samples, use_cache=True,
            should_cancel=should_stop,
        )
        geometry = AnalysisGeometry(
            info.width, info.height, AppConfig().video.analysis_width
        )
        analysis_ref = geometry.reference_to_analysis(reference)
        arena = getattr(job.roi_manager, "arena", None)
        mask = None if arena is None else analysis_mask(arena.points, geometry)
        windows = scan_video_windows(
            path, analysis_ref, geometry, config, info.total_frames,
            arena_mask=mask, should_stop=should_stop,
        )
        if windows is None:
            return None
        summary = summarize_windows(windows, hypot(info.width, info.height))
    except InterruptedError:
        return None
    except Exception as exc:        # unreadable video, bad frame, ...
        logger.info("Quick check of %s failed: %s", job.name, exc)
        return QuickCheckResult(job.key, LEVEL_ERROR, 0.0, str(exc))
    if summary is None:
        return QuickCheckResult(
            job.key, LEVEL_ERROR, 0.0, "The video has too few frames to check"
        )
    return QuickCheckResult(job.key, summary.level, summary.detection_rate,
                            summary.recommendation)


class QuickCheckWorker(QThread):
    """Checks the videos one after the other off the GUI thread.

    Parentless and joined by a :class:`TunerLifecycle` (see its docstring).
    ``progress(token, done, total, name)`` is emitted before each video,
    ``video_checked(token, result)`` after it, ``all_checked(token)`` when
    none was left; ``token`` tells a late signal of a cancelled worker from
    the current one's.
    """

    progress = pyqtSignal(int, int, int, str)       # token, done, total, name
    video_checked = pyqtSignal(int, object)         # token, result
    all_checked = pyqtSignal(int)                   # token

    def __init__(self, jobs: List[QuickCheckJob], config: DetectionConfig,
                 token: int = 0):
        super().__init__()
        self.token = int(token)
        self.jobs = list(jobs)
        self.config = copy.copy(config)
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()
        self.requestInterruption()

    def _should_stop(self) -> bool:
        return self._cancel.is_set() or self.isInterruptionRequested()

    def run(self) -> None:
        total = len(self.jobs)
        for index, job in enumerate(self.jobs):
            if self._should_stop():
                return
            self.progress.emit(self.token, index, total, job.name)
            result = check_video(job, self.config, self._should_stop)
            if result is None:
                return
            self.video_checked.emit(self.token, result)
        self.progress.emit(self.token, total, total, "")
        self.all_checked.emit(self.token)


# ----------------------------------------------------------------------
# The "Simple" card
# ----------------------------------------------------------------------
class SimpleDetectionCard(Card):
    """Sensitivity, animal size and the live quality chip.

    The card only reports what the user chose (``sensitivity_chosen``,
    ``size_chosen``); the stage forwards it to the tuner and shows the
    configuration back with :meth:`show_state`.
    """

    sensitivity_chosen = pyqtSignal(int)
    size_chosen = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("Detection settings", padding=Spacing.SM, parent=parent)
        top = len(SENSITIVITY_LEVELS) - 1

        # -- sensitivity
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        self.sensitivity_label = Label("Sensitivity", "body")
        self.sensitivity_value = Label("", "muted")
        self.sensitivity_value.setAlignment(Qt.AlignmentFlag.AlignRight
                                            | Qt.AlignmentFlag.AlignVCenter)
        head.addWidget(self.sensitivity_label)
        head.addWidget(self.sensitivity_value, 1)
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(0, top)
        self.sensitivity_slider.setPageStep(1)
        self.sensitivity_slider.setAccessibleName("Sensitivity")
        self.sensitivity_slider.setToolTip(SENSITIVITY_TIP)
        ends = QHBoxLayout()
        ends.setContentsMargins(0, 0, 0, 0)
        ends.addWidget(Label("Low", "caption"))
        ends.addStretch(1)
        ends.addWidget(Label("High", "caption"))
        self.body.addLayout(head)
        self.body.addWidget(self.sensitivity_slider)
        self.body.addLayout(ends)

        # -- animal size
        size_head = QHBoxLayout()
        size_head.setContentsMargins(0, 0, 0, 0)
        self.size_label = Label("Animal size", "body")
        self.size_value = Label("", "muted")
        self.size_value.setAlignment(Qt.AlignmentFlag.AlignRight
                                     | Qt.AlignmentFlag.AlignVCenter)
        size_head.addWidget(self.size_label)
        size_head.addWidget(self.size_value, 1)
        size_buttons = QHBoxLayout()
        size_buttons.setContentsMargins(0, 0, 0, 0)
        size_buttons.setSpacing(Spacing.XS)
        self.size_buttons: Dict[str, QPushButton] = {}
        for size in SIZE_ORDER:
            button = SecondaryButton(SIZE_NAMES[size])
            button.setCheckable(True)
            button.setToolTip(SIZE_TIP)
            button.setAccessibleName(f"Animal size {SIZE_NAMES[size]}")
            button.clicked.connect(lambda _checked=False, s=size: self._on_size(s))
            self.size_buttons[size] = button
            size_buttons.addWidget(button, 1)
        self.body.addSpacing(Spacing.XXS)
        self.body.addLayout(size_head)
        self.body.addLayout(size_buttons)

        # -- quality
        quality = QHBoxLayout()
        quality.setContentsMargins(0, 0, 0, 0)
        quality.setSpacing(Spacing.SM)
        self.quality_chip = StatusChip("neutral", "Not checked yet")
        self.quality_text = Label("", "muted", wrap=True)
        quality.addWidget(self.quality_chip, 0, Qt.AlignmentFlag.AlignTop)
        quality.addWidget(self.quality_text, 1)
        self.body.addSpacing(Spacing.XXS)
        self.body.addLayout(quality)

        self.sensitivity_slider.valueChanged.connect(self._on_sensitivity)
        self.sensitivity_slider.sliderReleased.connect(self._on_released)
        self.show_state(simple_state_of(DetectionConfig()))

    # -- what the user chose
    def _on_sensitivity(self, level: int) -> None:
        self.sensitivity_chosen.emit(int(level))

    def _on_released(self) -> None:
        # Applies the position even when it did not change: it snaps a
        # custom value onto the shown level.
        self.sensitivity_chosen.emit(self.sensitivity_slider.value())

    def _on_size(self, size: str) -> None:
        self._check_only(size)
        self.size_chosen.emit(size)

    def _check_only(self, size: Optional[str]) -> None:
        for name, button in self.size_buttons.items():
            button.setChecked(name == size)

    # -- what the stage shows
    def show_state(self, state: SimpleState) -> None:
        """Draw ``state``; a custom value is labelled, never rewritten."""
        blocked = self.sensitivity_slider.blockSignals(True)
        try:
            level = (state.nearest_sensitivity if state.sensitivity is None
                     else state.sensitivity)
            self.sensitivity_slider.setValue(level)
        finally:
            self.sensitivity_slider.blockSignals(blocked)
        self.sensitivity_value.setText(
            CUSTOM_TEXT if state.sensitivity is None
            else SENSITIVITY_NAMES[state.sensitivity]
        )
        self._check_only(state.size)
        self.size_value.setText(
            CUSTOM_TEXT if state.size is None else SIZE_NAMES[state.size]
        )

    def show_quality(self, summary: Optional[QualitySummary],
                     checking: bool = False, idle: str = "") -> None:
        """The chip: Good / Check / Poor and why, or a waiting message."""
        if summary is not None and not checking:
            text, kind = _BADGES[summary.level]
            self.quality_chip.set_status(kind, text)
            more = "" if summary.level == LEVEL_GOOD else f". {summary.recommendation}"
            self.quality_text.setText(summary.explanation + more)
            self.quality_text.setToolTip(summary.recommendation)
            return
        self.quality_text.setToolTip("")
        if checking:
            self.quality_chip.set_status("running", "Checking")
            self.quality_text.setText("Looking at a few frames of this video...")
        else:
            self.quality_chip.set_status("neutral", "Not checked")
            self.quality_text.setText(idle)

    def set_available(self, available: bool, reason: str = "") -> None:
        for widget in (self.sensitivity_slider, *self.size_buttons.values()):
            widget.setEnabled(available)
        self.setToolTip("" if available else reason)
        for widget in (self.sensitivity_slider, *self.size_buttons.values()):
            widget.setToolTip(
                (SENSITIVITY_TIP if widget is self.sensitivity_slider else SIZE_TIP)
                if available else reason
            )


# ----------------------------------------------------------------------
# The stage
# ----------------------------------------------------------------------
class DetectionStage(Stage):
    """Stage 3: tune the detection (or the motion classifier) with live writes."""

    stage_id = stage_status.STAGE_DETECTION
    title = stage_status.DETECTION_TITLE
    subtitle = DETECTION_SUBTITLE
    icon_name = "target"

    def __init__(self, loader: Optional[ThumbnailLoader] = None,
                 app_settings=None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._app_settings = app_settings
        self._tuner = None
        self._tuner_key: Optional[str] = None
        self._tuner_is_tst = False
        self._tst_baseline = 0
        self._mask_view = False
        self._pending_write = False
        self._writing = False
        self._write_error = ""
        self._signature_written: Optional[str] = None
        self._scan_wanted = False
        self._quality: Optional[QualitySummary] = None
        self._results: Dict[str, QuickCheckResult] = {}
        self._stale = False
        self._check_worker: Optional[QuickCheckWorker] = None
        self._check_progress = (0, 0)
        self._check_token = 0
        self._check_stopped = False
        self._frames: "OrderedDict[Tuple, object]" = OrderedDict()
        self._empty_target: Optional[str] = None
        self._extra_widgets: List[QWidget] = []
        # Joins the checking thread if the stage is deleted without
        # ``shutdown`` (a bound method: it also survives interpreter exit).
        self._checks = TunerLifecycle("detection quick check")
        self.destroyed.connect(self._checks.shutdown)

        self._write_timer = QTimer(self)
        self._write_timer.setSingleShot(True)
        self._write_timer.setInterval(WRITE_DEBOUNCE_MS)
        self._write_timer.timeout.connect(self.commit_pending)
        self._scan_timer = QTimer(self)
        self._scan_timer.setSingleShot(True)
        self._scan_timer.setInterval(SCAN_DEBOUNCE_MS)
        self._scan_timer.timeout.connect(self._run_quality_scan)
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(0)
        self._rebuild_timer.timeout.connect(self._rebuild_if_current)

        self._build_ui(loader)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_ui(self, loader: Optional[ThumbnailLoader]) -> None:
        self.pages = QStackedWidget()
        self.empty = EmptyState("zones", "Draw the arena zones first", "")
        self.pages.addWidget(self.empty)
        self.pages.addWidget(self._build_work_page(loader))
        self.set_content(self.pages)

    def _build_work_page(self, loader: Optional[ThumbnailLoader]) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(Spacing.MD)

        # -- hero: the preview
        hero = QWidget()
        hero_column = QVBoxLayout(hero)
        hero_column.setContentsMargins(0, 0, 0, 0)
        hero_column.setSpacing(Spacing.XS)
        self.caption = Label("", "muted")
        self.zones_note = Callout("warning", "")
        self.zones_note.setVisible(False)
        self.hero_layout = QVBoxLayout()
        self.hero_layout.setContentsMargins(0, 0, 0, 0)
        self.problem = EmptyState("warning", "This video could not be opened", "")
        self.problem_retry = self.problem.add_action("Try again")
        self.problem_retry.clicked.connect(self._on_retry)
        self.problem.setVisible(False)
        self.hero_layout.addWidget(self.problem, 1)
        hero_column.addWidget(self.caption)
        hero_column.addWidget(self.zones_note)
        hero_column.addLayout(self.hero_layout, 1)
        row.addWidget(hero, 1)

        # -- the side column
        side = QWidget()
        column = QVBoxLayout(side)
        column.setContentsMargins(0, 0, Spacing.XS, 0)
        column.setSpacing(Spacing.SM)

        self.simple = SimpleDetectionCard()
        self.simple.sensitivity_chosen.connect(self._on_sensitivity)
        self.simple.size_chosen.connect(self._on_size)
        column.addWidget(self.simple)

        self.motion_card = Card(
            "Motion settings",
            "Movement is told apart from stillness automatically.",
            padding=Spacing.SM,
        )
        self.motion_chip = StatusChip("neutral", "Waiting")
        self.motion_card.add_widget(self.motion_chip)
        self.motion_slot = QVBoxLayout()
        self.motion_slot.setContentsMargins(0, 0, 0, 0)
        self.motion_slot.setSpacing(Spacing.XS)
        self.motion_card.body.addLayout(self.motion_slot)
        self.motion_help = Label(
            "Stillness shorter than this is not counted as immobility.",
            "muted", wrap=True,
        )
        self.motion_card.add_widget(self.motion_help)
        column.addWidget(self.motion_card)

        self.videos_card = Card("Preview video", padding=Spacing.SM)
        self.filmstrip = Filmstrip(
            Qt.Orientation.Vertical, loader=loader, title="Preview video",
            show_header=False,
        )
        self.filmstrip.set_compact(True)
        self.filmstrip.setFixedHeight(112)
        self.filmstrip.video_selected.connect(self._on_video_picked)
        self.videos_card.add_widget(self.filmstrip)
        self.check_button = SecondaryButton("Quick check all videos", "search")
        self.check_button.clicked.connect(self.start_quick_check)
        self.videos_card.add_widget(self.check_button)
        progress_row = QWidget()
        progress_layout = QHBoxLayout(progress_row)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(Spacing.XS)
        self.check_bar = QProgressBar()
        self.check_bar.setTextVisible(True)
        self.check_bar.setFormat("%v of %m")
        self.check_cancel = GhostButton("Cancel")
        self.check_cancel.clicked.connect(self.cancel_quick_check)
        progress_layout.addWidget(self.check_bar, 1)
        progress_layout.addWidget(self.check_cancel)
        self.check_row = progress_row
        self.check_row.setVisible(False)
        self.check_label = Label("", "muted", wrap=True)
        self.check_label.setVisible(False)
        self.videos_card.add_widget(self.check_label)
        self.videos_card.add_widget(self.check_row)
        self.summary = Callout("info", "")
        self.summary.setVisible(False)
        self.videos_card.add_widget(self.summary)
        self.own_settings_note = Callout("info", "")
        self.own_settings_note.setVisible(False)
        self.videos_card.add_widget(self.own_settings_note)
        column.addWidget(self.videos_card)

        self.advanced = CollapsibleSection(
            "Advanced", expanded=False, animated=False,
            summary="All the sliders", settings=self._app_settings,
            settings_key="detection_advanced",
        )
        self.advanced_hint = Label(
            "Every setting, kept in step with the simple ones above. "
            "Changes are saved to the project as you make them.",
            "muted", wrap=True,
        )
        self.advanced.body.addWidget(self.advanced_hint)
        column.addWidget(self.advanced)
        column.addStretch(1)

        self.side_scroll = QScrollArea()
        self.side_scroll.setWidgetResizable(True)
        self.side_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.side_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.side_scroll.setFixedWidth(SIDE_WIDTH)
        self.side_scroll.setWidget(side)
        row.addWidget(self.side_scroll)
        return page

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def attach(self, window) -> None:
        super().attach(window)
        window.register_close_guard(self._close_guard)
        self._update_texts()

    @property
    def tuner(self):
        """The hosted tuner widget, or None when none is built."""
        return self._tuner

    @property
    def previewed_key(self) -> Optional[str]:
        return self._tuner_key

    @property
    def is_motion(self) -> bool:
        """True for the Tail Suspension Test (the motion tuner)."""
        return self._paradigm_step() == "tst_tuning"

    def _paradigm(self) -> str:
        project = self.project
        return "" if project is None else project.effective_paradigm()

    def _paradigm_step(self) -> str:
        return stage_status.detection_step(self._paradigm())

    def _update_texts(self) -> None:
        title = stage_status.detection_title(self._paradigm())
        subtitle = (MOTION_SUBTITLE if title == stage_status.MOTION_TITLE
                    else DETECTION_SUBTITLE)
        if title != self.title or subtitle != self.subtitle:
            self.set_texts(title, subtitle)

    # ------------------------------------------------------------------
    # The window's hooks
    # ------------------------------------------------------------------
    def on_project_changed(self, project: Optional[Project]) -> None:
        # A pending write belonged to the previous project (the close guard
        # flushed it before the switch): it must not land in the new one.
        self._pending_write = False
        self._write_timer.stop()
        self._stop_quick_check()
        self._dispose_tuner()
        self._tuner_key = None
        self._results.clear()
        self._stale = False
        self._frames.clear()
        self._quality = None
        self._write_error = ""
        self._signature_written = None
        self._update_texts()
        self._sync_filmstrip()
        self._show_summary()

    def on_project_modified(self) -> None:
        if self._writing:
            return
        self._update_texts()
        if self._is_current() and self._tuner is not None:
            if self._signature() != self._signature_written:
                self._enter()        # edited elsewhere: show the new template
                return
        self._sync_filmstrip()
        self.refresh_footer()

    def on_enter(self) -> None:
        self._enter()

    def on_leave(self) -> bool:
        self.commit_pending()
        if self._tuner is not None and self.project is not None:
            self._mark_reviewed()
        self._stop_quick_check()
        self._dispose_tuner()
        return True

    def commit_pending(self) -> bool:
        """Write a pending edit into the project now; False if it is refused."""
        self._write_timer.stop()
        if not self._pending_write:
            return True
        self._pending_write = False
        tuner, project = self._tuner, self.project
        if tuner is None or project is None or project.template_settings is None:
            return True
        try:
            candidate = self._candidate(project, tuner)
        except ValueError as exc:
            self._write_error = str(exc)
            self.refresh_footer()
            return False
        self._write_error = ""
        current = project.template_settings
        self._writing = True
        try:
            if canonical_json(candidate.to_dict()) != canonical_json(current.to_dict()):
                project.set_template(candidate)
            project.set_detection_reviewed(True)
        finally:
            self._writing = False
        self._signature_written = self._signature()
        self._refresh_own_settings_note()
        self.refresh_footer()
        return True

    def _close_guard(self) -> bool:
        """Closing the project or the window: flush, stop every thread."""
        self.commit_pending()
        self._stop_quick_check()
        if self._tuner is not None:
            self._dispose_tuner()
            self._rebuild_timer.start()     # only if closing is then refused
        return True

    def shutdown(self) -> None:
        """Stop everything (the window is closing)."""
        self.commit_pending()
        self._rebuild_timer.stop()
        self._scan_timer.stop()
        self._stop_quick_check()
        self._dispose_tuner()

    # ------------------------------------------------------------------
    # What can be shown
    # ------------------------------------------------------------------
    def _is_current(self) -> bool:
        return (self.workspace is not None
                and self.workspace.current_stage_id == self.stage_id)

    def _blocked(self) -> Optional[Tuple[str, str, str, Optional[str]]]:
        """``(title, body, action, target stage)`` when nothing can be tuned."""
        project = self.project
        if project is None:
            return ("Open or create a study first", "", "", None)
        if not stage_status.paradigm_known(project):
            return ("Choose the test first",
                    "Detection depends on the test these videos are from.",
                    "Go to Videos", stage_status.STAGE_VIDEOS)
        if not project.videos:
            return ("Add your videos first",
                    "There is nothing to preview yet.",
                    "Go to Videos", stage_status.STAGE_VIDEOS)
        template = project.template_settings
        motion = self.is_motion
        if template is None or (
            motion and not any(template.roi_manager.rois.values())
        ):
            body = (
                "Motion is measured inside each mouse's zone. Mark the zones "
                "on a reference frame, then come back."
                if motion else
                "Detection is tuned against the zones you draw. Mark the "
                "arena and zones on a reference frame, then come back."
            )
            return ("Draw the arena zones first", body,
                    "Go to Arena && zones", stage_status.STAGE_ARENA)
        return None

    def _enter(self) -> None:
        """Show the stage: the empty state, or the preview of a video."""
        self._update_texts()
        blocked = self._blocked()
        if blocked is not None:
            self._dispose_tuner()
            self._show_empty(*blocked)
            self.refresh_footer()
            return
        self.pages.setCurrentIndex(PAGE_WORK)
        self._apply_paradigm_layout()
        self._sync_filmstrip()
        self._refresh_own_settings_note()
        key = self._choose_key()
        self._show_video(key, force=True)
        self.refresh_footer()

    def _show_empty(self, title: str, body: str, action: str,
                    target: Optional[str]) -> None:
        self.pages.setCurrentIndex(PAGE_EMPTY)
        self.empty.set_texts(title, body)
        self.empty.clear_actions()
        self._empty_target = target
        if action:
            button = self.empty.add_action(action, True, "arrow-right")
            button.clicked.connect(self._on_empty_action)

    def _on_empty_action(self) -> None:
        if self._empty_target and self.workspace is not None:
            self.workspace.go_to_stage(self._empty_target)

    def _apply_paradigm_layout(self) -> None:
        motion = self.is_motion
        self.simple.setVisible(not motion)
        self.motion_card.setVisible(motion)
        self.check_button.setVisible(not motion)
        self.advanced.set_summary(
            "Classifier settings" if motion else "All the sliders")

    def _choose_key(self) -> Optional[str]:
        project = self.project
        keys = project.keys()
        if self._tuner_key in keys:
            return self._tuner_key
        selected = [] if self.workspace is None else self.workspace.selected_keys()
        for key in selected:
            if key in keys:
                return key
        return keys[0] if keys else None

    # ------------------------------------------------------------------
    # The video list
    # ------------------------------------------------------------------
    def _badge_of(self, key: str) -> Tuple[str, str]:
        result = self._results.get(key)
        if result is None or self._stale:
            return "", "neutral"
        return _BADGES[result.level]

    def _sync_filmstrip(self) -> None:
        project = self.project
        videos = [] if project is None else project.videos
        wanted = [(v.key, str(v.path)) for v in videos]
        current = [(item.key, str(item.path)) for item in self.filmstrip.items()]
        if wanted != current:
            items = []
            for video in videos:
                text, kind = self._badge_of(video.key)
                items.append(FilmstripItem(
                    video.key, video.path, video.name, text, kind,
                    tooltip=str(video.path),
                ))
            self.filmstrip.set_items(items)
        else:
            for video in videos:
                text, kind = self._badge_of(video.key)
                self.filmstrip.set_badge(video.key, text, kind)
        if self._tuner_key in self.filmstrip.keys():
            self.filmstrip.select(self._tuner_key)
        self.check_button.setEnabled(bool(videos) and self._check_worker is None)
        if not videos:
            self.check_button.setToolTip("Add videos first")

    def _on_video_picked(self, key: str) -> None:
        self.show_video(key)

    def show_video(self, key: str) -> bool:
        """Preview ``key``; False when it could not be opened or was cancelled."""
        return self._show_video(key, force=False)

    # ------------------------------------------------------------------
    # Building and disposing the tuner
    # ------------------------------------------------------------------
    def _reference_frame(self, path: Path):
        """The background frame of ``path`` (cached; generated off-thread)."""
        stat = Path(path).stat()
        cache_key = (str(path), stat.st_size, stat.st_mtime_ns)
        frame = self._frames.get(cache_key)
        if frame is not None:
            self._frames.move_to_end(cache_key)
            return frame
        samples = AppConfig().video.ref_frame_samples
        frame = run_with_progress(
            self.window(),
            f"Generating reference frame for {Path(path).name}...",
            lambda should_cancel: ReferenceFrameGenerator.generate(
                Path(path), samples, use_cache=True,
                should_cancel=should_cancel,
            ),
            title="Reference Frame",
        )
        self._frames[cache_key] = frame
        while len(self._frames) > MAX_REFERENCE_FRAMES:
            self._frames.popitem(last=False)
        return frame

    def _template_state(self, project: Project) -> SetupState:
        template = project.template_settings
        name = (None if template.paradigm_id in AVAILABLE_MAZES
                else project.effective_paradigm())
        return SetupState.from_batch_settings(template, maze_name=name)

    def _signature(self) -> Optional[str]:
        project = self.project
        template = None if project is None else project.template_settings
        if template is None:
            return None
        return canonical_json(template.to_dict())

    def _create_tuner(self, key: str):
        project = self.project
        entry = project.entry(key)
        template = project.template_settings
        settings = project.effective_settings(entry) or template
        if self.is_motion:
            maze = self._template_state(project).build_maze()
            return TstTunerWidget(
                str(entry.path), settings.roi_manager, maze,
                initial_motion_config=maze.motion_config, embedded=True,
            )
        frame = self._reference_frame(entry.path)
        return DetectionTunerWidget(
            str(entry.path), frame, settings.roi_manager,
            initial_config=template.detection_config, embedded=True,
        )

    def _show_video(self, key: Optional[str], force: bool) -> bool:
        if key is None or self.project is None:
            return False
        if not force and key == self._tuner_key and self._tuner is not None:
            return True
        self.commit_pending()
        try:
            tuner = self._create_tuner(key)
        except TaskCancelled:
            return self._failed(
                key, "Preparing the preview was cancelled.", cancelled=True)
        except Exception as exc:
            logger.warning("Could not open %s in the detection stage: %s",
                           key, exc)
            return self._failed(key, str(exc) or exc.__class__.__name__)
        self._dispose_tuner()
        self._tuner_key = key
        self._install_tuner(tuner)
        self.filmstrip.select(key)
        return True

    def _failed(self, key: str, message: str, cancelled: bool = False) -> bool:
        """The video could not be previewed: keep what is on screen if any."""
        if self._tuner is not None:
            if self._tuner_key in self.filmstrip.keys():
                self.filmstrip.select(self._tuner_key)
            if self.workspace is not None:
                self.workspace.show_message(message)
            return False
        self._tuner_key = key
        self.filmstrip.select(key)
        self.problem.set_texts(
            "The preview was cancelled" if cancelled
            else "This video could not be opened",
            message,
        )
        self.problem.setVisible(True)
        self.caption.setText(self._caption_for(key))
        self._set_tuner_controls(False)
        self.refresh_footer()
        return False

    def _caption_for(self, key: str) -> str:
        try:
            return f"Previewing {self.project.entry(key).name}"
        except KeyError:
            return ""

    def _install_tuner(self, tuner) -> None:
        project = self.project
        self._tuner = tuner
        self._tuner_is_tst = self.is_motion
        self.problem.setVisible(False)
        self.caption.setText(self._caption_for(self._tuner_key))
        self.caption.setVisible(not self.is_motion)
        self.hero_layout.insertWidget(0, tuner, 1)
        self._extra_widgets = []
        if self._tuner_is_tst:
            self._install_motion_controls(tuner)
        else:
            self._install_detection_controls(tuner)
        self._signature_written = self._signature()
        self._note_frame_size(tuner)
        self._set_tuner_controls(True)
        tuner.show()
        tuner.setFocus()
        self.refresh_footer()
        if project is not None and not self._tuner_is_tst:
            self._request_scan(immediate=True)

    def _install_detection_controls(self, tuner: DetectionTunerWidget) -> None:
        # The hero's footer row: the mask toggle and the key hints.
        self.mask_button = SecondaryButton("What the detector sees", "eye")
        self.mask_button.setCheckable(True)
        self.mask_button.setChecked(self._mask_view)
        self.mask_button.setToolTip(
            "Show the picture the detector works on instead of the video. "
            "The animal it picked is the green blob."
        )
        self.mask_button.toggled.connect(self._on_mask_toggled)
        hint_row = QWidget()
        hints = QHBoxLayout(hint_row)
        hints.setContentsMargins(0, 0, 0, 0)
        hints.setSpacing(Spacing.XXS)
        hints.addWidget(KeyHint("P"))
        hints.addWidget(KeyHint("N"))
        hints.addWidget(Label("step through frames", "muted"))
        tuner.footer_layout.addWidget(hint_row)
        tuner.footer_layout.addSpacing(Spacing.SM)
        tuner.footer_layout.addWidget(self.mask_button)
        self._extra_widgets += [hint_row, self.mask_button]
        self.advanced.body.addWidget(tuner.advanced_panel)
        self._extra_widgets.append(tuner.advanced_panel)
        if self._mask_view:
            tuner.set_view_mode(VIEW_MASK)
        self.simple.show_state(tuner.simple_state())
        self.simple.show_quality(None, checking=True)
        self._quality = None
        tuner.config_changed.connect(self._on_detection_config_changed)
        tuner.quality_changed.connect(self._on_quality_changed)
        tuner.scan_finished.connect(self._on_tuner_scan_finished)

    def _install_motion_controls(self, tuner: TstTunerWidget) -> None:
        # The classifier indicator is automatic and disabled: not shown.
        for name in ("energy_label", "energy_slider"):
            getattr(tuner, name).setVisible(False)
        self.motion_slot.addWidget(tuner.parameter_group)
        self._extra_widgets.append(tuner.parameter_group)
        self.advanced.body.addWidget(tuner.advanced_group)
        self._extra_widgets.append(tuner.advanced_group)
        self._tst_baseline = tuner.time_slider.value()
        self.motion_chip.set_status("running", "Scanning the video")
        tuner.config_changed.connect(self._on_motion_config_changed)
        tuner.scan_finished.connect(self._on_motion_scan_finished)

    def _note_frame_size(self, tuner) -> None:
        """Warn when the zones were drawn on a differently sized frame."""
        manager = getattr(tuner, "roi_manager", None)
        dims = getattr(manager, "source_dimensions", None)
        try:
            info = VideoHandler(Path(tuner.video_path))
            actual = (info.width, info.height)
        except Exception:
            actual = None
        mismatch = dims is not None and actual is not None and tuple(dims) != actual
        self.zones_note.setVisible(bool(mismatch))
        if mismatch:
            self.zones_note.set_text(
                f"This video is {actual[0]}x{actual[1]} but its zones were "
                f"drawn on {dims[0]}x{dims[1]}. The zones in the preview may "
                "be misplaced: fix them in Arena & zones."
            )

    def _set_tuner_controls(self, available: bool) -> None:
        reason = "There is no preview to tune: this video could not be opened"
        self.simple.set_available(available, reason)
        if not available:
            template = None if self.project is None else self.project.template_settings
            if template is not None:
                self.simple.show_state(simple_state_of(template.detection_config))
            self.simple.show_quality(None, idle="Nothing to check")
        self.advanced.setEnabled(available)
        self.advanced.setToolTip("" if available else reason)
        self.check_button.setToolTip(
            "" if available else "Open a preview first: the check uses its settings")

    def _dispose_tuner(self) -> None:
        """Stop the tuner's threads and free its widgets (idempotent)."""
        tuner = self._tuner
        self._scan_timer.stop()
        self._scan_wanted = False
        if tuner is None:
            return
        self._tuner = None
        for signal, slot in (
            (getattr(tuner, "config_changed", None),
             self._on_motion_config_changed if self._tuner_is_tst
             else self._on_detection_config_changed),
            (getattr(tuner, "quality_changed", None), self._on_quality_changed),
            (tuner.scan_finished,
             self._on_motion_scan_finished if self._tuner_is_tst
             else self._on_tuner_scan_finished),
        ):
            if signal is None:
                continue
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        tuner.shutdown()
        for widget in self._extra_widgets:
            widget.setParent(None)
            widget.deleteLater()
        self._extra_widgets = []
        self.hero_layout.removeWidget(tuner)
        tuner.setParent(None)
        tuner.deleteLater()

    def _rebuild_if_current(self) -> None:
        if (self._tuner is None and self.project is not None
                and self._is_current()
                and not getattr(self.workspace, "_closed", False)):
            self._enter()

    def _on_retry(self) -> None:
        self._enter()

    # ------------------------------------------------------------------
    # Tracking paradigms: the simple controls and live writes
    # ------------------------------------------------------------------
    def _on_sensitivity(self, level: int) -> None:
        if self._tuner is not None and not self._tuner_is_tst:
            self._tuner.set_sensitivity(level)

    def _on_size(self, size: str) -> None:
        if self._tuner is not None and not self._tuner_is_tst:
            self._tuner.set_animal_size(size)

    def _on_mask_toggled(self, checked: bool) -> None:
        self._mask_view = bool(checked)
        if self._tuner is not None and not self._tuner_is_tst:
            self._tuner.set_view_mode(VIEW_MASK if checked else VIEW_VIDEO)

    def _on_detection_config_changed(self, _config=None) -> None:
        tuner = self._tuner
        if tuner is None:
            return
        self.simple.show_state(tuner.simple_state())
        self.simple.show_quality(None, checking=True)
        self._quality = None
        self._invalidate_check()
        self._schedule_write()
        self._request_scan()

    def _on_motion_config_changed(self) -> None:
        self._schedule_write()

    def _on_motion_scan_finished(self) -> None:
        if self._tuner is not None:
            self.motion_chip.set_status("ok", "Preview ready")

    def _schedule_write(self) -> None:
        self._pending_write = True
        self._write_timer.start()

    def _candidate(self, project: Project, tuner):
        """The template ``BatchSettings`` this tuner's state produces (the
        classic wizard's conversion, so identical for identical choices)."""
        state = self._template_state(project)
        if self._tuner_is_tst:
            maze = state.build_maze()
            original_bout = maze.min_time_seconds
            tuner.apply_to_maze(maze)       # ValueError: invalid classifier
            if tuner.time_slider.value() == self._tst_baseline:
                # The slider only shows a rounded bout length: keep the exact
                # value of the parameters unless the user moved it.
                maze.min_time_seconds = original_bout
            state.set_maze(maze)
        else:
            state.detection_config = tuner.detection_config()
        return state.to_batch_settings(finalize=True)

    def _mark_reviewed(self) -> None:
        project = self.project
        if project is None or project.template_settings is None:
            return
        self._writing = True
        try:
            project.set_detection_reviewed(True)
        finally:
            self._writing = False

    # -- the live quality of the previewed video
    def _request_scan(self, immediate: bool = False) -> None:
        if self._tuner is None or self._tuner_is_tst:
            return
        self._scan_wanted = True
        self._scan_timer.start(0 if immediate else SCAN_DEBOUNCE_MS)

    def _run_quality_scan(self) -> None:
        tuner = self._tuner
        if tuner is None or self._tuner_is_tst or not self._scan_wanted:
            return
        worker = tuner.scan_worker
        if worker is not None and worker.isRunning():
            return              # re-requested when it finishes
        self._scan_wanted = False
        tuner.run_quick_scan()

    def _on_tuner_scan_finished(self) -> None:
        if self._tuner is None:
            return
        if self._scan_wanted:
            self._run_quality_scan()
        elif self._quality is None:
            summary = self._tuner.quality_summary()
            if summary is not None:
                self._on_quality_changed(summary)

    def _on_quality_changed(self, summary) -> None:
        if self._tuner is None or self._scan_wanted:
            return
        self._quality = summary
        self.simple.show_quality(summary)
        self.refresh_footer()

    # ------------------------------------------------------------------
    # Quick check of every video
    # ------------------------------------------------------------------
    @property
    def checking(self) -> bool:
        return self._check_worker is not None

    def _jobs(self) -> List[QuickCheckJob]:
        project = self.project
        template = project.template_settings
        jobs = []
        for entry in project.videos:
            settings = project.effective_settings(entry) or template
            jobs.append(QuickCheckJob(entry.key, entry.path, entry.name,
                                      settings.roi_manager))
        return jobs

    def start_quick_check(self) -> bool:
        """Check every video with the current settings (background thread)."""
        project = self.project
        if (self._check_worker is not None or self._tuner is None
                or self._tuner_is_tst or project is None
                or project.template_settings is None):
            return False
        self.commit_pending()
        config = project.template_settings.detection_config
        jobs = self._jobs()
        if not jobs:
            return False
        self._results.clear()
        self._stale = False
        self._check_stopped = False
        self._check_progress = (0, len(jobs))
        self._check_token += 1
        worker = QuickCheckWorker(jobs, config, self._check_token)
        worker.progress.connect(self._on_check_progress)
        worker.video_checked.connect(self._on_video_checked)
        worker.all_checked.connect(self._on_all_checked)
        worker.finished.connect(self._on_check_finished)
        self._check_worker = worker
        self._sync_filmstrip()
        self.check_button.setEnabled(False)
        self.check_bar.setRange(0, len(jobs))
        self.check_bar.setValue(0)
        self.check_row.setVisible(True)
        self.check_cancel.setEnabled(True)
        self.check_label.setText(f"Checking {jobs[0].name}...")
        self.check_label.setVisible(True)
        self.summary.setVisible(False)
        self._checks.start_thread(worker, cancel=worker.cancel, timeout_ms=15000)
        self.refresh_footer()
        return True

    def cancel_quick_check(self) -> None:
        worker = self._check_worker
        if worker is None:
            return
        self._check_stopped = True
        self.check_cancel.setEnabled(False)
        self.check_label.setText("Cancelling...")
        worker.cancel()

    def _stop_quick_check(self) -> None:
        """Interrupt and join the checking thread (leaving, closing)."""
        worker = self._check_worker
        if worker is not None:
            self._check_stopped = True
            self._checks.stop_threads()
        self._finish_check_ui()

    def _current_check(self, token: int) -> bool:
        return self._check_worker is not None and token == self._check_token

    def _on_check_progress(self, token: int, done: int, total: int,
                           name: str) -> None:
        if not self._current_check(token):
            return
        self._check_progress = (done, total)
        self.check_bar.setRange(0, max(1, total))
        self.check_bar.setValue(done)
        if name and not self._check_stopped:
            self.check_label.setText(f"Checking {name}...")
        self.refresh_footer()

    def _on_video_checked(self, token: int, result: QuickCheckResult) -> None:
        if not self._current_check(token):
            return
        self._results[result.key] = result
        text, kind = _BADGES[result.level]
        self.filmstrip.set_badge(result.key, text, kind)
        self.refresh_footer()

    def _on_all_checked(self, token: int) -> None:
        if self._current_check(token):
            self._mark_reviewed()

    def _on_check_finished(self) -> None:
        worker = self._check_worker
        if worker is not None and worker.isRunning():
            return      # the finish of an earlier, cancelled check
        self._finish_check_ui()
        self._show_summary()
        self.refresh_footer()

    def _finish_check_ui(self) -> None:
        self._check_worker = None
        self.check_row.setVisible(False)
        self.check_label.setVisible(False)
        self.check_button.setEnabled(self.project is not None
                                     and bool(self.project.videos))

    def _invalidate_check(self) -> None:
        """The settings changed: the chips of the last check are out of date."""
        if self._results and not self._stale:
            self._stale = True
            for key in self._results:
                self.filmstrip.set_badge(key, "", "neutral")
            self._show_summary()

    def check_counts(self) -> Optional[Tuple[int, int, int]]:
        """``(good, to check, unreadable)`` of the last check (None: none/stale)."""
        if not self._results or self._stale:
            return None
        levels = [r.level for r in self._results.values()]
        return (levels.count(LEVEL_GOOD),
                levels.count(LEVEL_CHECK) + levels.count(LEVEL_POOR),
                levels.count(LEVEL_ERROR))

    def attention_names(self) -> List[str]:
        """Names of the videos whose last check was not Good."""
        project = self.project
        if project is None or self._stale:
            return []
        names = []
        for entry in project.videos:
            result = self._results.get(entry.key)
            if result is not None and result.level != LEVEL_GOOD:
                names.append(f"{entry.name} ({_BADGES[result.level][0]})")
        return names

    def _show_summary(self) -> None:
        """The callout under the list: what the last check found."""
        total = 0 if self.project is None else len(self.project.videos)
        if self._results and self._stale:
            self.summary.set_kind("info")
            self.summary.set_text(
                "The settings changed since the last check. Run it again to "
                "check every video with the new settings.")
            self.summary.setVisible(True)
            return
        counts = self.check_counts()
        if counts is None:
            self.summary.setVisible(False)
            return
        good, to_check, unreadable = counts
        done = len(self._results)
        names = self.attention_names()
        if not names:
            self.summary.set_kind("success")
            self.summary.set_text(
                f"All {done} checked videos look good."
                if done == total else
                f"The {done} videos checked so far look good "
                f"({total - done} not checked).")
        else:
            shown = ", ".join(html.escape(n) for n in names[:5])
            more = f" and {len(names) - 5} more" if len(names) > 5 else ""
            self.summary.set_kind("warning")
            self.summary.set_text(
                f"<b>{len(names)} of {done} videos need attention:</b> "
                f"{shown}{more}. Pick one in the list to look at it and adjust "
                "the settings.")
        self.summary.setVisible(True)

    def _refresh_own_settings_note(self) -> None:
        project = self.project
        own = 0 if project is None else sum(
            1 for v in project.videos
            if v.settings is not None and v.settings_source != SOURCE_TEMPLATE
        )
        self.own_settings_note.setVisible(own > 0)
        if own:
            self.own_settings_note.set_text(
                f"{own} video{'s' if own != 1 else ''} "
                f"{'have' if own != 1 else 'has'} settings of "
                f"{'their' if own != 1 else 'its'} own and will not follow "
                "changes made here."
            )

    # ------------------------------------------------------------------
    # The footer
    # ------------------------------------------------------------------
    def next_label(self) -> Optional[str]:
        return "Trials"

    def can_advance(self) -> Tuple[bool, str]:
        blocked = self._blocked()
        if blocked is not None:
            return False, blocked[0]
        if self._write_error:
            return False, self._write_error
        found = stage_status.detection_status(self.project)
        if found.status == stage_status.STATUS_ATTENTION:
            return False, found.reason
        return True, ""

    def footer_text(self):
        text, chip = super().footer_text()
        if self._blocked() is not None:
            return text, chip
        if self._write_error:
            return self._write_error, ("error", "Needs attention")
        if self._check_worker is not None:
            done, total = self._check_progress
            return f"Checking videos... {done} of {total}", chip
        counts = self.check_counts()
        if counts is not None:
            good, to_check, unreadable = counts
            text = f"{good} good · {to_check} to check"
            if unreadable:
                text += f" · {unreadable} unreadable"
            return text, chip
        if self._quality is not None and not self._tuner_is_tst:
            return self._quality.explanation, chip
        return text, chip


__all__ = [
    "ANIMAL_SIZES", "DetectionStage", "LEVEL_CHECK", "LEVEL_ERROR",
    "LEVEL_GOOD", "LEVEL_POOR", "PAGE_EMPTY", "PAGE_WORK",
    "QuickCheckJob", "QuickCheckResult", "QuickCheckWorker",
    "SimpleDetectionCard", "check_video",
]
