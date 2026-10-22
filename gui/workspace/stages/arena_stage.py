"""
gui/workspace/stages/arena_stage.py
Stage 2: the arena and its zones.

::

    +--------+---------------------------------------------+------------+
    | Videos |  notice (a Callout, only when there is one)  | Editing    |
    | strip  |  +---------------------------------------+  | o All      |
    |  or    |  |                                       |  | o This one |
    |  rail  |  |   the ROI editor, full size           |  +------------+
    |        |  |   (or the template zones, read-only)  |  | Calibration|
    |        |  |                                       |  +------------+
    |        |  +---------------------------------------+  | Paradigm   |
    +--------+---------------------------------------------+------------+

What it edits
-------------
**All videos (template).** The default. The editor shows the template's
zones on the frame of a reference video (the first video, or the one picked
in the video list) and every edit goes straight to ``Project.set_template``.
The settings are built through :class:`~core.setup_state.SetupState`,
exactly as the old Setup tab did, so what is stored is byte-identical to what
that tab (and the classic wizard) stores for the same drawing.

**This video only.** The picked video shows its *effective* zones on its own
frame:

* a video that follows the template shows the template zones **read-only**
  with a note ("Looks off? Adjust zones for this video"), so a crooked
  camera is easy to spot;
* **Adjust zones for this video** gives it a zone override
  (``Project.set_zone_override(source="adjusted")``) and opens the editor on
  that video's frame; every edit is written live;
* **Auto-align to template** registers the video's background against the
  template video (``core.roi_alignment``) and stores an ``aligned`` override
  with the transform and the rescaled calibration;
* **Reset to template** drops the override.

Tools > Align template to videos... runs the same auto-alignment over every
video flagged "Check alignment", with one summary at the end.

Live writes
-----------
Nothing has an Apply button. Edits of the editor, the calibration length and
the paradigm options are written after ``WRITE_DEBOUNCE_MS`` of quiet (so a
vertex drag is one write), and *flushed* (never asked about) when the user
leaves the stage, asks whether "Next" is allowed, closes the project or the
window. An edit that leaves the zones incomplete is still written: the stage
then reports "needs attention" instead of silently dropping the drawing.

Video facts
-----------
The frame size of every video is what lets the project flag a template that
does not fit it (``zones_size_mismatch``). Nobody else records it yet, so the
stage reads it from the shared thumbnail loader and calls
``Project.set_video_info`` once per video.
"""
import copy
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (QAbstractButton, QButtonGroup, QFrame,
                             QHBoxLayout, QRadioButton, QScrollArea,
                             QSizePolicy, QStackedWidget, QVBoxLayout, QWidget)

from core import roi_alignment as ra
from core.calibration import REFERENCE_NONE
from core.models import AppConfig
from core.project import (ZONE_ADJUSTED, ZONE_ALIGNED, ZONES_ADJUSTED,
                          ZONES_ALIGNED, ZONES_CUSTOM, ZONES_TEMPLATE,
                          Project, ProjectIssue, VideoEntry, canonical_json)
from core.setup_state import SetupState
from gui.background_task import TaskCancelled, run_with_progress
from gui.config_checks import probe_dimensions
from gui.qt_image import bgr_to_qpixmap
from gui.roi_editor import TOOL_LINE, RoiEditorWidget
from gui.roi_editor.session import ARENA_COLOR, CENTER_COLOR, OTHER_COLOR
from gui.setup_panels import CalibrationPanel, ParadigmParamsPanel
from gui.template_alignment import read_middle_frame
from gui.workspace.prompts import VIDEO_FILTER
from gui.workspace.stages import status as stage_status
from gui.workspace.stages.base import Stage
from gui.workspace.theme import Spacing, current_palette
from gui.workspace.thumbnails import ThumbnailLoader
from gui.workspace.ui_kit import (Callout, Card, CollapsibleSection,
                                  EmptyState, Filmstrip, FilmstripItem,
                                  GhostButton, KeyHint, Label, SecondaryButton,
                                  StatusChip, _rounded)
from mazes import AVAILABLE_MAZES
from mazes.base_maze import STEP_CALIBRATION, STEP_ROIS

logger = logging.getLogger(__name__)

MODE_TEMPLATE = "template"
MODE_VIDEO = "video"

# Filmstrip badges: (text, chip kind).
BADGE_TEMPLATE = ("Template", "neutral")
BADGE_ADJUSTED = ("Adjusted", "info")
BADGE_ALIGNED = ("Aligned", "ok")
BADGE_CHECK = ("Check alignment", "warning")
BADGE_CUSTOM = ("Custom", "neutral")

CODE_SIZE_MISMATCH = "zones_size_mismatch"
CODE_OUTDATED = "zones_outdated"
CHECK_CODES = (CODE_SIZE_MISMATCH, CODE_OUTDATED)

WRITE_DEBOUNCE_MS = 250
COLUMN_WIDTH = 340
STRIP_WIDTH = 236
RAIL_WIDTH = 52
#: Below this window width the video list starts as a slim rail, so the
#: editor keeps well over 65 % of the width.
AUTO_COLLAPSE_BELOW = 1600
STRIP_FLAG = "arena/videos_collapsed"
FRAME_CACHE_SIZE = 4

TEMPLATE_NOTE = (
    "These are the template zones. Looks off? Adjust zones for this video."
)
DRAW_TIP = (
    "Draw every zone on this frame. They become the template: all videos "
    "use them unless you adjust a single video later."
)
LOCKED_NOTE = "A run is in progress: the zones are locked until it finishes."


def plural(count: int, noun: str) -> str:
    return stage_status.plural(count, noun)


# ----------------------------------------------------------------------
# Qt-free rules
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ZoneCounts:
    """How the videos get their zones."""
    template: int = 0       # follow the template
    own: int = 0            # adjusted, aligned or a custom copy
    check: int = 0          # "Check alignment"


def issue_codes_by_key(issues: Iterable[ProjectIssue]) -> Dict[str, Set[str]]:
    """The issue codes of each video (project-wide issues have no key)."""
    found: Dict[str, Set[str]] = {}
    for issue in issues:
        if issue.key is not None:
            found.setdefault(issue.key, set()).add(issue.code)
    return found


def template_dimensions(project: Project) -> Optional[Tuple[int, int]]:
    template = project.template_settings
    dims = None if template is None else template.roi_manager.source_dimensions
    return None if not dims else (int(dims[0]), int(dims[1]))


def frame_differs(project: Project, entry: VideoEntry) -> bool:
    """A video that follows the template but has another frame size."""
    expected = template_dimensions(project)
    return (
        expected is not None and entry.frame_size is not None
        and project.zone_status(entry) == ZONES_TEMPLATE
        and tuple(entry.frame_size) != expected
    )


def needs_check(project: Project, entry: VideoEntry, codes: Set[str]) -> bool:
    """True when the video's zones should be looked at ("Check alignment")."""
    return bool(codes & set(CHECK_CODES)) or frame_differs(project, entry)


def zone_badge(project: Project, entry: VideoEntry,
               codes: Optional[Set[str]] = None) -> Tuple[str, str]:
    """``(text, chip kind)`` of a video's zones for the filmstrip."""
    if needs_check(project, entry, codes or set()):
        return BADGE_CHECK
    status = project.zone_status(entry)
    if status == ZONES_ADJUSTED:
        return BADGE_ADJUSTED
    if status == ZONES_ALIGNED:
        return BADGE_ALIGNED
    if status == ZONES_CUSTOM:
        return BADGE_CUSTOM
    return BADGE_TEMPLATE


def zone_counts(project: Project, issues: Iterable[ProjectIssue]) -> ZoneCounts:
    by_key = issue_codes_by_key(issues)
    template = own = check = 0
    for entry in project.videos:
        if project.zone_status(entry) == ZONES_TEMPLATE:
            template += 1
        else:
            own += 1
        if needs_check(project, entry, by_key.get(entry.key, set())):
            check += 1
    return ZoneCounts(template, own, check)


def template_state(project: Project) -> Optional[SetupState]:
    """The template as a ``SetupState`` (or a blank one before it exists)."""
    paradigm = project.effective_paradigm()
    template = project.template_settings
    try:
        if template is not None:
            name = None if template.paradigm_id in AVAILABLE_MAZES else paradigm
            return SetupState.from_batch_settings(template, maze_name=name)
        if paradigm not in AVAILABLE_MAZES:
            return None
        state = SetupState.new(paradigm)
    except (ValueError, KeyError, TypeError):
        logger.warning("Cannot build the setup of the template", exc_info=True)
        return None
    if project.algorithm_parameters:
        state.algorithm_parameters = copy.deepcopy(project.algorithm_parameters)
    return state


def scale_factor_for(project: Project, manager) -> Optional[float]:
    """The calibration to store with a video's own zones.

    ``None`` keeps the template's scale (the zones carry no reference line,
    or the line gives the same scale). Otherwise the template's real-world
    length is spread over this video's line.
    """
    template = project.template_settings
    if template is None or manager is None:
        return None
    if manager.reference_length_pixels <= 0:
        return None
    state = template_state(project)
    if state is None:
        return None
    state.roi_manager = manager
    scale = state.scale_factor
    if math.isclose(scale, template.scale_factor, rel_tol=1e-12, abs_tol=0.0):
        return None
    return scale


def _same_scale(left: Optional[float], right: Optional[float]) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=0.0)


def _matrix3(matrix) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape == (2, 3):
        array = np.vstack([array, [0.0, 0.0, 1.0]])
    return array


def store_alignment(project: Project, key: str, matrix,
                    dims: Tuple[int, int]) -> None:
    """Give video ``key`` the template zones moved by ``matrix`` (aligned)."""
    template = project.template_settings
    transform = _matrix3(matrix)
    moved = ra.transform_roi_manager(template.roi_manager, transform, dims)
    scale = ra.scale_factor_after_alignment(
        template.scale_factor, template.roi_manager, moved
    )
    project.set_zone_override(
        key, moved, source=ZONE_ALIGNED, transform=transform.tolist(),
        scale_factor=None if _same_scale(scale, template.scale_factor) else scale,
    )


def _maze_signature(maze) -> Tuple:
    """What the ROI editor reads from a maze: when it changes, rebuild it."""
    return (
        maze.name, tuple(maze.get_roi_definitions()),
        getattr(maze, "max_polygons_per_roi", None),
        getattr(maze, "arena_enabled", None),
        getattr(maze, "arena_center_mode", None),
    )


# ----------------------------------------------------------------------
# Frames
# ----------------------------------------------------------------------
class _FrameCache:
    """The middle frame of the last few videos (cheap to ask for again)."""

    def __init__(self, size: int = FRAME_CACHE_SIZE):
        self._size = size
        self._frames: Dict[Tuple[str, int], np.ndarray] = {}

    @staticmethod
    def _key(path: Path) -> Optional[Tuple[str, int]]:
        try:
            return str(path), Path(path).stat().st_mtime_ns
        except OSError:
            return None

    def read(self, path: Path) -> Optional[np.ndarray]:
        key = self._key(Path(path))
        if key is None:
            return None
        frame = self._frames.pop(key, None)
        if frame is None:
            frame = read_middle_frame(Path(path))
            if frame is None:
                return None
        self._frames[key] = frame
        while len(self._frames) > self._size:
            self._frames.pop(next(iter(self._frames)))
        return frame

    def clear(self) -> None:
        self._frames.clear()


# ----------------------------------------------------------------------
# Read-only zones
# ----------------------------------------------------------------------
def category_colors(categories: Iterable[str]) -> Dict[str, Tuple[int, int, int]]:
    """The colours the ROI editor gives the zones of a paradigm."""
    base = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
    colored = [c for c in categories if "center" not in c]
    colors: Dict[str, Tuple[int, int, int]] = {}
    for category in categories:
        if "center" in category:
            colors[category] = CENTER_COLOR
        else:
            colors[category] = base[colored.index(category) % len(base)]
    return colors


class ReadOnlyZonesView(QWidget):
    """A frame with zones drawn over it that cannot be edited.

    The zones are drawn at their own pixel positions, whatever size the frame
    is, so a template that does not fit a video shows exactly how it is
    misplaced.
    """

    def __init__(self, caption: str = "Template zones (read-only)",
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._pixmap = None
        self._manager = None
        self._colors: Dict[str, Tuple[int, int, int]] = {}
        self._caption = caption
        self.setMinimumSize(320, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @property
    def manager(self):
        return self._manager

    @property
    def frame_size(self) -> Optional[Tuple[int, int]]:
        if self._pixmap is None:
            return None
        return self._pixmap.width(), self._pixmap.height()

    def set_view(self, frame: np.ndarray, manager, categories: Iterable[str],
                 caption: Optional[str] = None) -> None:
        self._pixmap = bgr_to_qpixmap(frame)
        self._manager = manager
        self._colors = category_colors(list(categories))
        if caption is not None:
            self._caption = caption
        self.update()

    def _image_rect(self) -> QRectF:
        area = QRectF(self.rect()).adjusted(0, 0, 0, 0)
        if self._pixmap is None or self._pixmap.isNull():
            return area
        scale = min(area.width() / self._pixmap.width(),
                    area.height() / self._pixmap.height())
        width, height = self._pixmap.width() * scale, self._pixmap.height() * scale
        return QRectF(area.center().x() - width / 2, area.center().y() - height / 2,
                      width, height)

    def paintEvent(self, event) -> None:
        palette = current_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(palette.surface_alt))
        if self._pixmap is None:
            return
        target = self._image_rect()
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))
        scale = target.width() / self._pixmap.width()

        def to_screen(points) -> QPolygonF:
            return QPolygonF([
                QPointF(target.left() + (float(x) + 0.5) * scale,
                        target.top() + (float(y) + 0.5) * scale)
                for x, y in points
            ])

        manager = self._manager
        if manager is not None:
            painter.save()
            painter.setClipRect(target)
            shapes = [(name, roi.points, self._colors.get(name, OTHER_COLOR))
                      for name, rois in manager.rois.items() for roi in rois]
            if manager.arena is not None:
                shapes.append(("arena", manager.arena.points, ARENA_COLOR))
            if manager.center_overlay is not None:
                shapes.append(("center", manager.center_overlay, CENTER_COLOR))
            label_font = QFont(self.font())
            label_font.setPixelSize(12)
            label_font.setBold(True)
            for name, points, rgb in shapes:
                polygon = to_screen(points)
                color = QColor(*rgb)
                painter.setPen(QPen(color, 2))
                painter.setBrush(QColor(color.red(), color.green(), color.blue(), 55))
                painter.drawPolygon(polygon)
                painter.setFont(label_font)
                centre = polygon.boundingRect().center()
                text = name.replace("_", " ")
                painter.setPen(QColor(0, 0, 0, 200))
                painter.drawText(centre + QPointF(1, 1), text)
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(centre, text)
            line = manager.reference_line
            if line is not None:
                painter.setPen(QPen(QColor(0, 230, 0), 2))
                first, second = (
                    QPointF(target.left() + x * scale, target.top() + y * scale)
                    for x, y in line
                )
                painter.drawLine(first, second)
            painter.restore()
        # The caption, as a pill in the corner.
        font = QFont(self.font())
        font.setPixelSize(12)
        font.setBold(True)
        painter.setFont(font)
        width = painter.fontMetrics().horizontalAdvance(self._caption) + 20
        pill = QRectF(target.left() + 10, target.top() + 10, width, 24)
        _rounded(painter, pill, 12, QColor(0, 0, 0, 150))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, self._caption)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        self.update()


# ----------------------------------------------------------------------
# The video list: a filmstrip, or a slim rail of numbered buttons
# ----------------------------------------------------------------------
class _RailButton(QAbstractButton):
    """One video of the rail: its number, tinted by its badge."""

    def __init__(self, key: str, number: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.key = key
        self.number = number
        self.kind = "neutral"
        self.setCheckable(True)
        self.setFixedSize(36, 36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_kind(self, kind: str) -> None:
        self.kind = kind
        self.update()

    def paintEvent(self, event) -> None:
        palette = current_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        foreground, soft = palette.kind_colors(self.kind)
        rect = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        selected = self.isChecked()
        _rounded(painter, rect, 8, soft,
                 palette.primary if selected else palette.border,
                 2.0 if selected else 1.0)
        font = QFont(self.font())
        font.setPixelSize(13)
        font.setBold(selected)
        painter.setFont(font)
        painter.setPen(QColor(palette.primary if selected else foreground))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(self.number))
        if self.kind in ("warning", "error"):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(foreground))
            painter.drawEllipse(QPointF(rect.right() - 3, rect.top() + 3), 3.5, 3.5)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        self.update()


class _Rail(QScrollArea):
    """A vertical column of numbered buttons: the collapsed video list."""

    clicked_key = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._body = QWidget()
        self._column = QVBoxLayout(self._body)
        self._column.setContentsMargins(0, 0, 0, 0)
        self._column.setSpacing(Spacing.XXS)
        self._column.addStretch(1)
        self.setWidget(self._body)
        self._buttons: Dict[str, _RailButton] = {}

    def set_items(self, items: List[FilmstripItem]) -> None:
        for button in self._buttons.values():
            button.setParent(None)
            button.deleteLater()
        self._buttons = {}
        for number, item in enumerate(items, start=1):
            button = _RailButton(item.key, number)
            button.setToolTip(self._tip(item))
            button.set_kind(item.badge_kind)
            button.clicked.connect(
                lambda _checked=False, key=item.key: self.clicked_key.emit(key)
            )
            self._column.insertWidget(self._column.count() - 1, button)
            self._buttons[item.key] = button

    @staticmethod
    def _tip(item: FilmstripItem) -> str:
        return f"{item.name}\n{item.badge_text}" if item.badge_text else item.name

    def set_badge(self, item: FilmstripItem) -> None:
        button = self._buttons.get(item.key)
        if button is not None:
            button.set_kind(item.badge_kind)
            button.setToolTip(self._tip(item))

    def select(self, key: Optional[str]) -> None:
        for name, button in self._buttons.items():
            button.setChecked(name == key)
        button = self._buttons.get(key)
        if button is not None:
            self.ensureWidgetVisible(button)

    def buttons(self) -> List[_RailButton]:
        return list(self._buttons.values())


class VideoNav(QWidget):
    """The project's videos with their zone badges, collapsible to a rail."""

    video_selected = pyqtSignal(str)
    collapsed_changed = pyqtSignal(bool)

    def __init__(self, loader: Optional[ThumbnailLoader] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._collapsed = False
        self._items: List[FilmstripItem] = []
        self.strip = Filmstrip(
            Qt.Orientation.Vertical, loader=loader, title="Videos",
            show_header=False,
        )
        self.strip.video_selected.connect(self.video_selected)
        self.rail = _Rail()
        self.rail.clicked_key.connect(self._rail_clicked)
        self.title_label = Label("Videos", "subtitle")
        self.count_label = Label("", "muted")
        self.toggle_button = GhostButton("", "chevron-left")
        self.toggle_button.setFixedSize(34, 34)
        self.toggle_button.clicked.connect(lambda _checked=False: self.toggle())
        self.header = QWidget()
        head = QHBoxLayout(self.header)
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(Spacing.XS)
        head.addWidget(self.title_label)
        head.addWidget(self.count_label)
        head.addStretch(1)
        head.addWidget(self.toggle_button)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(Spacing.XS)
        column.addWidget(self.header)
        column.addWidget(self.strip, 1)
        column.addWidget(self.rail, 1)
        self._apply()

    @property
    def thumbnail_loader(self) -> ThumbnailLoader:
        return self.strip.thumbnail_loader

    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        if collapsed == self._collapsed:
            return
        self._collapsed = collapsed
        self._apply()
        self.collapsed_changed.emit(collapsed)

    def toggle(self) -> None:
        self.set_collapsed(not self._collapsed)

    def _apply(self) -> None:
        collapsed = self._collapsed
        self.strip.setVisible(not collapsed)
        self.rail.setVisible(collapsed)
        self.title_label.setVisible(not collapsed)
        self.count_label.setVisible(not collapsed)
        self.setFixedWidth(RAIL_WIDTH if collapsed else STRIP_WIDTH)
        self.toggle_button.set_icon_name(
            "chevron-right" if collapsed else "chevron-left"
        )
        tip = "Show the video list" if collapsed else "Hide the video list"
        self.toggle_button.setToolTip(tip)
        self.toggle_button.setAccessibleName(tip)
        self.header.layout().setAlignment(
            self.toggle_button,
            Qt.AlignmentFlag.AlignHCenter if collapsed else Qt.AlignmentFlag.AlignRight,
        )

    # -- content -------------------------------------------------------------
    def keys(self) -> List[str]:
        return self.strip.keys()

    def items(self) -> List[FilmstripItem]:
        return self.strip.items()

    def set_items(self, items: List[FilmstripItem]) -> None:
        self._items = list(items)
        self.strip.set_items(items)
        self.rail.set_items(items)
        self.count_label.setText(f"({len(items)})" if items else "")
        self.rail.select(self.strip.selected_key())

    def set_badge(self, key: str, text: str, kind: str, tooltip: str = "") -> None:
        self.strip.set_badge(key, text, kind)
        for item in self._items:
            if item.key == key:
                item.badge_text, item.badge_kind = text, kind
                if tooltip:
                    item.tooltip = tooltip
                self.rail.set_badge(item)

    def select(self, key: Optional[str]) -> None:
        self.strip.select(key)
        self.rail.select(key)

    def selected_key(self) -> Optional[str]:
        return self.strip.selected_key()

    def _rail_clicked(self, key: str) -> None:
        self.rail.select(key)
        self.strip.select(key)
        self.video_selected.emit(key)


class _FittedParamsPanel(ParadigmParamsPanel):
    """The paradigm options, as big as the page in use.

    The stock panel keeps every paradigm's page in one stack, so it is as
    big as the biggest page (the open field's), shown or not: a Y-maze's two
    controls would float in a tall empty box, in a column too narrow for it.
    Here only the page in use stays in the stack.
    """

    def load(self, state) -> None:
        stack = self._stack
        for page in (self._empty, *self.panels.values()):
            if stack.indexOf(page) < 0:
                stack.addWidget(page)
        super().load(state)
        in_use = self.active_panel if self.active_panel is not None else self._empty
        for page in (self._empty, *self.panels.values()):
            if page is not in_use:
                stack.removeWidget(page)
                page.hide()
        self.updateGeometry()


# ----------------------------------------------------------------------
# The stage
# ----------------------------------------------------------------------
PAGE_MESSAGE = 0
PAGE_EDITOR = 1
PAGE_READONLY = 2


class ArenaStage(Stage):
    """Stage 2: the template's zones, and the zones of single videos."""

    stage_id = stage_status.STAGE_ARENA
    title = "Arena & zones"
    subtitle = "Mark the arena and its zones once; fix a single video only if its camera moved."
    icon_name = "zones"
    CONTENT_MARGIN = Spacing.MD

    def __init__(self, loader: Optional[ThumbnailLoader] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._mode = MODE_TEMPLATE
        self._key: Optional[str] = None
        self._state: Optional[SetupState] = None
        self._synced: Optional[Tuple] = None
        self._shown: Optional[Tuple] = None
        self._editor: Optional[RoiEditorWidget] = None
        self._editor_target: Optional[Tuple[str, Optional[str]]] = None
        self._manager_shown = None           # what the read-only view shows
        self._pending: Set[str] = set()
        self._rebuild_after_write = False
        self._editor_dirty = False
        self._writing = False
        self._loading_panels = False
        self._stale = False
        self._frames = _FrameCache()
        self._info_seen: Set[str] = set()
        self._notice_override: Optional[Tuple[str, str]] = None
        self._strip_decided = False
        self._setting_strip = False
        self._build(loader)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(WRITE_DEBOUNCE_MS)
        self._timer.timeout.connect(self.flush)
        self._refresh_chrome(footer=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build(self, loader: Optional[ThumbnailLoader]) -> None:
        self.nav = VideoNav(loader)
        self.nav.video_selected.connect(self._on_video_picked)
        self.nav.collapsed_changed.connect(self._on_collapsed)
        self.nav.thumbnail_loader.loaded.connect(self._on_thumbnail)

        # -- the centre: a notice and the canvas -----------------------------
        self.notice = Callout("tip", DRAW_TIP)
        self.notice.setVisible(False)
        self.message = EmptyState("zones", "")
        self.editor_host = QWidget()
        self.editor_layout = QVBoxLayout(self.editor_host)
        self.editor_layout.setContentsMargins(0, 0, 0, 0)
        self.readonly = ReadOnlyZonesView()
        self.stack = QStackedWidget()
        self.stack.addWidget(self.message)
        self.stack.addWidget(self.editor_host)
        self.stack.addWidget(self.readonly)
        centre = QWidget()
        centre_column = QVBoxLayout(centre)
        centre_column.setContentsMargins(0, 0, 0, 0)
        centre_column.setSpacing(Spacing.XS)
        centre_column.addWidget(self.notice)
        centre_column.addWidget(self.stack, 1)

        # -- the right column ----------------------------------------------------
        self.editing_card = self._build_editing_card()
        self.calibration_card = self._build_calibration_card()
        self.params_card = self._build_params_card()
        column_body = QWidget()
        column = QVBoxLayout(column_body)
        column.setContentsMargins(0, 0, Spacing.XXS, 0)
        column.setSpacing(Spacing.SM)
        column.addWidget(self.editing_card)
        column.addWidget(self.calibration_card)
        column.addWidget(self.params_card)
        column.addStretch(1)
        self.side = QScrollArea()
        self.side.setFrameShape(QFrame.Shape.NoFrame)
        self.side.setWidgetResizable(True)
        # Only the widest option panel (open field) can outgrow the column.
        self.side.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.side.setWidget(column_body)
        self.side.setFixedWidth(COLUMN_WIDTH)

        body = QWidget()
        row = QHBoxLayout(body)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(Spacing.SM)
        row.addWidget(self.nav)
        row.addWidget(centre, 1)
        row.addWidget(self.side)
        self.set_content(body)

    def _build_editing_card(self) -> Card:
        card = Card("Editing", padding=Spacing.SM)
        self.radio_template = QRadioButton("All videos (template)")
        self.radio_video = QRadioButton("This video only")
        self.radio_template.setToolTip(
            "Draw the zones once: every video that follows the template uses them"
        )
        self.radio_video.setToolTip(
            "Check, adjust or align the zones of the video picked in the list"
        )
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.radio_template)
        self.mode_group.addButton(self.radio_video)
        self.radio_template.setChecked(True)
        self.radio_template.clicked.connect(
            lambda _checked=False: self.set_mode(MODE_TEMPLATE))
        self.radio_video.clicked.connect(
            lambda _checked=False: self.set_mode(MODE_VIDEO))
        self.scope_label = Label("", "muted", wrap=True)

        # The per-video block.
        self.video_block = QWidget()
        block = QVBoxLayout(self.video_block)
        block.setContentsMargins(0, Spacing.XXS, 0, 0)
        block.setSpacing(Spacing.XS)
        self.video_name_label = Label("", "subtitle")
        self.video_name_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.video_chip = StatusChip("neutral", "Template")
        self.video_state_label = Label("", "muted", wrap=True)
        self.adjust_button = SecondaryButton("Adjust zones for this video", "zones")
        self.adjust_button.clicked.connect(lambda _checked=False: self.adjust_video())
        self.align_button = SecondaryButton("Auto-align to template", "align")
        self.align_button.clicked.connect(lambda _checked=False: self.align_video())
        self.reset_button = GhostButton("Reset to template", "reset")
        self.reset_button.clicked.connect(lambda _checked=False: self.reset_video())
        name_row = QHBoxLayout()
        name_row.setContentsMargins(0, 0, 0, 0)
        name_row.setSpacing(Spacing.XS)
        name_row.addWidget(self.video_name_label, 1)
        name_row.addWidget(self.video_chip)
        block.addLayout(name_row)
        block.addWidget(self.video_state_label)
        block.addWidget(self.adjust_button)
        block.addWidget(self.align_button)
        block.addWidget(self.reset_button)

        # Keyboard hints.
        hints = QWidget()
        hint_row = QHBoxLayout(hints)
        hint_row.setContentsMargins(0, Spacing.XXS, 0, 0)
        hint_row.setSpacing(Spacing.XXS)
        self.key_hints: List[KeyHint] = []
        for text, meaning in (("D", "draw"), ("E", "edit"), ("L", "line"),
                              ("Ctrl+Z", "undo"), ("F", "fit")):
            hint = KeyHint(text)
            self.key_hints.append(hint)
            hint_row.addWidget(hint)
            hint_row.addWidget(Label(meaning, "caption"))
            hint_row.addSpacing(Spacing.XXS)
        hint_row.addStretch(1)
        self.hints = hints

        card.add_widget(self.radio_template)
        card.add_widget(self.radio_video)
        card.add_widget(self.scope_label)
        card.add_widget(self.video_block)
        card.add_widget(self.hints)
        return card

    def _build_calibration_card(self) -> Card:
        card = Card("Calibration", padding=Spacing.SM)
        card.setToolTip("Turn pixels into centimetres: draw a line over something "
                        "whose real length you know.")
        self.calibration = CalibrationPanel()
        self.calibration.changed.connect(self._on_calibration_changed)
        self.draw_line_button = SecondaryButton("Draw reference line", "ruler")
        self.draw_line_button.clicked.connect(
            lambda _checked=False: self.draw_reference_line())
        card.add_widget(self.draw_line_button)
        card.add_widget(self.calibration)
        return card

    def _build_params_card(self) -> Card:
        card = Card(padding=Spacing.SM)
        self.params_section = CollapsibleSection(
            "Paradigm options", expanded=False, animated=False)
        # The panels were made for a wide form: use the column's full width.
        self.params_section.body.setContentsMargins(0, Spacing.XS, 0, Spacing.XS)
        self.params = _FittedParamsPanel()
        self.params.changed.connect(self._on_params_changed)
        self.params_section.body.addWidget(self.params)
        card.add_widget(self.params_section)
        return card

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def attach(self, window) -> None:
        super().attach(window)
        window.register_close_guard(self._close_guard)
        # This stage owns "Align template to videos...".
        action = window.align_action
        action.triggered.connect(self._on_align_action)
        action.setToolTip(
            "Carry the template's zones onto the videos marked "
            "\"Check alignment\" (camera moved or another resolution)."
        )
        self._refresh_actions()

    @property
    def prompts(self):
        return self.workspace.prompts

    # -- state helpers -------------------------------------------------------
    @property
    def mode(self) -> str:
        return self._mode

    @property
    def selected_key(self) -> Optional[str]:
        """The video picked in the list."""
        return self._key

    @property
    def editor(self) -> Optional[RoiEditorWidget]:
        """The ROI editor on screen (None while a message or the read-only
        view is shown)."""
        return self._editor

    @property
    def state(self) -> Optional[SetupState]:
        return None if self._state is None else self._state.copy()

    def has_pending_writes(self) -> bool:
        return bool(self._pending)

    def _is_current(self) -> bool:
        return (
            self.workspace is None
            or self.workspace.current_stage_id == self.stage_id
        )

    def _busy(self) -> bool:
        return bool(getattr(self.workspace, "is_busy", False))

    def _issues(self) -> List[ProjectIssue]:
        project = self.project
        if project is None:
            return []
        if self.workspace is not None:
            return self.workspace.project_issues()
        return project.validate()

    def _codes(self) -> Dict[str, Set[str]]:
        return issue_codes_by_key(self._issues())

    def _usable(self) -> bool:
        project = self.project
        return (
            project is not None and bool(project.videos)
            and stage_status.paradigm_known(project)
        )

    def _entry(self, key: Optional[str]) -> Optional[VideoEntry]:
        project = self.project
        if project is None or key is None or key not in project.keys():
            return None
        return project.entry(key)

    def _signature(self) -> Tuple:
        """What the stage shows of the project: a change from outside means
        the view must be rebuilt."""
        project = self.project
        if project is None:
            return ()
        template = project.template_settings
        return (
            project.effective_paradigm(), tuple(project.keys()),
            None if template is None else _json(template.to_dict()),
            tuple(
                (v.key, None if v.zones is None else _json(v.zones.to_dict()),
                 v.settings_source)
                for v in project.videos
            ),
        )

    # ------------------------------------------------------------------
    # Stage hooks
    # ------------------------------------------------------------------
    def on_project_changed(self, project: Optional[Project]) -> None:
        self._timer.stop()
        self._pending.clear()
        self._rebuild_after_write = False
        self._editor_dirty = False
        self._state = None
        self._key = None
        self._mode = MODE_TEMPLATE
        self._info_seen.clear()
        self._notice_override = None
        self._frames.clear()
        self._discard_editor()
        self._synced = None
        self._shown = None
        if self._is_current():
            self._reload()
        else:
            self._stale = True
        self._refresh_actions()

    def on_project_modified(self) -> None:
        if self.project is None:
            return
        self._refresh_actions()
        if self._writing:
            return
        if not self._is_current():
            self._stale = True
            return
        if self._pending:
            self.flush()
        if self._signature() != self._synced:
            self._reload()
        else:
            self._refresh_chrome()

    def on_selection_changed(self, keys: List[str]) -> None:
        """The video picked in the Videos table is the one shown here."""
        known = set(self.project.keys()) if self.project is not None else set()
        picked = [k for k in keys if k in known]
        if picked:
            self._key = picked[0]

    def on_enter(self) -> None:
        self._stale = False
        self._decide_strip()
        self._pick_initial_video()
        self._request_video_info()
        self._reload()

    def on_leave(self) -> bool:
        """Edits are written, never asked about."""
        self.flush()
        return True

    def _close_guard(self) -> bool:
        self.flush()
        return True

    def shutdown(self) -> None:
        """The window is closing: write what is pending, stop the timer."""
        self.flush()
        self._timer.stop()

    # -- the video list: strip or rail -----------------------------------------
    def _decide_strip(self) -> None:
        """Start with the full list on a wide window and a slim rail on a
        narrow one (the editor must keep well over 65 % of the width), unless
        the user chose before."""
        if self._strip_decided:
            return
        self._strip_decided = True
        settings = getattr(self.workspace, "app_settings", None)
        collapsed = None
        if settings is not None:
            unset = (settings.ui_flag(STRIP_FLAG, default=True)
                     != settings.ui_flag(STRIP_FLAG, default=False))
            if not unset:
                collapsed = settings.ui_flag(STRIP_FLAG)
        if collapsed is None:
            width = self.workspace.width() if self.workspace is not None else 0
            collapsed = 0 < width < AUTO_COLLAPSE_BELOW
        self._setting_strip = True
        try:
            self.nav.set_collapsed(collapsed)
        finally:
            self._setting_strip = False

    def _on_collapsed(self, collapsed: bool) -> None:
        """The user opened or closed the list: remember it."""
        self._strip_decided = True
        settings = getattr(self.workspace, "app_settings", None)
        if settings is not None and not self._setting_strip:
            settings.set_ui_flag(STRIP_FLAG, collapsed)

    # -- footer ----------------------------------------------------------------
    def next_label(self) -> Optional[str]:
        paradigm = "" if self.project is None else self.project.effective_paradigm()
        return stage_status.detection_title(paradigm)

    def can_advance(self) -> Tuple[bool, str]:
        self.flush()        # decide on what the user just drew
        status, reason = self.status()
        if status == stage_status.STATUS_DONE:
            return True, ""
        if status == stage_status.STATUS_ATTENTION and "aligned" in reason:
            return False, (
                f"{reason}. Pick \"This video only\" and Auto-align or adjust "
                "each one, or use Tools > Align template to videos..."
            )
        return False, reason or "Draw the arena and its zones first."

    def footer_text(self) -> Tuple[str, Optional[Tuple[str, str]]]:
        project = self.project
        status, reason = self.status()
        if not self._usable():
            return reason, None
        counts = zone_counts(project, self._issues())
        if status == stage_status.STATUS_TODO or (
            status == stage_status.STATUS_ATTENTION and not counts.check
        ):
            chip = ("warning", "Needs attention") if (
                status == stage_status.STATUS_ATTENTION) else None
            return reason, chip
        parts = []
        if counts.template:
            verb = "uses" if counts.template == 1 else "use"
            parts.append(f"{plural(counts.template, 'video')} {verb} the template")
        if counts.own:
            own = "its" if counts.own == 1 else "their"
            parts.append(f"{plural(counts.own, 'video')} with {own} own zones")
        text = ", ".join(parts)
        if counts.check:
            return f"{text}. {plural(counts.check, 'video')} to check", (
                "warning", "Check alignment")
        return text, ("ok", "Zones set")

    # ------------------------------------------------------------------
    # Rebuilding the view
    # ------------------------------------------------------------------
    def _pick_initial_video(self) -> None:
        project = self.project
        if project is None or not project.videos:
            self._key = None
            return
        keys = project.keys()
        if self._key in keys:
            return
        picked = [k for k in (self.workspace.selected_keys() if self.workspace else [])
                  if k in keys]
        if picked:
            self._key = picked[0]
            return
        expected = template_dimensions(project)
        if expected is not None:
            for entry in project.videos:
                if entry.frame_size is not None and tuple(entry.frame_size) == expected:
                    self._key = entry.key
                    return
        self._key = keys[0]

    def _reload(self) -> None:
        """Re-derive everything from the project and redraw."""
        self._stale = False
        self._shown = None
        self._discard_editor()
        project = self.project
        if not self._usable():
            self._state = None
            self._synced = self._signature()
            self._fill_nav()
            self._show_message(*self._unusable_message())
            self._refresh_chrome()
            return
        paradigm = project.effective_paradigm()
        if (
            project.template_settings is not None or self._state is None
            or self._state.maze_name != paradigm
        ):
            self._state = template_state(project)
        self._pick_initial_video()
        self._synced = self._signature()
        self._fill_nav()
        self._load_params()
        self._show_current()
        self._refresh_chrome()

    def _unusable_message(self) -> Tuple[str, str, str, str]:
        """(icon, title, body, action) for a project that cannot be edited yet."""
        project = self.project
        if project is None:
            return ("zones", "No study open",
                    "Open or create a study first.", "Go to Videos")
        if not stage_status.paradigm_known(project):
            return ("maze-field", "Choose the test first",
                    "The zones depend on the test: Y-maze, plus maze, open "
                    "field or tail suspension.", "Choose the test")
        return ("video", "Add a video first",
                "Zones are drawn on a frame of your videos.", "Add videos")

    def _show_message(self, icon_name: str, title: str, body: str = "",
                      action: str = "", callback=None, second: str = "",
                      second_callback=None) -> None:
        self._discard_editor()
        self.message.set_icon_name(icon_name)
        self.message.set_texts(title, body)
        self.message.clear_actions()
        if action:
            if callback is None:
                callback = self._go_to_videos
            button = self.message.add_action(action, True)
            button.clicked.connect(lambda _checked=False, cb=callback: cb())
        if second and second_callback is not None:
            button = self.message.add_action(second, False)
            button.clicked.connect(lambda _checked=False, cb=second_callback: cb())
        self.stack.setCurrentIndex(PAGE_MESSAGE)
        self._set_notice(None)

    def _go_to_videos(self) -> None:
        if self.workspace is not None:
            self.workspace.go_to_stage(stage_status.STAGE_VIDEOS)

    def _fill_nav(self) -> None:
        project = self.project
        if project is None or not project.videos:
            self.nav.set_items([])
            return
        codes = self._codes()
        items = [self._nav_item(entry, codes) for entry in project.videos]
        self.nav.set_items(items)
        self.nav.select(self._key)

    def _nav_item(self, entry: VideoEntry, codes: Dict[str, Set[str]]) -> FilmstripItem:
        text, kind = zone_badge(self.project, entry, codes.get(entry.key, set()))
        size = ""
        if entry.frame_size is not None:
            size = f"\n{entry.frame_size[0]}x{entry.frame_size[1]} px"
        return FilmstripItem(
            entry.key, entry.path, entry.name, text, kind,
            f"{entry.name}{size}\nZones: {text.lower()}",
        )

    def _refresh_badges(self) -> None:
        project = self.project
        if project is None:
            return
        if self.nav.keys() != project.keys():
            self._fill_nav()
            return
        codes = self._codes()
        for entry in project.videos:
            item = self._nav_item(entry, codes)
            self.nav.set_badge(entry.key, item.badge_text, item.badge_kind,
                               item.tooltip)

    # -- what the centre shows -------------------------------------------------
    def _reference_entry(self) -> Optional[VideoEntry]:
        project = self.project
        if project is None or not project.videos:
            return None
        return self._entry(self._key) or project.videos[0]

    def _show_current(self, force: bool = False) -> None:
        project = self.project
        if not self._usable() or self._state is None:
            return
        entry = self._reference_entry()
        if entry is None:
            return
        status = project.zone_status(entry)
        signature = (
            self._mode, entry.key, status,
            self._state.roi_manager is None, self._state.maze_name,
        )
        if signature == self._shown and not force:
            return
        self._shown = signature
        self._notice_override = None
        if self._mode == MODE_TEMPLATE:
            self._show_template_editor(entry)
        else:
            self._show_video(entry, status)
        self._refresh_calibration()

    def _frame_of(self, entry: VideoEntry) -> Optional[np.ndarray]:
        return self._frames.read(entry.path)

    def _unreadable(self, entry: VideoEntry) -> None:
        self._show_message(
            "warning", f"{entry.name} cannot be shown",
            "A frame could not be read from this video. Check that the file "
            "is still there and is a normal video, or pick another video.",
            "Go to Videos", self._go_to_videos,
            "Try again", lambda: (self._frames.clear(), self._show_current(True)),
        )

    def _show_template_editor(self, entry: VideoEntry) -> None:
        frame = self._frame_of(entry)
        if frame is None:
            self._unreadable(entry)
            return
        state = self._state
        self._open_editor(state.build_maze(), frame, state.roi_manager,
                          (MODE_TEMPLATE, None))
        self._set_notice(None)

    def _show_video(self, entry: VideoEntry, status: str) -> None:
        project = self.project
        state = self._state
        if state.roi_manager is None or project.template_settings is None:
            self._show_message(
                "zones", "Draw the template zones first",
                "Zones of a single video start from the template. Draw the "
                "template once, then come back to adjust a video.",
                "Draw the template zones",
                lambda: self.set_mode(MODE_TEMPLATE),
            )
            return
        frame = self._frame_of(entry)
        if frame is None:
            self._unreadable(entry)
            return
        settings = project.effective_settings(entry) or project.template_settings
        maze = state.build_maze()
        if status == ZONES_TEMPLATE:
            self._discard_editor()
            self._manager_shown = settings.roi_manager
            self.readonly.set_view(
                frame, settings.roi_manager,
                [name for name, _ in maze.get_roi_definitions()],
                f"Template zones on {entry.name} (read-only)",
            )
            self.stack.setCurrentIndex(PAGE_READONLY)
            return
        self._open_editor(maze, frame, settings.roi_manager, (MODE_VIDEO, entry.key))

    def _open_editor(self, maze, frame: np.ndarray, manager,
                     target: Tuple[str, Optional[str]]) -> None:
        self._discard_editor()
        has_shapes = manager is not None and (
            bool(manager.rois) or getattr(manager, "arena", None) is not None
            or manager.reference_length_pixels > 0
        )
        editor = RoiEditorWidget(
            maze, frame, parent=self.editor_host,
            initial_roi_manager=manager if has_shapes else None,
        )
        editor.changed.connect(self._on_editor_changed)
        self.editor_layout.addWidget(editor)
        self._editor = editor
        self._editor_target = target
        self._editor_dirty = False
        self.stack.setCurrentIndex(PAGE_EDITOR)
        editor.setEnabled(not self._busy())

    def _discard_editor(self) -> None:
        editor, self._editor = self._editor, None
        self._editor_target = None
        if editor is not None:
            self.editor_layout.removeWidget(editor)
            editor.setParent(None)
            editor.deleteLater()

    # -- notices ----------------------------------------------------------------
    def _set_notice(self, notice: Optional[Tuple[str, str]]) -> None:
        if notice is None:
            self.notice.setVisible(False)
            return
        kind, text = notice
        self.notice.set_kind(kind)
        self.notice.set_text(text)
        self.notice.setVisible(True)

    def current_notice(self) -> Optional[Tuple[str, str]]:
        """``(kind, text)`` of the note above the canvas (None: no note)."""
        if self.notice.isHidden():
            return None
        return self.notice.kind(), self.notice.text()

    def _compute_notice(self) -> Optional[Tuple[str, str]]:
        project = self.project
        if project is None or not self._usable() or self._state is None:
            return None
        if self._busy():
            return "info", LOCKED_NOTE
        entry = self._reference_entry()
        if entry is None:
            return None
        if self._notice_override is not None:
            return self._notice_override
        expected = template_dimensions(project)
        size = entry.frame_size
        differs = (
            expected is not None and size is not None and tuple(size) != expected
        )
        if self._mode == MODE_TEMPLATE:
            if self._state.roi_manager is None and self.stack.currentIndex() == PAGE_EDITOR:
                return "tip", DRAW_TIP
            if differs and self.stack.currentIndex() == PAGE_EDITOR:
                return "warning", (
                    f"{entry.name} is {size[0]}x{size[1]} but the template was "
                    f"drawn on {expected[0]}x{expected[1]} frames. Drawing here "
                    "redraws the template at this size; videos of the other "
                    "size will then need aligning."
                )
            return None
        status = project.zone_status(entry)
        codes = self._codes().get(entry.key, set())
        if status == ZONES_TEMPLATE and self.stack.currentIndex() == PAGE_READONLY:
            if differs:
                return "warning", (
                    f"{entry.name} is {size[0]}x{size[1]} but the template was "
                    f"drawn on {expected[0]}x{expected[1]} frames, so these zones "
                    "are misplaced. Auto-align them to this video."
                )
            return "tip", TEMPLATE_NOTE
        if CODE_OUTDATED in codes:
            return "warning", (
                "The template zones changed after this video's zones were "
                "made. Check that they still fit, or reset them to the template."
            )
        if CODE_SIZE_MISMATCH in codes:
            return "warning", (
                f"This video's zones were drawn on a different frame size than "
                f"its {size[0]}x{size[1]} px. Auto-align or redraw them."
                if size is not None else
                "This video's zones were drawn on a different frame size."
            )
        return None

    # ------------------------------------------------------------------
    # Chrome: buttons, cards, badges, footer
    # ------------------------------------------------------------------
    def _refresh_chrome(self, footer: bool = True) -> None:
        usable = self._usable() and self._state is not None
        busy = self._busy()
        self.side.setEnabled(not busy)
        if self._editor is not None:
            self._editor.setEnabled(not busy)
        for button, mode in ((self.radio_template, MODE_TEMPLATE),
                             (self.radio_video, MODE_VIDEO)):
            button.blockSignals(True)
            button.setChecked(self._mode == mode)
            button.blockSignals(False)
            button.setEnabled(usable)
        self.radio_video.setToolTip(
            "Check, adjust or align the zones of the video picked in the list"
            if usable else "Add a video and choose the test first"
        )
        self.video_block.setVisible(self._mode == MODE_VIDEO and usable)
        self.hints.setVisible(
            self._editor is not None and not busy
        )
        if usable:
            self._refresh_badges()
            self._refresh_scope_text()
            self._refresh_video_block()
        else:
            self.scope_label.setText("")
        self._refresh_calibration_card()
        self._set_notice(self._compute_notice())
        self._refresh_actions()
        if footer:
            self.refresh_footer()

    def _refresh_scope_text(self) -> None:
        project = self.project
        counts = zone_counts(project, self._issues())
        entry = self._reference_entry()
        if entry is None:
            self.scope_label.setText("")
        elif self._mode == MODE_TEMPLATE:
            self.scope_label.setText(
                f"Applies to {plural(counts.template, 'video')} using the "
                f"template. Frame: {entry.name}."
            )
        else:
            self.scope_label.setText(f"Applies to {entry.name} only.")

    def _refresh_video_block(self) -> None:
        project = self.project
        entry = self._reference_entry()
        if entry is None:
            return
        codes = self._codes().get(entry.key, set())
        text, kind = zone_badge(project, entry, codes)
        status = project.zone_status(entry)
        self.video_name_label.setText(entry.name)
        self.video_name_label.setToolTip(str(entry.path))
        self.video_chip.set_status(kind, text)
        expected = template_dimensions(project)
        if text == BADGE_CHECK[0] and CODE_OUTDATED in codes:
            state = "The template changed after these zones were made."
        elif text == BADGE_CHECK[0] and expected and entry.frame_size:
            state = (f"Frame {entry.frame_size[0]}x{entry.frame_size[1]} px, "
                     f"template {expected[0]}x{expected[1]} px.")
        else:
            state = {
                ZONES_TEMPLATE: "Follows the template zones.",
                ZONES_ADJUSTED: "Adjusted for this video.",
                ZONES_ALIGNED: "The template, moved onto this video.",
                ZONES_CUSTOM: "Its own settings copy (older Setup tab).",
            }.get(status, "")
        self.video_state_label.setText(state)
        has_template = (
            self._state is not None and self._state.roi_manager is not None
        )
        readable = entry.path.is_file()
        why_not = ""
        if not has_template:
            why_not = "Draw the template zones first."
        elif not readable:
            why_not = "The video file could not be found."
        self.adjust_button.setEnabled(
            not why_not and status in (ZONES_TEMPLATE, ZONES_CUSTOM)
        )
        self.adjust_button.setToolTip(
            why_not or (
                "Give this video its own copy of the zones and edit it"
                if status in (ZONES_TEMPLATE, ZONES_CUSTOM)
                else "This video already has its own zones: draw on the frame."
            )
        )
        self.adjust_button.setProperty(
            "class", "primary" if status == ZONES_TEMPLATE and not why_not
            else "secondary"
        )
        self.adjust_button.style().unpolish(self.adjust_button)
        self.adjust_button.style().polish(self.adjust_button)
        self.align_button.setEnabled(not why_not)
        self.align_button.setToolTip(
            why_not or "Find where the template's zones fall in this video "
                       "by comparing its background with the template's"
        )
        has_override = entry.zones is not None
        self.reset_button.setEnabled(has_override)
        self.reset_button.setToolTip(
            "Drop this video's own zones and follow the template again"
            if has_override else
            "This video has no zones of its own; it already follows the template."
            if status == ZONES_TEMPLATE else
            "This video's zones come from a full settings copy; use the Setup "
            "options to change that."
        )

    def _refresh_actions(self) -> None:
        window = self.workspace
        project = self.project
        if window is None:
            return
        window.align_action.setEnabled(
            project is not None and project.template_settings is not None
            and bool(project.videos)
        )

    # -- calibration -----------------------------------------------------------
    def _displayed_manager(self):
        if self._editor is not None:
            return self._editor.roi_manager()
        if self.stack.currentIndex() == PAGE_READONLY:
            return self._manager_shown
        return None if self._state is None else self._state.roi_manager

    def _refresh_calibration_card(self) -> None:
        state = self._state
        maze = None if state is None else state.build_maze()
        shown = maze is not None and STEP_CALIBRATION in maze.setup_steps()
        self.calibration_card.setVisible(shown)
        editable = self._editor is not None and not self._busy()
        self.draw_line_button.setEnabled(editable and TOOL_LINE in (
            self._editor.tool_buttons if self._editor is not None else {}))
        self.draw_line_button.setToolTip(
            "Switch the editor to the reference-line tool: click the two ends "
            "of something whose length you know"
            if self.draw_line_button.isEnabled() else
            "Open a frame you can draw on first (in \"This video only\", "
            "choose Adjust zones for this video)."
            if self._mode == MODE_VIDEO and self._editor is None else
            "The reference line is drawn on an editable frame."
        )
        self.params_card.setVisible(state is not None)

    def _refresh_calibration(self) -> None:
        """Show the reference line of the displayed zones in the panel."""
        state = self._state
        if state is None:
            return
        shown = state.copy()
        manager = self._displayed_manager()
        if manager is not None:
            shown.roi_manager = manager
        self._loading_panels = True
        try:
            self.calibration.load(shown)
        finally:
            self._loading_panels = False

    def _load_params(self) -> None:
        state = self._state
        if state is None:
            return
        self._loading_panels = True
        try:
            self.params.load(state)
            self.params_section.set_summary(self._params_summary(state))
        finally:
            self._loading_panels = False

    @staticmethod
    def _params_summary(state: SetupState) -> str:
        maze = state.build_maze()
        if hasattr(maze, "min_dwell_seconds") and hasattr(maze, "count_initial_arm"):
            return f"{maze.min_dwell_seconds:g} s dwell"
        if hasattr(maze, "num_mice"):
            count = int(maze.num_mice)
            return "1 mouse" if count == 1 else f"{count} mice"
        if hasattr(maze, "arena_enabled"):
            return "arena on" if maze.arena_enabled else "no arena"
        return ""

    # ------------------------------------------------------------------
    # User actions
    # ------------------------------------------------------------------
    def set_mode(self, mode: str) -> bool:
        """"All videos (template)" or "This video only"."""
        if mode not in (MODE_TEMPLATE, MODE_VIDEO) or not self._usable():
            return False
        self.flush()
        self._mode = mode
        self._shown = None
        self._show_current()
        self._refresh_chrome()
        return True

    def pick_video(self, key: str) -> bool:
        """Pick a video in the list (what a click on it does)."""
        if self._entry(key) is None:
            return False
        self.flush()
        self._key = key
        self.nav.select(key)
        self._show_current()
        self._refresh_chrome()
        return True

    def _on_video_picked(self, key: str) -> None:
        self.pick_video(key)

    def draw_reference_line(self) -> bool:
        """Switch the editor to its line tool."""
        if self._editor is None:
            return False
        return self._editor.set_tool(TOOL_LINE)

    def adjust_video(self) -> bool:
        """Give the picked video zones of its own, starting from what it has."""
        project = self.project
        entry = self._reference_entry()
        if project is None or entry is None or self._busy() or self._state is None:
            return False
        if self._state.roi_manager is None:
            return False
        self.flush()
        settings = project.effective_settings(entry) or project.template_settings
        template_scale = project.template_settings.scale_factor
        self._writing = True
        try:
            project.set_zone_override(
                entry.key, settings.roi_manager, source=ZONE_ADJUSTED,
                scale_factor=None if _same_scale(settings.scale_factor, template_scale)
                else settings.scale_factor,
            )
        finally:
            self._writing = False
        self._synced = self._signature()
        self._mode = MODE_VIDEO
        self._show_current(force=True)
        self._refresh_chrome()
        return True

    def reset_video(self) -> bool:
        """Back to the template's zones."""
        project = self.project
        entry = self._reference_entry()
        if project is None or entry is None or self._busy():
            return False
        self.flush()
        self._writing = True
        try:
            changed = project.clear_zone_override(entry.key)
        finally:
            self._writing = False
        self._synced = self._signature()
        self._show_current(force=True)
        self._refresh_chrome()
        return changed

    # -- auto-align --------------------------------------------------------------
    def _video_dims(self, entry: VideoEntry) -> Optional[Tuple[int, int]]:
        if entry.frame_size is not None:
            return tuple(entry.frame_size)
        dims = probe_dimensions(entry.path)
        if dims is not None:
            self._record_info(entry.key, frame_size=dims)
        return dims

    def _record_info(self, key: str, frame_size=None, duration=None) -> None:
        project = self.project
        if project is None or (frame_size is None and duration is None):
            return
        was = self._writing
        self._writing = True
        try:
            project.set_video_info(key, duration_s=duration, frame_size=frame_size)
        finally:
            self._writing = was

    def _template_video(self, exclude: Iterable[str] = ()) -> Optional[Path]:
        """A video that shows the template's own view of the arena: one of the
        template's frame size, never one of the videos being aligned."""
        project = self.project
        expected = template_dimensions(project)
        skip = set(exclude)
        for entry in project.videos:
            if entry.key in skip or not entry.path.is_file():
                continue
            dims = self._video_dims(entry)
            if dims is not None and (expected is None or tuple(dims) == expected):
                return entry.path
        size = f" ({expected[0]}x{expected[1]})" if expected else ""
        chosen = self.prompts.open_file(
            f"Which video was the template drawn on?{size}",
            str(project.base_dir or Path.home()), VIDEO_FILTER,
        )
        if chosen is None:
            return None
        dims = probe_dimensions(chosen)
        if dims is None:
            self.prompts.warn("Cannot read video", f"'{chosen.name}' could not be read.")
            return None
        if expected is not None and tuple(dims) != expected:
            self.prompts.warn(
                "Wrong video size",
                f"'{chosen.name}' is {dims[0]}x{dims[1]}, but the template "
                f"was drawn on {expected[0]}x{expected[1]} frames.",
            )
            return None
        return Path(chosen)

    def _register(self, template_video: Path, entries: List[VideoEntry],
                  label: str):
        """Auto-register each entry against the template video, off the GUI
        thread. Returns ``[(key, result or exception)]``; raises
        ``TaskCancelled`` when the user cancels."""
        samples = AppConfig().video.ref_frame_samples
        paths = [(entry.key, entry.path) for entry in entries]

        def work(should_cancel):
            found = []
            for key, path in paths:
                if should_cancel():
                    raise InterruptedError()
                try:
                    found.append((key, ra.register_videos(
                        template_video, path, samples, should_cancel)))
                except InterruptedError:
                    raise
                except Exception as exc:        # one bad video must not stop the rest
                    logger.warning("Aligning %s failed", path, exc_info=True)
                    found.append((key, exc))
            return found

        return run_with_progress(self.window(), label, work, title="Align zones")

    def _apply_registration(self, entry: VideoEntry, outcome) -> Tuple[bool, str]:
        """Store an aligned override; ``(stored, reason it was not)``."""
        if isinstance(outcome, Exception):
            return False, str(outcome) or type(outcome).__name__
        if not outcome.usable:
            return False, outcome.message or "not enough of the arena matched"
        dims = self._video_dims(entry)
        if dims is None:
            return False, "the video could not be read"
        store_alignment(self.project, entry.key, outcome.matrix, dims)
        return True, ""

    def align_video(self) -> bool:
        """Auto-align the picked video to the template."""
        project = self.project
        entry = self._reference_entry()
        if (
            project is None or entry is None or self._busy()
            or self._state is None or self._state.roi_manager is None
        ):
            return False
        self.flush()
        if project.zone_status(entry) == ZONES_ADJUSTED and not self.prompts.confirm(
            "Replace the adjusted zones?",
            f"The zones you adjusted for {entry.name} will be replaced by the "
            "template moved onto the video.",
            accept="Replace", informative="You can still adjust them afterwards.",
        ):
            return False
        template_video = self._template_video(exclude=[entry.key])
        if template_video is None:
            return False
        try:
            (_key, outcome), = self._register(
                template_video, [entry], f"Aligning {entry.name} to the template...")
        except TaskCancelled:
            return False
        except Exception as exc:
            logger.warning("Alignment failed", exc_info=True)
            self.prompts.warn("Alignment failed", f"The video could not be aligned:\n{exc}")
            return False
        self._writing = True
        try:
            stored, reason = self._apply_registration(entry, outcome)
        finally:
            self._writing = False
        if not stored:
            self.prompts.warn(
                "Could not align automatically",
                f"{entry.name}: {reason}.\n\nUse \"Adjust zones for this video\" "
                "to place the zones by hand.",
            )
            return False
        self._synced = self._signature()
        self._mode = MODE_VIDEO
        self._shown = None
        self._show_current(force=True)
        verdict = getattr(outcome, "verdict", ra.VERDICT_GOOD)
        if verdict == ra.VERDICT_GOOD:
            self._notice_override = ("success", (
                "Aligned automatically. Check that the zones sit on the arena; "
                "drag them if they do not."))
        else:
            self._notice_override = ("warning", (
                "The match is uncertain. Look closely at the zones and fix "
                "them by hand where they are off."))
        self._refresh_chrome()
        return True

    def align_all(self) -> int:
        """Tools > Align template to videos...: every "Check alignment"
        video, with a summary at the end. Returns how many were aligned."""
        project = self.project
        if project is None or self._busy():
            return 0
        if project.template_settings is None or (
            self._state is not None and self._state.roi_manager is None
        ):
            self.prompts.info(
                "Align template",
                "Draw the template zones first (stage Arena & zones).",
            )
            return 0
        self.flush()
        for entry in project.videos:
            if entry.path.is_file() and entry.frame_size is None:
                self._video_dims(entry)
        codes = self._codes()
        flagged = [
            e for e in project.videos
            if e.path.is_file() and needs_check(project, e, codes.get(e.key, set()))
        ]
        adjusted = [e for e in flagged if project.zone_status(e) == ZONES_ADJUSTED]
        todo = [e for e in flagged if e not in adjusted]
        if not todo:
            text = "Every video already fits the template."
            if adjusted:
                text = (
                    f"{plural(len(adjusted), 'video')} you adjusted by hand "
                    "need a look, but they were left alone: " +
                    ", ".join(e.name for e in adjusted) + "."
                )
            self.prompts.info("Align template", text)
            return 0
        template_video = self._template_video(exclude=[e.key for e in todo])
        if template_video is None:
            return 0
        todo = [e for e in todo if e.path != template_video]
        if not todo:
            self.prompts.info("Align template", "Every video already fits the template.")
            return 0
        try:
            outcomes = self._register(
                template_video, todo,
                f"Aligning {plural(len(todo), 'video')} to the template...")
        except TaskCancelled:
            return 0
        except Exception as exc:
            logger.warning("Batch alignment failed", exc_info=True)
            self.prompts.warn("Alignment failed", f"The videos could not be aligned:\n{exc}")
            return 0
        aligned: List[str] = []
        uncertain: List[str] = []
        failed: List[Tuple[str, str]] = []
        self._writing = True
        try:
            for (key, outcome), entry in zip(outcomes, todo):
                stored, reason = self._apply_registration(entry, outcome)
                if not stored:
                    failed.append((entry.name, reason))
                    continue
                aligned.append(entry.name)
                if getattr(outcome, "verdict", ra.VERDICT_GOOD) != ra.VERDICT_GOOD:
                    uncertain.append(entry.name)
        finally:
            self._writing = False
        self._synced = self._signature()
        if self._is_current():
            self._show_current(force=True)
            self._refresh_chrome()
        else:
            self._stale = True
        self.prompts.info("Align template", self._summary(
            len(todo), aligned, uncertain, failed, adjusted))
        return len(aligned)

    @staticmethod
    def _summary(total: int, aligned: List[str], uncertain: List[str],
                 failed: List[Tuple[str, str]], adjusted: List[VideoEntry]) -> str:
        lines = [f"Aligned {len(aligned)} of {plural(total, 'video')} to the template."]
        if uncertain:
            lines.append(
                "\nThe match is uncertain, so look at the zones of:\n" +
                "\n".join(f"  - {name}" for name in uncertain))
        if failed:
            lines.append(
                "\nCould not be aligned (adjust them by hand):\n" +
                "\n".join(f"  - {name}: {why}" for name, why in failed))
        if adjusted:
            lines.append(
                "\nLeft alone because you adjusted them by hand:\n" +
                "\n".join(f"  - {e.name}" for e in adjusted))
        return "\n".join(lines)

    def _on_align_action(self, _checked: bool = False) -> None:
        self.align_all()

    # ------------------------------------------------------------------
    # Live writes
    # ------------------------------------------------------------------
    def _schedule(self, kind: str) -> None:
        self._pending.add(kind)
        self._timer.start()

    def _on_editor_changed(self) -> None:
        if self._busy() or self._editor_target is None:
            return
        mode, _key = self._editor_target
        self._editor_dirty = True
        self._schedule("template" if mode == MODE_TEMPLATE else "zones")
        self._refresh_calibration()
        self._refresh_calibration_card()

    def _on_calibration_changed(self) -> None:
        if self._loading_panels or self._state is None or self._busy():
            return
        self._state.scale_cm = self.calibration.real_world_cm()
        self._schedule("template")

    def _on_params_changed(self) -> None:
        if self._loading_panels or self._state is None or self._busy():
            return
        if self.params.issues():
            return
        before = _maze_signature(self._state.build_maze())
        if not self.params.apply(self._state):
            return
        if _maze_signature(self._state.build_maze()) != before:
            self._rebuild_after_write = True
        self.params_section.set_summary(self._params_summary(self._state))
        self._schedule("template")

    def flush(self) -> None:
        """Write every pending edit now (never asks; safe to call any time)."""
        self._timer.stop()
        pending, self._pending = self._pending, set()
        project = self.project
        if not pending or project is None or self._busy():
            return
        self._writing = True
        try:
            if "template" in pending:
                self._write_template()
            if "zones" in pending:
                self._write_zones()
        finally:
            self._writing = False
            self._editor_dirty = False
        self._synced = self._signature()
        if self._rebuild_after_write:
            self._rebuild_after_write = False
            self._shown = None
            self._show_current(force=True)
        self._refresh_chrome(footer=False)

    def _write_template(self) -> None:
        project = self.project
        state = self._state
        if state is None:
            return
        editor = self._editor
        if (
            editor is not None and self._editor_dirty
            and self._editor_target == (MODE_TEMPLATE, None)
        ):
            state.roi_manager = editor.roi_manager()
            state.mark_done(STEP_ROIS)
        if state.roi_manager is None:
            return      # nothing to store until a zone exists
        if self.calibration.status.kind != REFERENCE_NONE:
            state.scale_cm = self.calibration.real_world_cm()
        settings = state.to_batch_settings(finalize=True)
        current = project.template_settings
        if current is not None and _json(current.to_dict()) == _json(settings.to_dict()):
            return
        project.set_template(settings)
        self._rescale_overrides()

    def _rescale_overrides(self) -> None:
        """Keep each own-zones video's calibration right after the template's
        scale (or its line) changed."""
        project = self.project
        for entry in project.videos:
            zones = entry.zones
            if zones is None:
                continue
            wanted = scale_factor_for(project, zones.roi_manager)
            if _same_scale(wanted, zones.scale_factor):
                continue
            reference = zones.reference
            project.set_zone_override(
                entry.key, zones.roi_manager, source=zones.source,
                transform=zones.transform, scale_factor=wanted,
            )
            # Only the calibration moved: the override is not "re-checked".
            entry.zones.reference = reference

    def _write_zones(self) -> None:
        project = self.project
        editor = self._editor
        if editor is None or self._editor_target is None:
            return
        mode, key = self._editor_target
        if mode != MODE_VIDEO or self._entry(key) is None:
            return
        manager = editor.roi_manager()
        settings = project.effective_settings(key)
        if settings is not None and _json(settings.roi_manager.to_dict()) == _json(
            manager.to_dict()
        ):
            return
        project.set_zone_override(
            key, manager, source=ZONE_ADJUSTED,
            scale_factor=scale_factor_for(project, manager),
        )

    # ------------------------------------------------------------------
    # Video facts
    # ------------------------------------------------------------------
    def _request_video_info(self) -> None:
        """Ask for the frame size of every video that has none yet."""
        project = self.project
        if project is None:
            return
        loader = self.nav.thumbnail_loader
        for entry in project.videos:
            if entry.frame_size is not None and entry.duration_s is not None:
                continue
            if not self._take_preview(entry):
                loader.request(entry.path)

    def _take_preview(self, entry: VideoEntry) -> bool:
        """Record what the cached preview of ``entry`` knows; False if none."""
        preview = self.nav.thumbnail_loader.get(entry.path)
        if preview is None:
            return False
        size = (preview.width, preview.height) if preview.width > 0 and (
            preview.height > 0) else None
        frame_size = size if entry.frame_size is None else None
        duration = preview.duration if (
            entry.duration_s is None and preview.duration > 0) else None
        if frame_size is not None or duration is not None:
            self._record_info(entry.key, frame_size, duration)
        return True

    def _on_thumbnail(self, path: str) -> None:
        project = self.project
        if project is None:
            return
        entry = project.find(Path(path))
        if entry is None:
            return
        before = entry.frame_size
        self._take_preview(entry)
        if entry.frame_size != before and self._is_current():
            self._refresh_chrome()


def _json(data) -> str:
    return canonical_json(data)


__all__ = [
    "ArenaStage", "BADGE_ADJUSTED", "BADGE_ALIGNED", "BADGE_CHECK",
    "BADGE_CUSTOM", "BADGE_TEMPLATE", "MODE_TEMPLATE", "MODE_VIDEO",
    "ReadOnlyZonesView", "VideoNav", "ZoneCounts", "needs_check",
    "scale_factor_for", "store_alignment", "template_state", "zone_badge",
    "zone_counts",
]
