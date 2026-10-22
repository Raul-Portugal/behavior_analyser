"""Qt-free ROI drawing rules and geometry used by the ROI editor.

Everything here is pure Python / numpy so the decisions (how many polygons a
zone may hold, whether a shape is big enough, which vertex a new point snaps
to) can be tested without opening a window.
"""
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from roi_manager import ARENA_CATEGORY

Point = Tuple[float, float]

# Smallest polygon the pipeline accepts; ROIManager.validate_rois reports
# anything below this as an issue that blocks the run, so refuse it here.
MIN_ROI_AREA_PX = 100.0
MIN_POLYGON_POINTS = 3

# The open-field arena: exactly one polygon, stored as ``ROIManager.arena``
# rather than as a zone.
ARENA_STEP_DESCRIPTION = "the open-field arena (the floor boundary)"

# New points and dragged vertices snap to an existing vertex within this many
# SCREEN pixels, so adjacent zones can share an exact border.
SNAP_RADIUS_SCREEN_PX = 6.0


@dataclass(frozen=True)
class RoiRule:
    """How many polygons one ROI category may hold."""
    minimum: int = 1
    maximum: Optional[int] = None


@dataclass(frozen=True)
class RoiStepState:
    """What the ROI editor allows for one zone right now, and why."""
    can_advance: bool
    can_add_point: bool
    hint: str


ARENA_RULE = RoiRule(1, 1)


def roi_rule_for_maze(maze) -> RoiRule:
    """TST mice hold exactly one polygon each; every other category 1 or more.

    The limit comes from the maze (``Maze.max_polygons_per_roi``).
    """
    return RoiRule(1, getattr(maze, 'max_polygons_per_roi', None))


def evaluate_roi_step(rule: RoiRule, finished: int, pending_points: int,
                      description: str) -> RoiStepState:
    """Decide whether more points / Next are allowed and give the hint.

    ``finished`` is the number of closed polygons in this category and
    ``pending_points`` the points of the shape still being drawn (a pending
    shape of three or more points is closed automatically when continuing).
    """
    at_capacity = rule.maximum is not None and finished >= rule.maximum
    exact = rule.maximum is not None and rule.maximum == rule.minimum
    need = (
        f"exactly {rule.minimum} polygon" + ("" if rule.minimum == 1 else "s")
        if exact else f"at least {rule.minimum} polygon"
        + ("" if rule.minimum == 1 else "s")
    )

    if 0 < pending_points < MIN_POLYGON_POINTS:
        missing = MIN_POLYGON_POINTS - pending_points
        return RoiStepState(
            False, True,
            f"Add {missing} more point{'s' if missing != 1 else ''}, or press "
            "Esc to cancel the shape."
        )
    if pending_points >= MIN_POLYGON_POINTS and finished + 1 >= rule.minimum:
        return RoiStepState(
            True, True,
            "Press Enter or double-click to close the shape."
        )
    if finished < rule.minimum:
        return RoiStepState(
            False, True,
            f"{description} needs {need}: click to add points, "
            "Enter or double-click to close the shape."
        )
    if at_capacity:
        return RoiStepState(
            True, False,
            f"{description} is done ({need}). Edit it, or delete it to "
            "undo and redraw it."
        )
    return RoiStepState(
        True, True,
        f"{description} has {finished} polygon{'s' if finished != 1 else ''}. "
        "Draw another, or pick another zone."
    )


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------
def polygon_area(points: Sequence[Point]) -> float:
    """Absolute shoelace area of a polygon (px squared)."""
    if len(points) < 3:
        return 0.0
    x = np.array([p[0] for p in points], dtype=np.float64)
    y = np.array([p[1] for p in points], dtype=np.float64)
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polygon_centroid(points: Sequence[Point]) -> Point:
    """Vertex average; stable even for tiny or degenerate polygons."""
    n = max(len(points), 1)
    return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n)


def snap_point(point: Point, candidates: Iterable[Point],
               radius: float) -> Point:
    """The nearest candidate within ``radius`` of ``point``, else ``point``."""
    best = None
    best_distance = radius
    for candidate in candidates:
        distance = math.hypot(candidate[0] - point[0], candidate[1] - point[1])
        if distance <= best_distance:
            best = candidate
            best_distance = distance
    return (float(best[0]), float(best[1])) if best is not None else point


def distance_to_segment(point: Point, a: Point, b: Point) -> Tuple[float, Point]:
    """(distance, closest point) from ``point`` to the segment a-b."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(point[0] - ax, point[1] - ay), (ax, ay)
    t = ((point[0] - ax) * dx + (point[1] - ay) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    closest = (ax + t * dx, ay + t * dy)
    return math.hypot(point[0] - closest[0], point[1] - closest[1]), closest


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Even-odd point-in-polygon test."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            crossing = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing:
                inside = not inside
    return inside


def rotate_about(points: Sequence[Point], centre: Point,
                 radians: float) -> List[Point]:
    cos, sin = math.cos(radians), math.sin(radians)
    return [
        (centre[0] + (x - centre[0]) * cos - (y - centre[1]) * sin,
         centre[1] + (x - centre[0]) * sin + (y - centre[1]) * cos)
        for x, y in points
    ]


def scale_about(points: Sequence[Point], centre: Point,
                factor: float) -> List[Point]:
    return [
        (centre[0] + (x - centre[0]) * factor,
         centre[1] + (y - centre[1]) * factor)
        for x, y in points
    ]


# ----------------------------------------------------------------------
# Zone rules
# ----------------------------------------------------------------------
STATUS_DONE = "done"
STATUS_MISSING = "missing"
STATUS_VIOLATED = "violated"


@dataclass(frozen=True)
class ZoneStatus:
    """The state of one zone row: status plus the issues behind it."""
    category: str
    status: str
    count: int
    issues: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == STATUS_DONE


def evaluate_zone(category: str, description: str, rule: RoiRule,
                  areas: Sequence[float]) -> ZoneStatus:
    """Judge a category's polygons (given as their areas) against its rule."""
    count = len(areas)
    label = description[0].upper() + description[1:] if description else category
    if category == ARENA_CATEGORY:
        label = "Arena"
    issues: List[str] = []
    if count < rule.minimum:
        plural = "" if rule.minimum == 1 else "s"
        issues.append(f"{label}: needs {rule.minimum} polygon{plural}, has {count}")
        return ZoneStatus(category, STATUS_MISSING, count, tuple(issues))
    if rule.maximum is not None and count > rule.maximum:
        plural = "" if rule.maximum == 1 else "s"
        issues.append(
            f"{label}: allows {'exactly ' if rule.maximum == rule.minimum else 'at most '}"
            f"{rule.maximum} polygon{plural}, has {count}"
        )
    for index, area in enumerate(areas, start=1):
        if area < MIN_ROI_AREA_PX:
            suffix = f" #{index}" if count > 1 else ""
            issues.append(
                f"{label}{suffix}: too small ({area:.0f} px²; at least "
                f"{MIN_ROI_AREA_PX:.0f} px² is needed)"
            )
    if issues:
        return ZoneStatus(category, STATUS_VIOLATED, count, tuple(issues))
    return ZoneStatus(category, STATUS_DONE, count)
