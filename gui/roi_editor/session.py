"""The ROI editor's editing logic, free of any widget.

``EditSession`` owns the document, the undo stack and the interaction state
(tool, current zone, polygon being drawn, selection, drags). The canvas turns
raw mouse / key events into calls on it, in SOURCE pixel coordinates; the
widgets only display what it reports. That keeps every editing rule testable
without a window.

Input positions are "model" coordinates: continuous source pixels where
integer values are pixel centres (the canvas subtracts the half pixel between
scene and model coordinates).
"""
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QUndoStack

from roi_manager import ARENA_CATEGORY, ROIManager

from .model import (AddShapeCommand, DeleteShapeCommand, Line, RoiDocument,
                    SetLineCommand, SetPointsCommand, Shape, clamp_point,
                    quantize)
from .rules import (ARENA_RULE, ARENA_STEP_DESCRIPTION, MIN_POLYGON_POINTS,
                    MIN_ROI_AREA_PX, SNAP_RADIUS_SCREEN_PX, STATUS_DONE,
                    STATUS_MISSING, Point, RoiRule, ZoneStatus,
                    distance_to_segment, evaluate_roi_step, evaluate_zone,
                    point_in_polygon, polygon_area, roi_rule_for_maze,
                    rotate_about, scale_about, snap_point)

TOOL_DRAW = "draw"
TOOL_EDIT = "edit"
TOOL_LINE = "line"
TOOLS = (TOOL_DRAW, TOOL_EDIT, TOOL_LINE)

KIND_ARENA = "arena"
KIND_ZONE = "zone"
KIND_EXTRA = "extra"

# Screen-pixel tolerances for picking things with the mouse.
VERTEX_HIT_PX = 9.0
EDGE_HIT_PX = 6.0
HANDLE_HIT_PX = 10.0
LINE_DRAG_PX = 4.0
ROTATE_HANDLE_OFFSET_PX = 30.0
SCALE_HANDLE_OFFSET_PX = 14.0

_BASE_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
ARENA_COLOR = (255, 165, 0)
CENTER_COLOR = (255, 255, 0)
OTHER_COLOR = (200, 200, 200)

def to_line_coords(point: Point) -> Point:
    """Model coordinates to the reference line's continuous source coordinates.

    Polygon vertices designate pixel centres; the line keeps the raw position
    under the cursor (pixel ``i`` spans ``[i, i + 1)``), as the old drawing
    dialog did.
    """
    return (point[0] + 0.5, point[1] + 0.5)


OverlayProvider = Callable[[np.ndarray], Optional[np.ndarray]]


@dataclass(frozen=True)
class ZoneRow:
    """One row of the zone panel."""
    category: str
    description: str
    rule: RoiRule
    kind: str


@dataclass
class _Drag:
    kind: str                      # vertex | move | rotate | scale | line-end | line-new
    start: Point
    shape_id: Optional[int] = None
    vertex: Optional[int] = None
    original: Optional[List[Tuple[int, int]]] = None
    original_line: Optional[Line] = None
    end: int = 0
    moved: bool = False


class EditSession(QObject):
    """Document + undo stack + tools: everything except drawing pixels."""

    # Any document edit (also undo / redo and live drag updates).
    changed = pyqtSignal()
    # Tool, current zone, draft, selection or hint changed (UI only).
    state_changed = pyqtSignal()
    # (valid, issues) whenever the validity result changes.
    validity_changed = pyqtSignal(bool, list)

    def __init__(self, maze, size: Tuple[int, int],
                 overlay_provider: Optional[OverlayProvider] = None,
                 parent=None):
        super().__init__(parent)
        self.maze = maze
        self.size = (int(size[0]), int(size[1]))
        self.document = RoiDocument(self)
        self.undo_stack = QUndoStack(self)
        self.overlay_provider = overlay_provider
        self.overlay: Optional[np.ndarray] = None
        self._overlay_key = None

        self.zone_definitions: List[Tuple[str, str]] = list(
            getattr(maze, "get_roi_definitions", lambda: [])()
        )
        self.arena_enabled = bool(getattr(maze, "arena_enabled", False))
        self.zone_rule = roi_rule_for_maze(maze) if maze is not None else RoiRule(1, None)
        needs_line = getattr(maze, "needs_reference_line", None)
        self.line_enabled = True if not callable(needs_line) else bool(needs_line())
        if maze is None:
            self.line_enabled = True

        self.tool = TOOL_DRAW
        self.current: Optional[str] = None
        self.draft: List[Tuple[int, int]] = []
        self.draft_redo: List[Tuple[int, int]] = []
        self.selected: Optional[int] = None
        self.selected_vertex: Optional[int] = None
        self.line_start: Optional[Point] = None
        self.hover: Optional[Point] = None
        self.snap_target: Optional[Point] = None
        self.view_scale = 1.0
        self.snapping = True
        self._message = ""
        self._drag: Optional[_Drag] = None
        self._last_validity: Optional[Tuple[bool, Tuple[str, ...]]] = None

        self.document.changed.connect(self._on_document_changed)
        self.undo_stack.indexChanged.connect(self._on_stack_moved)
        self.reset_view_state()
        self._emit_validity()

    # ------------------------------------------------------------------
    # Zone rows and status
    # ------------------------------------------------------------------
    def rows(self) -> List[ZoneRow]:
        rows: List[ZoneRow] = []
        has_arena_shape = bool(self.document.shapes_in(ARENA_CATEGORY))
        if self.arena_enabled:
            rows.append(ZoneRow(ARENA_CATEGORY, ARENA_STEP_DESCRIPTION, ARENA_RULE,
                                KIND_ARENA))
        elif has_arena_shape:
            rows.append(ZoneRow(ARENA_CATEGORY, "the arena (from the loaded ROIs)",
                                RoiRule(0, 1), KIND_ARENA))
        defined = {category for category, _ in self.zone_definitions}
        for category, description in self.zone_definitions:
            rows.append(ZoneRow(category, description, self.zone_rule, KIND_ZONE))
        for category in self.document.categories():
            if category != ARENA_CATEGORY and category not in defined:
                rows.append(ZoneRow(category, category, RoiRule(0, None), KIND_EXTRA))
        return rows

    def row(self, category: Optional[str]) -> Optional[ZoneRow]:
        for row in self.rows():
            if row.category == category:
                return row
        return None

    def rule_for(self, category: str) -> RoiRule:
        row = self.row(category)
        return row.rule if row is not None else RoiRule(0, None)

    def description_for(self, category: str) -> str:
        row = self.row(category)
        return row.description if row is not None else category

    def zone_status(self, category: str) -> ZoneStatus:
        row = self.row(category)
        rule = row.rule if row is not None else RoiRule(0, None)
        description = row.description if row is not None else category
        areas = [polygon_area(s.points) for s in self.document.shapes_in(category)]
        return evaluate_zone(category, description, rule, areas)

    def statuses(self) -> List[ZoneStatus]:
        return [self.zone_status(row.category) for row in self.rows()]

    def issues(self) -> List[str]:
        return [issue for status in self.statuses() for issue in status.issues]

    def is_valid(self) -> bool:
        return not self.issues()

    def category_color(self, category: str) -> Tuple[int, int, int]:
        if category == ARENA_CATEGORY:
            return ARENA_COLOR
        if "center" in category:
            return CENTER_COLOR
        colored = [c for c, _ in self.zone_definitions if "center" not in c]
        if category in colored:
            return _BASE_COLORS[colored.index(category) % len(_BASE_COLORS)]
        return OTHER_COLOR

    def can_add_shape(self, category: Optional[str]) -> bool:
        if category is None:
            return False
        rule = self.rule_for(category)
        return rule.maximum is None or len(self.document.shapes_in(category)) < rule.maximum

    def next_row_needing_attention(self, after: Optional[str] = None) -> Optional[str]:
        """The next zone (cycling) that is not done, or None if all are."""
        rows = self.rows()
        if not rows:
            return None
        start = 0
        for index, row in enumerate(rows):
            if row.category == after:
                start = index + 1
                break
        for offset in range(len(rows)):
            row = rows[(start + offset) % len(rows)]
            if row.category != after and not self.zone_status(row.category).ok:
                return row.category
        return None

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def reset_view_state(self) -> None:
        """Pick the starting zone and tool after (re)loading content."""
        self.draft = []
        self.draft_redo = []
        self.selected = None
        self.selected_vertex = None
        self.line_start = None
        self._drag = None
        self._message = ""
        rows = self.rows()
        first = self.next_row_needing_attention()
        self.current = first if first is not None else (rows[0].category if rows else None)
        needs_drawing = self.current is not None and (
            not self.zone_status(self.current).ok
        )
        self.tool = TOOL_DRAW if (needs_drawing or not self.document.shapes) else TOOL_EDIT
        if self.current is None:
            self.tool = TOOL_LINE if self.line_enabled else TOOL_EDIT

    def _notify(self, message: str = "") -> None:
        self._message = message
        self.state_changed.emit()

    def message(self) -> str:
        return self._message

    def show_message(self, text: str) -> None:
        """Show a transient hint (cleared by the next input)."""
        self._notify(text)

    def hint(self) -> str:
        """What the user can do right now (or why the last action was refused)."""
        if self._message:
            return self._message
        if self.tool == TOOL_LINE:
            if self.line_start is not None:
                return "Click the second end of the line (Esc cancels)."
            return ("Drag, or click twice, to draw a line over something of known "
                    "length. Drag an end to adjust it; right-click removes it.")
        if self.tool == TOOL_EDIT:
            return ("Drag a vertex or a polygon; double-click an edge to add a "
                    "vertex; Delete or right-click removes the selection.")
        if self.current is None:
            return "Nothing to draw."
        row = self.row(self.current)
        rule = row.rule if row is not None else RoiRule(0, None)
        description = row.description if row is not None else self.current
        return evaluate_roi_step(
            rule, len(self.document.shapes_in(self.current)), len(self.draft),
            description
        ).hint

    def has_unsaved_work(self) -> bool:
        return (not self.undo_stack.isClean()) or bool(self.draft)

    def mark_clean(self) -> None:
        self.undo_stack.setClean()

    # ------------------------------------------------------------------
    # Tools and current zone
    # ------------------------------------------------------------------
    def _settle_draft(self) -> bool:
        """Close a pending shape before changing tool / zone; False if stuck."""
        if not self.draft:
            return True
        if len(self.draft) >= MIN_POLYGON_POINTS and self.close_draft():
            return True
        self._notify(
            "Finish the shape in progress first (Enter closes it, Esc cancels it)."
        )
        return False

    def set_tool(self, tool: str) -> bool:
        if tool not in TOOLS or (tool == TOOL_LINE and not self.line_enabled):
            return False
        if tool != self.tool and not self._settle_draft():
            return False
        self.line_start = None
        self.tool = tool
        self._notify()
        return True

    def activate_zone(self, category: str) -> bool:
        """Make a zone current: draw it if it can take more, else edit it."""
        if self.row(category) is None:
            return False
        if category != self.current and not self._settle_draft():
            return False
        self.current = category
        self.line_start = None
        self.selected = None
        self.selected_vertex = None
        self.tool = TOOL_DRAW if self.can_add_shape(category) and (
            not self.zone_status(category).ok
            or not self.document.shapes_in(category)
        ) else TOOL_EDIT
        self._notify()
        return True

    # ------------------------------------------------------------------
    # Snapping and conversions
    # ------------------------------------------------------------------
    def tolerance(self, screen_px: float) -> float:
        return screen_px / max(self.view_scale, 1e-6)

    def snap_radius(self) -> float:
        return self.tolerance(SNAP_RADIUS_SCREEN_PX)

    def snap(self, point: Point, exclude: Optional[Tuple[int, int]] = None,
             enabled: bool = True) -> Point:
        """Pull ``point`` onto an existing vertex within the snap radius."""
        self.snap_target = None
        point = clamp_point(point, self.size)
        if not (self.snapping and enabled):
            return point
        snapped = snap_point(point, self.document.all_vertices(exclude),
                             self.snap_radius())
        if snapped != point:
            self.snap_target = snapped
        return snapped

    # ------------------------------------------------------------------
    # Drawing a new polygon
    # ------------------------------------------------------------------
    def add_draft_point(self, point: Point, snap: bool = True) -> bool:
        """Add a point to the polygon being drawn (draw tool)."""
        if self.current is None:
            return False
        if not self.draft and not self.can_add_shape(self.current):
            self._notify(
                f"{self.description_for(self.current)} is complete. Edit it, or "
                "delete it (Ctrl+Z undoes) to draw it again."
            )
            return False
        snapped = quantize(self.snap(point, enabled=snap))
        if self.draft and self.draft[-1] == snapped:
            return False
        self.draft.append(snapped)
        self.draft_redo = []
        self.selected = None
        self.selected_vertex = None
        self._notify()
        return True

    def cancel_draft(self) -> bool:
        had = bool(self.draft) or self.line_start is not None
        self.draft = []
        self.draft_redo = []
        self.line_start = None
        self._notify()
        return had

    def close_draft(self) -> bool:
        """Turn the pending points into a polygon; False if refused."""
        category = self.current
        if category is None:
            return False
        if 0 < len(self.draft) < MIN_POLYGON_POINTS:
            self._notify(
                f"{self.description_for(category)} needs at least three points."
            )
            return False
        if len(self.draft) < MIN_POLYGON_POINTS:
            return False
        points = list(self.draft)
        if points[-1] == points[0]:
            points.pop()
        deduped = [p for i, p in enumerate(points) if i == 0 or p != points[i - 1]]
        if not self.can_add_shape(category):
            self._notify(
                f"{self.description_for(category)} is complete. Edit it, or delete "
                "it (Ctrl+Z undoes) to draw it again."
            )
            return False
        area = polygon_area(deduped)
        if len(deduped) < MIN_POLYGON_POINTS or area < MIN_ROI_AREA_PX:
            self._notify(
                f"That shape is too small ({area:.0f} px²; at least "
                f"{MIN_ROI_AREA_PX:.0f} px² is needed). Cancel it (Esc) and "
                "draw it larger, or undo points."
            )
            return False
        shape = Shape(self.document.new_id(), category, deduped)
        self.draft = []
        self.draft_redo = []
        self.undo_stack.push(AddShapeCommand(
            self.document, shape, f"Add {self.description_for(category)}"
        ))
        self.selected = shape.id
        self.selected_vertex = None
        # A zone that cannot take another polygon moves the user along.
        if not self.can_add_shape(category):
            following = self.next_row_needing_attention(category)
            if following is not None:
                self.current = following
                self.tool = TOOL_DRAW
                self.selected = None
        self._notify()
        return True

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------
    def select(self, shape_id: Optional[int], vertex: Optional[int] = None) -> None:
        self.selected = shape_id
        self.selected_vertex = vertex
        shape = self.document.shape(shape_id)
        if shape is not None and self.row(shape.category) is not None:
            self.current = shape.category
        self._notify()

    def delete_shape(self, shape_id: int) -> bool:
        shape = self.document.shape(shape_id)
        if shape is None:
            return False
        self.undo_stack.push(DeleteShapeCommand(
            self.document, shape_id, f"Delete {self.description_for(shape.category)}"
        ))
        if self.selected == shape_id:
            self.selected = None
            self.selected_vertex = None
        self._notify()
        return True

    def delete_vertex(self, shape_id: int, index: int) -> bool:
        shape = self.document.shape(shape_id)
        if shape is None or not 0 <= index < len(shape.points):
            return False
        if len(shape.points) <= MIN_POLYGON_POINTS:
            self._notify(
                "A polygon needs at least three vertices; delete the whole "
                "polygon instead."
            )
            return False
        new_points = shape.points[:index] + shape.points[index + 1:]
        self.undo_stack.push(SetPointsCommand(
            self.document, shape_id, shape.points, new_points, "Delete vertex"
        ))
        self.selected_vertex = None
        self._notify()
        return True

    def insert_vertex(self, shape_id: int, after_index: int, point: Point) -> bool:
        shape = self.document.shape(shape_id)
        if shape is None:
            return False
        new_points = (shape.points[:after_index + 1]
                      + [quantize(clamp_point(point, self.size))]
                      + shape.points[after_index + 1:])
        self.undo_stack.push(SetPointsCommand(
            self.document, shape_id, shape.points, new_points, "Insert vertex"
        ))
        self.selected = shape_id
        self.selected_vertex = after_index + 1
        self._notify()
        return True

    def delete_selection(self) -> bool:
        if self.tool == TOOL_LINE:
            return self.delete_line()
        if self.selected is None:
            return False
        if self.selected_vertex is not None:
            shape = self.document.shape(self.selected)
            if shape is not None and len(shape.points) > MIN_POLYGON_POINTS:
                return self.delete_vertex(self.selected, self.selected_vertex)
        return self.delete_shape(self.selected)

    def clear_zone(self, category: str) -> int:
        """Delete every polygon of a zone as one undo step; returns the count."""
        shapes = self.document.shapes_in(category)
        if not shapes:
            return 0
        self.undo_stack.beginMacro(f"Clear {self.description_for(category)}")
        for shape in shapes:
            self.undo_stack.push(DeleteShapeCommand(self.document, shape.id))
        self.undo_stack.endMacro()
        self.selected = None
        self.selected_vertex = None
        self._notify()
        return len(shapes)

    def nudge(self, dx: int, dy: int) -> bool:
        shape = self.document.shape(self.selected)
        if shape is None:
            return False
        if self.selected_vertex is not None and self.selected_vertex < len(shape.points):
            moved = list(shape.points)
            x, y = moved[self.selected_vertex]
            moved[self.selected_vertex] = quantize(clamp_point((x + dx, y + dy), self.size))
            text = "Move vertex"
        else:
            dx, dy = self._clamp_translation(shape.points, dx, dy)
            moved = [(x + dx, y + dy) for x, y in shape.points]
            text = "Move polygon"
        if moved == shape.points:
            return False
        self.undo_stack.push(SetPointsCommand(
            self.document, shape.id, shape.points, moved, text
        ))
        return True

    def _clamp_translation(self, points, dx, dy) -> Tuple[int, int]:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        dx = max(-min(xs), min(self.size[0] - 1 - max(xs), dx))
        dy = max(-min(ys), min(self.size[1] - 1 - max(ys), dy))
        return int(dx), int(dy)

    # -- reference line -------------------------------------------------
    def set_line(self, line: Optional[Line], text: str = "Edit reference line") -> bool:
        if line is not None and tuple(line[0]) == tuple(line[1]):
            self._notify("The two ends of the reference line must differ.")
            return False
        if line is not None:
            line = tuple(
                (round(min(max(float(x), 0.0), float(self.size[0])), 2),
                 round(min(max(float(y), 0.0), float(self.size[1])), 2))
                for x, y in line
            )
        if line == self.document.line:
            return False
        self.undo_stack.push(SetLineCommand(self.document, self.document.line, line, text))
        self.line_start = None
        self._notify()
        return True

    def delete_line(self) -> bool:
        if self.document.line is None:
            return False
        return self.set_line(None, "Delete reference line")

    def line_length(self) -> float:
        line = self.document.line
        if line is None:
            return 0.0
        return float(math.hypot(line[1][0] - line[0][0], line[1][1] - line[0][1]))

    # ------------------------------------------------------------------
    # Undo / redo
    # ------------------------------------------------------------------
    def undo(self) -> bool:
        if self.draft:
            self.draft_redo.append(self.draft.pop())
            self._notify()
            return True
        if self.line_start is not None:
            self.line_start = None
            self._notify()
            return True
        if self.undo_stack.canUndo():
            self.undo_stack.undo()
            return True
        return False

    def redo(self) -> bool:
        if self.draft_redo and self.tool == TOOL_DRAW:
            self.draft.append(self.draft_redo.pop())
            self._notify()
            return True
        if self.undo_stack.canRedo():
            self.undo_stack.redo()
            return True
        return False

    def _on_stack_moved(self, _index: int) -> None:
        if self.selected is not None and self.document.shape(self.selected) is None:
            self.selected = None
            self.selected_vertex = None
        elif self.selected is not None:
            shape = self.document.shape(self.selected)
            if self.selected_vertex is not None and self.selected_vertex >= len(shape.points):
                self.selected_vertex = None
        self._message = ""
        self.state_changed.emit()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self, manager: Optional[ROIManager]) -> None:
        self.undo_stack.clear()
        self.document.load(manager)
        self.undo_stack.setClean()
        self.reset_view_state()
        self.state_changed.emit()

    def build_manager(self) -> ROIManager:
        manager = self.document.build_manager(
            self.size, [category for category, _ in self.zone_definitions]
        )
        overlay = self.overlay
        if overlay is not None and manager.arena is not None:
            try:
                manager.set_center_overlay(overlay)
            except ValueError:
                pass
        return manager

    # ------------------------------------------------------------------
    # Document change reactions
    # ------------------------------------------------------------------
    def _on_document_changed(self) -> None:
        if self._drag is None:
            self._update_overlay()
        self.changed.emit()
        self._emit_validity()

    def _emit_validity(self) -> None:
        issues = self.issues()
        key = (not issues, tuple(issues))
        if key == self._last_validity:
            return
        self._last_validity = key
        self.validity_changed.emit(not issues, list(issues))

    def _update_overlay(self) -> None:
        arenas = self.document.shapes_in(ARENA_CATEGORY)
        key = tuple(arenas[0].points) if arenas else None
        if key == self._overlay_key:
            return
        self._overlay_key = key
        self.overlay = None
        if key is None or self.overlay_provider is None or len(key) < MIN_POLYGON_POINTS:
            return
        try:
            self.overlay = self.overlay_provider(np.array(key, dtype=np.int32))
        except (ValueError, TypeError):
            self.overlay = None

    # ------------------------------------------------------------------
    # Hit testing
    # ------------------------------------------------------------------
    def _ordered_shapes(self) -> List[Shape]:
        """Selected shape first, then topmost (last drawn) to bottom."""
        shapes = list(reversed(self.document.shapes))
        selected = self.document.shape(self.selected)
        if selected is not None:
            shapes.remove(selected)
            shapes.insert(0, selected)
        return shapes

    def hit_vertex(self, point: Point) -> Optional[Tuple[int, int]]:
        tol = self.tolerance(VERTEX_HIT_PX)
        for shape in self._ordered_shapes():
            best, best_distance = None, tol
            for index, vertex in enumerate(shape.points):
                distance = math.hypot(vertex[0] - point[0], vertex[1] - point[1])
                if distance <= best_distance:
                    best, best_distance = index, distance
            if best is not None:
                return shape.id, best
        return None

    def hit_body(self, point: Point) -> Optional[int]:
        containing = [
            shape for shape in self.document.shapes
            if point_in_polygon(point, shape.points)
        ]
        if not containing:
            return None
        return min(containing, key=lambda s: polygon_area(s.points)).id

    def hit_edge(self, point: Point) -> Optional[Tuple[int, int, Point]]:
        tol = self.tolerance(EDGE_HIT_PX)
        for shape in self._ordered_shapes():
            best = None
            best_distance = tol
            count = len(shape.points)
            for index in range(count):
                a = shape.points[index]
                b = shape.points[(index + 1) % count]
                distance, closest = distance_to_segment(point, a, b)
                if distance <= best_distance:
                    best, best_distance = (index, closest), distance
            if best is not None:
                return shape.id, best[0], best[1]
        return None

    def transform_handles(self, shape: Shape) -> Dict[str, Point]:
        """Rotate / scale handle positions for a polygon (model coordinates)."""
        xs = [p[0] for p in shape.points]
        ys = [p[1] for p in shape.points]
        centre = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
        return {
            "rotate": (centre[0], min(ys) - self.tolerance(ROTATE_HANDLE_OFFSET_PX)),
            "scale": (max(xs) + self.tolerance(SCALE_HANDLE_OFFSET_PX),
                      max(ys) + self.tolerance(SCALE_HANDLE_OFFSET_PX)),
        }

    @staticmethod
    def _bbox_centre(points) -> Point:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)

    def hit_handle(self, point: Point) -> Optional[str]:
        shape = self.document.shape(self.selected)
        if shape is None or self.tool != TOOL_EDIT:
            return None
        tol = self.tolerance(HANDLE_HIT_PX)
        for name, position in self.transform_handles(shape).items():
            if math.hypot(position[0] - point[0], position[1] - point[1]) <= tol:
                return name
        return None

    def hit_line_end(self, point: Point) -> Optional[int]:
        line = self.document.line
        if line is None:
            return None
        tol = self.tolerance(VERTEX_HIT_PX)
        best, best_distance = None, tol
        scene = to_line_coords(point)
        for index in (0, 1):
            distance = math.hypot(line[index][0] - scene[0], line[index][1] - scene[1])
            if distance <= best_distance:
                best, best_distance = index, distance
        return best

    def hit_line(self, point: Point) -> bool:
        line = self.document.line
        if line is None:
            return False
        distance, _ = distance_to_segment(to_line_coords(point), line[0], line[1])
        return distance <= self.tolerance(EDGE_HIT_PX)

    def cursor_name(self, point: Optional[Point]) -> str:
        """arrow | cross | vertex | move | rotate — the cursor over ``point``."""
        if point is None or self._drag is not None:
            return "arrow" if self._drag is None else "move"
        if self.tool == TOOL_DRAW:
            return "cross"
        if self.tool == TOOL_LINE:
            return "vertex" if self.hit_line_end(point) is not None else "cross"
        if self.hit_handle(point) is not None:
            return "rotate"
        if self.hit_vertex(point) is not None:
            return "vertex"
        if self.hit_body(point) is not None:
            return "move"
        return "arrow"

    # ------------------------------------------------------------------
    # Mouse input (model coordinates)
    # ------------------------------------------------------------------
    @staticmethod
    def _no_snap(modifiers) -> bool:
        return bool(modifiers & Qt.KeyboardModifier.AltModifier)

    def press(self, point: Point, button=Qt.MouseButton.LeftButton,
              modifiers=Qt.KeyboardModifier.NoModifier) -> None:
        self._message = ""
        if button == Qt.MouseButton.RightButton:
            self._right_click(point)
            return
        if button != Qt.MouseButton.LeftButton:
            return
        if self.tool == TOOL_DRAW:
            self.add_draft_point(point, snap=not self._no_snap(modifiers))
        elif self.tool == TOOL_EDIT:
            self._press_edit(point)
        elif self.tool == TOOL_LINE:
            self._press_line(point, snap=not self._no_snap(modifiers))

    def _press_edit(self, point: Point) -> None:
        handle = self.hit_handle(point)
        shape = self.document.shape(self.selected)
        if handle is not None and shape is not None:
            self._drag = _Drag(handle, point, shape.id, original=list(shape.points))
            self._notify()
            return
        vertex = self.hit_vertex(point)
        if vertex is not None:
            shape_id, index = vertex
            self.select(shape_id, index)
            shape = self.document.shape(shape_id)
            self._drag = _Drag("vertex", point, shape_id, index, list(shape.points))
            return
        body = self.hit_body(point)
        if body is not None:
            self.select(body)
            shape = self.document.shape(body)
            self._drag = _Drag("move", point, body, original=list(shape.points))
            return
        self.select(None)

    def _press_line(self, point: Point, snap: bool) -> None:
        end = self.hit_line_end(point)
        if end is not None and self.line_start is None:
            self._drag = _Drag("line-end", point, end=end,
                               original_line=self.document.line)
            return
        snapped = to_line_coords(self.snap(point, enabled=snap))
        if self.line_start is None:
            self.line_start = snapped
            self._drag = _Drag("line-new", point, original_line=self.document.line)
            self._notify()
            return
        self.set_line((self.line_start, snapped), "Draw reference line")

    def move(self, point: Point,
             modifiers=Qt.KeyboardModifier.NoModifier) -> None:
        self.hover = point
        drag = self._drag
        if drag is None:
            previous = self.snap_target
            if self.tool in (TOOL_DRAW, TOOL_LINE):
                self.snap(point, enabled=not self._no_snap(modifiers))
            # Only redraw when the pointer moves something visible.
            if self.draft or self.line_start is not None or previous != self.snap_target:
                self.state_changed.emit()
            return
        distance_px = math.hypot(point[0] - drag.start[0],
                                 point[1] - drag.start[1]) * self.view_scale
        if not drag.moved and distance_px < 2.0:
            return
        drag.moved = True
        no_snap = self._no_snap(modifiers)
        if drag.kind == "vertex":
            target = self.snap(point, exclude=(drag.shape_id, drag.vertex), enabled=not no_snap)
            points = list(drag.original)
            points[drag.vertex] = quantize(target)
            self.document.set_points(drag.shape_id, points)
        elif drag.kind == "move":
            dx, dy = self._clamp_translation(
                drag.original, int(round(point[0] - drag.start[0])),
                int(round(point[1] - drag.start[1]))
            )
            self.document.set_points(
                drag.shape_id, [(x + dx, y + dy) for x, y in drag.original]
            )
        elif drag.kind in ("rotate", "scale"):
            self._drag_transform(drag, point, modifiers)
        elif drag.kind == "line-end":
            target = to_line_coords(self.snap(point, enabled=not no_snap))
            line = list(drag.original_line)
            line[drag.end] = (round(target[0], 2), round(target[1], 2))
            self.document.set_line((line[0], line[1]))
        elif drag.kind == "line-new":
            self.snap(point, enabled=not no_snap)
        self.state_changed.emit()

    def _drag_transform(self, drag: _Drag, point: Point, modifiers) -> None:
        centre = self._bbox_centre(drag.original)
        start_vector = (drag.start[0] - centre[0], drag.start[1] - centre[1])
        vector = (point[0] - centre[0], point[1] - centre[1])
        if drag.kind == "rotate":
            angle = math.atan2(vector[1], vector[0]) - math.atan2(
                start_vector[1], start_vector[0]
            )
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                step = math.radians(15.0)
                angle = round(angle / step) * step
            points = rotate_about(drag.original, centre, angle)
        else:
            start_length = math.hypot(*start_vector) or 1.0
            factor = max(0.05, math.hypot(*vector) / start_length)
            points = scale_about(drag.original, centre, factor)
        clamped = [clamp_point(p, self.size) for p in points]
        self.document.set_points(drag.shape_id, clamped)

    def release(self, point: Point,
                modifiers=Qt.KeyboardModifier.NoModifier) -> None:
        drag, self._drag = self._drag, None
        if drag is None:
            return
        if drag.kind == "line-new":
            if drag.moved and self.line_start is not None:
                end = to_line_coords(self.snap(point, enabled=not self._no_snap(modifiers)))
                start = self.line_start
                self.set_line((start, end), "Draw reference line")
            self.snap_target = None
            self._notify()
            return
        if drag.kind == "line-end":
            new_line = self.document.line
            self.document.set_line(drag.original_line)  # undo stack replays it
            if drag.moved and new_line != drag.original_line:
                self.set_line(new_line, "Move reference line end")
            self.snap_target = None
            self._notify()
            return
        shape = self.document.shape(drag.shape_id)
        self.snap_target = None
        if shape is not None and drag.moved and shape.points != drag.original:
            final = list(shape.points)
            text = {"vertex": "Move vertex", "move": "Move polygon",
                    "rotate": "Rotate polygon", "scale": "Scale polygon"}[drag.kind]
            self.undo_stack.push(SetPointsCommand(
                self.document, drag.shape_id, drag.original, final, text
            ))
        self._update_overlay()
        self._notify()

    def cancel_drag(self) -> bool:
        drag, self._drag = self._drag, None
        if drag is None:
            return False
        if drag.original is not None and drag.shape_id is not None:
            self.document.set_points(drag.shape_id, drag.original)
        if drag.kind == "line-end":
            self.document.set_line(drag.original_line)
        self.snap_target = None
        self._update_overlay()
        self._notify()
        return True

    def double_click(self, point: Point,
                     modifiers=Qt.KeyboardModifier.NoModifier) -> None:
        self._message = ""
        self._drag = None
        if self.tool == TOOL_DRAW:
            if self.draft:
                self.close_draft()
            return
        if self.tool != TOOL_EDIT:
            return
        if self.hit_vertex(point) is not None:
            return
        edge = self.hit_edge(point)
        if edge is not None:
            shape_id, index, closest = edge
            self.insert_vertex(shape_id, index, closest)

    def _right_click(self, point: Point) -> None:
        if self.tool == TOOL_DRAW:
            if len(self.draft) >= MIN_POLYGON_POINTS:
                self.close_draft()
            return
        if self.tool == TOOL_LINE:
            if self.hit_line(point) or self.hit_line_end(point) is not None:
                self.delete_line()
            return
        vertex = self.hit_vertex(point)
        if vertex is not None:
            self.delete_vertex(*vertex)
            return
        body = self.hit_body(point)
        if body is not None:
            self.delete_shape(body)

    def leave(self) -> None:
        self.hover = None
        self.snap_target = None
        self.state_changed.emit()

    # ------------------------------------------------------------------
    # Keys that edit (the widget handles view / tool keys)
    # ------------------------------------------------------------------
    def key_press(self, key, modifiers) -> bool:
        """Handle an editing key; True when it was consumed."""
        ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        if ctrl and key == Qt.Key.Key_Z:
            self._message = ""
            if shift:
                self.redo()
            else:
                self.undo()
            return True
        if ctrl and key == Qt.Key.Key_Y:
            self._message = ""
            self.redo()
            return True
        if ctrl:
            return False
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._message = ""
            if self.draft:
                self.close_draft()
            else:
                following = self.next_row_needing_attention(self.current)
                if following is not None:
                    self.activate_zone(following)
            return True
        if key == Qt.Key.Key_Escape:
            if self.cancel_drag():
                return True
            if self.draft or self.line_start is not None:
                self.cancel_draft()
                return True
            return False
        if key == Qt.Key.Key_Backspace:
            if self.draft:
                self.draft_redo.append(self.draft.pop())
                self._notify()
                return True
            return False
        if key == Qt.Key.Key_Delete:
            return self.delete_selection()
        arrows = {
            Qt.Key.Key_Left: (-1, 0), Qt.Key.Key_Right: (1, 0),
            Qt.Key.Key_Up: (0, -1), Qt.Key.Key_Down: (0, 1),
        }
        if key in arrows and self.tool == TOOL_EDIT and self.selected is not None:
            step = 10 if shift else 1
            dx, dy = arrows[key]
            self.nudge(dx * step, dy * step)
            return True
        return False


__all__ = [
    "ARENA_COLOR", "CENTER_COLOR", "EditSession", "KIND_ARENA", "KIND_EXTRA",
    "KIND_ZONE", "OverlayProvider", "STATUS_DONE", "STATUS_MISSING",
    "TOOL_DRAW", "TOOL_EDIT", "TOOL_LINE", "TOOLS", "ZoneRow",
]
