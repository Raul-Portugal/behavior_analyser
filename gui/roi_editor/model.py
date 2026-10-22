"""The ROI editor's document model: shapes, reference line, undoable commands.

The document is the single source of truth for what is drawn. The canvas only
displays it and turns mouse input into document edits; every user-visible edit
goes through a ``QUndoCommand`` so Ctrl+Z / Ctrl+Y cover all of them.

Coordinates are SOURCE pixels. A polygon vertex (x, y) is an integer: the
pixel it designates, exactly what ``ROIManager`` stores (int32). The canvas
draws it at the centre of that pixel, which is where OpenCV considers it.
The reference line keeps sub-pixel endpoints, like ``ROIManager`` does.
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QUndoCommand

from roi_manager import ARENA_CATEGORY, ROIManager

from .rules import Point

Line = Tuple[Point, Point]

REFERENCE_LINE_NAME = "User-drawn reference line"


def quantize(point: Point) -> Tuple[int, int]:
    """The pixel a source coordinate designates (nearest integer vertex)."""
    return (int(math.floor(point[0] + 0.5)), int(math.floor(point[1] + 0.5)))


def clamp_point(point: Point, size: Tuple[int, int]) -> Tuple[float, float]:
    """Keep a point inside the frame (pixel indices 0..size-1)."""
    return (
        min(max(float(point[0]), 0.0), float(size[0] - 1)),
        min(max(float(point[1]), 0.0), float(size[1] - 1)),
    )


@dataclass
class Shape:
    """One closed polygon of a category (the arena is the ARENA_CATEGORY one)."""
    id: int
    category: str
    points: List[Tuple[int, int]]

    @property
    def is_arena(self) -> bool:
        return self.category == ARENA_CATEGORY

    def copy(self) -> "Shape":
        return Shape(self.id, self.category, list(self.points))


@dataclass
class _Reference:
    """Calibration carried over from a loaded manager that has no endpoints."""
    length: float
    name: str


class RoiDocument(QObject):
    """Shapes (in drawing order), the reference line and the loaded extras."""

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.shapes: List[Shape] = []
        self.line: Optional[Line] = None
        self.line_name = REFERENCE_LINE_NAME
        # A calibration length loaded without endpoints (legacy or
        # alignment-scaled): kept on save until a line replaces it.
        self.legacy_reference: Optional[_Reference] = None
        self.category_order: List[str] = []
        self._next_id = 1

    # -- queries --------------------------------------------------------
    def new_id(self) -> int:
        shape_id = self._next_id
        self._next_id += 1
        return shape_id

    def shape(self, shape_id: Optional[int]) -> Optional[Shape]:
        for shape in self.shapes:
            if shape.id == shape_id:
                return shape
        return None

    def shapes_in(self, category: str) -> List[Shape]:
        return [shape for shape in self.shapes if shape.category == category]

    def index_of(self, shape_id: int) -> int:
        for index, shape in enumerate(self.shapes):
            if shape.id == shape_id:
                return index
        raise KeyError(shape_id)

    def categories(self) -> List[str]:
        return list(dict.fromkeys(shape.category for shape in self.shapes))

    def all_vertices(self, exclude: Optional[Tuple[int, int]] = None
                     ) -> List[Tuple[float, float]]:
        """Every vertex of every shape, optionally without one (id, index)."""
        vertices = []
        for shape in self.shapes:
            for index, point in enumerate(shape.points):
                if exclude is not None and exclude == (shape.id, index):
                    continue
                vertices.append((float(point[0]), float(point[1])))
        return vertices

    # -- primitive edits (commands call these) --------------------------
    def insert_shape(self, shape: Shape, index: Optional[int] = None) -> None:
        shape.points = [quantize(p) for p in shape.points]
        self._next_id = max(self._next_id, shape.id + 1)
        if index is None or index >= len(self.shapes):
            self.shapes.append(shape)
        else:
            self.shapes.insert(max(index, 0), shape)
        self.changed.emit()

    def remove_shape(self, shape_id: int) -> Tuple[Shape, int]:
        index = self.index_of(shape_id)
        shape = self.shapes.pop(index)
        self.changed.emit()
        return shape, index

    def set_points(self, shape_id: int, points: Sequence[Point]) -> None:
        shape = self.shape(shape_id)
        if shape is None:
            return
        new_points = [quantize(p) for p in points]
        if new_points == shape.points:
            return
        shape.points = new_points
        self.changed.emit()

    def set_line(self, line: Optional[Line]) -> None:
        normalised = None
        if line is not None:
            normalised = (
                (float(line[0][0]), float(line[0][1])),
                (float(line[1][0]), float(line[1][1])),
            )
        if normalised == self.line:
            return
        self.line = normalised
        self.changed.emit()

    # -- load / save ----------------------------------------------------
    def load(self, manager: Optional[ROIManager]) -> None:
        """Replace the content with a manager's ROIs, arena and line."""
        self.shapes = []
        self.line = None
        self.line_name = REFERENCE_LINE_NAME
        self.legacy_reference = None
        self.category_order = []
        self._next_id = 1
        if manager is not None:
            if manager.arena is not None:
                self.shapes.append(self._shape_from(ARENA_CATEGORY, manager.arena.points))
            for category, rois in manager.rois.items():
                self.category_order.append(category)
                for roi in rois:
                    self.shapes.append(self._shape_from(category, roi.points))
            if manager.has_reference:
                if manager.reference_line is not None:
                    self.line = manager.reference_line
                    self.line_name = manager.reference_name
                else:
                    self.legacy_reference = _Reference(
                        manager.reference_length_pixels, manager.reference_name
                    )
        self.changed.emit()

    def _shape_from(self, category: str, points) -> Shape:
        return Shape(
            self.new_id(), category,
            [(int(x), int(y)) for x, y in np.asarray(points).reshape(-1, 2)],
        )

    def build_manager(self, size: Tuple[int, int],
                      definition_order: Sequence[str]) -> ROIManager:
        """A NEW ROIManager with today's int32 rounding and the dimensions set.

        Categories are added in the maze's definition order (the order the
        step-by-step flow produced), then any extra loaded ones; polygons keep
        their drawing order inside a category.
        """
        manager = ROIManager()
        manager.set_source_dimensions((int(size[0]), int(size[1])))
        ordered = [c for c in definition_order if c != ARENA_CATEGORY]
        ordered += [c for c in self.category_order if c not in ordered]
        ordered += [c for c in self.categories()
                    if c not in ordered and c != ARENA_CATEGORY]
        for category in ordered:
            for shape in self.shapes_in(category):
                manager.add_roi(category, np.array(shape.points, dtype=np.int32))
        for shape in self.shapes_in(ARENA_CATEGORY)[:1]:
            manager.set_arena(np.array(shape.points, dtype=np.int32))
        if self.line is not None:
            try:
                manager.set_reference_line(self.line[0], self.line[1], self.line_name)
            except ValueError:
                pass
        elif self.legacy_reference is not None:
            manager.set_reference_length(
                self.legacy_reference.length, self.legacy_reference.name
            )
        return manager


# ----------------------------------------------------------------------
# Undoable commands
# ----------------------------------------------------------------------
class AddShapeCommand(QUndoCommand):
    def __init__(self, document: RoiDocument, shape: Shape, text: str = "Add polygon"):
        super().__init__(text)
        self._document = document
        self._shape = shape.copy()

    def redo(self):
        self._document.insert_shape(self._shape.copy())

    def undo(self):
        self._document.remove_shape(self._shape.id)


class DeleteShapeCommand(QUndoCommand):
    def __init__(self, document: RoiDocument, shape_id: int, text: str = "Delete polygon"):
        super().__init__(text)
        self._document = document
        self._shape_id = shape_id
        self._removed: Optional[Shape] = None
        self._index = 0

    def redo(self):
        self._removed, self._index = self._document.remove_shape(self._shape_id)

    def undo(self):
        if self._removed is not None:
            self._document.insert_shape(self._removed.copy(), self._index)


class SetPointsCommand(QUndoCommand):
    """Move / insert / delete a vertex, or move, rotate, scale a polygon."""

    def __init__(self, document: RoiDocument, shape_id: int,
                 old_points: Sequence[Point], new_points: Sequence[Point], text: str):
        super().__init__(text)
        self._document = document
        self._shape_id = shape_id
        self._old = [quantize(p) for p in old_points]
        self._new = [quantize(p) for p in new_points]

    def redo(self):
        self._document.set_points(self._shape_id, self._new)

    def undo(self):
        self._document.set_points(self._shape_id, self._old)


class SetLineCommand(QUndoCommand):
    def __init__(self, document: RoiDocument, old: Optional[Line],
                 new: Optional[Line], text: str = "Edit reference line"):
        super().__init__(text)
        self._document = document
        self._old = old
        self._new = new

    def redo(self):
        self._document.set_line(self._new)

    def undo(self):
        self._document.set_line(self._old)


__all__ = [
    "AddShapeCommand", "DeleteShapeCommand", "Line", "REFERENCE_LINE_NAME",
    "RoiDocument", "SetLineCommand", "SetPointsCommand", "Shape",
    "clamp_point", "quantize",
]
