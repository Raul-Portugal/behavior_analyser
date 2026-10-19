"""The ROI editor: a QGraphicsView-based editor for ROI polygons, the arena
and the calibration reference line.

``RoiEditorWidget`` is the embeddable editor; ``RoiEditorDialog`` wraps it for
the modal flow. Editing logic lives in ``EditSession`` (no widgets), drawing
in ``RoiCanvas``.
"""
from .model import REFERENCE_LINE_NAME, RoiDocument, Shape
from .rules import (ARENA_RULE, ARENA_STEP_DESCRIPTION, MIN_POLYGON_POINTS,
                    MIN_ROI_AREA_PX, SNAP_RADIUS_SCREEN_PX, RoiRule,
                    RoiStepState, evaluate_roi_step, roi_rule_for_maze)
from .session import (TOOL_DRAW, TOOL_EDIT, TOOL_LINE, EditSession, ZoneRow)
from .widget import RoiEditorWidget

__all__ = [
    "ARENA_RULE", "ARENA_STEP_DESCRIPTION", "EditSession",
    "MIN_POLYGON_POINTS", "MIN_ROI_AREA_PX", "REFERENCE_LINE_NAME",
    "RoiDocument", "RoiEditorWidget", "RoiRule", "RoiStepState",
    "SNAP_RADIUS_SCREEN_PX", "Shape", "TOOL_DRAW", "TOOL_EDIT", "TOOL_LINE",
    "ZoneRow", "evaluate_roi_step", "roi_rule_for_maze",
]
