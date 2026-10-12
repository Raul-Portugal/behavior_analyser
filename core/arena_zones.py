"""Pure cv2/numpy geometry for open-field center/periphery analysis.

An arena is a polygon in source-pixel coordinates. The center region is the
set of arena points whose distance to the nearest wall is at least a
threshold, so it is the arena shrunk (inset) by that distance. The threshold
is either given directly (pixels) or chosen so the center covers a fraction
of the arena's area.

Distances to the wall use ``cv2.pointPolygonTest`` in the continuous domain.
Area-based questions (center size for an area fraction, the center outline)
are answered on a raster whose polygon edges are placed with sub-pixel
accuracy, so they agree with the continuous classification to about a pixel.
"""
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

CENTER = "center"
PERIPHERY = "periphery"
OUTSIDE = "outside"
UNDETECTED = "undetected"

Position = Optional[Tuple[float, float]]

# Polygon vertices are rasterized with this many fractional bits.
_SUBPIXEL_SHIFT = 4
_SUBPIXEL_ONE = 1 << _SUBPIXEL_SHIFT
# Border of empty pixels around the polygon's bounding box.
_RASTER_PAD = 2
# Tolerance when simplifying a raster outline back to a polygon, in pixels.
_OUTLINE_EPSILON = 0.75


def _polygon(arena_points) -> np.ndarray:
    """Validate an arena and return it as an Nx2 float32 array."""
    polygon = np.asarray(arena_points, dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("Arena must be an Nx2 polygon with at least 3 points")
    if not np.all(np.isfinite(polygon)):
        raise ValueError("Arena points must be finite")
    if cv2.contourArea(polygon) <= 0:
        raise ValueError("Arena polygon has zero area")
    return polygon


def _contour(polygon: np.ndarray) -> np.ndarray:
    return polygon.reshape(-1, 1, 2)


def _fixed_point(polygon: np.ndarray, origin: Tuple[int, int]) -> np.ndarray:
    shifted = (polygon - np.asarray(origin, dtype=np.float32)) * _SUBPIXEL_ONE
    return np.round(shifted).astype(np.int32)


def signed_wall_distance(arena_points, point: Position) -> float:
    """Signed distance in pixels from ``point`` to the nearest arena wall.

    Positive inside the arena, negative outside, 0 on the wall.
    """
    if point is None:
        raise ValueError("point must not be None")
    polygon = _polygon(arena_points)
    return float(cv2.pointPolygonTest(
        _contour(polygon), (float(point[0]), float(point[1])), True
    ))


def wall_distances(
    positions: Sequence[Position], arena_points
) -> List[Optional[float]]:
    """Signed wall distance for every position; None where undetected."""
    contour = _contour(_polygon(arena_points))
    distances: List[Optional[float]] = []
    for position in positions:
        if position is None:
            distances.append(None)
        else:
            distances.append(float(cv2.pointPolygonTest(
                contour, (float(position[0]), float(position[1])), True
            )))
    return distances


class _ArenaRaster:
    """Inside mask and per-pixel wall distance for one arena polygon."""

    def __init__(self, arena_points):
        polygon = _polygon(arena_points)
        x0 = int(np.floor(polygon[:, 0].min())) - _RASTER_PAD
        y0 = int(np.floor(polygon[:, 1].min())) - _RASTER_PAD
        x1 = int(np.ceil(polygon[:, 0].max())) + _RASTER_PAD
        y1 = int(np.ceil(polygon[:, 1].max())) + _RASTER_PAD
        self.origin = (x0, y0)
        size = (y1 - y0 + 1, x1 - x0 + 1)
        fixed = _fixed_point(polygon, self.origin).reshape(-1, 1, 2)

        inside = np.zeros(size, dtype=np.uint8)
        cv2.fillPoly(inside, [fixed], 1, lineType=cv2.LINE_8,
                     shift=_SUBPIXEL_SHIFT)
        # Distance of every pixel to the nearest wall pixel, so inside pixels
        # measure the distance to the wall without a half-pixel bias.
        wall = np.full(size, 255, dtype=np.uint8)
        cv2.polylines(wall, [fixed], True, 0, thickness=1,
                      lineType=cv2.LINE_8, shift=_SUBPIXEL_SHIFT)
        self.inside = inside.astype(bool)
        self.distance = cv2.distanceTransform(
            wall, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        self.inside_distances = self.distance[self.inside]


def center_threshold_for_area_fraction(arena_points, fraction: float) -> float:
    """Wall distance (px) whose inner region covers ``fraction`` of the arena.

    Returns the distance d such that the part of the arena at least d away
    from every wall has ``fraction`` of the arena's area.
    """
    fraction = float(fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be strictly between 0 and 1")
    distances = _ArenaRaster(arena_points).inside_distances
    if distances.size == 0:
        raise ValueError("Arena polygon is too small to rasterize")
    return float(np.quantile(distances, 1.0 - fraction))


def max_wall_distance(arena_points) -> float:
    """Largest wall distance (px) inside the arena (its inscribed radius)."""
    distances = _ArenaRaster(arena_points).inside_distances
    return float(distances.max()) if distances.size else 0.0


def classify_center_periphery(
    positions: Sequence[Position],
    arena_points,
    threshold_px: float,
) -> List[str]:
    """Label every position ``center``, ``periphery``, ``outside`` or ``undetected``.

    ``undetected`` is a missing position, ``outside`` lies beyond the arena
    wall, ``center`` is at least ``threshold_px`` from every wall, and
    ``periphery`` is the rest of the arena (the wall itself included).
    """
    threshold_px = float(threshold_px)
    if not np.isfinite(threshold_px) or threshold_px < 0:
        raise ValueError("threshold_px must be a non-negative number")
    labels: List[str] = []
    for distance in wall_distances(positions, arena_points):
        if distance is None:
            labels.append(UNDETECTED)
        elif distance < 0:
            labels.append(OUTSIDE)
        elif distance >= threshold_px:
            labels.append(CENTER)
        else:
            labels.append(PERIPHERY)
    return labels


def center_polygons(arena_points, threshold_px: float) -> List[np.ndarray]:
    """Outlines (Nx2 int32 source pixels) of the center region, largest first.

    Convex arenas give one polygon; a concave arena can split its center.
    Empty when the threshold exceeds the arena's inscribed radius.
    """
    threshold_px = float(threshold_px)
    if not np.isfinite(threshold_px) or threshold_px < 0:
        raise ValueError("threshold_px must be a non-negative number")
    raster = _ArenaRaster(arena_points)
    center = (raster.inside & (raster.distance >= threshold_px)).astype(np.uint8)
    contours, _ = cv2.findContours(
        center, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    x0, y0 = raster.origin
    outlines = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        simplified = cv2.approxPolyDP(contour, _OUTLINE_EPSILON, True)
        points = simplified.reshape(-1, 2)
        if len(points) >= 3:
            outlines.append(
                (points + np.array([x0, y0], dtype=np.int32)).astype(np.int32)
            )
    return outlines


def center_polygon(arena_points, threshold_px: float) -> Optional[np.ndarray]:
    """Largest center-region outline, or None when the center is empty."""
    outlines = center_polygons(arena_points, threshold_px)
    return outlines[0] if outlines else None


def analysis_mask(arena_points, geometry) -> np.ndarray:
    """uint8 mask (255 inside the arena) at the geometry's analysis resolution.

    ``geometry`` is a ``core.analysis_geometry.AnalysisGeometry``; the arena
    vertices are mapped from source pixels to analysis pixels with it.
    """
    polygon = _polygon(arena_points)
    width, height = geometry.analysis_dims
    mapped = np.array(
        [geometry.to_analysis_point((float(x), float(y))) for x, y in polygon],
        dtype=np.float32,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    fixed = _fixed_point(mapped, (0, 0)).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [fixed], 255, lineType=cv2.LINE_8,
                 shift=_SUBPIXEL_SHIFT)
    return mask


def valid_step_distances(
    positions: Sequence[Position],
    valid: Sequence[bool],
    jump_limit: Optional[float],
) -> List[Optional[float]]:
    """Distance of the step ending at each frame, by the analyzer's rule.

    Mirrors ``Analyzer.finalize_result``: a step counts only between two
    consecutive valid observations and only when it does not exceed
    ``jump_limit`` (None disables the limit). Frames without a counted step
    hold None.
    """
    steps: List[Optional[float]] = [None] * len(positions)
    previous: Position = None
    for index, position in enumerate(positions):
        if position is None or not valid[index]:
            previous = None
            continue
        if previous is not None:
            distance = float(np.sqrt(
                (position[0] - previous[0]) ** 2
                + (position[1] - previous[1]) ** 2
            ))
            if jump_limit is None or distance <= jump_limit:
                steps[index] = distance
        previous = position
    return steps
