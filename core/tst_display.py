"""Shared colour and label mapping for TST state overlays.

The tuner preview and the validation video draw the same states, so they
share one mapping instead of keeping two copies in sync by hand.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

IMMOBILE_STATES = frozenset({"immobile_static", "passive_swing"})
KNOWN_STATES = IMMOBILE_STATES | {"active_escape"}

# BGR colours (OpenCV order) and overlay labels.
UNCERTAIN_STYLE = ((0, 165, 255), "UNCERTAIN")
PASSIVE_SWING_STYLE = ((180, 0, 180), "PASSIVE SWING")
IMMOBILE_STATIC_STYLE = ((0, 0, 255), "IMMOBILE STATIC")
SHORT_PAUSE_STYLE = ((0, 215, 255), "SHORT PAUSE")
ACTIVE_ESCAPE_STYLE = ((0, 255, 0), "ACTIVE ESCAPE")


def tst_display_style(
    behavior_state: str,
    is_immobile: Optional[bool],
) -> Tuple[Tuple[int, int, int], str]:
    """Return the (BGR colour, label) for one frame.

    ``is_immobile`` is the temporally filtered immobility flag: ``True`` for
    a scored immobile bout, ``False`` for mobile, ``None`` when unscorable. A
    frame whose raw state is immobile but that did not survive the minimum
    bout duration is drawn as a short pause rather than as an escape.
    """
    if behavior_state not in KNOWN_STATES or is_immobile is None:
        return UNCERTAIN_STYLE
    if is_immobile:
        if behavior_state == "passive_swing":
            return PASSIVE_SWING_STYLE
        return IMMOBILE_STATIC_STYLE
    if behavior_state in IMMOBILE_STATES:
        return SHORT_PAUSE_STYLE
    return ACTIVE_ESCAPE_STYLE


def blend_polygon(
    frame: np.ndarray,
    points: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.35,
    beta: float = 0.65,
) -> None:
    """Alpha-blend a filled polygon into ``frame`` in place.

    Only the polygon's bounding box (clamped to the frame) is copied and
    blended, which gives exactly the same pixels as copying and blending the
    whole frame because every pixel outside the polygon blends with itself.
    ``points`` is an int32 array reshapeable to (N, 1, 2); ``alpha`` weights
    the colour and ``beta`` the original frame.
    """
    points = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    if len(points) == 0:
        return
    frame_h, frame_w = frame.shape[:2]
    x, y, w, h = cv2.boundingRect(points)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame_w, x + w), min(frame_h, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    region = frame[y0:y1, x0:x1]
    overlay = region.copy()
    shifted = points - np.array([x0, y0], dtype=np.int32)
    cv2.fillPoly(overlay, [shifted], color)
    cv2.addWeighted(overlay, alpha, region, beta, 0, region)
