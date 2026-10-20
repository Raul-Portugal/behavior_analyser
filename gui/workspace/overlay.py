"""
gui/workspace/overlay.py
Overlays for the Results tab's player: draw a finished analysis on the
original video when no validation video was made.

* :func:`tracking_overlay` - the tracked position (dot and short trail), the
  zone label and the ROI outlines, from ``*_tracking_data.csv``.
* :func:`tst_overlay` - per mouse, the behavioural state of the frame in the
  colours the tuner and the validation video use (``core.tst_display``), from
  ``*_tst_motion_evidence.csv``.

Both return a callback ``(frame_bgr, frame_index) -> frame_bgr`` for
``VideoPlayerWidget.set_overlay``. They only draw; they never touch the disk
after loading and never raise on a frame without data.
"""
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np

from core.models import BatchSettings
from core.positional_validation import TrackingData, load_tracking_csv
from core.tst_display import tst_display_style
from core.tst_validation import load_tst_evidence_csv
from gui.run_resume import plan_key_to_path

logger = logging.getLogger(__name__)

OverlayCallback = Callable[[np.ndarray, int], np.ndarray]

DOT_COLOR = (0, 220, 0)             # BGR: detected position
LOST_COLOR = (0, 140, 255)          # no detection on this frame
TRAIL_COLOR = (255, 200, 0)
ROI_COLOR = (200, 200, 200)
TEXT_COLOR = (255, 255, 255)
PANEL_COLOR = (30, 30, 30)
TRAIL_FRAMES = 45


def _scale(frame: np.ndarray) -> float:
    """Drawing scale: 1.0 at 480 px of frame height."""
    return max(0.5, frame.shape[0] / 480.0)


def _label(frame: np.ndarray, text: str, origin, color=TEXT_COLOR,
           background=PANEL_COLOR) -> int:
    """Draw ``text`` on a filled box at ``origin`` (top-left); returns its height."""
    scale = 0.55 * _scale(frame)
    thickness = max(1, int(round(scale * 2)))
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x, y = int(origin[0]), int(origin[1])
    pad = max(2, int(4 * _scale(frame)))
    cv2.rectangle(
        frame, (x, y), (x + width + 2 * pad, y + height + baseline + 2 * pad),
        background, -1,
    )
    cv2.putText(
        frame, text, (x + pad, y + pad + height), cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA,
    )
    return height + baseline + 2 * pad


def load_video_rois(
    run_dir: os.PathLike, source: Optional[os.PathLike]
) -> Dict[str, List[np.ndarray]]:
    """The ROI polygons the run used for ``source`` (from ``run_plan.json``)."""
    plan_file = Path(run_dir) / "run_plan.json"
    if source is None or not plan_file.is_file():
        return {}
    try:
        plan: Mapping[str, BatchSettings] = BatchSettings.load_batch_plan(plan_file)
        wanted = os.path.normcase(os.path.abspath(str(source)))
        for key, settings in plan.items():
            target = plan_key_to_path(key, plan_file)
            if os.path.normcase(os.path.abspath(str(target))) == wanted:
                return {
                    name: [np.asarray(roi.points, dtype=np.int32) for roi in rois]
                    for name, rois in settings.roi_manager.rois.items()
                }
    except Exception:
        logger.warning("Could not read the ROIs of %s", plan_file, exc_info=True)
    return {}


def _draw_rois(
    frame: np.ndarray, rois: Mapping[str, Sequence[np.ndarray]],
    color=ROI_COLOR, thickness: int = 1,
) -> None:
    for polygons in rois.values():
        for points in polygons:
            cv2.polylines(
                frame, [points.reshape(-1, 1, 2)], True, color, thickness,
                cv2.LINE_AA,
            )


# ----------------------------------------------------------------------
# Positional paradigms
# ----------------------------------------------------------------------
def tracking_overlay(
    data: TrackingData,
    rois: Optional[Mapping[str, Sequence[np.ndarray]]] = None,
    trail: int = TRAIL_FRAMES,
) -> OverlayCallback:
    """Position dot, short trail, zone label and ROI outlines for each frame."""
    by_frame = {frame: row for row, frame in enumerate(data.frames)}
    rois = dict(rois or {})

    def draw(frame: np.ndarray, index: int) -> np.ndarray:
        scale = _scale(frame)
        if rois:
            _draw_rois(frame, rois, thickness=max(1, int(scale)))
        row = by_frame.get(index)
        if row is None:
            _label(frame, "outside the analysed window", (6, 6))
            return frame
        # Trail: the positions of the previous frames that were detected.
        points = [
            data.positions[r] for r in range(max(0, row - trail), row + 1)
            if data.positions[r] is not None
        ]
        if len(points) > 1:
            cv2.polylines(
                frame,
                [np.round(np.array(points)).astype(np.int32).reshape(-1, 1, 2)],
                False, TRAIL_COLOR, max(1, int(scale)), cv2.LINE_AA,
            )
        position = data.positions[row]
        radius = max(4, int(round(5 * scale)))
        zone = data.zones[row] or "-"
        if position is None:
            _label(frame, f"zone: {zone}  (not detected)", (6, 6), LOST_COLOR)
            return frame
        centre = (int(round(position[0])), int(round(position[1])))
        cv2.circle(frame, centre, radius, DOT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, centre, radius + 2, (0, 0, 0), 1, cv2.LINE_AA)
        _label(frame, f"zone: {zone}", (6, 6))
        return frame

    return draw


def tracking_overlay_from_file(
    csv_path: os.PathLike,
    rois: Optional[Mapping[str, Sequence[np.ndarray]]] = None,
) -> OverlayCallback:
    return tracking_overlay(load_tracking_csv(csv_path), rois)


# ----------------------------------------------------------------------
# Tail suspension test
# ----------------------------------------------------------------------
def tst_overlay(
    evidence: Mapping[str, Sequence[Mapping[str, object]]],
    rois: Optional[Mapping[str, Sequence[np.ndarray]]] = None,
) -> OverlayCallback:
    """Each mouse's state colour and label (ROI outlined in that colour)."""
    lookup: Dict[str, Dict[int, Mapping[str, object]]] = {
        mouse: {int(row["source_frame"]): row for row in rows}
        for mouse, rows in evidence.items()
    }
    rois = dict(rois or {})

    def draw(frame: np.ndarray, index: int) -> np.ndarray:
        scale = _scale(frame)
        y = 6
        for mouse, rows in lookup.items():
            row = rows.get(index)
            if row is None:
                continue
            color, label = tst_display_style(
                str(row.get("state") or "uncertain"), row.get("is_immobile")
            )
            for points in rois.get(mouse, []):
                cv2.polylines(
                    frame, [points.reshape(-1, 1, 2)], True, color,
                    max(2, int(2 * scale)), cv2.LINE_AA,
                )
            blue, green, red = color
            dark = 0.114 * blue + 0.587 * green + 0.299 * red < 110
            y += _label(
                frame, f"{mouse}: {label}", (6, y),
                (255, 255, 255) if dark else (0, 0, 0), color,
            )
            y += 2
        return frame

    return draw


def tst_overlay_from_file(
    csv_path: os.PathLike,
    rois: Optional[Mapping[str, Sequence[np.ndarray]]] = None,
) -> OverlayCallback:
    return tst_overlay(load_tst_evidence_csv(Path(csv_path)), rois)
