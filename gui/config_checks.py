"""
gui/config_checks.py
Checks that can run the moment a template or plan is chosen, before any
dialog asks the user to configure anything.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import cv2

Dimensions = Tuple[int, int]


@dataclass(frozen=True)
class DimensionMismatch:
    video: Path
    expected: Dimensions
    actual: Optional[Dimensions]  # None: the file could not be read


def probe_dimensions(path: Path) -> Optional[Dimensions]:
    """Frame size of a video without decoding it, or None if unreadable."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    return (width, height) if width > 0 and height > 0 else None


def dimension_mismatches(
    videos: Iterable[Path],
    expected_for: Callable[[Path], Optional[Dimensions]],
    prober: Callable[[Path], Optional[Dimensions]] = probe_dimensions,
) -> List[DimensionMismatch]:
    """Videos whose frame size differs from the size their ROIs were drawn on.

    ``expected_for`` returns the size recorded with a video's ROIs, or None
    when nothing is recorded (legacy settings), in which case it is skipped.
    """
    found: List[DimensionMismatch] = []
    for video in videos:
        expected = expected_for(video)
        if expected is None:
            continue
        actual = prober(video)
        if actual is None or tuple(actual) != tuple(expected):
            found.append(DimensionMismatch(
                video, (int(expected[0]), int(expected[1])), actual
            ))
    return found


def format_dimension_mismatches(
    mismatches: List[DimensionMismatch], limit: int = 8
) -> str:
    """Human-readable list for a message box (empty string if none)."""
    if not mismatches:
        return ""
    lines = []
    for item in mismatches[:limit]:
        want = f"{item.expected[0]}x{item.expected[1]}"
        have = (
            f"{item.actual[0]}x{item.actual[1]}"
            if item.actual else "could not be read"
        )
        lines.append(f"  - {item.video.name}: ROIs drawn for {want}, video is {have}")
    if len(mismatches) > limit:
        lines.append(f"  ... and {len(mismatches) - limit} more")
    count = len(mismatches)
    return (
        f"{count} video{'s' if count != 1 else ''} in the list "
        f"{'do' if count != 1 else 'does'} not match the frame size the "
        "loaded ROIs were drawn on:\n" + "\n".join(lines) +
        "\n\nRegions would be misplaced. Remove those videos, or choose "
        "'Set up from scratch' when starting."
    )
