"""Pure movement and zone-transition calculations.

This module deliberately has no GUI, OpenCV, NumPy, or plotting dependencies so
the behavioral data rules can be tested independently from the desktop runtime.
"""

from math import hypot
from typing import Dict, List, Optional, Tuple


Position = Optional[Tuple[int, int]]
Speed = Optional[float]


def calculate_instantaneous_speeds(
    positions: List[Position],
    fps: float,
    scale_factor: float = 1.0
) -> List[Speed]:
    """Return one speed per position without converting gaps into zero speed."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive")
    if not positions:
        return []

    speeds: List[Speed] = [None]
    for previous, current in zip(positions, positions[1:]):
        if previous is None or current is None:
            speeds.append(None)
            continue
        speeds.append(
            hypot(current[0] - previous[0], current[1] - previous[1])
            * fps
            * scale_factor
        )
    return speeds


def calculate_movement_efficiency(
    positions: List[Position],
    total_distance: float,
    scale_factor: float = 1.0
) -> float:
    """Return straight-line displacement divided by valid traveled distance."""
    if total_distance <= 0:
        return 0.0

    contiguous_displacement = 0.0
    segment_start: Position = None
    segment_end: Position = None

    def finish_segment() -> float:
        if segment_start is None or segment_end is None:
            return 0.0
        return hypot(
            segment_end[0] - segment_start[0],
            segment_end[1] - segment_start[1],
        )

    for position in positions:
        if position is None:
            contiguous_displacement += finish_segment()
            segment_start = None
            segment_end = None
            continue
        if segment_start is None:
            segment_start = position
        segment_end = position
    contiguous_displacement += finish_segment()

    scaled_displacement = contiguous_displacement * scale_factor
    return min(scaled_displacement / total_distance, 1.0)


def calculate_immobility_time(
    speeds: List[Speed],
    fps: float,
    threshold: float = 2.0
) -> Tuple[float, float]:
    """Calculate immobility using tracked speed samples only."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if threshold < 0:
        raise ValueError("threshold cannot be negative")

    valid_speeds = [speed for speed in speeds if speed is not None]
    if not valid_speeds:
        return 0.0, 0.0

    immobile_frames = sum(speed < threshold for speed in valid_speeds)
    return (
        immobile_frames / fps,
        immobile_frames / len(valid_speeds) * 100.0
    )


def calculate_zone_transitions(
    roi_labels: List[str]
) -> Tuple[List[Tuple[str, str, int]], int]:
    """Count observed transitions without bridging tracking gaps."""
    transitions: List[Tuple[str, str, int]] = []
    current_zone: Optional[str] = None

    for frame_offset, next_zone in enumerate(roi_labels):
        if next_zone == 'undetected':
            current_zone = None
            continue
        if current_zone is None:
            current_zone = next_zone
            continue
        if next_zone != current_zone:
            transitions.append((current_zone, next_zone, frame_offset))
            current_zone = next_zone

    return transitions, len(transitions)


def calculate_zone_first_entries(
    roi_labels: List[str],
    timestamps: List[float]
) -> Dict[str, float]:
    """Return first observed entry times, excluding non-spatial states."""
    first_entries: Dict[str, float] = {}
    for label, timestamp in zip(roi_labels, timestamps):
        if label not in {'outside', 'undetected'} and label not in first_entries:
            first_entries[label] = timestamp
    return first_entries
