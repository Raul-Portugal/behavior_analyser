"""Pure movement and zone-transition calculations.

This module deliberately has no GUI, OpenCV, NumPy, or plotting dependencies so
the behavioral data rules can be tested independently from the desktop runtime.
"""

from math import ceil, hypot
from typing import Dict, Iterator, List, Optional, Sequence, Tuple


Position = Optional[Tuple[float, float]]
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


def _contiguous_runs(positions: Sequence[Position]) -> Iterator[Tuple[int, int]]:
    """Yield (start, end_exclusive) for each run of non-None positions."""
    start: Optional[int] = None
    for index, position in enumerate(positions):
        if position is None:
            if start is not None:
                yield start, index
                start = None
        elif start is None:
            start = index
    if start is not None:
        yield start, len(positions)


def _window_half_width(fps: float, window_seconds: float) -> int:
    """Half-width in frames of a centered window, never below one frame."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if window_seconds < 0:
        raise ValueError("window_seconds cannot be negative")
    return max(1, int(round(window_seconds * fps / 2.0)))


def calculate_smoothed_speeds(
    positions: List[Position],
    fps: float,
    scale_factor: float = 1.0,
    window_seconds: float = 0.2
) -> List[Speed]:
    """Return speed from displacement across a centered time window.

    Speed at frame i is the distance between the positions half a window
    before and after i, divided by the elapsed time, which suppresses
    centroid jitter that frame-to-frame speed amplifies. The half-width is at
    least one frame. Windows are used only when every frame in them is
    tracked, so a tracking gap is never bridged and yields None instead.
    """
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive")
    half = _window_half_width(fps, window_seconds)
    speeds: List[Speed] = [None] * len(positions)
    seconds = 2 * half / fps
    for start, end in _contiguous_runs(positions):
        for index in range(start + half, end - half):
            before = positions[index - half]
            after = positions[index + half]
            speeds[index] = (
                hypot(after[0] - before[0], after[1] - before[1])
                / seconds * scale_factor
            )
    return speeds


def calculate_smoothed_path_length(
    positions: List[Position],
    fps: float,
    scale_factor: float = 1.0,
    window_seconds: float = 0.2
) -> float:
    """Return path length after a centered moving average of positions.

    Averaging is done inside each contiguous tracked run only, with the window
    shrinking symmetrically near run edges (so run endpoints stay raw), and
    step lengths are never summed across a gap. This removes the distance
    that centroid jitter adds to a raw frame-to-frame sum.
    """
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive")
    half = _window_half_width(fps, window_seconds)
    total = 0.0
    for start, end in _contiguous_runs(positions):
        length = end - start
        prefix_x = [0.0]
        prefix_y = [0.0]
        for position in positions[start:end]:
            prefix_x.append(prefix_x[-1] + position[0])
            prefix_y.append(prefix_y[-1] + position[1])
        previous: Optional[Tuple[float, float]] = None
        for offset in range(length):
            reach = min(half, offset, length - 1 - offset)
            low, high = offset - reach, offset + reach + 1
            count = high - low
            current = (
                (prefix_x[high] - prefix_x[low]) / count,
                (prefix_y[high] - prefix_y[low]) / count,
            )
            if previous is not None:
                total += hypot(
                    current[0] - previous[0], current[1] - previous[1]
                )
            previous = current
    return total * scale_factor


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


def calculate_immobility_bouts(
    speeds: List[Speed],
    fps: float,
    threshold: float,
    min_bout_seconds: float = 1.0
) -> Tuple[float, float, int, List[Optional[bool]]]:
    """Score immobility as sub-threshold speed sustained for a minimum time.

    Returns (immobile_time_s, immobile_pct_of_valid, bout_count, states).
    ``states`` has True for frames inside a qualifying bout, False for other
    tracked frames, and None where speed is missing. Missing speed breaks a
    bout, so a gap can neither extend nor join bouts.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if threshold < 0:
        raise ValueError("threshold cannot be negative")
    if min_bout_seconds < 0:
        raise ValueError("min_bout_seconds cannot be negative")

    min_frames = max(1, int(ceil(min_bout_seconds * fps - 1e-9)))
    states: List[Optional[bool]] = [
        None if speed is None else False for speed in speeds
    ]
    bout_count = 0
    run_start: Optional[int] = None

    def close_run(end: int) -> None:
        nonlocal bout_count
        if run_start is not None and end - run_start >= min_frames:
            bout_count += 1
            for index in range(run_start, end):
                states[index] = True

    for index, speed in enumerate(speeds):
        if speed is not None and speed < threshold:
            if run_start is None:
                run_start = index
            continue
        close_run(index)
        run_start = None
    close_run(len(speeds))

    valid_frames = sum(state is not None for state in states)
    immobile_frames = sum(state is True for state in states)
    if valid_frames == 0:
        return 0.0, 0.0, 0, states
    return (
        immobile_frames / fps,
        immobile_frames / valid_frames * 100.0,
        bout_count,
        states,
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
