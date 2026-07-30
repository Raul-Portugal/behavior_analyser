"""Pure temporal helpers for Tail Suspension Test state classification."""

from typing import Iterable, List, Optional, Sequence


def bridge_short_false_gaps(
    states: Iterable[bool],
    max_gap_frames: int
) -> List[bool]:
    """Fill short mobile gaps only when bounded by immobility on both sides."""
    values = [bool(value) for value in states]
    if max_gap_frames <= 0:
        return values

    index = 0
    while index < len(values):
        if values[index]:
            index += 1
            continue
        gap_start = index
        while index < len(values) and not values[index]:
            index += 1
        bounded = gap_start > 0 and index < len(values)
        if bounded and (index - gap_start) <= max_gap_frames:
            values[gap_start:index] = [True] * (index - gap_start)
    return values


def retain_minimum_true_runs(
    states: Iterable[bool],
    min_run_frames: int
) -> List[bool]:
    """Keep only True runs meeting the requested minimum duration."""
    values = [bool(value) for value in states]
    minimum = max(1, int(min_run_frames))
    filtered = [False] * len(values)
    index = 0
    while index < len(values):
        if not values[index]:
            index += 1
            continue
        run_start = index
        while index < len(values) and values[index]:
            index += 1
        if index - run_start >= minimum:
            filtered[run_start:index] = [True] * (index - run_start)
    return filtered


def smooth_tst_immobility_states(
    states: Sequence[str],
    min_bout_frames: int,
    max_active_gap_frames: int,
) -> List[Optional[bool]]:
    """Map multi-state evidence to immobility without crossing uncertainty."""
    mapped: List[Optional[bool]] = []
    for state in states:
        if state in {"immobile_static", "passive_swing"}:
            mapped.append(True)
        elif state == "active_escape":
            mapped.append(False)
        else:
            mapped.append(None)

    output: List[Optional[bool]] = [None] * len(mapped)
    start = 0
    while start < len(mapped):
        if mapped[start] is None:
            start += 1
            continue
        end = start
        while end < len(mapped) and mapped[end] is not None:
            end += 1
        segment = [bool(value) for value in mapped[start:end]]
        bridged = bridge_short_false_gaps(
            segment, max_active_gap_frames
        )
        retained = retain_minimum_true_runs(bridged, min_bout_frames)
        output[start:end] = retained
        start = end
    return output
