"""Pure zone debouncing, transition, and center-mediated entry rules."""

from math import ceil
from typing import Iterable, List, Optional, Sequence, Set, Tuple


UNDETECTED = "undetected"
OUTSIDE = "outside"


def dwell_frames(seconds: float, fps: float) -> int:
    if seconds < 0:
        raise ValueError("dwell time cannot be negative")
    if fps <= 0:
        raise ValueError("fps must be positive")
    return max(1, int(ceil(seconds * fps)))


def overlapping_zones_to_candidates(
    overlapping_zones: Sequence[Sequence[str]]
) -> List[Optional[str]]:
    """Convert overlaps to candidates; None means retain the stable zone."""
    candidates: List[Optional[str]] = []
    for zones in overlapping_zones:
        unique = list(dict.fromkeys(zones))
        if unique == [UNDETECTED]:
            candidates.append(UNDETECTED)
        elif len(unique) == 1:
            candidates.append(unique[0])
        else:
            candidates.append(None)
    return candidates


def debounce_zone_labels(
    candidates: Sequence[Optional[str]],
    min_dwell_frames: int
) -> List[str]:
    """Suppress brief zone changes and retain state through ROI overlaps."""
    if min_dwell_frames <= 0:
        raise ValueError("min_dwell_frames must be positive")

    output: List[Optional[str]] = [None] * len(candidates)
    stable: Optional[str] = None
    pending: Optional[str] = None
    pending_indices: List[int] = []

    def flush_pending(commit: bool = False):
        nonlocal stable, pending, pending_indices
        if not pending_indices:
            return
        if commit:
            stable = pending
            replacement = pending
        else:
            replacement = stable if stable is not None else OUTSIDE
        for pending_index in pending_indices:
            output[pending_index] = replacement
        pending = None
        pending_indices = []

    for index, candidate in enumerate(candidates):
        if candidate == UNDETECTED:
            flush_pending()
            output[index] = UNDETECTED
            stable = None
            continue

        if candidate is None:
            if pending is not None:
                pending_indices.append(index)
            else:
                output[index] = stable if stable is not None else OUTSIDE
            continue

        if candidate == stable:
            flush_pending()
            output[index] = stable
            continue

        if candidate != pending:
            flush_pending()
            pending = candidate
            pending_indices = []

        pending_indices.append(index)
        if len(pending_indices) >= min_dwell_frames:
            flush_pending(commit=True)

    flush_pending()
    return [
        label if label is not None else OUTSIDE
        for label in output
    ]


def extract_zone_transitions(
    labels: Sequence[str],
    *,
    excluded_labels: Iterable[str] = (UNDETECTED,)
) -> List[Tuple[str, str, int]]:
    """Extract transitions without bridging excluded/missing states."""
    excluded = set(excluded_labels)
    transitions: List[Tuple[str, str, int]] = []
    current: Optional[str] = None

    for index, label in enumerate(labels):
        if label in excluded:
            current = None
            continue
        if current is None:
            current = label
            continue
        if label != current:
            transitions.append((current, label, index))
            current = label

    return transitions


def extract_center_mediated_entries(
    labels: Sequence[str],
    timestamps: Sequence[float],
    *,
    arm_labels: Set[str],
    center_label: str = "center",
    count_initial_arm: bool = True
) -> List[Tuple[str, float]]:
    """Return arm entries that begin initially or follow a confirmed center."""
    entries: List[Tuple[str, float]] = []
    state: Optional[str] = None
    initial_segment = True

    for label, timestamp in zip(labels, timestamps):
        if label == UNDETECTED:
            state = None
            initial_segment = False
            continue
        if label == center_label:
            state = center_label
            continue
        if label not in arm_labels:
            continue

        if state == center_label:
            entries.append((label, timestamp))
        elif state is None and initial_segment and count_initial_arm:
            entries.append((label, timestamp))
        state = label

    return entries
