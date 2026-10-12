"""Pure zone debouncing, transition, and center-mediated entry rules."""

from math import ceil
from typing import (Iterable, List, NamedTuple, Optional, Sequence, Set,
                    Tuple)


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


class CenterEntries(NamedTuple):
    """Counted arm entries plus how many entries tracking gaps may have hidden."""
    entries: List[Tuple[str, float]]
    possibly_missed: int


def analyze_center_mediated_entries(
    labels: Sequence[str],
    timestamps: Sequence[float],
    *,
    arm_labels: Set[str],
    center_label: str = "center",
    count_initial_arm: bool = True
) -> CenterEntries:
    """Count center-mediated arm entries and flag entries hidden by gaps.

    Counting stays conservative: an arm seen right after an undetected gap is
    not counted, because the route taken during the gap is unknown. Such a
    resume is counted in ``possibly_missed`` when the animal may have entered
    that arm: it resumes in a different arm than the last one seen before the
    gap, or it was last seen in the center, and no center was confirmed
    after the gap. A gap before any zone was seen is not flagged.
    """
    entries: List[Tuple[str, float]] = []
    possibly_missed = 0
    state: Optional[str] = None
    initial_segment = True
    last_arm: Optional[str] = None
    gap_pending = False
    state_before_gap: Optional[str] = None

    for label, timestamp in zip(labels, timestamps):
        if label == UNDETECTED:
            if state is not None:
                gap_pending = True
                state_before_gap = state
            state = None
            initial_segment = False
            continue
        if label == center_label:
            state = center_label
            gap_pending = False
            continue
        if label not in arm_labels:
            continue

        if state == center_label:
            entries.append((label, timestamp))
        elif state is None and initial_segment and count_initial_arm:
            entries.append((label, timestamp))
        elif state is None and gap_pending:
            if state_before_gap == center_label or label != last_arm:
                possibly_missed += 1
        gap_pending = False
        state = label
        last_arm = label

    return CenterEntries(entries, possibly_missed)


def extract_center_mediated_entries(
    labels: Sequence[str],
    timestamps: Sequence[float],
    *,
    arm_labels: Set[str],
    center_label: str = "center",
    count_initial_arm: bool = True
) -> List[Tuple[str, float]]:
    """Return arm entries that begin initially or follow a confirmed center."""
    return analyze_center_mediated_entries(
        labels,
        timestamps,
        arm_labels=arm_labels,
        center_label=center_label,
        count_initial_arm=count_initial_arm,
    ).entries
