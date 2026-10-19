"""Criterion validation helpers for the positional paradigms.

Compares automated scores for the Y-maze, elevated plus maze and
position/zone tracking with blinded manual scoring. Everything here is a pure
computation on plain data: nothing reads video, and nothing changes an
analysis output. Manual annotations are read from CSV files in the same shape
the application writes (``entry_number,arm,timestamp_s``), and the exported
``<base>_tracking_data.csv`` can be loaded back so validation can run after an
analysis without the in-memory result.

Conventions: the automated value is always the first argument and the manual
value the second; differences are ``auto - manual``.
"""
import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import (
    Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence,
    Set, Tuple, Union,
)

import numpy as np

from core.agreement_stats import (
    AgreementSummary, KappaResult, cohens_kappa, percent_agreement,
    summarize_agreement, summarize_metrics,
)
from mazes.y_maze import sequence_alternation_metrics


NAN = float("nan")
UNDETECTED = "undetected"

EntryList = List[Tuple[str, float]]
Point = Optional[Tuple[float, float]]

# Tolerance comparisons are inclusive; this absorbs float round-off such as
# 1.3 - 0.3 > 1.0.
_TIME_EPSILON = 1e-9


# --------------------------------------------------------------------------
# Arm names
# --------------------------------------------------------------------------

_Y_ARM_PATTERN = re.compile(r"^(?:arm_?)?([abc])$")
_EPM_ARM_PATTERN = re.compile(r"^(open|closed)(?:_?arm)?(?:_?([12]))?$")
_EPM_SHORT_PATTERN = re.compile(r"^([oc])a?_?([12])$")


def _clean_name(name: object) -> str:
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def normalize_y_maze_arm(name: object) -> str:
    """Map ``arm_a``, ``A``, ``a``, ``Arm A``... to the app's ``arm_a``."""
    match = _Y_ARM_PATTERN.match(_clean_name(name))
    if not match:
        raise ValueError(f"Unrecognized Y-maze arm name: {name!r}")
    return f"arm_{match.group(1)}"


def normalize_epm_arm(name: object) -> str:
    """Map EPM arm names to the app's ``open_arm_1`` / ``closed_arm_2``.

    A bare ``open`` or ``closed`` (no arm number) is kept as ``open`` /
    ``closed``; ``OA1`` and ``CA2`` shorthands are accepted too.
    """
    cleaned = _clean_name(name)
    match = _EPM_ARM_PATTERN.match(cleaned)
    if match:
        kind, number = match.groups()
        return f"{kind}_arm_{number}" if number else kind
    match = _EPM_SHORT_PATTERN.match(cleaned)
    if match:
        kind = "open" if match.group(1) == "o" else "closed"
        return f"{kind}_arm_{match.group(2)}"
    raise ValueError(f"Unrecognized EPM arm name: {name!r}")


def epm_arm_class(arm: object) -> str:
    """Collapse an EPM arm name to ``open`` or ``closed``.

    Use as ``arm_key`` in :func:`match_events` when manual scoring only
    records the arm type.
    """
    normalized = normalize_epm_arm(arm)
    return "open" if normalized.startswith("open") else "closed"


def normalize_arm(name: object, maze: Optional[str] = None) -> str:
    """Normalize an arm name for ``maze`` (``"y_maze"``, ``"epm"``, or auto)."""
    if maze in ("y_maze", "y", "ymaze"):
        return normalize_y_maze_arm(name)
    if maze == "epm":
        return normalize_epm_arm(name)
    if maze is not None:
        raise ValueError(f"Unknown maze type: {maze!r}")
    try:
        return normalize_y_maze_arm(name)
    except ValueError:
        return normalize_epm_arm(name)


# --------------------------------------------------------------------------
# Manual annotation files
# --------------------------------------------------------------------------

_ARM_COLUMNS = ("arm", "arm_name", "zone")
_TIME_COLUMNS = ("timestamp_s", "timestamp", "time_s", "time")


def _first_column(fieldnames: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    return None


def load_entry_annotations(path, maze: Optional[str] = None) -> EntryList:
    """Read a manual arm-entry CSV as ``[(arm, timestamp_s), ...]``.

    The file uses the same columns the application writes in
    ``<base>_arm_entries.csv`` (``entry_number, arm, timestamp_s``);
    ``entry_number`` is optional and ``timestamp`` / ``time_s`` are accepted
    for the time column. Arm names are normalized (see
    :func:`normalize_y_maze_arm` and :func:`normalize_epm_arm`); with
    ``maze=None`` the name is tried as a Y-maze arm first, then as an EPM arm.
    Entries are returned in time order (ties keep ``entry_number`` order).
    """
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fieldnames = [
            _clean_name(name) for name in (reader.fieldnames or [])
        ]
        arm_column = _first_column(fieldnames, _ARM_COLUMNS)
        time_column = _first_column(fieldnames, _TIME_COLUMNS)
        if arm_column is None or time_column is None:
            raise ValueError(
                f"{path}: expected columns 'arm' and 'timestamp_s', "
                f"found {reader.fieldnames}"
            )
        entries = []
        for line, raw_row in enumerate(reader, start=2):
            row = {_clean_name(key): value for key, value in raw_row.items()
                   if key is not None}
            if not any(str(value or "").strip() for value in row.values()):
                continue
            try:
                timestamp = float(row[time_column])
            except (TypeError, ValueError):
                raise ValueError(
                    f"{path}, line {line}: invalid time {row[time_column]!r}"
                ) from None
            if not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError(
                    f"{path}, line {line}: time must be a finite value >= 0"
                )
            try:
                arm = normalize_arm(row[arm_column], maze)
            except ValueError as error:
                raise ValueError(f"{path}, line {line}: {error}") from None
            order = row.get("entry_number") or ""
            try:
                order_value = float(order)
            except ValueError:
                order_value = float(len(entries))
            entries.append((timestamp, order_value, arm))
    entries.sort(key=lambda item: (item[0], item[1]))
    return [(arm, timestamp) for timestamp, _, arm in entries]


# --------------------------------------------------------------------------
# Event matching
# --------------------------------------------------------------------------

@dataclass
class EventMatchResult:
    """Outcome of matching automated to manual events.

    ``matches`` holds ``(auto_index, manual_index, offset_s)`` with
    ``offset_s = auto - manual``; positive means the automated event is late.
    Precision is ``nan`` without automated events, recall ``nan`` without
    manual events, and F1 ``nan`` when there are no events at all.
    """
    tolerance_s: float
    n_auto: int
    n_manual: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_offset_s: float
    mean_abs_offset_s: float
    matches: List[Tuple[int, int, float]] = field(default_factory=list)
    unmatched_auto: List[int] = field(default_factory=list)
    unmatched_manual: List[int] = field(default_factory=list)


def match_events(
    auto: Sequence[Tuple[Hashable, float]],
    manual: Sequence[Tuple[Hashable, float]],
    tolerance_s: float,
    arm_key: Optional[Callable[[Hashable], Hashable]] = None,
) -> EventMatchResult:
    """Greedy one-to-one matching of ``(arm, time_s)`` events within a tolerance.

    Candidate pairs must have the same arm (after ``arm_key``, when given) and
    a time difference of at most ``tolerance_s`` (inclusive). Pairs are taken
    from the smallest time difference upward, each event used once, so the
    closest pairs win when events cluster. Unmatched automated events are
    false positives and unmatched manual events false negatives.
    """
    if tolerance_s < 0:
        raise ValueError("tolerance_s cannot be negative")
    key = arm_key or (lambda arm: arm)
    auto_events = [(key(arm), float(time)) for arm, time in auto]
    manual_events = [(key(arm), float(time)) for arm, time in manual]

    candidates = []
    for i, (auto_arm, auto_time) in enumerate(auto_events):
        for j, (manual_arm, manual_time) in enumerate(manual_events):
            if auto_arm != manual_arm:
                continue
            gap = abs(auto_time - manual_time)
            if gap <= tolerance_s + _TIME_EPSILON:
                candidates.append((gap, i, j))
    candidates.sort()

    used_auto: Set[int] = set()
    used_manual: Set[int] = set()
    matches = []
    for _, i, j in candidates:
        if i in used_auto or j in used_manual:
            continue
        used_auto.add(i)
        used_manual.add(j)
        matches.append((i, j, auto_events[i][1] - manual_events[j][1]))
    matches.sort()

    tp = len(matches)
    fp = len(auto_events) - tp
    fn = len(manual_events) - tp
    offsets = np.array([offset for _, _, offset in matches], dtype=float)
    return EventMatchResult(
        tolerance_s=float(tolerance_s),
        n_auto=len(auto_events),
        n_manual=len(manual_events),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=tp / (tp + fp) if tp + fp else NAN,
        recall=tp / (tp + fn) if tp + fn else NAN,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else NAN,
        mean_offset_s=float(offsets.mean()) if tp else NAN,
        mean_abs_offset_s=float(np.abs(offsets).mean()) if tp else NAN,
        matches=matches,
        unmatched_auto=[i for i in range(len(auto_events)) if i not in used_auto],
        unmatched_manual=[
            j for j in range(len(manual_events)) if j not in used_manual
        ],
    )


# --------------------------------------------------------------------------
# Sequence comparison
# --------------------------------------------------------------------------

@dataclass
class EditDistanceResult:
    """Levenshtein distance between two sequences and a 0-1 similarity."""
    distance: int
    length_a: int
    length_b: int
    similarity: float


def sequence_edit_distance(seq_a: Sequence, seq_b: Sequence) -> EditDistanceResult:
    """Levenshtein distance (insert, delete, substitute) between sequences.

    Works on strings or lists. ``similarity`` is ``1 - distance / max(len)``,
    and 1.0 when both sequences are empty.
    """
    a, b = list(seq_a), list(seq_b)
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (item_a != item_b),
            ))
        previous = current
    distance = previous[-1]
    longest = max(len(a), len(b))
    return EditDistanceResult(
        distance=distance,
        length_a=len(a),
        length_b=len(b),
        similarity=1.0 - distance / longest if longest else 1.0,
    )


@dataclass
class YMazeComparison:
    """Per-video comparison of automated and manual Y-maze scoring.

    Differences are ``auto - manual``. Alternation percentages are ``nan``
    when a sequence has fewer than three entries.
    """
    matching: EventMatchResult
    edit_distance: EditDistanceResult
    auto_sequence: str
    manual_sequence: str
    auto_total_entries: int
    manual_total_entries: int
    total_entries_difference: int
    auto_alternation_pct: float
    manual_alternation_pct: float
    alternation_difference_pct: float


def _y_sequence(entries: Sequence[Tuple[str, float]]) -> List[str]:
    return [normalize_y_maze_arm(arm).split("_")[-1].upper() for arm, _ in entries]


def compare_y_maze(
    auto_entries: Sequence[Tuple[str, float]],
    manual_entries: Sequence[Tuple[str, float]],
    tolerance_s: float = 1.0,
) -> YMazeComparison:
    """Compare automated and manual Y-maze arm entries for one video.

    Entries are ``(arm, timestamp_s)`` pairs such as
    ``YMazeAnalysisResult.arm_entries`` or :func:`load_entry_annotations`.
    The alternation percentage uses the same rule as the analysis itself
    (:func:`mazes.y_maze.sequence_alternation_metrics`).
    """
    auto = [(normalize_y_maze_arm(arm), float(t)) for arm, t in auto_entries]
    manual = [(normalize_y_maze_arm(arm), float(t)) for arm, t in manual_entries]
    auto_sequence = _y_sequence(auto)
    manual_sequence = _y_sequence(manual)

    def alternation(sequence):
        metrics = sequence_alternation_metrics(sequence)
        return metrics.alternation_score if metrics.possible_alternations else NAN

    auto_alt, manual_alt = alternation(auto_sequence), alternation(manual_sequence)
    return YMazeComparison(
        matching=match_events(auto, manual, tolerance_s),
        edit_distance=sequence_edit_distance(auto_sequence, manual_sequence),
        auto_sequence="".join(auto_sequence),
        manual_sequence="".join(manual_sequence),
        auto_total_entries=len(auto),
        manual_total_entries=len(manual),
        total_entries_difference=len(auto) - len(manual),
        auto_alternation_pct=auto_alt,
        manual_alternation_pct=manual_alt,
        alternation_difference_pct=auto_alt - manual_alt,
    )


# --------------------------------------------------------------------------
# EPM totals
# --------------------------------------------------------------------------

@dataclass
class MetricDifference:
    """One metric measured both ways; ``difference = auto - manual``."""
    name: str
    auto: float
    manual: float
    difference: float
    abs_difference: float
    pct_difference: float   # of the manual value; nan when it is 0


EPM_METRICS = (
    "entries_to_open_arms", "entries_to_closed_arms", "total_arm_entries",
    "time_in_open_arms_s", "time_in_closed_arms_s",
    "percent_time_in_open", "percent_entries_to_open",
)


def _field(source: Union[Mapping, object], name: str) -> Optional[float]:
    value = (
        source.get(name) if isinstance(source, Mapping)
        else getattr(source, name, None)
    )
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _metric_difference(name: str, auto: float, manual: float) -> MetricDifference:
    difference = auto - manual
    return MetricDifference(
        name=name, auto=auto, manual=manual, difference=difference,
        abs_difference=abs(difference),
        pct_difference=difference / manual * 100.0 if manual else NAN,
    )


def epm_totals_from_entries(entries: Sequence[Tuple[str, float]]) -> Dict[str, float]:
    """Entry counts and percentages from manual EPM entries (open vs closed)."""
    opens = sum(epm_arm_class(arm) == "open" for arm, _ in entries)
    closed = len(entries) - opens
    return {
        "entries_to_open_arms": float(opens),
        "entries_to_closed_arms": float(closed),
        "total_arm_entries": float(len(entries)),
        "percent_entries_to_open": (
            opens / len(entries) * 100.0 if entries else 0.0
        ),
    }


def compare_epm(
    auto_result_fields: Union[Mapping, object],
    manual_totals: Union[Mapping, object],
) -> Dict[str, MetricDifference]:
    """Per-video differences in EPM totals, for the metrics both sides have.

    ``auto_result_fields`` is an ``EPMAnalysisResult`` (or any mapping or
    object with the same field names) and ``manual_totals`` a mapping with
    some of ``entries_to_open_arms``, ``entries_to_closed_arms``,
    ``total_arm_entries``, ``time_in_open_arms_s``, ``time_in_closed_arms_s``,
    ``percent_time_in_open`` and ``percent_entries_to_open``. The total and
    the entry percentage are derived from the open/closed counts when only
    those are given. Metrics missing on either side are omitted.
    """
    def complete(source):
        values = {name: _field(source, name) for name in EPM_METRICS}
        opens = values["entries_to_open_arms"]
        closed = values["entries_to_closed_arms"]
        if opens is not None and closed is not None:
            if values["total_arm_entries"] is None:
                values["total_arm_entries"] = opens + closed
            if values["percent_entries_to_open"] is None:
                total = opens + closed
                values["percent_entries_to_open"] = (
                    opens / total * 100.0 if total else 0.0
                )
        return values

    auto_values, manual_values = complete(auto_result_fields), complete(manual_totals)
    return {
        name: _metric_difference(name, auto_values[name], manual_values[name])
        for name in EPM_METRICS
        if auto_values[name] is not None and manual_values[name] is not None
    }


# --------------------------------------------------------------------------
# Frame-level labels and positions
# --------------------------------------------------------------------------

@dataclass
class FrameLabelAgreement:
    """Agreement between automated and manual frame labels."""
    kappa: KappaResult
    percent_agreement: float
    n_frames: int
    n_ignored: int


def frame_label_agreement(
    auto_labels: Sequence[Optional[str]],
    manual_labels: Sequence[Optional[str]],
    ignore: Iterable[str] = (UNDETECTED,),
) -> FrameLabelAgreement:
    """Cohen's kappa and percent agreement on frame-level zone labels.

    Frames where either label is in ``ignore`` (default ``"undetected"``) or
    is missing (``None``) are dropped before scoring; the number dropped is
    ``n_ignored``.
    """
    if len(auto_labels) != len(manual_labels):
        raise ValueError(
            "auto_labels and manual_labels must cover the same frames "
            f"({len(auto_labels)} != {len(manual_labels)})"
        )
    skipped = set(ignore)
    kept_auto, kept_manual = [], []
    for auto, manual in zip(auto_labels, manual_labels):
        if auto is None or manual is None or auto in skipped or manual in skipped:
            continue
        kept_auto.append(auto)
        kept_manual.append(manual)
    return FrameLabelAgreement(
        kappa=cohens_kappa(kept_auto, kept_manual),
        percent_agreement=percent_agreement(kept_auto, kept_manual),
        n_frames=len(kept_auto),
        n_ignored=len(auto_labels) - len(kept_auto),
    )


@dataclass
class PositionErrors:
    """Localization error of automated positions against manual points.

    Errors are Euclidean distances over frames where both have a point.
    ``*_cm`` fields are ``nan`` when ``scale_factor`` is 0 (uncalibrated).
    ``miss_rate`` is the share of manual points the tracker missed;
    ``false_detections`` counts frames where the tracker reported a position
    but the manual scorer recorded none (the animal not visible).
    """
    n_manual: int
    n_matched: int
    n_missed: int
    miss_rate: float
    false_detections: int
    errors_px: np.ndarray
    errors_cm: np.ndarray
    mean_px: float
    median_px: float
    p95_px: float
    mean_cm: float
    median_cm: float
    p95_cm: float
    scale_factor: float


def _valid_point(point) -> Point:
    if point is None:
        return None
    x, y = float(point[0]), float(point[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def position_errors(
    auto_xy: Sequence[Point],
    manual_xy: Sequence[Point],
    scale_factor: float = 0.0,
) -> PositionErrors:
    """Per-frame position error in pixels and centimetres.

    ``auto_xy`` and ``manual_xy`` are aligned per frame and hold ``(x, y)``
    or ``None`` (also nan coordinates) when there is no point. ``scale_factor``
    is the centimetres-per-pixel value the analysis used. Pass only frames the
    manual scorer actually reviewed.
    """
    if len(auto_xy) != len(manual_xy):
        raise ValueError(
            "auto_xy and manual_xy must be aligned per frame "
            f"({len(auto_xy)} != {len(manual_xy)})"
        )
    if scale_factor < 0 or not math.isfinite(scale_factor):
        raise ValueError("scale_factor must be a finite value >= 0")
    errors, missed, false_detections, n_manual = [], 0, 0, 0
    for auto_point, manual_point in zip(auto_xy, manual_xy):
        auto_point, manual_point = _valid_point(auto_point), _valid_point(manual_point)
        if manual_point is None:
            false_detections += auto_point is not None
            continue
        n_manual += 1
        if auto_point is None:
            missed += 1
            continue
        errors.append(math.hypot(
            auto_point[0] - manual_point[0], auto_point[1] - manual_point[1]
        ))
    errors_px = np.asarray(errors, dtype=float)
    calibrated = scale_factor > 0
    errors_cm = errors_px * scale_factor if calibrated else np.full(errors_px.shape, NAN)

    def stats(values):
        if not values.size or np.all(np.isnan(values)):
            return NAN, NAN, NAN
        return (float(values.mean()), float(np.median(values)),
                float(np.percentile(values, 95)))

    mean_px, median_px, p95_px = stats(errors_px)
    mean_cm, median_cm, p95_cm = stats(errors_cm)
    return PositionErrors(
        n_manual=n_manual,
        n_matched=len(errors),
        n_missed=missed,
        miss_rate=missed / n_manual if n_manual else NAN,
        false_detections=false_detections,
        errors_px=errors_px,
        errors_cm=errors_cm,
        mean_px=mean_px, median_px=median_px, p95_px=p95_px,
        mean_cm=mean_cm, median_cm=median_cm, p95_cm=p95_cm,
        scale_factor=float(scale_factor),
    )


# --------------------------------------------------------------------------
# Exported tracking data
# --------------------------------------------------------------------------

@dataclass
class TrackingData:
    """Contents of an exported ``<base>_tracking_data.csv``.

    ``frames`` are absolute video frame numbers (``round(timestamp * fps)``
    as the exporter writes them). ``positions`` hold ``(x, y)`` in source
    pixels or ``None`` for undetected frames.
    """
    frames: List[int] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    positions: List[Point] = field(default_factory=list)
    raw_zones: List[str] = field(default_factory=list)
    zones: List[str] = field(default_factory=list)
    visual_zones: List[str] = field(default_factory=list)
    detected: List[bool] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def fps(self) -> float:
        """Frame rate estimated from the frame and timestamp columns (0 if unknown)."""
        if len(self.frames) < 2:
            return 0.0
        frame_span = self.frames[-1] - self.frames[0]
        time_span = self.timestamps[-1] - self.timestamps[0]
        return frame_span / time_span if time_span > 0 else 0.0

    def entries(
        self,
        arm_labels: Iterable[str],
        center_label: str = "center",
        count_initial_arm: bool = True,
    ) -> EntryList:
        """Arm entries recounted from the exported ``zone`` column.

        Uses the same center-mediated rule as the analysis
        (``core.zone_events.analyze_center_mediated_entries``).
        """
        from core.zone_events import analyze_center_mediated_entries
        counted = analyze_center_mediated_entries(
            self.zones, self.timestamps, arm_labels=set(arm_labels),
            center_label=center_label, count_initial_arm=count_initial_arm,
        )
        return list(counted.entries)


_TRACKING_REQUIRED = ("frame", "timestamp", "x", "y", "zone")


def load_tracking_csv(path) -> TrackingData:
    """Read an exported ``<base>_tracking_data.csv``.

    Columns (see ``output.DataExporter.export_to_csv``): ``frame, timestamp,
    x, y, raw_zone, zone, visual_zone, detected, confidence, failure_reason``.
    Empty ``x``/``y`` cells (undetected frames) load as ``None``. The TST
    motion-evidence file is a different format and is rejected here.
    """
    data = TrackingData()
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = [
            column for column in _TRACKING_REQUIRED
            if column not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(
                f"{path}: not a tracking data file (missing columns {missing})"
            )
        for line, row in enumerate(reader, start=2):
            try:
                data.frames.append(int(row["frame"]))
                data.timestamps.append(float(row["timestamp"]))
                x, y = (row["x"] or "").strip(), (row["y"] or "").strip()
                data.positions.append((float(x), float(y)) if x and y else None)
                data.confidences.append(float(row.get("confidence") or 0.0))
            except ValueError as error:
                raise ValueError(f"{path}, line {line}: {error}") from None
            zone = row["zone"]
            data.zones.append(zone)
            data.raw_zones.append(row.get("raw_zone") or zone)
            data.visual_zones.append(row.get("visual_zone") or zone)
            detected = (row.get("detected") or "").strip()
            data.detected.append(
                detected == "1" if detected else data.positions[-1] is not None
            )
            data.failure_reasons.append(row.get("failure_reason") or "")
    return data


__all__ = [
    "AgreementSummary", "EditDistanceResult", "EventMatchResult",
    "FrameLabelAgreement", "MetricDifference", "PositionErrors",
    "TrackingData", "YMazeComparison", "compare_epm", "compare_y_maze",
    "epm_arm_class", "epm_totals_from_entries", "frame_label_agreement",
    "load_entry_annotations", "load_tracking_csv", "match_events",
    "normalize_arm", "normalize_epm_arm", "normalize_y_maze_arm",
    "position_errors", "sequence_edit_distance", "summarize_agreement",
    "summarize_metrics",
]
