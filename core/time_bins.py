"""Per-time-bin metrics computed from an in-memory analysis result.

A *bin* is a fixed slice of the analysed video: bin 1 covers the first
``bin_seconds`` of the analysis, bin 2 the next ``bin_seconds`` and so on.
Bins are counted from the first analysed frame (the analysis start), not from
the start of the video; every bin also carries the absolute video times
(``abs_start_s`` / ``abs_end_s``). The last bin is *partial* (``partial_bin``
= 1) when the analysis ends more than one frame before the bin would; its
``bin_end_s`` is where the analysis ended and ``bin_duration_s`` its real
length. Metrics of a partial bin cover that shorter duration, so rates
computed from them must use ``bin_duration_s``.

Every count is attributed to exactly one bin, so each metric summed over the
bins equals the whole-video total:

* A frame (time in a zone, undetected time, immobility) belongs to the bin of
  its timestamp.
* A distance step (the move from the previous frame to this one) belongs to
  the bin of the frame it ends in, and to the zone of that frame; this is the
  rule of ``Analyzer.finalize_result``.
* An arm entry or zone entry belongs to the bin of the frame where it
  happens.
* Y-maze alternation: a triplet of consecutive arm entries is counted in the
  bin where its THIRD entry occurs. A triplet that crosses a bin boundary
  therefore uses the entries of the previous bin as context but is scored once,
  in the later bin. ``alternations_possible`` per bin is the number of triplets
  that end in it, so it sums to ``possible_alternations`` of the whole video;
  ``alternation_pct`` of a bin is correct / possible of that bin (blank when it
  has no triplet).

The long format has one row per (bin, subject, metric): see
``TIME_BIN_HEADERS``. ``subject`` is the mouse for TST and blank otherwise.

Nothing here touches Qt or the disk, so parallel and sequential runs build
identical tables from identical results.
"""
import math
from dataclasses import dataclass, field
from bisect import bisect_left
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

from core.analysis_engine import compute_step_distances
from core.arena_zones import CENTER, PERIPHERY
from core.zone_events import UNDETECTED, extract_zone_transitions
from mazes.epm import EPMAnalysisResult
from mazes.freestyle import FreestyleAnalysisResult
from mazes.y_maze import YMazeAnalysisResult

Number = Union[int, float]

TIME_BIN_HEADERS = [
    "bin_index",
    "bin_start_s",
    "bin_end_s",
    "abs_start_s",
    "abs_end_s",
    "bin_duration_s",
    "partial_bin",
    "subject",
    "metric",
    "value",
    "unit",
]

Y_MAZE_ARMS = ("arm_a", "arm_b", "arm_c")
EPM_ARMS = ("open_arm_1", "open_arm_2", "closed_arm_1", "closed_arm_2")

_NO_ZONE = frozenset({"outside", UNDETECTED})
_EPSILON = 1e-9


@dataclass(frozen=True)
class TimeBin:
    index: int                # 1-based
    start_s: float            # relative to the analysis start
    end_s: float
    abs_start_s: float        # absolute video time
    abs_end_s: float
    duration_s: float
    partial: bool
    frame_count: int


@dataclass
class BinMetric:
    metric: str
    unit: str
    values: List[Optional[Number]]       # one per bin; None = not defined
    subject: str = ""

    def total(self) -> float:
        return float(sum(v for v in self.values if v is not None))


@dataclass
class TimeBinTable:
    bin_seconds: float
    fps: float
    bins: List[TimeBin] = field(default_factory=list)
    metrics: List[BinMetric] = field(default_factory=list)

    def metric(self, name: str, subject: str = "") -> Optional[BinMetric]:
        for item in self.metrics:
            if item.metric == name and item.subject == subject:
                return item
        return None

    def rows(self) -> Iterator[List[object]]:
        """Long-format rows, aligned with ``TIME_BIN_HEADERS``."""
        for position, time_bin in enumerate(self.bins):
            for item in self.metrics:
                yield [
                    time_bin.index,
                    _fmt(time_bin.start_s, 3),
                    _fmt(time_bin.end_s, 3),
                    _fmt(time_bin.abs_start_s, 3),
                    _fmt(time_bin.abs_end_s, 3),
                    _fmt(time_bin.duration_s, 3),
                    int(time_bin.partial),
                    item.subject,
                    item.metric,
                    _fmt_value(item.values[position]),
                    item.unit,
                ]


def _fmt(value: float, places: int) -> str:
    return f"{value:.{places}f}"


def _fmt_value(value: Optional[Number]) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return f"{value:.4f}"


class _Grid:
    """Maps frames (by offset from the analysis start) to bin numbers."""

    def __init__(self, timestamps: Sequence[float], fps: float,
                 bin_seconds: float):
        self.fps = fps
        self.bin_seconds = bin_seconds
        self.t0 = timestamps[0]
        self.offsets = [
            int(round((stamp - self.t0) * fps)) for stamp in timestamps
        ]
        self.total_frames = self.offsets[-1] + 1
        self.total_s = self.total_frames / fps
        self.count = self.bin_of_offset(self.offsets[-1]) + 1
        self.frame_bins = [self.bin_of_offset(o) for o in self.offsets]

    def bin_of_offset(self, offset: int) -> int:
        return int(math.floor(offset / self.fps / self.bin_seconds + _EPSILON))

    def bin_of_time(self, timestamp: float) -> int:
        offset = int(round((timestamp - self.t0) * self.fps))
        return min(max(0, self.bin_of_offset(offset)), self.count - 1)

    def bins(self) -> List[TimeBin]:
        frames_in_bin = [0] * self.count
        for index in self.frame_bins:
            frames_in_bin[index] += 1
        bins = []
        for index in range(self.count):
            nominal_end = (index + 1) * self.bin_seconds
            partial = nominal_end - self.total_s > 1.0 / self.fps + _EPSILON
            end = min(nominal_end, self.total_s) if partial else nominal_end
            start = index * self.bin_seconds
            bins.append(TimeBin(
                index=index + 1,
                start_s=start,
                end_s=end,
                abs_start_s=self.t0 + start,
                abs_end_s=self.t0 + end,
                duration_s=end - start,
                partial=partial,
                frame_count=frames_in_bin[index],
            ))
        return bins

    def zeros(self) -> List[float]:
        return [0.0] * self.count

    def int_zeros(self) -> List[int]:
        return [0] * self.count


def compute_time_bins(result, bin_seconds: float) -> Optional[TimeBinTable]:
    """Per-bin metrics of ``result``, or None when there is nothing to bin.

    ``result`` is a finished analysis result (metrics calculated, scale
    applied). Works for every paradigm: TST results (those with
    ``motion_evidence``) give immobile/mobile/uncertain seconds per mouse;
    positional results give distance and time per zone plus the paradigm's
    own metrics.
    """
    if not bin_seconds or bin_seconds <= 0:
        return None
    timestamps = list(getattr(result, "timestamps", []))
    fps = float(getattr(result, "fps", 0.0) or 0.0)
    if not timestamps or fps <= 0:
        return None
    grid = _Grid(timestamps, fps, float(bin_seconds))
    table = TimeBinTable(float(bin_seconds), fps, grid.bins())
    if hasattr(result, "motion_evidence"):
        _add_tst_metrics(table, grid, result)
    else:
        _add_positional_metrics(table, grid, result)
    return table


# ----------------------------------------------------------------------
# Positional paradigms
# ----------------------------------------------------------------------
def _add_positional_metrics(table: TimeBinTable, grid: _Grid, result) -> None:
    fps = grid.fps
    count = len(grid.frame_bins)
    positions = list(result.positions)[:count]
    labels = list(result.roi_labels)
    scale = result.scale_factor if result.scale_factor > 0 else 1.0
    unit = result.distance_unit

    steps = compute_step_distances(
        positions, labels, getattr(result, "jump_limit_px", None)
    ).steps

    zone_time: Dict[str, List[float]] = {}
    zone_distance: Dict[str, List[float]] = {}
    total_distance = grid.zeros()
    undetected = grid.zeros()
    for index in range(len(positions)):
        label = labels[index] if index < len(labels) else UNDETECTED
        bin_index = grid.frame_bins[index]
        if positions[index] is None or label == UNDETECTED:
            undetected[bin_index] += 1.0 / fps
            continue
        zone_time.setdefault(label, grid.zeros())[bin_index] += 1.0 / fps
        step = steps[index]
        if step is not None:
            distance = step * scale
            total_distance[bin_index] += distance
            zone_distance.setdefault(label, grid.zeros())[bin_index] += distance

    table.metrics.append(BinMetric("distance_total", unit, total_distance))
    for zone in sorted(zone_time):
        table.metrics.append(BinMetric(
            f"distance_in_{zone}", unit,
            zone_distance.get(zone, grid.zeros()),
        ))
    for zone in sorted(zone_time):
        table.metrics.append(BinMetric(f"time_in_{zone}_s", "s", zone_time[zone]))
    table.metrics.append(BinMetric("undetected_time_s", "s", undetected))

    if isinstance(result, YMazeAnalysisResult):
        _add_y_maze_metrics(table, grid, result)
    elif isinstance(result, EPMAnalysisResult):
        _add_epm_metrics(table, grid, result)
    elif isinstance(result, FreestyleAnalysisResult):
        _add_freestyle_metrics(table, grid, result)


def _entries_per_bin(
    grid: _Grid, entries: Sequence[Tuple[str, float]]
) -> Dict[str, List[int]]:
    per_arm: Dict[str, List[int]] = {}
    for arm, timestamp in entries:
        per_arm.setdefault(arm, grid.int_zeros())[
            grid.bin_of_time(timestamp)
        ] += 1
    return per_arm


def _add_entry_metrics(
    table: TimeBinTable, grid: _Grid, entries, arms: Sequence[str]
) -> Dict[str, List[int]]:
    per_arm = _entries_per_bin(grid, entries)
    total = grid.int_zeros()
    for counts in per_arm.values():
        total = [a + b for a, b in zip(total, counts)]
    table.metrics.append(BinMetric("entries_total", "count", total))
    for arm in arms:
        table.metrics.append(BinMetric(
            f"entries_{arm}", "count", per_arm.get(arm, grid.int_zeros())
        ))
    return per_arm


def _add_y_maze_metrics(
    table: TimeBinTable, grid: _Grid, result: YMazeAnalysisResult
) -> None:
    entries = list(result.arm_entries)
    _add_entry_metrics(table, grid, entries, Y_MAZE_ARMS)

    correct = grid.int_zeros()
    possible = grid.int_zeros()
    sequence = [arm for arm, _ in entries]
    for third in range(2, len(sequence)):
        bin_index = grid.bin_of_time(entries[third][1])
        possible[bin_index] += 1
        if len(set(sequence[third - 2:third + 1])) == 3:
            correct[bin_index] += 1
    table.metrics.append(BinMetric("alternations_correct", "count", correct))
    table.metrics.append(BinMetric("alternations_possible", "count", possible))
    table.metrics.append(BinMetric(
        "alternation_pct", "percent",
        [
            c / p * 100.0 if p else None
            for c, p in zip(correct, possible)
        ],
    ))


def _add_epm_metrics(
    table: TimeBinTable, grid: _Grid, result: EPMAnalysisResult
) -> None:
    entries = list(result.arm_entries)
    per_arm = _add_entry_metrics(table, grid, entries, EPM_ARMS)
    for kind in ("open", "closed"):
        counts = grid.int_zeros()
        for arm, arm_counts in per_arm.items():
            if kind in arm:
                counts = [a + b for a, b in zip(counts, arm_counts)]
        table.metrics.append(BinMetric(f"entries_{kind}_arms", "count", counts))


def _add_freestyle_metrics(table: TimeBinTable, grid: _Grid, result) -> None:
    labels = list(result.roi_labels)

    # Zone entries follow Freestyle.calculate_metrics: every transition into
    # a user zone, plus the first zone the animal was seen in.
    entries: Dict[str, List[int]] = {}
    for _, to_zone, index in extract_zone_transitions(labels):
        if to_zone not in _NO_ZONE and index < len(grid.frame_bins):
            entries.setdefault(to_zone, grid.int_zeros())[
                grid.frame_bins[index]
            ] += 1
    first = next(
        (i for i, z in enumerate(labels) if z not in _NO_ZONE), None
    )
    if first is not None and first < len(grid.frame_bins):
        entries.setdefault(labels[first], grid.int_zeros())[
            grid.frame_bins[first]
        ] += 1
    if entries:
        total = grid.int_zeros()
        for counts in entries.values():
            total = [a + b for a, b in zip(total, counts)]
        table.metrics.append(BinMetric("entries_total", "count", total))
        for zone in sorted(entries):
            table.metrics.append(
                BinMetric(f"entries_{zone}", "count", entries[zone])
            )

    # Immobility exists only when calibrated (uncalibrated: empty states).
    states = list(getattr(result, "immobile_states", []) or [])
    if states:
        immobile = grid.zeros()
        for index, state in enumerate(states[:len(grid.frame_bins)]):
            if state is True:
                immobile[grid.frame_bins[index]] += 1.0 / grid.fps
        table.metrics.append(BinMetric("immobility_time_s", "s", immobile))

    arena_labels = list(getattr(result, "_arena_labels", []) or [])
    if arena_labels:
        center = grid.zeros()
        periphery = grid.zeros()
        for index, label in enumerate(arena_labels[:len(grid.frame_bins)]):
            if label == CENTER:
                center[grid.frame_bins[index]] += 1.0 / grid.fps
            elif label == PERIPHERY:
                periphery[grid.frame_bins[index]] += 1.0 / grid.fps
        table.metrics.append(BinMetric("center_time_s", "s", center))
        table.metrics.append(BinMetric("periphery_time_s", "s", periphery))
        table.metrics.append(BinMetric(
            "thigmotaxis_time_pct", "percent",
            [
                100.0 * p / (c + p) if (c + p) > 0 else None
                for c, p in zip(center, periphery)
            ],
        ))


# ----------------------------------------------------------------------
# TST
# ----------------------------------------------------------------------
def _tst_window(result, mouse: str, length: int) -> Tuple[int, int]:
    """Frame range scored for ``mouse``: the same rule as ``TST``'s metrics."""
    start, end = 0, length
    windows = result.per_roi_times
    if windows and mouse in windows:
        target_start, target_end = windows[mouse]
        stamps = list(result.timestamps)[:length]
        start = bisect_left(stamps, target_start)
        end = bisect_left(stamps, target_end)
    return start, end


def _add_tst_metrics(table: TimeBinTable, grid: _Grid, result) -> None:
    mice = sorted(
        set(result.motion_evidence) | set(result.immobile_states)
    )
    frames = len(grid.frame_bins)
    for mouse in mice:
        states = list(result.immobile_states.get(mouse, []))
        length = min(len(states), frames)
        start, end = _tst_window(result, mouse, length)
        immobile = grid.zeros()
        mobile = grid.zeros()
        uncertain = grid.zeros()
        for index in range(start, min(end, length)):
            bin_index = grid.frame_bins[index]
            state = states[index]
            if state is True:
                immobile[bin_index] += 1.0 / grid.fps
            elif state is False:
                mobile[bin_index] += 1.0 / grid.fps
            else:
                uncertain[bin_index] += 1.0 / grid.fps
        table.metrics.append(BinMetric("immobile_s", "s", immobile, mouse))
        table.metrics.append(BinMetric("mobile_s", "s", mobile, mouse))
        table.metrics.append(BinMetric("uncertain_s", "s", uncertain, mouse))
        table.metrics.append(BinMetric(
            "immobility_pct", "percent",
            [
                100.0 * i / (i + m) if (i + m) > 0 else None
                for i, m in zip(immobile, mobile)
            ],
            mouse,
        ))
