"""Qt-free model and file formats for blind manual scoring.

The manual-scoring dialog (``gui.manual_scoring_dialog``) is a thin shell
around :class:`ManualScoreSession`: hotkeys add events or bouts at the video
frame on screen, the session keeps them editable with undo, autosaves to a
sidecar JSON and writes the annotation files the validation helpers read:

=============================  ==============================================
File                           Format
=============================  ==============================================
``<video>_manual_arm_entries.csv``    ``entry_number,arm,timestamp_s`` (the same
                                      columns as ``<base>_arm_entries.csv``;
                                      read by ``load_entry_annotations``)
``<video>_manual_epm_totals.csv``     ``metric,value`` EPM totals derived from
                                      the scored arm/center sequence (optional)
``<video>_manual_zone_intervals.csv`` open field center/periphery intervals
                                      (see :func:`write_zone_intervals_csv`)
``<video>_manual_tst_annotations.csv`` ``mouse_name,frame,state`` frame-wise
                                      (``TST_VALIDATION.md``; read by
                                      ``load_annotations``)
``<video>_manual_scoring_<paradigm>_<initials>.json``  autosave / provenance
=============================  ==============================================

Frames are absolute video frame indices and times are ``frame / fps``, the
same convention the analysis uses for its exported timestamps.
"""
import bisect
import copy
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from core.tst_validation import VALID_STATES as TST_VALID_STATES
from core.version import APP_NAME, APP_VERSION

PARADIGM_Y_MAZE = "y_maze"
PARADIGM_EPM = "epm"
PARADIGM_OPEN_FIELD = "open_field"
PARADIGM_TST = "tst"

PARADIGM_LABELS = {
    PARADIGM_Y_MAZE: "Y-Maze",
    PARADIGM_EPM: "Elevated Plus Maze",
    PARADIGM_OPEN_FIELD: "Freestyle / Open Field",
    PARADIGM_TST: "Tail Suspension Test",
}

CENTER = "center"
PERIPHERY = "periphery"

Y_ARMS = ("arm_a", "arm_b", "arm_c")
EPM_ARMS = ("open_arm_1", "open_arm_2", "closed_arm_1", "closed_arm_2")
TST_STATES = ("immobile_static", "passive_swing", "active_escape", "uncertain")

FILL_NONE = "none"
FILL_ACTIVE = "active_escape"

SIDECAR_SCHEMA_VERSION = 1
MAX_UNDO = 500

ARM_ENTRY_COLUMNS = ("entry_number", "arm", "timestamp_s")
ZONE_INTERVAL_COLUMNS = (
    "interval_number", "zone", "start_frame", "end_frame",
    "start_s", "end_s", "duration_s",
)
TST_ANNOTATION_COLUMNS = ("mouse_name", "frame", "state")


def allowed_event_labels(paradigm: str) -> Tuple[str, ...]:
    """Labels an event may carry in ``paradigm`` (TST uses bouts, not events)."""
    if paradigm == PARADIGM_Y_MAZE:
        return Y_ARMS + (CENTER,)
    if paradigm == PARADIGM_EPM:
        return EPM_ARMS + (CENTER,)
    if paradigm == PARADIGM_OPEN_FIELD:
        return (CENTER, PERIPHERY)
    return ()


def sanitize_initials(initials: str) -> str:
    """Letters and digits only, upper case; '' when nothing usable is left."""
    return re.sub(r"[^A-Za-z0-9]", "", str(initials or "")).upper()


@dataclass
class ScoreEvent:
    """A hotkey press: ``label`` (an arm, ``center`` or a zone) at ``frame``."""
    label: str
    frame: int


@dataclass
class Bout:
    """A TST state held for frames ``start``..``end`` (inclusive)."""
    mouse: str
    state: str
    start: int
    end: int


@dataclass
class ZoneInterval:
    """Open-field zone over frames ``start_frame``..``end_frame`` (inclusive).

    ``start_s = start_frame / fps`` and ``end_s = (end_frame + 1) / fps``, so
    ``duration_s = end_s - start_s`` is the time covered by those frames.
    """
    zone: str
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


# --------------------------------------------------------------------------
# Derived quantities (pure functions of the scored data)
# --------------------------------------------------------------------------

def zone_at(events: Sequence[ScoreEvent], frame: int,
            initial_zone: str = PERIPHERY) -> str:
    """The open-field zone in force at ``frame`` (events are time sorted)."""
    zone = initial_zone
    for event in events:
        if event.frame > frame:
            break
        if event.label in (CENTER, PERIPHERY):
            zone = event.label
    return zone


def zone_intervals(
    events: Sequence[ScoreEvent],
    initial_zone: str,
    fps: float,
    window_start_frame: int,
    window_end_frame: int,
) -> List[ZoneInterval]:
    """Center/periphery intervals within ``[window_start, window_end)`` frames.

    ``events`` are zone changes (``center`` / ``periphery``) sorted by frame;
    the zone before the first change is ``initial_zone``. Consecutive spans of
    the same zone are merged and zero-length spans dropped.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    start, end = int(window_start_frame), int(window_end_frame)
    if end <= start:
        return []
    zone = zone_at(events, start, initial_zone)
    boundaries: List[Tuple[int, str]] = [(start, zone)]
    for event in events:
        if event.label not in (CENTER, PERIPHERY):
            continue
        if event.frame <= start:
            continue
        if event.frame >= end:
            break
        if event.label != boundaries[-1][1]:
            boundaries.append((event.frame, event.label))
    intervals: List[ZoneInterval] = []
    for index, (first, label) in enumerate(boundaries):
        stop = boundaries[index + 1][0] if index + 1 < len(boundaries) else end
        if stop <= first:
            continue
        # Dropping a zero-length span (two changes on one frame) can leave
        # neighbours with the same label; merge them.
        if intervals and intervals[-1].zone == label:
            intervals[-1].end_frame = stop - 1
            intervals[-1].end_s = stop / fps
            continue
        intervals.append(ZoneInterval(
            label, first, stop - 1, first / fps, stop / fps
        ))
    return intervals


def zone_entries(events: Sequence[ScoreEvent], initial_zone: str,
                 window_start_frame: int, window_end_frame: int) -> int:
    """Number of transitions into the center inside the window."""
    intervals = zone_intervals(
        events, initial_zone, 1.0, window_start_frame, window_end_frame
    )
    return sum(
        1 for index, interval in enumerate(intervals)
        if interval.zone == CENTER and index > 0
    )


def epm_time_totals(
    events: Sequence[ScoreEvent], window_end_frame: int, fps: float
) -> Optional[Dict[str, float]]:
    """Time per region implied by the scored arm/center sequence.

    Each event starts a stay that lasts until the next event (or the window
    end). Time before the first event is unknown and left out. Returns
    ``None`` unless ``center`` was scored at least once: without center
    markers the time spent crossing the center would be credited to an arm.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not any(event.label == CENTER for event in events):
        return None
    open_s = closed_s = center_s = 0.0
    ordered = list(events)
    for index, event in enumerate(ordered):
        stop = (
            ordered[index + 1].frame if index + 1 < len(ordered)
            else int(window_end_frame)
        )
        seconds = max(0, stop - event.frame) / fps
        if event.label == CENTER:
            center_s += seconds
        elif event.label.startswith("open"):
            open_s += seconds
        else:
            closed_s += seconds
    arms = open_s + closed_s
    return {
        "time_in_open_arms_s": open_s,
        "time_in_closed_arms_s": closed_s,
        "time_in_center_s": center_s,
        "percent_time_in_open": open_s / arms * 100.0 if arms > 0 else 0.0,
    }


def tst_frame_labels(
    bouts: Sequence[Bout],
    mice: Iterable[str],
    window_start_frame: int,
    window_end_frame: int,
    fill_unmarked: str = FILL_NONE,
) -> Dict[Tuple[str, int], str]:
    """Expand bouts to ``{(mouse, frame): state}``.

    Later bouts overwrite earlier ones where they overlap. With
    ``fill_unmarked="active_escape"`` every frame of ``[window_start,
    window_end)`` that no bout covers is labelled ``active_escape``, which
    turns "hold a key while immobile" into a complete frame-wise annotation.
    """
    labels: Dict[Tuple[str, int], str] = {}
    for bout in bouts:
        for frame in range(bout.start, bout.end + 1):
            labels[(bout.mouse, frame)] = bout.state
    if fill_unmarked == FILL_ACTIVE:
        for mouse in mice:
            for frame in range(int(window_start_frame), int(window_end_frame)):
                labels.setdefault((mouse, frame), "active_escape")
    return labels


# --------------------------------------------------------------------------
# File names and writers
# --------------------------------------------------------------------------

def output_names(video_stem: str, paradigm: str) -> Dict[str, str]:
    """File names (not paths) the paradigm's save writes, by role."""
    if paradigm in (PARADIGM_Y_MAZE, PARADIGM_EPM):
        names = {"arm_entries": f"{video_stem}_manual_arm_entries.csv"}
        if paradigm == PARADIGM_EPM:
            names["epm_totals"] = f"{video_stem}_manual_epm_totals.csv"
        return names
    if paradigm == PARADIGM_OPEN_FIELD:
        return {"zone_intervals": f"{video_stem}_manual_zone_intervals.csv"}
    if paradigm == PARADIGM_TST:
        return {"tst_annotations": f"{video_stem}_manual_tst_annotations.csv"}
    raise ValueError(f"Unknown paradigm: {paradigm!r}")


def sidecar_name(video_stem: str, paradigm: str, initials: str) -> str:
    return (
        f"{video_stem}_manual_scoring_{paradigm}_"
        f"{sanitize_initials(initials) or 'ANON'}.json"
    )


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence]) -> None:
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    temp.replace(path)


def write_arm_entries_csv(path, entries: Sequence[ScoreEvent], fps: float) -> int:
    """Write arm entries (``center`` markers excluded); returns the count."""
    arms = [event for event in entries if event.label != CENTER]
    _write_csv(
        path, ARM_ENTRY_COLUMNS,
        ((number, event.label, f"{event.frame / fps:.3f}")
         for number, event in enumerate(arms, start=1)),
    )
    return len(arms)


def write_epm_totals_csv(path, totals: Dict[str, float]) -> None:
    _write_csv(
        path, ("metric", "value"),
        ((name, f"{value:.4f}") for name, value in totals.items()),
    )


def load_epm_totals(path) -> Dict[str, float]:
    """Read ``metric,value`` rows (any ``EPM_METRICS`` names) as a mapping."""
    totals: Dict[str, float] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = [str(name).strip().lower() for name in (reader.fieldnames or [])]
        if "metric" not in fields or "value" not in fields:
            raise ValueError(
                f"{path}: expected columns 'metric' and 'value', "
                f"found {reader.fieldnames}"
            )
        for line, raw in enumerate(reader, start=2):
            row = {str(k).strip().lower(): v for k, v in raw.items() if k}
            name = str(row.get("metric") or "").strip()
            if not name:
                continue
            try:
                value = float(row["value"])
            except (TypeError, ValueError):
                raise ValueError(
                    f"{path}, line {line}: invalid value {row.get('value')!r}"
                ) from None
            totals[name] = value
    return totals


def write_zone_intervals_csv(path, intervals: Sequence[ZoneInterval]) -> None:
    """Open-field intervals: ``interval_number,zone,start_frame,end_frame,
    start_s,end_s,duration_s``.

    Frames are inclusive; ``start_s = start_frame / fps`` and ``end_s =
    (end_frame + 1) / fps``. The intervals tile the scoring window.
    """
    _write_csv(
        path, ZONE_INTERVAL_COLUMNS,
        ((number, item.zone, item.start_frame, item.end_frame,
          f"{item.start_s:.4f}", f"{item.end_s:.4f}",
          f"{item.duration_s:.4f}")
         for number, item in enumerate(intervals, start=1)),
    )


def load_zone_intervals(path) -> List[ZoneInterval]:
    """Read a ``_manual_zone_intervals.csv`` back (see the writer)."""
    intervals: List[ZoneInterval] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = [
            column for column in ("zone", "start_frame", "end_frame",
                                  "start_s", "end_s")
            if column not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(
                f"{path}: not a zone interval file (missing columns {missing})"
            )
        for line, row in enumerate(reader, start=2):
            try:
                intervals.append(ZoneInterval(
                    zone=str(row["zone"]).strip().lower(),
                    start_frame=int(row["start_frame"]),
                    end_frame=int(row["end_frame"]),
                    start_s=float(row["start_s"]),
                    end_s=float(row["end_s"]),
                ))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}, line {line}: {error}") from None
    return intervals


def center_time_s(intervals: Iterable[ZoneInterval]) -> float:
    """Total seconds spent in the center over ``intervals``."""
    return sum(item.duration_s for item in intervals if item.zone == CENTER)


def zone_frame_labels(
    intervals: Iterable[ZoneInterval],
) -> Dict[int, str]:
    """``{frame: zone}`` over the frames the intervals cover."""
    labels: Dict[int, str] = {}
    for item in intervals:
        for frame in range(item.start_frame, item.end_frame + 1):
            labels[frame] = item.zone
    return labels


def write_tst_annotations_csv(
    path, labels: Dict[Tuple[str, int], str]
) -> int:
    """Write ``mouse_name,frame,state`` rows sorted by mouse and frame."""
    for state in set(labels.values()):
        if state not in TST_VALID_STATES:
            raise ValueError(f"Unsupported TST annotation state: {state}")
    rows = [(mouse, frame, state)
            for (mouse, frame), state in sorted(labels.items())]
    _write_csv(path, TST_ANNOTATION_COLUMNS, rows)
    return len(rows)


# --------------------------------------------------------------------------
# The session
# --------------------------------------------------------------------------

class ManualScoreSession:
    """Everything one scorer has marked on one video, with undo."""

    def __init__(
        self,
        paradigm: str,
        video_name: str,
        fps: float,
        total_frames: int,
        scorer_initials: str,
        mice: Sequence[str] = ("mouse_1",),
        roi_file: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        if paradigm not in PARADIGM_LABELS:
            raise ValueError(f"Unknown paradigm: {paradigm!r}")
        if not (fps > 0 and math.isfinite(fps)):
            raise ValueError("fps must be positive")
        if total_frames <= 0:
            raise ValueError("total_frames must be positive")
        self.paradigm = paradigm
        self.video_name = str(video_name)
        self.fps = float(fps)
        self.total_frames = int(total_frames)
        self.scorer_initials = sanitize_initials(scorer_initials)
        self.mice: List[str] = [str(m) for m in mice] or ["mouse_1"]
        self.roi_file = roi_file
        self.output_dir = output_dir
        self.events: List[ScoreEvent] = []
        self.bouts: List[Bout] = []
        self.initial_zone = PERIPHERY
        self.window_start_s = 0.0
        self.window_end_s: Optional[float] = None
        self.fill_unmarked = FILL_NONE
        self.date_started = datetime.now().isoformat(timespec="seconds")
        self.date_saved: Optional[str] = None
        self.saved_files: List[str] = []
        self.dirty = False            # scored data not yet written to the CSVs
        self._undo: List[Tuple[str, List[ScoreEvent], List[Bout]]] = []
        self._open_bout: Optional[Bout] = None

    # ------------------------------------------------------------- basics
    @property
    def video_stem(self) -> str:
        return Path(self.video_name).stem

    def time_of(self, frame: int) -> float:
        return frame / self.fps

    def window_frames(self) -> Tuple[int, int]:
        """The scoring window as ``(start_frame, end_frame_exclusive)``."""
        start = int(round(self.window_start_s * self.fps))
        end = (
            self.total_frames if self.window_end_s is None
            else int(round(self.window_end_s * self.fps))
        )
        start = min(max(0, start), self.total_frames - 1)
        end = min(max(start + 1, end), self.total_frames)
        return start, end

    def set_window(self, start_s: float, end_s: Optional[float]) -> None:
        start_s = max(0.0, float(start_s))
        if end_s is not None and end_s <= start_s:
            raise ValueError("The window must end after it starts")
        self.window_start_s, self.window_end_s = start_s, end_s
        self.dirty = True

    def _snapshot(self, description: str) -> None:
        self._undo.append(
            (description, copy.deepcopy(self.events), copy.deepcopy(self.bouts))
        )
        if len(self._undo) > MAX_UNDO:
            del self._undo[0]

    def _check_frame(self, frame: int) -> int:
        frame = int(frame)
        if not 0 <= frame < self.total_frames:
            raise ValueError(
                f"Frame {frame} is outside the video (0..{self.total_frames - 1})"
            )
        return frame

    # ------------------------------------------------------------- events
    def add_event(self, label: str, frame: int) -> ScoreEvent:
        if label not in allowed_event_labels(self.paradigm):
            raise ValueError(
                f"{label!r} is not a {PARADIGM_LABELS[self.paradigm]} label"
            )
        frame = self._check_frame(frame)
        self._snapshot(f"add {label} at frame {frame}")
        event = ScoreEvent(label, frame)
        keys = [item.frame for item in self.events]
        self.events.insert(bisect.bisect_right(keys, frame), event)
        self.dirty = True
        return event

    def zone_now(self, frame: int) -> str:
        return zone_at(self.events, frame, self.initial_zone)

    def toggle_zone(self, frame: int) -> ScoreEvent:
        """Open field: switch to the other zone at ``frame``."""
        new = PERIPHERY if self.zone_now(frame) == CENTER else CENTER
        return self.add_event(new, frame)

    def delete_events(self, indices: Iterable[int]) -> int:
        chosen = sorted(
            {int(i) for i in indices if 0 <= int(i) < len(self.events)}
        )
        if not chosen:
            return 0
        self._snapshot(f"delete {len(chosen)} event(s)")
        for index in reversed(chosen):
            del self.events[index]
        self.dirty = True
        return len(chosen)

    # -------------------------------------------------------------- bouts
    def begin_bout(self, mouse: str, state: str, frame: int) -> Bout:
        if self.paradigm != PARADIGM_TST:
            raise ValueError("Bouts are only scored for the TST")
        if state not in TST_STATES:
            raise ValueError(f"Unsupported TST state: {state!r}")
        frame = self._check_frame(frame)
        self.end_bout()
        self._snapshot(f"{state} from frame {frame}")
        bout = Bout(str(mouse), state, frame, frame)
        self.bouts.append(bout)
        self._open_bout = bout
        self.dirty = True
        return bout

    @property
    def open_bout(self) -> Optional[Bout]:
        return self._open_bout

    def extend_bout(self, frame: int) -> None:
        """While a key is held: every frame up to ``frame`` gets the state."""
        bout = self._open_bout
        if bout is not None and frame > bout.end:
            bout.end = min(int(frame), self.total_frames - 1)

    def end_bout(self) -> Optional[Bout]:
        bout, self._open_bout = self._open_bout, None
        return bout

    def delete_bouts(self, indices: Iterable[int]) -> int:
        self.end_bout()
        chosen = sorted(
            {int(i) for i in indices if 0 <= int(i) < len(self.bouts)}
        )
        if not chosen:
            return 0
        self._snapshot(f"delete {len(chosen)} bout(s)")
        for index in reversed(chosen):
            del self.bouts[index]
        self.dirty = True
        return len(chosen)

    # --------------------------------------------------------- undo / clear
    def can_undo(self) -> bool:
        return bool(self._undo)

    def undo(self) -> Optional[str]:
        """Revert the last change; returns what was undone (None if nothing)."""
        if not self._undo:
            return None
        self.end_bout()
        description, self.events, self.bouts = self._undo.pop()
        self.dirty = True
        return description

    def clear(self) -> None:
        self.end_bout()
        if self.events or self.bouts:
            self._snapshot("clear all")
        self.events, self.bouts = [], []
        self.dirty = True

    # -------------------------------------------------------------- summary
    def event_count(self) -> int:
        """Arm entries (positional mazes) / zone changes / bouts."""
        if self.paradigm == PARADIGM_TST:
            return len(self.bouts)
        if self.paradigm in (PARADIGM_Y_MAZE, PARADIGM_EPM):
            return sum(1 for event in self.events if event.label != CENTER)
        return len(self.events)

    # --------------------------------------------------------------- outputs
    def planned_outputs(self, output_dir) -> Dict[str, Path]:
        base = Path(output_dir)
        return {
            role: base / name
            for role, name in output_names(self.video_stem, self.paradigm).items()
        }

    def write_outputs(self, output_dir) -> List[Path]:
        """Write the annotation files for this paradigm; returns the paths."""
        self.end_bout()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        planned = self.planned_outputs(output_dir)
        written: List[Path] = []
        start, end = self.window_frames()
        if self.paradigm in (PARADIGM_Y_MAZE, PARADIGM_EPM):
            write_arm_entries_csv(planned["arm_entries"], self.events, self.fps)
            written.append(planned["arm_entries"])
            if self.paradigm == PARADIGM_EPM:
                totals = epm_time_totals(self.events, end, self.fps)
                if totals is not None:
                    write_epm_totals_csv(planned["epm_totals"], totals)
                    written.append(planned["epm_totals"])
                else:
                    planned["epm_totals"].unlink(missing_ok=True)
        elif self.paradigm == PARADIGM_OPEN_FIELD:
            write_zone_intervals_csv(
                planned["zone_intervals"],
                zone_intervals(self.events, self.initial_zone, self.fps,
                               start, end),
            )
            written.append(planned["zone_intervals"])
        else:
            write_tst_annotations_csv(
                planned["tst_annotations"],
                tst_frame_labels(self.bouts, self.mice, start, end,
                                 self.fill_unmarked),
            )
            written.append(planned["tst_annotations"])
        self.date_saved = datetime.now().isoformat(timespec="seconds")
        self.saved_files = [str(path) for path in written]
        self.dirty = False
        return written

    # --------------------------------------------------------------- sidecar
    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "application": APP_NAME,
            "app_version": APP_VERSION,
            "paradigm": self.paradigm,
            "video_name": self.video_name,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "scorer_initials": self.scorer_initials,
            "date_started": self.date_started,
            "date_saved": self.date_saved,
            "date": self.date_started[:10],
            "blind": True,
            "roi_file": self.roi_file,
            "window": {"start_s": self.window_start_s, "end_s": self.window_end_s},
            "initial_zone": self.initial_zone,
            "events": [
                {"label": e.label, "frame": e.frame,
                 "timestamp_s": round(self.time_of(e.frame), 4)}
                for e in self.events
            ],
            "mice": list(self.mice),
            "fill_unmarked": self.fill_unmarked,
            "bouts": [
                {"mouse": b.mouse, "state": b.state,
                 "start": b.start, "end": b.end}
                for b in self.bouts
            ],
            "saved_files": list(self.saved_files),
            "csv_up_to_date": not self.dirty,
        }

    def save_sidecar(self, path) -> None:
        path = Path(path)
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        temp.replace(path)

    def load_sidecar(self, path) -> bool:
        """Restore a previous session; False if it does not belong to this one.

        A sidecar is accepted only for the same paradigm, scorer, video name
        and frame count, so it can never be mixed into another recording.
        """
        with Path(path).open(encoding="utf-8") as file:
            data = json.load(file)
        if (
            data.get("paradigm") != self.paradigm
            or data.get("video_name") != self.video_name
            or int(data.get("total_frames", -1)) != self.total_frames
            or str(data.get("scorer_initials", "")) != self.scorer_initials
        ):
            return False
        allowed = allowed_event_labels(self.paradigm)
        events = []
        for raw in data.get("events", []):
            label, frame = str(raw["label"]), int(raw["frame"])
            if label in allowed and 0 <= frame < self.total_frames:
                events.append(ScoreEvent(label, frame))
        events.sort(key=lambda event: event.frame)    # stable
        bouts = []
        for raw in data.get("bouts", []):
            state = str(raw["state"])
            start, end = int(raw["start"]), int(raw["end"])
            if state in TST_STATES and 0 <= start <= end < self.total_frames:
                bouts.append(Bout(str(raw["mouse"]), state, start, end))
        self.events, self.bouts = events, bouts
        self.mice = [str(m) for m in data.get("mice", self.mice)] or self.mice
        self.initial_zone = (
            data.get("initial_zone") if data.get("initial_zone") in
            (CENTER, PERIPHERY) else PERIPHERY
        )
        window = data.get("window") or {}
        self.window_start_s = float(window.get("start_s") or 0.0)
        end_s = window.get("end_s")
        self.window_end_s = None if end_s is None else float(end_s)
        self.fill_unmarked = (
            data.get("fill_unmarked")
            if data.get("fill_unmarked") in (FILL_NONE, FILL_ACTIVE)
            else FILL_NONE
        )
        self.date_started = str(data.get("date_started") or self.date_started)
        self.date_saved = data.get("date_saved")
        self.saved_files = [str(p) for p in data.get("saved_files", [])]
        self.dirty = not data.get("csv_up_to_date", False)
        self._undo = []
        self._open_bout = None
        return True


__all__ = [
    "Bout", "CENTER", "EPM_ARMS", "ManualScoreSession", "PARADIGM_EPM",
    "PARADIGM_LABELS", "PARADIGM_OPEN_FIELD", "PARADIGM_TST",
    "PARADIGM_Y_MAZE", "PERIPHERY", "ScoreEvent", "TST_STATES", "Y_ARMS",
    "ZoneInterval", "allowed_event_labels", "center_time_s",
    "epm_time_totals", "load_epm_totals", "load_zone_intervals",
    "output_names", "sanitize_initials", "sidecar_name", "tst_frame_labels",
    "write_arm_entries_csv", "write_epm_totals_csv",
    "write_tst_annotations_csv", "write_zone_intervals_csv",
    "zone_at", "zone_entries", "zone_frame_labels", "zone_intervals",
]
