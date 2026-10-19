"""Qt-free logic of the tracking accuracy spot-check.

A scorer is shown N randomly chosen frames of a video *without* the automated
position and clicks on the animal's center (or marks it not visible). The
clicks are compared with the exported ``<base>_tracking_data.csv`` through
:func:`core.positional_validation.position_errors`. The sample is drawn from a
fixed seed, shown in the dialog and saved with the results, so it can be
reproduced.

Outputs, written next to each other:

``<base>_position_spotcheck.csv``
    One row per reviewed frame: ``sample_number, frame, timestamp_s, visible,
    manual_x, manual_y, auto_x, auto_y, error_px, error_cm, outcome,
    scale_factor``. Positions are source pixels. ``outcome`` is ``matched``,
    ``missed`` (animal visible, tracker found nothing), ``false_detection``
    (animal not visible, tracker reported a position), ``correct_absent`` or
    ``unreadable`` (the frame could not be decoded; not counted).
``<base>_position_spotcheck_summary.json``
    The counts and error statistics, the seed and frames sampled, and the
    provenance (scorer, date, application version, input files).
"""
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from core.positional_validation import (PositionErrors, TrackingData,
                                        position_errors)
from core.version import APP_NAME, APP_VERSION

DEFAULT_SAMPLE_SIZE = 50
DEFAULT_SEED = 2024

STATUS_PENDING = "pending"
STATUS_CLICKED = "clicked"
STATUS_NOT_VISIBLE = "not_visible"
STATUS_UNREADABLE = "unreadable"

TRACKING_SUFFIX = "_tracking_data.csv"
SUMMARY_SUFFIX = "_summary.json"
SPOTCHECK_SUFFIX = "_position_spotcheck.csv"
SPOTCHECK_SUMMARY_SUFFIX = "_position_spotcheck_summary.json"

SPOTCHECK_COLUMNS = (
    "sample_number", "frame", "timestamp_s", "visible", "manual_x",
    "manual_y", "auto_x", "auto_y", "error_px", "error_cm", "outcome",
    "scale_factor",
)

Point = Optional[Tuple[float, float]]


def sample_frames(frames: Sequence[int], count: int, seed: int) -> List[int]:
    """``count`` distinct frames drawn at random from ``frames``.

    The same ``seed`` and frame set always give the same sample (in the same
    order). At most ``len(frames)`` frames are returned.
    """
    pool = sorted(set(int(frame) for frame in frames))
    count = max(0, min(int(count), len(pool)))
    return random.Random(int(seed)).sample(pool, count)


def read_scale_factor(summary_json) -> float:
    """cm per pixel the run used (0 when uncalibrated or unreadable)."""
    try:
        with Path(summary_json).open(encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, ValueError):
        return 0.0
    candidates = (
        (data.get("metadata") or {}).get("scale_factor"),
        (data.get("run_configuration") or {}).get("scale_factor"),
    )
    for value in candidates:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0:
            return value
    return 0.0


@dataclass
class RunVideo:
    """One analysed video found in a run folder."""
    base_name: str
    tracking_csv: Path
    summary_json: Optional[Path]
    source_video: Optional[Path]


def scan_run_folder(run_dir) -> List[RunVideo]:
    """Videos of a run folder that have an exported tracking CSV.

    The source video comes from ``manifest.json`` when it is there and exists.
    """
    run_dir = Path(run_dir)
    sources: Dict[str, Path] = {}
    try:
        manifest = json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8"))
        for entry in manifest.get("videos", []):
            source = Path(str(entry.get("source", "")))
            folder = Path(str(entry.get("output_folder", ""))).name
            for key in (folder, source.stem):
                if key:
                    sources.setdefault(key, source)
    except (OSError, ValueError, AttributeError):
        pass
    found: List[RunVideo] = []
    for csv_path in sorted(run_dir.rglob(f"*{TRACKING_SUFFIX}")):
        base = csv_path.name[:-len(TRACKING_SUFFIX)]
        summary = csv_path.with_name(f"{base}{SUMMARY_SUFFIX}")
        source = sources.get(base)
        found.append(RunVideo(
            base, csv_path, summary if summary.is_file() else None,
            source if source is not None and source.is_file() else None,
        ))
    return found


def base_name_of(tracking_csv, video_path=None) -> str:
    """``<base>`` of a ``<base>_tracking_data.csv`` (else the video's stem)."""
    name = Path(tracking_csv).name
    if name.endswith(TRACKING_SUFFIX):
        return name[:-len(TRACKING_SUFFIX)]
    return Path(video_path).stem if video_path else Path(tracking_csv).stem


@dataclass
class SpotcheckSample:
    """One sampled frame and what the scorer said about it."""
    number: int
    frame: int
    timestamp_s: float
    auto_xy: Point
    status: str = STATUS_PENDING
    manual_xy: Point = None

    @property
    def reviewed(self) -> bool:
        return self.status in (STATUS_CLICKED, STATUS_NOT_VISIBLE)

    @property
    def outcome(self) -> str:
        if self.status == STATUS_UNREADABLE:
            return "unreadable"
        if self.status == STATUS_PENDING:
            return "pending"
        if self.manual_xy is None:
            return "false_detection" if self.auto_xy is not None else "correct_absent"
        return "missed" if self.auto_xy is None else "matched"


class SpotcheckSession:
    """The sample, the answers and the statistics of one spot-check."""

    def __init__(
        self,
        tracking: TrackingData,
        count: int = DEFAULT_SAMPLE_SIZE,
        seed: int = DEFAULT_SEED,
        scale_factor: float = 0.0,
    ):
        if not len(tracking):
            raise ValueError("The tracking data has no frames")
        self.seed = int(seed)
        self.scale_factor = float(scale_factor)
        self.requested = int(count)
        by_frame = {
            frame: (timestamp, position)
            for frame, timestamp, position in zip(
                tracking.frames, tracking.timestamps, tracking.positions)
        }
        self.frames = sample_frames(by_frame, count, seed)
        self.samples: List[SpotcheckSample] = [
            SpotcheckSample(number, frame, by_frame[frame][0],
                            by_frame[frame][1])
            for number, frame in enumerate(self.frames, start=1)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def answer_click(self, index: int, x: float, y: float) -> None:
        sample = self.samples[index]
        sample.status, sample.manual_xy = STATUS_CLICKED, (float(x), float(y))

    def answer_not_visible(self, index: int) -> None:
        sample = self.samples[index]
        sample.status, sample.manual_xy = STATUS_NOT_VISIBLE, None

    def mark_unreadable(self, index: int) -> None:
        sample = self.samples[index]
        sample.status, sample.manual_xy = STATUS_UNREADABLE, None

    def first_pending(self) -> Optional[int]:
        for index, sample in enumerate(self.samples):
            if sample.status == STATUS_PENDING:
                return index
        return None

    @property
    def is_complete(self) -> bool:
        return self.first_pending() is None

    def reviewed_samples(self) -> List[SpotcheckSample]:
        return [sample for sample in self.samples if sample.reviewed]

    def errors(self) -> PositionErrors:
        """``position_errors`` over the frames the scorer reviewed."""
        reviewed = self.reviewed_samples()
        return position_errors(
            [sample.auto_xy for sample in reviewed],
            [sample.manual_xy for sample in reviewed],
            self.scale_factor,
        )

    # --------------------------------------------------------------- outputs
    def summary(self) -> Dict[str, object]:
        errors = self.errors()
        reviewed = self.reviewed_samples()

        def number(value):
            return None if value is None or not math.isfinite(value) else float(value)

        return {
            "seed": self.seed,
            "requested_samples": self.requested,
            "sampled_frames": list(self.frames),
            "reviewed": len(reviewed),
            "unreadable": sum(
                s.status == STATUS_UNREADABLE for s in self.samples),
            "visible": errors.n_manual,
            "not_visible": len(reviewed) - errors.n_manual,
            "matched": errors.n_matched,
            "missed": errors.n_missed,
            "miss_rate": number(errors.miss_rate),
            "false_detections": errors.false_detections,
            "mean_error_px": number(errors.mean_px),
            "median_error_px": number(errors.median_px),
            "p95_error_px": number(errors.p95_px),
            "mean_error_cm": number(errors.mean_cm),
            "median_error_cm": number(errors.median_cm),
            "p95_error_cm": number(errors.p95_cm),
            "scale_factor": self.scale_factor,
            "calibrated": self.scale_factor > 0,
        }

    def write_csv(self, path) -> None:
        scale = self.scale_factor
        path = Path(path)
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(SPOTCHECK_COLUMNS)
            for sample in self.samples:
                if sample.status == STATUS_PENDING:
                    continue
                manual, auto = sample.manual_xy, sample.auto_xy
                error_px = ""
                error_cm = ""
                if manual is not None and auto is not None:
                    distance = math.hypot(auto[0] - manual[0], auto[1] - manual[1])
                    error_px = f"{distance:.4f}"
                    if scale > 0:
                        error_cm = f"{distance * scale:.5f}"
                writer.writerow([
                    sample.number, sample.frame, f"{sample.timestamp_s:.3f}",
                    1 if sample.status == STATUS_CLICKED else
                    (0 if sample.status == STATUS_NOT_VISIBLE else ""),
                    "" if manual is None else f"{manual[0]:.2f}",
                    "" if manual is None else f"{manual[1]:.2f}",
                    "" if auto is None else f"{auto[0]:.2f}",
                    "" if auto is None else f"{auto[1]:.2f}",
                    error_px, error_cm, sample.outcome, f"{scale:.8g}",
                ])
        temp.replace(path)

    def write_summary_json(
        self, path, provenance: Optional[Dict[str, object]] = None
    ) -> None:
        data = {
            "application": APP_NAME,
            "app_version": APP_VERSION,
            "date": datetime.now().isoformat(timespec="seconds"),
        }
        data.update(provenance or {})
        data["results"] = self.summary()
        path = Path(path)
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        temp.replace(path)

    def write_outputs(
        self, output_dir, base_name: str,
        provenance: Optional[Dict[str, object]] = None,
    ) -> Tuple[Path, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{base_name}{SPOTCHECK_SUFFIX}"
        json_path = output_dir / f"{base_name}{SPOTCHECK_SUMMARY_SUFFIX}"
        self.write_csv(csv_path)
        self.write_summary_json(json_path, provenance)
        return csv_path, json_path


@dataclass
class LoadedSpotcheck:
    """A ``_position_spotcheck.csv`` read back for pooling."""
    auto_xy: List[Point]
    manual_xy: List[Point]
    scale_factor: float
    frames: List[int]


def load_spotcheck_csv(path) -> LoadedSpotcheck:
    """Read a spot-check CSV; unreadable and pending rows are left out."""
    auto: List[Point] = []
    manual: List[Point] = []
    frames: List[int] = []
    scale = 0.0

    def point(x, y) -> Point:
        x, y = (x or "").strip(), (y or "").strip()
        return (float(x), float(y)) if x and y else None

    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = [
            column for column in ("frame", "visible", "manual_x", "manual_y",
                                  "auto_x", "auto_y")
            if column not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(
                f"{path}: not a position spot-check file "
                f"(missing columns {missing})")
        for line, row in enumerate(reader, start=2):
            if (row.get("outcome") or "") in ("unreadable", "pending"):
                continue
            try:
                frames.append(int(row["frame"]))
                auto.append(point(row["auto_x"], row["auto_y"]))
                manual.append(point(row["manual_x"], row["manual_y"]))
                value = (row.get("scale_factor") or "").strip()
                if value:
                    scale = float(value)
            except ValueError as error:
                raise ValueError(f"{path}, line {line}: {error}") from None
    return LoadedSpotcheck(auto, manual, scale, frames)


__all__ = [
    "DEFAULT_SAMPLE_SIZE", "DEFAULT_SEED", "LoadedSpotcheck", "RunVideo",
    "SpotcheckSample", "SpotcheckSession", "base_name_of",
    "load_spotcheck_csv", "read_scale_factor",
    "sample_frames", "scan_run_folder",
]
