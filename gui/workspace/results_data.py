"""
gui/workspace/results_data.py
Reading a run folder for the Results tab, without Qt.

A run folder (see ``OUTPUTS.md``) holds ``manifest.json``, ``batch_summary.csv``,
optionally ``batch_time_bins.csv`` and one folder per video with ``data/``,
``plots/`` and ``media/``. :class:`RunData` loads the index of such a folder
(manifest schema 3, schema 2 and older best effort, or just the CSV when the
manifest is missing) and gives every row of the batch summary its
:class:`RunVideo`. The module also holds the data logic of the tab's charts
and of "Compare runs": :func:`time_bin_series` and :func:`diff_summaries`.

Everything is plain strings, floats and paths so it can be tested directly.
"""
import csv
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from core.project import ProjectError, read_manifest
from core.run_outputs import MANIFEST_FILENAME, RUN_PLAN_FILENAME

logger = logging.getLogger(__name__)

BATCH_SUMMARY = "batch_summary.csv"
BATCH_TIME_BINS = "batch_time_bins.csv"
FILENAME_COLUMN = "video_filename"
RUN_FOLDER_PREFIX = "analysis_run_"
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")
PLOT_SUFFIXES = (".png", ".jpg", ".jpeg")
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# Sections of ``*_summary.json`` that are settings, not results.
_SETTINGS_SECTIONS = frozenset(
    {"detection_parameters", "run_configuration"}
)
_MAX_FLATTEN_DEPTH = 3
_MAX_LIST_ITEMS = 6


class RunLoadError(ValueError):
    """The folder is not a readable run."""


def norm(path: os.PathLike) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
@dataclass
class Table:
    """A CSV as strings."""
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)

    def column(self, name: str) -> int:
        try:
            return self.headers.index(name)
        except ValueError:
            return -1

    def values(self, name: str) -> List[str]:
        index = self.column(name)
        return [] if index < 0 else [
            row[index] if index < len(row) else "" for row in self.rows
        ]


def read_table(path: os.PathLike) -> Table:
    """Read a CSV (UTF-8, optional BOM); short rows are padded."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return Table()
    headers = [cell.strip() for cell in rows[0]]
    width = len(headers)
    body = [
        (row + [""] * (width - len(row)))[:width] if len(row) != width else row
        for row in rows[1:] if any(cell.strip() for cell in row)
    ]
    return Table(headers, body)


def write_table(path: os.PathLike, headers: List[str], rows: Iterable[List[str]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def to_float(text: Any) -> Optional[float]:
    """A finite float, or None for blanks, ``N/A`` and anything else."""
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


# ----------------------------------------------------------------------
# One video of a run
# ----------------------------------------------------------------------
@dataclass
class RunVideo:
    name: str                         # source file name
    folder: Path
    source: Optional[Path] = None
    status: str = STATUS_COMPLETED
    metadata: Dict[str, str] = field(default_factory=dict)
    summary_json: Optional[Path] = None
    warnings: List[str] = field(default_factory=list)
    failure: str = ""

    def _files(self, sub: str, patterns: Iterable[str]) -> List[Path]:
        folder = self.folder / sub
        found: List[Path] = []
        try:
            for pattern in patterns:
                found.extend(sorted(folder.glob(pattern)))
        except OSError:
            return []
        return [path for path in found if path.is_file()]

    def plots(self) -> List[Path]:
        return self._files("plots", [f"*{s}" for s in PLOT_SUFFIXES])

    def media(self) -> List[Path]:
        return self._files("media", [f"*{s}" for s in VIDEO_SUFFIXES])

    def data_file(self, pattern: str) -> Optional[Path]:
        found = self._files("data", [pattern])
        return found[0] if found else None

    def tracking_csv(self) -> Optional[Path]:
        return self.data_file("*_tracking_data.csv")

    def evidence_csv(self) -> Optional[Path]:
        return self.data_file("*_tst_motion_evidence.csv")

    def time_bins_csv(self) -> Optional[Path]:
        return self.data_file("*_time_bins.csv")

    def load_summary(self) -> Dict[str, Any]:
        """The video's ``*_summary.json`` ({} when missing or unreadable)."""
        path = self.summary_json or self.data_file("*_summary.json")
        if path is None:
            return {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}


# ----------------------------------------------------------------------
# A run
# ----------------------------------------------------------------------
@dataclass
class RunData:
    run_dir: Path
    manifest: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 0
    paradigm: str = ""
    started_at: str = ""
    finished_at: str = ""
    cancelled: bool = False
    requested: int = 0
    sample_sheet: str = ""
    time_bin_seconds: float = 0.0
    videos: List[RunVideo] = field(default_factory=list)
    summary: Table = field(default_factory=Table)
    time_bins: Optional[Table] = None
    failures: List[Tuple[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    _row_videos: List[Optional[RunVideo]] = field(default_factory=list, repr=False)

    # -- loading ---------------------------------------------------------
    @classmethod
    def load(cls, run_dir: os.PathLike) -> "RunData":
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise RunLoadError(f"{run_dir} is not a folder")
        run = cls(run_dir=run_dir)
        manifest_file = run_dir / MANIFEST_FILENAME
        if manifest_file.is_file():
            try:
                run.manifest = read_manifest(run_dir)
            except ProjectError as exc:
                raise RunLoadError(str(exc)) from exc
        run._read_manifest_fields()
        run._read_videos()
        summary_file = run_dir / BATCH_SUMMARY
        if summary_file.is_file():
            try:
                run.summary = read_table(summary_file)
            except (OSError, csv.Error, UnicodeDecodeError) as exc:
                raise RunLoadError(f"Cannot read {BATCH_SUMMARY}: {exc}") from exc
        elif not run.manifest and not run.videos:
            raise RunLoadError(
                f"{run_dir.name} has no {MANIFEST_FILENAME}, "
                f"{BATCH_SUMMARY} or video folders: not a run folder"
            )
        else:
            run.notes.append(f"{BATCH_SUMMARY} is missing from this run.")
        run._read_time_bins()
        run._match_rows()
        if not run.manifest:
            run.notes.append(
                f"{MANIFEST_FILENAME} is missing; the folder was read as it is."
            )
        elif run.schema_version < 3:
            run.notes.append(
                f"Manifest schema {run.schema_version}: older run, some "
                "details (metadata, hashes) are not recorded."
            )
        return run

    def _read_manifest_fields(self) -> None:
        manifest = self.manifest
        try:
            self.schema_version = int(manifest.get("schema_version", 1)) if manifest else 0
        except (TypeError, ValueError):
            self.schema_version = 1
        self.paradigm = str(manifest.get("paradigm") or "")
        self.started_at = str(manifest.get("started_at") or "")
        self.finished_at = str(manifest.get("finished_at") or "")
        self.cancelled = bool(manifest.get("cancelled", False))
        try:
            self.requested = int(manifest.get("requested_videos") or 0)
        except (TypeError, ValueError):
            self.requested = 0
        self.sample_sheet = str(manifest.get("sample_sheet") or "")
        try:
            self.time_bin_seconds = float(manifest.get("time_bin_seconds") or 0.0)
        except (TypeError, ValueError):
            self.time_bin_seconds = 0.0
        self.failures = [
            (str(item.get("source", "")), str(item.get("message", "")))
            for item in manifest.get("failures") or [] if isinstance(item, dict)
        ]

    def _read_videos(self) -> None:
        warnings: Dict[str, List[str]] = {}
        for item in self.manifest.get("quality_warnings") or []:
            if isinstance(item, dict) and item.get("source"):
                warnings.setdefault(norm(item["source"]), []).append(
                    str(item.get("message", ""))
                )
        failure_text = {norm(src): msg for src, msg in self.failures if src}
        entries = [
            item for item in self.manifest.get("videos") or []
            if isinstance(item, dict)
        ]
        for item in entries:
            source = Path(item["source"]) if item.get("source") else None
            folder_name = str(item.get("output_folder") or "")
            folder = self.run_dir / folder_name if folder_name else self.run_dir
            summary = item.get("summary_json")
            marker = norm(source) if source is not None else ""
            video = RunVideo(
                name=source.name if source is not None else folder_name,
                folder=folder,
                source=source,
                status=str(item.get("status") or STATUS_COMPLETED),
                metadata={
                    str(k): "" if v is None else str(v)
                    for k, v in (item.get("metadata") or {}).items()
                },
                summary_json=(self.run_dir / summary) if summary else None,
                warnings=list(warnings.get(marker, [])),
                failure=failure_text.get(marker, ""),
            )
            self.videos.append(video)
        if self.videos:
            return
        # No manifest (or one without videos): the folders hold the run.
        try:
            children = sorted(
                child for child in self.run_dir.iterdir()
                if child.is_dir() and (child / "data").is_dir()
            )
        except OSError:
            children = []
        for child in children:
            self.videos.append(RunVideo(name=child.name, folder=child))

    def _read_time_bins(self) -> None:
        batch = self.run_dir / BATCH_TIME_BINS
        try:
            if batch.is_file():
                self.time_bins = read_table(batch)
                return
        except (OSError, csv.Error, UnicodeDecodeError):
            logger.warning("Cannot read %s", batch, exc_info=True)
        # Older runs: only the per-video files exist; put them together.
        combined: Optional[Table] = None
        for video in self.videos:
            path = video.time_bins_csv()
            if path is None:
                continue
            try:
                table = read_table(path)
            except (OSError, csv.Error, UnicodeDecodeError):
                continue
            if combined is None:
                combined = Table(
                    [FILENAME_COLUMN] + list(video.metadata) + table.headers
                )
            extras = [video.metadata.get(c, "") for c in video.metadata]
            wanted = combined.headers[1 + len(video.metadata):]
            if table.headers != wanted:
                continue
            combined.rows.extend([video.name] + extras + row for row in table.rows)
        self.time_bins = combined

    def _match_rows(self) -> None:
        """Give every row of the batch summary its video."""
        names = self.summary.values(FILENAME_COLUMN)
        completed = [v for v in self.videos if v.status == STATUS_COMPLETED]
        if (
            len(completed) == len(names)
            and all(v.name == name for v, name in zip(completed, names))
        ):
            self._row_videos = list(completed)
            return
        pool = list(completed or self.videos)
        matched: List[Optional[RunVideo]] = []
        for name in names:
            found = next((v for v in pool if v.name == name), None)
            if found is not None:
                pool.remove(found)
            matched.append(found)
        self._row_videos = matched

    # -- queries ---------------------------------------------------------
    @property
    def name(self) -> str:
        return self.run_dir.name

    def video_for_row(self, row: int) -> Optional[RunVideo]:
        if 0 <= row < len(self._row_videos):
            return self._row_videos[row]
        return None

    def row_for_source(self, path: os.PathLike) -> int:
        """The summary row of a source video (by path, else by file name)."""
        wanted = norm(path)
        for row, video in enumerate(self._row_videos):
            if video is not None and video.source is not None \
                    and norm(video.source) == wanted:
                return row
        name = Path(path).name
        for row, video in enumerate(self._row_videos):
            if video is not None and video.name == name:
                return row
        return -1

    def metadata_columns(self) -> List[str]:
        """Sample-sheet columns of the summary (and the time bins)."""
        columns: List[str] = []
        for video in self.videos:
            for column in video.metadata:
                if column not in columns:
                    columns.append(column)
        if columns:
            return columns
        table = self.time_bins
        if table is not None and "bin_index" in table.headers:
            return [
                h for h in table.headers[: table.headers.index("bin_index")]
                if h not in (FILENAME_COLUMN, "paradigm")
            ]
        return []

    def quality_warning_count(self) -> int:
        return sum(len(video.warnings) for video in self.videos)

    def describe(self) -> str:
        """One line for the run header."""
        parts = [self.paradigm or "paradigm unknown"]
        completed = sum(1 for v in self.videos if v.status == STATUS_COMPLETED)
        failed = sum(1 for v in self.videos if v.status == STATUS_FAILED) \
            or len(self.failures)
        parts.append(f"{completed} video(s) completed")
        if failed:
            parts.append(f"{failed} failed")
        if self.quality_warning_count():
            parts.append(f"{self.quality_warning_count()} quality warning(s)")
        if self.cancelled:
            parts.append("cancelled")
        if self.time_bin_seconds:
            parts.append(f"time bins of {self.time_bin_seconds:g} s")
        if self.started_at:
            parts.append(f"started {self.started_at}")
        return " - ".join(parts)


def looks_like_run(path: os.PathLike) -> bool:
    folder = Path(path)
    return folder.is_dir() and any(
        (folder / name).is_file()
        for name in (MANIFEST_FILENAME, BATCH_SUMMARY, RUN_PLAN_FILENAME)
    )


# ----------------------------------------------------------------------
# Key metrics of one video
# ----------------------------------------------------------------------
def format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "N/A"
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


def _flatten(
    prefix: str, value: Any, depth: int, out: List[Tuple[str, str]],
    notes: List[str],
) -> None:
    if isinstance(value, dict):
        if depth >= _MAX_FLATTEN_DEPTH:
            return
        for key, item in value.items():
            if key == "notes":
                if isinstance(item, list):
                    notes.extend(str(n) for n in item)
                continue
            _flatten(f"{prefix}{' / ' if prefix else ''}{key}", item,
                     depth + 1, out, notes)
    elif isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value) \
                and len(value) <= _MAX_LIST_ITEMS:
            out.append((prefix, ", ".join(format_value(i) for i in value)))
    else:
        out.append((prefix, format_value(value)))


def key_metrics(summary: Dict[str, Any]) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """``([(section, metric, value)], notes)`` from a video's summary.json.

    Sections are the top-level keys; settings sections (detection parameters,
    run configuration) are left out, long arrays are skipped.
    """
    rows: List[Tuple[str, str, str]] = []
    notes: List[str] = []
    for section, content in summary.items():
        if section in _SETTINGS_SECTIONS:
            continue
        pairs: List[Tuple[str, str]] = []
        _flatten("", content, 0, pairs, notes)
        title = section.replace("_", " ")
        for label, value in pairs:
            rows.append((title, label.replace("_", " ") or title, value))
    return rows, notes


# ----------------------------------------------------------------------
# Time bins
# ----------------------------------------------------------------------
@dataclass
class Series:
    label: str
    x: List[float]
    y: List[float]
    err: List[float]
    n: int                  # videos behind the series
    unit: str = ""


def time_bin_metrics(table: Optional[Table]) -> List[str]:
    if table is None:
        return []
    seen: List[str] = []
    for metric in table.values("metric"):
        if metric and metric not in seen:
            seen.append(metric)
    return seen


def time_bin_subjects(table: Optional[Table]) -> List[str]:
    if table is None:
        return []
    seen: List[str] = []
    for subject in table.values("subject"):
        if subject and subject not in seen:
            seen.append(subject)
    return seen


def time_bin_group_columns(table: Optional[Table]) -> List[str]:
    """Metadata columns of a long time-bin table (between video and bins)."""
    if table is None or "bin_index" not in table.headers:
        return []
    return [
        h for h in table.headers[: table.headers.index("bin_index")]
        if h not in (FILENAME_COLUMN, "paradigm")
    ]


def time_bin_series(
    table: Optional[Table],
    metric: str,
    subject: str = "",
    group_by: str = "",
) -> List[Series]:
    """The values of ``metric`` over the bins, per video or per group.

    ``subject`` picks one mouse (TST) or, empty, keeps them all as separate
    series. With ``group_by`` (a metadata column) the videos of a group are
    averaged per bin and ``err`` is the standard error of the mean; without
    it every video is its own series.
    """
    if table is None or not table.rows:
        return []
    h = table.headers
    try:
        i_video = h.index(FILENAME_COLUMN)
        i_bin, i_start, i_end = (
            h.index("bin_index"), h.index("bin_start_s"), h.index("bin_end_s")
        )
        i_metric, i_value = h.index("metric"), h.index("value")
    except ValueError:
        return []
    i_subject = h.index("subject") if "subject" in h else -1
    i_unit = h.index("unit") if "unit" in h else -1
    i_group = h.index(group_by) if group_by and group_by in h else -1

    # series key -> bin -> [(value, midpoint)]  and series key -> videos
    cells: Dict[Tuple, Dict[int, List[Tuple[float, float]]]] = {}
    members: Dict[Tuple, set] = {}
    unit = ""
    for row in table.rows:
        if row[i_metric] != metric:
            continue
        row_subject = row[i_subject] if i_subject >= 0 else ""
        if subject and row_subject != subject:
            continue
        value = to_float(row[i_value])
        start, end = to_float(row[i_start]), to_float(row[i_end])
        try:
            bin_index = int(float(row[i_bin]))
        except ValueError:
            continue
        if value is None or start is None or end is None:
            continue
        unit = unit or (row[i_unit] if i_unit >= 0 else "")
        who = (
            (row[i_group] or "(blank)",) if i_group >= 0 else (row[i_video],)
        )
        key = who + ((row_subject,) if row_subject and not subject else ())
        cells.setdefault(key, {}).setdefault(bin_index, []).append(
            (value, (start + end) / 2.0)
        )
        members.setdefault(key, set()).add(row[i_video])

    result: List[Series] = []
    for key in sorted(cells, key=lambda k: tuple(str(p) for p in k)):
        by_bin = cells[key]
        xs, ys, errs = [], [], []
        for bin_index in sorted(by_bin):
            values = [v for v, _ in by_bin[bin_index]]
            xs.append(float(np.mean([x for _, x in by_bin[bin_index]])))
            ys.append(float(np.mean(values)))
            errs.append(
                float(np.std(values, ddof=1) / math.sqrt(len(values)))
                if len(values) > 1 else 0.0
            )
        result.append(Series(
            label=" / ".join(str(part) for part in key),
            x=xs, y=ys, err=errs, n=len(members[key]), unit=unit,
        ))
    return result


# ----------------------------------------------------------------------
# Comparing two runs
# ----------------------------------------------------------------------
@dataclass
class DiffCell:
    a: str
    b: str
    changed: bool


@dataclass
class DiffRow:
    video: str
    status: str                     # "both", "only_a" or "only_b"
    cells: List[DiffCell] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.status != "both" or any(c.changed for c in self.cells)


@dataclass
class RunDiff:
    headers: List[str] = field(default_factory=list)
    rows: List[DiffRow] = field(default_factory=list)

    @property
    def changed_cells(self) -> int:
        return sum(
            1 for row in self.rows for cell in row.cells if cell.changed
        )

    @property
    def changed_rows(self) -> int:
        return sum(1 for row in self.rows if row.changed)


def values_equal(a: str, b: str, tolerance: float = 1e-9) -> bool:
    a, b = (a or "").strip(), (b or "").strip()
    if a == b:
        return True
    x, y = to_float(a), to_float(b)
    if x is None or y is None:
        return False
    return math.isclose(x, y, rel_tol=tolerance, abs_tol=tolerance)


def _keyed_rows(table: Table) -> Dict[Tuple[str, int], List[str]]:
    index = table.column(FILENAME_COLUMN)
    seen: Dict[str, int] = {}
    keyed: Dict[Tuple[str, int], List[str]] = {}
    for position, row in enumerate(table.rows):
        name = row[index] if index >= 0 else f"row {position + 1}"
        occurrence = seen.get(name, 0)
        seen[name] = occurrence + 1
        keyed[(name, occurrence)] = row
    return keyed


def diff_summaries(a: Table, b: Table, tolerance: float = 1e-9) -> RunDiff:
    """Per-video comparison of two batch summaries (values flagged if changed)."""
    headers = list(a.headers) + [h for h in b.headers if h not in a.headers]
    rows_a, rows_b = _keyed_rows(a), _keyed_rows(b)
    keys = list(rows_a) + [k for k in rows_b if k not in rows_a]
    diff = RunDiff(headers=headers)
    for key in keys:
        row_a, row_b = rows_a.get(key), rows_b.get(key)
        status = "both" if row_a is not None and row_b is not None else (
            "only_a" if row_a is not None else "only_b"
        )
        cells: List[DiffCell] = []
        for header in headers:
            va = row_a[a.headers.index(header)] if (
                row_a is not None and header in a.headers) else ""
            vb = row_b[b.headers.index(header)] if (
                row_b is not None and header in b.headers) else ""
            same = values_equal(va, vb, tolerance) if status == "both" else False
            cells.append(DiffCell(va, vb, not same and status == "both"))
        name = key[0] if key[1] == 0 else f"{key[0]} (#{key[1] + 1})"
        diff.rows.append(DiffRow(name, status, cells))
    return diff
