"""Animal/sample metadata for a batch, read from a CSV "sample sheet".

The first column of the sheet is ``video`` and every other column is free
metadata (``animal_id``, ``group``, ``sex``, ``treatment``, ...)::

    video,animal_id,group,sex
    mouse1.mp4,A01,control,F
    day2/mouse1.mp4,A01,control,F

A ``video`` cell is a path relative to the sheet's folder, an absolute path,
or a bare filename. Videos are matched to rows with the same rules as saved
batch plans (``core.batch_plan_keys.match_videos_to_plan``) with the sheet
file as the anchor, so a video is never matched by guesswork: a filename that
fits several rows or several selected videos is reported as ambiguous.

Nothing here touches Qt.
"""
import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from core.batch_plan_keys import match_videos_to_plan

VIDEO_COLUMN = "video"
# Delimiters spreadsheet programs write (semicolons in many locales).
_DELIMITERS = (",", ";", "\t")
# Reason ``match_videos_to_plan`` gives a video with no row at all; every
# other reason means the choice between several rows/videos was ambiguous.
_NOT_LISTED_REASON = "not in the plan"


class SampleSheetError(ValueError):
    """The sample sheet cannot be used; the message says why."""


@dataclass(frozen=True)
class SampleRow:
    line: int                    # 1-based line number in the file
    video: str                   # the ``video`` cell, as written
    values: Dict[str, str]       # metadata column -> value


@dataclass(frozen=True)
class SampleSheet:
    path: Path
    columns: Tuple[str, ...]     # metadata columns, in file order
    rows: Tuple[SampleRow, ...]


@dataclass
class SampleMetadata:
    """Run-level metadata: one value per metadata column for every video."""
    sheet_path: Path
    columns: Tuple[str, ...]
    by_video: Dict[Path, Dict[str, str]] = field(default_factory=dict)

    def values_for(self, video: Path) -> Dict[str, str]:
        """Metadata of ``video`` (blank values when the video has no row)."""
        found = self.by_video.get(video)
        if found is None:
            target = _norm(video)
            for candidate, values in self.by_video.items():
                if _norm(candidate) == target:
                    found = values
                    break
        if found is None:
            return {column: "" for column in self.columns}
        return {column: found.get(column, "") for column in self.columns}


@dataclass
class SampleSheetMatch:
    """Outcome of matching selected videos against a sample sheet."""
    metadata: SampleMetadata
    unmatched: List[Tuple[Path, str]] = field(default_factory=list)
    ambiguous: List[Tuple[Path, str]] = field(default_factory=list)
    # Sheet rows (``video`` cells) that no selected video matched.
    unused_rows: List[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return not self.unmatched and not self.ambiguous

    def problems(self) -> List[str]:
        """One human-readable line per video that has no unique row."""
        lines = [
            f"{video.name}: {reason}" for video, reason in self.unmatched
        ]
        lines.extend(
            f"{video.name}: {reason}" for video, reason in self.ambiguous
        )
        return lines


def _norm(path: Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def _row_key(cell: str) -> str:
    return os.path.normcase(cell.replace("\\", "/"))


def _detect_delimiter(header_line: str) -> str:
    counts = {d: header_line.count(d) for d in _DELIMITERS}
    best = max(_DELIMITERS, key=lambda d: counts[d])
    return best if counts[best] > 0 else ","


def load_sample_sheet(path: Path) -> SampleSheet:
    """Read a sample sheet CSV (UTF-8, with or without a BOM).

    Cells are stripped, blank rows are skipped, and the delimiter may be a
    comma, semicolon or tab. Raises ``SampleSheetError`` for an unreadable
    file, a first column that is not ``video``, a missing/duplicate/empty
    column name, a row without a video, or a video listed twice.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise SampleSheetError(f"Cannot read {path.name}: {exc}") from exc

    lines = text.splitlines()
    first_content = next((line for line in lines if line.strip()), "")
    delimiter = _detect_delimiter(first_content)
    reader = csv.reader(lines, delimiter=delimiter)

    header: Optional[List[str]] = None
    header_line = 0
    rows: List[SampleRow] = []
    seen: Dict[str, int] = {}
    for line_number, cells in enumerate(reader, 1):
        cells = [cell.strip() for cell in cells]
        if not any(cells):
            continue
        if header is None:
            header = cells
            header_line = line_number
            _validate_header(header, path)
            continue

        video = cells[0]
        width = len(header)
        extra = [cell for cell in cells[width:] if cell]
        if extra:
            raise SampleSheetError(
                f"{path.name} line {line_number}: more values than the "
                f"{width} columns of the header"
            )
        padded = (cells + [""] * width)[:width]
        if not video:
            raise SampleSheetError(
                f"{path.name} line {line_number}: the video column is empty"
            )
        key = _row_key(video)
        if key in seen:
            raise SampleSheetError(
                f"{path.name}: video {video!r} is listed twice "
                f"(lines {seen[key]} and {line_number})"
            )
        seen[key] = line_number
        rows.append(SampleRow(
            line=line_number,
            video=video,
            values=dict(zip(header[1:], padded[1:])),
        ))

    if header is None:
        raise SampleSheetError(f"{path.name} is empty")
    if not rows:
        raise SampleSheetError(
            f"{path.name} has a header (line {header_line}) but no rows"
        )
    return SampleSheet(path, tuple(header[1:]), tuple(rows))


def _validate_header(header: Sequence[str], path: Path) -> None:
    if header[0].lower() != VIDEO_COLUMN:
        raise SampleSheetError(
            f"{path.name}: the first column must be named "
            f"'{VIDEO_COLUMN}', found {header[0]!r}"
        )
    seen = set()
    for name in header:
        if not name:
            raise SampleSheetError(f"{path.name}: a column has no name")
        lowered = name.lower()
        if lowered in seen:
            raise SampleSheetError(
                f"{path.name}: column {name!r} appears twice"
            )
        seen.add(lowered)


def metadata_for_videos(
    sheet: SampleSheet,
    sheet_path: Optional[Path],
    videos: Iterable[Path],
) -> SampleSheetMatch:
    """Match ``videos`` to the rows of ``sheet`` without ever guessing.

    ``sheet_path`` anchors relative ``video`` cells (default: the sheet's own
    path). Videos with no row are reported in ``unmatched`` and videos whose
    filename fits several rows (or shares a filename with another selected
    video) in ``ambiguous``; neither receives metadata.
    """
    anchor = Path(sheet_path) if sheet_path is not None else sheet.path
    video_list = [Path(video) for video in videos]
    rows_by_key = {row.video: row for row in sheet.rows}
    plan_match = match_videos_to_plan(video_list, rows_by_key.keys(), anchor)

    metadata = SampleMetadata(anchor, sheet.columns)
    for video, key in plan_match.matched.items():
        metadata.by_video[video] = dict(rows_by_key[key].values)

    result = SampleSheetMatch(metadata)
    for video, reason in plan_match.unmatched:
        target = (
            result.unmatched if reason == _NOT_LISTED_REASON
            else result.ambiguous
        )
        target.append((video, reason))
    used = set(plan_match.matched.values())
    result.unused_rows = [row.video for row in sheet.rows if row.video not in used]
    return result


def reserved_column_conflicts(
    columns: Iterable[str], reserved: Iterable[str]
) -> List[str]:
    """Metadata columns whose name an output table already uses.

    Compared case-insensitively, so ``Paradigm`` still clashes with
    ``paradigm``. Existing output columns are never renamed; the run is
    refused instead.
    """
    taken = {name.lower() for name in reserved}
    return [column for column in columns if column.lower() in taken]


def describe_match(match: SampleSheetMatch, total_videos: int) -> str:
    """Short multi-line summary for a status label or a dialog."""
    matched = total_videos - len(match.unmatched) - len(match.ambiguous)
    lines = [f"{matched} of {total_videos} video(s) matched a sample row."]
    if match.unmatched:
        lines.append("No row for:")
        lines.extend(f"  {video.name}" for video, _ in match.unmatched)
    if match.ambiguous:
        lines.append("Ambiguous (several candidates):")
        lines.extend(
            f"  {video.name} - {reason}" for video, reason in match.ambiguous
        )
    if match.unused_rows:
        lines.append("Sheet rows that match no selected video:")
        lines.extend(f"  {row}" for row in match.unused_rows)
    return "\n".join(lines)

