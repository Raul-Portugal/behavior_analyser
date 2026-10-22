"""
gui/workspace/stages/trials_widgets.py
The building blocks of the Trials stage (``trials_stage.py``).

* **Time text**: :func:`parse_time` reads what a person types (``75``,
  ``75.5``, ``1:15``, ``1:15.5``, ``1:02:03``) and :func:`format_time` writes
  it back as ``m:ss.s``. Both are free of Qt.
* **Rows**: :func:`trial_rows` turns the project into the rows of the timing
  table (one per video, or for the Tail Suspension Test a header per video
  and one row per mouse), with the *effective* start and end and where each
  came from (the user, the protocol, the template). Free of Qt.
* :class:`TrialsModel` / :class:`TimeDelegate` / :class:`TrialsTable`: the
  table. Editing a cell asks the stage (through the ``commit`` callback) to
  write the project; an entry that cannot be written is *kept* in the cell,
  red, with the reason as its tooltip, so typing is never silently lost.
* :class:`TimelineBar`: the scrub bar under the player, with the trial window
  shaded and its start and end marked.
* :class:`PlayerPane`: the player, its bar and the "Set start here" /
  "Set end here" buttons.
"""
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PyQt6.QtCore import (QAbstractItemModel, QAbstractTableModel,
                          QModelIndex, QPointF, QRectF, QSize, Qt, pyqtSignal)
from PyQt6.QtGui import (QBrush, QColor, QFont, QKeyEvent, QPainter,
                         QPainterPath, QPen)
from PyQt6.QtWidgets import (QAbstractItemDelegate, QAbstractItemView,
                             QHBoxLayout, QHeaderView, QLineEdit, QSizePolicy,
                             QStyledItemDelegate, QTableView, QVBoxLayout,
                             QWidget)

from core.project import (START_PER_VIDEO, TRIAL_INVALID, TRIAL_NEEDS_START,
                          TRIAL_OK, Project)
from gui.video_player import VideoPlayerWidget
from gui.workspace.theme import Spacing, current_palette
from gui.workspace.ui_kit import (GhostButton, KeyHint, Label, PrimaryButton,
                                  SecondaryButton, _Themed, _font)
from gui.workspace.video_panel import ChipDelegate, ChipRole

# ----------------------------------------------------------------------
# Time text
# ----------------------------------------------------------------------
TIME_HINT = "m:ss.s"
_NUMBER = re.compile(r"^\d+(?:\.\d*)?$|^\.\d+$")


def parse_time(text: str) -> float:
    """Seconds from ``75``, ``75.5``, ``1:15``, ``1:15.5`` or ``1:02:03``.

    Raises ``ValueError`` with a sentence a person can act on.
    """
    cleaned = str(text).strip().replace(",", ".")
    if not cleaned:
        raise ValueError("Type a time such as 90 or 1:30")
    if cleaned.startswith("-"):
        raise ValueError("A time cannot be negative")
    bad = ValueError(f"“{cleaned}” is not a time. Use seconds (90) or m:ss (1:30)")
    parts = cleaned.split(":")
    if len(parts) > 3 or not _NUMBER.match(parts[-1]) or not all(
        p.isdigit() for p in parts[:-1]
    ):
        raise bad
    seconds = float(parts[-1])
    if len(parts) == 1:
        return seconds
    if seconds >= 60:
        raise ValueError("Seconds must be below 60 when minutes are given (1:30)")
    whole = [int(p) for p in parts[:-1]]
    if len(whole) == 2 and whole[1] >= 60:
        raise ValueError("Minutes must be below 60 when hours are given")
    total = seconds + 60 * whole[-1]
    if len(whole) == 2:
        total += 3600 * whole[0]
    return total


def format_time(seconds: float, decimals: int = 1) -> str:
    """``m:ss.s`` (minutes may exceed 59)."""
    total = round(max(0.0, float(seconds)), decimals)
    minutes, rest = divmod(total, 60)
    width = 3 + decimals if decimals else 2
    return f"{int(minutes)}:{rest:0{width}.{decimals}f}"


def format_time_exact(seconds: float) -> str:
    """Like :func:`format_time` with up to three decimals, zeros trimmed, so
    a value that is edited and accepted unchanged is not rewritten."""
    text = format_time(seconds, 3)
    head, _, tail = text.partition(".")
    return f"{head}.{tail.rstrip('0') or '0'}"


def round_time(seconds: float) -> float:
    """What the stage stores for a time taken from the player (milliseconds)."""
    return round(float(seconds), 3)


# ----------------------------------------------------------------------
# Rows
# ----------------------------------------------------------------------
KIND_VIDEO = "video"       # one row for a whole video
KIND_HEADER = "header"     # Tail Suspension Test: the video, above its mice
KIND_MOUSE = "mouse"       # Tail Suspension Test: one mouse of a video

ORIGIN_MANUAL = "manual"       # the user typed or captured it
ORIGIN_PROTOCOL = "protocol"   # start / end follow the trial protocol
ORIGIN_TEMPLATE = "template"   # the project template's value
ORIGIN_VIDEO = "video"         # no end: until the video ends

SOURCE_MANUAL = "manual"
SOURCE_PROTOCOL = "protocol"
SOURCE_TEMPLATE = "template"
SOURCE_WHOLE = "whole"

COL_STATUS, COL_VIDEO, COL_START, COL_END, COL_DURATION, COL_SOURCE = range(6)
COLUMN_TITLES = ("Status", "Video", "Start", "End", "Duration", "Source")
COLUMN_TIPS = (
    "Whether this trial can be analysed",
    "The video (Tail Suspension Test: video, then each mouse)",
    "When the trial starts, in minutes:seconds. Type 90 or 1:30, or press S "
    "on the video",
    "When the trial ends. Left alone it is the start plus the protocol's "
    "duration. Press E on the video to set it",
    "How long the trial lasts. Typing a duration sets the end",
    "Where the times come from: the protocol, or set by you for this video",
)

_STATUS_CHIPS = {
    TRIAL_OK: ("ok", "OK"),
    TRIAL_NEEDS_START: ("warning", "Needs start"),
    TRIAL_INVALID: ("error", "Invalid"),
}


@dataclass
class TrialRow:
    """One row of the timing table (the *effective* times, and their origin)."""
    kind: str
    key: str
    video: str
    subject: Optional[str] = None          # the mouse (kind "mouse")
    status: str = TRIAL_OK
    problem: Optional[str] = None
    start: Optional[float] = None
    start_known: bool = True               # False: the video still needs its start
    start_manual: bool = False
    end: Optional[float] = None
    end_origin: str = ORIGIN_VIDEO
    duration: Optional[float] = None
    source: str = SOURCE_PROTOCOL
    video_duration: Optional[float] = None
    protocol_duration: Optional[float] = None
    # The cells the problem is about (COL_START / COL_END).
    problem_cells: Tuple[int, ...] = ()
    # Past the end of the video (a warning, not an error).
    end_past_video: bool = False

    @property
    def is_header(self) -> bool:
        return self.kind == KIND_HEADER

    @property
    def end_manual(self) -> bool:
        return self.end_origin == ORIGIN_MANUAL

    @property
    def needs_attention(self) -> bool:
        return self.status != TRIAL_OK

    @property
    def chip(self) -> Tuple[str, str]:
        return _STATUS_CHIPS[self.status]

    @property
    def label(self) -> str:
        return self.video if self.subject is None else f"{self.video} › {self.subject}"

    def identity(self) -> Tuple[str, Optional[str]]:
        return self.key, self.subject


def _problem_cells(start, end, problem: Optional[str],
                   video_duration: Optional[float]) -> Tuple[int, ...]:
    """Which of the start / end cells a trial problem is about."""
    if problem is None:
        return ()
    cells = []
    if start is None or start < 0 or (
        video_duration is not None and start >= video_duration
    ):
        cells.append(COL_START)
    if end is not None and start is not None and end <= start:
        cells.append(COL_END)
    return tuple(cells) or (COL_START, COL_END)


def _window_problem(window: Tuple[float, float]) -> Optional[str]:
    a, b = window
    if a < 0 or b <= a:
        return "the end must be after the start, and neither can be negative"
    return None


def mouse_subjects(project: Project, settings, maze_for: Callable) -> List[str]:
    """The mice that get their own window (empty: one window per video)."""
    if settings is None or settings.roi_manager is None:
        return []
    maze = maze_for(project.effective_paradigm())
    if maze is None or not maze.has_per_roi_times:
        return []
    return list(maze.time_window_subjects(settings.roi_manager))


def trial_rows(project: Project, duration_of: Callable[[str], Optional[float]],
               maze_for: Callable) -> List[TrialRow]:
    """The rows of the timing table for every video of the project."""
    protocol = project.trial_protocol
    per_video = protocol.start_mode == START_PER_VIDEO
    rows: List[TrialRow] = []
    for entry in project.videos:
        trial = entry.trial
        settings = project.effective_settings(entry)
        status = project.trial_status(entry)
        problem = project.trial_problem(entry)
        vduration = duration_of(entry.key)
        subjects = mouse_subjects(project, settings, maze_for)

        if subjects:
            windows = settings.per_roi_times
            manual = set(trial.per_roi_times) if trial is not None else set()
            span = project.trial_span(entry)
            header = TrialRow(
                KIND_HEADER, entry.key, entry.name, None, status, problem,
                start=span[0] if span and windows else None,
                end=span[1] if span and windows else None,
                video_duration=vduration,
                source=SOURCE_MANUAL if manual else (
                    SOURCE_TEMPLATE if windows else SOURCE_WHOLE),
            )
            rows.append(header)
            for name in subjects:
                window = windows.get(name)
                if window is None:
                    rows.append(TrialRow(
                        KIND_MOUSE, entry.key, entry.name, name, TRIAL_OK,
                        start=0.0, start_known=False, end=None,
                        end_origin=ORIGIN_VIDEO, video_duration=vduration,
                        duration=vduration, source=SOURCE_WHOLE,
                    ))
                    continue
                issue = _window_problem(window)
                is_manual = name in manual
                origin = ORIGIN_MANUAL if is_manual else ORIGIN_TEMPLATE
                rows.append(TrialRow(
                    KIND_MOUSE, entry.key, entry.name, name,
                    TRIAL_INVALID if issue else TRIAL_OK, issue,
                    start=window[0], start_manual=is_manual, end=window[1],
                    end_origin=origin,
                    duration=window[1] - window[0] if not issue else None,
                    source=SOURCE_MANUAL if is_manual else SOURCE_TEMPLATE,
                    video_duration=vduration,
                    problem_cells=(COL_START, COL_END) if issue else (),
                    end_past_video=bool(
                        vduration is not None and window[1] > vduration + 0.05),
                ))
            continue

        span = project.trial_span(entry)
        start_manual = trial is not None and trial.start_set
        end_manual = trial is not None and trial.end_time is not None
        if span is not None:
            start, end = span
        else:                      # no template yet: what the layers would give
            start = trial.start_time if start_manual else 0.0
            end = trial.end_time if end_manual else (
                None if protocol.duration_s is None
                else start + protocol.duration_s)
        if end_manual:
            origin = ORIGIN_MANUAL
        elif end is None:
            origin = ORIGIN_VIDEO
        elif protocol.duration_s is not None:
            origin = ORIGIN_PROTOCOL
        else:
            origin = ORIGIN_TEMPLATE
        start_known = start_manual or not per_video or status == TRIAL_OK
        valid = problem is None and start is not None
        if valid and end is not None:
            length = end - start
        elif valid and vduration is not None:
            length = max(0.0, vduration - start)
        else:
            length = None
        rows.append(TrialRow(
            KIND_VIDEO, entry.key, entry.name, None, status, problem,
            start=start, start_known=start_known, start_manual=start_manual,
            end=end, end_origin=origin, duration=length,
            source=SOURCE_MANUAL if (start_manual or end_manual) else SOURCE_PROTOCOL,
            video_duration=vduration, protocol_duration=protocol.duration_s,
            problem_cells=_problem_cells(start, end, problem, vduration),
            end_past_video=bool(
                vduration is not None and end is not None
                and end > vduration + 0.05),
        ))
    return rows


# ----------------------------------------------------------------------
# The table model
# ----------------------------------------------------------------------
# commit(row, column, text) -> None when written, else why not
CommitFunction = Callable[[TrialRow, int, str], Optional[str]]


class TrialsModel(QAbstractTableModel):
    """The timing table. Rows come from :func:`trial_rows`; edits go to
    ``commit`` (the stage), which writes the project."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: List[TrialRow] = []
        self._bad: Dict[Tuple[str, Optional[str], int], Tuple[str, str]] = {}
        self.commit: Optional[CommitFunction] = None
        self.read_only = False

    # -- content ---------------------------------------------------------
    def rows(self) -> List[TrialRow]:
        return list(self._rows)

    def row_at(self, row: int) -> Optional[TrialRow]:
        return self._rows[row] if 0 <= row < len(self._rows) else None

    def row_of(self, key: str, subject: Optional[str] = None) -> int:
        """Table row of a video (its first selectable row) or of a mouse; -1."""
        for index, row in enumerate(self._rows):
            if row.key != key or row.is_header:
                continue
            if subject is None or row.subject == subject:
                return index
        return -1

    def set_rows(self, rows: Sequence[TrialRow]) -> None:
        old = [(r.kind, r.key, r.subject) for r in self._rows]
        new = [(r.kind, r.key, r.subject) for r in rows]
        live = {(r.key, r.subject) for r in rows}
        self._bad = {k: v for k, v in self._bad.items() if k[:2] in live}
        if old == new:
            self._rows = list(rows)
            if rows:
                self.dataChanged.emit(
                    self.index(0, 0),
                    self.index(len(rows) - 1, len(COLUMN_TITLES) - 1),
                )
            return
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def bad_cells(self) -> Dict[Tuple[str, Optional[str], int], Tuple[str, str]]:
        """``{(key, mouse, column): (typed text, reason)}`` for entries that
        could not be written."""
        return dict(self._bad)

    def clear_bad(self) -> None:
        if self._bad:
            self._bad.clear()
            self.refresh_all()

    def refresh_all(self) -> None:
        if self._rows:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(len(self._rows) - 1, len(COLUMN_TITLES) - 1),
            )

    # -- the model interface ---------------------------------------------
    def rowCount(self, parent: Optional[QModelIndex] = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self._rows)

    def columnCount(self, parent: Optional[QModelIndex] = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(COLUMN_TITLES)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and 0 <= section < len(COLUMN_TITLES):
            if role == Qt.ItemDataRole.DisplayRole:
                return COLUMN_TITLES[section]
            if role == Qt.ItemDataRole.ToolTipRole:
                return COLUMN_TIPS[section]
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        row = self.row_at(index.row()) if index.isValid() else None
        if row is None:
            return Qt.ItemFlag.NoItemFlags
        if row.is_header:
            return Qt.ItemFlag.ItemIsEnabled
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if index.column() in (COL_START, COL_END, COL_DURATION) and not self.read_only:
            flags |= Qt.ItemFlag.ItemIsEditable
        return flags

    # -- what a cell says ------------------------------------------------
    def _bad_entry(self, row: TrialRow, column: int):
        return self._bad.get((row.key, row.subject, column))

    def edit_text(self, row: TrialRow, column: int) -> str:
        """The text the editor opens with."""
        bad = self._bad_entry(row, column)
        if bad is not None:
            return bad[0]
        if column == COL_START:
            return format_time_exact(row.start) if row.start_known and row.start is not None else ""
        if column == COL_END:
            if row.end is None or (row.start is None or not row.start_known and row.end_origin != ORIGIN_MANUAL):
                return ""
            return format_time_exact(row.end)
        if column == COL_DURATION and row.duration is not None and row.end is not None:
            return format_time_exact(row.duration)
        return ""

    def display_text(self, row: TrialRow, column: int) -> str:
        bad = self._bad_entry(row, column)
        if bad is not None:
            return bad[0]
        if column == COL_VIDEO:
            if row.kind == KIND_MOUSE:
                return f"    › {row.subject}"
            return row.video
        if row.is_header:
            return ""
        if column == COL_START:
            if row.kind == KIND_MOUSE and not row.start_known:
                return format_time(0.0)
            if not row.start_known:
                return "Set start…"
            return format_time(row.start)
        if column == COL_END:
            return self._end_text(row)
        if column == COL_DURATION:
            if row.duration is None:
                return "—"
            return format_time(row.duration)
        if column == COL_SOURCE:
            return row.source
        return ""

    @staticmethod
    def _end_text(row: TrialRow) -> str:
        if row.end_origin == ORIGIN_VIDEO or row.end is None:
            return "video end"
        if row.end_origin == ORIGIN_MANUAL:
            return format_time(row.end)
        if row.end_origin == ORIGIN_TEMPLATE:
            return f"{format_time(row.end)}  template"
        if not row.start_known and row.protocol_duration is not None:
            return f"start + {format_time(row.protocol_duration, 0 if row.protocol_duration % 1 == 0 else 1)}"
        return f"{format_time(row.end)}  protocol"

    def _problem_text(self, row: TrialRow, column: int) -> Optional[str]:
        bad = self._bad_entry(row, column)
        if bad is not None:
            return bad[1]
        if row.problem and column in row.problem_cells:
            return row.problem[:1].upper() + row.problem[1:] + "."
        return None

    def _tooltip(self, row: TrialRow, column: int) -> Optional[str]:
        problem = self._problem_text(row, column)
        if problem:
            return problem
        if column == COL_STATUS:
            if row.problem:
                return row.problem[:1].upper() + row.problem[1:] + "."
            if row.status == TRIAL_NEEDS_START:
                return ("This video needs its own start: press S on the video "
                        "at the moment the mouse is placed.")
            return "This trial can be analysed."
        if row.is_header:
            return None
        if column == COL_START:
            if row.kind == KIND_MOUSE and not row.start_known:
                return "This mouse has no window: the whole video is analysed. Press S to give it one."
            if not row.start_known:
                return "Not set yet. Find the moment the mouse enters and press S."
            return "Set by you for this video." if row.start_manual else (
                "Every trial starts with the video (see the protocol)."
                if row.kind != KIND_MOUSE else None)
        if column == COL_END:
            if row.end_past_video:
                return "This is after the end of the video; the analysis stops with the video."
            if row.end_origin == ORIGIN_PROTOCOL:
                return ("End = start + the protocol's duration. Type a time or "
                        "press E to set this video's own end.")
            if row.end_origin == ORIGIN_VIDEO:
                return ("This mouse has no window: the whole video is analysed."
                        if row.kind == KIND_MOUSE else "The trial runs until the video ends.")
            if row.end_origin == ORIGIN_TEMPLATE:
                return "Taken from the project's settings."
            return "Set by you for this video. Clear the cell to follow the protocol again."
        if column == COL_DURATION:
            return "Type a duration to set the end to start + duration."
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        row = self.row_at(index.row()) if index.isValid() else None
        if row is None:
            return None
        column = index.column()
        palette = current_palette()
        if role == Qt.ItemDataRole.DisplayRole:
            if column == COL_STATUS:
                return "" if row.kind == KIND_MOUSE and row.status == TRIAL_OK else row.chip[1]
            return self.display_text(row, column)
        if role == Qt.ItemDataRole.EditRole:
            return self.edit_text(row, column)
        if role == ChipRole:
            if column != COL_STATUS:
                return None
            if row.kind == KIND_MOUSE and row.status == TRIAL_OK:
                return None
            return row.chip
        if role == Qt.ItemDataRole.ToolTipRole:
            return self._tooltip(row, column)
        if role == Qt.ItemDataRole.UserRole:
            return row
        bad = self._bad_entry(row, column) is not None
        problem = bad or (row.problem is not None and column in row.problem_cells)
        if role == Qt.ItemDataRole.BackgroundRole:
            if problem:
                return QBrush(QColor(palette.tint("danger", 0.16)))
            if row.is_header:
                return QBrush(QColor(palette.surface_alt))
            return None
        if role == Qt.ItemDataRole.ForegroundRole:
            if problem:
                return QBrush(QColor(palette.danger))
            if row.end_past_video and column == COL_END:
                return QBrush(QColor(palette.warning))
            if column == COL_START and not row.start_known and row.kind != KIND_MOUSE:
                return QBrush(QColor(palette.warning))
            muted = (
                (column == COL_START and not row.start_manual and row.kind != KIND_HEADER)
                or (column == COL_END and row.end_origin != ORIGIN_MANUAL)
                or (column == COL_DURATION and row.end_origin != ORIGIN_MANUAL)
                or column == COL_SOURCE
            )
            if muted:
                return QBrush(QColor(palette.text_muted))
            return None
        if role == Qt.ItemDataRole.FontRole:
            font = QFont()
            if self._bad_entry(row, column) is not None:
                return None
            if row.is_header and column == COL_VIDEO:
                font.setWeight(QFont.Weight.DemiBold)
                return font
            if (column == COL_END and row.end_origin in (ORIGIN_PROTOCOL, ORIGIN_VIDEO)) or (
                column == COL_START and not row.start_known
            ):
                font.setItalic(True)
                return font
            return None
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if column in (COL_START, COL_END, COL_DURATION):
                return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        return None

    def setData(self, index: QModelIndex, value, role=Qt.ItemDataRole.EditRole) -> bool:
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False
        row = self.row_at(index.row())
        if row is None or row.is_header or self.commit is None or self.read_only:
            return False
        column = index.column()
        text = str(value).strip()
        slot = (row.key, row.subject, column)
        if slot not in self._bad and text == self.edit_text(row, column):
            return False                      # unchanged: nothing to write
        reason = self.commit(row, column, text)
        if reason:
            self._bad[slot] = (text, reason)
            self.dataChanged.emit(index, index)
            return True
        self._bad.pop(slot, None)
        self.dataChanged.emit(index, index)
        return True


class TimeDelegate(QStyledItemDelegate):
    """Edits a time cell with a line edit that turns red while the text is
    not a time."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._editor: Optional[QLineEdit] = None

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.setPlaceholderText(TIME_HINT)
        editor.setToolTip("Seconds (90) or minutes:seconds (1:30.5). Enter to "
                          "accept, Esc to cancel. Empty goes back to the "
                          "protocol.")
        editor.textChanged.connect(lambda text, e=editor: self._validate(e, text))
        self._editor = editor
        return editor

    @staticmethod
    def _validate(editor: QLineEdit, text: str) -> None:
        ok = True
        if text.strip():
            try:
                parse_time(text)
            except ValueError:
                ok = False
        palette = current_palette()
        editor.setStyleSheet(
            "" if ok else
            f"QLineEdit {{ border: 1px solid {palette.danger}; color: {palette.danger}; }}"
        )

    def setEditorData(self, editor, index) -> None:
        editor.setText(str(index.data(Qt.ItemDataRole.EditRole) or ""))
        editor.selectAll()

    def setModelData(self, editor, model: QAbstractItemModel, index) -> None:
        model.setData(index, editor.text(), Qt.ItemDataRole.EditRole)

    def updateEditorGeometry(self, editor, option, index) -> None:
        editor.setGeometry(option.rect)

    def commit_open_editor(self) -> bool:
        """Accept what is being typed (leaving the stage must not lose it)."""
        editor = self._editor
        self._editor = None
        if editor is None:
            return False
        try:
            editor.isHidden()
        except RuntimeError:             # already deleted
            return False
        self.commitData.emit(editor)
        self.closeEditor.emit(editor, QAbstractItemDelegate.EndEditHint.NoHint)
        return True

    def has_open_editor(self) -> bool:
        editor = self._editor
        if editor is None:
            return False
        try:
            return not editor.isHidden()
        except RuntimeError:
            return False

    def destroyEditor(self, editor, index) -> None:
        if editor is self._editor:
            self._editor = None
        super().destroyEditor(editor, index)


class TrialsTable(QTableView):
    """The timing table; keys it does not use go to ``key_handler``."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.key_handler: Optional[Callable[[QKeyEvent], bool]] = None

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self.state() != QAbstractItemView.State.EditingState and (
            self.key_handler is not None and self.key_handler(event)
        ):
            event.accept()
            return
        super().keyPressEvent(event)


def make_table() -> Tuple[TrialsTable, TrialsModel, TimeDelegate]:
    """A styled table with its model and delegates."""
    from gui.workspace.theme import style_table_view
    table = TrialsTable()
    model = TrialsModel(table)
    table.setModel(model)
    style_table_view(table)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setEditTriggers(
        QAbstractItemView.EditTrigger.DoubleClicked
        | QAbstractItemView.EditTrigger.EditKeyPressed
    )
    header = table.horizontalHeader()
    header.setStretchLastSection(False)
    header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    header.setSectionResizeMode(COL_VIDEO, QHeaderView.ResizeMode.Stretch)
    header.setMinimumSectionSize(48)
    # The start column must fit "Set start…" (the call to action of a row
    # that has no start yet) without eliding it.
    for column, width in ((COL_STATUS, 100), (COL_START, 92), (COL_END, 104),
                          (COL_DURATION, 56), (COL_SOURCE, 66)):
        table.setColumnWidth(column, width)
    table.setItemDelegateForColumn(COL_STATUS, ChipDelegate(table))
    delegate = TimeDelegate(table)
    for column in (COL_START, COL_END, COL_DURATION):
        table.setItemDelegateForColumn(column, delegate)
    table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
    table.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    return table, model, delegate


# ----------------------------------------------------------------------
# The scrub bar
# ----------------------------------------------------------------------
_NICE_STEPS = (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600)


def tick_step(duration: float, width: float, min_gap: float = 56.0) -> float:
    """Seconds between labelled ticks so labels are at least ``min_gap`` px apart."""
    if duration <= 0 or width <= 0:
        return 60.0
    for step in _NICE_STEPS:
        if width * step / duration >= min_gap:
            return float(step)
    return float(_NICE_STEPS[-1])


class TimelineBar(_Themed, QWidget):
    """A scrub bar over the whole video with the trial window shaded and its
    start and end marked. Click or drag to move the player
    (``seek_requested(seconds)``)."""

    seek_requested = pyqtSignal(float)

    HEIGHT = 46
    MARGIN = 10

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._duration = 0.0
        self._start: Optional[float] = None
        self._end: Optional[float] = None
        self._end_default = False
        self._others: List[Tuple[float, float]] = []
        self._playhead = 0.0
        self._dragging = False
        self.setFixedHeight(self.HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(False)
        self.setToolTip("Click or drag to move through the video. The shaded part is the trial.")
        self.setAccessibleName("Trial timeline")

    # -- state -----------------------------------------------------------
    def set_video(self, duration: float) -> None:
        self._duration = max(0.0, float(duration))
        self.update()

    def set_window(self, start: Optional[float], end: Optional[float],
                   end_is_default: bool = False,
                   others: Sequence[Tuple[float, float]] = ()) -> None:
        """The trial of the selected row (``end`` None: until the video ends)."""
        self._start, self._end = start, end
        self._end_default = bool(end_is_default)
        self._others = list(others)
        self.update()

    def set_playhead(self, seconds: float) -> None:
        self._playhead = max(0.0, float(seconds))
        self.update()

    def duration(self) -> float:
        return self._duration

    def window(self) -> Tuple[Optional[float], Optional[float]]:
        return self._start, self._end

    def playhead(self) -> float:
        return self._playhead

    # -- geometry ----------------------------------------------------------
    def _track(self) -> QRectF:
        return QRectF(self.MARGIN, 12, max(1.0, self.width() - 2 * self.MARGIN), 10)

    def x_of(self, seconds: float) -> float:
        track = self._track()
        if self._duration <= 0:
            return track.left()
        fraction = min(max(seconds / self._duration, 0.0), 1.0)
        return track.left() + fraction * track.width()

    def time_at(self, x: float) -> float:
        track = self._track()
        if self._duration <= 0:
            return 0.0
        fraction = (x - track.left()) / track.width()
        return min(max(fraction, 0.0), 1.0) * self._duration

    # -- mouse -------------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._duration > 0:
            self._dragging = True
            self.seek_requested.emit(self.time_at(event.position().x()))

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self.seek_requested.emit(self.time_at(event.position().x()))

    def mouseReleaseEvent(self, event) -> None:
        self._dragging = False

    # -- painting ------------------------------------------------------------
    def sizeHint(self) -> QSize:
        return QSize(320, self.HEIGHT)

    def paintEvent(self, event) -> None:
        palette = current_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = self._track()
        painter.setPen(QPen(QColor(palette.border_strong), 1))
        painter.setBrush(QColor(palette.surface_alt))
        painter.drawRoundedRect(track, 5, 5)
        if self._duration <= 0:
            return

        # Where the video can be looked at, minute by minute.
        step = tick_step(self._duration, track.width())
        painter.setFont(_font(self.font(), 10))
        seconds = 0.0
        while seconds <= self._duration + 1e-6:
            x = self.x_of(seconds)
            painter.setPen(QPen(QColor(palette.border_strong), 1))
            painter.drawLine(QPointF(x, track.bottom() + 1), QPointF(x, track.bottom() + 5))
            painter.setPen(QColor(palette.text_muted))
            label = format_time(seconds, 0)
            width = painter.fontMetrics().horizontalAdvance(label)
            left = min(max(x - width / 2.0, 0.0), self.width() - width)
            painter.drawText(QPointF(left, track.bottom() + 17), label)
            seconds += step

        # The other mice (Tail Suspension Test), faintly.
        faint = QColor(palette.primary)
        faint.setAlpha(60)
        for a, b in self._others:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(faint)
            painter.drawRoundedRect(
                QRectF(self.x_of(a), track.top() + 2, max(2.0, self.x_of(b) - self.x_of(a)),
                       track.height() - 4), 3, 3)

        # The trial window.
        if self._start is not None:
            start_x = self.x_of(self._start)
            end_x = self.x_of(self._end if self._end is not None else self._duration)
            fill = QColor(palette.primary)
            fill.setAlpha(150 if not self._end_default else 100)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill)
            painter.drawRoundedRect(
                QRectF(start_x, track.top(), max(2.0, end_x - start_x), track.height()), 4, 4)
            self._mark(painter, start_x, track, palette.primary, "start")
            if self._end is not None:
                self._mark(painter, end_x, track, palette.primary, "end",
                           dashed=self._end_default)

        # The playhead.
        x = self.x_of(self._playhead)
        painter.setPen(QPen(QColor(palette.text), 2))
        painter.drawLine(QPointF(x, track.top() - 5), QPointF(x, track.bottom() + 3))
        painter.setBrush(QColor(palette.text))
        painter.setPen(Qt.PenStyle.NoPen)
        path = QPainterPath()
        path.moveTo(x - 4.5, track.top() - 8)
        path.lineTo(x + 4.5, track.top() - 8)
        path.lineTo(x, track.top() - 2)
        path.closeSubpath()
        painter.drawPath(path)

    @staticmethod
    def _mark(painter: QPainter, x: float, track: QRectF, color: str, side: str,
              dashed: bool = False) -> None:
        pen = QPen(QColor(color), 2.5)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(QPointF(x, track.top() - 3), QPointF(x, track.bottom() + 3))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(color))
        flag = 5.0 if side == "start" else -5.0
        path = QPainterPath()
        path.moveTo(x, track.bottom() + 3)
        path.lineTo(x + flag, track.bottom() + 3)
        path.lineTo(x, track.bottom() - 3)
        path.closeSubpath()
        painter.drawPath(path)


# ----------------------------------------------------------------------
# The player side
# ----------------------------------------------------------------------
class PlayerPane(QWidget):
    """The video, its scrub bar and the buttons that set the trial's start and
    end from the frame on screen. Keys it does not use go to ``key_handler``."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.key_handler: Optional[Callable[[QKeyEvent], bool]] = None
        self.player = VideoPlayerWidget(self)
        self.player.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # One scrub bar: ours replaces the player's slider (same place, same
        # job) and adds the trial window.
        self.bar = TimelineBar()
        self.player.slider.hide()
        scrub = self.player.layout().itemAt(1).layout()
        scrub.insertWidget(0, self.bar, 1)

        self.set_start_button = PrimaryButton("Set start here", "play")
        self.set_end_button = SecondaryButton("Set end here", "stop")
        self.to_start_button = GhostButton("Go to start", "arrow-left")
        for button in (self.set_start_button, self.set_end_button, self.to_start_button):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.start_hint = KeyHint("S")
        self.end_hint = KeyHint("E")
        self.readout = Label("", "muted")
        self.readout.setTextFormat(Qt.TextFormat.PlainText)
        self.readout.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(Spacing.XS)
        actions.addWidget(self.set_start_button)
        actions.addWidget(self.start_hint)
        actions.addSpacing(Spacing.XS)
        actions.addWidget(self.set_end_button)
        actions.addWidget(self.end_hint)
        actions.addSpacing(Spacing.SM)
        actions.addWidget(self.to_start_button)
        actions.addStretch(1)

        keys = QHBoxLayout()
        keys.setContentsMargins(0, 0, 0, 0)
        keys.setSpacing(Spacing.XXS)
        keys.addWidget(self.readout, 1)
        for text in ("←", "→"):
            keys.addWidget(KeyHint(text))
        keys.addWidget(Label("frame", "muted"))
        keys.addSpacing(Spacing.XXS)
        keys.addWidget(KeyHint("Shift"))
        keys.addWidget(Label("10 frames", "muted"))
        keys.addSpacing(Spacing.XS)
        keys.addWidget(KeyHint("Enter"))
        keys.addWidget(Label("next video needing a start", "muted"))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.XS)
        layout.addWidget(self.player, 1)
        layout.addLayout(actions)
        layout.addLayout(keys)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self.key_handler is not None and self.key_handler(event):
            event.accept()
            return
        super().keyPressEvent(event)


__all__ = [
    "COL_DURATION", "COL_END", "COL_SOURCE", "COL_START", "COL_STATUS",
    "COL_VIDEO", "KIND_HEADER", "KIND_MOUSE", "KIND_VIDEO", "ORIGIN_MANUAL",
    "ORIGIN_PROTOCOL", "ORIGIN_TEMPLATE", "ORIGIN_VIDEO", "PlayerPane",
    "TimeDelegate", "TimelineBar", "TrialRow", "TrialsModel", "TrialsTable",
    "format_time", "format_time_exact", "make_table", "mouse_subjects",
    "parse_time", "round_time", "tick_step", "trial_rows",
]
