"""
gui/video_list.py
The video list: finding videos in files and folders, reordering and removing
entries, and the per-row status shown while a batch runs.

The list logic is pure (paths in, paths out) so it can be tested without a
window; ``VideoListWidget`` is only the thin Qt layer on top.
"""
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QListWidget, QListWidgetItem

VIDEO_EXTENSIONS = frozenset({'.mp4', '.avi', '.mov', '.mkv'})
# Result folders hold validation videos and checkpoints; never re-import them.
RUN_FOLDER_PREFIX = "analysis_run_"

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_WARNING = "warning"

STATUS_LABELS = {
    STATUS_QUEUED: "Queued",
    STATUS_RUNNING: "Running",
    STATUS_DONE: "Done",
    STATUS_FAILED: "Failed",
    STATUS_WARNING: "Done, needs quality review",
}

_STATUS_STYLE = {
    STATUS_QUEUED: ("#8a8f98", "•"),
    STATUS_RUNNING: ("#1f6feb", "▶"),
    STATUS_DONE: ("#2da44e", "✓"),
    STATUS_FAILED: ("#d1242f", "✗"),
    STATUS_WARNING: ("#d29922", "!"),
}


# ----------------------------------------------------------------------
# Finding videos
# ----------------------------------------------------------------------
def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def _skipped_folder(name: str) -> bool:
    return name.startswith(RUN_FOLDER_PREFIX) or name.startswith(".")


def find_videos_in_folder(folder: Path, recursive: bool) -> List[Path]:
    """Videos directly in ``folder``, or under it when ``recursive``.

    Run output folders and hidden folders are skipped when recursing.
    """
    found: List[Path] = []
    try:
        if not recursive:
            found = [path for path in folder.iterdir() if is_video_file(path)]
        else:
            stack = [folder]
            while stack:
                current = stack.pop()
                for entry in current.iterdir():
                    if entry.is_dir():
                        if not _skipped_folder(entry.name):
                            stack.append(entry)
                    elif is_video_file(entry):
                        found.append(entry)
    except OSError:
        pass
    return sorted(found, key=lambda path: str(path).lower())


def collect_video_files(
    paths: Iterable[Path], recursive: bool
) -> Tuple[List[Path], List[Path]]:
    """Expand dropped or picked paths into videos.

    Returns ``(videos, ignored)``: files and folders are accepted, anything
    else (documents, missing paths) is listed as ignored. Duplicates are
    removed, keeping the first occurrence.
    """
    videos: List[Path] = []
    ignored: List[Path] = []
    seen = set()

    def add(path: Path):
        if path not in seen:
            seen.add(path)
            videos.append(path)

    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for video in find_videos_in_folder(path, recursive):
                add(video)
        elif is_video_file(path):
            add(path)
        else:
            ignored.append(path)
    return videos, ignored


# ----------------------------------------------------------------------
# Editing the order
# ----------------------------------------------------------------------
def remove_at(items: Sequence, indices: Iterable[int]) -> List:
    drop = set(indices)
    return [item for index, item in enumerate(items) if index not in drop]


def move_selected(
    items: Sequence, indices: Iterable[int], offset: int
) -> Tuple[List, List[int]]:
    """Move the selected rows one step up (-1) or down (+1).

    A block of selected rows moves together and stops at the list edge;
    the second value is the selection after the move.
    """
    result = list(items)
    selected = sorted({i for i in indices if 0 <= i < len(result)})
    if not selected or offset not in (-1, 1):
        return result, selected
    order = selected if offset < 0 else list(reversed(selected))
    blocked = set()
    new_selected = []
    for index in order:
        target = index + offset
        if target < 0 or target >= len(result) or target in blocked:
            blocked.add(index)
            new_selected.append(index)
            continue
        result[index], result[target] = result[target], result[index]
        new_selected.append(target)
    return result, sorted(new_selected)


def clear_list_choices(has_configuration: bool) -> List[Tuple[str, str]]:
    """(key, label) options for Clear List.

    A loaded configuration is only dropped when the user says so.
    """
    options = []
    if has_configuration:
        options.append(("videos", "Clear videos, keep configuration"))
        options.append(("all", "Clear videos and configuration"))
    else:
        options.append(("videos", "Clear videos"))
    return options


# ----------------------------------------------------------------------
# Per-video status
# ----------------------------------------------------------------------
def outcome_statuses(
    completed: Iterable[Path],
    failures: Iterable[Tuple[Path, str]],
    quality_warnings: Iterable[Tuple[Path, str]],
) -> Dict[Path, Tuple[str, str]]:
    """Final {video: (status, message)} derived from a finished run."""
    warnings: Dict[Path, List[str]] = {}
    for path, message in quality_warnings:
        warnings.setdefault(Path(path), []).append(message)
    statuses: Dict[Path, Tuple[str, str]] = {}
    for path in completed:
        path = Path(path)
        if path in warnings:
            statuses[path] = (STATUS_WARNING, "; ".join(warnings[path][:3]))
        else:
            statuses[path] = (STATUS_DONE, "")
    for path, message in failures:
        statuses[Path(path)] = (STATUS_FAILED, message)
    return statuses


def status_icon(status: str) -> QIcon:
    """A small colored badge with a glyph, so status never relies on color alone."""
    color, glyph = _STATUS_STYLE.get(status, _STATUS_STYLE[STATUS_QUEUED])
    size = 16
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(color))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(0, 0, size, size)
    painter.setPen(QColor("white"))
    font = QFont()
    font.setPixelSize(11)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, glyph)
    painter.end()
    return QIcon(pixmap)


class VideoListWidget(QListWidget):
    """Multi-select list of videos with keyboard shortcuts for editing it."""

    remove_requested = pyqtSignal()
    move_requested = pyqtSignal(int)  # -1 up, +1 down

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setIconSize(QSize(16, 16))
        self.setAlternatingRowColors(True)

    def keyPressEvent(self, event):
        key = event.key()
        alt = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and not alt:
            self.remove_requested.emit()
            event.accept()
        elif alt and key == Qt.Key.Key_Up:
            self.move_requested.emit(-1)
            event.accept()
        elif alt and key == Qt.Key.Key_Down:
            self.move_requested.emit(1)
            event.accept()
        else:
            super().keyPressEvent(event)

    def set_videos(
        self,
        videos: Sequence[Path],
        statuses: Optional[Dict[Path, Tuple[str, str]]] = None,
        selected: Iterable[int] = (),
    ):
        """Rebuild the rows, keeping each video's status and the selection."""
        statuses = statuses or {}
        wanted = set(selected)
        self.clear()
        for index, video in enumerate(videos):
            item = QListWidgetItem(str(video))
            item.setData(Qt.ItemDataRole.UserRole, str(video))
            status, message = statuses.get(video, (STATUS_QUEUED, ""))
            item.setIcon(status_icon(status))
            tip = STATUS_LABELS.get(status, status)
            item.setToolTip(f"{tip}: {message}" if message else tip)
            self.addItem(item)
            if index in wanted:
                item.setSelected(True)

    def selected_rows(self) -> List[int]:
        return sorted(self.row(item) for item in self.selectedItems())
