"""
gui/workspace/run_progress.py
The per-video progress table of a run: a model with one row per video
(status, percentage, ETA, last message) and the delegate that paints the
percentage as a progress bar. Shared by the Review & run stage.
"""
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtWidgets import (QApplication, QStyle, QStyledItemDelegate,
                             QStyleOptionProgressBar)

from gui.video_list import (STATUS_DONE, STATUS_FAILED, STATUS_LABELS,
                            STATUS_QUEUED, STATUS_RUNNING, STATUS_WARNING,
                            status_icon)

MAX_LISTED_PROBLEMS = 14
PROGRESS_ROLE = Qt.ItemDataRole.UserRole + 1

# ----------------------------------------------------------------------
# The per-video progress table
# ----------------------------------------------------------------------
class RunProgressModel(QAbstractTableModel):
    """One row per video of the run: status, percentage, ETA, last message."""

    COLUMNS = ("Video", "Status", "Progress", "ETA", "Message")
    VIDEO, STATUS, PROGRESS, ETA, MESSAGE = range(5)
    # Percentage that means "running, but no percentage is known".
    INDETERMINATE = -1

    def __init__(self, parent=None):
        super().__init__(parent)
        self._paths: List[Path] = []
        self._rows: Dict[Path, dict] = {}
        self._icons: Dict[str, object] = {}

    # -- content ---------------------------------------------------------
    def reset_rows(self, paths: List[Path]) -> None:
        self.beginResetModel()
        self._paths = list(paths)
        self._rows = {
            path: {"status": STATUS_QUEUED, "pct": 0, "eta": "", "message": ""}
            for path in self._paths
        }
        self.endResetModel()

    def paths(self) -> List[Path]:
        return list(self._paths)

    def row_of(self, path: Path) -> int:
        try:
            return self._paths.index(Path(path))
        except ValueError:
            return -1

    def state(self, path: Path) -> Optional[dict]:
        found = self._rows.get(Path(path))
        return dict(found) if found is not None else None

    def _touch(self, path: Path) -> None:
        row = self.row_of(path)
        if row >= 0:
            self.dataChanged.emit(
                self.index(row, 0), self.index(row, len(self.COLUMNS) - 1)
            )

    def set_status(self, path: Path, status: str, message: str = "") -> None:
        row = self._rows.get(Path(path))
        if row is None:
            return
        row["status"] = status
        if message or status in (STATUS_DONE, STATUS_WARNING, STATUS_FAILED):
            row["message"] = message
        if status in (STATUS_DONE, STATUS_WARNING):
            row["pct"], row["eta"] = 100, ""
        elif status == STATUS_FAILED:
            row["eta"] = ""
        self._touch(path)

    def set_progress(
        self, path: Path, pct: Optional[int], eta: str, message: str
    ) -> None:
        row = self._rows.get(Path(path))
        if row is None or row["status"] in (STATUS_DONE, STATUS_WARNING,
                                            STATUS_FAILED):
            return
        row["status"] = STATUS_RUNNING
        if pct is None:
            row["pct"] = self.INDETERMINATE if row["pct"] <= 0 else row["pct"]
        else:
            row["pct"] = max(int(pct), row["pct"], 0)
        if eta:
            row["eta"] = eta
        if message:
            row["message"] = message
        self._touch(path)

    # -- Qt model --------------------------------------------------------
    def rowCount(self, parent: Optional[QModelIndex] = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self._paths)

    def columnCount(self, parent: Optional[QModelIndex] = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if (
            role == Qt.ItemDataRole.DisplayRole
            and orientation == Qt.Orientation.Horizontal
        ):
            return self.COLUMNS[section]
        return None

    def _icon(self, status: str):
        if status not in self._icons:
            self._icons[status] = status_icon(status)
        return self._icons[status]

    @staticmethod
    def percent_text(pct: int) -> str:
        return "..." if pct == RunProgressModel.INDETERMINATE else f"{pct}%"

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._paths)):
            return None
        path = self._paths[index.row()]
        row = self._rows[path]
        column = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if column == self.VIDEO:
                return path.name
            if column == self.STATUS:
                return STATUS_LABELS.get(row["status"], row["status"])
            if column == self.PROGRESS:
                return self.percent_text(row["pct"])
            if column == self.ETA:
                return row["eta"]
            if column == self.MESSAGE:
                return row["message"]
        elif role == Qt.ItemDataRole.DecorationRole and column == self.STATUS:
            return self._icon(row["status"])
        elif role == Qt.ItemDataRole.ToolTipRole:
            if column == self.VIDEO:
                return str(path)
            if column == self.MESSAGE:
                return row["message"] or None
        elif role == PROGRESS_ROLE and column == self.PROGRESS:
            return row["pct"]
        return None


class ProgressBarDelegate(QStyledItemDelegate):
    """Paints the Progress column as a (possibly busy) progress bar."""

    def paint(self, painter, option, index) -> None:
        pct = index.data(PROGRESS_ROLE)
        if pct is None:
            super().paint(painter, option, index)
            return
        bar = QStyleOptionProgressBar()
        bar.rect = option.rect.adjusted(2, 3, -2, -3)
        bar.state = option.state
        bar.minimum = 0
        if pct == RunProgressModel.INDETERMINATE:
            bar.maximum = 0
            bar.progress = 0
            bar.text = ""
        else:
            bar.maximum = 100
            bar.progress = int(pct)
            bar.text = f"{int(pct)}%"
            bar.textVisible = True
        QApplication.style().drawControl(
            QStyle.ControlElement.CE_ProgressBar, bar, painter
        )



def format_problems(issues: List[str], limit: int = MAX_LISTED_PROBLEMS) -> str:
    lines = [f"- {issue}" for issue in issues[:limit]]
    if len(issues) > limit:
        lines.append(f"... and {len(issues) - limit} more")
    return "\n".join(lines)


__all__ = ["PROGRESS_ROLE", "ProgressBarDelegate", "RunProgressModel", "format_problems"]
