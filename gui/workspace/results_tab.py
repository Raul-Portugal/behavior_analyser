"""
gui/workspace/results_tab.py
The Results tab of the workspace: open a finished run and look at what it
produced.

::

    run: [analysis_run_... v]  [Open run folder...] [Compare runs]
    one-line summary of the run
    +- Videos ----------------------------------+- Time bins -+- Compare -+
    | batch_summary.csv (sort, filter, copy,    |
    |   export)                                 |
    | the selected video: Overview | Plots | Video |
    +-------------------------------------------+

* The run picker lists the project's runs (newest first) and any folder
  opened with "Open run folder..."; manifest schema 3 is read in full,
  schema 2 and older as far as they go.
* **Videos**: the batch summary as a sortable, filterable table. Selecting a
  row shows that video's key metrics, quality warnings, plots (click to
  enlarge) and its video: the validation video when one was made, else the
  original video with the tracking drawn on it.
* **Time bins**: a metric over the bins, per video or averaged per group of a
  metadata column.
* **Compare runs**: two batch summaries side by side, changed values marked.

The data logic is in :mod:`gui.workspace.results_data` and the overlays in
:mod:`gui.workspace.overlay`; this module is the Qt layer.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import (QAbstractTableModel, QModelIndex, QSize,
                          QSortFilterProxyModel, Qt, pyqtSignal)
from PyQt6.QtGui import QBrush, QColor, QIcon, QKeySequence, QPixmap
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                             QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QListWidget, QListWidgetItem, QMenu,
                             QPushButton, QSplitter, QTableView, QTableWidget,
                             QTableWidgetItem, QTabWidget, QVBoxLayout,
                             QWidget)

from core.positional_validation import load_tracking_csv
from core.project import Project
from core.tst_validation import load_tst_evidence_csv
from gui.video_list import STATUS_WARNING, status_icon
from gui.video_player import VideoPlayerWidget
from gui.workspace import overlay as overlays
from gui.workspace.results_data import (RunData, RunDiff,
                                        RunLoadError, RunVideo, Series,
                                        diff_summaries, key_metrics,
                                        looks_like_run, norm, time_bin_group_columns,
                                        time_bin_metrics, time_bin_series,
                                        time_bin_subjects, to_float,
                                        write_table)
from gui.workspace.tabs import WorkspaceTab

logger = logging.getLogger(__name__)

ALL_SUBJECTS = "(all)"
NO_GROUP = "(each video)"
CHANGED_BRUSH = QBrush(QColor(255, 224, 130))
CHANGED_TEXT = QBrush(QColor(40, 30, 0))
ONLY_BRUSH = QBrush(QColor(200, 200, 200))
THUMB_SIZE = QSize(180, 135)

SOURCE_VALIDATION = "validation"
SOURCE_OVERLAY = "overlay"


# ----------------------------------------------------------------------
# The summary table
# ----------------------------------------------------------------------
class SummaryModel(QAbstractTableModel):
    """``batch_summary.csv`` as a read-only table (the cells are the CSV's)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._run: Optional[RunData] = None
        self._headers: List[str] = []
        self._rows: List[List[str]] = []
        self._icons: Dict[str, object] = {}

    def set_run(self, run: Optional[RunData]) -> None:
        self.beginResetModel()
        self._run = run
        self._headers = list(run.summary.headers) if run else []
        self._rows = [list(r) for r in run.summary.rows] if run else []
        self.endResetModel()

    @property
    def headers(self) -> List[str]:
        return list(self._headers)

    def rowCount(self, parent: Optional[QModelIndex] = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self._rows)

    def columnCount(self, parent: Optional[QModelIndex] = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self._headers)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return self._headers[section] if section < len(self._headers) else None
            return section + 1
        return None

    def _icon(self, status: str):
        if status not in self._icons:
            self._icons[status] = status_icon(status)
        return self._icons[status]

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row, column = index.row(), index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            return self._rows[row][column]
        video = self._run.video_for_row(row) if self._run else None
        if column == 0 and video is not None:
            if role == Qt.ItemDataRole.DecorationRole and video.warnings:
                return self._icon(STATUS_WARNING)
            if role == Qt.ItemDataRole.ToolTipRole and video.warnings:
                return "\n".join(video.warnings)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if to_float(self._rows[row][column]) is not None:
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None


class SummaryProxy(QSortFilterProxyModel):
    """Numeric-aware sorting and a contains-text filter over every column."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text = ""
        self.setSortRole(Qt.ItemDataRole.DisplayRole)

    def set_text_filter(self, text: str) -> None:
        self._text = text.strip().lower()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, parent) -> bool:
        if not self._text:
            return True
        model = self.sourceModel()
        return any(
            self._text in str(
                model.index(source_row, column, parent).data() or ""
            ).lower()
            for column in range(model.columnCount())
        )

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        a, b = left.data() or "", right.data() or ""
        x, y = to_float(a), to_float(b)
        if x is not None and y is not None:
            return x < y
        if x is not None:       # numbers before text ("N/A", blanks)
            return True
        if y is not None:
            return False
        return str(a).lower() < str(b).lower()


class CopyableTableView(QTableView):
    """A table that tells its owner when Ctrl+C is pressed."""

    copy_requested = pyqtSignal()

    def keyPressEvent(self, event) -> None:
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copy_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
class PlotDialog(QDialog):
    """A plot at full size (scaled to the window)."""

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(Path(path).name)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self._source = QPixmap(str(path))
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(200, 150)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        if self._source.isNull():
            self.label.setText(f"Cannot read {Path(path).name}")
        else:
            self.resize(min(self._source.width() + 24, 1000),
                        min(self._source.height() + 24, 760))
        self._rescale()

    @property
    def shown_pixmap(self) -> QPixmap:
        return self._source

    def _rescale(self) -> None:
        if self._source.isNull():
            return
        self.label.setPixmap(self._source.scaled(
            self.label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()


# ----------------------------------------------------------------------
# The video of one result
# ----------------------------------------------------------------------
def overlay_for_video(
    run: RunData, video: RunVideo
) -> Tuple[Optional[Callable], str]:
    """``(overlay callback, message)`` for the original video of a result."""
    rois = overlays.load_video_rois(run.run_dir, video.source)
    tracking = video.tracking_csv()
    if tracking is not None:
        try:
            return overlays.tracking_overlay(load_tracking_csv(tracking), rois), ""
        except (OSError, ValueError) as exc:
            return None, f"Cannot read the tracking data: {exc}"
    evidence = video.evidence_csv()
    if evidence is not None:
        try:
            return overlays.tst_overlay(load_tst_evidence_csv(evidence), rois), ""
        except (OSError, ValueError) as exc:
            return None, f"Cannot read the motion evidence: {exc}"
    return None, "This result has no per-frame data to draw."


class VideoPane(QWidget):
    """The validation video, or the original video with an overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._run: Optional[RunData] = None
        self._video: Optional[RunVideo] = None
        self._active = False
        self._loaded_for: Optional[Tuple] = None
        self.source_combo = QComboBox()
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.player = VideoPlayerWidget(self)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Show:"))
        bar.addWidget(self.source_combo, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.message_label)
        layout.addWidget(self.player, 1)

    # -- content ---------------------------------------------------------
    def set_video(self, run: Optional[RunData], video: Optional[RunVideo]) -> None:
        self._run, self._video = run, video
        self._loaded_for = None
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        if run is not None and video is not None:
            for media in video.media():
                self.source_combo.addItem(
                    f"Validation video ({media.name})",
                    (SOURCE_VALIDATION, str(media)),
                )
            self.source_combo.addItem(
                "Original video with tracking overlay", (SOURCE_OVERLAY, "")
            )
        self.source_combo.blockSignals(False)
        self.player.close_video()
        self.message_label.setText(
            "" if video is not None else "Select a video in the table."
        )
        if self._active:
            self.load()

    def activate(self) -> None:
        self._active = True
        self.load()

    def deactivate(self) -> None:
        self._active = False
        self.player.pause()

    def _on_source_changed(self, _index: int) -> None:
        self._loaded_for = None
        if self._active:
            self.load()

    def current_source(self) -> Tuple[str, str]:
        data = self.source_combo.currentData()
        return data if data else ("", "")

    def load(self) -> bool:
        """Open the chosen source in the player (once per choice)."""
        run, video = self._run, self._video
        kind, path = self.current_source()
        if run is None or video is None or not kind:
            self.player.close_video()
            return False
        if self._loaded_for == (id(video), kind, path):
            return True
        self._loaded_for = (id(video), kind, path)
        self.player.set_overlay(None)
        if kind == SOURCE_VALIDATION:
            self.message_label.setText("")
            return self.player.open_video(Path(path))
        source = video.source
        if source is None or not Path(source).is_file():
            self.player.close_video()
            self.message_label.setText(
                f"The original video was not found: {source or video.name}"
            )
            return False
        callback, message = overlay_for_video(run, video)
        self.message_label.setText(message)
        if not self.player.open_video(Path(source)):
            return False
        self.player.set_overlay(callback)
        return True

    def close(self) -> bool:
        self.player.close_video()
        self._loaded_for = None
        return True

    def hideEvent(self, event) -> None:
        self.player.pause()
        super().hideEvent(event)


# ----------------------------------------------------------------------
# The per-video detail
# ----------------------------------------------------------------------
class VideoDetail(QWidget):
    """Overview, plots and video of the video selected in the summary."""

    plot_requested = pyqtSignal(object)      # Path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._run: Optional[RunData] = None
        self.video: Optional[RunVideo] = None
        self.title_label = QLabel("Select a video in the table.")
        self.title_label.setStyleSheet("font-weight: bold;")
        self.warnings_label = QLabel()
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setStyleSheet("color: #b45309;")
        self.warnings_label.setVisible(False)

        self.metrics_table = QTableWidget(0, 3)
        self.metrics_table.setHorizontalHeaderLabels(["Section", "Metric", "Value"])
        self.metrics_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.notes_label = QLabel()
        self.notes_label.setWordWrap(True)
        overview = QWidget()
        overview_layout = QVBoxLayout(overview)
        overview_layout.setContentsMargins(0, 0, 0, 0)
        overview_layout.addWidget(self.metrics_table, 1)
        overview_layout.addWidget(self.notes_label)

        self.plot_list = QListWidget()
        self.plot_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.plot_list.setIconSize(THUMB_SIZE)
        self.plot_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.plot_list.setMovement(QListWidget.Movement.Static)
        self.plot_list.setWordWrap(True)
        self.plot_list.setToolTip("Click a plot to enlarge it")
        self.plot_list.itemClicked.connect(self._on_plot_clicked)

        self.video_pane = VideoPane()
        self.tabs = QTabWidget()
        self.tabs.addTab(overview, "Overview")
        self.tabs.addTab(self.plot_list, "Plots")
        self.tabs.addTab(self.video_pane, "Video")
        self.tabs.currentChanged.connect(self._on_tab_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.title_label)
        layout.addWidget(self.warnings_label)
        layout.addWidget(self.tabs, 1)

    # -- content ---------------------------------------------------------
    def set_video(self, run: Optional[RunData], video: Optional[RunVideo]) -> None:
        self._run, self.video = run, video
        self.metrics_table.setRowCount(0)
        self.plot_list.clear()
        self.notes_label.setText("")
        if video is None:
            self.title_label.setText("Select a video in the table.")
            self.warnings_label.setVisible(False)
            self.video_pane.set_video(run, None)
            return
        status = "" if video.status == "completed" else f" ({video.status})"
        self.title_label.setText(f"{video.name}{status}")
        problems = list(video.warnings)
        if video.failure:
            problems.insert(0, f"Failed: {video.failure}")
        self.warnings_label.setText(
            "Quality warnings:\n" + "\n".join(f"- {p}" for p in problems)
            if problems else ""
        )
        self.warnings_label.setVisible(bool(problems))
        self._fill_metrics(video)
        self._fill_plots(video)
        self.video_pane.set_video(run, video)

    def _fill_metrics(self, video: RunVideo) -> None:
        summary = video.load_summary()
        rows, notes = key_metrics(summary)
        for key, value in video.metadata.items():
            rows.insert(0, ("sample", key, value))
        self.metrics_table.setRowCount(len(rows))
        for index, (section, metric, value) in enumerate(rows):
            for column, text in enumerate((section, metric, value)):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.metrics_table.setItem(index, column, item)
        self.metrics_table.resizeColumnsToContents()
        if not summary:
            self.notes_label.setText("This video has no summary.json.")
        elif notes:
            self.notes_label.setText("Notes: " + " ".join(notes[:4]))

    def _fill_plots(self, video: RunVideo) -> None:
        for path in video.plots():
            pixmap = QPixmap(str(path))
            if pixmap.isNull():
                continue
            label = path.stem
            if label.startswith(video.folder.name + "_"):
                label = label[len(video.folder.name) + 1:]
            item = QListWidgetItem(label.replace("_", " "))
            item.setIcon(QIcon(self._thumbnail(pixmap)))
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(str(path))
            self.plot_list.addItem(item)

    @staticmethod
    def _thumbnail(pixmap: QPixmap):
        return pixmap.scaled(
            THUMB_SIZE, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def plot_paths(self) -> List[Path]:
        return [
            Path(self.plot_list.item(i).data(Qt.ItemDataRole.UserRole))
            for i in range(self.plot_list.count())
        ]

    def metric_rows(self) -> List[Tuple[str, str, str]]:
        return [
            tuple(self.metrics_table.item(r, c).text() for c in range(3))
            for r in range(self.metrics_table.rowCount())
        ]

    def _on_plot_clicked(self, item: QListWidgetItem) -> None:
        self.plot_requested.emit(Path(item.data(Qt.ItemDataRole.UserRole)))

    def show_video_tab(self) -> None:
        self.tabs.setCurrentWidget(self.video_pane)

    def _on_tab_changed(self, _index: int) -> None:
        if self.tabs.currentWidget() is self.video_pane:
            self.video_pane.activate()
        else:
            self.video_pane.deactivate()

    def hideEvent(self, event) -> None:
        self.video_pane.deactivate()
        super().hideEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.tabs.currentWidget() is self.video_pane:
            self.video_pane.activate()


# ----------------------------------------------------------------------
# Time bins
# ----------------------------------------------------------------------
class TimeBinsPanel(QWidget):
    """A metric over the time bins, per video or per metadata group."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._run: Optional[RunData] = None
        self._loading = False
        self.series: List[Series] = []
        self.metric_combo = QComboBox()
        self.subject_combo = QComboBox()
        self.group_combo = QComboBox()
        for combo in (self.metric_combo, self.subject_combo, self.group_combo):
            combo.currentIndexChanged.connect(self._on_choice_changed)
        self.subject_label = QLabel("Mouse:")
        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.figure = Figure(figsize=(6, 3.5), layout="constrained")
        self.axes = self.figure.subplots()
        self.canvas = FigureCanvasQTAgg(self.figure)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Metric:"))
        bar.addWidget(self.metric_combo, 1)
        bar.addWidget(self.subject_label)
        bar.addWidget(self.subject_combo)
        bar.addWidget(QLabel("Group by:"))
        bar.addWidget(self.group_combo)
        layout = QVBoxLayout(self)
        layout.addLayout(bar)
        layout.addWidget(self.message_label)
        layout.addWidget(self.canvas, 1)
        self.set_run(None)

    def set_run(self, run: Optional[RunData]) -> None:
        self._run = run
        table = run.time_bins if run is not None else None
        self._loading = True
        try:
            keep = (self.metric_combo.currentText(), self.group_combo.currentText())
            self.metric_combo.clear()
            self.metric_combo.addItems(time_bin_metrics(table))
            self.subject_combo.clear()
            subjects = time_bin_subjects(table)
            if subjects:
                self.subject_combo.addItems([ALL_SUBJECTS] + subjects)
            self.subject_combo.setVisible(bool(subjects))
            self.subject_label.setVisible(bool(subjects))
            self.group_combo.clear()
            self.group_combo.addItems(
                [NO_GROUP] + time_bin_group_columns(table)
            )
            for combo, wanted in ((self.metric_combo, keep[0]),
                                  (self.group_combo, keep[1])):
                index = combo.findText(wanted)
                if index >= 0:
                    combo.setCurrentIndex(index)
        finally:
            self._loading = False
        self.render()

    def set_metric(self, metric: str) -> None:
        index = self.metric_combo.findText(metric)
        if index >= 0:
            self.metric_combo.setCurrentIndex(index)

    def set_group_by(self, column: Optional[str]) -> None:
        index = self.group_combo.findText(column or NO_GROUP)
        if index >= 0:
            self.group_combo.setCurrentIndex(index)

    def set_subject(self, subject: Optional[str]) -> None:
        index = self.subject_combo.findText(subject or ALL_SUBJECTS)
        if index >= 0:
            self.subject_combo.setCurrentIndex(index)

    def _on_choice_changed(self, _index: int) -> None:
        if not self._loading:
            self.render()

    def render(self) -> None:
        table = self._run.time_bins if self._run is not None else None
        axes = self.axes
        axes.clear()
        self.series = []
        metric = self.metric_combo.currentText()
        if table is None or not metric:
            self.message_label.setText(
                "This run has no time bins. Set Time bins in the Run tab "
                "before running to get them."
            )
            axes.set_axis_off()
            self.canvas.draw_idle()
            return
        axes.set_axis_on()
        subject = self.subject_combo.currentText()
        subject = "" if subject in ("", ALL_SUBJECTS) else subject
        group = self.group_combo.currentText()
        group = "" if group in ("", NO_GROUP) else group
        self.series = time_bin_series(table, metric, subject, group)
        for series in self.series:
            label = f"{series.label} (n={series.n})" if group else series.label
            if group or any(series.err):
                axes.errorbar(series.x, series.y, yerr=series.err, marker="o",
                              capsize=3, label=label)
            else:
                axes.plot(series.x, series.y, marker="o", label=label)
        unit = self.series[0].unit if self.series else ""
        axes.set_xlabel("Time since the start of the analysis (s)")
        axes.set_ylabel(f"{metric.replace('_', ' ')}" + (f" ({unit})" if unit else ""))
        axes.set_title(
            metric.replace("_", " ") + (f" by {group}" if group else "")
        )
        axes.grid(True, alpha=0.3)
        if 0 < len(self.series) <= 12:
            axes.legend(fontsize="small")
        self.message_label.setText(
            "" if self.series else "No numeric values for this selection."
        )
        self.canvas.draw_idle()


# ----------------------------------------------------------------------
# Compare runs
# ----------------------------------------------------------------------
class ComparePanel(QWidget):
    """The batch summaries of two runs, changed values highlighted."""

    def __init__(self, load_run: Callable[[Path], RunData], parent=None):
        super().__init__(parent)
        self._load_run = load_run
        self.diff: Optional[RunDiff] = None
        self.combo_a = QComboBox()
        self.combo_b = QComboBox()
        self.compare_button = QPushButton("Compare")
        self.compare_button.clicked.connect(self.compare)
        self.only_changed_check = QCheckBox("Only changed videos")
        self.only_changed_check.toggled.connect(self._fill)
        self.summary_label = QLabel(
            "Choose two runs (for example before and after re-analysing)."
        )
        self.summary_label.setWordWrap(True)
        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Before (A):"))
        bar.addWidget(self.combo_a, 1)
        bar.addWidget(QLabel("After (B):"))
        bar.addWidget(self.combo_b, 1)
        bar.addWidget(self.compare_button)
        layout = QVBoxLayout(self)
        layout.addLayout(bar)
        layout.addWidget(self.only_changed_check)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.table, 1)

    def set_runs(self, entries: List["RunEntry"]) -> None:
        keep = (self.combo_a.currentData(), self.combo_b.currentData())
        for combo, wanted in ((self.combo_a, keep[0]), (self.combo_b, keep[1])):
            combo.blockSignals(True)
            combo.clear()
            for entry in entries:
                combo.addItem(entry.label, str(entry.path))
            index = combo.findData(wanted) if wanted else -1
            if index >= 0:
                combo.setCurrentIndex(index)
            combo.blockSignals(False)

    def choose(self, a: Optional[Path], b: Optional[Path]) -> None:
        for combo, path in ((self.combo_a, a), (self.combo_b, b)):
            if path is not None:
                index = combo.findData(str(path))
                if index >= 0:
                    combo.setCurrentIndex(index)

    def compare(self) -> Optional[RunDiff]:
        a, b = self.combo_a.currentData(), self.combo_b.currentData()
        if not a or not b:
            self.summary_label.setText("There are no runs to compare.")
            return None
        if a == b:
            self.summary_label.setText("Choose two different runs.")
            self.diff = None
            self._fill()
            return None
        try:
            run_a, run_b = self._load_run(Path(a)), self._load_run(Path(b))
        except RunLoadError as exc:
            self.summary_label.setText(f"Cannot compare: {exc}")
            self.diff = None
            self._fill()
            return None
        self.diff = diff_summaries(run_a.summary, run_b.summary)
        diff = self.diff
        only = sum(1 for r in diff.rows if r.status != "both")
        text = (
            f"{diff.changed_cells} value(s) changed in "
            f"{sum(1 for r in diff.rows if r.status == 'both' and r.changed)}"
            f" of {len(diff.rows)} video(s)."
        )
        if only:
            text += f" {only} video(s) are in only one of the runs."
        self.summary_label.setText(text)
        self._fill()
        return diff

    def _fill(self, *_args) -> None:
        table = self.table
        table.clear()
        diff = self.diff
        if diff is None:
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        rows = [
            row for row in diff.rows
            if row.changed or not self.only_changed_check.isChecked()
        ]
        table.setColumnCount(len(diff.headers))
        table.setHorizontalHeaderLabels(diff.headers)
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, cell in enumerate(row.cells):
                if cell.changed:
                    text = f"{cell.a or '(blank)'} → {cell.b or '(blank)'}"
                elif row.status == "only_b":
                    text = cell.b
                else:
                    text = cell.a
                item = QTableWidgetItem(text)
                if cell.changed:
                    item.setBackground(CHANGED_BRUSH)
                    item.setForeground(CHANGED_TEXT)
                    item.setToolTip(f"A: {cell.a}\nB: {cell.b}")
                elif row.status != "both":
                    item.setBackground(ONLY_BRUSH)
                    item.setForeground(CHANGED_TEXT)
                    item.setToolTip("Only in run " + ("A" if row.status == "only_a" else "B"))
                table.setItem(r, c, item)
        table.resizeColumnsToContents()

    def changed_cells(self) -> List[Tuple[str, str]]:
        """``(video, column)`` of every changed value."""
        if self.diff is None:
            return []
        return [
            (row.video, self.diff.headers[c])
            for row in self.diff.rows
            for c, cell in enumerate(row.cells) if cell.changed
        ]


# ----------------------------------------------------------------------
# The tab
# ----------------------------------------------------------------------
@dataclass
class RunEntry:
    path: Path
    label: str
    external: bool = False
    exists: bool = True


def run_label(path: Path, completed: Optional[int] = None, failed: int = 0,
              cancelled: bool = False, started: str = "") -> str:
    parts: List[str] = []
    if completed is not None:
        parts.append(f"{completed} done")
    if failed:
        parts.append(f"{failed} failed")
    if cancelled:
        parts.append("cancelled")
    suffix = f"  ({', '.join(parts)})" if parts else ""
    return f"{path.name}{suffix}"


class ResultsTab(WorkspaceTab):
    """Open a run folder and review its results."""

    run_loaded = pyqtSignal(object)         # RunData

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.title = "Results"
        self.run: Optional[RunData] = None
        self._entries: List[RunEntry] = []
        self._external: List[Path] = []
        self._loading = False
        self._plot_dialogs: List[PlotDialog] = []
        self._build_ui()
        self._refresh_runs()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.run_combo = QComboBox()
        self.run_combo.setMinimumContentsLength(30)
        self.run_combo.currentIndexChanged.connect(self._on_run_picked)
        self.open_button = QPushButton("Open run folder...")
        self.open_button.clicked.connect(self.open_run_folder)
        self.folder_button = QPushButton("Show folder")
        self.folder_button.clicked.connect(self.show_run_folder)
        self.compare_button = QPushButton("Compare runs")
        self.compare_button.clicked.connect(self.show_compare)
        self.info_label = QLabel("Run a batch or open a run folder.")
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        # -- videos --------------------------------------------------------
        self.model = SummaryModel(self)
        self.proxy = SummaryProxy(self)
        self.proxy.setSourceModel(self.model)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter the table...")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self.proxy.set_text_filter)
        self.export_button = QPushButton("Export CSV...")
        self.export_button.clicked.connect(self.export_csv)
        self.copy_button = QPushButton("Copy")
        self.copy_button.setToolTip("Copy the selected cells (Ctrl+C)")
        self.copy_button.clicked.connect(self.copy_selection)
        self.rows_label = QLabel()
        filter_bar = QHBoxLayout()
        filter_bar.addWidget(self.filter_edit, 1)
        filter_bar.addWidget(self.rows_label)
        filter_bar.addWidget(self.copy_button)
        filter_bar.addWidget(self.export_button)

        self.table = CopyableTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(-1, Qt.SortOrder.AscendingOrder)   # CSV order
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_table_menu)
        self.table.copy_requested.connect(self.copy_selection)
        self.table.selectionModel().currentRowChanged.connect(self._on_row_changed)
        self.proxy.rowsInserted.connect(self._update_rows_label)
        self.proxy.rowsRemoved.connect(self._update_rows_label)
        self.proxy.modelReset.connect(self._update_rows_label)
        self.proxy.layoutChanged.connect(self._update_rows_label)

        table_box = QWidget()
        table_layout = QVBoxLayout(table_box)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addLayout(filter_bar)
        table_layout.addWidget(self.table, 1)

        self.detail = VideoDetail()
        self.detail.plot_requested.connect(self.open_plot)
        videos = QSplitter(Qt.Orientation.Vertical)
        videos.addWidget(table_box)
        videos.addWidget(self.detail)
        videos.setStretchFactor(0, 1)
        videos.setStretchFactor(1, 2)

        self.bins_panel = TimeBinsPanel()
        self.compare_panel = ComparePanel(self.run_data)

        self.sections = QTabWidget()
        self.sections.addTab(videos, "Videos")
        self.sections.addTab(self.bins_panel, "Time bins")
        self.sections.addTab(self.compare_panel, "Compare runs")

        top = QHBoxLayout()
        top.addWidget(QLabel("Run:"))
        top.addWidget(self.run_combo, 1)
        top.addWidget(self.open_button)
        top.addWidget(self.folder_button)
        top.addWidget(self.compare_button)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.info_label)
        layout.addWidget(self.sections, 1)

    # ------------------------------------------------------------------
    # Workspace hooks
    # ------------------------------------------------------------------
    def attach_workspace(self, window) -> None:
        # Release the decoder thread before the window goes away.
        window.register_close_guard(self._release_media)
        super().attach_workspace(window)

    @property
    def project(self) -> Optional[Project]:
        return None if self.workspace is None else self.workspace.project

    def on_project_changed(self, project: Optional[Project]) -> None:
        self._external.clear()
        self._clear_run()
        self._refresh_runs()

    def on_project_modified(self) -> None:
        signature = [str(e.path) for e in self._entries if not e.external]
        project = self.project
        wanted = [str(r.run_dir) for r in self._sorted_runs()] if project else []
        if signature != wanted:
            self._refresh_runs()

    def on_selection_changed(self, keys: List[str]) -> None:
        project, run = self.project, self.run
        if project is None or run is None or not keys:
            return
        try:
            entries = project.entries_for(keys[:1])
        except KeyError:
            return
        if not entries:
            return
        row = run.row_for_source(entries[0].path)
        if row >= 0:
            self.select_row(row)

    def on_tab_activated(self) -> None:
        if self.run is None and self._entries:
            self._load_newest()

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------
    def _sorted_runs(self):
        project = self.project
        if project is None:
            return []
        return sorted(
            project.runs,
            key=lambda r: (r.started_at or "", r.run_dir.name),
            reverse=True,
        )

    def _refresh_runs(self) -> None:
        entries: List[RunEntry] = []
        seen = set()
        for record in self._sorted_runs():
            exists = Path(record.run_dir).is_dir()
            label = run_label(
                record.run_dir, record.completed, record.failed, record.cancelled
            ) + ("" if exists else " - folder not found")
            entries.append(RunEntry(Path(record.run_dir), label, False, exists))
            seen.add(norm(record.run_dir))
        for path in self._external:
            if norm(path) not in seen:
                entries.append(RunEntry(path, f"{run_label(path)} - opened", True))
                seen.add(norm(path))
        self._entries = entries
        current = self.run.run_dir if self.run is not None else None
        self._loading = True
        try:
            self.run_combo.clear()
            for entry in entries:
                self.run_combo.addItem(entry.label, str(entry.path))
            if current is not None:
                index = self.run_combo.findData(str(current))
                if index >= 0:
                    self.run_combo.setCurrentIndex(index)
            elif entries:
                self.run_combo.setCurrentIndex(-1)
        finally:
            self._loading = False
        self.compare_panel.set_runs(entries)
        self.compare_button.setEnabled(len(entries) >= 2)

    def run_paths(self) -> List[Path]:
        return [entry.path for entry in self._entries]

    def _load_newest(self) -> bool:
        for entry in self._entries:
            if entry.exists and self.load_run(entry.path, quiet=True):
                return True
        return False

    def _on_run_picked(self, index: int) -> None:
        if self._loading or index < 0:
            return
        path = self.run_combo.itemData(index)
        if path:
            self.load_run(Path(path), quiet=True)

    @staticmethod
    def run_data(path: Path) -> RunData:
        """A fresh :class:`RunData` of ``path`` (what is on disk now)."""
        return RunData.load(path)

    def load_run(self, path: Path, quiet: bool = False) -> bool:
        """Load a run folder into the tab; False (with a message) if unreadable."""
        path = Path(path)
        try:
            run = self.run_data(path)
        except (RunLoadError, OSError, ValueError) as exc:
            self._clear_run()
            self.info_label.setText(f"Cannot open {path.name}: {exc}")
            if not quiet and self.workspace is not None:
                self.workspace.prompts.error("Open run", f"Cannot open {path}:\n{exc}")
            return False
        self._set_run(run)
        return True

    def _clear_run(self) -> None:
        self._set_run(None)
        self.info_label.setText("Run a batch or open a run folder.")

    def _set_run(self, run: Optional[RunData]) -> None:
        self.run = run
        self.detail.set_video(run, None)
        self.model.set_run(run)
        self.table.resizeColumnsToContents()
        self.bins_panel.set_run(run)
        self._loading = True
        try:
            if run is None:
                self.run_combo.setCurrentIndex(-1)
            else:
                index = self.run_combo.findData(str(run.run_dir))
                if index < 0:
                    self._external.append(run.run_dir)
                    self._refresh_runs()
                    index = self.run_combo.findData(str(run.run_dir))
                self.run_combo.setCurrentIndex(index)
        finally:
            self._loading = False
        if run is not None:
            text = run.describe()
            if run.notes:
                text += "\n" + " ".join(run.notes)
            if run.failures:
                text += "\nFailed: " + "; ".join(
                    f"{Path(src).name}: {msg}" for src, msg in run.failures[:3]
                )
            self.info_label.setText(text)
            self.info_label.setToolTip(str(run.run_dir))
            self.run_loaded.emit(run)
        self._update_rows_label()
        self.export_button.setEnabled(run is not None and bool(run.summary.rows))
        self.copy_button.setEnabled(run is not None and bool(run.summary.rows))
        self.folder_button.setEnabled(run is not None)

    def show_run(self, run_dir: Path) -> bool:
        """Show a run (the Run tab calls this when a run finishes)."""
        path = Path(run_dir)
        if norm(path) not in {norm(p) for p in self.run_paths()}:
            self._external.append(path)
            self._refresh_runs()
        return self.load_run(path, quiet=False)

    def open_run_folder(self) -> bool:
        """Pick any run folder (a run from another project or machine)."""
        if self.workspace is None:
            return False
        start = self.workspace.app_settings.dialog_dir("output")
        project = self.project
        if project is not None and project.base_dir is not None:
            runs = project.base_dir / "runs"
            start = str(runs if runs.is_dir() else project.base_dir)
        folder = self.workspace.prompts.choose_folder("Open a run folder", start)
        if folder is None:
            return False
        if not looks_like_run(folder):
            self.workspace.prompts.error(
                "Open run folder",
                f"{folder} does not look like an analysis run (no "
                "manifest.json or batch_summary.csv).",
            )
            return False
        return self.show_run(folder)

    def show_run_folder(self) -> bool:
        if self.run is None or self.workspace is None:
            return False
        return bool(self.workspace.prompts.open_folder(self.run.run_dir))

    def show_compare(self) -> None:
        """Switch to the comparison, preselecting this run and the one before."""
        paths = self.run_paths()
        current = self.run.run_dir if self.run is not None else None
        b = current or (paths[0] if paths else None)
        a = next((p for p in paths if norm(p) != norm(b)), None) if b else None
        self.compare_panel.choose(a, b)
        self.sections.setCurrentWidget(self.compare_panel)

    # ------------------------------------------------------------------
    # The summary table
    # ------------------------------------------------------------------
    def _update_rows_label(self, *_args) -> None:
        total = self.model.rowCount()
        shown = self.proxy.rowCount()
        self.rows_label.setText(
            "" if not total else
            f"{shown} of {total} row(s)" if shown != total else f"{total} row(s)"
        )

    def table_rows(self) -> List[List[str]]:
        """The rows as shown (filtered and sorted)."""
        return [
            [self.proxy.index(r, c).data() or "" for c in range(self.proxy.columnCount())]
            for r in range(self.proxy.rowCount())
        ]

    def select_row(self, source_row: int) -> bool:
        """Select a row of the batch summary (a row of the CSV)."""
        proxy_index = self.proxy.mapFromSource(self.model.index(source_row, 0))
        if not proxy_index.isValid():
            return False
        self.table.selectRow(proxy_index.row())
        self.table.setCurrentIndex(proxy_index)
        return True

    def selected_video(self) -> Optional[RunVideo]:
        return self.detail.video

    def _on_row_changed(self, current: QModelIndex, _previous) -> None:
        if self.run is None or not current.isValid():
            return
        source = self.proxy.mapToSource(current)
        self.detail.set_video(self.run, self.run.video_for_row(source.row()))

    def _selected_cells(self) -> List[List[Tuple[int, int]]]:
        indexes = sorted(
            self.table.selectionModel().selectedIndexes(),
            key=lambda i: (i.row(), i.column()),
        )
        rows: Dict[int, List[Tuple[int, int]]] = {}
        for index in indexes:
            rows.setdefault(index.row(), []).append((index.row(), index.column()))
        return [rows[r] for r in sorted(rows)]

    def copy_selection(self, with_headers: bool = False) -> str:
        """Copy the selected cells as tab-separated text; returns it."""
        if self.workspace is None:
            return ""
        lines: List[str] = []
        selected = self._selected_cells()
        if selected and with_headers is True:
            columns = [c for _, c in selected[0]]
            lines.append("\t".join(
                str(self.proxy.headerData(c, Qt.Orientation.Horizontal) or "")
                for c in columns
            ))
        for row in selected:
            lines.append("\t".join(
                str(self.proxy.index(r, c).data() or "") for r, c in row
            ))
        text = "\n".join(lines)
        if text:
            self.workspace.prompts.copy_text(text)
        return text

    def export_csv(self, path: Optional[Path] = None) -> bool:
        """Write the table as shown (filter and order applied) to a CSV file."""
        run = self.run
        if run is None or self.workspace is None:
            return False
        if path is None:
            path = self.workspace.prompts.save_file(
                "Export the table",
                str(run.run_dir / f"{run.name}_table.csv"),
                "CSV files (*.csv);;All Files (*.*)",
            )
            if path is None:
                return False
        try:
            write_table(path, self.model.headers, self.table_rows())
        except OSError as exc:
            self.workspace.prompts.error(
                "Export the table", f"Cannot write {Path(path).name}:\n{exc}"
            )
            return False
        return True

    def _show_table_menu(self, position) -> None:
        index = self.table.indexAt(position)
        menu = QMenu(self)
        menu.addAction("Copy", self.copy_selection)
        menu.addAction("Copy with headers",
                       lambda: self.copy_selection(with_headers=True))
        menu.addAction("Export table as CSV...", self.export_csv)
        video = None
        if index.isValid() and self.run is not None:
            video = self.run.video_for_row(self.proxy.mapToSource(index).row())
        if video is not None:
            menu.addSeparator()
            menu.addAction(
                "Open the video's results folder",
                lambda: self.workspace.prompts.open_folder(video.folder),
            )
        menu.exec(self.table.viewport().mapToGlobal(position))

    # ------------------------------------------------------------------
    # Plots and media
    # ------------------------------------------------------------------
    def open_plot(self, path: Path) -> PlotDialog:
        """Show a plot at full size in its own (non-modal) window."""
        dialog = PlotDialog(Path(path), self)
        dialog.setWindowFlag(Qt.WindowType.Window, True)
        self._plot_dialogs = [d for d in self._plot_dialogs if d.isVisible()]
        self._plot_dialogs.append(dialog)
        dialog.show()
        return dialog

    def _release_media(self) -> bool:
        self.detail.video_pane.close()
        return True

    def shutdown(self) -> None:
        """Release the video decoder and close plot windows."""
        self._release_media()
        for dialog in self._plot_dialogs:
            dialog.close()
        self._plot_dialogs.clear()


__all__ = ["ResultsTab", "SummaryModel", "SummaryProxy", "VideoDetail",
           "TimeBinsPanel", "ComparePanel", "overlay_for_video"]
