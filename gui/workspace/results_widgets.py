"""
gui/workspace/results_widgets.py
The widgets of the Results stage: the batch-summary table, the per-video
detail (key numbers, warnings, plots, the video), the time-bins chart and the
run comparison. They carry no project logic; :mod:`gui.workspace.stages.results_stage`
ties them to the project.

The per-video detail is built from the design system: key numbers as
:class:`~gui.workspace.ui_kit.StatTile`, quality warnings as
:class:`~gui.workspace.ui_kit.Callout`, plot thumbnails in a grid and the
video player, each in a :class:`~gui.workspace.ui_kit.Card`. The chart and
the comparison read their colours from the current theme.

The data logic is in :mod:`gui.workspace.results_data` and the overlays in
:mod:`gui.workspace.overlay`; this module is the Qt layer.
"""
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import (QAbstractTableModel, QModelIndex, QSize,
                          QSortFilterProxyModel, Qt, pyqtSignal)
from PyQt6.QtGui import QBrush, QColor, QIcon, QKeySequence, QPixmap
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                             QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                             QListWidget, QListWidgetItem, QScrollArea,
                             QTableView, QTableWidget, QTableWidgetItem,
                             QTabWidget, QVBoxLayout, QWidget)

from core.positional_validation import load_tracking_csv
from core.tst_validation import load_tst_evidence_csv
from gui.video_list import STATUS_WARNING, status_icon
from gui.video_player import VideoPlayerWidget
from gui.workspace import overlay as overlays
from gui.workspace.results_data import (RunData, RunDiff, RunLoadError,
                                        RunVideo, Series, diff_summaries,
                                        headline_metrics, key_metrics,
                                        time_bin_group_columns,
                                        time_bin_metrics, time_bin_series,
                                        time_bin_subjects, to_float)
from gui.workspace.theme import Spacing, current_palette, style_table_view
from gui.workspace.ui_kit import (Callout, Card, CollapsibleSection, Label,
                                  SecondaryButton, StatTile)

logger = logging.getLogger(__name__)

ALL_SUBJECTS = "(all)"
NO_GROUP = "(each video)"
THUMB_SIZE = QSize(180, 135)
TILE_COLUMNS = 3

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
        self.source_combo.setToolTip(
            "The validation video (when the run made one) or the original "
            "video with the tracking drawn on it"
        )
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        self.message_label = Label("", "muted", wrap=True)
        self.player = VideoPlayerWidget(self)
        bar = QHBoxLayout()
        bar.setSpacing(Spacing.XS)
        bar.addWidget(Label("Show", "caption"))
        bar.addWidget(self.source_combo, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.XS)
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
    """Key numbers, plots and video of the video selected in the summary."""

    plot_requested = pyqtSignal(object)      # Path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._run: Optional[RunData] = None
        self.video: Optional[RunVideo] = None
        self.title_label = Label("Select a video in the table.", "subtitle")
        self.warnings_label = Callout("warning", "", "Quality warnings")
        self.warnings_label.setVisible(False)
        self.tiles: List[StatTile] = []

        # -- overview: key numbers, then every measurement -------------------
        self.tiles_card = Card("Key numbers")
        self.tiles_grid = QGridLayout()
        self.tiles_grid.setSpacing(Spacing.XS)
        self.tiles_card.body.addLayout(self.tiles_grid)
        self.metrics_table = QTableWidget(0, 3)
        self.metrics_table.setHorizontalHeaderLabels(["Section", "Metric", "Value"])
        self.metrics_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        style_table_view(self.metrics_table)
        self.metrics_table.setMinimumHeight(220)
        self.metrics_section = CollapsibleSection(
            "All measurements", expanded=False, animated=False
        )
        self.metrics_section.body.addWidget(self.metrics_table)
        self.notes_label = Label("", "muted", wrap=True)
        overview = QWidget()
        overview_layout = QVBoxLayout(overview)
        overview_layout.setContentsMargins(0, Spacing.SM, 0, 0)
        overview_layout.setSpacing(Spacing.SM)
        overview_layout.addWidget(self.tiles_card)
        overview_layout.addWidget(self.metrics_section)
        overview_layout.addWidget(self.notes_label)
        overview_layout.addStretch(1)
        overview_scroll = QScrollArea()
        overview_scroll.setWidgetResizable(True)
        overview_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        overview_scroll.setWidget(overview)

        # -- plots: a grid of thumbnails ---------------------------------------
        self.plot_list = QListWidget()
        self.plot_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.plot_list.setIconSize(THUMB_SIZE)
        self.plot_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.plot_list.setMovement(QListWidget.Movement.Static)
        self.plot_list.setWordWrap(True)
        self.plot_list.setSpacing(Spacing.XS)
        self.plot_list.setToolTip("Click a plot to enlarge it")
        self.plot_list.setProperty("flat", True)
        self.plot_list.itemClicked.connect(self._on_plot_clicked)
        self.no_plots_label = Label("This video has no plots.", "muted", wrap=True)
        self.plots_card = Card("Plots", "Click a plot to enlarge it")
        self.plots_card.add_widget(self.no_plots_label)
        self.plots_card.add_widget(self.plot_list, 1)
        plots = QWidget()
        plots_layout = QVBoxLayout(plots)
        plots_layout.setContentsMargins(0, Spacing.SM, 0, 0)
        plots_layout.addWidget(self.plots_card, 1)

        # -- the video -------------------------------------------------------
        self.video_pane = VideoPane()
        self.video_card = Card(padding=Spacing.SM)
        self.video_card.add_widget(self.video_pane, 1)
        video = QWidget()
        video_layout = QVBoxLayout(video)
        video_layout.setContentsMargins(0, Spacing.SM, 0, 0)
        video_layout.addWidget(self.video_card, 1)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(overview_scroll, "Overview")
        self.tabs.addTab(plots, "Plots")
        self.tabs.addTab(video, "Video")
        self.video_tab = video
        self.tabs.currentChanged.connect(self._on_tab_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.XS)
        layout.addWidget(self.title_label)
        layout.addWidget(self.warnings_label)
        layout.addWidget(self.tabs, 1)

    # -- content ---------------------------------------------------------
    def set_video(self, run: Optional[RunData], video: Optional[RunVideo]) -> None:
        self._run, self.video = run, video
        self.metrics_table.setRowCount(0)
        self.plot_list.clear()
        self._clear_tiles()
        self.notes_label.setText("")
        self.tiles_card.setVisible(False)
        if video is None:
            self.title_label.setText("Select a video in the table.")
            self.warnings_label.setVisible(False)
            self.no_plots_label.setVisible(False)
            self.plot_list.setVisible(False)
            self.video_pane.set_video(run, None)
            return
        status = "" if video.status == "completed" else f" ({video.status})"
        self.title_label.setText(f"{video.name}{status}")
        problems = list(video.warnings)
        if video.failure:
            problems.insert(0, f"Failed: {video.failure}")
        self.warnings_label.set_kind("error" if video.failure else "warning")
        self.warnings_label.set_title(
            "This video failed" if video.failure else "Quality warnings"
        )
        self.warnings_label.set_text(
            "<br>".join(f"• {_escape(p)}" for p in problems)
        )
        self.warnings_label.setVisible(bool(problems))
        self._fill_metrics(video)
        self._fill_plots(video)
        self.video_pane.set_video(run, video)

    def _clear_tiles(self) -> None:
        for tile in self.tiles:
            self.tiles_grid.removeWidget(tile)
            tile.setParent(None)
            tile.deleteLater()
        self.tiles = []

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
        headline = headline_metrics(rows)
        for position, (name, value) in enumerate(headline):
            tile = StatTile(name.capitalize(), value)
            self.tiles.append(tile)
            self.tiles_grid.addWidget(
                tile, position // TILE_COLUMNS, position % TILE_COLUMNS
            )
        self.tiles_card.setVisible(bool(headline))
        self.metrics_section.set_expanded(not headline)
        if not summary:
            self.notes_label.setText("This video has no summary.json.")
        elif notes:
            self.notes_label.setText(
                "Notes: " + " ".join(dict.fromkeys(notes[:4]))
            )

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
        has_plots = self.plot_list.count() > 0
        self.plot_list.setVisible(has_plots)
        self.no_plots_label.setVisible(not has_plots)

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

    def tile_texts(self) -> List[Tuple[str, str]]:
        """``(caption, value)`` of the key-number tiles."""
        return [(t.caption(), t.value()) for t in self.tiles]

    def _on_plot_clicked(self, item: QListWidgetItem) -> None:
        self.plot_requested.emit(Path(item.data(Qt.ItemDataRole.UserRole)))

    def show_video_tab(self) -> None:
        self.tabs.setCurrentWidget(self.video_tab)

    def _on_tab_changed(self, _index: int) -> None:
        if self.tabs.currentWidget() is self.video_tab:
            self.video_pane.activate()
        else:
            self.video_pane.deactivate()

    def hideEvent(self, event) -> None:
        self.video_pane.deactivate()
        super().hideEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.tabs.currentWidget() is self.video_tab:
            self.video_pane.activate()


def _escape(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


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
        self.metric_combo.setToolTip("The measurement to plot over the time bins")
        self.subject_combo.setToolTip("Show one mouse of the videos, or all of them")
        self.group_combo.setToolTip(
            "Average the videos per value of a sample-sheet column"
        )
        self.subject_label = Label("Mouse", "caption")
        self.message_label = Label("", "muted", wrap=True)
        self.figure = Figure(figsize=(6, 3.5), layout="constrained")
        self.axes = self.figure.subplots()
        self.canvas = FigureCanvasQTAgg(self.figure)

        bar = QHBoxLayout()
        bar.setSpacing(Spacing.XS)
        bar.addWidget(Label("Metric", "caption"))
        bar.addWidget(self.metric_combo, 1)
        bar.addWidget(self.subject_label)
        bar.addWidget(self.subject_combo)
        bar.addWidget(Label("Group by", "caption"))
        bar.addWidget(self.group_combo)
        self.card = Card(padding=Spacing.SM)
        self.card.body.addLayout(bar)
        self.card.add_widget(self.message_label)
        self.card.add_widget(self.canvas, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, Spacing.SM, 0, 0)
        layout.addWidget(self.card, 1)
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

    def _apply_theme(self) -> None:
        """Paint the chart in the colours of the current theme."""
        palette = current_palette()
        figure, axes = self.figure, self.axes
        figure.set_facecolor(palette.surface)
        axes.set_facecolor(palette.surface)
        axes.tick_params(colors=palette.text_muted)
        axes.xaxis.label.set_color(palette.text)
        axes.yaxis.label.set_color(palette.text)
        axes.title.set_color(palette.text)
        for spine in axes.spines.values():
            spine.set_color(palette.border_strong)

    def render(self) -> None:
        table = self._run.time_bins if self._run is not None else None
        axes = self.axes
        axes.clear()
        self._apply_theme()
        self.series = []
        metric = self.metric_combo.currentText()
        if table is None or not metric:
            self.message_label.setText(
                "This run has no time bins. Set Time bins in Review & run "
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
            legend = axes.legend(fontsize="small", facecolor=current_palette().surface,
                                 edgecolor=current_palette().border)
            for text in legend.get_texts():
                text.set_color(current_palette().text)
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
        self.compare_button = SecondaryButton("Compare")
        self.compare_button.clicked.connect(lambda _checked=False: self.compare())
        self.only_changed_check = QCheckBox("Only changed videos")
        self.only_changed_check.toggled.connect(self._fill)
        self.summary_label = Label(
            "Choose two runs (for example before and after re-analysing).",
            "muted", wrap=True,
        )
        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        style_table_view(self.table)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setStretchLastSection(False)
        bar = QHBoxLayout()
        bar.setSpacing(Spacing.XS)
        bar.addWidget(Label("Before (A)", "caption"))
        bar.addWidget(self.combo_a, 1)
        bar.addWidget(Label("After (B)", "caption"))
        bar.addWidget(self.combo_b, 1)
        bar.addWidget(self.compare_button)
        self.card = Card(padding=Spacing.SM)
        self.card.body.addLayout(bar)
        self.card.add_widget(self.only_changed_check)
        self.card.add_widget(self.summary_label)
        self.card.add_widget(self.table, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, Spacing.SM, 0, 0)
        layout.addWidget(self.card, 1)

    def set_runs(self, entries: list) -> None:
        """``entries``: objects with ``label`` and ``path`` (the run picker's)."""
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
        palette = current_palette()
        changed_brush = QBrush(QColor(palette.tint("warning", 0.30)))
        only_brush = QBrush(QColor(palette.surface_alt))
        text_brush = QBrush(QColor(palette.text))
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
                    item.setBackground(changed_brush)
                    item.setForeground(text_brush)
                    item.setToolTip(f"A: {cell.a}\nB: {cell.b}")
                elif row.status != "both":
                    item.setBackground(only_brush)
                    item.setForeground(text_brush)
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


__all__ = [
    "ALL_SUBJECTS", "ComparePanel", "CopyableTableView", "NO_GROUP", "PlotDialog",
    "SummaryModel", "SummaryProxy", "TimeBinsPanel", "VideoDetail", "VideoPane",
    "overlay_for_video",
]
