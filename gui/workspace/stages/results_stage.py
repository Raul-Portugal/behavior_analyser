"""
gui/workspace/stages/results_stage.py
Stage 6: open a finished run and look at what it produced.

::

    run: [analysis_run_...  22 Oct 2026, 09:33  -  12 done v]   [Compare runs]
    one-line summary of the run (chips)
    ( Videos | Time bins | Compare runs )
    +- Batch summary --------------------------+- the selected video -------+
    | batch_summary.csv (sort, filter, copy,   | key numbers (stat tiles)   |
    |   export)                                | warnings, all metrics      |
    |                                          | Overview | Plots | Video   |
    +------------------------------------------+----------------------------+

* The run picker lists the project's runs (newest first, with date and
  counts) and any folder opened with "Open run folder..."; manifest schema 3
  is read in full, schema 2 and older as far as they go.
* **Videos**: the batch summary as a sortable, filterable table. Selecting a
  row shows that video's key numbers, quality warnings, plots (click to
  enlarge) and its video: the validation video when one was made, else the
  original video with the tracking drawn on it.
* **Time bins**: a metric over the bins, per video or averaged per group of a
  metadata column.
* **Compare runs**: two batch summaries side by side, changed values marked.
* Nothing to show yet: "No runs yet" with a way to Review & run.

The data logic is in :mod:`gui.workspace.results_data`, the overlays in
:mod:`gui.workspace.overlay` and the table, detail and chart widgets in
:mod:`gui.workspace.results_widgets`; this module is the stage that ties them
to the project.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import QModelIndex, Qt, pyqtSignal
from PyQt6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView,
                             QLineEdit, QMenu, QSplitter, QStackedWidget,
                             QTabWidget, QVBoxLayout, QWidget)

from core.project import Project
from gui.workspace.results_data import (RunData, RunLoadError, RunVideo,
                                        looks_like_run, norm, write_table)
from gui.workspace.results_widgets import (ComparePanel, CopyableTableView,
                                           PlotDialog, SummaryModel,
                                           SummaryProxy, TimeBinsPanel,
                                           VideoDetail)
from gui.workspace.stages import status as stage_status
from gui.workspace.stages.base import Stage
from gui.workspace.theme import Spacing, style_table_view
from gui.workspace.ui_kit import (Card, EmptyState, GhostButton, Label,
                                  SecondaryButton, StatusChip)

logger = logging.getLogger(__name__)

PAGE_EMPTY = 0
PAGE_MAIN = 1
SECTION_VIDEOS = 0

RESULTS_SUBTITLE = "Open and explore what your runs produced."


@dataclass
class RunEntry:
    path: Path
    label: str
    external: bool = False
    exists: bool = True


def format_started(text: str) -> str:
    """``22 Oct 2026, 09:33`` from a manifest time stamp ('' when unknown)."""
    if not text:
        return ""
    for parse in (datetime.fromisoformat,
                  lambda t: datetime.strptime(t, "%Y%m%d_%H%M%S")):
        try:
            return parse(text).strftime("%d %b %Y, %H:%M")
        except ValueError:
            continue
    return text


def run_label(path: Path, completed: Optional[int] = None, failed: int = 0,
              cancelled: bool = False, started: str = "") -> str:
    """The picker's text: date, folder and counts (``22 Oct 2026, 09:33 - ...``)."""
    parts: List[str] = []
    if completed is not None:
        parts.append(f"{completed} done")
    if failed:
        parts.append(f"{failed} failed")
    if cancelled:
        parts.append("cancelled")
    when = format_started(started)
    head = f"{when}  ·  {path.name}" if when else path.name
    suffix = f"  ·  {', '.join(parts)}" if parts else ""
    return f"{head}{suffix}"


class ResultsStage(Stage):
    """Stage 6: the runs of the project and their results."""

    stage_id = stage_status.STAGE_RESULTS
    title = "Results"
    subtitle = RESULTS_SUBTITLE
    icon_name = "chart"

    run_loaded = pyqtSignal(object)         # RunData

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.run: Optional[RunData] = None
        self._entries: List[RunEntry] = []
        self._external: List[Path] = []
        self._loading = False
        self._plot_dialogs: List[PlotDialog] = []
        self._build()
        self._refresh_runs()
        self._update_page()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build(self) -> None:
        self.pages = QStackedWidget()
        self.empty = EmptyState("chart", "", "")
        self.empty_run_button = self.empty.add_action("Go to Review && run")
        self.empty_run_button.clicked.connect(
            lambda _checked=False: self._go(stage_status.STAGE_REVIEW)
        )
        self.empty_open_button = self.empty.add_action("Open a run folder…")
        self.empty_open_button.clicked.connect(
            lambda _checked=False: self.open_run_folder()
        )
        self.pages.addWidget(self.empty)
        self.pages.addWidget(self._build_main())
        self.set_content(self.pages)

        # Stage-level actions.
        self.open_button = SecondaryButton("Open run folder…", "folder")
        self.open_button.setToolTip(
            "Open the results of another run: any folder made by an analysis"
        )
        self.open_button.clicked.connect(lambda _checked=False: self.open_run_folder())
        self.folder_button = GhostButton("Show in file manager", "folder")
        self.folder_button.setToolTip("Open this run's folder in the file manager")
        self.folder_button.clicked.connect(lambda _checked=False: self.show_run_folder())
        self.header.add_action(self.folder_button)
        self.header.add_action(self.open_button)

    def _build_main(self) -> QWidget:
        self.run_combo = self._make_combo()
        self.compare_button = SecondaryButton("Compare runs", "align")
        self.compare_button.clicked.connect(lambda _checked=False: self.show_compare())
        self.info_label = Label("Run a batch or open a run folder.", "muted", wrap=True)
        self.info_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.chips = QHBoxLayout()
        self.chips.setSpacing(Spacing.XS)
        self.paradigm_chip = StatusChip("info", "")
        self.completed_chip = StatusChip("ok", "")
        self.failed_chip = StatusChip("error", "")
        self.warnings_chip = StatusChip("warning", "")
        self.cancelled_chip = StatusChip("neutral", "Cancelled")
        self.bins_chip = StatusChip("neutral", "")
        for chip in (self.paradigm_chip, self.completed_chip, self.failed_chip,
                     self.warnings_chip, self.cancelled_chip, self.bins_chip):
            self.chips.addWidget(chip)
            chip.setVisible(False)
        self.chips.addStretch(1)

        # -- videos --------------------------------------------------------
        self.model = SummaryModel(self)
        self.proxy = SummaryProxy(self)
        self.proxy.setSourceModel(self.model)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter the table…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self.proxy.set_text_filter)
        self.export_button = SecondaryButton("Export CSV…")
        self.export_button.clicked.connect(lambda _checked=False: self.export_csv())
        self.copy_button = SecondaryButton("Copy")
        self.copy_button.setToolTip("Copy the selected cells (Ctrl+C)")
        self.copy_button.clicked.connect(lambda _checked=False: self.copy_selection())
        self.rows_label = Label("", "muted")
        filter_bar = QHBoxLayout()
        filter_bar.setSpacing(Spacing.XS)
        filter_bar.addWidget(self.filter_edit, 1)
        filter_bar.addWidget(self.rows_label)
        filter_bar.addWidget(self.copy_button)
        filter_bar.addWidget(self.export_button)

        self.table = CopyableTableView()
        self.table.setModel(self.proxy)
        style_table_view(self.table)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(-1, Qt.SortOrder.AscendingOrder)   # CSV order
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
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

        self.table_card = Card("Batch summary", padding=Spacing.SM)
        self.table_card.body.addLayout(filter_bar)
        self.table_card.add_widget(self.table, 1)

        self.detail = VideoDetail()
        self.detail.plot_requested.connect(self.open_plot)
        self.videos_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.videos_splitter.setChildrenCollapsible(False)
        self.videos_splitter.addWidget(self.table_card)
        self.videos_splitter.addWidget(self.detail)
        self.table_card.setMinimumWidth(420)
        self.detail.setMinimumWidth(440)
        self.videos_splitter.setStretchFactor(0, 3)
        self.videos_splitter.setStretchFactor(1, 2)
        self.videos_splitter.setSizes([700, 560])

        self.bins_panel = TimeBinsPanel()
        self.compare_panel = ComparePanel(self.run_data)

        self.sections = QTabWidget()
        self.sections.setDocumentMode(True)
        self.sections.addTab(self.videos_splitter, "Videos")
        self.sections.addTab(self.bins_panel, "Time bins")
        self.sections.addTab(self.compare_panel, "Compare runs")

        top = QHBoxLayout()
        top.setSpacing(Spacing.SM)
        top.addWidget(Label("Run", "caption"))
        top.addWidget(self.run_combo, 1)
        top.addWidget(self.compare_button)
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(Spacing.SM)
        column.addLayout(top)
        column.addLayout(self.chips)
        column.addWidget(self.info_label)
        column.addWidget(self.sections, 1)
        return page

    def _make_combo(self):
        from PyQt6.QtWidgets import QComboBox
        combo = QComboBox()
        combo.setMinimumContentsLength(30)
        combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        combo.setToolTip("Choose which run to look at (newest first)")
        combo.currentIndexChanged.connect(self._on_run_picked)
        return combo

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def attach(self, window) -> None:
        super().attach(window)
        # Release the decoder thread before the window goes away.
        window.register_close_guard(self._release_media)
        self.on_project_changed(window.project)

    def on_project_changed(self, project: Optional[Project]) -> None:
        self._external.clear()
        self._clear_run()
        self._refresh_runs()
        self._update_page()

    def on_project_modified(self) -> None:
        signature = [str(e.path) for e in self._entries if not e.external]
        project = self.project
        wanted = [str(r.run_dir) for r in self._sorted_runs()] if project else []
        if signature != wanted:
            self._refresh_runs()
            self._update_page()

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

    def on_enter(self) -> None:
        if self.run is None and self._entries:
            self._load_newest()
        self._update_page()
        self.refresh_footer()

    def on_leave(self) -> bool:
        self.detail.video_pane.deactivate()
        return True

    def _go(self, stage_id: str) -> None:
        if self.workspace is not None:
            self.workspace.go_to_stage(stage_id)

    def _update_page(self) -> None:
        if self.run is not None or self._entries:
            self.pages.setCurrentIndex(PAGE_MAIN)
            return
        if self.project is None:
            self.empty.set_texts(
                "No project open",
                "Open or create a project to see its results, or open the "
                "folder of an earlier run.",
            )
            self.empty_run_button.setVisible(False)
        else:
            self.empty.set_texts(
                "No runs yet",
                "Nothing has been analysed in this project. Go to Review & "
                "run to start the analysis, or open the folder of an earlier run.",
            )
            self.empty_run_button.setVisible(True)
        self.pages.setCurrentIndex(PAGE_EMPTY)
        self.folder_button.setEnabled(False)
        self.folder_button.setToolTip("There is no run to show yet")

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
                record.run_dir, record.completed, record.failed,
                record.cancelled, record.started_at,
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
        self.compare_button.setToolTip(
            "" if len(entries) >= 2 else "Compare needs two runs: this project has fewer"
        )

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
        """Load a run folder; False (with a message) if it cannot be read."""
        path = Path(path)
        try:
            run = self.run_data(path)
        except (RunLoadError, OSError, ValueError) as exc:
            self._clear_run()
            self.info_label.setText(f"Cannot open {path.name}: {exc}")
            self.info_label.setVisible(True)
            if not quiet and self.workspace is not None:
                self.workspace.prompts.error("Open run", f"Cannot open {path}:\n{exc}")
            return False
        self._set_run(run)
        return True

    def _clear_run(self) -> None:
        self._set_run(None)
        self.info_label.setText("Run a batch or open a run folder.")
        self.info_label.setVisible(True)

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
        self._show_run_info(run)
        self._update_rows_label()
        has_rows = run is not None and bool(run.summary.rows)
        self.export_button.setEnabled(has_rows)
        self.copy_button.setEnabled(has_rows)
        self.folder_button.setEnabled(run is not None)
        self.folder_button.setToolTip(
            "Open this run's folder in the file manager" if run is not None
            else "There is no run to show yet"
        )
        if run is not None:
            self.run_loaded.emit(run)
            self.pages.setCurrentIndex(PAGE_MAIN)
        self.refresh_footer()

    def _show_run_info(self, run: Optional[RunData]) -> None:
        if run is None:
            for chip in (self.paradigm_chip, self.completed_chip, self.failed_chip,
                         self.warnings_chip, self.cancelled_chip, self.bins_chip):
                chip.setVisible(False)
            return
        completed = sum(1 for v in run.videos if v.status == "completed")
        failed = sum(1 for v in run.videos if v.status == "failed") or len(run.failures)
        warnings = run.quality_warning_count()
        self.paradigm_chip.set_text(run.paradigm or "paradigm unknown")
        self.completed_chip.set_text(f"{completed} completed")
        self.failed_chip.set_text(f"{failed} failed")
        self.warnings_chip.set_text(f"{warnings} quality warning{'' if warnings == 1 else 's'}")
        self.bins_chip.set_text(f"time bins of {run.time_bin_seconds:g} s")
        self.paradigm_chip.setVisible(True)
        self.completed_chip.setVisible(True)
        self.failed_chip.setVisible(failed > 0)
        self.warnings_chip.setVisible(warnings > 0)
        self.cancelled_chip.setVisible(bool(run.cancelled))
        self.bins_chip.setVisible(bool(run.time_bin_seconds))
        text = run.describe()
        if run.notes:
            text += "\n" + " ".join(run.notes)
        if run.failures:
            text += "\nFailed: " + "; ".join(
                f"{Path(src).name}: {msg}" for src, msg in run.failures[:3]
            )
        self.info_label.setText(text)
        self.info_label.setToolTip(str(run.run_dir))
        # The chips say the counts; the line is for notes and failures.
        self.info_label.setVisible(bool(run.notes or run.failures))

    def show_run(self, run_dir: Path) -> bool:
        """Show a run (the Review & run stage calls this when a run finishes)."""
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
            f"{shown} of {total} rows" if shown != total else
            f"{total} row{'' if total == 1 else 's'}"
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

    # ------------------------------------------------------------------
    # What the window shows for this stage
    # ------------------------------------------------------------------
    def next_label(self) -> Optional[str]:
        return None

    def footer_text(self):
        if self.run is not None:
            run = self.run
            return f"{run.name}: {run.describe()}", ("ok", "Run loaded")
        return super().footer_text()


__all__ = ["ResultsStage", "RunEntry", "run_label", "format_started"]
