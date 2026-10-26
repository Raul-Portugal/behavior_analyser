"""
gui/workspace/stages/review_stage.py
Stage 5: check that everything is ready, then analyse the videos.

::

    +- Is everything ready? -----------------+  +- Run -----------------------+
    | Videos            [Done]               |  | 12 videos will be analysed  |
    | Arena & zones     [Done]               |  | [      Run analysis      ]  |
    | Detection         [Not reviewed] Fix ->|  +- Run options ---------------+
    | Trials            [Needs attention] Fix|  | Output folder  [Change...]  |
    +- Problems to fix -----------------------+  | Parallel videos  Auto       |
    | Start time needed: 3 videos      Fix -> |  | Time bins        Off        |
    +-----------------------------------------+  | Videos to run    All        |
                                                 | Sample sheet     Load Clear |
                                                 | [ ] Validation videos       |

* **Readiness** (:mod:`gui.workspace.run_readiness`): one row per stage with
  a status chip, a one-line reason and a "Fix ->" link that opens the stage;
  per-video problems are grouped by kind with the stage that fixes them. The
  big "Run analysis" button is disabled until nothing blocks the run, and its
  tooltip says what does.
* **Run options**: output folder, parallel videos (Auto or N), time bins
  (Off or N minutes), the videos to run (all, not yet done, selected), the
  sample sheet and the validation videos. The first two numbers and the
  sample sheet live in the project; the rest apply to the run in hand.
* **While running** the same stage shows the progress: overall bar and ETA,
  a per-video table (status, %, last message), a collapsible log and Cancel
  (after a confirmation). The window is busy (``set_busy``) and refuses to
  close without asking.
* **When it is over** a summary card (completed, failed, warnings) offers
  "View results", "Open folder" and, when videos were interrupted, "Resume
  interrupted run".

The run itself is :class:`gui.workspace.run_controller.RunController`: the
same worker the classic window and the CLI use, so the outputs are identical.
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QColor, QKeySequence, QPainter, QShortcut
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout,
                             QHBoxLayout,
                             QHeaderView, QLineEdit, QPlainTextEdit,
                             QProgressBar, QScrollArea, QSizePolicy,
                             QSpinBox, QStackedWidget, QTableView,
                             QVBoxLayout, QWidget)

from core.project import Project
from gui import ux_helpers
from gui.workspace.run_check import (SCOPE_ALL, SCOPE_LABELS, SCOPE_SELECTED,
                                     SCOPES, ResumeOffer)
from gui.workspace.run_controller import FORCE_QUIT_DELAY_MS, RunController
from gui.workspace.run_progress import (ProgressBarDelegate,
                                        RunProgressModel)
from gui.workspace.run_readiness import (CHIP_ERROR, CHIP_WARNING, Readiness,
                                         build_readiness)
from gui.workspace.stages import status as stage_status
from gui.workspace.stages.base import Stage
from gui.workspace.theme import (Spacing, current_palette, style_table_view)
from gui.workspace.ui_kit import (Callout, Card, CollapsibleSection,
                                  DangerButton, EmptyState, GhostButton,
                                  KeyHint, Label, PrimaryButton,
                                  SecondaryButton, StatusChip)

logger = logging.getLogger(__name__)

PAGE_EMPTY = 0
PAGE_SETUP = 1
PAGE_RUN = 2

REVIEW_SUBTITLE = "Check that everything is ready, then analyse your videos."
RUN_SHORTCUT = "Ctrl+Return"
SIDE_WIDTH = 330
MAX_LISTED_ISSUES = 8
TABLE_MIN_HEIGHT = 170
TITLE_WIDTH = 130        # the stage names of the checklist line up
CHECK_ROW_HEIGHT = 44
FIX_WIDTH = 72           # "Fix ->" buttons are as wide in every row


def plural(count: int, noun: str) -> str:
    return stage_status.plural(count, noun)


def _escape(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _scroll_page(content: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setWidget(content)
    return area


# ----------------------------------------------------------------------
# Small widgets
# ----------------------------------------------------------------------
class _Row(QWidget):
    """A row with a hairline under it (the readiness checklist)."""

    def __init__(self, line: bool = True, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._line = line

    def paintEvent(self, event) -> None:
        if not self._line:
            return
        painter = QPainter(self)
        painter.fillRect(0, self.height() - 1, self.width(), 1,
                         QColor(current_palette().border))

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange):
            self.update()


class CheckRow(_Row):
    """One stage of the readiness checklist: chip, title, reason, "Fix ->"."""

    def __init__(self, stage_id: str, parent: Optional[QWidget] = None):
        super().__init__(True, parent)
        self.stage_id = stage_id
        self.title_label = Label("", "subtitle")
        self.title_label.setMinimumWidth(TITLE_WIDTH)
        self.title_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.reason_label = Label("", "muted", wrap=True)
        self.chip = StatusChip("neutral", "")
        self.fix_button = SecondaryButton("Fix →")
        self.fix_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.fix_button.setFixedWidth(FIX_WIDTH)
        policy = self.fix_button.sizePolicy()
        policy.setRetainSizeWhenHidden(True)      # the chips stay in one column
        self.fix_button.setSizePolicy(policy)
        self.fix_target = ""
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(Spacing.SM)
        row.addWidget(self.title_label, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.reason_label, 1, Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.chip, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.fix_button, 0, Qt.AlignmentFlag.AlignVCenter)
        self.setFixedHeight(CHECK_ROW_HEIGHT)

    def show_check(self, check) -> None:
        self.title_label.setText(check.title)
        self.reason_label.setText(check.reason)
        self.chip.set_status(check.chip, check.label)
        self.fix_button.setVisible(bool(check.fix_stage))
        self.fix_button.setToolTip(
            f"Go to {check.title}" if check.stage_id == check.fix_stage
            else "Go to the first step that is missing"
        )
        self.fix_target = check.fix_stage


class ProblemRow(_Row):
    """A group of per-video problems: count chip, kind, the videos, "Fix ->"."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(True, parent)
        self.chip = StatusChip("warning", "")
        self.title_label = Label("", "subtitle", wrap=True)
        self.names_label = Label("", "muted", wrap=True)
        self.fix_button = SecondaryButton("Fix →")
        self.fix_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.fix_target = ""
        texts = QVBoxLayout()
        texts.setSpacing(2)
        texts.addWidget(self.title_label)
        texts.addWidget(self.names_label)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, Spacing.SM, 0, Spacing.SM)
        row.setSpacing(Spacing.SM)
        row.addLayout(texts, 1)
        row.addWidget(self.chip, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.fix_button, 0, Qt.AlignmentFlag.AlignVCenter)

    def show_group(self, group) -> None:
        self.title_label.setText(group.title)
        self.names_label.setText(group.listed_names() or "; ".join(group.messages))
        self.names_label.setVisible(bool(self.names_label.text()))
        self.chip.set_status(
            CHIP_ERROR if group.blocking else CHIP_WARNING,
            plural(len(group.keys), "video") if group.keys
            else ("Blocks the run" if group.blocking else "Note"),
        )
        self.fix_target = group.fix_stage
        self.fix_button.setVisible(bool(group.fix_stage))
        self.fix_button.setToolTip(
            "Open the stage where this is fixed" if group.fix_stage else ""
        )


# ----------------------------------------------------------------------
# The stage
# ----------------------------------------------------------------------
class ReviewStage(Stage):
    """Stage 5: the readiness checklist, the run options and the run itself."""

    stage_id = stage_status.STAGE_REVIEW
    title = "Review & run"
    subtitle = REVIEW_SUBTITLE
    icon_name = "list"
    FORCE_QUIT_DELAY_MS = FORCE_QUIT_DELAY_MS

    #: Emitted by the controller; re-exported so callers need not reach in.
    @property
    def run_started(self):
        return self.controller.run_started

    @property
    def run_finished(self):
        return self.controller.run_finished

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.controller = RunController(self)
        self._loading = False
        self._readiness: Optional[Readiness] = None
        self._start_issues: List[str] = []
        self._shown_row = -1
        self._check_rows = {}
        self._problem_rows: List[ProblemRow] = []
        self._build()
        controller = self.controller
        controller.state_changed.connect(self._on_state_changed)
        controller.progress_changed.connect(self._on_progress_changed)
        controller.log_added.connect(self._on_log_added)
        controller.log_cleared.connect(self.log_view.clear)
        controller.run_started.connect(self._on_run_started)
        self.refresh()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build(self) -> None:
        self.pages = QStackedWidget()
        self.empty = EmptyState("list", "", "")
        self.empty.add_action("Go to Videos")
        self.empty.buttons[0].clicked.connect(
            lambda _checked=False: self._go(stage_status.STAGE_VIDEOS)
        )
        self.pages.addWidget(self.empty)
        self.pages.addWidget(self._build_setup_page())
        self.pages.addWidget(self._build_run_page())
        self.set_content(self.pages)
        self.run_shortcut = QShortcut(QKeySequence(RUN_SHORTCUT), self)
        self.run_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.run_shortcut.activated.connect(self._on_shortcut)

    # -- the setup page --------------------------------------------------
    def _build_setup_page(self) -> QWidget:
        # Left: the checklist and the problems.
        self.checklist = Card("Is everything ready?", padding=Spacing.SM)
        self.checklist.body.setSpacing(0)
        for stage_id in (stage_status.STAGE_VIDEOS, stage_status.STAGE_ARENA,
                         stage_status.STAGE_DETECTION, stage_status.STAGE_TRIALS):
            row = CheckRow(stage_id)
            row.fix_button.clicked.connect(
                lambda _checked=False, r=row: self._go(r.fix_target)
            )
            self._check_rows[stage_id] = row
            self.checklist.add_widget(row)
        self.problems = Card("Videos that need attention", padding=Spacing.SM)
        self.problems_body = QVBoxLayout()
        self.problems_body.setContentsMargins(0, 0, 0, 0)
        self.problems_body.setSpacing(0)
        self.start_issues_callout = Callout(
            "error", "", "The last attempt to start the run found"
        )
        self.start_issues_callout.setVisible(False)
        self.problems.add_widget(self.start_issues_callout)
        self.problems.add_widget(self._wrap(self.problems_body))
        self.problems.setVisible(False)
        left = QVBoxLayout()
        left.setSpacing(Spacing.MD)
        # Right: the run card (always in view), left: the options under the checklist.
        self.run_card = Card(padding=Spacing.SM)
        self.run_card.body.setSpacing(Spacing.XS)
        self.ready_label = Label("", "subtitle", wrap=True)
        self.detail_label = Label("", "muted", wrap=True)
        self.run_button = PrimaryButton("Run analysis", "play")
        self.run_button.setMinimumHeight(42)
        self.run_button.clicked.connect(lambda _checked=False: self._on_run_clicked())
        hint = QHBoxLayout()
        hint.setSpacing(Spacing.XXS)
        hint.addStretch(1)
        hint.addWidget(KeyHint("Ctrl"))
        hint.addWidget(KeyHint("Enter"))
        hint.addWidget(Label("starts the run", "muted"))
        hint.addStretch(1)
        self.resume_callout = Callout("info", "", "Interrupted run")
        self.resume_button = SecondaryButton("Resume interrupted run", "reset")
        self.resume_button.clicked.connect(lambda _checked=False: self.start_resume())
        self.resume_box = QWidget()
        resume_layout = QVBoxLayout(self.resume_box)
        resume_layout.setContentsMargins(0, 0, 0, 0)
        resume_layout.setSpacing(Spacing.XS)
        resume_layout.addWidget(self.resume_callout)
        resume_layout.addWidget(self.resume_button)
        self.resume_box.setVisible(False)
        self.run_card.add_widget(self.ready_label)
        self.run_card.add_widget(self.detail_label)
        self.run_card.add_widget(self.run_button)
        self.run_card.body.addLayout(hint)
        self.run_card.add_widget(self.resume_box)
        self.resume_label = self.resume_callout    # the text the tests read

        self.options = self._build_options()
        left.addWidget(self.checklist)
        left.addWidget(self.problems)
        left.addWidget(self.options)
        left.addStretch(1)
        right = QVBoxLayout()
        right.setSpacing(Spacing.MD)
        right.addWidget(self.run_card)
        right.addStretch(1)

        body = QWidget()
        columns = QHBoxLayout(body)
        columns.setContentsMargins(0, 0, Spacing.XS, Spacing.MD)
        columns.setSpacing(Spacing.LG)
        columns.addLayout(left, 1)
        side = QWidget()
        side.setFixedWidth(SIDE_WIDTH)
        side.setLayout(right)
        right.setContentsMargins(0, 0, 0, 0)
        columns.addWidget(side, 0, Qt.AlignmentFlag.AlignTop)
        return _scroll_page(body)

    @staticmethod
    def _wrap(layout) -> QWidget:
        holder = QWidget()
        holder.setLayout(layout)
        return holder

    @staticmethod
    def _pair(caption: str, widget: QWidget) -> QVBoxLayout:
        """A caption right above its control."""
        column = QVBoxLayout()
        column.setSpacing(Spacing.XXS)
        column.addWidget(Label(caption, "caption"))
        column.addWidget(widget)
        return column

    def _build_options(self) -> Card:
        """The run options, two columns wide (they sit under the checklist)."""
        card = Card("Run options", padding=Spacing.SM)
        grid = QGridLayout()
        grid.setHorizontalSpacing(Spacing.LG)
        grid.setVerticalSpacing(Spacing.SM)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        # -- output folder (the whole width) ---------------------------------
        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setToolTip(
            "Each run creates an analysis_run_ folder inside this folder."
        )
        self.output_button = SecondaryButton("Change…")
        self.output_button.clicked.connect(lambda _checked=False: self.choose_output_dir())
        output_row = QHBoxLayout()
        output_row.setSpacing(Spacing.XS)
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(self.output_button)
        output = QHBoxLayout()
        output.setSpacing(Spacing.SM)
        caption = Label("Output folder", "caption")
        caption.setFixedWidth(88)
        output.addWidget(caption)
        output.addLayout(output_row, 1)
        grid.addLayout(output, 0, 0, 1, 2)

        # -- which videos | parallel videos and time bins -----------------------
        self.scope_combo = QComboBox()
        for scope in SCOPES:
            self.scope_combo.addItem(SCOPE_LABELS[scope], scope)
        self.scope_combo.setToolTip(
            "Which videos to analyse: every video, the ones that have no "
            "finished analysis yet, or the ones selected in the Videos stage."
        )
        self.scope_combo.currentIndexChanged.connect(self._on_scope_changed)
        self.scope_label = Label("", "muted", wrap=True)
        scope_column = self._pair("Videos to analyse", self.scope_combo)
        scope_column.addWidget(self.scope_label)
        grid.addLayout(scope_column, 1, 0)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 64)
        self.workers_spin.setSpecialValueText("Auto")
        self.workers_spin.setToolTip(
            "Videos analysed at once, each in its own process. Auto picks a "
            "number from the CPU cores and memory."
        )
        self.workers_spin.valueChanged.connect(self._on_workers_changed)
        self.bins_spin = QDoubleSpinBox()
        self.bins_spin.setRange(0.0, 240.0)
        self.bins_spin.setDecimals(2)
        self.bins_spin.setSingleStep(1.0)
        self.bins_spin.setSpecialValueText("Off")
        self.bins_spin.setSuffix(" min")
        self.bins_spin.setToolTip(
            "Also report the metrics in consecutive bins of this many "
            "minutes (batch_time_bins.csv). Off leaves the outputs as they "
            "are without this option."
        )
        self.bins_spin.valueChanged.connect(self._on_bins_changed)
        small = QHBoxLayout()
        small.setSpacing(Spacing.SM)
        small.addLayout(self._pair("Parallel videos", self.workers_spin), 1)
        small.addLayout(self._pair("Time bins", self.bins_spin), 1)
        grid.addLayout(small, 1, 1, Qt.AlignmentFlag.AlignTop)

        # -- sample sheet | validation videos -------------------------------------
        self.sheet_label = Label("", "muted", wrap=True)
        self.sheet_button = SecondaryButton("Load…")
        self.sheet_button.setToolTip(
            "Load animal metadata (group, sex, ...) from a CSV into the video "
            "table. The metadata columns are added to the batch summary."
        )
        self.sheet_button.clicked.connect(lambda _checked=False: self._load_sample_sheet())
        self.sheet_clear_button = GhostButton("Clear")
        self.sheet_clear_button.clicked.connect(lambda _checked=False: self._clear_sample_sheet())
        sheet_head = QHBoxLayout()
        sheet_head.setSpacing(Spacing.XS)
        sheet_head.addWidget(Label("Sample sheet", "caption"), 1)
        sheet_head.addWidget(self.sheet_button)
        sheet_head.addWidget(self.sheet_clear_button)
        sheet_column = QVBoxLayout()
        sheet_column.setSpacing(Spacing.XXS)
        sheet_column.addLayout(sheet_head)
        sheet_column.addWidget(self.sheet_label)
        grid.addLayout(sheet_column, 2, 0, Qt.AlignmentFlag.AlignTop)

        self.validation_check = QCheckBox("Validation videos")
        self.validation_check.setTristate(True)
        self.validation_check.setCheckState(Qt.CheckState.PartiallyChecked)
        self.validation_check.setToolTip(
            "Partly filled: each video keeps its own setting. Checked or "
            "unchecked: create (or skip) the validation video for every "
            "video of this run. The project's settings are not changed."
        )
        self.validation_check.stateChanged.connect(self._on_validation_changed)
        self.validation_label = Label("", "muted", wrap=True)
        validation = QVBoxLayout()
        validation.setSpacing(Spacing.XXS)
        validation.addWidget(Label("Validation video", "caption"))
        validation.addWidget(self.validation_check)
        validation.addWidget(self.validation_label)
        grid.addLayout(validation, 2, 1, Qt.AlignmentFlag.AlignTop)
        card.body.addLayout(grid)
        return card

    # -- the run page ----------------------------------------------------
    def _build_run_page(self) -> QWidget:
        self.progress_card = Card()
        self.run_title = Label("Analysing", "title")
        self.status_label = Label("Ready", "body")
        self.eta_label = Label("", "muted")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.cancel_button = DangerButton("Cancel run", "stop")
        self.cancel_button.clicked.connect(lambda _checked=False: self.request_cancel())
        top = QHBoxLayout()
        top.setSpacing(Spacing.MD)
        top.addWidget(self.run_title, 1)
        top.addWidget(self.cancel_button)
        self.progress_card.body.addLayout(top)
        status = QHBoxLayout()
        status.addWidget(self.status_label, 1)
        status.addWidget(self.eta_label)
        self.progress_card.body.addLayout(status)
        self.progress_card.add_widget(self.progress_bar)

        # The summary of a finished run.
        self.summary_card = Card("Analysis finished", padding=Spacing.SM)
        self.summary_callout = Callout("success", "")
        self.completed_chip = StatusChip("ok", "")
        self.failed_chip = StatusChip("error", "")
        self.warnings_chip = StatusChip("warning", "")
        self.cancelled_chip = StatusChip("neutral", "Cancelled")
        for chip in (self.completed_chip, self.failed_chip, self.warnings_chip,
                     self.cancelled_chip):
            self.summary_card.add_header_widget(chip)
        self.open_results_button = PrimaryButton("View results →")
        self.open_results_button.clicked.connect(lambda _checked=False: self.open_results())
        self.open_folder_button = SecondaryButton("Open folder", "folder")
        self.open_folder_button.clicked.connect(lambda _checked=False: self.open_run_folder())
        self.resume_run_button = SecondaryButton("Resume interrupted run", "reset")
        self.resume_run_button.clicked.connect(lambda _checked=False: self.start_resume())
        self.back_button = GhostButton("Back to run options", "arrow-left")
        self.back_button.clicked.connect(lambda _checked=False: self.dismiss_summary())
        buttons = QHBoxLayout()
        buttons.setSpacing(Spacing.SM)
        buttons.addWidget(self.open_results_button)
        buttons.addWidget(self.open_folder_button)
        buttons.addWidget(self.resume_run_button)
        buttons.addStretch(1)
        buttons.addWidget(self.back_button)
        self.summary_card.add_widget(self.summary_callout)
        self.summary_card.body.addLayout(buttons)
        self.summary_frame = self.summary_card       # the name the tests use
        self.summary_card.setVisible(False)

        self.progress_model = self.controller.progress_model
        self.progress_table = QTableView()
        self.progress_table.setModel(self.progress_model)
        style_table_view(self.progress_table)
        self.progress_table.setItemDelegateForColumn(
            RunProgressModel.PROGRESS, ProgressBarDelegate(self.progress_table)
        )
        self.progress_table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.progress_table.setSelectionMode(QTableView.SelectionMode.NoSelection)
        header = self.progress_table.horizontalHeader()
        header.setSectionResizeMode(
            RunProgressModel.VIDEO, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(
            RunProgressModel.STATUS, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(
            RunProgressModel.PROGRESS, QHeaderView.ResizeMode.Interactive)
        header.resizeSection(RunProgressModel.PROGRESS, 150)
        header.setSectionResizeMode(
            RunProgressModel.ETA, QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)

        self.log_section = CollapsibleSection("Log", expanded=False, animated=False)
        self.log_level_combo = QComboBox()
        for label, level in ux_helpers.LOG_LEVEL_CHOICES:
            self.log_level_combo.addItem(label, level)
        self.log_level_combo.setToolTip("Show messages of at least this level")
        self.log_level_combo.currentIndexChanged.connect(self._render_log)
        self.log_filter_edit = QLineEdit()
        self.log_filter_edit.setPlaceholderText("Filter the log…")
        self.log_filter_edit.setClearButtonEnabled(True)
        self.log_filter_edit.textChanged.connect(self._render_log)
        self.save_log_button = SecondaryButton("Save log…")
        self.save_log_button.clicked.connect(lambda _checked=False: self.controller.save_log())
        log_bar = QHBoxLayout()
        log_bar.setSpacing(Spacing.XS)
        log_bar.addWidget(self.log_level_combo)
        log_bar.addWidget(self.log_filter_edit, 1)
        log_bar.addWidget(self.save_log_button)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(20_000)
        self.log_view.setMinimumHeight(120)
        self.log_section.body.addLayout(log_bar)
        self.log_section.body.addWidget(self.log_view)

        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(Spacing.MD)
        column.addWidget(self.progress_card)
        column.addWidget(self.summary_card)
        column.addWidget(self.progress_table, 1)
        column.addWidget(self.log_section)
        self.progress_table.setMinimumHeight(TABLE_MIN_HEIGHT)
        # Squeezed (a small window, the log open) the page scrolls instead of
        # crushing the table.
        return _scroll_page(page)

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def attach(self, window) -> None:
        super().attach(window)
        self.controller.attach(window)
        window.run_requested.connect(self._on_run_requested)
        window.cancel_requested.connect(self.request_cancel)
        self.on_project_changed(window.project)

    def on_project_changed(self, project: Optional[Project]) -> None:
        if self.workspace is None:
            return
        self.controller.on_project_changed(project)
        if not self.is_running:
            self._update_page()
        self.refresh()

    def on_project_modified(self) -> None:
        self._start_issues = []        # what was found may have been fixed
        self.refresh(cheap=not self._is_current())

    def on_selection_changed(self, keys: List[str]) -> None:
        if self._is_current():
            self.refresh()

    def on_enter(self) -> None:
        self.controller.refresh_resume()
        self.refresh()

    def shutdown(self) -> None:
        """Stop the timers that could still close a window that is gone."""
        self.controller._force_timer.stop()
        self.controller._close_timer.stop()

    def _is_current(self) -> bool:
        return (
            self.workspace is not None
            and self.workspace.current_stage_id == self.stage_id
        )

    def _go(self, stage_id: str) -> None:
        if stage_id and self.workspace is not None:
            self.workspace.go_to_stage(stage_id)

    # ------------------------------------------------------------------
    # Delegates to the controller (the names the callers and tests use)
    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self.controller.is_running

    @property
    def summary_label(self):
        """The text of the finished run's summary."""
        return self.summary_callout

    @property
    def run_dir(self) -> Optional[Path]:
        return self.controller.run_dir

    @property
    def resume_offer(self) -> ResumeOffer:
        return self.controller.resume_offer

    @property
    def readiness(self) -> Optional[Readiness]:
        return self._readiness

    def check_rows(self) -> List[CheckRow]:
        """The checklist rows, in the order of the flow."""
        return list(self._check_rows.values())

    def check_row(self, stage_id: str) -> CheckRow:
        return self._check_rows[stage_id]

    def problem_rows(self) -> List[ProblemRow]:
        """The rows of the grouped per-video problems."""
        return list(self._problem_rows)

    @property
    def last_report(self):
        return self.controller.last_report

    def scope(self) -> str:
        return self.controller.scope()

    def set_scope(self, scope: str) -> None:
        index = self.scope_combo.findData(scope)
        if index >= 0:
            self.scope_combo.setCurrentIndex(index)

    def validation_override(self) -> Optional[bool]:
        return self.controller.validation_override()

    def set_validation_override(self, value: Optional[bool]) -> None:
        self.validation_check.setCheckState(
            Qt.CheckState.PartiallyChecked if value is None
            else Qt.CheckState.Checked if value else Qt.CheckState.Unchecked
        )

    def output_dir(self) -> Optional[Path]:
        return self.controller.output_dir()

    def set_output_dir(self, folder: Optional[Path]) -> None:
        self.controller.set_output_dir(folder)
        self.refresh()

    def choose_output_dir(self) -> bool:
        changed = self.controller.choose_output_dir()
        if changed:
            self.refresh()
        return changed

    def prepare(self, keys, strict):
        return self.controller.prepare(keys, strict)

    def start_run(self) -> bool:
        """Check the plan for the chosen scope and start the analysis."""
        return self._note_start(self.controller.start_run())

    def start_resume(self) -> bool:
        return self._note_start(self.controller.start_resume())

    def _note_start(self, started: bool) -> bool:
        """Keep what the pre-run check found on the page until something changes."""
        report = self.controller.last_report
        self._start_issues = (
            [] if started or report is None else list(report.issues)
        )
        self._update_start_issues()
        return started

    def _update_start_issues(self) -> None:
        issues = self._start_issues
        self.start_issues_callout.setVisible(bool(issues))
        self.start_issues_callout.set_text("<br>".join(
            f"• {_escape(text)}" for text in issues[:MAX_LISTED_ISSUES]
        ) + (f"<br>... and {len(issues) - MAX_LISTED_ISSUES} more"
             if len(issues) > MAX_LISTED_ISSUES else ""))
        if issues:
            self.problems.setVisible(True)
            self.problems.set_title(
                "Problems to fix", "Fix them, then run again.")

    def request_cancel(self) -> bool:
        """Ask for confirmation, then stop the run (progress is saved)."""
        return self.controller.request_cancel()

    def log_text(self) -> str:
        """The log lines currently shown (after the filters)."""
        return self.log_view.toPlainText()

    def dismiss_summary(self) -> None:
        self.controller.dismiss_summary()

    def open_run_folder(self) -> bool:
        return self.controller.open_run_folder()

    def open_results(self) -> bool:
        """Show the last run in the Results stage."""
        run_dir = self.controller.run_dir
        window = self.workspace
        if run_dir is None or window is None:
            return False
        results = window.stage(stage_status.STAGE_RESULTS)
        show_run = getattr(results, "show_run", None)
        if show_run is not None:
            show_run(run_dir)
        return window.go_to_stage(stage_status.STAGE_RESULTS)

    # ------------------------------------------------------------------
    # Reading the project into the page
    # ------------------------------------------------------------------
    def refresh(self, cheap: bool = False) -> None:
        """Bring the whole page up to date. ``cheap``: only the run action."""
        project = self.project
        window = self.workspace
        has_work = project is not None and any(
            entry.is_configured for entry in project.videos
        )
        if window is not None:
            window.set_run_enabled(has_work)
        if cheap:
            return
        self._loading = True
        try:
            options = project.run_options if project is not None else None
            self.workers_spin.setValue(
                options.parallel_workers if options is not None else 0
            )
            self.bins_spin.setValue(
                options.time_bin_seconds / 60.0 if options is not None else 0.0
            )
            index = self.scope_combo.findData(self.controller.scope())
            if index >= 0 and index != self.scope_combo.currentIndex():
                self.scope_combo.setCurrentIndex(index)
            self.validation_check.blockSignals(True)
            self.validation_check.setCheckState(self._check_state())
            self.validation_check.blockSignals(False)
        finally:
            self._loading = False
        output = self.output_dir()
        self.output_edit.setText(str(output) if output is not None else "")
        self.output_edit.setCursorPosition(0)
        self.output_edit.setToolTip(
            f"{output}\nEach run creates an analysis_run_ folder in it."
            if output is not None else
            "Each run creates an analysis_run_ folder inside this folder."
        )
        self._refresh_sheet_label()
        self._refresh_validation_label()
        self._update_readiness()
        self._update_page()
        self._update_controls()
        self.refresh_footer()

    def _check_state(self):
        value = self.controller.validation_override()
        return (Qt.CheckState.PartiallyChecked if value is None
                else Qt.CheckState.Checked if value else Qt.CheckState.Unchecked)

    def _update_readiness(self) -> None:
        project = self.project
        window = self.workspace
        if project is None or window is None:
            self._readiness = build_readiness(None, SCOPE_ALL, [])
        else:
            self._readiness = build_readiness(
                project, self.controller.scope(), self.controller.scope_keys(),
                window.project_issues(),
            )
        readiness = self._readiness
        for check in readiness.checks:
            self._check_rows[check.stage_id].show_check(check)
        for row in self._problem_rows:
            row.setParent(None)
            row.deleteLater()
        self._problem_rows = []
        for group in readiness.groups:
            row = ProblemRow()
            row.show_group(group)
            row.fix_button.clicked.connect(
                lambda _checked=False, r=row: self._go(r.fix_target)
            )
            self.problems_body.addWidget(row)
            self._problem_rows.append(row)
        self.problems.setVisible(bool(readiness.groups))
        errors = sum(1 for g in readiness.groups if g.blocking)
        self.problems.set_title(
            "Problems to fix" if errors else "Things to know",
            "These videos stop the run until they are fixed." if errors
            else "The run works, but take a look.",
        )
        self._update_start_issues()
        self._refresh_scope_label()

    def _refresh_scope_label(self) -> None:
        readiness = self._readiness
        project = self.project
        if project is None or readiness is None:
            self.scope_label.setText("No project open")
            return
        text = plural(readiness.configured_count, "video") + " will be analysed"
        if readiness.left_out:
            text += f"; {readiness.left_out} without settings left out"
        if self.controller.scope() == SCOPE_SELECTED and not readiness.scope_count:
            text = "Select videos in the Videos stage first"
        self.scope_label.setText(text)
        # Only worth a line when something is left out or missing; the run
        # card already says how many videos are analysed.
        notable = bool(readiness.left_out) or not readiness.scope_count
        self.scope_label.setVisible(notable)

    def _refresh_sheet_label(self) -> None:
        project = self.project
        has_sheet = False
        if project is None:
            self.sheet_label.setText("")
        else:
            columns = project.metadata_columns()
            sheet = project.run_options.sample_sheet
            has_sheet = bool(columns) or sheet is not None
            if not columns:
                self.sheet_label.setText("No metadata (no sample sheet loaded)")
                self.sheet_label.setToolTip("")
            else:
                filled = sum(
                    1 for video in project.videos
                    if any(video.metadata.get(c) for c in columns)
                )
                source = f"{sheet.name}: " if sheet is not None else ""
                text = (
                    f"{source}{', '.join(columns)} "
                    f"({filled} of {len(project.videos)} videos have values)"
                )
                if sheet is not None and not Path(sheet).is_file():
                    text += " - the sample sheet file was not found"
                self.sheet_label.setText(text)
                self.sheet_label.setToolTip(str(sheet) if sheet is not None else "")
        self.sheet_clear_button.setEnabled(has_sheet and not self.controller.is_busy)
        self.sheet_clear_button.setToolTip(
            "Remove the sample sheet and the metadata columns of every video"
            if has_sheet else "There is no sample sheet to clear"
        )

    def _refresh_validation_label(self) -> None:
        override = self.controller.validation_override()
        self.validation_label.setText(
            "each video keeps its own setting" if override is None
            else "created for every video of this run" if override
            else "not created for any video of this run"
        )

    # ------------------------------------------------------------------
    # State -> widgets
    # ------------------------------------------------------------------
    def _update_page(self) -> None:
        project = self.project
        controller = self.controller
        if project is None or not project.videos or not stage_status.paradigm_known(project):
            if project is None:
                self.empty.set_texts(
                    "No project open",
                    "Open or create a project, then add the videos to analyse.",
                )
            else:
                self.empty.set_texts(
                    "Nothing to run yet",
                    "Choose the test and add the videos first, then come back "
                    "to review and run.",
                )
            self.pages.setCurrentIndex(PAGE_EMPTY)
            return
        if controller.is_running or controller.summary is not None:
            self.pages.setCurrentIndex(PAGE_RUN)
        else:
            self.pages.setCurrentIndex(PAGE_SETUP)

    def _update_controls(self) -> None:
        controller = self.controller
        project = self.project
        readiness = self._readiness
        busy = controller.is_busy
        ready = readiness is not None and readiness.ready
        self.run_button.setEnabled(ready and not busy)
        if busy:
            self.run_button.setToolTip("An analysis is already in progress")
        elif ready:
            self.run_button.setToolTip(
                f"Analyse {plural(readiness.configured_count, 'video')}; the "
                f"results go into a new folder inside {self.output_dir()}"
            )
        else:
            self.run_button.setToolTip(
                readiness.why_not if readiness is not None else "")
        self.cancel_button.setEnabled(controller.can_cancel)
        self.cancel_button.setToolTip(
            "" if controller.can_cancel else
            ("Stopping the analysis..." if controller.is_running
             else "No analysis is running")
        )
        for widget in (self.scope_combo, self.workers_spin, self.bins_spin,
                       self.validation_check, self.output_button,
                       self.sheet_button):
            widget.setEnabled(not busy and project is not None)
            if busy:
                widget.setToolTip("Locked while an analysis is running")
        if not busy:
            self.scope_combo.setToolTip(
                "Which videos to analyse: every video, the ones that have no "
                "finished analysis yet, or the ones selected in the Videos stage."
            )
            self.workers_spin.setToolTip(
                "Videos analysed at once, each in its own process. Auto picks a "
                "number from the CPU cores and memory.")
            self.bins_spin.setToolTip(
                "Also report the metrics in consecutive bins of this many "
                "minutes (batch_time_bins.csv). Off leaves the outputs as they "
                "are without this option.")
            self.validation_check.setToolTip(
                "Partly filled: each video keeps its own setting. Checked or "
                "unchecked: create (or skip) the validation video for every "
                "video of this run. The project's settings are not changed.")
            self.output_button.setToolTip("Choose where the results are written")
            self.sheet_button.setToolTip(
                "Load animal metadata (group, sex, ...) from a CSV into the "
                "video table. The metadata columns are added to the batch summary.")
        offer = controller.resume_offer
        self.resume_box.setVisible(bool(offer) and not busy)
        self.resume_callout.set_text(controller.resume_text())
        self.resume_button.setEnabled(not busy and bool(offer))
        self.resume_button.setVisible(bool(offer))
        # The sentence in the run card.
        if readiness is None or project is None:
            self.ready_label.setText("")
            self.detail_label.setText("")
        elif ready:
            self.ready_label.setText(
                f"Ready to analyse {plural(readiness.configured_count, 'video')}"
            )
            notes = readiness.warnings
            self.detail_label.setText(
                ("Note: " + "; ".join(notes) + ". ") if notes else
                "Results are saved in a new folder of the output folder."
            )
        else:
            self.ready_label.setText("Not ready yet")
            self.detail_label.setText(readiness.why_not)
        self._update_run_page()

    def _update_run_page(self) -> None:
        controller = self.controller
        summary = controller.summary
        running = controller.is_running
        self.cancel_button.setVisible(running or controller.is_checking)
        self.run_title.setText(
            "Analysing" + (f" {plural(self.progress_model.rowCount(), 'video')}"
                           if self.progress_model.rowCount() else "")
        )
        # One card at a time tells the story: the progress while it runs, the
        # summary once it is over (the table stays in both).
        self.progress_card.setVisible(
            running or controller.is_checking or summary is None
        )
        self.summary_card.setVisible(summary is not None)
        if summary is None:
            return
        self.summary_card.set_title(
            "Analysis stopped" if summary.cancelled else "Analysis finished"
        )
        self.summary_callout.set_kind(summary.kind)
        self.summary_callout.set_text(summary.text())
        self.completed_chip.set_text(f"{summary.completed} completed")
        self.failed_chip.set_text(f"{summary.failed} failed")
        self.warnings_chip.set_text(f"{plural(summary.warnings, 'warning')}")
        self.failed_chip.setVisible(summary.failed > 0)
        self.warnings_chip.setVisible(summary.warnings > 0)
        self.cancelled_chip.setVisible(summary.cancelled)
        stopping = summary.stopping_for_close
        self.open_results_button.setEnabled(controller.has_results and not stopping)
        self.open_results_button.setToolTip(
            "" if controller.has_results else "This run did not write any results")
        self.open_folder_button.setEnabled(
            controller.run_dir is not None and Path(controller.run_dir).is_dir())
        offer = controller.resume_offer
        self.resume_run_button.setVisible(bool(offer))
        self.resume_run_button.setEnabled(bool(offer) and not controller.is_busy)
        self.back_button.setEnabled(not running)
        self.back_button.setToolTip(
            "Wait for the analysis to stop" if running else "")

    def _on_state_changed(self) -> None:
        if self.workspace is None:
            return
        self.refresh()

    def _on_run_started(self) -> None:
        self.log_view.clear()
        self.pages.setCurrentIndex(PAGE_RUN)
        self._shown_row = -1

    def _on_progress_changed(self) -> None:
        controller = self.controller
        self.progress_bar.setRange(0, controller.progress_maximum)
        self.progress_bar.setValue(controller.progress_value)
        self.status_label.setText(controller.status_text)
        self.eta_label.setText(controller.eta_text)
        self.eta_label.setVisible(bool(controller.eta_text))
        row = controller.current_row()
        if row >= 0 and row != self._shown_row:
            self._shown_row = row
            self.progress_table.scrollTo(self.progress_model.index(row, 0))
        if controller.summary is not None or controller.is_running:
            self._update_run_page()
        self.refresh_footer()

    # -- the log ---------------------------------------------------------
    def _log_matches(self, level: int, text: str) -> bool:
        minimum = self.log_level_combo.currentData() or 0
        needle = self.log_filter_edit.text().strip().lower()
        return level >= minimum and (not needle or needle in text.lower())

    def _on_log_added(self, level: int, text: str) -> None:
        if self._log_matches(level, text):
            self.log_view.appendPlainText(text)

    def _render_log(self, *_args) -> None:
        self.log_view.setPlainText("\n".join(
            text for level, text in self.controller.log_entries
            if self._log_matches(level, text)
        ))

    # ------------------------------------------------------------------
    # Edits of the options
    # ------------------------------------------------------------------
    def _on_workers_changed(self, value: int) -> None:
        project = self.project
        if self._loading or project is None:
            return
        project.set_run_options(parallel_workers=value)

    def _on_bins_changed(self, value: float) -> None:
        project = self.project
        if self._loading or project is None:
            return
        project.set_run_options(time_bin_seconds=round(float(value) * 60.0, 3))

    def _on_scope_changed(self, _index: int = 0) -> None:
        if self._loading:
            return
        self.controller.set_scope(self.scope_combo.currentData() or SCOPE_ALL)

    def _on_validation_changed(self, _state: int = 0) -> None:
        state = self.validation_check.checkState()
        self.controller.set_validation_override(
            True if state == Qt.CheckState.Checked
            else False if state == Qt.CheckState.Unchecked else None
        )
        self._refresh_validation_label()

    def _load_sample_sheet(self) -> None:
        if self.workspace is None or self.controller.is_busy:
            return
        self.workspace.video_panel.load_sample_sheet()

    def _clear_sample_sheet(self) -> bool:
        project = self.project
        if project is None or self.workspace is None or self.controller.is_busy:
            return False
        if not self.workspace.prompts.confirm(
            "Clear the sample sheet",
            "Remove the sample sheet and the metadata of every video?",
            accept="Clear",
            informative=(
                "Runs then add no metadata columns to the batch summary. "
                "You can load the sheet again at any time."
            ),
            destructive=True,
        ):
            return False
        return project.clear_sample_sheet()

    # -- starting --------------------------------------------------------
    def _on_run_clicked(self) -> bool:
        if not self.run_button.isEnabled():
            return False
        return self.start_run()

    def _on_shortcut(self) -> None:
        self._on_run_clicked()

    def _on_run_requested(self) -> None:
        """The Run action (menu): show this stage and start, if it can."""
        window = self.workspace
        if window is None or self.controller.is_busy:
            return
        if not self._is_current() and not window.go_to_stage(self.stage_id):
            return
        self.refresh()
        readiness = self._readiness
        if readiness is not None and not readiness.ready:
            window.show_message(readiness.why_not)
            return
        self.start_run()

    # ------------------------------------------------------------------
    # What the window shows for this stage
    # ------------------------------------------------------------------
    def can_advance(self) -> Tuple[bool, str]:
        return True, ""

    def footer_text(self):
        controller = self.controller
        if controller.is_running:
            return (
                f"{controller.status_text} ({controller.progress_value}%)",
                ("running", "Analysing"),
            )
        summary = controller.summary
        if summary is not None:
            chip = {"success": ("ok", "Finished"),
                    "warning": ("warning", "Needs a look"),
                    "error": ("error", "Finished with errors")}[summary.kind]
            return summary.headline(), chip
        readiness = self._readiness
        if readiness is None or self.project is None:
            return "", None
        if readiness.ready:
            return (
                f"{plural(readiness.configured_count, 'video')} will be analysed",
                ("ok", "Ready"),
            )
        return readiness.why_not, ("warning", "Needs attention")

    def next_label(self) -> Optional[str]:
        return self.next_stage_title


__all__ = ["ReviewStage", "CheckRow", "ProblemRow", "PAGE_EMPTY", "PAGE_RUN",
           "PAGE_SETUP"]
