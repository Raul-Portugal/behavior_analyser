"""
gui/workspace/stages/trials_stage.py
Stage 4: when does each trial start and end?

In the Y-maze, the elevated plus maze and the open field every mouse is put
in the apparatus at a different moment, so every video needs **its own start
time**; its end is usually that start plus a fixed trial duration (5 or 8
minutes). In the Tail Suspension Test every mouse of a video has its own
window instead. This stage lets the user tag a whole cohort quickly::

    +------------------------------+-----------------------------------+
    | Trial protocol (collapsible) |                                   |
    |                              |        the selected video         |
    | Status  Video  Start  End ...|                                   |
    | one row per video            |  [|-----====shaded trial====---|]  |
    | (TST: per mouse, grouped)    |  [Set start here S] [Set end E]   |
    +------------------------------+-----------------------------------+

* **Table (left)**: one row per video with its status, start, end, duration
  and where the times come from (the *protocol* or *manual*). Start, end and
  duration are edited in place (``90``, ``1:30`` or ``1:30.5``). A row that
  has no end of its own shows the protocol's end greyed with a "protocol"
  tag. Anything that cannot be used shows in red with the reason as a
  tooltip, and what was typed is kept.
* **Player (right)**: the selected video. ``S`` stores the frame on screen as
  the start, ``E`` as the end, ``Left`` / ``Right`` step a frame (``Shift``:
  ten), and ``Enter`` (or ``N``) jumps to the next video that still needs a
  start, so a cohort is tagged with the keyboard alone.
* **Protocol card**: the trial's duration ("until the video ends" or a fixed
  length) and whether trials start with the video or at a moment set per
  video; "Apply to all videos" makes every video follow it again.

Every edit is written to the project at once (``Project.set_trial``,
``set_trial_protocol``); there is no Apply button. Tail Suspension Test rows
are written with ``set_trial(per_roi_times=...)`` and the mice come from the
maze's ``time_window_subjects``; that test has no trial protocol (every mouse
has its own window), so the card is hidden and the protocol is kept at
"at the start of each video".

"Next: Review & run" is enabled once every video's trial is valid and has
its start. The "Show next" button in the footer does the same as ``Enter``.
"""
import logging
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import QModelIndex, Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QHBoxLayout,
                             QLineEdit, QRadioButton, QSplitter,
                             QStackedWidget, QVBoxLayout, QWidget)

from core.project import (START_PER_VIDEO, START_VIDEO, TRIAL_INVALID,
                          TRIAL_NEEDS_START, TRIAL_OK, UNSET, Project,
                          ProjectError)
from gui.workspace.stages import status as stage_status
from gui.workspace.stages.base import Stage
from gui.workspace.stages.trials_widgets import (
    COL_DURATION, COL_END, COL_START, KIND_MOUSE, ORIGIN_MANUAL, PlayerPane,
    TrialRow, format_time, format_time_exact, make_table, parse_time,
    round_time, trial_rows)
from gui.workspace.theme import Spacing, current_palette
from gui.workspace.ui_kit import (Callout, Card, CollapsibleSection,
                                  EmptyState, GhostButton, Label,
                                  SecondaryButton)
from mazes import create_maze

logger = logging.getLogger(__name__)

PAGE_EMPTY = 0
PAGE_MAIN = 1

TRIALS_SUBTITLE = (
    "Choose when each trial starts and ends. Every video can have its own times."
)
TST_SUBTITLE = "Choose the time window of each mouse in each video."
DEFAULT_DURATION_S = 300.0
NEXT_LABEL = "Review & run"
SHOW_NEXT_TEXT = "Show next"
END_HELP = "End = start + duration unless you set an end yourself."


def plural(count: int, noun: str) -> str:
    return stage_status.plural(count, noun)


# ----------------------------------------------------------------------
# The protocol card
# ----------------------------------------------------------------------
class ProtocolPanel(Card):
    """How long a trial lasts and where it starts, for the whole project.

    Emits ``changed(duration_s, start_mode)`` for every accepted edit
    (``duration_s`` None: until the video ends) and ``apply_requested`` for
    "Apply to all videos".
    """

    changed = pyqtSignal(object, str)
    apply_requested = pyqtSignal()

    def __init__(self, settings=None, parent: Optional[QWidget] = None):
        super().__init__(parent=parent, padding=Spacing.SM)
        self._duration_s: Optional[float] = None
        self._mode = START_VIDEO
        self.section = CollapsibleSection(
            "Trial protocol", expanded=True, settings=settings,
            settings_key="trials_protocol", animated=False,
        )
        self.add_widget(self.section)

        # -- duration ---------------------------------------------------
        self.fixed_radio = QRadioButton("Fixed length")
        self.until_end_radio = QRadioButton("Until the video ends")
        self.duration_group = QButtonGroup(self)
        self.duration_group.addButton(self.fixed_radio)
        self.duration_group.addButton(self.until_end_radio)
        self.duration_edit = QLineEdit()
        self.duration_edit.setPlaceholderText("m:ss")
        self.duration_edit.setFixedWidth(84)
        self.duration_edit.setToolTip(
            "How long each trial lasts, in minutes:seconds (5:00) or seconds (300)"
        )
        self.duration_error = Label("", "caption")
        self.duration_error.setStyleSheet(f"color: {current_palette().danger};")
        self.duration_error.setVisible(False)
        self.duration_error.setWordWrap(True)
        fixed_row = QHBoxLayout()
        fixed_row.setContentsMargins(0, 0, 0, 0)
        fixed_row.setSpacing(Spacing.XS)
        fixed_row.addWidget(self.fixed_radio)
        fixed_row.addWidget(self.duration_edit)
        fixed_row.addStretch(1)
        duration_col = QVBoxLayout()
        duration_col.setSpacing(Spacing.XXS)
        duration_col.addWidget(Label("Trial duration", "caption"))
        duration_col.addLayout(fixed_row)
        duration_col.addWidget(self.until_end_radio)
        duration_col.addWidget(self.duration_error)
        duration_col.addStretch(1)

        # -- start ------------------------------------------------------
        self.start_video_radio = QRadioButton("At the start of each video")
        self.start_per_video_radio = QRadioButton(
            "Set per video (when the mouse enters)"
        )
        self.start_group = QButtonGroup(self)
        self.start_group.addButton(self.start_video_radio)
        self.start_group.addButton(self.start_per_video_radio)
        start_col = QVBoxLayout()
        start_col.setSpacing(Spacing.XXS)
        start_col.addWidget(Label("Trial start", "caption"))
        start_col.addWidget(self.start_video_radio)
        start_col.addWidget(self.start_per_video_radio)
        start_col.addStretch(1)

        columns = QHBoxLayout()
        columns.setSpacing(Spacing.LG)
        columns.addLayout(duration_col)
        columns.addLayout(start_col, 1)
        self.section.body.addLayout(columns)

        # -- apply -------------------------------------------------------
        self.help_label = Label(END_HELP, "muted", wrap=True)
        self.apply_button = SecondaryButton("Apply to all videos")
        self.apply_button.setToolTip(
            "Make every video follow the protocol: its own ends are removed "
            "(and, when trials start with the video, its own starts too)"
        )
        apply_row = QHBoxLayout()
        apply_row.setSpacing(Spacing.SM)
        apply_row.addWidget(self.help_label, 1)
        apply_row.addWidget(self.apply_button)
        self.section.body.addLayout(apply_row)

        self.fixed_radio.clicked.connect(self._on_fixed_clicked)
        self.until_end_radio.clicked.connect(self._on_until_end_clicked)
        self.duration_edit.editingFinished.connect(self._on_duration_edited)
        self.duration_edit.textChanged.connect(self._on_duration_typing)
        self.start_video_radio.clicked.connect(
            lambda _checked=False: self._on_mode_clicked(START_VIDEO))
        self.start_per_video_radio.clicked.connect(
            lambda _checked=False: self._on_mode_clicked(START_PER_VIDEO))
        self.apply_button.clicked.connect(
            lambda _checked=False: self.apply_requested.emit())
        self.section.toggled.connect(lambda _expanded: self._update_summary())
        self.set_state(None, START_VIDEO)

    # -- state -----------------------------------------------------------
    def state(self) -> Tuple[Optional[float], str]:
        return self._duration_s, self._mode

    def set_state(self, duration_s: Optional[float], mode: str) -> None:
        """Show the project's protocol (never emits ``changed``)."""
        self._duration_s, self._mode = duration_s, mode
        typing = self.duration_edit.hasFocus()
        self.fixed_radio.setChecked(duration_s is not None)
        self.until_end_radio.setChecked(duration_s is None)
        if duration_s is not None and not typing:
            self.duration_edit.setText(format_time_exact(duration_s))
            self._show_duration_error("")
        elif duration_s is None and not typing and not self.duration_edit.text():
            self.duration_edit.setText(format_time(DEFAULT_DURATION_S, 0))
        self.duration_edit.setEnabled(duration_s is not None and self.isEnabled())
        self.start_video_radio.setChecked(mode == START_VIDEO)
        self.start_per_video_radio.setChecked(mode == START_PER_VIDEO)
        self._update_summary()

    def summary_text(self) -> str:
        duration = ("until the video ends" if self._duration_s is None
                    else f"{format_time(self._duration_s, 0 if self._duration_s % 1 == 0 else 1)} per trial")
        start = ("start set per video" if self._mode == START_PER_VIDEO
                 else "starts with the video")
        return f"{duration} · {start}"

    def _update_summary(self) -> None:
        self.section.set_summary("" if self.section.is_expanded() else self.summary_text())

    def set_editable(self, editable: bool, why: str = "") -> None:
        for widget in (self.fixed_radio, self.until_end_radio, self.start_video_radio,
                       self.start_per_video_radio, self.apply_button):
            widget.setEnabled(editable)
            widget.setToolTip("" if editable else why)
        self.duration_edit.setEnabled(editable and self._duration_s is not None)

    # -- edits ------------------------------------------------------------
    def _show_duration_error(self, text: str) -> None:
        self.duration_error.setText(text)
        self.duration_error.setVisible(bool(text))
        palette = current_palette()
        self.duration_edit.setStyleSheet(
            f"QLineEdit {{ border: 1px solid {palette.danger}; }}" if text else ""
        )
        self.duration_edit.setToolTip(
            text or "How long each trial lasts, in minutes:seconds (5:00) or seconds (300)"
        )

    def _on_duration_typing(self, text: str) -> None:
        if not self.duration_edit.hasFocus():
            return
        try:
            value = parse_time(text)
            if value <= 0:
                raise ValueError("The trial duration must be above 0")
        except ValueError as exc:
            self._show_duration_error(str(exc))
        else:
            self._show_duration_error("")

    def _on_fixed_clicked(self, _checked: bool = False) -> None:
        self.duration_edit.setEnabled(True)
        self.duration_edit.setFocus()
        self.duration_edit.selectAll()
        self._commit_duration()

    def _on_until_end_clicked(self, _checked: bool = False) -> None:
        self._show_duration_error("")
        self.duration_edit.setEnabled(False)
        self.changed.emit(None, self._mode)

    def _on_duration_edited(self) -> None:
        if self.fixed_radio.isChecked():
            self._commit_duration()

    def _commit_duration(self) -> bool:
        text = self.duration_edit.text()
        try:
            value = parse_time(text)
            if value <= 0:
                raise ValueError("The trial duration must be above 0")
        except ValueError as exc:
            self._show_duration_error(str(exc))
            return False
        self._show_duration_error("")
        if value != self._duration_s:
            self.changed.emit(value, self._mode)
        return True

    def flush(self) -> bool:
        """Accept a duration that is being typed. False: it is not valid (the
        card keeps showing why)."""
        if self.fixed_radio.isChecked() and self.duration_edit.isEnabled():
            return self._commit_duration()
        return True

    def _on_mode_clicked(self, mode: str) -> None:
        if mode != self._mode:
            self.changed.emit(self._duration_s, mode)


# ----------------------------------------------------------------------
# The stage
# ----------------------------------------------------------------------
class TrialsStage(Stage):
    """Stage 4: the start and end of every trial."""

    stage_id = stage_status.STAGE_TRIALS
    title = "Trials"
    subtitle = TRIALS_SUBTITLE
    icon_name = "clock"
    CONTENT_MARGIN = Spacing.MD

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._active = False
        self._dirty = False
        self._maze_cache: Dict[str, object] = {}
        self._video_seconds: Dict[str, float] = {}
        self._shown_key: Optional[str] = None
        self._loaded = False
        self._can_capture = False
        self._tst_checked = False
        self._pending_selection: Optional[Tuple[str, Optional[str]]] = None
        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build(self) -> None:
        self.pages = QStackedWidget()
        self.empty = EmptyState("clock", "", "")
        self.empty.add_action("Go to Videos")
        self.empty.buttons[0].clicked.connect(lambda _checked=False: self._empty_action())
        self._empty_target = stage_status.STAGE_VIDEOS
        self.pages.addWidget(self.empty)

        self.protocol = ProtocolPanel()
        self.protocol.changed.connect(self._on_protocol_changed)
        self.protocol.apply_requested.connect(self.apply_protocol_to_all)

        self.table, self.model, self.delegate = make_table()
        self.model.commit = self.commit_cell
        self.table.key_handler = self.handle_key
        self.table.selectionModel().currentRowChanged.connect(self._on_current_row)
        self.delegate.closeEditor.connect(lambda *_args: self._after_editor_closed())

        self.notice = Callout("info", "")
        self.notice.setVisible(False)
        self.summary_label = Label("", "muted", wrap=True)
        self.next_button = SecondaryButton("Next video needing a start", "arrow-right")
        self.next_button.clicked.connect(lambda _checked=False: self.next_needing_start())
        bottom = QHBoxLayout()
        bottom.setSpacing(Spacing.SM)
        bottom.addWidget(self.summary_label, 1)
        bottom.addWidget(self.next_button)

        left = QWidget()
        left_col = QVBoxLayout(left)
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(Spacing.SM)
        left_col.addWidget(self.protocol)
        left_col.addWidget(self.notice)
        left_col.addWidget(self.table, 1)
        left_col.addLayout(bottom)

        self.pane = PlayerPane()
        self.pane.key_handler = self.handle_key
        self.player = self.pane.player
        self.bar = self.pane.bar
        self.pane.set_start_button.clicked.connect(lambda _checked=False: self.set_start_here())
        self.pane.set_end_button.clicked.connect(lambda _checked=False: self.set_end_here())
        self.pane.to_start_button.clicked.connect(lambda _checked=False: self.jump_to_start())
        self.bar.seek_requested.connect(self._on_bar_seek)
        self.player.frame_changed.connect(self._on_frame_changed)
        self.player.video_loaded.connect(self._on_video_loaded)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(left)
        self.splitter.addWidget(self.pane)
        left.setMinimumWidth(430)
        self.pane.setMinimumWidth(460)
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 3)
        self.splitter.setSizes([400, 600])
        self.pages.addWidget(self.splitter)

        self.show_next_button = GhostButton(SHOW_NEXT_TEXT, "arrow-right")
        self.show_next_button.setToolTip("Select the next video that needs a start (Enter)")
        self.show_next_button.clicked.connect(lambda _checked=False: self.next_needing_start())
        self.show_next_button.setVisible(False)
        self.set_content(self.pages)
        self.refresh()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def attach(self, window) -> None:
        super().attach(window)
        footer = getattr(window, "footer", None)
        if footer is not None:
            footer.add_extra(self.show_next_button)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)
        self.refresh()

    def on_project_changed(self, project: Optional[Project]) -> None:
        self.shutdown()
        self._maze_cache.clear()
        self._video_seconds.clear()
        self.model.clear_bad()
        self._pending_selection = None
        self._tst_checked = False
        self._shown_key = None
        self.protocol.section.set_expanded(not self._protocol_configured())
        self.refresh()

    def on_project_modified(self) -> None:
        self.refresh()

    def on_selection_changed(self, keys: List[str]) -> None:
        pass

    def on_enter(self) -> None:
        self._active = True
        self._normalize_protocol()
        self.refresh()
        self._choose_initial_row()
        self._sync_extras()

    def on_leave(self) -> bool:
        """Accept what is being typed, then release the video."""
        self.flush()
        self._active = False
        self._close_player()
        self._sync_extras()
        return True

    def flush(self) -> None:
        """Write the edits in progress into the project (nothing is lost)."""
        self.delegate.commit_open_editor()
        self.protocol.flush()

    def shutdown(self) -> None:
        """Release the video (the window is closing or the project changed)."""
        try:
            self._close_player()
        except RuntimeError:         # the widget is already deleted
            return

    def _close_player(self) -> None:
        self.player.close_video()
        self._loaded = False
        self._shown_key = None

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def _protocol_configured(self) -> bool:
        project = self.project
        if project is None:
            return False
        protocol = project.trial_protocol
        return protocol.duration_s is not None or protocol.start_mode != START_VIDEO

    def _paradigm(self) -> str:
        project = self.project
        return "" if project is None else project.effective_paradigm()

    def maze_for(self, paradigm: str):
        """The (cached) maze of a paradigm, None for an unknown one."""
        if paradigm not in self._maze_cache:
            try:
                self._maze_cache[paradigm] = create_maze(paradigm)
            except (ValueError, TypeError, KeyError):
                self._maze_cache[paradigm] = None
        return self._maze_cache[paradigm]

    def per_mouse(self) -> bool:
        """True for a paradigm with one window per mouse (Tail Suspension Test)."""
        maze = self.maze_for(self._paradigm())
        return maze is not None and maze.has_per_roi_times

    def is_read_only(self) -> bool:
        return self.workspace is not None and self.workspace.is_busy

    def video_seconds(self, key: str) -> Optional[float]:
        """The length of a video when known (project, thumbnails or the player)."""
        project = self.project
        if project is None:
            return None
        try:
            entry = project.entry(key)
        except KeyError:
            return None
        if entry.duration_s:
            return entry.duration_s
        panel = getattr(self.workspace, "video_panel", None)
        found = panel.duration_of(key) if panel is not None else None
        return found or self._video_seconds.get(key)

    def _empty_reason(self) -> Optional[Tuple[str, str, str, str, str]]:
        """``(icon, title, text, button, target stage)`` while there is nothing to show."""
        project = self.project
        videos = stage_status.STAGE_VIDEOS
        if project is None:
            return ("folder", "Open a study first",
                    "Trial times belong to a study. Open or create one on the Videos page.",
                    "Go to Videos", videos)
        if not stage_status.paradigm_known(project):
            return ("maze-field", "Choose the test first",
                    "Trials depend on the test: a single start for most, one window "
                    "per mouse for the Tail Suspension Test.",
                    "Go to Videos", videos)
        if not project.videos:
            return ("video", "Add your videos first",
                    "Every video gets its own trial times. Add the recordings you want "
                    "to analyse.", "Go to Videos", videos)
        if self.per_mouse() and not self._has_zones():
            return ("zones", "Draw the mouse zones first",
                    "Each mouse has its own time window, and the mice come from the "
                    "zones. Draw them, then come back.",
                    "Go to Arena & zones", stage_status.STAGE_ARENA)
        return None

    def _has_zones(self) -> bool:
        project = self.project
        return project is not None and any(
            project.effective_settings(v) is not None for v in project.videos
        )

    def _empty_action(self) -> None:
        if self.workspace is not None:
            self.workspace.go_to_stage(self._empty_target)

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Show the project's trials (cheap; never interrupts typing)."""
        if self.delegate.has_open_editor():
            self._dirty = True
            return
        self._dirty = False
        project = self.project
        reason = self._empty_reason()
        if project is None or reason is not None:
            self._show_empty(reason)
            return
        self.pages.setCurrentIndex(PAGE_MAIN)
        tst = self.per_mouse()
        subtitle = TST_SUBTITLE if tst else TRIALS_SUBTITLE
        if subtitle != self.subtitle:
            self.set_texts(subtitle=subtitle)
        read_only = self.is_read_only()
        self.model.read_only = read_only
        self.protocol.setVisible(not tst)
        self.protocol.set_editable(not read_only, "A run is in progress")
        self.protocol.set_state(project.trial_protocol.duration_s,
                                project.trial_protocol.start_mode)
        self._show_notice(project, tst)

        selected = self.selected_identity() or self._pending_selection
        rows = trial_rows(project, self.video_seconds, self.maze_for)
        self.model.set_rows(rows)
        self._restore_selection(selected)
        self._update_summary(rows)
        self._update_pane()
        self._sync_extras()
        self.refresh_footer()

    def _show_empty(self, reason) -> None:
        self.pages.setCurrentIndex(PAGE_EMPTY)
        if reason is None:
            reason = ("clock", "Nothing to set yet", "", "Go to Videos",
                      stage_status.STAGE_VIDEOS)
        icon, title, text, button, target = reason
        self.empty.set_icon_name(icon)
        self.empty.set_texts(title, text)
        self.empty.buttons[0].setText(button)
        self._empty_target = target
        self.model.set_rows([])
        self._close_player()
        self._sync_extras()
        self.refresh_footer()

    def _show_notice(self, project: Project, tst: bool) -> None:
        text = ""
        if not tst and project.template_settings is None and not any(
            v.settings is not None for v in project.videos
        ):
            text = ("The arena and zones are not set yet. You can enter the times "
                    "now; set up the arena and zones before you run.")
        self.notice.setVisible(bool(text))
        if text:
            self.notice.set_text(text)

    def _after_editor_closed(self) -> None:
        if self._dirty:
            self.refresh()

    def _restore_selection(self, identity: Optional[Tuple[str, Optional[str]]]) -> None:
        if identity is None:
            return
        row = self.model.row_of(*identity)
        if row < 0:
            row = self.model.row_of(identity[0])
        if row >= 0 and self.table.currentIndex().row() != row:
            self._select_index(row, load=False)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def selected_row(self) -> Optional[TrialRow]:
        index = self.table.currentIndex()
        if not index.isValid():
            return None
        row = self.model.row_at(index.row())
        return None if row is None or row.is_header else row

    def selected_identity(self) -> Optional[Tuple[str, Optional[str]]]:
        row = self.selected_row()
        return None if row is None else row.identity()

    def select_row(self, key: str, subject: Optional[str] = None) -> bool:
        """Select a video (or one of its mice) and show it in the player."""
        index = self.model.row_of(key, subject)
        if index < 0:
            return False
        self._select_index(index, load=True)
        return True

    def _select_index(self, row: int, load: bool) -> None:
        column = self.table.currentIndex().column()
        if column not in (COL_START, COL_END, COL_DURATION):
            column = COL_START
        index = self.model.index(row, column)
        if not load:
            self.table.selectionModel().blockSignals(True)
        try:
            self.table.setCurrentIndex(index)
            self.table.selectRow(row)
            self.table.setCurrentIndex(index)
        finally:
            if not load:
                self.table.selectionModel().blockSignals(False)
        self.table.scrollTo(index)

    def _choose_initial_row(self) -> None:
        """Keep the selection, else the video picked on the Videos page, else
        the first one that needs a start, else the first."""
        if self.pages.currentIndex() != PAGE_MAIN:
            return
        current = self.selected_row()
        if current is not None:
            self._show_row(current, seek=True)
            return
        picked = [] if self.workspace is None else self.workspace.selected_keys()
        for key in picked:
            if self.select_row(key):
                return
        rows = self.model.rows()
        for index, row in enumerate(rows):
            if not row.is_header and row.needs_attention:
                self._select_index(index, load=True)
                return
        for index, row in enumerate(rows):
            if not row.is_header:
                self._select_index(index, load=True)
                return

    def _on_current_row(self, current: QModelIndex, _previous: QModelIndex) -> None:
        row = self.model.row_at(current.row()) if current.isValid() else None
        if row is None or row.is_header:
            return
        self._show_row(row, seek=True)

    # ------------------------------------------------------------------
    # The player
    # ------------------------------------------------------------------
    def _show_row(self, row: TrialRow, seek: bool) -> None:
        """Load the row's video (when it is not loaded) and go to its start."""
        project = self.project
        if project is None or not self._active:
            self._update_pane()
            return
        entry = project.entry(row.key)
        if self._shown_key != row.key or not self._loaded:
            self._close_player()
            self._shown_key = row.key
            if entry.path.exists():
                self._loaded = self.player.open_video(entry.path)
            else:
                self.player.view.set_frame(None)
                self.player.view.setText(f"The file of {entry.name} could not be found")
        if seek and self._loaded:
            self.player.seek_to_time(row.start if row.start_known and row.start else 0.0)
        self._update_pane()

    def _on_video_loaded(self, total: int, fps: float) -> None:
        row = self.selected_row()
        if row is not None and fps > 0:
            self._video_seconds[row.key] = total / fps
        self._update_pane()

    def _on_frame_changed(self, _index: int, seconds: float) -> None:
        self.bar.set_playhead(seconds)
        self._update_pane(readout_only=self._can_capture == (self._capture_block() is None))

    def _on_bar_seek(self, seconds: float) -> None:
        if self._loaded:
            self.player.pause()
            self.player.seek_to_time(seconds)

    def playhead_seconds(self) -> Optional[float]:
        """The time of the frame on screen (None without a video)."""
        if not self._loaded or self.player.current_frame < 0:
            return None
        return round_time(self.player.current_timestamp)

    def _capture_block(self) -> Optional[str]:
        """Why the playhead cannot be used now (None: it can)."""
        if self.is_read_only():
            return "A run is in progress"
        if self.selected_row() is None:
            return "Select a video in the table first"
        if not self._loaded:
            return "The video could not be opened, so there is no frame to take a time from"
        return None

    def _update_pane(self, readout_only: bool = False) -> None:
        row = self.selected_row()
        block = self._capture_block()
        self._can_capture = block is None
        duration = (self.player.total_frames / self.player.fps
                    if self._loaded and self.player.fps > 0
                    else (self.video_seconds(row.key) if row is not None else None))
        pane = self.pane
        if not readout_only:
            for button in (pane.set_start_button, pane.set_end_button):
                button.setEnabled(block is None)
            pane.set_start_button.setToolTip(
                block or "Use the time of the frame on screen as the start (S)")
            pane.set_end_button.setToolTip(
                block or "Use the time of the frame on screen as the end (E)")
            pane.to_start_button.setEnabled(row is not None and self._loaded)
            pane.to_start_button.setToolTip(
                "Go to this trial's start" if row is not None
                else "Select a video in the table first")
            self.bar.set_video(duration or 0.0)
            if row is None:
                self.bar.set_window(None, None)
            else:
                start = row.start if row.start_known else None
                end = row.end
                others = []
                if row.kind == KIND_MOUSE:
                    others = [
                        (r.start, r.end) for r in self.model.rows()
                        if r.key == row.key and r.kind == KIND_MOUSE
                        and r.subject != row.subject and r.end is not None
                        and r.start is not None
                    ]
                if row.kind == KIND_MOUSE and not row.start_known:
                    start = None
                self.bar.set_window(start, end, row.end_origin != ORIGIN_MANUAL, others)
        self.bar.set_playhead(self.player.current_timestamp if self._loaded else 0.0)
        pane.readout.setText(self._readout(row))

    def _readout(self, row: Optional[TrialRow]) -> str:
        if row is None:
            return "Select a video to set its trial"
        start = format_time(row.start) if row.start_known else "not set"
        if row.end is None:
            end = "video end"
        else:
            end = format_time(row.end) + ("" if row.end_manual else f" ({row.end_origin})")
        subject = row.label
        return f"{subject}   Start {start}   End {end}"

    # ------------------------------------------------------------------
    # Writing times
    # ------------------------------------------------------------------
    def commit_cell(self, row: TrialRow, column: int, text: str) -> Optional[str]:
        """Write what was typed in a cell; None when written, else the reason
        it was not (the cell keeps the text, in red)."""
        if self.project is None:
            return "No project is open"
        text = text.strip()
        if not text:
            value: Optional[float] = None
        else:
            try:
                value = parse_time(text)
            except ValueError as exc:
                return str(exc)
        return self.apply_time(row, column, value)

    def apply_time(self, row: TrialRow, column: int, value: Optional[float]) -> Optional[str]:
        """Set (or, with None, clear) the start, end or duration of a row."""
        project = self.project
        if project is None:
            return "No project is open"
        if self.is_read_only():
            return "A run is in progress"
        try:
            if row.kind == KIND_MOUSE:
                error = self._apply_mouse(project, row, column, value)
            else:
                error = self._apply_video(project, row, column, value)
        except ProjectError as exc:
            return str(exc)
        if error is None:
            self._seek_after_edit(row, column, value)
        return error

    def _apply_video(self, project: Project, row: TrialRow, column: int,
                     value: Optional[float]) -> Optional[str]:
        key = row.key
        if column == COL_START:
            project.set_trial(key, start=UNSET if value is None else value)
        elif column == COL_END:
            project.set_trial(key, end=UNSET if value is None else value)
        elif column == COL_DURATION:
            if value is None:
                project.set_trial(key, end=UNSET)
            elif value <= 0:
                return "A duration must be above 0"
            elif not row.start_known or row.start is None:
                return "Set the start first: the end is the start plus the duration"
            else:
                project.set_trial(key, end=round_time(row.start + value))
        return None

    def _apply_mouse(self, project: Project, row: TrialRow, column: int,
                     value: Optional[float]) -> Optional[str]:
        entry = project.entry(row.key)
        settings = project.effective_settings(entry)
        windows = {} if settings is None else {
            name: tuple(w) for name, w in settings.per_roi_times.items()
        }
        name = row.subject
        current = windows.get(name)
        if value is None:
            if column == COL_END and current is not None:
                windows[name] = (current[0], self._whole_end(row))
                if windows[name][1] is None:
                    return "Type an end: this mouse needs one while it has a start"
            else:
                windows.pop(name, None)
        elif column == COL_START:
            end = current[1] if current is not None else self._whole_end(row)
            if end is None:
                return "Also set this mouse's end (E, or type it): the video's length is not known yet"
            windows[name] = (value, end)
        elif column == COL_END:
            windows[name] = ((current[0] if current is not None else 0.0), value)
        elif column == COL_DURATION:
            if value <= 0:
                return "A duration must be above 0"
            start = current[0] if current is not None else 0.0
            windows[name] = (start, round_time(start + value))
        project.set_trial(row.key, per_roi_times=windows or UNSET)
        return None

    def _whole_end(self, row: TrialRow) -> Optional[float]:
        """The end a mouse gets when only its start is given: the video's end."""
        return self.video_seconds(row.key)

    def _seek_after_edit(self, row: TrialRow, column: int, value: Optional[float]) -> None:
        if value is None or not self._active or not self._loaded:
            return
        if self._shown_key != row.key or column not in (COL_START, COL_END):
            return
        self.player.pause()
        self.player.seek_to_time(value)

    def set_start_here(self) -> bool:
        """Store the time of the frame on screen as the selected trial's start."""
        return self._capture(COL_START)

    def set_end_here(self) -> bool:
        """Store the time of the frame on screen as the selected trial's end."""
        return self._capture(COL_END)

    def _capture(self, column: int) -> bool:
        block = self._capture_block()
        if block is not None:
            self._say(block)
            return False
        self.player.wait_idle(1000)
        seconds = self.playhead_seconds()
        row = self.selected_row()
        if seconds is None or row is None:
            self._say("Go to the frame you want first")
            return False
        error = self.apply_time(row, column, seconds)
        if error:
            self._say(error)
            return False
        what = "Start" if column == COL_START else "End"
        self._say(f"{what} of {row.label} set to {format_time(seconds)}")
        return True

    def jump_to_start(self) -> None:
        row = self.selected_row()
        if row is not None and self._loaded:
            self.player.pause()
            self.player.seek_to_time(row.start if row.start_known and row.start else 0.0)

    def clear_cell(self) -> bool:
        """Clear the current start / end / duration cell (back to the protocol)."""
        index = self.table.currentIndex()
        if not index.isValid() or index.column() not in (COL_START, COL_END, COL_DURATION):
            return False
        if not self.model.flags(index) & Qt.ItemFlag.ItemIsEditable:
            return False
        return self.model.setData(index, "", Qt.ItemDataRole.EditRole)

    def _say(self, text: str) -> None:
        if self.workspace is not None:
            self.workspace.show_message(text)

    # ------------------------------------------------------------------
    # Going through the cohort
    # ------------------------------------------------------------------
    def rows_needing_start(self) -> List[TrialRow]:
        return [r for r in self.model.rows() if not r.is_header and r.needs_attention]

    def next_needing_start(self) -> bool:
        """Select the next row after the current one that is not ready (wrapping
        around). False when there is none."""
        rows = self.model.rows()
        current = self.table.currentIndex().row()
        count = len(rows)
        for step in range(1, count + 1):
            index = (current + step) % count if count else 0
            row = rows[index]
            if row.is_header or not row.needs_attention:
                continue
            if index == current:
                break
            self._select_index(index, load=True)
            return True
        waiting = self.rows_needing_start()
        if not waiting:
            self._say("Every video has valid trial times")
        else:
            self._say("This is the only video that needs attention")
        return False

    # ------------------------------------------------------------------
    # The protocol
    # ------------------------------------------------------------------
    def _on_protocol_changed(self, duration_s, start_mode: str) -> None:
        project = self.project
        if project is None or self.is_read_only():
            return
        try:
            project.set_trial_protocol(duration_s, start_mode)
        except ProjectError as exc:
            self.protocol._show_duration_error(str(exc))

    def set_protocol(self, duration_s: Optional[float], start_mode: str) -> List[str]:
        """Set the project's protocol; returns the videos whose times changed."""
        project = self.project
        if project is None:
            return []
        return project.set_trial_protocol(duration_s, start_mode)

    def protocol_overwrites(self) -> List[str]:
        """Videos that "Apply to all" would change by dropping times the user set."""
        project = self.project
        if project is None:
            return []
        reset_start = project.trial_protocol.start_mode == START_VIDEO
        found = []
        for entry in project.videos:
            trial = entry.trial
            if trial is None:
                continue
            if trial.end_time is not None or (reset_start and trial.start_set):
                found.append(entry.key)
        return found

    def apply_protocol_to_all(self) -> Optional[List[str]]:
        """Make every video follow the protocol again (asks first when that
        drops times the user set). None when the user said no."""
        project = self.project
        if project is None or self.is_read_only():
            return None
        self.protocol.flush()
        overwritten = self.protocol_overwrites()
        if overwritten:
            reset_start = project.trial_protocol.start_mode == START_VIDEO
            what = "ends and starts" if reset_start else "ends"
            prompts = self.workspace.prompts
            if not prompts.confirm(
                "Apply the protocol to all videos?",
                f"{plural(len(overwritten), 'video')} {'has' if len(overwritten) == 1 else 'have'} "
                f"{what} you set yourself. Applying the protocol removes them, so every "
                "trial follows the protocol again.",
                accept="Apply to all", destructive=True,
            ):
                return None
        changed = project.apply_protocol_to_all()
        self._say(f"{plural(len(changed), 'video')} now follow the protocol")
        return changed

    def _normalize_protocol(self) -> None:
        """One window per mouse needs no per-video start: keep the protocol at
        "at the start of each video" for that test (once per project)."""
        project = self.project
        if project is None or self._tst_checked or self.is_read_only():
            return
        if not stage_status.paradigm_known(project):
            return
        self._tst_checked = True
        protocol = project.trial_protocol
        if self.per_mouse() and protocol.start_mode != START_VIDEO:
            project.set_trial_protocol(protocol.duration_s, START_VIDEO)

    # ------------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------------
    def handle_key(self, event: QKeyEvent) -> bool:
        """The keyboard flow: S / E set the start / end, Enter or N go to the
        next video needing a start, Left / Right step frames, Space plays,
        Delete clears a cell."""
        if self.pages.currentIndex() != PAGE_MAIN or self.project is None:
            return False
        modifiers = event.modifiers()
        if modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
                        | Qt.KeyboardModifier.MetaModifier):
            return False
        key = event.key()
        if key == Qt.Key.Key_S:
            self.set_start_here()
            return True
        if key == Qt.Key.Key_E:
            self.set_end_here()
            return True
        if key in (Qt.Key.Key_N, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.next_needing_start()
            return True
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Space):
            if not self._loaded:
                return False
            return self.player.handle_key_event(event)
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            return self.clear_cell()
        return False

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    def counts(self) -> Tuple[int, int, int, int]:
        """``(videos, ready, need a start, invalid)``."""
        project = self.project
        if project is None:
            return 0, 0, 0, 0
        states = [project.trial_status(v) for v in project.videos]
        return (len(states), states.count(TRIAL_OK),
                states.count(TRIAL_NEEDS_START), states.count(TRIAL_INVALID))

    def _update_summary(self, rows: List[TrialRow]) -> None:
        total, ready, needs, invalid = self.counts()
        self.summary_label.setText(self._summary_text(total, ready, needs, invalid))
        waiting = needs + invalid
        self.next_button.setEnabled(waiting > 0)
        project = self.project
        if waiting:
            self.next_button.setToolTip("Select the next video that needs a start or has invalid times (Enter)")
        elif project is not None and project.trial_protocol.start_mode == START_VIDEO:
            self.next_button.setToolTip(
                "Every trial starts with its video, so none needs a start. "
                "Choose “Set per video” in the protocol to give each its own.")
        else:
            self.next_button.setToolTip("Every video has valid trial times")

    @staticmethod
    def _summary_text(total: int, ready: int, needs: int, invalid: int) -> str:
        if not total:
            return ""
        parts = []
        if ready:
            parts.append(f"{ready} ready")
        if needs:
            parts.append(f"{needs} {'needs' if needs == 1 else 'need'} a start")
        if invalid:
            parts.append(f"{invalid} with invalid times")
        if ready == total:
            return f"All {plural(total, 'video')} {'is' if total == 1 else 'are'} ready"
        return " · ".join(parts)

    def _sync_extras(self) -> None:
        total, ready, needs, invalid = self.counts()
        visible = (self._active and self.pages.currentIndex() == PAGE_MAIN
                   and needs + invalid > 0)
        self.show_next_button.setVisible(bool(visible))
        # One window per mouse has no "start" to wait for: the button only
        # appears when a window is invalid.
        self.next_button.setVisible(
            self.pages.currentIndex() != PAGE_MAIN or not self.per_mouse()
            or needs + invalid > 0)

    def next_label(self) -> Optional[str]:
        return NEXT_LABEL

    def can_advance(self) -> Tuple[bool, str]:
        project = self.project
        if project is None:
            return False, stage_status.NEEDS_PROJECT
        if not stage_status.paradigm_known(project):
            return False, stage_status.NEEDS_PARADIGM
        if not project.videos:
            return False, stage_status.NEEDS_VIDEO
        total, ready, needs, invalid = self.counts()
        reasons = []
        if needs:
            reasons.append(f"{needs} of {plural(total, 'video')} {'needs' if needs == 1 else 'need'} a start time")
        if invalid:
            reasons.append(
                f"{invalid} of {plural(total, 'video')} {'has' if invalid == 1 else 'have'} "
                "times that cannot be used")
        if reasons:
            return False, "; ".join(reasons)
        return True, ""

    def footer_text(self):
        project = self.project
        if project is None or not project.videos:
            return "", None
        total, ready, needs, invalid = self.counts()
        text = self._summary_text(total, ready, needs, invalid)
        if needs or invalid:
            return text, ("warning", "Needs attention")
        return text, ("ok", "Done")


__all__ = ["ProtocolPanel", "TrialsStage", "PAGE_EMPTY", "PAGE_MAIN"]
