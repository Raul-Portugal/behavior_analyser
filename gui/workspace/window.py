"""
gui/workspace/window.py
``WorkspaceWindow``: the 4.0 single-window workspace, a guided flow of six
stages::

    +----------------------------------------------------------------+
    | app bar:  Behavioral Maze Analyzer | project name  [Saved]   ... |
    +----------------------------------------------------------------+
    | stepper:  1 Videos > 2 Arena & zones > 3 Detection > 4 Trials   |
    |           > 5 Review & run > 6 Results                          |
    +----------------------------------------------------------------+
    |                                                                |
    |   the current stage, filling the whole area        [details    |
    |   (see gui.workspace.stages)                        drawer,    |
    |                                                     hidden by  |
    |                                                     default]   |
    +----------------------------------------------------------------+
    | footer:  < Back      status              Next: <stage> >       |
    +----------------------------------------------------------------+

The stepper shows each stage's status (todo, current, done, needs
attention); every stage is clickable, so the flow is not linear. Stages are
:class:`~gui.workspace.stages.base.Stage` objects; the window calls their
hooks and draws the shared footer from them. The details of the selected
video live in a drawer that slides over the right edge (Ctrl+I, or the
"Details" button of a video row).

The window owns the *project session*: the open ``Project``, its autosaver,
the unsaved-changes flag and everything around them (new / open / recent /
save / save as / recovery / imports / export / close). It does not analyse
anything; the Run stage does, and talks to the window through the signals
and methods below.

For the stages and the old tabs (see also :mod:`gui.workspace.tabs`)
--------------------------------------------------------------------
Signals
    ``project_changed(project)``  a project was opened or closed (``None``);
    ``project_modified()``        the open project was edited (by anyone);
    ``selection_changed(keys)``   the table selection changed;
    ``stage_changed(stage_id)``   another stage became current;
    ``run_requested()``           the Run action was triggered;
    ``cancel_requested()``        the Cancel action was triggered.
State
    ``window.project`` (read-only property), ``selected_keys()``,
    ``selected_entries()``, ``video_panel``, ``context_panel``, ``prompts``,
    ``app_settings``, ``current_stage_id``, ``stage(stage_id)``,
    ``thumbnails`` (one shared ``ThumbnailLoader``).
Navigation
    ``go_to_stage(stage_id)``, ``next_stage()``, ``previous_stage()``,
    ``refresh_navigation()`` (stepper), ``refresh_footer()``,
    ``show_details(keys)``, ``show_message(text)`` (a sentence in the footer).
Control
    ``install_tab(name, widget)``   replace the Setup / Run / Results tab
                                    hosted by the temporary stages;
    ``show_tab(name)``              go to the stage that hosts that tab;
    ``set_run_enabled(bool)``       enable the Run action;
    ``set_busy(bool, reason)``      a run is in progress: lock project
                                    editing, enable Cancel, block closing
                                    the project;
    ``register_close_guard(fn)``    ``fn() -> bool`` may veto closing;
    ``save()``, ``statusBar()``;
    ``tools_menu`` / ``validation_menu`` / ``help_menu`` for extra actions.
"""
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PyQt6.QtCore import QEvent, QTimer, QUrl, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QKeySequence, QPainter
from PyQt6.QtWidgets import (QMainWindow, QMenu, QStackedWidget, QVBoxLayout,
                             QHBoxLayout, QWidget)

from core.models import BatchSettings
from core.project import (PROJECT_EXTENSION, Project, ProjectError,
                          ProjectIssue, VideoEntry)
from core.project_io import (ProjectAutosaver, autosave_path_for,
                             export_batch_plan, load_project,
                             project_from_batch_plan, project_from_run_folder,
                             project_from_template, recovery_available,
                             save_project, with_project_extension)
from core.version import APP_NAME, APP_VERSION
from gui.app_settings import AppSettings
from gui.video_list import is_video_file
from gui.workspace.context_panel import (ACTION_APPLY_TEMPLATE, ACTION_RELINK,
                                         ACTION_REMOVE, ACTION_REVEAL,
                                         ACTION_UNVERIFY, ACTION_VERIFY,
                                         ContextPanel, DetailsDrawer)
from gui.workspace.prompts import (DISCARD, OPEN_FILTER, PROJECT_FILTER,
                                   SAVE, SOURCE_FILTER, Prompts)
from gui.workspace.setup_tab import SetupTab  # Setup tab (X2b-2)
from gui.workspace.stages import status as stage_status
from gui.workspace.stages.arena_stage import ArenaStage
from gui.workspace.stages.base import Stage
from gui.workspace.stages.hosted import (DetectionStage, HostedStage,
                                         ResultsStage, ReviewStage, TrialsStage)
from gui.workspace.stages.videos_stage import VideosStage
from gui.workspace.tabs import (TAB_NAMES, TAB_RESULTS, TAB_RUN, TAB_SETUP,
                                PlaceholderTab, WorkspaceTab)
from gui.workspace.theme import Spacing, current_palette
from gui.workspace.thumbnails import ThumbnailLoader
from gui.workspace.ui_kit import (FooterBar, GhostButton, Label, StatusChip,
                                  StepInfo, Stepper, _IconBadge)

logger = logging.getLogger(__name__)

TITLE_SEPARATOR = "—"
QUICKSTART_URL = (
    "https://github.com/Raul-Portugal/behavior_analyser/blob/main/quickstart.md"
)
DEFAULT_SIZE = (1280, 680)
MINIMUM_SIZE = (960, 600)
MESSAGE_MS = 8000
APP_BAR_HEIGHT = 48

IMPORT_TEMPLATE = "template"
IMPORT_PLAN = "batch_plan"
IMPORT_RUN = "run_folder"
_IMPORT_LABELS = {
    IMPORT_TEMPLATE: "a saved template",
    IMPORT_PLAN: "a batch plan",
    IMPORT_RUN: "a previous run folder",
}

# The old tabs live inside the stages that host them (see stages/hosted.py).
_TAB_STAGES = {
    TAB_SETUP: stage_status.STAGE_ARENA,
    TAB_RUN: stage_status.STAGE_REVIEW,
    TAB_RESULTS: stage_status.STAGE_RESULTS,
}
_STAGE_TABS = {
    stage_status.STAGE_ARENA: TAB_SETUP,
    stage_status.STAGE_DETECTION: TAB_SETUP,
    stage_status.STAGE_TRIALS: TAB_SETUP,
    stage_status.STAGE_REVIEW: TAB_RUN,
    stage_status.STAGE_RESULTS: TAB_RESULTS,
}

# Classic windows opened by "Switch to classic interface" must outlive the
# workspace window that opened them.
_open_classic_windows: List[QWidget] = []


def quickstart_target() -> str:
    """The quickstart guide: the local file when shipped, else the web page."""
    local = Path(__file__).resolve().parents[2] / "quickstart.md"
    if local.is_file():
        return QUrl.fromLocalFile(str(local)).toString()
    return QUICKSTART_URL


class _Surface(QWidget):
    """A bar that paints the theme's surface colour and one hairline border."""

    def __init__(self, border: str = "bottom", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._border = border

    def paintEvent(self, event) -> None:
        palette = current_palette()
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(palette.surface))
        line = QColor(palette.border)
        if self._border == "bottom":
            painter.fillRect(0, self.height() - 1, self.width(), 1, line)
        elif self._border == "top":
            painter.fillRect(0, 0, self.width(), 1, line)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange):
            self.update()


class AppBar(_Surface):
    """The slim bar on top: the app name, the project and its save state."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("bottom", parent)
        self.setFixedHeight(APP_BAR_HEIGHT)
        self.logo = _IconBadge("maze-field", 28, shape="rounded")
        self.app_label = Label(APP_NAME, "subtitle")
        self.divider = Label("/", "muted")
        self.project_label = Label("", "body")
        self.save_chip = StatusChip("neutral", "")
        self.more_button = GhostButton("···")
        self.more_button.setFixedSize(44, 32)
        self.more_button.setStyleSheet("QPushButton { padding: 0px; font-size: 22px; font-weight: 700; }")
        self.more_button.setToolTip("More actions")
        self.more_button.setAccessibleName("More actions")
        row = QHBoxLayout(self)
        row.setContentsMargins(Spacing.LG, 0, Spacing.SM, 0)
        row.setSpacing(Spacing.SM)
        row.addWidget(self.logo)
        row.addWidget(self.app_label)
        row.addWidget(self.divider)
        row.addWidget(self.project_label)
        row.addWidget(self.save_chip)
        row.addStretch(1)
        row.addWidget(self.more_button)
        self.set_project(None, False)

    def set_project(self, project: Optional[Project], unsaved: bool) -> None:
        has = project is not None
        self.divider.setVisible(has)
        self.project_label.setVisible(has)
        self.save_chip.setVisible(has)
        if has:
            self.project_label.setText(project.name)
            self.project_label.setToolTip(
                str(project.file_path) if project.file_path else project.name
            )
            if unsaved:
                self.save_chip.set_status("warning", "Unsaved changes")
            else:
                self.save_chip.set_status("ok", "Saved")


class WorkspaceWindow(QMainWindow):
    """Single-window workspace: app bar, stepper, stages, footer, drawer."""

    project_changed = pyqtSignal(object)    # Project or None
    project_modified = pyqtSignal()
    selection_changed = pyqtSignal(list)    # keys of the selected videos
    stage_changed = pyqtSignal(str)         # id of the stage now showing
    run_requested = pyqtSignal()
    cancel_requested = pyqtSignal()

    AUTOSAVE_INTERVAL_MS = 1000

    def __init__(
        self,
        app_settings: Optional[AppSettings] = None,
        prompts: Optional[Prompts] = None,
    ):
        super().__init__()
        self.app_settings = app_settings or AppSettings()
        self.prompts = prompts or Prompts(self)
        self.prompts.parent = self
        self._project: Optional[Project] = None
        self._autosaver: Optional[ProjectAutosaver] = None
        self._unsaved = False
        self._busy = False
        self._run_enabled = False
        self._close_guards: List[Callable[[], bool]] = []
        self._skip_close_prompt = False
        self._closed = False
        self._issues_cache: Optional[List[ProjectIssue]] = None
        self._current_stage = stage_status.STAGE_VIDEOS
        self._stepper_signature: Optional[Tuple] = None
        self._message_active = False

        self.setAcceptDrops(True)
        self.resize(*DEFAULT_SIZE)
        self.setMinimumSize(*MINIMUM_SIZE)
        self._build_ui()
        self._build_actions()
        self._build_menus()
        self._build_shortcuts()
        self.install_tab(TAB_SETUP, SetupTab())  # Setup tab (X2b-2); also enables align_action
        self._restore_layout()

        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(self.AUTOSAVE_INTERVAL_MS)
        self._autosave_timer.timeout.connect(self._poll_autosave)
        self._autosave_timer.start()
        self._start_timer = QTimer(self)
        self._start_timer.setSingleShot(True)
        self._start_timer.setInterval(0)
        self._pending_start: Optional[Path] = None
        self._start_timer.timeout.connect(self._run_scheduled_start)
        # Coalesces the stepper refreshes one edit causes.
        self._nav_timer = QTimer(self)
        self._nav_timer.setSingleShot(True)
        self._nav_timer.setInterval(0)
        self._nav_timer.timeout.connect(self.refresh_navigation)
        self._message_timer = QTimer(self)
        self._message_timer.setSingleShot(True)
        self._message_timer.setInterval(MESSAGE_MS)
        self._message_timer.timeout.connect(self._message_done)

        self._update_title()
        self._update_ui_state()

        # --- Run and Results tabs (workstream X2b-3) -------------------
        from gui.workspace.results_tab import ResultsTab
        from gui.workspace.run_tab import RunTab
        self.install_tab(TAB_RESULTS, ResultsTab())
        self.install_tab(TAB_RUN, RunTab())
        # ---------------------------------------------------------------
        for stage in self.stages.values():
            stage.attach(self)
        self.refresh_navigation()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.thumbnails = ThumbnailLoader(self)
        self.videos_stage = VideosStage(
            self.app_settings, self.prompts, self.thumbnails
        )
        self.video_panel = self.videos_stage.video_panel
        self.context_panel = ContextPanel(
            self.prompts, self.video_panel.model.issues_for,
            loader=self.thumbnails, chip_for=self.video_panel.proxy.chip_for,
        )
        # Where the old tabs wait while no stage shows them.
        self._tab_parking = QWidget(self)
        self._tab_parking.hide()
        self.setup_tab: QWidget = PlaceholderTab(
            "Setup", "Setup: choose the paradigm and configure the regions "
                     "of interest, detection and calibration."
        )
        self.run_tab: QWidget = PlaceholderTab(
            "Run", "Run: start the analysis and follow its progress."
        )
        self.results_tab: QWidget = PlaceholderTab(
            "Results", "Results: open and review the output of earlier runs."
        )
        self._tab_widgets: Dict[str, QWidget] = {
            TAB_SETUP: self.setup_tab, TAB_RUN: self.run_tab,
            TAB_RESULTS: self.results_tab,
        }
        for widget in self._tab_widgets.values():
            widget.setParent(self._tab_parking)

        self.stages: Dict[str, Stage] = self._make_stages()
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.app_bar = AppBar()
        outer.addWidget(self.app_bar)

        self.stepper = Stepper(self._stepper_steps())
        holder = _Surface("bottom")
        holder_layout = QHBoxLayout(holder)
        holder_layout.setContentsMargins(Spacing.LG, Spacing.XS, Spacing.LG, Spacing.XS)
        holder_layout.addWidget(self.stepper)
        outer.addWidget(holder)
        self.stepper_holder = holder

        self.stack = QStackedWidget()
        for stage in self.stages.values():
            self.stack.addWidget(stage)
        # The drawer slides over this area, so the stages keep all of it.
        self.body = QWidget()
        body_layout = QVBoxLayout(self.body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.addWidget(self.stack)
        outer.addWidget(self.body, 1)
        self.drawer = DetailsDrawer(self.context_panel, self.body)

        self.footer = FooterBar()
        outer.addWidget(self.footer)
        self.setCentralWidget(central)

        self.stack.setCurrentWidget(self.videos_stage)
        self.stepper.set_current(stage_status.STAGE_VIDEOS)

        self.stepper.step_clicked.connect(self._on_step_clicked)
        self.footer.back_clicked.connect(self.previous_stage)
        self.footer.next_clicked.connect(self.next_stage)
        self.video_panel.selection_changed.connect(self._on_selection_changed)
        self.video_panel.paths_dropped.connect(self.handle_dropped_paths)
        self.video_panel.status_message.connect(self.show_message)
        self.video_panel.details_requested.connect(self.show_details)
        self.video_panel.model.projectChanged.connect(self._on_model_changed)
        self.context_panel.action_requested.connect(self._on_context_action)
        self.statusBar().hide()      # messages show in the footer instead

    def _make_stages(self) -> Dict[str, Stage]:
        """The stages in flow order.

        To replace a hosted stage by a real one, put the new class here
        (see :mod:`gui.workspace.stages`).
        """
        ordered = [
            self.videos_stage, ArenaStage(self.thumbnails), DetectionStage(), TrialsStage(),
            ReviewStage(), ResultsStage(),
        ]
        assert [s.stage_id for s in ordered] == list(stage_status.STAGE_ORDER)
        return {stage.stage_id: stage for stage in ordered}

    def _stepper_steps(self) -> List[StepInfo]:
        return [
            StepInfo(stage.stage_id, stage.title, stage.icon_name, "todo", "")
            for stage in self.stages.values()
        ]

    def _build_actions(self) -> None:
        def make(text, slot, shortcut=None, tip="", enabled=True):
            action = QAction(text, self)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
            if tip:
                action.setToolTip(tip)
                action.setStatusTip(tip)
            action.triggered.connect(lambda _checked=False: slot())
            action.setEnabled(enabled)
            return action

        self.new_action = make("&New project...", self.new_project, "Ctrl+N",
                               "Create an empty project")
        self.open_action = make("&Open project...", self.open_project, "Ctrl+O",
                                "Open a project, template or batch plan")
        self.save_action = make("&Save", self.save, "Ctrl+S",
                                "Save the project", enabled=False)
        self.save_as_action = make("Save &as...", self.save_as, "Ctrl+Shift+S",
                                   "Save the project under another name",
                                   enabled=False)
        self.import_template_action = make(
            "&Template...", self.import_template,
            tip="Start a project from a saved single-video configuration")
        self.import_plan_action = make(
            "&Batch plan...", self.import_batch_plan,
            tip="Start a project from a 3.x batch plan")
        self.import_run_action = make(
            "&Previous run folder...", self.import_run_folder,
            tip="Open an earlier analysis run as a project")
        self.export_plan_action = make(
            "&Export batch plan...", self.export_plan,
            tip="Write the configured videos as a 3.x batch plan",
            enabled=False)
        self.close_project_action = make(
            "&Close project", self.close_project, "Ctrl+W",
            enabled=False)
        self.classic_action = make(
            "Switch to &classic interface", self.switch_to_classic,
            tip="Open the classic wizard window instead")
        self.quit_action = make("&Quit", self.close, "Ctrl+Q")

        self.add_videos_action = make(
            "Add &videos...", self.video_panel.add_videos_dialog, "Ctrl+Shift+V",
            "Add video files to the project", enabled=False)
        self.add_folder_action = make(
            "Add &folder...", self.video_panel.add_folder_dialog, "Ctrl+Shift+O",
            "Add all the videos of a folder", enabled=False)
        self.remove_action = make(
            "&Remove", self.video_panel.remove_selected, None,
            "Remove the selected videos from the project (Del)", enabled=False)
        self.run_action = make("&Run analysis", self.run_requested.emit, None,
                               "Run the analysis", enabled=False)
        self.cancel_action = make("&Cancel run", self.cancel_requested.emit, None,
                                  "Stop the analysis", enabled=False)

        self.next_stage_action = make(
            "&Next stage", self.next_stage, "Ctrl+PgDown",
            "Go to the next stage")
        self.previous_stage_action = make(
            "&Previous stage", self.previous_stage, "Ctrl+PgUp",
            "Go back to the previous stage")
        self.review_action = make(
            "Go to Review && &run", self.go_to_review, "F5",
            "Review the study and run the analysis")
        self.details_action = make(
            "Video &details", self.toggle_details, "Ctrl+I",
            "Show or hide the details of the selected video")

        self.manual_scoring_action = make(
            "&Score a video manually...", self.open_manual_scoring,
            tip="Blind manual scoring with hotkeys, saved as annotation "
                "files for the validation report")
        self.position_spotcheck_action = make(
            "Tracking s&pot-check...", self.open_position_spotcheck,
            tip="Click the animal in randomly sampled frames to measure the "
                "tracker's position error")
        self.validation_report_action = make(
            "&Build validation report...", self.open_validation_report,
            tip="Compare automated results with manual scoring: agreement "
                "statistics, plots and acceptance criteria in one HTML report")
        self.align_action = make(
            "&Align template to videos...", lambda: None,
            tip="Carry the template's regions onto videos whose camera "
                "moved. Available from the Arena & zones stage.", enabled=False)
        self.quickstart_action = make("&Quickstart guide", self.open_quickstart)
        self.about_action = make(f"&About {APP_NAME}", self.show_about)

    def _build_menus(self) -> None:
        bar = self.menuBar()
        self.file_menu = bar.addMenu("&File")
        self.file_menu.addAction(self.new_action)
        self.file_menu.addAction(self.open_action)
        self.recent_menu = self.file_menu.addMenu("Open &recent")
        self.recent_menu.aboutToShow.connect(self._rebuild_recent_menu)
        self._rebuild_recent_menu()
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.save_action)
        self.file_menu.addAction(self.save_as_action)
        self.file_menu.addSeparator()
        self.import_menu = self.file_menu.addMenu("&Import")
        self.import_menu.addAction(self.import_template_action)
        self.import_menu.addAction(self.import_plan_action)
        self.import_menu.addAction(self.import_run_action)
        self.file_menu.addAction(self.export_plan_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.close_project_action)
        self.file_menu.addAction(self.classic_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.quit_action)

        self.tools_menu = bar.addMenu("&Tools")
        self.tools_menu.addAction(self.align_action)
        self.tools_menu.addSeparator()
        self.tools_menu.addAction(self.review_action)
        self.tools_menu.addAction(self.next_stage_action)
        self.tools_menu.addAction(self.previous_stage_action)
        self.tools_menu.addAction(self.details_action)
        self.tools_menu.addSeparator()
        self.tools_menu.addAction(self.run_action)
        self.tools_menu.addAction(self.cancel_action)

        self.validation_menu = bar.addMenu("&Validation")
        self.validation_menu.addAction(self.manual_scoring_action)
        self.validation_menu.addAction(self.position_spotcheck_action)
        self.validation_menu.addAction(self.validation_report_action)

        self.help_menu = bar.addMenu("&Help")
        self.help_menu.addAction(self.quickstart_action)
        self.help_menu.addAction(self.about_action)

        # The app bar's "⋯" button: the everyday actions in one place.
        self.more_menu = QMenu(self)
        for action in (self.new_action, self.open_action, self.save_action,
                       self.save_as_action):
            self.more_menu.addAction(action)
        self.more_menu.addSeparator()
        self.more_menu.addAction(self.details_action)
        self.more_menu.addAction(self.classic_action)
        self.more_menu.addSeparator()
        self.more_menu.addAction(self.quickstart_action)
        self.more_menu.addAction(self.about_action)
        self.app_bar.more_button.clicked.connect(
            lambda _checked=False: self.show_more_menu()
        )

    def _build_shortcuts(self) -> None:
        """Shortcuts work wherever the focus is inside the window."""
        for action in (self.next_stage_action, self.previous_stage_action,
                       self.review_action, self.details_action):
            action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            self.addAction(action)

    def show_more_menu(self) -> None:
        """Pop the "⋯" menu up under its button (does not block)."""
        button = self.app_bar.more_button
        point = button.mapToGlobal(button.rect().bottomRight())
        point.setX(point.x() - self.more_menu.sizeHint().width())
        self.more_menu.popup(point)

    # ------------------------------------------------------------------
    # Public state for the stages and tabs
    # ------------------------------------------------------------------
    @property
    def project(self) -> Optional[Project]:
        return self._project

    @property
    def has_unsaved_changes(self) -> bool:
        return self._project is not None and self._unsaved

    @property
    def is_busy(self) -> bool:
        return self._busy

    @property
    def autosaver(self) -> Optional[ProjectAutosaver]:
        return self._autosaver

    @property
    def welcome(self):
        """The no-project page of the Videos stage (recent projects, import)."""
        return self.videos_stage.welcome

    def selected_keys(self) -> List[str]:
        return self.video_panel.selected_keys()

    def selected_entries(self) -> List[VideoEntry]:
        project = self._project
        if project is None:
            return []
        keys = self.selected_keys()
        return project.entries_for(keys) if keys else []

    def project_issues(self) -> List[ProjectIssue]:
        """``project.validate()``, computed once until something changes."""
        project = self._project
        if project is None:
            return []
        if self._issues_cache is None:
            self._issues_cache = list(project.validate())
        return list(self._issues_cache)

    def invalidate_issues(self) -> None:
        self._issues_cache = None

    # -- the old tabs ------------------------------------------------------
    def tab_widget(self, name: str) -> QWidget:
        return self._tab_widgets[name]

    def install_tab(self, name: str, widget: QWidget) -> QWidget:
        """Replace the Setup, Run or Results tab; returns the old widget.

        The tab is hosted by the temporary stages (Setup by Arena & zones,
        Detection and Trials, Run by Review & run, Results by Results). A
        :class:`WorkspaceTab` is attached to the window (it then receives
        the project and selection hooks); any other widget is just shown.
        """
        if name not in TAB_NAMES:
            raise ValueError(f"Unknown tab {name!r}; use one of {TAB_NAMES}")
        old = self._tab_widgets[name]
        self._tab_widgets[name] = widget
        setattr(self, f"{name}_tab", widget)
        widget.setParent(self._tab_parking)
        if isinstance(widget, WorkspaceTab):
            widget.attach_workspace(self)
        for stage in self.stages.values():
            if isinstance(stage, HostedStage) and stage.tab_name == name:
                stage.on_tab_replaced(widget)
        old.setParent(None)
        old.deleteLater()
        return old

    def show_tab(self, name: str) -> None:
        """Go to the stage that hosts the Setup, Run or Results tab."""
        self.go_to_stage(_TAB_STAGES[name])

    def current_tab_name(self) -> str:
        """The old tab the current stage shows (``videos`` for stage 1)."""
        return _STAGE_TABS.get(self._current_stage, self._current_stage)

    def set_run_enabled(self, enabled: bool) -> None:
        """The Run stage calls this when a run can be started."""
        self._run_enabled = bool(enabled)
        self._update_ui_state()

    def set_busy(self, busy: bool, reason: str = "") -> None:
        """A run is in progress (or over): lock or unlock project editing."""
        self._busy = bool(busy)
        self.video_panel.set_read_only(self._busy)
        self.videos_stage.set_busy(self._busy)
        self._update_ui_state()
        if busy and reason:
            self.show_message(reason)

    def register_close_guard(self, guard: Callable[[], bool]) -> None:
        """``guard()`` returns False to veto closing the window or project.

        The guard explains the veto itself (it usually asks the user).
        """
        self._close_guards.append(guard)

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------
    @property
    def current_stage_id(self) -> str:
        return self._current_stage

    @property
    def current_stage(self) -> Stage:
        return self.stages[self._current_stage]

    def stage(self, stage_id: str) -> Stage:
        return self.stages[stage_id]

    def stage_ids(self) -> List[str]:
        return list(self.stages)

    def stage_available(self, stage_id: str) -> Tuple[bool, str]:
        """Whether the user can go to ``stage_id`` now, and why not."""
        status, reason = self.stages[stage_id].status()
        return status != stage_status.STATUS_DISABLED, reason

    def go_to_stage(self, stage_id: str, force: bool = False) -> bool:
        """Show ``stage_id``. False when the current stage vetoed leaving it
        (it asked the user) or the stage is not available yet.

        ``force`` skips both checks (used when the project changes).
        """
        if stage_id not in self.stages:
            raise KeyError(stage_id)
        if stage_id == self._current_stage:
            return True
        self.invalidate_issues()
        if not force:
            if not self.stage_available(stage_id)[0]:
                return False
            if not self.current_stage.on_leave():
                return False
        self._current_stage = stage_id
        stage = self.stages[stage_id]
        self.stack.setCurrentWidget(stage)
        self.refresh_navigation()
        stage.on_enter()
        self.refresh_navigation()
        self._remember_stage()
        self.stage_changed.emit(stage_id)
        return True

    def next_stage(self) -> bool:
        """Go to the following stage if the current one allows it."""
        order = self.stage_ids()
        index = order.index(self._current_stage)
        if index + 1 >= len(order):
            return False
        allowed, reason = self.current_stage.can_advance()
        if not allowed:
            if reason:
                self.show_message(reason)
            return False
        return self.go_to_stage(order[index + 1])

    def previous_stage(self) -> bool:
        order = self.stage_ids()
        index = order.index(self._current_stage)
        if index == 0:
            return False
        return self.go_to_stage(order[index - 1])

    def go_to_review(self) -> bool:
        """F5: the place where the analysis is started."""
        return self.go_to_stage(stage_status.STAGE_REVIEW)

    def _on_step_clicked(self, stage_id: str) -> None:
        self.go_to_stage(stage_id)

    def _remember_stage(self) -> None:
        project = self._project
        if project is not None and project.file_path is not None:
            self.app_settings.set_workspace_stage(
                project.file_path, self._current_stage
            )

    def _stage_for_project(self) -> str:
        """Where to land after a project was installed: where the user left
        off in it, else the first stage."""
        project = self._project
        if project is None or project.file_path is None:
            return stage_status.STAGE_VIDEOS
        saved = self.app_settings.workspace_stage(project.file_path)
        if saved in self.stages and self.stage_available(saved)[0]:
            return saved
        return stage_status.STAGE_VIDEOS

    # -- the stepper and the footer ---------------------------------------------
    def schedule_navigation_refresh(self) -> None:
        """Refresh the stepper and footer once the current burst of
        notifications is over."""
        if not self._closed:
            self._nav_timer.start()

    def refresh_navigation(self) -> None:
        """Redraw the stepper (statuses, titles) and the footer."""
        self._nav_timer.stop()
        if self._closed:
            return
        self.invalidate_issues()       # files may have changed on disk
        infos: List[StepInfo] = []
        for stage_id, stage in self.stages.items():
            status, reason = stage.status()
            if stage_id == self._current_stage:
                status = "current"
            infos.append(StepInfo(stage_id, stage.title, stage.icon_name, status, reason))
        signature = tuple((i.id, i.label, i.icon) for i in infos)
        if signature != self._stepper_signature:
            self._stepper_signature = signature
            self.stepper.set_steps(infos)
        else:
            for info in infos:
                self.stepper.set_status(info.id, info.status, info.reason)
        self.refresh_footer()

    def refresh_footer(self) -> None:
        """Draw the footer for the current stage."""
        if self._closed:
            return
        order = self.stage_ids()
        index = order.index(self._current_stage)
        stage = self.current_stage
        following = self.stages[order[index + 1]] if index + 1 < len(order) else None
        stage.next_stage_title = None if following is None else following.title
        label = stage.next_label()
        allowed, reason = stage.can_advance()
        if allowed and following is not None and label is not None:
            allowed, reason = self.stage_available(following.stage_id)
        self.footer.set_next(label, allowed, reason)
        self.footer.set_back(index > 0)
        if not self._message_active:
            text, chip = stage.footer_text()
            self.footer.set_status(text, chip)

    def show_message(self, text: str) -> None:
        """A sentence in the footer for a few seconds (then the stage's own)."""
        if self._closed:
            return
        self.statusBar().showMessage(text, MESSAGE_MS)
        self._message_active = True
        self.footer.set_status(text)
        self._message_timer.start()

    def _message_done(self) -> None:
        self._message_active = False
        self.refresh_footer()

    # -- the details drawer -----------------------------------------------------------
    def show_details(self, keys: Optional[List[str]] = None) -> None:
        """Open the drawer on the details of ``keys`` (default: the selection)."""
        if keys:
            self.video_panel.select_keys(keys)
            self.video_panel.flush_selection()
        self.drawer.open()

    def toggle_details(self) -> None:
        if self.drawer.is_open():
            self.drawer.close_drawer()
        else:
            self.show_details()

    # ------------------------------------------------------------------
    # UI state
    # ------------------------------------------------------------------
    def _update_title(self) -> None:
        project = self._project
        if project is None:
            self.setWindowTitle(APP_NAME)
            self.setWindowModified(False)
        else:
            self.setWindowTitle(f"{project.name}[*] {TITLE_SEPARATOR} {APP_NAME}")
            self.setWindowModified(self._unsaved)
        self.app_bar.set_project(project, self._unsaved)

    def _update_ui_state(self) -> None:
        has_project = self._project is not None
        idle = not self._busy
        editable = has_project and idle
        self.new_action.setEnabled(idle)
        self.open_action.setEnabled(idle)
        self.recent_menu.setEnabled(idle)
        self.import_menu.setEnabled(idle)
        self.save_action.setEnabled(has_project)
        self.save_as_action.setEnabled(has_project)
        self.export_plan_action.setEnabled(has_project)
        self.close_project_action.setEnabled(has_project and idle)
        self.add_videos_action.setEnabled(editable)
        self.add_folder_action.setEnabled(editable)
        self.remove_action.setEnabled(editable and bool(self.selected_keys()))
        self.run_action.setEnabled(has_project and idle and self._run_enabled)
        self.cancel_action.setEnabled(self._busy)
        self.review_action.setEnabled(has_project)
        self.details_action.setEnabled(has_project)

    def _show_status(self, text: str) -> None:
        self.show_message(text)

    def _rebuild_recent_menu(self) -> None:
        self.recent_menu.clear()
        recent = self.app_settings.recent_projects()
        for path in recent:
            action = self.recent_menu.addAction(path.name)
            action.setToolTip(str(path))
            action.setStatusTip(str(path))
            action.triggered.connect(
                lambda _checked=False, p=path: self.open_path(p)
            )
        if not recent:
            self.recent_menu.addAction("(none)").setEnabled(False)
        else:
            self.recent_menu.addSeparator()
            clear = self.recent_menu.addAction("Clear list")
            clear.triggered.connect(self._clear_recent)

    def _clear_recent(self) -> None:
        self.app_settings.clear_recent_projects()
        self._rebuild_recent_menu()
        self._refresh_welcome()

    def _refresh_welcome(self) -> None:
        if self.welcome is not None:
            self.welcome.refresh()

    # ------------------------------------------------------------------
    # Signals from the panels and the project
    # ------------------------------------------------------------------
    def _on_selection_changed(self, keys: List[str]) -> None:
        self.context_panel.set_selection(keys)
        self._update_ui_state()
        self.selection_changed.emit(list(keys))
        for stage in self.stages.values():
            stage.on_selection_changed(list(keys))

    def _on_model_changed(self) -> None:
        self.invalidate_issues()
        self.context_panel.refresh()
        self.schedule_navigation_refresh()

    def _on_project_edited(self) -> None:
        """Project listener: any change marks the project unsaved."""
        if self._closed:
            return
        self._unsaved = True
        self._update_title()    # a rename shows at once; also sets the flag
        self.invalidate_issues()
        self.project_modified.emit()
        for stage in self.stages.values():
            stage.on_project_modified()
        self.schedule_navigation_refresh()

    def _poll_autosave(self) -> None:
        autosaver = self._autosaver
        if autosaver is None or self._closed:
            return
        autosaver.poll()
        if autosaver.last_error:
            self._show_status(f"Autosave failed: {autosaver.last_error}")

    # ------------------------------------------------------------------
    # The project session
    # ------------------------------------------------------------------
    def _install_project(self, project: Optional[Project], unsaved: bool = False) -> None:
        """Make ``project`` (or nothing) the open project."""
        self._detach_project()
        self._project = project
        self._unsaved = bool(unsaved) and project is not None
        if project is not None:
            project.add_listener(self._on_project_edited)
            if project.file_path is not None:
                self._autosaver = ProjectAutosaver(project, project.file_path)
        self.invalidate_issues()
        self.video_panel.set_project(project)
        self.context_panel.set_project(project)
        if project is None:
            self.drawer.close_drawer(animated=False)
        self._update_title()
        self._update_ui_state()
        self.project_changed.emit(project)
        for stage in self.stages.values():
            stage.on_project_changed(project)
        self._switch_after_project_change()

    def _switch_after_project_change(self) -> None:
        """Show the stage the user left off at in this project (stage 1 for a
        new one, or when no project is open)."""
        target = self._stage_for_project()
        if target != self._current_stage:
            self.go_to_stage(target, force=True)
        else:
            self.current_stage.on_enter()
        self.refresh_navigation()

    def _detach_project(self) -> None:
        if self._project is not None:
            self._project.remove_listener(self._on_project_edited)
        if self._autosaver is not None:
            self._autosaver.close()
        self._autosaver = None
        self._project = None

    def _confirm_close_project(self) -> bool:
        """True when the open project may be closed (saved, or discarded)."""
        for guard in list(self._close_guards):
            if not guard():
                return False
        if not self.has_unsaved_changes:
            return True
        answer = self.prompts.ask_save_changes(self._project.name)
        if answer == SAVE:
            return self.save()
        if answer == DISCARD:
            # Keep a recoverable copy rather than silently losing the edits.
            if self._autosaver is not None:
                self._autosaver.flush()
            return True
        return False

    def close_project(self) -> bool:
        if self._project is None:
            return True
        if self._busy or not self._confirm_close_project():
            return False
        self._install_project(None)
        self.app_settings.clear_last_project()
        self.app_settings.sync()
        self._show_status("Project closed")
        return True

    # -- new -------------------------------------------------------------
    def _start_dir(self, kind: str = "config") -> str:
        if self._project is not None and self._project.base_dir is not None:
            return str(self._project.base_dir)
        return self.app_settings.dialog_dir(kind)

    def _ask_new_project_file(self, suggested: str = "") -> Optional[Path]:
        start = self._start_dir()
        if suggested:
            start = str(Path(start or ".") / suggested)
        path = self.prompts.save_file("Create project", start, PROJECT_FILTER)
        return with_project_extension(path) if path is not None else None

    def new_project(self) -> bool:
        """Create an empty project in a file the user chooses."""
        if self._busy:
            return False
        path = self._ask_new_project_file("untitled.bmaproj")
        if path is None or not self._confirm_close_project():
            return False
        return self._create_and_install(Project(name=path.stem), path)

    def _create_and_install(self, project: Project, path: Path) -> bool:
        """Save a brand-new project to ``path`` and make it the open one."""
        try:
            target = save_project(project, path)
        except (OSError, ValueError, TypeError) as exc:
            self.prompts.error("Create project", f"Cannot write {path.name}:\n{exc}")
            return False
        self._install_project(project)
        self._remember(target)
        self._show_status(f"Created {target.name}")
        return True

    def _remember(self, path: Path) -> None:
        self.app_settings.add_recent_project(path)
        self.app_settings.sync()
        self._rebuild_recent_menu()

    # -- open ------------------------------------------------------------
    def open_project(self, path: Optional[Path] = None) -> bool:
        """Open a project, or offer to import a template or batch plan."""
        if self._busy:
            return False
        if path is None:
            path = self.prompts.open_file(
                "Open project", self._start_dir(), OPEN_FILTER
            )
            if path is None:
                return False
        return self.open_path(path)

    def open_path(self, path: os.PathLike) -> bool:
        """Open whatever ``path`` is: a project, or a template / plan to import."""
        path = Path(path)
        if self._busy:
            return False
        if not path.is_file():
            self.prompts.error("Open", f"The file was not found:\n{path}")
            self.app_settings.remove_recent_project(path)
            self._rebuild_recent_menu()
            self._refresh_welcome()
            return False
        if path.suffix.lower() == PROJECT_EXTENSION:
            return self._open_project_file(path)
        kind = BatchSettings.detect_file_type(path)
        if kind == "project":
            return self._open_project_file(path)
        if kind in (IMPORT_TEMPLATE, IMPORT_PLAN):
            if self.prompts.confirm(
                "Import as a new project",
                f"“{path.name}” is {_IMPORT_LABELS[kind]}, not a project.",
                accept="Import",
                informative="Create a new project from it?",
            ):
                return self._import(kind, path)
            return False
        self.prompts.error(
            "Open",
            f"{path.name} is not a project, a template or a batch plan.",
        )
        return False

    def _load_with_recovery(self, path: Path):
        """``(project, recovered)`` or None when cancelled or unreadable."""
        recover = False
        if recovery_available(path):
            try:
                name = load_project(path).name if path.is_file() else path.stem
            except (ProjectError, OSError):
                name = path.stem
            answer = self.prompts.ask_recovery(name, autosave_path_for(path))
            if answer == "cancel":
                return None
            recover = answer == "recover"
        try:
            return load_project(path, recover=recover), recover
        except (ProjectError, OSError, ValueError) as exc:
            self.prompts.error("Open project", f"Cannot open {path.name}:\n{exc}")
            return None

    def _open_project_file(self, path: Path) -> bool:
        loaded = self._load_with_recovery(path)
        if loaded is None:
            return False
        project, recovered = loaded
        if not self._confirm_close_project():
            return False
        self._install_project(project, unsaved=recovered)
        self._remember(path)
        if recovered:
            self._show_status(
                f"Recovered the autosave of {path.name}; save to keep it"
            )
        else:
            self._show_status(f"Opened {path.name}")
        if project.load_warnings:
            shown = "\n".join(project.load_warnings[:10])
            more = len(project.load_warnings) - 10
            self.prompts.warn(
                "Project opened with warnings",
                shown + (f"\n... and {more} more" if more > 0 else ""),
            )
        return True

    # -- save ------------------------------------------------------------
    def save(self) -> bool:
        """Save to the project's file (asks for one if it has none)."""
        project = self._project
        if project is None:
            return False
        if project.file_path is None:
            return self.save_as()
        return self._write(project.file_path)

    def save_as(self) -> bool:
        project = self._project
        if project is None:
            return False
        current = project.file_path
        start = str(current) if current is not None else str(
            Path(self._start_dir() or ".") / with_project_extension(project.name)
        )
        path = self.prompts.save_file("Save project as", start, PROJECT_FILTER)
        if path is None:
            return False
        return self._write(with_project_extension(path))

    def _write(self, path: Path) -> bool:
        project = self._project
        if project is None:
            return False
        self.context_panel.commit_notes()
        self.videos_stage.commit_pending()
        try:
            target = save_project(project, path)
        except (OSError, ValueError, TypeError) as exc:
            self.prompts.error("Save project", f"Cannot save {path.name}:\n{exc}")
            return False
        old = self._autosaver
        if old is not None:
            old.saved()
            if old.path != target:
                old.close()
                old = None
        if old is None:
            self._autosaver = ProjectAutosaver(project, target)
        self._unsaved = False
        self._update_title()
        self.context_panel.refresh()
        self._remember(target)
        self._show_status(f"Saved {target.name}")
        return True

    # -- import / export -------------------------------------------------
    def import_template(self, source: Optional[Path] = None) -> bool:
        return self._import(IMPORT_TEMPLATE, source)

    def import_batch_plan(self, source: Optional[Path] = None) -> bool:
        return self._import(IMPORT_PLAN, source)

    def import_run_folder(self, source: Optional[Path] = None) -> bool:
        return self._import(IMPORT_RUN, source)

    def _import(self, kind: str, source: Optional[Path] = None) -> bool:
        """Build a new project from a template, plan or run folder."""
        if self._busy:
            return False
        if source is None:
            if kind == IMPORT_RUN:
                source = self.prompts.choose_folder(
                    "Select a previous run folder",
                    str(self.app_settings.output_dir() or self._start_dir("output")),
                )
            else:
                source = self.prompts.open_file(
                    "Select a template" if kind == IMPORT_TEMPLATE
                    else "Select a batch plan",
                    self._start_dir(), SOURCE_FILTER,
                )
            if source is None:
                return False
        source = Path(source)
        suggestion = f"{source.name if kind == IMPORT_RUN else source.stem}.bmaproj"
        destination = self.prompts.save_file(
            "Save the new project as",
            str(source.parent / suggestion),
            PROJECT_FILTER,
        )
        if destination is None:
            return False
        destination = with_project_extension(destination)
        try:
            if kind == IMPORT_TEMPLATE:
                project = project_from_template(source, project_file=destination)
            elif kind == IMPORT_PLAN:
                project = project_from_batch_plan(source, project_file=destination)
            else:
                project = project_from_run_folder(source, project_file=destination)
        except (ProjectError, OSError, ValueError, KeyError) as exc:
            self.prompts.error("Import", f"Cannot import {source.name}:\n{exc}")
            return False
        if not self._confirm_close_project():
            return False
        if kind != IMPORT_RUN:
            self.app_settings.add_recent_file(source)
        return self._create_and_install(project, destination)

    def export_plan(self, path: Optional[Path] = None) -> bool:
        """Write the configured videos as a 3.x batch plan."""
        project = self._project
        if project is None:
            return False
        if path is None:
            path = self.prompts.save_file(
                "Export batch plan",
                str(Path(self._start_dir() or ".") / f"{project.name}_plan.json"),
                "Batch plans (*.json)",
            )
            if path is None:
                return False
        try:
            skipped = export_batch_plan(project, path)
        except ProjectError as exc:
            self.prompts.warn("Export batch plan", str(exc))
            return False
        except OSError as exc:
            self.prompts.error("Export batch plan", f"Cannot write the plan:\n{exc}")
            return False
        self.app_settings.add_recent_file(path)
        message = f"Exported {Path(path).name}"
        if skipped:
            message += f" ({len(skipped)} video(s) without settings left out)"
            self.prompts.info(
                "Export batch plan",
                f"{len(skipped)} video(s) have no settings and were left "
                "out of the plan.",
            )
        self._show_status(message)
        return True

    # -- paradigm --------------------------------------------------------
    def change_paradigm(self, name: str) -> bool:
        """Choose the project's paradigm; confirms before clearing settings."""
        project = self._project
        if project is None or self._busy:
            self.videos_stage.refresh()
            return False
        if name == project.effective_paradigm():
            return True
        if project.has_configured_videos():
            count = sum(1 for v in project.videos if v.settings is not None)
            if not self.prompts.confirm(
                "Change paradigm",
                f"Change the paradigm to {name}?",
                accept="Change and clear settings",
                informative=(
                    f"The settings of {count} configured video(s) and the "
                    "project template belong to the current paradigm and will "
                    "be cleared."
                ),
                destructive=True,
            ):
                self.videos_stage.refresh()    # put the picker back
                return False
        project.set_paradigm(name)
        self.app_settings.set_last_paradigm(name)
        return True

    # -- context panel actions ------------------------------------------
    def _on_context_action(self, action: str) -> None:
        keys = self.selected_keys()
        panel = self.video_panel
        if not keys:
            return
        if action == ACTION_VERIFY:
            panel.mark_verified(keys, True)
        elif action == ACTION_UNVERIFY:
            panel.mark_verified(keys, False)
        elif action == ACTION_APPLY_TEMPLATE:
            panel.apply_template(keys)
        elif action == ACTION_REVEAL:
            panel.reveal(keys[0])
        elif action == ACTION_REMOVE:
            panel.remove_keys(keys)
        elif action == ACTION_RELINK:
            missing = [k for k in keys if panel.has_missing([k])]
            if missing:
                panel.relink(missing[0])

    # ------------------------------------------------------------------
    # Drag and drop
    # ------------------------------------------------------------------
    def dragEnterEvent(self, event) -> None:
        if not self._busy and event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        paths = [
            Path(url.toLocalFile()) for url in event.mimeData().urls()
            if url.isLocalFile() and url.toLocalFile()
        ]
        if self._busy or not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        self.handle_dropped_paths(paths)

    def handle_dropped_paths(self, paths: List[Path]) -> None:
        """What a drop means: open a project, import a config, add videos."""
        if self._busy:
            return
        paths = [Path(p) for p in paths]
        projects = [
            p for p in paths if p.suffix.lower() == PROJECT_EXTENSION and p.is_file()
        ]
        configs = [
            p for p in paths if p.suffix.lower() == ".json" and p.is_file()
        ]
        media = [p for p in paths if p.is_dir() or is_video_file(p)]
        ignored = [p for p in paths if p not in projects + configs + media]
        if len(projects) == 1 and not media and not configs:
            self.open_path(projects[0])
            return
        if len(configs) == 1 and not projects and not media:
            self.open_path(configs[0])
            return
        ignored.extend(projects + configs)
        if media:
            if self._project is None and not self._offer_project_for_drop():
                return
            self.video_panel.add_paths(media)
        if ignored and not media:
            names = ", ".join(p.name for p in ignored[:5])
            more = f" and {len(ignored) - 5} more" if len(ignored) > 5 else ""
            self.prompts.info(
                "Nothing To Add",
                "Drop video files (.mp4, .avi, .mov, .mkv), a folder of "
                "videos, a project (.bmaproj) or one saved configuration "
                f"(.json).\n\nIgnored: {names}{more}",
            )
        elif ignored:
            self._show_status(f"Ignored {len(ignored)} dropped item(s)")

    def _offer_project_for_drop(self) -> bool:
        if not self.prompts.confirm(
            "No project open",
            "Create a new project for the dropped videos?",
            accept="Create project",
        ):
            return False
        return self.new_project()

    # ------------------------------------------------------------------
    # Classic interface
    # ------------------------------------------------------------------
    def switch_to_classic(self) -> bool:
        """Open the classic wizard window and close this one."""
        if self._busy or not self._confirm_close_project():
            return False
        from gui.main_window import MainWindow
        classic = MainWindow(self.app_settings)
        classic.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        _open_classic_windows.append(classic)
        classic.destroyed.connect(
            lambda _obj=None, w=classic: _forget_classic(w)
        )
        classic.show()
        self._skip_close_prompt = True
        self.close()
        return True

    # ------------------------------------------------------------------
    # Validation, help
    # ------------------------------------------------------------------
    def open_manual_scoring(self) -> None:
        from gui.manual_scoring_dialog import open_manual_scoring
        open_manual_scoring(self, self.app_settings)

    def open_position_spotcheck(self) -> None:
        from gui.position_spotcheck_dialog import open_position_spotcheck
        open_position_spotcheck(self, self.app_settings)

    def open_validation_report(self) -> None:
        from gui.validation_report_dialog import open_validation_report
        open_validation_report(self, self.app_settings)

    def open_quickstart(self) -> None:
        self.prompts.open_url(quickstart_target())

    def show_about(self) -> None:
        self.prompts.about(
            f"About {APP_NAME}",
            f"<b>{APP_NAME}</b><br>Version {APP_VERSION}<br><br>"
            "Behavioral analysis of Y-maze, elevated plus maze, open field "
            "and tail suspension videos.",
        )

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    def start(self, path: Optional[os.PathLike] = None) -> bool:
        """Open ``path``, else offer to reopen the last project."""
        if path is not None:
            return self.open_path(path)
        last = self.app_settings.last_project()
        if last is not None and self.prompts.confirm(
            "Reopen last project",
            f"Reopen “{last.name}”?",
            accept="Reopen",
            informative=str(last),
        ):
            return self.open_path(last)
        if last is None and self._project is None:
            self.welcome.refresh()     # offer the 3.x import on a first start
        return False

    def schedule_start(self, path: Optional[os.PathLike] = None) -> None:
        """Run :meth:`start` from the event loop (after the window shows)."""
        self._pending_start = Path(path) if path is not None else None
        self._start_timer.start()

    def _run_scheduled_start(self) -> None:
        self.start(self._pending_start)

    # ------------------------------------------------------------------
    # Layout and closing
    # ------------------------------------------------------------------
    def _restore_layout(self) -> None:
        geometry = self.app_settings.workspace_geometry()
        if geometry is not None:
            self.restoreGeometry(geometry)

    def _save_layout(self) -> None:
        # A window that was never shown (e.g. --startup-check) has no layout
        # worth remembering and must not overwrite the user's saved one.
        if not self.isVisible():
            return
        self.app_settings.save_workspace_window(
            self.saveGeometry(), self.saveState()
        )
        self.app_settings.sync()

    def closeEvent(self, event) -> None:
        if not self._skip_close_prompt and not self._confirm_close_project():
            event.ignore()
            return
        self._shutdown()
        event.accept()

    def _shutdown(self) -> None:
        """Flush the autosave, remember the layout and stop all background work."""
        if self._closed:
            return
        self.context_panel.commit_notes()
        self.videos_stage.commit_pending()
        if self._autosaver is not None:
            self._autosaver.flush()
        self._save_layout()
        self._closed = True
        self._autosave_timer.stop()
        self._start_timer.stop()
        self._nav_timer.stop()
        self._message_timer.stop()
        self.drawer.close_drawer(animated=False)
        self.videos_stage.shutdown()
        self.context_panel.shutdown()
        self.thumbnails.shutdown()
        self._detach_project()
        self.video_panel.set_project(None)


def _forget_classic(window: QWidget) -> None:
    try:
        _open_classic_windows.remove(window)
    except ValueError:
        pass


__all__ = ["WorkspaceWindow", "quickstart_target"]
