"""
gui/workspace/window.py
``WorkspaceWindow``: the 4.0 single-window workspace.

::

    +--------------+-------------------------------+----------------+
    | video panel  |  Setup | Run | Results        | context panel  |
    | (table)      |  (tabs, installed by pieces   | (selected      |
    |              |   2 and 3)                    |  video)        |
    +--------------+-------------------------------+----------------+

The window owns the *project session*: the open ``Project``, its autosaver,
the unsaved-changes flag and everything around them (new / open / recent /
save / save as / recovery / imports / export / close). It does not analyse
anything; the Setup and Run tabs do, and talk to the window through the
signals and methods below.

For the tabs (see also :mod:`gui.workspace.tabs`)
-------------------------------------------------
Signals
    ``project_changed(project)``  a project was opened or closed (``None``);
    ``project_modified()``        the open project was edited (by anyone);
    ``selection_changed(keys)``   the table selection changed;
    ``run_requested()``           the Run action was triggered;
    ``cancel_requested()``        the Cancel action was triggered.
State
    ``window.project`` (read-only property), ``selected_keys()``,
    ``selected_entries()``, ``video_panel``, ``context_panel``, ``prompts``,
    ``app_settings``.
Control
    ``install_tab(name, widget)``   replace the Setup / Run / Results tab;
    ``show_tab(name)``;
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
from typing import Callable, List, Optional

from PyQt6.QtCore import QTimer, QUrl, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (QMainWindow, QSplitter, QTabWidget, QToolBar,
                             QWidget)

from core.models import BatchSettings
from core.project import (PROJECT_EXTENSION, Project, ProjectError,
                          VideoEntry)
from core.project_io import (ProjectAutosaver, autosave_path_for,
                             export_batch_plan, load_project,
                             project_from_batch_plan, project_from_run_folder,
                             project_from_template, recovery_available,
                             save_project, with_project_extension)
from core.version import APP_NAME, APP_VERSION
from gui.app_settings import AppSettings
from gui.video_list import is_video_file
from gui.workspace.context_panel import (ACTION_APPLY_TEMPLATE, ACTION_RELINK,
                                         ACTION_REMOVE, ACTION_UNVERIFY,
                                         ACTION_VERIFY, ContextPanel)
from gui.workspace.prompts import (DISCARD, OPEN_FILTER, PROJECT_FILTER,
                                   SAVE, SOURCE_FILTER, Prompts)
from gui.workspace.setup_tab import SetupTab  # Setup tab (X2b-2)
from gui.workspace.tabs import (TAB_NAMES, TAB_RESULTS, TAB_RUN, TAB_SETUP,
                                TAB_TITLES, PlaceholderTab, WorkspaceTab)
from gui.workspace.video_panel import VideoPanel

logger = logging.getLogger(__name__)

TITLE_SEPARATOR = "—"
QUICKSTART_URL = (
    "https://github.com/Raul-Portugal/behavior_analyser/blob/main/quickstart.md"
)
DEFAULT_SPLITTER_SIZES = (400, 640, 360)

IMPORT_TEMPLATE = "template"
IMPORT_PLAN = "batch_plan"
IMPORT_RUN = "run_folder"
_IMPORT_LABELS = {
    IMPORT_TEMPLATE: "a saved template",
    IMPORT_PLAN: "a batch plan",
    IMPORT_RUN: "a previous run folder",
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


class WorkspaceWindow(QMainWindow):
    """Single-window workspace: video table, tabs and context panel."""

    project_changed = pyqtSignal(object)    # Project or None
    project_modified = pyqtSignal()
    selection_changed = pyqtSignal(list)    # keys of the selected videos
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

        self.setAcceptDrops(True)
        self.resize(1400, 850)
        self._build_ui()
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
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

        self._update_title()
        self._update_ui_state()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.video_panel = VideoPanel(self.app_settings, self.prompts)
        self.context_panel = ContextPanel(
            self.prompts, self.video_panel.model.issues_for
        )
        self.tabs = QTabWidget()
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
        for name in TAB_NAMES:
            self.tabs.addTab(self._tab_widget(name), TAB_TITLES[name])
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self.video_panel)
        self.splitter.addWidget(self.tabs)
        self.splitter.addWidget(self.context_panel)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.setCentralWidget(self.splitter)

        self.video_panel.selection_changed.connect(self._on_selection_changed)
        self.video_panel.paths_dropped.connect(self.handle_dropped_paths)
        self.video_panel.status_message.connect(self._show_status)
        self.video_panel.model.projectChanged.connect(self._on_model_changed)
        self.context_panel.paradigm_chosen.connect(self.change_paradigm)
        self.context_panel.action_requested.connect(self._on_context_action)
        self.statusBar()

    def _tab_widget(self, name: str) -> QWidget:
        return {
            TAB_SETUP: self.setup_tab,
            TAB_RUN: self.run_tab,
            TAB_RESULTS: self.results_tab,
        }[name]

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
        self.run_action = make("&Run", self.run_requested.emit, "F5",
                               "Run the analysis", enabled=False)
        self.cancel_action = make("&Cancel", self.cancel_requested.emit, None,
                                  "Stop the analysis", enabled=False)

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
                "moved. Available from the Setup tab.", enabled=False)
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

        self.validation_menu = bar.addMenu("&Validation")
        self.validation_menu.addAction(self.manual_scoring_action)
        self.validation_menu.addAction(self.position_spotcheck_action)
        self.validation_menu.addAction(self.validation_report_action)

        self.help_menu = bar.addMenu("&Help")
        self.help_menu.addAction(self.quickstart_action)
        self.help_menu.addAction(self.about_action)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main", self)
        toolbar.setObjectName("workspace_toolbar")
        toolbar.setMovable(False)
        for action in (self.new_action, self.open_action, self.save_action):
            toolbar.addAction(action)
        toolbar.addSeparator()
        for action in (self.add_videos_action, self.add_folder_action,
                       self.remove_action):
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addAction(self.run_action)
        toolbar.addAction(self.cancel_action)
        self.toolbar = toolbar
        self.addToolBar(toolbar)

    # ------------------------------------------------------------------
    # Public state for the tabs
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

    def selected_keys(self) -> List[str]:
        return self.video_panel.selected_keys()

    def selected_entries(self) -> List[VideoEntry]:
        project = self._project
        if project is None:
            return []
        keys = self.selected_keys()
        return project.entries_for(keys) if keys else []

    def install_tab(self, name: str, widget: QWidget) -> QWidget:
        """Replace the Setup, Run or Results tab; returns the old widget.

        A :class:`WorkspaceTab` is attached to the window (it then receives
        the project and selection hooks); any other widget is just shown.
        """
        if name not in TAB_NAMES:
            raise ValueError(f"Unknown tab {name!r}; use one of {TAB_NAMES}")
        index = TAB_NAMES.index(name)
        old = self._tab_widget(name)
        was_current = self.tabs.currentIndex() == index
        self.tabs.removeTab(index)
        self.tabs.insertTab(index, widget, TAB_TITLES[name])
        setattr(self, f"{name}_tab", widget)
        old.setParent(None)
        old.deleteLater()
        if isinstance(widget, WorkspaceTab):
            widget.attach_workspace(self)
        if was_current:
            self.tabs.setCurrentIndex(index)
        return old

    def show_tab(self, name: str) -> None:
        self.tabs.setCurrentIndex(TAB_NAMES.index(name))

    def current_tab_name(self) -> str:
        return TAB_NAMES[self.tabs.currentIndex()]

    def _on_tab_changed(self, index: int) -> None:
        widget = self.tabs.widget(index)
        if isinstance(widget, WorkspaceTab):
            widget.on_tab_activated()

    def set_run_enabled(self, enabled: bool) -> None:
        """Piece 3 calls this when a run can be started."""
        self._run_enabled = bool(enabled)
        self._update_ui_state()

    def set_busy(self, busy: bool, reason: str = "") -> None:
        """A run is in progress (or over): lock or unlock project editing."""
        self._busy = bool(busy)
        self.video_panel.set_read_only(self._busy)
        self._update_ui_state()
        if busy and reason:
            self._show_status(reason)

    def register_close_guard(self, guard: Callable[[], bool]) -> None:
        """``guard()`` returns False to veto closing the window or project.

        The guard explains the veto itself (it usually asks the user).
        """
        self._close_guards.append(guard)

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
        self.context_panel.paradigm_combo.setEnabled(editable)

    def _show_status(self, text: str) -> None:
        self.statusBar().showMessage(text, 8000)

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

    # ------------------------------------------------------------------
    # Signals from the panels and the project
    # ------------------------------------------------------------------
    def _on_selection_changed(self, keys: List[str]) -> None:
        self.context_panel.set_selection(keys)
        self._update_ui_state()
        self.selection_changed.emit(list(keys))

    def _on_model_changed(self) -> None:
        self.context_panel.refresh()

    def _on_project_edited(self) -> None:
        """Project listener: any change marks the project unsaved."""
        if self._closed:
            return
        self._unsaved = True
        self.setWindowModified(True)
        self.project_modified.emit()

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
        self.video_panel.set_project(project)
        self.context_panel.set_project(project)
        self._update_title()
        self._update_ui_state()
        self.project_changed.emit(project)

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
            self.context_panel.refresh()
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
                self.context_panel.refresh()    # put the combo back
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
        state = self.app_settings.workspace_state()
        if state is not None:
            self.restoreState(state)
        sizes = self.app_settings.workspace_splitter_sizes()
        if len(sizes) != self.splitter.count() or not any(sizes):
            sizes = list(DEFAULT_SPLITTER_SIZES)
        self.splitter.setSizes(sizes)

    def _save_layout(self) -> None:
        # A window that was never shown (e.g. --startup-check) has no layout
        # worth remembering and must not overwrite the user's saved one.
        if not self.isVisible():
            return
        self.app_settings.save_workspace_window(
            self.saveGeometry(), self.saveState()
        )
        self.app_settings.set_workspace_splitter_sizes(self.splitter.sizes())
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
        if self._autosaver is not None:
            self._autosaver.flush()
        self._save_layout()
        self._closed = True
        self._autosave_timer.stop()
        self._start_timer.stop()
        self.context_panel.shutdown()
        self._detach_project()
        self.video_panel.set_project(None)


def _forget_classic(window: QWidget) -> None:
    try:
        _open_classic_windows.remove(window)
    except ValueError:
        pass


__all__ = ["WorkspaceWindow", "quickstart_target"]
