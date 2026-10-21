"""
gui/workspace/setup_tab.py
The Setup tab of the workspace: configure the project template, or give the
selected videos their own settings.

::

    +--------------------------------------------------------------+
    | [Project template v]  Project template (applies to 12 videos) |
    +------------------+-------------------------------------------+
    | 1 v Parameters   |                                           |
    | 2 v Regions      |      the active step's widget             |
    | 3 ! Calibration  |      (created when entered, shut down     |
    | 4 o Detection    |       when left)                          |
    | 5 o Time window  |                                           |
    | 6 o Review       |  issues of the step                       |
    +------------------+-------------------------------------------+
    | Previous  Next                 Revert  Apply  Mark verified  |
    +--------------------------------------------------------------+

How a change travels
--------------------
A step widget (panel, ROI editor, tuner) edits nothing until the user presses
**Apply**: the tab then asks the step to write itself into a *copy* of the
scope's ``SetupState``, converts that with ``to_batch_settings(finalize=True)``
(exactly what the classic review writes) and hands the result to the project:
``set_template`` for the template, ``customize`` for each selected video. The
project listener refreshes the table; the tab ignores its own echo and
refreshes the step list instead.

Steps are reachable in any order. Leaving a step, switching the scope or the
selection, closing the project or the window with unapplied edits asks
Apply / Discard / Cancel. A step's widget is torn down (tuners are shut down
and their threads joined) every time the step is left.

Until the regions exist there is no ``BatchSettings`` to store, so the edits
made before that (paradigm parameters) wait in a per-scope draft.
"""
import copy
import logging
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
                             QListWidget, QListWidgetItem, QPushButton,
                             QStackedWidget, QVBoxLayout, QWidget)

from core import roi_alignment as ra
from core.models import AppConfig, BatchSettings
from core.project import (SEVERITY_INFO, SOURCE_ALIGNED, SOURCE_CUSTOM,
                          SOURCE_TEMPLATE, ZONES_ADJUSTED, ZONES_ALIGNED,
                          VideoEntry, canonical_json)
from core.setup_state import (STATUS_DONE, STATUS_INVALID, STEP_LABELS,
                              SetupState)
from core.video import ReferenceFrameGenerator
from gui.background_task import run_with_progress
from gui.config_checks import probe_dimensions
from gui.roi_alignment_dialog import align_template_with_dialog
from gui.roi_editor import TOOL_LINE
from gui.template_alignment import read_middle_frame
from gui.workspace.prompts import SAVE, DISCARD, VIDEO_FILTER
from gui.workspace.setup_steps import (MessageHost, RoiHost, StepHost,
                                       StepServices, create_host, status_icon)
from gui.workspace.tabs import WorkspaceTab
from mazes import AVAILABLE_MAZES
from mazes.base_maze import (STEP_CALIBRATION, STEP_DETECTION_TUNING,
                             STEP_PARADIGM_PARAMS, STEP_REVIEW, STEP_ROIS,
                             STEP_TIME_WINDOW, STEP_TST_TUNING)

logger = logging.getLogger(__name__)

SCOPE_TEMPLATE = "template"
SCOPE_VIDEOS = "videos"
SCOPE_LABELS = {
    SCOPE_TEMPLATE: "Project template",
    SCOPE_VIDEOS: "Selected video(s)",
}
TEMPLATE_KEY = "\0template"

NO_PROJECT_TEXT = (
    "Open or create a project (File menu) to set up the analysis."
)
NO_PARADIGM_TEXT = (
    "Choose the paradigm in the project box on the right (the “Paradigm” "
    "chooser of the context panel): the setup steps depend on it."
)
NO_SELECTION_TEXT = (
    "Select one or more videos in the table to give them their own settings, "
    "or switch the scope back to the project template."
)
ISSUE_STYLE = "color: #d1242f;"
WARNING_STYLE = "color: #9a6700;"


def _fields_of_step(target: SetupState, source: SetupState, step: str) -> None:
    """Copy what ``step`` owns from ``source`` into ``target``."""
    if step in (STEP_PARADIGM_PARAMS, STEP_TST_TUNING):
        target.maze_name = source.maze_name
        target.algorithm_parameters = copy.deepcopy(source.algorithm_parameters)
        target.legacy_untagged = False
    elif step == STEP_ROIS:
        target.roi_manager = copy.deepcopy(source.roi_manager)
    elif step == STEP_CALIBRATION:
        target.scale_cm = source.scale_cm
    elif step == STEP_DETECTION_TUNING:
        target.detection_config = copy.deepcopy(source.detection_config)
    elif step == STEP_TIME_WINDOW:
        target.start_time = source.start_time
        target.end_time = source.end_time
        target.per_roi_times = dict(source.per_roi_times)
    elif step == STEP_REVIEW:
        target.create_timelapse = source.create_timelapse
    target.mark_done(step)


class SetupTab(WorkspaceTab):
    """Step-by-step setup of the project template or of selected videos."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._project = None
        self._scope = SCOPE_TEMPLATE
        self._state: Optional[SetupState] = None
        self._targets: List[str] = []
        self._state_key: Optional[str] = None
        self._steps: List[str] = []
        self._step: Optional[str] = None
        self._host: Optional[StepHost] = None
        self._completed: Dict[str, Set[str]] = {}
        self._drafts: Dict[str, SetupState] = {}
        self._ref_key: Optional[str] = None
        self._paradigm_seen = ""
        self._synced_json: Optional[str] = None
        self._apply_issues: List[str] = []
        self._dims: Dict[Path, Optional[Tuple[int, int]]] = {}
        self._committing = False
        self._syncing = False
        self._restoring_selection = False
        self.services = StepServices(
            read_frame=lambda path: read_middle_frame(path),
            reference_frame=self._generate_reference_frame,
            project_lines=self._project_issue_lines,
        )
        self._build_ui()
        self._show_empty(NO_PROJECT_TEXT)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.scope_combo = QComboBox()
        for scope in (SCOPE_TEMPLATE, SCOPE_VIDEOS):
            self.scope_combo.addItem(SCOPE_LABELS[scope], scope)
        self.scope_combo.setToolTip(
            "Edit the shared project template, or give the selected videos "
            "their own (custom) settings"
        )
        self.scope_combo.currentIndexChanged.connect(self._on_scope_changed)
        self.header_label = QLabel()
        self.header_label.setWordWrap(True)
        self.header_label.setTextFormat(Qt.TextFormat.RichText)
        top.addWidget(QLabel("Edit:"))
        top.addWidget(self.scope_combo)
        top.addWidget(self.header_label, 1)
        root.addLayout(top)

        reference = QHBoxLayout()
        self.reference_caption = QLabel("Reference video:")
        self.reference_combo = QComboBox()
        self.reference_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.reference_combo.setToolTip(
            "The video whose frame the regions and tuners are shown on"
        )
        self.reference_combo.currentIndexChanged.connect(
            self._on_reference_changed
        )
        reference.addWidget(self.reference_caption)
        reference.addWidget(self.reference_combo)
        reference.addStretch()
        self.reference_row = QWidget()
        self.reference_row.setLayout(reference)
        reference.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.reference_row)

        banner = QHBoxLayout()
        self.banner_label = QLabel()
        self.banner_label.setWordWrap(True)
        self.banner_label.setStyleSheet(WARNING_STYLE)
        self.banner_align_button = QPushButton("Align template...")
        self.banner_align_button.clicked.connect(
            lambda _checked=False: self.align_template()
        )
        banner.addWidget(self.banner_label, 1)
        banner.addWidget(self.banner_align_button)
        banner.setContentsMargins(0, 0, 0, 0)
        self.banner = QWidget()
        self.banner.setLayout(banner)
        root.addWidget(self.banner)

        middle = QHBoxLayout()
        self.step_list = QListWidget()
        self.step_list.setFixedWidth(200)
        self.step_list.currentRowChanged.connect(self._on_row_changed)
        middle.addWidget(self.step_list)

        right = QVBoxLayout()
        self.stack = QStackedWidget()
        self.empty_label = QLabel()
        self.empty_label.setWordWrap(True)
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stack.addWidget(self.empty_label)
        right.addWidget(self.stack, 1)
        self.issues_label = QLabel()
        self.issues_label.setWordWrap(True)
        self.issues_label.setTextFormat(Qt.TextFormat.RichText)
        right.addWidget(self.issues_label)
        middle.addLayout(right, 1)
        root.addLayout(middle, 1)

        buttons = QHBoxLayout()
        self.previous_button = QPushButton("Previous step")
        self.next_button = QPushButton("Next step")
        self.revert_button = QPushButton("Revert step")
        self.revert_button.setToolTip("Discard the unapplied edits of this step")
        self.apply_button = QPushButton("Apply")
        self.apply_button.setToolTip(
            "Write this step into the settings of the project"
        )
        self.verify_button = QPushButton("Mark selected videos verified")
        self.verify_button.setToolTip(
            "Record that you checked the regions of the selected videos"
        )
        self.previous_button.clicked.connect(
            lambda _checked=False: self.go_relative(-1)
        )
        self.next_button.clicked.connect(
            lambda _checked=False: self.go_relative(1)
        )
        self.revert_button.clicked.connect(
            lambda _checked=False: self.revert_step()
        )
        self.apply_button.clicked.connect(
            lambda _checked=False: self.apply_step()
        )
        self.verify_button.clicked.connect(
            lambda _checked=False: self.mark_selected_verified()
        )
        buttons.addWidget(self.previous_button)
        buttons.addWidget(self.next_button)
        buttons.addStretch()
        buttons.addWidget(self.revert_button)
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.verify_button)
        root.addLayout(buttons)

    # ------------------------------------------------------------------
    # The workspace contract
    # ------------------------------------------------------------------
    def attach_workspace(self, window) -> None:
        self.workspace = window
        window.register_close_guard(self._close_guard)
        window.align_action.triggered.connect(self._on_align_action)
        window.align_action.setToolTip(
            "Carry the template's regions onto videos whose camera moved or "
            "whose resolution differs."
        )
        # A tuner still scanning when the application quits must be joined.
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)
        super().attach_workspace(window)
        self._refresh_actions()

    @property
    def prompts(self):
        return self.workspace.prompts

    @property
    def project(self):
        return self._project

    @property
    def scope(self) -> str:
        return self._scope

    @property
    def step(self) -> Optional[str]:
        return self._step

    @property
    def steps(self) -> List[str]:
        return list(self._steps)

    @property
    def host(self) -> Optional[StepHost]:
        return self._host

    @property
    def state(self) -> Optional[SetupState]:
        """The scope's working state (the applied values; a copy)."""
        return None if self._state is None else self._state.copy()

    @property
    def targets(self) -> List[str]:
        return list(self._targets)

    def on_project_changed(self, project) -> None:
        self._teardown_host()
        self._project = project
        self._completed.clear()
        self._drafts.clear()
        self._dims.clear()
        self._ref_key = None
        self._apply_issues = []
        self._paradigm_seen = "" if project is None else project.effective_paradigm()
        self._reload()
        self._refresh_actions()

    def on_project_modified(self) -> None:
        if self._project is None:
            return
        if self._committing:
            self._refresh_chrome()
            return
        paradigm = self._project.effective_paradigm()
        if paradigm != self._paradigm_seen:
            # Settings belong to the paradigm: the project cleared them.
            self._paradigm_seen = paradigm
            self._completed.clear()
            self._drafts.clear()
            self._reload()
        elif (self._scope_json() != self._synced_json
              and not (self._host is not None and self._host.is_dirty())):
            self._reload(keep_step=True)
        else:
            self._sync_reference_combo()
            self._refresh_chrome()
        self._refresh_actions()

    def on_selection_changed(self, keys: List[str]) -> None:
        if self._restoring_selection or self._project is None:
            return
        if self._scope != SCOPE_VIDEOS:
            self._refresh_chrome()
            return
        targets = self._selected_targets()
        if targets == self._targets:
            self._refresh_chrome()
            return
        if not self._resolve_unapplied(rebuild=False):
            self._restoring_selection = True
            try:
                self.workspace.video_panel.select_keys(self._targets)
            finally:
                self._restoring_selection = False
            return
        self._reload(keep_step=True)

    def on_tab_activated(self) -> None:
        if isinstance(self._host, MessageHost) and self._step is not None:
            # A prerequisite may have been met on another tab.
            self._enter_step(self._step)
        self._refresh_chrome()
        self._refresh_actions()

    def shutdown(self) -> None:
        """Tear the active step down (tuners are joined)."""
        self._teardown_host()

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # What is being edited
    # ------------------------------------------------------------------
    def _is_busy(self) -> bool:
        return bool(getattr(self.workspace, "is_busy", False))

    def _selected_targets(self) -> List[str]:
        if self._project is None or self.workspace is None:
            return []
        known = set(self._project.keys())
        return [k for k in self.workspace.selected_keys() if k in known]

    def _scope_settings(self) -> Optional[BatchSettings]:
        project = self._project
        if project is None:
            return None
        if self._scope == SCOPE_TEMPLATE:
            return project.template_settings
        if not self._targets:
            return None
        return project.effective_settings(self._targets[0])

    def _scope_json(self) -> Optional[str]:
        settings = self._scope_settings()
        return None if settings is None else canonical_json(settings.to_dict())

    def _resolve_scope(self) -> Optional[str]:
        """Build the working state; the reason text when there is none."""
        self._state = None
        self._targets = []
        self._state_key = None
        project = self._project
        if project is None:
            return NO_PROJECT_TEXT
        paradigm = project.effective_paradigm()
        if not paradigm or paradigm not in AVAILABLE_MAZES:
            return NO_PARADIGM_TEXT
        if self._scope == SCOPE_VIDEOS:
            self._targets = self._selected_targets()
            if not self._targets:
                return NO_SELECTION_TEXT
            entry = project.entry(self._targets[0])
            settings = project.effective_settings(entry) or project.template_settings
            self._state_key = self._targets[0]
        else:
            settings = project.template_settings
            self._state_key = TEMPLATE_KEY
        key = self._state_key

        existing = self._scope_settings()
        if settings is not None:
            name = None if settings.paradigm_id in AVAILABLE_MAZES else paradigm
            state = SetupState.from_batch_settings(settings, maze_name=name)
            if key not in self._completed:
                seeded = set(state.steps) - {STEP_REVIEW}
                if self._scope == SCOPE_VIDEOS and entry.verified and existing:
                    seeded.add(STEP_REVIEW)
                self._completed[key] = seeded
            self._drafts.pop(key, None)
        elif key in self._drafts:
            state = self._drafts[key].copy()
        else:
            state = SetupState.new(paradigm)
            if project.algorithm_parameters:
                state.algorithm_parameters = copy.deepcopy(
                    project.algorithm_parameters
                )
        state.completed_steps = set(self._completed.get(key, set()))
        state.video_path = self._reference_path()
        self._state = state
        self._synced_json = self._scope_json()
        return None

    def _reference_path(self) -> Optional[Path]:
        project = self._project
        if project is None:
            return None
        if self._scope == SCOPE_VIDEOS:
            return project.entry(self._targets[0]).path if self._targets else None
        self._sync_reference_combo()
        if self._ref_key is None:
            return None
        return project.entry(self._ref_key).path

    def _sync_reference_combo(self) -> None:
        """Keep the reference chooser in step with the project's videos."""
        project = self._project
        keys = [] if project is None else project.keys()
        if self._ref_key not in keys:
            self._ref_key = keys[0] if keys else None
        current = [
            self.reference_combo.itemData(i)
            for i in range(self.reference_combo.count())
        ]
        self.reference_combo.blockSignals(True)
        try:
            if current != keys:
                self.reference_combo.clear()
                for video in ([] if project is None else project.videos):
                    self.reference_combo.addItem(video.name, video.key)
            if self._ref_key is not None:
                self.reference_combo.setCurrentIndex(
                    self.reference_combo.findData(self._ref_key)
                )
        finally:
            self.reference_combo.blockSignals(False)

    def _remember_completed(self, state: SetupState) -> None:
        if self._state_key is not None:
            self._completed[self._state_key] = set(state.completed_steps)

    # ------------------------------------------------------------------
    # Rebuilding the tab
    # ------------------------------------------------------------------
    def _reload(self, keep_step: bool = False) -> None:
        """Re-derive the working state and show the right step."""
        previous = self._step
        self._teardown_host()
        self._apply_issues = []
        reason = self._resolve_scope()
        if reason is not None:
            self._show_empty(reason)
            return
        self._steps = list(self._state.steps)
        self._populate_steps()
        if keep_step and previous in self._steps:
            step = previous
        else:
            step = self._first_open_step()
        self._enter_step(step)

    def _first_open_step(self) -> str:
        reports = self._state.step_reports()
        for step in self._steps:
            if reports[step].status != STATUS_DONE:
                return step
        return self._steps[0]

    def _show_empty(self, text: str) -> None:
        self._teardown_host()
        self._steps = []
        self._step = None
        self._state = None
        self._syncing = True
        try:
            self.step_list.clear()
        finally:
            self._syncing = False
        self.empty_label.setText(text)
        self.stack.setCurrentWidget(self.empty_label)
        self.issues_label.setText("")
        self._refresh_chrome()

    def _populate_steps(self) -> None:
        self._syncing = True
        try:
            self.step_list.clear()
            for index, step in enumerate(self._steps, start=1):
                item = QListWidgetItem(f"{index}. {STEP_LABELS[step]}")
                item.setData(Qt.ItemDataRole.UserRole, step)
                self.step_list.addItem(item)
        finally:
            self._syncing = False

    def _teardown_host(self) -> None:
        host, self._host = self._host, None
        if host is None:
            return
        try:
            self.stack.removeWidget(host)
            host.shutdown()
        finally:
            host.setParent(None)
            host.deleteLater()

    def _enter_step(self, step: str) -> None:
        """Show ``step``: build its widget now, shut the previous one down."""
        self._teardown_host()
        self._step = step
        self._apply_issues = []
        state = self._state
        video = state.video_path if state is not None else None
        host = create_host(step, state, video, self.services)
        host.step_requested.connect(self._on_step_requested)
        host.retry_requested.connect(self._on_retry)
        host.changed.connect(self._on_host_changed)
        self._host = host
        self.stack.addWidget(host)
        self.stack.setCurrentWidget(host)
        self._refresh_chrome()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def go_to_step(self, step: str, tool: Optional[str] = None) -> bool:
        """Open ``step``; False when unapplied edits stayed (user cancelled)."""
        if step not in self._steps:
            return False
        if step != self._step or self._host is None:
            if not self._resolve_unapplied(rebuild=False):
                return False
            self._enter_step(step)
        if tool == "line" and isinstance(self._host, RoiHost):
            self._host.set_tool(TOOL_LINE)
        return True

    def go_relative(self, offset: int) -> bool:
        if self._step not in self._steps:
            return False
        index = self._steps.index(self._step) + offset
        if not 0 <= index < len(self._steps):
            return False
        return self.go_to_step(self._steps[index])

    def _on_row_changed(self, row: int) -> None:
        if self._syncing or row < 0 or row >= len(self._steps):
            return
        if not self.go_to_step(self._steps[row]):
            self._select_row_of_active_step()

    def _on_step_requested(self, step: str, tool: str) -> None:
        self.go_to_step(step, tool or None)

    def _on_retry(self) -> None:
        if self._step is not None and self._state is not None:
            self._enter_step(self._step)

    def _on_host_changed(self) -> None:
        self._refresh_chrome()

    def _select_row_of_active_step(self) -> None:
        if self._step not in self._steps:
            return
        self._syncing = True
        try:
            self.step_list.setCurrentRow(self._steps.index(self._step))
        finally:
            self._syncing = False

    # -- scope and reference ------------------------------------------------
    def set_scope(self, scope: str) -> bool:
        index = self.scope_combo.findData(scope)
        if index < 0:
            return False
        self.scope_combo.setCurrentIndex(index)
        return self._scope == scope

    def _on_scope_changed(self, index: int) -> None:
        scope = self.scope_combo.itemData(index)
        if scope == self._scope:
            return
        if not self._resolve_unapplied(rebuild=False):
            self.scope_combo.blockSignals(True)
            self.scope_combo.setCurrentIndex(self.scope_combo.findData(self._scope))
            self.scope_combo.blockSignals(False)
            return
        self._scope = scope
        self._reload(keep_step=True)

    def set_reference_video(self, key: str) -> bool:
        index = self.reference_combo.findData(key)
        if index < 0:
            return False
        self.reference_combo.setCurrentIndex(index)
        return self._ref_key == key

    def _on_reference_changed(self, index: int) -> None:
        key = self.reference_combo.itemData(index)
        if key is None or key == self._ref_key:
            return
        if not self._resolve_unapplied(rebuild=False):
            self._sync_reference_combo()
            return
        self._ref_key = key
        if self._state is not None and self._scope == SCOPE_TEMPLATE:
            self._state.video_path = self._reference_path()
            if self._step is not None:
                self._enter_step(self._step)

    # ------------------------------------------------------------------
    # Applying
    # ------------------------------------------------------------------
    def apply_step(self) -> bool:
        """Write the active step into the project; False if it cannot be."""
        host, state = self._host, self._state
        if host is None or state is None or isinstance(host, MessageHost):
            return False
        if self._is_busy():
            self.prompts.warn(
                "Setup", "A run is in progress: the settings cannot be "
                         "changed until it finishes."
            )
            return False
        candidate = state.copy()
        if not host.apply(candidate):
            self._apply_issues = host.issues() or [
                "This step cannot be applied yet."
            ]
            self._refresh_chrome()
            return False
        if not self._commit(candidate, host.step_id):
            return False
        self._state = candidate
        self._remember_completed(candidate)
        host.mark_clean()
        self._apply_issues = []
        self._refresh_chrome()
        return True

    def _commit(self, candidate: SetupState, step: str) -> bool:
        project = self._project
        if candidate.roi_manager is None:
            # Nothing to store until the regions exist; keep a draft.
            self._drafts[self._state_key] = candidate.copy()
            return True
        self._committing = True
        try:
            if self._scope == SCOPE_TEMPLATE:
                project.set_template(candidate.to_batch_settings(finalize=True))
            else:
                self._commit_to_videos(candidate, step)
        except ValueError as exc:
            self._apply_issues = [str(exc)]
            self._refresh_chrome()
            return False
        finally:
            self._committing = False
        self._drafts.pop(self._state_key, None)
        self._state = candidate
        self._synced_json = self._scope_json()
        return True

    def _commit_to_videos(self, candidate: SetupState, step: str) -> None:
        project = self._project
        skipped: List[str] = []
        for position, key in enumerate(self._targets):
            entry = project.entry(key)
            if position == 0:
                settings = candidate.to_batch_settings(finalize=True)
            else:
                settings = self._merged_settings(entry, candidate, step)
                if settings is None:
                    skipped.append(entry.name)
                    continue
            current = project.effective_settings(entry)
            if (
                current is not None
                and entry.settings_source == SOURCE_TEMPLATE
                and canonical_json(current.to_dict())
                == canonical_json(settings.to_dict())
            ):
                continue    # nothing changed: it keeps following the template
            keep_alignment = (
                project.zone_status(entry) == ZONES_ALIGNED and step != STEP_ROIS
            )
            project.customize(
                key, settings,
                source=SOURCE_ALIGNED if keep_alignment else SOURCE_CUSTOM,
            )
        if skipped:
            self.prompts.warn(
                "Setup",
                "These videos have no regions yet, so only the first "
                "selected video was updated:\n" + "\n".join(skipped),
            )

    def _merged_settings(
        self, entry: VideoEntry, candidate: SetupState, step: str
    ) -> Optional[BatchSettings]:
        """``entry``'s own settings with just ``step`` taken from ``candidate``."""
        settings = (
            self._project.effective_settings(entry)
            or self._project.template_settings
        )
        if settings is None:
            if candidate.roi_manager is None or step != STEP_ROIS:
                return None
            target = SetupState.new(candidate.maze_name)
        else:
            name = None if settings.paradigm_id in AVAILABLE_MAZES else candidate.maze_name
            target = SetupState.from_batch_settings(settings, maze_name=name)
        _fields_of_step(target, candidate, step)
        try:
            return target.to_batch_settings(finalize=True)
        except ValueError:
            return None

    def revert_step(self) -> None:
        """Drop the unapplied edits of the active step."""
        if self._step is not None and self._state is not None:
            self._enter_step(self._step)

    def mark_selected_verified(self) -> bool:
        project = self._project
        keys = self._selected_targets()
        if project is None or not keys or self._is_busy():
            return False
        if not self._resolve_unapplied(rebuild=True):
            return False
        configured = [k for k in keys if project.entry(k).settings is not None]
        if len(configured) != len(keys):
            self.prompts.warn(
                "Mark verified",
                "Videos without settings cannot be verified; they were "
                "skipped.",
            )
        if configured:
            project.set_verified(configured, True)
        return bool(configured)

    # ------------------------------------------------------------------
    # The unapplied-edits guard
    # ------------------------------------------------------------------
    def has_unapplied_edits(self) -> bool:
        return (
            self._host is not None and self._project is not None
            and not isinstance(self._host, MessageHost) and self._host.is_dirty()
        )

    def _resolve_unapplied(self, rebuild: bool = True) -> bool:
        """Ask what to do with unapplied edits; True when it is fine to go on.

        ``rebuild`` re-creates the step from the applied state after Discard
        (when the caller is not going to leave the step anyway).
        """
        if not self.has_unapplied_edits():
            return True
        step = self._host.step_id
        answer = self.prompts.ask_save_changes(
            f"{STEP_LABELS.get(step, step)} ({self._scope_text()})"
        )
        if answer == SAVE:
            return self.apply_step()
        if answer == DISCARD:
            if rebuild:
                self.revert_step()
            return True
        return False

    def _close_guard(self) -> bool:
        """Window close guard: may the project be closed?"""
        return self._resolve_unapplied(rebuild=False)

    # ------------------------------------------------------------------
    # Status display
    # ------------------------------------------------------------------
    def _scope_text(self) -> str:
        if self._scope == SCOPE_TEMPLATE:
            return "project template"
        if len(self._targets) == 1:
            return self._project.entry(self._targets[0]).name
        return f"{len(self._targets)} selected videos"

    def _templated_count(self) -> int:
        return sum(
            1 for video in self._project.videos
            if video.settings is None or video.settings_source == SOURCE_TEMPLATE
        )

    def header_text(self) -> str:
        """Plain text of what is being edited."""
        project = self._project
        if project is None or self._state is None:
            return ""
        if self._scope == SCOPE_TEMPLATE:
            count = self._templated_count()
            noun = "video" if count == 1 else "videos"
            return (
                f"Project template (applies to {count} {noun} using the "
                "template)"
            )
        if len(self._targets) == 1:
            entry = project.entry(self._targets[0])
            text = f"Custom settings for {entry.name}"
            if entry.settings is None or entry.settings_source == SOURCE_TEMPLATE:
                text += " (starts from the template; applying makes it custom)"
            return text
        first = project.entry(self._targets[0]).name
        return (
            f"Custom settings for {len(self._targets)} selected videos "
            f"(shown: {first}; each step you apply is copied to all of them)"
        )

    def _dims_of(self, path: Path) -> Optional[Tuple[int, int]]:
        if path not in self._dims:
            self._dims[path] = probe_dimensions(path)
        return self._dims[path]

    def mismatch_text(self) -> str:
        """Why the template does not fit the selected video ('' if it does)."""
        project = self._project
        if (
            project is None or self._scope != SCOPE_VIDEOS or not self._targets
            or project.template_settings is None
        ):
            return ""
        expected = project.template_settings.roi_manager.source_dimensions
        entry = project.entry(self._targets[0])
        if expected is None or project.zone_status(entry) in (
            ZONES_ALIGNED, ZONES_ADJUSTED
        ):
            return ""
        actual = self._dims_of(entry.path)
        if actual is None or tuple(actual) == tuple(expected):
            return ""
        return (
            f"{entry.name} is {actual[0]}x{actual[1]} but the template was "
            f"drawn on {expected[0]}x{expected[1]} frames, so its regions "
            "would be misplaced."
        )

    def _refresh_chrome(self) -> None:
        has_state = self._state is not None
        busy = self._is_busy()
        self.header_label.setText(
            f"<b>{escape(self.header_text())}</b>" if has_state else ""
        )
        self.reference_row.setVisible(
            has_state and self._scope == SCOPE_TEMPLATE
        )
        mismatch = self.mismatch_text() if has_state else ""
        self.banner.setVisible(bool(mismatch))
        self.banner_label.setText(mismatch)
        self._refresh_step_items()
        host = self._host
        editable = (
            has_state and host is not None and not isinstance(host, MessageHost)
            and not busy
        )
        index = self._steps.index(self._step) if self._step in self._steps else -1
        self.previous_button.setEnabled(has_state and index > 0)
        self.next_button.setEnabled(
            has_state and 0 <= index < len(self._steps) - 1
        )
        self.apply_button.setEnabled(editable)
        self.revert_button.setEnabled(editable and host.is_dirty())
        self.verify_button.setEnabled(
            self._project is not None and not busy
            and bool(self._selected_targets())
        )
        self.scope_combo.setEnabled(self._project is not None)
        self._refresh_issues()

    def _refresh_step_items(self) -> None:
        if self._state is None:
            return
        reports = self._state.step_reports()
        self._syncing = True
        try:
            for row, step in enumerate(self._steps):
                item = self.step_list.item(row)
                if item is None:
                    continue
                report = reports[step]
                dirty = (
                    step == self._step and self.has_unapplied_edits()
                )
                item.setIcon(status_icon(report.status))
                item.setText(
                    f"{row + 1}. {STEP_LABELS[step]}" + (" *" if dirty else "")
                )
                tip = "\n".join(report.issues) or {
                    STATUS_DONE: "Done",
                    STATUS_INVALID: "Invalid",
                }.get(report.status, "Not done yet")
                item.setToolTip(tip)
            self._select_row_of_active_step_quietly()
        finally:
            self._syncing = False

    def _select_row_of_active_step_quietly(self) -> None:
        if self._step in self._steps:
            row = self._steps.index(self._step)
            if self.step_list.currentRow() != row:
                self.step_list.setCurrentRow(row)

    def step_status(self, step: str) -> str:
        return self._state.step_status(step) if self._state is not None else ""

    def displayed_issues(self) -> List[str]:
        """The issue lines under the step (apply failures, else the report)."""
        if self._state is None or self._step is None:
            return []
        if self._apply_issues:
            return list(self._apply_issues)
        host = self._host
        lines: List[str] = []
        if host is not None and not isinstance(host, MessageHost):
            lines.extend(host.issues())
        report = self._state.step_report(self._step)
        for issue in report.issues:
            if issue not in lines:
                lines.append(issue)
        return lines

    def _refresh_issues(self) -> None:
        issues = self.displayed_issues()
        warnings: List[str] = []
        if self._state is not None and self._step is not None:
            warnings = list(self._state.step_report(self._step).warnings)
        parts = []
        if issues:
            parts.append(
                f'<span style="{ISSUE_STYLE}">'
                + "<br>".join(f"&bull; {escape(i)}" for i in issues)
                + "</span>"
            )
        if warnings:
            parts.append(
                f'<span style="{WARNING_STYLE}">'
                + "<br>".join(f"&bull; {escape(w)}" for w in warnings)
                + "</span>"
            )
        self.issues_label.setText("<br>".join(parts))

    def _project_issue_lines(self) -> List[str]:
        """Per-video problems of the project and of the maze's validation."""
        project = self._project
        if project is None:
            return []
        lines: List[str] = []
        for issue in project.validate():
            if issue.severity != SEVERITY_INFO:
                lines.append(issue.message)
        maze = None if self._state is None else self._state.build_maze()
        if maze is not None:
            for video in project.videos:
                if video.settings is None:
                    continue
                for text in maze.validate_settings(
                    project.effective_settings(video)
                ):
                    lines.append(f"{video.name}: {text}")
        return lines

    # ------------------------------------------------------------------
    # Reference frame (off the GUI thread)
    # ------------------------------------------------------------------
    def _generate_reference_frame(self, video: Path):
        samples = AppConfig().video.ref_frame_samples
        return run_with_progress(
            self.window(),
            f"Generating reference frame for {Path(video).name}...",
            lambda should_cancel: ReferenceFrameGenerator.generate(
                Path(video), samples, use_cache=True,
                should_cancel=should_cancel,
            ),
            title="Reference Frame",
        )

    # ------------------------------------------------------------------
    # Template alignment
    # ------------------------------------------------------------------
    def _refresh_actions(self) -> None:
        window = self.workspace
        if window is None:
            return
        project = self._project
        window.align_action.setEnabled(
            project is not None and project.template_settings is not None
            and bool(project.videos)
        )

    def _on_align_action(self, _checked: bool = False) -> None:
        self.align_template()

    def _find_template_video(self, template: BatchSettings) -> Optional[Path]:
        """A project video showing the template's own frame size, or ask."""
        project = self._project
        expected = template.roi_manager.source_dimensions
        reference = (
            None if self._ref_key is None else project.entry(self._ref_key)
        )
        candidates = [v for v in ([reference] if reference else []) + project.videos]
        for video in candidates:
            if not video.path.is_file():
                continue
            dims = self._dims_of(video.path)
            if dims is not None and (expected is None or tuple(dims) == tuple(expected)):
                return video.path
        size = f" ({expected[0]}x{expected[1]})" if expected else ""
        chosen = self.prompts.open_file(
            f"Which video was the template drawn on?{size}",
            str(project.base_dir or Path.home()), VIDEO_FILTER,
        )
        if chosen is None:
            return None
        dims = probe_dimensions(chosen)
        if dims is None:
            self.prompts.warn("Cannot read video", f"'{chosen.name}' could not be read.")
            return None
        if expected is not None and tuple(dims) != tuple(expected):
            self.prompts.warn(
                "Wrong video size",
                f"'{chosen.name}' is {dims[0]}x{dims[1]}, but the template "
                f"was drawn on {expected[0]}x{expected[1]} frames.",
            )
            return None
        return chosen

    def align_template(self) -> int:
        """Carry the template onto the selected (or mismatching) videos.

        Opens the alignment dialog for each video; the result is stored as
        the video's own settings (source "aligned"). Returns how many videos
        were aligned.
        """
        project = self._project
        if project is None or self._is_busy():
            return 0
        template = project.template_settings
        if template is None:
            self.prompts.info(
                "Align template",
                "Set up the project template first (Setup tab, scope "
                "“Project template”).",
            )
            return 0
        if not self._resolve_unapplied(rebuild=True):
            return 0
        selected = self._selected_targets()
        entries = project.entries_for(selected) if selected else list(project.videos)
        entries = [e for e in entries if e.path.is_file()]
        expected = template.roi_manager.source_dimensions
        if not selected and expected is not None:
            entries = [
                e for e in entries
                if (self._dims_of(e.path) or tuple(expected)) != tuple(expected)
            ]
        if not entries:
            self.prompts.info(
                "Align template",
                "Every video already has the frame size of the template.",
            )
            return 0
        template_video = self._find_template_video(template)
        if template_video is None:
            return 0
        template_frame = read_middle_frame(template_video)
        if template_frame is None:
            self.prompts.warn(
                "Cannot read video",
                f"A frame could not be read from '{template_video.name}'.",
            )
            return 0
        samples = AppConfig().video.ref_frame_samples
        pending = [e for e in entries if e.path != template_video]
        aligned = 0
        while pending:
            entry = pending.pop(0)
            target_frame = read_middle_frame(entry.path)
            if target_frame is None:
                self.prompts.warn(
                    "Cannot read video",
                    f"A frame could not be read from '{entry.name}'.",
                )
                continue

            def auto_align(should_cancel, video=entry.path):
                return ra.register_videos(
                    template_video, video, samples, should_cancel
                )

            choice = align_template_with_dialog(
                self.window(), template.roi_manager, template_frame,
                target_frame, auto_align, template_name=template_video.name,
                target_name=entry.name,
            )
            if choice is None:
                break
            dims = self._dims_of(entry.path) or probe_dimensions(entry.path)
            self._store_alignment(entry, template, choice, dims)
            aligned += 1
            same = [e for e in pending if self._dims_of(e.path) == dims]
            if same and self.prompts.confirm(
                "Same camera?",
                f"{len(same)} more video(s) have the same frame size "
                f"({dims[0]}x{dims[1]}). Use this alignment for all of them?",
                accept="Use for all",
                informative="Choose this only if they were recorded with the "
                            "camera in the same place.",
            ):
                for other in same:
                    self._store_alignment(other, template, choice, dims)
                    aligned += 1
                pending = [e for e in pending if e not in same]
        return aligned

    def _store_alignment(self, entry: VideoEntry, template: BatchSettings,
                         choice, dims) -> None:
        settings = ra.apply_alignment(template, choice.matrix, dims)
        self._project.customize(entry.key, settings, source=SOURCE_ALIGNED)


__all__ = [
    "SCOPE_TEMPLATE", "SCOPE_VIDEOS", "SetupTab", "TEMPLATE_KEY",
]
