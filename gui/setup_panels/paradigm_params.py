"""Paradigm parameter panels: arm-entry rules, open field, TST."""
from typing import Any, Dict, List

from PyQt6.QtWidgets import (QCheckBox, QDoubleSpinBox, QFormLayout, QLabel,
                             QSpinBox, QStackedLayout, QVBoxLayout)

from gui.setup_panels.base import SetupPanel
from mazes.arm_entry import MIN_DWELL_RANGE_S
from mazes.base_maze import (KIND_ARMS, KIND_FREESTYLE, KIND_TST,
                             STEP_PARADIGM_PARAMS)


class MazeParamsPanel(SetupPanel):
    """Base of the per-paradigm panels: they edit a ``Maze`` object.

    ``load_from_maze`` / ``apply_to_maze`` are the maze-level API the
    dialogs use; ``load`` / ``apply`` adapt them to a ``SetupState``.
    """

    step_id = STEP_PARADIGM_PARAMS

    def load_from_maze(self, maze) -> None:
        raise NotImplementedError

    def apply_to_maze(self, maze) -> None:
        raise NotImplementedError

    def load(self, state) -> None:
        self.load_from_maze(state.build_maze())

    def apply(self, state) -> bool:
        if self.issues():
            return False
        maze = state.build_maze()
        self.apply_to_maze(maze)
        state.set_maze(maze)
        return self._finish_apply(state)


# ----------------------------------------------------------------------
# Y-maze / elevated plus maze
# ----------------------------------------------------------------------
class ArmEntryParamsPanel(MazeParamsPanel):
    """Minimum dwell time and the initial-arm rule (Y-maze and EPM)."""

    INFO = (
        "An arm entry is counted when the animal enters an arm from the "
        "center and stays long enough to be confirmed. Brief flickers "
        "across a zone border are ignored."
    )

    def __init__(self, maze=None, parent=None, preserve_exact: bool = True):
        super().__init__(parent, preserve_exact)
        self.info_label = QLabel(self.INFO)
        self.info_label.setWordWrap(True)

        low, high = MIN_DWELL_RANGE_S
        self.dwell_spin = QDoubleSpinBox(
            decimals=2, minimum=low, maximum=high, singleStep=0.05,
            value=0.10,
        )
        self.dwell_spin.setSuffix(" s")
        self.dwell_spin.setToolTip(
            "A zone change must last at least this long to count."
        )
        self.initial_check = QCheckBox(
            "Count the arm the animal starts in as an entry"
        )
        self.initial_check.setChecked(True)

        form = QFormLayout()
        form.addRow("Minimum dwell time:", self.dwell_spin)
        form.addRow(self.initial_check)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.info_label)
        layout.addLayout(form)

        self.dwell_spin.valueChanged.connect(self._emit_changed)
        self.initial_check.toggled.connect(self._emit_changed)
        if maze is not None:
            self.load_from_maze(maze)

    def load_from_maze(self, maze) -> None:
        self._loading = True
        try:
            self._exact.set(self.dwell_spin, float(maze.min_dwell_seconds))
            self.initial_check.setChecked(bool(maze.count_initial_arm))
        finally:
            self._loading = False

    def get_parameters(self) -> Dict[str, Any]:
        """Values in the shape ``configure_algorithm_parameters`` expects."""
        return {
            'min_dwell_seconds': float(self._exact.get(self.dwell_spin)),
            'count_initial_arm': self.initial_check.isChecked(),
        }

    def apply_to_maze(self, maze) -> None:
        maze.configure_algorithm_parameters(self.get_parameters())


# ----------------------------------------------------------------------
# TST
# ----------------------------------------------------------------------
class TstParamsPanel(MazeParamsPanel):
    """Mouse count, minimum immobility bout and the longest bridged gap."""

    def __init__(self, maze=None, parent=None, preserve_exact: bool = True):
        super().__init__(parent, preserve_exact)
        from mazes.tst import TST

        self.mice_spin = QSpinBox()
        self.mice_spin.setRange(1, TST.MAX_MICE)
        self.mice_spin.setToolTip(
            "How many mice are in the video; each gets its own region."
        )
        self.min_bout_spin = QDoubleSpinBox(
            decimals=2, minimum=0.0, maximum=60.0, singleStep=0.1,
            value=TST().min_time_seconds,
        )
        self.min_bout_spin.setSuffix(" s")
        self.min_bout_spin.setToolTip(
            "Immobility shorter than this is not counted as a bout."
        )
        self.max_gap_spin = QDoubleSpinBox(
            decimals=2, minimum=0.0, maximum=5.0, singleStep=0.05,
            value=TST().max_motion_gap_seconds,
        )
        self.max_gap_spin.setSuffix(" s")
        self.max_gap_spin.setToolTip(
            "A movement blip up to this long does not split an immobility "
            "bout."
        )
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Number of mice:", self.mice_spin)
        form.addRow("Minimum immobility bout:", self.min_bout_spin)
        form.addRow("Maximum bridged gap:", self.max_gap_spin)

        self.mice_spin.valueChanged.connect(self._emit_changed)
        self.min_bout_spin.valueChanged.connect(self._emit_changed)
        self.max_gap_spin.valueChanged.connect(self._emit_changed)
        if maze is not None:
            self.load_from_maze(maze)

    def load_from_maze(self, maze) -> None:
        self._loading = True
        try:
            self.mice_spin.setValue(
                min(max(int(maze.num_mice), self.mice_spin.minimum()),
                    self.mice_spin.maximum())
            )
            self._exact.set(self.min_bout_spin, float(maze.min_time_seconds))
            self._exact.set(
                self.max_gap_spin, float(maze.max_motion_gap_seconds)
            )
        finally:
            self._loading = False

    def apply_to_maze(self, maze) -> None:
        maze.configure_mice(self.mice_spin.value())
        maze.set_parameters(
            self._exact.get(self.min_bout_spin),
            self._exact.get(self.max_gap_spin),
        )


# ----------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------
class ParadigmParamsPanel(SetupPanel):
    """Shows the parameter panel of whichever paradigm the state holds."""

    step_id = STEP_PARADIGM_PARAMS

    def __init__(self, parent=None, preserve_exact: bool = True):
        super().__init__(parent, preserve_exact)
        from gui.setup_panels.freestyle_params import FreestyleParamsPanel

        self.panels: Dict[str, MazeParamsPanel] = {
            KIND_ARMS: ArmEntryParamsPanel(preserve_exact=preserve_exact),
            KIND_FREESTYLE: FreestyleParamsPanel(
                preserve_exact=preserve_exact),
            KIND_TST: TstParamsPanel(preserve_exact=preserve_exact),
        }
        self._empty = QLabel("This paradigm has no extra parameters.")
        self._empty.setWordWrap(True)
        self._stack = QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._stack.addWidget(self._empty)
        for panel in self.panels.values():
            self._stack.addWidget(panel)
            panel.changed.connect(self._emit_changed)
        self._active: SetupPanel = None

    @property
    def active_panel(self):
        """The paradigm panel currently shown (None: no parameters)."""
        return self._active

    def panel_for(self, kind: str):
        return self.panels.get(kind)

    def load(self, state) -> None:
        maze = state.build_maze()
        self._active = self.panels.get(maze.wizard_kind)
        self._loading = True
        try:
            if self._active is None:
                self._stack.setCurrentWidget(self._empty)
            else:
                self._stack.setCurrentWidget(self._active)
                self._active.load_from_maze(maze)
        finally:
            self._loading = False

    def apply(self, state) -> bool:
        if self._active is None:
            state.mark_done(self.step_id)
            return True
        return self._active.apply(state)

    def issues(self) -> List[str]:
        return self._active.issues() if self._active is not None else []
