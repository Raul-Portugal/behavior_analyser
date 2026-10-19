"""Embeddable setup panels shared by the classic dialogs and the workspace.

Every panel is a ``QWidget`` with ``load(state)``, ``apply(state)``,
``issues()`` and a ``changed`` signal (see ``SetupPanel``), editing one step
of a ``core.setup_state.SetupState``.
"""
from gui.setup_panels.base import ExactValues, SetupPanel
from gui.setup_panels.calibration_panel import CalibrationPanel
from gui.setup_panels.freestyle_params import FreestyleParamsPanel
from gui.setup_panels.paradigm_params import (ArmEntryParamsPanel,
                                              MazeParamsPanel,
                                              ParadigmParamsPanel,
                                              TstParamsPanel)
from gui.setup_panels.review_panel import ReviewPanel
from gui.setup_panels.time_window_panel import TimeWindowPanel

__all__ = [
    "ArmEntryParamsPanel", "CalibrationPanel", "ExactValues",
    "FreestyleParamsPanel", "MazeParamsPanel", "ParadigmParamsPanel",
    "ReviewPanel", "SetupPanel", "TimeWindowPanel", "TstParamsPanel",
]
