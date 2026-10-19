"""Small dialog for the arm-entry rules shared by the Y-maze and the EPM.

The controls live in ``ArmEntryParamsPanel`` (shared with the workspace);
this dialog only hosts that panel with OK/Cancel.
"""
from typing import Any, Dict

from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout

from gui.setup_panels.paradigm_params import ArmEntryParamsPanel


class ArmEntryParametersDialog(QDialog):
    """Edit min_dwell_seconds and count_initial_arm of a Y-maze / EPM."""

    def __init__(self, maze, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{maze.name}: Arm Entry Rules")

        self.panel = ArmEntryParamsPanel(maze, preserve_exact=False)
        # The controls keep their historic dialog attribute names.
        self.dwell_spin = self.panel.dwell_spin
        self.initial_check = self.panel.initial_check

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.panel)
        layout.addWidget(buttons)

    def get_parameters(self) -> Dict[str, Any]:
        """Return values in the shape ``configure_algorithm_parameters`` expects."""
        return self.panel.get_parameters()
