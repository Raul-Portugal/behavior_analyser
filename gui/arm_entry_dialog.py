"""Small dialog for the arm-entry rules shared by the Y-maze and the EPM."""
from typing import Any, Dict

from PyQt6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox,
                             QDoubleSpinBox, QFormLayout, QLabel, QVBoxLayout)


class ArmEntryParametersDialog(QDialog):
    """Edit min_dwell_seconds and count_initial_arm of a Y-maze / EPM."""

    def __init__(self, maze, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{maze.name}: Arm Entry Rules")

        info = QLabel(
            "An arm entry is counted when the animal enters an arm from the "
            "center and stays long enough to be confirmed. Brief flickers "
            "across a zone border are ignored."
        )
        info.setWordWrap(True)

        self.dwell_spin = QDoubleSpinBox(
            decimals=2, minimum=0.0, maximum=10.0, singleStep=0.05,
            value=float(maze.min_dwell_seconds)
        )
        self.dwell_spin.setSuffix(" s")
        self.dwell_spin.setToolTip(
            "A zone change must last at least this long to count."
        )
        self.initial_check = QCheckBox("Count the arm the animal starts in as an entry")
        self.initial_check.setChecked(bool(maze.count_initial_arm))

        form = QFormLayout()
        form.addRow("Minimum dwell time:", self.dwell_spin)
        form.addRow(self.initial_check)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(info)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def get_parameters(self) -> Dict[str, Any]:
        """Return values in the shape ``configure_algorithm_parameters`` expects."""
        return {
            'min_dwell_seconds': float(self.dwell_spin.value()),
            'count_initial_arm': self.initial_check.isChecked(),
        }
