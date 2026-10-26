"""
Freestyle Configuration Dialog.
Allows users to configure custom zones or choose zone-free tracking.

The controls live in ``FreestyleParamsPanel`` (shared with the workspace);
this dialog hosts that panel and adds the OK/Cancel flow: message boxes for
invalid zone names, the many-zones confirmation and the discard guard.
"""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDialog, QDialogButtonBox, QLabel, QMessageBox,
                             QScrollArea, QVBoxLayout, QWidget)
from gui import ux_helpers
from gui.dialog_layout import fit_dialog_to_available_screen
from gui.setup_panels.freestyle_params import (RESERVED_ZONE_NAMES,
                                               RESERVED_ZONE_PREFIX,
                                               FreestyleParamsPanel,
                                               internal_zone_name,
                                               is_reserved_zone_name)

__all__ = [
    "FreestyleConfigDialog", "RESERVED_ZONE_NAMES", "RESERVED_ZONE_PREFIX",
    "internal_zone_name", "is_reserved_zone_name",
]

# The panel's controls, exposed on the dialog under their historic names.
_PANEL_WIDGETS = (
    "zone_based_radio", "zone_free_radio", "mode_button_group",
    "immobility_threshold_spin", "speed_window_spin", "min_immobility_spin",
    "arena_checkbox", "arena_options", "center_distance_radio",
    "center_distance_spin", "center_area_radio", "center_area_spin",
    "center_mode_group", "center_note", "zone_config_group", "num_zones_spin",
    "min_dwell_spin", "zone_names_widget", "zone_names_layout",
    "zone_name_inputs", "zone_preview_group", "zone_list_widget",
    "info_label",
)
# ... and its behaviour.
_PANEL_METHODS = (
    "on_mode_changed", "on_arena_toggled", "on_num_zones_changed",
    "quick_fill", "update_zone_preview", "update_info_label", "center_mode",
    "center_area_fraction", "is_arena_enabled", "get_analysis_parameters",
    "load_from_maze",
)


class FreestyleConfigDialog(QDialog):
    """
    Dialog for configuring Freestyle/Open Field analysis.
    Allows users to define custom zones or proceed with zone-free tracking.
    """

    def __init__(self, parent=None, maze=None):
        """Create the dialog; pass the Freestyle maze to read and set its parameters."""
        super().__init__(parent)
        self.setWindowTitle("Freestyle / Open Field Configuration")

        self.maze = maze
        self.zone_names: list[tuple[str, str]] = []

        self.panel = FreestyleParamsPanel(preserve_exact=False)
        for name in _PANEL_WIDGETS:
            setattr(self, name, getattr(self.panel, name))
        for name in _PANEL_METHODS:
            setattr(self, name, getattr(self.panel, name))

        self.init_ui()
        if maze is not None:
            self.load_from_maze(maze)
        self._initial_state = self._current_state()

    @property
    def zone_free_mode(self) -> bool:
        return self.panel.zone_free_mode

    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()

        # Title and description
        title_label = QLabel("<h2>Freestyle / Open Field Configuration</h2>")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        description = QLabel(
            "Configure custom zones for your analysis, or choose zone-free tracking "
            "to focus purely on movement metrics without spatial regions."
        )
        description.setWordWrap(True)
        description.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        main_layout.addWidget(description)

        main_layout.addWidget(self.panel)

        # Button box
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.validate_and_accept)
        self.button_box.rejected.connect(self.reject)
        content_widget = QWidget()
        content_widget.setLayout(main_layout)
        self.content_scroll = QScrollArea()
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setWidget(content_widget)

        root_layout = QVBoxLayout()
        root_layout.addWidget(self.content_scroll, 1)
        root_layout.addWidget(self.button_box)
        self.setLayout(root_layout)
        fit_dialog_to_available_screen(
            self,
            preferred_size=(600, 500),
            minimum_size=(420, 340),
        )

    def validate_and_accept(self):
        """Validate the configuration before accepting."""
        self.zone_names = []
        problem = self.panel.first_zone_problem()
        if problem is not None:
            title, message, field = problem
            QMessageBox.warning(self, title, message)
            field.setFocus()
            return
        if self.panel.is_zone_free_mode():
            # Zone-free mode is always valid
            self._finish()
            return

        self.zone_names = self.panel.zone_definitions()

        # Confirm if many zones
        if len(self.zone_names) > 10:
            reply = QMessageBox.question(
                self, "Many Zones",
                f"You are defining {len(self.zone_names)} zones, which may make "
                f"ROI drawing time-consuming and analysis complex.\n\n"
                f"Are you sure you want to proceed?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._finish()

    def _finish(self):
        """Push parameters into the maze (if any) and close the dialog."""
        if self.maze is not None:
            self.apply_to_maze(self.maze)
        self.accept()

    def apply_to_maze(self, maze) -> None:
        """Set the dialog's analysis parameters on a Freestyle maze."""
        self.panel.apply_analysis_to_maze(maze)

    def _current_state(self) -> tuple:
        return self.panel.current_state()

    def has_unsaved_work(self) -> bool:
        return ux_helpers.should_confirm_discard(
            self._initial_state, self._current_state()
        )

    def reject(self):
        if self.has_unsaved_work() and not ux_helpers.confirm_discard(
            self, "this Freestyle configuration"
        ):
            return
        super().reject()

    def get_zone_definitions(self) -> list[tuple[str, str]]:
        """
        Get the configured zone definitions.

        Returns:
            List of (internal_name, display_name) tuples
            Empty list if zone-free mode
        """
        return self.zone_names

    def is_zone_free_mode(self) -> bool:
        """Check if zone-free mode is selected."""
        return self.panel.zone_free_mode
