"""
Freestyle Configuration Dialog.
Allows users to configure custom zones or choose zone-free tracking.
"""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QButtonGroup, QDialog, QDialogButtonBox,
                             QDoubleSpinBox, QFormLayout, QGroupBox, QLabel,
                             QLineEdit, QListWidget, QPushButton, QRadioButton,
                             QScrollArea, QSpinBox, QVBoxLayout, QHBoxLayout,
                             QWidget, QMessageBox)
from gui import ux_helpers
from gui.dialog_layout import fit_dialog_to_available_screen

# Internal labels the tracking pipeline reserves for non-zone states; "mouse_"
# prefixes are used for per-animal labels (e.g. TST detection).
RESERVED_ZONE_NAMES = frozenset({'outside', 'undetected'})
RESERVED_ZONE_PREFIX = 'mouse_'


def internal_zone_name(display_name: str) -> str:
    """Return the internal (lowercase, underscored) form of a zone name."""
    return display_name.strip().lower().replace(' ', '_')


def is_reserved_zone_name(display_name: str) -> bool:
    """True if a name collides with an internal tracking label."""
    internal = internal_zone_name(display_name)
    return (
        internal in RESERVED_ZONE_NAMES
        or internal.startswith(RESERVED_ZONE_PREFIX)
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
        self.zone_free_mode = False

        self.init_ui()
        if maze is not None:
            self.load_from_maze(maze)
        self._initial_state = self._current_state()

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
        
        # Mode selection
        mode_group = QGroupBox("Tracking Mode")
        mode_layout = QVBoxLayout()
        
        # Exclusive radio buttons: exactly one mode is always selected.
        self.zone_based_radio = QRadioButton("Zone-based tracking (define custom regions)")
        self.zone_based_radio.setChecked(True)

        self.zone_free_radio = QRadioButton("Zone-free tracking (pure movement analysis)")

        self.mode_button_group = QButtonGroup(self)
        self.mode_button_group.setExclusive(True)
        self.mode_button_group.addButton(self.zone_based_radio)
        self.mode_button_group.addButton(self.zone_free_radio)
        self.zone_based_radio.toggled.connect(self.on_mode_changed)

        mode_layout.addWidget(self.zone_based_radio)
        mode_layout.addWidget(self.zone_free_radio)
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)
        
        # Movement / immobility parameters
        movement_group = QGroupBox("Movement Parameters")
        movement_form = QFormLayout()

        self.immobility_threshold_spin = QDoubleSpinBox()
        self.immobility_threshold_spin.setRange(0.0, 50.0)
        self.immobility_threshold_spin.setDecimals(2)
        self.immobility_threshold_spin.setSingleStep(0.1)
        self.immobility_threshold_spin.setSuffix(" cm/s")
        self.immobility_threshold_spin.setValue(2.0)
        self.immobility_threshold_spin.setToolTip(
            "Smoothed speed below this value counts as immobile. Applied only "
            "when a reference line calibrates the video; uncalibrated runs "
            "report immobility as N/A."
        )
        movement_form.addRow("Immobility threshold:", self.immobility_threshold_spin)

        self.speed_window_spin = QDoubleSpinBox()
        self.speed_window_spin.setRange(0.05, 2.0)
        self.speed_window_spin.setDecimals(2)
        self.speed_window_spin.setSingleStep(0.05)
        self.speed_window_spin.setSuffix(" s")
        self.speed_window_spin.setValue(0.2)
        self.speed_window_spin.setToolTip(
            "Speed is the displacement across this centered time window, which "
            "suppresses centroid jitter. Longer windows give smoother but "
            "less responsive speed."
        )
        movement_form.addRow("Speed smoothing window:", self.speed_window_spin)

        self.min_immobility_spin = QDoubleSpinBox()
        self.min_immobility_spin.setRange(0.0, 30.0)
        self.min_immobility_spin.setDecimals(2)
        self.min_immobility_spin.setSingleStep(0.1)
        self.min_immobility_spin.setSuffix(" s")
        self.min_immobility_spin.setValue(1.0)
        self.min_immobility_spin.setToolTip(
            "Speed must stay below the threshold for at least this long to "
            "count as an immobility bout. Shorter pauses are scored as movement."
        )
        movement_form.addRow("Minimum immobility bout:", self.min_immobility_spin)

        movement_group.setLayout(movement_form)
        main_layout.addWidget(movement_group)

        # Zone configuration section
        self.zone_config_group = QGroupBox("Zone Configuration")
        zone_config_layout = QVBoxLayout()
        
        # Number of zones
        num_zones_layout = QHBoxLayout()
        num_zones_layout.addWidget(QLabel("Number of zones to define:"))
        self.num_zones_spin = QSpinBox()
        self.num_zones_spin.setRange(1, 20)
        self.num_zones_spin.setValue(3)
        self.num_zones_spin.setToolTip("Choose how many distinct zones/regions you want to define")
        self.num_zones_spin.valueChanged.connect(self.on_num_zones_changed)
        num_zones_layout.addWidget(self.num_zones_spin)
        num_zones_layout.addStretch()
        zone_config_layout.addLayout(num_zones_layout)
        
        # Minimum dwell before a zone change is accepted
        dwell_layout = QHBoxLayout()
        dwell_layout.addWidget(QLabel("Minimum zone dwell:"))
        self.min_dwell_spin = QDoubleSpinBox()
        self.min_dwell_spin.setRange(0.0, 5.0)
        self.min_dwell_spin.setDecimals(2)
        self.min_dwell_spin.setSingleStep(0.05)
        self.min_dwell_spin.setSuffix(" s")
        self.min_dwell_spin.setValue(0.10)
        self.min_dwell_spin.setToolTip(
            "A zone change is counted only after the animal stays in the new "
            "zone this long, which filters boundary flicker. 0 disables it."
        )
        dwell_layout.addWidget(self.min_dwell_spin)
        dwell_layout.addStretch()
        zone_config_layout.addLayout(dwell_layout)

        # Zone naming section
        zone_config_layout.addWidget(QLabel("Zone names (you will draw these regions next):"))
        
        # Scrollable area for zone name inputs
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(200)
        
        self.zone_names_widget = QWidget()
        self.zone_names_layout = QVBoxLayout(self.zone_names_widget)
        self.zone_name_inputs: list[QLineEdit] = []
        self._zone_name_rows: list[QWidget] = []
        
        scroll_area.setWidget(self.zone_names_widget)
        zone_config_layout.addWidget(scroll_area)
        
        # Quick fill buttons
        quick_fill_layout = QHBoxLayout()
        quick_fill_layout.addWidget(QLabel("Quick fill:"))
        
        btn_letters = QPushButton("A, B, C...")
        btn_letters.setToolTip("Fill with letters: Zone A, Zone B, Zone C...")
        btn_letters.clicked.connect(lambda: self.quick_fill('letters'))
        
        btn_numbers = QPushButton("1, 2, 3...")
        btn_numbers.setToolTip("Fill with numbers: Zone 1, Zone 2, Zone 3...")
        btn_numbers.clicked.connect(lambda: self.quick_fill('numbers'))
        
        btn_cardinal = QPushButton("N, S, E, W...")
        btn_cardinal.setToolTip("Fill with cardinal directions (up to 8 zones)")
        btn_cardinal.clicked.connect(lambda: self.quick_fill('cardinal'))
        
        quick_fill_layout.addWidget(btn_letters)
        quick_fill_layout.addWidget(btn_numbers)
        quick_fill_layout.addWidget(btn_cardinal)
        quick_fill_layout.addStretch()
        
        zone_config_layout.addLayout(quick_fill_layout)
        
        self.zone_config_group.setLayout(zone_config_layout)
        main_layout.addWidget(self.zone_config_group)
        
        # Zone preview list
        self.zone_preview_group = QGroupBox("Zone Summary")
        preview_layout = QVBoxLayout()
        self.zone_list_widget = QListWidget()
        self.zone_list_widget.setMaximumHeight(100)
        preview_layout.addWidget(self.zone_list_widget)
        self.zone_preview_group.setLayout(preview_layout)
        main_layout.addWidget(self.zone_preview_group)
        
        # Info label
        self.info_label = QLabel()
        self.info_label.setStyleSheet("color: #0066cc; font-style: italic; padding: 5px;")
        main_layout.addWidget(self.info_label)
        
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
        
        # Initialize with default zones
        self.on_num_zones_changed(3)
        self.quick_fill('letters')
        self.update_info_label()
    
    def on_mode_changed(self):
        """Handle mode selection change."""
        zone_based = self.zone_based_radio.isChecked()
        self.zone_free_mode = not zone_based
        
        # Enable/disable zone configuration
        self.zone_config_group.setEnabled(zone_based)
        self.zone_preview_group.setEnabled(zone_based)
        
        self.update_info_label()
    
    def on_num_zones_changed(self, count: int):
        """Resize the name list, keeping names the user already typed."""
        # Drop surplus rows from the end
        while len(self.zone_name_inputs) > count:
            self.zone_name_inputs.pop()
            row = self._zone_name_rows.pop()
            self.zone_names_layout.removeWidget(row)
            row.deleteLater()

        # Append rows for new zones
        for i in range(len(self.zone_name_inputs), count):
            zone_layout = QHBoxLayout()

            label = QLabel(f"Zone {i + 1}:")
            label.setMinimumWidth(60)

            line_edit = QLineEdit()
            line_edit.setPlaceholderText(f"e.g., Zone {chr(65 + i)}")
            line_edit.textChanged.connect(self.update_zone_preview)

            zone_layout.addWidget(label)
            zone_layout.addWidget(line_edit)

            container = QWidget()
            container.setLayout(zone_layout)

            self.zone_names_layout.addWidget(container)
            self.zone_name_inputs.append(line_edit)
            self._zone_name_rows.append(container)

        self.update_zone_preview()
        self.update_info_label()

    def quick_fill(self, pattern: str):
        """Quick fill zone names with a pattern."""
        if pattern == 'letters':
            for i, input_field in enumerate(self.zone_name_inputs):
                if i < 26:
                    input_field.setText(f"Zone {chr(65 + i)}")
                else:
                    input_field.setText(f"Zone {i + 1}")
        
        elif pattern == 'numbers':
            for i, input_field in enumerate(self.zone_name_inputs):
                input_field.setText(f"Zone {i + 1}")
        
        elif pattern == 'cardinal':
            directions = ['North', 'South', 'East', 'West', 
                         'Northeast', 'Northwest', 'Southeast', 'Southwest']
            for i, input_field in enumerate(self.zone_name_inputs):
                if i < len(directions):
                    input_field.setText(directions[i])
                else:
                    input_field.setText(f"Zone {i + 1}")
        
        self.update_zone_preview()
    
    def update_zone_preview(self):
        """Update the zone preview list."""
        self.zone_list_widget.clear()
        
        for i, input_field in enumerate(self.zone_name_inputs):
            text = input_field.text().strip()
            if text:
                self.zone_list_widget.addItem(f"{i + 1}. {text}")
            else:
                self.zone_list_widget.addItem(f"{i + 1}. (unnamed)")
    
    def update_info_label(self):
        """Update the information label based on current configuration."""
        if self.zone_free_mode:
            self.info_label.setText(
                "ℹ️ Zone-free mode: You will track movement without defining spatial regions. "
                "Analysis will focus on speed, distance, and movement patterns."
            )
        else:
            count = self.num_zones_spin.value()
            self.info_label.setText(
                f"ℹ️ You will draw {count} custom zone{'s' if count != 1 else ''} "
                f"on your video in the next step."
            )
    
    def validate_and_accept(self):
        """Validate the configuration before accepting."""
        if self.zone_free_mode:
            # Zone-free mode is always valid
            self.zone_names = []
            self._finish()
            return
        
        # Validate zone names
        self.zone_names = []
        seen_names = set()
        
        for i, input_field in enumerate(self.zone_name_inputs):
            name = input_field.text().strip()
            
            if not name:
                QMessageBox.warning(
                    self, "Empty Zone Name",
                    f"Zone {i + 1} has no name. Please provide a name or reduce the number of zones."
                )
                input_field.setFocus()
                return
            
            # Convert to internal format (lowercase with underscores)
            internal_name = internal_zone_name(name)

            if is_reserved_zone_name(name):
                QMessageBox.warning(
                    self, "Reserved Zone Name",
                    f"Zone name '{name}' is reserved for internal tracking "
                    f"states ('outside', 'undetected', or names starting with "
                    f"'mouse_'). Please choose a different name."
                )
                input_field.setFocus()
                self.zone_names = []
                return
            
            # Check for duplicates
            if internal_name in seen_names:
                QMessageBox.warning(
                    self, "Duplicate Zone Name",
                    f"Zone name '{name}' is used more than once. Please use unique names."
                )
                input_field.setFocus()
                return
            
            seen_names.add(internal_name)
            self.zone_names.append((internal_name, name))
        
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

    def load_from_maze(self, maze) -> None:
        """Show the maze's current analysis parameters."""
        self.immobility_threshold_spin.setValue(maze.immobility_threshold_cm_per_s)
        self.speed_window_spin.setValue(maze.speed_window_seconds)
        self.min_immobility_spin.setValue(maze.min_immobility_seconds)
        self.min_dwell_spin.setValue(maze.min_dwell_seconds)

    def get_analysis_parameters(self) -> dict:
        """Return the movement and zone-dwell parameters as Freestyle keys."""
        return {
            'immobility_threshold_cm_per_s': self.immobility_threshold_spin.value(),
            'speed_window_seconds': self.speed_window_spin.value(),
            'min_immobility_seconds': self.min_immobility_spin.value(),
            'min_dwell_seconds': self.min_dwell_spin.value(),
        }

    def apply_to_maze(self, maze) -> None:
        """Set the dialog's analysis parameters on a Freestyle maze."""
        maze.set_immobility_threshold(self.immobility_threshold_spin.value())
        maze.speed_window_seconds = self.speed_window_spin.value()
        maze.min_immobility_seconds = self.min_immobility_spin.value()
        maze.min_dwell_seconds = self.min_dwell_spin.value()

    def _current_state(self) -> tuple:
        return (
            self.zone_free_radio.isChecked(),
            self.num_zones_spin.value(),
            tuple(field.text() for field in self.zone_name_inputs),
            tuple(sorted(self.get_analysis_parameters().items())),
            self.min_dwell_spin.value(),
        )

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
        return self.zone_free_mode
