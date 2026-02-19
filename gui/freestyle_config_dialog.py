"""
Freestyle Configuration Dialog.
Allows users to configure custom zones or choose zone-free tracking.
"""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QFormLayout,
                             QGroupBox, QLabel, QLineEdit, QListWidget, 
                             QPushButton, QScrollArea, QSpinBox, QVBoxLayout,
                             QHBoxLayout, QWidget, QMessageBox)


class FreestyleConfigDialog(QDialog):
    """
    Dialog for configuring Freestyle/Open Field analysis.
    Allows users to define custom zones or proceed with zone-free tracking.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Freestyle / Open Field Configuration")
        self.setMinimumSize(600, 500)
        
        self.zone_names: list[str] = []
        self.zone_free_mode = False
        
        self.init_ui()
    
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
        
        self.zone_based_radio = QCheckBox("Zone-based tracking (define custom regions)")
        self.zone_based_radio.setChecked(True)
        self.zone_based_radio.toggled.connect(self.on_mode_changed)
        
        self.zone_free_radio = QCheckBox("Zone-free tracking (pure movement analysis)")
        self.zone_free_radio.toggled.connect(self.on_mode_changed)
        
        # Make them mutually exclusive
        self.zone_based_radio.toggled.connect(
            lambda checked: self.zone_free_radio.setChecked(not checked) if checked else None
        )
        self.zone_free_radio.toggled.connect(
            lambda checked: self.zone_based_radio.setChecked(not checked) if checked else None
        )
        
        mode_layout.addWidget(self.zone_based_radio)
        mode_layout.addWidget(self.zone_free_radio)
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)
        
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
        
        # Zone naming section
        zone_config_layout.addWidget(QLabel("Zone names (you will draw these regions next):"))
        
        # Scrollable area for zone name inputs
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(200)
        
        self.zone_names_widget = QWidget()
        self.zone_names_layout = QVBoxLayout(self.zone_names_widget)
        self.zone_name_inputs = []
        
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
        main_layout.addWidget(self.button_box)
        
        self.setLayout(main_layout)
        
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
        """Handle change in number of zones."""
        # Clear existing inputs properly
        while self.zone_names_layout.count():
            item = self.zone_names_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.zone_name_inputs.clear()
        
        # Create new inputs
        for i in range(count):
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
        
        # Don't add stretch here - it was causing issues
        self.update_zone_preview()
        self.update_info_label()
    
    def quick_fill(self, pattern: str):
        """Quick fill zone names with a pattern."""
        count = len(self.zone_name_inputs)
        
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
            self.accept()
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
            internal_name = name.lower().replace(' ', '_')
            
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
        
        self.accept()
    
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