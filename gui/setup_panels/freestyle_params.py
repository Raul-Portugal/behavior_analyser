"""Open-field parameters: zones, movement, smoothing and the arena."""
from typing import Dict, List, Optional, Tuple

from PyQt6.QtWidgets import (QButtonGroup, QCheckBox, QDoubleSpinBox,
                             QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QListWidget, QPushButton,
                             QRadioButton, QScrollArea, QSpinBox, QVBoxLayout,
                             QWidget)

from gui.setup_panels.paradigm_params import MazeParamsPanel
from mazes.freestyle import (CENTER_BY_AREA_FRACTION, CENTER_BY_DISTANCE,
                             DEFAULT_CENTER_AREA_FRACTION,
                             DEFAULT_CENTER_DISTANCE_CM)

# Internal labels the tracking pipeline reserves for non-zone states; "mouse_"
# prefixes are used for per-animal labels (e.g. TST detection). "arena" is the
# open-field arena polygon, which is stored apart from the zones.
RESERVED_ZONE_NAMES = frozenset({'outside', 'undetected', 'arena'})
RESERVED_ZONE_PREFIX = 'mouse_'

# Largest zone count the name list offers (more are kept if a maze has them).
DEFAULT_MAX_ZONES = 20
DEFAULT_ZONE_COUNT = 3


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


# (dialog title, message, the line edit at fault)
ZoneProblem = Tuple[str, str, QLineEdit]


class FreestyleParamsPanel(MazeParamsPanel):
    """Zone names, movement/immobility, zone dwell and arena options."""

    def __init__(self, maze=None, parent=None, preserve_exact: bool = True):
        super().__init__(parent, preserve_exact)
        self._loaded_internal: Dict[str, str] = {}
        self.zone_free_mode = False
        self._build_ui()
        if maze is not None:
            self.load_from_maze(maze)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Mode selection
        mode_group = QGroupBox("Tracking Mode")
        mode_layout = QVBoxLayout()

        # Exclusive radio buttons: exactly one mode is always selected.
        self.zone_based_radio = QRadioButton(
            "Zone-based tracking (define custom regions)"
        )
        self.zone_based_radio.setChecked(True)
        self.zone_free_radio = QRadioButton(
            "Zone-free tracking (pure movement analysis)"
        )
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
        movement_form.addRow("Immobility threshold:",
                             self.immobility_threshold_spin)

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
        movement_form.addRow("Minimum immobility bout:",
                             self.min_immobility_spin)

        movement_group.setLayout(movement_form)
        main_layout.addWidget(movement_group)

        # Open field arena: center/periphery zones and thigmotaxis
        arena_group = QGroupBox("Open Field Arena")
        arena_layout = QVBoxLayout()
        self.arena_checkbox = QCheckBox("Open field arena (center/periphery)")
        self.arena_checkbox.setToolTip(
            "Draw the arena boundary first. The center is derived from it "
            "automatically; reports center/periphery time, center entries, "
            "latency to center and thigmotaxis. Detection is also restricted "
            "to the arena."
        )
        arena_layout.addWidget(self.arena_checkbox)

        self.arena_options = QWidget()
        options_layout = QVBoxLayout(self.arena_options)
        options_layout.setContentsMargins(20, 0, 0, 0)
        self.center_distance_radio = QRadioButton(
            "Center starts this far from the walls:"
        )
        self.center_distance_radio.setChecked(True)
        self.center_distance_spin = QDoubleSpinBox()
        self.center_distance_spin.setRange(0.5, 200.0)
        self.center_distance_spin.setDecimals(1)
        self.center_distance_spin.setSingleStep(0.5)
        self.center_distance_spin.setSuffix(" cm")
        self.center_distance_spin.setValue(DEFAULT_CENTER_DISTANCE_CM)
        self.center_distance_spin.setToolTip(
            "Needs a reference line to calibrate the video. Without one the "
            "area rule below is used instead."
        )
        self.center_area_radio = QRadioButton(
            "Center covers this share of the arena area:"
        )
        self.center_area_spin = QDoubleSpinBox()
        self.center_area_spin.setRange(1.0, 99.0)
        self.center_area_spin.setDecimals(0)
        self.center_area_spin.setSingleStep(5.0)
        self.center_area_spin.setSuffix(" %")
        self.center_area_spin.setValue(100.0 * DEFAULT_CENTER_AREA_FRACTION)
        self.center_area_spin.setToolTip(
            "The center is the arena shrunk until it covers this share of "
            "its area. Works without calibration."
        )
        self.center_mode_group = QButtonGroup(self)
        self.center_mode_group.setExclusive(True)
        self.center_mode_group.addButton(self.center_distance_radio)
        self.center_mode_group.addButton(self.center_area_radio)
        for radio, spin in (
            (self.center_distance_radio, self.center_distance_spin),
            (self.center_area_radio, self.center_area_spin),
        ):
            row = QHBoxLayout()
            row.addWidget(radio)
            row.addWidget(spin)
            row.addStretch()
            options_layout.addLayout(row)
        self.center_note = QLabel(
            "Without a reference line (no calibration) the center is defined "
            "by area share instead."
        )
        self.center_note.setWordWrap(True)
        self.center_note.setStyleSheet("color: #555; font-style: italic;")
        options_layout.addWidget(self.center_note)
        arena_layout.addWidget(self.arena_options)
        arena_group.setLayout(arena_layout)
        main_layout.addWidget(arena_group)
        self.arena_options.setEnabled(False)
        self.arena_checkbox.toggled.connect(self.on_arena_toggled)

        # Zone configuration section
        self.zone_config_group = QGroupBox("Zone Configuration")
        zone_config_layout = QVBoxLayout()

        # Number of zones
        num_zones_layout = QHBoxLayout()
        num_zones_layout.addWidget(QLabel("Number of zones to define:"))
        self.num_zones_spin = QSpinBox()
        self.num_zones_spin.setRange(1, DEFAULT_MAX_ZONES)
        self.num_zones_spin.setValue(DEFAULT_ZONE_COUNT)
        self.num_zones_spin.setToolTip(
            "Choose how many distinct zones/regions you want to define"
        )
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
        zone_config_layout.addWidget(
            QLabel("Zone names (you will draw these regions next):")
        )

        # Scrollable area for zone name inputs
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(200)

        self.zone_names_widget = QWidget()
        self.zone_names_layout = QVBoxLayout(self.zone_names_widget)
        self.zone_name_inputs: List[QLineEdit] = []
        self._zone_name_rows: List[QWidget] = []

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
        self.info_label.setStyleSheet(
            "color: #0066cc; font-style: italic; padding: 5px;"
        )
        main_layout.addWidget(self.info_label)

        # Any edit is a change (connected last so building stays silent).
        for spin in (
            self.immobility_threshold_spin, self.speed_window_spin,
            self.min_immobility_spin, self.center_distance_spin,
            self.center_area_spin, self.min_dwell_spin,
        ):
            spin.valueChanged.connect(self._emit_changed)
        self.num_zones_spin.valueChanged.connect(self._emit_changed)
        self.arena_checkbox.toggled.connect(self._emit_changed)
        self.zone_based_radio.toggled.connect(self._emit_changed)
        self.center_area_radio.toggled.connect(self._emit_changed)

        # Initialize with default zones
        self.on_num_zones_changed(DEFAULT_ZONE_COUNT)
        self.quick_fill('letters')
        self.update_info_label()

    # ------------------------------------------------------------------
    # Behaviour
    # ------------------------------------------------------------------
    def on_mode_changed(self):
        """Handle mode selection change."""
        zone_based = self.zone_based_radio.isChecked()
        self.zone_free_mode = not zone_based

        # Enable/disable zone configuration
        self.zone_config_group.setEnabled(zone_based)
        self.zone_preview_group.setEnabled(zone_based)

        self.update_info_label()

    def on_arena_toggled(self, checked: bool):
        """Enable the center-definition controls only while the arena is on."""
        self.arena_options.setEnabled(checked)
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
            line_edit.textChanged.connect(self._emit_changed)

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

    def update_zone_preview(self, *_args):
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
        arena_text = (
            " You will first draw the arena boundary; the center and "
            "periphery are derived from it."
            if self.arena_checkbox.isChecked() else ""
        )
        if self.zone_free_mode:
            self.info_label.setText(
                "ℹ️ Zone-free mode: You will track movement without defining spatial regions. "
                "Analysis will focus on speed, distance, and movement patterns."
                + arena_text
            )
        else:
            count = self.num_zones_spin.value()
            self.info_label.setText(
                f"ℹ️ You will draw {count} custom zone{'s' if count != 1 else ''} "
                f"on your video in the next step." + arena_text
            )

    # ------------------------------------------------------------------
    # Zones
    # ------------------------------------------------------------------
    def zone_problems(self) -> List[ZoneProblem]:
        """Every reason the zone names cannot be used (none in zone-free)."""
        if self.zone_free_mode:
            return []
        problems: List[ZoneProblem] = []
        seen = set()
        for i, input_field in enumerate(self.zone_name_inputs):
            name = input_field.text().strip()
            if not name:
                problems.append((
                    "Empty Zone Name",
                    f"Zone {i + 1} has no name. Please provide a name or "
                    "reduce the number of zones.",
                    input_field,
                ))
                continue
            if is_reserved_zone_name(name):
                problems.append((
                    "Reserved Zone Name",
                    f"Zone name '{name}' is reserved for internal tracking "
                    f"states ('outside', 'undetected', 'arena', or names "
                    f"starting with 'mouse_'). Please choose a different name.",
                    input_field,
                ))
                continue
            internal = self._internal_name(name)
            if internal in seen:
                problems.append((
                    "Duplicate Zone Name",
                    f"Zone name '{name}' is used more than once. Please use "
                    "unique names.",
                    input_field,
                ))
                continue
            seen.add(internal)
        return problems

    def _internal_name(self, display_name: str) -> str:
        """Internal name: as loaded if the user left the name alone."""
        name = display_name.strip()
        return self._loaded_internal.get(name, internal_zone_name(name))

    def zone_definitions(self) -> List[Tuple[str, str]]:
        """``(internal, display)`` pairs; empty in zone-free mode."""
        if self.zone_free_mode:
            return []
        return [
            (self._internal_name(field.text()), field.text().strip())
            for field in self.zone_name_inputs
        ]

    def is_zone_free_mode(self) -> bool:
        return self.zone_free_mode

    def issues(self) -> List[str]:
        return [message for _title, message, _field in self.zone_problems()]

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    def center_mode(self) -> str:
        """The selected center rule (a Freestyle ``CENTER_BY_*`` value)."""
        return (
            CENTER_BY_AREA_FRACTION if self.center_area_radio.isChecked()
            else CENTER_BY_DISTANCE
        )

    def center_area_fraction(self) -> float:
        value = self._exact.get(self.center_area_spin)
        return value / 100.0

    def is_arena_enabled(self) -> bool:
        """True when the open-field arena step was requested."""
        return self.arena_checkbox.isChecked()

    def load_from_maze(self, maze) -> None:
        """Show the maze's analysis parameters and, if set, its zones."""
        self._loading = True
        try:
            self._exact.set(
                self.immobility_threshold_spin,
                maze.immobility_threshold_cm_per_s,
            )
            self._exact.set(self.speed_window_spin, maze.speed_window_seconds)
            self._exact.set(
                self.min_immobility_spin, maze.min_immobility_seconds
            )
            self._exact.set(self.min_dwell_spin, maze.min_dwell_seconds)
            self.arena_checkbox.setChecked(maze.arena_mode)
            self._exact.set(
                self.center_distance_spin, maze.arena_center_distance_cm
            )
            self._exact.set(
                self.center_area_spin, 100.0 * maze.arena_center_area_fraction
            )
            by_area = maze.arena_center_mode == CENTER_BY_AREA_FRACTION
            self.center_area_radio.setChecked(by_area)
            self.center_distance_radio.setChecked(not by_area)
            self._load_zones(maze)
        finally:
            self._loading = False

    def _load_zones(self, maze) -> None:
        self._loaded_internal = {}
        if not maze.zones_configured:
            # Nothing chosen yet: back to the defaults.
            self.zone_based_radio.setChecked(True)
            self.num_zones_spin.setValue(DEFAULT_ZONE_COUNT)
            self.quick_fill('letters')
            return
        zones = list(maze.user_defined_zones)
        if not zones:
            self.zone_free_radio.setChecked(True)
            return
        self.zone_based_radio.setChecked(True)
        self.num_zones_spin.setMaximum(max(DEFAULT_MAX_ZONES, len(zones)))
        self.num_zones_spin.setValue(len(zones))
        for field, (internal, display) in zip(self.zone_name_inputs, zones):
            field.setText(display)
            self._loaded_internal[display.strip()] = internal

    def get_analysis_parameters(self) -> dict:
        """Return the movement and zone-dwell parameters as Freestyle keys.

        The arena keys are included only while the arena is enabled, matching
        ``Freestyle.get_algorithm_parameters``.
        """
        parameters = {
            'immobility_threshold_cm_per_s':
                self._exact.get(self.immobility_threshold_spin),
            'speed_window_seconds': self._exact.get(self.speed_window_spin),
            'min_immobility_seconds':
                self._exact.get(self.min_immobility_spin),
            'min_dwell_seconds': self._exact.get(self.min_dwell_spin),
        }
        if self.is_arena_enabled():
            parameters.update({
                'arena_mode': True,
                'arena_center_mode': self.center_mode(),
                'arena_center_distance_cm':
                    self._exact.get(self.center_distance_spin),
                'arena_center_area_fraction': self.center_area_fraction(),
            })
        return parameters

    def apply_analysis_to_maze(self, maze) -> None:
        """Set the movement, dwell and arena parameters (not the zones)."""
        maze.set_immobility_threshold(
            self._exact.get(self.immobility_threshold_spin)
        )
        maze.speed_window_seconds = self._exact.get(self.speed_window_spin)
        maze.min_immobility_seconds = self._exact.get(self.min_immobility_spin)
        maze.min_dwell_seconds = self._exact.get(self.min_dwell_spin)
        maze.configure_arena(
            self.is_arena_enabled(),
            self.center_mode(),
            self._exact.get(self.center_distance_spin),
            self.center_area_fraction(),
        )

    def apply_to_maze(self, maze) -> None:
        """Set every parameter and the zones on a Freestyle maze."""
        self.apply_analysis_to_maze(maze)
        maze.configure_zones(self.zone_definitions())

    def current_state(self) -> tuple:
        """Everything the user can edit, for unsaved-work detection."""
        return (
            self.zone_free_radio.isChecked(),
            self.num_zones_spin.value(),
            tuple(field.text() for field in self.zone_name_inputs),
            tuple(sorted(self.get_analysis_parameters().items())),
            self.min_dwell_spin.value(),
            self.is_arena_enabled(),
            self.center_mode(),
            self.center_distance_spin.value(),
            self.center_area_spin.value(),
        )

    def first_zone_problem(self) -> Optional[ZoneProblem]:
        problems = self.zone_problems()
        return problems[0] if problems else None
