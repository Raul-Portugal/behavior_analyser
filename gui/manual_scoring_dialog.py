"""Blind manual scoring of a behavior video, with hotkeys.

Opened from Validation > "Score a video manually...". A small setup dialog asks
for the paradigm, the video, the scorer's initials and where to save; the
scoring dialog then plays the video and turns hotkey presses into events (arm
entries, center/periphery changes) or held-key bouts (TST states) at the frame
on screen. The scoring is *blind*: nothing from an analysis (tracking dots,
zone labels, results) is loaded or shown. The optional ROI overlay draws only
the zone outlines of the template, exactly as the experimenter drew them.

The scored data lives in :class:`core.manual_scoring.ManualScoreSession`; this
module is the keyboard/mouse shell around it and the video player
(:class:`gui.video_player.VideoPlayerWidget`).
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QKeySequence
from PyQt6.QtWidgets import (QAbstractButton, QCheckBox, QComboBox, QDialog,
                             QDoubleSpinBox, QFileDialog, QFormLayout,
                             QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QListWidget, QListWidgetItem, QMessageBox,
                             QPushButton, QVBoxLayout, QWidget)

from core.batch_plan_keys import match_videos_to_plan
from core.manual_scoring import (CENTER, FILL_ACTIVE, FILL_NONE,
                                 PARADIGM_EPM, PARADIGM_LABELS,
                                 PARADIGM_OPEN_FIELD, PARADIGM_TST,
                                 PARADIGM_Y_MAZE, PERIPHERY, TST_STATES,
                                 ManualScoreSession, sanitize_initials,
                                 sidecar_name)
from core.video import VideoHandler
from gui.dialog_layout import fit_dialog_to_available_screen
from gui.video_player import VideoPlayerWidget, format_timestamp
from roi_manager import ROIManager

logger = logging.getLogger(__name__)

# Open-field center key behaviour.
MODE_TOGGLE = "toggle"
MODE_HOLD = "hold"

_ZONE_COLORS = [
    (60, 60, 255), (60, 220, 60), (255, 120, 60), (220, 60, 220),
    (60, 220, 220), (220, 220, 60),
]


# --------------------------------------------------------------------------
# Hotkey layouts
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Hotkey:
    """One scoring key: the keys that trigger ``action`` and what it means."""
    keys: Tuple[str, ...]
    action: str
    description: str


def hotkeys_for(paradigm: str) -> List[Hotkey]:
    """The key layout of a paradigm (shown in the dialog and used for input).

    Y-maze arms are A/B/C (1/2/3 also work) and the optional center marker is
    0, so the center never clashes with arm C. The EPM uses 1-4 for the arms
    and 0 for the center. The open field has one key, C, for the center.
    TST states are held keys.
    """
    if paradigm == PARADIGM_Y_MAZE:
        return [
            Hotkey(("A", "1"), "arm_a", "Arm A entry"),
            Hotkey(("B", "2"), "arm_b", "Arm B entry"),
            Hotkey(("C", "3"), "arm_c", "Arm C entry"),
            Hotkey(("0",), CENTER, "Center marker (optional, not an entry)"),
        ]
    if paradigm == PARADIGM_EPM:
        return [
            Hotkey(("1",), "open_arm_1", "Open arm 1 entry"),
            Hotkey(("2",), "open_arm_2", "Open arm 2 entry"),
            Hotkey(("3",), "closed_arm_1", "Closed arm 1 entry"),
            Hotkey(("4",), "closed_arm_2", "Closed arm 2 entry"),
            Hotkey(("0",), CENTER,
                   "Center marker (needed for open-arm time)"),
        ]
    if paradigm == PARADIGM_OPEN_FIELD:
        return [Hotkey(("C",), CENTER,
                       "Center (toggle: press to switch zone; hold: held = center)")]
    if paradigm == PARADIGM_TST:
        return [
            Hotkey(("I",), "immobile_static", "Hold: immobile"),
            Hotkey(("P",), "passive_swing", "Hold: passive swing (immobile)"),
            Hotkey(("A",), "active_escape", "Hold: active / mobile"),
            Hotkey(("U",), "uncertain", "Hold: uncertain"),
        ]
    raise ValueError(f"Unknown paradigm: {paradigm!r}")


def _key_char(event: QKeyEvent) -> str:
    key = event.key()
    if 0x20 < key < 0x7F:
        return chr(key).upper()
    return ""


# --------------------------------------------------------------------------
# ROI overlay support
# --------------------------------------------------------------------------

def load_overlay_rois(config_path, video_path) -> Tuple[Optional[ROIManager], str]:
    """Read ROIs from a saved template or batch plan for the overlay.

    Returns ``(manager, note)``; ``manager`` is None (with an explanation in
    ``note``) when the file cannot be used. For a batch plan the entry of
    ``video_path`` is used, or the only entry when the plan has just one.
    """
    try:
        with Path(config_path).open(encoding="utf-8") as file:
            data = json.load(file)
        if data.get("type") == "batch_plan":
            plan = data["plan"]
            if len(plan) == 1:
                entry = next(iter(plan.values()))
            else:
                match = match_videos_to_plan(
                    [Path(video_path)], list(plan), Path(config_path)
                )
                key = match.matched.get(Path(video_path))
                if key is None:
                    return None, (
                        "This plan has no unambiguous entry for the video; "
                        "no zones are drawn."
                    )
                entry = plan[key]
        else:
            entry = data["data"] if "type" in data else data
        manager = ROIManager.from_dict(entry["roi_manager"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        return None, f"Could not read the ROIs: {error}"
    return manager, ""


def draw_zone_outlines(frame: np.ndarray, rois: ROIManager) -> np.ndarray:
    """Draw ROI outlines and their names (the template's, not the analysis')."""
    colors: Dict[str, Tuple[int, int, int]] = {}
    for index, category in enumerate(rois.rois):
        colors[category] = _ZONE_COLORS[index % len(_ZONE_COLORS)]
    output = rois.draw_on_frame(frame, colors)
    for category, shapes in rois.rois.items():
        for roi in shapes:
            center = roi.points.mean(axis=0).astype(int)
            cv2.putText(
                output, category, (int(center[0]) - 30, int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                colors.get(category, (255, 255, 255)), 1, cv2.LINE_AA,
            )
    return output


# --------------------------------------------------------------------------
# Setup dialog
# --------------------------------------------------------------------------

@dataclass
class ScoringSetup:
    paradigm: str
    video_path: Path
    scorer_initials: str
    output_dir: Path
    roi_file: Optional[Path] = None
    mice: Tuple[str, ...] = ("mouse_1",)


def parse_mouse_names(text: str) -> Tuple[str, ...]:
    names: List[str] = []
    for part in str(text).replace(";", ",").split(","):
        name = "_".join(part.split())
        if name and name not in names:
            names.append(name)
    return tuple(names) or ("mouse_1",)


class ManualScoringSetupDialog(QDialog):
    """Ask what to score: paradigm, video, scorer and where to save."""

    def __init__(self, parent=None, app_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Score a video manually")
        self._settings = app_settings
        self._output_follows_video = True

        intro = QLabel(
            "Blind manual scoring: the video is shown without any automated "
            "result. Press the hotkeys at the moment you judge each event "
            "happens, then save the annotation files for the validation "
            "report."
        )
        intro.setWordWrap(True)

        self.paradigm_combo = QComboBox()
        for paradigm, label in PARADIGM_LABELS.items():
            self.paradigm_combo.addItem(label, paradigm)
        self.video_edit = QLineEdit()
        self.video_edit.setPlaceholderText("Video file to score")
        video_browse = QPushButton("Browse...")
        video_browse.clicked.connect(self._browse_video)
        self.roi_edit = QLineEdit()
        self.roi_edit.setPlaceholderText(
            "Optional: template or batch plan to draw the zones"
        )
        roi_browse = QPushButton("Browse...")
        roi_browse.clicked.connect(self._browse_rois)
        self.mice_edit = QLineEdit("mouse_1")
        self.mice_edit.setToolTip(
            "Comma-separated mouse names, as in the analysis (mouse_1, ...)"
        )
        self.initials_edit = QLineEdit()
        self.initials_edit.setMaxLength(8)
        self.initials_edit.setPlaceholderText("e.g. AB")
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Defaults to the video's folder")
        output_browse = QPushButton("Browse...")
        output_browse.clicked.connect(self._browse_output)
        self.message_label = QLabel("")
        self.message_label.setStyleSheet("color: #a00;")
        self.message_label.setWordWrap(True)

        def row(edit, button):
            layout = QHBoxLayout()
            layout.addWidget(edit, 1)
            layout.addWidget(button)
            return layout

        form = QFormLayout()
        form.addRow("Paradigm:", self.paradigm_combo)
        form.addRow("Video:", row(self.video_edit, video_browse))
        form.addRow("Zones (optional):", row(self.roi_edit, roi_browse))
        self._mice_label = QLabel("Mice (TST):")
        form.addRow(self._mice_label, self.mice_edit)
        form.addRow("Scorer initials:", self.initials_edit)
        form.addRow("Save to:", row(self.output_edit, output_browse))

        start = QPushButton("Start scoring")
        start.setDefault(True)
        start.clicked.connect(self._try_accept)
        cancel = QPushButton("Cancel")
        cancel.setAutoDefault(False)
        cancel.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        buttons.addWidget(start)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(self.message_label)
        layout.addLayout(buttons)
        self.setMinimumWidth(560)

        self.paradigm_combo.currentIndexChanged.connect(self._paradigm_changed)
        self.video_edit.textChanged.connect(self._video_changed)
        self.output_edit.textEdited.connect(self._output_edited)
        self.roi_edit.textChanged.connect(self._roi_changed)
        self._paradigm_changed()

    # --- helpers
    def _start_dir(self, kind: str) -> str:
        if self._settings is not None:
            try:
                return self._settings.dialog_dir(kind)
            except Exception:
                logger.debug("dialog_dir failed", exc_info=True)
        return ""

    def _paradigm_changed(self, _index: int = 0) -> None:
        is_tst = self.paradigm() == PARADIGM_TST
        self.mice_edit.setVisible(is_tst)
        self._mice_label.setVisible(is_tst)

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select the video to score", self._start_dir("video"),
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)",
        )
        if path:
            self.video_edit.setText(path)

    def _browse_rois(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a template or batch plan", self._start_dir("config"),
            "Configuration (*.json);;All Files (*.*)",
        )
        if path:
            self.roi_edit.setText(path)

    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Save annotation files to", self.output_edit.text()
        )
        if folder:
            self.output_edit.setText(folder)
            self._output_follows_video = False

    def _output_edited(self, _text: str) -> None:
        self._output_follows_video = False

    def _video_changed(self, text: str) -> None:
        if self._output_follows_video and text.strip():
            self.output_edit.setText(str(Path(text.strip()).parent))

    def _roi_changed(self, text: str) -> None:
        path = Path(text.strip()) if text.strip() else None
        if (self.paradigm() != PARADIGM_TST or path is None
                or not path.is_file()
                or self.mice_edit.text().strip() not in ("", "mouse_1")):
            return
        manager, _ = load_overlay_rois(path, self.video_edit.text().strip())
        if manager is not None:
            names = sorted(k for k in manager.rois if k.startswith("mouse_"))
            if names:
                self.mice_edit.setText(", ".join(names))

    # --- results
    def paradigm(self) -> str:
        return str(self.paradigm_combo.currentData())

    def set_paradigm(self, paradigm: str) -> None:
        index = self.paradigm_combo.findData(paradigm)
        if index < 0:
            raise ValueError(f"Unknown paradigm: {paradigm!r}")
        self.paradigm_combo.setCurrentIndex(index)

    def validation_problem(self) -> str:
        video = self.video_edit.text().strip()
        if not video or not Path(video).is_file():
            return "Choose an existing video file."
        if not sanitize_initials(self.initials_edit.text()):
            return "Enter the scorer's initials (letters or digits)."
        roi = self.roi_edit.text().strip()
        if roi and not Path(roi).is_file():
            return "The zones file does not exist (leave it empty for none)."
        return ""

    def setup(self) -> ScoringSetup:
        video = Path(self.video_edit.text().strip())
        output = self.output_edit.text().strip()
        roi = self.roi_edit.text().strip()
        return ScoringSetup(
            paradigm=self.paradigm(),
            video_path=video,
            scorer_initials=sanitize_initials(self.initials_edit.text()),
            output_dir=Path(output) if output else video.parent,
            roi_file=Path(roi) if roi else None,
            mice=parse_mouse_names(self.mice_edit.text()),
        )

    def _try_accept(self) -> None:
        problem = self.validation_problem()
        self.message_label.setText(problem)
        if not problem:
            self.accept()


# --------------------------------------------------------------------------
# Scoring dialog
# --------------------------------------------------------------------------

class ManualScoringDialog(QDialog):
    """Play a video and score it with the keyboard; see the module docstring."""

    files_saved = pyqtSignal(list)

    def __init__(self, setup: ScoringSetup, parent=None):
        super().__init__(parent)
        self.setup = setup
        self.setWindowTitle(
            f"Manual scoring - {PARADIGM_LABELS[setup.paradigm]} - "
            f"{setup.video_path.name}"
        )
        handler = VideoHandler(setup.video_path)   # raises IOError if unreadable
        if handler.fps <= 0 or handler.total_frames <= 0:
            raise IOError(f"Cannot read frames or frame rate: {setup.video_path}")
        self.session = ManualScoreSession(
            setup.paradigm, setup.video_path.name, handler.fps,
            handler.total_frames, setup.scorer_initials, mice=setup.mice,
            roi_file=str(setup.roi_file) if setup.roi_file else None,
            output_dir=str(setup.output_dir),
        )
        self.sidecar_path = Path(setup.output_dir) / sidecar_name(
            self.session.video_stem, setup.paradigm, setup.scorer_initials
        )
        self.hotkeys = hotkeys_for(setup.paradigm)
        self._keymap: Dict[str, str] = {
            key: hotkey.action for hotkey in self.hotkeys for key in hotkey.keys
        }
        self._held_key: Optional[str] = None
        self._zone_mode = MODE_TOGGLE
        self._roi_manager: Optional[ROIManager] = None
        self._roi_note = ""
        self._closing = False
        self._resumed_note = ""

        self._build_ui(handler)
        self._load_rois(handler)
        self._resume_if_possible()
        self.player.frame_changed.connect(self._on_frame_changed)
        self.player.set_overlay(self._draw_overlay)
        if not self.player.open_video(setup.video_path):
            raise IOError(f"Cannot open video: {setup.video_path}")
        self._refresh_list()
        self._update_frame_status(0)
        fit_dialog_to_available_screen(
            self, preferred_size=(1280, 760), minimum_size=(900, 560)
        )

    # ------------------------------------------------------------------ UI
    def _build_ui(self, handler: VideoHandler) -> None:
        setup = self.setup
        self.player = VideoPlayerWidget(self)

        legend_rows = "".join(
            f"<tr><td><b>{' / '.join(h.keys)}</b>&nbsp;&nbsp;</td>"
            f"<td>{h.description}</td></tr>"
            for h in self.hotkeys
        )
        self.legend_label = QLabel(
            f"<table>{legend_rows}</table>"
            "<p><b>Space</b> play/pause, <b>Left/Right</b> step one frame, "
            "<b>Shift+Left/Right</b> jump 10 frames, <b>Home/End</b> "
            "start/end, <b>Ctrl+Z</b> undo, <b>Delete</b> remove the "
            "selected row.</p>"
        )
        self.legend_label.setWordWrap(True)
        self.legend_label.setTextFormat(Qt.TextFormat.RichText)

        blind = QLabel(
            "Blind mode: no automated tracking, zone labels or results are "
            "shown."
        )
        blind.setStyleSheet("color: #145a32; font-weight: bold;")
        blind.setWordWrap(True)
        self.blind_label = blind

        who = QLabel(
            f"Scorer: {self.session.scorer_initials}    "
            f"Date: {self.session.date_started[:10]}    "
            f"{handler.total_frames} frames at {handler.fps:g} fps"
        )
        who.setWordWrap(True)

        # Options (paradigm specific).
        options = QGroupBox("Options")
        form = QFormLayout(options)
        self.zones_check = QCheckBox("Show zone outlines")
        self.zones_check.setChecked(True)
        self.zones_check.toggled.connect(lambda _on: self.player.refresh())
        self.zones_check.setVisible(False)
        form.addRow(self.zones_check)
        self.roi_note_label = QLabel("")
        self.roi_note_label.setWordWrap(True)
        self.roi_note_label.setVisible(False)
        form.addRow(self.roi_note_label)

        duration = handler.total_frames / handler.fps
        self.window_start_spin = QDoubleSpinBox(
            decimals=2, minimum=0.0, maximum=max(duration, 0.01),
            singleStep=1.0)
        self.window_start_spin.setSuffix(" s")
        self.window_end_spin = QDoubleSpinBox(
            decimals=2, minimum=0.0, maximum=max(duration, 0.01),
            singleStep=1.0)
        self.window_end_spin.setSuffix(" s")
        self.window_end_spin.setSpecialValueText("end of video")
        self.window_start_spin.setToolTip(
            "Start of the scored window (score the same window the "
            "analysis scores)")
        self.window_end_spin.setToolTip("End of the scored window")
        self.window_start_spin.editingFinished.connect(self._window_changed)
        self.window_end_spin.editingFinished.connect(self._window_changed)
        needs_window = setup.paradigm != PARADIGM_Y_MAZE
        self._window_rows: List[QWidget] = []
        if needs_window:
            form.addRow("Window start:", self.window_start_spin)
            form.addRow("Window end:", self.window_end_spin)

        self.zone_mode_combo = QComboBox()
        self.zone_mode_combo.addItem("Toggle: C switches center / periphery",
                                     MODE_TOGGLE)
        self.zone_mode_combo.addItem("Hold: C held = center", MODE_HOLD)
        self.zone_mode_combo.currentIndexChanged.connect(self._zone_mode_changed)
        self.initial_zone_combo = QComboBox()
        self.initial_zone_combo.addItem("Starts in the periphery", PERIPHERY)
        self.initial_zone_combo.addItem("Starts in the center", CENTER)
        self.initial_zone_combo.currentIndexChanged.connect(
            self._initial_zone_changed)
        if setup.paradigm == PARADIGM_OPEN_FIELD:
            form.addRow("Center key:", self.zone_mode_combo)
            form.addRow("Initial zone:", self.initial_zone_combo)

        self.mouse_combo = QComboBox()
        self.mouse_combo.addItems(self.session.mice)
        self.mouse_combo.currentIndexChanged.connect(self._refresh_list)
        self.fill_combo = QComboBox()
        self.fill_combo.addItem("Leave unmarked frames unscored", FILL_NONE)
        self.fill_combo.addItem(
            "Unmarked frames in the window = active_escape", FILL_ACTIVE)
        self.fill_combo.currentIndexChanged.connect(self._fill_changed)
        if setup.paradigm == PARADIGM_TST:
            form.addRow("Mouse:", self.mouse_combo)
            form.addRow("No key held:", self.fill_combo)

        self.state_label = QLabel("")
        self.state_label.setStyleSheet("font-weight: bold;")
        self.state_label.setWordWrap(True)

        self.event_list = QListWidget()
        self.event_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection)
        self.event_list.itemDoubleClicked.connect(self._jump_to_item)
        self.event_list.setToolTip("Double-click a row to jump to its frame")

        self.undo_button = QPushButton("Undo (Ctrl+Z)")
        self.undo_button.clicked.connect(self.undo)
        self.delete_button = QPushButton("Delete selected")
        self.delete_button.clicked.connect(self.delete_selected)
        self.clear_button = QPushButton("Clear all")
        self.clear_button.clicked.connect(self.clear_all)
        edit_row = QHBoxLayout()
        for button in (self.undo_button, self.delete_button, self.clear_button):
            edit_row.addWidget(button)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.save_button = QPushButton("Save annotation files")
        self.save_button.clicked.connect(lambda: self.save())
        self.done_button = QPushButton("Save and close")
        self.done_button.clicked.connect(self.save_and_close)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.reject)
        save_row = QHBoxLayout()
        for button in (self.save_button, self.done_button, self.close_button):
            save_row.addWidget(button)

        side = QVBoxLayout()
        side.addWidget(blind)
        side.addWidget(who)
        side.addWidget(self.legend_label)
        side.addWidget(options)
        side.addWidget(self.state_label)
        side.addWidget(QLabel(self._list_title()))
        side.addWidget(self.event_list, 1)
        side.addLayout(edit_row)
        side.addWidget(self.status_label)
        side.addLayout(save_row)
        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setFixedWidth(360)

        layout = QHBoxLayout(self)
        layout.addWidget(self.player, 1)
        layout.addWidget(side_widget)

        # Keyboard focus stays on the dialog/player so hotkeys always arrive.
        for widget_type in (QAbstractButton, QComboBox, QListWidget):
            for widget in self.findChildren(widget_type):
                if not isinstance(widget, VideoPlayerWidget):
                    widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.player.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.player.setFocus()

    def _list_title(self) -> str:
        if self.setup.paradigm == PARADIGM_TST:
            return "Bouts of the selected mouse (double-click to jump):"
        if self.setup.paradigm == PARADIGM_OPEN_FIELD:
            return "Zone changes (double-click to jump):"
        return "Events (double-click to jump):"

    # ----------------------------------------------------------------- ROIs
    def _load_rois(self, handler: VideoHandler) -> None:
        if self.setup.roi_file is None:
            return
        manager, note = load_overlay_rois(
            self.setup.roi_file, self.setup.video_path
        )
        if manager is not None and manager.source_dimensions is not None \
                and tuple(manager.source_dimensions) != tuple(handler.dimensions):
            note = (
                f"The ROIs were drawn for {manager.source_dimensions[0]}x"
                f"{manager.source_dimensions[1]}, this video is "
                f"{handler.width}x{handler.height}: zones are not drawn."
            )
            manager = None
        self._roi_manager, self._roi_note = manager, note
        self.zones_check.setVisible(manager is not None)
        self.roi_note_label.setVisible(bool(note))
        self.roi_note_label.setText(note)

    def _draw_overlay(self, frame: np.ndarray, _index: int) -> np.ndarray:
        if self._roi_manager is None or not self.zones_check.isChecked():
            return frame
        return draw_zone_outlines(frame, self._roi_manager)

    # --------------------------------------------------------------- resume
    def _resume_if_possible(self) -> None:
        if not self.sidecar_path.is_file():
            return
        try:
            restored = self.session.load_sidecar(self.sidecar_path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            logger.warning("Could not resume %s: %s", self.sidecar_path, error)
            self._resumed_note = f"Could not resume the earlier session: {error}"
            return
        if not restored:
            return
        self._resumed_note = (
            f"Resumed an earlier session: {self.session.event_count()} item(s)."
        )
        self.window_start_spin.setValue(self.session.window_start_s)
        self.window_end_spin.setValue(self.session.window_end_s or 0.0)
        index = self.initial_zone_combo.findData(self.session.initial_zone)
        self.initial_zone_combo.setCurrentIndex(max(index, 0))
        index = self.fill_combo.findData(self.session.fill_unmarked)
        self.fill_combo.setCurrentIndex(max(index, 0))
        self.mouse_combo.clear()
        self.mouse_combo.addItems(self.session.mice)
        self.status_label.setText(self._resumed_note)

    # ---------------------------------------------------------------- options
    def _window_changed(self) -> None:
        start = self.window_start_spin.value()
        end = self.window_end_spin.value() or None
        try:
            self.session.set_window(start, end)
        except ValueError as error:
            self.status_label.setText(str(error))
            self.window_end_spin.setValue(0.0)
            self.session.set_window(start, None)
        self._autosave()
        self.player.setFocus()

    def _zone_mode_changed(self, _index: int = 0) -> None:
        self._zone_mode = str(self.zone_mode_combo.currentData())
        self.player.setFocus()

    def _initial_zone_changed(self, _index: int = 0) -> None:
        self.session.initial_zone = str(self.initial_zone_combo.currentData())
        self.session.dirty = True
        self._autosave()
        self._refresh_list()
        self._update_frame_status(self.player.current_frame)

    def _fill_changed(self, _index: int = 0) -> None:
        self.session.fill_unmarked = str(self.fill_combo.currentData())
        self.session.dirty = True
        self._autosave()

    @property
    def zone_mode(self) -> str:
        return self._zone_mode

    def set_zone_mode(self, mode: str) -> None:
        index = self.zone_mode_combo.findData(mode)
        if index < 0:
            raise ValueError(f"Unknown mode: {mode!r}")
        self.zone_mode_combo.setCurrentIndex(index)

    def selected_mouse(self) -> str:
        return self.mouse_combo.currentText() or self.session.mice[0]

    # ------------------------------------------------------------------ input
    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.matches(QKeySequence.StandardKey.Undo):
            self.undo()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Delete:
            self.delete_selected()
            event.accept()
            return
        if self.player.handle_key_event(event):
            event.accept()
            return
        modifiers = event.modifiers()
        blocked = (Qt.KeyboardModifier.ControlModifier
                   | Qt.KeyboardModifier.AltModifier
                   | Qt.KeyboardModifier.MetaModifier)
        action = None if modifiers & blocked else self._keymap.get(_key_char(event))
        if action is None:
            super().keyPressEvent(event)
            return
        event.accept()
        if event.isAutoRepeat():
            return
        self.handle_hotkey(_key_char(event), action)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if (not event.isAutoRepeat() and self._held_key is not None
                and _key_char(event) == self._held_key):
            self.release_hold()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def changeEvent(self, event) -> None:
        if event.type() == QEvent.Type.ActivationChange and not self.isActiveWindow():
            # A key released while another window had focus never arrives.
            self.release_hold()
        super().changeEvent(event)

    def handle_hotkey(self, key: str, action: str) -> None:
        """React to a scoring key press (also callable without a key event)."""
        if self.player.current_frame < 0:
            return
        paradigm = self.setup.paradigm
        if paradigm == PARADIGM_TST:
            self.begin_hold(key, action)
        elif paradigm == PARADIGM_OPEN_FIELD:
            self._open_field_key(key)
        else:
            self.log_event(action)

    def _open_field_key(self, key: str) -> None:
        frame = self.player.current_frame
        if self._zone_mode == MODE_HOLD:
            if self._held_key is None and self.session.zone_now(frame) != CENTER:
                self.session.add_event(CENTER, frame)
                self._held_key = key
                self._after_change(f"center from frame {frame}")
            elif self._held_key is None:
                self._held_key = key
            return
        event = self.session.toggle_zone(frame)
        self._after_change(f"{event.label} at frame {frame}")

    def log_event(self, label: str) -> None:
        """Log an arm entry / marker at the frame on screen."""
        frame = self.player.current_frame
        if frame < 0:
            return
        self.session.add_event(label, frame)
        self._after_change(
            f"{label} at {format_timestamp(self.session.time_of(frame))} "
            f"(frame {frame})"
        )

    def begin_hold(self, key: str, state: str) -> None:
        """TST: start (or switch to) a bout of ``state`` at the current frame."""
        if state not in TST_STATES:
            raise ValueError(f"Unsupported TST state: {state!r}")
        self.end_hold(save=False)
        frame = self.player.current_frame
        if frame < 0:
            return
        self.session.begin_bout(self.selected_mouse(), state, frame)
        self._held_key = key
        self._refresh_list()
        self._update_frame_status(frame)

    def release_hold(self) -> None:
        """The scoring key was released."""
        if self._held_key is None:
            return
        if self.setup.paradigm == PARADIGM_OPEN_FIELD:
            frame = self.player.current_frame
            self._held_key = None
            if self._zone_mode == MODE_HOLD and frame >= 0 \
                    and self.session.zone_now(frame) == CENTER:
                self.session.add_event(PERIPHERY, frame)
                self._after_change(f"periphery from frame {frame}")
            return
        self.end_hold(save=True)

    def end_hold(self, save: bool = True) -> None:
        self._held_key = None
        bout = self.session.end_bout()
        if bout is not None and save:
            self._after_change(
                f"{bout.state}: frames {bout.start}-{bout.end}"
            )

    def _on_frame_changed(self, frame: int, _timestamp: float) -> None:
        if self.session.open_bout is not None:
            self.session.extend_bout(frame)
            self._refresh_list()
        self._update_frame_status(frame)

    # ------------------------------------------------------------------ edits
    def _after_change(self, message: str) -> None:
        self.status_label.setText(message)
        self._refresh_list()
        self._update_frame_status(self.player.current_frame)
        self._autosave()

    def undo(self) -> None:
        self.end_hold(save=False)
        description = self.session.undo()
        if description is None:
            self.status_label.setText("Nothing to undo.")
            return
        self._after_change(f"Undid: {description}")

    def selected_rows(self) -> List[int]:
        """Model indices of the selected list rows."""
        return [
            int(item.data(Qt.ItemDataRole.UserRole))
            for item in self.event_list.selectedItems()
            if item.data(Qt.ItemDataRole.UserRole) is not None
        ]

    def delete_selected(self) -> None:
        indices = self.selected_rows()
        if not indices:
            return
        if self.setup.paradigm == PARADIGM_TST:
            removed = self.session.delete_bouts(indices)
        else:
            removed = self.session.delete_events(indices)
        if removed:
            self._after_change(f"Deleted {removed} row(s)")

    def clear_all(self) -> None:
        if not (self.session.events or self.session.bouts):
            return
        answer = QMessageBox.question(
            self, "Clear all",
            "Remove everything scored in this session? You can still undo it.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.session.clear()
            self._after_change("Cleared all rows")

    def _jump_to_item(self, item: QListWidgetItem) -> None:
        frame = item.data(Qt.ItemDataRole.UserRole + 1)
        if frame is not None:
            self.end_hold(save=False)
            self.player.pause()
            self.player.seek(int(frame))
            self.player.setFocus()

    # --------------------------------------------------------------- display
    def _event_text(self, number: int, label: str, frame: int) -> str:
        return (
            f"{number:>3}  {label:<14} "
            f"{format_timestamp(self.session.time_of(frame))}   frame {frame}"
        )

    def _refresh_list(self, *_args) -> None:
        selected = set(self.selected_rows())
        self.event_list.clear()
        paradigm = self.setup.paradigm
        if paradigm == PARADIGM_TST:
            mouse = self.selected_mouse()
            order = sorted(
                (i for i, b in enumerate(self.session.bouts) if b.mouse == mouse),
                key=lambda i: (self.session.bouts[i].start, i),
            )
            for row, index in enumerate(order, start=1):
                bout = self.session.bouts[index]
                text = (
                    f"{row:>3}  {bout.state:<16} frames {bout.start}-{bout.end}"
                    f"   {format_timestamp(self.session.time_of(bout.start))}"
                    f" - {format_timestamp(self.session.time_of(bout.end + 1))}"
                )
                self._add_item(text, index, bout.start, index in selected)
        else:
            arm_number = 0
            for index, event in enumerate(self.session.events):
                if paradigm == PARADIGM_OPEN_FIELD:
                    arm_number += 1
                    text = self._event_text(
                        arm_number, f"-> {event.label}", event.frame)
                    self._add_item(text, index, event.frame, index in selected)
                elif event.label == CENTER:
                    item = self._add_item(
                        f"     {'center marker':<14} "
                        f"{format_timestamp(self.session.time_of(event.frame))}"
                        f"   frame {event.frame}",
                        index, event.frame, index in selected)
                    item.setForeground(Qt.GlobalColor.gray)
                else:
                    arm_number += 1
                    self._add_item(
                        self._event_text(arm_number, event.label, event.frame),
                        index, event.frame, index in selected)
        self.event_list.scrollToBottom()

    def _add_item(self, text: str, index: int, frame: int,
                  selected: bool) -> QListWidgetItem:
        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, index)
        item.setData(Qt.ItemDataRole.UserRole + 1, frame)
        self.event_list.addItem(item)
        if selected:
            item.setSelected(True)
        return item

    def _update_frame_status(self, frame: int) -> None:
        paradigm = self.setup.paradigm
        if frame < 0:
            self.state_label.setText("")
        elif paradigm == PARADIGM_OPEN_FIELD:
            self.state_label.setText(
                f"Zone at this frame (your marks): {self.session.zone_now(frame)}"
            )
        elif paradigm == PARADIGM_TST:
            held = self.session.open_bout
            state = None
            for bout in reversed(self.session.bouts):
                if bout.mouse == self.selected_mouse() \
                        and bout.start <= frame <= bout.end:
                    state = bout.state
                    break
            if held is not None:
                self.state_label.setText(f"Recording: {held.state}")
            else:
                self.state_label.setText(
                    f"State at this frame: {state or 'not scored'}"
                )
        else:
            self.state_label.setText(
                f"{self.session.event_count()} arm entries scored"
            )

    # ------------------------------------------------------------------ saving
    def _autosave(self) -> None:
        try:
            self.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            self.session.save_sidecar(self.sidecar_path)
        except OSError as error:
            logger.warning("Autosave failed: %s", error)
            self.status_label.setText(f"Autosave failed: {error}")

    def _confirm_overwrite(self, paths: Sequence[Path]) -> bool:
        names = "\n".join(str(path) for path in paths)
        answer = QMessageBox.question(
            self, "Replace existing files?",
            "These files exist and were not written by this scoring "
            f"session (for example by another scorer):\n\n{names}\n\n"
            "Replace them?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def save(self) -> bool:
        """Write the annotation files; False if cancelled or failed."""
        self.end_hold(save=True)
        planned = self.session.planned_outputs(self.setup.output_dir)
        foreign = [
            path for path in planned.values()
            if path.exists() and str(path) not in self.session.saved_files
        ]
        if foreign and not self._confirm_overwrite(foreign):
            return False
        try:
            written = self.session.write_outputs(self.setup.output_dir)
        except (OSError, ValueError) as error:
            logger.error("Could not save the annotations", exc_info=True)
            QMessageBox.warning(self, "Could not save", str(error))
            return False
        self._autosave()
        self.status_label.setText(
            "Saved: " + ", ".join(path.name for path in written)
        )
        self.files_saved.emit([str(path) for path in written])
        return True

    def save_and_close(self) -> None:
        if self.save():
            self.accept()

    # ----------------------------------------------------------------- closing
    def _ask_close_action(self) -> str:
        """``"save"``, ``"discard"`` or ``"cancel"`` for unsaved annotations."""
        box = QMessageBox(self)
        box.setWindowTitle("Save before closing?")
        box.setText(
            "The annotation files have not been saved since your last "
            "change. Progress is autosaved and can be resumed."
        )
        save = box.addButton("Save and close", QMessageBox.ButtonRole.AcceptRole)
        discard = box.addButton("Close without saving",
                                QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is save:
            return "save"
        if clicked is discard:
            return "discard"
        return "cancel"

    def reject(self) -> None:
        if not self._closing and self.session.dirty and self.session.event_count():
            choice = self._ask_close_action()
            if choice == "cancel":
                return
            if choice == "save" and not self.save():
                return
        super().reject()

    def done(self, result: int) -> None:
        self._release()
        super().done(result)

    def closeEvent(self, event) -> None:
        if not self._closing and self.session.dirty and self.session.event_count():
            event.ignore()
            self.reject()
            return
        self._release()
        super().closeEvent(event)

    def _release(self) -> None:
        self._closing = True
        self.end_hold(save=True)
        self._autosave()
        self.player.close_video()


def open_manual_scoring(parent=None, app_settings=None) -> None:
    """Run the setup dialog, then the scoring dialog (both modal)."""
    setup_dialog = ManualScoringSetupDialog(parent, app_settings)
    try:
        if not setup_dialog.exec():
            return
        setup = setup_dialog.setup()
    finally:
        setup_dialog.deleteLater()
    if app_settings is not None:
        try:
            app_settings.set_last_video_dir(setup.video_path.parent)
        except Exception:
            logger.debug("Could not remember the video folder", exc_info=True)
    try:
        dialog = ManualScoringDialog(setup, parent)
    except (IOError, OSError) as error:
        QMessageBox.warning(parent, "Cannot open the video", str(error))
        return
    try:
        dialog.exec()
    finally:
        dialog.deleteLater()


__all__ = [
    "ManualScoringDialog", "ManualScoringSetupDialog", "ScoringSetup",
    "hotkeys_for", "load_overlay_rois", "open_manual_scoring",
    "parse_mouse_names",
]
