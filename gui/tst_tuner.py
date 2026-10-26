"""
TST tuner dialog.

The tuning UI (video preview, evidence graph, playback, classifier settings,
profiles) lives in ``gui.tuners.tst_tuner.TstTunerWidget`` so the workspace can
embed it; ``TstTunerDialog`` is the thin modal wrapper that adds the
Accept/Cancel buttons and the discard confirmation.
"""
from PyQt6.QtWidgets import (  # noqa: F401  (QFileDialog/QMessageBox: patch targets)
    QDialog, QFileDialog, QHBoxLayout, QMessageBox, QPushButton, QVBoxLayout,
)

from gui import ux_helpers
from gui.dialog_layout import fit_dialog_to_available_screen
from gui.tuners.tst_tuner import (  # noqa: F401  (re-exported API)
    EnergyScanner,
    OptimizedCanvas,
    TstTunerWidget,
)


def _forwarded(name: str) -> property:
    """Read/write property that forwards ``name`` to the wrapped widget."""
    return property(
        lambda self: getattr(self.widget, name),
        lambda self, value: setattr(self.widget, name, value),
    )


class TstTunerDialog(QDialog):
    """
    Modal wrapper around ``TstTunerWidget``: Accept writes the tuned settings
    into the maze, Cancel asks before discarding a finished scan.
    """

    # Attributes of the embedded widget that callers and tests use directly.
    video_path = _forwarded("video_path")
    roi_manager = _forwarded("roi_manager")
    tst = _forwarded("tst")
    preview_motion_config = _forwarded("preview_motion_config")
    roi_dict = _forwarded("roi_dict")
    selected_mouse = _forwarded("selected_mouse")
    energy_data = _forwarded("energy_data")
    evidence_data = _forwarded("evidence_data")
    state_data = _forwarded("state_data")
    frame_buffer = _forwarded("frame_buffer")
    scan_len = _forwarded("scan_len")
    fps = _forwarded("fps")
    analysis_fps = _forwarded("analysis_fps")
    orig_dims = _forwarded("orig_dims")
    is_playing = _forwarded("is_playing")
    current_frame = _forwarded("current_frame")
    play_timer = _forwarded("play_timer")
    timestamps = _forwarded("timestamps")
    filtered_immobile = _forwarded("filtered_immobile")
    figure = _forwarded("figure")
    canvas = _forwarded("canvas")
    ax = _forwarded("ax")
    fill_immobile = _forwarded("fill_immobile")
    fill_pause = _forwarded("fill_pause")
    scanner = _forwarded("scanner")
    image_label = _forwarded("image_label")
    frame_label = _forwarded("frame_label")
    fps_label = _forwarded("fps_label")
    progress_bar = _forwarded("progress_bar")
    mouse_selector = _forwarded("mouse_selector")
    graph_group = _forwarded("graph_group")
    play_button = _forwarded("play_button")
    scrub_slider = _forwarded("scrub_slider")
    speed_slider = _forwarded("speed_slider")
    time_slider = _forwarded("time_slider")
    energy_slider = _forwarded("energy_slider")
    advanced_controls = _forwarded("advanced_controls")
    playback_shortcuts = _forwarded("playback_shortcuts")
    content_scroll = _forwarded("content_scroll")

    def __init__(self, video_path, roi_manager, tst_maze_instance,
                 parent=None, initial_motion_config=None):
        super().__init__(parent)
        self.setWindowTitle("TST Tuner")
        self.widget = TstTunerWidget(
            video_path, roi_manager, tst_maze_instance,
            initial_motion_config=initial_motion_config, parent=self,
        )
        self.widget.scan_finished.connect(self._on_scan_finished)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.accept_button = QPushButton("Accept Parameters")
        self.accept_button.setEnabled(False)
        self.accept_button.setStyleSheet("""
            QPushButton {
                min-width: 120px;
                padding: 10px;
                font-weight: bold;
                background-color: #4CAF50;
                color: white;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #66BB6A;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
        """)
        self.accept_button.clicked.connect(self.accept)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                min-width: 120px;
                padding: 10px;
                border-radius: 4px;
            }
        """)
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.accept_button)
        button_layout.addWidget(self.cancel_button)

        # The final actions stay permanently visible; only the widget's
        # content area scrolls when the display is short.
        root_layout = QVBoxLayout()
        root_layout.addWidget(self.widget, 1)
        root_layout.addLayout(button_layout)
        self.setLayout(root_layout)
        fit_dialog_to_available_screen(
            self,
            preferred_size=(1400, 900),
            minimum_size=(640, 480),
        )

    def _on_scan_finished(self):
        self.accept_button.setEnabled(True)

    # -- forwarded behaviour -------------------------------------------------
    def start_scan(self):
        self.widget.start_scan()

    def on_scan_complete(self, energy_data, frame_buffer, orig_dims, fps):
        self.widget.on_scan_complete(energy_data, frame_buffer, orig_dims, fps)

    def init_plot_optimized(self, trace):
        self.widget.init_plot_optimized(trace)

    def on_param_change(self):
        self.widget.on_param_change()

    def toggle_playback(self):
        self.widget.toggle_playback()

    def toggle_playback_from_key(self):
        self.widget.toggle_playback_from_key()

    def step_frames(self, delta: int):
        self.widget.step_frames(delta)

    def zoom_in(self):
        self.widget.zoom_in()

    def zoom_out(self):
        self.widget.zoom_out()

    def reset_zoom(self):
        self.widget.reset_zoom()

    def _save_profile(self):
        self.widget._save_profile()

    def _load_profile(self):
        self.widget._load_profile()

    # -- dialog behaviour ----------------------------------------------------
    def accept(self):
        """Save parameters and close dialog."""
        try:
            self.widget.apply_to_maze()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid classifier settings", str(exc))
            return
        super().accept()

    def has_unsaved_work(self) -> bool:
        """The motion scan finished, so there is tuning worth keeping."""
        return self.widget.has_unsaved_changes()

    def reject(self):
        """Esc, the window button and Cancel all end here: confirm first."""
        if self.has_unsaved_work() and not ux_helpers.confirm_discard(
            self, "the TST tuning",
            "The motion scan and the parameters you tuned will be lost."
        ):
            return
        super().reject()

    def done(self, result: int):
        """Release the video capture and worker on every way of closing."""
        # accept()/reject()/Esc all end up here without going through closeEvent.
        self.cleanup()
        super().done(result)

    def cleanup(self):
        """Stop playback, join the scanner and free the captures."""
        self.widget.shutdown()

    def closeEvent(self, event):
        """Handle window close event."""
        self.cleanup()
        super().closeEvent(event)
