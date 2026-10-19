"""Tracking accuracy spot-check: click the animal in randomly sampled frames.

Opened from Validation > "Tracking spot-check...". The user picks a run
folder (or a video plus its ``*_tracking_data.csv``); the dialog samples N
frames with a fixed, visible seed and shows each *without* the automated dot.
The scorer clicks the animal's center, or presses "Not visible". At the end
the clicks are compared with the tracking CSV (pixel and cm error, miss rate,
false detections), shown, and saved as ``<base>_position_spotcheck.csv`` plus
a summary JSON (see :mod:`core.position_spotcheck`).

The click mapping follows the ROI selector: large frames are scaled down to at
most 1280 px wide for display, and a click is converted back to source pixels
with the display-to-source ratio.
"""
import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QPainter, QPen
from PyQt6.QtWidgets import (QComboBox, QDialog, QFileDialog, QFormLayout,
                             QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                             QPushButton, QScrollArea, QSpinBox,
                             QStackedWidget, QVBoxLayout, QWidget)

from core.manual_scoring import sanitize_initials
from core.position_spotcheck import (DEFAULT_SAMPLE_SIZE, DEFAULT_SEED,
                                     STATUS_CLICKED, RunVideo, SpotcheckSession,
                                     base_name_of, read_scale_factor,
                                     scan_run_folder)
from core.positional_validation import load_tracking_csv
from core.video import position_capture_at_frame
from gui.dialog_layout import fit_dialog_to_available_screen
from gui.qt_image import bgr_to_qpixmap

logger = logging.getLogger(__name__)

# Frames wider than this are scaled down for display (as in the ROI selector).
MAX_DISPLAY_WIDTH = 1280

PAGE_SETUP, PAGE_REVIEW, PAGE_RESULTS = 0, 1, 2


class ClickableFrameLabel(QLabel):
    """Shows a frame at display size and reports clicks in SOURCE pixels."""

    clicked = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source_width = 0
        self.source_height = 0
        self.display_to_source_x = 1.0
        self.display_to_source_y = 1.0
        self._has_frame = False
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_frame(self, frame: np.ndarray, marker_xy=None) -> None:
        """Show ``frame`` (BGR); ``marker_xy`` is a source-pixel point to mark."""
        source_h, source_w = frame.shape[:2]
        self.source_width, self.source_height = source_w, source_h
        shown = frame
        if source_w > MAX_DISPLAY_WIDTH:
            scale = MAX_DISPLAY_WIDTH / source_w
            shown = cv2.resize(
                frame, (int(source_w * scale), int(source_h * scale)))
        display_h, display_w = shown.shape[:2]
        self.display_to_source_x = source_w / display_w
        self.display_to_source_y = source_h / display_h
        pixmap = bgr_to_qpixmap(shown)
        if marker_xy is not None:
            painter = QPainter(pixmap)
            painter.setPen(QPen(Qt.GlobalColor.green, 2))
            x = int(round(marker_xy[0] / self.display_to_source_x))
            y = int(round(marker_xy[1] / self.display_to_source_y))
            painter.drawEllipse(x - 7, y - 7, 14, 14)
            painter.drawLine(x - 11, y, x + 11, y)
            painter.drawLine(x, y - 11, x, y + 11)
            painter.end()
        self.setPixmap(pixmap)
        self.setFixedSize(pixmap.size())
        self._has_frame = True

    def to_source(self, display_x: float, display_y: float):
        return (display_x * self.display_to_source_x,
                display_y * self.display_to_source_y)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._has_frame:
            position = event.position()
            self.clicked.emit(*self.to_source(position.x(), position.y()))
            event.accept()
        else:
            super().mousePressEvent(event)


class PositionSpotcheckDialog(QDialog):
    """Sample frames, collect clicks, evaluate and save; see the module doc."""

    completed = pyqtSignal(str, str)    # csv path, summary json path

    def __init__(self, parent=None, app_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Tracking accuracy spot-check")
        self._settings = app_settings
        self._output_follows_video = True
        self._run_videos: List[RunVideo] = []
        self.session: Optional[SpotcheckSession] = None
        self._cap = None
        self._index = 0
        self.saved_files = None
        self._build_ui()
        fit_dialog_to_available_screen(
            self, preferred_size=(1100, 800), minimum_size=(640, 480))

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_setup_page())
        self.pages.addWidget(self._build_review_page())
        self.pages.addWidget(self._build_results_page())
        layout = QVBoxLayout(self)
        layout.addWidget(self.pages)

    def _path_row(self, edit: QLineEdit, slot) -> QHBoxLayout:
        button = QPushButton("Browse...")
        button.clicked.connect(slot)
        row = QHBoxLayout()
        row.addWidget(edit, 1)
        row.addWidget(button)
        return row

    def _build_setup_page(self) -> QWidget:
        page = QWidget()
        intro = QLabel(
            "Check how accurately the tracker finds the animal. You will "
            "see randomly chosen frames without the automated position and "
            "click on the animal's center in each (or mark it not visible). "
            "Choose a run folder, or a video together with its tracking CSV."
        )
        intro.setWordWrap(True)

        self.run_edit = QLineEdit()
        self.run_edit.setPlaceholderText("Optional: analysis run folder")
        self.run_combo = QComboBox()
        self.run_combo.currentIndexChanged.connect(self._run_video_chosen)
        self.video_edit = QLineEdit()
        self.video_edit.setPlaceholderText("Video file")
        self.tracking_edit = QLineEdit()
        self.tracking_edit.setPlaceholderText("<video>_tracking_data.csv")
        self.summary_edit = QLineEdit()
        self.summary_edit.setPlaceholderText(
            "<video>_summary.json (for the cm scale; optional)")
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 5000)
        self.count_spin.setValue(DEFAULT_SAMPLE_SIZE)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(DEFAULT_SEED)
        self.seed_spin.setToolTip(
            "The same seed on the same tracking file gives the same frames")
        self.initials_edit = QLineEdit()
        self.initials_edit.setMaxLength(8)
        self.initials_edit.setPlaceholderText("optional, e.g. AB")
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Defaults to the video's folder")
        self.message_label = QLabel("")
        self.message_label.setStyleSheet("color: #a00;")
        self.message_label.setWordWrap(True)

        self.run_edit.editingFinished.connect(
            lambda: self.load_run_folder(self.run_edit.text().strip()))
        self.video_edit.textChanged.connect(self._video_changed)
        self.output_edit.textEdited.connect(self._output_edited)
        self.tracking_edit.textChanged.connect(self._tracking_changed)

        form = QFormLayout()
        form.addRow("Run folder:", self._path_row(self.run_edit, self._browse_run))
        form.addRow("Video in the run:", self.run_combo)
        form.addRow("Video:", self._path_row(self.video_edit, self._browse_video))
        form.addRow("Tracking CSV:",
                    self._path_row(self.tracking_edit, self._browse_tracking))
        form.addRow("Summary JSON:",
                    self._path_row(self.summary_edit, self._browse_summary))
        form.addRow("Frames to check:", self.count_spin)
        form.addRow("Random seed:", self.seed_spin)
        form.addRow("Scorer initials:", self.initials_edit)
        form.addRow("Save to:", self._path_row(self.output_edit,
                                               self._browse_output))

        self.start_button = QPushButton("Start")
        self.start_button.setDefault(True)
        self.start_button.clicked.connect(self.start)
        cancel = QPushButton("Cancel")
        cancel.setAutoDefault(False)
        cancel.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        buttons.addWidget(self.start_button)

        layout = QVBoxLayout(page)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(self.message_label)
        layout.addStretch(1)
        layout.addLayout(buttons)
        return page

    def _build_review_page(self) -> QWidget:
        page = QWidget()
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("font-weight: bold;")
        self.instruction_label = QLabel(
            "Click the animal's center. If it is not visible, press "
            "\"Not visible\" (N). Backspace goes back. The automated "
            "position is not shown."
        )
        self.instruction_label.setWordWrap(True)
        self.frame_label = ClickableFrameLabel()
        self.frame_label.clicked.connect(self._on_click)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.frame_label)
        self.not_visible_button = QPushButton("Not visible (N)")
        self.not_visible_button.clicked.connect(self.not_visible)
        self.back_button = QPushButton("Back (Backspace)")
        self.back_button.clicked.connect(self.back)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addWidget(self.back_button)
        buttons.addWidget(self.not_visible_button)
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        for button in (self.back_button, self.not_visible_button, cancel):
            button.setAutoDefault(False)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout = QVBoxLayout(page)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.instruction_label)
        layout.addWidget(self.scroll, 1)
        layout.addLayout(buttons)
        return page

    def _build_results_page(self) -> QWidget:
        page = QWidget()
        self.results_label = QLabel("")
        self.results_label.setTextFormat(Qt.TextFormat.RichText)
        self.results_label.setWordWrap(True)
        self.results_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.results_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        layout = QVBoxLayout(page)
        layout.addWidget(self.results_label, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close)
        layout.addLayout(row)
        return page

    # ------------------------------------------------------------- setup page
    def _start_dir(self, kind: str) -> str:
        if self._settings is not None:
            try:
                return self._settings.dialog_dir(kind)
            except Exception:
                logger.debug("dialog_dir failed", exc_info=True)
        return ""

    def _browse_run(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select an analysis run folder", self._start_dir("output"))
        if folder:
            self.run_edit.setText(folder)
            self.load_run_folder(folder)

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select the video", self._start_dir("video"),
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)")
        if path:
            self.video_edit.setText(path)

    def _browse_tracking(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select the tracking data CSV", self.run_edit.text(),
            "Tracking data (*_tracking_data.csv);;CSV (*.csv)")
        if path:
            self.tracking_edit.setText(path)

    def _browse_summary(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select the summary JSON", self.run_edit.text(),
            "Summary (*_summary.json);;JSON (*.json)")
        if path:
            self.summary_edit.setText(path)

    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Save results to", self.output_edit.text())
        if folder:
            self.output_edit.setText(folder)
            self._output_follows_video = False

    def _output_edited(self, _text: str) -> None:
        self._output_follows_video = False

    def _video_changed(self, text: str) -> None:
        if self._output_follows_video and text.strip():
            self.output_edit.setText(str(Path(text.strip()).parent))

    def _tracking_changed(self, text: str) -> None:
        path = Path(text.strip()) if text.strip() else None
        if path is None or self.summary_edit.text().strip():
            return
        base = base_name_of(path)
        summary = path.with_name(f"{base}_summary.json")
        if summary.is_file():
            self.summary_edit.setText(str(summary))

    def load_run_folder(self, folder: str) -> int:
        """List the run's videos that have a tracking CSV; returns the count."""
        self._run_videos = scan_run_folder(folder) if folder else []
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for item in self._run_videos:
            self.run_combo.addItem(item.base_name, item)
        self.run_combo.blockSignals(False)
        if self._run_videos:
            self.run_combo.setCurrentIndex(0)
            self.message_label.setText("")
            self._run_video_chosen(0)
        elif folder:
            self.message_label.setText(
                "No *_tracking_data.csv found in that folder (TST runs have "
                "no tracking CSV).")
        return len(self._run_videos)

    def _run_video_chosen(self, index: int) -> None:
        item = self.run_combo.itemData(index)
        if item is None:
            return
        self.tracking_edit.setText(str(item.tracking_csv))
        self.summary_edit.setText(
            str(item.summary_json) if item.summary_json else "")
        self.video_edit.setText(
            str(item.source_video) if item.source_video is not None else "")
        if item.source_video is None:
            self.message_label.setText(
                "The source video was not found from the run's manifest: "
                "choose it with Browse.")

    def validation_problem(self) -> str:
        video = self.video_edit.text().strip()
        tracking = self.tracking_edit.text().strip()
        if not video or not Path(video).is_file():
            return "Choose the video file (it was not found in the run)."
        if not tracking or not Path(tracking).is_file():
            return "Choose the tracking data CSV."
        summary = self.summary_edit.text().strip()
        if summary and not Path(summary).is_file():
            return "The summary JSON does not exist (leave it empty for none)."
        return ""

    # ----------------------------------------------------------------- start
    def start(self) -> bool:
        """Validate the inputs, draw the sample and show the first frame."""
        problem = self.validation_problem()
        self.message_label.setText(problem)
        if problem:
            return False
        try:
            tracking = load_tracking_csv(self.tracking_edit.text().strip())
            summary = self.summary_edit.text().strip()
            scale = read_scale_factor(summary) if summary else 0.0
            self.session = SpotcheckSession(
                tracking, self.count_spin.value(), self.seed_spin.value(),
                scale)
        except (OSError, ValueError) as error:
            self.message_label.setText(f"Cannot use the tracking data: {error}")
            return False
        self._release_capture()
        self._cap = cv2.VideoCapture(self.video_edit.text().strip())
        if not self._cap.isOpened():
            self._release_capture()
            self.message_label.setText("Cannot open the video.")
            return False
        self._index = 0
        self.pages.setCurrentIndex(PAGE_REVIEW)
        self.setFocus()
        self._show_sample()
        return True

    # ------------------------------------------------------------ review page
    def _show_sample(self) -> None:
        session = self.session
        while 0 <= self._index < len(session):
            sample = session.samples[self._index]
            position_capture_at_frame(self._cap, sample.frame)
            ok, frame = self._cap.read()
            if ok and frame is not None:
                marker = (sample.manual_xy
                          if sample.status == STATUS_CLICKED else None)
                self.frame_label.set_frame(frame, marker)
                done = len(session.reviewed_samples())
                self.progress_label.setText(
                    f"Frame {self._index + 1} of {len(session)}   "
                    f"(frame number {sample.frame}, "
                    f"seed {session.seed}, {done} answered)")
                self.back_button.setEnabled(self._index > 0)
                return
            logger.warning("Spot-check frame %d could not be read", sample.frame)
            session.mark_unreadable(self._index)
            self._index += 1
        self._advance_or_finish()

    def _on_click(self, x: float, y: float) -> None:
        if self.session is None or self.pages.currentIndex() != PAGE_REVIEW:
            return
        self.session.answer_click(self._index, x, y)
        self._next()

    def not_visible(self) -> None:
        """The animal cannot be seen in this frame."""
        if self.session is None or self.pages.currentIndex() != PAGE_REVIEW:
            return
        self.session.answer_not_visible(self._index)
        self._next()

    def back(self) -> None:
        if self.session is None or self._index <= 0 \
                or self.pages.currentIndex() != PAGE_REVIEW:
            return
        self._index -= 1
        self._show_sample()

    def _next(self) -> None:
        self._index += 1
        if self._index >= len(self.session):
            self._advance_or_finish()
        else:
            self._show_sample()

    def _advance_or_finish(self) -> None:
        pending = self.session.first_pending()
        if pending is None:
            self.finish()
        else:
            self._index = pending
            self._show_sample()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self.pages.currentIndex() == PAGE_REVIEW:
            if event.key() == Qt.Key.Key_N:
                self.not_visible()
                event.accept()
                return
            if event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Left):
                self.back()
                event.accept()
                return
        super().keyPressEvent(event)

    # ----------------------------------------------------------------- finish
    def finish(self) -> None:
        """Evaluate, save and show the results."""
        session = self.session
        video = Path(self.video_edit.text().strip())
        tracking = Path(self.tracking_edit.text().strip())
        output = self.output_edit.text().strip()
        output_dir = Path(output) if output else video.parent
        base = base_name_of(tracking, video)
        provenance = {
            "scorer_initials": sanitize_initials(self.initials_edit.text()),
            "video": str(video),
            "tracking_csv": str(tracking),
            "summary_json": self.summary_edit.text().strip() or None,
            "blind": True,
        }
        try:
            csv_path, json_path = session.write_outputs(
                output_dir, base, provenance)
            self.saved_files = (csv_path, json_path)
            saved = (f"Saved:<br>{csv_path}<br>{json_path}")
        except OSError as error:
            logger.error("Could not save the spot-check", exc_info=True)
            QMessageBox.warning(self, "Could not save", str(error))
            saved = f"<span style='color:#a00'>Not saved: {error}</span>"
        self._release_capture()
        self.results_label.setText(self._results_html(saved))
        self.pages.setCurrentIndex(PAGE_RESULTS)
        if self.saved_files:
            self.completed.emit(*(str(path) for path in self.saved_files))

    def _results_html(self, saved: str) -> str:
        data = self.session.summary()

        def fmt(value, digits=2):
            return "n/a" if value is None else f"{value:.{digits}f}"

        calibrated = data["calibrated"]
        miss = data["miss_rate"]
        rows = [
            ("Frames reviewed", data["reviewed"]),
            ("Animal visible", data["visible"]),
            ("Animal not visible", data["not_visible"]),
            ("Tracker found the animal (matched)", data["matched"]),
            ("Tracker missed a visible animal",
             f"{data['missed']} (miss rate "
             f"{'n/a' if miss is None else f'{miss * 100:.1f}%'})"),
            ("False detections (tracker reported a position, animal not "
             "visible)", data["false_detections"]),
            ("Position error, mean / median / 95th percentile (px)",
             f"{fmt(data['mean_error_px'])} / {fmt(data['median_error_px'])}"
             f" / {fmt(data['p95_error_px'])}"),
            ("Position error (cm)",
             (f"{fmt(data['mean_error_cm'], 3)} / "
              f"{fmt(data['median_error_cm'], 3)} / "
              f"{fmt(data['p95_error_cm'], 3)}") if calibrated
             else "n/a (the run was not calibrated)"),
            ("Random seed", data["seed"]),
        ]
        body = "".join(
            f"<tr><td>{label}&nbsp;&nbsp;</td><td><b>{value}</b></td></tr>"
            for label, value in rows)
        unreadable = (
            f"<p>{data['unreadable']} frame(s) could not be decoded and "
            "were skipped.</p>" if data["unreadable"] else "")
        return (f"<h3>Spot-check results</h3><table>{body}</table>"
                f"{unreadable}<p>{saved}</p>")

    # ---------------------------------------------------------------- closing
    def _release_capture(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def done(self, result: int) -> None:
        self._release_capture()
        super().done(result)

    def closeEvent(self, event) -> None:
        self._release_capture()
        super().closeEvent(event)


def open_position_spotcheck(parent=None, app_settings=None) -> None:
    dialog = PositionSpotcheckDialog(parent, app_settings)
    try:
        dialog.exec()
    finally:
        dialog.deleteLater()


__all__ = ["ClickableFrameLabel", "PositionSpotcheckDialog",
           "open_position_spotcheck"]
