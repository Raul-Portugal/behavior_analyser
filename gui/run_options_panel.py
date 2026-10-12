"""
gui/run_options_panel.py
Run-level options shown under the Start button: the sample sheet (animal
metadata) and, later, the time-bin length.

These are options of the *run*, not of a video's settings: they are never part
of ``BatchSettings`` and so never change the checkpoint settings hash. The
panel only collects them; ``MainWindow`` hands them to ``AnalysisWorker``.
"""
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

from PyQt6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QWidget,
)

from core.output_contract import COMMON_BATCH_HEADERS
from core.sample_sheet import (
    SampleMetadata, SampleSheet, SampleSheetError, SampleSheetMatch,
    describe_match, load_sample_sheet, metadata_for_videos,
    reserved_column_conflicts,
)
from gui.app_settings import AppSettings


class RunOptionsPanel(QWidget):
    """Sample-sheet button + status label (and the run's other options)."""

    def __init__(
        self,
        app_settings: AppSettings,
        videos_provider: Callable[[], Sequence[Path]],
        maze_provider: Callable[[], object],
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._settings = app_settings
        self._videos = videos_provider
        self._maze = maze_provider
        self.sheet: Optional[SampleSheet] = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.sheet_button = QPushButton("Sample sheet…")
        self.sheet_button.setToolTip(
            "A CSV whose first column is 'video' and whose other columns "
            "(animal_id, group, sex, treatment, ...) are added to the batch "
            "summary, each video's summary.json and the run manifest."
        )
        self.sheet_button.clicked.connect(self.choose_sheet)
        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Do not use a sample sheet")
        self.clear_button.clicked.connect(self.clear_sheet)
        self.sheet_label = QLabel()
        layout.addWidget(self.sheet_button)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.sheet_label, 1)

        remembered = app_settings.sample_sheet()
        if remembered is not None:
            try:
                self.sheet = load_sample_sheet(remembered)
            except SampleSheetError:
                app_settings.set_sample_sheet(None)
        self.refresh()

    # -- sheet ---------------------------------------------------------
    def choose_sheet(self) -> None:
        start = self._settings.sample_sheet()
        directory = (
            str(start.parent) if start is not None
            else self._settings.dialog_dir("config")
        )
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Sample Sheet", directory,
            "CSV files (*.csv);;All files (*)",
        )
        if not filename:
            return
        try:
            sheet = load_sample_sheet(Path(filename))
        except SampleSheetError as exc:
            QMessageBox.warning(self, "Invalid Sample Sheet", str(exc))
            return
        problem = self._column_conflict_message(sheet)
        if problem:
            QMessageBox.warning(self, "Invalid Sample Sheet", problem)
            return
        self.sheet = sheet
        self._settings.set_sample_sheet(sheet.path)
        self.refresh()
        match = self.match()
        if match is not None and not match.is_complete:
            QMessageBox.warning(
                self, "Sample Sheet Check",
                describe_match(match, len(self._videos())),
            )
        elif match is not None and match.unused_rows:
            QMessageBox.information(
                self, "Sample Sheet Check",
                describe_match(match, len(self._videos())),
            )

    def clear_sheet(self) -> None:
        self.sheet = None
        self._settings.set_sample_sheet(None)
        self.refresh()

    def _column_conflict_message(self, sheet: SampleSheet) -> str:
        maze = self._maze()
        reserved = list(COMMON_BATCH_HEADERS)
        if maze is not None:
            reserved += list(maze.get_batch_summary_headers())
        conflicts = reserved_column_conflicts(sheet.columns, reserved)
        if not conflicts:
            return ""
        return (
            f"{sheet.path.name}: column(s) {', '.join(conflicts)} clash with "
            "columns of the batch summary. Rename them in the sheet."
        )

    def match(self) -> Optional[SampleSheetMatch]:
        """The sheet matched against the videos now in the list."""
        if self.sheet is None:
            return None
        return metadata_for_videos(self.sheet, self.sheet.path, self._videos())

    def refresh(self) -> None:
        """Update the label (call when the sheet or the video list changes)."""
        self.clear_button.setEnabled(self.sheet is not None)
        if self.sheet is None:
            self.sheet_label.setText("No sample sheet")
            self.sheet_label.setToolTip("")
            return
        text = f"{self.sheet.path.name} ({len(self.sheet.rows)} rows)"
        videos = list(self._videos())
        tooltip = str(self.sheet.path)
        if videos:
            match = self.match()
            missing = len(match.unmatched) + len(match.ambiguous)
            if missing:
                text += f" - {missing} of {len(videos)} video(s) unmatched"
                tooltip += "\n" + describe_match(match, len(videos))
            else:
                text += f" - all {len(videos)} video(s) matched"
        self.sheet_label.setText(text)
        self.sheet_label.setToolTip(tooltip)

    # -- run options -----------------------------------------------------
    def resolve_for_run(
        self, videos: Sequence[Path]
    ) -> Tuple[bool, Optional[SampleMetadata]]:
        """``(proceed, metadata)`` for the videos about to be analysed.

        Without a sheet this is ``(True, None)``. A video with no unique row
        is never guessed: the user is asked whether to continue with blank
        metadata for it.
        """
        if self.sheet is None:
            return True, None
        problem = self._column_conflict_message(self.sheet)
        if problem:
            QMessageBox.critical(self, "Sample Sheet Problem", problem)
            return False, None
        match = metadata_for_videos(self.sheet, self.sheet.path, videos)
        if not match.is_complete:
            answer = QMessageBox.question(
                self, "Sample Sheet Check",
                describe_match(match, len(videos))
                + "\n\nContinue with blank metadata for the unmatched videos?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False, None
        return True, match.metadata
