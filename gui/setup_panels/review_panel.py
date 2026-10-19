"""Review panel: read-only summary, blocking issues, validation-video option."""
from typing import List

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QLabel, QVBoxLayout

from core.setup_state import STATUS_INVALID, STEP_LABELS
from gui.setup_panels.base import SetupPanel
from mazes.base_maze import STEP_REVIEW

VALIDATION_VIDEO_TEXT = "Create validation video (annotated)"


class ReviewPanel(SetupPanel):
    """What will run, what is wrong with it, and the validation-video option.

    ``show_summary=False`` leaves only the validation-video checkbox (the
    classic settings dialog already shows everything else as editable
    controls).
    """

    step_id = STEP_REVIEW

    def __init__(self, parent=None, show_summary: bool = True,
                 preserve_exact: bool = True):
        super().__init__(parent, preserve_exact)
        self.show_summary = show_summary
        self._issues: List[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(Qt.TextFormat.RichText)
        self.issues_label = QLabel()
        self.issues_label.setWordWrap(True)
        self.issues_label.setStyleSheet("color: #b00020;")
        if show_summary:
            layout.addWidget(self.summary_label)
            layout.addWidget(self.issues_label)
        else:
            self.summary_label.setVisible(False)
            self.issues_label.setVisible(False)

        # One name everywhere: the annotated video is the "validation video".
        self.timelapse_check = QCheckBox(VALIDATION_VIDEO_TEXT)
        self.timelapse_check.setToolTip(
            "Writes an annotated video so the tracking can be checked by eye. "
            "Slower, but recommended for new setups."
        )
        layout.addWidget(self.timelapse_check)
        self.timelapse_check.toggled.connect(self._on_option_toggled)
        self._state = None

    def _on_option_toggled(self, checked: bool) -> None:
        self._emit_changed()
        if self._state is not None and not self._loading:
            self._render(self._state)

    def load(self, state) -> None:
        self._state = state
        self._loading = True
        try:
            self.timelapse_check.setChecked(bool(state.create_timelapse))
        finally:
            self._loading = False
        self._render(state)

    def refresh(self) -> None:
        """Re-read the loaded state (other panels may have changed it)."""
        if self._state is not None:
            self._render(self._state)

    def _render(self, state) -> None:
        if not self.show_summary:
            return
        reviewable = state.copy()
        reviewable.create_timelapse = self.timelapse_check.isChecked()
        report = reviewable.step_report(STEP_REVIEW)
        issues = list(report.issues) if report.status == STATUS_INVALID else []
        # Steps not finished yet are listed too: a run needs them done.
        for step, step_report in reviewable.step_reports().items():
            if step != STEP_REVIEW and step_report.status == "todo":
                issues.append(f"{STEP_LABELS[step]} is not done yet")
        self._issues = issues
        rows = "".join(
            f"<tr><td><b>{label}:</b>&nbsp;</td><td>{value}</td></tr>"
            for label, value in reviewable.summary_lines()
        )
        self.summary_label.setText(f"<table>{rows}</table>")
        self.issues_label.setText(
            "<b>To fix before running:</b><br>"
            + "<br>".join(f"&bull; {issue}" for issue in issues)
            if issues else ""
        )

    def issues(self) -> List[str]:
        """Blocking problems (and unfinished steps) of the loaded state."""
        return list(self._issues)

    def apply(self, state) -> bool:
        state.create_timelapse = self.timelapse_check.isChecked()
        if self.show_summary:
            self._render(state)
            if self._issues:
                return False
        state.mark_done(self.step_id)
        return True
