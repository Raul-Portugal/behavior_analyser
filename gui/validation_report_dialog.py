"""Build the criterion-validation (agreement) report from the GUI.

Opened from Validation > "Build validation report...". The user adds
automated run folders and manual annotation folders or files; they are matched
per video by filename stem, and the matching table lets the user review it and
fix it (set the paradigm, choose another automated summary, manual files or
spot-check file for a row, remove a row). Then the event tolerance, the
acceptance criteria (for example ICC >= 0.90) and the output folder are chosen
and the report is built: a self-contained HTML file plus one CSV per table
(see :mod:`core.validation_report`). The report opens when it is done.
"""
import logging
import time
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                             QDoubleSpinBox, QFileDialog, QFormLayout,
                             QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                             QLineEdit, QMessageBox, QPlainTextEdit,
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from core import validation_report as vr
from core.manual_scoring import PARADIGM_LABELS
from gui.background_task import TaskCancelled, run_with_progress
from gui.dialog_layout import fit_dialog_to_available_screen

logger = logging.getLogger(__name__)

COL_VIDEO, COL_PARADIGM, COL_AUTO, COL_MANUAL, COL_SPOT, COL_STATUS = range(6)
COLUMN_TITLES = ["Video", "Paradigm", "Automated run", "Manual scoring",
                 "Position spot-check", "Status"]

CRIT_ENABLED, CRIT_KEY, CRIT_OP, CRIT_VALUE = range(4)

_MANUAL_ROLE_NAMES = {
    "manual_arm_entries": "arm entries",
    "manual_epm_totals": "EPM totals",
    "manual_zone_intervals": "zone intervals",
    "manual_tst_annotations": "TST annotations",
}


class ValidationReportDialog(QDialog):
    """Choose inputs, criteria and output, then build the report."""

    def __init__(self, parent=None, app_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Build validation report")
        self._settings = app_settings
        self.inputs: List[vr.VideoInputs] = []
        self.last_outcome: Optional[dict] = None
        # Replaceable so tests (and hosts) can intercept opening the report.
        self.open_url = QDesktopServices.openUrl
        self._output_set_by_user = False
        self._build_ui()
        self.reset_criteria()
        fit_dialog_to_available_screen(
            self, preferred_size=(1100, 860), minimum_size=(760, 560))

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        intro = QLabel(
            "Compare automated results with blinded manual scoring. Add the "
            "analysis run folder(s) and the folder(s) with the manual "
            "annotation files; they are matched by video name. Review the "
            "matching, set the acceptance criteria and build the report."
        )
        intro.setWordWrap(True)

        # --- inputs
        add_auto = QPushButton("Add automated run folder...")
        add_auto.clicked.connect(self._browse_automated)
        add_manual = QPushButton("Add manual annotations folder...")
        add_manual.clicked.connect(self._browse_manual_folder)
        add_files = QPushButton("Add manual files...")
        add_files.clicked.connect(self._browse_manual_files)
        self.remove_button = QPushButton("Remove selected rows")
        self.remove_button.clicked.connect(self.remove_selected)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear_inputs)
        buttons = QHBoxLayout()
        for button in (add_auto, add_manual, add_files, self.remove_button,
                       clear):
            buttons.addWidget(button)
        buttons.addStretch(1)

        self.table = QTableWidget(0, len(COLUMN_TITLES))
        self.table.setHorizontalHeaderLabels(COLUMN_TITLES)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        self.table.setColumnWidth(COL_VIDEO, 150)
        self.table.setColumnWidth(COL_PARADIGM, 150)
        self.table.setColumnWidth(COL_AUTO, 190)
        self.table.setColumnWidth(COL_MANUAL, 190)
        self.table.setColumnWidth(COL_SPOT, 160)
        self.table.cellDoubleClicked.connect(self._cell_double_clicked)
        self.table.setToolTip(
            "Double-click Automated run, Manual scoring or Position "
            "spot-check to choose a different file for that video.")
        self.matching_label = QLabel("")
        self.matching_label.setWordWrap(True)

        inputs_box = QGroupBox("1. Inputs and matching")
        inputs_layout = QVBoxLayout(inputs_box)
        inputs_layout.addLayout(buttons)
        inputs_layout.addWidget(self.table, 1)
        inputs_layout.addWidget(self.matching_label)

        # --- settings
        self.title_edit = QLineEdit("Criterion validation report")
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText(
            "Optional notes shown in the report (setup, scorers, blinding)")
        self.notes_edit.setFixedHeight(54)
        self.tolerance_spin = QDoubleSpinBox(
            decimals=2, minimum=0.0, maximum=60.0, singleStep=0.25, value=1.0)
        self.tolerance_spin.setSuffix(" s")
        self.tolerance_spin.setToolTip(
            "An automated and a manual arm entry match when they name the "
            "same arm and differ by at most this much")
        self.confidence_combo = QComboBox()
        for level in (0.90, 0.95, 0.99):
            self.confidence_combo.addItem(f"{level * 100:g}%", level)
        self.confidence_combo.setCurrentIndex(1)
        form = QFormLayout()
        form.addRow("Title:", self.title_edit)
        form.addRow("Notes:", self.notes_edit)
        form.addRow("Event tolerance:", self.tolerance_spin)
        form.addRow("Confidence level:", self.confidence_combo)
        settings_box = QGroupBox("2. Settings")
        settings_box.setLayout(form)

        # --- criteria
        self.criteria_table = QTableWidget(0, 4)
        self.criteria_table.setHorizontalHeaderLabels(
            ["Use", "Statistic", "Must be", "Threshold"])
        self.criteria_table.horizontalHeader().setSectionResizeMode(
            CRIT_KEY, QHeaderView.ResizeMode.Stretch)
        self.criteria_table.setColumnWidth(CRIT_ENABLED, 44)
        self.criteria_table.setColumnWidth(CRIT_OP, 70)
        self.criteria_table.setColumnWidth(CRIT_VALUE, 100)
        self.criteria_table.verticalHeader().setVisible(False)
        self.criteria_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        add_criterion = QPushButton("Add criterion")
        add_criterion.clicked.connect(lambda: self.add_criterion())
        remove_criterion = QPushButton("Remove selected")
        remove_criterion.clicked.connect(self.remove_selected_criteria)
        defaults = QPushButton("Protocol defaults")
        defaults.setToolTip(
            "ICC >= 0.90 (CI lower bound >= 0.75), event F1 >= 0.90, "
            "kappa >= 0.80 ... as in VALIDATION_PROTOCOL.md")
        defaults.clicked.connect(self.reset_criteria)
        criteria_buttons = QHBoxLayout()
        for button in (add_criterion, remove_criterion, defaults):
            criteria_buttons.addWidget(button)
        criteria_buttons.addStretch(1)
        criteria_box = QGroupBox(
            "3. Acceptance criteria (pre-register them before scoring)")
        criteria_layout = QVBoxLayout(criteria_box)
        criteria_layout.addWidget(self.criteria_table, 1)
        criteria_layout.addLayout(criteria_buttons)

        # --- output
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText(
            "Folder for the report and its CSV files")
        self.output_edit.textEdited.connect(self._output_edited)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_output)
        self.open_check = QCheckBox("Open the report when it is done")
        self.open_check.setChecked(True)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(browse)
        output_box = QGroupBox("4. Output")
        output_layout = QVBoxLayout(output_box)
        output_layout.addLayout(output_row)
        output_layout.addWidget(self.open_check)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.build_button = QPushButton("Build report")
        self.build_button.setDefault(True)
        self.build_button.clicked.connect(self.build)
        close = QPushButton("Close")
        close.setAutoDefault(False)
        close.clicked.connect(self.reject)
        bottom = QHBoxLayout()
        bottom.addWidget(self.status_label, 1)
        bottom.addWidget(close)
        bottom.addWidget(self.build_button)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(inputs_box, 3)
        side = QHBoxLayout()
        side.addWidget(settings_box, 2)
        side.addWidget(criteria_box, 3)
        side_widget = QWidget()
        side_widget.setLayout(side)
        layout.addWidget(side_widget, 3)
        layout.addWidget(output_box)
        layout.addLayout(bottom)

    # ------------------------------------------------------------- browsing
    def _start_dir(self, kind: str) -> str:
        if self._settings is not None:
            try:
                return self._settings.dialog_dir(kind)
            except Exception:
                logger.debug("dialog_dir failed", exc_info=True)
        return ""

    def _browse_automated(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select an analysis run folder (or its parent)",
            self._start_dir("output"))
        if folder:
            self.add_automated(folder)

    def _browse_manual_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select the folder with the manual annotation files",
            self._start_dir("video"))
        if folder:
            self.add_manual(folder)

    def _browse_manual_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select manual annotation files", self._start_dir("video"),
            "CSV (*.csv);;All Files (*.*)")
        if files:
            self.add_manual(files)

    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select the output folder", self.output_edit.text())
        if folder:
            self.output_edit.setText(folder)
            self._output_set_by_user = True

    def _output_edited(self, _text: str) -> None:
        self._output_set_by_user = True

    # ---------------------------------------------------------------- inputs
    def add_automated(self, sources) -> int:
        """Add run folders / summary files; returns the number of new runs."""
        found = vr.combine_inputs(vr.discover_automated(sources), {})
        return self._merge(found, sources)

    def add_manual(self, sources) -> int:
        """Add manual annotation folders / files; returns the files matched."""
        found = vr.combine_inputs({}, vr.discover_manual(sources))
        return self._merge(found, sources)

    def _merge(self, found: List[vr.VideoInputs], sources) -> int:
        existing = {item.stem.lower(): item for item in self.inputs}
        added = 0
        for item in found:
            current = existing.get(item.stem.lower())
            if current is None:
                self.inputs.append(item)
                existing[item.stem.lower()] = item
            else:
                current.merge(item)
            added += 1
        if not self._output_set_by_user and not self.output_edit.text().strip():
            first = Path(sources if isinstance(sources, (str, Path))
                         else list(sources)[0])
            base = first if first.is_dir() else first.parent
            self.output_edit.setText(str(
                base / f"validation_report_{time.strftime('%Y%m%d_%H%M%S')}"))
        self.inputs.sort(key=lambda v: v.stem.lower())
        self._refresh_table()
        return added

    def clear_inputs(self) -> None:
        self.inputs = []
        self._refresh_table()

    def remove_selected(self) -> None:
        rows = sorted({index.row() for index in
                       self.table.selectionModel().selectedRows()},
                      reverse=True)
        for row in rows:
            if 0 <= row < len(self.inputs):
                del self.inputs[row]
        self._refresh_table()

    def _manual_summary(self, item: vr.VideoInputs) -> str:
        names = [label for role, label in _MANUAL_ROLE_NAMES.items()
                 if getattr(item, role) is not None]
        return ", ".join(names) if names else "(none)"

    def status_of(self, item: vr.VideoInputs) -> str:
        """A one-word state of a row for the matching table."""
        has_manual = any(getattr(item, role) is not None
                         for role in _MANUAL_ROLE_NAMES)
        if item.paradigm is None and (has_manual or item.summary):
            return "set the paradigm"
        if item.summary is None and has_manual:
            return "no automated run"
        if item.summary is not None and not has_manual:
            return "no manual scoring"
        if item.summary is None and item.spotcheck is not None:
            return "spot-check only"
        if item.summary is None:
            return "empty"
        return "ready"

    def _refresh_table(self) -> None:
        self.table.setRowCount(0)
        for row, item in enumerate(self.inputs):
            self.table.insertRow(row)
            cells = {
                COL_VIDEO: item.stem,
                COL_AUTO: item.summary.name if item.summary else "(none)",
                COL_MANUAL: self._manual_summary(item),
                COL_SPOT: item.spotcheck.name if item.spotcheck else "(none)",
                COL_STATUS: self.status_of(item),
            }
            for column, text in cells.items():
                cell = QTableWidgetItem(text)
                tips = [f"{role}: {path}" for role, path in item.files().items()]
                cell.setToolTip("\n".join(tips))
                self.table.setItem(row, column, cell)
            combo = QComboBox()
            combo.addItem("(unknown)", None)
            for paradigm, label in PARADIGM_LABELS.items():
                combo.addItem(label, paradigm)
            index = combo.findData(item.paradigm)
            combo.setCurrentIndex(max(index, 0))
            combo.currentIndexChanged.connect(
                lambda _i, it=item, cb=combo: self._paradigm_changed(it, cb))
            self.table.setCellWidget(row, COL_PARADIGM, combo)
        ready = sum(self.status_of(item) == "ready" for item in self.inputs)
        self.matching_label.setText(
            f"{len(self.inputs)} video(s), {ready} ready (an automated run "
            "and manual scoring). Rows that are not ready are skipped with a "
            "warning in the report.")

    def _paradigm_changed(self, item: vr.VideoInputs, combo: QComboBox) -> None:
        item.paradigm = combo.currentData()
        row = self.inputs.index(item)
        self.table.item(row, COL_STATUS).setText(self.status_of(item))

    def set_paradigm(self, stem: str, paradigm: Optional[str]) -> None:
        for item in self.inputs:
            if item.stem == stem:
                item.paradigm = paradigm
        self._refresh_table()

    def _cell_double_clicked(self, row: int, column: int) -> None:
        if not 0 <= row < len(self.inputs):
            return
        item = self.inputs[row]
        if column == COL_AUTO:
            path, _ = QFileDialog.getOpenFileName(
                self, f"Automated summary for {item.stem}",
                str(item.summary.parent) if item.summary else
                self._start_dir("output"),
                "Summary (*_summary.json);;JSON (*.json)")
            if path:
                self.assign_automated(item.stem, path)
        elif column == COL_MANUAL:
            files, _ = QFileDialog.getOpenFileNames(
                self, f"Manual scoring files for {item.stem}",
                self._start_dir("video"), "CSV (*.csv);;All Files (*.*)")
            if files:
                self.assign_manual(item.stem, files)
        elif column == COL_SPOT:
            path, _ = QFileDialog.getOpenFileName(
                self, f"Position spot-check for {item.stem}",
                self._start_dir("video"), "CSV (*.csv);;All Files (*.*)")
            if path:
                self.assign_manual(item.stem, [path])

    def assign_automated(self, stem: str, summary_path) -> bool:
        """Use ``summary_path`` (and its sibling CSVs) for the row ``stem``."""
        found = vr.discover_automated(Path(summary_path))
        if not found:
            return False
        new = next(iter(found.values()))
        for item in self.inputs:
            if item.stem == stem:
                for role in vr.AUTO_ROLES:
                    setattr(item, role, getattr(new, role))
                item.paradigm = new.paradigm or item.paradigm
                self._refresh_table()
                return True
        return False

    def assign_manual(self, stem: str, paths) -> int:
        """Use manual files for the row ``stem``, whatever they are named."""
        count = 0
        for item in self.inputs:
            if item.stem != stem:
                continue
            for path in paths:
                classified = vr.classify_manual_file(path)
                if classified is None:
                    continue
                setattr(item, classified[0], Path(path))
                count += 1
        self._refresh_table()
        return count

    # --------------------------------------------------------------- criteria
    def reset_criteria(self) -> None:
        self.criteria_table.setRowCount(0)
        for criterion in vr.default_criteria():
            self.add_criterion(criterion)

    def add_criterion(self, criterion: Optional[vr.Criterion] = None) -> None:
        row = self.criteria_table.rowCount()
        self.criteria_table.insertRow(row)
        use = QCheckBox()
        use.setChecked(True)
        holder = QWidget()
        holder_layout = QHBoxLayout(holder)
        holder_layout.setContentsMargins(12, 0, 0, 0)
        holder_layout.addWidget(use)
        self.criteria_table.setCellWidget(row, CRIT_ENABLED, holder)
        holder.setProperty("use_check", True)
        key_combo = QComboBox()
        for key, label in vr.criteria_catalog():
            key_combo.addItem(label, key)
        op_combo = QComboBox()
        for op in (">=", "<=", ">", "<"):
            op_combo.addItem(op)
        value = QDoubleSpinBox(decimals=3, minimum=-1e9, maximum=1e9,
                               singleStep=0.05)
        if criterion is not None:
            key_combo.setCurrentIndex(max(key_combo.findData(criterion.key), 0))
            op_combo.setCurrentText(criterion.op)
            value.setValue(criterion.threshold)
        self.criteria_table.setCellWidget(row, CRIT_KEY, key_combo)
        self.criteria_table.setCellWidget(row, CRIT_OP, op_combo)
        self.criteria_table.setCellWidget(row, CRIT_VALUE, value)

    def remove_selected_criteria(self) -> None:
        rows = sorted({index.row() for index in
                       self.criteria_table.selectionModel().selectedRows()},
                      reverse=True)
        for row in rows:
            self.criteria_table.removeRow(row)

    def criteria(self) -> List[vr.Criterion]:
        """The checked criteria rows."""
        criteria = []
        for row in range(self.criteria_table.rowCount()):
            holder = self.criteria_table.cellWidget(row, CRIT_ENABLED)
            if not holder.findChild(QCheckBox).isChecked():
                continue
            criteria.append(vr.Criterion(
                self.criteria_table.cellWidget(row, CRIT_KEY).currentData(),
                self.criteria_table.cellWidget(row, CRIT_OP).currentText(),
                self.criteria_table.cellWidget(row, CRIT_VALUE).value()))
        return criteria

    def set_criterion_enabled(self, row: int, enabled: bool) -> None:
        holder = self.criteria_table.cellWidget(row, CRIT_ENABLED)
        holder.findChild(QCheckBox).setChecked(enabled)

    # ------------------------------------------------------------------ build
    def config(self) -> vr.ReportConfig:
        return vr.ReportConfig(
            title=self.title_edit.text().strip() or "Validation report",
            notes=self.notes_edit.toPlainText(),
            tolerance_s=self.tolerance_spin.value(),
            confidence=float(self.confidence_combo.currentData()),
            criteria=self.criteria(),
        )

    def output_dir(self) -> Optional[Path]:
        text = self.output_edit.text().strip()
        return Path(text) if text else None

    def build(self) -> Optional[dict]:
        """Compute and write the report; returns the outcome (None on failure)."""
        output = self.output_dir()
        if not self.inputs:
            self.status_label.setText("Add automated runs and manual files first.")
            return None
        if output is None:
            self.status_label.setText("Choose an output folder.")
            return None
        config = self.config()
        inputs = list(self.inputs)

        def work(_should_cancel):
            return vr.build_report(inputs, config, output)

        try:
            outcome = run_with_progress(
                self, "Building the validation report...", work,
                title="Validation report")
        except TaskCancelled:
            self.status_label.setText("Cancelled.")
            return None
        except Exception as error:
            logger.error("Validation report failed", exc_info=True)
            self.status_label.setText(f"The report could not be built: {error}")
            QMessageBox.warning(self, "Validation report", str(error))
            return None
        self.last_outcome = outcome
        results = outcome["results"]
        passed = sum(r.status == "pass" for r in results.criteria)
        failed = sum(r.status == "fail" for r in results.criteria)
        self.status_label.setText(
            f"Report written to {outcome['html']} - criteria: {passed} pass, "
            f"{failed} fail, {len(results.warnings)} warning(s). "
            f"{len(outcome['csv'])} CSV tables saved next to it.")
        if self.open_check.isChecked():
            self.open_url(QUrl.fromLocalFile(str(outcome["html"])))
        return outcome


def open_validation_report(parent=None, app_settings=None) -> None:
    dialog = ValidationReportDialog(parent, app_settings)
    try:
        dialog.exec()
    finally:
        dialog.deleteLater()


__all__ = ["ValidationReportDialog", "open_validation_report"]
