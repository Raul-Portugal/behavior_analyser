import unittest
from pathlib import Path
from unittest.mock import patch

from core import validation_report as vr
from tests.test_validation_report import ReportTestCase, write_entries

try:
    from PyQt6.QtWidgets import QApplication, QMessageBox

    from gui import validation_report_dialog as vrd
    from gui.validation_report_dialog import ValidationReportDialog
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class ValidationReportDialogTests(ReportTestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        self.dialog = ValidationReportDialog()
        self.addCleanup(self.dialog.deleteLater)
        self.opened = []
        self.dialog.open_url = self.opened.append

    def rows(self):
        return {
            item.stem: self.dialog.status_of(item)
            for item in self.dialog.inputs
        }

    def test_adding_folders_matches_by_stem_and_lists_status(self):
        self.make_y_maze(3)
        self.make_tst(1)
        self.dialog.add_automated(self.auto)
        self.assertEqual(set(self.rows().values()), {"no manual scoring"})
        self.dialog.add_manual(self.manual)
        self.assertEqual(self.rows(),
                         {"tst1": "ready", "ym1": "ready", "ym2": "ready",
                          "ym3": "ready"})
        self.assertEqual(self.dialog.table.rowCount(), 4)
        self.assertIn("4 ready", self.dialog.matching_label.text())
        paradigms = {item.stem: item.paradigm for item in self.dialog.inputs}
        self.assertEqual(paradigms["ym1"], "y_maze")
        self.assertEqual(paradigms["tst1"], "tst")
        combo = self.dialog.table.cellWidget(0, vrd.COL_PARADIGM)
        self.assertEqual(combo.currentData(), self.dialog.inputs[0].paradigm)

    def test_adding_manual_first_then_automated_merges_rows(self):
        self.make_y_maze(2)
        self.dialog.add_manual(self.manual)
        self.assertEqual(set(self.rows().values()), {"set the paradigm"})
        self.dialog.add_automated(self.auto)
        self.assertEqual(self.dialog.table.rowCount(), 2)
        self.assertEqual(set(self.rows().values()), {"ready"})

    def test_unmatched_files_are_shown_and_can_be_fixed(self):
        self.make_y_maze(2)
        (self.manual / "ym2_manual_arm_entries.csv").unlink()
        stray = self.manual / "scorer_b_clip2.csv"
        write_entries(stray, self.y_manual["ym2"])
        self.dialog.add_automated(self.auto)
        self.dialog.add_manual(self.manual)
        rows = self.rows()
        self.assertEqual(rows["ym1"], "ready")
        self.assertEqual(rows["ym2"], "no manual scoring")
        self.assertEqual(rows["scorer_b_clip2"], "set the paradigm")

        # The user points ym2 at the stray file and drops the stray row.
        self.assertEqual(self.dialog.assign_manual("ym2", [stray]), 1)
        self.dialog.table.selectRow(
            [i.stem for i in self.dialog.inputs].index("scorer_b_clip2"))
        self.dialog.remove_selected()
        self.assertEqual(self.rows(), {"ym1": "ready", "ym2": "ready"})
        self.assertEqual(self.dialog.assign_manual("ym2", [self.root]), 0)

    def test_choosing_another_automated_summary_and_paradigm(self):
        self.make_y_maze(2)
        self.make_epm(1)
        self.dialog.add_automated(self.auto)
        self.assertTrue(self.dialog.assign_automated(
            "ym1", self.auto / "epm1" / "epm1_summary.json"))
        item = next(i for i in self.dialog.inputs if i.stem == "ym1")
        self.assertEqual(item.summary.name, "epm1_summary.json")
        self.assertEqual(item.paradigm, "epm")
        self.assertFalse(self.dialog.assign_automated("ym1", self.manual))
        self.dialog.set_paradigm("ym1", "y_maze")
        self.assertEqual(item.paradigm, "y_maze")
        row = self.dialog.inputs.index(item)
        combo = self.dialog.table.cellWidget(row, vrd.COL_PARADIGM)
        combo.setCurrentIndex(0)                        # (unknown)
        self.assertIsNone(item.paradigm)
        self.assertEqual(self.dialog.status_of(item), "set the paradigm")
        self.assertEqual(self.dialog.table.item(row, vrd.COL_STATUS).text(),
                         "set the paradigm")

    def test_clear_and_default_output_folder(self):
        self.make_y_maze(1)
        self.dialog.add_automated(self.auto)
        output = self.dialog.output_dir()
        self.assertEqual(output.parent, self.auto)
        self.assertTrue(output.name.startswith("validation_report_"))
        self.dialog.clear_inputs()
        self.assertEqual(self.dialog.table.rowCount(), 0)
        self.assertEqual(self.dialog.inputs, [])

    def test_criteria_defaults_edit_and_disable(self):
        defaults = vr.default_criteria()
        got = self.dialog.criteria()
        self.assertEqual([(c.key, c.op, c.threshold) for c in got],
                         [(c.key, c.op, c.threshold) for c in defaults])
        self.dialog.set_criterion_enabled(0, False)
        self.assertEqual(len(self.dialog.criteria()), len(defaults) - 1)
        self.dialog.add_criterion(
            vr.Criterion("position.spotcheck.median_error_cm", "<=", 0.25))
        last = self.dialog.criteria()[-1]
        self.assertEqual((last.key, last.op, last.threshold),
                         ("position.spotcheck.median_error_cm", "<=", 0.25))
        self.dialog.criteria_table.selectRow(
            self.dialog.criteria_table.rowCount() - 1)
        self.dialog.remove_selected_criteria()
        self.assertEqual(len(self.dialog.criteria()), len(defaults) - 1)
        self.dialog.reset_criteria()
        self.assertEqual(len(self.dialog.criteria()), len(defaults))

    def test_config_reads_the_settings(self):
        self.dialog.title_edit.setText("Lab study")
        self.dialog.notes_edit.setPlainText("Two scorers")
        self.dialog.tolerance_spin.setValue(2.5)
        self.dialog.confidence_combo.setCurrentIndex(2)
        config = self.dialog.config()
        self.assertEqual(config.title, "Lab study")
        self.assertEqual(config.notes, "Two scorers")
        self.assertEqual(config.tolerance_s, 2.5)
        self.assertEqual(config.confidence, 0.99)
        self.assertEqual(len(config.criteria), len(vr.default_criteria()))

    def test_build_writes_the_report_and_opens_it(self):
        self.make_y_maze(5)
        self.make_tst(2)
        self.make_spotchecks()
        self.dialog.add_automated(self.auto)
        self.dialog.add_manual(self.manual)
        out = self.root / "report_out"
        self.dialog.output_edit.setText(str(out))
        self.dialog.tolerance_spin.setValue(0.5)
        outcome = self.dialog.build()
        self.assertIsNotNone(outcome)
        html = out / "validation_report.html"
        self.assertTrue(html.is_file())
        self.assertEqual(len(self.opened), 1)
        self.assertEqual(Path(self.opened[0].toLocalFile()), html)
        self.assertIn(str(html), self.dialog.status_label.text())
        direct = vr.build_results(
            self.dialog.inputs, vr.ReportConfig(
                tolerance_s=0.5, criteria=vr.default_criteria()))
        self.assertAlmostEqual(
            outcome["results"].paradigms["y_maze"].events.f1,
            direct.paradigms["y_maze"].events.f1)
        self.assertEqual(outcome["results"].config.tolerance_s, 0.5)
        self.assertTrue(list(out.glob("validation_report_*.csv")))

    def test_report_is_not_opened_when_unchecked(self):
        self.make_y_maze(3)
        self.dialog.add_automated(self.auto)
        self.dialog.add_manual(self.manual)
        self.dialog.output_edit.setText(str(self.root / "o"))
        self.dialog.open_check.setChecked(False)
        self.assertIsNotNone(self.dialog.build())
        self.assertEqual(self.opened, [])

    def test_build_needs_inputs_and_an_output_folder(self):
        self.assertIsNone(self.dialog.build())
        self.assertIn("Add automated", self.dialog.status_label.text())
        self.make_y_maze(2)
        self.dialog.add_automated(self.auto)
        self.dialog.output_edit.setText("")
        self.assertIsNone(self.dialog.build())
        self.assertIn("output folder", self.dialog.status_label.text())

    def test_a_failing_build_is_reported_not_raised(self):
        self.make_y_maze(2)
        self.dialog.add_automated(self.auto)
        self.dialog.output_edit.setText(str(self.root / "o"))
        with patch.object(vr, "build_report",
                          side_effect=OSError("disk full")), \
                patch.object(QMessageBox, "warning") as warning:
            self.assertIsNone(self.dialog.build())
        warning.assert_called_once()
        self.assertIn("disk full", self.dialog.status_label.text())
        self.assertEqual(self.opened, [])


if __name__ == "__main__":
    unittest.main()
