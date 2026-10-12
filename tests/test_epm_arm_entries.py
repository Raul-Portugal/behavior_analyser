import csv
import tempfile
import unittest
from pathlib import Path

try:
    from mazes.epm import EPM, EPMAnalysisResult
    from mazes.y_maze import YMazeAnalysisResult
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class EPMArmEntriesExportTests(unittest.TestCase):
    def test_writes_ordered_entries_like_the_y_maze(self):
        result = EPMAnalysisResult(
            arm_entries=[
                ("open_arm_1", 1.25), ("closed_arm_2", 4.0005),
                ("open_arm_2", 9.5),
            ]
        )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            EPM().export_specific_data(result, Path(tmp), "clip")
            path = Path(tmp) / "clip_arm_entries.csv"
            with open(path, newline="", encoding="utf-8") as file:
                rows = list(csv.reader(file))
        self.assertEqual(rows[0], ["entry_number", "arm", "timestamp_s"])
        self.assertEqual(
            rows[1:],
            [
                ["1", "open_arm_1", "1.250"],
                ["2", "closed_arm_2", "4.000"],
                ["3", "open_arm_2", "9.500"],
            ],
        )

    def test_header_only_when_there_are_no_entries(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            EPM().export_specific_data(EPMAnalysisResult(), Path(tmp), "x")
            text = (Path(tmp) / "x_arm_entries.csv").read_text(
                encoding="utf-8"
            )
        self.assertEqual(text.strip(), "entry_number,arm,timestamp_s")

    def test_other_result_types_write_nothing(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            EPM().export_specific_data(
                YMazeAnalysisResult(), Path(tmp), "x"
            )
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
