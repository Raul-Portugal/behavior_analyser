import unittest
from pathlib import Path
from types import SimpleNamespace

from core.output_contract import COMMON_BATCH_HEADERS, common_batch_row


class OutputContractTests(unittest.TestCase):
    def test_position_tracking_includes_total_distance_and_units(self):
        result = SimpleNamespace(
            positions=[(1, 1), (2, 2)],
            timestamps=[0.0, 0.1],
            total_distance=12.5,
            fps=10.0,
            detection_rate=1.0,
            tracking_quality_grade="Excellent",
            tracking_quality={},
            undetected_time_s=0.0,
            detection_confidences=[1.0, 1.0],
            distance_unit="pixels",
        )
        row = common_batch_row(Path("epm.mp4"), result, "Elevated Plus Maze")
        values = dict(zip(COMMON_BATCH_HEADERS, row))
        self.assertEqual(values["total_distance"], "12.50")
        self.assertEqual(values["distance_unit"], "pixels")
        self.assertEqual(values["analyzed_frames"], 2)

    def test_non_tracking_paradigm_marks_tracking_fields_not_applicable(self):
        result = SimpleNamespace(
            positions=[],
            timestamps=[0.0, 0.1],
            fps=10.0,
            detection_confidences=[],
            motion_evidence={"mouse_1": [{"state": "active_escape"}]},
        )
        row = common_batch_row(
            Path("tst.mp4"), result, "Tail Suspension Test"
        )
        values = dict(zip(COMMON_BATCH_HEADERS, row))
        self.assertEqual(values["total_distance"], "")
        self.assertEqual(values["tracking_quality"], "Not applicable")


if __name__ == "__main__":
    unittest.main()
