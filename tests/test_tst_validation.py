import unittest

from core.tst_validation import evaluate_annotations


class TSTValidationTests(unittest.TestCase):
    def test_evaluates_immobility_without_counting_unannotated_frames(self):
        evidence = {"mouse_1": [
            {"source_frame": 10, "state": "immobile_static"},
            {"source_frame": 20, "state": "active_escape"},
            {"source_frame": 30, "state": "passive_swing"},
        ]}
        report = evaluate_annotations(evidence, {
            ("mouse_1", 10): "immobile_static",
            ("mouse_1", 20): "active_escape",
        })
        self.assertEqual(report["mouse_1"]["annotated_frames"], 2.0)
        self.assertEqual(report["mouse_1"]["exact_state_agreement_pct"], 100.0)
        self.assertEqual(report["mouse_1"]["immobility_f1"], 1.0)

    def test_start_frame_fallback_aligns_rows_without_source_frame(self):
        states = ["immobile_static", "active_escape", "passive_swing"]
        evidence = {"mouse_1": [{"state": state} for state in states]}
        # A run that started at video frame 100: row 0 is frame 100.
        annotations = {
            ("mouse_1", 100): "immobile_static",
            ("mouse_1", 101): "active_escape",
            ("mouse_1", 102): "passive_swing",
        }
        aligned = evaluate_annotations(evidence, annotations, start_frame=100)
        self.assertEqual(aligned["mouse_1"]["annotated_frames"], 3.0)
        self.assertEqual(aligned["mouse_1"]["exact_state_agreement_pct"], 100.0)
        # Without the fallback the rows are compared with frames 0-2.
        self.assertNotIn("mouse_1", evaluate_annotations(evidence, annotations))

    def test_source_frame_wins_over_the_start_frame_fallback(self):
        evidence = {"mouse_1": [
            {"source_frame": 105, "state": "active_escape"},
        ]}
        report = evaluate_annotations(
            evidence, {("mouse_1", 105): "active_escape"}, start_frame=900
        )
        self.assertEqual(report["mouse_1"]["exact_state_agreement_pct"], 100.0)
