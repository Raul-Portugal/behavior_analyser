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
