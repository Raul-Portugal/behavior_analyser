import unittest

try:
    from mazes.tst import TST, TSTResult
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class TSTMetricTests(unittest.TestCase):
    def test_mean_confidence_uses_selected_mouse_window(self):
        result = TSTResult(
            fps=10.0,
            timestamps=[0.0, 0.1, 0.2, 0.3],
            per_roi_times={"mouse_1": (0.2, 0.4)},
            motion_evidence={
                "mouse_1": [
                    {"state": "uncertain", "confidence": 0.0},
                    {"state": "uncertain", "confidence": 0.0},
                    {"state": "immobile_static", "confidence": 1.0},
                    {"state": "immobile_static", "confidence": 1.0},
                ]
            },
        )

        TST().calculate_metrics(result)

        self.assertEqual(result.mean_state_confidence["mouse_1"], 1.0)

    def test_empty_selected_window_is_explicitly_unscored(self):
        result = TSTResult(
            fps=10.0,
            timestamps=[0.0, 0.1],
            per_roi_times={"mouse_1": (2.0, 3.0)},
            motion_evidence={
                "mouse_1": [
                    {"state": "immobile_static", "confidence": 1.0},
                    {"state": "immobile_static", "confidence": 1.0},
                ]
            },
        )

        TST().calculate_metrics(result)

        self.assertEqual(result.valid_time["mouse_1"], 0.0)
        self.assertEqual(result.mean_state_confidence["mouse_1"], 0.0)
        self.assertEqual(result.immobile_states["mouse_1"], [None, None])

    def test_mouse_with_no_evidence_still_receives_explicit_metrics(self):
        result = TSTResult(
            fps=10.0,
            activity_scores={"mouse_1": []},
            motion_evidence={"mouse_1": []},
            per_roi_times={"mouse_1": (0.0, 1.0)},
        )

        TST().calculate_metrics(result)

        self.assertEqual(result.valid_time["mouse_1"], 0.0)
        self.assertEqual(result.uncertain_time["mouse_1"], 0.0)
        self.assertIsNone(
            result.latency_to_first_immobility["mouse_1"]
        )


if __name__ == "__main__":
    unittest.main()
