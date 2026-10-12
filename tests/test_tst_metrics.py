import unittest

try:
    from mazes.tst import (
        TST, TSTResult, count_bouts_split_by_uncertainty,
        describe_window_truncation,
    )
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

    def _result(self, states, fps=10.0, per_roi_times=None):
        return TSTResult(
            fps=fps,
            timestamps=[index / fps for index in range(len(states))],
            per_roi_times=per_roi_times,
            motion_evidence={
                "mouse_1": [
                    {"state": state, "confidence": 1.0} for state in states
                ]
            },
        )

    def test_window_beyond_video_is_reported_as_truncated(self):
        result = self._result(
            ["immobile_static"] * 100, per_roi_times={"mouse_1": (0.0, 60.0)}
        )
        tst = TST()
        tst.calculate_metrics(result)

        self.assertEqual(result.requested_window_s["mouse_1"], (0.0, 60.0))
        self.assertEqual(result.window_coverage_s["mouse_1"], (0.0, 10.0))
        self.assertEqual(result.quality_metrics["mouse_1"]["window_truncated"], 1.0)
        self.assertEqual(
            describe_window_truncation(result, "mouse_1"),
            "mouse_1: requested window 0-60 s but video covers 0-10 s",
        )
        row = dict(zip(
            tst.get_batch_summary_headers(), tst.get_batch_summary_row(result)[0]
        ))
        self.assertEqual(row["window_start"], "0.0")
        self.assertEqual(row["window_end"], "10.0")
        self.assertEqual(row["requested_window_end"], "60.0")
        self.assertEqual(row["window_truncated"], 1)

    def test_window_within_tolerance_is_not_truncated(self):
        # Requested end is 0.4 s past the last frame: within frame + 0.5 s.
        result = self._result(
            ["immobile_static"] * 100, per_roi_times={"mouse_1": (0.0, 10.4)}
        )
        TST().calculate_metrics(result)
        self.assertEqual(result.quality_metrics["mouse_1"]["window_truncated"], 0.0)
        self.assertIsNone(describe_window_truncation(result, "mouse_1"))

    def test_inner_window_reports_covered_equal_to_request(self):
        result = self._result(
            ["immobile_static"] * 100, per_roi_times={"mouse_1": (2.0, 6.0)}
        )
        TST().calculate_metrics(result)
        self.assertEqual(result.window_coverage_s["mouse_1"], (2.0, 6.0))
        self.assertEqual(result.quality_metrics["mouse_1"]["window_truncated"], 0.0)

    def test_window_outside_trace_is_truncated_and_summary_is_rectangular(self):
        result = self._result(
            ["immobile_static"] * 10, per_roi_times={"mouse_1": (20.0, 30.0)}
        )
        tst = TST()
        tst.calculate_metrics(result)
        self.assertEqual(result.quality_metrics["mouse_1"]["window_truncated"], 1.0)
        rows = tst.get_batch_summary_row(result)
        self.assertEqual(len(rows[0]), len(tst.get_batch_summary_headers()))

    def test_bouts_split_only_by_uncertainty_are_counted(self):
        self.assertEqual(
            count_bouts_split_by_uncertainty(
                [True, True, None, None, True, False, True, None, False, True]
            ),
            1,
        )
        self.assertEqual(
            count_bouts_split_by_uncertainty(
                [True, None, True, None, True]
            ),
            2,
        )
        self.assertEqual(
            count_bouts_split_by_uncertainty([None, True, False, None, True]),
            0,
        )

    def test_split_bouts_are_exposed_in_metrics_and_summary(self):
        states = (
            ["immobile_static"] * 12 + ["uncertain"] * 3
            + ["immobile_static"] * 12 + ["active_escape"] * 12
            + ["immobile_static"] * 12
        )
        result = self._result(states)
        tst = TST()
        tst.calculate_metrics(result)

        self.assertEqual(result.immobility_bouts["mouse_1"], 3)
        self.assertEqual(result.bouts_split_by_uncertainty["mouse_1"], 1)
        self.assertEqual(
            result.quality_metrics["mouse_1"]["bouts_split_by_uncertainty"], 1.0
        )
        row = dict(zip(
            tst.get_batch_summary_headers(), tst.get_batch_summary_row(result)[0]
        ))
        self.assertEqual(row["bouts_split_by_uncertainty"], 1)
        self.assertEqual(row["window_truncated"], 0)


if __name__ == "__main__":
    unittest.main()
