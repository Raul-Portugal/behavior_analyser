import unittest

from core.tst_events import (
    bridge_short_false_gaps,
    retain_minimum_true_runs,
    smooth_tst_immobility_states,
)


class TSTEventTests(unittest.TestCase):
    def test_short_bounded_motion_gap_is_bridged(self):
        states = [True, True, False, False, True, True]
        self.assertEqual(
            bridge_short_false_gaps(states, 2),
            [True, True, True, True, True, True],
        )

    def test_long_or_unbounded_gaps_are_not_bridged(self):
        self.assertEqual(
            bridge_short_false_gaps([False, True, False, False, True], 1),
            [False, True, False, False, True],
        )

    def test_minimum_duration_is_applied_after_gap_bridging(self):
        bridged = bridge_short_false_gaps(
            [True, True, False, True, True, False, False, True], 1
        )
        self.assertEqual(
            retain_minimum_true_runs(bridged, 4),
            [True, True, True, True, True, False, False, False],
        )

    def test_uncertain_state_breaks_bouts_and_is_not_scored_mobile(self):
        states = [
            "immobile_static",
            "immobile_static",
            "uncertain",
            "passive_swing",
            "passive_swing",
        ]
        self.assertEqual(
            smooth_tst_immobility_states(states, 2, 1),
            [True, True, None, True, True],
        )

    def test_short_active_gap_between_passive_states_is_bridged(self):
        states = [
            "passive_swing",
            "passive_swing",
            "active_escape",
            "immobile_static",
            "immobile_static",
        ]
        self.assertEqual(
            smooth_tst_immobility_states(states, 2, 1),
            [True, True, True, True, True],
        )


if __name__ == "__main__":
    unittest.main()
