import unittest

from core.zone_events import (
    debounce_zone_labels,
    extract_center_mediated_entries,
    extract_zone_transitions,
    overlapping_zones_to_candidates,
)


class ZoneDebounceTests(unittest.TestCase):
    def test_short_excursion_is_suppressed(self):
        labels = debounce_zone_labels(
            ['center', 'center', 'arm_a', 'center', 'center'],
            min_dwell_frames=2
        )

        self.assertEqual(
            labels,
            ['center', 'center', 'center', 'center', 'center']
        )

    def test_unconfirmed_initial_zone_is_not_treated_as_an_entry(self):
        labels = debounce_zone_labels(
            ['arm_a', 'center', 'center'],
            min_dwell_frames=2
        )

        self.assertEqual(labels, ['outside', 'center', 'center'])

    def test_overlap_retains_stable_zone(self):
        candidates = overlapping_zones_to_candidates([
            ['center'],
            ['center'],
            ['center', 'arm_a'],
            ['arm_a'],
            ['arm_a']
        ])
        labels = debounce_zone_labels(candidates, min_dwell_frames=2)

        self.assertEqual(
            labels, ['center', 'center', 'center', 'arm_a', 'arm_a']
        )

    def test_missing_tracking_breaks_transition_continuity(self):
        transitions = extract_zone_transitions(
            ['zone_a', 'undetected', 'zone_b', 'outside']
        )

        self.assertEqual(transitions, [('zone_b', 'outside', 3)])


class CenterEntryTests(unittest.TestCase):
    def test_only_initial_or_center_mediated_arm_entries_count(self):
        labels = [
            'arm_a', 'arm_a', 'center', 'arm_b', 'arm_c',
            'undetected', 'arm_a', 'center', 'arm_c'
        ]
        timestamps = [index / 10 for index in range(len(labels))]

        entries = extract_center_mediated_entries(
            labels,
            timestamps,
            arm_labels={'arm_a', 'arm_b', 'arm_c'}
        )

        self.assertEqual(
            entries,
            [('arm_a', 0.0), ('arm_b', 0.3), ('arm_c', 0.8)]
        )


if __name__ == '__main__':
    unittest.main()
