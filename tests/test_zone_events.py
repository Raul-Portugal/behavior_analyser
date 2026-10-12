import unittest

from core.zone_events import (
    analyze_center_mediated_entries,
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


ARMS = {'arm_a', 'arm_b', 'arm_c'}


def _analyze(labels, **kwargs):
    timestamps = [index / 10 for index in range(len(labels))]
    return analyze_center_mediated_entries(
        labels, timestamps, arm_labels=ARMS, **kwargs
    )


class EntriesAcrossGapTests(unittest.TestCase):
    def test_arm_change_across_gap_is_flagged_not_counted(self):
        result = _analyze(
            ['center', 'arm_a', 'undetected', 'undetected', 'arm_b', 'center', 'arm_c']
        )
        self.assertEqual(result.entries, [('arm_a', 0.1), ('arm_c', 0.6)])
        self.assertEqual(result.possibly_missed, 1)

    def test_same_arm_after_gap_is_not_flagged(self):
        result = _analyze(
            ['center', 'arm_a', 'undetected', 'arm_a', 'center', 'arm_b']
        )
        self.assertEqual(result.entries, [('arm_a', 0.1), ('arm_b', 0.5)])
        self.assertEqual(result.possibly_missed, 0)

    def test_gap_after_center_resuming_in_arm_is_flagged(self):
        result = _analyze(['arm_a', 'center', 'undetected', 'arm_a'])
        self.assertEqual(result.entries, [('arm_a', 0.0)])
        self.assertEqual(result.possibly_missed, 1)

    def test_confirmed_center_after_gap_clears_the_flag(self):
        result = _analyze(
            ['arm_a', 'undetected', 'center', 'arm_b']
        )
        self.assertEqual(result.entries, [('arm_a', 0.0), ('arm_b', 0.3)])
        self.assertEqual(result.possibly_missed, 0)

    def test_leading_gap_is_not_a_missed_entry(self):
        result = _analyze(['undetected', 'undetected', 'arm_a', 'center', 'arm_b'])
        self.assertEqual(result.entries, [('arm_b', 0.4)])
        self.assertEqual(result.possibly_missed, 0)

    def test_outside_frames_do_not_hide_the_gap(self):
        result = _analyze(['arm_a', 'undetected', 'outside', 'arm_b'])
        self.assertEqual(result.possibly_missed, 1)

    def test_each_gap_is_counted_once(self):
        result = _analyze([
            'arm_a', 'undetected', 'arm_b',
            'undetected', 'arm_c',
            'undetected', 'arm_c',
        ])
        self.assertEqual(result.possibly_missed, 2)

    def test_counting_matches_the_plain_extractor(self):
        labels = ['arm_a', 'center', 'arm_b', 'undetected', 'arm_c', 'center', 'arm_a']
        stamps = [index / 10 for index in range(len(labels))]
        self.assertEqual(
            extract_center_mediated_entries(labels, stamps, arm_labels=ARMS),
            analyze_center_mediated_entries(labels, stamps, arm_labels=ARMS).entries,
        )

    def test_initial_arm_can_be_excluded(self):
        result = _analyze(['arm_a', 'center', 'arm_b'], count_initial_arm=False)
        self.assertEqual(result.entries, [('arm_b', 0.2)])


if __name__ == '__main__':
    unittest.main()
