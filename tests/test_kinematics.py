import unittest

from core.kinematics import (
    calculate_immobility_time,
    calculate_instantaneous_speeds,
    calculate_movement_efficiency,
    calculate_zone_first_entries,
    calculate_zone_transitions,
)


class KinematicsTests(unittest.TestCase):
    def test_speed_is_scaled_and_tracking_gaps_are_missing(self):
        positions = [(0, 0), (3, 4), None, (6, 8), (6, 8)]

        speeds = calculate_instantaneous_speeds(
            positions,
            fps=2.0,
            scale_factor=0.1
        )

        self.assertEqual(speeds, [None, 1.0, None, None, 0.0])

    def test_missing_speeds_do_not_count_as_immobility(self):
        immobile_time, immobile_percent = calculate_immobility_time(
            [None, 1.0, None, 0.0],
            fps=2.0,
            threshold=0.5
        )

        self.assertEqual(immobile_time, 0.5)
        self.assertEqual(immobile_percent, 50.0)

    def test_efficiency_uses_the_same_units_as_total_distance(self):
        efficiency = calculate_movement_efficiency(
            [(0, 0), (3, 4)],
            total_distance=0.5,
            scale_factor=0.1
        )

        self.assertEqual(efficiency, 1.0)

    def test_efficiency_does_not_bridge_tracking_gaps(self):
        efficiency = calculate_movement_efficiency(
            [(0, 0), (3, 4), None, (100, 100), (100, 105)],
            total_distance=20.0,
            scale_factor=1.0,
        )

        self.assertEqual(efficiency, 0.5)

    def test_transitions_do_not_bridge_undetected_frames(self):
        transitions, count = calculate_zone_transitions(
            ['zone_a', 'zone_b', 'undetected', 'zone_c', 'zone_c', 'outside']
        )

        self.assertEqual(
            transitions,
            [('zone_a', 'zone_b', 1), ('zone_c', 'outside', 5)]
        )
        self.assertEqual(count, 2)

    def test_first_entries_exclude_non_spatial_states(self):
        entries = calculate_zone_first_entries(
            ['undetected', 'outside', 'zone_a', 'zone_a', 'zone_b'],
            [0.0, 0.1, 0.2, 0.3, 0.4]
        )

        self.assertEqual(entries, {'zone_a': 0.2, 'zone_b': 0.4})


if __name__ == '__main__':
    unittest.main()
