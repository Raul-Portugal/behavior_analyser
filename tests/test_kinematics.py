import math
import random
import unittest

from core.kinematics import (
    calculate_immobility_bouts,
    calculate_immobility_time,
    calculate_instantaneous_speeds,
    calculate_movement_efficiency,
    calculate_smoothed_path_length,
    calculate_smoothed_speeds,
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


PAUSES = ((3.0, 5.0), (8.0, 9.5))


def scripted_track(fps, duration=12.0, speed_px_s=200.0):
    """Sample one physical, curved motion with scripted pauses at any fps."""
    positions = []
    for frame in range(int(duration * fps)):
        t = frame / fps
        moving_time = t
        for start, end in PAUSES:
            moving_time -= min(max(t - start, 0.0), end - start)
        travelled = speed_px_s * moving_time
        positions.append(
            (travelled * 0.8, 40.0 * math.sin(travelled / 80.0))
        )
    return positions


class SmoothedSpeedTests(unittest.TestCase):
    def test_window_uses_a_centered_displacement(self):
        positions = [(float(i), 0.0) for i in range(11)]

        speeds = calculate_smoothed_speeds(
            positions, fps=10.0, scale_factor=0.5, window_seconds=0.4
        )

        self.assertEqual(speeds[:2], [None, None])
        self.assertEqual(speeds[-2:], [None, None])
        for speed in speeds[2:-2]:
            self.assertAlmostEqual(speed, 10.0 * 0.5)

    def test_half_width_is_at_least_one_frame(self):
        positions = [(0.0, 0.0), (3.0, 4.0), (6.0, 8.0), (6.0, 8.0)]

        speeds = calculate_smoothed_speeds(
            positions, fps=1.0, window_seconds=0.0
        )

        self.assertEqual(len(speeds), 4)
        self.assertIsNone(speeds[0])
        self.assertIsNone(speeds[3])
        self.assertAlmostEqual(speeds[1], 5.0)
        self.assertAlmostEqual(speeds[2], 2.5)

    def test_windows_never_bridge_tracking_gaps(self):
        positions = [(float(i), 0.0) for i in range(5)]
        positions += [None]
        positions += [(100.0 + i, 0.0) for i in range(5)]

        speeds = calculate_smoothed_speeds(
            positions, fps=1.0, window_seconds=2.0
        )

        self.assertEqual(speeds[5], None)
        self.assertEqual(speeds[4], None)
        self.assertEqual(speeds[6], None)
        for index in (1, 2, 3, 7, 8, 9):
            self.assertAlmostEqual(speeds[index], 1.0)
        self.assertTrue(all(
            speed is None or speed < 2.0 for speed in speeds
        ))

    def test_accepts_float_positions_and_validates_arguments(self):
        speeds = calculate_smoothed_speeds(
            [(0.5, 0.25), (0.5, 0.25), (0.5, 0.25)], fps=10.0
        )
        self.assertEqual(speeds, [None, 0.0, None])
        with self.assertRaises(ValueError):
            calculate_smoothed_speeds([(0, 0)], fps=0.0)
        with self.assertRaises(ValueError):
            calculate_smoothed_speeds([(0, 0)], fps=30.0, scale_factor=0.0)


class ImmobilityBoutTests(unittest.TestCase):
    def test_short_dips_are_not_bouts(self):
        speeds = [5.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 5.0]

        time_s, pct, bouts, states = calculate_immobility_bouts(
            speeds, fps=4.0, threshold=1.0, min_bout_seconds=1.0
        )

        self.assertEqual(bouts, 1)
        self.assertEqual(time_s, 1.0)
        self.assertAlmostEqual(pct, 4 / 9 * 100.0)
        self.assertEqual(
            states,
            [False, False, False, False, True, True, True, True, False]
        )

    def test_missing_speed_breaks_a_bout_and_is_excluded(self):
        speeds = [0.0, 0.0, None, 0.0, 0.0, 5.0]

        time_s, pct, bouts, states = calculate_immobility_bouts(
            speeds, fps=2.0, threshold=1.0, min_bout_seconds=1.0
        )

        self.assertEqual((time_s, bouts), (2.0, 2))
        self.assertEqual(pct, 80.0)
        self.assertIsNone(states[2])

    def test_no_valid_speeds(self):
        self.assertEqual(
            calculate_immobility_bouts([None, None], 10.0, 1.0, 1.0),
            (0.0, 0.0, 0, [None, None])
        )


class SmoothedKinematicsBehaviorTests(unittest.TestCase):
    SCALE = 0.05
    THRESHOLD = 2.0

    def _immobility(self, positions, fps):
        speeds = calculate_smoothed_speeds(positions, fps, self.SCALE, 0.2)
        return calculate_immobility_bouts(speeds, fps, self.THRESHOLD, 1.0)

    def test_jittery_stationary_track_is_almost_fully_immobile(self):
        rng = random.Random(1234)
        positions = [
            (200 + rng.randint(-1, 1), 150 + rng.randint(-1, 1))
            for _ in range(600)
        ]

        raw = calculate_instantaneous_speeds(positions, 30.0, self.SCALE)
        _, pct, bouts, _ = self._immobility(positions, 30.0)

        self.assertGreater(
            sum(s is not None and s > self.THRESHOLD for s in raw), 100
        )
        self.assertGreaterEqual(pct, 99.0)
        self.assertEqual(bouts, 1)

    def test_scripted_pauses_are_recovered_within_half_a_second(self):
        for fps in (30.0, 60.0):
            with self.subTest(fps=fps):
                time_s, _, bouts, _ = self._immobility(
                    scripted_track(fps), fps
                )
                expected = sum(end - start for start, end in PAUSES)
                self.assertAlmostEqual(time_s, expected, delta=0.5)
                self.assertEqual(bouts, 2)

    def test_pauses_shorter_than_the_minimum_are_ignored(self):
        positions = [(float(i) * 4, 0.0) for i in range(60)]
        positions[20:30] = [positions[20]] * 10  # 0.33 s pause at 30 fps

        time_s, _, bouts, _ = self._immobility(positions, 30.0)

        self.assertEqual((time_s, bouts), (0.0, 0))

    def test_metrics_are_invariant_to_frame_rate(self):
        results = {}
        for fps in (30.0, 60.0):
            positions = scripted_track(fps)
            _, pct, _, _ = self._immobility(positions, fps)
            distance = calculate_smoothed_path_length(
                positions, fps, self.SCALE, 0.2
            )
            results[fps] = (pct, distance)

        self.assertAlmostEqual(results[30.0][0], results[60.0][0], delta=2.0)
        self.assertAlmostEqual(
            results[30.0][1] / results[60.0][1], 1.0, delta=0.03
        )

    def test_smoothed_path_length_removes_jitter_distance(self):
        rng = random.Random(7)
        positions = [
            (100 + rng.randint(-1, 1), 100 + rng.randint(-1, 1))
            for _ in range(300)
        ]
        raw = sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(positions, positions[1:])
        )

        smoothed = calculate_smoothed_path_length(positions, 30.0)

        self.assertLess(smoothed, raw * 0.25)

    def test_smoothed_path_length_follows_real_motion_and_skips_gaps(self):
        line = [(float(i), 0.0) for i in range(30)]
        positions = line + [None] + [(500.0 + i, 0.0) for i in range(30)]

        length = calculate_smoothed_path_length(
            positions, fps=30.0, scale_factor=2.0
        )

        self.assertAlmostEqual(length, 2.0 * 2 * 29, places=6)
        self.assertEqual(calculate_smoothed_path_length([], 30.0), 0.0)
        self.assertEqual(calculate_smoothed_path_length([None], 30.0), 0.0)


if __name__ == '__main__':
    unittest.main()
