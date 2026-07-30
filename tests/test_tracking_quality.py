import unittest

from core.tracking_quality import (
    quality_grade_label,
    score_detection_candidate,
    summarize_tracking_quality,
)


class CandidateScoringTests(unittest.TestCase):
    def test_rejects_components_outside_area_limits(self):
        self.assertEqual(
            score_detection_candidate(
                area=5,
                min_area=10,
                max_area=100,
                mean_contrast=1.0,
                foreground_dominance=1.0
            ),
            0.0
        )

    def test_proximity_improves_otherwise_equal_candidate(self):
        near = score_detection_candidate(
            area=40,
            min_area=10,
            max_area=100,
            mean_contrast=0.8,
            foreground_dominance=0.7,
            distance_from_last=5,
            max_expected_jump=100
        )
        far = score_detection_candidate(
            area=40,
            min_area=10,
            max_area=100,
            mean_contrast=0.8,
            foreground_dominance=0.7,
            distance_from_last=95,
            max_expected_jump=100
        )

        self.assertGreater(near, far)


class TrackingSummaryTests(unittest.TestCase):
    def test_missing_frames_reduce_coverage_without_creating_jumps(self):
        summary = summarize_tracking_quality(
            [(0, 0), (1, 0), None, (100, 100)],
            [0.9, 0.8, 0.0, 0.7],
            frame_diagonal=200
        )

        self.assertEqual(summary['detection_rate'], 0.75)
        self.assertEqual(summary['implausible_jump_rate'], 0.0)
        self.assertEqual(quality_grade_label(summary['quality_grade']), 'Poor')

    def test_high_quality_track_receives_excellent_grade(self):
        summary = summarize_tracking_quality(
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [0.9, 0.9, 0.8, 0.8],
            frame_diagonal=100
        )

        self.assertEqual(
            quality_grade_label(summary['quality_grade']),
            'Excellent'
        )


if __name__ == '__main__':
    unittest.main()
