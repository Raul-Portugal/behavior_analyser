import unittest

try:
    import cv2
    import numpy as np

    from core.detection import DetectionEngine
    from core.models import DetectionConfig

    HAS_OPENCV = True
except ModuleNotFoundError:
    HAS_OPENCV = False


@unittest.skipUnless(HAS_OPENCV, "OpenCV runtime is not installed")
class DetectionEngineTests(unittest.TestCase):
    def test_selects_plausible_high_contrast_component(self):
        reference = np.zeros((100, 100), dtype=np.uint8)
        frame = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(frame, (40, 45), (59, 64), 255, -1)
        config = DetectionConfig(
            threshold_percentile=95.0,
            use_weighting=False,
            min_component_area=20,
            morphology_kernel_size=3,
            min_detection_confidence=0.1
        )

        result = DetectionEngine(reference, config).detect(frame)

        self.assertTrue(result.detected)
        self.assertAlmostEqual(result.position[0], 50, delta=1)
        self.assertAlmostEqual(result.position[1], 55, delta=1)
        self.assertGreater(result.confidence, 0.5)

    def test_reports_no_foreground_for_unchanged_frame(self):
        reference = np.zeros((50, 50), dtype=np.uint8)
        config = DetectionConfig(use_weighting=False)

        result = DetectionEngine(reference, config).detect(reference.copy())

        self.assertFalse(result.detected)
        self.assertEqual(result.failure_reason, 'no_foreground')

    def test_far_candidate_is_rejected_until_reacquisition(self):
        reference = np.zeros((100, 100), dtype=np.uint8)
        first = reference.copy()
        far = reference.copy()
        cv2.rectangle(first, (10, 40), (29, 59), 255, -1)
        cv2.rectangle(far, (70, 40), (89, 59), 255, -1)
        config = DetectionConfig(
            threshold_percentile=95.0,
            use_weighting=False,
            min_component_area=20,
            morphology_kernel_size=3,
            min_detection_confidence=0.1,
            max_expected_jump_ratio=0.1,
        )
        engine = DetectionEngine(reference, config)

        self.assertTrue(engine.detect(first).detected)
        rejected = engine.detect(far)

        self.assertFalse(rejected.detected)
        self.assertEqual(rejected.failure_reason, "jump_exceeded")

    def test_detector_state_round_trip_preserves_failure_count(self):
        reference = np.zeros((20, 20), dtype=np.uint8)
        engine = DetectionEngine(reference, DetectionConfig())
        engine.last_position = (5, 7)
        engine.consecutive_failures = 12

        restored = DetectionEngine(reference, DetectionConfig())
        restored.restore_state(engine.export_state())

        self.assertEqual(restored.last_position, (5, 7))
        self.assertEqual(restored.consecutive_failures, 12)


if __name__ == '__main__':
    unittest.main()
