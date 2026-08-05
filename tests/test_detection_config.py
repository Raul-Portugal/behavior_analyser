import unittest

from core.models import DetectionConfig


class DetectionConfigTests(unittest.TestCase):
    def test_legacy_dictionary_receives_new_detector_defaults(self):
        config = DetectionConfig.from_dict({
            'threshold_percentile': 98.0,
            'window_size': 80
        })

        self.assertEqual(config.threshold_percentile, 98.0)
        self.assertEqual(config.window_size, 80)
        self.assertEqual(config.min_component_area, 20)
        self.assertEqual(config.min_detection_confidence, 0.15)

    def test_invalid_component_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            DetectionConfig(min_component_area=0)

        with self.assertRaises(ValueError):
            DetectionConfig(max_component_area_ratio=1.5)

        with self.assertRaises(ValueError):
            DetectionConfig(min_detection_confidence=float("nan"))


if __name__ == '__main__':
    unittest.main()
