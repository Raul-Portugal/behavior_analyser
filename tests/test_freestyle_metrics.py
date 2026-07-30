import unittest

try:
    from mazes.freestyle import Freestyle, FreestyleAnalysisResult
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class FreestyleMetricTests(unittest.TestCase):
    def _result(self, scale_factor):
        distance_scale = scale_factor if scale_factor > 0 else 1.0
        return FreestyleAnalysisResult(
            positions=[(0, 0), (1, 0), (2, 0)],
            fps=10.0,
            total_distance=2.0 * distance_scale,
            scale_factor=scale_factor,
        )

    def test_calibration_does_not_change_immobility_classification(self):
        uncalibrated = self._result(0.0)
        calibrated = self._result(0.05)

        Freestyle().calculate_metrics(uncalibrated)
        Freestyle().calculate_metrics(calibrated)

        self.assertEqual(
            uncalibrated.immobility_percentage,
            calibrated.immobility_percentage,
        )
        self.assertEqual(uncalibrated.immobility_percentage, 0.0)
        self.assertEqual(calibrated.immobility_threshold, 0.1)
        self.assertEqual(calibrated.immobility_threshold_unit, "cm/s")


if __name__ == "__main__":
    unittest.main()
