import unittest

try:
    import numpy as np

    from core.models import BatchSettings, DetectionConfig
    from mazes.tst import TST
    from roi_manager import ROIManager
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class ConfigurationMigrationTests(unittest.TestCase):
    def _settings_dict(self, algorithm_parameters):
        manager = ROIManager(source_dimensions=(100, 100))
        manager.add_roi(
            "mouse_1",
            np.asarray([[10, 10], [90, 10], [90, 90], [10, 90]]),
        )
        return {
            "schema_version": 4,
            "roi_manager": manager.to_dict(),
            "detection_config": DetectionConfig().to_dict(),
            "scale_factor": 0.0,
            "start_time": 0.0,
            "end_time": 10.0,
            "paradigm_id": "Tail Suspension Test",
            "algorithm_parameters": algorithm_parameters,
        }

    def test_legacy_tst_configuration_requires_review(self):
        with self.assertRaisesRegex(ValueError, "legacy motion engine"):
            BatchSettings.from_dict(self._settings_dict({
                "energy_threshold": 12.0,
                "min_time_seconds": 1.0,
            }))

    def test_v2_configuration_loads_and_upgrades_schema(self):
        parameters = TST().get_algorithm_parameters()
        settings = BatchSettings.from_dict(
            self._settings_dict(parameters)
        )

        self.assertEqual(
            settings.algorithm_parameters["engine_id"],
            "tst_motion_decomposition_v2",
        )
        self.assertEqual(
            settings.to_dict()["schema_version"],
            BatchSettings.SCHEMA_VERSION,
        )

    def test_tst_rejects_wrong_engine_identifier(self):
        with self.assertRaisesRegex(ValueError, "not compatible"):
            TST().configure_algorithm_parameters({
                "engine_id": "tst_motion_decomposition_v1",
            })


if __name__ == "__main__":
    unittest.main()
