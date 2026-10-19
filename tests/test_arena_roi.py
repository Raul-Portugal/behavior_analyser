import json
import unittest

import numpy as np

from core.models import BatchSettings, DetectionConfig
from roi_manager import ARENA_CATEGORY, ROIManager

SQUARE = np.array([[10, 10], [210, 10], [210, 210], [10, 210]])
ZONE = np.array([[50, 50], [150, 50], [150, 150], [50, 150]])


def _manager(with_arena: bool) -> ROIManager:
    manager = ROIManager(source_dimensions=(320, 240))
    manager.add_roi("zone_a", ZONE)
    manager.set_reference_length(120.0, "ref")
    if with_arena:
        manager.set_arena(SQUARE)
    return manager


class ArenaStorageTests(unittest.TestCase):
    def test_arena_is_not_a_zone(self):
        manager = _manager(True)
        self.assertNotIn(ARENA_CATEGORY, manager.rois)
        # (100, 100) is inside both the arena and zone_a: only zone_a counts.
        self.assertEqual(manager.get_overlapping_zones((100, 100)), ["zone_a"])
        self.assertEqual(manager.get_category((100, 100)), "zone_a")
        # (20, 20) is inside the arena only: arena membership never counts.
        self.assertEqual(manager.get_overlapping_zones((20, 20)), ["outside"])
        self.assertEqual(manager.get_category((20, 20)), "outside")

    def test_arena_only_manager_has_no_zones(self):
        manager = ROIManager(source_dimensions=(320, 240))
        manager.set_arena(SQUARE)
        self.assertTrue(manager.has_arena)
        self.assertEqual(manager.get_overlapping_zones((100, 100)), ["outside"])

    def test_invalid_arena_is_rejected(self):
        manager = ROIManager()
        with self.assertRaises(ValueError):
            manager.set_arena(np.array([[0, 0], [10, 10]]))
        self.assertFalse(manager.has_arena)

    def test_clear_arena_drops_center_overlay(self):
        manager = _manager(True)
        manager.set_center_overlay(ZONE)
        manager.clear_arena()
        self.assertIsNone(manager.arena)
        self.assertIsNone(manager.center_overlay)

    def test_center_overlay_validation(self):
        manager = _manager(True)
        with self.assertRaises(ValueError):
            manager.set_center_overlay(np.array([[0, 0], [1, 1]]))
        manager.set_center_overlay(None)
        self.assertIsNone(manager.center_overlay)


class ArenaSerializationTests(unittest.TestCase):
    def test_no_arena_serialization_is_unchanged(self):
        data = _manager(False).to_dict()
        self.assertEqual(data["schema_version"], 2)
        self.assertNotIn("arena", data)
        self.assertEqual(
            set(data),
            {"schema_version", "rois", "_reference_length_pixels",
             "_reference_name", "source_dimensions"},
        )

    def test_no_arena_batch_settings_keep_their_schema(self):
        settings = BatchSettings(_manager(False), DetectionConfig(), 0.0)
        data = settings.to_dict()
        self.assertEqual(data["schema_version"], BatchSettings.SCHEMA_VERSION)
        self.assertEqual(data["roi_manager"]["schema_version"], 2)

    def test_arena_round_trip(self):
        manager = _manager(True)
        data = manager.to_dict()
        self.assertEqual(data["schema_version"], 3)
        restored = ROIManager.from_dict(json.loads(json.dumps(data)))
        np.testing.assert_array_equal(restored.arena.points, SQUARE)
        self.assertEqual(restored.arena.category, ARENA_CATEGORY)
        self.assertEqual(list(restored.rois), ["zone_a"])
        self.assertEqual(restored.to_dict(), data)

    def test_batch_settings_round_trip_with_arena(self):
        settings = BatchSettings(_manager(True), DetectionConfig(), 0.0)
        data = settings.to_dict()
        self.assertEqual(data["schema_version"], BatchSettings.ARENA_SCHEMA_VERSION)
        restored = BatchSettings.from_dict(json.loads(json.dumps(data)))
        np.testing.assert_array_equal(restored.roi_manager.arena.points, SQUARE)
        self.assertEqual(restored.to_dict(), data)

    def test_old_schemas_still_load(self):
        for version in (1, 2):
            data = _manager(False).to_dict()
            data["schema_version"] = version
            restored = ROIManager.from_dict(data)
            self.assertIsNone(restored.arena)
            self.assertEqual(list(restored.rois), ["zone_a"])
        legacy = _manager(False).to_dict()
        del legacy["schema_version"]
        self.assertIsNone(ROIManager.from_dict(legacy).arena)

    def test_newer_roi_schema_is_refused(self):
        data = _manager(True).to_dict()
        data["schema_version"] = 4
        with self.assertRaisesRegex(ValueError, "newer"):
            ROIManager.from_dict(data)

    def test_arena_changes_checkpoint_hash_only_when_set(self):
        from checkpoint_manager import CheckpointManager
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            manager = CheckpointManager(Path(folder))
            plain = BatchSettings(_manager(False), DetectionConfig(), 0.0)
            with_arena = BatchSettings(_manager(True), DetectionConfig(), 0.0)
            self.assertNotIn("arena", plain.to_dict()["roi_manager"])
            self.assertNotEqual(
                manager._get_settings_hash(plain),
                manager._get_settings_hash(with_arena),
            )

    def test_center_overlay_is_not_serialized(self):
        manager = _manager(True)
        before = manager.to_dict()
        manager.set_center_overlay(ZONE)
        self.assertEqual(manager.to_dict(), before)


class ArenaDrawingTests(unittest.TestCase):
    def test_arena_is_drawn_distinctly(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        plain = _manager(False).draw_on_frame(frame)
        with_arena = _manager(True).draw_on_frame(frame)
        # The arena's top edge, away from zone_a, is only drawn with an arena.
        self.assertEqual(int(plain[10, 100].sum()), 0)
        self.assertGreater(int(with_arena[10, 100].sum()), 0)
        # Zone outlines are drawn identically either way.
        np.testing.assert_array_equal(plain[50, 100], with_arena[50, 100])

    def test_draw_without_arena_is_unchanged(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        manager = _manager(False)
        out = manager.draw_on_frame(frame)
        expected = frame.copy()
        import cv2
        cv2.polylines(expected, [manager.rois["zone_a"][0].points],
                      isClosed=True, color=(200, 200, 200), thickness=2)
        np.testing.assert_array_equal(out, expected)

    def test_center_overlay_is_drawn(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        manager = _manager(True)
        base = manager.draw_on_frame(frame)
        manager.set_center_overlay(np.array([[90, 90], [130, 90], [130, 130]]))
        overlay = manager.draw_on_frame(frame)
        self.assertFalse(np.array_equal(base, overlay))

    def test_stats_report_arena_only_when_set(self):
        self.assertNotIn("arena_area", _manager(False).get_stats())
        self.assertAlmostEqual(_manager(True).get_stats()["arena_area"], 40000.0)

    def test_tiny_arena_is_reported_by_validation(self):
        manager = _manager(False)
        manager.arena = None
        manager.set_arena(np.array([[0, 0], [5, 0], [5, 5], [0, 5]]))
        _, issues = manager.validate_rois()
        self.assertTrue(any("Arena" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
