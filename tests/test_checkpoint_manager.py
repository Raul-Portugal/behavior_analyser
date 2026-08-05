import tempfile
import unittest
from pathlib import Path

from checkpoint_manager import CheckpointManager


class FakeSettings:
    def __init__(self, threshold=99.0):
        self.threshold = threshold

    def to_dict(self):
        return {'threshold': self.threshold, 'nested': {'enabled': True}}


class CheckpointManagerTests(unittest.TestCase):
    def test_checkpoint_round_trip_uses_stable_settings_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / 'trial.mp4'
            video.write_bytes(b'video-data')
            manager = CheckpointManager(root)

            manager.save_checkpoint(
                video,
                frame_idx=123,
                partial_result={'positions': [[1, 2], None]},
                settings=FakeSettings()
            )
            restored = manager.load_checkpoint(video, FakeSettings())

            self.assertIsNotNone(restored)
            self.assertEqual(restored.last_frame_processed, 123)
            self.assertEqual(
                restored.partial_result,
                {'positions': [[1, 2], None]}
            )

    def test_changed_settings_invalidate_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / 'trial.mp4'
            video.write_bytes(b'video-data')
            manager = CheckpointManager(root)

            manager.save_checkpoint(
                video, 20, {'positions': []}, FakeSettings(99.0)
            )

            self.assertIsNone(
                manager.load_checkpoint(video, FakeSettings(98.0))
            )

    def test_same_stem_in_different_paths_has_distinct_checkpoint_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / 'first' / 'trial.mp4'
            second = root / 'second' / 'trial.mp4'
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b'first')
            second.write_bytes(b'second')
            manager = CheckpointManager(root)

            self.assertNotEqual(
                manager._get_checkpoint_path(first),
                manager._get_checkpoint_path(second)
            )


if __name__ == '__main__':
    unittest.main()
