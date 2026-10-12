import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import cv2
    import numpy as np

    from core.video import ReferenceFrameGenerator
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


@unittest.skipUnless(HAS_RUNTIME, "Video runtime is not installed")
class ReferenceCacheTests(unittest.TestCase):
    def test_cache_is_not_written_beside_source_video(self):
        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "example.mp4"
            cache_path = ReferenceFrameGenerator.get_cache_path(video_path)

            self.assertNotEqual(cache_path.parent, video_path.parent)
            self.assertIn(
                "BehavioralMazeAnalyzer",
                str(cache_path),
            )

    def _write_video(self, path, width=64, height=48):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (width, height)
        )
        for value in (30, 60, 90):
            writer.write(np.full((height, width, 3), value, dtype=np.uint8))
        writer.release()

    def test_none_and_native_dims_share_one_cache_entry(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            video_path = root / "clip.avi"
            self._write_video(video_path)
            with patch.object(
                ReferenceFrameGenerator, "get_cache_dir",
                return_value=root / "cache",
            ):
                first = ReferenceFrameGenerator.generate(video_path, 3)
                cache_path = ReferenceFrameGenerator.get_cache_path(
                    video_path
                )
                # Swap in a sentinel so a cache hit is distinguishable from
                # a regeneration.
                sentinel = np.full_like(first, 123)
                with np.load(cache_path, allow_pickle=False) as data:
                    arrays = {name: data[name] for name in data.files}
                arrays["frame"] = sentinel
                np.savez_compressed(cache_path, **arrays)

                from_none = ReferenceFrameGenerator.generate(video_path, 3)
                from_native = ReferenceFrameGenerator.generate(
                    video_path, 3, target_dims=(64, 48)
                )
                from_other = ReferenceFrameGenerator.generate(
                    video_path, 3, target_dims=(32, 24)
                )

        np.testing.assert_array_equal(from_none, sentinel)
        np.testing.assert_array_equal(from_native, sentinel)
        self.assertEqual(from_other.shape, (24, 32))
        self.assertEqual(first.shape, (48, 64))


if __name__ == "__main__":
    unittest.main()
