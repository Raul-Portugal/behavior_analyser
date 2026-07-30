import tempfile
import unittest
from pathlib import Path

try:
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


if __name__ == "__main__":
    unittest.main()
