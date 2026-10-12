"""The sequential/chunked reference frame must equal the v3.2.0 result."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import cv2
    import numpy as np

    from core.video import (
        BufferedVideoReader,
        ReferenceFrameGenerator,
        VideoHandler,
    )
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


def _reference_v320(path, num_samples, target_dims=None):
    """Verbatim v3.2.0 sampling and median (seek for every sample)."""
    handler = VideoHandler(path)
    indices = np.linspace(
        0, handler.total_frames - 1,
        min(num_samples, handler.total_frames), dtype=int,
    )
    frames = []
    cap = cv2.VideoCapture(str(path))
    try:
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if target_dims:
                    gray = cv2.resize(gray, target_dims)
                frames.append(gray)
    finally:
        cap.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def write_indexed_video(path, frame_count, width=96, height=64, fourcc="MJPG"):
    """Noisy frames whose brightness encodes the frame index."""
    rng = np.random.default_rng(5)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*fourcc), 25.0, (width, height)
    )
    for index in range(frame_count):
        frame = rng.integers(0, 200, (height, width, 3), dtype=np.uint8)
        frame[:, :8] = min(255, 20 + index * 2)
        writer.write(frame)
    writer.release()


@unittest.skipUnless(HAS_RUNTIME, "Video runtime is not installed")
class ReferenceFrameGenerationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.video = self.root / "clip.avi"
        write_indexed_video(self.video, 120)

    def _generate(self, **kwargs):
        with patch.object(
            ReferenceFrameGenerator, "get_cache_dir",
            return_value=self.root / "cache",
        ):
            return ReferenceFrameGenerator.generate(
                self.video, use_cache=False, **kwargs
            )

    def test_matches_v320_for_dense_and_sparse_sampling(self):
        # 120 samples -> every frame (sequential), 60 -> gap 2, 10 -> gap
        # ~13 (grab), 3 -> gap ~59 (seek), 1 and more samples than frames.
        for samples in (500, 120, 100, 60, 10, 3, 1):
            expected = _reference_v320(self.video, samples)
            actual = self._generate(num_samples=samples)
            np.testing.assert_array_equal(
                actual, expected, err_msg=f"samples={samples}"
            )
            self.assertEqual(actual.dtype, np.uint8)

    def test_matches_v320_with_resize(self):
        expected = _reference_v320(self.video, 20, target_dims=(48, 32))
        actual = self._generate(num_samples=20, target_dims=(48, 32))
        np.testing.assert_array_equal(actual, expected)

    def test_mixed_sequential_and_seek_gaps_use_both_paths(self):
        calls = {"set": 0, "grab": 0}
        real_capture = cv2.VideoCapture

        class CountingCapture:
            def __init__(self, *args):
                self._cap = real_capture(*args)

            def set(self, *args):
                calls["set"] += 1
                return self._cap.set(*args)

            def grab(self):
                calls["grab"] += 1
                return self._cap.grab()

            def __getattr__(self, name):
                return getattr(self._cap, name)

        with patch.object(cv2, "VideoCapture", CountingCapture):
            actual = self._generate(num_samples=10)
        expected = _reference_v320(self.video, 10)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(calls["set"], 1)  # only the first sample seeks
        self.assertGreater(calls["grab"], 50)

    def test_chunked_median_matches_numpy_median(self):
        rng = np.random.default_rng(11)
        for count in (1, 2, 7, 100):
            frames = [
                rng.integers(0, 256, (37, 29), dtype=np.uint8)
                for _ in range(count)
            ]
            expected = np.median(np.stack(frames), axis=0).astype(np.uint8)
            with patch.object(
                ReferenceFrameGenerator, "MEDIAN_BLOCK_BYTES", 200
            ):
                actual = ReferenceFrameGenerator._chunked_median(frames)
            np.testing.assert_array_equal(actual, expected)

    def test_cancel_is_polled_before_every_sample(self):
        polls = []

        def should_cancel():
            polls.append(1)
            return len(polls) > 4

        with self.assertRaises(InterruptedError):
            self._generate(num_samples=20, should_cancel=should_cancel)
        self.assertEqual(len(polls), 5)

    def test_cancel_before_first_sample_reads_nothing(self):
        with self.assertRaises(InterruptedError):
            self._generate(num_samples=20, should_cancel=lambda: True)


@unittest.skipUnless(HAS_RUNTIME, "Video runtime is not installed")
class ReaderDefaultsTests(unittest.TestCase):
    def test_default_buffer_is_small(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            video = Path(tmp) / "clip.avi"
            write_indexed_video(video, 5)
            with BufferedVideoReader(video) as reader:
                self.assertEqual(reader.buffer.maxsize, 16)


if __name__ == "__main__":
    unittest.main()
