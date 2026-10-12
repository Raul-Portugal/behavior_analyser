"""The optimized TST engine must reproduce the v3.2.0 algorithm exactly."""

import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np

    from core.tst_motion_engine import TSTMotionEngine
    from roi_manager import ROI
    from tests import tst_engine_runs as runs
    from tests import tst_synthetic as synthetic
    from tests.tst_reference_engine import ReferenceTSTMotionEngine
    HAS_OPENCV = True
except ModuleNotFoundError:
    HAS_OPENCV = False


def as_dicts(observations):
    return [
        {name: item.to_dict() for name, item in pair.items()}
        for pair in observations
    ]


def multi_mouse_frames(count, scale=2, mice=3):
    """Frames with ``mice`` side-by-side mice, upscaled ``scale`` times."""
    background = synthetic._background(synthetic.BASE_SIZE)
    frames = []
    for index in range(count):
        tiles = [
            synthetic.render_frame(2.0 + index / 30.0 + 0.37 * mouse, background)
            for mouse in range(mice)
        ]
        frame = np.hstack(tiles)
        frames.append(cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        ))
    rois = {}
    for mouse in range(mice):
        points = synthetic.clip_roi_points().astype(np.int64)
        points[:, 0] += mouse * synthetic.BASE_SIZE[0]
        rois[f"mouse_{mouse + 1}"] = ROI(
            (points * scale).astype(np.int32), f"mouse_{mouse + 1}"
        )
    return frames, rois


@unittest.skipUnless(HAS_OPENCV, "OpenCV runtime is not installed")
class TSTReferenceEquivalenceTests(unittest.TestCase):
    def test_full_clips_match_the_v320_algorithm_at_every_frame_rate(self):
        results = runs.runs("reference", "serial")
        for fps in runs.RATES:
            reference = as_dicts(results["reference"][fps])
            optimized = as_dicts(results["serial"][fps])
            self.assertGreater(len(reference), 100)
            self.assertEqual(optimized, reference, f"{fps} fps")

    def test_three_mice_on_a_downscaled_frame_match_the_reference(self):
        frames, rois = multi_mouse_frames(36)
        reference = runs.serial_run(
            ReferenceTSTMotionEngine, frames, 30.0, rois
        )
        optimized = runs.serial_run(TSTMotionEngine, frames, 30.0, rois)
        self.assertEqual(as_dicts(optimized), as_dicts(reference))
        self.assertEqual(
            {item.state for pair in reference for item in pair.values()}
            - {"uncertain"},
            {item.state for pair in optimized for item in pair.values()}
            - {"uncertain"},
        )


@unittest.skipUnless(HAS_OPENCV, "OpenCV runtime is not installed")
class TSTCropBeforeResizeTests(unittest.TestCase):
    ROI_POINTS = np.array(
        [[210, 90], [1200, 80], [1220, 700], [190, 710]], np.int32
    )

    @staticmethod
    def _frame(width, height, seed):
        rng = np.random.default_rng(seed)
        smooth = cv2.resize(
            rng.integers(0, 256, (height // 8, width // 8, 3)).astype(np.uint8),
            (width, height), interpolation=cv2.INTER_LINEAR,
        )
        noise = rng.integers(-20, 21, smooth.shape)
        return np.clip(smooth.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def _engines(self, width, height, analysis_width):
        scale = width / 1920.0
        roi = ROI(np.rint(self.ROI_POINTS * scale).astype(np.int32), "mouse_1")
        frame = self._frame(width, height, seed=width)
        engine = TSTMotionEngine({"mouse_1": roi}, analysis_width=analysis_width)
        reference = ReferenceTSTMotionEngine(
            {"mouse_1": roi}, analysis_width=analysis_width
        )
        engine._init_geometry(frame.shape)
        reference._init_geometry(frame.shape)
        return engine, reference, frame

    def test_integer_downscale_resizes_only_the_crop_without_any_change(self):
        for width, height, analysis_width in (
            (1920, 1080, 480), (2880, 1620, 480), (1920, 1080, 240),
            (960, 540, 480), (1920, 1080, 640),
        ):
            engine, reference, frame = self._engines(
                width, height, analysis_width
            )
            self.assertIsNotNone(
                engine._exact_crop_factors, (width, analysis_width)
            )
            for seed in range(3):
                frame = self._frame(width, height, seed)
                np.testing.assert_array_equal(
                    engine._analysis_gray(frame),
                    reference._analysis_gray(frame),
                    err_msg=f"{width}x{height} -> {analysis_width}",
                )

    def test_non_integer_downscale_keeps_the_full_frame_resize(self):
        for width, height, analysis_width in (
            (1280, 720, 480), (1920, 1080, 500), (1920, 1081, 480),
        ):
            engine, reference, frame = self._engines(
                width, height, analysis_width
            )
            self.assertIsNone(engine._exact_crop_factors, (width, height))
            np.testing.assert_array_equal(
                engine._analysis_gray(frame), reference._analysis_gray(frame)
            )

    def test_resize_does_not_touch_pixels_outside_the_crop(self):
        engine, _, frame = self._engines(1920, 1080, 480)
        with patch(
            "core.tst_motion_engine.cv2.resize", wraps=cv2.resize
        ) as resize:
            engine._analysis_gray(frame)
        source = resize.call_args.args[0]
        x0, y0, x1, y1 = engine._crop
        self.assertEqual(source.shape[:2], ((y1 - y0) * 4, (x1 - x0) * 4))
        self.assertLess(source.size, frame.size)


@unittest.skipUnless(HAS_OPENCV, "OpenCV runtime is not installed")
class TSTGrayCacheTests(unittest.TestCase):
    def setUp(self):
        self.frames = runs.clip_frames(30.0)[60:90]
        self.roi = runs.clip_roi()

    def _count_gray_calls(self, engine, pairs):
        original = engine._analysis_gray
        with patch.object(
            engine, "_analysis_gray", side_effect=original
        ) as spy:
            observations = [
                engine.calculate_observations(current, previous)
                for current, previous in pairs
            ]
        return observations, spy.call_count

    def test_each_frame_is_preprocessed_once_across_consecutive_pairs(self):
        engine = TSTMotionEngine({"mouse_1": self.roi}, fps=30.0)
        pairs = list(zip(self.frames[1:], self.frames))
        _, calls = self._count_gray_calls(engine, pairs)
        # N pairs need N + 1 distinct frames, not 2N images.
        self.assertEqual(calls, len(pairs) + 1)

    def test_cache_is_invalidated_when_the_previous_frame_changes(self):
        reference = ReferenceTSTMotionEngine({"mouse_1": self.roi}, fps=30.0)
        engine = TSTMotionEngine({"mouse_1": self.roi}, fps=30.0)
        # Non-consecutive pairs: "previous" is never the last "current".
        pairs = [
            (self.frames[index], self.frames[index - 2])
            for index in range(2, 20)
        ]
        optimized, calls = self._count_gray_calls(engine, pairs)
        self.assertEqual(calls, 2 * len(pairs))
        expected = [
            reference.calculate_observations(current, previous)
            for current, previous in pairs
        ]
        self.assertEqual(as_dicts(optimized), as_dicts(expected))

    def test_equal_but_distinct_frame_object_is_not_a_cache_hit(self):
        engine = TSTMotionEngine({"mouse_1": self.roi}, fps=30.0)
        engine.calculate_observations(self.frames[1], self.frames[0])
        changed = self.frames[1].copy()
        changed[:] = 255 - changed
        with patch.object(
            engine, "_analysis_gray", wraps=engine._analysis_gray
        ) as spy:
            engine.calculate_observations(self.frames[2], changed)
        self.assertEqual(spy.call_count, 2)

    def test_missing_frame_clears_the_cache(self):
        engine = TSTMotionEngine({"mouse_1": self.roi}, fps=30.0)
        engine.calculate_observations(self.frames[1], self.frames[0])
        self.assertIsNotNone(engine._gray_cache)
        engine.calculate_observations(None, None)
        self.assertIsNone(engine._gray_cache)


if __name__ == "__main__":
    unittest.main()
