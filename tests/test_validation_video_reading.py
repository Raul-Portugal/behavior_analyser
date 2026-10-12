"""Validation/timelapse writers must read sequentially yet stay frame-exact."""

import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import cv2
    import numpy as np

    from core.models import AppConfig
    from core.tst_display import blend_polygon
    from core.video import position_capture_at_frame
    from tests import worker_support as support
    from mazes.freestyle import Freestyle
    from mazes.tst import TST
    from roi_manager import ROIManager
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False

FRAME_COUNT = 80
SIZE = (160, 120)  # width, height
STEP = 3  # frame index is stored as brightness 3 * index + 10


def write_indexed_video(path):
    """A flat patch whose brightness encodes the frame index."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, SIZE
    )
    rng = np.random.default_rng(3)
    for index in range(FRAME_COUNT):
        frame = rng.integers(0, 40, (SIZE[1], SIZE[0], 3), dtype=np.uint8)
        frame[65:115, 5:75] = STEP * index + 10
        writer.write(frame)
    writer.release()


def decode_index(frame):
    return int(round((float(frame[70:110, 10:70].mean()) - 10) / STEP))


class RecordingWriter:
    """SafeVideoWriter stand-in that keeps the frames it is given."""

    frames = []

    def __init__(self, path, fps, dims, codec="mp4v"):
        self.path = Path(path)
        self.fps = fps
        self.path.write_bytes(b"")
        RecordingWriter.frames = []

    def write(self, frame):
        RecordingWriter.frames.append(frame.copy())

    def release(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class SeekOffCapture:
    """Capture whose POS_FRAMES seeks land a few frames early."""

    def __init__(self, cap, error=4):
        self._cap = cap
        self._error = error

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES and value > 0:
            value = max(0, value - self._error)
        return self._cap.set(prop, value)

    def __getattr__(self, name):
        return getattr(self._cap, name)


def _tracking_result(start_frame, count):
    fps = 30.0
    return SimpleNamespace(
        timestamps=[(start_frame + i) / fps for i in range(count)],
        start_frame=start_frame,
        positions=[(30.0 + i, 40.0) for i in range(count)],
        detection_confidences=[0.7] * count,
        visual_labels=["arena"] * count,
        roi_labels=["arena"] * count,
        detection_failure_reasons=[""] * count,
    )


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class ValidationVideoReadingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.video = self.root / "indexed.avi"
        write_indexed_video(self.video)
        (self.root / "out").mkdir()
        self.pipeline, _ = support.make_pipeline(self.root, Freestyle())
        manager = ROIManager(source_dimensions=SIZE)
        manager.add_roi(
            "arena", np.array([[0, 0], [159, 0], [159, 119], [0, 119]])
        )
        self.settings = SimpleNamespace(roi_manager=manager)
        self.handler = SimpleNamespace(fps=30.0, dimensions=SIZE)

    def _timelapse_indices(self, start_frame, count, capture_wrapper=None):
        patches = [patch("core.pipeline.SafeVideoWriter", RecordingWriter)]
        if capture_wrapper is not None:
            @contextlib.contextmanager
            def open_capture(path):
                cap = cv2.VideoCapture(str(path))
                try:
                    yield capture_wrapper(cap)
                finally:
                    cap.release()
            patches.append(patch("core.pipeline._open_capture", open_capture))
        with contextlib.ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            self.pipeline._generate_tracking_validation_video(
                _tracking_result(start_frame, count), self.video,
                self.settings, "clip", self.handler, self.root / "out",
            )
        return [decode_index(frame) for frame in RecordingWriter.frames]

    def test_timelapse_writes_every_nth_source_frame(self):
        interval = self._interval()
        indices = self._timelapse_indices(start_frame=17, count=60)
        self.assertEqual(indices, list(range(17, 77, interval)))

    def _interval(self):
        return max(1, AppConfig().visualization.timelapse_fps_divider)

    def test_timelapse_from_frame_zero(self):
        interval = self._interval()
        indices = self._timelapse_indices(start_frame=0, count=40)
        self.assertEqual(indices, list(range(0, 40, interval)))

    def test_timelapse_is_exact_when_seeks_land_early(self):
        interval = self._interval()
        indices = self._timelapse_indices(
            start_frame=23, count=50, capture_wrapper=SeekOffCapture
        )
        self.assertEqual(indices, list(range(23, 73, interval)))

    def test_timelapse_uses_grab_for_skipped_frames(self):
        calls = {"read": 0, "grab": 0}

        class Counting:
            def __init__(self, cap):
                self._cap = cap

            def read(self):
                calls["read"] += 1
                return self._cap.read()

            def grab(self):
                calls["grab"] += 1
                return self._cap.grab()

            def __getattr__(self, name):
                return getattr(self._cap, name)

        indices = self._timelapse_indices(
            start_frame=10, count=50, capture_wrapper=Counting
        )
        self.assertEqual(calls["read"], len(indices))
        self.assertEqual(calls["grab"], 50 - len(indices))

    def test_timelapse_stops_cleanly_at_end_of_video(self):
        indices = self._timelapse_indices(start_frame=70, count=40)
        self.assertEqual(indices, list(range(70, FRAME_COUNT, self._interval())))

    def _tst_result(self, start_frame, count, contour=None):
        fps = 30.0
        evidence = [
            {"confidence": 0.8, "qc_flag": "",
             **({"mask_contour": contour} if contour is not None else {})}
            for _ in range(count)
        ]
        return SimpleNamespace(
            timestamps=[(start_frame + i) / fps for i in range(count)],
            start_frame=start_frame,
            per_roi_times={"mouse_1": (0.0, 100.0)},
            immobile_states={"mouse_1": [i % 2 == 0 for i in range(count)]},
            behavior_states={"mouse_1": ["immobile_static"] * count},
            motion_evidence={"mouse_1": evidence},
        )

    def _tst_indices(self, start_frame, count, capture_wrapper=None,
                     contour=None):
        manager = ROIManager(source_dimensions=SIZE)
        manager.add_roi(
            "mouse_1", np.array([[100, 20], [150, 20], [150, 100], [100, 100]])
        )
        settings = SimpleNamespace(roi_manager=manager)
        tst_pipeline, _ = support.make_pipeline(self.root, TST())
        patches = [
            patch("core.pipeline.SafeVideoWriter", RecordingWriter),
            patch("core.pipeline.VideoHandler",
                  lambda _path: SimpleNamespace(fps=30.0, dimensions=SIZE)),
        ]
        if capture_wrapper is not None:
            @contextlib.contextmanager
            def open_capture(path):
                cap = cv2.VideoCapture(str(path))
                try:
                    yield capture_wrapper(cap)
                finally:
                    cap.release()
            patches.append(patch("core.pipeline._open_capture", open_capture))
        with contextlib.ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            tst_pipeline._generate_tst_validation_video(
                self._tst_result(start_frame, count, contour), self.video,
                settings, "clip", self.root / "out",
            )
        return [decode_index(frame) for frame in RecordingWriter.frames]

    def test_tst_video_writes_every_third_source_frame(self):
        indices = self._tst_indices(start_frame=11, count=45)
        self.assertEqual(indices, list(range(11, 56, 3)))

    def test_tst_video_is_exact_when_seeks_land_early(self):
        indices = self._tst_indices(
            start_frame=29, count=45, capture_wrapper=SeekOffCapture
        )
        self.assertEqual(indices, list(range(29, 74, 3)))

    def test_tst_video_tolerates_missing_and_empty_contours(self):
        self.assertEqual(
            self._tst_indices(5, 12, contour=None), list(range(5, 17, 3))
        )
        self.assertEqual(
            self._tst_indices(5, 12, contour=[]), list(range(5, 17, 3))
        )

    def test_tst_video_draws_contour_overlay(self):
        contour = [[105, 30], [140, 30], [140, 60], [105, 60]]
        self._tst_indices(start_frame=0, count=3, contour=contour)
        with_contour = RecordingWriter.frames[0].copy()
        self._tst_indices(start_frame=0, count=3, contour=None)
        without = RecordingWriter.frames[0]
        self.assertFalse(np.array_equal(with_contour, without))
        # Nothing changes away from the ROI/contour region.
        np.testing.assert_array_equal(
            with_contour[:, :90], without[:, :90]
        )


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class PositionCaptureTests(unittest.TestCase):
    def test_exact_seek_does_not_read_forward(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            video = Path(tmp) / "indexed.avi"
            write_indexed_video(video)
            cap = cv2.VideoCapture(str(video))
            try:
                position_capture_at_frame(cap, 40)
                ret, frame = cap.read()
                self.assertTrue(ret)
                self.assertEqual(decode_index(frame), 40)
            finally:
                cap.release()

    def test_inaccurate_seek_falls_back_to_forward_read(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            video = Path(tmp) / "indexed.avi"
            write_indexed_video(video)
            cap = SeekOffCapture(cv2.VideoCapture(str(video)))
            try:
                position_capture_at_frame(cap, 40)
                ret, frame = cap.read()
                self.assertTrue(ret)
                self.assertEqual(decode_index(frame), 40)
            finally:
                cap.release()

    def test_cancel_is_honoured_during_fallback(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            video = Path(tmp) / "indexed.avi"
            write_indexed_video(video)
            cap = SeekOffCapture(cv2.VideoCapture(str(video)))
            try:
                with self.assertRaises(InterruptedError):
                    position_capture_at_frame(
                        cap, 40, should_cancel=lambda: True
                    )
            finally:
                cap.release()


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class BlendPolygonTests(unittest.TestCase):
    @staticmethod
    def _full_frame_blend(frame, points, color, alpha, beta):
        """The v3.2.0 implementation: copy and blend the whole frame."""
        points = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, alpha, frame, beta, 0, frame)

    def test_matches_full_frame_blend_everywhere(self):
        rng = np.random.default_rng(21)
        polygons = [
            [[10, 10], [60, 15], [40, 70]],
            [[-20, 30], [50, -10], [90, 50], [20, 90]],   # crosses edges
            [[150, 100], [200, 110], [190, 160]],          # partly outside
            [[30, 30], [31, 30], [31, 31]],                # tiny
            [[400, 400], [450, 400], [450, 450]],          # fully outside
            [[0, 0], [159, 0], [159, 119], [0, 119]],      # whole frame
        ]
        for points in polygons:
            for alpha, beta in ((0.35, 0.65), (0.5, 0.5)):
                frame = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
                expected = frame.copy()
                self._full_frame_blend(
                    expected, points, (0, 165, 255), alpha, beta
                )
                actual = frame.copy()
                blend_polygon(actual, points, (0, 165, 255), alpha, beta)
                np.testing.assert_array_equal(actual, expected)

    def test_pixels_outside_bounding_box_are_untouched(self):
        rng = np.random.default_rng(2)
        frame = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
        original = frame.copy()
        blend_polygon(frame, [[40, 30], [90, 35], [70, 80]], (0, 0, 255))
        mask = np.ones(frame.shape[:2], dtype=bool)
        mask[30:81, 40:91] = False
        np.testing.assert_array_equal(frame[mask], original[mask])
        self.assertFalse(np.array_equal(frame[~mask], original[~mask]))

    def test_empty_polygon_is_a_no_op(self):
        frame = np.full((10, 10, 3), 77, dtype=np.uint8)
        blend_polygon(frame, np.empty((0, 2), dtype=np.int32), (1, 2, 3))
        self.assertTrue((frame == 77).all())


if __name__ == "__main__":
    unittest.main()
