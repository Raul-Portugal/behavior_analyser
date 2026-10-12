"""TSTFramePipeline must return exactly what serial analysis returns."""

import json
import threading
import time
import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np

    from core.tst_motion_engine import TSTFramePipeline, TSTMotionEngine
    from roi_manager import ROI
    from tests import tst_engine_runs as runs
    from tests.test_tst_engine_equivalence import as_dicts, multi_mouse_frames
    from tests.tst_reference_engine import ReferenceTSTMotionEngine
    HAS_OPENCV = True
except ModuleNotFoundError:
    HAS_OPENCV = False


def two_blob_frames(count=30):
    """Two near-equal blobs; the smaller one is the one the mouse was on.

    The largest component (what an engine without history would pick) is
    the right-hand blob, but proximity to the previous centroid favours the
    left one, so history-free speculation is wrong on purpose.
    """
    rng = np.random.default_rng(5)
    gray = np.clip(215 + rng.integers(-4, 5, (130, 700, 1)), 0, 255)
    background = np.repeat(gray.astype(np.uint8), 3, axis=2)
    frames = []
    for index in range(count):
        image = background.copy()
        cv2.circle(image, (40 + index % 3, 65), 17, (35, 35, 35), -1)
        if index >= 5:
            cv2.circle(image, (650, 65), 18, (35, 35, 35), -1)
        frames.append(image)
    roi = ROI(np.array([[10, 10], [690, 10], [690, 120], [10, 120]], np.int32),
              "mouse_1")
    return frames, {"mouse_1": roi}


@unittest.skipUnless(HAS_OPENCV, "OpenCV runtime is not installed")
class TSTPipelineEquivalenceTests(unittest.TestCase):
    def test_pipelined_clips_are_identical_to_serial_at_every_frame_rate(self):
        results = runs.runs("reference", "serial", "pipelined")
        for fps in runs.RATES:
            expected = as_dicts(results["reference"][fps])
            self.assertEqual(
                as_dicts(results["pipelined"][fps]), expected, f"{fps} fps"
            )
            self.assertEqual(
                as_dicts(results["serial"][fps]), expected, f"{fps} fps"
            )

    def test_worker_count_and_queue_depth_do_not_change_results(self):
        frames, rois = multi_mouse_frames(30)
        expected = as_dicts(runs.serial_run(
            ReferenceTSTMotionEngine, frames, 30.0, rois
        ))
        for options in (
            {"max_workers": 1}, {"max_workers": 2, "queue_depth": 1},
            {"max_workers": 3, "queue_depth": 9},
        ):
            observed = runs.pipelined_run(frames, 30.0, rois, **options)
            self.assertEqual(as_dicts(observed), expected, options)

    def test_history_dependent_blob_choice_is_corrected_in_order(self):
        frames, rois = two_blob_frames()
        serial_engine = TSTMotionEngine(rois, fps=30.0)
        expected = [
            serial_engine.calculate_observations(frames[i], frames[i - 1])
            for i in range(1, len(frames))
        ]
        reference = runs.serial_run(
            ReferenceTSTMotionEngine, frames, 30.0, rois
        )
        self.assertEqual(as_dicts(expected), as_dicts(reference))

        engine = TSTMotionEngine(rois, fps=30.0)
        fits = []
        original = engine._fit_motion

        def counted(*args, **kwargs):
            fits.append(threading.get_ident())
            return original(*args, **kwargs)

        with patch.object(engine, "_fit_motion", side_effect=counted):
            with TSTFramePipeline(engine, max_workers=3) as pipeline:
                observed = []
                for index in range(1, len(frames)):
                    while not pipeline.has_capacity():
                        observed.append(pipeline.pop()[1])
                    pipeline.submit(index, frames[index], frames[index - 1])
                while pipeline.pending:
                    observed.append(pipeline.pop()[1])
        self.assertEqual(as_dicts(observed), as_dicts(reference))
        # The speculation was wrong for the frames with two blobs, so those
        # fits were redone on the consuming thread.
        self.assertGreater(len(fits), len(frames) - 1)
        self.assertIn(threading.get_ident(), fits)

    def test_missing_frames_reset_history_like_serial_analysis(self):
        frames = runs.clip_frames(30.0)[70:110]
        pairs = [(frames[i], frames[i - 1]) for i in range(1, len(frames))]
        pairs[10] = (None, frames[9])
        pairs[11] = (frames[12], None)
        roi = {"mouse_1": runs.clip_roi()}

        serial = TSTMotionEngine(roi, fps=30.0)
        expected = [serial.calculate_observations(c, p) for c, p in pairs]

        engine = TSTMotionEngine(roi, fps=30.0)
        observed = []
        with TSTFramePipeline(engine, max_workers=3) as pipeline:
            for index, (current, previous) in enumerate(pairs):
                while not pipeline.has_capacity():
                    observed.append(pipeline.pop()[1])
                pipeline.submit(index, current, previous)
            while pipeline.pending:
                observed.append(pipeline.pop()[1])
        self.assertEqual(as_dicts(observed), as_dicts(expected))
        self.assertEqual(observed[10]["mouse_1"].qc_flag, "missing_frame")
        self.assertEqual(engine.export_state(), serial.export_state())

    def test_results_come_back_in_submission_order_with_their_tags(self):
        frames = runs.clip_frames(30.0)[:24]
        engine = TSTMotionEngine({"mouse_1": runs.clip_roi()}, fps=30.0)
        tags = []
        with TSTFramePipeline(engine, max_workers=4) as pipeline:
            for index in range(1, len(frames)):
                while not pipeline.has_capacity():
                    tags.append(pipeline.pop()[0])
                pipeline.submit(
                    ("frame", index), frames[index], frames[index - 1]
                )
                self.assertLessEqual(pipeline.pending, pipeline.queue_depth)
            self.assertEqual(pipeline.next_tag, ("frame", len(tags) + 1))
            while pipeline.pending:
                tags.append(pipeline.pop()[0])
        self.assertEqual(
            tags, [("frame", index) for index in range(1, len(frames))]
        )


@unittest.skipUnless(HAS_OPENCV, "OpenCV runtime is not installed")
class TSTPipelineCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.frames = runs.clip_frames(30.0)[40:150]
        self.roi = {"mouse_1": runs.clip_roi()}

    def _uninterrupted(self):
        engine = TSTMotionEngine(self.roi, fps=30.0)
        observations, states = [], []
        for index in range(1, len(self.frames)):
            observations.append(engine.calculate_observations(
                self.frames[index], self.frames[index - 1]
            ))
            states.append(json.dumps(engine.export_state(), sort_keys=True))
        return observations, states

    def test_state_reflects_only_fully_consumed_frames(self):
        _, states = self._uninterrupted()
        engine = TSTMotionEngine(self.roi, fps=30.0)
        with TSTFramePipeline(engine, max_workers=4) as pipeline:
            submitted = 0
            for consumed in range(1, 30):
                while pipeline.has_capacity():
                    submitted += 1
                    pipeline.submit(
                        submitted, self.frames[submitted],
                        self.frames[submitted - 1],
                    )
                pipeline.pop()
                # Frames beyond `consumed` are in flight but must not show.
                self.assertGreater(pipeline.pending, 0)
                self.assertEqual(
                    json.dumps(engine.export_state(), sort_keys=True),
                    states[consumed - 1],
                    f"after {consumed} consumed frames",
                )

    def test_resume_from_a_pipelined_checkpoint_matches_uninterrupted(self):
        expected, _ = self._uninterrupted()
        split = 37
        first = TSTMotionEngine(self.roi, fps=30.0)
        with TSTFramePipeline(first, max_workers=4) as pipeline:
            submitted = 0
            for _ in range(split):
                while pipeline.has_capacity():
                    submitted += 1
                    pipeline.submit(
                        submitted, self.frames[submitted],
                        self.frames[submitted - 1],
                    )
                pipeline.pop()
            # The checkpoint is taken while later frames are in flight, and
            # survives the JSON round trip the checkpoint file applies.
            state = json.loads(json.dumps(first.export_state()))

        resumed = TSTMotionEngine(self.roi, fps=30.0)
        resumed.restore_state(state)
        observed = []
        with TSTFramePipeline(resumed, max_workers=4) as pipeline:
            for index in range(split + 1, len(self.frames)):
                while not pipeline.has_capacity():
                    observed.append(pipeline.pop()[1])
                pipeline.submit(
                    index, self.frames[index], self.frames[index - 1]
                )
            while pipeline.pending:
                observed.append(pipeline.pop()[1])
        self.assertEqual(
            as_dicts(observed), as_dicts(expected[split:])
        )


@unittest.skipUnless(HAS_OPENCV, "OpenCV runtime is not installed")
class TSTPipelineLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.frames = runs.clip_frames(30.0)[:40]
        self.engine = TSTMotionEngine({"mouse_1": runs.clip_roi()}, fps=30.0)

    def test_cancelling_mid_stream_returns_promptly(self):
        original = self.engine._prepare_pair

        def slow(*args, **kwargs):
            time.sleep(0.4)
            return original(*args, **kwargs)

        pipeline = TSTFramePipeline(
            self.engine, max_workers=2, queue_depth=12
        )
        with patch.object(self.engine, "_prepare_pair", side_effect=slow):
            for index in range(1, 13):
                pipeline.submit(
                    index, self.frames[index], self.frames[index - 1]
                )
            pipeline.pop()
            started = time.monotonic()
            pipeline.close()
            elapsed = time.monotonic() - started
        # Uncancelled, the 11 remaining 0.4 s tasks on 2 workers take ~2 s;
        # cancelled, only the two running tasks finish.
        self.assertLess(elapsed, 1.2)
        self.assertEqual(pipeline.pending, 0)
        with self.assertRaises(RuntimeError):
            pipeline.pop()
        with self.assertRaises(RuntimeError):
            pipeline.submit(99, self.frames[2], self.frames[1])
        pipeline.close()  # idempotent

    def test_full_pipeline_rejects_submit_and_empty_pop_raises(self):
        with TSTFramePipeline(
            self.engine, max_workers=1, queue_depth=2
        ) as pipeline:
            with self.assertRaises(IndexError):
                pipeline.pop()
            pipeline.submit(1, self.frames[1], self.frames[0])
            pipeline.submit(2, self.frames[2], self.frames[1])
            self.assertFalse(pipeline.has_capacity())
            with self.assertRaises(RuntimeError):
                pipeline.submit(3, self.frames[3], self.frames[2])
            self.assertEqual(pipeline.pop()[0], 1)
            self.assertTrue(pipeline.has_capacity())

    def test_defaults_scale_with_the_cpu_count(self):
        with patch("core.tst_motion_engine.os.cpu_count", return_value=16):
            with TSTFramePipeline(self.engine) as pipeline:
                self.assertEqual(pipeline.max_workers, 4)
                self.assertEqual(pipeline.queue_depth, 8)
        with patch("core.tst_motion_engine.os.cpu_count", return_value=2):
            with TSTFramePipeline(self.engine) as pipeline:
                self.assertEqual(pipeline.max_workers, 2)
        with patch("core.tst_motion_engine.os.cpu_count", return_value=None):
            with TSTFramePipeline(self.engine) as pipeline:
                self.assertEqual(pipeline.max_workers, 1)

    def test_frame_size_change_is_raised_when_that_frame_is_popped(self):
        with TSTFramePipeline(self.engine, max_workers=2) as pipeline:
            pipeline.submit(1, self.frames[1], self.frames[0])
            wrong = cv2.resize(self.frames[2], (200, 150))
            pipeline.submit(2, wrong, self.frames[1])
            self.assertEqual(pipeline.pop()[0], 1)
            with self.assertRaises(ValueError):
                pipeline.pop()

    def test_worker_exception_surfaces_on_pop(self):
        with patch.object(
            self.engine, "_prepare_pair", side_effect=RuntimeError("boom")
        ):
            with TSTFramePipeline(self.engine, max_workers=2) as pipeline:
                pipeline.submit(1, self.frames[1], self.frames[0])
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    pipeline.pop()


if __name__ == "__main__":
    unittest.main()
