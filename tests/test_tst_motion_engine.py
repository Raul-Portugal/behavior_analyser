import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np

    from core.tst_motion_engine import (
        REFERENCE_FPS, TSTMotionConfig, TSTMotionEngine,
    )
    from roi_manager import ROI
    from tests import tst_synthetic as synthetic
    HAS_OPENCV = True
except ModuleNotFoundError:
    HAS_OPENCV = False


@unittest.skipUnless(HAS_OPENCV, "OpenCV runtime is not installed")
class TSTMotionClassificationTests(unittest.TestCase):
    def setUp(self):
        self.engine = TSTMotionEngine({})

    def classify(self, **overrides):
        values = {
            "rigid_motion": 0.0,
            "residual_motion": 0.0,
            "deformation": 0.0,
            "coherence": 0.9,
            "injection": 0.0,
            "area_ratio": 0.2,
            "inlier_ratio": 0.9,
        }
        values.update(overrides)
        return self.engine._classify(**values)

    def test_config_rejects_fractional_reversal_count(self):
        with self.assertRaises(ValueError):
            TSTMotionConfig(passive_min_reversals=1.5).validate()

    def test_temporally_supported_rigid_swing_is_passive(self):
        observation = self.classify(
            rigid_motion=0.02,
            residual_motion=0.003,
            deformation=0.04,
            coherence=0.92,
            injection=0.002,
            temporal_ready=True,
            pendulum_score=0.90,
        )
        self.assertEqual(observation.state, "passive_swing")

    def test_nonrigid_deformation_is_active(self):
        observation = self.classify(
            rigid_motion=0.01,
            residual_motion=0.025,
            deformation=0.24,
            coherence=0.30,
        )
        self.assertEqual(observation.state, "active_escape")

    def test_abrupt_rigid_motion_injection_is_active(self):
        observation = self.classify(
            rigid_motion=0.025,
            residual_motion=0.003,
            deformation=0.04,
            coherence=0.90,
            injection=0.05,
        )
        self.assertEqual(observation.state, "active_escape")

    def test_abrupt_injection_cannot_be_labelled_static(self):
        observation = self.classify(
            rigid_motion=0.001,
            residual_motion=0.002,
            deformation=0.02,
            coherence=0.90,
            injection=0.05,
        )
        self.assertEqual(observation.state, "active_escape")

    def test_weak_motion_is_static_immobility(self):
        observation = self.classify(
            rigid_motion=0.001,
            residual_motion=0.002,
            deformation=0.02,
        )
        self.assertEqual(observation.state, "immobile_static")

    def test_end_to_end_rigid_translation_and_deformation(self):
        rng = np.random.default_rng(4)
        background = np.full((240, 200, 1), 220, dtype=np.int16)
        background += rng.integers(-8, 9, background.shape)
        background = np.clip(background, 0, 255).astype(np.uint8)
        background = np.repeat(background, 3, axis=2)

        def frame(horizontal_shift=0, add_limb=False):
            image = background.copy()
            cv2.ellipse(
                image,
                (100 + horizontal_shift, 120),
                (25, 45),
                0,
                0,
                360,
                (35, 35, 35),
                -1,
            )
            for x, y in [(90, 100), (110, 110), (95, 130), (108, 145)]:
                cv2.circle(
                    image,
                    (x + horizontal_shift, y),
                    3,
                    (120, 120, 120),
                    -1,
                )
            if add_limb:
                cv2.line(
                    image,
                    (115 + horizontal_shift, 125),
                    (150 + horizontal_shift, 145),
                    (35, 35, 35),
                    8,
                )
            return image

        roi = ROI(
            np.array(
                [[40, 20], [160, 20], [160, 220], [40, 220]],
                dtype=np.int32,
            ),
            "mouse_1",
        )
        rigid_engine = TSTMotionEngine({"mouse_1": roi})
        rigid = rigid_engine.calculate_observations(
            frame(5), frame(0)
        )["mouse_1"]
        deform_engine = TSTMotionEngine({"mouse_1": roi})
        deforming = deform_engine.calculate_observations(
            frame(0, add_limb=True), frame(0)
        )["mouse_1"]
        camera_engine = TSTMotionEngine({"mouse_1": roi})
        camera_shifted = cv2.warpAffine(
            frame(0),
            np.float32([[1, 0, 3], [0, 1, 2]]),
            (200, 240),
            borderMode=cv2.BORDER_REFLECT,
        )
        camera_motion = camera_engine.calculate_observations(
            camera_shifted, frame(0)
        )["mouse_1"]

        self.assertEqual(rigid.state, "uncertain")
        self.assertEqual(rigid.qc_flag, "insufficient_temporal_context")
        self.assertEqual(deforming.state, "active_escape")
        self.assertEqual(camera_motion.state, "immobile_static")
        self.assertTrue(camera_motion.camera_stabilized)
        self.assertGreater(rigid.coherence, 0.9)
        self.assertGreater(deforming.deformation, 0.06)

    def test_engine_state_round_trip_preserves_temporal_history(self):
        roi = ROI(
            np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=np.int32),
            "mouse_1",
        )
        source = TSTMotionEngine({"mouse_1": roi}, fps=25.0)
        history = source.histories["mouse_1"]
        history.centroid = (5.0, 7.0)
        history.rigid_vector = np.asarray([1.0, 2.0])
        history.angular_velocity = 0.03
        history.centroid_trace = [(4.0, 7.0), (5.0, 7.0)]
        history.rotation_delta_trace = [0.01, 0.03]

        restored = TSTMotionEngine({"mouse_1": roi}, fps=25.0)
        restored.restore_state(source.export_state())

        restored_history = restored.histories["mouse_1"]
        self.assertEqual(restored_history.centroid, (5.0, 7.0))
        np.testing.assert_allclose(
            restored_history.rigid_vector, [1.0, 2.0]
        )
        self.assertEqual(
            restored_history.rotation_delta_trace, [0.01, 0.03]
        )

    def test_dense_flow_is_computed_once_for_multiple_mice(self):
        roi_1 = ROI(
            np.array([[10, 10], [90, 10], [90, 110], [10, 110]], np.int32),
            "mouse_1",
        )
        roi_2 = ROI(
            np.array([[110, 10], [190, 10], [190, 110], [110, 110]], np.int32),
            "mouse_2",
        )
        frame = np.full((120, 200, 3), 220, np.uint8)
        cv2.ellipse(frame, (50, 60), (15, 25), 0, 0, 360, (30, 30, 30), -1)
        cv2.ellipse(frame, (150, 60), (15, 25), 0, 0, 360, (30, 30, 30), -1)
        shifted = np.roll(frame, 2, axis=1)
        engine = TSTMotionEngine(
            {"mouse_1": roi_1, "mouse_2": roi_2}
        )

        original = cv2.calcOpticalFlowFarneback
        calls = 0

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        with patch(
            "core.tst_motion_engine.cv2.calcOpticalFlowFarneback",
            side_effect=counted,
        ):
            engine.calculate_observations(shifted, frame)
        self.assertEqual(calls, 1)

    def test_oscillating_rigid_rotation_becomes_passive(self):
        background = np.full((180, 160, 3), 220, dtype=np.uint8)

        def rotated_frame(angle):
            image = background.copy()
            cv2.ellipse(
                image, (80, 90), (20, 35), 0, 0, 360,
                (35, 35, 35), -1,
            )
            for x, y in [(72, 72), (88, 80), (75, 100), (87, 112)]:
                cv2.circle(image, (x, y), 3, (125, 125, 125), -1)
            transform = cv2.getRotationMatrix2D((80, 90), angle, 1.0)
            return cv2.warpAffine(
                image,
                transform,
                (160, 180),
                borderValue=(220, 220, 220),
            )

        roi = ROI(
            np.array(
                [[20, 10], [140, 10], [140, 170], [20, 170]],
                dtype=np.int32,
            ),
            "mouse_1",
        )
        config = TSTMotionConfig(
            temporal_window_seconds=1.5,
            passive_min_observation_seconds=0.4,
        )
        engine = TSTMotionEngine({"mouse_1": roi}, config, fps=20.0)
        angles = 10.0 * np.sin(np.linspace(0, 4 * np.pi, 70))
        previous = rotated_frame(float(angles[0]))
        states = []
        for angle in angles[1:]:
            current = rotated_frame(float(angle))
            states.append(
                engine.calculate_observations(
                    current, previous
                )["mouse_1"].state
            )
            previous = current

        self.assertIn("passive_swing", states)

    def test_monotonic_rigid_translation_is_not_assumed_passive(self):
        background = np.full((160, 180, 3), 220, dtype=np.uint8)

        def translated_frame(shift):
            image = background.copy()
            cv2.ellipse(
                image, (90 + shift, 80), (18, 30), 0, 0, 360,
                (35, 35, 35), -1,
            )
            cv2.circle(
                image, (85 + shift, 75), 3, (125, 125, 125), -1
            )
            return image

        roi = ROI(
            np.array(
                [[20, 10], [160, 10], [160, 150], [20, 150]],
                dtype=np.int32,
            ),
            "mouse_1",
        )
        config = TSTMotionConfig(
            temporal_window_seconds=1.5,
            passive_min_observation_seconds=0.4,
        )
        engine = TSTMotionEngine({"mouse_1": roi}, config, fps=20.0)
        shifts = np.linspace(-15, 15, 50).astype(int)
        previous = translated_frame(int(shifts[0]))
        states = []
        for shift in shifts[1:]:
            current = translated_frame(int(shift))
            states.append(
                engine.calculate_observations(
                    current, previous
                )["mouse_1"].state
            )
            previous = current

        self.assertNotIn("passive_swing", states)

    def test_checkpoint_resume_matches_uninterrupted_observations(self):
        background = np.full((180, 160, 3), 220, dtype=np.uint8)

        def rotated_frame(angle):
            image = background.copy()
            cv2.ellipse(
                image, (80, 90), (20, 35), 0, 0, 360,
                (35, 35, 35), -1,
            )
            for x, y in [(72, 72), (88, 80), (75, 100), (87, 112)]:
                cv2.circle(image, (x, y), 3, (125, 125, 125), -1)
            transform = cv2.getRotationMatrix2D((80, 90), angle, 1.0)
            return cv2.warpAffine(
                image, transform, (160, 180),
                borderValue=(220, 220, 220),
            )

        roi = ROI(
            np.array(
                [[20, 10], [140, 10], [140, 170], [20, 170]],
                dtype=np.int32,
            ),
            "mouse_1",
        )
        config = TSTMotionConfig(
            temporal_window_seconds=1.5,
            passive_min_observation_seconds=0.4,
        )
        frames = [
            rotated_frame(float(angle))
            for angle in 10.0 * np.sin(np.linspace(0, 4 * np.pi, 55))
        ]
        uninterrupted = TSTMotionEngine(
            {"mouse_1": roi}, config, fps=20.0
        )
        uninterrupted_observations = []
        for previous, current in zip(frames, frames[1:]):
            uninterrupted_observations.append(
                uninterrupted.calculate_observations(
                    current, previous
                )["mouse_1"]
            )

        first_half = TSTMotionEngine(
            {"mouse_1": roi}, config, fps=20.0
        )
        split = 28
        for previous, current in zip(frames[:split], frames[1:split]):
            first_half.calculate_observations(current, previous)
        resumed = TSTMotionEngine(
            {"mouse_1": roi}, config, fps=20.0
        )
        resumed.restore_state(first_half.export_state())

        resumed_observations = []
        for index in range(split, len(frames)):
            resumed_observations.append(
                resumed.calculate_observations(
                    frames[index], frames[index - 1]
                )["mouse_1"]
            )

        expected = uninterrupted_observations[split - 1:]
        self.assertEqual(
            [item.state for item in resumed_observations],
            [item.state for item in expected],
        )
        np.testing.assert_allclose(
            [item.pendulum_score for item in resumed_observations],
            [item.pendulum_score for item in expected],
            rtol=1e-7,
            atol=1e-7,
        )


@unittest.skipUnless(HAS_OPENCV, "OpenCV runtime is not installed")
class TSTRateNormalizationTests(unittest.TestCase):
    """The same physical motion must score alike at 15, 30 and 60 fps."""

    RATES = (15.0, 30.0, 60.0)
    observations = {}

    @classmethod
    def setUpClass(cls):
        roi = ROI(synthetic.clip_roi_points(), "mouse_1")
        for fps in cls.RATES:
            frames = synthetic.render_clip(fps)
            engine = TSTMotionEngine({"mouse_1": roi}, fps=fps)
            cls.observations[fps] = [
                engine.calculate_observations(
                    frames[index], frames[index - 1]
                )["mouse_1"]
                for index in range(1, len(frames))
            ]

    @classmethod
    def _at(cls, fps, time_s):
        """Observation for the frame interval ending at ``time_s``."""
        return cls.observations[fps][int(round(time_s * fps)) - 1]

    @classmethod
    def _median(cls, fps, name, start, end):
        times = np.arange(start, end, 1.0 / REFERENCE_FPS)
        return float(np.median([
            getattr(cls._at(fps, t), name) for t in times
        ]))

    def test_states_agree_across_frame_rates(self):
        grid = np.arange(1, int(synthetic.DURATION_S * 15)) / 15.0
        reference = [self._at(30.0, t).state for t in grid]
        for fps in (15.0, 60.0):
            states = [self._at(fps, t).state for t in grid]
            agreement = np.mean([
                left == right for left, right in zip(states, reference)
            ])
            self.assertGreaterEqual(agreement, 0.90, f"{fps} fps")
        self.assertIn("passive_swing", reference)
        self.assertIn("active_escape", reference)

    def test_normalized_features_agree_across_frame_rates(self):
        swing = (synthetic.STATIC_END_S + 1.0, synthetic.SWING_END_S - 0.1)
        wiggle = (
            synthetic.WIGGLE_START_S + 0.25, synthetic.WIGGLE_END_S
        )
        for name, window, tolerance in (
            ("rigid_motion", swing, 0.25),
            ("residual_motion", swing, 0.25),
            ("residual_motion", wiggle, 0.25),
            ("deformation", wiggle, 0.30),
        ):
            expected = self._median(30.0, name, *window)
            self.assertGreater(expected, 0.0)
            for fps in (15.0, 60.0):
                value = self._median(fps, name, *window)
                self.assertAlmostEqual(
                    value / expected, 1.0, delta=tolerance,
                    msg=f"{name} at {fps} fps: {value} vs {expected}",
                )

    def test_normalized_injection_stays_below_active_threshold(self):
        threshold = TSTMotionConfig().active_injection_threshold
        window = (synthetic.STATIC_END_S + 1.0, synthetic.SWING_END_S - 0.1)
        expected = self._median(30.0, "motion_injection", *window)
        for fps in (15.0, 60.0):
            value = self._median(fps, "motion_injection", *window)
            self.assertLess(abs(value - expected), 0.25 * threshold)

    def test_static_period_has_no_motion_at_any_rate(self):
        for fps in self.RATES:
            for name in ("rigid_motion", "residual_motion", "deformation"):
                self.assertLess(
                    self._median(fps, name, 0.3, 1.1), 0.003, name
                )

    def test_rate_scales_are_relative_to_reference_fps(self):
        engine = TSTMotionEngine({}, fps=60.0)
        self.assertAlmostEqual(engine.rate_scale, 60.0 / REFERENCE_FPS)
        self.assertAlmostEqual(
            engine.injection_scale, (60.0 / REFERENCE_FPS) ** 2
        )


@unittest.skipUnless(HAS_OPENCV, "OpenCV runtime is not installed")
class TSTHistoryResetTests(unittest.TestCase):
    ROI = np.array([[10, 10], [190, 10], [190, 150], [10, 150]], np.int32)

    @staticmethod
    def _frame(x=None):
        rng = np.random.default_rng(3)
        gray = np.clip(
            215 + rng.integers(-5, 6, (160, 200, 1)), 0, 255
        ).astype(np.uint8)
        image = np.repeat(gray, 3, axis=2)
        if x is not None:
            cv2.ellipse(image, (int(x), 80), (18, 30), 0, 0, 360,
                        (35, 35, 35), -1)
            for dx, dy in [(-6, -12), (7, -4), (-4, 10), (6, 16)]:
                cv2.circle(image, (int(x) + dx, 80 + dy), 3,
                           (130, 130, 130), -1)
        return image

    def _run(self, positions):
        engine = TSTMotionEngine({"mouse_1": ROI(self.ROI, "mouse_1")})
        frames = [self._frame(x) for x in positions]
        return engine, [
            engine.calculate_observations(current, previous)["mouse_1"]
            for previous, current in zip(frames, frames[1:])
        ]

    def test_velocity_change_without_gap_is_an_injection(self):
        slow = [60 + 2 * index for index in range(8)]
        fast = [slow[-1] + 7 * (index + 1) for index in range(3)]
        _, observations = self._run(slow + fast)
        self.assertGreater(
            observations[len(slow) - 1].motion_injection,
            TSTMotionConfig().active_injection_threshold,
        )

    def test_segmentation_gap_resets_history_before_new_velocity(self):
        slow = [60 + 2 * index for index in range(8)]
        fast = [90 + 7 * index for index in range(5)]
        positions = slow + [None, None] + fast
        engine, observations = self._run(positions)

        gap_start = len(slow) - 1
        self.assertEqual(
            observations[gap_start].qc_flag, "mouse_segmentation_failed"
        )
        history = engine.histories["mouse_1"]
        first_valid = observations[gap_start + 3]
        self.assertEqual(first_valid.motion_injection, 0.0)
        self.assertEqual(first_valid.angular_injection, 0.0)
        self.assertFalse(first_valid.temporal_ready)
        # Only the post-gap frames remain in the pendulum window.
        valid_after_gap = len(fast) - 1
        self.assertEqual(len(history.centroid_trace), valid_after_gap)
        self.assertEqual(len(history.rotation_delta_trace), valid_after_gap)

    def test_reset_clears_every_temporal_trace(self):
        engine, _ = self._run([60 + 2 * index for index in range(6)])
        history = engine.histories["mouse_1"]
        self.assertIsNotNone(history.rigid_vector)
        self.assertTrue(history.centroid_trace)
        engine.calculate_observations(None, None)
        for name in (
            "rigid_vector", "angular_velocity",
        ):
            self.assertIsNone(getattr(history, name))
        for name in (
            "centroid_trace", "rotation_delta_trace", "coherence_trace",
            "residual_trace", "deformation_trace",
        ):
            self.assertEqual(getattr(history, name), [])


if __name__ == "__main__":
    unittest.main()
