import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np

    from core.tst_motion_engine import TSTMotionConfig, TSTMotionEngine
    from roi_manager import ROI
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


if __name__ == "__main__":
    unittest.main()
