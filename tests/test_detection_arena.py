import unittest

import cv2
import numpy as np

from core.analysis_geometry import AnalysisGeometry
from core.arena_zones import analysis_mask
from core.detection import DetectionEngine, percentile_uint8
from core.models import DetectionConfig

SIZE = 200
# Square arena covering the left 100 x 100 corner region of a 200 x 200 frame.
ARENA = np.array([[20, 20], [119, 20], [119, 119], [20, 119]])


def _config(**overrides):
    values = dict(
        threshold_percentile=95.0,
        use_weighting=False,
        min_component_area=20,
        morphology_kernel_size=3,
        min_detection_confidence=0.1,
    )
    values.update(overrides)
    return DetectionConfig(**values)


def _mask():
    return analysis_mask(ARENA, AnalysisGeometry(SIZE, SIZE, 0))


def _scene(animal_value=120, distractor_value=255):
    """Reference of zeros; a small animal inside the arena and a larger,
    brighter blob outside it."""
    reference = np.zeros((SIZE, SIZE), dtype=np.uint8)
    frame = reference.copy()
    cv2.rectangle(frame, (60, 60), (71, 71), animal_value, -1)          # 144 px
    cv2.rectangle(frame, (150, 140), (179, 169), distractor_value, -1)  # 900 px
    return reference, frame


class ArenaMaskedDetectionTests(unittest.TestCase):
    def test_distractor_outside_the_arena_wins_without_an_arena(self):
        reference, frame = _scene()
        result = DetectionEngine(reference, _config()).detect(frame)
        self.assertTrue(result.detected)
        self.assertGreater(result.position[0], 140)

    def test_distractor_outside_the_arena_is_ignored_with_an_arena(self):
        reference, frame = _scene()
        engine = DetectionEngine(reference, _config())
        engine.set_analysis_mask(_mask())
        result = engine.detect(frame)
        self.assertTrue(result.detected)
        self.assertAlmostEqual(result.position[0], 65.5, delta=1.0)
        self.assertAlmostEqual(result.position[1], 65.5, delta=1.0)

    def test_nothing_inside_the_arena_is_not_detected(self):
        reference = np.zeros((SIZE, SIZE), dtype=np.uint8)
        frame = reference.copy()
        cv2.rectangle(frame, (150, 140), (179, 169), 255, -1)
        engine = DetectionEngine(reference, _config())
        engine.set_analysis_mask(_mask())
        result = engine.detect(frame)
        self.assertFalse(result.detected)
        self.assertEqual(result.failure_reason, "no_foreground")

    def test_debug_images_are_restricted_too(self):
        reference, frame = _scene()
        engine = DetectionEngine(reference, _config())
        engine.set_analysis_mask(_mask())
        engine.detect(frame)
        images = engine.get_intermediate_images()
        outside = _mask() == 0
        self.assertEqual(int(images['diff'][outside].max()), 0)
        self.assertEqual(int(images['threshold'][outside].max()), 0)
        direct = engine.get_intermediate_images(frame)
        self.assertEqual(int(direct['threshold'][outside].max()), 0)

    def test_full_frame_mask_equals_no_mask(self):
        rng = np.random.default_rng(5)
        for use_weighting in (False, True):
            reference = rng.integers(0, 30, (SIZE, SIZE), dtype=np.uint8)
            plain = DetectionEngine(reference, _config(use_weighting=use_weighting))
            full = DetectionEngine(reference, _config(use_weighting=use_weighting))
            full.set_analysis_mask(np.full((SIZE, SIZE), 255, dtype=np.uint8))
            for step in range(8):
                frame = reference.copy()
                cx = 30 + 12 * step
                cv2.circle(frame, (cx, 100), 9, 220, -1)
                a = plain.detect(frame)
                b = full.detect(frame)
                self.assertEqual(a.position, b.position)
                self.assertEqual(a.confidence, b.confidence)
                self.assertEqual(a.failure_reason, b.failure_reason)

    def test_no_mask_state_is_none(self):
        engine = DetectionEngine(np.zeros((10, 10), dtype=np.uint8), _config())
        self.assertIsNone(engine.analysis_mask)
        engine.set_analysis_mask(np.full((10, 10), 1, dtype=np.uint8))
        self.assertIsNotNone(engine.analysis_mask)
        self.assertEqual(set(np.unique(engine.analysis_mask)), {255})
        engine.set_analysis_mask(None)
        self.assertIsNone(engine.analysis_mask)

    def test_wrong_shape_or_empty_mask_is_refused(self):
        engine = DetectionEngine(np.zeros((10, 10), dtype=np.uint8), _config())
        with self.assertRaises(ValueError):
            engine.set_analysis_mask(np.ones((5, 5), dtype=np.uint8))
        with self.assertRaises(ValueError):
            engine.set_analysis_mask(np.zeros((10, 10), dtype=np.uint8))

    def test_threshold_percentile_is_computed_inside_the_arena(self):
        # The arena is ~25% of the frame. Inside it: noise 0-20 plus an
        # animal covering ~3% of the arena. The 99th percentile over the
        # whole frame would sit in the noise; inside the arena it sits in the
        # animal, so only the animal survives thresholding.
        rng = np.random.default_rng(1)
        reference = np.zeros((SIZE, SIZE), dtype=np.uint8)
        frame = rng.integers(0, 21, (SIZE, SIZE), dtype=np.uint8)
        frame[60:72, 60:72] = rng.integers(100, 161, (12, 12), dtype=np.uint8)
        config = _config(threshold_percentile=99.0, morphology_kernel_size=0)
        engine = DetectionEngine(reference, config)
        engine.set_analysis_mask(_mask())
        result = engine.detect(frame)
        binary = engine.get_intermediate_images()['threshold']
        self.assertTrue(result.detected)
        rows, columns = np.nonzero(binary)
        self.assertGreater(len(rows), 50)
        self.assertTrue(rows.min() >= 60 and rows.max() <= 71)
        self.assertTrue(columns.min() >= 60 and columns.max() <= 71)

        whole = DetectionEngine(reference, config)._segment(frame, False)[1]
        outside_masked = np.where(_mask() > 0, whole, 0).astype(np.uint8)
        diluted = percentile_uint8(outside_masked, 99.0)
        arena_only = percentile_uint8(outside_masked, 99.0, _mask())
        self.assertLess(diluted, 25.0)
        self.assertGreater(arena_only, 100.0)


class MaskedPercentileTests(unittest.TestCase):
    def test_matches_numpy_over_the_masked_pixels(self):
        rng = np.random.default_rng(9)
        for trial in range(25):
            image = rng.integers(0, 256, (60, 80), dtype=np.uint8)
            mask = (rng.random((60, 80)) < rng.uniform(0.05, 0.9)).astype(
                np.uint8
            ) * 255
            if not mask.any():
                continue
            for q in (0.0, 1.0, 50.0, 95.0, 99.0, 99.9, 100.0):
                expected = float(np.percentile(image[mask > 0], q))
                self.assertEqual(
                    percentile_uint8(image, q, mask), expected, (trial, q)
                )

    def test_full_mask_equals_unmasked(self):
        rng = np.random.default_rng(2)
        image = rng.integers(0, 256, (40, 50), dtype=np.uint8)
        full = np.full((40, 50), 255, dtype=np.uint8)
        for q in (10.0, 99.0):
            self.assertEqual(
                percentile_uint8(image, q, full), percentile_uint8(image, q)
            )

    def test_empty_mask_returns_zero(self):
        image = np.full((5, 5), 9, dtype=np.uint8)
        self.assertEqual(
            percentile_uint8(image, 50.0, np.zeros((5, 5), dtype=np.uint8)), 0.0
        )


class MinAbsoluteContrastTests(unittest.TestCase):
    def _noise_frame(self):
        rng = np.random.default_rng(4)
        reference = np.zeros((SIZE, SIZE), dtype=np.uint8)
        # Weak Gaussian noise (mean 20, sd 8) and no animal at all.
        noise = np.clip(rng.normal(20.0, 8.0, (SIZE, SIZE)), 0, 255)
        return reference, noise.astype(np.uint8)

    def test_default_is_disabled_and_changes_nothing(self):
        self.assertEqual(DetectionConfig().min_absolute_contrast, 0.0)
        reference = np.zeros((SIZE, SIZE), dtype=np.uint8)
        _, frame = _scene()
        base = DetectionEngine(reference, _config()).detect(frame)
        explicit = DetectionEngine(
            reference, _config(min_absolute_contrast=0.0)
        ).detect(frame)
        self.assertEqual(base, explicit)

    def test_floor_suppresses_noise_only_frames(self):
        reference, frame = self._noise_frame()
        config = _config(
            threshold_percentile=99.0, morphology_kernel_size=0,
            min_component_area=1,
        )
        free = DetectionEngine(reference, config).detect(frame)
        self.assertNotEqual(free.failure_reason, "no_foreground")

        floored = DetectionEngine(
            reference, _config(
                threshold_percentile=99.0, morphology_kernel_size=0,
                min_component_area=1, min_absolute_contrast=100.0,
            )
        ).detect(frame)
        self.assertFalse(floored.detected)
        self.assertEqual(floored.failure_reason, "no_foreground")

    def test_floor_keeps_a_real_animal(self):
        reference, frame = _scene(animal_value=150, distractor_value=0)
        result = DetectionEngine(
            reference, _config(min_absolute_contrast=40.0)
        ).detect(frame)
        self.assertTrue(result.detected)

    def test_validation_and_serialization(self):
        for bad in (-1.0, 256.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError, msg=str(bad)):
                DetectionConfig(min_absolute_contrast=bad)

        # Disabled: serialized exactly as before the field existed.
        self.assertNotIn('min_absolute_contrast', DetectionConfig().to_dict())
        self.assertEqual(
            set(DetectionConfig().to_dict()),
            {'threshold_percentile', 'use_weighting', 'weight_omega',
             'window_size', 'min_component_area', 'max_component_area_ratio',
             'morphology_kernel_size', 'min_detection_confidence',
             'max_expected_jump_ratio'},
        )

        enabled = DetectionConfig(min_absolute_contrast=25.0)
        data = enabled.to_dict()
        self.assertEqual(data['min_absolute_contrast'], 25.0)
        self.assertEqual(DetectionConfig.from_dict(data), enabled)

    def test_legacy_dictionary_loads_with_the_floor_disabled(self):
        legacy = {k: v for k, v in DetectionConfig().to_dict().items()}
        self.assertEqual(
            DetectionConfig.from_dict(legacy).min_absolute_contrast, 0.0
        )


if __name__ == "__main__":
    unittest.main()
