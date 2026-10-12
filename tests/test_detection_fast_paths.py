"""Exactness tests for the fast detection paths against the v3.2.0 detector."""

import unittest
from math import hypot
from typing import Dict, Optional, Tuple

try:
    import cv2
    import numpy as np

    from core.detection import (
        DetectionEngine,
        DetectionResult,
        percentile_uint8,
    )
    from core.models import DetectionConfig
    from core.tracking_quality import score_detection_candidate

    HAS_OPENCV = True
except ModuleNotFoundError:
    HAS_OPENCV = False


class ReferenceDetectionEngine:
    # Verbatim copy of the v3.2.0 detector, kept as the exactness oracle.

    MAX_FAILURES = 30

    def __init__(self, ref_frame: np.ndarray, config: DetectionConfig):
        if ref_frame is None or len(ref_frame.shape) != 2:
            raise ValueError("Invalid reference frame (must be grayscale 2D)")
        config.validate()

        self.ref_frame = ref_frame
        self.config = config
        self.h, self.w = ref_frame.shape
        self.last_position: Optional[Tuple[float, float]] = None
        self.consecutive_failures = 0
        self.last_result = DetectionResult(
            None, 0.0, failure_reason="not_processed"
        )
        self._last_debug_images: Dict[str, np.ndarray] = {}

    def _segment(
        self,
        frame: np.ndarray,
        use_last_pos: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if len(frame.shape) == 3 else frame
        )
        if gray.shape != self.ref_frame.shape:
            gray = cv2.resize(gray, (self.w, self.h))

        raw_diff = cv2.absdiff(gray, self.ref_frame)
        weighted_diff = raw_diff
        weighting_active = (
            self.config.use_weighting
            and use_last_pos
            and self.last_position is not None
            and self.consecutive_failures < self.MAX_FAILURES
        )
        if weighting_active:
            mask = np.full_like(
                raw_diff,
                round(255 * (1.0 - self.config.weight_omega)),
                dtype=np.uint8
            )
            x = int(round(self.last_position[0]))
            y = int(round(self.last_position[1]))
            half_window = self.config.window_size // 2
            x1, y1 = max(0, x - half_window), max(0, y - half_window)
            x2, y2 = min(self.w, x + half_window), min(self.h, y + half_window)
            mask[y1:y2, x1:x2] = 255
            weighted_diff = cv2.multiply(
                raw_diff, mask, scale=1.0 / 255.0
            ).astype(np.uint8)

        threshold_value = max(
            5.0,
            float(np.percentile(
                weighted_diff,
                self.config.threshold_percentile
            ))
        )
        _, binary = cv2.threshold(
            weighted_diff, threshold_value, 255, cv2.THRESH_BINARY
        )

        kernel_size = int(self.config.morphology_kernel_size)
        if kernel_size > 1:
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return raw_diff, weighted_diff, binary

    def detect(
        self,
        frame: np.ndarray,
        use_last_pos: bool = True
    ) -> DetectionResult:
        if frame is None:
            return self._record_failure("empty_frame")

        raw_diff, weighted_diff, binary = self._segment(frame, use_last_pos)
        foreground_pixels = int(cv2.countNonZero(binary))
        frame_area = self.w * self.h
        foreground_ratio = (
            foreground_pixels / frame_area if frame_area else 0.0
        )
        selected_mask = np.zeros_like(binary)

        self._last_debug_images = {
            'diff': raw_diff,
            'weighted_diff': weighted_diff,
            'threshold': binary,
            'selected_component': selected_mask
        }

        if foreground_pixels == 0:
            return self._record_failure(
                "no_foreground", foreground_ratio=foreground_ratio
            )

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        max_area = max(
            self.config.min_component_area,
            int(frame_area * self.config.max_component_area_ratio)
        )
        max_expected_jump = (
            hypot(self.w, self.h) * self.config.max_expected_jump_ratio
        )

        best = None
        jump_rejections = 0
        for label_index in range(1, count):
            area = int(stats[label_index, cv2.CC_STAT_AREA])
            if area < self.config.min_component_area or area > max_area:
                continue

            cx, cy = centroids[label_index]
            position = (float(cx), float(cy))
            left = int(stats[label_index, cv2.CC_STAT_LEFT])
            top = int(stats[label_index, cv2.CC_STAT_TOP])
            width = int(stats[label_index, cv2.CC_STAT_WIDTH])
            height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
            label_region = labels[top:top + height, left:left + width]
            component_mask = (
                (label_region == label_index).astype(np.uint8) * 255
            )
            mean_contrast = (
                cv2.mean(
                    raw_diff[top:top + height, left:left + width],
                    mask=component_mask
                )[0]
                / 255.0
            )
            dominance = area / foreground_pixels
            distance = (
                hypot(
                    position[0] - self.last_position[0],
                    position[1] - self.last_position[1]
                )
                if (
                    self.last_position is not None
                    and use_last_pos
                    and self.consecutive_failures < self.MAX_FAILURES
                )
                else None
            )
            if distance is not None and distance > max_expected_jump:
                jump_rejections += 1
                continue
            confidence = score_detection_candidate(
                area=area,
                min_area=self.config.min_component_area,
                max_area=max_area,
                mean_contrast=mean_contrast,
                foreground_dominance=dominance,
                distance_from_last=distance,
                max_expected_jump=max_expected_jump if distance is not None else None
            )

            if best is None or confidence > best['confidence']:
                best = {
                    'label': label_index,
                    'position': position,
                    'area': area,
                    'mean_contrast': mean_contrast,
                    'confidence': confidence
                }

        if best is None:
            return self._record_failure(
                (
                    "jump_exceeded"
                    if jump_rejections > 0 else
                    "no_plausible_component"
                ),
                foreground_ratio=foreground_ratio
            )

        selected_mask[labels == best['label']] = 255
        self._last_debug_images['selected_component'] = selected_mask
        confidence = float(best['confidence'])
        if confidence < self.config.min_detection_confidence:
            return self._record_failure(
                "low_confidence",
                confidence=confidence,
                component_area=best['area'],
                foreground_ratio=foreground_ratio,
                mean_contrast=best['mean_contrast']
            )

        self.last_position = best['position']
        self.consecutive_failures = 0
        self.last_result = DetectionResult(
            position=best['position'],
            confidence=confidence,
            component_area=best['area'],
            foreground_ratio=foreground_ratio,
            mean_contrast=best['mean_contrast']
        )
        return self.last_result

    def _record_failure(
        self,
        reason: str,
        *,
        confidence: float = 0.0,
        component_area: int = 0,
        foreground_ratio: float = 0.0,
        mean_contrast: float = 0.0
    ) -> DetectionResult:
        self.consecutive_failures += 1
        self.last_result = DetectionResult(
            None,
            float(confidence),
            failure_reason=reason,
            component_area=int(component_area),
            foreground_ratio=float(foreground_ratio),
            mean_contrast=float(mean_contrast)
        )
        return self.last_result

    def detect_position(
        self,
        frame: np.ndarray,
        use_last_pos: bool = True
    ) -> Optional[Tuple[float, float]]:
        """Compatibility wrapper returning only the selected centroid."""
        return self.detect(frame, use_last_pos).position

    def export_state(self) -> Dict[str, object]:
        """Return the temporal detector state required for exact resume."""
        return {
            "last_position": (
                list(self.last_position)
                if self.last_position is not None else None
            ),
            "consecutive_failures": int(self.consecutive_failures),
        }

    def restore_state(self, state: Optional[Dict[str, object]]) -> None:
        """Restore state produced by :meth:`export_state`."""
        if not state:
            return
        raw_position = state.get("last_position")
        if raw_position is None:
            self.last_position = None
        else:
            self.last_position = (
                float(raw_position[0]),
                float(raw_position[1]),
            )
        self.consecutive_failures = max(
            0, int(state.get("consecutive_failures", 0))
        )

    def get_intermediate_images(
        self,
        frame: Optional[np.ndarray] = None,
        use_last_pos: bool = False
    ) -> Dict[str, np.ndarray]:
        """Return exact segmentation stages used by the detector."""
        if frame is not None:
            raw_diff, weighted_diff, binary = self._segment(
                frame, use_last_pos
            )
            return {
                'diff': raw_diff,
                'weighted_diff': weighted_diff,
                'threshold': binary
            }
        return {
            name: image.copy()
            for name, image in self._last_debug_images.items()
        }


def _frame_sequence(seed, count=40, size=(96, 128)):
    """Noisy frames with a moving blob, a second blob and occasional gaps."""
    rng = np.random.default_rng(seed)
    height, width = size
    reference = rng.integers(60, 120, size, dtype=np.uint8)
    frames = []
    x, y = width // 2, height // 2
    for index in range(count):
        frame = np.clip(
            reference.astype(np.int16) + rng.integers(-6, 7, size), 0, 255
        ).astype(np.uint8)
        x = int(np.clip(x + rng.integers(-9, 10), 8, width - 9))
        y = int(np.clip(y + rng.integers(-9, 10), 8, height - 9))
        if index % 11 == 5:
            pass  # empty arena frame
        elif index % 13 == 7:
            cv2.circle(frame, (width - 12, 12), 6, 250, -1)  # far jump
        else:
            cv2.circle(frame, (x, y), int(rng.integers(4, 9)), 240, -1)
            if index % 4 == 0:
                cv2.circle(frame, (10, height - 10), 3, 200, -1)
        if index % 7 == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frames.append(frame)
    return reference, frames


@unittest.skipUnless(HAS_OPENCV, "OpenCV runtime is not installed")
class PercentileUint8Tests(unittest.TestCase):
    def test_matches_numpy_on_random_arrays_and_quantiles(self):
        rng = np.random.default_rng(12345)
        quantiles = [0.0, 100.0, 50.0, 99.0, 99.5, 95.0, 1.0, 33.3333,
                     97.123456, 99.999, 0.001, 12.5, 87.5]
        checked = 0
        for _ in range(1500):
            rows = int(rng.integers(1, 40))
            cols = int(rng.integers(1, 40))
            kind = int(rng.integers(0, 4))
            if kind == 0:
                image = rng.integers(0, 256, (rows, cols), dtype=np.uint8)
            elif kind == 1:
                image = rng.integers(
                    0, int(rng.integers(1, 8)), (rows, cols), dtype=np.uint8
                )
            elif kind == 2:
                image = np.full((rows, cols), int(rng.integers(0, 256)),
                                dtype=np.uint8)
            else:
                image = (rng.random((rows, cols)) ** 6 * 255).astype(np.uint8)
            for q in quantiles + list(rng.uniform(0, 100, 6)):
                self.assertEqual(
                    percentile_uint8(image, q),
                    float(np.percentile(image, q)),
                    msg=f"shape={image.shape} q={q}",
                )
                checked += 1
        self.assertGreater(checked, 25000)

    def test_matches_numpy_on_frame_sized_images(self):
        rng = np.random.default_rng(7)
        for _ in range(60):
            image = np.clip(
                rng.normal(rng.uniform(0, 40), rng.uniform(1, 30), (360, 640)),
                0, 255,
            ).astype(np.uint8)
            for q in (0, 50, 90, 99, 99.9, 100, rng.uniform(0, 100)):
                self.assertEqual(
                    percentile_uint8(image, q), float(np.percentile(image, q))
                )

    def test_edge_cases(self):
        single = np.array([[7]], dtype=np.uint8)
        for q in (0, 37.5, 100):
            self.assertEqual(percentile_uint8(single, q), 7.0)
        constant = np.full((5, 5), 200, dtype=np.uint8)
        self.assertEqual(percentile_uint8(constant, 99.0), 200.0)
        two = np.array([[10, 20]], dtype=np.uint8)
        self.assertEqual(percentile_uint8(two, 25), 12.5)
        self.assertEqual(percentile_uint8(two, 50), 15.0)
        self.assertEqual(percentile_uint8(two, 100), 20.0)
        self.assertEqual(percentile_uint8(two, 0), 10.0)

    def test_non_contiguous_and_other_dtypes_are_exact(self):
        rng = np.random.default_rng(3)
        image = rng.integers(0, 256, (30, 40), dtype=np.uint8)
        view = image[::2, 1::3]
        self.assertEqual(
            percentile_uint8(view, 90.0), float(np.percentile(view, 90.0))
        )
        floats = rng.random((10, 10))
        self.assertEqual(
            percentile_uint8(floats, 90.0), float(np.percentile(floats, 90.0))
        )


@unittest.skipUnless(HAS_OPENCV, "OpenCV runtime is not installed")
class WeightingExactnessTests(unittest.TestCase):
    def test_lookup_weighting_matches_full_mask_multiply(self):
        rng = np.random.default_rng(99)
        for omega in np.linspace(0.0, 1.0, 101):
            raw = rng.integers(0, 256, (48, 64), dtype=np.uint8)
            config = DetectionConfig(weight_omega=float(omega), window_size=20)
            engine = DetectionEngine(np.zeros_like(raw), config)
            engine.last_position = (
                float(rng.uniform(-5, 70)), float(rng.uniform(-5, 55))
            )
            reference = ReferenceDetectionEngine(np.zeros_like(raw), config)
            reference.last_position = engine.last_position
            _, weighted_new, _ = engine._segment(raw, True)
            _, weighted_old, _ = reference._segment(raw, True)
            np.testing.assert_array_equal(weighted_new, weighted_old)
            self.assertEqual(weighted_new.dtype, weighted_old.dtype)


@unittest.skipUnless(HAS_OPENCV, "OpenCV runtime is not installed")
class DetectEquivalenceTests(unittest.TestCase):
    CONFIGS = [
        DetectionConfig(),
        DetectionConfig(use_weighting=False, threshold_percentile=95.0),
        DetectionConfig(weight_omega=0.5, window_size=40,
                        threshold_percentile=97.3,
                        morphology_kernel_size=4, min_component_area=8),
        DetectionConfig(weight_omega=1.0, window_size=15,
                        min_detection_confidence=0.4,
                        max_expected_jump_ratio=0.1),
        DetectionConfig(weight_omega=0.0, threshold_percentile=99.9,
                        morphology_kernel_size=0, min_component_area=3),
    ]

    def test_detect_matches_reference_on_recorded_sequences(self):
        for seed in range(6):
            reference_frame, frames = _frame_sequence(seed)
            for config in self.CONFIGS:
                new = DetectionEngine(reference_frame, config)
                old = ReferenceDetectionEngine(reference_frame, config)
                for index, frame in enumerate(frames):
                    use_last = index % 9 != 8
                    got = new.detect(frame, use_last_pos=use_last)
                    expected = old.detect(frame, use_last_pos=use_last)
                    self.assertEqual(got, expected, msg=f"seed={seed} i={index}")
                    self.assertEqual(new.last_position, old.last_position)
                    self.assertEqual(
                        new.consecutive_failures, old.consecutive_failures
                    )
                    new_images = new.get_intermediate_images()
                    old_images = old.get_intermediate_images()
                    self.assertEqual(list(new_images), list(old_images))
                    for name in old_images:
                        np.testing.assert_array_equal(
                            new_images[name], old_images[name], err_msg=name
                        )

    def test_sequences_exercise_detections_and_failures(self):
        reasons = set()
        detected = 0
        reference_frame, frames = _frame_sequence(2)
        engine = DetectionEngine(reference_frame, DetectionConfig())
        for frame in frames:
            result = engine.detect(frame)
            detected += result.detected
            reasons.add(result.failure_reason)
        self.assertGreater(detected, 5)
        self.assertGreater(len(reasons), 1)

    def test_none_frame_and_resized_frame_match(self):
        reference_frame, frames = _frame_sequence(1, count=3)
        config = DetectionConfig()
        new = DetectionEngine(reference_frame, config)
        old = ReferenceDetectionEngine(reference_frame, config)
        self.assertEqual(new.detect(None), old.detect(None))
        bigger = cv2.resize(frames[0], (256, 192))
        self.assertEqual(new.detect(bigger), old.detect(bigger))

    def test_debug_images_empty_before_first_detect(self):
        engine = DetectionEngine(np.zeros((10, 10), np.uint8), DetectionConfig())
        self.assertEqual(engine.get_intermediate_images(), {})

    def test_selected_component_is_built_on_demand(self):
        reference = np.zeros((100, 100), dtype=np.uint8)
        frame = reference.copy()
        cv2.rectangle(frame, (40, 45), (59, 64), 255, -1)
        engine = DetectionEngine(
            reference,
            DetectionConfig(
                threshold_percentile=95.0,
                use_weighting=False,
                min_component_area=20,
                min_detection_confidence=0.1,
            ),
        )
        self.assertTrue(engine.detect(frame).detected)
        self.assertNotIn("selected_component", engine._last_debug_images)
        selected = engine.get_intermediate_images()["selected_component"]
        self.assertGreater(int(np.count_nonzero(selected)), 0)


if __name__ == "__main__":
    unittest.main()
