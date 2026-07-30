"""Background-subtraction detection and evidence-based quality monitoring."""

import logging
from collections import deque
from dataclasses import dataclass
from math import hypot
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from core.models import DetectionConfig
from core.tracking_quality import (
    quality_grade_label,
    score_detection_candidate,
    summarize_tracking_quality,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectionResult:
    """One detector decision and the evidence supporting it."""

    position: Optional[Tuple[int, int]]
    confidence: float
    failure_reason: str = ""
    component_area: int = 0
    foreground_ratio: float = 0.0
    mean_contrast: float = 0.0

    @property
    def detected(self) -> bool:
        return self.position is not None


class DetectionQualityMonitor:
    """Summarize recent detection coverage, confidence, and jump quality."""

    def __init__(self, window_size: int = 50, frame_diagonal: float = 1000.0):
        self.recent_detections = deque(maxlen=window_size)
        self.recent_confidences = deque(maxlen=window_size)
        self.recent_positions = deque(maxlen=window_size)
        self.frames_processed = 0
        self.frame_diagonal = max(float(frame_diagonal), 1.0)

    def update(
        self,
        detected: bool,
        confidence: float = 0.0,
        position: Optional[Tuple[int, int]] = None
    ):
        self.recent_detections.append(bool(detected))
        self.recent_confidences.append(float(confidence) if detected else 0.0)
        self.recent_positions.append(position if detected else None)
        self.frames_processed += 1

    def get_statistics(self) -> Dict[str, float]:
        summary = summarize_tracking_quality(
            list(self.recent_positions),
            list(self.recent_confidences),
            frame_diagonal=self.frame_diagonal
        )
        return {
            'detection_rate': summary['detection_rate'] * 100.0,
            'confidence': summary['mean_confidence'] * 100.0,
            'tracking_stability': summary['tracking_stability'] * 100.0,
            'implausible_jump_rate': summary['implausible_jump_rate'] * 100.0,
            'frames_processed': self.frames_processed,
            'quality_grade': summary['quality_grade']
        }

    def get_quality_status(self) -> Tuple[str, str, str]:
        stats = self.get_statistics()
        label = quality_grade_label(stats['quality_grade'])
        if label == "Excellent":
            return label, "green", "Tracking evidence is internally consistent"
        if label == "Review":
            return label, "orange", "Review low-confidence frames and large jumps"
        return label, "red", "Adjust segmentation or improve recording conditions"

    def reset(self):
        self.recent_detections.clear()
        self.recent_confidences.clear()
        self.recent_positions.clear()
        self.frames_processed = 0


class DetectionEngine:
    """Connected-component animal detector over a median background reference."""

    MAX_FAILURES = 30

    def __init__(self, ref_frame: np.ndarray, config: DetectionConfig):
        if ref_frame is None or len(ref_frame.shape) != 2:
            raise ValueError("Invalid reference frame (must be grayscale 2D)")
        config.validate()

        self.ref_frame = ref_frame
        self.config = config
        self.h, self.w = ref_frame.shape
        self.last_position: Optional[Tuple[int, int]] = None
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
            x, y = self.last_position
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
            position = (int(round(cx)), int(round(cy)))
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
    ) -> Optional[Tuple[int, int]]:
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
                int(raw_position[0]),
                int(raw_position[1]),
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
