"""Dependency-light scoring and quality summaries for positional tracking."""

from math import hypot
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple


Position = Optional[Tuple[int, int]]


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def score_detection_candidate(
    *,
    area: int,
    min_area: int,
    max_area: int,
    mean_contrast: float,
    foreground_dominance: float,
    distance_from_last: Optional[float] = None,
    max_expected_jump: Optional[float] = None
) -> float:
    """Score a foreground component using interpretable normalized evidence."""
    if area < min_area or area > max_area or min_area <= 0 or max_area < min_area:
        return 0.0

    area_target = max(min_area * 4, 1)
    area_score = clamp01(area / area_target)
    contrast_score = clamp01(mean_contrast)
    dominance_score = clamp01(foreground_dominance)

    if distance_from_last is None or not max_expected_jump:
        return clamp01(
            0.25 * area_score
            + 0.40 * contrast_score
            + 0.35 * dominance_score
        )

    proximity_score = clamp01(1.0 - distance_from_last / max_expected_jump)
    return clamp01(
        0.20 * area_score
        + 0.30 * contrast_score
        + 0.25 * dominance_score
        + 0.25 * proximity_score
    )


def summarize_tracking_quality(
    positions: Sequence[Position],
    confidences: Sequence[float],
    *,
    frame_diagonal: float,
    low_confidence_threshold: float = 0.40,
    implausible_jump_ratio: float = 0.20
) -> Dict[str, float]:
    """Summarize coverage, confidence, and consecutive-frame jump quality."""
    total_frames = len(positions)
    detected_indices = [
        index for index, position in enumerate(positions)
        if position is not None
    ]
    detected_count = len(detected_indices)
    detection_rate = detected_count / total_frames if total_frames else 0.0

    detected_confidences = [
        float(confidences[index])
        for index in detected_indices
        if index < len(confidences)
    ]
    mean_confidence = (
        mean(detected_confidences) if detected_confidences else 0.0
    )
    low_confidence_frames = sum(
        confidence < low_confidence_threshold
        for confidence in detected_confidences
    )

    jumps: List[float] = []
    for previous, current in zip(positions, positions[1:]):
        if previous is None or current is None:
            continue
        jumps.append(hypot(current[0] - previous[0], current[1] - previous[1]))

    jump_limit = max(0.0, frame_diagonal * implausible_jump_ratio)
    implausible_jumps = (
        sum(jump > jump_limit for jump in jumps) if jump_limit > 0 else 0
    )
    implausible_jump_rate = (
        implausible_jumps / len(jumps) if jumps else 0.0
    )
    tracking_stability = max(0.0, 1.0 - implausible_jump_rate)

    if (
        detection_rate >= 0.95
        and mean_confidence >= 0.60
        and implausible_jump_rate <= 0.01
    ):
        quality_grade = 2.0  # Excellent
    elif (
        detection_rate >= 0.80
        and mean_confidence >= 0.40
        and implausible_jump_rate <= 0.05
    ):
        quality_grade = 1.0  # Review
    else:
        quality_grade = 0.0  # Poor

    return {
        'total_frames': float(total_frames),
        'detected_frames': float(detected_count),
        'detection_rate': detection_rate,
        'mean_confidence': mean_confidence,
        'low_confidence_frames': float(low_confidence_frames),
        'mean_consecutive_jump_px': mean(jumps) if jumps else 0.0,
        'implausible_jump_rate': implausible_jump_rate,
        'tracking_stability': tracking_stability,
        'quality_grade': quality_grade
    }


def quality_grade_label(quality_grade: float) -> str:
    if quality_grade >= 2:
        return "Excellent"
    if quality_grade >= 1:
        return "Review"
    return "Poor"
