"""Stable, paradigm-independent columns for batch summaries."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from mazes.base_maze import BaseAnalysisResult
else:
    BaseAnalysisResult = Any


COMMON_BATCH_HEADERS = [
    "video_filename",
    "paradigm",
    "analyzed_duration_s",
    "analyzed_frames",
    "total_distance",
    "distance_unit",
    "excluded_jump_count",
    "excluded_jump_distance",
    "tracking_quality",
    "detection_rate_pct",
    "mean_detection_confidence",
    "undetected_time_s",
]


def common_batch_row(
    video_path: Path,
    result: BaseAnalysisResult,
    paradigm_name: str,
) -> List[object]:
    """Return common columns; unsupported TST tracking metrics stay blank."""
    analyzed_frames = len(result.timestamps)
    duration = analyzed_frames / result.fps if result.fps > 0 else 0.0
    has_position_tracking = not hasattr(result, "motion_evidence")
    if has_position_tracking:
        distance_cells = [
            f"{result.total_distance:.2f}",
            result.distance_unit,
            getattr(result, "excluded_jump_count", 0),
            f"{getattr(result, 'excluded_jump_distance', 0.0):.2f}",
        ]
        quality_cells = [
            result.tracking_quality_grade,
            f"{result.detection_rate * 100:.2f}",
            f"{result.tracking_quality.get('mean_confidence', 0.0):.4f}",
            f"{result.undetected_time_s:.2f}",
        ]
    else:
        distance_cells = ["", "", "", ""]
        quality_cells = ["Not applicable", "", "", ""]

    return [
        video_path.name,
        paradigm_name,
        f"{duration:.2f}",
        analyzed_frames,
        *distance_cells,
        *quality_cells,
    ]
