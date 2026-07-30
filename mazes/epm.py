"""
Contains the maze-specific logic for the Elevated Plus Maze (EPM) task.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Type

from mazes.base_maze import BaseAnalysisResult, Maze
from core.zone_events import (
    debounce_zone_labels,
    dwell_frames,
    extract_center_mediated_entries,
    overlapping_zones_to_candidates,
)


@dataclass
class EPMAnalysisResult(BaseAnalysisResult):
    """EPM specific result fields."""
    # Time metrics
    time_in_open_arms_s: float = 0.0
    time_in_closed_arms_s: float = 0.0
    time_in_center_s: float = 0.0
    
    # Distance metrics (NEW)
    distance_in_open_arms: float = 0.0
    distance_in_closed_arms: float = 0.0
    distance_in_center: float = 0.0
    
    # Entry metrics
    entries_to_open_arms: int = 0
    entries_to_closed_arms: int = 0
    total_arm_entries: int = 0
    
    # Percentages
    percent_time_in_open: float = 0.0
    percent_entries_to_open: float = 0.0
    
    # Store tuples of (full_roi_name, timestamp)
    arm_entries: List[Tuple[str, float]] = field(default_factory=list)


class EPM(Maze):
    name = "Elevated Plus Maze"

    def __init__(self):
        self.min_dwell_seconds = 0.10
        self.count_initial_arm = True

    def get_roi_definitions(self) -> List[Tuple[str, str]]:
        return [
            ('open_arm_1', "Open Arm 1"), ('open_arm_2', "Open Arm 2"),
            ('closed_arm_1', "Closed Arm 1"), ('closed_arm_2', "Closed Arm 2"),
            ('center', "Center")
        ]

    def get_result_class(self) -> Type[BaseAnalysisResult]:
        return EPMAnalysisResult

    def resolve_tracking_labels(self, result: BaseAnalysisResult) -> None:
        candidates = overlapping_zones_to_candidates(result.overlapping_zones)
        resolved = debounce_zone_labels(
            candidates,
            dwell_frames(self.min_dwell_seconds, result.fps)
        )
        result.roi_labels = resolved
        result.visual_labels = list(resolved)

    def calculate_metrics(self, result: BaseAnalysisResult) -> None:
        if not isinstance(result, EPMAnalysisResult):
            raise TypeError("EPM logic requires an EPMAnalysisResult instance.")

        # Time calculation
        result.time_in_open_arms_s = sum(v for k, v in result.time_in_roi.items() if 'open' in k)
        result.time_in_closed_arms_s = sum(v for k, v in result.time_in_roi.items() if 'closed' in k)
        result.time_in_center_s = result.time_in_roi.get('center', 0.0)

        # Distance calculation (NEW)
        result.distance_in_open_arms = sum(v for k, v in result.distance_in_roi.items() if 'open' in k)
        result.distance_in_closed_arms = sum(v for k, v in result.distance_in_roi.items() if 'closed' in k)
        result.distance_in_center = result.distance_in_roi.get('center', 0.0)

        arm_labels = {
            'open_arm_1', 'open_arm_2',
            'closed_arm_1', 'closed_arm_2'
        }
        result.arm_entries = extract_center_mediated_entries(
            result.roi_labels,
            result.timestamps,
            arm_labels=arm_labels,
            count_initial_arm=self.count_initial_arm
        )

        # Summarize entries
        result.entries_to_open_arms = sum(1 for arm, ts in result.arm_entries if 'open' in arm)
        result.entries_to_closed_arms = sum(1 for arm, ts in result.arm_entries if 'closed' in arm)
        result.total_arm_entries = len(result.arm_entries)

        # Calculate percentages
        total_time_on_arms = result.time_in_open_arms_s + result.time_in_closed_arms_s
        total_entries_to_arms = result.entries_to_open_arms + result.entries_to_closed_arms
        
        result.percent_time_in_open = (result.time_in_open_arms_s / total_time_on_arms * 100) if total_time_on_arms > 0 else 0
        result.percent_entries_to_open = (result.entries_to_open_arms / total_entries_to_arms * 100) if total_entries_to_arms > 0 else 0

    def get_algorithm_parameters(self) -> dict:
        return {
            'min_dwell_seconds': self.min_dwell_seconds,
            'count_initial_arm': self.count_initial_arm
        }

    def configure_algorithm_parameters(self, parameters: dict) -> None:
        if not parameters:
            return
        self.min_dwell_seconds = float(
            parameters.get('min_dwell_seconds', self.min_dwell_seconds)
        )
        self.count_initial_arm = bool(
            parameters.get('count_initial_arm', self.count_initial_arm)
        )

    def get_batch_summary_headers(self) -> List[str]:
        """Return column headers for batch summary CSV."""
        return [
            'time_in_open_arms_s', 
            'time_in_closed_arms_s', 
            'time_in_center_s',
            'distance_in_open_arms',
            'distance_in_closed_arms',
            'distance_in_center',
            'entries_to_open_arms',
            'entries_to_closed_arms',
            'total_arm_entries',
            'percent_open_of_arm_time',
            'percent_open_of_arm_entries'
        ]

    def get_batch_summary_row(self, result: BaseAnalysisResult) -> List[any]:
        """Return data row for batch summary CSV."""
        if not isinstance(result, EPMAnalysisResult): return ['N/A'] * 11
        return [
            f"{result.time_in_open_arms_s:.2f}",
            f"{result.time_in_closed_arms_s:.2f}",
            f"{result.time_in_center_s:.2f}",
            f"{result.distance_in_open_arms:.2f}",
            f"{result.distance_in_closed_arms:.2f}",
            f"{result.distance_in_center:.2f}",
            result.entries_to_open_arms,
            result.entries_to_closed_arms,
            result.total_arm_entries,
            f"{result.percent_time_in_open:.2f}",
            f"{result.percent_entries_to_open:.2f}"
        ]

    def generate_specific_plots(self, result: BaseAnalysisResult, output_dir: Path, base_name: str) -> None:
        # EPM doesn't have highly specific plots like the Y-Maze sequence required here.
        pass
