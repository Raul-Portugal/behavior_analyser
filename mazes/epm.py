"""
Contains the maze-specific logic for the Elevated Plus Maze (EPM) task.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Type

from mazes.base_maze import BaseAnalysisResult, Maze


@dataclass
class EPMAnalysisResult(BaseAnalysisResult):
    """EPM specific result fields."""
    # Time metrics
    time_in_open_arms_s: float = 0.0
    time_in_closed_arms_s: float = 0.0
    
    # Distance metrics (NEW)
    distance_in_open_arms: float = 0.0
    distance_in_closed_arms: float = 0.0
    
    # Entry metrics
    entries_to_open_arms: int = 0
    entries_to_closed_arms: int = 0
    
    # Percentages
    percent_time_in_open: float = 0.0
    percent_entries_to_open: float = 0.0
    
    # Store tuples of (full_roi_name, timestamp)
    arm_entries: List[Tuple[str, float]] = field(default_factory=list)


class EPM(Maze):
    name = "Elevated Plus Maze"

    def get_roi_definitions(self) -> List[Tuple[str, str]]:
        return [
            ('open_arm_1', "Open Arm 1"), ('open_arm_2', "Open Arm 2"),
            ('closed_arm_1', "Closed Arm 1"), ('closed_arm_2', "Closed Arm 2"),
            ('center', "Center")
        ]

    def get_result_class(self) -> Type[BaseAnalysisResult]:
        return EPMAnalysisResult

    def calculate_metrics(self, result: BaseAnalysisResult) -> None:
        if not isinstance(result, EPMAnalysisResult):
            raise TypeError("EPM logic requires an EPMAnalysisResult instance.")

        # Time calculation
        result.time_in_open_arms_s = sum(v for k, v in result.time_in_roi.items() if 'open' in k)
        result.time_in_closed_arms_s = sum(v for k, v in result.time_in_roi.items() if 'closed' in k)

        # Distance calculation (NEW)
        result.distance_in_open_arms = sum(v for k, v in result.distance_in_roi.items() if 'open' in k)
        result.distance_in_closed_arms = sum(v for k, v in result.distance_in_roi.items() if 'closed' in k)

        # Entry calculation using a simple state machine
        definitive_zone = None
        for i, zones in enumerate(result.overlapping_zones):
            timestamp = result.timestamps[i]
            current_arm = None
            if len(zones) == 1 and zones[0] != 'center' and zones[0] != 'outside':
                current_arm = zones[0]
            
            is_center = zones == ['center']

            if definitive_zone is None and current_arm:
                definitive_zone = current_arm
                result.arm_entries.append((current_arm, timestamp))
            elif definitive_zone is None and is_center:
                definitive_zone = 'center'
            elif definitive_zone == 'center' and current_arm:
                definitive_zone = current_arm
                result.arm_entries.append((current_arm, timestamp))
            elif definitive_zone is not None and definitive_zone != 'center' and is_center:
                definitive_zone = 'center'

        # Summarize entries
        result.entries_to_open_arms = sum(1 for arm, ts in result.arm_entries if 'open' in arm)
        result.entries_to_closed_arms = sum(1 for arm, ts in result.arm_entries if 'closed' in arm)

        # Calculate percentages
        total_time_on_arms = result.time_in_open_arms_s + result.time_in_closed_arms_s
        total_entries_to_arms = result.entries_to_open_arms + result.entries_to_closed_arms
        
        result.percent_time_in_open = (result.time_in_open_arms_s / total_time_on_arms * 100) if total_time_on_arms > 0 else 0
        result.percent_entries_to_open = (result.entries_to_open_arms / total_entries_to_arms * 100) if total_entries_to_arms > 0 else 0

    def get_batch_summary_headers(self) -> List[str]:
        """Return column headers for batch summary CSV."""
        return [
            'time_in_open_arms_s', 
            'time_in_closed_arms_s', 
            'distance_in_open_arms',   # NEW
            'distance_in_closed_arms', # NEW
            'entries_to_open_arms',
            'entries_to_closed_arms', 
            'percent_time_in_open', 
            'percent_entries_to_open'
        ]

    def get_batch_summary_row(self, result: BaseAnalysisResult) -> List[any]:
        """Return data row for batch summary CSV."""
        if not isinstance(result, EPMAnalysisResult): return ['N/A'] * 8
        return [
            f"{result.time_in_open_arms_s:.2f}",
            f"{result.time_in_closed_arms_s:.2f}",
            f"{result.distance_in_open_arms:.2f}",   # NEW
            f"{result.distance_in_closed_arms:.2f}", # NEW
            result.entries_to_open_arms,
            result.entries_to_closed_arms,
            f"{result.percent_time_in_open:.2f}",
            f"{result.percent_entries_to_open:.2f}"
        ]

    def generate_specific_plots(self, result: BaseAnalysisResult, output_dir: Path, base_name: str) -> None:
        # EPM doesn't have highly specific plots like the Y-Maze sequence required here.
        pass