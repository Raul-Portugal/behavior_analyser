"""
Defines the abstract base class for all maze types and their result containers.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type


@dataclass
class BaseAnalysisResult:
    """
    Container for generic analysis results, shared across maze types.
    This is the base class for all specific maze result dataclasses.
    """
    positions: List[Optional[Tuple[int, int]]] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    roi_labels: List[str] = field(default_factory=list)
    visual_labels: List[str] = field(default_factory=list)
    overlapping_zones: List[List[str]] = field(default_factory=list)
    time_in_roi: Dict[str, float] = field(default_factory=dict)
    distance_in_roi: Dict[str, float] = field(default_factory=dict)
    total_distance: float = 0.0
    detection_rate: float = 0.0
    start_frame: int = 0
    end_frame: int = 0
    fps: float = 0.0
    scale_factor: float = 0.0

    @property
    def distance_unit(self) -> str:
        """Returns the unit for distance measurements."""
        return "cm" if self.scale_factor > 0 else "pixels"

    def to_dict(self) -> Dict:
        """Serializes the result object to a dictionary for export."""
        return {
            'metadata': {
                'start_frame': self.start_frame, 'end_frame': self.end_frame,
                'total_frames': len(self.positions), 'fps': self.fps,
                'scale_factor': self.scale_factor, 'distance_unit': self.distance_unit,
                'detection_rate': self.detection_rate
            },
            'time_in_roi': self.time_in_roi,
            'distance_in_roi': self.distance_in_roi,
            'total_distance': self.total_distance,
        }


class Maze(ABC):
    """
    Abstract Base Class for a maze.
    This class defines the interface for all maze-specific logic.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """The user-friendly name of the maze."""
        pass

    @abstractmethod
    def get_roi_definitions(self) -> List[Tuple[str, str]]:
        """Returns a list of tuples for ROI selection: (internal_name, user_friendly_description)."""
        pass

    @abstractmethod
    def get_result_class(self) -> Type[BaseAnalysisResult]:
        """Returns the specific dataclass type for this maze's results."""
        pass

    @abstractmethod
    def calculate_metrics(self, result: BaseAnalysisResult) -> None:
        """
        Calculates maze-specific metrics and populates them into the result object.
        This is called *after* the main generic tracking loop is complete.
        """
        pass

    @abstractmethod
    def get_batch_summary_headers(self) -> List[str]:
        """Returns the specific headers for this maze's batch summary CSV."""
        pass

    @abstractmethod
    def get_batch_summary_row(self, result: BaseAnalysisResult) -> List[any]:
        """Returns a list of values for one row of the batch summary CSV."""
        pass

    @abstractmethod
    def generate_specific_plots(self, result: BaseAnalysisResult, output_dir: Path, base_name: str) -> None:
        """
        Generates any plots that are unique to this maze type.
        Generic plots like heatmaps are handled by the main visualizer.
        """
        pass