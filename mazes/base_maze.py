"""
Defines the abstract base class for all maze types and their result containers.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

# Ordered step ids a setup flow may contain (the workspace and the classic
# wizard both read them from ``Maze.setup_steps``).
STEP_PARADIGM_PARAMS = "paradigm_params"
STEP_ROIS = "rois"
STEP_CALIBRATION = "calibration"
STEP_DETECTION_TUNING = "detection_tuning"
STEP_TST_TUNING = "tst_tuning"
STEP_TIME_WINDOW = "time_window"
STEP_REVIEW = "review"
ALL_SETUP_STEPS = (
    STEP_PARADIGM_PARAMS, STEP_ROIS, STEP_CALIBRATION, STEP_DETECTION_TUNING,
    STEP_TST_TUNING, STEP_TIME_WINDOW, STEP_REVIEW,
)

# ``Maze.analysis_kind``: how a video is analysed.
ANALYSIS_TRACKING = "tracking"   # centroid tracking, then zone/movement metrics
ANALYSIS_MOTION = "motion"       # per-animal motion decomposition (TST)

# ``Maze.wizard_kind``: which parameter panel / classic wizard flow applies.
KIND_STANDARD = "standard"
KIND_ARMS = "arms"
KIND_FREESTYLE = "freestyle"
KIND_TST = "tst"


@dataclass
class BaseAnalysisResult:
    """
    Container for generic analysis results, shared across maze types.
    This is the base class for all specific maze result dataclasses.
    """
    positions: List[Optional[Tuple[float, float]]] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    raw_roi_labels: List[str] = field(default_factory=list)
    roi_labels: List[str] = field(default_factory=list)
    visual_labels: List[str] = field(default_factory=list)
    overlapping_zones: List[List[str]] = field(default_factory=list)
    detection_confidences: List[float] = field(default_factory=list)
    detection_failure_reasons: List[str] = field(default_factory=list)
    time_in_roi: Dict[str, float] = field(default_factory=dict)
    distance_in_roi: Dict[str, float] = field(default_factory=dict)
    total_distance: float = 0.0
    excluded_jump_count: int = 0
    excluded_jump_distance: float = 0.0
    detection_rate: float = 0.0
    undetected_frames: int = 0
    undetected_time_s: float = 0.0
    tracking_quality: Dict[str, float] = field(default_factory=dict)
    tracking_quality_grade: str = "Not assessed"
    start_frame: int = 0
    end_frame: int = 0
    fps: float = 0.0
    scale_factor: float = 0.0
    # Run context recorded by Analyzer.finalize_result for paradigm metrics.
    # Neither is exported: they are inputs, not results.
    # Open-field arena polygon (Nx2 source pixels), None when no arena is set.
    arena_polygon: Optional[List[List[int]]] = None
    # Largest step (source px) counted in distances; None when unknown.
    jump_limit_px: Optional[float] = None

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
                'detection_rate': self.detection_rate,
                'undetected_frames': self.undetected_frames,
                'undetected_time_s': self.undetected_time_s,
                'excluded_jump_count': self.excluded_jump_count,
                'excluded_jump_distance': self.excluded_jump_distance,
                'tracking_quality_grade': self.tracking_quality_grade
            },
            'tracking_quality': self.tracking_quality,
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

    def get_algorithm_parameters(self) -> Dict[str, Any]:
        """Return paradigm parameters required to reproduce this analysis."""
        return {}

    def configure_algorithm_parameters(self, parameters: Dict[str, Any]) -> None:
        """Restore paradigm parameters from a saved run configuration."""
        if parameters:
            raise ValueError(f"{self.name} does not accept algorithm parameters")

    def needs_reference_line(self) -> bool:
        """Whether the setup flow should offer distance calibration."""
        return self.reports_distance

    # ------------------------------------------------------------------
    # Setup hooks: what the setup flow (classic wizard or workspace) asks
    # for, so it never has to branch on the paradigm itself.
    # ------------------------------------------------------------------
    @property
    def analysis_kind(self) -> str:
        """``"tracking"`` (centroid + zones) or ``"motion"`` (TST engine)."""
        return ANALYSIS_TRACKING

    @property
    def wizard_kind(self) -> str:
        """Parameter-panel family: standard, arms, freestyle or tst."""
        return KIND_STANDARD

    @property
    def uses_detection_tuner(self) -> bool:
        """Whether the centroid detection tuner is part of the setup."""
        return self.analysis_kind == ANALYSIS_TRACKING

    @property
    def uses_motion_tuner(self) -> bool:
        """Whether the TST motion tuner is part of the setup."""
        return self.analysis_kind == ANALYSIS_MOTION

    @property
    def has_per_roi_times(self) -> bool:
        """Whether every ROI (mouse) gets its own start/end window."""
        return False

    @property
    def reports_distance(self) -> bool:
        """Whether results contain distances (so a calibration is useful).

        When False the scale factor is always 0 (pixel units are never
        reported either).
        """
        return True

    @property
    def max_polygons_per_roi(self) -> Optional[int]:
        """Most polygons one ROI category may hold; None means unlimited."""
        return None

    def setup_steps(self) -> List[str]:
        """Ordered setup step ids (a subset of ``ALL_SETUP_STEPS``)."""
        steps = [STEP_ROIS]
        if self.reports_distance:
            steps.append(STEP_CALIBRATION)
        if self.uses_detection_tuner:
            steps.append(STEP_DETECTION_TUNING)
        if self.uses_motion_tuner:
            steps.append(STEP_TST_TUNING)
        steps += [STEP_TIME_WINDOW, STEP_REVIEW]
        return steps

    def time_window_subjects(self, roi_manager) -> List[str]:
        """ROI names that need their own time window (empty: one global)."""
        return []

    def validate_parameters(self) -> List[str]:
        """Problems with the paradigm parameters that block a run."""
        return []

    def validate_rois(self, roi_manager) -> List[str]:
        """Problems with one video's ROIs that block a run (empty: fine).

        Messages are exactly what the run pre-checks report, without the
        video-name prefix.
        """
        issues: List[str] = []
        if roi_manager is None:
            return ["No ROIs defined"]
        required = [name for name, _ in self.get_roi_definitions()]
        if required:
            _, roi_issues = roi_manager.validate_rois(required)
            issues.extend(roi_issues)
        return issues

    def validate_time_windows(self, settings) -> List[str]:
        """Problems with the per-ROI time windows of one video's settings."""
        return []

    def validate_settings(self, settings) -> List[str]:
        """Every maze-level problem of one video's settings."""
        issues = list(self.validate_rois(settings.roi_manager))
        issues.extend(self.validate_time_windows(settings))
        if settings.paradigm_id and settings.paradigm_id != self.name:
            issues.append(
                f"configuration is for '{settings.paradigm_id}', "
                f"not '{self.name}'"
            )
        return issues

    @classmethod
    def matches_roi_names(cls, roi_names: Iterable[str]) -> bool:
        """Legacy heuristic: do these ROI names look like this paradigm?

        Used only for saved configurations without a ``paradigm_id``.
        """
        return False

    @classmethod
    def matches_settings(cls, settings) -> bool:
        """Whether saved ``BatchSettings`` belong to this paradigm.

        ``settings.paradigm_id`` decides when it names a known paradigm;
        otherwise the legacy ROI-name heuristics are used, tried in registry
        order (so the catch-all Freestyle only matches what no other maze
        claims).
        """
        from mazes import maze_class_for_settings
        return maze_class_for_settings(settings) is cls

    def infer_from_settings(self, settings) -> None:
        """Restore this maze's parameters from saved ``BatchSettings``."""
        parameters = getattr(settings, 'algorithm_parameters', None)
        if parameters:
            self.configure_algorithm_parameters(parameters)

    def resolve_tracking_labels(self, result: BaseAnalysisResult) -> None:
        """Resolve raw frame labels before occupancy and metric calculation."""
        source = result.raw_roi_labels or result.roi_labels
        result.roi_labels = list(source)
        result.visual_labels = list(source)

    def export_specific_data(
        self,
        result: BaseAnalysisResult,
        output_dir: Path,
        base_name: str
    ) -> None:
        """Export optional paradigm-specific tables."""
        return None
