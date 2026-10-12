"""
core/models.py
Unified data models and configuration settings.
Replaces config.py and batch_processing.py.
"""
import copy
import json
import logging
from math import isfinite
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple, Any, Optional

if TYPE_CHECKING:
    from roi_manager import ROIManager

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================

@dataclass
class VideoConfig:
    """Video processing configuration."""
    downsample_width: int = 640
    analysis_width: int = 640 
    ref_frame_samples: int = 100

    def __post_init__(self):
        if self.downsample_width <= 0:
            raise ValueError("downsample_width must be positive")
        if self.ref_frame_samples <= 0:
            raise ValueError("ref_frame_samples must be positive")

@dataclass
class DetectionConfig:
    """Detection algorithm parameters."""
    threshold_percentile: float = 99.0
    use_weighting: bool = True
    weight_omega: float = 0.9
    window_size: int = 100
    min_component_area: int = 20
    max_component_area_ratio: float = 0.10
    morphology_kernel_size: int = 3
    min_detection_confidence: float = 0.15
    max_expected_jump_ratio: float = 0.20

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        numeric_values = {
            'threshold_percentile': self.threshold_percentile,
            'weight_omega': self.weight_omega,
            'window_size': self.window_size,
            'min_component_area': self.min_component_area,
            'max_component_area_ratio': self.max_component_area_ratio,
            'morphology_kernel_size': self.morphology_kernel_size,
            'min_detection_confidence': self.min_detection_confidence,
            'max_expected_jump_ratio': self.max_expected_jump_ratio,
        }
        for name, value in numeric_values.items():
            if not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.threshold_percentile <= 100.0:
            raise ValueError("threshold_percentile must be 0-100")
        if not 0.0 <= self.weight_omega <= 1.0:
            raise ValueError("weight_omega must be 0-1")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.min_component_area <= 0:
            raise ValueError("min_component_area must be positive")
        if not 0.0 < self.max_component_area_ratio <= 1.0:
            raise ValueError("max_component_area_ratio must be in (0, 1]")
        if self.morphology_kernel_size < 0:
            raise ValueError("morphology_kernel_size cannot be negative")
        if not 0.0 <= self.min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be 0-1")
        if not 0.0 < self.max_expected_jump_ratio <= 1.0:
            raise ValueError("max_expected_jump_ratio must be in (0, 1]")

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, data: Dict) -> 'DetectionConfig':
        valid_keys = cls.__annotations__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

@dataclass
class VisualizationConfig:
    """Visualization settings."""
    heatmap_blur_sigma: float = 5.0
    timelapse_fps_divider: int = 5
    timelapse_speed_multiplier: int = 5
    roi_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.roi_colors:
            self.roi_colors = {
                'default': (200, 200, 200),
                'arm_a': (255, 0, 0), 'arm_b': (0, 255, 0), 'arm_c': (0, 0, 255),
                'center': (255, 255, 0)
            }

@dataclass
class AppConfig:
    """Global application defaults."""
    video: VideoConfig = field(default_factory=VideoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

# ============================================================================
# BATCH SETTINGS (Runtime Configuration)
# ============================================================================

class BatchSettings:
    """
    Container for all settings required for a single analysis run.
    """
    SCHEMA_VERSION = 5

    def __init__(
        self,
        roi_manager: 'ROIManager',
        detection_config: DetectionConfig,
        scale_factor: float,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        create_timelapse: bool = False,
        per_roi_times: Optional[Dict[str, Tuple[float, float]]] = None,
        paradigm_id: Optional[str] = None,
        algorithm_parameters: Optional[Dict[str, Any]] = None
    ):
        
        if roi_manager is None: raise ValueError("ROI manager cannot be None")
        detection_config.validate()
        if not isfinite(float(scale_factor)) or scale_factor < 0:
            raise ValueError("scale_factor cannot be negative")
        if not isfinite(float(start_time)) or start_time < 0:
            raise ValueError("start_time cannot be negative")
        if end_time is not None:
            if not isfinite(float(end_time)) or end_time <= start_time:
                raise ValueError("end_time must be greater than start_time")
        normalized_roi_times = {}
        for name, interval in (per_roi_times or {}).items():
            if not isinstance(interval, (tuple, list)) or len(interval) != 2:
                raise ValueError(
                    f"per_roi_times[{name!r}] must be a start/end pair"
                )
            interval_start = float(interval[0])
            interval_end = float(interval[1])
            if (
                not isfinite(interval_start)
                or not isfinite(interval_end)
                or interval_start < 0
                or interval_end <= interval_start
            ):
                raise ValueError(
                    f"per_roi_times[{name!r}] is not a valid interval"
                )
            normalized_roi_times[str(name)] = (
                interval_start,
                interval_end,
            )
        
        self.roi_manager = roi_manager
        self.detection_config = detection_config
        self.scale_factor = scale_factor
        self.start_time = start_time
        self.end_time = end_time
        self.create_timelapse = create_timelapse
        self.per_roi_times = normalized_roi_times
        self.paradigm_id = paradigm_id
        self.algorithm_parameters = algorithm_parameters or {}

    def copy(self) -> 'BatchSettings':
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'schema_version': self.SCHEMA_VERSION,
            'roi_manager': self.roi_manager.to_dict(),
            'detection_config': self.detection_config.to_dict(),
            'scale_factor': self.scale_factor,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'create_timelapse': self.create_timelapse,
            'per_roi_times': self.per_roi_times,
            'paradigm_id': self.paradigm_id,
            'algorithm_parameters': self.algorithm_parameters
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchSettings':
        schema_version = int(data.get('schema_version', 1))
        if schema_version > cls.SCHEMA_VERSION:
            raise ValueError(
                f"Configuration schema {schema_version} is newer than "
                f"supported schema {cls.SCHEMA_VERSION}"
            )
        paradigm_id = data.get('paradigm_id')
        algorithm_parameters = data.get('algorithm_parameters', {})
        is_tst = (
            paradigm_id == "Tail Suspension Test"
            or any(
                str(key).startswith("mouse_")
                for key in data.get("roi_manager", {}).get("rois", {})
            )
        )
        if is_tst:
            from core.tst_motion_engine import TSTMotionEngine
            engine_id = (
                algorithm_parameters.get("engine_id")
                if isinstance(algorithm_parameters, dict) else None
            )
            if engine_id != TSTMotionEngine.ENGINE_ID:
                raise ValueError(
                    "This TST configuration was created for a legacy motion "
                    f"engine ({engine_id or 'unversioned'}) and cannot be "
                    "migrated safely: the current engine is "
                    f"{TSTMotionEngine.ENGINE_ID}, which normalizes motion "
                    "features by frame rate. Re-open the source video, "
                    "re-tune the TST evidence and save a new configuration."
                )
        # ROI Manager must be imported to deserialize
        from roi_manager import ROIManager
        roi_manager = ROIManager.from_dict(data['roi_manager'])
        detection_config = DetectionConfig.from_dict(data['detection_config'])
        
        # Handle tuple conversion for JSON loaded data
        raw_times = data.get('per_roi_times', {})
        per_roi_times = {k: tuple(v) for k, v in raw_times.items()}
        
        return cls(
            roi_manager=roi_manager,
            detection_config=detection_config,
            scale_factor=data['scale_factor'],
            start_time=data.get('start_time', 0.0),
            end_time=data.get('end_time'),
            create_timelapse=data.get('create_timelapse', False),
            per_roi_times=per_roi_times,
            paradigm_id=paradigm_id,
            algorithm_parameters=algorithm_parameters
        )

    # I/O Helpers
    def save_to_file(self, filepath: Path) -> None:
        self._save_json({'type': 'single_template', 'data': self.to_dict()}, filepath)

    @classmethod
    def load_from_file(cls, filepath: Path) -> 'BatchSettings':
        data = cls._load_json(filepath)
        if data.get('type') == 'batch_plan':
            raise ValueError("File is a Batch Plan, use load_batch_plan")
        content = data['data'] if 'type' in data else data
        return cls.from_dict(content)

    @staticmethod
    def save_batch_plan(plan: Dict[str, 'BatchSettings'], filepath: Path):
        serialized = {name: s.to_dict() for name, s in plan.items()}
        BatchSettings._save_json({'type': 'batch_plan', 'plan': serialized}, filepath)

    @staticmethod
    def load_batch_plan(filepath: Path) -> Dict[str, 'BatchSettings']:
        data = BatchSettings._load_json(filepath)
        if data.get('type') != 'batch_plan':
            raise ValueError("File is not a Batch Plan")
        return {k: BatchSettings.from_dict(v) for k, v in data['plan'].items()}

    @staticmethod
    def detect_file_type(filepath: Path) -> str:
        try:
            data = BatchSettings._load_json(filepath)
            return 'batch_plan' if data.get('type') == 'batch_plan' else 'template'
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return 'unknown'

    @staticmethod
    def _save_json(content: Dict, path: Path):
        path = Path(path)
        temp_path = path.with_suffix(path.suffix + '.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        temp_path.replace(path)

    @staticmethod
    def _load_json(path: Path) -> Dict:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
