"""
core/models.py
Unified data models and configuration settings.
Replaces config.py and batch_processing.py.
"""
import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

# We will move ROIManager to core/roi.py in the next step, 
# for now, we assume it's still in the root or will be moved.
# If you haven't moved roi_manager yet, change this import to: from roi_manager import ROIManager
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
            raise ValueError(f"downsample_width must be positive")
        if self.ref_frame_samples <= 0:
            raise ValueError(f"ref_frame_samples must be positive")

@dataclass
class DetectionConfig:
    """Detection algorithm parameters."""
    threshold_percentile: float = 99.0
    use_weighting: bool = True
    weight_omega: float = 0.9
    window_size: int = 100

    def validate(self) -> None:
        if not 0.0 <= self.threshold_percentile <= 100.0:
            raise ValueError("threshold_percentile must be 0-100")
        if not 0.0 <= self.weight_omega <= 1.0:
            raise ValueError("weight_omega must be 0-1")

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
    def __init__(self, roi_manager: ROIManager, detection_config: DetectionConfig, 
                 scale_factor: float, start_time: float = 0.0, end_time: Optional[float] = None, 
                 create_timelapse: bool = False, per_roi_times: Optional[Dict[str, Tuple[float, float]]] = None):
        
        if roi_manager is None: raise ValueError("ROI manager cannot be None")
        
        self.roi_manager = roi_manager
        self.detection_config = detection_config
        self.scale_factor = scale_factor
        self.start_time = start_time
        self.end_time = end_time
        self.create_timelapse = create_timelapse
        self.per_roi_times = per_roi_times or {}

    def copy(self) -> 'BatchSettings':
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'roi_manager': self.roi_manager.to_dict(),
            'detection_config': self.detection_config.to_dict(),
            'scale_factor': self.scale_factor,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'create_timelapse': self.create_timelapse,
            'per_roi_times': self.per_roi_times
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchSettings':
        # ROI Manager must be imported to deserialize
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
            per_roi_times=per_roi_times
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
        except: return 'unknown'

    @staticmethod
    def _save_json(content: Dict, path: Path):
        with open(path, 'w') as f: json.dump(content, f, indent=2)

    @staticmethod
    def _load_json(path: Path) -> Dict:
        with open(path, 'r') as f: return json.load(f)