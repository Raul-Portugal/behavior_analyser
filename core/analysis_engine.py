"""
core/analysis_engine.py
Unified Analysis Engine.
Contains positional tracking and the TST motion-decomposition engine.
Replaces analysis.py and motion_engine.py.
"""
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Generator, List

from core.analysis_geometry import AnalysisGeometry
from core.arena_zones import analysis_mask
from core.video import BufferedVideoReader, VideoHandler
from core.detection import DetectionEngine
from core.models import AppConfig
# We still import ROI manager from root for now, to be moved later
from roi_manager import ROIManager
from mazes.base_maze import BaseAnalysisResult
from core.tracking_quality import quality_grade_label, summarize_tracking_quality

logger = logging.getLogger(__name__)

# ============================================================================
# POSITIONAL ANALYZER (Tracking)
# ============================================================================

class Analyzer:
    """Positional tracking engine using DetectionEngine."""

    def __init__(self, video_path: Path, detection_engine: DetectionEngine,
                 roi_manager: ROIManager, start_frame: int = 0,
                 end_frame: Optional[int] = None,
                 result_start_frame: Optional[int] = None):
        
        self.video_path = Path(video_path)
        self.engine = detection_engine
        self.roi_manager = roi_manager
        self.config = AppConfig()
        
        # Video properties
        handler = VideoHandler(self.video_path)
        self.fps = handler.fps
        self.orig_w, self.orig_h = handler.width, handler.height
        total_frames = handler.total_frames
        
        self.start_frame = max(0, start_frame)
        self.result_start_frame = (
            max(0, result_start_frame)
            if result_start_frame is not None
            else self.start_frame
        )
        self.end_frame = min(end_frame or total_frames, total_frames)
        
        # Optimization: Downscaling (shared with the tuner dialog)
        self.geometry = AnalysisGeometry(
            self.orig_w, self.orig_h, self.config.video.analysis_width
        )
        self.analysis_width = self.geometry.analysis_width
        self.scale_ratio = self.geometry.scale_ratio
        self.analysis_dims = (
            self.geometry.analysis_dims if self.geometry.is_scaled else None
        )

        # Resize detection engine ref frame if needed
        self.engine.ref_frame = self.geometry.reference_to_analysis(
            self.engine.ref_frame
        )
        self.engine.h, self.engine.w = self.engine.ref_frame.shape[:2]

        # With an open-field arena, detection only looks inside it. Without
        # one the engine is left untouched (no mask, whole-frame detection).
        arena = getattr(self.roi_manager, 'arena', None)
        if arena is not None:
            self.engine.set_analysis_mask(
                analysis_mask(arena.points, self.geometry)
            )

        # State for timelapse
        self.current_frame_img = None
        self.last_pos = None

    def process_frames(self) -> Generator[Tuple[int, Optional[Tuple[float, float]], List[str]], None, None]:
        """Yields: (frame_idx, (x,y), [zone_names])"""
        
        # Use new core.video.BufferedVideoReader
        reader = BufferedVideoReader(self.video_path, start_frame=self.start_frame)
        
        current_idx = self.start_frame
        
        try:
            while current_idx < self.end_frame:
                ret, frame = reader.read()
                if not ret: break # EOF
                
                self.current_frame_img = frame # Reference for timelapse
                
                # 1. Downscale
                detect_frame = self.geometry.to_analysis(frame)
                
                # 2. Detect
                raw_pos = self.engine.detect_position(detect_frame)
                
                # 3. Upscale & Map
                final_pos = None
                zones = ['undetected']
                
                if raw_pos:
                    final_pos = self.geometry.to_source(raw_pos)
                    zones = self.roi_manager.get_overlapping_zones(final_pos)
                
                self.last_pos = final_pos
                yield current_idx, final_pos, zones
                
                current_idx += 1
        finally:
            reader.release()

    def finalize_result(self, result: BaseAnalysisResult):
        """Populates summary metrics in result object."""
        result.fps = self.fps
        result.start_frame = self.result_start_frame
        result.end_frame = self.result_start_frame + len(result.positions)
        
        # Calculate distances
        total_dist = 0.0
        result.distance_in_roi.clear()
        result.time_in_roi.clear()
        
        result.undetected_frames = 0
        result.undetected_time_s = 0.0
        result.excluded_jump_count = 0
        result.excluded_jump_distance = 0.0
        jump_limit = (
            float(np.hypot(self.orig_w, self.orig_h))
            * self.engine.config.max_expected_jump_ratio
        )
        result.jump_limit_px = jump_limit
        arena = getattr(self.roi_manager, 'arena', None)
        result.arena_polygon = (
            None if arena is None else arena.points.tolist()
        )

        # Distance is accumulated only across consecutive valid observations.
        # A missing observation breaks the trajectory instead of creating a
        # synthetic jump from the last known point to the next known point.
        prev = None
        prev_idx = None
        for i, pos in enumerate(result.positions):
            zone = result.roi_labels[i] if i < len(result.roi_labels) else 'undetected'

            if pos is None or zone == 'undetected':
                result.undetected_frames += 1
                result.undetected_time_s += (1.0 / self.fps)
                prev = None
                prev_idx = None
                continue

            # Only valid spatial observations contribute to zone occupancy.
            result.time_in_roi[zone] = result.time_in_roi.get(zone, 0.0) + (1.0/self.fps)

            if prev is not None and prev_idx == i - 1:
                dist = np.sqrt((pos[0]-prev[0])**2 + (pos[1]-prev[1])**2)
                if dist <= jump_limit:
                    total_dist += dist
                    result.distance_in_roi[zone] = result.distance_in_roi.get(zone, 0.0) + dist
                else:
                    result.excluded_jump_count += 1
                    result.excluded_jump_distance += float(dist)
            prev = pos
            prev_idx = i
                
        result.total_distance = total_dist
        
        # Detection rate
        result.tracking_quality = summarize_tracking_quality(
            result.positions,
            result.detection_confidences,
            frame_diagonal=float(np.hypot(self.orig_w, self.orig_h)),
            implausible_jump_ratio=self.engine.config.max_expected_jump_ratio
        )
        result.detection_rate = result.tracking_quality['detection_rate']
        result.tracking_quality_grade = quality_grade_label(
            result.tracking_quality['quality_grade']
        )

def apply_scale_to_result(result: BaseAnalysisResult, scale_factor: float) -> None:
    """
    Helper function to convert pixel distances to real-world units.
    Moved from legacy analysis.py.
    """
    if result is None:
        raise ValueError("Result object cannot be None")
    
    if scale_factor <= 0:
        return
    
    result.scale_factor = scale_factor
    for category in result.distance_in_roi:
        result.distance_in_roi[category] *= scale_factor
    result.total_distance *= scale_factor
    result.excluded_jump_distance *= scale_factor
    logger.info(
        f"Distances converted to {result.distance_unit} "
        f"using scale factor {scale_factor:.4f}"
    )
