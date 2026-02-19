"""
core/analysis_engine.py
Unified Analysis Engine.
Contains logic for both Positional Tracking (Analyzer) and Motion Energy (MotionEngine).
Replaces analysis.py and motion_engine.py.
"""
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Generator, List, Dict

from core.video import BufferedVideoReader, VideoHandler
from core.detection import DetectionEngine
from core.models import AppConfig
# We still import ROI manager from root for now, to be moved later
from roi_manager import ROIManager, ROI
from mazes.base_maze import BaseAnalysisResult

logger = logging.getLogger(__name__)

# ============================================================================
# MOTION ENGINE (TST / Activity)
# ============================================================================

class MotionEngine:
    """Calculates pixel-wise motion energy within specific ROIs (For TST)."""
    
    def __init__(self, rois: Dict[str, ROI]):
        self.rois = rois
        self.masks = {}
        self._initialized = False

    def _init_masks(self, shape: Tuple[int, int]):
        h, w = shape[:2]
        for name, roi in self.rois.items():
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [roi.points.astype(np.int32)], 255)
            self.masks[name] = mask
        self._initialized = True

    def calculate_motion(self, frame: np.ndarray, prev_frame: np.ndarray) -> Dict[str, float]:
        if frame is None or prev_frame is None:
            return {n: 0.0 for n in self.rois}
            
        if not self._initialized:
            self._init_masks(frame.shape)

        g1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Simple absdiff for energy
        diff = cv2.absdiff(g1, g2)
        
        # Noise floor
        _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        
        results = {}
        for name, mask in self.masks.items():
            # Mean intensity inside the mask
            val = cv2.mean(thresh, mask=mask)[0]
            results[name] = val
            
        return results

# ============================================================================
# POSITIONAL ANALYZER (Tracking)
# ============================================================================

class Analyzer:
    """Positional tracking engine using DetectionEngine."""

    def __init__(self, video_path: Path, detection_engine: DetectionEngine, 
                 roi_manager: ROIManager, start_frame: int = 0, end_frame: Optional[int] = None):
        
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
        self.end_frame = min(end_frame or total_frames, total_frames)
        
        # Optimization: Downscaling
        self.analysis_width = self.config.video.analysis_width
        self.scale_ratio = 1.0
        self.analysis_dims = None
        
        if 0 < self.analysis_width < self.orig_w:
            self.scale_ratio = self.orig_w / self.analysis_width
            target_h = int(self.orig_h / self.scale_ratio)
            self.analysis_dims = (self.analysis_width, target_h)
            
            # Resize detection engine ref frame if needed
            if self.engine.ref_frame.shape[1] != self.analysis_width:
                self.engine.ref_frame = cv2.resize(self.engine.ref_frame, self.analysis_dims)
                self.engine.w = self.analysis_width
                self.engine.h = target_h

        # State for timelapse
        self.current_frame_img = None
        self.last_pos = None

    def process_frames(self) -> Generator[Tuple[int, Optional[Tuple[int, int]], List[str]], None, None]:
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
                detect_frame = frame
                if self.analysis_dims:
                    detect_frame = cv2.resize(frame, self.analysis_dims, interpolation=cv2.INTER_LINEAR)
                
                # 2. Detect
                raw_pos = self.engine.detect_position(detect_frame)
                
                # 3. Upscale & Map
                final_pos = None
                zones = ['outside']
                
                if raw_pos:
                    upscaled_x = int(raw_pos[0] * self.scale_ratio)
                    upscaled_y = int(raw_pos[1] * self.scale_ratio)
                    final_pos = (upscaled_x, upscaled_y)
                    zones = self.roi_manager.get_overlapping_zones(final_pos)
                
                self.last_pos = final_pos
                yield current_idx, final_pos, zones
                
                current_idx += 1
        finally:
            reader.release()

    def get_last_annotated_frame(self) -> np.ndarray:
        """Returns current frame with visualization for timelapse."""
        if self.current_frame_img is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
            
        vis = self.current_frame_img.copy()
        
        # Draw ROIs
        self.roi_manager.draw_on_frame(vis)
        
        # Draw Position
        if self.last_pos:
            cv2.circle(vis, self.last_pos, 8, (0, 0, 255), -1)
            
        return vis

    def finalize_result(self, result: BaseAnalysisResult):
        """Populates summary metrics in result object."""
        result.fps = self.fps
        result.start_frame = self.start_frame
        result.end_frame = self.end_frame
        
        # Calculate distances
        total_dist = 0.0
        result.distance_in_roi.clear()
        result.time_in_roi.clear()
        
        valid_positions = [p for p in result.positions if p]
        if not valid_positions: return

        # Simple distance accumulation
        prev = valid_positions[0]
        for i, pos in enumerate(result.positions):
            # Time
            zone = result.roi_labels[i] if i < len(result.roi_labels) else 'outside'
            result.time_in_roi[zone] = result.time_in_roi.get(zone, 0.0) + (1.0/self.fps)
            
            # Distance
            if pos:
                dist = np.sqrt((pos[0]-prev[0])**2 + (pos[1]-prev[1])**2)
                # Jump filter
                if dist < (self.orig_w / 3): 
                    total_dist += dist
                    result.distance_in_roi[zone] = result.distance_in_roi.get(zone, 0.0) + dist
                prev = pos
                
        result.total_distance = total_dist
        
        # Detection rate
        detected = sum(1 for p in result.positions if p)
        result.detection_rate = detected / len(result.positions) if result.positions else 0.0

def apply_scale_to_result(result: BaseAnalysisResult, scale_factor: float) -> None:
    """
    Helper function to convert pixel distances to real-world units.
    Moved from legacy analysis.py.
    """
    if result is None:
        raise ValueError("Result object cannot be None")
    
    if scale_factor <= 0:
        return
    
    try:
        result.scale_factor = scale_factor
        for category in result.distance_in_roi:
            result.distance_in_roi[category] *= scale_factor
        result.total_distance *= scale_factor
        logger.info(f"Distances converted to {result.distance_unit} using scale factor {scale_factor:.4f}")
    except Exception as e:
        logger.error(f"Error applying scale factor: {e}")