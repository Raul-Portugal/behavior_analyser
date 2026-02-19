"""
core/detection.py
Unified Detection Module: Background Subtraction Engine & Quality Monitoring.
Replaces detection.py and detection_monitor.py.
"""
import logging
import cv2
import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple, Any

from core.models import DetectionConfig

logger = logging.getLogger(__name__)

# ============================================================================
# QUALITY MONITOR (Formerly detection_monitor.py)
# ============================================================================

class DetectionQualityMonitor:
    """Monitors detection stability and confidence in real-time."""
    
    def __init__(self, window_size: int = 50):
        self.recent_detections = deque(maxlen=window_size)
        self.recent_confidences = deque(maxlen=window_size)
        self.recent_positions = deque(maxlen=window_size)
        self.frames_processed = 0
    
    def update(self, detected: bool, confidence: float = 0.0, position: Optional[Tuple[int, int]] = None):
        self.recent_detections.append(1 if detected else 0)
        self.recent_confidences.append(confidence if detected else 0.0)
        self.recent_positions.append(position)
        self.frames_processed += 1
    
    def get_statistics(self) -> Dict[str, float]:
        det_rate = (sum(self.recent_detections) / len(self.recent_detections) * 100) if self.recent_detections else 0.0
        avg_conf = (sum(self.recent_confidences) / len(self.recent_confidences) * 100) if self.recent_confidences else 0.0
        return {
            'detection_rate': det_rate,
            'confidence': avg_conf,
            'tracking_stability': self._calculate_stability(),
            'frames_processed': self.frames_processed
        }

    def get_quality_status(self) -> Tuple[str, str, str]:
        stats = self.get_statistics()
        dr, conf, stab = stats['detection_rate'], stats['confidence'], stats['tracking_stability']
        
        if dr > 90 and conf > 50 and stab > 70:
            return "Excellent", "green", "Optimal tracking"
        elif dr > 70 and conf > 30:
            return "Fair", "orange", "Adjust threshold slightly"
        return "Poor", "red", "Check lighting or threshold"

    def _calculate_stability(self) -> float:
        valid = [p for p in self.recent_positions if p]
        if len(valid) < 3: return 0.0
        
        changes = []
        for i in range(1, len(valid)):
            dist = np.linalg.norm(np.array(valid[i]) - np.array(valid[i-1]))
            changes.append(dist)
        
        if not changes: return 0.0
        cv = np.std(changes) / (np.mean(changes) + 1e-6)
        return max(0, min(100, 100 - (cv * 20)))

    def reset(self):
        self.recent_detections.clear()
        self.recent_confidences.clear()
        self.recent_positions.clear()
        self.frames_processed = 0

# ============================================================================
# DETECTION ENGINE (Formerly detection.py)
# ============================================================================

class DetectionEngine:
    """Background subtraction-based tracking engine."""

    def __init__(self, ref_frame: np.ndarray, config: DetectionConfig):
        if ref_frame is None or len(ref_frame.shape) != 2:
            raise ValueError("Invalid reference frame (must be grayscale 2D)")
        
        self.ref_frame = ref_frame
        self.config = config
        self.h, self.w = ref_frame.shape
        self.last_position: Optional[Tuple[int, int]] = None
        self.consecutive_failures = 0
        self.MAX_FAILURES = 30 # Reset spatial weighting after this

    def detect_position(self, frame: np.ndarray, use_last_pos: bool = True) -> Optional[Tuple[int, int]]:
        if frame is None: return None
        
        # Ensure grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        if gray.shape != self.ref_frame.shape:
             # Auto-resize if mismatch (robustness)
             gray = cv2.resize(gray, (self.w, self.h))

        # Diff & Weighting
        diff = cv2.absdiff(gray, self.ref_frame)
        
        use_weighting = use_last_pos and self.consecutive_failures < self.MAX_FAILURES
        if self.config.use_weighting and use_weighting and self.last_position:
            mask = np.full_like(diff, 255 * (1.0 - self.config.weight_omega), dtype=np.uint8)
            x, y = self.last_position
            hw = self.config.window_size // 2
            x1, y1 = max(0, x-hw), max(0, y-hw)
            x2, y2 = min(self.w, x+hw), min(self.h, y+hw)
            mask[y1:y2, x1:x2] = 255
            
            # Fast integer math approximation of weighting
            diff = cv2.multiply(diff, mask, scale=1.0/255.0).astype(np.uint8)

        # Threshold
        thresh_val = max(5.0, np.percentile(diff, self.config.threshold_percentile))
        _, binary = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Centroid
        M = cv2.moments(binary)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Bounds check
            if 0 <= cx < self.w and 0 <= cy < self.h:
                self.last_position = (cx, cy)
                self.consecutive_failures = 0
                return (cx, cy)
        
        self.consecutive_failures += 1
        return None

    def get_intermediate_images(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Debug helper for tuner."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        if gray.shape != self.ref_frame.shape: gray = cv2.resize(gray, (self.w, self.h))
        
        diff = cv2.absdiff(gray, self.ref_frame)
        thresh_val = max(5.0, np.percentile(diff, self.config.threshold_percentile))
        _, binary = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
        
        return {'diff': diff, 'threshold': binary}