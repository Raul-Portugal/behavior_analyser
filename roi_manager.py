"""
Generic ROI (Region of Interest) management and utilities with validation.
Enhanced with comprehensive error handling and validation.
"""
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ROI:
    """
    Represents a single polygonal Region of Interest with validation.
    """
    
    def __init__(self, points: np.ndarray, category: str):
        """
        Initialize an ROI.
        
        Args:
            points: Nx2 numpy array of polygon vertices
            category: Category/name of this ROI
            
        Raises:
            ValueError: If points or category are invalid
        """
        if points is None or points.size == 0:
            raise ValueError("ROI points cannot be None or empty")
        
        if len(points.shape) != 2 or points.shape[1] != 2:
            raise ValueError(f"ROI points must be Nx2 array, got shape {points.shape}")
        
        if points.shape[0] < 3:
            raise ValueError(f"ROI must have at least 3 points, got {points.shape[0]}")
        
        if not category or not isinstance(category, str):
            raise ValueError("ROI category must be a non-empty string")
        
        self.points = points.astype(np.int32)
        self.category = category
        
        # Calculate and cache area for validation
        self._area = cv2.contourArea(self.points.astype(np.float32))
        if self._area <= 0:
            logger.warning(f"ROI '{category}' has zero or negative area ({self._area:.2f})")

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Returns bounding box as (x_min, y_min, x_max, y_max)."""
        x_coords, y_coords = self.points[:, 0], self.points[:, 1]
        return (int(np.min(x_coords)), int(np.min(y_coords)), 
                int(np.max(x_coords)), int(np.max(y_coords)))

    @property
    def width(self) -> float:
        """Returns width of bounding box."""
        x1, _, x2, _ = self.bounds
        return float(x2 - x1)
    
    @property
    def height(self) -> float:
        """Returns height of bounding box."""
        _, y1, _, y2 = self.bounds
        return float(y2 - y1)
    
    @property
    def area(self) -> float:
        """Returns area of the polygon."""
        return self._area

    def contains_point(self, point: Tuple[int, int]) -> bool:
        """
        Check if a point is inside this ROI.
        
        Args:
            point: (x, y) tuple
            
        Returns:
            True if point is inside, False otherwise
        """
        if point is None:
            return False
        
        try:
            result = cv2.pointPolygonTest(self.points.astype(np.float32), point, False)
            return result >= 0
        except Exception as e:
            logger.warning(f"Error in point-in-polygon test: {e}")
            return False

    def overlaps_with(self, other: 'ROI') -> bool:
        """
        Check if this ROI overlaps with another ROI.
        
        Args:
            other: Another ROI object
            
        Returns:
            True if ROIs overlap, False otherwise
        """
        try:
            # Quick bounding box check first
            x1_min, y1_min, x1_max, y1_max = self.bounds
            x2_min, y2_min, x2_max, y2_max = other.bounds
            
            if x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min:
                return False
            
            # Check if any vertex of one is inside the other
            for point in self.points:
                if other.contains_point(tuple(point)):
                    return True
            
            for point in other.points:
                if self.contains_point(tuple(point)):
                    return True
            
            return False
        except Exception as e:
            logger.warning(f"Error checking ROI overlap: {e}")
            return False

    def to_dict(self) -> Dict:
        """Serialize ROI to dictionary."""
        return {
            'points': self.points.tolist(),
            'category': self.category
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ROI':
        """
        Create ROI from dictionary.
        
        Args:
            data: Dictionary with 'points' and 'category' keys
            
        Returns:
            ROI object
            
        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        
        if 'points' not in data or 'category' not in data:
            raise ValueError("Data must contain 'points' and 'category' keys")
        
        return cls(
            points=np.array(data['points'], dtype=np.int32),
            category=data['category']
        )


class ROIManager:
    """
    Manages all ROIs for analysis in a maze-agnostic way.
    Enhanced with validation and error handling.
    """

    def __init__(self):
        """Initialize an empty ROI manager."""
        self.rois: Dict[str, List[ROI]] = {}
        self._reference_length_pixels = 0.0
        self._reference_name = "N/A"

    def add_roi(self, category: str, points: np.ndarray) -> None:
        """
        Adds a new ROI to the manager.
        
        Args:
            category: Category name for this ROI
            points: Nx2 array of polygon vertices
            
        Raises:
            ValueError: If ROI is invalid
        """
        try:
            roi = ROI(points, category)
            self.rois.setdefault(category, []).append(roi)
            logger.debug(f"Added ROI: {category}, area={roi.area:.2f}, bounds={roi.bounds}")
        except Exception as e:
            logger.error(f"Failed to add ROI for category '{category}': {e}")
            raise ValueError(f"Invalid ROI: {e}")

    def get_category(self, point: Optional[Tuple[int, int]]) -> str:
        """
        Determines a single category for a point (first match).
        
        Args:
            point: (x, y) tuple or None
            
        Returns:
            Category name or 'outside' if not in any ROI
        """
        if point is None:
            return 'outside'
        
        try:
            for category, roi_list in self.rois.items():
                if any(roi.contains_point(point) for roi in roi_list):
                    return category
        except Exception as e:
            logger.warning(f"Error determining category for point {point}: {e}")
        
        return 'outside'

    def get_overlapping_zones(self, point: Optional[Tuple[int, int]]) -> List[str]:
        """
        Returns a list of all zones a point falls within.
        
        Args:
            point: (x, y) tuple or None
            
        Returns:
            List of category names, or ['outside'] if not in any ROI
        """
        if point is None:
            return ['outside']
        
        try:
            zones = [
                cat for cat, roi_list in self.rois.items()
                for roi in roi_list if roi.contains_point(point)
            ]
            return zones if zones else ['outside']
        except Exception as e:
            logger.warning(f"Error determining overlapping zones for point {point}: {e}")
            return ['outside']

    def validate_rois(self, required_categories: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """
        Validates that ROIs are properly defined.
        
        Args:
            required_categories: List of category names that must be present (optional)
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if any ROIs are defined
        if not self.rois:
            issues.append("No ROIs defined")
            return False, issues
        
        # Check for empty categories
        for category, roi_list in self.rois.items():
            if not roi_list:
                issues.append(f"Category '{category}' has no ROIs")
        
        # Check for required categories
        if required_categories:
            for required in required_categories:
                if required not in self.rois or not self.rois[required]:
                    issues.append(f"Required category '{required}' is missing")
        
        # Check for very small ROIs
        for category, roi_list in self.rois.items():
            for i, roi in enumerate(roi_list):
                if roi.area < 100:  # Arbitrary small threshold
                    issues.append(f"ROI {category}[{i}] has very small area ({roi.area:.2f})")
        
        # Check for overlapping ROIs within same category (usually not desired)
        for category, roi_list in self.rois.items():
            if len(roi_list) > 1:
                for i in range(len(roi_list)):
                    for j in range(i + 1, len(roi_list)):
                        if roi_list[i].overlaps_with(roi_list[j]):
                            logger.warning(f"ROIs in category '{category}' overlap (indices {i}, {j})")
        
        is_valid = len(issues) == 0
        return is_valid, issues

    def calculate_reference_length(self) -> float:
        """
        Calculates a default reference length from the first available ROI's width
        if no reference line was user-drawn. This is a generic fallback.
        
        Returns:
            Reference length in pixels
        """
        if self._reference_length_pixels > 0:
            return self._reference_length_pixels

        # Sort keys for deterministic behavior
        for category in sorted(self.rois.keys()):
            if self.rois[category]:
                self._reference_length_pixels = self.rois[category][0].width
                self._reference_name = f"Width of first {category.replace('_', ' ').title()} ROI"
                logger.info(f"Using reference length: {self._reference_length_pixels:.2f} px "
                          f"({self._reference_name})")
                return self._reference_length_pixels

        logger.warning("No ROIs available to calculate reference length")
        self._reference_length_pixels = 0.0
        self._reference_name = "N/A"
        return 0.0

    def set_reference_length(self, length_pixels: float, name: str = "User-defined") -> None:
        """
        Manually set the reference length.
        
        Args:
            length_pixels: Length in pixels
            name: Descriptive name for this reference
            
        Raises:
            ValueError: If length is invalid
        """
        if length_pixels <= 0:
            raise ValueError(f"Reference length must be positive, got {length_pixels}")
        
        self._reference_length_pixels = length_pixels
        self._reference_name = name
        logger.info(f"Reference length set: {length_pixels:.2f} px ({name})")

    @property
    def reference_length_pixels(self) -> float:
        """Returns the reference length in pixels."""
        return self._reference_length_pixels

    @property
    def reference_name(self) -> str:
        """Returns the descriptive name of the reference."""
        return self._reference_name

    def draw_on_frame(self, frame: np.ndarray, colors: Optional[Dict[str, Tuple]] = None, 
                     thickness: int = 2) -> np.ndarray:
        """
        Draws all defined ROIs onto a frame.
        
        Args:
            frame: Input frame (BGR)
            colors: Optional dict mapping category names to (B, G, R) colors
            thickness: Line thickness for drawing
            
        Returns:
            Frame with ROIs drawn
        """
        if frame is None or frame.size == 0:
            logger.warning("Cannot draw on empty frame")
            return frame
        
        try:
            output = frame.copy()
            default_color = (200, 200, 200)
            
            for category, roi_list in self.rois.items():
                color = default_color
                if colors is not None:
                    color = colors.get(category, colors.get('default', default_color))
                
                for roi in roi_list:
                    try:
                        cv2.polylines(output, [roi.points.astype(np.int32)], 
                                    isClosed=True, color=color, thickness=thickness)
                    except Exception as e:
                        logger.warning(f"Error drawing ROI {category}: {e}")
            
            return output
        except Exception as e:
            logger.error(f"Error drawing ROIs on frame: {e}")
            return frame

    def get_stats(self) -> Dict:
        """
        Returns statistics about the ROIs.
        
        Returns:
            Dictionary with ROI statistics
        """
        stats = {
            'num_categories': len(self.rois),
            'total_rois': sum(len(roi_list) for roi_list in self.rois.values()),
            'categories': {},
            'reference_length': self._reference_length_pixels,
            'reference_name': self._reference_name
        }
        
        for category, roi_list in self.rois.items():
            total_area = sum(roi.area for roi in roi_list)
            stats['categories'][category] = {
                'count': len(roi_list),
                'total_area': total_area
            }
        
        return stats

    def to_dict(self) -> Dict:
        """Serializes the entire ROIManager to a dictionary."""
        return {
            'rois': {cat: [r.to_dict() for r in r_list] for cat, r_list in self.rois.items()},
            '_reference_length_pixels': self._reference_length_pixels,
            '_reference_name': self._reference_name
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ROIManager':
        """
        Creates an ROIManager instance from a dictionary.
        
        Args:
            data: Dictionary with ROI data
            
        Returns:
            ROIManager object
            
        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        
        manager = cls()
        
        try:
            for category, roi_list_data in data.get('rois', {}).items():
                manager.rois[category] = [ROI.from_dict(roi_data) for roi_data in roi_list_data]
            
            manager._reference_length_pixels = data.get('_reference_length_pixels', 0.0)
            manager._reference_name = data.get('_reference_name', 'N/A')
            
            logger.info(f"Loaded ROIManager with {len(manager.rois)} categories")
            return manager
        except Exception as e:
            raise ValueError(f"Failed to load ROIManager from dict: {e}")