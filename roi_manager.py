"""
Generic ROI (Region of Interest) management and utilities with validation.
Enhanced with comprehensive error handling and validation.
"""
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Pre-3.2 configs may carry a reference named like "Width of first Arm A ROI".
LEGACY_BBOX_REFERENCE_PREFIX = "Width of first"

# Appended to the reference name when an alignment had to rescale a length
# that has no stored endpoints, so the result is flagged as approximate.
APPROXIMATE_REFERENCE_SUFFIX = " (approximate: scaled by alignment)"

# Reserved ROI name for the optional open-field arena. The arena is stored
# apart from ``ROIManager.rois`` so it is never treated as an analysis zone.
ARENA_CATEGORY = "arena"

# ROIManager.to_dict schema versions. Version 2 is written whenever no arena
# is set, so configurations without an arena serialize (and hash) exactly as
# they did before the arena existed.
ROI_SCHEMA_VERSION = 2
ROI_SCHEMA_VERSION_ARENA = 3

# Drawing styles (BGR) for the arena outline and its derived center region.
ARENA_DRAW_COLOR = (0, 165, 255)
CENTER_DRAW_COLOR = (255, 255, 0)


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
            # OpenCV 5 rejects numpy scalars here: hand it plain Python floats.
            point = (float(point[0]), float(point[1]))
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
                if other.contains_point((float(point[0]), float(point[1]))):
                    return True
            
            for point in other.points:
                if self.contains_point((float(point[0]), float(point[1]))):
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

    def __init__(self, source_dimensions: Optional[Tuple[int, int]] = None):
        """Initialize an empty ROI manager."""
        self.rois: Dict[str, List[ROI]] = {}
        # Optional open-field arena (floor boundary). Not an analysis zone:
        # it is excluded from zone assignment and from ``rois``.
        self.arena: Optional[ROI] = None
        # Cached center-region polygon, for overlays only. Never serialized.
        self.center_overlay: Optional[np.ndarray] = None
        self._reference_length_pixels = 0.0
        self._reference_name = "N/A"
        # Optional endpoints of the drawn reference line, in source pixels.
        # Kept so the calibration can follow a template to another camera
        # placement or resolution; ``None`` for length-only calibrations.
        self._reference_line: Optional[
            Tuple[Tuple[float, float], Tuple[float, float]]
        ] = None
        self._source_dimensions: Optional[Tuple[int, int]] = None
        if source_dimensions is not None:
            self.set_source_dimensions(source_dimensions)

    def set_source_dimensions(self, dimensions: Tuple[int, int]) -> None:
        """Record the source video dimensions for which ROI pixels are defined."""
        if not isinstance(dimensions, (tuple, list)) or len(dimensions) != 2:
            raise ValueError("Source dimensions must be a (width, height) pair")
        width, height = int(dimensions[0]), int(dimensions[1])
        if width <= 0 or height <= 0:
            raise ValueError("Source dimensions must be positive")
        self._source_dimensions = (width, height)

    @property
    def source_dimensions(self) -> Optional[Tuple[int, int]]:
        """Return the source video dimensions associated with these ROIs."""
        return self._source_dimensions

    def is_compatible_with(self, dimensions: Tuple[int, int]) -> bool:
        """Return whether raw ROI pixels can be used on the supplied dimensions."""
        if self._source_dimensions is None:
            return False
        return self._source_dimensions == (int(dimensions[0]), int(dimensions[1]))

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

    def set_arena(self, points: np.ndarray) -> None:
        """Define (or replace) the open-field arena polygon.

        Raises:
            ValueError: If the polygon is invalid.
        """
        try:
            self.arena = ROI(points, ARENA_CATEGORY)
        except Exception as e:
            logger.error(f"Failed to set arena: {e}")
            raise ValueError(f"Invalid arena: {e}")
        self.center_overlay = None

    def clear_arena(self) -> None:
        """Remove the arena and any cached center overlay."""
        self.arena = None
        self.center_overlay = None

    @property
    def has_arena(self) -> bool:
        return self.arena is not None

    def set_center_overlay(self, points: Optional[np.ndarray]) -> None:
        """Cache the center-region polygon that ``draw_on_frame`` should show."""
        if points is None:
            self.center_overlay = None
            return
        polygon = np.asarray(points)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            raise ValueError("Center overlay must be an Nx2 polygon (N >= 3)")
        self.center_overlay = polygon.astype(np.int32)

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
            # Multiple polygons may belong to the same category. Preserve
            # category insertion order without returning duplicate zone names.
            unique_zones = list(dict.fromkeys(zones))
            return unique_zones if unique_zones else ['outside']
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

        if self.arena is not None and self.arena.area < 100:
            issues.append(f"Arena has very small area ({self.arena.area:.2f})")
        
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
        Return the user-drawn reference length in pixels, or 0.0 if none.

        There is deliberately no fallback to an ROI bounding box: such a
        length is not the real-world quantity the user will type later.
        """
        return self._reference_length_pixels

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
        # A bare length no longer matches any stored endpoints.
        self._reference_line = None
        logger.info(f"Reference length set: {length_pixels:.2f} px ({name})")

    def set_reference_line(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        name: str = "User-defined",
    ) -> None:
        """Set the calibration from a drawn line, keeping its endpoints.

        The reference length is the distance between the endpoints (source
        pixels), so the two can never disagree.

        Raises:
            ValueError: If the endpoints are invalid or coincide.
        """
        line = self._validated_line(p1, p2)
        length = float(np.hypot(line[1][0] - line[0][0], line[1][1] - line[0][1]))
        self.set_reference_length(length, name)
        self._reference_line = line

    @staticmethod
    def _validated_line(p1, p2):
        try:
            line = (
                (float(p1[0]), float(p1[1])),
                (float(p2[0]), float(p2[1])),
            )
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"Reference line endpoints must be (x, y) pairs: {exc}")
        if not all(np.isfinite(v) for point in line for v in point):
            raise ValueError("Reference line endpoints must be finite")
        if line[0] == line[1]:
            raise ValueError("Reference line endpoints must differ")
        return line

    @property
    def reference_line(
        self,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Endpoints ((x1, y1), (x2, y2)) of the drawn reference line, or None."""
        return self._reference_line

    @property
    def reference_is_approximate(self) -> bool:
        """True when the length was scaled by an alignment without endpoints."""
        return self._reference_name.endswith(APPROXIMATE_REFERENCE_SUFFIX)

    @property
    def reference_length_pixels(self) -> float:
        """Returns the reference length in pixels."""
        return self._reference_length_pixels

    @property
    def reference_name(self) -> str:
        """Returns the descriptive name of the reference."""
        return self._reference_name

    @property
    def has_reference(self) -> bool:
        """True when a calibration length is available (user-drawn or legacy)."""
        return self._reference_length_pixels > 0

    @property
    def is_legacy_bbox_reference(self) -> bool:
        """True for old configs whose reference was an ROI bounding-box width."""
        return (
            self._reference_length_pixels > 0
            and self._reference_name.startswith(LEGACY_BBOX_REFERENCE_PREFIX)
        )

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
            
            if self.arena is not None:
                arena_color = ARENA_DRAW_COLOR
                if colors is not None:
                    arena_color = colors.get(ARENA_CATEGORY, arena_color)
                try:
                    cv2.polylines(output, [self.arena.points.astype(np.int32)],
                                  isClosed=True, color=arena_color,
                                  thickness=thickness + 1)
                except Exception as e:
                    logger.warning(f"Error drawing arena: {e}")
            if self.center_overlay is not None:
                try:
                    cv2.polylines(output, [self.center_overlay.astype(np.int32)],
                                  isClosed=True, color=CENTER_DRAW_COLOR,
                                  thickness=max(1, thickness - 1))
                except Exception as e:
                    logger.warning(f"Error drawing center overlay: {e}")

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
            'reference_name': self._reference_name,
            'source_dimensions': self._source_dimensions
        }
        
        for category, roi_list in self.rois.items():
            total_area = sum(roi.area for roi in roi_list)
            stats['categories'][category] = {
                'count': len(roi_list),
                'total_area': total_area
            }

        if self.arena is not None:
            stats['arena_area'] = self.arena.area

        return stats

    def to_dict(self) -> Dict:
        """Serializes the entire ROIManager to a dictionary."""
        data = {
            'schema_version': (
                ROI_SCHEMA_VERSION_ARENA if self.arena is not None
                else ROI_SCHEMA_VERSION
            ),
            'rois': {cat: [r.to_dict() for r in r_list] for cat, r_list in self.rois.items()},
            '_reference_length_pixels': self._reference_length_pixels,
            '_reference_name': self._reference_name,
            'source_dimensions': list(self._source_dimensions) if self._source_dimensions else None
        }
        # Written only when set, so configurations without an arena are
        # exactly what earlier versions wrote.
        if self.arena is not None:
            data['arena'] = self.arena.to_dict()
        # Same rule for the reference-line endpoints (older builds ignore
        # the key and keep using the stored length).
        if self._reference_line is not None:
            data['reference_line'] = [list(p) for p in self._reference_line]
        return data

    def _load_reference_line(self, raw_line) -> None:
        """Restore stored endpoints, dropping them if they disagree with the
        stored length (the length is what the calibration was made with)."""
        try:
            line = self._validated_line(raw_line[0], raw_line[1])
        except (TypeError, ValueError, IndexError) as exc:
            logger.warning("Ignoring invalid reference_line: %s", exc)
            return
        length = float(np.hypot(line[1][0] - line[0][0], line[1][1] - line[0][1]))
        stored = float(self._reference_length_pixels)
        if stored <= 0 or abs(length - stored) > 1e-3 * max(stored, 1.0):
            logger.warning(
                "Ignoring reference_line: endpoints span %.3f px but the "
                "stored reference length is %.3f px", length, stored
            )
            return
        self._reference_line = line

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
        
        schema_version = int(data.get('schema_version', 1))
        if schema_version > ROI_SCHEMA_VERSION_ARENA:
            raise ValueError(
                f"ROI schema {schema_version} is newer than supported "
                f"schema {ROI_SCHEMA_VERSION_ARENA}"
            )

        raw_dimensions = data.get('source_dimensions')
        manager = cls(tuple(raw_dimensions) if raw_dimensions else None)
        
        try:
            for category, roi_list_data in data.get('rois', {}).items():
                manager.rois[category] = [ROI.from_dict(roi_data) for roi_data in roi_list_data]
            
            raw_arena = data.get('arena')
            if raw_arena is not None:
                manager.arena = ROI.from_dict(raw_arena)

            manager._reference_length_pixels = data.get('_reference_length_pixels', 0.0)
            manager._reference_name = data.get('_reference_name', 'N/A')

            raw_line = data.get('reference_line')
            if raw_line is not None:
                manager._load_reference_line(raw_line)

            logger.info(f"Loaded ROIManager with {len(manager.rois)} categories")
            return manager
        except Exception as e:
            raise ValueError(f"Failed to load ROIManager from dict: {e}")
