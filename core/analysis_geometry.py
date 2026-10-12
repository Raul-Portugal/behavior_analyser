"""
core/analysis_geometry.py
Single source of truth for the analysis-resolution coordinate frame.

Detection runs on frames downsampled to ``analysis_width``; ROIs, exports
and overlays live in source-pixel space. Both the production Analyzer and
the interactive tuner use this class so pixel-valued detector parameters
(minimum area, window size, kernel, jump limit) mean the same thing in both.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# Interpolation used to shrink frames and the reference image for detection.
ANALYSIS_INTERPOLATION = cv2.INTER_LINEAR


@dataclass(frozen=True)
class AnalysisGeometry:
    """Immutable mapping between source frames and the analysis frame."""

    source_width: int
    source_height: int
    analysis_width: int = 0

    @property
    def is_scaled(self) -> bool:
        """True when frames are shrunk before detection."""
        return 0 < self.analysis_width < self.source_width

    @property
    def scale_ratio(self) -> float:
        """Source pixels per analysis pixel (1.0 when not downsampling)."""
        if not self.is_scaled:
            return 1.0
        return self.source_width / self.analysis_width

    @property
    def analysis_dims(self) -> Tuple[int, int]:
        """(width, height) of frames handed to the detector."""
        if not self.is_scaled:
            return self.source_width, self.source_height
        return (
            self.analysis_width,
            max(1, int(self.source_height / self.scale_ratio)),
        )

    def to_analysis(self, frame: np.ndarray) -> np.ndarray:
        """Resize a source-resolution frame to analysis resolution."""
        if not self.is_scaled:
            return frame
        return cv2.resize(
            frame, self.analysis_dims, interpolation=ANALYSIS_INTERPOLATION
        )

    def reference_to_analysis(self, ref: np.ndarray) -> np.ndarray:
        """Resize a reference frame to analysis resolution if needed."""
        if not self.is_scaled:
            return ref
        if (ref.shape[1], ref.shape[0]) == self.analysis_dims:
            return ref
        return cv2.resize(
            ref, self.analysis_dims, interpolation=ANALYSIS_INTERPOLATION
        )

    def to_source(
        self, point: Optional[Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        """Map an analysis-space point to source pixel coordinates.

        Uses the pixel-center convention ``(x + 0.5) * ratio - 0.5``.
        """
        if point is None:
            return None
        ratio = self.scale_ratio
        if ratio == 1.0:
            return float(point[0]), float(point[1])
        return (
            (float(point[0]) + 0.5) * ratio - 0.5,
            (float(point[1]) + 0.5) * ratio - 0.5,
        )

    def to_analysis_point(
        self, point: Optional[Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        """Inverse of :meth:`to_source`."""
        if point is None:
            return None
        ratio = self.scale_ratio
        if ratio == 1.0:
            return float(point[0]), float(point[1])
        return (
            (float(point[0]) + 0.5) / ratio - 0.5,
            (float(point[1]) + 0.5) / ratio - 0.5,
        )
