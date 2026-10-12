"""TST motion decomposition that separates passive rigid swing from activity.

The engine deliberately produces evidence rather than only a binary decision.
Whole-body translation/rotation is modelled separately from residual,
non-rigid motion so pendulum-like movement can be treated as passive.
"""

import weakref
from dataclasses import asdict, dataclass, field, replace
from math import ceil, isfinite
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from roi_manager import ROI

# Per-frame motion features are expressed relative to this frame rate so the
# thresholds in TSTMotionConfig keep one meaning regardless of the video fps.
REFERENCE_FPS = 30.0
# Frames are analysed at one canonical width (never upscaled) so features and
# their thresholds do not depend on the source resolution.
DEFAULT_ANALYSIS_WIDTH = 480
MIN_ANALYSIS_WIDTH = 160
# Processing is confined to the union bounding box of the ROIs plus this
# fraction of its size on every side (clamped to the frame).
ROI_MARGIN_FRACTION = 0.15

# Constant structuring elements; cv2 only reads them, so one instance is shared
# by every segmentation call (and every thread).
_ERODE_KERNEL = np.ones((7, 7), dtype=np.uint8)
_OPEN_KERNEL = np.ones((3, 3), dtype=np.uint8)
_CLOSE_KERNEL = np.ones((7, 7), dtype=np.uint8)
# A ROI border band with fewer pixels than this is too small to estimate the
# background from, so the whole ROI is used instead.
_MIN_BACKGROUND_PIXELS = 20


@dataclass
class TSTMotionConfig:
    segmentation_contrast: float = 12.0
    min_mouse_area_ratio: float = 0.005
    max_mouse_area_ratio: float = 0.65
    static_rigid_threshold: float = 0.003
    static_rotation_threshold: float = 0.003
    static_residual_threshold: float = 0.007
    active_residual_threshold: float = 0.014
    active_deformation_threshold: float = 0.06
    active_injection_threshold: float = 0.030
    active_angular_injection_threshold: float = 0.018
    passive_coherence_threshold: float = 0.65
    passive_pendulum_score_threshold: float = 0.65
    min_affine_inlier_ratio: float = 0.35
    temporal_window_seconds: float = 2.5
    passive_min_observation_seconds: float = 0.75
    passive_min_reversals: int = 1
    analysis_width: int = DEFAULT_ANALYSIS_WIDTH

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.segmentation_contrast < 0:
            raise ValueError("segmentation_contrast cannot be negative")
        if not 0 < self.min_mouse_area_ratio < self.max_mouse_area_ratio <= 1:
            raise ValueError("invalid mouse area ratio limits")
        for name, value in self.__dict__.items():
            if name.endswith(("threshold", "ratio")) and value < 0:
                raise ValueError(f"{name} cannot be negative")
        for name in (
            "passive_coherence_threshold",
            "passive_pendulum_score_threshold",
            "min_affine_inlier_ratio",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.static_residual_threshold >= self.active_residual_threshold:
            raise ValueError(
                "static_residual_threshold must be below "
                "active_residual_threshold"
            )
        if self.temporal_window_seconds <= 0:
            raise ValueError("temporal_window_seconds must be positive")
        if not (
            0 < self.passive_min_observation_seconds
            <= self.temporal_window_seconds
        ):
            raise ValueError(
                "passive_min_observation_seconds must be positive and no "
                "longer than temporal_window_seconds"
            )
        if self.passive_min_reversals < 1:
            raise ValueError("passive_min_reversals must be at least 1")
        if not isinstance(self.passive_min_reversals, int):
            raise ValueError("passive_min_reversals must be an integer")
        if (
            not isinstance(self.analysis_width, int)
            or self.analysis_width < MIN_ANALYSIS_WIDTH
        ):
            raise ValueError(
                "analysis_width must be an integer of at least "
                f"{MIN_ANALYSIS_WIDTH}"
            )


@dataclass
class TSTMotionObservation:
    state: str
    confidence: float
    activity_score: float
    rigid_motion: float
    rotation_motion: float
    rotation_radians: float
    residual_motion: float
    deformation: float
    coherence: float
    motion_injection: float
    angular_injection: float
    pendulum_score: float
    temporal_ready: bool
    mask_area_ratio: float
    affine_inlier_ratio: float
    rigid_dx: float = 0.0
    rigid_dy: float = 0.0
    camera_stabilized: bool = False
    qc_flag: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class _RoiGeometry:
    """Per-ROI constants of the cropped analysis space, built once."""

    mask: np.ndarray  # uint8 0/255 ROI mask in crop space
    selector: np.ndarray  # bool, ``mask > 0``
    area: int
    roi_index: np.ndarray  # flat raster indices of the ROI pixels
    background_index: np.ndarray  # flat indices used to estimate background


@dataclass
class _MouseHistory:
    centroid: Optional[Tuple[float, float]] = None
    rigid_vector: Optional[np.ndarray] = None
    angular_velocity: Optional[float] = None
    centroid_trace: list[Tuple[float, float]] = field(default_factory=list)
    rotation_delta_trace: list[float] = field(default_factory=list)
    coherence_trace: list[float] = field(default_factory=list)
    residual_trace: list[float] = field(default_factory=list)
    deformation_trace: list[float] = field(default_factory=list)

    def reset_temporal(self) -> None:
        """Forget velocities and pendulum traces after an unscorable frame.

        The centroid is kept only as a proximity hint for choosing the mouse
        blob; it never feeds a motion feature.
        """
        self.rigid_vector = None
        self.angular_velocity = None
        self.centroid_trace.clear()
        self.rotation_delta_trace.clear()
        self.coherence_trace.clear()
        self.residual_trace.clear()
        self.deformation_trace.clear()


# Observations whose motion estimate is missing or untrustworthy. The next
# valid frame must not difference against, or window across, such a gap.
_GAP_QC_FLAGS = frozenset({
    "missing_frame",
    "empty_roi",
    "mouse_segmentation_failed",
    "insufficient_mouse_pixels",
    "rigid_fit_failed",
    "weak_rigid_fit",
})


class TSTMotionEngine:
    """Extract per-mouse articulated and rigid-body motion evidence."""

    ENGINE_ID = "tst_motion_decomposition_v3"

    def __init__(
        self,
        rois: Dict[str, ROI],
        config: Optional[TSTMotionConfig] = None,
        fps: float = 30.0,
        analysis_width: Optional[int] = None,
    ):
        self.rois = rois
        self.config = config or TSTMotionConfig()
        if analysis_width is not None:
            self.config = replace(self.config, analysis_width=analysis_width)
        self.config.validate()
        self.analysis_width = int(self.config.analysis_width)
        if not isfinite(float(fps)) or fps <= 0:
            raise ValueError("fps must be a positive finite value")
        self.fps = float(fps)
        # Displacement-type features (per-frame translation, rotation,
        # residual and shape change) scale linearly with the frame rate;
        # injections are differences of per-frame velocities (roughly
        # accelerations), so they scale with its square.
        self.rate_scale = self.fps / REFERENCE_FPS
        self.injection_scale = self.rate_scale ** 2
        # Geometry is fixed on the first frame. ROI masks and every image
        # operation live in the cropped analysis space; contours are mapped
        # back to source pixels.
        self.masks: Dict[str, np.ndarray] = {}
        self._geometry: Dict[str, _RoiGeometry] = {}
        # Integer source->analysis downscale factors (kx, ky) when the
        # resize can be done on the ROI crop alone without changing a pixel.
        self._exact_crop_factors: Optional[Tuple[int, int]] = None
        # Gray analysis image of the most recent "current" frame, reused as
        # the next pair's "previous" image. Keyed by frame identity.
        self._gray_cache: Optional[Tuple[weakref.ref, np.ndarray]] = None
        self._source_size: Tuple[int, int] = (0, 0)
        self._analysis_size: Tuple[int, int] = (0, 0)
        self._scale: Tuple[float, float] = (1.0, 1.0)
        self._crop: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._background_feature_mask: Optional[np.ndarray] = None
        # Current segmented mouse masks are retained only for visual quality
        # control. They are deliberately not used as persistent model state.
        self.last_mouse_masks: Dict[str, Optional[np.ndarray]] = {
            name: None for name in rois
        }
        self.histories = {name: _MouseHistory() for name in rois}
        self.last_observations: Dict[str, TSTMotionObservation] = {}
        self._initialized = False

    def _init_geometry(self, shape: Tuple[int, ...]) -> None:
        """Fix the analysis scale, ROI crop box and cropped ROI masks."""
        height, width = shape[:2]
        ratio = min(1.0, self.analysis_width / max(1, width))
        analysis_width = max(1, int(round(width * ratio)))
        analysis_height = max(1, int(round(height * ratio)))
        self._source_size = (width, height)
        self._analysis_size = (analysis_width, analysis_height)
        self._scale = (analysis_width / width, analysis_height / height)

        scaled: Dict[str, np.ndarray] = {}
        for name, roi in self.rois.items():
            points = roi.points.astype(np.float64) * np.asarray(self._scale)
            scaled[name] = np.rint(points).astype(np.int32)
        if scaled:
            stacked = np.vstack(list(scaled.values()))
            x_min = max(0, int(stacked[:, 0].min()))
            y_min = max(0, int(stacked[:, 1].min()))
            x_max = min(analysis_width, int(stacked[:, 0].max()) + 1)
            y_max = min(analysis_height, int(stacked[:, 1].max()) + 1)
        else:
            x_min, y_min = 0, 0
            x_max, y_max = analysis_width, analysis_height
        margin_x = int(ceil(ROI_MARGIN_FRACTION * max(0, x_max - x_min)))
        margin_y = int(ceil(ROI_MARGIN_FRACTION * max(0, y_max - y_min)))
        x0 = max(0, x_min - margin_x)
        y0 = max(0, y_min - margin_y)
        x1 = min(analysis_width, max(x_max + margin_x, x0 + 1))
        y1 = min(analysis_height, max(y_max + margin_y, y0 + 1))
        self._crop = (x0, y0, x1, y1)

        crop_shape = (y1 - y0, x1 - x0)
        combined = np.zeros(crop_shape, dtype=np.uint8)
        for name, points in scaled.items():
            mask = np.zeros(crop_shape, dtype=np.uint8)
            cv2.fillPoly(mask, [points - np.asarray([x0, y0], np.int32)], 255)
            self.masks[name] = mask
            self._geometry[name] = self._build_roi_geometry(mask)
            combined = cv2.bitwise_or(combined, mask)
        # Camera-motion features come from background inside the crop box
        # but outside every ROI.
        self._background_feature_mask = cv2.bitwise_not(combined)
        factor_x, factor_y = width // analysis_width, height // analysis_height
        if (
            factor_x >= 2
            and factor_y >= 2
            and width == factor_x * analysis_width
            and height == factor_y * analysis_height
        ):
            # INTER_AREA with integer factors averages disjoint source
            # blocks, so resizing just the crop's source rows/columns is
            # pixel-identical to cropping the resized frame.
            self._exact_crop_factors = (factor_x, factor_y)
        self._initialized = True

    @staticmethod
    def _build_roi_geometry(mask: np.ndarray) -> _RoiGeometry:
        selector = mask > 0
        roi_index = np.flatnonzero(selector)
        inner = cv2.erode(mask, _ERODE_KERNEL, iterations=2)
        border_index = np.flatnonzero(cv2.subtract(mask, inner) > 0)
        return _RoiGeometry(
            mask=mask,
            selector=selector,
            area=int(roi_index.size),
            roi_index=roi_index,
            background_index=(
                border_index
                if border_index.size >= _MIN_BACKGROUND_PIXELS
                else roi_index
            ),
        )

    def _analysis_gray(self, frame: np.ndarray) -> np.ndarray:
        """Resize to the analysis width, crop to the ROI box, blur to gray."""
        x0, y0, x1, y1 = self._crop
        if (frame.shape[1], frame.shape[0]) == self._analysis_size:
            return self._gray(frame[y0:y1, x0:x1])
        if self._exact_crop_factors is not None:
            factor_x, factor_y = self._exact_crop_factors
            region = frame[
                y0 * factor_y:y1 * factor_y, x0 * factor_x:x1 * factor_x
            ]
            return self._gray(cv2.resize(
                region, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA
            ))
        frame = cv2.resize(
            frame, self._analysis_size, interpolation=cv2.INTER_AREA
        )
        return self._gray(frame[y0:y1, x0:x1])

    def _previous_gray(self, previous_frame: np.ndarray) -> np.ndarray:
        """Analysis image of ``previous_frame``, reused when it is cached."""
        cached = self._gray_cache
        if cached is not None and cached[0]() is previous_frame:
            return cached[1]
        return self._analysis_gray(previous_frame)

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    @staticmethod
    def _stabilize_previous(
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        feature_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool]:
        points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=250,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=5,
            mask=feature_mask,
        )
        if points is None or len(points) < 12:
            return previous_gray, False
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, points, None
        )
        if next_points is None or status is None:
            return previous_gray, False
        valid = status.reshape(-1).astype(bool)
        source = points.reshape(-1, 2)[valid]
        target = next_points.reshape(-1, 2)[valid]
        if len(source) < 12:
            return previous_gray, False
        transform, inliers = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
        )
        if transform is None:
            return previous_gray, False
        stabilized = cv2.warpAffine(
            previous_gray,
            transform,
            (current_gray.shape[1], current_gray.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return stabilized, True

    def _segment_mouse(
        self,
        gray: np.ndarray,
        geometry: _RoiGeometry,
        previous_centroid: Optional[Tuple[float, float]],
    ) -> Tuple[Optional[np.ndarray], float, str]:
        roi_area = geometry.area
        if roi_area == 0:
            return None, 0.0, "empty_roi"

        background_level = float(np.median(gray.take(geometry.background_index)))
        contrast = cv2.absdiff(
            gray, np.full_like(gray, int(round(background_level)))
        )
        roi_contrast = contrast.take(geometry.roi_index)
        otsu_threshold, _ = cv2.threshold(
            roi_contrast.reshape(-1, 1),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        threshold = max(self.config.segmentation_contrast, otsu_threshold)
        candidate = np.zeros_like(gray, dtype=np.uint8)
        candidate[(contrast >= threshold) & geometry.selector] = 255
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, _OPEN_KERNEL)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, _CLOSE_KERNEL)

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            candidate, connectivity=8
        )
        options = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            area_ratio = area / roi_area
            if not (
                self.config.min_mouse_area_ratio
                <= area_ratio
                <= self.config.max_mouse_area_ratio
            ):
                continue
            centroid = centroids[label]
            if previous_centroid is None:
                proximity = 0.0
            else:
                proximity = float(np.linalg.norm(
                    centroid - np.asarray(previous_centroid)
                ))
            score = area - (0.25 * proximity)
            options.append((score, label, area_ratio))

        if not options:
            return None, 0.0, "mouse_segmentation_failed"
        _, selected, area_ratio = max(options)
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[labels == selected] = 255
        return mask, float(area_ratio), ""

    @staticmethod
    def _centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
        moments = cv2.moments(mask, binaryImage=True)
        if moments["m00"] <= 0:
            return None
        return (
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        )

    def _decompose(
        self,
        flow: np.ndarray,
        source_gray: np.ndarray,
        target_gray: np.ndarray,
        previous_mask: np.ndarray,
        current_mask: np.ndarray,
        history: _MouseHistory,
        area_ratio: float,
        camera_stabilized: bool,
    ) -> TSTMotionObservation:
        ys, xs = np.nonzero(previous_mask)
        if len(xs) < 25:
            return self._uncertain(area_ratio, "insufficient_mouse_pixels")

        stride = max(1, len(xs) // 800)
        sample_points = np.column_stack(
            (xs[::stride], ys[::stride])
        ).astype(np.float32)
        sampled_flow = flow[ys[::stride], xs[::stride]]

        feature_points = cv2.goodFeaturesToTrack(
            source_gray,
            maxCorners=120,
            qualityLevel=0.005,
            minDistance=4,
            mask=previous_mask,
            blockSize=3,
        )
        source = sample_points
        target = source + sampled_flow
        if feature_points is not None and len(feature_points) >= 8:
            tracked, status, _ = cv2.calcOpticalFlowPyrLK(
                source_gray,
                target_gray,
                feature_points,
                None,
            )
            if tracked is not None and status is not None:
                valid = status.reshape(-1).astype(bool)
                if np.count_nonzero(valid) >= 8:
                    source = feature_points.reshape(-1, 2)[valid]
                    target = tracked.reshape(-1, 2)[valid]
        transform, inliers = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=1.5,
        )
        if transform is None or inliers is None:
            return self._uncertain(area_ratio, "rigid_fit_failed")

        inlier_ratio = float(np.mean(inliers))
        linear = transform[:, :2]
        offset = transform[:, 2]
        predicted = (
            sample_points @ linear.T + offset - sample_points
        )
        residual = np.linalg.norm(sampled_flow - predicted, axis=1)
        total = np.linalg.norm(sampled_flow, axis=1)

        body_scale = max(1.0, float(np.sqrt(np.count_nonzero(current_mask))))
        rate = self.rate_scale
        rigid_vector = np.median(predicted, axis=0)
        translation_motion = float(
            np.linalg.norm(rigid_vector) / body_scale * rate
        )
        rotation_radians = float(np.arctan2(linear[1, 0], linear[0, 0]))
        rotation_motion = abs(rotation_radians) * 0.5 * rate
        rigid_motion = float(np.hypot(
            translation_motion,
            rotation_motion,
        ))
        residual_motion = float(
            np.percentile(residual, 90) / body_scale * rate
        )
        coherence = float(np.clip(
            1.0 - np.median(residual) / (np.median(total) + 1e-6),
            0.0,
            1.0,
        ))

        warped_mask = cv2.warpAffine(
            previous_mask,
            transform,
            (current_mask.shape[1], current_mask.shape[0]),
            flags=cv2.INTER_NEAREST,
        )
        union = np.count_nonzero((warped_mask > 0) | (current_mask > 0))
        intersection = np.count_nonzero(
            (warped_mask > 0) & (current_mask > 0)
        )
        # IoU loss is a per-frame shape change, so it is treated as a
        # displacement-type feature and rate-normalized like residual motion.
        deformation = (
            1.0 - intersection / union if union > 0 else 1.0
        ) * rate

        if history.rigid_vector is None:
            injection = 0.0
        else:
            injection = float(
                np.linalg.norm(rigid_vector - history.rigid_vector)
                / body_scale * self.injection_scale
            )
        if history.angular_velocity is None:
            angular_injection = 0.0
        else:
            angular_injection = abs(
                rotation_radians - history.angular_velocity
            ) * 0.5 * self.injection_scale
        history.rigid_vector = rigid_vector
        history.angular_velocity = rotation_radians
        current_centroid = self._centroid(current_mask)
        history.centroid = current_centroid
        pendulum_score, temporal_ready = self._update_pendulum_evidence(
            history,
            current_centroid,
            rotation_radians,
            coherence,
            residual_motion,
            float(deformation),
            body_scale,
        )

        return self._classify(
            rigid_motion=rigid_motion,
            rotation_motion=rotation_motion,
            rotation_radians=rotation_radians * rate,
            residual_motion=residual_motion,
            deformation=float(deformation),
            coherence=coherence,
            injection=injection,
            angular_injection=angular_injection,
            pendulum_score=pendulum_score,
            temporal_ready=temporal_ready,
            area_ratio=area_ratio,
            inlier_ratio=inlier_ratio,
            rigid_vector=rigid_vector * rate,
            camera_stabilized=camera_stabilized,
        )

    def _update_pendulum_evidence(
        self,
        history: _MouseHistory,
        centroid: Optional[Tuple[float, float]],
        rotation_delta: float,
        coherence: float,
        residual_motion: float,
        deformation: float,
        body_scale: float,
    ) -> Tuple[float, bool]:
        """Estimate passive pendulum evidence from a bounded temporal window.

        ``rotation_delta`` is the raw per-frame rotation (its cumulative sum
        is a rate-invariant angle); ``residual_motion`` and ``deformation``
        are already rate-normalized.
        """
        if centroid is None:
            return 0.0, False

        history.centroid_trace.append(centroid)
        history.rotation_delta_trace.append(float(rotation_delta))
        history.coherence_trace.append(float(coherence))
        history.residual_trace.append(float(residual_motion))
        history.deformation_trace.append(float(deformation))

        max_samples = max(
            3, int(ceil(self.config.temporal_window_seconds * self.fps)) + 1
        )
        for trace in (
            history.centroid_trace,
            history.rotation_delta_trace,
            history.coherence_trace,
            history.residual_trace,
            history.deformation_trace,
        ):
            if len(trace) > max_samples:
                del trace[:-max_samples]

        min_samples = max(
            4,
            int(ceil(
                self.config.passive_min_observation_seconds * self.fps
            )),
        )
        if len(history.centroid_trace) < min_samples:
            return 0.0, False

        centroids = np.asarray(history.centroid_trace, dtype=np.float64)
        centered = centroids - np.mean(centroids, axis=0)
        covariance = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        principal_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        projected_centroid = centered @ principal_axis
        total_variance = float(np.sum(np.maximum(eigenvalues, 0.0)))
        axis_ratio = (
            float(np.max(eigenvalues) / total_variance)
            if total_variance > 1e-9 else 0.0
        )
        centroid_range = float(np.ptp(projected_centroid)) / body_scale

        rotation_position = np.cumsum(np.asarray(
            history.rotation_delta_trace,
            dtype=np.float64,
        ))
        rotation_position -= float(np.mean(rotation_position))
        rotation_range = float(np.ptp(rotation_position)) * 0.5

        if rotation_range > centroid_range:
            signal = rotation_position
            axis_score = 1.0
            amplitude = rotation_range
        else:
            signal = projected_centroid
            axis_score = float(np.clip(
                (axis_ratio - 0.60) / 0.40,
                0.0,
                1.0,
            ))
            amplitude = centroid_range

        # Ranges and amplitudes of position (centroid, rotation) are already
        # rate-invariant. The per-frame velocity floor is not, so it is
        # compared at the reference frame rate.
        velocity = np.diff(signal) * self.rate_scale
        nonzero = velocity[np.abs(velocity) > 1e-4]
        reversals = 0
        if len(nonzero) >= 2:
            reversals = int(np.count_nonzero(
                np.sign(nonzero[1:]) != np.sign(nonzero[:-1])
            ))
        reversal_score = float(np.clip(
            reversals / max(1, self.config.passive_min_reversals + 1),
            0.0,
            1.0,
        ))

        periodicity = 0.0
        normalized = signal - float(np.mean(signal))
        denominator = float(np.dot(normalized, normalized))
        min_lag = max(2, int(round(0.25 * self.fps)))
        max_lag = min(
            len(normalized) - 2,
            int(round(1.5 * self.fps)),
        )
        if denominator > 1e-9 and max_lag >= min_lag:
            correlations = []
            for lag in range(min_lag, max_lag + 1):
                left = normalized[:-lag]
                right = normalized[lag:]
                norm = float(np.linalg.norm(left) * np.linalg.norm(right))
                if norm > 1e-9:
                    correlations.append(abs(float(np.dot(left, right) / norm)))
            if correlations:
                periodicity = max(correlations)

        mean_coherence = float(np.mean(history.coherence_trace))
        coherence_score = float(np.clip(
            mean_coherence
            / max(self.config.passive_coherence_threshold, 1e-6),
            0.0,
            1.0,
        ))
        residual_cleanliness = 1.0 - float(np.clip(
            np.mean(history.residual_trace)
            / max(self.config.active_residual_threshold, 1e-6),
            0.0,
            1.0,
        ))
        deformation_cleanliness = 1.0 - float(np.clip(
            np.mean(history.deformation_trace)
            / max(self.config.active_deformation_threshold, 1e-6),
            0.0,
            1.0,
        ))
        amplitude_score = float(np.clip(
            amplitude / max(self.config.static_rigid_threshold * 4.0, 1e-6),
            0.0,
            1.0,
        ))

        if reversals < self.config.passive_min_reversals:
            return min(
                0.49,
                0.25 * axis_score
                + 0.15 * periodicity
                + 0.10 * coherence_score,
            ), True

        score = (
            0.15 * axis_score
            + 0.20 * reversal_score
            + 0.20 * periodicity
            + 0.15 * coherence_score
            + 0.10 * residual_cleanliness
            + 0.10 * deformation_cleanliness
            + 0.10 * amplitude_score
        )
        return float(np.clip(score, 0.0, 1.0)), True

    def _classify(
        self,
        *,
        rigid_motion: float,
        rotation_motion: float = 0.0,
        rotation_radians: float = 0.0,
        residual_motion: float,
        deformation: float,
        coherence: float,
        injection: float,
        angular_injection: float = 0.0,
        pendulum_score: float = 0.0,
        temporal_ready: bool = False,
        area_ratio: float,
        inlier_ratio: float,
        rigid_vector: Optional[np.ndarray] = None,
        camera_stabilized: bool = False,
    ) -> TSTMotionObservation:
        cfg = self.config
        residual_evidence = residual_motion / max(
            cfg.active_residual_threshold, 1e-6
        )
        deformation_evidence = deformation / max(
            cfg.active_deformation_threshold, 1e-6
        )
        injection_evidence = injection / max(
            cfg.active_injection_threshold, 1e-6
        )
        angular_injection_evidence = angular_injection / max(
            cfg.active_angular_injection_threshold, 1e-6
        )
        activity_score = float(np.clip(
            max(
                residual_evidence,
                deformation_evidence,
                injection_evidence,
                angular_injection_evidence,
            )
            * 50.0,
            0.0,
            100.0,
        ))

        qc_flag = ""
        if inlier_ratio < cfg.min_affine_inlier_ratio:
            state = "uncertain"
            qc_flag = "weak_rigid_fit"
            confidence = max(0.05, inlier_ratio)
        elif (
            residual_motion >= cfg.active_residual_threshold
            or deformation >= cfg.active_deformation_threshold
        ):
            state = "active_escape"
            confidence = 0.55 + 0.45 * np.clip(
                max(
                    residual_evidence,
                    deformation_evidence,
                ) - 1.0,
                0.0,
                1.0,
            )
        elif (
            rigid_motion <= cfg.static_rigid_threshold
            and rotation_motion <= cfg.static_rotation_threshold
            and residual_motion <= cfg.static_residual_threshold
            and deformation < cfg.active_deformation_threshold
            and injection < cfg.active_injection_threshold
            and angular_injection < cfg.active_angular_injection_threshold
        ):
            state = "immobile_static"
            margin = 1.0 - max(
                rigid_motion / max(cfg.static_rigid_threshold, 1e-6),
                rotation_motion / max(cfg.static_rotation_threshold, 1e-6),
                residual_motion / max(cfg.static_residual_threshold, 1e-6),
                injection / max(cfg.active_injection_threshold, 1e-6),
                angular_injection
                / max(cfg.active_angular_injection_threshold, 1e-6),
            )
            confidence = 0.65 + 0.35 * np.clip(margin, 0.0, 1.0)
        elif (
            temporal_ready
            and pendulum_score >= cfg.passive_pendulum_score_threshold
        ):
            state = "passive_swing"
            confidence = 0.55 + 0.45 * np.clip(
                (
                    pendulum_score
                    - cfg.passive_pendulum_score_threshold
                )
                / max(
                    1e-6,
                    1.0 - cfg.passive_pendulum_score_threshold,
                ),
                0.0,
                1.0,
            )
        elif (
            injection >= cfg.active_injection_threshold
            or angular_injection >= cfg.active_angular_injection_threshold
        ):
            state = "active_escape"
            confidence = 0.55 + 0.45 * np.clip(
                max(
                    injection_evidence,
                    angular_injection_evidence,
                ) - 1.0,
                0.0,
                1.0,
            )
        else:
            state = "uncertain"
            qc_flag = (
                "insufficient_temporal_context"
                if not temporal_ready and rigid_motion > cfg.static_rigid_threshold
                else "rigid_motion_without_passive_pattern"
            )
            confidence = 0.25

        return TSTMotionObservation(
            state=state,
            confidence=float(confidence),
            activity_score=activity_score,
            rigid_motion=rigid_motion,
            rotation_motion=rotation_motion,
            rotation_radians=rotation_radians,
            residual_motion=residual_motion,
            deformation=deformation,
            coherence=coherence,
            motion_injection=injection,
            angular_injection=angular_injection,
            pendulum_score=pendulum_score,
            temporal_ready=temporal_ready,
            mask_area_ratio=area_ratio,
            affine_inlier_ratio=inlier_ratio,
            rigid_dx=(
                float(rigid_vector[0]) if rigid_vector is not None else 0.0
            ),
            rigid_dy=(
                float(rigid_vector[1]) if rigid_vector is not None else 0.0
            ),
            camera_stabilized=camera_stabilized,
            qc_flag=qc_flag,
        )

    @staticmethod
    def _uncertain(
        area_ratio: float,
        reason: str,
    ) -> TSTMotionObservation:
        return TSTMotionObservation(
            state="uncertain",
            confidence=0.0,
            activity_score=0.0,
            rigid_motion=0.0,
            rotation_motion=0.0,
            rotation_radians=0.0,
            residual_motion=0.0,
            deformation=0.0,
            coherence=0.0,
            motion_injection=0.0,
            angular_injection=0.0,
            pendulum_score=0.0,
            temporal_ready=False,
            mask_area_ratio=area_ratio,
            affine_inlier_ratio=0.0,
            rigid_dx=0.0,
            rigid_dy=0.0,
            camera_stabilized=False,
            qc_flag=reason,
        )

    def calculate_observations(
        self,
        frame: np.ndarray,
        previous_frame: np.ndarray,
    ) -> Dict[str, TSTMotionObservation]:
        if frame is None or previous_frame is None:
            observations = {
                name: self._uncertain(0.0, "missing_frame")
                for name in self.rois
            }
            for history in self.histories.values():
                history.reset_temporal()
            self._gray_cache = None
            self.last_observations = observations
            return observations
        if not self._initialized:
            self._init_geometry(frame.shape)
        elif (frame.shape[1], frame.shape[0]) != self._source_size:
            raise ValueError(
                "TST frame size changed during analysis: expected "
                f"{self._source_size}, got "
                f"{(frame.shape[1], frame.shape[0])}"
            )

        current_gray = self._analysis_gray(frame)
        previous_gray = self._previous_gray(previous_frame)
        self._gray_cache = (weakref.ref(frame), current_gray)
        aligned_previous, camera_stabilized = self._stabilize_previous(
            previous_gray, current_gray, self._background_feature_mask
        )
        flow = cv2.calcOpticalFlowFarneback(
            aligned_previous,
            current_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        observations: Dict[str, TSTMotionObservation] = {}
        for name, geometry in self._geometry.items():
            history = self.histories[name]
            previous_mask, _, previous_error = self._segment_mouse(
                aligned_previous, geometry, history.centroid
            )
            current_mask, area_ratio, current_error = self._segment_mouse(
                current_gray, geometry, history.centroid
            )
            if previous_mask is None or current_mask is None:
                self.last_mouse_masks[name] = None
                observations[name] = self._uncertain(
                    area_ratio, current_error or previous_error
                )
                history.reset_temporal()
                continue
            self.last_mouse_masks[name] = current_mask.copy()
            observations[name] = self._decompose(
                flow,
                aligned_previous,
                current_gray,
                previous_mask,
                current_mask,
                history,
                area_ratio,
                camera_stabilized,
            )
            if observations[name].qc_flag in _GAP_QC_FLAGS:
                history.reset_temporal()

        self.last_observations = observations
        return observations

    def get_last_mask_contour(self, name: str) -> list[list[int]]:
        """Return a compact contour for the latest segmentation of ``name``.

        The contour supports preview/validation overlays without retaining a
        full raster mask for every processed frame. Points are in SOURCE
        pixel coordinates, whatever the internal analysis scale and crop.
        """
        mask = self.last_mouse_masks.get(name)
        if mask is None:
            return []
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return []
        contour = max(contours, key=cv2.contourArea)
        epsilon = max(1.0, 0.01 * cv2.arcLength(contour, True))
        simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        x0, y0, _, _ = self._crop
        scale_x, scale_y = self._scale
        return [
            [
                int(round((point[0] + x0) / scale_x)),
                int(round((point[1] + y0) / scale_y)),
            ]
            for point in simplified
        ]

    def export_state(self) -> Dict[str, object]:
        """Serialize temporal state for checkpoint-equivalent resume."""
        histories: Dict[str, object] = {}
        for name, history in self.histories.items():
            histories[name] = {
                "centroid": (
                    list(history.centroid)
                    if history.centroid is not None else None
                ),
                "rigid_vector": (
                    history.rigid_vector.tolist()
                    if history.rigid_vector is not None else None
                ),
                "angular_velocity": history.angular_velocity,
                "centroid_trace": [
                    list(value) for value in history.centroid_trace
                ],
                "rotation_delta_trace": list(history.rotation_delta_trace),
                "coherence_trace": list(history.coherence_trace),
                "residual_trace": list(history.residual_trace),
                "deformation_trace": list(history.deformation_trace),
            }
        return {
            "engine_id": self.ENGINE_ID,
            "fps": self.fps,
            "histories": histories,
        }

    def restore_state(self, state: Optional[Dict[str, object]]) -> None:
        """Restore state created by :meth:`export_state`."""
        if not state:
            return
        if state.get("engine_id") != self.ENGINE_ID:
            raise ValueError("TST checkpoint engine does not match current engine")
        histories = state.get("histories", {})
        if not isinstance(histories, dict):
            raise ValueError("Invalid TST checkpoint histories")
        for name, raw in histories.items():
            if name not in self.histories or not isinstance(raw, dict):
                continue
            history = self.histories[name]
            centroid = raw.get("centroid")
            history.centroid = (
                (float(centroid[0]), float(centroid[1]))
                if centroid is not None else None
            )
            rigid_vector = raw.get("rigid_vector")
            history.rigid_vector = (
                np.asarray(rigid_vector, dtype=np.float64)
                if rigid_vector is not None else None
            )
            angular_velocity = raw.get("angular_velocity")
            history.angular_velocity = (
                float(angular_velocity)
                if angular_velocity is not None else None
            )
            history.centroid_trace = [
                (float(value[0]), float(value[1]))
                for value in raw.get("centroid_trace", [])
            ]
            history.rotation_delta_trace = [
                float(value)
                for value in raw.get("rotation_delta_trace", [])
            ]
            history.coherence_trace = [
                float(value)
                for value in raw.get("coherence_trace", [])
            ]
            history.residual_trace = [
                float(value)
                for value in raw.get("residual_trace", [])
            ]
            history.deformation_trace = [
                float(value)
                for value in raw.get("deformation_trace", [])
            ]
