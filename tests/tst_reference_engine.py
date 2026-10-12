"""Verbatim v3.2.0 TST motion pipeline, kept as an exactness oracle.

The production engine was restructured for throughput (cached preprocessing,
hoisted constants, crop-before-resize, a pipelined flow stage). Every such
change must leave the observations identical, so tests compare the production
engine against this unoptimized implementation: full-frame resize, a gray
conversion for every frame of every pair, per-call kernels and masks, and
strictly serial per-mouse segmentation and decomposition.

Only the code that was changed is duplicated here; the unchanged helpers
(``_classify``, ``_update_pendulum_evidence``, ``_stabilize_previous``, ...)
are inherited from the production engine.
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from core.tst_motion_engine import (
    _GAP_QC_FLAGS, TSTMotionEngine, TSTMotionObservation, _MouseHistory,
)


class ReferenceTSTMotionEngine(TSTMotionEngine):
    """The unoptimized v3.2.0 algorithm, with the production state layout."""

    def _analysis_gray(self, frame: np.ndarray) -> np.ndarray:
        """Resize to the analysis width, crop to the ROI box, blur to gray."""
        if (frame.shape[1], frame.shape[0]) != self._analysis_size:
            frame = cv2.resize(
                frame, self._analysis_size, interpolation=cv2.INTER_AREA
            )
        x0, y0, x1, y1 = self._crop
        return self._gray(frame[y0:y1, x0:x1])

    def _reference_segment_mouse(
        self,
        gray: np.ndarray,
        roi_mask: np.ndarray,
        previous_centroid: Optional[Tuple[float, float]],
    ) -> Tuple[Optional[np.ndarray], float, str]:
        roi_area = int(np.count_nonzero(roi_mask))
        if roi_area == 0:
            return None, 0.0, "empty_roi"

        kernel = np.ones((7, 7), dtype=np.uint8)
        inner = cv2.erode(roi_mask, kernel, iterations=2)
        border = cv2.subtract(roi_mask, inner)
        background_pixels = gray[border > 0]
        if background_pixels.size < 20:
            background_pixels = gray[roi_mask > 0]
        background_level = float(np.median(background_pixels))
        contrast = cv2.absdiff(
            gray, np.full_like(gray, int(round(background_level)))
        )
        roi_contrast = contrast[roi_mask > 0]
        otsu_threshold, _ = cv2.threshold(
            roi_contrast.reshape(-1, 1),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        threshold = max(self.config.segmentation_contrast, otsu_threshold)
        candidate = np.zeros_like(gray, dtype=np.uint8)
        candidate[(contrast >= threshold) & (roi_mask > 0)] = 255
        candidate = cv2.morphologyEx(
            candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        candidate = cv2.morphologyEx(
            candidate, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
        )

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

    def _reference_decompose(
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
        previous_gray = self._analysis_gray(previous_frame)
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
        for name, roi_mask in self.masks.items():
            history = self.histories[name]
            previous_mask, _, previous_error = self._reference_segment_mouse(
                aligned_previous, roi_mask, history.centroid
            )
            current_mask, area_ratio, current_error = self._reference_segment_mouse(
                current_gray, roi_mask, history.centroid
            )
            if previous_mask is None or current_mask is None:
                self.last_mouse_masks[name] = None
                observations[name] = self._uncertain(
                    area_ratio, current_error or previous_error
                )
                history.reset_temporal()
                continue
            self.last_mouse_masks[name] = current_mask.copy()
            observations[name] = self._reference_decompose(
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
