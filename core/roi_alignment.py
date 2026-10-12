"""Pure helpers for carrying an ROI template to another camera placement.

An ROI template is drawn in the pixels of one video. When another video was
recorded with a slightly moved camera, or at another resolution, the same
polygons land in the wrong place. This module estimates the 2D transform
between two views of the same arena (from landmarks the user clicked, or
automatically from animal-free background frames) and applies it to a whole
``ROIManager``: zones, arena, reference line and calibration.

Everything here is Qt-free and works on plain arrays.

Conventions
-----------
* A transform ``M`` is a 3x3 matrix mapping TEMPLATE pixels to TARGET pixels
  (column vectors ``[x, y, 1]``), so ``M @ p_template ~ p_target``.
* Residuals are RMS point errors in TARGET pixels.
"""
import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from roi_manager import (APPROXIMATE_REFERENCE_SUFFIX, ROI, ROIManager)

logger = logging.getLogger(__name__)

MODE_SIMILARITY = "similarity"
MODE_AFFINE = "affine"
MODE_HOMOGRAPHY = "homography"
MODES = (MODE_SIMILARITY, MODE_AFFINE, MODE_HOMOGRAPHY)
MIN_POINTS = {MODE_SIMILARITY: 2, MODE_AFFINE: 3, MODE_HOMOGRAPHY: 4}

# Verdicts of an automatic registration.
VERDICT_GOOD = "good"
VERDICT_CHECK = "check"
VERDICT_POOR = "poor"
VERDICT_FAILED = "failed"

# Residual bands (target px) used for the human-readable quality hint.
RMS_GOOD_PX = 2.0
RMS_FAIR_PX = 4.0

# An alignment this far from identity (px) counts as "the camera moved".
SHIFT_FLAG_PX = 3.0
# Scale deviations below this are treated as "same scale".
SCALE_EPSILON = 1e-6

# ORB / matching settings for ``auto_register``.
ORB_FEATURES = 2000
ORB_WORK_SIZE = 960          # longest side the matcher works at
RATIO_TEST = 0.75
RANSAC_THRESHOLD_PX = 3.0    # in working pixels
MIN_INLIERS_GOOD = 30
MIN_INLIERS_CHECK = 12
MIN_INLIER_RATIO_GOOD = 0.3


@dataclass(frozen=True)
class RegistrationResult:
    """Outcome of ``auto_register``.

    ``matrix`` is None when no transform could be estimated at all.
    """
    matrix: Optional[np.ndarray]
    inliers: int
    matches: int
    rms: float
    verdict: str
    message: str = ""

    @property
    def usable(self) -> bool:
        return self.matrix is not None and self.verdict != VERDICT_FAILED


# ---------------------------------------------------------------------------
# Small matrix helpers
# ---------------------------------------------------------------------------

def identity() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def is_identity(matrix: np.ndarray, tolerance: float = 1e-9) -> bool:
    return bool(np.allclose(_normalise(matrix), np.eye(3), atol=tolerance))


def _normalise(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape == (2, 3):
        m = np.vstack([m, [0.0, 0.0, 1.0]])
    if m.shape != (3, 3) or not np.all(np.isfinite(m)):
        raise ValueError("A transform must be a finite 3x3 (or 2x3) matrix")
    if abs(m[2, 2]) > 1e-12 and m[2, 2] != 1.0:
        m = m / m[2, 2]
    return m


def _as_points(points: Sequence, name: str) -> np.ndarray:
    try:
        arr = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a list of (x, y) points: {exc}")
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must be an Nx2 list of points")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def transform_points(matrix: np.ndarray, points: Sequence) -> np.ndarray:
    """Apply ``matrix`` to Nx2 points; returns float64 Nx2."""
    m = _normalise(matrix)
    pts = _as_points(points, "points")
    if len(pts) == 0:
        return pts.copy()
    return cv2.perspectiveTransform(
        pts.reshape(-1, 1, 2), m
    ).reshape(-1, 2).astype(np.float64)


def _jacobian(matrix: np.ndarray, at: Tuple[float, float]) -> np.ndarray:
    """2x2 Jacobian of the (possibly projective) map at a template point."""
    m = _normalise(matrix)
    x, y = float(at[0]), float(at[1])
    w = m[2, 0] * x + m[2, 1] * y + m[2, 2]
    if abs(w) < 1e-12:
        raise ValueError("The transform is degenerate at this point")
    xp = (m[0, 0] * x + m[0, 1] * y + m[0, 2]) / w
    yp = (m[1, 0] * x + m[1, 1] * y + m[1, 2]) / w
    return np.array([
        [(m[0, 0] - m[2, 0] * xp) / w, (m[0, 1] - m[2, 1] * xp) / w],
        [(m[1, 0] - m[2, 0] * yp) / w, (m[1, 1] - m[2, 1] * yp) / w],
    ])


def mean_scale(matrix: np.ndarray, at: Tuple[float, float] = (0.0, 0.0)) -> float:
    """Local mean scale factor (square root of the area scale) at ``at``."""
    det = float(np.linalg.det(_jacobian(matrix, at)))
    return float(np.sqrt(abs(det)))


def is_similarity(matrix: np.ndarray, tolerance: float = 1e-6) -> bool:
    """True for rotation + uniform scale + shift (no shear or perspective)."""
    m = _normalise(matrix)
    if abs(m[2, 0]) > tolerance or abs(m[2, 1]) > tolerance:
        return False
    a = m[:2, :2]
    scale = np.sqrt(abs(np.linalg.det(a)))
    if scale <= 0:
        return False
    return bool(np.allclose(a @ a.T, scale ** 2 * np.eye(2),
                            atol=tolerance * max(1.0, scale ** 2)))


def decompose_similarity(matrix: np.ndarray) -> Tuple[float, float, float, float]:
    """(scale, rotation in degrees, tx, ty) of the affine part of ``matrix``."""
    m = _normalise(matrix)
    a = m[:2, :2]
    scale = float(np.sqrt(abs(np.linalg.det(a))))
    angle = float(np.degrees(np.arctan2(a[1, 0], a[0, 0])))
    return scale, angle, float(m[0, 2]), float(m[1, 2])


def rms_residual(matrix: np.ndarray, src_pts: Sequence, dst_pts: Sequence) -> float:
    """RMS distance (target px) between transformed src points and dst points."""
    src = _as_points(src_pts, "src_pts")
    dst = _as_points(dst_pts, "dst_pts")
    if len(src) != len(dst):
        raise ValueError("src_pts and dst_pts must have the same length")
    if len(src) == 0:
        return 0.0
    error = transform_points(matrix, src) - dst
    return float(np.sqrt(np.mean(np.sum(error ** 2, axis=1))))


# ---------------------------------------------------------------------------
# Transform estimation from point pairs
# ---------------------------------------------------------------------------

def _similarity_least_squares(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rotation + uniform scale + shift minimising the squared error."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s, d = src - mu_s, dst - mu_d
    variance = float(np.mean(np.sum(s ** 2, axis=1)))
    if variance <= 1e-12:
        raise ValueError("The landmark points coincide; pick distinct points")
    u, sing, vt = np.linalg.svd(d.T @ s / len(src))
    flip = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        flip[1, 1] = -1.0
    rotation = u @ flip @ vt
    scale = float(np.trace(np.diag(sing) @ flip) / variance)
    matrix = identity()
    matrix[:2, :2] = scale * rotation
    matrix[:2, 2] = mu_d - scale * rotation @ mu_s
    return matrix


def _check_not_collinear(src: np.ndarray, label: str) -> None:
    centred = src - src.mean(axis=0)
    sing = np.linalg.svd(centred, compute_uv=False)
    if len(sing) < 2 or sing[0] <= 1e-12 or sing[1] < 1e-3 * sing[0]:
        raise ValueError(
            f"The {label} landmarks lie on a line; spread them over the arena"
        )


def estimate_transform(
    src_pts: Sequence, dst_pts: Sequence, mode: str = MODE_SIMILARITY
) -> Tuple[np.ndarray, float]:
    """Fit a template -> target transform to matching points.

    ``similarity`` (rotation, uniform scale, shift) needs 2 or more points,
    ``affine`` 3 or more and ``homography`` 4 or more. All fits are least
    squares over every point given (landmarks are trusted, so no outlier
    rejection). Returns ``(3x3 matrix, RMS residual in target px)``.

    Raises:
        ValueError: For too few points, degenerate geometry or a mirrored fit.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown alignment mode '{mode}'")
    src = _as_points(src_pts, "src_pts")
    dst = _as_points(dst_pts, "dst_pts")
    if len(src) != len(dst):
        raise ValueError("src_pts and dst_pts must have the same length")
    if len(src) < MIN_POINTS[mode]:
        raise ValueError(
            f"{mode} alignment needs at least {MIN_POINTS[mode]} point pairs, "
            f"got {len(src)}"
        )

    if mode == MODE_SIMILARITY:
        matrix = _similarity_least_squares(src, dst)
    else:
        _check_not_collinear(src, "template")
        _check_not_collinear(dst, "target")
        if mode == MODE_AFFINE:
            if len(src) == 3:
                affine = cv2.getAffineTransform(
                    src.astype(np.float32), dst.astype(np.float32)
                ).astype(np.float64)
            else:
                design = np.hstack([src, np.ones((len(src), 1))])
                solved, *_ = np.linalg.lstsq(design, dst, rcond=None)
                affine = solved.T
            matrix = np.vstack([affine, [0.0, 0.0, 1.0]])
        else:
            homography, _ = cv2.findHomography(src, dst, 0)
            if homography is None:
                raise ValueError("Could not fit a homography to these points")
            matrix = homography.astype(np.float64)

    matrix = _normalise(matrix)
    centre = tuple(src.mean(axis=0))
    if float(np.linalg.det(_jacobian(matrix, centre))) <= 0:
        raise ValueError(
            "The landmarks produce a mirrored or collapsed alignment; check "
            "that the points were clicked in the same order on both frames"
        )
    return matrix, rms_residual(matrix, src, dst)


# ---------------------------------------------------------------------------
# Quality wording
# ---------------------------------------------------------------------------

def describe_residual(rms: float) -> Tuple[str, str]:
    """(level, text) for a residual, e.g. ("good", "good: 0.8 px")."""
    if rms <= RMS_GOOD_PX:
        level, label = VERDICT_GOOD, "good"
    elif rms <= RMS_FAIR_PX:
        level, label = VERDICT_CHECK, "fair, review the overlay"
    else:
        level, label = VERDICT_POOR, "check carefully"
    return level, f"{label}: {rms:.1f} px"


# ---------------------------------------------------------------------------
# Applying a transform to an ROIManager
# ---------------------------------------------------------------------------

def _rounded_polygon(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Transform polygon vertices and round to the nearest pixel.

    ``ROI`` stores int32 vertices (truncating casts); rounding first keeps
    the stored vertex within half a pixel of the exact position instead of
    biasing every point towards the origin.
    """
    moved = transform_points(matrix, points)
    return np.rint(moved).astype(np.int32)


def _template_centroid(manager: ROIManager) -> Tuple[float, float]:
    chunks = [roi.points for rois in manager.rois.values() for roi in rois]
    if manager.arena is not None:
        chunks.append(manager.arena.points)
    if not chunks:
        dims = manager.source_dimensions
        return (dims[0] / 2.0, dims[1] / 2.0) if dims else (0.0, 0.0)
    centre = np.vstack(chunks).astype(np.float64).mean(axis=0)
    return float(centre[0]), float(centre[1])


def transform_roi_manager(
    manager: ROIManager, matrix: np.ndarray, target_dims: Tuple[int, int]
) -> ROIManager:
    """Return a NEW manager with everything moved by ``matrix``.

    Zones and the arena are transformed (vertices rounded to pixels), the
    reference line follows and its length is recomputed so the calibration
    tracks the change of scale, and ``source_dimensions`` becomes
    ``target_dims``. A length without stored endpoints is scaled by the
    transform's mean scale factor and flagged as approximate. The input
    manager is never modified.
    """
    m = _normalise(matrix)
    moved = ROIManager(tuple(int(v) for v in target_dims))

    for category, rois in manager.rois.items():
        moved.rois[category] = [
            ROI(_rounded_polygon(m, roi.points), roi.category) for roi in rois
        ]
    if manager.arena is not None:
        moved.arena = ROI(
            _rounded_polygon(m, manager.arena.points), manager.arena.category
        )
    if manager.center_overlay is not None:
        moved.center_overlay = _rounded_polygon(m, manager.center_overlay)

    name = manager.reference_name
    line = manager.reference_line
    if line is not None:
        p1, p2 = transform_points(m, line)
        moved.set_reference_line(tuple(p1), tuple(p2), name)
    elif manager.reference_length_pixels > 0:
        scale = mean_scale(m, _template_centroid(manager))
        length = manager.reference_length_pixels
        if abs(scale - 1.0) > SCALE_EPSILON or not is_similarity(m):
            length = length * scale
            if not name.endswith(APPROXIMATE_REFERENCE_SUFFIX):
                name = name + APPROXIMATE_REFERENCE_SUFFIX
        moved.set_reference_length(length, name)
    else:
        moved._reference_name = manager.reference_name
    return moved


def scale_factor_after_alignment(
    scale_factor: float, old_manager: ROIManager, new_manager: ROIManager
) -> float:
    """cm-per-pixel for the aligned ROIs.

    The real-world length of the reference line is the invariant, so a
    reference that became ``k`` times longer in pixels makes a pixel ``k``
    times smaller. Without a calibration this is a no-op.
    """
    old_px = old_manager.reference_length_pixels
    new_px = new_manager.reference_length_pixels
    if scale_factor <= 0 or old_px <= 0 or new_px <= 0:
        return scale_factor
    return float(scale_factor) * old_px / new_px


# ---------------------------------------------------------------------------
# Automatic registration
# ---------------------------------------------------------------------------

def _to_gray(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Cannot register an empty image")
    if image.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        image = cv2.cvtColor(image, code)
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def _work_scale(gray: np.ndarray) -> float:
    longest = max(gray.shape[:2])
    return min(1.0, ORB_WORK_SIZE / float(longest))


def _verdict(inliers: int, matches: int, rms: float) -> str:
    ratio = inliers / matches if matches else 0.0
    if (
        inliers >= MIN_INLIERS_GOOD and ratio >= MIN_INLIER_RATIO_GOOD
        and rms <= RMS_GOOD_PX
    ):
        return VERDICT_GOOD
    if inliers >= MIN_INLIERS_CHECK and rms <= RMS_FAIR_PX * 2:
        return VERDICT_CHECK
    return VERDICT_POOR


def auto_register(ref_a_gray: np.ndarray, ref_b_gray: np.ndarray) -> RegistrationResult:
    """Find the similarity that maps frame A (template) onto frame B (target).

    Meant for animal-free background frames (the median reference frame), so
    the matched features are the arena itself. ORB features are matched with
    Lowe's ratio test and a similarity is fitted with RANSAC. Frames may have
    different resolutions: matching runs on a common working size and the
    result is expressed in the full-resolution pixels of each frame.

    Returns a ``RegistrationResult``; never raises for unmatchable images
    (``verdict`` is then ``"failed"`` and ``matrix`` is None).
    """
    gray_a = _to_gray(ref_a_gray)
    gray_b = _to_gray(ref_b_gray)
    scale_a, scale_b = _work_scale(gray_a), _work_scale(gray_b)

    def prepared(gray, scale):
        if scale != 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    work_a, work_b = prepared(gray_a, scale_a), prepared(gray_b, scale_b)

    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    kp_a, des_a = orb.detectAndCompute(work_a, None)
    kp_b, des_b = orb.detectAndCompute(work_b, None)
    if des_a is None or des_b is None or len(kp_a) < 3 or len(kp_b) < 3:
        return RegistrationResult(
            None, 0, 0, float("inf"), VERDICT_FAILED,
            "Too few image features to align (is the arena textured enough?)",
        )

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = []
    for pair in matcher.knnMatch(des_a, des_b, k=2):
        if len(pair) == 2 and pair[0].distance < RATIO_TEST * pair[1].distance:
            good.append(pair[0])
    if len(good) < 3:
        return RegistrationResult(
            None, 0, len(good), float("inf"), VERDICT_FAILED,
            "Too few matching features between the two frames",
        )

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])
    affine, mask = cv2.estimateAffinePartial2D(
        pts_a, pts_b, method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD_PX, maxIters=3000,
        confidence=0.995, refineIters=20,
    )
    if affine is None or mask is None:
        return RegistrationResult(
            None, 0, len(good), float("inf"), VERDICT_FAILED,
            "Could not fit an alignment to the matched features",
        )
    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())

    # Express the working-size transform in full-resolution pixels:
    # full_a -> work_a -> work_b -> full_b.
    to_work_a = np.diag([scale_a, scale_a, 1.0])
    from_work_b = np.diag([1.0 / scale_b, 1.0 / scale_b, 1.0])
    matrix = from_work_b @ _normalise(affine) @ to_work_a

    if inliers < 3:
        return RegistrationResult(
            None, inliers, len(good), float("inf"), VERDICT_FAILED,
            "No consistent alignment found between the two frames",
        )
    full_a = pts_a[inlier_mask].astype(np.float64) / scale_a
    full_b = pts_b[inlier_mask].astype(np.float64) / scale_b
    rms = rms_residual(matrix, full_a, full_b)
    verdict = _verdict(inliers, len(good), rms)
    return RegistrationResult(
        matrix, inliers, len(good), rms, verdict,
        f"{inliers} of {len(good)} matches agree",
    )


def displacement_px(matrix: np.ndarray, dims: Tuple[int, int]) -> float:
    """Largest corner displacement (target px) the transform causes on ``dims``.

    A cheap measure of how far the camera moved: 0 for the identity.
    """
    width, height = float(dims[0]), float(dims[1])
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]])
    return float(np.max(np.linalg.norm(transform_points(matrix, corners) - corners, axis=1)))
