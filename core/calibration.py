"""Pure helpers for distance calibration from a user-drawn reference line."""
from typing import Optional

NO_CALIBRATION_NOTE = (
    "No calibration line drawn — distances will be reported in pixels"
)
LEGACY_REFERENCE_WARNING = (
    "This calibration came from an ROI bounding box (older versions fell "
    "back to it when no line was drawn), not from a measured line. "
    "Redraw the reference line before trusting cm values."
)


def scale_factor_from_cm(real_world_cm: float, reference_px: float) -> float:
    """Return cm per pixel, or 0.0 (pixel units) if either input is unusable."""
    if real_world_cm is None or reference_px is None:
        return 0.0
    if real_world_cm <= 0 or reference_px <= 0:
        return 0.0
    return float(real_world_cm) / float(reference_px)


def real_world_cm_from_scale(
    scale_factor: float, reference_px: float
) -> Optional[float]:
    """Return the cm length a scale implies for a reference, or None."""
    if scale_factor is None or reference_px is None:
        return None
    if scale_factor <= 0 or reference_px <= 0:
        return None
    return float(scale_factor) * float(reference_px)


def rescale_for_new_reference(
    real_world_cm: float, new_reference_px: float
) -> float:
    """Recompute scale_factor after the reference line was redrawn.

    The cm length the user typed is the invariant; the old scale_factor
    belongs to the old pixel length and must not be carried over.
    """
    return scale_factor_from_cm(real_world_cm, new_reference_px)
