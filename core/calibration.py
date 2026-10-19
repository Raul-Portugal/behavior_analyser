"""Pure helpers for distance calibration from a user-drawn reference line."""
from dataclasses import dataclass
from typing import Optional

NO_CALIBRATION_NOTE = (
    "No calibration line drawn — distances will be reported in pixels"
)
LEGACY_REFERENCE_WARNING = (
    "This calibration came from an ROI bounding box (older versions fell "
    "back to it when no line was drawn), not from a measured line. "
    "Redraw the reference line before trusting cm values."
)
APPROXIMATE_REFERENCE_NOTE = (
    "This reference length was scaled when the template was aligned to "
    "this video (the original line's endpoints were not stored), so the "
    "calibration is approximate. Redraw the reference line to be sure."
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


# What a ROI manager says about its calibration reference.
REFERENCE_NONE = "none"
REFERENCE_DRAWN = "drawn"
REFERENCE_LEGACY = "legacy_bbox"
REFERENCE_APPROXIMATE = "approximate"


@dataclass(frozen=True)
class ReferenceStatus:
    """The state of the calibration reference line, in words.

    ``text`` names the reference, ``note`` is the explanation to show under
    it (empty when there is nothing to add) and ``warning`` marks notes the
    user should act on.
    """
    kind: str
    length_px: float
    text: str
    note: str = ""
    warning: bool = False


def describe_reference(roi_manager) -> ReferenceStatus:
    """Classify a ROI manager's reference: none, drawn, legacy or approximate."""
    length = (
        float(roi_manager.reference_length_pixels)
        if roi_manager is not None else 0.0
    )
    if length <= 0:
        return ReferenceStatus(
            REFERENCE_NONE, 0.0, "No reference line drawn",
            NO_CALIBRATION_NOTE,
        )
    if roi_manager.is_legacy_bbox_reference:
        return ReferenceStatus(
            REFERENCE_LEGACY, length,
            f"Legacy ROI bounding-box width ({length:.2f} px)",
            LEGACY_REFERENCE_WARNING, True,
        )
    if roi_manager.reference_is_approximate:
        return ReferenceStatus(
            REFERENCE_APPROXIMATE, length,
            f"Scaled reference length ({length:.2f} px)",
            APPROXIMATE_REFERENCE_NOTE, True,
        )
    return ReferenceStatus(
        REFERENCE_DRAWN, length,
        f"User-drawn reference line ({length:.2f} px)",
    )
