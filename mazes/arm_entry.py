"""Setup hooks shared by the arm-entry mazes (Y-maze and elevated plus maze)."""
from typing import Iterable, List

from mazes.base_maze import (
    KIND_ARMS, STEP_CALIBRATION, STEP_DETECTION_TUNING, STEP_PARADIGM_PARAMS,
    STEP_REVIEW, STEP_ROIS, STEP_TIME_WINDOW,
)

# Arm-entry parameter bounds, shared with the parameter panel.
MIN_DWELL_RANGE_S = (0.0, 10.0)


class ArmEntrySetupMixin:
    """Dwell time and initial-arm rule come first, then the usual steps."""

    # Names that identify the paradigm in configurations saved without a
    # ``paradigm_id`` (pre-3.x files).
    LEGACY_ROI_SIGNATURE: tuple = ()

    @property
    def wizard_kind(self) -> str:
        return KIND_ARMS

    def setup_steps(self) -> List[str]:
        return [
            STEP_PARADIGM_PARAMS, STEP_ROIS, STEP_CALIBRATION,
            STEP_DETECTION_TUNING, STEP_TIME_WINDOW, STEP_REVIEW,
        ]

    def validate_parameters(self) -> List[str]:
        low, high = MIN_DWELL_RANGE_S
        if not low <= float(self.min_dwell_seconds) <= high:
            return [
                f"Minimum dwell time must be between {low:g} and {high:g} s"
            ]
        return []

    @classmethod
    def matches_roi_names(cls, roi_names: Iterable[str]) -> bool:
        names = set(roi_names)
        return bool(cls.LEGACY_ROI_SIGNATURE) and all(
            name in names for name in cls.LEGACY_ROI_SIGNATURE
        )
