"""
gui/setup_progress.py
What the setup wizard has collected so far, kept in memory so cancelling a
late step does not throw away the ROIs, rules and tuning of the earlier ones.

Pure data and decisions: no widgets. The main window owns one instance at a
time and offers to resume it the next time the user starts an analysis.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STEP_MICE = "mice"
STEP_ZONES = "zones"
STEP_ROI = "roi"
STEP_ARM_RULES = "arm_rules"
STEP_TST_TUNER = "tst_tuner"
STEP_DETECTION = "detection"

STEP_LABELS = {
    STEP_MICE: "number of mice",
    STEP_ZONES: "zone names and movement parameters",
    STEP_ROI: "regions of interest",
    STEP_ARM_RULES: "arm-entry rules",
    STEP_TST_TUNER: "TST motion tuning",
    STEP_DETECTION: "detection tuning",
}

KIND_TST = "tst"
KIND_FREESTYLE = "freestyle"
KIND_ARMS = "arms"
KIND_STANDARD = "standard"

_STEPS_BY_KIND = {
    KIND_TST: (STEP_MICE, STEP_ROI, STEP_TST_TUNER),
    KIND_FREESTYLE: (STEP_ZONES, STEP_ROI, STEP_DETECTION),
    KIND_ARMS: (STEP_ROI, STEP_ARM_RULES, STEP_DETECTION),
    KIND_STANDARD: (STEP_ROI, STEP_DETECTION),
}


def wizard_steps(kind: str) -> Tuple[str, ...]:
    """Steps the wizard runs for a paradigm kind, in order."""
    return _STEPS_BY_KIND[kind]


@dataclass
class SetupProgress:
    maze_name: str
    video_path: Path
    kind: str
    completed: List[str] = field(default_factory=list)
    # Maze parameters (mice, zones, rules, motion tuning) after the last
    # completed step, so they can be restored on resume.
    algorithm_parameters: Dict[str, Any] = field(default_factory=dict)
    zone_free_mode: bool = False
    roi_manager: Any = None
    detection_config: Any = None
    # Set once every step is done and only the final review is still open.
    base_settings: Any = None

    @property
    def steps(self) -> Tuple[str, ...]:
        return wizard_steps(self.kind)

    def has(self, step: str) -> bool:
        return step in self.completed

    def mark(self, step: str, algorithm_parameters: Optional[Dict[str, Any]] = None):
        if step not in self.completed:
            self.completed.append(step)
        if algorithm_parameters is not None:
            self.algorithm_parameters = algorithm_parameters

    def next_step(self) -> Optional[str]:
        for step in self.steps:
            if not self.has(step):
                return step
        return None

    @property
    def is_resumable(self) -> bool:
        """Worth offering again: the ROIs (the costly part) are done."""
        return self.has(STEP_ROI) and self.roi_manager is not None

    def describe(self) -> str:
        done = [STEP_LABELS[step] for step in self.steps if self.has(step)]
        upcoming = self.next_step()
        next_text = (
            STEP_LABELS[upcoming] if upcoming is not None
            else "the final review of times and calibration"
        )
        return (
            f"Done: {', '.join(done) if done else 'nothing'}.\n"
            f"Next: {next_text}."
        )

    def applies_to(self, maze_name: str,
                   video_dimensions: Optional[Tuple[int, int]]) -> bool:
        """Reusable for this paradigm and a reference video of the same size."""
        if self.maze_name != maze_name or self.roi_manager is None:
            return False
        stored = getattr(self.roi_manager, "source_dimensions", None)
        if not stored or video_dimensions is None:
            return True
        return tuple(stored) == tuple(video_dimensions)
