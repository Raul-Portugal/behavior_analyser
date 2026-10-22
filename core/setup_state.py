"""
core/setup_state.py
Qt-free state of one analysis setup, shared by the classic dialogs and the
workspace.

A ``SetupState`` holds everything a setup collects for one reference video:
the paradigm and its parameters, the ROIs (with the calibration line), the
detection config, the TST motion config, the real-world calibration length,
the time windows, the validation-video option and which steps the user has
finished. It converts losslessly to and from ``BatchSettings`` and answers
"is this step done / todo / invalid, and why" from the maze's own hooks
(``Maze.setup_steps``, ``validate_rois``, ``validate_parameters`` ...), so no
paradigm knowledge lives here.
"""
import copy
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from core.calibration import (APPROXIMATE_REFERENCE_NOTE,
                              LEGACY_REFERENCE_WARNING,
                              real_world_cm_from_scale, scale_factor_from_cm)
from core.models import AppConfig, BatchSettings, DetectionConfig
from mazes import AVAILABLE_MAZES, create_maze, maze_class_for_settings
from mazes.base_maze import (ALL_SETUP_STEPS, STEP_CALIBRATION, STEP_DETECTION_TUNING,
                             STEP_PARADIGM_PARAMS, STEP_REVIEW, STEP_ROIS,
                             STEP_TIME_WINDOW, STEP_TST_TUNING, Maze)

STEP_LABELS = {
    STEP_PARADIGM_PARAMS: "Paradigm parameters",
    STEP_ROIS: "Regions of interest",
    STEP_CALIBRATION: "Calibration",
    STEP_DETECTION_TUNING: "Detection tuning",
    STEP_TST_TUNING: "TST motion tuning",
    STEP_TIME_WINDOW: "Time window",
    STEP_REVIEW: "Review",
}

STATUS_DONE = "done"
STATUS_TODO = "todo"
STATUS_INVALID = "invalid"

TimeWindow = Tuple[float, float]

GLOBAL_WINDOW_ISSUE = "The end time must be greater than the start time."


def per_roi_window_issue(name: str) -> str:
    return (
        f"{name.replace('_', ' ').title()} must have an end time greater "
        "than its start time."
    )


@dataclass(frozen=True)
class StepReport:
    """Status of one setup step plus what is wrong (or worth knowing)."""
    status: str
    issues: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == STATUS_DONE


class _SettingsShim:
    """The three attributes ``Maze.infer_from_settings`` reads."""

    def __init__(self, algorithm_parameters, roi_manager, paradigm_id):
        self.algorithm_parameters = algorithm_parameters
        self.roi_manager = roi_manager
        self.paradigm_id = paradigm_id


@dataclass
class SetupState:
    maze_name: str
    # Paradigm parameters exactly as ``Maze.get_algorithm_parameters`` writes
    # them (zones, dwell, TST mouse count and motion config ...).
    algorithm_parameters: Dict[str, Any] = field(default_factory=dict)
    roi_manager: Any = None
    detection_config: DetectionConfig = field(
        default_factory=lambda: AppConfig().detection
    )
    # Real-world length (cm) of the reference line; 0 means "no calibration".
    scale_cm: float = 0.0
    start_time: float = 0.0
    end_time: Optional[float] = None
    per_roi_times: Dict[str, TimeWindow] = field(default_factory=dict)
    create_timelapse: bool = False
    video_path: Optional[Path] = None
    completed_steps: Set[str] = field(default_factory=set)
    # True for settings loaded without a ``paradigm_id``: the maze is then
    # inferred from the ROIs and the result is written back untagged.
    legacy_untagged: bool = False
    # The scale factor a loaded configuration carried, valid while the cm
    # length and reference length it was made with are unchanged. Keeps the
    # BatchSettings round trip bit-exact.
    _loaded_scale: Optional[Tuple[float, float, float]] = field(
        default=None, repr=False, compare=False
    )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def new(cls, maze_name: str, video_path: Optional[Path] = None
            ) -> "SetupState":
        """A blank setup for a paradigm, with its default parameters."""
        maze = create_maze(maze_name)
        return cls(
            maze_name=maze_name,
            algorithm_parameters=copy.deepcopy(
                maze.get_algorithm_parameters()
            ),
            video_path=Path(video_path) if video_path else None,
        )

    @classmethod
    def from_batch_settings(
        cls,
        settings: BatchSettings,
        maze_name: Optional[str] = None,
        video_path: Optional[Path] = None,
        completed_steps: Optional[Set[str]] = None,
    ) -> "SetupState":
        """State for saved settings; ``to_batch_settings`` restores them.

        ``maze_name`` overrides the paradigm (used when the settings carry
        no usable ``paradigm_id``); otherwise the id, or failing that the
        legacy ROI-name inference, decides.
        """
        if maze_name is None:
            if settings.paradigm_id in AVAILABLE_MAZES:
                maze_name = settings.paradigm_id
            else:
                maze_name = maze_class_for_settings(settings).name
        settings = settings.copy()
        roi_manager = settings.roi_manager
        reference_px = roi_manager.reference_length_pixels
        scale_cm = real_world_cm_from_scale(
            settings.scale_factor, reference_px
        ) or 0.0
        return cls(
            maze_name=maze_name,
            algorithm_parameters=copy.deepcopy(settings.algorithm_parameters),
            roi_manager=roi_manager,
            detection_config=settings.detection_config,
            scale_cm=scale_cm,
            start_time=settings.start_time,
            end_time=settings.end_time,
            per_roi_times=dict(settings.per_roi_times),
            create_timelapse=settings.create_timelapse,
            video_path=Path(video_path) if video_path else None,
            completed_steps=set(completed_steps or ()),
            legacy_untagged=not settings.paradigm_id,
            _loaded_scale=(
                scale_cm, float(reference_px), float(settings.scale_factor)
            ),
        )

    def copy(self) -> "SetupState":
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # The maze
    # ------------------------------------------------------------------
    def build_maze(self) -> Maze:
        """A fresh maze configured from this state's parameters."""
        maze = create_maze(self.maze_name)
        if self.legacy_untagged and self.roi_manager is not None:
            maze.infer_from_settings(_SettingsShim(
                copy.deepcopy(self.algorithm_parameters), self.roi_manager,
                None,
            ))
        elif self.algorithm_parameters:
            maze.configure_algorithm_parameters(
                copy.deepcopy(self.algorithm_parameters)
            )
        return maze

    def set_maze(self, maze: Maze) -> None:
        """Adopt a maze's name and parameters (what a parameter panel does)."""
        self.maze_name = maze.name
        self.algorithm_parameters = copy.deepcopy(
            maze.get_algorithm_parameters()
        )
        self.legacy_untagged = False

    @property
    def steps(self) -> List[str]:
        """The setup steps of this paradigm, in order."""
        return self.build_maze().setup_steps()

    @property
    def tst_motion_config(self):
        """The TST motion config (None for paradigms without one)."""
        return getattr(self.build_maze(), "motion_config", None)

    @tst_motion_config.setter
    def tst_motion_config(self, config) -> None:
        maze = self.build_maze()
        if not hasattr(maze, "motion_config"):
            raise ValueError(f"{maze.name} has no motion config")
        config.validate()
        maze.motion_config = config
        self.algorithm_parameters = copy.deepcopy(
            maze.get_algorithm_parameters()
        )

    # ------------------------------------------------------------------
    # Calibration and time windows
    # ------------------------------------------------------------------
    @property
    def reference_length_px(self) -> float:
        if self.roi_manager is None:
            return 0.0
        return float(self.roi_manager.reference_length_pixels)

    @property
    def scale_factor(self) -> float:
        """cm per pixel, 0.0 for pixel units (or paradigms with no distance)."""
        if not self.build_maze().reports_distance:
            return 0.0
        reference_px = self.reference_length_px
        loaded = self._loaded_scale
        if loaded is not None and loaded[:2] == (
            float(self.scale_cm), reference_px
        ):
            return loaded[2]
        return scale_factor_from_cm(self.scale_cm, reference_px)

    def set_per_roi_times(self, times: Dict[str, TimeWindow]) -> None:
        """Set per-ROI windows and the global window that spans them."""
        self.per_roi_times = {
            name: (float(window[0]), float(window[1]))
            for name, window in times.items()
        }
        if self.per_roi_times:
            self.start_time = min(w[0] for w in self.per_roi_times.values())
            self.end_time = max(w[1] for w in self.per_roi_times.values())

    # ------------------------------------------------------------------
    # BatchSettings
    # ------------------------------------------------------------------
    def to_batch_settings(self, finalize: bool = False) -> BatchSettings:
        """The settings this state describes (independent copies).

        With ``finalize`` the paradigm id and the maze's full, normalised
        parameters are written, as the classic review does when it accepts.
        """
        if self.roi_manager is None:
            raise ValueError("The regions of interest are not defined yet")
        maze = self.build_maze()
        if finalize:
            paradigm_id = maze.name
            parameters = copy.deepcopy(maze.get_algorithm_parameters())
        else:
            paradigm_id = None if self.legacy_untagged else self.maze_name
            parameters = copy.deepcopy(self.algorithm_parameters)
        return BatchSettings(
            roi_manager=copy.deepcopy(self.roi_manager),
            detection_config=copy.deepcopy(self.detection_config),
            scale_factor=self.scale_factor,
            start_time=self.start_time,
            end_time=self.end_time,
            create_timelapse=self.create_timelapse,
            per_roi_times=dict(self.per_roi_times),
            paradigm_id=paradigm_id,
            algorithm_parameters=parameters,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary_lines(self) -> List[Tuple[str, str]]:
        """``(label, value)`` rows that describe the setup in plain words."""
        maze = self.build_maze()
        rows: List[Tuple[str, str]] = [("Paradigm", maze.name)]

        parameters = []
        for key, value in sorted(maze.get_algorithm_parameters().items()):
            if key in ("motion_config", "engine_id", "zones_configured"):
                continue
            if key == "zones":
                value = ", ".join(display for _, display in value) or "none"
            parameters.append(f"{key.replace('_', ' ')}: {value}")
        rows.append(("Parameters", "; ".join(parameters) or "none"))

        manager = self.roi_manager
        if manager is None or (not manager.rois and manager.arena is None):
            rows.append(("Regions", "not drawn yet"))
        else:
            parts = [
                f"{name} ({len(polygons)})"
                for name, polygons in manager.rois.items()
            ]
            if manager.arena is not None:
                parts.append("arena")
            rows.append(("Regions", ", ".join(parts)))

        if maze.reports_distance:
            scale = self.scale_factor
            if scale > 0:
                rows.append((
                    "Calibration",
                    f"{self.scale_cm:g} cm reference line "
                    f"({scale:.5f} cm per pixel)",
                ))
            else:
                rows.append(("Calibration", "none: distances in pixels"))
        if maze.uses_detection_tuner:
            detection = self.detection_config
            rows.append((
                "Detection",
                f"threshold {detection.threshold_percentile:g}th percentile, "
                f"minimum area {detection.min_component_area} px",
            ))
        if maze.uses_motion_tuner:
            rows.append(("Motion tuning", "TST motion evidence configured"))

        if maze.has_per_roi_times:
            windows = [
                f"{name}: {start:g}-{end:g} s"
                for name, (start, end) in sorted(self.per_roi_times.items())
            ]
            rows.append(("Time windows", "; ".join(windows) or "whole video"))
        else:
            end = "end of video" if self.end_time is None else f"{self.end_time:g} s"
            rows.append(("Time window", f"{self.start_time:g} s to {end}"))
        rows.append((
            "Validation video", "yes" if self.create_timelapse else "no"
        ))
        return rows

    # ------------------------------------------------------------------
    # Step bookkeeping
    # ------------------------------------------------------------------
    def mark_done(self, step_id: str) -> None:
        self.completed_steps.add(step_id)

    def mark_todo(self, step_id: str) -> None:
        self.completed_steps.discard(step_id)

    def step_status(self, step_id: str) -> str:
        """``"done"``, ``"todo"`` or ``"invalid"`` (see ``step_report``)."""
        return self.step_report(step_id).status

    def step_issues(self, step_id: str) -> List[str]:
        return list(self.step_report(step_id).issues)

    def step_report(self, step_id: str) -> StepReport:
        """Status, blocking issues and warnings of one step.

        A step the paradigm does not have (no calibration for TST, no
        detection tuner for TST ...) is reported as done.
        """
        if step_id not in ALL_SETUP_STEPS:
            raise ValueError(f"Unknown setup step: {step_id!r}")
        maze = self.build_maze()
        if step_id not in maze.setup_steps():
            return StepReport(STATUS_DONE)
        handler = {
            STEP_PARADIGM_PARAMS: self._report_params,
            STEP_ROIS: self._report_rois,
            STEP_CALIBRATION: self._report_calibration,
            STEP_DETECTION_TUNING: self._report_detection,
            STEP_TST_TUNING: self._report_motion,
            STEP_TIME_WINDOW: self._report_time_window,
            STEP_REVIEW: self._report_review,
        }[step_id]
        return handler(maze)

    def step_reports(self) -> Dict[str, StepReport]:
        """Report of every step of this paradigm, in order."""
        return {step: self.step_report(step) for step in self.steps}

    def is_ready(self) -> bool:
        """Every step is done (the review included)."""
        return all(report.ok for report in self.step_reports().values())

    def _marked(self, step_id: str, issues, warnings=()) -> StepReport:
        if issues:
            return StepReport(
                STATUS_INVALID, tuple(issues), tuple(warnings)
            )
        status = STATUS_DONE if step_id in self.completed_steps else STATUS_TODO
        return StepReport(status, (), tuple(warnings))

    def _report_params(self, maze: Maze) -> StepReport:
        return self._marked(STEP_PARADIGM_PARAMS, maze.validate_parameters())

    def _report_rois(self, maze: Maze) -> StepReport:
        manager = self.roi_manager
        if manager is None:
            return StepReport(STATUS_TODO)
        issues = maze.validate_rois(manager)
        if not manager.rois and getattr(manager, "arena", None) is None:
            # Nothing drawn yet: still to do if the paradigm needs shapes,
            # otherwise (zone-free open field) done once the user confirms.
            if issues:
                return StepReport(STATUS_TODO)
            return self._marked(STEP_ROIS, ())
        if issues:
            return StepReport(STATUS_INVALID, tuple(issues))
        return StepReport(STATUS_DONE)

    def _report_calibration(self, maze: Maze) -> StepReport:
        issues: List[str] = []
        warnings: List[str] = []
        if self.roi_manager is None:
            return StepReport(STATUS_TODO)
        if not isfinite(float(self.scale_cm)) or self.scale_cm < 0:
            issues.append("The real-world length cannot be negative")
        elif self.scale_cm > 0 and self.reference_length_px <= 0:
            issues.append(
                "A real-world length needs a reference line: draw one in "
                "the regions step"
            )
        if self.reference_length_px > 0:
            if self.roi_manager.is_legacy_bbox_reference:
                warnings.append(LEGACY_REFERENCE_WARNING)
            elif self.roi_manager.reference_is_approximate:
                warnings.append(APPROXIMATE_REFERENCE_NOTE)
        if issues:
            return StepReport(STATUS_INVALID, tuple(issues), tuple(warnings))
        if self.scale_cm > 0:
            return StepReport(STATUS_DONE, (), tuple(warnings))
        # No calibration: pixel units, fine once the user chose that.
        return self._marked(STEP_CALIBRATION, (), warnings)

    def _report_detection(self, maze: Maze) -> StepReport:
        try:
            self.detection_config.validate()
        except ValueError as exc:
            return StepReport(STATUS_INVALID, (str(exc),))
        return self._marked(STEP_DETECTION_TUNING, ())

    def _report_motion(self, maze: Maze) -> StepReport:
        config = getattr(maze, "motion_config", None)
        if config is not None:
            try:
                config.validate()
            except ValueError as exc:
                return StepReport(STATUS_INVALID, (str(exc),))
        return self._marked(STEP_TST_TUNING, ())

    def window_issues(self, maze: Maze) -> List[str]:
        """Problems with the start/end (or per-ROI) windows."""
        issues: List[str] = []
        if not isfinite(float(self.start_time)) or self.start_time < 0:
            issues.append("The start time cannot be negative")
        if self.end_time is not None and not (
            isfinite(float(self.end_time)) and self.end_time > self.start_time
        ):
            issues.append(GLOBAL_WINDOW_ISSUE)
        if maze.has_per_roi_times:
            subjects = (
                maze.time_window_subjects(self.roi_manager)
                if self.roi_manager is not None else []
            )
            for name, window in sorted(self.per_roi_times.items()):
                if not (window[0] >= 0 and window[1] > window[0]):
                    issues.append(per_roi_window_issue(name))
                if self.roi_manager is not None and name not in subjects:
                    issues.append(
                        f"Time range for unknown mouse: {name}"
                    )
        return issues

    def _report_time_window(self, maze: Maze) -> StepReport:
        return self._marked(STEP_TIME_WINDOW, self.window_issues(maze))

    def _report_review(self, maze: Maze) -> StepReport:
        issues: List[str] = []
        others = [s for s in maze.setup_steps() if s != STEP_REVIEW]
        not_done = []
        for step in others:
            report = self.step_report(step)
            issues.extend(
                f"{STEP_LABELS[step]}: {issue}" for issue in report.issues
            )
            if report.status == STATUS_TODO:
                not_done.append(STEP_LABELS[step])
        if self.roi_manager is not None and not issues:
            try:
                settings = self.to_batch_settings()
            except ValueError as exc:
                issues.append(str(exc))
            else:
                issues.extend(maze.validate_settings(settings))
        if issues:
            return StepReport(STATUS_INVALID, tuple(issues))
        if not_done:
            return StepReport(STATUS_TODO)
        status = (
            STATUS_DONE if STEP_REVIEW in self.completed_steps
            else STATUS_TODO
        )
        return StepReport(status)
