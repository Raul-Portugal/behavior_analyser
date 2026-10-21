"""
gui/workspace/stages/status.py
What the stepper shows for each stage, computed from the project alone.

This module is free of Qt, so it is tested without a display. The stepper
statuses it returns are the ones of :class:`gui.workspace.ui_kit.Stepper`
(``todo``, ``done``, ``attention`` and ``disabled``); the window adds
``current`` for the stage on screen.

Rules, per stage
----------------
Videos
    done: a paradigm is chosen, there is at least one video and none of
    their files is missing. attention: a file is missing.
Arena & zones
    done: the project template's regions are valid and no video needs its
    regions aligned (``zones_size_mismatch``). attention: invalid regions,
    or videos whose frame size differs from the one the regions were drawn
    on.
Detection (Motion tuning for the Tail Suspension Test)
    done: the template's detection (motion) settings validate and the user
    has reviewed them (``Project.detection_reviewed``, set by the stage when
    the preview was looked at or edited). attention: an invalid
    configuration.
Trials
    done: ``Project.trial_status`` is ``ok`` for every video.
    attention: some video has an invalid trial or no start.
Review & run
    done: ready to run (every video configured, found and free of errors).
    attention: errors or missing videos block a run.
Results
    done: at least one run is recorded.

Stages that need earlier ones are ``disabled`` (with the reason) until a
project, a paradigm and a video exist.
"""
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from core.project import (SEVERITY_ERROR, TRIAL_INVALID, TRIAL_NEEDS_START,
                          TRIAL_OK, Project, ProjectIssue)
from core.setup_state import STATUS_DONE as REPORT_DONE
from core.setup_state import SetupState
from mazes import AVAILABLE_MAZES, create_maze
from mazes.base_maze import STEP_DETECTION_TUNING, STEP_ROIS, STEP_TST_TUNING

STAGE_VIDEOS = "videos"
STAGE_ARENA = "arena"
STAGE_DETECTION = "detection"
STAGE_TRIALS = "trials"
STAGE_REVIEW = "review"
STAGE_RESULTS = "results"
STAGE_ORDER: Tuple[str, ...] = (
    STAGE_VIDEOS, STAGE_ARENA, STAGE_DETECTION, STAGE_TRIALS, STAGE_REVIEW,
    STAGE_RESULTS,
)

STATUS_TODO = "todo"
STATUS_DONE = "done"
STATUS_ATTENTION = "attention"
STATUS_DISABLED = "disabled"

MOTION_TITLE = "Motion tuning"
DETECTION_TITLE = "Detection"

NEEDS_PROJECT = "Open or create a project first"
NEEDS_PARADIGM = "Choose the test first"
NEEDS_VIDEO = "Add a video first"


@dataclass(frozen=True)
class StageStatus:
    """A stage's stepper status and the sentence that explains it."""
    status: str
    reason: str = ""


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def paradigm_known(project: Optional[Project]) -> bool:
    """True when the project has a paradigm this build can run."""
    return project is not None and project.effective_paradigm() in AVAILABLE_MAZES


def detection_step(paradigm: str) -> str:
    """The setup step of the detection stage: the tuner of the paradigm."""
    if paradigm in AVAILABLE_MAZES:
        try:
            if create_maze(paradigm).uses_motion_tuner:
                return STEP_TST_TUNING
        except (ValueError, TypeError):
            pass
    return STEP_DETECTION_TUNING


def detection_title(paradigm: str) -> str:
    """"Detection", or "Motion tuning" for the Tail Suspension Test."""
    return MOTION_TITLE if detection_step(paradigm) == STEP_TST_TUNING else DETECTION_TITLE


def _issues_of(project: Project, issues: Optional[Iterable[ProjectIssue]]) -> List[ProjectIssue]:
    return list(project.validate()) if issues is None else list(issues)


def _count_codes(issues: Iterable[ProjectIssue], code: str) -> int:
    return len({i.key for i in issues if i.code == code})


def _setup_state(project: Project) -> Optional[SetupState]:
    """The template as a ``SetupState`` with every step marked accepted."""
    template = project.template_settings
    if template is None:
        return None
    try:
        name = None if template.paradigm_id in AVAILABLE_MAZES else project.effective_paradigm()
        state = SetupState.from_batch_settings(template, maze_name=name)
    except (ValueError, KeyError, TypeError):
        return None
    state.completed_steps = set(state.steps)
    return state


# ----------------------------------------------------------------------
# One function per stage
# ----------------------------------------------------------------------
def videos_status(project: Optional[Project]) -> StageStatus:
    if project is None:
        return StageStatus(STATUS_TODO, "Open or create a project to start")
    if not paradigm_known(project):
        return StageStatus(STATUS_TODO, "Choose the test these videos are from")
    if not project.videos:
        return StageStatus(STATUS_TODO, "Add the videos you want to analyse")
    missing = len(project.missing_files())
    if missing:
        return StageStatus(
            STATUS_ATTENTION, f"{plural(missing, 'video')} could not be found on disk"
        )
    return StageStatus(STATUS_DONE, plural(len(project.videos), "video") + " added")


def arena_status(project: Optional[Project],
                 issues: Optional[Iterable[ProjectIssue]] = None) -> StageStatus:
    blocked = _blocked(project)
    if blocked is not None:
        return blocked
    state = _setup_state(project)
    if state is None:
        return StageStatus(STATUS_TODO, "Draw the arena and zones")
    report = state.step_report(STEP_ROIS)
    if report.issues:
        return StageStatus(STATUS_ATTENTION, report.issues[0])
    if report.status != REPORT_DONE:
        return StageStatus(STATUS_TODO, "Draw the arena and zones")
    misfits = _count_codes(_issues_of(project, issues), "zones_size_mismatch")
    if misfits:
        return StageStatus(
            STATUS_ATTENTION,
            f"{plural(misfits, 'video')} need their zones aligned",
        )
    return StageStatus(STATUS_DONE, "Arena and zones are set")


def detection_status(project: Optional[Project]) -> StageStatus:
    blocked = _blocked(project)
    if blocked is not None:
        return blocked
    step = detection_step(project.effective_paradigm())
    state = _setup_state(project)
    if state is None:
        return StageStatus(STATUS_TODO, "Set up the arena first, then tune detection")
    report = state.step_report(step)
    if report.issues:
        return StageStatus(STATUS_ATTENTION, report.issues[0])
    if report.status == REPORT_DONE and project.detection_reviewed:
        return StageStatus(STATUS_DONE, "Detection settings accepted")
    return StageStatus(STATUS_TODO, "Check that the animal is detected well")


def trials_status(project: Optional[Project]) -> StageStatus:
    blocked = _blocked(project)
    if blocked is not None:
        return blocked
    if project.template_settings is None and not any(
        v.settings is not None for v in project.videos
    ):
        return StageStatus(STATUS_TODO, "Set up the arena first, then set the trials")
    states = [project.trial_status(v) for v in project.videos if v.settings is not None]
    invalid = states.count(TRIAL_INVALID)
    needs_start = states.count(TRIAL_NEEDS_START)
    if invalid:
        return StageStatus(
            STATUS_ATTENTION, f"{plural(invalid, 'video')} have times that cannot be used"
        )
    if needs_start:
        return StageStatus(
            STATUS_ATTENTION, f"{plural(needs_start, 'video')} need a start time"
        )
    if states and all(state == TRIAL_OK for state in states):
        return StageStatus(STATUS_DONE, "Every trial has valid times")
    return StageStatus(STATUS_TODO, "Set when each trial starts and ends")


def review_status(project: Optional[Project],
                  issues: Optional[Iterable[ProjectIssue]] = None) -> StageStatus:
    blocked = _blocked(project)
    if blocked is not None:
        return blocked
    unconfigured = sum(1 for v in project.videos if v.settings is None)
    if unconfigured == len(project.videos):
        return StageStatus(STATUS_TODO, "Set up the arena and detection first")
    errors = [i for i in _issues_of(project, issues) if i.severity == SEVERITY_ERROR]
    if errors:
        videos = len({i.key for i in errors if i.key is not None})
        if videos:
            reason = f"{plural(videos, 'video')} need attention before a run"
        else:
            reason = errors[0].message
        return StageStatus(STATUS_ATTENTION, reason)
    if unconfigured:
        return StageStatus(
            STATUS_TODO, f"{plural(unconfigured, 'video')} still need settings"
        )
    return StageStatus(STATUS_DONE, "Ready to run")


def results_status(project: Optional[Project]) -> StageStatus:
    if project is None:
        return StageStatus(STATUS_DISABLED, NEEDS_PROJECT)
    if project.runs:
        return StageStatus(STATUS_DONE, plural(len(project.runs), "run") + " recorded")
    return StageStatus(STATUS_TODO, "Nothing has been analysed yet")


def _blocked(project: Optional[Project]) -> Optional[StageStatus]:
    """The disabled status of a stage that needs a project, paradigm and video."""
    if project is None:
        return StageStatus(STATUS_DISABLED, NEEDS_PROJECT)
    if not paradigm_known(project):
        return StageStatus(STATUS_DISABLED, NEEDS_PARADIGM)
    if not project.videos:
        return StageStatus(STATUS_DISABLED, NEEDS_VIDEO)
    return None


def stage_status(stage_id: str, project: Optional[Project],
                 issues: Optional[Iterable[ProjectIssue]] = None) -> StageStatus:
    """The status of one stage. ``issues`` (``project.validate()``) may be
    passed in to save a second validation."""
    if stage_id == STAGE_VIDEOS:
        return videos_status(project)
    if stage_id == STAGE_ARENA:
        return arena_status(project, issues)
    if stage_id == STAGE_DETECTION:
        return detection_status(project)
    if stage_id == STAGE_TRIALS:
        return trials_status(project)
    if stage_id == STAGE_REVIEW:
        return review_status(project, issues)
    if stage_id == STAGE_RESULTS:
        return results_status(project)
    raise KeyError(stage_id)


def all_stage_statuses(project: Optional[Project]) -> Dict[str, StageStatus]:
    """Every stage's status, validating the project once."""
    issues = None if project is None else project.validate()
    return {stage: stage_status(stage, project, issues) for stage in STAGE_ORDER}


__all__ = [
    "DETECTION_TITLE", "MOTION_TITLE", "STAGE_ARENA", "STAGE_DETECTION",
    "STAGE_ORDER", "STAGE_RESULTS", "STAGE_REVIEW", "STAGE_TRIALS",
    "STAGE_VIDEOS", "STATUS_ATTENTION", "STATUS_DISABLED", "STATUS_DONE",
    "STATUS_TODO", "StageStatus", "all_stage_statuses", "arena_status",
    "detection_status", "detection_step", "detection_title", "paradigm_known",
    "results_status", "review_status", "stage_status", "trials_status",
    "videos_status",
]
