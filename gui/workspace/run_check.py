"""
gui/workspace/run_check.py
What the Run tab decides *before* it starts a worker, without any Qt.

* :func:`scope_keys` turns the run scope (all / selected / not yet done) into
  project video keys.
* :func:`prepare_run` builds the analysis plan for those keys and checks it
  the way the classic window and the CLI do: unconfigured and missing videos,
  the paradigm, ``Maze.validate_settings``, reserved sample-sheet columns and
  the batch pre-flight (video, disk and system checks). It returns a
  :class:`RunPreparation` whose ``issues`` block the run and whose ``warnings``
  the user may accept.
* :func:`find_resumable` pairs the checkpoints in an output folder with the
  project's videos, so an interrupted run can be offered for resuming.

Nothing here writes to the project or starts an analysis.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from checkpoint_manager import CheckpointManager
from core.models import BatchSettings
from core.output_contract import COMMON_BATCH_HEADERS
from core.project import (STATUS_DONE, STATUS_WARNING, Project,
                          SEVERITY_ERROR, SEVERITY_WARNING)
from core.run_outputs import apply_effective_parameters
from core.sample_sheet import SampleMetadata, reserved_column_conflicts
from core.time_bins import TIME_BIN_HEADERS
from gui.run_resume import InterruptedVideo, scan_for_resumable_runs
from mazes import AVAILABLE_MAZES, create_maze
from mazes.base_maze import Maze
from preflight_checker import PreflightChecker

logger = logging.getLogger(__name__)

SCOPE_ALL = "all"
SCOPE_SELECTED = "selected"
SCOPE_PENDING = "pending"
SCOPES = (SCOPE_ALL, SCOPE_SELECTED, SCOPE_PENDING)
SCOPE_LABELS = {
    SCOPE_ALL: "All configured videos",
    SCOPE_SELECTED: "Selected videos",
    SCOPE_PENDING: "Videos not yet done",
}

# Project issue codes the pre-run check repeats (the others are about
# results folders or unconfigured videos, which it reports itself).
_PROJECT_ISSUE_CODES = frozenset(
    {"paradigm_mismatch", "sample_sheet_missing", "video_changed"}
)
MAX_LISTED = 6


def scope_keys(
    project: Project, scope: str, selected: Sequence[str] = ()
) -> List[str]:
    """Video keys in project order for a run scope.

    ``selected`` videos are taken as they are; ``all`` and ``pending`` cover
    every video of the project, and :func:`prepare_run` then leaves out the
    ones without settings (``strict`` decides whether that blocks the run).
    """
    if scope == SCOPE_SELECTED:
        wanted = set(selected)
        return [video.key for video in project.videos if video.key in wanted]
    if scope == SCOPE_PENDING:
        return [
            video.key for video in project.videos
            if video.last_status not in (STATUS_DONE, STATUS_WARNING)
        ]
    return [video.key for video in project.videos]


@dataclass
class RunPreparation:
    """The checked, ready-to-start plan of one run (or why there is none)."""
    keys: List[str] = field(default_factory=list)           # videos in the plan
    plan: Dict[Path, BatchSettings] = field(default_factory=dict)
    video_keys: Dict[Path, str] = field(default_factory=dict)
    maze: Optional[Maze] = None
    sample_metadata: Optional[SampleMetadata] = None
    issues: List[str] = field(default_factory=list)         # block the run
    warnings: List[str] = field(default_factory=list)       # may be accepted
    skipped: List[Path] = field(default_factory=list)       # not configured

    @property
    def ok(self) -> bool:
        return not self.issues and bool(self.plan)


def _listed(names: Sequence[str]) -> str:
    shown = ", ".join(names[:MAX_LISTED])
    extra = len(names) - MAX_LISTED
    return shown + (f" and {extra} more" if extra > 0 else "")


def build_maze(
    project: Project, plan: Dict[Path, BatchSettings]
) -> Maze:
    """The maze of the project's paradigm, configured from the plan.

    Raises ``ValueError`` (unknown paradigm) or ``TypeError`` / ``ValueError``
    (parameters the maze does not accept).
    """
    paradigm = project.effective_paradigm()
    maze = create_maze(paradigm)
    first = next(iter(plan.values()), None)
    if first is not None and first.algorithm_parameters:
        maze.configure_algorithm_parameters(first.algorithm_parameters)
    return maze


def apply_validation_video(
    plan: Dict[Path, BatchSettings], create: Optional[bool]
) -> None:
    """Force the validation video on or off for this run (None = leave it).

    ``plan`` holds copies of the project's settings, so the project itself is
    never changed; the checkpoint hash of an overridden video does change.
    """
    if create is None:
        return
    for settings in plan.values():
        settings.create_timelapse = bool(create)


def prepare_run(
    project: Project,
    keys: Sequence[str],
    output_dir: Optional[Path],
    strict: bool = False,
    create_validation_video: Optional[bool] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    preflight: bool = True,
) -> RunPreparation:
    """Build and check the plan for ``keys``.

    ``strict`` makes videos without settings a blocking problem (used when
    the user picked the videos); otherwise they are left out with a warning.
    Videos whose file is gone always block the run: a plan quietly missing a
    video is the kind of surprise a batch should not have.
    """
    result = RunPreparation()
    if not keys:
        result.issues.append("No videos to analyse in this scope.")
        return result

    analysis = project.analysis_plan(keys)
    if analysis.unconfigured:
        names = [path.name for path in analysis.unconfigured]
        message = (
            f"{len(names)} video(s) have no settings: {_listed(names)}. "
            "Configure them in the Setup tab."
        )
        if strict:
            result.issues.append(message)
        else:
            result.warnings.append(message.replace(
                "Configure them", "They are left out of this run; configure them"
            ))
        result.skipped = list(analysis.unconfigured)
    if analysis.missing:
        names = [path.name for path in analysis.missing]
        result.issues.append(
            f"{len(names)} video file(s) not found: {_listed(names)}. "
            "Relink them in the video table."
        )
    plan = {
        path: settings for path, settings in analysis.plan.items()
        if path not in analysis.missing
    }
    result.plan = plan
    result.video_keys = {path: analysis.keys[path] for path in plan}
    result.keys = [result.video_keys[path] for path in plan]
    if not plan:
        if not result.issues:
            result.issues.append("None of the videos has settings yet.")
        return result

    paradigm = project.effective_paradigm()
    if not paradigm:
        result.issues.append(
            "The project has no paradigm. Choose one in the context panel."
        )
        return result
    if paradigm not in AVAILABLE_MAZES:
        result.issues.append(f"Unknown paradigm {paradigm!r}.")
        return result
    try:
        maze = build_maze(project, plan)
    except (TypeError, ValueError) as exc:
        result.issues.append(f"Incompatible configuration: {exc}")
        return result
    result.maze = maze
    apply_validation_video(plan, create_validation_video)

    # -- paradigm ------------------------------------------------------
    expected = None
    for path, settings in plan.items():
        result.issues.extend(
            f"{path.name}: {issue}" for issue in maze.validate_settings(settings)
        )
        parameters = settings.algorithm_parameters
        if expected is None:
            expected = parameters
        elif parameters != expected:
            result.issues.append(
                f"{path.name}: paradigm parameters differ from the rest of "
                "the batch; mixed-parameter batches are not supported"
            )
    wanted = set(result.keys)
    for issue in project.validate():
        if issue.code not in _PROJECT_ISSUE_CODES:
            continue
        if issue.key is not None and issue.key not in wanted:
            continue
        if issue.severity == SEVERITY_ERROR:
            if issue.message not in result.issues:
                result.issues.append(issue.message)
        elif issue.severity == SEVERITY_WARNING:
            result.warnings.append(issue.message)

    # -- sample sheet / metadata ----------------------------------------
    metadata = project.sample_metadata()
    result.sample_metadata = metadata
    if metadata is not None:
        reserved = (
            list(COMMON_BATCH_HEADERS) + list(maze.get_batch_summary_headers())
            + ["video_filename", "paradigm"] + list(TIME_BIN_HEADERS)
        )
        conflicts = reserved_column_conflicts(metadata.columns, reserved)
        if conflicts:
            result.issues.append(
                f"Metadata column(s) {', '.join(conflicts)} clash with output "
                "columns (batch summary or time bins). Rename them in the "
                "video table."
            )
        blank = [
            path.name for path in plan
            if not any(metadata.values_for(path).values())
        ]
        if blank:
            result.warnings.append(
                f"{len(blank)} video(s) have no metadata values: "
                f"{_listed(blank)}"
            )

    # -- pre-flight (videos, disk, system) ---------------------------------
    if preflight and not result.issues:
        try:
            pre_issues, pre_warnings = PreflightChecker().validate_batch(
                plan, output_dir, paradigm_name=maze.name, progress=progress
            )
        except Exception as exc:    # a probe bug must not block the app
            logger.warning("Pre-flight checks failed", exc_info=True)
            pre_issues, pre_warnings = [], [f"Pre-flight checks failed: {exc}"]
        result.issues.extend(pre_issues)
        result.warnings.extend(pre_warnings)
    return result


# ----------------------------------------------------------------------
# Resuming
# ----------------------------------------------------------------------
@dataclass
class ResumeOffer:
    """Interrupted videos of the project that a new run can continue."""
    keys: List[str] = field(default_factory=list)        # to run again
    resumable: List[InterruptedVideo] = field(default_factory=list)
    changed: List[InterruptedVideo] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.resumable)


def find_resumable(
    project: Project,
    output_dir: Optional[Path],
    create_validation_video: Optional[bool] = None,
) -> ResumeOffer:
    """Checkpoints in ``output_dir`` that belong to videos of ``project``.

    A checkpoint resumes only when its settings hash equals the hash the next
    run would use, so the project's current settings (with the validation
    video choice applied) are hashed exactly like the worker does. Videos of
    an interrupted run that had not started yet are added to ``keys``.
    """
    offer = ResumeOffer()
    if output_dir is None or not Path(output_dir).is_dir():
        return offer
    try:
        scan = scan_for_resumable_runs(Path(output_dir))
    except Exception:
        logger.warning("Could not inspect resumable analyses", exc_info=True)
        return offer
    if not scan:
        return offer

    interrupted = [item for run in scan.runs for item in run.interrupted]
    interrupted += list(scan.unmatched)
    mine = {
        item.video: project.find(item.video) for item in interrupted
        if project.find(item.video) is not None
    }
    if not mine:
        return offer
    entries = [entry for entry in mine.values()]
    analysis = project.analysis_plan([entry.key for entry in entries])
    if not analysis.plan:
        return offer
    try:
        maze = build_maze(project, analysis.plan)
    except (TypeError, ValueError):
        return offer
    apply_validation_video(analysis.plan, create_validation_video)
    apply_effective_parameters(analysis.plan, maze)
    hasher = CheckpointManager(Path(output_dir), create_dir=False)
    keys: List[str] = []
    for item in interrupted:
        entry = mine.get(item.video)
        settings = analysis.plan.get(entry.path) if entry is not None else None
        if settings is None:
            continue
        if hasher.get_settings_hash(settings) == item.settings_hash:
            offer.resumable.append(item)
            keys.append(entry.key)
        else:
            offer.changed.append(item)
    if not offer.resumable:
        return offer
    for run in scan.runs:
        for video in run.pending_videos:
            entry = project.find(video)
            if entry is not None and entry.key not in keys:
                if any(item.video == video for item in run.interrupted):
                    continue
                keys.append(entry.key)
    order = {key: index for index, key in enumerate(project.keys())}
    offer.keys = sorted(set(keys), key=lambda key: order.get(key, 0))
    return offer
