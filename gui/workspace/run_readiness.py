"""
gui/workspace/run_readiness.py
What the Review & run stage tells the user *before* a run, without any Qt.

* :func:`build_readiness` turns the project's stage statuses
  (:mod:`gui.workspace.stages.status`) and its validation issues into one
  :class:`Readiness`: a checklist row per stage (Videos, Arena & zones,
  Detection, Trials), the per-video problems grouped by kind, and the list of
  things that block a run of the chosen scope (with the sentence the disabled
  Run button explains itself with).
* The groups are what the user can *fix*: every one names the stage that
  fixes it (``fix_stage``), which the stage turns into a "Fix ->" link.

The authoritative pre-run check stays :func:`gui.workspace.run_check.prepare_run`
(it also probes the video files and the disk); this module only mirrors the
cheap, project-level part of it so the Run button can be enabled or disabled
and explained at a glance.
"""
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from core.project import (SEVERITY_ERROR, SEVERITY_WARNING, Project,
                          ProjectIssue)
from gui.workspace.run_check import SCOPE_SELECTED
from gui.workspace.stages import status as stage_status

# Chip kinds of the checklist rows (``ui_kit.CHIP_KINDS``).
CHIP_OK = "ok"
CHIP_WARNING = "warning"
CHIP_ERROR = "error"
CHIP_NEUTRAL = "neutral"

CHECK_STAGES: Tuple[str, ...] = (
    stage_status.STAGE_VIDEOS, stage_status.STAGE_ARENA,
    stage_status.STAGE_DETECTION, stage_status.STAGE_TRIALS,
)
MAX_NAMES = 6

# Issue code -> (group title, stage that fixes it, short label of the fix).
_GROUPS: Dict[str, Tuple[str, str]] = {
    "missing_file": ("Video files not found", stage_status.STAGE_VIDEOS),
    "unconfigured": ("No settings yet", stage_status.STAGE_ARENA),
    "paradigm_mismatch": ("Set up for another test", stage_status.STAGE_VIDEOS),
    "trial_needs_start": ("Start time needed", stage_status.STAGE_TRIALS),
    "trial_invalid": ("Trial times that cannot be used", stage_status.STAGE_TRIALS),
    "zones_size_mismatch": ("Zones drawn on another frame size", stage_status.STAGE_ARENA),
    "video_changed": ("Video changed since it was set up", stage_status.STAGE_ARENA),
    "sample_sheet_missing": ("Sample sheet not found", ""),
}
# The order the groups are listed in.
_GROUP_ORDER = tuple(_GROUPS)


@dataclass(frozen=True)
class StageCheck:
    """One row of the readiness checklist."""
    stage_id: str
    title: str
    chip: str                 # ok / warning / error / neutral
    label: str                # the chip's text
    reason: str               # one line explaining the status
    fix_stage: str = ""       # where "Fix ->" goes ("" = nothing to fix)

    @property
    def done(self) -> bool:
        return self.chip == CHIP_OK


@dataclass(frozen=True)
class ProblemGroup:
    """Per-video problems of one kind."""
    code: str
    title: str
    kind: str                         # error / warning
    fix_stage: str                    # "" when there is nothing to open
    keys: Tuple[str, ...] = ()        # videos concerned (empty: the project)
    names: Tuple[str, ...] = ()
    messages: Tuple[str, ...] = ()

    @property
    def count(self) -> int:
        return len(self.keys) or len(self.messages)

    @property
    def blocking(self) -> bool:
        return self.kind == SEVERITY_ERROR

    def summary(self) -> str:
        """``Start time needed: 3 videos`` (no count for project-level notes)."""
        if not self.keys:
            return self.title
        return f"{self.title}: {stage_status.plural(len(self.keys), 'video')}"

    def listed_names(self, limit: int = MAX_NAMES) -> str:
        shown = ", ".join(self.names[:limit])
        extra = len(self.names) - limit
        return shown + (f" and {extra} more" if extra > 0 else "")


@dataclass
class Readiness:
    """The result of :func:`build_readiness`."""
    checks: List[StageCheck] = field(default_factory=list)
    groups: List[ProblemGroup] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    scope_count: int = 0          # videos in the chosen scope
    configured_count: int = 0     # ... that have settings (they are analysed)
    left_out: int = 0             # ... that have none (left out of the run)

    @property
    def ready(self) -> bool:
        return not self.blockers

    @property
    def why_not(self) -> str:
        """The sentence a disabled Run button shows ('' when it is ready)."""
        if not self.blockers:
            return ""
        first = self.blockers[0]
        extra = len(self.blockers) - 1
        return first + (f" (and {extra} more problem{'s' if extra != 1 else ''})"
                        if extra > 0 else "")

    def check(self, stage_id: str) -> Optional[StageCheck]:
        return next((c for c in self.checks if c.stage_id == stage_id), None)

    def group(self, code: str) -> Optional[ProblemGroup]:
        return next((g for g in self.groups if g.code == code), None)

    @property
    def warnings(self) -> List[str]:
        return [g.summary() for g in self.groups if not g.blocking]


# ----------------------------------------------------------------------
# The checklist rows
# ----------------------------------------------------------------------
def _check_for(stage_id: str, project: Optional[Project],
               issues: Optional[Sequence[ProjectIssue]]) -> StageCheck:
    found = stage_status.stage_status(stage_id, project, issues)
    title = _title(stage_id, project)
    status = found.status
    if status == stage_status.STATUS_DONE:
        return StageCheck(stage_id, title, CHIP_OK, "Done", found.reason)
    if status == stage_status.STATUS_ATTENTION:
        return StageCheck(stage_id, title, CHIP_ERROR, "Needs attention",
                          found.reason, stage_id)
    if status == stage_status.STATUS_DISABLED:
        # Go to the stage that makes this one available.
        return StageCheck(stage_id, title, CHIP_NEUTRAL, "Waiting", found.reason,
                          stage_status.STAGE_VIDEOS if stage_id != stage_status.STAGE_VIDEOS
                          else "")
    label = "Not reviewed" if stage_id == stage_status.STAGE_DETECTION else "To do"
    return StageCheck(stage_id, title, CHIP_WARNING, label, found.reason, stage_id)


def _title(stage_id: str, project: Optional[Project]) -> str:
    if stage_id == stage_status.STAGE_VIDEOS:
        return "Videos"
    if stage_id == stage_status.STAGE_ARENA:
        return "Arena & zones"
    if stage_id == stage_status.STAGE_TRIALS:
        return "Trials"
    paradigm = project.effective_paradigm() if project is not None else ""
    return stage_status.detection_title(paradigm)


# ----------------------------------------------------------------------
# The per-video problems
# ----------------------------------------------------------------------
def _group_issues(
    issues: Iterable[ProjectIssue], names: Dict[str, str], wanted: Optional[set],
    strict: bool,
) -> List[ProblemGroup]:
    by_code: Dict[str, List[ProjectIssue]] = {}
    for issue in issues:
        if issue.code not in _GROUPS:
            continue
        if issue.key is not None and wanted is not None and issue.key not in wanted:
            continue
        by_code.setdefault(issue.code, []).append(issue)
    groups: List[ProblemGroup] = []
    for code in _GROUP_ORDER:
        found = by_code.get(code)
        if not found:
            continue
        title, fix_stage = _GROUPS[code]
        kind = SEVERITY_ERROR if any(i.severity == SEVERITY_ERROR for i in found) \
            else SEVERITY_WARNING
        if code == "unconfigured" and strict:
            kind = SEVERITY_ERROR          # the user picked these videos
        keys: List[str] = []
        shown: List[str] = []
        for issue in found:
            if issue.key is not None and issue.key not in keys:
                keys.append(issue.key)
                shown.append(names.get(issue.key, issue.key))
        groups.append(ProblemGroup(
            code, title, kind, fix_stage, tuple(keys), tuple(shown),
            tuple(dict.fromkeys(i.message for i in found)),
        ))
    return groups


def build_readiness(
    project: Optional[Project],
    scope: str,
    scope_keys: Sequence[str],
    issues: Optional[Sequence[ProjectIssue]] = None,
) -> Readiness:
    """The checklist, the problems and the blockers for ``scope_keys``.

    ``issues`` (``project.validate()``) may be passed in to save a second
    validation. The checklist always covers the whole project; the problems
    and blockers only the videos of the scope (plus the project itself).
    """
    result = Readiness()
    if project is None:
        result.blockers.append("Open or create a project first")
        return result
    if issues is None:
        issues = project.validate()
    issues = list(issues)
    result.checks = [_check_for(stage, project, issues) for stage in CHECK_STAGES]
    entries = {v.key: v for v in project.videos}
    keys = [k for k in scope_keys if k in entries]
    result.scope_count = len(keys)
    wanted = set(keys)
    configured = [k for k in keys if k in entries and entries[k].settings is not None]
    result.configured_count = len(configured)
    result.left_out = len(keys) - len(configured)
    names = {key: entry.name for key, entry in entries.items()}
    strict = scope == SCOPE_SELECTED
    result.groups = _group_issues(issues, names, wanted, strict)

    # -- what blocks a run -------------------------------------------------
    if not project.videos:
        result.blockers.append("Add the videos you want to analyse first")
    elif not stage_status.paradigm_known(project):
        result.blockers.append("Choose the test these videos are from first")
    elif not keys:
        result.blockers.append(
            "Select the videos to analyse in the Videos stage"
            if scope == SCOPE_SELECTED else "There are no videos in this scope"
        )
    elif not configured:
        result.blockers.append(
            "None of the videos has settings yet: draw the arena and zones first"
        )
    for group in result.groups:
        if group.blocking:
            result.blockers.append(group.summary())
    return result


__all__ = [
    "CHECK_STAGES", "CHIP_ERROR", "CHIP_NEUTRAL", "CHIP_OK", "CHIP_WARNING",
    "ProblemGroup", "Readiness", "StageCheck", "build_readiness",
]
