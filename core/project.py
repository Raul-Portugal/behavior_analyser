"""A persistent project: everything about one study in a single file.

A :class:`Project` holds the videos of a study, the settings of every video,
free metadata per video, the run options and the history of the runs made
from it. It is the data foundation of the 4.0 workspace: the workspace window
edits a project and the analysis is started from :meth:`Project.analysis_plan`.

Nothing here touches Qt. Reading and writing ``.bmaproj`` files (and the
importers for templates, batch plans and 3.x run folders) lives in
:mod:`core.project_io`.

Template and overrides
----------------------
Every video follows the project template. What is different about one video
is stored as a layer over it, never as a copy of every setting: a
:class:`TrialWindow` (when its trial starts and ends, or one window per
mouse) and a :class:`ZoneOverride` (its own regions, adjusted or aligned).
:meth:`Project.effective_settings` composes template, zones and trial into
the exact ``BatchSettings`` a run uses, identical (``to_dict`` and settings
hash) to a full copy with those values, so analysis outputs and checkpoints
do not depend on how a difference is stored. A full copy per video
(``customize``) remains for anything the layers cannot express. Project
files of schema 1 (full copies only) are migrated on load where that is
exact.

Paths
-----
In memory every path is absolute. In the file, a path is stored relative to
the project file when that is possible (so a study folder can be moved or
shared as a whole) and absolute otherwise (another drive). A video's ``key``
is an opaque, stable identifier created from the path relative to the project
file; it never changes when the video is relinked, which keeps run history
and exports pointing at the same entry.
"""
import copy
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Tuple, Union)

from core.models import BatchSettings
from core.run_outputs import (MANIFEST_FILENAME, MANIFEST_SCHEMA_VERSION,
                              RUN_PLAN_FILENAME)
from core.sample_sheet import (SampleMetadata, SampleSheetMatch,
                               load_sample_sheet, metadata_for_videos)
from core.version import APP_VERSION

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
FILE_TYPE = "project"
PROJECT_EXTENSION = ".bmaproj"

# VideoEntry.last_status
STATUS_NEW = "new"
STATUS_CONFIGURED = "configured"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_WARNING = "warning"
STATUS_INTERRUPTED = "interrupted"
STATUSES = (
    STATUS_NEW, STATUS_CONFIGURED, STATUS_DONE, STATUS_FAILED,
    STATUS_WARNING, STATUS_INTERRUPTED,
)

# VideoEntry.settings_source
SOURCE_TEMPLATE = "template"
SOURCE_CUSTOM = "custom"
SOURCE_ALIGNED = "aligned"
SETTINGS_SOURCES = (SOURCE_TEMPLATE, SOURCE_CUSTOM, SOURCE_ALIGNED)

# Written in place of a settings object identical to the project template.
_TEMPLATE_REFERENCE = "template"

# ZoneOverride.source
ZONE_ADJUSTED = "adjusted"
ZONE_ALIGNED = "aligned"
ZONE_SOURCES = (ZONE_ADJUSTED, ZONE_ALIGNED)

# Project.zone_status: where a video's regions come from.
ZONES_TEMPLATE = "template"
ZONES_ADJUSTED = "adjusted"
ZONES_ALIGNED = "aligned"
ZONES_CUSTOM = "custom"      # a full legacy copy of the settings
ZONE_STATUSES = (ZONES_TEMPLATE, ZONES_ADJUSTED, ZONES_ALIGNED, ZONES_CUSTOM)

# TrialProtocol.start_mode
START_VIDEO = "video_start"    # every trial begins when the video does
START_PER_VIDEO = "per_video"  # each video has its own start (mouse entry)
START_MODES = (START_VIDEO, START_PER_VIDEO)

# Project.trial_status
TRIAL_OK = "ok"
TRIAL_NEEDS_START = "needs_start"
TRIAL_INVALID = "invalid"
TRIAL_STATUSES = (TRIAL_OK, TRIAL_NEEDS_START, TRIAL_INVALID)


class ProjectError(ValueError):
    """A project file or operation is invalid; the message says why."""


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _now() -> str:
    """ISO timestamp in the format the run manifest uses."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _is_absolute(text: str) -> bool:
    return PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute()


def norm_path(path: os.PathLike) -> str:
    """Case-normalised absolute spelling, for comparing paths."""
    return os.path.normcase(str(Path(path).resolve()))


def store_path(path: Optional[os.PathLike], base_dir: Optional[Path]) -> Optional[str]:
    """How a path is written to a file: relative to ``base_dir`` if possible.

    Always POSIX separators; the absolute path is the fallback (no base
    folder, or another drive).
    """
    if path is None:
        return None
    resolved = Path(path).resolve()
    if base_dir is not None:
        try:
            return Path(
                os.path.relpath(resolved, Path(base_dir).resolve())
            ).as_posix()
        except ValueError:  # different drive
            pass
    return resolved.as_posix()


def resolve_stored_path(text: Optional[str], base_dir: Optional[Path]) -> Optional[Path]:
    """Inverse of :func:`store_path`: an absolute path for a stored one."""
    if text is None or text == "":
        return None
    if _is_absolute(text) or base_dir is None:
        return Path(text).resolve()
    return Path(os.path.normpath(Path(base_dir) / text)).resolve()


def rebase_path(
    path: Optional[Path], old_root: Path, new_root: Path
) -> Optional[Path]:
    """``path`` moved from under ``old_root`` to under ``new_root``.

    Returns ``None`` when ``path`` is not under ``old_root``.
    """
    if path is None:
        return None
    try:
        relative = Path(norm_path(path)).relative_to(norm_path(old_root))
    except ValueError:
        return None
    return (Path(new_root) / relative).resolve()


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def video_fingerprint(path: Path) -> Optional[str]:
    """``CheckpointManager.get_video_fingerprint``; None if unreadable."""
    from checkpoint_manager import CheckpointManager
    try:
        return CheckpointManager.get_video_fingerprint(Path(path))
    except OSError:
        return None


def settings_hash(settings: BatchSettings) -> str:
    """The hash a checkpoint and the run manifest record for ``settings``."""
    from checkpoint_manager import CheckpointManager
    return CheckpointManager(
        Path("."), create_dir=False
    ).get_settings_hash(settings)


def _settings_to_json(settings: Optional[BatchSettings]) -> Optional[str]:
    return None if settings is None else canonical_json(settings.to_dict())


# ----------------------------------------------------------------------
# Run options
# ----------------------------------------------------------------------
@dataclass
class RunOptions:
    """Run-level choices that are not part of any video's settings."""
    parallel_workers: int = 0              # 0 = automatic
    time_bin_seconds: float = 0.0          # 0 = no time bins
    sample_sheet: Optional[Path] = None
    create_validation_video: bool = False  # default for new setups

    def to_dict(self, base_dir: Optional[Path] = None) -> Dict[str, Any]:
        return {
            "parallel_workers": int(self.parallel_workers),
            "time_bin_seconds": float(self.time_bin_seconds),
            "sample_sheet": store_path(self.sample_sheet, base_dir),
            "create_validation_video": bool(self.create_validation_video),
        }

    @classmethod
    def from_dict(
        cls, data: Optional[Mapping[str, Any]], base_dir: Optional[Path] = None
    ) -> "RunOptions":
        data = data or {}
        workers = int(data.get("parallel_workers", 0))
        bin_seconds = float(data.get("time_bin_seconds", 0.0))
        if workers < 0:
            raise ProjectError("parallel_workers cannot be negative")
        if not bin_seconds >= 0.0 or bin_seconds == float("inf"):
            raise ProjectError("time_bin_seconds must be a finite value >= 0")
        return cls(
            parallel_workers=workers,
            time_bin_seconds=bin_seconds,
            sample_sheet=resolve_stored_path(data.get("sample_sheet"), base_dir),
            create_validation_video=bool(data.get("create_validation_video", False)),
        )


# ----------------------------------------------------------------------
# Layered overrides: trial windows, zone overrides, the trial protocol
# ----------------------------------------------------------------------
class _Unset:
    """Type of :data:`UNSET`."""

    def __repr__(self) -> str:
        return "UNSET"


# Passed to :meth:`Project.set_trial` to clear one field (None leaves it).
UNSET = _Unset()


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _check_number(value: Any, what: str) -> Any:
    """``value`` unchanged if it is a finite number (types are kept so the
    composed settings hash like the ones a full copy would have had)."""
    if not _is_number(value):
        raise ProjectError(f"{what} must be a finite number, not {value!r}")
    return value


def _interval(value: Any, name: str) -> Tuple[Any, Any]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ProjectError(f"per_roi_times[{name!r}] must be a start/end pair")
    return (
        _check_number(value[0], f"per_roi_times[{name!r}] start"),
        _check_number(value[1], f"per_roi_times[{name!r}] end"),
    )


@dataclass
class TrialWindow:
    """When the analysed part of one video starts and ends.

    A layer over the project template: only what the user chose for this
    video is stored. ``start_time`` counts only when ``start_set`` (the user
    chose the start, even if it is 0); otherwise the video starts where the
    template does. ``end_time`` None follows the project's trial protocol
    (start plus its duration) or the template. ``per_roi_times`` (Tail
    Suspension Test: one window per mouse) replaces the template's windows
    when it is not empty.
    """
    start_time: float = 0.0
    end_time: Optional[float] = None
    per_roi_times: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    start_set: bool = False

    def copy(self) -> "TrialWindow":
        return TrialWindow(
            self.start_time, self.end_time,
            {name: (a, b) for name, (a, b) in self.per_roi_times.items()},
            self.start_set,
        )

    def is_empty(self) -> bool:
        """True when it changes nothing (no start, no end, no windows)."""
        return (
            not self.start_set and self.end_time is None
            and not self.per_roi_times
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "per_roi_times": {
                name: [a, b] for name, (a, b) in self.per_roi_times.items()
            },
            "start_set": bool(self.start_set),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrialWindow":
        if not isinstance(data, Mapping):
            raise ProjectError("A trial window must be an object")
        end = data.get("end_time")
        raw_windows = data.get("per_roi_times") or {}
        if not isinstance(raw_windows, Mapping):
            raise ProjectError("per_roi_times must be an object")
        return cls(
            start_time=_check_number(data.get("start_time", 0.0), "start_time"),
            end_time=None if end is None else _check_number(end, "end_time"),
            per_roi_times={
                str(name): _interval(value, str(name))
                for name, value in raw_windows.items()
            },
            start_set=bool(data.get("start_set", False)),
        )


def _check_transform(matrix: Any) -> Optional[List[List[float]]]:
    if matrix is None:
        return None
    try:
        rows = [[float(v) for v in row] for row in matrix]
    except (TypeError, ValueError) as exc:
        raise ProjectError("transform must be a 3x3 matrix") from exc
    if len(rows) != 3 or any(len(row) != 3 for row in rows) or not all(
        math.isfinite(v) for row in rows for v in row
    ):
        raise ProjectError("transform must be a finite 3x3 matrix")
    return rows


@dataclass
class ZoneOverride:
    """The regions of one video when they differ from the template's.

    ``source`` says how they came about: ``adjusted`` (edited by hand) or
    ``aligned`` (the template's regions moved by ``transform``, a 3x3 matrix
    from template pixels to this video's). ``reference`` is the fingerprint
    of the template regions the override was derived from
    (:meth:`Project.zones_reference`), so a later change of the template can
    be noticed. ``scale_factor`` is set when the override also changes the
    calibration (an alignment rescales it); None keeps the template's.
    """
    roi_manager: Any                       # roi_manager.ROIManager
    source: str = ZONE_ADJUSTED
    transform: Optional[List[List[float]]] = None
    reference: Optional[str] = None
    scale_factor: Optional[float] = None

    def copy(self) -> "ZoneOverride":
        return ZoneOverride(
            copy.deepcopy(self.roi_manager), self.source,
            copy.deepcopy(self.transform), self.reference, self.scale_factor,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "roi_manager": self.roi_manager.to_dict(),
            "source": self.source,
            "transform": copy.deepcopy(self.transform),
            "reference": self.reference,
            "scale_factor": self.scale_factor,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ZoneOverride":
        from roi_manager import ROIManager
        if not isinstance(data, Mapping) or "roi_manager" not in data:
            raise ProjectError("A zone override needs its regions")
        source = data.get("source", ZONE_ADJUSTED)
        if source not in ZONE_SOURCES:
            raise ProjectError(f"Unknown zone source {source!r}")
        scale = data.get("scale_factor")
        reference = data.get("reference")
        return cls(
            roi_manager=ROIManager.from_dict(data["roi_manager"]),
            source=source,
            transform=_check_transform(data.get("transform")),
            reference=None if reference is None else str(reference),
            scale_factor=(
                None if scale is None else _check_number(scale, "scale_factor")
            ),
        )


@dataclass
class TrialProtocol:
    """How long a trial lasts and where it starts, for the whole project.

    ``duration_s`` None means "until the video ends". With ``start_mode``
    ``per_video`` (a mouse enters the apparatus at its own moment) every
    video needs its own start before it can be analysed.
    """
    duration_s: Optional[float] = None
    start_mode: str = START_VIDEO

    def __post_init__(self) -> None:
        if self.start_mode not in START_MODES:
            raise ProjectError(f"Unknown start mode {self.start_mode!r}")
        if self.duration_s is not None:
            _check_number(self.duration_s, "duration_s")
            if self.duration_s <= 0:
                raise ProjectError("The trial duration must be above 0")

    def to_dict(self) -> Dict[str, Any]:
        return {"duration_s": self.duration_s, "start_mode": self.start_mode}

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "TrialProtocol":
        data = data or {}
        if not isinstance(data, Mapping):
            raise ProjectError("trial_protocol must be an object")
        return cls(
            duration_s=data.get("duration_s"),
            start_mode=data.get("start_mode", START_VIDEO),
        )


_TIME_KEYS = ("start_time", "end_time", "per_roi_times")
# What a trial and a zone override can express. ``schema_version`` is a
# function of the regions (an arena raises it), so it follows them.
_LAYERED_KEYS = frozenset(
    _TIME_KEYS + ("roi_manager", "scale_factor", "schema_version")
)


def _differs(left: Mapping[str, Any], right: Mapping[str, Any], key: str) -> bool:
    return canonical_json(left.get(key)) != canonical_json(right.get(key))


def decompose_settings(
    template: BatchSettings,
    settings: BatchSettings,
    source: str = SOURCE_CUSTOM,
) -> Optional[Tuple[Optional[TrialWindow], Optional[ZoneOverride]]]:
    """``settings`` as a trial and/or zone override on top of ``template``.

    Returns ``(trial, zones)``, either of which is None when ``settings``
    agrees with the template there; ``(None, None)`` for identical settings.
    Returns None when the settings differ from the template in anything a
    layer cannot express (other fields, an end time or per-mouse windows that
    the template has and the settings lack, a calibration that changed
    without the regions, or an aligned copy whose regions did not move).
    The values are copied untouched so the composition hashes the same.
    """
    wanted, base = settings.to_dict(), template.to_dict()
    for key in (set(wanted) | set(base)) - _LAYERED_KEYS:
        if _differs(wanted, base, key):
            return None
    zones_differ = _differs(wanted, base, "roi_manager")
    scale_differs = _differs(wanted, base, "scale_factor")
    if scale_differs and not zones_differ:
        return None
    if source == SOURCE_ALIGNED and not zones_differ:
        return None
    trial = TrialWindow()
    if _differs(wanted, base, "start_time"):
        trial.start_time, trial.start_set = settings.start_time, True
    if _differs(wanted, base, "end_time"):
        if settings.end_time is None:
            return None
        trial.end_time = settings.end_time
    if _differs(wanted, base, "per_roi_times"):
        if not settings.per_roi_times:
            return None
        trial.per_roi_times = {
            name: (a, b) for name, (a, b) in settings.per_roi_times.items()
        }
    zones = None
    if zones_differ:
        zones = ZoneOverride(
            roi_manager=copy.deepcopy(settings.roi_manager),
            source=ZONE_ALIGNED if source == SOURCE_ALIGNED else ZONE_ADJUSTED,
            scale_factor=settings.scale_factor if scale_differs else None,
        )
    return (None if trial.is_empty() else trial), zones


# ----------------------------------------------------------------------
# Videos
# ----------------------------------------------------------------------
@dataclass
class VideoEntry:
    """One video of the project and everything configured for it."""
    key: str
    path: Path
    metadata: Dict[str, str] = field(default_factory=dict)
    # The stored settings: a mirror of the project template for a video that
    # follows it (``settings_source`` template), or a legacy full copy of
    # its own (custom / aligned). None = not configured. What a run really
    # uses is :meth:`Project.effective_settings`, which layers ``zones`` and
    # ``trial`` on top.
    settings: Optional[BatchSettings] = None
    settings_source: str = SOURCE_TEMPLATE
    verified: bool = False
    # ``get_video_fingerprint`` when the video was configured; a later
    # difference means the file changed (or moved) since.
    fingerprint: Optional[str] = None
    last_status: str = STATUS_NEW
    last_run_dir: Optional[Path] = None
    notes: str = ""
    # Layered overrides: when this video's trial starts and ends, and its
    # regions when they differ from the template's. None = the template's.
    trial: Optional[TrialWindow] = None
    zones: Optional[ZoneOverride] = None
    # Facts about the file, when known (``Project.set_video_info``): they
    # cap a protocol-computed end and check the regions' frame size.
    duration_s: Optional[float] = None
    frame_size: Optional[Tuple[int, int]] = None
    # Settings found in a file that this version cannot read. Kept verbatim
    # and written back, so opening and saving never destroys them.
    _raw_settings: Optional[Dict[str, Any]] = field(
        default=None, init=False, repr=False, compare=False
    )
    # Same for a trial window or zone override that could not be read.
    _raw_trial: Optional[Any] = field(
        default=None, init=False, repr=False, compare=False
    )
    _raw_zones: Optional[Any] = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def is_configured(self) -> bool:
        return self.settings is not None

    def to_dict(
        self,
        base_dir: Optional[Path] = None,
        template_json: Optional[str] = None,
    ) -> Dict[str, Any]:
        settings_value: Any = None
        if self.settings is not None:
            settings_dict = self.settings.to_dict()
            if (
                template_json is not None
                and self.settings_source == SOURCE_TEMPLATE
                and canonical_json(settings_dict) == template_json
            ):
                settings_value = _TEMPLATE_REFERENCE
            else:
                settings_value = settings_dict
        elif self._raw_settings is not None:
            settings_value = self._raw_settings
        return {
            "key": self.key,
            "path": store_path(self.path, base_dir),
            "metadata": dict(self.metadata),
            "settings": settings_value,
            "settings_source": self.settings_source,
            "verified": bool(self.verified),
            "fingerprint": self.fingerprint,
            "last_status": self.last_status,
            "last_run_dir": store_path(self.last_run_dir, base_dir),
            "notes": self.notes,
            "trial": (
                self.trial.to_dict() if self.trial is not None
                else self._raw_trial
            ),
            "zones": (
                self.zones.to_dict() if self.zones is not None
                else self._raw_zones
            ),
            "duration_s": self.duration_s,
            "frame_size": (
                None if self.frame_size is None else list(self.frame_size)
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        base_dir: Optional[Path] = None,
        template: Optional[BatchSettings] = None,
        warnings: Optional[List[str]] = None,
    ) -> "VideoEntry":
        """Rebuild an entry. Unreadable settings leave it unconfigured.

        The reason is appended to ``warnings`` and the raw settings are kept
        so a later save writes them back unchanged.
        """
        try:
            path = resolve_stored_path(data["path"], base_dir)
            key = str(data["key"])
        except KeyError as exc:
            raise ProjectError(f"A video entry has no {exc.args[0]!r}") from exc
        if path is None:
            raise ProjectError(f"Video {key!r} has an empty path")
        source = data.get("settings_source", SOURCE_TEMPLATE)
        status = data.get("last_status", STATUS_NEW)
        entry = cls(
            key=key,
            path=path,
            metadata={
                str(k): "" if v is None else str(v)
                for k, v in (data.get("metadata") or {}).items()
            },
            settings_source=source if source in SETTINGS_SOURCES else SOURCE_CUSTOM,
            verified=bool(data.get("verified", False)),
            fingerprint=data.get("fingerprint"),
            last_status=status if status in STATUSES else STATUS_NEW,
            last_run_dir=resolve_stored_path(data.get("last_run_dir"), base_dir),
            notes=str(data.get("notes") or ""),
        )
        raw = data.get("settings")
        if raw == _TEMPLATE_REFERENCE:
            if template is None:
                raise ProjectError(
                    f"Video {key!r} uses the project template, which is missing"
                )
            entry.settings = template.copy()
        elif raw is not None:
            try:
                entry.settings = BatchSettings.from_dict(raw)
            except Exception as exc:  # unreadable here, maybe valid elsewhere
                entry._raw_settings = raw if isinstance(raw, dict) else None
                if warnings is not None:
                    warnings.append(
                        f"{path.name}: settings could not be read ({exc})"
                    )
        for attribute, kind, reader in (
            ("trial", "trial window", TrialWindow.from_dict),
            ("zones", "zone override", ZoneOverride.from_dict),
        ):
            raw_layer = data.get(attribute)
            if raw_layer is None:
                continue
            try:
                setattr(entry, attribute, reader(raw_layer))
            except Exception as exc:  # keep it verbatim, like unreadable settings
                setattr(entry, f"_raw_{attribute}", raw_layer)
                if warnings is not None:
                    warnings.append(
                        f"{path.name}: the {kind} could not be read ({exc})"
                    )
        entry.duration_s = (
            data["duration_s"] if _is_number(data.get("duration_s"))
            and data["duration_s"] > 0 else None
        )
        size = data.get("frame_size")
        if (
            isinstance(size, (list, tuple)) and len(size) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) and v > 0
                    for v in size)
        ):
            entry.frame_size = (size[0], size[1])
        return entry


# ----------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------
@dataclass
class RunRecord:
    """One analysis run made from the project."""
    run_dir: Path
    started_at: str = ""
    finished_at: str = ""
    cancelled: bool = False
    completed: int = 0
    failed: int = 0
    manifest: Optional[Path] = None
    # Video key -> settings hash the video was analysed with.
    settings_hashes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self, base_dir: Optional[Path] = None) -> Dict[str, Any]:
        return {
            "run_dir": store_path(self.run_dir, base_dir),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "cancelled": bool(self.cancelled),
            "completed": int(self.completed),
            "failed": int(self.failed),
            "manifest": store_path(self.manifest, base_dir),
            "settings_hashes": dict(self.settings_hashes),
        }

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], base_dir: Optional[Path] = None
    ) -> "RunRecord":
        run_dir = resolve_stored_path(data.get("run_dir"), base_dir)
        if run_dir is None:
            raise ProjectError("A run record has no 'run_dir'")
        return cls(
            run_dir=run_dir,
            started_at=str(data.get("started_at") or ""),
            finished_at=str(data.get("finished_at") or ""),
            cancelled=bool(data.get("cancelled", False)),
            completed=int(data.get("completed", 0)),
            failed=int(data.get("failed", 0)),
            manifest=resolve_stored_path(data.get("manifest"), base_dir),
            settings_hashes={
                str(k): str(v)
                for k, v in (data.get("settings_hashes") or {}).items()
            },
        )


# ----------------------------------------------------------------------
# Results of operations
# ----------------------------------------------------------------------
SEVERITY_ERROR = "error"      # a run would fail or be wrong
SEVERITY_WARNING = "warning"  # a run works but something needs a look
SEVERITY_INFO = "info"


@dataclass(frozen=True)
class ProjectIssue:
    """One problem found by :meth:`Project.validate`."""
    severity: str
    code: str            # missing_file, video_changed, unconfigured, ...
    message: str
    key: Optional[str]   # the video concerned; None for project-level issues


@dataclass
class AddVideosResult:
    added: List[VideoEntry] = field(default_factory=list)
    duplicates: List[Path] = field(default_factory=list)


@dataclass
class AnalysisPlan:
    """What :meth:`Project.analysis_plan` hands to the analysis worker."""
    # {video path: its own copy of the settings}, in project order; exactly
    # what ``AnalysisWorker`` / ``BatchRunner`` take as ``analysis_plan``.
    plan: Dict[Path, BatchSettings] = field(default_factory=dict)
    # Videos that have no settings yet (not in ``plan``).
    unconfigured: List[Path] = field(default_factory=list)
    # Configured videos whose file cannot be found (not in ``plan``, unless
    # the plan was built with ``include_missing``).
    missing: List[Path] = field(default_factory=list)
    # {video path: project key} for the videos in ``plan``.
    keys: Dict[Path, str] = field(default_factory=dict)


def read_manifest(run_dir: os.PathLike) -> Dict[str, Any]:
    """The ``manifest.json`` of a run folder (any schema up to 3)."""
    path = Path(run_dir) / MANIFEST_FILENAME
    try:
        with open(path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ProjectError(f"{path} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise ProjectError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ProjectError(f"{path} is not a run manifest")
    version = manifest.get("schema_version", 1)
    if not isinstance(version, int) or version > MANIFEST_SCHEMA_VERSION:
        raise ProjectError(
            f"{path.name} has schema {version}, newer than the supported "
            f"schema {MANIFEST_SCHEMA_VERSION}"
        )
    return manifest


def _plan_video_paths(run_dir: Path) -> List[str]:
    """Normalised paths of the videos listed in a run's ``run_plan.json``."""
    try:
        with open(run_dir / RUN_PLAN_FILENAME, "r", encoding="utf-8") as handle:
            plan = json.load(handle).get("plan") or {}
        return [norm_path(resolve_stored_path(key, run_dir)) for key in plan]
    except (OSError, ValueError, AttributeError, TypeError):
        return []


# ----------------------------------------------------------------------
# The project
# ----------------------------------------------------------------------
@dataclass
class Project:
    name: str = "Untitled project"
    created_at: str = field(default_factory=_now)
    modified_at: str = field(default_factory=_now)
    app_version: str = APP_VERSION
    paradigm: str = ""                       # maze name; "" = not chosen yet
    algorithm_parameters: Dict[str, Any] = field(default_factory=dict)
    run_options: RunOptions = field(default_factory=RunOptions)
    # The shared configuration new videos inherit until they are customised.
    template_settings: Optional[BatchSettings] = None
    # How long a trial lasts and whether every video has its own start.
    trial_protocol: TrialProtocol = field(default_factory=TrialProtocol)
    videos: List[VideoEntry] = field(default_factory=list)
    runs: List[RunRecord] = field(default_factory=list)
    notes: str = ""
    # Runtime only (never serialized): where the project lives on disk, what
    # went wrong while loading it, and who wants to hear about changes.
    file_path: Optional[Path] = field(default=None, repr=False, compare=False)
    load_warnings: List[str] = field(
        default_factory=list, repr=False, compare=False
    )
    _listeners: List[Callable[[], None]] = field(
        default_factory=list, init=False, repr=False, compare=False
    )

    schema_version = SCHEMA_VERSION  # class attribute, not a dataclass field

    # -- location and keys ---------------------------------------------
    @property
    def base_dir(self) -> Optional[Path]:
        """Folder of the project file; None until the project is saved."""
        if self.file_path is None:
            return None
        return Path(self.file_path).resolve().parent

    def key_for(self, video_path: Path) -> str:
        """Stable key of a video: its path relative to the project file."""
        video = Path(video_path).resolve()
        if self.file_path is not None:
            from core.batch_plan_keys import plan_key_for
            return plan_key_for(video, Path(self.file_path))
        return video.as_posix()

    # -- change notification -------------------------------------------
    def add_listener(self, callback: Callable[[], None]) -> None:
        """Call ``callback()`` after every change made through the API."""
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[], None]) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _changed(self, touch: bool = True) -> None:
        if touch:
            self.modified_at = _now()
        for callback in list(self._listeners):
            try:
                callback()
            except Exception:
                logger.warning("Project listener failed", exc_info=True)

    # -- lookups ---------------------------------------------------------
    def entry(self, key: str) -> VideoEntry:
        for video in self.videos:
            if video.key == key:
                return video
        raise KeyError(key)

    def find(self, video_path: Path) -> Optional[VideoEntry]:
        """The entry for a video path (any spelling), or None."""
        target = norm_path(video_path)
        for video in self.videos:
            if norm_path(video.path) == target:
                return video
        return None

    def keys(self) -> List[str]:
        return [video.key for video in self.videos]

    def metadata_columns(self) -> List[str]:
        """Metadata names used by any video, in first-seen order."""
        columns: List[str] = []
        seen = set()
        for video in self.videos:
            for name in video.metadata:
                if name not in seen:
                    seen.add(name)
                    columns.append(name)
        return columns

    # -- serialization ---------------------------------------------------
    def to_dict(self, base_dir: Optional[Path] = None) -> Dict[str, Any]:
        """The ``.bmaproj`` payload; paths are relative to ``base_dir``."""
        template_json = _settings_to_json(self.template_settings)
        return {
            "type": FILE_TYPE,
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "app_version": self.app_version,
            "paradigm": self.paradigm,
            "algorithm_parameters": copy.deepcopy(self.algorithm_parameters),
            "run_options": self.run_options.to_dict(base_dir),
            "template_settings": (
                None if self.template_settings is None
                else self.template_settings.to_dict()
            ),
            "trial_protocol": self.trial_protocol.to_dict(),
            "notes": self.notes,
            "videos": [
                video.to_dict(base_dir, template_json) for video in self.videos
            ],
            "runs": [run.to_dict(base_dir) for run in self.runs],
        }

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], base_dir: Optional[Path] = None
    ) -> "Project":
        """Rebuild a project; paths resolve against ``base_dir``.

        Raises :class:`ProjectError` for a file that is not a project or comes
        from a newer schema than this build understands.
        """
        if not isinstance(data, Mapping) or data.get("type") != FILE_TYPE:
            raise ProjectError("The file is not a project")
        try:
            version = int(data.get("schema_version", 1))
        except (TypeError, ValueError) as exc:
            raise ProjectError("The project has no valid schema_version") from exc
        if version > SCHEMA_VERSION:
            raise ProjectError(
                f"Project schema {version} is newer than the supported "
                f"schema {SCHEMA_VERSION}; update the application to open it"
            )
        warnings: List[str] = []
        template = None
        raw_template = data.get("template_settings")
        if raw_template is not None:
            try:
                template = BatchSettings.from_dict(raw_template)
            except Exception as exc:
                raise ProjectError(
                    f"The project template settings cannot be read: {exc}"
                ) from exc
        project = cls(
            name=str(data.get("name") or "Untitled project"),
            created_at=str(data.get("created_at") or _now()),
            modified_at=str(data.get("modified_at") or _now()),
            app_version=str(data.get("app_version") or ""),
            paradigm=str(data.get("paradigm") or ""),
            algorithm_parameters=copy.deepcopy(
                dict(data.get("algorithm_parameters") or {})
            ),
            run_options=RunOptions.from_dict(data.get("run_options"), base_dir),
            template_settings=template,
            trial_protocol=TrialProtocol.from_dict(data.get("trial_protocol")),
            notes=str(data.get("notes") or ""),
        )
        seen = set()
        for raw in data.get("videos") or []:
            video = VideoEntry.from_dict(raw, base_dir, template, warnings)
            if video.key in seen:
                raise ProjectError(f"Duplicate video key {video.key!r}")
            seen.add(video.key)
            project.videos.append(video)
        project.runs = [
            RunRecord.from_dict(raw, base_dir) for raw in data.get("runs") or []
        ]
        if version < 2:
            project._migrate_full_copies(warnings)
        project.load_warnings = warnings
        return project

    def _migrate_full_copies(self, warnings: List[str]) -> None:
        """Schema 1 stored every per-video change as a full copy of the
        settings. A copy that differs from the template only in its times
        and/or regions becomes a trial and/or zone override; any other copy
        stays as it is. The change is reported once in ``warnings``."""
        moved: List[str] = []
        for video in self.videos:
            if video.settings is None or video.settings_source == SOURCE_TEMPLATE:
                continue
            if self._store_as_layers(video, video.settings, video.settings_source):
                moved.append(video.name)
        if moved:
            shown = ", ".join(moved[:5]) + (
                f" and {len(moved) - 5} more" if len(moved) > 5 else ""
            )
            warnings.append(
                f"Updated to the new project format: {len(moved)} video(s) "
                "that differed from the template only in their start/end "
                f"times or regions now follow the template ({shown}). "
                "Their settings and results are unchanged."
            )

    def _store_as_layers(
        self, entry: VideoEntry, settings: BatchSettings, source: str
    ) -> bool:
        """Store ``settings`` as the template plus a trial and/or zone override.

        Only done when the settings differ from the template in nothing but
        their times and/or regions, and when the composition then gives
        exactly the settings it replaces (``to_dict`` and so the hash).
        Returns False, leaving ``entry`` untouched, when that is not so.
        """
        template = self.template_settings
        if template is None:
            return False
        layers = decompose_settings(template, settings, source)
        if layers is None or layers == (None, None):
            return False
        trial, zones = layers
        if zones is not None:
            zones.reference = self.zones_reference()
        candidate = VideoEntry(
            key=entry.key, path=entry.path, settings=template.copy(),
            settings_source=SOURCE_TEMPLATE, trial=trial, zones=zones,
            duration_s=entry.duration_s, frame_size=entry.frame_size,
        )
        wanted = _settings_to_json(self.effective_settings(
            VideoEntry(key=entry.key, path=entry.path, settings=settings,
                       settings_source=SOURCE_CUSTOM)
        ))
        if _settings_to_json(self.effective_settings(candidate)) != wanted:
            return False
        entry.settings = candidate.settings
        entry.settings_source = SOURCE_TEMPLATE
        entry._raw_settings = None
        entry.trial, entry.zones = trial, zones
        return True

    # ------------------------------------------------------------------
    # Videos
    # ------------------------------------------------------------------
    def entries_for(self, keys: Iterable[str]) -> List[VideoEntry]:
        """Entries for ``keys`` in project order; unknown keys raise KeyError."""
        wanted = list(keys)
        known = {video.key for video in self.videos}
        for key in wanted:
            if key not in known:
                raise KeyError(key)
        wanted_set = set(wanted)
        return [video for video in self.videos if video.key in wanted_set]

    def _unique_key(self, key: str, taken: Iterable[str]) -> str:
        used = set(taken)
        candidate, sequence = key, 2
        while candidate in used:
            candidate = f"{key}#{sequence}"
            sequence += 1
        return candidate

    def add_videos(self, paths: Iterable[os.PathLike]) -> AddVideosResult:
        """Append videos, skipping any already in the project.

        Duplicates are detected by resolved path (also within ``paths``) and
        reported, never added twice. Order is kept. Each new video inherits
        the project template when there is one (status ``configured``),
        otherwise it is ``new`` and not configured.
        """
        result = AddVideosResult()
        known = {norm_path(video.path) for video in self.videos}
        taken = {video.key for video in self.videos}
        for raw in paths:
            path = Path(raw).resolve()
            marker = norm_path(path)
            if marker in known:
                result.duplicates.append(path)
                continue
            known.add(marker)
            key = self._unique_key(self.key_for(path), taken)
            taken.add(key)
            entry = VideoEntry(key=key, path=path)
            if self.template_settings is not None:
                self._configure(entry, self.template_settings, SOURCE_TEMPLATE)
            self.videos.append(entry)
            result.added.append(entry)
        if result.added:
            self._changed()
        return result

    def remove_videos(self, keys: Iterable[str]) -> List[VideoEntry]:
        """Remove videos; returns the removed entries (project order)."""
        drop = set(keys)
        removed = [video for video in self.videos if video.key in drop]
        if removed:
            self.videos = [v for v in self.videos if v.key not in drop]
            self._changed()
        return removed

    def move_videos(self, keys: Iterable[str], offset: int) -> bool:
        """Move the selected videos ``offset`` rows (negative = up).

        A block of selected rows moves together and stops at the list edge.
        Returns True if anything moved.
        """
        selected = set(keys)
        step = 1 if offset > 0 else -1
        moved_any = False
        for _ in range(abs(int(offset))):
            moved = False
            count = len(self.videos)
            order = range(count - 1, -1, -1) if step > 0 else range(count)
            for index in order:
                target = index + step
                if (
                    self.videos[index].key in selected
                    and 0 <= target < count
                    and self.videos[target].key not in selected
                ):
                    self.videos[index], self.videos[target] = (
                        self.videos[target], self.videos[index]
                    )
                    moved = True
            if not moved:
                break
            moved_any = True
        if moved_any:
            self._changed()
        return moved_any

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def _configure(
        self,
        entry: VideoEntry,
        settings: BatchSettings,
        source: str,
        verified: bool = False,
    ) -> bool:
        """Give ``entry`` a copy of ``settings``; True if they differ.

        Different settings invalidate the verification and any earlier
        result (status back to ``configured``, the last run stays listed).
        Identical settings only update the source.
        """
        same = (
            entry.settings is not None
            and canonical_json(entry.settings.to_dict())
            == canonical_json(settings.to_dict())
        )
        entry.settings_source = source
        entry._raw_settings = None
        if same:
            entry.verified = entry.verified or verified
            if entry.fingerprint is None:
                entry.fingerprint = video_fingerprint(entry.path)
            return False
        entry.settings = settings.copy()
        entry.verified = verified
        entry.fingerprint = video_fingerprint(entry.path)
        entry.last_status = STATUS_CONFIGURED
        return True

    def set_template(
        self, settings: Optional[BatchSettings], update_inheriting: bool = True
    ) -> List[str]:
        """Set the shared configuration (None clears it).

        With ``update_inheriting`` every video that has no settings yet or
        still uses the template receives the new one; customised and aligned
        videos are left alone. The paradigm and algorithm parameters of an
        empty project are taken from the template. Returns the keys updated.
        """
        self.template_settings = None if settings is None else settings.copy()
        updated: List[str] = []
        if settings is not None:
            if not self.paradigm and settings.paradigm_id:
                self.paradigm = settings.paradigm_id
            if not self.algorithm_parameters and settings.algorithm_parameters:
                self.algorithm_parameters = copy.deepcopy(
                    settings.algorithm_parameters
                )
            if update_inheriting:
                for video in self.videos:
                    if video.settings is None or video.settings_source == SOURCE_TEMPLATE:
                        self._configure(video, settings, SOURCE_TEMPLATE)
                        updated.append(video.key)
        self._changed()
        return updated

    def has_configured_videos(self) -> bool:
        """True if any video (or the template) already has settings."""
        return self.template_settings is not None or any(
            video.settings is not None or video._raw_settings is not None
            for video in self.videos
        )

    def set_paradigm(self, paradigm: str) -> List[str]:
        """Choose the project's paradigm.

        Settings are paradigm specific, so *changing* the paradigm of a
        project that has configured videos clears them: the template, the
        algorithm parameters and every video's settings, verification and
        fingerprint (videos go back to ``new``; their run history stays).
        Choosing the first paradigm, or the one already effective, clears
        nothing. Returns the keys of the videos that lost their settings.
        """
        paradigm = paradigm or ""
        cleared: List[str] = []
        current = self.effective_paradigm()
        if paradigm != current and self.has_configured_videos():
            self.template_settings = None
            self.algorithm_parameters = {}
            for video in self.videos:
                if video.settings is not None or video._raw_settings is not None:
                    cleared.append(video.key)
                video.settings = None
                video._raw_settings = None
                video.zones = None      # regions are paradigm specific
                if video.trial is not None:
                    video.trial.per_roi_times = {}
                    if video.trial.is_empty():
                        video.trial = None
                video.settings_source = SOURCE_TEMPLATE
                video.verified = False
                video.fingerprint = None
                video.last_status = STATUS_NEW
        if paradigm != self.paradigm or cleared:
            self.paradigm = paradigm
            self._changed()
        return cleared

    def apply_template_to(
        self, keys: Optional[Iterable[str]] = None, keep_trials: bool = True
    ) -> List[str]:
        """Reset videos to the template (all videos when ``keys`` is None).

        Legacy full copies and zone overrides are dropped. Trial windows
        (when each mouse entered) are data about the recording rather than
        configuration, so they are kept unless ``keep_trials`` is False.
        """
        if self.template_settings is None:
            raise ProjectError("The project has no template settings")
        entries = self.videos if keys is None else self.entries_for(keys)
        for video in entries:
            had_zones = video.zones is not None
            video.zones = None
            if not keep_trials:
                video.trial = None
            changed = self._configure(
                video, self.template_settings, SOURCE_TEMPLATE
            )
            if had_zones and not changed:
                video.verified = False
                video.last_status = STATUS_CONFIGURED
        if entries:
            self._changed()
        return [video.key for video in entries]

    def customize(
        self,
        key: str,
        settings: BatchSettings,
        source: str = SOURCE_CUSTOM,
        verified: bool = False,
    ) -> VideoEntry:
        """Give one video its own full settings (``source``: custom or aligned).

        This is the legacy full copy: it forks every setting from the
        template. To change only a video's times or regions, use
        :meth:`set_trial` and :meth:`set_zone_override`, which keep it
        following the rest of the template. Because ``settings`` are final,
        the video's zone override is dropped (the copy carries its regions)
        and an existing trial is rewritten to the copy's own times.
        """
        if source not in (SOURCE_CUSTOM, SOURCE_ALIGNED):
            raise ProjectError(f"Unknown settings source {source!r}")
        entry = self.entry(key)
        entry.zones = None
        entry._raw_zones = None
        if entry.trial is not None:
            entry.trial = TrialWindow(
                start_time=settings.start_time,
                end_time=settings.end_time,
                per_roi_times=dict(settings.per_roi_times),
                start_set=entry.trial.start_set,
            )
        self._configure(entry, settings, source, verified)
        self._changed()
        return entry

    def set_verified(self, keys: Iterable[str], verified: bool = True) -> None:
        """Mark videos as checked by the user (ROIs confirmed on the frame)."""
        for video in self.entries_for(keys):
            video.verified = bool(verified)
        self._changed()

    def accept_video_changes(self, keys: Optional[Iterable[str]] = None) -> List[str]:
        """Record the current fingerprint of videos that changed on disk."""
        entries = self.videos if keys is None else self.entries_for(keys)
        accepted = []
        for video in entries:
            current = video_fingerprint(video.path)
            if current is not None and current != video.fingerprint:
                video.fingerprint = current
                accepted.append(video.key)
        if accepted:
            self._changed()
        return accepted

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def set_metadata_value(self, key: str, column: str, value: str) -> None:
        entry = self.entry(key)
        if entry.metadata.get(column) != value or column not in entry.metadata:
            entry.metadata[column] = value
            self._changed()

    def add_metadata_column(self, name: str) -> bool:
        """Add an empty column to every video; False if it already exists."""
        name = name.strip()
        if not name:
            raise ProjectError("A metadata column needs a name")
        if name in self.metadata_columns():
            return False
        for video in self.videos:
            video.metadata[name] = ""
        self._changed()
        return True

    def remove_metadata_column(self, name: str) -> bool:
        found = False
        for video in self.videos:
            found = video.metadata.pop(name, None) is not None or found
        if found:
            self._changed()
        return found

    def set_video_notes(self, key: str, notes: str) -> None:
        entry = self.entry(key)
        if entry.notes != notes:
            entry.notes = notes
            self._changed()

    def set_metadata_from_sheet(self, sheet_path: os.PathLike) -> SampleSheetMatch:
        """Fill metadata from a sample sheet CSV.

        Matching follows ``core.sample_sheet`` (never by guesswork). Matched
        videos get the sheet's columns (other metadata is kept); videos with
        no row, or an ambiguous one, keep what they have and are listed in
        the returned match's ``unmatched`` / ``ambiguous``. The sheet becomes
        the run option ``sample_sheet``. Raises ``SampleSheetError`` if the
        sheet cannot be read.
        """
        sheet_file = Path(sheet_path).resolve()
        sheet = load_sample_sheet(sheet_file)
        match = metadata_for_videos(
            sheet, sheet_file, [video.path for video in self.videos]
        )
        for video in self.videos:
            values = match.metadata.by_video.get(video.path)
            if values is not None:
                video.metadata.update(values)
        self.run_options.sample_sheet = sheet_file
        self._changed()
        return match

    def sample_metadata(self) -> Optional[SampleMetadata]:
        """The project's metadata as the ``SampleMetadata`` a run expects.

        None when no video has metadata. The ``sheet_path`` is the run
        option sample sheet, or the project file when there is none.
        """
        columns = tuple(self.metadata_columns())
        if not columns:
            return None
        anchor = self.run_options.sample_sheet or self.file_path or Path(".")
        return SampleMetadata(
            sheet_path=Path(anchor),
            columns=columns,
            by_video={
                video.path: {c: video.metadata.get(c, "") for c in columns}
                for video in self.videos
            },
        )

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------
    def set_run_options(
        self,
        parallel_workers: Optional[int] = None,
        time_bin_seconds: Optional[float] = None,
        create_validation_video: Optional[bool] = None,
    ) -> bool:
        """Change run-level options (None leaves one alone); True if any changed."""
        options = self.run_options
        changed = False
        if parallel_workers is not None:
            if int(parallel_workers) < 0:
                raise ProjectError("parallel_workers cannot be negative")
            if int(parallel_workers) != options.parallel_workers:
                options.parallel_workers = int(parallel_workers)
                changed = True
        if time_bin_seconds is not None:
            seconds = float(time_bin_seconds)
            if not seconds >= 0.0 or seconds == float("inf"):
                raise ProjectError("time_bin_seconds must be a finite value >= 0")
            if seconds != options.time_bin_seconds:
                options.time_bin_seconds = seconds
                changed = True
        if (
            create_validation_video is not None
            and bool(create_validation_video) != options.create_validation_video
        ):
            options.create_validation_video = bool(create_validation_video)
            changed = True
        if changed:
            self._changed()
        return changed

    def effective_paradigm(self) -> str:
        """The project's paradigm, or the first one its settings name."""
        if self.paradigm:
            return self.paradigm
        if self.template_settings is not None and self.template_settings.paradigm_id:
            return self.template_settings.paradigm_id
        for video in self.videos:
            if video.settings is not None and video.settings.paradigm_id:
                return video.settings.paradigm_id
        return ""

    # ------------------------------------------------------------------
    # Template + layered overrides
    # ------------------------------------------------------------------
    def _entry_of(self, key: Union[str, VideoEntry]) -> VideoEntry:
        return key if isinstance(key, VideoEntry) else self.entry(key)

    @staticmethod
    def _is_full_copy(entry: VideoEntry) -> bool:
        """True for a legacy full copy (custom / aligned source)."""
        return entry.settings is not None and entry.settings_source != SOURCE_TEMPLATE

    def _protocol_sets_end(self, entry: VideoEntry) -> bool:
        """Does the protocol duration decide this video's end?

        It does for every video built from the template, and for a legacy
        full copy only once it has a trial (a copy keeps its own end).
        """
        return self.trial_protocol.duration_s is not None and (
            entry.trial is not None or not self._is_full_copy(entry)
        )

    def _composed_times(self, entry: VideoEntry) -> Tuple[Any, Any, Dict]:
        """``(start, end, per_roi_times)`` of the entry's effective settings.

        The base is the stored settings (the template or the legacy copy),
        then the trial: the start when it was set, the end when it has one,
        otherwise start plus the protocol duration (capped at the video
        duration when that is known), and the per-mouse windows when it
        has some. Values are passed through untouched.
        """
        base = entry.settings if entry.settings is not None else self.template_settings
        if base is None:
            start, end, windows = 0.0, None, {}
        else:
            start, end, windows = base.start_time, base.end_time, base.per_roi_times
        trial = entry.trial
        if trial is not None and trial.start_set:
            start = trial.start_time
        if trial is not None and trial.end_time is not None:
            end = trial.end_time
        elif self._protocol_sets_end(entry):
            end = start + self.trial_protocol.duration_s
            known = entry.duration_s
            if known is not None and _is_number(start) and end > known:
                end = known if known > start else None
        if trial is not None and trial.per_roi_times:
            windows = trial.per_roi_times
        return start, end, {name: tuple(v) for name, v in windows.items()}

    def effective_settings(
        self, key: Union[str, VideoEntry]
    ) -> Optional[BatchSettings]:
        """The settings a run of this video uses; None if it has none.

        Composed in layers: the stored settings (a copy of the template, or
        the video's legacy full copy), then its zone override (regions and
        calibration), then its trial (start, end, per-mouse windows; see
        :meth:`_composed_times`), then the project's paradigm and algorithm
        parameters where the settings carry none. The result is a fresh
        copy, equal (``to_dict`` and settings hash) to the full copy that
        had the same values written into it. Times that are not valid are
        still applied; :meth:`trial_status` and :meth:`validate` report them.
        """
        entry = self._entry_of(key)
        if entry.settings is None:
            return None
        settings = entry.settings.copy()
        zones = entry.zones
        if zones is not None:
            settings.roi_manager = copy.deepcopy(zones.roi_manager)
            if zones.scale_factor is not None:
                settings.scale_factor = zones.scale_factor
        start, end, windows = self._composed_times(entry)
        settings.start_time = start
        settings.end_time = end
        settings.per_roi_times = windows
        if not settings.paradigm_id and self.paradigm:
            settings.paradigm_id = self.paradigm
        if not settings.algorithm_parameters and self.algorithm_parameters:
            settings.algorithm_parameters = copy.deepcopy(self.algorithm_parameters)
        return settings

    def zones_reference(self) -> Optional[str]:
        """Fingerprint of the template's regions (None without a template).

        Recorded in each zone override, so a later edit of the template's
        regions can be told apart from an override that is still current.
        """
        if self.template_settings is None:
            return None
        return hashlib.sha1(
            canonical_json(self.template_settings.roi_manager.to_dict()).encode()
        ).hexdigest()[:16]

    def zone_status(self, key: Union[str, VideoEntry]) -> str:
        """Where the video's regions come from.

        ``adjusted`` / ``aligned``: it has a zone override of that kind.
        ``custom``: a legacy full copy (``aligned`` when it came from an
        alignment). ``template``: the project template's regions.
        """
        entry = self._entry_of(key)
        if entry.zones is not None:
            return entry.zones.source
        if self._is_full_copy(entry):
            if entry.settings_source == SOURCE_ALIGNED:
                return ZONES_ALIGNED
            return ZONES_CUSTOM
        return ZONES_TEMPLATE

    def trial_span(
        self, key: Union[str, VideoEntry]
    ) -> Optional[Tuple[Any, Optional[Any]]]:
        """``(start, end)`` in seconds of what is analysed; None if unconfigured.

        ``end`` None means "until the video ends". With per-mouse windows
        (Tail Suspension Test) it is the span from the earliest start to the
        latest end. The values may be invalid; see :meth:`trial_status`.
        """
        entry = self._entry_of(key)
        if entry.settings is None:
            return None
        start, end, windows = self._composed_times(entry)
        if windows:
            return (
                min(a for a, _ in windows.values()),
                max(b for _, b in windows.values()),
            )
        return start, end

    def _has_start(self, entry: VideoEntry) -> bool:
        if entry.trial is not None and entry.trial.start_set:
            return True
        # A legacy copy made before trials existed: a start other than 0 was
        # chosen by hand.
        return (
            self._is_full_copy(entry)
            and _is_number(entry.settings.start_time)
            and entry.settings.start_time > 0
        )

    def trial_problem(self, key: Union[str, VideoEntry]) -> Optional[str]:
        """Why the video's effective times cannot be analysed (None if fine)."""
        entry = self._entry_of(key)
        start, end, windows = self._composed_times(entry)
        if not _is_number(start) or start < 0:
            return "the start time must be 0 or more"
        if end is not None and (not _is_number(end) or end <= start):
            return "the end time must be after the start time"
        for name, (a, b) in windows.items():
            if not _is_number(a) or not _is_number(b) or a < 0 or b <= a:
                return f"the window of {name} is not a valid interval"
        if entry.duration_s is not None and start >= entry.duration_s:
            return (
                f"the start time is beyond the end of the video "
                f"({entry.duration_s:g} s)"
            )
        return None

    def trial_status(self, key: Union[str, VideoEntry]) -> str:
        """``ok``, ``needs_start`` or ``invalid`` for the video's trial.

        ``needs_start``: the protocol gives every video its own start and
        this one has none yet. ``invalid``: the times cannot be analysed
        (see :meth:`trial_problem`).
        """
        entry = self._entry_of(key)
        if self.trial_problem(entry) is not None:
            return TRIAL_INVALID
        if (
            self.trial_protocol.start_mode == START_PER_VIDEO
            and not self._has_start(entry)
        ):
            return TRIAL_NEEDS_START
        return TRIAL_OK

    def _effective_json(self, entry: VideoEntry) -> Optional[str]:
        return _settings_to_json(self.effective_settings(entry))

    def _mark_changed_settings(
        self, entry: VideoEntry, before: Optional[str], reset_verified: bool
    ) -> bool:
        """Effective settings changed: results are stale, verification may be."""
        if entry.settings is None or self._effective_json(entry) == before:
            return False
        if reset_verified:
            entry.verified = False
        entry.last_status = STATUS_CONFIGURED
        return True

    def set_trial(
        self,
        key: str,
        start: Union[float, None, _Unset] = None,
        end: Union[float, None, _Unset] = None,
        per_roi_times: Union[
            Mapping[str, Tuple[float, float]], None, _Unset
        ] = None,
    ) -> Optional[TrialWindow]:
        """Set when the trial of one video starts and ends (seconds).

        Each argument that is None is left as it is; :data:`UNSET` clears it
        (the start goes back to the template's, the end to the protocol or
        template, the windows to the template's). Setting a start counts as
        the user having chosen it, also when it is 0. Only finite numbers
        are accepted; whether the times make sense is reported by
        :meth:`trial_status`, so a value being typed can be kept. Returns
        the video's trial (None when nothing is left of it).
        """
        entry = self.entry(key)
        before = self._effective_json(entry)
        trial = entry.trial.copy() if entry.trial is not None else TrialWindow()
        if start is UNSET:
            trial.start_time, trial.start_set = 0.0, False
        elif start is not None:
            trial.start_time = _check_number(start, "start")
            trial.start_set = True
        if end is UNSET:
            trial.end_time = None
        elif end is not None:
            trial.end_time = _check_number(end, "end")
        if per_roi_times is UNSET:
            trial.per_roi_times = {}
        elif per_roi_times is not None:
            trial.per_roi_times = {
                str(name): _interval(value, str(name))
                for name, value in per_roi_times.items()
            }
        entry.trial = None if trial.is_empty() else trial
        entry._raw_trial = None
        self._mark_changed_settings(entry, before, reset_verified=False)
        self._changed()
        return entry.trial

    def clear_trial(self, key: str) -> bool:
        """Drop the video's own trial times; True if it had any."""
        entry = self.entry(key)
        if entry.trial is None and entry._raw_trial is None:
            return False
        before = self._effective_json(entry)
        entry.trial = None
        entry._raw_trial = None
        self._mark_changed_settings(entry, before, reset_verified=False)
        self._changed()
        return True

    def set_trial_protocol(
        self,
        duration_s: Optional[float] = None,
        start_mode: str = START_VIDEO,
    ) -> List[str]:
        """Set the project's trial protocol.

        ``duration_s`` None means "until the video ends"; otherwise a video
        without an explicit end of its own ends that long after its start.
        ``start_mode`` ``per_video`` makes every video need its own start.
        Returns the keys of the videos whose effective settings changed.
        """
        protocol = TrialProtocol(duration_s, start_mode)   # validates
        before = {v.key: self._effective_json(v) for v in self.videos}
        self.trial_protocol = protocol
        changed = [
            v.key for v in self.videos
            if self._mark_changed_settings(v, before[v.key], reset_verified=False)
        ]
        self._changed()
        return changed

    def apply_protocol_to_all(self) -> List[str]:
        """Make every video follow the protocol again.

        Explicit ends are dropped, so each trial ends at its start plus the
        protocol duration (or at the template's end); in ``video_start``
        mode the explicit starts go too. Per-mouse windows are kept.
        Returns the keys that changed.
        """
        before = {v.key: self._effective_json(v) for v in self.videos}
        reset_start = self.trial_protocol.start_mode == START_VIDEO
        changed: List[str] = []
        for entry in self.videos:
            trial = entry.trial
            if trial is None:
                continue
            trial = trial.copy()
            trial.end_time = None
            if reset_start:
                trial.start_time, trial.start_set = 0.0, False
            entry.trial = None if trial.is_empty() else trial
            if self._mark_changed_settings(entry, before[entry.key], False):
                changed.append(entry.key)
        self._changed()
        return changed

    def set_video_info(
        self,
        key: str,
        duration_s: Optional[float] = None,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Record what is known about a video file (None leaves a value).

        ``duration_s`` caps a protocol-computed end and rejects a start
        beyond the video; ``frame_size`` (width, height) lets
        :meth:`validate` check the regions were drawn on that size.
        """
        entry = self.entry(key)
        before = self._effective_json(entry)
        if duration_s is not None:
            if _check_number(duration_s, "duration_s") <= 0:
                raise ProjectError("duration_s must be above 0")
            entry.duration_s = duration_s
        if frame_size is not None:
            try:
                width, height = (int(v) for v in frame_size)
            except (TypeError, ValueError) as exc:
                raise ProjectError("frame_size must be (width, height)") from exc
            if width <= 0 or height <= 0:
                raise ProjectError("frame_size must be positive")
            entry.frame_size = (width, height)
        self._mark_changed_settings(entry, before, reset_verified=False)
        self._changed()

    def set_zone_override(
        self,
        key: str,
        roi_manager: Any,
        source: str = ZONE_ADJUSTED,
        transform: Optional[Any] = None,
        scale_factor: Optional[float] = None,
    ) -> ZoneOverride:
        """Give one video its own regions; everything else follows the template.

        ``roi_manager`` is copied. ``source`` is ``adjusted`` (edited) or
        ``aligned`` (the template's regions moved by ``transform``, a 3x3
        matrix from template pixels to this video's). ``scale_factor`` is the
        video's calibration when it differs from the template's; for an
        aligned override it defaults to the template's rescaled to the
        moved reference line, as an alignment does. The video loses its
        verification (the regions changed) and any earlier result.
        """
        if source not in ZONE_SOURCES:
            raise ProjectError(f"Unknown zone source {source!r}")
        if scale_factor is not None:
            if _check_number(scale_factor, "scale_factor") < 0:
                raise ProjectError("scale_factor cannot be negative")
        entry = self.entry(key)
        before = self._effective_json(entry)
        template = self.template_settings
        manager = copy.deepcopy(roi_manager)
        if source == ZONE_ALIGNED and scale_factor is None and template is not None:
            from core.roi_alignment import scale_factor_after_alignment
            scale_factor = scale_factor_after_alignment(
                template.scale_factor, template.roi_manager, manager
            )
            if scale_factor == template.scale_factor:
                scale_factor = None
        entry.zones = ZoneOverride(
            roi_manager=manager,
            source=source,
            transform=_check_transform(transform),
            reference=self.zones_reference(),
            scale_factor=scale_factor,
        )
        entry._raw_zones = None
        self._mark_changed_settings(entry, before, reset_verified=True)
        entry.verified = False
        self._changed()
        return entry.zones

    def clear_zone_override(self, key: str) -> bool:
        """Back to the template's regions; True if there was an override."""
        entry = self.entry(key)
        if entry.zones is None and entry._raw_zones is None:
            return False
        before = self._effective_json(entry)
        entry.zones = None
        entry._raw_zones = None
        self._mark_changed_settings(entry, before, reset_verified=True)
        entry.verified = False
        self._changed()
        return True

    def analysis_plan(
        self,
        keys: Optional[Iterable[str]] = None,
        include_missing: bool = False,
    ) -> AnalysisPlan:
        """The plan ``AnalysisWorker`` / ``BatchRunner`` expect.

        ``plan`` maps the video path to its own *copy* of the settings (the
        worker stamps them), in project order, for every configured video
        whose file exists. Videos without settings are listed in
        ``unconfigured`` and configured videos whose file is gone in
        ``missing`` (also put in ``plan`` when ``include_missing``, which is
        what exporting a plan wants). ``keys`` limits the plan to those
        videos.
        """
        entries = self.videos if keys is None else self.entries_for(keys)
        result = AnalysisPlan()
        for video in entries:
            if video.settings is None:
                result.unconfigured.append(video.path)
                continue
            exists = video.path.is_file()
            if not exists:
                result.missing.append(video.path)
            if exists or include_missing:
                result.plan[video.path] = self.effective_settings(video)
                result.keys[video.path] = video.key
        return result

    def _override_issues(
        self, video: VideoEntry, zones_now: Optional[str]
    ) -> List[ProjectIssue]:
        """Problems of a video's trial window and zone override."""
        issues: List[ProjectIssue] = []
        problem = self.trial_problem(video)
        if problem is not None:
            issues.append(ProjectIssue(
                SEVERITY_ERROR, "trial_invalid",
                f"{video.name}: {problem}", video.key,
            ))
        elif self.trial_status(video) == TRIAL_NEEDS_START:
            issues.append(ProjectIssue(
                SEVERITY_ERROR, "trial_needs_start",
                f"{video.name} needs its start time (each video has its own "
                "start in this project)", video.key,
            ))
        if video.settings is None:
            return issues
        manager = (
            video.zones.roi_manager if video.zones is not None
            else video.settings.roi_manager
        )
        drawn = getattr(manager, "source_dimensions", None)
        if (
            video.frame_size is not None and drawn is not None
            and tuple(drawn) != tuple(video.frame_size)
        ):
            kind = "adjusted regions" if video.zones is not None else "regions"
            issues.append(ProjectIssue(
                SEVERITY_ERROR, "zones_size_mismatch",
                f"{video.name} is {video.frame_size[0]}x{video.frame_size[1]} "
                f"but its {kind} were drawn on {drawn[0]}x{drawn[1]} frames",
                video.key,
            ))
        if (
            video.zones is not None and video.zones.reference is not None
            and zones_now is not None and video.zones.reference != zones_now
        ):
            issues.append(ProjectIssue(
                SEVERITY_INFO, "zones_outdated",
                f"The template regions changed since {video.name} was "
                f"{video.zones.source}; check its regions", video.key,
            ))
        return issues

    def validate(self) -> List[ProjectIssue]:
        """Problems that would hurt a run, one issue each.

        Checks, per video: the file exists, it is unchanged since it was
        configured (fingerprint), it has settings, and the settings' paradigm
        matches the project's; its trial (``trial_invalid``,
        ``trial_needs_start``) and regions (``zones_size_mismatch`` against
        a known frame size, ``zones_outdated`` when the template's regions
        changed after an override was made); plus the template's paradigm,
        the sample sheet and the results folder of earlier runs.
        """
        issues: List[ProjectIssue] = []
        paradigm = self.effective_paradigm()
        template = self.template_settings
        if (
            template is not None and template.paradigm_id and paradigm
            and template.paradigm_id != paradigm
        ):
            issues.append(ProjectIssue(
                SEVERITY_ERROR, "paradigm_mismatch",
                f"The template is for {template.paradigm_id}, "
                f"the project for {paradigm}", None,
            ))
        sheet = self.run_options.sample_sheet
        if sheet is not None and not sheet.is_file():
            issues.append(ProjectIssue(
                SEVERITY_WARNING, "sample_sheet_missing",
                f"Sample sheet not found: {sheet}", None,
            ))
        zones_now = self.zones_reference()
        for video in self.videos:
            exists = video.path.is_file()
            if not exists:
                issues.append(ProjectIssue(
                    SEVERITY_ERROR, "missing_file",
                    f"Video file not found: {video.path}", video.key,
                ))
            elif video.fingerprint is not None:
                if video_fingerprint(video.path) != video.fingerprint:
                    issues.append(ProjectIssue(
                        SEVERITY_WARNING, "video_changed",
                        f"{video.name} changed since it was configured; "
                        "check its settings", video.key,
                    ))
            if video.settings is None:
                issues.append(ProjectIssue(
                    SEVERITY_WARNING, "unconfigured",
                    f"{video.name} has no settings", video.key,
                ))
            elif (
                video.settings.paradigm_id and paradigm
                and video.settings.paradigm_id != paradigm
            ):
                issues.append(ProjectIssue(
                    SEVERITY_ERROR, "paradigm_mismatch",
                    f"{video.name} is configured for "
                    f"{video.settings.paradigm_id}, the project is {paradigm}",
                    video.key,
                ))
            issues.extend(self._override_issues(video, zones_now))
            if video.last_run_dir is not None and not video.last_run_dir.is_dir():
                issues.append(ProjectIssue(
                    SEVERITY_INFO, "run_folder_missing",
                    f"Results folder of {video.name} not found: "
                    f"{video.last_run_dir}", video.key,
                ))
        return issues

    # ------------------------------------------------------------------
    # Run history
    # ------------------------------------------------------------------
    def record_run(self, run_dir: os.PathLike) -> RunRecord:
        """Add a finished (or interrupted) run folder to the history.

        Reads ``manifest.json`` (schema 3; older ones are read as far as they
        go) and updates every video of the run: ``done``, ``warning`` (it has
        a quality warning), ``failed``, and ``interrupted`` for videos that
        ``run_plan.json`` lists but the manifest does not. A folder with only
        a ``run_plan.json`` (the run died before writing the manifest) marks
        all its videos interrupted. Recording the same folder again replaces
        its record.
        """
        run_dir = Path(run_dir).resolve()
        manifest_path = run_dir / MANIFEST_FILENAME
        plan_path = run_dir / RUN_PLAN_FILENAME
        manifest: Dict[str, Any] = {}
        if manifest_path.is_file():
            manifest = read_manifest(run_dir)
        elif not plan_path.is_file():
            raise ProjectError(
                f"{run_dir} has neither a {MANIFEST_FILENAME} nor a "
                f"{RUN_PLAN_FILENAME}"
            )

        by_path = {norm_path(video.path): video for video in self.videos}
        warned = {
            norm_path(item["source"])
            for item in manifest.get("quality_warnings") or []
            if isinstance(item, dict) and item.get("source")
        }
        outcomes: Dict[str, str] = {}
        hashes: Dict[str, str] = {}
        failed_sources = set()
        for item in manifest.get("videos") or []:
            if not isinstance(item, dict) or not item.get("source"):
                continue
            marker = norm_path(item["source"])
            if item.get("status") == "completed":
                outcomes[marker] = (
                    STATUS_WARNING if marker in warned else STATUS_DONE
                )
            elif item.get("status") == "failed":
                outcomes[marker] = STATUS_FAILED
                failed_sources.add(marker)
            video = by_path.get(marker)
            if video is not None and item.get("settings_hash"):
                hashes[video.key] = str(item["settings_hash"])
        for item in manifest.get("failures") or []:
            if isinstance(item, dict) and item.get("source"):
                marker = norm_path(item["source"])
                outcomes[marker] = STATUS_FAILED
                failed_sources.add(marker)
        for marker in _plan_video_paths(run_dir):
            outcomes.setdefault(marker, STATUS_INTERRUPTED)

        for marker, status in outcomes.items():
            video = by_path.get(marker)
            if video is not None:
                video.last_status = status
                video.last_run_dir = run_dir
        done = sum(
            1 for status in outcomes.values()
            if status in (STATUS_DONE, STATUS_WARNING)
        )
        record = RunRecord(
            run_dir=run_dir,
            started_at=str(manifest.get("started_at") or ""),
            finished_at=str(manifest.get("finished_at") or ""),
            cancelled=bool(manifest.get("cancelled", not manifest)),
            completed=int(manifest.get("completed_videos", done)),
            failed=len(failed_sources),
            manifest=manifest_path if manifest else None,
            settings_hashes=hashes,
        )
        self.runs = [r for r in self.runs if norm_path(r.run_dir) != norm_path(run_dir)]
        self.runs.append(record)
        self._changed()
        return record

    # ------------------------------------------------------------------
    # Moved data
    # ------------------------------------------------------------------
    def missing_files(self) -> List[VideoEntry]:
        """Videos whose file cannot be found."""
        return [video for video in self.videos if not video.path.is_file()]

    def relink(self, old_root: os.PathLike, new_root: os.PathLike) -> List[str]:
        """Repoint everything under ``old_root`` to ``new_root``.

        For a data folder that moved: videos, results folders, manifests and
        the sample sheet that lived under ``old_root`` follow it. Keys do not
        change. A video's fingerprint contains its path, so it is recorded
        again at the new place when the file is there. Returns the keys of the
        relinked videos.
        """
        old_root, new_root = Path(old_root), Path(new_root)
        relinked: List[str] = []
        for video in self.videos:
            moved = rebase_path(video.path, old_root, new_root)
            if moved is not None:
                video.path = moved
                if video.fingerprint is not None and moved.is_file():
                    video.fingerprint = video_fingerprint(moved)
                relinked.append(video.key)
            moved_run = rebase_path(video.last_run_dir, old_root, new_root)
            if moved_run is not None:
                video.last_run_dir = moved_run
        for run in self.runs:
            run.run_dir = rebase_path(run.run_dir, old_root, new_root) or run.run_dir
            run.manifest = rebase_path(run.manifest, old_root, new_root) or run.manifest
        sheet = rebase_path(self.run_options.sample_sheet, old_root, new_root)
        if sheet is not None:
            self.run_options.sample_sheet = sheet
        if relinked or sheet is not None or self.runs:
            self._changed()
        return relinked

    def bind_to_file(self, path: os.PathLike) -> None:
        """Remember where the project is saved.

        The first time (no file yet) the video keys, which were absolute,
        become relative to the project file; the run history follows. Later
        moves of the file keep every key.
        """
        first_time = self.file_path is None
        self.file_path = Path(path).resolve()
        if first_time and self.rekey():
            self._changed(touch=False)

    def rekey(self) -> Dict[str, str]:
        """Recompute every key from the current file location.

        Returns ``{old key: new key}`` for the keys that changed.
        """
        mapping: Dict[str, str] = {}
        taken: List[str] = []
        for video in self.videos:
            new_key = self._unique_key(self.key_for(video.path), taken)
            taken.append(new_key)
            if new_key != video.key:
                mapping[video.key] = new_key
                video.key = new_key
        if mapping:
            for run in self.runs:
                run.settings_hashes = {
                    mapping.get(key, key): value
                    for key, value in run.settings_hashes.items()
                }
        return mapping
