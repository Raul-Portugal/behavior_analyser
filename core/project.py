"""A persistent project: everything about one study in a single file.

A :class:`Project` holds the videos of a study, the settings of every video,
free metadata per video, the run options and the history of the runs made
from it. It is the data foundation of the 4.0 workspace: the workspace window
edits a project and the analysis is started from :meth:`Project.analysis_plan`.

Nothing here touches Qt. Reading and writing ``.bmaproj`` files (and the
importers for templates, batch plans and 3.x run folders) lives in
:mod:`core.project_io`.

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
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from core.models import BatchSettings
from core.run_outputs import (MANIFEST_FILENAME, MANIFEST_SCHEMA_VERSION,
                              RUN_PLAN_FILENAME)
from core.sample_sheet import (SampleMetadata, SampleSheetMatch,
                               load_sample_sheet, metadata_for_videos)
from core.version import APP_VERSION

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
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
# Videos
# ----------------------------------------------------------------------
@dataclass
class VideoEntry:
    """One video of the project and everything configured for it."""
    key: str
    path: Path
    metadata: Dict[str, str] = field(default_factory=dict)
    settings: Optional[BatchSettings] = None     # None = not configured
    settings_source: str = SOURCE_TEMPLATE
    verified: bool = False
    # ``get_video_fingerprint`` when the video was configured; a later
    # difference means the file changed (or moved) since.
    fingerprint: Optional[str] = None
    last_status: str = STATUS_NEW
    last_run_dir: Optional[Path] = None
    notes: str = ""
    # Settings found in a file that this version cannot read. Kept verbatim
    # and written back, so opening and saving never destroys them.
    _raw_settings: Optional[Dict[str, Any]] = field(
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
        project.load_warnings = warnings
        return project

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
                video.settings_source = SOURCE_TEMPLATE
                video.verified = False
                video.fingerprint = None
                video.last_status = STATUS_NEW
        if paradigm != self.paradigm or cleared:
            self.paradigm = paradigm
            self._changed()
        return cleared

    def apply_template_to(self, keys: Optional[Iterable[str]] = None) -> List[str]:
        """Reset videos to the template (all videos when ``keys`` is None)."""
        if self.template_settings is None:
            raise ProjectError("The project has no template settings")
        entries = self.videos if keys is None else self.entries_for(keys)
        for video in entries:
            self._configure(video, self.template_settings, SOURCE_TEMPLATE)
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
        """Give one video its own settings (``source``: custom or aligned)."""
        if source not in (SOURCE_CUSTOM, SOURCE_ALIGNED):
            raise ProjectError(f"Unknown settings source {source!r}")
        entry = self.entry(key)
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

    def _effective_settings(self, entry: VideoEntry) -> BatchSettings:
        """A copy of the entry's settings, carrying paradigm and parameters."""
        settings = entry.settings.copy()
        if not settings.paradigm_id and self.paradigm:
            settings.paradigm_id = self.paradigm
        if not settings.algorithm_parameters and self.algorithm_parameters:
            settings.algorithm_parameters = copy.deepcopy(self.algorithm_parameters)
        return settings

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
                result.plan[video.path] = self._effective_settings(video)
                result.keys[video.path] = video.key
        return result

    def validate(self) -> List[ProjectIssue]:
        """Problems that would hurt a run, one issue each.

        Checks, per video: the file exists, it is unchanged since it was
        configured (fingerprint), it has settings, and the settings' paradigm
        matches the project's; plus the template's paradigm, the sample sheet
        and the results folder of earlier runs.
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
