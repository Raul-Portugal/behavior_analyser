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
from typing import Any, Callable, Dict, List, Mapping, Optional

from core.models import BatchSettings
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
