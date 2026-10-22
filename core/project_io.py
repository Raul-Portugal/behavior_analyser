"""Reading, writing and importing ``.bmaproj`` project files.

* :func:`save_project` / :func:`load_project`: atomic JSON, paths relative to
  the project file, newer schemas rejected.
* Importers: :func:`project_from_template`, :func:`project_from_batch_plan`
  and :func:`project_from_run_folder` (open an existing 3.x run as a project).
* :func:`export_batch_plan`: a project back to a plain 3.x batch plan.
* :class:`ProjectAutosaver`: debounced ``<name>.bmaproj.autosave`` writes and
  crash recovery. It has no timer of its own; the shell calls
  :meth:`ProjectAutosaver.poll` from one.

Nothing here touches Qt.
"""
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from core.batch_plan_keys import build_plan_by_key, match_videos_to_plan
from core.models import BatchSettings
from core.project import (FILE_TYPE, PROJECT_EXTENSION, SOURCE_CUSTOM,
                          SOURCE_TEMPLATE, Project, ProjectError,
                          canonical_json, read_manifest, resolve_stored_path)
from core.run_outputs import MANIFEST_FILENAME, RUN_PLAN_FILENAME
from core.version import APP_VERSION

logger = logging.getLogger(__name__)

AUTOSAVE_SUFFIX = ".autosave"


# ----------------------------------------------------------------------
# Save and load
# ----------------------------------------------------------------------
def with_project_extension(path: os.PathLike) -> Path:
    """``path`` with the ``.bmaproj`` extension added if it has another."""
    path = Path(path)
    if path.suffix.lower() != PROJECT_EXTENSION:
        path = path.with_name(path.name + PROJECT_EXTENSION)
    return path


def autosave_path_for(path: os.PathLike) -> Path:
    """``<name>.bmaproj.autosave`` next to the project file."""
    path = Path(path)
    return path.with_name(path.name + AUTOSAVE_SUFFIX)


def _write_json_atomic(content: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    BatchSettings._save_json(content, path)


def save_project(project: Project, path: os.PathLike) -> Path:
    """Write the project atomically to ``path`` and return the final path.

    The extension ``.bmaproj`` is added when missing. The project is bound to
    the file (so ``project.file_path`` is where it was saved; the first save
    turns the video keys into paths relative to the file), every path is
    stored relative to the file where possible, and a stale autosave is
    removed.
    """
    target = with_project_extension(path).resolve()
    project.bind_to_file(target)
    project.app_version = APP_VERSION
    _write_json_atomic(project.to_dict(target.parent), target)
    discard_autosave(target)
    return target


def _read_project_file(path: Path) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ProjectError(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProjectError(f"{path.name} is not a project file")
    return data


def load_project(path: os.PathLike, recover: bool = False) -> Project:
    """Read a project file; paths resolve against the file's folder.

    With ``recover`` the newer autosave next to ``path`` is read instead (see
    :func:`recovery_available`); the project is still bound to ``path``, so
    saving it replaces the real file. Raises ``ProjectError`` for a file that
    is not a project or has a newer schema, ``OSError`` if it cannot be read.
    """
    target = Path(path).resolve()
    source = autosave_path_for(target) if recover else target
    project = Project.from_dict(_read_project_file(source), target.parent)
    project.file_path = target
    return project


def recovery_available(path: os.PathLike) -> bool:
    """True if an autosave exists that is newer than the project file."""
    target = Path(path)
    autosave = autosave_path_for(target)
    try:
        if not autosave.is_file():
            return False
        if not target.is_file():
            return True
        return autosave.stat().st_mtime > target.stat().st_mtime
    except OSError:
        return False


def discard_autosave(path: os.PathLike) -> None:
    try:
        autosave_path_for(path).unlink()
    except OSError:
        pass


def is_project_file(path: os.PathLike) -> bool:
    try:
        return _read_project_file(Path(path)).get("type") == FILE_TYPE
    except (OSError, ProjectError):
        return False


# ----------------------------------------------------------------------
# Importers
# ----------------------------------------------------------------------
def _new_project(name: str, project_file: Optional[os.PathLike]) -> Project:
    project = Project(name=name)
    if project_file is not None:
        project.bind_to_file(with_project_extension(project_file))
    return project


def _without_times(settings: BatchSettings) -> str:
    """Everything of ``settings`` but its times (and what follows the regions)."""
    return canonical_json({
        key: value for key, value in settings.to_dict().items()
        if key not in ("start_time", "end_time", "per_roi_times", "schema_version")
    })


def _adopt_settings(
    project: Project, assignments: List
) -> None:
    """Configure videos from ``[(key, settings)]`` and find the template.

    Settings shared by at least two videos, apart from their start/end times,
    become the project template. A video that differs from it only in its
    times gets a trial window, one that also (or only) has other regions gets
    a zone override, and both keep following the template for everything
    else. A video that differs in anything more keeps its own full copy as
    ``custom``. When the shared settings carry different times, the template
    starts at 0 and runs to the end of the video, so each trial states its
    own times. The paradigm and algorithm parameters are taken from the first
    settings that name them.
    """
    if not assignments:
        return
    counts = Counter(_without_times(s) for _, s in assignments)
    common, count = counts.most_common(1)[0]
    template = None
    if count >= 2:
        group = [s for _, s in assignments if _without_times(s) == common]
        template = group[0].copy()
        if len({
            canonical_json([s.start_time, s.end_time, s.per_roi_times])
            for s in group
        }) > 1:
            template.start_time = 0.0
            template.end_time = None
            template.per_roi_times = {}
        project.set_template(template, update_inheriting=False)
    for key, settings in assignments:
        entry = project.entry(key)
        if template is None:
            project._configure(entry, settings, SOURCE_CUSTOM)
            continue
        project._configure(entry, template, SOURCE_TEMPLATE)
        if canonical_json(settings.to_dict()) == canonical_json(template.to_dict()):
            continue
        if not project._store_as_layers(entry, settings, SOURCE_CUSTOM):
            project._configure(entry, settings, SOURCE_CUSTOM)
    for _, settings in assignments:
        if not project.paradigm and settings.paradigm_id:
            project.paradigm = settings.paradigm_id
        if not project.algorithm_parameters and settings.algorithm_parameters:
            project.algorithm_parameters = dict(settings.algorithm_parameters)


def project_from_template(
    template_path: os.PathLike,
    videos: Iterable[os.PathLike] = (),
    name: Optional[str] = None,
    project_file: Optional[os.PathLike] = None,
) -> Project:
    """A project whose template is a saved single-video configuration.

    Every video inherits the template. ``project_file`` (optional) is where
    the project will be saved, so keys are relative from the start.
    """
    template_path = Path(template_path)
    try:
        settings = BatchSettings.load_from_file(template_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ProjectError(f"Cannot read {template_path.name}: {exc}") from exc
    project = _new_project(name or template_path.stem, project_file)
    project.set_template(settings)
    project.add_videos(videos)
    return project


def project_from_batch_plan(
    plan_path: os.PathLike,
    videos: Optional[Iterable[os.PathLike]] = None,
    name: Optional[str] = None,
    project_file: Optional[os.PathLike] = None,
) -> Project:
    """A project from a 3.x batch plan.

    Without ``videos`` the plan's own entries are used (keys resolve against
    the plan file). With ``videos`` each is matched to a plan entry exactly
    like the CLI does (``match_videos_to_plan``); a video the plan does not
    cover is added without settings, so ``validate`` flags it.
    """
    plan_path = Path(plan_path)
    try:
        by_key = BatchSettings.load_batch_plan(plan_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ProjectError(f"Cannot read {plan_path.name}: {exc}") from exc
    project = _new_project(name or plan_path.stem, project_file)
    pairs: List = []  # (video path, plan settings or None)
    if videos is None:
        for key, settings in by_key.items():
            pairs.append((resolve_stored_path(key, plan_path.resolve().parent), settings))
    else:
        selected = [Path(video) for video in videos]
        match = match_videos_to_plan(selected, by_key.keys(), plan_path)
        for video in selected:
            key = match.matched.get(video)
            pairs.append((video, None if key is None else by_key[key]))
    project.add_videos([path for path, _ in pairs])
    assignments = []
    for path, settings in pairs:
        if settings is not None:
            assignments.append((project.find(path).key, settings))
    _adopt_settings(project, assignments)
    return project


def project_from_run_folder(
    run_dir: os.PathLike,
    name: Optional[str] = None,
    project_file: Optional[os.PathLike] = None,
) -> Project:
    """Open a finished or interrupted run folder as a project.

    ``run_plan.json`` supplies the videos and their exact settings,
    ``manifest.json`` the metadata, the sample sheet, the time-bin length,
    the video fingerprints and the outcome of each video. Works with 3.x runs
    (plan and manifest schema 3); a folder with only one of the two files
    still opens.
    """
    run_dir = Path(run_dir).resolve()
    plan_path = run_dir / RUN_PLAN_FILENAME
    manifest = read_manifest(run_dir) if (run_dir / MANIFEST_FILENAME).is_file() else {}
    by_key: Dict[str, BatchSettings] = {}
    if plan_path.is_file():
        try:
            by_key = BatchSettings.load_batch_plan(plan_path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ProjectError(f"Cannot read {plan_path.name}: {exc}") from exc
    if not by_key and not manifest:
        raise ProjectError(
            f"{run_dir} has neither a {RUN_PLAN_FILENAME} nor a "
            f"{MANIFEST_FILENAME}"
        )

    project = _new_project(name or run_dir.name, project_file)
    plan_paths = [resolve_stored_path(key, run_dir) for key in by_key]
    manifest_videos = [
        item for item in manifest.get("videos") or []
        if isinstance(item, dict) and item.get("source")
    ]
    project.add_videos(plan_paths + [Path(item["source"]) for item in manifest_videos])

    _adopt_settings(project, [
        (project.find(path).key, settings)
        for path, settings in zip(plan_paths, by_key.values())
    ])
    for item in manifest_videos:
        entry = project.find(Path(item["source"]))
        if entry is None:
            continue
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            entry.metadata.update(
                {str(k): "" if v is None else str(v) for k, v in metadata.items()}
            )
        if item.get("video_fingerprint"):
            # The video as it was analysed, so a later edit is detected.
            entry.fingerprint = str(item["video_fingerprint"])
    if not project.paradigm and manifest.get("paradigm"):
        project.paradigm = str(manifest["paradigm"])
    if not project.algorithm_parameters and isinstance(
        manifest.get("algorithm_parameters"), dict
    ):
        project.algorithm_parameters = dict(manifest["algorithm_parameters"])
    if manifest.get("sample_sheet"):
        project.run_options.sample_sheet = Path(manifest["sample_sheet"])
    if manifest.get("time_bin_seconds"):
        project.run_options.time_bin_seconds = float(manifest["time_bin_seconds"])
    project.record_run(run_dir)
    return project


def export_batch_plan(project: Project, path: os.PathLike) -> List[Path]:
    """Write the configured videos as a 3.x batch plan (File > Load).

    Keys are relative to the plan file, exactly as the application writes
    them, so any 3.x build and the CLI read the result. Videos without
    settings are left out and returned. Raises ``ProjectError`` when no video
    is configured.
    """
    exported = project.analysis_plan(include_missing=True)
    if not exported.plan:
        raise ProjectError("No video has settings to export")
    target = Path(path)
    _write_json_atomic(
        {
            "type": "batch_plan",
            "plan": {
                key: settings.to_dict()
                for key, settings in build_plan_by_key(
                    exported.plan, target
                ).items()
            },
        },
        target,
    )
    return exported.unconfigured


# ----------------------------------------------------------------------
# Autosave
# ----------------------------------------------------------------------
class ProjectAutosaver:
    """Debounced autosave of a project to ``<file>.autosave``.

    The autosaver listens to the project: every change marks it dirty, and
    :meth:`poll` (called from a timer by the shell) writes the autosave once
    the project has been quiet for ``debounce_s`` seconds, or after
    ``max_delay_s`` seconds of continuous edits so a long editing session is
    never unprotected. The clock is injectable for tests. A real save should
    be followed by :meth:`saved`, which removes the autosave.
    """

    def __init__(
        self,
        project: Project,
        path: os.PathLike,
        debounce_s: float = 2.0,
        max_delay_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.project = project
        self.path = with_project_extension(path).resolve()
        self.debounce_s = max(0.0, float(debounce_s))
        self.max_delay_s = max(self.debounce_s, float(max_delay_s))
        self._clock = clock
        self._first_dirty: Optional[float] = None
        self._last_change: Optional[float] = None
        self.last_error: Optional[str] = None
        project.add_listener(self.mark_dirty)

    @property
    def autosave_path(self) -> Path:
        return autosave_path_for(self.path)

    @property
    def is_dirty(self) -> bool:
        return self._first_dirty is not None

    def mark_dirty(self) -> None:
        now = self._clock()
        if self._first_dirty is None:
            self._first_dirty = now
        self._last_change = now

    def due(self) -> bool:
        """True if :meth:`poll` would write now."""
        if self._first_dirty is None or self._last_change is None:
            return False
        now = self._clock()
        return (
            now - self._last_change >= self.debounce_s
            or now - self._first_dirty >= self.max_delay_s
        )

    def poll(self) -> bool:
        """Write the autosave if one is due; True if it was written."""
        return self.flush() if self.due() else False

    def flush(self) -> bool:
        """Write the autosave now if there are unsaved changes."""
        if not self.is_dirty:
            return False
        try:
            _write_json_atomic(
                self.project.to_dict(self.path.parent), self.autosave_path
            )
        except OSError as exc:
            self.last_error = str(exc)
            logger.warning("Autosave failed: %s", exc)
            return False  # still dirty: the next poll retries
        self.last_error = None
        self._first_dirty = self._last_change = None
        return True

    def saved(self) -> None:
        """The project was saved for real: drop the autosave and the dirt."""
        self._first_dirty = self._last_change = None
        discard_autosave(self.path)

    def close(self) -> None:
        self.project.remove_listener(self.mark_dirty)

    @staticmethod
    def recovery_available(path: os.PathLike) -> bool:
        return recovery_available(path)
