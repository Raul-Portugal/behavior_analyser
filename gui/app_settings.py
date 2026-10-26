"""
gui/app_settings.py
Thin typed wrapper over QSettings for what the app remembers between sessions:
default folders for the file dialogs, window geometry, the last paradigm and a
short list of recently used templates / plans.

Nothing here is analysis state; losing or corrupting these values is harmless,
so every getter falls back to a safe default instead of raising.
"""
import json
import os
from pathlib import Path
from typing import List, Optional, Union

from PyQt6.QtCore import QByteArray, QSettings

ORGANIZATION = "BehavioralMazeAnalyzer"
APPLICATION = "BehavioralMazeAnalyzer"
MAX_RECENT_FILES = 8
MAX_RECENT_PROJECTS = 8

# Lets tests, portable installs and CI keep settings in a plain ini file
# instead of the registry / user config folder.
SETTINGS_FILE_ENV_VAR = "BEHAVIORAL_MAZE_ANALYZER_SETTINGS_FILE"

PathLike = Union[str, os.PathLike]


def _same_file(a: PathLike, b: PathLike) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


class AppSettings:
    """Typed access to the persisted application preferences."""

    KEY_VIDEO_DIR = "folders/last_video"
    KEY_OUTPUT_DIR = "folders/output"
    KEY_CONFIG_DIR = "folders/last_config"
    KEY_GEOMETRY = "window/geometry"
    KEY_STATE = "window/state"
    KEY_WS_GEOMETRY = "workspace/geometry"
    KEY_WS_STATE = "workspace/state"
    KEY_WS_SPLITTER = "workspace/splitter_sizes"
    KEY_PARADIGM = "setup/last_paradigm"
    KEY_RECENT = "recent/files"
    KEY_RECENT_PROJECTS = "recent/projects"
    KEY_WELCOME_DISMISSED = "workspace/welcome_dismissed"
    KEY_LAST_PROJECT = "projects/last"
    KEY_RECURSIVE = "videos/add_folder_recursive"
    KEY_WORKERS = "run/parallel_workers"
    KEY_SAMPLE_SHEET = "run/sample_sheet"
    KEY_THEME_MODE = "ui/theme_mode"
    KEY_UI_FLAG_PREFIX = "ui/flags/"

    def __init__(self, qsettings: Optional[QSettings] = None):
        self._qs = qsettings if qsettings is not None else self._default_qsettings()

    @staticmethod
    def _default_qsettings() -> QSettings:
        override = os.environ.get(SETTINGS_FILE_ENV_VAR)
        if override:
            return QSettings(override, QSettings.Format.IniFormat)
        return QSettings(ORGANIZATION, APPLICATION)

    @classmethod
    def for_file(cls, path: PathLike) -> "AppSettings":
        """Settings stored in an explicit ini file (tests, portable use)."""
        return cls(QSettings(str(path), QSettings.Format.IniFormat))

    def sync(self) -> None:
        self._qs.sync()

    # ------------------------------------------------------------------
    # Folders
    # ------------------------------------------------------------------
    def _get_dir(self, key: str) -> Optional[Path]:
        raw = self._qs.value(key, "")
        if not isinstance(raw, str) or not raw:
            return None
        path = Path(raw)
        return path if path.is_dir() else None

    def _set_dir(self, key: str, path: Optional[PathLike]) -> None:
        if path is None or str(path) == "":
            self._qs.remove(key)
        else:
            self._qs.setValue(key, str(path))

    def last_video_dir(self) -> Optional[Path]:
        return self._get_dir(self.KEY_VIDEO_DIR)

    def set_last_video_dir(self, path: Optional[PathLike]) -> None:
        self._set_dir(self.KEY_VIDEO_DIR, path)

    def output_dir(self) -> Optional[Path]:
        """The output folder the user chose explicitly (None if never chosen)."""
        return self._get_dir(self.KEY_OUTPUT_DIR)

    def set_output_dir(self, path: Optional[PathLike]) -> None:
        self._set_dir(self.KEY_OUTPUT_DIR, path)

    def last_config_dir(self) -> Optional[Path]:
        return self._get_dir(self.KEY_CONFIG_DIR)

    def set_last_config_dir(self, path: Optional[PathLike]) -> None:
        self._set_dir(self.KEY_CONFIG_DIR, path)

    def dialog_dir(self, kind: str, fallback: Optional[PathLike] = None) -> str:
        """Directory string for a file dialog ("" lets the OS choose).

        ``kind`` is "video", "output" or "config". An explicit output folder
        is the second choice for config dialogs, then ``fallback``.
        """
        getters = {
            "video": (self.last_video_dir,),
            "output": (self.output_dir, self.last_video_dir),
            "config": (self.last_config_dir, self.output_dir),
        }[kind]
        for getter in getters:
            found = getter()
            if found is not None:
                return str(found)
        if fallback is not None and Path(fallback).is_dir():
            return str(fallback)
        return ""

    # ------------------------------------------------------------------
    # Window
    # ------------------------------------------------------------------
    def window_geometry(self) -> Optional[QByteArray]:
        value = self._qs.value(self.KEY_GEOMETRY)
        return value if isinstance(value, QByteArray) and not value.isEmpty() else None

    def window_state(self) -> Optional[QByteArray]:
        value = self._qs.value(self.KEY_STATE)
        return value if isinstance(value, QByteArray) and not value.isEmpty() else None

    def save_window(self, geometry: QByteArray, state: QByteArray) -> None:
        self._qs.setValue(self.KEY_GEOMETRY, geometry)
        self._qs.setValue(self.KEY_STATE, state)

    # ------------------------------------------------------------------
    # Workspace window (4.0): its own layout, separate from the classic one
    # ------------------------------------------------------------------
    def workspace_geometry(self) -> Optional[QByteArray]:
        value = self._qs.value(self.KEY_WS_GEOMETRY)
        return value if isinstance(value, QByteArray) and not value.isEmpty() else None

    def workspace_state(self) -> Optional[QByteArray]:
        value = self._qs.value(self.KEY_WS_STATE)
        return value if isinstance(value, QByteArray) and not value.isEmpty() else None

    def save_workspace_window(self, geometry: QByteArray, state: QByteArray) -> None:
        self._qs.setValue(self.KEY_WS_GEOMETRY, geometry)
        self._qs.setValue(self.KEY_WS_STATE, state)

    def workspace_splitter_sizes(self) -> List[int]:
        """Widths of the video / center / context panes ([] if never saved)."""
        raw = self._qs.value(self.KEY_WS_SPLITTER, "")
        if not isinstance(raw, str) or not raw:
            return []
        try:
            loaded = json.loads(raw)
        except ValueError:
            return []
        if (
            not isinstance(loaded, list)
            or not loaded
            or not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0
                       for v in loaded)
        ):
            return []
        return loaded

    def set_workspace_splitter_sizes(self, sizes: List[int]) -> None:
        self._qs.setValue(
            self.KEY_WS_SPLITTER, json.dumps([int(v) for v in sizes])
        )

    # ------------------------------------------------------------------
    # Setup choices
    # ------------------------------------------------------------------
    def last_paradigm(self) -> Optional[str]:
        value = self._qs.value(self.KEY_PARADIGM, "")
        return value if isinstance(value, str) and value else None

    def set_last_paradigm(self, name: Optional[str]) -> None:
        if name:
            self._qs.setValue(self.KEY_PARADIGM, name)
        else:
            self._qs.remove(self.KEY_PARADIGM)

    def add_folder_recursive(self) -> bool:
        return str(self._qs.value(self.KEY_RECURSIVE, "false")).lower() == "true"

    def set_add_folder_recursive(self, enabled: bool) -> None:
        self._qs.setValue(self.KEY_RECURSIVE, "true" if enabled else "false")

    def parallel_workers(self) -> int:
        """Videos analysed at once; 0 means "Auto" (the default)."""
        try:
            return max(0, int(self._qs.value(self.KEY_WORKERS, 0)))
        except (TypeError, ValueError):
            return 0

    def set_parallel_workers(self, count: int) -> None:
        self._qs.setValue(self.KEY_WORKERS, max(0, int(count)))

    def sample_sheet(self) -> Optional[Path]:
        """The last sample sheet chosen (None if never chosen or deleted)."""
        raw = self._qs.value(self.KEY_SAMPLE_SHEET, "")
        if not isinstance(raw, str) or not raw:
            return None
        path = Path(raw)
        return path if path.is_file() else None

    def set_sample_sheet(self, path: Optional[PathLike]) -> None:
        if path is None or str(path) == "":
            self._qs.remove(self.KEY_SAMPLE_SHEET)
        else:
            self._qs.setValue(self.KEY_SAMPLE_SHEET, str(path))

    # ------------------------------------------------------------------
    # Look and feel (workspace theme, remembered section states)
    # ------------------------------------------------------------------
    THEME_MODES = ("auto", "light", "dark")

    def theme_mode(self) -> str:
        """"auto" (follow the OS), "light" or "dark"; "auto" if unset."""
        value = self._qs.value(self.KEY_THEME_MODE, "auto")
        return value if isinstance(value, str) and value in self.THEME_MODES else "auto"

    def set_theme_mode(self, mode: str) -> None:
        if mode not in self.THEME_MODES:
            raise ValueError(f"Unknown theme mode: {mode!r}")
        self._qs.setValue(self.KEY_THEME_MODE, mode)

    def ui_flag(self, key: str, default: bool = False) -> bool:
        """A remembered on/off interface state (e.g. a collapsed section)."""
        raw = self._qs.value(self.KEY_UI_FLAG_PREFIX + key, None)
        if raw is None:
            return default
        return str(raw).lower() == "true"

    def set_ui_flag(self, key: str, value: bool) -> None:
        self._qs.setValue(
            self.KEY_UI_FLAG_PREFIX + key, "true" if value else "false"
        )

    # ------------------------------------------------------------------
    # Recent templates and plans
    # ------------------------------------------------------------------
    def _stored_recent(self, key: Optional[str] = None) -> List[str]:
        raw = self._qs.value(key or self.KEY_RECENT, "")
        if not isinstance(raw, str) or not raw:
            return []
        try:
            loaded = json.loads(raw)
        except ValueError:
            return []
        if not isinstance(loaded, list):
            return []
        return [item for item in loaded if isinstance(item, str) and item]

    def recent_files(self, existing_only: bool = True) -> List[Path]:
        """Most recent first; files that no longer exist are left out."""
        paths = [Path(item) for item in self._stored_recent()]
        if existing_only:
            paths = [path for path in paths if path.is_file()]
        return paths[:MAX_RECENT_FILES]

    def add_recent_file(self, path: PathLike) -> None:
        entry = str(Path(path).resolve())
        items = [
            item for item in self._stored_recent() if not _same_file(item, entry)
        ]
        items.insert(0, entry)
        self._qs.setValue(
            self.KEY_RECENT, json.dumps(items[:MAX_RECENT_FILES])
        )

    def remove_recent_file(self, path: PathLike) -> None:
        items = [
            item for item in self._stored_recent() if not _same_file(item, path)
        ]
        self._qs.setValue(self.KEY_RECENT, json.dumps(items))

    def clear_recent_files(self) -> None:
        self._qs.remove(self.KEY_RECENT)

    # ------------------------------------------------------------------
    # Recent projects (.bmaproj)
    # ------------------------------------------------------------------
    def recent_projects(self, existing_only: bool = True) -> List[Path]:
        """Most recent first (at most 8); missing files are left out."""
        paths = [Path(item) for item in self._stored_recent(self.KEY_RECENT_PROJECTS)]
        if existing_only:
            paths = [path for path in paths if path.is_file()]
        return paths[:MAX_RECENT_PROJECTS]

    def add_recent_project(self, path: PathLike) -> None:
        """Put a project first in the list; it also becomes ``last_project``."""
        entry = str(Path(path).resolve())
        items = [
            item for item in self._stored_recent(self.KEY_RECENT_PROJECTS)
            if not _same_file(item, entry)
        ]
        items.insert(0, entry)
        self._qs.setValue(
            self.KEY_RECENT_PROJECTS, json.dumps(items[:MAX_RECENT_PROJECTS])
        )
        self._qs.setValue(self.KEY_LAST_PROJECT, entry)

    def remove_recent_project(self, path: PathLike) -> None:
        items = [
            item for item in self._stored_recent(self.KEY_RECENT_PROJECTS)
            if not _same_file(item, path)
        ]
        self._qs.setValue(self.KEY_RECENT_PROJECTS, json.dumps(items))
        raw = self._qs.value(self.KEY_LAST_PROJECT, "")
        if isinstance(raw, str) and raw and _same_file(raw, path):
            self._qs.remove(self.KEY_LAST_PROJECT)

    def clear_recent_projects(self) -> None:
        self._qs.remove(self.KEY_RECENT_PROJECTS)
        self._qs.remove(self.KEY_LAST_PROJECT)

    def last_project(self) -> Optional[Path]:
        """The project opened or saved last, if its file still exists.

        Closing a project on purpose can ``clear_last_project`` so the next
        start does not reopen it; the recent list is kept.
        """
        raw = self._qs.value(self.KEY_LAST_PROJECT, "")
        if not isinstance(raw, str) or not raw:
            return None
        path = Path(raw)
        return path if path.is_file() else None

    def clear_last_project(self) -> None:
        self._qs.remove(self.KEY_LAST_PROJECT)

    # ------------------------------------------------------------------
    # First-run welcome card (workspace empty state)
    # ------------------------------------------------------------------
    def welcome_dismissed(self) -> bool:
        """True once the one-time 3.x migration card was dealt with."""
        return str(self._qs.value(self.KEY_WELCOME_DISMISSED, "false")).lower() == "true"

    def set_welcome_dismissed(self, dismissed: bool = True) -> None:
        self._qs.setValue(
            self.KEY_WELCOME_DISMISSED, "true" if dismissed else "false"
        )

    # ------------------------------------------------------------------
    # Where the user was in each project (workspace stage)
    # ------------------------------------------------------------------
    KEY_STAGES = "workspace/project_stages"

    def _stage_map(self) -> dict:
        raw = self._qs.value(self.KEY_STAGES, "")
        try:
            data = json.loads(raw) if isinstance(raw, str) and raw else {}
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    def workspace_stage(self, project_file: PathLike) -> Optional[str]:
        """The stage the user left a project at (None if never recorded)."""
        found = self._stage_map().get(os.path.normcase(os.path.abspath(project_file)))
        return found if isinstance(found, str) and found else None

    def set_workspace_stage(self, project_file: PathLike, stage: Optional[str]) -> None:
        """Remember (or, with None, forget) the stage a project was left at."""
        mapping = self._stage_map()
        key = os.path.normcase(os.path.abspath(project_file))
        if not stage:
            mapping.pop(key, None)
        else:
            mapping[key] = str(stage)
        self._qs.setValue(self.KEY_STAGES, json.dumps(mapping))

    # ------------------------------------------------------------------
    # Run output folder of each project (workspace Review & run stage)
    # ------------------------------------------------------------------
    KEY_RUN_OUTPUTS = "run/project_output_dirs"

    def _run_output_map(self) -> dict:
        raw = self._qs.value(self.KEY_RUN_OUTPUTS, "")
        try:
            data = json.loads(raw) if isinstance(raw, str) and raw else {}
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    def run_output_dir(self, project_file: PathLike) -> Optional[Path]:
        """The output folder chosen for a project (it may not exist yet)."""
        found = self._run_output_map().get(os.path.normcase(os.path.abspath(project_file)))
        return Path(found) if isinstance(found, str) and found else None

    def set_run_output_dir(
        self, project_file: PathLike, path: Optional[PathLike]
    ) -> None:
        """Remember (or, with None, forget) a project's run output folder."""
        mapping = self._run_output_map()
        key = os.path.normcase(os.path.abspath(project_file))
        if path is None or str(path) == "":
            mapping.pop(key, None)
        else:
            mapping[key] = str(path)
        self._qs.setValue(self.KEY_RUN_OUTPUTS, json.dumps(mapping))
