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
    KEY_PARADIGM = "setup/last_paradigm"
    KEY_RECENT = "recent/files"
    KEY_RECURSIVE = "videos/add_folder_recursive"

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

    # ------------------------------------------------------------------
    # Recent templates and plans
    # ------------------------------------------------------------------
    def _stored_recent(self) -> List[str]:
        raw = self._qs.value(self.KEY_RECENT, "")
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
