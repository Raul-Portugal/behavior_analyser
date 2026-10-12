"""Helpers for tests that build the main window without touching real settings."""
import tempfile
from pathlib import Path

from gui.app_settings import AppSettings


def temp_app_settings(testcase) -> AppSettings:
    """An AppSettings backed by an ini file in a throw-away folder."""
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    testcase.addCleanup(tmp.cleanup)
    return AppSettings.for_file(Path(tmp.name) / "settings.ini")
