"""Helpers for the project tests: cheap settings objects and fake videos."""
import tempfile
import unittest
from pathlib import Path
from typing import Optional

import numpy as np

from core.models import AppConfig, BatchSettings
from roi_manager import ROIManager


def make_settings(
    paradigm_id: Optional[str] = "Y-Maze", offset: int = 0, scale: float = 0.0
) -> BatchSettings:
    """A valid BatchSettings; ``offset`` makes the ROI (and hash) differ."""
    manager = ROIManager()
    points = np.array(
        [[offset, 0], [offset + 40, 0], [offset + 40, 40], [offset, 40]]
    )
    manager.add_roi("arm_a", points)
    return BatchSettings(
        manager, AppConfig().detection, scale, paradigm_id=paradigm_id
    )


class ProjectFixture(unittest.TestCase):
    """A temp folder with ``write_video`` for stat-only fake videos."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()

    def write_video(self, relative: str, content: bytes = b"video") -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path.resolve()
