"""Helpers for the Setup tab tests: a workspace with real videos and a project."""
import time
from pathlib import Path
from typing import Iterable, List, Sequence

from tests.workspace_support import HAS_RUNTIME, WorkspaceFixture

if HAS_RUNTIME:
    from gui.workspace.setup_tab import SCOPE_TEMPLATE, SCOPE_VIDEOS  # noqa: F401
    from tests import worker_support as support

Y, E, F, T = (
    "Y-Maze", "Elevated Plus Maze", "Freestyle / Open Field",
    "Tail Suspension Test",
)


def wait_until(app, predicate, seconds: float = 30.0) -> bool:
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    return predicate()


def square(index: int, size: int = 30):
    """Four corners of a square that does not touch its neighbours."""
    x = 4 + (index % 4) * 38
    y = 4 + (index // 4) * 40
    return [(x, y), (x + size, y), (x + size, y + size), (x, y + size)]


def draw_polygon(editor, category: str, points) -> None:
    """Draw one polygon with the given corners in a zone of the editor."""
    assert editor.set_current_zone(category), category
    for x, y in points:
        assert editor.add_point(x, y, snap=False)
    assert editor.close_shape(), category


def draw_zones(editor, categories: Sequence[str], first_index: int = 0) -> None:
    """Draw one square per category on a ROI editor (scripted, no mouse)."""
    for offset, category in enumerate(categories):
        assert editor.set_current_zone(category), category
        for x, y in square(first_index + offset):
            assert editor.add_point(x, y, snap=False)
        assert editor.close_shape(), category


class SetupTabFixture(WorkspaceFixture):
    """A project of two real 160x120 clips with a paradigm chosen."""

    PARADIGM = Y
    FRAMES = 60

    def setUp(self):
        super().setUp()
        self.tab = self.attach_setup_tab()
        self.new_project_at()
        folder = self.root / "real"
        folder.mkdir()
        self.clips: List[Path] = [
            support.write_tracking_video(
                folder, frames=self.FRAMES, name=f"clip{n}.avi"
            )
            for n in (1, 2)
        ]
        self.add_videos(self.clips)
        self.window.video_panel.clear_selection()
        self.project = self.window.project
        self.keys = self.project.keys()
        if self.PARADIGM:
            self.choose_paradigm(self.PARADIGM)

    def attach_setup_tab(self):
        """The legacy Setup tab, attached to the window by the test.

        No stage shows it any more (Arena & zones, Detection and Trials
        replaced it), but its own tests and the stage tests that compare
        against what it stores still drive it. The Tools > Align action
        belongs to the Arena & zones stage, so the tab's handler is detached
        from it; tests call ``tab.align_template()`` directly.
        """
        from PyQt6.QtWidgets import QWidget
        from gui.workspace.setup_tab import SetupTab
        window = self.window
        parking = QWidget(window)
        parking.hide()
        tab = SetupTab()
        tab.setParent(parking)
        tab.attach_workspace(window)
        window.align_action.triggered.disconnect(tab._on_align_action)
        window.setup_tab = tab
        return tab

    def choose_paradigm(self, name: str) -> None:
        self.assertTrue(self.window.change_paradigm(name))
        self.app.processEvents()

    def select(self, keys: Iterable[str]) -> None:
        panel = self.window.video_panel
        panel.select_keys(list(keys))
        # Deliver the table's (coalesced) selection signal, and any snap-back
        # the Setup tab asks for in answer to it.
        for _ in range(2):
            panel.flush_selection()
            self.app.processEvents()
