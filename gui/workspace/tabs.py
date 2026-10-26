"""
gui/workspace/tabs.py
The contract of a legacy workspace "tab".

Since the six-stage redesign no tab is built by the workspace window: every
step is a :class:`~gui.workspace.stages.base.Stage`. The only remaining
:class:`WorkspaceTab` is the old :class:`~gui.workspace.setup_tab.SetupTab`,
which no stage shows any more. It is kept as a reference implementation that
the tests drive (``tests/setup_tab_support.py`` attaches it to a window) to
prove that the Arena & zones, Detection and Trials stages store exactly what
the original Setup steps stored.

What a tab can rely on (all on the GUI thread):

* ``self.workspace`` - the window; ``workspace.project`` is the current
  ``Project`` (or None) and ``workspace.selected_keys()`` the selected video
  keys in table order.
* ``on_project_changed(project)`` - a different project was opened or the
  project was closed (``None``). Rebuild everything.
* ``on_project_modified()`` - the open project was edited by anyone.
* ``on_selection_changed(keys)`` - the table selection changed.
* ``on_tab_activated()`` - the user switched to this tab.
"""
from typing import List, Optional

from PyQt6.QtWidgets import QWidget

from core.project import Project


class WorkspaceTab(QWidget):
    """Base class for the legacy tabs (all hooks are optional)."""

    workspace = None  # set by attach_workspace

    def attach_workspace(self, window) -> None:
        """Wire the hooks to the window's signals and bring the tab up to date."""
        self.workspace = window
        window.project_changed.connect(self.on_project_changed)
        window.project_modified.connect(self.on_project_modified)
        window.selection_changed.connect(self.on_selection_changed)
        self.on_project_changed(window.project)
        self.on_selection_changed(window.selected_keys())

    def on_project_changed(self, project: Optional[Project]) -> None:
        pass

    def on_project_modified(self) -> None:
        pass

    def on_selection_changed(self, keys: List[str]) -> None:
        pass

    def on_tab_activated(self) -> None:
        pass
