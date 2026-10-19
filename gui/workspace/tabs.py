"""
gui/workspace/tabs.py
The contract between the workspace window and the Setup / Run / Results tabs.

The window ships a :class:`PlaceholderTab` for each tab. A real tab is a
:class:`WorkspaceTab` subclass handed to ``WorkspaceWindow.install_tab``; the
window then calls :meth:`WorkspaceTab.attach_workspace` once and the tab hears
about the project and the selection through the ``on_*`` hooks below, so it
never has to dig into the window's widgets.

What a tab can rely on (all on the GUI thread):

* ``self.workspace`` - the window; ``workspace.project`` is the current
  ``Project`` (or None) and ``workspace.selected_keys()`` the selected video
  keys in table order.
* ``on_project_changed(project)`` - a different project was opened or the
  project was closed (``None``). Rebuild everything.
* ``on_project_modified()`` - the open project was edited by anyone (the
  table, the context panel, another tab). Cheap refresh.
* ``on_selection_changed(keys)`` - the table selection changed.
* ``on_tab_activated()`` - the user switched to this tab.

Tabs change the project only through its API (``Project.customize``,
``set_template``, ``record_run`` ...); the window notices every change, marks
the project modified and autosaves it.
"""
from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from core.project import Project


class WorkspaceTab(QWidget):
    """Base class for the tabs of the workspace (all hooks are optional)."""

    workspace = None  # set by attach_workspace

    def attach_workspace(self, window) -> None:
        """Called once when the tab is installed; wires the hooks."""
        self.workspace = window
        window.project_changed.connect(self.on_project_changed)
        window.project_modified.connect(self.on_project_modified)
        window.selection_changed.connect(self.on_selection_changed)
        # Bring a tab installed late up to date.
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


class PlaceholderTab(WorkspaceTab):
    """What a tab shows until its real content is installed."""

    def __init__(self, title: str, text: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.title = title
        layout = QVBoxLayout(self)
        self.label = QLabel(text)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setEnabled(False)
        layout.addWidget(self.label)


TAB_SETUP = "setup"
TAB_RUN = "run"
TAB_RESULTS = "results"
TAB_NAMES = (TAB_SETUP, TAB_RUN, TAB_RESULTS)
TAB_TITLES = {TAB_SETUP: "Setup", TAB_RUN: "Run", TAB_RESULTS: "Results"}
