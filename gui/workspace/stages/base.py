"""
gui/workspace/stages/base.py
The base class of the six stages of the workspace.

A stage owns the whole content area of the window. Its widget is built
from a :class:`~gui.workspace.ui_kit.StageHeader` at the top and a content
area below it; the window supplies the stepper above and the shared
:class:`~gui.workspace.ui_kit.FooterBar` below, and calls the hooks of this
class. A stage never navigates by itself: it answers ``can_advance`` and
the window moves on (``window.go_to_stage`` is there for the rare stage
that must send the user elsewhere, such as the Videos stage after a run).

Hooks (all optional, all on the GUI thread)
-------------------------------------------
``attach(window)``
    once, when the window installs the stage; ``self.workspace`` is set.
``on_project_changed(project)``
    a project was opened or closed (``None``): rebuild everything.
``on_project_modified()``
    the open project was edited by anyone: a cheap refresh.
``on_selection_changed(keys)``
    the selection of the video table changed.
``on_enter()`` / ``on_leave() -> bool``
    the stage becomes / stops being the current one. ``on_leave`` may
    return False to veto the move (it asks the user itself, through
    ``window.prompts``, e.g. about unsaved work).
``status() -> (status, reason)``
    what the stepper shows (see :mod:`gui.workspace.stages.status`).
``next_label()`` / ``can_advance() -> (bool, reason)``
    the footer's primary button: "Next: <label> →", disabled with the
    reason as its tooltip when ``can_advance`` says no.
``footer_text() -> (text, chip)``
    the muted sentence (and optional ``(kind, text)`` chip) in the footer.
"""
from typing import List, Optional, Tuple

from PyQt6.QtWidgets import QVBoxLayout, QWidget

from core.project import Project
from gui.workspace.stages import status as stage_status
from gui.workspace.theme import Spacing
from gui.workspace.ui_kit import StageHeader

__all__ = ["Stage"]

# Stepper status -> footer chip kind
_CHIP_KINDS = {
    stage_status.STATUS_DONE: "ok",
    stage_status.STATUS_ATTENTION: "warning",
}


class Stage(QWidget):
    """One stage of the workspace flow."""

    #: Identifier used by the stepper, the settings and ``go_to_stage``.
    stage_id: str = ""
    #: Stepper label and header title.
    title: str = ""
    #: One sentence under the title: what to do here.
    subtitle: str = ""
    #: Name of the stepper / header icon (see ``gui.workspace.icons``).
    icon_name: str = "info"
    #: Whether the stage draws its own header (the Videos stage hides it
    #: while there is no project, where the empty state is the header).
    show_header: bool = True
    #: Margins around header and content (the content may override them).
    CONTENT_MARGIN = Spacing.LG

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.workspace = None
        # Set by the window: the title of the following stage ("Next: ...").
        self.next_stage_title: Optional[str] = None
        self.header = StageHeader(self.title, self.subtitle, self.icon_name)
        self.content = QWidget()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(
            self.CONTENT_MARGIN, Spacing.MD, self.CONTENT_MARGIN, Spacing.SM
        )
        outer.setSpacing(Spacing.MD)
        outer.addWidget(self.header)
        outer.addWidget(self.content, 1)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(Spacing.MD)

    # -- wiring -----------------------------------------------------------
    def attach(self, window) -> None:
        """Called once by the window; ``window.project`` is already valid."""
        self.workspace = window

    @property
    def project(self) -> Optional[Project]:
        return None if self.workspace is None else self.workspace.project

    # -- content ----------------------------------------------------------
    def set_texts(self, title: Optional[str] = None,
                  subtitle: Optional[str] = None) -> None:
        """Change the header texts (and ``title`` / ``subtitle``)."""
        if title is not None:
            self.title = title
        if subtitle is not None:
            self.subtitle = subtitle
        self.header.set_texts(self.title, self.subtitle)

    def set_content(self, widget: QWidget) -> None:
        """Put ``widget`` as the only child of the content area."""
        self.clear_content()
        self.content_layout.addWidget(widget, 1)

    def clear_content(self) -> List[QWidget]:
        """Detach everything from the content area (not deleted); returns it."""
        taken = []
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                taken.append(widget)
        return taken

    # -- hooks (override what you need) -----------------------------------
    def on_project_changed(self, project: Optional[Project]) -> None:
        pass

    def on_project_modified(self) -> None:
        pass

    def on_selection_changed(self, keys: List[str]) -> None:
        pass

    def on_enter(self) -> None:
        pass

    def on_leave(self) -> bool:
        """Return False to keep the user on this stage."""
        return True

    # -- what the window shows for this stage -----------------------------
    def status(self) -> Tuple[str, str]:
        """``(status, reason)`` for the stepper, from the project alone.

        The default delegates to :func:`gui.workspace.stages.status.stage_status`;
        a stage may refine it with what only it knows.
        """
        if not self.stage_id:
            return stage_status.STATUS_TODO, ""
        issues = None if self.workspace is None else self.workspace.project_issues()
        found = stage_status.stage_status(self.stage_id, self.project, issues)
        return found.status, found.reason

    def next_label(self) -> Optional[str]:
        """The label after "Next: " (None hides the button, as on the last stage)."""
        return self.next_stage_title

    def can_advance(self) -> Tuple[bool, str]:
        """Whether "Next" is enabled, and why not."""
        return True, ""

    def footer_text(self) -> Tuple[str, Optional[Tuple[str, str]]]:
        """The footer's status sentence and an optional ``(kind, text)`` chip."""
        status, reason = self.status()
        kind = _CHIP_KINDS.get(status)
        chip = None
        if kind == "ok":
            chip = ("ok", "Done")
        elif kind == "warning":
            chip = ("warning", "Needs attention")
        return reason, chip

    def refresh_footer(self) -> None:
        """Push this stage's footer state to the window (when it is current)."""
        if self.workspace is not None:
            self.workspace.refresh_footer()
