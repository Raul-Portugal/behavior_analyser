"""
gui/workspace/stages/hosted.py
Stages 2 to 6 in their temporary form: the old Setup, Run and Results tabs
hosted inside a stage, so nothing breaks while the real stages are written.

* No setup stage is hosted any more: Arena & zones, Detection and
  Trials are real stages (``arena_stage.py``, ``detection_stage.py``,
  ``trials_stage.py``). ``HostedSetupStage`` is kept for reuse.
* Review & run hosts the ``RunTab``; Results hosts the ``ResultsTab``.

The tabs keep working through their ``WorkspaceTab`` hooks (the window still
attaches them with ``install_tab``); the stage adds the header, the footer
and the navigation around them. See :mod:`gui.workspace.stages` for how a
real stage replaces one of these.
"""
from typing import Optional, Tuple

from PyQt6.QtWidgets import QWidget

from gui.workspace.stages import status as stage_status
from gui.workspace.stages.base import Stage
from gui.workspace.tabs import TAB_RESULTS, TAB_RUN, TAB_SETUP, WorkspaceTab


class HostedStage(Stage):
    """A stage whose content is one of the window's old tabs."""

    #: ``tab_setup`` / ``tab_run`` / ``tab_results`` of ``gui.workspace.tabs``.
    tab_name: str = ""

    def hosted_tab(self) -> Optional[QWidget]:
        """The tab widget currently installed in the window."""
        if self.workspace is None:
            return None
        return self.workspace.tab_widget(self.tab_name)

    def is_current(self) -> bool:
        return (
            self.workspace is not None
            and self.workspace.current_stage_id == self.stage_id
        )

    def host(self) -> None:
        """Move the tab into this stage's content area."""
        tab = self.hosted_tab()
        if tab is None:
            return
        if tab.parentWidget() is not self.content:
            self.set_content(tab)
        tab.show()

    def on_tab_replaced(self, widget: QWidget) -> None:
        """``install_tab`` swapped the tab: show the new one if we are on screen."""
        if self.is_current():
            self.host()

    def on_enter(self) -> None:
        self.host()
        tab = self.hosted_tab()
        if isinstance(tab, WorkspaceTab):
            tab.on_tab_activated()


class HostedSetupStage(HostedStage):
    """A stage that shows one step of the old Setup tab."""

    tab_name = TAB_SETUP

    def setup_step(self) -> str:
        """The ``SetupTab`` step this stage opens."""
        raise NotImplementedError

    def on_enter(self) -> None:
        super().on_enter()
        tab = self.hosted_tab()
        go_to_step = getattr(tab, "go_to_step", None)
        if go_to_step is not None:
            go_to_step(self.setup_step())

    def on_leave(self) -> bool:
        """Unapplied edits of the step are resolved (Apply / Discard / Cancel)
        before the user moves on."""
        tab = self.hosted_tab()
        resolve = getattr(tab, "_resolve_unapplied", None)    # scaffolding only
        if resolve is None:
            return True
        return bool(resolve(rebuild=False))




class ReviewStage(HostedStage):
    """Review & run: the old Run tab."""

    stage_id = stage_status.STAGE_REVIEW
    title = "Review & run"
    subtitle = "Check everything, then analyse your videos."
    icon_name = "list"
    tab_name = TAB_RUN

    def can_advance(self) -> Tuple[bool, str]:
        return True, ""


class ResultsStage(HostedStage):
    """Results: the old Results tab (the last stage has no Next)."""

    stage_id = stage_status.STAGE_RESULTS
    title = "Results"
    subtitle = "Open and explore what your runs produced."
    icon_name = "chart"
    tab_name = TAB_RESULTS

    def next_label(self) -> Optional[str]:
        return None


__all__ = [
    "HostedSetupStage", "HostedStage", "ResultsStage", "ReviewStage",
]
