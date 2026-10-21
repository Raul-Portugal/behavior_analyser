"""
gui/workspace/stages
The six stages of the 4.0 workspace::

    1 Videos -> 2 Arena & zones -> 3 Detection -> 4 Trials -> 5 Review & run -> 6 Results

(Detection is called "Motion tuning" for the Tail Suspension Test.)

Modules
-------
``base``          the :class:`Stage` base class (header, content area, hooks);
``status``        Qt-free rules for the stepper statuses;
``videos_stage``  stage 1, finished: empty state, paradigm picker, video table;
``hosted``        stages 2 to 6 in their *temporary* form: the old Setup, Run
                  and Results tabs hosted inside a stage.

The replacement contract (for the authors of stages 2 to 6)
----------------------------------------------------------
Each hosted stage is scaffolding. To replace one:

1. Write ``YourStage(Stage)`` with the same ``stage_id`` as the hosted class
   it replaces (``STAGE_ARENA``, ``STAGE_DETECTION``, ``STAGE_TRIALS``,
   ``STAGE_REVIEW`` or ``STAGE_RESULTS``). Build the content with
   ``self.set_content(widget)`` or ``self.content_layout``; use the
   ``self.header`` (``set_texts``, ``add_action``) and set ``title``,
   ``subtitle`` and ``icon_name`` as class attributes.
2. Register it in ``WorkspaceWindow._make_stages`` (``gui/workspace/window.py``)
   instead of the hosted class. Nothing else in the window changes: it calls
   the hooks below and reads ``status()``, ``next_label()``, ``can_advance()``
   and ``footer_text()`` to draw the stepper and the footer.
3. Delete the matching hosted class from ``hosted.py``. The old tabs are
   attached to the window with ``install_tab``; when the last hosted stage
   that needs one (Setup: stages 2 to 4, Run: 5, Results: 6) goes, remove the
   tab from ``WorkspaceWindow.__init__``. Anything that still uses
   ``window.setup_tab`` / ``run_tab`` / ``results_tab`` or ``show_tab`` must
   move to ``window.go_to_stage(stage_id)``.

What a stage can rely on (all on the GUI thread)
------------------------------------------------
``self.workspace``
    the window: ``project``, ``selected_keys()``, ``prompts``,
    ``app_settings``, ``video_panel``, ``is_busy``, ``set_busy(bool, why)``,
    ``register_close_guard(fn)``, ``set_run_enabled(bool)``,
    ``go_to_stage(id)``, ``refresh_footer()``, ``show_details(keys)``,
    ``thumbnails`` (a shared ``ThumbnailLoader``).
Hooks
    ``on_project_changed(project)``, ``on_project_modified()``,
    ``on_selection_changed(keys)``, ``on_enter()``, ``on_leave() -> bool``;
    the window calls them on every stage, current or not, so a stage must be
    cheap while hidden.
Footer
    ``next_label()`` (default: the next stage's title), ``can_advance()``,
    ``footer_text()``, ``status()`` (default: ``status.stage_status``), and
    ``refresh_footer()`` when any of them changed.
Changing the project
    only through the ``Project`` API; the window notices, marks the project
    modified, autosaves it and calls ``on_project_modified`` on every stage.
Dialogs
    only through ``window.prompts``, so tests never block.
Timers
    must be children of a widget (a stage or one of its children); the
    window's closing path stops them. A stage is deleted with the window.
"""
from gui.workspace.stages import status
from gui.workspace.stages.base import Stage

__all__ = ["Stage", "status"]
