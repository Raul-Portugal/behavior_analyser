"""
gui/workspace/stages
The six stages of the 4.0 workspace::

    1 Videos -> 2 Arena & zones -> 3 Detection -> 4 Trials -> 5 Review & run -> 6 Results

(Detection is called "Motion tuning" for the Tail Suspension Test.)

Modules
-------
``base``             the :class:`Stage` base class (header, content area, hooks);
``status``           Qt-free rules for the stepper statuses;
``videos_stage``     stage 1: empty state, paradigm picker, video table;
``arena_stage``      stage 2: the template's zones and the zones of single
                     videos (live edits, alignment);
``detection_stage``  stage 3: live-written detection / motion tuning;
``trials_stage``     stage 4: per-video start and end of each trial;
``review_stage``     stage 5: readiness checklist, run options, the run and its
                     progress (the machinery is
                     :mod:`gui.workspace.run_controller`);
``results_stage``    stage 6: the runs of the project and their results.

Writing or changing a stage
---------------------------
Subclass ``Stage`` with a ``stage_id`` of :mod:`~gui.workspace.stages.status`,
set ``title``, ``subtitle`` and ``icon_name`` as class attributes, build the
content with ``self.set_content(widget)`` or ``self.content_layout`` and use
the ``self.header`` (``set_texts``, ``add_action``). Register the class in
``WorkspaceWindow._make_stages`` (``gui/workspace/window.py``): the window
calls the hooks below and reads ``status()``, ``next_label()``,
``can_advance()`` and ``footer_text()`` to draw the stepper and the footer. A
stage that owns threads or capture devices defines ``shutdown()``; the window
calls it on every stage when it closes.

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
