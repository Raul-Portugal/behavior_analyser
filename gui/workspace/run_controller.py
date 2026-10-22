"""
gui/workspace/run_controller.py
The run machinery of the workspace, without any layout: the Review & run
stage owns one :class:`RunController` and only draws what it reports.

The controller

* keeps the run choices that are not stored in the project (scope, the
  validation-video override, the output folder of this project);
* builds and checks the plan (:func:`gui.workspace.run_check.prepare_run`),
  asks the user about pre-flight warnings and starts ``AnalysisWorker`` on a
  ``QThread`` exactly like the classic window and the CLI (the analysis
  itself is untouched, so the outputs are byte-identical);
* feeds the per-video progress table, the overall progress, the ETA and the
  log, and records the finished run in the project;
* offers to resume interrupted videos from their checkpoints;
* cancels a run (with confirmation) and holds the window open while the
  worker saves its checkpoint when the user closes it during a run.

All of it runs on the GUI thread. The timers belong to the controller, so a
pending force-quit offer or close never touches a deleted object.

Validation videos
-----------------
The override is tri-state. ``None`` (the default) leaves
``BatchSettings.create_timelapse`` as configured per video; ``True`` /
``False`` force it on or off for every video of **this run only** (the run
works on copies of the settings, the project is not changed).
"""
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication

from core.batch_runner import BatchRunOutcome, default_worker_count
from core.project import Project, ProjectError
from gui import ux_helpers
from gui.video_list import STATUS_RUNNING, outcome_statuses
from gui.worker import AnalysisWorker
from gui.workspace.run_check import (SCOPE_ALL, SCOPE_SELECTED, ResumeOffer,
                                     RunPreparation, find_resumable,
                                     prepare_run, scope_keys)
from gui.workspace.run_progress import RunProgressModel, format_problems

logger = logging.getLogger(__name__)

FORCE_QUIT_DELAY_MS = 10_000
MAX_LOG_ENTRIES = 20_000

_VIDEO_ETA = re.compile(r"(?:Video\s+)?ETA:\s*([^|]+?)(?:\s*\||$)")

# A running QThread must outlive a window that was destroyed under it (tests,
# force quit), or Qt aborts. Entries leave this set when the thread finishes.
_live_threads: Set[QThread] = set()


@dataclass
class RunSummary:
    """What the finished (or cancelled) run reports to the user."""
    completed: int = 0
    failed: int = 0
    warnings: int = 0
    cancelled: bool = False
    run_dir: Optional[Path] = None
    failures: List[Tuple[str, str]] = field(default_factory=list)
    quality_warnings: List[Tuple[str, str]] = field(default_factory=list)
    problem: str = ""                 # the run could not be added to the project
    stopping_for_close: bool = False  # the window closes once saved

    @property
    def kind(self) -> str:
        """``success``, ``warning`` or ``error`` (a Callout kind)."""
        if self.failed or self.problem:
            return "error"
        if self.cancelled or self.warnings:
            return "warning"
        return "success"

    def headline(self) -> str:
        if self.stopping_for_close:
            return "Saving progress; the window closes automatically when done."
        if self.cancelled:
            return (f"Analysis cancelled. Completed: {self.completed} video(s). "
                    "The interrupted video can be resumed from its checkpoint.")
        if self.failed:
            return (f"Completed with errors. Completed: {self.completed} "
                    f"video(s), failed: {self.failed}.")
        if self.warnings:
            return (f"Completed: {self.completed} video(s), with quality "
                    "warnings. Review the per-frame evidence and the "
                    "validation video before interpreting the results.")
        return f"Analysis finished. Processed: {self.completed} video(s)."

    def detail_lines(self) -> List[str]:
        lines = [f"- {name}: {message}" for name, message in self.failures[:5]]
        if not self.failures:
            lines = [f"- {name}: {message}"
                     for name, message in self.quality_warnings[:5]]
        if self.run_dir is not None:
            lines.append(f"Saved to: {self.run_dir}")
        if self.problem:
            lines.append(self.problem)
        return lines

    def text(self) -> str:
        return "\n".join([self.headline()] + self.detail_lines())


class RunController(QObject):
    """Start, follow, cancel and resume the analysis of the project's videos."""

    run_started = pyqtSignal()
    run_finished = pyqtSignal(object)        # BatchRunOutcome
    #: Something the UI shows changed (running, checking, resume offer, ...).
    state_changed = pyqtSignal()
    #: The overall progress, status sentence or ETA changed.
    progress_changed = pyqtSignal()
    #: A log line was added (level, text) / the log was cleared.
    log_added = pyqtSignal(int, str)
    log_cleared = pyqtSignal()

    FORCE_QUIT_DELAY_MS = FORCE_QUIT_DELAY_MS

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.workspace = None
        self.progress_model = RunProgressModel(self)
        self._scope = SCOPE_ALL
        self._validation: Optional[bool] = None
        self._explicit_output: Optional[Path] = None
        self._checking = False
        self._thread: Optional[QThread] = None
        self._worker: Optional[AnalysisWorker] = None
        self._close_requested = False
        self._force_close = False
        self._cancelling = False
        self._run_started = 0.0
        self._worker_eta: Optional[str] = None
        self._sequential = True
        self._base_paths: Dict[str, Path] = {}
        self._current_path: Optional[Path] = None
        self._last_run_dir: Optional[Path] = None
        self._last_outcome: Optional[BatchRunOutcome] = None
        self._resume = ResumeOffer()
        self._log_entries: List[Tuple[int, str]] = []
        self.last_report: Optional[RunPreparation] = None
        self.summary: Optional[RunSummary] = None
        # What the progress panel shows.
        self.status_text = "Ready"
        self.progress_value = 0
        self.progress_maximum = 100
        self.eta_text = ""
        # Owned by this object: they die with it and never touch a deleted one.
        self._close_timer = QTimer(self)
        self._close_timer.setSingleShot(True)
        self._close_timer.setInterval(0)
        self._close_timer.timeout.connect(self._close_window)
        self._force_timer = QTimer(self)
        self._force_timer.setSingleShot(True)
        self._force_timer.timeout.connect(self._offer_force_quit)

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def attach(self, window) -> None:
        self.workspace = window
        window.register_close_guard(self._close_guard)

    @property
    def project(self) -> Optional[Project]:
        return None if self.workspace is None else self.workspace.project

    def on_project_changed(self, project: Optional[Project]) -> None:
        self._explicit_output = None
        if project is not None and project.file_path is not None:
            settings = self.workspace.app_settings
            self._explicit_output = settings.run_output_dir(project.file_path)
        if not self.is_running:
            self.summary = None
            self.progress_model.reset_rows([])
            self._last_run_dir = None
            self._reset_progress()
        self.refresh_resume()

    # ------------------------------------------------------------------
    # Choices
    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.isRunning()

    @property
    def is_checking(self) -> bool:
        return self._checking

    @property
    def is_busy(self) -> bool:
        return self.is_running or self._checking

    @property
    def close_requested(self) -> bool:
        return self._close_requested

    @property
    def run_dir(self) -> Optional[Path]:
        """The run folder of the last or current run."""
        return self._last_run_dir

    @property
    def last_outcome(self) -> Optional[BatchRunOutcome]:
        return self._last_outcome

    def scope(self) -> str:
        return self._scope

    def set_scope(self, scope: str) -> None:
        if scope != self._scope:
            self._scope = scope
            self.state_changed.emit()

    def validation_override(self) -> Optional[bool]:
        """None = each video's own setting; True / False = force for this run."""
        return self._validation

    def set_validation_override(self, value: Optional[bool]) -> None:
        if value is not None:
            value = bool(value)
        if value != self._validation:
            self._validation = value
            self.refresh_resume()

    def default_output_dir(self) -> Optional[Path]:
        project = self.project
        if project is not None and project.base_dir is not None:
            return project.base_dir / "runs"
        if self.workspace is not None:
            return self.workspace.app_settings.output_dir()
        return None

    def output_dir(self) -> Optional[Path]:
        """Where run folders are created: the chosen folder, else ``runs/``."""
        return self._explicit_output or self.default_output_dir()

    def set_output_dir(self, folder: Optional[Path]) -> None:
        project = self.project
        self._explicit_output = Path(folder) if folder is not None else None
        if project is not None and project.file_path is not None:
            settings = self.workspace.app_settings
            settings.set_run_output_dir(project.file_path, folder)
            settings.sync()
        self.refresh_resume()

    def choose_output_dir(self) -> bool:
        """Ask for the output folder; False when cancelled or not allowed."""
        if self.is_running or self.workspace is None:
            return False
        start = self.output_dir()
        chosen = self.workspace.prompts.choose_folder(
            "Choose the output folder",
            str(start if start is not None else self.workspace.app_settings
                .dialog_dir("output")),
        )
        if chosen is None:
            return False
        self.set_output_dir(chosen)
        return True

    def scope_keys(self, scope: Optional[str] = None) -> List[str]:
        project = self.project
        if project is None:
            return []
        return scope_keys(
            project, scope or self._scope, self.workspace.selected_keys()
        )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    def refresh_resume(self) -> None:
        project = self.project
        offer = ResumeOffer()
        if project is not None and not self.is_running:
            offer = find_resumable(
                project, self.output_dir(), self.validation_override()
            )
        self._resume = offer
        self.state_changed.emit()

    @property
    def resume_offer(self) -> ResumeOffer:
        return self._resume

    def resume_text(self) -> str:
        offer = self._resume
        if offer:
            text = (
                f"{len(offer.resumable)} interrupted video(s) can continue "
                "from their checkpoint."
            )
            if offer.changed:
                text += (
                    f" {len(offer.changed)} more changed settings and would "
                    "start over."
                )
            return text
        if offer.changed:
            return (
                f"{len(offer.changed)} interrupted video(s) have changed "
                "settings; running them starts over."
            )
        return ""

    def start_resume(self) -> bool:
        """Run the videos of an interrupted run again, from their checkpoints."""
        if self.is_running or not self._resume:
            return False
        self.last_report = None
        return self._start(list(self._resume.keys), strict=False)

    # ------------------------------------------------------------------
    # Starting
    # ------------------------------------------------------------------
    def start_run(self) -> bool:
        """Check the plan for the chosen scope and start the analysis."""
        project = self.project
        if project is None or self.is_running or self._checking:
            return False
        self.last_report = None
        keys = self.scope_keys()
        if self._scope == SCOPE_SELECTED and not keys:
            self.workspace.prompts.info(
                "Nothing selected",
                "Select the videos to analyse in the Videos stage.",
            )
            return False
        return self._start(keys, strict=self._scope == SCOPE_SELECTED)

    def prepare(self, keys: List[str], strict: bool) -> RunPreparation:
        """Build and check the plan (pre-flight included)."""
        project = self.project
        output = self.output_dir()
        self._checking = True
        self.state_changed.emit()

        def on_progress(done: int, total: int, name: str) -> None:
            self.status_text = (
                f"Checking {name}..." if name else "Finishing checks..."
            )
            self.progress_maximum = max(total, 1)
            self.progress_value = done
            self.progress_changed.emit()
            QApplication.processEvents()

        try:
            self.status_text = "Checking the videos..."
            self.progress_changed.emit()
            return prepare_run(
                project, keys, output, strict=strict,
                create_validation_video=self.validation_override(),
                progress=on_progress,
            )
        finally:
            self._checking = False
            self._reset_progress()
            self.state_changed.emit()

    def _start(self, keys: List[str], strict: bool) -> bool:
        prompts = self.workspace.prompts
        output = self.output_dir()
        if output is None:
            chosen = prompts.choose_folder(
                "Choose the output folder",
                self.workspace.app_settings.dialog_dir("output"),
            )
            if chosen is None:
                return False
            self.set_output_dir(chosen)
            output = chosen
        report = self.prepare(keys, strict)
        self.last_report = report
        if report.issues:
            prompts.error(
                "Cannot start the analysis", format_problems(report.issues)
            )
            return False
        if not report.plan:
            prompts.info("Nothing to analyse", "No video of this scope is ready.")
            return False
        if report.warnings:
            count = len(report.plan)
            noun = "video" if count == 1 else "videos"
            if not prompts.confirm(
                "Ready with warnings",
                f"Ready to analyse {count} {noun} ({report.maze.name}).\n\n"
                + format_problems(report.warnings),
                accept="Start analysis",
                informative=f"Results go into a new folder inside {output}",
            ):
                return False
        return self._launch(report, output)

    def worker_count(self, video_count: int) -> int:
        chosen = self.project.run_options.parallel_workers
        if chosen <= 0:
            return default_worker_count(video_count)
        return max(1, min(chosen, video_count))

    def _launch(self, report: RunPreparation, output: Path) -> bool:
        project = self.project
        prompts = self.workspace.prompts
        try:
            Path(output).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            prompts.error(
                "Cannot start the analysis",
                f"Cannot create the output folder:\n{exc}",
            )
            return False
        workers = self.worker_count(len(report.plan))
        thread = QThread()
        try:
            worker = AnalysisWorker(
                report.plan, Path(output), report.maze,
                workers=workers,
                sample_metadata=report.sample_metadata,
                bin_seconds=project.run_options.time_bin_seconds,
            )
        except Exception as exc:
            prompts.error(
                "Cannot start the analysis",
                f"Could not create the run output folder:\n{exc}",
            )
            return False
        self._sequential = workers <= 1 or len(report.plan) <= 1
        self._base_paths = {
            f"Video: {name}": path
            for path, name in worker.output_base_names.items()
        }
        self._current_path = None
        self._last_run_dir = worker.output_dir
        self._last_outcome = None
        self._worker_eta = None
        self._run_started = time.monotonic()
        self._close_requested = False
        self._force_close = False
        self._cancelling = False
        self._log_entries.clear()
        self.log_cleared.emit()
        self.summary = None
        self.progress_model.reset_rows(list(report.plan))
        self.progress_maximum = 100
        self.progress_value = 0
        self.status_text = "Starting the analysis..."
        self.eta_text = "Batch ETA: calculating..."
        self.progress_changed.emit()

        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_finished)
        worker.detailed_progress.connect(self._on_detailed_progress)
        worker.log.connect(self._on_worker_log)
        worker.video_status.connect(self._on_video_status)
        # Qt frees both objects once the thread has stopped; _on_finished
        # drops our references first so nothing touches them afterwards.
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        _live_threads.add(thread)
        thread.finished.connect(lambda t=thread: _live_threads.discard(t))
        self._thread, self._worker = thread, worker

        self.workspace.set_busy(
            True, f"Analysing {len(report.plan)} video(s)..."
        )
        thread.start()
        self.state_changed.emit()
        self.run_started.emit()
        return True

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------
    def _reset_progress(self) -> None:
        self.progress_maximum = 100
        self.progress_value = 0
        self.status_text = "Ready"
        self.eta_text = ""
        self.progress_changed.emit()

    def _on_video_status(self, video: str, status: str, message: str) -> None:
        path = Path(video)
        self.progress_model.set_status(path, status, message)
        if status == STATUS_RUNNING:
            self._current_path = path

    def current_row(self) -> int:
        """The row of the video being analysed (-1 when none)."""
        if self._current_path is None:
            return -1
        return self.progress_model.row_of(self._current_path)

    def _on_detailed_progress(self, value: int, main: str, sub: str) -> None:
        self.progress_value = max(0, min(100, int(value)))
        self.status_text = main
        path = self._base_paths.get(main) or self._current_path
        if path is not None:
            self._update_video(path, int(value), sub)
        self._update_eta(int(value), sub)
        self.progress_changed.emit()

    def _update_video(self, path: Path, batch_pct: int, sub: str) -> None:
        model = self.progress_model
        total = max(len(model.paths()), 1)
        pct: Optional[int] = None
        if self._sequential:
            row = model.row_of(path)
            if row >= 0:
                pct = int(max(0, min(100, batch_pct * total - row * 100)))
        match = _VIDEO_ETA.search(sub or "")
        eta = match.group(1).strip() if match else ""
        if eta.lower().startswith("calculating"):
            eta = ""
        message = _VIDEO_ETA.sub("", sub or "").strip(" |")
        model.set_progress(path, pct, eta, message or (sub or ""))

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._run_started

    def _update_eta(self, percent: int, sub_text: str = "") -> None:
        """The worker's own batch ETA, else one from the elapsed time."""
        reported = ux_helpers.extract_batch_eta(sub_text)
        if reported:
            self._worker_eta = reported
        elapsed = self.elapsed_seconds()
        eta_text = self._worker_eta
        if eta_text is None:
            remaining = ux_helpers.estimate_batch_eta(elapsed, percent)
            if remaining is not None:
                eta_text = "~" + ux_helpers.format_duration(remaining)
        self.eta_text = (
            f"Batch ETA: {eta_text or 'calculating...'}   "
            f"(elapsed {ux_helpers.format_duration(elapsed)})"
        )

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------
    def _on_worker_log(self, text: str) -> None:
        level = ux_helpers.level_for_worker_message(text)
        self._log_entries.append((level, text))
        if len(self._log_entries) > MAX_LOG_ENTRIES:
            del self._log_entries[: len(self._log_entries) - MAX_LOG_ENTRIES]
        self.log_added.emit(level, text)

    @property
    def log_entries(self) -> List[Tuple[int, str]]:
        return list(self._log_entries)

    def save_log(self) -> bool:
        """Write the complete (unfiltered) log to a text file."""
        if self.workspace is None or not self._log_entries:
            return False
        start = self.output_dir() or Path(".")
        path = self.workspace.prompts.save_file(
            "Save log", str(Path(start) / "analysis_log.txt"),
            "Text Files (*.txt);;All Files (*.*)",
        )
        if path is None:
            return False
        try:
            Path(path).write_text(
                "\n".join(text for _, text in self._log_entries),
                encoding="utf-8",
            )
        except OSError as exc:
            self.workspace.prompts.error(
                "Save log", f"Could not save the log:\n{exc}"
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------
    def request_cancel(self) -> bool:
        """Ask for confirmation, then stop the run (progress is saved)."""
        if not self.is_running or self._close_requested or self._cancelling:
            return False
        if not self.workspace.prompts.confirm(
            "Stop the analysis?",
            "Stop the running analysis?",
            accept="Stop analysis",
            informative=(
                "Progress is saved. The video in progress can be resumed "
                "from its checkpoint later."
            ),
            destructive=True,
        ):
            return False
        self._cancel_worker()
        return True

    def _cancel_worker(self) -> None:
        worker = self._worker
        if worker is None:
            return
        worker.stop()
        self._cancelling = True
        self.status_text = "Cancelling..."
        self.progress_changed.emit()
        self.state_changed.emit()

    @property
    def can_cancel(self) -> bool:
        return self.is_running and not self._close_requested and not self._cancelling

    # ------------------------------------------------------------------
    # Finishing
    # ------------------------------------------------------------------
    def _on_finished(self, outcome: BatchRunOutcome) -> None:
        thread = self._thread
        # Drop our references before the thread's finished signal lets Qt
        # delete the worker and thread.
        self._thread = None
        self._worker = None
        if thread is not None:
            thread.quit()
            thread.wait(5000)
        closing = self._close_requested
        self._cancelling = False
        self._force_timer.stop()
        self._last_outcome = outcome
        run_dir = outcome.output_dir or self._last_run_dir
        self._last_run_dir = run_dir

        for video, (status, message) in outcome_statuses(
            (path for path, _ in outcome.results),
            outcome.failures, outcome.quality_warnings,
        ).items():
            self.progress_model.set_status(video, status, message)
        problem = self._record_run(run_dir)

        self.eta_text = ""
        if outcome.cancelled:
            self.status_text = "Cancelled - checkpoint saved"
        else:
            self.progress_value = self.progress_maximum = 100
            self.status_text = (
                "Finished with errors" if outcome.failures
                else "Finished - quality review required"
                if outcome.quality_warnings else "Finished"
            )
        self.summary = RunSummary(
            completed=len(outcome.results),
            failed=len(outcome.failures),
            warnings=len(outcome.quality_warnings),
            cancelled=bool(outcome.cancelled),
            run_dir=run_dir,
            failures=[(Path(p).name, str(m)) for p, m in outcome.failures],
            quality_warnings=[
                (Path(p).name, str(m)) for p, m in outcome.quality_warnings
            ],
            problem=problem,
        )
        self.progress_changed.emit()
        self.workspace.set_busy(False)
        self.refresh_resume()
        self.run_finished.emit(outcome)
        if closing:
            self._close_timer.start()

    def _record_run(self, run_dir: Optional[Path]) -> str:
        """Add the run to the project; returns a problem text ('' when fine)."""
        project = self.project
        if project is None or run_dir is None:
            return ""
        try:
            project.record_run(run_dir)
        except (ProjectError, OSError, ValueError) as exc:
            logger.warning("Could not record the run", exc_info=True)
            return f"The run could not be added to the project: {exc}"
        return ""

    def dismiss_summary(self) -> None:
        self.summary = None
        self.state_changed.emit()

    def open_run_folder(self) -> bool:
        run_dir = self._last_run_dir
        if run_dir is None or self.workspace is None:
            return False
        return bool(self.workspace.prompts.open_folder(run_dir))

    @property
    def has_results(self) -> bool:
        """Whether the last run folder holds results the Results stage can read."""
        run_dir = self._last_run_dir
        return (
            run_dir is not None and Path(run_dir).is_dir()
            and (Path(run_dir) / "manifest.json").is_file()
        )

    # ------------------------------------------------------------------
    # Closing the window during a run
    # ------------------------------------------------------------------
    def _close_guard(self) -> bool:
        """Veto closing while a run is in progress: cancel first, close later."""
        if not self.is_running:
            return True
        if self._force_close:
            # Last resort after the worker failed to stop in time.
            self._stop_thread_forcefully()
            return True
        if self._close_requested:
            return False        # already stopping; the window closes by itself
        if not self.workspace.prompts.confirm(
            "Analysis in progress",
            "Quit and cancel the analysis?",
            accept="Cancel analysis and quit",
            informative=(
                "The analysis stops and its progress is saved; the window "
                "closes when it has stopped."
            ),
            destructive=True,
        ):
            return False
        # Never block the GUI thread waiting for the worker: ask it to stop,
        # keep the window open, and close from _on_finished.
        self._close_requested = True
        self._cancel_worker()
        self.status_text = "Stopping the analysis..."
        self.summary = RunSummary(cancelled=True, stopping_for_close=True)
        self.progress_changed.emit()
        self._force_timer.start(self.FORCE_QUIT_DELAY_MS)
        self.state_changed.emit()
        return False

    def _close_window(self) -> None:
        if self.workspace is not None:
            self.workspace.close()

    def _offer_force_quit(self) -> None:
        """The worker did not stop in time; let the user stop waiting."""
        if not (self._close_requested and self.is_running):
            return
        if self.workspace.prompts.confirm(
            "Analysis still stopping",
            "The analysis has not stopped yet.\n\nForce quit now? The "
            "checkpoint of the video in progress may be stale, so the next "
            "run may redo some of its frames, and any video or plot being "
            "written will be incomplete.",
            accept="Force quit",
            destructive=True,
        ):
            self._force_close = True
            self.workspace.close()
        else:
            self._force_timer.start(self.FORCE_QUIT_DELAY_MS)

    def _stop_thread_forcefully(self) -> None:
        thread = self._thread
        if thread is None:
            return
        thread.quit()
        if not thread.wait(2000):
            logger.warning("Analysis thread did not stop; terminating it")
            thread.terminate()
            thread.wait(2000)


__all__ = ["RunController", "RunSummary", "FORCE_QUIT_DELAY_MS", "_live_threads"]
