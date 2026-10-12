"""
gui/worker.py
Qt-facing analysis worker.

All analysis lives in the Qt-free ``core`` modules: ``core.pipeline`` runs one
video and ``core.batch_runner`` runs a batch (sequentially or in parallel
child processes) and writes the batch summary and manifest. This class only
owns the run folder, translates reporter calls into Qt signals and exposes
``stop()``.
"""
import logging
from pathlib import Path
from typing import Dict, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from checkpoint_manager import CheckpointManager
from core.batch_runner import BatchRunner, BatchRunOutcome
from core.models import BatchSettings
from core.run_outputs import (
    RUN_PLAN_FILENAME, apply_effective_parameters, create_run_folder,
    output_base_names, run_timestamp, write_run_manifest, write_run_plan,
)
from mazes.base_maze import Maze

logger = logging.getLogger(__name__)

__all__ = ["AnalysisWorker", "BatchRunOutcome", "RUN_PLAN_FILENAME"]


class _SignalReporter:
    """Batch reporter that re-emits everything as the worker's Qt signals."""

    def __init__(self, worker: "AnalysisWorker"):
        self._worker = worker

    def progress(self, pct, main, sub=""):
        self._worker.emit_progress(pct, main, sub)

    def log(self, message):
        self._worker.log.emit(message)

    def video_status(self, video_path, status, message):
        self._worker.video_status.emit(str(video_path), status, message)


class _WorkerCancel:
    """Cancel token that reads the worker's flag (set by ``stop()``)."""

    def __init__(self, worker: "AnalysisWorker"):
        self._worker = worker

    def is_set(self) -> bool:
        return self._worker.is_cancelled


class AnalysisWorker(QObject):
    # Signals
    progress = pyqtSignal(int, str)
    detailed_progress = pyqtSignal(int, str, str)
    log = pyqtSignal(str)
    finished = pyqtSignal(object)
    # (video path, "running" | "done" | "warning" | "failed" | "queued", message)
    video_status = pyqtSignal(str, str, str)

    def __init__(
        self,
        analysis_plan: Dict[Path, BatchSettings],
        output_dir: Path,
        maze: Maze,
        workers: int = 1,
    ):
        """``workers`` > 1 analyses several videos at once in child processes."""
        super().__init__()
        self.analysis_plan = analysis_plan
        self.output_root = output_dir
        self.maze = maze
        self.workers = max(1, int(workers))
        apply_effective_parameters(self.analysis_plan, self.maze)
        self.is_cancelled = False
        self.checkpoint_mgr = CheckpointManager(output_dir)
        self.output_dir = create_run_folder(output_dir)
        self.run_started_at = run_timestamp()
        self.output_base_names = output_base_names(list(analysis_plan))

    def stop(self):
        """Request the worker to stop processing."""
        self.log.emit("Cancellation requested by user - saving progress...")
        self.is_cancelled = True

    def emit_progress(self, overall_pct: int, main_task: str, sub_task: str = ""):
        """Helper to emit detailed progress signals."""
        self.detailed_progress.emit(overall_pct, main_task, sub_task)
        self.progress.emit(overall_pct, main_task)

    def run(self):
        """Run the batch and emit ``finished`` exactly once."""
        runner = BatchRunner(
            self.analysis_plan,
            self.output_root,
            self.output_dir,
            self.maze,
            self.output_base_names,
            _SignalReporter(self),
            _WorkerCancel(self),
            workers=self.workers,
            checkpoint_mgr=self.checkpoint_mgr,
            started_at=self.run_started_at,
            write_plan=self._write_run_plan,
            write_manifest=self._write_run_manifest,
        )
        outcome: Optional[BatchRunOutcome] = None
        try:
            outcome = runner.run()
        except Exception as exc:
            logger.error("Batch run failed", exc_info=True)
            self.log.emit(f"✗ Batch run failed: {exc}")
            outcome = BatchRunOutcome(
                results=runner.results,
                failures=runner.failures + [(Path("batch"), str(exc))],
                quality_warnings=runner.quality_warnings,
                cancelled=runner.cancelled,
                output_dir=self.output_dir,
            )
        finally:
            if outcome is not None:
                self.finished.emit(outcome)

    def _write_run_plan(self):
        """Save the exact per-video settings of this run as a batch plan."""
        write_run_plan(self.analysis_plan, self.output_dir)

    def _write_run_manifest(
        self, results, failures, quality_warnings, cancelled
    ):
        """Write a compact index so a run can be understood at a glance."""
        write_run_manifest(
            self.output_dir,
            self.maze,
            self.run_started_at,
            len(self.analysis_plan),
            self.output_base_names,
            results,
            failures,
            quality_warnings,
            cancelled,
        )
