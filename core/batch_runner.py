"""
core/batch_runner.py
Qt-free batch orchestration: runs every video of a plan, sequentially in this
process or in parallel child processes, then writes the batch summary and the
run manifest in the original video order.

The GUI worker and the headless CLI both drive ``BatchRunner``; they differ
only in the ``BatchReporter`` (Qt signals vs. console) and the cancel token.
"""
import logging
import multiprocessing
import os
import queue as queue_module
import time
from concurrent.futures import (
    CancelledError, FIRST_COMPLETED, ProcessPoolExecutor, wait,
)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Protocol, Tuple

from checkpoint_manager import CheckpointManager, ProgressEstimator
from core.models import BatchSettings
from core.pipeline import (
    CancelToken, VideoJob, VideoOutcome, init_child_process, run_video,
)
from core.run_outputs import write_run_manifest, write_run_plan
from core.sample_sheet import SampleMetadata
from core.time_bins import TimeBinTable
from core.video import VideoHandler
from mazes import create_maze
from mazes.base_maze import BaseAnalysisResult, Maze
from output import DataExporter

logger = logging.getLogger(__name__)

# Seconds between cancel-flag reads in a child process. Reading a Manager
# Event is an inter-process call, so it is not done once per frame.
CHILD_CANCEL_POLL_SECONDS = 0.1
# Seconds the parent waits for child results before draining progress again.
PARENT_POLL_SECONDS = 0.1
# Assumed frame count when a video cannot be probed for the batch ETA.
FALLBACK_FRAME_ESTIMATE = 1000


@dataclass
class BatchRunOutcome:
    results: List[Tuple[Path, BaseAnalysisResult]] = field(default_factory=list)
    failures: List[Tuple[Path, str]] = field(default_factory=list)
    quality_warnings: List[Tuple[Path, str]] = field(default_factory=list)
    cancelled: bool = False
    output_dir: Optional[Path] = None


class BatchReporter(Protocol):
    """Receives batch-level progress (``pct`` is 0-100 across the batch)."""

    def progress(self, pct: int, main: str, sub: str = "") -> None:
        ...

    def log(self, message: str) -> None:
        ...

    def video_status(self, video_path: Path, status: str, message: str) -> None:
        ...


def _total_ram_gb() -> Optional[float]:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return None


def default_worker_count(
    n_videos: int,
    cpu_count: Optional[int] = None,
    ram_gb: Optional[float] = None,
) -> int:
    """``min(cpu_count // 4, n_videos, ram_gb // 3)``, at least 1."""
    cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    memory = ram_gb if ram_gb is not None else _total_ram_gb()
    limits = [int(cpus) // 4, int(n_videos)]
    if memory is not None:
        limits.append(int(memory // 3))
    return max(1, min(limits))


def maze_rebuild_is_exact(maze: Maze, parameters: Mapping) -> bool:
    """Whether ``create_maze`` + the parameters reproduce ``maze`` exactly.

    A maze can carry state its parameter dict does not (for example the
    legacy-threshold note on Freestyle); such a maze must travel to child
    processes as a snapshot instead of being rebuilt.
    """
    try:
        rebuilt = create_maze(maze.name)
        if parameters:
            rebuilt.configure_algorithm_parameters(dict(parameters))
        return type(rebuilt) is type(maze) and vars(rebuilt) == vars(maze)
    except Exception:
        return False


# ----------------------------------------------------------------------
# Reporters
# ----------------------------------------------------------------------
class _SequentialVideoReporter:
    """Maps one video's local progress onto the batch bar and status."""

    def __init__(self, batch: BatchReporter, video_path: Path,
                 video_number: int, total_videos: int):
        self._batch = batch
        self._video_path = video_path
        self._number = video_number
        self._total = total_videos

    def progress(self, pct, main, sub=""):
        batch_pct = int(
            ((self._number - 1) + (pct / 100.0)) / self._total * 100
        )
        self._batch.progress(batch_pct, main, sub)

    def log(self, message):
        self._batch.log(message)

    def status(self, status, message=""):
        self._batch.video_status(self._video_path, status, message)


class _QueueVideoReporter:
    """Child-process reporter: forwards every event to the parent's queue."""

    def __init__(self, events, index: int):
        self._events = events
        self._index = index

    def progress(self, pct, main, sub=""):
        self._events.put(("progress", self._index, pct, main, sub))

    def log(self, message):
        self._events.put(("log", self._index, message))

    def status(self, status, message=""):
        self._events.put(("status", self._index, status, message))


class _PolledCancel:
    """Rate-limited view of a (Manager) Event shared with the parent."""

    def __init__(self, event, interval: float = CHILD_CANCEL_POLL_SECONDS):
        self._event = event
        self._interval = interval
        self._last_poll = float("-inf")
        self._cancelled = False

    def is_set(self) -> bool:
        if self._cancelled:
            return True
        now = time.monotonic()
        if now - self._last_poll >= self._interval:
            self._last_poll = now
            try:
                self._cancelled = bool(self._event.is_set())
            except (OSError, EOFError):
                # The parent is gone: nothing is listening any more.
                self._cancelled = True
        return self._cancelled


def run_video_in_child(
    job: VideoJob, events, cancel_event, index: int
) -> VideoOutcome:
    """Process-pool entry point (module level so 'spawn' can import it)."""
    cancel = _PolledCancel(cancel_event)
    if cancel.is_set():
        return VideoOutcome(job.video_path, cancelled=True)
    return run_video(job, _QueueVideoReporter(events, index), cancel)


# ----------------------------------------------------------------------
# The runner
# ----------------------------------------------------------------------
class BatchRunner:
    """Runs a batch plan and produces the run folder's batch-level outputs."""

    def __init__(
        self,
        analysis_plan: Mapping[Path, BatchSettings],
        output_root: Path,
        run_dir: Path,
        maze: Maze,
        base_names: Mapping[Path, str],
        reporter: BatchReporter,
        cancel: CancelToken,
        workers: int = 1,
        checkpoint_mgr: Optional[CheckpointManager] = None,
        started_at: str = "",
        write_plan: Optional[Callable[[], None]] = None,
        write_manifest: Optional[Callable[..., None]] = None,
        sample_metadata: Optional[SampleMetadata] = None,
        bin_seconds: float = 0.0,
    ):
        self.analysis_plan = analysis_plan
        self.output_root = output_root
        self.run_dir = run_dir
        self.maze = maze
        self.base_names = base_names
        self.reporter = reporter
        self.cancel = cancel
        self.workers = max(1, int(workers))
        self.checkpoint_mgr = checkpoint_mgr or CheckpointManager(output_root)
        self.started_at = started_at
        # Run-level options: neither is part of BatchSettings, so neither
        # changes the checkpoint settings hash.
        self.sample_metadata = sample_metadata
        self.bin_seconds = max(0.0, float(bin_seconds or 0.0))
        self.time_bins: Dict[Path, TimeBinTable] = {}
        self._write_plan = write_plan or (
            lambda: write_run_plan(self.analysis_plan, self.run_dir)
        )
        self._write_manifest = write_manifest or self._default_write_manifest
        self.results: List[Tuple[Path, BaseAnalysisResult]] = []
        self.failures: List[Tuple[Path, str]] = []
        self.quality_warnings: List[Tuple[Path, str]] = []
        self.cancelled = False

    def _default_write_manifest(
        self, results, failures, quality_warnings, cancelled
    ):
        write_run_manifest(
            self.run_dir, self.maze, self.started_at,
            len(self.analysis_plan), self.base_names,
            results, failures, quality_warnings, cancelled,
            sample_metadata=self.sample_metadata,
        )

    # -- helpers -------------------------------------------------------
    def _make_job(self, video_number: int, video_path: Path,
                  settings: BatchSettings, total_videos: int,
                  **extra) -> VideoJob:
        return VideoJob(
            video_path=video_path,
            settings=settings,
            maze_name=self.maze.name,
            algorithm_parameters=self.maze.get_algorithm_parameters(),
            output_dir=self.run_dir / self.base_names[video_path],
            base_name=self.base_names[video_path],
            checkpoint_root=self.output_root,
            animal_metadata=(
                None if self.sample_metadata is None
                else self.sample_metadata.values_for(video_path)
            ),
            bin_seconds=self.bin_seconds,
            video_number=video_number,
            total_videos=total_videos,
            **extra,
        )

    def _ingest(self, video_path: Path, outcome: VideoOutcome) -> None:
        self.quality_warnings.extend(
            (video_path, message) for message in outcome.warnings
        )
        if outcome.cancelled:
            self.cancelled = True
        elif outcome.failure is not None:
            self.failures.append((video_path, outcome.failure))
        elif outcome.result is not None:
            self.results.append((video_path, outcome.result))
            if outcome.time_bins is not None:
                self.time_bins[video_path] = outcome.time_bins

    # -- public --------------------------------------------------------
    def run(self) -> BatchRunOutcome:
        videos = list(self.analysis_plan.items())
        total_videos = len(videos)
        self._write_plan()
        self.reporter.log("Preparing batch analysis...")

        parallel_workers = min(self.workers, total_videos)
        try:
            if total_videos > 1 and parallel_workers > 1:
                self._run_parallel(videos, parallel_workers)
            else:
                self._run_sequential(videos)
        finally:
            self._finish(total_videos)
        return BatchRunOutcome(
            results=self.results,
            failures=self.failures,
            quality_warnings=self.quality_warnings,
            cancelled=self.cancelled,
            output_dir=self.run_dir,
        )

    def _finish(self, total_videos: int) -> None:
        if self.results:
            try:
                DataExporter.export_batch_summary_csv(
                    self.results, self.run_dir, self.maze,
                    metadata=self.sample_metadata,
                )
            except Exception as exc:
                self.failures.append((Path("batch_summary.csv"), str(exc)))
                logger.error("Batch summary export failed", exc_info=True)

        tables = [
            (path, self.time_bins[path]) for path, _ in self.results
            if path in self.time_bins
        ]
        if tables:
            try:
                DataExporter.export_batch_time_bins_csv(
                    tables, self.run_dir, self.maze.name,
                    metadata=self.sample_metadata,
                )
            except Exception as exc:
                self.failures.append((Path("batch_time_bins.csv"), str(exc)))
                logger.error("Batch time bins export failed", exc_info=True)

        try:
            self._write_manifest(
                self.results, self.failures, self.quality_warnings,
                self.cancelled,
            )
        except Exception as exc:
            self.failures.append((Path("manifest.json"), str(exc)))
            logger.error("Run manifest export failed", exc_info=True)
        finally:
            self.reporter.log(
                f"\n{'='*60}\nBatch Analysis Complete\n{'='*60}\n"
                f"Processed: {len(self.results)}/{total_videos}"
            )

    # -- sequential ------------------------------------------------------
    def _run_sequential(self, videos) -> None:
        total_videos = len(videos)

        # Total frames feed the batch-wide ETA shown in the progress text.
        total_batch_frames = 0
        for v_path, v_settings in videos:
            try:
                vh = VideoHandler(v_path)
                s = int(v_settings.start_time * vh.fps)
                e = int(v_settings.end_time * vh.fps) if v_settings.end_time else vh.total_frames
                total_batch_frames += max(0, e - s)
            except Exception:
                total_batch_frames += FALLBACK_FRAME_ESTIMATE
        global_est = ProgressEstimator(total_batch_frames)
        global_offset = 0

        for video_num, (video_path, settings) in enumerate(videos, 1):
            if self.cancel.is_set():
                self.reporter.log(
                    "Analysis cancelled - checkpoints saved for resume"
                )
                self.cancelled = True
                break

            job = self._make_job(
                video_num, video_path, settings, total_videos,
                batch_estimator=global_est,
                batch_frame_offset=global_offset,
                maze=self.maze,
            )
            outcome = run_video(
                job,
                _SequentialVideoReporter(
                    self.reporter, video_path, video_num, total_videos
                ),
                self.cancel,
                checkpoint_mgr=self.checkpoint_mgr,
            )
            self._ingest(video_path, outcome)
            if outcome.cancelled:
                break
            global_offset += outcome.frames_done

    # -- parallel --------------------------------------------------------
    def _run_parallel(self, videos, workers: int) -> None:
        total_videos = len(videos)
        context = multiprocessing.get_context("spawn")
        local_pct = [0.0] * total_videos
        outcomes: List[Optional[VideoOutcome]] = [None] * total_videos
        snapshot = (
            None if maze_rebuild_is_exact(
                self.maze, self.maze.get_algorithm_parameters()
            ) else self.maze
        )
        self.reporter.log(
            f"Running {total_videos} videos on {workers} parallel workers"
        )

        manager = context.Manager()
        try:
            events = manager.Queue()
            cancel_event = manager.Event()

            def drain() -> None:
                while True:
                    try:
                        message = events.get_nowait()
                    except queue_module.Empty:
                        return
                    self._relay(message, videos, local_pct)

            with ProcessPoolExecutor(
                max_workers=workers, mp_context=context,
                initializer=init_child_process,
            ) as pool:
                futures = {}
                for index, (video_path, settings) in enumerate(videos):
                    job = self._make_job(
                        index + 1, video_path, settings, total_videos,
                        maze=snapshot,
                    )
                    future = pool.submit(
                        run_video_in_child, job, events, cancel_event, index
                    )
                    futures[future] = index

                pending = set(futures)
                while pending:
                    if self.cancel.is_set() and not cancel_event.is_set():
                        cancel_event.set()
                        # Videos no child has picked up yet never start.
                        for future in list(pending):
                            future.cancel()
                    done, pending = wait(
                        pending, timeout=PARENT_POLL_SECONDS,
                        return_when=FIRST_COMPLETED,
                    )
                    drain()
                    for future in done:
                        index = futures[future]
                        outcomes[index] = self._collect(
                            future, videos[index][0]
                        )
                        local_pct[index] = 100.0
                drain()
        finally:
            manager.shutdown()

        # Aggregate in the original video order.
        not_started = False
        for index, (video_path, _) in enumerate(videos):
            outcome = outcomes[index]
            if outcome is None:
                not_started = True
                continue
            self._ingest(video_path, outcome)
        if not_started:
            self.reporter.log(
                "Analysis cancelled - checkpoints saved for resume"
            )
            self.cancelled = True

    @staticmethod
    def _collect(future, video_path: Path) -> Optional[VideoOutcome]:
        try:
            return future.result()
        except CancelledError:
            return None
        except BaseException as exc:  # a crashed child must not sink the batch
            logger.error("Parallel worker failed", exc_info=True)
            return VideoOutcome(
                video_path,
                failure=str(exc) or type(exc).__name__,
            )

    def _relay(self, message, videos, local_pct: List[float]) -> None:
        """Re-emit one child event on the parent's reporter."""
        kind, index = message[0], message[1]
        video_path = videos[index][0]
        if kind == "progress":
            _, _, pct, main, sub = message
            local_pct[index] = max(local_pct[index], float(pct))
            self.reporter.progress(
                int(sum(local_pct) / len(local_pct)), main, sub
            )
        elif kind == "log":
            text = message[2]
            if not text.startswith("\n"):  # banners already name the video
                text = f"[{video_path.name}] {text}"
            self.reporter.log(text)
        elif kind == "status":
            self.reporter.video_status(video_path, message[2], message[3])
