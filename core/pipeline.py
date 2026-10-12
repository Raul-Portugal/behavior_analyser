"""
core/pipeline.py
Qt-free analysis of ONE video.

``run_video`` does everything the GUI worker used to do per video: tracked or
TST analysis, checkpoints, metrics, exports, plots, the optional validation
video and the quality warnings. It reports through a ``ProgressReporter`` and
polls a ``CancelToken``, so the same code runs on the GUI worker thread, in a
child process of a parallel batch, and from the headless CLI.
"""
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import cv2
import numpy as np

from core.models import AppConfig, BatchSettings
from core.video import (
    VideoHandler,
    ReferenceFrameGenerator,
    SafeVideoWriter,
    BufferedVideoReader,
    position_capture_at_frame,
)
from core.detection import DetectionEngine, DetectionQualityMonitor
from core.analysis_engine import Analyzer
from core.time_bins import TimeBinTable, compute_time_bins
from core.tst_display import blend_polygon, tst_display_style
from core.tst_motion_engine import (
    TST_VALIDATION_FRAME_STEP, TSTFramePipeline, TSTMotionEngine,
)

# --- LEGACY/UTILITY IMPORTS ---
from core.analysis_engine import apply_scale_to_result
from mazes import create_maze
from mazes.base_maze import BaseAnalysisResult, Maze
from mazes.tst import TST, describe_window_truncation
from output import DataExporter, Visualizer
from checkpoint_manager import CheckpointManager, ProgressEstimator

logger = logging.getLogger(__name__)

# Threads OpenCV may use inside each child process of a parallel batch, so
# several videos do not oversubscribe the machine.
CHILD_PROCESS_CV_THREADS = 2


@contextmanager
def _open_capture(video_path):
    """Open a cv2.VideoCapture that is released even when a phase aborts."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        yield cap
    finally:
        cap.release()


# ----------------------------------------------------------------------
# Public contract
# ----------------------------------------------------------------------
class ProgressReporter(Protocol):
    """Where one video's progress goes (signals, a queue, the console)."""

    def progress(self, pct: float, main: str, sub: str = "") -> None:
        """``pct`` is 0-100 within this video."""

    def log(self, message: str) -> None:
        ...

    def status(self, status: str, message: str = "") -> None:
        """``status``: "running" | "done" | "warning" | "failed" | "queued"."""


class CancelToken(Protocol):
    """Anything with ``is_set()``: threading and multiprocessing Events."""

    def is_set(self) -> bool:
        ...


class NullReporter:
    """Reporter that discards everything."""

    def progress(self, pct, main, sub=""):
        pass

    def log(self, message):
        pass

    def status(self, status, message=""):
        pass


@dataclass
class VideoJob:
    """Everything needed to analyse one video; picklable for child processes."""
    video_path: Path
    settings: BatchSettings
    maze_name: str
    algorithm_parameters: Dict[str, Any]
    output_dir: Path          # this video's output folder inside the run
    base_name: str
    checkpoint_root: Path     # the CheckpointManager root (not the run folder)
    video_number: int = 1     # position in the batch, for the log header
    total_videos: int = 1
    # Batch-wide ETA bookkeeping. Only meaningful for a sequential in-process
    # batch, where the estimator spans every video.
    batch_estimator: Optional[ProgressEstimator] = None
    batch_frame_offset: int = 0
    # A pre-configured maze to use instead of rebuilding one from
    # ``maze_name`` and ``algorithm_parameters`` (state a parameter dict does
    # not capture, such as a legacy Freestyle threshold note).
    maze: Optional[Maze] = None
    # Sample-sheet values of this video (run-level, not part of the settings
    # hash); None when the run has no sample sheet.
    animal_metadata: Optional[Dict[str, str]] = None
    # Time-bin length in seconds (0 = no time bins). Run-level, like the
    # metadata: not part of the settings, so not part of the checkpoint hash.
    bin_seconds: float = 0.0


@dataclass
class VideoOutcome:
    """What happened to one video. Picklable."""
    video_path: Path
    result: Optional[BaseAnalysisResult] = None
    warnings: List[str] = field(default_factory=list)
    failure: Optional[str] = None
    cancelled: bool = False
    frames_done: int = 0
    # Per-bin metrics when the job asked for time bins (computed once here so
    # a parallel batch can write the batch-level file from the parent).
    time_bins: Optional[TimeBinTable] = None

    @property
    def succeeded(self) -> bool:
        return self.result is not None and self.failure is None and not self.cancelled


def build_maze(job: VideoJob) -> Maze:
    """The maze for a job: its snapshot, or a fresh one from name + parameters."""
    if job.maze is not None:
        return job.maze
    maze = create_maze(job.maze_name)
    parameters = job.algorithm_parameters
    if parameters:
        maze.configure_algorithm_parameters(parameters)
    return maze


def init_child_process() -> None:
    """Process-pool initializer: keep OpenCV from oversubscribing the CPU."""
    cv2.setNumThreads(CHILD_PROCESS_CV_THREADS)


def run_video(
    job: VideoJob,
    reporter: ProgressReporter,
    cancel: CancelToken,
    checkpoint_mgr: Optional[CheckpointManager] = None,
) -> VideoOutcome:
    """Analyse one video. Never raises for analysis errors or cancellation.

    A failure comes back as ``VideoOutcome.failure``; a cancellation as
    ``VideoOutcome.cancelled`` (the checkpoint is saved so it can resume).
    ``checkpoint_mgr`` lets an in-process caller share its manager; otherwise
    one is created under ``job.checkpoint_root``.
    """
    if checkpoint_mgr is None:
        job.checkpoint_root.mkdir(parents=True, exist_ok=True)
        checkpoint_mgr = CheckpointManager(job.checkpoint_root)
    pipeline = VideoPipeline(
        job, build_maze(job), reporter, cancel, checkpoint_mgr
    )
    return pipeline.run()


class VideoPipeline:
    """The per-video analysis, with its collaborators injected."""

    def __init__(
        self,
        job: VideoJob,
        maze: Maze,
        reporter: ProgressReporter,
        cancel: CancelToken,
        checkpoint_mgr: CheckpointManager,
    ):
        self.job = job
        self.maze = maze
        self.reporter = reporter
        self.cancel = cancel
        self.checkpoint_mgr = checkpoint_mgr
        self.base_name = job.base_name
        self.global_est = job.batch_estimator
        self.global_offset = job.batch_frame_offset
        self.time_bins: Optional[TimeBinTable] = None
        self._export_warnings: List[str] = []

    def _raise_if_cancelled(self):
        """Cooperative cancellation point for phases outside the frame loops."""
        if self.cancel.is_set():
            raise InterruptedError("Analysis cancelled")

    def _progress(self, pct: float, task: str):
        self.reporter.progress(pct, f"Video: {self.base_name}", task)

    def run(self) -> VideoOutcome:
        job = self.job
        video_path = job.video_path
        settings = job.settings
        base_name = job.base_name
        config = AppConfig()
        quality_warnings: List[str] = []
        frames_done = 0

        self.checkpoint_mgr.reset_timer()
        video_output_dir = job.output_dir
        data_dir = video_output_dir / "data"
        plots_dir = video_output_dir / "plots"
        media_dir = video_output_dir / "media"
        self.reporter.log(
            f"\n{'='*60}\nAnalyzing {self.maze.name} video "
            f"{job.video_number}/{job.total_videos}: {video_path.name}\n"
            f"{'='*60}"
        )
        self.reporter.status("running", "")

        try:
            # Inside the try so an unwritable output folder is recorded
            # as a per-video failure instead of escaping.
            for directory in (data_dir, plots_dir):
                directory.mkdir(parents=True, exist_ok=True)

            # Initialize VideoHandler
            handler = VideoHandler(video_path)
            start_frame = int(settings.start_time * handler.fps)
            end_frame = int(settings.end_time * handler.fps) if settings.end_time else handler.total_frames

            # Checkpoint loading
            checkpoint = self.checkpoint_mgr.load_checkpoint(video_path, settings)
            resume_from_frame = (
                checkpoint.last_frame_processed + 1
                if checkpoint else start_frame
            )
            if checkpoint:
                self.reporter.log(
                    f"✓ Resuming from checkpoint (frame {resume_from_frame})"
                )

            # === BRANCH: TST vs STANDARD TRACKING ===
            if isinstance(self.maze, TST):
                # --- TST MOTION-DECOMPOSITION LOGIC ---
                self._progress(
                    10, "Decomposing rigid and articulated motion..."
                )

                result, frames_done = self._run_tst_analysis(
                    video_path, settings, start_frame, end_frame, handler,
                    checkpoint, resume_from_frame
                )
                tst_first_frame = max(start_frame, resume_from_frame)
                early_end = self._early_end_warning(
                    end_frame - tst_first_frame,
                    frames_done,
                    tst_first_frame,
                    end_frame,
                )
                if early_end:
                    quality_warnings.append(early_end)
                    self.reporter.log(f"⚠ {early_end}")
                    logger.warning("%s: %s", video_path.name, early_end)

                self._raise_if_cancelled()
                self._progress(60, "Applying Temporal Filters...")
                self.maze.calculate_metrics(result)
                for mouse_name, uncertain_time in result.uncertain_time.items():
                    total_time = (
                        result.valid_time.get(mouse_name, 0.0)
                        + uncertain_time
                    )
                    uncertain_rate = (
                        uncertain_time / total_time if total_time else 1.0
                    )
                    if uncertain_rate > 0.05:
                        quality_message = (
                            f"{mouse_name}: {uncertain_rate*100:.1f}% "
                            "of TST frames were unscorable"
                        )
                        quality_warnings.append(quality_message)
                    mean_confidence = result.mean_state_confidence.get(
                        mouse_name, 0.0
                    )
                    if mean_confidence < 0.60:
                        quality_warnings.append(
                            f"{mouse_name}: low mean TST state confidence "
                            f"({mean_confidence:.2f})"
                        )
                    qc = result.quality_metrics.get(mouse_name, {})
                    truncation = describe_window_truncation(
                        result, mouse_name
                    )
                    if truncation:
                        quality_warnings.append(truncation)
                    if qc.get("segmentation_failure_pct", 0.0) > 5.0:
                        quality_warnings.append(
                            f"{mouse_name}: segmentation failed in "
                            f"{qc['segmentation_failure_pct']:.1f}% of TST frames"
                        )
                    if qc.get("mask_area_cv_pct", 0.0) > 35.0:
                        quality_warnings.append(
                            f"{mouse_name}: unstable TST body-mask area "
                            f"(CV {qc['mask_area_cv_pct']:.1f}%)"
                        )

                self._raise_if_cancelled()
                self._progress(70, "Exporting Data...")
                self._export_all_data(result, base_name, settings, data_dir)

                if settings.create_timelapse:
                    self._raise_if_cancelled()
                    self._progress(80, "Creating Validation Video...")
                    media_dir.mkdir(parents=True, exist_ok=True)
                    self._generate_tst_validation_video(
                        result, video_path, settings, base_name, media_dir
                    )

                self._raise_if_cancelled()
                self._progress(90, "Generating Plots...")
                try:
                    self.maze.generate_specific_plots(
                        result, plots_dir, base_name
                    )
                except Exception as exc:
                    message = f"plot generation failed: {exc}"
                    quality_warnings.append(message)
                    logger.error(message, exc_info=True)
                self.checkpoint_mgr.clear_checkpoint(video_path)

            else:
                # --- STANDARD TRACKING LOGIC (Centroid) ---
                video_progress_func = lambda pct, task: self._progress(pct, task)

                self._progress(5, "Generating reference frame...")

                # Core: Generate Reference
                ref_frame = ReferenceFrameGenerator.generate(
                    video_path,
                    num_samples=config.video.ref_frame_samples,
                    target_dims=handler.dimensions,
                    use_cache=True,
                    should_cancel=lambda: self.cancel.is_set()
                )

                self._progress(15, "Initializing detection engine...")

                # Core: Initialize Engines
                engine = DetectionEngine(ref_frame, settings.detection_config)
                analyzer = Analyzer(
                    video_path,
                    engine,
                    settings.roi_manager,
                    resume_from_frame,
                    end_frame,
                    result_start_frame=start_frame
                )

                # Initialize Result Container
                result_class = self.maze.get_result_class()
                result_container = result_class()

                # Restore Partial Result if resuming
                if checkpoint and checkpoint.partial_result:
                    for k, v in checkpoint.partial_result.items():
                        if hasattr(result_container, k): setattr(result_container, k, v)
                    # JSON turns (x, y) tuples into lists.
                    result_container.positions = [
                        tuple(p) if p is not None else None
                        for p in result_container.positions
                    ]
                    if (
                        not result_container.raw_roi_labels
                        and result_container.roi_labels
                    ):
                        result_container.raw_roi_labels = list(
                            result_container.roi_labels
                        )
                    last_valid = next(
                        (p for p in reversed(result_container.positions) if p is not None),
                        None
                    )
                    if last_valid is not None:
                        engine.last_position = (
                            analyzer.geometry.to_analysis_point(
                                last_valid
                            )
                        )
                    engine.restore_state(
                        checkpoint.partial_result.get("_detector_state")
                    )

                self._progress(20, "Tracking animal position...")

                quality_monitor = DetectionQualityMonitor(
                    frame_diagonal=float(
                        np.hypot(handler.width, handler.height)
                    )
                )
                local_est = ProgressEstimator(end_frame - resume_from_frame)

                # Run Analysis Loop
                result, timelapse_success, frames_done = self._run_tracked_analysis(
                    analyzer, result_container, settings, video_path, handler,
                    local_est, quality_monitor
                )

                early_end = self._early_end_warning(
                    analyzer.end_frame - resume_from_frame,
                    frames_done,
                    resume_from_frame,
                    analyzer.end_frame,
                )
                if early_end:
                    quality_warnings.append(early_end)
                    self.reporter.log(f"⚠ {early_end}")
                    logger.warning("%s: %s", video_path.name, early_end)

                result.fps = handler.fps
                self.maze.resolve_tracking_labels(result)
                analyzer.finalize_result(result)

                self._raise_if_cancelled()
                self._progress(70, "Calculating Metrics...")
                if settings.scale_factor > 0:
                    apply_scale_to_result(result, settings.scale_factor)
                self.maze.calculate_metrics(result)
                if result.tracking_quality_grade != "Excellent":
                    quality_message = (
                        f"{result.tracking_quality_grade}: "
                        f"{result.detection_rate*100:.1f}% coverage, "
                        f"{result.tracking_quality.get('mean_confidence', 0.0):.2f} "
                        "mean confidence"
                    )
                    quality_warnings.append(quality_message)
                    self.reporter.log(f"⚠ Tracking quality review: {quality_message}")
                missed_entries = getattr(result, 'entries_possibly_missed', 0)
                if missed_entries > 0:
                    gap_message = (
                        f"{missed_entries} arm entr"
                        f"{'y' if missed_entries == 1 else 'ies'} may be "
                        "missing because tracking was lost and resumed in a "
                        "different arm; entry counts are conservative"
                    )
                    quality_warnings.append(gap_message)
                    self.reporter.log(f"⚠ {gap_message}")

                self._raise_if_cancelled()
                self._progress(78, "Exporting Data...")
                self._export_all_data(result, base_name, settings, data_dir)

                self._raise_if_cancelled()
                self._progress(85, "Generating Plots...")
                plot_errors = self._generate_all_visualizations(
                    result, handler, base_name, video_progress_func, plots_dir
                )
                quality_warnings.extend(plot_errors)

                if settings.create_timelapse and timelapse_success:
                    self._raise_if_cancelled()
                    self._progress(95, "Creating validation video...")
                    media_dir.mkdir(parents=True, exist_ok=True)
                    self._generate_tracking_validation_video(
                        result, video_path, settings, base_name, handler,
                        media_dir
                    )
                self.checkpoint_mgr.clear_checkpoint(video_path)

            quality_warnings.extend(self._export_warnings)
            self.reporter.log(f"✓ Successfully processed: {video_path.name}")
            if quality_warnings:
                self.reporter.status("warning", "; ".join(quality_warnings[:3]))
            else:
                self.reporter.status("done", "")
            return VideoOutcome(
                video_path, result=result, warnings=quality_warnings,
                frames_done=frames_done, time_bins=self.time_bins,
            )

        except InterruptedError:
            self.reporter.log(
                f"Analysis cancelled while processing {video_path.name}"
            )
            self.reporter.status(
                "queued", "interrupted; resumes from its checkpoint"
            )
            return VideoOutcome(
                video_path, warnings=quality_warnings, cancelled=True,
                frames_done=frames_done,
            )
        except Exception as e:
            self.reporter.log(f"✗ Error processing {video_path.name}: {str(e)}")
            logger.error("Worker error", exc_info=True)
            self.reporter.status("failed", str(e))
            return VideoOutcome(
                video_path, warnings=quality_warnings, failure=str(e),
                frames_done=frames_done,
            )

    def _run_tst_analysis(
        self,
        video_path,
        settings,
        start_frame,
        end_frame,
        handler,
        checkpoint=None,
        resume_from_frame=None
    ):
        """Core TST Analysis Loop."""
        result = self.maze.get_result_class()()
        result.fps = handler.fps
        result.timestamps = []
        result.per_roi_times = settings.per_roi_times
        result.start_frame = start_frame
        result.end_frame = end_frame

        if checkpoint and checkpoint.partial_result:
            for key, value in checkpoint.partial_result.items():
                if hasattr(result, key):
                    setattr(result, key, value)
            # Run setup always wins over what the checkpoint recorded.
            result.fps = handler.fps
            result.per_roi_times = settings.per_roi_times
            result.start_frame = start_frame
            result.end_frame = end_frame

        # Setup ROI dict
        roi_dict = {cat: rois[0] for cat, rois in settings.roi_manager.rois.items() if rois}
        for name in roi_dict:
            result.activity_scores.setdefault(name, [])
            result.motion_evidence.setdefault(name, [])

        # TST motion-decomposition engine
        engine = TSTMotionEngine(
            roi_dict,
            self.maze.motion_config,
            fps=handler.fps,
        )
        if checkpoint:
            engine.restore_state(
                checkpoint.partial_result.get("_tst_engine_state")
            )

        processing_start = (
            max(start_frame, int(resume_from_frame))
            if resume_from_frame is not None
            else start_frame
        )
        previous_frame_idx = max(start_frame, processing_start - 1)
        reader = BufferedVideoReader(
            video_path,
            start_frame=previous_frame_idx,
        )

        local_est = ProgressEstimator(max(0, end_frame - processing_start))
        next_read = previous_frame_idx + 1
        frames_processed = 0
        # Dense flow and the history-independent segmentation of the next
        # frames run on worker threads while this thread consumes the oldest
        # frame in order, so histories (and checkpoints) only ever reflect
        # frames that were fully consumed.
        pipeline = TSTFramePipeline(engine)

        try:
            # The first read can raise VideoReadError, so it belongs inside
            # the try/finally that releases the reader.
            ret, prev_frame = reader.read()
            if not ret:
                return result, 0

            reader_exhausted = False
            while True:
                while (
                    not reader_exhausted
                    and next_read < end_frame
                    and pipeline.has_capacity()
                ):
                    ret, frame = reader.read()
                    if not ret:
                        reader_exhausted = True
                        break
                    pipeline.submit(next_read, frame, prev_frame)
                    prev_frame = frame
                    next_read += 1
                if not pipeline.pending:
                    break

                curr = pipeline.next_tag
                if self.cancel.is_set():
                    checkpoint_payload = dict(result.__dict__)
                    checkpoint_payload["_tst_engine_state"] = (
                        engine.export_state()
                    )
                    self.checkpoint_mgr.save_checkpoint(
                        video_path,
                        curr - 1,
                        checkpoint_payload,
                        settings
                    )
                    raise InterruptedError("Analysis cancelled")

                _, observations = pipeline.pop()
                # Position within the result, which is what the validation
                # video indexes (also after a resume: lists are restored).
                result_index = len(result.timestamps)
                result.timestamps.append(curr / handler.fps)
                keep_contour = (
                    settings.create_timelapse
                    and result_index % TST_VALIDATION_FRAME_STEP == 0
                )
                for name, observation in observations.items():
                    result.activity_scores[name].append(
                        observation.activity_score
                    )
                    evidence = observation.to_dict()
                    # Absolute video frame, so annotations line up when the
                    # run does not start at frame 0.
                    evidence["source_frame"] = curr
                    # A compact contour gives the optional validation video a
                    # direct view of the body segmentation that produced this
                    # classification. Only the frames that video draws keep
                    # one: contours were most of the per-frame memory.
                    if keep_contour:
                        evidence["mask_contour"] = (
                            engine.get_last_mask_contour(name)
                        )
                    result.motion_evidence[name].append(evidence)

                processed_frame_idx = curr
                frames_processed += 1

                local_est.update(frames_processed)
                if self.global_est is not None:
                    self.global_est.update(self.global_offset + frames_processed)
                if frames_processed % 30 == 0:
                    l_eta = local_est.get_eta_string(frames_processed)
                    pct = (
                        (frames_processed / max(1, end_frame - processing_start))
                        * 60
                    )
                    task = f"Motion Analysis | Video ETA: {l_eta}"
                    if self.global_est is not None:
                        g_eta = self.global_est.get_eta_string(
                            self.global_offset + frames_processed
                        )
                        task += f" | Batch ETA: {g_eta}"
                    self._progress(pct, task)

                if self.checkpoint_mgr.should_save_checkpoint():
                    checkpoint_payload = dict(result.__dict__)
                    checkpoint_payload["_tst_engine_state"] = (
                        engine.export_state()
                    )
                    self.checkpoint_mgr.save_checkpoint(
                        video_path,
                        processed_frame_idx,
                        checkpoint_payload,
                        settings
                    )
        finally:
            # Cancel queued frame work first so a cancelled run never waits
            # on more than the few frame pairs already being computed.
            pipeline.close()
            reader.release()

        result.end_frame = (
            int(round(result.timestamps[-1] * handler.fps)) + 1
            if result.timestamps else start_frame
        )
        return result, frames_processed

    @staticmethod
    def _early_end_warning(
        expected_frames: int,
        frames_processed: int,
        first_frame: int,
        end_frame: int,
    ) -> Optional[str]:
        """Describe a short read, or None if within max(2, 1%) of expected."""
        shortfall = expected_frames - frames_processed
        if shortfall <= max(2, 0.01 * expected_frames):
            return None
        return (
            f"analysis ended early at frame {first_frame + frames_processed} "
            f"of {end_frame} ({shortfall} frames missing)"
        )

    def _run_tracked_analysis(self, analyzer, result_container, settings, video_path,
                             handler, local_est, quality_monitor):
        """Core Positional Tracking Loop (Optimized for RAM)."""
        frame_count = 0
        total_local_frames = analyzer.end_frame - analyzer.start_frame

        # Core: Analyzer Generator
        for frame_idx, position, overlapping_zones in analyzer.process_frames():
            result_container.positions.append(position)
            result_container.timestamps.append(frame_idx / handler.fps)
            result_container.overlapping_zones.append(overlapping_zones)
            raw_label = overlapping_zones[0] if overlapping_zones else 'outside'
            result_container.raw_roi_labels.append(raw_label)
            result_container.roi_labels.append(raw_label)
            
            detected = position is not None
            detection_result = analyzer.engine.last_result
            result_container.detection_confidences.append(
                detection_result.confidence
            )
            result_container.detection_failure_reasons.append(
                detection_result.failure_reason
            )
            # Update Quality Monitor
            quality_monitor.update(
                detected,
                detection_result.confidence,
                position
            )
            
            frame_count += 1
            
            # Progress Updates
            local_est.update(frame_count)
            if self.global_est is not None:
                self.global_est.update(self.global_offset + frame_count)
            
            if frame_count % 30 == 0:
                tracking_progress = (frame_count / total_local_frames) * 50
                overall_pct = 20 + tracking_progress
                l_eta = local_est.get_eta_string(frame_count)
                speed =local_est.get_speed()
                self._progress(
                    overall_pct,
                    f"Frame {frame_idx} | ETA: {l_eta} | Speed: {speed:.1f} fps"
                )
            
            # Cancel always saves; otherwise save on the wall-clock interval.
            cancelled = self.cancel.is_set()
            if cancelled or self.checkpoint_mgr.should_save_checkpoint():
                checkpoint_payload = dict(result_container.__dict__)
                checkpoint_payload["_detector_state"] = (
                    analyzer.engine.export_state()
                )
                self.checkpoint_mgr.save_checkpoint(
                    video_path, frame_idx, checkpoint_payload, settings
                )
            if cancelled:
                raise InterruptedError("Analysis cancelled")

        return result_container, True, frame_count

    def _generate_tracking_validation_video(
        self, result, video_path, settings, base_name, handler, output_dir
    ):
        """Generate tracking validation video with ROIs and Text directly to disk."""
        try:
            output_path = output_dir / f"{base_name}_timelapse.mp4"
            config = AppConfig()
            interval = config.visualization.timelapse_fps_divider
            if interval < 1: interval = 1
            out_fps = (handler.fps / interval) * config.visualization.timelapse_speed_multiplier
            
            # Use resolved visual labels if available, else raw roi labels
            labels = result.visual_labels if len(result.visual_labels) == len(result.timestamps) else result.roi_labels

            with SafeVideoWriter(
                output_path, out_fps, handler.dimensions
            ) as writer, _open_capture(video_path) as cap:
                first_result_frame = (
                    int(round(result.timestamps[0] * handler.fps))
                    if result.timestamps else result.start_frame
                )
                position_capture_at_frame(
                    cap, first_result_frame,
                    should_cancel=lambda: self.cancel.is_set(),
                )
                self._raise_if_cancelled()

                curr_idx = 0
                total_frames = len(result.timestamps)

                while curr_idx < total_frames:
                    self._raise_if_cancelled()
                    if curr_idx % interval != 0:
                        # Skipped frames are only decoded, never converted.
                        if not cap.grab():
                            break
                        curr_idx += 1
                        continue
                    ret, frame = cap.read()
                    if not ret: break

                    if curr_idx % interval == 0:
                        # 1. Draw ROIs
                        frame = settings.roi_manager.draw_on_frame(frame, thickness=2)
                        
                        # 2. Draw Position Dot
                        pos = result.positions[curr_idx]
                        confidence = (
                            result.detection_confidences[curr_idx]
                            if curr_idx < len(result.detection_confidences)
                            else 0.0
                        )
                        if pos:
                            if confidence >= 0.60:
                                dot_color = (0, 200, 0)
                            elif confidence >= 0.40:
                                dot_color = (0, 165, 255)
                            else:
                                dot_color = (0, 0, 255)
                            center = (
                                int(round(pos[0])), int(round(pos[1]))
                            )
                            cv2.circle(frame, center, 8, dot_color, -1)
                        
                        # 3. Draw Zone Text Overlay
                        current_zone = labels[curr_idx] if curr_idx < len(labels) else "outside"
                        if pos:
                            evidence_text = f"Confidence: {confidence:.2f}"
                        else:
                            reason = (
                                result.detection_failure_reasons[curr_idx]
                                if curr_idx < len(result.detection_failure_reasons)
                                else "not_detected"
                            )
                            evidence_text = f"Missing: {reason}"
                        text = (
                            f"Zone: {current_zone.replace('_', ' ').title()} | "
                            f"{evidence_text}"
                        )
                        
                        # Text background for visibility
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.rectangle(frame, (10, 10), (760, 60), (0, 0, 0), -1)
                        cv2.putText(frame, text, (20, 45), font, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
                        
                        writer.write(frame)
                    curr_idx += 1
            self.reporter.log("✓ Timelapse saved")
        except InterruptedError:
            self._discard_partial_video(output_path)
            raise
        except Exception as e:
            logger.error(f"Error creating timelapse video: {e}", exc_info=True)
            raise

    @staticmethod
    def _discard_partial_video(output_path):
        """Remove the half-written video left behind by a cancelled export."""
        try:
            Path(output_path).unlink(missing_ok=True)
        except OSError:
            logger.debug("Could not remove partial video", exc_info=True)

    def _export_all_data(self, result, base_name, settings, output_dir):
        """Export CSVs and JSONs."""
        frame_data_name = (
            f"{base_name}_tst_motion_evidence.csv"
            if isinstance(self.maze, TST) else
            f"{base_name}_tracking_data.csv"
        )
        self._raise_if_cancelled()
        DataExporter.export_to_csv(
            result, output_dir / frame_data_name
        )
        self._raise_if_cancelled()
        DataExporter.export_summary_csv(
            result,
            output_dir / f"{base_name}_analysis_summary.csv",
            self.maze
        )
        
        self._raise_if_cancelled()
        # JSON Export - Handle differences in settings objects
        detection_params = None
        if hasattr(settings, 'detection_config'):
             detection_params = settings.detection_config.to_dict()
        
        DataExporter.export_summary_json(
            result,
            output_dir / f"{base_name}_summary.json",
            detection_params,
            settings.to_dict(),
            animal_metadata=self.job.animal_metadata,
        )
        self._raise_if_cancelled()
        self.maze.export_specific_data(result, output_dir, base_name)
        self._export_time_bins(result, base_name, output_dir)
        self.reporter.log("✓ Data exported")

    def _export_time_bins(self, result, base_name, output_dir):
        """Per-bin metrics (only when the run asked for time bins).

        A failure here costs the bin files, not the video: it is reported as
        a quality warning like a failed plot.
        """
        if not self.job.bin_seconds or self.job.bin_seconds <= 0:
            return
        self._raise_if_cancelled()
        try:
            table = compute_time_bins(result, self.job.bin_seconds)
            if table is None:
                return
            DataExporter.export_time_bins_csv(
                table, output_dir / f"{base_name}_time_bins.csv"
            )
            self.time_bins = table
        except Exception as exc:
            message = f"time bins failed: {exc}"
            self._export_warnings.append(message)
            logger.error(message, exc_info=True)

    def _generate_all_visualizations(
        self, result, handler, base_name, cb, output_dir
    ):
        """Generate generic plots."""
        errors = []
        jobs = [
            (
                "heatmap",
                lambda: Visualizer.generate_heatmap(
                    result.positions,
                    handler.dimensions,
                    output_dir / f"{base_name}_heatmap.png",
                ),
            ),
            (
                "trajectory",
                lambda: Visualizer.generate_trajectory_plot(
                    result,
                    handler.dimensions,
                    output_dir / f"{base_name}_trajectory.png",
                ),
            ),
            (
                "time series",
                lambda: Visualizer.generate_time_series(
                    result,
                    output_dir / f"{base_name}_timeseries.png",
                ),
            ),
            (
                "paradigm-specific plots",
                lambda: self.maze.generate_specific_plots(
                    result, output_dir, base_name
                ),
            ),
        ]
        for label, job in jobs:
            self._raise_if_cancelled()
            try:
                job()
            except Exception as exc:
                message = f"{label} generation failed: {exc}"
                errors.append(message)
                logger.error(message, exc_info=True)
        return errors

    def _generate_tst_validation_video(
        self, result, video_path, settings, base_name, output_dir
    ):
        """Generate color-coded TST validation video."""
        try:
            output_path = output_dir / f"{base_name}_validation.mp4"
            vh = VideoHandler(video_path)
            
            # Subsample for validation video speed
            # Shared with _run_tst_analysis, which stores contours only for
            # the frames drawn here.
            step = TST_VALIDATION_FRAME_STEP
            out_fps = vh.fps / step
            
            # Core: Use SafeVideoWriter
            with SafeVideoWriter(
                output_path, out_fps, vh.dimensions
            ) as writer, _open_capture(video_path) as cap:
                first_result_frame = (
                    int(round(result.timestamps[0] * vh.fps))
                    if result.timestamps else result.start_frame
                )
                position_capture_at_frame(
                    cap, first_result_frame,
                    should_cancel=lambda: self.cancel.is_set(),
                )
                self._raise_if_cancelled()

                roi_dict = {cat: rois[0] for cat, rois in settings.roi_manager.rois.items() if rois}
                curr_idx = 0
                total_frames = len(result.timestamps)

                while curr_idx < total_frames:
                    self._raise_if_cancelled()
                    if curr_idx % step != 0:
                        # Skipped frames are only decoded, never converted.
                        if not cap.grab():
                            break
                        curr_idx += 1
                        continue
                    ret, frame = cap.read()
                    if not ret: break

                    if curr_idx % step == 0:
                        time_sec = result.timestamps[curr_idx]
                        
                        for name, roi in roi_dict.items():
                            valid_start, valid_end = 0, float('inf')
                            if result.per_roi_times and name in result.per_roi_times:
                                valid_start, valid_end = result.per_roi_times[name]
                            
                            pts = roi.points.astype(np.int32)
                            
                            if time_sec < valid_start or time_sec >= valid_end:
                                # Grey out excluded time
                                blend_polygon(
                                    frame, pts, (50, 50, 50), 0.5, 0.5
                                )
                                cv2.putText(frame, "EXCLUDED", (pts[0][0], pts[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)
                            else:
                                # Show active status (Mobile/Immobile)
                                if name in result.immobile_states and curr_idx < len(result.immobile_states[name]):
                                    is_immobile = result.immobile_states[name][curr_idx]
                                    behavior_trace = result.behavior_states.get(
                                        name, []
                                    )
                                    behavior_state = (
                                        behavior_trace[curr_idx]
                                        if curr_idx < len(behavior_trace)
                                        else "uncertain"
                                    )
                                    color, status = tst_display_style(
                                        behavior_state, is_immobile
                                    )
                                    evidence_trace = result.motion_evidence.get(
                                        name, []
                                    )
                                    evidence = (
                                        evidence_trace[curr_idx]
                                        if curr_idx < len(evidence_trace)
                                        else {}
                                    )
                                    contour = evidence.get("mask_contour", [])
                                    if len(contour) >= 3:
                                        contour_points = np.asarray(
                                            contour, dtype=np.int32
                                        ).reshape(-1, 1, 2)
                                        blend_polygon(
                                            frame, contour_points, color,
                                            0.35, 0.65,
                                        )
                                        cv2.polylines(
                                            frame, [contour_points], True,
                                            (255, 255, 255), 1,
                                        )
                                    cv2.polylines(frame, [pts], True, color, 2)
                                    confidence = float(
                                        evidence.get("confidence", 0.0)
                                    )
                                    qc_flag = str(evidence.get("qc_flag", ""))
                                    label = f"{status} ({confidence:.2f})"
                                    if qc_flag:
                                        label += f" | {qc_flag}"
                                    cv2.putText(frame, label, (pts[0][0], pts[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                        
                        writer.write(frame)
                    curr_idx += 1

            self.reporter.log("✓ Validation video saved")
        except InterruptedError:
            self._discard_partial_video(output_path)
            raise
        except Exception as e:
            logger.error(f"Error creating validation video: {e}")
            raise
