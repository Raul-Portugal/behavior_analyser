"""
gui/worker.py
Enhanced analysis worker.
Uses the unified Core modules for TST and Tracking analysis.
"""
import logging
import json
import copy
import cv2
import numpy as np
import hashlib
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import QObject, pyqtSignal

# --- CORE IMPORTS ---
from core.batch_plan_keys import build_plan_by_key
from core.models import AppConfig, BatchSettings
from core.video import VideoHandler, ReferenceFrameGenerator, SafeVideoWriter, BufferedVideoReader
from core.detection import DetectionEngine, DetectionQualityMonitor
from core.analysis_engine import Analyzer
from core.tst_display import tst_display_style
from core.tst_motion_engine import TSTMotionEngine
from core.version import APP_NAME, APP_VERSION

# --- LEGACY/UTILITY IMPORTS ---
from core.analysis_engine import apply_scale_to_result
from mazes.base_maze import BaseAnalysisResult, Maze
from mazes.tst import TST, describe_window_truncation
from output import DataExporter, Visualizer
from checkpoint_manager import CheckpointManager, ProgressEstimator

logger = logging.getLogger(__name__)

# Written into every run folder so the run can be reloaded or resumed later.
RUN_PLAN_FILENAME = "run_plan.json"


@contextmanager
def _open_capture(video_path):
    """Open a cv2.VideoCapture that is released even when a phase aborts."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        yield cap
    finally:
        cap.release()


@dataclass
class BatchRunOutcome:
    results: List[Tuple[Path, BaseAnalysisResult]] = field(default_factory=list)
    failures: List[Tuple[Path, str]] = field(default_factory=list)
    quality_warnings: List[Tuple[Path, str]] = field(default_factory=list)
    cancelled: bool = False
    output_dir: Optional[Path] = None


class AnalysisWorker(QObject):
    # Signals
    progress = pyqtSignal(int, str)
    detailed_progress = pyqtSignal(int, str, str)
    log = pyqtSignal(str)
    finished = pyqtSignal(object)
    # (video path, "running" | "done" | "warning" | "failed" | "queued", message)
    video_status = pyqtSignal(str, str, str)

    def __init__(self, analysis_plan: Dict[Path, BatchSettings], output_dir: Path, maze: Maze):
        super().__init__()
        self.analysis_plan = analysis_plan
        self.output_root = output_dir
        self.maze = maze
        effective_parameters = self.maze.get_algorithm_parameters()
        for settings in self.analysis_plan.values():
            settings.paradigm_id = self.maze.name
            settings.algorithm_parameters = copy.deepcopy(
                effective_parameters
            )
        self.is_cancelled = False
        self.checkpoint_mgr = CheckpointManager(output_dir)
        run_suffix = time.strftime("%Y%m%d_%H%M%S")
        candidate = output_dir / f"analysis_run_{run_suffix}"
        sequence = 2
        while candidate.exists():
            candidate = output_dir / f"analysis_run_{run_suffix}_{sequence}"
            sequence += 1
        candidate.mkdir(parents=True, exist_ok=False)
        self.output_dir = candidate
        self.run_started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        stem_counts = Counter(path.stem for path in analysis_plan)
        self.output_base_names = {}
        for path in analysis_plan:
            base_name = path.stem
            if stem_counts[path.stem] > 1:
                path_hash = hashlib.sha256(
                    str(path.resolve()).encode('utf-8')
                ).hexdigest()[:8]
                base_name = f"{base_name}_{path_hash}"
            self.output_base_names[path] = base_name

    def stop(self):
        """Request the worker to stop processing."""
        self.log.emit("Cancellation requested by user - saving progress...")
        self.is_cancelled = True

    def _raise_if_cancelled(self):
        """Cooperative cancellation point for phases outside the frame loops."""
        if self.is_cancelled:
            raise InterruptedError("Analysis cancelled")

    def emit_progress(self, overall_pct: int, main_task: str, sub_task: str = ""):
        """Helper to emit detailed progress signals."""
        self.detailed_progress.emit(overall_pct, main_task, sub_task)
        self.progress.emit(overall_pct, main_task)

    def run(self):
        """Main execution loop."""
        config = AppConfig()
        all_results: List[Tuple[Path, BaseAnalysisResult]] = []
        failures: List[Tuple[Path, str]] = []
        quality_warnings: List[Tuple[Path, str]] = []
        run_cancelled = False
        videos_to_process = list(self.analysis_plan.items())
        total_videos = len(videos_to_process)

        self._write_run_plan()

        # --- PRE-CALCULATION FOR ETA ---
        self.log.emit("Preparing batch analysis...")
        total_batch_frames = 0
        
        # Calculate total frames for global progress bar
        for v_path, v_settings in videos_to_process:
            try:
                vh = VideoHandler(v_path)
                s = int(v_settings.start_time * vh.fps)
                e = int(v_settings.end_time * vh.fps) if v_settings.end_time else vh.total_frames
                total_batch_frames += max(0, e - s)
            except Exception: 
                total_batch_frames += 1000 # Fallback estimate
        
        global_est = ProgressEstimator(total_batch_frames)
        global_offset = 0

        for video_num, (video_path, settings) in enumerate(videos_to_process, 1):
            if self.is_cancelled:
                self.log.emit("Analysis cancelled - checkpoints saved for resume")
                run_cancelled = True
                break

            self.checkpoint_mgr.reset_timer()
            base_name = self.output_base_names[video_path]
            video_output_dir = self.output_dir / base_name
            data_dir = video_output_dir / "data"
            plots_dir = video_output_dir / "plots"
            media_dir = video_output_dir / "media"
            self.log.emit(f"\n{'='*60}\nAnalyzing {self.maze.name} video {video_num}/{total_videos}: {video_path.name}\n{'='*60}")
            self.video_status.emit(str(video_path), "running", "")
            warnings_before = len(quality_warnings)
            
            to_batch_pct = lambda local_pct: int(
                ((video_num - 1) + (local_pct / 100.0))
                / total_videos
                * 100
            )
            frames_done = 0

            try:
                # Inside the try so an unwritable output folder is recorded
                # as a per-video failure instead of escaping run().
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
                    self.log.emit(
                        f"✓ Resuming from checkpoint (frame {resume_from_frame})"
                    )

                # === BRANCH: TST vs STANDARD TRACKING ===
                if isinstance(self.maze, TST):
                    # --- TST MOTION-DECOMPOSITION LOGIC ---
                    self.emit_progress(
                        to_batch_pct(10),
                        f"Video: {base_name}",
                        "Decomposing rigid and articulated motion..."
                    )
                    
                    result, frames_done = self._run_tst_analysis(
                        video_path, settings, start_frame, end_frame, handler,
                        global_est, global_offset, to_batch_pct, base_name,
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
                        quality_warnings.append((video_path, early_end))
                        self.log.emit(f"⚠ {early_end}")
                        logger.warning("%s: %s", video_path.name, early_end)
                    
                    self._raise_if_cancelled()
                    self.emit_progress(to_batch_pct(60), f"Video: {base_name}", "Applying Temporal Filters...")
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
                            quality_warnings.append(
                                (video_path, quality_message)
                            )
                        mean_confidence = result.mean_state_confidence.get(
                            mouse_name, 0.0
                        )
                        if mean_confidence < 0.60:
                            quality_warnings.append((
                                video_path,
                                f"{mouse_name}: low mean TST state confidence "
                                f"({mean_confidence:.2f})"
                            ))
                        qc = result.quality_metrics.get(mouse_name, {})
                        truncation = describe_window_truncation(
                            result, mouse_name
                        )
                        if truncation:
                            quality_warnings.append((video_path, truncation))
                        if qc.get("segmentation_failure_pct", 0.0) > 5.0:
                            quality_warnings.append((
                                video_path,
                                f"{mouse_name}: segmentation failed in "
                                f"{qc['segmentation_failure_pct']:.1f}% of TST frames",
                            ))
                        if qc.get("mask_area_cv_pct", 0.0) > 35.0:
                            quality_warnings.append((
                                video_path,
                                f"{mouse_name}: unstable TST body-mask area "
                                f"(CV {qc['mask_area_cv_pct']:.1f}%)",
                            ))
                    
                    self._raise_if_cancelled()
                    self.emit_progress(to_batch_pct(70), f"Video: {base_name}", "Exporting Data...")
                    self._export_all_data(result, base_name, settings, data_dir)
                    
                    if settings.create_timelapse:
                        self._raise_if_cancelled()
                        self.emit_progress(to_batch_pct(80), f"Video: {base_name}", "Creating Validation Video...")
                        media_dir.mkdir(parents=True, exist_ok=True)
                        self._generate_tst_validation_video(
                            result, video_path, settings, base_name, media_dir
                        )
                    
                    self._raise_if_cancelled()
                    self.emit_progress(to_batch_pct(90), f"Video: {base_name}", "Generating Plots...")
                    try:
                        self.maze.generate_specific_plots(
                            result, plots_dir, base_name
                        )
                    except Exception as exc:
                        message = f"plot generation failed: {exc}"
                        quality_warnings.append((video_path, message))
                        logger.error(message, exc_info=True)
                    self.checkpoint_mgr.clear_checkpoint(video_path)

                else:
                    # --- STANDARD TRACKING LOGIC (Centroid) ---
                    video_progress_func = lambda pct, task: self.emit_progress(to_batch_pct(pct), f"Video: {base_name}", task)
                    
                    self.emit_progress(to_batch_pct(5), f"Video: {base_name}", "Generating reference frame...")
                    
                    # Core: Generate Reference
                    ref_frame = ReferenceFrameGenerator.generate(
                        video_path, 
                        num_samples=config.video.ref_frame_samples,
                        target_dims=handler.dimensions, 
                        use_cache=True,
                        should_cancel=lambda: self.is_cancelled
                    )
                    
                    self.emit_progress(to_batch_pct(15), f"Video: {base_name}", "Initializing detection engine...")
                    
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

                    self.emit_progress(to_batch_pct(20), f"Video: {base_name}", "Tracking animal position...")
                    
                    quality_monitor = DetectionQualityMonitor(
                        frame_diagonal=float(
                            np.hypot(handler.width, handler.height)
                        )
                    )
                    local_est = ProgressEstimator(end_frame - resume_from_frame)
                    
                    # Run Analysis Loop
                    result, timelapse_success, frames_done = self._run_tracked_analysis(
                        analyzer, result_container, settings, video_path, handler,
                        local_est, global_est, quality_monitor,
                        to_batch_pct, base_name, global_offset
                    )

                    early_end = self._early_end_warning(
                        analyzer.end_frame - resume_from_frame,
                        frames_done,
                        resume_from_frame,
                        analyzer.end_frame,
                    )
                    if early_end:
                        quality_warnings.append((video_path, early_end))
                        self.log.emit(f"⚠ {early_end}")
                        logger.warning("%s: %s", video_path.name, early_end)

                    result.fps = handler.fps
                    self.maze.resolve_tracking_labels(result)
                    analyzer.finalize_result(result)
                    
                    self._raise_if_cancelled()
                    self.emit_progress(to_batch_pct(70), f"Video: {base_name}", "Calculating Metrics...")
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
                        quality_warnings.append((video_path, quality_message))
                        self.log.emit(f"⚠ Tracking quality review: {quality_message}")
                    missed_entries = getattr(result, 'entries_possibly_missed', 0)
                    if missed_entries > 0:
                        gap_message = (
                            f"{missed_entries} arm entr"
                            f"{'y' if missed_entries == 1 else 'ies'} may be "
                            "missing because tracking was lost and resumed in a "
                            "different arm; entry counts are conservative"
                        )
                        quality_warnings.append((video_path, gap_message))
                        self.log.emit(f"⚠ {gap_message}")

                    self._raise_if_cancelled()
                    self.emit_progress(to_batch_pct(78), f"Video: {base_name}", "Exporting Data...")
                    self._export_all_data(result, base_name, settings, data_dir)
                    
                    self._raise_if_cancelled()
                    self.emit_progress(to_batch_pct(85), f"Video: {base_name}", "Generating Plots...")
                    plot_errors = self._generate_all_visualizations(
                        result, handler, base_name, video_progress_func, plots_dir
                    )
                    quality_warnings.extend(
                        (video_path, message) for message in plot_errors
                    )
                    
                    if settings.create_timelapse and timelapse_success:
                        self._raise_if_cancelled()
                        self.emit_progress(to_batch_pct(95), f"Video: {base_name}", "Creating Timelapse...")
                        media_dir.mkdir(parents=True, exist_ok=True)
                        self._generate_tracking_validation_video(
                            result, video_path, settings, base_name, handler,
                            media_dir
                        )
                    self.checkpoint_mgr.clear_checkpoint(video_path)

                self.log.emit(f"✓ Successfully processed: {video_path.name}")
                all_results.append((video_path, result))
                new_warnings = quality_warnings[warnings_before:]
                if new_warnings:
                    self.video_status.emit(
                        str(video_path), "warning",
                        "; ".join(message for _, message in new_warnings[:3]),
                    )
                else:
                    self.video_status.emit(str(video_path), "done", "")

            except InterruptedError:
                run_cancelled = True
                self.log.emit(f"Analysis cancelled while processing {video_path.name}")
                self.video_status.emit(
                    str(video_path), "queued",
                    "interrupted; resumes from its checkpoint",
                )
                break
            except Exception as e:
                self.log.emit(f"✗ Error processing {video_path.name}: {str(e)}")
                logger.error("Worker error", exc_info=True)
                failures.append((video_path, str(e)))
                self.video_status.emit(str(video_path), "failed", str(e))
            
            global_offset += frames_done

        if all_results:
            try:
                DataExporter.export_batch_summary_csv(
                    all_results, self.output_dir, self.maze
                )
            except Exception as exc:
                failures.append((Path("batch_summary.csv"), str(exc)))
                logger.error("Batch summary export failed", exc_info=True)

        try:
            self._write_run_manifest(
                all_results, failures, quality_warnings, run_cancelled
            )
        except Exception as exc:
            failures.append((Path("manifest.json"), str(exc)))
            logger.error("Run manifest export failed", exc_info=True)
        finally:
            self.log.emit(
                f"\n{'='*60}\nBatch Analysis Complete\n{'='*60}\n"
                f"Processed: {len(all_results)}/{total_videos}"
            )
            self.finished.emit(
                BatchRunOutcome(
                    results=all_results,
                    failures=failures,
                    quality_warnings=quality_warnings,
                    cancelled=run_cancelled,
                    output_dir=self.output_dir
                )
            )

    def _run_tst_analysis(
        self,
        video_path,
        settings,
        start_frame,
        end_frame,
        handler,
        global_est,
        global_offset,
        to_batch_pct,
        base_name,
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
            buffer_size=64
        )

        local_est = ProgressEstimator(max(0, end_frame - processing_start))
        curr = previous_frame_idx + 1
        frames_processed = 0

        try:
            # The first read can raise VideoReadError, so it belongs inside
            # the try/finally that releases the reader.
            ret, prev_frame = reader.read()
            if not ret:
                return result, 0

            while curr < end_frame:
                if self.is_cancelled:
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

                ret, frame = reader.read()
                if not ret:
                    break

                observations = engine.calculate_observations(frame, prev_frame)
                result.timestamps.append(curr / handler.fps)
                for name, observation in observations.items():
                    result.activity_scores[name].append(
                        observation.activity_score
                    )
                    evidence = observation.to_dict()
                    # A compact contour gives the optional validation video a
                    # direct view of the body segmentation that produced this
                    # classification, without exporting full frame masks.
                    evidence["mask_contour"] = engine.get_last_mask_contour(
                        name
                    )
                    result.motion_evidence[name].append(evidence)

                processed_frame_idx = curr
                prev_frame = frame
                curr += 1
                frames_processed += 1

                local_est.update(frames_processed)
                global_est.update(global_offset + frames_processed)
                if frames_processed % 30 == 0:
                    l_eta = local_est.get_eta_string(frames_processed)
                    g_eta = global_est.get_eta_string(global_offset + frames_processed)
                    pct = to_batch_pct(
                        (frames_processed / max(1, end_frame - processing_start))
                        * 60
                    )
                    self.emit_progress(
                        pct,
                        f"Video: {base_name}",
                        f"Motion Analysis | Video ETA: {l_eta} | Batch ETA: {g_eta}"
                    )

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
                             handler, local_est, global_est, quality_monitor,
                             to_batch_pct, base_name, global_offset):
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
            global_est.update(global_offset + frame_count)
            
            if frame_count % 30 == 0:
                tracking_progress = (frame_count / total_local_frames) * 50
                overall_pct = to_batch_pct(20 + tracking_progress)
                l_eta = local_est.get_eta_string(frame_count)
                speed =local_est.get_speed()
                self.emit_progress(overall_pct, f"Video: {base_name}", 
                                 f"Frame {frame_idx} | ETA: {l_eta} | Speed: {speed:.1f} fps")
            
            # Cancel always saves; otherwise save on the wall-clock interval.
            cancelled = self.is_cancelled
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
                cap.set(cv2.CAP_PROP_POS_FRAMES, first_result_frame)
                
                curr_idx = 0
                total_frames = len(result.timestamps)
                
                while curr_idx < total_frames:
                    self._raise_if_cancelled()
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
            self.log.emit("✓ Timelapse saved")
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
            settings.to_dict()
        )
        self._raise_if_cancelled()
        self.maze.export_specific_data(result, output_dir, base_name)
        self.log.emit("✓ Data exported")

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
            step = 3
            out_fps = vh.fps / step
            
            # Core: Use SafeVideoWriter
            with SafeVideoWriter(
                output_path, out_fps, vh.dimensions
            ) as writer, _open_capture(video_path) as cap:
                first_result_frame = (
                    int(round(result.timestamps[0] * vh.fps))
                    if result.timestamps else result.start_frame
                )
                cap.set(cv2.CAP_PROP_POS_FRAMES, first_result_frame)
                
                roi_dict = {cat: rois[0] for cat, rois in settings.roi_manager.rois.items() if rois}
                curr_idx = 0
                total_frames = len(result.timestamps)
                
                while curr_idx < total_frames:
                    self._raise_if_cancelled()
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
                                overlay = frame.copy()
                                cv2.fillPoly(overlay, [pts], (50, 50, 50))
                                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
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
                                        mask_overlay = frame.copy()
                                        cv2.fillPoly(
                                            mask_overlay,
                                            [contour_points],
                                            color,
                                        )
                                        cv2.addWeighted(
                                            mask_overlay, 0.35, frame, 0.65,
                                            0, frame,
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

            self.log.emit("✓ Validation video saved")
        except InterruptedError:
            self._discard_partial_video(output_path)
            raise
        except Exception as e:
            logger.error(f"Error creating validation video: {e}")
            raise

    def _write_run_plan(self):
        """Save the exact per-video settings of this run as a batch plan.

        Keys are relative to the run folder, so the plan can be reloaded with
        File > Load Configuration or resumed after a restart. The paradigm
        and algorithm parameters are part of every entry. A failure here is
        logged but never stops the analysis.
        """
        plan_file = self.output_dir / RUN_PLAN_FILENAME
        try:
            BatchSettings.save_batch_plan(
                build_plan_by_key(self.analysis_plan, plan_file), plan_file
            )
        except Exception:
            logger.warning("Could not write %s", plan_file, exc_info=True)

    def _write_run_manifest(
        self, results, failures, quality_warnings, cancelled
    ):
        """Write a compact index so a run can be understood at a glance."""
        generated_files = ["manifest.json"] + sorted(
            str(path.relative_to(self.output_dir)).replace("\\", "/")
            for path in self.output_dir.rglob("*")
            if path.is_file()
        )
        manifest = {
            "schema_version": 2,
            "application": {
                "name": APP_NAME,
                "version": APP_VERSION,
            },
            "paradigm": self.maze.name,
            "started_at": self.run_started_at,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "cancelled": cancelled,
            "requested_videos": len(self.analysis_plan),
            "completed_videos": len(results),
            "videos": [
                {
                    "source": str(video_path),
                    "output_folder": self.output_base_names[video_path],
                }
                for video_path, _ in results
            ],
            "failures": [
                {"source": str(path), "message": message}
                for path, message in failures
            ],
            "quality_warnings": [
                {"source": str(path), "message": message}
                for path, message in quality_warnings
            ],
            "algorithm_parameters": self.maze.get_algorithm_parameters(),
            "generated_files": generated_files,
        }
        manifest_path = self.output_dir / "manifest.json"
        temp_path = manifest_path.with_suffix(".json.tmp")
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)
            file.flush()
        temp_path.replace(manifest_path)
