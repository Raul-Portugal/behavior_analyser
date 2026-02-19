"""
gui/worker.py
Enhanced analysis worker.
Uses the unified Core modules for TST and Tracking analysis.
"""
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from PyQt6.QtCore import QObject, pyqtSignal, QThread

# --- CORE IMPORTS ---
from core.models import AppConfig, BatchSettings
from core.video import VideoHandler, ReferenceFrameGenerator, SafeVideoWriter, BufferedVideoReader
from core.detection import DetectionEngine, DetectionQualityMonitor
from core.analysis_engine import Analyzer, MotionEngine

# --- LEGACY/UTILITY IMPORTS ---
from core.analysis_engine import apply_scale_to_result
from mazes.base_maze import BaseAnalysisResult, Maze
from mazes.tst import TST
from output import DataExporter, Visualizer
from checkpoint_manager import CheckpointManager, ProgressEstimator

logger = logging.getLogger(__name__)


class AnalysisWorker(QObject):
    # Signals
    progress = pyqtSignal(int, str)
    detailed_progress = pyqtSignal(int, str, str)
    log = pyqtSignal(str)
    finished = pyqtSignal(list)

    def __init__(self, analysis_plan: Dict[Path, BatchSettings], output_dir: Path, maze: Maze):
        super().__init__()
        self.analysis_plan = analysis_plan
        self.output_dir = output_dir
        self.maze = maze
        self.is_cancelled = False
        self.checkpoint_mgr = CheckpointManager(output_dir)

    def stop(self):
        """Request the worker to stop processing."""
        self.log.emit("Cancellation requested by user - saving progress...")
        self.is_cancelled = True

    def emit_progress(self, overall_pct: int, main_task: str, sub_task: str = ""):
        """Helper to emit detailed progress signals."""
        self.detailed_progress.emit(overall_pct, main_task, sub_task)
        self.progress.emit(overall_pct, main_task)

    def run(self):
        """Main execution loop."""
        config = AppConfig()
        all_results: List[Tuple[Path, BaseAnalysisResult]] = []
        videos_to_process = list(self.analysis_plan.items())
        total_videos = len(videos_to_process)

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
                break

            base_name = video_path.stem
            self.log.emit(f"\n{'='*60}\nAnalyzing {self.maze.name} video {video_num}/{total_videos}: {video_path.name}\n{'='*60}")
            
            video_start_pct = int(((video_num - 1) / total_videos) * 100)
            frames_done = 0

            try:
                # Initialize VideoHandler
                handler = VideoHandler(video_path)
                start_frame = int(settings.start_time * handler.fps)
                end_frame = int(settings.end_time * handler.fps) if settings.end_time else handler.total_frames
                
                # Checkpoint loading
                checkpoint = self.checkpoint_mgr.load_checkpoint(video_path, settings)
                resume_from_frame = checkpoint.last_frame_processed if checkpoint else None
                if resume_from_frame:
                    self.log.emit(f"✓ Resuming from checkpoint (frame {resume_from_frame})")

                # === BRANCH: TST vs STANDARD TRACKING ===
                if isinstance(self.maze, TST):
                    # --- TST LOGIC (Motion Energy) ---
                    self.emit_progress(video_start_pct + 10, f"Video: {base_name}", "Calculating Motion Energy...")
                    
                    result, frames_done = self._run_tst_analysis(
                        video_path, settings, start_frame, end_frame, handler,
                        global_est, global_offset, video_start_pct, base_name
                    )
                    
                    self.checkpoint_mgr.clear_checkpoint(video_path)
                    
                    self.emit_progress(video_start_pct + 60, f"Video: {base_name}", "Applying Temporal Filters...")
                    self.maze.calculate_metrics(result)
                    
                    self.emit_progress(video_start_pct + 70, f"Video: {base_name}", "Exporting Data...")
                    self._export_all_data(result, base_name, settings)
                    
                    if settings.create_timelapse:
                        self.emit_progress(video_start_pct + 80, f"Video: {base_name}", "Creating Validation Video...")
                        self._generate_tst_validation_video(result, video_path, settings, base_name)
                    
                    self.emit_progress(video_start_pct + 90, f"Video: {base_name}", "Generating Plots...")
                    self.maze.generate_specific_plots(result, self.output_dir, base_name)

                else:
                    # --- STANDARD TRACKING LOGIC (Centroid) ---
                    video_progress_func = lambda pct, task: self.emit_progress(video_start_pct + pct, f"Video: {base_name}", task)
                    
                    self.emit_progress(video_start_pct + 5, f"Video: {base_name}", "Generating reference frame...")
                    
                    # Core: Generate Reference
                    ref_frame = ReferenceFrameGenerator.generate(
                        video_path, 
                        num_samples=settings.detection_config.window_size,
                        target_dims=handler.dimensions, 
                        use_cache=True
                    )
                    
                    self.emit_progress(video_start_pct + 15, f"Video: {base_name}", "Initializing detection engine...")
                    
                    # Core: Initialize Engines
                    engine = DetectionEngine(ref_frame, settings.detection_config)
                    analyzer = Analyzer(video_path, engine, settings.roi_manager, start_frame, end_frame)
                    
                    # Initialize Result Container
                    result_class = self.maze.get_result_class()
                    result_container = result_class()
                    
                    # Restore Partial Result if resuming
                    if checkpoint and checkpoint.partial_result:
                        for k, v in checkpoint.partial_result.items():
                            if hasattr(result_container, k): setattr(result_container, k, v)

                    self.emit_progress(video_start_pct + 20, f"Video: {base_name}", "Tracking animal position...")
                    
                    quality_monitor = DetectionQualityMonitor()
                    local_est = ProgressEstimator(end_frame - start_frame)
                    
                    # Run Analysis Loop
                    result, timelapse_success, frames_done = self._run_tracked_analysis(
                        analyzer, result_container, settings, video_path, handler,
                        local_est, global_est, quality_monitor,
                        video_start_pct, base_name, global_offset
                    )
                    
                    self.checkpoint_mgr.clear_checkpoint(video_path)
                    
                    self.emit_progress(video_start_pct + 70, f"Video: {base_name}", "Calculating Metrics...")
                    self.maze.calculate_metrics(result)
                    
                    if settings.scale_factor > 0:
                        apply_scale_to_result(result, settings.scale_factor)

                    self.emit_progress(video_start_pct + 78, f"Video: {base_name}", "Exporting Data...")
                    self._export_all_data(result, base_name, settings)
                    
                    self.emit_progress(video_start_pct + 85, f"Video: {base_name}", "Generating Plots...")
                    self._generate_all_visualizations(result, handler, base_name, video_progress_func)
                    
                    if settings.create_timelapse and timelapse_success:
                        self.emit_progress(video_start_pct + 95, f"Video: {base_name}", "Creating Timelapse...")
                        self._generate_tracking_validation_video(result, video_path, settings, base_name, handler)

                self.log.emit(f"✓ Successfully processed: {video_path.name}")
                all_results.append((video_path, result))

            except Exception as e:
                self.log.emit(f"✗ Error processing {video_path.name}: {str(e)}")
                logger.error("Worker error", exc_info=True)
            
            global_offset += frames_done

        self.log.emit(f"\n{'='*60}\nBatch Analysis Complete\n{'='*60}\nProcessed: {len(all_results)}/{total_videos}")
        self.finished.emit(all_results)

    def _run_tst_analysis(self, video_path, settings, start_frame, end_frame, handler,
                          global_est, global_offset, video_start_pct, base_name):
        """Core TST Analysis Loop."""
        result = self.maze.get_result_class()()
        result.fps = handler.fps
        result.timestamps = []
        result.per_roi_times = settings.per_roi_times
        result.start_frame = start_frame
        result.end_frame = end_frame
        
        # Setup ROI dict
        roi_dict = {cat: rois[0] for cat, rois in settings.roi_manager.rois.items() if rois}
        for name in roi_dict: result.motion_energy[name] = []
            
        # Core: Motion Engine
        engine = MotionEngine(roi_dict)
        
        # Core: Buffered Reader
        reader = BufferedVideoReader(video_path, start_frame=start_frame, buffer_size=64)
        
        local_est = ProgressEstimator(end_frame - start_frame)
        ret, prev_frame = reader.read()
        if not ret:
            reader.release()
            return result, 0
            
        curr = start_frame + 1
        frames_processed = 0
        
        while curr < end_frame:
            if self.is_cancelled: break
            
            ret, frame = reader.read()
            if not ret: break
            
            # Calculate Motion
            energies = engine.calculate_motion(frame, prev_frame)
            
            result.timestamps.append(curr / handler.fps)
            for name, val in energies.items():
                result.motion_energy[name].append(val)
            
            prev_frame = frame
            curr += 1
            frames_processed += 1
            
            # Progress Updates
            local_est.update(curr)
            global_est.update(global_offset + frames_processed)
            if frames_processed % 30 == 0:
                l_eta = local_est.get_eta_string(curr)
                g_eta = global_est.get_eta_string(global_offset + frames_processed)
                pct = video_start_pct + int((frames_processed / (end_frame-start_frame)) * 60)
                self.emit_progress(pct, f"Video: {base_name}", f"Motion Analysis | Video ETA: {l_eta} | Batch ETA: {g_eta}")
        
        reader.release()
        return result, frames_processed

    def _run_tracked_analysis(self, analyzer, result_container, settings, video_path,
                             handler, local_est, global_est, quality_monitor,
                             video_start_pct, base_name, global_offset):
        """Core Positional Tracking Loop (Optimized for RAM)."""
        frame_count = 0
        total_local_frames = analyzer.end_frame - analyzer.start_frame

        # Core: Analyzer Generator
        for frame_idx, position, overlapping_zones in analyzer.process_frames():
            result_container.positions.append(position)
            result_container.timestamps.append(frame_idx / handler.fps)
            result_container.overlapping_zones.append(overlapping_zones)
            result_container.roi_labels.append(overlapping_zones[0] if overlapping_zones else 'outside')
            
            detected = position is not None
            # Update Quality Monitor
            quality_monitor.update(detected, 1.0 if detected else 0.0, position)
            
            frame_count += 1
            
            # Progress Updates
            local_est.update(frame_count)
            global_est.update(global_offset + frame_count)
            
            if frame_count % 30 == 0:
                tracking_progress = (frame_count / total_local_frames) * 50
                overall_pct = video_start_pct + int(20 + tracking_progress)
                l_eta = local_est.get_eta_string(frame_count)
                g_eta = global_est.get_eta_string(global_offset + frame_count)
                speed = local_est.get_speed()
                self.emit_progress(overall_pct, f"Video: {base_name}", 
                                 f"Frame {frame_idx} | ETA: {l_eta} | Speed: {speed:.1f} fps")
            
            # Save Checkpoint
            if self.is_cancelled:
                self.checkpoint_mgr.save_checkpoint(video_path, frame_idx, result_container.__dict__, settings)
                raise InterruptedError("Analysis cancelled")
            
            if self.checkpoint_mgr.should_save_checkpoint(frame_idx):
                self.checkpoint_mgr.save_checkpoint(video_path, frame_idx, result_container.__dict__, settings)

        analyzer.finalize_result(result_container)
        return result_container, True, frame_count

    def _generate_tracking_validation_video(self, result, video_path, settings, base_name, handler):
        """Generate tracking validation video with ROIs and Text directly to disk."""
        try:
            output_path = self.output_dir / f"{base_name}_timelapse.mp4"
            config = AppConfig()
            interval = config.visualization.timelapse_fps_divider
            if interval < 1: interval = 1
            out_fps = (handler.fps / interval) * config.visualization.timelapse_speed_multiplier
            
            # Use resolved visual labels if available, else raw roi labels
            labels = result.visual_labels if len(result.visual_labels) == len(result.timestamps) else result.roi_labels

            with SafeVideoWriter(output_path, out_fps, handler.dimensions) as writer:
                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, result.start_frame)
                
                curr_idx = 0
                total_frames = len(result.timestamps)
                
                while curr_idx < total_frames:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    if curr_idx % interval == 0:
                        # 1. Draw ROIs
                        frame = settings.roi_manager.draw_on_frame(frame, thickness=2)
                        
                        # 2. Draw Position Dot
                        pos = result.positions[curr_idx]
                        if pos:
                            cv2.circle(frame, pos, 8, (0, 0, 255), -1)
                        
                        # 3. Draw Zone Text Overlay
                        current_zone = labels[curr_idx] if curr_idx < len(labels) else "outside"
                        text = f"Zone: {current_zone.replace('_', ' ').title()}"
                        
                        # Text background for visibility
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.rectangle(frame, (10, 10), (350, 60), (0, 0, 0), -1)
                        cv2.putText(frame, text, (20, 45), font, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
                        
                        writer.write(frame)
                    curr_idx += 1
                cap.release()
            self.log.emit(f"✓ Timelapse saved")
        except Exception as e:
            logger.error(f"Error creating timelapse video: {e}", exc_info=True)

    def _export_all_data(self, result, base_name, settings):
        """Export CSVs and JSONs."""
        DataExporter.export_to_csv(result, self.output_dir / f"{base_name}_tracking_data.csv")
        DataExporter.export_summary_csv(result, self.output_dir / f"{base_name}_analysis_summary.csv")
        
        # JSON Export - Handle differences in settings objects
        detection_params = None
        if hasattr(settings, 'detection_config'):
             detection_params = settings.detection_config.to_dict()
        
        DataExporter.export_summary_json(result, self.output_dir / f"{base_name}_summary.json", detection_params)
        self.log.emit("✓ Data exported")

    def _generate_all_visualizations(self, result, handler, base_name, cb):
        """Generate generic plots."""
        Visualizer.generate_heatmap(result.positions, handler.dimensions, self.output_dir / f"{base_name}_heatmap.png")
        Visualizer.generate_trajectory_plot(result, handler.dimensions, self.output_dir / f"{base_name}_trajectory.png")
        Visualizer.generate_time_series(result, self.output_dir / f"{base_name}_timeseries.png")
        self.maze.generate_specific_plots(result, self.output_dir, base_name)
        self.log.emit("✓ Visualizations generated")

    def _generate_tst_validation_video(self, result, video_path, settings, base_name):
        """Generate color-coded TST validation video."""
        try:
            output_path = self.output_dir / f"{base_name}_validation.mp4"
            vh = VideoHandler(video_path)
            
            # Subsample for validation video speed
            step = 3
            out_fps = vh.fps / step
            
            # Core: Use SafeVideoWriter
            with SafeVideoWriter(output_path, out_fps, vh.dimensions) as writer:
                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, result.start_frame)
                
                roi_dict = {cat: rois[0] for cat, rois in settings.roi_manager.rois.items() if rois}
                curr_idx = 0
                total_frames = len(result.timestamps)
                
                while curr_idx < total_frames:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    if curr_idx % step == 0:
                        time_sec = (result.start_frame + curr_idx) / vh.fps
                        
                        for name, roi in roi_dict.items():
                            valid_start, valid_end = 0, float('inf')
                            if result.per_roi_times and name in result.per_roi_times:
                                valid_start, valid_end = result.per_roi_times[name]
                            
                            pts = roi.points.astype(np.int32)
                            
                            if time_sec < valid_start or time_sec > valid_end:
                                # Grey out excluded time
                                overlay = frame.copy()
                                cv2.fillPoly(overlay, [pts], (50, 50, 50))
                                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                                cv2.putText(frame, "EXCLUDED", (pts[0][0], pts[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)
                            else:
                                # Show active status (Mobile/Immobile)
                                if name in result.immobile_states and curr_idx < len(result.immobile_states[name]):
                                    is_immobile = result.immobile_states[name][curr_idx]
                                    color = (0, 0, 255) if is_immobile else (0, 255, 0) # Red/Green
                                    cv2.polylines(frame, [pts], True, color, 2)
                                    status = "IMMOBILE" if is_immobile else "MOBILE"
                                    cv2.putText(frame, status, (pts[0][0], pts[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        writer.write(frame)
                    curr_idx += 1
                cap.release()
                
            self.log.emit(f"✓ Validation video saved")
        except Exception as e:
            logger.error(f"Error creating validation video: {e}")