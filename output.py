"""
output.py
Generic data export and visualization utilities.
Updated to use Core Architecture (SafeVideoWriter).
"""
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
# Force Agg backend for thread safety
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cm
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from mazes.base_maze import BaseAnalysisResult, Maze
from core.video import SafeVideoWriter  # Replaces VideoWriter from video_io

logger = logging.getLogger(__name__)


class DataExporter:
    """A collection of static methods for exporting analysis data to various formats."""

    @staticmethod
    def export_batch_summary_csv(all_results: List[Tuple[Path, BaseAnalysisResult]], output_dir: Path, maze: Maze):
        if not all_results:
            logger.warning("No results to export for batch summary.")
            return

        safe_name = "".join([c if c.isalnum() else "_" for c in maze.name])
        output_path = output_dir / f"_BATCH_SUMMARY_{safe_name}.csv"
        
        logger.info(f"Exporting {maze.name} batch summary to: {output_path}")

        # Basic headers
        generic_headers = ['video_filename', 'total_duration_s']
        headers = generic_headers + maze.get_batch_summary_headers()

        try:
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

                for video_path, result in all_results:
                    duration = (result.end_frame - result.start_frame) / result.fps if result.fps > 0 else 0
                    maze_data = maze.get_batch_summary_row(result)
                    
                    # Handle multi-row output (e.g., TST multiple mice)
                    if maze_data and isinstance(maze_data[0], list):
                        for sub_row in maze_data:
                            row = [video_path.name, f"{duration:.2f}"] + sub_row
                            writer.writerow(row)
                    else:
                        row = [video_path.name, f"{duration:.2f}"] + maze_data
                        writer.writerow(row)
                        
        except IOError as e:
            logger.error(f"Failed to write batch summary file: {e}")

    @staticmethod
    def export_to_csv(result: BaseAnalysisResult, output_path: Path):
        logger.info(f"Exporting raw data to CSV: {output_path}")
        
        # Check if TST (has motion_energy instead of positions)
        is_tst = hasattr(result, 'motion_energy') and len(result.motion_energy) > 0
        
        try:
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                if is_tst:
                    # --- TST EXPORT FORMAT ---
                    mice = sorted(list(result.motion_energy.keys()))
                    header = ['frame', 'timestamp']
                    for m in mice:
                        header.extend([f'{m}_energy', f'{m}_is_immobile'])
                    writer.writerow(header)
                    
                    if not mice: return
                    length = len(result.timestamps)
                    
                    for i in range(length):
                        ts = result.timestamps[i]
                        row = [result.start_frame + i, f"{ts:.3f}"]
                        for m in mice:
                            energy_trace = result.motion_energy.get(m, [])
                            energy = energy_trace[i] if i < len(energy_trace) else 0.0
                            state_trace = result.immobile_states.get(m, [])
                            state = 1 if (i < len(state_trace) and state_trace[i]) else 0
                            row.extend([f"{energy:.1f}", state])
                        writer.writerow(row)
                        
                else:
                    # --- STANDARD TRACKING FORMAT ---
                    writer.writerow(['frame', 'timestamp', 'x', 'y', 'zone', 'visual_zone', 'detected'])
                    
                    for i in range(len(result.positions)):
                        if i >= len(result.timestamps): break
                        pos = result.positions[i]
                        x, y = (pos[0], pos[1]) if pos else ('', '')
                        roi = result.roi_labels[i] if i < len(result.roi_labels) else ''
                        vis = result.visual_labels[i] if i < len(result.visual_labels) else ''
                        
                        writer.writerow([
                            result.start_frame + i, 
                            f"{result.timestamps[i]:.3f}",
                            x, y, roi, vis,
                            1 if pos else 0
                        ])
                        
        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")

    @staticmethod
    def export_summary_csv(result: BaseAnalysisResult, output_path: Path):
        logger.info(f"Exporting summary statistics: {output_path}")
        try:
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                if hasattr(result, 'total_immobility_time') and result.total_immobility_time:
                    writer.writerow(['Metric', 'Mouse', 'Value', 'Unit'])
                    writer.writerow(['--- TST RESULTS ---'])
                    mice = sorted(result.total_immobility_time.keys())
                    for m in mice:
                        writer.writerow(['Total Immobility', m, f"{result.total_immobility_time[m]:.2f}", 'seconds'])
                        writer.writerow(['Immobility Bouts', m, result.immobility_bouts.get(m, 0), 'count'])
                        writer.writerow(['Latency to Immobile', m, f"{result.latency_to_first_immobility.get(m, 0):.2f}", 'seconds'])
                else:
                    writer.writerow(['Metric', 'Value', 'Unit'])
                    writer.writerow(['Total Distance', f'{result.total_distance:.2f}', result.distance_unit])
                    writer.writerow(['Detection Rate', f'{result.detection_rate*100:.1f}%', ''])
                    writer.writerow(['--- TIME IN ZONES ---'])
                    for zone, time in sorted(result.time_in_roi.items()):
                        writer.writerow([f'Time in {zone}', f'{time:.2f}', 'seconds'])
                    writer.writerow(['--- DISTANCE IN ZONES ---'])
                    for zone, dist in sorted(result.distance_in_roi.items()):
                        writer.writerow([f'Distance in {zone}', f'{dist:.2f}', result.distance_unit])

        except Exception as e:
            logger.error(f"Error exporting summary CSV: {e}")

    @staticmethod
    def export_summary_json(result: BaseAnalysisResult, output_path: Path, detection_params: Optional[Dict] = None):
        try:
            data = result.to_dict()
            if detection_params:
                data['detection_parameters'] = detection_params
            
            if hasattr(result, '__dict__'):
                base_keys = set(BaseAnalysisResult().__dict__.keys())
                for k, v in result.__dict__.items():
                    if k not in data and k not in base_keys and not k.startswith('_'):
                        data.setdefault('specific_metrics', {})[k] = v
                        
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error exporting JSON: {e}")

    @staticmethod
    def export_sequence_details_csv(result: BaseAnalysisResult, output_path: Path):
        if not hasattr(result, 'arm_entries') or not result.arm_entries: return
        try:
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['entry_num', 'arm', 'timestamp'])
                for i, (arm, ts) in enumerate(result.arm_entries):
                    writer.writerow([i+1, arm, f"{ts:.3f}"])
        except Exception: pass


class Visualizer:
    """A collection of static methods for creating generic plots and visualizations."""

    @staticmethod
    def generate_heatmap(positions, dimensions, output_path, blur_sigma=5.0):
        logger.info(f"Generating heatmap: {output_path}")
        valid_positions = [p for p in positions if p]
        if not valid_positions:
            logger.warning("No valid positions to generate a heatmap.")
            return

        w, h = dimensions
        heatmap = np.zeros((h, w), dtype=np.float32)
        xs, ys = zip(*valid_positions)
        xs, ys = np.array(xs), np.array(ys)
        mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        np.add.at(heatmap, (ys[mask].astype(int), xs[mask].astype(int)), 1)
        
        if blur_sigma > 0:
            heatmap = gaussian_filter(heatmap, sigma=blur_sigma)

        try:
            fig = Figure(figsize=(10, 10 * (h / w)))
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            im = ax.imshow(heatmap, cmap='viridis', interpolation='nearest', aspect='auto')
            ax.set_title('Location Heatmap')
            ax.axis('off')
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(output_path, dpi=150)
        except Exception as e:
            logger.error(f"Failed to generate heatmap: {e}")

    @staticmethod
    def generate_trajectory_plot(result: BaseAnalysisResult, dimensions, output_path: Path):
        logger.info("Generating trajectory plot...")
        if len(result.positions) < 2: return

        w, h = dimensions
        try:
            fig = Figure(figsize=(10, 10 * (h / w)))
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            valid_pos = np.array([p for p in result.positions if p is not None])
            labels = [result.roi_labels[i] for i, p in enumerate(result.positions) if p is not None]
            unique_labels = sorted(list(set(labels)))
            cmap = cm.get_cmap('viridis', len(unique_labels))
            color_map = {label: cmap(i) for i, label in enumerate(unique_labels)}
            point_colors = [color_map.get(label, (0.5, 0.5, 0.5, 1.0)) for label in labels]

            if len(valid_pos) > 0:
                ax.scatter(valid_pos[:, 0], valid_pos[:, 1], c=point_colors, s=1, alpha=0.5)
                ax.plot(valid_pos[0, 0], valid_pos[0, 1], 'o', color='lime', markersize=10, label='Start')
                ax.plot(valid_pos[-1, 0], valid_pos[-1, 1], 'X', color='magenta', markersize=12, label='End')
            
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.set_title('Animal Trajectory')
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_path, dpi=150)
        except Exception as e:
            logger.error(f"Failed to generate trajectory plot: {e}")

    @staticmethod
    def generate_time_series(result: BaseAnalysisResult, output_path: Path):
        logger.info(f"Generating time series plot: {output_path}")
        if not result.timestamps: return

        try:
            # FIX: Fallback to roi_labels safely if visual_labels hasn't been populated
            labels_to_plot = result.visual_labels if len(result.visual_labels) == len(result.timestamps) else result.roi_labels
            
            if not labels_to_plot:
                logger.warning("No labels available for time series plot.")
                return

            unique_zones = sorted(list(set(labels_to_plot)))
            zone_map = {zone: i for i, zone in enumerate(unique_zones)}
            y_values = [zone_map.get(label, -1) for label in labels_to_plot]

            fig = Figure(figsize=(15, 6))
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            ax.step(result.timestamps, y_values, where='post')
            ax.set_yticks(range(len(unique_zones)))
            # Format zone names beautifully (e.g., "arm_a" -> "Arm A")
            ax.set_yticklabels([z.replace('_', ' ').title() for z in unique_zones])
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Current Zone')
            ax.set_title('Zone Occupancy Over Time')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            fig.tight_layout()
            fig.savefig(output_path, dpi=150)
        except Exception as e:
            logger.error(f"Failed to generate time series: {e}", exc_info=True)

    @staticmethod
    def create_timelapse_video(frames, output_path, fps, show_progress=True):
        """Used for pure array-based timelapse writing (deprecated for tracking, kept for compatibility)"""
        if not frames:
            logger.warning("No frames for timelapse.")
            return
        logger.info(f"Creating timelapse: {output_path}")
        h, w = frames[0].shape[:2]
        
        try:
            from core.video import SafeVideoWriter
            with SafeVideoWriter(output_path, fps, (w, h)) as writer:
                iterator = tqdm(frames, desc="Writing timelapse") if show_progress else frames
                for frame in iterator:
                    writer.write(frame)
        except Exception as e:
            logger.error(f"Failed to create timelapse: {e}")