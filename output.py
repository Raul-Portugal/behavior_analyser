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
from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from mazes.base_maze import BaseAnalysisResult, Maze
from core.video import SafeVideoWriter  # Replaces VideoWriter from video_io
from core.output_contract import COMMON_BATCH_HEADERS, common_batch_row

logger = logging.getLogger(__name__)


class DataExporter:
    """A collection of static methods for exporting analysis data to various formats."""

    @staticmethod
    def export_batch_summary_csv(all_results: List[Tuple[Path, BaseAnalysisResult]], output_dir: Path, maze: Maze):
        if not all_results:
            logger.warning("No results to export for batch summary.")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "batch_summary.csv"
        
        logger.info(f"Exporting {maze.name} batch summary to: {output_path}")

        headers = COMMON_BATCH_HEADERS + maze.get_batch_summary_headers()

        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

                for video_path, result in all_results:
                    maze_data = maze.get_batch_summary_row(result)
                    common_cells = common_batch_row(
                        video_path, result, maze.name
                    )
                    
                    # Handle multi-row output (e.g., TST multiple mice)
                    if maze_data and isinstance(maze_data[0], list):
                        for sub_row in maze_data:
                            writer.writerow(common_cells + sub_row)
                    else:
                        writer.writerow(common_cells + maze_data)
                        
        except IOError as e:
            logger.error(f"Failed to write batch summary file: {e}")
            raise

    @staticmethod
    def export_to_csv(result: BaseAnalysisResult, output_path: Path):
        logger.info(f"Exporting raw data to CSV: {output_path}")
        
        is_tst = (
            hasattr(result, 'motion_evidence')
            and len(result.motion_evidence) > 0
        )
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                if is_tst:
                    # --- TST EXPORT FORMAT ---
                    mice = sorted(
                        set(result.activity_scores)
                        | set(result.motion_evidence)
                    )
                    header = ['frame', 'timestamp_s']
                    for m in mice:
                        header.extend([
                            f'{m}_active_evidence',
                            f'{m}_behavior_state',
                            f'{m}_is_immobile',
                            f'{m}_confidence',
                            f'{m}_rigid_motion',
                            f'{m}_rotation_motion',
                            f'{m}_rotation_radians',
                            f'{m}_rigid_dx',
                            f'{m}_rigid_dy',
                            f'{m}_residual_motion',
                            f'{m}_deformation',
                            f'{m}_coherence',
                            f'{m}_motion_injection',
                            f'{m}_angular_injection',
                            f'{m}_pendulum_score',
                            f'{m}_temporal_ready',
                            f'{m}_mask_area_ratio',
                            f'{m}_affine_inlier_ratio',
                            f'{m}_camera_stabilized',
                            f'{m}_qc_flag',
                        ])
                    writer.writerow(header)
                    
                    if not mice: return
                    length = len(result.timestamps)
                    
                    for i in range(length):
                        ts = result.timestamps[i]
                        row = [int(round(ts * result.fps)), f"{ts:.3f}"]
                        for m in mice:
                            activity_trace = result.activity_scores.get(m, [])
                            activity_score = (
                                activity_trace[i]
                                if i < len(activity_trace) else None
                            )
                            state_trace = result.immobile_states.get(m, [])
                            immobile = (
                                state_trace[i] if i < len(state_trace) else None
                            )
                            evidence_trace = result.motion_evidence.get(m, [])
                            evidence = (
                                evidence_trace[i]
                                if i < len(evidence_trace) else {}
                            )
                            row.extend([
                                (
                                    "" if activity_score is None
                                    else f"{activity_score:.1f}"
                                ),
                                evidence.get("state", "uncertain"),
                                "" if immobile is None else int(immobile),
                                f"{float(evidence.get('confidence', 0.0)):.4f}",
                                f"{float(evidence.get('rigid_motion', 0.0)):.6f}",
                                f"{float(evidence.get('rotation_motion', 0.0)):.6f}",
                                f"{float(evidence.get('rotation_radians', 0.0)):.6f}",
                                f"{float(evidence.get('rigid_dx', 0.0)):.6f}",
                                f"{float(evidence.get('rigid_dy', 0.0)):.6f}",
                                f"{float(evidence.get('residual_motion', 0.0)):.6f}",
                                f"{float(evidence.get('deformation', 0.0)):.6f}",
                                f"{float(evidence.get('coherence', 0.0)):.6f}",
                                f"{float(evidence.get('motion_injection', 0.0)):.6f}",
                                f"{float(evidence.get('angular_injection', 0.0)):.6f}",
                                f"{float(evidence.get('pendulum_score', 0.0)):.6f}",
                                int(bool(evidence.get(
                                    'temporal_ready', False
                                ))),
                                f"{float(evidence.get('mask_area_ratio', 0.0)):.6f}",
                                f"{float(evidence.get('affine_inlier_ratio', 0.0)):.6f}",
                                int(bool(evidence.get(
                                    'camera_stabilized', False
                                ))),
                                evidence.get("qc_flag", ""),
                            ])
                        writer.writerow(row)
                        
                else:
                    # --- STANDARD TRACKING FORMAT ---
                    writer.writerow([
                        'frame', 'timestamp', 'x', 'y', 'raw_zone', 'zone', 'visual_zone',
                        'detected', 'confidence', 'failure_reason'
                    ])
                    
                    for i in range(len(result.positions)):
                        if i >= len(result.timestamps): break
                        pos = result.positions[i]
                        x, y = (pos[0], pos[1]) if pos else ('', '')
                        roi = result.roi_labels[i] if i < len(result.roi_labels) else ''
                        raw_roi = (
                            result.raw_roi_labels[i]
                            if i < len(result.raw_roi_labels) else roi
                        )
                        vis = result.visual_labels[i] if i < len(result.visual_labels) else ''
                        confidence = (
                            result.detection_confidences[i]
                            if i < len(result.detection_confidences) else 0.0
                        )
                        failure_reason = (
                            result.detection_failure_reasons[i]
                            if i < len(result.detection_failure_reasons) else ''
                        )
                        
                        writer.writerow([
                            int(round(result.timestamps[i] * result.fps)),
                            f"{result.timestamps[i]:.3f}",
                            x, y, raw_roi, roi, vis,
                            1 if pos else 0,
                            f"{confidence:.4f}",
                            failure_reason
                        ])
                        
        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")
            raise

    @staticmethod
    def export_summary_csv(
        result: BaseAnalysisResult,
        output_path: Path,
        maze: Optional[Maze] = None
    ):
        logger.info(f"Exporting summary statistics: {output_path}")
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                if hasattr(result, 'motion_evidence'):
                    writer.writerow(['Metric', 'Mouse', 'Value', 'Unit'])
                    writer.writerow(['--- TST RESULTS ---', '', '', ''])
                    mice = sorted(result.total_immobility_time.keys())
                    for m in mice:
                        latency = result.latency_to_first_immobility.get(m)
                        writer.writerow(['Total Immobility', m, f"{result.total_immobility_time[m]:.2f}", 'seconds'])
                        writer.writerow(['Immobility', m, f"{result.immobility_percentage.get(m, 0.0):.2f}", 'percent of valid time'])
                        writer.writerow(['Static Immobility', m, f"{result.static_immobility_time.get(m, 0.0):.2f}", 'seconds'])
                        writer.writerow(['Passive Swing', m, f"{result.passive_swing_time.get(m, 0.0):.2f}", 'seconds'])
                        writer.writerow(['Bridged Gap Immobility', m, f"{result.bridged_gap_time.get(m, 0.0):.2f}", 'seconds'])
                        writer.writerow(['Final Mobile Time', m, f"{result.mobile_time.get(m, 0.0):.2f}", 'seconds'])
                        writer.writerow(['Uncertain / Excluded', m, f"{result.uncertain_time.get(m, 0.0):.2f}", 'seconds'])
                        writer.writerow(['Valid Scored Time', m, f"{result.valid_time.get(m, 0.0):.2f}", 'seconds'])
                        writer.writerow(['Mean State Confidence', m, f"{result.mean_state_confidence.get(m, 0.0):.4f}", '0-1'])
                        writer.writerow(['Immobility Bouts', m, result.immobility_bouts.get(m, 0), 'count'])
                        writer.writerow([
                            'Latency to Immobile',
                            m,
                            '' if latency is None else f"{latency:.2f}",
                            'seconds'
                        ])
                else:
                    writer.writerow(['Metric', 'Subject', 'Value', 'Unit'])
                    writer.writerow(['Total Distance', '', f'{result.total_distance:.2f}', result.distance_unit])
                    writer.writerow(['Excluded Jumps', '', result.excluded_jump_count, 'count'])
                    writer.writerow(['Excluded Jump Distance', '', f'{result.excluded_jump_distance:.2f}', result.distance_unit])
                    writer.writerow(['Detection Rate', '', f'{result.detection_rate*100:.1f}', 'percent'])
                    writer.writerow(['Undetected Frames', '', result.undetected_frames, 'frames'])
                    writer.writerow(['Undetected Time', '', f'{result.undetected_time_s:.2f}', 'seconds'])
                    writer.writerow(['Tracking Quality', '', result.tracking_quality_grade, ''])
                    writer.writerow([
                        'Mean Detection Confidence',
                        '',
                        f"{result.tracking_quality.get('mean_confidence', 0.0):.3f}",
                        '0-1'
                    ])
                    writer.writerow([
                        'Implausible Jump Rate',
                        '',
                        f"{result.tracking_quality.get('implausible_jump_rate', 0.0)*100:.2f}",
                        'percent'
                    ])
                    writer.writerow(['--- TIME IN ZONES ---', '', '', ''])
                    for zone, time in sorted(result.time_in_roi.items()):
                        writer.writerow([f'Time in {zone}', zone, f'{time:.2f}', 'seconds'])
                    writer.writerow(['--- DISTANCE IN ZONES ---', '', '', ''])
                    for zone, dist in sorted(result.distance_in_roi.items()):
                        writer.writerow([f'Distance in {zone}', zone, f'{dist:.2f}', result.distance_unit])

                    if maze is not None:
                        headers = maze.get_batch_summary_headers()
                        values = maze.get_batch_summary_row(result)
                        if values and not isinstance(values[0], list):
                            writer.writerow(['--- PARADIGM METRICS ---', '', '', ''])
                            for metric, value in zip(headers, values):
                                writer.writerow([
                                    metric,
                                    '',
                                    value,
                                    DataExporter._metric_unit(
                                        metric, result
                                    ),
                                ])

        except Exception as e:
            logger.error(f"Error exporting summary CSV: {e}")
            raise

    @staticmethod
    def _metric_unit(metric: str, result: BaseAnalysisResult) -> str:
        """Attach explicit units to paradigm-specific human summary rows."""
        normalized = metric.lower()
        if normalized in {"speed_unit", "immobility_threshold_unit"}:
            return ""
        if "distance" in normalized:
            return result.distance_unit
        if normalized in {"average_speed", "peak_speed"}:
            return f"{result.distance_unit}/s"
        if normalized == "immobility_threshold":
            return getattr(
                result,
                "immobility_threshold_unit",
                f"{result.distance_unit}/s",
            )
        if normalized.endswith("_s") or normalized.endswith("_time"):
            return "seconds"
        if normalized.endswith("_pct") or "percent" in normalized:
            return "percent"
        if "efficiency" in normalized:
            return "0-1"
        if any(
            token in normalized
            for token in ("entries", "count", "transitions", "bouts")
        ):
            return "count"
        return ""

    @staticmethod
    def export_summary_json(
        result: BaseAnalysisResult,
        output_path: Path,
        detection_params: Optional[Dict] = None,
        run_configuration: Optional[Dict] = None
    ):
        try:
            data = result.to_dict()
            if detection_params:
                data['detection_parameters'] = detection_params
            if run_configuration:
                data['run_configuration'] = run_configuration
            
            if hasattr(result, '__dict__'):
                base_keys = set(BaseAnalysisResult().__dict__.keys())
                frame_trace_keys = {
                    'activity_scores',
                    'motion_evidence',
                    'immobile_states',
                    'behavior_states',
                    'state_confidences',
                    'instantaneous_speeds',
                }
                represented_specific_keys = set()
                if 'tst_metrics' in data:
                    represented_specific_keys.update({
                        'total_immobility_time',
                        'immobility_percentage',
                        'static_immobility_time',
                        'passive_swing_time',
                        'bridged_gap_time',
                        'mobile_time',
                        'uncertain_time',
                        'valid_time',
                        'mean_state_confidence',
                        'immobility_bouts',
                        'latency_to_first_immobility',
                        'per_roi_times',
                    })
                for k, v in result.__dict__.items():
                    if (
                        k not in data
                        and k not in base_keys
                        and k not in frame_trace_keys
                        and k not in represented_specific_keys
                        and not k.startswith('_')
                    ):
                        data.setdefault('specific_metrics', {})[k] = v
                        
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error exporting JSON: {e}")
            raise

    @staticmethod
    def export_sequence_details_csv(result: BaseAnalysisResult, output_path: Path):
        if not hasattr(result, 'arm_entries') or not result.arm_entries: return
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['entry_num', 'arm', 'timestamp'])
                for i, (arm, ts) in enumerate(result.arm_entries):
                    writer.writerow([i+1, arm, f"{ts:.3f}"])
        except Exception as exc:
            logger.error("Error exporting sequence details: %s", exc)
            raise


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
            raise

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
            cmap = colormaps.get_cmap('viridis').resampled(
                max(1, len(unique_labels))
            )
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
            raise

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
            raise

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
