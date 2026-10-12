"""
Freestyle / Open Field module for flexible behavioral analysis.
Thread-safe plotting implementation.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Type, Dict, Optional
import logging

# Thread-safe plotting
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

from mazes.base_maze import BaseAnalysisResult, Maze
from core.kinematics import (
    calculate_immobility_bouts,
    calculate_instantaneous_speeds,
    calculate_movement_efficiency,
    calculate_smoothed_path_length,
    calculate_smoothed_speeds,
    calculate_zone_first_entries,
)
from core.zone_events import (
    debounce_zone_labels,
    dwell_frames,
    extract_zone_transitions,
    overlapping_zones_to_candidates,
)

logger = logging.getLogger(__name__)

UNCALIBRATED_IMMOBILITY_UNIT = "N/A (uncalibrated)"
NOT_AVAILABLE = "N/A"


def build_transition_matrix(transitions: List[Tuple[str, str, int]], 
                            zone_names: List[str]) -> np.ndarray:
    n = len(zone_names)
    matrix = np.zeros((n, n), dtype=int)
    zone_to_idx = {name: i for i, name in enumerate(zone_names)}
    
    for from_zone, to_zone, _ in transitions:
        if from_zone in zone_to_idx and to_zone in zone_to_idx:
            from_idx = zone_to_idx[from_zone]
            to_idx = zone_to_idx[to_zone]
            matrix[from_idx, to_idx] += 1
    
    return matrix


# ============================================================================
# RESULT DATACLASS
# ============================================================================

@dataclass
class FreestyleAnalysisResult(BaseAnalysisResult):
    """Extended result container for Freestyle/Open Field analysis."""
    # Movement metrics. Per-frame traces below are exported to
    # _speed_data.csv, not to the JSON summary.
    instantaneous_speeds: List[Optional[float]] = field(default_factory=list)
    smoothed_speeds: List[Optional[float]] = field(default_factory=list)
    immobile_states: List[Optional[bool]] = field(default_factory=list)
    average_speed: float = 0.0
    peak_speed: float = 0.0
    # Immobility fields are None when uncalibrated: a cm/s threshold cannot
    # be applied to pixel speeds, so immobility is reported as unavailable.
    immobility_time: Optional[float] = 0.0
    immobility_percentage: Optional[float] = 0.0
    immobility_bouts: Optional[int] = 0
    immobility_threshold: Optional[float] = 0.0
    immobility_threshold_unit: str = "pixels/s"
    speed_window_seconds: float = 0.2
    min_immobility_seconds: float = 1.0
    smoothed_total_distance: float = 0.0
    movement_efficiency: float = 0.0
    notes: List[str] = field(default_factory=list)
    
    # Zone exploration metrics
    zone_transitions: List[Tuple[str, str, int]] = field(default_factory=list)
    total_transitions: int = 0
    zone_entry_counts: Dict[str, int] = field(default_factory=dict)
    zone_first_entries: Dict[str, float] = field(default_factory=dict)
    
    # Custom metrics
    custom_metrics: Dict[str, any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        base_dict = super().to_dict()
        
        base_dict['freestyle_metrics'] = {
            'movement': {
                'average_speed': self.average_speed,
                'peak_speed': self.peak_speed,
                'immobility_time': self.immobility_time,
                'immobility_percentage': self.immobility_percentage,
                'immobility_threshold': self.immobility_threshold,
                'immobility_threshold_unit': self.immobility_threshold_unit,
                'immobility_bouts': self.immobility_bouts,
                'min_immobility_seconds': self.min_immobility_seconds,
                'speed_window_seconds': self.speed_window_seconds,
                'smoothed_total_distance': self.smoothed_total_distance,
                'movement_efficiency': self.movement_efficiency,
                'notes': self.notes
            },
            'exploration': {
                'total_transitions': self.total_transitions,
                'zone_entry_counts': self.zone_entry_counts,
                'zone_first_entries': self.zone_first_entries
            },
            'custom': self.custom_metrics
        }
        
        return base_dict


# ============================================================================
# FREESTYLE MAZE CLASS
# ============================================================================

class Freestyle(Maze):
    """Freestyle/Open Field maze type with flexible zone configuration."""
    
    def __init__(self):
        self.user_defined_zones: List[Tuple[str, str]] = []
        self._zones_configured = False
        self.immobility_threshold_cm_per_s = 2.0
        self.speed_window_seconds = 0.2
        self.min_immobility_seconds = 1.0
        self.min_dwell_seconds = 0.10
        # Pre-3.2 configs stored a px/s threshold; kept only for reporting.
        self._legacy_threshold_px_per_s: Optional[float] = None
    
    @property
    def name(self) -> str:
        return "Freestyle / Open Field"
    
    def configure_zones(self, zone_definitions: List[Tuple[str, str]]) -> None:
        self.user_defined_zones = zone_definitions
        self._zones_configured = True
        if not zone_definitions:
            logger.info("Freestyle configured for zone-free tracking")
        else:
            logger.info(f"Freestyle configured with {len(zone_definitions)} custom zones")
    
    def get_roi_definitions(self) -> List[Tuple[str, str]]:
        if not self._zones_configured:
            return []
        return self.user_defined_zones
    
    def needs_reference_line(self) -> bool:
        return True
    
    def get_result_class(self) -> Type[BaseAnalysisResult]:
        return FreestyleAnalysisResult

    def resolve_tracking_labels(self, result: BaseAnalysisResult) -> None:
        source = (
            overlapping_zones_to_candidates(result.overlapping_zones)
            if result.overlapping_zones else
            list(result.raw_roi_labels or result.roi_labels)
        )
        result.roi_labels = debounce_zone_labels(
            source, dwell_frames(self.min_dwell_seconds, result.fps)
        )
        result.visual_labels = list(result.roi_labels)
    
    def calculate_metrics(self, result: BaseAnalysisResult) -> None:
        if not isinstance(result, FreestyleAnalysisResult):
            raise TypeError("Freestyle logic requires a FreestyleAnalysisResult instance.")
        
        logger.info("Calculating Freestyle metrics...")
        
        # Movement metrics
        calibrated = result.scale_factor > 0
        speed_scale = result.scale_factor if calibrated else 1.0
        result.notes = []
        result.speed_window_seconds = self.speed_window_seconds
        result.min_immobility_seconds = self.min_immobility_seconds

        result.instantaneous_speeds = calculate_instantaneous_speeds(
            result.positions, result.fps, speed_scale
        )
        result.smoothed_speeds = calculate_smoothed_speeds(
            result.positions, result.fps, speed_scale,
            self.speed_window_seconds
        )
        valid_speeds = [s for s in result.smoothed_speeds if s is not None]
        result.average_speed = float(np.mean(valid_speeds)) if valid_speeds else 0.0
        result.peak_speed = float(np.max(valid_speeds)) if valid_speeds else 0.0

        # Smoothed path length removes the distance centroid jitter adds to
        # the raw sum; total_distance stays the raw metric.
        result.smoothed_total_distance = calculate_smoothed_path_length(
            result.positions, result.fps, speed_scale,
            self.speed_window_seconds
        )

        if calibrated:
            result.immobility_threshold = self.immobility_threshold_cm_per_s
            result.immobility_threshold_unit = "cm/s"
            (
                result.immobility_time,
                result.immobility_percentage,
                result.immobility_bouts,
                result.immobile_states,
            ) = calculate_immobility_bouts(
                result.smoothed_speeds,
                result.fps,
                self.immobility_threshold_cm_per_s,
                self.min_immobility_seconds,
            )
            if self._legacy_threshold_px_per_s is not None:
                result.notes.append(
                    "Legacy immobility threshold "
                    f"{self._legacy_threshold_px_per_s:g} px/s "
                    f"(= {self._legacy_threshold_px_per_s * speed_scale:g} "
                    f"cm/s at this calibration) was not applied; "
                    f"immobility used "
                    f"{self.immobility_threshold_cm_per_s:g} cm/s."
                )
        else:
            result.immobility_threshold = None
            result.immobility_threshold_unit = UNCALIBRATED_IMMOBILITY_UNIT
            result.immobility_time = None
            result.immobility_percentage = None
            result.immobility_bouts = None
            result.immobile_states = []
            result.notes.append(
                "Immobility is not reported: no spatial calibration, so a "
                "cm/s threshold cannot be applied. Speeds are in pixels/s. "
                "Draw a reference line to calibrate."
            )

        # Efficiency compares net displacement per tracked segment with the
        # smoothed path, so jitter does not deflate it. Segment endpoints are
        # raw positions in both, keeping the ratio consistent (capped at 1).
        result.movement_efficiency = calculate_movement_efficiency(
            result.positions,
            result.smoothed_total_distance,
            speed_scale
        )
        
        # Exploration metrics
        if self.user_defined_zones:
            result.zone_transitions = extract_zone_transitions(
                result.roi_labels
            )
            result.total_transitions = len(result.zone_transitions)
            for from_zone, to_zone, _ in result.zone_transitions:
                if to_zone not in {'outside', 'undetected'}:
                    result.zone_entry_counts[to_zone] = result.zone_entry_counts.get(to_zone, 0) + 1
            first_observed_zone = next(
                (z for z in result.roi_labels if z not in {'outside', 'undetected'}),
                None
            )
            if first_observed_zone:
                result.zone_entry_counts[first_observed_zone] = (
                    result.zone_entry_counts.get(first_observed_zone, 0) + 1
                )
            result.zone_first_entries = calculate_zone_first_entries(
                result.roi_labels, result.timestamps
            )
        
        logger.info("Freestyle metrics calculation complete")
    
    def get_batch_summary_headers(self) -> List[str]:
        headers = [
            'average_speed', 'peak_speed', 'speed_unit',
            'immobility_threshold', 'immobility_threshold_unit',
            'immobility_time_s', 'immobility_pct', 'immobility_bouts',
            'smoothed_total_distance', 'movement_efficiency'
        ]
        
        if self.user_defined_zones:
            headers.extend([
                'total_transitions',
                'num_zones_visited'
            ])
            for internal_name, _ in self.user_defined_zones:
                headers.append(f'{internal_name}_entries')
        
        return headers
    
    def get_batch_summary_row(self, result: BaseAnalysisResult) -> List[any]:
        if not isinstance(result, FreestyleAnalysisResult):
            n_cols = len(self.get_batch_summary_headers())
            return ['N/A'] * n_cols
        
        def fmt(value, spec):
            return NOT_AVAILABLE if value is None else format(value, spec)

        row = [
            f"{result.average_speed:.2f}",
            f"{result.peak_speed:.2f}",
            f"{result.distance_unit}/s",
            fmt(result.immobility_threshold, ".2f"),
            result.immobility_threshold_unit,
            fmt(result.immobility_time, ".2f"),
            fmt(result.immobility_percentage, ".2f"),
            NOT_AVAILABLE if result.immobility_bouts is None
            else result.immobility_bouts,
            f"{result.smoothed_total_distance:.2f}",
            f"{result.movement_efficiency:.4f}"
        ]

        if self.user_defined_zones:
            row.extend([
                result.total_transitions,
                len(result.zone_entry_counts)
            ])
            for internal_name, _ in self.user_defined_zones:
                count = result.zone_entry_counts.get(internal_name, 0)
                row.append(count)
        
        return row
    
    def generate_specific_plots(self, result: BaseAnalysisResult, 
                               output_dir: Path, base_name: str) -> None:
        """Generate Freestyle-specific plots and exports."""
        if not isinstance(result, FreestyleAnalysisResult): return
        
        logger.info("Generating Freestyle-specific plots...")
        
        self._generate_speed_plot(result, output_dir / f"{base_name}_speed.png")
        if self.user_defined_zones and result.total_transitions > 0:
            self._generate_transition_matrix_plot(result, output_dir / f"{base_name}_transition_matrix.png")
            self._generate_zone_entries_plot(result, output_dir / f"{base_name}_zone_entries.png")
    
    def _generate_speed_plot(self, result: FreestyleAnalysisResult, output_path: Path) -> None:
        try:
            if not result.smoothed_speeds or not result.timestamps: return

            # Thread-safe plotting
            fig = Figure(figsize=(14, 6))
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)

            unit = f"{result.distance_unit}/s"
            plot_speeds = [
                np.nan if speed is None else speed
                for speed in result.smoothed_speeds
            ]
            ax.plot(result.timestamps, plot_speeds,
                   'b-', alpha=0.7, linewidth=0.8, label=(
                       f'Smoothed speed ({result.speed_window_seconds:g} s window)'
                   ))

            ax.axhline(y=result.average_speed, color='g', linestyle='--',
                      linewidth=2, label=f'Average: {result.average_speed:.1f} {unit}')

            if result.immobility_threshold is not None:
                ax.axhline(y=result.immobility_threshold, color='r', linestyle='--',
                          linewidth=1, alpha=0.5, label=(
                              f'Immobility threshold: '
                              f'{result.immobility_threshold:.2f} {unit}'
                          ))
                if result.immobile_states:
                    ax.fill_between(
                        result.timestamps[:len(result.immobile_states)],
                        0, 1,
                        where=[state is True for state in result.immobile_states],
                        color='red', alpha=0.12, step='mid',
                        transform=ax.get_xaxis_transform(),
                        label=(
                            f'Immobile (>= {result.min_immobility_seconds:g} s)'
                        ))
            else:
                ax.plot([], [], ' ', label='Immobility: N/A (uncalibrated)')

            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel(f'Speed ({result.distance_unit}/second)')
            ax.set_title('Movement Speed Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(output_path, dpi=150)

        except Exception as e:
            logger.error(f"Error generating speed plot: {e}")
            raise

    def _generate_transition_matrix_plot(self, result: FreestyleAnalysisResult, output_path: Path) -> None:
        try:
            zone_names = [name for name, _ in self.user_defined_zones]
            # Only include zones that were visited
            used_zones = set(result.zone_entry_counts.keys())
            for f, t, _ in result.zone_transitions:
                used_zones.add(f)
                used_zones.add(t)
            
            filtered_names = [z for z in zone_names if z in used_zones]
            if len(filtered_names) < 2: return
            
            matrix = build_transition_matrix(result.zone_transitions, filtered_names)
            
            fig = Figure(figsize=(10, 8))
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            
            im = ax.imshow(matrix, cmap='YlOrRd')
            
            # Annotate
            for i in range(len(filtered_names)):
                for j in range(len(filtered_names)):
                    ax.text(j, i, matrix[i, j], ha="center", va="center", color="black")
            
            display_names = [n.replace('_', ' ').title() for n in filtered_names]
            ax.set_xticks(np.arange(len(display_names)))
            ax.set_yticks(np.arange(len(display_names)))
            ax.set_xticklabels(display_names, rotation=45)
            ax.set_yticklabels(display_names)
            
            fig.colorbar(im, ax=ax, label='Transition Count')
            ax.set_xlabel('To Zone')
            ax.set_ylabel('From Zone')
            ax.set_title('Zone Transition Matrix')
            
            fig.tight_layout()
            fig.savefig(output_path, dpi=150)
            
        except Exception as e:
            logger.error(f"Error generating transition matrix: {e}")
            raise
    
    def _generate_zone_entries_plot(self, result: FreestyleAnalysisResult, output_path: Path) -> None:
        try:
            if not result.zone_entry_counts: return
            
            sorted_zones = sorted(result.zone_entry_counts.items(), key=lambda x: x[1], reverse=True)
            zones = [name.replace('_', ' ').title() for name, _ in sorted_zones]
            counts = [count for _, count in sorted_zones]
            
            fig = Figure(figsize=(12, 6))
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            
            bars = ax.bar(zones, counts, color='steelblue', alpha=0.8, edgecolor='black')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{int(height)}', ha='center', va='bottom')
            
            ax.set_xlabel('Zone')
            ax.set_ylabel('Number of Entries')
            ax.set_title('Zone Entry Counts')
            
            fig.tight_layout()
            fig.savefig(output_path, dpi=150)
            
        except Exception as e:
            logger.error(f"Error generating zone entries plot: {e}")
            raise
    
    def _export_speed_csv(self, result: FreestyleAnalysisResult, output_path: Path) -> None:
        try:
            import csv
            if not result.smoothed_speeds or not result.timestamps: return

            with open(
                output_path, 'w', newline='', encoding='utf-8'
            ) as f:
                writer = csv.writer(f)
                unit = result.distance_unit
                writer.writerow([
                    'frame', 'timestamp_s',
                    f"speed_{unit}_per_s",
                    f"smoothed_speed_{unit}_per_s",
                    'is_immobile', 'distance_unit'
                ])

                states = result.immobile_states
                for i, (timestamp, raw, smooth) in enumerate(zip(
                    result.timestamps,
                    result.instantaneous_speeds,
                    result.smoothed_speeds,
                )):
                    # Blank when immobility is unavailable (uncalibrated) or
                    # the frame has no valid smoothed speed.
                    state = states[i] if i < len(states) else None
                    writer.writerow([
                        int(round(timestamp * result.fps)),
                        f"{timestamp:.3f}",
                        '' if raw is None else f"{raw:.2f}",
                        '' if smooth is None else f"{smooth:.2f}",
                        '' if state is None else int(state),
                        unit
                    ])
        except Exception as e:
            logger.error(f"Error exporting speed CSV: {e}")
            raise

    def export_specific_data(
        self,
        result: BaseAnalysisResult,
        output_dir: Path,
        base_name: str
    ) -> None:
        if isinstance(result, FreestyleAnalysisResult):
            self._export_speed_csv(
                result, output_dir / f"{base_name}_speed_data.csv"
            )
    
    def set_immobility_threshold(self, threshold: float) -> None:
        """Set the immobility speed threshold in cm/s (calibrated runs only)."""
        if threshold < 0: raise ValueError("Threshold must be non-negative")
        self.immobility_threshold_cm_per_s = float(threshold)

    def get_algorithm_parameters(self) -> Dict[str, any]:
        return {
            'zones': [list(zone) for zone in self.user_defined_zones],
            'zones_configured': self._zones_configured,
            'immobility_threshold_cm_per_s': self.immobility_threshold_cm_per_s,
            'speed_window_seconds': self.speed_window_seconds,
            'min_immobility_seconds': self.min_immobility_seconds,
            'min_dwell_seconds': self.min_dwell_seconds
        }

    def configure_algorithm_parameters(self, parameters: Dict[str, any]) -> None:
        if not parameters:
            return
        zones = [tuple(zone) for zone in parameters.get('zones', [])]
        if parameters.get('zones_configured', bool(zones)):
            self.configure_zones(zones)
        if 'immobility_threshold_cm_per_s' in parameters:
            self.set_immobility_threshold(
                float(parameters['immobility_threshold_cm_per_s'])
            )
            self._legacy_threshold_px_per_s = None
        else:
            # Legacy px/s thresholds were applied to raw jittery speeds and
            # are not comparable; keep the value for reporting only.
            legacy = parameters.get(
                'immobility_threshold_px_per_s',
                parameters.get('immobility_threshold')
            )
            self._legacy_threshold_px_per_s = (
                None if legacy is None else float(legacy)
            )
        self.speed_window_seconds = max(0.0, float(parameters.get(
            'speed_window_seconds', self.speed_window_seconds
        )))
        self.min_immobility_seconds = max(0.0, float(parameters.get(
            'min_immobility_seconds', self.min_immobility_seconds
        )))
        self.min_dwell_seconds = max(
            0.0,
            float(parameters.get('min_dwell_seconds', self.min_dwell_seconds))
        )
